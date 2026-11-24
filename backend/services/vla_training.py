"""Behavior cloning for continuous action chunks or discrete action classes."""
import hashlib
import json
from pathlib import Path
import random


def signature(config, datasets=()):
    excluded={'max_steps','save_every','eval_every','output_dir','experiment_name','run_id','started_at',
              'resume_from_checkpoint','resumed_from_run','vla_worker_version'}
    payload={k:v for k,v in config.items() if k not in excluded and (config.get('training_algorithm')=='ngu' or not k.startswith('ngu_'))}
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    for key in ('data_path','val_data_path'):
        if config.get(key):payload[key]=digest(config[key])
    images=sorted({row['image'] for data in datasets if data for row in data.rows})
    payload['images']=[(path,digest(path)) for path in images]
    if config.get('initial_policy_path'):
        root=Path(config['initial_policy_path'])
        payload['initial_policy_weights']=[(str(path.relative_to(root)),digest(path)) for path in sorted(root.rglob('*')) if path.is_file() and (path.suffix=='.safetensors' or path.name=='policy.json')]
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def action_loss(predictions, targets, mask, config):
    action_type = config['action_type']
    import torch.nn.functional as F
    if action_type == 'continuous':
        values = F.smooth_l1_loss(predictions.float(), targets, reduction='none').mean(-1)
    elif action_type == 'hybrid':
        dim = config['action_dim']
        values = F.smooth_l1_loss(predictions[..., :dim].float(), targets[..., :dim], reduction='none').mean(-1)
        offset = dim
        discrete = []
        for index, classes in enumerate(config['discrete_classes']):
            discrete.append(F.cross_entropy(predictions[..., offset:offset + classes].float().transpose(1, 2), targets[..., dim + index].long(), reduction='none'))
            offset += classes
        values = values + sum(discrete) / len(discrete)
    else:
        values = F.cross_entropy(predictions.float().transpose(1, 2), targets, reduction='none')
    return (values * mask).sum() / mask.sum().clamp(min=1)


def train(config, metrics):
    import time
    import torch
    import numpy as np
    from contextlib import nullcontext
    from backend.services.vla_data import Demonstrations, batch
    from backend.services.vla_model import create, save, restore_weights
    from backend.services import vla_grpo as rl
    torch.set_num_threads(min(4,torch.get_num_threads()))
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    algorithm=config.get('training_algorithm','sft');is_rl=algorithm!='sft'
    cuda=config['device']!='cpu' and torch.cuda.is_available()
    if config['device']=='cuda' and not cuda:raise ValueError('CUDA is unavailable')
    if config['architecture']=='vlm_action' and not cuda:raise ValueError('VLM action fine-tuning requires CUDA')
    device='cuda' if cuda else 'cpu'
    if cuda and config['mixed_precision']=='bf16' and not torch.cuda.is_bf16_supported():raise ValueError('Selected GPU does not support bf16')
    metrics.update(status='loading_dataset',stage='vla_grpo' if is_rl else 'vla_sft',training_algorithm=algorithm,
        reward_source=('demonstration_verifier' if algorithm=='ngu' else 'demonstration_distance') if is_rl else None,policy_format='mah_vla_v1',device=device,
        effective_precision=config['mixed_precision'] if cuda else 'float32')
    data=Demonstrations(config['data_path'],config)
    validation=Demonstrations(config['val_data_path'],config) if config.get('val_data_path') else None
    if max(len(row['actions']) for row in data.rows)<config['action_horizon']:raise ValueError('No training targets for the full action horizon; reduce action_horizon or supply full chunks')
    # A deterministic observation bank avoids a second GPU-resident reference backbone.
    output_dim=config['action_dim']+(sum(config['discrete_classes']) if config['action_type']=='hybrid' else 0)
    if is_rl and config['grpo_kl_beta'] and len(data)*config['action_horizon']*output_dim>64_000_000:
        raise ValueError('Reference output bank exceeds 256 MB; reduce dataset/horizon/head or set KL beta to 0')
    initial=rl.initial_policy(config)
    stats=initial['statistics'] if initial else data.statistics()
    fingerprint=signature(config,(data,validation))
    metrics.update(status='loading_model')
    model,processor=create(config);model.to(device)
    if initial and not config.get('resume_from_checkpoint'):restore_weights(model,config['initial_policy_path'])
    if config['mixed_precision']=='fp16':
        for parameter in model.parameters():
            if parameter.requires_grad:parameter.data=parameter.data.float()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
    def schedule(step):
        # Plan-independent schedule makes checkpoint extension reproduce the original optimizer path.
        return min(1.,(step+1)/max(1,config['warmup_steps']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    scaler=torch.amp.GradScaler('cuda',enabled=cuda and config['mixed_precision']=='fp16')
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16 if config['mixed_precision']=='bf16' else torch.float16) if cuda and config['mixed_precision']!='no' else nullcontext()
    def prepare(indices,dataset=data):
        inputs,states,targets,mask=batch([dataset.rows[i] for i in indices],config,stats,processor)
        inputs={key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in inputs.items()}
        return inputs,states.to(device),targets.to(device),mask.to(device)
    step=cursor=0;pending=None;reference_bank=None;ngu_stats={}
    generator=torch.Generator().manual_seed(42);order=torch.randperm(len(data),generator=generator).tolist()
    if config.get('resume_from_checkpoint'):
        directory=Path(config['resume_from_checkpoint']);state=torch.load(directory/'training_state.pt',map_location='cpu',weights_only=False)
        if state['signature']!=fingerprint or state['format']!='mah_vla_training_v1':raise ValueError('VLA checkpoint data/images/model/reward/optimizer configuration changed')
        restore_weights(model,directory);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        stats=state['statistics'];step,cursor,order=state['step'],state['cursor'],state['order']
        generator.set_state(state['generator']);random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'])
        if cuda:torch.cuda.set_rng_state_all(state['cuda_rng'])
        pending=state.get('pending');reference_bank=state.get('reference_bank');ngu_stats=state.get('ngu_stats',{})
        if is_rl and config['grpo_kl_beta'] and reference_bank is None:raise ValueError('GRPO checkpoint has no immutable reference bank')
        if step>=config['max_steps']:raise ValueError('Checkpoint reached max_steps; extend the training plan')
    elif is_rl and config['grpo_kl_beta']:
        metrics.update(status='building_reference')
        # Reference uses the same deterministic mode as sampling and gradient updates.
        rl.training_mode(model,algorithm)
        reference_bank=torch.empty(len(data),config['action_horizon'],output_dim,dtype=torch.float32)
        with torch.no_grad(),amp():
            for offset in range(0,len(data),config['batch_size']):
                indices=list(range(offset,min(offset+config['batch_size'],len(data))))
                inputs,states,_,_=prepare(indices);reference_bank[indices]=model(inputs,states).float().cpu()
    def take(count=None):
        nonlocal cursor,order
        indices=[]
        for _ in range(config['batch_size'] if count is None else count):
            if cursor>=len(order):order=torch.randperm(len(data),generator=generator).tolist();cursor=0
            indices.append(order[cursor]);cursor+=1
        return indices
    def cpu_tree(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu_tree(v) for v in value]
        return value
    def checkpoint(directory,final=False):
        save(directory,model,processor,stats,config)
        if not final:
            torch.save(dict(format='mah_vla_training_v1',signature=fingerprint,step=step,cursor=cursor,order=order,
                generator=generator.get_state(),statistics=stats,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if cuda else [],python_rng=random.getstate(),
                pending=cpu_tree(pending),reference_bank=reference_bank,ngu_stats=ngu_stats),directory/'training_state.pt')
            from backend.paths import atomic_json
            atomic_json(directory/'complete.json',{'format':'mah_vla_training_v1','step':step})
        metrics.log_checkpoint(str(directory),step=step)
    metrics.update(status='training',total_steps=config['max_steps'],current_step=step,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))
    started=time.perf_counter();start_step=step
    while step<config['max_steps']:
        rl.training_mode(model,algorithm)
        if not is_rl:
            work=[dict(indices=take()) for _ in range(config['gradient_accumulation'])]
        else:
            if pending is None:
                work=[]
                with torch.no_grad(),amp():
                    if algorithm=='ngu':
                        from backend.services.ngu import collect,record
                        required=config['batch_size']*config['gradient_accumulation']
                        # Each observation gets its own variable-sized group and baseline.
                        attempts=0
                        while len(work)<required and attempts<required*(1+config['ngu_max_refill_rounds']):
                            indices=take(1);attempts+=1
                            inputs,states,targets,mask=prepare(indices);output=model(inputs,states).float()
                            def draw():
                                actions=rl.sample(output,config)
                                return actions,rl.verified_success(actions,targets,mask,config)[0]
                            result=collect(draw,max_attempts=config['ngu_max_attempts'],continue_probability=config['ngu_continue_probability'])
                            record(ngu_stats,result)
                            if result.status!='mixed':continue
                            actions=torch.cat(result.chunks,dim=1);rewards=result.rewards.unsqueeze(0)
                            work.append(cpu_tree(dict(indices=indices,actions=actions,old_lp=rl.log_probs(output,actions,config),
                                advantages=rl.relative_advantages(rewards,config),rewards=rewards)))
                        metrics.update(ngu_stats=dict(ngu_stats))
                        if len(work)<required:raise ValueError('NGU exhausted its bounded sampling/refill budget: no mixed success groups. Use an SFT policy, check the verifier tolerance or increase the budget.')
                    for _ in range(0 if algorithm=='ngu' else config['gradient_accumulation']):
                        indices=take();inputs,states,targets,mask=prepare(indices)
                        output=model(inputs,states).float();actions=rl.sample(output,config)
                        rewards=rl.reward(actions,targets,mask,config)
                        work.append(cpu_tree(dict(indices=indices,actions=actions,old_lp=rl.log_probs(output,actions,config),
                            advantages=rl.relative_advantages(rewards,config),rewards=rewards)))
                pending=dict(work=work,updates=0)
            work=pending['work']
        for attempt in range(16):
            optimizer.zero_grad(set_to_none=True);total_loss=total_bc=mean_reward=kl=clip=0.
            for item in work:
                inputs,states,targets,mask=prepare(item['indices'])
                with amp():
                    output=model(inputs,states);bc=action_loss(output,targets,mask,config)
                    if is_rl:
                        ref=reference_bank[item['indices']].to(device) if reference_bank is not None else None
                        loss,parts=rl.objective(output,item['actions'].to(device),item['old_lp'].to(device),
                            item['advantages'].to(device),mask,config,reference=ref)
                        loss=loss+config['grpo_sft_weight']*bc
                        mean_reward+=float(item['rewards'].mean())/len(work);kl+=parts['kl']/len(work);clip+=parts['clip_fraction']/len(work)
                    else:loss=bc
                if not torch.isfinite(loss):raise ValueError('Action loss is not finite')
                scaler.scale(loss/len(work)).backward();total_loss+=float(loss.detach())/len(work);total_bc+=float(bc.detach())/len(work)
            scaler.unscale_(optimizer)
            finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if not finite:
                if not scaler.is_enabled():raise ValueError('Action gradients are not finite')
                scaler.step(optimizer);scaler.update();metrics.metrics['amp_overflows']=metrics.metrics.get('amp_overflows',0)+1
                continue
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();break
        else:raise ValueError('FP16 action gradients still overflow after 16 scale reductions')
        scheduler.step();step+=1
        if is_rl:
            pending['updates']+=1
            if pending['updates']>=config['grpo_updates']:pending=None
            metrics.metrics.setdefault('reward_history',[]).append(mean_reward)
            metrics.update(current_reward=mean_reward,kl_divergence=kl,clip_fraction=clip)
        metrics.log_step(step,total_loss,optimizer.param_groups[0]['lr']);metrics.metrics.setdefault('grad_norm_history',[]).append(float(norm))
        metrics.update(action_loss=total_bc,optimizer_step=step,estimated_remaining_seconds=(time.perf_counter()-started)/(step-start_step)*(config['max_steps']-step))
        if validation and (step%config['eval_every']==0 or step==config['max_steps']):
            model.eval();losses=valid_count=total_error=class_correct=0.
            with torch.no_grad(),amp():
                for offset in range(0,len(validation),config['batch_size']):
                    inputs,states,targets,mask=prepare(list(range(offset,min(offset+config['batch_size'],len(validation)))),validation)
                    predictions=model(inputs,states);loss=action_loss(predictions,targets,mask,config);count=float(mask.sum())
                    losses+=float(loss)*count;valid_count+=count
                    if config['action_type']!='discrete':
                        error=(predictions[...,:config['action_dim']].float()-targets[...,:config['action_dim']]).abs()*predictions.new_tensor(stats['action']['std'])
                        total_error+=float((error.mean(-1)*mask).sum())
                    else:total_error+=float(((predictions.argmax(-1)==targets)*mask).sum())
                    if config['action_type']=='hybrid':
                        category_offset=config['action_dim']
                        for index,classes in enumerate(config['discrete_classes']):
                            class_correct+=float(((predictions[...,category_offset:category_offset+classes].argmax(-1)==targets[...,config['action_dim']+index].long())*mask).sum());category_offset+=classes
            if config['action_type']=='hybrid':metrics.metrics['action_accuracy']=class_correct/(valid_count*len(config['discrete_classes']))
            metrics.metrics.setdefault('val_steps_history',[]).append(step);metrics.metrics.setdefault('val_loss_history',[]).append(losses/valid_count)
            metrics.update(current_val_loss=losses/valid_count,**({'action_mae':total_error/valid_count} if config['action_type']!='discrete' else {'action_accuracy':total_error/valid_count}))
        if step%config['save_every']==0 or step==config['max_steps']:checkpoint(Path(config['output_dir'])/f'checkpoint-{step}')
    final=Path(config['output_dir'])/'final_model';checkpoint(final,final=True)
    metrics.update(status='completed',final_model_dir=str(final))

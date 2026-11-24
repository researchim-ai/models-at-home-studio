"""Action distributions and group sampling, bridged to our homellm GRPO loss.
Rewards compare sampled chunks to demonstrations; no environment success is assumed.
"""
from types import SimpleNamespace
import torch
from torch.distributions import Normal, Categorical, kl_divergence


def distributions(output, c):
    output=output.float()
    if c['action_type']=='discrete':return [Categorical(logits=output/c['grpo_temperature'])]
    dim=c['action_dim'];parts=[Normal(output[..., :dim],c['grpo_action_std'])]
    if c['action_type']=='hybrid':
        offset=dim
        for classes in c['discrete_classes']:
            parts.append(Categorical(logits=output[..., offset:offset+classes]/c['grpo_temperature']));offset+=classes
    return parts


def sample(output,c):
    parts=distributions(output,c);g=c['grpo_group_size']
    if c['action_type']=='discrete':return parts[0].sample((g,)).permute(1,0,2).detach()
    continuous=parts[0].sample((g,)).permute(1,0,2,3)
    if len(parts)>1:
        categories=torch.stack([p.sample((g,)).permute(1,0,2) for p in parts[1:]],-1)
        continuous=torch.cat([continuous,categories.float()],-1)
    return continuous.detach()


def log_probs(output,actions,c):
    # [observations, group, horizon], joint likelihood for each action vector.
    parts=distributions(output.unsqueeze(1),c)
    if c['action_type']=='discrete':return parts[0].log_prob(actions.long())
    dim=c['action_dim'];lp=parts[0].log_prob(actions[..., :dim]).sum(-1)
    for index,part in enumerate(parts[1:]):lp=lp+part.log_prob(actions[..., dim+index].long())
    return lp


def reward(actions,targets,mask,c):
    """Negative mean normalized squared error + mean categorical accuracy; no padding reward."""
    if c['action_type']=='discrete':value=(actions==targets[:,None]).float()
    else:
        dim=c['action_dim'];value=-(actions[..., :dim]-targets[:,None,...,:dim]).square().mean(-1)
        if c['action_type']=='hybrid':value=value+(actions[..., dim:]==targets[:,None,...,dim:]).float().mean(-1)
    valid=mask[:,None].to(value)
    return ((value*valid).sum(-1)/valid.sum(-1).clamp_min(1)).detach()


def verified_success(actions,targets,mask,c):
    """A VLA demonstration verifier, not an environment-success oracle."""
    if c['action_type']=='discrete':ok=(actions==targets[:,None])
    else:
        dim=c['action_dim']
        mse=(actions[...,:dim]-targets[:,None,...,:dim]).square().mean(-1)
        valid=mask[:,None].to(mse)
        rmse=((mse*valid).sum(-1)/valid.sum(-1).clamp_min(1)).sqrt()
        continuous_ok=rmse<=c['ngu_action_tolerance']
        if c['action_type']=='continuous':return continuous_ok.float().detach()
        ok=(actions[...,dim:]==targets[:,None,...,dim:]).all(-1)
        return (continuous_ok & (ok | ~mask[:,None]).all(-1)).float().detach()
    return (ok | ~mask[:,None]).all(-1).float().detach()


def relative_advantages(rewards,c):
    from homellm.training.rl.loss import compute_advantages
    return compute_advantages(rewards,use_std_normalization=c['training_algorithm'] not in ('dr_grpo','ngu')).detach()


def objective(output,actions,old_lp,advantages,mask,c,reference=None):
    from homellm.training.rl.loss import GRPOLoss
    lp=log_probs(output,actions,c);shape=lp.shape
    valid=mask[:,None].expand(shape).reshape(-1,shape[-1])
    exp=SimpleNamespace(action_log_probs=old_lp.reshape(-1,shape[-1]),log_probs_ref=None,
        action_mask=valid,advantages=advantages[...,None].expand(shape).reshape(-1,shape[-1]))
    core=GRPOLoss(clip_eps_low=c['grpo_clip_epsilon'],
        clip_eps_high=c.get('grpo_clip_epsilon_high',.28) if c['training_algorithm']=='dapo' else c['grpo_clip_epsilon'],
        kl_weight=0.,token_level_loss=c['training_algorithm']=='dapo',
        fixed_length_normalizer=c['action_horizon'] if c['training_algorithm']=='dr_grpo' else None)
    loss,components=core(lp.reshape(-1,shape[-1]),exp)
    kl=output.new_zeros((),dtype=torch.float32)
    if reference is not None and c['grpo_kl_beta']:
        terms=[]
        for current,ref in zip(distributions(output,c),distributions(reference,c)):
            value=kl_divergence(current,ref)
            terms.append(value.sum(-1) if isinstance(current,Normal) else value)
        per_step=sum(terms);kl=((per_step*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
        loss=loss+c['grpo_kl_beta']*kl
    return loss,dict(kl=float(kl.detach()),clip_fraction=float(((lp-old_lp).exp().sub(1).abs()>c['grpo_clip_epsilon']).float().masked_select(mask[:,None].expand(shape)).mean()))


def training_mode(model,algorithm):
    model.train()
    if algorithm!='sft':
        # Same deterministic network for sampling and likelihood updates; gradients stay enabled.
        for module in model.modules():
            if isinstance(module,torch.nn.modules.dropout._DropoutNd):module.eval()
            for name in ('attention_dropout','hidden_dropout','activation_dropout','dropout'):
                if isinstance(getattr(module,name,None),(float,int)):setattr(module,name,0.)


def initial_policy(c):
    """Validate the warm-start policy contract before loading its weights."""
    import json
    from pathlib import Path
    if not c.get('initial_policy_path'):return None
    value=json.loads((Path(c['initial_policy_path'])/'policy.json').read_text())
    if value.get('format')!='mah_vla_v1':raise ValueError('Initial policy is not a Studio VLA export')
    root=Path(c['initial_policy_path'])
    expected=['policy.safetensors'] if c['architecture']=='native_vla' else ['action_head.safetensors']
    if any(not (root/name).is_file() or not (root/name).stat().st_size for name in expected):raise ValueError('Initial VLA policy has missing/empty weight files')
    if c['architecture']=='vlm_action' and not any((root/'backbone').glob('*.safetensors')):raise ValueError('Initial VLA backbone weights are missing')
    old=value['config']
    fields=['architecture','action_type','action_dim','action_horizon','state_dim','hidden_dim']
    fields+=['vision_grid','image_size','max_instruction_bytes'] if c['architecture']=='native_vla' else ['model_path','tuning_method','training_backend','lora_r','lora_alpha','lora_target_modules','lora_modules_to_save']
    for key in fields+['discrete_classes','action_names','discrete_names','discrete_labels']:
        if old.get(key)!=c.get(key):raise ValueError(f'Initial VLA policy contract differs: {key}')
    return value

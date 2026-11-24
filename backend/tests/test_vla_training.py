"""Actual optimizer/resume checks, independent of the desktop smoke harness."""
import json
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('homellm.training.vlm_common')
from homellm.training.vlm_common import MetricsLogger
from safetensors.torch import load_file
from backend.routes.vla import VLATrainingRequest
from backend.services.vla_training import train


@pytest.fixture
def setup_policy(tmp_path):
    from PIL import Image
    def setup(algorithm='sft',kind='continuous'):
        rows=[]
        for i in range(4):
            image=tmp_path/f'{i}.png';Image.new('RGB',(16,16),'red' if i%2 else 'blue').save(image)
            action=[float(i),-float(i)] if kind=='continuous' else i%2 if kind=='discrete' else {'continuous':[float(i),-float(i)],'discrete':[i%2]}
            rows.append(dict(image=str(image),instruction=f'move {i%2}',state=[],actions=[action,action]))
        data=tmp_path/'data.jsonl';data.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        c=VLATrainingRequest(training_algorithm=algorithm,action_type=kind,action_dim=2,action_horizon=2,state_dim=0,
            discrete_classes=[2] if kind=='hybrid' else [],hidden_dim=16,image_size=16,device='cpu',mixed_precision='no',
            batch_size=2,max_steps=3,save_every=1,eval_every=1,grpo_group_size=4).model_dump()
        c.update(data_path=str(data),dataset_path=str(data),val_data_path=str(data),output_dir=str(tmp_path/'out'))
        return c
    return setup


@pytest.mark.parametrize('algorithm,kind',[('sft','continuous'),('grpo','continuous'),('grpo','hybrid'),('dr_grpo','discrete'),('dapo','hybrid'),('ngu','continuous'),('ngu','hybrid'),('ngu','discrete')])
def test_actual_optimizer_and_mid_group_exact_resume(setup_policy,tmp_path,algorithm,kind):
    c=setup_policy(algorithm,kind);log=MetricsLogger(tmp_path/'metrics.json');train(c,log)
    final=load_file(str(Path(c['output_dir'])/'final_model/policy.safetensors'))
    first=load_file(str(Path(c['output_dir'])/'checkpoint-1/policy.safetensors'))
    assert any(not torch.equal(first[key],final[key]) for key in final)
    assert log.metrics['current_step']==3 and all(v>0 for v in log.metrics['grad_norm_history'])
    resumed={**c,'resume_from_checkpoint':str(Path(c['output_dir'])/'checkpoint-1'),'output_dir':str(tmp_path/'resumed')}
    train(resumed,MetricsLogger(tmp_path/'resume.json'))
    restored=load_file(str(tmp_path/'resumed/final_model/policy.safetensors'))
    assert all(torch.equal(final[key],restored[key]) for key in final)
    if algorithm!='sft':
        state=torch.load(Path(c['output_dir'])/'checkpoint-1/training_state.pt',weights_only=False)
        assert state['pending']['updates']==1 and state['reference_bank'] is not None
        assert len(log.metrics['reward_history'])==3


def test_sft_to_grpo_warm_start_and_image_change_rejection(setup_policy,tmp_path):
    c=setup_policy();train(c,MetricsLogger(tmp_path/'sft.json'))
    policy=Path(c['output_dir'])/'final_model'
    rl={**c,'training_algorithm':'grpo','stage':'vla_grpo','initial_policy_path':str(policy),'output_dir':str(tmp_path/'rl')}
    train(rl,MetricsLogger(tmp_path/'grpo.json'))
    assert json.loads((tmp_path/'rl/final_model/policy.json').read_text())['statistics']==json.loads((policy/'policy.json').read_text())['statistics']
    from PIL import Image
    Image.new('RGB',(16,16),'green').save(tmp_path/'0.png')
    with pytest.raises(ValueError,match='images/model/reward'):
        train({**rl,'resume_from_checkpoint':str(tmp_path/'rl/checkpoint-1'),'output_dir':str(tmp_path/'bad')},MetricsLogger(tmp_path/'bad.json'))


def test_ngu_unreachable_verifier_stops_without_an_optimizer_update(setup_policy,tmp_path,monkeypatch):
    from backend.services import vla_grpo
    c=setup_policy('ngu');c.update(ngu_max_attempts=2,ngu_max_refill_rounds=0)
    monkeypatch.setattr(vla_grpo,'verified_success',lambda actions,*args:torch.zeros(actions.shape[:2]))
    log=MetricsLogger(tmp_path/'failed.json')
    with pytest.raises(ValueError,match='bounded sampling/refill'):
        train(c,log)
    assert log.metrics['current_step']==0 and log.metrics['ngu_stats']['samples']<=c['batch_size']*2*c['grpo_group_size']
    assert not (Path(c['output_dir'])/'final_model').exists()

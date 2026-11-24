import json
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('homellm.training.rl.loss')
from backend.routes.vla import VLATrainingRequest
from backend.services import vla_grpo as rl


def config(kind='continuous',**extra):
    return VLATrainingRequest(training_algorithm='grpo',action_type=kind,action_dim=2,action_horizon=3,
        discrete_classes=[2,3] if kind=='hybrid' else [],**extra).model_dump()


@pytest.mark.parametrize('kind',['continuous','discrete','hybrid'])
def test_real_distribution_likelihood_gradient_and_padding(kind):
    torch.manual_seed(1);c=config(kind);dim=2+(5 if kind=='hybrid' else 0)
    output=torch.zeros(2,3,dim,requires_grad=True)
    actions=rl.sample(output.detach(),c)
    assert not actions.requires_grad and actions.shape[:3]==(2,8,3)
    lp=rl.log_probs(output,actions,c)
    targets=torch.zeros((2,3) if kind=='discrete' else (2,3,4 if kind=='hybrid' else 2))
    mask=torch.tensor([[True,False,False],[True,True,False]])
    rewards=rl.reward(actions,targets,mask,c);adv=rl.relative_advantages(rewards,c)
    assert torch.allclose(adv.mean(-1),torch.zeros(2),atol=1e-6)
    loss,_=rl.objective(output,actions,lp.detach(),adv,mask,c,reference=output.detach())
    loss.backward()
    assert torch.isfinite(output.grad).all() and output.grad.abs().sum()>0
    assert output.grad[0,1:].count_nonzero()==0 and output.grad[1,2:].count_nonzero()==0
    changed=actions.clone();changed[:, :, 2]=100 if kind!='discrete' else 1
    assert torch.equal(rewards,rl.reward(changed,targets,mask,c))


@pytest.mark.parametrize('algorithm',['grpo','dr_grpo','dapo'])
def test_constant_reward_has_no_policy_gradient(algorithm):
    c=config();c['training_algorithm']=algorithm
    out=torch.zeros(1,3,2,requires_grad=True);samples=rl.sample(out,c)
    adv=rl.relative_advantages(torch.ones(1,8),c)
    assert adv.count_nonzero()==0
    lp=rl.log_probs(out,samples,c)
    loss,_=rl.objective(out,samples,lp.detach(),adv,torch.ones(1,3,dtype=torch.bool),c)
    loss.backward();assert out.grad.count_nonzero()==0


def test_analytic_kl_anchors_policy_and_ratio_clips():
    c=config();c['grpo_kl_beta']=1.
    out=torch.full((1,3,2),.2,requires_grad=True);ref=torch.zeros_like(out)
    actions=rl.sample(out,c);lp=rl.log_probs(out,actions,c).detach()
    loss,parts=rl.objective(out,actions,lp,torch.zeros(1,8),torch.ones(1,3,dtype=torch.bool),c,reference=ref)
    loss.backward();assert parts['kl']>0 and (out.grad>0).all()
    out.grad=None
    loss,parts=rl.objective(out,actions,lp-1,torch.ones(1,8),torch.ones(1,3,dtype=torch.bool),c)
    loss.backward();assert parts['clip_fraction']==1 and out.grad.count_nonzero()==0


def test_grpo_api_records_method_and_resumes(client,workspace,monkeypatch):
    from backend.routes import vla
    data=workspace/'demo.jsonl';data.write_text('{}')
    monkeypatch.setattr(vla.processes,'launch',lambda *a,**kw:None)
    for algorithm in ('grpo','dr_grpo','dapo'):
        result=client.post('/api/vla/start',json=dict(dataset_path=str(data),training_algorithm=algorithm))
        assert result.status_code==200,result.text
        row=next(row for row in client.get('/api/vla/runs').json()['runs'] if row['run_id']==result.json()['run_id'])
        assert row['stage']=='vla_grpo' and row['config']['training_algorithm']==algorithm
    assert client.post('/api/vla/start',json=dict(dataset_path=str(data),training_algorithm='ppo')).status_code==422


def test_sampling_buffer_bound():
    with pytest.raises(ValueError,match='sample buffer'):
        VLATrainingRequest(training_algorithm='grpo',batch_size=128,gradient_accumulation=128,
            grpo_group_size=64,action_horizon=64,action_dim=64)

import pytest

torch=pytest.importorskip('torch')
from backend.services.ngu import collect,record


def test_later_success_keeps_every_failed_attempt_and_its_baseline():
    batches=iter([(['bad1','bad2'],torch.zeros(2)),(['good','bad3'],torch.tensor([1.,0.]))])
    result=collect(lambda:next(batches),max_attempts=4,continue_probability=1)
    assert result.status=='mixed' and result.rounds==2
    assert result.chunks==[['bad1','bad2'],['good','bad3']]
    assert result.rewards.tolist()==[0,0,1,0]
    from homellm.training.rl.loss import compute_advantages
    assert compute_advantages(result.rewards,use_std_normalization=False).tolist()==[-.25,-.25,.75,-.25]
    stats={};record(stats,result);assert stats==dict(prompts=1,rounds=2,samples=4,retries=1,mixed=1)


@pytest.mark.parametrize('values,p,expected,rounds',[(torch.ones(2),1,'easy',1),(torch.zeros(2),0,'give_up',1),(torch.zeros(2),1,'budget',3),(torch.tensor([0.,1.]),1,'mixed',1)])
def test_easy_hard_probability_and_budget_are_bounded(values,p,expected,rounds):
    calls=[]
    def draw():calls.append(1);return len(calls),values
    result=collect(draw,max_attempts=3,continue_probability=p,random_value=lambda:.5)
    assert result.status==expected and len(calls)==rounds


@pytest.mark.parametrize('rewards',[torch.tensor([.5]),torch.tensor([float('nan')]),torch.empty(0)])
def test_a_dense_reward_cannot_silently_be_treated_as_a_verifier(rewards):
    with pytest.raises(ValueError,match='binary verifier'):collect(lambda:(None,rewards),max_attempts=2,continue_probability=1)


def test_vla_verifier_ignores_padding_and_requires_all_categories():
    from backend.services.vla_grpo import verified_success
    c=dict(action_type='hybrid',action_dim=2,ngu_action_tolerance=.5)
    targets=torch.zeros(1,2,3);mask=torch.tensor([[True,False]])
    samples=torch.tensor([[[[.3,.3,0.],[100.,100.,2.]],[[.3,.3,1.],[0.,0.,0.]],[[.8,.8,0.],[0.,0.,0.]]]])
    assert verified_success(samples,targets,mask,c).tolist()==[[1.,0.,0.]]


def test_ngu_configuration_and_rollout_budget(client):
    from backend.routes.vla import VLATrainingRequest
    from backend.services.vlm_grpo import native_config
    from backend.routes.vlm import VLMTrainingRequest
    c=VLATrainingRequest(training_algorithm='ngu');assert c.stage=='vla_grpo' and c.reward_type=='demonstration_verifier'
    c=VLMTrainingRequest(grpo_algorithm='ngu');loss=native_config(c.model_dump())
    assert loss.algorithm.value=='grpo' and loss.dynamic_sampling and not loss.use_std_normalization
    with pytest.raises(ValueError,match='512 completions'):VLMTrainingRequest(stage='vlm_grpo',grpo_algorithm='ngu',grpo_group_size=64,ngu_max_attempts=64)
    assert VLMTrainingRequest(grpo_algorithm='ngu',grpo_group_size=64,ngu_max_attempts=64).stage=='vlm_sft'
    assert client.post('/api/vla/preflight',json={'training_algorithm':'ngu','ngu_continue_probability':1.1}).status_code==422


def test_vlm_collector_merges_real_experiences_before_advantages(tmp_path):
    from types import SimpleNamespace as NS
    from contextlib import nullcontext
    from PIL import Image
    from backend.services.vlm_native_trainer import NativeVLMTrainer
    from backend.services.vlm_grpo import native_config
    from homellm.training.rl.experience import ReplayBuffer
    image=tmp_path/'red.png';Image.new('RGB',(8,8),'red').save(image)
    trainer=NativeVLMTrainer.__new__(NativeVLMTrainer)
    trainer.config=dict(grpo_algorithm='ngu',grpo_group_size=2,grpo_max_new_tokens=1,grpo_temperature=1.,grpo_clip_eps_low=.2,grpo_clip_eps_high=.2,grpo_kl_weight=0.,grpo_reward_fn='exact_match',training_backend='transformers',ngu_max_attempts=3,ngu_continue_probability=1.)
    class Model:
        config=NS();device='cpu';generation_config=NS(eos_token_id=9)
        count=0
        def eval(self):return self
        def generate(self,**kwargs):
            self.count+=1
            return torch.tensor([[1,2,3],[1,2,3]] if self.count==1 else [[1,2,4],[1,2,3]])
    trainer.model=Model();trainer.torch=torch;trainer.rl=native_config(trainer.config)
    trainer.data=[dict(image_refs=[str(image)],messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'color?'}]},{'role':'assistant','content':'red'}])]
    trainer.processor=NS(tokenizer=NS(pad_token_id=0),batch_decode=lambda rows,**kw:['red' if row[0]==4 else 'blue' for row in rows])
    trainer.cursor=trainer.generated=trainer.filtered=0;trainer.ngu_stats={};trainer.buffer=ReplayBuffer();trainer.sidecars={}
    trainer.metrics=NS(update=lambda **kw:None,log_sample_output=lambda *a:None)
    trainer._encode=lambda *a:dict(input_ids=torch.tensor([[1,2]]),pixel_values=torch.ones(1,3,8,8))
    trainer._amp=nullcontext;trainer._training_mode=lambda:None
    trainer._log_probs=lambda model,inputs,length:torch.zeros(1,length)
    trainer._reference_log_probs=lambda *a:None
    added,rewards=trainer._collect()
    assert added and trainer.model.count==2 and len(trainer.buffer.items)==4
    assert rewards.tolist()==[0,0,1,0]
    assert [item.advantages.item() for item in trainer.buffer.items]==[-.25,-.25,.75,-.25]
    assert all(item.prompt_id==0 for item in trainer.buffer.items)
    assert trainer.ngu_stats['retries']==1

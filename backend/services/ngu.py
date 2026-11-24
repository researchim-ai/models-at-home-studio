"""Bounded synchronous Never Give Up sampling (Noukhovitch et al., 2026).
Same policy version across retries: no stale rollouts or asynchronous workers.
All failed attempts remain in the successful group's baseline and policy loss.
"""
from dataclasses import dataclass
import random


@dataclass
class SamplingResult:
    chunks: list
    rewards: object
    status: str
    rounds: int


def collect(draw, *, max_attempts, continue_probability, random_value=None):
    """draw -> (payload, 1D binary rewards). Stop easy/mixed groups, bound hard ones."""
    import torch
    if not 1<=max_attempts<=64 or not 0<=continue_probability<=1:raise ValueError('Invalid NGU sampling budget')
    chunks=[];values=[];rng=random_value or random.random
    for attempt in range(1,max_attempts+1):
        payload,rewards=draw();rewards=rewards.detach().float().reshape(-1)
        if not rewards.numel() or not torch.isfinite(rewards).all() or not ((rewards==0)|(rewards==1)).all():
            raise ValueError('NGU requires finite binary verifier rewards')
        chunks.append(payload);values.append(rewards)
        merged=torch.cat(values)
        if merged.min()!=merged.max():return SamplingResult(chunks,merged,'mixed',attempt)
        if bool(merged.all()):return SamplingResult(chunks,merged,'easy',attempt)
        if attempt==max_attempts:return SamplingResult(chunks,merged,'budget',attempt)
        if rng()>=continue_probability:return SamplingResult(chunks,merged,'give_up',attempt)
    raise AssertionError('Unreachable')


def record(stats,result):
    stats['prompts']=stats.get('prompts',0)+1
    stats['rounds']=stats.get('rounds',0)+result.rounds
    stats['samples']=stats.get('samples',0)+result.rewards.numel()
    stats['retries']=stats.get('retries',0)+result.rounds-1
    stats[result.status]=stats.get(result.status,0)+1

"""Integrated fly language model: approved 4+4, token current retained post-attention."""
from dataclasses import dataclass,asdict
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from fly_attention import AttentionCache


@dataclass(frozen=True)
class LMConfig:
    pre_steps: int = 4
    post_steps: int = 4
    pad: int = 0
    bos: int = 1
    eos: int = 2


class LMState(NamedTuple):
    voltage: torch.Tensor
    spike: torch.Tensor
    cache: AttentionCache


def state_tensors(state):
    return (state.voltage,state.spike,*state.cache)


def copy_state_(destination,source):
    """Inference/staging only, after backward has finished; never inside functional forward."""
    with torch.no_grad():
        for dst,src in zip(state_tensors(destination),state_tensors(source)):dst.copy_(src)


class FlyLM(nn.Module):
    def __init__(self,core,interfaces,attention,config=None):
        super().__init__()
        self.core=core;self.interfaces=interfaces;self.attention=attention
        self.config=c=config or LMConfig()
        if min(c.pre_steps,c.post_steps)<1:raise ValueError('positive phase lengths required')
        if interfaces.config.dim!=attention.config.dim:raise ValueError('interface/attention dimension mismatch')
        if core.n!=interfaces.input.n:raise ValueError('interface/core size mismatch')

    def initial_state(self,batch):
        v,s=self.core.initial_state(batch)
        return LMState(v,s,self.attention.empty(batch))

    def detach(self,state):
        return LMState(state.voltage.detach(),state.spike.detach(),self.attention.detach(state.cache))

    def step(self,ids,state=None,active=None,reset=None,weights=None):
        if ids.ndim!=1 or ids.dtype!=torch.long:raise ValueError('ids must be int64[B]')
        if state is None:state=self.initial_state(ids.shape[0])
        active=ids.ne(self.config.pad) if active is None else active & ids.ne(self.config.pad)
        reset=ids.eq(self.config.bos) if reset is None else reset | ids.eq(self.config.bos)
        cache=self.attention.reset(state.cache,reset)
        w=self.core.weights() if weights is None else weights
        current=self.interfaces.input_current(ids)
        pre,pre_rate=self.core.advance(current,self.config.pre_steps,(state.voltage,state.spike),
                                       weights=w,active=active,reset=reset)
        provisional=self.interfaces.representation(pre,pre_rate)
        recalled=self.attention.read(provisional,cache,active)
        post,post_rate=self.core.advance(current+self.interfaces.feedback_current(recalled),
                                         self.config.post_steps,pre,weights=w,active=active)
        final=self.interfaces.representation(post,post_rate)
        logits=F.linear(final,self.interfaces.embedding.weight)
        logits=torch.where(active[:,None],logits,torch.zeros_like(logits))
        cache=self.attention.append(final,cache,active)
        return logits,LMState(*post,cache)

    def forward(self,ids,state=None,active=None,reset=None):
        if ids.ndim!=2 or ids.shape[0]<1:raise ValueError('ids must be nonempty [length,batch]')
        if state is None:state=self.initial_state(ids.shape[1])
        weights=self.core.weights();outputs=[]
        for t in range(ids.shape[0]):
            logits,state=self.step(ids[t],state,None if active is None else active[t],
                                    None if reset is None else reset[t],weights)
            outputs.append(logits)
        return torch.stack(outputs),state

    def loss(self,logits,targets,ids,active=None):
        valid=ids.ne(self.config.pad)&targets.ne(self.config.pad)
        if active is not None:valid=valid&active
        safe_targets=torch.where(valid,targets,torch.zeros_like(targets))
        losses=F.cross_entropy(logits.flatten(0,1),safe_targets.flatten(),reduction='none').reshape_as(targets)
        count=valid.sum()
        return (losses*valid).sum()/count.clamp_min(1),count

    def specification(self):
        return dict(lm=asdict(self.config),interfaces=asdict(self.interfaces.config),
                    attention=asdict(self.attention.config),neurons=self.core.n,edges=self.core.raw.numel(),
                    parameters=sum(p.numel() for p in self.parameters()),post_current='token_plus_feedback')


def select_token(logits,config,temperature=0.,uniform=None):
    """GPU greedy or inverse-CDF categorical sample. PAD/BOS excluded from generation."""
    if temperature<0:raise ValueError('temperature must be nonnegative')
    allowed=torch.arange(logits.shape[-1],device=logits.device)
    allowed=(allowed!=config.pad)&(allowed!=config.bos)
    scores=logits.masked_fill(~allowed,float('-inf'))
    if temperature==0:return scores.argmax(-1)
    if uniform is None:uniform=torch.rand(logits.shape[0],device=logits.device)
    cdf=(scores/temperature).softmax(-1).cumsum(-1)
    # First bin strictly above u; clamp to last allowed token for FP rounding.
    index=(cdf<=uniform[:,None]).sum(-1)
    last=torch.where(allowed,torch.arange(logits.shape[-1],device=logits.device),0).max()
    return index.clamp_max(last)


class CUDATokenStep:
    """Fixed-batch captured inference. Owns state; not valid across parameter updates."""
    def __init__(self,model,batch,temperature=0.):
        if model.core.raw.device.type!='cuda':raise ValueError('CUDA required')
        self.model=model;self.temperature=temperature
        self.state=model.initial_state(batch)
        self.ids=torch.full((batch,),model.config.bos,device='cuda',dtype=torch.long)
        self.active=torch.ones(batch,device='cuda',dtype=torch.bool)
        self.reset=torch.zeros_like(self.active)
        self.uniform=torch.zeros(batch,device='cuda')
        self.graph=torch.cuda.CUDAGraph()
        def run():
            logits,new=model.step(self.ids,self.state,self.active,self.reset)
            selected=select_token(logits,model.config,temperature,self.uniform)
            selected=torch.where(self.active & self.ids.ne(model.config.pad),selected,
                                 torch.full_like(selected,model.config.pad))
            copy_state_(self.state,new)
            return logits,selected
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad():
            with torch.cuda.stream(stream):
                for _ in range(3):run()
            torch.cuda.current_stream().wait_stream(stream)
            copy_state_(self.state,model.initial_state(batch))
            with torch.cuda.graph(self.graph):self.logits,self.selected=run()
            copy_state_(self.state,model.initial_state(batch))

    @torch.no_grad()
    def replay(self,ids,active=None,reset=None,uniform=None):
        self.ids.copy_(ids)
        if active is None:self.active.fill_(True)
        else:self.active.copy_(active)
        if reset is None:self.reset.fill_(False)
        else:self.reset.copy_(reset)
        if uniform is None:self.uniform.uniform_()
        else:self.uniform.copy_(uniform)
        self.graph.replay()
        return self.logits,self.selected  # Static outputs overwritten on next replay.

    @torch.no_grad()
    def generate(self,prompt,max_new_tokens):
        if prompt.ndim!=2 or prompt.shape[0]<1 or max_new_tokens<1:raise ValueError('nonempty prompt/positive token count required')
        copy_state_(self.state,self.model.initial_state(prompt.shape[1]))
        candidate=torch.full_like(prompt[0],self.model.config.pad)
        for ids in prompt:
            active=ids.ne(self.model.config.pad)
            _,selected=self.replay(ids,active)
            candidate=torch.where(active,selected,candidate)
        result=[candidate.clone()]
        finished=candidate.eq(self.model.config.eos)|candidate.eq(self.model.config.pad)
        for _ in range(max_new_tokens-1):
            _,selected=self.replay(candidate,~finished)
            candidate=selected.clone();result.append(candidate)
            finished=finished|candidate.eq(self.model.config.eos)
        return torch.stack(result)  # EOS followed by PAD; fixed length for graph execution.

    def close(self):
        self.graph.reset()

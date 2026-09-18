"""Diagnostic FP32 Moonlight-style Muon / ROOT fallback + PyTorch AdamW.

Equations: MoonshotAI/Moonlight examples/toy_train.py;
huawei-noah/noah-research ROOT/examples/example_train.py (access 2026-09-12).
ROOT has no calibrated coefficients for our shapes: explicitly use its fallback.
Not a production optimizer or an exact reproduction of the authors' training.
"""
import math
import torch


def orthogonalize(g):
    tall=g.shape[0]>g.shape[1]
    x=g.T if tall else g
    x=x/(x.norm()+1e-7)
    for _ in range(5):
        a=x@x.T
        x=3.4445*x+(-4.775*a+2.0315*(a@a))@x
    return x.T if tall else x


def robust_component(g):
    # Static Python ranks; CUDA values. No tensor-dependent host branch in capture.
    rank=.9*(g.numel()-1);lo=math.floor(rank);hi=math.ceil(rank)
    flat=g.abs().flatten()
    low=flat.kthvalue(lo+1).values
    high=flat.kthvalue(hi+1).values
    epsilon=torch.lerp(low,high,rank-lo)+1e-9
    return g-g.sign()*torch.relu(g.abs()-epsilon)


def matrix_names(model):
    # Sparse port weights are indexed edge lists, despite their 2D storage.
    return {name+'.weight' for name,module in model.named_modules()
            if isinstance(module,torch.nn.Linear)}


class Hybrid:
    def __init__(self,model,variant,lr=1e-4,betas=(.9,.95)):
        self.params=list(model.parameters());self.variant=variant;self.lr=lr
        names=matrix_names(model)
        self.matrices=[p for n,p in model.named_parameters() if n in names]
        aux=[p for n,p in model.named_parameters() if n not in names]
        self.adam=torch.optim.AdamW(aux,lr=lr,betas=betas,eps=1e-8,
                                    weight_decay=0,capturable=True,foreach=False)
        self.state=self.adam.state
        for p in self.matrices:self.state[p]={'momentum':torch.zeros_like(p)}

    def zero_grad(self,set_to_none=False):
        for p in self.params:
            if set_to_none:p.grad=None
            elif p.grad is not None:p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p in self.matrices:
            if p.grad is None:continue
            buf=self.state[p]['momentum'];buf.mul_(.95).add_(p.grad)
            g=p.grad+.95*buf
            if self.variant=='root':g=robust_component(g)
            update=orthogonalize(g)
            p.add_(update,alpha=-self.lr*.2*math.sqrt(max(p.shape)))
        self.adam.step()


class HybridMuon:
    """Phase 7e (17 September 2026): Muon (Moonlight equations above, FP32) on the dense nn.Linear matrices, plain
    torch.optim.Adam (capturable, same betas as the baseline) on everything else, built inside FairCapture through the
    pretrain_control monkeypatch. Every state entry is a CUDA tensor so that FairCapture.restore() can zero it and
    pretrain_resumable's 'step' assertion holds for the Muon parameters too."""
    def __init__(self,model,params,lr,muon_lr,include_head=False,adam_groups=None,adam_cls=None,accumulate=None,**adam_kw):
        # adam_cls: the ORIGINAL torch.optim.Adam. pretrain_control monkeypatches torch.optim.Adam to build this object,
        # so calling torch.optim.Adam from here would recurse (phase 7e smoke, 17 September 04:30).
        # accumulate: (param_list, lr, every) — E5 (phase 8, 18 September 2026, after the E0 panel: gradient SNR on the
        # synapses 0.22 = pure noise at 2 lanes x 16 tokens): these parameters get Adam on the gradient summed over
        # `every` consecutive updates (effective batch x every for them only), applied every `every`-th update. Pure
        # tensor arithmetic (gate = 0/1 tensor), so it captures into the CUDA graph like the rest of the step.
        Adam=adam_cls or torch.optim.Adam
        params=list(params)
        names=matrix_names(model)
        if include_head and hasattr(model,'head'):names=names|{'head'}
        named={id(p):n for n,p in model.named_parameters()}
        self.matrices=[p for p in params if named.get(id(p)) in names]
        matrix_ids={id(p) for p in self.matrices}
        self.accumulated=list(accumulate[0]) if accumulate else []
        acc_ids={id(p) for p in self.accumulated}
        self.acc_lr,self.acc_every=(float(accumulate[1]),int(accumulate[2])) if accumulate else (0.,1)
        self.acc_betas=adam_kw.get('betas',(.9,.999));self.acc_eps=adam_kw.get('eps',1e-8)
        self.acc_state={p:{'acc':torch.zeros_like(p),'m':torch.zeros_like(p),'v':torch.zeros_like(p),
                           'applied':torch.zeros((),dtype=torch.float32,device=p.device),
                           'step':torch.zeros((),dtype=torch.float32,device=p.device)} for p in self.accumulated}
        aux=[p for p in params if id(p) not in matrix_ids and id(p) not in acc_ids]
        if adam_groups:  # [(param_list, lr)] for parameters wanting their own learning rate (per-type core parameters)
            special={id(p):g_lr for group,g_lr in adam_groups for p in group}
            rest=[p for p in aux if id(p) not in special]
            groups=[dict(params=rest)]+[dict(params=[p for p in aux if special.get(id(p))==g_lr],lr=g_lr) for _,g_lr in adam_groups]
            self.adam=Adam(groups,lr=lr,**adam_kw)
        else:
            self.adam=Adam(aux,lr=lr,**adam_kw)
        self.params=params;self.lr=lr;self.muon_lr=muon_lr
        # Muon states live in their own dict (Adam's state_dict must only see Adam parameters); `state` is a merged
        # read view for FairCapture.restore() (zeroes every tensor) and pretrain_resumable's 'step' assertion.
        self.muon_state={p:{'momentum':torch.zeros_like(p),'step':torch.zeros((),dtype=torch.float32,device=p.device)} for p in self.matrices}
        import collections
        self.state=collections.ChainMap(self.muon_state,self.acc_state,self.adam.state)  # muon first: Adam's state is a defaultdict
        self.param_groups=self.adam.param_groups
        self.matrix_names=sorted(named[id(p)] for p in self.matrices)
        self.accumulated_names=sorted(named[id(p)] for p in self.accumulated)

    def zero_grad(self,set_to_none=False):
        for p in self.params:
            if set_to_none:p.grad=None
            elif p.grad is not None:p.grad.zero_()

    def state_dict(self):
        return dict(adam=self.adam.state_dict(),muon=dict(lr=self.muon_lr,matrices=self.matrix_names,
                    state=[{k:v for k,v in self.muon_state[p].items()} for p in self.matrices]),
                    accumulated=dict(lr=self.acc_lr,every=self.acc_every,names=self.accumulated_names,
                    state=[{k:v for k,v in self.acc_state[p].items()} for p in self.accumulated]))

    @torch.no_grad()
    def step(self):
        for p in self.matrices:
            if p.grad is None:continue
            st=self.muon_state[p];buf=st['momentum'];buf.mul_(.95).add_(p.grad)
            g=p.grad+.95*buf
            p.add_(orthogonalize(g),alpha=-self.muon_lr*.2*math.sqrt(max(p.shape)))
            st['step'].add_(1)
        b1,b2=self.acc_betas;every=self.acc_every
        for p in self.accumulated:
            if p.grad is None:continue
            st=self.acc_state[p];st['acc'].add_(p.grad);st['step'].add_(1)
            gate=(torch.remainder(st['step'],every)==0).to(p.dtype)  # 1 on every `every`-th update, else 0 (no host branch)
            st['applied'].add_(gate)
            g=st['acc']/every
            st['m'].lerp_(b1*st['m']+(1-b1)*g,gate)  # moments move only on gated updates
            st['v'].lerp_(b2*st['v']+(1-b2)*g*g,gate)
            t=st['applied'].clamp_min(1)
            p.add_(-(gate*self.acc_lr)*(st['m']/(1-b1**t))/((st['v']/(1-b2**t)).sqrt()+self.acc_eps))
            st['acc'].mul_(1-gate)
        self.adam.step()


def make_optimizer(model,variant):
    if variant=='adamw':
        return torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),
                                weight_decay=0,capturable=True,foreach=False)
    if variant not in ('muon','root'):raise ValueError(variant)
    return Hybrid(model,variant)

"""Causal graph-state attention. Functional, fixed-shape cache for CUDA Graphs."""
from dataclasses import dataclass
from typing import NamedTuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


@dataclass(frozen=True)
class AttentionConfig:
    dim: int = 256
    heads: int = 4
    window: int = 128
    rope_base: float = 10000.0
    seed: int = 23


class AttentionCache(NamedTuple):
    k: torch.Tensor       # [B,H,W,Dh], already rotated
    v: torch.Tensor       # [B,H,W,Dh]
    positions: torch.Tensor  # [B,W], absolute positions within each story
    valid: torch.Tensor   # [B,W]
    next_position: torch.Tensor  # [B], position of current token


class MathSDPA(nn.Module):
    """Backend contract: q,k,v plus bool mask (True = allowed), no dropout."""
    def forward(self,q,k,v,mask):
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
                                                  dropout_p=0.0,is_causal=False)


class CausalAttention(nn.Module):
    def __init__(self,config=None,backend=None):
        super().__init__()
        self.config=c=config or AttentionConfig()
        if c.dim<1 or c.heads<1 or c.dim%c.heads or (c.dim//c.heads)%2:
            raise ValueError('dim must divide heads with an even head dimension')
        if c.window<1 or c.rope_base<=1: raise ValueError('invalid window/RoPE base')
        self.head_dim=c.dim//c.heads
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(c.seed)
            self.norm=nn.LayerNorm(c.dim)
            self.q=nn.Linear(c.dim,c.dim,bias=False)
            self.k=nn.Linear(c.dim,c.dim,bias=False)
            self.v=nn.Linear(c.dim,c.dim,bias=False)
            self.o=nn.Linear(c.dim,c.dim,bias=False)
        self.register_buffer('inv_frequency',c.rope_base**(-torch.arange(0,self.head_dim,2).float()/self.head_dim))
        self.register_buffer('first_slot',torch.arange(c.window)==0)
        self.backend=backend if backend is not None else MathSDPA()

    def empty(self,batch):
        if not isinstance(batch,int) or batch<1: raise ValueError('positive batch required')
        p=self.q.weight;c=self.config
        shape=(batch,c.heads,c.window,self.head_dim)
        return AttentionCache(p.new_zeros(shape),p.new_zeros(shape),
            torch.zeros(batch,c.window,dtype=torch.long,device=p.device),
            torch.zeros(batch,c.window,dtype=torch.bool,device=p.device),
            torch.zeros(batch,dtype=torch.long,device=p.device))

    @staticmethod
    def detach(cache):
        """Explicit TBPTT boundary; values/storage preserved, autograd cut."""
        return AttentionCache(*(x.detach() for x in cache))

    def _mask(self,mask,cache):
        if mask is None: return torch.ones_like(cache.next_position,dtype=torch.bool)
        if mask.shape!=cache.next_position.shape or mask.dtype!=torch.bool or mask.device!=cache.k.device:
            raise ValueError('mask must be bool[B] on cache device')
        return mask

    def reset(self,cache,reset):
        """Reset before reading a new story, including inactive reset slots."""
        if reset is None: return cache
        reset=self._mask(reset,cache)
        return AttentionCache(*(torch.where(reset.reshape((-1,)+(1,)*(x.ndim-1)),torch.zeros_like(x),x)
                                for x in cache))

    def _heads(self,x):
        return x.reshape(x.shape[0],self.config.heads,1,self.head_dim)

    def rotate(self,x,positions):
        """Adjacent-pair RoPE; x[B,H,1,Dh], positions[B]. Absolute, not modulo W."""
        angle=positions.to(self.inv_frequency.dtype)[:,None,None,None]*self.inv_frequency
        even,odd=x[...,0::2],x[...,1::2]
        cos,sin=angle.cos(),angle.sin()
        return torch.stack((even*cos-odd*sin,even*sin+odd*cos),dim=-1).flatten(-2)

    def read(self,provisional,cache,active=None):
        """Query current provisional representation; only previous final states readable."""
        active=self._mask(active,cache)
        q=self.rotate(self._heads(self.q(self.norm(provisional))),cache.next_position)
        allowed=cache.valid & (cache.positions<cache.next_position[:,None]) & active[:,None]
        any_valid=allowed.any(-1)
        # Guarantee one harmless zero slot for empty rows; no all-masked softmax.
        mask=allowed | (~any_valid[:,None] & self.first_slot)
        k=torch.where(allowed[:,None,:,None],cache.k,torch.zeros_like(cache.k))
        v=torch.where(allowed[:,None,:,None],cache.v,torch.zeros_like(cache.v))
        recalled=self.backend(q,k,v,mask[:,None,None,:]).reshape(provisional.shape[0],self.config.dim)
        return torch.where(any_valid[:,None],self.o(recalled),torch.zeros_like(recalled))

    def append(self,final,cache,active=None):
        """Call AFTER post-attention graph update; append K/V with gradients intact."""
        active=self._mask(active,cache)
        z=self.norm(final)
        k=self.rotate(self._heads(self.k(z)),cache.next_position)
        v=self._heads(self.v(z))
        candidates=(torch.cat((cache.k[:,:,1:],k),dim=2),
                    torch.cat((cache.v[:,:,1:],v),dim=2),
                    torch.cat((cache.positions[:,1:],cache.next_position[:,None]),dim=1),
                    torch.cat((cache.valid[:,1:],active[:,None]),dim=1),
                    cache.next_position+1)
        return AttentionCache(*(torch.where(active.reshape((-1,)+(1,)*(old.ndim-1)),new,old)
                                for new,old in zip(candidates,cache)))

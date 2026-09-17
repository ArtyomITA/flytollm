"""Trainable text interfaces; Anatomy setup followed by CUDA Graph tensor execution."""
from dataclasses import dataclass, asdict
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OUTPUT_CLASSES = ('descending_neuron', 'descending_neuron_tbc', 'vnc_motor', 'cb_motor',
                  'vnc_efferent', 'cb_efferent', 'efferent_descending', 'efferent_ascending')


@dataclass(frozen=True)
class InterfaceConfig:
    vocab: int = 4096
    dim: int = 256
    fan_in: int = 8
    pool_size: int = 32
    seed: int = 17
    input_gain: float = .6
    input_bias: float = .6
    feedback_gate: float = .1


def anatomical_ports(body_ids, annotations, pool_size=32):
    """CPU-only mapping; preserve graph node order; chunks are NOT physical regions."""
    import pyarrow.feather as pf
    if pool_size < 1: raise ValueError('pool_size must be positive')
    ids = np.asarray(body_ids)
    if len(np.unique(ids)) != len(ids): raise ValueError('body IDs must be unique')
    t = pf.read_table(annotations, columns=['bodyId','superclass','rootSide'])
    lookup = {row['bodyId']:(row['superclass'],row['rootSide']) for row in t.to_pylist()}
    sensory, buckets = [], {}
    for i, body in enumerate(ids):
        superclass, side = lookup[int(body)]
        if superclass and 'sensory' in superclass: sensory.append(i)
        if superclass in OUTPUT_CLASSES:
            buckets.setdefault((superclass,side or 'unknown'),[]).append(i)
    read_nodes, groups, labels = [], [], []
    for key, indices in sorted(buckets.items()):
        indices.sort(key=lambda i:int(ids[i]))
        for start in range(0,len(indices),pool_size):
            chunk = indices[start:start+pool_size]
            read_nodes.extend(chunk); groups.extend([len(labels)]*len(chunk))
            labels.append(dict(superclass=key[0],root_side=key[1],chunk=start//pool_size,nodes=len(chunk)))
    if not sensory or not read_nodes: raise ValueError('empty input/output anatomical selection')
    return (torch.tensor(sensory,dtype=torch.long), torch.tensor(read_nodes,dtype=torch.long),
            torch.tensor(groups,dtype=torch.long), labels)


class SparseInjector(nn.Module):
    def __init__(self, n, nodes, dim, fan_in, seed, gain=.6, bias=.6, bounded=True):
        super().__init__()
        if not 1 <= fan_in <= dim or nodes.numel()==0: raise ValueError('invalid ports/fan-in')
        nodes=nodes.to(device='cpu',dtype=torch.long)
        if nodes.unique().numel()!=nodes.numel() or nodes.min()<0 or nodes.max()>=n:
            raise ValueError('ports must be distinct compact node IDs')
        generator=torch.Generator().manual_seed(seed)
        # Consecutive fan-in positions in shuffled global dimension permutation:
        # no repeated channel per neuron; global channel load differs by at most1.
        permutation=torch.randperm(dim,generator=generator)
        channels=permutation[torch.arange(len(nodes)*fan_in).reshape(len(nodes),fan_in)%dim]
        self.register_buffer('nodes',nodes)
        self.register_buffer('channels',channels)
        self.weight=nn.Parameter(torch.randn(len(nodes),fan_in,generator=generator)/fan_in**.5)
        self.bias=nn.Parameter(torch.full((len(nodes),),float(bias))) if bounded else None
        self.n,self.dim,self.gain,self.bounded=n,dim,gain,bounded

    def forward(self, vector):
        values=(vector[:,self.channels]*self.weight).sum(-1)
        if self.bounded: values=self.bias+self.gain*torch.tanh(values)
        return vector.new_zeros(vector.shape[0],self.n).index_copy(1,self.nodes,values)


class PopulationReadout(nn.Module):
    def __init__(self,nodes,groups,dim):
        super().__init__()
        nodes=nodes.to(device='cpu',dtype=torch.long); groups=groups.to(device='cpu',dtype=torch.long)
        if len(nodes)!=len(groups) or not len(nodes) or nodes.unique().numel()!=len(nodes):
            raise ValueError('invalid readout nodes/groups')
        self.groups_count=int(groups.max())+1
        counts=torch.bincount(groups,minlength=self.groups_count)
        if groups.min()<0 or (counts==0).any(): raise ValueError('groups must be contiguous and nonempty')
        self.register_buffer('nodes',nodes); self.register_buffer('groups',groups)
        self.register_buffer('counts',counts.float())
        self.norm=nn.LayerNorm(2*self.groups_count)
        self.projection=nn.Linear(2*self.groups_count,dim,bias=False)

    def features(self,state,rate):
        # Preserve voltage and spike-rate as separate channels; mean pooling explicit.
        def pool(x):
            return x.new_zeros(x.shape[0],self.groups_count).index_add(1,self.groups,x[:,self.nodes])/self.counts
        return torch.cat((pool(state[0]),pool(rate)),dim=-1)

    def forward(self,state,rate):
        return self.projection(self.norm(self.features(state,rate)))


class TextInterfaces(nn.Module):
    def __init__(self,n,sensory,read_nodes,groups,config=None):
        super().__init__()
        self.config=config or InterfaceConfig()
        c=self.config
        if not 3 < c.vocab <= 65536 or c.dim<1: raise ValueError('invalid vocabulary/dimension')
        if torch.isin(sensory.cpu(),read_nodes.cpu()).any(): raise ValueError('input/output ports must be disjoint')
        # Constructor deterministic without changing caller's CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(c.seed)
            self.embedding=nn.Embedding(c.vocab,c.dim)
            nn.init.normal_(self.embedding.weight,std=.02)
            self.input_norm=nn.LayerNorm(c.dim)
            self.input=SparseInjector(n,sensory,c.dim,c.fan_in,c.seed,c.input_gain,c.input_bias)
            self.feedback=SparseInjector(n,sensory,c.dim,c.fan_in,c.seed+1,bounded=False)
            self.gate=nn.Parameter(torch.tensor(float(c.feedback_gate)))
            self.readout=PopulationReadout(read_nodes,groups,c.dim)
            self.output_norm=nn.LayerNorm(c.dim)

    def input_current(self,ids):
        return self.input(self.input_norm(self.embedding(ids)))

    def feedback_current(self,recalled):
        return self.gate*self.feedback(recalled)

    def representation(self,state,rate):
        return self.output_norm(self.readout(state,rate))

    def logits(self,state,rate):
        # No extra LM-head parameter; same E used for lookup and output.
        return F.linear(self.representation(state,rate),self.embedding.weight)

    def describe(self):
        return dict(config=asdict(self.config),sensory_nodes=len(self.input.nodes),
                    read_nodes=len(self.readout.nodes),groups=self.readout.groups_count,
                    parameters=sum(p.numel() for p in self.parameters()),
                    embedding_parameters=self.embedding.weight.numel(),
                    parameter_groups={name:p.numel() for name,p in self.named_parameters()},
                    mapping_sha256={name:hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
                                    for name,x in self.named_buffers()})

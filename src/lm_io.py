"""Host data/checkpoint I/O only; all model evaluation runs on CUDA."""
from dataclasses import asdict
import numpy as np
import torch
from fly_lm import FlyLM,LMConfig
from fly_core import Core
from fly_interfaces import TextInterfaces,InterfaceConfig,anatomical_ports
from fly_attention import CausalAttention,AttentionConfig


def build_cns(threshold=10):
    from fly_graph import load_graph,ROOT
    data=load_graph(threshold)
    n=len(data['body_ids']);counts=np.log1p(data['weight'])
    incoming=np.bincount(data['dst'],weights=counts,minlength=n)
    magnitude=(.5*counts/np.maximum(incoming[data['dst']],1)).astype(np.float32)
    core=Core(torch.from_numpy(data['src']),torch.from_numpy(data['dst']),torch.from_numpy(magnitude),torch.from_numpy(data['sign']),n)
    ports=anatomical_ports(data['body_ids'],ROOT/'dataset/male_cns/body-annotations-male-cns-v1.0.feather')
    interfaces=TextInterfaces(n,*ports[:3])
    return FlyLM(core,interfaces,CausalAttention()).cuda()


def story_batch(dataset,indices,starts,length):
    """No cross-story targets. EOS is predicted; EOS itself has no next target."""
    if len(indices)!=len(starts) or not indices or length<1:raise ValueError('invalid batch layout')
    ids=np.zeros((length,len(indices)),dtype=np.int64);targets=np.zeros_like(ids)
    for b,(index,start) in enumerate(zip(indices,starts)):
        if start<0:raise ValueError('negative story offset')
        story=dataset[index];count=max(0,min(length,len(story)-1-start))
        ids[:count,b]=story[start:start+count];targets[:count,b]=story[start+1:start+count+1]
    return torch.from_numpy(ids).cuda(),torch.from_numpy(targets).cuda()


def save_model(path,model,metadata=None):
    torch.save(dict(schema=1,specification=model.specification(),chunk=model.core.chunk,
                    state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    metadata=metadata or {}),path)


def load_model(path):
    data=torch.load(path,weights_only=True,map_location='cpu')
    if data['schema']!=1:raise ValueError('unknown checkpoint schema')
    spec=data['specification'];state=data['state_dict'];n=spec['neurons']
    core=Core(state['core.src'],state['core.dst'],torch.ones_like(state['core.raw']),torch.ones(n),n,data['chunk'])
    interfaces=TextInterfaces(n,state['interfaces.input.nodes'],state['interfaces.readout.nodes'],
                              state['interfaces.readout.groups'],InterfaceConfig(**spec['interfaces']))
    model=FlyLM(core,interfaces,CausalAttention(AttentionConfig(**spec['attention'])),LMConfig(**spec['lm']))
    model.load_state_dict(state)
    if model.specification()!=spec:raise ValueError('checkpoint configuration mismatch')
    return model.cuda(),data['metadata']

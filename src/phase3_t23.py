"""T2/T3: unchanged fly training, instrumented CUDA collection and linear probes."""
import argparse, gc, hashlib, json, random, sys, time, traceback
from pathlib import Path
from bench_runtime import emit, supervise, memory

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def manifest():
    out={}
    for split,seed,count in [('train',93016,80),('dev',93017,32)]:
        rng=random.Random(seed); rows=[]
        for _ in range(count):
            row=list(range(400,408))*4;rng.shuffle(row);rows.append(row)
        out[split]=rows
    assert not set(map(tuple,out['train'])) & set(map(tuple,out['dev']))
    p=ROOT/'configs/phase3_t23_manifest.json'
    if p.exists():assert json.loads(p.read_text())==out
    else:p.write_text(json.dumps(out,indent=2))
    return out

def pairs(rows):
    import torch
    result=[]
    for b in range(0,len(rows),2):
        x=torch.tensor([[1]+r+[0]*7 for r in rows[b:b+2]],device='cuda').T.contiguous()
        y=x.clone();y[0]=0
        result.append([(x[t:t+8],y[t:t+8]) for t in range(0,40,8)])
    return result

def train(a,rows):
    import torch
    from lm_io import build_cns,save_model
    from phase3_extended import initialize,Captured
    model=build_cns();initialize(model,89)
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    save_model(ROOT/'results/phase3_t2_initial.pt',model,dict(protocol='PROTOCOLLO_T2_T3.md',updates=0))
    opt=torch.optim.Adam(model.parameters(),lr=.0001,capturable=True,foreach=False)
    engine=Captured(model,opt);tr=pairs(rows['train']);dev=pairs(rows['dev'])
    emit('initial_evaluation');before=engine.evaluate(dev)
    times=[];log=[];count=0;slow=0
    for pair in tr:
        engine.reset()
        for x,y in pair:
            torch.cuda.synchronize();started=time.perf_counter();metric=engine.update(x,y);torch.cuda.synchronize()
            if metric is None:continue
            elapsed=time.perf_counter()-started;times.append(elapsed);count+=metric['count']
            slow=slow+1 if elapsed>2 else 0
            if slow>=5:raise RuntimeError('Five consecutive updates >2s')
            if len(times)%32==0:log.append(dict(update=len(times),**metric));emit('training',update=len(times),loss=metric['loss'])
    assert len(times)==200 and count==2560
    emit('final_evaluation');after=engine.evaluate(dev);after_train=engine.evaluate(tr[:16])
    changes={}
    with torch.no_grad():
        for group in ('core','interfaces','attention'):
            selected=[(p,initial[n]) for n,p in model.named_parameters() if n.startswith(group+'.')]
            delta=sum((p-v).square().sum().item() for p,v in selected)
            base=sum(v.square().sum().item() for p,v in selected)
            changes[group]=dict(delta_l2=delta**.5,initial_l2=base**.5,relative_update=(delta/max(base,1e-30))**.5)
    save_model(ROOT/'results/phase3_t2_final.pt',model,dict(protocol='PROTOCOLLO_T2_T3.md',updates=200,targets=count))
    engine.close()
    return dict(before_dev=before,after_dev=after,after_train32=after_train,updates=200,valid_targets=count,train_s=sum(times),first_step_s=times[0],mean_step_s=sum(times)/200,changes=changes,log=log,
                checkpoints={n:sha(ROOT/'results'/n) for n in ['phase3_t2_initial.pt','phase3_t2_final.pt']})

def regions(model):
    import torch
    sensory=model.interfaces.input.nodes;read=model.interfaces.readout.nodes
    mask=torch.ones(model.core.n,device='cuda',dtype=torch.bool);mask[sensory]=False;mask[read]=False
    return {'sensory':sensory,'readout':read,'other':mask.nonzero().flatten()}

def summary(v):
    import torch
    return torch.stack([v.mean(),v.square().mean().sqrt(),v.min(),v.max(),(v==0).float().mean(),((v>=.9)&(v<=1.1)).float().mean(),(v>1).float().mean()])

class Collector:
    def __init__(self,model):
        import torch
        from fly_lm import copy_state_,state_tensors
        from fly_core import LIFReset
        self.model=model;self.regions=regions(model)
        self.ids=torch.ones(2,device='cuda',dtype=torch.long)
        self.state=model.initial_state(2);self.empty=model.initial_state(2)
        self.observed={};self.hooks=[];self.us=[]
        obs=self.observed
        self.hooks.append(model.interfaces.input_norm.register_forward_hook(lambda m,a,o:obs.__setitem__('embedding',o)))
        self.hooks.append(model.interfaces.input.register_forward_hook(lambda m,a,o:obs.__setitem__('current',o)))
        self.hooks.append(model.interfaces.feedback.register_forward_hook(lambda m,a,o:obs.__setitem__('feedback',o)))
        self.hooks.append(model.interfaces.readout.register_forward_pre_hook(lambda m,a:obs.setdefault('raw',[]).append(a)))
        self.hooks.append(model.interfaces.readout.norm.register_forward_pre_hook(lambda m,a:obs.setdefault('pooled',[]).append(a[0])))
        self.hooks.append(model.interfaces.output_norm.register_forward_hook(lambda m,a,o:obs.setdefault('representation',[]).append(o)))
        original=LIFReset.forward
        def instrument(ctx,u):
            self.us.append(torch.stack([summary(u[:,idx]) for idx in self.regions.values()]))
            return original(ctx,u)
        def capture_fn():
            obs.clear();self.us.clear()
            logits,new=model.step(self.ids,self.state)
            features={'embedding':obs['embedding'],'sensory_current':obs['current'][:,model.interfaces.input.nodes]}
            stats=[]
            for k,phase in enumerate(('pre','post')):
                state,rate=obs['raw'][k]
                nodes=model.interfaces.readout.nodes
                features['raw_'+phase]=torch.cat([state[0][:,nodes],rate[:,nodes]],-1)
                features['pooled_'+phase]=obs['pooled'][k]
                features['repr_'+phase]=obs['representation'][k]
                stats.append(torch.stack([torch.stack([summary(state[0][:,idx]),summary(rate[:,idx])]) for idx in self.regions.values()]))
            features['logits']=logits
            currents=torch.stack([obs['current'][:,model.interfaces.input.nodes].square().mean().sqrt(),
                                  (model.interfaces.gate*obs['feedback'][:,model.interfaces.input.nodes]).square().mean().sqrt()])
            substeps=torch.stack(self.us)
            copy_state_(self.state,model.detach(new))
            return features,torch.stack(stats),substeps,currents
        # Baseline forward before patch; hooks observe but do not change outputs.
        with torch.no_grad():reference,reference_state=model.step(self.ids,self.empty)
        self.graph=torch.cuda.CUDAGraph()
        try:
            LIFReset.forward=staticmethod(instrument)
            with torch.no_grad():capture_fn()
            copy_state_(self.state,self.empty)
            with torch.no_grad(),torch.cuda.graph(self.graph):self.out=capture_fn()
        finally:
            LIFReset.forward=staticmethod(original)
            for hook in self.hooks:hook.remove()
            self.hooks.clear();obs.clear();self.us.clear()
        copy_state_(self.state,self.empty);self.graph.replay()
        torch.testing.assert_close(self.out[0]['logits'],reference,rtol=2e-4,atol=2e-5)
        for x,y in zip(state_tensors(self.state),state_tensors(reference_state)):
            torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-5)
        self.reset()

    def reset(self):
        from fly_lm import copy_state_
        copy_state_(self.state,self.empty)

    def replay(self,ids):
        self.ids.copy_(ids);self.graph.replay()
        return self.out

    def close(self):self.graph.reset()

def probe(features,labels,permuted=False):
    import torch
    x,z=features[:1024],features[1024:]
    mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-6)
    x=(x-mean)/std;z=(z-mean)/std
    y=labels[:1024];truth=labels[1024:]
    if permuted:
        generator=torch.Generator(device='cuda').manual_seed(92)
        y=y[torch.randperm(len(y),device='cuda',generator=generator)]
    torch.manual_seed(91);model=torch.nn.Linear(x.shape[1],8,device='cuda')
    opt=torch.optim.Adam(model.parameters(),lr=.03,capturable=True,foreach=False)
    initial=[p.detach().clone() for p in model.parameters()]
    def step():
        opt.zero_grad(set_to_none=False);loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();opt.step()
        return loss
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):loss=step()
    with torch.no_grad():
        for p,v in zip(model.parameters(),initial):p.copy_(v)
        for state in opt.state.values():
            for v in state.values():v.zero_()
    for _ in range(100):graph.replay()
    eval_graph=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(eval_graph):
        scores=torch.stack([(model(x).argmax(-1)==y).float().mean(),(model(z).argmax(-1)==truth).float().mean()])
    eval_graph.replay();values=scores.tolist();last=float(loss.detach())
    assert __import__('math').isfinite(last)
    graph.reset();eval_graph.reset()
    return dict(train_accuracy=values[0],dev_accuracy=values[1],loss=last,updates=100,permuted=permuted)

def separation(features,labels):
    import torch
    centers=torch.stack([features[labels==k].mean(0) for k in range(8)])
    between=(centers-centers.mean(0)).square().mean()
    within=(features-centers[labels]).square().mean()
    return dict(between=float(between),within=float(within),ratio=float(between/within.clamp_min(1e-30)))

def gradient_diagnostic(model,rows):
    import torch
    regs=regions(model);current=[]
    ids=torch.tensor([[1]+row[:7] for row in rows[:2]],device='cuda').T.contiguous()
    hook=model.interfaces.input.register_forward_hook(lambda m,a,o:current.append(o))
    parameters=list(model.named_parameters())
    def calculate():
        current.clear();logits,_=model(ids)
        loss=torch.nn.functional.cross_entropy(logits[-1],ids[-1])
        grads=torch.autograd.grad(loss,current+[p for _,p in parameters],allow_unused=True)
        current_stats=torch.stack([torch.stack([torch.stack([g[:,idx].square().mean().sqrt(),g[:,idx].abs().max(),g[:,idx].ne(0).float().mean()]) for idx in regs.values()]) for g in grads[:8]])
        param_stats=torch.stack([torch.stack([g.square().mean().sqrt(),g.abs().max(),g.ne(0).float().mean()]) if g is not None else torch.zeros(3,device='cuda') for g in grads[8:]])
        return loss,current_stats,param_stats
    graph=torch.cuda.CUDAGraph()
    try:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):calculate()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):out=calculate()
    finally:hook.remove();current.clear()
    graph.replay()
    loss,stats,ps=out
    assert bool(torch.isfinite(stats).all() & torch.isfinite(ps).all() & torch.isfinite(loss))
    result=dict(loss=float(loss.detach()),current_by_position=stats.tolist(),regions=list(regs),columns=['rms','abs_max','nonzero_fraction'],parameter_gradients={n:v for (n,_),v in zip(parameters,ps.tolist())},scope='CE final token; BOS plus7 symbols; state empty; no gradient beyond8')
    graph.reset()
    return result

def diagnose(a,rows):
    import torch
    from lm_io import load_model
    model,_=load_model(ROOT/'results/phase3_t2_final.pt')
    emit('collector_capture');collector=Collector(model)
    features={k:torch.empty(2048,v.shape[-1],device='cuda') for k,v in collector.out[0].items()}
    labels=[];stats_sum=None;u_sum=None;curr_sum=None;position=0;stats_max=None;u_max=None
    started=time.perf_counter()
    for split in ('train','dev'):
        selected=rows[split][:32]
        for b in range(0,32,2):
            collector.reset();ids=torch.tensor([[1]+r for r in selected[b:b+2]],device='cuda').T
            for t,token in enumerate(ids):
                observed,stats,u,curr=collector.replay(token)
                if t==0:continue
                for name,value in observed.items():features[name][position:position+2].copy_(value)
                labels.extend((token-400).cpu().tolist());position+=2
                stats_sum=stats.clone() if stats_sum is None else stats_sum+stats
                u_sum=u.clone() if u_sum is None else u_sum+u
                curr_sum=curr.clone() if curr_sum is None else curr_sum+curr
                stats_max=stats.clone() if stats_max is None else torch.maximum(stats_max,stats)
                u_max=u.clone() if u_max is None else torch.maximum(u_max,u)
            emit('collection',split=split,pair=b//2+1,positions=position)
    labels=torch.tensor(labels,device='cuda');assert position==2048
    isolated={name:[] for name in features}
    for token in range(400,408,2):
        collector.reset();observed,_,_,_=collector.replay(torch.tensor([token,token+1],device='cuda'))
        for name,value in observed.items():isolated[name].append(value.clone())
    isolated_summary={}
    for name,values in isolated.items():
        v=torch.cat(values);dist=torch.pdist(v)
        isolated_summary[name]=dict(min_pair_distance=float(dist.min()),max_pair_distance=float(dist.max()),unique_rows=int(v.unique(dim=0).shape[0]))
    generation=[];collector.reset()
    prompt=rows['dev'][0][:8]
    for token in [1]+prompt:
        observed,stats,u,curr=collector.replay(torch.full((2,),token,device='cuda',dtype=torch.long))
    for _ in range(16):
        token=int(observed['logits'][0].argmax())
        if token in (0,2):break
        observed,stats,u,curr=collector.replay(torch.full((2,),token,device='cuda',dtype=torch.long))
        generation.append(dict(token=token,regional_state=stats.tolist(),pre_reset=u.tolist(),current_rms=curr.tolist()))
    telemetry=dict(regions=list(collector.regions),region_sizes={k:len(v) for k,v in collector.regions.items()},
                   columns=['mean','rms','min','max','zero_fraction','near_threshold_fraction','over_threshold_fraction'],
                   state_axes=['pre_post','region','voltage_rate','stat'],pre_reset_axes=['substep','region','stat'],
                   mean_state=(stats_sum/1024).tolist(),max_over_samples=stats_max.tolist(),mean_pre_reset=(u_sum/1024).tolist(),max_pre_reset=u_max.tolist(),
                   mean_current_rms=(curr_sum/1024).tolist(),generation=generation)
    collector.close();del collector,observed,stats,u,curr,isolated,values,v
    emit('probes')
    results={}
    for name,value in features.items():
        assert bool(torch.isfinite(value).all())
        results[name]=dict(probe=probe(value,labels),permuted=probe(value,labels,True),
                           train_separation=separation(value[:1024],labels[:1024]),dev_separation=separation(value[1024:],labels[1024:]))
        emit('probe_done',name=name,**results[name]['probe'],permuted_dev=results[name]['permuted']['dev_accuracy'])
    constant=probe(torch.ones(2048,1,device='cuda'),labels)
    logits=features['logits'];head={}
    for split,sl in [('train',slice(0,1024)),('dev',slice(1024,None))]:
        head[split]=dict(accuracy=float((logits[sl].argmax(-1)==labels[sl]+400).float().mean()),accuracy8=float((logits[sl,400:408].argmax(-1)==labels[sl]).float().mean()))
    del features,logits,value
    emit('current_gradients');gradients=gradient_diagnostic(model,rows['dev'])
    return dict(probes=results,constant_probe=constant,isolated=isolated_summary,telemetry=telemetry,gradients=gradients,head=head,positions=2048,collection_and_probes_s=time.perf_counter()-started,collector_equivalence=True,checkpoint_sha256=sha(ROOT/'results/phase3_t2_final.pt'))

def worker(a):
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(.75)
    emit('load',gpu=torch.cuda.get_device_name(),torch=torch.__version__,memory=memory())
    rows=manifest()
    sources=['phase3_t23.py','PROTOCOLLO_T2_T3.md','configs/phase3_t23_manifest.json','fly_core.py','fly_lm.py','fly_interfaces.py','fly_attention.py','phase3_extended.py','bench_runtime.py']
    hashes={f:sha(ROOT/f) for f in sources}
    result=train(a,rows) if a.case=='train' else diagnose(a,rows)
    peak=torch.cuda.max_memory_allocated()/2**20
    gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    final=torch.cuda.memory_allocated()/2**20
    result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak,source_sha256=hashes,config=vars(a),post_cleanup_memory=memory(),fallback_count=0)
    emit('cleanup',allocated_mb=final)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['train','diagnose'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_t23')
    try:result=worker(a)
    except Exception as exc:traceback.print_exc();result=dict(ok=False,error_type=type(exc).__name__,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

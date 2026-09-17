"""One-step cache interventions and TBPTT with equal optimizer frequency."""
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
from bench_runtime import emit,supervise,memory
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_rows():return json.loads((ROOT/'configs/phase3_t01b_manifest.json').read_text())['streams']

def blocks(rows,delay):
    import torch
    from phase3_t01 import examples
    result=[]
    for b in range(0,len(rows),2):
        x,y=examples(rows[b:b+2],delay)
        x=torch.tensor([r+[0]*8 for r in x],device='cuda').T.contiguous()
        y=torch.tensor([r+[0]*8 for r in y],device='cuda').T.contiguous()
        result.append([(x[t:t+16],y[t:t+16]) for t in range(0,48,16)])
    return result

class FairCapture:
    def __init__(self,model,length,x,y):
        import torch
        from fly_lm import copy_state_
        self.model=model;self.length=length
        self.ids=x.clone();self.targets=y.clone();self.state=model.initial_state(2);self.empty=model.initial_state(2)
        self.opt=opt=torch.optim.Adam(model.parameters(),lr=.0001,capturable=True,foreach=False)
        initial=[p.detach().clone() for p in model.parameters()]
        def restore():
            with torch.no_grad():
                for p,v in zip(model.parameters(),initial):p.copy_(v)
                for s in opt.state.values():
                    for v in s.values():v.zero_()
            opt.zero_grad(set_to_none=False);self.reset()
        def step():
            opt.zero_grad(set_to_none=False)
            total=(self.ids.ne(0)&self.targets.ne(0)).sum()
            state=self.state;loss_sum=self.ids.new_zeros((),dtype=torch.float32)
            for start in range(0,16,length):
                ids=self.ids[start:start+length];targets=self.targets[start:start+length]
                logits,new=model(ids,state);loss,n=model.loss(logits,targets,ids)
                part=loss*n/total.clamp_min(1);part.backward()
                loss_sum=loss_sum+part.detach();state=model.detach(new)
            finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False);opt.step()
            copy_state_(self.state,state)
            finite=finite & torch.isfinite(loss_sum) & torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()
            return loss_sum,total,norm,finite
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        emit('warmup',length=length)
        with torch.cuda.stream(stream):
            for _ in range(3):step()
        torch.cuda.current_stream().wait_stream(stream);restore()
        # Same detach contract in independent eager execution; no optimizer warmup retained.
        expected_loss,_,_,_=step();expected_loss=float(expected_loss)
        expected=[p.detach().clone() for p in model.parameters()]
        grads=[p.grad.detach().clone() for p in model.parameters()]
        restore();self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.out=step()
        restore();self.graph.replay();torch.cuda.synchronize()
        assert abs(float(self.out[0])-expected_loss)<1e-4
        for p,v,g in zip(model.parameters(),expected,grads):
            torch.testing.assert_close(p,v,rtol=1e-3,atol=1e-5)
            torch.testing.assert_close(p.grad,g,rtol=1e-3,atol=1e-5)
        self.equivalence=dict(loss_error=abs(float(self.out[0])-expected_loss),gradient_max_error=max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),grads)),weight_max_error=max(float((p-v).abs().max()) for p,v in zip(model.parameters(),expected)))
        restore()
        self.eval_ids=torch.ones(2,device='cuda',dtype=torch.long);self.eval_targets=self.eval_ids.clone()
        self.eval_graph=torch.cuda.CUDAGraph()
        def evaluate_step():
            logits,new=model.step(self.eval_ids,self.state)
            loss,n=model.loss(logits[None],self.eval_targets[None],self.eval_ids[None]);valid=self.eval_ids.ne(0)&self.eval_targets.ne(0)
            stats=torch.stack([loss*n,n,((logits.argmax(-1)==self.eval_targets)&valid).sum(),((logits[:,400:408].argmax(-1)+400==self.eval_targets)&valid).sum()])
            copy_state_(self.state,model.detach(new));return stats
        with torch.no_grad(),torch.cuda.graph(self.eval_graph):self.eval_out=evaluate_step()
        self.reset()

    def reset(self):
        from fly_lm import copy_state_
        copy_state_(self.state,self.empty)

    def update(self,x,y):
        self.ids.copy_(x);self.targets.copy_(y);self.graph.replay()
        loss,n,norm,finite=self.out
        if not bool(finite):raise RuntimeError('Nonfinite training')
        return dict(loss=float(loss),count=int(n),grad_norm=float(norm))

    def evaluate(self,pairs):
        sums=[0.]*4
        for index,pair in enumerate(pairs):
            self.reset()
            for x,y in pair:
                for token,target in zip(x,y):
                    self.eval_ids.copy_(token);self.eval_targets.copy_(target);self.eval_graph.replay()
                    values=self.eval_out.tolist()
                    for k,v in enumerate(values):sums[k]+=v
            if (index+1)%8==0:emit('evaluation',pairs=index+1)
        return dict(ce=sums[0]/sums[1],targets=int(sums[1]),accuracy=sums[2]/sums[1],accuracy8=sums[3]/sums[1])

    def close(self):self.graph.reset();self.eval_graph.reset();self.model.zero_grad(set_to_none=True)

def sensitivity(model,rows,delay,length):
    import torch
    pair=blocks(rows[:2],delay)[0][0];ids,targets=pair;currents=[]
    hook=model.interfaces.input.register_forward_hook(lambda m,a,o:currents.append(o))
    def calculate():
        currents.clear();state=model.initial_state(2)
        for start in range(0,16,length):
            logits,new=model(ids[start:start+length],state);state=model.detach(new)
        loss=torch.nn.functional.cross_entropy(logits[-1],targets[-1])
        grads=torch.autograd.grad(loss,currents,allow_unused=True)
        nodes=model.interfaces.input.nodes
        rms=torch.stack([g[:,nodes].square().mean().sqrt() if g is not None else loss.new_zeros(()) for g in grads])
        return loss.detach(),rms
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):calculate()
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):out=calculate()
    finally:hook.remove();currents.clear()
    graph.replay();loss,rms=out
    assert bool(torch.isfinite(rms).all())
    result=dict(loss=float(loss),sensory_current_gradient_rms=rms.tolist(),source_index=15-delay,source_rms=float(rms[15-delay]),length=length)
    if length==8:assert not bool(rms[:8].any())
    graph.reset();return result

def t5(a):
    import torch
    from lm_io import load_model,save_model
    model,_=load_model(ROOT/'results/phase3_t2_final.pt');rows=load_rows()
    train=blocks(rows['train'][:668],a.delay);dev=blocks(rows['dev'],a.delay)
    engine=FairCapture(model,a.length,*train[0][0]);equivalence=engine.equivalence
    entries=[];times=[];total=0;slow=0;curve=[]
    emit('training_start',length=a.length,delay=a.delay)
    for index,pair in enumerate(train):
        engine.reset()
        for segment,(x,y) in enumerate(pair):
            if len(times)==1000:break
            torch.cuda.synchronize();t=time.perf_counter();m=engine.update(x,y);torch.cuda.synchronize();dt=time.perf_counter()-t
            times.append(dt);total+=m['count'];entries.append([index*2,segment,m['count']])
            slow=slow+1 if dt>2 else 0
            if slow>=5:raise RuntimeError('Five consecutive updates >2s')
            if len(times)%32==0:emit('training',update=len(times),**m)
            if len(times) in (200,500,1000):
                from fly_lm import state_tensors
                saved=[v.clone() for v in state_tensors(engine.state)]
                curve.append(dict(update=len(times),dev=engine.evaluate(dev)))
                with torch.no_grad():
                    for v,s in zip(state_tensors(engine.state),saved):v.copy_(s)
                del saved,v,s
                emit('curve',**curve[-1])
        if len(times)==1000:break
    assert len(times)==1000
    emit('train_evaluation');tr=engine.evaluate(train[:16])
    emit('dev_evaluation');dv=engine.evaluate(dev)
    engine.close();del engine
    emit('sensitivity');grad=sensitivity(model,rows['dev'],a.delay,a.length)
    checkpoint=Path(a.output).with_suffix('.diagnostic.pt');save_model(checkpoint,model,dict(protocol='PROTOCOLLO_ESTENSIONE_FASE_3.md',length=a.length,delay=a.delay,updates=1000))
    return dict(curve=curve,length=a.length,delay=a.delay,updates=1000,valid_targets=total,stream_segments=entries,train=tr,dev=dv,train_s=sum(times),mean_step_s=sum(times)/1000,first_step_s=times[0],equivalence=equivalence,sensitivity=grad,
                initial_checkpoint_sha256=sha(ROOT/'results/phase3_t2_final.pt'),checkpoint_sha256=sha(checkpoint),data_sha256=sha(ROOT/'configs/phase3_t01b_manifest.json'))

def transformed(cache,attention,variant):
    import torch
    from fly_attention import AttentionCache
    if variant=='real':return cache
    if variant=='empty':return AttentionCache(torch.zeros_like(cache.k),torch.zeros_like(cache.v),cache.positions,torch.zeros_like(cache.valid),cache.next_position)
    if variant=='joint':return AttentionCache(cache.k.flip(2),cache.v.flip(2),cache.positions.flip(1),cache.valid.flip(1),cache.next_position)
    width=cache.valid.shape[1];slots=torch.arange(width,device=cache.k.device)[None].expand_as(cache.positions)
    start=width-cache.valid.sum(-1,keepdim=True)
    order=torch.where(cache.valid,width-1-(slots-start),slots)
    angle=cache.positions[:,None,:,None]*attention.inv_frequency[None,None,None,:]
    c,s=angle.cos(),angle.sin()
    even,odd=cache.k[...,0::2],cache.k[...,1::2]
    unrot=torch.stack([even*c+odd*s,odd*c-even*s],-1).flatten(-2)
    indices=order[:,None,:,None].expand_as(cache.k)
    k=unrot.gather(2,indices);v=cache.v.gather(2,indices)
    even,odd=k[...,0::2],k[...,1::2]
    rotated=torch.stack([even*c-odd*s,even*s+odd*c],-1).flatten(-2)
    return AttentionCache(rotated,v,cache.positions,cache.valid,cache.next_position)

def small_probe(data,labels,shuffle=False):
    import torch
    n=len(labels)//2;x,z=data[:n],data[n:];y=labels[:n];truth=labels[n:]
    mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(1e-6);x=(x-mean)/std;z=(z-mean)/std
    if shuffle:
        gen=torch.Generator(device='cuda').manual_seed(92);y=y[torch.randperm(n,device='cuda',generator=gen)]
    torch.manual_seed(91);model=torch.nn.Linear(x.shape[1],8,device='cuda');opt=torch.optim.Adam(model.parameters(),lr=.03,capturable=True,foreach=False)
    init=[p.detach().clone() for p in model.parameters()]
    def step():
        opt.zero_grad(set_to_none=False);loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();opt.step();return loss
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):loss=step()
    with torch.no_grad():
        for p,v in zip(model.parameters(),init):p.copy_(v)
        for state in opt.state.values():
            for v in state.values():v.zero_()
    for _ in range(100):graph.replay()
    evaluation=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(evaluation):scores=torch.stack([(model(x).argmax(-1)==y).float().mean(),(model(z).argmax(-1)==truth).float().mean()])
    evaluation.replay();v=scores.tolist();result=dict(train_accuracy=v[0],dev_accuracy=v[1],loss=float(loss.detach()),permuted=shuffle,train_targets_repeated=100*n)
    graph.reset();evaluation.reset();return result

def t4(a):
    import torch
    from lm_io import load_model,story_batch
    from fly_lm import LMState,copy_state_
    from text_dataset import StoryDataset
    checkpoint='phase3_t2_final.pt' if a.dataset=='symbols' else 'phase35_adam_s17.diagnostic.pt'
    model,_=load_model(ROOT/'results'/checkpoint)
    state=model.initial_state(2);empty=model.initial_state(2);ids=torch.ones(2,device='cuda',dtype=torch.long);targets=ids.clone()
    variants=['real','empty','content_reverse','joint']
    def step():
        weights=model.core.weights();outputs=[];newreal=None
        for variant in variants:
            changed=LMState(state.voltage,state.spike,transformed(state.cache,model.attention,variant))
            logits,new=model.step(ids,changed,weights=weights)
            if variant=='real':newreal=new
            outputs.append(logits)
        rows=[];valid=ids.ne(0)&targets.ne(0)
        for logits in outputs:
            loss,n=model.loss(logits[None],targets[None],ids[None]);diff=logits-outputs[0]
            rows.append(torch.stack([loss*n,n,((logits.argmax(-1)==targets)&valid).sum(),(diff.square().mean(-1)*valid).sum(),(diff.abs()*valid[:,None]).max(),((logits.argmax(-1)!=outputs[0].argmax(-1))&valid).sum()]))
        k=newreal.cache.k[:,:,-1,:];v=newreal.cache.v[:,:,-1,:]
        pos=newreal.cache.positions[:,-1];angle=pos[:,None,None]*model.attention.inv_frequency[None,None,:]
        c,s=angle.cos(),angle.sin();even,odd=k[...,0::2],k[...,1::2]
        unrot=torch.stack([even*c+odd*s,odd*c-even*s],-1).flatten(1)
        copy_state_(state,model.detach(newreal))
        return torch.stack(rows),unrot,v.flatten(1)
    emit('capture',dataset=a.dataset)
    with torch.no_grad():step()
    copy_state_(state,empty);graph=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(graph):out=step()
    all_pairs=[]
    if a.dataset=='symbols':
        data=json.loads((ROOT/'configs/phase3_t23_manifest.json').read_text())
        for split in ('train','dev'):
            for b in range(0,16,2):
                x=torch.tensor([[1]+r for r in data[split][b:b+2]],device='cuda').T.contiguous();y=x.clone();y[0]=0
                all_pairs.append((split,b,x,y))
    else:
        indices=json.loads((ROOT/'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
        with StoryDataset(ROOT/'dataset/prepared_v1','validation') as data:
            for b in range(0,16,2):
                x,y=story_batch(data,indices[b:b+2],[0,0],64);all_pairs.append(('dev',indices[b:b+2],x,y))
    metrics=[];keys=[];values=[];labels=[];started=time.perf_counter()
    for split,index,x,y in all_pairs:
        copy_state_(state,empty);sums=[[0.]*6 for _ in variants]
        for token,target in zip(x,y):
            ids.copy_(token);targets.copy_(target);graph.replay();stats,k,v=out
            assert bool(torch.isfinite(stats).all() & torch.isfinite(k).all() & torch.isfinite(v).all())
            for row,value in zip(sums,stats.tolist()):
                for j in range(6):row[j]=max(row[j],value[j]) if j==4 else row[j]+value[j]
            if a.dataset=='symbols' and bool(target.ne(0).all()):keys.append(k.clone());values.append(v.clone());labels.extend((target-400).tolist())
        metrics.append(dict(split=split,index=index,raw=sums));emit('interventions',dataset=a.dataset,pair=len(metrics))
    graph.reset();probe_results={}
    if keys:
        labels=torch.tensor(labels,device='cuda')
        for name,data in [('k_unrotated',torch.cat(keys)),('v',torch.cat(values))]:
            probe_results[name]=dict(real=small_probe(data,labels),permuted=small_probe(data,labels,True))
    aggregate={}
    for split in sorted(set(m['split'] for m in metrics)):
        agg=[]
        for i,name in enumerate(variants):
            rows=[m['raw'][i] for m in metrics if m['split']==split];n=sum(r[1] for r in rows)
            agg.append(dict(variant=name,targets=int(n),ce=sum(r[0] for r in rows)/n,accuracy=sum(r[2] for r in rows)/n,logit_delta_rms=(sum(r[3] for r in rows)/n)**.5,logit_delta_max=max(r[4] for r in rows),argmax_changed_fraction=sum(r[5] for r in rows)/n))
        aggregate[split]=agg
    return dict(checkpoint=checkpoint,checkpoint_sha256=sha(ROOT/'results'/checkpoint),dataset=a.dataset,aggregate=aggregate,per_pair=metrics,cache_probes=probe_results,elapsed_s=time.perf_counter()-started,one_step_only=True,
                joint_control_small_error=all(r[-1]['logit_delta_max']<1e-4 for r in aggregate.values()))

def worker(a):
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(.75);emit('load',memory=memory(),gpu=torch.cuda.get_device_name())
    hashes={f:sha(ROOT/f) for f in ['phase3_t5_long.py','PROTOCOLLO_ESTENSIONE_FASE_3.md','fly_lm.py','fly_core.py','fly_attention.py','fly_interfaces.py','bench_runtime.py']}
    result=t4(a) if a.case=='t4' else t5(a)
    peak=torch.cuda.max_memory_allocated()/2**20
    gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect();final=torch.cuda.memory_allocated()/2**20
    result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak,source_sha256=hashes,config=vars(a),post_cleanup_memory=memory(),fallback_count=0)
    emit('cleanup',allocated_mb=final);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['t4','t5'],required=True);p.add_argument('--dataset',choices=['symbols','text'],default='symbols');p.add_argument('--length',choices=[8,16],type=int,default=8);p.add_argument('--delay',choices=[1,8],type=int,default=1)
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true');p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_t5_long')
    try:r=worker(a)
    except Exception as exc:traceback.print_exc();r=dict(ok=False,error_type=type(exc).__name__,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(r,indent=2))
    if not r['ok']:sys.exit(1)

if __name__=='__main__':main()


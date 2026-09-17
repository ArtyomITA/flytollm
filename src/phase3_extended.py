"""Guarded phase3 memory, optimizer grid and validation experiments."""
import argparse,gc,hashlib,json,math,random,sys,time,traceback
from pathlib import Path
from bench_runtime import emit,supervise,memory
ROOT=Path(__file__).resolve().parent


def initialize(model,seed):
    import torch
    with torch.no_grad(),torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear):module.reset_parameters()
        model.interfaces.embedding.weight.normal_(0,.02)
        for port in (model.interfaces.input,model.interfaces.feedback):
            port.weight.normal_(0,1/math.sqrt(port.weight.shape[1]))


class Captured:
    def __init__(self,model,opt):
        import torch
        from fly_lm import copy_state_
        self.model=model;self.opt=opt
        self.ids=torch.ones((8,2),device='cuda',dtype=torch.long)
        self.targets=self.ids.clone();self.state=model.initial_state(2);self.empty=model.initial_state(2)
        self.eval_ids=torch.ones(2,device='cuda',dtype=torch.long);self.eval_targets=self.eval_ids.clone()
        self.graph=torch.cuda.CUDAGraph();self.eval_graph=torch.cuda.CUDAGraph()
        observed={}
        hooks=[model.interfaces.input.register_forward_hook(lambda m,args,out:observed.__setitem__('token',out)),
               model.interfaces.feedback.register_forward_hook(lambda m,args,out:observed.__setitem__('feedback',out)),
               model.interfaces.readout.register_forward_pre_hook(lambda m,args:observed.__setitem__('rate',args[1]))]
        def forward():
            logits,new=model.step(self.eval_ids,self.state)
            loss,count=model.loss(logits[None],self.eval_targets[None],self.eval_ids[None])
            valid=self.eval_ids.ne(0)&self.eval_targets.ne(0)
            correct=((logits.argmax(-1)==self.eval_targets)&valid).sum()
            restricted=((logits[:,400:408].argmax(-1)+400==self.eval_targets)&valid).sum()
            copy_state_(self.state,model.detach(new))
            rate=observed['rate'];nodes=model.interfaces.input.nodes
            stats=torch.stack([new.voltage.square().mean().sqrt(),new.voltage.abs().max(),rate.mean(),
                               (rate==0).float().mean(),(rate==1).float().mean(),model.interfaces.gate.clone(),
                               observed['token'][:,nodes].square().mean().sqrt(),
                               (model.interfaces.gate*observed['feedback'][:,nodes]).square().mean().sqrt()])
            observed.clear()
            return loss,count,correct,restricted,stats
        def train_function():
            opt.zero_grad(set_to_none=False)
            logits,new=model(self.ids,self.state)
            loss,count=model.loss(logits,self.targets,self.ids)
            with torch.no_grad():
                rate=observed['rate'];nodes=model.interfaces.input.nodes
                stats=torch.stack([new.voltage.square().mean().sqrt(),new.voltage.abs().max(),rate.mean(),
                                   (rate==0).float().mean(),(rate==1).float().mean(),model.interfaces.gate.clone(),
                                   observed['token'][:,nodes].square().mean().sqrt(),
                                   (model.interfaces.gate*observed['feedback'][:,nodes]).square().mean().sqrt()])
            observed.clear();loss.backward()
            finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
            opt.step();copy_state_(self.state,model.detach(new))
            return loss,count,stats,norm,finite & torch.isfinite(stats).all() & torch.isfinite(loss)
        initial=[p.detach().clone() for p in model.parameters()]
        def restore():
            with torch.no_grad():
                for p,v in zip(model.parameters(),initial):p.copy_(v)
                for st in opt.state.values():
                    for v in st.values():v.zero_()
            self.reset();opt.zero_grad(set_to_none=False)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        emit('warmup')
        with torch.cuda.stream(stream):
            for _ in range(3):train_function()
            with torch.no_grad():forward()
        torch.cuda.current_stream().wait_stream(stream);restore()
        emit('capture')
        with torch.cuda.graph(self.graph):self.train_out=train_function()
        restore()
        with torch.no_grad(),torch.cuda.graph(self.eval_graph):self.eval_out=forward()
        restore()
        for hook in hooks:hook.remove()
        observed.clear()

    def reset(self):
        from fly_lm import copy_state_
        copy_state_(self.state,self.empty)

    def stage(self,x,y):
        self.ids.copy_(x);self.targets.copy_(y)

    def update(self,x,y):
        import torch
        if not bool((x.ne(0)&y.ne(0)).any()):
            # Unscored prefix still supplies context; Adam is skipped, state is advanced.
            with torch.no_grad():
                for token,target in zip(x,y):
                    self.eval_ids.copy_(token);self.eval_targets.copy_(target);self.eval_graph.replay()
            return None
        self.stage(x,y);self.graph.replay()
        loss,count,stats,norm,finite=self.train_out
        if not bool(finite):raise RuntimeError('Nonfinite train state or gradients')
        return dict(loss=loss.item(),count=count.item(),stats=stats.tolist(),grad_norm=norm.item())

    def evaluate(self,pairs):
        import torch
        total=0.;count=correct=restricted=0;states=[];per_pair=[]
        with torch.no_grad():
            for pair in pairs:
                self.reset()
                pair_total=0.;pair_count=0
                for x,y in pair:
                    for token,target in zip(x,y):
                        self.eval_ids.copy_(token);self.eval_targets.copy_(target);self.eval_graph.replay()
                        loss,n,c,r,stats=self.eval_out
                        if not torch.isfinite(stats).all() or not torch.isfinite(loss):raise RuntimeError('Nonfinite eval')
                        total+=loss.item()*n.item();count+=n.item();correct+=c.item();restricted+=r.item()
                        pair_total+=loss.item()*n.item();pair_count+=n.item()
                    states.append(stats.tolist())
                per_pair.append(dict(ce=pair_total/max(pair_count,1),targets=pair_count))
        return dict(ce=total/max(count,1),accuracy=correct/max(count,1),restricted_accuracy=restricted/max(count,1),
                    targets=count,stats=states,per_pair=per_pair)

    def close(self):
        self.graph.reset();self.eval_graph.reset();self.model.zero_grad(set_to_none=True)


def text_pairs(indices,length,split='train'):
    from lm_io import story_batch
    from text_dataset import StoryDataset
    with StoryDataset(ROOT/'dataset/prepared_v1',split) as data:
        return [[story_batch(data,indices[b:b+2],[t,t],8) for t in range(0,length,8)]
                for b in range(0,len(indices),2)]


def synthetic_pairs(seed,delay):
    import torch
    rng=random.Random(seed)
    rows=[[rng.randrange(400,408) for _ in range(32)] for _ in range(8)]
    pairs=[]
    for b in range(0,8,2):
        xs=[];ys=[]
        for row in rows[b:b+2]:
            xs.append([1]+row+[0]*7)
            ys.append([0]+[0]*delay+row[:-delay]+[0]*7)
        x=torch.tensor(xs,device='cuda').T.contiguous();y=torch.tensor(ys,device='cuda').T.contiguous()
        pairs.append([(x[t:t+8],y[t:t+8]) for t in range(0,40,8)])
    return pairs,rows


def optimizer(model,a):
    import torch
    from optimizer_variants import Hybrid
    if a.variant in ('muon','root'):return Hybrid(model,a.variant,a.lr,betas=(.9,.999))
    cls=torch.optim.Adam if a.variant=='adam' else torch.optim.AdamW
    return cls(model.parameters(),lr=a.lr,betas=(.9,.999),weight_decay=0,capturable=True,foreach=False)


def long_stress(model):
    import torch
    from fly_lm import CUDATokenStep
    from text_dataset import StoryDataset
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as data:payload=data[0][1:-1].astype('int64').tolist()
    ids=[1]+[payload[i%len(payload)] for i in range(255)]
    decoder=CUDATokenStep(model,2);samples=[]
    try:
        for i,x in enumerate(ids):
            decoder.replay(torch.full((2,),x,device='cuda',dtype=torch.long))
            if (i+1)%32==0:
                v=decoder.state.voltage
                assert torch.isfinite(v).all()
                row=dict(tokens=i+1,voltage_rms=v.square().mean().sqrt().item(),voltage_abs_max=v.abs().max().item(),
                         next_position=decoder.state.cache.next_position.tolist(),valid=decoder.state.cache.valid.sum(-1).tolist())
                samples.append(row);emit('long_state',**row)
        assert samples[-1]['valid']==[128,128]
        return dict(samples=samples,passed=True,scope='cyclic payload, inference only, 256 tokens')
    finally:decoder.close()


def experiment(model,a):
    import torch
    from lm_io import save_model
    if a.case=='long':return long_stress(model)
    config=json.loads((ROOT/'configs/phase3_protocol_v1.json').read_text(encoding='utf8'))
    if a.case=='memory':
        train,train_rows=synthetic_pairs(171,a.delay);dev,dev_rows=synthetic_pairs(172,a.delay)
        assert not set(map(tuple,train_rows)) & set(map(tuple,dev_rows))
        epochs=10
    else:
        indices=list(range(2,18)) if a.case=='grid' else list(range(18,82))
        random.Random(a.seed).shuffle(indices)
        train=text_pairs(indices,32 if a.case=='grid' else 128)
        dev=text_pairs(config['validation_reserved'][:4] if a.case=='grid' else config['validation_reserved'],
                       32 if a.case=='grid' else 128,'validation')
        epochs=3 if a.case=='grid' else 1
    engine=Captured(model,optimizer(model,a))
    try:
        # Single-token captured evaluation must match full-window GPU eager reference.
        with torch.no_grad():
            x,y=train[0][0]
            expected_logits,expected_state=model(x)
            expected_loss,_=model.loss(expected_logits,y,x)
            check=engine.evaluate([[train[0][0]]])
            from fly_lm import state_tensors
            for actual,expected in zip(state_tensors(engine.state),state_tensors(expected_state)):
                torch.testing.assert_close(actual,expected,rtol=5e-4,atol=1e-5)
            assert abs(check['ce']-expected_loss.item())<1e-4
        del expected_logits,expected_state,expected_loss,actual,expected
        emit('initial_validation');initial=engine.evaluate(dev)
        initial_parameters={n:p.detach().clone() for n,p in model.named_parameters()}
        # Fixed output after completed training, not a validation-selected checkpoint.
        rows=[];updates=0;elapsed=0.;seen=0;timed_snapshot=None;times=[];slow_steps=0
        for epoch in range(epochs):
            for pair in train:
                engine.reset()
                for x,y in pair:
                    if a.case=='stability' and elapsed>=240:break
                    torch.cuda.synchronize();t=time.perf_counter()
                    metric=engine.update(x,y);torch.cuda.synchronize();dt=time.perf_counter()-t
                    if metric is None:continue
                    slow_steps=slow_steps+1 if dt>2 else 0
                    if a.allow_paging and slow_steps>=5:raise RuntimeError('Paging profile: five consecutive steps over2s')
                    elapsed+=dt;times.append(dt);updates+=1;seen+=metric['count']
                    if a.case=='grid' and elapsed<=20:
                        timed_snapshot=[p.detach().clone() for p in model.parameters()]
                        timed_updates=updates;timed_elapsed=elapsed
                    if updates%32==0:
                        row=dict(updates=updates,train_s=elapsed,**metric);rows.append(row);emit('train_progress',**row)
                if a.case=='stability' and elapsed>=240:break
        emit('final_validation');final=engine.evaluate(dev)
        group_changes={}
        for prefix in ('core','interfaces','attention'):
            selected=[(p,initial_parameters[n]) for n,p in model.named_parameters() if n.startswith(prefix+'.')]
            group_changes[prefix]=dict(delta_rms=math.sqrt(sum((p-old).square().sum().item() for p,old in selected)/sum(p.numel() for p,_ in selected)),
                                       finite=all(bool(torch.isfinite(p).all()) for p,_ in selected))
        if not all(x['finite'] for x in group_changes.values()):raise RuntimeError('Nonfinite weights')
        checkpoint=Path(a.output).with_suffix('.diagnostic.pt')
        if a.case=='stability':save_model(checkpoint,model,dict(protocol='PROTOCOLLO_FASE_3_B.md',seed=a.seed,lr=a.lr,updates=updates))
        timed=None
        if timed_snapshot is not None:
            with torch.no_grad():
                for p,v in zip(model.parameters(),timed_snapshot):p.copy_(v)
            timed=dict(updates=timed_updates,train_s=timed_elapsed,validation=engine.evaluate(dev))
        extra={}
        if a.case=='stability':
            extra['heldout12_initial']=aggregate_stats_eval(initial,dev,4)
            extra['heldout12_final']=aggregate_stats_eval(final,dev,4)
        passed=final['restricted_accuracy']>=.8 if a.case=='memory' else final['ce']<=.95*initial['ce'] if a.case=='stability' else None
        drift=False
        if a.case=='stability' and len(rows)>=6:
            import statistics
            first=statistics.median(r['stats'][0] for r in rows[:3]);last=[r['stats'][0] for r in rows[-3:]]
            drift=last[0]<last[1]<last[2] and min(last)>2*first
            passed=passed and not drift
        return dict(initial=initial,final=final,passed=passed,drift_flag=drift,updates=updates,seen_targets=seen,
                    train_s=elapsed,mean_step_s=sum(times)/len(times),rows=rows,group_changes=group_changes,
                    captured_eval_reference_verified=True,timed20s=timed,
                    checkpoint=str(checkpoint) if a.case=='stability' else None,**extra)
    finally:engine.close()


def aggregate_stats_eval(result,pairs,skip_stories):
    # Per-pair scalar totals are recorded by evaluate; no re-evaluation needed.
    records=result['per_pair'][skip_stories//2:]
    n=sum(r['targets'] for r in records)
    return dict(ce=sum(r['ce']*r['targets'] for r in records)/n,targets=n)


def worker(a):
    import torch
    from lm_io import build_cns,load_model
    torch.set_num_threads(1);emit('load',memory=memory(),torch=torch.__version__,gpu=torch.cuda.get_device_name())
    if a.case=='long':model,_=load_model(ROOT/'results/phase32_adam_lr1e4.diagnostic.pt')
    else:model=build_cns();initialize(model,a.seed)
    initial_sha=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    torch.cuda.reset_peak_memory_stats();result=experiment(model,a)
    peak=torch.cuda.max_memory_allocated()/2**20;del model
    emit('cleanup');gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    cleanup=dict(**memory(),allocated_mb=torch.cuda.memory_allocated()/2**20,reserved_mb=torch.cuda.memory_reserved()/2**20)
    assert cleanup['allocated_mb']==0,cleanup
    return dict(ok=True,case=a.case,variant=a.variant,lr=a.lr,seed=a.seed,delay=a.delay,result=result,
                initial_parameter_sha256=initial_sha,peak_vram_mb=peak,post_cleanup_memory=cleanup,fallback_count=0,
                source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                               ['phase3_extended.py','optimizer_variants.py','bench_runtime.py','PROTOCOLLO_FASE_3_B.md','fly_lm.py','fly_core.py']},
                allow_paging=a.allow_paging)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['long','memory','grid','stability'],required=True)
    p.add_argument('--variant',choices=['adam','adamw','muon','root'],default='adam')
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--delay',type=int,choices=[1,4,8],default=1)
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=300)
    p.add_argument('--max-vram-mb',type=int,default=7000)
    p.add_argument('--allow-paging',action='store_true');a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_extended')
    try:result=worker(a)
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error=str(exc),error_type=type(exc).__name__)
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if not result['ok']:sys.exit(1)


if __name__=='__main__':main()

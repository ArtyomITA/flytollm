"""Approved isolated variants; original core files remain untouched."""
import argparse,gc,json,sys,time,traceback,hashlib
from pathlib import Path
from bench_runtime import emit,supervise,memory
from phase3_t23 import ROOT,manifest,pairs,sha

def variant(model,name):
    import torch,numpy as np,pyarrow.feather as pf
    if name in ('p1','p2'):
        from fly_graph import load_graph
        ids=load_graph(10)['body_ids']
        ann=pf.read_table(ROOT/'dataset/male_cns/body-annotations-male-cns-v1.0.feather',columns=['bodyId','class','subclass','rootSide']).to_pylist()
        lookup={r['bodyId']:r for r in ann}
        def key(index):
            r=lookup[int(ids[index])]
            return (r['class'] or '',r['rootSide'] or '',r['subclass'] or '',int(ids[index]))
        if name=='p1':
            port=model.interfaces.input
            nodes=port.nodes.cpu().tolist();order=sorted(range(len(nodes)),key=lambda i:key(nodes[i]))
            old=port.channels.clone();new=old.clone()
            new[torch.tensor(order,device='cuda')]=old
            with torch.no_grad():port.channels.copy_(new)
            assert torch.equal(torch.bincount(old.flatten()),torch.bincount(new.flatten()))
        else:
            read=model.interfaces.readout;nodes=read.nodes.cpu().tolist()
            order=sorted(range(len(nodes)),key=lambda i:key(nodes[i]))
            new=read.groups.clone()
            # Keep every group capacity and all selected nodes; change membership only.
            new[torch.tensor(order,device='cuda')]=read.groups.sort().values
            with torch.no_grad():read.groups.copy_(new)
            assert torch.equal(torch.bincount(new).float(),read.counts)
    elif name=='d1':
        from fly_core import LIFReset
        def backward(ctx,gv,gs):
            (u,)=ctx.saved_tensors;s=(u>1).to(u.dtype);p=torch.sigmoid(2*(u-1))
            return gv*(1-s)+(gs-gv*u)*(2*p*(1-p))
        LIFReset.backward=staticmethod(backward)
    elif name=='d2':
        with torch.no_grad():model.interfaces.input.bias.fill_(.5)
    elif name=='h1':
        from fly_lm import FlyLM,LMState
        from torch.nn import functional as F
        class Untied(FlyLM):
            def step(self,ids,state=None,active=None,reset=None,weights=None):
                if state is None:state=self.initial_state(ids.shape[0])
                active=ids.ne(self.config.pad) if active is None else active & ids.ne(self.config.pad)
                reset=ids.eq(self.config.bos) if reset is None else reset | ids.eq(self.config.bos)
                cache=self.attention.reset(state.cache,reset)
                w=self.core.weights() if weights is None else weights
                current=self.interfaces.input_current(ids)
                pre,rate=self.core.advance(current,self.config.pre_steps,(state.voltage,state.spike),weights=w,active=active,reset=reset)
                provisional=self.interfaces.representation(pre,rate)
                recalled=self.attention.read(provisional,cache,active)
                post,rate=self.core.advance(current+self.interfaces.feedback_current(recalled),self.config.post_steps,pre,weights=w,active=active)
                final=self.interfaces.representation(post,rate)
                logits=F.linear(final,self.head)
                logits=torch.where(active[:,None],logits,torch.zeros_like(logits))
                return logits,LMState(*post,self.attention.append(final,cache,active))
        model.__class__=Untied;model.head=torch.nn.Parameter(model.interfaces.embedding.weight.detach().clone())
    return model

def run(a):
    import torch
    from lm_io import load_model
    from phase3_extended import Captured
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    model,_=load_model(ROOT/'results/phase3_t2_initial.pt')
    ids=torch.tensor([400,401],device='cuda')
    with torch.no_grad():before,_=model.step(ids)
    model=variant(model,a.variant)
    if a.variant in ('d1','h1'):
        with torch.no_grad():after,_=model.step(ids)
        torch.testing.assert_close(before,after,rtol=2e-4,atol=2e-5)
    rows=manifest();train=pairs(rows['train']);dev=pairs(rows['dev'])
    opt=torch.optim.Adam(model.parameters(),lr=.0001,capturable=True,foreach=False)
    engine=Captured(model,opt);curve=[];times=[];targets=0;slow=0
    initial=engine.evaluate(dev)
    for pair in train:
        engine.reset()
        for x,y in pair:
            torch.cuda.synchronize();start=time.perf_counter();m=engine.update(x,y);torch.cuda.synchronize()
            if m is None:continue
            dt=time.perf_counter()-start;times.append(dt);targets+=m['count'];slow=slow+1 if dt>2 else 0
            if slow>=5:raise RuntimeError('Five consecutive updates >2s')
            if len(times)%32==0:emit('training',variant=a.variant,update=len(times),**m)
        if len(times) in (50,100,200):
            curve.append(dict(update=len(times),dev=engine.evaluate(dev)));emit('curve',**curve[-1])
    assert len(times)==200 and targets==2560
    final=engine.evaluate(dev);tr=engine.evaluate(train[:16]);engine.close()
    path=Path(a.output).with_suffix('.experimental.pt')
    torch.save(dict(variant=a.variant,state_dict={n:p.detach().cpu() for n,p in model.state_dict().items()},
                    base_checkpoint='phase3_t2_initial.pt',experimental_loader='phase3_variants.variant'),path)
    return dict(variant=a.variant,initial_dev=initial,dev=final,train=tr,curve=curve,updates=200,targets=targets,train_s=sum(times),
                parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(path),
                initial_checkpoint_sha256=sha(ROOT/'results/phase3_t2_initial.pt'),audit_evaluated=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['base','p1','p2','d1','d2','h1'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true');p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_variants')
    try:
        import torch
        emit('load',memory=memory());result=run(a);peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20
        result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak,source_sha256=sha(__file__))
    except Exception as e:traceback.print_exc();result=dict(ok=False,error=str(e))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

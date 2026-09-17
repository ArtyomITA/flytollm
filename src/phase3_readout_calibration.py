"""R1/R2: calibrate existing readout, then evaluate the recurrent circuit."""
import argparse, gc, json, sys, traceback, time
from pathlib import Path
from bench_runtime import emit, supervise, memory
from phase3_t23 import ROOT, Collector, manifest, sha

def collect(model, rows):
    import torch
    c=Collector(model); features=[]; labels=[]; logits=[]; telemetry=[]
    for b in range(0,len(rows),2):
        c.reset()
        ids=torch.tensor([[1]+r for r in rows[b:b+2]],device='cuda').T
        for t,token in enumerate(ids):
            f,s,u,current=c.replay(token)
            if t:
                features.append(f['pooled_post'].clone());labels.append(token.clone())
                logits.append(f['logits'].clone());telemetry.append(torch.cat([s.flatten(),current]).clone())
        emit('collection',pair=b//2+1)
    c.close()
    return torch.cat(features),torch.cat(labels),torch.cat(logits),torch.stack(telemetry).mean(0)

def run(a):
    import torch
    from torch.nn import functional as F
    from lm_io import load_model, save_model
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    model,_=load_model(ROOT/'results/phase3_t2_final.pt');rows=manifest()
    x,y,base_train,_=collect(model,rows['train'][:32])
    z,truth,base_dev,base_stats=collect(model,rows['dev'])
    def metrics(scores,targets):
        return dict(ce=float(F.cross_entropy(scores,targets)),accuracy=float((scores.argmax(-1)==targets).float().mean()))
    before=dict(train=metrics(base_train,y),dev=metrics(base_dev,truth))
    for p in model.parameters():p.requires_grad_(False)
    modules=[model.interfaces.readout.norm,model.interfaces.readout.projection,model.interfaces.output_norm]
    params=[p for m in modules for p in m.parameters()]
    for p in params:p.requires_grad_(True)
    initial=[p.detach().clone() for p in params]
    def head(v):
        return F.linear(modules[2](modules[1](modules[0](v))),model.interfaces.embedding.weight)
    opt=torch.optim.Adam(params,lr=.001,capturable=True,foreach=False)
    def step():
        opt.zero_grad(set_to_none=False);loss=F.cross_entropy(head(x),y);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,foreach=False);opt.step()
        return torch.stack([loss.detach(),norm.detach()])
    def restore():
        with torch.no_grad():
            for p,v in zip(params,initial):p.copy_(v)
            for state in opt.state.values():
                for v in state.values():v.zero_()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):stats=step()
    restore()
    with torch.cuda.stream(stream):reference=step().clone()
    torch.cuda.current_stream().wait_stream(stream)
    expected=[p.detach().clone() for p in params]
    restore();graph.replay();torch.cuda.synchronize()
    torch.testing.assert_close(stats,reference,rtol=2e-4,atol=2e-5)
    for p,v in zip(params,expected):torch.testing.assert_close(p,v,rtol=2e-4,atol=2e-6)
    restore();curve=[];started=time.perf_counter()
    eg=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(eg):train_scores=head(x);dev_scores=head(z)
    for update in range(100):
        graph.replay()
        if (update+1)%10==0:
            eg.replay();torch.cuda.synchronize()
            if not bool(torch.isfinite(stats).all()):raise RuntimeError('Nonfinite calibration')
            curve.append(dict(update=update+1,train=metrics(train_scores,y),dev=metrics(dev_scores,truth)))
            emit('calibration',**curve[-1])
    train_s=time.perf_counter()-started;graph.reset();eg.reset()
    offline=curve[-1]
    save_model(Path(a.output).with_suffix('.diagnostic.pt'),model,dict(experiment='R1',updates=100,targets=102400))
    emit('recurrent_validation')
    _,yr,lr,after_stats=collect(model,rows['dev'])
    _,yn,ln,_=collect(model,rows['train'][32:64])
    return dict(before=before,offline=offline,recurrent_dev=metrics(lr,yr),new_contexts_train=metrics(ln,yn),
                telemetry_before=base_stats.tolist(),telemetry_after=after_stats.tolist(),curve=curve,
                train_s=train_s,updates=100,target_presentations=102400,trainable_parameters=sum(p.numel() for p in params),
                equivalence=True,architecture_changed=False,audit_evaluated=False,
                checkpoint_sha256=sha(ROOT/'results/phase3_t2_final.pt'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_readout_calibration')
    try:
        import torch
        emit('load',memory=memory());result=run(a)
        peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20
        result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak,source_sha256=sha(__file__),post_cleanup_memory=memory())
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

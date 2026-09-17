"""R0: exact tied-embedding gradient decomposition in CUDA Graph."""
import argparse,gc,json,sys,traceback
from pathlib import Path
from bench_runtime import emit,supervise,memory
from phase3_t23 import ROOT,sha,manifest

def diagnose(model,batches):
    import torch
    from torch.nn import functional as F
    ids=torch.ones(8,2,device='cuda',dtype=torch.long);target=torch.ones(2,device='cuda',dtype=torch.long)
    reps=[];currents=[]
    hooks=[model.interfaces.output_norm.register_forward_hook(lambda m,a,o:reps.append(o)),
           model.interfaces.input.register_forward_hook(lambda m,a,o:currents.append(o))]
    E=model.interfaces.embedding.weight
    def calc():
        reps.clear();currents.clear();logits,_=model(ids)
        loss=F.cross_entropy(logits[-1],target)
        total,*cg=torch.autograd.grad(loss,[E]+currents)
        direct_loss=F.cross_entropy(F.linear(reps[-1].detach(),E),target)
        direct,=torch.autograd.grad(direct_loss,E)
        lookup=total-direct
        dot=(direct*lookup).sum();den=direct.norm()*lookup.norm()
        summary=torch.stack([loss.detach(),total.norm(),direct.norm(),lookup.norm(),dot/den.clamp_min(1e-30),
                             (total-direct-lookup).abs().max(),den])
        current=torch.stack([g[:,model.interfaces.input.nodes].square().mean().sqrt() for g in cg])
        return summary,current
    graph=torch.cuda.CUDAGraph()
    try:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):calc()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):out=calc()
    finally:
        for h in hooks:h.remove()
        reps.clear();currents.clear()
    result=[]
    for index,(x,y) in enumerate(batches):
        ids.copy_(x);target.copy_(y);graph.replay()
        assert bool(torch.isfinite(out[0]).all() & torch.isfinite(out[1]).all())
        result.append(dict(pair=index,summary=out[0].tolist(),current_rms=out[1].tolist()))
        emit('gradient_pair',pair=index)
    graph.reset();return result

def run():
    import torch
    from lm_io import load_model,story_batch
    from text_dataset import StoryDataset
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    rows=manifest()['dev'];batches=[]
    for b in range(0,16,2):
        x=torch.tensor([[1]+r[:7] for r in rows[b:b+2]],device='cuda').T.contiguous()
        batches.append((x,x[-1].clone()))
    model,_=load_model(ROOT/'results/phase3_t2_final.pt')
    symbols=diagnose(model,batches);del model,batches
    model,_=load_model(ROOT/'results/phase35_adam_s17.diagnostic.pt');batches=[]
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds:
        for i in range(82,98,2):
            x,y=story_batch(ds,[i,i+1],[0,0],8)
            if bool((y[-1]==0).any()):raise RuntimeError('Invalid last target')
            batches.append((x,y[-1].clone()))
    return dict(symbols=symbols,text=diagnose(model,batches),columns=['ce','total_l2','output_l2','lookup_l2','cosine','reconstruction_max','cosine_denominator'],
                scope='CE last token;8 positions; fixed checkpoints; same E parameter space',audit_evaluated=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_embedding_gradients')
    try:
        import torch
        emit('load',memory=memory());result=run();peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20
        result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak,source_sha256=sha(__file__))
    except Exception as e:traceback.print_exc();result=dict(ok=False,error=str(e))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

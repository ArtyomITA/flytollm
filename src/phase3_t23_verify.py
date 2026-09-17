"""Focused validation of T2/T3 instrumentation on mixed causal prefixes."""
import argparse,gc,json,sys,traceback
from pathlib import Path
from bench_runtime import emit,supervise,memory

def verify():
    import torch
    from phase3_t23 import Collector,sha,ROOT
    from lm_io import load_model
    from fly_lm import state_tensors
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    emit('load',memory=memory())
    model,_=load_model(ROOT/'results/phase3_t2_final.pt')
    initial=torch.load(ROOT/'results/phase3_t2_initial.pt',map_location='cpu',weights_only=True)['state_dict']
    raw=initial['core.raw'].cuda();del initial
    with torch.no_grad():
        old=torch.nn.functional.softplus(raw);new=torch.nn.functional.softplus(model.core.raw)
        delta=float((new-old).norm()/old.norm())
    del raw,old,new
    collector=Collector(model);state=model.initial_state(2)
    tokens=[[1,1],[400,407],[401,406],[402,0],[403,405],[1,404],[406,403],[407,402]]
    errors=[]
    with torch.no_grad():
        for pair in tokens:
            ids=torch.tensor(pair,device='cuda')
            ref,state=model.step(ids,state)
            observed,stats,u,current=collector.replay(ids)
            actual=observed['logits']
            torch.testing.assert_close(actual,ref,rtol=2e-4,atol=2e-5)
            for x,y in zip(state_tensors(collector.state),state_tensors(state)):
                torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-5)
            assert bool(torch.isfinite(stats).all() & torch.isfinite(u).all() & torch.isfinite(current).all())
            errors.append(float((actual-ref).abs().max()))
    collector.close()
    return dict(prefixes=tokens,max_logit_errors=errors,all_state_tensors_close=True,telemetry_finite=True,effective_core_weight_relative_update=delta,
                checkpoints={n:sha(ROOT/'results'/n) for n in ['phase3_t2_initial.pt','phase3_t2_final.pt']},
                source_sha256={n:sha(ROOT/n) for n in ['phase3_t23.py','phase3_t23_verify.py']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=120);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_t23_verify')
    try:
        result=verify()
        import torch
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20
        result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=torch.cuda.max_memory_allocated()/2**20)
    except Exception as exc:traceback.print_exc();result=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

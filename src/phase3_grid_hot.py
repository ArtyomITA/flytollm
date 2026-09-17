"""Serial24-case grid in one guarded runtime; fresh weights/moments per case."""
import argparse,gc,hashlib,json,os,sys,threading,time,traceback
from pathlib import Path
from bench_runtime import supervise,emit,memory


def run_case(model,baseline,a,name):
    import torch
    from phase3_extended import initialize,experiment
    with torch.no_grad():
        for p,v in zip(model.parameters(),baseline):p.copy_(v)
    initialize(model,a.seed)
    initial_sha=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();pre=memory()
    emit('grid_case_begin',case=name,variant=a.variant,lr=a.lr,seed=a.seed)
    timer=threading.Timer(600,lambda:os._exit(124));timer.daemon=True;timer.start()
    try:result=experiment(model,a)
    finally:timer.cancel();timer.join()
    peak=torch.cuda.max_memory_allocated()/2**20
    gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    return dict(ok=True,case='grid',variant=a.variant,lr=a.lr,seed=a.seed,delay=1,result=result,
                initial_parameter_sha256=initial_sha,peak_vram_mb=peak,
                between_case_memory=dict(**memory(),allocated_mb=torch.cuda.memory_allocated()/2**20),
                pre_memory=pre,elapsed_s=time.perf_counter()-start,allow_paging=a.allow_paging,fallback_count=0,
                source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                               ['phase3_grid_hot.py','phase3_extended.py','optimizer_variants.py','bench_runtime.py','PROTOCOLLO_FASE_3_B.md','fly_lm.py','fly_core.py']})


def run(a):
    import torch
    from lm_io import build_cns
    from summarize_phase3 import read_case
    torch.set_num_threads(1);emit('load',memory=memory())
    model=build_cns();baseline=[p.detach().clone() for p in model.parameters()]
    resident=torch.cuda.memory_allocated()/2**20
    a.case='grid';a.delay=1;completed=[]
    for seed in (17,23):
        for lr in ('0.0001','0.0003','0.001'):
            for variant in ('adam','adamw','muon','root'):
                name=f'phase34_{variant}_s{seed}_lr{lr}'
                previous=read_case(name)
                if previous:
                    for f,h in previous['worker']['source_sha256'].items():
                        assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,(name,f)
                    completed.append(name);emit('grid_skip_verified',case=name);continue
                a.seed=seed;a.lr=float(lr);a.variant=variant
                result=run_case(model,baseline,a,name)
                remaining=result['between_case_memory']['allocated_mb']
                assert abs(remaining-resident)<1,dict(resident=resident,remaining=remaining)
                out=Path('results')/(name+'.json')
                assert not out.exists(),'Preserve existing failed case before retry'
                out.with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf8')
                out.write_text(json.dumps(dict(ok=True,result=result,guard_profile='paging_opt_in' if a.allow_paging else 'physical_ram_default',
                                               supervision='Shared live watchdog results/phase34_hot_run.jsonl; per-case600s timer',
                                               elapsed_s=result['elapsed_s']),indent=2),encoding='utf8')
                completed.append(name)
                emit('grid_case_done',case=name,ce=result['result']['final']['ce'],updates=result['result']['updates'])
    del model,baseline
    emit('cleanup');gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    assert torch.cuda.memory_allocated()==0
    return dict(ok=True,completed=completed,final_allocator_mb=0,
                notes='Cases sequential; weights reset and optimizer rebuilt each case; shared runtime only.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--output',default='results/phase34_hot_run.json')
    p.add_argument('--timeout',type=float,default=7200);p.add_argument('--phase-timeout',type=float,default=90)
    p.add_argument('--max-vram-mb',type=int,default=7000);a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_grid_hot')
    try:r=run(a)
    except Exception as exc:
        traceback.print_exc();r=dict(ok=False,error=str(exc),error_type=type(exc).__name__)
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    if not r['ok']:sys.exit(1)


if __name__=='__main__':main()

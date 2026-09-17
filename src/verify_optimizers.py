"""Guarded six-update optimizer compatibility probe, not a quality comparison."""
import argparse,gc,hashlib,json,sys,traceback
from pathlib import Path
from bench_runtime import supervise,emit


def run(a):
    import torch
    from verify_phase25 import build,carry_training
    from optimizer_variants import make_optimizer,matrix_names,robust_component,orthogonalize
    torch.set_num_threads(1);torch.manual_seed(89)
    emit('load');model=build(a.toy)
    # Independent primitive reference: sorted interpolated percentile, both shapes.
    with torch.no_grad():
        for shape in ((16,16),(16,6)):
            g=torch.randn(*shape,device='cuda');g[0,0]=100
            threshold=torch.quantile(g.abs(),.9)+1e-9
            torch.testing.assert_close(robust_component(g),g.clamp(-threshold,threshold))
            assert torch.isfinite(orthogonalize(robust_component(g))).all()
    del g,threshold
    names=matrix_names(model)
    groups={n:list(p.shape) for n,p in model.named_parameters() if n in names}
    matrix_count=sum(p.numel() for n,p in model.named_parameters() if n in names)
    torch.cuda.reset_peak_memory_stats()
    result=carry_training(model,lambda m:make_optimizer(m,a.variant))
    for row in result['metrics']:row['gpu_ms']['optimizer']=row['gpu_ms'].pop('adam')
    peak=torch.cuda.max_memory_allocated()/2**20
    spec=model.specification();del model
    emit('cleanup');gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    allocated=torch.cuda.memory_allocated()/2**20
    assert allocated==0,allocated
    return dict(ok=True,variant=a.variant,toy=a.toy,result=result,specification=spec,
                matrix_groups=groups,matrix_parameter_count=matrix_count,peak_vram_mb=peak,
                final_allocator_mb=allocated,fallback_count=0,
                root_coefficient_mode='official unknown-shape fallback; not calibrated',
                precision='FP32',lr=1e-4,betas=[.9,.95],weight_decay=0,
                source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                               ['optimizer_variants.py','verify_optimizers.py','verify_phase25.py']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['adamw','muon','root'],required=True)
    p.add_argument('--toy',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--output',required=True)
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=300)
    p.add_argument('--max-vram-mb',type=int,default=7000);a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='verify_optimizers')
    try:result=run(a)
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if not result['ok']:sys.exit(1)


if __name__=='__main__':main()

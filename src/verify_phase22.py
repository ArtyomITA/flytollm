"""Guarded GPU-only phase 2.2 checks; CUDA Graph failure is fatal."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from bench_runtime import supervise, emit, memory

ROOT=Path(__file__).resolve().parent


def worker(a):
    emit('load')
    started=time.monotonic()
    import numpy as np
    import torch
    from fly_core import Core
    from fly_interfaces import TextInterfaces, anatomical_ports
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1)
    torch.manual_seed(17)
    if a.toy:
        import unittest
        from test_interfaces import InterfaceTests, fixture
        emit('gpu_unit_tests')
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(InterfaceTests)
        tests=unittest.TextTestRunner(verbosity=2).run(suite)
        if not tests.wasSuccessful(): raise RuntimeError('GPU unit tests failed')
        model,core=fixture()
        anatomy={}
    else:
        from fly_graph import load_graph
        emit('anatomy_load')
        data=load_graph(10,progress=lambda message:emit('graph_load',message=message))
        path=ROOT/'dataset/male_cns/body-annotations-male-cns-v1.0.feather'
        sensory,read,groups,labels=anatomical_ports(data['body_ids'],path)
        n=len(data['body_ids'])
        magnitude=np.log1p(data['weight'])
        sums=np.bincount(data['dst'],weights=magnitude,minlength=n)
        magnitude=(.5*magnitude/np.maximum(sums[data['dst']],1)).astype(np.float32)
        core=Core(torch.from_numpy(data['src']),torch.from_numpy(data['dst']),
                  torch.from_numpy(magnitude),torch.from_numpy(data['sign']),n).cuda()
        model=TextInterfaces(n,sensory,read,groups).cuda()
        anatomy=dict(groups=labels,annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     body_ids_sha256=hashlib.sha256(data['body_ids'].tobytes()).hexdigest())
        del data,magnitude,sums
    description=model.describe()
    load_s=time.monotonic()-started
    params=list(model.parameters())+list(core.parameters())
    initial=[p.detach().clone() for p in params]
    optimizer=torch.optim.Adam(params,lr=1e-4,capturable=True,foreach=False,weight_decay=0)
    # Fixed storage, different contents across replay. Numeric fixtures are on GPU.
    ids=torch.tensor([4,5],device='cuda')
    targets=torch.tensor([6,7],device='cuda')
    feedback=torch.randn(2,model.config.dim,device='cuda')*.1
    batches=[(ids+i,targets+i,feedback*(i+1)) for i in range(3)]
    def stage(i):
        ids.copy_(batches[i][0]); targets.copy_(batches[i][1]); feedback.copy_(batches[i][2])
    def step():
        optimizer.zero_grad(set_to_none=False)
        state,rate=core.advance(model.input_current(ids)+model.feedback_current(feedback),8)
        loss=torch.nn.functional.cross_entropy(model.logits(state,rate),targets)
        loss.backward()
        optimizer.step()
        return loss,rate
    def reset():
        with torch.no_grad():
            for p,x in zip(params,initial): p.copy_(x)
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value): value.zero_()
        optimizer.zero_grad(set_to_none=False)
    emit('forward_backward_optimizer_warmup')
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    first=time.monotonic()
    with torch.cuda.stream(stream):
        for _ in range(3): step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    warmup_s=time.monotonic()-first
    reset()
    emit('gpu_eager_reference')
    expected_losses=[]
    for i in range(3):
        stage(i); loss,rate=step(); expected_losses.append(loss.detach().clone())
    expected=[p.detach().clone() for p in params]
    expected_grads=[p.grad.detach().clone() for p in params]
    expected_opt=[{k:v.detach().clone() for k,v in optimizer.state[p].items()} for p in params]
    del loss,rate
    reset(); stage(0)
    emit('capture')
    graph=torch.cuda.CUDAGraph()
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.graph(graph): static_loss,static_rate=step()
    reset()
    metrics=[]
    for i in range(3):
        stage(i); torch.cuda.synchronize(); start=time.perf_counter()
        graph.replay(); torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        torch.testing.assert_close(static_loss,expected_losses[i],rtol=2e-4,atol=2e-6)
        metrics.append(dict(loss=static_loss.item(),step_s=elapsed,rate=static_rate.mean().item()))
        emit('replay',step=i,**metrics[-1])
    emit('gpu_equivalence')
    errors={}
    names=[name for name,_ in model.named_parameters()]+['core.'+name for name,_ in core.named_parameters()]
    for name,p,e,g,o,x in zip(names,params,expected,expected_grads,expected_opt,initial):
        torch.testing.assert_close(p,e,rtol=2e-4,atol=2e-6)
        torch.testing.assert_close(p.grad,g,rtol=5e-4,atol=2e-6)
        for key,value in o.items():
            torch.testing.assert_close(optimizer.state[p][key],value,rtol=5e-4,atol=2e-6)
        if not torch.isfinite(p.grad).all() or p.grad.abs().max()==0:
            raise RuntimeError('Missing/nonfinite/zero gradient: '+name)
        if (p-x).abs().max()==0: raise RuntimeError('Unchanged parameter: '+name)
        errors[name]=dict(parameter_max_abs=(p-e).abs().max().item(),gradient_max_abs=(p.grad-g).abs().max().item(),
                          gradient_max=p.grad.abs().max().item(),update_max=(p-x).abs().max().item())
    result=dict(ok=True,scenario='phase22_toy' if a.toy else 'phase22_cns_t10',variant='cuda_graph_fp32',
                torch=torch.__version__,python=sys.executable,gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),interfaces=description,anatomy=anatomy,
                neurons=core.n,edges=core.raw.numel(),batch=2,substeps=8,metrics=metrics,equivalence=errors,
                model_load_s=load_s,warmup_three_steps_s=warmup_s,first_step_s=metrics[0]['step_s'],
                mean_step_s=sum(m['step_s'] for m in metrics)/len(metrics),loss_finite=True,grad_finite=True,
                peak_vram_mb=torch.cuda.max_memory_allocated()/2**20,
                gpu_alloc_mb=torch.cuda.memory_allocated()/2**20,gpu_reserved_mb=torch.cuda.memory_reserved()/2**20,
                fallback_count=0,blocking_fallback_count=0,semantic_verdict='interface_gradient_probe_only',
                notes='Three changing synthetic token/feedback batches; no attention, sequence LM or quality claim. GPU eager is correctness reference only.')
    if a.toy: result['unit_tests']=tests.testsRun
    emit('cleanup')
    graph.reset()
    # Drop every graph/closure/snapshot reference before checking the allocator.
    del graph,stream,step,stage,reset,model,core,params,initial,optimizer,ids,targets,feedback,batches
    del expected,expected_grads,expected_opt,expected_losses,static_loss,static_rate,p,e,g,o,x,value
    gc.collect();torch.cuda.synchronize()
    # Local torch build retains cuBLAS workspaces independently of live tensors.
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache();gc.collect()
    result['post_cleanup_memory']=dict(**memory(),gpu_alloc_mb=torch.cuda.memory_allocated()/2**20,
                                       gpu_reserved_mb=torch.cuda.memory_reserved()/2**20)
    emit('allocator_cleanup',**result['post_cleanup_memory'])
    if result['post_cleanup_memory']['gpu_alloc_mb']>=1:
        emit('live_local_tensors',tensors={k:list(v.shape) for k,v in locals().items() if torch.is_tensor(v) and v.is_cuda})
        raise RuntimeError('CUDA references remain after cleanup')
    result['stability_verdict']='stable'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toy',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--output',default=str(ROOT/'results/phase22_toy.json'))
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=300)
    p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker: return supervise(a,worker_module='verify_phase22')
    try: result=worker(a)
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error_type=type(exc).__name__,error=str(exc),fallback_count=0)
    gc.collect()
    if 'torch' in sys.modules:
        torch=sys.modules['torch']
        if torch.cuda.is_initialized():
            torch.cuda.synchronize();torch.cuda.empty_cache();gc.collect()
            result['final_allocator_mb']=torch.cuda.memory_allocated()/2**20
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['ok']:sys.exit(1)


if __name__=='__main__': main()

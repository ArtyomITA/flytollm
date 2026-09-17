"""Phase 2.3: guarded GPU numerical tests and mandatory CUDA Graph probes."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from bench_runtime import supervise,emit,memory

ROOT=Path(__file__).resolve().parent


def training_probe():
    import torch
    from fly_attention import CausalAttention
    model=CausalAttention().cuda()
    params=list(model.parameters());initial=[p.detach().clone() for p in params]
    opt=torch.optim.Adam(params,lr=1e-4,capturable=True,foreach=False,weight_decay=0)
    q=torch.randn(4,2,256,device='cuda');final=torch.randn_like(q);target=torch.randn_like(q)
    active=torch.ones(4,2,device='cuda',dtype=torch.bool)
    reset=torch.zeros_like(active)
    cases=[]
    for i in range(3):
        a=active.clone();r=reset.clone();a[1,i%2]=False;r[2,1]=i!=0
        cases.append((q*(i+1),final/(i+1),target+i*.1,a,r))
    def stage(i):
        for dest,src in zip((q,final,target,active,reset),cases[i]):dest.copy_(src)
    def step():
        opt.zero_grad(set_to_none=False)
        cache=model.empty(2);outputs=[]
        for t in range(4):
            cache=model.reset(cache,reset[t])
            outputs.append(model.read(q[t],cache,active[t]))
            cache=model.append(final[t],cache,active[t])
        output=torch.stack(outputs)
        loss=((output-target).square()*active[:,:,None]).sum()/(active.sum().clamp_min(1)*256)
        loss.backward();opt.step()
        return loss,output,cache
    def restore():
        with torch.no_grad():
            for p,x in zip(params,initial):p.copy_(x)
            for state in opt.state.values():
                for value in state.values():value.zero_()
        opt.zero_grad(set_to_none=False)
    emit('training_warmup')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    start=time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    warmup_s=time.perf_counter()-start;restore()
    emit('training_gpu_reference')
    expected=[]
    for i in range(3):
        stage(i);loss,output,cache=step()
        expected.append((loss.detach().clone(),output.detach().clone(),tuple(x.detach().clone() for x in cache)))
    expected_params=[p.detach().clone() for p in params]
    expected_grads=[p.grad.detach().clone() for p in params]
    expected_opt=[{k:v.detach().clone() for k,v in opt.state[p].items()} for p in params]
    del loss,output,cache
    restore();stage(0)
    emit('training_capture')
    graph=torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):static_loss,static_output,static_cache=step()
        restore();metrics=[]
        for i in range(3):
            stage(i);torch.cuda.synchronize();start=time.perf_counter()
            graph.replay();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            torch.testing.assert_close(static_loss,expected[i][0],rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(static_output,expected[i][1],rtol=2e-5,atol=2e-6)
            for x,y in zip(static_cache,expected[i][2]):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
            metrics.append(dict(loss=static_loss.item(),step_s=elapsed))
            emit('training_replay',step=i,**metrics[-1])
        errors={}
        for (name,p),x,g,state,start in zip(model.named_parameters(),expected_params,expected_grads,expected_opt,initial):
            torch.testing.assert_close(p,x,rtol=2e-4,atol=2e-6)
            torch.testing.assert_close(p.grad,g,rtol=2e-4,atol=2e-6)
            for key,value in state.items():torch.testing.assert_close(opt.state[p][key],value,rtol=2e-4,atol=2e-6)
            if not torch.isfinite(p.grad).all() or p.grad.abs().max()==0:raise RuntimeError('Invalid gradient: '+name)
            if (p-start).abs().max()==0:raise RuntimeError('Parameter not updated: '+name)
            errors[name]=dict(parameter_max_abs=(p-x).abs().max().item(),gradient_max_abs=(p.grad-g).abs().max().item(),
                              gradient_max=p.grad.abs().max().item(),update_max=(p-start).abs().max().item())
        return dict(parameters=sum(p.numel() for p in params),batch=2,length=4,window=128,heads=4,head_dim=64,
                    warmup_three_steps_s=warmup_s,metrics=metrics,equivalence=errors,loss_finite=True,grad_finite=True)
    finally:graph.reset()


def inference_probe():
    import torch
    from fly_attention import CausalAttention
    m=CausalAttention().cuda().eval()
    q=torch.randn(2,256,device='cuda');final=torch.randn_like(q)
    active=torch.ones(2,device='cuda',dtype=torch.bool);reset=torch.zeros_like(active)
    storage=m.empty(2)
    def step():
        cache=m.reset(storage,reset)
        out=m.read(q,cache,active)
        new=m.append(final,cache,active)
        # Inference only: no retained autograd graph. Persistent storage for next replay.
        for dest,src in zip(storage,new):dest.copy_(src)
        return out
    emit('inference_warmup')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad():
        with torch.cuda.stream(stream):
            for _ in range(3):step()
        torch.cuda.current_stream().wait_stream(stream)
        for x in storage:x.zero_()
        graph=torch.cuda.CUDAGraph()
        try:
            emit('inference_capture')
            with torch.cuda.graph(graph):out=step()
            for x in storage:x.zero_()
            expected=m.empty(2);max_error=0.;start=time.perf_counter()
            for t in range(134):
                q.normal_();final.normal_()
                active.fill_(True);active[1]=t%5!=1
                reset.fill_(False);reset[1]=t in (3,130)
                expected=m.reset(expected,reset)
                expected_out=m.read(q,expected,active)
                expected=m.append(final,expected,active)
                graph.replay()
                torch.testing.assert_close(out,expected_out,rtol=2e-5,atol=2e-6)
                for x,y in zip(storage,expected):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)
                max_error=max(max_error,(out-expected_out).abs().max().item())
                if t in (0,64,128,133):emit('inference_replay',step=t)
            torch.testing.assert_close(storage.positions[0],torch.arange(6,134,device='cuda'))
            return dict(replays=134,max_output_error=max_error,next_positions=storage.next_position.tolist(),
                        valid_counts=storage.valid.sum(-1).tolist(),elapsed_with_reference_s=time.perf_counter()-start)
        finally:graph.reset()


def worker():
    emit('load');start=time.perf_counter()
    import torch
    import unittest
    from test_attention import AttentionTests
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1);torch.manual_seed(23)
    load_s=time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    emit('gpu_unit_tests')
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AttentionTests))
    if not tests.wasSuccessful():raise RuntimeError('GPU attention tests failed')
    train=training_probe()
    inference=inference_probe()
    peak=torch.cuda.max_memory_allocated()/2**20
    emit('cleanup');gc.collect();torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    cleanup=dict(**memory(),gpu_alloc_mb=torch.cuda.memory_allocated()/2**20,gpu_reserved_mb=torch.cuda.memory_reserved()/2**20)
    emit('allocator_cleanup',**cleanup)
    if cleanup['gpu_alloc_mb']!=0:raise RuntimeError('GPU allocations remain after cleanup')
    return dict(ok=True,scenario='phase23_attention_only',variant='cuda_graph_math_sdpa_fp32',unit_tests=tests.testsRun,
                model_load_s=load_s,torch=torch.__version__,python=sys.executable,gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),training=train,inference=inference,
                peak_vram_mb=peak,post_cleanup_memory=cleanup,fallback_count=0,blocking_fallback_count=0,
                semantic_verdict='attention_module_verified_not_complete_lm',stability_verdict='stable',
                source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                               ['fly_attention.py','test_attention.py','verify_phase23.py','bench_runtime.py']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--output',default=str(ROOT/'results/phase23_attention.json'))
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=300)
    p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='verify_phase23')
    try:result=worker()
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error_type=type(exc).__name__,error=str(exc),fallback_count=0)
    gc.collect()
    if 'torch' in sys.modules:
        torch=sys.modules['torch']
        if torch.cuda.is_initialized():
            torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
            result['final_allocator_mb']=torch.cuda.memory_allocated()/2**20
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if not result['ok']:sys.exit(1)


if __name__=='__main__':main()

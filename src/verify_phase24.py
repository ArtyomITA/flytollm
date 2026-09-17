"""Guarded integrated LM training/generation, mandatory CUDA Graph, no CPU compute."""
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
from bench_runtime import supervise,emit,memory
ROOT=Path(__file__).resolve().parent


def probe(a):
    import torch
    from fly_lm import CUDATokenStep,state_tensors,select_token
    from lm_io import build_cns,story_batch,save_model,load_model
    from test_lm import toy
    start=time.perf_counter()
    model=toy() if a.toy else build_cns()
    batch=2 if a.toy else 1
    if a.toy:
        ids=torch.tensor([[1,1],[4,5],[6,7],[8,9]],device='cuda')
        cases=[(ids.clone(),ids+3) for _ in range(3)]
        cases[1][0][2,1]=0;cases[1][1][2,1]=0
        cases[2][0][2,1]=1
    else:
        from text_dataset import StoryDataset
        with StoryDataset(ROOT/'dataset/prepared_v1','train') as data:
            cases=[story_batch(data,[i],[0],4) for i in range(3)]
    ids=cases[0][0].clone();targets=cases[0][1].clone()
    params=list(model.parameters());initial=[p.detach().clone() for p in params]
    opt=torch.optim.Adam([{'params':list(model.core.parameters()),'lr':1e-4},
                          {'params':list(model.interfaces.parameters())+list(model.attention.parameters()),'lr':1e-4}],
                         capturable=True,foreach=False,weight_decay=0)
    load_s=time.perf_counter()-start
    def stage(i):ids.copy_(cases[i][0]);targets.copy_(cases[i][1])
    def step():
        opt.zero_grad(set_to_none=False)
        logits,state=model(ids)
        loss,count=model.loss(logits,targets,ids)
        loss.backward();opt.step()
        return loss,logits
    def restore():
        with torch.no_grad():
            for p,x in zip(params,initial):p.copy_(x)
            for state in opt.state.values():
                for value in state.values():value.zero_()
        opt.zero_grad(set_to_none=False)
    emit('training_warmup');stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    start=time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    warmup_s=time.perf_counter()-start;restore();expected=[]
    emit('training_reference')
    for i in range(3):
        stage(i);loss,logits=step();expected.append((loss.detach().clone(),logits.detach().clone()))
    expected_params=[p.detach().clone() for p in params]
    expected_grads=[p.grad.detach().clone() for p in params]
    expected_opt=[{k:v.detach().clone() for k,v in opt.state[p].items()} for p in params]
    del loss,logits
    restore();stage(0);graph=torch.cuda.CUDAGraph()
    try:
        emit('training_capture')
        with torch.cuda.graph(graph):loss,logits=step()
        restore();metrics=[]
        for i in range(3):
            stage(i);torch.cuda.synchronize();start=time.perf_counter();graph.replay();torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            torch.testing.assert_close(loss,expected[i][0],rtol=3e-5,atol=3e-6)
            torch.testing.assert_close(logits,expected[i][1],rtol=3e-4,atol=3e-6)
            metrics.append(dict(loss=loss.item(),step_s=elapsed));emit('training_replay',step=i,**metrics[-1])
        errors={}
        for (name,p),x,g,o,before in zip(model.named_parameters(),expected_params,expected_grads,expected_opt,initial):
            torch.testing.assert_close(p,x,rtol=3e-4,atol=3e-6)
            torch.testing.assert_close(p.grad,g,rtol=5e-4,atol=3e-6)
            for key,value in o.items():torch.testing.assert_close(opt.state[p][key],value,rtol=5e-4,atol=3e-6)
            if not torch.isfinite(p.grad).all() or p.grad.abs().max()==0:raise RuntimeError('Invalid gradient '+name)
            if (p-before).abs().max()==0:raise RuntimeError('Unchanged parameter '+name)
            errors[name]=dict(parameter_error=(p-x).abs().max().item(),gradient_error=(p.grad-g).abs().max().item(),
                              gradient_max=p.grad.abs().max().item(),update_max=(p-before).abs().max().item())
        emit('checkpoint')
        path=Path(a.output).with_suffix('.smoke.pt')
        metadata=dict(status='diagnostic_three_updates_not_pretrained',data='synthetic' if a.toy else 'train stories0,1,2; first4 input tokens',
                      tokenizer_sha256=None if a.toy else hashlib.sha256((ROOT/'dataset/prepared_v1/tokenizer-4096.json').read_bytes()).hexdigest())
        save_model(path,model,metadata)
        restored,_=load_model(path)
        with torch.no_grad():torch.testing.assert_close(model(ids)[0],restored(ids)[0],rtol=3e-4,atol=3e-6)
        del restored
    finally:graph.reset()
    # Finished backward/capture: remove training graph references before inference capture.
    del loss,logits,graph,step,restore,stage,opt,expected,expected_params,expected_grads,expected_opt,initial,params
    del p,x,g,o,before,value
    gc.collect()
    emit('generation_capture')
    inference=[]
    with torch.no_grad():
        for temperature in (0.,.8):
            decoder=CUDATokenStep(model,batch,temperature)
            try:
                state=model.initial_state(batch);max_error=0.
                for t in range(8):
                    token=cases[0][0][t%4].clone()
                    if t==5:token.fill_(0)
                    uniform=torch.full((batch,),.1+t*.1,device='cuda')
                    expected_logits,state=model.step(token,state)
                    expected_token=select_token(expected_logits,model.config,temperature,uniform)
                    expected_token=torch.where(token.ne(0),expected_token,torch.zeros_like(expected_token))
                    actual,selected=decoder.replay(token,uniform=uniform)
                    torch.testing.assert_close(actual,expected_logits,rtol=3e-4,atol=3e-6)
                    torch.testing.assert_close(selected,expected_token)
                    for x,y in zip(state_tensors(state),state_tensors(decoder.state)):torch.testing.assert_close(x,y,rtol=3e-4,atol=3e-6)
                    max_error=max(max_error,(actual-expected_logits).abs().max().item())
                generated=decoder.generate(cases[0][0][:2],6)
                inference.append(dict(temperature=temperature,replays_verified=8,max_logit_error=max_error,generated_ids=generated.tolist()))
            finally:decoder.close()
    result=dict(specification=model.specification(),model_load_s=load_s,warmup_three_steps_s=warmup_s,
                batch=batch,length=4,training=metrics,equivalence=errors,inference=inference,checkpoint=str(path),
                notes=metadata,loss_finite=True,grad_finite=True)
    return result


def worker(a):
    emit('load')
    import torch,unittest
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1);torch.manual_seed(73);torch.cuda.reset_peak_memory_stats()
    tests_count=0
    if a.toy:
        from test_lm import LMTests
        emit('gpu_tests');tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LMTests))
        if not tests.wasSuccessful():raise RuntimeError('LM tests failed')
        tests_count=tests.testsRun
    result=probe(a);peak=torch.cuda.max_memory_allocated()/2**20
    emit('cleanup');gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    cleanup=dict(**memory(),gpu_alloc_mb=torch.cuda.memory_allocated()/2**20,gpu_reserved_mb=torch.cuda.memory_reserved()/2**20)
    emit('allocator_cleanup',**cleanup)
    if cleanup['gpu_alloc_mb']!=0:raise RuntimeError('Live CUDA allocations after cleanup')
    result.update(ok=True,unit_tests=tests_count,scenario='phase24_toy' if a.toy else 'phase24_cns_t10',
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                  peak_vram_mb=peak,post_cleanup_memory=cleanup,fallback_count=0,stability_verdict='stable',
                  semantic_verdict='integrated_lm_smoke_not_quality_validation',
                  source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                 ['fly_lm.py','lm_io.py','test_lm.py','verify_phase24.py','fly_core.py','fly_interfaces.py','fly_attention.py']})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--toy',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--output',default=str(ROOT/'results/phase24_toy.json'))
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=300)
    p.add_argument('--max-vram-mb',type=int,default=7000);a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='verify_phase24')
    try:result=worker(a)
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error=str(exc),error_type=type(exc).__name__,fallback_count=0)
    gc.collect()
    if 'torch' in sys.modules:
        torch=sys.modules['torch']
        if torch.cuda.is_initialized():
            torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
            result['final_allocator_mb']=torch.cuda.memory_allocated()/2**20
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if not result['ok']:sys.exit(1)


if __name__=='__main__':main()

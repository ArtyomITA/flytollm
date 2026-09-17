"""Guarded full-LM verification, carry training, activation checkpoint and GPU profile."""
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
from bench_runtime import supervise,emit,memory
ROOT=Path(__file__).resolve().parent


def build(toy):
    from lm_io import build_cns
    if not toy:return build_cns()
    from test_interfaces import fixture
    from fly_attention import CausalAttention,AttentionConfig
    from fly_lm import FlyLM
    interfaces,core=fixture()
    return FlyLM(core,interfaces,CausalAttention(AttentionConfig(dim=16,heads=2,window=128)).cuda())


def correctness(model):
    import torch
    from fly_lm import CUDATokenStep,state_tensors,LMState
    from fly_attention import AttentionCache
    vocab=model.interfaces.config.vocab
    decoder=CUDATokenStep(model,2)
    positions=[0,0];histories=[[],[]];max_error=0.
    with torch.no_grad():
        expected=model.initial_state(2)
        try:
            emit('long_cache_134_replays')
            for t in range(134):
                ids=torch.tensor([3+t%(vocab-3),3+(t+9)%(vocab-3)],device='cuda')
                active=torch.tensor([True,t%7!=3],device='cuda')
                reset=torch.tensor([False,t in (43,45,130)],device='cuda')
                if t==0:ids.fill_(1)
                old=tuple(x.clone() for x in state_tensors(expected))
                logits,expected=model.step(ids,expected,active,reset)
                actual,_=decoder.replay(ids,active,reset)
                torch.testing.assert_close(actual,logits,rtol=5e-4,atol=1e-5)
                for x,y in zip(state_tensors(expected),state_tensors(decoder.state)):
                    torch.testing.assert_close(x,y,rtol=5e-4,atol=1e-5)
                for b in range(2):
                    clear=t==0 or (b==1 and t in (43,45,130))
                    enabled=b==0 or t%7!=3
                    if clear:positions[b]=0;histories[b]=[]
                    if enabled:
                        histories[b]=(histories[b]+[positions[b]])[-128:];positions[b]+=1
                    elif not clear:
                        for x,y in zip(old,state_tensors(expected)):torch.testing.assert_close(x[b],y[b])
                    valid=expected.cache.valid[b]
                    torch.testing.assert_close(expected.cache.positions[b][valid],torch.tensor(histories[b],device='cuda',dtype=torch.long))
                self_positions=expected.cache.next_position.tolist()
                if self_positions!=positions:raise RuntimeError('Incorrect independent slot positions')
                max_error=max(max_error,(actual-logits).abs().max().item())
                if t in (0,64,133):emit('long_cache_progress',step=t)
            assert len(histories[0])==128 and histories[0][0]==6
        finally:decoder.close()
        # Full state reset for one slot must match a fresh independent example.
        ids=torch.tensor([1,4],device='cuda')
        a,s=model.step(ids,expected)
        fresh,fresh_state=model.step(ids[:1])
        torch.testing.assert_close(a[:1],fresh,rtol=5e-4,atol=1e-5)
        for x,y in zip(state_tensors(s),state_tensors(fresh_state)):torch.testing.assert_close(x[:1],y,rtol=5e-4,atol=1e-5)
    emit('causality_gradients_reset')
    ids=torch.tensor([[1,1],[4,5],[6,7],[8,9],[10,11],[12,13]],device='cuda')
    output,state=model(ids);changed=ids.clone();changed[4:]+=1
    torch.testing.assert_close(output[:4],model(changed)[0][:4],rtol=5e-4,atol=1e-5)
    prefix=model.detach(model(ids[:2])[1])
    leaf=LMState(prefix.voltage.clone().requires_grad_(),prefix.spike.clone().requires_grad_(),
        AttentionCache(prefix.cache.k.clone().requires_grad_(),prefix.cache.v.clone().requires_grad_(),
                       prefix.cache.positions,prefix.cache.valid,prefix.cache.next_position))
    logits,new=model.step(torch.tensor([1,6],device='cuda'),leaf)
    gradients=torch.autograd.grad(logits.square().sum(),(leaf.voltage,leaf.spike,leaf.cache.k,leaf.cache.v))
    for grad in gradients:
        if grad[0].abs().max()!=0:raise RuntimeError('Gradient crosses story reset')
        if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite state gradient')
    if sum(g[1].abs().sum().item() for g in gradients)==0:raise RuntimeError('No gradient to unreset history')
    model.zero_grad(set_to_none=True)
    model.loss(output,ids+1,ids)[0].backward()
    groups={}
    for name,p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max()==0:raise RuntimeError('Invalid gradient '+name)
        groups[name]=p.grad.abs().max().item()
    return dict(replays=134,cache_window=128,max_logit_error=max_error,next_positions=positions,
                valid_counts=[len(x) for x in histories],gradient_max=groups,reset_cuts_gradients=True)


def checkpoint_check(model):
    import torch
    from lm_checkpointing import checkpoint_forward
    from fly_lm import state_tensors
    ids=torch.tensor([[1,1],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]],device='cuda')
    active=torch.ones_like(ids,dtype=torch.bool);active[3,1]=False
    reset=torch.zeros_like(active);reset[4,1]=True
    reference=None;results=[];observed_peak=0
    for segment in (0,2,4,8):
        emit('activation_checkpoint',segment=segment)
        # Warm every variant before timing; first checkpoint call loads extra machinery.
        model.zero_grad(set_to_none=True)
        if segment:warm_out,warm_state=checkpoint_forward(model,ids,active=active,reset=reset,segment=segment)
        else:warm_out,warm_state=model(ids,active=active,reset=reset)
        model.loss(warm_out,ids+1,ids,active)[0].backward();torch.cuda.synchronize()
        observed_peak=max(observed_peak,torch.cuda.max_memory_allocated())
        del warm_out,warm_state
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        base=torch.cuda.memory_allocated();start=time.perf_counter()
        if segment:out,state=checkpoint_forward(model,ids,active=active,reset=reset,segment=segment)
        else:out,state=model(ids,active=active,reset=reset)
        loss,_=model.loss(out,ids+1,ids,active);loss.backward();torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;peak=(torch.cuda.max_memory_allocated()-base)/2**20
        snap=(out.detach().clone(),tuple(x.detach().clone() for x in state_tensors(state)),
              [p.grad.detach().clone() for p in model.parameters()])
        observed_peak=max(observed_peak,torch.cuda.max_memory_allocated())
        gradient_relative_error=0.
        if reference is None:reference=snap
        else:
            torch.testing.assert_close(snap[0],reference[0],rtol=5e-4,atol=1e-5)
            for x,y in zip(snap[1],reference[1]):torch.testing.assert_close(x,y,rtol=5e-4,atol=1e-5)
            for x,y in zip(snap[2],reference[2]):
                torch.testing.assert_close(x,y,rtol=5e-4,atol=1e-8)
                relative=((x-y).norm()/y.norm().clamp_min(1e-12)).item()
                gradient_relative_error=max(gradient_relative_error,relative)
                if relative>0.003:raise RuntimeError('Checkpoint gradient relative error too large')
        results.append(dict(segment=segment,forward_backward_s=elapsed,incremental_peak_mb=peak,loss=loss.item(),max_gradient_relative_l2=gradient_relative_error))
        del out,state,loss,snap
    model.zero_grad(set_to_none=True)
    return dict(comparisons=results,adopted=False,observed_peak_mb=observed_peak/2**20,
                notes='Single warmed GPU eager correctness/timing probe; not CUDA Graph performance. Baseline unchanged.')


def carry_training(model, optimizer_factory=None):
    import torch
    from fly_lm import copy_state_,state_tensors
    from lm_io import story_batch
    from text_dataset import StoryDataset
    initial=[p.detach().clone() for p in model.parameters()]
    # Real TinyStories for CNS; synthetic matching toy vocabulary otherwise.
    if model.interfaces.config.vocab==4096:
        with StoryDataset(ROOT/'dataset/prepared_v1','train') as data:
            batches=[story_batch(data,[0,1],[i*4,i*4],4) for i in range(6)]
    else:
        batches=[]
        for i in range(6):
            x=(torch.arange(8,device='cuda').reshape(4,2)+i*4)%20+3
            if i==0:x[0].fill_(1)
            batches.append((x,x+1))
    ids,targets=[x.clone() for x in batches[0]]
    storage=model.initial_state(2)
    opt=(optimizer_factory(model) if optimizer_factory else
         torch.optim.Adam(model.parameters(),lr=1e-4,capturable=True,foreach=False,weight_decay=0))
    events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(5)]
    def step():
        opt.zero_grad(set_to_none=False)
        events[0].record()
        logits,new=model(ids,storage);loss,count=model.loss(logits,targets,ids)
        events[1].record();loss.backward();events[2].record();opt.step();events[3].record()
        copy_state_(storage,model.detach(new))
        events[4].record()
        return loss
    def restore():
        with torch.no_grad():
            for p,x in zip(model.parameters(),initial):p.copy_(x)
            for st in opt.state.values():
                for x in st.values():x.zero_()
        copy_state_(storage,model.initial_state(2));opt.zero_grad(set_to_none=False)
    def stage(i):ids.copy_(batches[i][0]);targets.copy_(batches[i][1])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    emit('carry_warmup')
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream)
    restore();expected=[]
    for i in range(6):
        stage(i);loss=step();expected.append((loss.detach().clone(),tuple(x.detach().clone() for x in state_tensors(storage))))
    params=[p.detach().clone() for p in model.parameters()]
    moments=[{k:v.detach().clone() for k,v in opt.state[p].items()} for p in model.parameters()]
    del loss
    restore();stage(0);graph=torch.cuda.CUDAGraph()
    try:
        emit('carry_capture')
        with torch.cuda.graph(graph):loss=step()
        restore();metrics=[]
        for i in range(6):
            stage(i);torch.cuda.synchronize();start=time.perf_counter();graph.replay();torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            torch.testing.assert_close(loss,expected[i][0],rtol=5e-4,atol=1e-5)
            for x,y in zip(state_tensors(storage),expected[i][1]):torch.testing.assert_close(x,y,rtol=5e-4,atol=1e-5)
            for name,p in model.named_parameters():
                if not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite carry gradient '+name)
            metrics.append(dict(step=i,loss=loss.item(),seconds=elapsed,positions=storage.cache.next_position.tolist(),
                                gpu_ms={name:events[j].elapsed_time(events[j+1]) for j,name in
                                        enumerate(['forward_loss','backward','adam','carry_copy'])}))
            emit('carry_replay',**metrics[-1])
        error=0.
        for p,x,st in zip(model.parameters(),params,moments):
            torch.testing.assert_close(p,x,rtol=5e-4,atol=1e-5)
            error=max(error,(p-x).abs().max().item())
            for key,value in st.items():torch.testing.assert_close(opt.state[p][key],value,rtol=5e-4,atol=1e-5)
        return dict(updates=6,batch=2,length=4,metrics=metrics,max_parameter_error=error,
                    state_policy='Detached carry after update; stale-state TBPTT approximation explicit')
    finally:
        graph.reset()
        with torch.no_grad():
            for p,x in zip(model.parameters(),initial):p.copy_(x)
        model.zero_grad(set_to_none=True)


def profile(model):
    import torch
    from fly_lm import LMState,copy_state_
    with torch.no_grad():
        state=model.initial_state(2)
        for t in range(6):_,state=model.step(torch.tensor([3+t,10+t],device='cuda'),state)
        ids=torch.tensor([20,21],device='cuda');w=model.core.weights()
        current=model.interfaces.input_current(ids)
        pre,pr=model.core.advance(current,4,(state.voltage,state.spike),weights=w)
        provisional=model.interfaces.representation(pre,pr)
        recalled=model.attention.read(provisional,state.cache)
        feedback=model.interfaces.feedback_current(recalled)
        post,po=model.core.advance(current+feedback,4,pre,weights=w)
        final=model.interfaces.representation(post,po)
        new=LMState(*post,model.attention.append(final,state.cache))
        scratch=model.initial_state(2)
        functions={
            'core_weights':lambda:model.core.weights(),
            'embedding_injection':lambda:model.interfaces.input_current(ids),
            'core_pre':lambda:model.core.advance(current,4,(state.voltage,state.spike),weights=w),
            'read_pre':lambda:model.interfaces.representation(pre,pr),
            'attention_read':lambda:model.attention.read(provisional,state.cache),
            'feedback_injection':lambda:model.interfaces.feedback_current(recalled),
            'core_post':lambda:model.core.advance(current+feedback,4,pre,weights=w),
            'read_head':lambda:torch.nn.functional.linear(model.interfaces.representation(post,po),model.interfaces.embedding.weight),
            'cache_append':lambda:model.attention.append(final,state.cache),
            'state_copy':lambda:copy_state_(scratch,new)}
        timings={}
        for name,fn in functions.items():
            emit('profile_module',module=name)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):fn()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):out=fn()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):graph.replay()
                end.record();end.synchronize();timings[name]=start.elapsed_time(end)/30
            finally:graph.reset()
            del out
        sensory=model.interfaces.input.nodes;read=model.interfaces.readout.nodes
        def stats(x):return dict(rms=x.square().mean().sqrt().item(),max_abs=x.abs().max().item())
        telemetry=dict(token_current=stats(current[:,sensory]),feedback_current=stats(feedback[:,sensory]),gate=model.interfaces.gate.item(),
                       pre_rate=pr.mean().item(),post_rate=po.mean().item(),
                       pre_silent_fraction=(pr==0).float().mean().item(),post_silent_fraction=(po==0).float().mean().item(),
                       pre_every_step_fraction=(pr==1).float().mean().item(),post_every_step_fraction=(po==1).float().mean().item(),
                       output_active_fraction=(po[:,read]>0).float().mean().item(),voltage=stats(post[0]))
        # Directed structural reachability; GPU propagation, no numeric CPU graph walk.
        reachable=torch.zeros(model.core.n,device='cuda',dtype=torch.bool);reachable[sensory]=True
        distances=[]
        for hop in range(1,33):
            counts=torch.zeros(model.core.n,device='cuda',dtype=torch.int32)
            counts.index_add_(0,model.core.dst,reachable[model.core.src].int())
            newer=reachable|(counts>0)
            distances.append(dict(hops=hop,reachable_nodes=newer.sum().item(),reachable_read_nodes=newer[read].sum().item()))
            converged=torch.equal(newer,reachable);reachable=newer
            if converged:break
        return dict(module_gpu_ms=timings,telemetry=telemetry,reachability=distances,reachability_converged=converged,
                    notes='Isolated captured forward modules, B2; not additive end-to-end training profile. Reachability ignores signs/threshold firing.')


def worker(a):
    import torch
    emit('load');torch.set_num_threads(1);torch.manual_seed(89)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    start=time.perf_counter();model=build(a.toy);load_s=time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    result=globals()[a.case](model)
    peak=max(torch.cuda.max_memory_allocated()/2**20,result.get('observed_peak_mb',0))
    spec=model.specification();del model
    emit('cleanup');gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    cleanup=dict(**memory(),gpu_alloc_mb=torch.cuda.memory_allocated()/2**20,gpu_reserved_mb=torch.cuda.memory_reserved()/2**20)
    emit('allocator_cleanup',**cleanup)
    if cleanup['gpu_alloc_mb']!=0:raise RuntimeError('Live GPU allocations')
    return dict(ok=True,case=a.case,toy=a.toy,result=result,specification=spec,model_load_s=load_s,peak_vram_mb=peak,
                post_cleanup_memory=cleanup,fallback_count=0,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                               ['verify_phase25.py','lm_checkpointing.py','fly_lm.py','fly_core.py','fly_attention.py','fly_interfaces.py']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['correctness','checkpoint_check','carry_training','profile'],required=True)
    p.add_argument('--toy',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--output',required=True)
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=300)
    p.add_argument('--max-vram-mb',type=int,default=7000);a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='verify_phase25')
    try:result=worker(a)
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error=str(exc),error_type=type(exc).__name__)
    gc.collect()
    if 'torch' in sys.modules:
        torch=sys.modules['torch']
        if torch.cuda.is_initialized():
            torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
            result['final_allocator_mb']=torch.cuda.memory_allocated()/2**20
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if not result['ok']:sys.exit(1)


if __name__=='__main__':main()

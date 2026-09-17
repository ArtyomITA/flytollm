"""Phase3 guarded CUDA Graph overfit; fixed protocol, no architecture changes."""
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
from bench_runtime import supervise,emit,memory
ROOT=Path(__file__).resolve().parent


def experiment(a,model):
    import torch
    from lm_io import story_batch,save_model,load_model
    from text_dataset import StoryDataset
    from fly_lm import CUDATokenStep,copy_state_,state_tensors
    from tokenizers import Tokenizer
    folder=ROOT/'dataset/prepared_v1'
    tokenizer=Tokenizer.from_file(str(folder/'tokenizer-4096.json'))
    with StoryDataset(folder,'train') as data:
        batches=[story_batch(data,[0,1],[i,i],8) for i in (0,8)]
        sample=[data[i][:17].astype('int64').tolist() for i in (0,1)]
    prompt=torch.tensor([x[:4] for x in sample],device='cuda').T.contiguous()

    def evaluate(generate=False):
        decoder=CUDATokenStep(model,2)
        losses=[];correct=0;count=0
        try:
            with torch.no_grad():
                for ids,targets in batches:
                    for x,y in zip(ids,targets):
                        logits,_=decoder.replay(x)
                        loss,n=model.loss(logits[None],y[None],x[None])
                        losses.append((loss*n).item());count+=n.item()
                        correct+=((logits.argmax(-1)==y)&y.ne(0)&x.ne(0)).sum().item()
                result=dict(ce=sum(losses)/count,accuracy=correct/count,targets=count)
                if generate:
                    tokens=decoder.generate(prompt,12).T.cpu().tolist()
                    result['generation']=[dict(prompt_ids=sample[b][:4],prompt=tokenizer.decode(sample[b][:4]),
                                               generated_ids=t,text=tokenizer.decode(t)) for b,t in enumerate(tokens)]
                return result
        finally:decoder.close()

    emit('initial_evaluation');initial_eval=evaluate(True)
    initial=[p.detach().clone() for p in model.parameters()]
    groups=[dict(name=prefix,params=[p for n,p in model.named_parameters() if n.startswith(prefix+'.')])
            for prefix in ('core','interfaces','attention')]
    opt=torch.optim.Adam(groups,lr=a.lr,betas=(.9,.999),weight_decay=0,foreach=False,capturable=True)
    ids,targets=[x.clone() for x in batches[0]]
    state=model.initial_state(2);empty=model.initial_state(2)
    observations={}
    # Hooks only observe existing port outputs/rates; equations and gradients unchanged.
    hooks=[model.interfaces.input.register_forward_hook(lambda m,args,out:observations.__setitem__('token',out)),
           model.interfaces.feedback.register_forward_hook(lambda m,args,out:observations.__setitem__('feedback',out)),
           model.interfaces.readout.register_forward_pre_hook(lambda m,args:observations.__setitem__('rate',args[1]))]
    def step():
        opt.zero_grad(set_to_none=False)
        logits,new=model(ids,state);loss,count=model.loss(logits,targets,ids)
        loss.backward()
        finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
        copy_state_(state,model.detach(new))
        with torch.no_grad():
            rate=observations['rate'];nodes=model.interfaces.input.nodes
            token=observations['token'][:,nodes];feedback=model.interfaces.gate*observations['feedback'][:,nodes]
            metrics=torch.stack([rate.mean(),(rate==0).float().mean(),(rate==1).float().mean(),
                                 state.voltage.square().mean().sqrt(),state.voltage.abs().max(),
                                 token.square().mean().sqrt(),feedback.square().mean().sqrt(),model.interfaces.gate.clone(),norm])
            finite=finite & torch.isfinite(metrics).all() & torch.isfinite(loss)
        opt.step();observations.clear()
        return loss,finite,metrics
    def restore():
        with torch.no_grad():
            for p,x in zip(model.parameters(),initial):p.copy_(x)
            for st in opt.state.values():
                for value in st.values():value.zero_()
        copy_state_(state,empty);opt.zero_grad(set_to_none=False)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    emit('warmup');start=time.perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    warmup_s=time.perf_counter()-start;restore()
    graph=torch.cuda.CUDAGraph()
    emit('capture')
    with torch.cuda.graph(graph):loss,finite,metrics=step()
    for hook in hooks:hook.remove()
    observations.clear();restore()

    def replay(x,y):
        # Host control decision from staged validity; never replay Adam on empty targets.
        if not bool((x.ne(0)&y.ne(0)).any()):return False
        ids.copy_(x);targets.copy_(y);graph.replay()
        if not bool(finite):raise RuntimeError('Nonfinite training metrics/gradients')
        return True
    def diagnostics():
        result={}
        for (name,p),old in zip(model.named_parameters(),initial):
            result[name]=dict(grad_rms=p.grad.square().mean().sqrt().item(),grad_max=p.grad.abs().max().item(),
                             delta_rms=(p-old).square().mean().sqrt().item(),finite=bool(torch.isfinite(p).all()))
        if not all(x['finite'] for x in result.values()):raise RuntimeError('Nonfinite weights')
        return result
    try:
        # Skip tested with nonzero Adam moments, against full parameter/state snapshots.
        emit('skip_reset_checks');replay(*batches[0]);torch.cuda.synchronize()
        before=[p.detach().clone() for p in model.parameters()]
        moments=[v.clone() for st in opt.state.values() for v in st.values()]
        carry=[x.clone() for x in state_tensors(state)]
        zeros=torch.zeros_like(ids);assert replay(zeros,zeros) is False
        for p,x in zip(model.parameters(),before):assert torch.equal(p,x)
        for v,x in zip((v for st in opt.state.values() for v in st.values()),moments):assert torch.equal(v,x)
        for v,x in zip(state_tensors(state),carry):assert torch.equal(v,x)
        del before,moments,carry,zeros;restore()
        rows=[];times=[];final=initial_eval;passed=False
        for epoch in range(1,a.epochs+1):
            copy_state_(state,empty)
            for window,batch in enumerate(batches):
                torch.cuda.synchronize();start=time.perf_counter();assert replay(*batch);torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                assert state.cache.next_position.tolist()==[8*(window+1)]*2
            if epoch%20==0 or epoch==a.epochs:
                telemetry=dict(zip(['post_rate','silent_fraction','every_step_fraction','voltage_rms','voltage_abs_max',
                                    'token_rms','feedback_rms','gate','unclipped_grad_norm'],metrics.tolist()))
                grads=diagnostics();final=evaluate()
                row=dict(epoch=epoch,updates=epoch*2,train_last_window_loss=loss.item(),evaluation=final,
                         telemetry=telemetry,parameters=grads)
                rows.append(row);emit('overfit_evaluation',epoch=epoch,**final,telemetry=telemetry)
                passed=final['ce']<=.5*initial_eval['ce'] and final['accuracy']>=.5
                if passed and not a.smoke:break
        final=evaluate(True)
        changed={prefix:any(x['delta_rms']>0 for n,x in rows[-1]['parameters'].items() if n.startswith(prefix+'.'))
                 for prefix in ('core','interfaces','attention')}
        passed=passed and all(changed.values())
        emit('checkpoint')
        checkpoint=Path(a.output).with_suffix('.diagnostic.pt')
        save_model(checkpoint,model,dict(protocol='PROTOCOLLO_FASE_3.md',lr=a.lr,epochs=epoch,
                                        tokenizer_sha256=hashlib.sha256((folder/'tokenizer-4096.json').read_bytes()).hexdigest()))
        loaded,metadata=load_model(checkpoint)
        for x,y in zip(model.state_dict().values(),loaded.state_dict().values()):assert torch.equal(x,y)
        del loaded
        return dict(initial=initial_eval,final=final,overfit_passed=passed,smoke=a.smoke,epochs=epoch,updates=epoch*2,
                    distinct_targets=32,trained_targets=epoch*32,lr=a.lr,rows=rows,changed_groups=changed,
                    skip_empty_verified=True,reset_carry_verified=True,checkpoint_roundtrip=True,
                    checkpoint=str(checkpoint),sample_token_ids=sample,warmup_s=warmup_s,
                    first_step_s=times[0],mean_step_s=sum(times)/len(times),training_step_tokens_per_sec=16/(sum(times)/len(times)))
    finally:
        graph.reset()
        for hook in hooks:hook.remove()
        observations.clear();model.zero_grad(set_to_none=True)


def worker(a):
    import torch
    from lm_io import build_cns
    torch.set_num_threads(1);torch.manual_seed(89)
    emit('load',torch=torch.__version__,gpu=torch.cuda.get_device_name(),memory=memory())
    start=time.perf_counter();model=build_cns();load_s=time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats();result=experiment(a,model)
    peak=torch.cuda.max_memory_allocated()/2**20;spec=model.specification();del model
    emit('cleanup');gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    cleanup=dict(**memory(),allocated_mb=torch.cuda.memory_allocated()/2**20,reserved_mb=torch.cuda.memory_reserved()/2**20)
    assert cleanup['allocated_mb']==0,cleanup
    return dict(ok=True,result=result,specification=spec,model_load_s=load_s,peak_vram_mb=peak,
                post_cleanup_memory=cleanup,fallback_count=0,semantic_verdict='diagnostic overfit only',
                source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                               ['train_diagnostic.py','PROTOCOLLO_FASE_3.md','fly_lm.py','fly_core.py','fly_interfaces.py','fly_attention.py']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--smoke',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--phase-timeout',type=float,default=300);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not 1<=a.epochs<=200 or a.lr not in (1e-4,1e-3):p.error('protocol bounds')
    if a.smoke and a.epochs!=2:p.error('smoke requires epochs2')
    if not a.worker:return supervise(a,worker_module='train_diagnostic')
    try:result=worker(a)
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error=str(exc),error_type=type(exc).__name__)
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    if not result['ok']:sys.exit(1)


if __name__=='__main__':main()

"""Versioned recovery experiments; unchanged model and captured compute."""
import argparse, gc, hashlib, json, random, sys, time, traceback
from pathlib import Path
from bench_runtime import emit, memory, supervise


def language_diagnostics(model, train, dev):
    import torch
    from fly_lm import CUDATokenStep
    from text_dataset import StoryDataset
    from tokenizers import Tokenizer
    counts=torch.ones(4096,device='cuda')
    for pair in train:
        for x,y in pair:
            valid=(x!=0)&(y!=0)
            counts+=torch.bincount(y[valid],minlength=4096)
    logp=(counts/counts.sum()).log()
    raw=counts-1
    decoder=CUDATokenStep(model,2)
    try:
        histogram=torch.zeros(4096,device='cuda',dtype=torch.long)
        entropy=ce=base=0.;total=correct=0
        with torch.no_grad():
            for pair in dev:
                from fly_lm import copy_state_
                copy_state_(decoder.state,model.initial_state(2))
                for x,y in pair:
                    for token,target in zip(x,y):
                        logits,_=decoder.replay(token)
                        valid=(token!=0)&(target!=0)
                        p=logits.softmax(-1); n=int(valid.sum())
                        if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite diagnostics')
                        entropy+=float((-(p*p.clamp_min(1e-30).log()).sum(-1)*valid).sum())
                        ce+=float((torch.nn.functional.cross_entropy(logits,target,reduction='none')*valid).sum())
                        base+=float((-logp[target]*valid).sum());total+=n
                        pred=logits.argmax(-1);correct+=int(((pred==target)&valid).sum())
                        histogram+=torch.bincount(pred[valid],minlength=4096)
            tokenizer=Tokenizer.from_file('dataset/prepared_v1/tokenizer-4096.json')
            indices=[14447,9560]
            with StoryDataset(Path('dataset/prepared_v1'),'validation') as data:
                prompts=[data[i][:8].astype('int64').tolist() for i in indices]
            ids=torch.tensor(prompts,device='cuda').T.contiguous()
            generated=decoder.generate(ids,24).T.cpu().tolist()
            top=histogram.topk(5)
            return dict(ce=ce/total,accuracy=correct/total,targets=total,entropy=entropy/total,
                        unigram_ce=base/total,train_point_fraction=float(raw[16]/raw.sum()),
                        predicted_point_fraction=float(histogram[16])/total,
                        prediction_top=[dict(id=i,text=tokenizer.decode([i]),count=c) for i,c in zip(top.indices.tolist(),top.values.tolist())],
                        prompts=[tokenizer.decode(p) for p in prompts],generated_ids=generated,
                        generated_text=[tokenizer.decode(g) for g in generated],
                        constant_generation=any(len(set(g))==1 for g in generated))
    finally:decoder.close()


def reblock(pairs,length):
    import torch
    result=[]
    for pair in pairs:
        x=torch.cat([v[0] for v in pair]);y=torch.cat([v[1] for v in pair])
        pad=(-len(x))%length
        if pad:
            x=torch.cat([x,x.new_zeros((pad,2))]);y=torch.cat([y,y.new_zeros((pad,2))])
        result.append([(x[i:i+length],y[i:i+length]) for i in range(0,len(x),length)])
    return result


def verify_capture(model,engine,pair):
    import torch
    params=list(model.parameters());saved=[p.detach().clone() for p in params]
    x,y=pair[0];opt=engine.opt
    def restore():
        with torch.no_grad():
            for p,v in zip(params,saved):p.copy_(v)
            for state in opt.state.values():
                for value in state.values():value.zero_()
        opt.zero_grad(set_to_none=False);engine.reset()
    restore();logits,state=model(x);loss,count=model.loss(logits,y,x);loss.backward()
    torch.nn.utils.clip_grad_norm_(params,1.,foreach=False)
    grads=[p.grad.detach().clone() for p in params];expected_loss=float(loss)
    opt.step();expected=[p.detach().clone() for p in params]
    del logits,state,loss
    restore();metric=engine.update(x,y)
    assert abs(metric['loss']-expected_loss)<1e-4
    max_weight=max(float((p-v).abs().max()) for p,v in zip(params,expected))
    max_grad=max(float((p.grad-v).abs().max()) for p,v in zip(params,grads))
    for p,v,g in zip(params,expected,grads):
        torch.testing.assert_close(p,v,rtol=1e-3,atol=1e-5)
        torch.testing.assert_close(p.grad,g,rtol=1e-3,atol=1e-5)
    restore()
    return dict(loss_abs_error=abs(metric['loss']-expected_loss),weight_max_abs=max_weight,grad_max_abs=max_grad)


def execute(a):
    import torch
    from lm_io import build_cns,save_model
    from phase3_extended import initialize,Captured,optimizer,synthetic_pairs,text_pairs,aggregate_stats_eval
    torch.set_num_threads(1);emit('load',memory=memory(),gpu=torch.cuda.get_device_name())
    model=build_cns(a.threshold);initialize(model,a.seed)
    specification=model.specification()
    artificial_sha=hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for n,p in model.named_parameters() if not n.startswith("core."))).hexdigest()
    if a.case=='memory':
        train,tr=synthetic_pairs(171,a.delay);dev,dr=synthetic_pairs(172,a.delay)
        assert not set(map(tuple,tr))&set(map(tuple,dr))
        epochs=a.epochs
    else:
        indices=list(range(18,82));random.Random(17).shuffle(indices)
        train=text_pairs(indices,128)
        reserved=json.loads(Path('configs/phase3_protocol_v1.json').read_text())['validation_reserved']
        dev=text_pairs(reserved,128,'validation');epochs=a.epochs
    train=reblock(train,a.length);dev=reblock(dev,a.length)
    if a.length==16:
        from phase3_capture16 import Captured16
        Captured=Captured16
    before=language_diagnostics(model,train,dev) if a.case=='language' else None
    engine=Captured(model,optimizer(model,a))
    emit('verify_capture');equivalence=verify_capture(model,engine,train[0])
    updates=seen=0;elapsed=0.;rows=[];evaluations=[];slow=0
    try:
        emit('initial_validation')
        initial=engine.evaluate(dev)
        initial_train=engine.evaluate(train) if a.case=='memory' else None
        emit('training_start')
        for epoch in range(epochs):
            for pair in train:
                engine.reset()
                for x,y in pair:
                    if elapsed>=400:break
                    torch.cuda.synchronize();start=time.perf_counter()
                    metric=engine.update(x,y);torch.cuda.synchronize();dt=time.perf_counter()-start
                    if metric is None:continue
                    elapsed+=dt;updates+=1;seen+=metric['count']
                    slow=slow+1 if dt>2 else 0
                    if slow>=5:raise RuntimeError('Five consecutive steps over2s')
                    if updates%32==0 and updates%64!=0:emit('progress',updates=updates,train_s=elapsed)
                    if updates%64==0:
                        row=dict(updates=updates,train_s=elapsed,**metric);rows.append(row);emit('train_progress',**row)
                if elapsed>=400:break
            if a.case=='memory' and (epoch+1)%10==0:
                ev=dict(epoch=epoch+1,train=engine.evaluate(train),dev=engine.evaluate(dev))
                evaluations.append(ev);emit('memory_validation',epoch=epoch+1,train_accuracy=ev['train']['restricted_accuracy'],dev_accuracy=ev['dev']['restricted_accuracy'])
                if ev['dev']['restricted_accuracy']>=.8:break
            if elapsed>=400:break
        emit('final_validation');final=engine.evaluate(dev)
        final_train=engine.evaluate(train) if a.case=='memory' else None
        finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
        assert finite
        drift=False
        if len(rows)>=6:
            import statistics
            ref=statistics.median(r['stats'][0] for r in rows[:3]);last=[r['stats'][0] for r in rows[-3:]]
            drift=last[0]<last[1]<last[2] and min(last)>2*ref
        passed=final['restricted_accuracy']>=.8 if a.case=='memory' else final['ce']<=.95*initial['ce'] and not drift
        checkpoint=str(Path(a.output).with_suffix('.diagnostic.pt'))
        save_model(checkpoint,model,dict(protocol='PROTOCOLLO_FASE_3_D.md',case=a.case,delay=a.delay,updates=updates,lr=a.lr))
        result=dict(initial=initial,final=final,initial_train=initial_train,final_train=final_train,evaluations=evaluations,
                    passed=passed,drift_flag=drift,updates=updates,seen_targets=seen,train_s=elapsed,rows=rows,checkpoint=checkpoint)
        if a.case=='language':
            result['heldout12_initial']=aggregate_stats_eval(initial,dev,4)
            result['heldout12_final']=aggregate_stats_eval(final,dev,4)
    finally:engine.close()
    del engine
    gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
    if a.case=='language':
        emit('generation_diagnostics');result['before_diagnostics']=before;result['after_diagnostics']=language_diagnostics(model,train,dev)
    peak=torch.cuda.max_memory_allocated()/2**20
    del model,train,dev,pair,x,y
    gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    alloc=torch.cuda.memory_allocated()/2**20
    assert alloc==0,alloc
    emit('cleanup',allocated_mb=alloc)
    return dict(ok=True,result=result,equivalence=equivalence,specification=specification,artificial_parameter_sha256=artificial_sha,config=vars(a),peak_allocated_mb=peak,final_allocator_mb=alloc,
                source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['phase3_compare_v2.py','phase3_capture16.py','phase3_extended.py','PROTOCOLLO_FASE_3_D.md','fly_lm.py','fly_core.py','fly_interfaces.py','bench_runtime.py']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['memory','language'],required=True)
    p.add_argument('--threshold',type=int,choices=[5,10],default=10);p.add_argument('--length',type=int,choices=[8,16],default=16);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--delay',type=int,default=1);p.add_argument('--seed',type=int,default=89)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--variant',default='adam')
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--allow-paging',action='store_true');p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_compare_v2')
    try:r=execute(a)
    except Exception as e:traceback.print_exc();r=dict(ok=False,error=str(e),error_type=type(e).__name__)
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    if not r['ok']:sys.exit(1)


if __name__=='__main__':main()

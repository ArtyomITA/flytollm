"""T6: 100k unique training targets, frozen DEV protocol, CUDA Graph LM."""
import argparse,gc,json,sys,time,traceback,math
from pathlib import Path
from collections import Counter
from bench_runtime import emit,supervise,memory
from phase3_t23 import ROOT,sha

def run(a):
    import torch,numpy as np
    from lm_io import load_model,story_batch,save_model,build_cns
    from text_dataset import StoryDataset
    from phase3_extended import initialize
    from phase3_t45 import FairCapture
    from fly_lm import state_tensors,CUDATokenStep
    if a.threshold==10:model,_=load_model(ROOT/'results/phase3_t2_initial.pt')
    else:model=build_cns(a.threshold)
    initialize(model,a.seed)
    import hashlib
    artificial_hash=hashlib.sha256()
    for name,p in model.named_parameters():
        if not name.startswith('core.'):artificial_hash.update(name.encode());artificial_hash.update(p.detach().cpu().numpy().tobytes())
    dev_ids=json.loads((ROOT/'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
    with StoryDataset(ROOT/'dataset/prepared_v1','validation') as ds:
        dev=[[story_batch(ds,dev_ids[b:b+2],[t,t],16) for t in range(0,128,16)] for b in range(0,16,2)]
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds:
        first=story_batch(ds,[130,131],[0,0],16)
    engine=FairCapture(model,8,*first);equivalence=engine.equivalence
    total=0;updates=0;times=[];slow=0;curve=[];telemetry=[];unigram=Counter();bigram=Counter();contexts=Counter();used=[];thresholds=sorted(set([v for v in (10000,30000,a.budget) if v<=a.budget]))
    initial=engine.evaluate(dev)
    def baseline():
        n=sum(unigram.values());uni=bi=0.;count=0
        for pair in dev:
            for x,y in pair:
                xx=x.cpu().flatten().tolist();yy=y.cpu().flatten().tolist()
                for u,v in zip(xx,yy):
                    if u==0 or v==0:continue
                    p=(unigram[v]+1)/(n+4096)
                    uni-=math.log(p);bi-=math.log((bigram[u,v]+10*p)/(contexts[u]+10));count+=1
        return dict(unigram_ce=uni/count,bigram_ce=bi/count,training_targets=n)
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds:
        index=130
        while total<a.budget:
            engine.reset();used.extend([index,index+1])
            for start in range(0,128,16):
                x,y=story_batch(ds,[index,index+1],[start,start],16)
                valid=(x!=0)&(y!=0);n=int(valid.sum())
                if not n:continue
                if total+n>a.budget:
                    rank=valid.flatten().cumsum(0).reshape_as(valid);y=torch.where(rank<=a.budget-total,y,0);n=a.budget-total
                torch.cuda.synchronize();t=time.perf_counter();m=engine.update(x,y);torch.cuda.synchronize();dt=time.perf_counter()-t
                total+=m['count'];updates+=1;times.append(dt);slow=slow+1 if dt>2 else 0
                if slow>=5:raise RuntimeError('Five consecutive updates >2s')
                for u,v in zip(x.cpu().flatten().tolist(),y.cpu().flatten().tolist()):
                    if u and v:unigram[v]+=1;bigram[u,v]+=1;contexts[u]+=1
                if updates%32==0:
                    sample=engine.state.voltage[:,::21].detach().cpu().numpy()
                    telemetry.append(dict(update=updates,targets=total,voltage_sample_quantiles=np.quantile(sample,[0,.01,.5,.99,1]).tolist(),grad_norm=m['grad_norm']))
                    emit('training',targets=total,update=updates,**m)
                if thresholds and total>=thresholds[0]:
                    threshold=thresholds.pop(0);saved=[v.clone() for v in state_tensors(engine.state)]
                    dv=engine.evaluate(dev)
                    with torch.no_grad():
                        for v,s in zip(state_tensors(engine.state),saved):v.copy_(s)
                    del saved,v,s
                    curve.append(dict(requested_targets=threshold,actual_targets=total,dev=dv,baseline=baseline()))
                    emit('pilot_curve',**curve[-1])
                if total==a.budget:break
            index+=2
    assert total==a.budget and sum(unigram.values())==total
    engine.close();del engine
    path=Path(a.output).with_suffix('.diagnostic.pt');save_model(path,model,dict(experiment='T6',seed=a.seed,targets=total))
    generation=[];step=CUDATokenStep(model,2)
    from fly_lm import copy_state_
    context_scores=[]
    for pair_index,pair in enumerate(dev):
        copy_state_(step.state,model.initial_state(2));totals=np.zeros((2,2,4));dominant=[Counter(),Counter()]
        for x,y in pair:
            for token,target in zip(x,y):
                before_state=[v.clone() for v in state_tensors(step.state)]
                real,_=step.replay(token);real=real.clone()
                after_state=[v.clone() for v in state_tensors(step.state)]
                with torch.no_grad():
                    for v,s in zip(state_tensors(step.state),before_state):v.copy_(s)
                isolated,_=step.replay(token,reset=torch.ones(2,device='cuda',dtype=torch.bool))
                for k,scores in enumerate((real,isolated)):
                    lp=scores.log_softmax(-1);valid=(token!=0)&(target!=0)
                    ce=-lp.gather(1,target[:,None]).flatten();entropy=-(lp.exp()*lp).sum(-1)
                    pred=scores.argmax(-1)
                    totals[k]+=torch.stack([ce*valid,valid,(pred==target)&valid,entropy*valid],-1).cpu().numpy()
                    for v in pred[valid].cpu().tolist():dominant[k][v]+=1
                with torch.no_grad():
                    for v,s in zip(state_tensors(step.state),after_state):v.copy_(s)
        context_scores.append(dict(stories=dev_ids[2*pair_index:2*pair_index+2],raw=totals.tolist(),dominant=[c.most_common(1) for c in dominant]))
        emit('context_evaluation',pair=pair_index+1)
    step.close();del step
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(ROOT/'dataset/prepared_v1/tokenizer-4096.json'))
    for temperature in (0.,.8):
        torch.manual_seed(93018);step=CUDATokenStep(model,2,temperature)
        for prompt in ['Once upon a time','The little girl','Tom wanted to']:
            tokens=[1]+tokenizer.encode(prompt).ids
            prompt_ids=torch.tensor([tokens,tokens],device='cuda').T.contiguous()
            generated=step.generate(prompt_ids,32)[:,0].cpu().tolist()
            if 2 in generated:generated=generated[:generated.index(2)+1]
            generation.append(dict(prompt=prompt,temperature=temperature,seed=93018,top_p=1.,ids=generated,text=tokenizer.decode(generated)))
        step.close()
    return dict(seed=a.seed,threshold=a.threshold,specification=model.specification(),artificial_initial_sha256=artificial_hash.hexdigest(),initial=initial,curve=curve,context_scores=context_scores,telemetry=telemetry,generation=generation,targets=total,updates=updates,train_s=sum(times),train_story_indices=used,
                checkpoint_sha256=sha(path),equivalence=equivalence,audit_evaluated=False,test_evaluated=False,
                promising=curve[-1]['dev']['ce']<curve[-1]['baseline']['unigram_ce'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--threshold',type=int,choices=[10,5],default=10);p.add_argument('--budget',type=int,default=100000);p.add_argument('--seed',type=int,default=17);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=4000);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_text_pilot')
    try:
        import torch
        emit('load',memory=memory());result=run(a);peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20;result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak,source_sha256=sha(__file__))
    except Exception as e:traceback.print_exc();result=dict(ok=False,error=str(e))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()


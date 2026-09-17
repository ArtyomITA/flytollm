"""Guarded CUDA Graph training, full stories, resumable at any update boundary."""
import argparse, gc, hashlib, json, math, os, random, shutil, sys, time, traceback
from collections import Counter
from pathlib import Path
from bench_runtime import ROOT, emit, memory, supervise

SCHEMA = 1
SOURCES = ['pretrain_resumable.py', 'phase3_t45.py', 'phase3_variants.py',
           'fly_core.py', 'fly_interfaces.py', 'fly_attention.py', 'fly_lm.py', 'lm_io.py',
           'text_dataset.py', 'phase3_extended.py']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprints():
    return {p: digest(ROOT / p) for p in SOURCES + [
        'dataset/prepared_v1/manifest.json', 'dataset/prepared_v1/tokenizer-4096.json',
        'configs/phase3_protocol_v1.json']}

def cpu_tree(value):
    import torch
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value

def model_payload(model):
    return dict(schema=1, specification=model.specification(), chunk=model.core.chunk,
                state_dict=cpu_tree(model.state_dict()))

def load_payload(data, head):
    import torch
    from fly_core import Core
    from fly_interfaces import TextInterfaces, InterfaceConfig
    from fly_attention import CausalAttention, AttentionConfig
    from fly_lm import FlyLM, LMConfig
    from phase3_variants import variant
    state=data['state_dict']; spec=data['specification']; n=spec['neurons']
    core=Core(state['core.src'], state['core.dst'], torch.ones_like(state['core.raw']),
              torch.ones(n), n, data['chunk'])
    interfaces=TextInterfaces(n, state['interfaces.input.nodes'], state['interfaces.readout.nodes'],
                             state['interfaces.readout.groups'], InterfaceConfig(**spec['interfaces']))
    model=FlyLM(core, interfaces, CausalAttention(AttentionConfig(**spec['attention'])), LMConfig(**spec['lm'])).cuda()
    if head == 'separate': variant(model, 'h1')
    model.load_state_dict(state, strict=True)
    assert model.specification() == spec
    return model

def restore_optimizer(engine, saved):
    """Copy into captured tensors; load_state_dict would replace their addresses."""
    import torch
    live=engine.opt.state_dict()
    assert len(live['param_groups']) == len(saved['param_groups'])
    with torch.no_grad():
        for current, source in zip(live['param_groups'], saved['param_groups']):
            assert current == source, 'Optimizer configuration mismatch'
            for idx, parameter in zip(current['params'], engine.opt.param_groups[0]['params']):
                target=engine.opt.state[parameter]; origin=saved['state'][idx]
                assert target.keys() == origin.keys()
                for key, tensor in target.items():
                    assert tensor.shape == origin[key].shape and tensor.dtype == origin[key].dtype
                    tensor.copy_(origin[key])
                    assert torch.equal(tensor.cpu(), origin[key]), f'Optimizer restore: {key}'

def restore_runtime(engine, saved):
    import torch
    from fly_lm import state_tensors
    restore_optimizer(engine, saved['optimizer'])
    with torch.no_grad():
        for target, source in zip(state_tensors(engine.state), saved['recurrent']):
            target.copy_(source)
            assert torch.equal(target.cpu(), source)
    torch.set_rng_state(saved['rng_cpu'])
    torch.cuda.set_rng_state_all(saved['rng_cuda'])
    random.setstate(saved['rng_python'])

def checkpoint(engine, progress, config, path):
    import torch
    from fly_lm import state_tensors
    emit('checkpoint', update=progress['updates'], path=str(path))
    data=dict(schema=SCHEMA, config=config, fingerprints=fingerprints(), model=model_payload(engine.model),
              optimizer=cpu_tree(engine.opt.state_dict()), recurrent=cpu_tree(state_tensors(engine.state)),
              progress=progress, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
              rng_python=random.getstate(), numerical_note='Full state restored; CUDA atomic reductions are not bitwise deterministic.')
    temporary=path.with_suffix('.tmp')
    torch.save(data, temporary)
    with temporary.open('r+b') as stream: stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    emit('checkpoint_saved', update=progress['updates'], bytes=path.stat().st_size)

def evaluate_preserving(engine, dev):
    import torch
    from fly_lm import state_tensors
    saved=[x.clone() for x in state_tensors(engine.state)]
    try: return engine.evaluate(dev)
    finally:
        with torch.no_grad():
            for target, source in zip(state_tensors(engine.state), saved): target.copy_(source)

def baselines(dev, counts):
    uni=Counter(dict(counts['unigram'])); bi=Counter({tuple(k):v for k,v in counts['bigram']})
    contexts=Counter(dict(counts['contexts'])); n=sum(uni.values()); a=b=0.; total=0
    for pair in dev:
        for x,y in pair:
            for u,v in zip(x.cpu().flatten().tolist(), y.cpu().flatten().tolist()):
                if not u or not v: continue
                p=(uni[v]+1)/(n+4096)
                a-=math.log(p); b-=math.log((bi[u,v]+10*p)/(contexts[u]+10)); total+=1
    return dict(unigram_ce=a/total, bigram_ce=b/total, train_targets=n)

def samples(model):
    import torch
    from fly_lm import CUDATokenStep
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(ROOT/'dataset/prepared_v1/tokenizer-4096.json'))
    entries=[]
    with torch.random.fork_rng(devices=[0]):
        for temperature in (0., .8):
            torch.manual_seed(93018); step=CUDATokenStep(model,2,temperature)
            try:
                for prompt in ('Once upon a time', 'The little girl', 'Tom wanted to'):
                    token_ids=[1]+tokenizer.encode(prompt).ids
                    x=torch.tensor([token_ids,token_ids],device='cuda').T.contiguous()
                    generated=step.generate(x,32)[:,0].cpu().tolist()
                    if 2 in generated: generated=generated[:generated.index(2)+1]
                    entries.append(dict(prompt=prompt,temperature=temperature,seed=93018,ids=generated,text=tokenizer.decode(generated)))
            finally: step.close()
    return entries

def next_pair(ds, cursor, engine):
    """Both batch lanes reset together; never cross a story with a target."""
    index=cursor['story']; offset=cursor['offset']
    if index+1 >= len(ds):
        cursor.update(story=130, offset=0, epoch=cursor.get('epoch',0)+1)
        index=130; offset=0
    length=max(len(ds[index]),len(ds[index+1]))-1
    if offset >= length:
        cursor.update(story=index+2, offset=0)
        return next_pair(ds,cursor,engine)
    if offset == 0: engine.reset()
    from lm_io import story_batch
    x,y=story_batch(ds,[index,index+1],[offset,offset],16)
    cursor['offset']+=16
    return x,y

def run(a):
    import torch, numpy as np
    from lm_io import build_cns, story_batch
    from phase3_extended import initialize
    from phase3_variants import variant
    from phase3_t45 import FairCapture
    from text_dataset import StoryDataset
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.75)
    torch.manual_seed(a.seed); random.seed(a.seed)
    config=dict(threshold=a.threshold,head=a.head,seed=a.seed,batch=2,positions=16,tbptt=8,
                lr=.0001,optimizer='Adam',clip=1.,story_policy='full stories, sequential pairs from130, reset each pair, repeat epochs',
                eval_prefix=128,baseline_frozen_after_update=2000,backend='CUDA Graph FP32',architecture_variants=['h1'] if a.head=='separate' else [])
    saved=None
    if a.resume:
        saved=torch.load(a.resume, map_location='cpu', weights_only=True)
        assert saved['schema']==SCHEMA and saved['config']==config, 'Resume configuration mismatch'
        assert saved['fingerprints']==fingerprints(), 'Code/data changed since checkpoint'
        model=load_payload(saved['model'],a.head); progress=saved['progress']
    else:
        model=build_cns(a.threshold); initialize(model,a.seed)
        if a.head=='separate': variant(model,'h1')
        initial_hash=hashlib.sha256()
        for name,p in model.named_parameters():
            if not name.startswith('core.') and name!='head':
                initial_hash.update(name.encode()); initial_hash.update(p.detach().cpu().numpy().tobytes())
        progress=dict(updates=0,targets=0,cursor=dict(story=130,offset=0,epoch=0),curve=[],train_seconds=0.,
                      counts=dict(unigram=[],bigram=[],contexts=[]),artificial_initial_sha256=initial_hash.hexdigest(),
                      best_dev_ce=None,initial_dev=None,telemetry=[],resume_checks=[])
    dev_ids=json.loads((ROOT/'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
    with StoryDataset(ROOT/'dataset/prepared_v1','validation') as ds:
        dev=[[story_batch(ds,dev_ids[b:b+2],[t,t],16) for t in range(0,128,16)] for b in range(0,16,2)]
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds: first=story_batch(ds,[130,131],[0,0],16)
    engine=FairCapture(model,8,*first); equivalence=engine.equivalence
    if saved:
        restore_runtime(engine,saved)
        for key,value in model.state_dict().items(): assert torch.equal(value.cpu(),saved['model']['state_dict'][key]),key
        progress['resume_checks'].append(dict(update=progress['updates'],model_exact=True,optimizer_exact=True,recurrent_exact=True))
        del saved
    for state in engine.opt.state.values(): assert float(state['step'])==progress['updates'], 'Adam step count mismatch'
    if progress['initial_dev'] is None: progress['initial_dev']=evaluate_preserving(engine,dev)
    counts=progress['counts']; uni=Counter(dict(counts['unigram'])); bi=Counter({tuple(k):v for k,v in counts['bigram']}); ctx=Counter(dict(counts['contexts']))
    def sync_counts(): progress['counts']=dict(unigram=list(uni.items()),bigram=[[list(k),v] for k,v in bi.items()],contexts=list(ctx.items()))
    prefix=Path(a.output).with_suffix(''); latest=prefix.with_suffix('.latest.pt')
    slow=0; measured=[]; stopped='update_budget'
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds:
        while a.updates==0 or progress['updates'] < a.updates:
            if (ROOT/'STOP_PRETRAINING').exists(): stopped='user_stop_file'; break
            if progress['updates']%128==0 and shutil.disk_usage(ROOT).free<2*2**30:
                stopped='disk_headroom_below_2GiB'; break
            x,y=next_pair(ds,progress['cursor'],engine)
            torch.cuda.synchronize(); started=time.perf_counter(); m=engine.update(x,y); torch.cuda.synchronize(); dt=time.perf_counter()-started
            assert m['count']>0
            progress['updates']+=1; progress['targets']+=m['count']; progress['train_seconds']+=dt; measured.append(dt)
            update=progress['updates']
            slow=slow+1 if dt>2 else 0
            if slow>=5: raise RuntimeError('Five consecutive updates >2s')
            if update <= 2000:
                for u,v in zip(x.cpu().flatten().tolist(),y.cpu().flatten().tolist()):
                    if u and v: uni[v]+=1; bi[u,v]+=1; ctx[u]+=1
            if update%16==0:
                voltage=engine.state.voltage[:,::21].detach().cpu().numpy()
                spike=engine.state.spike[:,::21].float().mean().item()
                q=np.quantile(voltage,[0,.01,.5,.99,1]).tolist()
                if not np.isfinite(q).all(): raise RuntimeError('Nonfinite recurrent state')
                tele=dict(update=update,targets=progress['targets'],voltage_quantiles=q,spike_sample_mean=spike,**m)
                progress['telemetry'].append(tele); emit('training',seconds=dt,**tele)
            if update%a.checkpoint_every==0:
                sync_counts(); checkpoint(engine,progress,config,latest)
                checkpoint(engine,progress,config,prefix.with_suffix(f'.step{update:08d}.pt'))
            if update%a.eval_every==0 or update==a.updates:
                sync_counts(); dv=evaluate_preserving(engine,dev); baseline=baselines(dev,progress['counts'])
                entry=dict(update=update,targets=progress['targets'],dev=dv,baseline=baseline)
                progress['curve'].append(entry); emit('validation',**entry)
                progress.setdefault('samples',[]).append(dict(update=update,generation=samples(model)))
                previous=progress['best_dev_ce']
                if previous is None or dv['ce']<previous:
                    progress['best_dev_ce']=dv['ce']; checkpoint(engine,progress,config,prefix.with_suffix('.best.pt'))
                checkpoint(engine,progress,config,latest)
                if previous is not None and dv['ce']>previous+1.0:
                    stopped='validation_regression_gt_1_nat'; break
    sync_counts(); checkpoint(engine,progress,config,latest)
    repeated_dev=[evaluate_preserving(engine,dev) for _ in range(2)]
    generated=samples(model)
    engine.close(); del engine
    summary={k:v for k,v in progress.items() if k not in ('counts','telemetry')}
    return dict(**summary,config=config,specification=model.specification(),parameters=sum(p.numel() for p in model.parameters()),
                generation=generated,repeated_dev=repeated_dev,telemetry=progress['telemetry'],equivalence=equivalence,stop_reason=stopped,
                mean_step_s=sum(measured)/len(measured) if measured else None,checkpoint=str(latest),
                checkpoint_sha256=digest(latest),audit_evaluated=False,test_evaluated=False,
                semantic_verdict='exploratory_pretraining; synthetic memory criteria remain open')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--threshold',type=int,choices=[10,5],default=10);p.add_argument('--head',choices=['tied','separate'],default='tied')
    p.add_argument('--seed',type=int,default=17);p.add_argument('--updates',type=int,required=True);p.add_argument('--resume')
    p.add_argument('--checkpoint-every',type=int,default=2000);p.add_argument('--eval-every',type=int,default=2000)
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=31536000);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if a.updates<0 or min(a.checkpoint_every,a.eval_every)<1:p.error('updates >=0 (0=until stop), positive intervals required')
    if not a.worker:return supervise(a,worker_module='pretrain_resumable')
    try:
        import torch
        emit('load',python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,memory=memory())
        result=run(a);peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20
        result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak,fingerprints=fingerprints())
    except Exception as exc:traceback.print_exc();result=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

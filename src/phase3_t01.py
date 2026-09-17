"""Bounded task-contract checks and independent CUDA Graph GRU controls."""
import argparse, collections, gc, hashlib, json, random, sys, time, traceback
from pathlib import Path
from bench_runtime import emit, supervise, memory

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def oracle(tokens, delay, history=None):
    history = [] if history is None else history
    out = []
    for token in tokens:
        if token == 1:
            history.clear()
            out.append(0)
        elif token == 0:
            out.append(0)
        else:
            history.append(token)
            out.append(history[-delay-1] if len(history) > delay else 0)
    return out

def examples(rows, delay):
    xs = [[1] + row + [0]*7 for row in rows]
    ys = [[0] + [0]*delay + (row[:-delay] if delay else row) + [0]*7 for row in rows]
    return xs, ys

def prepare():
    config = json.loads((ROOT/'configs/phase3_protocol_v1.json').read_text())
    evaluation = json.loads((ROOT/'dataset/prepared_v1/evaluation_indices.json').read_text())
    excluded = set(config['validation_reserved']) | set(range(82))
    excluded.update(r['story_index'] for r in evaluation['splits']['validation'])
    n = json.loads((ROOT/'dataset/prepared_v1/manifest.json').read_text())['splits']['validation']['stories']
    audit = random.Random(93013).sample([i for i in range(n) if i not in excluded], 64)
    streams = {}
    for split, seed, count in [('train',93014,400), ('dev',93015,64)]:
        rng = random.Random(seed)
        streams[split] = [[rng.randrange(400,408) for _ in range(32)] for _ in range(count)]
    assert not set(map(tuple,streams['train'])) & set(map(tuple,streams['dev']))
    manifest = dict(audit_ids=audit, excluded_ids=sorted(excluded), audit_evaluated=False,
                    dev_text_ids=config['validation_reserved'], streams=streams,
                    seeds=dict(audit=93013,train=93014,dev=93015))
    path = ROOT/'configs/phase3_t01_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest,indent=2))
    return manifest

def contracts(manifest):
    checks = []
    for delay in (0,1,4,8):
        xs, ys = examples(manifest['streams']['dev'],delay)
        for x,y in zip(xs,ys):
            assert oracle(x,delay) == y
            assert sum(v != 0 for v in y) == 32-delay
            for width in (8,16):
                history=[]; actual=[]
                for start in range(0,len(x),width):
                    actual += oracle(x[start:start+width],delay,history)
                assert actual == y
            changed=x[:17]+[407 if v!=407 else 400 for v in x[17:33]]+x[33:]
            assert oracle(changed,delay)[:17] == y[:17]
        prefix=[400,401,402,403,404,405,406,407,400,401]
        fixture=[1]+prefix+[0,0,1]+prefix+[0]
        assert oracle(fixture,delay)[14:24] == oracle([1]+prefix,delay)[1:]
        assert oracle([1]+prefix[:5]+[0]+prefix[5:],delay) == oracle([1]+prefix,delay)[:6]+[0]+oracle([1]+prefix,delay)[6:]
        checks.append(dict(delay=delay,streams=64,valid_targets=64*(32-delay),oracle_accuracy=1.,chunks=[8,16],reset=True,pad_freeze=True,causal_prefix=True))
    return checks

def text_baselines(manifest):
    import numpy as np
    import torch
    from text_dataset import StoryDataset
    data={}
    for split,indices in [('train',list(range(18,82))),('validation',manifest['dev_text_ids'])]:
        with StoryDataset(ROOT/'dataset/prepared_v1',split) as ds:
            data[split]=[np.array(ds[i][:129],dtype=np.int64) for i in indices]
    counts=np.ones(4096,dtype=np.float64); pairs=collections.Counter(); totals=collections.Counter()
    for row in data['train']:
        assert row[0]==1 and not np.any(row==0)
        for x,y in zip(row[:-1],row[1:]):
            assert y != 1
            counts[y]+=1; pairs[int(x),int(y)]+=1; totals[int(x)]+=1
    uni=counts/counts.sum(); rows=[]
    for split,stories in data.items():
        for index,row in enumerate(stories):
            # Build only observed contexts; avoid allocating a 4096-square matrix.
            probabilities=[]
            for x in row[:-1]:
                p=10*uni.copy()
                for (context,y),value in pairs.items():
                    if context==x:p[y]+=value
                p/=totals[int(x)]+10
                probabilities.append(p)
            p=torch.tensor(np.array(probabilities),device='cuda',dtype=torch.float32)
            u=torch.tensor(uni,device='cuda',dtype=torch.float32)
            y=torch.tensor(row[1:],device='cuda')
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                stats=torch.stack([-u[y].log().mean(),-p.gather(1,y[:,None]).log().mean(),
                                   (u.argmax()==y).float().mean(),(p.argmax(-1)==y).float().mean()])
            graph.replay(); values=stats.tolist()
            rows.append(dict(split=split,story_index=(list(range(18,82)) if split=='train' else manifest['dev_text_ids'])[index],targets=len(y),unigram_ce=values[0],bigram_ce=values[1],unigram_accuracy=values[2],bigram_accuracy=values[3]))
            graph.reset(); del graph,stats,p,u,y
        emit('baseline',split=split,stories=len(stories))
    result={}
    for split in data:
        selected=[r for r in rows if r['split']==split]; total=sum(r['targets'] for r in selected)
        result[split]=dict(targets=total,**{k:sum(r[k]*r['targets'] for r in selected)/total for k in ['unigram_ce','bigram_ce','unigram_accuracy','bigram_accuracy']})
    return dict(summary=result,per_story=rows,fit_split='train',audit_evaluated=False)

def text_contract():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from lm_io import story_batch
    from fly_lm import FlyLM
    from text_dataset import StoryDataset
    fixtures=[np.array([1,400,401,2]),np.array([1,402,2])]
    ids,targets=story_batch(fixtures,[0,1],[0,1],8)
    assert ids.T.cpu().tolist()==[[1,400,401,0,0,0,0,0],[402,0,0,0,0,0,0,0]]
    assert targets.T.cpu().tolist()==[[400,401,2,0,0,0,0,0],[2,0,0,0,0,0,0,0]]
    logits=torch.zeros(8,2,4096,device='cuda'); graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):loss,count=FlyLM.loss(SimpleNamespace(config=SimpleNamespace(pad=0)),logits,targets,ids)
    graph.replay()
    assert count.item()==4 and abs(loss.item()-__import__('math').log(4096))<1e-5
    graph.reset()
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds:
        for index in (18,81):
            row=np.array(ds[index]); start=len(row)-3
            x,y=story_batch(ds,[index],[start],8)
            assert y[:,0].cpu().tolist()==row[start+1:].tolist()+[0]*6
            assert y[1,0].item()==2 and x[2:,0].count_nonzero().item()==0
    return dict(fixture_mixed_padding=True,eos_scored=True,bos_targets_absent=True,no_cross_story=True,production_loss_valid_count=4,real_train_ids=[18,81])

def gru_case(manifest, delay):
    import torch
    torch.manual_seed(89)
    class Control(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=torch.nn.Embedding(4096,32)
            self.cell=torch.nn.GRUCell(32,64)
            self.head=torch.nn.Linear(64,4096)
        def forward(self,ids):
            h=torch.zeros(2,64,device='cuda'); out=[]
            for token in ids:
                h=h*token.ne(1)[:,None]
                new=self.cell(self.embedding(token),h)
                h=torch.where(token.ne(0)[:,None],new,h)
                out.append(self.head(h))
            return torch.stack(out)
    started=time.perf_counter(); model=Control().cuda()
    opt=torch.optim.Adam(model.parameters(),lr=.003,capturable=True,foreach=False)
    ids=torch.ones(40,2,device='cuda',dtype=torch.long); targets=torch.full_like(ids,400)
    initial=[p.detach().clone() for p in model.parameters()]
    def restore():
        with torch.no_grad():
            for p,v in zip(model.parameters(),initial):p.copy_(v)
            for state in opt.state.values():
                for v in state.values():v.zero_()
        opt.zero_grad(set_to_none=False)
    def forward():
        logits=model(ids); valid=targets.ne(0)&ids.ne(0)
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten(),reduction='none').view_as(targets)
        return torch.stack([(loss*valid).sum(),((logits.argmax(-1)==targets)&valid).sum(),
                            ((logits[:,:,400:408].argmax(-1)+400==targets)&valid).sum(),valid.sum()])
    def step():
        opt.zero_grad(set_to_none=False); stats=forward(); loss=stats[0]/stats[3]
        loss.backward()
        finite=torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()]).all()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False); opt.step()
        finite=finite & torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all() & torch.isfinite(loss)
        return stats,finite
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
    torch.cuda.current_stream().wait_stream(stream); restore()
    emit('capture',delay=delay)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): train_stats,finite=step()
    restore()
    eval_graph=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(eval_graph): eval_stats=forward()
    # Numerical reference uses the capture stream, avoiding gradient stream migration.
    restore()
    with torch.cuda.stream(stream): expected_stats,_=step()
    torch.cuda.current_stream().wait_stream(stream)
    expected=[p.detach().clone() for p in model.parameters()]; expected_stats=expected_stats.detach().clone()
    restore(); graph.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(train_stats,expected_stats,rtol=2e-4,atol=2e-5)
    for p,v in zip(model.parameters(),expected):torch.testing.assert_close(p,v,rtol=2e-4,atol=2e-6)
    del expected,expected_stats,p,v
    restore(); del initial
    load_s=time.perf_counter()-started
    def stage(rows):
        x,y=examples(rows,delay)
        ids.copy_(torch.tensor(x,device='cuda').T); targets.copy_(torch.tensor(y,device='cuda').T)
    def evaluate(rows):
        sums=[0.]*4
        for b in range(0,len(rows),2):
            stage(rows[b:b+2]); eval_graph.replay()
            for i,v in enumerate(eval_stats.tolist()):sums[i]+=v
        return dict(ce=sums[0]/sums[3],accuracy=sums[1]/sums[3],accuracy8=sums[2]/sums[3],targets=int(sums[3]),constant400_accuracy=sum(y==400 for row in examples(rows,delay)[1] for y in row)/sums[3],chance8=.125)
    torch.cuda.reset_peak_memory_stats(); times=[]; slow=0
    emit('training_start',delay=delay)
    for update in range(200):
        t=time.perf_counter(); stage(manifest['streams']['train'][2*update:2*update+2]); graph.replay(); torch.cuda.synchronize()
        times.append(time.perf_counter()-t)
        if not bool(finite):raise RuntimeError('Nonfinite GRU step')
        slow=slow+1 if times[-1]>2 else 0
        if slow>=5:raise RuntimeError('Five consecutive updates >2s')
        if (update+1)%32==0:emit('training',delay=delay,update=update+1,loss=float(train_stats[0]/train_stats[3]))
    result=dict(delay=delay,updates=200,parameters=sum(p.numel() for p in model.parameters()),model_load_s=load_s,
                train_s=sum(times),first_step_s=times[0],mean_step_s=sum(times)/200,valid_training_targets=400*(32-delay),new_symbol_positions=12800,
                train=evaluate(manifest['streams']['train'][:64]),dev=evaluate(manifest['streams']['dev']),equivalence=True,
                peak_vram_mb=torch.cuda.max_memory_allocated()/2**20,loss_finite=True,grad_finite=True,fallback_count=0)
    graph.reset();eval_graph.reset()
    return result

def worker(a):
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(.75)
    emit('load',torch=torch.__version__,gpu=torch.cuda.get_device_name(),memory=memory())
    manifest=prepare()
    hashes={str(p):sha(ROOT/p) for p in ['phase3_t01.py','PROTOCOLLO_T0_T1.md','configs/phase3_t01_manifest.json','dataset/prepared_v1/manifest.json','bench_runtime.py']}
    if a.case=='t0':
        checks=contracts(manifest)
        from phase3_extended import synthetic_pairs
        for delay in (1,4,8):
            pairs,rows=synthetic_pairs(171,delay)
            for b,pair in enumerate(pairs):
                x=torch.cat([v[0] for v in pair]).T.cpu().tolist()
                y=torch.cat([v[1] for v in pair]).T.cpu().tolist()
                assert [oracle(row,delay) for row in x]==y
            del pairs,pair,x,y
        result=dict(checks=checks,historical_generator_match=True,text_contract=text_contract(),text=text_baselines(manifest))
    else:
        result=gru_case(manifest,a.delay)
    gc.collect(); torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    final=torch.cuda.memory_allocated()/2**20
    result.update(ok=final==0,source_sha256=hashes,final_allocator_mb=final,post_cleanup_memory=memory(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),audit_evaluated=False)
    if final!=0:result['error']='Allocator references remain'
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',choices=['t0','gru'],required=True);p.add_argument('--delay',type=int,choices=[0,1,4,8],default=0)
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_t01')
    try:result=worker(a)
    except Exception as exc:
        traceback.print_exc();result=dict(ok=False,error_type=type(exc).__name__,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

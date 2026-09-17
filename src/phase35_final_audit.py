"""Frozen selected model, reserved64 audit once; CUDA Graph inference only."""
import argparse,gc,json,math,sys,traceback
from pathlib import Path
from bench_runtime import ROOT,emit,memory,supervise

def run(a):
    import torch
    from pretrain_resumable import load_payload,digest,baselines,fingerprints
    from fly_lm import copy_state_
    from lm_io import story_batch
    from text_dataset import StoryDataset
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    source=Path(a.checkpoint);data=torch.load(source,map_location='cpu',weights_only=True)
    assert data['fingerprints']==fingerprints()
    model=load_payload(data['model'],data['config']['head'])
    chosen_hash=digest(source);state=model.initial_state(2);empty=model.initial_state(2)
    ids=torch.ones(2,device='cuda',dtype=torch.long);targets=ids.clone()
    def step():
        scores,new=model.step(ids,state);valid=ids.ne(0)&targets.ne(0)
        lp=scores.log_softmax(-1);ce=-lp.gather(1,targets[:,None]).flatten();entropy=-(lp.exp()*lp).sum(-1)
        stats=torch.stack([ce*valid,valid,((scores.argmax(-1)==targets)&valid),entropy*valid],-1)
        copy_state_(state,model.detach(new));return stats
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(),torch.cuda.stream(stream):step()
    torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(graph):out=step()
    if a.smoke:
        rows=json.loads((ROOT/'configs/phase3_protocol_v1.json').read_text())['validation_reserved'][:2];length=16
    else:
        rows=json.loads((ROOT/'configs/phase3_t01_manifest.json').read_text())['audit_ids'];length=128
        assert len(rows)==64
        marker=ROOT/'results/phase35_reserved64_audit_started.json'
        with marker.open('x') as f:json.dump(dict(checkpoint=str(source),sha256=chosen_hash,ids=rows,selection_frozen=True),f,indent=2)
    pairs=[]
    with StoryDataset(ROOT/'dataset/prepared_v1','validation') as ds:
        for b in range(0,len(rows),2):pairs.append([story_batch(ds,rows[b:b+2],[0,0],length)])
    baseline=baselines(pairs,data['progress']['counts']);per_story=[]
    for b,pair in enumerate(pairs):
        copy_state_(state,empty);sums=torch.zeros(2,4,device='cpu')
        for x,y in pair:
            for token,target in zip(x,y):ids.copy_(token);targets.copy_(target);graph.replay();sums+=out.cpu()
        for lane in range(2):per_story.append(dict(id=rows[b*2+lane],ce_sum=float(sums[lane,0]),targets=int(sums[lane,1]),correct=int(sums[lane,2]),entropy_sum=float(sums[lane,3])))
        emit('audit_pair' if not a.smoke else 'smoke_pair',pair=b+1)
    count=sum(r['targets'] for r in per_story);ce=sum(r['ce_sum'] for r in per_story)/count
    assert count>0 and math.isfinite(ce)
    smoke_reference_error=None
    if a.smoke:
        reference=json.loads((ROOT/'results/graph_suite_smoke_h5_anatomical_validation.json').read_text())['result']
        assert reference['checkpoint_sha256']==chosen_hash, 'Smoke reference uses a different checkpoint'
        assert reference['observed_tokens']==count
        smoke_reference_error=abs(ce-reference['ce'])
        assert smoke_reference_error<1e-3, 'Audit CE disagrees with independent probe'
    assert digest(source)==chosen_hash
    graph.reset()
    return dict(checkpoint=str(source),checkpoint_sha256=chosen_hash,config=data['config'],update=data['progress']['updates'],
                audit_evaluated=not a.smoke,test_evaluated=False,selection_frozen=True,ce=ce,targets=count,
                smoke_reference_ce_error=smoke_reference_error,
                accuracy=sum(r['correct'] for r in per_story)/count,per_story=per_story,baseline=baseline,
                beats_unigram=ce<baseline['unigram_ce'],beats_bigram=ce<baseline['bigram_ce'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--output',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true');p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000);a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase35_final_audit')
    try:
        import torch
        emit('load',memory=memory());result=run(a);peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20;result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak)
    except Exception as exc:traceback.print_exc();result=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

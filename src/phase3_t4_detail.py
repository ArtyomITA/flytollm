"""Per-story CE for the existing T4 interventions, no training or new data."""
import argparse,gc,json,sys,traceback
from pathlib import Path
from bench_runtime import emit,supervise,memory

def execute():
    import torch
    from phase3_t45 import ROOT,sha,transformed
    from lm_io import load_model,story_batch
    from text_dataset import StoryDataset
    from fly_lm import LMState,copy_state_
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    emit('load',memory=memory())
    model,_=load_model(ROOT/'results/phase35_adam_s17.diagnostic.pt')
    state=model.initial_state(2);empty=model.initial_state(2);ids=torch.ones(2,device='cuda',dtype=torch.long);targets=ids.clone()
    variants=['real','empty','content_reverse','joint']
    def step():
        weights=model.core.weights();out=[];valid=ids.ne(0)&targets.ne(0)
        for variant in variants:
            changed=LMState(state.voltage,state.spike,transformed(state.cache,model.attention,variant))
            logits,new=model.step(ids,changed,weights=weights)
            if variant=='real':newreal=new
            losses=torch.nn.functional.cross_entropy(logits,targets,reduction='none')
            out.append(torch.stack([losses*valid,valid,((logits.argmax(-1)==targets)&valid)]))
        copy_state_(state,model.detach(newreal));return torch.stack(out)
    with torch.no_grad():step()
    copy_state_(state,empty);graph=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(graph):stats=step()
    indices=json.loads((ROOT/'configs/phase3_protocol_v1.json').read_text())['validation_reserved'];result=[]
    with StoryDataset(ROOT/'dataset/prepared_v1','validation') as ds:
        for start in range(0,16,2):
            x,y=story_batch(ds,indices[start:start+2],[0,0],64);copy_state_(state,empty)
            total=torch.zeros(4,3,2,device='cuda')
            for token,target in zip(x,y):
                ids.copy_(token);targets.copy_(target);graph.replay();total.add_(stats)
            assert bool(torch.isfinite(total).all())
            values=total.cpu().tolist()
            for slot,index in enumerate(indices[start:start+2]):
                scores={name:dict(ce=values[j][0][slot]/values[j][1][slot],targets=int(values[j][1][slot]),accuracy=values[j][2][slot]/values[j][1][slot]) for j,name in enumerate(variants)}
                result.append(dict(story_index=index,scores=scores))
            emit('per_story',stories=len(result))
    graph.reset()
    old=json.loads((ROOT/'results/phase3_t4_text.worker.json').read_text())['aggregate']['dev']
    for row in old:
        name=row['variant'];n=sum(r['scores'][name]['targets'] for r in result)
        ce=sum(r['scores'][name]['ce']*r['scores'][name]['targets'] for r in result)/n
        assert n==row['targets'] and abs(ce-row['ce'])<1e-5
    deltas={}
    for variant in variants[1:]:
        vals=[r['scores'][variant]['ce']-r['scores']['real']['ce'] for r in result]
        deltas[variant]=dict(values=vals,improved_stories=sum(v<0 for v in vals),worsened_stories=sum(v>0 for v in vals),mean=sum(vals)/len(vals))
    return dict(per_story=result,delta_ce=deltas,aggregate_matches_original=True,
                source_sha256={f:sha(ROOT/f) for f in ['phase3_t4_detail.py','phase3_t45.py']},checkpoint_sha256=sha(ROOT/'results/phase35_adam_s17.diagnostic.pt'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true');p.add_argument('--timeout',type=float,default=120);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_t4_detail')
    try:
        r=execute()
        import torch
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20;r.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=torch.cuda.max_memory_allocated()/2**20)
    except Exception as exc:traceback.print_exc();r=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(r,indent=2))
    if not r['ok']:sys.exit(1)

if __name__=='__main__':main()

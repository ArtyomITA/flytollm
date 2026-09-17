"""Fresh GPU process: restore observed checkpoint and execute one next update."""
import argparse,gc,json,sys,traceback
from pathlib import Path
from bench_runtime import ROOT,emit,memory,supervise

def run(a):
    import torch
    from pretrain_resumable import load_payload,restore_runtime,digest,fingerprints,next_pair
    from phase3_t45 import FairCapture
    from text_dataset import StoryDataset
    from lm_io import story_batch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    path=Path(a.checkpoint);original=digest(path)
    data=torch.load(path,map_location='cpu',weights_only=True)
    assert data['fingerprints']==fingerprints()
    model=load_payload(data['model'],data['config']['head'])
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds:first=story_batch(ds,[130,131],[0,0],16)
    engine=FairCapture(model,8,*first);restore_runtime(engine,data)
    for key,value in model.state_dict().items():assert torch.equal(value.cpu(),data['model']['state_dict'][key]),key
    previous=data['progress']['updates'];cursor=dict(data['progress']['cursor'])
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds:x,y=next_pair(ds,cursor,engine)
    before=[float(st['step']) for st in engine.opt.state.values()];assert all(v==previous for v in before)
    metrics=engine.update(x,y)
    after=[float(st['step']) for st in engine.opt.state.values()];assert all(v==previous+1 for v in after)
    assert digest(path)==original
    engine.close()
    return dict(checkpoint=str(path),checkpoint_sha256=original,restored_step=previous,next_step=previous+1,
                cursor_after=cursor,metrics=metrics,model_exact=True,optimizer_and_recurrent_exact=True,
                immutable_source=True,checkpoint_not_overwritten=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true');p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000);a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='verify_resume_after_probe')
    try:
        import torch
        emit('load',memory=memory());result=run(a);peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20;result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak)
    except Exception as exc:traceback.print_exc();result=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

"""Fixed held-out prompts before/after phase3.5; guarded GPU-only generation."""
import argparse,gc,json,sys,traceback
from pathlib import Path
from bench_runtime import supervise,emit


def run_model(model,prompt,tokenizer):
    import torch
    from fly_lm import CUDATokenStep
    decoder=CUDATokenStep(model,2)
    try:
        with torch.no_grad():
            generated=decoder.generate(prompt,24).T.cpu().tolist()
            assert torch.isfinite(decoder.state.voltage).all()
            return [dict(ids=ids,text=tokenizer.decode(ids)) for ids in generated]
    finally:decoder.close()


def run():
    import torch
    from tokenizers import Tokenizer
    from text_dataset import StoryDataset
    from lm_io import build_cns,load_model
    from phase3_extended import initialize
    torch.set_num_threads(1)
    folder=Path('dataset/prepared_v1')
    tokenizer=Tokenizer.from_file(str(folder/'tokenizer-4096.json'))
    indices=json.loads(Path('configs/phase3_protocol_v1.json').read_text(encoding='utf8'))['validation_reserved'][4:6]
    with StoryDataset(folder,'validation') as data:ids=[data[i][:8].astype('int64').tolist() for i in indices]
    prompt=torch.tensor(ids,device='cuda').T.contiguous()
    emit('initial_model');model=build_cns();initialize(model,17)
    before=run_model(model,prompt,tokenizer);del model
    gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache()
    # Locate the one successful stability run; never pick by loss.
    candidates=[]
    for p in Path('results').glob('phase35_adam_s17*.json'):
        if p.name.endswith('.worker.json'):continue
        d=json.loads(p.read_text(encoding='utf8'))
        if d.get('ok'):candidates.append(d['result']['result']['checkpoint'])
    assert len(candidates)==1,candidates
    emit('trained_model');model,_=load_model(candidates[0]);after=run_model(model,prompt,tokenizer)
    del model,prompt
    emit('cleanup');gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
    assert torch.cuda.memory_allocated()==0
    return dict(ok=True,indices=indices,prompts=[tokenizer.decode(x) for x in ids],prompt_ids=ids,
                before=before,after=after,checkpoint=candidates[0],final_allocator_mb=0,
                note='Fixed two validation prompts unused in optimizer selection, greedy24; not a quality benchmark.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--output',default='results/phase35_generation.json');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90)
    p.add_argument('--max-vram-mb',type=int,default=7000);a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='generate_phase35')
    try:r=run()
    except Exception as exc:
        traceback.print_exc();r=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    if not r['ok']:sys.exit(1)


if __name__=='__main__':main()

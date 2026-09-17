"""Verify immutable evidence and equal T5 budgets; no model computation."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    names=['phase3_t1b_d1','phase3_t4_symbols','phase3_t4_text','phase3_t4_detail']+[f'phase3_t5_l{length}_d{delay}' for delay in (1,8) for length in (8,16)]
    runs={};missing=[]
    for name in names:
        path=ROOT/'results'/f'{name}.json'
        if not path.exists():missing.append(name);continue
        root=json.loads(path.read_text());assert root['ok'],name
        result=root['result'];assert result['final_allocator_mb']==0,name
        for file,value in result['source_sha256'].items():assert sha(ROOT/file)==value,(name,file)
        runs[name]=result
    comparisons=[]
    for delay in (1,8):
        keys=[f'phase3_t5_l{length}_d{delay}' for length in (8,16)]
        if not all(k in runs for k in keys):continue
        left,right=[runs[k] for k in keys]
        for field in ['updates','valid_targets','stream_segments','initial_checkpoint_sha256','data_sha256']:assert left[field]==right[field],field
        assert left['updates']==200
        for result in (left,right):
            assert sha(ROOT/'results/phase3_t2_final.pt')==result['initial_checkpoint_sha256']
            assert result['valid_targets']==(4152 if delay==1 else 3214)
        comparisons.append(dict(delay=delay,matched_budget=True,targets=left['valid_targets'],updates=200,
            l8={k:left[k] for k in ['train','dev','train_s','peak_vram_mb','sensitivity']},
            l16={k:right[k] for k in ['train','dev','train_s','peak_vram_mb','sensitivity']},
            dev_accuracy_delta_l16_minus_l8=right['dev']['accuracy']-left['dev']['accuracy'],
            dev_ce_delta_l16_minus_l8=right['dev']['ce']-left['dev']['ce']))
    output=dict(status='complete' if not missing else 'partial',missing=missing,runs=list(runs),hashes_verified=True,cleanup_zero=True,
        t1_d1_calibrated=runs.get('phase3_t1b_d1',{}).get('dev',{}).get('accuracy',0)>=.8,
        architecture_changed=False,nodes_removed=0,audit_evaluated=False,phase3_closed=False,t5=comparisons)
    if 'phase3_t4_detail' in runs:
        output['text_per_story']=runs['phase3_t4_detail']['delta_ce']
        assert runs['phase3_t4_detail']['aggregate_matches_original']
    (ROOT/'results/phase3_t1b_t45_summary.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()

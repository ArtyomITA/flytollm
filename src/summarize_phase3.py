"""Read-only experimental aggregation; no model loads or score selection on test."""
import csv,json,statistics
from pathlib import Path


def read_case(stem):
    choices=list(Path('results').glob(stem+'.json'))+list(Path('results').glob(stem+'_attempt*.json'))
    valid=[]
    for p in choices:
        if p.name.endswith('.worker.json'):continue
        d=json.loads(p.read_text(encoding='utf8'))
        if d.get('ok'):valid.append((p,d))
    if not valid:return None
    if len(valid)>1:raise RuntimeError('Multiple successful attempts require explicit selection: '+stem)
    p,d=valid[0];r=d['result'];return dict(file=str(p),summary=d,worker=r,result=r['result'])


def main():
    grid=[]
    for variant in ('adam','adamw','muon','root'):
        for lr in ('0.0001','0.0003','0.001'):
            for seed in (17,23):
                case=read_case(f'phase34_{variant}_s{seed}_lr{lr}')
                if case:
                    w=case['worker'];r=case['result']
                    grid.append(dict(variant=variant,lr=float(lr),seed=seed,file=case['file'],
                                     initial_ce=r['initial']['ce'],final_ce=r['final']['ce'],accuracy=r['final']['accuracy'],
                                     updates=r['updates'],seen=r['seen_targets'],train_s=r['train_s'],
                                     timed_ce=r['timed20s']['validation']['ce'] if r['timed20s'] else None,
                                     timed_updates=r['timed20s']['updates'] if r['timed20s'] else None,
                                     initial_parameter_sha256=w['initial_parameter_sha256'],peak_allocated_mb=w['peak_vram_mb']))
    selected={}
    if len(grid)==24:
        for seed in (17,23):
            subset=[r for r in grid if r['seed']==seed]
            assert len({r['initial_parameter_sha256'] for r in subset})==1
            assert len({(r['updates'],r['seen']) for r in subset})==1
        for variant in ('adam','adamw','muon','root'):
            candidates=[]
            for lr in (1e-4,3e-4,1e-3):
                rows=sorted([r for r in grid if r['variant']==variant and r['lr']==lr],key=lambda r:r['seed'])
                candidates.append(dict(lr=lr,mean_ce=statistics.mean(r['final_ce'] for r in rows),
                                       per_seed_ce={str(r['seed']):r['final_ce'] for r in rows}))
            selected[variant]=min(candidates,key=lambda r:r['mean_ce'])
        for variant,result in selected.items():
            gains={seed:1-ce/selected['adam']['per_seed_ce'][seed] for seed,ce in result['per_seed_ce'].items()}
            result['relative_gain_vs_adam']=gains
            result['meets_2pct_both_seeds']=all(v>=.02 for v in gains.values())
    memory=[]
    for delay in (1,4,8):
        case=read_case(f'phase33_memory_d{delay}')
        if case:
            r=case['result'];memory.append(dict(delay=delay,file=case['file'],passed=r['passed'],
                                              initial_accuracy=r['initial']['restricted_accuracy'],
                                              final_accuracy=r['final']['restricted_accuracy'],full_accuracy=r['final']['accuracy'],
                                              final_ce=r['final']['ce'],updates=r['updates'],targets=r['final']['targets']))
    stability=read_case('phase35_adam_s17')
    out=dict(memory=memory,grid_complete=len(grid)==24,grid=grid,selected=selected,
             stability=None if stability is None else dict(file=stability['file'],result=stability['result']))
    Path('results/phase3_extended_summary.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    if grid:
        with Path('results/phase34_grid.csv').open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(grid[0]));writer.writeheader();writer.writerows(grid)
    print(json.dumps(dict(memory=memory,grid_cases=len(grid),selected=selected,
                         stability=None if not stability else stability['result']['passed'])))


if __name__=='__main__':main()

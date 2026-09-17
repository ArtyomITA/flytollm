"""Aggregate completed comparison runs; retain missing/failed cases explicitly."""
import csv,hashlib,json
from pathlib import Path


def main():
    rows=[];missing=[]
    for threshold in (10,5):
        for case in ('d1','d4','d8','language'):
            name=f'phase3d_t{threshold}_'+('language' if case=='language' else 'l16_'+case)
            p=Path('results',name+'.json')
            choices=[q for q in [p,*Path('results').glob(name+'_attempt*.json')] if q.exists() and not q.name.endswith('.worker.json')]
            valid=[(q,json.loads(q.read_text())) for q in choices if json.loads(q.read_text()).get('ok')]
            if not valid:missing.append(name+': absent or failed');continue
            assert len(valid)==1,name
            p,s=valid[0]
            w=s['result'];r=w['result']
            for f,h in w['source_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,(name,f)
            assert w['final_allocator_mb']==0
            row=dict(threshold=threshold,case=case,edges=w['specification']['edges'],parameters=w['specification']['parameters'],
                     train_accuracy=r['final_train']['restricted_accuracy'] if r['final_train'] else None,
                     dev_accuracy=r['final']['restricted_accuracy'] if case!='language' else r['final']['accuracy'],
                     initial_ce=r['initial']['ce'],final_ce=r['final']['ce'],passed=r['passed'],
                     updates=r['updates'],seen=r['seen_targets'],train_s=r['train_s'],peak_allocated_mb=w['peak_allocated_mb'],
                     artificial_sha=w['artificial_parameter_sha256'],source=str(p))
            if case=='language':
                row.update(unigram_ce=r['after_diagnostics']['unigram_ce'],generated=r['after_diagnostics']['generated_text'],
                           heldout12_ce=r['heldout12_final']['ce'],constant_generation=r['after_diagnostics']['constant_generation'])
            rows.append(row)
    pairs=[]
    for case in ('d1','d4','d8','language'):
        pair=[r for r in rows if r['case']==case]
        if len(pair)==2:
            a,b=pair
            assert a['artificial_sha']==b['artificial_sha'],case
            matched=(a['updates'],a['seen'])==(b['updates'],b['seen'])
            pairs.append(dict(case=case,equal_budget=matched,train_time_ratio_5_over_10=b['train_s']/a['train_s'],
                              ce_change_5_minus_10=b['final_ce']-a['final_ce'],accuracy_change_5_minus_10=b['dev_accuracy']-a['dev_accuracy']))
    out=dict(complete=not missing,missing=missing,runs=rows,comparisons=pairs)
    Path('results/phase3d_summary.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with Path('results/phase3d_comparison.csv').open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(complete=out['complete'],runs=[{k:r[k] for k in ('threshold','case','dev_accuracy','final_ce','train_s')} for r in rows],missing=missing)))


if __name__=='__main__':main()

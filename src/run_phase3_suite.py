"""Serial orchestration only. Each case retains its existing guarded supervisor."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from bench_runtime import memory,emit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['memory','grid','stability'],required=True)
    p.add_argument('--allow-paging',action='store_true')
    a=p.parse_args();cases=[]
    if a.suite=='memory':
        for d in (1,4,8):cases.append((f'phase33_memory_d{d}', ['--case','memory','--delay',str(d),'--seed','89','--lr','0.001']))
    elif a.suite=='grid':
        for seed in (17,23):
            for lr in ('0.0001','0.0003','0.001'):
                for variant in ('adam','adamw','muon','root'):
                    cases.append((f'phase34_{variant}_s{seed}_lr{lr}',
                                  ['--case','grid','--variant',variant,'--seed',str(seed),'--lr',lr]))
    else:cases=[('phase35_adam_s17',['--case','stability','--seed','17','--lr','0.0001'])]
    for name,args in cases:
        out=Path('results')/(name+'.json')
        if out.exists():
            previous=json.loads(out.read_text(encoding='utf8'))
            if previous.get('ok'):
                hashes=previous['result']['source_sha256']
                if any(hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h for f,h in hashes.items()):
                    raise RuntimeError('Existing successful case has different sources; preserve and review: '+name)
                emit('suite_skip_verified',case=name);continue
            # Preserve every failed attempt instead of overwriting its evidence.
            trial=2
            while out.with_name(name+f'_attempt{trial}.json').exists():trial+=1
            out=out.with_name(name+f'_attempt{trial}.json')
        available=memory()
        low=(available['available_gib']<.125 or available['commit_available_gib']<6) if a.allow_paging else available['available_gib']<2.5
        if low:
            emit('suite_pause_headroom',next_case=name,memory=memory());return 2
        emit('suite_case',case=name,output=str(out))
        result=subprocess.run([sys.executable,'phase3_extended.py',*args,'--output',str(out),
                               *(['--allow-paging'] if a.allow_paging else [])])
        record=json.loads(out.read_text(encoding='utf8')) if out.exists() else {}
        if result.returncode or not record.get('ok'):
            emit('suite_stopped',case=name,record=record);return 1
    emit('suite_complete',suite=a.suite,cases=len(cases));return 0


if __name__=='__main__':sys.exit(main())

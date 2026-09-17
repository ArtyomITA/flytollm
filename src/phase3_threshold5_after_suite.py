"""Run the requested larger-graph comparison after the active GPU queue ends."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

if __name__=='__main__':
    deadline=time.monotonic()+15000
    while time.monotonic()<deadline:
        state=json.loads((ROOT/'results/phase3_suite_live.json').read_text())
        if state['status'] in ('failed','interrupted'):raise RuntimeError('Main suite stopped; no automatic follow-up')
        if state['status']=='completed':break
        time.sleep(10)
    else:raise TimeoutError('Main suite wait exceeded')
    output=ROOT/'results/phase3_variant_threshold5.json'
    if output.exists():raise RuntimeError('Refuse overwrite')
    env=os.environ.copy();env['TEMP']=env['TMP']=str(ROOT/'.runtime-tmp')
    command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'phase3_threshold5_trial.py'),'--variant','threshold5',
             '--output',str(output),'--allow-paging','--timeout','900']
    with (ROOT/'results/phase3_variant_threshold5.console.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise RuntimeError('Threshold5 comparison failed')
    data=json.loads(output.read_text())
    print(json.dumps(dict(ok=data.get('ok'),output=str(output))))

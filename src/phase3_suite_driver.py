"""Serial authorized experiment queue. Each child retains bench_runtime guards."""
import argparse,json,os,subprocess,time
from pathlib import Path
import ctypes
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);a=p.parse_args()
    if a.wait_pid:
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x00100000,False,a.wait_pid)
        if handle:
            try:
                if kernel.WaitForSingleObject(handle,1300000)!=0:raise RuntimeError('Previous process wait failed')
            finally:kernel.CloseHandle(handle)
        result=json.loads((ROOT/'results/phase3_t5_long_l8_d1.json').read_text())
        if not result.get('ok'):raise RuntimeError('Previous run failed')
    jobs=[]
    def add(module,name,args,timeout=600):jobs.append(dict(module=module,name=name,args=args,timeout=timeout))
    for d in (4,8):add('phase3_t01c',f'phase3_t1c_d{d}',['--case','gru','--delay',str(d)])
    add('phase3_topology','phase3_p0',[])
    for variant in ('base','p1','p2','d1','d2','h1'):
        add('phase3_variants',f'phase3_variant_{variant}',['--variant',variant])
    add('phase3_text_pilot','phase3_t6_smoke',['--budget','128'])
    for delay in (1,8):
        for length in (8,16):add('phase3_t5_2000',f'phase3_t5_2000_l{length}_d{delay}',['--case','t5','--length',str(length),'--delay',str(delay)],1800)
    add('phase3_text_pilot','phase3_t6_s17',['--seed','17'],4000)
    state=dict(status='running',jobs=jobs,completed=[])
    def save():
        (ROOT/'results/phase3_suite_live.json').write_text(json.dumps(state,indent=2))
    env=os.environ.copy();env['TEMP']=env['TMP']=str(ROOT/'.runtime-tmp')
    for job in jobs:
        output=ROOT/'results'/f"{job['name']}.json"
        if output.exists():raise RuntimeError(f'Refuse overwrite: {output}')
        command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/(job['module']+'.py')),*job['args'],
                 '--output',str(output),'--allow-paging','--timeout',str(job['timeout'])]
        state.update(current=job['name'],started=time.time());save();print(json.dumps(dict(phase='job_start',job=job)),flush=True)
        log=ROOT/'results'/f"{job['name']}.console.log"
        with log.open('w',encoding='utf8') as handle:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            try:code=proc.wait(timeout=job['timeout']+120)
            except BaseException:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                state['status']='interrupted';save();raise
        result=json.loads(output.read_text()) if output.exists() else {}
        state['completed'].append(dict(name=job['name'],ok=code==0 and result.get('ok',False),finished=time.time()))
        print(json.dumps(dict(phase='job_end',**state['completed'][-1])),flush=True)
        if not state['completed'][-1]['ok']:
            state['status']='failed';save();return
        save()
    state['status']='completed';save()

if __name__=='__main__':main()

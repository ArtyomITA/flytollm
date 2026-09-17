"""Guarded synthetic core benchmark; no LM throughput extrapolation."""
import argparse, ctypes, gc, hashlib, json, os, statistics, subprocess, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def memory():
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ['total','avail','page','apage','virtual','avirtual','ext']]
    m=M();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return dict(ram_pct=m.load,available_gib=m.avail/2**30,
                commit_available_gib=m.apage/2**30,commit_limit_gib=m.page/2**30)

def gpu_used():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
        return int(r.stdout.splitlines()[0])
    except Exception: return None

def rss_mb(pid):
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in ['peak','rss','pp','qp','pn','qn','page','peakpage']]
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    fn=ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    handle=kernel.OpenProcess(0x1000|0x10,False,pid)
    if not handle:return 0.
    try:
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        return counters.rss/2**20 if fn(handle,ctypes.byref(counters),counters.cb) else 0.
    finally:kernel.CloseHandle(handle)

def emit(phase,**data):
    print(json.dumps(dict(phase=phase,pid=os.getpid(),time=time.time(),**data)),flush=True)

def worker(a):
    import numpy as np
    import torch
    from fly_core import Core
    from fly_graph import load_graph
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['bench_runtime.py','fly_core.py','fly_graph.py']}
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.cuda.set_per_process_memory_fraction(.75)
    prop=torch.cuda.get_device_properties(0)
    emit('load',python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=prop.name,memory=memory())
    t=time.perf_counter()
    if a.toy:
        n=24;s=np.arange(n,dtype=np.int64)
        g=dict(src=np.tile(s,2),dst=np.concatenate([(s+1)%n,(s+5)%n]),weight=np.ones(2*n,dtype=np.float32),sign=np.where(s%4==0,-1,1).astype(np.float32),sensory=s%3==0,body_ids=s)
    else: g=load_graph(a.soglia,progress=lambda msg:emit('load',detail=msg))
    n=len(g['body_ids']);e=len(g['src'])
    saved_steps=a.tronca*a.T if not a.checkpoint_chars else min(a.checkpoint_chars,a.tronca)*a.T
    estimate=5*a.batch*n*saved_steps+12*a.batch*n*a.tronca+96*e+24*a.batch*a.chunk+128*2**20
    if estimate>5.5*2**30: raise RuntimeError(f'Estimated {estimate/2**30:.2f} GiB exceeds 5.5 GiB')
    emit('init',neurons=n,edges=e,estimated_gib=estimate/2**30)
    src=torch.from_numpy(g['src']).cuda();dst=torch.from_numpy(g['dst']).cuda()
    counts=np.log1p(g['weight']);sums=np.bincount(g['dst'],weights=counts,minlength=n)
    mag=(a.gain*counts/np.maximum(sums[g['dst']],1)).astype(np.float32)
    model=Core(src,dst,torch.from_numpy(mag).cuda(),torch.from_numpy(g['sign']).cuda(),n,a.chunk)
    sensory=torch.from_numpy(np.flatnonzero(g['sensory'])).cuda()
    if sensory.numel()==0: raise RuntimeError('No sensory nodes')
    drive=torch.zeros(a.tronca,a.batch,n,device='cuda')
    drive[:,:,sensory]=.4+1.2*torch.rand(a.tronca,a.batch,len(sensory),device='cuda')
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,capturable=True,foreach=False)
    initial=model.raw.detach().clone()
    del g,counts,sums,mag
    gc.collect();load_s=time.perf_counter()-t
    def step():
        opt.zero_grad(set_to_none=False)
        loss,rate,_=model.window(drive,steps=a.T,checkpoint_chars=a.checkpoint_chars)
        loss.backward();opt.step()
        return loss,rate
    def reset():
        with torch.no_grad():
            model.raw.copy_(initial)
            for state in opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value): value.zero_()
            opt.zero_grad(set_to_none=False)
        torch.cuda.synchronize()
    def diagnose(loss,rate):
        grad=model.raw.grad
        d=dict(loss=float(loss.detach()),firing_rate=float(rate),grad_finite=bool(torch.isfinite(grad).all()),grad_abs_max=float(grad.abs().max()),params_finite=bool(torch.isfinite(model.raw).all()))
        if not(np.isfinite(d['loss']) and 0<d['firing_rate']<1 and d['grad_finite'] and d['grad_abs_max']>0 and d['params_finite']): raise RuntimeError(f'Invalid core: {d}')
        return d
    def measure(fn,name,reset_peak=True):
        times=[];metrics=[]
        if reset_peak:torch.cuda.reset_peak_memory_stats()
        for i in range(a.passi):
            emit(name,iteration=i);torch.cuda.synchronize();t=time.perf_counter()
            loss,rate=fn();torch.cuda.synchronize();times.append(time.perf_counter()-t)
            metrics.append(diagnose(loss,rate));emit(name+'_done',seconds=times[-1],**metrics[-1])
        return dict(times_s=times,median_s=statistics.median(times),peak_vram_mb=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20,metrics=metrics,
                    synthetic_positions_per_sec=a.batch*a.tronca/statistics.median(times))
    emit('warmup')
    for i in range(2):
        emit('warmup',iteration=i)
        loss,rate=step()
        diagnose(loss,rate)
    del loss,rate
    torch.cuda.synchronize();reset()
    eager=measure(step,'eager')
    expected={k:v.detach().cpu().clone() for k,v in opt.state[model.raw].items() if torch.is_tensor(v)}
    expected_raw=model.raw.detach().cpu().clone()
    expected_grad=model.raw.grad.detach().cpu().clone()
    changed=float((model.raw.detach()-initial).abs().max())
    if changed==0: raise RuntimeError('Weights unchanged')
    result=dict(ok=True,scenario='synthetic_core',config=vars(a),source_sha256=hashes,neurons=n,edges=e,model_load_s=load_s,torch=torch.__version__,gpu=prop.name,
                eager=eager,parameter_change_max=changed,fallback_count=0,semantic_verdict='active_core_only',
                notes=f'Synthetic sensory current each substep; incoming log-count magnitude normalized to {a.gain}. Independent windows. No tokenizer/readout/CE.')
    graph=None
    if not a.no_graph:
        try:
            emit('graph_warmup');reset()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): step()
            torch.cuda.current_stream().wait_stream(stream)
            reset();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            emit('capture');graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): static_loss,static_rate=step()
            reset()
            def replay():
                graph.replay()
                return static_loss,static_rate
            measured=measure(replay,'graph',reset_peak=False)
            actual=model.raw.detach().cpu()
            torch.testing.assert_close(actual,expected_raw,rtol=2e-5,atol=2e-6)
            errors=dict(raw_max_abs=float((actual-expected_raw).abs().max()))
            grad=model.raw.grad.detach().cpu()
            torch.testing.assert_close(grad,expected_grad,rtol=2e-4,atol=1e-10)
            errors['grad_max_abs']=float((grad-expected_grad).abs().max())
            for k,v in expected.items():
                current=opt.state[model.raw][k].detach().cpu()
                torch.testing.assert_close(current,v,rtol=2e-4,atol=2e-7)
                errors[k+'_max_abs']=float((current-v).abs().max())
            for x,y in zip(measured['metrics'],eager['metrics']):
                for key in ['loss','firing_rate']:
                    if not np.isclose(x[key],y[key],rtol=2e-5,atol=2e-7): raise RuntimeError(f'Graph mismatch: {key}')
            measured['equivalence']=errors;result['graph']=measured
            result['graph_speedup_pct']=100*(eager['median_s']/measured['median_s']-1)
            del replay,static_loss,static_rate
        except Exception as exc:
            result.update(ok=False,graph_error=f'{type(exc).__name__}: {exc}',semantic_verdict='eager_valid_graph_unvalidated',fallback_count=1)
            emit('graph_error',error=result['graph_error'])
    emit('cleanup')
    if graph is not None: graph.reset()
    del graph,step,reset,diagnose,measure,opt,model,initial,drive,src,dst,sensory
    gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();gc.collect()
    result['post_cleanup_memory']=dict(**memory(),gpu_alloc_mb=torch.cuda.memory_allocated()/2**20,gpu_reserved_mb=torch.cuda.memory_reserved()/2**20)
    result['stability_verdict']='stable' if result['post_cleanup_memory']['gpu_alloc_mb']<1 else 'allocator_refs_remain'
    return result

def supervise(a, worker_module="bench_runtime"):
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);log=out.with_suffix('.jsonl')
    pre=memory();used=gpu_used();summary=dict(pre_memory=dict(**pre,gpu_used_mb=used),guard_abort=None)
    paging=getattr(a,'allow_paging',False)
    summary['guard_profile']='paging_opt_in' if paging else 'physical_ram_default'
    pre_low=(pre['available_gib']<.125 or pre['commit_available_gib']<6) if paging else pre['available_gib']<1.75
    if pre_low or (used is not None and used>6500):
        summary.update(ok=False,guard_abort='preflight_headroom');out.write_text(json.dumps(summary,indent=2));emit('finished',**summary);return
    child=out.with_suffix('.worker.json')
    if child.exists(): child.unlink()
    # Bypass Windows venv launcher. One worker process; no subprocesses inside worker.
    # -S excludes global site-packages. Use this interpreter's environment paths.
    bootstrap='import sys; sys.path[:0]='+repr([str(ROOT),str(Path(sys.prefix)/'Lib'/'site-packages')])+f'; from {worker_module} import main; main()'
    cmd=[sys._base_executable,'-S','-u','-c',bootstrap,*sys.argv[1:],'--worker']
    started=time.monotonic();last=started;heartbeat=0;peak=pre['ram_pct'];peak_gpu=used or 0;peak_rss=0.;tracked=set()
    with log.open('w',encoding='utf-8') as sink:
        p=subprocess.Popen(cmd,stdout=sink,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        tracked.add(p.pid)
        def kill():
            # Held process handle works when taskkill PID lookup is access-denied.
            # Worker uses threads only; terminating it also terminates CUDA work.
            p.kill()
            p.wait(timeout=15)
        try:
            with log.open('r',encoding='utf-8') as reader:
                while p.poll() is None:
                    for line in reader:
                        print(line.rstrip(),flush=True);last=time.monotonic()
                        try:tracked.add(json.loads(line)['pid'])
                        except (ValueError,KeyError):pass
                    now=time.monotonic();m=memory();v=gpu_used();peak=max(peak,m['ram_pct']);peak_gpu=max(peak_gpu,v or 0)
                    peak_rss=max(peak_rss,sum(rss_mb(pid) for pid in tracked))
                    ram_low=m['available_gib']<(.125 if paging else 1.25)
                    commit_low=paging and m['commit_available_gib']<4
                    phase_limit=min(a.phase_timeout,90) if paging else a.phase_timeout
                    reason=('host_ram' if ram_low else 'commit_headroom' if commit_low else 'vram' if v is not None and v>a.max_vram_mb else 'total_timeout' if now-started>a.timeout else 'phase_timeout' if now-last>phase_limit else None)
                    if reason: summary['guard_abort']=reason;kill();break
                    if now-heartbeat>10: emit('watchdog',memory=m,gpu_used_mb=v);heartbeat=now
                    time.sleep(2)
                for line in reader: print(line.rstrip(),flush=True)
        finally:
            if p.poll() is None: kill()
    time.sleep(2)
    summary.update(exit_code=p.returncode,peak_system_ram_pct=peak,peak_gpu_used_mb=peak_gpu,peak_process_tree_ram_mb=peak_rss,post_exit_memory=dict(**memory(),gpu_used_mb=gpu_used()),elapsed_s=time.monotonic()-started)
    if child.exists(): summary['result']=json.loads(child.read_text())
    summary['ok']=p.returncode==0 and summary['guard_abort'] is None and summary.get('result',{}).get('ok',False)
    out.write_text(json.dumps(summary,indent=2),encoding='utf-8');emit('finished',ok=summary['ok'],output=str(out),guard_abort=summary['guard_abort'])
    if not summary['ok']:sys.exit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,default in [('soglia',10),('batch',1),('tronca',4),('T',8),('chunk',262144),('passi',3),('checkpoint-chars',0),('seed',123)]: p.add_argument('--'+key,type=int,default=default)
    for key in ['no-graph','toy','worker']: p.add_argument('--'+key,action='store_true')
    p.add_argument('--gain',type=float,default=.5)
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90)
    p.add_argument('--max-vram-mb',type=int,default=7000);p.add_argument('--output',default=str(ROOT/'results'/'benchmark.json'))
    a=p.parse_args()
    for key in ['soglia','batch','tronca','T','chunk','passi']:
        if getattr(a,key)<1:p.error(key+' must be positive')
    if a.checkpoint_chars<0:p.error('checkpoint-chars must be nonnegative')
    if not 0<a.gain<=100:p.error('gain must be in (0,100]')
    if a.worker:
        try: result=worker(a)
        except Exception as exc:
            traceback.print_exc();result=dict(ok=False,error_type=type(exc).__name__,error=str(exc))
        gc.collect()
        if 'torch' in sys.modules:
            try:
                torch=sys.modules['torch']
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize();torch.cuda.empty_cache();gc.collect()
                    result['final_allocator_mb']=torch.cuda.memory_allocated()/2**20
            except Exception as exc:result['cleanup_error']=str(exc)
        Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        if not result['ok']:sys.exit(1)
    else:supervise(a)

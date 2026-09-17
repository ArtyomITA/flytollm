"""Approved: compare text candidates, train to20k, isolated probes, resume until stop."""
import argparse, ctypes, json, math, os, subprocess, time
from pathlib import Path
from bench_runtime import ROOT

STATE=ROOT/'results/pretrain_night_live.json'
STOP=ROOT/'STOP_PRETRAINING'

def write_state(state):
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));os.replace(temporary,STATE)

def score(result):
    values=[r['ce'] for r in result['repeated_dev']]
    assert values and all(math.isfinite(value) for value in values), 'Invalid validation CE'
    return sum(values)/len(values)

def choose(results):
    """Sequential interventions: tied10 -> separate10 -> separate5."""
    assert all(r['stop_reason']=='update_budget' for r in results.values())
    assert len({r['updates'] for r in results.values()})==1
    assert len({r['targets'] for r in results.values()})==1
    assert len({r['artificial_initial_sha256'] for r in results.values()})==1
    selected='t10'
    if score(results['h10']) < score(results['t10']):
        selected='h10'
        if score(results['h5']) <= score(results['h10']):selected='h5'
    return selected

def choose_night_pair(results):
    assert all(r['stop_reason']=='update_budget' for r in results.values())
    assert len({r['updates'] for r in results.values()})==1
    assert len({r['targets'] for r in results.values()})==1
    if set(results)=={'t10','h10'}:return 'h10' if score(results['h10'])<score(results['t10']) else 't10'
    if set(results)=={'h10','h5'}:return 'h5' if score(results['h5'])<=score(results['h10']) else 'h10'
    raise ValueError('Unapproved night comparison pair')

def run_job(state, name, module, args, timeout, optional=False):
    if STOP.exists(): raise InterruptedError('User stop file present')
    output=ROOT/'results'/f'{name}.json'
    if output.exists():
        if not optional:raise RuntimeError(f'Refuse overwrite {output}')
        state['completed'].append(dict(name=name,ok=False,optional=True,error='Output already exists; preserve it and skip probe',finished=time.time()))
        write_state(state);return None
    pause=ROOT/'PAUSE_BETWEEN_RUNS'  # user request 15/9 20:00: rest the GPU between runs; file holds the seconds
    if pause.exists():
        seconds=float((pause.read_text().strip() or '300'))
        print(json.dumps(dict(event='pause',name=name,seconds=seconds)),flush=True);time.sleep(seconds)
    # user rule 17/9 02:50: three minutes of rest after roughly every hour and a half of GPU activity (accumulated
    # across queues in a file; "un'ora non per forza precisa, anche un'ora e mezza")
    activity=ROOT/'results'/'.gpu_activity.json'
    try:acc=json.loads(activity.read_text())
    except (OSError,ValueError):acc={'seconds_since_rest':0.0}
    if acc.get('seconds_since_rest',0.0)>=5400:
        print(json.dumps(dict(event='pause',name=name,seconds=180,reason='hourly rest')),flush=True);time.sleep(180)
        acc['seconds_since_rest']=0.0;activity.write_text(json.dumps(acc))
    command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/f'{module}.py'),*map(str,args),
             '--output',str(output),'--allow-paging','--timeout',str(timeout)]
    state.update(status='running',current=name,command=command,started=time.time());write_state(state)
    print(json.dumps(dict(event='start',name=name)),flush=True)
    env=os.environ.copy();env['TEMP']=env['TMP']=str(ROOT/'.runtime-tmp')
    with output.with_suffix('.console.log').open('w',encoding='utf8') as log:
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        except OSError as exc:
            if not optional:raise
            state['completed'].append(dict(name=name,ok=False,optional=True,error=str(exc),finished=time.time()))
            write_state(state);return None
        state['supervisor_pid']=process.pid;write_state(state)
        code=process.wait()
    try:
        acc=json.loads(activity.read_text()) if activity.exists() else {'seconds_since_rest':0.0}
        acc['seconds_since_rest']=float(acc.get('seconds_since_rest',0.0))+(time.time()-state['started']);acc['last_job']=name
        activity.write_text(json.dumps(acc))
    except (OSError,ValueError):pass
    try:data=json.loads(output.read_text()) if output.exists() else {}
    except (ValueError,OSError):data={}
    good=code==0 and data.get('ok',False) and isinstance(data.get('result'),dict)
    state['completed'].append(dict(name=name,ok=good,optional=optional,finished=time.time()));write_state(state)
    print(json.dumps(dict(event='end',name=name,ok=good)),flush=True)
    if not good and not optional: raise RuntimeError(f'{name} failed; see {output}')
    return data.get('result') if good else None

def test_checkpoint_path(result):
    path=Path(result['checkpoint'])
    return path if path.is_absolute() else ROOT/path

def probe_plan(source, prefix, smoke=False):
    jobs=[]
    for null in (False,True):
        for split in ('train','validation'):
            name=f'{prefix}_{"null" if null else "anatomical"}_{split}'
            args=['--checkpoint',str(source),'--split',split]
            if null:args+=['--null-groups']
            if smoke:args+=['--stories',2,'--tokens',16]
            jobs.append((name,args))
    reset_args=['--checkpoint',str(source),'--split','validation','--reset-each-token']
    if smoke:reset_args+=['--stories',2,'--tokens',16]
    jobs.append((f'{prefix}_reset_validation',reset_args))
    return jobs

def optional_suite(state,source,prefix):
    state.pop('context_reset_delta_ce',None)
    state.pop('analysis',None)
    results=[]
    for name,args in probe_plan(source,prefix):
        results.append(run_job(state,name,'graph_specialization_probe',args,600,optional=True))
    if all(r is not None for r in results[:4]):
        try:
            command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'analyze_graph_specialization.py'),
                     '--train',results[0]['observations'],'--dev',results[1]['observations'],
                     '--null-train',results[2]['observations'],'--null-dev',results[3]['observations'],
                     '--output',str(ROOT/'results'/f'{prefix}_analysis.json')]
            analyzed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
            state['analysis']=dict(ok=analyzed.returncode==0,stderr=analyzed.stderr[-2000:])
        except Exception as exc:state['analysis']=dict(ok=False,error=str(exc))
    else:state['analysis']=dict(ok=False,reason='one or more observational workers failed; resume original checkpoint')
    if results[1] is not None and results[4] is not None and 'ce' in results[1] and 'ce' in results[4]:
        state['context_reset_delta_ce']=results[4]['ce']-results[1]['ce']
    write_state(state)
    return all(r is not None for r in results)

def main():
    global STATE
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--from-comparison',action='store_true');a=p.parse_args()
    if a.self_test:
        def item(ce):return dict(repeated_dev=[dict(ce=ce),dict(ce=ce)],updates=2000,targets=60000,artificial_initial_sha256='same',stop_reason='update_budget')
        assert choose(dict(t10=item(6),h10=item(5.9),h5=item(5.8)))=='h5'
        assert choose(dict(t10=item(6),h10=item(6.1),h5=item(5.8)))=='t10'
        assert choose(dict(t10=item(6),h10=item(5.9),h5=item(6)))=='h10'
        assert choose_night_pair(dict(t10=item(6),h10=item(5.9)))=='h10'
        assert choose_night_pair(dict(h10=item(6),h5=item(6.1)))=='h10'
        plan=probe_plan('untouched.pt','probe');assert len(plan)==5
        assert all(args[1]=='untouched.pt' for _,args in plan)
        print('selection smoke passed');return
    if (ROOT/'HOLD_PRETRAINING').exists():
        raise RuntimeError('Long pretraining on hold: user must explicitly authorize bedtime launch first.')
    if a.from_comparison:STATE=ROOT/'results/pretrain_actual_night_live.json'
    if STATE.exists(): raise RuntimeError('Night queue already has state; do not launch twice')
    if a.from_comparison:
        review=json.loads((ROOT/'results/phase35_4_final_review.json').read_text())
        if not review.get('ok') or not review.get('night_launch_authorized_by_user'):
            raise RuntimeError('Phase3.5/4 final verification missing or failed')
    checks=['pretrain_smoke_t10','pretrain_resume_smoke_t10','pretrain_smoke_h5','pretrain_resume_smoke_h5']
    checks += ['resume_after_suite_smoke','phase35_audit_harness_smoke']
    checks += [f'resume_candidate_{key}_smoke' for key in ('t10','h10','h5')]
    checks += [name for name,_ in probe_plan('unused','graph_suite_smoke_h5',smoke=True)]
    for name in checks:
        data=json.loads((ROOT/'results'/f'{name}.json').read_text())
        if not data.get('ok'):raise RuntimeError(f'Required smoke failed: {name}')
    state=dict(status='starting',completed=[],smokes=checks,policy='phase3.5+4 -> approved pair to8000 each ->winner to20000 ->observational suite ->until stop',
               checkpoint_interval=2000,all_research_phase3_criteria_closed=False)
    write_state(state)
    # Keep the system awake during the authorized unattended job; display may sleep.
    keep_awake=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:raise OSError('Cannot inhibit automatic sleep for unattended training')
    try:
        candidates={};config={'t10':(10,'tied'),'h10':(10,'separate'),'h5':(5,'separate')}
        for key,(threshold,head) in config.items():
            if a.from_comparison:
                ready=json.loads((ROOT/'results/phase35_4_compare_live.json').read_text())
                if ready.get('status')!='completed_waiting_user':raise RuntimeError('Phase3.5+4 comparison not finished')
                completed=json.loads((ROOT/'results'/f'pretrain_pilot_{key}.json').read_text())
                if not completed.get('ok'):raise RuntimeError(f'Candidate failed: {key}')
                candidates[key]=completed['result']
            else:
                candidates[key]=run_job(state,f'pretrain_pilot_{key}','pretrain_resumable',
                                       ['--threshold',threshold,'--head',head,'--updates',2000],7200)
        selected=choose(candidates);chosen=candidates[selected]
        if a.from_comparison:
            assert review['selected']==selected
            assert review['checkpoint_sha256']==chosen['checkpoint_sha256']
        if score(chosen)>=chosen['initial_dev']['ce']:raise RuntimeError('No validation improvement from initialization')
        state.update(selected=selected,comparison={k:dict(ce=score(v),parameters=v['parameters'],mean_step_s=v['mean_step_s'],
                                                        targets=v['targets'],baseline=v['curve'][-1]['baseline']) for k,v in candidates.items()})
        write_state(state)
        pair=('t10','h10') if selected=='t10' else ('h10','h5')
        order=[selected]+[key for key in pair if key!=selected];night_results={}
        for key in order:
            threshold,head=config[key]
            night_results[key]=run_job(state,f'pretrain_night8000_{key}','pretrain_resumable',
                                       ['--threshold',threshold,'--head',head,'--updates',8000,'--resume',test_checkpoint_path(candidates[key])],18000)
            if night_results[key]['stop_reason']!='update_budget':
                state.update(status='stopped',reason=night_results[key]['stop_reason']);write_state(state);return
            probe_ok=optional_suite(state,test_checkpoint_path(night_results[key]),f'graph_probe_8000_{key}')
            state.setdefault('probe_8000',{})[key]=dict(ok=probe_ok,analysis=state.get('analysis'),
                                                      context_reset_delta_ce=state.get('context_reset_delta_ce'))
            write_state(state)
        assert len({r['targets'] for r in night_results.values()})==1
        selected=choose_night_pair(night_results);chosen=night_results[selected]
        state.update(selected_after8000=selected,night_comparison={k:dict(ce=score(v),targets=v['targets'],updates=v['updates'],
                                                                        checkpoint=v['checkpoint'],generation=v['generation']) for k,v in night_results.items()})
        write_state(state)
        threshold,head=config[selected]
        trained=run_job(state,'pretrain_to20000','pretrain_resumable',
                        ['--threshold',threshold,'--head',head,'--updates',20000,'--resume',test_checkpoint_path(chosen)],86400)
        if trained['stop_reason']!='update_budget' or trained['updates']!=20000:
            state.update(status='stopped',reason=trained['stop_reason']);write_state(state);return
        source=test_checkpoint_path(trained)
        # Failure of observational probes is recorded, not used to alter/replace training state.
        state['probe_20000_ok']=optional_suite(state,source,'graph_probe_20000');write_state(state)
        resumed=run_job(state,'pretrain_continuous','pretrain_resumable',
                        ['--threshold',threshold,'--head',head,'--updates',0,'--resume',source],31536000)
        state.update(status='stopped',reason=resumed['stop_reason']);write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped',reason=str(exc));write_state(state)
    except Exception as exc:
        state.update(status='failed',error=str(exc));write_state(state);raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()

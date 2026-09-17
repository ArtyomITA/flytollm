"""Finish phase3.5/4 checks serially after the running three-profile comparison."""
import ctypes
import hashlib
import json
import time
from pathlib import Path
import pretrain_night_queue as q


def main():
    q.STATE=q.ROOT/'results/phase35_completion_live.json'
    if q.STATE.exists(): raise RuntimeError('Completion queue already exists; refuse duplicate')
    state=dict(status='waiting_comparison', completed=[])
    q.write_state(state)
    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
        raise OSError('Cannot inhibit automatic system sleep')
    try:
        deadline=time.monotonic()+3*3600
        while True:
            if q.STOP.exists():raise InterruptedError('User stop file present')
            ready=json.loads((q.ROOT/'results/phase35_4_compare_live.json').read_text())
            if ready['status']=='completed_waiting_user':break
            if ready['status'] in ('failed','stopped'):raise RuntimeError(f'Comparison {ready}')
            if time.monotonic()>deadline:raise TimeoutError('Comparison did not finish in3h')
            time.sleep(5)
        candidates={key:json.loads((q.ROOT/'results'/f'pretrain_pilot_{key}.json').read_text()) for key in ('t10','h10','h5')}
        assert all(item['ok'] and item['result']['final_allocator_mb']==0 for item in candidates.values())
        candidates={key:item['result'] for key,item in candidates.items()}
        selected=q.choose(candidates);chosen=candidates[selected]
        assert q.score(chosen)<chosen['initial_dev']['ce']
        source=q.test_checkpoint_path(chosen)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==chosen['checkpoint_sha256']
        state.update(selected=selected, frozen_checkpoint=str(source), frozen_sha256=chosen['checkpoint_sha256'])
        q.write_state(state)
        smoke=q.ROOT/'results/pretrain_resume_smoke_h5.latest.pt'
        q.run_job(state,'resume_after_suite_smoke','verify_resume_after_probe',['--checkpoint',smoke],600)
        q.run_job(state,'phase35_audit_harness_smoke','phase35_final_audit',['--checkpoint',smoke,'--smoke'],600)
        threshold,head={'t10':(10,'tied'),'h10':(10,'separate'),'h5':(5,'separate')}[selected]
        replica=q.run_job(state,'phase35_selected_replica_s23','pretrain_resumable',
                          ['--threshold',threshold,'--head',head,'--seed',23,'--updates',2000],7200)
        assert replica['updates']==chosen['updates'] and replica['targets']==chosen['targets']
        assert replica['stop_reason']=='update_budget'
        audit=q.run_job(state,'phase35_reserved64_audit','phase35_final_audit',['--checkpoint',source],600)
        assert audit['checkpoint_sha256']==chosen['checkpoint_sha256'] and audit['audit_evaluated']
        report=dict(ok=True, selected=selected, selection_frozen_before_replica_and_audit=True,
                    checkpoint=str(source), checkpoint_sha256=chosen['checkpoint_sha256'],
                    comparison={key:dict(ce=q.score(r),targets=r['targets'],updates=r['updates'],
                                         parameters=r['parameters'],mean_step_s=r['mean_step_s'],
                                         peak_vram_mb=r['peak_vram_mb'],baseline=r['curve'][-1]['baseline'],
                                         generation=r['generation']) for key,r in candidates.items()},
                    replica=dict(ce=q.score(replica),initial_ce=replica['initial_dev']['ce'],
                                 improves_initial=q.score(replica)<replica['initial_dev']['ce'],
                                 targets=replica['targets'],seed=23),
                    audit=audit, all_phase3_quality_criteria_passed=False,
                    night_launch_authorized_by_user=True, long_pretraining_launched=False)
        (q.ROOT/'results/phase35_4_final_review.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        state.update(status='ready_for_night_launch',report='results/phase35_4_final_review.json')
        q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped',error=str(exc));q.write_state(state)
    except Exception as exc:
        state.update(status='failed',error=str(exc));q.write_state(state);raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()

"""Phase 7b: what the user's stop of 22:22 (16 September) left undone: the G4 run (three "tengo" levers together,
2000 updates) and the inference suite on the 1000-update checkpoint of the standard baseline. Generous supervisor
timeouts so that a pause (pause_runs.py) does not expire the run. One GPU worker; PAUSE_BETWEEN_RUNS honoured."""
import ctypes, json, subprocess, time
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD, T8, SHORTPATH, RELTHR5, inference

STOP = ROOT / 'STOP_PHASE7'


def main():
    q.STATE = ROOT / 'results/phase7b_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7b queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#G')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        try:
            q.run_job(state, 'phase7_tengo_all_2000', 'pretrain_control', STANDARD + T8 + SHORTPATH + RELTHR5 + ['--updates', 2000, '--checkpoint-every', 2000], 21600)
        except RuntimeError as exc:
            state['failed'].append(dict(name='phase7_tengo_all_2000', error=str(exc))); q.write_state(state)
        checkpoint = ROOT / 'results/phase7_baseline_4000.step00001000.pt'
        if checkpoint.exists():
            inference(state, checkpoint, 'baseline_1000')
        else:
            state['failed'].append(dict(name='inference_baseline_1000', error='checkpoint missing')); q.write_state(state)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

"""Phase 7h (proposed 17 September 2026, 14:10, after K1 Muon 1e-3 = 4.276 beat Muon 3e-4 by 0.135; launch is the
user's decision): the pre-registered follow-up of K1 and the promotion of the best Muon learning rate to 8000, in place
of K6 (Muon 3e-4 at 8000, superseded). Section K of SUITE_TEST_FASE_6B.md (K7, K8). One GPU worker, hourly 3-minute
rest handled by pretrain_night_queue.run_job, generous timeouts (pausable with pause_runs.py).

  K7 Muon 3e-3 on the baseline (K1 reading: "the optimal lr is still higher")
  K8 promotion to 8000 of the best Muon lr among 1e-3 (K1) and 3e-3 (K7), picked by rule; the recorder references it to
     the standard baseline at 8000 (K0) once that exists
Estimated GPU time: 13 + 55 min."""
import ctypes, json
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD
from phase7g_queue import best

STOP = ROOT / 'STOP_PHASE7'


def main():
    q.STATE = ROOT / 'results/phase7h_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7h queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#K')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')

    def job(name, extra, updates, timeout, checkpoint_every=2000):
        try:
            q.run_job(state, name, 'pretrain_control', STANDARD + extra + ['--updates', updates, '--checkpoint-every', checkpoint_every], timeout)
        except RuntimeError as exc:
            state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)

    try:
        job('phase7_lever_muon3e3_2000', ['--optimizer', 'muon', '--muon-lr', 3e-3], 2000, 14400)
        muon_lr = best([(1e-3, 'phase7_lever_muon1e3_2000'), (3e-3, 'phase7_lever_muon3e3_2000')])
        state['picked'] = dict(muon_lr=muon_lr); q.write_state(state)
        job('phase7_promote_muon_8000', ['--optimizer', 'muon', '--muon-lr', muon_lr], 8000, 21600)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

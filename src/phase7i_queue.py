"""Phase 7i (user, 17 September 2026 17:17: "fermare ora K8 e rilanciarla a 1e-3"; "non usiamo robe non buone sennò
sfancula tutti i test"): the promotion of Muon to 8000 with the learning rate the pre-registration of K7 said to keep
(1e-3: K7 at 3e-3 was within 0.10 of K1, plateau). The 7h queue had picked 3e-3 by a naive lowest-CE rule and was
killed at about 2000 updates; its partial files stay as phase7_promote_muon_8000.* (no results JSON).
Section K8 of SUITE_TEST_FASE_6B.md. One GPU worker, hourly rest handled by pretrain_night_queue.run_job."""
import ctypes
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD

STOP = ROOT / 'STOP_PHASE7'


def main():
    q.STATE = ROOT / 'results/phase7i_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7i queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#K8')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        try:
            q.run_job(state, 'phase7_promote_muon1e3_8000', 'pretrain_control',
                      STANDARD + ['--optimizer', 'muon', '--muon-lr', 1e-3, '--updates', 8000, '--checkpoint-every', 2000], 21600)
        except RuntimeError as exc:
            state['failed'].append(dict(name='phase7_promote_muon1e3_8000', error=str(exc))); q.write_state(state)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

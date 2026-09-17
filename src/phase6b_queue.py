"""Phase 6b, point 1 short runs (PREREGISTRAZIONE_FASE_6B.md): configuration-model smoke, second rewire seed at 2000,
configuration-model null at 2000. Sequential GPU jobs through pretrain_night_queue.run_job."""
import ctypes, json, time
import pretrain_night_queue as q
from bench_runtime import ROOT

STOP = ROOT / 'STOP_PHASE6'
COMMON = ['--threshold', 10, '--head', 'separate', '--seed', 17, '--eval-every', 500, '--checkpoint-every', 2000]


def main():
    q.STATE = ROOT / 'results/phase6b_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase6b queue state exists; do not launch twice')
    state = dict(status='starting', completed=[], protocol='PREREGISTRAZIONE_FASE_6B.md')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        q.run_job(state, 'phase6b_smoke_configmodel', 'pretrain_control',
                  COMMON + ['--rewire-seed', 41, '--rewire-kind', 'config', '--updates', 4], 1800)
        q.run_job(state, 'phase6b_rewired_s43_2000', 'pretrain_control',
                  COMMON + ['--rewire-seed', 43, '--rewire-kind', 'degree', '--updates', 2000], 7200)
        q.run_job(state, 'phase6b_configmodel_s41_2000', 'pretrain_control',
                  COMMON + ['--rewire-seed', 41, '--rewire-kind', 'config', '--updates', 2000], 7200)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

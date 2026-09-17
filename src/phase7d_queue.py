"""Phase 7d (user "si" of 17 September 2026, 02:05): follow-ups of section H on the standard baseline, 2000 updates,
curve every 500. H1b/H1c: the per-type bias with a learning rate that does not destabilise (1e-4 = same as the rest,
no dedicated group; 3e-4). Further follow-ups appended before launch if section H asks for them. Pausable, generous
timeouts, one GPU worker."""
import ctypes
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD

STOP = ROOT / 'STOP_PHASE7'
RUNS = [
    ('phase7_lever_bias_lr1e4_2000', ['--core-variant', 'bias_type']),
    ('phase7_lever_bias_lr3e4_2000', ['--core-variant', 'bias_type', '--type-param-lr', 3e-4]),
]


def main():
    q.STATE = ROOT / 'results/phase7d_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7d queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#H')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        for name, extra in RUNS:
            try:
                q.run_job(state, name, 'pretrain_control', STANDARD + extra + ['--updates', 2000, '--checkpoint-every', 2000], 14400)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

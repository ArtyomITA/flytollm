"""Phase 8a (user, 17 September 2026 18:30: "fai per ogni categoria i corti ... segnati il nuovo candidato sicuro per
default e parti con quello"): the short tests that need no new code, on the safe default candidate
(STANDARD_MOSCA_6B.md + Muon 1e-3 on the dense matrices, confirmed at 8000 by K8: 3.541 vs 3.652). Section L of
SUITE_TEST_FASE_6B.md. No automatic choice inside this queue: every configuration is fixed.
One GPU worker, hourly 3-minute rest handled by pretrain_night_queue.run_job, pausable with pause_runs.py.

  L4 default + 8+8 substeps                        (measured 8+8: 26-29 min)
  F9 default + 8+8 + anatomical short-path ports   (about 28 min)
  L5 default + 12+12 substeps                      (not measured, about 40 min)
  L3 K5 (default + 8+8 + conductance + tau 1e-2) with data seed 23   (measured K5: 46 min)
  L7 K5 with initial weight scale x0.5             (46 min)"""
import ctypes
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD, T8, SHORTPATH

STOP = ROOT / 'STOP_PHASE7'
DEFAULT = STANDARD + ['--optimizer', 'muon', '--muon-lr', 1e-3]          # safe default candidate, 17 September 2026
K5 = DEFAULT + T8 + ['--core-variant', 'tau_type+reversal', '--type-param-lr', 1e-2]

JOBS = [
    ('phase8_L4_T8_2000', DEFAULT + T8, 14400),
    ('phase8_F9_shortpath_T8_2000', DEFAULT + T8 + SHORTPATH, 14400),
    ('phase8_L5_T12_2000', DEFAULT + ['--pre-steps', 12, '--post-steps', 12], 21600),
    ('phase8_L3_best_seed23_2000', K5 + ['--seed', 23], 21600),
    ('phase8_L7_best_scale05_2000', K5 + ['--weight-scale', 0.5], 21600),
]


def main():
    q.STATE = ROOT / 'results/phase8a_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase8a queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#L')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        for name, args, timeout in JOBS:
            try:
                q.run_job(state, name, 'pretrain_control', args + ['--updates', 2000, '--checkpoint-every', 2000], timeout)
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

"""Phase 7e (user request 17 September 2026, 02:20: "run con Muon dove pensi che possano servire"): Muon on the dense
matrices of the external channel (attention q/k/v/o and readout projection, 301.568 values), Adam on the rest, where
the external channel matters most: the standard baseline (random ports, attention worth 0.19-0.27 nat) and the 8+8
configuration. Section I of SUITE_TEST_FASE_6B.md. 2000 updates, curve every 500, pausable, generous timeouts.

  I1  baseline + Muon base lr 1e-4 (effective 3.2e-4 on 256-dim matrices)
  I2  baseline + Muon base lr 3e-4 (effective ~1e-3, the phase-3.4 grid winner)
  I3  8+8 substeps + Muon base lr 1e-4 (the best phase-7 configuration with Muon)
  I4  baseline + Muon 1e-4 also on the output head (4096x256)"""
import ctypes
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD, T8

STOP = ROOT / 'STOP_PHASE7'
RUNS = [
    ('muon', ['--optimizer', 'muon', '--muon-lr', 1e-4]),
    ('muon3e4', ['--optimizer', 'muon', '--muon-lr', 3e-4]),
    ('T8muon', T8 + ['--optimizer', 'muon', '--muon-lr', 1e-4]),
    ('muonhead', ['--optimizer', 'muon', '--muon-lr', 1e-4, '--muon-head']),
    ('T8rev', T8 + ['--core-variant', 'reversal']),  # user 'si' 02:45: the two levers above threshold together (section G5)
]


def main():
    q.STATE = ROOT / 'results/phase7e_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7e queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#I')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    failed = set()
    try:
        for tag, extra in RUNS:
            if tag in ('muon3e4', 'T8muon', 'T8rev'):
                continue  # same mechanisms as 'muon' / already smoked in phase 7 and 7c
            name = f'phase7_smoke_{tag}'
            try:
                q.run_job(state, name, 'pretrain_control', STANDARD + extra + ['--updates', 4, '--checkpoint-every', 4], 1800)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state); failed.add(tag)
        if 'muon' in failed:
            failed |= {'muon3e4', 'T8muon'}
        for tag, extra in RUNS:
            name = f'phase7_lever_{tag}_2000'
            if tag in failed:
                state['failed'].append(dict(name=name, error='smoke failed, run skipped')); q.write_state(state); continue
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

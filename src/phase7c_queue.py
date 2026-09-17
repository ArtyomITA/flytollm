"""Phase 7c (SUITE_TEST_FASE_6B.md, section H; user go "2" of 17 September 2026 00:58): the "aggiunte / modificate"
levers of RICERCA_ATTIVITA_CERVELLO_MOSCA.md, each added alone to the standard baseline (STANDARD_MOSCA_6B.md), 2000
updates with the DEV curve every 500. Generous supervisor timeouts (pausable with pause_runs.py). One GPU worker.

  H1 bias_type      trainable resting drive per cell type (11.752), init 0, dedicated lr 1e-3
  H2 tau_type       trainable leak per cell type, dedicated lr 1e-3 (D6 redone with a learning rate that can move it)
  H3 reversal       conductance-like synapses on the standard (D7 was on anatomical ports)
  H4 excitability   bias_type + tau_type + reversal together, dedicated lr 1e-3
  H5 monoamines     dopamine / octopamine / serotonin neurons (541) with sign -1 (Pospisil convention) instead of +1
  H6 weight scale   initial synaptic magnitudes x0.5 and x2 (global scale, Shiu-style single scalar)
A failed smoke skips its run; a failed run is recorded and the queue continues."""
import ctypes
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD

STOP = ROOT / 'STOP_PHASE7'
LR = ['--type-param-lr', 1e-3]
LEVERS = [
    ('bias', ['--core-variant', 'bias_type'] + LR),
    ('tau', ['--core-variant', 'tau_type'] + LR),
    ('reversal', ['--core-variant', 'reversal']),
    ('excit', ['--core-variant', 'bias_type+tau_type+reversal'] + LR),
    ('mono', ['--rewire-kind', 'relthr_mono']),
    ('scale05', ['--weight-scale', 0.5]),
    ('scale2', ['--weight-scale', 2.0]),
]


def main():
    q.STATE = ROOT / 'results/phase7c_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7c queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#H')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    failed = set()
    try:
        for tag, extra in LEVERS:
            if tag == 'scale05':
                continue  # same mechanism as scale2: one smoke is enough
            name = f'phase7_smoke_{tag}'
            try:
                q.run_job(state, name, 'pretrain_control', STANDARD + extra + ['--updates', 4, '--checkpoint-every', 4], 1800)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state); failed.add(tag)
        if 'scale2' in failed:
            failed.add('scale05')
        for tag, extra in LEVERS:
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

"""Phase 7f (user, 17 September 2026 02:50, going to sleep: "continua te e i test che valuti consigliati, falli"):
tests I judge worth running after 7e, section J of SUITE_TEST_FASE_6B.md, on the standard baseline, one GPU worker,
hourly 3-minute rest handled by pretrain_night_queue.run_job, generous timeouts (pausable with pause_runs.py).

  J1 tau_type with lr 3e-3 (H2 gave +0.06 at 1e-3: does a larger step give more?)
  J2 tau_type (lr 1e-3) + conductance synapses (H2 + H3, the two stable positive dynamics levers)
  J3 inference suite on the 8+8 checkpoint at 2000 (how the extra substeps change activity / lesions / attention)
  J4 promotions to 8000 of the levers above threshold: 8+8 (G1), conductance (H3), and 8+8 + conductance (G5) if
     it beats G1 by >= 0.10 at 2000 (promotion rule of the suite)"""
import ctypes, json
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD, T8, inference

STOP = ROOT / 'STOP_PHASE7'
REV = ['--core-variant', 'reversal']


def ce_at(name, update):
    path = ROOT / 'results' / f'{name}.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not data.get('ok'):
        return None
    return next((e['dev']['ce'] for e in data['result']['curve'] if e['update'] == update), None)


def main():
    q.STATE = ROOT / 'results/phase7f_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7f queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#J')
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
        job('phase7_lever_tau3e3_2000', ['--core-variant', 'tau_type', '--type-param-lr', 3e-3], 2000, 14400)
        job('phase7_lever_revtau_2000', ['--core-variant', 'tau_type+reversal', '--type-param-lr', 1e-3], 2000, 14400)
        checkpoint = ROOT / 'results/phase7_tengo_T8_2000.latest.pt'
        if checkpoint.exists():
            inference(state, checkpoint, 'T8_2000')
        job('phase7_promote_T8_8000', T8, 8000, 21600)
        job('phase7_promote_reversal_8000', REV, 8000, 21600)
        g5, g1 = ce_at('phase7_lever_T8rev_2000', 2000), ce_at('phase7_tengo_T8_2000', 2000)
        if g5 is not None and g1 is not None and g5 <= g1 - 0.10:
            job('phase7_promote_T8rev_8000', T8 + REV, 8000, 28800)
        else:
            state['skipped'] = dict(name='phase7_promote_T8rev_8000', reason=f'G5 {g5} vs G1 {g1}: promotion rule not met'); q.write_state(state)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

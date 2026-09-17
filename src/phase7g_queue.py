"""Phase 7g (proposed 17 September 2026, 13:30, after the night chain 7c-7f; launch is the user's decision):
follow-ups of the winning levers, section K of SUITE_TEST_FASE_6B.md, on the standard baseline, one GPU worker,
hourly 3-minute rest handled by pretrain_night_queue.run_job, generous timeouts (pausable with pause_runs.py).

  K1 Muon 1e-3 on the baseline (I2 gave +0.169 at 3e-4, I1 -0.068 at 1e-4: steep lr curve)
  K2 tau per type with lr 1e-2 (J1 doubled H2 by going 1e-3 -> 3e-3)
  K3 8+8 + Muon 3e-4 (I3 used the wrong lr 1e-4)
  K4 conductance + tau 3e-3 (J2 used tau 1e-3)
  K5 all winners together: 8+8 + conductance + tau (best lr of J1/K2) + Muon (best lr of I2/K1), picked by rule
  K0 standard baseline at 8000 (the missing reference for the J4 promotions)
  K6 promotion of Muon 3e-4 to 8000 (suite rule: I2 +0.169 >= 0.10)
Estimated GPU time: 13 + 15 + 24 + 25 + 45 + 55 + 55 min = about 3 h 50, plus rests."""
import ctypes, json
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD, T8

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


def best(candidates, update=2000, margin=0.10):
    """(value, name) pairs, the FIRST is the value in use. A later candidate replaces it only if its run beats the
    current pick by at least `margin` nat at `update` (the suite's effect threshold): differences inside the threshold
    are a plateau and the pre-registered reading is "keep the value in use". Written on 17 September 2026 after the 7h
    queue picked Muon 3e-3 over 1e-3 for a 0.026 difference, against the K7 pre-registration. The margin rule is the
    assistant's PROPOSAL, not a user rule: state it in the queue proposal and get it approved before the next launch."""
    pick, pick_ce = candidates[0][0], ce_at(candidates[0][1], update)
    for value, name in candidates[1:]:
        ce = ce_at(name, update)
        if ce is None:
            continue
        if pick_ce is None or ce <= pick_ce - margin:
            pick, pick_ce = value, ce
    return pick


def main():
    q.STATE = ROOT / 'results/phase7g_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7g queue state exists; do not launch twice')
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
        job('phase7_lever_muon1e3_2000', ['--optimizer', 'muon', '--muon-lr', 1e-3], 2000, 14400)
        job('phase7_lever_tau1e2_2000', ['--core-variant', 'tau_type', '--type-param-lr', 1e-2], 2000, 14400)
        job('phase7_lever_T8muon3e4_2000', T8 + ['--optimizer', 'muon', '--muon-lr', 3e-4], 2000, 14400)
        job('phase7_lever_revtau3e3_2000', ['--core-variant', 'tau_type+reversal', '--type-param-lr', 3e-3], 2000, 14400)
        muon_lr = best([(3e-4, 'phase7_lever_muon3e4_2000'), (1e-3, 'phase7_lever_muon1e3_2000')])
        tau_lr = best([(3e-3, 'phase7_lever_tau3e3_2000'), (1e-2, 'phase7_lever_tau1e2_2000')])
        state['picked'] = dict(muon_lr=muon_lr, tau_lr=tau_lr); q.write_state(state)
        job('phase7_lever_best_2000', T8 + ['--core-variant', 'tau_type+reversal', '--type-param-lr', tau_lr,
                                            '--optimizer', 'muon', '--muon-lr', muon_lr], 2000, 21600)
        job('phase7_baseline_8000', [], 8000, 21600)
        job('phase7_promote_muon3e4_8000', ['--optimizer', 'muon', '--muon-lr', 3e-4], 8000, 21600)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

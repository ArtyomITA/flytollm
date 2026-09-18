"""Phase 8c (18 September 2026): follow-ups of queue 8b, on the safe default candidate (standard + Muon 1e-3). Section O of
SUITE_TEST_FASE_6B.md. Fixed configurations, no automatic choice. Resumable like 8b (finished runs are skipped).

  C15   synapses put back to their initial values on the K8 checkpoint (8b run failed on a script bug, fixed)
  N0    diagnostic re-run for the cross-entropy only (the 8b JSON used the tied embedding instead of the separate head)
  L8    ALL levers at 12+12: 12+12 + conductance + tau 1e-2 + Muon 1e-3 (user question of 18 September 00:55)
  H5    smooth arousal at 12+12
  H6    "wake up and learn" at 8+8: per-type homeostasis + widened weight gradient + synaptic lr SYN_LR
  H7    the same at 12+12
  L9    default candidate + 8+8 at 8000 updates (the missing reference at the new base depth)
  L10   default candidate + 12+12 at 8000 updates (reference of C18)
  C18   "wake up and learn" at 12+12, 8000 updates
  C17   widened gradient + synaptic lr SYN_LR at 8+8, 8000 updates (user, 18 September 01:00: "2000 steps bastano?"): does training more of the
        wiring pay late, once the interfaces have saturated? Reference K8 3.541
SYN_LR = 3e-3: proposed by the assistant after the M0 micro-sweep and APPROVED by the user on 18 September 2026 01:05
("3e-3 va bene"). (3e-3: coverage 17.7% of the synapses, CE unchanged
at 300 updates)."""
import ctypes, json, time
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase8b_queue import DEFAULT, diagnostic

STOP = ROOT / 'STOP_PHASE7'
SYN_LR = 3e-3
T8 = ['--pre-steps', 8, '--post-steps', 8]     # base depth since the user decision of 18 September 00:55
T12 = ['--pre-steps', 12, '--post-steps', 12]
WAKE = ['--core-variant', 'homeo+soft_gw', '--homeo', '0.02,2e-5', '--core-lr', SYN_LR]
RUNS = [
    # shorts (2000 updates)
    ('phase8_L8_best_T12_2000', T12 + ['--core-variant', 'tau_type+reversal', '--type-param-lr', 1e-2], 21600),
    ('phase8_H5_arousal_smooth_T12_2000', T12 + ['--core-variant', 'arousal', '--arousal', 'smooth,0.3,500,0'], 14400),
    ('phase8_H6_wake_learn_T8_2000', T8 + WAKE, 21600),
    ('phase8_H7_wake_learn_T12_2000', T12 + WAKE, 21600),
    # longs (8000 updates): the references at the same depth first, then the 'more synapses learn' runs (user: 12+12 at 8000 too)
    ('phase8_L9_T8_8000', T8, 43200, 8000),
    ('phase8_C17_soft_corelr_T8_8000', T8 + ['--core-variant', 'soft_gw', '--core-lr', SYN_LR], 43200, 8000),
    ('phase8_L10_T12_8000', T12, 43200, 8000),
    ('phase8_C18_wake_learn_T12_8000', T12 + WAKE, 43200, 8000),
]


def main():
    q.STATE = ROOT / 'results/phase8c_live.json'; q.STOP = STOP
    if q.STATE.exists():
        previous = json.loads(q.STATE.read_text())
        if previous.get('status') in ('running', 'completed'):
            raise RuntimeError(f"Phase8c queue state is {previous.get('status')}; do not launch twice")
        q.STATE.replace(q.STATE.with_name(f'phase8c_live.stopped_{int(time.time())}.json'))
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#O')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')

    def done(run):
        path = ROOT / 'results' / f'{run}.json'
        return path.exists() and json.loads(path.read_text()).get('ok') is True

    try:
        # 18 September 01:45: the core-learning experiments (section P, phase8d_queue) run BEFORE the 8c runs
        import phase8d_queue, phase8e_queue
        phase8d_queue.main()
        phase8e_queue.main()   # section Q: input channels by modality (user yes, 18 September 01:45)
        q.STATE = ROOT / 'results/phase8c_live.json'
        k8 = ROOT / 'results/phase7_promote_muon1e3_8000.latest.pt'
        diagnostic(state, k8, 'default8000_v2', script='phase8_c15_revert_weights.py', prefix='c15', extra=('--kind', 'relthr'))
        diagnostic(state, k8, 'default8000_v2')
        try:
            if not done('phase8_smoke_wake'):
                q.run_job(state, 'phase8_smoke_wake', 'pretrain_control', DEFAULT + T12 + WAKE + ['--updates', 4, '--checkpoint-every', 4], 1800)
            smoke = True
        except RuntimeError as exc:
            smoke = False
            state['failed'].append(dict(name='phase8_smoke_wake', error=str(exc))); q.write_state(state)
        for name, extra, timeout, *rest in RUNS:
            updates = rest[0] if rest else 2000
            if done(name):
                state['completed'].append(dict(name=name, ok=True, skipped='result exists (resumed queue)', finished=time.time())); q.write_state(state)
                continue
            if 'wake' in name and not smoke:
                state['failed'].append(dict(name=name, error='smoke wake failed: skipped')); q.write_state(state)
                continue
            try:
                q.run_job(state, name, 'pretrain_control', DEFAULT + extra + ['--updates', updates, '--checkpoint-every', 2000], timeout)
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

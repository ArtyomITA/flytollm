"""Phase 8d (18 September 2026 01:45, after RICERCA_CORE_NON_IMPARA.md; user question: how do we make the core itself
learn, isolating it from the other components?): the core-learning experiments E1-E4 of the report, section P of
SUITE_TEST_FASE_6B.md. Base: standard + Muon 1e-3 + 8+8 (reference L4 4.187), except E4 which continues the 8000-update
4+4 checkpoint K8 (3.541) with every interface frozen. Fixed configurations, no automatic choice; resumable; one smoke
per code family. Runs by phase8c_queue.main() before its own runs (chained after 8b2)."""
import ctypes, json, time
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase8b_queue import DEFAULT

STOP = ROOT / 'STOP_PHASE7'
T8 = ['--pre-steps', 8, '--post-steps', 8]
K8 = str(ROOT / 'results/phase7_promote_muon1e3_8000.latest.pt')
CORE_ONLY = ['--init-from', K8, '--freeze-interfaces']
RUNS = [
    # E4: the decisive test of the residual capacity of the core: only the synapses train, from the trained checkpoint (4+4)
    ('phase8_E4_core_only_lr1e4_2000', CORE_ONLY, 'coreonly'),
    ('phase8_E4_core_only_lr1e2_2000', CORE_ONLY + ['--core-lr', 1e-2], 'coreonly'),
    ('phase8_E4_core_only_soft_lr1e2_2000', CORE_ONLY + ['--core-lr', 1e-2, '--core-variant', 'soft_gw'], 'coreonly'),
    # E1: fluctuation-driven initialisation (variance normalisation), 8+8
    ('phase8_E1_fluct_g1_T8_2000', T8 + ['--init-norm', 'fluct', '--weight-scale', 1.0], 'fluct'),
    ('phase8_E1_fluct_g22_T8_2000', T8 + ['--init-norm', 'fluct', '--weight-scale', 2.2], 'fluct'),
    # E3: slow interfaces, fast core, 8+8
    ('phase8_E3_slow_iface10_T8_2000', T8 + ['--interface-lr', 1e-5, '--muon-lr', 1e-4, '--core-lr', 3e-3], 'iface'),
    ('phase8_E3_slow_iface100_T8_2000', T8 + ['--interface-lr', 1e-6, '--muon-lr', 1e-5, '--core-lr', 3e-3], 'iface'),
    # E2: output scale, 8+8
    ('phase8_E2_outscale01_T8_2000', T8 + ['--output-scale', 0.1], 'outscale'),
    ('phase8_E2_outscale001_T8_2000', T8 + ['--output-scale', 0.01], 'outscale'),
]
SMOKES = dict(coreonly=CORE_ONLY + ['--core-lr', 1e-2, '--core-variant', 'soft_gw'], fluct=T8 + ['--init-norm', 'fluct', '--weight-scale', 2.2],
              iface=T8 + ['--interface-lr', 1e-6, '--muon-lr', 1e-5, '--core-lr', 3e-3], outscale=T8 + ['--output-scale', 0.01])


def done(run):
    path = ROOT / 'results' / f'{run}.json'
    return path.exists() and json.loads(path.read_text()).get('ok') is True


def main():
    q.STATE = ROOT / 'results/phase8d_live.json'; q.STOP = STOP
    if q.STATE.exists():
        previous = json.loads(q.STATE.read_text())
        if previous.get('status') in ('running', 'completed'):
            raise RuntimeError(f"Phase8d queue state is {previous.get('status')}; do not launch twice")
        q.STATE.replace(q.STATE.with_name(f'phase8d_live.stopped_{int(time.time())}.json'))
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#P')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        smoke_ok = {}
        for name, extra, family in RUNS:
            if done(name):
                state['completed'].append(dict(name=name, ok=True, skipped='result exists (resumed queue)', finished=time.time())); q.write_state(state)
                continue
            if family not in smoke_ok:
                if done(f'phase8_smoke_{family}'):
                    smoke_ok[family] = True
                else:
                    try:
                        q.run_job(state, f'phase8_smoke_{family}', 'pretrain_control', DEFAULT + SMOKES[family] + ['--updates', 4, '--checkpoint-every', 4], 1800)
                        smoke_ok[family] = True
                    except RuntimeError as exc:
                        smoke_ok[family] = False
                        state['failed'].append(dict(name=f'phase8_smoke_{family}', error=str(exc))); q.write_state(state)
            if not smoke_ok[family]:
                state['failed'].append(dict(name=name, error=f'smoke {family} failed: skipped')); q.write_state(state)
                continue
            try:
                q.run_job(state, name, 'pretrain_control', DEFAULT + extra + ['--updates', 2000, '--checkpoint-every', 2000], 21600)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state); raise
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

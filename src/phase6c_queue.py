"""Phase 6c training variants (SUITE_TEST_FASE_6B.md, categories B and D, plus E1/E2): 2000-update controls on the
whole graph with the identical pipeline. Waits for the inference suite to finish (one GPU worker at a time), runs a
4-update smoke for every new mechanism, then the 2000-update runs in priority order; a failed smoke or run is
recorded and the queue continues."""
import ctypes, json, subprocess, time
import pretrain_night_queue as q
from bench_runtime import ROOT

STOP = ROOT / 'STOP_PHASE6'
COMMON = ['--threshold', 10, '--head', 'separate', '--seed', 17, '--eval-every', 500, '--checkpoint-every', 2000, '--fast-mode', 'fused']
INFERENCE_LOG = ROOT / 'results/phase6c_inference.console.log'

# (name, extra args) in priority order
RUNS = [
    ('phase6c_permweights_2000', ['--rewire-kind', 'weights', '--rewire-seed', 41]),
    ('phase6c_permsigns_2000', ['--rewire-kind', 'signs', '--rewire-seed', 41]),
    ('phase6c_ports_random_2000', ['--rewire-kind', 'none', '--ports', 'random_matched']),
    ('phase6c_erdosrenyi_2000', ['--rewire-kind', 'er', '--rewire-seed', 41]),
    ('phase6c_withinsuperclass_2000', ['--rewire-kind', 'superclass', '--rewire-seed', 41]),
    ('phase6c_relthr_2000', ['--rewire-kind', 'relthr', '--rewire-seed', 41]),
    ('phase6c_softsign_2000', ['--rewire-kind', 'softsign', '--rewire-seed', 41]),
    ('phase6c_ports_olfactory_2000', ['--rewire-kind', 'none', '--ports', 'olfactory']),
    ('phase6c_readout_anatomical_2000', ['--rewire-kind', 'none', '--readout', 'anatomical']),
    ('phase6c_core_apl_2000', ['--rewire-kind', 'none', '--core-variant', 'apl']),
    ('phase6c_core_graded_ol_2000', ['--rewire-kind', 'none', '--core-variant', 'graded_ol']),
    ('phase6c_core_tau_type_2000', ['--rewire-kind', 'none', '--core-variant', 'tau_type']),
    ('phase6c_core_reversal_2000', ['--rewire-kind', 'none', '--core-variant', 'reversal']),
]
SMOKES = [
    ('phase6c_smoke_relthr', ['--rewire-kind', 'relthr', '--rewire-seed', 41]),
    ('phase6c_smoke_readout_anatomical', ['--rewire-kind', 'none', '--readout', 'anatomical']),
    ('phase6c_smoke_core_apl', ['--rewire-kind', 'none', '--core-variant', 'apl']),
    ('phase6c_smoke_core_graded_ol', ['--rewire-kind', 'none', '--core-variant', 'graded_ol']),
    ('phase6c_smoke_core_tau_type', ['--rewire-kind', 'none', '--core-variant', 'tau_type']),
    ('phase6c_smoke_core_reversal', ['--rewire-kind', 'none', '--core-variant', 'reversal']),
]


def wait_for_inference():
    while True:
        done = INFERENCE_LOG.exists() and INFERENCE_LOG.read_text(encoding='utf8', errors='replace').count('=== exit') >= 6
        if done:
            return
        time.sleep(30)


def main():
    q.STATE = ROOT / 'results/phase6c_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase6c queue state exists; do not launch twice')
    state = dict(status='waiting_for_inference', completed=[], protocol='SUITE_TEST_FASE_6B.md', failed=[])
    q.write_state(state)
    wait_for_inference()  # inference log already complete
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    skip = set()
    try:
        for name, extra in SMOKES:
            try:
                q.run_job(state, name, 'pretrain_control', COMMON + extra + ['--updates', 4], 1800)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
                skip.add(tuple(extra))
        for name, extra in RUNS:
            if tuple(extra) in skip:
                state['failed'].append(dict(name=name, error='smoke failed, run skipped')); q.write_state(state); continue
            try:
                q.run_job(state, name, 'pretrain_control', COMMON + extra + ['--updates', 2000], 7200)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
        try:
            q.run_job(state, 'phase6c_reorder_100', 'pretrain_control',
                      COMMON + ['--rewire-kind', 'reorder', '--rewire-seed', 41, '--updates', 100], 1800)
        except RuntimeError as exc:
            state['failed'].append(dict(name='phase6c_reorder_100', error=str(exc))); q.write_state(state)
        bench = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), str(ROOT / 'phase6c_microbench.py')], cwd=ROOT,
                               capture_output=True, text=True, timeout=1200, creationflags=subprocess.CREATE_NO_WINDOW)
        state['microbench'] = dict(ok=bench.returncode == 0, stdout=bench.stdout[-3000:], stderr=bench.stderr[-2000:])
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

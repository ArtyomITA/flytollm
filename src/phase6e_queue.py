"""Phase 6e: reruns owed after the 13:19 RAM-guard incident (two smokes launched in parallel with the queue): the permuted-
weights run, the reversal smoke and run, and the graded/tau re-smokes with the corrected FusedVariantCore. One GPU worker."""
import ctypes, json
import pretrain_night_queue as q
from bench_runtime import ROOT

STOP = ROOT / 'STOP_PHASE6'
COMMON = ['--threshold', 10, '--head', 'separate', '--seed', 17, '--eval-every', 500, '--checkpoint-every', 2000, '--fast-mode', 'fused']
JOBS = [
    ('phase6e_resmoke_core_graded_ol', ['--rewire-kind', 'none', '--core-variant', 'graded_ol', '--updates', 4], 1800),
    ('phase6e_resmoke_core_tau_type', ['--rewire-kind', 'none', '--core-variant', 'tau_type', '--updates', 4], 1800),
    ('phase6e_smoke_core_reversal', ['--rewire-kind', 'none', '--core-variant', 'reversal', '--updates', 4], 1800),
    ('phase6c_permweights_2000', ['--rewire-kind', 'weights', '--rewire-seed', 41, '--updates', 2000], 7200),
    ('phase6c_core_reversal_2000', ['--rewire-kind', 'none', '--core-variant', 'reversal', '--updates', 2000], 7200),
]


def main():
    q.STATE = ROOT / 'results/phase6e_live.json'; q.STOP = STOP
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md')
    q.write_state(state)
    keep = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for name, extra, timeout in JOBS:
            out = ROOT / 'results' / f'{name}.json'
            if out.exists():
                data = json.loads(out.read_text())
                if data.get('ok'):
                    continue
                for suffix in ('.json', '.worker.json', '.console.log', '.jsonl'):
                    p = ROOT / 'results' / f'{name}{suffix}'
                    if p.exists():
                        p.replace(ROOT / 'results' / f'{name}.failed_attempt{suffix}')
            try:
                q.run_job(state, name, 'pretrain_control', COMMON + extra, timeout)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
        state.update(status='completed'); q.write_state(state)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

"""Phase 6f: promotion to 8000 updates (SUITE_TEST_FASE_6B.md, regola di promozione). Every finished 2000-update control run
whose DEV CE beats the real fly (5.087) by at least 0.10 nat is re-run from scratch to 8000 updates with the same
configuration and the fused core (a fresh run rather than a resume: variant cores carry extra buffers/parameters that
pretrain_resumable.load_payload cannot rebuild). Compared afterwards with real 3.959 and rewired 3.814 at 8000."""
import ctypes, json, glob
import pretrain_night_queue as q
from bench_runtime import ROOT

REAL_2000 = 5.087
STOP = ROOT / 'STOP_PHASE6'
COMMON = ['--threshold', 10, '--head', 'separate', '--seed', 17, '--eval-every', 500, '--checkpoint-every', 2000, '--fast-mode', 'fused', '--updates', 8000]


def candidates():
    out = []
    for path in sorted(glob.glob(str(ROOT / 'results' / 'phase6*_2000.json'))):
        name = __import__('os').path.basename(path)[:-5]
        if name.startswith('phase6s_') or name in ('phase6_rewired_h10_2000', 'phase6b_rewired_s43_2000'):
            continue  # rewired already at 8000 (3,814); seed 43 is a replicate of it
        data = json.loads(open(path).read())
        if not data.get('ok') or not isinstance(data.get('result'), dict):
            continue
        r = data['result']
        if r.get('updates') != 2000 or not r.get('curve'):
            continue
        ce = r['curve'][-1]['dev']['ce']
        if REAL_2000 - ce < 0.10:
            continue
        c = r.get('control', {})
        args = ['--rewire-kind', c.get('rewire_kind', 'none'), '--rewire-seed', c.get('rewire_seed', 41),
                '--ports', c.get('ports', 'anatomical'), '--readout', c.get('readout', 'chunks'), '--core-variant', c.get('core_variant', 'lif')]
        out.append((name.replace('_2000', '_8000'), ce, args))
    return out


def main():
    q.STATE = ROOT / 'results/phase6f_live.json'; q.STOP = STOP
    cands = candidates()
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md', candidates=[dict(name=n, ce_2000=ce) for n, ce, _ in cands])
    q.write_state(state)
    keep = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for name, ce, args in sorted(cands, key=lambda x: x[1]):
            if (ROOT / 'results' / f'{name}.json').exists():
                continue
            try:
                q.run_job(state, name, 'pretrain_control', COMMON + args, 14400)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
        state.update(status='completed'); q.write_state(state)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    print(candidates()) if __import__('sys').argv[1:] == ['--list'] else main()

"""Phase 6s speed suite (SUITE_TEST_FASE_6B.md, category S). Each configuration is a real 100-update training run through
the identical pipeline (pretrain_control -> pretrain_resumable.run, FairCapture CUDA Graph), so every run reports
mean_step_s, the eager/graph equivalence, the training loss every 16 updates and DEV CE at 100. Equivalence between
configurations = final-weight and loss-trajectory distance against the reference S0, compared with the noise floor
S0a vs S0b (two identical reference runs differ only by atomic-order noise)."""
import argparse, ctypes, json, time
import torch
import pretrain_night_queue as q
from bench_runtime import ROOT

STOP = ROOT / 'STOP_PHASE6S'
COMMON = ['--threshold', 10, '--head', 'separate', '--seed', 17, '--updates', 100, '--eval-every', 100, '--checkpoint-every', 100, '--rewire-kind', 'none']
CONFIGS = [
    ('phase6s_S0a_reference', []),
    ('phase6s_S0b_reference', []),
    ('phase6s_S1_chunk1M', ['--fast-mode', 'gather', '--chunk', 1048576]),
    ('phase6s_S1_chunk_all', ['--fast-mode', 'gather', '--chunk', 2753975]),
    ('phase6s_S2_int32', ['--fast-mode', 'gather', '--chunk', 1048576, '--index-dtype', 'int32']),
    ('phase6s_S3_csr', ['--fast-mode', 'csr']),
    ('phase6s_S5_fp16', ['--fast-mode', 'fp16', '--chunk', 1048576]),
    ('phase6s_S7_adam_fused', ['--adam', 'fused']),
    ('phase6s_S7_adam_foreach', ['--adam', 'foreach']),
    ('phase6s_S4_reorder', ['--rewire-kind', 'reorder', '--rewire-seed', 41]),
    ('phase6s_S4_reorder_chunk1M', ['--rewire-kind', 'reorder', '--rewire-seed', 41, '--fast-mode', 'gather', '--chunk', 1048576]),
    ('phase6s_S6_fused', ['--fast-mode', 'fused']),
    ('phase6s_S6_fused_reorder', ['--rewire-kind', 'reorder', '--rewire-seed', 41, '--fast-mode', 'fused']),
]


def compare(reference, name):
    ref = json.loads((ROOT / 'results' / f'{reference}.json').read_text())['result']
    cur = json.loads((ROOT / 'results' / f'{name}.json').read_text())['result']
    out = dict(name=name, mean_step_s=cur['mean_step_s'], speedup=ref['mean_step_s'] / cur['mean_step_s'],
               dev_ce=cur['curve'][-1]['dev']['ce'], dev_ce_ref=ref['curve'][-1]['dev']['ce'],
               equivalence=cur['equivalence'], peak_vram_mb=cur['peak_vram_mb'])
    lr = {t['update']: t['loss'] for t in ref['telemetry']}; lc = {t['update']: t['loss'] for t in cur['telemetry']}
    out['loss_max_abs_diff'] = max(abs(lr[k] - lc[k]) for k in lr if k in lc)
    a = torch.load(ROOT / ref['checkpoint'], map_location='cpu', weights_only=True)['model']['state_dict']
    b = torch.load(ROOT / cur['checkpoint'], map_location='cpu', weights_only=True)['model']['state_dict']
    diffs = {}
    for k, v in a.items():
        if k in b and v.dtype.is_floating_point and v.shape == b[k].shape:
            diffs[k] = float((v - b[k]).abs().max())
    out['weight_max_abs_diff'] = max(diffs.values()); out['weight_diff_core_raw'] = diffs.get('core.raw')
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--only', nargs='*'); p.add_argument('--compare-only', action='store_true'); a = p.parse_args()
    q.STATE = ROOT / 'results/phase6s_live.json'; q.STOP = STOP
    if not a.compare_only:
        state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#S')
        q.write_state(state)
        keep = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            for name, extra in CONFIGS:
                if a.only and name not in a.only:
                    continue
                if (ROOT / 'results' / f'{name}.json').exists():
                    continue
                try:
                    q.run_job(state, name, 'pretrain_control', COMMON + extra, 1800)
                except RuntimeError as exc:
                    state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
            state.update(status='completed'); q.write_state(state)
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    rows = []
    for name, _ in CONFIGS:
        if name == 'phase6s_S0a_reference' or not (ROOT / 'results' / f'{name}.json').exists():
            continue
        try:
            rows.append(compare('phase6s_S0a_reference', name))
        except Exception as exc:
            rows.append(dict(name=name, error=str(exc)))
    (ROOT / 'results/phase6s_speed_summary.json').write_text(json.dumps(rows, indent=2))
    for r in rows:
        if 'error' in r:
            print(r['name'], 'ERROR', r['error'][:200]); continue
        print(f"{r['name']:34s} {r['mean_step_s']*1000:7.1f} ms/update  x{r['speedup']:.2f}  loss maxdiff {r['loss_max_abs_diff']:.2e}  weights maxdiff {r['weight_max_abs_diff']:.2e} (core {r['weight_diff_core_raw']:.2e})  devCE {r['dev_ce']:.4f} (ref {r['dev_ce_ref']:.4f})  vram {r['peak_vram_mb']:.0f}")


if __name__ == '__main__':
    main()

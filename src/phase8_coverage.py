"""How much of the brain is learning (phase 8, user concern of 17 September 2026: "se lavoriamo su un campione di pochi
neuroni non va bene"). CPU only. For every run name given: DEV curve, last training gradient norm and sampled spike rate
(from the .jsonl), and from the .latest.pt checkpoint against the initial weights of the same graph: relative change of the
synaptic weights, fraction of synapses moved by more than 1e-4, largest change, and, when present, the statistics of the
per-type threshold offsets (homeo), leaks (tau_type) and group gains.
Run: .venv/Scripts/python.exe phase8_coverage.py run_name [run_name ...] [--output results/phase8_coverage_<tag>.json]"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from bench_runtime import ROOT
from pretrain_control import load_control_graph

_init = {}


def init_weights(kind, seed, scale):
    key = (kind, seed, scale)
    if key not in _init:
        data, _ = load_control_graph(10, seed, kind)
        n = len(data['body_ids']); counts = np.log1p(data['weight']); incoming = np.bincount(data['dst'], weights=counts, minlength=n)
        _init[key] = torch.from_numpy((scale * .5 * counts / np.maximum(incoming[data['dst']], 1)).astype(np.float32))
    return _init[key]


def describe(name):
    R = ROOT / 'results'; out = dict(run=name)
    result = R / f'{name}.json'
    control = {}
    if result.exists():
        d = json.loads(result.read_text())
        if d.get('ok'):
            r = d['result']; control = r.get('control', {})
            out['curve'] = {str(e['update']): round(e['dev']['ce'], 4) for e in r['curve']}
            out['ms_per_update'] = round(r['mean_step_s'] * 1000)
    log = R / f'{name}.jsonl'
    if log.exists():
        rows = [json.loads(l) for l in log.open(encoding='utf-8', errors='replace') if l.strip().startswith('{')]
        train = [x for x in rows if x.get('phase') == 'training' and 'grad_norm' in x]
        if train:
            tail = train[-max(1, len(train) // 5):]
            out['grad_norm_last_fifth_median'] = float(np.median([x['grad_norm'] for x in tail]))
            out['spike_sample_mean_last'] = train[-1].get('spike_sample_mean')
    ckpt = R / f'{name}.latest.pt'
    if ckpt.exists():
        sd = torch.load(ckpt, map_location='cpu', weights_only=True)['model']['state_dict']
        kind = control.get('rewire_kind', 'relthr'); seed = int(control.get('rewire_seed', 41)); scale = float(control.get('weight_scale', 1.0))
        w0 = init_weights(kind, seed, scale)
        w = torch.nn.functional.softplus(sd['core.raw'].float())
        if w.shape == w0.shape:
            d = w - w0
            out.update(weights_relative_change=float(d.norm() / w0.norm()), synapses_moved_fraction=float((d.abs() > 1e-4).float().mean()),
                       synapses_moved=int((d.abs() > 1e-4).sum()), largest_change=float(d.abs().max()), mean_weight=float(w.mean()), mean_weight_init=float(w0.mean()))
        for key, label, f in (('core.thr_offset', 'threshold_offset', lambda x: x), ('core.leak_logit', 'leak', torch.sigmoid), ('core.group_gain', 'group_gain', lambda x: x)):
            if key in sd:
                x = f(sd[key].float())
                out[label] = dict(min=float(x.min()), median=float(x.median()), max=float(x.max()), mean=float(x.mean()),
                                  above_0_1=int((x > .1).sum()) if label == 'threshold_offset' else None, n=int(x.numel()))
        for key in ('calls', 'core.tokens'):
            if key in sd:
                out[key] = int(sd[key])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+'); p.add_argument('--output', default=None)
    a = p.parse_args()
    table = [describe(name) for name in a.runs]
    for row in table:
        curve = ' '.join(f'{k}:{v:.3f}' for k, v in row.get('curve', {}).items())
        print(f"{row['run']:44s} {curve} | dw {row.get('weights_relative_change', float('nan')) * 100:.3f}% moved {row.get('synapses_moved_fraction', float('nan')) * 100:.2f}% "
              f"max {row.get('largest_change', float('nan')):.4f} | grad {row.get('grad_norm_last_fifth_median', float('nan')):.2f} spikes {row.get('spike_sample_mean_last')}")
        for label in ('threshold_offset', 'leak', 'group_gain'):
            if label in row:
                print('   ', label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row[label].items()})
    if a.output:
        Path(a.output).write_text(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()

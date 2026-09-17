"""Collect the phase-6c inference results (results/phase6c_<checkpoint>.json) into one table for SUITE_TEST_FASE_6B.md."""
import json
from bench_runtime import ROOT

NAMES = ['fly_real_2000', 'rewired_2000', 'configmodel_2000', 'fly_real_8000', 'rewired_8000', 'fly_real_26247']


def load(name):
    p = ROOT / 'results' / f'phase6c_{name}.json'
    return json.loads(p.read_text())['tests'] if p.exists() else None


def main():
    data = {n: load(n) for n in NAMES}
    out = ['## Riepilogo inferenza fase 6c (generato da phase6c_summary.py)', '']
    out.append('### A1 Profilo di contesto: CE con KV cache limitata a k token, gate a zero, PAPA')
    out.append('| Checkpoint | k=128 | 64 | 32 | 8 | 1 | 0 | gate 0 | PAPA | Δ(0−128) |'); out.append('|---|---|---|---|---|---|---|---|---|---|')
    for n, t in data.items():
        if not t or 'context' not in t:
            continue
        c = t['context']; row = [c[f'kv_{k}']['ce'] for k in (128, 64, 32, 8, 1, 0)]
        out.append(f"| {n} | " + ' | '.join(f'{v:.3f}' for v in row) + f" | {c['gate_0']['ce']:.3f} | {c['papa_mean']['ce']:.3f} | {row[-1] - row[0]:+.3f} |")
    out.append(''); out.append('### A5 Loss per posizione nella finestra di 16 (1 … 16) e indice d\'uso del contesto loss(1) − loss(16)')
    for n, t in data.items():
        if t and 'context' in t:
            pw = t['context']['baseline']['per_window']
            out.append(f"- {n}: " + ' '.join(f'{v:.2f}' for v in pw) + f" · indice {pw[0] - pw[-1]:+.3f}")
    out.append(''); out.append('### A2 / C7 / C9 Attività: gate, rapporto correnti, frazione attiva, Gini, KC')
    out.append('| Checkpoint | gate | feedback/token | frazione attiva per token | mai attivi | Gini | KC attive | sonda KC lag0 | ρ trained | persist. 8/16 |'); out.append('|---|---|---|---|---|---|---|---|---|---|')
    for n, t in data.items():
        if not t or 'activity' not in t:
            continue
        a = t['activity']; kc = t.get('kc_probe', {}); sp = t.get('spectral', {}); pe = t.get('persistence', {})
        out.append(f"| {n} | {a['gate']:.3f} | {a['feedback_to_token_current_ratio_mean'] if a['feedback_to_token_current_ratio_mean'] is None else round(a['feedback_to_token_current_ratio_mean'], 3)} | {a['active_fraction_per_token_mean']:.4f} | {a['fraction_never_active']:.3f} | {a['gini_mean_rate']:.3f} | {a['kc_active_fraction_mean']:.4f} | {kc.get('accuracy_lag0', float('nan')):.3f} | {sp.get('spectral_radius_trained', float('nan')):.3f} | {pe.get('corr_8', float('nan')):.2f}/{pe.get('corr_16', float('nan')):.2f} |")
    out.append(''); out.append('### C10 Sonda posizionale (R² cross-fit): complesso centrale contro popolazione random')
    for n, t in data.items():
        if t and 'activity' in t:
            a = t['activity']; cx = a.get('cx_position_probe') or {}; rn = a.get('random_position_probe') or {}
            out.append(f"- {n}: CX ({a.get('cx_nodes')} nodi) R² {cx.get('r2_crossfit', float('nan')):.3f} · random R² {rn.get('r2_crossfit', float('nan')):.3f}")
    out.append(''); out.append('### C2 / C6 Popolazioni: tasso medio, selettività, spente (superclasse)')
    for n, t in data.items():
        if not t or 'activity' not in t:
            continue
        pops = t['activity']['populations']['superclass']; g = t['activity']['global_mean_rate']
        silent = [k for k, v in pops.items() if v['silent']]
        top = sorted(pops.items(), key=lambda kv: -kv[1]['selectivity'])[:6]
        out.append(f"- {n}: tasso globale {g:.4f}; spente: {', '.join(silent) or 'nessuna'}; più selettive: " + ', '.join(f"{k} ({v['selectivity']:.2f})" for k, v in top))
    out.append(''); out.append('### C3 / C4 / C5 Lesioni: ΔCE lesione, ΔCE controllo random a pari grado, eccesso')
    keys = None
    for n, t in data.items():
        if t and 'lesions' in t:
            keys = list(t['lesions']['lesions'].keys()); break
    if keys:
        out.append('| Popolazione (nodi) | ' + ' | '.join(n for n, t in data.items() if t and 'lesions' in t) + ' |')
        out.append('|---|' + '---|' * sum(1 for t in data.values() if t and 'lesions' in t))
        for k in keys:
            cells = []
            for n, t in data.items():
                if t and 'lesions' in t:
                    l = t['lesions']['lesions'].get(k)
                    cells.append(f"{l['delta']:+.3f} / {l['delta_control']:+.3f} / {l['excess']:+.3f}" if l else '—')
            nodes = next((t['lesions']['lesions'][k]['nodes'] for t in data.values() if t and 'lesions' in t and k in t['lesions']['lesions']), '?')
            out.append(f'| {k} ({nodes}) | ' + ' | '.join(cells) + ' |')
    out.append(''); out.append('### C8 Robustezza al rumore in ingresso: ΔCE con 5% / 10% / 20% di token sostituiti')
    for n, t in data.items():
        if t and 'noise' in t:
            z = t['noise']; out.append(f"- {n}: {z['noise_0.05']:+.3f} / {z['noise_0.1']:+.3f} / {z['noise_0.2']:+.3f}")
    text = '\n'.join(out)
    (ROOT / 'results/phase6c_summary.md').write_text(text, encoding='utf8')
    print(text)


if __name__ == '__main__':
    main()

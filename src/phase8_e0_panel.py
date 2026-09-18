"""E0 diagnostic panel (RICERCA_CORE_NON_IMPARA.md; user ok 18 September 2026 02:05): which of the five causes of the
non-learning core dominates. Inference plus a few gradient samples on one plain-LIF control checkpoint, no training.
  1. excitability (cause 3, quiescent init): per neuron over DEV, pre-reset potential u at every substep: mean, std
     (sigma_U), distance to threshold xi = (1 - mean) / std, occupancy of the surrogate window |u - 1| < 0.5, spike rate;
     Rossbroich-Gygax-Zenke 2022 fluctuation window: sigma_U in [1/3, 1], xi in [1, 3]
  2. effective step (cause 4, softplus): dw/draw = sigmoid(raw) over the synapses, the factor cutting every optimizer step
  3. gradient signal-to-noise (cause 5): gradient of the loss on core.raw over K samples of 2 lanes x 16 tokens (as in
     training): per synapse |mean| / std, share of synapses with SNR > 1 (pure noise gives ~1/sqrt(K)), with a nonzero
     gradient in at least one sample, in every sample
  4. where the weights moved (causes 1-2, lazy regime / starvation): Spearman correlation of |delta w| (vs init) with the
     source rate, the target rate, |mean gradient| and w0; share of the total |delta w| carried by the synapses whose
     source is among the 10% most active neurons; synapses moved (> 1e-4) among those with a silent source
Run (GPU idle, ~1 min): .venv/Scripts/python.exe phase8_e0_panel.py --checkpoint-path results/<run>.latest.pt --output results/phase8_e0_<tag>.json
With --run <name> the model is rebuilt through phase8_load_variant (any variant, init_norm / weight_scale read from the run's control record)."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from phase6c_inference import Suite
from pretrain_control import load_control_graph
from fly_core import Propagate, LIFReset


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum() + 1e-30))


def quantiles(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, np.float64)
    return {str(q): float(np.quantile(x, q)) for q in qs} if x.size else {}


@torch.no_grad()
def excitability_pass(suite, stats):
    """DEV pass with the core substeps re-implemented (same arithmetic as Core.advance) to record u per neuron."""
    model = suite.model; core = model.core; original = core.advance

    def advance(current, steps=1, state=None, *, weights=None, active=None, reset=None, reference=False):
        v, s = core.reset_state(state, reset) if state is not None else core.initial_state(current.shape[0])
        w = core.weights() if weights is None else weights
        rate = torch.zeros_like(current)
        lanes = active if active is not None else torch.ones(current.shape[0], dtype=torch.bool, device=current.device)
        for _ in range(steps):
            syn = Propagate.apply(s, w, core.src, core.dst, core.chunk)
            u = .95 * v + syn + current
            um = u[lanes]
            stats['sum'] += um.sum(0); stats['sq'] += (um * um).sum(0); stats['near'] += ((um - 1).abs() < .5).float().sum(0)
            stats['count'] += float(lanes.sum())
            next_v, next_s = LIFReset.apply(u)
            stats['spikes'] += next_s[lanes].sum(0)
            if active is not None:
                mask = active[:, None]
                next_v = torch.where(mask, next_v, v); next_s = torch.where(mask, next_s, s)
                rate = rate + torch.where(mask, next_s, torch.zeros_like(next_s))
            else:
                rate = rate + next_s
            v, s = next_v, next_s
        return (v, s), rate / steps

    core.advance = advance; loss = 0.; count = 0
    try:
        state = model.initial_state(16)
        for x, y in suite.windows:
            for pos in range(16):
                logits, state = model.step(x[pos], state)
                valid = (x[pos] != 0) & (y[pos] != 0)
                losses = torch.nn.functional.cross_entropy(logits, torch.where(valid, y[pos], torch.zeros_like(y[pos])), reduction='none')
                loss += float((losses * valid).sum()); count += int(valid.sum())
    finally:
        core.advance = original
    return loss / max(count, 1)


def gradient_samples(suite, k):
    model = suite.model; core = model.core
    g_sum = torch.zeros_like(core.raw); g_sq = torch.zeros_like(core.raw); nonzero = torch.zeros_like(core.raw)
    windows = suite.windows
    for i in range(k):
        x, y = windows[i % len(windows)]; lane = (i // len(windows)) * 2 % 16
        ids, targets = x[:, lane:lane + 2], y[:, lane:lane + 2]
        model.zero_grad(set_to_none=True)
        logits, _ = model(ids); loss, _ = model.loss(logits, targets, ids); loss.backward()
        g = core.raw.grad.detach()
        g_sum += g; g_sq += g * g; nonzero += (g != 0).float()
    model.zero_grad(set_to_none=True)
    mean = g_sum / k; std = (g_sq / k - mean * mean).clamp_min(0).sqrt()
    return mean, std, nonzero


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--kind', default='relthr')
    p.add_argument('--rewire-seed', type=int, default=41)
    p.add_argument('--weight-scale', type=float, default=1.0)
    p.add_argument('--grad-samples', type=int, default=16)
    p.add_argument('--run', default=None, help='result name: load through phase8_load_variant.VariantSuite and take init_norm / weight_scale / graph from its control record')
    a = p.parse_args(); started = time.time()
    path = Path(a.checkpoint_path)
    init_norm = 'sum'
    if a.run:
        from phase8_load_variant import VariantSuite
        suite = VariantSuite(a.run, a.checkpoint_path); c = suite.control
        a.kind, a.rewire_seed, a.weight_scale, init_norm = c['rewire_kind'], c['rewire_seed'], float(c.get('weight_scale', 1.0)), c.get('init_norm', 'sum')
    else:
        suite = Suite(path.stem, path)
    model = suite.model; core = model.core; n = core.n
    if getattr(core, 'flags', None):
        raise SystemExit(f'E0 re-implements the plain LIF substeps; core variant {sorted(core.flags)} would be measured with the wrong dynamics')
    data, _ = load_control_graph(10, a.rewire_seed, a.kind)
    assert np.array_equal(core.src.cpu().numpy(), data['src']) and np.array_equal(core.dst.cpu().numpy(), data['dst']), 'checkpoint graph differs'
    counts = np.log1p(data['weight']); incoming = np.bincount(data['dst'], weights=counts, minlength=n)
    if init_norm == 'fluct':  # E1: same formula as pretrain_control.build_control_cns
        incoming_sq = np.bincount(data['dst'], weights=counts ** 2, minlength=n)
        magnitude = torch.from_numpy((a.weight_scale * counts / np.sqrt(np.maximum(incoming_sq[data['dst']], 1e-12))).astype(np.float32)).cuda()
    else:
        magnitude = torch.from_numpy((a.weight_scale * .5 * counts / np.maximum(incoming[data['dst']], 1)).astype(np.float32)).cuda()
    raw0 = magnitude + torch.log(-torch.expm1(-magnitude))
    out = dict(checkpoint=str(path), run=a.run, init_norm=init_norm, weight_scale=a.weight_scale, depth=[model.config.pre_steps, model.config.post_steps], neurons=n, synapses=int(core.raw.numel()))

    # 1. excitability
    stats = {key: torch.zeros(n, device='cuda') for key in ('sum', 'sq', 'near', 'spikes')}; stats['count'] = 0.
    out['dev_ce'] = excitability_pass(suite, stats)
    c = stats['count']; mean_u = stats['sum'] / c; std_u = (stats['sq'] / c - mean_u * mean_u).clamp_min(0).sqrt()
    near = stats['near'] / c; rate = stats['spikes'] / c
    xi = ((1 - mean_u) / std_u.clamp_min(1e-6)).cpu().numpy(); std_np = std_u.cpu().numpy(); mean_np = mean_u.cpu().numpy()
    rate_np = rate.cpu().numpy(); near_np = near.cpu().numpy()
    live = std_np > 1e-4
    out['excitability'] = dict(
        substeps_recorded=c, neurons_with_any_fluctuation=float(live.mean()), mute_neurons=float((rate_np == 0).mean()),
        mean_u=quantiles(mean_np[live]), sigma_u=quantiles(std_np[live]), xi=quantiles(xi[live]),
        sigma_in_window_1_3_to_1=float(((std_np >= 1 / 3) & (std_np <= 1)).mean()), sigma_below_0_1=float((std_np < .1).mean()),
        xi_in_window_1_to_3=float(((xi >= 1) & (xi <= 3) & live).mean()), xi_above_3=float(((xi > 3) & live).mean()),
        xi_below_1=float(((xi < 1) & live).mean()),
        surrogate_window_occupancy=quantiles(near_np), neurons_never_in_window=float((near_np == 0).mean()),
        rate=quantiles(rate_np[rate_np > 0]))
    e = out['excitability']
    print(f'{path.stem}: DEV CE {out["dev_ce"]:.4f}; neurons fluctuating {100 * e["neurons_with_any_fluctuation"]:.1f}%, mute {100 * e["mute_neurons"]:.1f}%; '
          f'sigma_U median {e["sigma_u"].get("0.5", float("nan")):.3f} (in [1/3,1]: {100 * e["sigma_in_window_1_3_to_1"]:.1f}%, < 0.1: {100 * e["sigma_below_0_1"]:.1f}%); '
          f'xi median {e["xi"].get("0.5", float("nan")):.2f} (in [1,3]: {100 * e["xi_in_window_1_to_3"]:.1f}%, > 3: {100 * e["xi_above_3"]:.1f}%); '
          f'never in surrogate window {100 * e["neurons_never_in_window"]:.1f}%', flush=True)

    # 2. effective step
    sig = torch.sigmoid(core.raw.detach()).cpu().numpy()
    out['effective_step'] = dict(sigmoid_raw=quantiles(sig), below_0_05=float((sig < .05).mean()), mean=float(sig.mean()))
    print(f'effective step dw/draw: median {out["effective_step"]["sigmoid_raw"]["0.5"]:.4f}, mean {out["effective_step"]["mean"]:.4f}, '
          f'< 0.05: {100 * out["effective_step"]["below_0_05"]:.1f}% (cut factor ~{1 / max(out["effective_step"]["mean"], 1e-9):.0f}x)', flush=True)

    # 3. gradient SNR
    k = a.grad_samples
    mean_g, std_g, nonzero = gradient_samples(suite, k)
    any_nz = nonzero > 0; all_nz = nonzero == k
    snr = (mean_g.abs() / std_g.clamp_min(1e-30))[any_nz].cpu().numpy()
    out['gradient'] = dict(samples=k, synapses_any_nonzero=float(any_nz.float().mean()), synapses_all_nonzero=float(all_nz.float().mean()),
                          snr=quantiles(snr), snr_above_1=float((snr > 1).mean()), snr_above_2=float((snr > 2).mean()),
                          noise_reference=1 / k ** .5, abs_mean_gradient=quantiles(mean_g.abs()[any_nz].cpu().numpy()))
    g = out['gradient']
    print(f'gradient over {k} samples: synapses with any gradient {100 * g["synapses_any_nonzero"]:.1f}%, in every sample {100 * g["synapses_all_nonzero"]:.1f}%; '
          f'SNR median {g["snr"].get("0.5", float("nan")):.3f} (noise ~{g["noise_reference"]:.2f}), SNR > 1: {100 * g["snr_above_1"]:.1f}%, > 2: {100 * g["snr_above_2"]:.1f}%', flush=True)

    # 4. where the weights moved
    w_t = torch.nn.functional.softplus(core.raw.detach()); w_0 = torch.nn.functional.softplus(raw0)
    dw = (w_t - w_0).abs().cpu().numpy(); w0 = w_0.cpu().numpy()
    src = data['src']; dst = data['dst']
    src_rate = rate_np[src]; dst_rate = rate_np[dst]; gm = mean_g.abs().cpu().numpy()
    top = rate_np >= np.quantile(rate_np, .9)
    silent_src = src_rate == 0
    out['movement'] = dict(
        relative_movement=float(np.linalg.norm(w_t.cpu().numpy() - w0) / np.linalg.norm(w0)), moved_above_1e4=float((dw > 1e-4).mean()),
        spearman_dw_source_rate=spearman(dw, src_rate), spearman_dw_target_rate=spearman(dw, dst_rate),
        spearman_dw_abs_mean_gradient=spearman(dw, gm), spearman_dw_w0=spearman(dw, w0),
        share_of_dw_from_top10pct_sources=float(dw[top[src]].sum() / max(dw.sum(), 1e-30)), synapses_from_top10pct_sources=float(top[src].mean()),
        moved_with_silent_source=float((dw[silent_src] > 1e-4).mean()) if silent_src.any() else None, synapses_with_silent_source=float(silent_src.mean()),
        moved_with_active_source=float((dw[~silent_src] > 1e-4).mean()) if (~silent_src).any() else None)
    m = out['movement']
    print(f'weights: moved {100 * m["relative_movement"]:.2f}% of ||w0||, > 1e-4 on {100 * m["moved_above_1e4"]:.1f}%; Spearman |dw| vs source rate {m["spearman_dw_source_rate"]:.3f}, '
          f'target rate {m["spearman_dw_target_rate"]:.3f}, |mean grad| {m["spearman_dw_abs_mean_gradient"]:.3f}, w0 {m["spearman_dw_w0"]:.3f}; '
          f'top-10% sources ({100 * m["synapses_from_top10pct_sources"]:.1f}% of synapses) carry {100 * m["share_of_dw_from_top10pct_sources"]:.1f}% of |dw|; '
          f'silent-source synapses {100 * m["synapses_with_silent_source"]:.1f}%, moved {100 * (m["moved_with_silent_source"] or 0):.2f}% vs active-source {100 * (m["moved_with_active_source"] or 0):.2f}%', flush=True)
    out['seconds'] = time.time() - started
    Path(a.output).write_text(json.dumps(out, indent=2))
    print(f'done in {out["seconds"] / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()

"""C15 (SUITE_TEST_FASE_6B.md): how much of the result is carried by the TRAINED synapses, and how wide the learning is.
Inference only, on a plain-LIF control checkpoint of the standard configuration:
  1. DEV cross-entropy of the checkpoint as trained
  2. the same with the 2.75 M synaptic weights put back to their initial values (everything else as trained):
     the difference is the work done by synaptic learning
  3. the same with ONLY the synapses that moved by more than 1e-4 put back, and with only the others put back
  4. coverage: neurons active on DEV, synapses from active sources, synapses moved, per superclass
Run (GPU idle): .venv/Scripts/python.exe phase8_c15_revert_weights.py --checkpoint-path results/<run>.latest.pt --kind relthr --output results/phase8_c15_<tag>.json"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from phase6c_inference import Suite
from pretrain_control import load_control_graph


@torch.no_grad()
def dev_ce(suite, record=None):
    model = suite.model; loss = 0.; count = 0
    state = model.initial_state(16)
    original = model.core.advance
    if record is not None:
        def advance(current, steps=1, state=None, **kw):
            out = original(current, steps, state, **kw); record.add_(out[1].sum(0) * steps); return out
        model.core.advance = advance
    try:
        for x, y in suite.windows:
            for pos in range(16):
                logits, state = model.step(x[pos], state)
                valid = (x[pos] != 0) & (y[pos] != 0)
                losses = torch.nn.functional.cross_entropy(logits, torch.where(valid, y[pos], torch.zeros_like(y[pos])), reduction='none')
                loss += float((losses * valid).sum()); count += int(valid.sum())
    finally:
        model.core.advance = original
    return loss / max(count, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-path', required=True)
    p.add_argument('--kind', default='relthr')
    p.add_argument('--rewire-seed', type=int, default=41)
    p.add_argument('--weight-scale', type=float, default=1.0)
    p.add_argument('--output', required=True)
    p.add_argument('--run', default=None, help='result name: load any control variant through phase8_load_variant.VariantSuite (graph kind, seed and weight scale from its control record)')
    a = p.parse_args(); started = time.time()
    path = Path(a.checkpoint_path)
    if a.run:
        from phase8_load_variant import VariantSuite
        suite = VariantSuite(a.run, a.checkpoint_path); c = suite.control
        assert c.get('init_norm', 'sum') == 'sum', 'fluctuation init: use the E0 panel formula'
        a.kind, a.rewire_seed, a.weight_scale = c['rewire_kind'], c['rewire_seed'], float(c.get('weight_scale', 1.0))
    else:
        suite = Suite(path.stem, path)
    core = suite.model.core
    data, _ = load_control_graph(10, a.rewire_seed, a.kind)
    assert np.array_equal(core.src.cpu().numpy(), data['src']) and np.array_equal(core.dst.cpu().numpy(), data['dst']), 'checkpoint graph differs'
    n = core.n; counts = np.log1p(data['weight']); incoming = np.bincount(data['dst'], weights=counts, minlength=n)
    magnitude = torch.from_numpy((a.weight_scale * .5 * counts / np.maximum(incoming[data['dst']], 1)).astype(np.float32)).cuda()
    raw0 = magnitude + torch.log(-torch.expm1(-magnitude))
    trained = core.raw.data.clone()
    w_t = torch.nn.functional.softplus(trained); w_0 = torch.nn.functional.softplus(raw0)
    moved = (w_t - w_0).abs() > 1e-4
    rate = torch.zeros(n, device='cuda')
    out = dict(checkpoint=str(path), ce_trained=dev_ce(suite, rate))
    active = rate > 0
    core.raw.data.copy_(raw0); out['ce_synapses_at_init'] = dev_ce(suite)
    core.raw.data.copy_(torch.where(moved, raw0, trained)); out['ce_moved_synapses_at_init'] = dev_ce(suite)
    core.raw.data.copy_(torch.where(moved, trained, raw0)); out['ce_unmoved_synapses_at_init'] = dev_ce(suite)
    core.raw.data.copy_(trained)
    from_active = active[core.src]
    out.update(delta_synaptic_learning=out['ce_synapses_at_init'] - out['ce_trained'],
               neurons_active=int(active.sum()), neurons_active_fraction=float(active.float().mean()),
               synapses=int(moved.numel()), synapses_from_active=int(from_active.sum()), synapses_moved=int(moved.sum()),
               synapses_moved_fraction=float(moved.float().mean()), moved_from_active_fraction=float(moved[from_active].float().mean()),
               relative_change=float((w_t - w_0).norm() / w_0.norm()))
    superclass = suite.ann['superclass']; per = {}
    act = active.cpu().numpy(); src = core.src.cpu().numpy(); mv = moved.cpu().numpy()
    for name in sorted(set(superclass) - {''}):
        nodes = superclass == name
        edges = nodes[src]
        per[name] = dict(neurons=int(nodes.sum()), active_fraction=float(act[nodes].mean()), outgoing_synapses=int(edges.sum()),
                         moved_fraction=float(mv[edges].mean()) if edges.any() else 0.)
    out['per_superclass'] = per; out['seconds'] = time.time() - started
    Path(a.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_superclass'}, indent=1))


if __name__ == '__main__':
    main()

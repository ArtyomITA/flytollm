"""CPU-only analyses of phase 8 (category C shorts, PIANO_TEST_RIMASTI.md), no GPU, a few minutes:
  C12  spectral radius of the effective signed weight matrix AT INIT (power iteration, as phase6c_inference.spectral does
       on trained checkpoints), for the graphs used by the suite: real threshold 10, relative threshold (standard),
       degree-preserving rewire, configuration model
  D2b  signed reachability ports -> readout at init: a unit drive on the port set is propagated linearly k hops through
       the signed init weights; for each readout population: net signed drive, absolute drive, coherence = |net| / abs,
       fraction of readout nodes reached. Pairs: visual / random ports x fru / hub / descending-chunk readouts (F1, F3,
       F4, F7 of the suite)
Output: results/phase8_cpu_analyses.json"""
import json, time
import numpy as np, torch
from bench_runtime import ROOT
from pretrain_control import load_control_graph, variant_ports, variant_readout, ANNOTATIONS
from fly_interfaces import anatomical_ports


def init_weights(data):
    n = len(data['body_ids']); counts = np.log1p(data['weight'])
    incoming = np.bincount(data['dst'], weights=counts, minlength=n)
    magnitude = (.5 * counts / np.maximum(incoming[data['dst']], 1)).astype(np.float32)
    sign = np.asarray(data['sign'], dtype=np.float32)[data['src']]
    return torch.from_numpy(data['src'].astype(np.int64)), torch.from_numpy(data['dst'].astype(np.int64)), torch.from_numpy(magnitude * sign), n


def spectral_radius(src, dst, w, n, iterations=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g); v /= v.norm(); radius = None; history = []
    for _ in range(iterations):
        out = torch.zeros(n).index_add_(0, dst, w * v[src]); radius = float(out.norm()); v = out / max(radius, 1e-12); history.append(radius)
    return radius, history[-5:]


def reach(src, dst, w, n, ports, readout, hops=6):
    x = torch.zeros(n); x[ports] = 1. / len(ports)
    rows = []
    for k in range(1, hops + 1):
        x = torch.zeros(n).index_add_(0, dst, w * x[src])
        r = x[readout]
        net, absolute = float(r.sum()), float(r.abs().sum())
        rows.append(dict(hop=k, net=net, abs=absolute, coherence=abs(net) / absolute if absolute else 0.,
                         reached=float((r != 0).float().mean()), positive_fraction=float((r > 0).float().sum() / max(int((r != 0).sum()), 1))))
    return rows


def main():
    started = time.time(); out = dict(C12={}, D2b={})
    for kind in ('none', 'relthr', 'degree', 'config'):
        data, _ = load_control_graph(10, 41, kind)
        src, dst, w, n = init_weights(data)
        radius, tail = spectral_radius(src, dst, w, n)
        absolute, _ = spectral_radius(src, dst, w.abs(), n)
        out['C12'][kind] = dict(edges=int(len(w)), spectral_radius_init=radius, last_iterations=tail, spectral_radius_abs=absolute,
                                inhibitory_edge_fraction=float((w < 0).float().mean()))
        print('C12', kind, round(radius, 4), 'abs', round(absolute, 4), flush=True)
        if kind == 'none':
            body_ids = data['body_ids']
            sensory, read_nodes, groups, _ = anatomical_ports(body_ids, ANNOTATIONS)
            ports = {name: variant_ports(body_ids, sensory, read_nodes, name)[0] for name in ('visual', 'random_matched', 'auditory', 'anatomical')}
            readouts = dict(descending_chunks=read_nodes)
            for name in ('fru', 'hub'):
                readouts[name] = variant_readout(body_ids, data['src'], data['dst'], sensory, read_nodes, groups, name)[0]
            for pname, pnodes in ports.items():
                for rname, rnodes in readouts.items():
                    keep = rnodes[~torch.isin(rnodes, pnodes)]
                    out['D2b'][f'{pname}->{rname}'] = reach(src, dst, w, n, pnodes, keep)
                    best = max(out['D2b'][f'{pname}->{rname}'], key=lambda r: r['abs'])
                    print('D2b', pname, '->', rname, 'peak hop', best['hop'], 'net', f"{best['net']:+.2e}", 'coherence', round(best['coherence'], 3), 'reached', round(best['reached'], 3), flush=True)
    out['seconds'] = time.time() - started
    (ROOT / 'results/phase8_cpu_analyses.json').write_text(json.dumps(out, indent=2))
    print('done', round(out['seconds'], 1), 's')


if __name__ == '__main__':
    main()

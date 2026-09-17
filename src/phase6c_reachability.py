"""D2: hop distance from the sensory ports to every node and to the readout nodes, on the real graph and on the null
graphs (CPU, breadth-first on the directed edge list). Also A4/C8 from existing logs: local log-log slopes of the DEV
curves and the train/dev gap over the last 500 updates."""
import json
import numpy as np
from bench_runtime import ROOT
from fly_graph import load_graph
from fly_rewire import load_rewired_graph
from fly_interfaces import OUTPUT_CLASSES
import pyarrow.feather as pf


def layers_from(src, dst, n, seeds, max_hops=16):
    dist = np.full(n, -1, dtype=np.int64); dist[seeds] = 0
    front = np.zeros(n, dtype=bool); front[seeds] = True
    for hop in range(1, max_hops + 1):
        nodes = np.unique(dst[front[src]]); nodes = nodes[dist[nodes] < 0]
        if len(nodes) == 0:
            break
        dist[nodes] = hop; front.fill(False); front[nodes] = True
    return dist


def reachability(data, name):
    n = len(data['body_ids'])
    rows = pf.read_table(ROOT / 'dataset/male_cns/body-annotations-male-cns-v1.0.feather', columns=['bodyId', 'superclass']).to_pylist()
    lookup = {r['bodyId']: r['superclass'] for r in rows}
    read = np.array([i for i, b in enumerate(data['body_ids']) if lookup[int(b)] in OUTPUT_CLASSES])
    sensory = np.flatnonzero(data['sensory'])
    dist = layers_from(data['src'], data['dst'], n, sensory)
    reached = dist >= 0
    hist = np.bincount(dist[reached], minlength=17)[:17].tolist()
    out = dict(name=name, nodes=int(n), edges=int(len(data['src'])), sensory=int(len(sensory)), readout=int(len(read)),
               nodes_reached=int(reached.sum()), nodes_by_hop=hist,
               readout_reached=int(reached[read].sum()),
               readout_within_4=int(((dist[read] >= 0) & (dist[read] <= 4)).sum()),
               readout_within_8=int(((dist[read] >= 0) & (dist[read] <= 8)).sum()),
               readout_median_hops=float(np.median(dist[read][dist[read] >= 0])) if reached[read].any() else None,
               readout_mean_hops=float(np.mean(dist[read][dist[read] >= 0])) if reached[read].any() else None,
               nodes_median_hops=float(np.median(dist[reached])))
    # reverse: how many sensory nodes can reach any readout within 8 hops
    rdist = layers_from(data['dst'], data['src'], n, read)
    out['sensory_reaching_readout_within_8'] = int(((rdist[sensory] >= 0) & (rdist[sensory] <= 8)).sum())
    return out


def curve_slopes():
    def load(name):
        p = ROOT / 'results' / f'{name}.json'
        return json.loads(p.read_text()).get('result') if p.exists() else None
    out = {}
    for name in ['pretrain_night8000_h10', 'pretrain_continuous', 'phase6_rewired_h10_8000', 'phase6_gru_lr1e-3', 'phase6_transformer_lr1e-4', 'phase6_gru_lr1e-4']:
        r = load(name)
        if not r:
            continue
        curve = [(e['targets'], e['dev']['ce']) for e in r.get('curve', []) if e.get('targets', 0) > 0]
        slopes = [dict(from_targets=a[0], to_targets=b[0], slope=float((np.log(b[1]) - np.log(a[1])) / (np.log(b[0]) - np.log(a[0]))))
                  for a, b in zip(curve[:-1], curve[1:]) if b[0] > a[0]]
        tele = r.get('telemetry') or []
        last = [t for t in tele if t.get('update', 0) > r['updates'] - 500 and t.get('loss') is not None]
        train_tail = float(np.average([t['loss'] for t in last], weights=[t.get('count') or 1 for t in last])) if last else None
        final_ce = curve[-1][1] if curve else None
        out[name] = dict(updates=r['updates'], curve=curve, log_log_slopes=slopes, train_loss_last_500=train_tail, final_dev_ce=final_ce,
                         gap_train_minus_dev=(train_tail - final_ce) if (train_tail is not None and final_ce is not None) else None)
    return out


def main():
    results = dict(reachability=[], curves=curve_slopes())
    real = load_graph(10, progress=lambda m: None)
    results['reachability'].append(reachability(real, 'real'))
    for kind, seed in (('degree', 41), ('degree', 43), ('config', 41)):
        data, _ = load_rewired_graph(10, seed, kind=kind, progress=lambda m: None)
        results['reachability'].append(reachability(data, f'{kind}_s{seed}'))
    (ROOT / 'results/phase6c_reachability.json').write_text(json.dumps(results, indent=2))
    for r in results['reachability']:
        print(json.dumps(r))
    for k, v in results['curves'].items():
        print(k, 'slopes', [round(s['slope'], 3) for s in v['log_log_slopes']], 'gap', v['gap_train_minus_dev'])


if __name__ == '__main__':
    main()

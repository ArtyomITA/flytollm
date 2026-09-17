"""Category F follow-up (SUITE_TEST_FASE_6B.md): synaptic hop distance from every port set used in the suite to every
readout population, on the real threshold-10 graph (CPU breadth-first search on the directed edge list, no GPU).
Explains F1-F7: a readout that the ports reach in few hops learns, one that sits many hops away or behind silent
hubs does not. Writes results/phase6d_hops.json and prints a table."""
import json
import numpy as np
from bench_runtime import ROOT
from fly_graph import load_graph
from fly_interfaces import OUTPUT_CLASSES
from pretrain_control import node_column
from phase6c_reachability import layers_from

OUT = ROOT / 'results/phase6d_hops.json'


def port_sets(body_ids, sensory, read_nodes, n):
    sup = node_column(body_ids, 'superclass'); cls = node_column(body_ids, 'class'); sub = node_column(body_ids, 'subclass')
    is_sens = np.isin(np.arange(n), sensory)
    sets = {'anatomical': sensory}
    rng = np.random.default_rng(17)
    sets['random_matched'] = np.sort(rng.choice(np.setdiff1d(np.arange(n), read_nodes), len(sensory), replace=False))
    sets['olfactory'] = np.setdiff1d(np.flatnonzero(cls == 'olfactory'), read_nodes)
    sets['visual'] = np.setdiff1d(np.flatnonzero((sup == 'ol_sensory') | ((cls == 'visual') & is_sens)), read_nodes)
    sets['auditory'] = np.setdiff1d(np.flatnonzero(((cls == 'mechanosensory') | (sub == 'auditory')) & is_sens), read_nodes)
    return sets


def readout_sets(body_ids, src, dst, sensory, read_nodes, n):
    ports = set(sensory.tolist())
    fru = node_column(body_ids, 'fruDsx'); cls = node_column(body_ids, 'class')
    indeg = np.bincount(dst, minlength=n)
    hub = np.array([i for i in np.argsort(-indeg, kind='stable') if i not in ports][:len(read_nodes)])
    return {'descending': read_nodes,
            'fru': np.array([i for i in np.flatnonzero(fru != '') if i not in ports]),
            'hub': hub,
            'cx': np.array([i for i in np.flatnonzero(cls == 'CX') if i not in ports])}


def main():
    data = load_graph(10, progress=lambda *a, **k: None)
    body_ids, src, dst = data['body_ids'], data['src'], data['dst']
    n = len(body_ids)
    sensory = np.flatnonzero(data['sensory'])
    sup = node_column(body_ids, 'superclass')
    read_nodes = np.flatnonzero(np.isin(sup, OUTPUT_CLASSES))
    ports = port_sets(body_ids, sensory, read_nodes, n)
    reads = readout_sets(body_ids, src, dst, sensory, read_nodes, n)
    indeg = np.bincount(dst, minlength=n)
    rows = []
    for pname, pset in ports.items():
        dist = layers_from(src, dst, n, pset)
        for rname, rset in reads.items():
            d = dist[rset]; ok = d >= 0
            rows.append(dict(ports=pname, n_ports=int(len(pset)), readout=rname, n_readout=int(len(rset)),
                             reached=float(ok.mean()), median_hops=float(np.median(d[ok])) if ok.any() else None,
                             mean_hops=float(d[ok].mean()) if ok.any() else None,
                             within_2=float((ok & (d <= 2)).mean()), within_3=float((ok & (d <= 3)).mean()),
                             within_4=float((ok & (d <= 4)).mean()),
                             hist=np.bincount(d[ok], minlength=12)[:12].tolist()))
    readout_indeg = {r: dict(median_indeg=float(np.median(indeg[s])), mean_indeg=float(indeg[s].mean())) for r, s in reads.items()}
    OUT.write_text(json.dumps(dict(rows=rows, readout_indeg=readout_indeg), indent=1))
    print(f"{'ports':15s} {'readout':11s} {'reach':>6s} {'med':>4s} {'mean':>5s} {'<=2':>6s} {'<=3':>6s} {'<=4':>6s}  hist")
    for r in rows:
        print(f"{r['ports']:15s} {r['readout']:11s} {r['reached']:6.3f} {r['median_hops'] or 0:4.0f} {r['mean_hops'] or 0:5.2f} {r['within_2']:6.3f} {r['within_3']:6.3f} {r['within_4']:6.3f}  {r['hist']}")
    print('readout in-degree', readout_indeg)


if __name__ == '__main__':
    main()

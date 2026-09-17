"""Degree-preserving directed rewiring control. Nodes, signs, per-source weights and in/out degrees kept."""
import json
import pathlib
import time
import numpy as np
from fly_graph import load_graph, ROOT

SCHEMA = 1


def rewire(src, dst, n, seed, max_rounds=500, progress=print):
    """Permute the destination column, then repair self-loops/duplicates by disjoint swaps."""
    rng = np.random.default_rng(seed)
    new = dst[rng.permutation(len(dst))].copy()
    src64 = src.astype(np.int64)
    for round_index in range(max_rounds):
        bad = new == src
        key = src64 * n + new
        order = np.argsort(key, kind='stable')
        sorted_key = key[order]
        duplicate = np.zeros(len(key), dtype=bool)
        duplicate[order[1:]] = sorted_key[1:] == sorted_key[:-1]
        bad |= duplicate
        conflicts = int(bad.sum())
        if conflicts == 0:
            return new, round_index
        index = np.flatnonzero(bad)
        partner = rng.integers(0, len(dst), size=len(index))
        used = np.zeros(len(dst), dtype=bool)
        left, right = [], []
        for i, p in zip(index.tolist(), partner.tolist()):
            if i == p or used[i] or used[p]:
                continue
            used[i] = used[p] = True
            left.append(i)
            right.append(p)
        if left:
            a = np.asarray(left)
            b = np.asarray(right)
            temporary = new[a].copy()
            new[a] = new[b]
            new[b] = temporary
        progress(f'rewire round {round_index}: conflicts {conflicts}')
    raise RuntimeError('rewiring did not converge')


def repair(src, new, n, rng, max_rounds=500, progress=print):
    """Remove self-loops/duplicates by swapping destinations between disjoint edge pairs (in place)."""
    src64 = src.astype(np.int64)
    for round_index in range(max_rounds):
        bad = new == src
        key = src64 * n + new
        order = np.argsort(key, kind='stable')
        sorted_key = key[order]
        duplicate = np.zeros(len(key), dtype=bool)
        duplicate[order[1:]] = sorted_key[1:] == sorted_key[:-1]
        bad |= duplicate
        conflicts = int(bad.sum())
        if conflicts == 0:
            return round_index
        index = np.flatnonzero(bad)
        partner = rng.integers(0, len(new), size=len(index))
        used = np.zeros(len(new), dtype=bool)
        left, right = [], []
        for i, p in zip(index.tolist(), partner.tolist()):
            if i == p or used[i] or used[p]:
                continue
            used[i] = used[p] = True
            left.append(i)
            right.append(p)
        if left:
            a = np.asarray(left)
            b = np.asarray(right)
            temporary = new[a].copy()
            new[a] = new[b]
            new[b] = temporary
        progress(f'repair round {round_index}: conflicts {conflicts}')
    raise RuntimeError('repair did not converge')


def rewire_config(src, dst, weight, n, seed, progress=print):
    """Configuration-model null: degree multisets kept, degrees re-assigned to random nodes.

    Source and destination roles get two independent node permutations, so which node is a hub, how much
    sensory nodes project and how much readout nodes receive all change; per-node sign (Dale) is untouched
    because signs live on nodes. Edges are then re-sorted by (src, dst) with their weights."""
    rng = np.random.default_rng(seed)
    new_src = rng.permutation(n)[src]
    new_dst = rng.permutation(n)[dst].copy()
    rounds = repair(new_src, new_dst, n, rng, progress=progress)
    order = np.lexsort((new_dst, new_src))
    return new_src[order].astype(src.dtype), new_dst[order].astype(dst.dtype), weight[order], rounds


def verify_config(original, rewired, n):
    src, dst = original['src'], original['dst']
    new_src, new_dst = rewired['src'], rewired['dst']
    assert np.array_equal(np.sort(rewired['weight']), np.sort(original['weight'])), 'weight multiset changed'
    assert np.array_equal(rewired['sign'], original['sign']), 'signs changed'
    assert np.array_equal(rewired['sensory'], original['sensory']), 'sensory flags changed'
    assert np.array_equal(rewired['body_ids'], original['body_ids']), 'body ids changed'
    in_old, in_new = np.bincount(dst, minlength=n), np.bincount(new_dst, minlength=n)
    out_old, out_new = np.bincount(src, minlength=n), np.bincount(new_src, minlength=n)
    assert np.array_equal(np.sort(in_new), np.sort(in_old)), 'in-degree multiset changed'
    assert np.array_equal(np.sort(out_new), np.sort(out_old)), 'out-degree multiset changed'
    assert not (new_dst == new_src).any(), 'self loop present'
    key = new_src.astype(np.int64) * n + new_dst
    assert len(np.unique(key)) == len(key), 'duplicate edge present'
    old_key = src.astype(np.int64) * n + dst
    sensory = original['sensory'].astype(bool)
    sign = original['sign']
    return dict(edges=int(len(src)), nodes=int(n), method='configuration_model',
                preserved_edge_fraction=float(np.isin(key, old_key).mean()),
                reciprocity_original=reciprocity(src, dst, n),
                reciprocity_rewired=reciprocity(new_src, new_dst, n),
                in_degree_multiset_preserved=True, out_degree_multiset_preserved=True,
                per_node_in_degree_correlation=float(np.corrcoef(in_old, in_new)[0, 1]),
                per_node_out_degree_correlation=float(np.corrcoef(out_old, out_new)[0, 1]),
                sensory_out_edges_original=int(out_old[sensory].sum()), sensory_out_edges_rewired=int(out_new[sensory].sum()),
                excitatory_edge_fraction_original=float((sign[src] > 0).mean()),
                excitatory_edge_fraction_rewired=float((sign[new_src] > 0).mean()),
                self_loops=0, duplicates=0)


def node_labels(body_ids, column, folder):
    import pyarrow.feather as pf
    t = pf.read_table(folder / 'body-annotations-male-cns-v1.0.feather', columns=['bodyId', column]).to_pandas().set_index('bodyId')
    values = t[column].reindex(np.asarray(body_ids)).fillna('').astype(str).to_numpy()
    uniq, index = np.unique(values, return_inverse=True)
    return index.astype(np.int64), uniq


def rewire_within_groups(src, dst, n, seed, groups, max_rounds=500, progress=print):
    """Permute destinations only among edges whose destination carries the same group label.

    The group-to-group block matrix (who projects to which compartment, with how many edges) is conserved;
    the fine wiring inside each block is destroyed. Repair swaps stay inside the destination group."""
    rng = np.random.default_rng(seed)
    new = dst.copy()
    g = groups[dst]
    for label in np.unique(g):
        idx = np.flatnonzero(g == label)
        new[idx] = dst[idx][rng.permutation(len(idx))]
    src64 = src.astype(np.int64)
    pools = {label: np.flatnonzero(g == label) for label in np.unique(g)}
    for round_index in range(max_rounds):
        bad = new == src
        key = src64 * n + new
        order = np.argsort(key, kind='stable')
        sorted_key = key[order]
        duplicate = np.zeros(len(key), dtype=bool)
        duplicate[order[1:]] = sorted_key[1:] == sorted_key[:-1]
        bad |= duplicate
        conflicts = int(bad.sum())
        if conflicts == 0:
            return new, round_index
        index = np.flatnonzero(bad)
        used = np.zeros(len(new), dtype=bool)
        left, right = [], []
        for i in index.tolist():
            pool = pools[g[i]]
            p = int(pool[rng.integers(len(pool))])
            if i == p or used[i] or used[p]:
                continue
            used[i] = used[p] = True
            left.append(i)
            right.append(p)
        if left:
            a = np.asarray(left)
            b = np.asarray(right)
            temporary = new[a].copy()
            new[a] = new[b]
            new[b] = temporary
        progress(f'group repair round {round_index}: conflicts {conflicts}')
    raise RuntimeError('group-aware repair did not converge')


def rewire_er(src, dst, weight, n, seed, progress=print):
    """Erdős-Rényi null: uniform random sources and destinations, same edge count, weights permuted."""
    rng = np.random.default_rng(seed)
    new_src = rng.integers(0, n, len(src)).astype(np.int64)
    new_dst = rng.integers(0, n, len(dst)).astype(np.int64)
    rounds = repair(new_src, new_dst, n, rng, progress=progress)
    order = np.lexsort((new_dst, new_src))
    return new_src[order].astype(src.dtype), new_dst[order].astype(dst.dtype), weight[rng.permutation(len(weight))][order], rounds


THRESHOLD5_EDGES = 6242118  # edges of core-graph-5.npz (absolute threshold 5 synapses)


def relative_threshold_graph(original, n, folder, cache, progress=print, edges=None):
    """Keep the edges whose synapse count is the largest fraction of the target's total input (from the threshold-1
    graph), as many edges as the absolute-threshold graph (or `edges` if given)."""
    full = load_graph(1, folder=folder, cache=cache, progress=progress)
    assert np.array_equal(full['body_ids'], original['body_ids']), 'node sets differ between thresholds'
    total_in = np.bincount(full['dst'], weights=full['weight'], minlength=n)
    ratio = full['weight'] / np.maximum(total_in[full['dst']], 1)
    k = int(edges) if edges else len(original['src'])
    keep = np.sort(np.argsort(-ratio, kind='stable')[:k])
    data = dict(body_ids=original['body_ids'], src=full['src'][keep].astype(original['src'].dtype),
                dst=full['dst'][keep].astype(original['dst'].dtype), weight=full['weight'][keep].astype(original['weight'].dtype),
                sign=original['sign'], sensory=original['sensory'])
    old_key = original['src'].astype(np.int64) * n + original['dst']
    new_key = data['src'].astype(np.int64) * n + data['dst']
    stats = dict(edges=int(k), nodes=int(n), method='relative_threshold_matched_edges', ratio_threshold=float(ratio[keep].min()),
                 overlap_with_absolute_threshold=float(np.isin(new_key, old_key).mean()),
                 edges_below_10_synapses=float((data['weight'] < 10).mean()), total_synapses_kept=float(data['weight'].sum()),
                 total_synapses_absolute=float(original['weight'].sum()), reciprocity_original=reciprocity(original['src'], original['dst'], n),
                 reciprocity_rewired=reciprocity(data['src'], data['dst'], n), self_loops=int((data['src'] == data['dst']).sum()),
                 duplicates=int(len(new_key) - len(np.unique(new_key))))
    return data, stats


def reorder_nodes(original, n, folder):
    """Relabel nodes by (superclass, total degree): a pure permutation, mathematically identical model."""
    labels, _ = node_labels(original['body_ids'], 'superclass', folder)
    degree = np.bincount(original['src'], minlength=n) + np.bincount(original['dst'], minlength=n)
    order = np.lexsort((-degree, labels))          # new position -> old node
    position = np.empty(n, dtype=np.int64); position[order] = np.arange(n)   # old node -> new position
    src, dst = position[original['src']], position[original['dst']]
    edge_order = np.lexsort((dst, src))
    data = dict(body_ids=original['body_ids'][order], src=src[edge_order].astype(original['src'].dtype),
                dst=dst[edge_order].astype(original['dst'].dtype), weight=original['weight'][edge_order],
                sign=original['sign'][order], sensory=original['sensory'][order])
    return data, dict(edges=int(len(src)), nodes=int(n), method='node_reorder_superclass_degree', permutation_identity=bool((order == np.arange(n)).all()))


def reciprocity(src, dst, n):
    key = src.astype(np.int64) * n + dst
    reverse = dst.astype(np.int64) * n + src
    return float(np.isin(reverse, key).mean())


def verify(original, rewired, n):
    src, dst, weight = original['src'], original['dst'], original['weight']
    new = rewired['dst']
    assert np.array_equal(rewired['src'], src), 'source column changed'
    assert np.array_equal(rewired['weight'], weight), 'weights changed'
    assert np.array_equal(rewired['sign'], original['sign']), 'signs changed'
    assert np.array_equal(rewired['sensory'], original['sensory']), 'sensory flags changed'
    assert np.array_equal(rewired['body_ids'], original['body_ids']), 'body ids changed'
    assert np.array_equal(np.bincount(new, minlength=n), np.bincount(dst, minlength=n)), 'in-degree changed'
    assert not (new == src).any(), 'self loop present'
    key = src.astype(np.int64) * n + new
    assert len(np.unique(key)) == len(key), 'duplicate edge present'
    return dict(edges=int(len(src)), nodes=int(n),
                preserved_edge_fraction=float((new == dst).mean()),
                reciprocity_original=reciprocity(src, dst, n),
                reciprocity_rewired=reciprocity(src, new, n),
                in_degree_preserved=True, out_degree_preserved=True,
                self_loops=0, duplicates=0)


KINDS = dict(degree=('dst_permutation_repair', 'rewired'), config=('configuration_model_two_permutations', 'configmodel'),
             weights=('weight_permutation', 'permweights'), signs=('node_sign_permutation', 'permsigns'),
             er=('erdos_renyi_repair', 'erdosrenyi'), superclass=('dst_permutation_within_superclass', 'withinsuperclass'),
             softsign=('flip_6pct_node_signs', 'softsign'), relthr=('relative_threshold_matched_edges', 'relthr'),
             relthr5=('relative_threshold_threshold5_edge_count', 'relthr5'),
             relthr_mono=('relative_threshold_matched_edges_monoamines_negative', 'relthrmono'),
             reorder=('node_reorder_superclass_degree', 'reorder'))
MONOAMINES = ('dopamine', 'octopamine', 'serotonin')


def load_rewired_graph(threshold, seed, folder=None, cache=True, progress=print, kind='degree'):
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {sorted(KINDS)}')
    folder = pathlib.Path(folder or ROOT / 'dataset' / 'male_cns')
    paths = [folder / x for x in ['body-annotations-male-cns-v1.0.feather',
             'body-neurotransmitters-male-cns-v1.0.feather', 'connectome-weights-male-cns-v1.0.feather']]
    method, tag = KINDS[kind]
    fingerprint = dict(schema=SCHEMA, threshold=threshold, rewire_seed=seed, method=method,
                       files=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
    stamp = json.dumps(fingerprint, sort_keys=True)
    cached = folder / f'core-graph-{threshold}-{tag}-s{seed}.npz'
    if cache and cached.exists():
        with np.load(cached, allow_pickle=False) as z:
            if str(z['fingerprint']) == stamp:
                progress('rewired cache hit')
                data = {k: z[k] for k in ['body_ids', 'src', 'dst', 'weight', 'sign', 'sensory']}
                return data, json.loads(str(z['stats']))
    original = load_graph(threshold, folder=folder, cache=cache, progress=progress)
    n = len(original['body_ids'])
    started = time.monotonic()
    data = dict(original)
    rounds = 0
    rng = np.random.default_rng(seed)
    if kind == 'degree':
        new_dst, rounds = rewire(original['src'], original['dst'], n, seed, progress=progress)
        data['dst'] = new_dst
        stats = verify(original, data, n)
    elif kind == 'config':
        new_src, new_dst, new_weight, rounds = rewire_config(original['src'], original['dst'], original['weight'], n, seed, progress=progress)
        data.update(src=new_src, dst=new_dst, weight=new_weight)
        stats = verify_config(original, data, n)
    elif kind == 'weights':
        data['weight'] = original['weight'][rng.permutation(len(original['weight']))]
        assert np.array_equal(np.sort(data['weight']), np.sort(original['weight']))
        stats = dict(edges=int(len(original['src'])), nodes=int(n), method=method,
                     weight_rank_correlation=float(np.corrcoef(original['weight'], data['weight'])[0, 1]))
    elif kind == 'signs':
        data['sign'] = original['sign'][rng.permutation(n)]
        stats = dict(edges=int(len(original['src'])), nodes=int(n), method=method,
                     excitatory_node_fraction=float((original['sign'] > 0).mean()),
                     excitatory_edge_fraction_original=float((original['sign'][original['src']] > 0).mean()),
                     excitatory_edge_fraction_rewired=float((data['sign'][original['src']] > 0).mean()),
                     signs_unchanged_fraction=float((data['sign'] == original['sign']).mean()))
    elif kind == 'softsign':
        flip = rng.choice(n, int(round(.06 * n)), replace=False)
        sign = original['sign'].copy(); sign[flip] = -sign[flip]; data['sign'] = sign
        stats = dict(edges=int(len(original['src'])), nodes=int(n), method=method, flipped_nodes=int(len(flip)),
                     flipped_edge_fraction=float(np.isin(original['src'], flip).mean()))
    elif kind == 'er':
        new_src, new_dst, new_weight, rounds = rewire_er(original['src'], original['dst'], original['weight'], n, seed, progress=progress)
        data.update(src=new_src, dst=new_dst, weight=new_weight)
        key = new_src.astype(np.int64) * n + new_dst
        stats = dict(edges=int(len(new_src)), nodes=int(n), method=method, self_loops=int((new_src == new_dst).sum()),
                     duplicates=int(len(key) - len(np.unique(key))), reciprocity_original=reciprocity(original['src'], original['dst'], n),
                     reciprocity_rewired=reciprocity(new_src, new_dst, n), isolated_nodes=int(((np.bincount(new_src, minlength=n) + np.bincount(new_dst, minlength=n)) == 0).sum()),
                     max_in_degree_original=int(np.bincount(original['dst'], minlength=n).max()), max_in_degree_rewired=int(np.bincount(new_dst, minlength=n).max()),
                     sensory_out_edges_original=int(np.bincount(original['src'], minlength=n)[original['sensory'].astype(bool)].sum()),
                     sensory_out_edges_rewired=int(np.bincount(new_src, minlength=n)[original['sensory'].astype(bool)].sum()))
    elif kind == 'superclass':
        groups, uniq = node_labels(original['body_ids'], 'superclass', folder)
        new_dst, rounds = rewire_within_groups(original['src'], original['dst'], n, seed, groups, progress=progress)
        data['dst'] = new_dst
        stats = verify(original, data, n)
        stats.update(method=method, groups=int(len(uniq)), destination_group_preserved=bool((groups[new_dst] == groups[original['dst']]).all()),
                     in_degree_preserved=bool(np.array_equal(np.bincount(new_dst, minlength=n), np.bincount(original['dst'], minlength=n))))
    elif kind == 'relthr':
        data, stats = relative_threshold_graph(original, n, folder, cache, progress=progress)
    elif kind == 'relthr5':
        # same relative criterion, but as many edges as the absolute threshold-5 graph (6.242.118): "more brain" lever
        data, stats = relative_threshold_graph(original, n, folder, cache, progress=progress, edges=THRESHOLD5_EDGES)
    elif kind == 'relthr_mono':
        # standard graph (relative threshold) with the monoaminergic neurons (dopamine / octopamine / serotonin) made
        # inhibitory (-1) instead of the default +1: the Pospisil 2024 sign convention against Shiu 2024 (phase 7c, H5)
        import pyarrow.feather as pf
        data, stats = relative_threshold_graph(original, n, folder, cache, progress=progress)
        nt = pf.read_table(paths[1], columns=['body', 'consensus_nt'])
        nt_map = dict(zip(nt['body'].to_pylist(), nt['consensus_nt'].to_pylist()))
        mono = np.array([nt_map.get(int(b)) in MONOAMINES for b in original['body_ids']])
        sign = data['sign'].copy(); sign[mono] = -1.; data['sign'] = sign
        stats.update(method=method, monoamine_nodes=int(mono.sum()), monoamine_out_edges=int(np.isin(data['src'], np.flatnonzero(mono)).sum()),
                     excitatory_node_fraction=float((sign > 0).mean()))
    elif kind == 'reorder':
        data, stats = reorder_nodes(original, n, folder)
    stats.update(seed=seed, threshold=threshold, kind=kind, repair_rounds=rounds, seconds=time.monotonic() - started)
    if cache:
        temporary = cached.with_suffix('.tmp.npz')
        np.savez(temporary, fingerprint=stamp, stats=json.dumps(stats), **data)
        temporary.replace(cached)
    return data, stats


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threshold', type=int, default=10)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--kind', choices=sorted(KINDS), default='degree')
    p.add_argument('--output', default=str(ROOT / 'results' / 'phase6_rewire_stats.json'))
    a = p.parse_args()
    graph, stats = load_rewired_graph(a.threshold, a.seed, kind=a.kind)
    pathlib.Path(a.output).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

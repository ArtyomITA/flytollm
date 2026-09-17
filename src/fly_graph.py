"""Stream graph; fixed node IDs across thresholds. Cache source fingerprints."""
import json
import pathlib
import time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf

ROOT = pathlib.Path(__file__).resolve().parent


def load_graph(threshold, folder=None, cache=True, progress=print):
    folder = pathlib.Path(folder or ROOT / 'dataset' / 'male_cns')
    paths = [folder / x for x in ['body-annotations-male-cns-v1.0.feather',
             'body-neurotransmitters-male-cns-v1.0.feather', 'connectome-weights-male-cns-v1.0.feather']]
    fingerprint = dict(schema=1, threshold=threshold,
                       files=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
    stamp = json.dumps(fingerprint, sort_keys=True)
    cached = folder / f'core-graph-{threshold}.npz'
    if cache and cached.exists():
        with np.load(cached, allow_pickle=False) as z:
            if str(z['fingerprint']) == stamp:
                progress('cache hit')
                return {k: z[k] for k in ['body_ids','src','dst','weight','sign','sensory']}
    ann = pf.read_table(paths[0])
    ann = ann.filter(pc.is_valid(ann['superclass']))
    order = pc.sort_indices(ann, sort_keys=[('bodyId','ascending')])
    ann = ann.take(order)
    ids = ann['bodyId'].combine_chunks()
    body_ids = ids.to_numpy()
    sensory = np.array(['sensory' in x for x in ann['superclass'].to_pylist()])
    nt = pf.read_table(paths[1])
    nt = nt.filter(pc.is_in(nt['body'], value_set=ids))
    nt_map = dict(zip(nt['body'].to_pylist(), nt['consensus_nt'].to_pylist()))
    sign = np.array([-1 if nt_map.get(int(b)) in {'gaba','glutamate','histamine'} else 1
                     for b in body_ids], dtype=np.float32)
    parts = [[],[],[]]
    last = time.monotonic()
    with pa.memory_map(str(paths[2]), 'r') as f:
        reader = pa.ipc.open_file(f)
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            b = b.filter(pc.greater_equal(b['weight'], threshold))
            b = b.filter(pc.and_(pc.is_in(b['body_pre'], value_set=ids), pc.is_in(b['body_post'], value_set=ids)))
            if b.num_rows:
                parts[0].append(np.searchsorted(body_ids, b['body_pre'].to_numpy()))
                parts[1].append(np.searchsorted(body_ids, b['body_post'].to_numpy()))
                parts[2].append(b['weight'].to_numpy().astype(np.float32))
            if time.monotonic() - last > 5:
                progress(f'batch {i+1}/{reader.num_record_batches}')
                last = time.monotonic()
    src, dst, weight = [np.concatenate(p) if p else np.empty(0, dtype=np.int64 if j<2 else np.float32)
                        for j,p in enumerate(parts)]
    result = dict(body_ids=body_ids, src=src, dst=dst, weight=weight, sign=sign, sensory=sensory)
    if cache:
        temporary = cached.with_suffix('.tmp.npz')
        np.savez(temporary, fingerprint=stamp, **result)
        temporary.replace(cached)
    return result

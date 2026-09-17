"""T7 observational only: threshold changes possible paths, not running model."""
import json,gc
import numpy as np
from fly_graph import load_graph,ROOT
from bench_runtime import memory

def measure(threshold):
    import pyarrow.feather as pf
    from fly_interfaces import OUTPUT_CLASSES
    data=load_graph(threshold);ids=data['body_ids'];n=len(ids)
    rows=pf.read_table(ROOT/'dataset/male_cns/body-annotations-male-cns-v1.0.feather',columns=['bodyId','superclass']).to_pylist()
    lookup={r['bodyId']:r['superclass'] for r in rows}
    sensory=np.flatnonzero(data['sensory']);read=np.array([i for i,b in enumerate(ids) if lookup[int(b)] in OUTPUT_CLASSES])
    def reach(src,dst,seeds):
        seen=np.zeros(n,dtype=bool);seen[seeds]=True;front=seen.copy();layers=[]
        while front.any():
            nodes=np.unique(dst[front[src]]);nodes=nodes[~seen[nodes]];seen[nodes]=True
            front.fill(False);front[nodes]=True;layers.append(len(nodes))
        return seen,layers
    f,fl=reach(data['src'],data['dst'],sensory);r,rl=reach(data['dst'],data['src'],read)
    return dict(threshold=threshold,nodes=n,edges=len(data['src']),readout_reached=int(f[read].sum()),readout_total=len(read),
                sensory_reaching_output=int(r[sensory].sum()),sensory_total=len(sensory),forward_layers=fl,reverse_layers=rl)

if __name__=='__main__':
    before=memory()
    if before['available_gib']<2 or before['commit_available_gib']<6:raise RuntimeError('Insufficient host headroom for graph audit')
    result=[]
    for threshold in (10,5):result.append(measure(threshold));gc.collect()
    (ROOT/'results/phase3_t7_reachability.json').write_text(json.dumps(dict(results=result,pre_memory=before,post_memory=memory(),model_changed=False,scope='CPU metadata/topology only; no training or performance conclusion'),indent=2))
    print(json.dumps(result))

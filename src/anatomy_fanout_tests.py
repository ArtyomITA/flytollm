"""CPU topology-only tests: few seeds -> many nodes; whole10/5 graphs intact."""
import argparse,ast,json,time
from pathlib import Path
import numpy as np
from bench_runtime import ROOT,memory

def expansion(src,dst,n,seeds,read,groups,deadline):
    seen=np.zeros(n,dtype=bool);seen[seeds]=True;front=seen.copy();rows=[]
    for depth in range(1,17):
        if time.monotonic()>deadline:raise TimeoutError('Bounded topology budget exceeded')
        m=memory()
        if m['available_gib']<.5 or m['commit_available_gib']<5:raise RuntimeError('Topology RAM headroom')
        reached=np.unique(dst[front[src]]);fresh=reached[~seen[reached]]
        front.fill(False);front[fresh]=True;seen[fresh]=True
        if depth in (1,2,4,8,16):
            rows.append(dict(hops=depth,reachable_nodes=int(seen.sum()),new_nodes=int(len(fresh)),
                             expansion_ratio=float(seen.sum()/len(seeds)),readout_nodes=int(seen[read].sum()),
                             anatomical_groups=int(len(np.unique(groups[seen])))))
    return rows

def matched(seeds,indegree,outdegree,sides,rng):
    ib=np.floor(np.log2(indegree+1));ob=np.floor(np.log2(outdegree+1));used=np.zeros(len(sides),dtype=bool);used[seeds]=True
    choices=[];distance=[]
    for seed in seeds:
        scores=abs(ib-ib[seed])+abs(ob-ob[seed])+100*(sides!=sides[seed]);scores[used]=1e9
        best=scores.min();pool=np.flatnonzero(scores==best);pick=int(rng.choice(pool));used[pick]=True
        choices.append(pick);distance.append(float(best))
    return np.array(choices),dict(mean_degree_bin_distance=float(np.mean(distance)),max_degree_bin_distance=max(distance),
                                rule='exact side and log2-degree bins where available; nearest bins otherwise; seeds excluded')

def run():
    from fly_graph import load_graph
    import pyarrow.feather as pf
    start=time.monotonic();deadline=start+90;pre=memory()
    if pre['available_gib']<1 or pre['commit_available_gib']<6:raise RuntimeError('Topology preflight headroom')
    graph=load_graph(10);ids=graph['body_ids'];n=len(ids)
    ann=pf.read_table(ROOT/'dataset/male_cns/body-annotations-male-cns-v1.0.feather',columns=['bodyId','superclass','rootSide']).to_pylist()
    lookup={r['bodyId']:r for r in ann};classes=np.array([lookup[int(i)]['superclass'] or 'unknown' for i in ids]);sides=np.array([lookup[int(i)]['rootSide'] or 'unknown' for i in ids]);del ann,lookup
    names=sorted(set(zip(classes,sides)));index={k:i for i,k in enumerate(names)};groups=np.array([index[k] for k in zip(classes,sides)])
    # Read the existing declaration without importing torch or constructing a model.
    tree=ast.parse((ROOT/'fly_interfaces.py').read_text());output_classes=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='OUTPUT_CLASSES' for t in node.targets):output_classes=ast.literal_eval(node.value)
    assert output_classes
    read=np.flatnonzero(np.isin(classes,output_classes));indegree=np.bincount(graph['dst'],minlength=n);outdegree=np.bincount(graph['src'],minlength=n)
    selections={};matching={};rng=np.random.default_rng(93025)
    for label,mask in [('ascending',np.char.find(classes,'ascending')>=0),('descending',np.char.find(classes,'descending')>=0),('sensory',np.char.find(classes,'sensory')>=0),('whole_graph',np.ones(n,dtype=bool))]:
        candidates=np.flatnonzero(mask);order=np.lexsort((ids[candidates],-outdegree[candidates]));seeds=candidates[order[:32]]
        assert len(seeds)==32
        selections[label]=seeds;control,info=matched(seeds,indegree,outdegree,sides,rng);selections[label+'_matched_null']=control;matching[label]=info
    reports=[]
    for threshold in (10,5):
        if threshold==5:
            del graph;graph=load_graph(5);assert np.array_equal(graph['body_ids'],ids)
        src,dst=graph['src'],graph['dst'];inc=np.bincount(dst,minlength=n);out=np.bincount(src,minlength=n)
        buckets=[]
        for label in ('ascending','descending','sensory','vnc_intrinsic','cb_intrinsic','ol_intrinsic'):
            nodes=np.flatnonzero(np.char.find(classes,label)>=0);mask=np.zeros(n,dtype=bool);mask[nodes]=True
            buckets.append(dict(label=label,nodes=len(nodes),incoming_edges=int(inc[nodes].sum()),outgoing_edges=int(out[nodes].sum()),
                                incoming_degree_quantiles=np.quantile(inc[nodes],[0,.5,.9,.99,1]).tolist(),
                                outgoing_degree_quantiles=np.quantile(out[nodes],[0,.5,.9,.99,1]).tolist(),
                                outgoing_to_other_classes=int((mask[src]&(classes[src]!=classes[dst])).sum())))
        probes={}
        for name,seeds in selections.items():
            probes[name]=dict(seed_body_ids=ids[seeds].tolist(),expansion=expansion(src,dst,n,seeds,read,groups,deadline))
        reports.append(dict(threshold=threshold,nodes=n,edges=len(src),group_statistics=buckets,fanout=probes))
    return dict(ok=True,elapsed_s=time.monotonic()-start,pre_memory=pre,post_memory=memory(),
                labels=names,matching=matching,thresholds=reports,selection='Top32 outgoing-degree seeds chosen once on threshold10; identical body IDs tested on5',
                limitations=['Structural paths, not activation propagation or language specialization.',
                             'Hop count is not a guarantee of useful signal through LIF dynamics.',
                             'Sampled soma geometry is not used to choose seeds.',
                             'Controls degree-binned and side-matched, exact fallback distance reported.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--output');a=p.parse_args()
    if a.self_test:
        src=np.array([0,0,1,2,3]);dst=np.array([1,2,3,3,4]);groups=np.arange(5)
        r=expansion(src,dst,5,np.array([0]),np.array([4]),groups,time.monotonic()+5)
        assert r[0]['reachable_nodes']==3 and r[1]['reachable_nodes']==4 and r[2]['readout_nodes']==1
        reverse=expansion(dst,src,5,np.array([0]),np.array([4]),groups,time.monotonic()+5)
        assert all(x['reachable_nodes']==1 for x in reverse)
        print('Directed fan-out smoke passed');return
    result=run();Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(dict(ok=True,output=a.output,seconds=result['elapsed_s'])))

if __name__=='__main__':main()

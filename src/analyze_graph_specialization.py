"""CPU descriptive statistics, train-only conditional tables, held-out DEV/nulls."""
import argparse,json
from pathlib import Path
import numpy as np

def distributions(profiles):
    # Per-neuron group averages already correct for group cardinality.
    energy=np.maximum(profiles[:,:,0].astype(float),0)
    totals=energy.sum(1);valid=totals>0
    return energy[valid]/totals[valid,None],valid

def fit_table(tokens,observed):
    count=np.bincount(tokens,minlength=4096)
    sums=np.zeros((4096,observed.shape[1]));np.add.at(sums,tokens,observed)
    prior=(observed.sum(0)+1e-3);prior/=prior.sum()
    # Fixed ten-observation shrinkage, not tuned on DEV.
    return (sums+10*prior[None])/(count[:,None]+10),prior

def measure(table,prior,tokens,observed):
    predicted=table[tokens]
    conditional=float(-(observed*np.log(predicted.clip(1e-12))).sum(1).mean())
    marginal=float(-(observed*np.log(prior.clip(1e-12))).sum(1).mean())
    return dict(conditional_cross_entropy=conditional,global_cross_entropy=marginal,
                gain_nats=marginal-conditional,
                top_group_agreement=float((predicted.argmax(1)==observed.argmax(1)).mean()))

def profile_tests(observed,tokens,stories):
    entropy=-(observed*np.log(observed.clip(1e-12))).sum(1)
    order=np.sort(observed,axis=1)[:,::-1]
    effective=1/(observed*observed).sum(1)
    counts=np.bincount(tokens,minlength=4096);rng=np.random.default_rng(93027)
    positions=np.zeros(len(stories),dtype=int);seen={}
    for i,s in enumerate(stories):positions[i]=seen.get(s,0);seen[s]=positions[i]+1
    same=[];different=[]
    norms=np.linalg.norm(observed,axis=1)
    for i in rng.permutation(len(tokens)):
        matches=np.flatnonzero((tokens==tokens[i])&(stories!=stories[i]))
        nulls=np.flatnonzero((tokens!=tokens[i])&(stories!=stories[i])&(positions//16==positions[i]//16)
                            &(np.floor(np.log2(counts[tokens]))==np.floor(np.log2(counts[tokens[i]]))))
        if not len(matches) or not len(nulls):continue
        j=int(rng.choice(matches));k=int(rng.choice(nulls))
        same.append(float(observed[i]@observed[j]/(norms[i]*norms[j])))
        different.append(float(observed[i]@observed[k]/(norms[i]*norms[k])))
        if len(same)==256:break
    by_story=[]
    for story in np.unique(stories):
        mask=stories==story
        by_story.append(dict(story=int(story),observations=int(mask.sum()),entropy=float(entropy[mask].mean()),
                             effective_groups=float(effective[mask].mean())))
    return dict(entropy_mean=float(entropy.mean()),effective_group_count_quantiles=np.quantile(effective,[0,.25,.5,.75,1]).tolist(),
                top_k_activity_mass={str(k):float(order[:,:min(k,order.shape[1])].sum(1).mean()) for k in (1,2,4,8)},
                aggregate_group_load=observed.mean(0).tolist(),per_story=by_story,
                context_stability=dict(pairs=len(same),same_token_cosine=float(np.mean(same)) if same else None,
                                       different_token_matched_position_frequency_cosine=float(np.mean(different)) if different else None),
                note='Activity shares are observational proxies, not expert selection or compute savings. Matching uses coarse16-position and log2-frequency bins.')

def bootstrap_gain(table,prior,tokens,observed,stories):
    difference=(observed*(np.log(table[tokens].clip(1e-12))-np.log(prior.clip(1e-12)))).sum(1)
    unique=np.unique(stories);sums=np.array([difference[stories==s].sum() for s in unique]);counts=np.array([(stories==s).sum() for s in unique])
    rng=np.random.default_rng(93028);values=[]
    for _ in range(200):
        draw=rng.integers(len(unique),size=len(unique));values.append(float(sums[draw].sum()/counts[draw].sum()))
    return dict(stories=len(unique),replicates=200,percentile95=np.quantile(values,[.025,.975]).tolist(),
                unit='resampled stories; conditional table fixed from train; exploratory interval')

def analyze(train_path,dev_path,permutations=32):
    with np.load(train_path,allow_pickle=False) as t,np.load(dev_path,allow_pickle=False) as d:
        assert str(t['split'])=='train' and str(d['split'])=='validation'
        assert str(t['checkpoint_sha256'])==str(d['checkpoint_sha256'])
        assert str(t['groups_json'])==str(d['groups_json'])
        assert bool(t['null_groups'])==bool(d['null_groups'])
        train,vt=distributions(t['profiles']);dev,vd=distributions(d['profiles'])
        if not len(train) or not len(dev):return dict(estimable=False,reason='no post4 spiking group energy')
        tt=t['token_ids'][vt];dt=d['token_ids'][vd];stories=t['story_ids'][vt];dev_stories=d['story_ids'][vd]
        table,prior=fit_table(tt,train);real=measure(table,prior,dt,dev)
        rng=np.random.default_rng(93023);null=[]
        for _ in range(permutations):
            shuffled=tt.copy()
            for story in np.unique(stories):
                rows=np.flatnonzero(stories==story);shuffled[rows]=rng.permutation(shuffled[rows])
            tab,p=fit_table(shuffled,train);null.append(measure(tab,p,dt,dev)['gain_nats'])
        return dict(estimable=True,train_rows=len(train),dev_rows=len(dev),excluded_silent_train=int((~vt).sum()),
                    excluded_silent_dev=int((~vd).sum()),null_groups=bool(t['null_groups']),real=real,
                    train_token_label_null_gains=null,permutations=permutations,
                    activity_tests=profile_tests(dev,dt,dev_stories),gain_story_bootstrap=bootstrap_gain(table,prior,dt,dev,dev_stories),
                    permutation_tail_fraction=float((1+sum(x>=real['gain_nats'] for x in null))/(permutations+1)),
                    held_out='DEV16 for development; no audit',
                    interpretation='Predictability of a group-activity proxy by token ID; not LM CE, not a learned router, not causality.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--train');p.add_argument('--dev');p.add_argument('--null-train');p.add_argument('--null-dev')
    p.add_argument('--output');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        tokens=np.tile(np.array([3,4]),64);profiles=np.eye(2)[tokens-3]
        table,prior=fit_table(tokens,profiles);m=measure(table,prior,tokens,profiles)
        assert m['gain_nats']>.5 and m['top_group_agreement']==1
        constant=np.tile([.5,.5],(len(tokens),1));tab,prior=fit_table(tokens,constant)
        assert abs(measure(tab,prior,tokens,constant)['gain_nats'])<1e-12
        summary=profile_tests(constant,tokens,np.repeat(np.arange(4),32))
        assert abs(summary['effective_group_count_quantiles'][2]-2)<1e-12
        assert abs(summary['top_k_activity_mass']['1']-.5)<1e-12
        assert bootstrap_gain(tab,prior,tokens,constant,np.repeat(np.arange(4),32))['percentile95']==[0.,0.]
        print('conditional/null statistics smoke passed');return
    result=dict(ok=True,anatomical=analyze(a.train,a.dev),matched_group_null=analyze(a.null_train,a.null_dev))
    structural=Path(__file__).resolve().parent/'results/anatomy_fanout_thresholds.json'
    if structural.exists():
        topology=json.loads(structural.read_text())
        result['static_fanout_comparison']=dict(path=str(structural),ok=topology.get('ok'),
                                               note='Fixed topology measured before training; compare with learned weighted flow from current probes.')
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(dict(ok=True,output=a.output)))

if __name__=='__main__':main()

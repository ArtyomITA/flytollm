"""Read-only MoE-motivated observability. No routing or parameter changes."""
import argparse, gc, json, sys, traceback
from pathlib import Path
from bench_runtime import ROOT, emit, supervise, memory

def run(a):
    import torch, numpy as np, pyarrow.feather as feather
    from pretrain_resumable import load_payload, digest, fingerprints
    from fly_graph import load_graph
    from fly_lm import copy_state_
    from lm_io import story_batch
    from text_dataset import StoryDataset
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.75)
    source=Path(a.checkpoint); before=digest(source)
    saved=torch.load(source,map_location='cpu',weights_only=True)
    assert saved['fingerprints']==fingerprints(), 'Probe/checkpoint code mismatch'
    model=load_payload(saved['model'],saved['config']['head']); step_number=saved['progress']['updates']
    threshold=saved['config']['threshold']; del saved
    graph_data=load_graph(threshold); body_ids=graph_data['body_ids']
    indegree=np.bincount(graph_data['dst'],minlength=len(body_ids))
    outdegree=np.bincount(graph_data['src'],minlength=len(body_ids)); del graph_data
    annotations=feather.read_table(ROOT/'dataset/male_cns/body-annotations-male-cns-v1.0.feather',
                                  columns=['bodyId','superclass','class','rootSide']).to_pylist()
    lookup={r['bodyId']:r for r in annotations}
    labels=[(lookup[int(i)]['superclass'] or 'unknown',lookup[int(i)]['rootSide'] or 'unknown') for i in body_ids]
    names=sorted(set(labels)); mapping={k:i for i,k in enumerate(names)}
    group=np.array([mapping[k] for k in labels],dtype=np.int64)
    sides=np.array([k[1] for k in labels]); del lookup, annotations, body_ids, labels
    if a.null_groups:
        original_sizes=np.bincount(group,minlength=len(names)); rng=np.random.default_rng(93022)
        inbin=np.floor(np.log2(indegree+1)).astype(int);outbin=np.floor(np.log2(outdegree+1)).astype(int)
        cells={}
        for i,key in enumerate(zip(sides,inbin,outbin)):cells.setdefault(key,[]).append(i)
        for members in cells.values():
            positions=np.array(members);group[positions]=rng.permutation(group[positions])
        assert np.array_equal(np.bincount(group,minlength=len(names)),original_sizes)
    del indegree,outdegree,sides
    groups=len(names); n=model.core.n if hasattr(model.core,'n') else len(group)
    assert n==len(group)
    counts=np.bincount(group,minlength=groups).astype(float)
    group_gpu=torch.from_numpy(group).cuda(); index=group_gpu[None].expand(2,-1)
    counts_gpu=torch.from_numpy(counts).float().cuda()[None]
    state=model.initial_state(2); empty=model.initial_state(2)
    ids=torch.ones(2,device='cuda',dtype=torch.long); targets=ids.clone()
    observed={}
    def read_hook(module,args): observed['rate']=args[1]
    def input_hook(module,args,out): observed['input']=out
    hook=model.interfaces.readout.register_forward_pre_hook(read_hook)
    hook2=model.interfaces.input.register_forward_hook(input_hook)
    active_nodes=torch.zeros(n,device='cuda',dtype=torch.bool)
    accumulated=torch.zeros(n,device='cuda')
    def step():
        weights=model.core.weights()
        reset=torch.ones_like(ids,dtype=torch.bool) if a.reset_each_token else None
        logits,new=model.step(ids,state,weights=weights,reset=reset)
        rate=observed['rate']; current=observed['input']; active=ids.ne(0)
        metrics=[]
        for feature in (rate,rate.ne(0).float(),new.voltage.abs(),current.abs()):
            total=torch.zeros(2,groups,device='cuda').scatter_add_(1,index,feature)
            metrics.append(total/counts_gpu)
        accumulated.add_((rate*active[:,None]).sum(0))
        active_nodes.logical_or_((rate.ne(0)&active[:,None]).any(0))
        lp=logits.log_softmax(-1);valid=active&targets.ne(0)
        ce=-lp.gather(1,targets[:,None]).flatten()
        entropy=-(lp.exp()*lp).sum(-1)
        quality=torch.stack([ce*valid,valid,((logits.argmax(-1)==targets)&valid),entropy*valid],-1)
        copy_state_(state,model.detach(new))
        return torch.stack(metrics,-1),quality,weights
    try:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(),torch.cuda.stream(stream):step()
        torch.cuda.current_stream().wait_stream(stream)
        capture=torch.cuda.CUDAGraph()
        with torch.no_grad(),torch.cuda.graph(capture): outputs=step()
    finally: hook.remove();hook2.remove();observed.clear()
    copy_state_(state,empty);active_nodes.zero_();accumulated.zero_()
    validation=(json.loads((ROOT/'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
                if a.split=='validation' else list(range(9000,9016)))[:a.stories]
    assert len(validation)%2==0
    group_sum=np.zeros((groups,4)); token_sum={}; token_count={}; quality_sum=np.zeros(4); seen=0
    records=[]; record_tokens=[]; record_stories=[]
    with StoryDataset(ROOT/'dataset/prepared_v1',a.split) as ds:
        for b in range(0,len(validation),2):
            copy_state_(state,empty)
            x,y=story_batch(ds,validation[b:b+2],[0,0],a.tokens)
            for t,(token,target) in enumerate(zip(x,y)):
                ids.copy_(token);targets.copy_(target);capture.replay()
                profiles=outputs[0].cpu().numpy();quality_sum+=outputs[1].cpu().numpy().sum(0)
                for lane,token_id in enumerate(token.cpu().tolist()):
                    if not token_id:continue
                    group_sum+=profiles[lane];seen+=1
                    records.append(profiles[lane].copy());record_tokens.append(token_id);record_stories.append(validation[b+lane])
                    token_sum.setdefault(token_id,np.zeros(groups));token_sum[token_id]+=profiles[lane,:,0]
                    token_count[token_id]=token_count.get(token_id,0)+1
                if (t+1)%16==0:emit('probe',pair=b//2+1,token=t+1)
    means=accumulated.cpu().numpy()/max(seen,1);active=active_nodes.cpu().numpy()
    weights=outputs[2].cpu().numpy(); src=model.core.src.cpu().numpy();dst=model.core.dst.cpu().numpy()
    signed=np.zeros(groups*groups);absolute=np.zeros_like(signed);edge_count=np.zeros_like(signed)
    for start in range(0,len(src),262144):
        s=src[start:start+262144];d=dst[start:start+262144]
        bin_id=group[s]*groups+group[d];proxy=means[s]*weights[start:start+262144]
        signed+=np.bincount(bin_id,weights=proxy,minlength=groups*groups)
        absolute+=np.bincount(bin_id,weights=np.abs(proxy),minlength=groups*groups)
        edge_count+=np.bincount(bin_id,minlength=groups*groups)
    top=np.argsort(absolute)[-20:][::-1]
    flows=[dict(source=names[i//groups],destination=names[i%groups],absolute_proxy=float(absolute[i]),
                signed_proxy=float(signed[i]),edges=int(edge_count[i])) for i in top]
    profiles=[dict(token_id=k,count=token_count[k],mean_post_rate=(v/token_count[k]).tolist())
              for k,v in sorted(token_sum.items(),key=lambda kv:-token_count[kv[0]])[:64]]
    group_metrics=group_sum/max(seen,1)
    assert np.isfinite(group_metrics).all() and np.isfinite(quality_sum).all()
    observation_path=Path(a.output).with_suffix('.observations.npz')
    np.savez_compressed(observation_path,profiles=np.array(records,dtype=np.float32),token_ids=np.array(record_tokens),
                        story_ids=np.array(record_stories),group_sizes=counts,checkpoint_sha256=before,
                        groups_json=json.dumps(names),null_groups=a.null_groups,split=a.split)
    result=dict(checkpoint=str(source),checkpoint_update=step_number,checkpoint_sha256=before,
                observations=str(observation_path),observations_sha256=digest(observation_path),
                data=a.split,reset_each_token=a.reset_each_token,
                reset_note='Clears graph state and attention cache, including position; not pure attention ablation' if a.reset_each_token else None,
                group_null='permute within rootSide/log2(indegree+1)/log2(outdegree+1), sizes exact' if a.null_groups else None,
                audit_evaluated=False,test_evaluated=False,stories=validation,tokens_per_story=a.tokens,
                nodes=n,nodes_ever_spiked=int(active.sum()),observed_tokens=seen,
                group_names=names,group_sizes=counts.tolist(),group_metric_columns=['mean_post4_spike_rate','fraction_post4_spiking','mean_abs_voltage','mean_abs_token_current'],
                group_metrics=group_metrics.tolist(),token_profiles=profiles,top_intergroup_proxy=flows,
                ce=float(quality_sum[0]/quality_sum[1]),accuracy=float(quality_sum[2]/quality_sum[1]),
                entropy=float(quality_sum[3]/quality_sum[1]),
                limitations=['Correlation only; no evidence of expert routing or causal specialization.',
                             'Group labels are superclass/side, not anatomical neuropil segmentation.',
                             'Flow proxy is mean post4 spike rate times current weight, not measured trajectory or full8-step current.',
                             'No nodes removed, no activation mask, no weights changed.'])
    assert digest(source)==before, 'Source checkpoint modified'
    capture.reset()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--stories',type=int,default=16);p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--split',choices=['train','validation'],default='validation');p.add_argument('--null-groups',action='store_true')
    p.add_argument('--reset-each-token',action='store_true')
    p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if a.stories<2 or a.stories>16 or a.stories%2 or not 1<=a.tokens<=128:p.error('stories even2..16, tokens1..128')
    if not a.worker:return supervise(a,worker_module='graph_specialization_probe')
    try:
        import torch
        emit('load',memory=memory());result=run(a);peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20;result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak)
    except Exception as exc:traceback.print_exc();result=dict(ok=False,error=str(exc))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

"""P0: graph reachability and train-only sensory sensitivities."""
import argparse,gc,json,sys,traceback
from pathlib import Path
from bench_runtime import emit,supervise,memory
from phase3_t23 import ROOT,sha

def run():
    import torch,numpy as np,pyarrow.feather as pf
    from fly_graph import load_graph
    from lm_io import load_model,story_batch
    from text_dataset import StoryDataset
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.75)
    data=load_graph(10);n=len(data['body_ids']);src,dst=data['src'],data['dst']
    model,_=load_model(ROOT/'results/phase35_adam_s17.diagnostic.pt')
    sensory=model.interfaces.input.nodes.cpu().numpy();read=model.interfaces.readout.nodes.cpu().numpy()
    def bfs(source,target,seeds):
        distance=np.full(n,-1,dtype=np.int32);distance[seeds]=0;front=np.zeros(n,dtype=bool);front[seeds]=True;depth=0
        while front.any():
            reached=np.unique(target[front[source]]);reached=reached[distance[reached]<0]
            depth+=1;distance[reached]=depth;front.fill(False);front[reached]=True
            if depth%8==0:emit('topology_depth',depth=depth)
        return distance
    forward=bfs(src,dst,sensory);reverse=bfs(dst,src,read)
    fields=['bodyId','class','subclass','rootSide','entryNerve','exitNerve','superclass']
    ann=pf.read_table(ROOT/'dataset/male_cns/body-annotations-male-cns-v1.0.feather',columns=fields).to_pylist()
    lookup={r['bodyId']:r for r in ann};records=[lookup[int(b)] for b in data['body_ids']]
    groups={}
    for i,r in enumerate(records):groups.setdefault((r['class'] or 'unknown',r['rootSide'] or 'unknown'),[]).append(i)
    keys=sorted(groups);group_idx=np.empty(n,dtype=np.int64)
    for k,key in enumerate(keys):group_idx[groups[key]]=k
    gi=torch.tensor(group_idx[sensory],device='cuda');counts=torch.bincount(gi,minlength=len(keys)).clamp_min(1)
    ids=torch.ones(16,2,device='cuda',dtype=torch.long);targets=torch.ones(2,device='cuda',dtype=torch.long);curr=[]
    hook=model.interfaces.input.register_forward_hook(lambda m,a,o:curr.append(o))
    def calculate():
        curr.clear();logits,_=model(ids);loss=torch.nn.functional.cross_entropy(logits[-1],targets)
        grads=torch.autograd.grad(loss,curr)
        node=torch.stack([g[:,model.interfaces.input.nodes].square().mean(0) for g in grads]).mean(0)
        pooled=node.new_zeros(len(keys)).index_add(0,gi,node)/counts
        return pooled.sqrt(),loss.detach()
    graph=torch.cuda.CUDAGraph()
    try:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):calculate()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):out=calculate()
    finally:hook.remove();curr.clear()
    sensitivity=[]
    with StoryDataset(ROOT/'dataset/prepared_v1','train') as ds:
        for i in range(98,130,2):
            x,y=story_batch(ds,[i,i+1],[0,0],16)
            if bool((y[-1]==0).any()):raise RuntimeError('Short P0 story')
            ids.copy_(x);targets.copy_(y[-1]);graph.replay();sensitivity.append(out[0].clone())
            emit('sensitivity_pair',pair=(i-98)//2)
    grad=torch.stack(sensitivity).mean(0).tolist();graph.reset()
    by_group=[]
    for k,key in enumerate(keys):
        ix=np.array(groups[key]);f=forward[ix];r=reverse[ix]
        by_group.append(dict(group=key,nodes=len(ix),reachable_from_input=int((f>=0).sum()),can_reach_readout=int((r>=0).sum()),
                             on_input_output_path=int(((f>=0)&(r>=0)).sum()),sensory_gradient_rms=grad[k]))
    return dict(nodes=n,edges=len(src),readout_reached=int((forward[read]>=0).sum()),readout_total=len(read),
                sensory_can_reach_readout=int((reverse[sensory]>=0).sum()),sensory_total=len(sensory),
                max_min_distance_input_to_readout=int(forward[read].max()),groups=by_group,
                training_story_indices=list(range(98,130)),audit_evaluated=False,
                caveat='Reachability is possible paths; surrogate sensitivity is local, not causal proof. No nodes removed.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--allow-paging',action='store_true')
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--phase-timeout',type=float,default=90);p.add_argument('--max-vram-mb',type=int,default=7000)
    a=p.parse_args()
    if not a.worker:return supervise(a,worker_module='phase3_topology')
    try:
        import torch
        emit('load',memory=memory());result=run();peak=torch.cuda.max_memory_allocated()/2**20
        gc.collect();torch.cuda.synchronize();torch._C._cuda_clearCublasWorkspaces();torch.cuda.empty_cache();gc.collect()
        final=torch.cuda.memory_allocated()/2**20;result.update(ok=final==0,final_allocator_mb=final,peak_vram_mb=peak,source_sha256=sha(__file__))
    except Exception as e:traceback.print_exc();result=dict(ok=False,error=str(e))
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result,indent=2))
    if not result['ok']:sys.exit(1)

if __name__=='__main__':main()

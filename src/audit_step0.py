"""Read-only audit of original benchmark; tiny CPU probe or streaming data count."""
import argparse
import ast
import collections
import ctypes
import gc
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent

def memory():
    class M(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(s, ctypes.c_ulonglong) for s in ['total','avail','page','availpage','virtual','availvirtual','extended']]
    m = M(); m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {'ram_pct': m.load, 'available_GiB': m.avail / 2**30}

def toy():
    import torch
    torch.set_num_threads(1)
    original = ROOT / 'bench_throughput.pre_step1.py'
    if not original.exists(): original = ROOT / 'bench_throughput.py'
    tree = ast.parse(original.read_text(encoding='utf-8'))
    main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
    selected = [x for x in main.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in {'Surrogato','Propaga','finestra'}]
    B,N,L,T,E = 2,5,64,8,7
    env = dict(torch=torch, B=B,N=N,L=L,T=T,CHUNK=3,dev=torch.device('cpu'),tau=.95,soglia_spike=1.,
               src_t=torch.tensor([0,1,2,3,4,0,2]), dst_t=torch.tensor([1,2,3,4,0,4,1]),
               mag=torch.nn.Parameter(torch.ones(E)), segno_arco=torch.ones(E))
    exec(compile(ast.Module(body=selected,type_ignores=[]),'original_benchmark_extracted','exec'), env)
    env['spike_fn'] = env['Surrogato'].apply
    env['propaga'] = env['Propaga'].apply
    saved = {}; shapes = collections.Counter()
    def pack(t):
        key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        saved[key] = t
        shapes[str((tuple(t.shape),str(t.dtype)))] += 1
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        loss = env['finestra'](torch.full((L,B,N), .08))
    bn = sum(k[1] for k,t in saved.items() if t.shape == (B,N))
    loss.backward()
    results = {'torch':torch.__version__, 'device':'cpu', 'max_drive_loss':loss.item(),
               'edge_grad_abs_max':env['mag'].grad.abs().max().item(),
               'max_pre_spike_voltage_bound':.95*.08/(1-.95**8),
               'unique_BN_saved_bytes':bn,'bytes_per_BN_substep':bn/(B*N*L*T),
               'projected_B8_N163743_saved_GiB':bn/(B*N)*8*163743/2**30,
               'saved_tensor_calls':dict(shapes)}
    torch.manual_seed(123)
    s=torch.randint(0,2,(B,N)).double().requires_grad_(); w=torch.randn(E,dtype=torch.double,requires_grad=True)
    upstream=torch.randn(B,N,dtype=torch.double)
    y=env['propaga'](s,w)
    grads=torch.autograd.grad(y,(s,w),upstream)
    ref=torch.zeros_like(s).index_add(1,env['dst_t'],s[:,env['src_t']]*w)
    refs=torch.autograd.grad(ref,(s,w),upstream)
    results['binary_propagation_errors']=[(y-ref).abs().max().item()]+[(a-b).abs().max().item() for a,b in zip(grads,refs)]
    del saved
    gc.collect()
    return results

def data():
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as pf
    folder=ROOT/'dataset'/'male_cns'
    ann=pf.read_table(folder/'body-annotations-male-cns-v1.0.feather')
    selected=ann.filter(pc.is_valid(ann['superclass']))
    ids=selected['bodyId'].combine_chunks()
    nt=pf.read_table(folder/'body-neurotransmitters-male-cns-v1.0.feather')
    nt=nt.filter(pc.is_in(nt['body'],value_set=ids))
    nts=collections.Counter(nt['consensus_nt'].to_pylist())
    counts={s:0 for s in [1,2,3,5,10]}; total=0; syn=0; used={5:set(),10:set()}
    central_ids=ann.filter(pc.is_in(ann['superclass'],value_set=pa.array(['cb_intrinsic','cb_sensory','descending_neuron','cb_motor'])))['bodyId'].combine_chunks()
    central={s:0 for s in counts}
    with pa.memory_map(str(folder/'connectome-weights-male-cns-v1.0.feather'),'r') as f:
        reader=pa.ipc.open_file(f)
        for i in range(reader.num_record_batches):
            if memory()['available_GiB'] < 1.5: raise RuntimeError('RAM guard')
            b=reader.get_batch(i); total+=b.num_rows
            b=b.filter(pc.and_(pc.is_in(b['body_pre'],value_set=ids),pc.is_in(b['body_post'],value_set=ids)))
            w=b['weight'].to_numpy(); syn+=int(w.sum())
            c=b.filter(pc.and_(pc.is_in(b['body_pre'],value_set=central_ids),pc.is_in(b['body_post'],value_set=central_ids)))['weight'].to_numpy()
            for s in counts:
                counts[s]+=int(np.count_nonzero(w>=s)); central[s]+=int(np.count_nonzero(c>=s))
            for s in used:
                filt=b.filter(pc.greater_equal(b['weight'],s))
                used[s].update(filt['body_pre'].to_pylist()); used[s].update(filt['body_post'].to_pylist())
            if i%400==0: print(json.dumps({'phase':'stream','batch':i,'batches':reader.num_record_batches,**memory()}),flush=True)
    return dict(annotation_rows=ann.num_rows,selected_neurons=len(ids),superclasses=collections.Counter(selected['superclass'].to_pylist()),
                neurotransmitters=nts,total_edge_rows=total,threshold_counts=counts,synapse_sum=syn,
                incident_neurons={s:len(v) for s,v in used.items()}, central_neurons=len(central_ids),central_counts=central)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['toy','data']);a=p.parse_args()
    start=time.time();pre=memory();print(json.dumps({'phase':'start',**pre}),flush=True)
    result=(toy if a.case=='toy' else data)()
    result.update(pre_memory=pre,post_memory=memory(),elapsed_s=time.time()-start)
    path=ROOT/f'audit_step0_{a.case}.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

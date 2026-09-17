"""Rewired-graph control: identical trainer (pretrain_resumable.run untouched), anatomy replaced."""
import argparse, gc, hashlib, json, sys, traceback
from pathlib import Path
from bench_runtime import ROOT, emit, memory, supervise
import pretrain_resumable as base

CONTROL_SOURCES = ['pretrain_control.py', 'fly_rewire.py', 'fly_graph.py', 'fly_core_variants.py', 'fly_core_fast.py']


ANNOTATIONS = ROOT / 'dataset/male_cns/body-annotations-male-cns-v1.0.feather'


def load_control_graph(threshold, seed, kind):
    from fly_rewire import load_rewired_graph
    from fly_graph import load_graph
    if kind == 'none':
        return load_graph(threshold, progress=lambda m: emit('graph', detail=m)), dict(kind='none', method='original_graph')
    return load_rewired_graph(threshold, seed, progress=lambda m: emit('rewire', detail=m), kind=kind)


def node_column(body_ids, column):
    import numpy as np, pyarrow.feather as pf
    t = pf.read_table(ANNOTATIONS, columns=['bodyId', column]).to_pandas().set_index('bodyId')
    return t[column].reindex(np.asarray(body_ids)).fillna('').astype(str).to_numpy()


def variant_ports(body_ids, sensory, read_nodes, ports, seed=17):
    import numpy as np, torch
    n = len(body_ids)
    if ports == 'anatomical':
        return sensory, dict(ports='anatomical', count=int(sensory.numel()))
    if ports == 'random_matched':
        rng = np.random.default_rng(seed)
        candidates = np.setdiff1d(np.arange(n), read_nodes.numpy())
        chosen = np.sort(rng.choice(candidates, sensory.numel(), replace=False))
        return torch.from_numpy(chosen), dict(ports='random_matched', count=int(len(chosen)), overlap_with_anatomical=float(np.isin(chosen, sensory.numpy()).mean()))
    if ports == 'olfactory':
        cls = node_column(body_ids, 'class')
        chosen = np.flatnonzero(cls == 'olfactory')
        chosen = np.setdiff1d(chosen, read_nodes.numpy())
        return torch.from_numpy(chosen), dict(ports='olfactory', count=int(len(chosen)))
    if ports == 'visual':
        # the eyes: photoreceptors and ocelli (superclass ol_sensory) plus every sensory node of class visual
        sup = node_column(body_ids, 'superclass'); cls = node_column(body_ids, 'class')
        chosen = np.flatnonzero((sup == 'ol_sensory') | ((cls == 'visual') & np.isin(np.arange(n), sensory.numpy())))
        chosen = np.setdiff1d(chosen, read_nodes.numpy())
        return torch.from_numpy(chosen), dict(ports='visual', count=int(len(chosen)))
    if ports == 'auditory':
        # hearing and touch: Johnston's organ and mechanosensory afferents (how the fly hears courtship song)
        cls = node_column(body_ids, 'class'); sub = node_column(body_ids, 'subclass')
        chosen = np.flatnonzero(((cls == 'mechanosensory') | (sub == 'auditory')) & np.isin(np.arange(n), sensory.numpy()))
        chosen = np.setdiff1d(chosen, read_nodes.numpy())
        return torch.from_numpy(chosen), dict(ports='auditory', count=int(len(chosen)))
    if ports == 'shortpath':
        # anatomical input-side populations 1-2 synapses from the descending/motor readout (RICERCA_ATTIVITA_CERVELLO_MOSCA.md):
        # brain mechanosensory afferents (Johnston's organ, wind/gravity, bristles), VNC tactile and proprioceptive afferents
        # (leg/body sensory onto motor circuits) and second-order visual projection neurons (LC/LPLC/MeTu... onto descending neurons)
        sup = node_column(body_ids, 'superclass'); cls = node_column(body_ids, 'class')
        is_sens = np.isin(np.arange(n), sensory.numpy())
        mech = np.isin(cls, ['mechanosensory', 'mechanosensory_tbc', 'mechanosensory_tactile', 'mechanosensory_proprioceptive']) & is_sens
        chosen = np.flatnonzero(mech | (sup == 'visual_projection'))
        chosen = np.setdiff1d(chosen, read_nodes.numpy())
        return torch.from_numpy(chosen), dict(ports='shortpath', count=int(len(chosen)), mechanosensory=int(mech.sum()),
                                              visual_projection=int((sup == 'visual_projection').sum()))
    raise ValueError(ports)


def variant_readout(body_ids, src, dst, sensory, read_nodes, groups, readout, pool_size=32):
    """Alternative readout populations. Returns (read_nodes, groups, info); chunks of pool_size inside each label like
    anatomical_ports, so the readout head keeps the same construction."""
    import numpy as np, torch
    n = len(body_ids); ports = set(sensory.tolist())
    if readout == 'chunks':
        return read_nodes, groups, dict(readout='chunks', groups=int(groups.max()) + 1, count=int(read_nodes.numel()))
    side = node_column(body_ids, 'rootSide')
    if readout == 'fru':
        # courtship / communication circuitry: fruitless- and doublesex-expressing neurons
        fru = node_column(body_ids, 'fruDsx')
        labels = {i: (fru[i], side[i] or 'unknown') for i in np.flatnonzero(fru != '') if i not in ports}
    elif readout == 'hub':
        # best-connected nodes: highest in-degree outside the ports, same count as the anatomical readout
        indeg = np.bincount(dst, minlength=n)
        order = [i for i in np.argsort(-indeg, kind='stable') if i not in ports][:int(read_nodes.numel())]
        labels = {int(i): ('hub_decile_%d' % (k * 10 // len(order)), side[i] or 'unknown') for k, i in enumerate(order)}
    elif readout == 'cx':
        # integration centre: central complex neurons
        cls = node_column(body_ids, 'class')
        labels = {i: ('CX', side[i] or 'unknown') for i in np.flatnonzero(cls == 'CX') if i not in ports}
    else:
        raise ValueError(readout)
    buckets = {}
    for i, key in labels.items():
        buckets.setdefault(key, []).append(int(i))
    nodes, group_ids, names = [], [], []
    for key in sorted(buckets):
        members = sorted(buckets[key], key=lambda i: int(body_ids[i]))
        for start in range(0, len(members), pool_size):
            chunk = members[start:start + pool_size]
            nodes.extend(chunk); group_ids.extend([len(names)] * len(chunk)); names.append(dict(label=key[0], side=key[1], nodes=len(chunk)))
    if not nodes:
        raise ValueError('empty readout selection')
    return (torch.tensor(nodes, dtype=torch.long), torch.tensor(group_ids, dtype=torch.long),
            dict(readout=readout, groups=len(names), count=len(nodes), labels=names[:200]))


def anatomical_readout_groups(body_ids, read_nodes):
    """Group readout nodes by (superclass, exit nerve | soma neuromere | root side) instead of arbitrary chunks of 32."""
    import numpy as np, torch
    superclass = node_column(body_ids, 'superclass'); nerve = node_column(body_ids, 'exitNerve')
    neuromere = node_column(body_ids, 'somaNeuromere'); side = node_column(body_ids, 'rootSide')
    keys = []
    for i in read_nodes.tolist():
        second = nerve[i] or neuromere[i] or side[i] or 'unknown'
        keys.append((superclass[i], second))
    uniq = sorted(set(keys)); index = {k: j for j, k in enumerate(uniq)}
    groups = torch.tensor([index[k] for k in keys], dtype=torch.long)
    sizes = torch.bincount(groups).tolist()
    return groups, dict(readout='anatomical', groups=len(uniq), min_group=min(sizes), max_group=max(sizes),
                        labels=[dict(superclass=k[0], key=k[1], nodes=s) for k, s in zip(uniq, sizes)])


def core_variant_arg(value):
    from fly_core_variants import parse_variant
    parse_variant(value)  # raises on unknown flags
    return value


def build_control_cns(threshold, seed, kind='degree', ports='anatomical', readout='chunks', core_variant='lif', fast_mode='off', chunk=262144, index_dtype='int64',
                      pre_steps=4, post_steps=4, weight_scale=1.0):
    import numpy as np, torch
    from fly_core import Core
    from fly_core_variants import parse_variant
    from fly_interfaces import TextInterfaces, anatomical_ports
    from fly_attention import CausalAttention
    from fly_lm import FlyLM, LMConfig
    data, stats = load_control_graph(threshold, seed, kind)
    n = len(data['body_ids']); counts = np.log1p(data['weight'])
    incoming = np.bincount(data['dst'], weights=counts, minlength=n)
    magnitude = (float(weight_scale) * .5 * counts / np.maximum(incoming[data['dst']], 1)).astype(np.float32)
    args = (torch.from_numpy(data['src']), torch.from_numpy(data['dst']), torch.from_numpy(magnitude), torch.from_numpy(data['sign']), n)
    extra = {}
    if weight_scale != 1.0:
        extra['weight_scale'] = float(weight_scale)
    flags = parse_variant(core_variant)
    if fast_mode == 'fused':
        from fly_core_fast import _fused_kernels
        _fused_kernels()  # compile before the core exists: a lazy NVRTC compile inside propagate() kept that frame (and the core) alive
    if not flags and fast_mode == 'fused':
        from fly_core_fast import FusedCore
        core = FusedCore(*args, chunk=chunk)
        extra['core'] = core.describe()
    elif not flags and fast_mode != 'off':
        from fly_core_fast import FastCore
        core = FastCore(*args, mode=fast_mode, chunk=chunk, index_dtype=torch.int32 if index_dtype == 'int32' else torch.long)
        extra['core'] = core.describe()
    elif not flags:
        core = Core(*args)
    else:
        from fly_core_variants import VariantCore
        if fast_mode == 'fused':
            from fly_core_fast import FusedVariantCore as VariantCore
        kwargs = {}
        if 'apl' in flags:
            kwargs['kc'] = torch.from_numpy(np.flatnonzero(node_column(data['body_ids'], 'class') == 'Kenyon_Cell'))
        if 'graded_ol' in flags:
            kwargs['ol'] = torch.from_numpy(np.flatnonzero(node_column(data['body_ids'], 'superclass') == 'ol_intrinsic'))
        if flags & {'tau_type', 'bias_type'}:
            _, index = np.unique(node_column(data['body_ids'], 'type'), return_inverse=True)
            kwargs['type_index'] = torch.from_numpy(index.astype(np.int64))
        core = VariantCore(*args, core_variant, **kwargs)
        extra['core'] = core.describe()
    sensory, read_nodes, groups, labels = anatomical_ports(data['body_ids'], ANNOTATIONS)
    if readout in ('fru', 'hub', 'cx'):
        read_nodes, groups, readout_info = variant_readout(data['body_ids'], data['src'], data['dst'], sensory, read_nodes, groups, readout)
        extra['readout'] = readout_info
    sensory, port_info = variant_ports(data['body_ids'], sensory, read_nodes, ports)
    extra['ports'] = port_info
    if readout == 'anatomical':
        groups, readout_info = anatomical_readout_groups(data['body_ids'], read_nodes)
        extra['readout'] = readout_info
    interfaces = TextInterfaces(n, sensory, read_nodes, groups)
    model = FlyLM(core, interfaces, CausalAttention(), LMConfig(pre_steps=int(pre_steps), post_steps=int(post_steps))).cuda()
    extra['steps'] = dict(pre=int(pre_steps), post=int(post_steps))
    stats = dict(stats, **extra)
    return model, data, stats


def build_rewired_cns(threshold, seed, kind='degree'):
    return build_control_cns(threshold, seed, kind)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threshold', type=int, choices=[10, 5], default=10)
    p.add_argument('--head', choices=['tied', 'separate'], default='separate')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--rewire-seed', type=int, default=41)
    from fly_rewire import KINDS
    from fly_core_variants import VARIANTS
    p.add_argument('--rewire-kind', choices=sorted(KINDS) + ['none'], default='degree',
                   help='graph null model (see fly_rewire.KINDS); none = original graph, for ports/readout/core variants')
    p.add_argument('--ports', choices=['anatomical', 'random_matched', 'olfactory', 'visual', 'auditory', 'shortpath'], default='anatomical')
    p.add_argument('--pre-steps', type=int, default=4, help='core substeps per token before the attention read (main model: 4)')
    p.add_argument('--post-steps', type=int, default=4, help='core substeps per token after the attention read (main model: 4)')
    p.add_argument('--readout', choices=['chunks', 'anatomical', 'fru', 'hub', 'cx'], default='chunks')
    p.add_argument('--core-variant', type=core_variant_arg, default='lif', help="one of %s or several joined by '+', e.g. bias_type+tau_type+reversal" % (VARIANTS,))
    p.add_argument('--weight-scale', type=float, default=1.0, help='multiply the initial synaptic magnitudes by this factor (phase 7c, global scale lever)')
    p.add_argument('--type-param-lr', type=float, default=None, help='dedicated Adam learning rate for the per-cell-type parameters (leak_logit, bias_type); default = same as the rest')
    p.add_argument('--optimizer', choices=['adam', 'muon'], default='adam', help='muon = Moonlight-style Muon on the dense nn.Linear matrices (attention q/k/v/o, readout projection), Adam on the rest (phase 7e)')
    p.add_argument('--muon-lr', type=float, default=1e-4, help='Muon base learning rate; effective per matrix = base * 0.2 * sqrt(max(shape))')
    p.add_argument('--muon-head', action='store_true', help='also give the separate output head matrix (4096x256) to Muon')
    p.add_argument('--fast-mode', choices=['off', 'gather', 'fp16', 'csr', 'fused'], default='off', help='speed core (fly_core_fast.FastCore); off = fly_core.Core')
    p.add_argument('--chunk', type=int, default=262144, help='edges per propagation chunk (fast core only)')
    p.add_argument('--index-dtype', choices=['int64', 'int32'], default='int64', help='edge index dtype (fast core only)')
    p.add_argument('--adam', choices=['default', 'fused', 'foreach'], default='default', help='Adam implementation (monkeypatched; phase3_t45 untouched)')
    p.add_argument('--updates', type=int, required=True)
    p.add_argument('--resume')
    p.add_argument('--checkpoint-every', type=int, default=2000)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--output', required=True)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--allow-paging', action='store_true')
    p.add_argument('--timeout', type=float, default=31536000)
    p.add_argument('--phase-timeout', type=float, default=90)
    p.add_argument('--max-vram-mb', type=int, default=7000)
    a = p.parse_args()
    if a.updates < 0 or min(a.checkpoint_every, a.eval_every) < 1:
        p.error('updates >=0, positive intervals required')
    if not a.worker:
        return supervise(a, worker_module='pretrain_control')
    try:
        import torch, lm_io
        emit('load', python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda, memory=memory())
        holder = {}

        def patched(threshold=10):
            model, data, stats = build_control_cns(threshold, a.rewire_seed, a.rewire_kind, a.ports, a.readout, a.core_variant, a.fast_mode, a.chunk, a.index_dtype,
                                                   a.pre_steps, a.post_steps, a.weight_scale)
            holder['data'] = data; holder['stats'] = stats; holder['model'] = model
            return model
        lm_io.build_cns = patched
        if a.adam != 'default' or a.type_param_lr is not None or a.optimizer == 'muon':
            original_adam = torch.optim.Adam

            def patched_adam(params, **kw):
                kw.pop('foreach', None)
                if a.adam == 'fused':
                    kw['fused'] = True
                elif a.adam == 'foreach':
                    kw['foreach'] = True
                params = list(params)
                if params and isinstance(params[0], dict):
                    raise RuntimeError('param groups already present; the control optimizer patch expects a flat parameter list')
                typed = []
                if a.type_param_lr is not None:
                    # dedicated learning rate for the per-cell-type parameters (phase 7c: D6 was budget-limited at 1e-4)
                    core = holder['model'].core
                    special = {id(t) for name, t in core.named_parameters() if name in ('leak_logit', 'bias_type')}
                    typed = [p for p in params if id(p) in special]
                    if not typed:
                        raise RuntimeError('type-param-lr given but the core has no per-type parameters')
                    holder['stats']['type_param_lr'] = dict(lr=float(a.type_param_lr), tensors=len(typed), values=int(sum(p.numel() for p in typed)))
                if a.optimizer == 'muon':
                    from optimizer_variants import HybridMuon
                    kw.pop('fused', None)
                    opt = HybridMuon(holder['model'], params, kw.pop('lr'), a.muon_lr, include_head=a.muon_head,
                                     adam_groups=[(typed, float(a.type_param_lr))] if typed else None, foreach=False, **kw)
                    holder['stats']['optimizer'] = dict(kind='muon', muon_lr=float(a.muon_lr), matrices=opt.matrix_names,
                                                        matrix_values=int(sum(p.numel() for p in opt.matrices)))
                    return opt
                if typed:
                    typed_ids = {id(p) for p in typed}
                    params = [dict(params=[p for p in params if id(p) not in typed_ids]), dict(params=typed, lr=float(a.type_param_lr))]
                return original_adam(params, **kw)
            torch.optim.Adam = patched_adam
        data, stats = load_control_graph(a.threshold, a.rewire_seed, a.rewire_kind)
        holder.setdefault('stats', stats)
        if a.resume:
            saved = torch.load(a.resume, map_location='cpu', weights_only=True)
            state = saved['model']['state_dict']
            assert torch.equal(state['core.src'], torch.from_numpy(data['src'])), 'checkpoint topology is not this rewired graph'
            assert torch.equal(state['core.dst'], torch.from_numpy(data['dst'])), 'checkpoint topology is not this rewired graph'
            del saved, state
        result = base.run(a)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        gc.collect(); torch.cuda.synchronize(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache(); gc.collect()
        final = torch.cuda.memory_allocated() / 2 ** 20
        # fused cupy kernels leave a stable CUDA-graph pool residue (~100 MB, no growth during training): tolerated and recorded
        clean = final == 0 or (a.fast_mode == 'fused' and final < 256)
        result.update(ok=clean, final_allocator_mb=final, allocator_residue_tolerated=(final != 0), peak_vram_mb=peak, fingerprints=base.fingerprints(),
                      control=dict(kind={'degree': 'degree_preserving_rewire', 'config': 'configuration_model'}.get(a.rewire_kind, a.rewire_kind),
                                   rewire_seed=a.rewire_seed, rewire_kind=a.rewire_kind, ports=a.ports, readout=a.readout,
                                   core_variant=a.core_variant, fast_mode=a.fast_mode, chunk=a.chunk, index_dtype=a.index_dtype, adam=a.adam,
                                   pre_steps=a.pre_steps, post_steps=a.post_steps, weight_scale=a.weight_scale, type_param_lr=a.type_param_lr,
                                   optimizer=a.optimizer, muon_lr=a.muon_lr if a.optimizer == 'muon' else None, stats=holder['stats'],
                                   sources={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CONTROL_SOURCES}),
                      semantic_verdict=f'control run: graph={a.rewire_kind}, ports={a.ports}, readout={a.readout}, core={a.core_variant}; same nodes and init; 2000-update test, not a model change')
    except Exception as exc:
        traceback.print_exc(); result = dict(ok=False, error=str(exc), error_type=type(exc).__name__)
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result, indent=2))
    if not result['ok']:
        sys.exit(1)


if __name__ == '__main__':
    main()

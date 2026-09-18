"""Rewired-graph control: identical trainer (pretrain_resumable.run untouched), anatomy replaced."""
import argparse, gc, hashlib, json, sys, traceback
from pathlib import Path
from bench_runtime import ROOT, emit, memory, supervise
import pretrain_resumable as base

CONTROL_SOURCES = ['pretrain_control.py', 'fly_rewire.py', 'fly_graph.py', 'fly_core_variants.py', 'fly_core_fast.py', 'fly_lm_variants.py', 'optimizer_variants.py', 'fly_interfaces_variants.py']


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


def variant_ports(body_ids, sensory, read_nodes, ports, seed=17, count=None, graph=None):
    import numpy as np, torch
    n = len(body_ids)
    if ports == 'convergent':
        # P4 (phase 8): the neurons reached by every sensory modality within 3 synapses (multisensory convergence zones)
        from fly_interfaces_variants import convergent_nodes
        cls = node_column(body_ids, 'class'); sup = node_column(body_ids, 'superclass')
        chosen = convergent_nodes(graph['src'], graph['dst'], n, cls, sup, sensory.numpy(), hops=4)   # 3 hops: 204 nodes on relthr; 4 hops: 8,129
        chosen = np.setdiff1d(chosen, read_nodes.numpy())
        return torch.from_numpy(chosen.astype(np.int64)), dict(ports='convergent', count=int(len(chosen)), hops=4, overlap_with_anatomical=float(np.isin(chosen, sensory.numpy()).mean()))
    if ports == 'anatomical':
        return sensory, dict(ports='anatomical', count=int(sensory.numel()))
    if ports == 'random_matched':
        rng = np.random.default_rng(seed)
        candidates = np.setdiff1d(np.arange(n), read_nodes.numpy())
        chosen = np.sort(rng.choice(candidates, int(count) if count else sensory.numel(), replace=False))
        return torch.from_numpy(chosen), dict(ports='random_matched', count=int(len(chosen)), seed=int(seed), overlap_with_anatomical=float(np.isin(chosen, sensory.numpy()).mean()))
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
                      pre_steps=4, post_steps=4, weight_scale=1.0, loop=None, freeze_core=False, ports_seed=17, ports_count=None, homeo=None, arousal=None,
                      init_norm='sum', port_channels='mixed', port_encoder='node', gain_groups='superclass'):
    import numpy as np, torch
    from fly_core import Core
    from fly_core_variants import parse_variant
    from fly_interfaces import TextInterfaces, anatomical_ports
    from fly_attention import CausalAttention
    from fly_lm import FlyLM, LMConfig
    data, stats = load_control_graph(threshold, seed, kind)
    n = len(data['body_ids']); counts = np.log1p(data['weight'])
    incoming = np.bincount(data['dst'], weights=counts, minlength=n)
    if init_norm == 'fluct':
        # E1 (phase 8, Rossbroich-Gygax-Zenke 2022): variance normalisation, weight_scale = target root-sum-square of the
        # incoming weights (fluctuation-driven), instead of the sum normalised to 0.5 (mean-driven, quiescent)
        incoming_sq = np.bincount(data['dst'], weights=counts ** 2, minlength=n)
        magnitude = (float(weight_scale) * counts / np.sqrt(np.maximum(incoming_sq[data['dst']], 1e-12))).astype(np.float32)
    else:
        magnitude = (float(weight_scale) * .5 * counts / np.maximum(incoming[data['dst']], 1)).astype(np.float32)
    args = (torch.from_numpy(data['src']), torch.from_numpy(data['dst']), torch.from_numpy(magnitude), torch.from_numpy(data['sign']), n)
    extra = {}
    if weight_scale != 1.0:
        extra['weight_scale'] = float(weight_scale)
    if init_norm != 'sum':
        extra['init_norm'] = init_norm
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
        if 'homeo' in flags and homeo:
            kwargs['homeo'] = homeo
        if 'arousal' in flags and arousal:
            kwargs['arousal'] = arousal
        if flags & {'tau_type', 'bias_type', 'homeo'}:
            _, index = np.unique(node_column(data['body_ids'], 'type'), return_inverse=True)
            kwargs['type_index'] = torch.from_numpy(index.astype(np.int64))
        if 'gain_group' in flags:
            if gain_groups == 'modality':
                from fly_interfaces_variants import modality_index, MODALITIES
                sens0, _, _, _ = anatomical_ports(data['body_ids'], ANNOTATIONS)
                index = modality_index(node_column(data['body_ids'], 'class'), node_column(data['body_ids'], 'superclass'), sens0.numpy())
                names = ['non porta'] + list(MODALITIES)
            else:
                names, index = np.unique(node_column(data['body_ids'], 'superclass'), return_inverse=True)
            kwargs['group_index'] = torch.from_numpy(index.astype(np.int64))
            extra['gain_groups'] = [str(x) for x in names]
        core = VariantCore(*args, core_variant, **kwargs)
        extra['core'] = core.describe()
    sensory, read_nodes, groups, labels = anatomical_ports(data['body_ids'], ANNOTATIONS)
    if readout in ('fru', 'hub', 'cx'):
        read_nodes, groups, readout_info = variant_readout(data['body_ids'], data['src'], data['dst'], sensory, read_nodes, groups, readout)
        extra['readout'] = readout_info
    sensory, port_info = variant_ports(data['body_ids'], sensory, read_nodes, ports, seed=ports_seed, count=ports_count, graph=data)
    extra['ports'] = port_info
    if readout == 'anatomical':
        groups, readout_info = anatomical_readout_groups(data['body_ids'], read_nodes)
        extra['readout'] = readout_info
    interfaces = TextInterfaces(n, sensory, read_nodes, groups)
    if port_channels == 'modal' or port_encoder == 'type':
        from fly_interfaces_variants import modality_index, modal_channels_, TypeSharedInjector
        cls_all = node_column(data['body_ids'], 'class'); sup_all = node_column(data['body_ids'], 'superclass')
        if port_channels == 'modal':
            modality = modality_index(cls_all, sup_all, sensory.numpy())[sensory.numpy()]
            extra['port_channels'] = modal_channels_(interfaces.input, modality, interfaces.config.dim, interfaces.config.fan_in, interfaces.config.seed)
        if port_encoder == 'type':
            types = node_column(data['body_ids'], 'type')[sensory.numpy()]
            types = np.where(types == '', 'untyped', types)
            names_t, group = np.unique(types, return_inverse=True)
            interfaces.input = TypeSharedInjector(interfaces.input, torch.from_numpy(group.astype(np.int64)), interfaces.config.seed)
            extra['port_encoder'] = dict(kind='type', groups=int(len(names_t)), untyped_ports=int((types == 'untyped').sum()), weights=int(interfaces.input.weight.numel()))
    if freeze_core:
        # reservoir / identity controls (phase 8): the synaptic weights become a buffer, never trained
        raw = core._parameters.pop('raw')
        core.register_buffer('raw', raw.detach().clone())
        extra['freeze_core'] = True
    config = LMConfig(pre_steps=int(pre_steps), post_steps=int(post_steps))
    if loop is None:
        model = FlyLM(core, interfaces, CausalAttention(), config).cuda()
    else:
        from fly_lm_variants import LoopedFlyLM
        from dataclasses import asdict
        model = LoopedFlyLM(core, interfaces, CausalAttention(), config, loop).cuda()
        extra['loop'] = dict(asdict(loop), reads=list(model.reads), rows=model.rows)
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
    p.add_argument('--ports', choices=['anatomical', 'random_matched', 'olfactory', 'visual', 'auditory', 'shortpath', 'convergent'], default='anatomical')
    p.add_argument('--port-channels', choices=['mixed', 'modal'], default='mixed', help='modal = every anatomical port reads only the embedding channel block of its sensory modality (phase 8, P1)')
    p.add_argument('--port-encoder', choices=['node', 'type'], default='node', help='type = ports of the same sensory cell type share channels, weight and bias (phase 8, P2)')
    p.add_argument('--gain-groups', choices=['superclass', 'modality'], default='superclass', help='grouping of the gain_group core flag (phase 8, P3: one gain per sensory modality)')
    p.add_argument('--pre-steps', type=int, default=4, help='core substeps per token before the attention read (main model: 4)')
    p.add_argument('--post-steps', type=int, default=4, help='core substeps per token after the attention read (main model: 4)')
    p.add_argument('--readout', choices=['chunks', 'anatomical', 'fru', 'hub', 'cx'], default='chunks')
    p.add_argument('--core-variant', type=core_variant_arg, default='lif', help="one of %s or several joined by '+', e.g. bias_type+tau_type+reversal" % (VARIANTS,))
    p.add_argument('--weight-scale', type=float, default=1.0, help='multiply the initial synaptic magnitudes by this factor (phase 7c, global scale lever)')
    p.add_argument('--type-param-lr', type=float, default=None, help='dedicated Adam learning rate for the per-cell-type parameters (leak_logit, bias_type); default = same as the rest')
    p.add_argument('--core-accumulate', type=int, default=0, help='E5 (phase 8): Adam on core.raw applied every N updates on the gradient summed over those N updates (effective batch x N for the synapses only; needs --optimizer muon and --core-lr)')
    p.add_argument('--core-lr', type=float, default=None, help='dedicated Adam learning rate for the synaptic weights (core.raw); phase 8: at 1e-4 the 2.75 M weights move 0.3%% in 8000 updates')
    p.add_argument('--optimizer', choices=['adam', 'muon'], default='adam', help='muon = Moonlight-style Muon on the dense nn.Linear matrices (attention q/k/v/o, readout projection), Adam on the rest (phase 7e)')
    p.add_argument('--muon-lr', type=float, default=1e-4, help='Muon base learning rate; effective per matrix = base * 0.2 * sqrt(max(shape))')
    p.add_argument('--muon-head', action='store_true', help='also give the separate output head matrix (4096x256) to Muon')
    p.add_argument('--attn-reads', default='default', help="substeps after which the attention is read: 'default' (after pre-steps, the main model), 'off', 'every', or a comma list such as 2,6 (phase 8, LoopedFlyLM)")
    p.add_argument('--attn-kv', choices=['final', 'per_read', 'first'], default='final', help='KV cache regime of the looped model')
    p.add_argument('--attn-step-id', action='store_true', help='per-read gain and bias on the query input (step identifier)')
    p.add_argument('--attn-inject', choices=['none', 'concat'], default='none', help='concat = query input is adapter([state ; token embedding])')
    p.add_argument('--token-injection', choices=['all', 'first', 'first_tonic'], default='all', help='token current at every substep (main model), at the first substep only, or at the first substep with the tonic injector bias kept at every substep (N5b)')
    p.add_argument('--depth-schedule', default='', help="substeps per half changing during training, e.g. 0:12,500:8,1000:4,1500:12 (start_update:depth); needs --pre-steps = --post-steps = the maximum depth (phase 8, N6d)")
    p.add_argument('--homeo', default='', help='target,eta of the wake-up homeostasis (core flag homeo), e.g. 0.02,2e-5')
    p.add_argument('--arousal', default='', help='shape,amplitude,period_updates,duty_updates of the arousal schedule (core flag arousal), e.g. pulse,0.3,500,100 or smooth,0.3,500,0')
    p.add_argument('--init-norm', choices=['sum', 'fluct'], default='sum', help='synaptic init: sum of incoming weights = 0.5 (main model) or fluctuation-driven, weight-scale = root-sum-square of the incoming weights (phase 8, E1)')
    p.add_argument('--output-scale', type=float, default=1.0, help='logits multiplier (phase 8, E2: small values push training out of the lazy regime)')
    p.add_argument('--interface-lr', type=float, default=None, help='Adam lr of the interface parameters (embedding, injectors, readout, head, norms) (phase 8, E3: slow interfaces)')
    p.add_argument('--init-from', default=None, help='checkpoint (.pt) whose model weights initialise this run (same graph); optimizer fresh (phase 8, E4)')
    p.add_argument('--freeze-interfaces', action='store_true', help='train only the core parameters; everything else frozen as buffers (phase 8, E4)')
    p.add_argument('--freeze-core', action='store_true', help='synaptic weights frozen at their initial values (reservoir / identity controls)')
    p.add_argument('--ports-seed', type=int, default=17, help='seed of the random ports (random_matched)')
    p.add_argument('--ports-count', type=int, default=None, help='number of random ports (random_matched); default = as many as the anatomical ports')
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
            loop = None
            if a.attn_reads != 'default' or a.attn_kv != 'final' or a.attn_step_id or a.attn_inject != 'none' or a.token_injection != 'all' or a.depth_schedule or a.output_scale != 1.0 or a.freeze_interfaces:
                from fly_lm_variants import LoopConfig, parse_reads, parse_schedule
                loop = LoopConfig(reads=parse_reads(a.attn_reads, a.pre_steps, a.post_steps), kv=a.attn_kv, step_id=a.attn_step_id,
                                  inject=a.attn_inject, token_injection=a.token_injection, depth_schedule=parse_schedule(a.depth_schedule), output_scale=a.output_scale)
            model, data, stats = build_control_cns(threshold, a.rewire_seed, a.rewire_kind, a.ports, a.readout, a.core_variant, a.fast_mode, a.chunk, a.index_dtype,
                                                   a.pre_steps, a.post_steps, a.weight_scale, loop=loop, freeze_core=a.freeze_core,
                                                   ports_seed=a.ports_seed, ports_count=a.ports_count,
                                                   init_norm=a.init_norm, port_channels=a.port_channels, port_encoder=a.port_encoder, gain_groups=a.gain_groups,
                                                   homeo=tuple(float(x) for x in a.homeo.split(',')) if a.homeo else None,
                                                   arousal=(lambda f: (f[0], float(f[1]), float(f[2]), float(f[3])))(a.arousal.split(',')) if a.arousal else None)
            holder['data'] = data; holder['stats'] = stats; holder['model'] = model
            return model
        lm_io.build_cns = patched
        import phase3_variants
        original_variant = phase3_variants.variant

        def variant_keeping_loop(model, name):
            # 'h1' swaps model.__class__ to a FlyLM subclass with its own step(): a LoopedFlyLM must keep its step
            if name == 'h1' and hasattr(model, 'add_separate_head'):
                model = model.add_separate_head()
            else:
                model = original_variant(model, name)
            if name == 'h1' and a.init_from:
                saved = torch.load(a.init_from, map_location='cpu', weights_only=True)['model']['state_dict']
                assert torch.equal(saved['core.src'].long(), model.core.src.cpu().long()) and torch.equal(saved['core.dst'].long(), model.core.dst.cpu().long()), 'init-from: different graph'
                for key in ('core.src32', 'core.dst32'):
                    saved.pop(key, None)
                missing, unexpected = model.load_state_dict({k: v for k, v in saved.items()}, strict=False)
                holder['stats']['init_from'] = dict(path=a.init_from, missing=[k for k in missing if not k.startswith('core.src') and not k.startswith('core.dst')][:20], unexpected=list(unexpected)[:20])
                emit('init_from', path=a.init_from, missing=len(missing), unexpected=len(unexpected))
            if name == 'h1' and a.freeze_interfaces:
                model.freeze_interfaces_()
                holder['stats']['freeze_interfaces'] = dict(trainable=[n for n, _ in model.named_parameters()])
            return model
        phase3_variants.variant = variant_keeping_loop
        if a.adam != 'default' or a.type_param_lr is not None or a.optimizer == 'muon' or a.core_lr is not None:
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
                    special = {id(t) for name, t in core.named_parameters() if name in ('leak_logit', 'bias_type', 'group_gain')}
                    typed = [p for p in params if id(p) in special]
                    if not typed:
                        raise RuntimeError('type-param-lr given but the core has no per-type parameters')
                    holder['stats']['type_param_lr'] = dict(lr=float(a.type_param_lr), tensors=len(typed), values=int(sum(p.numel() for p in typed)))
                groups = [(typed, float(a.type_param_lr))] if typed else []
                if a.interface_lr is not None:
                    # E3: every trainable parameter outside the core (and outside the Muon matrices, which keep --muon-lr) gets its own lr
                    model_ = holder['model']
                    core_ids = {id(p) for _, p in model_.core.named_parameters()}
                    from optimizer_variants import matrix_names
                    matrices = matrix_names(model_) if a.optimizer == 'muon' else set()
                    named_ = {id(p): n for n, p in model_.named_parameters()}
                    iface = [p for p in params if id(p) not in core_ids and named_.get(id(p)) not in matrices]
                    groups.append((iface, float(a.interface_lr)))
                    holder['stats']['interface_lr'] = dict(lr=float(a.interface_lr), tensors=len(iface))
                accumulate = None
                if a.core_lr is not None:
                    raw = [p for p in params if p is holder['model'].core.raw]
                    if not raw:
                        raise RuntimeError('core-lr given but core.raw is not a trainable parameter')
                    if a.core_accumulate:
                        if a.optimizer != 'muon':
                            raise RuntimeError('core-accumulate is implemented in HybridMuon only (--optimizer muon)')
                        accumulate = (raw, float(a.core_lr), int(a.core_accumulate))
                        holder['stats']['core_lr'] = dict(lr=float(a.core_lr), values=int(raw[0].numel()), accumulate=int(a.core_accumulate))
                    else:
                        same = [g for g in groups if g[1] == float(a.core_lr)]
                        if same:
                            same[0][0].extend(raw)
                        else:
                            groups.append((raw, float(a.core_lr)))
                        holder['stats']['core_lr'] = dict(lr=float(a.core_lr), values=int(raw[0].numel()))
                elif a.core_accumulate:
                    raise RuntimeError('core-accumulate needs --core-lr')
                if a.optimizer == 'muon':
                    from optimizer_variants import HybridMuon
                    kw.pop('fused', None)
                    opt = HybridMuon(holder['model'], params, kw.pop('lr'), a.muon_lr, include_head=a.muon_head,
                                     adam_groups=groups or None, adam_cls=original_adam, accumulate=accumulate, foreach=False, **kw)
                    holder['stats']['optimizer'] = dict(kind='muon', muon_lr=float(a.muon_lr), matrices=opt.matrix_names,
                                                        matrix_values=int(sum(p.numel() for p in opt.matrices)))
                    return opt
                if groups:
                    special_ids = {id(p) for group, _ in groups for p in group}
                    params = [dict(params=[p for p in params if id(p) not in special_ids])] + [dict(params=group, lr=lr) for group, lr in groups]
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
                                   optimizer=a.optimizer, muon_lr=a.muon_lr if a.optimizer == 'muon' else None,
                                   attn_reads=a.attn_reads, attn_kv=a.attn_kv, attn_step_id=a.attn_step_id, attn_inject=a.attn_inject,
                                   token_injection=a.token_injection, freeze_core=a.freeze_core, core_lr=a.core_lr, core_accumulate=a.core_accumulate, depth_schedule=a.depth_schedule, homeo=a.homeo, arousal=a.arousal,
                                   init_norm=a.init_norm, output_scale=a.output_scale, interface_lr=a.interface_lr, init_from=a.init_from, freeze_interfaces=a.freeze_interfaces,
                                   port_channels=a.port_channels, port_encoder=a.port_encoder, gain_groups=a.gain_groups, ports_seed=a.ports_seed, ports_count=a.ports_count,
                                   stats=holder['stats'],
                                   sources={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CONTROL_SOURCES}),
                      semantic_verdict=f'control run: graph={a.rewire_kind}, ports={a.ports}, readout={a.readout}, core={a.core_variant}; same nodes and init; 2000-update test, not a model change')
    except Exception as exc:
        traceback.print_exc(); result = dict(ok=False, error=str(exc), error_type=type(exc).__name__)
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result, indent=2))
    if not result['ok']:
        sys.exit(1)


if __name__ == '__main__':
    main()

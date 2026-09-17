"""Control-code variants of the text interfaces (fly_interfaces.py untouched), phase 8 section Q: input channels by
modality (user proposal of 18 September 2026).

modality_index      per-node modality id of the anatomical sensory ports: 0 = not a port, then vista, meccano/udito,
                    olfatto, propriocezione, gusto, termo/igro/chemo, altro (unknown sensory)
modal_channels_     P1: the 256 embedding channels are split into one block per modality and every port draws its
                    fan-in channels only from the block of its modality (labelled lines: what goes in through the eyes
                    is a different part of the token vector from what goes in through the antennae)
TypeSharedInjector  P2: all ports of the same sensory cell type share channels, weight and bias: one calibrated channel
                    per type (369 types among the 17,937 ports; R1-R6 photoreceptors = one channel), like a glomerulus
convergent_nodes    P4: the neurons reached by every sensory modality within `hops` synapses (multisensory convergence
                    zones, e.g. oviIN, MBON32, AVLP080, pIP1), used as input ports instead of the sensory neurons"""
import numpy as np, torch
from torch import nn

MODALITIES = ('vista', 'meccano/udito', 'olfatto', 'propriocezione', 'gusto', 'termo/igro/chemo', 'altro')


def modality_masks(cls, sup, sensory_mask):
    m = {
        'vista': (sup == 'ol_sensory') | ((cls == 'visual') & sensory_mask),
        'meccano/udito': np.isin(cls, ['mechanosensory', 'mechanosensory_tbc', 'mechanosensory_tactile']) & sensory_mask,
        'olfatto': (cls == 'olfactory') & sensory_mask,
        'propriocezione': (cls == 'mechanosensory_proprioceptive') & sensory_mask,
        'gusto': (cls == 'gustatory') & sensory_mask,
        'termo/igro/chemo': np.isin(cls, ['thermosensory', 'hygrosensory', 'chemosensory']) & sensory_mask,
    }
    covered = np.zeros_like(sensory_mask)
    for v in m.values():
        covered |= v
    m['altro'] = sensory_mask & ~covered
    return m


def modality_index(cls, sup, sensory):
    n = len(cls); sensory_mask = np.zeros(n, bool); sensory_mask[np.asarray(sensory)] = True
    index = np.zeros(n, np.int64)
    for k, name in enumerate(MODALITIES, start=1):
        index[modality_masks(cls, sup, sensory_mask)[name]] = k
    return index


def modal_channels_(injector, port_modality, dim, fan_in, seed):
    """Overwrite injector.channels so that each port only reads the channel block of its modality."""
    port_modality = torch.as_tensor(port_modality, dtype=torch.long)
    groups = int(port_modality.max()) + 1
    block = dim // (groups - 1) if groups > 1 else dim
    if block < fan_in:
        raise ValueError(f'{groups - 1} modality blocks of {block} channels cannot hold fan-in {fan_in}')
    generator = torch.Generator().manual_seed(seed)
    channels = injector.channels.clone()
    info = {}
    for g in range(1, groups):
        ports = torch.nonzero(port_modality == g).flatten()
        if ports.numel() == 0:
            continue
        start = (g - 1) * block; width = block if g < groups - 1 else dim - start
        perm = start + torch.randperm(width, generator=generator)
        channels[ports] = perm[torch.arange(ports.numel() * fan_in).reshape(ports.numel(), fan_in) % width]
        info[MODALITIES[g - 1]] = dict(ports=int(ports.numel()), channels=[int(start), int(start + width)])
    injector.channels.copy_(channels)
    return info


class TypeSharedInjector(nn.Module):
    def __init__(self, base, group_of_port, seed):
        super().__init__()
        group_of_port = torch.as_tensor(group_of_port, dtype=torch.long)
        groups = int(group_of_port.max()) + 1
        fan_in = base.channels.shape[1]; dim = base.dim
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(dim, generator=generator)
        self.register_buffer('nodes', base.nodes.clone())
        self.register_buffer('group', group_of_port)
        self.register_buffer('channels', permutation[torch.arange(groups * fan_in).reshape(groups, fan_in) % dim])
        self.weight = nn.Parameter(torch.randn(groups, fan_in, generator=generator) / fan_in ** .5)
        self.bias = nn.Parameter(torch.full((groups,), float(base.bias.detach().mean()))) if base.bounded else None
        self.n, self.dim, self.gain, self.bounded, self.groups = base.n, dim, base.gain, base.bounded, groups

    def forward(self, vector):
        values = (vector[:, self.channels[self.group]] * self.weight[self.group]).sum(-1)
        if self.bounded:
            values = self.bias[self.group] + self.gain * torch.tanh(values)
        return vector.new_zeros(vector.shape[0], self.n).index_copy(1, self.nodes, values)


def convergent_nodes(src, dst, n, cls, sup, sensory, hops=3):
    sensory_mask = np.zeros(n, bool); sensory_mask[np.asarray(sensory)] = True
    masks = modality_masks(cls, sup, sensory_mask)
    src_t = torch.as_tensor(src, dtype=torch.long); dst_t = torch.as_tensor(dst, dtype=torch.long)
    reached_all = np.ones(n, bool)
    for name in MODALITIES[:-1]:
        dist = torch.full((n,), 99, dtype=torch.long); frontier = torch.from_numpy(masks[name].copy()); dist[frontier] = 0
        for h in range(1, hops + 1):
            reach = torch.zeros(n, dtype=torch.bool); reach[dst_t[frontier[src_t]]] = True
            new = reach & (dist == 99); dist[new] = h; frontier = new
            if not new.any():
                break
        reached_all &= (dist <= hops).numpy()
    return np.flatnonzero(reached_all)

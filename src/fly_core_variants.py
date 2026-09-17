"""Core variants for the phase-6b tests D4-D7 and the phase-7c levers (SUITE_TEST_FASE_6B.md). Same interface as
fly_core.Core; only the substep dynamics in advance() differ. fly_core.py itself is untouched (checkpoint fingerprints
intact). `variant` is one flag or several joined by '+', e.g. 'bias_type+tau_type+reversal'.

apl        k-winner-take-all on the Kenyon cells: at every substep only the 5% of KC with the highest potential may spike.
graded_ol  optic-lobe intrinsic neurons emit a graded activation sigmoid(4(u-1)) instead of a binary spike.
tau_type   the leak factor 0.95 becomes a trainable parameter per cell type (sigmoid of a logit, init 0.95).
reversal   conductance-like synapses: excitatory drive scaled by (E_e - v)/E_e, inhibitory by (v - E_i)/(-E_i),
           E_e = 3, E_i = -1 in threshold units; identical to the LIF at v = 0.
bias_type  a trainable resting drive (bias) per cell type added to the potential at every substep, init 0 (identical
           model at init); the flyvis-style "excitability per type" lever (RICERCA_ATTIVITA_CERVELLO_MOSCA.md).
"""
import torch
from fly_core import Core, Propagate, LIFReset


class PropagateFloat(torch.autograd.Function):
    """Propagate with float (graded) presynaptic activations saved for the weight gradient."""
    @staticmethod
    def forward(ctx, s, w, src, dst, chunk):
        out = torch.zeros_like(s)
        for i in range(0, w.numel(), chunk):
            sl = slice(i, i + chunk)
            out.index_add_(1, dst[sl], s[:, src[sl]] * w[sl])
        ctx.save_for_backward(s, w, src, dst)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        s, w, src, dst = ctx.saved_tensors
        gs, gw = torch.zeros_like(g), torch.empty_like(w)
        for i in range(0, w.numel(), ctx.chunk):
            sl = slice(i, i + ctx.chunk)
            gd = g[:, dst[sl]]
            gw[sl] = (gd * s[:, src[sl]]).sum(0)
            gs.index_add_(1, src[sl], gd * w[sl])
        return gs, gw, None, None, None


VARIANTS = ('lif', 'apl', 'graded_ol', 'tau_type', 'reversal', 'bias_type')


def parse_variant(variant):
    flags = [f for f in str(variant).split('+') if f]
    if not flags:
        raise ValueError('empty variant')
    for f in flags:
        if f not in VARIANTS:
            raise ValueError(f'variant flags must be among {VARIANTS}, got {f!r}')
    return set(flags) - {'lif'}


class VariantCore(Core):
    def __init__(self, src, dst, magnitude, signs, n, variant, kc=None, ol=None, type_index=None, chunk=262144):
        super().__init__(src, dst, magnitude, signs, n, chunk)
        self.variant = variant
        self.flags = parse_variant(variant)
        if 'apl' in self.flags:
            if kc is None or kc.numel() == 0:
                raise ValueError('apl variant needs Kenyon cell indices')
            self.register_buffer('kc', kc.to(torch.long))
            self.k = max(1, int(round(.05 * kc.numel())))
        if 'graded_ol' in self.flags:
            if ol is None or ol.numel() == 0:
                raise ValueError('graded_ol variant needs optic-lobe indices')
            self.register_buffer('ol', ol.to(torch.long))
        if self.flags & {'tau_type', 'bias_type'}:
            if type_index is None:
                raise ValueError('tau_type / bias_type variants need a type index per node')
            self.register_buffer('type_index', type_index.to(torch.long))
            types = int(type_index.max()) + 1
            if 'tau_type' in self.flags:
                self.leak_logit = torch.nn.Parameter(torch.full((types,), 2.9444389791664403))  # sigmoid -> 0.95
            if 'bias_type' in self.flags:
                self.bias_type = torch.nn.Parameter(torch.zeros(types))
        if 'reversal' in self.flags:
            self.e_rev, self.i_rev = 3., -1.

    def propagate(self, s, w):
        if 'graded_ol' in self.flags:
            return PropagateFloat.apply(s, w, self.src, self.dst, self.chunk)
        return Propagate.apply(s, w, self.src, self.dst, self.chunk)

    def describe(self):
        d = dict(variant=self.variant)
        if 'apl' in self.flags:
            d.update(kc=int(self.kc.numel()), winners=self.k)
        if 'graded_ol' in self.flags:
            d.update(graded_nodes=int(self.ol.numel()))
        if 'tau_type' in self.flags:
            d.update(types=int(self.leak_logit.numel()))
        if 'bias_type' in self.flags:
            d.update(bias_types=int(self.bias_type.numel()))
        if 'reversal' in self.flags:
            d.update(e_rev=self.e_rev, i_rev=self.i_rev)
        return d

    def advance(self, current, steps=1, state=None, *, weights=None, active=None, reset=None, reference=False):
        if not isinstance(steps, int) or steps < 1:
            raise ValueError('steps must be a positive integer')
        if (current.ndim != 2 or current.shape[1] != self.n or current.shape[0] < 1
                or current.device != self.raw.device or current.dtype != self.raw.dtype):
            raise ValueError('current must be [batch,n] with core dtype/device')
        batch = current.shape[0]
        self._check_mask(active, batch)
        self._check_mask(reset, batch)
        if state is None:
            state = self.initial_state(batch)
        if len(state) != 2 or any(x.shape != current.shape or x.device != current.device
                                 or x.dtype != current.dtype for x in state):
            raise ValueError('state must be (voltage, spike), matching current')
        v, s = self.reset_state(state, reset)
        w = self.weights() if weights is None else weights
        if w.shape != self.raw.shape or w.dtype != self.raw.dtype or w.device != self.raw.device:
            raise ValueError('effective weights must match core raw shape/dtype/device')
        rate = torch.zeros_like(current)
        if 'tau_type' in self.flags:
            leak = torch.sigmoid(self.leak_logit)[self.type_index][None, :]
        else:
            leak = .95
        drive = current
        if 'bias_type' in self.flags:
            drive = current + self.bias_type[self.type_index][None, :]
        if 'reversal' in self.flags:
            w_exc, w_inh = w.clamp(min=0), (-w).clamp(min=0)
        for _ in range(steps):
            if 'reversal' in self.flags:
                ge = self.propagate(s, w_exc)
                gi = self.propagate(s, w_inh)
                syn = ge * (self.e_rev - v) / self.e_rev - gi * (v - self.i_rev) / (-self.i_rev)
            else:
                syn = self.propagate(s, w)
            u = leak * v + syn + drive
            if 'apl' in self.flags:
                ukc = u[:, self.kc]
                threshold = ukc.topk(self.k, dim=1).values[:, -1:]
                ukc = torch.where(ukc >= threshold, ukc, ukc.clamp(max=.99))
                u = u.index_copy(1, self.kc, ukc)
            next_v, next_s = LIFReset.apply(u)
            if 'graded_ol' in self.flags:
                uo = u[:, self.ol]
                so = torch.sigmoid(4 * (uo - 1))
                next_s = next_s.index_copy(1, self.ol, so)
                next_v = next_v.index_copy(1, self.ol, uo * (1 - so))
            if active is not None:
                mask = active[:, None]
                next_v = torch.where(mask, next_v, v)
                next_s = torch.where(mask, next_s, s)
                rate = rate + torch.where(mask, next_s, torch.zeros_like(next_s))
            else:
                rate = rate + next_s
            v, s = next_v, next_s
        return (v, s), rate / steps

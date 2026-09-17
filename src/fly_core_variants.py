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
cond_ports conductance-like injection of the external drive (token current + attention feedback): the positive part is
           scaled by (E_e - v)/E_e, the negative part by (v - E_i)/(-E_i); identical to the LIF at v = 0 (phase 8, M4).
soft_gw    wider weight gradient (phase 8, C16): forward unchanged (binary spikes), but in the backward pass the weight
           gradient uses a SOFT presynaptic activity max(s, sigmoid(4(u-1))) instead of the binary spike, so synapses
           from neurons that were near threshold also receive gradient (with the binary rule only the synapses of
           neurons that fired are trained: 19% of the edges, C13w). The first substep of every advance() call uses
           the binary spikes of the incoming state.
homeo      wake-up homeostasis (phase 8, user + assistant brainstorming of 17 September 2026): every cell type keeps a
           gradient-free threshold offset (threshold = 1 - offset, offset in [0, 0.9], init 0 = the LIF). A running mean
           of the type's firing rate is compared with a target: types below target slowly lower their threshold, types
           above target give the offset back ten times more slowly; nothing is pushed below the LIF excitability.
           Buffers are updated only in training forwards (grad enabled), so evaluation uses them frozen. homeo=(target, eta).
arousal    open-loop global excitability modulation (the fly's arousal / octopamine gain states): a threshold offset common
           to every neuron follows a schedule over training updates read from a GPU token counter:
           arousal=(shape, amplitude, period_updates, duty_updates); 'pulse' = amplitude during the last duty updates of
           each period, 'smooth' = amplitude * sin^2(pi * phase / period), no abrupt change.
gain_group a trainable gain per anatomical group (superclass) on the external drive, init 1 (identical model at init);
           the "gain per region" lever (phase 8, 2.7.2). Needs group_index; train it with --type-param-lr.
"""
import math
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


class PropagateSoft(torch.autograd.Function):
    """Propagate with binary spikes forward; the weight gradient uses the soft presynaptic activity p (no gradient to p)."""
    @staticmethod
    def forward(ctx, s, p, w, src, dst, chunk):
        out = torch.zeros_like(s)
        for i in range(0, w.numel(), chunk):
            sl = slice(i, i + chunk)
            out.index_add_(1, dst[sl], s[:, src[sl]] * w[sl])
        ctx.save_for_backward(p, w, src, dst)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        p, w, src, dst = ctx.saved_tensors
        gs, gw = torch.zeros_like(g), torch.empty_like(w)
        for i in range(0, w.numel(), ctx.chunk):
            sl = slice(i, i + ctx.chunk)
            gd = g[:, dst[sl]]
            gw[sl] = (gd * p[:, src[sl]]).sum(0)
            gs.index_add_(1, src[sl], gd * w[sl])
        return gs, None, gw, None, None, None


VARIANTS = ('lif', 'apl', 'graded_ol', 'tau_type', 'reversal', 'bias_type', 'cond_ports', 'gain_group', 'soft_gw', 'homeo', 'arousal')
# bias_type reparametrisation (phase 7f, 17 September 04:20): bias = BIAS_SCALE * parameter. The raw bias gradient sums
# over every substep and every neuron of a type, dominated the global norm clip (1.0) and starved the other parameters
# (H1: grad norm 10-13 vs 4-5). Adam is scale-invariant per parameter, so only the clip contribution changes; use a
# dedicated lr of 5e-3 to keep the step in threshold units at 1e-4. Runs before this date used BIAS_SCALE = 1.
BIAS_SCALE = 0.02


def parse_variant(variant):
    flags = [f for f in str(variant).split('+') if f]
    if not flags:
        raise ValueError('empty variant')
    for f in flags:
        if f not in VARIANTS:
            raise ValueError(f'variant flags must be among {VARIANTS}, got {f!r}')
    return set(flags) - {'lif'}


class VariantCore(Core):
    def __init__(self, src, dst, magnitude, signs, n, variant, kc=None, ol=None, type_index=None, chunk=262144, group_index=None,
                 homeo=None, arousal=None, tokens_per_update=16):
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
        if self.flags & {'tau_type', 'bias_type', 'homeo'}:
            if type_index is None:
                raise ValueError('tau_type / bias_type variants need a type index per node')
            self.register_buffer('type_index', type_index.to(torch.long))
            types = int(type_index.max()) + 1
            if 'tau_type' in self.flags:
                self.leak_logit = torch.nn.Parameter(torch.full((types,), 2.9444389791664403))  # sigmoid -> 0.95
            if 'bias_type' in self.flags:
                self.bias_type = torch.nn.Parameter(torch.zeros(types))
        if self.flags & {'reversal', 'cond_ports'}:
            self.e_rev, self.i_rev = 3., -1.
        if 'homeo' in self.flags:
            self.homeo_target, self.homeo_eta = (float(x) for x in (homeo or (0.02, 2e-5)))
            types = int(self.type_index.max()) + 1
            self.register_buffer('type_count', torch.bincount(self.type_index, minlength=types).clamp_min(1).float())
            self.register_buffer('homeo_rate', torch.full((types,), self.homeo_target))
            self.register_buffer('thr_offset', torch.zeros(types))
        if 'arousal' in self.flags:
            shape, amplitude, period, duty = arousal or ('pulse', 0.3, 500, 100)
            if shape not in ('pulse', 'smooth'):
                raise ValueError('arousal shape must be pulse or smooth')
            self.arousal = (shape, float(amplitude), float(period), float(duty)); self.tokens_per_update = float(tokens_per_update)
            self.register_buffer('tokens', torch.zeros((), dtype=torch.long))
        if 'soft_gw' in self.flags and 'graded_ol' in self.flags:
            raise ValueError('soft_gw and graded_ol cannot be combined')
        if 'gain_group' in self.flags:
            if group_index is None:
                raise ValueError('gain_group variant needs a group index per node')
            self.register_buffer('group_index', group_index.to(torch.long))
            self.group_gain = torch.nn.Parameter(torch.ones(int(group_index.max()) + 1))

    def propagate(self, s, w):
        if 'graded_ol' in self.flags:
            return PropagateFloat.apply(s, w, self.src, self.dst, self.chunk)
        return Propagate.apply(s, w, self.src, self.dst, self.chunk)

    def arousal_level(self):
        shape, amplitude, period, duty = self.arousal
        phase = torch.remainder(self.tokens.float() / self.tokens_per_update, period)
        if shape == 'pulse':
            return amplitude * (phase >= period - duty).float()
        return amplitude * torch.sin(math.pi * phase / period) ** 2

    def propagate_soft(self, s, p, w):
        return PropagateSoft.apply(s, p, w, self.src, self.dst, self.chunk)

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
        if self.flags & {'reversal', 'cond_ports'}:
            d.update(e_rev=self.e_rev, i_rev=self.i_rev)
        if 'gain_group' in self.flags:
            d.update(gain_groups=int(self.group_gain.numel()))
        if 'homeo' in self.flags:
            d.update(homeo_target=self.homeo_target, homeo_eta=self.homeo_eta)
        if 'arousal' in self.flags:
            d.update(arousal=list(self.arousal))
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
        if 'gain_group' in self.flags:
            drive = drive * self.group_gain[self.group_index][None, :]
        if 'cond_ports' in self.flags:
            drive_exc, drive_inh = drive.clamp(min=0), (-drive).clamp(min=0)
        if 'bias_type' in self.flags:
            drive = drive + BIAS_SCALE * self.bias_type[self.type_index][None, :]
        if 'reversal' in self.flags:
            w_exc, w_inh = w.clamp(min=0), (-w).clamp(min=0)
        soft = 'soft_gw' in self.flags
        p = s
        offset = None
        if 'homeo' in self.flags:
            offset = self.thr_offset[self.type_index][None, :]
        if 'arousal' in self.flags:
            level = self.arousal_level()
            offset = level if offset is None else offset + level
        for _ in range(steps):
            if 'reversal' in self.flags:
                ge = self.propagate_soft(s, p, w_exc) if soft else self.propagate(s, w_exc)
                gi = self.propagate_soft(s, p, w_inh) if soft else self.propagate(s, w_inh)
                syn = ge * (self.e_rev - v) / self.e_rev - gi * (v - self.i_rev) / (-self.i_rev)
            else:
                syn = self.propagate_soft(s, p, w) if soft else self.propagate(s, w)
            if 'cond_ports' in self.flags:
                # the per-type bias (if any) stays a plain current; only the external drive is conductance-like
                external = drive_exc * (self.e_rev - v) / self.e_rev - drive_inh * (v - self.i_rev) / (-self.i_rev)
                u = leak * v + syn + external + (drive - (drive_exc - drive_inh))
            else:
                u = leak * v + syn + drive
            if 'apl' in self.flags:
                ukc = u[:, self.kc]
                threshold = ukc.topk(self.k, dim=1).values[:, -1:]
                ukc = torch.where(ukc >= threshold, ukc, ukc.clamp(max=.99))
                u = u.index_copy(1, self.kc, ukc)
            if offset is not None:
                # threshold 1 - offset without touching fly_core: shift the potential, spike, give the shift back to the silent ones
                u = u + offset
                next_v, next_s = LIFReset.apply(u)
                next_v = next_v - offset * (1 - next_s)
            else:
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
            if soft:
                p = torch.maximum(next_s, torch.sigmoid(4 * (u - 1))).detach()
            v, s = next_v, next_s
        if torch.is_grad_enabled():   # training forwards only (captured in the training CUDA Graph; evaluation runs under no_grad)
            with torch.no_grad():
                if 'homeo' in self.flags:
                    per_node = (rate / steps).mean(0)
                    per_type = torch.zeros_like(self.homeo_rate).index_add_(0, self.type_index, per_node) / self.type_count
                    self.homeo_rate.mul_(0.99).add_(0.01 * per_type)
                    error = (self.homeo_target - self.homeo_rate) / self.homeo_target
                    self.thr_offset.add_(self.homeo_eta * torch.where(error > 0, error, 0.1 * error)).clamp_(0., 0.9)
                if 'arousal' in self.flags and reset is not None:
                    self.tokens.add_(1)   # the first advance() of every token carries the reset mask
        return (v, s), rate / steps

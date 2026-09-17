"""Sparse LIF core. First-order surrogate gradients; binary recurrent spikes."""
import torch
from torch.utils.checkpoint import checkpoint


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > 1).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        p = torch.sigmoid(4 * (u - 1))
        return g * 4 * p * (1 - p)


class LIFReset(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        s = (u > 1).to(u.dtype)
        return u * (1 - s), s

    @staticmethod
    def backward(ctx, gv, gs):
        (u,) = ctx.saved_tensors
        s = (u > 1).to(u.dtype)
        p = torch.sigmoid(4 * (u - 1))
        return gv * (1 - s) + (gs - gv * u) * (4 * p * (1 - p))


class Propagate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, w, src, dst, chunk):
        out = torch.zeros_like(s)
        for i in range(0, w.numel(), chunk):
            sl = slice(i, i + chunk)
            out.index_add_(1, dst[sl], s[:, src[sl]] * w[sl])
        ctx.save_for_backward(s.bool(), w, src, dst)
        ctx.chunk = chunk
        return out

    @staticmethod
    def backward(ctx, g):
        sb, w, src, dst = ctx.saved_tensors
        gs, gw = torch.zeros_like(g), torch.empty_like(w)
        for i in range(0, w.numel(), ctx.chunk):
            sl = slice(i, i + ctx.chunk)
            gd = g[:, dst[sl]]
            gw[sl] = (gd * sb[:, src[sl]].to(g.dtype)).sum(0)
            gs.index_add_(1, src[sl], gd * w[sl])
        return gs, gw, None, None, None


class Core(torch.nn.Module):
    def __init__(self, src, dst, magnitude, signs, n, chunk=262144):
        super().__init__()
        self.n, self.chunk = n, chunk
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)
        self.register_buffer('signs', signs[src])
        # Stable inverse softplus. Positive magnitude throughout training.
        self.raw = torch.nn.Parameter(magnitude + torch.log(-torch.expm1(-magnitude)))

    def weights(self):
        return torch.nn.functional.softplus(self.raw) * self.signs

    def initial_state(self, batch):
        """Fresh (voltage, binary spike), on the core's device and dtype."""
        if not isinstance(batch, int) or batch < 1:
            raise ValueError('batch must be a positive integer')
        return self.raw.new_zeros(batch, self.n), self.raw.new_zeros(batch, self.n)

    @staticmethod
    def detach_state(state):
        """TBPTT boundary; retains values, cuts history. No in-place writes allowed."""
        return tuple(x.detach() for x in state)

    def _check_mask(self, mask, batch):
        if mask is not None and (mask.shape != (batch,) or mask.dtype != torch.bool
                                 or mask.device != self.raw.device):
            raise ValueError('mask must be bool[batch] on the core device')

    def reset_state(self, state, reset):
        """Functional per-example story reset, including gradients to old state."""
        self._check_mask(reset, state[0].shape[0])
        if reset is None:
            return state
        return tuple(torch.where(reset[:, None], torch.zeros_like(x), x) for x in state)

    def advance(self, current, steps=1, state=None, *, weights=None,
                active=None, reset=None, reference=False):
        """Pure substeps: constant current[B,N] -> ((v,s), mean_spike_rate[B,N]).

        No loss/cache mutation/detach. Reset happens before stepping; inactive
        slots then preserve state and report zero rate. Supplied effective
        weights can be shared within one forward, never across optimizer steps.
        Caller must supply binary spike state and finite inputs. First-order
        surrogate derivatives only. Compose calls for pre/post attention phases.
        """
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
        for _ in range(steps):
            if reference:
                syn = torch.zeros_like(s).index_add(1, self.dst, s[:, self.src] * w)
            else:
                syn = Propagate.apply(s, w, self.src, self.dst, self.chunk)
            u = .95 * v + syn + current
            if reference:
                next_s = Spike.apply(u)
                next_v = u * (1 - next_s)
            else:
                next_v, next_s = LIFReset.apply(u)
            if active is not None:
                mask = active[:, None]
                next_v = torch.where(mask, next_v, v)
                next_s = torch.where(mask, next_s, s)
                rate = rate + torch.where(mask, next_s, torch.zeros_like(next_s))
            else:
                rate = rate + next_s
            v, s = next_v, next_s
        return (v, s), rate / steps

    def window(self, drive, steps=8, state=None, reference=False, checkpoint_chars=0):
        b = drive.shape[1]
        v, s = self.initial_state(b) if state is None else state
        w = self.weights()

        def segment(v, s, currents, weights):
            loss = currents.new_zeros(())
            activity = currents.new_zeros(())
            for current in currents:
                (v, s), rate = self.advance(current, steps, (v, s), weights=weights,
                                           reference=reference)
                loss = loss + (rate - .15).square().mean()
                activity = activity + rate.detach().mean()
            return v, s, loss, activity

        loss, activity = drive.new_zeros(()), drive.new_zeros(())
        width = checkpoint_chars or len(drive)
        for i in range(0, len(drive), width):
            args = (v, s, drive[i:i + width], w)
            if checkpoint_chars:
                v, s, part, count = checkpoint(segment, *args, use_reentrant=False)
            else:
                v, s, part, count = segment(*args)
            loss, activity = loss + part, activity + count
        return loss / len(drive), activity / len(drive), (v, s)

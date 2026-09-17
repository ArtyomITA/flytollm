"""Speed variants of the sparse LIF core (SUITE_TEST_FASE_6B.md, category S). Same mathematics as fly_core.Core:
only the propagation kernel, the index dtype and the chunking change. fly_core.py is untouched.

modes
  gather   gather + index_add (as fly_core.Propagate) with configurable chunk and index dtype   (S1, S2)
  fp16     weights and spikes stored in half for the gather, products accumulated in float      (S5)
  csr      cuSPARSE SpMM on a CSR matrix ordered by destination; backward with the transposed CSR
           (ordered by source) and an element-wise gather for the weight gradient                 (S3)
"""
import numpy
import torch
from fly_core import Core, LIFReset

MODES = ('gather', 'fp16', 'csr')


class PropagateGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, w, src, dst, chunk, half):
        out = torch.zeros_like(s)
        if half:
            s16, w16 = s.half(), w.half()
            for i in range(0, w.numel(), chunk):
                sl = slice(i, i + chunk)
                out.index_add_(1, dst[sl], (s16[:, src[sl]] * w16[sl]).float())
        else:
            for i in range(0, w.numel(), chunk):
                sl = slice(i, i + chunk)
                out.index_add_(1, dst[sl], s[:, src[sl]] * w[sl])
        ctx.save_for_backward(s.bool(), w, src, dst)
        ctx.chunk, ctx.half = chunk, half
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
        return gs, gw, None, None, None, None


class PropagateCSR(torch.autograd.Function):
    """out = A s  with A[dst, src] = w.  crow/col describe A in CSR (rows = dst); crow_t/col_t describe A^T (rows = src).
    perm maps CSR value order -> edge order (values = w[perm]); perm_t likewise for A^T."""
    @staticmethod
    def forward(ctx, s, w, crow, col, perm, crow_t, col_t, perm_t, src, dst):
        n = s.shape[1]
        a = torch.sparse_csr_tensor(crow, col, w[perm], size=(n, n))
        out = torch.sparse.mm(a, s.t().contiguous()).t().contiguous()
        ctx.save_for_backward(s.bool(), w, crow_t, col_t, perm_t, src, dst)
        return out

    @staticmethod
    def backward(ctx, g):
        sb, w, crow_t, col_t, perm_t, src, dst = ctx.saved_tensors
        n = g.shape[1]
        at = torch.sparse_csr_tensor(crow_t, col_t, w[perm_t], size=(n, n))
        gs = torch.sparse.mm(at, g.t().contiguous()).t().contiguous()
        gw = (g[:, dst] * sb[:, src].to(g.dtype)).sum(0)
        return gs, gw, None, None, None, None, None, None, None, None


class FastCore(Core):
    def __init__(self, src, dst, magnitude, signs, n, mode='gather', chunk=262144, index_dtype=torch.long):
        super().__init__(src.to(index_dtype), dst.to(index_dtype), magnitude, signs, n, chunk)
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode = mode
        self.index_dtype = index_dtype
        if mode == 'csr':
            s64, d64 = src.long(), dst.long()
            order = torch.argsort(d64 * n + s64)            # CSR by destination
            order_t = torch.argsort(s64 * n + d64)          # CSR of the transpose, by source
            self.register_buffer('perm', order)
            self.register_buffer('col', s64[order].to(torch.int32))
            self.register_buffer('crow', torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(d64, minlength=n)]), 0).to(torch.int32))
            self.register_buffer('perm_t', order_t)
            self.register_buffer('col_t', d64[order_t].to(torch.int32))
            self.register_buffer('crow_t', torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(s64, minlength=n)]), 0).to(torch.int32))
            self.register_buffer('src64', s64)
            self.register_buffer('dst64', d64)

    def describe(self):
        return dict(core='FastCore', mode=self.mode, chunk=self.chunk, index_dtype=str(self.index_dtype))

    def propagate(self, s, w):
        if self.mode == 'csr':
            return PropagateCSR.apply(s, w, self.crow, self.col, self.perm, self.crow_t, self.col_t, self.perm_t, self.src64, self.dst64)
        return PropagateGather.apply(s, w, self.src, self.dst, self.chunk, self.mode == 'fp16')

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
        for _ in range(steps):
            syn = self.propagate(s, w)
            u = .95 * v + syn + current
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


# ----------------------------------------------------------------------------------------------------------------------
# S6: fused gather-multiply-scatter CUDA kernels (cupy RawKernel), edge-parallel, int32 indices, spike-driven skips.
_FUSED_SRC = r'''
extern "C" __global__ void prop_fwd(const float* __restrict__ s, const float* __restrict__ w,
                                    const int* __restrict__ src, const int* __restrict__ dst,
                                    float* __restrict__ out, const int E, const int B, const int N) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int a = src[e], d = dst[e];
    float we = 0.f; bool loaded = false;
    for (int b = 0; b < B; ++b) {
        float sv = s[(size_t)b * N + a];
        if (sv != 0.f) {
            if (!loaded) { we = w[e]; loaded = true; }
            atomicAdd(&out[(size_t)b * N + d], sv * we);
        }
    }
}
extern "C" __global__ void prop_bwd(const float* __restrict__ s, const float* __restrict__ w,
                                    const int* __restrict__ src, const int* __restrict__ dst,
                                    const float* __restrict__ g, float* __restrict__ gs, float* __restrict__ gw,
                                    const int E, const int B, const int N) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    int a = src[e], d = dst[e];
    float we = w[e]; float acc = 0.f;
    for (int b = 0; b < B; ++b) {
        float gd = g[(size_t)b * N + d];
        if (gd != 0.f) {
            acc += gd * s[(size_t)b * N + a];
            atomicAdd(&gs[(size_t)b * N + a], gd * we);
        }
    }
    gw[e] = acc;
}
'''
_kernels = {}


def _fused_kernels():
    if not _kernels:
        import cupy
        module = cupy.RawModule(code=_FUSED_SRC, options=('-std=c++11',))
        _kernels['fwd'] = module.get_function('prop_fwd')
        _kernels['bwd'] = module.get_function('prop_bwd')
    return _kernels


def _ptr(t):
    return t.data_ptr()


def _launch(kernel, args, n_threads):
    """Launch on torch's current stream so CUDA Graph capture sees it. Arguments are raw pointers/ints."""
    import cupy
    block = 256
    grid = (n_threads + block - 1) // block
    stream = cupy.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)
    with stream:
        kernel((grid,), (block,), args, stream=stream)


class PropagateFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, w, src32, dst32):
        s = s.contiguous(); w = w.contiguous()
        out = torch.zeros_like(s)
        B, N = s.shape; E = w.numel()
        k = _fused_kernels()
        import cupy
        _launch(k['fwd'], (numpy.uintp(s.data_ptr()),
                           numpy.uintp(w.data_ptr()),
                           numpy.uintp(src32.data_ptr()),
                           numpy.uintp(dst32.data_ptr()),
                           numpy.uintp(out.data_ptr()),
                           cupy.int32(E), cupy.int32(B), cupy.int32(N)), E)
        ctx.save_for_backward(s, w, src32, dst32)
        return out

    @staticmethod
    def backward(ctx, g):
        s, w, src32, dst32 = ctx.saved_tensors
        g = g.contiguous()
        gs = torch.zeros_like(g); gw = torch.empty_like(w)
        B, N = g.shape; E = w.numel()
        k = _fused_kernels()
        import cupy
        _launch(k['bwd'], (numpy.uintp(s.data_ptr()),
                           numpy.uintp(w.data_ptr()),
                           numpy.uintp(src32.data_ptr()),
                           numpy.uintp(dst32.data_ptr()),
                           numpy.uintp(g.data_ptr()),
                           numpy.uintp(gs.data_ptr()),
                           numpy.uintp(gw.data_ptr()),
                           cupy.int32(E), cupy.int32(B), cupy.int32(N)), E)
        return gs, gw, None, None


class PropagateFusedSoft(torch.autograd.Function):
    """Fused propagation with binary spikes forward; the backward kernel receives the soft presynaptic activity p in
    place of the spikes, so gw = sum_b g[b, dst] * p[b, src] (phase 8, C16). gs does not depend on the activity."""
    @staticmethod
    def forward(ctx, s, p, w, src32, dst32):
        s = s.contiguous(); w = w.contiguous(); p = p.contiguous()
        out = torch.zeros_like(s)
        B, N = s.shape; E = w.numel()
        k = _fused_kernels()
        import cupy
        _launch(k['fwd'], (numpy.uintp(s.data_ptr()), numpy.uintp(w.data_ptr()), numpy.uintp(src32.data_ptr()), numpy.uintp(dst32.data_ptr()),
                           numpy.uintp(out.data_ptr()), cupy.int32(E), cupy.int32(B), cupy.int32(N)), E)
        ctx.save_for_backward(p, w, src32, dst32)
        return out

    @staticmethod
    def backward(ctx, g):
        p, w, src32, dst32 = ctx.saved_tensors
        g = g.contiguous()
        gs = torch.zeros_like(g); gw = torch.empty_like(w)
        B, N = g.shape; E = w.numel()
        k = _fused_kernels()
        import cupy
        _launch(k['bwd'], (numpy.uintp(p.data_ptr()), numpy.uintp(w.data_ptr()), numpy.uintp(src32.data_ptr()), numpy.uintp(dst32.data_ptr()),
                           numpy.uintp(g.data_ptr()), numpy.uintp(gs.data_ptr()), numpy.uintp(gw.data_ptr()),
                           cupy.int32(E), cupy.int32(B), cupy.int32(N)), E)
        return gs, None, gw, None, None


class FusedCore(FastCore):
    """FastCore with the fused cupy kernels (mode 'fused')."""
    def __init__(self, src, dst, magnitude, signs, n, chunk=262144):
        super().__init__(src, dst, magnitude, signs, n, mode='gather', chunk=chunk)
        self.mode = 'fused'
        self.register_buffer('src32', src.to(torch.int32))
        self.register_buffer('dst32', dst.to(torch.int32))

    def describe(self):
        return dict(core='FusedCore', mode='fused', kernel='cupy prop_fwd/prop_bwd', index_dtype='int32')

    def propagate(self, s, w):
        return PropagateFused.apply(s, w, self.src32, self.dst32)


class FusedVariantCore(__import__('fly_core_variants').VariantCore):
    """VariantCore (apl / graded_ol / tau_type / reversal) with the fused cupy propagation kernel. The kernel reads float
    presynaptic activations, so graded units work unchanged; reversal calls it twice (excitatory / inhibitory)."""
    def __init__(self, src, dst, magnitude, signs, n, variant, **kw):
        super().__init__(src, dst, magnitude, signs, n, variant, **kw)
        self.register_buffer('src32', src.to(torch.int32))
        self.register_buffer('dst32', dst.to(torch.int32))

    def describe(self):
        d = super().describe(); d.update(core='FusedVariantCore', kernel='cupy prop_fwd/prop_bwd', index_dtype='int32'); return d

    def propagate(self, s, w):
        return PropagateFused.apply(s, w, self.src32, self.dst32)

    def propagate_soft(self, s, p, w):
        return PropagateFusedSoft.apply(s, p, w, self.src32, self.dst32)

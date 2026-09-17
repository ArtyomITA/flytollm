"""E2: propagation micro-benchmark on the real graph (batch 2, as in training): FP32 weights vs FP16 storage
(weights and spikes stored in half, accumulated in FP32 through a half-precision gather and float index_add),
and gather+index_add vs torch.sparse CSR matmul. INT8 dp4a needs a custom kernel: not measured."""
import json, time
import torch
from bench_runtime import ROOT
from fly_graph import load_graph


def timed(fn, repeats=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeats * 1000


def main():
    g = load_graph(10, progress=lambda m: None)
    n = len(g['body_ids']); src = torch.from_numpy(g['src']).cuda().long(); dst = torch.from_numpy(g['dst']).cuda().long()
    w = torch.rand(len(src), device='cuda') * .01
    s = (torch.rand(2, n, device='cuda') < .05).float()
    chunk = 262144
    def fp32():
        out = torch.zeros_like(s)
        for i in range(0, w.numel(), chunk):
            sl = slice(i, i + chunk); out.index_add_(1, dst[sl], s[:, src[sl]] * w[sl])
        return out
    w16 = w.half(); s16 = s.half()
    def fp16_storage():
        out = torch.zeros_like(s)
        for i in range(0, w.numel(), chunk):
            sl = slice(i, i + chunk); out.index_add_(1, dst[sl], (s16[:, src[sl]] * w16[sl]).float())
        return out
    src32 = src.int(); dst32 = dst.int()
    def fp32_int32_index():
        out = torch.zeros_like(s)
        for i in range(0, w.numel(), chunk):
            sl = slice(i, i + chunk); out.index_add_(1, dst32[sl].long(), s[:, src32[sl].long()] * w[sl])
        return out
    csr = torch.sparse_coo_tensor(torch.stack([dst, src]), w, (n, n)).coalesce().to_sparse_csr()
    def sparse_csr():
        return torch.sparse.mm(csr, s.T).T
    result = dict(edges=int(len(src)), nodes=int(n), batch=2,
                  fp32_gather_index_add_ms=timed(fp32), fp16_storage_ms=timed(fp16_storage),
                  int32_index_cast_ms=timed(fp32_int32_index), torch_sparse_csr_mm_ms=timed(sparse_csr),
                  note='INT8 dp4a not measured: needs a custom CUDA kernel; Pascal has no tensor cores; FP16 compute is 1/64 of FP32 on GP104')
    ref = fp32(); result['fp16_max_abs_error'] = float((fp16_storage() - ref).abs().max()); result['csr_max_abs_error'] = float((sparse_csr() - ref).abs().max())
    (ROOT / 'results/phase6c_microbench.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

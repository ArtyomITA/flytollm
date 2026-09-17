"""
Benchmark del throughput reale per FlyToLLM sul connettoma male-CNS.

Misura ms/carattere per forward + backward del nucleo ricorrente spiking,
in modalita' eager e con cattura CUDA graph, sulla GPU presente.
Non allena nulla: gira per un numero fisso di passi e riporta i tempi,
la memoria di picco e l'estrapolazione a epoche su dataset di varie dimensioni.

Su Pascal (sm_61) torch.compile/Inductor non e' disponibile perche' Triton
richiede compute capability >= 7.0: i CUDA graph sono la leva principale,
e questo script misura quanto valgono su questo carico.

Uso:
    python bench_throughput.py --soglia 10
    python bench_throughput.py --soglia 5 --batch 32 --T 8
    python bench_throughput.py --soglia 10 --no-graph
"""

import argparse
import time
import numpy as np
import pyarrow.feather as pf
import pyarrow.compute as pc
import pyarrow as pa

DATA = r"E:\claudecode pesante\flytollm\dataset\male_cns"


def carica_grafo(soglia):
    """Lista di archi del CNS intero sopra la soglia, piu' segno per la legge di Dale."""
    t0 = time.time()
    ann = pf.read_feather(rf"{DATA}\body-annotations-male-cns-v1.0.feather")
    reali = ann.loc[ann["superclass"].notna(), "bodyId"].to_numpy()

    tab = pf.read_table(
        rf"{DATA}\connectome-weights-male-cns-v1.0.feather", memory_map=True
    )
    ids = pa.array(reali)
    tab = tab.filter(
        pc.and_(
            pc.is_in(tab.column("body_pre"), value_set=ids),
            pc.is_in(tab.column("body_post"), value_set=ids),
        )
    )
    tab = tab.filter(pc.greater_equal(tab.column("weight"), soglia))

    pre = tab.column("body_pre").to_numpy()
    post = tab.column("body_post").to_numpy()
    w = tab.column("weight").to_numpy().astype(np.float32)

    usati, inverse = np.unique(np.concatenate([pre, post]), return_inverse=True)
    src = inverse[: len(pre)].astype(np.int64)
    dst = inverse[len(pre) :].astype(np.int64)

    nt = pf.read_feather(rf"{DATA}\body-neurotransmitters-male-cns-v1.0.feather")
    nt_map = dict(zip(nt["body"].to_numpy(), nt["consensus_nt"].to_numpy()))
    INIB = {"gaba", "glutamate", "histamine"}
    segno = np.ones(len(usati), dtype=np.float32)
    for i, b in enumerate(usati):
        if nt_map.get(b, "unclear") in INIB:
            segno[i] = -1.0
    n_inib = int((segno < 0).sum())

    print(f"grafo caricato in {time.time()-t0:.1f} s")
    print(f"  neuroni  : {len(usati):,}")
    print(f"  archi    : {len(src):,}")
    print(f"  inibitori: {n_inib:,} ({100*n_inib/len(usati):.1f}%)")
    return src, dst, w, segno, len(usati)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soglia", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=400_000, help="archi per blocco")
    ap.add_argument("--T", type=int, default=8, help="sottopassi per carattere")
    ap.add_argument("--tronca", type=int, default=64, help="troncamento BPTT")
    ap.add_argument("--passi", type=int, default=6)
    ap.add_argument("--no-graph", action="store_true", help="salta la prova CUDA graph")
    args = ap.parse_args()

    import torch

    print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\nGPU non disponibile: questa build di torch e' CPU-only.")
        print("  pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126")
        return

    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name} | sm_{prop.major}{prop.minor} | {prop.total_memory/2**30:.1f} GB\n")

    src, dst, w, segno, N = carica_grafo(args.soglia)
    E = len(src)

    src_t = torch.from_numpy(src).to(dev)
    dst_t = torch.from_numpy(dst).to(dev)
    segno_t = torch.from_numpy(segno).to(dev)

    mag = torch.nn.Parameter(torch.from_numpy(np.log1p(w) / 8.0).to(dev))
    opt = torch.optim.Adam([mag], lr=1e-3, capturable=True)

    # legge di Dale: il segno dell'arco e' quello del neurone presinaptico
    segno_arco = segno_t[src_t].contiguous()

    B, T, L = args.batch, args.T, args.tronca
    tau, soglia_spike = 0.95, 1.0

    class Surrogato(torch.autograd.Function):
        """Heaviside in avanti, derivata della sigmoide all'indietro."""

        @staticmethod
        def forward(ctx, v):
            ctx.save_for_backward(v)
            return (v > 0).float()

        @staticmethod
        def backward(ctx, g):
            (v,) = ctx.saved_tensors
            sg = torch.sigmoid(4.0 * v)
            return g * 4.0 * sg * (1 - sg)

    spike_fn = Surrogato.apply

    CHUNK = args.chunk

    class Propaga(torch.autograd.Function):
        """
        I[b, dst[e]] += S[b, src[e]] * w[e]

        Non materializza mai un tensore [batch x archi]: scorre gli archi a
        blocchi e nel backward ricostruisce i gradienti da S e da grad_I, che
        sono entrambi [batch x neuroni].
        """

        @staticmethod
        def forward(ctx, S, w):
            I = torch.zeros_like(S)
            for i in range(0, w.numel(), CHUNK):
                s_ = src_t[i : i + CHUNK]
                d_ = dst_t[i : i + CHUNK]
                I.index_add_(1, d_, S[:, s_] * w[i : i + CHUNK])
            ctx.save_for_backward(S.to(torch.bool), w)
            return I

        @staticmethod
        def backward(ctx, gI):
            Sb, w = ctx.saved_tensors
            S = Sb.to(gI.dtype)
            gS = torch.zeros_like(gI)
            gw = torch.zeros_like(w)
            gI = gI.contiguous()
            for i in range(0, w.numel(), CHUNK):
                s_ = src_t[i : i + CHUNK]
                d_ = dst_t[i : i + CHUNK]
                g = gI[:, d_]
                gw[i : i + CHUNK] = (g * S[:, s_]).sum(0)
                gS.index_add_(1, s_, g * w[i : i + CHUNK])
            return gS, gw

    propaga = Propaga.apply

    def finestra(drive):
        """Una finestra di troncamento completa: L caratteri x T sottopassi."""
        V = torch.zeros(B, N, device=dev)
        S = torch.zeros(B, N, device=dev)
        perdita = torch.zeros((), device=dev)
        pesi = mag * segno_arco
        for c in range(L):
            V = V + drive[c]
            for _ in range(T):
                V = tau * V + propaga(S, pesi)
                S = spike_fn(V - soglia_spike)
                V = V * (1 - S)
            perdita = perdita + S.mean()
        return perdita

    print(f"config: batch {B} · T {T} · troncamento {L} · archi {E:,} · neuroni {N:,}")
    print(f"passi sequenziali per finestra: {L*T:,}\n")

    drive = (torch.rand(L, B, N, device=dev) * 0.08).contiguous()

    def cronometra(nome, esegui, passi):
        tempi = []
        for it in range(passi):
            torch.cuda.synchronize()
            t0 = time.time()
            esegui()
            torch.cuda.synchronize()
            dt = time.time() - t0
            tempi.append(dt)
            car = B * L
            print(f"  [{nome}] passo {it+1}/{passi}: {dt:6.3f} s  ({1000*dt/car:6.3f} ms/car)")
        stabili = tempi[1:] if len(tempi) > 1 else tempi
        return float(np.median(stabili))

    # ---------- eager ----------
    def passo_eager():
        opt.zero_grad(set_to_none=False)
        finestra(drive).backward()
        opt.step()

    print("MODALITA' EAGER")
    med_eager = cronometra("eager", passo_eager, args.passi)
    vram_eager = torch.cuda.max_memory_allocated() / 2**30

    med_graph = None
    vram_graph = None

    # ---------- cuda graph ----------
    if not args.no_graph:
        print("\nMODALITA' CUDA GRAPH")
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    opt.zero_grad(set_to_none=False)
                    finestra(drive).backward()
                    opt.step()
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            opt.zero_grad(set_to_none=False)
            with torch.cuda.graph(g):
                perdita_statica = finestra(drive)
                perdita_statica.backward()
                opt.step()

            print(f"  cattura riuscita · {L*T:,} passi in un solo grafo")
            med_graph = cronometra("graph", g.replay, args.passi)
            vram_graph = torch.cuda.max_memory_allocated() / 2**30
        except Exception as exc:
            print(f"  cattura fallita: {type(exc).__name__}: {exc}")
            print("  (si prosegue con i soli numeri eager)")

    # ---------- esito ----------
    car = B * L
    print("\n" + "=" * 62)
    ms_e = 1000 * med_eager / car
    cs_e = car / med_eager
    print(f"EAGER       : {ms_e:.4f} ms/carattere · {cs_e:>10,.0f} car/s · {vram_eager:.2f} GB")
    if med_graph:
        ms_g = 1000 * med_graph / car
        cs_g = car / med_graph
        print(f"CUDA GRAPH  : {ms_g:.4f} ms/carattere · {cs_g:>10,.0f} car/s · {vram_graph:.2f} GB")
        print(f"GUADAGNO    : {100*(med_eager/med_graph - 1):+.1f}%")
    print("=" * 62)

    migliore = car / (med_graph or med_eager)
    print(f"\nTempo per epoca, dal throughput misurato ({migliore:,.0f} caratteri/s):\n")
    for nome, nchar in [
        ("20 MB", 20e6),
        ("50 MB", 50e6),
        ("500 MB", 500e6),
        ("TinyStories intero (2,23 GB)", 2.23e9),
    ]:
        sec = nchar / migliore
        if sec < 7200:
            q = f"{sec/60:.0f} minuti"
        elif sec < 172800:
            q = f"{sec/3600:.1f} ore"
        else:
            q = f"{sec/86400:.1f} giorni"
        print(f"  {nome:32s} {q}")
    print()


if __name__ == "__main__":
    main()

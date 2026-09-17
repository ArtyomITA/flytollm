"""Standard baselines (GRU, small decoder Transformer) on the exact fly data feed; PROTOCOLLO_VALUTAZIONE."""
import argparse, gc, hashlib, json, math, random, sys, time, traceback
from collections import Counter
from pathlib import Path
from bench_runtime import ROOT, emit, memory, supervise
import pretrain_resumable as base

PAD, BOS, EOS, VOCAB, DIM = 0, 1, 2, 4096, 256


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(kind):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class GRULM(nn.Module):
        kind = 'gru'

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB, DIM)
            self.rnn = nn.GRU(DIM, DIM, num_layers=2)
            self.projection = nn.Linear(DIM, DIM)

        def initial_state(self, batch):
            return torch.zeros(2, batch, DIM, device=self.embedding.weight.device)

        def step(self, ids, state):
            active = ids.ne(PAD)
            output, new = self.rnn(self.embedding(ids)[None], state)
            new = torch.where(active[None, :, None], new, state)
            logits = F.linear(self.projection(output[0]), self.embedding.weight)
            return logits, new

        def forward(self, ids, state):
            # Whole segment through cuDNN. A lane past its story end keeps evolving on PAD
            # embeddings, but its outputs are masked and its state is discarded at the next reset.
            output, new = self.rnn(self.embedding(ids), state)
            return F.linear(self.projection(output), self.embedding.weight), new

    class Attention(nn.Module):
        def __init__(self, heads=4):
            super().__init__()
            self.heads, self.head_dim = heads, DIM // heads
            self.q = nn.Linear(DIM, DIM, bias=False); self.k = nn.Linear(DIM, DIM, bias=False)
            self.v = nn.Linear(DIM, DIM, bias=False); self.o = nn.Linear(DIM, DIM, bias=False)
            self.register_buffer('inv', 1. / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)))

        def rope(self, x, pos):
            angle = pos[:, None, :, None].float() * self.inv[None, None, None, :]
            c, s = angle.cos(), angle.sin(); even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack([even * c - odd * s, even * s + odd * c], -1).flatten(-2)

        def forward(self, x, pos, mask):
            b, l, _ = x.shape
            shape = (b, l, self.heads, self.head_dim)
            q = self.rope(self.q(x).view(shape).transpose(1, 2), pos)
            k = self.rope(self.k(x).view(shape).transpose(1, 2), pos)
            v = self.v(x).view(shape).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return self.o(y.transpose(1, 2).reshape(b, l, DIM))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(DIM); self.attention = Attention()
            self.norm2 = nn.LayerNorm(DIM)
            self.ff1 = nn.Linear(DIM, 4 * DIM); self.ff2 = nn.Linear(4 * DIM, DIM)

        def forward(self, x, pos, mask):
            x = x + self.attention(self.norm1(x), pos, mask)
            return x + self.ff2(F.gelu(self.ff1(self.norm2(x))))

    class TransformerLM(nn.Module):
        kind = 'transformer'
        context = 128

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(VOCAB, DIM)
            self.blocks = nn.ModuleList([Block() for _ in range(2)])
            self.norm = nn.LayerNorm(DIM)

        def forward(self, ids, pos):
            """ids,pos [L,B]; left padding allowed (PAD keys masked; PAD queries attend themselves)."""
            x = self.embedding(ids.T); b, l, _ = x.shape
            valid = ids.T.ne(PAD)
            causal = torch.ones(l, l, dtype=torch.bool, device=x.device).tril()
            mask = causal[None, None] & valid[:, None, None, :]
            mask = mask | torch.eye(l, dtype=torch.bool, device=x.device)[None, None]
            for block in self.blocks:
                x = block(x, pos.T, mask)
            return F.linear(self.norm(x), self.embedding.weight).transpose(0, 1)

    model = GRULM() if kind == 'gru' else TransformerLM()
    with torch.no_grad():
        nn.init.normal_(model.embedding.weight, std=.02)
    return model.cuda()


def masked_ce(logits, targets, ids):
    import torch
    from torch.nn import functional as F
    valid = ids.ne(PAD) & targets.ne(PAD)
    safe = torch.where(valid, targets, torch.zeros_like(targets))
    losses = F.cross_entropy(logits.flatten(0, 1), safe.flatten(), reduction='none').reshape_as(targets)
    correct = ((logits.argmax(-1) == targets) & valid).sum()
    return (losses * valid).sum(), valid.sum(), correct


class Trainer:
    """Same window semantics as the fly engine: 16 positions per update, TBPTT 8 (GRU), reset per story pair."""

    def __init__(self, model, lr):
        import torch
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999), foreach=False)
        self.history = None
        self.reset()

    def reset(self):
        if self.model.kind == 'gru':
            self.state = self.model.initial_state(2)
        else:
            self.history = [[], []]
            self.offset = 0

    def window(self, x):
        """Transformer: story prefix (<=128) plus the new tokens; returns ids,pos [Lw,B] and slot offset."""
        import torch
        rows = []
        for lane in range(x.shape[1]):
            count = int(x[:, lane].ne(PAD).sum())
            tokens = self.history[lane] + x[:count, lane].tolist()
            start = max(0, len(tokens) - self.model.context)
            rows.append((tokens[start:], self.offset + start - len(self.history[lane])))
        width = max(len(r[0]) for r in rows)
        ids = torch.zeros(width, len(rows), dtype=torch.long, device='cuda')
        pos = torch.zeros_like(ids)
        for lane, (tokens, first) in enumerate(rows):
            if tokens:
                ids[width - len(tokens):, lane] = torch.tensor(tokens, device='cuda')
                pos[width - len(tokens):, lane] = torch.arange(first, first + len(tokens), device='cuda')
        return ids, pos

    def update(self, x, y):
        import torch
        self.opt.zero_grad(set_to_none=False)
        total = (x.ne(PAD) & y.ne(PAD)).sum()
        if self.model.kind == 'gru':
            state = self.state; loss_sum = torch.zeros((), device='cuda')
            for start in range(0, x.shape[0], 8):
                ids, targets = x[start:start + 8], y[start:start + 8]
                logits, new = self.model(ids, state)
                ce, count, _ = masked_ce(logits, targets, ids)
                part = ce / total.clamp_min(1); part.backward()
                loss_sum = loss_sum + part.detach(); state = new.detach()
            self.state = state
        else:
            ids, pos = self.window(x)
            logits = self.model(ids, pos)
            tail = x.shape[0]
            targets = torch.zeros_like(ids)
            for lane in range(x.shape[1]):
                count = int(x[:, lane].ne(PAD).sum())
                if count:
                    targets[ids.shape[0] - count:, lane] = y[:count, lane]
            ce, count, _ = masked_ce(logits[-tail:], targets[-tail:], ids[-tail:])
            loss_sum = ce / total.clamp_min(1); loss_sum.backward()
            for lane in range(x.shape[1]):
                count = int(x[:, lane].ne(PAD).sum())
                self.history[lane] = (self.history[lane] + x[:count, lane].tolist())[-self.model.context:]
            self.offset += x.shape[0]
        finite = all(bool(torch.isfinite(p.grad).all()) for p in self.model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., foreach=False)
        self.opt.step()
        if not finite or not torch.isfinite(loss_sum):
            raise RuntimeError('Nonfinite training')
        return dict(loss=float(loss_sum), count=int(total), grad_norm=float(norm))

    @staticmethod
    def evaluate(model, pairs):
        import torch
        sums = [0., 0, 0]
        with torch.no_grad():
            for pair in pairs:
                if model.kind == 'gru':
                    state = model.initial_state(2)
                    for x, y in pair:
                        logits, state = model(x, state)
                        ce, count, correct = masked_ce(logits, y, x)
                        sums[0] += float(ce); sums[1] += int(count); sums[2] += int(correct)
                else:
                    x = torch.cat([w[0] for w in pair]); y = torch.cat([w[1] for w in pair])
                    pos = torch.arange(x.shape[0], device='cuda')[:, None].expand_as(x)
                    ce, count, correct = masked_ce(model(x, pos), y, x)
                    sums[0] += float(ce); sums[1] += int(count); sums[2] += int(correct)
        return dict(ce=sums[0] / sums[1], targets=sums[1], accuracy=sums[2] / sums[1])


def samples(model):
    import torch
    from fly_lm import LMConfig, select_token
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT / 'dataset/prepared_v1/tokenizer-4096.json'))
    config = LMConfig(); entries = []
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        for temperature in (0., .8):
            torch.manual_seed(93018)
            for prompt in ('Once upon a time', 'The little girl', 'Tom wanted to'):
                tokens = [BOS] + tokenizer.encode(prompt).ids
                generated = []
                for _ in range(32):
                    seq = torch.tensor(tokens + generated, device='cuda')[:, None].expand(-1, 2).contiguous()
                    if model.kind == 'gru':
                        logits, _ = model(seq, model.initial_state(2)); last = logits[-1]
                    else:
                        seq = seq[-model.context:]
                        pos = torch.arange(seq.shape[0], device='cuda')[:, None].expand_as(seq)
                        last = model(seq, pos)[-1]
                    token = int(select_token(last, config, temperature)[0])
                    generated.append(token)
                    if token == EOS:
                        break
                entries.append(dict(prompt=prompt, temperature=temperature, seed=93018, ids=generated,
                                    text=tokenizer.decode(generated)))
    return entries


def run(a):
    import torch
    from lm_io import story_batch
    from text_dataset import StoryDataset
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.75)
    torch.manual_seed(a.seed); random.seed(a.seed)
    model = build(a.model); trainer = Trainer(model, a.lr)
    parameters = sum(p.numel() for p in model.parameters())
    config = dict(model=a.model, lr=a.lr, seed=a.seed, batch=2, positions=16, tbptt=8 if a.model == 'gru' else None,
                  context=None if a.model == 'gru' else model.context, optimizer='Adam', clip=1., eval_prefix=128,
                  story_policy='full stories, sequential pairs from130, reset each pair, repeat epochs (pretrain_resumable.next_pair)',
                  baseline_frozen_after_update=2000, backend='GPU FP32 eager', dropout=0.)
    dev_ids = json.loads((ROOT / 'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
    with StoryDataset(ROOT / 'dataset/prepared_v1', 'validation') as ds:
        dev = [[story_batch(ds, dev_ids[b:b + 2], [t, t], 16) for t in range(0, 128, 16)] for b in range(0, 16, 2)]
    progress = dict(updates=0, targets=0, cursor=dict(story=130, offset=0, epoch=0), curve=[], train_seconds=0., samples=[])
    progress['initial_dev'] = Trainer.evaluate(model, dev)
    uni, bi, ctx = Counter(), Counter(), Counter()

    def counts():
        return dict(unigram=list(uni.items()), bigram=[[list(k), v] for k, v in bi.items()], contexts=list(ctx.items()))
    measured = []; slow = 0; stopped = 'update_budget'
    with StoryDataset(ROOT / 'dataset/prepared_v1', 'train') as ds:
        while progress['updates'] < a.updates:
            if (ROOT / 'STOP_PHASE6').exists():
                stopped = 'user_stop_file'; break
            x, y = base.next_pair(ds, progress['cursor'], trainer)
            torch.cuda.synchronize(); started = time.perf_counter(); m = trainer.update(x, y); torch.cuda.synchronize()
            dt = time.perf_counter() - started
            assert m['count'] > 0
            progress['updates'] += 1; progress['targets'] += m['count']; progress['train_seconds'] += dt; measured.append(dt)
            update = progress['updates']
            slow = slow + 1 if dt > 2 else 0
            if slow >= 5:
                raise RuntimeError('Five consecutive updates >2s')
            if update <= 2000:
                for u, v in zip(x.cpu().flatten().tolist(), y.cpu().flatten().tolist()):
                    if u and v:
                        uni[v] += 1; bi[u, v] += 1; ctx[u] += 1
            if update % 16 == 0:
                tele = dict(update=update, targets=progress['targets'], seconds=dt, **m)
                progress.setdefault('telemetry', []).append(tele); emit('training', **tele)
            if update % a.eval_every == 0 or update == a.updates:
                dv = Trainer.evaluate(model, dev); baseline = base.baselines(dev, counts())
                entry = dict(update=update, targets=progress['targets'], dev=dv, baseline=baseline)
                progress['curve'].append(entry); emit('validation', **entry)
                progress['samples'].append(dict(update=update, generation=samples(model)))
    repeated = [Trainer.evaluate(model, dev) for _ in range(2)]
    generated = samples(model)
    checkpoint = Path(a.output).with_suffix('.final.pt')
    torch.save(dict(model=a.model, config=config, state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                    progress={k: v for k, v in progress.items() if k != 'samples'}), checkpoint)
    return dict(**progress, config=config, parameters=parameters, generation=generated, repeated_dev=repeated,
                stop_reason=stopped, mean_step_s=sum(measured) / len(measured) if measured else None,
                checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), audit_evaluated=False, test_evaluated=False,
                data_feed='pretrain_resumable.next_pair; identical story order and windows as the fly runs',
                semantic_verdict='standard baseline; no connectome')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['gru', 'transformer'], required=True)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--seed', type=int, default=17)
    p.add_argument('--updates', type=int, required=True); p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--output', required=True); p.add_argument('--worker', action='store_true')
    p.add_argument('--allow-paging', action='store_true'); p.add_argument('--timeout', type=float, default=7200)
    p.add_argument('--phase-timeout', type=float, default=300); p.add_argument('--max-vram-mb', type=int, default=7000)
    a = p.parse_args()
    if a.updates < 1 or a.eval_every < 1:
        p.error('positive updates/eval interval required')
    if not a.worker:
        return supervise(a, worker_module='baseline_lm')
    try:
        import torch
        emit('load', python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda, memory=memory())
        result = run(a); peak = torch.cuda.max_memory_allocated() / 2 ** 20
        gc.collect(); torch.cuda.synchronize(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache(); gc.collect()
        final = torch.cuda.memory_allocated() / 2 ** 20
        result.update(ok=final == 0, final_allocator_mb=final, peak_vram_mb=peak,
                      fingerprints={f: digest(ROOT / f) for f in ['baseline_lm.py', 'pretrain_resumable.py', 'lm_io.py', 'text_dataset.py',
                                                                  'dataset/prepared_v1/manifest.json', 'dataset/prepared_v1/tokenizer-4096.json']})
    except Exception as exc:
        traceback.print_exc(); result = dict(ok=False, error=str(exc), error_type=type(exc).__name__)
    Path(a.output).with_suffix('.worker.json').write_text(json.dumps(result, indent=2))
    if not result['ok']:
        sys.exit(1)


if __name__ == '__main__':
    main()

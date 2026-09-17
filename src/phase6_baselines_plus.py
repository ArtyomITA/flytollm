"""Phase 6 follow-up, CPU only: n-gram baselines refitted on the exact training stream seen by every model
(same pairs, same order, cumulative up to 2000 and 8000 updates), with declared smoothing, plus a prequential
proxy (mean training loss along the stream, from the telemetry every 16 updates) for every phase-6 model.
No GPU, no model code touched. Output: results/phase6_baselines_plus.json."""
import json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from bench_runtime import ROOT
import pretrain_resumable as base
from lm_io import story_batch
from text_dataset import StoryDataset

V = 4096
CHECKPOINTS = (2000, 8000)


class DummyEngine:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


def to_np(t):
    return np.asarray(t.cpu()) if hasattr(t, 'cpu') else np.asarray(t)


def lane_sequences(pairs_by_update):
    """Yield (update, lane, [(prev2, prev1, u, v), ...]) keeping lane continuity across windows of one story."""
    history = {0: [], 1: []}
    for update, (x, y, reset) in enumerate(pairs_by_update, start=1):
        if reset:
            history = {0: [], 1: []}
        for lane in (0, 1):
            seq = []
            for pos in range(x.shape[0]):
                u, v = int(x[pos, lane]), int(y[pos, lane])
                if not u or not v:
                    continue
                h = history[lane]
                seq.append((h[-1] if len(h) >= 1 else 0, u, v))
                history[lane].append(u)
            yield update, lane, seq


class NGram:
    """Counts for interpolated Kneser-Ney (trigram/bigram) and additive smoothing."""

    def __init__(self):
        self.uni = Counter(); self.bi = Counter(); self.tri = Counter()
        self.ctx1 = Counter(); self.ctx2 = Counter()
        self.types_after1 = defaultdict(set); self.types_after2 = defaultdict(set)
        self.continuation = defaultdict(set)  # word -> set of left contexts (for KN unigram)
        self.n = 0

    def add(self, p, u, v):
        self.uni[v] += 1; self.n += 1
        self.bi[u, v] += 1; self.ctx1[u] += 1; self.types_after1[u].add(v); self.continuation[v].add(u)
        if p:
            self.tri[p, u, v] += 1; self.ctx2[p, u] += 1; self.types_after2[p, u].add(v)

    def p_add(self, v, k):
        return (self.uni[v] + k) / (self.n + k * V)

    def p_project_bigram(self, u, v):
        p = (self.uni[v] + 1) / (self.n + V)
        return (self.bi[u, v] + 10 * p) / (self.ctx1[u] + 10)

    def p_kn_uni(self, v):
        total_types = sum(len(s) for s in self.continuation.values())
        return (len(self.continuation[v]) + 0.5) / (total_types + 0.5 * V)

    def p_kn_bi(self, u, v, d=0.75):
        c = self.ctx1[u]
        if c == 0:
            return self.p_kn_uni(v)
        lam = d * len(self.types_after1[u]) / c
        return max(self.bi[u, v] - d, 0) / c + lam * self.p_kn_uni(v)

    def p_kn_tri(self, p, u, v, d=0.75):
        c = self.ctx2[p, u]
        if not p or c == 0:
            return self.p_kn_bi(u, v, d)
        lam = d * len(self.types_after2[p, u]) / c
        return max(self.tri[p, u, v] - d, 0) / c + lam * self.p_kn_bi(u, v, d)


def dev_sequences(dev):
    """DEV16 as lane-continuous sequences (same construction as pretrain_resumable.run)."""
    out = []
    for pair_windows in dev:
        history = {0: [], 1: []}
        for x, y in pair_windows:
            x, y = to_np(x), to_np(y)
            for lane in (0, 1):
                for pos in range(x.shape[0]):
                    u, v = int(x[pos, lane]), int(y[pos, lane])
                    if not u or not v:
                        continue
                    h = history[lane]
                    out.append((h[-1] if h else 0, u, v))
                    history[lane].append(u)
    return out


def evaluate(model, seqs):
    scores = {'unigram_add1': 0., 'unigram_add0.1': 0., 'bigram_project': 0., 'bigram_kn': 0., 'trigram_kn': 0.}
    for p, u, v in seqs:
        scores['unigram_add1'] -= math.log(model.p_add(v, 1.))
        scores['unigram_add0.1'] -= math.log(model.p_add(v, .1))
        scores['bigram_project'] -= math.log(model.p_project_bigram(u, v))
        scores['bigram_kn'] -= math.log(model.p_kn_bi(u, v))
        scores['trigram_kn'] -= math.log(model.p_kn_tri(p, u, v))
    return {k: s / len(seqs) for k, s in scores.items()}, len(seqs)


def prequential(name, limits):
    path = ROOT / 'results' / f'{name}.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    tele = (data.get('result') or {}).get('telemetry') or []
    out = {}
    for limit in limits:
        rows = [t for t in tele if t.get('update', 0) <= limit and t.get('loss') is not None]
        if not rows:
            continue
        weights = [t.get('count') or 1 for t in rows]
        out[str(limit)] = dict(mean_train_loss=sum(t['loss'] * w for t, w in zip(rows, weights)) / sum(weights),
                               samples=len(rows), last_update=rows[-1]['update'])
    return out


def main():
    dev_ids = json.loads((ROOT / 'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
    with StoryDataset(ROOT / 'dataset/prepared_v1', 'validation') as ds:
        dev = [[story_batch(ds, dev_ids[b:b + 2], [t, t], 16) for t in range(0, 128, 16)] for b in range(0, 16, 2)]
    dev_seqs = dev_sequences(dev)
    engine = DummyEngine(); cursor = dict(story=130, offset=0, epoch=0); stream = []
    with StoryDataset(ROOT / 'dataset/prepared_v1', 'train') as ds:
        for _ in range(max(CHECKPOINTS)):
            before = engine.resets
            x, y = base.next_pair(ds, cursor, engine)
            stream.append((to_np(x), to_np(y), engine.resets != before))
    model = NGram(); result = dict(dev_targets=len(dev_seqs), smoothing=dict(
        unigram='additive k in {1, 0.1} over V=4096', bigram_project='(c(u,v)+10*p_add1(v))/(c(u)+10), as pretrain_resumable.baselines',
        bigram_kn='interpolated Kneser-Ney, discount 0.75, continuation unigram with 0.5 smoothing',
        trigram_kn='interpolated Kneser-Ney, discount 0.75, backing off to bigram_kn; history reset at each story'), ngram={})
    targets = 0
    for update, lane, seq in lane_sequences(stream):
        for p, u, v in seq:
            model.add(p, u, v); targets += 1
        if lane == 1 and update in CHECKPOINTS:
            scores, n = evaluate(model, dev_seqs)
            result['ngram'][str(update)] = dict(train_targets=targets, dev_ce=scores)
            print(update, 'train_targets', targets, json.dumps({k: round(v, 4) for k, v in scores.items()}))
    result['prequential_proxy'] = {name: prequential(name, CHECKPOINTS) for name in
                                   ['pretrain_night8000_h10', 'pretrain_pilot_h10', 'phase6_rewired_h10_8000', 'phase6_gru_lr1e-4',
                                    'phase6_gru_lr1e-3', 'phase6_transformer_lr1e-4', 'phase6_transformer_lr1e-3']}
    for name, value in result['prequential_proxy'].items():
        print(name, json.dumps(value))
    (ROOT / 'results/phase6_baselines_plus.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

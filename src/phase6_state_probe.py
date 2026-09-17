"""Linear probe on the core state (PREREGISTRAZIONE_FASE_6B.md, test 1.3). Inference only, one checkpoint at a time.

For every token position the story is fed exactly as in evaluation (attention on, state carried within the story,
fresh state per story pair). Features: membrane potential of a fixed random subset of 8,192 neurons (seed 5) plus all
readout neurons, and the last-substep spike of the readout neurons. A ridge classifier (dual form, lambda = number of
training rows, standardized features) trained on training-split stories 0-47 predicts the token at lag k in {0,1,2,4,8}
and is scored on DEV16. Chance control: same probe with training labels permuted across rows."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from bench_runtime import ROOT
import pretrain_resumable as base
from lm_io import story_batch
from text_dataset import StoryDataset

LAGS = (0, 1, 2, 4, 8)
V = 4096
CHECKPOINTS = {'fly_real_8000': 'results/pretrain_night8000_h10.latest.pt',
               'rewired_8000': 'results/phase6_rewired_h10_8000.latest.pt',
               'fly_real_26247': 'results/pretrain_continuous.latest.pt'}


def collect(model, ds, story_ids, subset, readout):
    features, labels = [], []
    for b in range(0, len(story_ids), 2):
        pair = story_ids[b:b + 2]
        state = model.initial_state(2)
        history = [[], []]
        for t in range(0, 128, 16):
            x, _ = story_batch(ds, pair, [t, t], 16)
            x = torch.as_tensor(x).cuda()
            for pos in range(x.shape[0]):
                ids = x[pos]
                with torch.no_grad():
                    _, state = model.step(ids, state)
                for lane in range(2):
                    tok = int(ids[lane])
                    if tok == 0:
                        continue
                    history[lane].append(tok)
                    v = state.voltage[lane, subset]
                    s = state.spike[lane, readout].float()
                    features.append(torch.cat([v, s]).float().cpu())
                    labels.append([history[lane][-1 - k] if len(history[lane]) > k else -1 for k in LAGS])
    return torch.stack(features), np.asarray(labels, dtype=np.int64)


def ridge_dual(x_train, y_train, x_eval, lam):
    """Return eval scores (rows x V) of ridge regression onto one-hot labels; dual form since rows << features."""
    mu = x_train.mean(0, keepdim=True); sd = x_train.std(0, keepdim=True) + 1e-6
    xt = ((x_train - mu) / sd).cuda(); xe = ((x_eval - mu) / sd).cuda()
    y = torch.zeros(len(y_train), V, device='cuda'); y[torch.arange(len(y_train)), torch.as_tensor(y_train).cuda()] = 1.
    gram = xt @ xt.T
    gram.diagonal().add_(lam)
    alpha = torch.linalg.solve(gram, y)
    scores = xe @ (xt.T @ alpha)
    return scores.cpu()


def probe(name, path, train_stories, dev_ids, seed=5):
    saved = torch.load(ROOT / path, map_location='cpu', weights_only=True)
    model = base.load_payload(saved['model'], 'separate'); model.eval()
    n = model.core.n
    readout = model.interfaces.readout.nodes.cuda().long()
    rng = np.random.default_rng(seed)
    subset = torch.as_tensor(np.union1d(rng.choice(n, 8192, replace=False), readout.cpu().numpy())).cuda().long()
    started = time.time()
    with StoryDataset(ROOT / 'dataset/prepared_v1', 'train') as ds:
        x_train, y_train = collect(model, ds, list(range(train_stories)), subset, readout)
    with StoryDataset(ROOT / 'dataset/prepared_v1', 'validation') as ds:
        x_dev, y_dev = collect(model, ds, dev_ids, subset, readout)
    del model; torch.cuda.empty_cache()
    out = dict(checkpoint=path, features=int(x_train.shape[1]), train_rows=int(len(x_train)), dev_rows=int(len(x_dev)),
               subset_neurons=int(subset.numel()), collect_seconds=time.time() - started, lags={})
    perm = np.random.default_rng(seed + 1).permutation(len(y_train))
    for i, k in enumerate(LAGS):
        tr = (y_train[:, i] > 2); ev = (y_dev[:, i] > 2)
        lam = float(tr.sum())
        real = ridge_dual(x_train[tr], y_train[tr, i], x_dev[ev], lam).argmax(1).numpy()
        shuffled_labels = y_train[perm, i]
        ctrl_mask = tr & (shuffled_labels > 2)
        chance = ridge_dual(x_train[ctrl_mask], shuffled_labels[ctrl_mask], x_dev[ev], float(ctrl_mask.sum())).argmax(1).numpy()
        majority = np.bincount(y_train[tr, i], minlength=V).argmax()
        truth = y_dev[ev, i]
        out['lags'][str(k)] = dict(accuracy=float((real == truth).mean()), chance_permuted=float((chance == truth).mean()),
                                   majority=float((truth == majority).mean()), rows=int(ev.sum()))
        print(name, 'lag', k, json.dumps(out['lags'][str(k)]), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-stories', type=int, default=48)
    p.add_argument('--only', nargs='*', default=None)
    p.add_argument('--output', default=str(ROOT / 'results/phase6b_state_probe.json'))
    a = p.parse_args()
    dev_ids = json.loads((ROOT / 'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
    results = dict(protocol='PREREGISTRAZIONE_FASE_6B.md', lags=list(LAGS), train_stories=a.train_stories, checkpoints={})
    for name, path in CHECKPOINTS.items():
        if a.only and name not in a.only:
            continue
        if not (ROOT / path).exists():
            results['checkpoints'][name] = dict(missing=path); continue
        results['checkpoints'][name] = probe(name, path, a.train_stories, dev_ids)
        Path(a.output).write_text(json.dumps(results, indent=2))
    Path(a.output).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

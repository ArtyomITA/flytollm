"""N0 (SUITE_TEST_FASE_6B.md section N; PIANO_TEST_RIMASTI.md): what changes between hypothetical attention reads.
Inference only, on a plain-LIF control checkpoint. For every DEV token the core is advanced one substep at a time with
the real dynamics (token current for the pre substeps, the real attention read after pre_steps, token + feedback for
the post substeps); after EVERY substep r the script computes, without feeding anything back:
  - the read-out representation (mean rate of the token so far) and its cosine distance from the previous substep and
    from the representation at the real read;
  - the attention a read at r would produce with the SAME cache: cosine of the recalled vector with the real one,
    agreement of the most attended position, Jensen-Shannon divergence of the attention weights (head-averaged);
  - spike pattern overlap between substeps r and r+k (fixed point vs cycle).
If representation and attention weights do not move between substeps, re-reading the attention cannot help.
Run (GPU idle): .venv/Scripts/python.exe phase8_n0_diagnostic.py --checkpoint-path results/<run>.latest.pt --output results/phase8_n0_<tag>.json"""
import argparse, json, math, time
from pathlib import Path
import torch
from phase6c_inference import Suite, ROOT


def attention_weights(attention, query, cache, active):
    """Head-averaged attention probabilities [B, W] of a read with `query`, and the allowed mask."""
    q = attention.rotate(attention._heads(attention.q(attention.norm(query))), cache.next_position)      # [B,H,1,Dh]
    allowed = cache.valid & (cache.positions < cache.next_position[:, None]) & active[:, None]           # [B,W]
    scores = (q @ cache.k.transpose(-1, -2)).squeeze(2) / math.sqrt(attention.head_dim)                # [B,H,W]
    scores = scores.masked_fill(~allowed[:, None, :], float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(allowed.any(-1)[:, None, None], probs, torch.zeros_like(probs))
    return probs.mean(1), allowed


def js_divergence(p, q, eps=1e-9):
    m = .5 * (p + q)
    kl = lambda a, b: (a * ((a + eps) / (b + eps)).log()).sum(-1)
    return .5 * kl(p, m) + .5 * kl(q, m)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-path', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    started = time.time()
    path = Path(a.checkpoint_path)
    suite = Suite(path.stem, path)
    model = suite.model; core, interfaces, attention = model.core, model.interfaces, model.attention
    pre, post = model.config.pre_steps, model.config.post_steps; total = pre + post
    w = core.weights()
    cos = torch.nn.functional.cosine_similarity
    acc = {k: [0.] * total for k in ('rep_cos_prev', 'rep_cos_real', 'recall_cos_real', 'argmax_agree', 'js_real', 'entropy')}
    overlap = {k: [0., 0] for k in (1, 2, 3, 4)}
    count = [0] * total
    ce_sum = ce_n = 0.
    for ids, targets in suite.windows:
        state = model.initial_state(ids.shape[1])
        for t in range(ids.shape[0]):
            tok = ids[t]
            active = tok.ne(model.config.pad); reset = tok.eq(model.config.bos)
            cache = attention.reset(state.cache, reset)
            current = interfaces.input_current(tok)
            vs = (state.voltage, state.spike); rate_sum = None; feedback = None
            reps, recalls, weights_r, spikes = [], [], [], []
            for r in range(1, total + 1):
                drive = current if feedback is None else current + feedback
                vs, rate = core.advance(drive, 1, vs, weights=w, active=active, reset=reset if r == 1 else None)
                rate_sum = rate if rate_sum is None else rate_sum + rate
                rep = interfaces.representation(vs, rate_sum / r)
                reps.append(rep); recalls.append(attention.read(rep, cache, active)); spikes.append(vs[1])
                weights_r.append(attention_weights(attention, rep, cache, active)[0])
                if r == pre:
                    # the real read of the main model: mean rate of the pre substeps, fed back for the post substeps
                    feedback = interfaces.feedback_current(recalls[-1]); rate_sum_pre = rate_sum; post_sum = None
                elif r > pre:
                    post_sum = rate if post_sum is None else post_sum + rate
            final = interfaces.representation(vs, post_sum / post)
            logits = torch.nn.functional.linear(final, interfaces.embedding.weight)
            valid = active & targets[t].ne(model.config.pad)
            if valid.any():
                ce_sum += float(torch.nn.functional.cross_entropy(logits[valid], targets[t][valid], reduction='sum')); ce_n += int(valid.sum())
            state = type(state)(vs[0], vs[1], attention.append(final, cache, active))
            has_context = (cache.valid & (cache.positions < cache.next_position[:, None])).any(-1) & active
            real = pre - 1
            for r in range(total):
                m = has_context
                if not m.any():
                    continue
                n = int(m.sum()); count[r] += n
                acc['rep_cos_real'][r] += float(cos(reps[r][m], reps[real][m]).sum())
                acc['rep_cos_prev'][r] += float(cos(reps[r][m], reps[r - 1][m]).sum()) if r else float(n)
                acc['recall_cos_real'][r] += float(cos(recalls[r][m], recalls[real][m]).sum())
                acc['argmax_agree'][r] += float((weights_r[r][m].argmax(-1) == weights_r[real][m].argmax(-1)).sum())
                acc['js_real'][r] += float(js_divergence(weights_r[r][m], weights_r[real][m]).sum())
                pr = weights_r[r][m]
                acc['entropy'][r] += float(-(pr * (pr + 1e-9).log()).sum(-1).sum())
            for k in overlap:
                for r in range(total - k):
                    x, y = spikes[r][active], spikes[r + k][active]
                    inter = (x * y).sum(-1); union = ((x + y) > 0).sum(-1).clamp_min(1)
                    overlap[k][0] += float((inter / union).sum()); overlap[k][1] += int(active.sum())
    out = dict(checkpoint=str(path), substeps=total, real_read_after=pre, ce=ce_sum / max(ce_n, 1), tokens_with_context=count[0],
               per_substep={k: [v / max(c, 1) for v, c in zip(vals, count)] for k, vals in acc.items()},
               spike_jaccard_by_lag={str(k): v[0] / max(v[1], 1) for k, v in overlap.items()},
               seconds=time.time() - started)
    Path(a.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('ce', 'tokens_with_context', 'spike_jaccard_by_lag', 'seconds')}))
    for k, v in out['per_substep'].items():
        print(k, ' '.join(f'{x:.3f}' for x in v))


if __name__ == '__main__':
    main()

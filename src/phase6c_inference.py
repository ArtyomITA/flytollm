"""Phase 6c inference suite on one checkpoint (SUITE_TEST_FASE_6B.md: A1, A2, A5, C2-C11).

Inference only; the model code is not modified: behaviour is changed by wrapping methods on the loaded instance
(KV-cache limit, feedback gate scaling, PAPA mean attention, node lesions through effective weights and masked
currents, input-token noise). DEV16 is run as one batch of 16 lanes (8 story pairs), which changes only the
order of CUDA atomic reductions relative to the training-time batch of 2."""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import pyarrow.feather as pf
from bench_runtime import ROOT
import pretrain_resumable as base
from lm_io import story_batch
from text_dataset import StoryDataset
from fly_graph import load_graph

V = 4096
CHECKPOINTS = {'fly_real_8000': 'results/pretrain_night8000_h10.latest.pt',
               'rewired_8000': 'results/phase6_rewired_h10_8000.latest.pt',
               'fly_real_26247': 'results/pretrain_continuous.latest.pt',
               'fly_real_2000': 'results/pretrain_pilot_h10.latest.pt',
               'rewired_2000': 'results/phase6_rewired_h10_2000.latest.pt',
               'configmodel_2000': 'results/phase6b_configmodel_s41_2000.latest.pt'}


def annotations(body_ids):
    t = pf.read_table(ROOT / 'dataset/male_cns/body-annotations-male-cns-v1.0.feather',
                      columns=['bodyId', 'superclass', 'class', 'type', 'dimorphism']).to_pandas().set_index('bodyId')
    t = t.loc[np.asarray(body_ids)]
    return {k: t[k].fillna('').astype(str).to_numpy() for k in ['superclass', 'class', 'type', 'dimorphism']}


class Suite:
    def __init__(self, name, path, seed=5):
        saved = torch.load(ROOT / path, map_location='cpu', weights_only=True)
        for key in ('core.src32', 'core.dst32'):  # FusedCore index buffers (phase 6c+ control runs): derived from src/dst, not part of the plain Core
            saved['model']['state_dict'].pop(key, None)
        self.model = base.load_payload(saved['model'], 'separate'); self.model.eval()
        self.name, self.path = name, path
        self.n = self.model.core.n
        graph = load_graph(10, progress=lambda m: None)
        self.ann = annotations(graph['body_ids'])
        src, dst = self.model.core.src.cpu().numpy(), self.model.core.dst.cpu().numpy()
        self.degree = np.bincount(src, minlength=self.n) + np.bincount(dst, minlength=self.n)
        self.rng = np.random.default_rng(seed)
        dev_ids = json.loads((ROOT / 'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
        with StoryDataset(ROOT / 'dataset/prepared_v1', 'validation') as ds:
            self.windows = []
            for t in range(0, 128, 16):
                xs, ys = zip(*[story_batch(ds, dev_ids[b:b + 2], [t, t], 16) for b in range(0, 16, 2)])
                self.windows.append((torch.cat([torch.as_tensor(x) for x in xs], 1).cuda(),
                                    torch.cat([torch.as_tensor(y) for y in ys], 1).cuda()))
        self.readout_nodes = self.model.interfaces.readout.nodes.cpu().numpy()
        self.port_nodes = self.model.interfaces.input.nodes.cpu().numpy()
        self.gate0 = float(self.model.interfaces.gate.item())

    # ---------------------------------------------------------------- core pass
    def run(self, kv_limit=None, gate=None, papa=None, keep=None, noise=0., record=False, noise_seed=11):
        model = self.model; attention = model.attention; interfaces = model.interfaces
        original_read, original_advance = attention.read, model.core.advance
        original_input, original_feedback = interfaces.input_current, interfaces.feedback_current
        rec = dict(recalled=[], ratio=[], rate_sum=torch.zeros(self.n, device='cuda'), active_frac=[], pop=None,
                   cx_state=[], positions=[], kc_frac=[])
        if kv_limit is not None:
            def read(provisional, cache, active=None):
                valid = cache.valid & (cache.positions >= (cache.next_position[:, None] - kv_limit))
                return original_read(provisional, cache._replace(valid=valid), active)
            attention.read = read
        collect = isinstance(papa, str) and papa == 'collect'
        if papa is not None and not collect:
            def read_papa(provisional, cache, active=None):
                return papa[None, :].expand(provisional.shape[0], -1).clone()
            attention.read = read_papa
        if collect:
            def read_collect(provisional, cache, active=None):
                out = original_read(provisional, cache, active); rec['recalled'].append(out.detach()); return out
            attention.read = read_collect
        if gate is not None:
            interfaces.gate.data.fill_(gate)
        keep_t = None
        if keep is not None:
            keep_t = torch.as_tensor(keep, device='cuda', dtype=torch.float32)
            def input_current(ids):
                return original_input(ids) * keep_t[None, :]
            def feedback_current(recalled):
                return original_feedback(recalled) * keep_t[None, :]
            interfaces.input_current, interfaces.feedback_current = input_current, feedback_current
        if record:
            holder = dict(rate=None)
            def advance(current, steps=1, state=None, **kw):
                out = original_advance(current, steps, state, **kw)
                holder['rate'] = out[1] if holder['rate'] is None else holder['rate'] + out[1]
                return out
            model.core.advance = advance
            cur = dict(inp=None)
            def input_ratio(ids):
                out = original_input(ids); cur['inp'] = out; return out
            def feedback_ratio(recalled):
                out = original_feedback(recalled)
                ports = torch.as_tensor(self.port_nodes, device='cuda')
                rec['ratio'].append((out[:, ports].abs().mean(1) / (cur['inp'][:, ports].abs().mean(1) + 1e-9)).detach())
                return out
            interfaces.input_current, interfaces.feedback_current = input_ratio, feedback_ratio
        weights = None
        if keep_t is not None:
            with torch.no_grad():
                w = model.core.weights()
                weights = w * keep_t[model.core.src] * keep_t[model.core.dst]
        types = self.ann['type'].astype(str)
        cx_mask = np.char.startswith(types, 'EPG') | np.char.startswith(types, 'PEN') | np.char.startswith(types, 'PEG')
        cx = torch.as_tensor(np.flatnonzero(cx_mask), device='cuda')
        kc = torch.as_tensor(np.flatnonzero(self.ann['class'] == 'Kenyon_Cell'), device='cuda')
        gen = torch.Generator(device='cuda'); gen.manual_seed(noise_seed)
        loss_sum = 0.; count = 0; correct = 0
        per_window = torch.zeros(16, device='cuda'); per_window_n = torch.zeros(16, device='cuda')
        per_story = torch.zeros(128, device='cuda'); per_story_n = torch.zeros(128, device='cuda')
        pop_index = None
        try:
            with torch.no_grad():
                state = model.initial_state(16)
                for w_index, (x, y) in enumerate(self.windows):
                    for pos in range(16):
                        ids = x[pos].clone(); targets = y[pos]
                        if noise > 0:
                            mask = (ids > 2) & (torch.rand(ids.shape, generator=gen, device='cuda') < noise)
                            ids = torch.where(mask, torch.randint(3, V, ids.shape, generator=gen, device='cuda'), ids)
                        if record:
                            holder['rate'] = None
                        logits, state = model.step(ids, state, weights=weights)
                        valid = (x[pos] != 0) & (targets != 0)
                        losses = torch.nn.functional.cross_entropy(logits, torch.where(valid, targets, torch.zeros_like(targets)), reduction='none')
                        losses = losses * valid
                        loss_sum += float(losses.sum()); count += int(valid.sum())
                        correct += int(((logits.argmax(-1) == targets) & valid).sum())
                        per_window[pos] += losses.sum(); per_window_n[pos] += valid.sum()
                        per_story[w_index * 16 + pos] += losses.sum(); per_story_n[w_index * 16 + pos] += valid.sum()
                        if record:
                            rate = holder['rate'] / 2.  # pre + post phases, each already averaged over its substeps
                            lanes = x[pos] != 0
                            rec['rate_sum'] += rate[lanes].sum(0)
                            rec['active_frac'].extend((rate[lanes] > 0).float().mean(1).tolist())
                            rec['kc_frac'].extend((rate[lanes][:, kc] > 0).float().mean(1).tolist())
                            rec['cx_state'].append(state.voltage[lanes][:, cx].cpu()); rec['positions'].extend([w_index * 16 + pos] * int(lanes.sum()))
                            if pop_index is None:
                                pop_index = {}
                                for key in ('superclass', 'class'):
                                    labels = self.ann[key]; uniq = sorted(set(labels) - {''})
                                    idx = torch.full((self.n,), -1, dtype=torch.long, device='cuda')
                                    for j, u in enumerate(uniq):
                                        idx[torch.as_tensor(np.flatnonzero(labels == u), device='cuda')] = j
                                    pop_index[key] = (uniq, idx)
                                rec['pop'] = {key: [] for key in pop_index}
                            for key, (uniq, idx) in pop_index.items():
                                sel = idx >= 0
                                sums = torch.zeros(len(uniq), device='cuda').index_add(0, idx[sel], rate[lanes][:, sel].sum(0))
                                counts = torch.bincount(idx[sel], minlength=len(uniq)).float() * int(lanes.sum())
                                rec['pop'][key].append((sums / counts).cpu())
        finally:
            attention.read = original_read; model.core.advance = original_advance
            interfaces.input_current, interfaces.feedback_current = original_input, original_feedback
            interfaces.gate.data.fill_(self.gate0)
        out = dict(ce=loss_sum / max(count, 1), accuracy=correct / max(count, 1), targets=count,
                   per_window=(per_window / per_window_n.clamp_min(1)).tolist(),
                   per_story=(per_story / per_story_n.clamp_min(1)).tolist())
        if record:
            out['record'] = rec
        if collect:
            out['recalled_mean'] = torch.cat(rec['recalled']).mean(0)
        return out

    # ---------------------------------------------------------------- tests
    def context_profile(self):
        result = {}
        for k in (128, 64, 32, 8, 1, 0):
            r = self.run(kv_limit=k); result[f'kv_{k}'] = dict(ce=r['ce'], accuracy=r['accuracy'])
            print(self.name, 'kv', k, round(r['ce'], 4), flush=True)
        r = self.run(gate=0.); result['gate_0'] = dict(ce=r['ce'], accuracy=r['accuracy'])
        collected = self.run(papa='collect')
        r = self.run(papa=collected['recalled_mean']); result['papa_mean'] = dict(ce=r['ce'], accuracy=r['accuracy'])
        result['baseline'] = dict(ce=collected['ce'], accuracy=collected['accuracy'], per_window=collected['per_window'],
                                  per_story=collected['per_story'])
        return result

    def lesions(self, populations):
        """populations: list of (label, node index array). Returns delta CE for the lesion and a degree-matched random control."""
        base_ce = self.run()['ce']; result = dict(baseline_ce=base_ce, lesions={})
        bins = np.searchsorted(np.quantile(self.degree, np.linspace(0, 1, 21)[1:-1]), self.degree)
        for label, nodes in populations:
            nodes = np.asarray(nodes)
            if len(nodes) == 0:
                continue
            keep = np.ones(self.n, bool); keep[nodes] = False
            lesion = self.run(keep=keep)['ce']
            control_nodes = []
            available = np.ones(self.n, bool); available[nodes] = False
            for b in np.unique(bins[nodes]):
                need = int((bins[nodes] == b).sum()); pool = np.flatnonzero((bins == b) & available)
                pick = self.rng.choice(pool, min(need, len(pool)), replace=False); control_nodes.extend(pick.tolist()); available[pick] = False
            keep_c = np.ones(self.n, bool); keep_c[control_nodes] = False
            control = self.run(keep=keep_c)['ce']
            result['lesions'][label] = dict(nodes=int(len(nodes)), lesion_ce=lesion, control_ce=control, delta=lesion - base_ce,
                                            delta_control=control - base_ce, excess=(lesion - base_ce) - (control - base_ce))
            print(self.name, 'lesion', label, len(nodes), round(lesion - base_ce, 4), round(control - base_ce, 4), flush=True)
        return result

    def activity(self):
        r = self.run(record=True); rec = r['record']
        mean_rate = (rec['rate_sum'] / max(len(rec['active_frac']), 1)).cpu().numpy()
        sorted_rate = np.sort(mean_rate); cum = np.cumsum(sorted_rate)
        gini = float(1 - 2 * (cum / cum[-1]).mean()) if cum[-1] > 0 else None
        out = dict(ce=r['ce'], global_mean_rate=float(mean_rate.mean()), fraction_never_active=float((mean_rate == 0).mean()),
                   active_fraction_per_token_mean=float(np.mean(rec['active_frac'])), active_fraction_per_token_std=float(np.std(rec['active_frac'])),
                   gini_mean_rate=gini, kc_active_fraction_mean=float(np.mean(rec['kc_frac'])),
                   feedback_to_token_current_ratio_mean=float(torch.cat(rec['ratio']).mean()) if rec['ratio'] else None,
                   gate=self.gate0, feedback_weight_norm=float(self.model.interfaces.feedback.weight.norm()),
                   input_weight_norm=float(self.model.interfaces.input.weight.norm()), populations={})
        for key, entries in rec['pop'].items():
            uniq = sorted(set(self.ann[key]) - {''}); m = torch.stack(entries).numpy()  # tokens x populations
            mean = m.mean(0); selectivity = m.std(0) / (mean + 1e-9)
            out['populations'][key] = {u: dict(nodes=int((self.ann[key] == u).sum()), mean_rate=float(mean[j]), selectivity=float(selectivity[j]),
                                               silent=bool(mean[j] < 0.1 * mean_rate.mean())) for j, u in enumerate(uniq)}
        # ring attractor probe: ridge from CX voltage to story position, cross-fit on two halves of DEV
        cx_states = torch.cat(rec['cx_state']).numpy(); positions = np.asarray(rec['positions'], dtype=np.float64)
        out['cx_position_probe'] = self.position_probe(cx_states, positions)
        rand = self.rng.choice(self.n, cx_states.shape[1], replace=False)
        out['cx_nodes'] = int(cx_states.shape[1])
        # random control population: needs another recorded pass restricted to those nodes; reuse by recording voltage subset
        out['random_position_probe'] = self.position_probe_random(rand, positions)
        return out, mean_rate

    def position_probe(self, features, positions):
        if features.shape[1] == 0:
            return None
        half = len(features) // 2; scores = []
        for a, b in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
            xa, xb = features[a], features[b]; mu, sd = xa.mean(0), xa.std(0) + 1e-6
            xa = np.c_[(xa - mu) / sd, np.ones(len(xa))]; xb = np.c_[(xb - mu) / sd, np.ones(len(xb))]
            w = np.linalg.solve(xa.T @ xa + len(xa) * np.eye(xa.shape[1]), xa.T @ positions[a])
            pred = xb @ w; scores.append(1 - ((pred - positions[b]) ** 2).mean() / positions[b].var())
        return dict(r2_crossfit=float(np.mean(scores)))

    def position_probe_random(self, nodes, positions):
        model = self.model; states = []
        with torch.no_grad():
            state = model.initial_state(16)
            for w_index, (x, y) in enumerate(self.windows):
                for pos in range(16):
                    _, state = model.step(x[pos], state)
                    lanes = x[pos] != 0
                    states.append(state.voltage[lanes][:, torch.as_tensor(nodes, device='cuda')].cpu())
        return self.position_probe(torch.cat(states).numpy(), positions)

    def kc_probe(self, train_stories=48):
        """Linear probe lag 0 on Kenyon cells only (ridge dual), train stories 0-47, eval DEV16."""
        kc = torch.as_tensor(np.flatnonzero(self.ann['class'] == 'Kenyon_Cell'), device='cuda')
        def collect(split, story_ids):
            feats, labels = [], []
            with StoryDataset(ROOT / 'dataset/prepared_v1', split) as ds, torch.no_grad():
                for b in range(0, len(story_ids), 2):
                    state = self.model.initial_state(2)
                    for t in range(0, 128, 16):
                        x, _ = story_batch(ds, story_ids[b:b + 2], [t, t], 16); x = torch.as_tensor(x).cuda()
                        for pos in range(16):
                            _, state = self.model.step(x[pos], state)
                            for lane in range(2):
                                tok = int(x[pos, lane])
                                if tok > 2:
                                    feats.append(torch.cat([state.voltage[lane, kc], state.spike[lane, kc]]).cpu()); labels.append(tok)
            return torch.stack(feats), np.asarray(labels)
        dev_ids = json.loads((ROOT / 'configs/phase3_protocol_v1.json').read_text())['validation_reserved']
        xt, yt = collect('train', list(range(train_stories))); xd, yd = collect('validation', dev_ids)
        mu = xt.mean(0, keepdim=True); sd = xt.std(0, keepdim=True) + 1e-6
        a = ((xt - mu) / sd).cuda(); b = ((xd - mu) / sd).cuda()
        y = torch.zeros(len(yt), V, device='cuda'); y[torch.arange(len(yt)), torch.as_tensor(yt).cuda()] = 1.
        gram = a @ a.T; gram.diagonal().add_(float(len(yt)))
        pred = (b @ (a.T @ torch.linalg.solve(gram, y))).argmax(1).cpu().numpy()
        return dict(kc_nodes=int(kc.numel()), accuracy_lag0=float((pred == yd).mean()), majority=float((yd == np.bincount(yt).argmax()).mean()), rows=int(len(yd)))

    def spectral(self, iterations=60):
        core = self.model.core
        with torch.no_grad():
            # 17 September 2026: the iteration oscillates with period 2 (dominant +/- pair); a single norm is one phase of
            # the oscillation, not the radius. Report the geometric mean of two consecutive norms, fixed seed.
            g = torch.Generator(device='cuda').manual_seed(0)
            w = core.weights(); v = torch.randn(self.n, device='cuda', generator=g); v /= v.norm(); history = []
            for _ in range(iterations):
                out = torch.zeros_like(v).index_add_(0, core.dst, w * v[core.src]); radius = float(out.norm()); v = out / max(radius, 1e-12); history.append(radius)
        return dict(spectral_radius_trained=(history[-1] * history[-2]) ** .5, last_two_norms=history[-2:], iterations=iterations)

    def persistence(self):
        """Correlation of the voltage with itself after 8 and 16 silent substeps from the DEV end state."""
        model = self.model
        with torch.no_grad():
            state = model.initial_state(16)
            for x, _ in self.windows:
                for pos in range(16):
                    _, state = model.step(x[pos], state)
            v0 = state.voltage.clone(); s = (state.voltage, state.spike); zero = torch.zeros_like(v0); out = {}
            for steps in (8, 16):
                (v, sp), _ = model.core.advance(zero, steps, s)
                out[f'corr_{steps}'] = float(torch.nn.functional.cosine_similarity(v.flatten(), v0.flatten(), dim=0))
                out[f'spike_fraction_after_{steps}'] = float(sp.mean())
        return out

    def noise(self):
        base_ce = self.run()['ce']; out = dict(baseline_ce=base_ce)
        for p in (.05, .1, .2):
            out[f'noise_{p}'] = self.run(noise=p)['ce'] - base_ce
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', choices=list(CHECKPOINTS), help='named checkpoint (phase 6c)')
    p.add_argument('--checkpoint-path', help='any .pt checkpoint of a plain-LIF control run (name = file stem); phase 7')
    p.add_argument('--tests', nargs='+', default=['context', 'activity', 'lesions', 'noise', 'kc', 'spectral', 'persistence'])
    p.add_argument('--output', default=None)
    a = p.parse_args()
    if bool(a.checkpoint) == bool(a.checkpoint_path):
        p.error('give exactly one of --checkpoint / --checkpoint-path')
    name, path = (a.checkpoint, CHECKPOINTS[a.checkpoint]) if a.checkpoint else (Path(a.checkpoint_path).name.replace('.pt', ''), a.checkpoint_path)
    started = time.time(); suite = Suite(name, path)
    out = dict(checkpoint=path, tests={})
    output = Path(a.output or ROOT / 'results' / f'phase6c_{name}.json')
    def save():
        out['seconds'] = time.time() - started; output.write_text(json.dumps(out, indent=2))
    if 'context' in a.tests:
        out['tests']['context'] = suite.context_profile(); save()
    if 'activity' in a.tests:
        act, mean_rate = suite.activity(); out['tests']['activity'] = act; save()
        np.save(output.with_suffix('.mean_rate.npy'), mean_rate)
    if 'lesions' in a.tests:
        pops = []
        for key in ('superclass',):
            for u in sorted(set(suite.ann[key]) - {''}):
                nodes = np.flatnonzero(suite.ann[key] == u)
                if len(nodes) >= 500:
                    pops.append((f'{key}:{u}', nodes))
        for u in ('Kenyon_Cell', 'CX', 'olfactory', 'visual'):
            pops.append((f'class:{u}', np.flatnonzero(suite.ann['class'] == u)))
        rich = np.flatnonzero(suite.degree >= 37)
        if len(rich) > 0.1 * suite.n:
            rich = np.argsort(-suite.degree)[: suite.n // 10]
        pops.append(('rich_club', rich)); pops.append(('dimorphic', np.flatnonzero(suite.ann['dimorphism'] != '')))
        out['tests']['lesions'] = suite.lesions(pops); save()
    if 'noise' in a.tests:
        out['tests']['noise'] = suite.noise(); save()
    if 'kc' in a.tests:
        out['tests']['kc_probe'] = suite.kc_probe(); save()
    if 'spectral' in a.tests:
        out['tests']['spectral'] = suite.spectral(); save()
    if 'persistence' in a.tests:
        out['tests']['persistence'] = suite.persistence(); save()
    save(); print(json.dumps({k: (v if k != 'tests' else list(v)) for k, v in out.items()}))


if __name__ == '__main__':
    main()

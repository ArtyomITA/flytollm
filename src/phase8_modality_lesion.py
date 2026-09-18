"""Q1 follow-up (SUITE_TEST_FASE_6B.md section Q): which sense carries the text? Inference only, on a plain-LIF
control checkpoint whose input ports are the anatomical sensory neurons (P0 mixed channels, P1 modal channels).

For every sensory modality (vista, meccano/udito, olfatto, propriocezione, gusto, termo/igro/chemo, altro) the DEV
cross-entropy is measured with that modality's ports lesioned, in two ways:
  info      the ports keep their token-independent baseline (the injector bias, 0.6) and lose the token information:
            what the model reads through that sense
  silence   input and feedback currents of the ports set to zero: the sense goes dark
Each lesion has a random control: the same number of ports drawn at random among all ports (seeded), same mode.
Also reported: the share of the total input drive (|current| summed over DEV) carried by each modality at baseline.
Reading: delta CE (lesion minus baseline) per modality against its random control. With modal channels (P1) a modality
that carries more than its port share is a sense the model has specialised; with mixed channels (P0) every modality is
expected to behave like its random control.
Run (GPU idle): .venv/Scripts/python.exe phase8_modality_lesion.py --checkpoint-path results/<run>.latest.pt --output results/phase8_modlesion_<tag>.json"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from phase6c_inference import Suite
from fly_interfaces_variants import MODALITIES, modality_index


@torch.no_grad()
def dev_ce(suite, keep_input=None, keep_feedback=None, baseline=None, drive=None):
    """keep_*: float vectors over the n neurons (1 keep, 0 lesion). baseline: current given to lesioned ports in
    place of the token-dependent input (None = zero). drive: tensor over n accumulating |input current| on DEV."""
    model = suite.model; interfaces = model.interfaces; loss = 0.; count = 0
    original_input, original_feedback = interfaces.input_current, interfaces.feedback_current

    def input_current(ids):
        out = original_input(ids)
        if drive is not None:
            drive.add_(out.abs().sum(0))
        if keep_input is not None:
            out = out * keep_input[None, :]
            if baseline is not None:
                out = out + baseline[None, :] * (1. - keep_input[None, :])
        return out

    def feedback_current(recalled):
        out = original_feedback(recalled)
        return out if keep_feedback is None else out * keep_feedback[None, :]

    interfaces.input_current, interfaces.feedback_current = input_current, feedback_current
    try:
        state = model.initial_state(16)
        for x, y in suite.windows:
            for pos in range(16):
                logits, state = model.step(x[pos], state)
                valid = (x[pos] != 0) & (y[pos] != 0)
                losses = torch.nn.functional.cross_entropy(logits, torch.where(valid, y[pos], torch.zeros_like(y[pos])), reduction='none')
                loss += float((losses * valid).sum()); count += int(valid.sum())
    finally:
        interfaces.input_current, interfaces.feedback_current = original_input, original_feedback
    return loss / max(count, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--modes', nargs='+', default=['info', 'silence'], choices=['info', 'silence'])
    p.add_argument('--controls', type=int, default=1, help='random port controls per modality and mode')
    p.add_argument('--seed', type=int, default=23)
    a = p.parse_args(); started = time.time()
    path = Path(a.checkpoint_path); suite = Suite(path.stem, path); model = suite.model; n = model.core.n
    injector = model.interfaces.input
    ports = injector.nodes.cpu().numpy()
    index = modality_index(suite.ann['class'], suite.ann['superclass'], ports)
    groups = {name: ports[index[ports] == k] for k, name in enumerate(MODALITIES, start=1)}
    unassigned = int((index[ports] == 0).sum())
    assert unassigned == 0, f'{unassigned} ports outside every modality (are the ports anatomical?)'
    bias_full = torch.zeros(n, device='cuda')
    if injector.bounded:
        bias_full[injector.nodes] = injector.bias.detach().float()
    ones = torch.ones(n, device='cuda')
    drive = torch.zeros(n, device='cuda')
    ce0 = dev_ce(suite, drive=drive)
    total_drive = float(drive[injector.nodes].sum())
    out = dict(checkpoint=str(path), depth=[model.config.pre_steps, model.config.post_steps], ports=int(len(ports)),
               baseline_ce=ce0, modalities={}, controls={})
    print(f'{path.stem}: baseline CE {ce0:.4f}, {len(ports)} ports', flush=True)
    for name, nodes in groups.items():
        out['modalities'][name] = dict(ports=int(len(nodes)), port_share=len(nodes) / len(ports),
                                       drive_share=float(drive[torch.as_tensor(nodes, device='cuda')].sum()) / max(total_drive, 1e-9))
    rng = np.random.default_rng(a.seed)
    for mode in a.modes:
        for name, nodes in groups.items():
            if len(nodes) == 0:
                continue
            keep = ones.clone(); keep[torch.as_tensor(nodes, device='cuda')] = 0.
            if mode == 'info':
                ce = dev_ce(suite, keep_input=keep, baseline=bias_full)
            else:
                ce = dev_ce(suite, keep_input=keep, keep_feedback=keep)
            out['modalities'][name][f'ce_{mode}'] = ce; out['modalities'][name][f'delta_{mode}'] = ce - ce0
            controls = []
            for c in range(a.controls):
                sample = rng.choice(ports, size=len(nodes), replace=False)
                keep = ones.clone(); keep[torch.as_tensor(sample, device='cuda')] = 0.
                if mode == 'info':
                    cec = dev_ce(suite, keep_input=keep, baseline=bias_full)
                else:
                    cec = dev_ce(suite, keep_input=keep, keep_feedback=keep)
                controls.append(cec - ce0)
            out['modalities'][name][f'control_delta_{mode}'] = controls
            print(f'{mode:8s} {name:18s} ports {len(nodes):6d} ({100 * len(nodes) / len(ports):5.1f}%): CE {ce:.4f} '
                  f'delta {ce - ce0:+.4f}, random control {np.mean(controls):+.4f}', flush=True)
    out['seconds'] = time.time() - started
    Path(a.output).write_text(json.dumps(out, indent=2))
    print(f'done in {out["seconds"] / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()

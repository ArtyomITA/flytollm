"""C13 (SUITE_TEST_FASE_6B.md): load ANY control-run checkpoint (core variants, looped attention, modal ports, type
encoder, frozen core / interfaces) for inference, rebuilding the model exactly as pretrain_control.py built it from the
`control` record of the run's result JSON, then loading the state dict strictly. Plain (non-fused) cores are used for
inference so that no cupy kernel is needed; the fused index buffers core.src32/dst32 are dropped.

    from phase8_load_variant import load_variant_model, VariantSuite
    model, control = load_variant_model('phase8_L8_best_T12_2000')
    suite = VariantSuite('phase8_L8_best_T12_2000')       # phase6c_inference.Suite with the variant model inside
"""
import json
from pathlib import Path
import torch
from bench_runtime import ROOT
import phase6c_inference


def _loop_from(c):
    from fly_lm_variants import LoopConfig, parse_reads, parse_schedule
    reads = c.get('attn_reads', 'default'); kv = c.get('attn_kv', 'final'); step_id = bool(c.get('attn_step_id'))
    inject = c.get('attn_inject', 'none'); token_injection = c.get('token_injection', 'all'); schedule = c.get('depth_schedule')
    output_scale = c.get('output_scale', 1.0) or 1.0; freeze_interfaces = bool(c.get('freeze_interfaces'))
    if reads == 'default' and kv == 'final' and not step_id and inject == 'none' and token_injection == 'all' and not schedule and output_scale == 1.0 and not freeze_interfaces:
        return None
    return LoopConfig(reads=parse_reads(reads, c['pre_steps'], c['post_steps']), kv=kv, step_id=step_id, inject=inject,
                      token_injection=token_injection, depth_schedule=parse_schedule(schedule), output_scale=output_scale)


def load_variant_model(run, checkpoint=None, head='separate'):
    """run: result name (results/<run>.worker.json holds the control record); checkpoint defaults to results/<run>.latest.pt."""
    from pretrain_control import build_control_cns
    import phase3_variants
    worker = ROOT / 'results' / f'{run}.worker.json'
    if worker.exists():
        c = json.loads(worker.read_text())['control']
    else:
        result = json.loads((ROOT / 'results' / f'{run}.json').read_text())
        c = result.get('control') or result['result']['control']
    homeo = tuple(float(x) for x in c['homeo'].split(',')) if c.get('homeo') else None
    arousal = (lambda f: (f[0], float(f[1]), float(f[2]), float(f[3])))(c['arousal'].split(',')) if c.get('arousal') else None
    model, data, stats = build_control_cns(10, c['rewire_seed'], c['rewire_kind'], c['ports'], c['readout'], c['core_variant'], 'off', c['chunk'], 'int64',
                                           c['pre_steps'], c['post_steps'], c.get('weight_scale', 1.0), loop=_loop_from(c), freeze_core=bool(c.get('freeze_core')),
                                           ports_seed=c.get('ports_seed', 17), ports_count=c.get('ports_count'), homeo=homeo, arousal=arousal,
                                           init_norm=c.get('init_norm', 'sum'), port_channels=c.get('port_channels', 'mixed'),
                                           port_encoder=c.get('port_encoder', 'node'), gain_groups=c.get('gain_groups', 'superclass'))
    if head == 'separate':
        model = model.add_separate_head() if hasattr(model, 'add_separate_head') else phase3_variants.variant(model, 'h1')
    if c.get('freeze_interfaces'):
        model.freeze_interfaces_()
    path = Path(checkpoint) if checkpoint else ROOT / 'results' / f'{run}.latest.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved['model']['state_dict']
    for key in ('core.src32', 'core.dst32'):
        state.pop(key, None)
    assert torch.equal(state['core.src'].long(), model.core.src.cpu().long()) and torch.equal(state['core.dst'].long(), model.core.dst.cpu().long()), 'checkpoint graph differs'
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, dict(control=c, stats=stats, checkpoint=str(path), data=data)


class VariantSuite(phase6c_inference.Suite):
    """phase6c_inference.Suite built around a variant checkpoint (same DEV windows, same helper methods)."""

    def __init__(self, run, checkpoint=None, seed=5):
        model, info = load_variant_model(run, checkpoint)
        self._variant_model, self._variant_info = model, info
        original = phase6c_inference.base.load_payload
        phase6c_inference.base.load_payload = lambda data, head: model
        try:
            super().__init__(run, Path(info['checkpoint']).relative_to(ROOT) if Path(info['checkpoint']).is_absolute() else info['checkpoint'], seed)
        finally:
            phase6c_inference.base.load_payload = original
        self.control = info['control']; self.stats = info['stats']


if __name__ == '__main__':
    import argparse, time
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--checkpoint-path')
    p.add_argument('--output')
    a = p.parse_args(); started = time.time()
    suite = VariantSuite(a.run, a.checkpoint_path)
    from phase8_c15_revert_weights import dev_ce
    ce = dev_ce(suite)
    print(f'{a.run}: variant checkpoint loaded ({suite.control["core_variant"]}, {suite.control["pre_steps"]}+{suite.control["post_steps"]}), DEV CE {ce:.4f} in {(time.time() - started) / 60:.1f} min', flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(dict(run=a.run, dev_ce=ce, control=suite.control, seconds=time.time() - started), indent=2))

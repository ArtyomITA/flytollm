"""CPU self-test of the phase-8 control variants on a tiny random graph (no GPU, seconds):
  1. LoopedFlyLM with the default loop reproduces FlyLM.step (logits, state, cache) over several tokens
  2. every loop variant runs forward + backward and leaves a finite gradient on EVERY parameter (FairCapture requires it)
  3. attention off leaves no attention / feedback parameter
  4. new core flags (cond_ports, gain_group) are identical to the LIF at init where they should be, and train
Run: .venv/Scripts/python.exe phase8_selftest.py"""
import torch
from fly_core import Core
from fly_core_variants import VariantCore
from fly_interfaces import TextInterfaces
from fly_attention import CausalAttention
from fly_lm import FlyLM, LMConfig
from fly_lm_variants import LoopedFlyLM, LoopConfig


def tiny(n=300, edges=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (edges,), generator=g); dst = torch.randint(0, n, (edges,), generator=g)
    magnitude = torch.rand(edges, generator=g) * .3 + .05
    signs = torch.where(torch.rand(n, generator=g) < .7, 1., -1.)
    sensory = torch.arange(0, 60); read = torch.arange(200, 296); groups = torch.arange(96) // 32
    return src, dst, magnitude, signs, n, sensory, read, groups


def build(cls, loop=None, core_variant=None, **core_kw):
    src, dst, magnitude, signs, n, sensory, read, groups = tiny()
    core = Core(src, dst, magnitude, signs, n) if core_variant is None else VariantCore(src, dst, magnitude, signs, n, core_variant, **core_kw)
    interfaces = TextInterfaces(n, sensory, read, groups)
    attention = CausalAttention()
    if cls is FlyLM:
        return FlyLM(core, interfaces, attention, LMConfig())
    return LoopedFlyLM(core, interfaces, attention, LMConfig(), loop)


def tokens(length=6, batch=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 4096, (length, batch), generator=g)
    ids[0] = 1  # BOS
    return ids


def check_equivalence():
    base = build(FlyLM); looped = build(LoopedFlyLM)
    looped.load_state_dict(base.state_dict())
    ids = tokens()
    a, sa = base(ids); b, sb = looped(ids)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for x, y in zip((sa.voltage, sa.spike, *sa.cache), (sb.voltage, sb.spike, *sb.cache)):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
    print('1. default loop == FlyLM: ok')


def check_gradients():
    variants = {
        'N1 two reads per_read + step id': LoopConfig(reads=(2, 6), kv='per_read', step_id=True),
        'N2 every substep per_read + concat + step id': LoopConfig(reads=tuple(range(1, 8)), kv='per_read', step_id=True, inject='concat'),
        'N3 every substep, first-read cache': LoopConfig(reads=tuple(range(1, 8)), kv='first', step_id=True, inject='concat'),
        'N1b every substep, final cache': LoopConfig(reads=tuple(range(1, 8)), kv='final'),
        'N4 two reads, no step id': LoopConfig(reads=(2, 6), kv='per_read'),
        'N5 token injected at first substep only': LoopConfig(reads=(4,), token_injection='first'),
        'attention off': LoopConfig(reads=()),
    }
    ids = tokens(); targets = tokens(seed=2)
    for name, loop in variants.items():
        model = build(LoopedFlyLM, loop)
        logits, state = model(ids)
        loss, n = model.loss(logits, targets, ids)
        loss.backward()
        missing = [k for k, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not missing, (name, missing)
        rows = state.cache.k.shape[0] // ids.shape[1]
        print(f'2. {name}: loss {float(loss):.4f}, params {sum(p.numel() for p in model.parameters())}, cache rows {rows}: ok')
        if not loop.reads:
            names = [k for k, _ in model.named_parameters()]
            assert not any(k.startswith('attention.') or k.startswith('interfaces.feedback') or k == 'interfaces.gate' for k in names), names
            print('3. attention off: no attention / feedback parameter left: ok')
    # identity at init for the options that must not change the model before training
    base = build(LoopedFlyLM, LoopConfig(reads=(2, 6), kv='per_read'))
    for loop in (LoopConfig(reads=(2, 6), kv='per_read', step_id=True), LoopConfig(reads=(2, 6), kv='per_read', inject='concat')):
        other = build(LoopedFlyLM, loop)
        other.load_state_dict(base.state_dict(), strict=False)
        torch.testing.assert_close(base(ids)[0], other(ids)[0], rtol=1e-5, atol=1e-6)
    print('2b. step id and concat adapter are the identity at init: ok')


def check_core_flags():
    ids = tokens(); targets = tokens(seed=2)
    src, dst, magnitude, signs, n, *_ = tiny()
    group_index = torch.arange(n) % 5
    lif = build(FlyLM)
    gain = build(FlyLM, core_variant='gain_group', group_index=group_index)
    gain.load_state_dict(lif.state_dict(), strict=False)
    torch.testing.assert_close(lif(ids)[0], gain(ids)[0], rtol=1e-5, atol=1e-6)
    soft = build(FlyLM, core_variant='soft_gw')
    soft.load_state_dict(lif.state_dict(), strict=False)
    torch.testing.assert_close(lif(ids)[0], soft(ids)[0], rtol=1e-5, atol=1e-6)
    a, b = build(FlyLM), build(FlyLM, core_variant='soft_gw'); b.load_state_dict(a.state_dict(), strict=False)
    for m in (a, b):
        lg, _ = m(ids); ls, _ = m.loss(lg, targets, ids); ls.backward()
    ga, gb = a.core.raw.grad, b.core.raw.grad
    print(f'4c. soft_gw: forward identical to LIF; synapses with nonzero gradient {int((ga != 0).sum())} -> {int((gb != 0).sum())} of {ga.numel()}: ok')
    assert int((gb != 0).sum()) > int((ga != 0).sum())
    for variant, kw in (('gain_group', dict(group_index=group_index)), ('cond_ports', {}), ('cond_ports+reversal', {}), ('soft_gw', {}), ('soft_gw+reversal+tau_type', dict(type_index=torch.arange(n) % 7))):
        model = build(FlyLM, core_variant=variant, **kw)
        logits, _ = model(ids); loss, _ = model.loss(logits, targets, ids); loss.backward()
        missing = [k for k, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not missing, (variant, missing)
        print(f'4. core flag {variant}: loss {float(loss):.4f}: ok')
    print('4b. gain_group is the LIF at init: ok')


def check_depth_schedule():
    ids = tokens(); targets = tokens(seed=2)

    def lm(pre, loop=None, looped=False):
        src, dst, magnitude, signs, n, sensory, read, groups = tiny()
        core = Core(src, dst, magnitude, signs, n); interfaces = TextInterfaces(n, sensory, read, groups); attention = CausalAttention()
        cfg = LMConfig(pre_steps=pre, post_steps=pre)
        return LoopedFlyLM(core, interfaces, attention, cfg, loop) if looped else FlyLM(core, interfaces, attention, cfg)

    shallow = lm(2)
    gated = lm(4, LoopConfig(reads=(4,), depth_schedule=((0, 2), (3, 4))), True)
    gated.load_state_dict(shallow.state_dict(), strict=False)
    with torch.no_grad():
        torch.testing.assert_close(shallow(ids)[0], gated(ids)[0], rtol=1e-5, atol=1e-6)
    print('5. depth schedule: maximum depth 4 gated to 2 == FlyLM 2+2: ok')
    deep = lm(4); deep.load_state_dict(shallow.state_dict())
    for _ in range(6):   # 2 calls per update: after update 3 (6 calls) the depth switches to 4
        logits, _ = gated(ids); loss, _ = gated.loss(logits, targets, ids); loss.backward(); gated.zero_grad()
    assert int(gated.calls) == 6 and float(gated.current_depth()) == 4.
    with torch.no_grad():
        before = int(gated.calls)
        torch.testing.assert_close(deep(ids)[0], gated(ids)[0], rtol=1e-5, atol=1e-6)
        assert int(gated.calls) == before
    print('5b. after the switch == FlyLM 4+4; evaluation under no_grad does not advance the counter: ok')


def check_homeo_arousal():
    ids = tokens(); targets = tokens(seed=2)
    src, dst, magnitude, signs, n, *_ = tiny()
    type_index = torch.arange(n) % 11
    lif = build(FlyLM)
    homeo = build(FlyLM, core_variant='homeo', type_index=type_index, homeo=(0.9, 1e-2))   # target far above the toy rates: every type must wake
    homeo.load_state_dict(lif.state_dict(), strict=False)
    with torch.no_grad():
        torch.testing.assert_close(lif(ids)[0], homeo(ids)[0], rtol=1e-5, atol=1e-6)
        assert float(homeo.core.thr_offset.abs().sum()) == 0.
    for _ in range(5):
        logits, _ = homeo(ids); loss, _ = homeo.loss(logits, targets, ids); loss.backward(); homeo.zero_grad()
    off = homeo.core.thr_offset
    assert int((off > 0).sum()) == off.numel(), 'homeostasis did not raise the excitability of types below target'
    assert float(off.min()) >= 0. and float(off.max()) <= 0.9
    missing = [k for k, p in homeo.named_parameters() if p.grad is None]
    print(f'6. homeo: identical to the LIF at init, frozen under no_grad; after 5 training forwards offsets in [{float(off.min()):.4f}, {float(off.max()):.4f}], types woken {int((off > 0).sum())}/{off.numel()}: ok')
    for shape in ('pulse', 'smooth'):
        model = build(FlyLM, core_variant='arousal', arousal=(shape, 0.3, 4, 1), tokens_per_update=6)
        levels = []
        for _ in range(8):
            levels.append(round(float(model.core.arousal_level()), 3))
            logits, _ = model(ids[:, :1].expand(-1, 2).contiguous() if False else ids); loss, _ = model.loss(logits, targets, ids); loss.backward(); model.zero_grad()
        print(f'6b. arousal {shape}: level per update {levels}: ok')
        assert max(levels) > 0 and min(levels) == 0


def check_separate_head():
    from phase3_variants import variant
    ids = tokens(); targets = tokens(seed=2)
    untied = variant(build(FlyLM), 'h1')
    looped = build(LoopedFlyLM).add_separate_head()
    with torch.no_grad():
        untied.head.mul_(1.5); looped.load_state_dict(untied.state_dict())
        torch.testing.assert_close(untied(ids)[0], looped(ids)[0], rtol=1e-5, atol=1e-6)
    model = build(LoopedFlyLM, LoopConfig(reads=(2, 6), kv='per_read', step_id=True)).add_separate_head()
    logits, _ = model(ids); loss, _ = model.loss(logits, targets, ids); loss.backward()
    missing = [k for k, p in model.named_parameters() if p.grad is None]
    assert not missing and model.head.grad.abs().sum() > 0, missing
    print('7. separate output head: looped default == trainer Untied model; head trained in the looped variants: ok')


def check_output_scale_and_freeze():
    ids = tokens(); targets = tokens(seed=2)
    base = build(LoopedFlyLM)
    scaled = build(LoopedFlyLM, LoopConfig(reads=(4,), output_scale=0.1)); scaled.load_state_dict(base.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(base(ids)[0] * 0.1, scaled(ids)[0], rtol=1e-5, atol=1e-6)
    print('8. output scale 0.1 multiplies the logits: ok')
    frozen = build(LoopedFlyLM).add_separate_head().freeze_interfaces_()
    names = [n for n, _ in frozen.named_parameters()]
    assert names == ['core.raw'], names
    logits, _ = frozen(ids); loss, _ = frozen.loss(logits, targets, ids); loss.backward()
    assert frozen.core.raw.grad is not None and torch.isfinite(frozen.core.raw.grad).all()
    print('8b. freeze interfaces: only core.raw trainable, gradient finite: ok')


def check_port_variants():
    from fly_interfaces_variants import modal_channels_, TypeSharedInjector
    ids = tokens(); targets = tokens(seed=2)
    model = build(FlyLM)
    ports = model.interfaces.input.nodes.numel()
    modality = torch.arange(ports) % 4 + 1        # 4 fake modalities, ids 1..4
    info = modal_channels_(model.interfaces.input, modality, 256, 8, 17)
    ch = model.interfaces.input.channels
    block = 256 // 4
    for g in range(1, 5):
        sel = ch[modality == g]
        lo = (g - 1) * block; hi = 256 if g == 4 else lo + block
        assert int(sel.min()) >= lo and int(sel.max()) < hi, (g, int(sel.min()), int(sel.max()))
    logits, _ = model(ids); loss, _ = model.loss(logits, targets, ids); loss.backward()
    print(f'9. modal channels: every port reads only its modality block ({len(info)} blocks), training ok')
    model = build(FlyLM)
    group = torch.arange(ports) % 7
    model.interfaces.input = TypeSharedInjector(model.interfaces.input, group, 17)
    out = model.interfaces.input(torch.randn(2, 256))
    assert out.shape == (2, model.core.n) and torch.isfinite(out).all()
    logits, _ = model(ids); loss, _ = model.loss(logits, targets, ids); loss.backward()
    assert model.interfaces.input.weight.grad is not None and model.interfaces.input.weight.shape == (7, 8)
    print('9b. type-shared injector: 7 shared channels of fan-in 8, forward finite, gradient on the shared weights: ok')


if __name__ == '__main__':
    torch.manual_seed(0)
    check_port_variants()
    check_output_scale_and_freeze()
    check_separate_head()
    check_homeo_arousal()
    check_depth_schedule()
    check_equivalence()
    check_gradients()
    check_core_flags()
    print('ALL OK')

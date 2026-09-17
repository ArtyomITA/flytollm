"""Phase 8b (user, 17 September 2026 18:30: "fai per ogni categoria i corti ... dove si richiedono modifiche anche al
core, falle comunque"): the short tests that need new control code, on the safe default candidate (standard + Muon 1e-3,
4.275 at 2000). Section M of SUITE_TEST_FASE_6B.md. Every configuration is fixed: no automatic choice in this queue.
New code (main model files untouched): fly_lm_variants.LoopedFlyLM (attention inside the loop of substeps),
fly_core_variants flags cond_ports / gain_group, pretrain_control --attn-* --token-injection --freeze-core --ports-seed
--ports-count, phase8_n0_diagnostic.py. CPU self-test: phase8_selftest.py (passed 17 September 18:41).
A smoke (4 updates) runs before each code family; a failed smoke skips its runs. One GPU worker, hourly 3-minute rest
handled by pretrain_night_queue.run_job, pausable with pause_runs.py. Measured duration of a 2000-update run at 4+4:
15 min; estimated total about 5 h."""
import ctypes, json, subprocess, time
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD

STOP = ROOT / 'STOP_PHASE7'
DEFAULT = STANDARD + ['--optimizer', 'muon', '--muon-lr', 1e-3]
N1 = ['--attn-reads', '2,6', '--attn-kv', 'per_read', '--attn-step-id']
N2 = ['--attn-reads', 'every', '--attn-kv', 'per_read', '--attn-step-id', '--attn-inject', 'concat']
N3 = ['--attn-reads', 'every', '--attn-kv', 'first', '--attn-step-id', '--attn-inject', 'concat']
N4 = ['--attn-reads', '2,6', '--attn-kv', 'per_read']
N5 = ['--token-injection', 'first']
OFF = ['--attn-reads', 'off']
COND = ['--core-variant', 'cond_ports']
GAIN = ['--core-variant', 'gain_group', '--type-param-lr', 1e-2]
SOFT = ['--core-variant', 'soft_gw']
SHOCK = ['--pre-steps', 12, '--post-steps', 12, '--depth-schedule', '0:12,500:8,1000:4,1500:12']
HOMEO = ['--core-variant', 'homeo', '--homeo', '0.02,2e-5']
FREEZE = ['--freeze-core']
IDENTITY = ['--weight-scale', 1e-6, '--freeze-core']
REAL10 = ['--rewire-kind', 'none']      # original graph at threshold 10 (the factorial uses the pair real / degree-preserving rewire)
REWIRED = ['--rewire-kind', 'degree']

# (run name, extra args, smoke family)
MICRO = ['--eval-every', 100]
# (run name, extra args, smoke family[, updates]); micro runs: 300 updates, evaluation every 100, to find the synaptic
# learning rate BEFORE spending 2000-update runs (user request of 17 September 2026 20:30)
RUNS = [
    ('phase8_micro_default_300', MICRO, None, 300),
    ('phase8_micro_corelr1e3_300', MICRO + ['--core-lr', 1e-3], 'corelr', 300),
    ('phase8_micro_corelr3e3_300', MICRO + ['--core-lr', 3e-3], 'corelr', 300),
    ('phase8_micro_corelr1e2_300', MICRO + ['--core-lr', 1e-2], 'corelr', 300),
    ('phase8_micro_corelr3e2_300', MICRO + ['--core-lr', 3e-2], 'corelr', 300),
    ('phase8_micro_soft_corelr1e3_300', MICRO + SOFT + ['--core-lr', 1e-3], 'softgw', 300),
    ('phase8_micro_soft_corelr3e3_300', MICRO + SOFT + ['--core-lr', 3e-3], 'softgw', 300),
    ('phase8_micro_soft_corelr1e2_300', MICRO + SOFT + ['--core-lr', 1e-2], 'softgw', 300),
    ('phase8_micro_soft_corelr3e2_300', MICRO + SOFT + ['--core-lr', 3e-2], 'softgw', 300),
    ('phase8_C14_core_lr1e3_2000', ['--core-lr', 1e-3], 'corelr'),   # C14: the synaptic weights move 0.3% in 8000 updates at lr 1e-4
    ('phase8_C14_core_lr1e2_2000', ['--core-lr', 1e-2], 'corelr'),
    ('phase8_C16_soft_gw_2000', SOFT, 'softgw'),                      # C16: wider weight gradient (soft presynaptic activity in the backward)
    ('phase8_C16_soft_gw_corelr1e3_2000', SOFT + ['--core-lr', 1e-3], 'softgw'),
    ('phase8_C16_soft_gw_corelr1e2_2000', SOFT + ['--core-lr', 1e-2], 'softgw'),
    ('phase8_N6d_depth_shock_2000', SHOCK, 'shock'),                  # user idea: 12+12 -> 8+8 -> 4+4 -> shock back to 12+12, 500 updates each
    ('phase8_H1_homeo_2000', HOMEO, 'homeo'),                          # wake-up homeostasis per cell type (user: 'mi piace tantissimo')
    ('phase8_H2_arousal_pulse_2000', ['--core-variant', 'arousal', '--arousal', 'pulse,0.3,500,100'], 'arousal'),
    ('phase8_H3_arousal_smooth_2000', ['--core-variant', 'arousal', '--arousal', 'smooth,0.3,500,0'], 'arousal'),
    ('phase8_H4_homeo_T12_2000', HOMEO + ['--pre-steps', 12, '--post-steps', 12], 'homeo'),
    ('phase8_N1_reads2_2000', N1, 'loop'),
    ('phase8_N2_every_2000', N2, 'loop'),
    ('phase8_N5_inject_first_2000', N5, 'loop'),
    ('phase8_M4_cond_ports_2000', COND, 'cond'),
    ('phase8_A272_gain_group_2000', GAIN, 'gain'),
    ('phase8_B253a_identity_2000', IDENTITY, 'freeze'),
    ('phase8_B10_reservoir_2000', FREEZE, 'freeze'),
    ('phase8_N4_reads2_noid_2000', N4, 'loop'),
    ('phase8_N3_every_firstkv_2000', N3, 'loop'),
    ('phase8_B8_ports_seed23_2000', ['--ports-seed', 23], 'ports'),
    ('phase8_B9a_ports2639_2000', ['--ports-count', 2639], 'ports'),
    ('phase8_B9b_ports1733_2000', ['--ports-count', 1733], 'ports'),
    ('phase8_B252a_real_on_2000', REAL10, None),
    ('phase8_B252b_real_off_2000', REAL10 + OFF, 'off'),
    ('phase8_B252c_rewired_on_2000', REWIRED, None),
    ('phase8_B252d_rewired_off_2000', REWIRED + OFF, 'off'),
]
SMOKES = dict(homeo=HOMEO, arousal=['--core-variant', 'arousal', '--arousal', 'smooth,0.3,500,0'], shock=SHOCK, corelr=['--core-lr', 1e-2], softgw=SOFT + ['--core-lr', 1e-2], loop=N2, cond=COND, gain=GAIN, freeze=IDENTITY, ports=['--ports-count', 1733, '--ports-seed', 23], off=OFF)


def diagnostic(state, checkpoint, tag, script='phase8_n0_diagnostic.py', prefix='n0', extra=()):
    output = ROOT / 'results' / f'phase8_{prefix}_{tag}.json'
    if q.STOP.exists():
        raise InterruptedError('User stop file present')
    if output.exists() or not checkpoint.exists():
        state['completed'].append(dict(name=f'{prefix}_{tag}', ok=output.exists(), skipped='output exists' if output.exists() else 'checkpoint missing', finished=time.time()))
        q.write_state(state); return
    print(json.dumps(dict(event='start', name=f'{prefix}_{tag}')), flush=True)
    state.update(status='running', current=f'{prefix}_{tag}', started=time.time()); q.write_state(state)
    with (ROOT / 'results/phase8_n0.console.log').open('a', encoding='utf8') as log:
        r = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-u', str(ROOT / script), *map(str, extra),
                            '--checkpoint-path', str(checkpoint.relative_to(ROOT)), '--output', str(output)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=3600, creationflags=subprocess.CREATE_NO_WINDOW)
    ok = r.returncode == 0 and output.exists()
    state['completed'].append(dict(name=f'{prefix}_{tag}', ok=ok, finished=time.time())); q.write_state(state)
    print(json.dumps(dict(event='end', name=f'{prefix}_{tag}', ok=ok)), flush=True)
    if not ok:
        state['failed'].append(dict(name=f'{prefix}_{tag}', error=f'exit {r.returncode}')); q.write_state(state)


def main():
    q.STATE = ROOT / 'results/phase8b_live.json'; q.STOP = STOP
    if q.STATE.exists():
        # resumable (user, 17 September 2026 20:40: the PC is needed around 22:12-22:30, the queue is stopped at a job
        # boundary with STOP_PHASE7 and relaunched later): a finished queue or a live one is never relaunched
        previous = json.loads(q.STATE.read_text())
        if previous.get('status') in ('running', 'completed'):
            raise RuntimeError(f"Phase8b queue state is {previous.get('status')}; do not launch twice")
        q.STATE.replace(q.STATE.with_name(f'phase8b_live.stopped_{int(time.time())}.json'))
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#M')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        diagnostic(state, ROOT / 'results/phase7_promote_muon1e3_8000.latest.pt', 'default8000')
        diagnostic(state, ROOT / 'results/phase7_promote_muon1e3_8000.latest.pt', 'default8000', script='phase8_c15_revert_weights.py', prefix='c15', extra=('--kind', 'relthr'))
        for tag, ckpt in (('trained4', 'phase7_lever_muon1e3_2000'), ('trained8', 'phase8_L4_T8_2000'), ('trained12', 'phase8_L5_T12_2000')):
            diagnostic(state, ROOT / f'results/{ckpt}.latest.pt', tag, script='phase8_depth_transfer.py', prefix='depth')  # N6c: depth-schedule viability
        smoke_ok = {}
        def done(run):
            path = ROOT / 'results' / f'{run}.json'
            return path.exists() and json.loads(path.read_text()).get('ok') is True

        for name, extra, family, *rest in RUNS:
            updates = rest[0] if rest else 2000
            if done(name):
                state['completed'].append(dict(name=name, ok=True, skipped='result exists (resumed queue)', finished=time.time())); q.write_state(state)
                continue
            if family is not None and family not in smoke_ok and done(f'phase8_smoke_{family}'):
                smoke_ok[family] = True
            if family is not None and family not in smoke_ok:
                try:
                    q.run_job(state, f'phase8_smoke_{family}', 'pretrain_control', DEFAULT + SMOKES[family] + ['--updates', 4, '--checkpoint-every', 4], 1800)
                    smoke_ok[family] = True
                except RuntimeError as exc:
                    smoke_ok[family] = False
                    state['failed'].append(dict(name=f'phase8_smoke_{family}', error=str(exc))); q.write_state(state)
            if family is not None and not smoke_ok[family]:
                state['failed'].append(dict(name=name, error=f'smoke {family} failed: skipped')); q.write_state(state)
                continue
            try:
                q.run_job(state, name, 'pretrain_control', DEFAULT + extra + ['--updates', updates, '--checkpoint-every', updates], 14400)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

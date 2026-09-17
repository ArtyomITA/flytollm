"""Phase 8b2 (user, 18 September 2026 00:55: "tutti i 4+4 rendili 8+8"): the runs of queue 8b that had not started yet,
now on the base standard + Muon 1e-3 + 8+8 substeps (reference L4 = 4.187 at 2000, 27 min per run). Run names carry _T8.
Unchanged by design: the depth shock (maximum 12) and the homeostasis run at 12+12. The two-read variants read after
substeps 4 and 12 of 16 (they were 2 and 6 of 8). Done at 4+4 before the decision and kept: the M0 micro-sweep and C14
(synaptic lr 1e-3 and 1e-2). Generated from phase8b_queue.py by make_8b2.py; same structure, resumable, fixed
configurations, no automatic choice. Section M of SUITE_TEST_FASE_6B.md."""
import ctypes, json, subprocess, time
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase7_queue import STANDARD

STOP = ROOT / 'STOP_PHASE7'
DEFAULT = STANDARD + ['--optimizer', 'muon', '--muon-lr', 1e-3]
T8 = ['--pre-steps', 8, '--post-steps', 8]
N1 = ['--attn-reads', '4,12', '--attn-kv', 'per_read', '--attn-step-id']
N2 = ['--attn-reads', 'every', '--attn-kv', 'per_read', '--attn-step-id', '--attn-inject', 'concat']
N3 = ['--attn-reads', 'every', '--attn-kv', 'first', '--attn-step-id', '--attn-inject', 'concat']
N4 = ['--attn-reads', '4,12', '--attn-kv', 'per_read']
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
# (run name, extra args, smoke family)
RUNS = [
    ('phase8_C14_core_lr1e2_T8_2000', ['--core-lr', 1e-2], 'corelr'),
    ('phase8_C16_soft_gw_T8_2000', SOFT, 'softgw'),                      # C16: wider weight gradient (soft presynaptic activity in the backward)
    ('phase8_C16_soft_gw_corelr1e3_T8_2000', SOFT + ['--core-lr', 1e-3], 'softgw'),
    ('phase8_C16_soft_gw_corelr1e2_T8_2000', SOFT + ['--core-lr', 1e-2], 'softgw'),
    ('phase8_N6d_depth_shock_2000', SHOCK, 'shock'),                  # user idea: 12+12 -> 8+8 -> 4+4 -> shock back to 12+12, 500 updates each
    ('phase8_H1_homeo_T8_2000', HOMEO, 'homeo'),                          # wake-up homeostasis per cell type (user: 'mi piace tantissimo')
    ('phase8_H2_arousal_pulse_T8_2000', ['--core-variant', 'arousal', '--arousal', 'pulse,0.3,500,100'], 'arousal'),
    ('phase8_H3_arousal_smooth_T8_2000', ['--core-variant', 'arousal', '--arousal', 'smooth,0.3,500,0'], 'arousal'),
    ('phase8_H4_homeo_T12_2000', HOMEO + ['--pre-steps', 12, '--post-steps', 12], 'homeo'),
    ('phase8_N1_reads2_T8_2000', N1, 'loop'),
    ('phase8_N2_every_T8_2000', N2, 'loop'),
    ('phase8_N5_inject_first_T8_2000', N5, 'loop'),
    ('phase8_M4_cond_ports_T8_2000', COND, 'cond'),
    ('phase8_A272_gain_group_T8_2000', GAIN, 'gain'),
    ('phase8_B253a_identity_T8_2000', IDENTITY, 'freeze'),
    ('phase8_B10_reservoir_T8_2000', FREEZE, 'freeze'),
    ('phase8_N4_reads2_noid_T8_2000', N4, 'loop'),
    ('phase8_N3_every_firstkv_T8_2000', N3, 'loop'),
    ('phase8_B8_ports_seed23_T8_2000', ['--ports-seed', 23], 'ports'),
    ('phase8_B9a_ports2639_T8_2000', ['--ports-count', 2639], 'ports'),
    ('phase8_B9b_ports1733_T8_2000', ['--ports-count', 1733], 'ports'),
    ('phase8_B252a_real_on_T8_2000', REAL10, None),
    ('phase8_B252b_real_off_T8_2000', REAL10 + OFF, 'off'),
    ('phase8_B252c_rewired_on_T8_2000', REWIRED, None),
    ('phase8_B252d_rewired_off_T8_2000', REWIRED + OFF, 'off'),
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
    q.STATE = ROOT / 'results/phase8b2_live.json'; q.STOP = STOP
    if q.STATE.exists():
        # resumable (user, 17 September 2026 20:40: the PC is needed around 22:12-22:30, the queue is stopped at a job
        # boundary with STOP_PHASE7 and relaunched later): a finished queue or a live one is never relaunched
        previous = json.loads(q.STATE.read_text())
        if previous.get('status') in ('running', 'completed'):
            raise RuntimeError(f"Phase8b2 queue state is {previous.get('status')}; do not launch twice")
        q.STATE.replace(q.STATE.with_name(f'phase8b2_live.stopped_{int(time.time())}.json'))
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
            if family is not None and family not in smoke_ok and done(f'phase8_smoke_{family}_T8'):
                smoke_ok[family] = True
            if family is not None and family not in smoke_ok:
                try:
                    q.run_job(state, f'phase8_smoke_{family}_T8', 'pretrain_control', DEFAULT + T8 + SMOKES[family] + ['--updates', 4, '--checkpoint-every', 4], 1800)
                    smoke_ok[family] = True
                except RuntimeError as exc:
                    smoke_ok[family] = False
                    state['failed'].append(dict(name=f'phase8_smoke_{family}_T8', error=str(exc))); q.write_state(state)
            if family is not None and not smoke_ok[family]:
                state['failed'].append(dict(name=name, error=f'smoke {family} failed: skipped')); q.write_state(state)
                continue
            try:
                q.run_job(state, name, 'pretrain_control', DEFAULT + T8 + extra + ['--updates', updates, '--checkpoint-every', updates], 21600)
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

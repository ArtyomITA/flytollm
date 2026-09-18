"""Phase 8e (user, 18 September 2026 01:45: yes to the four P proposals): input channels by sensory modality, section Q
of SUITE_TEST_FASE_6B.md. Base: standard + Muon 1e-3 + 8+8 (27 min per run). Fixed configurations, no automatic choice,
resumable, one smoke per code family. Run by phase8c_queue.main() after phase8d_queue.main() and before the 8c runs.

  P0   reference: full anatomical sensory ports (17,937), mixed channels, at 8+8 + Muon (did not exist at this base)
  P1   modal channels: every port reads only the embedding block of its modality (7 blocks of 36 channels)
  P2   type-shared encoder: ports of the same sensory cell type share channels, weight and bias (369 types + untyped)
  P3   P1 + one trainable gain per modality on the external drive (core flag gain_group, lr 1e-2)
  P4   ports = multisensory convergence neurons (reached by every modality within 4 synapses on the standard graph: 8,129)
  P4b  control: 8,129 random ports"""
import ctypes, json, time
import pretrain_night_queue as q
from bench_runtime import ROOT
from phase8b_queue import DEFAULT, diagnostic

STOP = ROOT / 'STOP_PHASE7'
T8 = ['--pre-steps', 8, '--post-steps', 8]
ANAT = T8 + ['--ports', 'anatomical']
RUNS = [
    ('phase8_P0_anatomical_T8_2000', ANAT, 'anat'),
    ('phase8_P1_modal_channels_T8_2000', ANAT + ['--port-channels', 'modal'], 'modal'),
    ('phase8_P2_type_encoder_T8_2000', ANAT + ['--port-encoder', 'type'], 'typeenc'),
    ('phase8_P3_modal_gain_T8_2000', ANAT + ['--port-channels', 'modal', '--core-variant', 'gain_group', '--gain-groups', 'modality', '--type-param-lr', 1e-2], 'modalgain'),
    ('phase8_P4_convergent_T8_2000', T8 + ['--ports', 'convergent'], 'conv'),
    ('phase8_P4b_random8129_T8_2000', T8 + ['--ports', 'random_matched', '--ports-count', 8129], 'anat'),
    # user yes, 18 September 18:25 (after C15b: with anatomical ports the contribution of the trained synapses grows with
    # training, 0.008 / 0.089 / 0.243 at 2000 / 8000 / 26,247 updates in the old standard, 0.0006 with random ports):
    # the anatomical ports inside the current default (Muon 1e-3, 8+8) promoted to 8000 updates, and its twin with the
    # smooth arousal wave (H3: stable, +0.031 at 2000, moved synapses x2); ~100 min each; C15 on both at the end
    ('phase8_P0L_anatomical_T8_8000', ANAT, 'anat', 8000),
    ('phase8_P0A_anatomical_arousal_T8_8000', ANAT + ['--core-variant', 'arousal', '--arousal', 'smooth,0.3,500,0'], 'anatarousal', 8000),
]
SMOKES = dict(anat=ANAT, modal=ANAT + ['--port-channels', 'modal'], typeenc=ANAT + ['--port-encoder', 'type'],
              modalgain=ANAT + ['--port-channels', 'modal', '--core-variant', 'gain_group', '--gain-groups', 'modality', '--type-param-lr', 1e-2],
              conv=T8 + ['--ports', 'convergent'],
              anatarousal=ANAT + ['--core-variant', 'arousal', '--arousal', 'smooth,0.3,500,0'])


def done(run):
    path = ROOT / 'results' / f'{run}.json'
    return path.exists() and json.loads(path.read_text()).get('ok') is True


def main():
    q.STATE = ROOT / 'results/phase8e_live.json'; q.STOP = STOP
    if q.STATE.exists():
        previous = json.loads(q.STATE.read_text())
        if previous.get('status') in ('running', 'completed'):
            raise RuntimeError(f"Phase8e queue state is {previous.get('status')}; do not launch twice")
        q.STATE.replace(q.STATE.with_name(f'phase8e_live.stopped_{int(time.time())}.json'))
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#Q')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        smoke_ok = {}
        for name, extra, family, *rest in RUNS:
            updates = rest[0] if rest else 2000
            if done(name):
                state['completed'].append(dict(name=name, ok=True, skipped='result exists (resumed queue)', finished=time.time())); q.write_state(state)
                continue
            if family not in smoke_ok:
                if done(f'phase8_smoke_{family}'):
                    smoke_ok[family] = True
                else:
                    try:
                        q.run_job(state, f'phase8_smoke_{family}', 'pretrain_control', DEFAULT + SMOKES[family] + ['--updates', 4, '--checkpoint-every', 4], 1800)
                        smoke_ok[family] = True
                    except RuntimeError as exc:
                        smoke_ok[family] = False
                        state['failed'].append(dict(name=f'phase8_smoke_{family}', error=str(exc))); q.write_state(state)
            if not smoke_ok[family]:
                state['failed'].append(dict(name=name, error=f'smoke {family} failed: skipped')); q.write_state(state)
                continue
            try:
                q.run_job(state, name, 'pretrain_control', DEFAULT + extra + ['--updates', updates, '--checkpoint-every', 2000], 21600 if updates == 2000 else 43200)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state)
        # 18 September 02:10 (added, not a change of the runs above): which sense carries the text? lesion by modality at
        # inference on the P0 (mixed channels) and P1 (modal channels) checkpoints, ~15 min each (29 DEV passes at 8+8)
        for tag, run in (('P0_anat', 'phase8_P0_anatomical_T8_2000'), ('P1_modal', 'phase8_P1_modal_channels_T8_2000')):
            diagnostic(state, ROOT / 'results' / f'{run}.latest.pt', tag, script='phase8_modality_lesion.py', prefix='modlesion')
        # how much of the result is carried by the trained synapses (C15) on the two 8000-update anatomical runs and on P0
        for run in ('phase8_P0_anatomical_T8_2000', 'phase8_P0L_anatomical_T8_8000', 'phase8_P0A_anatomical_arousal_T8_8000'):
            diagnostic(state, ROOT / 'results' / f'{run}.latest.pt', run.replace('phase8_', ''), script='phase8_c15_revert_weights.py', prefix='c15', extra=('--run', run))
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state); raise
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()

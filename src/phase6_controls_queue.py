"""Phase6 controls: rewired-graph fly and standard baselines at equal data. Sequential GPU jobs."""
import argparse, ctypes, json, math, time
from pathlib import Path
import pretrain_night_queue as q
from bench_runtime import ROOT

STOP = ROOT / 'STOP_PHASE6'


def score(result):
    values = [r['ce'] for r in result['repeated_dev']]
    assert values and all(math.isfinite(v) for v in values)
    return sum(values) / len(values)


def existing(name):
    path = ROOT / 'results' / f'{name}.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get('result') if data.get('ok') else None


def curve_at(result, update):
    for entry in result.get('curve', []):
        if entry['update'] == update:
            return entry
    return None


def summary(state):
    rows = {}
    pilot = existing('pretrain_pilot_h10'); night = existing('pretrain_night8000_h10')
    review = json.loads((ROOT / 'results/phase35_4_final_review.json').read_text())
    if pilot:
        rows['fly_real_h10_seed17'] = dict(update=2000, targets=pilot['targets'], ce=score(pilot), parameters=pilot['parameters'],
                                          mean_step_s=pilot['mean_step_s'], baseline=pilot['curve'][-1]['baseline'])
    rows['fly_real_h10_seed23'] = dict(update=2000, targets=review['replica']['targets'], ce=review['replica']['ce'])
    if night:
        rows['fly_real_h10_8000'] = dict(update=8000, targets=night['targets'], ce=score(night), parameters=night['parameters'],
                                        mean_step_s=night['mean_step_s'])
    for name in ['phase6_rewired_h10_2000', 'phase6_rewired_h10_8000']:
        r = existing(name)
        if r:
            rows[name] = dict(update=r['updates'], targets=r['targets'], ce=score(r), parameters=r['parameters'],
                              mean_step_s=r['mean_step_s'], rewire=r.get('control', {}).get('stats'),
                              artificial_initial_sha256=r.get('artificial_initial_sha256'))
    for name in ['phase6_gru_lr1e-4', 'phase6_transformer_lr1e-4', 'phase6_gru_lr1e-3', 'phase6_transformer_lr1e-3']:
        r = existing(name)
        if r:
            for update in (2000, 8000):
                entry = curve_at(r, update)
                if entry:
                    rows[f'{name}_{update}'] = dict(update=update, targets=entry['targets'], ce=entry['dev']['ce'],
                                                    accuracy=entry['dev']['accuracy'], parameters=r['parameters'],
                                                    mean_step_s=r['mean_step_s'], baseline=entry['baseline'])
    if pilot:
        rows['artificial_init_match'] = dict(fly=pilot.get('artificial_initial_sha256'),
                                             rewired=rows.get('phase6_rewired_h10_2000', {}).get('artificial_initial_sha256'))
    out = dict(protocol='PROTOCOLLO_FASE_6_CONTROLLI.md', generated=time.time(), rows=rows, jobs=state['completed'])
    (ROOT / 'results/phase6_controls_summary.json').write_text(json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--summary-only', action='store_true'); a = p.parse_args()
    q.STATE = ROOT / 'results/phase6_controls_live.json'; q.STOP = STOP
    if a.summary_only:
        state = json.loads(q.STATE.read_text()) if q.STATE.exists() else dict(completed=[])
        print(json.dumps(summary(state), indent=2)); return
    if q.STATE.exists():
        raise RuntimeError('Phase6 queue state exists; do not launch twice')
    state = dict(status='starting', completed=[], protocol='PROTOCOLLO_FASE_6_CONTROLLI.md')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    try:
        first = q.run_job(state, 'phase6_rewired_h10_2000', 'pretrain_control',
                          ['--threshold', 10, '--head', 'separate', '--seed', 17, '--rewire-seed', 41, '--updates', 2000,
                           '--eval-every', 500, '--checkpoint-every', 2000], 7200)
        for model in ('gru', 'transformer'):
            for lr in ('1e-4', '1e-3'):
                q.run_job(state, f'phase6_{model}_lr{lr}', 'baseline_lm',
                          ['--model', model, '--lr', lr, '--seed', 17, '--updates', 8000, '--eval-every', 500], 7200)
        checkpoint = Path(first['checkpoint'])
        q.run_job(state, 'phase6_rewired_h10_8000', 'pretrain_control',
                  ['--threshold', 10, '--head', 'separate', '--seed', 17, '--rewire-seed', 41, '--updates', 8000,
                   '--eval-every', 500, '--checkpoint-every', 2000,
                   '--resume', str(checkpoint if checkpoint.is_absolute() else ROOT / checkpoint)], 18000)
        state.update(status='completed'); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        summary(state)


if __name__ == '__main__':
    main()

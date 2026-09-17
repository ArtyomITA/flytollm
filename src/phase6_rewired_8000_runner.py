"""Phase 6 follow-up: finish the rewired control to 8000 updates after the resume equivalence assertion failed.

Strategy, fixed before running:
1. Retry the resume from the 2000-update checkpoint once (the CUDA-graph/eager equivalence check compares an
   Adam first step from zeroed moments, which is a discontinuous function of near-zero gradients under
   nondeterministic atomics; a retry can pass or fail again).
2. If it fails again with the same AssertionError, run the rewired control from scratch to 8000 updates.
   Resume restores model, optimizer, cursor, counts and recurrent state exactly, so a fresh 8000-update run
   follows the same trajectory as 2000 + resume up to atomic-order noise.
No source file listed in pretrain_resumable.SOURCES is modified: existing checkpoints stay resumable.
"""
import ctypes, json, time
from pathlib import Path
import pretrain_night_queue as q
import phase6_controls_queue as p6
from bench_runtime import ROOT

NAME = 'phase6_rewired_h10_8000'
COMMON = ['--threshold', 10, '--head', 'separate', '--seed', 17, '--rewire-seed', 41, '--updates', 8000,
          '--eval-every', 500, '--checkpoint-every', 2000]


def archive(tag):
    """Move the failed job outputs aside so run_job can write fresh ones."""
    moved = []
    for suffix in ('.json', '.worker.json', '.console.log', '.jsonl', '.best.pt', '.latest.pt'):
        path = ROOT / 'results' / f'{NAME}{suffix}'
        if path.exists():
            target = ROOT / 'results' / f'{NAME}.{tag}{suffix}'
            path.replace(target); moved.append(target.name)
    return moved


def failed_equivalence():
    worker = ROOT / 'results' / f'{NAME}.worker.json'
    if not worker.exists():
        return False
    data = json.loads(worker.read_text())
    return data.get('error_type') == 'AssertionError' and 'not close' in data.get('error', '')


def main():
    q.STATE = ROOT / 'results/phase6_controls_live.json'; q.STOP = p6.STOP
    state = json.loads(q.STATE.read_text())
    assert state.get('status') == 'failed', f"unexpected queue status {state.get('status')!r}"
    state['rewired_8000_followup'] = dict(archived_first_failure=archive('failed_resume_1'), started=time.time())
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    checkpoint = ROOT / 'results/phase6_rewired_h10_2000.latest.pt'
    try:
        try:
            q.run_job(state, NAME, 'pretrain_control', COMMON + ['--resume', str(checkpoint)], 18000)
            state['rewired_8000_followup']['mode'] = 'resume_retry'
        except RuntimeError:
            if not failed_equivalence():
                raise
            state['rewired_8000_followup']['archived_second_failure'] = archive('failed_resume_2')
            state['rewired_8000_followup']['mode'] = 'fresh_8000'
            q.write_state(state)
            q.run_job(state, NAME, 'pretrain_control', COMMON, 18000)
        state.update(status='completed'); state.pop('error', None); q.write_state(state)
    except InterruptedError as exc:
        state.update(status='stopped', reason=str(exc)); q.write_state(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc)); q.write_state(state); raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        p6.summary(state)


if __name__ == '__main__':
    main()

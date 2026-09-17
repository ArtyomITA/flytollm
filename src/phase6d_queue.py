"""Phase 6d (SUITE_TEST_FASE_6B.md, category F): specialised brain regions as text ports and readout.
Eyes (photoreceptors) or ears (Johnston's organ / mechanosensory) as input; courtship-communication circuitry
(fru/dsx), best-connected hubs, or central complex as output. Same pipeline, 2000 updates, one GPU worker.
Waits for the phase-6c queue to finish."""
import ctypes, json, time
import pretrain_night_queue as q
from bench_runtime import ROOT

STOP = ROOT / 'STOP_PHASE6'
COMMON = ['--threshold', 10, '--head', 'separate', '--seed', 17, '--eval-every', 500, '--checkpoint-every', 2000, '--rewire-kind', 'none']
PREVIOUS = ROOT / 'results/phase6c_live.json'

SMOKES = [('phase6d_smoke_readout_fru', ['--readout', 'fru']), ('phase6d_smoke_ports_visual', ['--ports', 'visual'])]
RUNS = [
    ('phase6d_ports_visual_2000', ['--ports', 'visual']),
    ('phase6d_readout_fru_2000', ['--readout', 'fru']),
    ('phase6d_visual_fru_2000', ['--ports', 'visual', '--readout', 'fru']),
    ('phase6d_readout_hub_2000', ['--readout', 'hub']),
    ('phase6d_readout_cx_2000', ['--readout', 'cx']),
    ('phase6d_ports_auditory_2000', ['--ports', 'auditory']),
    ('phase6d_visual_hub_2000', ['--ports', 'visual', '--readout', 'hub']),
]


def wait_for_previous():
    while True:
        if PREVIOUS.exists():
            status = json.loads(PREVIOUS.read_text()).get('status')
            if status in ('completed', 'failed', 'stopped'):
                return status
        time.sleep(60)


def main():
    q.STATE = ROOT / 'results/phase6d_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase6d queue state exists; do not launch twice')
    state = dict(status='waiting_for_phase6c', completed=[], protocol='SUITE_TEST_FASE_6B.md#F', failed=[])
    q.write_state(state)
    state['previous_status'] = wait_for_previous()
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    skip = set()
    try:
        for name, extra in SMOKES:
            try:
                q.run_job(state, name, 'pretrain_control', COMMON + extra + ['--updates', 4], 1800)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state); skip.add(tuple(extra))
        for name, extra in RUNS:
            if any(tuple(s) == tuple(extra) for s in skip):
                state['failed'].append(dict(name=name, error='smoke failed, run skipped')); q.write_state(state); continue
            try:
                q.run_job(state, name, 'pretrain_control', COMMON + extra + ['--updates', 2000], 7200)
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

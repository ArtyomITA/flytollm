"""Phase 7 (SUITE_TEST_FASE_6B.md, section G; user plan of 16 September 2026): the standard configuration
(STANDARD_MOSCA_6B.md: whole graph with relative synaptic threshold, random ports, chunk readout, LIF, fused kernel)
as the new baseline, then the three "tengo" levers from RICERCA_ATTIVITA_CERVELLO_MOSCA.md added one at a time, then
all three together. One GPU worker at a time; PAUSE_BETWEEN_RUNS honoured by pretrain_night_queue.run_job.

  1. phase7_baseline_4000: standard, 4000 updates, eval every 500, checkpoints at 1000/2000/3000/4000
  2. inference suite (phase6c_inference.py) on the 1000 / 2000 / 4000 checkpoints
  3. phase7_tengo_T8_2000 (8+8 substeps), phase7_tengo_shortpath_2000 (anatomical short-path ports),
     phase7_tengo_relthr5_2000 (relative threshold at the threshold-5 edge count, 6.242.118 edges): 2000 updates each
  4. phase7_tengo_all_2000: the three levers together
A failed smoke skips its runs; a failed run is recorded and the queue continues."""
import ctypes, json, subprocess, time
import pretrain_night_queue as q
from bench_runtime import ROOT

STOP = ROOT / 'STOP_PHASE7'
STANDARD = ['--threshold', 10, '--head', 'separate', '--seed', 17, '--eval-every', 500, '--fast-mode', 'fused',
            '--rewire-kind', 'relthr', '--rewire-seed', 41, '--ports', 'random_matched', '--readout', 'chunks', '--core-variant', 'lif']
T8 = ['--pre-steps', 8, '--post-steps', 8]
SHORTPATH = ['--ports', 'shortpath']
RELTHR5 = ['--rewire-kind', 'relthr5']
INFERENCE_LOG = ROOT / 'results/phase7_inference.console.log'

SMOKES = [('phase7_smoke_standard', []), ('phase7_smoke_T8', T8), ('phase7_smoke_shortpath', SHORTPATH),
          ('phase7_smoke_relthr5', RELTHR5), ('phase7_smoke_all', T8 + SHORTPATH + RELTHR5)]
TENGO = [('phase7_tengo_T8_2000', T8, 'phase7_smoke_T8', 7200),
         ('phase7_tengo_shortpath_2000', SHORTPATH, 'phase7_smoke_shortpath', 7200),
         ('phase7_tengo_relthr5_2000', RELTHR5, 'phase7_smoke_relthr5', 7200),
         ('phase7_tengo_all_2000', T8 + SHORTPATH + RELTHR5, 'phase7_smoke_all', 10800)]


def pause():
    marker = ROOT / 'PAUSE_BETWEEN_RUNS'
    if marker.exists():
        seconds = float(marker.read_text().strip() or '300')
        print(json.dumps(dict(event='pause', name='inference', seconds=seconds)), flush=True); time.sleep(seconds)


def inference(state, checkpoint, tag):
    """phase6c_inference.py on one checkpoint; output results/phase7_inference_<tag>.json, console appended to one log."""
    if STOP.exists():
        raise InterruptedError('User stop file present')
    output = ROOT / 'results' / f'phase7_inference_{tag}.json'
    if output.exists():
        state['completed'].append(dict(name=f'inference_{tag}', ok=True, skipped='output exists', finished=time.time())); q.write_state(state); return
    pause()
    print(json.dumps(dict(event='start', name=f'inference_{tag}')), flush=True)
    state.update(status='running', current=f'inference_{tag}', started=time.time()); q.write_state(state)
    with INFERENCE_LOG.open('a', encoding='utf8') as log:
        log.write(f'=== inference_{tag} {time.strftime("%H:%M:%S")}\n'); log.flush()
        r = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-u', str(ROOT / 'phase6c_inference.py'),
                            '--checkpoint-path', str(checkpoint), '--output', str(output)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=3600, creationflags=subprocess.CREATE_NO_WINDOW)
        log.write(f'=== exit {r.returncode} {time.strftime("%H:%M:%S")}\n')
    ok = r.returncode == 0 and output.exists()
    state['completed'].append(dict(name=f'inference_{tag}', ok=ok, finished=time.time())); q.write_state(state)
    print(json.dumps(dict(event='end', name=f'inference_{tag}', ok=ok)), flush=True)
    if not ok:
        state['failed'].append(dict(name=f'inference_{tag}', error=f'exit {r.returncode}')); q.write_state(state)


def main():
    q.STATE = ROOT / 'results/phase7_live.json'; q.STOP = STOP
    if q.STATE.exists():
        raise RuntimeError('Phase7 queue state exists; do not launch twice')
    state = dict(status='running', completed=[], failed=[], protocol='SUITE_TEST_FASE_6B.md#G')
    q.write_state(state)
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if not keep_awake:
        raise OSError('Cannot inhibit sleep')
    failed_smokes = set()
    try:
        for name, extra in SMOKES:
            try:
                q.run_job(state, name, 'pretrain_control', STANDARD + extra + ['--updates', 4, '--checkpoint-every', 4], 1800)
            except RuntimeError as exc:
                state['failed'].append(dict(name=name, error=str(exc))); q.write_state(state); failed_smokes.add(name)
        baseline_ok = 'phase7_smoke_standard' not in failed_smokes
        if baseline_ok:
            try:
                q.run_job(state, 'phase7_baseline_4000', 'pretrain_control', STANDARD + ['--updates', 4000, '--checkpoint-every', 1000], 10800)
            except RuntimeError as exc:
                state['failed'].append(dict(name='phase7_baseline_4000', error=str(exc))); q.write_state(state); baseline_ok = False
        if baseline_ok:
            for update in (1000, 2000, 4000):
                checkpoint = ROOT / 'results' / f'phase7_baseline_4000.step{update:08d}.pt'
                if checkpoint.exists():
                    inference(state, checkpoint, f'baseline_{update}')
                else:
                    state['failed'].append(dict(name=f'inference_baseline_{update}', error='checkpoint missing')); q.write_state(state)
        for name, extra, smoke, timeout in TENGO:
            if smoke in failed_smokes:
                state['failed'].append(dict(name=name, error='smoke failed, run skipped')); q.write_state(state); continue
            try:
                q.run_job(state, name, 'pretrain_control', STANDARD + extra + ['--updates', 2000, '--checkpoint-every', 2000], timeout)
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

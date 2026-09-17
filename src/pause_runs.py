"""Pause / resume the running control queue without losing the run (user request, 16 September 2026).

  python pause_runs.py status   list the queue, supervisor and worker processes and whether they are suspended
  python pause_runs.py pause    suspend queue -> supervisor -> worker (NtSuspendProcess); GPU compute stops, VRAM stays allocated
  python pause_runs.py resume   resume worker first, wait for a fresh progress line in its console log (so the
                                supervisor's 90 s phase guard does not fire), then supervisor and queue

Limit: the supervisor's total timeout (--timeout passed by the queue, wall clock) keeps running while paused; a pause
longer than the remaining budget makes the supervisor kill the worker at resume ('total_timeout'). Queues launched
after 16 September pass generous timeouts for this reason. Runs cannot be resumed from a checkpoint (control runs
rebuild the model from source), so killing a run means restarting it from scratch."""
import ctypes, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESS_SUSPEND_RESUME = 0x0800
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
k32 = ctypes.windll.kernel32; ntdll = ctypes.windll.ntdll


def processes():
    """python.exe processes of the queue / supervisor / worker, from the command line (WMI via PowerShell)."""
    cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId, ParentProcessId, CommandLine | ConvertTo-Json -Compress")
    out = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True, timeout=60).stdout.strip()
    rows = json.loads(out) if out else []
    if isinstance(rows, dict):
        rows = [rows]
    found = []
    for r in rows:
        line = r.get('CommandLine') or ''
        role = None
        if '_queue.py' in line or 'phase6f_promote.py' in line:
            role = 'queue'
        elif 'pretrain_control.py' in line or 'pretrain_resumable.py' in line:
            role = 'worker' if '--worker' in line else 'supervisor'
        elif 'phase6c_inference.py' in line:
            role = 'inference'
        if role:
            found.append(dict(pid=int(r['ProcessId']), parent=int(r['ParentProcessId']), role=role, cmd=line[:120]))
    return found


def act(pid, resume):
    h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, int(pid))
    if not h:
        return f'{pid}: OpenProcess failed ({k32.GetLastError()})'
    status = (ntdll.NtResumeProcess if resume else ntdll.NtSuspendProcess)(h); k32.CloseHandle(h)
    return f'{pid}: {"resumed" if resume else "suspended"} (status {status})'


def worker_log(procs):
    """The worker writes results/<name>.console.log (read by the supervisor): take the newest console log."""
    logs = sorted((ROOT / 'results').glob('*.console.log'), key=lambda x: x.stat().st_mtime)
    return logs[-1] if logs else None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'status'
    procs = processes()
    order = {'queue': 0, 'supervisor': 1, 'worker': 2, 'inference': 2}
    procs.sort(key=lambda p: order[p['role']])
    if mode == 'status' or not procs:
        for p in procs:
            print(p['role'], p['pid'], p['cmd'][:100])
        if not procs:
            print('no queue/supervisor/worker running')
        return
    if mode == 'pause':
        for p in procs:  # queue first so it cannot react, then supervisor, then worker
            print(act(p['pid'], resume=False))
        print('paused', time.strftime('%H:%M:%S'), '- VRAM stays allocated; resume with: python pause_runs.py resume')
        return
    if mode == 'resume':
        workers = [p for p in procs if p['role'] in ('worker', 'inference')]
        others = [p for p in procs if p['role'] in ('supervisor', 'queue')]
        log = worker_log(procs)
        size = log.stat().st_size if log else 0
        for p in workers:
            print(act(p['pid'], resume=True))
        deadline = time.time() + 120
        while log and time.time() < deadline and log.stat().st_size == size:
            time.sleep(1)
        print('worker progressed' if (not log or log.stat().st_size > size) else 'WARNING: no new worker output in 120 s')
        for p in sorted(others, key=lambda p: -order[p['role']]):  # supervisor before queue
            print(act(p['pid'], resume=True))
        print('resumed', time.strftime('%H:%M:%S'))
        return
    print('usage: pause_runs.py status|pause|resume')


if __name__ == '__main__':
    main()

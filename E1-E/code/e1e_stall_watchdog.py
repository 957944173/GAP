"""E1-0 stall watchdog: on-demand faulthandler stack capture (no semantics).

Attempts 14/16/17/18 each stalled with a completely silent log and no usable
stack (ptrace is unavailable on this box: /proc/sys/kernel/yama/ptrace_scope=1
and no passwordless sudo, so py-spy/gdb/strace are all unusable).  The one
mechanism that *does* work is the on-demand `faulthandler.register(SIGUSR1)`
handler installed by code/e1e_hang_diagnostics.py in every process that runs
with PYTHONPATH=<code>/pythonpath_e10 and E1E_RUN_DIR set: delivering SIGUSR1
appends an all-thread Python stack to RUN_DIR/logs/stackdumps/stackdump_pid<PID>.txt
without disturbing the process.

This watchdog automates exactly that delivery, but only after the stall is
independently confirmed by the same criteria the previous attempts used
manually (log file not growing for N seconds while the process is alive):

  phase "pre":  armed once the log contains the resume line and disarmed as
                soon as the first validation batch print appears.  The two
                observed hangs both sat in this window.
  phase "post": after the first validation batch print, a much longer static
                window (default 45 min) is required, because real training
                steps legitimately take ~14-20 min each on this box.

CRITICAL SAFETY: SIGUSR1 is only ever sent to pids that have a stackdump file
*created after this watchdog started* -- such a file proves the process loaded
e1e_hang_diagnostics, i.e. it has a SIGUSR1 handler and will not be killed by
the signal.  Stale files from previous attempts are snapshotted and ignored, so
a recycled pid can never be signalled.

Usage:
    python3 e1e_stall_watchdog.py --log <launcher log> --run-dir <RUN_DIR> \
        [--pre-stall-seconds 480] [--post-stall-seconds 2700] [--interval 15]
"""
import argparse
import glob
import os
import re
import signal
import subprocess
import time

RESUME_RE = re.compile(r"Resuming from ")
FIRST_BATCH_RE = re.compile(r"test_gen_batch meta info")
# NOTE: deliberately NOT matching tqdm's "Training Progress" bar -- that line
# appears right after the first validation and would otherwise retire the
# watchdog before the training loop it is supposed to monitor.
DONE_RE = re.compile(r"Final validation metrics:")


def log_line(msg, path):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")


def stackdump_files(d):
    return set(glob.glob(os.path.join(d, "stackdump_pid*.txt")))


def pid_of(path):
    m = re.search(r"stackdump_pid(\d+)\.txt$", path)
    return int(m.group(1)) if m else None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def snapshot_gpu():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "<nvidia-smi unavailable>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--pre-stall-seconds", type=int, default=480)
    ap.add_argument("--post-stall-seconds", type=int, default=2700)
    ap.add_argument("--max-captures", type=int, default=3)
    args = ap.parse_args()

    dump_dir = os.path.join(args.run_dir, "logs", "stackdumps")
    os.makedirs(dump_dir, exist_ok=True)
    wlog = os.path.join(args.run_dir, "logs", "watchdog.log")

    baseline = stackdump_files(dump_dir)
    start_wall = time.time()
    log_line(f"watchdog started; log={args.log}; baseline stackdumps={len(baseline)}", wlog)
    log_line(f"pre_stall={args.pre_stall_seconds}s post_stall={args.post_stall_seconds}s interval={args.interval}s", wlog)

    armed = False
    disarmed_first_batch = False
    captures = 0
    seen_pids_signaled = set()

    while True:
        time.sleep(args.interval)
        try:
            content = open(args.log, "r", errors="replace").read()
        except FileNotFoundError:
            continue

        if DONE_RE.search(content):
            log_line("watchdog: training finished (final metrics/progress marker seen); exiting", wlog)
            return 0

        if not armed and RESUME_RE.search(content):
            armed = True
            log_line("watchdog: armed (resume line seen)", wlog)

        if not disarmed_first_batch and FIRST_BATCH_RE.search(content):
            disarmed_first_batch = True
            log_line("watchdog: first validation batch print seen -> post phase (long window)", wlog)

        if not armed:
            continue

        try:
            mtime = os.path.getmtime(args.log)
        except OSError:
            continue
        quiet = time.time() - mtime
        threshold = args.post_stall_seconds if disarmed_first_batch else args.pre_stall_seconds
        if quiet < threshold:
            continue

        if captures >= args.max_captures:
            log_line(f"watchdog: quiet {quiet:.0f}s but max_captures={args.max_captures} reached; exiting", wlog)
            return 1

        # ---- confirmed stall: capture stacks ----
        captures += 1
        new_files = sorted(stackdump_files(dump_dir) - baseline)
        fresh = []
        for f in new_files:
            pid = pid_of(f)
            if pid is None:
                continue
            try:
                if os.path.getmtime(f) < start_wall:
                    continue
            except OSError:
                continue
            if alive(pid):
                fresh.append((pid, f))

        log_line(
            f"STALL #{captures}: log quiet {quiet:.0f}s (threshold {threshold}s), "
            f"phase={'post' if disarmed_first_batch else 'pre'}; "
            f"signalling {len(fresh)} pids with SIGUSR1",
            wlog,
        )
        for pid, f in fresh:
            try:
                os.kill(pid, signal.SIGUSR1)
                seen_pids_signaled.add(pid)
                log_line(f"  SIGUSR1 -> pid {pid} ({os.path.basename(f)})", wlog)
            except OSError as e:
                log_line(f"  SIGUSR1 -> pid {pid} failed: {e}", wlog)

        log_line(f"  GPU snapshot: {snapshot_gpu()}", wlog)
        try:
            ps = subprocess.run(
                ["ps", "-eo", "pid,ppid,stat,pcpu,rss,etime,args", "--sort=-pcpu"],
                capture_output=True, text=True, timeout=30,
            ).stdout
            with open(os.path.join(dump_dir, f"ps_at_stall_{captures}.txt"), "w") as fh:
                fh.write(ps)
            log_line(f"  ps snapshot -> {dump_dir}/ps_at_stall_{captures}.txt", wlog)
        except Exception as e:  # noqa: BLE001
            log_line(f"  ps snapshot failed: {e}", wlog)

        # let the dumps flush, then keep watching (the operator/agent decides)
        time.sleep(45)


if __name__ == "__main__":
    raise SystemExit(main())

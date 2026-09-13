"""E1-0 passive diagnostic instrumentation: periodic all-thread stack dumps.

Purpose: attempt 14 hung silently for 4+ hours (0% GPU util, ~100% CPU spin on
SGLang scheduler subprocesses, training log completely silent after all 4
WorkerDict ranks finished `_init_sampling_params()`) with no evidence of
*where* execution was actually stuck -- static code review narrowed the
candidate window to the pre-training validation rollout's generation call or
the FSDP->SGLang weight-sync (`FSDPSGLangShardingManager.update_weights()`,
specifically the per-tensor `dist.gather_object()` collective or SGLang's
own `resume_memory_occupation()`/`update_weights_from_tensor()` RPC), but
this was not confirmed. Rather than guess further and risk a wrong "fix",
this module adds PURE OBSERVATION so that if the hang recurs, the next
attempt produces an actual stack trace instead of silence.

This is diagnostic-only (module-level side effect: registers a background
watcher via the stdlib `faulthandler` module). It does not alter any
training control flow, reward, sampling, GRPO, optimizer, checkpoint, or
tool-execution semantics -- it only writes periodic all-thread Python stack
snapshots to RUN_DIR/logs/stackdumps/. Loaded from
pythonpath_e10/sitecustomize.py alongside the other E1-0 patches, so it runs
in every process (TaskRunner driver and every WorkerDict/Ray worker) that
inherits PYTHONPATH.

Two mechanisms, both belt-and-suspenders:
  1. faulthandler.dump_traceback_later(interval, repeat=True): a C-level
     watchdog thread that dumps all-thread Python stacks every N seconds.
     Implemented in C so it can fire even if the main thread is blocked in a
     long-running Python-level call, as long as that call periodically
     releases the GIL (true for NCCL/gloo collective ops, socket I/O, and
     most torch CUDA calls) -- it will NOT fire if the main thread is stuck
     in a tight C-extension loop that never releases the GIL, but this is
     still strictly more informative than nothing.
  2. faulthandler.register(SIGUSR1, ...): allows an on-demand dump by
     sending SIGUSR1 to any pid whose log shows it has entered this module,
     without killing the process -- useful for a manual "is rank pid X truly
     stuck, and where" check mid-run before deciding to SIGTERM.

Failure mode: if E1D_RUN_DIR is unset, or faulthandler is unavailable for any
reason, this module no-ops (wrapped in try/except) rather than raising --
diagnostics must never be able to crash or block real training.

*** POST-ATTEMPT-15 UPDATE (disabled the periodic watchdog) ***
Attempt 15 (2026-09-10, this run) proved that the periodic mechanism
(`faulthandler.dump_traceback_later`) is NOT safe in this environment: at
09:23:00, ~90s after this module loaded in WorkerDict pid=1845765 (an
SGLang-TP0-engine-holding rank), the C-level watchdog thread fired mid CUDA
graph-capture warmup and produced:
    Fatal Python error: PyThreadState_Get: the function must be called with
    the GIL held, but the GIL is released (the current Python thread state
    is NULL)
    Segmentation fault
This killed that Ray actor outright (see launcher_attempt15.log lines
619-731), cascaded into a gloo "Connection closed by peer" error in a
sibling rank, and degraded the whole run (Ray cluster left holding only
3/4 GPUs). This is a real, first-hand-observed crash caused BY this
diagnostic module, not by anything in the original training code -- i.e.
exactly the kind of self-inflicted corruption risk the E1-0 mandate forbids
introducing. Root cause (consistent with the crash signature): SGLang's
per-rank engine runs native CUDA/NCCL background threads (watchdog thread,
tp_worker_overlap forward thread) whose Python thread state can be
mid-transition (GIL released for a blocking native call) at the instant the
watchdog thread fires; `dump_traceback_later`'s C-level timer thread does
not wait for a safe interpreter checkpoint before walking thread states,
unlike a delivered POSIX signal (which CPython only actually runs at a
bytecode-boundary safepoint). The periodic watchdog is therefore DISABLED
below. Only the SIGUSR1 on-demand handler remains active: it is driven by
an actual delivered signal, checked by the interpreter at a safe point,
which is the standard documented-safe faulthandler usage pattern.
"""
import atexit
import faulthandler
import os
import signal

_E1D_RUN_DIR = os.environ.get("E1D_RUN_DIR")

_dump_file = None

try:
    if _E1D_RUN_DIR:
        _diag_dir = os.path.join(_E1D_RUN_DIR, "logs", "stackdumps")
        os.makedirs(_diag_dir, exist_ok=True)
        _pid = os.getpid()
        _dump_path = os.path.join(_diag_dir, f"stackdump_pid{_pid}.txt")
        _dump_file = open(_dump_path, "a", buffering=1)
        _dump_file.write(f"=== E1-0 hang-diagnostics stack watcher started, pid={_pid} ===\n")
        _dump_file.flush()

        # DISABLED after attempt 15: faulthandler.dump_traceback_later's C-level
        # periodic watchdog thread caused a real SIGSEGV crash (see docstring
        # above). Do NOT re-enable this call.
        # faulthandler.dump_traceback_later(90, repeat=True, file=_dump_file, exit=False)

        # On-demand dump only: `kill -USR1 <pid>` appends a fresh all-thread trace
        # without disturbing the process otherwise. Safe: driven by an actual
        # delivered signal, handled at a CPython bytecode-boundary safepoint.
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file=_dump_file, all_threads=True, chain=False)

        def _e1d_close_dump_file():
            try:
                if _dump_file and not _dump_file.closed:
                    _dump_file.write("=== process exiting, stack watcher stopping ===\n")
                    _dump_file.close()
            except Exception:
                pass

        atexit.register(_e1d_close_dump_file)
except Exception as _e:  # noqa: BLE001 -- diagnostics must never break training
    try:
        import sys
        sys.stderr.write(f"E1-0 hang_diagnostics: failed to install (non-fatal, continuing): {type(_e).__name__}: {_e}\n")
    except Exception:
        pass

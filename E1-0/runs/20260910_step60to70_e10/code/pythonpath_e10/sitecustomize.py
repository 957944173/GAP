"""E1-0 registration shim.

Placed on PYTHONPATH (verl/verl/workers/reward_manager/__init__.py itself is
NOT modified) so that Python auto-imports this module at interpreter startup
in every process that has this directory on PYTHONPATH -- including the
driver and any Ray worker processes that inherit the parent's PYTHONPATH env
var (OS-level PYTHONPATH propagates into Ray remote workers without needing
runtime_env edits).

Importing e10_reward_manager here runs its @register("e10_batch_audit")
decorator exactly once per process, before load_reward_manager() does its
string-name lookup in reward.py. This is the entire registration mechanism;
no original verl file is modified.
"""
import os
import sys

_E10_SHIM_DIR = os.path.dirname(os.path.abspath(__file__))
_E10_CODE_DIR = os.path.abspath(os.path.join(_E10_SHIM_DIR, ".."))
if _E10_CODE_DIR not in sys.path:
    sys.path.insert(0, _E10_CODE_DIR)

# e10_hang_diagnostics: periodic watchdog PERMANENTLY DISABLED, on-demand
# SIGUSR1 handler RE-ENABLED for attempt 17+.
#
# History: added and loaded for attempt 15 (2026-09-10) to help root-cause a
# suspected recurrence of attempt 14's silent hang. Its periodic in-process
# watchdog (faulthandler.dump_traceback_later, firing every 90s in every
# process on PYTHONPATH) caused a genuine SIGSEGV in WorkerDict pid=1845765
# at 09:23:00 during CUDA graph-capture warmup ("Fatal Python error:
# PyThreadState_Get: the function must be called with the GIL held, but the
# GIL is released" -> Segmentation fault), which killed that Ray actor and
# degraded the whole run to 3/4 GPUs. See e10_hang_diagnostics.py's updated
# docstring and RUN_DIR/logs/launcher_attempt15.log lines 619-731 for full
# evidence. That periodic mechanism is permanently commented out in
# e10_hang_diagnostics.py and must never be re-enabled.
#
# For attempt 16 (2026-09-10, same day) this module was fully unloaded (not
# imported at all) as an extra precaution. Attempt 16 then reproduced
# attempt 14's original hang cleanly (log static 25+ min at the exact same
# "Resuming from global_step_60" / CUDA-graph-capture-complete milestone,
# GPU 0% on all 4 devices, SGLang scheduler main threads in `stat=R` /
# on-CPU per /proc/<pid>/task/*/wchan inspection -- see
# RUN_DIR/logs/attempt16_hang_diagnosis.md). Attempting to use `py-spy dump
# --pid <PID>` (the alternative floated in the attempt-15 doc) and a direct
# `strace -p <pid>` both FAILED in this environment with ptrace permission
# errors (/proc/sys/kernel/yama/ptrace_scope=1, no passwordless sudo) --
# neither tool is usable here, so no external non-invasive stack-trace tool
# exists in this environment.
#
# Given that, the on-demand `faulthandler.register(SIGUSR1, ...)` path in
# e10_hang_diagnostics.py is re-enabled starting attempt 17. Unlike the
# periodic watchdog, this only fires when a signal is actually delivered,
# and CPython only services pending signals at a bytecode-boundary
# safepoint (via the eval-loop's pending-signal check) -- it does not walk
# thread stacks from an independent C-level timer thread the way
# dump_traceback_later's watchdog does, so it cannot observe a thread
# mid-GIL-transition the way that call could. It will be sent manually
# (`kill -USR1 <pid>`) to at most one or two specific suspect pids, once,
# only after the hang criteria are independently confirmed (log static
# 15+min, GPU 0%, CPU spin) -- not installed as any kind of periodic timer.
try:
    import e10_hang_diagnostics  # noqa: F401 -- on-demand SIGUSR1 dump only, periodic watchdog stays disabled, see module docstring
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-0 sitecustomize.py: failed to import e10_hang_diagnostics (non-fatal): {e}\n")

try:
    import e10_reward_manager  # noqa: F401  -- runs @register("e10_batch_audit")
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-0 sitecustomize.py: failed to import e10_reward_manager: {e}\n")
    raise

try:
    import e10_checkpoint_patch  # noqa: F401 -- monkeypatches FSDPCheckpointManager.load_checkpoint (map_location="cpu"), see that module's docstring for the CUDA OOM this fixes
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-0 sitecustomize.py: failed to import e10_checkpoint_patch: {e}\n")
    raise

try:
    import e10_sglang_gpu_patch  # noqa: F401 -- monkeypatches Worker._setup_env_cuda_visible_devices to narrow CUDA_VISIBLE_DEVICES under RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1, see that module's docstring
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-0 sitecustomize.py: failed to import e10_sglang_gpu_patch: {e}\n")
    raise

# Attempt-19 passive diagnostics (pure prints, no semantics): worker-side
# load_checkpoint / generate_sequences boundaries and the FSDP->SGLang
# sharding-manager handoff, added because attempt 18 stalled with no output
# between "load_checkpoint: fully returned" and _validate()'s first print.
# Non-fatal: if the wrap fails, training must still proceed unchanged.
try:
    import e10_diag_patch  # noqa: F401
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"E1-0 sitecustomize.py: failed to import e10_diag_patch (non-fatal): {e}\n")

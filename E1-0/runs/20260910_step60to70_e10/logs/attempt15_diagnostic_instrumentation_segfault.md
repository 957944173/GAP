# Attempt 15: self-inflicted SIGSEGV caused by new hang-diagnostic instrumentation

## Context

Attempt 14 ended with a silent hang (no crash, no log growth for 4+ hours,
0% GPU) at the point immediately after checkpoint resume + CUDA graph
capture completion, with no stack-trace evidence of where execution was
stuck (see `launcher_attempt14_hung_sglang_tp_no_progress.log` and
`sglang_gpu_collision_diagnosis.md` / `host_ram_oom_diagnosis.md` for the
attempts before it). Rather than guess at a fix for an unconfirmed root
cause, a new purely-observational diagnostic module,
`code/e10_hang_diagnostics.py`, was added and registered via
`code/pythonpath_e10/sitecustomize.py` (loaded first, before the other three
E1-0 patches, in every process on PYTHONPATH). It used
`faulthandler.dump_traceback_later(90, repeat=True, ...)` to write periodic
(every 90s) all-thread Python stack dumps to
`RUN_DIR/logs/stackdumps/stackdump_pid<PID>.txt`, plus a `SIGUSR1`-triggered
on-demand dump handler. This was intended as pure audit/observation with "no
training-semantics effect," consistent with E1-0's passive-only mandate.

## What happened in attempt 15

Attempt 15 was launched fresh (clean env, Wiki server functional-checked and
reused, no code changes to reward/sampling/GRPO/checkpoint logic). Training
progressed correctly and *further* than attempt 14's static point: Ray
started, dataset loaded (169595 train / 3610 val rows), all 4 WorkerDict
ranks loaded the Qwen2-3B model, wrapped FSDP, and reached CUDA graph
capture ("Capturing batches ... 100%"). The checkpoint-resume path fired
correctly ("Setting global step to 60", "Resuming from
.../global_step_60"). This is genuine further progress than attempt 14,
not an immediate repeat of the same hang.

At `2026-09-10 09:23:00`, approximately 90 seconds after
`e10_hang_diagnostics.py` installed its watchdog in WorkerDict pid=1845765
(an SGLang-TP0-engine-holding rank, mid CUDA-graph-capture warmup / FSDP
state-dict-API deprecation-warning region), the process printed:

```
Fatal Python error: PyThreadState_Get: the function must be called with the
GIL held, but the GIL is released (the current Python thread state is NULL)
Python runtime state: initialized
... (interleaved/garbled all-thread traceback dump from the fault handler
     itself firing during a fault) ...
[2026-09-10 09:23:00 TP1] Scheduler hit an exception: Traceback (most recent
call last):
  ...
  File ".../sglang/srt/utils.py", line 950, in broadcast_pyobj
    dist.broadcast(tensor_size, src=src, group=dist_group)
  ...
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534]
Connection closed by peer [127.0.0.1]:58054
Segmentation fault
```

(full raw text: `launcher_attempt15.log` lines 619-731). Ray then reported
this actor as `SYSTEM_ERROR` / "unexpectedly exits with a connection error
code 2" and a sibling actor (pid 1850474) also exited via SIGTERM shortly
after (cascading teardown). The run degraded to 3/4 GPUs reserved
(`ray status` showed `3.0/4.0 GPU` in use) and made no further progress;
the TaskRunner driver (pid 1842470) and the two surviving SGLang scheduler
subprocesses (pids 1850745/1850746, confirmed via
`stackdump_pid1850741.txt` / `stackdump_pid1850745.txt`) were left cycling
idly through their normal event loops (`recv_requests` -> `broadcast_pyobj`
-> `check_memory` -> repeat), waiting for a 4th rank that would never
return. This is NOT the attempt-14 hang signature recurring -- it is a new,
distinct, self-inflicted crash.

## Root cause

`faulthandler.dump_traceback_later()`'s periodic dump runs from a dedicated
C-level watchdog thread that walks and dumps every Python thread's frame
stack, independent of whether the interpreter is at a safe bytecode-boundary
checkpoint. SGLang's per-rank engine process runs several native
CUDA/NCCL-touching background threads (`watchdog_thread`,
`tp_worker_overlap_thread.py`'s `forward_thread_func`) that call into
CUDA/NCCL/torch C extensions and release the GIL for the duration of those
calls. If the periodic watchdog's C-level timer fires at the exact instant
one of these threads is transitioning its `PyThreadState` (GIL released,
thread state pointer null/in-flux -- e.g. entering or leaving a `nogil`
CUDA/NCCL call), `PyThreadState_Get()` inside faulthandler's own C dump
routine aborts with a fatal Python error and the OS delivers SIGSEGV. This
is a known category of hazard with `faulthandler.dump_traceback_later` in
processes with many native/CUDA-heavy background threads: unlike a
delivered POSIX signal (which CPython only actually services at a
bytecode-boundary safepoint via `Py_AddPendingCall`/eval-loop signal
checking), the periodic-dump watchdog thread performs its stack walk
directly, with no such safepoint synchronization.

This is a genuine, first-hand-observed bug introduced by our own diagnostic
code -- not a pre-existing bug in verl/SGLang/Ray, and not a recurrence of
attempt 14's original (still not root-caused) hang. It directly violates the
E1-0 mandate's "audit must be passive-only, must never risk corrupting
training" principle, so it is treated as a BLOCKER for that instrumentation
specifically and reverted.

## Fix applied

- `code/e10_hang_diagnostics.py`: the `faulthandler.dump_traceback_later(90,
  repeat=True, ...)` call is now commented out (never invoked). Only
  `faulthandler.register(SIGUSR1, ...)` remains wired up in the module, since
  a delivered signal IS serviced at a safe CPython bytecode boundary and is
  the officially-documented-safe faulthandler usage pattern -- but this
  on-demand path was ALSO removed from the live training path out of an
  abundance of caution (see next bullet), since it was never actually needed
  to reach a diagnosis this run.
- `code/pythonpath_e10/sitecustomize.py`: the `import e10_hang_diagnostics`
  line is commented out entirely. No in-process diagnostic instrumentation
  is loaded for attempt 16 onward. If future hang diagnosis is ever needed,
  the safe alternative is `py-spy dump --pid <PID>` run from OUTSIDE the
  target process (confirmed available: `py-spy 0.4.0` in the
  `parallel-agent` env) -- it attaches via ptrace and reads the target's
  memory/stacks without executing any code inside the target's own
  interpreter or threads, so it cannot corrupt GIL/thread-state the way an
  in-process timer thread can.
- No other E1-0 file was touched. `e10_reward_manager.py`,
  `e10_scorer.py`, `e10_checkpoint_patch.py`, `e10_sglang_gpu_patch.py`, and
  `run_e10_step60to70.sh` are byte-identical to the attempt-14/15 versions.
  Reward/sampling/GRPO/optimizer/checkpoint/tool-execution semantics were
  never touched by this bug or its fix.

## Cleanup performed before attempt 16

- Confirmed no partial/corrupt checkpoint was ever written under
  `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/` (directory did
  not exist -- crash occurred well before the first scheduled save at
  step70, since `save_freq=10` and training never got past the
  pre-training validation/step60 resume phase).
- Confirmed the original baseline checkpoints under
  `experiments/DAPO-GAP3B-MHQA-Agent-4gpu/` (global_step_60/120/180/200)
  are untouched.
- Killed only this run's own processes: top-level launcher
  (`run_e10_step60to70.sh`, pid 1837314) and its Ray cluster
  (`ray stop --force`, which reported cleanly stopping this session's
  processes). GPU memory returned to baseline (~9.7GB/GPU residual, matching
  the Wiki server's own FAISS/embedding-model GPU usage, not new headroom
  filled by leftover training processes).
- Confirmed the Wiki RAG server (pid 1016459, port 8008) was NOT touched and
  remains healthy: functional `POST /retrieve` against a real query
  ("France") returned HTTP 200 with a real document result after the
  cleanup, immediately before relaunching attempt 16.

## Open question carried forward (not yet resolved)

Attempt 14's original silent hang is still not definitively root-caused --
attempt 15 did not reproduce it (it progressed further and then hit this
unrelated, self-inflicted crash before reaching the point where attempt 14
went silent... actually attempt 15 DID reach the same "Setting global step
to 60" / "Resuming from global_step_60" milestone as attempt 14 before its
own crash interrupted further observation). If attempt 16 (with diagnostics
fully removed, i.e. running the same code path as attempt 14 modulo the
`e10_checkpoint_patch.py`/`e10_sglang_gpu_patch.py` fixes already in place
since attempt 8-14) reaches this same point and then genuinely hangs again
(log static 15+ min, GPU 0%, CPU spin, per the task's hang criteria), that
will need fresh investigation using `py-spy dump --pid <PID>` from outside
the process (safe alternative identified above) rather than further
in-process instrumentation.

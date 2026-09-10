# Attempt 16 diagnosis: hung, partial DP-group CUDA-graph capture

## Summary

Attempt 16 (driver pid 1877604, launched 09:41, killed 11:22 after 79+ min of
zero progress) hung with a *new* signature, distinct from every prior
attempt's root cause. The GPU-assignment fix (`e10_sglang_gpu_patch.py`) is
confirmed working correctly this time (see below) -- this is not a repeat of
the attempt 5/8/9 GPU-collision bug, nor the attempt 1-9 NCCL "Duplicate GPU"
bug, nor the attempt 13 CUDA-IPC device-ordinal bug, nor the attempt 15
faulthandler segfault.

## Evidence

- `nvidia-smi --query-compute-apps`: each of GPU 0/1/2/3 hosts exactly one
  Wiki-server slice + one WorkerDict + one sglang scheduler process, ~34-35GB
  scheduler memory per GPU. No collision. Confirms `e10_sglang_gpu_patch.py`'s
  per-rank GPU assignment fix is working correctly.
- Checkpoint resume succeeded: log shows `Setting global step to 60`,
  `Resuming from .../global_step_60`.
- Per-WorkerDict `.out`/`.err` progress diverged sharply between the two DP
  groups (GEN_TP=2, so 4 WorkerDicts = 2 DP groups x TP2):
  - DP group A (WorkerDict pids 1886013, 1887427): completed CUDA graph
    capture ("Capturing batches ... 100%|##########| 23/23") and moved on to
    the FSDP FutureWarning about `state_dict_type` deprecation -- i.e. reached
    the weight-sync / state-dict handling step.
  - DP group B (WorkerDict pids 1887426, 1887428): `.out`/`.err` files stop
    immediately after `use_xml_tool_parser open, add more stop words` (25
    lines each, vs 61/26 for group A) -- **no "Capturing batches" line at
    all**. This is not a log-deduplication artifact (checked: Ray's
    `[repeated Nx across cluster]` dedup only collapses identical lines from
    *different* pids, and grepping each `.out` file individually by exact
    pid-specific log path, not the merged launcher log, confirms these two
    processes genuinely never started CUDA graph capture).
- CPU tick-delta over a fixed 10s window (steady-state, ~90 min into the
  hang): sglang scheduler pids 1891035/1891036 (DP group A, TP0/TP1) sleeping
  most of the time but occasionally runnable; pids 1891037/1891039 (also
  observed at various points in R state) show ~97-117% single-core CPU
  consumption sustained across repeated checks -- busy-poll signature, not
  idle.
- Per-thread inspection of all 4 scheduler processes (`/proc/<pid>/task/*/`):
  main scheduler thread alternates S/R; every other thread (ZMQ IO, NCCL
  watchdog/heartbeat, CUDA event handler) parked in `futex_wait_queue_me` /
  `ep_poll` / `poll_schedule_timeout` -- consistent with a process waiting on
  a distributed collective or IPC handoff that never completes, not a crash.
- `py-spy dump --pid <pid>` and `strace -p <pid>` both fail with
  `ptrace attach ... was attempted by ...` in dmesg but no readable output --
  confirmed `/proc/sys/kernel/yama/ptrace_scope=1` and no passwordless sudo on
  this box. No native stack trace could be obtained for any process in this
  attempt (or attempt 14). This is an environment limitation, not something
  fixable from E1-0's scope.
- GPU utilization: 0% on all 4 GPUs throughout the stall, memory pinned at
  their post-capture footprint (~54GB used per GPU on the full node, majority
  from other users' unrelated jobs sharing these GPUs -- see `nvidia-smi`
  full node output; E1-0's own portion is ~44GB/GPU: 10GB Wiki + 34GB
  scheduler).
- Ray core-worker logs for pids 1886013/1887427: only routine internal Ray
  telemetry (task/IO stats), no errors -- Ray's own RPC layer still considers
  these actors alive.
- `dmesg`: no Xid errors, no NCCL/CUDA fatal errors, no OOM killer activity
  during this attempt's window.

## Interpretation (INFERRED, not proven -- no stack trace available)

DP group A's SGLang engines completed CUDA graph capture and reached the
FSDP->SGLang weight-sync handoff point. DP group B's engines never started
capture. The most likely explanation given the observed thread states
(NCCL watchdog/heartbeat threads present and parked, not crashed) is a
cross-DP-group or cross-TP-group synchronization/barrier during rollout
engine initialization that group B's WorkerDicts stalled on before even
reaching the capture step -- but this is not confirmed against source, and no
code fix is applied based on this alone (would risk violating the section-7
"do not modify SGLang initialization/concurrency" constraint without a
verified root cause).

**SUPERSEDED (post-attempt-17 update):** the framing above treats "DP group
B never starts CUDA graph capture" as the anomaly needing explanation. Cross-
referencing against attempts 14 and 17 shows this diagnosis was for a hang
that occurs *after* checkpoint-resume printed "Setting global step to 60" /
"Resuming from .../global_step_60" but *before* any WorkerDict reaches
`load_checkpoint`'s per-rank torch.load() -- i.e. TP-rank-0-only SGLang
engine ownership within a DP group finishing capture while its sibling
group's WorkerDicts sit idle pre-capture is very plausibly just normal
scheduling skew (Ray/GPU scheduling jitter across two independent process
groups), not a cross-DP-group barrier bug. Attempts 14, 16, and 17 all share
one confirmed common signature instead: **zero occurrences, in any of the
three attempts, of any of the four `FSDPCheckpointManager.load_checkpoint`
log lines** ("Loaded model from" / "Loaded optimizer from" / "Loaded rng
from" / "Loaded lr_scheduler from"), despite these being unconditionally
printed (INFO level, unset `VERL_LOGGING_LEVEL`, for every rank) as soon as
each per-rank load completes. The driver's `_load_checkpoint()` call
(`ray_trainer.py`) happens strictly BEFORE `_validate()`/the training loop,
and dispatches a blocking `Dispatch.ONE_TO_ALL` RPC into every WorkerDict's
`FSDPCheckpointManager.load_checkpoint` (E1-0-patched via
`e10_checkpoint_patch.py`). This is now the leading hypothesis for attempts
14/16/17's shared hang, not a cross-DP-group SGLang init barrier. See
`checkpoint_load_hang_investigation.md` for the full narrowing and the
isolation test that ruled out raw `torch.load()` I/O contention as the
bottleneck (44s for all 4 ranks loading concurrently on this host, tested
live under current memory conditions). Passive per-rank timing
instrumentation has been added directly inside `e10_checkpoint_patch.py`
(bracketing `torch.load()`, `model.load_state_dict()`,
`optimizer.load_state_dict()`, and the final `torch.distributed.barrier()`)
so the next authorized attempt will produce definitive evidence instead of
silence.

## Action taken

No optimizer update occurred (zero `step:`/`num_gen_batches` log lines this
attempt). Per task section 16 (ordinary-error self-fix, no need for a new
attempt subdirectory since no optimizer update happened), attempt 16 was
cleanly terminated:

    kill -TERM 1877604

Verified after termination:
- No stray python3/ray/sglang processes remain (checked via `ps aux`).
- GPU memory returned to Wiki-server-only baseline (~9.7-10.2GB/GPU).
- Wiki server (pid 1016459) confirmed still healthy via functional
  `/retrieve` POST returning valid results.
- No partial checkpoint exists under
  `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/`.

Proceeding to attempt 17 using the identical, unmodified
`run_e10_step60to70.sh` / `e10_sglang_gpu_patch.py` / `e10_checkpoint_patch.py`
(no code-level root-cause fix identified yet for this specific hang; retrying
is consistent with this box's observed non-deterministic hang rate across
attempts 14 and 16, both of which got further than earlier attempts and
involved no code defect that's been pinpointed).

# Checkpoint-load hang investigation: attempts 14, 16, 17 (unresolved, GPU relaunch paused pending user confirmation)

## Context

Per explicit user instruction, all E1-0 GPU processes were stopped after
attempt 17 hung, and no further GPU training may be launched until (a) the
root cause is identified/fixed and (b) the user has explicitly confirmed.
This document consolidates the investigation performed entirely via static
code/config review and off-GPU isolation testing -- no training was
relaunched to produce it.

## Shared signature across attempts 14, 16, 17

All three: driver prints exactly

    Load from checkpoint folder: .../global_step_60
    Setting global step to 60
    Resuming from .../global_step_60

and then nothing else, indefinitely (killed after 70-90+ minutes of zero
GPU utilization / zero new log lines). GPU-collision, NCCL "Duplicate GPU",
and CUDA-IPC device-ordinal bugs (attempts 1-13's distinct root causes) are
all confirmed absent this time -- `nvidia-smi` shows correct, non-colliding
per-rank GPU assignment throughout.

## Ruled out this investigation

1. **Async-rollout-manager code path** (`AsyncLLMServerManager`,
   `AsyncSglangServer`, FastAPI/uvicorn, `ChatCompletionScheduler`): this run
   uses `actor_rollout_ref.rollout.mode=sync` (the Hydra default -- never
   overridden in `run_e10_step60to70.sh`, confirmed via `ppo_trainer.yaml`
   line 366 `mode: sync` and `main_ppo.py` lines 106/114's
   `AsyncActorRolloutRefWorker if mode == "async" else ActorRolloutRefWorker`
   selection). None of that machinery is ever constructed for this run.
2. **`copy_to_local`/FileLock contention**: local (non-`hdfs://`) checkpoint
   paths are a pure passthrough in `verl/verl/utils/fs.py`
   (`copy_local_path_from_hdfs` returns `src` unchanged for local paths) --
   no copy, no lock.
3. **`get_fsdp_state_ctx`'s `FSDP.state_dict_type()`**: confirmed
   non-blocking (thread-local config setter), not a collective.
4. **Disk I/O / disk space**: `dd` read test of a checkpoint shard achieved
   893 MB/s; `df -h` shows 529G free on `/data01`.
5. **Logging-level suppression**: `FSDPCheckpointManager`'s logger defaults
   to INFO (`os.getenv("VERL_LOGGING_LEVEL", "INFO")`), confirmed unset
   anywhere in the E1-0 launch environment -- the "Loaded model from" etc.
   lines would print if that code path executed, for every rank.
6. **Raw `torch.load()` I/O contention / host memory-swap thrashing during
   deserialization**: directly tested via isolation script (see below) --
   RULED OUT. Loading all 4 ranks' model+optimizer shards concurrently, live
   on this host, under the *current* memory conditions (15Gi swap in use,
   other unrelated jobs running, GPUs at 95%+ util from an unrelated vllm
   job) completed in **44 seconds total** with per-rank times of 2.4-22.3s.
   No hang, no unusual slowness. A single-rank load alone took ~15-18s. This
   was the previously-leading hypothesis (based on the absence of any
   "Loaded model from" print in attempts 14/16/17); it is now considered
   unlikely to be the sole cause, though it cannot fully rule out a *worse*
   contention scenario at the exact moment of a live 4-rank training run
   (real training also holds ~34-44GB/GPU of SGLang engine memory and a full
   torch/CUDA context per rank, which the isolation script did not
   replicate).

## Current leading hypothesis (not yet confirmed)

`FSDPCheckpointManager.load_checkpoint`'s per-rank `model.load_state_dict()`
/ `optimizer.load_state_dict()` calls, AFTER `torch.load()` returns.
FSDP's SHARDED_STATE_DICT load path
(`torch.distributed.fsdp._state_dict_utils._sharded_pre_load_state_dict_hook`)
performs a `dist.all_gather_into_tensor(tensor, local_tensor,
group=fsdp_state.process_group)` collective per parameter while
reconstructing each sharded `DTensor` back onto its owning rank's GPU. This
is a genuine NCCL/gloo collective (unlike anything upstream of it in this
function), runs strictly after `torch.load()` (already ruled out as slow)
and strictly before the existing "Loaded model from" print -- fully
consistent with all 3 hangs never printing that line. A stall here (rather
than a crash) would look exactly like the observed symptom: 0% GPU util,
parked NCCL watchdog threads, no forward progress, no error.

This is INFERRED, not VERIFIED -- no stack trace has been obtained in this
environment (`ptrace_scope=1`, no passwordless sudo, confirmed in earlier
attempts) and the isolation test above did not (and structurally could not,
without initializing full torch.distributed + FSDP-wrapped modules matching
the real run) reproduce this specific code path.

## Action taken this investigation

Added passive, non-blocking print-based timing instrumentation directly
inside `e10_checkpoint_patch.py`'s `_e10_patched_load_checkpoint`, bracketing:
  - function entry
  - `get_fsdp_state_ctx` entry
  - each `torch.load()` call (before/after)
  - each `.load_state_dict()` call (before/after)
  - `get_fsdp_state_ctx` exit
  - the final `torch.distributed.barrier()` (before/after)

Each line is tagged `[E1-0 ckpt-diag][rank N] <event> (t=<unix ts>)`, printed
unbuffered to stdout (captured by the existing `tee` to the launcher log and
by Ray's per-worker `.out` files). This is pure observation: it does not
alter control flow, reward, sampling, GRPO, optimizer, checkpoint, or tool
semantics -- consistent with the task's passive-audit mandate. Verified the
patched module still compiles and imports cleanly (no GPU/Ray/training
activity involved in that verification).

## Status

Root cause NOT yet confirmed. No GPU training has been relaunched. Per the
user's explicit standing instruction, attempt 18 (which would exercise this
new instrumentation and should finally show exactly which line inside
`load_checkpoint` is last reached before a hang, if one recurs) will not be
launched without the user's explicit go-ahead.

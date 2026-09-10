# E1-0 root cause of the silent post-checkpoint-load hang (attempt 19, 2026-09-10)

## Summary

**`monkey_patch_torch_reductions()` — sglang's CUDA-IPC device-UUID reduction
patch — was being executed at `e10_sglang_gpu_patch.py` MODULE IMPORT time, and
sitecustomize.py puts that module on the PYTHONPATH of every process,
including the Ray `TaskRunner` driver. With the patch active, the driver's
fork-based `StatefulDataLoader` never receives a batch from its workers.**

This is the single root cause of the silent hangs seen in attempts 14, 16, 17,
18 and 19a. It is not a distributed-collective problem, not a memory problem,
not a Ray scheduling problem, and not specific to the checkpoint load path.

Fix: the call is deferred to `_ensure_monkey_patched()`, invoked from the
patched `SGLangRollout._init_distributed_env`, i.e. **only inside the
WorkerDict/FSDP actor process** that actually needs it (it is the process that
serializes FSDP weights to the SGLang engine over CUDA IPC). The receiving
SGLang scheduler subprocesses already call the same function themselves
(`sglang/srt/model_executor/model_runner.py:1295`), so no coverage is lost.

## Direct evidence (attempt 19 / 19a, live run)

`sigusr1`-triggered `faulthandler` stacks (`logs/stackdumps/stackdump_pid2494707.txt`)
at the stall, ~8 minutes after the log froze:

Driver main thread:
```
stateful_dataloader.py:1215 _try_get_data
stateful_dataloader.py:1379 _get_data
stateful_dataloader.py:1423 _next_data
stateful_dataloader.py:450  __next__
ray_trainer.py:818          _validate          <-- waiting for the FIRST val batch
```
plus ~20 `multiprocessing/queues.py:231 _feed` (idle queue feeder) threads.

DataLoader worker (SIGUSR1 to pid 2507762, stack appended to the same file):
```
multiprocessing/queues.py:113 get          <-- never receives an index
torchdata/stateful_dataloader/worker.py:187 _worker_loop
```
`/proc/<pid>/wchan` for all 8 workers: main thread `poll_schedule_timeout`,
feeder thread `futex_wait_queue_me`; all 8 alive, 0% GPU.

`nvidia-smi` at stall: 0% utilisation on all four GPUs.

## Isolated reproduction (no Ray, no GPU, CPU-only)

`/tmp/dl_probe.py`: build the real `RLHFDataset` (nq.cleaned.parquet) + real
Qwen tokenizer, wrap in `StatefulDataLoader(num_workers=8)` and iterate.

| preloaded before the probe | result |
|---|---|
| nothing (baseline) | first batch in **0.5 s** |
| full E1-0 sitecustomize chain (before fix) | **no batch in 240 s** (hang) |
| `verl.workers.fsdp_workers` only | 0.5 s, 0 sglang modules, 1 thread |
| `verl.workers.sharding_manager.fsdp_sglang` | 0.5 s (181 sglang modules + `_read_thread` started — **harmless**) |
| `verl.workers.rollout.sglang_rollout.sglang_rollout` | 0.5 s (**harmless**) |
| `e10_hang_diagnostics` + `e10_reward_manager` + `e10_checkpoint_patch` + `e10_diag_patch` | 0.5 s (**harmless**) |
| `from sglang.srt.patch_torch import monkey_patch_torch_reductions; monkey_patch_torch_reductions()` | **hang** |
| `import e10_sglang_gpu_patch` (before fix; calls it) | **hang** |
| `import e10_sglang_gpu_patch` (after fix; call deferred) | **0.5 s** |
| full sitecustomize chain (after fix) | **0.5 s** |

Finer micro-repro: with the patch applied, a forked child hangs inside
`multiprocessing.Queue.put(torch.zeros(3))`; without the patch the same child
completes immediately. The patched `reduce_tensor` wrapper
(`_reduce_tensor_modified` in `sglang/srt/patch_torch.py`) maps the serialized
tensor's device through `_device_to_uuid` →
`torch.cuda.get_device_properties(device)`, so **pickling any torch tensor in a
forked child performs CUDA work in that child, which deadlocks.** Every
`StatefulDataLoader` worker pickles its collated batch (torch tensors) to send
it back, so no worker can ever deliver a batch.

## Why the previous attempts looked like several different bugs

* Attempts 14/16/17: the log froze right after `Resuming from .../global_step_60`
  because the *next* thing after the checkpoint RPC is the pre-training
  `_validate()` -> first `val_dataloader` iteration -> the fork hang. (Those
  attempts had no driver-side instrumentation, so "no output after Resuming"
  was the whole observable symptom.)
* Attempt 18: the checkpoint-load diagnostics happened to complete, which moved
  the visible freeze one step downstream to `_validate()` — same underlying
  fork hang, just observed at a slightly later boundary.
* Attempt 19a: identical to 18, now with per-phase prints that localised it to
  `_validate: entered` with no `test_gen_batch meta info`, and the SIGUSR1
  stacks above gave the exact frames.
* This also explains why the ORIGINAL 200-step GAP run never hit it: the
  original driver process never had sglang imported, so the reduction patch was
  never applied there and its `DataLoader` forks were unaffected.

## Fix and verification

* `e10_sglang_gpu_patch.py`: module-level `monkey_patch_torch_reductions()` call
  removed; added `_ensure_monkey_patched()` (idempotent) called at the top of
  `_e10_patched_init_distributed_env` (runs in the WorkerDict/FSDP actor before
  `_init_inference_engine`, i.e. before any `MultiprocessingSerializer.serialize()`).
* Verified: importing the module no longer breaks the DataLoader; the deferred
  call still sets `torch.multiprocessing.reductions._reduce_tensor_original`
  (i.e. the patch is really applied where needed); full sitecustomize chain
  now yields "GOT FIRST BATCH after 0.5s".
* Training semantics untouched: no reward/sampling/GRPO/optimizer/rollout/tool
  change; the patch is applied in exactly the process that needs it, only later
  in that process's lifetime.

Classification: **VERIFIED** (isolated, deterministic reproduction plus a live
run stack trace; fix verified by the same reproduction).

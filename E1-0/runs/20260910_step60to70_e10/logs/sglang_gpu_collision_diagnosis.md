# SGLang engine GPU collision under RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1: diagnosis and self-fix (attempt 5 -> attempt 8)

## Symptom (attempt 5)

```
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

Raised in `sglang/srt/model_executor/model_runner.py:init_memory_pool`, inside
the SGLang scheduler subprocess, for both TP0 and TP1 ranks simultaneously.
Occurred after the attempt-4 NCCL fix (`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`)
successfully cleared the FSDP/NCCL "Duplicate GPU detected" crash and after
the checkpoint-load OOM fix (`code/e10_checkpoint_patch.py`, see
`checkpoint_load_oom_diagnosis.md`) successfully cleared checkpoint resume.
Saved as `logs/launcher_attempt5_sglang_gpu_collision_crash.log`.

## Root cause

`verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`'s
`_init_distributed_env()` (line ~354) computes each TP group's SGLang engine
GPU assignment via:

```python
torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp"))
self.visible_devices_set = set(",".join(visible_devices).split(","))
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))
```

This implicitly assumes each rank's `CUDA_VISIBLE_DEVICES` has *already* been
narrowed by Ray to that rank's own single assigned physical GPU (exactly
Ray's *default* per-actor remap behavior). Under that assumption, unioning a
TP group's per-rank single-GPU sets correctly reconstructs that DP group's
2-GPU pair, distinct from the other DP group's pair.

`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` disables exactly that
narrowing: every rank's process sees the full unremapped `CUDA_VISIBLE_DEVICES=0,1,2,3`.
Ray never sets a `LOCAL_RANK` env var itself (only `RAY_LOCAL_RANK`); under
default remap, `worker.py`'s later `local_rank = int(os.getenv("LOCAL_RANK",
"0"))` read (line 161) picks up the unset-default `"0"`, which
`_configure_with_store` then writes back explicitly -- i.e. `LOCAL_RANK == "0"`
and `CUDA_VISIBLE_DEVICES` narrowed to one GPU is the standing invariant every
downstream consumer (FSDP/NCCL and SGLang alike) is built around. With
NOSET=1, `worker.py`'s own `_setup_env_cuda_visible_devices()`
(`verl/verl/single_controller/base/worker.py:225-232`) only does
`torch.cuda.set_device(RAY_LOCAL_RANK)` -- it steers which device index torch
treats as current, but never narrows `os.environ["CUDA_VISIBLE_DEVICES"]`.

Consequence: every rank's `CUDA_VISIBLE_DEVICES` stays `"0,1,2,3"`, so every
TP group's `all_gather_object` union in `sglang_rollout.py` collapses to the
same 4-GPU set regardless of which physical GPUs that DP group's ranks are
actually running `torch.cuda.set_device()` against. `_init_inference_engine`
then launches both DP groups' SGLang `AsyncEngine`s with `base_gpu_id=0`
against the same (mis-)computed 4-GPU visible set, so both groups' engines
target the same physical GPU pair (0,1) -- a direct memory collision (each
engine independently reserves ~`gpu_memory_utilization=0.5` of GPU 0/1's 80GiB,
which the second engine's `init_memory_pool` then finds insufficient).

## Fix applied (attempt 8)

Cannot edit `sglang_rollout.py` in place (baseline file, off-limits per task
sections 2/7 -- this is exactly the "no modification of SGLang" prohibition
in section 7). Instead, added `code/e10_sglang_gpu_patch.py`: a runtime
monkeypatch (same mechanism as `e10_checkpoint_patch.py`), imported from
`code/pythonpath_e10/sitecustomize.py`, which replaces
`Worker._setup_env_cuda_visible_devices` with a copy that additionally
narrows `os.environ["CUDA_VISIBLE_DEVICES"]` to the single physical GPU
indicated by `RAY_LOCAL_RANK` (indexing into the full visible-devices list)
and sets `LOCAL_RANK="0"`, `set_device(0)` -- i.e. deterministically
reproducing, in our own code, the exact per-rank single-GPU narrowing
invariant that Ray's default remap normally provides, without re-enabling
Ray's own internal remap machinery (whose race is what caused the original
NCCL crash in the first place).

This:
- Preserves `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` (the only setting
  that has ever cleared the FSDP/NCCL "Duplicate GPU detected" crash across 8
  attempts).
- Restores the single-GPU-per-rank `CUDA_VISIBLE_DEVICES` invariant that
  `sglang_rollout.py`'s (unmodified) TP-group union logic depends on, so each
  DP group's SGLang engine is correctly assigned its own distinct GPU pair.
- Changes zero GPU count, placement-group strategy, `max_colocate_count`,
  `tensor_model_parallel_size`, rollout/reward/GRPO semantics, or any
  section-7-audited config -- it only fixes which physical GPU string a rank
  reports for itself, matching the value the original (unpatched,
  default-remap) code path would have produced for that same rank.
- Qualifies as a section-16 "ordinary error" self-fix (Ray/CUDA device
  plumbing, not experiment logic), same category as the two prior fixes.

No optimizer update occurred in attempts 5, 6, or 7, so attempt 8 reruns
within the same `RUN_DIR` (no new attempt-subdirectory needed), with all
prior logs preserved (`launcher_attempt{1..7}_*.log`).

## UPDATE: attempt 8 crashed with IndexError -- idempotency fix (attempt 9)

`WorkerDict.__init__` (`verl/verl/single_controller/ray/base.py`)
instantiates multiple role classes (actor/rollout/ref) inside the **same**
process, each calling `Worker.__init__()` -> `_setup_env_cuda_visible_devices()`
again. The attempt-8 patch narrowed `CUDA_VISIBLE_DEVICES` from 4 entries to 1
on the first call; the second call in the same process then tried
`visible[int(local_rank)]` against an already-length-1 list with
`local_rank` e.g. `"3"`, raising `IndexError: list index out of range`
(`logs/launcher_attempt8_indexerror_crash.log`). Fixed by guarding the
narrowing to only run when more than one device is still visible. This still
crashed at attempt 9 for a deeper reason (see next update) and has since been
abandoned in favor of a different fix location.

## UPDATE: attempt 9 crashed with the original NCCL "Duplicate GPU detected" error -- wrong fix location, corrected (attempt 10)

Attempt 9 (idempotency-fixed `Worker._setup_env_cuda_visible_devices` patch)
progressed further than attempt 8 but crashed with the **identical** FSDP/NCCL
"Duplicate GPU detected" error attempt 4 was supposed to have fixed
(`logs/launcher_attempt9_nccl_crash_all_gpu0.log`), now with every rank
reporting collision against rank 0 specifically (rank1-vs-0, rank2-vs-0,
rank3-vs-0), rather than the more varied rank-pair pattern seen in
attempts 1/2/3/6/7.

Root cause of this regression: mutating `os.environ["CUDA_VISIBLE_DEVICES"]`
inside `Worker.__init__` has no effect on the *current* process's actual CUDA
device visibility, because CUDA's device enumeration is established (cached)
the first time any code in that process touches the CUDA runtime (via
earlier imports of `torch`/`sglang`/etc., which happens before
`Worker.__init__` runs) -- it is not re-read from the environment on each
subsequent `torch.cuda.set_device()` call. So narrowing the env var and then
calling `get_torch_device().set_device(0)` did not select "the one physical
GPU now named by index 0 in the narrowed list" -- it selected "index 0 of
whatever device list CUDA cached at process start", i.e. physical GPU 0, for
every rank simultaneously. This is a worse regression than the original
problem: it collapsed all 4 ranks onto GPU 0 instead of preserving attempt 4's
correct per-rank distinct-GPU assignment via `set_device(RAY_LOCAL_RANK)`.

**Fix (attempt 10): abandoned patching `Worker.__init__` entirely** (reverted
to exactly attempt 4's proven-working, unpatched state: only
`set_device(RAY_LOCAL_RANK)`, no `CUDA_VISIBLE_DEVICES` narrowing). Moved the
fix to `SGLangRollout._init_distributed_env`
(`verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`, ~line 317)
instead: this method's `AsyncEngine(...)` call
(`_init_inference_engine`, invoked immediately after) spawns **brand-new OS
subprocesses** for the SGLang engine, which read `CUDA_VISIBLE_DEVICES` fresh
from the environment at their own spawn time -- unlike the long-lived
`WorkerDict` actor process, a freshly `fork`/`spawn`-ed child process has no
pre-existing cached CUDA context, so narrowing the env var immediately before
that spawn *does* take effect (this is exactly the mechanism the original,
unpatched code already relies on under Ray's default remap; only the
*computed value* being gathered was wrong under NOSET=1, not the mechanism
itself).

`code/e10_sglang_gpu_patch.py` was rewritten accordingly: it now patches
`SGLangRollout._init_distributed_env` to gather each rank's true single
physical GPU (derived by indexing the full `CUDA_VISIBLE_DEVICES` list with
`RAY_LOCAL_RANK` -- the same identifier attempt 4's working FSDP fix already
depends on) instead of gathering each rank's raw, now-unreliable-under-NOSET=1
`os.environ["CUDA_VISIBLE_DEVICES"]`. Under default remap (NOSET unset) this
reduces to the original behavior exactly (a no-op passthrough), since
`os.environ["CUDA_VISIBLE_DEVICES"]` is already that rank's single GPU in
that regime. `Worker.__init__`/`_setup_env_cuda_visible_devices` is no longer
patched at all.

No optimizer update occurred in attempts 8 or 9, so attempt 10 reruns within
the same `RUN_DIR`, with logs preserved as `launcher_attempt8_indexerror_crash.log`
and `launcher_attempt9_nccl_crash_all_gpu0.log`.

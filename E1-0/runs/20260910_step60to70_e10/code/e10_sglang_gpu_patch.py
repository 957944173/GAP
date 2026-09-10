"""E1-0 self-fix: correct SGLang per-DP-group GPU assignment under RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1.

Background (full history in ../logs/nccl_duplicate_gpu_diagnosis.md,
../logs/checkpoint_load_oom_diagnosis.md, ../logs/sglang_gpu_collision_diagnosis.md):

Attempts 1/2/3/6/7 (Ray's *default* per-actor CUDA_VISIBLE_DEVICES remap)
crashed 5/5 with NCCL "Duplicate GPU detected" during FSDP's first
_sync_module_params_and_buffers. RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
(attempt 4) is the only setting that has ever avoided this crash: it disables
Ray's own per-actor CUDA_VISIBLE_DEVICES remap, and verl's own
Worker._setup_env_cuda_visible_devices() (verl/verl/single_controller/base/worker.py,
~line 225) instead does `torch.cuda.set_device(RAY_LOCAL_RANK)` directly --
UNPATCHED, this is correct and sufficient for FSDP/NCCL; do not touch it.

That setting's side effect is on a *different* consumer:
verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py's
_init_distributed_env() (~line 354) computes each TP group's SGLang engine
GPU assignment by torch.distributed.all_gather_object-ing every rank's own
os.environ["CUDA_VISIBLE_DEVICES"] and unioning it:

    torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], tp_group)
    self.visible_devices_set = set(",".join(visible_devices).split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

Under Ray's default remap, each rank's CUDA_VISIBLE_DEVICES is *already*
narrowed to that rank's own single physical GPU, so this union correctly
reconstructs each DP group's distinct GPU pair. Under NOSET=1, every rank's
CUDA_VISIBLE_DEVICES is the full unremapped "0,1,2,3" (Worker.__init__ never
narrows it -- it only steers set_device()), so every TP group's union
collapses to the same full set, and _init_inference_engine's base_gpu_id=0
sends every DP group's SGLang AsyncEngine subprocess to the same physical
GPUs -- a memory collision (attempt 5's `RuntimeError: Not enough memory`).

FIRST FIX ATTEMPT (superseded, see git-free history below): patching
Worker._setup_env_cuda_visible_devices itself to narrow CUDA_VISIBLE_DEVICES
before set_device() was tried and FAILED (attempt 9): mutating the env var
inside the *same* process that already has an initialized CUDA context (from
earlier imports such as torch/sglang) has no effect on that process's actual
device visibility -- CUDA's device enumeration is cached at first
context-touching call, not re-read from the environment afterward. The result
was every rank's set_device(0) resolving to physical GPU 0 for all 4 ranks
simultaneously, i.e. the *same* "Duplicate GPU detected" NCCL crash attempt 4
was supposed to fix. That approach is abandoned; Worker.__init__ is left
completely unpatched (exactly attempt 4's proven-working state).

ACTUAL FIX (attempt 10): patch SGLangRollout._init_distributed_env instead.
This method runs inside the long-lived WorkerDict actor process (whose CUDA
context is already pinned via Worker.__init__'s set_device(RAY_LOCAL_RANK) --
that pinning is correct and untouched), but the value it computes
(os.environ["CUDA_VISIBLE_DEVICES"]) is consumed by _init_inference_engine's
`AsyncEngine(...)` call immediately afterward, which spawns *brand-new* OS
subprocesses (SGLang scheduler/detokenizer/etc.) that read CUDA_VISIBLE_DEVICES
fresh at spawn time -- so setting it correctly *here*, right before that
spawn, does take effect (this is exactly what the original code already
relies on under default remap; we are only fixing what value gets computed,
not introducing a new mechanism).

The fix replaces the union step: instead of gathering each rank's (now
unreliable, always-full-under-NOSET=1) os.environ["CUDA_VISIBLE_DEVICES"],
gather each rank's true single physical GPU id, derived from indexing the
original full CUDA_VISIBLE_DEVICES list with RAY_LOCAL_RANK (the same
proven-reliable per-rank identifier attempt 4's FSDP fix already depends on).
Under default remap (RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES unset), this
patch is a no-op passthrough to the original behavior (os.environ["CUDA_VISIBLE_DEVICES"]
is already this rank's single GPU in that regime, so indexing produces the
same value).

This is a pure GPU-device-assignment plumbing fix: it does not touch GPU
count, placement-group strategy, max_colocate_count, tensor_model_parallel_size,
rollout sampling/reward/GRPO semantics, or any section-7-audited config.
Cannot edit verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py in
place (baseline file, off-limits per task sections 2/7 -- the explicit
"no modification of SGLang" prohibition); this is a runtime monkeypatch, same
pattern as e10_checkpoint_patch.py.

UPDATE (attempt 13 -> attempt 14): a second, distinct bug surfaced further
into the pipeline, during the pre-training validation rollout's weight sync
(FSDP -> SGLang engine), also a side effect of NOSET=1:

    RuntimeError: CUDA error: invalid device ordinal
    ... torch/multiprocessing/reductions.py:181 rebuild_cuda_tensor ->
    torch.UntypedStorage._new_shared_cuda(...)

verl/verl/workers/sharding_manager/fsdp_sglang.py's update_weights() calls
sglang's MultiprocessingSerializer.serialize() (ForkingPickler) on each
weight tensor *inside the FSDP/WorkerDict actor process* to hand it to the
SGLang engine subprocess via CUDA IPC. sglang ships a fix for exactly this
cross-process CUDA_VISIBLE_DEVICES-mismatch class of bug
(sglang/srt/patch_torch.py:monkey_patch_torch_reductions, referencing
https://github.com/pytorch/pytorch/pull/149248): it makes both the
serializing (reduce_tensor) and deserializing (rebuild_cuda_tensor) paths
encode/resolve the GPU by UUID instead of raw CUDA device index, so it works
regardless of whether sender and receiver see different-sized/ordered
CUDA_VISIBLE_DEVICES. sglang already calls this patch on the *receiving* side
(model_executor/model_runner.py's _unwrap_tensor) and in VerlEngine.__init__
-- but verl's actual codepath here uses sglang_rollout.py's own AsyncEngine
directly, never VerlEngine, so the patch is never applied in the *sending*
process (the FSDP/WorkerDict actor calling .serialize()). Under Ray's default
per-actor remap this never surfaces (every process narrows to exactly 1 GPU
seen as index 0 on both ends, so raw index happens to match). Under NOSET=1
(required for the FSDP/NCCL fix above), the FSDP process sees all 4 physical
GPUs unremapped -- a tensor can carry a raw device index of 2 or 3 -- while
the receiving SGLang scheduler subprocess only sees its TP group's narrowed
2-GPU (or 1-GPU) view, so that raw index is out of range there.

Fix: call sglang's own monkey_patch_torch_reductions() here too (idempotent,
no-ops if already applied), so it runs in the WorkerDict/FSDP process before
any MultiprocessingSerializer.serialize() call, in addition to sitecustomize
loading this same module in the spawned SGLang scheduler subprocess (which
also gets it, redundantly-safely, via model_runner.py's own call). This does
not change training semantics -- it only fixes which bytes an existing
CUDA-IPC device handle resolves to across two processes with different GPU
visibility, using a mechanism sglang itself ships and already relies on for
its own VerlEngine entrypoint. Not a rollout/reward/GRPO/sampling change.

ATTEMPT-19 CORRECTION to the paragraph above: `monkey_patch_torch_reductions()`
must NOT be executed at this module's import time (sitecustomize imports this
module in EVERY process, including the driver). Doing so deadlocks fork-based
StatefulDataLoader workers and was the actual cause of the silent
post-checkpoint-load hangs of attempts 14/16/17/18/19a. It is now deferred to
`_ensure_monkey_patched()`, called from `_e10_patched_init_distributed_env`,
i.e. only inside the WorkerDict/FSDP actor process. See that function's
docstring and analysis/root_cause_dataloader_fork.md for the isolated
reproduction and the evidence chain.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from sglang.srt.patch_torch import monkey_patch_torch_reductions
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.workers.rollout.sglang_rollout import sglang_rollout as _sglang_rollout_mod
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

logger = _sglang_rollout_mod.logger
sglang_ps = _sglang_rollout_mod.sglang_ps

_MONKEY_PATCHED = False


def _ensure_monkey_patched():
    """Apply sglang's CUDA-IPC reductions patch exactly once, in the calling
    process only.

    E1-0 FIX (attempt 19): this call used to run at MODULE IMPORT time, i.e. in
    every process that has sitecustomize on PYTHONPATH -- including the
    TaskRunner driver.  That broke fork-based `StatefulDataLoader` completely:
    with the patch active, the very first `_validate()` never received a batch
    from any of its 8 workers (driver blocked in
    `stateful_dataloader._try_get_data`; workers blocked in
    `torchdata...worker._worker_loop -> index_queue.get()`), the launcher log
    froze and GPU util stayed at 0% -- the exact signature of attempts 14/16/
    17/18/19a.

    Root cause (OBSERVED, reproduced in isolation on this box, no Ray, no GPU
    needed -- see analysis/root_cause_dataloader_fork.md):
      * `from sglang.srt.patch_torch import monkey_patch_torch_reductions;
        monkey_patch_torch_reductions()` followed by a fork-based
        `multiprocessing.Queue` + `q.put(torch.zeros(3))` in the child hangs;
        without the call it completes immediately.
      * The patch rebinds `torch.multiprocessing.reductions.reduce_tensor` to
        `_reduce_tensor_modified`, whose `_modify_tuple(..., 6, _device_to_uuid)`
        unconditionally maps the serialized tensor's device through
        `torch.cuda.get_device_properties()`.  Any forked child that then
        pickles a torch tensor (exactly what every DataLoader worker does with
        its collated batch) performs CUDA work inside a forked process, which
        deadlocks.
    Importing sglang alone is harmless (verified): only the *call* breaks fork.

    The patch is genuinely needed only in the long-lived WorkerDict/FSDP actor
    process that hands weights to the SGLang engine via
    `MultiprocessingSerializer.serialize()` -- so it is now applied lazily from
    `_e10_patched_init_distributed_env`, which runs inside that process (and,
    per sglang_rollout.py, strictly before `_init_inference_engine`).  The
    receiving SGLang scheduler subprocesses call the same function themselves
    (sglang/srt/model_executor/model_runner.py:1295), so nothing is lost.
    """
    global _MONKEY_PATCHED
    if _MONKEY_PATCHED:
        return
    monkey_patch_torch_reductions()
    _MONKEY_PATCHED = True


def _e10_patched_init_distributed_env(self, device_mesh_cpu, **kwargs):
    _ensure_monkey_patched()
    self._device_mesh_cpu = device_mesh_cpu
    os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
    self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
    assert self.tensor_parallel_size <= dist.get_world_size(), "tensor parallel size should be less than or equal to the world size"
    self.train_tp = kwargs.get("train_tp", None)
    if self.train_tp is not None:
        # deployed with megatron
        os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
        os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
        train_tp = kwargs.get("train_tp", None)
        num_tp_per_train_tp = train_tp // self.tensor_parallel_size
        sglang_ps.initialize_parallel_state(
            tensor_model_parallel_size=self.tensor_parallel_size,
            num_tp_per_train_tp=num_tp_per_train_tp,
        )

    tp_size = self.tensor_parallel_size
    world_size = int(os.getenv("WORLD_SIZE", "-1"))

    # init device mesh
    if self._device_mesh_cpu is None:
        device_mesh_kwargs = dict(
            mesh_shape=(world_size // tp_size, tp_size, 1),
            mesh_dim_names=["dp", "tp", "pp"],
        )

        self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

    self._rank = self._device_mesh_cpu.get_rank()
    self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
    self._tp_size = self._device_mesh_cpu["tp"].size()
    if self._rank == 0:
        logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
    # get tp_rank of this process in this tp group
    visible_devices = [None] * self._device_mesh_cpu.size(1)

    # E1-0 fix: under RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1,
    # os.environ["CUDA_VISIBLE_DEVICES"] is the full unremapped set on every
    # rank (Ray's own per-actor narrowing is disabled and nothing else
    # narrows it), so gathering it directly collapses every TP group's union
    # to the same full set. Gather this rank's true single physical GPU
    # (derived from RAY_LOCAL_RANK, the same identifier the FSDP/NCCL fix
    # already relies on) instead. Under default remap this reduces to the
    # original behavior (os.environ["CUDA_VISIBLE_DEVICES"] is already this
    # rank's single GPU in that regime).
    if ray_noset_visible_devices():
        full_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        this_rank_visible_devices = full_visible_devices[int(os.environ["RAY_LOCAL_RANK"])]
    else:
        this_rank_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]

    torch.distributed.all_gather_object(visible_devices, this_rank_visible_devices, self._device_mesh_cpu.get_group("tp"))
    self.visible_devices_set = set(",".join(visible_devices).split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))


SGLangRollout._init_distributed_env = _e10_patched_init_distributed_env

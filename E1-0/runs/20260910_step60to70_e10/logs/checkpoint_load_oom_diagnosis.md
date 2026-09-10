# CUDA OOM during checkpoint load: diagnosis and self-fix (attempt 4 -> attempt 5)

## Symptom

Occurred immediately after the NCCL crash was resolved (see
`nccl_duplicate_gpu_diagnosis.md`), during genuine checkpoint resume, i.e.
strictly *after* `Setting global step to 60` / `Resuming from
.../global_step_60` was logged, and strictly *before* any optimizer step:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 74.00 MiB. GPU 0
has a total capacity of 79.25 GiB of which 38.06 MiB is free. Process 1016459
has 9.92 GiB memory in use. Process 1275461 has 9.10 GiB memory in use.
Process 1276213 has 2.43 GiB memory in use. Process 1276214 has 2.36 GiB
memory in use. Including non-PyTorch memory, this process has 2.29 GiB memory
in use. Process 1278687 has 30.26 GiB memory in use. Process 1278689 has
22.81 GiB memory in use. Of the allocated memory 1.88 GiB is allocated by
PyTorch, and 2.50 MiB is reserved by PyTorch but unallocated.
```

Traceback: `ray_trainer.py:1510 _load_checkpoint()` ->
`actor_rollout_wg.load_checkpoint()` -> `fsdp_workers.py:805` ->
`fsdp_checkpoint_manager.py:117 torch.load(local_optim_path,
weights_only=False)` -> `torch/_utils.py:_to()` ->
`torch.UntypedStorage(self.size(), device=device)`. Multiple `WorkerDict`
ranks (pids 1276212, 1276213, 1276214) failed identically. Saved as
`logs/launcher_attempt4_oom_crash.log`.

Confirmed clean aftermath: all training processes exited; only the Wiki
server (pid 1016459) remained on the GPUs; no optimizer update occurred; no
partial/corrupted checkpoint output directory was created under
`experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/`.

## Root cause

This is a **direct, mechanistic side effect of the attempt-4 NCCL fix**, not
an independent issue:

1. The step60 checkpoint (`experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60`)
   was originally saved under Ray's *default* regime, where Ray remaps
   `CUDA_VISIBLE_DEVICES` per actor so each rank's process sees exactly one
   physical GPU, always enumerated as that process's `cuda:0`. Consequently,
   every rank's pickled `model_world_size_4_rank_{r}.pt` /
   `optim_world_size_4_rank_{r}.pt` tensors carry a `cuda:0` storage-location
   tag in their pickle metadata (`torch.load`'s per-tensor device comes from
   this tag, not from the current process's device topology).
2. `verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py`'s
   `load_checkpoint()` calls `torch.load(local_model_path, weights_only=False)`
   / `torch.load(local_optim_path, weights_only=False)` with **no
   `map_location` argument**. PyTorch's default restore behavior places each
   deserialized tensor directly onto the device named by its pickled storage
   location -- i.e., literal `cuda:0` -- via `torch._utils._to()` ->
   `torch.UntypedStorage(size, device=device)`. This happens *before* the
   surrounding `get_fsdp_state_ctx(..., ShardedStateDictConfig(offload_to_cpu=True),
   ShardedOptimStateDictConfig(offload_to_cpu=True))` context's
   sharding/offload logic ever gets a chance to act -- `offload_to_cpu=True`
   only governs where FSDP *re-shards state onto after* `load_state_dict`
   ingests it, not where `torch.load` initially deserializes the raw pickle.
3. Attempt 4's NCCL fix (`RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`, see
   `nccl_duplicate_gpu_diagnosis.md`) makes every rank's process see **all 4**
   physical GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3` unremapped), with verl's own
   `worker.py:_setup_env_cuda_visible_devices()` instead doing
   `torch.cuda.set_device(RAY_LOCAL_RANK)`. This is necessary and correct for
   avoiding the NCCL race, but it means `"cuda:0"` in every rank's pickle now
   resolves to the *same physical device* (physical GPU 0) for all 4 ranks
   simultaneously, instead of each rank's own distinct physical GPU as it did
   under the original save-time regime. All 4 ranks therefore attempt to
   materialize their full local model+optimizer shard onto physical GPU 0 at
   the same time, on top of GPU 0 already hosting the Wiki server (9.92 GiB)
   and this rank's own SGLang rollout engine allocation -- exhausting GPU 0's
   79.25 GiB.
4. This is a **pre-existing latent bug in `fsdp_checkpoint_manager.py`**
   (missing `map_location="cpu"` on a checkpoint-resume path that explicitly
   declares `offload_to_cpu=True` everywhere else), only *exposed* by
   E1-0's necessary NCCL fix. The original run (`train_dapo_mhqa_agent_wiki.sh`,
   step0->200) never exercised `load_checkpoint()` at all -- it trained
   continuously from a freshly-initialized model and only ever *saved*
   checkpoints, never *resumed* one -- so this bug had no prior chance to
   manifest on this box.

## Fix applied (attempt 5)

Cannot edit `verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py` in place
(original baseline file, off-limits per task sections 2/7). Instead, added
`code/e10_checkpoint_patch.py`: a runtime monkeypatch, imported from
`code/pythonpath_e10/sitecustomize.py` (the same mechanism already used to
register the `e10_batch_audit` reward manager), which replaces
`FSDPCheckpointManager.load_checkpoint` with a byte-identical copy except
that all three `torch.load(...)` calls (model, optimizer, extra_state) gain
an explicit `map_location="cpu"`. This:

- Forces PyTorch to deserialize every pickled tensor onto CPU regardless of
  its pickled storage-location tag, eliminating the physical-GPU-0 contention
  entirely.
- Changes **zero training semantics**: the resulting `model_state_dict` /
  `optimizer_state_dict` / `extra_state_dict` are bit-identical in content
  (dtype, shape, values) to what the unpatched code would have produced --
  only the transient device during deserialization differs. FSDP's
  `ShardedStateDictConfig`/`ShardedOptimStateDictConfig` sharding logic (via
  `self.model.load_state_dict(...)` / `self.optimizer.load_state_dict(...)`
  inside `get_fsdp_state_ctx(...)`) then scatters this CPU-resident full
  state onto each rank's assigned GPU exactly as it already does for any
  `offload_to_cpu=True` load -- this is the standard, intended codepath, not
  a novel one.
- Is not a rollout/reward/GRPO/optimizer-math change and does not touch any
  section-7-audited semantic; it only fixes *where* the same bytes are
  deserialized during a resume. Qualifies as a section-16 "ordinary error"
  self-fix (import/library-boundary-adjacent, not experiment-logic).

No optimizer update occurred in attempt 4, so attempt 5 reruns within the
same `RUN_DIR` (no new attempt-subdirectory needed), with attempt 4's log
preserved as `logs/launcher_attempt4_oom_crash.log`.

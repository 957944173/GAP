"""E1-0 runtime monkeypatch: fix CUDA OOM in FSDPCheckpointManager.load_checkpoint.

Root cause (attempt 4, see logs/nccl_duplicate_gpu_diagnosis.md addendum):
E1-0's NCCL self-fix sets RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1, which
means every rank's process now sees all 4 physical GPUs (verl's own
worker.py:_setup_env_cuda_visible_devices does torch.cuda.set_device(RAY_LOCAL_RANK)
instead of relying on Ray's per-actor CUDA_VISIBLE_DEVICES remap). The original
step60 checkpoint, however, was saved under the OLD regime where Ray remapped
each actor to see exactly one physical GPU as "cuda:0" -- so every rank's
pickled *.pt tensors carry a "cuda:0" storage-location tag.

verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py's load_checkpoint() calls
torch.load(local_model_path, weights_only=False) / torch.load(local_optim_path,
weights_only=False) with NO map_location argument. torch.load's default
behavior restores each tensor to the device its pickled storage location names
-- i.e. literally "cuda:0" -- BEFORE FSDP's surrounding
get_fsdp_state_ctx(..., ShardedStateDictConfig(offload_to_cpu=True), ...)
context ever gets a chance to shard/redistribute anything. With 4 ranks now
all physically sharing GPU 0 (because CUDA_VISIBLE_DEVICES is un-remapped) and
each rank trying to deserialize its own full local model+optimizer shard onto
literal GPU 0, GPU 0 (already hosting the Wiki server + this rank's own
SGLang rollout engine) runs out of memory.

Fix: monkeypatch FSDPCheckpointManager.load_checkpoint with an identical copy
that adds map_location="cpu" to the three torch.load() calls, matching the
offload_to_cpu=True intent the surrounding FSDP context config already
declares. This changes ZERO training semantics (model/optimizer/RNG/scheduler
state is bit-identical; only the transient deserialization device differs --
FSDP's ShardedStateDictConfig/ShardedOptimStateDictConfig sharding logic
scatters the (CPU-resident) full state back onto each rank's assigned GPU
exactly as it already does for the offload_to_cpu=True path). No original
verl file is modified; this is applied purely at runtime via monkeypatch,
imported from pythonpath_e10/sitecustomize.py before training starts.
"""
import os
import time

import torch
import torch.distributed

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.fs import copy_to_local, is_non_local
from verl.utils.fsdp_utils import get_fsdp_state_ctx
from verl.utils.logger import log_with_rank
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType

logger = FSDPCheckpointManager.__module__ and __import__("logging").getLogger(FSDPCheckpointManager.__module__)


def _e1d_diag(self, msg):
    # E1-0 PASSIVE DIAGNOSTIC (added after attempts 14/16/17 hung with zero
    # "Loaded model from"/"Loaded optimizer from" log lines from ANY rank,
    # while an isolated 4-way-concurrent torch.load() reproduction on this
    # same host completed in ~44s with no stall -- ruling out raw
    # deserialization I/O as the bottleneck. This narrows the leading
    # hypothesis to the FSDP-internal collective inside
    # model.load_state_dict()/optimizer.load_state_dict() themselves (e.g.
    # _sharded_pre_load_state_dict_hook's dist.all_gather_into_tensor on
    # fsdp_state.process_group), which runs AFTER torch.load() returns but
    # BEFORE the existing "Loaded model from" print. These prints are pure
    # observation (unbuffered stdout via print(), rank-tagged) -- they do not
    # alter control flow, reward, sampling, GRPO, optimizer, or checkpoint
    # semantics in any way.
    print(f"[E1-D ckpt-diag][rank {self.rank}] {msg} (t={time.time():.1f})", flush=True)


def _e1d_patched_load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
    if local_path is None:
        return

    if self.should_load_model:
        assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
    if self.should_load_optimizer:
        assert self.optimizer is not None, "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"

    _e1d_diag(self, "load_checkpoint: entered")
    state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True) if self.should_load_model else None
    optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True) if self.should_load_optimizer else None
    with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
        _e1d_diag(self, "entered get_fsdp_state_ctx")
        if self.should_load_model:
            remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
            local_model_path = copy_to_local(remote_model_path)
            # E1-0 FIX: map_location="cpu" (see module docstring) -- avoids
            # deserializing every rank's shard onto literal GPU 0.
            _e1d_diag(self, f"torch.load(model) starting: {local_model_path}")
            model_state_dict = torch.load(local_model_path, weights_only=False, map_location="cpu")
            _e1d_diag(self, "torch.load(model) done; calling model.load_state_dict()")
            self.model.load_state_dict(model_state_dict)
            _e1d_diag(self, "model.load_state_dict() returned")
            log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

        if self.should_load_optimizer:
            remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
            local_optim_path = copy_to_local(remote_optim_path)
            # E1-0 FIX: map_location="cpu" (see module docstring).
            _e1d_diag(self, f"torch.load(optim) starting: {local_optim_path}")
            optimizer_state_dict = torch.load(local_optim_path, weights_only=False, map_location="cpu")
            _e1d_diag(self, "torch.load(optim) done; calling optimizer.load_state_dict()")
            self.optimizer.load_state_dict(optimizer_state_dict)
            _e1d_diag(self, "optimizer.load_state_dict() returned")
            log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)
    _e1d_diag(self, "exited get_fsdp_state_ctx")

    if self.should_load_extra:
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        local_extra_state_path = copy_to_local(remote_extra_state_path)
        # E1-0 FIX: map_location="cpu" -- extra_state holds RNG/lr_scheduler
        # state, not sharded by FSDP, so it's small, but keep it consistent.
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False, map_location="cpu")
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])
            log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
        if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

    if self.rank == 0 and del_local_after_load:
        try:
            os.remove(local_model_path) if is_non_local(local_model_path) else None
            os.remove(local_optim_path) if is_non_local(local_optim_path) else None
            os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
        except Exception as e:
            log_with_rank(f"remove local resume ckpt file after loading failed, exception {e} will be ignored", rank=self.rank, logger=logger)

    _e1d_diag(self, "about to enter final torch.distributed.barrier()")
    torch.distributed.barrier()
    _e1d_diag(self, "load_checkpoint: fully returned")


FSDPCheckpointManager.load_checkpoint = _e1d_patched_load_checkpoint

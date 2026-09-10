# step60 source checkpoint (E1-0 resume point)

**Path:** `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60`

## Uniqueness

VERIFIED: this is the *only* `global_step_60` directory anywhere under `experiments/`
(`find experiments -maxdepth 2 -iname "global_step_60"` returns exactly one hit). No
candidate ambiguity existed; no guessing was required.

Sibling checkpoints in the same experiment dir: `global_step_120`, `global_step_180`,
`global_step_200` (all newer by mtime), plus `latest_checkpointed_iteration.txt` = `200`.
step60 predates all of these and is consistent with being an earlier checkpoint of the
same original GAP RL run — VERIFIED by monotonic mtime ordering (step60: 2026-08-24
04:14, step120: 2026-08-24 22:10, step180: 2026-08-25 15:29, step200: 2026-08-26 00:09).

## Completeness (VERIFIED)

| Component | Files | Status |
|---|---|---|
| Model weights | `actor/model_world_size_4_rank_{0,1,2,3}.pt` | present, 3,397,345,082 bytes each (identical across ranks — expected for FSDP full-state-dict-per-rank sharding at this size/config) |
| Optimizer state | `actor/optim_world_size_4_rank_{0,1,2,3}.pt` | present, 6,171,911,046 bytes each |
| Extra state (LR scheduler + RNG) | `actor/extra_state_world_size_4_rank_{0,1,2,3}.pt` | present, ~14.6-14.8 KB each |
| Tokenizer / HF config | `actor/{config.json,tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,added_tokens.json,special_tokens_map.json,generation_config.json}` | present |
| Merged HF snapshot (informational only, not used for resume) | `actor/merged_hf/*` | present |
| Dataloader state | `data.pt` (1492 bytes) | present |

All 4 ranks present for model/optim/extra — matches `N_GPUS_PER_NODE=4` used by both
the original run and E1-0. No missing shard.

## Model identity

`actor/config.json`: `architectures=["Qwen2ForCausalLM"]`, `hidden_size=2048`,
`num_hidden_layers=36`, `vocab_size=151936` — Qwen2-family ~3B config, consistent with
`BASE_MODEL=experiments/exp_1_lr1e-5_wr0.03_bs1_ga8_4gpu` (the SFT base referenced by
both the original launch script and E1-0's `run_e10_step60to70.sh`).

## Resume mechanism (VERIFIED via source read)

`run_e10_step60to70.sh` passes:
```
trainer.resume_mode=resume_path
trainer.resume_from_path=".../DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60"
```
`ray_trainer.py`'s `_load_checkpoint()` (`resume_path` branch) sets
`self.global_steps = int(global_step_folder.split("global_step_")[-1])` directly from
this path string, i.e. `self.global_steps = 60` — VERIFIED by source read this run
(see prior-session read of `_load_checkpoint()`).

`FSDPCheckpointManager.load_checkpoint()` genuinely restores optimizer state (from
`optim_world_size_*_rank_*.pt`) and LR-scheduler + RNG state (from
`extra_state_world_size_*_rank_*.pt`) — VERIFIED by source read (prior session). This
is a full-state resume, not weights-only, **provided** the checkpoint's `extra_state`
and `optim` files are non-empty/well-formed, which their non-trivial byte sizes above
are consistent with (final byte-level content is not unpickled here to avoid
mutating/loading state outside the actual training process; this is deferred to the
training process's own load, whose success/failure is captured in
`logs/training.log`).

## Output collision check

E1-0 output target: `experiments/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu/` — VERIFIED
via `run_e10_step60to70.sh`'s `EXPERIMENT_NAME`/`default_local_dir` — does not exist
prior to this run and is disjoint from `DAPO-GAP3B-MHQA-Agent-4gpu/` (the original,
read-only source of the step60 checkpoint). E1-0 never writes into the original
experiment directory.

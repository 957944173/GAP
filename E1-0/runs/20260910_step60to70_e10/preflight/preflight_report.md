# E1-0 Preflight Report

RUN_DIR: `/data01/wyy/Graph-Agent-Planning/E1-0/runs/20260910_step60to70_e10`

## A. Repository state

- **git root:** `/data01/wyy/Graph-Agent-Planning` (VERIFIED: `.git` exists at repo root).
- **branch:** `main`.
- **commit:** none — `git log` reports "your current branch 'main' does not have any
  commits yet". VERIFIED this is a freshly-initialized empty git repo (`.git` dir
  mtime 2026-09-10 01:13, i.e. created moments before this preflight run, almost
  certainly by tooling unrelated to E1-0) with **zero commits** — there is no baseline
  commit to diff against.
- **git status:** every top-level directory (`Agent/`, `verl/`, `experiments/`,
  `E1-0/`, etc.) shows as untracked (`??`). This is a **deviation** from task section 2's
  instruction to snapshot `git status`/`git diff` before/after changes, but it is not
  an E1-0-caused deviation: the repo had no commit history before E1-0 touched
  anything. DECISION (E1-0 self-fix, documented per task's self-fix/document
  instruction): since there is no meaningful git baseline, file-level integrity is
  instead proven via explicit sha256/size/mtime capture of every original file E1-0
  depends on, both listed once in `code/original_files_manifest.txt` (captured before
  and re-checked after the run — see FINAL_REPORT for the after-run re-hash). This
  gives the same "original files unchanged" guarantee git diff would have, without
  relying on an absent commit history. No commit was created by E1-0 (creating one
  was out of scope and risked capturing unrelated large binary/checkpoint files into
  a 2.9GB-and-growing `.git`).
- **Python import path:** `python3` resolves to `/home/nf5468m6/miniconda3/bin/python3`
  in the base env at preflight-check time; the actual training run activates
  `parallel-agent` per task section 6 before launch (see `environment/environment.txt`).

## B. Input training files (traced from `run_gen_bs_supervisor.sh` -> `train_dapo_mhqa_agent_wiki.sh`)

| File | Exists | Size | mtime | Rows | Columns |
|---|---|---|---|---|---|
| `Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet` (train) | YES | 61,978,764 bytes | 2026-07-02 15:38:06 | 169,595 | question, answer, data_source, prompt, reward_model, extra_info |
| `Agent/data/mhqa_agent/test_benchmarks/nq.cleaned.parquet` (val) | YES | 1,108,051 bytes | 2026-07-02 18:24:40 | 3,610 | data_source, prompt, reward_model, extra_info |
| `verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml` (tool config) | YES | 502 bytes | 2026-07-01 10:52:23 | n/a | n/a |
| `experiments/exp_1_lr1e-5_wr0.03_bs1_ga8_4gpu` (BASE_MODEL, SFT base) | YES | directory | 2026-09-04 16:41:50 | n/a | n/a |

No file was missing; no dataset substitution was needed. These are exactly the paths
`run_e10_step60to70.sh` uses (verbatim copy of the original script's paths — see
`code/run_e10_step60to70.sh` lines 100-107).

## C. step60 checkpoint

See `checkpoints_manifest/step60_source.md` for full detail. Summary:
- **Path:** `experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60` — the ONLY
  `global_step_60` directory in the repo (VERIFIED via `find`).
- **Completeness:** model/optimizer/extra_state present for all 4 ranks
  (`world_size_4_rank_{0,1,2,3}`), plus tokenizer files and `data.pt` dataloader
  state. No missing shard.
- **Resume mechanism:** `trainer.resume_mode=resume_path` +
  `trainer.resume_from_path=<step60 dir>` — VERIFIED via source read of
  `ray_trainer.py`'s `_load_checkpoint()` that this sets `self.global_steps = 60`
  directly, and that `FSDPCheckpointManager.load_checkpoint()` genuinely restores
  optimizer + LR-scheduler + RNG state (not weights-only).

## D. Output collision risk

- E1-0 experiment name: `DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu` — VERIFIED does
  not exist under `experiments/` prior to launch (checked immediately before writing
  this report).
- E1-0 log file: `verl/logs/DAPO-GAP3B-MHQA-Agent-E10-step60to70-4gpu.log` — unique,
  does not collide with the original run's log filename.
- E1-0 RUN_DIR: `E1-0/runs/20260910_step60to70_e10/` — newly created this run, no
  pre-existing E1-0 content overwritten (this is the first and only run under
  `E1-0/runs/`).
- `trainer.logger=['console']` and `WANDB_MODE=disabled` — VERIFIED no experiment
  tracker (W&B/SwanLab) run will be created or collide.
- Original experiment dir `experiments/DAPO-GAP3B-MHQA-Agent-4gpu/` (containing
  step60/120/180/200) is never written to by E1-0; E1-0 only *reads* step60 from it.

## Wiki service (see wiki/wiki_status.json, wiki/wiki_health_check.txt for full detail)

- Reused existing healthy server (pid 1016459), bound to `127.0.0.1:8008` only.
- Functional check with a real query ("what is the capital of France") returned
  genuine Wikipedia passages, HTTP 200 — stronger than a port-LISTEN-only check.
- `environment.sh`'s `WIKI_RAG_SERVER_URL` (derived from `hostname -I`, LAN IP
  `10.29.128.7`) is unreachable for this server (`HTTP_CODE=000`) — E1-0 launch
  script overrides this to `127.0.0.1` post-`environment.sh`-source, following the
  exact precedent already in `gen_bs_supervisor.py:251`. `environment.sh` itself was
  NOT modified.
- `WIKI_SERVER_AUTOSTART=0` set in the E1-0 launch script — guarantees no duplicate
  server is started and the existing healthy server is never at risk of being killed
  by E1-0.

## Config validation

A full Hydra `--cfg job --resolve` dry run of the actual `run_e10_step60to70.sh`
command (GPU-free) was executed and saved to `configs/resolved_training_config.yaml`.
Confirmed: `resume_mode=resume_path`, `resume_from_path` = step60 dir,
`total_training_steps=70`, `save_freq=10`, `test_freq=10`,
`reward_manager=e10_batch_audit`, `custom_reward_function.train_path/train_name` point
at the E1-0 scorer, `rollout.n=8`, `algorithm.filter_groups.enable=true`,
`adv_estimator=grpo` — all as intended, with every other original parameter
(LR, batch sizes, clip ratios, multi_turn config, tool_config_path) unchanged from
`train_dapo_mhqa_agent_wiki.sh`.

## Registration smoke test

End-to-end synthetic-DataProto test of `E10BatchAuditRewardManager` +
`compute_score_em_batch_e10` (4 rows, 1 uid group, mixed correct/incorrect, mixed
search-tag presence, one row with structured `detailed_tool_metrics` and one with
fallback `None`) passed: reward_tensor values correct, `reward_extra_info` keys
`['score','em']`, `acc` tensor correct, rollout/group audit JSONL fields all correct
including `fallback_degraded` distinction. Reward identity
(`compute_score_em_batch_e10` output == direct `compute_score_em` output) confirmed
byte-identical on the same synthetic inputs.

## PRE-FLIGHT VERDICT: PASS

All checks in sections A-D, Wiki health, and config validation pass. No BLOCKER
condition was encountered. Proceeding to launch per task section 15.

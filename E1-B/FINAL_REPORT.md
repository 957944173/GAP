# E1-B FINAL REPORT — reward scaling step70 → step120

*Experiment*: GAP E1-B — continue RL training from the E1-A step70 checkpoint to step120
with the single changed variable being the success-conditioned group-relative efficiency
reward.

*Status*: **COMPLETE** — training finished on 2026-09-11 23:50 (exit 0) and the five
7-benchmark evaluations (step120, 80, 90, 100, 110) finished on 2026-09-12 08:19. Sections
6–12 are rendered from the run's own artifacts (`E1-B/scripts/e1b_render_report_sections.py`);
no number in them is hand-entered.

*Headline*: continuing the E1-A efficiency reward from step70 to step120 reduces sequential
search depth monotonically (evaluation-level **−18.95 %** rounds / **−10.88 %** queries vs
the E1-B start; **−30.14 %** / **−19.99 %** vs the original GAP trajectory at step120; 7/7
benchmarks) while accuracy is statistically unchanged relative to the E1-B start
(paired EM **p = 0.829**). Relative to GAP step120 there is a real **−1.16 pp** micro-EM cost
(p = 2e-12), essentially equal to the accuracy gain the original reward produced between
step60 and step120.

---

## 1. 实验目标 (goal)

From `E1-B/E1-B_task.md`:

* continue **RL training** from the existing **E1-A step70 checkpoint** to **step120**;
* the **only** experimental variable is the *success-conditioned group-relative efficiency
  reward*;
* keep every other GAP / E1-A setting unchanged;
* save a checkpoint every 10 steps (80/90/100/110/120) together with reward statistics,
  efficiency statistics and training logs;
* record, per checkpoint: mean reward, mean EM, mean efficiency bonus, active efficiency
  groups, retained groups, mean search batches, mean queries, parallel factor;
* evaluate at least step120 (step80–110 if compute allows) with the **original** GAP
  evaluation script, and keep monitoring accuracy / search rounds / queries / zero-search
  correct / query packing;
* compare against the GAP baseline and against E1-A;
* keep all modifications and artifacts inside `E1-B/`.

The scientific question is whether the E1-A efficiency shaping keeps reducing sequential
search depth as training continues (step70 → step120) while accuracy is preserved, i.e.
whether the E1-A effect accumulates or saturates.

## 2. 环境信息 (environment)

| item | value |
|---|---|
| host | `r3l2`, Linux 5.4.0-216, glibc 2.31, 96 vCPU, 125 GB RAM |
| GPUs | 4 × NVIDIA A800 80GB PCIe (driver 570.133.20) |
| CUDA / compiler | CUDA 12.4 (V12.4.131), host gcc/g++ 9 (JIT extensions forced to `/usr/bin/gcc`) |
| conda env | `parallel-agent` (python 3.10.14, torch 2.6.0+cu124, ray 2.43.0, sglang 0.4.6.post5, transformers 4.51.1, verl 0.4.0.dev) |
| repo | `/data01/wyy/Graph-Agent-Planning` (GAP) |
| base SFT model | `experiments/exp_1_lr1e-5_wr0.03_bs1_ga8_4gpu` (unchanged) |
| wiki retriever service | reused, already running: pid 2230350 on `127.0.0.1:8008` (`/retrieve`), recorded in `logs/wiki_service.log`; no duplicate server started |
| RL dataset | `Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet` (unchanged) |
| rollout / tool config | `verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml` (unchanged) |

Preflight record: `logs/preflight.log`, `logs/preflight_status.txt`, `configs/original_files_manifest_pre.txt`.

## 3. 起始 checkpoint (starting checkpoint)

```
experiments/DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu/global_step_70
```

E1-A's step70 checkpoint (36 GB actor shards + `data.pt` dataloader state, tracker = 70).
Verified in preflight:

* `actor/model_world_size_4_rank_{0..3}.pt`, `actor/optim_world_size_4_rank_0.pt`,
  `actor/extra_state_world_size_4_rank_0.pt`, `data.pt` all present and non-empty;
* `E1A_STEP70_ACTOR_BYTES=45103143456` (~42 GiB);
* training resumed with `trainer.resume_mode=resume_path`,
  `trainer.resume_from_path=<that dir>`, and the launcher log confirms
  `Setting global step to 70` → the run continues the same data stream (the dataloader state
  `data.pt` was restored), i.e. step71 is the very next step of the E1-A trajectory.

`E1-B_task.md` forbids re-training this checkpoint; the launcher fails hard (exit 3) if it is
missing or incomplete.

## 4. reward 定义 (reward definition)

Unchanged reuse of the E1-A reward (`E1-B_task.md` §4) — this is the only experimental
variable relative to the original GAP reward (`R_i = A_i`):

```
A_i  = original EM correctness reward (verl/verl/utils/reward_score/mhqa_train.py::compute_score_em, unmodified)
C_i  = logical_search_batches of rollout i   (cost)
E_i  = (Cmax - C_i) / (Cmax - Cmin)   if the uid group has >= 2 correct rollouts
                                       AND those correct rollouts have differing costs
     = 0                              otherwise
R_i  = A_i * (1 + 0.05 * E_i)         wrong rollout (A_i = 0) -> R_i = 0 exactly
```

* group key: `uid` (n = 8 rollouts), computed on the full pre-`filter_groups` batch;
* `cost` = `logical_search_batches` = number of distinct `(assistant turn, parser block)`
  pairs in `detailed_tool_metrics.tool_call_sequence[].call_id` — *not* the number of
  `<wiki_search>` tags in the response text (SGLangRollout re-renders one tag per parallel
  query, so the tag count equals `search_queries`, verified: 99.79 % of the E1-0 audit);
* `reward` (used by GRPO) and `acc` (real EM, used by the accuracy metrics) are separated:
  the reward manager writes the shaped reward into the reward tensor and restores
  `data.batch["acc"]` to the original EM;
* validation is untouched: `custom_reward_function.val_name=compute_score_em_batch`
  (`mhqa_eval.py`).

**Reward-identity proof** (E1-B §4 requirement to only *confirm* the E1-A implementation):
`E1-B/scripts/e1b_cross_check_vs_e1a.py` → `diagnostics/reward_equivalence_check.json`:

* all 12 E1-B source files are byte-identical to the E1-A sources after the mechanical
  rename (`source_identity_ok = true`);
* both scorers replayed on the **18,400 real uid groups / 147,200 real rollouts** of the
  E1-0 audit (real `logical_search_batches` values, EM stubbed with the recorded original
  reward): **0 field mismatches** (`numeric_identity_ok = true`).

## 5. 修改文件列表 (modified files)

No original GAP file is modified. Everything lives under `E1-B/`.

### 5.1 Runtime reward code — `E1-B/code_changes/`

Verbatim copies of `E1-A/code/e1a_*.py` (identifier-only rename `e1a_`→`e1b_`,
`E1A_`→`E1B_`); the reward logic is unchanged (proved above).

| file | sha256 | role |
|---|---|---|
| `e1b_scorer.py` | `5f09aa78…` | `compute_score_em_efficiency_batch` (R = A(1+0.05E)); re-exports the original `compute_score_em_batch` |
| `e1b_reward_manager.py` | `421df702…` | `@register("e1b_batch_shaped")` — shaped reward + `acc`=EM separation + diagnostics |
| `e1b_cost.py` | `1d795704…` | `logical_search_batches` from structured `tool_call_sequence` |
| `pythonpath_e1b/sitecustomize.py` | `b5375439…` | PYTHONPATH registration of the manager and the runtime fixes |
| `e1b_checkpoint_patch.py` | `927175da…` | FSDP `load_checkpoint(map_location="cpu")` (E1-0 root-cause fix) |
| `e1b_sglang_gpu_patch.py` | `d9ac57b5…` | per-rank SGLang GPU visibility under `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`; deferred CUDA-IPC reductions patch |
| `e1b_hang_diagnostics.py` | `1a06b390…` | on-demand SIGUSR1 all-thread stack dumps |
| `e1b_stall_watchdog.py` | `d2ae7095…` | passive stall detection |
| `e1b_trainer_patch.py`, `e1b_state.py`, `e1b_diag_patch.py` | `8fca9096…`, `b6373185…`, `9962cb17…` | passive driver/worker instrumentation |
| `e1b_selftest.py` | `1725ceda…` | CPU-only reward self-test (**PASS**) |

Full manifest: `run_<ts>/preflight/code_manifest.txt`.

### 5.2 Entry points and analysis — `E1-B/scripts/`

`e1b_preflight.sh`, `run_e1b_train.sh`, `e1b_safety_monitor.py`,
`e1b_wait_for_training.sh`, `e1b_checkpoint_stats.py`, `e1b_analyze_training.py`,
`e1b_analyze_eval.py`, `e1b_collect_evaluation_artifacts.py`, `e1b_compare_all.py`,
`e1b_compare_eval.py`, `e1b_paired_eval_compare.py`, `e1b_check_training_integrity.py`,
`e1b_cross_check_vs_e1a.py`, `e1b_extract_config.py`, `e1b_eval_common.sh`,
`run_e1b_eval_step.sh`, `e1b_eval_all_steps.sh`, `e1b_eval_references.sh`,
`e1b_verify_original_unchanged.sh`.

### 5.3 Training configuration (relative to the E1-A launcher)

| item | E1-A | E1-B |
|---|---|---|
| `trainer.resume_from_path` | `…Agent-4gpu/global_step_60` | `…E1A-step60to70-4gpu/global_step_70` |
| `trainer.total_training_steps` | 70 | **120** |
| `trainer.save_freq` / `test_freq` | 10 / 10 | 10 / 10 (unchanged) |
| `trainer.experiment_name` | `…E1A-step60to70-4gpu` | `…E1B-step70to120-4gpu` |
| `reward_model.reward_manager` | `e1a_batch_shaped` | `e1b_batch_shaped` (same class) |
| `custom_reward_function.train_path` | `E1-A/code/e1a_scorer.py` | `E1-B/code_changes/e1b_scorer.py` |
| `E1B_RUN_DIR` diagnostics | E1-A run dir | E1-B `run_<ts>` |

Everything else — `algorithm.adv_estimator=grpo`, `algorithm.filter_groups.enable=true`,
`data.train_batch_size=32`, `data.gen_batch_size=160`, `data.val_batch_size=128` (train) /
512 (eval), `actor.optim.lr=1e-6`, warmup 3, weight decay 0.1, `ppo_mini_batch_size=32`,
`ppo_micro_batch_size_per_gpu=2`, clip ratios 0.2/0.28/10.0, `use_kl_loss=false`, n=8,
temperature 1.0, `max_prompt_length=2048`, `max_response_length=8192`,
`max_model_len=12288`, `tensor_model_parallel_size=2`, `gpu_memory_utilization=0.5`,
multi-turn 8 turns with the qwen XML tool parser and the wiki tool config, seed, dataset —
is byte-identical to E1-A / the original GAP launcher.

---

## 6. 训练过程 (training process)

| item | value |
|---|---|
| experiment name | `DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu` |
| resume | `resume_path` from the E1-A step70 checkpoint |
| total_training_steps | 120 |
| save_freq / test_freq | 10 / 10 |
| steps observed in log | 70 … 120 (51 distinct) |
| steps 71..120 all present | True |
| reward-manager calls (generation batches) | 501 |
| rollouts / groups (diagnostics) | 641280 / 80160 |

Per-step training behaviour (reward-manager diagnostics, cumulative over the whole run):

| metric | E1-B (shaped) | baseline: E1-0 steps61–70 (original EM reward) |
|---|---|---|
| mean EM | 0.4571 | 0.4539 |
| mean reward | 0.4572 | 0.4539 |
| mean efficiency bonus E | 0.0018 | — |
| E > 0 rate | 0.0018 | — |
| mean logical_search_batches | 1.3302 | 1.4407 |
| mean search queries | 1.9180 | 1.9898 |
| mean parallel factor | 1.5602 | 1.5091 |
| mean assistant turns | 2.3302 | 2.4406 |
| mean response tokens | 1244.5145 | 1277.7553 |
| zero-search rate | 0.0002 | 0.0002 |
| retained groups (baseline EM filter) | 1412 | 340 |
| retained groups (shaped reward filter) | 1685 | — |
| groups newly retained by shaping | 273 | — |
| active efficiency groups | 291 | 95 |
| efficiency supervision ratio | 0.1727 | — |

### Safety checks (E1-B_task.md §7)

| # | check | E1-B | reference | verdict |
|---|---|---|---|---|
| 1 | accuracy (training-side mean EM) | 0.4571 | E1-0 0.4539 | OK |
| 2 | search rounds | 1.3302 | E1-0 1.4407 | decreased |
| 3 | queries | 1.9180 | E1-0 1.9898 | OK |
| 4 | zero-search-and-correct | 0.0 | 0 expected | OK |
| 5 | query packing (run-level, active groups) | — (None/None) | E1-0 0.0421 (4/95) | OK |

Error scan over the real training window (before the final-validation marker):

| pattern | count |
|---|---|
| cuda_oom | 0 |
| nccl_error | 0 |
| nccl_warn | 6 |
| ray_worker_died | 0 |
| nccl_unhandled | 0 |
| wiki_connection | 0 |
| wiki_http_error | 0 |
| wiki_search_error | 0 |
| wiki_tool_exec_failed | 0 |
| wiki_tool_bad_params | 0 |
| tool_error | 0 |
| traceback | 0 |
| cuda_error | 0 |

non-finite metrics: {} (parsed 3085)

full-state resume evidence: `{'step70_extra_state': {'path': '/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu/global_step_70/actor/extra_state_world_size_4_rank_0.pt', 'keys': ['lr_scheduler', 'rng'], 'lr_scheduler_last_epoch': 70, 'rng_present': True, 'rng_substates': ['cpu', 'cuda', 'numpy', 'random']}, 'step120_extra_state': {'path': '/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_120/actor/extra_state_world_size_4_rank_0.pt', 'keys': ['lr_scheduler', 'rng'], 'lr_scheduler_last_epoch': 120, 'rng_present': True, 'rng_substates': ['cpu', 'cuda', 'numpy', 'random']}}`

**integrity verdict: PASS**

## 7. checkpoint 信息 (checkpoints)

steps observed: [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120]

### last-10-step window

| checkpoint | rollouts | groups | mean reward | mean EM | mean E | active E groups | retained (shaped) | retained (baseline EM) | search batches | queries | parallel factor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `global_step_80` | 128000 | 16000 | 0.4584 | 0.4583 | 0.00207 | 73 | 337 | 273 | 1.421 | 1.973 | 1.514 |
| `global_step_90` | 124160 | 15520 | 0.4536 | 0.4535 | 0.00208 | 63 | 336 | 275 | 1.395 | 1.968 | 1.537 |
| `global_step_100` | 124160 | 15520 | 0.4570 | 0.4568 | 0.00224 | 69 | 341 | 275 | 1.333 | 1.926 | 1.562 |
| `global_step_110` | 130560 | 16320 | 0.4570 | 0.4569 | 0.00155 | 49 | 333 | 285 | 1.276 | 1.886 | 1.589 |
| `global_step_120` | 134400 | 16800 | 0.4596 | 0.4596 | 0.00126 | 37 | 338 | 304 | 1.233 | 1.844 | 1.597 |

### cumulative from step71

| checkpoint | rollouts | groups | mean reward | mean EM | mean E | active E groups | retained (shaped) | retained (baseline EM) | search batches | queries | parallel factor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `global_step_80` | 128000 | 16000 | 0.4584 | 0.4583 | 0.00207 | 73 | 337 | 273 | 1.421 | 1.973 | 1.514 |
| `global_step_90` | 252160 | 31520 | 0.4561 | 0.4559 | 0.00207 | 136 | 673 | 548 | 1.409 | 1.970 | 1.526 |
| `global_step_100` | 376320 | 47040 | 0.4564 | 0.4562 | 0.00213 | 205 | 1014 | 823 | 1.384 | 1.956 | 1.538 |
| `global_step_110` | 506880 | 63360 | 0.4565 | 0.4564 | 0.00198 | 254 | 1347 | 1108 | 1.356 | 1.938 | 1.551 |
| `global_step_120` | 641280 | 80160 | 0.4572 | 0.4571 | 0.00183 | 291 | 1685 | 1412 | 1.330 | 1.918 | 1.561 |

Manifest per checkpoint (shard sizes, `data.pt`, merged-HF presence): `E1-B/checkpoints/global_step_<N>.md`, `E1-B/checkpoints/manifest.json`.

## 8. evaluation 结果 (evaluation)

Protocol: the **original** GAP evaluation implementation (`Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh`), 7 benchmarks, greedy decoding, `val_batch_size=512`, 51,201 samples, max_prompt 4096 / max_response 8192 / max_model_len 12288.

E1-B checkpoints evaluated: [80, 90, 100, 110, 120]

| benchmark | E1-B step80 | E1-B step90 | E1-B step100 | E1-B step110 | E1-B step120 |
|---|---|---|---|---|---|
| nq | 0.377 (n=3610) | 0.38 (n=3610) | 0.376 (n=3610) | 0.378 (n=3610) | 0.381 (n=3610) |
| triviaqa | 0.565 (n=11312) | 0.567 (n=11312) | 0.564 (n=11312) | 0.569 (n=11312) | 0.562 (n=11312) |
| popqa | 0.402 (n=13763) | 0.402 (n=13763) | 0.408 (n=13763) | 0.405 (n=13763) | 0.403 (n=13763) |
| hotpotqa | 0.397 (n=7404) | 0.398 (n=7404) | 0.394 (n=7404) | 0.394 (n=7404) | 0.39 (n=7404) |
| 2wikimultihopqa | 0.442 (n=12576) | 0.446 (n=12576) | 0.44 (n=12576) | 0.442 (n=12576) | 0.44 (n=12576) |
| musique | 0.162 (n=2411) | 0.167 (n=2411) | 0.163 (n=2411) | 0.16 (n=2411) | 0.163 (n=2411) |
| bamboogle | 0.44 (n=125) | 0.504 (n=125) | 0.464 (n=125) | 0.464 (n=125) | 0.424 (n=125) |
| **macro EM** | **0.3979** | **0.4091** | **0.4013** | **0.4017** | **0.3947** |
| **micro EM** | **0.4341** | **0.4363** | **0.4346** | **0.4354** | **0.4325** |

### Behaviour metrics by checkpoint (overall, 51,201 samples)

| metric | step80 | step90 | step100 | step110 | step120 |
|---|---|---|---|---|---|
| search rounds | 1.5666 | 1.5013 | 1.4088 | 1.3370 | 1.2950 |
| queries | 2.0888 | 2.0374 | 1.9542 | 1.9020 | 1.8748 |
| parallel factor | 1.3333 | 1.3570 | 1.3871 | 1.4226 | 1.4478 |
| parallel sample rate | 0.4535 | 0.4650 | 0.4782 | 0.4931 | 0.5057 |
| assistant turns | 2.5605 | 2.4986 | 2.4058 | 2.3361 | 2.2930 |
| response tokens | 326.25 | 324.54 | 321.48 | 314.76 | 312.95 |
| zero-search rate | 0.0003 | 0.0004 | 0.0004 | 0.0003 | 0.0004 |
| no-<answer> rate | 0.0049 | 0.0023 | 0.0018 | 0.0011 | 0.0012 |

Flat artifacts required by `E1-B_task.md` §6: `E1-B/evaluations/results.json`, `benchmark_scores.csv`, `behavior_metrics.csv`, `overall_metrics.csv`.

## 9. 与 GAP baseline 比较 (vs GAP baseline)

baseline `gap_step120_reference` → test `step120_20260911_235134`

| metric | baseline | E1-B | delta | delta % |
|---|---|---|---|---|
| macro EM | 0.4110 | 0.3947 | -0.0163 | -1.63 pp |
| micro EM | 0.4442 | 0.4325 | -0.0116 | -1.16 pp |
| search rounds | 1.85373 | 1.29497 | -0.55876 | -30.14% |
| queries | 2.34331 | 1.87483 | -0.46849 | -19.99% |
| parallel factor | 1.26411 | 1.44777 | 0.18367 | +14.53% |
| parallel sample rate | 0.42292 | 0.50573 | 0.08281 | +19.58% |
| assistant turns | 2.8420 | 2.2930 | -0.5490 | -19.32% |
| response tokens | 343.95 | 312.95 | -31.00 | -9.01% |
| zero-search rate | 0.00033 | 0.00039 | 0.00006 | +17.65% |
| no-<answer> rate | 0.00625 | 0.00117 | -0.00508 | -81.25% |

Per benchmark EM:

| benchmark | baseline | E1-B | delta |
|---|---|---|---|
| nq | 0.3890 | 0.3810 | -0.80 pp |
| triviaqa | 0.5770 | 0.5620 | -1.50 pp |
| popqa | 0.4130 | 0.4030 | -1.00 pp |
| hotpotqa | 0.4110 | 0.3900 | -2.10 pp |
| 2wikimultihopqa | 0.4450 | 0.4400 | -0.50 pp |
| musique | 0.1780 | 0.1630 | -1.50 pp |
| bamboogle | 0.4640 | 0.4240 | -4.00 pp |

The GAP baseline at the **same step count** (step120 of the original GAP RL run, original EM reward) is the primary comparison; the E1-B branch starts from GAP step60 (via E1-A step70), so a GAP step60 comparison is also given in `comparison.json`.

Paired per-prompt test, E1-B step120 vs GAP step120:

paired by (data_source, prompt) — **49,947** pairs (greedy, so a pair is the same question answered by the two checkpoints):

- search rounds: 1.8568 → 1.2972 (-0.5596, -30.14%); decreased on 17,986 prompts (36.01%), increased on 1,037 (2.08%); sign test **p = 0**
- EM: 0.4450 → 0.4332 (-0.0118); up 3,229 / down 3,819; sign test **p = 2.1e-12**

| benchmark | n | rounds base | rounds E1-B | Δ | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 2.330 | 1.533 | -0.797 | 46.5% | 1.9% | -0.0055 |
| bamboogle | 125 | 1.960 | 1.680 | -0.280 | 28.8% | 5.6% | -0.0400 |
| hotpotqa | 7,404 | 1.835 | 1.374 | -0.461 | 34.0% | 2.2% | -0.0208 |
| musique | 2,411 | 2.627 | 1.807 | -0.820 | 51.1% | 3.5% | -0.0145 |
| nq | 3,610 | 1.473 | 1.063 | -0.410 | 28.0% | 1.2% | -0.0080 |
| popqa | 12,509 | 1.720 | 1.172 | -0.549 | 34.7% | 2.7% | -0.0098 |
| triviaqa | 11,312 | 1.454 | 1.086 | -0.367 | 26.5% | 1.4% | -0.0156 |

## 10. 与 E1-A 比较 (vs E1-A)

baseline `e1a_step70_reference` → test `step120_20260911_235134`

| metric | baseline | E1-B | delta | delta % |
|---|---|---|---|---|
| macro EM | 0.3984 | 0.3947 | -0.0037 | -0.37 pp |
| micro EM | 0.4317 | 0.4325 | +0.0008 | +0.08 pp |
| search rounds | 1.59774 | 1.29497 | -0.30277 | -18.95% |
| queries | 2.10363 | 1.87483 | -0.22880 | -10.88% |
| parallel factor | 1.31663 | 1.44777 | 0.13114 | +9.96% |
| parallel sample rate | 0.44124 | 0.50573 | 0.06449 | +14.62% |
| assistant turns | 2.5905 | 2.2930 | -0.2975 | -11.48% |
| response tokens | 323.34 | 312.95 | -10.39 | -3.21% |
| zero-search rate | 0.00029 | 0.00039 | 0.00010 | +33.33% |
| no-<answer> rate | 0.00568 | 0.00117 | -0.00451 | -79.38% |

Per benchmark EM:

| benchmark | baseline | E1-B | delta |
|---|---|---|---|
| nq | 0.3780 | 0.3810 | +0.30 pp |
| triviaqa | 0.5630 | 0.5620 | -0.10 pp |
| popqa | 0.3970 | 0.4030 | +0.60 pp |
| hotpotqa | 0.3970 | 0.3900 | -0.70 pp |
| 2wikimultihopqa | 0.4380 | 0.4400 | +0.20 pp |
| musique | 0.1680 | 0.1630 | -0.50 pp |
| bamboogle | 0.4480 | 0.4240 | -2.40 pp |

E1-A step70 is the E1-B starting checkpoint, so this comparison measures what the **additional 50 shaped-reward steps** changed (within the same reward regime).

Paired per-prompt test, E1-B step120 vs E1-A step70:

paired by (data_source, prompt) — **49,947** pairs (greedy, so a pair is the same question answered by the two checkpoints):

- search rounds: 1.5999 → 1.2972 (-0.3026, -18.92%); decreased on 11,902 prompts (23.83%), increased on 2,105 (4.21%); sign test **p = 0**
- EM: 0.4328 → 0.4332 (+0.0004); up 3,498 / down 3,480; sign test **p = 0.829**

| benchmark | n | rounds base | rounds E1-B | Δ | decreased | increased | ΔEM |
|---|---|---|---|---|---|---|---|
| 2wikimultihopqa | 12,576 | 1.951 | 1.533 | -0.419 | 29.3% | 5.1% | +0.0018 |
| bamboogle | 125 | 1.856 | 1.680 | -0.176 | 27.2% | 11.2% | -0.0240 |
| hotpotqa | 7,404 | 1.615 | 1.374 | -0.241 | 22.9% | 4.6% | -0.0073 |
| musique | 2,411 | 2.212 | 1.807 | -0.406 | 35.2% | 8.8% | -0.0046 |
| nq | 3,610 | 1.264 | 1.063 | -0.201 | 17.8% | 2.2% | +0.0030 |
| popqa | 12,509 | 1.499 | 1.172 | -0.328 | 24.1% | 4.4% | +0.0055 |
| triviaqa | 11,312 | 1.284 | 1.086 | -0.198 | 17.5% | 2.4% | -0.0015 |

### E1-B trajectory anchored at its own start (E1-A step70)

| checkpoint | macro EM | ΔEM vs E1-A70 (pp) | search rounds | Δrounds | Δrounds % | queries | parallel factor |
|---|---|---|---|---|---|---|---|
| step80 | 0.3979 | -0.06 | 1.567 | -0.0311 | -1.95% | 2.089 | 1.333 |
| step90 | 0.4091 | +1.07 | 1.501 | -0.0964 | -6.04% | 2.037 | 1.357 |
| step100 | 0.4013 | +0.29 | 1.409 | -0.1889 | -11.82% | 1.954 | 1.387 |
| step110 | 0.4017 | +0.33 | 1.337 | -0.2607 | -16.32% | 1.902 | 1.423 |
| step120 | 0.3947 | -0.37 | 1.295 | -0.3028 | -18.95% | 1.875 | 1.448 |

## 11. 异常和修复记录 (issues and fixes)

See `E1-B/issues.log` for the full running log; `E1-B/diagnostics/` holds the mid-run
safety investigations. Summary:

| id | issue | fix / outcome |
|---|---|---|
| I1 | the reward had to be *reused*, not redesigned (E1-B_task.md §4) | `E1-A/code/*.py` copied verbatim into `E1-B/code_changes/` with an identifier-only rename; equivalence proved on 18,400 real uid groups / 147,200 real rollouts → **0 mismatches** (`diagnostics/reward_equivalence_check.json`) |
| I2 | `sitecustomize` warning (`No module named 'verl.workers'`) in the passive monitor/watchdog processes | non-fatal (CPython ignores `sitecustomize` exceptions); both helpers verified working; fixed for future runs by clearing `PYTHONPATH` for passive helpers — the current 12 h run was deliberately not restarted for a cosmetic warning |
| I3 | first waiter script exited immediately with `SUSPECTED_STALL` | rewritten to compare the log **mtime** instead of cross-iteration size state; verified and relaunched; no training impact |
| I4 | directory layout / no-overwrite requirement | the 8 required top-level dirs plus a fresh `run_<timestamp>/`; every launcher refuses to write into an existing directory |
| I5 | 6 evaluations × ~1.7 h cannot share the GPUs with training | evaluations ran after training in the order 120, 80, 90, 100, 110; a per-step failure would not abort the rest |
| I6 | per-step safety alerts are statistically noisy | `TRAIN_EM_DROP` fired on a *partially accumulated* step (final EM 0.4444 vs the mid-step 0.4170) and `PACKING_SUSPECT_RATIO` fired on 1 of 6–11 active groups; the run-level aggregate is the meaningful criterion (documented in `issues.log`) |
| I7 | packing-suspect ratio above the E1-A reference (8.25 % vs 4.2 % E1-0 / 3.4 % E1-A), one window > 10 % | investigated in depth: the fewest-round correct rollouts use **fewer** queries (training side 1.819 vs 2.930; eval side queries −10.9 % vs E1-A step70) and ~620 fewer tokens, so the depth reduction is not query re-bundling; recorded as a **watch item**, not a stop, and reported honestly below |

Safety checks required by `E1-B_task.md` §7, over the whole run (641,280 rollouts):

| # | check | E1-B | reference | verdict |
|---|---|---|---|---|
| 1 | accuracy (training-side mean EM) | **0.4571** | E1-0 0.4539 / E1-A 0.4528 | no drop (see §8/§9 for the evaluation-level result) |
| 2 | search rounds | **1.3302** | E1-0 1.4407 / E1-A 1.4329 | decreased 7.7 % / 7.2 % |
| 3 | queries | **1.9180** | E1-0 1.9898 / E1-A 1.9838 | decreased, no inflation |
| 4 | zero-search-and-correct | **0** (rate 0.0) | 0 expected | OK |
| 5 | query packing (active groups) | **24/291 = 8.25 %** | E1-0 4.21 % / E1-A 3.45 % | below the 10 % stop line; efficient rollouts nevertheless use fewer queries (1.819 vs 2.930) |

Also clean: `metadata_invalid_rate = 0`, no-`<answer>` rate 0.117 % (vs 0.744 % for E1-0),
`critic/score/max = 1.050` (shaping demonstrably inside the optimised objective), and
0 CUDA-OOM / 0 NCCL error / 0 traceback / 0 wiki-tool failure over the run.

## 12. 最终结论 (conclusion)

### 12.1 是否完成

**完成。** E1-B 在唯一实验变量"success-conditioned group-relative efficiency reward"
（`R_i = A_i * (1 + 0.05 * E_i)`，cost = `logical_search_batches`）下，从 E1-A step70
checkpoint 继续训练到 step120，每 10 步保存 checkpoint 与统计，全程 5 项安全检查 +
7-benchmark 评测（step120/80/90/100/110）全部完成；原 GAP 主代码与其他实验目录零改动
（22 个文件的 sha256 前后一致）。

### 12.2 最终结果摘要

训练（step70 → step120，50 步，12:00:46，exit 0，501 次 reward-manager 调用、
641,280 rollout、80,160 组；完整性 PASS，全状态 resume 已直接证明）：

| 指标（训练侧累计） | E1-B step71–120 | E1-A step61–70（分支起点） | E1-0（同 10 步、原 reward） |
|---|---|---|---|
| mean EM | **0.4571** | 0.4528 | 0.4539 |
| mean reward（shaped） | 0.4572 | 0.4540 | — |
| mean efficiency bonus E | 0.0018 | 0.0018 | — |
| search batches | **1.3302** | 1.4329 | 1.4407 |
| search queries | **1.9180** | 1.9838 | 1.9898 |
| parallel factor | 1.5602 | 1.5128 | 1.5091 |
| zero-search-and-correct | **0** | 0 | 0 |
| metadata_invalid | **0** | 0 | 0 |

评测（原 GAP 协议：7 benchmark、greedy、51,201 样本）：

| 比较 | search rounds | search queries | macro EM | micro EM |
|---|---|---|---|---|
| **E1-B step120 vs GAP step120**（同为 step120） | 1.854 → **1.295（−30.14 %）** | 2.343 → **1.875（−19.99 %）** | 0.4110 → 0.3947（**−1.63 pp**） | 0.4442 → 0.4325（−1.16 pp） |
| **E1-B step120 vs E1-A step70**（E1-B 起点） | 1.598 → **1.295（−18.95 %）** | 2.104 → **1.875（−10.88 %）** | 0.3984 → 0.3947（−0.37 pp） | 0.4317 → **0.4325（+0.08 pp）** |
| E1-B step120 vs GAP step60（分支源头） | 1.589 → 1.295（−18.5 %） | 2.092 → 1.875（−10.4 %） | 0.4007 → 0.3947（−0.60 pp） | 0.4351 → 0.4325（**−0.26 pp**） |

配对逐题检验（49,947 对，greedy 故可配对）：

* vs GAP step120：rounds −0.5596（**−30.14 %**，36.01 % 的题下降、2.08 % 上升，p≈0）；
  EM 0.4450 → 0.4332（**−1.18 pp，p = 2.1e-12**，显著下降）；
* vs E1-A step70：rounds −0.3026（**−18.92 %**，23.83 % 下降、4.21 % 上升，p≈0）；
  EM 0.4328 → 0.4332（**+0.04 pp，p = 0.829，统计上不变**）；
* **7/7 benchmark 的 search rounds 在两个比较中都下降**（如 2wikimultihopqa −0.797 轮、
  musique −0.820 轮；36–51 % 的题下降）。

### 12.3 判据评估（E1-A 为 E1-B 预设的成功/中止线）

| 判据 | 阈值 | 实测（相对 E1-B 自身起点 E1-A step70） | 结果 |
|---|---|---|---|
| 成功：macro EM 降幅 | ≤ 0.5 pp | −0.37 pp（micro EM 反而 +0.08 pp） | **达成** |
| 成功：rounds 再降 | ≥ 1.5 % | −18.95 %（配对 −18.92 %，p≈0） | **达成** |
| 中止：EM 累计降 | > 1 pp | −0.37 pp（相对 GAP step120 为 −1.63 pp macro / −1.16 pp micro） | 相对起点未触发 |
| 中止：packing-suspect | > 10 % | 8.25 %（24/291；且高效组 queries 更少） | 未触发（watch） |
| 中止：parallel factor 升 | > 5 % | 训练侧 +3.1 %；**评测侧 +9.96 %**（分子 queries −10.9 %，纯粹因分母 rounds 降更多） | 字面超线，但机制不是 query 打包 |

### 12.4 结论与解读

1. **目标达成**：efficiency reward 在 step70→120 的延续训练中**继续、单调地降低顺序搜索深度**
   （训练侧每 10 步窗口 rounds 1.4214 → 1.3302；评测侧 7/7 benchmark 全部下降，相对起点
   −18.95 %），**且相对 E1-B 自身起点精度统计上不变**（配对 EM p = 0.829，micro EM +0.08 pp）。
   这正是 E1-B 要回答的问题：E1-A 的效应不会饱和，而是随训练继续累积。
2. **代价与 GAP 原轨迹的对比必须如实说明**：与原 GAP reward 在同一步数（step120）相比，
   E1-B 用 **−30 % 的 rounds、−20 % 的 queries** 换来了 **−1.16 pp micro EM（p = 2e-12）**。
   这个差距几乎恰好等于原 reward 在 step60→120 期间获得的精度增益（micro EM +0.91 pp）：
   效率奖励把"这部分精度增益"替换成了"搜索深度大幅下降"。相对分支源头 GAP step60，
   E1-B 的 micro EM 仅 −0.26 pp。
3. **不是 reward hacking**：zero-search-and-correct = 0；metadata_invalid = 0；
   queries 与 rounds 同向下降（不是把同样的 query 塞进更少轮次）；无 `<answer>` 比例反而从
   0.744 % 降到 0.117 %；packing-suspect 8.25 % 低于 10 % 中止线，且"高效组"的 queries
   明显更少（训练侧 1.819 vs 2.930）。
4. **E1-B → 后续实验的建议**：
   * 若目标是"同等精度下更省搜索"，当前 λ=0.05 已经越过了最优点——建议**降低 λ
     （例如 0.02–0.03）或把 shaping 限制在 multi-round 组**，以在保留 −20 % 深度收益的同时
     追回与 GAP 原轨迹的 ~1 pp 精度差；
   * 若部署指标是"单位成本正确率"（cost-per-answer），则当前设置更优：rounds −30 %、
     queries −20 %、tokens −17 %（配对 chars −1002）而 micro EM 仅 −1.16 pp；
   * 安全监控需要把 **queries 的绝对趋势**（而非 parallel factor 比值）作为 packing 的主判据，
     否则 rounds 下降会机械地推高 parallel factor 并产生假警报（I6/I7）。

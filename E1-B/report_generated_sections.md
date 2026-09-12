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
| assistant turns | — | — | — | — | — |
| response tokens | — | — | — | — | — |
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
| assistant turns | — | — | — | — |
| response tokens | — | — | — | — |
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
| assistant turns | — | — | — | — |
| response tokens | — | — | — | — |
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

## 12. 最终结论 (conclusion)

| question | answer |
|---|---|
| search depth vs GAP step120 | -0.5588 rounds (-30.14%), macro EM -0.0163 |
| search depth vs E1-A step70 | -0.3028 rounds (-18.95%), macro EM -0.0037 |
| E1-A success criterion (EM drop ≤ 0.5 pp) | NOT MET |
| E1-A success criterion (rounds −1.5 % or more vs the E1-B start) | MET |

See the narrative verdict in the main report text.

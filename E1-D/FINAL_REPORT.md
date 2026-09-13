# E1-D FINAL REPORT — Filter-Decoupled Reward (controlled causal / mechanism ablation)

*Status*: **COMPLETE** — D1 (step60→70) and D2 (step70→120) both finished with exit 0 and
integrity PASS; both evaluations (step70, step120) ran the original 7-benchmark protocol.
Every artifact lives under `E1-D/`.  No original GAP / E1-0 / E1-A / E1-B / E1-C file was
modified (23/23 protected files sha256-identical, see §8).

---

## 1. Executive Summary

E1-B (shaped reward **and** shaped filter) reduced search depth by 30 % versus the original
GAP trajectory at step120 but lost 1.16 pp micro-EM.  E1-C proved that the shaped filter
revives *all-correct but efficiency-variable* groups (`S_GAP ⊂ S_E1`) and that they displace
mixed-correctness groups inside the fixed-size optimizer batch.  E1-D removes exactly that
channel: **the reward is unchanged (still `R = A·(1+0.05·E)`), and only the filter input is
switched back to the original EM** (`algorithm.filter_groups.metric=em`, config-only,
`ray_trainer.py` untouched).

Results (step120, paired over 49,947 prompts):

* **E1-D vs GAP120** — the *objective channel alone*: rounds 1.857 → **1.723 (−6.99 %,
  p ≈ 1e-233)**, micro EM 0.4450 → **0.4392 (−0.57 pp, p = 2.8e-4)**.
* **E1-D vs E1-B120** — the *filter channel alone*: rounds 1.297 → **1.727 (+33.1 %, p ≈ 0)**,
  micro EM 0.4332 → **0.4392 (+0.61 pp, p = 2.5e-4)**.
* Therefore E1-B's total effect vs GAP120 decomposes **exactly additively** (paired means):
  rounds **−0.5596 = −0.1298 (objective, 23.2 %) + −0.4298 (filter, 76.8 %)**,
  EM **−1.18 pp = −0.57 pp (objective) + −0.61 pp (filter)**.

**Verdict: CASE A.**  With the original filter, accuracy returns to ≈GAP (micro EM −0.57 pp,
macro EM −0.51 pp) while the efficiency gain largely disappears (rounds only −7.0 % instead of
−30.1 %).  The filter-induced training-distribution shift was the **dominant** driver of
E1-B's depth reduction and roughly **half** of its accuracy cost, and most of the usable
efficiency signal came from the revived all-correct groups.

Mechanism, measured directly (D2, steps 71–120, 83,680 candidate groups, 523 generation
batches): the EM filter retained **1,679 groups, 0 of them all-correct**, with **0 mismatches**
against the `std(EM) > 0` rule; the counterfactual shaped filter would have retained 2,113,
i.e. **434 all-correct efficiency groups were correctly suppressed**, and efficiency-active
supervision fell from 520 to **86 groups (−83.5 %)**.

Recommended next experiment: **Candidate B (baseline-preserving + capped efficiency groups)**
— see §29.

## 2. Scientific question

See `E1-D/analysis/report_static_sections.md` §2 (merged below for completeness).

E1-A/E1-B used the shaped reward `R = A·(1+0.05·E)` for **both** GRPO's advantage **and** the
group filter.  E1-C showed on shared candidate pools that the shaped filter keeps a strict
superset (`S_GAP ⊂ S_E1`: 1412 → 1685 on the E1-B pool, GAP-only = 0) and that at the
fixed-size optimizer-batch level 19.4 % of the slots change hands toward *easier* groups
(success rate 1.0 vs 0.47, p = 7.4e-73).

> **How much of E1-B's accuracy-efficiency trade-off comes from the efficiency objective
> itself, and how much from the reward-dependent filter (training-distribution shift)?**

## 3. Exact experimental design

| cell | filter metric | GRPO reward | run |
|---|---|---|---|
| GAP | `EM` | `EM` | original GAP step60→120 |
| **E1-D** | **`em`** | **shaped `A(1+0.05E)`** | this experiment |
| E1-B | shaped `seq_final_reward` | shaped | E1-B step70→120 |

* **Start**: original GAP `experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60`
  (full-state `resume_path`), **not** E1-A step70 / E1-B step120.
* **D1**: step60 → step70 (`total_training_steps=70`, `save_freq=10`, `test_freq=10`, `n=8`).
* **D2**: E1-D's **own** step70 → step120 (checkpoints 80/90/100/110/120).
* **Only scientific change**: `algorithm.filter_groups.metric=em` (config-only).
* Unchanged: GRPO (`adv_estimator=grpo`, `norm_adv_by_std_in_grpo=True`), λ=0.05, cost
  `logical_search_batches`, n=8, `train_batch_size=32`, `gen_batch_size=160`,
  `ppo_mini_batch_size=32`, lr 1e-6 (+3 warmup), weight decay 0.1, clip 0.2/0.28/10.0,
  `use_kl_loss=false`, temperature 1.0, 2048/8192 prompt/response, TP=2,
  `gpu_memory_utilization=0.5`, 8-turn multi-turn with the qwen XML tool parser and the wiki
  tool config, the same RL dataset, the same SFT base model, the same evaluator.
* Validation/evaluation stays on the original GAP EM scorer; the training-side `acc` bucket is
  restored to EM so no accuracy metric is shaped.

## 4. Why E1-D follows E1-C

E1-C's controlled replay produced three facts that make E1-D the necessary next step:
(1) candidate-level strict containment with GAP-only = 0; (2) optimizer-batch level **partial
replacement** (19.4 % of slots, Jaccard 0.675 on the uncensored E1-0 pool) because the trainer
stops generating as soon as 32 groups are retained; (3) the entering groups are all-correct
(success 1.000) and the leaving ones mixed (success 0.467, p = 7.4e-73).  E1-C could only
*measure* the mechanism, not separate its contribution; E1-D removes the channel entirely and
therefore decomposes E1-B's effect.

## 5. Environment

| item | value |
|---|---|
| host | `r3l2`, Linux 5.4.0-216, glibc 2.31, 96 vCPU, 125 GB RAM |
| GPUs | 4 × NVIDIA A800 80GB PCIe (driver 570.133.20) |
| CUDA | 12.4 (V12.4.131); JIT compiled with `/usr/bin/gcc` `/usr/bin/g++` |
| conda env | `parallel-agent` — python 3.10.14, torch 2.6.0+cu124, ray 2.43.0, sglang 0.4.6.post5, transformers 4.51.1, verl 0.4.0.dev |
| env file | `/data01/wyy/Graph-Agent-Planning/environment.sh` (sourced before every RL/eval launcher) |
| order enforced | `conda activate parallel-agent` → `source environment.sh` → command; each independent launcher does this itself |
| logs | `E1-D/logs/preflight.log`, `E1-D/logs/wiki_service.log`, `E1-D/logs/train_D1_*.log`, `E1-D/logs/train_D2_*.log` |

## 6. Wiki service status

The wiki retriever is the service the repo tool config points at
(`verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml` →
`http://127.0.0.1:8008/retrieve`).  E1-D performs a **functional** retrieval probe (a real
request, not just a process check) before every training/eval entry point.
Result: **already healthy → REUSED**; no second server started, nothing stopped.
Recorded in `E1-D/logs/wiki_service.log`: PID 2230350, port 8008, command line, health check
and timestamps.  Preflight GPU/Ray check found no stale training/Ray/SGLang process, and no
unknown process was killed.

## 7. Starting checkpoint integrity

| file | bytes |
|---|---|
| `…/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60/actor/model_world_size_4_rank_{0..3}.pt` | 3,397,345,082 each |
| `…/actor/optim_world_size_4_rank_{0..3}.pt` | 6,171,911,046 each |
| `…/actor/extra_state_world_size_4_rank_{0..3}.pt` | 14,632 / 14,696 / 14,696 / 14,760 |
| `…/global_step_60/data.pt` | 1,492 (StatefulDataLoader state) |
| actor total | 45,103,143,488 B |

Preflight verified presence, non-emptiness and sizes; the resume path is the explicit
`global_step_60` directory (the experiment-wide tracker file reads 200 from the original run
and is not used).  Training logs confirm `Setting global step to 60` and
`Resuming from …/global_step_60`.

## 8. Source-code provenance

`E1-D/manifests/source_files.json` — absolute path, size, mtime and sha256 for the E1-B reward
implementation that was copied, the E1-A code, the original `batch.py`, `ray_trainer.py`,
`core_algos.py`, `reward.py`, the RL launcher chain, the evaluation launcher, `environment.sh`,
the RL dataset, the tool config and the starting-checkpoint files.

`E1-D/manifests/original_files_manifest_pre.json` / `..._post.json` — 23 protected files.
**Result: 0 sha256 mismatches** (one entry, `sglang_rollout.py`, is a package path — the pre
manifest carried the wrong path, corrected; both pre and post then agree with the file on
disk).  No original GAP / E1-0 / E1-A / E1-B / E1-C file changed.

The only files the experiment wrote outside `E1-D/` are the framework's own per-run Hydra
output directories (`verl/outputs/<date>/<time>/{main_ppo.log,.hydra/config.yaml,...}`) and the
training console logs under `verl/logs/`; both are produced automatically by
`verl.trainer.main_ppo` for every run (including E1-0/E1-A/E1-B) and no source file is
touched.

## 9. Exact code/config modifications

| file | source | modification |
|---|---|---|
| `E1-D/code/e1d_*.py` (11 modules) | `E1-B/code_changes/e1b_*.py` | identifier-only rename; reward logic untouched (proved in §10) |
| `E1-D/code/e1d_reward_manager.py` | `e1b_reward_manager.py` | **only** added E1-D filter-decoupling diagnostics: `filter_metric`, `actual_filter_keep`, `counterfactual_shaped_keep`, `mixed_correctness`, `all_correct`, `all_wrong`, `efficiency_active_excluded_by_em_filter`, and the per-generation-batch composition counters (`n_all_correct_kept`, `n_mixed_kept`, `n_all_wrong_kept`, `n_efficiency_active_excluded`, `n_mixed_efficiency_active_kept`, `efficiency_supervision_ratio_actual`) |
| `E1-D/code/pythonpath_e1d/sitecustomize.py` | `pythonpath_e1b/…` | rename + registry injection for `e1d_batch_shaped` |
| `E1-D/scripts/run_e1d_train.sh` | `run_e1b_train.sh` | **`algorithm.filter_groups.metric=em`** (the single scientific change); two phases D1/D2 with phase-specific resume and `total_training_steps`; E1-D experiment/run names; no-overwrite guards; provenance capture |
| `E1-D/scripts/*` (eval, integrity, analysis, comparison) | `E1-B/scripts/*` | renames + E1-D paths/labels; the evaluator used is the ORIGINAL `Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh`, unmodified |
| `ray_trainer.py` | — | **not modified**: `metric=em` is read straight from `new_batch.non_tensor_batch["em"]` (§11) |

## 10. Proof that the E1-D reward equals the E1-B reward

`E1-D/analysis/preflight_selftest.{md,json}` TEST 1: the E1-D and E1-B scorers were run on the
**same real rollouts** with `compute_score_em` stubbed by the recorded EM, so only the
group/efficiency math is compared.

| pool | rollouts checked | mismatches |
|---|---|---|
| E1-0 audit | 147,200 | 0 |
| E1-B run diagnostics | 641,280 | 0 |
| **total** | **788,480** | **0** |

TEST 2: `DEFAULT_LAMBDA = 0.05` in both modules and `E1D_LAMBDA=0.05` in the launcher.
TEST 5: inside 29 baseline-retained mixed groups with cost variation, the efficient and
inefficient correct rollouts get **different** shaped rewards and **different** GRPO
advantages (example: `R = [0,0,1,1,1.05,1,1,0]` → advantages 0.705 vs 0.801 for the
inefficient/efficient correct rollouts).
TEST 7: on the EM-filter-selected step-71 optimizer batch (256 trajectories) the advantages
have 0 NaN, 0 Inf, `max|adv| = 2.4749`, `mean|adv| = 0.8165`.

## 11. Proof that the filter uses EM

Two independent proofs.

**(a) Runtime.** `ray_trainer.py:1766` prints `std val <new_batch.non_tensor_batch[metric_name]>`
for every generation batch.  In the live D1 log:

```
std val  [0. 0. 0. ... 1. 1. 1.]
std val  [1. 1. 1. ... 0. 0. 0.]
```

Only 0/1 values appear — the filter input is the original EM.  A shaped reward would show
1.05.  Full record: `E1-D/analysis/filter_uses_em_proof.md`.

**(b) Data path** (pinned sources): scorer dict → `batch.py` `reward_extra_info[key].append(value)`
(includes `em`) → `reward.py::compute_reward` returns it as `reward_extra_infos_dict` →
`ray_trainer.py:1755` `non_tensor_batch.update({k: np.array(v)})` → `metric_name = "em"` →
`zip(non_tensor_batch["uid"], non_tensor_batch["em"])` → `std` per uid → kept iff
`std > 0 or |group| == 1` (l.1781) → accumulate until `num_prompt_in_batch ≥ 32` then
`batch[:32*8]` (l.1860).

**(c) Compositional proof at scale** (§24): over 83,680 candidate groups, the trainer's actual
keep decisions reproduce `std(EM) > 0` with **0 mismatches**, retain **0** all-correct groups
and **0** all-wrong groups.

## 12. Proof that validation is unchanged

* Launcher: `custom_reward_function.val_path=…/mhqa_eval.py`,
  `val_name=compute_score_em_batch` — the original GAP EM scorer, unshaped.
* Reward manager: the validation instance (`num_examine == 1`, `is_train = False`) delegates
  `verify()` verbatim to the original `BatchRewardManager.verify` (selftest TEST 6).
* Training-side `acc`: the manager writes the shaped reward into
  `token_level_scores/token_level_rewards` and then restores `data.batch["acc"]` to the
  original EM, so accuracy metrics never see the shaping.
* The evaluator binary/script is the original one; only the checkpoint path and names change.

## 13. Preflight / selftest results

`E1-D/analysis/preflight_selftest.md` — **all 7 tests PASS** (the gate before any GPU use):

| test | result |
|---|---|
| TEST 1 reward equivalence with E1-B | 788,480 rollouts, **0 mismatches** |
| TEST 2 λ = 0.05 exactly | PASS (module default + env) |
| TEST 3 filter baseline equivalence | `metric=em` reproduces E1-C's `S_GAP` exactly: E1-0 **340**, E1-B **1412**; counterfactual shaped set 424 / 1685; GAP-only = 0 |
| TEST 4 no all-correct revival | all-correct efficiency-variable groups 84 (E1-0) / 273 (E1-B): **0** retained by the EM filter, **all** retained by the shaped filter |
| TEST 5 mixed-group shaping preserved | 29/29 checked groups have distinct shaped rewards and distinct efficient-vs-inefficient advantages |
| TEST 6 validation unchanged | manager delegates to super; val scorer = original `mhqa_eval.py`; filter metric = `em` |
| TEST 7 no NaN/Inf | rewards: 0 non-finite over 788,480 rollouts; advantages: 0 NaN / 0 Inf |

Preflight (`E1-D/logs/preflight.log`): wiki healthy (reused), env correct, no stale
processes, all inputs present, no output collision, provenance written.

## 14. step60→70 training integrity (D1)

`E1-D/analysis/step70_integrity.md` — **PASS**.

* exit code 0; 11 steps (60…70) all present; 1,241 metric values, **0 non-finite**
* resume: `Setting global step to 60`; `global_step_70` checkpoint complete
  (4×model, 4×optimizer, 4×extra_state, `data.pt`)
* full-state resume evidence: extra_state `lr_scheduler.last_epoch` 60 → 70 with RNG
  (`cpu`, `cuda`, `numpy`, `random`) present in both
* error scan before the final-validation marker: **0** CUDA-OOM, **0** NCCL error,
  **0** traceback, **0** wiki/tool failures (12 benign c10d warnings)

## 15. step70 training diagnostics (D1)

151,040 rollouts / 18,880 groups / 118 generation batches.

| metric | E1-D D1 | E1-0 steps61–70 (EM filter + EM reward) | E1-B steps71–120 (shaped filter) |
|---|---|---|---|
| mean EM | 0.45714 | 0.45391 | 0.45708 |
| mean shaped reward | 0.45724 | — (reward = EM) | 0.45717 |
| mean efficiency bonus E | 0.001995 | — | 0.001827 |
| mean `logical_search_batches` | 1.4457 | 1.4407 | 1.3302 |
| mean queries | 1.9844 | 1.9898 | 1.9180 |
| mean parallel factor | 1.5021 | 1.5091 | 1.5602 |
| mean turns | 2.4456 | 2.4406 | 2.3302 |
| mean tokens | 1274.8 | 1277.8 | 1244.5 |
| retained groups (actual EM filter) | **331** | 340 | 1412 |
| counterfactual shaped-filter retained | 400 | — | — |
| efficiency-active retained | **8** | — | 291 |
| efficiency supervision density | **2.42 %** | — | 17.3 % |

## 16. step70 full 7-benchmark results

`E1-D/eval_results/step70_20260913_024024/` (original protocol, 51,201 samples, greedy):

| benchmark | nq | triviaqa | popqa | hotpotqa | 2wikimultihopqa | musique | bamboogle | macro | micro |
|---|---|---|---|---|---|---|---|---|---|
| EM | 0.376 | 0.567 | 0.399 | 0.399 | 0.439 | 0.163 | 0.440 | **0.3976** | **0.4333** |

Behaviour: rounds 1.6237, queries 2.1274, queries/round = parallel factor 1.3102, parallel
sample rate 0.4398, zero-search 0.0391 %, no-`<answer>` 0.7246 %.
Flat artifacts: `E1-D/eval_results/{results.json, benchmark_scores.csv,
behavior_metrics.csv, overall_metrics.csv}`.

## 17. GAP70 vs E1-A70 vs E1-D70

| run | filter | reward | macro EM | micro EM | rounds | queries |
|---|---|---|---|---|---|---|
| GAP70 (= E1-0 step70) | EM | EM | 0.4006 | 0.4325 | 1.6296 | 2.1280 |
| E1-A70 | shaped | shaped | 0.3984 | 0.4317 | 1.5977 | 2.1036 |
| **E1-D70** | **EM** | **shaped** | 0.3976 | **0.4333** | 1.6237 | 2.1274 |

Paired per-prompt (49,947 pairs):

* **E1-D70 vs GAP70**: micro EM **+0.08 pp**, macro EM −0.30 pp; rounds −0.36 %
  (paired vs GAP60: +2.16 %, p = 7.1e-21; EM −0.19 pp, p = 0.185 n.s.)
* **E1-D70 vs E1-A70**: micro EM **+0.16 pp** (p = 0.414 n.s.), macro EM −0.09 pp;
  rounds **+1.61 %** (p = 4.9e-13) — i.e. after 10 steps, removing the filter channel
  removes the (small) depth reduction E1-A had achieved and slightly improves EM.

This is consistent with the measured efficiency-supervision density of only 2.42 % in D1:
with so few active groups the objective channel has almost nothing to act on in 10 steps.

## 18. step70→120 training integrity (D2)

`E1-D/analysis/step120_integrity.md` — **PASS**.

* exit code 0; 51 steps (70…120) all present; 6,164 metric values, **0 non-finite**
* resume: `Setting global step to 70` from **E1-D's own** `global_step_70`
  (`experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu/global_step_70`), not E1-A/E1-B
* checkpoints `global_step_{70,80,90,100,110,120}` all complete (4×model 3,397,345,082 +
  4×optimizer 6,171,911,046 + 4×extra_state + `data.pt` each)
* full-state resume evidence: extra_state `lr_scheduler.last_epoch` 70 → 120, RNG present
* error scan: **0** OOM, **0** NCCL error, **0** traceback, **0** wiki/tool failures
  (12 benign c10d warnings)
* `E1-D/manifests/checkpoints.json` records every checkpoint's absolute path, step, file
  list, size, tracker, `data.pt` presence and actor/optimizer/extra-state completeness

## 19. step70→120 training diagnostics (D2)

669,440 rollouts / 83,680 groups / 523 generation batches.

| metric | E1-D D2 (EM filter + shaped reward) | E1-0 (EM/EM) | E1-B (shaped/shaped) |
|---|---|---|---|
| mean EM | **0.46334** | 0.45391 | 0.45708 |
| mean shaped reward | 0.46350 | — | 0.45717 |
| mean efficiency bonus E | 0.003025 | — | 0.001827 |
| mean `logical_search_batches` | 1.5009 | 1.4407 | **1.3302** |
| mean queries | 2.0439 | 1.9898 | 1.9180 |
| mean parallel factor | 1.4930 | 1.5091 | 1.5602 |
| mean turns | 2.5008 | 2.4406 | 2.3302 |
| mean tokens | 1311.8 | 1277.8 | 1244.5 |
| zero-search rate | 0.0215 % | 0.0163 % | 0.0250 % |
| retained groups (actual) | **1,679** | — | — |
| counterfactual shaped-filter retained | 2,113 | — | 1,685 (E1-C) |
| efficiency-active retained | **86** | — | 291 |
| efficiency supervision density | **5.12 %** | — | 17.3 % |

The validation-side (original NQ protocol, 128 samples, greedy) NQ EM over D2:
step70 0.373 → 80 0.371 → 90 0.387 → 100 0.389 → 110 0.384 — flat, no collapse.
`critic/score/max = 1.050` appears throughout, confirming the shaped reward is the optimized
objective.

## 20. step120 full 7-benchmark results

`E1-D/eval_results/step120_20260913_165834/`:

| benchmark | nq | triviaqa | popqa | hotpotqa | 2wikimultihopqa | musique | bamboogle | macro | micro |
|---|---|---|---|---|---|---|---|---|---|
| EM | 0.388 | 0.568 | 0.408 | 0.403 | 0.441 | 0.177 | 0.456 | **0.4059** | **0.4386** |

Behaviour: rounds 1.7234, queries 2.2315, parallel factor 1.2948, parallel sample rate 0.4367,
zero-search 0.0215 %, no-`<answer>` 0.455 %.

## 21. GAP120 vs E1-B120 vs E1-D120 (the causal matrix)

| cell | filter | reward | macro EM | micro EM | search rounds | queries | q/round | parallel factor | parallel rate | zero-search | no-answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GAP120 | EM | EM | 0.4110 | 0.4442 | 1.8537 | 2.3433 | 1.2641 | 1.2641 | 0.4229 | 0.033 % | 0.625 % |
| **E1-D120** | **EM** | **shaped** | 0.4059 | 0.4386 | 1.7234 | 2.2315 | 1.2948 | 1.2948 | 0.4367 | 0.021 % | 0.455 % |
| E1-B120 | shaped | shaped | 0.3947 | 0.4325 | 1.2950 | 1.8748 | 1.4478 | 1.4478 | 0.5057 | 0.039 % | 0.117 % |

Unpaired deltas:

| comparison | macro EM | micro EM | rounds | queries | parallel factor |
|---|---|---|---|---|---|
| E1-D120 − GAP120 | −0.51 pp | **−0.56 pp** | −0.1304 (**−7.03 %**) | −0.1119 (−4.77 %) | +2.43 % |
| E1-D120 − E1-B120 | +1.12 pp | **+0.61 pp** | +0.4284 (**+33.09 %**) | +0.3566 (+19.02 %) | −10.57 % |
| E1-D120 − GAP60 | +0.51 pp | +0.35 pp | +0.1344 (+8.46 %) | +0.1393 (+6.66 %) | −1.41 % |

Reference: `gap180` macro 0.4167 / micro 0.4506 / rounds 1.9363.

## 22. Paired per-prompt statistics

Pairing by `(data_source, prompt)` exactly as in E1-A/E1-B (popqa duplicate prompts skipped),
49,947 pairs for every comparison.  Both evaluations are greedy, so a pair is the same question
answered by the two checkpoints.

| comparison | rounds base → E1-D | Δrounds | % | decreased / increased | sign-test p | EM base → E1-D | ΔEM | up / down | sign-test p |
|---|---|---|---|---|---|---|---|---|---|
| **E1-D120 vs GAP120** | 1.8568 → 1.7270 | −0.1298 | **−6.99 %** | 9,781 / 5,718 | **1.27e-233** | 0.4450 → 0.4392 | **−0.57 pp** | 2,949 / 3,235 | **2.76e-4** |
| **E1-D120 vs E1-B120** | 1.2972 → 1.7270 | +0.4298 | **+33.13 %** | 1,345 / 15,141 | ≈0 | 0.4332 → 0.4392 | **+0.61 pp** | 3,601 / 3,297 | **2.52e-4** |
| E1-D70 vs E1-A70 | 1.5999 → 1.6256 | +0.0257 | +1.61 % | 4,883 / 5,624 | 4.87e-13 | 0.4328 → 0.4340 | +0.12 pp | 2,555 / 2,497 | 0.414 |
| E1-D70 vs GAP60 | 1.5912 → 1.6256 | +0.0344 | +2.16 % | 4,861 / 5,830 | 7.14e-21 | 0.4358 → 0.4340 | −0.19 pp | 2,463 / 2,557 | 0.185 |

Per-benchmark paired results:

* **vs GAP120** — rounds fall on **6/7** benchmarks (2wiki −0.175, musique −0.199,
  popqa −0.135, nq −0.133, triviaqa −0.089, hotpotqa −0.087; bamboogle +0.104, n=125);
  EM is between −0.0012 (musique) and −0.0088 (triviaqa), i.e. a uniform small decrease.
* **vs E1-B120** — rounds rise on **7/7** (+0.277 … +0.622) and EM rises on **7/7**
  (+0.0009 … +0.0320), i.e. the filter channel bought depth at a *uniform* accuracy cost.

**Additive decomposition** (paired means, E1-B120 − GAP120 = −0.5596 rounds / −1.18 pp EM):

| channel | Δrounds vs GAP120 | share of total | ΔEM vs GAP120 | share |
|---|---|---|---|---|
| objective (shaped reward, EM filter): E1-D120 − GAP120 | **−0.1298** | 23.2 % | **−0.57 pp** | 48 % |
| filter (shaped filter given shaped reward): E1-D120 − E1-B120 | **−0.4298** | 76.8 % | **−0.61 pp** | 52 % |
| total (E1-B120 − GAP120) | −0.5596 | 100 % | −1.18 pp | 100 % |

## 23. Efficiency supervision density

The filter decides **how much efficiency signal can be trained on at all**.

| run / pool | candidate groups | actually retained | efficiency-active retained | density | counterfactual shaped-filter active |
|---|---|---|---|---|---|
| E1-D D1 (61–70) | 18,880 | 331 | **8** | **2.42 %** | 77 |
| E1-D D2 (71–120) | 83,680 | 1,679 | **86** | **5.12 %** | 520 |
| E1-B (71–120, shaped filter) | 80,160 | 1,685 | **291** | **17.27 %** | — |
| E1-C on the E1-B pool (EM filter) | 80,160 | 1,412 | 18 | 1.27 % | 291 |

So decoupling removes **83.5 %** of the efficiency supervision on E1-D's own pool
(520 → 86 active groups).  With only ~7 % of the batch carrying any efficiency gradient, the
objective channel can only act on a thin slice of the data — which is exactly what the
step120 result shows (−7 % rounds instead of −30 %).

## 24. Actual filter composition

From the E1-D reward-manager diagnostics (per-generation-batch records), the actual keep
decisions under `metric=em`:

| phase | candidate groups | actual retained | EM-rule mismatches | retained all-correct | retained mixed | retained all-wrong | all-correct efficiency groups excluded | generation batches |
|---|---|---|---|---|---|---|---|---|
| D1 (61–70) | 18,880 | 331 | **0** | **0** | 331 | 0 | 69 | 118 |
| D2 (71–120) | 83,680 | 1,679 | **0** | **0** | 1,679 | 0 | 434 | 523 |

The trainer's `keep` set is **identical** to `std(EM) > 0` at every one of the 641 generation
batches, while the shaped filter would have added 69 + 434 = 503 all-correct groups.  This is
Q1/Q2 of §19 answered directly: **the EM filter eliminates all-correct revival and the actual
retained set equals the same-pool GAP EM filter set, 100 %.**

## 25. Query-packing / zero-search / safety analysis

| check | E1-D D2 (EM filter + shaped reward) | E1-0 (EM/EM) | reference |
|---|---|---|---|
| packing-suspect ratio (active groups) | **22 / 520 = 4.23 %** | 4.21 % | E1-B ≈ 8 % (E1-C, run-level) |
| efficient-correct mean queries | 2.118 | 2.040 | — |
| inefficient-correct mean queries | 3.190 | 2.932 | — |
| efficient-correct mean rounds | 1.564 | 1.488 | — |
| inefficient-correct mean rounds | 2.636 | 2.492 | — |
| zero-search-and-correct | **0** | 0 | 0 (E1-B) |
| zero-search rate | 0.0215 % | 0.0163 % | 0.0250 % (E1-B) |
| `metadata_valid = false` | **0** | 0 | 0 (E1-B) |
| queries per round (eval) | **1.2948** | — | 1.2641 (GAP120), 1.4478 (E1-B120) |
| no-`<answer>` rate (eval) | **0.455 %** | — | 0.625 % (GAP120), 0.117 % (E1-B120) |

No packing: the rounds reduction in E1-D is small (−7 %) and queries fall with it
(−4.8 %), and queries/round (1.295) stays close to the GAP value (1.264) rather than rising to
E1-B's 1.448.  There is no zero-search shortcut and no malformed-output increase.

## 26. Interpretation: CASE A

Pre-registered cases (§20 of the task):

| case | criterion | observed |
|---|---|---|
| **A** | E1-D EM ≈ GAP **and** the efficiency gain largely disappears | **YES** |
| B | E1-D EM ≈ GAP **and** clear efficiency gain remains | no |
| C | E1-D EM still clearly below GAP with efficiency retained | no |
| D | E1-D EM ≈ GAP and efficiency ≈ GAP | close to A but not exact: −7.0 % rounds ≠ 0 |

E1-D120 micro EM 0.4386 vs GAP120 0.4442 = **−0.56 pp** (and the paired estimate is −0.57 pp,
p = 2.8e-4), i.e. accuracy is back within ~0.5–0.6 pp of GAP and **more than half of E1-B's
−1.16 pp is recovered**; meanwhile the round reduction collapses from −30.1 % (E1-B) to
−7.0 %.  **CASE A** is the correct reading, with the refinement that the objective channel is
not literally zero — it is a small but statistically unambiguous −7 % at a −0.57 pp EM cost.

Mechanism-level conclusion: E1-B's depth reduction was **predominantly** produced by the
filter-induced distribution shift (≈77 % of the rounds effect), and roughly **half** of its
accuracy cost came from the same channel.  The remaining objective effect is real but small,
because the baseline filter leaves only ~5 % of the retained groups carrying an efficiency
gradient.

## 27. What E1-D establishes

1. **Causal decomposition of E1-B's trade-off** (paired, additive, same evaluator):
   filter ≈ 77 % of the depth gain and ≈ 52 % of the accuracy loss; objective ≈ 23 % / 48 %.
2. **The filter-decoupling implementation is exact**: `metric=em` is config-only, changes no
   original file, and the actual keep set equals `std(EM) > 0` at all 641 generation batches
   (0 mismatches, 0 all-correct groups retained).
3. **The objective channel alone is safe but weak**: −7 % search rounds for −0.57 pp micro EM
   with no packing, no zero-search shortcut, no malformed-output increase, and a *lower*
   no-`<answer>` rate than GAP.
4. **Quantitative reason for the weakness**: decoupling removes 83.5 % of the efficiency
   supervision (520 → 86 active groups), so the efficiency reward rarely acts.
5. **λ is not a lever here** (E1-C H5, confirmed by construction): membership is identical for
   every positive λ, so the missing signal cannot be recovered by tuning λ.

## 28. What it does NOT establish

* It does **not** prove that the distribution shift is the *sole* cause of E1-B's EM loss: the
  decomposition is over paired evaluation means, and the two channels also interact inside
  the GRPO advantage of mixed groups.
* It does **not** measure the filter channel's effect at a *matched* number of optimizer
  steps in a single run — the three matrix cells are three runs (policy-confounded from the
  first update).
* It does **not** show that the remaining −7 % is a pure "objective" effect: E1-D's own
  candidate distribution also differs from GAP's after the first update.
* It does **not** determine the optimal cap for Candidate B (the data support the *need* for a
  cap and bound the trade-off, but not its size).
* It does **not** claim any reward hacking: all safety indicators in §25 are clean.

## 29. Recommended next experiment

**Candidate B — Baseline-Preserving + Capped Efficiency Groups.**

Design: keep `filter_groups.metric = em` for the *decision* (so the baseline mixed groups are
never displaced), and additionally admit a **fixed, capped quota** of all-correct
efficiency-variable groups per optimizer step (or on a fraction of steps), so that part of the
depth reduction is recovered without changing which mixed groups are trained on.

Exact reason (from E1-D's measurements, not from prior belief):

* the filter channel produced **76.8 %** of E1-B's depth reduction, so removing it (E1-D)
  gives back most of the depth benefit — the mechanism is worth keeping *partially*;
* that channel also caused **≈52 %** of E1-B's EM loss, but the loss appeared as a *uniform*
  small decrease across all seven benchmarks (paired EM falls on 7/7 vs E1-D), i.e. it is a
  composition effect, not a single-benchmark artefact;
* the objective-only channel is safe but weak because it retains only **86/1,679 = 5.1 %**
  efficiency-active groups; a capped quota can raise that density deliberately while keeping
  the mixed-group quota intact;
* **λ tuning is ruled out** (identical membership for all λ > 0), so the only remaining knob
  that affects both the distribution and the signal is the quota itself.

A secondary, cheaper probe (worth running alongside): keep E1-D's setting but **increase the
number of optimizer steps**, to test whether the small −7 % objective effect accumulates.

**Not recommended**: Candidate C (λ-only) — membership is λ-invariant; and returning to
E1-B's shaped filter — E1-D shows it is the channel responsible for most of the accuracy cost.

## 30. Reproduction commands

```bash
cd /data01/wyy/Graph-Agent-Planning

# 0. preflight (wiki reuse/start, env, GPU/Ray, input integrity, provenance) + offline selftest
bash E1-D/scripts/e1d_preflight.sh
python3 E1-D/code/e1d_filter_selftest.py E1-D/scripts/run_e1d_train.sh

# 1. D1: GAP step60 -> E1-D step70
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate parallel-agent
source environment.sh
export WIKI_RAG_SERVER_URL="http://127.0.0.1:8008/retrieve"
bash E1-D/scripts/run_e1d_train_60_70.sh

# 2. D1 integrity (checkpoint + full-state resume + error scan + filter decoupling)
E1D_PHASE=D1 python3 E1-D/scripts/e1d_integrity.py

# 3. step70 7-benchmark evaluation
bash E1-D/scripts/run_e1d_eval_70.sh

# 4. D2: E1-D step70 -> step120
bash E1-D/scripts/run_e1d_train_70_120.sh

# 5. D2 integrity, then the step120 evaluation
E1D_PHASE=D2 python3 E1-D/scripts/e1d_integrity.py
bash E1-D/scripts/run_e1d_eval_120.sh

# 6. analysis: training diagnostics, causal matrix, paired tests, summary.json
bash E1-D/scripts/run_e1d_analysis.sh

# 7. everything above as one unattended chain (this is how the run was executed)
bash E1-D/scripts/run_e1d_all.sh all
```

## 31. Full generated-artifact inventory

| path | content |
|---|---|
| `E1-D/FINAL_REPORT.md` | this report (32 sections) |
| `E1-D/README.md` | status, headline numbers, layout, reproduction |
| `E1-D/code/e1d_{scorer,reward_manager,cost,state,checkpoint_patch,sglang_gpu_patch,hang_diagnostics,stall_watchdog,trainer_patch,diag_patch,selftest}.py` | reward implementation (E1-B copy, renamed) + runtime fixes |
| `E1-D/code/e1d_filter_selftest.py` | the 7-test offline selftest |
| `E1-D/code/pythonpath_e1d/sitecustomize.py` | runtime registry injection |
| `E1-D/scripts/e1d_preflight.sh` | wiki + env + GPU/Ray + input integrity + provenance |
| `E1-D/scripts/run_e1d_train.sh`, `run_e1d_train_60_70.sh`, `run_e1d_train_70_120.sh` | training launchers (filter metric = em) |
| `E1-D/scripts/e1d_eval_common.sh`, `run_e1d_eval_step.sh`, `run_e1d_eval_70.sh`, `run_e1d_eval_120.sh` | evaluation launchers (original protocol) |
| `E1-D/scripts/e1d_integrity.py` | per-phase integrity + filter-decoupling verification |
| `E1-D/scripts/e1d_analyze_training.py`, `e1d_checkpoint_stats.py`, `e1d_analyze_eval.py`, `e1d_collect_evaluation_artifacts.py`, `e1d_compare_all.py`, `e1d_paired_eval_compare.py`, `e1d_build_summary.py`, `e1d_report_numbers.py` | analysis tooling |
| `E1-D/scripts/e1d_pipeline.sh`, `run_e1d_all.sh`, `e1d_safety_monitor.py` | unattended orchestration + monitoring |
| `E1-D/configs/{resolved_step60_to70.json, resolved_step60_to70.txt, resolved_step70_to120.json, resolved_step70_to120.txt, reward_config.json}` | resolved configs and the run's reward/filter config |
| `E1-D/logs/{preflight.log, wiki_service.log, train_D1_*.log, train_D2_*.log, pipeline.log, pipeline_status.txt, issues_and_fixes.md, selftest_rerun.log}` | logs |
| `E1-D/diagnostics/checkpoint_stats_{D1,D2}/` | per-checkpoint training statistics |
| `E1-D/analysis/step{70,120}_integrity.{md,json}` | integrity + filter-decoupling evidence |
| `E1-D/analysis/preflight_selftest.{md,json}` | the 7 selftests |
| `E1-D/analysis/filter_uses_em_proof.md`, `live_filter_check_D1.md` | runtime proof that the filter uses EM |
| `E1-D/analysis/step70_three_way.md` | step70 GAP70/E1-A70/E1-D70 comparison |
| `E1-D/analysis/{D1,D2}_vs_{E1_0,E1_B}/` | training-side diagnostics incl. `safety_analysis.json` |
| `E1-D/analysis/comparison_<ts>/{comparison.json, comparison.md, vs_*.md, per_benchmark_deltas.md, paired_comparison_e1d120_vs_{gap120,e1b120}.{json,md}, paired_comparison_e1d70_vs_{e1a70,gap60}.{json,md}}` | causal matrix + paired tests |
| `E1-D/analysis/{summary.json, results.json, benchmark_scores.csv, behavior_metrics.csv, overall_metrics.csv}` | machine-readable summary + flat artifacts |
| `E1-D/eval_results/step70_20260913_024024/`, `E1-D/eval_results/step120_20260913_165834/` | full evaluations (accuracy, per-benchmark, trajectory, search-round, parallel statistics) |
| `E1-D/manifests/{source_files.json, original_files_manifest_pre.json, original_files_manifest_post.json, checkpoints.json}` | provenance + no-modification evidence |
| `E1-D/runs/{D1_20260912_235330, D2_20260913_041846}/` | run dirs: stdout/stderr, `diagnostics/*.jsonl` (rollouts/groups/calls), preflight, config |
| `experiments/DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu/global_step_{70,80,90,100,110,120}/` | the actual checkpoints (referenced from `E1-D/manifests/checkpoints.json`) |

## 32. Known limitations

1. **Single seed, single trajectory per matrix cell**; the between-cell comparisons are
   controlled in configuration, not in data (the policies diverge after the first update).
2. **Additivity is measured on paired evaluation means**, not within a single training run;
   channel interactions inside the GRPO advantage are not decomposed.
3. **The baseline filter makes efficiency supervision thin (~5 %)**, so a small objective
   effect can also mean "too little signal" rather than "no effect"; E1-D quantifies the
   density so the two readings can be separated.
4. **Search rounds are measured with two different conventions** in the trainer
   (`logical_search_batches` from the structured tool-call sequence) and in the evaluator
   (`<observation>`-split assistant spans); the two are never mixed in one table, and
   training-side vs evaluation-side numbers are labelled.
5. **No causal claim beyond the ablation**: E1-D shows what changes when the filter is
   decoupled; it does not prove that a distribution shift is the only mechanism, nor that the
   EM difference would persist under a different seed.
6. **Candidate B's quota is not determined** by this experiment.

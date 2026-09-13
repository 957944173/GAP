# E1-D static report sections (2-6, 8-9, 30, 32)

> Drafted while D1/eval70 ran; merged into `E1-D/FINAL_REPORT.md` at the end. All numbers
> that could change are kept out of this file and generated from the artifacts instead.

## 2. Scientific question

E1-A/E1-B replaced the reward `R = EM` with the success-conditioned group-relative
efficiency reward

```
A_i = original EM correctness in {0,1}
C_i = logical_search_batches
E_i = (Cmax - C_i)/(Cmax - Cmin)   if the uid group has >= 2 correct rollouts with differing cost
    = 0                            otherwise
R_i = A_i * (1 + 0.05 * E_i)       wrong rollout -> exactly 0
```

and used **that same shaped quantity for the GRPO group filter**
(`algorithm.filter_groups.metric = seq_final_reward`).  E1-C proved on the shared candidate
pools that this makes the retained set strictly larger (`S_GAP ⊂ S_E1`: 1412 → 1685 on the
E1-B pool) because all-correct groups whose correct rollouts differ in search cost acquire
`std(R) > 0`; at the fixed-size optimizer-batch level 19.4 % of the slots change hands and
the added groups are *systematically easier* (success rate 1.0 vs 0.47).

The scientific question of E1-D is therefore:

> In E1-B's accuracy-efficiency trade-off, how much comes from the efficiency **objective**
> itself, and how much from the reward-dependent **filter** (training-distribution shift)?

E1-D answers it by changing **only** the filter input:

| run | filter metric | GRPO reward |
|---|---|---|
| original GAP | `EM` | `EM` |
| **E1-D** | **`EM`** | **shaped `R = A(1+0.05E)`** |
| E1-B | shaped `R` | shaped `R` |

## 3. Exact experimental design

* **Start**: the ORIGINAL GAP checkpoint
  `experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60` (not E1-A step70, not E1-B
  step120), restored with `trainer.resume_mode=resume_path` /
  `resume_from_path=<that dir>` so that model, optimizer, lr-scheduler, RNG and the
  dataloader state (`data.pt`) all continue.
* **D1**: `global_step 60 -> 70`, `total_training_steps=70`, `save_freq=10`,
  `test_freq=10`, `rollout.n=8`.
* **D2**: E1-D's own step70 -> step120 (full-state resume inside the same experiment
  directory), checkpoints 80/90/100/110/120.
* **The only scientific variable**: `algorithm.filter_groups.metric=em` (config-only;
  `ray_trainer.py` is untouched).
* Everything else is copied byte-for-byte from the E1-B launcher: GRPO
  (`algorithm.adv_estimator=grpo`, `norm_adv_by_std_in_grpo=True`), λ=0.05, cost
  `logical_search_batches`, n=8, train_batch_size 32, gen_batch_size 160, ppo_mini_batch
  32, lr 1e-6 with 3 warmup steps, weight decay 0.1, clip 0.2/0.28/10.0, `use_kl_loss=false`,
  temperature 1.0, max_prompt 2048 / max_response 8192 / max_model_len 10240, TP=2,
  `gpu_memory_utilization=0.5`, multi-turn (8 turns, qwen XML tool parser, wiki tool config),
  the same RL dataset, the same SFT base model and the same evaluation protocol.
* **Validation/evaluation** keeps the original GAP EM scorer
  (`mhqa_eval.py:compute_score_em_batch`); the training-side `acc` bucket is restored to the
  original EM after the shaped reward is written, so no accuracy metric is contaminated.
* **Prohibited and not done**: λ changes, query/length/reflection penalties, dataset or SFT
  changes, DAPO/GRPO/advantage/`norm_adv_by_std_in_grpo` changes, n changes, wiki corpus or
  tool-parser changes, async-execution changes, backbone changes, and any automatic entry
  into Candidate B.

## 4. Why E1-D follows E1-C

E1-C's controlled replay produced three facts that make E1-D the necessary next experiment:

1. at candidate level the shaped filter **strictly contains** the EM filter
   (`GAP-only = 0`, +273 groups on the E1-B pool, +84 on the E1-0 pool);
2. at optimizer-batch level it is **not** containment but **partial replacement**
   (62/320 = 19.4 % of slots on the uncensored E1-0 pool; Jaccard 0.675), because the
   trainer stops generating as soon as 32 groups are retained;
3. the groups that enter are **all-correct** (success rate 1.000) and the ones that leave are
   **mixed** (success rate 0.467, p = 7.4e-73).

E1-C could not separate the two channels: both its recommendations (Candidate A,
filter-decoupled reward) and the alternative (Candidate B, capped quota) hinge on how much
of E1-B's behaviour survives once the filter no longer moves.  E1-D is the clean ablation
that measures exactly that, and E1-C's own measurement predicts the cost: under the EM
filter only **18 of 1412** retained groups are efficiency-active (1.3 %) versus
**291 of 1685** (17.3 %) under the shaped filter.

## 5. Environment

| item | value |
|---|---|
| host | `r3l2`, Linux 5.4.0-216, glibc 2.31, 96 vCPU, 125 GB RAM |
| GPUs | 4 × NVIDIA A800 80GB PCIe (driver 570.133.20) |
| CUDA / compiler | CUDA 12.4 (V12.4.131); JIT extensions forced to `/usr/bin/gcc` / `/usr/bin/g++` |
| conda env | `parallel-agent` (python 3.10.14, torch 2.6.0+cu124, ray 2.43.0, sglang 0.4.6.post5, transformers 4.51.1, verl 0.4.0.dev) |
| environment file | `/data01/wyy/Graph-Agent-Planning/environment.sh` (sourced before every RL/eval launcher) |
| required order | `conda activate parallel-agent` → `source environment.sh` → run; every independent launcher does this itself |
| recorded in | `E1-D/logs/preflight.log`, `E1-D/logs/wiki_service.log` |

## 6. Wiki service status

The wiki retriever is the service the repo's own search-tool config points at
(`verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml`,
`http://127.0.0.1:8008/retrieve`).  E1-D probes it functionally (a real retrieval request,
not just "a process exists") before every RL/eval entry point.  It was already healthy and
was **REUSED** — no second server was started and nothing was stopped.  PID / port / command
/ health-check / timestamp are recorded in `E1-D/logs/wiki_service.log`.

## 8. Source-code provenance

`E1-D/manifests/source_files.json` records absolute path + size + mtime + sha256 for the
E1-B reward implementation that was copied, the original `batch.py`, `ray_trainer.py`,
`core_algos.py`, `reward.py`, the RL launcher chain, the evaluation launcher, `environment.sh`,
the RL dataset, the tool config and the starting checkpoint.
`E1-D/manifests/original_files_manifest_pre.json` / `..._post.json` record the same hashes for
every original GAP / E1-0 / E1-A / E1-B / E1-C file that must not change, so the
"zero modification of the originals" claim is checkable after the fact.

## 9. Exact code/config modifications

| file | derived from | modification |
|---|---|---|
| `E1-D/code/e1d_scorer.py` … `e1d_*.py` (11 files) | `E1-B/code_changes/e1b_*.py` | identifier-only rename (`e1b_`→`e1d_`, `E1B_`→`E1D_`, `[E1-B`→`[E1-D`); reward logic untouched (proved by TEST 1 of the selftest on 788,480 real rollouts) |
| `E1-D/code/e1d_reward_manager.py` | `e1b_reward_manager.py` | added E1-D filter-decoupling diagnostics only (`filter_metric`, `actual_filter_keep`, `counterfactual_shaped_keep`, `mixed_correctness`, `all_correct`, `all_wrong`, `efficiency_active_excluded_by_em_filter`, and the per-generation-batch composition counters). The reward/`acc`/validation paths are unchanged |
| `E1-D/scripts/run_e1d_train.sh` | `E1-B/scripts/run_e1b_train.sh` | `algorithm.filter_groups.metric=em` (the single scientific change), two phases (D1/D2) with phase-specific resume path and `total_training_steps`, E1-D run/experiment names, no-overwrite guards, provenance capture |
| `E1-D/scripts/*` (analyzers, eval, comparison) | `E1-B/scripts/*` | renames + E1-D paths/labels; the evaluator itself is the ORIGINAL `Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh`, unmodified |
| original GAP / verl files | — | **not modified** (checked by the pre/post manifests) |

`ray_trainer.py` was **not** touched: `metric=em` is read straight out of
`new_batch.non_tensor_batch["em"]`, which the reward manager publishes via
`reward_extra_info` (see §11 of the final report for the runtime proof).

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

# 2. D1 integrity (checkpoints + full-state resume + error scan + filter decoupling)
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

# 7. all of the above in one unattended chain (used for this run)
bash E1-D/scripts/run_e1d_all.sh all
```

## 32. Known limitations

1. **Single seed / single trajectory.**  Each cell of the 2×2 causal matrix is one run;
   E1-D is an ablation, not a repeated-measures experiment, so differences are not given
   confidence intervals beyond the paired per-prompt tests.
2. **The three matrix cells are not step-aligned in a single run.**  GAP120 is the original
   run's step120, E1-B120 is a different branch, E1-D120 is this run.  Only E1-D's own
   step70/step120 pair is a within-run comparison.
3. **The candidate streams differ across cells** from the first policy update onwards, so the
   between-cell comparisons are controlled in *configuration*, not in *data*.
4. **Efficiency supervision under the EM filter is very sparse** (measured: see §23 of the
   final report).  A null efficiency result can therefore mean "the mechanism does not work"
   or "the signal was too thin", and E1-D is designed to distinguish these only in
   combination with the measured density.
5. **Query-packing / safety** is monitored from the training-side diagnostics and the
   evaluation dumps; "search rounds" is measured with the two different conventions used by
   the trainer (`logical_search_batches` from the structured tool-call sequence) and by the
   evaluator (`<observation>`-split assistant spans), and they are never mixed in one table.
6. **No reward-hacking claim is made from accuracy alone**; §25 reports packing, zero-search,
   malformed and no-answer rates separately.

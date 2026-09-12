# E1-C FINAL REPORT — filter-distribution diagnosis of the E1 efficiency reward

*Status*: **COMPLETE** — offline diagnosis only, no RL training, no server started.
All artifacts under `E1-C/`. Sources were opened read-only.

---

## 1. Executive Summary

E1-B (efficiency-shaped reward, step70→120) reduced sequential search depth but ended
**1.16 pp micro-EM below the original GAP trajectory at the same step** (paired
p = 2.1e-12).  The working hypothesis was that the shaped reward does not only change the
*objective* but also the **set of groups that survive GRPO's `filter_groups`**, and thereby
the effective training distribution.  E1-C tests that hypothesis by replaying both filters
on **identical candidate rollout pools**.

What is now established, with exact IDs and exact arithmetic:

1. **At the candidate level the E1 retained set strictly contains the GAP set.**
   On the E1-B candidate pool (80,160 groups): `S_GAP = 1412`,
   `S_E1 = 1685`, intersection `1412`, **GAP-only = 0**, E1-only `273`,
   Jaccard `0.838` → relation **`GAP_STRICT_SUBSET_OF_E1`**.  The same holds on the E1-0
   pool (340 ⊂ 424) and the E1-A pool (294 ⊂ 349).  This is what the filter algebra
   predicts, and it is confirmed empirically.
2. **The 273 added groups are 100 % all-correct + efficiency-varying** (k = 8/8, all
   efficiency-active).  They are 0.76 % of all 35,927 all-correct groups, and they are
   16.2 % of the E1 retained set.  They come from `hotpotqa` (189) and `nq` (84).
3. **At the optimizer-batch level the relation is NOT containment but partial overlap.**
   Replaying `ray_trainer.py`'s loop (accumulate kept groups per generation batch until
   `num_prompt_in_batch ≥ train_batch_size = 32`, then `batch[:32*8]`) on the E1-B pool, the
   E1 filter fills **all 1600** optimizer slots (32/step) while the counterfactual GAP
   filter is **right-censored in 46 of 50 steps**, filling only **1409** slots (28.2/step):
   on this stream E1 retains **1.136×** more groups per generation batch.  On the
   **uncensored** E1-0 pool (whose stream was generated under the GAP filter), E1 fills 32
   groups in **94** generation batches versus GAP's **115** (**−18.3 %**) and **62 of 320
   (19.4 %) of the optimizer slots are replaced** by E1-only groups (Jaccard 0.675).
4. **The replacement is systematically asymmetric in difficulty.**  E1-added groups have
   group success rate **1.000** (k = 8/8, all-correct); the GAP groups they displace have
   success rate **0.467** (k = 3.73/8, 100 % mixed).  Mann-Whitney **p = 7.4e-73**
   (E1-B pool); the same gap appears on the E1-0 (1.000 vs 0.518) and E1-A
   (1.000 vs 0.422) pools.
5. **λ is an on/off switch, not a strength knob.**  Retained membership is a step function
   of λ: λ = 0 → 1412 groups, **every λ > 0 ∈ {0.01, 0.025, 0.05, 0.1, 0.2} → the same
   1685 groups** (pairwise Jaccard = 1.0, nothing added or removed).  On a *fixed* optimizer
   batch the efficiency gradient (advantage of efficient-correct minus inefficient-correct)
   is **−0.626 at λ = 0** and **+0.211 … +0.219 for λ > 0** — i.e. the sign flips as soon
   as shaping exists, and the magnitude is essentially λ-independent (GRPO's
   `norm_adv_by_std_in_grpo` normalises it away).
6. **Hypotheses:** H1, H2, H3, H4, H5 **SUPPORTED**; H6
   **SUPPORTED BY ASSOCIATION** (not causally proven).
7. **Recommended next experiment: Candidate A (Filter-Decoupled Reward) as the primary
   ablation**, with the explicit quantitative caveat — and therefore Candidate B
   (quota-preserving hybrid) as the immediate follow-up — that decoupling the filter leaves
   only **18 efficiency-active groups out of 1412 retained (1.3 %, down from 291/1685 =
   17.3 %)**, i.e. it removes ~94 % of the efficiency supervision.  **λ adjustment
   (Candidate C) is ruled out by the data.**

## 2. Motivation

E1-A/E1-B replaced the reward `R = EM` by `R = EM·(1 + 0.05·E)` with
`E = (Cmax − C)/(Cmax − Cmin)` over the correct rollouts of an "active" group
(≥ 2 correct with differing `logical_search_batches`).  Because GRPO's group filter keeps a
group iff `np.std(reward) > 0`, and because the reward now varies **inside an all-correct
group** whenever the correct rollouts differ in search cost, the shaping can revive groups
that the original reward discarded.  Those revived groups are, by construction, *fully
solved* prompts whose only gradient is an efficiency ordering.  If they consume optimizer
slots that the original run would have spent on mixed-correctness prompts, the effective
training distribution shifts — a mechanism entirely separate from "the objective now
prefers shorter searches".

## 3. Exact research questions

* **RQ1** On the *same* candidate pool, how do `S_GAP` and `S_E1` relate (equal / subset /
  overlap / disjoint)?
* **RQ2** What kind of groups does E1 add (all-correct + efficiency variation? mixed? other)?
* **RQ3** Does the fixed-size optimizer batch + "generate until full" loop make the
  *actual* optimizer composition differ, even when the candidate-level relation is a
  containment?
* **RQ4** What does λ do: (A) λ = 0 vs λ > 0, (B) different positive λ, (C) if the positive-λ
  retained sets coincide, does λ only change reward/advantage magnitude?
* **RQ5** Is the E1-B accuracy drop statistically consistent with the
  training-distribution-shift hypothesis (association only)?

## 4. Source artifacts / provenance

See `E1-C/data_inventory.md` (with sha256 of every input) for the full table.  Summary:

| pool | file | rollouts | groups | gen batches | steps | reward actually used |
|---|---|---|---|---|---|---|
| `E1_0` | `E1-0/runs/20260910_step60to70_e10/audit/rollouts_pid2552655.jsonl` (251 MB) | 147,200 | 18,400 | 115 | 61–70 | original EM |
| `E1_A` | `E1-A/run_20260911_024902/diagnostics/rollouts_pid2893768.jsonl` (75 MB) | 139,520 | 17,440 | 109 | 61–70 | E1 shaped (λ=0.05) |
| `E1_B` | `E1-B/run_20260911_113432/diagnostics/rollouts_pid3388475.jsonl` (347 MB) | 641,280 | 80,160 | 501 | 71–120 | E1 shaped (λ=0.05) |

Training-loop semantics were taken from the pinned sources
`verl/verl/trainer/ppo/ray_trainer.py` (filter at l.1762-1871) and
`verl/verl/trainer/ppo/core_algos.py` (`compute_grpo_outcome_advantage`), and the
parameters from `E1-B/run_20260911_113432/config/resolved_config.json`
(`train_batch_size = 32`, `gen_batch_size = 160`, `n = 8`,
`filter_groups.metric = seq_final_reward`, `norm_adv_by_std_in_grpo = True`).

For the observational part, the original GAP run's console logs
(`verl/logs/gen_bs_supervisor_run/train_console.run{1..4}.log`, `supervisor.log`) supply the
real generation-loop counts for steps 0–200.

`uid` is the **dataset row index** (`rl_dataset.py:309-315` → `extra_info["index"]` →
`ray_trainer.py:1734`), so the training-time group — and the group the filter sees — is
`(call_index, uid)`, exactly as in E1-0.5.

## 5. Code changes

| E1-C file | derived from | change |
|---|---|---|
| `code/e1c_common.py` | `E1-0.5/scripts/e105_common.py` (validated in E1-0.5) | kept the group reconstruction `(call_index, uid)` and `gap_filter_keep` verbatim; added three-pool adapters, the λ-parameterised reward, the sequential batch-fill replay returning **selected group IDs**, the exact GRPO advantage, and dependency-free rank-sum / two-proportion tests (no scipy in this environment) |
| `code/e1c_run_all.py` | `E1-0.5/scripts/run_analysis.py` (its `part7` filter replay) | extended from "counts per step on one pool" to "IDs + censoring diagnostics on three pools", plus Analyses B, D, E (λ membership / batch-fill / advantage), F (console-log parsing) and G (hypothesis verdicts) |
| `code/e1c_figures.py` | new | 5 figures from the result CSVs |
| `code/e1c_docs.py` | new | inventory / config / error log / README generation |

No original GAP / verl file was modified (`E1-C/configs/source_provenance.json` records the
sha256 of every source that was read, including `ray_trainer.py` and `core_algos.py`).

## 6. Controlled same-pool set analysis (Analysis A)

For every group of a pool the two rewards are computed from the *same* rollouts
(`R_gap = EM`, `R_e1 = EM·(1+0.05·E)`) and the trainer's rule
`kept iff np.std(group reward) > 0 or |group| = 1` is applied.

| pool | candidates | `S_GAP` | `S_E1` | intersection | GAP-only | E1-only | Jaccard | containment ∩/GAP | containment ∩/E1 | relation |
|---|---|---|---|---|---|---|---|---|---|---|
| `E1_0` | 18,400 | 340 | 424 | 340 | **0** | 84 | 0.802 | 1.000 | 0.802 | `GAP_STRICT_SUBSET_OF_E1` |
| `E1_A` | 17,440 | 294 | 349 | 294 | **0** | 55 | 0.842 | 1.000 | 0.842 | `GAP_STRICT_SUBSET_OF_E1` |
| `E1_B` | 80,160 | 1,412 | 1,685 | 1,412 | **0** | 273 | 0.838 | 1.000 | 0.838 | `GAP_STRICT_SUBSET_OF_E1` |

Exact IDs: `results/same_pool_set_relationship.csv` (all 80,160 groups with both keep flags),
`same_pool_gap_only.csv` (empty), `same_pool_e1_only.csv` (273 rows),
`same_pool_intersection.csv` (1,412 rows), `same_pool_set_summary.csv`.

## 7. Candidate-level set relationship

**Answer to A: containment, strict, and complete.**  `S_GAP ⊂ S_E1`, i.e. the E1 filter
never drops a group the GAP filter keeps, and it adds 273 (E1-B) / 84 (E1-0) / 55 (E1-A)
new ones.  The theoretical reason is exact: a mixed group always has rewards `{0, ≥1}` and
is kept by both; an all-wrong group has all-zero rewards and is kept by neither; an
all-correct group has reward `1 + λE`, which varies iff the correct rollouts differ in cost,
so it is kept iff `λ > 0` and the group is efficiency-active.

Candidate-pool composition (E1-B, 80,160 groups):

| k = #correct | groups | share | kept by GAP | kept by E1 |
|---|---|---|---|---|
| 0 (all-wrong) | 42,821 | 53.4 % | 0 | 0 |
| 1–7 (mixed) | 1,412 | 1.76 % | **1,412 (100 %)** | 1,412 |
| 8 (all-correct) | 35,927 | 44.8 % | 0 | **273 (0.76 %)** |

So the candidate stream is extremely bimodal — the policy nearly always solves a prompt 8/8
or fails it 8/8 — and the *only* thing shaping can add at candidate level is the 0.76 % of
all-correct groups whose correct rollouts disagree about how long to search.

## 8. Sequential fixed-size batch-fill replay (Analysis C)

The replay reproduces `ray_trainer.py` exactly: generation batches are consumed in
`call_index` order; each contributes its kept groups in batch order; accumulation stops as
soon as `num_prompt_in_batch ≥ 32`; the optimizer batch is the **first 32 kept groups**.
`data.train_batch_size = 32` was auto-read from the resolved config (not hard-coded).

**Verification that the replay is the real loop** (sanity checks, all pass):

| check | result |
|---|---|
| E1 replay on the E1-B pool vs the 501 generation batches actually used | 50/50 steps identical |
| E1 replay on the E1-A pool vs the 109 batches actually used | 10/10 steps identical |
| GAP replay on the E1-0 pool vs the 115 batches actually used | 10/10 steps identical |
| GAP replay on the E1-0 pool vs E1-0.5's reported `baseline_retained` | 340 = 340 ✓ |

**Per-pool batch-fill result:**

| pool | GAP batches needed | E1 batches needed | reduction | GAP slots filled | E1 slots | E1/GAP keep rate | GAP censored steps |
|---|---|---|---|---|---|---|---|
| `E1_0` (uncensored counterfactual) | **115** (= observed 115) | **94** | **−18.3 %** | 320/320 | 320/320 | 1.000 → −18.3 % in batches | 0/10 |
| `E1_A` | 109 (censored) | 109 (= observed) | 0 % | 292/320 | 320/320 | 1.096 | 9/10 |
| `E1_B` | 501 (censored) | 501 (= observed) | 0 % | **1,409/1,600** | 1,600/1,600 | **1.136** | **46/50** |

**Reading.**  On the E1-B stream the E1 filter always fills the batch (32/32 every step,
mean 32.0) while the counterfactual GAP filter reaches the target in only 4 of 50 steps and
otherwise exhausts the generated batches — because the E1 run stopped generating as soon as
*its* batch was full.  The censoring-free statement is therefore the **keep rate per
generation batch**: 3.194 (E1) vs 2.812 (GAP) = **1.136×**, which on an unconstrained stream
translates into ≈1.9 fewer generation batches per optimizer step.  The uncensored E1-0 pool,
where the *same* stream was generated under the GAP filter, gives the direct measurement:
**94 vs 115 batches, −18.3 %**.

Per-step detail: `results/batchfill_overlap_by_step.csv`,
`results/batchfill_all_pools_by_step.csv`; per-step E1-B mean GAP-selected = 28.18 vs
E1-selected = 32.00.

## 9. Optimizer-batch overlap / replacement analysis

**Answer to B: partial replacement, not containment.**

| pool | selected slots | intersection | GAP-selected-only (displaced) | E1-selected-only (added) | Jaccard | steps with identical selection |
|---|---|---|---|---|---|---|
| `E1_0` (uncensored) | 320 / 320 | 258 | **62 (19.4 %)** | **62 (19.4 %)** | 0.675 | 0/10 |
| `E1_A` | 292 / 320 | 268 | 24 | 52 | 0.779 | 1/10 |
| `E1_B` | 1,409 / 1,600 | 1,338 | 71 | **262** | 0.801 | 0/50 |

On the uncensored pool, **62 of the 320 optimizer-batch group slots (19.4 %) change hands**.
The E1-B pool gives the same picture ((262 added)/(1600) = 16.4 %): every optimizer step
loses 1–4 mixed groups and gains 2–10 all-correct efficiency groups.
Group-level ID lists: `results/batchfill_gap_only.csv` (displaced),
`results/batchfill_e1_only.csv` (added), `results/batchfill_cohort_members*.csv`.

## 10. GAP-only vs E1-only difficulty/composition (Analysis D)

| metric (E1-B pool) | E1-added (n = 262) | GAP-displaced (n = 71) | test |
|---|---|---|---|
| correct count k / 8 | **8.00** | 3.73 | Mann-Whitney **p = 7.4e-73** |
| group success rate k/8 | **1.000** | 0.467 | p = 7.4e-73 |
| all-correct rate | **1.000** | 0.000 | two-proportion p = 2.1e-74 |
| mixed rate | 0.000 | **1.000** | p = 2.1e-74 |
| mean `logical_search_batches` | 1.821 | 1.657 | p = 4.3e-05 |
| mean queries | 2.327 | 2.481 | p = 0.845 (n.s.) |
| mean response tokens | 1,518 | 1,544 | p = 0.556 (n.s.) |

The same asymmetry holds on the other two pools
(`results/difficulty_by_pool.csv`):

| pool | E1-added k/8 | displaced k/8 | E1-added success | displaced success |
|---|---|---|---|---|
| `E1_0` | 8.00 | 4.15 | 1.000 | 0.518 |
| `E1_A` | 8.00 | 3.38 | 1.000 | 0.422 |
| `E1_B` | 8.00 | 3.73 | 1.000 | 0.467 |

**Answer to the key question in §6 of the task:** yes — the groups E1 introduces are
**solved, efficiency-variable groups**, and the groups they push out are
**mixed-correctness groups**.  The exchange is one-for-one in slots and only partially in
number (because the E1 batch also fills faster).

## 11. λ membership analysis (Analysis E, part A/B)

| λ | retained `S_λ` | ∩ with λ=0 | Jaccard vs λ=0 | new vs λ=0 | removed vs λ=0 | Jaccard vs λ=0.05 | identical to λ=0.05 |
|---|---|---|---|---|---|---|---|
| 0 | 1,412 | 1,412 | 1.000 | 0 | 0 | 0.838 | no |
| 0.01 | **1,685** | 1,412 | 0.838 | 273 | 0 | **1.000** | **yes** |
| 0.025 | **1,685** | 1,412 | 0.838 | 273 | 0 | **1.000** | **yes** |
| 0.05 | **1,685** | 1,412 | 0.838 | 273 | 0 | **1.000** | **yes** |
| 0.1 | **1,685** | 1,412 | 0.838 | 273 | 0 | **1.000** | **yes** |
| 0.2 | **1,685** | 1,412 | 0.838 | 273 | 0 | **1.000** | **yes** |

Batch-fill is identical for every positive λ (`results/lambda_batchfill.csv`: 1600 selected
slots, Jaccard 1.0 with the λ = 0.05 selection, 0.839 with the λ = 0 GAP selection).

**Answer to C:** λ = 0.05 is *not* special.  **Any λ > 0 activates exactly the same new
groups**; λ = 0 is the only distinguished value because it reproduces the GAP reward
**exactly** (verified: `reward_vector(λ=0) == EM` for all 80,160 groups, 0 mismatches).

## 12. λ advantage analysis (Analysis E, part C)

Advantages were computed with the exact `compute_grpo_outcome_advantage`
(`norm_adv_by_std_in_grpo = True`, `ε = 1e-6`, `std` with ddof = 1) on the trajectories of
the selected optimizer batch.  Two views are given:

* **per-λ batch** (`lambda_advantage.csv`): the batch selected under that λ (identical for
  all λ > 0);
* **fixed batch** (`lambda_advantage_fixed_batch.csv`): the λ = 0.05 selection held fixed and
  only the reward recomputed — this isolates the advantage effect from membership.

| λ | mean abs advantage | max abs advantage | correct − wrong gap | efficient-correct mean | inefficient-correct mean | **efficient − inefficient gap** |
|---|---|---|---|---|---|---|
| 0 | 0.6485 | 2.4749 | 1.3376 | **+0.0207** | **+0.6466** | **−0.626** |
| 0.01 | 0.7746 | 2.4749 | 1.3376 | +0.7316 | +0.5203 | **+0.2113** |
| 0.025 | 0.7746 | 2.4749 | 1.3376 | +0.7323 | +0.5202 | **+0.2121** |
| 0.05 | 0.7746 | 2.4749 | 1.3376 | +0.7332 | +0.5200 | **+0.2131** |
| 0.1 | 0.7746 | 2.4749 | 1.3376 | +0.7349 | +0.5197 | **+0.2152** |
| 0.2 | 0.7745 | 2.4749 | 1.3374 | +0.7379 | +0.5191 | **+0.2188** |

`nan = 0`, `inf = 0` at every λ.

Two directly observed facts worth highlighting:

1. **Under the original reward there is a systematic *anti*-efficiency gradient.**  On the
   same batch, correct rollouts that used *fewer* rounds receive **lower** advantage
   (0.021) than correct rollouts that used *more* rounds (0.647) — a −0.626 gap.  Within a
   mixed group all correct rollouts get the *identical* advantage under λ = 0, so this gap is
   a **composition effect**: the fewer-round correct rollouts sit in easier (higher-k) groups,
   where the per-correct advantage is smaller.  The efficiency reward therefore does not
   merely add a bonus; it **reverses an existing bias**.
2. **λ does not scale that gradient.**  The gap jumps to ≈ +0.21 as soon as λ > 0 and then
   stays flat from λ = 0.01 to λ = 0.2 (+0.211 → +0.219).  This is the expected consequence of
   GRPO's per-group normalisation (`norm_adv_by_std_in_grpo = True`): for an all-correct
   group the reward is `1 + λE`, so `(R − mean)/std = (E − Ē)/std(E)` is λ-free, and the
   same cancellation largely applies inside mixed groups.

## 13. Actual-run comparison (Analysis F) — OBSERVATIONAL / POLICY-CONFOUNDED

Generation-loop counts recovered from the real console logs (steps 70–120):

| quantity | original GAP run | E1-B run |
|---|---|---|
| generation batches per optimizer step (mean) | **10.36** | **10.02** |
| retained groups per generation batch (mean) | 3.380 | 3.461 |
| steps compared | 50 | 50 |

The two runs used almost the same number of generation batches per step.  This is *not* a
contradiction of §8: E1-B's stream is denser in keepable groups (1.136×) but its policy also
produces a different candidate mix, and the counts are policy-confounded.

**Explicitly unavailable:** the original GAP run keeps **no per-group or per-rollout
records**.  It prints `std val <reward array>` but numpy truncates it to
`[0. 0. 0. ... 0. 0. 0.]`, so per-rollout correctness inside a generation batch cannot be
recovered; therefore the original run's *composition* (k-histogram, all-correct/mixed ratio,
active-efficiency count, prompt IDs) is **unavailable** and no proxy is fabricated.  The
E1-B composition per step is given in `results/e1b_actual_composition_by_step.csv`
(candidate EM, all-correct rate, mixed rate, active-efficiency groups, mean success rate,
mean search rounds for steps 71–120).

## 14. Limitations / confounders

1. **Right-censoring of the counterfactual GAP arm** on the E1-A/E1-B pools (they were
   generated under the E1 filter).  This is quantified, not ignored, and the uncensored
   E1-0 pool is used for the headline batch-fill number.
2. **The E1-0 pool is 10 steps (61–70) of a different branch**; its policy is the step-60
   model for all 10 steps' worth of early updates.  It is the only pool whose stream was
   generated under the GAP filter, hence the only uncensored counterfactual — but it is a
   *shorter* and *earlier* run than E1-B.
3. **Replay assumes the candidate stream would be identical under the other filter.**
   True only up to the first policy update; the replay is a controlled counterfactual about
   *filtering and batch filling*, not about how the policy would have evolved.
4. **`filter_groups.max_num_gen_batches = 0`** (unlimited) — the trainer would in principle
   have kept generating; the replay's "censoring" is an artifact of the recorded stream, not
   of a trainer limit.
5. **Advantage analysis ignores** `token_level_rewards` shaping by KL (not used:
   `use_kl_in_reward = False`, `use_kl_loss = False`), sequence packing and the
   actor's mini-batch split; it reproduces the outcome-advantage step only.
6. **H6 is association only.**  The EM gap is measured on 51,201-sample evaluations of two
   different runs; the replay establishes a *mechanism that is present*, not that it caused
   the gap.

## 15. Hypothesis verdict table (Analysis G)

| id | hypothesis | verdict | key evidence |
|---|---|---|---|
| **H1** | E1 adds new retained groups on the same pool | **SUPPORTED** | +273 groups on the E1-B pool (1412 → 1685), GAP-only = 0, `GAP_STRICT_SUBSET_OF_E1`; same on E1-0 (+84) and E1-A (+55) |
| **H2** | the added groups are all-correct but efficiency-variable | **SUPPORTED** | 273/273 = 100 % all-correct (k = 8/8) and 273/273 efficiency-active; 0.76 % of all all-correct groups |
| **H3** | fixed-size batch + early stop changes the actual optimizer composition | **SUPPORTED** | uncensored E1-0 pool: E1 needs 94 vs GAP 115 generation batches (−18.3 %), 62/320 (19.4 %) optimizer slots replaced; E1-B pool: GAP censored in 46/50 steps, E1 keeps 1.136× per batch |
| **H4** | the added groups are easier than the displaced ones | **SUPPORTED** | success rate 1.000 vs 0.467 (p = 7.4e-73); all-correct rate 1.000 vs 0.000; replicated on all three pools |
| **H5** | positive λ change advantage magnitude, not membership | **SUPPORTED** | membership identical for every λ ∈ {0.01 … 0.2}; the efficiency gradient is −0.626 at λ = 0 and +0.211…+0.219 for λ > 0 |
| **H6** | the E1-B EM drop is consistent with the distribution-shift hypothesis | **SUPPORTED BY ASSOCIATION** | the mechanism is present and quantified (19.4 % slot replacement toward solved groups), and E1-B step120 is micro-EM −1.16 pp vs GAP step120 (paired p = 2.1e-12); the replay does not prove causation |

## 16. Whether the current hypothesis is supported

**Yes, as stated — with one correction of emphasis.**  The hypothesis was "the reward change
makes some previously `std = 0` groups `std > 0`, so they pass the GRPO filter, so the
effective RL training distribution changes; the EM drop may come from the training-sample
selection mechanism as well as from the efficiency objective."

* The *filter* part is **confirmed exactly**: containment at candidate level, 273 revived
  all-correct groups, all efficiency-active.
* The *distribution* part is **confirmed at optimizer-batch level** and quantified:
  partial overlap (not containment), 19.4 % of slots replaced on the uncensored pool, and the
  replacements are systematically easier (success rate 1.0 vs 0.47).
* What the data **cannot** say is that this shift *caused* the EM drop; the efficiency
  objective itself also changes the gradient on the retained mixed groups.  E1-C therefore
  supports the hypothesis **by association** and, more usefully, makes it **testable**: the
  shift can be removed without touching the objective (Candidate A), which is exactly the
  ablation the data call for.

## 17. Recommended next experiment

**Primary: Candidate A — Filter-Decoupled Reward.**

Keep the filter exactly as in GAP (`kept iff std(R_EM) > 0 or |group| = 1`, i.e. the
original correctness-only rule), and apply the efficiency-shaped reward **only inside the
groups that the baseline filter already retains**.

*Why this one:* E1-C shows the distribution shift is real, large (19.4 % of optimizer slots)
and directional (solved groups replacing mixed groups) — and that λ cannot remove it (H5).
Candidate A is the only design that removes the filter channel completely while leaving the
objective channel intact, so it cleanly separates the two mechanisms.

**Required companion measurement (and the reason Candidate B must follow):**
under Candidate A the retained set is `S_GAP = 1412` groups of which only **18 are
efficiency-active** (1.3 %, versus 291/1685 = 17.3 % today) — decoupling would cut the
efficiency supervision by ≈94 %.  If Candidate A restores GAP-level EM *and* the depth
reduction disappears, that is strong evidence that the effect was carried by the revived
groups; if Candidate A keeps most of the depth reduction, the objective channel is doing the
work.

**Secondary (run only if Candidate A shows the expected signal loss):
Candidate B — Baseline-Preserving + Capped Efficiency Groups.**  Keep the GAP filter and add
a *capped quota* of all-correct efficiency groups, so the baseline mixed groups are never
displaced.  E1-C supports the *need* for such a cap and bounds the trade-off (each added
group costs exactly one mixed group), but it **does not determine the quota** — the cap must
be chosen by a sweep.

**Explicitly not recommended: Candidate C (λ adjustment only).**  Because membership is
λ-invariant for every λ > 0, and batch-fill is identical too, **changing λ cannot address
the filter-distribution shift**; it would only change the (already ≈λ-invariant) advantage
magnitudes.

## 18. Exact reason for recommendation

* H1–H4 are all SUPPORTED with large effect sizes and exact IDs, so the filter channel is a
  genuine, quantified confound of the E1-B result — it must be removed before any conclusion
  about the efficiency objective's effect on EM can be drawn.
* H5 shows λ is not a knob for that confound: the retained set is identical for every
  positive λ, so tuning λ would leave the shift untouched (this rules out Candidate C as the
  *primary* next step, even though it is the cheapest).
* The measured cost of removing the confound is concrete: efficiency supervision would drop
  from 291 to 18 active groups on a 10-step-equivalent stream (≈94 % less), which is why the
  recommendation is Candidate A **with** a pre-registered check on whether any efficiency
  signal survives, and Candidate B as the designed fallback that keeps the baseline
  distribution intact while reserving a capped budget for efficiency groups.

## 19. Reproduction commands

```bash
cd /data01/wyy/Graph-Agent-Planning

# (1) main analyses A-G, sanity checks, all result CSVs + summary.json  (~7 min, CPU only)
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate parallel-agent
python3 E1-C/code/e1c_run_all.py

# (2) figures (parallel-agent has no matplotlib)
/home/nf5468m6/miniconda3/envs/huatuo/bin/python3 E1-C/code/e1c_figures.py

# (3) documentation (inventory / config / error log / README)
python3 E1-C/code/e1c_docs.py

# (4) single-analysis spot checks
python3 - <<'PY'
import sys; sys.path.insert(0, "E1-C/code")
import e1c_common as C
print("target train_batch_size:", C.train_batch_size())
df = C.load_pool("E1_B"); g = C.build_group_table(df)
print("groups:", len(g), "all n=8:", set(g.n_rollouts) == {8})
PY
```

## 20. List of every generated artifact

`E1-C/` (all files below were created by E1-C; nothing outside `E1-C/` was written):

| path | content |
|---|---|
| `FINAL_REPORT.md` | this report |
| `data_inventory.md` | sources, fields, hashes, strict-vs-proxy analysis list |
| `README.md` | layout, headline numbers, reproduction |
| `configs/analysis_config.json` | filter rule, truncation rule, advantage spec, pools |
| `configs/source_provenance.json` | sha256 + size of every input artifact |
| `logs/run.log` | driver run log (all analyses + sanity results) |
| `logs/run_all_stderr.log` | stdout+stderr of the driver |
| `logs/errors_and_fixes.md` | bugs found and fixed (B1–B6) |
| `code/e1c_common.py` | pools, group reconstruction, replay, GRPO advantage, tests |
| `code/e1c_run_all.py` | Analyses A–G driver |
| `code/e1c_figures.py` | figure generation |
| `code/e1c_docs.py` | documentation generation |
| `results/summary.json` | machine-readable summary of every analysis + sanity checks |
| `results/same_pool_set_summary.csv` | Analysis A, per pool |
| `results/same_pool_set_relationship.csv` | all 80,160 E1-B groups with both keep flags |
| `results/same_pool_gap_only.csv` | `S_GAP − S_E1` (empty) |
| `results/same_pool_e1_only.csv` | `S_E1 − S_GAP` (273 rows) |
| `results/same_pool_intersection.csv` | the 1,412 shared groups |
| `results/group_type_summary.csv` | Analysis B, per pool × set |
| `results/e1_only_details.csv`, `results/gap_only_details.csv` | group-type detail |
| `results/e1_only_source_breakdown.csv` | sources of the added groups |
| `results/batchfill_overlap_by_step.csv` | Analysis C per optimizer step (E1-B pool) |
| `results/batchfill_all_pools_by_step.csv` | Analysis C, all pools, both modes |
| `results/batchfill_gap_only.csv`, `results/batchfill_e1_only.csv` | displaced / added group IDs |
| `results/batchfill_cohort_members{,_E1_0,_E1_A,_E1_B}.csv` | cohort membership per pool |
| `results/difficulty_comparison.csv` | Analysis D metric-by-metric with p-values |
| `results/difficulty_by_pool.csv` | Analysis D summary for all three pools |
| `results/lambda_membership.csv` | Analysis E part A/B |
| `results/lambda_batchfill.csv` | Analysis E part B (batch-fill) |
| `results/lambda_advantage.csv` | Analysis E part C (per-λ batch) |
| `results/lambda_advantage_fixed_batch.csv` | Analysis E part C (fixed batch) |
| `results/actual_run_comparison.csv` | Analysis F, steps 70–120 |
| `results/e1b_actual_composition_by_step.csv` | E1-B observed composition per step |
| `figures/group_composition.png` | candidate pool composition (bimodality) |
| `figures/set_relationship.png` | `S_GAP` ⊂ `S_E1` |
| `figures/difficulty_cohorts.png` | E1-added vs GAP-displaced |
| `figures/lambda_effect.png` | membership step function + advantage gap vs λ |
| `figures/actual_gen_batches.png` | observed generation batches, GAP vs E1-B |

---

## Answers to the five questions posed in §12 of the task

**A. On the SAME candidate pool, is the reward-change relation containment, partial, or
completely different?**
**Containment — strictly.**  `S_GAP ⊂ S_E1` on all three pools, with **0** GAP-only groups.
E1-B: 1412 ⊂ 1685 (Jaccard 0.838, ∩/GAP = 1.000); E1-0: 340 ⊂ 424; E1-A: 294 ⊂ 349.

**B. At the fixed-size optimizer-batch level: containment, partial replacement, or
completely different?**
**Partial replacement.**  On the uncensored E1-0 pool, 62 of 320 slots (19.4 %) are replaced
(Jaccard 0.675); on the E1-B pool, 262 of 1600 E1 slots are E1-only and the counterfactual
GAP arm cannot even fill its batch (46/50 steps censored, 1409/1600 slots).  Containment at
candidate level therefore does **not** imply containment at optimizer-batch level.

**C. Does λ = 0.05 specifically cause the membership change, or does any λ > 0 activate the
same new groups?**
**Any λ > 0.**  λ = 0 gives 1412 groups; λ ∈ {0.01, 0.025, 0.05, 0.1, 0.2} give the *same*
1685 groups with pairwise Jaccard 1.0 and zero additions/removals among themselves.  λ is an
on/off switch: λ = 0 is exact GAP, λ > 0 is a different filter, and the value in between
does not matter for membership (only marginally for advantage magnitude).

**D. Is there evidence that newly added all-correct/easy efficiency groups crowd out the
harder/mixed groups the correctness-only GRPO used?**
**Yes, on the same pool and in the same optimizer step.**  19.4 % of the 32 group slots per
step change hands (uncensored pool); every added group is all-correct (success rate 1.0)
while every displaced group is mixed (success rate 0.47, p = 7.4e-73).  In E1-B's own stream
this happens in every one of the 50 steps (16.4 % of slots).  Note the correct statement is
"capacity is reallocated", not "the EM drop is caused by it".

**E. Given this, what is most worth changing next: λ, the filter, the reward, or something
else?**
**The filter.**  λ cannot change membership (H5), and the reward formula itself is already
minimal.  The next experiment should decouple the two roles of the reward: keep the GAP
correctness filter for *who trains* and use the efficiency shaping only for *how the retained
groups are ranked* (Candidate A), with the pre-registered expectation — measured here — that
this removes ~94 % of the efficiency supervision (18 vs 291 active efficiency groups), hence
Candidate B (baseline-preserving + capped efficiency quota) as the designed follow-up.

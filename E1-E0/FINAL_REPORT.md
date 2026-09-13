# E1-E0 FINAL REPORT — offline quota replay for the E1-E efficiency-group mixture

*Status*: **COMPLETE** — pure offline replay on the real E1-D D2 candidate stream.  No training,
no GPU, no model / trainer / reward / dataset modification.  Every artifact is under `E1-E0/`.

---

## 1. Executive Summary

E1-C and E1-D showed that E1-B's shaped **filter** — which admits all-correct
efficiency-variable groups — was responsible for most of E1-B's search-depth reduction
(≈77 %) and about half of its accuracy cost, while decoupling the filter entirely (E1-D) left
too little efficiency supervision (~5 % of retained groups) for the objective to act.  The
obvious next design is therefore a **quota**: keep GAP's correctness training intact and admit
at most **q** all-correct efficiency groups per 32-group optimizer batch.

E1-E0 answers, offline and on the real stream, which **q** that should be.

Method: on the E1-D D2 candidate stream (669,440 rollouts, 83,680 n=8 groups, 523 generation
batches, 50 optimizer steps), the generation loop is replayed with the **original GAP stopping
rule — only mixed-correctness groups (1 ≤ k ≤ 7) count toward `data.train_batch_size = 32`** —
while all-correct efficiency-variable groups (k = 8 with `logical_search_batches` variation)
are cached in the same pass, in stream order, and never influence the stopping point.  For a
quota q the optimizer batch is `mixed[:32−m] + efficiency_cache[:m]` with `m = min(q, |cache|)`;
the displaced mixed groups are the last m of the q = 0 batch.

Results:

| q | full-quota rate | mean efficiency groups/step | replacement fraction | efficiency supervision |
|---|---|---|---|---|
| 1 | **50/50 = 100 %** | 1.00 | 3.125 % | 3.125 % |
| 2 | **50/50 = 100 %** | 2.00 | **6.25 %** | **6.25 %** |
| 3 | **50/50 = 100 %** | 3.00 | 9.375 % | 9.375 % |
| 4 | 47/50 = 94 % | 3.94 | 12.3125 % | 12.3125 % |
| *unrestricted (E1-B-style)* | — | *8.06* | *25.19 %* | *25.19 %* |

* every step has **≥ 3** cached efficiency groups (min 3, mean 8.06), so q = 1/2/3 fill on
  **every** step and q = 4 on 47/50;
* the replaced mixed groups show **no difficulty bias** (k/8 p = 0.94 at q = 2) and **no source
  bias** (χ² p = 0.92 at q = 2); the displaced-group k/8 histogram is flat across 1/8 … 7/8;
* the inserted groups carry a **real** efficiency signal: mean cost gap 1.109 rounds
  (efficient 1.47 vs inefficient 2.58 rounds), efficient rollouts use **fewer** queries in
  99.3 % of groups and fewer tokens in 98.0 % → **no query packing**;
* E1-E reward semantics replay cleanly at every q: 0 NaN, 0 Inf, group `std > 0` in 100 %,
  efficiency ranking correct in 100 %, GRPO advantage |max| 2.4749.

**Recommendation: q = 2** (replacement 6.25 %, i.e. 2 of 32 groups per step), with q = 3 as the
pre-registered ablation and q = 4/“unrestricted” explicitly rejected as too disruptive.
**Proceed to formal E1-E training** (§20) with the configuration in §21.

## 2. Motivation

E1-B's shaped reward both changed the objective and changed **who trains** (the filter).  E1-D
removed the second channel and recovered accuracy (micro EM −0.57 pp vs GAP120, +0.61 pp vs
E1-B120) but also removed most of the depth reduction (−7 % vs GAP120 instead of −30 %), because
the EM filter leaves only 86/1,679 = 5.1 % efficiency-active groups.  So neither extreme is
right: a *bounded* amount of efficiency supervision is wanted, without displacing the
mixed-correctness groups that carry correctness supervision.

## 3. Relationship to E1-C / E1-D

| experiment | filter | reward | what it established |
|---|---|---|---|
| E1-C | — (offline) | — | `S_GAP ⊂ S_E1` on the same pool; at optimizer-batch level partially replaced (19.4 % of slots on the uncensored E1-0 pool); added groups success rate 1.0 vs displaced 0.47 |
| E1-D | EM | shaped | decoupling recovers accuracy (micro −0.57 pp vs GAP120) but leaves only ~5 % efficiency supervision → most of the depth gain disappears |
| **E1-E0** | — (offline) | — | **how large a quota of all-correct efficiency groups is safe and useful** — the first step of the E1-E design that E1-D's §29 recommended |

E1-E0 reuses E1-C's replay logic and E1-D's candidate stream; it does not re-derive anything
about model behaviour.

## 4. Exact E1-E0 design

1. **Stream**: E1-D D2's own generation stream, in trainer order (`src_order` = file order,
   `call_index` monotone).
2. **Group** = `(call_index, uid)` with n = 8 (the trainer's filter unit).
3. **Classes**: mixed-correctness `1 ≤ k ≤ 7`; all-correct efficiency group `k = 8` **and**
   `max(logical_search_batches) > min(logical_search_batches)` over the correct rollouts;
   all-wrong `k = 0` (never used).
4. **Stopping rule (control variable)**: consume generation batches in order and count **only**
   mixed groups until the target `data.train_batch_size` is reached.  The target is read from
   the resolved config (32), never hard-coded.
5. **Efficiency cache**: all-correct efficiency groups encountered inside the *same* stopping
   prefix, in stream order.  q never changes the stopping point.
6. **Batch construction**: `m = min(q, |cache|)`;
   batch = `mixed[:32−m] + cache[:m]` (total 32).
7. **Displaced groups**: the last m mixed groups of the q = 0 batch.
8. **Selection**: stream order only — no difficulty / reward / source / random selection.

## 5. Data provenance

| item | value |
|---|---|
| stream | `E1-D/runs/D2_20260913_041846/diagnostics/rollouts_pid1249393.jsonl` (361,966,409 B, sha256 `90528cb98af497aa…`) |
| rollouts / groups | 669,440 / 83,680 |
| generation batches / steps | 523 / 50 (global_step 71–120) |
| sources | nq 39,127 groups (46.8 %), hotpotqa 44,553 (53.2 %) |
| classes | mixed 1,679 · all-correct 37,926 · all-wrong 44,075 · **all-correct efficiency-variable 434** |
| target B | 32 (`E1-D/configs/resolved_step70_to120.json`, `data.train_batch_size`) |
| other config | `gen_batch_size=160`, `rollout.n=8`, `filter_groups.metric=em` |

Full hashes and the field inventory are in `E1-E0/data_inventory.md` and
`E1-E0/configs/source_provenance.json`.

## 6. Reused code and minimal modifications

| E1-E0 file | source | change |
|---|---|---|
| `code/e1e_common.py` | `E1-C/code/e1c_common.py` | kept the group reconstruction `(call_index, uid)`, the `logical_search_batches` cost and the filter rule verbatim; added mixed/efficiency classification, the mixed-only stopping rule, the efficiency cache, the quota batch construction, and dependency-free χ²/rank-sum tests |
| `code/e1e_quota_replay.py` | new driver | §4–§13 analyses on top of the above |
| `code/e1e_figures.py`, `code/e1e_docs.py` | new | figures and documentation |

No E1-C / E1-D / E1-B / E1-A / E1-0 / E1-0.5 file was written to (read-only reuse).

## 7. Baseline q = 0 replay validation

| check | result |
|---|---|
| generation batches consumed by the q = 0 replay | **523** |
| generation batches actually used by the E1-D D2 run (`calls_pid*.jsonl`) | **523** — exact match |
| optimizer steps | 50, each reaching exactly 32 mixed groups (`stopping_reached = True` for all) |
| selected mixed groups | 1,600 (= 50 × 32) |
| mixed groups available in the stopping prefixes | 1,600 → **zero surplus dropped** |
| group-level cross-check | q = 0's selected set == the first 32 mixed groups of each stopping prefix (`q0_equals_baseline_replay = True`) |
| E1-D cross-check | E1-D's EM filter retained 1,679 mixed groups over the whole stream; 1,600 of them are the optimizer batches' first 32 per step — consistent with E1-D report §24 |
| per-step generation batches | min 7, max 14, mean 10.46 |

Output: `results/q0_baseline_replay.csv` (+ `results/q0_batch_members.csv` with the exact IDs).

## 8. q = 1/2/3/4 quota availability

| q | steps | steps with full quota | FullQuotaRate | steps with 0 efficiency groups | mean inserted | p10 / p50 / p90 | max | total inserted | total replaced | ActualReplacementRate |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 50 | 50 | 1.00 | 50 | 0.00 | 0/0/0 | 0 | 0 | 0 | 0 |
| 1 | 50 | **50** | **1.000** | **0** | 1.00 | 1/1/1 | 1 | 50 | 50 | 0.03125 |
| 2 | 50 | **50** | **1.000** | **0** | 2.00 | 2/2/2 | 2 | 100 | 100 | 0.0625 |
| 3 | 50 | **50** | **1.000** | **0** | 3.00 | 3/3/3 | 3 | 150 | 150 | 0.09375 |
| 4 | 50 | 47 | 0.940 | 0 | 3.94 | 4/4/4 | 4 | 197 | 197 | 0.123125 |

The efficiency cache per step is min 3, mean 8.06, max 15 and **never empty**, which is why
q ≤ 3 saturates completely; q = 4 misses on 3 steps (each with exactly 3 available).
Full numbers: `results/quota_availability.csv`, per-step detail: `results/quota_by_step.csv`.

## 9. Generation stopping invariance

* Every q consumes the **same** generation batches in every step
  (`generation_stopping_invariant_across_q = True`; the stopping point is computed once, before
  any quota is applied).
* Consequence: the q = 0..4 comparison is a pure *batch-composition* change on an identical
  candidate set — the only intended variable.
* `results/quota_by_step.csv` carries `generation_batches_consumed` and
  `candidate_groups_seen` identically for all q in a step, so this is auditable row by row.

## 10. Optimizer-batch replacement analysis

| q | optimizer groups | efficiency groups | fraction | mixed groups used | mixed replaced | mixed supervision retained |
|---|---|---|---|---|---|---|
| 0 | 1,600 | 0 | 0 % | 1,600 | 0 | 100 % |
| 1 | 1,600 | 50 | 3.125 % | 1,550 | 50 | 96.875 % |
| 2 | 1,600 | 100 | 6.25 % | 1,500 | 100 | 93.75 % |
| 3 | 1,600 | 150 | 9.375 % | 1,450 | 150 | 90.625 % |
| 4 | 1,600 | 197 | 12.3125 % | 1,403 | 197 | 87.6875 % |

Every batch keeps exactly 32 groups (sanity check 4) and the total replacement is monotone in
q (0 → 50 → 100 → 150 → 197; sanity check 11).

## 11. Mixed-group displacement analysis

Replaced groups per q, with `k = 1..7` histograms:

| q | n replaced | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | mean k/8 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 50 | 8 | 7 | 4 | 6 | 7 | 10 | 8 | 0.5225 |
| 2 | 100 | 18 | 13 | 11 | 13 | 15 | 13 | 17 | 0.5012 |
| 3 | 150 | 26 | 19 | 18 | 14 | 21 | 21 | 31 | 0.5183 |
| 4 | 197 | 40 | 24 | 25 | 18 | 24 | 24 | 42 | 0.5032 |

Bias tests — displaced groups **vs all mixed groups in the same batches**:

| q | k/8 (p) | search rounds (p) | tokens (p) | queries (p) | source χ² (p) |
|---|---|---|---|---|---|
| 1 | 0.648 | 0.0545 | 0.190 | 0.171 | 0.684 |
| 2 | **0.943** | 0.119 | 0.119 | 0.119 | **0.920** |
| 3 | 0.509 | 0.0806 | 0.112 | 0.123 | 0.914 |
| 4 | 0.978 | 0.199 | 0.153 | 0.203 | 0.995 |

**No statistically significant difficulty or source bias at any q.**  The mean k/8 of the
displaced groups is 0.50–0.52 versus 0.5025 for all mixed groups, and the k-histogram is flat
across the 1/8…7/8 range.  The only marginal signal is q = 1's search rounds (p = 0.055,
displaced slightly *longer*-chain groups), which weakens as q grows; it is reported rather than
smoothed over (a stream-tail effect cannot be assumed away).

For contrast, the same test for inserted-vs-replaced groups is p ≈ 1e-20 (they are k = 8 vs
k ≈ 4 by construction) — that is the intended difference, not a bias.

Outputs: `results/mixed_displacement_by_quota.csv`, `figures/displaced_mixed_k_distribution.png`.

## 12. Efficiency-group composition

Of the 434 all-correct efficiency-variable groups in the stream, **403 (92.9 %) fall inside the
stopping prefixes** and are therefore cacheable; the rest appear after the stopping point of
their step.

| quantity | value |
|---|---|
| cached groups | 403 (mean 8.06/step, min 3, max 15) |
| sources | hotpotqa 263, nq 140 |
| mean / median cost gap (Cmax − Cmin) | **1.109 / 1.0** rounds |
| mean efficient-correct rounds | 1.474 |
| mean inefficient-correct rounds | 2.583 |
| mean efficient-correct queries | 2.036 |
| mean inefficient-correct queries | 3.116 |
| mean efficient-correct tokens | 1,346 |
| mean inefficient-correct tokens | 1,957 |
| efficient uses **≤** inefficient queries | **99.26 %** of groups |
| efficient uses **≤** inefficient tokens | **98.01 %** of groups |
| efficiency ranking correct (argmin cost == argmax E) | **100 %** |

Details: `results/efficiency_group_details.csv`.

## 13. Source balance

| q | nq share | hotpotqa share | Δ nq vs q = 0 | Δ hotpotqa vs q = 0 |
|---|---|---|---|---|
| 0 | 43.19 % | 56.81 % | — | — |
| 1 | 42.81 % | 57.19 % | −0.38 pp | +0.38 pp |
| 2 | 42.75 % | 57.25 % | −0.44 pp | +0.44 pp |
| 3 | 42.50 % | 57.50 % | −0.69 pp | +0.69 pp |
| 4 | 42.31 % | 57.69 % | −0.88 pp | +0.88 pp |

The largest shift is **0.88 pp at q = 4** — no meaningful tilt toward either dataset
(χ² p = 0.92 at q = 2).  Output: `results/source_shift_by_quota.csv`,
`figures/q_vs_source_distribution.png`.

## 14. Query-packing risk

Packing would mean "fewer rounds bought by cramming the same queries into them".  Within the
inserted efficiency groups, the *fewer-round* correct rollouts use **fewer** queries in 99.26 %
of groups (2.04 vs 3.12) and fewer tokens in 98.01 % (1,346 vs 1,957).  Their mean
queries-per-round is not inflated, and the batch-level effect (a quota replaces mixed groups,
it does not change how trajectories are generated) cannot create packing by construction.

**Verdict: no query-packing risk in the E1-E design.**  Note this differs from the *unrestricted*
shaped-filter regime, where the whole retained set is re-shaped: E1-E shapes only the q inserted
groups and leaves the other 32−q groups on the pure EM reward, so the packing pressure is
bounded to q/B of the batch.

## 15. E1-E reward sanity check

The formal E1-E reward semantics were replayed offline:

* mixed groups (1 ≤ k ≤ 7): `R_i = A_i` — the **original GAP correctness reward, no shaping**;
* inserted all-correct efficiency groups (k = 8): `R_i = 1 + 0.05 · E_i`,
  `E_i = (Cmax − C_i)/(Cmax − Cmin)`.

| q | trajectories | reward NaN/Inf | advantage NaN/Inf | group std > 0 | ranking correct | mean \|adv\| | max \|adv\| |
|---|---|---|---|---|---|---|---|
| 1 | 12,800 | 0 / 0 | 0 / 0 | 100 % | 100 % | 0.7723 | 2.4749 |
| 2 | 12,800 | 0 / 0 | 0 / 0 | 100 % | 100 % | 0.7725 | 2.4749 |
| 3 | 12,800 | 0 / 0 | 0 / 0 | 100 % | 100 % | 0.7740 | 2.4749 |
| 4 | 12,800 | 0 / 0 | 0 / 0 | 100 % | 100 % | 0.7758 | 2.4749 |

Advantages use the exact `compute_grpo_outcome_advantage` replica
(`norm_adv_by_std_in_grpo=True`, ε = 1e-6, std with ddof = 1).  Output:
`results/e1e_reward_sanity.csv`.

## 16. Comparison with unrestricted admission (E1-B)

| regime | efficiency groups inserted | per step | fraction of optimizer groups |
|---|---|---|---|
| E1-B (shaped filter, its own run) | 262–273 slots over 50 steps (E1-C/E1-D measurements) | ≈5.2–5.8 | **≈16–18 %** |
| E1-E0 unrestricted on the D2 stream | 403 | 8.06 | **25.19 %** |
| E1-E0 q = 2 | 100 | 2.00 | **6.25 %** |
| E1-E0 q = 3 | 150 | 3.00 | 9.375 % |

Strict comparability caveat: E1-B's numbers come from a **different policy's** pool (E1-C's
replay of the E1-B run); the D2 stream is E1-D's own policy.  The comparison is therefore
**indicative**, and is used only to say that q = 2/3 are clearly less aggressive than the
unrestricted regime that produced E1-B's accuracy cost.

## 17. Recommended q

**q = 2** (at most 2 all-correct efficiency groups per 32-group optimizer batch; mean 2.00,
100 % of steps saturated; 6.25 % of optimizer groups).

Rationale against the task's own decision rules:

| criterion | q = 1 | **q = 2** | q = 3 | q = 4 |
|---|---|---|---|---|
| full-quota rate (≥ ~80–90 % wanted) | 1.000 | **1.000** | 1.000 | 0.940 |
| replacement fraction « unrestricted (25.2 %) | 3.1 % | **6.25 %** | 9.4 % | 12.3 % |
| mixed supervision retained | 96.9 % | **93.8 %** | 90.6 % | 87.7 % |
| difficulty bias p (k/8) | 0.648 | **0.943** | 0.509 | 0.978 |
| source bias p (χ²) | 0.684 | **0.920** | 0.914 | 0.995 |
| efficiency signal present every step | yes | **yes** | yes | 47/50 |

q = 1 also saturates and is the most conservative, but it delivers only half the efficiency
signal of q = 2 while (at this sample size) showing the *only* marginal difficulty signal
(rounds p = 0.055).  q = 3 remains safe by every measured criterion and is therefore proposed
as the **pre-registered ablation**, not as the first training configuration.  q = 4 fails the
100 % saturation requirement and is rejected.

## 18. Exact reason for the q recommendation

1. **Correctness supervision first.**  q = 2 removes 6.25 % of the mixed-group slots
   (93.75 % retained), versus 12.3 % for q = 4 and ≈16–18 % for the unrestricted E1-B regime
   that E1-D showed to be harmful.
2. **Stable efficiency supervision.**  q = 2 is filled on **50/50** steps (the cache is never
   smaller than 2; in fact never smaller than 3), so every optimizer step carries efficiency
   gradient — unlike E1-D, where only ~5 % of retained groups were efficiency-active.
3. **No measurable bias.**  Displaced groups match the batch's mixed groups on k/8
   (p = 0.943), search rounds (p = 0.119), tokens, queries and source (χ² p = 0.920).
4. **The inserted groups carry real signal.**  Cost gap 1.109 rounds on average, efficient
   rollouts use fewer queries (99.3 %) and tokens (98.0 %), efficiency ranking correct 100 %,
   reward/advantage finite everywhere.
5. **Bounded packing pressure.**  Only 2/32 of the batch is shaped, and the shaped groups'
   own efficient trajectories are shorter in both rounds *and* queries.

## 19. Risks / limitations

1. **Single-policy stream.**  Availability and bias are properties of the E1-D D2 candidate
   distribution; a different checkpoint could have a different cache size per step.  The
   margin at q = 2 (min 3 cached per step) is comfortable but not unbounded.
2. **Offline by construction.**  E1-E0 cannot predict EM or search-round effects; §11's proxy
   metrics are batch-composition statements only.
3. **Stream-tail displacement is a real (if small) effect.**  The displaced groups are the last
   m mixed groups in stream order; the data show no significant bias, but q = 1's marginal
   rounds signal is a reminder that "no bias detected at n = 50–197 groups" is not "no bias".
4. **The mixed groups' own cost variation is not rewarded in E1-E.**  Some mixed groups are
   efficiency-active too (86 such groups in E1-D), and the E1-E design gives them `R = A` only.
   That is deliberate (cleaner control), but it means the total efficiency supervision is
   exactly q/B and no more.
5. **No λ sweep** — λ = 0.05 is inherited for comparability; E1-E0 does not test it.

## 20. Whether E1-E formal training should proceed

**Yes — proceed.**  The pre-registered bar was: a q that (1) saturates on most steps,
(2) replaces clearly less than the unrestricted regime, (3) shows no serious source/difficulty
bias, (4) inserts groups with genuine search-round variation, and (5) carries low packing risk.
q = 2 satisfies all five, and all 12 sanity checks pass (including the q = 0 baseline replay
reproducing the E1-D D2 run's 523 generation batches exactly).

**Do not launch it from this experiment** — E1-E0 is diagnosis only; this report stops at the
configuration proposal in §21.

## 21. Proposed E1-E formal configuration

| element | value |
|---|---|
| start checkpoint | original GAP `experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60` (full state) |
| optimizer batch target | unchanged (`data.train_batch_size = 32`) |
| **quota** | **q = 2** (pre-registered ablation: q = 3; rejected: q = 4 / unrestricted) |
| mixed-group reward | `R_i = A_i` (original EM only, no shaping) |
| all-correct efficiency-group reward | `R_i = 1 + 0.05 · E_i` |
| generation stopping | only mixed-correctness groups count toward the baseline target |
| efficiency groups | cached during the same candidate stream; at most q inserted per optimizer batch |
| selection | stream order only |
| GRPO / advantage / `norm_adv_by_std_in_grpo` | unchanged |
| rollout n, dataset, tool/wiki config, evaluator | unchanged |
| filter metric | **must stop being `seq_final_reward`**: the stopping rule is the correctness rule, so E1-E needs either `metric=em` (E1-D style) for stopping **plus** an explicit quota insertion step, or the small runtime patch that implements exactly §4 — this is the one implementation decision E1-E must make, and E1-E0's replay defines the target behaviour precisely |

## 22. Reproduction commands

```bash
cd /data01/wyy/Graph-Agent-Planning

# all analyses (CPU only, ~4 min; reads the 361 MB E1-D D2 rollout stream)
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate parallel-agent
python3 E1-E0/code/e1e_quota_replay.py

# figures (parallel-agent has no matplotlib) + unrestricted reference
/home/nf5468m6/miniconda3/envs/huatuo/bin/python3 E1-E0/code/e1e_figures.py

# documentation (inventory / config / error log / README)
python3 E1-E0/code/e1e_docs.py
```

## 23. Complete artifact list

| path | content |
|---|---|
| `E1-E0/FINAL_REPORT.md` | this report |
| `E1-E0/data_inventory.md` | stream provenance, fields, hashes, strict-vs-proxy scope |
| `E1-E0/README.md` | status, headline table, layout, reproduction |
| `E1-E0/code/e1e_common.py` | stream loader, group reconstruction, replay, GRPO advantage, tests |
| `E1-E0/code/e1e_quota_replay.py` | main driver (§4–§13 + sanity checks + recommendation) |
| `E1-E0/code/e1e_figures.py` | figures + unrestricted-admission reference |
| `E1-E0/code/e1e_docs.py` | documentation generation |
| `E1-E0/configs/analysis_config.json` | definitions, quotas, reward semantics, recommended q |
| `E1-E0/configs/source_provenance.json` | sha256 of every input |
| `E1-E0/logs/run.log` | driver log (per-q availability, all sanity checks, recommendation) |
| `E1-E0/logs/errors_and_fixes.md` | bugs found and fixed, plus three "not a bug" notes |
| `E1-E0/results/summary.json` | machine-readable summary of everything |
| `E1-E0/results/recommended_quota.json` | the recommendation + mixture rule |
| `E1-E0/results/q0_baseline_replay.csv` | per-step q = 0 baseline replay |
| `E1-E0/results/q0_batch_members.csv` | exact q = 0 selected group IDs |
| `E1-E0/results/quota_by_step.csv` | the full §5 field set for every (step, q) |
| `E1-E0/results/optimizer_batch_members_by_quota.csv` | exact selected group IDs for every (step, q) |
| `E1-E0/results/quota_availability.csv` | §6 availability / saturation |
| `E1-E0/results/quota_supervision_density.csv` | §9 density vs theoretical bound |
| `E1-E0/results/mixed_displacement_by_quota.csv` | §7 displacement + bias tests |
| `E1-E0/results/efficiency_group_details.csv` | §8 per-group efficiency composition |
| `E1-E0/results/source_shift_by_quota.csv` | §10 source balance |
| `E1-E0/results/group_success_distribution.csv` | k = 0..8 distribution per q |
| `E1-E0/results/batch_level_proxy_by_quota.csv` | §11 transparent batch-level proxies |
| `E1-E0/results/e1e_reward_sanity.csv` | §13 reward/advantage sanity per q |
| `E1-E0/results/unrestricted_reference.csv` | §16 unrestricted-admission reference |
| `E1-E0/figures/{q_vs_efficiency_fraction, q_vs_fullquota_and_replacement, q_vs_source_distribution, displaced_mixed_k_distribution, tradeoff_curve}.png` | 5 figures |

---

## Answers to the questions in §17 of the task

**A. How many efficiency groups can actually enter training at q = 1/2/3/4?**
Per optimizer step: 1.00 / 2.00 / 3.00 / 3.94 (totals over 50 steps: 50 / 100 / 150 / 197).
As a fraction of optimizer groups: 3.125 % / 6.25 % / 9.375 % / 12.3125 % (unrestricted
reference: 8.06/step = 25.19 %).

**B. Does q = 2 fill on the vast majority of optimizer steps?**
**Yes — 50/50 steps (100 %)**, and not marginally: the cache is never smaller than 3 groups per
step, so q = 2 has a one-group margin on every step and q = 3 exactly saturates.

**C. How much mixed-correctness supervision does each q replace?**
50 / 100 / 150 / 197 groups, i.e. 3.125 % / 6.25 % / 9.375 % / 12.3125 % of all optimizer
groups; mixed supervision retained = 96.875 % / 93.75 % / 90.625 % / 87.6875 %.  (Unrestricted
E1-B-style admission on the same stream would replace 25.19 %.)

**D. Are the displaced mixed groups harder, or source-biased?**
**No.**  Displaced vs all mixed groups in the same batches: k/8 p = 0.648 / 0.943 / 0.509 /
0.978, search rounds p = 0.0545 / 0.119 / 0.0806 / 0.199, tokens and queries all p > 0.11,
source χ² p = 0.684 / 0.920 / 0.914 / 0.995.  The displaced k/8 histogram is flat across
1/8…7/8, and the nq/hotpotqa split moves by at most 0.88 pp (q = 4).

**E. Do the added all-correct groups provide a stable efficiency signal?**
**Yes.**  All inserted groups have k = 8 and a positive cost gap (mean 1.109 rounds, efficient
1.47 vs inefficient 2.58 rounds); `std(R) > 0` in 100 % of them, the efficiency ranking is
correct in 100 %, and their GRPO advantages are finite (|max| 2.4749) with 0 NaN/Inf.

**F. Is there an obvious query-packing risk?**
**No.**  Within the inserted groups the efficient (fewer-round) correct rollouts use **fewer**
queries in 99.26 % of cases (2.04 vs 3.12) and fewer tokens in 98.01 % (1,346 vs 1,957); and
only q/B of the batch is shaped at all, so the packing pressure is bounded by construction.

**G. Which q is best for the first formal E1-E training?**
**q = 2** (6.25 % replacement, 100 % saturation, no detected bias, real efficiency signal).
q = 3 is the pre-registered ablation; q = 4 and unrestricted admission are rejected.

**H. Should formal E1-E training proceed?**
**Yes** — the design passes every pre-registered criterion and all 12 sanity checks.  E1-E0
itself launches nothing; the configuration is proposed in §21.

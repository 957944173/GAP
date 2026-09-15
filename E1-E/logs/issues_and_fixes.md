# E1-E issues and fixes

## D1 — disk cleanup (authorised by E1-E_task.md section 10)

* **When**: 2026-09-14 02:35, before any E1-E training.
* **Issue**: only 233 GB free on `/data01` while E1-E needs ~229 GB of its own checkpoints
  (step70 + 80/90/100/110 + 120, ~36 GB actor each, plus two FSDP->HF merges), leaving no
  safety margin.
* **Action**: deleted four **redundant intermediate** checkpoints of the finished E1-B run
  (`experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_{80,90,100,110}`,
  45,103,148,956 / 45,103,149,028 / 45,103,149,076 / 45,103,149,028 bytes).
  Freed **168 GB** (233 -> 401 GB free).  E1-B's final `global_step_120` and its whole
  experiment dir remain; nothing else was touched.
* **Why safe** (all conditions of section 10 satisfied): the E1-B final step120 is complete and
  untouched; the run is finished so no intermediate is needed for resume; the evaluations of
  those steps (`E1-B/evaluations/step{80,90,100,110}_*`) and their statistics
  (`E1-B/checkpoints/checkpoint_stats.*`, `E1-B/checkpoints/manifest.json`) are already saved;
  E1-B's FINAL_REPORT comparisons were already computed; they are not part of the current run.
* **Never touched**: original GAP step60/120/180/200, the base/SFT model, E1-0/E1-A step70
  comparison checkpoints, E1-B step120, E1-D step70/step120, all code / parquet / eval results
  / reports / logs.
* **Full record**: `E1-E/logs/disk_cleanup.log` (path, size, mtime, owning experiment, reason,
  timestamp, post-delete state, freed space).
* **Scientific variable changed**: no.

## Scaffold audit (after the cleanup, before training) — E1-D residue removed

The E1-E tree was built by copying the E1-D implementation and renaming `e1d_*` -> `e1e_*`.
A full audit of the copied files against the E1-E pre-registered design found four real defects
and two stale artifacts.  All were fixed **before** any GPU work started.

1. **`E1-E/scripts/run_e1e_train.sh`: `pythonpath_e1d` was still on `PYTHONPATH`** (and in the
   launcher's own preflight input list).  Consequence if unfixed: `sitecustomize.py` would not
   be imported, `e1e_quota_gated` would never be registered, and `main_ppo` would die at
   start-up — the experiment could not run at all.  Fixed to `pythonpath_e1e`.
2. **`run_e1e_train.sh` still exported `E1E_FILTER_METRIC="em"`** and the generated
   `config/reward_config.json` still described E1-D semantics
   (`R_i = A_i*(1+0.05E_i)` for every group, `filter_metric=em`, manager `e1e_batch_shaped`).
   The command line was already correct (`metric=e1e_quota_metric`,
   `reward_manager=e1e_quota_gated`), but the provenance file would have misdocumented the
   frozen design.  Fixed: metric, manager name, reward formula, reward-name and comments now
   state the E1-E semantics (mixed group `R_i = A_i` with no shaping; quota-selected k=8
   efficiency group `R_i = 1 + 0.05*E_i`).
3. **`E1-E/scripts/run_e1e_analysis.sh` had lost the genuine E1-D baselines**: after the
   rename, `E1D120_DUMP` pointed at E1-E's own future eval dir, so the mandated four-way
   comparison (GAP120 / E1-B120 / E1-D120 / E1-E120) would have compared E1-E with itself and
   dropped E1-D entirely.  Fixed: `E1E70_DUMP`/`E1E120_DUMP` for E1-E's own evals, plus a new
   `E1D120_DUMP` reading
   `experiments/DAPO-GAP3B-MHQA-Agent-E1D-eval-step120-4gpu/val_generations/0.jsonl`; added
   paired comparisons `e1e120_vs_e1d120` and `e1e120_vs_e1e70` (and relabelled the rest
   `e1e*`); added a symlink loop over `E1-D/eval_results/step*_*` so E1-D's step70/step120
   evaluations enter the comparison matrix read-only.
4. **`E1-E/scripts/e1e_integrity.py` still ran the E1-D check** `check_filter_decoupling()`,
   which reads `groups_pid*.jsonl` fields (`baseline_keep_em_std_gt_0`,
   `shaped_keep_reward_std_gt_0`, `all_correct`, ...) that the E1-E reward manager does not
   write.  Replaced with `check_quota_semantics()` + `check_log_vs_diagnostics()`: the full
   section 11/12 invariant set measured on E1-E's own `quota_steps/quota_calls/quota_groups/
   rollouts` diagnostics (32 groups per step, `quota_actual <= 2`, mixed-only stopping,
   `mixed reward == EM` per rollout, efficiency reward in [1,1.05] with `em == 1`, wrong reward
   exactly 0, published metric == EM or 1+0.05E, advantage NaN/Inf == 0, no quota error lines,
   fill rate / shortfall / displaced counts per step) **plus** an independent cross-check of
   the trainer's own logged `critic/score/max` against the quota diagnostics (1.05 exactly on
   steps with an insertion, <= 1.0 on steps without).
5. **Stale E1-D self-tests parked**: `E1-E/code/e1e_filter_selftest.py` and `e1e_selftest.py`
   were E1-D-semantics copies that wrote to the same output paths as the real E1-E gate.  They
   were moved (not deleted) to `E1-E/legacy_e1d_unused/` with a README explaining why they
   cannot be used; the E1-D originals remain untouched in `E1-D/code/`.  `run_e1e_all.sh` now
   gates on `e1e_runtime_selftest.py` instead.
6. **Missing automatic gate**: E1-E_task.md section 15 requires an unattended D1 -> D2
   decision.  The copied chain had no gate script at all.  Added `E1-E/scripts/e1e_d1_gate.py`
   implementing exactly the section 15 hard conditions (integrity fatal patterns, quota
   semantics, checkpoint completeness, evaluation completeness, micro-EM consistency drop
   > 1.0 pp vs E1-0 step70 and E1-D step70, query-explosion-with-rounds-decrease, mass
   zero-search, malformed/no-answer burst) while explicitly **not** stopping on a 1-3% step70
   efficiency gain.  Wired into `run_e1e_all.sh` (`stage_gate`) and `e1e_pipeline.sh`.

Cosmetic renames done in the same pass: `e1e_eval_common.sh` now exports
`EVAL_MODEL_KIND="e1e"` (the original GAP evaluator only requires a non-`rl` kind),
`e1e_paired_eval_compare.py` takes `--e1e-dump` (`--e1d-dump` kept as an alias),
`sitecustomize.py` docstring path corrected.

Section 11 gate evidence: `E1-E/analysis/preflight_selftest.{md,json}` — PART A planner replay
matches E1-E0's `optimizer_batch_members_by_quota.csv` for **q=0 and q=2 on 50/50 optimizer
steps**, PART B replays real generation batches through the actual `E1EQuotaRewardManager` and
the trainer's own accumulate/truncate loop (same batch size and same group set on the steps
exercised), PART C asserts 13 named invariants covering all 12 mandated section 11 items.

**Scientific variable changed**: no (all fixes are bug repairs in the E1-E scaffolding or in
provenance/reporting; the pre-registered design, λ, q, reward formulas, batch sizes, stopping
rule and learning-rate/batch configuration are untouched).

## D1 (during training) — reward-manager diagnostic bug in two id lists

* **When**: found 2026-09-14 10:03 while inspecting the first completed optimizer step (step 61)
  of D1, i.e. during training.
* **Symptom**: `quota_steps_pid*.jsonl` recorded
  `inserted_efficiency_group_ids: ["61:158926", "61:158926"]` (the same id twice) and an empty
  `mixed_selected_group_ids`, while `quota_actual` was correctly 2.
* **Root cause**: in `E1EQuotaRewardManager._record` the two comprehensions used `gid` (a loop
  variable left over from the preceding `for g, role in kept_meta:` loop) instead of
  `g["gid"]`, so every element of those lists became the *last* iterated group's id.
* **Impact**: diagnostics only. The group selection, the quota, the batch composition, the
  rewards, the advantage, `acc`, the stopping rule and every other recorded field are
  unaffected (the buggy lists are written after the batch has been materialised).
  `len()` of those lists stayed correct, so no downstream count was wrong.
* **Fix**: use `g["gid"]` (fixed for D2, whose processes start fresh). D1 is **not** restarted:
  the defect cannot change the trained model and a restart would discard ~30 min of GPU work.
* **Authoritative re-derivation**: `E1-E/scripts/e1e_integrity.py` now re-derives the inserted
  efficiency-group ids (and the stopping call's kept mixed ids) from
  `quota_groups_pid*.jsonl` for **both** phases, verifies that they are distinct and that their
  count equals `quota_actual`, and records the manager's original values alongside
  (`inserted_ids_as_recorded`). `e1e_prediction_vs_observation.py` does the same. The gate and
  the integrity verdict use the derived, verified lists.
* **First-step runtime values (step 61)**: 10 generation batches, 30 mixed groups kept + 2
  distinct cached efficiency groups = **32 groups** = the optimizer batch, quota filled,
  `eff_shortfall` 0, `advantage_nan`/`inf` 0, mixed-group reward exactly equal to EM.

## D1 (during training) — reward-manager advantage diagnostic empty at stopping calls

* **When**: found 2026-09-14 10:25, while checking the first two completed optimizer steps.
* **Symptom**: the step-level records for steps 61 and 62 had
  `advantage_mean/std/min/max = null`, `reward_mean/min/max = null`, `em_mean = null`, even
  though `quota_actual = 2`, the batch was 32 groups, and the two injected efficiency groups do
  appear in `quota_groups_pid*.jsonl` with correct rewards (1.0125 / 1.01875, in [1, 1.05]).
* **Root cause**: `_record` built the advantage list with
  `for j, s in zip(g["idx"], vals)` where `vals` is the cached per-rollout score list of an
  **injected** group and `g["idx"]` is empty for injected groups, so `zip` yielded nothing.
  Whenever the stopping call keeps no mixed groups (the typical case with the frozen mixed cap)
  the diagnostic arrays were therefore empty.
* **Impact**: diagnostics only. The advantage actually used for training is computed by
  `core_algos.compute_grpo_outcome_advantage` from the shaped `token_level_rewards` inside the
  materialised `DataProto`; it does not read these fields. Per-call records of non-stopping
  batches and all per-group reward/EM/efficiency/cost rows were already correct.
* **Fix**: iterate `vals` directly (equivalent for in-batch groups, correct for injected ones)
  and use the same list for the `em_mean` field (fixed for D2, whose processes start fresh).
  D1 is not restarted.
* **Independent verification**: `E1-E/scripts/e1e_integrity.py` now recomputes the exact GRPO
  advantage **from the recorded per-rollout rewards** of each step's materialised batch
  (`advantage_recomputed_per_step`) and reports NaN/Inf from that recomputation alongside the
  manager's own field, so the finiteness claim is no longer vacuous. Recomputed values for
  D1 steps 61-62: no NaN/Inf, mean ~0, std 0.937.
* **Second diagnostic defect fixed at the same time** (same root pattern): the leaked-variable
  id lists recorded in the previous entry.

## D1 — live verification of the frozen reward formula (informational)

From the passive safety monitor's per-step aggregate over the materialised optimizer batches
(`E1-E/runs/D1_*/logs/safety_monitor_agg.json`, 256 rollouts = 32 groups x n=8 per step):

* step 61: `sum_em = 134.0`, `sum_reward = 134.25000000000003`, `sum_eff = 5.0`
  -> `sum_reward - sum_em = 0.25 = 0.05 x sum_eff` **exactly**.
* the trainer's own logged metric is `critic/score/max = 1.050`, `critic/score/min = 0.000` on
  every completed step, i.e. the shaped efficiency bonus (1 + 0.05 x E, max 1.05) is present in
  the reward tensor that feeds GRPO, while mixed groups carry the unshaped EM (0/1).

This is the pre-registered reward semantics verified in the live run (mixed: `R_i = A_i`; gated
k=8 group: `R_i = 1 + 0.05 E_i`; wrong rollout: 0), independently of the diagnostics.

The monitor also emits `TRAIN_EM_DROP` / `PARALLEL_FACTOR_INCREASE` alerts. These are **passive**
and are not stop conditions: the monitor compares each step's *training-batch* statistics with
the first observed step, and a 32-group mixed-only batch (all groups have 1 <= k <= 7 by
construction) has a large step-to-step sampling variance (the observed per-step training EM
moves between ~0.40 and ~0.52 with no trend). The section-15 gate deliberately uses the
7-benchmark **evaluation** micro EM (51,201 samples) against the frozen reference runs, plus the
quota-semantics/warning checks, not these sampling-noise alerts.

## D1 RESTARTED — empty-keep batches leaked raw groups into the trainer (design violation)

* **When**: detected 2026-09-14 11:49, during D1 step 69/70; the invalid run was stopped at
  11:52 and archived to `E1-E/runs/INVALID_D1_20260914_093225/` (with a `WHY_INVALID.md`).
* **Symptom**: step 67 produced no `quota_steps` record while the trainer's own log showed it
  training normally; the manager's recorded calls for step 67 were gb 1,2,4,5,6,7,8,9 (gap at 3
  and 10) with only 27 groups kept, yet the trainer accumulated 34 groups and trained on 32.
* **Direct evidence**: `rollouts_pid*.jsonl` contained calls that returned the **untouched raw
  generation batch** (1280 rollouts = 160 groups x n=8) with **no matching `quota_calls` record** —
  D1 steps 63, 64, 65, 67, 68 (6 calls in total; the new integrity invariant reports
  `n_unrecorded_calls = 6` and `steps_with_generation_batch_gaps = [63,65,67,68]`).
* **Root cause**: in `E1EQuotaRewardManager._apply_quota_inplace`, when the frozen plan selected
  nothing from a generation batch (all candidates all-correct *without* cost variation or
  all-wrong, or the frozen mixed cap already reached) the method did
  `self.quota_errors += 1; return`, leaving the `DataProto` **unchanged**. The trainer then ran
  its own `std > 0` filter on the raw 160 candidate groups, kept the uncached efficiency groups
  (published metric `1 + 0.05 E` has std > 0), accumulated them into the optimizer batch and let
  them advance the **mixed-only** stopping counter. Step 67's optimizer batch was therefore
  27 mixed + 5 unbudgeted efficiency groups instead of 30 mixed + 2 cached efficiency groups.
  This is exactly the section-15 hard stop condition "quota 导致 generation stopping 被改变" and
  it violates the frozen design (section 1: only mixed groups count toward B = 32, efficiency
  groups are cached and injected at the stopping batch as mixed[:B-m] + cache[:m]).
* **How often**: the E1-E0 candidate stream contains an empty-keep generation batch in **18 of
  50** optimizer steps (steps 74, 75, 76, 82, 86-89, 93-96, 101, 106, 110, 113, 115, 116 for
  q = 2), i.e. the defect would have contaminated roughly a third of D2, not a corner case.
* **Fix** (in the manager, no design change):
  1. the efficiency-group stash now happens *before* the empty-keep test, so a later injection
     still finds its cached slices;
  2. when the plan keeps nothing the method sets `self._keep_nothing`, records the call
     (`kept_nothing: true`, `quota_actual: 0`, `note` explaining the neutralisation) and returns
     with the planner state correctly advanced;
  3. `__call__` then replaces the **published** filter metric
     (`result["reward_extra_info"]["e1e_quota_metric"]`) with a constant, so the trainer's
     `std > 0` filter drops **every** group of that batch: the accumulation and the stopping
     counter are provably untouched, which is what the frozen design requires. Nothing else
     about the batch is modified and no original verl file is touched.
* **Regression test (section-11 gate extended, `PART D`)**: `e1e_runtime_selftest.py` now
  (D1) inserts a synthetic no-keep batch into real steps 71-76 and asserts the materialised
  optimizer batch is **identical to the run without the insertion** and still 32 groups, (D2)
  runs the real-stream steps that do contain empty-keep batches (74, 75, 76) and asserts they
  reproduce E1-E0's frozen membership exactly, and (D3) asserts the manager wrote the
  `kept_nothing` diagnostic records. All PART D checks pass; PART A (q=0 and q=2 planner match
  on 50/50 steps), PART B (runtime replay) and PART C (13 invariants) still pass as before.
* **New runtime invariant in the integrity checker**: `manager_calls_served` (distinct
  `(global_step, timestamp)` groups in `rollouts_pid`) must equal `manager_calls_recorded`, the
  per-step `generation_batch_in_step` counter must be gapless, and every non-`kept_nothing` call
  must have returned exactly `8 x kept_groups` rollouts. On the archived invalid run this check
  reports `all_manager_calls_recorded = false` with gaps in steps 63/65/67/68, i.e. the gate
  would have refused D2 even without the manual inspection.
* **Cost**: D1 had reached optimizer step 69/70 at 11:51 (1 h 50 min of the 2 h 20 min phase)
  plus ~30 min of startup; the step-70 checkpoint had not been written yet, so nothing existed in
  `experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/` and no cleanup was required. The
  whole D1 -> step70 evaluation -> D2 chain was restarted from the original GAP step-60
  full state; the invalid run's diagnostics are kept for the audit trail but are **never** used
  for results.
* **Scientific variable changed**: none (bug fix only; the frozen design, q = 2, lambda = 0.05,
  reward formulas and stopping rule are untouched).

## Restarted D1 — live confirmation that the empty-keep fix works

Run `E1-E/runs/D1_20260914_115858` (started 2026-09-14 11:58), checked at 13:03 after steps
61-63: every step produced exactly 32 groups (30 mixed + 2 injected efficiency), `eff_shortfall`
0, `manager_calls_served == manager_calls_recorded` (37), **0 unrecorded calls, 0 oversized
returned batches** (> 256 rollouts), no `generation_batch_in_step` gaps and no call whose
returned rollout count differs from `8 x kept_groups`.  The previously leaking case has already
occurred **twice** in this run and is now recorded as `kept_nothing` (2 calls) with the published
filter metric neutralised, i.e. the trainer provably drops those batches and the mixed-only
stopping counter is untouched.

## Driver bug: the `all` chain swallowed a stage failure (fixed) + two integrity-checker bugs

* **When**: 2026-09-14 15:07, immediately after D1.
* **Symptom**: `run_e1e_all.sh all` printed "E1-E ALL STAGES DONE" and exited 0 even though its
  D1-integrity stage had just printed `VERDICT: CHECK FAILED`; the step70 evaluation, gate and D2
  therefore never started (GPUs went idle).
* **Root cause**: the chain was `stage_a && stage_b && ... ` followed by an unconditional
  `echo "ALL STAGES DONE"` and then `rc=$?`.  The `&&` list short-circuited on the integrity
  failure (its status was never captured) and the `echo` reset `$?` to 0, so `rc` was 0.
* **Fix**: the driver now iterates an explicit stage list, captures each stage's status
  immediately, writes `logs/pipeline_status.txt = STATUS=BLOCKED_<stage>` and exits non-zero on
  the first failure, and prints which stage failed.  A new `allfrom <stage>` mode resumes the
  chain at a given stage (used here to restart at `eval70`, since D1's training itself is valid).
* **Two checker bugs found while auditing the D1 failure** (the integrity verdict was a false
  negative, not an experiment problem):
  1. the log cross-check required the metric line to *start* with `step:` while the console
     prefixes it with `(TaskRunner pid=...)`, so it found 0 of 10 steps; it now uses a
     prefix-tolerant search and merges per-step metrics instead of overwriting them;
  2. the reward-metric consistency check flagged the 6,400 rollouts of the five `kept_nothing`
     calls, whose published metric is *deliberately* neutralised to 0.0, and the efficiency
     cost-variation check read the injected groups' `cost_gap: null` record field instead of the
     authoritative per-rollout costs.  Both now use the correct sources (neutralised calls are
     matched with a 3 s timestamp tolerance because a call record and its rollout rows are
     timestamped separately; cost variation is derived from the rollouts).
* **Result after the fixes**: `E1-E/analysis/step70_integrity.json` -> `integrity_ok = true`,
  `quota_semantics_ok = true`, 0 metric inconsistencies among 2,560 trained rollouts, 0
  cost-variation violations, log cross-check 10/10 steps (`critic/score/max = 1.05` exactly on
  every step with an insertion).  The same checker still flags the archived invalid run
  (`quota_semantics_ok = false`, 5 unrecorded / 5 oversized calls, gaps at steps 63/65/67/68),
  so it discriminates the real defect.

## Post-D2 — two more checker/driver bugs (both fixed; no experiment impact)

* **When**: 2026-09-15 05:38-06:30, after D2 reached step 120 (all six checkpoints present).
* **Driver `rc` capture bug**: `run_one` used `if ! "stage_$st"; then local rc=$?`, and `$?` there
  is the status of the `!` operator (0 when the stage failed), so a failed stage was reported as
  `FAILED (rc=0)` and the loop continued to the next stage.  Fixed by running the stage first and
  capturing `$?` before testing it.  (Effect of the bug this time: D2's integrity stage reported
  a false failure, which was itself a checker bug, and the chain went on to run eval120 -- which
  was needed anyway, so no work was lost.)
* **Over-strict quota criterion**: the integrity check required `steps_full_quota == steps_used`,
  i.e. a 100 % full-quota rate, but the frozen design is `m = min(q, |cache|)`, so a step whose
  stopping batch has only one cached efficiency group *must* insert one, not two.  The criterion
  now verifies the frozen formula on every step (`quota_actual == min(quota_target,
  efficiency_cache_size)`, or a shortfall exactly explained by the recorded `eff_shortfall` from
  the documented mixed cap) and reports the fill rate as a statistic instead of a pass/fail gate.
* **Observed D2 values after the fix**: `integrity_ok = true`; the formula holds on **50/50**
  steps; fill rate 47/50 = 97 % (steps 110 and 119 had `|cache| = 1` at their stopping batch, so
  `m = min(2, 1) = 1` exactly as pre-registered); one cap-induced shortfall at step 72
  (`|cache| = 2`, `mixed_kept_cum = 31`, quota 1, recorded `eff_shortfall = 1`); 97 efficiency
  groups inserted in total, i.e. an observed replacement rate of 97/3200 = **6.06 %** versus the
  6.25 % predicted by E1-E0 on the E1-D stream; 24 `kept_nothing` calls handled correctly;
  0 unrecorded, 0 oversized, 0 generation-batch gaps, 0 metric inconsistencies, 0 NaN/Inf
  advantages, 0 zero-search and 0 malformed rollouts; log cross-check 50/50 steps.
  D1: fill 10/10, 20 inserted, 0 shortfall, `integrity_ok = true`.
* **Scientific variable changed**: none.

## Completion record (2026-09-15)

* Chain: preflight + section-11 selftest (PART A/B/C/D) PASS -> D1 (61-70) -> D1 integrity PASS
  -> step70 7-benchmark evaluation -> automatic section-15 gate PASS -> disk guard (no deletion
  needed) -> D2 (71-120, checkpoints 80/90/100/110/120) -> D2 integrity PASS -> step120
  7-benchmark evaluation -> analysis (four-way comparison, 6 paired tests, summary) -> post-run
  provenance manifest -> 6 figures -> `E1-E/FINAL_REPORT.md` (31 sections).
* One process-hygiene incident at the very end: `run_e1e_all.sh` was rewritten while the running
  chain was still reading it, so bash hit a partially written line (`syntax error near ';;'`) and
  the chain died after the analysis stage. The two remaining stages were re-run individually
  (`run_e1e_all.sh postmanifest`, `run_e1e_all.sh figures`) with no loss of results. Lesson: do
  not edit a launcher that a live chain is executing.
* Final numbers (detail in `E1-E/FINAL_REPORT.md`): E1-E120 micro EM 0.439475 / rounds 1.5203 vs
  GAP120 0.444170 / 1.8537, E1-B120 0.432535 / 1.2950, E1-D120 0.438561 / 1.7234. E1-E120 meets
  both preregistered criteria vs GAP120 (-0.47 pp EM, -17.99 % rounds) and Pareto-dominates
  E1-D120 (+0.09 pp micro EM, -11.78 % rounds; paired accuracy p = 0.568, paired rounds
  p ~ 0 over 49,947 prompts). q = 2 runtime fill rate 95 % overall (D1 100 %, D2 94 %), observed
  replacement rate 6.09 %, 1 recorded cap shortfall, 0 semantic violations.
* Disk: `/data01` 172 GB free at the end; E1-E's own six checkpoints total 227 GB; the only
  deletion in the whole experiment was the E1-B intermediate cleanup before training (168 GB
  freed); the pre-D2 guard deleted nothing.
* Provenance: `E1-E/analysis/original_files_unchanged.json` -> original GAP/verl files
  byte-identical (23/23 files), and the one path the pre-run list misspelled
  (`sglang_rollout/sglang_rollout.py`) matches E1-0's independent 2026-09-10 hash.

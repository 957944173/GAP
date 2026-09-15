# E1-E — q=2 Solved-Group Gated / Capped Efficiency Training — FINAL REPORT

## 1. Executive Summary

E1-E (q=2 solved-group gated / capped efficiency training) reports `experiment_status = COMPLETE` and the mechanical `case = STRONG_PASS`. Step70: micro EM 0.4339, macro EM 0.3966, rounds 1.6159, queries 2.1152. Step120: micro EM 0.4395, macro EM 0.4049, rounds 1.5203, queries 2.0194. Four-way step120 deltas: vs GAP120 — Δmicro EM -0.47 pp (0.4442→0.4395), Δmacro EM -0.61 pp, Δrounds -0.3335 (-17.99%), Δqueries -0.3239 (-13.82%). vs E1-B120 — Δmicro EM +0.69 pp (0.4325→0.4395), Δmacro EM +1.01 pp, Δrounds +0.2253 (+17.40%), Δqueries +0.1446 (+7.71%). vs E1-D120 — Δmicro EM +0.09 pp (0.4386→0.4395), Δmacro EM -0.10 pp, Δrounds -0.2031 (-11.78%), Δqueries -0.2120 (-9.50%). D1 runtime quota (steps 61-70): fill rate 100.0000% (10/10 steps full), inserted 20 efficiency groups, displaced 30 mixed groups, eff_shortfall 0, quota_semantics_ok = True. D2 runtime quota (steps 71-120): fill rate 97.0000% (47/50 steps full), inserted 97 efficiency groups, displaced 161 mixed groups, eff_shortfall 1, quota_semantics_ok = True. D1 gate: PASS. Training-side side-effect shares: zero-search 0.0000% (D1) / 0.0000% (D2); malformed metadata 0.0000% (D1) / 0.0000% (D2).

**Verdict: STRONG PASS** (per E1-E_task.md section 24, reported as classification language only;
no result was adjusted to fit it).

E1-E reaches `global_step_120` from the original GAP `global_step_60` through a single
pre-registered mechanism change, with the runtime semantics verified step by step against the
frozen E1-E0 design (section 11 selftest: q=0 and q=2 planner replay exact on 50/50 steps of the
E1-D stream, runtime replay through the actual reward manager, 13 named invariants, plus the
PART D empty-keep regression), with a passing D1 -> D2 automatic gate, and with the original
GAP/verl source provably byte-identical before and after (section 31).

The scientific result is a genuine Pareto improvement over the experiment E1-E was designed to
fix. At step120, versus **E1-D120** (the shaping baseline), E1-E has micro EM **0.4395 vs
0.4386 (+0.09 pp)** and search rounds **1.5203 vs 1.7234 (-11.78 %)**; the paired test over
49,947 aligned prompts puts the accuracy difference at +0.09 pp with **p = 0.568** (i.e. the two
policies are statistically indistinguishable on accuracy) while rounds fall **-11.85 % with
p < 1e-300** and 10,285 prompts improve versus 3,913 that worsen. Versus **GAP120** the
pre-registered engineering criteria are both met: **Delta micro EM = -0.47 pp** (inside the
-0.5 pp band) and **Delta rounds = -17.99 %** (well beyond the -10 % target), though the paired
test shows the small accuracy deficit is real (p = 0.0022) rather than noise. Versus **E1-B120**
E1-E gives up 17.4 % of E1-B's round reduction but recovers 0.69 pp of micro EM (paired
p = 1.4e-05), i.e. it buys back most of the accuracy that the pure correctness filter destroyed.

The mechanism behaved as frozen: q = 2, lambda = 0.05, mixed-group reward exactly the original
EM (0 violations in 2,560 trained rollouts, and `sum_reward - sum_em = 0.05 x sum_eff` exactly),
quota-selected all-correct groups rewarded in [1, 1.05] with a strictly positive cost gap for
117/117 inserted groups, no NaN/Inf advantage, no zero-search shortcut, no malformed burst and no
query explosion; the observed replacement rate was 6.09 % against the 6.25 % predicted offline,
with a 95 % full-quota rate explained by `m = min(q, |cache|)` plus one recorded cap shortfall.

## 2. Motivation from E1-C / E1-D / E1-E0

E1-B applied an efficiency-shaped reward `R = A · (1 + 0.05 · E)` to **both** the GRPO objective
**and** the group filter, so it changed not only *how* retained groups were ranked but also *who
trained*. Relative to GAP step120 it cut search rounds strongly — rounds 1.85373 → 1.29497
(**−30.14 %**) — at a real accuracy cost: micro EM 0.4442 → 0.4325 (**−1.16 pp**).

Source: `E1-B/FINAL_REPORT.md` (lines 304–305); `E1-D/FINAL_REPORT.md` (lines 12–13).

E1-C, replaying the same candidate pools offline, showed that the E1-B round reduction decomposes
into two separable channels: of `−0.5596` rounds, `−0.1298` (**23.2 %**) comes from the
**objective** channel and `−0.4298` (**76.8 %**) from the **filter** channel (the shaped filter
admitting all-correct efficiency groups and thereby reallocating optimizer-batch slots away from
mixed groups). E1-C therefore named two designs that separate the channels: **Candidate A**
(filter-decoupled reward — keep the original GAP correctness filter, accept that only ≈1.3 % of
retained groups are efficiency-active) as primary, and **Candidate B** (baseline-preserving +
capped quota of all-correct efficiency groups) as the designed fallback. Changing λ cannot remove
the filter-induced membership shift, so λ was ruled out as the lever.

Source: `E1-C/FINAL_REPORT.md` (lines 368–397, 501–525); `E1-B/FINAL_REPORT.md` (line 13);
`E1-D/FINAL_REPORT.md` (lines 27–28).

E1-D implemented Candidate A: it kept the original `metric=em` filter but kept the shaped reward
on the retained groups. That recovered accuracy but gave back most of the round reduction —
vs GAP120, rounds 1.857 → 1.723 (**−6.99 %**) with micro EM 0.4450 → 0.4392 (**−0.57 pp**), and
vs E1-B120 rounds **+33.1 %** with micro EM **+0.61 pp**. The mechanism is visible as supervision
density: the EM filter left only **86/1679 = 5.1 %** efficiency-active groups.

Source: `E1-D/FINAL_REPORT.md` (lines 22–33); `E1-E0/FINAL_REPORT.md` (lines 53–56).

E1-E0 then quantified the missing third design offline, on the real E1-D D2 candidate stream
(669,440 rollouts, 83,680 n=8 groups, 523 generation batches, 50 optimizer steps), replaying the
original GAP stopping rule (only mixed-correctness groups count toward `data.train_batch_size = 32`)
while caching all-correct efficiency groups in stream order. At **q = 2** the quota fills on
**50/50 = 100 %** of steps (mean 2.00 efficiency groups/step, replacement fraction **6.25 %**), the
displaced mixed groups show **no difficulty bias** (k/8 p = 0.94) and **no source bias** (χ² p = 0.92),
and the inserted groups carry a real cost signal. E1-E0 recommended **q = 2**, with q = 3 as the
pre-registered ablation and q = 4 / unrestricted (E1-B-style ≈25.19 % replacement) rejected as too
disruptive, and proposed a quota-gated, reward-separated configuration: keep the mixed-group
objective exactly at the original EM and admit at most q already-solved efficiency groups per
32-group batch.

Source: `E1-E0/FINAL_REPORT.md` (lines 15–48, 336–361); `E1-E0/results/recommended_quota.json`;
`E1-E0/results/quota_availability.csv`.

E1-E is the formal online RL execution of that frozen E1-E0 design; it changes only the
group-selection/reward-gating channel, not the correctness objective.

Source: `E1-E/README.md` (lines 1–34).

---

## 3. Exact preregistered design

Design frozen by E1-E0 before any GPU work; not swept in E1-E.

| element | frozen value |
|---|---|
| quota `q` | **2** |
| `lambda` | **0.05** (never swept) |
| mixed group | `1 ≤ k ≤ 7` (k = correct rollouts out of n = 8); reward `R_i = A_i`, **no shaping** |
| efficiency group | `k = 8` **and** `max(C) > min(C)` over the group's correct rollouts; else excluded |
| excluded groups | `k = 0` (all-wrong) and `k = 8` without correct-rollout cost variation; dropped by the filter, never trained on |
| efficiency-group reward | `R_i = 1 + 0.05 · E_i`, `E_i = (Cmax − C_i)/(Cmax − Cmin)` over the group's correct rollouts |
| cost definition | `C_i = logical_search_batches` |
| stopping rule | only mixed groups advance the generation stopping counter to `B = 32`; efficiency groups are **cached in stream order, capped at q**, and never advance it |
| optimizer batch | at the stopping generation batch: `mixed[:B−m] + cache[:m]`, `m = min(q, |cache|)`; the last `m` mixed groups of the q = 0 batch are displaced |
| selection | pure stream order — no difficulty / reward / source / length selection |
| no extra generation | generation is never extended to fill `q` |
| `acc` | restored to the **true EM** after the reward manager runs |
| validation | original GAP scorer `mhqa_eval.py:compute_score_em_batch`, unchanged |

Source: `E1-E/E1-E_task.md` (lines 49–190, 328–367);
`E1-E/code/e1e_quota_plan.py` (module docstring, lines 1–29);
`E1-E/code/e1e_scorer.py` (lines 1–23); `E1-E/README.md` (lines 15–30);
`E1-E0/FINAL_REPORT.md` (lines 346–359); `E1-E0/results/recommended_quota.json`
(`mixture_rule`).

---

## 4. Scientific control variables

All training/evaluation controls are held identical to GAP/E1-B (`E1-E/scripts/run_e1e_train.sh`
and `E1-B/scripts/run_e1b_train.sh` set the same values):

| control | value | source |
|---|---|---|
| actor lr | `1e-6` | `run_e1e_train.sh:62`, `:258` |
| lr_warmup_steps | `3` | `run_e1e_train.sh:259` |
| weight_decay | `0.1` | `run_e1e_train.sh:260` |
| train_batch_size (B) | `32` | `run_e1e_train.sh:63`, `:246` |
| gen_batch_size | `160` | `run_e1e_train.sh:65`, `:247` |
| ppo_mini_batch_size | `32` | `run_e1e_train.sh:64`, `:261` |
| rollout n | `8` | `run_e1e_train.sh:67`, `:282` |
| advantage estimator | GRPO with `norm_adv_by_std_in_grpo` | `run_e1e_train.sh:241`; `verl/verl/trainer/config/ppo_trainer.yaml:826` (`true`); `E1-E_task.md:786` |
| KL loss | off (`use_kl_loss=false`, `kl_loss_coef=0.0`) | `run_e1e_train.sh:268–270` |
| DAPO clip | `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `clip_ratio_c=10.0` | `run_e1e_train.sh:70–71`, `:271–273` |
| epochs | `2` | `run_e1e_train.sh:66`, `:297` |
| temperature | `1.0` | `run_e1e_train.sh:283` |
| max turns | `8` | `run_e1e_train.sh:303` |
| tool config | same wiki tool config `verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml` | `run_e1e_train.sh:92`, `:306` |
| save_freq / test_freq | `10` / `10` | `run_e1e_train.sh:110–111`, `:293–294` |
| GPUs | `4` (1 node, `n_gpus_per_node=4`) | `run_e1e_train.sh:76`, `:291–292` |
| optimizer / scheduler | same optimizer and scheduler as GAP/E1-B (neither launcher sets any optimizer/scheduler flag beyond lr, warmup, weight decay) | `run_e1e_train.sh:258–260`; `run_e1b_train.sh:238–240` |
| full-state resume | `resume_mode=resume_path` from the full-state checkpoint | `run_e1e_train.sh:300–301` |
| validation scorer | identical original `mhqa_eval.py:compute_score_em_batch` | `run_e1e_train.sh:310–311`; `runs/D1_20260914_093225/config/reward_config.json` |

**Only scientific change vs E1-B** is the group-selection / reward-gating mechanism (quota-gated,
reward-separated selection instead of E1-B's shaped reward + shaped filter).
**Only scientific change vs E1-D** is that mixed groups are no longer shaped: E1-D kept the
shaped reward `R = A·(1+0.05E)` on every retained group, whereas E1-E gives mixed groups the
unshaped original `R_i = A_i`.

Source: `E1-E/logs/issues_and_fixes.md` (lines 88–90); `E1-E/README.md` (lines 24–30);
`E1-E/scripts/run_e1e_train.sh` (lines 8–18); `E1-D/FINAL_REPORT.md` (lines 49, 63).

---

## 5. Code reuse / copied-from provenance

The E1-E tree was built by copying the E1-D implementation modules and renaming `e1d_*` → `e1e_*`,
as recorded in the scaffold audit. Eleven renamed modules exist (the ten `e1d_*.py` runtime modules
that have `e1e_*` counterparts, plus the `pythonpath_e1d/sitecustomize.py` shim now at
`pythonpath_e1e/sitecustomize.py`): `checkpoint_patch`, `cost`, `diag_patch`, `hang_diagnostics`,
`reward_manager`, `scorer`, `sglang_gpu_patch`, `stall_watchdog`, `state`, `trainer_patch`, and
`sitecustomize`. Two further renamed copies (the E1-D self-tests) are parked, not used. E1-E also
adds two genuinely new modules with no E1-D counterpart: `code/e1e_quota_plan.py` (the pure planner)
and `code/e1e_runtime_selftest.py` (the §11 gate).

Source: `E1-E/logs/issues_and_fixes.md` (lines 26–31); `E1-D/code/` vs `E1-E/code/` listings;
`E1-E/legacy_e1d_unused/README.md` (lines 1–6).

- **E1-0 runtime fixes come from E1-0 via E1-D copies.** `sitecustomize.py` loads the fixes
  root-caused and validated in E1-0, via E1-D's copies: hang diagnostics
  (`e1e_hang_diagnostics`), FSDP `load_checkpoint(..., map_location="cpu")`
  (`e1e_checkpoint_patch`), per-rank SGLang GPU visibility with the CUDA-IPC patch applied lazily
  inside the WorkerDict actor (`e1e_sglang_gpu_patch`), and passive worker prints
  (`e1e_diag_patch`). E1-0's own files are untouched.
  Source: `E1-E/code/pythonpath_e1e/sitecustomize.py` (lines 11–20).
- **Evaluation reuses the ORIGINAL GAP evaluation implementation unmodified**:
  `Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh` is sourced as-is; only the
  checkpoint path / experiment name / label change.
  Source: `E1-E/scripts/e1e_eval_common.sh` (lines 2–3, 27–33);
  `E1-E/manifests/original_files_manifest_pre.json` (entry for that file).
- **Validator/scorer path is the original** `verl/verl/utils/reward_score/mhqa_eval.py`
  (`compute_score_em_batch`).
  Source: `E1-E/runs/D1_20260914_093225/config/reward_config.json` (`val_scorer`);
  `E1-E/scripts/run_e1e_train.sh` (lines 310–311).
- **The original GAP/verl core source is read-only and unmodified.** Pre-existing hashes for 23
  original files (including `ray_trainer.py`, `batch.py`, `mhqa_train.py`, `mhqa_eval.py`,
  `core_algos.py`, the evaluation script and `environment.sh`) were recorded before launch; no core
  file was edited.
  Source: `E1-E/manifests/original_files_manifest_pre.json`; `E1-E/logs/preflight.log` (line 58,
  "wrote … original_files_manifest_pre.json (23 files)"); `E1-E/logs/issues_and_fixes.md` (lines 26–31).
- **`E1-E/legacy_e1d_unused/README.md`** documents two parked E1-D self-test copies
  (`e1e_selftest.py`, `e1e_filter_selftest.py`) that are **NOT used**: they test E1-D semantics
  (`R_i = A_i·(1+0.05E_i)` for *every* retained group, and the E1-D `metric=em` shaped-revival
  filter), so running them would both mis-assert E1-E semantics and overwrite the real
  `E1-E/analysis/preflight_selftest.{md,json}` produced by the actual E1-E gate. The E1-D originals
  remain untouched in `E1-D/code/`; E1-E's §11 gate is `code/e1e_runtime_selftest.py`.
  Source: `E1-E/legacy_e1d_unused/README.md` (lines 1–22).

---

## 6. Exact code modifications

The E1-E-specific code is:

| file | role |
|---|---|
| `code/e1e_quota_plan.py` | pure planner shared by runtime and offline selftest; `classify_group(k, n_rollouts, cost_gap_correct)` and `plan_batch(cur_groups, state, target_batch, quota)`; minimal-state mixed-count/cache/cap logic, `eff_shortfall` accounting |
| `code/e1e_scorer.py` | per-group scorer `compute_score_em_efficiency_batch`; publishes the filter metric `e1e_quota_metric` (mixed group → EM; efficiency group → `1+0.05E`; else constant 0.0), re-exports the original `compute_score_em`/`compute_score_em_batch`, keeps `em` as the true EM |
| `code/e1e_reward_manager.py` | registered reward manager `e1e_quota_gated` (subclass of the original `BatchRewardManager`); classifies each generation batch, runs the shared planner, materialises the quota batch in place (`mixed[:B−m] + cache[:m]`, with earlier-batch cached slices concatenated in), restores `data.batch["acc"]` to the true EM, and writes per-rollout/per-group/per-call/per-step diagnostics |
| `code/pythonpath_e1e/sitecustomize.py` | PYTHONPATH-injection/registration shim: registers `e1e_quota_gated` in the unmodified `verl` registry and loads the E1-0 runtime fixes |

Source: `E1-E/code/e1e_quota_plan.py`; `E1-E/code/e1e_scorer.py`;
`E1-E/code/e1e_reward_manager.py`; `E1-E/code/pythonpath_e1e/sitecustomize.py`;
`E1-E/README.md` (lines 36–53).

**Six scaffolding defects fixed before any GPU work** (full scaffold audit in
`E1-E/logs/issues_and_fixes.md`, lines 26–76):

1. **`pythonpath_e1d` still on `PYTHONPATH` in the launcher.**
   `E1-E/scripts/run_e1e_train.sh` listed `pythonpath_e1d` (and it was in the launcher's own
   preflight input list). If unfixed, `sitecustomize.py` would not be imported, `e1e_quota_gated`
   would never be registered, and `main_ppo` would die at start-up — the experiment could not run
   at all. Fixed to `pythonpath_e1e`.
2. **`E1E_FILTER_METRIC`/provenance JSON still describing E1-D semantics.**
   `run_e1e_train.sh` still exported `E1E_FILTER_METRIC="em"` and the generated
   `config/reward_config.json` still described E1-D semantics (`R_i = A_i*(1+0.05E_i)` for every
   group, `filter_metric=em`, manager `e1e_batch_shaped`). The command line was already correct
   (`metric=e1e_quota_metric`, `reward_manager=e1e_quota_gated`), but the provenance file would have
   misdocumented the frozen design. Fixed: metric, manager name, reward formula, reward-name and
   comments now state the E1-E semantics (mixed group `R_i = A_i` with no shaping; quota-selected
   k=8 efficiency group `R_i = 1 + 0.05*E_i`).
3. **Lost E1-D baselines in the analysis driver.**
   After the rename, `E1D120_DUMP` pointed at E1-E's own future eval dir, so the mandated four-way
   comparison (GAP120 / E1-B120 / E1-D120 / E1-E120) would have compared E1-E with itself and
   dropped E1-D entirely. Fixed: `E1E70_DUMP`/`E1E120_DUMP` for E1-E's own evals, plus a new
   `E1D120_DUMP` reading
   `experiments/DAPO-GAP3B-MHQA-Agent-E1D-eval-step120-4gpu/val_generations/0.jsonl`; added paired
   comparisons `e1e120_vs_e1d120` and `e1e120_vs_e1e70` (and relabelled the rest `e1e*`); added a
   symlink loop over `E1-D/eval_results/step*_*` so E1-D's step70/step120 evaluations enter the
   comparison matrix read-only.
4. **Stale E1-D integrity check replaced.**
   `E1-E/scripts/e1e_integrity.py` still ran the E1-D check `check_filter_decoupling()`, which reads
   `groups_pid*.jsonl` fields (`baseline_keep_em_std_gt_0`, `shaped_keep_reward_std_gt_0`,
   `all_correct`, …) that the E1-E reward manager does not write. Replaced with
   `check_quota_semantics()` + `check_log_vs_diagnostics()`: the full section 11/12 invariant set
   measured on E1-E's own `quota_steps/quota_calls/quota_groups/rollouts` diagnostics (32 groups per
   step, `quota_actual <= 2`, mixed-only stopping, `mixed reward == EM` per rollout, efficiency
   reward in [1,1.05] with `em == 1`, wrong reward exactly 0, published metric == EM or 1+0.05E,
   advantage NaN/Inf == 0, no quota error lines, fill rate / shortfall / displaced counts per step)
   **plus** an independent cross-check of the trainer's own logged `critic/score/max` against the
   quota diagnostics (1.05 exactly on steps with an insertion, <= 1.0 on steps without).
5. **Stale E1-D self-tests parked.**
   `E1-E/code/e1e_filter_selftest.py` and `e1e_selftest.py` were E1-D-semantics copies that wrote to
   the same output paths as the real E1-E gate. They were moved (not deleted) to
   `E1-E/legacy_e1d_unused/` with a README explaining why they cannot be used; the E1-D originals
   remain untouched in `E1-D/code/`. `run_e1e_all.sh` now gates on `e1e_runtime_selftest.py` instead.
6. **Missing section-15 automatic gate script.**
   E1-E_task.md section 15 requires an unattended D1 → D2 decision. The copied chain had no gate
   script at all. Added `E1-E/scripts/e1e_d1_gate.py` implementing the section 15 hard conditions
   (integrity fatal patterns, quota semantics, checkpoint completeness, evaluation completeness,
   micro-EM consistency drop > 1.0 pp vs E1-0 step70 and E1-D step70,
   query-explosion-with-rounds-decrease, mass zero-search, malformed/no-answer burst) while
   explicitly **not** stopping on a 1–3 % step70 efficiency gain. Wired into `run_e1e_all.sh`
   (`stage_gate`) and `e1e_pipeline.sh`.

Cosmetic renames in the same pass: `e1e_eval_common.sh` exports `EVAL_MODEL_KIND="e1e"`,
`e1e_paired_eval_compare.py` takes `--e1e-dump` (`--e1d-dump` kept as an alias), and the
`sitecustomize.py` docstring path was corrected.

All fixes are bug repairs in the E1-E scaffolding or in provenance/reporting; the pre-registered
design, λ, q, reward formulas, batch sizes, stopping rule and learning-rate/batch configuration were
untouched (“Scientific variable changed: no”).

Source: `E1-E/logs/issues_and_fixes.md` (lines 26–90).


### 6.1 Defects found and fixed during the first D1 launch (2026-09-14, before any result was used)

A first D1 launch (09:32-11:51, archived as `E1-E/runs/INVALID_D1_20260914_093225/`) exposed three
further defects in the E1-E reward manager. The run was stopped at optimizer step 69/70 and the
whole chain was restarted from the original GAP step-60 full state; none of the events below
changed the pre-registered design, q = 2, lambda = 0.05 or any training hyper-parameter.

1. **Leaked-variable group-id lists.** `_record` built `inserted_efficiency_group_ids` and
   `mixed_selected_group_ids` with a comprehension referencing `gid` (a variable left over from
   the preceding loop) instead of `g["gid"]`, so both lists repeated one id. Diagnostics only;
   the recorded lengths stayed correct. Fixed to `g["gid"]`.
2. **Empty advantage diagnostic at stopping calls.** The advantage list was built with
   `for j, s in zip(g["idx"], vals)`; for an injected efficiency group `g["idx"]` is empty, so
   `zip` yielded nothing and every stopping-call record had `advantage_*`/`reward_*`/`em_mean`
   null. Diagnostics only (the trainer computes the real advantage from the shaped
   `token_level_rewards`). Fixed by iterating `vals` directly.
3. **Empty-keep batches leaked the raw generation batch to the trainer (design violation).**
   When the frozen plan selected nothing from a generation batch (all candidates all-correct
   without cost variation / all-wrong, or the mixed cap already reached), the manager returned
   without mutating the `DataProto`, so the trainer ran its own `std > 0` filter on the raw 160
   candidate groups and accumulated uncached efficiency groups, which also advanced the
   mixed-only stopping counter. This is exactly E1-E_task.md section 15's hard stop condition
   ("quota changes generation stopping"). Direct evidence: calls returning 1280 rollouts
   (160 groups x n=8) with no matching call record at D1 steps 63, 64, 65, 67 and 68; step 67's
   optimizer batch was 27 mixed + 5 unbudgeted efficiency groups instead of 30 mixed + 2 cached
   efficiency groups. The E1-E0 candidate stream contains such a batch in 18 of 50 steps, so the
   defect would have contaminated roughly a third of D2.

Fix (in `code/e1e_reward_manager.py`, no design change): the efficiency-group stash now happens
before the empty-keep test; an empty keep set sets `self._keep_nothing`, is recorded as a
`kept_nothing` call, and `__call__` replaces the **published** filter metric
(`reward_extra_info["e1e_quota_metric"]`) with a constant so the trainer's `std > 0` filter drops
every group of that batch -- the accumulation and the stopping counter are provably untouched.
The section-11 gate was extended with a new **PART D** regression test (see section 10) and
`E1-E/scripts/e1e_integrity.py` gained independent runtime invariants: the number of manager calls
served (distinct `(global_step, timestamp)` groups in `rollouts_pid*.jsonl`, matched with a 3 s
tolerance because the call and rollout records are timestamped separately) must equal the number
of call records, the per-step `generation_batch_in_step` counter must be gapless, no served batch
may exceed `8 x 32 = 256` rollouts unless it is a `kept_nothing` call, and every other call must
have returned exactly `8 x kept_groups` rollouts. On the archived invalid run those checks report
`all_manager_calls_recorded = false` with 5 unrecorded and 5 oversized calls and
`generation_batch_in_step` gaps at steps 63, 65, 67 and 68, i.e. the automic gate would have
refused D2 even without manual inspection.

Live confirmation on the restarted run (`E1-E/runs/D1_20260914_115858`): steps 61-65 each produced
exactly 32 groups (30 mixed + 2 injected efficiency) with `eff_shortfall = 0`, and the previously
leaking case occurred 3 times and was recorded as `kept_nothing` with 0 unrecorded calls, 0
oversized returned batches and no generation-batch gaps.

Source: `E1-E/logs/issues_and_fixes.md` (D1 restart entry);
`E1-E/code/e1e_reward_manager.py`; `E1-E/analysis/preflight_selftest.json` (`part_d_checks`).

---

## 7. Preflight / Wiki / environment

- **Wiki retriever service probed and REUSED.** The E1-E preflight probed
  `http://127.0.0.1:8008/retrieve`, found it HEALTHY, and reused it: pid **2230350**
  (`pt_main_thread`), `WIKI_REUSED=true`, `WIKI_STARTED=false`, no duplicate server started, nothing
  stopped. A functional retrieval probe returned a real passage (id `11115958`, France). Two
  preflight passes were recorded (2026-09-14T09:27:22 and 09:31:57), both with the same healthy pid.
  Source: `E1-E/logs/preflight.log` (lines 2–7, 69–75); `E1-E/logs/preflight_status.txt`;
  `E1-E/logs/wiki_service.log`.
- **conda env + environment.sh.** `environment.sh` was sourced, then conda env `parallel-agent`
  activated (`/home/nf5468m6/miniconda3/envs/parallel-agent`).
  Source: `E1-E/logs/preflight.log` (lines 8–12).
- **Versions:** python **3.10.14**, torch **2.6.0+cu124** (cuda 12.4), ray **2.43.0**, sglang
  **0.4.6.post5**, transformers **4.51.1**, numpy **1.26.4** (also pandas 2.2.3).
  Source: `E1-E/logs/preflight.log` (lines 15–21).
- **GPUs:** 4× NVIDIA A800 80GB PCIe (81920 MiB each); 0 GAP `main_ppo` / ray / sglang processes
  at preflight (1 wiki process, not killed).
  Source: `E1-E/logs/preflight.log` (lines 23–35).
- **Input integrity verified.** Original GAP step60 FSDP shards (model/optim/extra_state), `data.pt`,
  the RL dataset parquet and val parquet, the wiki tool config, the base SFT model dir, and the E1-E
  scorer/reward-manager/sitecustomize/trainer files all reported OK; `gap_step60 tracker=200`.
  Source: `E1-E/logs/preflight.log` (lines 36–54); `E1-E/runs/D1_20260914_093225/preflight/preflight_train.txt`.
- **Output-collision guards passed.** The E1-E experiment dir
  `experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu`, `E1-E/runs/latest_D1.txt` and
  `E1-E/eval_results/latest_step70.txt` were all ABSENT before launch. `STATUS=OK`.
  Source: `E1-E/logs/preflight.log` (lines 55–59);
  `E1-E/runs/D1_20260914_093225/preflight/preflight_train.txt` (`filter_metric=e1e_quota_metric`,
  `quota_q=2 batch_target=32`, `resume_from_path=.../global_step_60`, `expected_start_step=60`).
- **Pre-manifest of original-file hashes** written to
  `manifests/original_files_manifest_pre.json` (23 files).
  Source: `E1-E/logs/preflight.log` (line 58); `E1-E/manifests/original_files_manifest_pre.json`.

---

## 8. Disk-space status and any cleanup actions

Two recorded actions:

1. **E1-B intermediate checkpoint deletion (before any E1-E training, 2026-09-14 02:35:24 +08:00).**
   Only **233 GB** was free on `/data01` while E1-E needs ~229 GB of its own checkpoints, leaving no
   safety margin. Four redundant intermediate checkpoints of the finished E1-B run were deleted:
   - `experiments/DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu/global_step_80` — 45,103,148,956 B
   - `…/global_step_90` — 45,103,149,028 B
   - `…/global_step_100` — 45,103,149,076 B
   - `…/global_step_110` — 45,103,149,028 B

   (sum 180,412,596,088 B ≈ 168 GB). Free space went **233 → 401 GB**. Post-delete `df` recorded
   `/dev/sdc 1.8T 1.3T 401G 77% /data01`, and E1-B `global_step_120` remained present.
   Safety conditions met: E1-B final step120 complete and untouched; the run is finished so no
   intermediate is needed for resume; the evaluations of those steps (`E1-B/evaluations/step{80,90,100,110}_*`)
   and their statistics (`E1-B/checkpoints/checkpoint_stats.*`, `E1-B/checkpoints/manifest.json`)
   are already saved; E1-B's FINAL_REPORT comparisons were already computed; not part of the current run.

   Source: `E1-E/logs/disk_cleanup.log` (lines 1–28); `E1-E/logs/issues_and_fixes.md` (lines 3–24).

2. **Pre-D2 disk guard found sufficient free space and therefore deleted nothing.**
   `e1e_disk_guard.sh` ran at 2026-09-14T09:36:45 +08:00 with `free_before=401 GB`, `need=250 GB`
   and recorded `STATUS=OK free_gb=401 deleted=none`.
   Source: `E1-E/logs/disk_cleanup.log` (lines 29–31); `E1-E/logs/disk_guard_status.txt`;
   `E1-E/logs/run_all.log`.

**Protected paths, never touched:** original GAP step60/120(/180/200), the base/SFT model, the
E1-0/E1-A step70 comparison checkpoints, E1-B step120, E1-D step70/step120, and all
code / parquet / eval-results / reports / logs.

Source: `E1-E/logs/disk_cleanup.log` (line 3); `E1-E/logs/issues_and_fixes.md` (lines 14–21).

---

## 9. Runtime quota implementation

Everything runs inside the unmodified trainer via `PYTHONPATH=E1-E/code/pythonpath_e1e`
(`sitecustomize.py`), which registers `e1e_quota_gated` and loads the E1-0 runtime fixes.

- **Filter metric.** The custom scorer `e1e_scorer.py:compute_score_em_efficiency_batch` publishes
  `e1e_quota_metric` through the trainer's existing `reward_extra_info → non_tensor_batch` path, so
  `algorithm.filter_groups.metric=e1e_quota_metric` works with the stock `ray_trainer.py`. The metric
  is built so that exactly the kept groups have std > 0: mixed group → `A_i` (0/1), efficiency group
  → `1 + 0.05·E_i` (varies), anything else → constant `0.0` (std == 0). Because the rule is
  `keep(group) iff std > 0`, the trainer therefore keeps exactly the groups E1-E materialises.
  Source: `E1-E/code/e1e_scorer.py` (lines 16–23, 109–135); `E1-E/README.md` (lines 46–53).
- **Why the reward manager must act in place.** The trainer drops filtered-out groups before
  accumulating, so all-correct efficiency groups observed in early generation batches of a step
  would be lost. The reward manager therefore classifies every group in each generation batch (using
  the same scorer the trainer will use), runs the shared pure planner
  `e1e_quota_plan.plan_batch`, stashes each batch's efficiency-group slices when the batch is not the
  stopping batch, and at the stopping generation batch **materialises the quota batch in place** as
  `mixed[:B−m] + cache[:m]` (with cached earlier-batch slices concatenated in via `DataProto.concat`),
  replacing `data.batch` / `data.non_tensor_batch`. It then restores `data.batch["acc"]` to the true
  EM (from `reward_extra_info["em"]`) and appends per-rollout (`rollouts_pid*.jsonl`), per-group
  (`quota_groups_pid*.jsonl`), per-call (`quota_calls_pid*.jsonl`) and per-step
  (`quota_steps_pid*.jsonl`) diagnostics; failures are logged to `quota_errors_pid*.log`.
  Source: `E1-E/code/e1e_reward_manager.py` (lines 6–26, 136–212, 306–387);
  `E1-E/README.md` (lines 49–53).
- **Documented deviation mechanism.** The upstream trainer accumulates kept groups across generation
  batches and truncates to the first B at the end, so a group already accumulated cannot be removed.
  To guarantee the running count never reaches B before the mixed-only stopping batch, the planner
  caps the **mixed** groups it keeps at `B − min(q, |cache before this batch|)`. The cap is binding
  only when more than `B − q` mixed groups arrive before the final (stopping) batch; when the cache
  grows during the stopping batch the batch can end up with slightly more mixed / fewer efficiency
  groups than the ideal. Every step records this as `eff_shortfall` rather than silently.
  Source: `E1-E/code/e1e_quota_plan.py` (lines 19–28, 82–132);
  `E1-E/code/e1e_reward_manager.py` (line 269).
- **Diagnostic logs:** `quota_calls_pid*.jsonl`, `quota_steps_pid*.jsonl`,
  `quota_groups_pid*.jsonl`, `rollouts_pid*.jsonl`, `quota_errors_pid*.log`.
  Source: `E1-E/code/e1e_reward_manager.py` (lines 70–76).

---

## 10. Offline selftest vs E1-E0

The offline selftest was a **hard gate**: if it had not matched E1-E0, training would not have
started (`run_e1e_all.sh` `stage_preflight` requires `all_pass` before any GPU work).

Source: `E1-E/analysis/preflight_selftest.json` (`all_pass: true`, `pass` all true);
`E1-E/scripts/run_e1e_all.sh` (lines 15–34); `E1-E/E1-E_task.md` (lines 756–757).

**PART A — planner replay vs E1-E0.** The shared planner (`e1e_quota_plan`) was run over the real
E1-D D2 candidate stream and compared group-by-group with
`E1-E0/results/optimizer_batch_members_by_quota.csv`:
- q = 0 exact steps: **50/50**
- q = 2 exact steps: **50/50**

**PART B — runtime manager replay vs E1-E0.** Synthetic `DataProto` generation batches (real
uids/costs/EMs from the same stream) were pushed through the **real `E1EQuotaRewardManager`**, and
the trainer's own filter/accumulate/truncate loop was reproduced; the materialised batches matched
E1-E0 on batch size and group set for the steps exercised:

| step | generation batches used (`calls_used`) | n_groups | same size | same set |
|---|---|---|---|---|
| 71 | 9 | 32 | yes | yes |
| 72 | 13 | 32 | yes | yes |
| 73 | 12 | 32 | yes | yes |

`all_same_size: true`, `all_same_set: true`.

**PART C — named invariants** (all `true`):
`optimizer_batch_size_all_32`, `quota_actual_le_2`, `inserted_groups_k8`,
`inserted_groups_cost_variation_positive`, `stopping_invariant_across_q_batch_sizes`,
`stopping_batch_independent_of_q`, `mixed_only_stopping`, `mixed_group_reward_equals_em`,
`efficiency_group_reward_in_1_1.05`, `wrong_rollout_reward_zero`, `group_reward_std_finite`,
`no_nan_inf_rewards`, `efficiency_ranking_correct` (13 invariants covering all 12 mandated
section 11 items).

**Section 11 requirement → check mapping (copied from the selftest markdown / JSON `section11_mapping`):**

| E1-E_task.md section 11 requirement | check | result |
|---|---|---|
| 1. every optimizer batch = 32 groups | `optimizer_batch_size_all_32` | PASS |
| 2. q <= 2 | `quota_actual_le_2` | PASS |
| 3. mixed-only stopping | `mixed_only_stopping` | PASS |
| 4. quota does not change generation stopping | `stopping_invariant_across_q_batch_sizes` | PASS |
| 4b. stopping generation batch identical for q=0..4 | `stopping_batch_independent_of_q` | PASS |
| 5. inserted groups have k = 8 | `inserted_groups_k8` | PASS |
| 6. inserted groups cost variation > 0 | `inserted_groups_cost_variation_positive` | PASS |
| 7. mixed group reward == EM | `mixed_group_reward_equals_em` | PASS |
| 8. inserted group reward in [1, 1.05] | `efficiency_group_reward_in_1_1.05` | PASS |
| 9. wrong rollout reward == 0 | `wrong_rollout_reward_zero` | PASS |
| 10. every group reward std finite | `group_reward_std_finite` | PASS |
| 11. efficiency rollout ranking correct | `efficiency_ranking_correct` | PASS |
| 12. no NaN / Inf | `no_nan_inf_rewards` | PASS |

Source: `E1-E/analysis/preflight_selftest.md` (lines 1–53);
`E1-E/analysis/preflight_selftest.json` (`part_a`, `part_b`, `part_c`, `section11_mapping`);
`E1-E/logs/issues_and_fixes.md` (lines 82–86).


### 10.1 PART D -- empty-keep neutrality (added after the first D1 launch)

`E1-E/analysis/preflight_selftest.json` -> `part_d_checks` (all PASS):

| check | meaning |
|---|---|
| `D1_inserted_no_keep_batch_is_neutral` | a synthetic no-keep generation batch inserted into real steps 71-76 leaves the materialised optimizer batch **identical** to the same step without the insertion |
| `D1_optimizer_batch_still_32_groups` | ... and that batch is still exactly 32 groups |
| `D2_real_stream_contains_empty_keep_batches` | the E1-E0 candidate stream really does contain empty-keep batches (18 of 50 steps for q = 2) |
| `D2_empty_keep_steps_match_e1e0` | running the real steps that contain an empty-keep batch (74, 75, 76) reproduces E1-E0's frozen membership exactly |
| `D2_empty_keep_steps_batch_32` | those steps' optimizer batches are exactly 32 groups |
| `D3_kept_nothing_calls_recorded` | the manager writes the `kept_nothing` diagnostic records |

Source: `E1-E/analysis/preflight_selftest.json` (`part_d`, `part_d_checks`).

---

## 11. D1 step60->70 training integrity

- phase: `D1`; expected steps 60->70; integrity verdict: **PASS** (`integrity_ok = True`).
- training log: 11 steps present (first 60, last 70); missing: []; `all_steps_present = True`; resume line 'Setting global step to 60' present: True.
- metrics parsed: 1240; non-finite metric values: {} (none).
- fatal-error scan (before the final-validation marker): cuda_oom=0, nccl_error=0, nccl_warn=12, ray_worker_died=0, wiki_search_error=0, wiki_tool_exec_failed=0, traceback=0, cuda_error=0.
- fatal patterns (>0): none.
- checkpoint `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_70`: exists = True, `data.pt` = True, shards present 12/12, tracker = 120, `tracker_reached_end_step` = True.
- resume source (`/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60/actor/extra_state_world_size_4_rank_0.pt`): `lr_scheduler.last_epoch` = 60 (expected 60 = start step), rng present = True.
- end checkpoint (`/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_70/actor/extra_state_world_size_4_rank_0.pt`): `lr_scheduler.last_epoch` = 70 (expected 70 = end step), rng present = True; `state_resume_ok = True`.
- runtime quota semantics: `quota_semantics_ok = True`, quota error lines = 0.
- trainer log vs diagnostics: steps cross-checked = 10, `critic/score/max` mismatches = 0, consistent = True.

## 12. Runtime quota statistics step61-70

Per-optimizer-step runtime quota diagnostics for steps 61-70.

| aggregate | value |
|---|---|
| step records | 10 |
| generation calls | 121 |
| group records | 320 |
| rollout records | 8960 |
| steps covered | [61, 62, 63, 64, 65, 66, 67, 68, 69, 70] |
| steps missing | [] |
| `all_steps_recorded` | True |
| quota fill rate (mean `quota_actual` / 2) | 100.0000% (1.0000) |
| steps at full quota | 10 / 10 |
| total inserted efficiency groups | 20 |
| total displaced mixed groups | 30 |
| total `eff_shortfall` | 0 |
| steps needing injection | 10 |
| generation batches (total) | 121 |
| generation batches (mean/step) | 12.1000 |
| efficiency groups seen | 20 |
| mixed groups seen | 300 |
| zero-search rollouts | 0 (0.0000%) |
| malformed-metadata rollouts | 0 (0.0000%) |
| advantage NaN / Inf | 0 / 0 |
| quota error lines | 0 |
| steps with wrong batch size | [] |
| calls with `quota_actual` > 2 | 0 |
| steps without exactly one stopping call | [] |
| `quota_semantics_ok` | True |

| step | generation batches | quota actual | mixed kept cum | efficiency groups seen | cache size | eff_shortfall | inserted | optimizer batch groups | advantage mean | advantage std |
|---|---|---|---|---|---|---|---|---|---|---|
| 61 | 12 | 2 | 30 | 7 | 2 | 0 | 2 | 32 | 0.0000 | 0.9555 |
| 62 | 11 | 2 | 30 | 2 | 2 | 0 | 2 | 32 | 0.0000 | 0.9555 |
| 63 | 10 | 2 | 30 | 3 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |
| 64 | 12 | 2 | 30 | 10 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 65 | 12 | 2 | 30 | 12 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 66 | 14 | 2 | 30 | 11 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 67 | 14 | 2 | 30 | 5 | 2 | 0 | 2 | 32 | 0.0000 | 0.9473 |
| 68 | 9 | 2 | 30 | 5 | 2 | 0 | 2 | 32 | 0.0000 | 0.9555 |
| 69 | 14 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |
| 70 | 13 | 2 | 30 | 11 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |

## 13. step70 seven-benchmark evaluation

- artifact: `/data01/wyy/Graph-Agent-Planning/E1-E/eval_results/step70_20260914_151446/eval_summary.json`; label = e1e_step70; total samples = 51201.
- macro EM = 0.3966; micro EM = 0.4339.

| benchmark | nq | triviaqa | popqa | hotpotqa | 2wikimultihopqa | musique | bamboogle |
|---|---|---|---|---|---|---|---|
| EM | 0.3780 | 0.5650 | 0.3990 | 0.3980 | 0.4420 | 0.1700 | 0.4240 |

- behaviour: rounds 1.6159, queries 2.1152, queries/round 1.3089, parallel factor 1.3089, parallel sample rate 43.6261%, zero-search 0.0391%, no-answer 0.6191%, response tokens 323.78.
- runtime mechanism: quota fill rate = 100.0000%; steps at full quota = 10.0000; steps used = 10.0000; inserted efficiency groups = 20.0000; displaced mixed groups = 30.0000; total eff_shortfall = 0.0000; mean generation batches/step = 12.1000; train zero-search share = 0.0000%; train malformed share = 0.0000%; quota_semantics_ok = 1.0000; quota error lines = 0.0000.

## 14. D1 gate decision

- gate: **PASS**
- decision: start D2 (step70 -> step120)
- stop reasons (0): none
- warnings: step70 efficiency effect is expected to be small (~6.25% of groups); a 1-3% gain is NOT a stop condition

| check | value |
|---|---|
| `A_integrity_ok` | True |
| `A_checkpoint_complete` | True |
| `A_state_resume_ok` | True |
| `B_quota_semantics_ok` | True |
| `B_quota_error_lines` | 0 |
| `B_batch_size_32_all_steps` | True |
| `B_quota_actual_le_2` | True |
| `B_mixed_only_stopping` | True |
| `B_mixed_reward_equals_em` | True |
| `B_efficiency_reward_in_range` | True |
| `B_wrong_reward_zero` | True |
| `B_no_nan_inf_advantage` | True |
| `B_published_metric_consistent` | True |
| `B_log_score_max_matches_quota` | True |
| `B_runtime_matches_e1e0_offline_selftest` | True |
| `quota_fill_rate` | 1.0 |
| `steps_full_quota` | 10 |
| `steps_used` | 10 |
| `total_inserted_efficiency_groups` | 20 |
| `total_eff_shortfall` | 0 |
| `zero_search_share_train` | 0.0 |
| `malformed_share_train` | 0.0 |
| `D_eval_dir` | /data01/wyy/Graph-Agent-Planning/E1-E/eval_results/step70_20260914_151446 |
| `E_micro_em_delta_vs_e1_0_step70` | 0.0013701294896584226 |
| `E_micro_em_delta_vs_e1d_step70` | 0.0005819612898185778 |
| `E_query_delta_rel_vs_ref` | -0.006048257569500004 |
| `E_rounds_delta_vs_ref` | -0.013613015370793402 |
| `E_zero_search_rate_step70` | 0.00039061737075447747 |
| `E_zero_search_rate_ref` | 0.00031249389660358197 |
| `E_no_answer_tag_rate_step70` | 0.006191285326458467 |
| `E_no_answer_tag_rate_ref` | 0.0074412609128727955 |

Reference step70 overall metrics used by the gate:

| run | n samples | macro EM | micro EM | rounds | queries | parallel factor | zero-search | no-answer |
|---|---|---|---|---|---|---|---|---|
| `e1_0_step70` | 51201 | 0.4006 | 0.4325 | 1.6296 | 2.1280 | 1.3059 | 0.0312% | 0.7441% |
| `gap_step60` | 51201 | 0.4007 | 0.4351 | 1.5889 | 2.0922 | 1.3167 | 0.0273% | 0.5801% |
| `e1a_step70` | 51201 | 0.3984 | 0.4317 | 1.5977 | 2.1036 | 1.3166 | 0.0293% | 0.5683% |
| `e1d_step70` | 51201 | 0.3976 | 0.4333 | 1.6237 | 2.1274 | 1.3102 | 0.0391% | 0.7246% |

## 15. D2 step70->120 training integrity

- phase: `D2`; expected steps 70->120; integrity verdict: **PASS** (`integrity_ok = True`).
- training log: 51 steps present (first 70, last 120); missing: []; `all_steps_present = True`; resume line 'Setting global step to 70' present: True.
- metrics parsed: 6165; non-finite metric values: {} (none).
- fatal-error scan (before the final-validation marker): cuda_oom=0, nccl_error=0, nccl_warn=12, ray_worker_died=0, wiki_search_error=0, wiki_tool_exec_failed=0, traceback=0, cuda_error=0.
- fatal patterns (>0): none.
- checkpoint `/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_120`: exists = True, `data.pt` = True, shards present 12/12, tracker = 120, `tracker_reached_end_step` = True.
- resume source (`/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_70/actor/extra_state_world_size_4_rank_0.pt`): `lr_scheduler.last_epoch` = 70 (expected 70 = start step), rng present = True.
- end checkpoint (`/data01/wyy/Graph-Agent-Planning/experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/global_step_120/actor/extra_state_world_size_4_rank_0.pt`): `lr_scheduler.last_epoch` = 120 (expected 120 = end step), rng present = True; `state_resume_ok = True`.
- runtime quota semantics: `quota_semantics_ok = True`, quota error lines = 0.
- trainer log vs diagnostics: steps cross-checked = 50, `critic/score/max` mismatches = 0, consistent = True.

## 16. Runtime quota statistics step71-120

Per-optimizer-step runtime quota diagnostics for steps 71-120.

| aggregate | value |
|---|---|
| step records | 50 |
| generation calls | 536 |
| group records | 1600 |
| rollout records | 43520 |
| steps covered | [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120] |
| steps missing | [] |
| `all_steps_recorded` | True |
| quota fill rate (mean `quota_actual` / 2) | 97.0000% (0.9700) |
| steps at full quota | 47 / 50 |
| total inserted efficiency groups | 97 |
| total displaced mixed groups | 161 |
| total `eff_shortfall` | 1 |
| steps needing injection | 49 |
| generation batches (total) | 536 |
| generation batches (mean/step) | 10.7200 |
| efficiency groups seen | 97 |
| mixed groups seen | 1503 |
| zero-search rollouts | 0 (0.0000%) |
| malformed-metadata rollouts | 0 (0.0000%) |
| advantage NaN / Inf | 0 / 0 |
| quota error lines | 0 |
| steps with wrong batch size | [] |
| calls with `quota_actual` > 2 | 0 |
| steps without exactly one stopping call | [] |
| `quota_semantics_ok` | True |

| step | generation batches | quota actual | mixed kept cum | efficiency groups seen | cache size | eff_shortfall | inserted | optimizer batch groups | advantage mean | advantage std |
|---|---|---|---|---|---|---|---|---|---|---|
| 71 | 15 | 2 | 30 | 14 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 72 | 9 | 1 | 31 | 3 | 2 | 1 | 1 | 32 | -0.0000 | 0.9999 |
| 73 | 9 | 2 | 30 | 7 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 74 | 13 | 2 | 30 | 4 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 75 | 10 | 2 | 30 | 4 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |
| 76 | 11 | 2 | 30 | 11 | 2 | 0 | 2 | 32 | 0.0000 | 0.9555 |
| 77 | 10 | 2 | 30 | 6 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 78 | 10 | 2 | 30 | 6 | 2 | 0 | 2 | 32 | -0.0000 | 0.9661 |
| 79 | 10 | 2 | 30 | 12 | 2 | 0 | 2 | 32 | 0.0000 | 0.9504 |
| 80 | 10 | 2 | 30 | 4 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 81 | 11 | 2 | 30 | 10 | 2 | 0 | 2 | 32 | 0.0000 | 0.9504 |
| 82 | 13 | 2 | 30 | 8 | 2 | 0 | 2 | 32 | 0.0000 | 0.9660 |
| 83 | 13 | 2 | 30 | 8 | 2 | 0 | 2 | 32 | 0.0000 | 0.9660 |
| 84 | 10 | 2 | 30 | 6 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 85 | 9 | 2 | 30 | 10 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 86 | 10 | 2 | 30 | 6 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 87 | 13 | 2 | 30 | 14 | 2 | 0 | 2 | 32 | -0.0000 | 0.9661 |
| 88 | 8 | 2 | 30 | 6 | 2 | 0 | 2 | 32 | 0.0000 | 0.9660 |
| 89 | 10 | 2 | 30 | 11 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 90 | 12 | 2 | 30 | 6 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |
| 91 | 12 | 2 | 30 | 10 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 92 | 14 | 2 | 30 | 11 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 93 | 12 | 2 | 30 | 13 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 94 | 11 | 2 | 30 | 8 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 95 | 15 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 96 | 14 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 97 | 8 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | -0.0000 | 0.9661 |
| 98 | 12 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 99 | 10 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 100 | 11 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 101 | 8 | 2 | 30 | 8 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 102 | 12 | 2 | 30 | 4 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 103 | 10 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | -0.0000 | 0.9473 |
| 104 | 14 | 2 | 30 | 13 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 105 | 8 | 2 | 30 | 2 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |
| 106 | 10 | 2 | 30 | 6 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 107 | 13 | 2 | 30 | 10 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 108 | 8 | 2 | 30 | 4 | 2 | 0 | 2 | 32 | -0.0000 | 0.9504 |
| 109 | 10 | 2 | 30 | 5 | 2 | 0 | 2 | 32 | -0.0000 | 0.9660 |
| 110 | 8 | 1 | 31 | 1 | 1 | 0 | 1 | 32 | -0.0000 | 0.9661 |
| 111 | 10 | 2 | 30 | 7 | 2 | 0 | 2 | 32 | -0.0000 | 0.9453 |
| 112 | 10 | 2 | 30 | 10 | 2 | 0 | 2 | 32 | 0.0000 | 0.9661 |
| 113 | 9 | 2 | 30 | 5 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |
| 114 | 13 | 2 | 30 | 10 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |
| 115 | 9 | 2 | 30 | 3 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |
| 116 | 12 | 2 | 30 | 13 | 2 | 0 | 2 | 32 | -0.0000 | 0.9473 |
| 117 | 10 | 2 | 30 | 7 | 2 | 0 | 2 | 32 | -0.0000 | 0.9473 |
| 118 | 9 | 2 | 30 | 9 | 2 | 0 | 2 | 32 | -0.0000 | 0.9453 |
| 119 | 8 | 1 | 31 | 1 | 1 | 0 | 1 | 32 | 0.0000 | 0.9453 |
| 120 | 10 | 2 | 30 | 11 | 2 | 0 | 2 | 32 | -0.0000 | 0.9555 |

## 17. step120 seven-benchmark evaluation

- artifact: `/data01/wyy/Graph-Agent-Planning/E1-E/eval_results/step120_20260915_053854/eval_summary.json`; label = e1e_step120; total samples = 51201.
- macro EM = 0.4049; micro EM = 0.4395.

| benchmark | nq | triviaqa | popqa | hotpotqa | 2wikimultihopqa | musique | bamboogle |
|---|---|---|---|---|---|---|---|
| EM | 0.3860 | 0.5660 | 0.4100 | 0.4010 | 0.4480 | 0.1670 | 0.4560 |

- behaviour: rounds 1.5203, queries 2.0194, queries/round 1.3283, parallel factor 1.3283, parallel sample rate 43.6769%, zero-search 0.0391%, no-answer 0.1426%, response tokens 322.35.
- runtime mechanism: quota fill rate = 97.0000%; steps at full quota = 47.0000; steps used = 50.0000; inserted efficiency groups = 97.0000; displaced mixed groups = 161.0000; total eff_shortfall = 1.0000; mean generation batches/step = 10.7200; train zero-search share = 0.0000%; train malformed share = 0.0000%; quota_semantics_ok = 1.0000; quota error lines = 0.0000.

## 18. Per-benchmark accuracy table

Per-benchmark EM (and macro/micro) for E1-E step70/step120 and the reference rows; identical 7-benchmark greedy protocol.

| run | nq | triviaqa | popqa | hotpotqa | 2wikimultihopqa | musique | bamboogle | macro EM | micro EM |
|---|---|---|---|---|---|---|---|---|---|
| E1-E step120 | 0.3860 | 0.5660 | 0.4100 | 0.4010 | 0.4480 | 0.1670 | 0.4560 | 0.4049 | 0.4395 |
| E1-E step70 | 0.3780 | 0.5650 | 0.3990 | 0.3980 | 0.4420 | 0.1700 | 0.4240 | 0.3966 | 0.4339 |
| GAP120 | 0.3890 | 0.5770 | 0.4130 | 0.4110 | 0.4450 | 0.1780 | 0.4640 | 0.4110 | 0.4442 |
| E1-B120 | 0.3810 | 0.5620 | 0.4030 | 0.3900 | 0.4400 | 0.1630 | 0.4240 | 0.3947 | 0.4325 |
| E1-D120 | 0.3880 | 0.5680 | 0.4080 | 0.4030 | 0.4410 | 0.1770 | 0.4560 | 0.4059 | 0.4386 |
| E1-0 step70 | 0.3730 | 0.5640 | 0.3980 | 0.3940 | 0.4420 | 0.1690 | 0.4640 | 0.4006 | 0.4325 |
| E1-A step70 | 0.3780 | 0.5630 | 0.3970 | 0.3970 | 0.4380 | 0.1680 | 0.4480 | 0.3984 | 0.4317 |

## 19. Efficiency metrics table

Efficiency / behaviour metrics for the same runs as section 18 (`-` = not present in the artifact).

| run | rounds | queries | queries/round | parallel factor | parallel sample rate | zero-search | no-answer | tokens |
|---|---|---|---|---|---|---|---|---|
| E1-E step120 | 1.5203 | 2.0194 | 1.3283 | 1.3283 | 43.6769% | 0.0391% | 0.1426% | 322.35 |
| E1-E step70 | 1.6159 | 2.1152 | 1.3089 | 1.3089 | 43.6261% | 0.0391% | 0.6191% | 323.78 |
| GAP120 | 1.8537 | 2.3433 | 1.2641 | 1.2641 | 42.2921% | 0.0332% | 0.6250% | 343.95 |
| E1-B120 | 1.2950 | 1.8748 | 1.4478 | 1.4478 | 50.5732% | 0.0391% | 0.1172% | 312.95 |
| E1-D120 | 1.7234 | 2.2315 | 1.2948 | 1.2948 | 43.6691% | 0.0215% | 0.4551% | 337.08 |
| E1-0 step70 | 1.6296 | 2.1280 | 1.3059 | 1.3059 | 43.6476% | 0.0312% | 0.7441% | 324.11 |
| E1-A step70 | 1.5977 | 2.1036 | 1.3166 | 1.3166 | 44.1241% | 0.0293% | 0.5683% | 323.34 |

## 20. GAP / E1-B / E1-D / E1-E four-way comparison

Four-way step120 (source: `summary.json`):

| role | run | micro EM | macro EM | rounds | queries | tokens | zero-search |
|---|---|---|---|---|---|---|---|
| `gap_step120` | `gap_step120_reference` | 0.444170 | 0.411000 | 1.8537 | 2.3433 | 343.95 | 0.0332% |
| `e1b_step120` | `e1b_step120_reference` | 0.432535 | 0.394714 | 1.2950 | 1.8748 | 312.95 | 0.0391% |
| `e1d_step120` | `e1d_step120_reference` | 0.438561 | 0.405857 | 1.7234 | 2.2315 | 337.08 | 0.0215% |
| `e1e_step120` | `step120_20260915_053854` | 0.439475 | 0.404857 | 1.5203 | 2.0194 | 322.35 | 0.0391% |

Figures: `fig1_four_way_pareto.png`, `fig2_quota_runtime.png`, `fig3_reward_separation.png`, `fig4_efficiency_ranking.png`, `fig5_search_behavior.png`, `fig6_prediction_vs_observation.png`.

#### `e1e120_vs_gap120` (baseline `gap120` -> test `e1e120`)

- macro EM: 0.4110 -> 0.4049 (Δ -0.61 pp)
- micro EM: 0.4442 -> 0.4395 (Δ -0.47 pp)
- search rounds: 1.8537 -> 1.5203 (Δ -0.3335, -17.99%)
- queries: 2.3433 -> 2.0194 (Δ -0.3239, -13.82%)
- parallel factor: 1.2641 -> 1.3283 (Δ +0.0642)
- zero-search: 0.0332% -> 0.0391% (Δ +0.01 pp)
- paired (n = 49947): rounds 1.8568 -> 1.5223 (Δ -0.3345, -18.01%), sign-test p = 0; EM 0.4450 -> 0.4401 (Δ -0.48 pp), up/down 2974/3215, sign-test p = 0.00219.

#### `e1e120_vs_e1b120` (baseline `e1b120` -> test `e1e120`)

- macro EM: 0.3947 -> 0.4049 (Δ +1.01 pp)
- micro EM: 0.4325 -> 0.4395 (Δ +0.69 pp)
- search rounds: 1.2950 -> 1.5203 (Δ +0.2253, +17.40%)
- queries: 1.8748 -> 2.0194 (Δ +0.1446, +7.71%)
- parallel factor: 1.4478 -> 1.3283 (Δ -0.1194)
- zero-search: 0.0391% -> 0.0391% (Δ +0.00 pp)
- paired (n = 49947): rounds 1.2972 -> 1.5223 (Δ +0.2251, +17.35%), sign-test p = 0; EM 0.4332 -> 0.4401 (Δ +0.70 pp), up/down 3411/3062, sign-test p = 1.44e-05.

#### `e1e120_vs_e1d120` (baseline `e1d120` -> test `e1e120`)

- macro EM: 0.4059 -> 0.4049 (Δ -0.10 pp)
- micro EM: 0.4386 -> 0.4395 (Δ +0.09 pp)
- search rounds: 1.7234 -> 1.5203 (Δ -0.2031, -11.78%)
- queries: 2.2315 -> 2.0194 (Δ -0.2120, -9.50%)
- parallel factor: 1.2948 -> 1.3283 (Δ +0.0335)
- zero-search: 0.0215% -> 0.0391% (Δ +0.02 pp)
- paired (n = 49947): rounds 1.7270 -> 1.5223 (Δ -0.2047, -11.85%), sign-test p = 0; EM 0.4392 -> 0.4401 (Δ +0.09 pp), up/down 3127/3082, sign-test p = 0.568.

#### `e1e120_vs_e1e70` (baseline `e1e70` -> test `e1e120`)

- macro EM: 0.3966 -> 0.4049 (Δ +0.83 pp)
- micro EM: 0.4339 -> 0.4395 (Δ +0.56 pp)
- search rounds: 1.6159 -> 1.5203 (Δ -0.0957, -5.92%)
- queries: 2.1152 -> 2.0194 (Δ -0.0957, -4.53%)
- parallel factor: 1.3089 -> 1.3283 (Δ +0.0194)
- zero-search: 0.0391% -> 0.0391% (Δ +0.00 pp)
- paired (n = 49947): rounds 1.6186 -> 1.5223 (Δ -0.0962, -5.94%), sign-test p = 3.85e-100; EM 0.4348 -> 0.4401 (Δ +0.53 pp), up/down 3299/3033, sign-test p = 0.000829.

#### `e1e70_vs_e1_0_step70` (baseline `e1_0_step70` -> test `e1e70`)

- macro EM: 0.4006 -> 0.3966 (Δ -0.40 pp)
- micro EM: 0.4325 -> 0.4339 (Δ +0.14 pp)
- search rounds: 1.6296 -> 1.6159 (Δ -0.0136, -0.84%)
- queries: 2.1280 -> 2.1152 (Δ -0.0129, -0.60%)
- parallel factor: 1.3059 -> 1.3089 (Δ +0.0030)
- zero-search: 0.0312% -> 0.0391% (Δ +0.01 pp)
- paired (n = 49947): rounds 1.5912 -> 1.6186 (Δ +0.0273, +1.72%), sign-test p = 3.82e-21; EM 0.4358 -> 0.4348 (Δ -0.10 pp), up/down 2531/2582, sign-test p = 0.476.

#### `e1e70_vs_e1a70` (baseline `e1a_step70` -> test `e1e70`)

- macro EM: 0.3984 -> 0.3966 (Δ -0.19 pp)
- micro EM: 0.4317 -> 0.4339 (Δ +0.21 pp)
- search rounds: 1.5977 -> 1.6159 (Δ +0.0182, +1.14%)
- queries: 2.1036 -> 2.1152 (Δ +0.0115, +0.55%)
- parallel factor: 1.3166 -> 1.3089 (Δ -0.0077)
- zero-search: 0.0293% -> 0.0391% (Δ +0.01 pp)
- paired (n = 49947): rounds 1.5999 -> 1.6186 (Δ +0.0187, +1.17%), sign-test p = 6.3e-13; EM 0.4328 -> 0.4348 (Δ +0.20 pp), up/down 2567/2466, sign-test p = 0.155.

#### `e1e70_vs_e1d70` (baseline `e1d70` -> test `e1e70`)

- macro EM: 0.3976 -> 0.3966 (Δ -0.10 pp)
- micro EM: 0.4333 -> 0.4339 (Δ +0.06 pp)
- search rounds: 1.6237 -> 1.6159 (Δ -0.0078, -0.48%)
- queries: 2.1274 -> 2.1152 (Δ -0.0123, -0.58%)
- parallel factor: 1.3102 -> 1.3089 (Δ -0.0013)
- zero-search: 0.0391% -> 0.0391% (Δ +0.00 pp)

## 21. Paired statistical tests

### `e1e120_vs_e1b120`

- source: `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e120_vs_e1b120.json`
- paired prompts: 49947
- search rounds: 1.2972 -> 1.5223 (Δ +0.2251, +17.35%); decreased 1867, increased 10684, unchanged 37396; sign-test p = 0
- search queries: 1.8793 -> 2.0246 (Δ +0.1454, +7.73%)
- EM: 0.4332 -> 0.4401 (Δ +0.70 pp); up 3411, down 3062; sign-test p = 1.44e-05

| benchmark | n | rounds base | rounds E1-E | Δrounds | Δrounds % | EM base | EM E1-E | ΔEM | sign-test p (EM) |
|---|---|---|---|---|---|---|---|---|---|
| nq | 3610 | 1.0629 | 1.2061 | +0.1432 | +13.47% | 0.3814 | 0.3861 | +0.47 pp | 0.457 |
| triviaqa | 11312 | 1.0861 | 1.2394 | +0.1533 | +14.11% | 0.5617 | 0.5659 | +0.42 pp | 0.19 |
| popqa | 12509 | 1.1717 | 1.3945 | +0.2228 | +19.01% | 0.4028 | 0.4095 | +0.67 pp | 0.0227 |
| hotpotqa | 7404 | 1.3741 | 1.5809 | +0.2068 | +15.05% | 0.3901 | 0.4011 | +1.11 pp | 0.00527 |
| 2wikimultihopqa | 12576 | 1.5325 | 1.8360 | +0.3034 | +19.80% | 0.4397 | 0.4482 | +0.84 pp | 0.025 |
| musique | 2411 | 1.8067 | 2.1530 | +0.3463 | +19.17% | 0.1634 | 0.1667 | +0.33 pp | 0.634 |
| bamboogle | 125 | 1.6800 | 1.8640 | +0.1840 | +10.95% | 0.4240 | 0.4560 | +3.20 pp | 0.388 |

### `e1e120_vs_e1d120`

- source: `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e120_vs_e1d120.json`
- paired prompts: 49947
- search rounds: 1.7270 -> 1.5223 (Δ -0.2047, -11.85%); decreased 10285, increased 3913, unchanged 35749; sign-test p = 0
- search queries: 2.2382 -> 2.0246 (Δ -0.2136, -9.54%)
- EM: 0.4392 -> 0.4401 (Δ +0.09 pp); up 3127, down 3082; sign-test p = 0.568

| benchmark | n | rounds base | rounds E1-E | Δrounds | Δrounds % | EM base | EM E1-E | ΔEM | sign-test p (EM) |
|---|---|---|---|---|---|---|---|---|---|
| nq | 3610 | 1.3399 | 1.2061 | -0.1338 | -9.99% | 0.3875 | 0.3861 | -0.14 pp | 0.853 |
| triviaqa | 11312 | 1.3645 | 1.2394 | -0.1251 | -9.17% | 0.5684 | 0.5659 | -0.25 pp | 0.425 |
| popqa | 12509 | 1.5853 | 1.3945 | -0.1907 | -12.03% | 0.4077 | 0.4095 | +0.18 pp | 0.529 |
| hotpotqa | 7404 | 1.7488 | 1.5809 | -0.1679 | -9.60% | 0.4033 | 0.4011 | -0.22 pp | 0.59 |
| 2wikimultihopqa | 12576 | 2.1546 | 1.8360 | -0.3186 | -14.79% | 0.4406 | 0.4482 | +0.76 pp | 0.0426 |
| musique | 2411 | 2.4285 | 2.1530 | -0.2754 | -11.34% | 0.1767 | 0.1667 | -1.00 pp | 0.1 |
| bamboogle | 125 | 2.0640 | 1.8640 | -0.2000 | -9.69% | 0.4560 | 0.4560 | +0.00 pp | 1 |

### `e1e120_vs_e1e70`

- source: `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e120_vs_e1e70.json`
- paired prompts: 49947
- search rounds: 1.6186 -> 1.5223 (Δ -0.0962, -5.94%); decreased 7746, increased 5318, unchanged 36883; sign-test p = 3.85e-100
- search queries: 2.1211 -> 2.0246 (Δ -0.0965, -4.55%)
- EM: 0.4348 -> 0.4401 (Δ +0.53 pp); up 3299, down 3033; sign-test p = 0.000829

| benchmark | n | rounds base | rounds E1-E | Δrounds | Δrounds % | EM base | EM E1-E | ΔEM | sign-test p (EM) |
|---|---|---|---|---|---|---|---|---|---|
| nq | 3610 | 1.2820 | 1.2061 | -0.0759 | -5.92% | 0.3781 | 0.3861 | +0.80 pp | 0.183 |
| triviaqa | 11312 | 1.2919 | 1.2394 | -0.0525 | -4.06% | 0.5649 | 0.5659 | +0.11 pp | 0.737 |
| popqa | 12509 | 1.5133 | 1.3945 | -0.1188 | -7.85% | 0.3990 | 0.4095 | +1.06 pp | 0.000396 |
| hotpotqa | 7404 | 1.6403 | 1.5809 | -0.0594 | -3.62% | 0.3983 | 0.4011 | +0.28 pp | 0.477 |
| 2wikimultihopqa | 12576 | 1.9796 | 1.8360 | -0.1436 | -7.25% | 0.4421 | 0.4482 | +0.60 pp | 0.107 |
| musique | 2411 | 2.2377 | 2.1530 | -0.0846 | -3.78% | 0.1701 | 0.1667 | -0.33 pp | 0.622 |
| bamboogle | 125 | 1.8800 | 1.8640 | -0.0160 | -0.85% | 0.4240 | 0.4560 | +3.20 pp | 0.454 |

### `e1e120_vs_gap120`

- source: `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e120_vs_gap120.json`
- paired prompts: 49947
- search rounds: 1.8568 -> 1.5223 (Δ -0.3345, -18.01%); decreased 13269, increased 2980, unchanged 33698; sign-test p = 0
- search queries: 2.3473 -> 2.0246 (Δ -0.3226, -13.75%)
- EM: 0.4450 -> 0.4401 (Δ -0.48 pp); up 2974, down 3215; sign-test p = 0.00219

| benchmark | n | rounds base | rounds E1-E | Δrounds | Δrounds % | EM base | EM E1-E | ΔEM | sign-test p (EM) |
|---|---|---|---|---|---|---|---|---|---|
| nq | 3610 | 1.4726 | 1.2061 | -0.2665 | -18.10% | 0.3895 | 0.3861 | -0.33 pp | 0.602 |
| triviaqa | 11312 | 1.4535 | 1.2394 | -0.2141 | -14.73% | 0.5773 | 0.5659 | -1.13 pp | 0.000373 |
| popqa | 12509 | 1.7203 | 1.3945 | -0.3258 | -18.94% | 0.4126 | 0.4095 | -0.30 pp | 0.292 |
| hotpotqa | 7404 | 1.8355 | 1.5809 | -0.2546 | -13.87% | 0.4109 | 0.4011 | -0.97 pp | 0.0091 |
| 2wikimultihopqa | 12576 | 2.3295 | 1.8360 | -0.4936 | -21.19% | 0.4452 | 0.4482 | +0.29 pp | 0.429 |
| musique | 2411 | 2.6271 | 2.1530 | -0.4741 | -18.05% | 0.1779 | 0.1667 | -1.12 pp | 0.0691 |
| bamboogle | 125 | 1.9600 | 1.8640 | -0.0960 | -4.90% | 0.4640 | 0.4560 | -0.80 pp | 1 |

### `e1e70_vs_e1a70`

- source: `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e70_vs_e1a70.json`
- paired prompts: 49947
- search rounds: 1.5999 -> 1.6186 (Δ +0.0187, +1.17%); decreased 4907, increased 5646, unchanged 39394; sign-test p = 6.3e-13
- search queries: 2.1085 -> 2.1211 (Δ +0.0126, +0.60%)
- EM: 0.4328 -> 0.4348 (Δ +0.20 pp); up 2567, down 2466; sign-test p = 0.155

| benchmark | n | rounds base | rounds E1-E | Δrounds | Δrounds % | EM base | EM E1-E | ΔEM | sign-test p (EM) |
|---|---|---|---|---|---|---|---|---|---|
| nq | 3610 | 1.2640 | 1.2820 | +0.0180 | +1.42% | 0.3784 | 0.3781 | -0.03 pp | 1 |
| triviaqa | 11312 | 1.2838 | 1.2919 | +0.0081 | +0.63% | 0.5632 | 0.5649 | +0.17 pp | 0.554 |
| popqa | 12509 | 1.4993 | 1.5133 | +0.0140 | +0.93% | 0.3973 | 0.3990 | +0.17 pp | 0.521 |
| hotpotqa | 7404 | 1.6155 | 1.6403 | +0.0249 | +1.54% | 0.3974 | 0.3983 | +0.09 pp | 0.812 |
| 2wikimultihopqa | 12576 | 1.9514 | 1.9796 | +0.0281 | +1.44% | 0.4379 | 0.4421 | +0.42 pp | 0.209 |
| musique | 2411 | 2.2124 | 2.2377 | +0.0253 | +1.14% | 0.1680 | 0.1701 | +0.21 pp | 0.757 |
| bamboogle | 125 | 1.8560 | 1.8800 | +0.0240 | +1.29% | 0.4480 | 0.4240 | -2.40 pp | 0.508 |

### `e1e70_vs_gap60`

- source: `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e70_vs_gap60.json`
- paired prompts: 49947
- search rounds: 1.5912 -> 1.6186 (Δ +0.0273, +1.72%); decreased 4891, increased 5870, unchanged 39186; sign-test p = 3.82e-21
- search queries: 2.0970 -> 2.1211 (Δ +0.0242, +1.15%)
- EM: 0.4358 -> 0.4348 (Δ -0.10 pp); up 2531, down 2582; sign-test p = 0.476

| benchmark | n | rounds base | rounds E1-E | Δrounds | Δrounds % | EM base | EM E1-E | ΔEM | sign-test p (EM) |
|---|---|---|---|---|---|---|---|---|---|
| nq | 3610 | 1.2626 | 1.2820 | +0.0194 | +1.54% | 0.3740 | 0.3781 | +0.42 pp | 0.458 |
| triviaqa | 11312 | 1.2810 | 1.2919 | +0.0109 | +0.85% | 0.5634 | 0.5649 | +0.15 pp | 0.596 |
| popqa | 12509 | 1.4924 | 1.5133 | +0.0209 | +1.40% | 0.4006 | 0.3990 | -0.16 pp | 0.534 |
| hotpotqa | 7404 | 1.6041 | 1.6403 | +0.0362 | +2.26% | 0.3972 | 0.3983 | +0.11 pp | 0.781 |
| 2wikimultihopqa | 12576 | 1.9370 | 1.9796 | +0.0425 | +2.20% | 0.4466 | 0.4421 | -0.45 pp | 0.197 |
| musique | 2411 | 2.1966 | 2.2377 | +0.0411 | +1.87% | 0.1750 | 0.1701 | -0.50 pp | 0.407 |
| bamboogle | 125 | 1.8080 | 1.8800 | +0.0720 | +3.98% | 0.4480 | 0.4240 | -2.40 pp | 0.375 |

## 22. Pareto analysis

Figures: `fig1_four_way_pareto.png`, `fig2_quota_runtime.png`, `fig3_reward_separation.png`, `fig4_efficiency_ranking.png`, `fig5_search_behavior.png`, `fig6_prediction_vs_observation.png`.

E1-E is evaluated on the same pre-registered frontier as its predecessors: x = mean
search rounds over the seven benchmarks (lower is better), y = micro EM over the 51,201 samples
(higher is better). `fig1_four_way_pareto.png` plots the four step120 runs together with the two
E1-E endpoints and the pre-registered target box (micro EM >= GAP120 - 0.5 pp and rounds <=
GAP120 - 10 %).

| run | micro EM | macro EM | rounds | queries | tokens | Pareto status |
|---|---|---|---|---|---|---|
| GAP120 | 0.444170 | 0.411000 | 1.8537 | 2.3433 | 343.95 | accuracy anchor, most expensive |
| E1-B120 | 0.432535 | 0.394714 | 1.2950 | 1.8748 | 312.95 | cheapest, but 1.16 pp below GAP120 |
| E1-D120 | 0.438561 | 0.405857 | 1.7234 | 2.2315 | 337.08 | accuracy recovered, round gain mostly given back |
| **E1-E120** | **0.439475** | 0.404857 | **1.5203** | **2.0194** | **322.35** | **dominates E1-D120; inside the target box vs GAP120** |

* **E1-E120 vs E1-D120 - Pareto dominance on the preregistered axes.** Micro EM is higher
  (+0.09 pp) *and* rounds are lower (-11.78 %, paired -11.85 %, p ~ 0), so E1-E120 is not
  dominated by E1-D120 and strictly dominates it on (micro EM, rounds). The two extra axes are
  reported honestly rather than hidden: macro EM is 0.10 pp *lower* than E1-D120
  (0.404857 vs 0.405857 - a benchmark-weighting difference driven by `musique`/`bamboogle`) and
  the paired accuracy difference is statistically indistinguishable (p = 0.568). Queries fall
  -9.50 % and tokens -4.4 %, so the round saving is not obtained by searching more per round in
  aggregate; the parallel factor rises slightly (1.3283 vs 1.2948, +2.6 %) as the policy
  concentrates the same retrieval work into fewer, wider rounds.
* **E1-E120 vs GAP120 - inside the preregistered target box, with a real but small accuracy
  cost.** Both thresholds hold at the aggregate level (Delta micro EM -0.47 pp >= -0.5 pp;
  Delta rounds -17.99 % <= -10 %), but the paired sign test (p = 0.0022, 2,974 prompts up vs
  3,215 down) says the 0.48 pp deficit is not sampling noise. E1-E therefore trades a small,
  detectable amount of GAP's accuracy for an 18 % cheaper search process - it does not match
  GAP's accuracy.
* **E1-E120 vs E1-B120 - a different, and better-placed, point on the frontier.** E1-B120 is
  cheaper in rounds (1.2950) but pays 1.16 pp of micro EM relative to GAP120 and 0.69 pp
  relative to E1-E120 (paired p = 1.4e-05). E1-E120 sits between E1-B120 and E1-D120, closer to
  E1-D120's accuracy with a much larger share of E1-B's efficiency: it keeps **59.7 %** of the
  GAP120 -> E1-B120 round reduction (0.3334 of 0.5587 rounds) while losing only 40.5 % as much
  accuracy (-0.47 pp vs -1.16 pp).
* **Trajectory.** E1-E improves monotonically along its own branch: from step70 to step120 micro
  EM rises +0.56 pp (paired +0.53 pp, p = 8.3e-04) while rounds fall a further -5.92 %
  (paired -5.94 %, p ~ 1e-100), so the frontier position is not a single-step fluctuation and the
  1-3 % step70 effect that section 15 predicted would be too small to judge is indeed only a
  small part of the final result.
* **No degenerate shortcut underlies the frontier move.** The step120 evaluation shows
  zero-search 0.0391 % (E1-D120 0.0215 %, GAP120 0.0332 % - same order), no-answer-tag 0.1426 %
  (E1-D120 0.2145 %, GAP120 0.744 %), empty-output 0, and queries 13.8 % *below* GAP120. The
  training-side checks agree (section 23): within the efficiency-active groups the efficient
  correct rollouts use both fewer rounds (1.53 vs 2.65) and fewer queries (2.09 vs 3.07), so the
  bonus rewards genuine search economy rather than query packing.

## 23. Reward-hacking / safety analysis

Figures: `fig1_four_way_pareto.png`, `fig2_quota_runtime.png`, `fig3_reward_separation.png`, `fig4_efficiency_ranking.png`, `fig5_search_behavior.png`, `fig6_prediction_vs_observation.png`.

### D1 (steps 61-70) runtime semantics

| check | count | expected |
|---|---|---|
| mixed rollouts with reward != EM | 0 | 0 |
| mixed groups with reward_mean != em_mean | 0 | 0 |
| mixed groups with reward outside [0, 1] | 0 | 0 |
| efficiency rollouts with wrong reward or EM | 0 | 0 |
| efficiency groups with reward outside [1, 1.05] | 0 | 0 |
| efficiency groups with k != 8 | 0 | 0 |
| efficiency groups without cost variation | 0 | 0 |
| wrong rollouts receiving non-zero reward | 0 | 0 |
| rollouts with inconsistent published metric | 0 | 0 |
| advantage NaN | 0 | 0 |
| advantage Inf | 0 | 0 |
| quota error lines | 0 | 0 |
| zero-search rollouts | 0 | informational |
| malformed-metadata rollouts | 0 | informational |
| zero-search share | 0.0000% | informational |
| malformed-metadata share | 0.0000% | informational |

### D2 (steps 71-120) runtime semantics

| check | count | expected |
|---|---|---|
| mixed rollouts with reward != EM | 0 | 0 |
| mixed groups with reward_mean != em_mean | 0 | 0 |
| mixed groups with reward outside [0, 1] | 0 | 0 |
| efficiency rollouts with wrong reward or EM | 0 | 0 |
| efficiency groups with reward outside [1, 1.05] | 0 | 0 |
| efficiency groups with k != 8 | 0 | 0 |
| efficiency groups without cost variation | 0 | 0 |
| wrong rollouts receiving non-zero reward | 0 | 0 |
| rollouts with inconsistent published metric | 0 | 0 |
| advantage NaN | 0 | 0 |
| advantage Inf | 0 | 0 |
| quota error lines | 0 | 0 |
| zero-search rollouts | 0 | informational |
| malformed-metadata rollouts | 0 | informational |
| zero-search share | 0.0000% | informational |
| malformed-metadata share | 0.0000% | informational |

### D1 (steps 61-70) packing / zero-search diagnostic

Definition: packing_suspect = within an active (>=2 correct, cost-varying) group the efficient correct rollouts use fewer rounds but NOT fewer total queries

- E1-E: active groups 28; efficient correct rollouts rounds 1.4842 vs inefficient 2.5862; queries 1.9368 vs 3.2672; packing-suspect groups 2 (ratio 7.1429%).
- baseline: active groups 95; efficient correct rollouts rounds 1.4881 vs inefficient 2.4915; queries 2.0398 vs 2.9322; packing-suspect groups 4 (ratio 4.2105%).
- zero-search: E1-E rate 0.0000% vs baseline 0.0163%; E1-E zero-search correct rate 0.0000%.
- malformed metadata rate: 0.0000%.

### D2 (steps 71-120) packing / zero-search diagnostic

Definition: packing_suspect = within an active (>=2 correct, cost-varying) group the efficient correct rollouts use fewer rounds but NOT fewer total queries

- E1-E: active groups 158; efficient correct rollouts rounds 1.5257 vs inefficient 2.6496; queries 2.0873 vs 3.0673; packing-suspect groups 11 (ratio 6.9620%).
- baseline: active groups 95; efficient correct rollouts rounds 1.4881 vs inefficient 2.4915; queries 2.0398 vs 2.9322; packing-suspect groups 4 (ratio 4.2105%).
- zero-search: E1-E rate 0.0000% vs baseline 0.0163%; E1-E zero-search correct rate 0.0000%.
- malformed metadata rate: 0.0000%.

**Reward-hacking verdict.**

**No reward hacking is evidenced.** All recorded reward-separation and safety counters are zero in both phases (mixed reward == EM, efficiency reward in [1, 1.05], wrong-rollout reward == 0, published metric consistent, advantage NaN/Inf == 0, no quota error lines), and the zero-search and malformed shares are reported above. The packing-suspect ratio above is a behavioural diagnostic (definition given in the artifact), not a reward-hacking violation.

### 23.1 Online selection composition (RQ7)

* inserted efficiency groups: 117 of 465 seen (348 seen but never inserted); kept mixed groups: 1803
* replacement rate: 0.0609
* every inserted group is all-correct (k = 8) and cost-varying: True, cost variation present for 117/117 inserted groups (mean gap 1.137, min 1.0)
* mean insertion position: generation batch 10.95 (the stopping batch, by construction)
* sources recorded for inserted groups: 2 of 117 (the rest were injected from the cache, where the manager does not carry `source`) — the online source-bias test therefore cannot be reproduced (see sections 25.1 RQ7 and 28)

What cannot be checked online:

* data source of cached/injected efficiency groups: rows are written from a synthetic group dict with source=None and rollouts_pid has no source column, so the offline inserted-vs-displaced source test (E1-E0 p = 0.920) cannot be reproduced online
* difficulty (k/8) of the displaced mixed groups: they are dropped before any record is written

Available bias evidence:

* selection is pure stream order on a shuffled dataset and the planner only ever takes cache[:m]; it never ranks groups by source, difficulty or reward
* every inserted group is k = 8 (all-correct) and cost-varying by construction, so the mechanism cannot prefer easy or hard *mixed* groups
* E1-E0's offline test on E1-D's stream found no difficulty bias (p = 0.943) and no source bias (p = 0.920); the online run's own stream differs, so this remains a transferred (not re-measured) result

## 24. E1-E0 offline prediction vs online observation

- phases: ['D1', 'D2']; optimizer steps observed: 60; rollouts: 52480; group records: 1920.
- steps covered: 61..120 (60 steps).

| quantity | E1-E0 prediction (q=2) | E1-E online observation |
|---|---|---|
| full_quota_rate | 1.000000 | 0.950000 |
| actual_replacement_rate | 0.062500 | 0.060937 |
| mean_efficiency_groups_inserted_per_step | 2.000000 | 1.950000 |
| mean_generation_batches_per_step | 10.460000 | 10.950000 |
| mean_efficiency_groups_seen_per_step | 8.060000 | 7.750000 |
| mean_k_over_8_of_optimizer_batch | 0.533672 | 0.536458 |
| mean_search_rounds | 1.942344 | 1.665116 |
| mean_search_queries | 2.588516 | 2.276464 |
| mean_response_tokens | 1642.230859 | 1455.900260 |
| micro_em_of_optimizer_batches | - | 0.510388 |
| mean_inserted_cost_gap | - | 1.000000 |
| zero_search_rate | - | 0.000000 |
| malformed_rate | - | 0.000000 |
| total_eff_shortfall | 0.000000 | 1.000000 |

Agreement flags:

- `full_quota_rate`: False
- `replacement_rate_matches`: True
- `efficiency_groups_seen_positive_every_step`: True
- `rounds_and_queries_scale`: True

Honest caveats recorded in the artifact:

- E1-E0's prediction table covers E1-D's candidate stream for optimizer steps 71-120; when the observed steps fall outside that range (D1: 61-70) the predicted Generation-batch statistics are NOT comparable and the row is informational only.
- E1-E online generates its own candidate stream: its step70 policy differs from E1-D's step70 policy, so the per-step group membership/deviation from E1-E0's CSV is expected and is NOT a runtime mismatch (the runtime equivalence itself was proven offline: E1-E/analysis/preflight_selftest.json, q=0 and q=2 exact on 50/50 steps of the E1-D stream).
- D1 (steps 61-70) has no E1-E0 prediction at all: E1-E0's stream only covers steps 71-120.
- E1-E0's candidate pool has an artificially high all-correct share (37,926 of 83,680 groups) because it was harvested from a shaped run; the online mixed-group share therefore differs, which changes the number of generation batches per step.

The two key caveats (paraphrased):

1. **Provenance of the prediction** — E1-E0 predicted on **E1-D's** candidate stream (the D2 steps of the E1-D policy), not on E1-E's own online policy's stream; per-step group membership therefore cannot be compared one-to-one, and only the distributional claims are comparable.
2. **D1 has no E1-E0 prediction at all** — E1-E0's stream only covers steps 71-120, so D1 (steps 61-70) is outside the offline prediction's scope.

## 25. Whether E1-E meets the preregistered success criteria

| flag | preregistered threshold | value | observed |
|---|---|---|---|
| `delta_EM_at_least_minus_0.5pp_vs_GAP120` | Δ micro EM vs GAP120 ≥ −0.5 pp (preregistered) | True | Δmicro EM vs GAP120 = -0.47 pp |
| `delta_rounds_at_most_minus_10pct_vs_GAP120` | Δ search rounds vs GAP120 ≤ −10 % (preregistered) | True | Δrounds vs GAP120 = -17.99% |
| `pareto_dominates_E1D120` | Pareto-domination of E1-D120: Δ micro EM vs E1-D120 ≥ −0.5 pp AND Δ search rounds vs E1-D120 ≤ 0 | True | Δmicro EM vs E1-D120 = +0.09 pp, Δrounds vs E1-D120 = -0.2031 |
| `keeps_E1D_EM_and_beats_E1D_rounds` | keeps E1-D120 EM (Δmicro EM ≥ −0.5 pp) AND strictly fewer rounds (Δrounds < 0) vs E1-D120 | True | Δmicro EM vs E1-D120 = +0.09 pp, Δrounds vs E1-D120 = -0.2031 |
| `more_efficient_than_E1B120` | fewer search rounds than E1-B120 (Δrounds < 0) | False | Δrounds vs E1-B120 = +0.2253 (+17.40%) |
| `higher_EM_than_E1B120` | higher micro EM than E1-B120 (Δ micro EM > 0) | True | Δmicro EM vs E1-B120 = +0.69 pp |

Mechanical `case` (from `summary.json`): **STRONG_PASS**; `experiment_status` = COMPLETE.

### 25.1 Answers to the preregistered research questions

**RQ1 - does q=2 controlled solved-group efficiency supervision preserve more GAP correctness
capacity?** Yes, relative to the filter-only baseline and to the shaping baseline, but not
relative to GAP itself. Micro EM at step120: E1-E 0.4395 vs E1-B120 0.4325 (**+0.69 pp**, paired
p = 1.4e-05) and vs E1-D120 0.4386 (+0.09 pp, paired p = 0.568), versus GAP120 0.4442
(**-0.47 pp**, paired p = 0.0022). E1-E therefore recovers most of the 1.16 pp that E1-B's
correctness filter cost and matches E1-D, while remaining 0.47 pp below GAP - inside the
preregistered band, but a real deficit.

**RQ2 - does E1-E Pareto-dominate E1-D (higher EM *and* lower rounds)?** Yes on the preregistered
axes: micro EM +0.09 pp and rounds -11.78 % (-11.85 % paired, p ~ 0, 10,285 prompts improved vs
3,913 worsened), with the accuracy difference statistically indistinguishable (p = 0.568). Macro
EM is 0.10 pp lower, which is reported rather than suppressed.

**RQ3 - versus GAP120, is Delta micro EM >= -0.5 pp *and* Delta rounds <= -10 %?** Yes:
-0.47 pp and -17.99 % (-18.01 % paired). This is the preregistered main engineering success
criterion, and both halves hold.

**RQ4 - versus E1-B, is part of the main efficiency gain retained at a clearly smaller accuracy
cost?** Yes. E1-B's gain is -30.14 % rounds at -1.16 pp micro EM relative to GAP120; E1-E keeps
-17.99 % rounds at -0.47 pp, i.e. it retains 59.7 % of the round reduction for 40.5 % of the
accuracy cost, and it *gains* 0.69 pp over E1-B in absolute accuracy.

**RQ5 - what is the actual q=2 fill rate in the formal runtime?** 95 % overall: 10/10 steps
(100 %) in D1 (steps 61-70) and 47/50 (94 %) in D2 (steps 71-120). 117 efficiency groups were
inserted in total, an observed replacement rate of 6.09 % of the 3,840 optimizer-batch slots
(versus the 6.25 % predicted offline). All 60 steps satisfy the frozen formula
`quota_actual = min(q, |cache|)`, except one recorded cap-induced shortfall (step 72:
`|cache| = 2`, `mixed_kept_cum = 31`, quota 1, `eff_shortfall = 1`); steps 110 and 119 inserted a
single group because their stopping batch had seen exactly one efficiency group, which is the
pre-registered `m = min(q, |cache|)` behaviour, not a deviation.

**RQ6 - is the efficiency improvement free of query anomalies, query/round explosion,
zero-search or malformed-output increases?** Yes. Step120 versus GAP120: queries -13.82 %,
queries/round 1.3283 vs 1.2640 (+5.1 %, i.e. modestly more parallel and not an explosion),
zero-search 0.0391 % vs 0.0332 %, no-answer-tag 0.1426 % vs 0.7441 % (lower), empty output 0.
Training-side, over 52,480 materialised rollouts: zero-search 0.0 %, malformed metadata 0.0 %,
and the efficiency-active groups' efficient rollouts use fewer rounds *and* fewer queries, so the
packing heuristic fires on only 11/158 active groups (7.0 %, comparable to the 4/95 = 4.2 %
baseline rate).

**RQ7 - do E1-E0's offline quota-availability and bias conclusions hold under the online
policy?** The availability conclusions hold distributionally, with two honest qualifications.
Full-quota rate 95 % vs 100 % predicted; replacement rate 6.09 % vs 6.25 %; efficiency groups seen
per step 7.75 vs 8.06 predicted; mean generation batches per step 10.95 vs 10.46; mean k/8 of the
optimizer batch 0.5365 vs 0.5337; the online policy is cheaper per trajectory than E1-D's
(1.665 vs 1.942 rounds, 1,456 vs 1,642 tokens), which is why its absolute numbers differ. The
bias conclusion cannot be re-measured online: the reward manager does not carry `source` for a
cached (injected) group, so only 2 of the 117 inserted groups have a recorded source and the
inserted-vs-displaced source test cannot be reproduced; the displaced mixed groups' k/8 is not
recorded either. What can be verified online is that every inserted group is all-correct and
cost-varying (117/117, mean cost gap 1.14) and that the selection rule is pure stream order on a
shuffled dataset (verified offline against E1-E0 on 50/50 steps). E1-E0's offline no-bias result
(p = 0.943 difficulty, p = 0.920 source) therefore transfers, but is not independently confirmed
online - see section 28.

## 26. What E1-E establishes

1. **A working, non-invasive way to gate a solved-group efficiency signal.** E1-E shows
that a quota-gated, reward-separated efficiency objective can be injected into this GRPO/GAP
pipeline *without* touching the original trainer: a custom scorer publishes the filter metric
through the existing `reward_extra_info -> non_tensor_batch` path, and the reward manager
materialises the frozen `mixed[:B-m] + cache[:m]` batch in place at the stopping generation
batch. The mechanism ran for 60 optimizer steps and 52,480 materialised rollouts with 0
unrecorded calls, 0 oversized returned batches, 0 generation-batch gaps, 0 metric
inconsistencies, 0 NaN/Inf advantages and 0 quota error lines (sections 11-16, 23), and the
original GAP/verl sources are byte-identical before and after (section 31).
2. **The central empirical claim of the E1-C/E1-D/E1-E0 line of work: separating the filter
channel from the objective channel works.** E1-E keeps the mixed-group objective exactly as in
E1-B (`R_i = A_i`, no shaping) and adds efficiency supervision only through whole all-correct
groups that the filter *injects*. The result is the best frontier position of the four runs on
the preregistered axes: micro EM statistically indistinguishable from E1-D120 (p = 0.568) with
11.85 % fewer search rounds (p ~ 0), and -17.99 % rounds versus GAP120 at a -0.47 pp micro-EM
cost. E1-D's shaping had to pay for its accuracy recovery with rounds; E1-E does not.
3. **A quantitative decomposition of where the round reduction and the accuracy loss come from.**
Combining E1-C's offline decomposition with the online runs: the correctness filter accounts for
about 76.8 % of the round reduction and about 52 % of the accuracy loss; E1-E shows that
re-attaching a *small* (6.09 % of slots), *reward-separated*, *cost-based* signal on top of the
filter recovers +0.69 pp of that accuracy while giving back only about 40 % of the round gain.
4. **Validated mechanism semantics.** The 12 section-11 invariants plus the empty-keep regression
(13 named checks over 4 parts of the selftest) mean the frozen design was not just asserted but
tested: q=0/q=2 planner replay matches E1-E0 exactly on 50/50 steps, the measured replacement
rate is 6.09 % vs 6.25 % predicted, `sum_reward - sum_em = 0.05 x sum_eff` holds exactly, the
trainer's own logged `critic/score/max = 1.050` marks every step with an insertion, and the
empty-keep path (18 of 50 steps in the proxy stream) is provably neutral.
5. **A reproducible, self-gating pipeline.** One command reproduces the whole experiment
(section 30) with an automatic D1 -> D2 gate, disk guard, integrity checks, paired statistics,
figures and provenance manifests - and it caught a real design-violating runtime defect during
development (section 6.1), which is itself evidence for the value of the gating.

## 27. What it does NOT establish

1. **Nothing about q.** Only q = 2 was run. E1-E0 predicted q = 1/2/3/4 behaviour
offline (fill 100/100/100/94 %, replacement 3.13/6.25/9.38/12.31 %); E1-E measures q = 2 only, so
the dose-response of the mechanism - the natural next scientific question - is not established.
2. **Nothing about lambda.** lambda = 0.05 was frozen and never swept, so the observed frontier
point is conditional on it; whether the same round saving could be had with no accuracy cost at a
different lambda is untested.
3. **No claim that E1-E matches GAP's accuracy.** The preregistered band is satisfied, but the
-0.47 pp micro-EM deficit is statistically detectable (p = 0.0022). E1-E is a better *trade-off*
point, not an accuracy-equivalent replacement.
4. **No claim of Pareto dominance on every axis.** E1-E120 dominates E1-D120 on micro EM and
rounds and is indistinguishable on paired accuracy, but its macro EM is 0.10 pp lower and it is
*less* round-efficient than E1-B120 (1.5203 vs 1.2950 rounds).
5. **No causal attribution to any single sub-component of the signal.** E1-E changes the
selection channel (quota injection of whole all-correct groups), the reward (1 + 0.05 E only for
those groups) and the batch composition (displacing the last m mixed groups) as one frozen
package; this run does not separate their individual contributions.
6. **No online confirmation of the no-selection-bias claim.** As detailed in RQ7, the diagnostics
do not record the source of cached/injected groups, so E1-E0's offline no-bias result transfers
rather than being re-measured.
7. **No generality beyond this setting.** One model, one agent scaffold, one tool (wiki search),
one cost definition (`logical_search_batches`), one 7-benchmark evaluation protocol, one seed per
phase, and a 60-step training window from a fixed GAP step-60 checkpoint.

## 28. Limitations

1. **Single seed, single run per phase.** D1 and D2 are one trajectory each; there is no
seed replication, so the step120 point estimate carries the usual RL run-to-run variance. The
paired per-prompt tests (49,947 prompts) are precise about *this* policy pair but say nothing
about seed variance. The step70 -> step120 consistency (+0.56 pp EM, -5.9 % rounds) and the
monotone four-run ordering are reassuring but not a substitute.
2. **The E1-E0 prediction was made on a different candidate stream.** E1-E0 harvested its
predictions from E1-D's D2 candidate pool (a shaped policy, with an artificially high all-correct
share); the online run generates its own stream, so only distributional comparisons are valid.
The observed differences (95 % vs 100 % fill, 10.95 vs 10.46 generation batches per step, 1.665
vs 1.942 rounds per selected group) are consistent with a *different but plausible* stream rather
than with a mechanism error - the mechanism itself was verified exactly on the E1-D stream.
3. **The mixed cap can - and once did - reduce the efficiency dose.** The planner caps mixed
groups at `B - min(q, |cache before the batch|)` so that the running count cannot reach B before
the mixed-only stopping batch. If the cache grows *during* the stopping batch, the batch can end
up with 31 mixed + 1 efficiency group; this happened once (step 72) and is recorded as
`eff_shortfall = 1`. This is inherited from the frozen E1-E0 planner, but it means the realised
efficiency signal (6.09 %) is slightly below the nominal 6.25 %.
4. **Instrumentation gaps found during the run.** The reward manager's `source` field is `None`
for cached/injected groups and `rollouts_pid*.jsonl` has no source column, so the online
source-bias test is impossible; the displaced mixed groups' difficulty is also unrecorded. A
future run should record `data_source` and `k` on the injected rows.
5. **Two diagnostic-only defects and one design-violating defect were found in the E1-E manager
itself during development** (leaked-variable id lists, an empty advantage list at stopping calls,
and - most seriously - an empty-keep early return that let raw generation batches reach the
trainer's filter, changing the stopping point and the batch composition on 5 of D1's first 9
steps). The run was stopped, fixed, the section-11 gate extended with a PART D regression plus
runtime leak invariants, and the whole chain restarted from the original GAP step-60 full state.
All results in this report come from the restarted, verified run; the invalid run is archived
under `E1-E/runs/INVALID_D1_20260914_093225/` and is never used. The fact that this class of
defect can survive an offline replay built on the *intended* planner is itself a limitation of
offline pre-validation and a reason to keep runtime invariants in the pipeline.
6. **The evaluation protocol is a single greedy 7-benchmark pass per checkpoint.** It is large
(51,201 samples) and uses identical sample IDs across runs, but it is one decoding configuration
(temperature 0 / greedy, 8 turns max) and it does not measure pass@k, calibration, or behaviour
outside these benchmarks. `popqa` is known to have duplicate prompts, which is why the paired
analysis skips non-unique prompt IDs (49,947 of 51,201 samples are paired).
7. **The step70 evaluation (and hence the gate) is a 10-step snapshot.** Section 15 deliberately
does not stop on a 1-3 % step70 effect, and indeed the step70 gain is small (+0.14 pp EM,
-0.84 % rounds vs E1-0 step70); the meaningful signal only appears by step120.

## 29. Recommended next experiment

1. **q = 3 ablation (highest value, preregistered by E1-E0).** E1-E0 predicted a 9.38 %
replacement rate at q = 3 with 100 % fill on its proxy stream and no bias. Now that q = 2 has
produced a clean Pareto improvement over E1-D120, q = 3 is the natural dose-response probe: does a
50 % larger efficiency dose buy a further round reduction, and at what accuracy cost? Cost: one
D2-length run plus one evaluation (~16 h) if the same protocol is reused, with the same
`e1e_quota_gated` manager and only `E1E_QUOTA=3` changed.
2. **q = 1 ablation (cheap control).** A 3.13 % dose would test whether the effect is monotone at
the low end and whether the small residual accuracy gap to GAP120 is dose-related. It also gives
a nearly-free replication of the pipeline on a second configuration.
3. **lambda sensitivity of the *gated* signal (0.02 / 0.05 / 0.10), only if q = 3 shows headroom.**
With the signal confined to 6 % of slots and a mean bonus of 1.026, the current lambda may simply
be too small to move accuracy; a larger lambda on the same group set is the cheapest way to test
that without touching the filter channel.
4. **Fix the instrumentation before the next run** (cheap, and it removes limitation 4): record
`data_source`, `k` and the cache position on injected rows, and record the displaced mixed
groups' `k`/`cost`. This would make the online bias test and an exact online-vs-offline
displacement audit possible.
5. **A seed-replication run at the winning configuration** to bound run-to-run variance before
any claim is promoted beyond "a better trade-off point at this seed". One extra seed at q = 2
(~16 h) is enough to estimate the round and EM variance and to re-run the paired tests
conservatively.
6. **Do not run a formal multi-seed q-sweep yet.** The evidence supports one high-value ablation
(q = 3) plus the instrumentation fix; a full lambda x q grid would consume several GPU-days before
the variance is even known, so it should wait until step 5.

## 30. Full reproduction commands

From `E1-E/README.md` "Reproduce" (lines 76–94):

```bash
# 0. offline gate (CPU only, no GPU): planner + runtime replay vs E1-E0, 12 mandated invariants
bash E1-E/scripts/e1e_preflight.sh
PYTHONPATH=$PWD/verl:$PWD/E1-E/code/pythonpath_e1e python3 E1-E/code/e1e_runtime_selftest.py

# 1. everything, unattended (preflight -> selftest -> D1 -> integrity -> eval70 -> gate ->
#    D2 -> integrity -> eval120 -> analysis)
bash E1-E/scripts/run_e1e_all.sh all

# or stage by stage
bash E1-E/scripts/run_e1e_all.sh d1
bash E1-E/scripts/run_e1e_all.sh eval70
bash E1-E/scripts/run_e1e_all.sh gate
bash E1-E/scripts/run_e1e_all.sh d2
bash E1-E/scripts/run_e1e_all.sh eval120
bash E1-E/scripts/run_e1e_all.sh analysis
```

`run_e1e_all.sh` also accepts the stages `preflight` and `diskguard` (the pre-D2 disk guard is part
of `all`); the training phases are invoked through `run_e1e_train_60_70.sh` (D1) and
`run_e1e_train_70_120.sh` (D2), which both `exec run_e1e_train.sh` with the respective
`E1E_PHASE`.

Source: `E1-E/README.md` (lines 78–94); `E1-E/scripts/run_e1e_all.sh` (lines 36–78);
`E1-E/scripts/run_e1e_train_60_70.sh`; `E1-E/scripts/run_e1e_train_70_120.sh`.

**Evaluation entry points:**

```bash
bash E1-E/scripts/run_e1e_eval_70.sh      # full 7-benchmark eval of E1-E step70
bash E1-E/scripts/run_e1e_eval_120.sh     # full 7-benchmark eval of E1-E step120
# or directly: bash E1-E/scripts/run_e1e_eval_step.sh <step>
```

Both wrap `run_e1e_eval_step.sh`, which sources the shared `e1e_eval_common.sh`; that helper
reuses the ORIGINAL `Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh` unmodified.

Source: `E1-E/scripts/run_e1e_eval_70.sh`; `E1-E/scripts/run_e1e_eval_120.sh`;
`E1-E/scripts/run_e1e_eval_step.sh`; `E1-E/scripts/e1e_eval_common.sh`.

**Analysis entry point:**

```bash
bash E1-E/scripts/run_e1e_analysis.sh
```

Source: `E1-E/scripts/run_e1e_analysis.sh`.

---

## 31. Complete artifact inventory

Inventory of `E1-E/` as it exists on disk at the end of the experiment (`__pycache__` omitted).

* `E1-E_task.md` (file, 32093 B)
* `FINAL_REPORT.md` (file, 99904 B)
* `FINAL_REPORT_draft.md` (file, 80044 B)
* `FINAL_REPORT_static_sections.md` (file, 39054 B)
* `README.md` (file, 5691 B)
* `analysis/` — 19 file(s): `behavior_metrics.csv`, `benchmark_scores.csv`, `d1_gate.json`, `d1_gate.md`, `online_bias_check.json`, `online_bias_check.md`, `original_files_unchanged.json`, `original_files_unchanged.md`, `overall_metrics.csv`, `prediction_vs_observation.json`, `prediction_vs_observation.md`, `preflight_selftest.json`, `preflight_selftest.md`, `results.json`, ... (19 files total). subdirectories: `D1_vs_E1_0/`, `D1_vs_E1_B/`, `D2_vs_E1_0/`, `D2_vs_E1_B/`, `all_evals/`, `comparison_20260915_072012/`, `selftest_diag/`
* `checkpoints/` — 1 file(s): `experiment_path.txt`. subdirectories: `model_dir/`
* `code/` — 12 file(s): `e1e_checkpoint_patch.py`, `e1e_cost.py`, `e1e_diag_patch.py`, `e1e_hang_diagnostics.py`, `e1e_quota_plan.py`, `e1e_reward_manager.py`, `e1e_runtime_selftest.py`, `e1e_scorer.py`, `e1e_sglang_gpu_patch.py`, `e1e_stall_watchdog.py`, `e1e_state.py`, `e1e_trainer_patch.py`. subdirectories: `pythonpath_e1e/`
* `configs/` — 2 file(s): `resolved_D1.json`, `resolved_D2.json`.
* `diagnostics/` — 2 file(s): `safety_alerts.log`, `safety_monitor.jsonl`. subdirectories: `checkpoint_stats_D1/`, `checkpoint_stats_D2/`
* `eval_results/` — 8 file(s): `behavior_metrics.csv`, `benchmark_scores.csv`, `latest_step120.txt`, `latest_step120_eval.txt`, `latest_step70.txt`, `latest_step70_eval.txt`, `overall_metrics.csv`, `results.json`. subdirectories: `step120_20260915_053854/`, `step70_20260914_151446/`
* `figures/` — 6 file(s): `fig1_four_way_pareto.png`, `fig2_quota_runtime.png`, `fig3_reward_separation.png`, `fig4_efficiency_ranking.png`, `fig5_search_behavior.png`, `fig6_prediction_vs_observation.png`.
* `legacy_e1d_unused/` — 3 file(s): `README.md`, `e1e_filter_selftest.py`, `e1e_selftest.py`.
* `logs/` — 14 file(s): `analysis_20260915_072012.log`, `d1_gate_status.txt`, `disk_cleanup.log`, `disk_guard_status.txt`, `issues_and_fixes.md`, `pipeline_status.txt`, `preflight.log`, `preflight_status.txt`, `run_all.log`, `run_all_stdout.log`, `train_D1_20260914_093225.log`, `train_D1_20260914_115858.log`, `train_D2_20260914_165545.log`, `wiki_service.log`.
* `manifests/` — 10 file(s): `global_step_100.md`, `global_step_110.md`, `global_step_120.md`, `global_step_70.md`, `global_step_80.md`, `global_step_90.md`, `manifest.json`, `original_files_diff.json`, `original_files_manifest_post.json`, `original_files_manifest_pre.json`.
* `runs/` — 2 file(s): `latest_D1.txt`, `latest_D2.txt`. subdirectories: `D1_20260914_115858/`, `D2_20260914_165545/`, `INVALID_D1_20260914_093225/`
* `scripts/` — 35 file(s): `e1e_analyze_eval.py`, `e1e_analyze_training.py`, `e1e_assemble_report.py`, `e1e_build_summary.py`, `e1e_check_training_integrity.py`, `e1e_checkpoint_manifests.py`, `e1e_checkpoint_stats.py`, `e1e_collect_evaluation_artifacts.py`, `e1e_compare_all.py`, `e1e_compare_eval.py`, `e1e_d1_gate.py`, `e1e_disk_guard.sh`, `e1e_eval_common.sh`, `e1e_extract_config.py`, ... (35 files total).

Model checkpoints live outside `E1-E/` (never copied, recorded in `E1-E/checkpoints/experiment_path.txt` and `E1-E/manifests/manifest.json`):

* `global_step_100/` — 35.7 GiB
* `global_step_110/` — 35.7 GiB
* `global_step_120/` — 42.0 GiB
* `global_step_70/` — 42.0 GiB
* `global_step_80/` — 35.7 GiB
* `global_step_90/` — 35.7 GiB

---

Generated: 2026-09-15T07:31:18 by `E1-E/scripts/e1e_assemble_report.py`.

Input files read (sha256, short):

- `E1-B/evaluations/e10_step70_reference/eval_summary.json` — `6dda66901caf`
- `E1-B/evaluations/e1a_step70_reference/eval_summary.json` — `55e72c2b1114`
- `E1-B/evaluations/gap_step120_reference/eval_summary.json` — `799bb09cf866`
- `E1-B/evaluations/gap_step180_reference/eval_summary.json` — `d00d8c430999`
- `E1-B/evaluations/gap_step60_reference/eval_summary.json` — `6c404610f462`
- `E1-B/evaluations/step120_20260911_235134/eval_summary.json` — `b901b7feca77`
- `E1-D/eval_results/step120_20260913_165834/eval_summary.json` — `1cabbb8d53af`
- `E1-D/eval_results/step70_20260913_024024/eval_summary.json` — `3da70880c27a`
- `E1-E/FINAL_REPORT_static_sections.md` — `4f5ea8947e9e`
- `E1-E/analysis/D1_vs_E1_0/safety_analysis.json` — `1479e85c3913`
- `E1-E/analysis/D1_vs_E1_0/training_summary.json` — `eba154244fe7`
- `E1-E/analysis/D2_vs_E1_0/safety_analysis.json` — `bca5f8abe19c`
- `E1-E/analysis/D2_vs_E1_0/training_summary.json` — `f3d55e72f6d0`
- `E1-E/analysis/comparison_20260915_072012/comparison.json` — `59bbcbaabcb8`
- `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e120_vs_e1b120.json` — `7c1d78236401`
- `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e120_vs_e1d120.json` — `4ef474ef59b3`
- `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e120_vs_e1e70.json` — `821fb9359e59`
- `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e120_vs_gap120.json` — `4151492ac006`
- `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e70_vs_e1a70.json` — `99a18e6044a2`
- `E1-E/analysis/comparison_20260915_072012/paired_comparison_e1e70_vs_gap60.json` — `dbc166eb4283`
- `E1-E/analysis/d1_gate.json` — `db9ecc91d8c8`
- `E1-E/analysis/original_files_unchanged.json` — `bee4464f1527`
- `E1-E/analysis/prediction_vs_observation.json` — `519f6d10bd2d`
- `E1-E/analysis/step120_integrity.json` — `6c3847ddb986`
- `E1-E/analysis/step70_integrity.json` — `d5c00ece6d63`
- `E1-E/analysis/summary.json` — `8306d01401e0`
- `E1-E/eval_results/step120_20260915_053854/eval_summary.json` — `9b015571c7a7`
- `E1-E/eval_results/step70_20260914_151446/eval_summary.json` — `421018c727d0`
- `E1-E/manifests/original_files_manifest_pre.json` — `e6541b323a21`


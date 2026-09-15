# E1-E — static report sections

This file contains only the pre-registered/static sections of the E1-E final report:
§2–§10 and §30–§31. Result, integrity, evaluation, comparison, Pareto, safety,
prediction-vs-observation, verdict and next-step sections are written separately by the caller.

---

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

Inventory of `E1-E/` as it exists on disk (directories and their actual contents; empty directories
listed as empty):

| path | contents |
|---|---|
| `E1-E/E1-E_task.md` | pre-registered task specification |
| `E1-E/README.md` | experiment README (design, layout, reproduction) |
| `E1-E/FINAL_REPORT_static_sections.md` | this file |
| `code/` | `e1e_checkpoint_patch.py`, `e1e_cost.py`, `e1e_diag_patch.py`, `e1e_hang_diagnostics.py`, `e1e_quota_plan.py`, `e1e_reward_manager.py`, `e1e_runtime_selftest.py`, `e1e_scorer.py`, `e1e_sglang_gpu_patch.py`, `e1e_stall_watchdog.py`, `e1e_state.py`, `e1e_trainer_patch.py`, and `pythonpath_e1e/sitecustomize.py` (plus `__pycache__/`) |
| `scripts/` | `e1e_analyze_eval.py`, `e1e_analyze_training.py`, `e1e_build_summary.py`, `e1e_checkpoint_manifests.py`, `e1e_checkpoint_stats.py`, `e1e_check_training_integrity.py`, `e1e_collect_evaluation_artifacts.py`, `e1e_compare_all.py`, `e1e_compare_eval.py`, `e1e_d1_gate.py`, `e1e_disk_guard.sh`, `e1e_eval_common.sh`, `e1e_extract_config.py`, `e1e_integrity.py`, `e1e_paired_eval_compare.py`, `e1e_pipeline.sh`, `e1e_prediction_vs_observation.py`, `e1e_preflight.sh`, `e1e_render_report_sections.py`, `e1e_report_numbers.py`, `e1e_safety_monitor.py`, `e1e_verify_original_unchanged.sh`, `run_e1e_all.sh`, `run_e1e_analysis.sh`, `run_e1e_eval_120.sh`, `run_e1e_eval_70.sh`, `run_e1e_eval_step.sh`, `run_e1e_train_60_70.sh`, `run_e1e_train_70_120.sh`, `run_e1e_train.sh` |
| `configs/` | empty (no resolved-config files present at the time of writing) |
| `logs/` | `d1_gate_status.txt`, `disk_cleanup.log`, `disk_guard_status.txt`, `issues_and_fixes.md`, `preflight.log`, `preflight_status.txt`, `run_all.log`, `run_all_stdout.log`, `train_D1_20260914_093225.log`, `wiki_service.log` |
| `runs/` | `latest_D1.txt`; `D1_20260914_093225/` with `stdout.log`, `stderr.log`, `config/` (`git_diff.txt`, `launch_command.txt`, `reward_config.json`), `diagnostics/` (empty), `logs/` (`safety_monitor.jsonl`, `safety_monitor_agg.json`, `safety_monitor_state.json`, `safety_monitor_stdout.log`, `watchdog.log`, `watchdog_stdout.log`, `stackdumps/stackdump_pid*.txt`), `preflight/` (`code_manifest.txt`, `preflight_train.txt`) |
| `diagnostics/` | `safety_monitor.jsonl` |
| `analysis/` | `d1_gate.json`, `d1_gate.md`, `prediction_vs_observation.json`, `preflight_selftest.json`, `preflight_selftest.md` |
| `eval_results/` | empty (no step evaluation directories yet) |
| `manifests/` | `original_files_manifest_pre.json` |
| `checkpoints/` | `experiment_path.txt` and the symlink `model_dir -> experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu` (model weights stay in `experiments/`) |
| `figures/` | empty |
| `legacy_e1d_unused/` | `README.md`, `e1e_filter_selftest.py`, `e1e_selftest.py` (parked E1-D self-test copies, not used) |

Source: direct `find` / `ls` of `E1-E/` on disk.

---

### Sources read

`E1-E/README.md`, `E1-E/E1-E_task.md`, `E1-E/logs/issues_and_fixes.md`,
`E1-E/logs/preflight.log`, `E1-E/logs/preflight_status.txt`, `E1-E/logs/disk_cleanup.log`,
`E1-E/logs/disk_guard_status.txt`, `E1-E/logs/wiki_service.log`,
`E1-E/runs/D1_20260914_093225/preflight/preflight_train.txt`,
`E1-E/runs/D1_20260914_093225/config/reward_config.json`,
`E1-E/analysis/preflight_selftest.md`, `E1-E/analysis/preflight_selftest.json`,
`E1-E/manifests/original_files_manifest_pre.json`, `E1-E/checkpoints/experiment_path.txt`,
`E1-E/legacy_e1d_unused/README.md`, `E1-E/code/e1e_quota_plan.py`,
`E1-E/code/e1e_reward_manager.py`, `E1-E/code/e1e_scorer.py`,
`E1-E/code/e1e_cost.py`, `E1-E/code/e1e_runtime_selftest.py`,
`E1-E/code/pythonpath_e1e/sitecustomize.py`, `E1-E/scripts/run_e1e_train.sh`,
`E1-E/scripts/run_e1e_all.sh`, `E1-E/scripts/run_e1e_eval_common.sh`,
`E1-E/scripts/run_e1e_eval_step.sh`, `E1-E/scripts/run_e1e_eval_70.sh`,
`E1-E/scripts/run_e1e_eval_120.sh`, `E1-E/scripts/run_e1e_analysis.sh`,
`E1-E/scripts/run_e1e_train_60_70.sh`, `E1-E/scripts/run_e1e_train_70_120.sh`,
`E1-E/scripts/e1e_pipeline.sh`, `E1-B/FINAL_REPORT.md`, `E1-D/FINAL_REPORT.md`,
`E1-C/FINAL_REPORT.md`, `E1-E0/FINAL_REPORT.md`, `E1-E0/results/recommended_quota.json`,
`E1-E0/results/quota_availability.csv`, `E1-E0/results/quota_supervision_density.csv`,
`verl/verl/trainer/config/ppo_trainer.yaml`, `E1-B/scripts/run_e1b_train.sh`.

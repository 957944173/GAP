# E1-D issues and fixes

Every problem encountered while executing E1-D, with timestamp, root cause, affected stage,
fix, whether it changed any scientific variable, and how it was verified.

---

## I1 — the D1 integrity check reports FAIL when re-run after D2 (fixed, reporting only)

* **When**: 2026-09-13 18:41 (analysis stage) — first noticed while assembling the report.
* **Symptom**: `E1-D/analysis/step70_integrity.json` was rewritten by the analysis stage with
  `integrity_ok = false` (the original D1 check at 02:40 had passed), because
  `latest_checkpointed_iteration.txt` had legitimately advanced to 120 by then and the check
  required `tracker == 70` exactly.
* **Root cause**: the tracker file is **experiment-wide**, not per-phase, so an exact-equality
  test is only valid immediately after the phase.  The scientifically meaningful requirement
  is "the phase's checkpoint exists, is complete, and the tracker has reached at least the
  phase's end step".
* **Fix**: `e1d_integrity.py` now uses `int(tracker) >= END_STEP`, records
  `tracker_reached_end_step` and a note explaining the semantics; both phases re-checked
  (`step70`: PASS, `step120`: PASS).
* **Scientific variable changed**: no — the check is a post-hoc verification, not part of
  training.  The D1 checkpoint itself (global_step_70, 38,292,967,072 B actor + `data.pt`)
  is unchanged and was verified complete at 02:40 before D2 started.
* **Verified**: `E1-D/analysis/step70_integrity.{json,md}` and `step120_integrity.{json,md}`
  both report `integrity_ok = true`, `state_resume_ok = true`,
  `tracker_reached_end_step = true`.

## I2 — `KeyError: 'rounds_e1b_mean'` in the summary / report-number helpers (fixed)

* **When**: 2026-09-13 18:49 (analysis stage) and again 19:0x while collecting report numbers.
* **Symptom**: `e1d_build_summary.py` aborted with `KeyError: 'rounds_e1b_mean'` (wrapped in a
  `WARN build_summary`, so the pipeline still finished), and `e1d_report_numbers.py` printed
  the references but crashed at the paired sections.
* **Root cause**: the E1-D copies of the paired-comparison script renamed the output fields
  (`rounds_e1b_mean` → `rounds_e1d_mean`, `em_e1b_mean` → `em_e1d_mean`), but the two
  consumers still read the old names.
* **Fix**: consumers updated to the new field names; `analysis/summary.json` rebuilt
  successfully and the report-number helper runs to completion.
* **Scientific variable changed**: no (analysis/reporting code only; every number comes from
  the same paired-comparison JSONs).

## I3 — selftest TEST 7 originally measured an uninformative batch (fixed)

* **When**: 2026-09-13 00:0x (selftest, before any GPU use).
* **Symptom**: the first version computed the GRPO advantage on `list(groups)[:32]` — the first
  32 groups of the candidate stream — which happen to be all-wrong/all-correct, so every
  advantage was exactly 0 (`max_abs = 0.0`).  The test "passed" but proved nothing.
* **Fix**: TEST 7 now reproduces the trainer's batch fill for step 71 (the first 32 groups kept
  by the EM filter, in generation order) and reports `n_groups`, `nan`, `inf`, `max_abs`,
  `mean_abs`.  Result: 256 trajectories, 0 NaN, 0 Inf, `max|adv| = 2.4749`,
  `mean|adv| = 0.8165`.
* **Scientific variable changed**: no (test-only; it strengthened the evidence that the shaped
  reward reaches GRPO).

## I4 — note on the efficiency-supervision density (not a bug)

E1-C measured, on the **E1-B run's own candidate pool** (steps 71–120), that the EM filter
would retain 1,412 groups of which only 18 are efficiency-active (1.3 %).  E1-D measures, on
**its own** candidate pool over the same step range, 1,679 retained groups with 86
efficiency-active (5.1 %).  The two numbers are not contradictory: they are different pools
(different policies, hence different mixed/active frequencies).  Both say the same thing — the
EM filter keeps only a small fraction of the efficiency-active groups that the shaped filter
would have kept (E1-C: 18 vs 291; E1-D's own pool: 86 vs 520, i.e. **−83.5 %**).

## I5 — disk headroom

`/data01` held E1-D's own 6 checkpoints (~258 GB) plus the earlier experiments.  Free space was
monitored at every checkpoint; the pipeline refuses to start D2 with < 320 GB free.  No
existing artifact was deleted at any point.

## I6 — evaluation directory naming in the analysis merge (fixed)

`run_e1d_analysis.sh` merges every evaluation into `analysis/all_evals/` via symlinks; the
first version prefixed the link names with the parent directory, which hid E1-D's own runs
from the `step<N>_<ts>` prefix detection used by `e1d_compare_all.py`.  Fixed so E1-D's own
evaluations keep their `step<N>_<timestamp>` name; references keep theirs.  Not overwritten:
`ln -sfn` only ever points at new names.

## I7 — no unknown process was killed

Preflight found no stale Ray/SGLang/`main_ppo` process at any point; nothing was killed.  The
wiki retriever (pid 2230350) was already running and was **reused**, never restarted or stopped.

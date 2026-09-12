# E1-C errors, bugs and fixes

## B1 — `KeyError: 'selected_count'` in the Analysis C aggregation (fixed)

* The per-step replay rows written into the batch-fill summary use the keys
  `GAP_selected` / `E1_selected`, but the aggregate block still summed
  `selected_count` (left over from an earlier draft).  First E1-C run aborted with
  `KeyError: 'selected_count'` after Analysis A.
* Fix: sum `GAP_selected` / `E1_selected`.  Re-run completed; this is a code bug in E1-C
  only, no earlier experiment was affected.

## B2 — first Analysis C draft did not distinguish containment from batch membership

* The candidate-level result is `S_GAP ⊂ S_E1` (strict subset), yet the fixed-size
  optimizer batch is only a **partial overlap**.  The first draft reported the aggregate
  "calls needed" without flagging that the counterfactual GAP arm is **right-censored** on
  pools that were generated under the E1 filter (the E1 run stops generating as soon as 32
  groups are retained, so the GAP arm may never reach 32 within the generated batches).
* Fix: added `GAP_censored_steps`, `GAP_kept_per_batch`, `E1_kept_per_batch`,
  `fill_rate_ratio_E1_over_GAP` and the projected-calls fields, and re-anchored the H3
  evidence on the **uncensored** `E1_0` pool.  This is a reporting/analysis bug fixed in E1-C.

## B3 — duplicate optimizer steps when merging the original GAP console logs

* `train_console.run3.log` and `train_console.run4.log` both contain steps 191–199, so the
  merge concatenates them (recorded in `summary.json → actual_run.duplicate_steps_across_gap_logs`).
* Impact on E1-C: **none** — the actual-run comparison is restricted to steps 70–120 and
  the duplicates are at steps 191–199.  The field is kept in the artifact for transparency.

## B4 — `E1-0.5` result cross-check

* E1-C recomputes the baseline retained set on the E1-0 audit pool from scratch and gets
  **340**, identical to `E1-0.5/outputs/filter_replay_results.json → baseline_retained`.  No
  bug found in the earlier report for this quantity.
* E1-C does **not** modify any E1-0.5 file; the comparison is recorded in
  `results/summary.json → sanity.matches_E1_0.5`.

## B5 — environment

* `scipy` is not installed in either usable conda env, so the rank-sum and two-proportion
  tests are implemented from scratch (`e1c_common.mannwhitney_u`, `two_prop_z`, normal
  approximation with tie correction).  p-values are reported to 2–3 significant digits.
* `matplotlib` is absent in `parallel-agent`; the figures were rendered with
  `/home/nf5468m6/miniconda3/envs/huatuo/bin/python3` reading only `E1-C/results/*.csv`.

## B6 — disk headroom

* `/data01` was at ~97 % when E1-C started (the E1-B checkpoints occupy ~215 GB).  All E1-C
  artifacts are CSV/JSON/PNG (a few MB); no large intermediate file is produced and no
  existing artifact was deleted to make room.

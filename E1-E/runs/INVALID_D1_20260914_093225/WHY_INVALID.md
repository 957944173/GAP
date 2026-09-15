# Invalid D1 run (2026-09-14 09:32 - 11:51) — do not use for results

This D1 run was stopped at optimizer step 69/70 on purpose.  Its runtime diagnostics proved
that the reward manager's **empty-keep early return** let the raw (unmutated) generation batch
reach the trainer's group filter, so unbudgeted efficiency groups were accumulated into the
optimizer batch and advanced the generation stopping counter.  That violates the pre-registered
E1-E design (E1-E_task.md section 1: only mixed groups count toward B = 32; efficiency groups
are cached and injected at the stopping batch as mixed[:B-m] + cache[:m]) and section 15's hard
stop condition "quota 导致 generation stopping 被改变".

Direct evidence (see `D1_20260914_093225/diagnostics/`):
* `rollouts_pid2594120.jsonl` contains calls that returned the full raw generation batch
  (1280 rollouts = 160 groups x n=8) which have **no matching record** in
  `quota_calls_pid2594120.jsonl`: step 67 twice (11:21:22, 11:29:33) and step 68 once (11:38:08).
* step 67 therefore has 8 recorded calls (gb 1,2,4,5,6,7,8,9) with only 27 mixed groups kept and
  no stopping call, while the trainer's own log shows 10 generation batches and an accumulated
  count of 34 groups (27 mixed + 2 leaked + 5 leaked), truncated to 32: the optimizer batch was
  27 mixed + 5 unbudgeted efficiency groups, not 30 mixed + 2 cached efficiency groups.
* steps 61-66 and 68-69 reached exactly 32 groups (30 mixed + 2 injected) and are individually
  consistent, but a run that violates the frozen mechanism on any step cannot be used for the
  pre-registered comparison.

The step-70 checkpoint was never saved (the run was stopped before step 70), so
`experiments/DAPO-GAP3B-MHQA-Agent-E1E-q2-step60to120-4gpu/` does not exist and nothing has to
be deleted.  The fix is documented in `E1-E/logs/issues_and_fixes.md`; the restarted run uses a
new run dir and re-verifies the mechanism with the section-11 selftest (extended with an
empty-keep case) before training.

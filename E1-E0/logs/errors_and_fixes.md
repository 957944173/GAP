# E1-E0 errors and fixes

## B1 — `NameError: name 'members' is not defined` (fixed)

* The per-step quota row used `members` (the name used in the group-membership helper) inside
  the `inserted_mean_cost_gap` expression while the loop variable is `mem`.  The run aborted
  after writing `q0_baseline_replay.csv`.
* Fix: use `mem`.

## B2 — `ValueError: too many values to unpack (expected 2)` (fixed)

* After B1 the same expression still unpacked `mem` as `(gid, role)` tuples, but `mem` is a
  list of **dicts** (the batch-member rows).  Fixed to `gidx.loc[r["group_id"]] ... for r in mem
  if r["role"] == "efficiency"`.
* Both bugs were in the E1-E0 driver only; they wrote no wrong number into any artifact (the
  affected CSV is written after the statement, and the run was re-executed from scratch).

## N1 — note: two pools, two different efficiency densities (not a bug)

* E1-D's own run retained 1,679 groups with **86** efficiency-active ones (**5.12 %**) because
  the EM filter drops all-correct groups.
* On the *same* D2 stream, **403** efficiency groups appear inside the stopping prefixes
  (**8.06 per step**), i.e. an unrestricted admission would be **25.19 %** of the optimizer
  groups.  E1-C measured **291/1,685 = 17.3 %** efficiency-active groups on E1-B's own pool.
* These are three different quantities (retained-by-EM-filter, available-in-stream,
  retained-by-shaped-filter on a different policy) and are never mixed in one comparison.

## N2 — note: uid repeats are possible but were not observed

* Sanity check 8 requires no duplicate group ID **or uid** inside one optimizer batch.  Both
  are **0** for all 5 quotas × 50 steps: the same prompt never appears twice in one batch in
  this stream, although the stream itself re-samples uids across steps.

## N3 — no silent drops

* All 669,440 rollout records were parsed and grouped; `sum(group sizes) ==
  number of rollout records`; no malformed record was skipped (`malformed_records_dropped = 0`).

## N4 — reuse boundary

* `E1-E0/code/e1e_common.py` is a copy-and-extend of `E1-C/code/e1c_common.py`; the group
  reconstruction, the cost definition (`logical_search_batches`) and the filter rule are kept
  verbatim, and the new code only adds the mixed/efficiency classification, the mixed-only
  stopping rule, the efficiency cache and the quota batch construction.
* E1-C / E1-D files were opened read-only; nothing under those directories was modified.

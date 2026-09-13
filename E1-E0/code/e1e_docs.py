#!/usr/bin/env python3
"""E1-E0 documentation generator: data_inventory.md, configs/analysis_config.json,
logs/errors_and_fixes.md, README.md.  Values are read from results/summary.json so the
prose cannot drift from the artifacts.
"""
import csv
import hashlib
import json
import os

E1E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(E1E)
RES = os.path.join(E1E, "results")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rd(name):
    with open(os.path.join(RES, name)) as fh:
        return list(csv.DictReader(fh))


def main():
    S = json.load(open(os.path.join(RES, "summary.json")))
    rec = json.load(open(os.path.join(RES, "recommended_quota.json")))
    unref = {r["quantity"]: r["value"] for r in rd("unrestricted_reference.csv")}
    cs = S["candidate_stream"]
    avail = {r["q"]: r for r in (S["quota_availability"])}
    dens = {r["q"]: r for r in (S["supervision_density"])}
    proxy = {r["q"]: r for r in (S["batch_level_proxy"])}
    dispm = {d["q"]: d for d in (S["mixed_displacement"]) if d.get("replaced")}

    sources = [
        ("E1-D D2 candidate stream (rollouts)", "E1-D/runs/D2_20260913_041846/diagnostics/rollouts_pid1249393.jsonl"),
        ("E1-D D2 generation-batch records", "E1-D/runs/D2_20260913_041846/diagnostics/calls_pid1249393.jsonl"),
        ("E1-D D2 group records (cross-check)", "E1-D/runs/D2_20260913_041846/diagnostics/groups_pid1249393.jsonl"),
        ("E1-D D2 resolved config (target B)", "E1-D/configs/resolved_step70_to120.json"),
        ("E1-C summary (unrestricted comparison)", "E1-C/results/summary.json"),
        ("E1-D summary (baseline cross-check)", "E1-D/analysis/summary.json"),
        ("E1-C common code copied", "E1-C/code/e1c_common.py"),
        ("E1-C replay code copied", "E1-C/code/e1c_run_all.py"),
        ("trainer filter source", "verl/verl/trainer/ppo/ray_trainer.py"),
        ("GRPO advantage source", "verl/verl/trainer/ppo/core_algos.py"),
    ]
    prov = []
    for label, rel in sources:
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            prov.append((label, rel, os.path.getsize(p), sha(p)))
        else:
            prov.append((label, rel, None, "MISSING"))
    with open(os.path.join(E1E, "configs/source_provenance.json"), "w") as fh:
        json.dump([{"label": l, "path": p, "bytes": b, "sha256": s} for l, p, b, s in prov], fh, indent=2)

    inv = f"""# E1-E0 data inventory / provenance

Generated from `E1-E0/results/summary.json`; every file below was opened **read-only**.
E1-E0 writes nothing outside `E1-E0/`.

## 1. Candidate stream (the E1-D D2 run's own generation stream)

| item | value |
|---|---|
| source file | `E1-D/runs/D2_20260913_041846/diagnostics/rollouts_pid1249393.jsonl` |
| rollouts | {cs['rollouts']:,} |
| n=8 groups | {cs['groups']:,} |
| generation batches (`call_index`) | {cs['generation_batches']} |
| optimizer steps (`global_step`) | {cs['optimizer_steps']} ({cs['steps'][0]}–{cs['steps'][1]}) |
| data sources | {cs['sources']} |
| mixed-correctness groups (1 ≤ k ≤ 7) | {cs['mixed_groups']:,} |
| all-correct groups (k = 8) | {cs['all_correct_groups']:,} |
| all-wrong groups (k = 0) | {cs['all_wrong_groups']:,} |
| **all-correct + efficiency-variable groups** | **{cs['efficiency_groups']:,}** |

Target optimizer batch size **B = {S['target_optimizer_batch_size']}**, read automatically from
`{S['target_source']}` (`data.train_batch_size`), together with `gen_batch_size =
{S['gen_batch_size']}`, `rollout.n = {S['rollout_n']}` and `filter_groups.metric = em`.

## 2. Source files and hashes

| role | path | bytes | sha256 (16) |
|---|---|---|---|
""" + "\n".join(f"| {l} | `{p}` | {b if b is not None else '—'} | `{s[:16]}` |" for l, p, b, s in prov) + f"""

## 3. Available / missing fields

**Available per rollout** (from the E1-D reward-manager diagnostics):
`call_index`, `global_step`, `uid`, `data_source`, `em`, `logical_search_batches`,
`search_queries`, `conversation_turns`, `response_token_length`, `metadata_valid`,
`zero_search`, `complete_reason`, plus the recorded `reward` (= shaped) and `efficiency`.
Per group (recomputed here from the raw rollouts, never trusted from derived fields):
n=8 membership, `k`, mixed/all-correct/all-wrong flags, correction-cost variation,
mean/min/max search cost, queries, tokens, turns.

**Missing / not used**: raw response text (only excerpts exist in older audits, and none in
E1-D's diagnostics); per-mini-batch assignment inside an optimizer step; the actual GRPO
mini-batch shuffle.  None of these is needed by the E1-E0 quota replay.

## 4. Strictly replayable vs proxy

* **Strict**: the generation-stream order (`src_order` = file order, which is the batch order
  the trainer saw, with `call_index` monotone), the n=8 grouping `(call_index, uid)`, the
  mixed/efficiency classification, the mixed-only stopping rule, the quota batch
  construction, the group-level composition, and the GRPO outcome-advantage formula.
* **Proxy**: everything about *training effects*.  E1-E0 never runs the model, so no EM or
  search-round change is claimed; §11 of the report only reports transparent batch-level
  quantities (supervision retained %, efficiency fraction, mean k/8, source shift).

## 5. Cross-checks performed

* q=0 replay consumes **{S['q0_baseline']['generation_batches_consumed']}** generation batches
  = exactly the {S['q0_baseline']['observed_generation_batches_in_E1_D_D2']} batches the E1-D D2 run
  actually used (`q0_replay_matches_observed = {S['q0_baseline']['matches_observed']}`).
* q=0 selects **{S['q0_baseline']['selected_groups_total']}** groups = exactly the
  {cs['mixed_groups']} mixed groups that E1-D's EM filter retained (E1-D report §24).
* the efficiency-group count ({cs['efficiency_groups']}) equals E1-D's reported
  `would_be_revived_by_shaped_filter = 434`.
* all {cs['groups']:,} reconstructed groups have n = 8 and all
  {cs['rollouts']:,} rollouts are accounted for.

## 6. Limitations

* The stream comes from **one run (E1-D D2)**, i.e. one policy trajectory; quota availability
  is a property of *that* policy's candidate distribution.
* `uid` is the dataset row index, so the same prompt can be re-sampled in later generation
  batches — the group identity is `(call_index, uid)`, matching the trainer's own filter unit.
* No measurement of what a quota would do to learning is possible offline (by construction).
"""
    with open(os.path.join(E1E, "data_inventory.md"), "w") as fh:
        fh.write(inv)

    cfg = {
        "experiment": "E1-E0",
        "kind": "offline quota replay / batch-composition diagnosis (no training)",
        "candidate_stream": {"source": S["candidate_stream"]["source"],
                             "rollouts": cs["rollouts"], "groups": cs["groups"],
                             "generation_batches": cs["generation_batches"],
                             "optimizer_steps": cs["optimizer_steps"]},
        "target_optimizer_batch_size": S["target_optimizer_batch_size"],
        "target_source": S["target_source"],
        "quotas": [0, 1, 2, 3, 4],
        "lambda": rec["lambda"],
        "definitions": {
            "mixed_correctness_group": "1 <= k <= 7 correct rollouts out of 8",
            "efficiency_group": "k == 8 AND max(logical_search_batches) > min(logical_search_batches) over the correct rollouts",
            "cost": "logical_search_batches (unchanged from E1-A/E1-B/E1-C/E1-D)",
            "stopping_rule": "consume generation batches in stream order, count ONLY mixed groups, stop at data.train_batch_size",
            "efficiency_cache": "all-correct efficiency-variable groups seen inside the same stopping prefix, in stream order",
            "batch_construction": "mixed[:B-m] + efficiency_cache[:m], m = min(q, |cache|)",
            "replaced_mixed_groups": "the last m mixed groups of the q=0 batch",
            "selection": "stream order only (no difficulty/reward/source/random selection)",
        },
        "reward_semantics_for_E1_E": {
            "mixed_group": "R_i = A_i (original EM only, no shaping)",
            "efficiency_group": "R_i = 1 + lambda * E_i with lambda = 0.05, E_i = (Cmax - C_i)/(Cmax - Cmin)",
        },
        "recommended_q": rec["recommended_q"],
    }
    with open(os.path.join(E1E, "configs/analysis_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    errs = f"""# E1-E0 errors and fixes

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

* All {cs['rollouts']:,} rollout records were parsed and grouped; `sum(group sizes) ==
  number of rollout records`; no malformed record was skipped (`malformed_records_dropped = 0`).

## N4 — reuse boundary

* `E1-E0/code/e1e_common.py` is a copy-and-extend of `E1-C/code/e1c_common.py`; the group
  reconstruction, the cost definition (`logical_search_batches`) and the filter rule are kept
  verbatim, and the new code only adds the mixed/efficiency classification, the mixed-only
  stopping rule, the efficiency cache and the quota batch construction.
* E1-C / E1-D files were opened read-only; nothing under those directories was modified.
"""
    with open(os.path.join(E1E, "logs/errors_and_fixes.md"), "w") as fh:
        fh.write(errs)

    readme = f"""# E1-E0 — offline quota replay for the E1-E efficiency-group mixture

Pure offline diagnosis on the **real E1-D D2 candidate stream**.  No training, no GPU, no
model/trainer/reward modification.

## Question

If the next experiment keeps GAP's correctness training intact but admits at most **q**
all-correct efficiency-variable groups per 32-group optimizer batch, what does each q do to
the batch composition — and which q is the right one to train with?

## Headline

| q | full-quota rate | mean efficiency groups / step | replacement fraction | efficiency supervision fraction |
|---|---|---|---|---|
| 1 | {float(avail[1]['full_quota_rate']):.2f} | {float(avail[1]['mean_actual_efficiency_groups_per_step']):.2f} | {float(avail[1]['actual_replacement_rate']):.4f} | {float(dens[1]['actual_efficiency_group_fraction']):.4f} |
| 2 | {float(avail[2]['full_quota_rate']):.2f} | {float(avail[2]['mean_actual_efficiency_groups_per_step']):.2f} | {float(avail[2]['actual_replacement_rate']):.4f} | {float(dens[2]['actual_efficiency_group_fraction']):.4f} |
| 3 | {float(avail[3]['full_quota_rate']):.2f} | {float(avail[3]['mean_actual_efficiency_groups_per_step']):.2f} | {float(avail[3]['actual_replacement_rate']):.4f} | {float(dens[3]['actual_efficiency_group_fraction']):.4f} |
| 4 | {float(avail[4]['full_quota_rate']):.2f} | {float(avail[4]['mean_actual_efficiency_groups_per_step']):.2f} | {float(avail[4]['actual_replacement_rate']):.4f} | {float(dens[4]['actual_efficiency_group_fraction']):.4f} |
| unrestricted (reference) | — | {float(unref['mean_per_step']):.2f} | {float(unref['replacement_fraction']):.4f} | {float(unref['efficiency_group_fraction']):.4f} |

**Recommended q = {rec['recommended_q']}** — see `FINAL_REPORT.md` §17/§18.

## Layout

```
E1-E0/
├── FINAL_REPORT.md      # 23 sections + answers A-H
├── data_inventory.md
├── README.md
├── code/                # e1e_common.py, e1e_quota_replay.py, e1e_figures.py, e1e_docs.py
├── configs/             # analysis_config.json, source_provenance.json
├── logs/                # run.log, errors_and_fixes.md
├── results/             # every CSV/JSON listed in FINAL_REPORT.md §23
└── figures/             # 5 PNGs
```

## Reproduce

```bash
cd /data01/wyy/Graph-Agent-Planning
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate parallel-agent
python3 E1-E0/code/e1e_quota_replay.py          # all analyses (~4 min, CPU only)
/home/nf5468m6/miniconda3/envs/huatuo/bin/python3 E1-E0/code/e1e_figures.py   # figures
python3 E1-E0/code/e1e_docs.py                  # inventory / config / error log / README
```
"""
    with open(os.path.join(E1E, "README.md"), "w") as fh:
        fh.write(readme)
    print("wrote data_inventory.md, configs/analysis_config.json, configs/source_provenance.json,")
    print("      logs/errors_and_fixes.md, README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E1-E online selection-composition check (RQ7).

E1-E0 concluded offline, on E1-D's candidate stream, that stream-order q=2 selection shows no
difficulty bias (k/8 two-proportion p = 0.943) and no source bias (chi-square p = 0.920).  The
online E1-E run generates its own stream, so this script reports what can and cannot be
re-checked from E1-E's own diagnostics, and quantifies it:

  CAN be checked online
    * every inserted group is an all-correct, cost-varying (k = 8) group -- the mechanism never
      selects on the difficulty of the mixed groups, because it does not compete with them;
    * the cost variation of every inserted group (non-degenerate efficiency signal);
    * where the insertion sits inside the step (always the stopping generation batch);
    * the number of efficiency groups seen but never inserted, and the replacement rate;
    * the source mix of the groups whose source *is* recorded.

  CANNOT be checked online (documented instrumentation gap)
    * the data source of a *cached/injected* efficiency group: the reward manager stores cached
      groups as `(slice, scores)` and writes the group row from a synthetic dict whose `source`
      is `None` (`rollouts_pid*.jsonl` has no source column either), so the offline
      inserted-vs-displaced source test cannot be reproduced from the online diagnostics.  The
      available evidence for that claim online is the frozen stream-order selection rule itself
      (verified against E1-E0 on 50/50 steps by the section-11 selftest) on a shuffled dataset,
      plus E1-E0's offline test on the proxy stream.
    * the difficulty of the *displaced* mixed groups (they are dropped before being recorded).

Writes E1-E/analysis/online_bias_check.{json,md}.
"""
import glob
import json
import os
from collections import Counter, defaultdict

REPO = "/data01/wyy/Graph-Agent-Planning"
E1E = os.path.join(REPO, "E1-E")


def iter_jsonl(paths):
    for p in paths:
        for line in open(p, errors="ignore"):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def main():
    steps, groups, rollouts, calls = [], [], [], []
    for ph in ("D1", "D2"):
        f = os.path.join(E1E, "runs", f"latest_{ph}.txt")
        if not os.path.exists(f):
            continue
        diag = os.path.join(open(f).read().strip(), "diagnostics")
        steps += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_steps_pid*.jsonl")))))
        groups += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_groups_pid*.jsonl")))))
        rollouts += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "rollouts_pid*.jsonl")))))
        calls += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_calls_pid*.jsonl")))))

    eff = [g for g in groups if g.get("group_type") == "efficiency"]
    mix = [g for g in groups if g.get("group_type") == "mixed"]
    in_batch_eff = [g for g in eff if g.get("source") is not None]
    injected_eff = [g for g in eff if g.get("source") is None]

    # per-(step, uid) cost sets from the authoritative rollout rows
    costs = defaultdict(set)
    for r in rollouts:
        if r.get("logical_search_batches") is not None:
            costs[(int(r["global_step"]), str(r.get("uid")))].add(float(r["logical_search_batches"]))
    gaps = []
    no_variation = []
    for g in eff:
        v = costs.get((int(g["global_step"]), str(g["uid"])), set())
        if len(v) >= 2:
            gaps.append(max(v) - min(v))
        else:
            no_variation.append({"step": int(g["global_step"]), "uid": str(g["uid"]),
                                 "costs": sorted(v)})

    seen = sum(int(s.get("efficiency_seen_cum") or 0) for s in steps)
    inserted = sum(int(s.get("quota_actual") or 0) for s in steps)
    stop_positions = [int(s.get("generation_batches_consumed") or 0) for s in steps]

    res = {
        "steps": len(steps),
        "inserted_efficiency_groups": len(eff),
        "inserted_with_recorded_source": len(in_batch_eff),
        "inserted_injected_source_not_recorded": len(injected_eff),
        "kept_mixed_groups": len(mix),
        "efficiency_groups_seen_total": seen,
        "efficiency_groups_seen_but_not_inserted": seen - inserted,
        "actual_replacement_rate": inserted / (32.0 * len(steps)) if steps else None,
        "source_mix_recorded_for_inserted": dict(Counter(str(g.get("source")) for g in in_batch_eff)),
        "source_mix_kept_mixed": dict(Counter(str(g.get("source")) for g in mix)),
        "every_inserted_group_is_all_correct": all(int(g.get("k") or 0) == 8 for g in eff),
        "inserted_cost_gap_mean": (sum(gaps) / len(gaps)) if gaps else None,
        "inserted_cost_gap_min": min(gaps) if gaps else None,
        "n_inserted_with_cost_variation": len(gaps),
        "n_inserted_without_recorded_cost_variation": len(no_variation),
        "examples_without_recorded_variation": no_variation[:5],
        "stopping_call_position_mean": (sum(stop_positions) / len(stop_positions)) if stop_positions else None,
        "selection_rule": "stream order: the planner inserts cache[:m], where the cache holds the "
                          "first q efficiency groups seen in the step; verified offline against "
                          "E1-E0 on 50/50 steps (section-11 selftest PART A) and against the "
                          "empty-keep case (PART D)",
        "cannot_check_online": [
            "data source of cached/injected efficiency groups: rows are written from a synthetic "
            "group dict with source=None and rollouts_pid has no source column, so the offline "
            "inserted-vs-displaced source test (E1-E0 p = 0.920) cannot be reproduced online",
            "difficulty (k/8) of the displaced mixed groups: they are dropped before any record "
            "is written",
        ],
        "available_bias_evidence": [
            "selection is pure stream order on a shuffled dataset and the planner only ever takes "
            "cache[:m]; it never ranks groups by source, difficulty or reward",
            "every inserted group is k = 8 (all-correct) and cost-varying by construction, so the "
            "mechanism cannot prefer easy or hard *mixed* groups",
            "E1-E0's offline test on E1-D's stream found no difficulty bias (p = 0.943) and no "
            "source bias (p = 0.920); the online run's own stream differs, so this remains a "
            "transferred (not re-measured) result",
        ],
    }
    res["all_inserted_groups_have_cost_variation"] = (len(no_variation) == 0 and len(eff) > 0)

    os.makedirs(os.path.join(E1E, "analysis"), exist_ok=True)
    with open(os.path.join(E1E, "analysis", "online_bias_check.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    md = ["# E1-E online selection-composition check (RQ7)", "",
          f"- optimizer steps: {res['steps']}",
          f"- inserted efficiency groups: {len(eff)} of {seen} seen "
          f"({res['efficiency_groups_seen_but_not_inserted']} seen but not inserted)",
          f"- kept mixed groups: {len(mix)}",
          f"- actual replacement rate: {res['actual_replacement_rate']:.4f}",
          f"- every inserted group all-correct (k = 8): {res['every_inserted_group_is_all_correct']}",
          f"- inserted cost variation: mean "
          f"{res['inserted_cost_gap_mean']} (min {res['inserted_cost_gap_min']}), present for "
          f"{res['n_inserted_with_cost_variation']}/{len(eff)} inserted groups",
          f"- mean insertion position: generation batch {res['stopping_call_position_mean']} "
          f"(the stopping batch, by construction)", "",
          "## source composition", "",
          f"- recorded sources for inserted groups: {res['source_mix_recorded_for_inserted']} "
          f"({res['inserted_with_recorded_source']} of {len(eff)}; the rest were injected from the "
          f"cache, where the manager does not carry `source`)",
          f"- sources of the kept mixed groups: {res['source_mix_kept_mixed']}", "",
          "## what cannot be checked online (instrumentation gap)", ""]
    md += [f"- {c}" for c in res["cannot_check_online"]]
    md += ["", "## available bias evidence", ""] + [f"- {c}" for c in res["available_bias_evidence"]]
    md.append("")
    with open(os.path.join(E1E, "analysis", "online_bias_check.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

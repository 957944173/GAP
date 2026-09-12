#!/usr/bin/env python3
"""E1-B per-checkpoint statistics (E1-B_task.md section 5).

For every saved checkpoint (step 80, 90, 100, 110, 120) reports, from the reward
manager diagnostics of the E1-B run:

    mean reward            (shaped reward R = A*(1+0.05E), what GRPO optimises)
    mean EM                (original EM accuracy, from data.batch["acc"] semantics)
    mean efficiency bonus  (E)
    active efficiency groups
    retained groups        (post-filter_groups, baseline-EM vs shaped-reward criteria)
    mean search batches    (logical_search_batches)
    mean queries
    parallel factor        (search_queries / logical_search_batches)

Two windows are reported per checkpoint:
    window_10step  -- only the 10 steps that produced that checkpoint
    cumulative     -- all steps from 71 to that checkpoint (the training trajectory)

Outputs (into --out, default E1-B/checkpoints):
    checkpoint_stats.json
    checkpoint_stats.csv
    checkpoint_stats.md
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

STEPS = [80, 90, 100, 110, 120]


def iter_jsonl(patterns):
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            with open(p, errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", default=",".join(str(s) for s in STEPS))
    args = ap.parse_args()

    e1b_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(e1b_root, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    diag = os.path.join(args.run_dir, "diagnostics")

    # ---------------- rollouts: streaming per-step aggregation ----------------
    step_acc = defaultdict(lambda: {
        "n_rollouts": 0, "sum_em": 0.0, "sum_reward": 0.0, "sum_eff": 0.0,
        "n_eff_nonzero": 0, "cost_sum": 0.0, "cost_n": 0,
        "query_sum": 0.0, "query_n": 0, "parallel_sum": 0.0, "parallel_n": 0,
        "turns_sum": 0.0, "turns_n": 0, "tokens_sum": 0.0, "tokens_n": 0,
        "zero_search": 0, "zero_search_correct": 0, "metadata_invalid": 0,
        "group_ids": set(),
    })
    for r in iter_jsonl([os.path.join(diag, "rollouts_pid*.jsonl")]):
        gs = r.get("global_step")
        if gs is None:
            continue
        d = step_acc[int(gs)]
        d["n_rollouts"] += 1
        em = 1.0 if (r.get("em") in (1, 1.0)) else 0.0
        d["sum_em"] += em
        if r.get("reward") is not None:
            d["sum_reward"] += float(r["reward"])
        if r.get("efficiency") is not None:
            e = float(r["efficiency"])
            d["sum_eff"] += e
            if e > 0:
                d["n_eff_nonzero"] += 1
        c = r.get("logical_search_batches")
        if c is not None:
            d["cost_sum"] += float(c)
            d["cost_n"] += 1
        q = r.get("search_queries")
        if q is not None:
            d["query_sum"] += float(q)
            d["query_n"] += 1
        if c and q:
            d["parallel_sum"] += float(q) / float(c)
            d["parallel_n"] += 1
        t = r.get("conversation_turns")
        if t is not None:
            d["turns_sum"] += float(t)
            d["turns_n"] += 1
        tk = r.get("response_token_length")
        if tk is not None:
            d["tokens_sum"] += float(tk)
            d["tokens_n"] += 1
        if r.get("zero_search"):
            d["zero_search"] += 1
            if em == 1.0:
                d["zero_search_correct"] += 1
        if not r.get("metadata_valid", True):
            d["metadata_invalid"] += 1
        d["group_ids"].add((r.get("call_index"), str(r.get("uid"))))

    # ---------------- groups: active / retained ----------------
    step_groups = defaultdict(lambda: {
        "n_groups": 0, "n_active": 0, "n_baseline_keep": 0, "n_shaped_keep": 0,
        "n_newly_added": 0, "sum_mean_eff": 0.0,
    })
    for g in iter_jsonl([os.path.join(diag, "groups_pid*.jsonl")]):
        gs = g.get("global_step")
        if gs is None:
            continue
        d = step_groups[int(gs)]
        d["n_groups"] += 1
        if g.get("efficiency_active"):
            d["n_active"] += 1
        if g.get("baseline_keep_em_std_gt_0"):
            d["n_baseline_keep"] += 1
        if g.get("shaped_keep_reward_std_gt_0"):
            d["n_shaped_keep"] += 1
        if g.get("newly_added_by_shaping"):
            d["n_newly_added"] += 1
        d["sum_mean_eff"] += float(g.get("mean_efficiency") or 0.0)

    def merge(steps):
        agg = {"n_rollouts": 0, "sum_em": 0.0, "sum_reward": 0.0, "sum_eff": 0.0,
               "n_eff_nonzero": 0, "cost_sum": 0.0, "cost_n": 0,
               "query_sum": 0.0, "query_n": 0, "parallel_sum": 0.0, "parallel_n": 0,
               "turns_sum": 0.0, "turns_n": 0, "tokens_sum": 0.0, "tokens_n": 0,
               "zero_search": 0, "zero_search_correct": 0, "metadata_invalid": 0,
               "group_ids": set()}
        gagg = {"n_groups": 0, "n_active": 0, "n_baseline_keep": 0, "n_shaped_keep": 0,
                "n_newly_added": 0, "sum_mean_eff": 0.0}
        for s in steps:
            d = step_acc.get(s)
            if d:
                for k in agg:
                    if k == "group_ids":
                        agg[k] |= d[k]
                    else:
                        agg[k] += d[k]
            g = step_groups.get(s)
            if g:
                for k in gagg:
                    gagg[k] += g[k]
        n = agg["n_rollouts"]
        res = {
            "steps": sorted(steps),
            "n_rollouts": n,
            "n_groups": len(agg["group_ids"]),
            "mean_reward": (agg["sum_reward"] / n) if n else None,
            "mean_em": (agg["sum_em"] / n) if n else None,
            "mean_efficiency_bonus": (agg["sum_eff"] / n) if n else None,
            "efficiency_nonzero_rate": (agg["n_eff_nonzero"] / n) if n else None,
            "active_efficiency_groups": gagg["n_active"],
            "retained_groups_shaped": gagg["n_shaped_keep"],
            "retained_groups_baseline_em": gagg["n_baseline_keep"],
            "newly_added_by_shaping": gagg["n_newly_added"],
            "mean_search_batches": (agg["cost_sum"] / agg["cost_n"]) if agg["cost_n"] else None,
            "mean_queries": (agg["query_sum"] / agg["query_n"]) if agg["query_n"] else None,
            "parallel_factor": (agg["parallel_sum"] / agg["parallel_n"]) if agg["parallel_n"] else None,
            "mean_conversation_turns": (agg["turns_sum"] / agg["turns_n"]) if agg["turns_n"] else None,
            "mean_response_tokens": (agg["tokens_sum"] / agg["tokens_n"]) if agg["tokens_n"] else None,
            "zero_search_rate": (agg["zero_search"] / n) if n else None,
            "zero_search_correct_count": agg["zero_search_correct"],
            "metadata_invalid": agg["metadata_invalid"],
        }
        return res

    all_steps = sorted(step_acc)
    rows = []
    for ck in [int(s) for s in args.steps.split(",") if s.strip()]:
        window = [s for s in all_steps if ck - 9 <= s <= ck]
        cum = [s for s in all_steps if 71 <= s <= ck]
        rows.append({
            "checkpoint": f"global_step_{ck}",
            "step": ck,
            "window_10step": merge(window) if window else None,
            "cumulative": merge(cum) if cum else None,
        })

    result = {
        "run_dir": args.run_dir,
        "steps_observed": all_steps,
        "checkpoints": rows,
    }
    with open(os.path.join(out_dir, "checkpoint_stats.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    fields = ["n_rollouts", "n_groups", "mean_reward", "mean_em", "mean_efficiency_bonus",
              "active_efficiency_groups", "retained_groups_shaped", "retained_groups_baseline_em",
              "newly_added_by_shaping", "mean_search_batches", "mean_queries", "parallel_factor",
              "mean_conversation_turns", "mean_response_tokens", "zero_search_rate",
              "zero_search_correct_count", "metadata_invalid"]
    with open(os.path.join(out_dir, "checkpoint_stats.csv"), "w") as fh:
        fh.write("checkpoint,window," + ",".join(fields) + "\n")
        for r in rows:
            for win in ("window_10step", "cumulative"):
                d = r[win]
                if not d:
                    continue
                fh.write(f"{r['checkpoint']},{win}," +
                         ",".join(("" if d.get(f) is None else f"{d[f]:.6f}" if isinstance(d[f], float) else str(d[f]))
                                  for f in fields) + "\n")

    def fmt(v, n=4):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.{n}f}"
        return str(v)

    with open(os.path.join(out_dir, "checkpoint_stats.md"), "w") as fh:
        fh.write("# E1-B per-checkpoint statistics\n\n")
        fh.write(f"run dir: `{args.run_dir}`\n\nsteps observed: {all_steps}\n\n")
        for win, title in (("window_10step", "last-10-step window"),
                           ("cumulative", "cumulative (step71..checkpoint)")):
            fh.write(f"## {title}\n\n")
            fh.write("| checkpoint | rollouts | groups | mean reward | mean EM | mean E | active E groups | "
                     "retained (shaped) | retained (EM) | search batches | queries | parallel factor |\n")
            fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                d = r[win]
                if not d:
                    continue
                fh.write(f"| {r['checkpoint']} | {d['n_rollouts']} | {d['n_groups']} | "
                         f"{fmt(d['mean_reward'])} | {fmt(d['mean_em'])} | {fmt(d['mean_efficiency_bonus'])} | "
                         f"{d['active_efficiency_groups']} | {d['retained_groups_shaped']} | "
                         f"{d['retained_groups_baseline_em']} | {fmt(d['mean_search_batches'])} | "
                         f"{fmt(d['mean_queries'])} | {fmt(d['parallel_factor'])} |\n")
            fh.write("\n")
    print(json.dumps({"out_dir": out_dir, "steps_observed": all_steps,
                      "checkpoints": [r["checkpoint"] for r in rows]}, indent=2))
    for r in rows:
        d = r["cumulative"]
        if d:
            print(f"step {r['step']:>3} cumulative: reward={fmt(d['mean_reward'])} em={fmt(d['mean_em'])} "
                  f"E={fmt(d['mean_efficiency_bonus'])} activeE={d['active_efficiency_groups']} "
                  f"retained={d['retained_groups_shaped']} rounds={fmt(d['mean_search_batches'])} "
                  f"queries={fmt(d['mean_queries'])} pf={fmt(d['parallel_factor'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

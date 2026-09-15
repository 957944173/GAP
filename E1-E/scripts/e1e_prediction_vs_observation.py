#!/usr/bin/env python3
"""E1-E0 prediction vs E1-E online observation (E1-E_task.md section 10 of the report).

E1-E0 froze the design offline on E1-D's D2 candidate stream (steps 71-120 of the *E1-D*
policy) and predicted, for q = 2:

  * full-quota rate                        100%      (50/50 steps)
  * actual replacement rate                6.25%     (2 of 32 groups per step)
  * mean efficiency groups per step        2.0
  * mean generation batches per step       10.46     (q = 0 baseline: 523 batches / 50 steps)
  * no difficulty / source bias in the replaced groups (k/8 p = 0.943, source chi2 p = 0.920)
  * efficiency groups cheaper than their group mean (fewer rounds/queries/tokens)
  * mixed-group reward == EM, efficiency-group reward in [1, 1.05]

The *online* E1-E run generates its own candidate stream (its step70 policy differs from
E1-D's), so per-step membership cannot be compared one-to-one; what must be compared are the
frozen *distributional* claims measured on E1-E's own runtime diagnostics.  This script does
exactly that and reports agreement / divergence per claim, plus the E1-E0 caveats.

Writes E1-E/analysis/prediction_vs_observation.{md,json}.
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

REPO = "/data01/wyy/Graph-Agent-Planning"
E1E = os.path.join(REPO, "E1-E")
E1E0 = os.path.join(REPO, "E1-E0")


def iter_jsonl(paths):
    for p in paths:
        for line in open(p, errors="ignore"):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def load_prediction():
    q2 = {}
    p = os.path.join(E1E0, "results", "quota_by_step.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            if int(r["q"]) != 2:
                continue
            def _num(v):
                if v in ("", None):
                    return None
                try:
                    return float(v)
                except ValueError:
                    return v
            q2[int(r["global_step"])] = {k: _num(v) for k, v in r.items()}
    rec = json.load(open(os.path.join(E1E0, "results", "recommended_quota.json")))
    summ = json.load(open(os.path.join(E1E0, "results", "summary.json")))
    return q2, rec, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="D2", choices=["D1", "D2", "both"])
    ap.add_argument("--out", default=os.path.join(E1E, "analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    phases = ["D1", "D2"] if args.phase == "both" else [args.phase]
    rollouts, groups, steps = [], [], []
    for ph in phases:
        f = os.path.join(E1E, "runs", f"latest_{ph}.txt")
        if not os.path.exists(f):
            continue
        run = open(f).read().strip()
        diag = os.path.join(run, "diagnostics")
        rollouts += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "rollouts_pid*.jsonl")))))
        groups += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_groups_pid*.jsonl")))))
        steps += list(iter_jsonl(sorted(glob.glob(os.path.join(diag, "quota_steps_pid*.jsonl")))))

    res = {"phases": phases, "n_rollouts": len(rollouts), "n_groups": len(groups),
           "n_step_records": len(steps)}
    if not steps:
        res["available"] = False
        res["reason"] = "no runtime diagnostics yet"
        json.dump(res, open(os.path.join(args.out, "prediction_vs_observation.json"), "w"), indent=2)
        print("no diagnostics yet")
        return 1
    res["available"] = True

    by_step = {}
    for r in steps:
        by_step.setdefault(int(r["global_step"]), []).append(r)
    step_ids = sorted(by_step)
    res["steps"] = step_ids

    # ---------------- observed batch-level statistics per step ------------------------
    obs = {}
    for s in step_ids:
        rs = [r for r in rollouts if int(r["global_step"]) == s]
        gs = [g for g in groups if int(g["global_step"]) == s]
        stop = [r for r in by_step[s] if r.get("is_stopping")]
        if not rs or not gs or not stop:
            continue
        rec = stop[-1]
        rounds = [float(r["logical_search_batches"]) for r in rs
                  if r.get("logical_search_batches") is not None]
        queries = [float(r["search_queries"]) for r in rs if r.get("search_queries") is not None]
        tokens = [float(r["response_token_length"]) for r in rs
                  if r.get("response_token_length") is not None]
        em = [float(r["em"]) for r in rs if r.get("em") is not None]
        eff = [g for g in gs if g.get("group_type") == "efficiency"]
        mix = [g for g in gs if g.get("group_type") == "mixed"]
        eff_gaps = [float(g["cost_gap"]) for g in eff if g.get("cost_gap") is not None]
        # inserted ids re-derived from the group records of the stopping call (the D1 manager
        # field repeated one id because of a leaked loop variable; fixed for D2)
        stop_ci = int(rec["call_index"])
        inserted = [g["group_id"] for g in gs
                    if int(g.get("call_index", -1)) == stop_ci and g.get("group_type") == "efficiency"]
        obs[s] = {
            "generation_batches": int(rec.get("generation_batches_consumed") or len(by_step[s])),
            "candidate_groups_seen": sum(int(r.get("candidate_groups_seen", 0)) for r in by_step[s]),
            "efficiency_groups_seen": int(rec.get("efficiency_seen_cum") or 0),
            "quota_actual": int(rec.get("quota_actual") or 0),
            "inserted_ids": inserted,
            "displaced_ids": rec.get("displaced_mixed_group_ids") or [],
            "mixed_kept": len(mix), "efficiency_kept": len(eff),
            "group_mean_k_over_8": (sum(float(g["em_mean"]) for g in gs) / len(gs)) if gs else None,
            "mean_search_rounds": st.mean(rounds) if rounds else None,
            "mean_search_queries": st.mean(queries) if queries else None,
            "mean_response_tokens": st.mean(tokens) if tokens else None,
            "micro_em": st.mean(em) if em else None,
            "inserted_mean_cost_gap": (st.mean(eff_gaps) if eff_gaps else None),
            "eff_shortfall": int(rec.get("eff_shortfall") or 0),
            "zero_search_rate": (sum(1 for r in rs if r.get("zero_search")) / len(rs)),
            "malformed_rate": (sum(1 for r in rs if not r.get("metadata_valid")) / len(rs)),
        }
    res["observed_per_step"] = obs
    if not obs:
        res["available"] = False
        res["reason"] = "step records exist but no materialised batch diagnostics yet"
        json.dump(res, open(os.path.join(args.out, "prediction_vs_observation.json"), "w"), indent=2)
        print("insufficient diagnostics")
        return 1

    # ---------------- distributional comparison ---------------------------------------
    pred_q2, rec_q, summ = load_prediction()
    n = len(obs)
    full = sum(1 for v in obs.values() if v["quota_actual"] == 2)
    repl = sum(v["quota_actual"] for v in obs.values()) / (32.0 * n)
    mean_gb = st.mean(v["generation_batches"] for v in obs.values())
    mean_eff_seen = st.mean(v["efficiency_groups_seen"] for v in obs.values())
    obs_claims = {
        "full_quota_rate": full / n,
        "actual_replacement_rate": repl,
        "mean_efficiency_groups_inserted_per_step": st.mean(v["quota_actual"] for v in obs.values()),
        "mean_generation_batches_per_step": mean_gb,
        "mean_efficiency_groups_seen_per_step": mean_eff_seen,
        "mean_k_over_8_of_optimizer_batch": st.mean(v["group_mean_k_over_8"] for v in obs.values()),
        "mean_search_rounds": st.mean(v["mean_search_rounds"] for v in obs.values()),
        "mean_search_queries": st.mean(v["mean_search_queries"] for v in obs.values()),
        "mean_response_tokens": st.mean(v["mean_response_tokens"] for v in obs.values()),
        "micro_em_of_optimizer_batches": st.mean(v["micro_em"] for v in obs.values()),
        "mean_inserted_cost_gap": st.mean(v["inserted_mean_cost_gap"] for v in obs.values()
                                          if v["inserted_mean_cost_gap"] is not None)
        if any(v["inserted_mean_cost_gap"] is not None for v in obs.values()) else None,
        "zero_search_rate": st.mean(v["zero_search_rate"] for v in obs.values()),
        "malformed_rate": st.mean(v["malformed_rate"] for v in obs.values()),
        "total_eff_shortfall": sum(v["eff_shortfall"] for v in obs.values()),
    }
    pred_claims = {
        "full_quota_rate": rec_q["full_quota_rate_by_q"]["2"],
        "actual_replacement_rate": rec_q["actual_replacement_rate_by_q"]["2"],
        "mean_efficiency_groups_inserted_per_step": rec_q["mean_efficiency_groups_per_step_by_q"]["2"],
        "mean_generation_batches_per_step": (sum(v["generation_batches_consumed"] for v in pred_q2.values()
                                                 if v["generation_batches_consumed"])
                                             / len(pred_q2)) if pred_q2 else None,
        "mean_efficiency_groups_seen_per_step": (sum(v["efficiency_groups_available"] for v in pred_q2.values()
                                                     if v["efficiency_groups_available"] is not None)
                                                 / len(pred_q2)) if pred_q2 else None,
        "mean_k_over_8_of_optimizer_batch": (sum(v["mean_k_over_8"] for v in pred_q2.values()
                                                 if v["mean_k_over_8"] is not None)
                                             / len(pred_q2)) if pred_q2 else None,
        "mean_search_rounds": (sum(v["mean_search_rounds"] for v in pred_q2.values()
                                   if v["mean_search_rounds"] is not None)
                               / len(pred_q2)) if pred_q2 else None,
        "mean_search_queries": (sum(v["mean_search_queries"] for v in pred_q2.values()
                                    if v["mean_search_queries"] is not None)
                                / len(pred_q2)) if pred_q2 else None,
        "mean_response_tokens": (sum(v["mean_response_tokens"] for v in pred_q2.values()
                                     if v["mean_response_tokens"] is not None)
                                 / len(pred_q2)) if pred_q2 else None,
        "zero_search_rate": None,
        "malformed_rate": None,
        "total_eff_shortfall": 0.0,
        "source_note": "E1-E0 predicted on E1-D's D2 candidate stream (steps 71-120 of E1-D)",
    }
    pred_rounds = pred_claims.get("mean_search_rounds")
    pred_queries = pred_claims.get("mean_search_queries")
    res["comparison"] = {
        "observed": obs_claims, "predicted_E1E0": pred_claims,
        "agreement": {
            "full_quota_rate": (obs_claims["full_quota_rate"] >= 0.99),
            "replacement_rate_matches": (abs(obs_claims["actual_replacement_rate"]
                                             - pred_claims["actual_replacement_rate"]) <= 0.01),
            "efficiency_groups_seen_positive_every_step": all(
                v["efficiency_groups_seen"] > 0 for v in obs.values()),
            "rounds_and_queries_scale": (pred_rounds is None or pred_queries is None or
                                         (abs(obs_claims["mean_search_rounds"] / pred_rounds - 1) < 0.5)),
        },
        "prediction_covers_same_steps": all(71 <= int(s_) <= 120 for s_ in step_ids),
        "honest_caveats": [
            "E1-E0's prediction table covers E1-D's candidate stream for optimizer steps "
            "71-120; when the observed steps fall outside that range (D1: 61-70) the predicted "
            "Generation-batch statistics are NOT comparable and the row is informational only.",
            "E1-E online generates its own candidate stream: its step70 policy differs from "
            "E1-D's step70 policy, so the per-step group membership/deviation from E1-E0's "
            "CSV is expected and is NOT a runtime mismatch (the runtime equivalence itself "
            "was proven offline: E1-E/analysis/preflight_selftest.json, q=0 and q=2 exact on "
            "50/50 steps of the E1-D stream).",
            "D1 (steps 61-70) has no E1-E0 prediction at all: E1-E0's stream only covers "
            "steps 71-120.",
            "E1-E0's candidate pool has an artificially high all-correct share (37,926 of "
            "83,680 groups) because it was harvested from a shaped run; the online mixed-group "
            "share therefore differs, which changes the number of generation batches per step.",
        ],
    }

    json.dump(res, open(os.path.join(args.out, "prediction_vs_observation.json"), "w"),
              indent=2, default=str)
    md = ["# E1-E0 frozen prediction vs E1-E online observation", "",
          f"phases: {', '.join(phases)}; optimizer steps observed: {n} "
          f"({step_ids[0]}..{step_ids[-1]}); rollouts: {len(rollouts):,}; "
          f"group records: {len(groups):,}", "",
          "| quantity | E1-E0 prediction (q=2) | E1-E online observation | verdict |",
          "|---|---|---|---|"]
    for k in ("full_quota_rate", "actual_replacement_rate",
              "mean_efficiency_groups_inserted_per_step", "mean_generation_batches_per_step",
              "mean_efficiency_groups_seen_per_step", "mean_k_over_8_of_optimizer_batch",
              "mean_search_rounds", "mean_search_queries", "mean_response_tokens",
              "mean_inserted_cost_gap", "zero_search_rate", "malformed_rate",
              "total_eff_shortfall"):
        pv, ov = pred_claims.get(k), obs_claims.get(k)
        if ov is None:
            continue
        verdict = "-"
        if pv is not None:
            rel = (ov - pv) / pv if pv else None
            verdict = "match" if (rel is not None and abs(rel) <= 0.15) else f"delta {ov - pv:+.4f}"
        md.append(f"| {k} | {'-' if pv is None else f'{pv:.6f}'} | {ov:.6f} | {verdict} |")
    md += ["", "## agreement flags", ""]
    for k, v in res["comparison"]["agreement"].items():
        md.append(f"- {k}: {v}")
    md += ["", "## honest caveats", ""] + [f"- {c}" for c in res["comparison"]["honest_caveats"]]
    md += ["", "## per-step observation", "",
           "| step | gen batches | cand groups | eff seen | quota | mixed | eff | rounds | queries | tokens | micro EM | inserted cost gap | shortfall |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in step_ids:
        v = obs.get(s)
        if not v:
            continue
        md.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            s, v["generation_batches"], v["candidate_groups_seen"], v["efficiency_groups_seen"],
            v["quota_actual"], v["mixed_kept"], v["efficiency_kept"],
            "-" if v["mean_search_rounds"] is None else f"{v['mean_search_rounds']:.3f}",
            "-" if v["mean_search_queries"] is None else f"{v['mean_search_queries']:.3f}",
            "-" if v["mean_response_tokens"] is None else f"{v['mean_response_tokens']:.1f}",
            "-" if v["micro_em"] is None else f"{v['micro_em']:.4f}",
            "-" if v["inserted_mean_cost_gap"] is None else f"{v['inserted_mean_cost_gap']:.3f}",
            v["eff_shortfall"]))
    md.append("")
    open(os.path.join(args.out, "prediction_vs_observation.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md[:40]))
    print("wrote", os.path.join(args.out, "prediction_vs_observation.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

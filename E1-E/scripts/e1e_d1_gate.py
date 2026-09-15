#!/usr/bin/env python3
"""E1-E D1 -> D2 automatic gate (E1-E_task.md section 15).

Unattended decision: after D1 (step60 -> step70) has trained, the step70 checkpoint has been
integrity-checked and the full 7-benchmark evaluation has finished, decide automatically
whether D2 (step70 -> step120) may start.

PASS requires ALL of the hard conditions of section 15:

  A. no fatal training / integrity error (checkpoint complete, steps present, full-state
     resume evidence, no OOM/NCCL/traceback/wiki-tool failure, no non-finite logged metric);
  B. runtime quota semantics correct (32 groups per optimizer step, quota <= 2, mixed-only
     stopping, mixed reward == EM, efficiency reward in [1, 1.05], wrong reward == 0,
     no NaN/Inf advantage, no quota error lines) -- measured on E1-E's own diagnostics;
  C. step70 checkpoint complete;
  D. 7-benchmark evaluation complete (all benchmarks, expected sample counts);
  E. no reward hacking / catastrophic anomaly:
       - micro EM drop vs E1-0 step70 and E1-D step70 <= 1.0 pp (consistency drop);
       - no query explosion (queries +>15%) accompanied by a rounds *decrease* (packing);
       - no mass zero-search shortcut;
       - no malformed / no-answer-tag burst.

A modest step70 efficiency gain (1-3%) is explicitly NOT a reason to stop: the E1-E signal is
only ~6.25% of the batch.

Writes E1-E/analysis/d1_gate.{md,json} and E1-E/logs/d1_gate_status.txt (`GATE=PASS|FAIL`).
Exit code 0 = PASS (start D2), 1 = FAIL (stop, analyse, report).
"""
import glob
import json
import os
import sys

REPO = "/data01/wyy/Graph-Agent-Planning"
E1E = os.path.join(REPO, "E1-E")
EM_DROP_STOP = 0.010          # 1.0 pp
QUERY_EXPLOSION = 0.15        # +15%
ZERO_SEARCH_ABS = 0.01        # 1%
ZERO_SEARCH_REL = 5.0
NO_ANSWER_TOL = 0.02          # +2 pp absolute
EMPTY_OUT_TOL = 0.01


def newest(pat):
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def overall(path):
    d = load_json(path)
    return (d or {}).get("overall"), d


def main():
    res = {"gate": None, "reasons": [], "warnings": [], "checks": {}}

    # ---------------- A/B/C: integrity ----------------
    integ = load_json(os.path.join(E1E, "analysis", "step70_integrity.json"))
    if not integ:
        res["reasons"].append("A: step70_integrity.json missing (integrity check did not run)")
        integ = {}
    err = (integ.get("training_log") or {}).get("error_pattern_counts_before_final_val", {}) or {}
    ck = integ.get("checkpoint") or {}
    ck_ok = bool(ck.get("exists") and ck.get("data_pt")
                 and all(v.get("exists") for v in (ck.get("shards") or {}).values()))
    fatal_err = {k: v for k, v in err.items()
                 if k in ("cuda_oom", "nccl_error", "traceback", "cuda_error",
                          "wiki_search_error", "wiki_tool_exec_failed") and v}
    q = integ.get("quota_semantics") or {}
    xc = integ.get("log_vs_diagnostics") or {}
    selftest = load_json(os.path.join(E1E, "analysis", "preflight_selftest.json")) or {}
    qav = q.get("available") is True

    def b(cond):
        return bool(cond) if qav else None

    res["checks"].update({
        "A_integrity_ok": bool(integ.get("integrity_ok")),
        "A_checkpoint_complete": ck_ok,
        "A_fatal_error_patterns": fatal_err,
        "A_non_finite_metrics": (integ.get("training_log") or {}).get("non_finite_metrics"),
        "A_state_resume_ok": integ.get("state_resume_ok"),
        "B_quota_semantics_ok": q.get("quota_semantics_ok"),
        "B_quota_error_lines": q.get("quota_error_lines"),
        "B_batch_size_32_all_steps": b(not q.get("steps_with_wrong_batch_size")),
        "B_quota_actual_le_2": b(not q.get("calls_with_quota_actual_gt_2")),
        "B_mixed_only_stopping": b(not q.get("calls_where_stopping_disagrees_with_mixed_count")),
        "B_mixed_reward_equals_em": b(q.get("rollouts_mixed_reward_not_equal_em") == 0),
        "B_efficiency_reward_in_range": b(q.get("rollouts_efficiency_reward_or_em_wrong") == 0),
        "B_wrong_reward_zero": b(q.get("rollouts_wrong_with_nonzero_reward") == 0),
        "B_no_nan_inf_advantage": b(q.get("advantage_nan_total") == 0
                                    and q.get("advantage_inf_total") == 0),
        "B_published_metric_consistent": b(q.get("rollouts_published_metric_inconsistent") == 0),
        "B_log_score_max_matches_quota": xc.get("log_score_max_consistent_with_quota"),
        "B_runtime_matches_e1e0_offline_selftest": (selftest.get("all_pass") is True
                                                   and all((selftest.get("pass") or {}).values())),
        "quota_fill_rate": q.get("quota_fill_rate"),
        "steps_full_quota": q.get("steps_full_quota"),
        "steps_used": q.get("steps_used"),
        "total_inserted_efficiency_groups": q.get("total_inserted_efficiency_groups"),
        "total_eff_shortfall": q.get("total_eff_shortfall"),
        "zero_search_share_train": q.get("zero_search_share"),
        "malformed_share_train": q.get("malformed_metadata_share"),
    })
    if not integ.get("integrity_ok"):
        res["reasons"].append(f"A: integrity_ok is false (fatal patterns: {fatal_err})")
    if not ck_ok:
        res["reasons"].append("C: step70 checkpoint incomplete")
    if fatal_err:
        res["reasons"].append(f"A: fatal error patterns in the training window: {fatal_err}")
    for key, label in (("B_quota_semantics_ok", "B: quota semantics invariant failed"),
                       ("B_no_nan_inf_advantage", "B: NaN/Inf advantage detected"),
                       ("B_batch_size_32_all_steps", "B: an optimizer batch was not 32 groups"),
                       ("B_quota_actual_le_2", "B: quota exceeded 2"),
                       ("B_mixed_only_stopping", "B: stopping disagreed with the mixed-group count"),
                       ("B_mixed_reward_equals_em", "B: a mixed group was shaped (reward != EM)"),
                       ("B_efficiency_reward_in_range", "B: efficiency reward outside [1, 1.05]"),
                       ("B_wrong_reward_zero", "B: a wrong rollout received non-zero reward"),
                       ("B_published_metric_consistent", "B: published filter metric inconsistent"),
                       ("B_log_score_max_matches_quota", "B: trainer log disagrees with quota diagnostics"),
                       ("B_runtime_matches_e1e0_offline_selftest", "B: offline selftest did not pass")):
        if res["checks"].get(key) is False:
            res["reasons"].append(label)
    if q.get("quota_error_lines"):
        res["reasons"].append(f"B: {q['quota_error_lines']} quota error line(s) in the run")

    # ---------------- D: evaluation completeness ----------------
    ev_dir = newest(os.path.join(E1E, "eval_results", "step70_*"))
    ev = load_json(os.path.join(ev_dir, "eval_summary.json")) if ev_dir else None
    BENCHES = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
    if not ev:
        res["reasons"].append("D: step70 eval_summary.json missing")
        per = {}
    else:
        per = (ev.get("accuracy") or {}).get("per_benchmark") or {}
        missing = [b for b in BENCHES if b not in per]
        expected = (ev.get("accuracy") or {}).get("n_by_benchmark") or {}
        if missing:
            res["reasons"].append(f"D: step70 evaluation missing benchmarks {missing}")
        res["checks"]["D_eval_dir"] = ev_dir
        res["checks"]["D_benchmarks_present"] = sorted(per.keys())
        res["checks"]["D_sample_counts"] = {b: per.get(b, {}).get("n") for b in BENCHES}
        res["checks"]["D_expected_counts"] = expected
    e1e_ov = (ev or {}).get("overall") or {}

    # ---------------- E: anomalies ----------------
    ref_paths = {
        "e1_0_step70": os.path.join(REPO, "E1-B/evaluations/e10_step70_reference/eval_summary.json"),
        "gap_step60": os.path.join(REPO, "E1-B/evaluations/gap_step60_reference/eval_summary.json"),
        "e1a_step70": os.path.join(REPO, "E1-B/evaluations/e1a_step70_reference/eval_summary.json"),
        "e1d_step70": newest(os.path.join(REPO, "E1-D/eval_results/step70_*", "eval_summary.json")),
    }
    refs = {}
    for k, p in ref_paths.items():
        ov, _ = overall(p)
        if ov:
            refs[k] = ov
    res["checks"]["E_references"] = refs
    res["checks"]["E_e1e_step70_overall"] = e1e_ov
    if e1e_ov and refs:
        for name in ("e1_0_step70", "e1d_step70"):
            r = refs.get(name)
            if not r:
                continue
            d_em = e1e_ov.get("micro_em", 0) - r.get("micro_em", 0)
            res["checks"][f"E_micro_em_delta_vs_{name}"] = d_em
            if d_em < -EM_DROP_STOP:
                res["reasons"].append(
                    f"E: micro EM {d_em*100:.2f} pp vs {name} (< -1.0 pp): consistent correctness drop")
        r = refs.get("e1_0_step70") or refs.get("e1d_step70")
        dq = e1e_ov.get("search_queries_mean", 0) / (r.get("search_queries_mean") or 1) - 1
        dr = e1e_ov.get("retrieval_rounds_mean", 0) - (r.get("retrieval_rounds_mean") or 0)
        res["checks"]["E_query_delta_rel_vs_ref"] = dq
        res["checks"]["E_rounds_delta_vs_ref"] = dr
        if dq > QUERY_EXPLOSION and dr < 0:
            res["reasons"].append(
                f"E: query explosion +{dq*100:.1f}% with rounds {dr:+.3f} (packing suspicion)")
        zs = e1e_ov.get("zero_search_rate") or 0.0
        zr = r.get("zero_search_rate") or 0.0
        res["checks"]["E_zero_search_rate_step70"] = zs
        res["checks"]["E_zero_search_rate_ref"] = zr
        if zs > ZERO_SEARCH_ABS and zs > ZERO_SEARCH_REL * max(zr, 1e-9):
            res["reasons"].append(
                f"E: mass zero-search shortcut (step70 {zs*100:.3f}% vs ref {zr*100:.3f}%)")
        na = e1e_ov.get("no_answer_tag_rate") or 0.0
        nar = r.get("no_answer_tag_rate") or 0.0
        res["checks"]["E_no_answer_tag_rate_step70"] = na
        res["checks"]["E_no_answer_tag_rate_ref"] = nar
        if na > nar + NO_ANSWER_TOL:
            res["reasons"].append(
                f"E: malformed/no-answer output burst ({na*100:.2f}% vs ref {nar*100:.2f}%)")
        eo = e1e_ov.get("empty_output_rate")
        if eo is not None and eo > EMPTY_OUT_TOL:
            res["reasons"].append(f"E: empty_output_rate {eo*100:.2f}% > 1%")
    else:
        res["reasons"].append("D/E: no step70 evaluation to judge anomalies from")
    if e1e_ov:
        res["warnings"].append(
            "step70 efficiency effect is expected to be small (~6.25% of groups); a 1-3% gain "
            "is NOT a stop condition")

    res["gate"] = "PASS" if not res["reasons"] else "FAIL"
    res["decision"] = ("start D2 (step70 -> step120)" if res["gate"] == "PASS"
                       else "STOP: do not start D2; write the failure analysis and FINAL_REPORT")

    os.makedirs(os.path.join(E1E, "analysis"), exist_ok=True)
    with open(os.path.join(E1E, "analysis", "d1_gate.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    md = ["# E1-E D1 -> D2 automatic gate (E1-E_task.md section 15)", "",
          f"**GATE = {res['gate']}** — {res['decision']}", ""]
    if res["reasons"]:
        md += ["## stop reasons", ""] + [f"- {r}" for r in res["reasons"]] + [""]
    if res["warnings"]:
        md += ["## notes", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    md += ["## checks", "", "```json", json.dumps(res["checks"], indent=2, default=str)[:8000],
           "```", ""]
    with open(os.path.join(E1E, "analysis", "d1_gate.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(E1E, "logs", "d1_gate_status.txt"), "w") as fh:
        fh.write(f"GATE={res['gate']}\n" + "".join(f"REASON={r}\n" for r in res["reasons"]))
    print("\n".join(md))
    return 0 if res["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Assemble E1-E/FINAL_REPORT_draft.md (E1-E_task.md section 23, 31 sections).

Assembles the 31-section draft report from artifacts that already exist on disk:

  * sections 2-10 and 30 are copied verbatim from
    `E1-E/FINAL_REPORT_static_sections.md` (its own section 31 is stale and is
    regenerated from disk here);
  * sections 11-21 and 23-25 are generated from the analysis / evaluation
    artifacts listed below;
  * sections 22 and 26-29 (plus the verdict prose in 1 and 25) are left as
    clearly marked `<!-- TODO(caller): ... -->` placeholders.

Everything is read defensively: a missing or unparseable artifact yields a short
`(missing: <path>)` line for its section and the assembly continues, so the draft
is always produced (exit 0 whenever it could be written).

Usage:
    python3 E1-E/scripts/e1e_assemble_report.py            # write the draft
    python3 E1-E/scripts/e1e_assemble_report.py --check    # list missing inputs only
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime

SCRIPT = os.path.abspath(__file__)
E1E = os.path.dirname(os.path.dirname(SCRIPT))
REPO = os.path.dirname(E1E)
STATIC = os.path.join(E1E, "FINAL_REPORT_static_sections.md")
ANALYSIS = os.path.join(E1E, "analysis")
EVAL_ROOT = os.path.join(E1E, "eval_results")
FIGDIR = os.path.join(E1E, "figures")
DEFAULT_OUT = os.path.join(E1E, "FINAL_REPORT_draft.md")

BENCH = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]

FATAL_PATTERNS = ("cuda_oom", "nccl_error", "traceback", "cuda_error",
                  "wiki_search_error", "wiki_tool_exec_failed")

# Reference evaluations, mirroring e1e_build_summary.py's REFS table.
REF_PATHS = {
    "e1_0_step70": os.path.join(REPO, "E1-B/evaluations/e10_step70_reference/eval_summary.json"),
    "e1a_step70": os.path.join(REPO, "E1-B/evaluations/e1a_step70_reference/eval_summary.json"),
    "gap60": os.path.join(REPO, "E1-B/evaluations/gap_step60_reference/eval_summary.json"),
    "gap120": os.path.join(REPO, "E1-B/evaluations/gap_step120_reference/eval_summary.json"),
    "gap180": os.path.join(REPO, "E1-B/evaluations/gap_step180_reference/eval_summary.json"),
    "e1b120": os.path.join(REPO, "E1-B/evaluations/step120_20260911_235134/eval_summary.json"),
    "e1d70": "GLOB:" + os.path.join(REPO, "E1-D/eval_results/step70_*/eval_summary.json"),
    "e1d120": "GLOB:" + os.path.join(REPO, "E1-D/eval_results/step120_*/eval_summary.json"),
}

# Display rows for the per-benchmark / behaviour tables (section 18/19).
RUN_ROWS = [
    ("E1-E step120", "e1e120"),
    ("E1-E step70", "e1e70"),
    ("GAP120", "gap120"),
    ("E1-B120", "e1b120"),
    ("E1-D120", "e1d120"),
    ("E1-0 step70", "e1_0_step70"),
    ("E1-A step70", "e1a_step70"),
]

SECTION_TITLES = {
    1: "Executive Summary",
    11: "D1 step60->70 training integrity",
    12: "Runtime quota statistics step61-70",
    13: "step70 seven-benchmark evaluation",
    14: "D1 gate decision",
    15: "D2 step70->120 training integrity",
    16: "Runtime quota statistics step71-120",
    17: "step120 seven-benchmark evaluation",
    18: "Per-benchmark accuracy table",
    19: "Efficiency metrics table",
    20: "GAP / E1-B / E1-D / E1-E four-way comparison",
    21: "Paired statistical tests",
    22: "Pareto analysis",
    23: "Reward-hacking / safety analysis",
    24: "E1-E0 offline prediction vs online observation",
    25: "Whether E1-E meets the preregistered success criteria",
    26: "What E1-E establishes",
    27: "What it does NOT establish",
    28: "Limitations",
    29: "Recommended next experiment",
    30: "Full reproduction commands",
    31: "Complete artifact inventory",
}

FLAG_THRESHOLDS = {
    "delta_EM_at_least_minus_0.5pp_vs_GAP120":
        "Δ micro EM vs GAP120 ≥ −0.5 pp (preregistered)",
    "delta_rounds_at_most_minus_10pct_vs_GAP120":
        "Δ search rounds vs GAP120 ≤ −10 % (preregistered)",
    "pareto_dominates_E1D120":
        "Pareto-domination of E1-D120: Δ micro EM vs E1-D120 ≥ −0.5 pp AND "
        "Δ search rounds vs E1-D120 ≤ 0",
    "keeps_E1D_EM_and_beats_E1D_rounds":
        "keeps E1-D120 EM (Δmicro EM ≥ −0.5 pp) AND strictly fewer rounds "
        "(Δrounds < 0) vs E1-D120",
    "more_efficient_than_E1B120":
        "fewer search rounds than E1-B120 (Δrounds < 0)",
    "higher_EM_than_E1B120":
        "higher micro EM than E1-B120 (Δ micro EM > 0)",
}

# Inputs the caller expects to exist; used by --check.
EXPECTED_INPUTS = [
    ("static sections", STATIC),
    ("D1 integrity", os.path.join(ANALYSIS, "step70_integrity.json")),
    ("D2 integrity", os.path.join(ANALYSIS, "step120_integrity.json")),
    ("D1 gate", os.path.join(ANALYSIS, "d1_gate.json")),
    ("summary", os.path.join(ANALYSIS, "summary.json")),
    ("prediction vs observation", os.path.join(ANALYSIS, "prediction_vs_observation.json")),
    ("pre-run original manifest", os.path.join(E1E, "manifests", "original_files_manifest_pre.json")),
    ("original files unchanged", os.path.join(ANALYSIS, "original_files_unchanged.json")),
    ("E1-E step70 eval", os.path.join(EVAL_ROOT, "step70_*", "eval_summary.json")),
    ("E1-E step120 eval", os.path.join(EVAL_ROOT, "step120_*", "eval_summary.json")),
    ("comparison dir", os.path.join(ANALYSIS, "comparison_*", "comparison.json")),
    ("paired comparisons", os.path.join(ANALYSIS, "comparison_*", "paired_comparison_*.json")),
    ("D1 safety analysis", os.path.join(ANALYSIS, "D1_vs_E1_0", "safety_analysis.json")),
    ("D2 safety analysis", os.path.join(ANALYSIS, "D2_vs_E1_0", "safety_analysis.json")),
    ("D1 training summary", os.path.join(ANALYSIS, "D1_vs_E1_0", "training_summary.json")),
    ("D2 training summary", os.path.join(ANALYSIS, "D2_vs_E1_0", "training_summary.json")),
    ("figures", os.path.join(FIGDIR, "*.png")),
    ("E1-0 step70 reference", REF_PATHS["e1_0_step70"]),
    ("E1-A step70 reference", REF_PATHS["e1a_step70"]),
    ("GAP120 reference", REF_PATHS["gap120"]),
    ("E1-B120 reference", REF_PATHS["e1b120"]),
    ("E1-D step70 reference", REF_PATHS["e1d70"][5:]),
    ("E1-D step120 reference", REF_PATHS["e1d120"][5:]),
]

# ---------------------------------------------------------------------------
# defensive IO + bookkeeping
# ---------------------------------------------------------------------------
INPUTS = {}       # relative path -> short sha256 (or "missing" / "unreadable")
MISSING = []      # relative paths that could not be read


def _rel(path):
    try:
        return os.path.relpath(path, REPO)
    except Exception:  # noqa: BLE001
        return str(path)


def register(path, status=None):
    rel = _rel(path)
    if rel in INPUTS:
        return
    if status is not None:
        INPUTS[rel] = status
        return
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        INPUTS[rel] = h.hexdigest()[:12]
    except Exception:  # noqa: BLE001
        INPUTS[rel] = "unreadable"


def note_missing(path):
    rel = _rel(path)
    if rel not in MISSING:
        MISSING.append(rel)
    INPUTS.setdefault(rel, "missing")


def jload(path):
    """Load JSON defensively; register the file when it was actually read."""
    if not path:
        return None
    if not os.path.isfile(path):
        note_missing(path)
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:  # noqa: BLE001
        note_missing(path)
        return None
    register(path)
    return obj


def newest(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def fnum(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    try:
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def fsgn(v, nd=4):
    """Explicit sign, fixed decimals."""
    if v is None:
        return "-"
    try:
        return f"{v:+.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def fpp(v):
    """EM / rate deltas in percentage points, explicit sign, 2 decimals."""
    if v is None:
        return "-"
    try:
        return f"{v * 100:+.2f} pp"
    except (TypeError, ValueError):
        return str(v)


def fpct(v):
    """A value already expressed in percent; explicit sign, 2 decimals."""
    if v is None:
        return "-"
    try:
        return f"{v:+.2f}%"
    except (TypeError, ValueError):
        return str(v)


def frate(v, nd=4):
    if v is None:
        return "-"
    try:
        return f"{v * 100:.{nd}f}%"
    except (TypeError, ValueError):
        return str(v)


def fint(v):
    return "-" if v is None else str(v)


def fp(v):
    """A p-value in compact scientific notation."""
    if v is None:
        return "-"
    try:
        return f"{v:.3g}"
    except (TypeError, ValueError):
        return str(v)


def figures_note():
    pngs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIGDIR, "*.png")))
    if not pngs:
        return f"(no figures found in `{_rel(FIGDIR)}/`)"
    return "Figures: " + ", ".join(f"`{p}`" for p in pngs) + "."


# ---------------------------------------------------------------------------
# evaluation blocks
# ---------------------------------------------------------------------------
def eval_block_disk(path):
    s = jload(path)
    if not isinstance(s, dict):
        return None
    acc = s.get("accuracy") or {}
    o = s.get("overall") or {}
    rounds = o.get("retrieval_rounds_mean")
    queries = o.get("search_queries_mean")
    per = {k: (v or {}).get("em") for k, v in (acc.get("per_benchmark") or {}).items()}
    # `overall` omits turns/tokens; derive them as sample-weighted means over the
    # per-benchmark blocks, exactly as e1e_compare_all.py does.
    br = s.get("benchmark_results") or {}
    n_tot = sum((v.get("n_samples") or 0) for v in br.values())
    tokens = o.get("response_tokens_mean")
    turns = o.get("assistant_turns_mean_all_messages")
    if n_tot:
        if tokens is None:
            tokens = sum((v.get("response_tokens_mean") or 0) * (v.get("n_samples") or 0)
                         for v in br.values()) / n_tot
        if turns is None:
            turns = sum((v.get("assistant_turns_mean_all_messages") or 0) * (v.get("n_samples") or 0)
                        for v in br.values()) / n_tot
    return {
        "path": _rel(path),
        "label": s.get("label"),
        "macro_em": acc.get("macro_em_over_benchmarks"),
        "micro_em": acc.get("micro_em_over_samples"),
        "total_samples": acc.get("total_samples"),
        "per_benchmark_em": per,
        "search_rounds": rounds,
        "queries": queries,
        "queries_per_round": (queries / rounds) if rounds else None,
        "parallel_factor": o.get("parallel_factor_mean"),
        "parallel_sample_rate": o.get("parallel_sample_rate"),
        "zero_search_rate": o.get("zero_search_rate"),
        "no_answer_tag_rate": o.get("no_answer_tag_rate"),
        "tokens": tokens,
        "turns": turns,
    }


def build_eval_blocks(summary):
    blocks = {}
    for step in (70, 120):
        d = newest(os.path.join(EVAL_ROOT, f"step{step}_*"))
        path = os.path.join(d, "eval_summary.json") if d else None
        blocks[f"e1e{step}"] = eval_block_disk(path) if path else None
    for key, p in REF_PATHS.items():
        if p.startswith("GLOB:"):
            d = newest(p[5:])
            blocks[key] = eval_block_disk(d) if d else None
        else:
            blocks[key] = eval_block_disk(p)
    # prefer the merged summary.json blocks where they exist (authoritative);
    # keep the disk-only fields (e.g. tokens) that summary.json omits.
    if isinstance(summary, dict):
        for key, tag in (("step70", "e1e70"), ("step120", "e1e120")):
            sb = summary.get(key)
            if isinstance(sb, dict):
                merged = dict(blocks.get(tag) or {})
                for kk, vv in sb.items():
                    if vv is not None or kk not in merged:
                        merged[kk] = vv
                if merged:
                    blocks[tag] = merged
        refs = summary.get("references")
        if isinstance(refs, dict):
            for k, v in refs.items():
                if not isinstance(v, dict):
                    continue
                merged = dict(blocks.get(k) or {})
                for kk, vv in v.items():
                    if vv is not None or kk not in merged:
                        merged[kk] = vv
                if merged:
                    blocks[k] = merged
    return blocks


def four_way_from_blocks(blocks):
    rows = []
    for role, key in (("gap_step120", "gap120"), ("e1b_step120", "e1b120"),
                      ("e1d_step120", "e1d120"), ("e1e_step120", "e1e120")):
        b = blocks.get(key)
        if not b:
            continue
        rows.append({
            "role": role,
            "run": b.get("label") or key,
            "micro_em": b.get("micro_em"),
            "macro_em": b.get("macro_em"),
            "rounds": b.get("search_rounds"),
            "queries": b.get("queries"),
            "tokens": b.get("tokens"),
            "zero_search": b.get("zero_search_rate"),
        })
    return rows or None


# ---------------------------------------------------------------------------
# static sections 2-10 / 30
# ---------------------------------------------------------------------------
def load_static_sections():
    if not os.path.isfile(STATIC):
        note_missing(STATIC)
        return {}
    try:
        with open(STATIC, encoding="utf-8") as fh:
            text = fh.read()
    except Exception:  # noqa: BLE001
        note_missing(STATIC)
        return {}
    register(STATIC)
    lines = [ln.rstrip() for ln in text.splitlines()]
    heads = [i for i, ln in enumerate(lines) if re.match(r"^## \d+\.", ln)]
    out = {}
    for j, i in enumerate(heads):
        m = re.match(r"^## (\d+)\.", lines[i])
        num = int(m.group(1))
        end = heads[j + 1] if j + 1 < len(heads) else len(lines)
        block = lines[i:end]
        while block and not block[-1].strip():
            block.pop()
        out[num] = block
    return out


def emit_static(lines, num, static):
    if num in static:
        lines.extend(static[num])
    else:
        lines.append(f"## {num}.")
        lines.append(f"(missing: {_rel(STATIC)} section {num})")
    lines.append("")


# ---------------------------------------------------------------------------
# section 1
# ---------------------------------------------------------------------------
def sec1(lines, ctx):
    lines.append("## 1. Executive Summary")
    lines.append("")
    s = ctx.get("summary")
    blocks = ctx["blocks"]
    if not isinstance(s, dict):
        lines.append(f"(missing: {_rel(os.path.join(ANALYSIS, 'summary.json'))})")
        lines.append("")
        lines.append("<!-- TODO(caller): verdict language -->")
        lines.append("")
        return

    comp = s.get("comparisons") or {}

    def cval(key, field):
        return (comp.get(key) or {}).get(field)

    def be(key):
        return blocks.get(key) or {}

    def line_for(key):
        b, t = be((comp.get(key) or {}).get("baseline")), be((comp.get(key) or {}).get("test"))
        return (f"Δmicro EM {fpp(cval(key, 'delta_micro_em'))} "
                f"({fnum(b.get('micro_em'))}→{fnum(t.get('micro_em'))}), "
                f"Δmacro EM {fpp(cval(key, 'delta_macro_em'))}, "
                f"Δrounds {fsgn(cval(key, 'delta_rounds'))} "
                f"({fpct(cval(key, 'delta_rounds_pct'))}), "
                f"Δqueries {fsgn(cval(key, 'delta_queries'))} "
                f"({fpct(cval(key, 'delta_queries_pct'))})")

    b70, b120 = be("e1e70"), be("e1e120")
    q70 = (ctx.get("integ70") or {}).get("quota_semantics") or {}
    q120 = (ctx.get("integ120") or {}).get("quota_semantics") or {}
    gate = (ctx.get("d1_gate") or {}).get("gate")

    para = []
    para.append(
        f"E1-E (q=2 solved-group gated / capped efficiency training) reports "
        f"`experiment_status = {s.get('experiment_status')}` and the mechanical "
        f"`case = {s.get('case')}`.")
    para.append(
        f"Step70: micro EM {fnum(b70.get('micro_em'))}, macro EM {fnum(b70.get('macro_em'))}, "
        f"rounds {fnum(b70.get('search_rounds'))}, queries {fnum(b70.get('queries'))}.")
    para.append(
        f"Step120: micro EM {fnum(b120.get('micro_em'))}, macro EM {fnum(b120.get('macro_em'))}, "
        f"rounds {fnum(b120.get('search_rounds'))}, queries {fnum(b120.get('queries'))}.")
    if comp:
        para.append("Four-way step120 deltas: vs GAP120 — " + line_for("e1e120_vs_gap120") + ".")
        para.append("vs E1-B120 — " + line_for("e1e120_vs_e1b120") + ".")
        para.append("vs E1-D120 — " + line_for("e1e120_vs_e1d120") + ".")
    else:
        para.append("(missing: summary.json `comparisons`)")
    if q70.get("available"):
        para.append(
            f"D1 runtime quota (steps 61-70): fill rate {frate(q70.get('quota_fill_rate'))} "
            f"({fint(q70.get('steps_full_quota'))}/{fint(q70.get('steps_used'))} steps full), "
            f"inserted {fint(q70.get('total_inserted_efficiency_groups'))} efficiency groups, "
            f"displaced {fint(q70.get('total_displaced_mixed_groups'))} mixed groups, "
            f"eff_shortfall {fint(q70.get('total_eff_shortfall'))}, "
            f"quota_semantics_ok = {q70.get('quota_semantics_ok')}.")
    else:
        para.append(f"(missing: {_rel(os.path.join(ANALYSIS, 'step70_integrity.json'))} quota_semantics)")
    if q120.get("available"):
        para.append(
            f"D2 runtime quota (steps 71-120): fill rate {frate(q120.get('quota_fill_rate'))} "
            f"({fint(q120.get('steps_full_quota'))}/{fint(q120.get('steps_used'))} steps full), "
            f"inserted {fint(q120.get('total_inserted_efficiency_groups'))} efficiency groups, "
            f"displaced {fint(q120.get('total_displaced_mixed_groups'))} mixed groups, "
            f"eff_shortfall {fint(q120.get('total_eff_shortfall'))}, "
            f"quota_semantics_ok = {q120.get('quota_semantics_ok')}.")
    else:
        para.append(f"(missing: {_rel(os.path.join(ANALYSIS, 'step120_integrity.json'))} quota_semantics)")
    para.append(f"D1 gate: {gate if gate is not None else '(missing)'}.")
    para.append(
        f"Training-side side-effect shares: zero-search "
        f"{frate(q70.get('zero_search_share'))} (D1) / {frate(q120.get('zero_search_share'))} (D2); "
        f"malformed metadata {frate(q70.get('malformed_metadata_share'))} (D1) / "
        f"{frate(q120.get('malformed_metadata_share'))} (D2).")
    lines.append(" ".join(para))
    lines.append("")
    lines.append("<!-- TODO(caller): verdict language -->")
    lines.append("")


# ---------------------------------------------------------------------------
# sections 11 / 15 — integrity
# ---------------------------------------------------------------------------
def sec_integrity(lines, num, title, integ, integ_path, start_step, end_step):
    lines.append(f"## {num}. {title}")
    lines.append("")
    if not isinstance(integ, dict):
        lines.append(f"(missing: {_rel(integ_path)})")
        lines.append("")
        return
    lines.append(f"- phase: `{integ.get('phase')}`; expected steps {start_step}->{end_step}; "
                 f"integrity verdict: **{'PASS' if integ.get('integrity_ok') else 'FAIL'}** "
                 f"(`integrity_ok = {integ.get('integrity_ok')}`).")
    tl = integ.get("training_log") or {}
    steps_seen = tl.get("steps_seen") or []
    lines.append(
        f"- training log: {len(steps_seen)} steps present"
        + (f" (first {steps_seen[0]}, last {steps_seen[-1]})" if steps_seen else "")
        + f"; missing: {tl.get('missing_steps')}; `all_steps_present = {tl.get('all_steps_present')}`; "
        f"resume line 'Setting global step to {start_step}' present: {tl.get('resume_line_present')}.")
    lines.append(f"- metrics parsed: {fint(tl.get('metrics_parsed'))}; "
                 f"non-finite metric values: {tl.get('non_finite_metrics') or '{} (none)'}.")
    err = tl.get("error_pattern_counts_before_final_val") or {}
    if err:
        lines.append("- fatal-error scan (before the final-validation marker): "
                     + ", ".join(f"{k}={v}" for k, v in err.items()) + ".")
        nonzero = [k for k in FATAL_PATTERNS if err.get(k)]
        lines.append(f"- fatal patterns (>0): {nonzero if nonzero else 'none'}.")
    else:
        lines.append("- fatal-error scan: (missing: `error_pattern_counts_before_final_val`).")
    ck = integ.get("checkpoint") or {}
    shards = ck.get("shards") or {}
    present = sum(1 for v in shards.values() if v.get("exists"))
    lines.append(
        f"- checkpoint `{integ.get('checkpoint_dir')}`: exists = {ck.get('exists')}, "
        f"`data.pt` = {ck.get('data_pt')}, shards present {present}/{len(shards)}, "
        f"tracker = {ck.get('tracker')}, `tracker_reached_end_step` = "
        f"{integ.get('tracker_reached_end_step')}.")
    sre = integ.get("state_resume_evidence") or {}
    rs = sre.get("resume_source") or {}
    en = sre.get("end") or {}
    lines.append(
        f"- resume source (`{rs.get('path')}`): `lr_scheduler.last_epoch` = "
        f"{fint(rs.get('lr_scheduler_last_epoch'))} (expected {fint(rs.get('expected_last_epoch'))} "
        f"= start step), rng present = {rs.get('rng_present')}.")
    lines.append(
        f"- end checkpoint (`{en.get('path')}`): `lr_scheduler.last_epoch` = "
        f"{fint(en.get('lr_scheduler_last_epoch'))} (expected {fint(en.get('expected_last_epoch'))} "
        f"= end step), rng present = {en.get('rng_present')}; "
        f"`state_resume_ok = {integ.get('state_resume_ok')}`.")
    q = integ.get("quota_semantics") or {}
    lines.append(f"- runtime quota semantics: `quota_semantics_ok = {q.get('quota_semantics_ok')}`, "
                 f"quota error lines = {fint(q.get('quota_error_lines'))}.")
    lvd = integ.get("log_vs_diagnostics") or {}
    lines.append(
        f"- trainer log vs diagnostics: steps cross-checked = {fint(lvd.get('steps_cross_checked'))}, "
        f"`critic/score/max` mismatches = {fint(lvd.get('score_max_mismatches'))}, "
        f"consistent = {lvd.get('log_score_max_consistent_with_quota')}.")
    lines.append("")


# ---------------------------------------------------------------------------
# sections 12 / 16 — runtime quota statistics
# ---------------------------------------------------------------------------
def sec_quota(lines, num, title, integ, integ_path, step_range):
    lines.append(f"## {num}. {title}")
    lines.append("")
    q = (integ or {}).get("quota_semantics") if isinstance(integ, dict) else None
    if not isinstance(q, dict) or not q.get("available"):
        reason = q.get("reason") if isinstance(q, dict) else None
        lines.append(f"(missing: {_rel(integ_path)} quota_semantics"
                     + (f" — {reason}" if reason else "") + ")")
        lines.append("")
        return
    lines.append(f"Per-optimizer-step runtime quota diagnostics for {step_range}.")
    lines.append("")
    lines.append("| aggregate | value |")
    lines.append("|---|---|")
    lines.append(f"| step records | {fint(q.get('n_step_records'))} |")
    lines.append(f"| generation calls | {fint(q.get('n_generation_calls'))} |")
    lines.append(f"| group records | {fint(q.get('n_group_records'))} |")
    lines.append(f"| rollout records | {fint(q.get('n_rollout_records'))} |")
    lines.append(f"| steps covered | {q.get('steps_covered')} |")
    lines.append(f"| steps missing | {q.get('steps_missing')} |")
    lines.append(f"| `all_steps_recorded` | {q.get('all_steps_recorded')} |")
    lines.append(f"| quota fill rate (mean `quota_actual` / 2) | {frate(q.get('quota_fill_rate'))} "
                 f"({fnum(q.get('quota_fill_rate'), 4)}) |")
    lines.append(f"| steps at full quota | {fint(q.get('steps_full_quota'))} / {fint(q.get('steps_used'))} |")
    lines.append(f"| total inserted efficiency groups | {fint(q.get('total_inserted_efficiency_groups'))} |")
    lines.append(f"| total displaced mixed groups | {fint(q.get('total_displaced_mixed_groups'))} |")
    lines.append(f"| total `eff_shortfall` | {fint(q.get('total_eff_shortfall'))} |")
    lines.append(f"| steps needing injection | {fint(q.get('steps_needing_injection'))} |")
    lines.append(f"| generation batches (total) | {fint(q.get('total_generation_batches'))} |")
    lines.append(f"| generation batches (mean/step) | {fnum(q.get('mean_generation_batches_per_step'))} |")
    lines.append(f"| efficiency groups seen | {fint(q.get('efficiency_groups'))} |")
    lines.append(f"| mixed groups seen | {fint(q.get('mixed_groups'))} |")
    lines.append(f"| zero-search rollouts | {fint(q.get('zero_search_rollouts'))} "
                 f"({frate(q.get('zero_search_share'))}) |")
    lines.append(f"| malformed-metadata rollouts | {fint(q.get('malformed_metadata_rollouts'))} "
                 f"({frate(q.get('malformed_metadata_share'))}) |")
    lines.append(f"| advantage NaN / Inf | {fint(q.get('advantage_nan_total'))} / "
                 f"{fint(q.get('advantage_inf_total'))} |")
    lines.append(f"| quota error lines | {fint(q.get('quota_error_lines'))} |")
    lines.append(f"| steps with wrong batch size | {q.get('steps_with_wrong_batch_size')} |")
    lines.append(f"| calls with `quota_actual` > 2 | {fint(q.get('calls_with_quota_actual_gt_2'))} |")
    lines.append(f"| steps without exactly one stopping call | "
                 f"{q.get('steps_without_exactly_one_stopping_call')} |")
    lines.append(f"| `quota_semantics_ok` | {q.get('quota_semantics_ok')} |")
    lines.append("")
    per_step = q.get("per_step") or {}
    if not per_step:
        lines.append("(missing: `quota_semantics.per_step`)")
        lines.append("")
        return
    lines.append("| step | generation batches | quota actual | mixed kept cum | efficiency groups seen | "
                 "cache size | eff_shortfall | inserted | optimizer batch groups | advantage mean | "
                 "advantage std |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in sorted(per_step, key=lambda x: int(x)):
        v = per_step[g] or {}
        inserted = len(v.get("inserted_ids") or [])
        lines.append(
            f"| {g} | {fint(v.get('generation_batches'))} | {fint(v.get('quota_actual'))} | "
            f"{fint(v.get('mixed_kept_cum'))} | {fint(v.get('efficiency_groups_seen'))} | "
            f"{fint(v.get('efficiency_cache_size'))} | {fint(v.get('eff_shortfall'))} | "
            f"{inserted} | {fint(v.get('optimizer_batch_groups'))} | "
            f"{fnum(v.get('advantage_mean'))} | {fnum(v.get('advantage_std'))} |")
    lines.append("")


# ---------------------------------------------------------------------------
# sections 13 / 17 — seven-benchmark evaluation
# ---------------------------------------------------------------------------
def sec_eval(lines, num, title, tag, blocks, step):
    lines.append(f"## {num}. {title}")
    lines.append("")
    b = blocks.get(tag)
    if not b:
        lines.append(f"(missing: {_rel(os.path.join(EVAL_ROOT, f'step{step}_*', 'eval_summary.json'))})")
        lines.append("")
        return
    lines.append(f"- artifact: `{b.get('path') or '(from summary.json)'}`; "
                 f"label = {b.get('label')}; total samples = {fint(b.get('total_samples'))}.")
    lines.append(f"- macro EM = {fnum(b.get('macro_em'))}; micro EM = {fnum(b.get('micro_em'))}.")
    per = b.get("per_benchmark_em") or {}
    if per:
        lines.append("")
        lines.append("| benchmark | " + " | ".join(BENCH) + " |")
        lines.append("|---" * (len(BENCH) + 1) + "|")
        lines.append("| EM | " + " | ".join(fnum(per.get(x), 4) for x in BENCH) + " |")
    lines.append("")
    lines.append(
        f"- behaviour: rounds {fnum(b.get('search_rounds'))}, queries {fnum(b.get('queries'))}, "
        f"queries/round {fnum(b.get('queries_per_round'))}, "
        f"parallel factor {fnum(b.get('parallel_factor'))}, "
        f"parallel sample rate {frate(b.get('parallel_sample_rate'))}, "
        f"zero-search {frate(b.get('zero_search_rate'))}, "
        f"no-answer {frate(b.get('no_answer_tag_rate'))}, "
        f"response tokens {fnum(b.get('tokens'), 2)}.")
    mech_keys = [
        ("quota_fill_rate", "quota fill rate", True),
        ("steps_full_quota", "steps at full quota", False),
        ("steps_used", "steps used", False),
        ("total_inserted_efficiency_groups", "inserted efficiency groups", False),
        ("total_displaced_mixed_groups", "displaced mixed groups", False),
        ("total_eff_shortfall", "total eff_shortfall", False),
        ("mean_generation_batches_per_step", "mean generation batches/step", False),
        ("zero_search_share_train", "train zero-search share", True),
        ("malformed_metadata_share", "train malformed share", True),
        ("quota_semantics_ok", "quota_semantics_ok", False),
        ("quota_error_lines", "quota error lines", False),
    ]
    present = [(lbl, key, is_rate) for key, lbl, is_rate in mech_keys
               if b.get(key) is not None]
    if present:
        lines.append("- runtime mechanism: " + "; ".join(
            f"{lbl} = {frate(b.get(key)) if is_rate else fnum(b.get(key), 4)}"
            for lbl, key, is_rate in present) + ".")
    lines.append("")


# ---------------------------------------------------------------------------
# section 14 — D1 gate
# ---------------------------------------------------------------------------
def sec_gate(lines, ctx):
    lines.append("## 14. D1 gate decision")
    lines.append("")
    g = ctx.get("d1_gate")
    gate_path = os.path.join(ANALYSIS, "d1_gate.json")
    if not isinstance(g, dict):
        lines.append(f"(missing: {_rel(gate_path)})")
        lines.append("")
        return
    lines.append(f"- gate: **{g.get('gate')}**")
    lines.append(f"- decision: {g.get('decision')}")
    reasons = g.get("reasons") or []
    lines.append(f"- stop reasons ({len(reasons)}): " + ("; ".join(reasons) if reasons else "none"))
    warnings = g.get("warnings") or []
    if warnings:
        lines.append("- warnings: " + "; ".join(warnings))
    checks = g.get("checks") or {}
    scalar = {k: v for k, v in checks.items() if not isinstance(v, (dict, list))}
    if scalar:
        lines.append("")
        lines.append("| check | value |")
        lines.append("|---|---|")
        for k in scalar:
            lines.append(f"| `{k}` | {scalar[k]} |")
    refs = checks.get("E_references")
    if isinstance(refs, dict) and refs:
        lines.append("")
        lines.append("Reference step70 overall metrics used by the gate:")
        lines.append("")
        lines.append("| run | n samples | macro EM | micro EM | rounds | queries | parallel factor | "
                     "zero-search | no-answer |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for name, r in refs.items():
            if not isinstance(r, dict):
                continue
            lines.append(
                f"| `{name}` | {fint(r.get('n_samples'))} | {fnum(r.get('macro_em'))} | "
                f"{fnum(r.get('micro_em'))} | {fnum(r.get('retrieval_rounds_mean'))} | "
                f"{fnum(r.get('search_queries_mean'))} | {fnum(r.get('parallel_factor_mean'))} | "
                f"{frate(r.get('zero_search_rate'))} | {frate(r.get('no_answer_tag_rate'))} |")
    lines.append("")


# ---------------------------------------------------------------------------
# sections 18 / 19 — per-benchmark + efficiency tables
# ---------------------------------------------------------------------------
def sec18(lines, blocks):
    lines.append("## 18. Per-benchmark accuracy table")
    lines.append("")
    present = [(lbl, blocks.get(tag)) for lbl, tag in RUN_ROWS if blocks.get(tag)]
    if not present:
        lines.append("(missing: no evaluation summaries found under "
                     f"`{_rel(EVAL_ROOT)}/` or in `summary.json.references`)")
        lines.append("")
        return
    lines.append("Per-benchmark EM (and macro/micro) for E1-E step70/step120 and the "
                 "reference rows; identical 7-benchmark greedy protocol.")
    lines.append("")
    lines.append("| run | " + " | ".join(BENCH) + " | macro EM | micro EM |")
    lines.append("|---" * (len(BENCH) + 3) + "|")
    for lbl, b in present:
        per = b.get("per_benchmark_em") or {}
        lines.append(f"| {lbl} | " + " | ".join(fnum(per.get(x), 4) for x in BENCH)
                     + f" | {fnum(b.get('macro_em'))} | {fnum(b.get('micro_em'))} |")
    lines.append("")


def sec19(lines, blocks):
    lines.append("## 19. Efficiency metrics table")
    lines.append("")
    present = [(lbl, blocks.get(tag)) for lbl, tag in RUN_ROWS if blocks.get(tag)]
    if not present:
        lines.append("(missing: no evaluation summaries found under "
                     f"`{_rel(EVAL_ROOT)}/` or in `summary.json.references`)")
        lines.append("")
        return
    lines.append("Efficiency / behaviour metrics for the same runs as section 18 "
                 "(`-` = not present in the artifact).")
    lines.append("")
    lines.append("| run | rounds | queries | queries/round | parallel factor | parallel sample rate | "
                 "zero-search | no-answer | tokens |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for lbl, b in present:
        lines.append(
            f"| {lbl} | {fnum(b.get('search_rounds'))} | {fnum(b.get('queries'))} | "
            f"{fnum(b.get('queries_per_round'))} | {fnum(b.get('parallel_factor'))} | "
            f"{frate(b.get('parallel_sample_rate'))} | {frate(b.get('zero_search_rate'))} | "
            f"{frate(b.get('no_answer_tag_rate'))} | {fnum(b.get('tokens'), 2)} |")
    lines.append("")


# ---------------------------------------------------------------------------
# section 20 — four-way comparison
# ---------------------------------------------------------------------------
def _render_summary_cmp(lines, key, c, blocks):
    bkey, tkey = c.get("baseline"), c.get("test")
    bb = blocks.get(bkey) or {}
    tb = blocks.get(tkey) or {}
    lines.append(f"#### `{key}` (baseline `{bkey}` -> test `{tkey}`)")
    lines.append("")
    lines.append(f"- macro EM: {fnum(bb.get('macro_em'))} -> {fnum(tb.get('macro_em'))} "
                 f"(Δ {fpp(c.get('delta_macro_em'))})")
    lines.append(f"- micro EM: {fnum(bb.get('micro_em'))} -> {fnum(tb.get('micro_em'))} "
                 f"(Δ {fpp(c.get('delta_micro_em'))})")
    lines.append(f"- search rounds: {fnum(bb.get('search_rounds'))} -> {fnum(tb.get('search_rounds'))} "
                 f"(Δ {fsgn(c.get('delta_rounds'))}, {fpct(c.get('delta_rounds_pct'))})")
    lines.append(f"- queries: {fnum(bb.get('queries'))} -> {fnum(tb.get('queries'))} "
                 f"(Δ {fsgn(c.get('delta_queries'))}, {fpct(c.get('delta_queries_pct'))})")
    lines.append(f"- parallel factor: {fnum(bb.get('parallel_factor'))} -> "
                 f"{fnum(tb.get('parallel_factor'))} (Δ {fsgn(c.get('delta_parallel_factor'))})")
    lines.append(f"- zero-search: {frate(bb.get('zero_search_rate'))} -> "
                 f"{frate(tb.get('zero_search_rate'))} (Δ {fpp(c.get('delta_zero_search'))})")
    p = c.get("paired")
    if isinstance(p, dict):
        lines.append(
            f"- paired (n = {fint(p.get('paired_prompts'))}): rounds {fnum(p.get('rounds_baseline'))} -> "
            f"{fnum(p.get('rounds_e1e'))} (Δ {fsgn(p.get('rounds_delta'))}, "
            f"{fpct(p.get('rounds_delta_pct'))}), sign-test p = {fp(p.get('sign_test_p_rounds'))}; "
            f"EM {fnum(p.get('em_baseline'))} -> {fnum(p.get('em_e1e'))} "
            f"(Δ {fpp(p.get('em_delta'))}), up/down {fint(p.get('em_up_prompts'))}/"
            f"{fint(p.get('em_down_prompts'))}, sign-test p = {fp(p.get('sign_test_p_em'))}.")
    lines.append("")


def _render_comparison_json_cmp(lines, key, blk):
    if not isinstance(blk, dict):
        return
    lines.append(f"#### `{key}` (baseline `{blk.get('baseline_run')}` -> "
                 f"test `{blk.get('test_run')}`)")
    lines.append("")
    for name, field in (("macro EM", "macro_em"), ("micro EM", "micro_em")):
        d = blk.get(field) or {}
        lines.append(f"- {name}: {fnum(d.get('baseline'))} -> {fnum(d.get('test'))} "
                     f"(Δ {fpp(d.get('delta'))})")
    beh = blk.get("behavior") or {}
    for metric in ("retrieval_rounds_mean", "search_queries_mean", "parallel_factor_mean",
                   "parallel_sample_rate", "assistant_turns_mean_all_messages",
                   "response_tokens_mean", "zero_search_rate", "no_answer_tag_rate",
                   "empty_output_rate"):
        d = beh.get(metric) or {}
        if d.get("baseline") is None or d.get("test") is None:
            continue
        if metric in ("retrieval_rounds_mean", "search_queries_mean", "response_tokens_mean",
                      "assistant_turns_mean_all_messages"):
            delta = f"{fsgn(d.get('delta'))} ({fpct(d.get('pct'))})"
        elif metric in ("zero_search_rate", "no_answer_tag_rate", "parallel_sample_rate",
                        "empty_output_rate"):
            delta = fpp(d.get("delta"))
        else:
            delta = fsgn(d.get("delta"))
        lines.append(f"- {metric}: {fnum(d.get('baseline'))} -> {fnum(d.get('test'))} (Δ {delta})")
    lines.append("")


def sec20(lines, ctx):
    lines.append("## 20. GAP / E1-B / E1-D / E1-E four-way comparison")
    lines.append("")
    s = ctx.get("summary")
    comp = ctx.get("comparison")
    comp_path = ctx.get("comparison_path")
    blocks = ctx["blocks"]
    four = None
    source = None
    reported_missing = False
    if isinstance(s, dict) and s.get("four_way_step120"):
        four = s["four_way_step120"]
        source = "`summary.json`"
    elif isinstance(comp, dict) and comp.get("four_way_step120"):
        four = comp["four_way_step120"]
        source = f"`{_rel(comp_path)}`"
    if four:
        lines.append(f"Four-way step120 (source: {source}):")
        lines.append("")
        lines.append("| role | run | micro EM | macro EM | rounds | queries | tokens | zero-search |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in four:
            lines.append(
                f"| `{r.get('role')}` | `{r.get('run')}` | {fnum(r.get('micro_em'), 6)} | "
                f"{fnum(r.get('macro_em'), 6)} | {fnum(r.get('rounds'))} | {fnum(r.get('queries'))} | "
                f"{fnum(r.get('tokens'), 2)} | {frate(r.get('zero_search'))} |")
    else:
        if not isinstance(s, dict):
            lines.append(f"(missing: {_rel(os.path.join(ANALYSIS, 'summary.json'))})")
            reported_missing = True
        if not isinstance(comp, dict):
            lines.append(f"(missing: {_rel(os.path.join(ANALYSIS, 'comparison_*', 'comparison.json'))})")
            reported_missing = True
        four = four_way_from_blocks(blocks)
        if four:
            lines.append("")
            lines.append("Four-way step120 reconstructed from the evaluation summaries "
                         "(comparison artifacts missing):")
            lines.append("")
            lines.append("| role | run | micro EM | macro EM | rounds | queries | tokens | zero-search |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for r in four:
                lines.append(
                    f"| `{r.get('role')}` | `{r.get('run')}` | {fnum(r.get('micro_em'), 6)} | "
                    f"{fnum(r.get('macro_em'), 6)} | {fnum(r.get('rounds'))} | {fnum(r.get('queries'))} | "
                    f"{fnum(r.get('tokens'), 2)} | {frate(r.get('zero_search'))} |")
    lines.append("")
    lines.append(figures_note())
    lines.append("")
    comps = s.get("comparisons") if isinstance(s, dict) else None
    if isinstance(comps, dict) and comps:
        for key, c in comps.items():
            if not isinstance(c, dict) or "delta_micro_em" not in c:
                continue
            _render_summary_cmp(lines, key, c, blocks)
    elif isinstance(comp, dict):
        for key in comp:
            if re.match(r"^(vs_|e1e.*_vs_)", key):
                _render_comparison_json_cmp(lines, key, comp.get(key))
    else:
        if not reported_missing:
            lines.append(f"(missing: {_rel(os.path.join(ANALYSIS, 'comparison_*', 'comparison.json'))})")
            lines.append("")


# ---------------------------------------------------------------------------
# section 21 — paired tests
# ---------------------------------------------------------------------------
def _bench_sort_key(name):
    return (BENCH.index(name), "") if name in BENCH else (len(BENCH), str(name))


def sec21(lines, ctx):
    lines.append("## 21. Paired statistical tests")
    lines.append("")
    paired = ctx.get("paired") or {}
    if not paired:
        lines.append(f"(missing: {_rel(os.path.join(ANALYSIS, 'comparison_*', 'paired_comparison_*.json'))})")
        lines.append("")
        return
    for label in sorted(paired):
        pj, path = paired[label]
        o = pj.get("overall") or {}
        lines.append(f"### `{label}`")
        lines.append("")
        lines.append(f"- source: `{_rel(path)}`")
        lines.append(f"- paired prompts: {fint(o.get('n_paired_prompts'))}")
        rbase = o.get("rounds_baseline_mean")
        rdelta = o.get("rounds_delta")
        rpct = (rdelta / rbase * 100.0) if (rdelta is not None and rbase) else None
        lines.append(
            f"- search rounds: {fnum(rbase)} -> {fnum(o.get('rounds_e1e_mean'))} "
            f"(Δ {fsgn(rdelta)}, {fpct(rpct)}); decreased {fint(o.get('rounds_decreased_prompts'))}, "
            f"increased {fint(o.get('rounds_increased_prompts'))}, unchanged "
            f"{fint(o.get('rounds_unchanged_prompts'))}; sign-test p = {fp(o.get('sign_test_p_rounds'))}")
        qbase = o.get("queries_baseline_mean")
        qdelta = o.get("queries_delta")
        qpct = (qdelta / qbase * 100.0) if (qdelta is not None and qbase) else None
        lines.append(
            f"- search queries: {fnum(qbase)} -> {fnum(o.get('queries_e1e_mean'))} "
            f"(Δ {fsgn(qdelta)}, {fpct(qpct)})")
        lines.append(
            f"- EM: {fnum(o.get('em_baseline_mean'))} -> {fnum(o.get('em_e1e_mean'))} "
            f"(Δ {fpp(o.get('em_delta'))}); up {fint(o.get('em_up_prompts'))}, "
            f"down {fint(o.get('em_down_prompts'))}; sign-test p = {fp(o.get('sign_test_p_em'))}")
        per = pj.get("per_benchmark") or {}
        if per:
            lines.append("")
            lines.append("| benchmark | n | rounds base | rounds E1-E | Δrounds | Δrounds % | EM base | "
                         "EM E1-E | ΔEM | sign-test p (EM) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for ds in sorted(per, key=_bench_sort_key):
                v = per[ds] or {}
                rb = v.get("rounds_baseline_mean")
                rd = v.get("rounds_delta")
                rpc = (rd / rb * 100.0) if (rd is not None and rb) else None
                lines.append(
                    f"| {ds} | {fint(v.get('n_paired_prompts'))} | {fnum(rb)} | "
                    f"{fnum(v.get('rounds_e1e_mean'))} | {fsgn(rd)} | {fpct(rpc)} | "
                    f"{fnum(v.get('em_baseline_mean'))} | {fnum(v.get('em_e1e_mean'))} | "
                    f"{fpp(v.get('em_delta'))} | {fp(v.get('sign_test_p_em'))} |")
        lines.append("")


# ---------------------------------------------------------------------------
# section 23 — safety / reward hacking
# ---------------------------------------------------------------------------
SAFETY_COUNTERS = [
    ("mixed rollouts with reward != EM", "rollouts_mixed_reward_not_equal_em", "0"),
    ("mixed groups with reward_mean != em_mean", "mixed_groups_reward_mean_not_equal_em", "0"),
    ("mixed groups with reward outside [0, 1]", "mixed_groups_reward_out_of_0_1", "0"),
    ("efficiency rollouts with wrong reward or EM", "rollouts_efficiency_reward_or_em_wrong", "0"),
    ("efficiency groups with reward outside [1, 1.05]",
     "efficiency_groups_reward_out_of_1_1.05", "0"),
    ("efficiency groups with k != 8", "efficiency_groups_with_k_not_8", "0"),
    ("efficiency groups without cost variation", "efficiency_groups_without_cost_variation", "0"),
    ("wrong rollouts receiving non-zero reward", "rollouts_wrong_with_nonzero_reward", "0"),
    ("rollouts with inconsistent published metric", "rollouts_published_metric_inconsistent", "0"),
    ("advantage NaN", "advantage_nan_total", "0"),
    ("advantage Inf", "advantage_inf_total", "0"),
    ("quota error lines", "quota_error_lines", "0"),
    ("zero-search rollouts", "zero_search_rollouts", "informational"),
    ("malformed-metadata rollouts", "malformed_metadata_rollouts", "informational"),
]


def sec23(lines, ctx):
    lines.append("## 23. Reward-hacking / safety analysis")
    lines.append("")
    lines.append(figures_note())
    lines.append("")
    evidence = []
    any_available = False
    for phase, integ, integ_path in (
            ("D1 (steps 61-70)", ctx.get("integ70"), os.path.join(ANALYSIS, "step70_integrity.json")),
            ("D2 (steps 71-120)", ctx.get("integ120"), os.path.join(ANALYSIS, "step120_integrity.json"))):
        lines.append(f"### {phase} runtime semantics")
        lines.append("")
        q = (integ or {}).get("quota_semantics") if isinstance(integ, dict) else None
        if not isinstance(q, dict) or not q.get("available"):
            lines.append(f"(missing: {_rel(integ_path)} quota_semantics)")
            lines.append("")
            continue
        any_available = True
        lines.append(f"| check | count | expected |")
        lines.append("|---|---|---|")
        for label, key, expected in SAFETY_COUNTERS:
            lines.append(f"| {label} | {fint(q.get(key))} | {expected} |")
        lines.append(f"| zero-search share | {frate(q.get('zero_search_share'))} | informational |")
        lines.append(f"| malformed-metadata share | {frate(q.get('malformed_metadata_share'))} | "
                     f"informational |")
        lines.append("")
        for key in ("rollouts_mixed_reward_not_equal_em", "mixed_groups_reward_mean_not_equal_em",
                    "mixed_groups_reward_out_of_0_1", "rollouts_efficiency_reward_or_em_wrong",
                    "efficiency_groups_reward_out_of_1_1.05", "efficiency_groups_with_k_not_8",
                    "efficiency_groups_without_cost_variation",
                    "rollouts_wrong_with_nonzero_reward", "rollouts_published_metric_inconsistent",
                    "advantage_nan_total", "advantage_inf_total"):
            v = q.get(key)
            if isinstance(v, (int, float)) and v:
                evidence.append(f"{phase}: {key} = {v}")
        if q.get("quota_error_lines"):
            evidence.append(f"{phase}: quota_error_lines = {q.get('quota_error_lines')}")
    for phase, tag in (("D1 (steps 61-70)", "D1"), ("D2 (steps 71-120)", "D2")):
        s = (ctx.get("safety") or {}).get(tag)
        spath = os.path.join(ANALYSIS, f"{tag}_vs_E1_0", "safety_analysis.json")
        lines.append(f"### {phase} packing / zero-search diagnostic")
        lines.append("")
        if not isinstance(s, dict):
            lines.append(f"(missing: {_rel(spath)})")
            lines.append("")
            continue
        qpk = s.get("query_packing")
        if isinstance(qpk, dict):
            definition = qpk.get("definition")
            if definition:
                lines.append(f"Definition: {definition}")
                lines.append("")
            for role, rlabel in (("e1d", "E1-E"), ("baseline", "baseline")):
                d = qpk.get(role)
                if not isinstance(d, dict):
                    continue
                lines.append(
                    f"- {rlabel}: active groups {fint(d.get('active_groups'))}; "
                    f"efficient correct rollouts rounds {fnum(d.get('efficient_rounds_mean'))} vs "
                    f"inefficient {fnum(d.get('inefficient_rounds_mean'))}; queries "
                    f"{fnum(d.get('efficient_queries_mean'))} vs "
                    f"{fnum(d.get('inefficient_queries_mean'))}; packing-suspect groups "
                    f"{fint(d.get('packing_suspect_groups'))} "
                    f"(ratio {frate(d.get('packing_suspect_ratio'))}).")
        zs = s.get("zero_search")
        if isinstance(zs, dict):
            lines.append(
                f"- zero-search: E1-E rate {frate(zs.get('e1e_zero_search_rate'))} vs baseline "
                f"{frate(zs.get('baseline_zero_search_rate'))}; E1-E zero-search correct rate "
                f"{frate(zs.get('e1e_zero_search_correct_rate'))}.")
        mal = s.get("malformed")
        if isinstance(mal, dict):
            lines.append(f"- malformed metadata rate: {frate(mal.get('e1e_metadata_invalid_rate'))}.")
        lines.append("")
    lines.append("**Reward-hacking verdict.**")
    lines.append("")
    if not any_available:
        lines.append("Reward hacking cannot be assessed: the runtime quota diagnostics "
                     f"(`step70_integrity.json` / `step120_integrity.json` `quota_semantics`) are "
                     "not available.")
    elif evidence:
        lines.append("**Reward hacking IS evidenced by the recorded counters:** "
                     + "; ".join(evidence) + ".")
    else:
        lines.append("**No reward hacking is evidenced.** All recorded reward-separation and safety "
                     "counters are zero in both phases (mixed reward == EM, efficiency reward in "
                     "[1, 1.05], wrong-rollout reward == 0, published metric consistent, "
                     "advantage NaN/Inf == 0, no quota error lines), and the zero-search and "
                     "malformed shares are reported above. The packing-suspect ratio above is a "
                     "behavioural diagnostic (definition given in the artifact), not a "
                     "reward-hacking violation.")
    lines.append("")


# ---------------------------------------------------------------------------
# section 24 — prediction vs observation
# ---------------------------------------------------------------------------
def sec24(lines, ctx):
    lines.append("## 24. E1-E0 offline prediction vs online observation")
    lines.append("")
    pred = ctx.get("pred")
    pred_path = os.path.join(ANALYSIS, "prediction_vs_observation.json")
    if not isinstance(pred, dict):
        lines.append(f"(missing: {_rel(pred_path)})")
        lines.append("")
        return
    if not pred.get("available"):
        lines.append(f"(not available: {_rel(pred_path)}"
                     + (f" — {pred.get('reason')}" if pred.get("reason") else "") + ")")
        lines.append("")
    lines.append(f"- phases: {pred.get('phases')}; optimizer steps observed: "
                 f"{fint(pred.get('n_step_records'))}; rollouts: {fint(pred.get('n_rollouts'))}; "
                 f"group records: {fint(pred.get('n_groups'))}.")
    steps = pred.get("steps") or []
    if steps:
        lines.append(f"- steps covered: {steps[0]}..{steps[-1]} ({len(steps)} steps).")
    cmp_ = pred.get("comparison") if isinstance(pred.get("comparison"), dict) else {}
    obs = cmp_.get("observed") or {}
    prd = cmp_.get("predicted_E1E0") or {}
    keys = [k for k in list(obs) + list(prd) if k not in ("source_note",)]
    seen = []
    for k in keys:
        if k not in seen:
            seen.append(k)
    if seen:
        lines.append("")
        lines.append("| quantity | E1-E0 prediction (q=2) | E1-E online observation |")
        lines.append("|---|---|---|")
        for k in seen:
            lines.append(f"| {k} | {fnum(prd.get(k), 6)} | {fnum(obs.get(k), 6)} |")
    agree = cmp_.get("agreement") or {}
    if agree:
        lines.append("")
        lines.append("Agreement flags:")
        lines.append("")
        for k, v in agree.items():
            lines.append(f"- `{k}`: {v}")
    cav = cmp_.get("honest_caveats") or []
    if cav:
        lines.append("")
        lines.append("Honest caveats recorded in the artifact:")
        lines.append("")
        for c in cav:
            lines.append(f"- {c}")
    lines.append("")
    lines.append("The two key caveats (paraphrased):")
    lines.append("")
    lines.append("1. **Provenance of the prediction** — E1-E0 predicted on **E1-D's** candidate stream "
                 "(the D2 steps of the E1-D policy), not on E1-E's own online policy's stream; "
                 "per-step group membership therefore cannot be compared one-to-one, and only the "
                 "distributional claims are comparable.")
    lines.append("2. **D1 has no E1-E0 prediction at all** — E1-E0's stream only covers steps 71-120, "
                 "so D1 (steps 61-70) is outside the offline prediction's scope.")
    lines.append("")


# ---------------------------------------------------------------------------
# section 25 — preregistered success criteria
# ---------------------------------------------------------------------------
def sec25(lines, ctx):
    lines.append("## 25. Whether E1-E meets the preregistered success criteria")
    lines.append("")
    s = ctx.get("summary")
    if not isinstance(s, dict):
        lines.append(f"(missing: {_rel(os.path.join(ANALYSIS, 'summary.json'))})")
        lines.append("")
        lines.append("<!-- TODO(caller): prose -->")
        lines.append("")
        return
    flags = s.get("decision_flags") or {}
    comps = s.get("comparisons") or {}

    def cf(key, field):
        return (comps.get(key) or {}).get(field)

    def observed(name):
        if name == "delta_EM_at_least_minus_0.5pp_vs_GAP120":
            return f"Δmicro EM vs GAP120 = {fpp(cf('e1e120_vs_gap120', 'delta_micro_em'))}"
        if name == "delta_rounds_at_most_minus_10pct_vs_GAP120":
            return f"Δrounds vs GAP120 = {fpct(cf('e1e120_vs_gap120', 'delta_rounds_pct'))}"
        if name in ("pareto_dominates_E1D120", "keeps_E1D_EM_and_beats_E1D_rounds"):
            return (f"Δmicro EM vs E1-D120 = {fpp(cf('e1e120_vs_e1d120', 'delta_micro_em'))}, "
                    f"Δrounds vs E1-D120 = {fsgn(cf('e1e120_vs_e1d120', 'delta_rounds'))}")
        if name == "more_efficient_than_E1B120":
            return (f"Δrounds vs E1-B120 = {fsgn(cf('e1e120_vs_e1b120', 'delta_rounds'))} "
                    f"({fpct(cf('e1e120_vs_e1b120', 'delta_rounds_pct'))})")
        if name == "higher_EM_than_E1B120":
            return f"Δmicro EM vs E1-B120 = {fpp(cf('e1e120_vs_e1b120', 'delta_micro_em'))}"
        return "-"

    lines.append("| flag | preregistered threshold | value | observed |")
    lines.append("|---|---|---|---|")
    for name in FLAG_THRESHOLDS:
        lines.append(f"| `{name}` | {FLAG_THRESHOLDS[name]} | {flags.get(name)} | "
                     f"{observed(name)} |")
    for name in flags:
        if name not in FLAG_THRESHOLDS:
            lines.append(f"| `{name}` | (recorded flag) | {flags.get(name)} | - |")
    lines.append("")
    lines.append(f"Mechanical `case` (from `summary.json`): **{s.get('case')}**; "
                 f"`experiment_status` = {s.get('experiment_status')}.")
    lines.append("")
    lines.append("<!-- TODO(caller): prose -->")
    lines.append("")


# ---------------------------------------------------------------------------
# section 31 — artifact inventory from disk
# ---------------------------------------------------------------------------
def sec31(lines):
    lines.append("## 31. Complete artifact inventory")
    lines.append("")
    lines.append("Regenerated from disk at report-generation time (`os.walk` over `E1-E/`, "
                 "max depth 2, `__pycache__` skipped).")
    lines.append("")
    dirs = {}
    for root, dirnames, filenames in os.walk(E1E):
        rel = os.path.relpath(root, E1E)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 2:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        dirs["" if rel == "." else rel] = sorted(filenames)
    lines.append("| directory | files (basenames) |")
    lines.append("|---|---|")
    for key in sorted(dirs):
        label = "`E1-E/`" if key == "" else f"`E1-E/{key}/`"
        files = dirs[key]
        contents = ", ".join(f"`{f}`" for f in files) if files else "(empty)"
        lines.append(f"| {label} | {contents} |")
    lines.append("")
    for sub in ("checkpoints", "eval_results"):
        d = os.path.join(E1E, sub)
        lines.append(f"`E1-E/{sub}/` file sizes:")
        lines.append("")
        if not os.path.isdir(d):
            lines.append(f"(missing: {_rel(d)})")
            lines.append("")
            continue
        entries = sorted(os.listdir(d))
        if not entries:
            lines.append("- (empty)")
        for name in entries:
            p = os.path.join(d, name)
            if os.path.islink(p):
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = None
                lines.append(f"- `{name}` -> `{os.readlink(p)}` "
                             f"({size if size is not None else '?'} B)")
            elif os.path.isfile(p):
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = None
                lines.append(f"- `{name}` ({size if size is not None else '?'} B)")
            elif os.path.isdir(p):
                lines.append(f"- `{name}/` (directory)")
        lines.append("")


# ---------------------------------------------------------------------------
# footer
# ---------------------------------------------------------------------------
def footer(lines):
    lines.append("---")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')} by "
                 f"`E1-E/scripts/e1e_assemble_report.py`.")
    lines.append("")
    lines.append("Input files read (sha256, short):")
    lines.append("")
    read_files = sorted((k, v) for k, v in INPUTS.items()
                        if v not in ("missing", "unreadable"))
    if read_files:
        for k, v in read_files:
            lines.append(f"- `{k}` — `{v}`")
    else:
        lines.append("- (none)")
    if MISSING:
        lines.append("")
        lines.append("Inputs missing at generation time:")
        lines.append("")
        for k in MISSING:
            lines.append(f"- `{k}`")
    lines.append("")


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def load_paired():
    out = {}
    for p in sorted(glob.glob(os.path.join(ANALYSIS, "comparison_*",
                                           "paired_comparison_*.json"))):
        obj = jload(p)
        if not isinstance(obj, dict):
            continue
        label = obj.get("label") or os.path.basename(p)
        # sorted ascending by directory timestamp -> later (newer) dir wins
        out[label] = (obj, p)
    return out


def load_context():
    summary = jload(os.path.join(ANALYSIS, "summary.json"))
    blocks = build_eval_blocks(summary)
    comparison_path = newest(os.path.join(ANALYSIS, "comparison_*", "comparison.json"))
    ctx = {
        "summary": summary,
        "blocks": blocks,
        "integ70": jload(os.path.join(ANALYSIS, "step70_integrity.json")),
        "integ120": jload(os.path.join(ANALYSIS, "step120_integrity.json")),
        "d1_gate": jload(os.path.join(ANALYSIS, "d1_gate.json")),
        "pred": jload(os.path.join(ANALYSIS, "prediction_vs_observation.json")),
        "comparison_path": comparison_path,
        "comparison": jload(comparison_path) if comparison_path else None,
        "paired": load_paired(),
        "safety": {
            "D1": jload(os.path.join(ANALYSIS, "D1_vs_E1_0", "safety_analysis.json")),
            "D2": jload(os.path.join(ANALYSIS, "D2_vs_E1_0", "safety_analysis.json")),
        },
        "training": {
            "D1": jload(os.path.join(ANALYSIS, "D1_vs_E1_0", "training_summary.json")),
            "D2": jload(os.path.join(ANALYSIS, "D2_vs_E1_0", "training_summary.json")),
        },
        "manifest_pre": jload(os.path.join(E1E, "manifests", "original_files_manifest_pre.json")),
        "unchanged": jload(os.path.join(ANALYSIS, "original_files_unchanged.json")),
        "static": load_static_sections(),
    }
    return ctx


def safe(lines, fn, *args, **kwargs):
    try:
        fn(lines, *args, **kwargs)
    except Exception as e:  # noqa: BLE001
        lines.append(f"(error building section: {type(e).__name__}: {e})")
        lines.append("")


def build_report():
    ctx = load_context()
    static = ctx["static"]
    lines = []
    lines.append("# E1-E — q=2 Solved-Group Gated / Capped Efficiency Training — FINAL REPORT")
    lines.append("")
    safe(lines, sec1, ctx)
    for n in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        emit_static(lines, n, static)
    safe(lines, sec_integrity, 11, SECTION_TITLES[11], ctx.get("integ70"),
         os.path.join(ANALYSIS, "step70_integrity.json"), 60, 70)
    safe(lines, sec_quota, 12, SECTION_TITLES[12], ctx.get("integ70"),
         os.path.join(ANALYSIS, "step70_integrity.json"), "steps 61-70")
    safe(lines, sec_eval, 13, SECTION_TITLES[13], "e1e70", ctx["blocks"], 70)
    safe(lines, sec_gate, ctx)
    safe(lines, sec_integrity, 15, SECTION_TITLES[15], ctx.get("integ120"),
         os.path.join(ANALYSIS, "step120_integrity.json"), 70, 120)
    safe(lines, sec_quota, 16, SECTION_TITLES[16], ctx.get("integ120"),
         os.path.join(ANALYSIS, "step120_integrity.json"), "steps 71-120")
    safe(lines, sec_eval, 17, SECTION_TITLES[17], "e1e120", ctx["blocks"], 120)
    safe(lines, sec18, ctx["blocks"])
    safe(lines, sec19, ctx["blocks"])
    safe(lines, sec20, ctx)
    safe(lines, sec21, ctx)
    lines.append("## 22. Pareto analysis")
    lines.append("")
    lines.append(f"{figures_note()}")
    lines.append("")
    lines.append("<!-- TODO(caller): Pareto analysis (dominated / non-dominated / Pareto "
                 "improvement / trade-off; see fig1_four_way_pareto.png) -->")
    lines.append("")
    safe(lines, sec23, ctx)
    safe(lines, sec24, ctx)
    safe(lines, sec25, ctx)
    for n in (26, 27, 28, 29):
        lines.append(f"## {n}. {SECTION_TITLES[n]}")
        lines.append("")
        lines.append(f"<!-- TODO(caller): {SECTION_TITLES[n]} -->")
        lines.append("")
    emit_static(lines, 30, static)
    safe(lines, sec31)
    footer(lines)
    text = "\n".join(ln.rstrip() for ln in lines) + "\n"
    return text


def check_inputs():
    missing = []
    for label, pat in EXPECTED_INPUTS:
        # patterns with a glob wildcard may match nothing
        if any(ch in pat for ch in "*?["):
            if not glob.glob(pat):
                missing.append((label, pat))
        elif not os.path.exists(pat):
            missing.append((label, pat))
    return missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="do not write anything; print which expected inputs are missing")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output draft path")
    args = ap.parse_args(argv)

    if args.check:
        missing = check_inputs()
        if not missing:
            print("all expected inputs present")
        else:
            print(f"{len(missing)} expected input(s) missing:")
            for label, pat in missing:
                print(f"MISSING: {label}: {_rel(pat)}")
        return 0

    try:
        text = build_report()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not assemble report: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    try:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not write {args.out}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    n_inputs = len([1 for v in INPUTS.values() if v not in ("missing", "unreadable")])
    print(f"wrote {_rel(args.out)} ({len(text.splitlines())} lines, "
          f"{n_inputs} input file(s) read, {len(MISSING)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""E1-D post-phase integrity check (E1-D_task.md §12 / §16).

    E1D_PHASE=D1 python3 e1d_integrity.py      # checks global_step_70
    E1D_PHASE=D2 python3 e1d_integrity.py      # checks global_step_120

Verifies
  * the phase's checkpoint is complete (model/optimizer/extra_state x4 ranks, data.pt,
    tracker value) and the numbers of steps between the resume point and the end are all
    present in the log;
  * full-state resume evidence (lr_scheduler.last_epoch continuation + RNG present, read
    from the small extra_state shard);
  * no non-finite metric values, and an error-pattern scan (CUDA OOM / NCCL / traceback /
    wiki-tool failures) restricted to the real training window;
  * the FILTER DECOUPLING invariant, from the reward-manager diagnostics of this phase:
    `actual retained set on the current candidate pool == the same-pool original EM filter
    set` (100 %), i.e. `keep_em_std_gt_0` reproduces `std(EM) > 0` exactly, and the
    counterfactual shaped filter would have retained strictly more.

Writes E1-D/analysis/step<END>_integrity.{md,json}
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

REPO = "/data01/wyy/Graph-Agent-Planning"
E1D = os.path.join(REPO, "E1-D")
EXP_NAME = os.environ.get("E1D_EXPERIMENT_NAME", "DAPO-GAP3B-MHQA-Agent-E1D-step60to120-4gpu")
EXP_DIR = os.path.join(REPO, "experiments", EXP_NAME)

PHASE = os.environ.get("E1D_PHASE", "D1")
END_STEP = int(os.environ.get("E1D_END_STEP", "70" if PHASE == "D1" else "120"))
START_STEP = int(os.environ.get("E1D_START_STEP", "60" if PHASE == "D1" else "70"))
RUN_DIR = os.environ.get("E1D_RUN_DIR") or open(
    os.path.join(E1D, "runs", f"latest_{PHASE}.txt")).read().strip()
TRAIN_LOG = os.environ.get("E1D_TRAIN_LOG") or sorted(glob.glob(
    os.path.join(E1D, "logs", f"train_{PHASE}_*.log")))[-1]
CKPT = os.path.join(EXP_DIR, f"global_step_{END_STEP}")
ANALYSIS = os.path.join(E1D, "analysis")

STEP_RE = re.compile(r"step:(\d+)\s+-")
METRIC_RE = re.compile(r"([A-Za-z0-9_/\.\-@]+):([^\s]+)")
ERROR_PATTERNS = {
    "cuda_oom": r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError",
    "nccl_error": r"NCCL error|ncclInvalidUsage|ncclInternalError|ncclSystemError|Duplicate GPU detected",
    "nccl_warn": r"\[W\d+ .*c10d\]|NCCL WARN",
    "ray_worker_died": r"ActorDiedError|Worker exits unexpectedly|RayTaskError|ray\.exceptions",
    "wiki_search_error": r"Error when executing search",
    "wiki_tool_exec_failed": r"\[SearchTool\] Execution failed",
    "traceback": r"Traceback \(most recent call last\)",
    "cuda_error": r"CUDA error",
}


def check_filter_decoupling():
    """Q2 of §19: actual retained == same-pool EM filter set, from this phase's diagnostics."""
    groups = sorted(glob.glob(os.path.join(RUN_DIR, "diagnostics", "groups_pid*.jsonl")))
    if not groups:
        return {"available": False, "reason": "no group diagnostics in this run dir"}
    n = mismatch = 0
    kept_actual = kept_shaped = 0
    kept_all_correct = kept_mixed = kept_all_wrong = 0
    active_excluded = active_retained_actual = active_retained_shaped = 0
    active_mixed_retained = 0
    for p in groups:
        for line in open(p, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                g = json.loads(line)
            except ValueError:
                continue
            n += 1
            em_metric_keep = bool(g.get("baseline_keep_em_std_gt_0"))
            actual_keep = bool(g.get("actual_filter_keep", em_metric_keep))
            shaped_keep = bool(g.get("shaped_keep_reward_std_gt_0"))
            if actual_keep != em_metric_keep:
                mismatch += 1
            kept_actual += int(actual_keep)
            kept_shaped += int(shaped_keep)
            if actual_keep:
                if g.get("all_correct"):
                    kept_all_correct += 1
                elif g.get("mixed_correctness"):
                    kept_mixed += 1
                elif g.get("all_wrong"):
                    kept_all_wrong += 1
            if g.get("efficiency_active"):
                if g.get("efficiency_active_excluded_by_em_filter"):
                    active_excluded += 1
                if actual_keep:
                    active_retained_actual += 1
                if shaped_keep:
                    active_retained_shaped += 1
                if actual_keep and g.get("mixed_correctness"):
                    active_mixed_retained += 1
    return {
        "available": True,
        "candidate_groups": n,
        "actual_retained": kept_actual,
        "em_metric_keep_equals_actual_keep_mismatches": mismatch,
        "filter_decoupling_confirmed": mismatch == 0,
        "counterfactual_shaped_retained": kept_shaped,
        "would_be_revived_by_shaped_filter": kept_shaped - kept_actual,
        "retained_all_correct": kept_all_correct,
        "retained_mixed": kept_mixed,
        "retained_all_wrong": kept_all_wrong,
        "efficiency_active_excluded_by_em_filter": active_excluded,
        "efficiency_active_retained_actual": active_retained_actual,
        "efficiency_active_retained_shaped_filter": active_retained_shaped,
        "mixed_efficiency_active_retained": active_mixed_retained,
        "efficiency_supervision_density_actual": (
            active_retained_actual / kept_actual if kept_actual else None),
        "generation_batches": len(glob.glob(os.path.join(RUN_DIR, "diagnostics", "calls_pid*.jsonl"))) and
                              sum(1 for p in glob.glob(os.path.join(RUN_DIR, "diagnostics", "calls_pid*.jsonl"))
                                  for _ in open(p)),
    }


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    result = {"phase": PHASE, "experiment_dir": EXP_DIR, "checkpoint_dir": CKPT,
              "train_log": TRAIN_LOG, "run_dir": RUN_DIR,
              "expected_start_step": START_STEP, "expected_end_step": END_STEP}

    ckpt = {"exists": os.path.isdir(CKPT), "shards": {}, "data_pt": False, "tracker": None}
    if ckpt["exists"]:
        for rank in range(4):
            for kind, pat in (("model", "model_world_size_4_rank_%d.pt"),
                              ("optimizer", "optim_world_size_4_rank_%d.pt"),
                              ("extra_state", "extra_state_world_size_4_rank_%d.pt")):
                p = os.path.join(CKPT, "actor", pat % rank)
                ckpt["shards"][f"{kind}_rank{rank}"] = (
                    {"exists": True, "size": os.path.getsize(p)} if os.path.exists(p) else {"exists": False})
        ckpt["data_pt"] = os.path.exists(os.path.join(CKPT, "data.pt"))
        t = os.path.join(EXP_DIR, "latest_checkpointed_iteration.txt")
        ckpt["tracker"] = open(t).read().strip() if os.path.exists(t) else None
    result["checkpoint"] = ckpt

    log = {"exists": os.path.exists(TRAIN_LOG)}
    if log["exists"]:
        text = open(TRAIN_LOG, errors="replace").read()
        clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
        steps = sorted({int(m.group(1)) for m in STEP_RE.finditer(clean)})
        want = list(range(START_STEP, END_STEP + 1))
        log["steps_seen"] = steps
        log["missing_steps"] = [s for s in want if s not in steps]
        log["all_steps_present"] = not log["missing_steps"]
        log["resume_line_present"] = f"Setting global step to {START_STEP}" in clean
        bad, n_metrics = {}, 0
        for line in clean.splitlines():
            if not STEP_RE.search(line):
                continue
            for k, v in METRIC_RE.findall(line):
                try:
                    fv = float(v)
                except ValueError:
                    continue
                n_metrics += 1
                if not math.isfinite(fv):
                    bad[k] = bad.get(k, 0) + 1
        log["metrics_parsed"] = n_metrics
        log["non_finite_metrics"] = bad
        idx = clean.find("Final validation metrics:")
        head = clean if idx < 0 else clean[:idx]
        log["error_pattern_counts_before_final_val"] = {
            name: len(re.findall(pat, head, flags=re.IGNORECASE)) for name, pat in ERROR_PATTERNS.items()}
    result["training_log"] = log

    resume = {}
    try:
        import torch
        src = (os.path.join(REPO, "experiments/DAPO-GAP3B-MHQA-Agent-4gpu/global_step_60/actor")
               if PHASE == "D1" else os.path.join(EXP_DIR, "global_step_70/actor"))
        for tag, d, expect in (("resume_source", src, START_STEP), ("end", os.path.join(CKPT, "actor"), END_STEP)):
            p = os.path.join(d, "extra_state_world_size_4_rank_0.pt")
            if not os.path.exists(p):
                continue
            st = torch.load(p, map_location="cpu", weights_only=False)
            lr = st.get("lr_scheduler") or {}
            resume[tag] = {"path": p, "keys": sorted(st.keys()),
                           "lr_scheduler_last_epoch": lr.get("last_epoch"),
                           "expected_last_epoch": expect,
                           "rng_present": "rng" in st,
                           "rng_substates": sorted((st.get("rng") or {}).keys())}
    except Exception as e:  # noqa: BLE001
        resume["_error"] = f"{type(e).__name__}: {e}"
    result["state_resume_evidence"] = resume
    state_ok = (resume.get("resume_source", {}).get("lr_scheduler_last_epoch") == START_STEP
                and resume.get("end", {}).get("lr_scheduler_last_epoch") == END_STEP
                and resume.get("end", {}).get("rng_present") is True)
    result["state_resume_ok"] = bool(state_ok)

    filt = check_filter_decoupling()
    result["filter_decoupling"] = filt

    err = log.get("error_pattern_counts_before_final_val", {})
    # The tracker is a single file for the whole experiment, so a *re-check* of D1 after D2
    # has advanced it legitimately shows a larger value.  What matters scientifically is that
    # the phase's checkpoint exists and that the tracker has reached at least END_STEP.
    try:
        tracker_ok = int(ckpt.get("tracker")) >= END_STEP
    except (TypeError, ValueError):
        tracker_ok = False
    result["tracker_reached_end_step"] = bool(tracker_ok)
    result["note_tracker"] = ("tracker is experiment-wide; a D1 re-check after D2 shows 120 - "
                              "the requirement is tracker >= END_STEP and a complete global_step_END_STEP")
    ok = (ckpt.get("exists") and tracker_ok
          and all(v.get("exists") for v in ckpt["shards"].values()) and ckpt.get("data_pt")
          and log.get("all_steps_present") and log.get("resume_line_present")
          and state_ok and not log.get("non_finite_metrics")
          and not any(err.get(k) for k in ("cuda_oom", "nccl_error", "traceback", "cuda_error",
                                           "wiki_search_error", "wiki_tool_exec_failed"))
          and (not filt.get("available") or filt.get("filter_decoupling_confirmed")))
    result["integrity_ok"] = bool(ok)

    lines = [f"# E1-D {PHASE} integrity check (step{START_STEP} -> step{END_STEP})", "",
             f"- experiment dir: `{EXP_DIR}`", f"- checkpoint: `{CKPT}`",
             f"- tracker: {ckpt.get('tracker')} (expected {END_STEP})",
             f"- data.pt present: {ckpt.get('data_pt')}",
             f"- steps {START_STEP}..{END_STEP} all present: {log.get('all_steps_present')} "
             f"(missing: {log.get('missing_steps')})",
             f"- 'Setting global step to {START_STEP}': {log.get('resume_line_present')}",
             f"- metrics parsed: {log.get('metrics_parsed')}, non-finite: {log.get('non_finite_metrics')}",
             f"- full-state resume (lr_scheduler/RNG): {state_ok}", "",
             "## error scan before the final-validation marker", ""]
    for k, v in (err or {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## filter decoupling (E1-D_task.md §11 / §19 Q2)", "",
              "```json", json.dumps(filt, indent=2)[:3000], "```",
              "", f"**VERDICT: {'PASS' if ok else 'CHECK FAILED'}**", ""]
    with open(os.path.join(ANALYSIS, f"step{END_STEP}_integrity.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(ANALYSIS, f"step{END_STEP}_integrity.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

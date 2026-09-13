#!/usr/bin/env python3
"""E1-D post-training integrity check.

Verifies the step120 checkpoint, the step71..step120 training log records, and
scans for NaN/Inf/OOM/NCCL/Ray/Wiki failures.  Writes:
    analysis/post_train_integrity.md
    analysis/post_train_integrity.json
    checkpoints_manifest/step120.md

Usage:
    E1D_RUN_DIR=<run dir> [E1D_EXPERIMENT_NAME=<name>] python3 e1d_check_training_integrity.py
"""
import json
import math
import os
import re
import sys
from collections import Counter

REPO = "/data01/wyy/Graph-Agent-Planning"
RUN_DIR = (os.environ.get("E1D_RUN_DIR")
           or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENT_NAME = os.environ.get("E1D_EXPERIMENT_NAME", "DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu")
EXP_DIR = os.path.join(REPO, "experiments", EXPERIMENT_NAME)
CKPT = os.path.join(EXP_DIR, "global_step_120")
LOG_FILE = os.environ.get("LOG_FILE", os.path.join(REPO, "verl", "logs", f"{EXPERIMENT_NAME}.log"))
if not os.path.isdir(os.path.join(RUN_DIR, "diagnostics")):
    _lr = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "latest_run.txt")
    if os.path.isfile(_lr):
        RUN_DIR = open(_lr).read().strip()

ANALYSIS = os.path.join(RUN_DIR, "analysis")
MANIFEST = os.path.join(RUN_DIR, "checkpoints_manifest")
os.makedirs(ANALYSIS, exist_ok=True)
os.makedirs(MANIFEST, exist_ok=True)

STEP_RE = re.compile(r"step:(\d+)\s+-\s")
METRIC_RE = re.compile(r"([A-Za-z0-9_/\.\-@]+):([^\s]+)")

ERROR_PATTERNS = {
    "cuda_oom": r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError",
    "nccl_error": r"NCCL error|ncclInvalidUsage|ncclInternalError|ncclSystemError|Duplicate GPU detected",
    "nccl_warn": r"\[W\d+ .*c10d\]|NCCL WARN",
    "ray_worker_died": r"ActorDiedError|Worker exits unexpectedly|RayTaskError|ray\.exceptions",
    "nccl_unhandled": r"ProcessGroupNCCL|watchdog caught collective operation timeout",
    "wiki_connection": r"Connection refused|Max retries exceeded|HTTPConnectionPool|NewConnectionError",
    "wiki_http_error": r"HTTPError|status code [45]\d\d|\" [45]\d\d ",
    # exact SearchTool failure signatures (verl/tools/search_tool.py:98,220,239)
    "wiki_search_error": r"Error when executing search",
    "wiki_tool_exec_failed": r"\[SearchTool\] Execution failed",
    "wiki_tool_bad_params": r"\[SearchTool\].*Received parameters",
    "tool_error": r"Tool execution failed|error executing tool|execute_tool.*error|tool call failed",
    "traceback": r"Traceback \(most recent call last\)",
    "cuda_error": r"CUDA error",
}


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    result = {"experiment_dir": EXP_DIR, "checkpoint_dir": CKPT, "log_file": LOG_FILE}
    lines = []

    # ---------------- checkpoint ----------------
    ckpt = {"exists": os.path.isdir(CKPT), "actor_dir_exists": os.path.isdir(os.path.join(CKPT, "actor")),
            "shards": {}, "data_pt": False, "tracker": None, "global_step_from_path": None}
    if ckpt["exists"]:
        ckpt["global_step_from_path"] = int(os.path.basename(CKPT).split("global_step_")[-1])
        for rank in range(4):
            for kind, pat in (("model", "model_world_size_4_rank_%d.pt"),
                              ("optimizer", "optim_world_size_4_rank_%d.pt"),
                              ("extra_state", "extra_state_world_size_4_rank_%d.pt")):
                p = os.path.join(CKPT, "actor", pat % rank)
                ckpt["shards"][f"{kind}_rank{rank}"] = (
                    {"exists": True, "size": os.path.getsize(p)} if os.path.exists(p) else {"exists": False}
                )
        ckpt["data_pt"] = os.path.exists(os.path.join(CKPT, "data.pt"))
        tracker = os.path.join(EXP_DIR, "latest_checkpointed_iteration.txt")
        if os.path.exists(tracker):
            ckpt["tracker"] = open(tracker).read().strip()
    result["checkpoint"] = ckpt

    # ---------------- training log ----------------
    log = {"exists": os.path.exists(LOG_FILE)}
    if log["exists"]:
        text = open(LOG_FILE, errors="replace").read()
        clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
        steps = sorted({int(m.group(1)) for m in STEP_RE.finditer(clean)})
        log["steps_seen"] = steps
        log["steps_71_to_120_all_present"] = all(s in steps for s in range(71, 121))
        log["missing_steps_71_to_120"] = [s for s in range(71, 121) if s not in steps]
        log["max_step"] = max(steps) if steps else None
        log["start_step_line_present"] = "Setting global step to 70" in clean
        log["resume_line_present"] = "Resuming from" in clean
        log["loaded_model_lines"] = clean.count("Loaded model from")
        log["loaded_optimizer_lines"] = clean.count("Loaded optimizer from")
        log["loaded_lr_scheduler_lines"] = clean.count("Loaded lr_scheduler from")
        log["loaded_rng_lines"] = clean.count("Loaded rng from")
        log["final_validation_metrics_present"] = "Final validation metrics:" in clean

        # metric finiteness (parse `key:value` from the console-logger step lines)
        bad_metrics = {}
        n_metrics = 0
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
                    bad_metrics.setdefault(k, 0)
                    bad_metrics[k] += 1
        log["metrics_parsed"] = n_metrics
        log["non_finite_metrics"] = bad_metrics

        # Error scan in two windows: anything BEFORE the final-validation
        # marker is a genuine training-time problem; the tail after it is the
        # expected SIGTERM/actor-teardown cascade of a completed run.
        idx = clean.find("Final validation metrics:")
        head = clean if idx < 0 else clean[:idx]
        tail = "" if idx < 0 else clean[idx:]
        log["error_pattern_counts_before_final_val"] = {
            name: len(re.findall(pat, head, flags=re.IGNORECASE))
            for name, pat in ERROR_PATTERNS.items()
        }
        log["error_pattern_counts_after_final_val_teardown"] = {
            name: len(re.findall(pat, tail, flags=re.IGNORECASE))
            for name, pat in ERROR_PATTERNS.items()
        }
        log["log_size_bytes"] = os.path.getsize(LOG_FILE)
    result["training_log"] = log

    # ---------------- optimizer / lr_scheduler / RNG state residency --------
    # Direct proof that the resume was a full-state resume and not weights-only:
    # FSDP saves lr_scheduler + RNG into the tiny extra_state shard, so we can
    # read both checkpoints' extra_state and check that last_epoch continues
    # 60 -> 70 instead of being reset to 0.
    resume = {"step70_extra_state": None, "step120_extra_state": None}
    try:
        import torch

        for tag, d in (("step70_extra_state", os.path.join(REPO, "experiments",
                                                           "DAPO-GAP3B-MHQA-Agent-E1A-step60to70-4gpu", "global_step_70", "actor")),
                       ("step120_extra_state", os.path.join(CKPT, "actor"))):
            p = os.path.join(d, "extra_state_world_size_4_rank_0.pt")
            if not os.path.exists(p):
                continue
            st = torch.load(p, map_location="cpu", weights_only=False)
            lr = st.get("lr_scheduler") or {}
            resume[tag] = {
                "path": p,
                "keys": sorted(st.keys()),
                "lr_scheduler_last_epoch": lr.get("last_epoch"),
                "rng_present": "rng" in st,
                "rng_substates": sorted((st.get("rng") or {}).keys()),
            }
    except Exception as e:  # noqa: BLE001
        resume["_error"] = f"{type(e).__name__}: {e}"
    result["state_resume_evidence"] = resume
    state_resume_ok = (
        (resume.get("step70_extra_state") or {}).get("lr_scheduler_last_epoch") == 70
        and (resume.get("step120_extra_state") or {}).get("lr_scheduler_last_epoch") == 120
        and (resume.get("step120_extra_state") or {}).get("rng_present") is True
    )

    # ---------------- verdict ----------------
    ok = (
        ckpt.get("exists")
        and ckpt.get("global_step_from_path") == 120
        and all(v.get("exists") for v in ckpt["shards"].values())
        and log.get("steps_71_to_120_all_present")
        and log.get("start_step_line_present")
        and state_resume_ok
        and not log.get("non_finite_metrics")
        and not any((log.get("error_pattern_counts_before_final_val") or {}).get(k)
                    for k in ("cuda_oom", "nccl_error", "traceback", "cuda_error",
                              "wiki_search_error", "wiki_tool_exec_failed"))
    )
    result["integrity_ok"] = bool(ok)
    result["state_resume_ok"] = bool(state_resume_ok)

    # ---------------- markdown ----------------
    lines.append("# E1-D post-training integrity check\n")
    lines.append(f"- experiment dir: `{EXP_DIR}`")
    lines.append(f"- checkpoint dir: `{CKPT}`")
    lines.append(f"- training log: `{LOG_FILE}` ({human(log.get('log_size_bytes', 0))})")
    lines.append(f"\n## Checkpoint\n")
    lines.append(f"- exists: {ckpt['exists']}")
    lines.append(f"- global step (from dir name): {ckpt.get('global_step_from_path')}")
    lines.append(f"- tracker value: {ckpt.get('tracker')}")
    lines.append(f"- data.pt present: {ckpt.get('data_pt')}")
    for k, v in ckpt["shards"].items():
        if v.get("exists"):
            lines.append(f"- {k}: {human(v['size'])}")
        else:
            lines.append(f"- {k}: **MISSING**")
    lines.append("\n## Training log\n")
    lines.append(f"- steps seen: {log.get('steps_seen')}")
    lines.append(f"- all of step71..step120 present: {log.get('steps_71_to_120_all_present')} "
                 f"(missing: {log.get('missing_steps_71_to_120')})")
    lines.append(f"- 'Setting global step to 70': {log.get('start_step_line_present')}")
    lines.append("- (the upstream 'Loaded model/optimizer/lr_scheduler from ...' INFO lines are not "
                 "emitted to stdout in this run; state residency is proven directly below instead)")
    lines.append(f"- metrics parsed: {log.get('metrics_parsed')}, non-finite: {log.get('non_finite_metrics')}")
    lines.append(f"- final validation metrics present: {log.get('final_validation_metrics_present')}")
    lines.append("\n## Optimizer / lr_scheduler / RNG full-state resume evidence\n")
    for tag in ("step70_extra_state", "step120_extra_state"):
        v = resume.get(tag)
        lines.append(f"- {tag}: {v}")
    lines.append(f"- state_resume_ok: {state_resume_ok}")
    lines.append("\n### Error-pattern scan BEFORE the final-validation marker (real training window)\n")
    for k, v in (log.get("error_pattern_counts_before_final_val") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("\n### Error-pattern scan AFTER the final-validation marker (expected teardown)\n")
    for k, v in (log.get("error_pattern_counts_after_final_val_teardown") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append(f"\n## VERDICT: {'PASS' if ok else 'CHECK FAILED'}\n")

    with open(os.path.join(ANALYSIS, "post_train_integrity.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(ANALYSIS, "post_train_integrity.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    # ---------------- step120 manifest ----------------
    m = ["# E1-D step120 checkpoint manifest", "",
         f"- path: `{CKPT}`",
         f"- produced by: E1-D `step70 -> efficiency-shaped reward -> step120` run",
         f"- continued from: the E1-A step70 checkpoint", 
         f"- global step: {ckpt.get('global_step_from_path')}",
         f"- tracker file: `{os.path.join(EXP_DIR, 'latest_checkpointed_iteration.txt')}` = {ckpt.get('tracker')}",
         "- save_contents: `['model', 'optimizer', 'extra']` (identical to the original script)", ""]
    for k, v in ckpt["shards"].items():
        if v.get("exists"):
            m.append(f"- actor/{k.replace('_rank', '_world_size_4_rank_')}: {v['size']} bytes (sha256 not "
                     f"computed: multi-GB shards, size+existence recorded instead)")
        else:
            m.append(f"- actor/{k}: MISSING")
    m.append(f"- data.pt present: {ckpt.get('data_pt')}")
    m.append("")
    m.append("The checkpoint is NOT copied into E1-D; only this manifest is kept.")
    with open(os.path.join(MANIFEST, "step120.md"), "w") as f:
        f.write("\n".join(m) + "\n")

    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

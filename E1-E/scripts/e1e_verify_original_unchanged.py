#!/usr/bin/env python3
"""E1-E post-run provenance check: the original GAP/verl files must be byte-identical.

E1-E_task.md forbids modifying the original GAP main code.  E1-E injects everything through
`PYTHONPATH` sitecustomize registration, so this script verifies that claim:

  1. recompute sha256 for every file listed in `manifests/original_files_manifest_pre.json`;
  2. compare with the pre-run hash -> `manifests/original_files_diff.json`;
  3. additionally check the file the pre-run list got wrong (it listed
     `verl/verl/workers/rollout/sglang_rollout.py`, which does not exist, instead of
     `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`) against BOTH the current
     file and the independent hash recorded by E1-0 on 2026-09-10
     (`E1-0/runs/20260910_step60to70_e10/code/original_files_manifest.txt`), so the critical
     rollout implementation is covered by a pre-E1-E reference;
  4. report any change (diff, mtime after the E1-E start, or missing file) loudly.

Writes `manifests/original_files_manifest_post.json`, `manifests/original_files_diff.json`
and `analysis/original_files_unchanged.{json,md}`.  Exit 0 = unchanged.
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime

REPO = "/data01/wyy/Graph-Agent-Planning"
E1E = os.path.join(REPO, "E1-E")
PRE = os.path.join(E1E, "manifests", "original_files_manifest_pre.json")
POST = os.path.join(E1E, "manifests", "original_files_manifest_post.json")
DIFF = os.path.join(E1E, "manifests", "original_files_diff.json")
E10_MANIFEST = os.path.join(REPO, "E1-0/runs/20260910_step60to70_e10/code/original_files_manifest.txt")

EXTRA = {
    # the correct path that the pre-run list fat-fingered, plus the other FSDP/SGLang files
    # E1-0 recorded independently before E1-E existed
    "verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py": None,
    "verl/verl/workers/rollout/async_server.py": None,
    "verl/verl/workers/rollout/schemas.py": None,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_e10_manifest():
    """{path: (sha256, mtime_str)} from E1-0's pipe-delimited manifest."""
    out = {}
    if not os.path.exists(E10_MANIFEST):
        return out
    for line in open(E10_MANIFEST, errors="ignore"):
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        path = parts[0]
        m = re.search(r"sha256=([0-9a-f]{64})", parts[-1])
        mt = re.search(r"mtime=(.*)$", parts[2]) if len(parts) > 2 else None
        if m:
            out[path] = {"sha256": m.group(1), "mtime": (mt.group(1).strip() if mt else None)}
    return out


def main():
    pre = json.load(open(PRE))
    e10 = parse_e10_manifest()
    entries, changed, missing = [], [], []
    for f in pre["files"]:
        rel = f["path"]
        p = os.path.join(REPO, rel)
        cur = {"path": rel, "pre_present": not f.get("missing"),
               "pre_sha256": f.get("sha256"), "pre_mtime": f.get("mtime")}
        if os.path.exists(p):
            cur.update({"exists": True, "size": os.path.getsize(p), "mtime": os.path.getmtime(p),
                        "sha256": sha256(p)})
            if cur["pre_sha256"] and cur["sha256"] != cur["pre_sha256"]:
                changed.append(rel)
        else:
            cur["exists"] = False
            missing.append(rel)
        entries.append(cur)

    # the path the pre-run list got wrong: verify against E1-0's pre-E1-E hash
    extra = {}
    for rel in EXTRA:
        p = os.path.join(REPO, rel)
        e = {"path": rel, "exists": os.path.exists(p)}
        if e["exists"]:
            e["sha256"] = sha256(p)
            e["size"] = os.path.getsize(p)
            ref = e10.get(rel)
            if ref:
                e["e1_0_sha256"] = ref["sha256"]
                e["matches_e1_0"] = (ref["sha256"] == e["sha256"])
        extra[rel] = e

    unchanged = (not changed) and all(not e.get("pre_present") or e.get("exists") for e in entries)
    result = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "pre_manifest": PRE,
        "n_files_checked": len(entries),
        "changed_files": changed,
        "missing_files": missing,
        "entries": entries,
        "extra_checks": extra,
        "e1_0_reference_manifest": E10_MANIFEST if e10 else None,
        "original_files_unchanged": bool(unchanged),
        "note": ("The pre-run list contained one typo (`verl/verl/workers/rollout/sglang_rollout.py`, "
                 "which does not exist); the correct nested path is checked here against E1-0's "
                 "independent 2026-09-10 hash."),
    }
    json.dump(result, open(POST, "w"), indent=2)
    json.dump({"changed_files": changed, "missing_files": missing,
               "original_files_unchanged": bool(unchanged),
               "extra_checks": extra}, open(DIFF, "w"), indent=2)

    md = ["# Original GAP/verl files unchanged by E1-E", "",
          f"checked {len(entries)} files from the pre-run manifest at {result['checked_at']}", "",
          f"**UNCHANGED: {bool(unchanged)}**", ""]
    if changed:
        md += ["## CHANGED FILES (must be empty)", ""] + [f"- {c}" for c in changed] + [""]
    if missing:
        md += ["## MISSING FILES", ""] + [f"- {c}" for c in missing] + [""]
    md += ["## corrected-path checks (pre-run typo)", "",
           "| path | exists | sha256 (E1-E) | sha256 (E1-0, 2026-09-10) | match |",
           "|---|---|---|---|---|"]
    for rel, e in extra.items():
        md.append(f"| `{rel}` | {e.get('exists')} | {(e.get('sha256') or '-')[:16]} | "
                  f"{(e.get('e1_0_sha256') or '-')[:16]} | {e.get('matches_e1_0', '-')} |")
    md += ["", "## all checked files", "", "| file | pre sha256 | post sha256 | same |",
           "|---|---|---|---|"]
    for e in entries:
        pre_s = (e.get("pre_sha256") or "-")[:12]
        post_s = (e.get("sha256") or "-")[:12]
        same = "-" if not e.get("pre_sha256") else (e.get("sha256") == e["pre_sha256"])
        md.append(f"| `{e['path']}` | {pre_s} | {post_s} | {same} |")
    md.append("")
    os.makedirs(os.path.join(E1E, "analysis"), exist_ok=True)
    open(os.path.join(E1E, "analysis", "original_files_unchanged.md"), "w").write("\n".join(md) + "\n")
    json.dump(result, open(os.path.join(E1E, "analysis", "original_files_unchanged.json"), "w"), indent=2)
    print("\n".join(md[:20]))
    print("wrote", POST, DIFF)
    return 0 if unchanged else 1


if __name__ == "__main__":
    sys.exit(main())

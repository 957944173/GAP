#!/usr/bin/env python3
"""Write one manifest per E1-E checkpoint into E1-E/checkpoints/ (E1-E_task.md §5).

For every global_step_<N> of the E1-E experiment:
    * existence + byte size of every actor shard (model / optimizer / extra_state x 4 ranks)
    * data.pt (dataloader state) presence + sha256 + size
    * tracker file value
    * the per-checkpoint statistics from checkpoint_stats.json (mean reward, mean EM,
      mean efficiency bonus, active efficiency groups, retained groups, mean search
      batches, mean queries, parallel factor)

Outputs: E1-E/checkpoints/global_step_<N>.md and E1-E/checkpoints/manifest.json
"""
import argparse
import hashlib
import json
import os

REPO = "/data01/wyy/Graph-Agent-Planning"
EXP = "DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu"
STEPS = [70, 80, 90, 100, 110, 120]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "checkpoints"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stats_path = os.path.join(args.out, "checkpoint_stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as fh:
            sj = json.load(fh)
        stats = {r["step"]: r for r in sj.get("checkpoints", [])}

    exp_dir = os.path.join(REPO, "experiments", EXP)
    tracker = None
    tp = os.path.join(exp_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(tp):
        tracker = open(tp).read().strip()

    manifest = {"experiment": EXP, "experiment_dir": exp_dir, "tracker": tracker, "checkpoints": {}}
    for s in STEPS:
        d = os.path.join(exp_dir, f"global_step_{s}")
        entry = {"path": d, "exists": os.path.isdir(d), "shards": {}, "data_pt": None}
        if entry["exists"]:
            for rank in range(4):
                for kind, pat in (("model", "model_world_size_4_rank_%d.pt"),
                                  ("optimizer", "optim_world_size_4_rank_%d.pt"),
                                  ("extra_state", "extra_state_world_size_4_rank_%d.pt")):
                    p = os.path.join(d, "actor", pat % rank)
                    entry["shards"][f"{kind}_rank{rank}"] = (
                        {"exists": True, "size": os.path.getsize(p)} if os.path.exists(p)
                        else {"exists": False})
            dp = os.path.join(d, "data.pt")
            if os.path.exists(dp):
                entry["data_pt"] = {"exists": True, "size": os.path.getsize(dp),
                                    "sha256": sha256(dp) if os.path.getsize(dp) < (1 << 28) else "skipped-large"}
            else:
                entry["data_pt"] = {"exists": False}
            # merged HF model is created by the evaluation step (may not exist yet)
            entry["merged_hf"] = os.path.exists(os.path.join(d, "actor", "merged_hf", "config.json"))
            entry["stats"] = stats.get(s)
        manifest["checkpoints"][str(s)] = entry

        md = [f"# E1-E checkpoint global_step_{s}", ""]
        md.append(f"- path: `{d}`")
        md.append(f"- tracker (`latest_checkpointed_iteration.txt`): {tracker}")
        if not entry["exists"]:
            md.append("- **MISSING**")
        else:
            for k, v in entry["shards"].items():
                md.append(f"- actor/{k.replace('_rank', '_world_size_4_rank_')}: "
                          f"{v['size']} bytes" if v.get("exists") else f"- actor/{k}: MISSING")
            md.append(f"- data.pt: {entry['data_pt']}")
            md.append(f"- merged HF present: {entry['merged_hf']}")
            st = entry.get("stats")
            if st:
                for win in ("window_10step", "cumulative"):
                    w = st.get(win)
                    if not w:
                        continue
                    md.append(f"\n## statistics ({win})")
                    for k in ("n_rollouts", "n_groups", "mean_reward", "mean_em",
                              "mean_efficiency_bonus", "active_efficiency_groups",
                              "retained_groups_shaped", "retained_groups_baseline_em",
                              "newly_added_by_shaping", "mean_search_batches", "mean_queries",
                              "parallel_factor", "mean_conversation_turns", "mean_response_tokens",
                              "zero_search_rate", "zero_search_correct_count", "metadata_invalid"):
                        md.append(f"- {k}: {w.get(k)}")
        with open(os.path.join(args.out, f"global_step_{s}.md"), "w") as fh:
            fh.write("\n".join(md) + "\n")

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(json.dumps({k: {"exists": v["exists"], "merged_hf": v.get("merged_hf"),
                          "has_stats": bool(v.get("stats"))} for k, v in manifest["checkpoints"].items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

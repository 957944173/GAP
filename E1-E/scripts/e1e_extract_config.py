#!/usr/bin/env python3
"""Extract the resolved Hydra training config from an E1-E/E1-0 training log.

The verl TaskRunner prints `pprint(OmegaConf.to_container(config, resolve=True))`
right after "TaskRunner hostname:".  This pulls that dict out and saves it so the
run directory keeps a faithful record of what was actually executed.

Usage: python3 e1e_extract_config.py --log <stdout.log> --out <path.txt>
"""
import argparse
import json
import os
import re

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None, help="optional JSON output path")
    args = ap.parse_args()

    start = None
    depth = 0
    collected = []
    with open(args.log, errors="ignore") as f:
        for raw in f:
            line = ANSI.sub("", raw).rstrip("\n")
            if start is None:
                if "TaskRunner hostname:" in line:
                    start = "armed"          # config pprint follows
                continue
            if start == "armed":
                s0 = re.sub(r"^\(.*?\)\s*", "", line)
                if s0.strip().startswith("{"):
                    start = "collecting"
                    collected.append(s0)
                    depth += s0.count("{") - s0.count("}")
                    if depth <= 0:
                        break
                continue
            if start == "collecting":
                # strip Ray's "(TaskRunner pid=N) " prefix if present
                s = re.sub(r"^\(.*?\)\s*", "", line)
                collected.append(s)
                depth += s.count("{") - s.count("}")
                if depth <= 0 and collected:
                    break

    text = "\n".join(collected)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"extracted {len(collected)} config lines -> {args.out}")

    if args.json and text.strip().startswith("{"):
        # pprint uses single quotes; normalise the common cases for JSON
        try:
            import ast
            obj = ast.literal_eval(text)
            with open(args.json, "w") as f:
                json.dump(obj, f, indent=2, default=str)
            print(f"wrote JSON config -> {args.json}")
        except Exception as e:  # noqa: BLE001
            print(f"(could not convert to JSON: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

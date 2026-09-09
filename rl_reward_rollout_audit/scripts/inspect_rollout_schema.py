#!/usr/bin/env python3
"""Inspect a few rows of GAP's RL parquet without launching training.

The script reports the source parquet schema and the fields that RLHFDataset
places into DataProto. It reads only the first parquet row group when pyarrow is
available; it never modifies the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = REPO_ROOT / "Agent/data/mhqa_agent/GAP-MHQA-RL-Dataset/GAP-RL-16w.cleaned.parquet"


def short_value(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {str(k): short_value(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 3:
            return [short_value(v, depth + 1) for v in value[:3]] + [f"... ({len(value)} items)"]
        return [short_value(v, depth + 1) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return short_value(value.tolist(), depth + 1)
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = value if not isinstance(value, str) else value[:400]
        return text
    return repr(value)[:400]


def read_parquet_sample(path: Path, rows: int) -> tuple[list[dict], list[dict]]:
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        schema = [
            {"name": field.name, "type": str(field.type)}
            for field in parquet_file.schema_arrow
        ]
        table = parquet_file.read_row_group(0)
        sample = table.slice(0, min(rows, table.num_rows)).to_pylist()
        return sample, schema
    except Exception:
        import pandas as pd

        dataframe = pd.read_parquet(path).head(rows)
        schema = [{"name": name, "type": str(dtype)} for name, dtype in dataframe.dtypes.items()]
        return dataframe.to_dict(orient="records"), schema


def inspect_jsonl(path: Path, rows: int) -> dict:
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(samples) >= rows:
                break
            if line.strip():
                samples.append(json.loads(line))
    return {
        "path": str(path),
        "sample_count": len(samples),
        "keys": sorted({key for sample in samples for key in sample}),
        "samples": [short_value(sample) for sample in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=2)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "parquet": {"path": str(args.parquet), "exists": args.parquet.exists()},
        "dataproto_runtime_schema": {
            "batch": [
                "input_ids",
                "attention_mask",
                "position_ids",
                "prompts",
                "responses",
                "loss_mask",
                "response_mask",
                "token_level_scores",
                "token_level_rewards",
                "advantages",
                "returns",
            ],
            "non_tensor_batch": [
                "data_source",
                "prompt",
                "reward_model",
                "extra_info",
                "answer",
                "index",
                "uid",
                "messages",
                "complete_reason",
                "finish_reason",
                "tool_call_info",
                "detailed_tool_metrics",
            ],
            "note": "Fields are stage-dependent; this is the union observed from RLHFDataset, trainer, and sglang rollout construction.",
        },
    }

    if args.parquet.exists():
        samples, schema = read_parquet_sample(args.parquet, args.rows)
        result["parquet"]["schema"] = schema
        result["parquet"]["samples"] = [short_value(sample) for sample in samples]

    if args.jsonl is not None and args.jsonl.exists():
        result["jsonl"] = inspect_jsonl(args.jsonl, args.rows)

    print(json.dumps(result, indent=2, ensure_ascii=False, default=repr))


if __name__ == "__main__":
    main()

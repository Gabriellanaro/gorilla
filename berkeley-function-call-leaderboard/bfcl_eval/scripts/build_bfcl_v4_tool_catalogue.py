#!/usr/bin/env python3
"""
build_bfcl_v4_tool_catalogue.py

Creates a tool catalogue from bfcl_v4_all_internal.jsonl.

- One row per unique tool_name
- Tool uniqueness = tool.name only
- Keeps ALL distinct orig_descriptions and parameter schemas
- Tracks provenance (source_files, task_types)
- Counts occurrences, conversations, and GT usage
- Outputs JSONL

Output schema:
{
  "tool_name": str,
  "orig_descriptions": [
    {"description": str, "source_files": [str, ...]}
  ],
  "parameters_variants": [
    {"parameters": dict, "source_files": [str, ...]}
  ],
  "source_files": [str, ...],
  "task_types": ["single_turn", "multi_turn"],
  "num_occurrences": int,
  "num_conversations": int,
  "num_as_ground_truth": int
}
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any


INPUT_PATH = Path("bfcl_eval/data/internal/bfcl_v4_all_internal.jsonl")
OUTPUT_PATH = Path("bfcl_eval/data/internal/bfcl_v4_tool_catalogue.jsonl")


def base_id(full_id: str) -> str:
    """Remove __callN suffix if present."""
    return re.sub(r"__call\d+$", "", full_id)


def canonicalize(obj: Any) -> str:
    """Stable string representation for dict comparison."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def main():
    assert INPUT_PATH.exists(), f"Missing input file: {INPUT_PATH}"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tools = {}

    # bookkeeping for per-conversation counting
    tool_to_conversations = defaultdict(set)

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            src = row["source_file"]
            task_type = row["task_type"]
            bid = base_id(row["id"])
            gt_tool = row.get("ground_truth", {}).get("tool_name")

            for tool in row.get("tools", []):
                name = tool.get("name")
                if not name:
                    continue

                if name not in tools:
                    tools[name] = {
                        "tool_name": name,
                        "orig_descriptions": {},
                        "parameters_variants": {},
                        "source_files": set(),
                        "task_types": set(),
                        "num_occurrences": 0,
                        "num_conversations": set(),
                        "num_as_ground_truth": 0,
                    }

                entry = tools[name]

                # occurrences
                entry["num_occurrences"] += 1
                entry["source_files"].add(src)
                entry["task_types"].add(task_type)
                entry["num_conversations"].add(bid)

                # description variants
                desc = tool.get("orig_description")
                if isinstance(desc, str) and desc.strip():
                    key = desc.strip()
                    if key not in entry["orig_descriptions"]:
                        entry["orig_descriptions"][key] = set()
                    entry["orig_descriptions"][key].add(src)

                # parameter variants
                params = tool.get("parameters")
                if isinstance(params, dict):
                    pkey = canonicalize(params)
                    if pkey not in entry["parameters_variants"]:
                        entry["parameters_variants"][pkey] = {
                            "parameters": params,
                            "source_files": set(),
                        }
                    entry["parameters_variants"][pkey]["source_files"].add(src)

                # ground truth usage
                if gt_tool == name:
                    entry["num_as_ground_truth"] += 1

    # write JSONL
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for tool_name, entry in sorted(tools.items()):
            out_row = {
                "tool_name": tool_name,
                "orig_descriptions": [
                    {
                        "description": desc,
                        "source_files": sorted(list(srcs)),
                    }
                    for desc, srcs in entry["orig_descriptions"].items()
                ],
                "parameters_variants": [
                    {
                        "parameters": pv["parameters"],
                        "source_files": sorted(list(pv["source_files"])),
                    }
                    for pv in entry["parameters_variants"].values()
                ],
                "source_files": sorted(list(entry["source_files"])),
                "task_types": sorted(list(entry["task_types"])),
                "num_occurrences": entry["num_occurrences"],
                "num_conversations": len(entry["num_conversations"]),
                "num_as_ground_truth": entry["num_as_ground_truth"],
            }

            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print("✅ Tool catalogue built")
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Unique tools: {len(tools)}")


if __name__ == "__main__":
    main()

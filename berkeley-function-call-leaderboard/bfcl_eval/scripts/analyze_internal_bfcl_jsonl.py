#!/usr/bin/env python3
"""
analyze_internal_bfcl_jsonl.py

Analyze the combined BFCL v4 internal JSONL dataset produced by your builder.

Usage:
  python scripts/analyze_internal_bfcl_jsonl.py \
    --input data/internal/bfcl_v4_all_internal.jsonl \
    --outdir data/internal/_analysis_bfcl_v4

Outputs:
  - summary.json
  - REPORT.md (human-readable)
  - by_source_file.csv
  - by_task_type.csv
  - decisions_per_conversation_dist.csv
  - candidate_tools_count_dist.csv
  - top_candidate_tool_names.csv
  - top_ground_truth_tool_names.csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Bad JSON at line {i} in {path}: {e}") from e
    return rows


def base_id(full_id: str) -> str:
    # remove trailing __callN (your builder uses this for multiple GT calls)
    return re.sub(r"__call\d+$", "", full_id)


def safe_len(x: Any) -> int:
    return len(x) if isinstance(x, list) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Internal JSONL file to analyze")
    ap.add_argument("--outdir", type=str, required=True, help="Output folder for analysis artifacts")
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(inp)
    if not rows:
        raise RuntimeError(f"No rows found in {inp}")

    # Build a flat dataframe of per-row diagnostics
    records = []
    all_candidate_tool_names: List[str] = []

    for r in rows:
        rid = str(r.get("id", ""))
        src = r.get("source_file")
        task_type = r.get("task_type")

        msgs = r.get("messages") or []
        tools = r.get("tools") or []
        gt = r.get("ground_truth") or {}

        gt_tool = gt.get("tool_name")
        gt_args = gt.get("tool_arguments")
        turn_index = gt.get("turn_index")

        n_tools = safe_len(tools)
        n_msgs = safe_len(msgs)

        n_user = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "user")
        n_assistant = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant")
        n_tool_msgs = sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "tool")

        for t in tools:
            if isinstance(t, dict) and isinstance(t.get("name"), str):
                all_candidate_tool_names.append(t["name"])

        records.append(
            {
                "id": rid,
                "base_id": base_id(rid),
                "source_file": src,
                "task_type": task_type,
                "n_tools": n_tools,
                "n_messages": n_msgs,
                "n_user_messages": n_user,
                "n_assistant_messages": n_assistant,
                "n_tool_messages": n_tool_msgs,
                "gt_tool_name": gt_tool,
                "gt_has_args": isinstance(gt_args, dict) and len(gt_args) > 0,
                "turn_index": turn_index,
                "gt_missing": gt_tool is None,
                "gt_is_final_answer": gt_tool == "final_answer",
            }
        )

    df = pd.DataFrame.from_records(records)

    summary = {
        "rows_total": int(len(df)),
        "unique_conversations_base_id": int(df["base_id"].nunique()),
        "unique_source_files": int(df["source_file"].nunique()),
        "task_type_counts": df["task_type"].value_counts(dropna=False).to_dict(),
        "rows_with_multiple_candidate_tools": int((df["n_tools"] > 1).sum()),
        "rows_with_empty_tools": int((df["n_tools"] == 0).sum()),
        "rows_with_empty_messages": int((df["n_messages"] == 0).sum()),
        "rows_with_missing_gt_tool_name": int(df["gt_missing"].sum()),
        "rows_with_gt_args_present": int(df["gt_has_args"].sum()),
    }

    # by source_file
    by_source = (
        df.groupby("source_file")
        .agg(
            rows=("id", "size"),
            conversations=("base_id", "nunique"),
            single_turn_rows=("task_type", lambda s: int((s == "single_turn").sum())),
            multi_turn_rows=("task_type", lambda s: int((s == "multi_turn").sum())),
            avg_tools=("n_tools", "mean"),
            p_multi_tool=("n_tools", lambda s: float((s > 1).mean())),
            p_empty_tools=("n_tools", lambda s: float((s == 0).mean())),
            p_missing_gt=("gt_missing", "mean"),
            p_empty_messages=("n_messages", lambda s: float((s == 0).mean())),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    # by task_type
    by_task = (
        df.groupby("task_type")
        .agg(
            rows=("id", "size"),
            conversations=("base_id", "nunique"),
            avg_tools=("n_tools", "mean"),
            p_multi_tool=("n_tools", lambda s: float((s > 1).mean())),
            p_empty_tools=("n_tools", lambda s: float((s == 0).mean())),
            p_missing_gt=("gt_missing", "mean"),
            p_empty_messages=("n_messages", lambda s: float((s == 0).mean())),
        )
        .reset_index()
    )

    # distribution: decisions per conversation
    decisions_per_conv = df.groupby("base_id").size().rename("n_decisions").reset_index()
    decisions_dist = (
        decisions_per_conv["n_decisions"]
        .value_counts()
        .sort_index()
        .rename_axis("n_decisions")
        .reset_index(name="n_conversations")
    )

    # distribution: number of candidate tools per row
    tools_dist = (
        df["n_tools"].value_counts().sort_index().rename_axis("n_tools").reset_index(name="n_rows")
    )

    # top candidate tool names
    tool_freq = pd.DataFrame(Counter(all_candidate_tool_names).most_common(50), columns=["tool_name", "count"])

    # top GT tool names
    gt_tool_freq = (
        df["gt_tool_name"]
        .fillna("<<MISSING>>")
        .value_counts()
        .head(50)
        .rename_axis("gt_tool_name")
        .reset_index(name="count")
    )

    # Write outputs
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    by_source.to_csv(outdir / "by_source_file.csv", index=False)
    by_task.to_csv(outdir / "by_task_type.csv", index=False)
    decisions_dist.to_csv(outdir / "decisions_per_conversation_dist.csv", index=False)
    tools_dist.to_csv(outdir / "candidate_tools_count_dist.csv", index=False)
    tool_freq.to_csv(outdir / "top_candidate_tool_names.csv", index=False)
    gt_tool_freq.to_csv(outdir / "top_ground_truth_tool_names.csv", index=False)

    # Human-readable report (with red flags)
    red_flags = []
    if summary["rows_with_empty_messages"] == summary["rows_total"]:
        red_flags.append("ALL rows have empty `messages` (builder likely failed to parse `question/messages`).")
    if summary["rows_with_empty_tools"] > 0:
        red_flags.append("Some rows have empty `tools` (tool list parsing failed for some sources).")
    if summary["rows_with_missing_gt_tool_name"] > 0:
        red_flags.append("Some rows have missing `ground_truth.tool_name` (GT parsing mismatch).")

    report = []
    report.append("# BFCL v4 internal dataset analysis\n")
    report.append(f"- Rows (tool-call decisions): **{summary['rows_total']}**")
    report.append(f"- Unique conversations (base_id): **{summary['unique_conversations_base_id']}**")
    report.append(f"- Source files: **{summary['unique_source_files']}**")
    report.append(f"- Task type counts: {summary['task_type_counts']}")
    report.append(f"- Rows with >1 candidate tool: **{summary['rows_with_multiple_candidate_tools']}** ({summary['rows_with_multiple_candidate_tools']/summary['rows_total']:.1%})")
    report.append(f"- Rows with empty tools: **{summary['rows_with_empty_tools']}** ({summary['rows_with_empty_tools']/summary['rows_total']:.1%})")
    report.append(f"- Rows with missing GT tool_name: **{summary['rows_with_missing_gt_tool_name']}** ({summary['rows_with_missing_gt_tool_name']/summary['rows_total']:.1%})")
    report.append(f"- Rows with empty messages: **{summary['rows_with_empty_messages']}** ({summary['rows_with_empty_messages']/summary['rows_total']:.1%})\n")

    report.append("## Red flags\n")
    if red_flags:
        for rf in red_flags:
            report.append(f"- **{rf}**")
    else:
        report.append("- None detected.\n")

    report.append("\n## Output files\n")
    report += [
        "- summary.json",
        "- by_source_file.csv",
        "- by_task_type.csv",
        "- decisions_per_conversation_dist.csv",
        "- candidate_tools_count_dist.csv",
        "- top_candidate_tool_names.csv",
        "- top_ground_truth_tool_names.csv",
    ]

    (outdir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("✅ Analysis complete")
    print(f"Input: {inp}")
    print(f"Outdir: {outdir}")
    print("Key summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

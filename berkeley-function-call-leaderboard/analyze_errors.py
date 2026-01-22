#!/usr/bin/env python
"""
Error analysis pipeline for BFCL-style JSONL evaluation outputs.

Central assumption: these JSONL files only log errors, so missing task ids in a
condition imply that id was correct (valid=True) for that condition. This
assumption is required for paired transition analysis across conditions.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


CONDITION_LABELS = {
    "score_desc_original_name_original": "OO",
    "score_desc_original_name_augmented": "OA",
    "score_desc_augmented_name_original": "AO",
    "score_desc_augmented_name_augmented": "AA",
}

DEFAULT_CONDITIONS = ["OO", "OA", "AO", "AA"]

MACRO_BUCKET_RULES: List[Tuple[str, str]] = [
    (r"(tool|function)[ _-]?select|wrong tool|no tool|tool mismatch", "tool_selection"),
    (r"schema|json|format|parse|decode|malformed", "schema_compliance"),
    (r"missing required|required field|missing argument|missing key", "missing_required"),
    (r"format|decode|deserialize|output parse|invalid json", "formatting_decoding"),
    (r"runtime|execution|timeout|exception|error during inference", "execution_runtime"),
]


@dataclass
class ParseResult:
    summary: Dict[str, Any]
    rows: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BFCL error analysis pipeline")
    parser.add_argument(
        "--root",
        default=".",
        help="Project root containing score_desc_* folders (default: current dir)",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated substrings to filter model_name (optional)",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated condition labels to include (default: OO,OA,AO,AA)",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_out",
        help="Output directory (default: analysis_out)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional limit for number of JSON files to process",
    )
    return parser.parse_args()


def find_json_files(root: Path, condition_folder: str) -> List[Path]:
    base = root / condition_folder
    if not base.exists():
        return []
    return [p for p in base.rglob("*.json") if p.is_file()]


def read_jsonl_file(path: Path) -> ParseResult:
    summary: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        if first_line:
            try:
                summary = json.loads(first_line)
            except Exception:
                summary = {}
        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                print(f"Warning: skipping malformed JSON line {line_num} in {path}")
                continue
    return ParseResult(summary=summary, rows=rows)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except Exception:
        return str(value)


def extract_error_type(row: Dict[str, Any]) -> str:
    if "error_type" in row and row["error_type"]:
        return str(row["error_type"])
    err = row.get("error")
    if isinstance(err, dict):
        et = err.get("error_type")
        if et:
            return str(et)
    return ""


def extract_error_text(row: Dict[str, Any]) -> str:
    err = row.get("error")
    if isinstance(err, dict):
        msg = err.get("error_message")
        if isinstance(msg, list):
            return "; ".join([normalize_text(x) for x in msg])
        if msg:
            return normalize_text(msg)
        return normalize_text(err)
    if isinstance(err, list):
        return "; ".join([normalize_text(x) for x in err])
    if isinstance(err, str):
        return err
    return ""


def infer_model_name(row: Dict[str, Any], path: Path) -> str:
    if row.get("model_name"):
        return str(row["model_name"])
    parts = path.parts
    for i, part in enumerate(parts):
        if part in CONDITION_LABELS:
            if i + 1 < len(parts):
                return parts[i + 1]
    return "unknown_model"


def infer_category(row: Dict[str, Any], path: Path) -> str:
    if row.get("test_category"):
        return str(row["test_category"])
    parts = path.parts
    for i, part in enumerate(parts):
        if part in CONDITION_LABELS:
            if i + 2 < len(parts):
                candidate = parts[i + 2 : -1]
                if candidate:
                    return "_".join(candidate)
    return "unknown_category"


def macro_bucket(error_type: str, error_text: str) -> str:
    blob = f"{error_type} {error_text}".lower()
    for pattern, bucket in MACRO_BUCKET_RULES:
        if re.search(pattern, blob):
            return bucket
    return "other"


def parse_json_like(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if not (text.startswith("{") or text.startswith("[")):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_expected_tools(possible_answer: Any) -> List[str]:
    tools: List[str] = []
    if isinstance(possible_answer, list):
        for item in possible_answer:
            if isinstance(item, dict):
                name = item.get("name") or item.get("function") or item.get("tool")
                if name:
                    tools.append(str(name))
            elif isinstance(item, str):
                tools.append(item)
    elif isinstance(possible_answer, dict):
        name = possible_answer.get("name") or possible_answer.get("function") or possible_answer.get("tool")
        if name:
            tools.append(str(name))
    elif isinstance(possible_answer, str):
        tools.append(possible_answer)
    return list({t.strip() for t in tools if t and isinstance(t, str)})


def compute_near_miss(
    model_result_decoded: Any, possible_answer: Any, error_type: str
) -> Dict[str, Any]:
    decoded_text = normalize_text(model_result_decoded).lower()
    expected_tools = [t.lower() for t in extract_expected_tools(possible_answer)]
    tool_match = None
    if expected_tools:
        tool_match = any(t in decoded_text for t in expected_tools)

    expected_args_keys: List[str] = []
    possible_obj = parse_json_like(possible_answer)
    if isinstance(possible_obj, dict):
        args = possible_obj.get("arguments") or possible_obj.get("args") or possible_obj.get("parameters")
        if isinstance(args, dict):
            expected_args_keys = list(args.keys())
    elif isinstance(possible_obj, list):
        for item in possible_obj:
            if isinstance(item, dict):
                args = item.get("arguments") or item.get("args") or item.get("parameters")
                if isinstance(args, dict):
                    expected_args_keys.extend(args.keys())

    predicted_args_keys: List[str] = []
    decoded_obj = parse_json_like(model_result_decoded)
    if isinstance(decoded_obj, dict):
        args = decoded_obj.get("arguments") or decoded_obj.get("args") or decoded_obj.get("parameters")
        if isinstance(args, dict):
            predicted_args_keys = list(args.keys())

    arg_overlap = None
    if expected_args_keys and predicted_args_keys:
        arg_overlap = len(set(expected_args_keys) & set(predicted_args_keys))

    near_miss = False
    if tool_match:
        if re.search(r"schema|json|format|parse|decode|enum|type|key|missing", error_type.lower()):
            near_miss = True

    return {
        "tool_match": tool_match,
        "arg_overlap": arg_overlap,
        "near_miss_tool_correct_schema_wrong": near_miss,
    }


def parse_condition_label(condition_folder: str) -> str:
    return CONDITION_LABELS.get(condition_folder, condition_folder)


def collect_error_rows(
    root: Path, conditions: List[str], model_filters: List[str], max_files: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    files_seen = 0

    for condition_folder, label in CONDITION_LABELS.items():
        if label not in conditions:
            continue
        json_files = find_json_files(root, condition_folder)
        if max_files is not None:
            json_files = json_files[: max_files - files_seen]
        for path in json_files:
            parsed = read_jsonl_file(path)
            summary = dict(parsed.summary)
            summary.update(
                {
                    "condition": label,
                    "filepath": str(path),
                    "model_name": None,
                    "category": None,
                }
            )
            summaries.append(summary)

            for row in parsed.rows:
                model_name = infer_model_name(row, path)
                if model_filters and not any(f in model_name for f in model_filters):
                    continue
                category = infer_category(row, path)
                error_type = extract_error_type(row)
                error_text = extract_error_text(row)
                near_miss = compute_near_miss(
                    row.get("model_result_decoded"), row.get("possible_answer"), error_type
                )
                rows.append(
                    {
                        "condition": label,
                        "model_name": model_name,
                        "category": category,
                        "id": row.get("id"),
                        "valid": bool(row.get("valid", False)),
                        "error_type": error_type,
                        "error_text": error_text,
                        "macro_bucket": macro_bucket(error_type, error_text),
                        "tool_match": near_miss["tool_match"],
                        "arg_overlap": near_miss["arg_overlap"],
                        "near_miss_tool_correct_schema_wrong": near_miss[
                            "near_miss_tool_correct_schema_wrong"
                        ],
                        "prompt": normalize_text(row.get("prompt"))
                        if row.get("prompt") is not None
                        else None,
                        "model_result_decoded": normalize_text(row.get("model_result_decoded"))
                        if row.get("model_result_decoded") is not None
                        else None,
                        "possible_answer": normalize_text(row.get("possible_answer"))
                        if row.get("possible_answer") is not None
                        else None,
                        "filepath": str(path),
                    }
                )
            files_seen += 1
            if max_files is not None and files_seen >= max_files:
                break
        if max_files is not None and files_seen >= max_files:
            break

    rows_df = pd.DataFrame(rows)
    summaries_df = pd.DataFrame(summaries)
    return rows_df, summaries_df


def build_paired_outcomes(error_rows: pd.DataFrame, conditions: List[str]) -> pd.DataFrame:
    if error_rows.empty:
        return pd.DataFrame()

    universe = (
        error_rows[["model_name", "category", "id"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    outcomes = universe.copy()
    for cond in conditions:
        cond_errors = (
            error_rows[error_rows["condition"] == cond][["model_name", "category", "id"]]
            .dropna()
            .drop_duplicates()
        )
        cond_errors["error_present"] = True
        merged = outcomes.merge(cond_errors, on=["model_name", "category", "id"], how="left")
        error_present = merged["error_present"].fillna(False).astype(bool)
        outcomes[cond] = ~error_present

    ordered = conditions
    def pattern(row: pd.Series) -> str:
        bits = ["1" if row[c] else "0" for c in ordered]
        return "".join(bits)

    outcomes["pattern"] = outcomes.apply(pattern, axis=1)

    if "OO" in ordered:
        for cond in ordered:
            if cond == "OO":
                continue
            outcomes[f"fixed_by_{cond}"] = (~outcomes["OO"]) & outcomes[cond]
            outcomes[f"broken_by_{cond}"] = outcomes["OO"] & (~outcomes[cond])

    outcomes["unchanged_invalid"] = outcomes[ordered].apply(lambda r: (~r).all(), axis=1)
    outcomes["unchanged_valid"] = outcomes[ordered].apply(lambda r: r.all(), axis=1)

    return outcomes


def summarize_tables(
    error_rows: pd.DataFrame, outcomes: pd.DataFrame, outdir: Path
) -> None:
    summary_dir = outdir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if not error_rows.empty:
        err_counts = (
            error_rows.groupby(["condition", "error_type"])
            .size()
            .reset_index(name="count")
        )
        err_counts.to_csv(summary_dir / "error_type_counts_by_condition.csv", index=False)

        macro_share = (
            error_rows.groupby(["condition", "macro_bucket"])
            .size()
            .reset_index(name="count")
        )
        macro_share.to_csv(summary_dir / "macro_bucket_share_by_condition.csv", index=False)

        if "OO" in error_rows["condition"].unique() and "AA" in error_rows["condition"].unique():
            oo = error_rows[error_rows["condition"] == "OO"]
            aa = error_rows[error_rows["condition"] == "AA"]
            oo_counts = oo["error_type"].value_counts().rename("oo_count")
            aa_counts = aa["error_type"].value_counts().rename("aa_count")
            delta = pd.concat([oo_counts, aa_counts], axis=1).fillna(0)
            delta["delta"] = delta["aa_count"] - delta["oo_count"]
            delta = delta.sort_values("delta", ascending=False).reset_index()
            delta.rename(columns={"index": "error_type"}, inplace=True)
            delta.to_csv(summary_dir / "top_error_types_delta_OO_to_AA.csv", index=False)

    if not outcomes.empty:
        if "pattern" in outcomes.columns:
            pattern_counts = (
                outcomes.groupby("pattern").size().reset_index(name="count")
            )
            pattern_counts.to_csv(summary_dir / "pattern_counts_16way.csv", index=False)

        if "OO" in outcomes.columns:
            trans_cols = [c for c in outcomes.columns if c.startswith("fixed_by_") or c.startswith("broken_by_")]
            if trans_cols:
                trans = outcomes.groupby(["category"])[trans_cols].sum().reset_index()
                trans.to_csv(summary_dir / "transitions_by_category_and_condition.csv", index=False)


def save_outputs(
    error_rows: pd.DataFrame, outcomes: pd.DataFrame, summaries: pd.DataFrame, outdir: Path
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        error_rows.to_parquet(outdir / "error_rows.parquet", index=False)
        outcomes.to_parquet(outdir / "paired_outcomes.parquet", index=False)
        summaries.to_parquet(outdir / "file_summaries.parquet", index=False)
    except ImportError as exc:
        print(f"Warning: parquet support unavailable ({exc}). Writing CSV fallback.")
        error_rows.to_csv(outdir / "error_rows.csv", index=False)
        outcomes.to_csv(outdir / "paired_outcomes.csv", index=False)
        summaries.to_csv(outdir / "file_summaries.csv", index=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    model_filters = [m.strip() for m in args.models.split(",") if m.strip()]

    error_rows, summaries = collect_error_rows(
        root=root,
        conditions=conditions,
        model_filters=model_filters,
        max_files=args.max_files,
    )
    outcomes = build_paired_outcomes(error_rows, conditions)
    save_outputs(error_rows, outcomes, summaries, outdir)
    summarize_tables(error_rows, outcomes, outdir)

    print("Parsed error rows:", len(error_rows))
    print("Paired outcomes:", len(outcomes))
    print("Output directory:", outdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Error analysis pipeline for BFCL-style evaluation outputs.

Central assumption: score files only log errors, so missing task ids imply
correct for that run. The universe of ids per run is taken from the matching
result file; paired transition analysis uses the intersection across conditions.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


SCORE_CONDITION_FOLDERS = {
    "score_desc_original_name_original": "OO",
    "score_desc_original_name_augmented": "OA",
    "score_desc_augmented_name_original": "AO",
    "score_desc_augmented_name_augmented": "AA",
}

RESULT_CONDITION_FOLDERS = {
    "result_desc_original_name_original": "OO",
    "result_desc_original_name_augmented": "OA",
    "result_desc_augmented_name_original": "AO",
    "result_desc_augmented_name_augmented": "AA",
}

CONDITION_FOLDER_LABELS = {}
CONDITION_FOLDER_LABELS.update(SCORE_CONDITION_FOLDERS)
CONDITION_FOLDER_LABELS.update(RESULT_CONDITION_FOLDERS)

DEFAULT_CONDITIONS = ["OO", "OA", "AO", "AA"]

MACRO_BUCKET_RULES: List[Tuple[str, str]] = [
    (r"(tool|function)[ _-]?select|wrong tool|no tool|tool mismatch", "tool_selection"),
    (r"schema|json|format|parse|decode|malformed", "schema_compliance"),
    (r"missing required|required field|missing argument|missing key", "missing_required"),
    (r"format|decode|deserialize|output parse|invalid json", "formatting_decoding"),
    (r"runtime|execution|timeout|exception|error during inference", "execution_runtime"),
]


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
    files = list(base.rglob("*.json"))
    files.extend(list(base.rglob("*.jsonl")))
    return [p for p in files if p.is_file()]


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        first = None
        for line in f:
            if line.strip():
                first = line
                break
        if first is None:
            return []
        if first.lstrip().startswith("["):
            rest = first + f.read()
            try:
                data = json.loads(rest)
            except Exception:
                print(f"Warning: failed to parse JSON array in {path}")
                return []
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return []
        second = None
        for line in f:
            if line.strip():
                second = line
                break
        if second is None:
            try:
                obj = json.loads(first)
            except Exception:
                print(f"Warning: failed to parse JSON object in {path}")
                return []
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
            return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"Warning: skipping malformed JSON line {line_num} in {path}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            elif isinstance(obj, list):
                rows.extend([x for x in obj if isinstance(x, dict)])
    return rows


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


def is_score_file(path: Path) -> bool:
    name = path.name
    return name.endswith("_score.json") or name.endswith("_score.jsonl")


def is_result_file(path: Path) -> bool:
    name = path.name
    return name.endswith("_result.json") or name.endswith("_result.jsonl")


def strip_suffix(filename: str) -> str:
    return re.sub(r"(_score|_result)\.jsonl?$", "", filename)


def extract_summary_and_rows(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary: Dict[str, Any] = {}
    if rows:
        first = rows[0]
        if isinstance(first, dict) and any(
            k in first for k in ("accuracy", "correct_count", "total_count")
        ):
            summary = first
            rows = rows[1:]
    return summary, rows


def load_error_file(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = read_json_or_jsonl(path)
    summary, rows = extract_summary_and_rows(rows)
    error_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "id" not in row:
            continue
        error_type = extract_error_type(row)
        error_text = extract_error_text(row)
        error_obj = row.get("error")
        sub_error_type = None
        model_output_item = None
        possible_answer_item = None
        if isinstance(error_obj, dict):
            sub_error_type = error_obj.get("sub_error_type") or error_obj.get("sub_error")
            model_output_item = error_obj.get("model_output_item")
            possible_answer_item = error_obj.get("possible_answer_item")
        elif isinstance(error_obj, list):
            for item in error_obj:
                if isinstance(item, dict):
                    if "sub_error_type" in item:
                        sub_error_type = item.get("sub_error_type")
                        model_output_item = item.get("model_output_item")
                        possible_answer_item = item.get("possible_answer_item")
                        break
                    for value in item.values():
                        if isinstance(value, dict) and "sub_error_type" in value:
                            sub_error_type = value.get("sub_error_type")
                            model_output_item = value.get("model_output_item")
                            possible_answer_item = value.get("possible_answer_item")
                            break
        error_rows.append(
            {
                "id": row.get("id"),
                "model_name": row.get("model_name"),
                "test_category": row.get("test_category"),
                "error_type": error_type,
                "error_text": error_text,
                "sub_error_type": sub_error_type,
                "model_output_item": normalize_text(model_output_item)
                if model_output_item is not None
                else None,
                "possible_answer_item": normalize_text(possible_answer_item)
                if possible_answer_item is not None
                else None,
                "raw_error": normalize_text(error_obj) if error_obj is not None else None,
                "prompt": normalize_text(row.get("prompt"))
                if row.get("prompt") is not None
                else None,
            }
        )
    return summary, error_rows


def load_result_ids(path: Path) -> Tuple[List[str], int]:
    rows = read_json_or_jsonl(path)
    ids: List[str] = []
    seen = set()
    dup_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = row.get("id")
        if not item_id:
            continue
        if item_id in seen:
            dup_count += 1
            continue
        seen.add(item_id)
        ids.append(item_id)
    if dup_count:
        print(f"Warning: {dup_count} duplicate ids in result file {path}")
    return ids, dup_count


def derive_path_metadata(path: Path, condition_root: Path) -> Tuple[str, str]:
    try:
        rel = path.relative_to(condition_root)
    except ValueError:
        rel = path
    parts = rel.parts
    model_name = parts[0] if parts else "unknown_model"
    category_parts = parts[1:-1]
    category = "_".join(category_parts) if category_parts else "unknown_category"
    return model_name, category


def find_matching_result_file(score_path: Path, score_root: Path, results_root: Path) -> Optional[Path]:
    target_name = re.sub(r"_score(\.jsonl?)$", r"_result\1", score_path.name)
    try:
        rel_parent = score_path.parent.relative_to(score_root)
    except ValueError:
        rel_parent = score_path.parent
    candidate = results_root / rel_parent / target_name
    if candidate.exists():
        return candidate

    matches = list(results_root.rglob(target_name))
    if not matches:
        print(f"Warning: no matching result file for {score_path}")
        return None
    if len(matches) == 1:
        return matches[0]

    def score_match(path: Path) -> Tuple[int, int]:
        try:
            rel = path.parent.relative_to(results_root)
        except ValueError:
            rel = path.parent
        rel_parts = rel.parts
        suffix_len = 0
        for a, b in zip(reversed(rel_parts), reversed(rel_parent.parts)):
            if a == b:
                suffix_len += 1
            else:
                break
        return (-suffix_len, len(path.parts))

    matches.sort(key=score_match)
    print(f"Warning: multiple result matches for {score_path}; using {matches[0]}")
    return matches[0]


def infer_model_name(row: Dict[str, Any], path: Path) -> str:
    if row.get("model_name"):
        return str(row["model_name"])
    parts = path.parts
    for i, part in enumerate(parts):
        if part in CONDITION_FOLDER_LABELS:
            if i + 1 < len(parts):
                return parts[i + 1]
    return "unknown_model"


def infer_category(row: Dict[str, Any], path: Path) -> str:
    if row.get("test_category"):
        return str(row["test_category"])
    parts = path.parts
    for i, part in enumerate(parts):
        if part in CONDITION_FOLDER_LABELS:
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
    return CONDITION_FOLDER_LABELS.get(condition_folder, condition_folder)


def collect_master_tables(
    root: Path, conditions: List[str], model_filters: List[str], max_files: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []

    label_to_result_folder = {v: k for k, v in RESULT_CONDITION_FOLDERS.items()}

    for score_folder, label in SCORE_CONDITION_FOLDERS.items():
        if label not in conditions:
            continue
        score_root = root / score_folder
        result_folder = label_to_result_folder.get(label)
        if result_folder is None:
            print(f"Warning: no result folder mapping for condition {label}")
            continue
        results_root = root / result_folder
        if not results_root.exists():
            print(f"Warning: results folder missing for condition {label}: {results_root}")
            continue

        score_files = [p for p in find_json_files(root, score_folder) if is_score_file(p)]
        if max_files is not None:
            score_files = score_files[:max_files]

        for score_path in score_files:
            result_path = find_matching_result_file(score_path, score_root, results_root)
            if result_path is None:
                continue
            error_summary, error_rows = load_error_file(score_path)
            result_ids, dup_count = load_result_ids(result_path)
            if not result_ids:
                print(f"Warning: no ids found in result file {result_path}")
                continue

            model_name, category = derive_path_metadata(score_path, score_root)
            if model_filters and not any(f in model_name for f in model_filters):
                continue
            run_name = strip_suffix(score_path.name)

            error_by_id = {row["id"]: row for row in error_rows if row.get("id")}
            error_ids = set(error_by_id.keys())
            total_ids = len(result_ids)
            error_count = len([i for i in result_ids if i in error_ids])
            correct_count = total_ids - error_count

            top_errors = (
                pd.Series([error_by_id[i]["error_type"] for i in result_ids if i in error_by_id])
                .value_counts()
                .head(5)
                .to_dict()
            )

            print(
                f"Run {run_name} ({label}/{category}): total={total_ids} errors={error_count} "
                f"correct={correct_count} top_errors={top_errors}"
            )

            if error_summary:
                summary_total = error_summary.get("total_count")
                summary_correct = error_summary.get("correct_count")
                if summary_total is not None and int(summary_total) != total_ids:
                    print(
                        f"Warning: total_count mismatch for {score_path} "
                        f"(summary {summary_total} vs computed {total_ids})"
                    )
                if summary_correct is not None and int(summary_correct) != correct_count:
                    print(
                        f"Warning: correct_count mismatch for {score_path} "
                        f"(summary {summary_correct} vs computed {correct_count})"
                    )

            run_summaries.append(
                {
                    "condition": label,
                    "model_name": model_name,
                    "category": category,
                    "run_name": run_name,
                    "score_file": str(score_path),
                    "result_file": str(result_path),
                    "total_ids": total_ids,
                    "error_ids": error_count,
                    "correct_ids": correct_count,
                    "summary_accuracy": error_summary.get("accuracy") if error_summary else None,
                    "summary_correct_count": error_summary.get("correct_count") if error_summary else None,
                    "summary_total_count": error_summary.get("total_count") if error_summary else None,
                    "duplicate_result_ids": dup_count,
                }
            )

            for item_id in result_ids:
                err = error_by_id.get(item_id)
                is_error = err is not None
                error_type = err.get("error_type") if err else None
                error_text = err.get("error_text") if err else None
                rows.append(
                    {
                        "condition": label,
                        "model_name": model_name,
                        "category": category,
                        "run_name": run_name,
                        "id": item_id,
                        "is_error": is_error,
                        "is_correct": not is_error,
                        "error_type": error_type,
                        "error_text": error_text,
                        "sub_error_type": err.get("sub_error_type") if err else None,
                        "model_output_item": err.get("model_output_item") if err else None,
                        "possible_answer_item": err.get("possible_answer_item") if err else None,
                        "raw_error": err.get("raw_error") if err else None,
                        "prompt": err.get("prompt") if err else None,
                        "macro_bucket": macro_bucket(error_type or "", error_text or ""),
                        "score_file": str(score_path),
                        "result_file": str(result_path),
                    }
                )

    rows_df = pd.DataFrame(rows)
    run_summary_df = pd.DataFrame(run_summaries)
    return rows_df, run_summary_df


def build_paired_outcomes(master_rows: pd.DataFrame, conditions: List[str]) -> pd.DataFrame:
    if master_rows.empty:
        return pd.DataFrame()

    outcomes_rows: List[Dict[str, Any]] = []
    grouped = master_rows.groupby(["model_name", "category"])
    for (model_name, category), group in grouped:
        cond_sets = {}
        cond_maps = {}
        for cond in conditions:
            subset = group[group["condition"] == cond][["id", "is_correct"]]
            if subset.empty:
                continue
            cond_sets[cond] = set(subset["id"])
            cond_maps[cond] = dict(zip(subset["id"], subset["is_correct"]))

        if len(cond_sets) < len(conditions):
            missing = [c for c in conditions if c not in cond_sets]
            print(f"Warning: missing conditions {missing} for {model_name}/{category}")
            continue

        ids_intersection = set.intersection(*cond_sets.values())
        for cond, id_set in cond_sets.items():
            if id_set != ids_intersection:
                print(
                    f"Warning: id mismatch in {model_name}/{category}/{cond} "
                    f"(total {len(id_set)} vs intersect {len(ids_intersection)})"
                )

        for item_id in ids_intersection:
            row = {
                "model_name": model_name,
                "category": category,
                "id": item_id,
            }
            for cond in conditions:
                row[cond] = bool(cond_maps[cond].get(item_id, False))
            outcomes_rows.append(row)

    outcomes = pd.DataFrame(outcomes_rows)
    if outcomes.empty:
        return outcomes

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
    master_rows: pd.DataFrame, outcomes: pd.DataFrame, outdir: Path, conditions: List[str]
) -> None:
    summary_dir = outdir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    if not outcomes.empty:
        pattern_counts = (
            outcomes.groupby(["model_name", "category", "pattern"])
            .size()
            .reset_index(name="count")
        )
        total_per_group = pattern_counts.groupby(["model_name", "category"])["count"].transform("sum")
        pattern_counts["rate"] = pattern_counts["count"] / total_per_group
        pattern_counts.to_csv(summary_dir / "pattern_counts_by_category.csv", index=False)

        trans_cols = [c for c in outcomes.columns if c.startswith("fixed_by_") or c.startswith("broken_by_")]
        if trans_cols:
            trans = outcomes.groupby(["model_name", "category"])[trans_cols].sum().reset_index()
            trans.to_csv(summary_dir / "transitions_by_category_and_condition.csv", index=False)

        category_rows: List[Dict[str, Any]] = []
        for (model_name, category), group in outcomes.groupby(["model_name", "category"]):
            total = len(group)
            row = {
                "model_name": model_name,
                "category": category,
                "total_tasks": total,
            }
            for cond in conditions:
                if cond in group.columns:
                    row[f"error_rate_{cond}"] = 1.0 - group[cond].mean()
            row["always_correct_rate"] = (group["pattern"] == "1111").mean()
            if "fixed_by_AA" in group.columns:
                row["fixed_by_AA_rate"] = group["fixed_by_AA"].mean()
            if "broken_by_AA" in group.columns:
                row["broken_by_AA_rate"] = group["broken_by_AA"].mean()
            category_rows.append(row)

        if category_rows:
            pd.DataFrame(category_rows).to_csv(
                summary_dir / "category_summary.csv", index=False
            )

    if not master_rows.empty:
        err_only = master_rows[master_rows["is_error"]]
        if not err_only.empty:
            err_counts = (
                err_only.groupby(["condition", "macro_bucket"])
                .size()
                .reset_index(name="count")
            )
            err_counts.to_csv(summary_dir / "macro_bucket_counts_by_condition.csv", index=False)

            error_type_counts = (
                err_only.groupby(["condition", "error_type"])
                .size()
                .reset_index(name="count")
            )
            error_type_counts.to_csv(summary_dir / "error_type_counts_by_condition.csv", index=False)


def save_outputs(
    master_rows: pd.DataFrame, outcomes: pd.DataFrame, run_summaries: pd.DataFrame, outdir: Path
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        master_rows.to_parquet(outdir / "master_rows.parquet", index=False)
        outcomes.to_parquet(outdir / "paired_outcomes.parquet", index=False)
        run_summaries.to_parquet(outdir / "run_summaries.parquet", index=False)
    except ImportError as exc:
        print(f"Warning: parquet support unavailable ({exc}). Writing CSV fallback.")
        master_rows.to_csv(outdir / "master_rows.csv", index=False)
        outcomes.to_csv(outdir / "paired_outcomes.csv", index=False)
        run_summaries.to_csv(outdir / "run_summaries.csv", index=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    model_filters = [m.strip() for m in args.models.split(",") if m.strip()]

    master_rows, run_summaries = collect_master_tables(
        root=root,
        conditions=conditions,
        model_filters=model_filters,
        max_files=args.max_files,
    )
    outcomes = build_paired_outcomes(master_rows, conditions)
    save_outputs(master_rows, outcomes, run_summaries, outdir)
    summarize_tables(master_rows, outcomes, outdir, conditions)

    print("Master rows:", len(master_rows))
    print("Paired outcomes:", len(outcomes))
    print("Output directory:", outdir)


if __name__ == "__main__":
    main()

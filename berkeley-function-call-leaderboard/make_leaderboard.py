"""
README:
Run: python make_leaderboard.py
Output: leaderboard.html at the project root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bfcl_eval.eval_checker.eval_runner_helper import generate_leaderboard_csv

try:  # Optional dependency for image export.
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional
    go = None


CONDITIONS = {
    "OO": "score_desc_original_name_original",
    "OA": "score_desc_original_name_augmented",
    "AO": "score_desc_augmented_name_original",
    "AA": "score_desc_augmented_name_augmented",
}
TOP_ERROR_SHIFT_K = 5

ERROR_TYPES_ORDER = [
    "WRONG_TOOL",
    "HALLUCINATED_TOOL",
    "SHOULD_ABSTAIN_BUT_CALLED",
    "SHOULD_CALL_BUT_ABSTAINED",
    "MISSING_REQUIRED_ARGS",
    "EXTRA_ARGS",
    "TYPE_MISMATCH",
    "VALUE_FORMAT_ERROR",
    "WRONG_ARGS",
    "INVALID_JSON_OR_SCHEMA",
    "OTHER",
]
KNOWN_ERROR_TYPES = [err for err in ERROR_TYPES_ORDER if err != "OTHER"]

REQUIRED_TEST_CATEGORIES = {
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "web_search_base",
    "web_search_no_snippet",
    "memory_kv",
    "memory_vector",
    "memory_rec_sum",
}

TEST_CATEGORY_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "data_non_live": {
        "Non-Live Overall Acc": [
            "simple_python",
            "simple_java",
            "simple_javascript",
            "multiple",
            "parallel",
            "parallel_multiple",
        ],
        "AST Summary": [
            "simple_python",
            "simple_java",
            "simple_javascript",
            "multiple",
            "parallel",
            "parallel_multiple",
        ],
        "Simple AST": ["simple_python", "simple_java", "simple_javascript"],
        "Python Simple AST": ["simple_python"],
        "Java Simple AST": ["simple_java"],
        "JavaScript Simple AST": ["simple_javascript"],
        "Multiple AST": ["multiple"],
        "Parallel AST": ["parallel"],
        "Parallel Multiple AST": ["parallel_multiple"],
        "Irrelevance Detection": ["irrelevance"],
    },
    "data_live": {
        "Live Overall Acc": [
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
        ],
        "AST Summary": [
            "live_simple",
            "live_multiple",
            "live_parallel",
            "live_parallel_multiple",
        ],
        "Python Simple AST": ["live_simple"],
        "Python Multiple AST": ["live_multiple"],
        "Python Parallel AST": ["live_parallel"],
        "Python Parallel Multiple AST": ["live_parallel_multiple"],
        "Irrelevance Detection": ["live_irrelevance"],
        "Relevance Detection": ["live_relevance"],
    },
    "data_multi_turn": {
        "Multi Turn Overall Acc": [
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        ],
        "Base": ["multi_turn_base"],
        "Miss Func": ["multi_turn_miss_func"],
        "Miss Param": ["multi_turn_miss_param"],
        "Long Context": ["multi_turn_long_context"],
    },
    "data_agentic": {
        "Agentic Overall Acc": [
            "web_search_base",
            "web_search_no_snippet",
            "memory_kv",
            "memory_vector",
            "memory_rec_sum",
        ],
        "Web Search Summary": ["web_search_base", "web_search_no_snippet"],
        "Web Search Base": ["web_search_base"],
        "Web Search No Snippet": ["web_search_no_snippet"],
        "Memory Summary": ["memory_kv", "memory_vector", "memory_rec_sum"],
        "Memory KV": ["memory_kv"],
        "Memory Vector": ["memory_vector"],
        "Memory Recursive Summarization": ["memory_rec_sum"],
    },
    "data_agentic-web": {
        "Web Search Summary": ["web_search_base", "web_search_no_snippet"],
        "Web Search Base": ["web_search_base"],
        "Web Search No Snippet": ["web_search_no_snippet"],
    },
    "data_agentic-memory": {
        "Memory Summary": ["memory_kv", "memory_vector", "memory_rec_sum"],
        "Memory KV": ["memory_kv"],
        "Memory Vector": ["memory_vector"],
        "Memory Recursive Summarization": ["memory_rec_sum"],
    },
}


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def load_json_cache(path: Path) -> Optional[object]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"failed to read cache {path}: {exc}")
        return None


def write_json_cache(path: Path, payload: object) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"failed to write cache {path}: {exc}")


def read_json_or_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        try:
            data = json.loads(text)
        except Exception:
            return []
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        return []
    rows: List[Dict[str, object]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        elif isinstance(obj, list):
            rows.extend([row for row in obj if isinstance(row, dict)])
    return rows


def extract_summary_and_rows(
    rows: List[Dict[str, object]]
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    summary: Dict[str, object] = {}
    if rows:
        first = rows[0]
        if isinstance(first, dict) and any(
            key in first for key in ("accuracy", "correct_count", "total_count")
        ):
            summary = first
            rows = rows[1:]
    return summary, rows


def parse_test_category_from_filename(filename: str) -> Optional[str]:
    match = re.match(r"BFCL_v4_(.+?)_score\.jsonl?$", filename)
    if not match:
        return None
    return match.group(1)


def extract_error_fields(row: Dict[str, object]) -> Tuple[str, str, str]:
    error_type = ""
    error_text = ""
    sub_error = ""
    if "error_type" in row and row["error_type"]:
        error_type = str(row["error_type"])
    error_obj = row.get("error")
    if isinstance(error_obj, dict):
        if not error_type and error_obj.get("error_type"):
            error_type = str(error_obj.get("error_type"))
        if error_obj.get("sub_error_type"):
            sub_error = str(error_obj.get("sub_error_type"))
        msg = error_obj.get("error_message")
        if isinstance(msg, list):
            error_text = " ".join([str(m) for m in msg if m])
        elif msg:
            error_text = str(msg)
    elif isinstance(error_obj, list):
        error_text = " ".join([str(item) for item in error_obj if item])
    elif isinstance(error_obj, str):
        error_text = error_obj
    if not error_text and row.get("error"):
        error_text = str(row.get("error"))
    return error_type, sub_error, error_text


def infer_primary_error_type(
    test_category: str, error_type: str, sub_error: str, error_text: str
) -> str:
    combined = " ".join([error_type or "", sub_error or "", error_text or ""]).lower()
    if any(token in test_category for token in ("irrelevance", "live_irrelevance")):
        return "SHOULD_ABSTAIN_BUT_CALLED"

    if re.search(r"(invalid json|schema|parse|decode|malformed)", combined):
        return "INVALID_JSON_OR_SCHEMA"
    if re.search(r"(tool_not_found|unknown tool|not in tool catalog|hallucinated tool)", combined):
        return "HALLUCINATED_TOOL"
    if re.search(r"(wrong tool|tool mismatch|tool_select|function select|tool selection)", combined):
        return "WRONG_TOOL"
    if re.search(r"(no tool call|missing tool call|no function call)", combined):
        return "SHOULD_CALL_BUT_ABSTAINED"
    if re.search(r"(missing required|required field|missing argument|missing key)", combined):
        return "MISSING_REQUIRED_ARGS"
    if re.search(r"(extra arguments|unexpected argument|extra key|unsupported argument)", combined):
        return "EXTRA_ARGS"
    if re.search(r"(type mismatch|invalid type|expected .*? type)", combined):
        return "TYPE_MISMATCH"
    if re.search(r"(format|invalid value|invalid enum|date format)", combined):
        return "VALUE_FORMAT_ERROR"
    if re.search(r"(answer_not_found|incorrect|wrong value|mismatch)", combined):
        return "WRONG_ARGS"
    return "OTHER"


ERROR_TAG_RULES = [
    ("INVALID_JSON_OR_SCHEMA", r"(invalid json|schema|parse|decode|malformed)"),
    ("HALLUCINATED_TOOL", r"(tool_not_found|unknown tool|not in tool catalog|hallucinated tool)"),
    ("WRONG_TOOL", r"(wrong tool|tool mismatch|tool_select|function select|tool selection)"),
    ("SHOULD_CALL_BUT_ABSTAINED", r"(no tool call|missing tool call|no function call)"),
    ("MISSING_REQUIRED_ARGS", r"(missing required|required field|missing argument|missing key)"),
    ("EXTRA_ARGS", r"(extra arguments|unexpected argument|extra key|unsupported argument)"),
    ("TYPE_MISMATCH", r"(type mismatch|invalid type|expected .*? type)"),
    ("VALUE_FORMAT_ERROR", r"(format|invalid value|invalid enum|date format)"),
    ("WRONG_ARGS", r"(answer_not_found|incorrect|wrong value|mismatch)"),
]


def infer_error_tags(
    test_category: str, error_type: str, sub_error: str, error_text: str
) -> List[str]:
    if any(token in test_category for token in ("irrelevance", "live_irrelevance")):
        return ["SHOULD_ABSTAIN_BUT_CALLED"]
    tags = set()
    combined = " ".join([error_type or "", sub_error or "", error_text or ""]).lower()
    for tag, pattern in ERROR_TAG_RULES:
        if re.search(pattern, combined):
            tags.add(tag)
    upper_error = str(error_type or "").upper()
    upper_sub = str(sub_error or "").upper()
    if upper_error in KNOWN_ERROR_TYPES:
        tags.add(upper_error)
    if upper_sub in KNOWN_ERROR_TYPES:
        tags.add(upper_sub)
    return sorted(tags)


def get_row_id(row: Dict[str, object], fallback: str) -> str:
    for key in ("id", "task_id", "example_id"):
        value = row.get(key)
        if value:
            return str(value)
    prompt = row.get("prompt")
    if isinstance(prompt, dict):
        prompt_id = prompt.get("id")
        if prompt_id:
            return str(prompt_id)
    return fallback


def is_row_incorrect(row: Dict[str, object]) -> bool:
    if "valid" in row:
        return not bool(row.get("valid"))
    if "correct" in row:
        return not bool(row.get("correct"))
    if "is_correct" in row:
        return not bool(row.get("is_correct"))
    if row.get("error") or row.get("error_type"):
        return True
    return True


def build_error_summary(
    root: Path,
    data_map: Dict[str, Dict[str, pd.DataFrame]],
    sanity_check: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, List[str]]]]:
    records: List[Dict[str, object]] = []
    display_name_map: Dict[str, str] = {}
    sanity_printed = False
    for condition in CONDITIONS:
        df = data_map.get(condition, {}).get("data_overall.csv")
        if df is None or "Model" not in df.columns:
            continue
        for model in df["Model"].tolist():
            normalized = re.sub(r"[^a-z0-9]+", "", str(model).lower())
            display_name_map[normalized] = str(model)
    for condition, folder in CONDITIONS.items():
        condition_path = root / folder
        if not condition_path.exists():
            continue
        for path in condition_path.rglob("*_score.json*"):
            test_category = parse_test_category_from_filename(path.name)
            if not test_category:
                continue
            rel_parts = path.relative_to(condition_path).parts
            model = rel_parts[0] if rel_parts else ""
            normalized_model = re.sub(r"[^a-z0-9]+", "", str(model).lower())
            model_display = display_name_map.get(normalized_model, str(model))
            rows = read_json_or_jsonl(path)
            summary, error_rows = extract_summary_and_rows(rows)
            total_count = summary.get("total_count")
            correct_count = summary.get("correct_count")
            if isinstance(total_count, str):
                try:
                    total_count = int(total_count)
                except ValueError:
                    total_count = None
            if isinstance(correct_count, str):
                try:
                    correct_count = int(correct_count)
                except ValueError:
                    correct_count = None
            if total_count is None:
                total_count = len(error_rows)
            incorrect_count = (
                total_count - correct_count
                if isinstance(correct_count, int)
                else len(error_rows)
            )
            tag_ids: Dict[str, set] = {tag: set() for tag in KNOWN_ERROR_TYPES}
            wrong_ids: set = set()
            union_known: set = set()
            for idx, row in enumerate(error_rows):
                if not isinstance(row, dict):
                    continue
                if not is_row_incorrect(row):
                    continue
                row_id = get_row_id(row, f"{path.name}:{idx}")
                wrong_ids.add(row_id)
                error_type, sub_error, error_text = extract_error_fields(row)
                tags = infer_error_tags(test_category, error_type, sub_error, error_text)
                for tag in tags:
                    if tag in tag_ids:
                        tag_ids[tag].add(row_id)
                        union_known.add(row_id)
            other_ids = wrong_ids - union_known
            error_counts = Counter({tag: len(ids) for tag, ids in tag_ids.items() if ids})
            if other_ids:
                error_counts["OTHER"] = len(other_ids)
            if wrong_ids:
                incorrect_count = len(wrong_ids)

            if sanity_check:
                known_total = sum(len(ids) for ids in tag_ids.values())
                if wrong_ids and not union_known:
                    print(
                        f"[sanity] no known tags: {model_display} {condition} {test_category} "
                        f"wrong={len(wrong_ids)}"
                    )
                if other_ids and known_total > 0 and len(other_ids) == len(wrong_ids):
                    print(
                        f"[sanity] OTHER equals wrong_ids with known tags present: "
                        f"{model_display} {condition} {test_category} other={len(other_ids)}"
                    )
                if (
                    not sanity_printed
                    and test_category == "memory_vector"
                    and wrong_ids
                ):
                    print(
                        "[sanity] memory_vector counts:",
                        {
                            "model": model_display,
                            "condition": condition,
                            "total": total_count,
                            "wrong": len(wrong_ids),
                            "known_union": len(union_known),
                            "other": len(other_ids),
                            "tags": {k: len(v) for k, v in tag_ids.items() if v},
                        },
                    )
                    sanity_printed = True
            records.append(
                {
                    "condition": condition,
                    "model": model_display,
                    "test_category": test_category,
                    "total": int(total_count),
                    "incorrect": int(incorrect_count),
                    "errors": dict(error_counts),
                }
            )
    return records, TEST_CATEGORY_GROUPS

def read_generic_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"failed to read {csv_path}: {exc}")
        return None
    if "Model" not in df.columns:
        warn(f"missing Model column in {csv_path}")
        return None
    df = df.copy()
    df["Model"] = df["Model"].astype(str)
    for col in df.columns:
        if col in ("Model", "Rank"):
            continue
        series = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
        df[col] = pd.to_numeric(series, errors="coerce")
    df = df.dropna(subset=["Model"], how="any")
    return df


def parse_model_run_name(model_run: str) -> Tuple[str, int]:
    match = re.match(r"^(?P<base>.+)_(?P<run>\d+)$", model_run)
    if match:
        return match.group("base"), int(match.group("run"))
    return model_run, 0


def coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def coerce_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def extract_score_summary(path: Path) -> Optional[Tuple[float, int]]:
    rows = read_json_or_jsonl(path)
    summary, _ = extract_summary_and_rows(rows)
    accuracy = coerce_float(summary.get("accuracy"))
    total_count = coerce_int(summary.get("total_count"))
    if accuracy is None or total_count is None:
        return None
    return accuracy, total_count


def collect_run_scores(run_dir: Path) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for path in run_dir.rglob("BFCL_v4_*_score.json*"):
        test_category = parse_test_category_from_filename(path.name)
        if not test_category:
            continue
        summary = extract_score_summary(path)
        if not summary:
            warn(f"missing summary in score file: {path}")
            continue
        accuracy, total_count = summary
        scores[test_category] = {"accuracy": accuracy, "total_count": total_count}
    return scores


def compute_run_overall(
    condition: str,
    model_name: str,
    scores: Dict[str, Dict[str, float]],
    temp_root: Path,
) -> Tuple[float, Optional[float], str]:
    with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
        output_path = Path(temp_dir)
        generate_leaderboard_csv({model_name: scores}, output_path)
        overall_df = read_generic_csv(output_path / "data_overall.csv")
        if overall_df is None or overall_df.empty:
            raise ValueError(f"failed to generate data_overall.csv for {model_name}")
        row = overall_df.iloc[0]
        model_display = str(row.get("Model"))
        overall_acc = coerce_float(row.get("Overall Acc"))
        if overall_acc is None:
            raise ValueError(f"missing Overall Acc for {model_name} in {output_path}")
        excl_map = compute_overall_excl_web({condition: {"data_overall.csv": overall_df}})
        excl_value = excl_map.get(condition, {}).get(model_display)
        return overall_acc, excl_value, model_display


def summarize_values(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    cleaned = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not cleaned:
        return None, None, None
    return sum(cleaned) / len(cleaned), min(cleaned), max(cleaned)


def build_aggregated_main_rows(
    root: Path,
    data_map: Dict[str, Dict[str, pd.DataFrame]],
) -> List[Dict[str, object]]:
    fallback_overall: Dict[str, pd.Series] = {}
    for condition in CONDITIONS:
        df = data_map.get(condition, {}).get("data_overall.csv")
        if df is None or "Overall Acc" not in df.columns:
            continue
        fallback_overall[condition] = df.set_index("Model")["Overall Acc"]

    fallback_excl = compute_overall_excl_web(data_map)

    legacy_scores: Dict[Tuple[str, str], Dict[str, List[Optional[float]]]] = {}
    legacy_root = root / "score"
    temp_root = root / "analysis_out"
    temp_root.mkdir(parents=True, exist_ok=True)

    if legacy_root.exists():
        for condition in CONDITIONS:
            condition_root = legacy_root / condition
            if not condition_root.exists():
                continue
            for run_dir in condition_root.iterdir():
                if not run_dir.is_dir():
                    continue
                model_base, run_id = parse_model_run_name(run_dir.name)
                scores = collect_run_scores(run_dir)
                missing = sorted(REQUIRED_TEST_CATEGORIES.difference(scores.keys()))
                if missing:
                    warn(
                        "skipping legacy run with missing categories "
                        f"{run_dir} (run_id={run_id}): {', '.join(missing)}"
                    )
                    continue
                try:
                    overall_acc, excl_acc, model_display = compute_run_overall(
                        condition, model_base, scores, temp_root
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    warn(f"failed to compute overall for {run_dir}: {exc}")
                    continue
                key = (condition, model_display)
                entry = legacy_scores.setdefault(key, {"overall": [], "excl": []})
                entry["overall"].append(overall_acc)
                entry["excl"].append(excl_acc)

    models: set[str] = set()
    for (condition, model_display) in legacy_scores.keys():
        models.add(model_display)
    for condition, series in fallback_overall.items():
        models.update(series.index.tolist())

    rows: List[Dict[str, object]] = []
    for model in sorted(models):
        row: Dict[str, object] = {"model": str(model)}
        for condition in CONDITIONS:
            key = (condition, model)
            if key in legacy_scores and legacy_scores[key]["overall"]:
                overall_mean, overall_min, overall_max = summarize_values(
                    legacy_scores[key]["overall"]
                )
                excl_mean, excl_min, excl_max = summarize_values(legacy_scores[key]["excl"])
            else:
                fallback_series = fallback_overall.get(condition)
                if fallback_series is not None and model in fallback_series.index:
                    raw_value = fallback_series.get(model)
                    fallback_value = None if pd.isna(raw_value) else float(raw_value)
                else:
                    fallback_value = None
                overall_mean = overall_min = overall_max = fallback_value
                fallback_excl_value = fallback_excl.get(condition, {}).get(model)
                if fallback_excl_value is None or pd.isna(fallback_excl_value):
                    excl_mean = excl_min = excl_max = None
                else:
                    excl_mean = excl_min = excl_max = float(fallback_excl_value)

            row[condition] = overall_mean
            row[f"{condition}_min"] = overall_min
            row[f"{condition}_max"] = overall_max
            row[f"{condition}_excl_web"] = excl_mean
            row[f"{condition}_excl_web_min"] = excl_min
            row[f"{condition}_excl_web_max"] = excl_max
        rows.append(row)
    return rows


def load_all_condition_data(root: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_map: Dict[str, Dict[str, pd.DataFrame]] = {}
    for condition, folder in CONDITIONS.items():
        folder_path = root / folder
        if not folder_path.exists():
            warn(f"missing folder for {condition}: {folder_path}")
            continue
        condition_map: Dict[str, pd.DataFrame] = {}
        for csv_path in folder_path.glob("data_*.csv"):
            df = read_generic_csv(csv_path)
            if df is None:
                continue
            condition_map[csv_path.name] = df
        if not condition_map:
            warn(f"no CSVs found for {condition} in {folder_path}")
        data_map[condition] = condition_map
    return data_map


def build_wide_table(data_map: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[pd.DataFrame, List[str]]:
    condition_frames: Dict[str, pd.DataFrame] = {}
    missing_conditions: List[str] = []
    for condition in CONDITIONS:
        df = data_map.get(condition, {}).get("data_overall.csv")
        if df is None:
            missing_conditions.append(condition)
            continue
        if "Overall Acc" not in df.columns:
            warn(f"missing Overall Acc in data_overall.csv for {condition}")
            missing_conditions.append(condition)
            continue
        condition_frames[condition] = df

    models: List[str] = []
    if condition_frames:
        model_sets = [set(df["Model"].tolist()) for df in condition_frames.values()]
        models = sorted(set().union(*model_sets))
    else:
        warn("no condition CSVs loaded; output will be empty")

    wide = pd.DataFrame(index=models)
    for condition, df in condition_frames.items():
        mapping = df.set_index("Model")["Overall Acc"]
        wide[f"{condition}_acc"] = mapping.reindex(models)

    return wide, missing_conditions


def compute_overall_excl_web(
    data_map: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, Dict[str, float]]:
    overall_excl: Dict[str, Dict[str, float]] = {}
    required_cols = [
        "Non-Live AST Acc",
        "Live Acc",
        "Irrelevance Detection",
        "Multi Turn Acc",
        "Memory Acc",
    ]
    weights = [10, 10, 10, 30, 40]
    for condition in CONDITIONS:
        df = data_map.get(condition, {}).get("data_overall.csv")
        if df is None:
            continue
        if any(col not in df.columns for col in required_cols):
            warn(f"missing columns for overall excl web in {condition}")
            continue
        condition_map: Dict[str, float] = {}
        for _, row in df.iterrows():
            model = str(row.get("Model"))
            values = []
            for col in required_cols:
                value = row.get(col)
                if pd.isna(value):
                    value = 0.0
                values.append(float(value))
            weighted_sum = sum(val * weight for val, weight in zip(values, weights))
            condition_map[model] = weighted_sum / sum(weights)
        overall_excl[condition] = condition_map
    return overall_excl


def build_overall_components(
    data_map: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    components: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for condition in CONDITIONS:
        df = data_map.get(condition, {}).get("data_overall.csv")
        if df is None:
            continue
        condition_map: Dict[str, Dict[str, Optional[float]]] = {}
        for _, row in df.iterrows():
            model = str(row.get("Model"))
            non_live = row.get("Non-Live AST Acc")
            live = row.get("Live Acc")
            irrelevance = row.get("Irrelevance Detection")
            multi_turn = row.get("Multi Turn Acc")
            web = row.get("Web Search Acc")
            memory = row.get("Memory Acc")
            agentic = None
            if pd.notna(web) and pd.notna(memory):
                agentic = (float(web) + float(memory)) / 2.0
            condition_map[model] = {
                "non_live": float(non_live) if pd.notna(non_live) else None,
                "live": float(live) if pd.notna(live) else None,
                "irrelevance": float(irrelevance) if pd.notna(irrelevance) else None,
                "multi_turn": float(multi_turn) if pd.notna(multi_turn) else None,
                "web_search": float(web) if pd.notna(web) else None,
                "memory": float(memory) if pd.notna(memory) else None,
                "agentic": agentic,
            }
        components[condition] = condition_map
    return components


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "model"


def get_metric_display_label(
    metric: str,
    table_title: str,
    exclude_web_search: bool,
) -> str:
    if (
        exclude_web_search
        and "agentic" in table_title.lower()
        and "agentic overall" in str(metric).lower()
    ):
        return "Agentic (memory-only)"
    return str(metric or "")


def get_agentic_memory_summary(
    summary_tables: List[Dict[str, object]],
    model: str,
    condition: str,
) -> Optional[float]:
    table = next(
        (item for item in summary_tables if item.get("key") == "data_agentic-memory"),
        None,
    )
    if not table:
        return None
    for row in table.get("rows", []):
        if row.get("model") != model:
            continue
        metric_values = row.get("metrics", {}).get("Memory Summary", {})
        value = metric_values.get(condition)
        return float(value) if value is not None else None
    return None


def normalize_acc_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val / 100.0 if val > 1.5 else val


def compute_delta_pp(combo: Optional[float], baseline: Optional[float]) -> Optional[float]:
    combo_val = normalize_acc_value(combo)
    base_val = normalize_acc_value(baseline)
    if combo_val is None or base_val is None:
        return None
    return (combo_val - base_val) * 100.0


def find_table_by_key(
    tables: List[Dict[str, object]], key: str
) -> Optional[Dict[str, object]]:
    return next((table for table in tables if table.get("key") == key), None)


def print_overall_delta_diagnostics(
    rows: List[Dict[str, object]],
    overall_components: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    category_tables_full: List[Dict[str, object]],
    summary_tables: List[Dict[str, object]],
    overall_excl_web: Optional[Dict[str, Dict[str, float]]] = None,
    combos: Tuple[str, ...] = ("OA", "AO", "AA"),
) -> None:
    components_full = [
        ("non_live", "non_live", 0.1),
        ("live", "live", 0.1),
        ("irrelevance", "irrelevance", 0.1),
        ("multi_turn", "multi_turn", 0.3),
        ("agentic", "agentic", 0.4),
    ]
    components_memory = [
        ("non_live", "non_live", 0.1),
        ("live", "live", 0.1),
        ("irrelevance", "irrelevance", 0.1),
        ("multi_turn", "multi_turn", 0.3),
        ("memory", "memory", 0.4),
    ]
    model_rows = {str(row.get("model")): row for row in rows}
    agentic_table = find_table_by_key(category_tables_full, "data_agentic")
    if not agentic_table:
        agentic_table = find_table_by_key(category_tables_full, "data_agentic-memory")

    def print_mode(
        title: str,
        components: List[Tuple[str, str, float]],
        overall_map: Optional[Dict[str, Dict[str, float]]],
    ) -> None:
        print(f"\n[diagnose] {title} (OO baseline)\n")
        for model, row in model_rows.items():
            print(f"MODEL: {model}")
            for combo in combos:
                if overall_map:
                    base_val = overall_map.get("OO", {}).get(model)
                    combo_val = overall_map.get(combo, {}).get(model)
                    if base_val is None or combo_val is None:
                        print(f"  Combo {combo} vs OO: missing overall values")
                        continue
                    overall_delta_pp = compute_delta_pp(combo_val, base_val)
                else:
                    overall_base = normalize_acc_value(row.get("OO"))
                    overall_combo = normalize_acc_value(row.get(combo))
                    if overall_base is None or overall_combo is None:
                        print(f"  Combo {combo} vs OO: missing overall values")
                        continue
                    overall_delta_pp = (overall_combo - overall_base) * 100.0
                print(f"  Combo {combo} vs OO:")
                print(f"    Overall Δ: {overall_delta_pp:+.2f}pp")
                contrib_sum = 0.0
                category_lines: List[Tuple[str, Optional[float], float, Optional[float]]] = []
                for key, label, weight in components:
                    base = overall_components.get("OO", {}).get(model, {}).get(key)
                    combo_component = overall_components.get(combo, {}).get(model, {}).get(key)
                    delta_pp = compute_delta_pp(combo_component, base)
                    contrib_pp = delta_pp * weight if delta_pp is not None else None
                    if contrib_pp is not None:
                        contrib_sum += contrib_pp
                    category_lines.append((label, delta_pp, weight, contrib_pp))
                print("    Category deltas (Δpp | weight | contrib_pp):")
                for label, delta_pp, weight, contrib_pp in category_lines:
                    if delta_pp is None or contrib_pp is None:
                        print(f"      {label:<12}: missing")
                        continue
                    print(f"      {label:<12}: {delta_pp:+.2f} | {weight:.2f} | {contrib_pp:+.2f}")

                valid_raw = [
                    (label, delta_pp)
                    for label, delta_pp, _, _ in category_lines
                    if delta_pp is not None
                ]
                valid_contrib = [
                    (label, contrib_pp)
                    for label, _, _, contrib_pp in category_lines
                    if contrib_pp is not None
                ]
                raw_winner = max(valid_raw, key=lambda item: item[1]) if valid_raw else None
                contrib_winner = (
                    max(valid_contrib, key=lambda item: item[1]) if valid_contrib else None
                )
                if raw_winner:
                    print(f"    Biggest raw delta: {raw_winner[0]} ({raw_winner[1]:+.2f}pp)")
                if contrib_winner:
                    print(f"    Biggest contrib:   {contrib_winner[0]} ({contrib_winner[1]:+.2f}pp)")
                diff = contrib_sum - overall_delta_pp
                print(
                    f"    Note: contrib sum = {contrib_sum:+.2f}pp vs overall Δ = {overall_delta_pp:+.2f}pp (diff={diff:+.2f}pp)"
                )

                if raw_winner and contrib_winner:
                    raw_memory = raw_winner[0] in ("agentic", "memory")
                    contrib_memory = contrib_winner[0] in ("agentic", "memory")
                    if raw_memory and contrib_memory:
                        note = "both"
                    elif raw_memory:
                        note = "raw_delta"
                    elif contrib_memory:
                        note = "weighting"
                    else:
                        note = "neither"
                    print(f"    Memory dominates due to: {note}")

                if agentic_table:
                    sub_rows = [
                        r for r in agentic_table.get("rows", []) if r.get("model") == model
                    ]
                    if sub_rows:
                        metric_values = sub_rows[0].get("metrics", {})
                        memory_metrics = [
                            metric
                            for metric in metric_values.keys()
                            if "memory" in str(metric).lower()
                        ]
                        if memory_metrics:
                            deltas = []
                            for metric in memory_metrics:
                                base_val = metric_values.get(metric, {}).get("OO")
                                combo_val = metric_values.get(metric, {}).get(combo)
                                delta_pp = compute_delta_pp(combo_val, base_val)
                                if delta_pp is not None:
                                    deltas.append((metric, delta_pp))
                            if deltas:
                                deltas.sort(key=lambda item: abs(item[1]), reverse=True)
                                print("    Agentic subcats Δpp (top movers):")
                                for metric, delta_pp in deltas[:5]:
                                    print(f"      {metric}: {delta_pp:+.2f}")
            print("")

    print_mode("Overall delta diagnostics (includes web_search)", components_full, None)
    if overall_excl_web:
        print_mode("Overall delta diagnostics (excludes web_search)", components_memory, overall_excl_web)

def get_metric_display_value(
    row: Dict[str, object],
    metric: str,
    table_title: str,
    condition: str,
    exclude_web_search: bool,
    summary_tables: List[Dict[str, object]],
) -> Optional[float]:
    if (
        exclude_web_search
        and "agentic" in table_title.lower()
        and "agentic overall" in str(metric).lower()
    ):
        return get_agentic_memory_summary(summary_tables, str(row.get("model")), condition)
    metric_values = row.get("metrics", {}).get(metric, {})
    value = metric_values.get(condition)
    return float(value) if value is not None else None


def is_web_search_label(label: str) -> bool:
    lowered = label.lower()
    tokens = ["web search", "websearch", "web_search", "yandex", "fetch", "browser", "search"]
    return any(token in lowered for token in tokens)


def build_thesis_scatter_facet_specs(
    category_tables: List[Dict[str, object]],
    summary_tables: List[Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, int], List[str]]:
    model_points: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    excluded_labels: set[str] = set()
    for table in category_tables:
        title = str(table.get("title") or "")
        metrics = table.get("metrics", [])
        for row in table.get("rows", []):
            model = str(row.get("model") or "")
            if not model:
                continue
            model_points.setdefault(model, {"OA": [], "AO": [], "AA": []})
            for metric in metrics:
                metric_label = get_metric_display_label(str(metric), title, True)
                if is_web_search_label(title) or is_web_search_label(metric_label):
                    excluded_labels.add(f"{title} :: {metric_label}")
                    continue
                oo = get_metric_display_value(row, str(metric), title, "OO", True, summary_tables)
                if oo is None:
                    continue
                for combo in ("OA", "AO", "AA"):
                    combo_val = get_metric_display_value(
                        row, str(metric), title, combo, True, summary_tables
                    )
                    if combo_val is None:
                        continue
                    delta_raw = combo_val - oo
                    model_points[model][combo].append(
                        {
                            "x": float(oo),
                            "y": float(delta_raw),
                            "label": metric_label,
                            "category": title,
                        }
                    )
    specs: List[Dict[str, object]] = []
    total_points = 0
    total_labels = 0
    for model, combo_map in model_points.items():
        any_points = any(combo_map[combo] for combo in ("OA", "AO", "AA"))
        if not any_points:
            continue
        combos = ["OA", "AO", "AA"]
        x_values = [pt["x"] for combo in combos for pt in combo_map[combo]]
        y_values = [pt["y"] for combo in combos for pt in combo_map[combo]]
        if not x_values or not y_values:
            continue
        total_points += len(x_values)
        scale_max = max(x_values)
        use_percent = scale_max > 1.5
        x_scale = 1.0 if use_percent else 100.0
        y_scale = 1.0 if use_percent else 100.0
        x_values_scaled = [val * x_scale for val in x_values]
        y_values_scaled = [val * y_scale for val in y_values]
        x_min = max(0.0, min(x_values_scaled) - (3 if use_percent else 3))
        x_max = min(100.0, max(x_values_scaled) + (3 if use_percent else 3))
        x_pad = 2.0
        x_min_plot = max(0.0, x_min - x_pad)
        x_max_plot = min(100.0, x_max + x_pad)
        pad = 1.0
        y_min = min(y_values_scaled) - pad
        y_max = max(y_values_scaled) + pad
        round_step = 0.5
        y_range = [
            math.floor(y_min / round_step) * round_step,
            math.ceil(y_max / round_step) * round_step,
        ]

        data: List[Dict[str, object]] = []
        shapes: List[Dict[str, object]] = []
        annotations: List[Dict[str, object]] = []
        for idx, combo in enumerate(combos, start=1):
            points = combo_map[combo]
            if not points:
                continue
            pos_sorted = sorted(points, key=lambda pt: pt["y"], reverse=True)
            neg_sorted = sorted(points, key=lambda pt: pt["y"])
            pos_cutoff = pos_sorted[2]["y"] if len(pos_sorted) >= 3 else None
            neg_cutoff = neg_sorted[2]["y"] if len(neg_sorted) >= 3 else None
            pos_set = {
                id(pt)
                for pt in pos_sorted
                if pos_cutoff is None or pt["y"] >= pos_cutoff
            }
            neg_set = {
                id(pt)
                for pt in neg_sorted
                if neg_cutoff is None or pt["y"] <= neg_cutoff
            }
            threshold_set = {
                id(pt) for pt in points if abs(pt["y"] * y_scale) >= 1.0
            }
            top_set = pos_set.union(neg_set)
            use_threshold = len(threshold_set) <= len(top_set)
            label_set = threshold_set if use_threshold else top_set
            jitter_x = 0.003
            jitter_y = 0.08
            xs = []
            ys = []
            labels = []
            bg_colors = []
            highlight_colors = []
            for pt in points:
                seed = f"{model}|{combo}|{pt['label']}".encode("utf-8")
                base = int(hashlib.md5(seed).hexdigest()[:8], 16) % 1000
                jitter = (base / 1000.0) - 0.5
                xs.append(pt["x"] * x_scale + jitter * jitter_x * x_scale)
                ys.append(pt["y"] * y_scale + jitter * jitter_y)
                labels.append(pt["label"])
                if id(pt) in pos_set:
                    highlight_colors.append("#2f855a")
                elif id(pt) in neg_set:
                    highlight_colors.append("#c43d3d")
                else:
                    highlight_colors.append("rgba(0,0,0,0)")
                bg_colors.append("#3b4a6b")
            data.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": xs,
                    "y": ys,
                    "marker": {"size": 5, "opacity": 0.22, "color": bg_colors},
                    "customdata": [
                        [pt["category"], pt["label"]] for pt in points
                    ],
                    "hovertemplate": (
                        "Category: %{customdata[0]}<br>"
                        "Row: %{customdata[1]}<br>"
                        "OO: %{x:.2f}<br>"
                        "Delta: %{y:.2f}pp<extra></extra>"
                    ),
                    "xaxis": f"x{idx}" if idx > 1 else "x",
                    "yaxis": f"y{idx}" if idx > 1 else "y",
                    "showlegend": False,
                }
            )
            data.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": xs,
                    "y": ys,
                    "marker": {"size": 9, "opacity": 1.0, "color": highlight_colors},
                    "customdata": [
                        [pt["category"], pt["label"]] for pt in points
                    ],
                    "hovertemplate": (
                        "Category: %{customdata[0]}<br>"
                        "Row: %{customdata[1]}<br>"
                        "OO: %{x:.2f}<br>"
                        "Delta: %{y:.2f}pp<extra></extra>"
                    ),
                    "xaxis": f"x{idx}" if idx > 1 else "x",
                    "yaxis": f"y{idx}" if idx > 1 else "y",
                    "showlegend": False,
                }
            )

            label_candidates = [pt for pt in points if id(pt) in label_set]
            label_candidates = sorted(label_candidates, key=lambda pt: abs(pt["y"]), reverse=True)
            placed: List[Tuple[float, float]] = []
            for pt in label_candidates:
                label = pt["label"]
                label = re.sub(r"^web search\\s*", "", label, flags=re.IGNORECASE)
                if len(label) > 22:
                    label = label[:19] + "..."
                base_x = pt["x"] * x_scale
                base_y = pt["y"] * y_scale
                direction = -1 if id(pt) in neg_set else 1
                dx = 0.7 * direction
                dy = 0.6 * direction
                pad_x = 1.5
                candidate_x = min(
                    max(base_x + dx, x_min_plot + pad_x),
                    x_max_plot - pad_x,
                )
                candidate_y = base_y + dy
                too_close = False
                for placed_x, placed_y in placed:
                    if abs(candidate_x - placed_x) < 3.2 and abs(candidate_y - placed_y) < 1.6:
                        too_close = True
                        break
                if too_close:
                    continue
                placed.append((candidate_x, candidate_y))
                annotations.append(
                    {
                        "x": candidate_x,
                        "y": candidate_y,
                        "xref": f"x{idx}" if idx > 1 else "x",
                        "yref": f"y{idx}" if idx > 1 else "y",
                        "text": label,
                        "showarrow": False,
                        "font": {"size": 9, "color": "#444"},
                        "xanchor": "center",
                        "yanchor": "middle",
                    }
                )
                total_labels += 1

            shapes.append(
                {
                    "type": "line",
                    "xref": f"x{idx}" if idx > 1 else "x",
                    "yref": f"y{idx}" if idx > 1 else "y",
                    "x0": x_min_plot,
                    "x1": x_max_plot,
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": "rgba(0,0,0,0.18)", "width": 1, "dash": "dot"},
                }
            )

        layout: Dict[str, object] = {
            "margin": {"l": 80, "r": 40, "t": 30, "b": 50},
            "plot_bgcolor": "#ffffff",
            "paper_bgcolor": "#ffffff",
            "annotations": annotations,
            "shapes": shapes,
        }
        x_domains = [
            [0.0, 0.31],
            [0.345, 0.655],
            [0.69, 1.0],
        ]
        titles = ["OA - OO", "AO - OO", "AA - OO"]
        for idx, title in enumerate(titles, start=1):
            axis_suffix = "" if idx == 1 else str(idx)
            layout[f"xaxis{axis_suffix}"] = {
                "title": "OO accuracy (%)" if idx == 2 else "",
                "range": [x_min_plot, x_max_plot],
                "showgrid": True,
                "gridcolor": "rgba(0,0,0,0.06)",
                "zeroline": False,
                "domain": x_domains[idx - 1],
                "anchor": f"y{axis_suffix}" if axis_suffix else "y",
            }
            layout[f"yaxis{axis_suffix}"] = {
                "title": "Delta accuracy (pp)" if idx == 1 else "",
                "range": y_range,
                "showgrid": True,
                "gridcolor": "rgba(0,0,0,0.06)",
                "zeroline": False,
                "domain": [0.0, 1.0],
                "anchor": f"x{axis_suffix}" if axis_suffix else "x",
            }
            annotations.append(
                {
                    "x": 0.5,
                    "y": 1.1,
                    "xref": f"x{axis_suffix} domain" if axis_suffix else "x domain",
                    "yref": f"y{axis_suffix} domain" if axis_suffix else "y domain",
                    "text": title,
                    "showarrow": False,
                    "font": {"size": 12, "color": "#333"},
                }
            )
        specs.append({"model": model, "data": data, "layout": layout})
    info = {
        "models": len(specs),
        "points": total_points,
        "labels": total_labels,
    }
    return specs, info, sorted(excluded_labels)


def export_thesis_scatter_facets(
    specs: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    if go is None:
        warn("plotly is not available; skipping thesis scatter exports.")
        return
    if not specs:
        warn("no thesis scatter data; skipping thesis scatter exports.")
        return
    for spec in specs:
        model = str(spec.get("model") or "model")
        fig = go.Figure(data=spec.get("data", []), layout=spec.get("layout", {}))
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = sanitize_filename(model)
        basename = (
            f"baseline_vs_delta_facet_no_websearch_{suffix}"
            if len(specs) > 1
            else "baseline_vs_delta_facet_no_websearch"
        )
        export_thesis_plot_images(fig.data, fig.layout.to_plotly_json(), output_dir, basename)

def compute_error_shift_summary(
    error_records: List[Dict[str, object]],
    combo: str,
    top_k: int,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, int]]:
    models = sorted({str(rec.get("model")) for rec in error_records if rec.get("model")})
    error_types: List[str] = []
    seen = set()
    for error_type in ERROR_TYPES_ORDER:
        error_types.append(error_type)
        seen.add(error_type)
    for rec in error_records:
        for key in (rec.get("errors") or {}).keys():
            if key not in seen:
                error_types.append(str(key))
                seen.add(str(key))

    results: Dict[str, List[Dict[str, float]]] = {}
    movers_count = 0
    for model in models:
        totals = {"OO": 0, combo: 0}
        counts = {"OO": Counter(), combo: Counter()}
        for rec in error_records:
            if rec.get("model") != model:
                continue
            condition = rec.get("condition")
            if condition not in totals:
                continue
            total = int(rec.get("total") or 0)
            totals[condition] += total
            errors = rec.get("errors") or {}
            for key, value in errors.items():
                counts[condition][str(key)] += int(value or 0)
        if totals["OO"] <= 0 or totals[combo] <= 0:
            continue
        deltas: List[Dict[str, float]] = []
        for error_type in error_types:
            oo_rate = counts["OO"][error_type] / totals["OO"] if totals["OO"] else 0.0
            combo_rate = counts[combo][error_type] / totals[combo] if totals[combo] else 0.0
            delta = (combo_rate - oo_rate) * 100.0
            deltas.append({"error_type": error_type, "delta": delta})
        positives = sorted(
            [row for row in deltas if row["delta"] > 0],
            key=lambda row: row["delta"],
            reverse=True,
        )[:top_k]
        negatives = sorted(
            [row for row in deltas if row["delta"] < 0],
            key=lambda row: row["delta"],
        )[:top_k]
        selected = sorted(positives + negatives, key=lambda row: row["delta"])
        movers_count += len(selected)
        results[model] = selected
    info = {
        "error_types": len(error_types),
        "models": len(results),
        "movers": movers_count,
    }
    return results, info


def export_thesis_error_shift_plot(
    error_records: List[Dict[str, object]],
    output_dir: Path,
) -> None:
    shifts, info = compute_error_shift_summary(error_records, "OO", TOP_ERROR_SHIFT_K)
    print(
        "[info] error shifts (AA-OO): "
        f"{info['error_types']} error types, "
        f"{info['models']} models, "
        f"{info['movers']} movers"
    )
    if go is None:
        warn("plotly is not available; skipping thesis error shift exports.")
        return
    if not shifts:
        warn("no error shift data; skipping thesis error shift exports.")
        return
    try:
        from plotly.subplots import make_subplots
    except Exception:
        warn("plotly.subplots unavailable; skipping thesis error shift exports.")
        return
    models = list(shifts.keys())
    max_abs = 0.1
    for rows in shifts.values():
        for row in rows:
            max_abs = max(max_abs, abs(row["delta"]))
    x_range = [-max_abs - 0.5, max_abs + 0.5]
    fig = make_subplots(
        rows=len(models),
        cols=1,
        shared_xaxes=True,
        subplot_titles=models,
        vertical_spacing=0.12,
    )
    for idx, model in enumerate(models, start=1):
        rows = shifts[model]
        if not rows:
            continue
        y = [row["error_type"] for row in rows]
        x = [row["delta"] for row in rows]
        colors = ["#2f855a" if val >= 0 else "#c43d3d" for val in x]
        text = [f"{val:+.1f}pp" for val in x]
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                orientation="h",
                marker={"color": colors},
                text=text,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="Error: %{y}<br>Δ: %{x:+.2f}pp<extra></extra>",
                showlegend=False,
            ),
            row=idx,
            col=1,
        )
        fig.update_xaxes(
            row=idx,
            col=1,
            range=x_range,
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.35)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
        )
        fig.update_yaxes(
            row=idx,
            col=1,
            showgrid=False,
        )
    fig.update_layout(
        height=260 * len(models) + 120,
        margin={"l": 140, "r": 30, "t": 40, "b": 40},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        title_text="Top-K error-type shifts (AA − OO)",
    )
    export_thesis_plot_images(
        fig.data,
        fig.layout.to_plotly_json(),
        output_dir,
        "top_error_shifts_AA_vs_OO",
    )


def short_model_label(model: str) -> str:
    lowered = model.lower()
    if "gpt-4o" in lowered:
        return "GPT-4o"
    if "gpt-5 mini" in lowered or "gpt-5-mini" in lowered:
        return "GPT-5 mini"
    if "gpt-5.1" in lowered or "gpt-5-1" in lowered:
        return "GPT-5.1"
    if len(model) <= 14:
        return model
    for sep in (" ", "-", "/"):
        if sep in model:
            head, tail = model.split(sep, 1)
            if head and tail:
                return f"{head}<br>{tail}"
    return model


def normalize_overall_values(
    rows: List[Dict[str, object]],
    key_suffix: str = "",
) -> Tuple[List[Dict[str, object]], bool]:
    values: List[float] = []
    for row in rows:
        for condition in CONDITIONS:
            value = row.get(f"{condition}{key_suffix}")
            if value is None or pd.isna(value):
                continue
            values.append(float(value))
    if not values:
        return rows, False
    max_value = max(values)
    needs_percent = max_value <= 1.2
    if not needs_percent:
        return rows, False
    scaled: List[Dict[str, object]] = []
    for row in rows:
        updated = dict(row)
        for condition in CONDITIONS:
            for key in (
                f"{condition}{key_suffix}",
                f"{condition}{key_suffix}_min",
                f"{condition}{key_suffix}_max",
            ):
                value = updated.get(key)
                if value is None or pd.isna(value):
                    continue
                updated[key] = float(value) * 100.0
        scaled.append(updated)
    return scaled, True


def build_thesis_main_result_plot(
    aggregated_rows: List[Dict[str, object]],
    *,
    use_excl_web: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object], str]:
    if not aggregated_rows:
        return [], {}, "No data available."

    key_suffix = "_excl_web" if use_excl_web else ""
    normalized_rows, scaled = normalize_overall_values(aggregated_rows, key_suffix)
    models = [row.get("model") for row in normalized_rows if row.get("model")]
    model_labels = [short_model_label(str(model)) for model in models]
    model_positions = list(range(len(models)))

    condition_labels = {
        "OO": "OO",
        "OA": "OA",
        "AO": "AO",
        "AA": "AA",
    }
    condition_offsets = {
        "OO": -0.27,
        "OA": -0.09,
        "AO": 0.09,
        "AA": 0.27,
    }
    condition_colors = {
        "OO": "#2b6cb0",
        "OA": "#2f855a",
        "AO": "#c05621",
        "AA": "#6b46c1",
    }

    traces: List[Dict[str, object]] = []
    y_values_all: List[float] = []
    value_lookup: Dict[Tuple[str, str], float] = {}
    for condition in CONDITIONS:
        xs: List[float] = []
        ys: List[float] = []
        err_plus: List[float] = []
        err_minus: List[float] = []
        model_names: List[str] = []
        for idx, row in enumerate(normalized_rows):
            mean_value = row.get(f"{condition}{key_suffix}")
            if mean_value is None or pd.isna(mean_value):
                continue
            mean_value = float(mean_value)
            min_value = row.get(f"{condition}{key_suffix}_min")
            max_value = row.get(f"{condition}{key_suffix}_max")
            if min_value is None or pd.isna(min_value) or max_value is None or pd.isna(max_value):
                min_value = max_value = mean_value
            min_value = float(min_value)
            max_value = float(max_value)
            xs.append(model_positions[idx] + condition_offsets[condition])
            ys.append(mean_value)
            err_plus.append(max_value - mean_value)
            err_minus.append(mean_value - min_value)
            y_values_all.extend([mean_value, min_value, max_value])
            model_name = str(row.get("model"))
            model_names.append(model_name)
            value_lookup[(model_name, condition)] = mean_value

        if not xs:
            continue

        include_error = any((plus > 0 or minus > 0) for plus, minus in zip(err_plus, err_minus))
        trace: Dict[str, object] = {
            "type": "scatter",
            "mode": "markers",
            "name": condition_labels[condition],
            "x": xs,
            "y": ys,
            "marker": {"size": 9, "color": condition_colors[condition]},
            "customdata": model_names,
        }
        if include_error:
            trace["error_y"] = {
                "type": "data",
                "array": err_plus,
                "arrayminus": err_minus,
                "visible": True,
                "thickness": 1,
                "width": 4,
            }
        traces.append(trace)

    if not y_values_all:
        return [], {}, "No data available."

    y_min = min(y_values_all)
    y_max = max(y_values_all)
    pad = 0.5
    y_low = y_min - pad
    y_high = y_max + pad
    round_step = 0.5
    y_range = [
        math.floor(y_low / round_step) * round_step,
        math.ceil(y_high / round_step) * round_step,
    ]

    annotations: List[Dict[str, object]] = []
    for model in models:
        if (model, "OO") not in value_lookup or (model, "AA") not in value_lookup:
            continue
        oo_value = value_lookup[(model, "OO")]
        aa_value = value_lookup[(model, "AA")]
        delta = aa_value - oo_value
        if abs(delta) < 0.1:
            continue
        base_x = model_positions[models.index(model)] + condition_offsets["AA"]
        label = f"{delta:+.1f}pp"
        annotations.append(
            {
                "x": base_x + 0.03,
                "y": aa_value + 0.12,
                "text": label,
                "showarrow": False,
                "font": {"size": 11, "color": "#444"},
                "xanchor": "left",
                "yanchor": "bottom",
            }
        )

    layout = {
        "margin": {"l": 70, "r": 30, "t": 20, "b": 80},
        "xaxis": {
            "title": "",
            "tickvals": model_positions,
            "ticktext": model_labels,
            "tickangle": 0,
            "showgrid": False,
            "zeroline": False,
        },
        "yaxis": {
            "title": "Overall accuracy (%)" if scaled else "Overall accuracy",
            "range": y_range,
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.08)",
            "zeroline": False,
        },
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.2},
        "annotations": annotations,
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
    }
    caption = "Mean across runs; error bars show min–max."
    return traces, layout, caption


def render_thesis_figures_html(
    with_web_traces: List[Dict[str, object]],
    with_web_layout: Dict[str, object],
    with_web_caption: str,
    excl_web_traces: List[Dict[str, object]],
    excl_web_layout: Dict[str, object],
    excl_web_caption: str,
    error_records: List[Dict[str, object]],
    error_groups: Dict[str, Dict[str, List[str]]],
    scatter_specs: List[Dict[str, object]],
) -> str:
    with_web_traces_json = json.dumps(with_web_traces)
    with_web_layout_json = json.dumps(with_web_layout)
    excl_web_traces_json = json.dumps(excl_web_traces)
    excl_web_layout_json = json.dumps(excl_web_layout)
    error_records_json = json.dumps(error_records)
    error_types_json = json.dumps(ERROR_TYPES_ORDER)
    error_groups_json = json.dumps(error_groups)
    scatter_specs_json = json.dumps(scatter_specs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Thesis Figures</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      font-family: "Times New Roman", Georgia, serif;
      color: #111;
      background: #ffffff;
    }}
    .figure {{
      max-width: 900px;
      margin: 0 auto;
    }}
    #thesis_main_result,
    #thesis_main_result_excl_web {{
      width: 100%;
      height: 420px;
    }}
    .caption {{
      margin-top: 12px;
      font-size: 14px;
      color: #333;
    }}
    .figure + .figure {{
      margin-top: 32px;
    }}
    .plot-controls {{
      display: flex;
      gap: 8px;
      margin-bottom: 8px;
      align-items: center;
      font-size: 14px;
    }}
    .plot-warning {{
      margin: 8px 0;
      color: #9c2c2c;
      font-size: 13px;
    }}
    .scatter-panel {{
      margin-top: 12px;
    }}
    .movers-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 12px;
    }}
    .movers-table th,
    .movers-table td {{
      border: 1px solid #d5d5d5;
      padding: 6px 8px;
      text-align: left;
    }}
    .movers-table th {{
      background: #f1f1f1;
    }}
  </style>
</head>
<body>
  <div class="figure">
    <div id="thesis_main_result"></div>
    <div class="caption">{with_web_caption} Includes web-search tasks.</div>
  </div>
  <div class="figure">
    <div id="thesis_main_result_excl_web"></div>
    <div class="caption">{excl_web_caption} Overall accuracy excluding web-search tasks.</div>
  </div>
  <div class="figure">
    <div id="thesis_error_mix_plot"></div>
    <div class="caption">
      Error distribution (share of incorrect outputs). Multi-label errors can push raw shares above 100%; bars are normalized to 100%.
    </div>
  </div>
  <div class="figure">
    <div class="plot-warning" id="thesisErrorHeatmapWarning"></div>
    <div id="thesis_error_heatmap_plot"></div>
    <div class="caption">
      Error shift heatmap (delta pp) for top-moving rows; tables list top movers by absolute delta.
    </div>
  </div>
  <div class="figure">
    <div id="thesis_scatter_facets"></div>
    <div class="caption">
      Baseline (OO) vs delta accuracy faceted by combo; y-axis in percentage points.
    </div>
  </div>
  <div class="figure">
    <div class="plot-controls">
      <label>Comparison
        <select id="thesisErrorShiftCombo">
          <option value="AA">AA - OO</option>
          <option value="OA">OA - OO</option>
          <option value="AO">AO - OO</option>
        </select>
      </label>
    </div>
    <div class="plot-warning" id="thesisErrorShiftWarning"></div>
    <div id="thesis_error_shift_plot"></div>
    <div class="caption">
      Rates are error_count / total_tasks for the slice; multi-label errors can sum &gt; 100%.
    </div>
  </div>
  <script>
    const thesisData = {with_web_traces_json};
    const thesisLayout = {with_web_layout_json};
    Plotly.newPlot("thesis_main_result", thesisData, thesisLayout, {{ displayModeBar: false }});
    const thesisExclWebData = {excl_web_traces_json};
    const thesisExclWebLayout = {excl_web_layout_json};
    Plotly.newPlot("thesis_main_result_excl_web", thesisExclWebData, thesisExclWebLayout, {{ displayModeBar: false }});

    const thesisErrorRecords = {error_records_json};
    const thesisErrorTypes = {error_types_json};
    const thesisErrorGroups = {error_groups_json};
    const thesisTopShiftK = {TOP_ERROR_SHIFT_K};
    const thesisTopRows = 15;
    const thesisTopMovers = 10;
    const thesisScatterSpecs = {scatter_specs_json};
    const thesisErrorMixContainer = document.getElementById("thesis_error_mix_plot");

    function renderThesisScatterFacets() {{
      const container = document.getElementById("thesis_scatter_facets");
      if (!container || typeof Plotly === "undefined") {{
        return;
      }}
      container.innerHTML = "";
      thesisScatterSpecs.forEach((spec) => {{
        const section = document.createElement("div");
        section.className = "scatter-panel";
        const heading = document.createElement("h2");
        heading.textContent = spec.model || "";
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "360px";
        section.appendChild(plotDiv);
        container.appendChild(section);
        Plotly.newPlot(plotDiv, spec.data, spec.layout, {{ displayModeBar: false }});
      }});
    }}

    function buildThesisErrorIndex() {{
      const index = {{}};
      (thesisErrorRecords || []).forEach((rec) => {{
        const model = rec.model;
        const condition = rec.condition;
        if (!model || !condition) {{
          return;
        }}
        if (!index[model]) {{
          index[model] = {{}};
        }}
        if (!index[model][condition]) {{
          index[model][condition] = {{ total: 0, incorrect: 0, errorCounts: {{}} }};
        }}
        index[model][condition].total += rec.total || 0;
        index[model][condition].incorrect += rec.incorrect || 0;
        const errors = rec.errors || {{}};
        Object.keys(errors).forEach((key) => {{
          index[model][condition].errorCounts[key] =
            (index[model][condition].errorCounts[key] || 0) + (errors[key] || 0);
        }});
      }});
      return index;
    }}

    const thesisErrorIndex = buildThesisErrorIndex();

    function formatGroupTitle(key) {{
      return String(key || "")
        .replace("data_", "")
        .replace(/_/g, " ")
        .replace(/\b\w/g, (m) => m.toUpperCase());
    }}

    function aggregateThesisErrors(model, condition, testCategories) {{
      let total = 0;
      let incorrect = 0;
      const errorCounts = {{}};
      testCategories.forEach((testCategory) => {{
        const rec =
          thesisErrorIndex[model] &&
          thesisErrorIndex[model][condition] &&
          thesisErrorIndex[model][condition][testCategory]
            ? thesisErrorIndex[model][condition][testCategory]
            : null;
        if (!rec) {{
          return;
        }}
        total += rec.total || 0;
        incorrect += rec.incorrect || 0;
        const errors = rec.errors || {{}};
        Object.keys(errors).forEach((key) => {{
          errorCounts[key] = (errorCounts[key] || 0) + errors[key];
        }});
      }});
      return {{ total, incorrect, errorCounts }};
    }}

    function renderThesisTopMovers(entries) {{
      if (!entries.length) {{
        return null;
      }}
      const table = document.createElement("table");
      table.className = "movers-table";
      table.innerHTML =
        "<thead><tr>" +
        "<th>Subcategory</th>" +
        "<th>Error type</th>" +
        "<th>Delta (pp)</th>" +
        "<th>OO rate</th>" +
        "<th>Combo rate</th>" +
        "<th>N all</th>" +
        "<th>N wrong OO</th>" +
        "<th>N wrong combo</th>" +
        "</tr></thead>";
      const tbody = document.createElement("tbody");
      entries.forEach((row) => {{
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td>${{row.label}}</td>` +
          `<td>${{row.errorType}}</td>` +
          `<td>${{row.delta.toFixed(2)}}</td>` +
          `<td>${{(row.ooRate * 100).toFixed(2)}}%</td>` +
          `<td>${{(row.comboRate * 100).toFixed(2)}}%</td>` +
          `<td>${{row.total}}</td>` +
          `<td>${{row.ooIncorrect}}</td>` +
          `<td>${{row.comboIncorrect}}</td>`;
        tbody.appendChild(tr);
      }});
      table.appendChild(tbody);
      return table;
    }}

    function updateThesisErrorHeatmap() {{
      const container = document.getElementById("thesis_error_heatmap_plot");
      const warning = document.getElementById("thesisErrorHeatmapWarning");
      if (!container || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      container.innerHTML = "";
      warning.style.display = "none";
      const comboSelect = document.getElementById("thesisErrorShiftCombo");
      const combo = comboSelect ? comboSelect.value : "AA";

      const models = Object.keys(thesisErrorIndex || {{}}).sort();
      if (!models.length) {{
        warning.style.display = "";
        warning.textContent = "No error data available for thesis heatmap.";
        return;
      }}

      models.forEach((model) => {{
        const rows = [];
        let maxAbs = 0;
        Object.entries(thesisErrorGroups || {{}}).forEach(([groupKey, group]) => {{
          const categoryTitle = formatGroupTitle(groupKey);
          Object.entries(group || {{}}).forEach(([metricLabel, categories]) => {{
            if (!categories || !categories.length) {{
              return;
            }}
            const oo = aggregateThesisErrors(model, "OO", categories);
            const comboAgg = aggregateThesisErrors(model, combo, categories);
            if (!oo.total || !comboAgg.total) {{
              return;
            }}
            const row = {{
              label: metricLabel,
              category: categoryTitle,
              values: {{}},
              maxAbs: 0,
              total: oo.total,
              ooIncorrect: oo.incorrect,
              comboIncorrect: comboAgg.incorrect
            }};
            thesisErrorTypes.forEach((errorType) => {{
              const ooRate = oo.total ? (oo.errorCounts[errorType] || 0) / oo.total : 0;
              const comboRate = comboAgg.total
                ? (comboAgg.errorCounts[errorType] || 0) / comboAgg.total
                : 0;
              const delta = (comboRate - ooRate) * 100;
              row.values[errorType] = {{
                ooRate,
                comboRate,
                delta,
                total: oo.total,
                ooIncorrect: oo.incorrect,
                comboIncorrect: comboAgg.incorrect
              }};
              row.maxAbs = Math.max(row.maxAbs, Math.abs(delta));
              maxAbs = Math.max(maxAbs, Math.abs(delta));
            }});
            rows.push(row);
          }});
        }});

        if (!rows.length) {{
          return;
        }}
        const topRows = rows
          .sort((a, b) => b.maxAbs - a.maxAbs)
          .slice(0, thesisTopRows);
        if (!maxAbs) {{
          maxAbs = 0.1;
        }}
        const xLabels = thesisErrorTypes.slice();
        const yLabels = topRows.map((row) => row.label);
        const z = topRows.map((row) =>
          xLabels.map((errorType) => row.values[errorType].delta)
        );
        const custom = topRows.map((row) =>
          xLabels.map((errorType) => ({{
            label: row.label,
            category: row.category,
            errorType,
            ooRate: row.values[errorType].ooRate,
            comboRate: row.values[errorType].comboRate,
            delta: row.values[errorType].delta,
            total: row.values[errorType].total,
            ooIncorrect: row.values[errorType].ooIncorrect,
            comboIncorrect: row.values[errorType].comboIncorrect
          }}))
        );

        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "480px";
        section.appendChild(plotDiv);
        const tableWrap = document.createElement("div");
        container.appendChild(section);

        const trace = {{
          type: "heatmap",
          z,
          x: xLabels,
          y: yLabels,
          zmid: 0,
          zmin: -maxAbs,
          zmax: maxAbs,
          customdata: custom,
          hovertemplate:
            "Row: %{{customdata.label}}<br>" +
            "Category: %{{customdata.category}}<br>" +
            "Error: %{{customdata.errorType}}<br>" +
            "OO rate: %{{customdata.ooRate:.2%}}<br>" +
            "Combo rate: %{{customdata.comboRate:.2%}}<br>" +
            "Delta: %{{customdata.delta:+.2f}}pp<br>" +
            "N: %{{customdata.total}}<extra></extra>"
        }};
        const layout = {{
          margin: {{ t: 20, l: 140, r: 20, b: 120 }},
          xaxis: {{ tickangle: -30 }},
          yaxis: {{ automargin: true }}
        }};
        Plotly.newPlot(plotDiv, [trace], layout, {{ displayModeBar: false }});

        const movers = [];
        topRows.forEach((row) => {{
          thesisErrorTypes.forEach((errorType) => {{
            const cell = row.values[errorType];
            if (!cell) {{
              return;
            }}
            movers.push({{
              label: row.label,
              errorType,
              delta: cell.delta,
              ooRate: cell.ooRate,
              comboRate: cell.comboRate,
              total: cell.total,
              ooIncorrect: row.ooIncorrect,
              comboIncorrect: row.comboIncorrect
            }});
          }});
        }});
        movers.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
        const moversTable = renderThesisTopMovers(movers.slice(0, thesisTopMovers));
        if (moversTable) {{
          tableWrap.appendChild(moversTable);
          section.appendChild(tableWrap);
        }}
      }});
    }}

    function updateThesisErrorMix() {{
      if (!thesisErrorMixContainer || typeof Plotly === "undefined") {{
        return;
      }}
      thesisErrorMixContainer.innerHTML = "";
      const models = Object.keys(thesisErrorIndex || {{}}).sort();
      models.forEach((model) => {{
        const conditions = ["OO", "OA", "AO", "AA"].filter(
          (condition) => thesisErrorIndex[model] && thesisErrorIndex[model][condition]
        );
        if (!conditions.length) {{
          return;
        }}
        if (!conditions.includes("OO")) {{
          return;
        }}
        const avgShares = thesisErrorTypes.map((errorType) => {{
          let totalShare = 0;
          let count = 0;
          conditions.forEach((condition) => {{
            const agg = thesisErrorIndex[model][condition];
            const share = agg.incorrect
              ? (agg.errorCounts[errorType] || 0) / agg.incorrect
              : 0;
            totalShare += share;
            count += 1;
          }});
          return {{ errorType, avgShare: count ? totalShare / count : 0 }};
        }});
        const topErrors = avgShares
          .filter((row) => row.errorType !== "OTHER")
          .sort((a, b) => b.avgShare - a.avgShare)
          .slice(0, 5)
          .map((row) => row.errorType);
        const orderedErrors = topErrors.concat("OTHER");
        console.log("Thesis error mix labels:", orderedErrors);

        const traces = orderedErrors.map((errorType) => {{
          const y = [];
          const custom = [];
          conditions.forEach((condition) => {{
            const agg = thesisErrorIndex[model][condition];
            let count = 0;
            if (errorType === "OTHER") {{
              count = agg.errorCounts["OTHER"] || 0;
            }} else {{
              count = agg.errorCounts[errorType] || 0;
            }}
            const rawShare = agg.incorrect ? count / agg.incorrect : 0;
            custom.push({{
              errorType,
              count,
              share: rawShare,
              incorrect: agg.incorrect
            }});
            y.push(rawShare);
          }});
          return {{
            name: errorType,
            type: "bar",
            x: conditions,
            y,
            customdata: custom,
            hovertemplate:
              "Error: %{{customdata.errorType}}<br>" +
              "Count: %{{customdata.count}}<br>" +
              "Share: %{{customdata.share:.2%}}<br>" +
              "Total incorrect: %{{customdata.incorrect}}<extra></extra>"
          }};
        }});
        const totalsByCondition = conditions.map((condition) => {{
          let totalShare = 0;
          traces.forEach((trace) => {{
            const idx = trace.x.indexOf(condition);
            if (idx >= 0) {{
              totalShare += trace.y[idx] || 0;
            }}
          }});
          return totalShare;
        }});
        traces.forEach((trace) => {{
          trace.y = trace.y.map((val, idx) => {{
            const denom = totalsByCondition[idx] || 1;
            return denom ? val / denom : 0;
          }});
        }});

        const annotations = conditions.map((condition, idx) => ({{
          x: condition,
          y: 1.04,
          text: "n=" + (thesisErrorIndex[model][condition].incorrect || 0),
          showarrow: false,
          yref: "y",
          xref: "x",
          font: {{ size: 11, color: "#444" }}
        }}));

        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "360px";
        section.appendChild(plotDiv);
        thesisErrorMixContainer.appendChild(section);

        const layout = {{
          barmode: "stack",
          margin: {{ t: 60, l: 60, r: 20, b: 40 }},
          yaxis: {{
            title: "Share of incorrect",
            tickformat: ".0%",
            range: [0, 1.08]
          }},
          annotations
        }};
        Plotly.newPlot(plotDiv, traces, layout, {{ displayModeBar: false }});
      }});
    }}

    function updateThesisErrorShifts() {{
      const combo = document.getElementById("thesisErrorShiftCombo").value;
      const container = document.getElementById("thesis_error_shift_plot");
      const warning = document.getElementById("thesisErrorShiftWarning");
      if (!container || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      container.innerHTML = "";
      const models = Object.keys(thesisErrorIndex || {{}}).sort();
      let movers = 0;
      let plotted = 0;
      models.forEach((model) => {{
        const oo = thesisErrorIndex[model] ? thesisErrorIndex[model]["OO"] : null;
        const comboAgg = thesisErrorIndex[model] ? thesisErrorIndex[model][combo] : null;
        if (!oo || !comboAgg || !oo.total || !comboAgg.total) {{
          return;
        }}
        const deltas = thesisErrorTypes.map((errorType) => {{
          const ooRate = oo.total ? (oo.errorCounts[errorType] || 0) / oo.total : 0;
          const comboRate = comboAgg.total
            ? (comboAgg.errorCounts[errorType] || 0) / comboAgg.total
            : 0;
          return {{
            errorType,
            delta: (comboRate - ooRate) * 100
          }};
        }});
        const selected = deltas
          .filter((row) => Math.abs(row.delta) > 0.1)
          .sort((a, b) => a.delta - b.delta);
        if (!selected.length) {{
          return;
        }}
        movers += selected.length;
        plotted += 1;

        const y = selected.map((row) => row.errorType);
        const x = selected.map((row) => row.delta);
        const colors = x.map((val) => (val >= 0 ? "#2f855a" : "#c43d3d"));
        const text = x.map((val) => `${{val >= 0 ? "+" : ""}}${{val.toFixed(1)}}pp`);
        const trace = {{
          type: "bar",
          orientation: "h",
          x,
          y,
          marker: {{ color: colors }},
          text,
          textposition: "outside",
          cliponaxis: false,
          hovertemplate: "Error: %{{y}}<br>Delta %{{x:+.2f}}pp<extra></extra>"
        }};

        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "320px";
        section.appendChild(plotDiv);
        container.appendChild(section);

        const minX = Math.min(...x);
        const maxX = Math.max(...x);
        const pad = Math.max(0.6, (Math.abs(minX) + Math.abs(maxX)) * 0.08);
        const layout = {{
          margin: {{ t: 20, l: 200, r: 40, b: 40 }},
          xaxis: {{
            title: "Delta error rate (pp)",
            zeroline: true,
            zerolinecolor: "rgba(0,0,0,0.35)",
            showgrid: true,
            gridcolor: "rgba(0,0,0,0.08)",
            range: [minX - pad, maxX + pad]
          }},
          yaxis: {{
            automargin: true,
            showgrid: false
          }}
        }};
        Plotly.newPlot(plotDiv, [trace], layout, {{ displayModeBar: false }});
      }});

      if (!container.childNodes.length) {{
        warning.style.display = "";
        warning.textContent = "No error data for the current selection.";
      }} else {{
        warning.style.display = "none";
      }}
      console.log("[thesis error shifts]", {{
        combo,
        errorTypes: thesisErrorTypes.length,
        models: plotted,
        movers
      }});
    }}

    document.getElementById("thesisErrorShiftCombo").addEventListener("change", () => {{
      updateThesisErrorShifts();
      updateThesisErrorHeatmap();
    }});
    updateThesisErrorShifts();
    updateThesisErrorHeatmap();
    updateThesisErrorMix();
    renderThesisScatterFacets();
  </script>
</body>
</html>
"""


def export_thesis_plot_images(
    thesis_traces: List[Dict[str, object]],
    thesis_layout: Dict[str, object],
    output_dir: Path,
    basename: str,
) -> None:
    if go is None:
        warn("plotly is not available; skipping thesis figure exports.")
        return
    if not thesis_traces or not thesis_layout:
        warn("no thesis plot data; skipping thesis figure exports.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(data=thesis_traces, layout=thesis_layout)
    svg_path = output_dir / f"{basename}.svg"
    png_path = output_dir / f"{basename}.png"
    try:
        fig.write_image(str(svg_path))
    except Exception as exc:  # pragma: no cover - optional
        warn(
            "failed to export SVG (install kaleido with `pip install -U kaleido`): "
            f"{exc}"
        )
    try:
        fig.write_image(str(png_path), width=2000, height=1200, scale=1)
    except Exception as exc:  # pragma: no cover - optional
        warn(
            "failed to export PNG (install kaleido with `pip install -U kaleido`): "
            f"{exc}"
        )


def compute_deltas(wide: pd.DataFrame) -> pd.DataFrame:
    result = wide.copy()
    if "OO_acc" in result.columns:
        baseline = result["OO_acc"]
    else:
        baseline = pd.Series([math.nan] * len(result), index=result.index)

    for condition in ("OO", "OA", "AO", "AA"):
        acc_col = f"{condition}_acc"
        delta_col = f"{condition}_delta"
        if acc_col not in result.columns:
            result[acc_col] = math.nan
        if condition == "OO":
            result[delta_col] = 0.0
        else:
            delta = result[acc_col] - baseline
            delta[baseline.isna()] = math.nan
            result[delta_col] = delta
    return result


def make_rows(
    table: pd.DataFrame,
    overall_excl_web: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, row in table.iterrows():
        accs = {
            "OO": row.get("OO_acc"),
            "OA": row.get("OA_acc"),
            "AO": row.get("AO_acc"),
            "AA": row.get("AA_acc"),
        }
        max_acc = None
        for val in accs.values():
            if pd.notna(val):
                if max_acc is None or val > max_acc:
                    max_acc = float(val)

        excl_values = {}
        for condition in ("OO", "OA", "AO", "AA"):
            excl = overall_excl_web.get(condition, {}).get(str(model))
            excl_values[condition] = float(excl) if excl is not None else None

        rows.append(
            {
                "model": str(model),
                "OO": float(accs["OO"]) if pd.notna(accs["OO"]) else None,
                "OA": float(accs["OA"]) if pd.notna(accs["OA"]) else None,
                "AO": float(accs["AO"]) if pd.notna(accs["AO"]) else None,
                "AA": float(accs["AA"]) if pd.notna(accs["AA"]) else None,
                "OO_excl_web": excl_values["OO"],
                "OA_excl_web": excl_values["OA"],
                "AO_excl_web": excl_values["AO"],
                "AA_excl_web": excl_values["AA"],
                "bestAcc": max_acc,
            }
        )
    return rows


def filename_to_title(filename: str) -> str:
    stem = filename.replace("data_", "").replace(".csv", "")
    return stem.replace("_", " ").title()


def build_category_tables(
    data_map: Dict[str, Dict[str, pd.DataFrame]],
    mode: str,
) -> List[Dict[str, object]]:
    filenames: set[str] = set()
    for condition_data in data_map.values():
        filenames.update(condition_data.keys())
    filenames.discard("data_overall.csv")
    filenames.discard("data_format_sensitivity.csv")

    tables: List[Dict[str, object]] = []
    for filename in sorted(filenames):
        condition_frames: Dict[str, pd.DataFrame] = {}
        missing_conditions: List[str] = []
        for condition in CONDITIONS:
            df = data_map.get(condition, {}).get(filename)
            if df is None:
                missing_conditions.append(condition)
                continue
            condition_frames[condition] = df
        if not condition_frames:
            continue

        sample_df = next(iter(condition_frames.values()))
        metrics = [col for col in sample_df.columns if col not in ("Rank", "Model")]

        def build_table(
            key_suffix: str,
            title: str,
            metric_list: List[str],
        ) -> Optional[Dict[str, object]]:
            if not metric_list:
                return None
            models: List[str] = []
            model_sets = [set(df["Model"].tolist()) for df in condition_frames.values()]
            if model_sets:
                models = sorted(set().union(*model_sets))

            per_condition_maps: Dict[str, pd.DataFrame] = {}
            for condition, df in condition_frames.items():
                per_condition_maps[condition] = df.set_index("Model")

            rows: List[Dict[str, object]] = []
            for model in models:
                metric_values: Dict[str, Dict[str, Optional[float]]] = {}
                for metric in metric_list:
                    metric_values[metric] = {}
                    for condition in CONDITIONS:
                        table = per_condition_maps.get(condition)
                        if table is None or metric not in table.columns:
                            metric_values[metric][condition] = None
                            continue
                        value = table.at[model, metric] if model in table.index else math.nan
                        metric_values[metric][condition] = (
                            float(value) if pd.notna(value) else None
                        )
                rows.append({"model": str(model), "metrics": metric_values})

            return {
                "key": f"{filename.replace('.csv', '')}{key_suffix}",
                "title": title,
                "metrics": metric_list,
                "rows": rows,
                "missing": missing_conditions,
            }

        if mode == "summary":
            if filename == "data_agentic.csv":
                overall_metrics = [m for m in metrics if "overall" in m.lower()]
                web_summary = [
                    m
                    for m in metrics
                    if "web search" in m.lower() and "summary" in m.lower()
                ]
                memory_summary = [
                    m for m in metrics if "memory" in m.lower() and "summary" in m.lower()
                ]
                table = build_table("", filename_to_title(filename), overall_metrics)
                if table:
                    tables.append(table)
                table = build_table("-web-search", "Web Search", web_summary)
                if table:
                    tables.append(table)
                table = build_table("-memory", "Memory", memory_summary)
                if table:
                    tables.append(table)
                continue

            summary_metrics = [m for m in metrics if "overall" in m.lower()]
            table = build_table("", filename_to_title(filename), summary_metrics)
            if table:
                tables.append(table)
            continue

        table = build_table("", filename_to_title(filename), metrics)
        if table:
            tables.append(table)

    return tables


def render_html(
    rows: List[Dict[str, object]],
    aggregated_rows: List[Dict[str, object]],
    missing_conditions: List[str],
    category_tables: List[Dict[str, object]],
    summary_tables: List[Dict[str, object]],
    all_models: List[str],
    overall_components: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    error_records: List[Dict[str, object]],
    error_groups: Dict[str, Dict[str, List[str]]],
) -> str:
    data_json = json.dumps(rows)
    aggregated_json = json.dumps(aggregated_rows)
    categories_json = json.dumps(category_tables)
    summaries_json = json.dumps(summary_tables)
    models_json = json.dumps(all_models)
    components_json = json.dumps(overall_components)
    error_records_json = json.dumps(error_records)
    error_groups_json = json.dumps(error_groups)
    missing_note = ""
    if missing_conditions:
        missing_note = (
            "Missing data for conditions: "
            + ", ".join(sorted(missing_conditions))
            + ". Showing available results."
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Function Calling Leaderboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --bg: #f3f6ff;
      --panel: #ffffff;
      --border: #d7e3ff;
      --text: #1d2b4f;
      --muted: #5b6b8a;
      --cell: #c8d7ff;
      --best: #7ef07b;
      --baseline: #bcd0ff;
    }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #f0f5ff, #e5efff);
      color: var(--text);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }}
    .wrap {{
      padding: 24px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 24px;
      font-weight: 700;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      background: var(--panel);
      padding: 12px 16px;
      border: 1px solid var(--border);
      border-radius: 10px;
      box-shadow: 0 6px 16px rgba(23, 51, 94, 0.08);
      margin-bottom: 16px;
    }}
    .tabs {{
      display: inline-flex;
      gap: 8px;
      margin: 4px 0 16px;
    }}
    .tab-button {{
      border: 1px solid var(--border);
      background: #f8fbff;
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 13px;
      color: var(--text);
      cursor: pointer;
    }}
    .tab-button.active {{
      background: #2b5cff;
      border-color: #2b5cff;
      color: #ffffff;
    }}
    .tab-panel {{
      display: none;
    }}
    .tab-panel.active {{
      display: block;
    }}
    .plot-panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 6px 16px rgba(23, 51, 94, 0.08);
    }}
    .plot-subtabs {{
      display: inline-flex;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .plot-subtabs.secondary {{
      margin-top: 6px;
    }}
    .plot-view {{
      display: none;
    }}
    .plot-view.active {{
      display: block;
    }}
    .plot-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .plot-warning {{
      padding: 12px;
      border: 1px dashed var(--border);
      border-radius: 10px;
      background: #f6f8ff;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
      margin-bottom: 12px;
    }}
    .plot-warning + .plot-warning {{
      margin-top: 8px;
    }}
    .plot-info {{
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    #heatmap_stack .section {{
      margin: 24px 0;
    }}
    #plot_scatter .section {{
      margin: 20px 0;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}
    #plot_contrib .section {{
      margin: 20px 0;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}
    .controls label {{
      font-size: 13px;
      color: var(--muted);
    }}
    select, button {{
      margin-left: 6px;
      padding: 6px 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #f8fbff;
      color: var(--text);
      font-size: 13px;
    }}
    button {{
      cursor: pointer;
    }}
    .filter {{
      min-width: 200px;
    }}
    .model-filter {{
      width: 220px;
      min-height: 120px;
    }}
    .sortable {{
      cursor: pointer;
      user-select: none;
    }}
    .sortable:hover {{
      text-decoration: underline;
      background: #eef3ff;
    }}
    .overall-header {{
      background: #e2e6ef;
      font-weight: 700;
    }}
    .sort-indicator {{
      margin-left: 6px;
      font-size: 11px;
      color: var(--muted);
    }}
    .note {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .section {{
      margin-top: 24px;
    }}
    .section h2 {{
      margin: 16px 0 8px;
      font-size: 18px;
    }}
    .section .note {{
      margin-top: 0;
    }}
    .chart-card {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: var(--panel);
    }}
    .chart-title {{
      font-size: 14px;
      font-weight: 700;
      margin: 0 0 8px;
    }}
    .chart-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .chart-controls select {{
      margin-left: 6px;
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      overflow: visible;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 760px;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #f9fbff;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      padding: 10px;
      text-align: center;
    }}
    tbody td {{
      padding: 12px 10px;
      text-align: center;
      background: var(--cell);
      border-bottom: 1px solid #e6ecff;
      border-right: 1px solid #e6ecff;
      font-size: 14px;
    }}
    tbody td:first-child {{
      text-align: left;
      background: #f2f6ff;
      font-weight: 600;
      color: #15307a;
    }}
    tbody td.baseline {{
      background: var(--baseline);
      font-weight: 600;
    }}
    tbody td.best {{
      background: var(--best);
      font-weight: 700;
      color: #0c3a0c;
    }}
    tbody tr:last-child td {{
      border-bottom: none;
    }}
    tbody td:last-child, thead th:last-child {{
      border-right: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>OpenFunctions Leaderboard</h1>
    <div class="controls">
      <label>Delta format
        <select id="deltaMode">
          <option value="pp">pp</option>
          <option value="rel">relative%</option>
        </select>
      </label>
      <label>Detail mode
        <select id="detailMode">
          <option value="full">Full</option>
          <option value="summary">Summary</option>
        </select>
      </label>
      <label>Table
        <select id="tableSelect"></select>
      </label>
      <label class="filter">Models
        <select id="modelFilter" class="model-filter" multiple></select>
      </label>
      <button id="selectAllModels">All</button>
      <button id="clearModels">None</button>
      <label>
        <input type="checkbox" id="excludeWebSearch" checked>
        Exclude web_search from overall
      </label>
    </div>
    <div class="note" id="note">{missing_note}</div>
    <div class="note" id="overallLabel"></div>
    <div class="tabs">
      <button class="tab-button active" id="tabResults" type="button">Results</button>
      <button class="tab-button" id="tabPlots" type="button">Plots</button>
    </div>
    <div class="tab-panel active" id="resultsTab">
      <div class="table-wrap">
        <table data-table-key="main">
          <thead>
            <tr>
              <th>Model</th>
              <th>OO</th>
              <th>OA</th>
              <th>AO</th>
              <th>AA</th>
            </tr>
          </thead>
          <tbody id="tableBody"></tbody>
        </table>
      </div>
      <div class="section" id="mainOverallSection">
        <div class="chart-card" id="mainOverallCard">
          <div class="chart-title">Main Result (Overall Acc)</div>
          <div class="note">Bars show mean across runs; error bars show min–max.</div>
          <div id="main_overall_chart"></div>
        </div>
      </div>
      <div id="detailSections"></div>
    </div>
    <div class="tab-panel" id="plotsTab">
      <div class="plot-panel">
        <div class="plot-subtabs">
          <button class="tab-button active" id="plotTabScatter" type="button">Scatter</button>
          <button class="tab-button" id="plotTabHeatmap" type="button">Heatmap</button>
          <button class="tab-button" id="plotTabContribution" type="button">Contribution</button>
          <button class="tab-button" id="plotTabErrors" type="button">Errors</button>
        </div>
        <div class="plot-view active" id="plots_scatter_view">
          <div class="plot-controls">
            <label>Metric mode
              <select id="plotMetricMode">
                <option value="accuracy">Accuracy</option>
                <option value="delta">Delta</option>
              </select>
            </label>
          <label>Slice
            <select id="plotCategoryFilter"></select>
          </label>
            <label>
              <input type="checkbox" class="plot-combo" value="OA" checked>
              OA
            </label>
            <label>
              <input type="checkbox" class="plot-combo" value="AO" checked>
              AO
            </label>
            <label>
              <input type="checkbox" class="plot-combo" value="AA" checked>
              AA
            </label>
          </div>
          <div class="plot-warning" id="plotWarning"></div>
          <div id="plot_scatter"></div>
          <div class="plot-info">Points show OO vs OA/AO/AA for the current table slice and detail mode.</div>
        </div>
        <div class="plot-view" id="plots_heatmap_view">
          <div class="plot-controls">
            <label>Combo
              <select id="heatmapCombo">
                <option value="OA">OA</option>
                <option value="AO">AO</option>
                <option value="AA">AA</option>
              </select>
            </label>
          </div>
          <div class="plot-info">Cells show Δ accuracy = combo − OO, per model and subcategory.</div>
          <div class="plot-warning" id="heatmapWarning"></div>
          <div id="heatmap_stack"></div>
        </div>
        <div class="plot-view" id="plots_contrib_view">
          <div class="plot-info">
            Stacked bars show how each top-level component contributes to the overall delta vs OO.
          </div>
          <div class="plot-warning" id="contribWarning"></div>
          <div id="plot_contrib"></div>
        </div>
        <div class="plot-view" id="plots_errors_view">
          <div class="plot-subtabs secondary">
            <button class="tab-button active" id="errorTabMix" type="button">Error mix</button>
            <button class="tab-button" id="errorTabHeatmap" type="button">Delta heatmap</button>
            <button class="tab-button" id="errorTabBuckets" type="button">BFCL buckets</button>
            <button class="tab-button" id="errorTabShifts" type="button">Top shifts</button>
          </div>
          <div class="plot-view active" id="error_mix_view">
            <div class="plot-controls">
              <label>Display
                <select id="errorMixMode">
                  <option value="share">Share</option>
                  <option value="count">Counts</option>
                </select>
              </label>
            </div>
            <div class="plot-warning" id="errorMixWarning"></div>
            <div id="error_mix_plot"></div>
            <div class="plot-info">Stacked bars show composition of incorrect examples by error type.</div>
          </div>
          <div class="plot-view" id="error_heatmap_view">
            <div class="plot-controls">
              <label>Condition
                <select id="errorHeatmapCombo">
                  <option value="OA">OA</option>
                  <option value="AO">AO</option>
                  <option value="AA">AA</option>
                </select>
              </label>
            </div>
            <div class="plot-warning" id="errorHeatmapWarning"></div>
            <div id="error_heatmap_plot"></div>
            <div class="plot-info">
              Cells show Δ error rate (condition − OO), over total examples in each slice.
            </div>
          </div>
          <div class="plot-view" id="error_bucket_view">
            <div class="plot-controls">
              <label>Display
                <select id="errorBucketMode">
                  <option value="share">Share of errors</option>
                  <option value="rate">Error rate</option>
                </select>
              </label>
            </div>
            <div class="plot-warning" id="errorBucketWarning"></div>
            <div id="error_bucket_plot"></div>
            <div class="plot-info">
              Buckets follow BFCL-style root causes; error rate is per total examples.
            </div>
          </div>
          <div class="plot-view" id="error_shift_view">
            <div class="plot-controls">
              <label>Comparison
                <select id="errorShiftCombo">
                  <option value="AA">AA - OO</option>
                  <option value="OA">OA - OO</option>
                  <option value="AO">AO - OO</option>
                </select>
              </label>
            </div>
            <div class="plot-warning" id="errorShiftWarning"></div>
            <div id="error_shift_plot"></div>
            <div class="plot-info">
              Rates are error_count / total_tasks for the slice; multi-label errors can sum &gt; 100%.
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script>
    const MISSING = "--";
    const mainRows = {data_json};
    const mainRowsAggregated = {aggregated_json};
    const categoryTablesFull = {categories_json};
    const categoryTablesSummary = {summaries_json};
    const allModels = {models_json};
    const overallComponents = {components_json};
    const errorRecords = {error_records_json};
    const errorGroups = {error_groups_json};
    const conditions = ["OO", "OA", "AO", "AA"];
    const tableState = {{}};
    const barState = {{}};

    function setActiveTab(tabId) {{
      const resultsTab = document.getElementById("resultsTab");
      const plotsTab = document.getElementById("plotsTab");
      const tabResults = document.getElementById("tabResults");
      const tabPlots = document.getElementById("tabPlots");
      if (!resultsTab || !plotsTab || !tabResults || !tabPlots) {{
        return;
      }}
      const showPlots = tabId === "plots";
      resultsTab.classList.toggle("active", !showPlots);
      plotsTab.classList.toggle("active", showPlots);
      tabResults.classList.toggle("active", !showPlots);
      tabPlots.classList.toggle("active", showPlots);
      updatePlot();
      updateHeatmaps();
      updateContribution();
    }}

    function setPlotView(viewId) {{
      const scatterView = document.getElementById("plots_scatter_view");
      const heatmapView = document.getElementById("plots_heatmap_view");
      const contribView = document.getElementById("plots_contrib_view");
      const errorsView = document.getElementById("plots_errors_view");
      const tabScatter = document.getElementById("plotTabScatter");
      const tabHeatmap = document.getElementById("plotTabHeatmap");
      const tabContrib = document.getElementById("plotTabContribution");
      const tabErrors = document.getElementById("plotTabErrors");
      if (
        !scatterView ||
        !heatmapView ||
        !contribView ||
        !errorsView ||
        !tabScatter ||
        !tabHeatmap ||
        !tabContrib ||
        !tabErrors
      ) {{
        return;
      }}
      const showScatter = viewId === "scatter";
      const showHeatmap = viewId === "heatmap";
      const showContrib = viewId === "contrib";
      const showErrors = viewId === "errors";
      scatterView.classList.toggle("active", showScatter);
      heatmapView.classList.toggle("active", showHeatmap);
      contribView.classList.toggle("active", showContrib);
      errorsView.classList.toggle("active", showErrors);
      tabScatter.classList.toggle("active", showScatter);
      tabHeatmap.classList.toggle("active", showHeatmap);
      tabContrib.classList.toggle("active", showContrib);
      tabErrors.classList.toggle("active", showErrors);
      buildPlotCategoryFilter();
      updatePlot();
      updateHeatmaps();
      updateContribution();
      updateErrorPlots();
    }}

    function isNumber(value) {{
      return typeof value === "number" && !Number.isNaN(value);
    }}

    function isExcludeWebSearch() {{
      const checkbox = document.getElementById("excludeWebSearch");
      return checkbox ? checkbox.checked : false;
    }}

    function getOverallRowValue(row, condition) {{
      if (!isExcludeWebSearch()) {{
        return row[condition];
      }}
      const key = `${{condition}}_excl_web`;
      return isNumber(row[key]) ? row[key] : row[condition];
    }}

    function getSummaryTableByKey(tableKey) {{
      return categoryTablesSummary.find((table) => table.key === tableKey) || null;
    }}

    function getAgenticMemorySummary(model, condition) {{
      const memoryTable = getSummaryTableByKey("data_agentic-memory");
      if (!memoryTable) {{
        return null;
      }}
      const row = memoryTable.rows.find((item) => item.model === model);
      if (!row) {{
        return null;
      }}
      const metricValues = row.metrics["Memory Summary"] || {{}};
      return metricValues[condition];
    }}

    function getMetricDisplayLabel(metric, table) {{
      if (
        isExcludeWebSearch() &&
        table.title.toLowerCase().includes("agentic") &&
        String(metric).toLowerCase().includes("agentic overall")
      ) {{
        return "Agentic (memory-only)";
      }}
      return String(metric || "");
    }}

    function getMetricDisplayValue(row, metric, table, condition) {{
      if (
        isExcludeWebSearch() &&
        table.title.toLowerCase().includes("agentic") &&
        String(metric).toLowerCase().includes("agentic overall")
      ) {{
        return getAgenticMemorySummary(row.model, condition);
      }}
      const metricValues = row.metrics[metric] || {{}};
      return metricValues[condition];
    }}

    function updateOverallLabel() {{
      const label = document.getElementById("overallLabel");
      if (!label) {{
        return;
      }}
      if (isExcludeWebSearch()) {{
        label.textContent = "Overall (excl. web_search)";
        label.style.display = "";
      }} else {{
        label.textContent = "";
        label.style.display = "none";
      }}
    }}

    function toFixed(value, digits) {{
      return isNumber(value) ? value.toFixed(digits) : MISSING;
    }}

    function formatDelta(acc, baseline, mode) {{
      if (!isNumber(acc) || !isNumber(baseline)) {{
        return MISSING;
      }}
      if (mode === "rel") {{
        if (baseline === 0) {{
          return MISSING;
        }}
        const rel = ((acc - baseline) / baseline) * 100;
        return (rel >= 0 ? "+" : "") + rel.toFixed(2) + "%";
      }}
      const delta = acc - baseline;
      return (delta >= 0 ? "+" : "") + delta.toFixed(2);
    }}

    function computeDeltaValue(acc, baseline, mode) {{
      if (!isNumber(acc) || !isNumber(baseline)) {{
        return null;
      }}
      if (mode === "rel") {{
        if (baseline === 0) {{
          return null;
        }}
        return ((acc - baseline) / baseline) * 100;
      }}
      return acc - baseline;
    }}

    function buildTableIndex(tables) {{
      const index = {{}};
      tables.forEach((table) => {{
        const modelMap = {{}};
        table.rows.forEach((row) => {{
          modelMap[row.model] = row.metrics;
        }});
        index[table.key] = modelMap;
      }});
      return index;
    }}

    const tableIndexFull = buildTableIndex(categoryTablesFull);
    const tableIndexSummary = buildTableIndex(categoryTablesSummary);

    function buildTableSelect() {{
      const select = document.getElementById("tableSelect");
      const tables = getCategoryTables();
      select.innerHTML = "";
      const mainOption = document.createElement("option");
      mainOption.value = "main";
      mainOption.textContent = "Main";
      select.appendChild(mainOption);
      const allOption = document.createElement("option");
      allOption.value = "all";
      allOption.textContent = "All";
      select.appendChild(allOption);
      tables.forEach((table) => {{
        const option = document.createElement("option");
        option.value = table.key;
        option.textContent = table.title;
        select.appendChild(option);
      }});
      if (!Array.from(select.options).some((option) => option.value === select.value)) {{
        select.value = "main";
      }}
    }}

    function buildPlotCategoryFilter() {{
      const select = document.getElementById("plotCategoryFilter");
      if (!select) {{
        return;
      }}
      const tables = getCategoryTables();
      const selected = select.value || "all";
      select.innerHTML = "";
      const allOption = document.createElement("option");
      allOption.value = "all";
      allOption.textContent = "All";
      select.appendChild(allOption);
      tables.forEach((table) => {{
        const option = document.createElement("option");
        option.value = table.key;
        option.textContent = table.title;
        select.appendChild(option);
      }});
      if (Array.from(select.options).some((option) => option.value === selected)) {{
        select.value = selected;
      }}
    }}

    function buildModelFilter() {{
      const select = document.getElementById("modelFilter");
      select.innerHTML = "";
      allModels.forEach((model) => {{
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        option.selected = true;
        select.appendChild(option);
      }});
    }}

    function getSelectedModels() {{
      const select = document.getElementById("modelFilter");
      const values = Array.from(select.selectedOptions).map((option) => option.value);
      if (values.length === 0) {{
        return new Set(allModels);
      }}
      return new Set(values);
    }}

    function renderMainTable() {{
      const body = document.getElementById("tableBody");
      const mode = document.getElementById("deltaMode").value;
      const selected = getSelectedModels();
      body.innerHTML = "";
      const tableKey = "main";
      const sortedRows = getSortedRows(tableKey, mainRowsAggregated);
      sortedRows.forEach((row) => {{
        if (!selected.has(row.model)) {{
          return;
        }}
        const tr = document.createElement("tr");
        const baseline = getOverallRowValue(row, "OO");
        const values = conditions.map((condition) => getOverallRowValue(row, condition));
        const bestAcc = Math.max(...values.filter((value) => isNumber(value)));

        function buildCell(label, value, isBaseline, condition) {{
          const td = document.createElement("td");
          if (label === "Model") {{
            td.textContent = value ?? MISSING;
            return td;
          }}
          const accText = toFixed(value, 2);
          if (condition === "OO") {{
            td.textContent = accText;
          }} else {{
            const deltaText = formatDelta(value, baseline, mode);
            td.textContent = accText + " (" + deltaText + ")";
          }}
          if (isBaseline) {{
            td.classList.add("baseline");
          }}
          if (isNumber(value) && isNumber(bestAcc) && value === bestAcc) {{
            td.classList.add("best");
          }}
          return td;
        }}

        tr.appendChild(buildCell("Model", row.model, false, "Model"));
        tr.appendChild(buildCell("OO", getOverallRowValue(row, "OO"), true, "OO"));
        tr.appendChild(buildCell("OA", getOverallRowValue(row, "OA"), false, "OA"));
        tr.appendChild(buildCell("AO", getOverallRowValue(row, "AO"), false, "AO"));
        tr.appendChild(buildCell("AA", getOverallRowValue(row, "AA"), false, "AA"));
        body.appendChild(tr);
      }});
    }}

    function renderMainOverallChart() {{
      const container = document.getElementById("main_overall_chart");
      if (!container || typeof Plotly === "undefined") {{
        return;
      }}
      const selectedModels = Array.from(getSelectedModels());
      const rowIndex = {{}};
      mainRowsAggregated.forEach((row) => {{
        rowIndex[row.model] = row;
      }});
      const models = selectedModels.filter((model) => rowIndex[model]);
      container.innerHTML = "";
      if (!models.length) {{
        const empty = document.createElement("div");
        empty.className = "note";
        empty.textContent = "No data available for the current selection.";
        container.appendChild(empty);
        return;
      }}

      const useExcl = isExcludeWebSearch();
      const traces = conditions.map((condition) => {{
        const y = [];
        const plus = [];
        const minus = [];
        models.forEach((model) => {{
          const row = rowIndex[model];
          const value = getOverallRowValue(row, condition);
          y.push(isNumber(value) ? value : null);
          const minKey = useExcl ? `${{condition}}_excl_web_min` : `${{condition}}_min`;
          const maxKey = useExcl ? `${{condition}}_excl_web_max` : `${{condition}}_max`;
          const minVal = row[minKey];
          const maxVal = row[maxKey];
          if (isNumber(value) && isNumber(minVal) && isNumber(maxVal)) {{
            plus.push(Math.max(0, maxVal - value));
            minus.push(Math.max(0, value - minVal));
          }} else {{
            plus.push(0);
            minus.push(0);
          }}
        }});
        const hasError = plus.some((v) => v > 0) || minus.some((v) => v > 0);
        return {{
          name: condition,
          type: "bar",
          x: models,
          y,
          error_y: {{
            type: "data",
            array: plus,
            arrayminus: minus,
            visible: hasError,
            color: "#2f3a56",
            thickness: 1
          }},
        }};
      }});

      const layout = {{
        barmode: "group",
        margin: {{ t: 20, l: 60, r: 20, b: 120 }},
        xaxis: {{ tickangle: -30 }},
        yaxis: {{ title: "Overall Acc" }},
        legend: {{ orientation: "h", y: -0.2 }}
      }};
      Plotly.newPlot(container, traces, layout, {{ displayModeBar: false }});
    }}

    function getSortValueMain(row, key, mode) {{
      if (key === "model") {{
        return row.model ? row.model.toLowerCase() : "";
      }}
      if (key.endsWith("_delta")) {{
        const condition = key.split("_")[0];
        if (condition === "OO") {{
          return 0;
        }}
        const acc = getOverallRowValue(row, condition);
        const baseline = getOverallRowValue(row, "OO");
        if (!isNumber(acc) || !isNumber(baseline)) {{
          return Number.NEGATIVE_INFINITY;
        }}
        if (mode === "rel") {{
          return baseline === 0 ? Number.NEGATIVE_INFINITY : ((acc - baseline) / baseline) * 100;
        }}
        return acc - baseline;
      }}
      const value = getOverallRowValue(row, key);
      if (isNumber(value)) {{
        return value;
      }}
      return Number.NEGATIVE_INFINITY;
    }}

    function getSortValueDetail(row, sortSpec, table) {{
      if (sortSpec.key === "model") {{
        return row.model ? row.model.toLowerCase() : "";
      }}
      const metric = sortSpec.metric;
      const condition = sortSpec.condition;
      const value = table
        ? getMetricDisplayValue(row, metric, table, condition)
        : (row.metrics[metric] || {{}})[condition];
      if (isNumber(value)) {{
        return value;
      }}
      return Number.NEGATIVE_INFINITY;
    }}

    function getSortedRows(tableKey, rows) {{
      const mode = document.getElementById("deltaMode").value;
      const state = tableState[tableKey];
      const sortKey = state ? state.sortKey : null;
      const sortDir = state ? state.sortDir : null;
      const multiplier = sortDir === "asc" ? 1 : -1;
      const sortSpec = sortKey ? buildSortSpec(tableKey, sortKey) : null;
      const table = tableKey === "main" ? null : getTableByKey(tableKey);
      const sorted = rows.slice();
      sorted.sort((a, b) => {{
        let av;
        let bv;
        if (sortKey) {{
          if (tableKey === "main") {{
            av = getSortValueMain(a, sortKey, mode);
            bv = getSortValueMain(b, sortKey, mode);
          }} else {{
            av = getSortValueDetail(a, sortSpec, table);
            bv = getSortValueDetail(b, sortSpec, table);
          }}
          if (av > bv) return 1 * multiplier;
          if (av < bv) return -1 * multiplier;
        }}
        return a.model.localeCompare(b.model);
      }});
      return sorted;
    }}

    function buildSortSpec(tableKey, sortKey) {{
      if (tableKey === "main") {{
        return {{ key: sortKey }};
      }}
      const [metric, condition] = sortKey.split("::");
      return {{ key: sortKey, metric, condition }};
    }}

    function getTableByKey(tableKey) {{
      return getCategoryTables().find((table) => table.key === tableKey) || null;
    }}

    function renderDetailTables() {{
      const container = document.getElementById("detailSections");
      const mode = document.getElementById("deltaMode").value;
      const selected = getSelectedModels();
      const selectedTable = document.getElementById("tableSelect").value;
      const tables = getCategoryTables();
      container.innerHTML = "";
      tables.forEach((table) => {{
        if (selectedTable !== "all" && selectedTable !== table.key) {{
          return;
        }}
        const section = document.createElement("div");
        section.className = "section";
        section.id = `section-${{table.key}}`;

        const heading = document.createElement("h2");
        heading.textContent = table.title;
        section.appendChild(heading);

        if (table.missing && table.missing.length) {{
          const note = document.createElement("div");
          note.className = "note";
          note.textContent = `Missing data for conditions: ${{table.missing.join(", ")}}. Showing available results.`;
          section.appendChild(note);
        }}

        const controls = document.createElement("div");
        controls.className = "chart-controls";
        controls.dataset.tableKey = table.key;

        const state = barState[table.key] || {{
          metric: table.metrics[0] || "",
          condition: "AA"
        }};
        if (!table.metrics.includes(state.metric)) {{
          state.metric = table.metrics[0] || "";
        }}
        barState[table.key] = state;

        const metricLabel = document.createElement("label");
        metricLabel.textContent = "Metric";
        const metricSelect = document.createElement("select");
        metricSelect.id = `bar-metric-${{table.key}}`;
          table.metrics.forEach((metric) => {{
            const option = document.createElement("option");
            option.value = metric;
            option.textContent = getMetricDisplayLabel(metric, table);
            metricSelect.appendChild(option);
          }});
        metricSelect.value = state.metric;
        metricLabel.appendChild(metricSelect);
        controls.appendChild(metricLabel);

        const conditionLabel = document.createElement("label");
        conditionLabel.textContent = "Condition";
        const conditionSelect = document.createElement("select");
        conditionSelect.id = `bar-condition-${{table.key}}`;
        ["OA", "AO", "AA"].forEach((condition) => {{
          const option = document.createElement("option");
          option.value = condition;
          option.textContent = `${{condition}}-OO`;
          conditionSelect.appendChild(option);
        }});
        conditionSelect.value = state.condition;
        conditionLabel.appendChild(conditionSelect);
        controls.appendChild(conditionLabel);

        section.appendChild(controls);

        const chartCard = document.createElement("div");
        chartCard.className = "chart-card";
        const chartTitle = document.createElement("div");
        chartTitle.className = "chart-title";
        chartTitle.textContent = "Category Delta Bars";
        chartCard.appendChild(chartTitle);
        const chartContainer = document.createElement("div");
        chartContainer.id = `bar-container-${{table.key}}`;
        chartCard.appendChild(chartContainer);
        section.appendChild(chartCard);

        const wrap = document.createElement("div");
        wrap.className = "table-wrap";
        const tableEl = document.createElement("table");
        tableEl.dataset.tableKey = table.key;

        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        const modelTh = document.createElement("th");
        modelTh.textContent = "Model";
        modelTh.classList.add("sortable");
        modelTh.dataset.sortKey = "model";
        headerRow.appendChild(modelTh);
          table.metrics.forEach((metric) => {{
            const displayLabel = getMetricDisplayLabel(metric, table);
            const metricLower = displayLabel.toLowerCase();
            if (
              isExcludeWebSearch() &&
              table.title.toLowerCase().includes("agentic") &&
              metricLower.includes("web search")
            ) {{
              return;
            }}
            const isOverall = displayLabel.toLowerCase().includes("overall");
            conditions.forEach((condition) => {{
              const th = document.createElement("th");
              th.classList.add("sortable");
              th.dataset.sortKey = `${{metric}}::${{condition}}`;
              th.textContent = `${{displayLabel}} ${{condition}}`;
              if (isOverall) {{
                th.classList.add("overall-header");
              }}
              headerRow.appendChild(th);
            }});
        }});
        thead.appendChild(headerRow);
        tableEl.appendChild(thead);

        const tbody = document.createElement("tbody");
        const tableKey = table.key;
        const sortedRows = getSortedRows(tableKey, table.rows);
        sortedRows.forEach((row) => {{
          if (!selected.has(row.model)) {{
            return;
          }}
          const tr = document.createElement("tr");
          const modelTd = document.createElement("td");
          modelTd.textContent = row.model ?? MISSING;
          tr.appendChild(modelTd);
            table.metrics.forEach((metric) => {{
              const displayLabel = getMetricDisplayLabel(metric, table);
              const metricLower = displayLabel.toLowerCase();
              if (
                isExcludeWebSearch() &&
                table.title.toLowerCase().includes("agentic") &&
                metricLower.includes("web search")
              ) {{
                return;
              }}
              const bestAcc = Math.max(
                ...conditions
                  .map((condition) => getMetricDisplayValue(row, metric, table, condition))
                  .filter((value) => isNumber(value))
              );
              conditions.forEach((condition) => {{
                const value = getMetricDisplayValue(row, metric, table, condition);
                const td = document.createElement("td");
                const baseline = getMetricDisplayValue(row, metric, table, "OO");
                const accText = toFixed(value, 2);
                if (condition === "OO") {{
                  td.textContent = accText;
                  td.classList.add("baseline");
              }} else {{
                const deltaText = formatDelta(value, baseline, mode);
                td.textContent = accText + " (" + deltaText + ")";
              }}
              if (isNumber(value) && isNumber(bestAcc) && value === bestAcc) {{
                td.classList.add("best");
              }}
              tr.appendChild(td);
            }});
          }});
          tbody.appendChild(tr);
        }});
        tableEl.appendChild(tbody);
        wrap.appendChild(tableEl);
        section.appendChild(wrap);
        container.appendChild(section);
        updateSortIndicators(tableEl, table.key);
      }});
    }}

    function renderAllTables() {{
      const selectedTable = document.getElementById("tableSelect").value;
      const mainSection = document.querySelector(".table-wrap");
      const mainChart = document.getElementById("mainOverallSection");
      const note = document.getElementById("note");
      if (selectedTable === "main" || selectedTable === "all") {{
        mainSection.style.display = "";
        if (mainChart) {{
          mainChart.style.display = "";
        }}
        note.style.display = "";
        renderMainTable();
        renderMainOverallChart();
        const mainTable = document.querySelector("table[data-table-key='main']");
        if (mainTable) {{
          updateSortIndicators(mainTable, "main");
        }}
      }} else {{
        mainSection.style.display = "none";
        if (mainChart) {{
          mainChart.style.display = "none";
        }}
        note.style.display = "none";
      }}
      renderDetailTables();
      updatePlot();
      updateHeatmaps();
      updateContribution();
      updateErrorPlots();
      updateOverallLabel();
    }}

    function bindControls() {{
      document.getElementById("deltaMode").addEventListener("change", () => {{
        renderAllTables();
        attachHeaderSorting();
        renderBarCharts();
        updatePlot();
        updateErrorPlots();
      }});
      document.getElementById("detailMode").addEventListener("change", () => {{
        buildTableSelect();
        buildPlotCategoryFilter();
        renderAllTables();
        attachHeaderSorting();
        renderBarCharts();
        updatePlot();
        updateErrorPlots();
      }});
      document.getElementById("tableSelect").addEventListener("change", (event) => {{
        buildPlotCategoryFilter();
        renderAllTables();
        attachHeaderSorting();
        renderBarCharts();
        updatePlot();
        updateErrorPlots();
      }});
      document.getElementById("modelFilter").addEventListener("change", () => {{
        renderAllTables();
        attachHeaderSorting();
        renderBarCharts();
        updatePlot();
        updateErrorPlots();
      }});
      document.getElementById("selectAllModels").addEventListener("click", () => {{
        const select = document.getElementById("modelFilter");
        Array.from(select.options).forEach((option) => {{
          option.selected = true;
        }});
        renderAllTables();
        attachHeaderSorting();
        renderBarCharts();
        updatePlot();
        updateErrorPlots();
      }});
      document.getElementById("clearModels").addEventListener("click", () => {{
        const select = document.getElementById("modelFilter");
        Array.from(select.options).forEach((option) => {{
          option.selected = false;
        }});
        renderAllTables();
        attachHeaderSorting();
        renderBarCharts();
        updatePlot();
        updateErrorPlots();
      }});
      document.getElementById("plotMetricMode").addEventListener("change", () => {{
        updatePlot();
        updateErrorPlots();
      }});
      document.getElementById("plotCategoryFilter").addEventListener("change", () => {{
        updatePlot();
        updateErrorPlots();
      }});
      document.getElementById("excludeWebSearch").addEventListener("change", () => {{
        renderAllTables();
        attachHeaderSorting();
        renderBarCharts();
        updatePlot();
        updateHeatmaps();
        updateErrorPlots();
      }});
      document.getElementById("heatmapCombo").addEventListener("change", () => {{
        updateHeatmaps();
        updateErrorPlots();
      }});
      document.querySelectorAll(".plot-combo").forEach((el) => {{
        el.addEventListener("change", () => {{
          updatePlot();
          updateErrorPlots();
        }});
      }});
      document.getElementById("plotTabScatter").addEventListener("click", () => {{
        setPlotView("scatter");
      }});
      document.getElementById("plotTabHeatmap").addEventListener("click", () => {{
        setPlotView("heatmap");
      }});
      document.getElementById("plotTabContribution").addEventListener("click", () => {{
        setPlotView("contrib");
      }});
      document.getElementById("plotTabErrors").addEventListener("click", () => {{
        setPlotView("errors");
      }});
      document.getElementById("tabResults").addEventListener("click", () => {{
        setActiveTab("results");
      }});
      document.getElementById("tabPlots").addEventListener("click", () => {{
        setActiveTab("plots");
      }});
      document.getElementById("errorMixMode").addEventListener("change", () => {{
        updateErrorMix();
      }});
      document.getElementById("errorHeatmapCombo").addEventListener("change", () => {{
        updateErrorHeatmap();
      }});
      document.getElementById("errorBucketMode").addEventListener("change", () => {{
        updateErrorBuckets();
      }});
      document.getElementById("errorShiftCombo").addEventListener("change", () => {{
        updateErrorShifts();
      }});
      document.getElementById("errorTabMix").addEventListener("click", () => {{
        setErrorView("mix");
      }});
      document.getElementById("errorTabHeatmap").addEventListener("click", () => {{
        setErrorView("heatmap");
      }});
      document.getElementById("errorTabBuckets").addEventListener("click", () => {{
        setErrorView("buckets");
      }});
      document.getElementById("errorTabShifts").addEventListener("click", () => {{
        setErrorView("shifts");
      }});
    }}

    function updateSortIndicators(tableElement, tableKey) {{
      const state = tableState[tableKey] || {{ sortKey: null, sortDir: null }};
      tableElement.querySelectorAll("th.sortable").forEach((th) => {{
        const indicator = th.querySelector(".sort-indicator");
        if (indicator) {{
          indicator.remove();
        }}
        if (state.sortKey && th.dataset.sortKey === state.sortKey) {{
          const span = document.createElement("span");
          span.className = "sort-indicator";
          span.textContent = state.sortDir === "asc" ? "▲" : "▼";
          th.appendChild(span);
        }}
      }});
    }}

    function attachHeaderSorting() {{
      document.querySelectorAll("table").forEach((table) => {{
        table.querySelectorAll("th.sortable").forEach((th) => {{
          if (th.dataset.bound === "1") {{
            return;
          }}
          th.dataset.bound = "1";
          th.addEventListener("click", () => {{
            const tableKey = table.dataset.tableKey || "main";
            const key = th.dataset.sortKey;
            const state = tableState[tableKey] || {{ sortKey: null, sortDir: null }};
            if (state.sortKey !== key) {{
              state.sortKey = key;
              state.sortDir = "asc";
            }} else if (state.sortDir === "asc") {{
              state.sortDir = "desc";
            }} else {{
              state.sortKey = null;
              state.sortDir = null;
            }}
            tableState[tableKey] = state;
            renderAllTables();
            attachHeaderSorting();
          }});
        }});
      }});
    }}

    function renderMainHeader() {{
      const headerRow = document.querySelector("table thead tr");
      if (!headerRow) {{
        return;
      }}
      headerRow.querySelectorAll("th").forEach((th, index) => {{
        if (index === 0) {{
          th.classList.add("sortable");
          th.dataset.sortKey = "model";
        }} else {{
          const condition = conditions[index - 1];
          th.classList.add("sortable");
          th.dataset.sortKey = condition;
        }}
      }});
    }}

    function renderAllTablesWithHeaders() {{
      renderMainHeader();
      renderAllTables();
      attachHeaderSorting();
      renderBarCharts();
      updatePlot();
      updateHeatmaps();
      updateContribution();
      updateErrorPlots();
    }}

    function getCategoryTables() {{
      const mode = document.getElementById("detailMode").value;
      return mode === "summary" ? categoryTablesSummary : categoryTablesFull;
    }}

    function getCategoryIndex() {{
      const mode = document.getElementById("detailMode").value;
      return mode === "summary" ? tableIndexSummary : tableIndexFull;
    }}

    function normalizeAcc(value) {{
      if (!isNumber(value)) {{
        return null;
      }}
      return value > 1.5 ? value / 100 : value;
    }}

    function getOverallComponents(model, condition) {{
      if (!overallComponents || !overallComponents[condition]) {{
        return null;
      }}
      return overallComponents[condition][model] || null;
    }}

    const ERROR_TYPES_ORDER = [
      "WRONG_TOOL",
      "HALLUCINATED_TOOL",
      "SHOULD_ABSTAIN_BUT_CALLED",
      "SHOULD_CALL_BUT_ABSTAINED",
      "MISSING_REQUIRED_ARGS",
      "EXTRA_ARGS",
      "TYPE_MISMATCH",
      "VALUE_FORMAT_ERROR",
      "WRONG_ARGS",
      "INVALID_JSON_OR_SCHEMA",
      "OTHER"
    ];
    const TOP_SHIFT_K = {TOP_ERROR_SHIFT_K};

    const ROOT_CAUSE_MAP = {{
      WRONG_TOOL: "TOOL_DOC_UNDERSTANDING",
      HALLUCINATED_TOOL: "TOOL_DOC_UNDERSTANDING",
      MISSING_REQUIRED_ARGS: "TOOL_DOC_UNDERSTANDING",
      EXTRA_ARGS: "TOOL_DOC_UNDERSTANDING",
      TYPE_MISMATCH: "TOOL_DOC_UNDERSTANDING",
      VALUE_FORMAT_ERROR: "TOOL_DOC_UNDERSTANDING",
      WRONG_ARGS: "TOOL_DOC_UNDERSTANDING",
      SHOULD_ABSTAIN_BUT_CALLED: "USER_INTENT_UNDERSTANDING",
      SHOULD_CALL_BUT_ABSTAINED: "USER_INTENT_UNDERSTANDING",
      INVALID_JSON_OR_SCHEMA: "OTHER",
      OTHER: "OTHER"
    }};

    const ROOT_CAUSE_ORDER = [
      "TOOL_DOC_UNDERSTANDING",
      "USER_INTENT_UNDERSTANDING",
      "STATE_ENV_UNDERSTANDING",
      "OTHER"
    ];

    function buildErrorIndex() {{
      const index = {{}};
      (errorRecords || []).forEach((rec) => {{
        const condition = rec.condition;
        const model = rec.model;
        const testCategory = rec.test_category;
        if (!index[condition]) {{
          index[condition] = {{}};
        }}
        if (!index[condition][model]) {{
          index[condition][model] = {{}};
        }}
        index[condition][model][testCategory] = rec;
      }});
      return index;
    }}

    const errorIndex = buildErrorIndex();

    function isWebSearchCategory(testCategory) {{
      return String(testCategory || "").startsWith("web_search");
    }}

    function isWebSearchLabel(label) {{
      const lowered = String(label || "").toLowerCase();
      return (
        lowered.includes("web search") ||
        lowered.includes("websearch") ||
        lowered.includes("web_search")
      );
    }}

    function getSelectedTestCategories() {{
      const selectedTable = document.getElementById("tableSelect").value;
      const tableKey = selectedTable === "main" ? "all" : selectedTable;
      const tables = getCategoryTables();
      const selected = new Set();
      tables.forEach((table) => {{
        if (tableKey !== "all" && tableKey !== table.key) {{
          return;
        }}
        const groups = errorGroups[table.key] || {{}};
        Object.values(groups).forEach((group) => {{
          group.forEach((testCategory) => selected.add(testCategory));
        }});
      }});
      let list = Array.from(selected);
      if (isExcludeWebSearch()) {{
        list = list.filter((testCategory) => !isWebSearchCategory(testCategory));
      }}
      return list;
    }}

    function aggregateErrorsForCategories(model, condition, testCategories) {{
      let total = 0;
      let incorrect = 0;
      const errorCounts = {{}};
      testCategories.forEach((testCategory) => {{
        const byCondition = errorIndex && errorIndex[condition];
        const byModel = byCondition && byCondition[model];
        const rec = byModel && byModel[testCategory] ? byModel[testCategory] : null;
        if (!rec) {{
          return;
        }}
        total += rec.total || 0;
        incorrect += rec.incorrect || 0;
        const errors = rec.errors || {{}};
        Object.keys(errors).forEach((key) => {{
          errorCounts[key] = (errorCounts[key] || 0) + errors[key];
        }});
      }});
      return {{ total, incorrect, errorCounts }};
    }}

    function setErrorView(viewId) {{
      const mixView = document.getElementById("error_mix_view");
      const heatmapView = document.getElementById("error_heatmap_view");
      const bucketView = document.getElementById("error_bucket_view");
      const shiftView = document.getElementById("error_shift_view");
      const tabMix = document.getElementById("errorTabMix");
      const tabHeatmap = document.getElementById("errorTabHeatmap");
      const tabBuckets = document.getElementById("errorTabBuckets");
      const tabShifts = document.getElementById("errorTabShifts");
      if (
        !mixView ||
        !heatmapView ||
        !bucketView ||
        !shiftView ||
        !tabMix ||
        !tabHeatmap ||
        !tabBuckets ||
        !tabShifts
      ) {{
        return;
      }}
      const showMix = viewId === "mix";
      const showHeatmap = viewId === "heatmap";
      const showBuckets = viewId === "buckets";
      const showShifts = viewId === "shifts";
      mixView.classList.toggle("active", showMix);
      heatmapView.classList.toggle("active", showHeatmap);
      bucketView.classList.toggle("active", showBuckets);
      shiftView.classList.toggle("active", showShifts);
      tabMix.classList.toggle("active", showMix);
      tabHeatmap.classList.toggle("active", showHeatmap);
      tabBuckets.classList.toggle("active", showBuckets);
      tabShifts.classList.toggle("active", showShifts);
      updateErrorPlots();
    }}

    function updatePlot() {{
      const plot = document.getElementById("plot_scatter");
      const warning = document.getElementById("plotWarning");
      if (!plot || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      const scatterView = document.getElementById("plots_scatter_view");
      if (scatterView && !scatterView.classList.contains("active")) {{
        return;
      }}
      const selectedModels = Array.from(getSelectedModels());
      plot.innerHTML = "";
      if (selectedModels.length === 0) {{
        warning.style.display = "";
        warning.textContent = "Select at least one model to view the scatter plot.";
        return;
      }}
      const selectedTable = document.getElementById("tableSelect").value;
      const tables = getCategoryTables();
      const metricMode = document.getElementById("plotMetricMode").value;
      const categoryFilter = document.getElementById("plotCategoryFilter").value;
      const enabledCombos = new Set(
        Array.from(document.querySelectorAll(".plot-combo:checked")).map((el) => el.value)
      );
      if (enabledCombos.size === 0) {{
        warning.style.display = "";
        warning.textContent = "Select at least one combo to view the scatter plot.";
        return;
      }}
      warning.style.display = "none";
      selectedModels.forEach((model) => {{
        const points = {{ OA: [], AO: [], AA: [] }};
        tables.forEach((table) => {{
          if (selectedTable !== "all" && selectedTable !== table.key) {{
            return;
          }}
          if (categoryFilter !== "all" && categoryFilter !== table.key) {{
            return;
          }}
          if (isWebSearchLabel(table.title)) {{
            return;
          }}
          table.rows.forEach((row) => {{
            if (row.model !== model) {{
              return;
            }}
            table.metrics.forEach((metric) => {{
              const metricLabel = getMetricDisplayLabel(metric, table);
              if (isWebSearchLabel(metricLabel)) {{
                return;
              }}
              const oo = normalizeAcc(getMetricDisplayValue(row, metric, table, "OO"));
              if (!isNumber(oo)) {{
                return;
              }}
              ["OA", "AO", "AA"].forEach((condition) => {{
                if (!enabledCombos.has(condition)) {{
                  return;
                }}
                const combo = normalizeAcc(getMetricDisplayValue(row, metric, table, condition));
                if (!isNumber(combo)) {{
                  return;
                }}
                const delta = combo - oo;
                const y = metricMode === "delta" ? delta : combo;
                points[condition].push({{
                  x: oo,
                  y,
                  meta: [condition, table.title, metricLabel, oo, combo, delta]
                }});
              }});
            }});
          }});
        }});

        const totalPoints = Object.values(points).reduce((sum, arr) => sum + arr.length, 0);
        if (!totalPoints) {{
          return;
        }}

        const container = document.createElement("div");
        container.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        container.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "420px";
        container.appendChild(plotDiv);
        plot.appendChild(container);

        const traces = ["OA", "AO", "AA"]
          .filter((condition) => enabledCombos.has(condition))
          .map((condition) => {{
            const tracePoints = points[condition];
            const sortedPos = [...tracePoints].sort((a, b) => b.y - a.y);
            const sortedNeg = [...tracePoints].sort((a, b) => a.y - b.y);
            const posCutoff = sortedPos.length >= 3 ? sortedPos[2].y : null;
            const negCutoff = sortedNeg.length >= 3 ? sortedNeg[2].y : null;
            const posSet = new Set(
              sortedPos.filter((pt) => posCutoff === null || pt.y >= posCutoff)
            );
            const negSet = new Set(
              sortedNeg.filter((pt) => negCutoff === null || pt.y <= negCutoff)
            );
            const colors = tracePoints.map((pt) => {{
              if (posSet.has(pt)) return "#2f852f";
              if (negSet.has(pt)) return "#c43d3d";
              return "#2b6cb0";
            }});
            return {{
              name: condition,
              mode: "markers",
              type: "scatter",
              x: tracePoints.map((pt) => pt.x),
              y: tracePoints.map((pt) => pt.y),
              marker: {{ size: 10, color: colors, opacity: 0.9 }},
              customdata: tracePoints.map((pt) => pt.meta),
              hovertemplate:
                "Combo: %{{customdata[0]}}<br>" +
                "Category: %{{customdata[1]}}<br>" +
                "Row: %{{customdata[2]}}<br>" +
                "OO: %{{customdata[3]:.3f}}<br>" +
                "Combo: %{{customdata[4]:.3f}}<br>" +
                "Delta: %{{customdata[5]:.3f}}<extra></extra>"
            }};
          }});

        const accuracyLayout = {{
          xaxis: {{ title: "OO accuracy", range: [0, 1] }},
          yaxis: {{ title: "Combo accuracy", range: [0, 1] }},
          shapes: [
            {{
              type: "line",
              x0: 0,
              y0: 0,
              x1: 1,
              y1: 1,
              line: {{ color: "#8892b0", width: 1, dash: "dot" }}
            }}
          ]
        }};
        const deltaLayout = {{
          xaxis: {{ title: "OO accuracy", range: [0, 1] }},
          yaxis: {{ title: "Combo - OO", range: [-1, 1] }},
          shapes: [
            {{
              type: "line",
              x0: 0,
              y0: 0,
              x1: 1,
              y1: 0,
              line: {{ color: "#8892b0", width: 1, dash: "dot" }}
            }}
          ]
        }};

        const layout = {{
          margin: {{ t: 20, l: 40, r: 20, b: 40 }},
          legend: {{ orientation: "h" }},
          ...(metricMode === "delta" ? deltaLayout : accuracyLayout)
        }};
        Plotly.newPlot(plotDiv, traces, layout, {{ displayModeBar: false }});
      }});
    }}

    function updateContribution() {{
      const container = document.getElementById("plot_contrib");
      const warning = document.getElementById("contribWarning");
      if (!container || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      const contribView = document.getElementById("plots_contrib_view");
      if (contribView && !contribView.classList.contains("active")) {{
        return;
      }}
      const selectedModels = Array.from(getSelectedModels());
      container.innerHTML = "";
      if (selectedModels.length === 0) {{
        warning.style.display = "";
        warning.textContent =
          "Select at least one model to view contribution breakdown.";
        return;
      }}
      warning.style.display = "none";

      const useMemoryOnly = isExcludeWebSearch();
      const modeLabel = useMemoryOnly
        ? "overall excludes web_search"
        : "overall includes web_search";
      const components = [
        {{ key: "non_live", label: "Non-live", weight: 0.1 }},
        {{ key: "live", label: "Live", weight: 0.1 }},
        {{ key: "irrelevance", label: "Irrelevance", weight: 0.1 }},
        {{ key: "multi_turn", label: "Multi-turn", weight: 0.3 }},
        {{
          key: useMemoryOnly ? "memory" : "agentic",
          label: useMemoryOnly ? "Agentic (memory-only)" : "Agentic",
          weight: 0.4
        }}
      ];

      selectedModels.forEach((model) => {{
        const oo = getOverallComponents(model, "OO");
        if (!oo) {{
          return;
        }}
        const combos = ["OA", "AO", "AA"];
        const traces = components.map((component) => {{
          const y = [];
          const custom = [];
          combos.forEach((combo) => {{
            const comboComp = getOverallComponents(model, combo);
            const ooVal = normalizeAcc(oo[component.key]);
            const comboVal = normalizeAcc(comboComp ? comboComp[component.key] : null);
            let delta = null;
            let contrib = null;
            if (isNumber(ooVal) && isNumber(comboVal)) {{
              delta = comboVal - ooVal;
              contrib = delta * component.weight;
            }}
            y.push(contrib);
            custom.push({{
              component: component.label,
              weight: component.weight,
              oo: ooVal,
              combo: comboVal,
              delta,
              contrib,
              modeLabel
            }});
          }});
          return {{
            name: component.label,
            type: "bar",
            x: combos,
            y,
            customdata: custom,
            hovertemplate:
              "Component: %{{customdata.component}}<br>" +
              "Weight: %{{customdata.weight:.2f}}<br>" +
              "OO: %{{customdata.oo:.3f}}<br>" +
              "Combo: %{{customdata.combo:.3f}}<br>" +
              "Delta: %{{customdata.delta:.3f}}<br>" +
              "Contribution: %{{customdata.contrib:.3f}}<br>" +
              "%{{customdata.modeLabel}}<extra></extra>"
          }};
        }});

        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "420px";
        section.appendChild(plotDiv);
        container.appendChild(section);

        const layout = {{
          barmode: "relative",
          margin: {{ t: 20, l: 50, r: 20, b: 40 }},
          shapes: [
            {{
              type: "line",
              x0: -0.5,
              x1: 2.5,
              y0: 0,
              y1: 0,
              line: {{ color: "#8892b0", width: 1, dash: "dot" }}
            }}
          ],
          yaxis: {{ title: "Contribution (pp)" }},
          xaxis: {{ title: "Combo" }}
        }};
        Plotly.newPlot(plotDiv, traces, layout, {{ displayModeBar: false }});
      }});
    }}

    function updateErrorPlots() {{
      const errorsView = document.getElementById("plots_errors_view");
      if (errorsView && !errorsView.classList.contains("active")) {{
        return;
      }}
      const mixView = document.getElementById("error_mix_view");
      const heatmapView = document.getElementById("error_heatmap_view");
      const bucketView = document.getElementById("error_bucket_view");
      const shiftView = document.getElementById("error_shift_view");
      if (mixView && mixView.classList.contains("active")) {{
        updateErrorMix();
      }}
      if (heatmapView && heatmapView.classList.contains("active")) {{
        updateErrorHeatmap();
      }}
      if (bucketView && bucketView.classList.contains("active")) {{
        updateErrorBuckets();
      }}
      if (shiftView && shiftView.classList.contains("active")) {{
        updateErrorShifts();
      }}
    }}

    function updateErrorMix() {{
      const container = document.getElementById("error_mix_plot");
      const warning = document.getElementById("errorMixWarning");
      if (!container || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      const selectedModels = Array.from(getSelectedModels());
      container.innerHTML = "";
      if (selectedModels.length === 0) {{
        warning.style.display = "";
        warning.textContent = "Select at least one model to view error mix.";
        return;
      }}
      warning.style.display = "none";
      const testCategories = getSelectedTestCategories();
      if (!testCategories.length) {{
        warning.style.display = "";
        warning.textContent = "No error data for the current selection.";
        return;
      }}

      selectedModels.forEach((model) => {{
        const conditions = ["OO", "OA", "AO", "AA"];
        const conditionAgg = {{}};
        const availableConditions = [];
        conditions.forEach((condition) => {{
          const agg = aggregateErrorsForCategories(model, condition, testCategories);
          conditionAgg[condition] = agg;
          if ((agg && agg.incorrect) || (agg && agg.total)) {{
            availableConditions.push(condition);
          }}
        }});
        if (!availableConditions.length) {{
          return;
        }}
        if (!availableConditions.includes("OO")) {{
          warning.style.display = "";
          warning.textContent = "Missing OO error data for the current selection.";
          return;
        }}

        const errorTypes = Object.keys(
          availableConditions.reduce((acc, condition) => {{
            const errors = conditionAgg[condition].errorCounts || {{}};
            Object.keys(errors).forEach((key) => {{
              acc[key] = true;
            }});
            return acc;
          }}, {{}})
        );
        const avgShares = errorTypes.map((errorType) => {{
          let totalShare = 0;
          let count = 0;
          availableConditions.forEach((condition) => {{
            const agg = conditionAgg[condition];
            const share = agg.incorrect ? (agg.errorCounts[errorType] || 0) / agg.incorrect : 0;
            totalShare += share;
            count += 1;
          }});
          return {{ errorType, avgShare: count ? totalShare / count : 0 }};
        }});
        const topErrors = avgShares
          .filter((row) => row.errorType !== "OTHER")
          .sort((a, b) => b.avgShare - a.avgShare)
          .slice(0, 5)
          .map((row) => row.errorType);
        const orderedErrors = topErrors.concat("OTHER");
        console.log("Error mix labels:", orderedErrors);

        const traces = orderedErrors.map((errorType) => {{
          const y = [];
          const custom = [];
          availableConditions.forEach((condition) => {{
            const agg = conditionAgg[condition];
            let count = 0;
            if (errorType === "OTHER") {{
              count = agg.errorCounts["OTHER"] || 0;
            }} else {{
              count = agg.errorCounts[errorType] || 0;
            }}
            const rawShare = agg.incorrect ? count / agg.incorrect : 0;
            custom.push({{
              errorType,
              count,
              share: rawShare,
              incorrect: agg.incorrect
            }});
            y.push(rawShare);
          }});
          return {{
            name: errorType,
            type: "bar",
            x: availableConditions,
            y,
            customdata: custom,
            hovertemplate:
              "Error: %{{customdata.errorType}}<br>" +
              "Count: %{{customdata.count}}<br>" +
              "Share: %{{customdata.share:.2%}}<br>" +
              "Total incorrect: %{{customdata.incorrect}}<extra></extra>"
          }};
        }});
        const totalsByCondition = availableConditions.map((condition) => {{
          let totalShare = 0;
          traces.forEach((trace) => {{
            const idx = trace.x.indexOf(condition);
            if (idx >= 0) {{
              totalShare += trace.y[idx] || 0;
            }}
          }});
          return totalShare;
        }});
        traces.forEach((trace) => {{
          trace.y = trace.y.map((val, idx) => {{
            const denom = totalsByCondition[idx] || 1;
            return denom ? val / denom : 0;
          }});
        }});

        const totals = availableConditions.map((condition) => conditionAgg[condition].incorrect || 0);
        const annotations = availableConditions.map((condition, idx) => ({{
          x: condition,
          y: 1.04,
          text: "n=" + totals[idx],
          showarrow: false,
          yref: "y",
          xref: "x",
          font: {{ size: 11, color: "#444" }}
        }}));

        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "420px";
        section.appendChild(plotDiv);
        container.appendChild(section);

        const layout = {{
          barmode: "stack",
          margin: {{ t: 60, l: 60, r: 20, b: 40 }},
          yaxis: {{
            title: "Share of incorrect",
            tickformat: ".0%",
            range: [0, 1.08]
          }},
          annotations
        }};
        Plotly.newPlot(plotDiv, traces, layout, {{ displayModeBar: false }});
      }});
    }}

    function updateErrorHeatmap() {{
      const container = document.getElementById("error_heatmap_plot");
      const warning = document.getElementById("errorHeatmapWarning");
      if (!container || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      const TOPK_POS = 10;
      const TOPK_NEG = 10;
      const MIN_ABS_PP = 0.5;
      const selectedModels = Array.from(getSelectedModels());
      container.innerHTML = "";
      if (selectedModels.length === 0) {{
        warning.style.display = "";
        warning.textContent = "Select at least one model to view the error heatmap.";
        return;
      }}
      warning.style.display = "none";
      const selectedCombo = document.getElementById("errorHeatmapCombo").value;
      const tables = getCategoryTables();
      const selectedTable = document.getElementById("tableSelect").value;
      const tableKey = selectedTable === "main" ? "all" : selectedTable;

      selectedModels.forEach((model) => {{
        const rows = [];
        let maxAbs = 0;
        tables.forEach((table) => {{
          if (tableKey !== "all" && tableKey !== table.key) {{
            return;
          }}
          const groups = errorGroups[table.key] || {{}};
          table.metrics.forEach((metric) => {{
            const label = getMetricDisplayLabel(metric, table);
            if (label.toLowerCase().includes("agentic (memory-only)")) {{
              return;
            }}
            const categories = groups[metric] || groups[label] || [];
            const filtered = isExcludeWebSearch()
              ? categories.filter((cat) => !isWebSearchCategory(cat))
              : categories;
            if (!filtered.length) {{
              return;
            }}
            const oo = aggregateErrorsForCategories(model, "OO", filtered);
            const combo = aggregateErrorsForCategories(model, selectedCombo, filtered);
            if (!oo.total || !combo.total) {{
              return;
            }}
            const row = {{
              label,
              category: table.title,
              values: {{}}
            }};
            ERROR_TYPES_ORDER.forEach((errorType) => {{
              const ooRate = oo.total ? (oo.errorCounts[errorType] || 0) / oo.total : 0;
              const comboRate = combo.total
                ? (combo.errorCounts[errorType] || 0) / combo.total
                : 0;
              const delta = (comboRate - ooRate) * 100;
              row.values[errorType] = {{
                ooRate,
                comboRate,
                delta,
                total: oo.total,
                comboTotal: combo.total,
                ooCount: oo.errorCounts[errorType] || 0,
                comboCount: combo.errorCounts[errorType] || 0
              }};
            }});
            rows.push(row);
          }});
        }});

        if (!rows.length) {{
          return;
        }}
        const rowByLabel = {{}};
        rows.forEach((row) => {{
          rowByLabel[row.label] = row;
        }});
        const candidates = [];
        rows.forEach((row) => {{
          ERROR_TYPES_ORDER.forEach((errorType) => {{
            const entry = row.values[errorType];
            if (!entry || !Number.isFinite(entry.delta)) {{
              return;
            }}
            if (Math.abs(entry.delta) < MIN_ABS_PP) {{
              return;
            }}
            candidates.push({{
              row: row.label,
              col: errorType,
              delta: entry.delta
            }});
          }});
        }});
        const pos = candidates
          .filter((item) => item.delta > 0)
          .sort((a, b) => b.delta - a.delta)
          .slice(0, TOPK_POS);
        const neg = candidates
          .filter((item) => item.delta < 0)
          .sort((a, b) => a.delta - b.delta)
          .slice(0, TOPK_NEG);
        const selectedPairs = new Set();
        pos.concat(neg).forEach((item) => {{
          selectedPairs.add(`${{item.row}}|||${{item.col}}`);
        }});
        if (!selectedPairs.size) {{
          const section = document.createElement("div");
          section.className = "section";
          const heading = document.createElement("h2");
          heading.textContent = model;
          section.appendChild(heading);
          const plotDiv = document.createElement("div");
          plotDiv.style.height = "200px";
          section.appendChild(plotDiv);
          container.appendChild(section);
          const layout = {{
            margin: {{ t: 20, l: 20, r: 20, b: 20 }},
            annotations: [
              {{
                text: "No movers above threshold.",
                x: 0.5,
                y: 0.5,
                xref: "paper",
                yref: "paper",
                showarrow: false,
                font: {{ size: 12, color: "#666" }}
              }}
            ]
          }};
          Plotly.newPlot(plotDiv, [], layout, {{ displayModeBar: false }});
          return;
        }}
        const rowScores = {{}};
        const colScores = {{}};
        selectedPairs.forEach((key) => {{
          const [rowLabel, colLabel] = key.split("|||");
          const entry = rowByLabel[rowLabel]?.values[colLabel];
          if (!entry) {{
            return;
          }}
          const absVal = Math.abs(entry.delta);
          if (!rowScores[rowLabel] || absVal > rowScores[rowLabel].abs) {{
            rowScores[rowLabel] = {{ abs: absVal }};
          }}
          if (!colScores[colLabel] || absVal > colScores[colLabel].abs) {{
            colScores[colLabel] = {{ abs: absVal }};
          }}
          maxAbs = Math.max(maxAbs, absVal);
        }});
        if (!maxAbs) {{
          maxAbs = 0.1;
        }}
        const rowStrength = (label) => (rowScores[label]?.abs ?? 0);
        const isMemoryRow = (label) => label.toLowerCase().includes("memory");
        const memoryRows = Object.keys(rowScores)
          .filter((label) => isMemoryRow(label))
          .sort((a, b) => rowStrength(b) - rowStrength(a));
        const otherRows = Object.keys(rowScores)
          .filter((label) => !isMemoryRow(label))
          .sort((a, b) => rowStrength(b) - rowStrength(a));
        const yLabels = memoryRows.concat(otherRows);
        const fixedOrder = ERROR_TYPES_ORDER.slice();
        const xLabels = fixedOrder.filter((label) => Object.prototype.hasOwnProperty.call(colScores, label));
        const xDisplay = xLabels.map((label) => (label === "OTHER" ? "UNCLASSIFIED" : label));
        const z = yLabels.map((rowLabel) =>
          xLabels.map((errorType) => {{
            const entry = rowByLabel[rowLabel]?.values[errorType];
            if (!entry) {{
              return NaN;
            }}
            const key = `${{rowLabel}}|||${{errorType}}`;
            if (!selectedPairs.has(key)) {{
              return NaN;
            }}
            return entry.delta;
          }})
        );
        const custom = yLabels.map((rowLabel) =>
          xLabels.map((errorType) => {{
            const entry = rowByLabel[rowLabel]?.values[errorType];
            const key = `${{rowLabel}}|||${{errorType}}`;
            if (!entry || !selectedPairs.has(key)) {{
              return null;
            }}
            return {{
              label: rowLabel,
              category: rowByLabel[rowLabel]?.category,
              errorType,
              errorDisplay: errorType === "OTHER" ? "UNCLASSIFIED" : errorType,
              ooRate: entry.ooRate,
              comboRate: entry.comboRate,
              delta: entry.delta,
              total: entry.total,
              comboTotal: entry.comboTotal,
              ooCount: entry.ooCount,
              comboCount: entry.comboCount
            }};
          }})
        );

        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "520px";
        section.appendChild(plotDiv);
        container.appendChild(section);

        const trace = {{
          type: "heatmap",
          z,
          x: xDisplay,
          y: yLabels,
          zmid: 0,
          zmin: -maxAbs,
          zmax: maxAbs,
          customdata: custom,
          hoverongaps: false,
          text: z.map((row) =>
            row.map((val) => {{
              if (!Number.isFinite(val) || Math.abs(val) < 0.05) {{
                return "";
              }}
              return `${{val >= 0 ? "+" : ""}}${{val.toFixed(1)}}pp`;
            }})
          ),
          texttemplate: "%{{text}}",
          textfont: {{ size: 14, color: "#222" }},
          hovertemplate:
            "Row: %{{customdata.label}}<br>" +
            "Category: %{{customdata.category}}<br>" +
            "Error: %{{customdata.errorDisplay}}<br>" +
            "OO rate: %{{customdata.ooRate:.2%}}<br>" +
            "Combo rate: %{{customdata.comboRate:.2%}}<br>" +
            "Delta: %{{customdata.delta:+.2f}}pp<br>" +
            "OO errors: %{{customdata.ooCount}} / %{{customdata.total}}<br>" +
            "Combo errors: %{{customdata.comboCount}} / %{{customdata.comboTotal}}<extra></extra>"
        }};
        const layout = {{
          margin: {{ t: 20, l: 120, r: 20, b: 120 }},
          xaxis: {{ tickangle: -30 }},
          yaxis: {{ automargin: true }}
        }};
        Plotly.newPlot(plotDiv, [trace], layout, {{ displayModeBar: false }});
        const caption = document.createElement("div");
        caption.className = "plot-caption";
        caption.textContent = `Cells shown are top movers (|Δ| ≥ ${{MIN_ABS_PP}}pp); others hidden.`;
        section.appendChild(caption);
      }});
      if (!container.childNodes.length) {{
        warning.style.display = "";
        warning.textContent = "No error data for the current selection.";
      }}
    }}

    function updateErrorBuckets() {{
      const container = document.getElementById("error_bucket_plot");
      const warning = document.getElementById("errorBucketWarning");
      if (!container || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      const selectedModels = Array.from(getSelectedModels());
      container.innerHTML = "";
      if (selectedModels.length === 0) {{
        warning.style.display = "";
        warning.textContent = "Select at least one model to view BFCL buckets.";
        return;
      }}
      warning.style.display = "none";
      const testCategories = getSelectedTestCategories();
      if (!testCategories.length) {{
        warning.style.display = "";
        warning.textContent = "No error data for the current selection.";
        return;
      }}
      const mode = document.getElementById("errorBucketMode").value;

      selectedModels.forEach((model) => {{
        const traces = ROOT_CAUSE_ORDER.map((bucket) => {{
          const y = [];
          const custom = [];
          ["OO", "OA", "AO", "AA"].forEach((condition) => {{
            const agg = aggregateErrorsForCategories(model, condition, testCategories);
            let bucketCount = 0;
            Object.entries(agg.errorCounts).forEach(([errorType, count]) => {{
              const mapped = ROOT_CAUSE_MAP[errorType] || "OTHER";
              if (mapped === bucket) {{
                bucketCount += count;
              }}
            }});
            const share = agg.incorrect ? bucketCount / agg.incorrect : 0;
            const rate = agg.total ? bucketCount / agg.total : 0;
            y.push(mode === "share" ? share : rate);
            custom.push({{
              bucket,
              count: bucketCount,
              share,
              rate,
              incorrect: agg.incorrect,
              total: agg.total
            }});
          }});
          return {{
            name: bucket,
            type: "bar",
            x: ["OO", "OA", "AO", "AA"],
            y,
            customdata: custom,
            hovertemplate:
              "Bucket: %{{customdata.bucket}}<br>" +
              "Count: %{{customdata.count}}<br>" +
              "Share: %{{customdata.share:.2%}}<br>" +
              "Rate: %{{customdata.rate:.2%}}<br>" +
              "Total incorrect: %{{customdata.incorrect}}<br>" +
              "Total examples: %{{customdata.total}}<extra></extra>"
          }};
        }});

        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "420px";
        section.appendChild(plotDiv);
        container.appendChild(section);

        const layout = {{
          barmode: "stack",
          margin: {{ t: 20, l: 60, r: 20, b: 40 }},
          yaxis: {{
            title: mode === "share" ? "Share of incorrect" : "Error rate",
            tickformat: ".0%"
          }}
        }};
        Plotly.newPlot(plotDiv, traces, layout, {{ displayModeBar: false }});
      }});
    }}

    function updateErrorShifts() {{
      const container = document.getElementById("error_shift_plot");
      const warning = document.getElementById("errorShiftWarning");
      if (!container || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      const selectedModels = Array.from(getSelectedModels());
      container.innerHTML = "";
      if (selectedModels.length === 0) {{
        warning.style.display = "";
        warning.textContent = "Select at least one model to view top shifts.";
        return;
      }}
      warning.style.display = "none";
      const testCategories = getSelectedTestCategories();
      if (!testCategories.length) {{
        warning.style.display = "";
        warning.textContent = "No error data for the current selection.";
        return;
      }}
      const combo = document.getElementById("errorShiftCombo").value;
      let movers = 0;
      let plotted = 0;

      selectedModels.forEach((model) => {{
        const oo = aggregateErrorsForCategories(model, "OO", testCategories);
        const comboAgg = aggregateErrorsForCategories(model, combo, testCategories);
        if (!oo.total || !comboAgg.total) {{
          return;
        }}
        const deltas = ERROR_TYPES_ORDER.map((errorType) => {{
          const ooRate = oo.total ? (oo.errorCounts[errorType] || 0) / oo.total : 0;
          const comboRate = comboAgg.total
            ? (comboAgg.errorCounts[errorType] || 0) / comboAgg.total
            : 0;
          return {{
            errorType,
            delta: (comboRate - ooRate) * 100
          }};
        }});
        const selected = deltas
          .filter((row) => Math.abs(row.delta) > 0.1)
          .sort((a, b) => a.delta - b.delta);
        if (!selected.length) {{
          return;
        }}
        movers += selected.length;
        plotted += 1;

        const y = selected.map((row) => row.errorType);
        const x = selected.map((row) => row.delta);
        const colors = x.map((val) => (val >= 0 ? "#2f855a" : "#c43d3d"));
        const text = x.map((val) => `${{val >= 0 ? "+" : ""}}${{val.toFixed(1)}}pp`);
        const trace = {{
          type: "bar",
          orientation: "h",
          x,
          y,
          marker: {{ color: colors }},
          text,
          textposition: "outside",
          cliponaxis: false,
          hovertemplate: "Error: %{{y}}<br>Delta %{{x:+.2f}}pp<extra></extra>"
        }};

        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = model;
        section.appendChild(heading);
        const plotDiv = document.createElement("div");
        plotDiv.style.height = "320px";
        section.appendChild(plotDiv);
        container.appendChild(section);

        const minX = Math.min(...x);
        const maxX = Math.max(...x);
        const pad = Math.max(0.6, (Math.abs(minX) + Math.abs(maxX)) * 0.08);
        const layout = {{
          margin: {{ t: 20, l: 200, r: 40, b: 40 }},
          xaxis: {{
            title: "Delta error rate (pp)",
            zeroline: true,
            zerolinecolor: "rgba(0,0,0,0.35)",
            showgrid: true,
            gridcolor: "rgba(0,0,0,0.08)",
            range: [minX - pad, maxX + pad]
          }},
          yaxis: {{
            automargin: true,
            showgrid: false
          }}
        }};
        Plotly.newPlot(plotDiv, [trace], layout, {{ displayModeBar: false }});
      }});

      if (!container.childNodes.length) {{
        warning.style.display = "";
        warning.textContent = "No error data for the current selection.";
      }}
      console.log("[error shifts]", {{
        combo,
        errorTypes: ERROR_TYPES_ORDER.length,
        models: plotted,
        movers
      }});
    }}

    function updateHeatmaps() {{
      const stack = document.getElementById("heatmap_stack");
      const warning = document.getElementById("heatmapWarning");
      if (!stack || !warning || typeof Plotly === "undefined") {{
        return;
      }}
      const heatmapView = document.getElementById("plots_heatmap_view");
      if (heatmapView && !heatmapView.classList.contains("active")) {{
        return;
      }}
      const selectedTable = document.getElementById("tableSelect").value;
      const tables = getCategoryTables();
      const selectedCombo = document.getElementById("heatmapCombo").value;
      const selectedModels = Array.from(getSelectedModels());
      const tableList = tables.filter((table) => {{
        if (selectedTable === "main") {{
          return false;
        }}
        if (selectedTable !== "all" && selectedTable !== table.key) {{
          return false;
        }}
        if (isExcludeWebSearch() && table.title.toLowerCase().includes("web search")) {{
          return false;
        }}
        return true;
      }});

      stack.innerHTML = "";
      warning.style.display = "none";
      warning.textContent = "";
      if (!tableList.length) {{
        warning.style.display = "";
        warning.textContent = "No categories selected for the heatmap.";
        return;
      }}

      const modelSet = new Set(selectedModels);
      let maxAbs = 0;
      const specs = [];

      tableList.forEach((table) => {{
        const rowIndex = {{}};
        table.rows.forEach((row) => {{
          rowIndex[row.model] = row;
        }});
        const yLabels = selectedModels.filter((model) => modelSet.has(model));
        const metrics = table.metrics.map((metric) => getMetricDisplayLabel(metric, table));
        const isAgentic = table.title.toLowerCase().includes("agentic");

        const slices = isAgentic
          ? [
              {{
                key: "web_search",
                title: "Agentic (Web Search)",
                filter: (label) => label.toLowerCase().includes("web search")
              }},
              {{
                key: "memory",
                title: "Agentic (Memory)",
                filter: (label) => {{
                  const lower = label.toLowerCase();
                  return (
                    lower.includes("memory") &&
                    !lower.includes("agentic overall") &&
                    !lower.includes("agentic (memory-only)")
                  );
                }}
              }}
            ]
          : [
              {{
                key: "default",
                title: table.title,
                filter: () => true
              }}
            ];

        slices.forEach((slice) => {{
          if (isExcludeWebSearch() && slice.key === "web_search") {{
            return;
          }}
          const metricIndexes = metrics
            .map((label, idx) => (slice.filter(label) ? idx : -1))
            .filter((idx) => idx >= 0);
          if (!metricIndexes.length) {{
            return;
          }}
          const xLabels = metricIndexes.map((idx) => metrics[idx]);
          const z = [];
          const custom = [];

          yLabels.forEach((model) => {{
            const row = rowIndex[model];
            const rowZ = [];
            const rowCustom = [];
            metricIndexes.forEach((metricIdx) => {{
              const metricKey = table.metrics[metricIdx];
              const metricLabel = metrics[metricIdx];
              const oo = normalizeAcc(
                row ? getMetricDisplayValue(row, metricKey, table, "OO") : null
              );
              const combo = normalizeAcc(
                row ? getMetricDisplayValue(row, metricKey, table, selectedCombo) : null
              );
              let delta = null;
              if (isNumber(oo) && isNumber(combo)) {{
                delta = combo - oo;
                maxAbs = Math.max(maxAbs, Math.abs(delta));
              }}
              rowZ.push(delta);
              rowCustom.push({{
                model,
                category: slice.title,
                label: metricLabel,
                oo,
                combo,
                delta,
                n: "not available"
              }});
            }});
            z.push(rowZ);
            custom.push(rowCustom);
          }});

          specs.push({{
            title: slice.title,
            xLabels,
            yLabels,
            z,
            custom
          }});
        }});
      }});

      if (!isNumber(maxAbs) || maxAbs === 0) {{
        maxAbs = 0.05;
      }}

      specs.forEach((spec, index) => {{
        const section = document.createElement("div");
        section.className = "section";
        const heading = document.createElement("h2");
        heading.textContent = spec.title;
        section.appendChild(heading);
        const container = document.createElement("div");
        container.id = `heatmap-${{index}}`;
        section.appendChild(container);
        stack.appendChild(section);

        const trace = {{
          type: "heatmap",
          z: spec.z,
          x: spec.xLabels,
          y: spec.yLabels,
          zmid: 0,
          zmin: -maxAbs,
          zmax: maxAbs,
          customdata: spec.custom,
          hovertemplate:
            "Model: %{{customdata.model}}<br>" +
            "Category: %{{customdata.category}}<br>" +
            "Row: %{{customdata.label}}<br>" +
            "OO: %{{customdata.oo:.3f}}<br>" +
            "Combo: %{{customdata.combo:.3f}}<br>" +
            "Delta: %{{customdata.delta:.3f}}<br>" +
            "n: %{{customdata.n}}<extra></extra>"
        }};
        const layout = {{
          margin: {{ t: 10, l: 80, r: 20, b: 120 }},
          xaxis: {{
            tickangle: -30
          }},
          yaxis: {{
            automargin: true
          }}
        }};
        Plotly.newPlot(container, [trace], layout, {{ displayModeBar: false }});
      }});
    }}

    function renderBarCharts() {{
      const tables = getCategoryTables();
      const selectedTable = document.getElementById("tableSelect").value;
      tables.forEach((table) => {{
        if (selectedTable !== "all" && selectedTable !== table.key) {{
          return;
        }}
        renderBarChart(table);
      }});
      attachBarControlHandlers();
    }}

    function renderBarChart(table) {{
      const container = document.getElementById(`bar-container-${{table.key}}`);
      if (!container) {{
        return;
      }}
      const state = barState[table.key] || {{
        metric: table.metrics[0] || "",
        condition: "AA"
      }};
      barState[table.key] = state;
      container.innerHTML = "";
      const selectedModels = Array.from(getSelectedModels());
      const rowIndex = {{}};
      table.rows.forEach((row) => {{
        rowIndex[row.model] = row;
      }});
      const mode = document.getElementById("deltaMode").value;
      const rows = selectedModels.map((model) => {{
        const row = rowIndex[model];
        if (!row) {{
          return null;
        }}
        const baseline = getMetricDisplayValue(row, state.metric, table, "OO");
        const value = getMetricDisplayValue(row, state.metric, table, state.condition);
        const delta = computeDeltaValue(value, baseline, mode);
        if (!isNumber(delta)) {{
          return null;
        }}
        return {{
          model,
          baseline,
          value,
          delta,
        }};
      }}).filter((item) => item);
      if (!rows.length) {{
        const empty = document.createElement("div");
        empty.className = "note";
        empty.textContent = "No data available for the current selection.";
        container.appendChild(empty);
        return;
      }}

      rows.sort((a, b) => b.delta - a.delta);
      const visible = rows;
      const maxAbs = Math.max(1, ...rows.map((row) => Math.abs(row.delta)));
      const labelWidth = 180;
      const barAreaWidth = 420;
      const barHeight = 18;
      const rowGap = 6;
      const width = labelWidth + barAreaWidth + 40;
      const height = visible.length * (barHeight + rowGap) + 30;
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.classList.add("chart-svg");
      const zeroX = labelWidth + barAreaWidth / 2;

      const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
      axis.setAttribute("x1", zeroX);
      axis.setAttribute("x2", zeroX);
      axis.setAttribute("y1", 10);
      axis.setAttribute("y2", height - 10);
      axis.setAttribute("stroke", "#9aa7c5");
      axis.setAttribute("stroke-width", "1");
      svg.appendChild(axis);

      visible.forEach((row, index) => {{
        const y = 20 + index * (barHeight + rowGap);
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", 4);
        label.setAttribute("y", y + barHeight - 4);
        label.setAttribute("font-size", "11");
        label.textContent = row.model;
        svg.appendChild(label);

        const barLength = Math.abs(row.delta) / maxAbs * (barAreaWidth / 2 - 8);
        const x = row.delta >= 0 ? zeroX : zeroX - barLength;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", x);
        rect.setAttribute("y", y);
        rect.setAttribute("width", barLength);
        rect.setAttribute("height", barHeight);
        rect.setAttribute("rx", "3");
        rect.setAttribute("fill", row.delta >= 0 ? "#2fbf71" : "#e06a5f");
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        const deltaText = row.delta.toFixed(2) + (mode === "rel" ? "%" : "");
        title.textContent = `${{table.title}} | ${{row.model}} OO: ${{row.baseline?.toFixed(2) ?? "NA"}}, ${{state.condition}}: ${{row.value?.toFixed(2) ?? "NA"}}, Δ: ${{deltaText}}`;
        rect.appendChild(title);
        svg.appendChild(rect);

        const valueText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        const textX = row.delta >= 0 ? x + barLength + 4 : x - 4;
        valueText.setAttribute("x", textX);
        valueText.setAttribute("y", y + barHeight - 4);
        valueText.setAttribute("text-anchor", row.delta >= 0 ? "start" : "end");
        valueText.setAttribute("font-size", "10");
        valueText.setAttribute("fill", "#2f3a56");
        valueText.textContent = deltaText;
        svg.appendChild(valueText);
      }});
      container.appendChild(svg);
    }}

    function attachBarControlHandlers() {{
      document.querySelectorAll(".chart-controls[data-table-key]").forEach((controls) => {{
        const tableKey = controls.dataset.tableKey;
        const metricSelect = document.getElementById(`bar-metric-${{tableKey}}`);
        const conditionSelect = document.getElementById(`bar-condition-${{tableKey}}`);
        if (!metricSelect || !conditionSelect) {{
          return;
        }}
        if (metricSelect.dataset.bound === "1") {{
          return;
        }}
        metricSelect.dataset.bound = "1";
        metricSelect.addEventListener("change", (event) => {{
          const state = barState[tableKey];
          state.metric = event.currentTarget.value;
          renderBarCharts();
        }});
        conditionSelect.addEventListener("change", (event) => {{
          const state = barState[tableKey];
          state.condition = event.currentTarget.value;
          renderBarCharts();
        }});
      }});
    }}

    buildTableSelect();
    buildModelFilter();
    tableState["main"] = {{ sortKey: null, sortDir: null }};
    categoryTablesFull.forEach((table) => {{
      tableState[table.key] = {{ sortKey: null, sortDir: null }};
    }});
    categoryTablesSummary.forEach((table) => {{
      if (!tableState[table.key]) {{
        tableState[table.key] = {{ sortKey: null, sortDir: null }};
      }}
    }});
    bindControls();
    document.getElementById("detailMode").value = "full";
    buildTableSelect();
    document.getElementById("tableSelect").value = "all";
    buildPlotCategoryFilter();
    renderAllTablesWithHeaders();
    setErrorView("mix");
    setPlotView("scatter");
    setActiveTab("results");
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BFCL leaderboard outputs.")
    parser.add_argument(
        "--export-images",
        action="store_true",
        help="Export thesis figures to SVG/PNG via Plotly/Kaleido.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute cached aggregated rows and error summary.",
    )
    parser.add_argument(
        "--error-sanity-check",
        action="store_true",
        help="Log per-slice error sanity checks when building error summaries.",
    )
    parser.add_argument(
        "--diagnose-overall",
        action="store_true",
        help="Print overall delta diagnostics (category deltas and contributions).",
    )
    args = parser.parse_args()

    root = Path.cwd()
    cache_root = root / "analysis_out"
    agg_cache_path = cache_root / "cached_aggregated_main_rows.json"
    error_cache_path = cache_root / "cached_error_summary.json"
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    data_map = load_all_condition_data(root)
    timings["load_all_condition_data"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    wide, missing_conditions = build_wide_table(data_map)
    timings["build_wide_table"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    table = compute_deltas(wide)
    timings["compute_deltas"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    overall_excl_web = compute_overall_excl_web(data_map)
    timings["compute_overall_excl_web"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = make_rows(table, overall_excl_web)
    timings["make_rows"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    aggregated_rows: List[Dict[str, object]] = []
    if not args.refresh_cache:
        cached = load_json_cache(agg_cache_path)
        if isinstance(cached, list):
            aggregated_rows = [row for row in cached if isinstance(row, dict)]
            if aggregated_rows:
                print(f"[info] using cached aggregated rows: {agg_cache_path}")
    if not aggregated_rows:
        aggregated_rows = build_aggregated_main_rows(root, data_map)
        write_json_cache(agg_cache_path, aggregated_rows)
        print(f"[info] cached aggregated rows: {agg_cache_path}")
    timings["build_aggregated_main_rows"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    category_tables = build_category_tables(data_map, mode="full")
    timings["build_category_tables_full"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    summary_tables = build_category_tables(data_map, mode="summary")
    timings["build_category_tables_summary"] = time.perf_counter() - t0
    all_models = sorted(
        {
            row["model"]
            for row in rows + aggregated_rows
        }.union(
            {
                detail_row["model"]
                for table_info in category_tables + summary_tables
                for detail_row in table_info["rows"]
            }
        )
    )
    t0 = time.perf_counter()
    overall_components = build_overall_components(data_map)
    timings["build_overall_components"] = time.perf_counter() - t0
    if args.diagnose_overall:
        print_overall_delta_diagnostics(
            rows, overall_components, category_tables, summary_tables, overall_excl_web
        )
    t0 = time.perf_counter()
    error_records: List[Dict[str, object]] = []
    error_groups: Dict[str, Dict[str, List[str]]] = {}
    if not args.refresh_cache:
        cached = load_json_cache(error_cache_path)
        if isinstance(cached, dict):
            records = cached.get("records")
            groups = cached.get("groups")
            if isinstance(records, list) and isinstance(groups, dict):
                error_records = [row for row in records if isinstance(row, dict)]
                error_groups = {
                    key: val for key, val in groups.items() if isinstance(val, dict)
                }
                if error_records and error_groups:
                    print(f"[info] using cached error summary: {error_cache_path}")
    if not error_records or not error_groups:
        error_records, error_groups = build_error_summary(
            root, data_map, args.error_sanity_check
        )
        write_json_cache(error_cache_path, {"records": error_records, "groups": error_groups})
        print(f"[info] cached error summary: {error_cache_path}")
    timings["build_error_summary"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    html = render_html(
        rows,
        aggregated_rows,
        missing_conditions,
        category_tables,
        summary_tables,
        all_models,
        overall_components,
        error_records,
        error_groups,
    )
    timings["render_html"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    out_path = root / "leaderboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"leaderboard written: {out_path.resolve()}")
    timings["write_leaderboard_html"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    thesis_traces, thesis_layout, thesis_caption = build_thesis_main_result_plot(
        aggregated_rows,
        use_excl_web=False,
    )
    thesis_excl_traces, thesis_excl_layout, thesis_excl_caption = (
        build_thesis_main_result_plot(aggregated_rows, use_excl_web=True)
    )
    scatter_specs, scatter_info, scatter_excluded = build_thesis_scatter_facet_specs(
        category_tables,
        summary_tables,
    )
    timings["build_thesis_figures"] = time.perf_counter() - t0
    print(
        "[info] thesis scatter facets: "
        f"{scatter_info['models']} models, "
        f"{scatter_info['points']} points, "
        f"{scatter_info['labels']} labels"
    )
    if scatter_excluded:
        preview = ", ".join(scatter_excluded[:10])
        suffix = "..." if len(scatter_excluded) > 10 else ""
        print(f"[info] thesis scatter excluded (web search): {preview}{suffix}")
    t0 = time.perf_counter()
    thesis_html = render_thesis_figures_html(
        thesis_traces,
        thesis_layout,
        thesis_caption,
        thesis_excl_traces,
        thesis_excl_layout,
        thesis_excl_caption,
        error_records,
        error_groups,
        scatter_specs,
    )
    timings["render_thesis_html"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    thesis_path = root / "thesis_figures.html"
    thesis_path.write_text(thesis_html, encoding="utf-8")
    print(f"thesis figures written: {thesis_path.resolve()}")
    timings["write_thesis_html"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if args.export_images:
        export_thesis_plot_images(
            thesis_traces,
            thesis_layout,
            root / "figures",
            "main_result_overall_acc",
        )
        export_thesis_plot_images(
            thesis_excl_traces,
            thesis_excl_layout,
            root / "figures",
            "main_result_overall_acc_excl_web",
        )
        export_thesis_error_shift_plot(error_records, root / "figures")
        export_thesis_scatter_facets(scatter_specs, root / "figures")
    else:
        print("[info] skipping thesis image exports (use --export-images to enable)")
    timings["export_thesis_images"] = time.perf_counter() - t0

    # print("[timing] seconds by section:")
    # for key, value in timings.items():
    #     print(f"[timing] {key}: {value:.3f}s")
    # return 0


if __name__ == "__main__":
    raise SystemExit(main())

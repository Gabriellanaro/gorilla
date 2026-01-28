"""
README:
Run: python make_leaderboard.py
Output: leaderboard.html at the project root.
"""

from __future__ import annotations

import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bfcl_eval.eval_checker.eval_runner_helper import generate_leaderboard_csv


CONDITIONS = {
    "OO": "score_desc_original_name_original",
    "OA": "score_desc_original_name_augmented",
    "AO": "score_desc_augmented_name_original",
    "AA": "score_desc_augmented_name_augmented",
}

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


def build_error_summary(
    root: Path,
    data_map: Dict[str, Dict[str, pd.DataFrame]],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, List[str]]]]:
    records: List[Dict[str, object]] = []
    display_name_map: Dict[str, str] = {}
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
            error_counts = Counter()
            for row in error_rows:
                error_type, sub_error, error_text = extract_error_fields(row)
                primary = infer_primary_error_type(
                    test_category, error_type, sub_error, error_text
                )
                error_counts[primary] += 1
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
      document.getElementById("errorTabMix").addEventListener("click", () => {{
        setErrorView("mix");
      }});
      document.getElementById("errorTabHeatmap").addEventListener("click", () => {{
        setErrorView("heatmap");
      }});
      document.getElementById("errorTabBuckets").addEventListener("click", () => {{
        setErrorView("buckets");
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

    function getSelectedTestCategories() {{
      const selectedTable = document.getElementById("tableSelect").value;
      const tables = getCategoryTables();
      const selected = new Set();
      tables.forEach((table) => {{
        if (selectedTable !== "all" && selectedTable !== table.key) {{
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
      const tabMix = document.getElementById("errorTabMix");
      const tabHeatmap = document.getElementById("errorTabHeatmap");
      const tabBuckets = document.getElementById("errorTabBuckets");
      if (!mixView || !heatmapView || !bucketView || !tabMix || !tabHeatmap || !tabBuckets) {{
        return;
      }}
      const showMix = viewId === "mix";
      const showHeatmap = viewId === "heatmap";
      const showBuckets = viewId === "buckets";
      mixView.classList.toggle("active", showMix);
      heatmapView.classList.toggle("active", showHeatmap);
      bucketView.classList.toggle("active", showBuckets);
      tabMix.classList.toggle("active", showMix);
      tabHeatmap.classList.toggle("active", showHeatmap);
      tabBuckets.classList.toggle("active", showBuckets);
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
          table.rows.forEach((row) => {{
            if (row.model !== model) {{
              return;
            }}
            table.metrics.forEach((metric) => {{
              const metricLabel = getMetricDisplayLabel(metric, table);
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
                  meta: {{
                    condition,
                    category: table.title,
                    label: metricLabel,
                    oo,
                    combo,
                    delta
                  }}
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
            return {{
              name: condition,
              mode: "markers",
              type: "scatter",
              x: tracePoints.map((pt) => pt.x),
              y: tracePoints.map((pt) => pt.y),
              marker: {{ size: 10 }},
              customdata: tracePoints.map((pt) => pt.meta),
              hovertemplate:
                "Combo: %{{customdata.condition}}<br>" +
                "Category: %{{customdata.category}}<br>" +
                "Row: %{{customdata.label}}<br>" +
                "OO: %{{customdata.oo:.3f}}<br>" +
                "Combo: %{{customdata.combo:.3f}}<br>" +
                "Delta: %{{customdata.delta:.3f}}<extra></extra>"
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
      if (mixView && mixView.classList.contains("active")) {{
        updateErrorMix();
      }}
      if (heatmapView && heatmapView.classList.contains("active")) {{
        updateErrorHeatmap();
      }}
      if (bucketView && bucketView.classList.contains("active")) {{
        updateErrorBuckets();
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
      const mode = document.getElementById("errorMixMode").value;

      selectedModels.forEach((model) => {{
        const traces = ERROR_TYPES_ORDER.map((errorType) => {{
          const y = [];
          const custom = [];
          ["OO", "OA", "AO", "AA"].forEach((condition) => {{
            const agg = aggregateErrorsForCategories(model, condition, testCategories);
            const count = agg.errorCounts[errorType] || 0;
            const share = agg.incorrect ? count / agg.incorrect : 0;
            y.push(mode === "share" ? share : count);
            custom.push({{
              errorType,
              count,
              share,
              incorrect: agg.incorrect
            }});
          }});
          return {{
            name: errorType,
            type: "bar",
            x: ["OO", "OA", "AO", "AA"],
            y,
            customdata: custom,
            hovertemplate:
              "Error: %{{customdata.errorType}}<br>" +
              "Count: %{{customdata.count}}<br>" +
              "Share: %{{customdata.share:.2%}}<br>" +
              "Total incorrect: %{{customdata.incorrect}}<extra></extra>"
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
            title: mode === "share" ? "Share of incorrect" : "Incorrect count",
            tickformat: mode === "share" ? ".0%" : undefined
          }}
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

      selectedModels.forEach((model) => {{
        const rows = [];
        let maxAbs = 0;
        tables.forEach((table) => {{
          if (selectedTable !== "all" && selectedTable !== table.key) {{
            return;
          }}
          const groups = errorGroups[table.key] || {{}};
          table.metrics.forEach((metric) => {{
            const label = getMetricDisplayLabel(metric, table);
            const categories = groups[metric] || groups[label] || [];
            const filtered = isExcludeWebSearch()
              ? categories.filter((cat) => !isWebSearchCategory(cat))
              : categories;
            if (!filtered.length) {{
              return;
            }}
            const oo = aggregateErrorsForCategories(model, "OO", filtered);
            const combo = aggregateErrorsForCategories(model, selectedCombo, filtered);
            if (!oo.total) {{
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
              const delta = comboRate - ooRate;
              row.values[errorType] = {{
                ooRate,
                comboRate,
                delta,
                total: oo.total
              }};
              maxAbs = Math.max(maxAbs, Math.abs(delta));
            }});
            rows.push(row);
          }});
        }});

        if (!rows.length) {{
          return;
        }}
        if (!maxAbs) {{
          maxAbs = 0.01;
        }}
        const xLabels = ERROR_TYPES_ORDER.slice();
        const yLabels = rows.map((row) => row.label);
        const z = rows.map((row) =>
          xLabels.map((errorType) => row.values[errorType].delta)
        );
        const custom = rows.map((row) =>
          xLabels.map((errorType) => ({{
            label: row.label,
            category: row.category,
            errorType,
            ooRate: row.values[errorType].ooRate,
            comboRate: row.values[errorType].comboRate,
            delta: row.values[errorType].delta,
            total: row.values[errorType].total
          }}))
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
            "Delta: %{{customdata.delta:.2%}}<br>" +
            "N: %{{customdata.total}}<extra></extra>"
        }};
        const layout = {{
          margin: {{ t: 20, l: 120, r: 20, b: 120 }},
          xaxis: {{ tickangle: -30 }},
          yaxis: {{ automargin: true }}
        }};
        Plotly.newPlot(plotDiv, [trace], layout, {{ displayModeBar: false }});
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
    root = Path.cwd()
    data_map = load_all_condition_data(root)
    wide, missing_conditions = build_wide_table(data_map)
    table = compute_deltas(wide)
    overall_excl_web = compute_overall_excl_web(data_map)
    rows = make_rows(table, overall_excl_web)
    aggregated_rows = build_aggregated_main_rows(root, data_map)
    category_tables = build_category_tables(data_map, mode="full")
    summary_tables = build_category_tables(data_map, mode="summary")
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
    overall_components = build_overall_components(data_map)
    error_records, error_groups = build_error_summary(root, data_map)
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
    out_path = root / "leaderboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"leaderboard written: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

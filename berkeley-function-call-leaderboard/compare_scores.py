import argparse
import csv
import json
import sys
from pathlib import Path
import re


def normalize_key(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_model_key(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"n/a", "na", "nan", "none"}:
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def detect_subcategory_field(headers):
    if not headers:
        return None
    candidates = [
        "subcategory",
        "sub_category",
        "subcat",
        "split",
        "subset",
        "subtask",
        "task",
    ]
    for candidate in candidates:
        if candidate in headers:
            return candidate
    for header in headers:
        if "subcat" in header or "subset" in header or "split" in header:
            return header
    return None


def find_first_csv(directory):
    csv_files = sorted(directory.glob("*.csv"))
    return csv_files[0] if csv_files else None


def find_root_category_csv(scores_root, category):
    candidate = scores_root / f"data_{category}.csv"
    if candidate.exists():
        return candidate
    for csv_path in sorted(scores_root.glob("*.csv")):
        if category.lower() in csv_path.stem.lower():
            return csv_path
    return None


def count_error_items(leaf_dir):
    file_count = 0
    row_count = 0
    for json_path in leaf_dir.glob("*.json"):
        file_count += 1
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                first_line = ""
                while True:
                    pos = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        first_line = line.strip()
                        break
                if first_line:
                    try:
                        first_payload = json.loads(first_line)
                    except json.JSONDecodeError:
                        first_payload = None
                    if isinstance(first_payload, dict):
                        total = parse_float(first_payload.get("total_count"))
                        correct = parse_float(first_payload.get("correct_count"))
                        if total is not None and correct is not None:
                            row_count += max(int(total - correct), 0)
                            continue
                handle.seek(0)
                try:
                    payload = json.load(handle)
                except json.JSONDecodeError:
                    handle.seek(pos if first_line else 0)
                    line_rows = 0
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        line_rows += 1
                    row_count += line_rows
                    continue
        except OSError:
            continue
        if isinstance(payload, list):
            row_count += len(payload)
        elif isinstance(payload, dict):
            # Common pattern: errors are stored under a list field.
            if isinstance(payload.get("errors"), list):
                row_count += len(payload["errors"])
            elif isinstance(payload.get("data"), list):
                row_count += len(payload["data"])
            else:
                row_count += 1
        else:
            row_count += 1
    return file_count, row_count


def read_csv_metrics(csv_path, model, category, model_filter=None):
    records = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return records
            normalized_headers = [normalize_key(h) for h in reader.fieldnames]
            header_map = dict(zip(reader.fieldnames, normalized_headers))
            subcategory_field = detect_subcategory_field(normalized_headers)

            model_column = None
            if model_filter is not None:
                if "model" in normalized_headers:
                    model_column = "model"
                else:
                    for header in normalized_headers:
                        if "model" in header and header != "model_link":
                            model_column = header
                            break
                if model_column is None:
                    print(
                        f"Warning: no model column found in {csv_path} for filtering.",
                        file=sys.stderr,
                    )

            model_key = normalize_model_key(model_filter) if model_filter else None

            for row in reader:
                normalized_row = {}
                for key, value in row.items():
                    normalized_row[header_map[key]] = value.strip() if isinstance(value, str) else value

                if model_key and model_column:
                    row_model = normalized_row.get(model_column, "")
                    if normalize_model_key(row_model) != model_key:
                        continue

                subcategory_value = (
                    normalized_row.get(subcategory_field) if subcategory_field else "overall"
                )
                subcategory = normalize_key(subcategory_value) or "overall"

                for key, value in normalized_row.items():
                    if subcategory_field and key == subcategory_field:
                        continue
                    numeric_value = parse_float(value)
                    if numeric_value is None:
                        continue
                    records.append(
                        {
                            "condition": None,
                            "model": model,
                            "category": category,
                            "subcategory": subcategory,
                            "metric_name": key,
                            "metric_value": numeric_value,
                        }
                    )
    except OSError as exc:
        print(f"Warning: failed to read {csv_path}: {exc}", file=sys.stderr)
    return records


def collect_metrics(scores_root, condition):
    records = []
    error_counts = {}

    if not scores_root.exists():
        print(f"Warning: scores root {scores_root} does not exist.", file=sys.stderr)
        return records, error_counts

    model_dirs = sorted([p for p in scores_root.iterdir() if p.is_dir()])
    for model_dir in model_dirs:
        model_name = model_dir.name
        category_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
        for category_dir in category_dirs:
            category_name = category_dir.name
            error_file_count, error_item_count = count_error_items(category_dir)
            error_counts[(model_name, category_name)] = (error_file_count, error_item_count)

            leaf_csv = find_first_csv(category_dir)
            if leaf_csv:
                csv_records = read_csv_metrics(leaf_csv, model_name, category_name)
            else:
                root_csv = find_root_category_csv(scores_root, category_name)
                if not root_csv:
                    print(
                        f"Warning: no CSV found for {model_name}/{category_name} under {scores_root}.",
                        file=sys.stderr,
                    )
                    continue
                csv_records = read_csv_metrics(
                    root_csv, model_name, category_name, model_filter=model_name
                )

            if not csv_records:
                print(
                    f"Warning: no metrics parsed for {model_name}/{category_name} from {scores_root}.",
                    file=sys.stderr,
                )
                continue

            for record in csv_records:
                record["condition"] = condition
                record["error_file_count"] = error_file_count
                record["error_item_count"] = error_item_count
                records.append(record)

    return records, error_counts


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_comparison(orig_records, aug_records, orig_errors, aug_errors):
    orig_map = {
        (r["model"], r["category"], r["subcategory"], r["metric_name"]): r
        for r in orig_records
    }
    aug_map = {
        (r["model"], r["category"], r["subcategory"], r["metric_name"]): r
        for r in aug_records
    }
    common_keys = sorted(set(orig_map) & set(aug_map))
    comparison_rows = []
    for key in common_keys:
        model, category, subcategory, metric_name = key
        orig_value = orig_map[key]["metric_value"]
        aug_value = aug_map[key]["metric_value"]
        delta = aug_value - orig_value
        delta_pct = None
        if orig_value != 0:
            delta_pct = delta / orig_value
        orig_error = orig_errors.get((model, category), (0, 0))
        aug_error = aug_errors.get((model, category), (0, 0))
        comparison_rows.append(
            {
                "model": model,
                "category": category,
                "subcategory": subcategory,
                "metric_name": metric_name,
                "orig_value": orig_value,
                "aug_value": aug_value,
                "delta": delta,
                "delta_pct": delta_pct,
                "orig_error_file_count": orig_error[0],
                "orig_error_item_count": orig_error[1],
                "aug_error_file_count": aug_error[0],
                "aug_error_item_count": aug_error[1],
            }
        )
    return comparison_rows


def build_wide(comparison_rows, orig_errors, aug_errors):
    grouped = {}
    metrics_by_group = {}
    for row in comparison_rows:
        group_key = (row["model"], row["category"], row["subcategory"])
        grouped.setdefault(group_key, {})
        metrics_by_group.setdefault(group_key, set()).add(row["metric_name"])
        metric = row["metric_name"]
        grouped[group_key][f"orig_{metric}"] = row["orig_value"]
        grouped[group_key][f"aug_{metric}"] = row["aug_value"]
        grouped[group_key][f"delta_{metric}"] = row["delta"]

    wide_rows = []
    for (model, category, subcategory), values in sorted(grouped.items()):
        orig_error = orig_errors.get((model, category), (0, 0))
        aug_error = aug_errors.get((model, category), (0, 0))
        base = {
            "model": model,
            "category": category,
            "subcategory": subcategory,
            "orig_error_file_count": orig_error[0],
            "orig_error_item_count": orig_error[1],
            "aug_error_file_count": aug_error[0],
            "aug_error_item_count": aug_error[1],
        }
        base.update(values)
        wide_rows.append(base)

    all_metrics = sorted(
        {metric for metrics in metrics_by_group.values() for metric in metrics}
    )
    fieldnames = [
        "model",
        "category",
        "subcategory",
        "orig_error_file_count",
        "orig_error_item_count",
        "aug_error_file_count",
        "aug_error_item_count",
    ]
    for metric in all_metrics:
        fieldnames.extend([f"orig_{metric}", f"aug_{metric}", f"delta_{metric}"])
    return wide_rows, fieldnames


def write_readme(out_dir, scores_orig, scores_aug):
    content = f"""# Score Comparison Outputs

This folder contains CSVs produced by `compare_scores.py`, comparing original vs augmented
score summaries.

## Outputs
- `metrics_tidy.csv`: Long-form metrics for both conditions.
- `metrics_comparison_long.csv`: Joined original vs augmented metrics with deltas.
- `metrics_comparison_wide.csv`: Pivoted comparison table by model/category/subcategory.

## Run
```bash
python compare_scores.py --scores_orig {scores_orig} --scores_aug {scores_aug} --out_dir {out_dir}
```

Notes:
- The script scans `{scores_orig}` and `{scores_aug}` for model/category subfolders.
- If no CSV is found under a model/category leaf, it falls back to `data_<category>.csv`
  at the scores root (matching rows by model name).
"""
    readme_path = out_dir / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Compare original vs augmented score summaries.")
    parser.add_argument(
        "--scores_orig",
        default="score_desc_original_name_original",
        help="Path to original scores root.",
    )
    parser.add_argument(
        "--scores_aug",
        default="score_desc_augmented_name_augmented",
        help="Path to augmented scores root.",
    )
    parser.add_argument(
        "--out_dir",
        default="comparison_out",
        help="Output directory for comparison CSVs.",
    )
    args = parser.parse_args()

    scores_orig = Path(args.scores_orig)
    scores_aug = Path(args.scores_aug)
    out_dir = Path(args.out_dir)

    orig_records, orig_errors = collect_metrics(scores_orig, "orig")
    aug_records, aug_errors = collect_metrics(scores_aug, "aug")

    tidy_rows = sorted(
        orig_records + aug_records,
        key=lambda r: (r["condition"], r["model"], r["category"], r["subcategory"], r["metric_name"]),
    )

    tidy_fields = [
        "condition",
        "model",
        "category",
        "subcategory",
        "metric_name",
        "metric_value",
        "error_file_count",
        "error_item_count",
    ]

    write_csv(out_dir / "metrics_tidy.csv", tidy_rows, tidy_fields)

    comparison_rows = build_comparison(orig_records, aug_records, orig_errors, aug_errors)
    comparison_fields = [
        "model",
        "category",
        "subcategory",
        "metric_name",
        "orig_value",
        "aug_value",
        "delta",
        "delta_pct",
        "orig_error_file_count",
        "orig_error_item_count",
        "aug_error_file_count",
        "aug_error_item_count",
    ]
    write_csv(out_dir / "metrics_comparison_long.csv", comparison_rows, comparison_fields)

    wide_rows, wide_fields = build_wide(comparison_rows, orig_errors, aug_errors)
    write_csv(out_dir / "metrics_comparison_wide.csv", wide_rows, wide_fields)

    write_readme(out_dir, scores_orig, scores_aug)

    print(
        f"Wrote {len(tidy_rows)} tidy rows, {len(comparison_rows)} comparison rows to {out_dir}."
    )


if __name__ == "__main__":
    main()

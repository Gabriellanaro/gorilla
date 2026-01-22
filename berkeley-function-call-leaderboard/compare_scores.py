import argparse
import csv
import json
import sys
from pathlib import Path
import re
from collections import OrderedDict


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


def index_records(records):
    return {
        (r["model"], r["category"], r["subcategory"], r["metric_name"]): r
        for r in records
    }


def build_baseline_comparison(records_by_condition, errors_by_condition, baseline_label):
    if baseline_label not in records_by_condition:
        raise ValueError(f"Baseline condition '{baseline_label}' was not provided.")
    baseline_map = index_records(records_by_condition[baseline_label])
    comparison_rows = []
    for condition, records in records_by_condition.items():
        if condition == baseline_label:
            continue
        condition_map = index_records(records)
        common_keys = sorted(set(baseline_map) & set(condition_map))
        for key in common_keys:
            model, category, subcategory, metric_name = key
            baseline_value = baseline_map[key]["metric_value"]
            compare_value = condition_map[key]["metric_value"]
            delta = compare_value - baseline_value
            delta_pct = None
            if baseline_value != 0:
                delta_pct = delta / baseline_value
            baseline_error = errors_by_condition[baseline_label].get((model, category), (0, 0))
            compare_error = errors_by_condition[condition].get((model, category), (0, 0))
            comparison_rows.append(
                {
                    "baseline_condition": baseline_label,
                    "compare_condition": condition,
                    "model": model,
                    "category": category,
                    "subcategory": subcategory,
                    "metric_name": metric_name,
                    "baseline_value": baseline_value,
                    "compare_value": compare_value,
                    "delta": delta,
                    "delta_pct": delta_pct,
                    "baseline_error_file_count": baseline_error[0],
                    "baseline_error_item_count": baseline_error[1],
                    "compare_error_file_count": compare_error[0],
                    "compare_error_item_count": compare_error[1],
                }
            )
    return comparison_rows


def build_wide_by_metric(
    records_by_condition, errors_by_condition, conditions_order, baseline_label
):
    indexed = {label: index_records(records) for label, records in records_by_condition.items()}
    all_keys = sorted({key for mapping in indexed.values() for key in mapping})
    rows = []
    for key in all_keys:
        model, category, subcategory, metric_name = key
        row = {
            "model": model,
            "category": category,
            "subcategory": subcategory,
            "metric_name": metric_name,
        }
        for label in conditions_order:
            record = indexed.get(label, {}).get(key)
            row[f"value_{label}"] = record["metric_value"] if record else None
            error = errors_by_condition.get(label, {}).get((model, category), (0, 0))
            row[f"error_file_count_{label}"] = error[0]
            row[f"error_item_count_{label}"] = error[1]
        baseline_value = row.get(f"value_{baseline_label}")
        for label in conditions_order:
            if label == baseline_label:
                continue
            compare_value = row.get(f"value_{label}")
            delta = None
            delta_pct = None
            if baseline_value is not None and compare_value is not None:
                delta = compare_value - baseline_value
                if baseline_value != 0:
                    delta_pct = delta / baseline_value
            row[f"delta_{label}"] = delta
            row[f"delta_pct_{label}"] = delta_pct
        rows.append(row)

    fieldnames = ["model", "category", "subcategory", "metric_name"]
    for label in conditions_order:
        fieldnames.extend(
            [
                f"value_{label}",
                f"error_file_count_{label}",
                f"error_item_count_{label}",
            ]
        )
    for label in conditions_order:
        if label == baseline_label:
            continue
        fieldnames.extend([f"delta_{label}", f"delta_pct_{label}"])
    return rows, fieldnames


def write_readme(out_dir, score_roots, baseline_label, subcategory_filter):
    score_lines = "\n".join([f"- `{label}`: `{path}`" for label, path in score_roots.items()])
    content = f"""# Score Comparison Outputs

This folder contains CSVs produced by `compare_scores.py`, comparing multiple score roots
against a baseline.

## Outputs
- `metrics_tidy.csv`: Long-form metrics for all conditions.
- `metrics_comparison_long.csv`: Baseline vs each comparison condition with deltas.
- `metrics_comparison_wide.csv`: Baseline and all conditions in one wide table.
- `plots/`: PNG plots for selected metrics (overall only).

## Run
```bash
python compare_scores.py --out_dir {out_dir}
```

Notes:
- Baseline condition: `{baseline_label}`.
- Subcategory filter: `{subcategory_filter}`.
- Score roots:
{score_lines}
- The script scans each score root for model/category subfolders.
- If no CSV is found under a model/category leaf, it falls back to `data_<category>.csv`
  at the score root (matching rows by model name).
"""
    readme_path = out_dir / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(content, encoding="utf-8")


def normalize_metric_list(metric_list):
    normalized = []
    seen = set()
    for metric in metric_list:
        norm = normalize_key(metric)
        if norm and norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


def plot_metrics(
    tidy_rows,
    conditions_order,
    metrics_to_plot,
    out_dir,
    subcategory_filter,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available; skipping plots.", file=sys.stderr)
        return

    metric_set = set(metrics_to_plot)
    rows = [
        row
        for row in tidy_rows
        if row["subcategory"] == subcategory_filter and row["metric_name"] in metric_set
    ]
    if not rows:
        print("Warning: no rows matched plot filters; skipping plots.", file=sys.stderr)
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def safe_name(value):
        return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")

    grouped = {}
    for row in rows:
        key = (row["metric_name"], row["category"])
        grouped.setdefault(key, []).append(row)

    for (metric_name, category), metric_rows in grouped.items():
        models = sorted({row["model"] for row in metric_rows})
        value_map = {
            (row["model"], row["condition"]): row["metric_value"] for row in metric_rows
        }

        count = len(conditions_order)
        width = 0.8 / max(count, 1)
        x_positions = list(range(len(models)))

        plt.figure(figsize=(max(6, len(models) * 0.6), 4))
        for idx, condition in enumerate(conditions_order):
            offset = (idx - (count - 1) / 2) * width
            bar_x = [x + offset for x in x_positions]
            bar_y = [
                value_map.get((model, condition), float("nan")) for model in models
            ]
            plt.bar(bar_x, bar_y, width=width, label=condition)

        plt.xticks(x_positions, models, rotation=45, ha="right")
        display_metric = metric_name.replace("_", " ")
        plt.title(f"{display_metric} - {category}")
        plt.legend()
        plt.tight_layout()

        filename = f"{safe_name(metric_name)}_{safe_name(category)}.png"
        plt.savefig(plots_dir / filename, dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare original vs augmented score summaries.")
    parser.add_argument(
        "--scores",
        nargs="*",
        help="List of label=path entries to compare. Overrides default score roots.",
    )
    parser.add_argument(
        "--scores_orig",
        default=None,
        help="(Legacy) Path to original scores root for 2-way comparison.",
    )
    parser.add_argument(
        "--scores_aug",
        default=None,
        help="(Legacy) Path to augmented scores root for 2-way comparison.",
    )
    parser.add_argument(
        "--baseline",
        default="orig_orig",
        help="Baseline condition label for comparison.",
    )
    parser.add_argument(
        "--subcategory",
        default="overall",
        help="Subcategory filter (e.g., overall).",
    )
    parser.add_argument(
        "--plot_metrics",
        nargs="*",
        default=[
            "Overall Acc",
            "Non-Live AST Acc",
            "Non-Live Simple AST",
            "Non-Live Multiple AST",
            "Non-Live Parallel AST",
            "Non-Live Parallel Multiple AST",
            "Live Acc",
            "Live Simple AST",
            "Live Multiple AST",
            "Live Parallel AST",
            "Live Parallel Multiple AST",
            "Multi Turn Acc",
            "Multi Turn Base",
            "Multi Turn Miss Func",
            "Multi Turn Miss Param",
            "Multi Turn Long Context",
            "Web Search Acc",
            "Web Search Base",
            "Web Search No Snippet",
            "Memory Acc",
            "Memory KV",
            "Memory Vector",
            "Memory Recursive Summarization",
            "Relevance Detection",
            "Irrelevance Detection",
        ],
        help="Metrics to plot (match CSV headers).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plot generation.",
    )
    parser.add_argument(
        "--out_dir",
        default="comparison_out",
        help="Output directory for comparison CSVs.",
    )
    args = parser.parse_args()

    score_roots = OrderedDict()
    base_dir = Path(__file__).resolve().parent
    if args.scores:
        for entry in args.scores:
            if "=" not in entry:
                raise ValueError(f"Invalid score entry '{entry}', expected label=path.")
            label, path = entry.split("=", 1)
            score_roots[label.strip()] = Path(path.strip())
    elif args.scores_orig or args.scores_aug:
        if not (args.scores_orig and args.scores_aug):
            raise ValueError("Both --scores_orig and --scores_aug are required together.")
        score_roots["orig"] = Path(args.scores_orig)
        score_roots["aug"] = Path(args.scores_aug)
        if args.baseline == "orig_orig":
            args.baseline = "orig"
    else:
        score_roots["orig_orig"] = base_dir / "score_desc_original_name_original"
        score_roots["aug_aug"] = base_dir / "score_desc_augmented_name_augmented"
        score_roots["aug_orig"] = base_dir / "score_desc_augmented_name_original"
        score_roots["orig_aug"] = base_dir / "score_desc_original_name_augmented"

    subcategory_filter = normalize_key(args.subcategory)
    out_dir = Path(args.out_dir)

    records_by_condition = OrderedDict()
    errors_by_condition = OrderedDict()
    for condition, scores_root in score_roots.items():
        records, errors = collect_metrics(scores_root, condition)
        if subcategory_filter:
            records = [r for r in records if r["subcategory"] == subcategory_filter]
        records_by_condition[condition] = records
        errors_by_condition[condition] = errors

    tidy_rows = sorted(
        [record for records in records_by_condition.values() for record in records],
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

    comparison_rows = build_baseline_comparison(
        records_by_condition, errors_by_condition, args.baseline
    )
    comparison_fields = [
        "baseline_condition",
        "compare_condition",
        "model",
        "category",
        "subcategory",
        "metric_name",
        "baseline_value",
        "compare_value",
        "delta",
        "delta_pct",
        "baseline_error_file_count",
        "baseline_error_item_count",
        "compare_error_file_count",
        "compare_error_item_count",
    ]
    write_csv(out_dir / "metrics_comparison_long.csv", comparison_rows, comparison_fields)

    wide_rows, wide_fields = build_wide_by_metric(
        records_by_condition,
        errors_by_condition,
        list(score_roots.keys()),
        args.baseline,
    )
    write_csv(out_dir / "metrics_comparison_wide.csv", wide_rows, wide_fields)

    if not args.no_plots:
        plot_metrics(
            tidy_rows,
            list(score_roots.keys()),
            normalize_metric_list(args.plot_metrics),
            out_dir,
            subcategory_filter or "overall",
        )

    write_readme(out_dir, score_roots, args.baseline, subcategory_filter or "overall")

    print(
        f"Wrote {len(tidy_rows)} tidy rows, {len(comparison_rows)} comparison rows to {out_dir}."
    )


if __name__ == "__main__":
    main()

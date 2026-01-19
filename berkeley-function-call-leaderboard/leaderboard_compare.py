import argparse
import csv
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
    candidates = ["subcategory", "sub_category", "subset", "split"]
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


def detect_model_column(headers):
    if "model" in headers:
        return "model"
    for header in headers:
        if "model" in header and header != "model_link":
            return header
    return None


def read_summary_csv(csv_path, model_filter=None):
    rows = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return rows

            normalized_headers = [normalize_key(h) for h in reader.fieldnames]
            header_map = dict(zip(reader.fieldnames, normalized_headers))
            subcategory_field = detect_subcategory_field(normalized_headers)
            model_column = detect_model_column(normalized_headers)

            model_key = normalize_model_key(model_filter) if model_filter else None

            for row in reader:
                try:
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

                    metrics = {}
                    for key, value in normalized_row.items():
                        if subcategory_field and key == subcategory_field:
                            continue
                        if key == model_column:
                            continue
                        numeric_value = parse_float(value)
                        if numeric_value is None:
                            continue
                        metrics[key] = numeric_value

                    if not metrics:
                        print(
                            f"Warning: no numeric metrics parsed in row for {csv_path}.",
                            file=sys.stderr,
                        )
                        continue

                    rows.append({"subcategory": subcategory, "metrics": metrics})
                except Exception as exc:
                    print(f"Warning: failed to parse row in {csv_path}: {exc}", file=sys.stderr)
                    continue
    except OSError as exc:
        print(f"Warning: failed to read {csv_path}: {exc}", file=sys.stderr)
    return rows


def choose_primary_metric(metric_names):
    if not metric_names:
        return None
    if "overall_accuracy" in metric_names:
        return "overall_accuracy"
    for name in metric_names:
        if name == "acc":
            return name
    for name in metric_names:
        if "accuracy" in name:
            return name
    return metric_names[0]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Build leaderboard comparison between orig and aug scores.")
    parser.add_argument(
        "--orig",
        default="score_desc_original_name_original",
        help="Path to original scores root.",
    )
    parser.add_argument(
        "--aug",
        default="score_desc_augmented_name_augmented",
        help="Path to augmented scores root.",
    )
    parser.add_argument(
        "--out",
        default="comparison_out/leaderboard",
        help="Output directory for leaderboard comparison.",
    )
    args = parser.parse_args()

    orig_root = Path(args.orig)
    aug_root = Path(args.aug)
    out_dir = Path(args.out)

    if not orig_root.exists() or not aug_root.exists():
        print("Error: one or both score roots do not exist.", file=sys.stderr)
        return 1

    orig_models = {p.name for p in orig_root.iterdir() if p.is_dir()}
    aug_models = {p.name for p in aug_root.iterdir() if p.is_dir()}
    models = sorted(orig_models & aug_models)

    leaderboard_rows = []
    wide_rows = []
    data_map = {}
    primary_metric_map = {}
    model_primary_metrics = {}
    categories_compared = set()

    for model in models:
        orig_model_dir = orig_root / model
        aug_model_dir = aug_root / model
        orig_categories = {p.name for p in orig_model_dir.iterdir() if p.is_dir()}
        aug_categories = {p.name for p in aug_model_dir.iterdir() if p.is_dir()}
        categories = sorted(orig_categories & aug_categories)

        for category in categories:
            categories_compared.add(category)
            orig_leaf = orig_model_dir / category
            aug_leaf = aug_model_dir / category

            orig_csv = find_first_csv(orig_leaf)
            aug_csv = find_first_csv(aug_leaf)

            orig_filter = None
            aug_filter = None
            if not orig_csv:
                orig_csv = find_root_category_csv(orig_root, category)
                orig_filter = model
            if not aug_csv:
                aug_csv = find_root_category_csv(aug_root, category)
                aug_filter = model

            if not orig_csv or not aug_csv:
                print(
                    f"Warning: missing CSV for {model}/{category}; skipping.",
                    file=sys.stderr,
                )
                continue

            orig_rows = read_summary_csv(orig_csv, model_filter=orig_filter)
            aug_rows = read_summary_csv(aug_csv, model_filter=aug_filter)

            orig_map = {}
            aug_map = {}
            metric_names = set()

            for row in orig_rows:
                for metric_name, value in row["metrics"].items():
                    orig_map[(row["subcategory"], metric_name)] = value
            for row in aug_rows:
                for metric_name, value in row["metrics"].items():
                    aug_map[(row["subcategory"], metric_name)] = value

            common_keys = sorted(set(orig_map) & set(aug_map))
            if not common_keys:
                print(
                    f"Warning: no common metrics for {model}/{category}; skipping.",
                    file=sys.stderr,
                )
                continue

            for subcategory, metric_name in common_keys:
                orig_value = orig_map[(subcategory, metric_name)]
                aug_value = aug_map[(subcategory, metric_name)]
                delta = aug_value - orig_value
                delta_pct = None
                if orig_value != 0:
                    delta_pct = delta / orig_value
                row = {
                    "model": model,
                    "category": category,
                    "subcategory": subcategory,
                    "metric_name": metric_name,
                    "orig_value": orig_value,
                    "aug_value": aug_value,
                    "delta": delta,
                    "delta_pct": delta_pct,
                }
                leaderboard_rows.append(row)
                data_map[(model, category, subcategory, metric_name)] = row
                metric_names.add(metric_name)

            metric_list = sorted(metric_names)
            primary_metric = choose_primary_metric(metric_list)
            if primary_metric:
                primary_metric_map[(model, category)] = primary_metric
                model_primary_metrics.setdefault(model, set()).add(primary_metric)

    # Build wide rows
    grouped = {}
    metrics_by_group = {}
    for row in leaderboard_rows:
        group_key = (row["model"], row["category"], row["subcategory"])
        grouped.setdefault(group_key, {})
        metrics_by_group.setdefault(group_key, set()).add(row["metric_name"])
        metric = row["metric_name"]
        grouped[group_key][f"orig_{metric}"] = row["orig_value"]
        grouped[group_key][f"aug_{metric}"] = row["aug_value"]
        grouped[group_key][f"delta_{metric}"] = row["delta"]

    all_metrics = sorted(
        {metric for metrics in metrics_by_group.values() for metric in metrics}
    )
    wide_fields = ["model", "category", "subcategory"]
    for metric in all_metrics:
        wide_fields.extend([f"orig_{metric}", f"aug_{metric}", f"delta_{metric}"])

    for (model, category, subcategory), values in sorted(grouped.items()):
        base = {"model": model, "category": category, "subcategory": subcategory}
        base.update(values)
        wide_rows.append(base)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "leaderboard_long.csv", leaderboard_rows, [
        "model",
        "category",
        "subcategory",
        "metric_name",
        "orig_value",
        "aug_value",
        "delta",
        "delta_pct",
    ])
    write_csv(out_dir / "leaderboard_wide.csv", wide_rows, wide_fields)

    # Markdown report
    markdown_lines = ["# Leaderboard Comparison", ""]
    markdown_lines.append("This report compares augmented vs original scores by model/category.")
    markdown_lines.append("")

    # Top improvements/regressions by primary metric
    primary_rows = []
    for (model, category), primary_metric in primary_metric_map.items():
        for row in leaderboard_rows:
            if row["model"] != model or row["category"] != category:
                continue
            if row["metric_name"] != primary_metric:
                continue
            primary_rows.append(row)

    def format_primary_table(title, rows):
        markdown_lines.append(f"## {title}")
        markdown_lines.append("")
        markdown_lines.append("| Model | Category | Subcategory | Metric | Orig | Aug | Delta |")
        markdown_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            markdown_lines.append(
                f"| {row['model']} | {row['category']} | {row['subcategory']} | "
                f"{row['metric_name']} | {row['orig_value']:.4g} | {row['aug_value']:.4g} | "
                f"{row['delta']:.4g} |"
            )
        markdown_lines.append("")

    top_improvements = sorted(primary_rows, key=lambda r: r["delta"], reverse=True)[:10]
    top_regressions = sorted(primary_rows, key=lambda r: r["delta"])[:10]
    format_primary_table("Top 10 Improvements (Primary Accuracy Metric)", top_improvements)
    format_primary_table("Top 10 Regressions (Primary Accuracy Metric)", top_regressions)

    # Per-model tables
    for model in models:
        markdown_lines.append(f"## Model: {model}")
        markdown_lines.append("")
        model_categories = sorted({row["category"] for row in leaderboard_rows if row["model"] == model})
        if not model_categories:
            markdown_lines.append("_No comparable categories found._")
            markdown_lines.append("")
            continue
        for category in model_categories:
            primary_metric = primary_metric_map.get((model, category))
            markdown_lines.append(f"### Category: {category}")
            if primary_metric:
                markdown_lines.append(f"Primary accuracy metric: `{primary_metric}`")
            markdown_lines.append("")

            metric_names = sorted(
                {
                    row["metric_name"]
                    for row in leaderboard_rows
                    if row["model"] == model and row["category"] == category
                }
            )
            if not metric_names:
                markdown_lines.append("_No metrics found._")
                markdown_lines.append("")
                continue

            metric_list = []
            if primary_metric and primary_metric in metric_names:
                metric_list.append(primary_metric)
            for metric in metric_names:
                if metric == primary_metric:
                    continue
                metric_list.append(metric)
                if len(metric_list) >= 5:
                    break

            headers = ["Subcategory"]
            for metric in metric_list:
                headers.extend([f"{metric} (orig)", f"{metric} (aug)", f"{metric} (delta)"])
            markdown_lines.append("| " + " | ".join(headers) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

            subcategories = sorted(
                {
                    row["subcategory"]
                    for row in leaderboard_rows
                    if row["model"] == model and row["category"] == category
                }
            )
            for subcategory in subcategories:
                row_cells = [subcategory]
                for metric in metric_list:
                    entry = data_map.get((model, category, subcategory, metric))
                    if not entry:
                        row_cells.extend(["", "", ""])
                        continue
                    row_cells.extend(
                        [
                            f"{entry['orig_value']:.4g}",
                            f"{entry['aug_value']:.4g}",
                            f"{entry['delta']:.4g}",
                        ]
                    )
                markdown_lines.append("| " + " | ".join(row_cells) + " |")
            markdown_lines.append("")

    (out_dir / "leaderboard.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    # Console summary
    print(f"Models compared: {len(models)}")
    print(f"Categories compared: {len(categories_compared)}")
    print(f"Total rows written: {len(leaderboard_rows)}")
    for model in models:
        metrics = model_primary_metrics.get(model)
        if not metrics:
            continue
        if len(metrics) == 1:
            metric = next(iter(metrics))
            print(f"Primary metric for {model}: {metric}")
        else:
            per_category = {
                category: metric
                for (m, category), metric in primary_metric_map.items()
                if m == model
            }
            print(f"Primary metrics for {model}: {per_category}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

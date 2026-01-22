import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


DESCRIPTION_KEYS = [
    "augmented_description",
    "aug_description",
    "description",
    "tool_description",
    "docstring",
]
ORIG_DESCRIPTION_KEYS = [
    "description",
    "tool_description",
    "docstring",
]
CATEGORY_KEYS = [
    "category",
    "task_category",
    "tool_category",
    "group",
    "split",
    "source_files",
]
SUBCATEGORY_KEYS = [
    "subcategory",
    "task_category",
    "tool_category",
    "group",
    "split",
    "task_types",
]
SUBSUBCATEGORY_KEYS = [
    "subsubcategory",
]
SCHEMA_KEYS = [
    "parameters",
    "args_schema",
    "input_schema",
    "json_schema",
    "schema",
]
OUTPUT_SCHEMA_KEYS = [
    "returns_schema",
    "output_schema",
]


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def sentence_split(text):
    return [s.strip() for s in re.split(r"[.?!]+", text) if s.strip()]


def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(words, sentences):
    # readability tests designed to indicate how difficult a passage in English is to understand.
    if not words or sentences == 0:
        return None
    syllables = sum(count_syllables(w) for w in words)
    words_count = len(words)
    return 206.835 - 1.015 * (words_count / sentences) - 84.6 * (syllables / words_count)


def flesch_kincaid_grade(words, sentences):
    # readability tests designed to indicate how difficult a passage in English is to understand.
    if not words or sentences == 0:
        return None
    syllables = sum(count_syllables(w) for w in words)
    words_count = len(words)
    return 0.39 * (words_count / sentences) + 11.8 * (syllables / words_count) - 15.59


def detect_bullets(text):
    bullet_re = re.compile(r"^\s*(?:[-*]|\u2022|\d+\.)\s+")
    return sum(1 for line in text.splitlines() if bullet_re.match(line))


def count_json_like_blocks(text):
    brace_re = re.compile(r"\{[^{}]{0,400}:[^{}]{0,400}\}", re.DOTALL)
    return len(brace_re.findall(text))


def text_metrics(text):
    if not text:
        return {}
    words = tokenize(text)
    word_count = len(words)
    unique_word_count = len(set(words))
    sentences = sentence_split(text)
    sentence_count = len(sentences)
    line_count = len(text.splitlines())
    bullet_count = detect_bullets(text)
    avg_sentence_len = (word_count / sentence_count) if sentence_count else None
    long_word_ratio = (sum(1 for w in words if len(w) >= 10) / word_count) if word_count else None

    entropy = None
    if word_count:
        counts = Counter(words)
        entropy = -sum((c / word_count) * math.log2(c / word_count) for c in counts.values())

    example_block_count = count_json_like_blocks(text)
    text_lower = text.lower()
    has_example = int("example" in text_lower or example_block_count > 0)
    has_step_by_step = int(
        "step by step" in text_lower
        or re.search(r"\bstep\s+\d+\b", text_lower) is not None
        or re.search(r"^\s*\d+\.\s+", text, re.MULTILINE) is not None
    )
    edge_case_markers = [
        "edge case",
        "corner case",
        "error",
        "exception",
        "invalid",
        "out of range",
        "failure",
    ]
    has_edge_case_guidance = int(any(marker in text_lower for marker in edge_case_markers))

    return {
        "char_count": len(text),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "line_count": line_count,
        "bullet_count": bullet_count,
        "avg_sentence_len": avg_sentence_len,
        "flesch_reading_ease": flesch_reading_ease(words, sentence_count),
        "flesch_kincaid_grade": flesch_kincaid_grade(words, sentence_count),
        "long_word_ratio": long_word_ratio,
        "unique_word_count": unique_word_count,
        "ttr": (unique_word_count / word_count) if word_count else None,
        "word_entropy": entropy,
        "repetition_rate": (1 - (unique_word_count / word_count)) if word_count else None,
        "has_example": has_example,
        "example_block_count": example_block_count,
        "has_step_by_step": has_step_by_step,
        "has_edge_case_guidance": has_edge_case_guidance,
    }


def normalize_name_tokens(name):
    return re.findall(r"[a-zA-Z]+", str(name).lower())


def name_mentioned(name, text_lower, word_set):
    tokens = normalize_name_tokens(name)
    if not tokens:
        return False
    if len(tokens) == 1:
        return tokens[0] in word_set
    phrase = " ".join(tokens)
    if phrase in text_lower:
        return True
    return all(token in word_set for token in tokens)


def extract_schema_info(schema):
    param_names = []
    required_names = set()
    enum_fields = {}
    field_types = {}
    leaf_paths = []

    def walk(node, path_prefix):
        if not isinstance(node, dict):
            return
        properties = node.get("properties", {})
        required = set(node.get("required", []) or [])
        for prop, child in properties.items():
            path = f"{path_prefix}.{prop}" if path_prefix else prop
            param_names.append(path)
            if prop in required:
                required_names.add(path)

            if isinstance(child, dict):
                if "enum" in child:
                    enum_fields[path] = set(child.get("enum") or [])
                field_type = child.get("type")
                if field_type is not None:
                    if isinstance(field_type, list):
                        field_types[path] = [t for t in field_type if isinstance(t, str)]
                    elif isinstance(field_type, str):
                        field_types[path] = [field_type]

                child_props = child.get("properties")
                if child_props:
                    walk(child, path)
                    continue

                if child.get("type") == "array":
                    items = child.get("items")
                    if isinstance(items, dict) and (items.get("properties") or items.get("type") == "object"):
                        walk(items, path)
                        continue

            leaf_paths.append(path)

    walk(schema, "")
    return {
        "param_names": param_names,
        "required_names": required_names,
        "enum_fields": enum_fields,
        "field_types": field_types,
        "leaf_paths": leaf_paths,
    }


def schema_metrics(schema_info, text):
    if not schema_info or not text:
        return {}
    text_lower = text.lower()
    word_set = set(tokenize(text))

    param_names = schema_info["param_names"]
    required_names = schema_info["required_names"]
    enum_fields = schema_info["enum_fields"]
    field_types = schema_info["field_types"]
    leaf_paths = schema_info["leaf_paths"]
    nested_leaf_paths = [p for p in leaf_paths if "." in p]

    def coverage_for(names):
        if not names:
            return None
        hit = sum(1 for name in names if name_mentioned(name, text_lower, word_set))
        return hit / len(names)

    param_name_coverage = coverage_for(param_names)
    required_param_coverage = coverage_for(required_names)
    nested_leaf_coverage = coverage_for(nested_leaf_paths)

    type_keywords = {
        "string": ["string", "str", "text"],
        "integer": ["int", "integer", "number"],
        "number": ["number", "float", "double"],
        "boolean": ["boolean", "bool"],
        "array": ["array", "list"],
        "object": ["object", "dict", "map"],
        "null": ["null"],
    }
    type_hits = 0
    type_total = 0
    for name, types in field_types.items():
        if not types:
            continue
        type_total += 1
        keywords = []
        for t in types:
            keywords.extend(type_keywords.get(t.lower(), [t.lower()]))
        if any(kw in text_lower for kw in keywords):
            type_hits += 1
    type_coverage = (type_hits / type_total) if type_total else None

    enum_value_mentions = 0
    enum_field_hits = 0
    for field, values in enum_fields.items():
        field_hit = False
        for value in values:
            if isinstance(value, (int, float)):
                if re.search(rf"\b{re.escape(str(value))}\b", text_lower):
                    enum_value_mentions += 1
                    field_hit = True
            else:
                value_str = str(value).lower()
                if value_str and value_str in text_lower:
                    enum_value_mentions += 1
                    field_hit = True
        if field_hit:
            enum_field_hits += 1
    enum_field_coverage = (enum_field_hits / len(enum_fields)) if enum_fields else None

    constraint_terms = [
        "min",
        "max",
        "minimum",
        "maximum",
        "format",
        "iso",
        "date",
        "datetime",
        "time",
        "range",
        "pattern",
        "regex",
        "must",
        "required",
        "optional",
        "length",
        "between",
        "less than",
        "greater than",
    ]
    constraint_mention_count = sum(len(re.findall(rf"\b{re.escape(term)}\b", text_lower)) for term in constraint_terms)

    return {
        "param_name_coverage": param_name_coverage,
        "required_param_coverage": required_param_coverage,
        "type_coverage": type_coverage,
        "enum_value_mentions": enum_value_mentions,
        "enum_field_coverage": enum_field_coverage,
        "constraint_mention_count": constraint_mention_count,
        "nested_leaf_coverage": nested_leaf_coverage,
    }


def output_schema_metrics(output_schema, text):
    if not output_schema or not text:
        return {}
    text_lower = text.lower()
    word_set = set(tokenize(text))
    info = extract_schema_info(output_schema)
    return_fields = info["param_names"]
    return_field_coverage = None
    if return_fields:
        hit = sum(1 for name in return_fields if name_mentioned(name, text_lower, word_set))
        return_field_coverage = hit / len(return_fields)
    mentions_returns = int(
        any(term in text_lower for term in ["return", "returns", "output", "result"])
    )
    return {
        "mentions_returns": mentions_returns,
        "return_field_coverage": return_field_coverage,
    }


def safe_get_first(obj, keys):
    for key in keys:
        if key in obj and obj[key]:
            return obj[key]
    return None


def extract_texts(entry):
    augmented = safe_get_first(entry, DESCRIPTION_KEYS)
    original_texts = []
    if isinstance(entry.get("orig_descriptions"), list):
        for item in entry["orig_descriptions"]:
            if isinstance(item, dict):
                text = safe_get_first(item, ORIG_DESCRIPTION_KEYS)
                if text:
                    original_texts.append(text)
            elif isinstance(item, str):
                original_texts.append(item)
    if not original_texts:
        for key in ORIG_DESCRIPTION_KEYS:
            if key in entry and entry[key] and entry[key] != augmented:
                original_texts.append(entry[key])
                break
    return augmented, original_texts


def extract_category(entry):
    category = "unknown"
    subcategory = "unknown"
    subsubcategory = "unknown"
    for key in SUBSUBCATEGORY_KEYS:
        if key in entry and entry[key]:
            value = entry[key]
            if isinstance(value, list) and value:
                subsubcategory = str(value[0])
            else:
                subsubcategory = str(value)
            break
    for key in CATEGORY_KEYS:
        if key in entry and entry[key]:
            value = entry[key]
            if isinstance(value, list) and value:
                category = str(value[0])
            else:
                category = str(value)
            break
    for key in SUBCATEGORY_KEYS:
        if key in entry and entry[key]:
            value = entry[key]
            if isinstance(value, list) and value:
                subcategory = str(value[0])
            else:
                subcategory = str(value)
            break
    if category == "unknown":
        source_files = entry.get("source_files")
        if isinstance(source_files, list) and source_files:
            category = str(source_files[0])
    if subcategory == "unknown":
        task_types = entry.get("task_types")
        if isinstance(task_types, list) and task_types:
            subcategory = str(task_types[0])
    return category, subcategory, subsubcategory


def extract_schema(entry):
    if "parameters_variants" in entry and isinstance(entry["parameters_variants"], list):
        for variant in entry["parameters_variants"]:
            if isinstance(variant, dict) and "parameters" in variant:
                return variant["parameters"]
    return safe_get_first(entry, SCHEMA_KEYS)


def extract_output_schema(entry):
    return safe_get_first(entry, OUTPUT_SCHEMA_KEYS)


def compute_metrics_for_text(text, schema_info, output_schema, include_schema_metrics=True):
    if not text:
        return {}
    metrics = {}
    metrics.update(text_metrics(text))
    if include_schema_metrics:
        metrics.update(schema_metrics(schema_info, text))
        metrics.update(output_schema_metrics(output_schema, text))
    return metrics


def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    all_keys = set().union(*metrics_list)
    averaged = {}
    for key in all_keys:
        values = [m.get(key) for m in metrics_list if m.get(key) is not None]
        if not values:
            averaged[key] = None
        else:
            averaged[key] = mean(values)
    return averaged


def quantiles(values):
    if not values:
        return None, None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    def pct(p):
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)
    return pct(0.25), pct(0.75)


def summarize_group(rows, metric_names):
    summary = []
    for metric in metric_names:
        values = [row.get(metric) for row in rows if row.get(metric) is not None]
        if not values:
            continue
        p25, p75 = quantiles(values)
        summary.append({
            "metric_name": metric,
            "mean": mean(values),
            "median": median(values),
            "p25": p25,
            "p75": p75,
            "n": len(values),
        })
    return summary


def detect_mapping(entries):
    def find_key(keys, nested=False):
        for key in keys:
            if nested:
                if any(isinstance(e.get("orig_descriptions"), list) for e in entries):
                    return "orig_descriptions[].description"
            if any(key in e for e in entries):
                return key
        return "unknown"

    mapping = {
        "augmented_description": find_key(DESCRIPTION_KEYS),
        "original_description": "orig_descriptions[].description"
        if any(isinstance(e.get("orig_descriptions"), list) for e in entries)
        else find_key(ORIG_DESCRIPTION_KEYS),
        "category": find_key(CATEGORY_KEYS),
        "subcategory": find_key(SUBCATEGORY_KEYS),
        "schema": "parameters_variants[].parameters"
        if any("parameters_variants" in e for e in entries)
        else find_key(SCHEMA_KEYS),
        "output_schema": find_key(OUTPUT_SCHEMA_KEYS),
        "tool_id": find_key(["tool_id", "tool_name", "name", "function_name", "id"]),
    }
    return mapping


def print_plan_and_mapping(entries):
    mapping = detect_mapping(entries)
    print("Plan:")
    print("- Load catalogue JSON/JSONL and normalize description/category/schema fields.")
    print("- Compute per-tool metrics for augmented and original descriptions, plus deltas.")
    print("- Aggregate metrics overall, by category, and by subcategory; write CSVs and report.")
    print("Assumed field mapping (first 20 entries):")
    for key, value in mapping.items():
        print(f"- {key}: {value}")


def read_catalogue(path):
    entries = []
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            entries = data
        else:
            entries = [data]
        return entries
    except json.JSONDecodeError:
        entries = []
        for idx, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: bad JSON on line {idx}, skipping")
        return entries


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue_path", required=True)
    parser.add_argument("--out_dir", default="descriptiveness_out")
    args = parser.parse_args()

    catalogue_path = Path(args.catalogue_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = read_catalogue(catalogue_path)
    print_plan_and_mapping(entries[:20])

    tool_rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            print("Warning: entry is not an object, skipping")
            continue

        tool_id = (
            entry.get("tool_id")
            or entry.get("tool_name")
            or entry.get("name")
            or entry.get("function_name")
            or entry.get("id")
            or ""
        )
        category, subcategory, subsubcategory = extract_category(entry)
        augmented_text, original_texts = extract_texts(entry)

        schema = extract_schema(entry)
        schema_info = extract_schema_info(schema) if isinstance(schema, dict) else None
        output_schema = extract_output_schema(entry)

        aug_metrics = compute_metrics_for_text(augmented_text, schema_info, output_schema, include_schema_metrics=True)
        orig_metrics_list = [
            compute_metrics_for_text(text, schema_info, output_schema, include_schema_metrics=False)
            for text in original_texts
            if text
        ]
        orig_metrics = average_metrics(orig_metrics_list)

        combined = {
            "tool_id": tool_id,
            "category": category,
            "subcategory": subcategory,
            "subsubcategory": subsubcategory,
        }

        metric_keys = set(aug_metrics) | set(orig_metrics)
        for key in metric_keys:
            combined[f"aug_{key}"] = aug_metrics.get(key)
            combined[f"orig_{key}"] = orig_metrics.get(key)
            aug_val = aug_metrics.get(key)
            orig_val = orig_metrics.get(key)
            if aug_val is None or orig_val is None:
                combined[f"delta_{key}"] = None
            else:
                combined[f"delta_{key}"] = aug_val - orig_val

        tool_rows.append(combined)

    if not tool_rows:
        print("No valid tool entries found.")
        return

    metric_names = sorted({k for row in tool_rows for k in row.keys() if k.startswith(("aug_", "orig_", "delta_"))})
    tool_fieldnames = ["tool_id", "category", "subcategory", "subsubcategory"] + metric_names
    write_csv(out_dir / "tool_level_metrics.csv", tool_rows, tool_fieldnames)

    grouped = defaultdict(list)
    grouped_overall = {"overall": tool_rows}
    for row in tool_rows:
        grouped[row["category"]].append(row)

    sub_grouped = defaultdict(list)
    for row in tool_rows:
        key = f"{row['category']}/{row['subcategory']}"
        sub_grouped[key].append(row)

    sub_sub_grouped = defaultdict(list)
    for row in tool_rows:
        key = f"{row['category']}/{row['subcategory']}/{row['subsubcategory']}"
        sub_sub_grouped[key].append(row)

    def build_aggregate_rows(group_level, groups):
        rows = []
        for group_name, rows_in_group in groups.items():
            summaries = summarize_group(rows_in_group, metric_names)
            for summary in summaries:
                rows.append({
                    "group_level": group_level,
                    "group_name": group_name,
                    **summary,
                })
        return rows

    aggregate_overall = build_aggregate_rows("overall", grouped_overall)
    aggregate_category = build_aggregate_rows("category", grouped)
    aggregate_subcategory = build_aggregate_rows("subcategory", sub_grouped)
    aggregate_subsubcategory = build_aggregate_rows("subsubcategory", sub_sub_grouped)

    aggregate_fields = ["group_level", "group_name", "metric_name", "mean", "median", "p25", "p75", "n"]
    write_csv(out_dir / "aggregate_overall.csv", aggregate_overall, aggregate_fields)
    write_csv(out_dir / "aggregate_by_category.csv", aggregate_category, aggregate_fields)
    write_csv(out_dir / "aggregate_by_subcategory.csv", aggregate_subcategory, aggregate_fields)
    write_csv(out_dir / "aggregate_by_subsubcategory.csv", aggregate_subsubcategory, aggregate_fields)

    report_lines = []
    report_lines.append(f"Tools: {len(tool_rows)}")
    report_lines.append(f"Categories: {len(grouped)}")
    report_lines.append(f"Subcategories: {len(sub_grouped)}")
    report_lines.append(f"Subsubcategories: {len(sub_sub_grouped)}")

    def median_metric(rows, metric):
        vals = [r.get(metric) for r in rows if r.get(metric) is not None]
        return median(vals) if vals else None

    category_param_medians = []
    for cat, rows in grouped.items():
        median_val = median_metric(rows, "aug_param_name_coverage")
        if median_val is not None:
            category_param_medians.append((cat, median_val))

    category_param_medians.sort(key=lambda x: x[1], reverse=True)
    report_lines.append("")
    report_lines.append("Top 5 categories by median augmented param_name_coverage:")
    for cat, val in category_param_medians[:5]:
        report_lines.append(f"- {cat}: {val:.3f}")
    report_lines.append("")
    report_lines.append("Bottom 5 categories by median augmented param_name_coverage:")
    for cat, val in category_param_medians[-5:]:
        report_lines.append(f"- {cat}: {val:.3f}")

    def summarize_metric(metric):
        vals = [r.get(metric) for r in tool_rows if r.get(metric) is not None]
        if not vals:
            return "n/a"
        return f"mean={mean(vals):.3f}, median={median(vals):.3f}"

    report_lines.append("")
    report_lines.append("Key metrics summary (augmented vs original):")
    for base in ["word_count", "param_name_coverage", "required_param_coverage", "has_example"]:
        report_lines.append(
            f"- {base}: aug {summarize_metric('aug_' + base)} | orig {summarize_metric('orig_' + base)}"
        )
    report_lines.append("")
    report_lines.append("Key metric deltas (aug - orig):")
    for base in ["word_count", "param_name_coverage", "required_param_coverage", "has_example"]:
        report_lines.append(f"- {base}: {summarize_metric('delta_' + base)}")

    (out_dir / "descriptiveness_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

import argparse
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def load_catalogue(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    mapping: dict[str, str] = {}
    normalized: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            name = obj.get("tool_name")
            if name:
                mapping[name] = obj.get("aug_description")
                normalized_name = name.replace(".", "_")
                normalized.setdefault(normalized_name, []).append(name)
    return mapping, normalized


def extract_category_from_id(test_id: str) -> str:
    # Mirror bfcl_eval.utils.extract_test_category_from_id behavior.
    if ":" in test_id:
        test_id = test_id.split(":")[0]
    return test_id.rsplit("_", 1)[0]


def iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*_result.json"))


def parse_callable_name(desc: str) -> str | None:
    marker = "Callable name:"
    if marker not in desc:
        return None
    after = desc.split(marker, 1)[1]
    end_idx = after.find(". ")
    if end_idx == -1:
        return after.strip() or None
    return after[:end_idx].strip() or None


def detect_mode(results_dir: Path) -> str:
    for path in results_dir.rglob("*_result.json"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries = [obj] if isinstance(obj, dict) else [x for x in obj if isinstance(x, dict)]
                for entry in entries:
                    inf_logs = [
                        x
                        for x in entry.get("inference_log", [])
                        if isinstance(x, dict) and x.get("role") == "inference_input"
                    ]
                    if not inf_logs:
                        continue
                    tools = inf_logs[0].get("content", {}).get("tools", [])
                    for tool in tools:
                        if not isinstance(tool, dict):
                            continue
                        desc = tool.get("description")
                        if isinstance(desc, str) and desc.startswith("Display name:"):
                            return "augmented_names"
        break
    return "original_names"


def best_ratio(desc: str, catalogue: dict[str, str], candidates: list[str]) -> float:
    best = 0.0
    for candidate in candidates:
        aug = catalogue.get(candidate)
        if not aug:
            continue
        ratio = SequenceMatcher(None, desc, aug).ratio()
        if ratio > best:
            best = ratio
    return best


def find_missing_augmented_ids(
    results_dir: Path, catalogue: dict[str, str], normalized_map: dict[str, list[str]]
) -> dict[str, set[str]]:
    ids_by_category: dict[str, set[str]] = defaultdict(set)
    prefix = "Display name:"
    lang_suffixes = (
        " Note that the provided function is in Java 8 SDK syntax.",
        " Note that the provided function is in JavaScript syntax.",
        " Note that the provided function is in Python 3 syntax.",
    )
    mode = detect_mode(results_dir)
    for path in iter_result_files(results_dir):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries = [obj] if isinstance(obj, dict) else [x for x in obj if isinstance(x, dict)]
                for entry in entries:
                    test_id = entry.get("id")
                    if not test_id:
                        continue
                    inf_logs = [
                        x
                        for x in entry.get("inference_log", [])
                        if isinstance(x, dict) and x.get("role") == "inference_input"
                    ]
                    if not inf_logs:
                        continue
                    tools = inf_logs[0].get("content", {}).get("tools", [])
                    for tool in tools:
                        if not isinstance(tool, dict):
                            continue
                        name = tool.get("name")
                        desc = tool.get("description")
                        if not name or not isinstance(desc, str):
                            continue
                        callable_name = parse_callable_name(desc)
                        trimmed = desc
                        if trimmed.startswith(prefix) and "Callable name:" in trimmed:
                            parts = trimmed.split("Callable name:", 1)
                            if len(parts) == 2:
                                after = parts[1]
                                dot_idx = after.find(". ")
                                if dot_idx != -1:
                                    trimmed = after[dot_idx + 2 :]
                        for suffix in lang_suffixes:
                            if trimmed.endswith(suffix):
                                trimmed = trimmed[: -len(suffix)]
                                break
                        candidates: list[str] = []
                        if mode == "augmented_names":
                            if callable_name:
                                candidates.append(callable_name)
                        else:
                            if callable_name:
                                candidates.append(callable_name)
                            if name:
                                candidates.append(name)
                            candidates.extend(normalized_map.get(name, []))
                        candidates = list(dict.fromkeys(candidates))
                        if not candidates:
                            continue
                        ratio = best_ratio(trimmed, catalogue, candidates)
                        if ratio < 0.50:
                            category = extract_category_from_id(test_id)
                            ids_by_category[category].add(test_id)
                            break
    return ids_by_category


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find result entries missing augmented descriptions and write run-ids JSON."
    )
    parser.add_argument(
        "--results-dir",
        # default="result_desc_augmented_name_augmented",
        default="result_desc_augmented_name_original",
        help="Root results directory to scan.",
    )
    parser.add_argument(
        "--catalogue",
        default="bfcl_eval/data/internal/bfcl_v4_tool_catalogue_augmented_51_v2.jsonl",
        help="Augmented tool catalogue JSONL.",
    )
    parser.add_argument(
        "--output",
        default="test_case_ids_to_generate.json",
        help="Output JSON file with test ids to regenerate.",
    )
    parser.add_argument(
        "--model",
        default="azure-gpt-5.1-responses-FC",
        help="Restrict scan to a single model folder under results-dir (e.g. azure-gpt-5.1-responses-FC).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "augmented_names", "original_names"],
        default="auto",
        help="How to match tool names; auto detects Display name prefix in descriptions.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    catalogue_path = Path(args.catalogue)
    output_path = Path(args.output)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")
    if not catalogue_path.exists():
        raise FileNotFoundError(f"Catalogue not found: {catalogue_path}")

    catalogue, normalized_map = load_catalogue(catalogue_path)

    if args.model:
        model_dir = results_dir / args.model
        if not model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {model_dir}")
        if args.mode != "auto":
            global detect_mode
            detect_mode = lambda _path: args.mode
        ids_by_category = find_missing_augmented_ids(model_dir, catalogue, normalized_map)
        payload = {k: sorted(list(v)) for k, v in sorted(ids_by_category.items())}
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        total_ids = sum(len(v) for v in ids_by_category.values())
        print(f"Wrote {total_ids} ids across {len(ids_by_category)} categories to {output_path}")
        return

    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        raise FileNotFoundError(f"No model folders found under: {results_dir}")

    for model_dir in sorted(model_dirs, key=lambda p: p.name):
        ids_by_category = find_missing_augmented_ids(model_dir, catalogue, normalized_map)
        payload = {k: sorted(list(v)) for k, v in sorted(ids_by_category.items())}
        model_output = output_path.with_name(
            f"{output_path.stem}_{model_dir.name}{output_path.suffix}"
        )
        with model_output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        total_ids = sum(len(v) for v in ids_by_category.values())
        print(
            f"Wrote {total_ids} ids across {len(ids_by_category)} categories to {model_output}"
        )


if __name__ == "__main__":
    main()

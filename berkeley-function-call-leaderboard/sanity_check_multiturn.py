import argparse
import csv
import json
import os
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


DECODE_FAIL_STR = "Failed to decode the model response. Proceed to next turn."
DECODE_FAIL_LOG_STR = "Error decoding the model response. Proceed to next turn."

TOOL_KEYS = {
    "tool_calls",
    "function_call",
    "function_calls",
    "calls",
    "invoked_tool",
    "invoked_tools",
    "tool",
    "tool_call",
}
FINAL_KEYS = {"final_answer", "answer", "final", "response"}


def _iter_json_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        content = f.read().strip()
    # Some "json" files are actually JSONL.
    if "\n" in content:
        return list(_read_jsonl(path))
    return json.loads(content)


def _walk(node: Any) -> Iterable[Any]:
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def _find_strings(node: Any) -> Iterable[str]:
    for cur in _walk(node):
        if isinstance(cur, str):
            yield cur


def _has_final_marker(node: Any) -> bool:
    for cur in _walk(node):
        if isinstance(cur, dict):
            for key, value in cur.items():
                if key in FINAL_KEYS and value not in (None, "", [], {}):
                    return True
    return False


def _looks_like_tool_call(node: Any) -> bool:
    if isinstance(node, dict):
        if any(k in node for k in TOOL_KEYS):
            return True
        if len(node) == 1:
            key = next(iter(node.keys()))
            value = node[key]
            if isinstance(value, (dict, list, str)):
                return True
    if isinstance(node, list):
        for item in node:
            if _looks_like_tool_call(item):
                return True
    return False


def _classify_step(step: Any) -> str:
    if _looks_like_tool_call(step):
        return "tool"
    if isinstance(step, str):
        return "text"
    if isinstance(step, list):
        if any(isinstance(item, str) for item in step):
            return "text"
    return "empty"


def _extract_steps_from_result(result: Any) -> List[Any]:
    if isinstance(result, list):
        steps = []
        for turn in result:
            if isinstance(turn, list):
                steps.extend(turn)
        if steps:
            return steps
    return []


def _extract_steps_from_inference_log(inference_log: Any) -> List[Any]:
    steps = []
    if not isinstance(inference_log, list):
        return steps
    for turn in inference_log:
        if not isinstance(turn, dict):
            continue
        for key, value in turn.items():
            if key.startswith("step_") and isinstance(value, list):
                steps.append(value)
    return steps


def _pick_first_snippet(node: Any, max_len: int = 220) -> str:
    for s in _find_strings(node):
        snippet = s.replace("\n", " ")
        if snippet:
            return snippet[:max_len]
    return ""


def _load_scores(
    scores_root: Path, model: str, category: str
) -> Tuple[Dict[str, bool], Optional[str], Dict[str, int]]:
    scores_dir = scores_root / model / category
    id_to_correct: Dict[str, bool] = {}
    summary_counts = {"correct_count": 0, "total_count": 0, "valid_true": 0}
    if not scores_dir.exists():
        return id_to_correct, None, summary_counts

    score_files = list(scores_dir.glob("*_score.json"))
    if not score_files:
        return id_to_correct, None, summary_counts

    for score_file in score_files:
        try:
            entries = _read_json(score_file)
        except Exception as exc:
            print(f"[warn] Failed to read score file {score_file}: {exc}")
            continue

        if isinstance(entries, dict):
            entries = [entries]

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "correct_count" in entry and "total_count" in entry:
                summary_counts["correct_count"] += int(entry["correct_count"])
                summary_counts["total_count"] += int(entry["total_count"])
            if "id" in entry:
                valid = entry.get("valid")
                if valid is None and "error" in entry:
                    valid = False
                if isinstance(valid, bool):
                    id_to_correct[entry["id"]] = valid
                    if valid:
                        summary_counts["valid_true"] += 1
            elif "errors" in entry and isinstance(entry["errors"], list):
                for err in entry["errors"]:
                    if isinstance(err, dict) and "id" in err:
                        id_to_correct[err["id"]] = False

    return id_to_correct, str(scores_dir), summary_counts


def _collect_items(results_root: Path, model: str, category: str) -> List[Tuple[Path, dict]]:
    base = results_root / model / category
    if not base.exists():
        return []
    items = []
    for path in base.rglob("*.json"):
        try:
            entries = _read_json(path)
        except Exception as exc:
            print(f"[warn] Failed to read result file {path}: {exc}")
            continue
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, dict):
                items.append((path, entry))
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--scores_root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--category", default="multi_turn")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    scores_root = Path(args.scores_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _collect_items(results_root, args.model, args.category)
    id_to_correct, scores_dir, summary_counts = _load_scores(
        scores_root, args.model, args.category
    )

    rows = []
    decode_fail_items = []
    ok_items = []

    for path, entry in items:
        item_id = entry.get("id") or path.stem
        result = entry.get("result", [])
        inference_log = entry.get("inference_log", [])

        steps = _extract_steps_from_result(result)
        if not steps:
            steps = _extract_steps_from_inference_log(inference_log)

        steps_total = len(steps)
        decode_fail_count = sum(
            1
            for s in _find_strings(entry)
            if s in (DECODE_FAIL_STR, DECODE_FAIL_LOG_STR)
        )

        tool_call_turns = 0
        text_only_turns = 0
        empty_turns = 0
        for step in steps:
            kind = _classify_step(step)
            if kind == "tool":
                tool_call_turns += 1
            elif kind == "text":
                text_only_turns += 1
            else:
                empty_turns += 1

        completed_flag = _has_final_marker(entry)
        correct = id_to_correct.get(item_id)

        snippet = ""
        if decode_fail_count > 0:
            snippet = _pick_first_snippet(inference_log)
            decode_fail_items.append((decode_fail_count, item_id, path, snippet))
        else:
            ok_items.append((item_id, path, _pick_first_snippet(result)))

        rows.append(
            {
                "id": item_id,
                "file_path": str(path),
                "steps_total": steps_total,
                "decode_fail_count": decode_fail_count,
                "tool_call_turns": tool_call_turns,
                "text_only_turns": text_only_turns,
                "empty_or_unparsed_turns": empty_turns,
                "completed_flag": completed_flag,
                "correct": correct,
            }
        )

    csv_path = out_dir / "multiturn_item_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()) if rows else [],
        )
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    total_items = len(rows)
    items_with_fail = sum(1 for r in rows if r["decode_fail_count"] > 0)
    steps_list = [r["steps_total"] for r in rows] or [0]
    tool_list = [r["tool_call_turns"] for r in rows] or [0]
    text_list = [r["text_only_turns"] for r in rows] or [0]

    def _acc(subset: List[dict]) -> Optional[float]:
        vals = [r for r in subset if r["correct"] is not None]
        if not vals:
            return None
        return sum(1 for r in vals if r["correct"]) / len(vals)

    with_fail = [r for r in rows if r["decode_fail_count"] > 0]
    without_fail = [r for r in rows if r["decode_fail_count"] == 0]

    summary_path = out_dir / "multiturn_sanity_summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Multi-turn Sanity Summary\n\n")
        f.write(f"- results_root: `{results_root}`\n")
        f.write(f"- scores_root: `{scores_root}`\n")
        f.write(f"- model: `{args.model}`\n")
        f.write(f"- category: `{args.category}`\n")
        if scores_dir:
            f.write(f"- scores_dir: `{scores_dir}`\n")
        f.write("\n## Aggregates\n\n")
        f.write(f"- total_items: {total_items}\n")
        f.write(
            f"- items_with_any_decode_fail: {items_with_fail} "
            f"({(items_with_fail/total_items*100) if total_items else 0:.2f}%)\n"
        )
        f.write(f"- avg_steps_total: {sum(steps_list)/len(steps_list):.2f}\n")
        f.write(f"- median_steps_total: {median(steps_list):.2f}\n")
        f.write(f"- avg_tool_call_turns: {sum(tool_list)/len(tool_list):.2f}\n")
        f.write(f"- avg_text_only_turns: {sum(text_list)/len(text_list):.2f}\n")

        f.write("\n## Accuracy Split (if available)\n\n")
        acc_fail = _acc(with_fail)
        acc_ok = _acc(without_fail)
        f.write(f"- accuracy_with_decode_fail: {acc_fail if acc_fail is not None else 'n/a'}\n")
        f.write(f"- accuracy_without_decode_fail: {acc_ok if acc_ok is not None else 'n/a'}\n")
        f.write(f"- items_with_known_correctness: {len([r for r in rows if r['correct'] is not None])}\n")
        if summary_counts["total_count"]:
            f.write(
                f"- score_summary_correct_count: {summary_counts['correct_count']}\n"
            )
            f.write(f"- score_summary_total_count: {summary_counts['total_count']}\n")
            f.write(f"- score_entries_valid_true: {summary_counts['valid_true']}\n")

        f.write("\n## Examples\n\n")
        decode_fail_items.sort(reverse=True)
        ok_items.sort()
        borderline = sorted(with_fail, key=lambda r: r["decode_fail_count"])[:1]

        f.write("### High decode-fail items\n\n")
        for count, item_id, path, snippet in decode_fail_items[:2]:
            f.write(f"- `{item_id}` ({count} fails) in `{path}`\n")
            if snippet:
                f.write(f"  - snippet: {snippet}\n")

        f.write("\n### Zero decode-fail items\n\n")
        for item_id, path, snippet in ok_items[:2]:
            f.write(f"- `{item_id}` in `{path}`\n")
            if snippet:
                f.write(f"  - snippet: {snippet}\n")

        f.write("\n### Borderline item\n\n")
        if borderline:
            item = borderline[0]
            f.write(
                f"- `{item['id']}` ({item['decode_fail_count']} fails) in `{item['file_path']}`\n"
            )
        else:
            f.write("- n/a\n")

        f.write("\n## Conclusion\n\n")
        if acc_fail is None or acc_ok is None or summary_counts["valid_true"] == 0:
            f.write(
                "Cannot determine accuracy impact because per-item correctness "
                "was not found in scores (or only invalid entries are present). "
                "Use score JSONL with per-item validity for this split.\n"
            )
        else:
            if acc_fail < acc_ok:
                f.write(
                    "Items with decode-fail logs show lower accuracy, suggesting "
                    "decode failures likely contaminate evaluation.\n"
                )
            else:
                f.write(
                    "Accuracy is similar across decode-fail groups, suggesting "
                    "decode failures may not be the dominant source of error.\n"
                )

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

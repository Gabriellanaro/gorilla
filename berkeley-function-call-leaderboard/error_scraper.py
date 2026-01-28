import argparse
import csv
import os
import re
from collections import defaultdict


ERROR_LINE_RE = re.compile(
    r"Error during inference:\s*(.*?)(?:\"\\s*,\\s*\"traceback\"|\\r?\\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def categorize_error(msg: str) -> str:
    msg_lower = msg.lower()

    if "error code:" in msg_lower:
        m = re.search(r"error code:\s*(\d+)", msg_lower)
        code = m.group(1) if m else "unknown"
        if "internalservererror" in msg_lower:
            return f"Error code {code} InternalServerError"
        if "content_filter" in msg_lower:
            return f"Error code {code} content_filter"
        if "context window" in msg_lower:
            return f"Error code {code} context_window"
        if "no tool output found" in msg_lower:
            return f"Error code {code} no_tool_output"
        if "invalid 'input" in msg_lower and "string too long" in msg_lower:
            return f"Error code {code} output_too_long"
        if "timeout" in msg_lower:
            return f"Error code {code} timeout"
        return f"Error code {code} other"

    if "connection error" in msg_lower:
        return "Connection error"
    if "failed to invoke the azure cli" in msg_lower:
        return "Failed to invoke the Azure CLI"
    if "timeout" in msg_lower:
        return "Timeout"

    return "Unknown error"


def extract_errors(text: str) -> list[str]:
    errors = []
    for match in ERROR_LINE_RE.finditer(text):
        msg = match.group(1).strip()
        if not msg:
            continue
        errors.append(categorize_error(msg))
    return errors


def iter_result_json_files(root: str) -> list[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        in_results = os.sep + "results" + os.sep in (os.sep + dirpath + os.sep)
        in_result_desc = "result_desc" in dirpath
        if not in_results and not in_result_desc:
            continue
        for fn in filenames:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(dirpath, fn))
    return files


def read_json_or_jsonl(path: str) -> list[dict]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
                    return []
                if isinstance(data, list):
                    return [row for row in data if isinstance(row, dict)]
                if isinstance(data, dict):
                    return [data]
                return []
            # JSONL
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
            return rows
    except OSError:
        return []


def extract_row_text(row: dict) -> str:
    parts = []
    for key in ("result", "traceback", "error", "message"):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def scan(root: str) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, set[str]]]:
    folder_summary = defaultdict(lambda: defaultdict(int))
    file_summary = defaultdict(lambda: defaultdict(int))
    file_ids = defaultdict(set)
    files = iter_result_json_files(root)

    for path in files:
        rel = os.path.relpath(path, root)
        folder = rel.split(os.sep)[0] if rel else rel
        rows = read_json_or_jsonl(path)
        if not rows:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = extract_row_text(row)
            if not text:
                continue
            errors = extract_errors(text)
            if not errors:
                continue
            item_id = row.get("id")
            if item_id:
                file_ids[rel].add(str(item_id))
            for err in errors:
                folder_summary[folder][err] += 1
                file_summary[rel][err] += 1

    return folder_summary, file_summary, file_ids


def write_csv(summary: dict[str, dict[str, int]], out_path: str, id_label: str) -> None:
    error_types = sorted(
        {
            err
            for err_map in summary.values()
            for err, count in err_map.items()
            if count > 0
        }
    )

    if not error_types:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([id_label])
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([id_label, *error_types])
        for key in sorted(summary.keys()):
            err_map = summary[key]
            if not any(err_map.get(err, 0) > 0 for err in error_types):
                continue
            row = [key]
            for err in error_types:
                count = err_map.get(err, 0)
                row.append("" if count == 0 else count)
            writer.writerow(row)


def write_csv_with_file(
    summary: dict[str, dict[str, int]],
    out_path: str,
    file_ids: dict[str, set[str]],
) -> None:
    error_types = sorted(
        {
            err
            for err_map in summary.values()
            for err, count in err_map.items()
            if count > 0
        }
    )

    if not error_types:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["folder", "model", "file", "task_ids"])
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "model", "file", "task_ids", *error_types])
        for rel_path in sorted(summary.keys()):
            err_map = summary[rel_path]
            if not any(err_map.get(err, 0) > 0 for err in error_types):
                continue
            parts = rel_path.split(os.sep)
            folder = parts[0] if parts else rel_path
            model = parts[1] if len(parts) > 1 else ""
            filename = parts[-1] if parts else rel_path
            ids = ";".join(sorted(file_ids.get(rel_path, set())))
            row = [folder, model, filename, ids]
            for err in error_types:
                count = err_map.get(err, 0)
                row.append("" if count == 0 else count)
            writer.writerow(row)


def write_json(summary: dict[str, dict[str, int]], out_path: str) -> None:
    data = {
        key: {err: count for err, count in err_map.items() if count > 0}
        for key, err_map in summary.items()
        if any(count > 0 for count in err_map.values())
    }
    with open(out_path, "w", encoding="utf-8") as f:
        import json

        json.dump(data, f, indent=2, sort_keys=True)


def write_json_with_file(summary: dict[str, dict[str, int]], out_path: str) -> None:
    rows = []
    for rel_path, err_map in summary.items():
        if not any(count > 0 for count in err_map.values()):
            continue
        parts = rel_path.split(os.sep)
        folder = parts[0] if parts else rel_path
        filename = parts[-1] if parts else rel_path
        rows.append(
            {
                "folder": folder,
                "file": filename,
                "errors": {err: count for err, count in err_map.items() if count > 0},
            }
        )
    with open(out_path, "w", encoding="utf-8") as f:
        import json

        json.dump(rows, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan result folders for inference errors and output a CSV summary."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root folder to scan (default: current directory).",
    )
    parser.add_argument(
        "--out",
        default="error_summary.csv",
        help="CSV output path (default: error_summary.csv).",
    )
    parser.add_argument(
        "--json-out",
        default="error_summary.json",
        help="JSON output path (default: error_summary.json).",
    )
    parser.add_argument(
        "--by-folder",
        action="store_true",
        help="Summarize by top-level folder instead of folder+file.",
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.exists(root):
        print(f"Warning: root not found: {root}. Falling back to current directory.")
        root = "."

    folder_summary, file_summary, file_ids = scan(root)
    if args.by_folder:
        write_csv(folder_summary, args.out, "folder")
        write_json(folder_summary, args.json_out)
    else:
        write_csv_with_file(file_summary, args.out, file_ids)
        # Keep JSON at folder summary level by default.
        write_json(folder_summary, args.json_out)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

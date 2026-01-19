#!/usr/bin/env python3
"""
build_bfcl_v4_internal_jsonl.py

Loops over ALL BFCL_v4_*.json files in a folder, merges each example with its
corresponding entry in possible_answer/, and writes ONE combined internal JSONL
dataset in the schema we agreed on.

Output item schema (exact labels):
{
  "id": str,
  "source_file": str,          # filename without "BFCL_v4_" and without extension
  "task_type": "single_turn" | "multi_turn",
  "messages": [
    {"role": str, "content": str, "tool_name": str|None, "tool_arguments": dict|None}
  ],
  "tools": [
    {"name": str, "orig_description": str, "aug_description": None, "parameters": dict}
  ],
  "ground_truth": {"turn_index": int|None, "tool_name": str|None, "tool_arguments": dict|None}
}

Key decisions implemented:
- Include ALL BFCL_v4_*.json files.
- source_file = filename without BFCL_v4_ prefix and .json extension.
- task_type = "multi_turn" if "multi_turn" in source_file else "single_turn".
- Join by raw id between question file and possible_answer file.
- Output ONE ROW PER GT TOOL-CALL DECISION when we can extract multiple calls.
- Output ids are made unique across files by prefixing with source_file:
    id = f"{source_file}__{raw_id}"
  and if multiple calls:
    id = f"{source_file}__{raw_id}__call{i}"

Usage:
  python scripts/build_bfcl_v4_internal_jsonl.py \
    --bfcl_dir path/to/berkeley-function-call-leaderboard/bfcl_eval \
    --output data/internal/bfcl_v4_all_internal.jsonl

Assumptions:
- possible_answer/<same filename> may be missing; we still emit rows with null ground_truth.
- Files are either JSON arrays OR JSONL (one JSON object per line). Both supported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# IO helpers
# -----------------------------
def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
      - JSON array: [ {...}, {...} ]
      - JSONL: one JSON object per line
    Returns list of dict rows.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try JSON first (array or object)
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            # Rare, but wrap a single object
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path} line {i}: {e}") from e
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# Normalization helpers
# -----------------------------
def get_id(row: Dict[str, Any]) -> str:
    rid = row.get("id") or row.get("example_id") or row.get("item_id") or row.get("qid")
    if rid is None:
        raise ValueError("Row missing an id field (id/example_id/item_id/qid).")
    return str(rid)


def safe_json_loads(s: str) -> Optional[Any]:
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def normalize_messages(qrow: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[List[Optional[int]]]]:
    """
    BFCL v4 variants commonly use:
      - messages: [...]
      - question: [...]
      - conversation: [...]
    Returns (messages, turn_index_map). turn_index_map maps turn index -> user message index.
    """
    raw = (
        qrow.get("messages")
        or qrow.get("question")
        or qrow.get("conversation")
        or qrow.get("chat")
        or qrow.get("dialogue")
    )

    def iter_message_dicts(node: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(node, dict):
            yield node
        elif isinstance(node, list):
            for item in node:
                yield from iter_message_dicts(item)

    def normalize_message_obj(m: Dict[str, Any]) -> Dict[str, Any]:
        role = m.get("role") or m.get("type") or m.get("speaker") or "user"
        content = m.get("content") or m.get("text") or m.get("value") or ""
        tool_name = None
        tool_arguments = None

        # Some BFCL variants include assistant tool_calls embedded
        if isinstance(m.get("tool_calls"), list) and m["tool_calls"]:
            first = m["tool_calls"][0]
            if isinstance(first, dict):
                tool_name = (
                    first.get("name")
                    or first.get("tool_name")
                    or (first.get("function") or {}).get("name")
                )
                args = first.get("arguments") or (first.get("function") or {}).get("arguments")
                if isinstance(args, str):
                    parsed = safe_json_loads(args)
                    tool_arguments = parsed if isinstance(parsed, dict) else {"__raw__": args}
                elif isinstance(args, dict):
                    tool_arguments = args

        return {
            "role": str(role),
            "content": "" if content is None else str(content),
            "tool_name": str(tool_name) if tool_name is not None else None,
            "tool_arguments": tool_arguments if isinstance(tool_arguments, dict) else None,
        }

    def flatten_messages(node: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in iter_message_dicts(node):
            out.append(normalize_message_obj(m))
        return out

    # Turn-structured input: question = [ [ {msg} ], [ {msg} ], ... ] or deeper
    if isinstance(raw, list) and any(isinstance(x, list) for x in raw):
        messages: List[Dict[str, Any]] = []
        turn_index_map: List[Optional[int]] = []
        for turn in raw:
            start_idx = len(messages)
            messages.extend(flatten_messages(turn))
            turn_user_idx = None
            for i in range(start_idx, len(messages)):
                if messages[i].get("role") == "user":
                    turn_user_idx = i
                    break
            turn_index_map.append(turn_user_idx)
        return messages, turn_index_map

    if raw is not None:
        messages = flatten_messages(raw)
        if messages:
            return messages, None

    # Fallback if only a string prompt exists
    prompt = qrow.get("prompt") or qrow.get("instruction") or qrow.get("input") or qrow.get("query")
    if isinstance(prompt, str) and prompt.strip():
        return (
            [{"role": "user", "content": prompt.strip(), "tool_name": None, "tool_arguments": None}],
            None,
        )

    return ([], None)


def normalize_tools(qrow: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    BFCL v4 variants commonly use:
      - function: [...]
      - functions: [...]
      - tools: [...]
    Each tool is normalized to:
      {"name","orig_description","aug_description":None,"parameters":{...}}
    """
    raw = qrow.get("tools") or qrow.get("functions") or qrow.get("function")

    # BFCL v4 multi_turn_base style: tools are implicit via `path` (fully-qualified function names)
    if (not raw) and isinstance(qrow.get("path"), list) and qrow["path"]:
        tools = []
        for fn in qrow["path"]:
            if not isinstance(fn, str):
                continue
            tools.append(
                {
                    "name": fn,  # e.g., "GorillaFileSystem.mv"
                    "orig_description": f"BFCL available function: {fn}",
                    "aug_description": None,
                    "parameters": {},
                }
            )
        return tools


    # BFCL v4 memory/web_search style: tools are implicit via involved_classes
    if (not raw) and isinstance(qrow.get("involved_classes"), list) and qrow["involved_classes"]:
        tools = []
        for cls in qrow["involved_classes"]:
            if not isinstance(cls, str):
                continue
            tools.append(
                {
                    "name": cls,
                    "orig_description": f"BFCL available class: {cls}",
                    "aug_description": None,
                    "parameters": {},
                }
            )
        return tools

    if not isinstance(raw, list):
        return []

    tools: List[Dict[str, Any]] = []
    for t in raw:
        if not isinstance(t, dict):
            continue

        # OpenAI style: {"type":"function","function":{...}}
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            name = fn.get("name")
            desc = fn.get("description") or ""
            params = fn.get("parameters") or {}
        else:
            # BFCL style: {"name","description","parameters"}
            name = t.get("name") or t.get("tool_name")
            desc = t.get("description") or t.get("doc") or t.get("summary") or ""
            params = t.get("parameters") or t.get("schema") or {}
            if not isinstance(params, dict):
                params = {}

        if not name:
            continue

        tools.append(
            {
                "name": str(name),
                "orig_description": "" if desc is None else str(desc),
                "aug_description": None,
                "parameters": params,
            }
        )

    return tools


# -----------------------------
# Ground truth extraction
# -----------------------------
def _tool_names_set(tools: List[Dict[str, Any]]) -> set:
    return {t.get("name") for t in tools if isinstance(t.get("name"), str)}

import ast

def parse_call_string(call: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Parses strings like: mv(source='final_report.pdf', destination='temp')
    Returns: ("mv", {"source": "...", "destination": "..."})
    """
    call = call.strip()
    if not call or "(" not in call or not call.endswith(")"):
        return (None, None)

    try:
        node = ast.parse(call, mode="eval").body
        if not isinstance(node, ast.Call):
            return (None, None)

        # function name (unqualified, e.g. mv)
        fn = None
        if isinstance(node.func, ast.Name):
            fn = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fn = node.func.attr

        args: Dict[str, Any] = {}
        positional = []
        for arg in node.args:
            try:
                positional.append(ast.literal_eval(arg))
            except Exception:
                positional.append(None)
        if positional:
            args["__args__"] = positional
        for kw in node.keywords:
            if kw.arg is None:
                continue
            val = ast.literal_eval(kw.value)
            args[kw.arg] = val

        return (fn, args)
    except Exception:
        return (None, {"__raw__": call})


def resolve_full_tool_name(short_name: str, candidate_full_names: set) -> Optional[str]:
    """
    Map short function name like 'mv' to a fully-qualified one in candidates, like 'GorillaFileSystem.mv'
    If multiple match, returns first sorted match (deterministic).
    """
    if not short_name:
        return None
    matches = sorted([c for c in candidate_full_names if c.endswith("." + short_name) or c == short_name])
    return matches[0] if matches else None

def extract_gt_calls(
    q_tools: List[Dict[str, Any]],
    arow: Dict[str, Any],
) -> List[Tuple[Optional[int], Optional[str], Optional[Dict[str, Any]]]]:
    """
    Returns a list of (turn_index, tool_name, tool_arguments) tuples.

    possible_answer format (what you described):
      {"id": "...", "ground_truth": ...}

    We support several common shapes for ground_truth:
      A) list of dicts, each dict is either:
         - {"some_tool_name": {...args...}}
         - {"name": "...", "arguments": {...}} or similar
      B) dict:
         - {"some_tool_name": {...args...}}
         - {"name": "...", "arguments": {...}}
      C) list of strings:
         - if string matches a candidate tool name, treat as tool_name
         - otherwise tool_name=None (still returns one call record)
      D) string:
         - if matches a tool, tool_name=that string else None
    """
    gt = arow.get("ground_truth")

    candidate_names = _tool_names_set(q_tools)
    source_info = arow.get("source")

    def final_answer_calls(answers: List[str]) -> List[Tuple[Optional[int], Optional[str], Optional[Dict[str, Any]]]]:
        calls: List[Tuple[Optional[int], Optional[str], Optional[Dict[str, Any]]]] = []
        for ans in answers:
            if not isinstance(ans, str):
                continue
            args: Dict[str, Any] = {"answer": ans}
            if source_info is not None:
                args["source"] = source_info
            calls.append((None, "final_answer", args))
        return calls if calls else [(None, "final_answer", {"answer": None, "source": source_info})]

    # Multi-turn base style:
    # ground_truth = [ [ "cd(...)", "mkdir(...)" ], [ "grep(...)" ], ... ]
    if isinstance(gt, list) and gt and all(isinstance(step, list) for step in gt):
        calls = []
        for step_i, step in enumerate(gt):
            for call_str in step:
                if isinstance(call_str, str):
                    short_fn, args = parse_call_string(call_str)
                    full_name = resolve_full_tool_name(short_fn, candidate_names) if short_fn else None
                    calls.append((step_i, full_name or short_fn, args))
                else:
                    calls.append((step_i, None, None))
        return calls if calls else [(None, None, None)]


    def normalize_args(x: Any) -> Optional[Dict[str, Any]]:
        if x is None:
            return None
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            parsed = safe_json_loads(x)
            if isinstance(parsed, dict):
                return parsed
            return {"__raw__": x}
        return None

    def from_toolcall_obj(obj: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[Dict[str, Any]]]:
        t_idx = obj.get("turn_index") if isinstance(obj.get("turn_index"), int) else None
        name = obj.get("tool_name") or obj.get("name") or (obj.get("function") or {}).get("name")
        args = obj.get("tool_arguments") or obj.get("arguments") or obj.get("args") or (obj.get("function") or {}).get("arguments")
        return (t_idx, str(name) if name is not None else None, normalize_args(args))

    # A) list
    if isinstance(gt, list):
        if gt and all(isinstance(x, str) for x in gt) and not any(
            isinstance(x, str) and (x in candidate_names) for x in gt
        ):
            return final_answer_calls([x for x in gt if isinstance(x, str)])

        calls: List[Tuple[Optional[int], Optional[str], Optional[Dict[str, Any]]]] = []
        for entry in gt:
            # list of dict tool calls
            if isinstance(entry, dict):
                # Pattern: {"toolname": {...}}
                keys = [k for k in entry.keys() if k != "turn_index"]
                if len(keys) == 1 and isinstance(keys[0], str):
                    maybe_tool = keys[0]
                    maybe_args = entry.get(maybe_tool)
                    # Only treat as tool call mapping if it looks like a candidate tool
                    if maybe_tool in candidate_names:
                        t_idx = entry.get("turn_index") if isinstance(entry.get("turn_index"), int) else None
                        calls.append((t_idx, maybe_tool, normalize_args(maybe_args)))
                        continue
                # Otherwise interpret as a tool-call object
                calls.append(from_toolcall_obj(entry))
                continue

            # list of strings (sometimes "answer", sometimes tool name)
            if isinstance(entry, str):
                tool_name = entry if entry in candidate_names else resolve_full_tool_name(entry, candidate_names)
                calls.append((None, tool_name, None))
                continue

            # other
            calls.append((None, None, None))

        return calls if calls else [(None, None, None)]

    # B) dict
    if isinstance(gt, dict):
        # Mapping {"toolname": {...}}
        keys = list(gt.keys())
        if len(keys) == 1 and isinstance(keys[0], str) and keys[0] in candidate_names:
            return [(None, keys[0], normalize_args(gt[keys[0]]))]
        # Tool-call object dict
        return [from_toolcall_obj(gt)]

    # C/D) string or other scalar
    if isinstance(gt, str):
        if gt in candidate_names:
            return [(None, gt, None)]
        resolved = resolve_full_tool_name(gt, candidate_names)
        if resolved:
            return [(None, resolved, None)]
        return final_answer_calls([gt])

    return [(None, None, None)]


def infer_turn_index(
    messages,
    gt_tool_name,
    provided_turn_index,
    task_type,
    turn_index_map: Optional[List[Optional[int]]] = None,
):
    # If already an absolute message index, keep it
    if isinstance(provided_turn_index, int) and task_type == "single_turn":
        return provided_turn_index

    # For multi_turn_base: provided_turn_index is step_i (0..n_steps-1),
    # map it to the index of the step_i-th user message in the flattened messages list.
    if task_type == "multi_turn" and isinstance(provided_turn_index, int):
        if turn_index_map is not None:
            if 0 <= provided_turn_index < len(turn_index_map):
                return turn_index_map[provided_turn_index]
            return None
        user_idxs = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if 0 <= provided_turn_index < len(user_idxs):
            return user_idxs[provided_turn_index]
        return None

    return None



# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bfcl_dir",
        type=str,
        required=True,
        help="Folder containing BFCL_v4_*.json files and a possible_answer/ subfolder.",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path (combined internal dataset).",
    )
    args = ap.parse_args()

    bfcl_dir = Path(args.bfcl_dir)
    answers_dir = bfcl_dir / "possible_answer"
    out_path = Path(args.output)

    if not bfcl_dir.exists():
        raise FileNotFoundError(f"bfcl_dir not found: {bfcl_dir}")
    if not answers_dir.exists():
        # Allow processing without possible_answer; we will emit null ground_truth.
        answers_dir = None

    question_files = sorted(bfcl_dir.glob("BFCL_v4_*.json"))
    if not question_files:
        raise FileNotFoundError(f"No BFCL_v4_*.json found in: {bfcl_dir}")

    missing_answer_files: List[str] = []
    missing_answer_ids: List[Tuple[str, str]] = []  # (filename, missing_id)
    skipped_non_tool_files: List[str] = []
    out_rows: List[Dict[str, Any]] = []
    total_q = 0
    total_out = 0

    for qfile in question_files:
        source_file = qfile.stem
        if source_file.startswith("BFCL_v4_"):
            source_label = source_file[len("BFCL_v4_") :]
        else:
            source_label = source_file  # fallback, should not happen

        task_type = "multi_turn" if "multi_turn" in source_label else "single_turn"

        afile = (answers_dir / qfile.name) if answers_dir is not None else None
        qrows = load_json_or_jsonl(qfile)
        arows = load_json_or_jsonl(afile) if afile is not None and afile.exists() else []
        if afile is None or not afile.exists():
            missing_answer_files.append(qfile.name)
        # Skip files that are not tool-calling (no tool list schema in the questions file)
        if not qrows:
            skipped_non_tool_files.append(qfile.name)
            continue

        sample = qrows[0]
        has_tool_schema = any(k in sample for k in ("tools", "functions", "function"))
        has_path_schema = isinstance(sample.get("path"), list) and len(sample.get("path")) > 0
        has_involved_schema = isinstance(sample.get("involved_classes"), list) and len(sample.get("involved_classes")) > 0

        if not (has_tool_schema or has_path_schema or has_involved_schema):
            skipped_non_tool_files.append(qfile.name)
            continue



        aidx: Dict[str, Dict[str, Any]] = {}
        for ar in arows:
            aidx[get_id(ar)] = ar

        for qrow in qrows:
            total_q += 1
            raw_id = get_id(qrow)
            if raw_id not in aidx:
                # Still emit a row with null ground_truth so tools are catalogued.
                missing_answer_ids.append((afile.name, raw_id))
                arow = {"ground_truth": None}
            else:
                arow = aidx[raw_id]

            messages, turn_index_map = normalize_messages(qrow)
            tools = normalize_tools(qrow)

            gt_calls = extract_gt_calls(tools, arow)

            # always prefix ids with source_label to avoid collisions across files
            base_out_id = f"{source_label}__{raw_id}"

            # One row per tool-call decision
            if len(gt_calls) > 1:
                for i, (t_idx, gt_tool, gt_args) in enumerate(gt_calls):
                    out_id = f"{base_out_id}__call{i}"
                    out_rows.append(
                        {
                            "id": out_id,
                            "source_file": source_label,
                            "task_type": task_type,
                            "messages": messages,
                            "tools": tools,
                            "ground_truth": {
                                "turn_index": infer_turn_index(messages, gt_tool, t_idx, task_type, turn_index_map),
                                "tool_name": gt_tool,
                                "tool_arguments": gt_args,
                            },
                        }
                    )
                    total_out += 1
            else:
                (t_idx, gt_tool, gt_args) = gt_calls[0]
                out_rows.append(
                    {
                        "id": base_out_id,
                        "source_file": source_label,
                        "task_type": task_type,
                        "messages": messages,
                        "tools": tools,
                        "ground_truth": {
                        "turn_index": infer_turn_index(messages, gt_tool, t_idx, task_type, turn_index_map),
                            "tool_name": gt_tool,
                            "tool_arguments": gt_args,
                        },
                    }
                )
                total_out += 1

    write_jsonl(out_path, out_rows)

    print("Done")
    print(f"BFCL dir: {bfcl_dir}")
    print(f"Question files processed: {len(question_files)}")
    print(f"Question examples read: {total_q}")
    print(f"Internal rows written (tool-call decisions): {total_out}")
    print(f"Output: {out_path}")
    if missing_answer_files:
        print("\nFiles without possible_answer (ground_truth set to None):")
        for fn in missing_answer_files:
            print(f"  - {fn}")

    if missing_answer_ids:
        print("\nExamples with missing possible_answer id (ground_truth set to None):")
        # show only first 50 to avoid flooding terminal
        for fn, mid in missing_answer_ids[:50]:
            print(f"  - file={fn} missing_id={mid}")
        if len(missing_answer_ids) > 50:
            print(f"  ... and {len(missing_answer_ids) - 50} more")
    if skipped_non_tool_files:
        print("\nSkipped non-tool BFCL files (no tools/functions/function field):")
        for fn in skipped_non_tool_files:
            print(f"  - {fn}")


if __name__ == "__main__":
    main()

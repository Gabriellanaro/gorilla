"""
Augment BFCL v4 tool catalogue using Azure OpenAI (Responses API).

This script reads one tool per line (JSONL) and writes an augmented description
for each tool to a new JSONL file. It keeps the Azure OpenAI call logic and
system prompt loading, but removes API Bench-specific assumptions.

Input:  berkeley-function-call-leaderboard/bfcl_eval/data/internal/bfcl_v4_tool_catalogue.jsonl
Output: berkeley-function-call-leaderboard/bfcl_eval/data/internal/bfcl_v4_tool_catalogue_augmented.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tqdm import tqdm


# ---------------------- Environment ---------------------- #

load_dotenv(override=True)

REQUIRED_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
]

missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
if missing:
    raise RuntimeError(f"Missing required Azure env vars: {missing}")


# ---------------------- IO Helpers ---------------------- #

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path} line {i}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Legacy API Bench helpers (not used for BFCL v4 augmentation):
# def normalize_args(raw_args: Any) -> List[str]: ...
# def compute_tool_id(api_name: str, args: List[str]) -> str: ...
# def get_api_data(row: Dict[str, Any]) -> Dict[str, Any]: ...


def is_rewrite_issue(desc: str, max_desc_chars: int, allow_backticks: bool) -> Optional[str]:
    """
    Returns:
      None         -> acceptable
      "empty"      -> empty output
      "backticks"  -> contains backticks when not allowed
      "too_long"   -> longer than max_desc_chars
    """
    if not desc:
        return "empty"
    if (not allow_backticks) and ("`" in desc):
        return "backticks"
    if len(desc) > max_desc_chars:
        return "too_long"
    return None


# ---------------------- Data Structures ---------------------- #
# Legacy ToolDef class not needed for BFCL v4 catalogue processing.


# ---------------------- Azure OpenAI Client ---------------------- #

def build_client() -> AzureOpenAI:
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


# ---------------------- TOP-K Lexical Retrieval ---------------------- #
# Legacy retrieval utilities not used for BFCL v4 catalogue processing.


# ---------------------- Rewriter ---------------------- #

def render_user_content(tool_name: str, description: str) -> str:
    desc = description.strip() or "No description provided."
    return f"Tool name: {tool_name}\nOriginal description: {desc}"


def extract_output_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    data = resp.model_dump()
    for item in data.get("output", []) or []:
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text" and part.get("text"):
                return part["text"].strip()

    raise RuntimeError("Could not extract text output from response.")


def _json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    chunk = text[start : end + 1]
    try:
        obj = json.loads(chunk)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def validate_augmented_description(parsed: Dict[str, Any]) -> str:
    aug_desc = str(parsed.get("augmented_description") or "").strip()
    if len(aug_desc) > 7000:
        aug_desc = aug_desc[:7000].rstrip()
    return aug_desc


def rewrite_tool_once(
    client: AzureOpenAI,
    model: str,
    system_prompt: str,
    tool_name: str,
    description: str,
    max_retries: int = 3,
    max_desc_chars: int = 1200,
    allow_backticks: bool = False,
) -> Tuple[Optional[str], str, str]:
    """
    Returns (augmented_description | None, raw_model_output, parse_mode).
    """
    user_content = render_user_content(tool_name, description)

    last_text = ""
    last_reject_reason: Optional[str] = None
    last_valid_aug_desc: Optional[str] = None
    parse_mode = "unknown"

    for attempt in range(max_retries + 1):
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        last_text = extract_output_text(resp)

        parsed = _json_from_text(last_text)
        if parsed is None or "augmented_description" not in parsed:
            last_reject_reason = "non_json"
            if attempt < max_retries:
                continue
            break

        aug_desc = validate_augmented_description(parsed)
        last_valid_aug_desc = aug_desc

        reject_reason = is_rewrite_issue(
            aug_desc,
            max_desc_chars=max_desc_chars,
            allow_backticks=allow_backticks,
        )

        if reject_reason is None:
            parse_mode = "json_ok"
            return aug_desc, last_text, parse_mode

        last_reject_reason = reject_reason
        if attempt < max_retries:
            continue
        break

    if last_reject_reason == "too_long" and last_valid_aug_desc:
        truncated = last_valid_aug_desc[:max_desc_chars].rstrip()
        parse_mode = "json_truncated"
        return truncated, last_text, parse_mode

    parse_mode = "parse_failed"
    return None, last_text, parse_mode


# ---------------------- Dataset Parsing ---------------------- #
# API Bench parsing helpers removed for BFCL v4 tool catalogue.


# ---------------------- Pipeline ---------------------- #

def augment_dataset(
    input_path: Path,
    output_path: Path,
    model: str,
    prompt_path: Path,
    limit: Optional[int],
    max_retries: int,
    max_desc_chars: int,
    allow_backticks: bool,
    skip_existing: bool,
    resume_from: Optional[str],
    dry_run: bool,
):
    rows = load_jsonl(input_path)
    if limit:
        rows = rows[:limit]

    system_prompt = load_prompt(prompt_path)
    prompt_hash = sha256_text(system_prompt)

    client = None if dry_run else build_client()

    existing_tools: Dict[str, Dict[str, Any]] = {}
    if skip_existing and output_path.exists():
        for row in load_jsonl(output_path):
            name = row.get("tool_name")
            if isinstance(name, str) and name.strip():
                existing_tools[name.strip()] = row

    if dry_run:
        print(f"[dry-run] input rows: {len(rows)}")
        print(f"[dry-run] existing augmented rows: {len(existing_tools)}")

    augmented_rows: List[Dict[str, Any]] = []
    resume_hit = resume_from is None
    for row in tqdm(rows, desc="Rewriting tool docs", unit="tool"):
        updated = dict(row)
        tool_name = str(updated.get("tool_name") or "").strip()
        if not tool_name:
            print("[warn] missing tool_name; skipping tool.")
            continue

        if not resume_hit:
            if tool_name == resume_from:
                resume_hit = True
            else:
                continue

        if skip_existing and tool_name in existing_tools:
            augmented_rows.append(existing_tools[tool_name])
            continue

        orig_descs = updated.get("orig_descriptions") or []
        first_desc = ""
        if isinstance(orig_descs, list) and orig_descs:
            first = orig_descs[0]
            if isinstance(first, dict):
                first_desc = str(first.get("description") or "")

        if dry_run:
            updated["aug_description"] = first_desc.strip() or "No description provided."
            augmented_rows.append(updated)
            continue

        aug_desc, raw_out, parse_mode = rewrite_tool_once(
            client=client,
            model=model,
            system_prompt=system_prompt,
            tool_name=tool_name,
            description=first_desc,
            max_retries=max_retries,
            max_desc_chars=max_desc_chars,
            allow_backticks=allow_backticks,
        )

        if not aug_desc:
            print(f"[warn] JSON parse failed for tool={tool_name}; using original description.")
            aug_desc = first_desc.strip() or "No description provided."

        updated["aug_description"] = aug_desc
        updated["_rewrite_meta"] = {
            "rewriter_model": model,
            "rewritten_at_utc": utc_now_iso(),
            "prompt_hash": prompt_hash,
            "parse_mode": parse_mode,
            "raw_output_hash": sha256_text(raw_out or ""),
        }
        augmented_rows.append(updated)

    dump_jsonl(output_path, augmented_rows)
    print(f"[done] wrote augmented dataset to {output_path}")


# ---------------------- CLI ---------------------- #

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Rewrite tool docs for BFCL v4 tool catalogue.")
    parser.add_argument(
        "--input-json",
        default="bfcl_eval/data/internal/bfcl_v4_tool_catalogue.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="bfcl_eval/data/internal/bfcl_v4_tool_catalogue_augmented.jsonl",
    )
    parser.add_argument("--model", default="gpt-5.1-responses")
    parser.add_argument(
        "--prompt-file",
        default="bfcl_eval\data\prompts\docstring_rewriter.txt",
    )
    parser.add_argument("--max-desc-chars", type=int, default=1200)
    parser.add_argument("--allow-backticks", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tools already present in the output JSONL (by tool_name).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Start processing at the first matching tool_name and skip prior entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the model; just validate input and write placeholders.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries when JSON parsing fails or the rewrite is rejected (too long / backticks / empty).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    augment_dataset(
        input_path=Path(args.input_json),
        output_path=Path(args.output_json),
        model=args.model,
        prompt_path=Path(args.prompt_file),
        max_desc_chars=args.max_desc_chars,
        allow_backticks=args.allow_backticks,
        limit=args.limit,
        max_retries=args.max_retries,
        skip_existing=args.skip_existing,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

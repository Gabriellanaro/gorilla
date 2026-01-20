import argparse
import json
from pathlib import Path

from bfcl_eval.model_handler.utils import coerce_fc_response, convert_to_function_call


def _find_first_failed_response(obj: dict):
    inference_log = obj.get("inference_log", [])
    result = obj.get("result", [])

    turn_map = {}
    result_turn_idx = -1
    for log_idx, turn in enumerate(inference_log):
        if isinstance(turn, dict) and "begin_of_turn_query" in turn:
            result_turn_idx += 1
            turn_map[log_idx] = result_turn_idx

    for log_idx, turn in enumerate(inference_log):
        if not isinstance(turn, dict):
            continue
        for step_key, step_entries in turn.items():
            if not isinstance(step_entries, list):
                continue
            for entry in step_entries:
                if entry.get("role") == "handler_log" and "error" in entry:
                    step_num = int(step_key.split("_")[-1])
                    mapped_turn = turn_map.get(log_idx)
                    if mapped_turn is None:
                        return None
                    if mapped_turn >= len(result):
                        return None
                    if step_num >= len(result[mapped_turn]):
                        return None
                    return result[mapped_turn][step_num]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=(
            "berkeley-function-call-leaderboard/result_desc_augmented_name_augmented/"
            "azure-gpt-5.1-responses-FC/multi_turn/BFCL_v4_multi_turn_base_result.json"
        ),
        help="Path to a BFCL result JSONL file.",
    )
    parser.add_argument("--id", default=None, help="Entry id to inspect.")
    parser.add_argument(
        "--response-format",
        default="responses",
        choices=["auto", "chatcompletions", "responses"],
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if args.id:
        target = next((e for e in entries if e.get("id") == args.id), None)
        if target is None:
            raise SystemExit(f"Missing id: {args.id}")
    else:
        target = next((e for e in entries if e.get("inference_log")), None)
        if target is None:
            raise SystemExit("No entries with inference_log found.")

    model_responses = _find_first_failed_response(target)
    if model_responses is None:
        raise SystemExit("No failed response found in inference_log.")

    coerced = coerce_fc_response(model_responses, args.response_format)
    if coerced is not None:
        if coerced == []:
            print("OK: response decoded as empty tool call list.")
            return 0
        convert_to_function_call(coerced)
        print("OK: response decoded to tool call list.")
        return 0

    convert_to_function_call(model_responses)
    print("OK: response decoded to tool call list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

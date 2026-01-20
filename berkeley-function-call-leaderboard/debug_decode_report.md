# Debug Decode Report

## Where the decode failure is emitted

- Print location: `berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py:298` (`BaseHandler.inference_multi_turn_FC`).
- Same message in prompting loop: `berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py:590` (`BaseHandler.inference_multi_turn_prompting`).
- Exception origin: `berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py:207-226` (`convert_to_function_call`, `value.items()` on a string).

## Exception details

- Exception: `AttributeError: 'str' object has no attribute 'items'`.
- Variable being decoded: `model_responses` (type `str`).
- Source of `model_responses`: `AzureOpenAIResponsesHandler._parse_query_response_FC` assigns `api_response.output_text` when no function calls are detected; see `berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/azure_openai.py:136-155`.

## Raw saved payloads (3 failing items)

Raw payloads are not persisted as full API objects; BFCL only saves the parsed `result` field and the `inference_log` in the JSONL result files. The failing payload is the assistant `content` string saved in the `result` list at the same turn/step that logged the decode error.

File: `berkeley-function-call-leaderboard/result_desc_augmented_name_augmented/azure-gpt-5.1-responses-FC/multi_turn/BFCL_v4_multi_turn_base_result.json`

1) `multi_turn_base_164` (turn 0, step 2)
```
result[0][2] -> type: str
"I checked from Rivermist's nearest airport (RMS) to New York (NYC) on 2026-12-01 ... no bookable RMS -> NYC flight ..."
```

2) `multi_turn_base_165` (turn 0, step 1)
```
result[0][1] -> type: str
"Eleanor Smith's traveler information has been successfully verified ... Verification status: Verified (valid)."
```

3) `multi_turn_base_166` (turn 0, step 3)
```
result[0][3] -> type: str
"Here's what I found for your trip: Route: Crescent Hollow (CRH) -> Rivermist (RMS) ..."
```

These entries are valid JSON (JSONL line objects). They are not truncated; each string is a complete assistant response and is stored as a JSON string.

## Expected vs actual schema

Expected (for FC decoding):
```
list[dict[str, dict]]  or  list[dict[str, json_str]]
Example: [{"get_flight_cost": {"travel_from": "RMS", "travel_to": "JFK", "travel_date": "2026-12-01", "travel_class": "first"}}]
```

Actual (observed in failing items):
```
str
Example: "Here's what I found for your trip: ..."
```

## Root cause (one sentence)

In FC multi-turn mode, `convert_to_function_call` is invoked on a plain text `output_text` string from the Responses API, and it attempts `value.items()` as if the payload were a dict of tool arguments, causing an AttributeError and the repeated "Failed to decode" messages.

## Call chain (single failing item trace)

1) Generation loop: `berkeley-function-call-leaderboard/bfcl_eval/_llm_response_generation.py:195-254` -> `multi_threaded_inference()` -> `handler.inference()`.
2) Multi-turn FC loop: `berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py:101-350` -> `BaseHandler.inference_multi_turn_FC()`.
3) Model call: `berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/azure_openai.py:91-111` -> `_query_FC()` -> `AzureOpenAI.responses.create(...)`.
4) Response parsing: `berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/azure_openai.py:136-155` -> `_parse_query_response_FC()` -> `model_responses = api_response.output_text` when no tool calls.
5) Decode attempt: `berkeley-function-call-leaderboard/bfcl_eval/model_handler/base_handler.py:273-299` -> `decode_execute()` -> `convert_to_function_call()` -> error at `utils.py:226`.

## Recommended fixes

Option A (minimal patch; decode Responses-format text safely):
- Update `berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py`:
  - Add `coerce_fc_response(...)` to detect plain text outputs or JSON-like tool calls and return `[]` for text.
  - Log possible truncation when JSON-like strings fail to parse (length and tail).
- Update `berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/azure_openai.py` and `berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/openai_response.py`:
  - In `decode_execute`, call `coerce_fc_response` before `convert_to_function_call`.
- Add CLI flag:
  - `berkeley-function-call-leaderboard/bfcl_eval/__main__.py` + `berkeley-function-call-leaderboard/bfcl_eval/_llm_response_generation.py`:
    - `--response-format {auto,chatcompletions,responses}` and store in run metadata.
- Add script test:
  - `berkeley-function-call-leaderboard/bfcl_eval/scripts/test_decode_responses_payload.py` loads a failing payload and asserts decode succeeds.

Option B (force schema that BFCL already expects):
- Change the model call to always return structured tool calls:
  - In `berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/azure_openai.py:_query_FC`, pass tool-choice parameters so the model must produce a tool call (if supported by the model).
  - Alternatively, switch the model to `AzureOpenAICompletionsHandler` in `berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py` for this model alias, so `tool_calls` are always parsed from ChatCompletions.


# Experiment CLI Reference

All commands below are intended to run from the `berkeley-function-call-leaderboard` repo root unless noted.

## bfcl CLI (preferred)

Install entrypoint (if needed):
```bash
pip install -e .
```

### `bfcl version`
Show the installed bfcl version.

### `bfcl test-categories`
List available test categories.

### `bfcl models`
List available models.

### `bfcl generate`
Generate model responses.

Arguments:
- `--model`: comma-separated list of model names (default: `gorilla-openfunctions-v2`)
- `--test-category`: comma-separated list of categories (default: `all`)
- `--temperature`
- `--include-input-log`
- `--exclude-state-log`
- `--num-gpus`
- `--num-threads`
- `--gpu-memory-utilization`
- `--backend` (default: `sglang`)
- `--skip-server-setup`
- `--local-model-path`
- `--result-dir`
- `--allow-overwrite`, `-o`
- `--run-ids`
- `--tool-desc` (default: `original`)
- `--tool-name` (default: `original`)
- `--aug-tool-catalog`
- `--response-format` (default: `auto`)

Examples:
```bash
bfcl generate --model gpt-4o-2024-11-20-FC --test-category simple_python,parallel
bfcl generate --model claude-3-5-sonnet-20241022-FC,gpt-4o-2024-11-20-FC --test-category multi_turn
bfcl generate --model gorilla-openfunctions-v2 --run-ids
bfcl generate --model llama-3.1-8b --backend vllm --num-gpus 1 --gpu-memory-utilization 0.9
bfcl generate --model local-oss-model --local-model-path C:\models\my_model --skip-server-setup
bfcl generate --model gpt-4o-2024-11-20-FC --tool-desc augmented --tool-name augmented --aug-tool-catalog bfcl_eval/data/internal/bfcl_v4_tool_catalogue_augmented.jsonl
```

### `bfcl results`
List available result folders.

Arguments:
- `--result-dir`

Example:
```bash
bfcl results --result-dir result_desc_augmented_name_augmented
```

### `bfcl evaluate`
Evaluate model results.

Arguments:
- `--model`: comma-separated list of model names
- `--test-category`: comma-separated list of categories (default: `all`)
- `--result-dir`
- `--score-dir`
- `--partial-eval`
- `--tool-desc` (default: `original`)
- `--tool-name` (default: `original`)

Examples:
```bash
bfcl evaluate --model gpt-4o-2024-11-20-FC --test-category simple_python,parallel
bfcl evaluate --model gpt-4o-2024-11-20-FC --test-category multi_turn --partial-eval
bfcl evaluate --model gpt-4o-2024-11-20-FC --tool-desc augmented --tool-name augmented
```

### `bfcl scores`
Display leaderboard table from a score directory.

Arguments:
- `--score-dir`

Example:
```bash
bfcl scores --score-dir score_desc_augmented_name_augmented
```

## Legacy generate/evaluate entrypoints

These mirror the `bfcl` CLI but use space-separated lists instead of comma-separated lists.

### `python -m bfcl_eval.openfunctions_evaluation`
Generate model responses (legacy).

Arguments:
- `--model` (default: `gorilla-openfunctions-v2`)
- `--test-category` (default: `all`)
- `--temperature`
- `--include-input-log`
- `--exclude-state-log`
- `--num-threads`
- `--num-gpus`
- `--backend` (`vllm` or `sglang`)
- `--gpu-memory-utilization`
- `--result-dir`
- `--run-ids`
- `--tool-desc` (`original` or `augmented`)
- `--tool-name` (`original` or `augmented`)
- `--response-format` (`auto`, `chatcompletions`, `responses`)
- `--allow-overwrite`, `-o`
- `--skip-server-setup`
- `--local-model-path`

Examples:
```bash
python -m bfcl_eval.openfunctions_evaluation --model gpt-4o-2024-11-20-FC --test-category simple_python parallel
python -m bfcl_eval.openfunctions_evaluation --model gpt-4o-2024-11-20-FC --run-ids
```

### `python -m bfcl_eval.eval_checker.eval_runner`
Evaluate model results (legacy).

Arguments:
- `--model` (space-separated list)
- `--test-category` (space-separated list, default: `all`)
- `--result-dir`
- `--score-dir`
- `--partial-eval`

Examples:
```bash
python -m bfcl_eval.eval_checker.eval_runner --model gpt-4o-2024-11-20-FC --test-category simple_python parallel
python -m bfcl_eval.eval_checker.eval_runner --model gpt-4o-2024-11-20-FC --test-category multi_turn --partial-eval
```

### `python openfunctions_evaluation.py`
Wrapper for legacy generation (same args as `-m bfcl_eval.openfunctions_evaluation`).

Example:
```bash
python openfunctions_evaluation.py --model gpt-4o-2024-11-20-FC --test-category simple_python
```

## Leaderboard and score comparison

### `python leaderboard_compare.py`
Compare score roots and generate leaderboard comparison artifacts.

Arguments:
- `--orig` (default: `score_desc_original_name_original`)
- `--aug` (default: `score_desc_augmented_name_augmented`)
- `--out` (default: `comparison_out/leaderboard`)

Examples:
```bash
python leaderboard_compare.py --orig score_desc_original_name_original --aug score_desc_augmented_name_augmented --out comparison_out/leaderboard
```

### `python compare_scores.py` (repo root)
Print per-category deltas between two score roots for a single model.

Arguments:
- `--model` (required)
- `--category` (optional list)
- `--score-dir-original` (default: `score`)
- `--score-dir-augmented` (default: `score_augmented`)

Examples:
```bash
python compare_scores.py --model azure-gpt-5.1-responses-FC
python compare_scores.py --model azure-gpt-5.1-responses-FC --category simple_python multi_turn
python compare_scores.py --model azure-gpt-5.1-responses-FC --score-dir-original score_desc_original_name_original --score-dir-augmented score_desc_augmented_name_augmented
```

### `python bfcl_eval/scripts/compare_scores.py`
Build comparison CSVs across all models and categories.

Arguments:
- `--scores_orig` (default: `score_desc_original_name_original`)
- `--scores_aug` (default: `score_desc_augmented_name_augmented`)
- `--out_dir` (default: `comparison_out`)

Example:
```bash
python bfcl_eval/scripts/compare_scores.py --scores_orig score_desc_original_name_original --scores_aug score_desc_augmented_name_augmented --out_dir comparison_out
```

## Sanity checks and debugging

### `python sanity_check_multiturn.py`
Analyze multi-turn decode failures and write summary artifacts.

Arguments:
- `--results_root` (required)
- `--scores_root` (required)
- `--model` (required)
- `--category` (default: `multi_turn`)
- `--out_dir` (required)

Example:
```bash
python sanity_check_multiturn.py --results_root result_desc_augmented_name_augmented --scores_root score_desc_augmented_name_augmented --model azure-gpt-5.1-responses-FC --out_dir comparison_out/sanity
```

### `python bfcl_eval/scripts/test_decode_responses_payload.py`
Inspect a failing response payload for decoding.

Arguments:
- `--path` (default points to a BFCL result JSONL file)
- `--id` (optional entry id)
- `--response-format` (`auto`, `chatcompletions`, `responses`)

Examples:
```bash
python bfcl_eval/scripts/test_decode_responses_payload.py --path result_desc_augmented_name_augmented/azure-gpt-5.1-responses-FC/multi_turn/BFCL_v4_multi_turn_base_result.json
python bfcl_eval/scripts/test_decode_responses_payload.py --path result_desc_augmented_name_augmented/azure-gpt-5.1-responses-FC/multi_turn/BFCL_v4_multi_turn_base_result.json --id multi_turn_base_15 --response-format responses
```

### `python bfcl_eval/scripts/literal_parse_sanity.py`
Sanity check for literal parsing (no args).

Example:
```bash
python bfcl_eval/scripts/literal_parse_sanity.py
```

### `python bfcl_eval/scripts/ddg_sanity_test.py`
Sanity check for web search integration (requires SerpAPI in `.env`).

Example:
```bash
python bfcl_eval/scripts/ddg_sanity_test.py
```

## Dataset/tooling utilities

### `python bfcl_eval/scripts/build_bfcl_v4_internal_jsonl.py`
Build combined internal JSONL from BFCL v4 datasets.

Arguments:
- `--bfcl_dir` (required)
- `--output` (required)

Example:
```bash
python bfcl_eval/scripts/build_bfcl_v4_internal_jsonl.py --bfcl_dir bfcl_eval --output bfcl_eval/data/internal/bfcl_v4_all_internal.jsonl
```

### `python bfcl_eval/scripts/build_bfcl_v4_tool_catalogue.py`
Build tool catalogue JSONL from the internal JSONL (no args).

Example:
```bash
python bfcl_eval/scripts/build_bfcl_v4_tool_catalogue.py
```

### `python bfcl_eval/scripts/augment_docstrings_bfcl_v4.py`
Rewrite tool descriptions using Azure OpenAI.

Arguments:
- `--input-json`
- `--output-json`
- `--model`
- `--prompt-file`
- `--max-desc-chars`
- `--allow-backticks`
- `--limit`
- `--skip-existing`
- `--resume-from`
- `--dry-run`
- `--max-retries`

Example:
```bash
python bfcl_eval/scripts/augment_docstrings_bfcl_v4.py --input-json bfcl_eval/data/internal/bfcl_v4_tool_catalogue.jsonl --output-json bfcl_eval/data/internal/bfcl_v4_tool_catalogue_augmented.jsonl --model gpt-4o --prompt-file bfcl_eval/data/prompts/docstring_rewriter_V1.txt --max-desc-chars 1200 --skip-existing
```

### `python bfcl_eval/scripts/analyze_internal_bfcl_jsonl.py`
Analyze combined internal JSONL and write reports.

Arguments:
- `--input` (required)
- `--outdir` (required)

Example:
```bash
python bfcl_eval/scripts/analyze_internal_bfcl_jsonl.py --input bfcl_eval/data/internal/bfcl_v4_all_internal.jsonl --outdir bfcl_eval/data/internal/_analysis_bfcl_v4
```

### `python bfcl_eval/scripts/compile_multi_turn_func_doc.py`
Compile multi-turn function docs (no args).

Example:
```bash
python bfcl_eval/scripts/compile_multi_turn_func_doc.py
```

### `python bfcl_eval/scripts/visualize_multi_turn_ground_truth_conversation.py`
Generate ground-truth multi-turn conversation logs (no args).

Example:
```bash
python bfcl_eval/scripts/visualize_multi_turn_ground_truth_conversation.py
```

### `python bfcl_eval/scripts/check_func_doc_format.py`
Validate function schema format (no args).

Example:
```bash
python bfcl_eval/scripts/check_func_doc_format.py
```

### `python bfcl_eval/scripts/check_illegal_python_param_name.py`
Scan and fix illegal Python param names in datasets (writes changes).

Example:
```bash
python bfcl_eval/scripts/check_illegal_python_param_name.py
```

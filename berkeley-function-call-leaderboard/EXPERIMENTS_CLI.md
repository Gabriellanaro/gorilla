# Experiment CLI Reference

All commands below are intended to run from the `berkeley-function-call-leaderboard` repo root unless noted.

## bfcl CLI (primary)

Install entrypoint (if needed):
```bash
pip install -e .
```

### `bfcl generate`
Generate model responses.

Example:
```bash
bfcl generate --model azure-gpt-4o-FC --test-category simple_python,parallel --tool-desc original --tool-name augmented
```

### `bfcl evaluate`
Evaluate model results.

Example:
```bash
bfcl evaluate --test-category all --tool-desc augmented --tool-name original --model azure-gpt-5.1-responses-FC
```

## Scripts and notebooks (curated)

### `python analyze_errors.py`
Parses BFCL error JSON/JSONL files across score folders, builds paired outcomes, and writes parquet/CSV plus summary tables under `analysis_out`.

Example:
```bash
python analyze_errors.py --root . --outdir analysis_out
```

### `python catalogue_descriptiveness.py`
Computes descriptiveness/readability metrics for tool descriptions (aug vs orig) and writes `tool_level_metrics.csv` plus aggregate CSVs.

Example:
```bash
python catalogue_descriptiveness.py --input bfcl_eval/data/internal/bfcl_v4_tool_catalogue_augmented.jsonl --output_dir descriptiveness_out
```

### `python compare_scores.py`
Compares multiple score roots; outputs comparison CSVs, plots, and a README under `comparison_out`.

Example:
```bash
python compare_scores.py --scores_orig score_desc_original_name_original --scores_aug score_desc_augmented_name_augmented --out_dir comparison_out
```

### `python make_leaderboard.py`
Builds `leaderboard.html` from score CSVs and computes delta columns.

Example:
```bash
python make_leaderboard.py
```

### `error_analysis.ipynb`
Notebook for loading `analyze_errors.py` outputs and generating tables/plots.

Example:
```bash
jupyter notebook error_analysis.ipynb
```

### `compare_scores.ipynb`
Notebook for visualizing score deltas and regressions.

Example:
```bash
jupyter notebook compare_scores.ipynb
```

### `descriptiveness_visualizer.ipynb`
Notebook linking descriptiveness metrics to performance deltas.

Example:
```bash
jupyter notebook descriptiveness_visualizer.ipynb
```

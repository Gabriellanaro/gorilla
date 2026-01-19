# Score Comparison Outputs

This folder contains CSVs produced by `compare_scores.py`, comparing original vs augmented
score summaries.

## Outputs
- `metrics_tidy.csv`: Long-form metrics for both conditions.
- `metrics_comparison_long.csv`: Joined original vs augmented metrics with deltas.
- `metrics_comparison_wide.csv`: Pivoted comparison table by model/category/subcategory.

## Run
```bash
python compare_scores.py --scores_orig score_desc_original_name_original --scores_aug score_desc_augmented_name_augmented --out_dir comparison_out
```

Notes:
- The script scans `score_desc_original_name_original` and `score_desc_augmented_name_augmented` for model/category subfolders.
- If no CSV is found under a model/category leaf, it falls back to `data_<category>.csv`
  at the scores root (matching rows by model name).

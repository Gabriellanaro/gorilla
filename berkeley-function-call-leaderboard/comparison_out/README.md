# Score Comparison Outputs

This folder contains CSVs produced by `compare_scores.py`, comparing multiple score roots
against a baseline.

## Outputs
- `metrics_tidy.csv`: Long-form metrics for all conditions.
- `metrics_comparison_long.csv`: Baseline vs each comparison condition with deltas.
- `metrics_comparison_wide.csv`: Baseline and all conditions in one wide table.
- `plots/`: PNG plots for selected metrics (overall only).

## Run
```bash
python compare_scores.py --out_dir comparison_out
```

Notes:
- Baseline condition: `orig_orig`.
- Subcategory filter: `overall`.
- Score roots:
- `orig_orig`: `C:\Users\b-glanaro\gorilla\berkeley-function-call-leaderboard\score_desc_original_name_original`
- `aug_aug`: `C:\Users\b-glanaro\gorilla\berkeley-function-call-leaderboard\score_desc_augmented_name_augmented`
- `aug_orig`: `C:\Users\b-glanaro\gorilla\berkeley-function-call-leaderboard\score_desc_augmented_name_original`
- `orig_aug`: `C:\Users\b-glanaro\gorilla\berkeley-function-call-leaderboard\score_desc_original_name_augmented`
- The script scans each score root for model/category subfolders.
- If no CSV is found under a model/category leaf, it falls back to `data_<category>.csv`
  at the score root (matching rows by model name).

# Leaderboard Comparison

This report compares augmented vs original scores by model/category.

## Top 10 Improvements (Primary Accuracy Metric)

| Model | Category | Subcategory | Metric | Orig | Aug | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| azure-gpt-4o-FC | multi_turn | overall | base | 53.5 | 56 | 2.5 |
| azure-gpt-5.1-responses-FC | multi_turn | overall | base | 38 | 38.5 | 0.5 |
| azure-gpt-4o-FC | non_live | overall | ast_summary | 84.67 | 83.79 | -0.88 |
| azure-gpt-4o-FC | live | overall | ast_summary | 70.1 | 68.1 | -2 |
| azure-gpt-5.1-responses-FC | live | overall | ast_summary | 63.8 | 60.03 | -3.77 |
| azure-gpt-5.1-responses-FC | non_live | overall | ast_summary | 81.81 | 77.56 | -4.25 |

## Top 10 Regressions (Primary Accuracy Metric)

| Model | Category | Subcategory | Metric | Orig | Aug | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| azure-gpt-5.1-responses-FC | non_live | overall | ast_summary | 81.81 | 77.56 | -4.25 |
| azure-gpt-5.1-responses-FC | live | overall | ast_summary | 63.8 | 60.03 | -3.77 |
| azure-gpt-4o-FC | live | overall | ast_summary | 70.1 | 68.1 | -2 |
| azure-gpt-4o-FC | non_live | overall | ast_summary | 84.67 | 83.79 | -0.88 |
| azure-gpt-5.1-responses-FC | multi_turn | overall | base | 38 | 38.5 | 0.5 |
| azure-gpt-4o-FC | multi_turn | overall | base | 53.5 | 56 | 2.5 |

## Model: azure-gpt-4o-FC

### Category: live
Primary accuracy metric: `ast_summary`

| Subcategory | ast_summary (orig) | ast_summary (aug) | ast_summary (delta) | irrelevance_detection (orig) | irrelevance_detection (aug) | irrelevance_detection (delta) | live_overall_acc (orig) | live_overall_acc (aug) | live_overall_acc (delta) | python_multiple_ast (orig) | python_multiple_ast (aug) | python_multiple_ast (delta) | python_parallel_ast (orig) | python_parallel_ast (aug) | python_parallel_ast (delta) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 70.1 | 68.1 | -2 | 79.41 | 81.22 | 1.81 | 70.1 | 68.1 | -2 | 70.09 | 68.09 | -2 | 62.5 | 68.75 | 6.25 |

### Category: multi_turn
Primary accuracy metric: `base`

| Subcategory | base (orig) | base (aug) | base (delta) | long_context (orig) | long_context (aug) | long_context (delta) | multi_turn_overall_acc (orig) | multi_turn_overall_acc (aug) | multi_turn_overall_acc (delta) | rank (orig) | rank (aug) | rank (delta) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 53.5 | 56 | 2.5 | 48.5 | 51 | 2.5 | 42.38 | 26.75 | -15.63 | 1 | 1 | 0 |

### Category: non_live
Primary accuracy metric: `ast_summary`

| Subcategory | ast_summary (orig) | ast_summary (aug) | ast_summary (delta) | irrelevance_detection (orig) | irrelevance_detection (aug) | irrelevance_detection (delta) | java_simple_ast (orig) | java_simple_ast (aug) | java_simple_ast (delta) | javascript_simple_ast (orig) | javascript_simple_ast (aug) | javascript_simple_ast (delta) | multiple_ast (orig) | multiple_ast (aug) | multiple_ast (delta) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 84.67 | 83.79 | -0.88 | 85 | 85.83 | 0.83 | 66 | 65 | -1 | 74 | 72 | -2 | 91.5 | 88.5 | -3 |

## Model: azure-gpt-5.1-responses-FC

### Category: live
Primary accuracy metric: `ast_summary`

| Subcategory | ast_summary (orig) | ast_summary (aug) | ast_summary (delta) | irrelevance_detection (orig) | irrelevance_detection (aug) | irrelevance_detection (delta) | live_overall_acc (orig) | live_overall_acc (aug) | live_overall_acc (delta) | python_multiple_ast (orig) | python_multiple_ast (aug) | python_multiple_ast (delta) | python_parallel_ast (orig) | python_parallel_ast (aug) | python_parallel_ast (delta) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 63.8 | 60.03 | -3.77 | 84.39 | 86.65 | 2.26 | 63.8 | 60.03 | -3.77 | 63.15 | 60.87 | -2.28 | 56.25 | 68.75 | 12.5 |

### Category: multi_turn
Primary accuracy metric: `base`

| Subcategory | base (orig) | base (aug) | base (delta) | multi_turn_overall_acc (orig) | multi_turn_overall_acc (aug) | multi_turn_overall_acc (delta) | rank (orig) | rank (aug) | rank (delta) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 38 | 38.5 | 0.5 | 31.63 | 9.62 | -22.01 | 2 | 2 | 0 |

### Category: non_live
Primary accuracy metric: `ast_summary`

| Subcategory | ast_summary (orig) | ast_summary (aug) | ast_summary (delta) | irrelevance_detection (orig) | irrelevance_detection (aug) | irrelevance_detection (delta) | java_simple_ast (orig) | java_simple_ast (aug) | java_simple_ast (delta) | javascript_simple_ast (orig) | javascript_simple_ast (aug) | javascript_simple_ast (delta) | multiple_ast (orig) | multiple_ast (aug) | multiple_ast (delta) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 81.81 | 77.56 | -4.25 | 88.75 | 88.75 | 0 | 63 | 63 | 0 | 66 | 64 | -2 | 88.5 | 82.5 | -6 |

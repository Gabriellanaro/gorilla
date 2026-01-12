# BFCL v4 internal dataset analysis

- Rows (tool-call decisions): **8279**
- Unique conversations (base_id): **3556**
- Source files: **16**
- Task type counts: {'multi_turn': 4625, 'single_turn': 3654}
- Rows with >1 candidate tool: **6544** (79.0%)
- Rows with empty tools: **0** (0.0%)
- Rows with missing GT tool_name: **0** (0.0%)
- Rows with empty messages: **0** (0.0%)

## Red flags

- None detected.


## Output files

- summary.json
- by_source_file.csv
- by_task_type.csv
- decisions_per_conversation_dist.csv
- candidate_tools_count_dist.csv
- top_candidate_tool_names.csv
- top_ground_truth_tool_names.csv
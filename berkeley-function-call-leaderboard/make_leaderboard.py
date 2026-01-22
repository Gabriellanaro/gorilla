"""
README:
Run: python make_leaderboard.py
Output: leaderboard.html at the project root.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CONDITIONS = {
    "OO": "score_desc_original_name_original",
    "OA": "score_desc_original_name_augmented",
    "AO": "score_desc_augmented_name_original",
    "AA": "score_desc_augmented_name_augmented",
}


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def read_generic_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - defensive
        warn(f"failed to read {csv_path}: {exc}")
        return None
    if "Model" not in df.columns:
        warn(f"missing Model column in {csv_path}")
        return None
    df = df.copy()
    df["Model"] = df["Model"].astype(str)
    for col in df.columns:
        if col in ("Model", "Rank"):
            continue
        series = df[col].astype(str).str.replace("%", "", regex=False).str.strip()
        df[col] = pd.to_numeric(series, errors="coerce")
    df = df.dropna(subset=["Model"], how="any")
    return df


def load_all_condition_data(root: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    data_map: Dict[str, Dict[str, pd.DataFrame]] = {}
    for condition, folder in CONDITIONS.items():
        folder_path = root / folder
        if not folder_path.exists():
            warn(f"missing folder for {condition}: {folder_path}")
            continue
        condition_map: Dict[str, pd.DataFrame] = {}
        for csv_path in folder_path.glob("data_*.csv"):
            df = read_generic_csv(csv_path)
            if df is None:
                continue
            condition_map[csv_path.name] = df
        if not condition_map:
            warn(f"no CSVs found for {condition} in {folder_path}")
        data_map[condition] = condition_map
    return data_map


def build_wide_table(data_map: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[pd.DataFrame, List[str]]:
    condition_frames: Dict[str, pd.DataFrame] = {}
    missing_conditions: List[str] = []
    for condition in CONDITIONS:
        df = data_map.get(condition, {}).get("data_overall.csv")
        if df is None:
            missing_conditions.append(condition)
            continue
        if "Overall Acc" not in df.columns:
            warn(f"missing Overall Acc in data_overall.csv for {condition}")
            missing_conditions.append(condition)
            continue
        condition_frames[condition] = df

    models: List[str] = []
    if condition_frames:
        model_sets = [set(df["Model"].tolist()) for df in condition_frames.values()]
        models = sorted(set().union(*model_sets))
    else:
        warn("no condition CSVs loaded; output will be empty")

    wide = pd.DataFrame(index=models)
    for condition, df in condition_frames.items():
        mapping = df.set_index("Model")["Overall Acc"]
        wide[f"{condition}_acc"] = mapping.reindex(models)

    return wide, missing_conditions


def compute_deltas(wide: pd.DataFrame) -> pd.DataFrame:
    result = wide.copy()
    if "OO_acc" in result.columns:
        baseline = result["OO_acc"]
    else:
        baseline = pd.Series([math.nan] * len(result), index=result.index)

    for condition in ("OO", "OA", "AO", "AA"):
        acc_col = f"{condition}_acc"
        delta_col = f"{condition}_delta"
        if acc_col not in result.columns:
            result[acc_col] = math.nan
        if condition == "OO":
            result[delta_col] = 0.0
        else:
            delta = result[acc_col] - baseline
            delta[baseline.isna()] = math.nan
            result[delta_col] = delta
    return result


def make_rows(table: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, row in table.iterrows():
        accs = {
            "OO": row.get("OO_acc"),
            "OA": row.get("OA_acc"),
            "AO": row.get("AO_acc"),
            "AA": row.get("AA_acc"),
        }
        max_acc = None
        for val in accs.values():
            if pd.notna(val):
                if max_acc is None or val > max_acc:
                    max_acc = float(val)

        rows.append(
            {
                "model": str(model),
                "OO": float(accs["OO"]) if pd.notna(accs["OO"]) else None,
                "OA": float(accs["OA"]) if pd.notna(accs["OA"]) else None,
                "AO": float(accs["AO"]) if pd.notna(accs["AO"]) else None,
                "AA": float(accs["AA"]) if pd.notna(accs["AA"]) else None,
                "bestAcc": max_acc,
            }
        )
    return rows


def filename_to_title(filename: str) -> str:
    stem = filename.replace("data_", "").replace(".csv", "")
    return stem.replace("_", " ").title()


def build_category_tables(
    data_map: Dict[str, Dict[str, pd.DataFrame]]
) -> List[Dict[str, object]]:
    filenames: set[str] = set()
    for condition_data in data_map.values():
        filenames.update(condition_data.keys())
    filenames.discard("data_overall.csv")
    filenames.discard("data_format_sensitivity.csv")

    tables: List[Dict[str, object]] = []
    for filename in sorted(filenames):
        condition_frames: Dict[str, pd.DataFrame] = {}
        missing_conditions: List[str] = []
        for condition in CONDITIONS:
            df = data_map.get(condition, {}).get(filename)
            if df is None:
                missing_conditions.append(condition)
                continue
            condition_frames[condition] = df
        if not condition_frames:
            continue

        sample_df = next(iter(condition_frames.values()))
        metrics = [col for col in sample_df.columns if col not in ("Rank", "Model")]

        models: List[str] = []
        model_sets = [set(df["Model"].tolist()) for df in condition_frames.values()]
        if model_sets:
            models = sorted(set().union(*model_sets))

        per_condition_maps: Dict[str, pd.DataFrame] = {}
        for condition, df in condition_frames.items():
            per_condition_maps[condition] = df.set_index("Model")

        rows: List[Dict[str, object]] = []
        for model in models:
            metric_values: Dict[str, Dict[str, Optional[float]]] = {}
            for metric in metrics:
                metric_values[metric] = {}
                for condition in CONDITIONS:
                    table = per_condition_maps.get(condition)
                    if table is None or metric not in table.columns:
                        metric_values[metric][condition] = None
                        continue
                    value = table.at[model, metric] if model in table.index else math.nan
                    metric_values[metric][condition] = (
                        float(value) if pd.notna(value) else None
                    )
            rows.append({"model": str(model), "metrics": metric_values})

        tables.append(
            {
                "key": filename.replace(".csv", ""),
                "title": filename_to_title(filename),
                "metrics": metrics,
                "rows": rows,
                "missing": missing_conditions,
            }
        )

    return tables


def render_html(
    rows: List[Dict[str, object]],
    missing_conditions: List[str],
    category_tables: List[Dict[str, object]],
    all_models: List[str],
) -> str:
    data_json = json.dumps(rows)
    categories_json = json.dumps(category_tables)
    models_json = json.dumps(all_models)
    missing_note = ""
    if missing_conditions:
        missing_note = (
            "Missing data for conditions: "
            + ", ".join(sorted(missing_conditions))
            + ". Showing available results."
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Function Calling Leaderboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {{
      --bg: #f3f6ff;
      --panel: #ffffff;
      --border: #d7e3ff;
      --text: #1d2b4f;
      --muted: #5b6b8a;
      --cell: #c8d7ff;
      --best: #7ef07b;
      --baseline: #bcd0ff;
    }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #f0f5ff, #e5efff);
      color: var(--text);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }}
    .wrap {{
      padding: 24px;
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 24px;
      font-weight: 700;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      background: var(--panel);
      padding: 12px 16px;
      border: 1px solid var(--border);
      border-radius: 10px;
      box-shadow: 0 6px 16px rgba(23, 51, 94, 0.08);
      margin-bottom: 16px;
    }}
    .controls label {{
      font-size: 13px;
      color: var(--muted);
    }}
    select, button {{
      margin-left: 6px;
      padding: 6px 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #f8fbff;
      color: var(--text);
      font-size: 13px;
    }}
    button {{
      cursor: pointer;
    }}
    .filter {{
      min-width: 200px;
    }}
    .model-filter {{
      width: 220px;
      min-height: 120px;
    }}
    .sortable {{
      cursor: pointer;
      user-select: none;
    }}
    .sortable:hover {{
      text-decoration: underline;
      background: #eef3ff;
    }}
    .overall-header {{
      background: #e2e6ef;
      font-weight: 700;
    }}
    .sort-indicator {{
      margin-left: 6px;
      font-size: 11px;
      color: var(--muted);
    }}
    .note {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .section {{
      margin-top: 24px;
    }}
    .section h2 {{
      margin: 16px 0 8px;
      font-size: 18px;
    }}
    .section .note {{
      margin-top: 0;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--panel);
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 760px;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #f9fbff;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      padding: 10px;
      text-align: center;
    }}
    tbody td {{
      padding: 12px 10px;
      text-align: center;
      background: var(--cell);
      border-bottom: 1px solid #e6ecff;
      border-right: 1px solid #e6ecff;
      font-size: 14px;
    }}
    tbody td:first-child {{
      text-align: left;
      background: #f2f6ff;
      font-weight: 600;
      color: #15307a;
    }}
    tbody td.baseline {{
      background: var(--baseline);
      font-weight: 600;
    }}
    tbody td.best {{
      background: var(--best);
      font-weight: 700;
      color: #0c3a0c;
    }}
    tbody tr:last-child td {{
      border-bottom: none;
    }}
    tbody td:last-child, thead th:last-child {{
      border-right: none;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>OpenFunctions Leaderboard</h1>
    <div class="controls">
      <label>Delta format
        <select id="deltaMode">
          <option value="pp">pp</option>
          <option value="rel">relative%</option>
        </select>
      </label>
      <label>Table
        <select id="tableSelect"></select>
      </label>
      <label class="filter">Models
        <select id="modelFilter" class="model-filter" multiple></select>
      </label>
      <button id="selectAllModels">All</button>
      <button id="clearModels">None</button>
    </div>
    <div class="note" id="note">{missing_note}</div>
    <div class="table-wrap">
      <table data-table-key="main">
        <thead>
          <tr>
            <th>Model</th>
            <th>OO</th>
            <th>OA</th>
            <th>AO</th>
            <th>AA</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
    <div id="detailSections"></div>
  </div>
  <script>
    const MISSING = "--";
    const mainRows = {data_json};
    const categoryTables = {categories_json};
    const allModels = {models_json};
    const conditions = ["OO", "OA", "AO", "AA"];
    const tableState = {{}};

    function isNumber(value) {{
      return typeof value === "number" && !Number.isNaN(value);
    }}

    function toFixed(value, digits) {{
      return isNumber(value) ? value.toFixed(digits) : MISSING;
    }}

    function formatDelta(acc, baseline, mode) {{
      if (!isNumber(acc) || !isNumber(baseline)) {{
        return MISSING;
      }}
      if (mode === "rel") {{
        if (baseline === 0) {{
          return MISSING;
        }}
        const rel = ((acc - baseline) / baseline) * 100;
        return (rel >= 0 ? "+" : "") + rel.toFixed(2) + "%";
      }}
      const delta = acc - baseline;
      return (delta >= 0 ? "+" : "") + delta.toFixed(2);
    }}

    function buildTableSelect() {{
      const select = document.getElementById("tableSelect");
      select.innerHTML = "";
      const mainOption = document.createElement("option");
      mainOption.value = "main";
      mainOption.textContent = "Main";
      select.appendChild(mainOption);
      const allOption = document.createElement("option");
      allOption.value = "all";
      allOption.textContent = "All";
      select.appendChild(allOption);
      categoryTables.forEach((table) => {{
        const option = document.createElement("option");
        option.value = table.key;
        option.textContent = table.title;
        select.appendChild(option);
      }});
      select.value = "main";
    }}

    function buildModelFilter() {{
      const select = document.getElementById("modelFilter");
      select.innerHTML = "";
      allModels.forEach((model) => {{
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        option.selected = true;
        select.appendChild(option);
      }});
    }}

    function getSelectedModels() {{
      const select = document.getElementById("modelFilter");
      const values = Array.from(select.selectedOptions).map((option) => option.value);
      if (values.length === 0) {{
        return new Set(allModels);
      }}
      return new Set(values);
    }}

    function renderMainTable() {{
      const body = document.getElementById("tableBody");
      const mode = document.getElementById("deltaMode").value;
      const selected = getSelectedModels();
      body.innerHTML = "";
      const tableKey = "main";
      const sortedRows = getSortedRows(tableKey, mainRows);
      sortedRows.forEach((row) => {{
        if (!selected.has(row.model)) {{
          return;
        }}
        const tr = document.createElement("tr");
        const baseline = row.OO;
        const bestAcc = row.bestAcc;

        function buildCell(label, value, isBaseline, condition) {{
          const td = document.createElement("td");
          if (label === "Model") {{
            td.textContent = value ?? MISSING;
            return td;
          }}
          const accText = toFixed(value, 2);
          if (condition === "OO") {{
            td.textContent = accText;
          }} else {{
            const deltaText = formatDelta(value, baseline, mode);
            td.textContent = accText + " (" + deltaText + ")";
          }}
          if (isBaseline) {{
            td.classList.add("baseline");
          }}
          if (isNumber(value) && isNumber(bestAcc) && value === bestAcc) {{
            td.classList.add("best");
          }}
          return td;
        }}

        tr.appendChild(buildCell("Model", row.model, false, "Model"));
        tr.appendChild(buildCell("OO", row.OO, true, "OO"));
        tr.appendChild(buildCell("OA", row.OA, false, "OA"));
        tr.appendChild(buildCell("AO", row.AO, false, "AO"));
        tr.appendChild(buildCell("AA", row.AA, false, "AA"));
        body.appendChild(tr);
      }});
    }}

    function getSortValueMain(row, key, mode) {{
      if (key === "model") {{
        return row.model ? row.model.toLowerCase() : "";
      }}
      if (key.endsWith("_delta")) {{
        const condition = key.split("_")[0];
        if (condition === "OO") {{
          return 0;
        }}
        const acc = row[condition];
        const baseline = row.OO;
        if (!isNumber(acc) || !isNumber(baseline)) {{
          return Number.NEGATIVE_INFINITY;
        }}
        if (mode === "rel") {{
          return baseline === 0 ? Number.NEGATIVE_INFINITY : ((acc - baseline) / baseline) * 100;
        }}
        return acc - baseline;
      }}
      const value = row[key];
      if (isNumber(value)) {{
        return value;
      }}
      return Number.NEGATIVE_INFINITY;
    }}

    function getSortValueDetail(row, sortSpec) {{
      if (sortSpec.key === "model") {{
        return row.model ? row.model.toLowerCase() : "";
      }}
      const metric = sortSpec.metric;
      const condition = sortSpec.condition;
      const metricValues = row.metrics[metric] || {{}};
      const value = metricValues[condition];
      if (isNumber(value)) {{
        return value;
      }}
      return Number.NEGATIVE_INFINITY;
    }}

    function getSortedRows(tableKey, rows) {{
      const mode = document.getElementById("deltaMode").value;
      const state = tableState[tableKey];
      const sortKey = state ? state.sortKey : null;
      const sortDir = state ? state.sortDir : null;
      const multiplier = sortDir === "asc" ? 1 : -1;
      const sortSpec = sortKey ? buildSortSpec(tableKey, sortKey) : null;
      const sorted = rows.slice();
      sorted.sort((a, b) => {{
        let av;
        let bv;
        if (sortKey) {{
          if (tableKey === "main") {{
            av = getSortValueMain(a, sortKey, mode);
            bv = getSortValueMain(b, sortKey, mode);
          }} else {{
            av = getSortValueDetail(a, sortSpec);
            bv = getSortValueDetail(b, sortSpec);
          }}
          if (av > bv) return 1 * multiplier;
          if (av < bv) return -1 * multiplier;
        }}
        return a.model.localeCompare(b.model);
      }});
      return sorted;
    }}

    function buildSortSpec(tableKey, sortKey) {{
      if (tableKey === "main") {{
        return {{ key: sortKey }};
      }}
      const [metric, condition] = sortKey.split("::");
      return {{ key: sortKey, metric, condition }};
    }}

    function renderDetailTables() {{
      const container = document.getElementById("detailSections");
      const mode = document.getElementById("deltaMode").value;
      const selected = getSelectedModels();
      const selectedTable = document.getElementById("tableSelect").value;
      container.innerHTML = "";
      categoryTables.forEach((table) => {{
        if (selectedTable !== "all" && selectedTable !== table.key) {{
          return;
        }}
        const section = document.createElement("div");
        section.className = "section";
        section.id = `section-${{table.key}}`;

        const heading = document.createElement("h2");
        heading.textContent = table.title;
        section.appendChild(heading);

        if (table.missing && table.missing.length) {{
          const note = document.createElement("div");
          note.className = "note";
          note.textContent = `Missing data for conditions: ${{table.missing.join(", ")}}. Showing available results.`;
          section.appendChild(note);
        }}

        const wrap = document.createElement("div");
        wrap.className = "table-wrap";
        const tableEl = document.createElement("table");
        tableEl.dataset.tableKey = table.key;

        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        const modelTh = document.createElement("th");
        modelTh.textContent = "Model";
        modelTh.classList.add("sortable");
        modelTh.dataset.sortKey = "model";
        headerRow.appendChild(modelTh);
        table.metrics.forEach((metric) => {{
          const isOverall = metric.toLowerCase().includes("overall");
          conditions.forEach((condition) => {{
            const th = document.createElement("th");
            th.classList.add("sortable");
            th.dataset.sortKey = `${{metric}}::${{condition}}`;
            th.textContent = `${{metric}} ${{condition}}`;
            if (isOverall) {{
              th.classList.add("overall-header");
            }}
            headerRow.appendChild(th);
          }});
        }});
        thead.appendChild(headerRow);
        tableEl.appendChild(thead);

        const tbody = document.createElement("tbody");
        const tableKey = table.key;
        const sortedRows = getSortedRows(tableKey, table.rows);
        sortedRows.forEach((row) => {{
          if (!selected.has(row.model)) {{
            return;
          }}
          const tr = document.createElement("tr");
          const modelTd = document.createElement("td");
          modelTd.textContent = row.model ?? MISSING;
          tr.appendChild(modelTd);
          table.metrics.forEach((metric) => {{
            const metricValues = row.metrics[metric] || {{}};
            const bestAcc = Math.max(
              ...conditions
                .map((condition) => metricValues[condition])
                .filter((value) => isNumber(value))
            );
            conditions.forEach((condition) => {{
              const value = metricValues[condition];
              const td = document.createElement("td");
              const baseline = metricValues["OO"];
              const accText = toFixed(value, 2);
              if (condition === "OO") {{
                td.textContent = accText;
                td.classList.add("baseline");
              }} else {{
                const deltaText = formatDelta(value, baseline, mode);
                td.textContent = accText + " (" + deltaText + ")";
              }}
              if (isNumber(value) && isNumber(bestAcc) && value === bestAcc) {{
                td.classList.add("best");
              }}
              tr.appendChild(td);
            }});
          }});
          tbody.appendChild(tr);
        }});
        tableEl.appendChild(tbody);
        wrap.appendChild(tableEl);
        section.appendChild(wrap);
        container.appendChild(section);
        updateSortIndicators(tableEl, table.key);
      }});
    }}

    function renderAllTables() {{
      const selectedTable = document.getElementById("tableSelect").value;
      const mainSection = document.querySelector(".table-wrap");
      const note = document.getElementById("note");
      if (selectedTable === "main" || selectedTable === "all") {{
        mainSection.style.display = "";
        note.style.display = "";
        renderMainTable();
        const mainTable = document.querySelector("table[data-table-key='main']");
        if (mainTable) {{
          updateSortIndicators(mainTable, "main");
        }}
      }} else {{
        mainSection.style.display = "none";
        note.style.display = "none";
      }}
      renderDetailTables();
    }}

    function bindControls() {{
      document.getElementById("deltaMode").addEventListener("change", () => {{
        renderAllTables();
        attachHeaderSorting();
      }});
      document.getElementById("tableSelect").addEventListener("change", (event) => {{
        renderAllTables();
        attachHeaderSorting();
      }});
      document.getElementById("modelFilter").addEventListener("change", () => {{
        renderAllTables();
        attachHeaderSorting();
      }});
      document.getElementById("selectAllModels").addEventListener("click", () => {{
        const select = document.getElementById("modelFilter");
        Array.from(select.options).forEach((option) => {{
          option.selected = true;
        }});
        renderAllTables();
        attachHeaderSorting();
      }});
      document.getElementById("clearModels").addEventListener("click", () => {{
        const select = document.getElementById("modelFilter");
        Array.from(select.options).forEach((option) => {{
          option.selected = false;
        }});
        renderAllTables();
        attachHeaderSorting();
      }});
    }}

    function updateSortIndicators(tableElement, tableKey) {{
      const state = tableState[tableKey] || {{ sortKey: null, sortDir: null }};
      tableElement.querySelectorAll("th.sortable").forEach((th) => {{
        const indicator = th.querySelector(".sort-indicator");
        if (indicator) {{
          indicator.remove();
        }}
        if (state.sortKey && th.dataset.sortKey === state.sortKey) {{
          const span = document.createElement("span");
          span.className = "sort-indicator";
          span.textContent = state.sortDir === "asc" ? "▲" : "▼";
          th.appendChild(span);
        }}
      }});
    }}

    function attachHeaderSorting() {{
      document.querySelectorAll("table").forEach((table) => {{
        table.querySelectorAll("th.sortable").forEach((th) => {{
          if (th.dataset.bound === "1") {{
            return;
          }}
          th.dataset.bound = "1";
          th.addEventListener("click", () => {{
            const tableKey = table.dataset.tableKey || "main";
            const key = th.dataset.sortKey;
            const state = tableState[tableKey] || {{ sortKey: null, sortDir: null }};
            if (state.sortKey !== key) {{
              state.sortKey = key;
              state.sortDir = "asc";
            }} else if (state.sortDir === "asc") {{
              state.sortDir = "desc";
            }} else {{
              state.sortKey = null;
              state.sortDir = null;
            }}
            tableState[tableKey] = state;
            renderAllTables();
            attachHeaderSorting();
          }});
        }});
      }});
    }}

    function renderMainHeader() {{
      const headerRow = document.querySelector("table thead tr");
      if (!headerRow) {{
        return;
      }}
      headerRow.querySelectorAll("th").forEach((th, index) => {{
        if (index === 0) {{
          th.classList.add("sortable");
          th.dataset.sortKey = "model";
        }} else {{
          const condition = conditions[index - 1];
          th.classList.add("sortable");
          th.dataset.sortKey = condition;
        }}
      }});
    }}

    function renderAllTablesWithHeaders() {{
      renderMainHeader();
      renderAllTables();
      attachHeaderSorting();
    }}

    buildTableSelect();
    buildModelFilter();
    tableState["main"] = {{ sortKey: null, sortDir: null }};
    categoryTables.forEach((table) => {{
      tableState[table.key] = {{ sortKey: null, sortDir: null }};
    }});
    bindControls();
    renderAllTablesWithHeaders();
  </script>
</body>
</html>
"""


def main() -> int:
    root = Path.cwd()
    data_map = load_all_condition_data(root)
    wide, missing_conditions = build_wide_table(data_map)
    table = compute_deltas(wide)
    rows = make_rows(table)
    category_tables = build_category_tables(data_map)
    all_models = sorted(
        {
            row["model"]
            for row in rows
        }.union(
            {
                detail_row["model"]
                for table_info in category_tables
                for detail_row in table_info["rows"]
            }
        )
    )
    html = render_html(rows, missing_conditions, category_tables, all_models)
    out_path = root / "leaderboard.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"leaderboard written: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

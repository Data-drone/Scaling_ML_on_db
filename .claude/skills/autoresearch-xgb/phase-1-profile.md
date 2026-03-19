# Phase 1: PROFILE

Profile the dataset using the SQL Statement API. No cluster needed — runs
against a SQL warehouse.

## Prerequisites

- `table`: full UC table name (e.g. `brian_gen_ai.xgb_scaling.imbalanced_10m`)
- `target_col`: target column name
- `warehouse_id`: SQL warehouse ID (default: `148ccb90800933a1`)

## Step 1: Get row count

SQL via Statement API (see api-reference.md):

```sql
SELECT COUNT(*) as row_count FROM {table}
```

Parse `row_count` from `result.data_array[0][0]`.

## Step 2: Get column metadata

```sql
DESCRIBE TABLE {table}
```

Parse the response to classify each column:
- `int`, `bigint`, `float`, `double`, `decimal` → numeric
- `string` → categorical
- `boolean` → boolean
- `timestamp`, `date` → timestamp (drop for v1)

Record: column name, type classification, data type.

## Step 3: Get null percentages

For each column (batch up to ~50 per query to avoid SQL length limits):

```sql
SELECT
  SUM(CASE WHEN col_1 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as null_pct_col_1,
  SUM(CASE WHEN col_2 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as null_pct_col_2,
  ...
FROM {table}
```

## Step 4: Get categorical cardinality

For each categorical column (batched):

```sql
SELECT
  COUNT(DISTINCT cat_col_1) as card_1,
  COUNT(DISTINCT cat_col_2) as card_2,
  ...
FROM {table}
```

## Step 5: Get class distribution

```sql
SELECT {target_col}, COUNT(*) as cnt
FROM {table}
GROUP BY {target_col}
ORDER BY {target_col}
```

## Step 6: Get numeric basic stats

For numeric columns (batched):

```sql
SELECT
  MIN(num_col) as min_val,
  MAX(num_col) as max_val,
  AVG(num_col) as mean_val,
  STDDEV(num_col) as std_val
FROM {table}
```

**v1 simplification:** Skip skewness in SQL. Compute it on the cluster in
Phase 3 after loading data — avoids complex SQL.

## Step 7: Get sample rows

```sql
SELECT * FROM {table} LIMIT 5
```

Useful for inspecting column names and values.

## Output

Collect all results into a profile dict. Example:

```python
profile = {
    "table": "brian_gen_ai.xgb_scaling.imbalanced_10m",
    "target_col": "label",
    "row_count": 10_000_000,
    "columns": [
        {"name": "feat_0", "type": "numeric", "null_pct": 0.0},
        {"name": "cat_0", "type": "categorical", "null_pct": 2.1, "cardinality": 12},
        ...
    ],
    "numeric_count": 200,
    "categorical_count": 50,
    "boolean_count": 0,
    "class_distribution": {0: 9_500_000, 1: 500_000},
    "sample_rows": [...],
}
```

This profile drives Phase 2 (cluster sizing) and Phase 3 (feature decisions).

## Notebook Cells

Add to the notebook after this phase:

1. **Markdown cell:** `## 1. Data Profile` with a summary table:

```
# MAGIC %md
# MAGIC ## 1. Data Profile
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Table | `{table}` |
# MAGIC | Rows | {row_count:,} |
# MAGIC | Numeric features | {numeric_count} |
# MAGIC | Categorical features | {categorical_count} |
# MAGIC | Estimated raw size | {estimated_gb:.1f} GB |
# MAGIC | Class balance | {class_0_pct:.1f}% / {class_1_pct:.1f}% |
```

2. **Code cell:** Profiling code (so notebook is reproducible when re-run)

3. **Markdown cell:** Key findings and decisions for Phase 3

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

Query `SUM(CASE WHEN col IS NULL THEN 1 ELSE 0 END)*100.0/COUNT(*)` for each column, batched ~50 cols/query to avoid SQL length limits.

## Step 4: Get categorical cardinality

Query `COUNT(DISTINCT cat_col)` for each categorical column (batched).

## Step 5: Get class distribution

`SELECT {target_col}, COUNT(*) as cnt FROM {table} GROUP BY {target_col} ORDER BY {target_col}`

## Step 6: Get numeric basic stats

Query `MIN, MAX, AVG, STDDEV` for numeric columns (batched). Skip skewness in SQL — compute on cluster in Phase 3.

## Step 7: Get sample rows

```sql
SELECT * FROM {table} LIMIT 5
```

Useful for inspecting column names and values.

## Output

Collect into a `profile` dict with fields: `table`, `target_col`, `row_count`, `columns` (list of {name, type, null_pct, cardinality}), `numeric_count`, `categorical_count`, `boolean_count`, `class_distribution`, `sample_rows`. This drives Phase 2 (cluster sizing) and Phase 3 (feature decisions).

## Notebook Cells

Add to notebook (see notebook-format.md for `# MAGIC` syntax):

1. **Markdown cell:** `## 1. Data Profile` — summary table with table name, row count, numeric/categorical counts, estimated size, class balance.
2. **Code cell:** Profiling code (so notebook is reproducible when re-run).
3. **Markdown cell:** Key findings and decisions for Phase 3.

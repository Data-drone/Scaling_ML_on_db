# Notebook Format Reference

Databricks `.py` source notebook, built incrementally. Stored locally, uploaded after each phase.

## Format Rules

- First line: `# Databricks notebook source`
- Cell separator: `# COMMAND ----------`
- Markdown cells: every line prefixed with `# MAGIC` (first line `# MAGIC %md`)
- Pip install: `# MAGIC %pip install -U mlflow psutil`
- Python restart: `# MAGIC %restart_python`

## Local File Path

    /workspace/group/scaling_xgb_work/notebooks/autoresearch/{table_name}_{YYYYMMDD}.py

Use Write tool for initial skeleton, Edit tool to append cells.

## Notebook Skeleton (created in Phase 2)

```python
# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # AutoResearch: {table_name}
# MAGIC
# MAGIC **Generated:** {date}
# MAGIC **Table:** `{catalog}.{schema}.{table_name}`
# MAGIC **Target:** `{target_col}`
# MAGIC **Budget:** {budget_minutes} minutes
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC _Summary will be added after experiments complete._

# COMMAND ----------

# MAGIC %pip install -U mlflow psutil

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Imports and setup will be added in Phase 3
```

## Appending Cells

Append after the last line of the notebook. Format (same for code and markdown):

```python

# COMMAND ----------

# code goes here (or # MAGIC %md for markdown)
```

For markdown cells, prefix every line with `# MAGIC`:

```
# MAGIC %md
# MAGIC ## Section Title
# MAGIC
# MAGIC Explanation text.
```

**Key principle:** Write markdown cells AFTER the code cell runs so explanations describe actual outcomes, not intentions.

Upload notebook via Workspace Import API (api-reference.md) after each phase. Summary cell details in phase-5-finalize.md.

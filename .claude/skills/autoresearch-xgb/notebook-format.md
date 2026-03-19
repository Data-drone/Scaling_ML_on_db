# Notebook Format Reference

The autoresearch agent builds a Databricks `.py` source notebook incrementally.
It is stored locally during the session and uploaded to the workspace periodically
(after each phase) and at the end.

## Databricks .py Source Format

The first line MUST be:

    # Databricks notebook source

Each cell is separated by:

    # COMMAND ----------

Markdown cells use the `# MAGIC %md` prefix (each line):

    # MAGIC %md
    # MAGIC ## Section Title
    # MAGIC
    # MAGIC Some explanation text here.

Pip install cells:

    # MAGIC %pip install -U mlflow psutil

Python restart:

    # MAGIC %restart_python

## Local File Path

Store the notebook locally at:

    /workspace/group/scaling_xgb_work/notebooks/autoresearch/{table_name}_{YYYYMMDD}.py

Use the Write tool to create the initial skeleton and the Edit tool to append
cells as the agent progresses through phases.

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

To add a new cell to the notebook, append after the last line:

```python
# (blank line)
# COMMAND ----------
# (blank line)
# cell content here
```

For a markdown cell:

```python
# (blank line)
# COMMAND ----------
# (blank line)
# MAGIC %md
# MAGIC ## New Section
# MAGIC
# MAGIC Explanation of what this section does.
```

**Key principle:** Markdown cells are written AFTER the code cell runs and the
agent sees the output. This ensures explanations describe actual outcomes, not
just intentions.

## Uploading to Workspace

After each phase and at the end, upload via the Workspace Import API (see
api-reference.md):

```bash
NB_CONTENT=$(base64 -w0 notebooks/autoresearch/{notebook_file})
# POST to /api/2.0/workspace/import with content=${NB_CONTENT}
```

Upload path: `/Users/{user_email}/autoresearch/{table_name}_{YYYYMMDD}`

Set `"overwrite": true` to update the existing notebook on each upload.

## Summary Cell (Phase 5)

After all experiments, use the Edit tool to replace the placeholder summary at
the top of the notebook with the final results:

- Best model metrics (AUC-PR, AUC-ROC, F1)
- Track chosen and why
- Feature engineering decisions
- Experiments run and comparison table
- Total time and cluster config
- MLflow run ID and model registry name

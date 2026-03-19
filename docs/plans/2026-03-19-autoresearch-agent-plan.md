# AutoResearch Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Claude Code skill that autonomously profiles a UC table, creates a right-sized cluster, runs iterative XGBoost experiments via the Command Execution API, and produces a polished notebook artifact.

**Architecture:** A new skill `autoresearch-xgb` containing a SKILL.md orchestrator, phase-specific recipe files, and a cluster/execution API interaction reference. The agent follows a 5-phase pipeline: PROFILE → SIZE & CREATE → FEATURE ENG → TRAIN → FINALIZE. All cluster interaction is via Databricks REST APIs called from the Claude Code sandbox.

**Tech Stack:** Databricks REST APIs (Clusters 2.0, Command Execution 1.2, Workspace 2.0, SQL Statements), MLflow, XGBoost, Ray (for distributed track), databricks CLI / curl.

**Design doc:** `docs/plans/2026-03-19-autoresearch-agent-design.md`

---

## Task 1: Create skill skeleton — SKILL.md

**Files:**
- Create: `.claude/skills/autoresearch-xgb/SKILL.md`

**Step 1: Write the SKILL.md orchestrator**

This is the main entry point. It tells Claude what inputs to extract, what phases to run, and links to the phase-specific recipe files.

```markdown
---
name: autoresearch-xgb
description: Autonomous XGBoost research agent. Given a Unity Catalog table and target column, profiles the data, creates a right-sized cluster, runs iterative experiments via Command Execution API, and produces a polished Databricks notebook. Use when asked to "research", "auto-train", "find the best model", or "explore and train" on a dataset.
---

# AutoResearch: XGBoost on Databricks

Autonomous agent that explores a dataset and produces the best XGBoost model
it can find within a time budget. The output is a well-documented Databricks
notebook uploaded to the workspace.

## Inputs

Extract from the user's request:

| Input | Required | Example | Default |
|-------|----------|---------|---------|
| `table` | yes | `brian_gen_ai.xgb_scaling.imbalanced_10m` | — |
| `target_col` | yes | `label` | — |
| `budget_minutes` | no | `45` | `60` |

Parse `catalog`, `schema`, `table_name` from the three-part table identifier.

## Checklist

Copy this and track progress:

- [ ] Phase 1: PROFILE — Profile the dataset via SQL warehouse
- [ ] Phase 2: SIZE & CREATE — Calculate cluster size, create cluster + context
- [ ] Phase 3: FEATURE ENG — Run feature engineering cells on live cluster
- [ ] Phase 4: TRAIN — Run baseline + experiments, compare results
- [ ] Phase 5: FINALIZE — Clean up notebook, register model, teardown

## Phase 1: PROFILE

See [phase-1-profile.md](phase-1-profile.md).

Quick summary: Run SQL queries via the SQL Statement API against a warehouse
to collect row count, column types, cardinality, nulls, class distribution,
and basic stats. No cluster needed.

## Phase 2: SIZE & CREATE

See [phase-2-cluster.md](phase-2-cluster.md).

Quick summary: Estimate memory from profile, pick cluster config from decision
table, create cluster via REST API, wait for RUNNING, create execution context.
While waiting, scaffold the notebook skeleton locally.

## Phase 3: FEATURE ENGINEERING

See [phase-3-features.md](phase-3-features.md).

Quick summary: Run cells on the live cluster to load data, encode categoricals,
handle nulls, detect skew, cap feature count, split train/test. Each step is
a code cell with markdown explanation. Upload notebook to workspace after this
phase (crash recovery checkpoint).

## Phase 4: TRAIN & EVALUATE

See [phase-4-train.md](phase-4-train.md).

Quick summary: Run baseline XGBoost, then 1-2 variations if budget allows.
Read metrics from MLflow API. Stop early if no improvement after 2 experiments.
Upload notebook after each training run.

## Phase 5: FINALIZE

See [phase-5-finalize.md](phase-5-finalize.md).

Quick summary: Compare results, register best model to UC, add summary to
notebook top, upload final notebook, terminate cluster, report to user.

## API Reference

See [api-reference.md](api-reference.md) for the exact REST API calls used
in each phase (cluster create, command execute, workspace import, etc.).

## Notebook Format

See [notebook-format.md](notebook-format.md) for the .py source notebook
structure and how to build it incrementally.

## Error Handling

- Cell error → read error, one retry with fix, if still fails log in markdown
- Cluster dies → save local notebook, upload partial, report to user
- OOM → log error, try fewer features or smaller sample
- Budget exceeded → skip remaining experiments, go to Phase 5
- No metric improvement after 2 experiments → stop early, go to Phase 5
```

**Step 2: Commit**

```bash
git add .claude/skills/autoresearch-xgb/SKILL.md
git commit -m "feat(autoresearch): add skill skeleton SKILL.md"
```

---

## Task 2: API reference — how to call Databricks REST APIs

**Files:**
- Create: `.claude/skills/autoresearch-xgb/api-reference.md`

**Step 1: Write the API reference**

This file documents every REST API call the agent needs, with exact curl/CLI
commands and expected responses. The agent copies these patterns verbatim.

```markdown
# API Reference — Databricks REST APIs

All calls use the `databricks` CLI or `curl` with bearer token authentication.
Token: `$(databricks-token)` (outputs a fresh M2M OAuth token).
Host: from `$DATABRICKS_HOST` env var.

## Authentication Pattern

For all curl calls:
    curl -s \
      -H "Authorization: Bearer $(databricks-token)" \
      -H "Content-Type: application/json" \
      "${DATABRICKS_HOST}/api/..."

For databricks CLI:
    databricks api post /api/... --json '{...}'

## Cluster Lifecycle

### Create cluster

    curl -s -X POST \
      -H "Authorization: Bearer $(databricks-token)" \
      -H "Content-Type: application/json" \
      "${DATABRICKS_HOST}/api/2.0/clusters/create" \
      -d '{
        "cluster_name": "autoresearch-{table_name}-{timestamp}",
        "spark_version": "17.3.x-cpu-ml-scala2.13",
        "node_type_id": "Standard_D16s_v5",
        "num_workers": 0,
        "autotermination_minutes": 70,
        "data_security_mode": "SINGLE_USER",
        "spark_conf": {
          "spark.databricks.cluster.profile": "singleNode",
          "spark.master": "local[*, 4]"
        },
        "custom_tags": {
          "ResourceClass": "SingleNode",
          "project": "autoresearch"
        },
        "azure_attributes": {
          "availability": "SPOT_WITH_FALLBACK_AZURE",
          "spot_bid_max_price": -1
        }
      }'

Response: `{"cluster_id": "0319-123456-abc123"}`

For Ray distributed (multi-node), change:
- `"num_workers": 4` (or calculated N)
- Remove `singleNode` profile and `local[*,4]` from spark_conf
- Add `"spark.executorEnv.OMP_NUM_THREADS": "15"` to spark_conf
- Change `ResourceClass` to `"default"`

### Poll cluster status

    curl -s \
      -H "Authorization: Bearer $(databricks-token)" \
      "${DATABRICKS_HOST}/api/2.0/clusters/get?cluster_id={CLUSTER_ID}" \
      | jq -r '.state'

Poll every 15 seconds until state = `RUNNING`. Typical wait: 3-5 minutes.
Possible states: PENDING → RUNNING (success) or TERMINATED/ERROR (failure).

### Terminate cluster

    curl -s -X POST \
      -H "Authorization: Bearer $(databricks-token)" \
      -H "Content-Type: application/json" \
      "${DATABRICKS_HOST}/api/2.0/clusters/delete" \
      -d '{"cluster_id": "{CLUSTER_ID}"}'

## Command Execution

### Create execution context

    curl -s -X POST \
      -H "Authorization: Bearer $(databricks-token)" \
      -H "Content-Type: application/json" \
      "${DATABRICKS_HOST}/api/1.2/contexts/create" \
      -d '{"clusterId": "{CLUSTER_ID}", "language": "python"}'

Response: `{"id": "1234567890"}`

### Execute a command (run a cell)

    curl -s -X POST \
      -H "Authorization: Bearer $(databricks-token)" \
      -H "Content-Type: application/json" \
      "${DATABRICKS_HOST}/api/1.2/commands/execute" \
      -d '{
        "clusterId": "{CLUSTER_ID}",
        "contextId": "{CONTEXT_ID}",
        "language": "python",
        "command": "print(\"hello world\")"
      }'

Response: `{"id": "cmd-abc123"}`

**Important:** The `command` field is a JSON string. Escape quotes, newlines,
and special characters properly. For multi-line code, use `\n` or pass via
a file/heredoc.

### Poll command status

    curl -s \
      -H "Authorization: Bearer $(databricks-token)" \
      "${DATABRICKS_HOST}/api/1.2/commands/status?clusterId={CLUSTER_ID}&contextId={CONTEXT_ID}&commandId={COMMAND_ID}"

Response when running:
    {"id": "cmd-abc123", "status": "Running"}

Response when finished:
    {
      "id": "cmd-abc123",
      "status": "Finished",
      "results": {
        "resultType": "text",
        "data": "hello world\n"
      }
    }

Response on error:
    {
      "id": "cmd-abc123",
      "status": "Error",
      "results": {
        "resultType": "error",
        "cause": "NameError: name 'undefined_var' is not defined",
        "summary": "NameError: ..."
      }
    }

Poll every 5 seconds. Status values: Queued → Running → Finished/Error/Cancelled.

### Destroy execution context

    curl -s -X POST \
      -H "Authorization: Bearer $(databricks-token)" \
      -H "Content-Type: application/json" \
      "${DATABRICKS_HOST}/api/1.2/contexts/destroy" \
      -d '{"clusterId": "{CLUSTER_ID}", "contextId": "{CONTEXT_ID}"}'

## SQL Statement API (for profiling)

### Execute SQL statement

    curl -s -X POST \
      -H "Authorization: Bearer $(databricks-token)" \
      -H "Content-Type: application/json" \
      "${DATABRICKS_HOST}/api/2.0/sql/statements/" \
      -d '{
        "warehouse_id": "148ccb90800933a1",
        "statement": "SELECT COUNT(*) as cnt FROM {table}",
        "wait_timeout": "30s",
        "disposition": "INLINE",
        "format": "JSON_ARRAY"
      }'

Response (inline):
    {
      "status": {"state": "SUCCEEDED"},
      "result": {
        "data_array": [["10000000"]],
        "row_count": 1
      },
      "manifest": {
        "columns": [{"name": "cnt", "type_name": "BIGINT"}]
      }
    }

If `state` is `PENDING`, poll:
    GET /api/2.0/sql/statements/{statement_id}

## Workspace Import (upload notebook)

### Import/overwrite notebook

    curl -s -X POST \
      -H "Authorization: Bearer $(databricks-token)" \
      -H "Content-Type: application/json" \
      "${DATABRICKS_HOST}/api/2.0/workspace/import" \
      -d '{
        "path": "/Users/{user_email}/autoresearch/{notebook_name}",
        "format": "SOURCE",
        "language": "PYTHON",
        "content": "{base64_encoded_notebook_content}",
        "overwrite": true
      }'

The `content` field must be base64-encoded. Generate it:
    base64 -w0 /path/to/notebook.py

## MLflow API (read metrics)

### Get run metrics

    curl -s \
      -H "Authorization: Bearer $(databricks-token)" \
      "${DATABRICKS_HOST}/api/2.0/mlflow/runs/get?run_id={RUN_ID}" \
      | jq '.run.data.metrics'

Use this to read metric values (source of truth) rather than parsing cell output.
```

**Step 2: Commit**

```bash
git add .claude/skills/autoresearch-xgb/api-reference.md
git commit -m "feat(autoresearch): add API reference for REST interactions"
```

---

## Task 3: Notebook format — how to build the .py notebook incrementally

**Files:**
- Create: `.claude/skills/autoresearch-xgb/notebook-format.md`

**Step 1: Write the notebook format reference**

```markdown
# Notebook Format Reference

The autoresearch agent builds a Databricks `.py` source notebook incrementally
as it works. The notebook is stored locally during the session and uploaded to
the workspace periodically (after each phase) and at the end.

## Databricks .py Source Format

The first line MUST be:
    # Databricks notebook source

Each cell is separated by:
    # COMMAND ----------

Markdown cells use the `# MAGIC %md` prefix:
    # MAGIC %md
    # MAGIC ## Section Title
    # MAGIC
    # MAGIC Some explanation text here.

Pip install cells:
    # MAGIC %pip install -U mlflow

Python restart:
    # MAGIC %restart_python

## Building the Notebook

The agent maintains a local file at:
    /workspace/group/scaling_xgb_work/notebooks/autoresearch/{table_name}_{date}.py

To add a cell, append to the file:

For a markdown cell:
    echo '' >> notebook.py
    echo '# COMMAND ----------' >> notebook.py
    echo '' >> notebook.py
    echo '# MAGIC %md' >> notebook.py
    echo '# MAGIC ## Section Title' >> notebook.py
    echo '# MAGIC' >> notebook.py
    echo '# MAGIC Explanation here.' >> notebook.py

For a code cell:
    echo '' >> notebook.py
    echo '# COMMAND ----------' >> notebook.py
    echo '' >> notebook.py
    echo 'import pandas as pd' >> notebook.py
    echo 'df = spark.table("my_table").toPandas()' >> notebook.py

**Important:** In practice, the agent should use the Write or Edit tools to
build the notebook content, not echo commands. The above is just to illustrate
the format.

## Notebook Skeleton

The agent creates the skeleton in Phase 2 (while the cluster spins up):

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
    # MAGIC %restart_python

    # COMMAND ----------

    # Imports and setup — filled in during Phase 3

Then each phase appends its cells to this file.

## Uploading to Workspace

After each phase and at the end, upload via the Workspace Import API:

    base64 -w0 notebooks/autoresearch/{notebook_file} > /tmp/nb_b64.txt
    # Then POST to /api/2.0/workspace/import with content from /tmp/nb_b64.txt

Upload path: `/Users/{user_email}/autoresearch/{table_name}_{date}`

Set `"overwrite": true` to update the existing notebook.

## Summary Cell (Phase 5)

After all experiments, the agent prepends (or updates) the summary cell at
the top of the notebook with:

- Best model metrics (AUC-PR, AUC-ROC, F1)
- Track chosen and why
- Feature engineering decisions
- Experiments run and results
- Total time and cluster config
- MLflow run ID and model registry name
```

**Step 2: Commit**

```bash
git add .claude/skills/autoresearch-xgb/notebook-format.md
git commit -m "feat(autoresearch): add notebook format reference"
```

---

## Task 4: Phase 1 recipe — PROFILE

**Files:**
- Create: `.claude/skills/autoresearch-xgb/phase-1-profile.md`

**Step 1: Write the profile phase recipe**

```markdown
# Phase 1: PROFILE

Profile the dataset using the SQL Statement API. No cluster needed — this
runs against a SQL warehouse.

## Prerequisites

- `table`: full UC table name (e.g. `brian_gen_ai.xgb_scaling.imbalanced_10m`)
- `target_col`: target column name
- `warehouse_id`: SQL warehouse ID (default: `148ccb90800933a1`)

## Step 1: Get row count and column list

Run this SQL via the SQL Statement API (see api-reference.md):

    SELECT COUNT(*) as row_count FROM {table}

Then:

    DESCRIBE TABLE {table}

Parse the response to get column names and types. Classify each column:
- `IntegerType`, `LongType`, `FloatType`, `DoubleType` → numeric
- `StringType` → categorical
- `BooleanType` → boolean
- `TimestampType`, `DateType` → timestamp (drop for v1)

## Step 2: Get null percentages

    SELECT
      {for each column: SUM(CASE WHEN col IS NULL THEN 1 ELSE 0 END) / COUNT(*) as null_pct_col}
    FROM {table}

For tables with many columns, batch this into multiple queries (max ~50 columns
per query to avoid SQL length limits).

## Step 3: Get categorical cardinality

For each categorical column:

    SELECT COUNT(DISTINCT {cat_col}) as cardinality FROM {table}

Or batched:

    SELECT
      COUNT(DISTINCT cat_col_1) as card_1,
      COUNT(DISTINCT cat_col_2) as card_2,
      ...
    FROM {table}

## Step 4: Get class distribution

    SELECT {target_col}, COUNT(*) as cnt
    FROM {table}
    GROUP BY {target_col}
    ORDER BY {target_col}

## Step 5: Get numeric stats

    SELECT
      MIN({num_col}) as min_val,
      MAX({num_col}) as max_val,
      AVG({num_col}) as mean_val,
      STDDEV({num_col}) as std_val
    FROM {table}

Batch across numeric columns. For skewness (used in Phase 3 to decide
log transforms):

    SELECT
      (SUM(POWER({num_col} - avg_val, 3)) / (COUNT(*) * POWER(std_val, 3))) as skewness
    FROM {table}
    CROSS JOIN (SELECT AVG({num_col}) as avg_val, STDDEV({num_col}) as std_val FROM {table})

**Simplification for v1:** Skip skewness in profiling. Compute it in Phase 3
on the cluster after loading data. This avoids complex SQL.

## Step 6: Get sample rows

    SELECT * FROM {table} LIMIT 5

Useful for inspecting column names and values.

## Output

Collect all results into a profile summary dict:

    profile = {
        "table": table,
        "target_col": target_col,
        "row_count": 10_000_000,
        "columns": [
            {"name": "feat_0", "type": "numeric", "null_pct": 0.0},
            {"name": "cat_0", "type": "categorical", "null_pct": 0.02, "cardinality": 12},
            ...
        ],
        "numeric_count": 200,
        "categorical_count": 50,
        "boolean_count": 0,
        "class_distribution": {0: 9_500_000, 1: 500_000},
        "sample_rows": [...],
    }

This profile drives Phase 2 (cluster sizing) and Phase 3 (feature decisions).

## Notebook Cells

Add to the notebook:

1. Markdown cell: "## 1. Data Profile" with a table summarizing the profile
2. Code cell: the profiling queries (so the notebook is reproducible)
3. Markdown cell: auto-generated summary of findings
```

**Step 2: Commit**

```bash
git add .claude/skills/autoresearch-xgb/phase-1-profile.md
git commit -m "feat(autoresearch): add Phase 1 PROFILE recipe"
```

---

## Task 5: Phase 2 recipe — SIZE & CREATE CLUSTER

**Files:**
- Create: `.claude/skills/autoresearch-xgb/phase-2-cluster.md`

**Step 1: Write the cluster sizing and creation recipe**

```markdown
# Phase 2: SIZE & CREATE CLUSTER

Use the profile from Phase 1 to calculate the right cluster size, create it,
and open an execution context.

## Step 1: Estimate memory

    raw_data_gb = row_count * numeric_count * 8 / 1e9
    xgb_overhead_gb = raw_data_gb * 3
    feature_headroom_gb = xgb_overhead_gb * 1.3
    spark_overhead_gb = 4
    total_needed_gb = feature_headroom_gb + spark_overhead_gb

## Step 2: Pick cluster config

| total_needed_gb | Track | node_type_id | num_workers | RAM |
|-----------------|-------|--------------|-------------|-----|
| < 20 | single-node CPU | Standard_D16s_v5 | 0 | 64 GB |
| 20–50 | single-node high-mem | Standard_E32s_v5 | 0 | 256 GB |
| 50–150 | Ray distributed (small) | Standard_D16s_v5 | 2 | 192 GB total |
| 150–400 | Ray distributed (medium) | Standard_D16s_v5 | 4 | 320 GB total |
| > 400 | Ray distributed (large) | Standard_D16s_v5 | 8 | 576 GB total |

For Ray: usable RAM per node ≈ 40 GB (64 GB minus Spark/Ray/OS overhead).
N workers = ceil(total_needed_gb / 40).

**Budget constraint:** If budget_minutes < 30 and total_needed_gb is 20-50,
prefer Standard_E32s_v5 single-node over Ray distributed (less setup overhead).

Record the chosen track as `track` variable: `"single-node-cpu"`,
`"single-node-highmem"`, or `"ray-distributed"`.

## Step 3: Create cluster

Use the cluster create API from api-reference.md with the chosen config.

For single-node clusters:
- `spark_version`: `"17.3.x-cpu-ml-scala2.13"`
- `num_workers`: 0
- `spark_conf`: singleNode profile + `local[*, 4]`
- `autotermination_minutes`: budget_minutes + 10

For Ray distributed:
- `spark_version`: `"17.3.x-cpu-ml-scala2.13"`
- `num_workers`: N (from table above)
- `spark_conf`: `spark.executorEnv.OMP_NUM_THREADS` = vCPUs - 1
- `autotermination_minutes`: budget_minutes + 10

Name: `autoresearch-{table_name}-{YYYYMMDD-HHMM}`

## Step 4: Wait for cluster RUNNING

Poll `GET /api/2.0/clusters/get` every 15 seconds.

Timeout: 10 minutes. If not RUNNING after 10 min, report error and exit.

## Step 5: Create execution context

    POST /api/1.2/contexts/create
    {"clusterId": "{cluster_id}", "language": "python"}

Save the `context_id` for all subsequent cell executions.

## Step 6: While waiting — scaffold notebook

While the cluster is spinning up (Steps 3-4), build the notebook skeleton
locally. See notebook-format.md for the skeleton structure.

Also pre-compute feature engineering decisions from the profile:
- Which categoricals to encode (cardinality < 50)
- Which to drop (cardinality >= 50)
- Which columns have high nulls (> 50% → drop)
- scale_pos_weight from class distribution

## Step 7: Record budget start time

    budget_start = time.time()

This is used by the budget guard in Phase 4.

## Notebook Cells

Add to the notebook:

1. Markdown cell: "## 2. Cluster Configuration" explaining the sizing decision
2. Markdown cell: table showing estimated memory, chosen track, node type, workers
```

**Step 2: Commit**

```bash
git add .claude/skills/autoresearch-xgb/phase-2-cluster.md
git commit -m "feat(autoresearch): add Phase 2 SIZE & CREATE recipe"
```

---

## Task 6: Phase 3 recipe — FEATURE ENGINEERING

**Files:**
- Create: `.claude/skills/autoresearch-xgb/phase-3-features.md`

**Step 1: Write the feature engineering recipe**

This is the first phase that runs cells on the live cluster. Each step creates
a code cell, sends it via Command Execution API, reads the output, then adds
a markdown explanation cell.

```markdown
# Phase 3: FEATURE ENGINEERING

Run interactive cells on the live cluster to prepare features for training.

**Prerequisites:** cluster_id, context_id from Phase 2. Profile from Phase 1.

## How to run a cell

For every code cell in this phase:

1. Build the Python code as a string
2. Send via Command Execution API (see api-reference.md)
3. Poll until Finished or Error
4. Read the output from `results.data` (for text) or `results.cause` (for errors)
5. If Error: try to fix the code and retry once. If still fails, log error in
   a markdown cell and move on.
6. Append the code cell and a markdown cell to the local notebook file

## Cell 1: Imports and setup

Run on cluster:

    import os, time, psutil
    import pandas as pd
    import numpy as np
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    user_email = spark.sql("SELECT current_user()").collect()[0][0]
    experiment_path = f"/Users/{user_email}/autoresearch"
    mlflow.set_experiment(experiment_path)
    print(f"MLflow experiment: {experiment_path}")
    print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")

Read the output. Record the available RAM for the memory check later.

## Cell 2: Load data

For single-node tracks:

    load_start = time.time()
    df = spark.table("{table}").toPandas()
    load_time = time.time() - load_start
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols in {load_time:.1f}s")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

For Ray distributed track, the data loading happens in Phase 4 (inside the
Ray training function). In Phase 3, just verify the table is accessible:

    row_count = spark.table("{table}").count()
    print(f"Table accessible: {row_count:,} rows")

**Runtime memory check:** After loading, parse the memory usage from output.
If usage > 80% of available RAM, add a warning markdown cell.

## Cell 3: Handle categoricals

For each categorical column identified in profiling:

    from pyspark.sql.types import StringType

    # Categoricals to encode (cardinality < 50): {list}
    # Categoricals to drop (cardinality >= 50): {list}

    # Drop high-cardinality
    df = df.drop(columns=[{high_card_cols}])

    # Ordinal-encode low-cardinality
    from sklearn.preprocessing import OrdinalEncoder
    cat_cols = [{low_card_cols}]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[cat_cols] = enc.fit_transform(df[cat_cols])
        print(f"Encoded {len(cat_cols)} categorical columns")

## Cell 4: Handle booleans and types

    bool_cols = [{bool_col_list}]
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Ensure all features are numeric
    feature_cols = [c for c in df.columns if c != "{target_col}"]
    print(f"Features: {len(feature_cols)}")

## Cell 5: Handle nulls

    null_pcts = df[feature_cols].isnull().mean()
    high_null_cols = null_pcts[null_pcts > 0.5].index.tolist()
    if high_null_cols:
        df = df.drop(columns=high_null_cols)
        feature_cols = [c for c in feature_cols if c not in high_null_cols]
        print(f"Dropped {len(high_null_cols)} high-null columns: {high_null_cols}")

    # Median impute remaining nulls
    null_remaining = df[feature_cols].isnull().sum().sum()
    if null_remaining > 0:
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        print(f"Imputed {null_remaining} remaining nulls with median")

## Cell 6: Feature count guardrail

    if len(feature_cols) > 1000:
        variances = df[feature_cols].var().sort_values()
        drop_n = len(feature_cols) - 1000
        low_var_cols = variances.head(drop_n).index.tolist()
        df = df.drop(columns=low_var_cols)
        feature_cols = [c for c in feature_cols if c not in low_var_cols]
        print(f"Dropped {drop_n} low-variance features (cap at 1000)")

    print(f"Final feature count: {len(feature_cols)}")

## Cell 7: Class balance and train/test split

    from sklearn.model_selection import train_test_split

    X = df[feature_cols]
    y = df["{target_col}"]

    class_counts = y.value_counts().sort_index()
    scale_pos_weight = class_counts.iloc[0] / class_counts.iloc[1]
    print(f"Class distribution: {dict(class_counts)}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

## Checkpoint: Upload notebook

After all Phase 3 cells complete, upload the notebook-so-far to the workspace.
This is a crash recovery checkpoint — if the cluster dies in Phase 4, we still
have profiling + feature eng.
```

**Step 2: Commit**

```bash
git add .claude/skills/autoresearch-xgb/phase-3-features.md
git commit -m "feat(autoresearch): add Phase 3 FEATURE ENG recipe"
```

---

## Task 7: Phase 4 recipe — TRAIN & EVALUATE

**Files:**
- Create: `.claude/skills/autoresearch-xgb/phase-4-train.md`

**Step 1: Write the training recipe**

```markdown
# Phase 4: TRAIN & EVALUATE

Run XGBoost training experiments on the live cluster. Start with a baseline,
then try variations if budget allows.

## Budget Guard

Before each experiment, check:

    elapsed_minutes = (time.time() - budget_start) / 60
    remaining = budget_minutes - elapsed_minutes
    phase5_reserve = budget_minutes * 0.2  # 20% for finalize

    if remaining < phase5_reserve:
        print("Budget guard: skipping remaining experiments")
        # Go to Phase 5
        break

## Baseline Experiment (always runs)

### Cell: Baseline training

For single-node:

    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix
    from mlflow.models import infer_signature

    xgb_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": os.cpu_count(),
        "random_state": 42,
    }

    with mlflow.start_run(run_name="baseline") as run:
        mlflow.log_params({
            "experiment": "baseline",
            "table": "{table}",
            "n_rows": len(X_train) + len(X_test),
            "n_features": len(feature_cols),
            **{f"xgb_{k}": v for k, v in xgb_params.items()},
        })

        train_start = time.time()
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        mlflow.log_metric("train_time_sec", train_time)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics = {
            "auc_pr": average_precision_score(y_test, y_proba),
            "auc_roc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_metric("total_time_sec", load_time + train_time)

        sig = infer_signature(X_test.head(100), model.predict_proba(X_test.head(100)))
        mlflow.sklearn.log_model(model, "model", signature=sig)

        baseline_run_id = run.info.run_id
        print(f"Baseline: AUC-PR={metrics['auc_pr']:.4f}, train={train_time:.1f}s")

After the cell runs, read metrics from MLflow API (source of truth):

    curl -s -H "Authorization: Bearer $(databricks-token)" \
      "${DATABRICKS_HOST}/api/2.0/mlflow/runs/get?run_id={baseline_run_id}" \
      | jq '.run.data.metrics[] | select(.key == "auc_pr") | .value'

Record: `best_auc_pr = metrics["auc_pr"]`, `best_run_id = baseline_run_id`,
`experiments_without_improvement = 0`.

## Experiment 2: Shallower trees (if budget allows)

Check budget guard. Then run:

    Same as baseline but: "max_depth": 4, run_name="exp2_depth4"

After run, compare AUC-PR to best. If improved, update best. If not,
increment `experiments_without_improvement`.

## Experiment 3: Deeper trees + more rounds (if budget allows)

Check budget guard. Check progress guard (stop if experiments_without_improvement >= 2).

    Same as baseline but: "max_depth": 8, "n_estimators": 200, run_name="exp3_depth8_200r"

## Experiment 4: Feature importance pruning (if budget allows)

Check budget guard. Check progress guard.

    # Get feature importances from baseline model
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance")

    # Drop bottom 20%
    n_drop = int(len(feature_cols) * 0.2)
    drop_cols = importance_df.head(n_drop)["feature"].tolist()
    X_train_pruned = X_train.drop(columns=drop_cols)
    X_test_pruned = X_test.drop(columns=drop_cols)

    # Retrain with pruned features, run_name="exp4_pruned"

## After all experiments

### Cell: Results comparison

    import mlflow
    experiment = mlflow.get_experiment_by_name(experiment_path)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    print(runs[["run_id", "params.experiment", "metrics.auc_pr",
                 "metrics.auc_roc", "metrics.train_time_sec"]].to_string())

Add a markdown cell to the notebook with a comparison table.

### Checkpoint: Upload notebook

Upload notebook to workspace after Phase 4.

## For Ray Distributed Track

The training is different — use the Ray `DataParallelTrainer` pattern from
the existing `track-ray-distributed.md` skill. Key differences:

- Data is loaded via `ray.data.read_databricks_tables()` instead of pandas
- Training uses `DataParallelTrainer` with `XGBoostConfig`
- OMP 3-layer fix must be applied
- Model extraction from Ray checkpoint

Refer to `.claude/skills/train-xgb-databricks/track-ray-distributed.md` for
the exact cell code when the chosen track is `ray-distributed`.
```

**Step 2: Commit**

```bash
git add .claude/skills/autoresearch-xgb/phase-4-train.md
git commit -m "feat(autoresearch): add Phase 4 TRAIN recipe"
```

---

## Task 8: Phase 5 recipe — FINALIZE

**Files:**
- Create: `.claude/skills/autoresearch-xgb/phase-5-finalize.md`

**Step 1: Write the finalize recipe**

```markdown
# Phase 5: FINALIZE

Compare results, register the best model, finalize the notebook, and clean up.

## Step 1: Pick the winner

From the experiments run in Phase 4, identify the run with the highest AUC-PR.
Use the MLflow API to get the definitive metric values.

## Step 2: Register best model to Unity Catalog

Run on cluster:

    import mlflow
    from mlflow.models import infer_signature

    best_run_id = "{best_run_id}"
    uc_model_name = "{catalog}.{schema}.autoresearch_{table_name}"

    # The model was already logged in Phase 4.
    # Register it to UC:
    mlflow.register_model(
        f"runs:/{best_run_id}/model",
        uc_model_name,
    )
    print(f"Registered: {uc_model_name}")

## Step 3: Update notebook summary

Edit the top of the local notebook file to replace the placeholder summary
with actual results:

    # MAGIC %md
    # MAGIC # AutoResearch: {table_name}
    # MAGIC
    # MAGIC **Generated:** {date}
    # MAGIC **Table:** `{catalog}.{schema}.{table_name}`
    # MAGIC **Target:** `{target_col}`
    # MAGIC
    # MAGIC ## Results
    # MAGIC
    # MAGIC | Experiment | AUC-PR | AUC-ROC | F1 | Train Time |
    # MAGIC |------------|--------|---------|-----|------------|
    # MAGIC | Baseline   | 0.xxxx | 0.xxxx  | 0.xx | xx.xs    |
    # MAGIC | Exp 2      | ...    | ...     | ... | ...        |
    # MAGIC
    # MAGIC **Best model:** {best_experiment} (AUC-PR: {best_auc_pr:.4f})
    # MAGIC **Registered as:** `{uc_model_name}`
    # MAGIC **MLflow run:** {best_run_id}
    # MAGIC
    # MAGIC ## Decisions Made
    # MAGIC - **Track:** {track} ({reason})
    # MAGIC - **Cluster:** {node_type} x {num_workers} workers
    # MAGIC - **Features:** {n_features} (dropped {n_dropped_high_card} high-cardinality, {n_dropped_null} high-null)
    # MAGIC - **Encoded:** {n_encoded} categorical columns (ordinal)
    # MAGIC - **Total time:** {total_minutes:.1f} minutes

## Step 4: Upload final notebook

Upload the final notebook to workspace:

    /Users/{user_email}/autoresearch/{table_name}_{YYYYMMDD}

Use `overwrite: true`.

## Step 5: Terminate cluster

    POST /api/2.0/clusters/delete
    {"cluster_id": "{cluster_id}"}

## Step 6: Report to user

Send a message summarizing:

- Best model metrics
- Notebook location (workspace path)
- MLflow experiment link
- UC model registry name
- Total time and cost estimate
- Key decisions made (track, features, experiments tried)
```

**Step 2: Commit**

```bash
git add .claude/skills/autoresearch-xgb/phase-5-finalize.md
git commit -m "feat(autoresearch): add Phase 5 FINALIZE recipe"
```

---

## Task 9: Integration test — dry run the full skill on a small dataset

**Files:**
- All files from Tasks 1-8

**Step 1: Verify skill is discoverable**

Check that Claude Code can find the skill:

    ls -la .claude/skills/autoresearch-xgb/

Verify all files exist:
- SKILL.md
- api-reference.md
- notebook-format.md
- phase-1-profile.md
- phase-2-cluster.md
- phase-3-features.md
- phase-4-train.md
- phase-5-finalize.md

**Step 2: Test Phase 1 (profiling) against a real table**

Run the SQL Statement API calls from phase-1-profile.md against the
`brian_gen_ai.xgb_scaling.imbalanced_10k` table (tiny dataset).
Verify you get valid row counts, column types, and class distribution.

**Step 3: Test cluster creation**

Create a small test cluster (Standard_D16s_v5, 0 workers) via the API.
Verify it reaches RUNNING state. Then:
- Create an execution context
- Execute `print("hello from autoresearch")`
- Verify output comes back as `"hello from autoresearch"`
- Destroy context and terminate cluster

**Step 4: Commit test results**

Document what worked and any issues in the commit message.

```bash
git add -A
git commit -m "feat(autoresearch): complete skill files and integration test"
```

---

## Task 10: Push branch and open PR

**Step 1: Rebase on latest main**

```bash
git fetch origin
git rebase origin/main
```

**Step 2: Push and create PR**

```bash
git push -u origin HEAD
gh pr create --fill
```

**Step 3: Verify checks**

```bash
gh pr checks
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Skill skeleton | `.claude/skills/autoresearch-xgb/SKILL.md` |
| 2 | API reference | `.claude/skills/autoresearch-xgb/api-reference.md` |
| 3 | Notebook format | `.claude/skills/autoresearch-xgb/notebook-format.md` |
| 4 | Phase 1: Profile | `.claude/skills/autoresearch-xgb/phase-1-profile.md` |
| 5 | Phase 2: Cluster | `.claude/skills/autoresearch-xgb/phase-2-cluster.md` |
| 6 | Phase 3: Features | `.claude/skills/autoresearch-xgb/phase-3-features.md` |
| 7 | Phase 4: Train | `.claude/skills/autoresearch-xgb/phase-4-train.md` |
| 8 | Phase 5: Finalize | `.claude/skills/autoresearch-xgb/phase-5-finalize.md` |
| 9 | Integration test | Dry run against real Databricks |
| 10 | Push + PR | Branch + CI |

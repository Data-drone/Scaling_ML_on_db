# AutoResearch Agent Design — XGBoost on Databricks

**Date:** 2026-03-19
**Status:** Draft — pending implementation plan

## Overview

An autonomous research agent that takes a Unity Catalog table, a target column,
and an optional time budget, then produces a polished, well-documented Databricks
notebook containing the best XGBoost model it can find.

The agent is implemented as a Claude Code skill. It orchestrates everything via
Databricks REST APIs — creates a right-sized cluster, runs code interactively,
builds the notebook incrementally, and tears down when done.

## Entry Point

The user says something like:

> Train the best XGBoost model on brian_gen_ai.xgb_scaling.imbalanced_10m,
> target column is "label", budget 45 minutes.

The skill extracts:

| Input | Required | Example | Default |
|-------|----------|---------|---------|
| `table` | yes | `brian_gen_ai.xgb_scaling.imbalanced_10m` | — |
| `target_col` | yes | `label` | — |
| `budget_minutes` | no | `45` | `60` |

Derived values: `catalog`, `schema`, `table_name` parsed from the three-part
table identifier. `warehouse_id` from project config or default.

## Architecture

```
User request
    │
    ▼
Claude Code (skill: autoresearch-xgb)
    │
    ├─ Phase 1: PROFILE ──────► SQL Warehouse (lightweight query)
    │
    ├─ Phase 2: SIZE & CREATE ─► Databricks Clusters API
    │                              │
    │                              ▼
    │                           Running cluster + execution context
    │                              │
    ├─ Phase 3: FEATURE ENG ───► Command Execution API (interactive cells)
    │                              │
    ├─ Phase 4: TRAIN ─────────► Command Execution API (interactive cells)
    │                              │
    ├─ Phase 5: FINALIZE ──────► Workspace Import API (upload notebook)
    │                              │
    └─ Teardown ───────────────► Clusters API (terminate)
```

All REST calls go through `databricks api post` CLI or
`curl -H "Authorization: Bearer $(databricks-token)"`.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cluster lifecycle | Agent-managed (create/destroy) | Right-sized for the data; no wasted spend on idle clusters |
| Code execution | Command Execution API (1.2) | Tightest feedback loop — run cell, read output, decide next |
| Artifact format | Databricks .py source notebook | Built incrementally; uploaded to workspace at end and after each phase |
| Autonomy | Fully autonomous with budget guard | Agent runs end-to-end, reports results when done |
| Feature engineering | Automated basics + light experimentation | Ordinal encode, null handling, skew detection; no deep AutoML |
| Notebook style | Well-structured with markdown explanations | Each section has rationale, not just code |

## The Five Phases

### Phase 1 — PROFILE

Run a lightweight SQL query via the SQL warehouse (no cluster needed) to collect:

- Row count, column count
- Column types (numeric, categorical, boolean, timestamp)
- Null percentages per column
- Cardinality of each categorical column
- Class distribution of target column
- Basic stats on numeric columns (min, max, mean, stddev, skewness)
- Sample of 5 rows (for column name inspection)

This profile drives all subsequent decisions. It becomes the first section of the
notebook (markdown summary + profiling code).

### Phase 2 — SIZE & CREATE CLUSTER

**Memory estimation:**

```
raw_data_gb = rows × numeric_features × 8 / 1e9
xgb_overhead = raw_data_gb × 3        # gradient/hessian buffers, tree structures
feature_headroom = xgb_overhead × 1.3  # 30% headroom for new features from encoding
spark_overhead_gb = 4                   # Spark driver/executor JVM overhead
total_needed_gb = feature_headroom + spark_overhead_gb
```

**Cluster decision table:**

| total_needed_gb | Track | Cluster Config |
|-----------------|-------|----------------|
| < 20 GB | Single-node CPU | Standard_D16s_v5 (64 GB RAM) |
| 20–50 GB | Single-node high-mem | Standard_E32s_v5 (256 GB RAM) |
| > 50 GB | Ray distributed | Standard_D16s_v5 × N workers |

For Ray distributed: `N = ceil(total_needed_gb / 40)` (each D16s_v5 has 64 GB,
use ~60% for data).

**Budget constraint:** if budget < 30 min and data is large, prefer a bigger
single node over distributed (less setup overhead).

**Actions:**
1. `POST /api/2.0/clusters/create` with the chosen config
2. Poll `GET /api/2.0/clusters/get` until state = RUNNING
3. `POST /api/1.2/contexts/create` with language = python
4. While cluster spins up (3-5 min): scaffold notebook skeleton locally,
   pre-compute feature engineering decisions from the profile

**Cluster settings:**
- `autotermination_minutes`: budget_minutes + 10 (safety net)
- `data_security_mode`: SINGLE_USER
- `azure_attributes.availability`: SPOT_WITH_FALLBACK_AZURE (not pure SPOT)

### Phase 3 — FEATURE ENGINEERING

Run cells on the live cluster via Command Execution API. Each step is a code cell
with a markdown explanation above it.

Steps:
1. Load data via `spark.table()`, convert to pandas (single-node) or keep as
   Spark/Ray dataset (distributed)
2. **Categoricals:** ordinal-encode if cardinality < 50, drop if >= 50
3. **Booleans:** cast to int
4. **Nulls:** drop columns with > 50% null, median-impute the rest
5. **Skewed numerics:** log1p transform if skewness > 2.0 (adds new column,
   keeps original)
6. **Feature count guardrail:** if total features > 1000, drop lowest-variance
   columns until under 1000
7. **Imbalance:** calculate `scale_pos_weight = count(neg) / count(pos)`
8. **Train/test split:** 80/20 stratified, `random_state=42`
9. **Runtime memory check:** after data is loaded and transformed, check
   `psutil.virtual_memory()`. If usage > 80% of available RAM, log a warning
   in the notebook and consider switching to distributed (v2: auto-resize)

**Periodic save:** after Phase 3 completes, upload notebook-so-far to workspace
via `/api/2.0/workspace/import`. This is crash recovery — if the cluster dies
in Phase 4, we still have profiling + feature eng.

### Phase 4 — TRAIN & EVALUATE

**Baseline run:**
```python
xgb_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
}
```
- Train, log to MLflow, evaluate: AUC-ROC, AUC-PR, F1, confusion matrix
- Read metrics from MLflow API (source of truth, not cell output parsing)

**Experiments (if budget allows):**
- Experiment 2: `max_depth=4` (shallower trees)
- Experiment 3: `max_depth=8, n_estimators=200` (deeper, more trees)
- Experiment 4: drop low-importance features (bottom 20% by gain from baseline)

**Guards:**
- Before each experiment: check `elapsed < budget * 0.8` (reserve 20% for
  Phase 5)
- **Progress guard:** if the best metric (AUC-PR) hasn't improved after 2
  consecutive experiments, stop early
- **Hard cutoff:** if `elapsed > budget_minutes`, terminate experiments
  immediately, proceed to Phase 5 with best result so far

**Periodic save:** upload notebook to workspace after each training run completes.

### Phase 5 — FINALIZE

1. Compare all experiment results (table in notebook)
2. Pick the winner (highest AUC-PR)
3. Register best model to Unity Catalog via MLflow:
   ```python
   mlflow.xgboost.log_model(
       xgb_model=best_booster,
       artifact_path="model",
       signature=infer_signature(X_test, predictions),
       registered_model_name=f"{catalog}.{schema}.autoresearch_{table_name}",
   )
   ```
4. Add summary markdown cell at the top of the notebook:
   - Table name, row count, feature count
   - Track chosen (CPU/GPU/Ray) and why
   - Feature engineering decisions made
   - Best model metrics
   - Total time, cluster config
5. Upload final notebook to workspace:
   `/Users/{user}/autoresearch/{table_name}_{date}.py`
6. Terminate cluster via `POST /api/2.0/clusters/delete`
7. Report results to user (chat message with key metrics + notebook link)

## Notebook Format

The output is a Databricks `.py` source notebook:

```python
# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # AutoResearch: {table_name}
# MAGIC
# MAGIC **Generated:** {date}
# MAGIC **Best Model:** AUC-PR {value}, AUC-ROC {value}
# MAGIC **Track:** {single-node-cpu | single-node-gpu | ray-distributed}
# MAGIC **Cluster:** {node_type} × {workers}
# MAGIC **Training Time:** {time}
# MAGIC
# MAGIC ## Summary of Decisions
# MAGIC - Dropped {n} high-cardinality categoricals
# MAGIC - Log-transformed {n} skewed features
# MAGIC - ...

# COMMAND ----------

# Section 2: Environment Setup
# ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Profile
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Rows   | 10,000,000 |
# MAGIC | ...    | ... |

# COMMAND ----------

# profiling code here
# ...
```

Key principle: markdown cells are written AFTER code cells execute, so they
describe actual outcomes, not just intentions.

## API Interaction Pattern

### Core loop (for each cell)

```
1. Build Python code string for the cell
2. POST /api/1.2/commands/execute
   Body: {"clusterId": cluster_id, "contextId": ctx_id,
          "language": "python", "command": code}
   → command_id
3. Poll GET /api/1.2/commands/status
   Params: clusterId, contextId, commandId
   Until: status = "Finished" or "Error" or "Cancelled"
4. Read results:
   - results.resultType = "text" → cell output (print statements)
   - results.resultType = "error" → error trace
   - results.data → the output content
5. Agent reasons about output, decides next cell
6. Append code cell + markdown cell to local .py notebook file
```

### Cluster lifecycle

```
Create:  POST /api/2.0/clusters/create → cluster_id
Poll:    GET  /api/2.0/clusters/get?cluster_id=X (until RUNNING)
Context: POST /api/1.2/contexts/create → context_id
...cells...
Upload:  POST /api/2.0/workspace/import (notebook .py file)
Destroy: POST /api/1.2/contexts/destroy
         POST /api/2.0/clusters/delete
```

## Error Handling

| Scenario | Response |
|----------|----------|
| Cluster fails to create | Report error to user, exit |
| Cell returns error | Read error, attempt one fix, if still fails: log error in markdown, move on |
| Cluster terminates unexpectedly (spot preemption) | Save local notebook state, upload partial notebook, report to user |
| OOM during training | Log error, try with fewer features or smaller sample, move on |
| Budget exceeded | Stop experiments, proceed to Phase 5 with best result |
| No metric improvement after 2 experiments | Stop experiments early, proceed to Phase 5 |
| Feature count > 1000 after encoding | Drop lowest-variance columns until under 1000 |
| Data load uses > 80% RAM | Log warning in notebook, continue (v2: auto-resize) |

All errors are documented in the notebook as markdown cells (transparency).

## What This Does NOT Do (v1)

- No hyperparameter search (just a few manual variations)
- No parallel experiments (sequential on cluster)
- No deep AutoML feature search (basic transforms only)
- No LLM-driven domain reasoning about column semantics
- No automatic cluster resizing mid-session
- No GPU support (CPU and Ray distributed only)
- No external memory mode (ExtMemQuantileDMatrix)
- No governance isolation (uses caller's permissions)

## Future Enhancements (v2+)

- **Parallel experiments** on Ray clusters (partition workers across trials)
- **GPU track** with automatic VRAM sizing
- **LLM-driven feature reasoning** (interpret column names, suggest transforms)
- **External memory mode** for datasets that don't fit in RAM
- **MLflow Tracing** for agent decision audit trail
- **Service Principal isolation** for production use
- **Auto-resume** from partial notebooks after cluster failure
- **Helper function library** (Approach 3 from brainstorming) for structured
  agent-cluster interaction

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Profiling via SQL warehouse or training cluster? | SQL warehouse (Phase 1 runs before cluster exists) |
| GPU support in v1? | No — CPU and Ray only for v1 |
| Notebook naming? | `/Users/{user}/autoresearch/{table_name}_{date}.py` |
| Results to Discord? | Chat only for v1 |
| 3x overhead multiplier correct? | Roughly — add runtime memory check as safety net |
| Tune num_workers or fix? | Fix based on data size for v1 |

## External Review

Reviewed by Gemini (2026-03-19). Key feedback incorporated:

1. **Periodic notebook save** — upload to workspace after each phase (crash recovery)
2. **Progress guard** — stop if no metric improvement after 2 experiments
3. **Metrics from MLflow API** — source of truth, not cell output parsing
4. **Runtime memory check** — verify actual usage after data load
5. **Feature count cap** — max 1000 features guardrail
6. **SPOT_WITH_FALLBACK** — explicit in cluster config (not pure SPOT)

Feedback deferred to v2: governance isolation, MLflow Tracing, ExtMemQuantileDMatrix,
Databricks Apps UI, Tool Library approach.

# Train XGBoost Skill — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Claude Code skill that guides an agent through training XGBoost on Databricks end-to-end — profiling data, preparing features, selecting the right approach, and generating a complete notebook.

**Architecture:** Decision-tree skill with progressive disclosure. SKILL.md contains the decision logic and links to track-specific recipe files that hold complete notebook code. The skill lives in `.claude/skills/train-xgb-databricks/` within the repo.

**Tech Stack:** Claude Code skills (markdown), Databricks, XGBoost, Ray on Spark, MLflow, Unity Catalog

---

### Task 1: Create skill directory and SKILL.md

**Files:**
- Create: `.claude/skills/train-xgb-databricks/SKILL.md`

**Step 1: Create the directory**

```bash
mkdir -p .claude/skills/train-xgb-databricks
```

**Step 2: Write SKILL.md**

Write the main skill file with:

```markdown
---
name: train-xgb-databricks
description: Trains XGBoost on a Databricks dataset end-to-end. Profiles the data, prepares features, selects training approach (single-node CPU, GPU, or Ray distributed) based on data size and hardware, generates a complete notebook with MLflow tracking. Use when asked to train XGBoost, fit gradient boosted trees, or build a classifier/regressor on Databricks.
---

# Train XGBoost on Databricks

End-to-end skill for training XGBoost on Databricks. Given a dataset and target column, this skill guides you through profiling, feature prep, approach selection, and notebook generation.

## Checklist

- [ ] Step 1: ASSESS — Profile the dataset
- [ ] Step 2: PREPARE — Feature engineering
- [ ] Step 3: SELECT TRACK — Pick training approach
- [ ] Step 4: GENERATE — Write the notebook
- [ ] Step 5: DEPLOY — (optional) DAB job definition

## Step 1: ASSESS — Profile the Dataset

Before writing any training code, profile the dataset:

```python
# Run in a Databricks notebook cell
df = spark.table("<catalog>.<schema>.<table>")
row_count = df.count()
col_count = len(df.columns)
print(f"Rows: {row_count:,}, Columns: {col_count}")

# Feature types
from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType, StringType
numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, LongType, FloatType, DoubleType))
                and f.name != "<target_column>"]
cat_cols = [f.name for f in df.schema.fields
            if isinstance(f.dataType, StringType)]
print(f"Numeric features: {len(numeric_cols)}, Categorical: {len(cat_cols)}")

# Memory estimate
estimated_gb = (row_count * len(numeric_cols) * 8) / 1e9
print(f"Estimated memory: {estimated_gb:.1f} GB")

# Class distribution (for classification)
df.groupBy("<target_column>").count().show()
```

Record: row_count, numeric feature count, categorical feature count, estimated_gb, class distribution.

## Step 2: PREPARE — Feature Engineering

See [feature-prep.md](feature-prep.md) for the complete feature engineering reference.

Quick summary:
- **Numeric features**: Use directly (XGBoost handles them natively)
- **Categorical features**: Ordinal-encode low-cardinality (<50 unique). Drop or hash high-cardinality.
- **Imbalanced targets**: Calculate `scale_pos_weight = count(negative) / count(positive)`
- **Train/test split**: Stratified 80/20 with `random_state=42`

## Step 3: SELECT TRACK — Pick Training Approach

Use this decision table:

| Row Count | Hardware | Track | Recipe |
|-----------|----------|-------|--------|
| ≤ 10M | CPU only | Single-node CPU | [track-single-node-cpu.md](track-single-node-cpu.md) |
| ≤ 10M | GPU available | Single-node GPU | [track-single-node-gpu.md](track-single-node-gpu.md) |
| 10M–100M | CPU only | Ray distributed | [track-ray-distributed.md](track-ray-distributed.md) |
| 10M–100M | GPU available | Single-node GPU* | [track-single-node-gpu.md](track-single-node-gpu.md) |
| > 100M | Any | Ray distributed | [track-ray-distributed.md](track-ray-distributed.md) |

*GPU path for 10M–100M only if estimated GPU memory (data_gb × 6) fits in GPU VRAM. Otherwise use Ray distributed.

**To check GPU availability:**
```python
import subprocess
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                         "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
if result.returncode == 0:
    print(f"GPU available: {result.stdout.strip()}")
else:
    print("No GPU — use CPU track")
```

## Step 4: GENERATE — Write the Notebook

Follow the track-specific recipe file selected in Step 3. Each recipe produces a complete Databricks notebook with:

- Environment validation gate (`src.validate_env`)
- Widget parameters (data_size, node_type, run_mode)
- Data loading from Unity Catalog
- Feature prep and train/test split
- XGBoost training with correct `tree_method`
- MLflow tracking (params, metrics, model registration, system metrics)
- Evaluation (AUC-PR, AUC-ROC, F1, confusion matrix)
- Structured JSON exit via `dbutils.notebook.exit()`

## Step 5: DEPLOY — DAB Job Definition (Optional)

If the user wants to deploy as a Databricks job, add a job definition to `databricks.yml`.

See [gotchas.md](gotchas.md) for Databricks-specific pitfalls to avoid.

### Cluster config by track:

**Single-node CPU:**
```yaml
job_clusters:
  - job_cluster_key: single_cpu
    new_cluster:
      spark_version: "17.3.x-cpu-ml-scala2.13"
      node_type_id: "Standard_D16s_v5"
      num_workers: 0
      data_security_mode: SINGLE_USER
      spark_conf:
        spark.master: "local[*, 4]"
        spark.databricks.cluster.profile: singleNode
```

**Single-node GPU:**
```yaml
job_clusters:
  - job_cluster_key: single_gpu
    new_cluster:
      spark_version: "17.3.x-gpu-ml-scala2.13"
      node_type_id: "Standard_NC4as_T4_v3"
      num_workers: 0
      data_security_mode: SINGLE_USER
      spark_conf:
        spark.master: "local[*, 4]"
        spark.databricks.cluster.profile: singleNode
```

**Ray distributed:**
```yaml
job_clusters:
  - job_cluster_key: ray_distributed
    new_cluster:
      spark_version: "17.3.x-cpu-ml-scala2.13"
      node_type_id: "Standard_D16s_v5"
      num_workers: 4
      data_security_mode: SINGLE_USER
      spark_conf:
        spark.executorEnv.OMP_NUM_THREADS: "15"
```

## XGBoost Hyperparameter Defaults

Good starting point for imbalanced binary classification:

```python
xgb_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",       # or "gpu_hist" for GPU
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": scale_pos_weight,  # from Step 2
    "random_state": 42,
    "verbosity": 1,
}
```

## References

- **Feature prep details**: [feature-prep.md](feature-prep.md)
- **Databricks gotchas**: [gotchas.md](gotchas.md)
- **Project learnings**: See `docs/LEARNINGS.md` in the repo
- **Deployment guide**: See `docs/DEPLOYMENT.md` in the repo
```

**Step 3: Verify the file**

```bash
wc -l .claude/skills/train-xgb-databricks/SKILL.md
```

Expected: ~180-200 lines (well under 500 limit).

**Step 4: Commit**

```bash
git add .claude/skills/train-xgb-databricks/SKILL.md
git commit -m "feat: add train-xgb-databricks skill — main decision tree"
```

---

### Task 2: Write feature-prep.md

**Files:**
- Create: `.claude/skills/train-xgb-databricks/feature-prep.md`

**Step 1: Write the file**

Content covers:
- Type detection from Spark schema
- Numeric feature handling (pass-through)
- Categorical encoding (ordinal for low-cardinality, drop for high)
- Imbalance handling (scale_pos_weight formula)
- Stratified train/test split
- Memory estimation formula
- Spark-to-pandas conversion pattern

```markdown
# Feature Preparation Reference

## Type Detection from Spark Schema

```python
from pyspark.sql.types import (IntegerType, LongType, FloatType,
                                DoubleType, StringType, BooleanType)

numeric_types = (IntegerType, LongType, FloatType, DoubleType)
cat_types = (StringType,)

target_col = "label"  # adjust to your target column

numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, numeric_types) and f.name != target_col]
cat_cols = [f.name for f in df.schema.fields
            if isinstance(f.dataType, cat_types)]
bool_cols = [f.name for f in df.schema.fields
             if isinstance(f.dataType, BooleanType)]

# Booleans: cast to int
for col in bool_cols:
    df = df.withColumn(col, df[col].cast("integer"))
    numeric_cols.append(col)
```

## Handling Categoricals

XGBoost (CPU `hist` method) does not natively handle string features.

**Low-cardinality (< 50 unique values):** Ordinal encode.

```python
from pyspark.ml.feature import StringIndexer

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx",
                          handleInvalid="keep")
            for c in cat_cols]

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

# Use indexed columns instead of originals
feature_cols = numeric_cols + [f"{c}_idx" for c in cat_cols]
```

**High-cardinality (≥ 50 unique):** Drop the column or use feature hashing. High-cardinality ordinal encoding creates misleading ordinal relationships.

```python
high_card = [c for c in cat_cols
             if df.select(c).distinct().count() >= 50]
cat_cols = [c for c in cat_cols if c not in high_card]
# high_card columns are excluded from features
```

## Imbalance Handling

For binary classification with imbalanced classes:

```python
class_counts = df.groupBy(target_col).count().collect()
counts = {row[target_col]: row["count"] for row in class_counts}
scale_pos_weight = counts[0] / counts[1]  # negative / positive
```

This tells XGBoost to weight the minority class higher during training.

## Train/Test Split

**Single-node (pandas):**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Distributed (Ray Data):**
```python
train_ds, test_ds = ray_dataset.train_test_split(test_size=0.2, seed=42)
```

## Memory Estimation

```python
import psutil

estimated_gb = (row_count * feature_count * 8) / 1e9  # 8 bytes per float64
available_gb = psutil.virtual_memory().available / 1e9

if estimated_gb > available_gb * 0.8:
    print(f"WARNING: {estimated_gb:.1f} GB > 80% of {available_gb:.1f} GB available")
    print("Consider distributed training (Ray) or a larger node")
```

**GPU memory estimation:**
```python
gpu_mem_needed_gb = estimated_gb * 6  # XGBoost GPU needs ~6x raw data for histograms
```

## Spark to Pandas Conversion

For single-node training, convert Spark DataFrame to pandas:

```python
import time
load_start = time.time()
pdf = df.select(feature_cols + [target_col]).toPandas()
print(f"Loaded {len(pdf):,} rows in {time.time() - load_start:.1f}s")
print(f"Memory: {pdf.memory_usage(deep=True).sum() / 1e9:.2f} GB")

X = pdf[feature_cols]
y = pdf[target_col]
```
```

**Step 2: Commit**

```bash
git add .claude/skills/train-xgb-databricks/feature-prep.md
git commit -m "feat: add feature-prep reference for train-xgb skill"
```

---

### Task 3: Write track-single-node-cpu.md

**Files:**
- Create: `.claude/skills/train-xgb-databricks/track-single-node-cpu.md`

**Step 1: Write the file**

Complete notebook recipe derived from `notebooks/train_xgb_single.ipynb`. Includes:
- Widget setup
- Environment validation gate
- MLflow setup with system metrics
- Data loading from UC with memory check
- Feature prep and stratified split
- XGBoost training (`tree_method="hist"`, `n_jobs=cpu_count()`)
- Model registration to Unity Catalog
- Full evaluation (AUC-PR, AUC-ROC, F1, confusion matrix)
- Structured JSON exit

```markdown
# Track: Single-Node CPU

Complete notebook recipe for training XGBoost on a single Databricks node using CPU.

**When to use:** Dataset ≤ 10M rows, CPU-only cluster.

**Runtime:** `17.3.x-cpu-ml-scala2.13`
**Node types:** D16s_v5 (16 vCPU, 64 GB), E16s_v5 (16 vCPU, 128 GB), E32s_v5 (32 vCPU, 256 GB)

## Notebook Structure

### Cell 1: Install and restart

```python
%pip install -U mlflow
%restart_python
```

### Cell 2: Widget parameters

```python
dbutils.widgets.dropdown("data_size", "tiny",
    ["tiny", "small", "medium", "medium_large", "large", "xlarge"], "Data Size")
dbutils.widgets.text("node_type", "D16sv5", "Node Type")
dbutils.widgets.dropdown("run_mode", "full", ["full", "smoke"], "Run Mode")
dbutils.widgets.text("catalog", "brian_gen_ai", "Catalog")
dbutils.widgets.text("schema", "xgb_scaling", "Schema")
dbutils.widgets.text("table_name", "", "Table Name (override)")
```

### Cell 3: Environment validation

```python
import sys, os
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/".join(notebook_path.split("/")[:-2])
sys.path.insert(0, f"/Workspace{repo_root}")

from src.validate_env import validate_environment
validate_environment(track="single-node-scaling", expected_workers=0, require_gpu=False)
```

### Cell 4: Configuration

```python
data_size = dbutils.widgets.get("data_size")
node_type = dbutils.widgets.get("node_type")
run_mode = dbutils.widgets.get("run_mode")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name_override = dbutils.widgets.get("table_name").strip()

from src.config import get_preset

if table_name_override:
    input_table = f"{catalog}.{schema}.{table_name_override}"
    data_size_label = table_name_override.replace("imbalanced_", "")
else:
    preset = get_preset(data_size)
    input_table = f"{catalog}.{schema}.imbalanced_{preset.table_suffix}"
    data_size_label = data_size

run_name = f"smoke_{node_type}" if run_mode == "smoke" else f"{data_size_label}_{node_type}"
```

### Cell 5: MLflow setup

```python
import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import mlflow
mlflow.enable_system_metrics_logging()
mlflow.set_registry_uri("databricks-uc")

uc_model_name = f"{catalog}.{schema}.xgb_single_cpu"
user_email = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{user_email}/xgb_scaling_benchmark"
mlflow.set_experiment(experiment_path)
```

### Cell 6: Load data with memory check

```python
import psutil, time

# Memory check (if using preset)
if not table_name_override:
    estimated_gb = (preset.rows * preset.total_features * 8) / 1e9
    available_gb = psutil.virtual_memory().available / 1e9
    if estimated_gb > available_gb * 0.8:
        print(f"WARNING: {estimated_gb:.1f} GB exceeds 80% of {available_gb:.1f} GB RAM")

load_start = time.time()
df = spark.table(input_table).toPandas()
load_time = time.time() - load_start

mlflow_dataset = mlflow.data.from_pandas(df, source=input_table,
                                          name=data_size_label, targets="label")
```

### Cell 7: Feature prep and split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["label"])
y = df["label"]

class_counts = y.value_counts().sort_index()
scale_pos_weight = class_counts[0] / class_counts[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Cell 8: Train and evaluate

```python
import xgboost as xgb
from sklearn.metrics import (average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score, confusion_matrix)
from mlflow.models import infer_signature

n_cores = os.cpu_count()
xgb_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": n_cores,
    "random_state": 42,
}

with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
    mlflow.log_input(mlflow_dataset, context="training")
    mlflow.log_params({
        "data_size": data_size, "node_type": node_type,
        "n_rows": len(df), "n_features": X.shape[1], "n_cores": n_cores,
    })
    for k, v in xgb_params.items():
        mlflow.log_param(f"xgb_{k}", v)
    mlflow.log_metric("data_load_time_sec", load_time)

    train_start = time.time()
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    train_time = time.time() - train_start

    signature = infer_signature(X_train.head(1000), model.predict_proba(X_train.head(1000)))
    mlflow.sklearn.log_model(model, "model",
        registered_model_name=uc_model_name, signature=signature)

    mlflow.log_metric("train_time_sec", train_time)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_proba)
    auc_roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_metrics({"auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1,
        "total_time_sec": load_time + train_time})
```

### Cell 9: Exit

```python
import json
result = {"status": "ok", "run_name": run_name, "run_id": run.info.run_id,
          "n_rows": len(df), "auc_pr": round(auc_pr, 4),
          "train_time_sec": round(train_time, 1)}
dbutils.notebook.exit(json.dumps(result))
```
```

**Step 2: Commit**

```bash
git add .claude/skills/train-xgb-databricks/track-single-node-cpu.md
git commit -m "feat: add single-node CPU track recipe"
```

---

### Task 4: Write track-single-node-gpu.md

**Files:**
- Create: `.claude/skills/train-xgb-databricks/track-single-node-gpu.md`

**Step 1: Write the file**

Same structure as CPU track but with GPU-specific changes derived from `notebooks/train_xgb_gpu.ipynb`:
- `%pip install -U mlflow pynvml`
- `tree_method: "gpu_hist"`, `gpu_id: 0`
- nvidia-smi validation and GPU info capture
- GPU memory estimation (data_gb × 6 vs VRAM)
- GPU-specific MLflow params (gpu_type, gpu_memory_gb, gpu_count)
- Runtime: `17.3.x-gpu-ml-scala2.13`
- Node types: NC4as_T4_v3 (1x T4, 16 GB), NC16as_T4_v3 (4x T4)

Key differences from CPU track:
- Cell 1: `%pip install -U mlflow pynvml`
- Cell 3: `validate_environment(track="gpu-scaling", require_gpu=True)`
- Cell 6: GPU memory check alongside RAM check
- Cell 8: `tree_method="gpu_hist"` instead of `"hist"`, no `n_jobs`

**Step 2: Commit**

```bash
git add .claude/skills/train-xgb-databricks/track-single-node-gpu.md
git commit -m "feat: add single-node GPU track recipe"
```

---

### Task 5: Write track-ray-distributed.md

**Files:**
- Create: `.claude/skills/train-xgb-databricks/track-ray-distributed.md`

**Step 1: Write the file**

Most complex track, derived from `notebooks/train_xgb_ray.ipynb`. Key elements:
- Ray on Spark initialization (`setup_ray_cluster`)
- 3-layer OMP_NUM_THREADS fix (L1: spark_conf, L2: runtime_env, L3: worker-level env+ctypes)
- `OmpDiagnosticsCollector` zero-CPU actor
- `WorkerMetricsMonitor` per-worker system metrics
- `DataParallelTrainer` with custom `xgb_train_fn`
- `ray.data.read_databricks_tables` for distributed loading
- `XGBoostConfig` backend
- Requires `warehouse_id` widget for SQL Warehouse
- Ray cluster shutdown in cleanup cell

**Step 2: Commit**

```bash
git add .claude/skills/train-xgb-databricks/track-ray-distributed.md
git commit -m "feat: add Ray distributed track recipe"
```

---

### Task 6: Write gotchas.md

**Files:**
- Create: `.claude/skills/train-xgb-databricks/gotchas.md`

**Step 1: Write the file**

Distilled from `docs/LEARNINGS.md`, covering only the gotchas an agent needs to avoid:

```markdown
# Databricks Gotchas for XGBoost Training

## G1: OMP_NUM_THREADS=1 (Critical — 3.4x performance impact)

Databricks silently sets `OMP_NUM_THREADS=1` on Spark executors. XGBoost's C++ layer
caps threads at `min(nthread, omp_get_max_threads())`, so your `nthread=14` silently
becomes 1.

**Fix (3 layers):**
1. `spark.executorEnv.OMP_NUM_THREADS: "15"` in cluster Spark config
2. `ray.init(runtime_env={"env_vars": {"OMP_NUM_THREADS": "15"}})`
3. Worker-level: `os.environ["OMP_NUM_THREADS"]` + ctypes BEFORE `import xgboost`

Layer 1 is most important — layers 2 and 3 may be too late if libgomp is already loaded.

## G2: ML Runtime suffix required

Use `17.3.x-cpu-ml-scala2.13` (CPU) or `17.3.x-gpu-ml-scala2.13` (GPU).
The plain `-scala2.13` runtime does NOT include XGBoost, Ray, or MLflow.

## G3: src/__init__.py must start with docstring, not comment

If `src/__init__.py` starts with `# comment`, Databricks DBR 17.3 misidentifies it
as a notebook → `NotebookImportException` blocks all imports.

**Fix:** Use `"""docstring."""` instead of `# comment` as first line.

## G4: Unity Catalog requires SINGLE_USER mode

Set `data_security_mode: SINGLE_USER` in cluster config for UC access.

## G5: Jobs API needs task-level run_id for error details

`GET /api/2.1/jobs/runs/get` only returns generic errors.
For stack traces: get task-level `run_id` from `tasks[].run_id`,
then call `GET /api/2.1/jobs/runs/get-output?run_id=<task_run_id>`.

## G6: Workspace code drift

When multiple agents push code, deployed code diverges from git.
Always deploy via `databricks bundle deploy` (single source of truth).
Never edit workspace files directly.

## G7: Azure Spot preemption on large VMs

E32s_v5 spot instances get preempted frequently. Use `SPOT_WITH_FALLBACK_AZURE`
or pin to on-demand for long experiments (> 30 min).
```

**Step 2: Commit**

```bash
git add .claude/skills/train-xgb-databricks/gotchas.md
git commit -m "feat: add Databricks gotchas reference for train-xgb skill"
```

---

### Task 7: Push branch and open PR

**Step 1: Create branch**

```bash
cd /workspace/group/scaling_xgb_work
git fetch origin
git switch main && git pull --rebase
git switch -c agent/andy/train-xgb-skill origin/main
```

Note: If commits were made on main during earlier tasks, cherry-pick or recommit on this branch.

**Step 2: Verify all files exist**

```bash
ls -la .claude/skills/train-xgb-databricks/
# Expected: SKILL.md, feature-prep.md, track-single-node-cpu.md,
#           track-single-node-gpu.md, track-ray-distributed.md, gotchas.md
```

**Step 3: Verify SKILL.md is under 500 lines**

```bash
wc -l .claude/skills/train-xgb-databricks/SKILL.md
```

**Step 4: Push and create PR**

```bash
git push -u origin HEAD
gh pr create --title "feat: add train-xgb-databricks Claude skill" \
  --body "## Summary
- New Claude Code skill for training XGBoost on Databricks end-to-end
- Decision-tree approach: profiles data → selects track → generates notebook
- Three tracks: single-node CPU, single-node GPU, Ray distributed
- Encodes all Databricks gotchas (OMP fix, UC, DAB patterns)
- Progressive disclosure: SKILL.md overview + track-specific recipes

## Files
- .claude/skills/train-xgb-databricks/SKILL.md — main decision tree
- .claude/skills/train-xgb-databricks/feature-prep.md — feature engineering reference
- .claude/skills/train-xgb-databricks/track-single-node-cpu.md — CPU recipe
- .claude/skills/train-xgb-databricks/track-single-node-gpu.md — GPU recipe
- .claude/skills/train-xgb-databricks/track-ray-distributed.md — Ray recipe
- .claude/skills/train-xgb-databricks/gotchas.md — Databricks pitfalls

## Test plan
- [ ] Verify SKILL.md < 500 lines
- [ ] Verify all track files are reachable from SKILL.md links
- [ ] Manual test: give an agent a UC table and verify it produces working notebook code
"
```

**Step 5: Verify PR checks**

```bash
gh pr checks
```

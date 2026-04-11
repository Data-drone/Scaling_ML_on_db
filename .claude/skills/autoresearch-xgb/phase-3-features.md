# Phase 3: FEATURE ENGINEERING

Run interactive cells on the live cluster to prepare features for training.

**Prerequisites:** `cluster_id`, `context_id` from Phase 2. Profile from Phase 1.

## How to Run a Cell

Execute cells via Command Execution API (api-reference.md). On error, retry once with fix; if still fails, log error in markdown cell and continue. Append each code cell + markdown explanation to notebook.

## Cell 1: Imports and setup

Import pandas, numpy, psutil, mlflow. Set `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true` (before mlflow import). `mlflow.set_registry_uri("databricks-uc")`. Get user email via `spark.sql("SELECT current_user()")`. Set MLflow experiment to `/Users/{email}/autoresearch`. Print available RAM and CPU count. Record `available_ram_gb` and `experiment_path` from output.

Add to notebook with markdown: `## 3. Environment Setup`

## Cell 2: Load data

**Single-node:** `df = spark.table("{table}").toPandas()`. Time it, print row/col count, memory usage, remaining RAM.
**Ray distributed:** Just verify table access (`spark.table("{table}").count()`) — actual loading happens in Phase 4.

**Runtime memory check:** If memory usage > 80% available RAM, add warning markdown cell. Do NOT abort.

Add to notebook with markdown: `## 4. Data Loading`

## Cell 2.5: Auto-detect string-typed numerics and datetimes

For each string column (excl target): sample 200 non-null values. Try numeric first: strip `$`, commas, `%`, whitespace via regex `[\$,\s%]` then `pd.to_numeric`; if >80% success, convert whole column. Then try datetime: `pd.to_datetime(format='mixed')`; if >80% success, extract `hour`/`dayofweek`/`month` features and drop original column.

Print summary of conversions. After this cell, string column lists will have changed.

## Cell 3: Handle categoricals

Drop high-cardinality columns (>=50 unique). OrdinalEncoder on low-cardinality (<50) with `handle_unknown='use_encoded_value'`, `unknown_value=-1`. Build column lists dynamically from Phase 1 profile.

## Cell 4: Handle booleans and types

Cast boolean cols to int. Drop timestamp cols (not supported by XGBoost). Recompute `feature_cols = [c for c in df.columns if c != target_col]`.

## Cell 5: Handle nulls

Drop columns with >50% null. Impute remaining nulls with median.

## Cell 6: Feature count guardrail

If features > 1000, drop lowest-variance columns to cap at 1000.

## Cell 7: Target encoding, class balance, and train/test split

Drop null targets. LabelEncoder for string targets (object/string dtype). Detect binary (2 classes, compute `scale_pos_weight = count_class0 / count_class1`) vs multiclass (>2 classes). Set `task_type` to `"binary"` or `"multiclass"`.

Stratified 80/20 split (`train_test_split` with `stratify=y`); fall back to non-stratified if minority class has < 2 samples (catches ValueError).

Outputs: `X_train`, `X_test`, `y_train`, `y_test`, `task_type`, `scale_pos_weight`, `n_classes`.

Add to notebook with markdown: `## 5. Feature Engineering` with subsections for each transform.

## Checkpoint: Upload notebook

After all Phase 3 cells complete, upload the notebook-so-far to the workspace
(see api-reference.md, Workspace Import). This is a crash recovery checkpoint —
if the cluster dies in Phase 4, we still have profiling + feature eng preserved.

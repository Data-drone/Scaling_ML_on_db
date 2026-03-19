# Phase 3: FEATURE ENGINEERING

Run interactive cells on the live cluster to prepare features for training.

**Prerequisites:** `cluster_id`, `context_id` from Phase 2. Profile from Phase 1.

## How to Run a Cell

For every code cell in this phase:

1. Write the Python code to a temp file (`/tmp/cell_N.py`)
2. JSON-encode it and send via Command Execution API (see api-reference.md)
3. Poll until `Finished` or `Error`
4. Read output from `results.data` (text) or `results.cause` (error)
5. If Error: try to fix the code and retry once. If still fails, add a markdown
   cell documenting the error and move on.
6. Append the code cell + a markdown explanation cell to the local notebook file

## Cell 1: Imports and setup

Run on cluster:

```python
import os, sys, time, psutil
import pandas as pd
import numpy as np
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import mlflow
mlflow.set_registry_uri("databricks-uc")
user_email = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{user_email}/autoresearch"
mlflow.set_experiment(experiment_path)
available_ram_gb = psutil.virtual_memory().available / 1e9
print(f"MLflow experiment: {experiment_path}")
print(f"Available RAM: {available_ram_gb:.1f} GB")
print(f"CPU cores: {os.cpu_count()}")
```

Record `available_ram_gb` and `experiment_path` from the output.

Add to notebook with markdown: `## 3. Environment Setup`

## Cell 2: Load data

**For single-node tracks:**

```python
load_start = time.time()
df = spark.table("{table}").toPandas()
load_time = time.time() - load_start
mem_gb = df.memory_usage(deep=True).sum() / 1e9
print(f"Loaded {len(df):,} rows x {len(df.columns)} cols in {load_time:.1f}s")
print(f"Memory usage: {mem_gb:.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB")
```

**For Ray distributed track:** Data loading happens in Phase 4 inside the Ray
training function. In Phase 3, just verify the table is accessible:

```python
row_count = spark.table("{table}").count()
cols = spark.table("{table}").columns
print(f"Table verified: {row_count:,} rows, {len(cols)} columns")
```

**Runtime memory check:** Parse memory usage from output. If usage > 80% of
available RAM, add a warning markdown cell. Do NOT abort — just document it.

Add to notebook with markdown: `## 4. Data Loading`

## Cell 3: Handle categoricals

Build this cell dynamically from the profile (Phase 1 cardinality data):

```python
# Categoricals to ENCODE (cardinality < 50): {list from profile}
# Categoricals to DROP (cardinality >= 50): {list from profile}

cols_to_drop = [{high_cardinality_cols}]
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} high-cardinality columns: {cols_to_drop}")

cat_cols = [{low_cardinality_cols}]
if cat_cols:
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = enc.fit_transform(df[cat_cols])
    print(f"Ordinal-encoded {len(cat_cols)} categorical columns")
```

Add to notebook with markdown explaining encoding decisions and cardinality
thresholds.

## Cell 4: Handle booleans and types

```python
bool_cols = [{boolean_cols_from_profile}]
for col in bool_cols:
    df[col] = df[col].astype(int)
if bool_cols:
    print(f"Cast {len(bool_cols)} boolean columns to int")

# Drop timestamp columns (not supported by XGBoost)
ts_cols = [{timestamp_cols_from_profile}]
if ts_cols:
    df = df.drop(columns=ts_cols)
    print(f"Dropped {len(ts_cols)} timestamp columns: {ts_cols}")

feature_cols = [c for c in df.columns if c != "{target_col}"]
print(f"Features after type handling: {len(feature_cols)}")
```

## Cell 5: Handle nulls

```python
null_pcts = df[feature_cols].isnull().mean()
high_null_cols = null_pcts[null_pcts > 0.5].index.tolist()
if high_null_cols:
    df = df.drop(columns=high_null_cols)
    feature_cols = [c for c in feature_cols if c not in high_null_cols]
    print(f"Dropped {len(high_null_cols)} high-null columns (>50%): {high_null_cols}")

null_remaining = df[feature_cols].isnull().sum().sum()
if null_remaining > 0:
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    print(f"Imputed {null_remaining:,} remaining null values with median")
else:
    print("No nulls remaining")
```

## Cell 6: Feature count guardrail

```python
if len(feature_cols) > 1000:
    variances = df[feature_cols].var().sort_values()
    n_drop = len(feature_cols) - 1000
    low_var_cols = variances.head(n_drop).index.tolist()
    df = df.drop(columns=low_var_cols)
    feature_cols = [c for c in feature_cols if c not in low_var_cols]
    print(f"Dropped {n_drop} lowest-variance features to cap at 1000")

print(f"Final feature count: {len(feature_cols)}")
```

## Cell 7: Class balance and train/test split

```python
from sklearn.model_selection import train_test_split

X = df[sorted(feature_cols)]
y = df["{target_col}"]

class_counts = y.value_counts().sort_index()
scale_pos_weight = float(class_counts.iloc[0] / class_counts.iloc[1])
minority_pct = class_counts.min() / len(y) * 100

print(f"Class distribution: { {k: v for k, v in class_counts.items()} }")
print(f"Minority class: {minority_pct:.1f}%")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
```

Add to notebook with markdown: `## 5. Feature Engineering` followed by
subsections for each transform.

## Checkpoint: Upload notebook

After all Phase 3 cells complete, upload the notebook-so-far to the workspace
(see api-reference.md, Workspace Import). This is a crash recovery checkpoint —
if the cluster dies in Phase 4, we still have profiling + feature eng preserved.

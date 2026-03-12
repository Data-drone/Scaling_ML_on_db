# Track: Single-Node GPU

Complete notebook recipe for training XGBoost on a single Databricks node using GPU acceleration.

**When to use:** Dataset <= 10M rows with GPU available, or 10M-100M rows if estimated GPU memory (data_gb x 6) fits in GPU VRAM.

**Runtime:** `17.3.x-gpu-ml-scala2.13`
**Node types:** NC4as_T4_v3 (1x T4, 16 GB VRAM), NC16as_T4_v3 (4x T4, 64 GB total)

**Current limitation:** Uses single-GPU training only. On multi-GPU nodes, only the GPU specified by `device` is used.

## Notebook Structure

### Cell 1: Install and restart

```python
%pip install -U mlflow pynvml
%restart_python
```

### Cell 2: Widget parameters

```python
dbutils.widgets.dropdown("data_size", "tiny",
    ["tiny", "small", "medium", "medium_large", "large", "xlarge"], "Data Size")
dbutils.widgets.text("node_type", "NC4asT4v3", "Node Type")
dbutils.widgets.text("gpu_id", "0", "GPU ID")
dbutils.widgets.dropdown("run_mode", "full", ["full", "smoke"], "Run Mode")
dbutils.widgets.text("catalog", "brian_gen_ai", "Catalog")
dbutils.widgets.text("schema", "xgb_scaling", "Schema")
dbutils.widgets.text("table_name", "", "Table Name (override)")
```

### Cell 3: Error tracking and configuration

```python
_notebook_errors = []
def log_error(msg, exc=None):
    import traceback
    entry = {"error": str(msg)}
    if exc: entry["traceback"] = traceback.format_exc()
    _notebook_errors.append(entry)
    print(f"ERROR LOGGED: {msg}")

data_size = dbutils.widgets.get("data_size")
node_type = dbutils.widgets.get("node_type")
gpu_id = dbutils.widgets.get("gpu_id")
run_mode = dbutils.widgets.get("run_mode")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name_override = dbutils.widgets.get("table_name").strip()

import sys, os
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/".join(notebook_path.split("/")[:-2])
sys.path.insert(0, f"/Workspace{repo_root}")

from src.config import get_preset, PRESETS

if table_name_override:
    input_table = f"{catalog}.{schema}.{table_name_override}"
    data_size_label = table_name_override.replace("imbalanced_", "")
    preset = None
else:
    preset = get_preset(data_size)
    input_table = f"{catalog}.{schema}.imbalanced_{preset.table_suffix}"
    data_size_label = data_size

run_name = f"smoke_gpu_{node_type}" if run_mode == "smoke" else f"{data_size_label}_gpu_{node_type}"
print(f"Config: {data_size} | {node_type} | GPU {gpu_id} | {run_mode} | table={input_table}")
```

### Cell 4: Environment validation and GPU detection

```python
from src.validate_env import validate_environment
validate_environment(
    track="gpu-scaling",
    expected_workers=0,
    require_gpu=True,
)

import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=10
)
if result.returncode == 0:
    gpu_lines = result.stdout.strip().split('\n')
    gpu_count = len(gpu_lines)
    print(f"Detected {gpu_count} GPU(s):")
    gpu_names = []
    gpu_mem_totals = []
    for i, line in enumerate(gpu_lines):
        parts = line.split(',')
        name = parts[0].strip()
        mem_total = parts[1].strip()
        mem_free = parts[2].strip() if len(parts) > 2 else "N/A"
        gpu_names.append(name)
        gpu_mem_totals.append(float(mem_total.split()[0]))  # MiB
        print(f"  GPU {i}: {name}, {mem_total} total, {mem_free} free")

    selected_gpu = int(gpu_id)
    if selected_gpu >= gpu_count:
        print(f"WARNING: gpu_id={selected_gpu} but only {gpu_count} GPU(s). Falling back to GPU 0.")
        selected_gpu = 0

    gpu_name = gpu_names[selected_gpu]
    gpu_mem_total = gpu_mem_totals[selected_gpu]
    print(f"\nUsing GPU {selected_gpu}: {gpu_name}, {gpu_mem_total/1024:.1f} GB")
    if gpu_count > 1:
        print(f"NOTE: {gpu_count} GPUs available but only GPU {selected_gpu} used (single-GPU training).")
else:
    raise RuntimeError("No GPU detected! Use GPU ML Runtime (17.3.x-gpu-ml-scala2.13)")
```

### Cell 5: MLflow setup

```python
import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import mlflow
mlflow.enable_system_metrics_logging()
mlflow.set_registry_uri("databricks-uc")

uc_model_name = f"{catalog}.{schema}.xgb_gpu_single"
user_email = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{user_email}/xgb_scaling_benchmark"
mlflow.set_experiment(experiment_path)
```

### Cell 6: Load data with memory checks

```python
import psutil, time

# RAM check
if preset:
    estimated_gb = (preset.rows * preset.total_features * 8) / 1e9
    available_gb = psutil.virtual_memory().available / 1e9
    if estimated_gb > available_gb * 0.8:
        log_error(f"Data {estimated_gb:.1f} GB exceeds 80% of {available_gb:.1f} GB RAM")

load_start = time.time()
df = spark.table(input_table).toPandas()
load_time = time.time() - load_start
print(f"Loaded {len(df):,} rows x {len(df.columns)} cols in {load_time:.1f}s")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# GPU memory check
raw_data_gb = df.memory_usage(deep=True).sum() / 1e9
estimated_gpu_gb = raw_data_gb * 6  # XGBoost GPU needs ~6x raw data for histograms
print(f"Raw data: {raw_data_gb:.2f} GB | Estimated GPU memory needed: {estimated_gpu_gb:.2f} GB")
if estimated_gpu_gb > gpu_mem_total / 1024:
    print(f"WARNING: {estimated_gpu_gb:.1f} GB may exceed GPU memory {gpu_mem_total/1024:.1f} GB")
else:
    print(f"GPU memory check OK: {estimated_gpu_gb:.1f} GB < {gpu_mem_total/1024:.1f} GB")

mlflow_dataset = mlflow.data.from_pandas(df, source=input_table,
                                          name=data_size_label, targets="label")
```

### Cell 7: Feature prep and split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["label"])
y = df["label"]

class_counts = y.value_counts().sort_index()
minority_ratio = class_counts[1] / len(y)
scale_pos_weight = class_counts[0] / class_counts[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | scale_pos_weight: {scale_pos_weight:.2f}")
```

### Cell 8: Train and evaluate

```python
import xgboost as xgb
from sklearn.metrics import (average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score, confusion_matrix, classification_report)
from mlflow.models import infer_signature

xgb_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",            # XGBoost 2.0+: "hist" auto-uses GPU when device is set
    "device": f"cuda:{selected_gpu}", # XGBoost 2.0+: replaces deprecated gpu_hist + gpu_id
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "verbosity": 1,
}

try:
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        run_id = run.info.run_id
        mlflow.log_input(mlflow_dataset, context="training")
        mlflow.log_params({
            "data_size": data_size, "node_type": node_type,
            "n_rows": len(df), "n_features": X.shape[1],
            "minority_ratio": round(minority_ratio, 4),
            "training_mode": "gpu_single",
            "gpu_type": gpu_name,
            "gpu_memory_gb": round(gpu_mem_total / 1024, 1),
            "gpu_count": gpu_count,
            "gpu_id": selected_gpu,
        })
        for k, v in xgb_params.items():
            mlflow.log_param(f"xgb_{k}", v)
        mlflow.log_metric("data_load_time_sec", load_time)

        print("Training XGBoost with GPU (tree_method=gpu_hist)...")
        train_start = time.time()
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        mlflow.log_metric("train_time_sec", train_time)

        sig = infer_signature(X_train.head(1000), model.predict_proba(X_train.head(1000)))
        mlflow.sklearn.log_model(model, "model",
            registered_model_name=uc_model_name, signature=sig)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc_pr = average_precision_score(y_test, y_proba)
        auc_roc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metrics({
            "auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1,
            "raw_data_gb": round(raw_data_gb, 2),
            "estimated_gpu_mem_gb": round(estimated_gpu_gb, 2),
            "true_negatives": int(cm[0,0]), "false_positives": int(cm[0,1]),
            "false_negatives": int(cm[1,0]), "true_positives": int(cm[1,1]),
            "total_time_sec": load_time + train_time,
        })
        print(f"AUC-PR: {auc_pr:.4f} | Train: {train_time:.1f}s")
        print(classification_report(y_test, y_pred))

except Exception as e:
    log_error(f"Training failed: {e}", exc=e)
    print(f"TRAINING FAILED: {e}")
```

### Cell 9: Exit

```python
import json
try:
    result = {
        "status": "ok" if not _notebook_errors else "error",
        "run_name": run_name, "run_id": run_id,
        "gpu_type": gpu_name, "gpu_id": selected_gpu,
        "tree_method": "gpu_hist",
        "n_rows": len(df), "auc_pr": round(auc_pr, 4),
        "train_time_sec": round(train_time, 1),
    }
    if _notebook_errors:
        result["errors"] = [e["error"] for e in _notebook_errors]
except NameError as e:
    result = {"status": "error", "error": str(e),
              "errors": [e["error"] for e in _notebook_errors] if _notebook_errors else []}

dbutils.notebook.exit(json.dumps(result))
```

## Key Differences from CPU Track

| Aspect | CPU Track | GPU Track |
|--------|-----------|-----------|
| `tree_method` | `"hist"` | `"gpu_hist"` |
| Threading | `n_jobs=os.cpu_count()` | Not needed (GPU handles parallelism) |
| Extra param | -- | `gpu_id=selected_gpu` |
| Memory check | RAM only | RAM + GPU VRAM (data_gb x 6) |
| Pip install | `mlflow` | `mlflow pynvml` |
| Validation | `require_gpu=False` | `require_gpu=True` |
| Runtime | `17.3.x-cpu-ml-scala2.13` | `17.3.x-gpu-ml-scala2.13` |

## DAB Job Definition

```yaml
train_xgb_gpu_nc4:
  name: "[${var.env}] Train XGBoost GPU (NC4as_T4_v3)"
  parameters:
    - name: data_size
      default: tiny
    - name: node_type
      default: NC4asT4v3
    - name: gpu_id
      default: "0"
    - name: run_mode
      default: full
    - name: table_name
      default: ""
  tasks:
    - task_key: train
      notebook_task:
        notebook_path: ./notebooks/train_xgb_gpu.ipynb
        base_parameters:
          data_size: "{{job.parameters.data_size}}"
          node_type: "{{job.parameters.node_type}}"
          gpu_id: "{{job.parameters.gpu_id}}"
          run_mode: "{{job.parameters.run_mode}}"
          table_name: "{{job.parameters.table_name}}"
          catalog: ${var.catalog}
          schema: ${var.schema}
        source: WORKSPACE
      new_cluster:
        spark_version: "17.3.x-gpu-ml-scala2.13"
        node_type_id: "Standard_NC4as_T4_v3"
        num_workers: 0
        data_security_mode: SINGLE_USER
        spark_conf:
          spark.databricks.cluster.profile: singleNode
          spark.master: "local[*, 4]"
        custom_tags:
          ResourceClass: SingleNode
        azure_attributes:
          availability: SPOT_WITH_FALLBACK_AZURE
          spot_bid_max_price: -1
      libraries:
        - pypi:
            package: psutil
        - pypi:
            package: pynvml
  tags:
    project: scaling_xgb
    environment: ${var.env}
```

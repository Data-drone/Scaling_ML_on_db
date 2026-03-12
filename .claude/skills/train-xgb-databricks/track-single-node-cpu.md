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
_notebook_errors = []
def log_error(msg, exc=None):
    import traceback
    entry = {"error": str(msg)}
    if exc: entry["traceback"] = traceback.format_exc()
    _notebook_errors.append(entry)
    print(f"ERROR LOGGED: {msg}")

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
    preset = None
else:
    preset = get_preset(data_size)
    input_table = f"{catalog}.{schema}.imbalanced_{preset.table_suffix}"
    data_size_label = data_size

run_name = f"smoke_{node_type}" if run_mode == "smoke" else f"{data_size_label}_{node_type}"
print(f"Config: {data_size} | {node_type} | {run_mode} | table={input_table}")
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

if preset:
    estimated_gb = (preset.rows * preset.total_features * 8) / 1e9
    available_gb = psutil.virtual_memory().available / 1e9
    if estimated_gb > available_gb * 0.8:
        log_error(f"Data {estimated_gb:.1f} GB exceeds 80% of {available_gb:.1f} GB RAM")

load_start = time.time()
df = spark.table(input_table).toPandas()
load_time = time.time() - load_start
print(f"Loaded {len(df):,} rows x {len(df.columns)} cols in {load_time:.1f}s")

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
    "verbosity": 1,
}

try:
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        run_id = run.info.run_id
        mlflow.log_input(mlflow_dataset, context="training")
        mlflow.log_params({
            "data_size": data_size, "node_type": node_type,
            "n_rows": len(df), "n_features": X.shape[1],
            "minority_ratio": round(minority_ratio, 4), "n_cores": n_cores,
        })
        for k, v in xgb_params.items():
            mlflow.log_param(f"xgb_{k}", v)
        mlflow.log_metric("data_load_time_sec", load_time)

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

## DAB Job Definition

```yaml
train_xgb_single_d16:
  name: "[${var.env}] Train XGBoost Single (D16s_v5)"
  parameters:
    - name: data_size
      default: tiny
    - name: node_type
      default: D16sv5
    - name: run_mode
      default: full
    - name: table_name
      default: ""
  tasks:
    - task_key: train
      notebook_task:
        notebook_path: ./notebooks/train_xgb_single.ipynb
        base_parameters:
          data_size: "{{job.parameters.data_size}}"
          node_type: "{{job.parameters.node_type}}"
          run_mode: "{{job.parameters.run_mode}}"
          table_name: "{{job.parameters.table_name}}"
          catalog: ${var.catalog}
          schema: ${var.schema}
        source: WORKSPACE
      new_cluster:
        spark_version: "17.3.x-cpu-ml-scala2.13"
        node_type_id: "Standard_D16s_v5"
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
  tags:
    project: scaling_xgb
    environment: ${var.env}
```

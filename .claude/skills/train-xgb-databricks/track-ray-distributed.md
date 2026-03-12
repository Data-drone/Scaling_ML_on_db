# Track: Ray Distributed

Complete notebook recipe for distributed XGBoost training using Ray on Spark with the OMP_NUM_THREADS fix, per-worker diagnostics, and system metrics collection.

**When to use:** Dataset 10M-100M+ rows, or when data exceeds single-node memory.

**Runtime:** `17.3.x-cpu-ml-scala2.13`
**Node types:** D16s_v5 (16 vCPU, 64 GB) x 4-8 workers, D8s_v5 (8 vCPU, 32 GB) x 8 workers

**Critical:** Requires the 3-layer OMP_NUM_THREADS fix (see [gotchas.md](gotchas.md) G1). Without it, XGBoost silently uses 1 core per worker -- a 3.4x slowdown.

## Notebook Structure

### Cell 1: Install and restart

```python
%pip install -U mlflow
%restart_python
```

### Cell 2: Widget parameters

```python
_notebook_errors = []
def log_error(msg, exc=None):
    import traceback
    entry = {"error": str(msg)}
    if exc: entry["traceback"] = traceback.format_exc()
    _notebook_errors.append(entry)
    print(f"ERROR LOGGED: {msg}")

dbutils.widgets.dropdown("data_size", "tiny",
    ["tiny", "small", "medium", "medium_large", "large", "xlarge"], "Data Size")
dbutils.widgets.text("node_type", "D8sv5", "Node Type")
dbutils.widgets.dropdown("run_mode", "full", ["full", "smoke"], "Run Mode")
dbutils.widgets.text("num_workers", "0", "Num Workers (0=auto)")
dbutils.widgets.text("cpus_per_worker", "0", "CPUs per Worker (0=auto)")
dbutils.widgets.text("warehouse_id", "148ccb90800933a1", "Databricks SQL Warehouse ID")
dbutils.widgets.text("catalog", "brian_gen_ai", "Catalog")
dbutils.widgets.text("schema", "xgb_scaling", "Schema")
dbutils.widgets.text("table_name", "", "Table Name (override)")
```

### Cell 3: Configuration

```python
data_size = dbutils.widgets.get("data_size")
node_type = dbutils.widgets.get("node_type")
run_mode = dbutils.widgets.get("run_mode")
num_workers_input = int(dbutils.widgets.get("num_workers"))
cpus_per_worker_input = int(dbutils.widgets.get("cpus_per_worker"))
warehouse_id = dbutils.widgets.get("warehouse_id").strip()
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
table_name_override = dbutils.widgets.get("table_name").strip()

import sys, os
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
repo_root = "/".join(notebook_path.split("/")[:-2])
sys.path.insert(0, f"/Workspace{repo_root}")
from src.config import PRESETS as CONFIG_PRESETS, get_preset

SIZE_PRESETS = {n: {"suffix": p.table_suffix, "rows": p.rows, "features": p.total_features}
                for n, p in CONFIG_PRESETS.items()}

if table_name_override:
    input_table = f"{catalog}.{schema}.{table_name_override}"
    data_size_label = table_name_override.replace("imbalanced_", "")
else:
    preset = get_preset(data_size)
    input_table = f"{catalog}.{schema}.imbalanced_{preset.table_suffix}"
    data_size_label = data_size

run_name = (f"ray_smoke_{node_type}" if run_mode == "smoke"
            else f"ray_{data_size_label}{'_'+str(num_workers_input)+'w' if num_workers_input > 0 else ''}_{node_type}")
print(f"Config: {data_size} | {node_type} | {run_mode} | table={input_table} | run={run_name}")
```

### Cell 4: Environment validation

```python
from src.validate_env import validate_environment
_env_report = validate_environment(
    track="ray-scaling",
    expected_workers=num_workers_input if num_workers_input > 0 else None,
    raise_on_failure=False,
)
if not _env_report.passed:
    print(f"WARNING: {len(_env_report.errors)} validation error(s) -- continuing for debugging")
```

### Cell 5: MLflow setup

```python
import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
import mlflow
mlflow.enable_system_metrics_logging()
mlflow.set_registry_uri("databricks-uc")

uc_model_name = f"{catalog}.{schema}.xgb_ray_cpu"
user_email = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{user_email}/xgb_scaling_benchmark"
mlflow.set_experiment(experiment_path)
```

### Cell 6: Initialize Ray on Spark

This cell handles Ray cluster startup and the OMP_NUM_THREADS Layer 2 fix.

```python
import time, os, re
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

print(f"Ray version: {ray.__version__}")

# Databricks credentials for Ray workers (needed for Ray Data + MLflow)
databricks_host_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
os.environ["DATABRICKS_HOST"] = databricks_host_url
os.environ["DATABRICKS_TOKEN"] = databricks_token

# Get cluster info — retry as executors may be slow to register on new clusters
sc = spark.sparkContext
import time as _time
for _attempt in range(5):
    num_executors = sc._jsc.sc().getExecutorMemoryStatus().size() - 1
    if num_executors >= 1:
        break
    print(f"Waiting for executors... attempt {_attempt+1} (got {num_executors})")
    _time.sleep(10)
print(f"Executors available: {num_executors}")

# Determine per-node vCPU from node_type string (e.g., D16sv5 -> 16)
node_type_lower = node_type.lower()
node_vcpus_match = re.search(r"[de](\d+)", node_type_lower)
node_vcpus = int(node_vcpus_match.group(1)) if node_vcpus_match else 8
allocatable_cpus_per_node = max(1, node_vcpus - 1)  # Leave 1 for Spark/system

# Determine worker count (one per executor by default)
if num_workers_input > 0:
    num_workers = min(num_workers_input, num_executors)
else:
    num_workers = max(1, num_executors)

# CPUs per worker
if cpus_per_worker_input > 0:
    cpus_per_worker = min(cpus_per_worker_input, allocatable_cpus_per_node)
else:
    cpus_per_worker = allocatable_cpus_per_node

num_cpus_worker_node = allocatable_cpus_per_node

print(f"Sizing: {node_type} ({node_vcpus} vCPU) | {num_executors} executors")
print(f"Ray: {num_workers}W x {cpus_per_worker}CPU = {num_workers * cpus_per_worker} total")
```

### Cell 7: Start Ray cluster with OMP fix

```python
# OMP Layer 1: spark.executorEnv.OMP_NUM_THREADS must be in databricks.yml / cluster config
# OMP Layer 2: ray.init runtime_env (applied here)
print("Starting Ray cluster...")
ray_start = time.time()
os.environ["OMP_NUM_THREADS"] = str(allocatable_cpus_per_node)

setup_ray_cluster(
    min_worker_nodes=num_executors,
    max_worker_nodes=num_executors,
    num_cpus_worker_node=num_cpus_worker_node,
    num_gpus_worker_node=0,
    collect_log_to_path="/tmp/ray_logs",
)
ray_init_time = time.time() - ray_start

# CRITICAL OMP FIX (Layer 2): Reconnect with runtime_env
omp_threads_str = str(cpus_per_worker)
if ray.is_initialized():
    ray.shutdown()
ray.init(runtime_env={
    "env_vars": {
        "OMP_NUM_THREADS": omp_threads_str,
        "DATABRICKS_HOST": databricks_host_url,
        "DATABRICKS_TOKEN": databricks_token,
    }
})
print(f"Ray reconnected. OMP_NUM_THREADS={omp_threads_str}, init={ray_init_time:.1f}s")

# Verify resources and adjust if needed
cluster_resources = ray.cluster_resources()
available_cpus = int(cluster_resources.get("CPU", 0))
required = num_workers * cpus_per_worker + 1  # +1 for trainer overhead
if required > available_cpus:
    usable = max(1, available_cpus - 1)
    cpus_per_worker = max(1, usable // max(1, num_workers))
    num_workers = max(1, usable // max(1, cpus_per_worker))
print(f"Final: {num_workers}W x {cpus_per_worker}CPU = {num_workers*cpus_per_worker}+1 overhead")
print(f"Cluster resources: {cluster_resources}")
```

### Cell 8: Define WorkerMetricsMonitor and OmpDiagnosticsCollector

These are zero-CPU Ray actors for collecting per-worker system metrics and OMP diagnostic state.

```python
import ray

@ray.remote(num_cpus=0)
class WorkerMetricsMonitor:
    """Runs MLflow SystemMetricsMonitor on a specific worker node."""
    def __init__(self, run_id, node_id, db_host, db_token, sampling_interval=10.0):
        import os
        os.environ.update({
            "DATABRICKS_HOST": db_host,
            "DATABRICKS_TOKEN": db_token,
            "MLFLOW_TRACKING_URI": "databricks",
        })
        self._run_id = run_id
        self._node_id = node_id
        self._si = sampling_interval
        self._mon = None
        self._rn = ray.get_runtime_context().get_node_id()[:8]

    def start(self):
        from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
        self._mon = SystemMetricsMonitor(
            run_id=self._run_id, node_id=self._node_id,
            sampling_interval=self._si, samples_before_logging=1,
        )
        self._mon.start()
        return f"{self._node_id} on {self._rn}"

    def stop(self):
        if self._mon:
            self._mon.finish()
            self._mon = None
            return f"{self._node_id} stopped"
        return f"{self._node_id} n/a"


@ray.remote(num_cpus=0)
class OmpDiagnosticsCollector:
    """Collects OMP diagnostic state from each worker via .report() calls."""
    def __init__(self):
        self._results = {}

    def report(self, worker_rank, diagnostics):
        self._results[worker_rank] = diagnostics

    def get_all(self):
        return dict(self._results)


def start_worker_monitors(run_id, db_host, db_token, num_nodes, si=10.0):
    """Start a WorkerMetricsMonitor on each non-driver node."""
    head = ray.get_runtime_context().get_node_id()
    nodes = [n for n in ray.nodes()
             if n.get("Alive") and n["NodeID"] != head][:num_nodes]
    actors, futs = [], []
    for i, n in enumerate(nodes):
        a = WorkerMetricsMonitor.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=n["NodeID"], soft=False),
            name=f"metrics_w{i}",
        ).remote(run_id, f"worker_{i}", db_host, db_token, si)
        futs.append(a.start.remote())
        actors.append(a)
    for r in ray.get(futs):
        print(f"  {r}")
    return actors


def stop_worker_monitors(actors):
    """Gracefully stop all worker monitors."""
    if not actors:
        return
    try:
        for r in ray.get([a.stop.remote() for a in actors], timeout=30):
            print(f"  {r}")
    except Exception as e:
        print(f"  WARN: {e}")
    for a in actors:
        try:
            ray.kill(a)
        except Exception:
            pass


print("WorkerMetricsMonitor + OmpDiagnosticsCollector defined.")
```

### Cell 9: Load data with Ray Data

Ray Data loads directly from Unity Catalog via a SQL Warehouse -- no Spark-to-pandas conversion needed.

```python
import ray.data

if not warehouse_id:
    raise ValueError("warehouse_id is required for distributed Ray Data loading")

# Re-assert Databricks credentials on the driver.
# ray.init(runtime_env={"env_vars": ...}) sets env vars for WORKERS only;
# the driver process may lose them after ray.shutdown()+ray.init().
# read_databricks_tables() checks the driver env for DATABRICKS_TOKEN.
# CRITICAL: Strip https:// from HOST — Ray constructs URLs from it and
# double-prefixing causes host='https' resolution errors (see G9).
host_for_ray = databricks_host_url.replace("https://", "").replace("http://", "")
os.environ["DATABRICKS_HOST"] = host_for_ray
os.environ["DATABRICKS_TOKEN"] = databricks_token

print(f"Loading: {input_table} via warehouse {warehouse_id}")
load_start = time.time()

query = f"SELECT * FROM {input_table}"
full_ray_ds = ray.data.read_databricks_tables(
    warehouse_id=warehouse_id,
    query=query,
)

n_rows = full_ray_ds.count()
all_columns = list(full_ray_ds.schema().names)
if "label" not in all_columns:
    raise ValueError(f"Expected 'label' column, got: {all_columns}")

feature_columns = [c for c in all_columns if c != "label"]
load_time = time.time() - load_start
print(f"Loaded {n_rows:,} rows x {len(all_columns)} cols in {load_time:.1f}s")
```

### Cell 10: Class distribution and train/test split

```python
import numpy as np

positive_count = int(full_ray_ds.sum("label"))
negative_count = int(n_rows - positive_count)
minority_ratio = positive_count / n_rows if n_rows else 0.0
scale_pos_weight = negative_count / max(positive_count, 1)

print(f"Class 0: {negative_count:,} | Class 1: {positive_count:,} | scale_pos_weight: {scale_pos_weight:.2f}")

# Distributed train/test split
split_start = time.time()
train_ray_ds, test_ray_ds = full_ray_ds.train_test_split(test_size=0.2, seed=42)
split_time = time.time() - split_start

train_count = train_ray_ds.count()
test_count = test_ray_ds.count()
print(f"Train: {train_count:,} | Test: {test_count:,} | Split: {split_time:.1f}s")

# Bounded evaluation sample for local sklearn metrics
def _ray_dataset_to_numpy(dataset, feature_cols, label_col="label", batch_size=65536):
    x_batches, y_batches = [], []
    for batch in dataset.iter_batches(batch_format="numpy", batch_size=batch_size):
        x_batch = np.column_stack([batch[c] for c in feature_cols]).astype(np.float32, copy=False)
        y_batch = batch[label_col].astype(np.float32, copy=False)
        x_batches.append(x_batch)
        y_batches.append(y_batch)
    if not x_batches:
        return np.empty((0, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.concatenate(x_batches, axis=0), np.concatenate(y_batches, axis=0)

eval_sample_rows = min(200_000, test_count)
eval_test_ds = test_ray_ds.limit(eval_sample_rows)
X_test_eval, y_test_eval = _ray_dataset_to_numpy(eval_test_ds, feature_columns)
print(f"Evaluation sample: {X_test_eval.shape[0]:,} rows")
```

### Cell 11: Define XGBoost train function with OMP Layer 3 fix

This is the per-worker training function. The OMP Layer 3 fix sets `OMP_NUM_THREADS` and calls ctypes `omp_set_num_threads()` **before** importing xgboost.

```python
from ray.train.xgboost import RayTrainReportCallback, XGBoostConfig
from ray.train import ScalingConfig, RunConfig
from ray.train.data_parallel_trainer import DataParallelTrainer
import ray.data, ray.train

xgb_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "nthread": cpus_per_worker,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": scale_pos_weight,
    "seed": 42,
    "verbosity": 1,
}
num_boost_round = 100

scaling_config = ScalingConfig(
    num_workers=num_workers,
    use_gpu=False,
    resources_per_worker={"CPU": cpus_per_worker},
)

ray_storage_path = f"/Volumes/{catalog}/{schema}/ray_results/"
os.makedirs(ray_storage_path, exist_ok=True)
run_config = RunConfig(storage_path=ray_storage_path, name="xgb_ray_train")


def xgb_train_fn(config):
    """Per-worker train function with OMP Layer 3 fix."""
    import os, ctypes
    import numpy as np

    nthread = config.get("nthread", 1)
    diag_ref = config.get("_omp_diag_ref")
    rank = ray.train.get_context().get_world_rank()
    diag = {"omp_before": os.environ.get("OMP_NUM_THREADS", "NOT_SET")}

    # OMP Layer 3: Set env var + ctypes BEFORE import xgboost
    os.environ["OMP_NUM_THREADS"] = str(nthread)
    diag["omp_set_to"] = str(nthread)

    for lib_name in ["libgomp.so.1", "libomp.so", "libomp.so.5"]:
        try:
            lib = ctypes.CDLL(lib_name)
            lib.omp_get_max_threads.restype = ctypes.c_int
            lib.omp_set_num_threads.argtypes = [ctypes.c_int]
            before = lib.omp_get_max_threads()
            lib.omp_set_num_threads(nthread)
            diag[f"ctypes_{lib_name}"] = f"{before}->{lib.omp_get_max_threads()}"
        except OSError:
            pass

    # NOW import xgboost (after OMP is configured)
    import xgboost

    # Verify OMP state post-import
    for lib_name in ["libgomp.so.1"]:
        try:
            lib = ctypes.CDLL(lib_name)
            lib.omp_get_max_threads.restype = ctypes.c_int
            diag[f"post_{lib_name}"] = str(lib.omp_get_max_threads())
        except Exception:
            pass

    # Report diagnostics to collector
    if diag_ref:
        try:
            ray.get(diag_ref.report.remote(rank, diag), timeout=10)
        except Exception:
            pass

    label_col = config["label_column"]
    n_rounds = config["num_boost_round"]
    xp = {k: v for k, v in config.items()
          if k not in ("label_column", "num_boost_round", "dataset_keys", "_omp_diag_ref")}

    def shard_to_dmatrix(shard):
        x_batches, y_batches = [], []
        for batch in shard.iter_batches(batch_format="numpy", batch_size=65536):
            feature_cols = sorted(c for c in batch.keys() if c != label_col)
            x_batch = np.column_stack([batch[c] for c in feature_cols]).astype(np.float32, copy=False)
            y_batch = batch[label_col].astype(np.float32, copy=False)
            x_batches.append(x_batch)
            y_batches.append(y_batch)
        if not x_batches:
            return xgboost.DMatrix(
                np.empty((0, 0), dtype=np.float32),
                label=np.empty((0,), dtype=np.float32),
            )
        X = np.concatenate(x_batches, axis=0)
        y = np.concatenate(y_batches, axis=0)
        return xgboost.DMatrix(X, label=y)

    train_shard = ray.train.get_dataset_shard("train").materialize()
    dtrain = shard_to_dmatrix(train_shard)
    evals = [(dtrain, "train")]

    valid_shard = ray.train.get_dataset_shard("valid")
    if valid_shard:
        evals.append((shard_to_dmatrix(valid_shard.materialize()), "valid"))

    # Resume from checkpoint if available
    ckpt = ray.train.get_checkpoint()
    start_model, iters = None, n_rounds
    if ckpt:
        start_model = RayTrainReportCallback.get_model(ckpt)
        iters = n_rounds - start_model.num_boosted_rounds()

    xgboost.train(
        xp, dtrain, evals=evals,
        num_boost_round=iters,
        xgb_model=start_model,
        callbacks=[RayTrainReportCallback()],
    )


train_loop_config = {**xgb_params, "label_column": "label", "num_boost_round": num_boost_round}
print(f"XGB: {xgb_params}")
print(f"OMP: spark_conf(L1) + runtime_env(L2) + env+ctypes before import(L3)")
print(f"DataParallelTrainer: {num_workers}W x {cpus_per_worker}CPU")
```

### Cell 12: Train with MLflow tracking

```python
_mons = []
_omp = OmpDiagnosticsCollector.remote()
train_loop_config["_omp_diag_ref"] = _omp

with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
    run_id = run.info.run_id
    print(f"MLflow: {run_id} ({run_name})")

    # Start per-worker metrics monitors
    try:
        _mons = start_worker_monitors(run_id, databricks_host_url, databricks_token, num_executors)
        mlflow.log_param("worker_metrics_monitors", len(_mons))
    except Exception as e:
        print(f"WARN: monitors failed: {e}")
        mlflow.log_param("worker_metrics_monitors", 0)

    # Log params
    for k, v in {
        "training_mode": "ray_distributed",
        "data_size": data_size, "node_type": node_type,
        "run_mode": run_mode, "input_table": input_table,
        "n_rows": n_rows, "n_features": len(feature_columns),
        "num_workers": num_workers, "cpus_per_worker": cpus_per_worker,
        "num_boost_round": num_boost_round,
        "omp_fix": "spark_conf+runtime_env+env_before_import+ctypes+diag",
        "omp_target": cpus_per_worker,
    }.items():
        mlflow.log_param(k, v)
    for k, v in xgb_params.items():
        mlflow.log_param(f"xgb_{k}", v)
    mlflow.log_metric("ray_init_time_sec", ray_init_time)
    mlflow.log_metric("data_load_time_sec", load_time)
    mlflow.log_metric("split_time_sec", split_time)

    # Train
    try:
        print("\nTraining with DataParallelTrainer + OMP fix...")
        t0 = time.time()
        trainer = DataParallelTrainer(
            train_loop_per_worker=xgb_train_fn,
            train_loop_config=train_loop_config,
            scaling_config=scaling_config,
            run_config=run_config,
            datasets={"train": train_ray_ds, "valid": test_ray_ds},
            backend_config=XGBoostConfig(),
        )
        result = trainer.fit()
        train_time = time.time() - t0
        print(f"Training done in {train_time:.1f}s")
        mlflow.log_metric("train_time_sec", train_time)
    finally:
        stop_worker_monitors(_mons)
        _mons = []

    # Collect OMP diagnostics
    try:
        omp_diag = ray.get(_omp.get_all.remote(), timeout=15)
        print(f"\nOMP diagnostics ({len(omp_diag)} workers):")
        for rank in sorted(omp_diag):
            for k, v in sorted(omp_diag[rank].items()):
                print(f"  w{rank}/{k}: {v}")
                mlflow.log_param(f"omp_w{rank}_{k}", str(v)[:500])
    except Exception as e:
        print(f"WARN: OMP diag failed: {e}")

    # Extract model and evaluate
    import xgboost as xgb
    from mlflow.models import infer_signature
    t0 = time.time()
    booster = RayTrainReportCallback.get_model(result.checkpoint)

    # BUG-2 fix: signature is REQUIRED for Unity Catalog model registration
    sig = infer_signature(X_test_eval, booster.predict(xgb.DMatrix(X_test_eval)))
    mlflow.xgboost.log_model(
        xgb_model=booster,
        artifact_path="model",
        signature=sig,
        registered_model_name=uc_model_name,
    )
    mlflow.log_param("registered_model_name", uc_model_name)

    y_proba = booster.predict(xgb.DMatrix(X_test_eval))
    y_pred = (y_proba > 0.5).astype(int)
    pred_time = time.time() - t0
    mlflow.log_metric("predict_time_sec", pred_time)

    from sklearn.metrics import (average_precision_score, roc_auc_score,
        f1_score, precision_score, recall_score, confusion_matrix, classification_report)

    auc_pr = average_precision_score(y_test_eval, y_proba)
    auc_roc = roc_auc_score(y_test_eval, y_proba)
    f1 = f1_score(y_test_eval, y_pred)
    prec = precision_score(y_test_eval, y_pred, zero_division=0)
    rec = recall_score(y_test_eval, y_pred, zero_division=0)

    for name, val in [("auc_pr", auc_pr), ("auc_roc", auc_roc), ("f1", f1),
                      ("precision", prec), ("recall", rec)]:
        mlflow.log_metric(name, val)
        print(f"  {name}: {val:.4f}")

    cm = confusion_matrix(y_test_eval, y_pred)
    for name, val in [("true_negatives", cm[0,0]), ("false_positives", cm[0,1]),
                      ("false_negatives", cm[1,0]), ("true_positives", cm[1,1])]:
        mlflow.log_metric(name, val)

    print(classification_report(y_test_eval, y_pred, zero_division=0))

    total_time = ray_init_time + load_time + split_time + train_time + pred_time
    mlflow.log_metric("total_time_sec", total_time)
    print(f"\nDone: {run_name} | {total_time:.1f}s | OMP={cpus_per_worker} | {run_id}")
```

### Cell 13: Shutdown Ray cluster

```python
from ray.util.spark import shutdown_ray_cluster
print("Shutting down Ray cluster...")
shutdown_ray_cluster()
print("Ray cluster shutdown complete.")
```

### Cell 14: Exit

```python
import json
try:
    result = {
        "status": "ok" if not _notebook_errors else "error",
        "run_name": run_name,
        "run_id": run_id,
        "training_mode": "ray_distributed",
        "data_size": data_size,
        "node_type": node_type,
        "warehouse_id": warehouse_id,
        "n_rows": n_rows,
        "num_workers": num_workers,
        "cpus_per_worker": cpus_per_worker,
        "auc_pr": round(auc_pr, 4),
        "train_time_sec": round(train_time, 1),
        "total_time_sec": round(total_time, 1),
    }
    if _notebook_errors:
        result["errors"] = [e["error"] for e in _notebook_errors]
except NameError as e:
    result = {"status": "error", "error": str(e),
              "errors": [e["error"] for e in _notebook_errors] if _notebook_errors else []}

dbutils.notebook.exit(json.dumps(result))
```

## Key Architectural Patterns

### OMP_NUM_THREADS 3-Layer Fix

Without this fix, XGBoost uses 1 CPU core per worker regardless of `nthread` parameter.

| Layer | Where | Code | When Applied |
|-------|-------|------|-------------|
| L1 | Spark cluster config | `spark.executorEnv.OMP_NUM_THREADS: "15"` | JVM startup (most reliable) |
| L2 | Ray init | `ray.init(runtime_env={"env_vars": {"OMP_NUM_THREADS": "15"}})` | Worker process start |
| L3 | Worker train function | `os.environ` + `ctypes.CDLL("libgomp.so.1").omp_set_num_threads()` | Before `import xgboost` |

### WorkerMetricsMonitor

Zero-CPU Ray actor pinned to each worker node via `NodeAffinitySchedulingStrategy`. Runs `mlflow.system_metrics.SystemMetricsMonitor` to capture per-worker CPU/memory metrics like `system/worker_0/cpu_utilization_percentage`.

### OmpDiagnosticsCollector

Zero-CPU Ray actor that collects OMP state from each worker's `xgb_train_fn`. Workers call `diag_ref.report.remote(rank, diagnostics)` and the collector aggregates results. Logged as MLflow params: `omp_w{rank}_omp_before`, `omp_w{rank}_omp_set_to`, etc.

### Ray Data Loading

Uses `ray.data.read_databricks_tables(warehouse_id=..., query=...)` to load data directly from Unity Catalog via a SQL Warehouse. Requires the `warehouse_id` widget. No Spark-to-pandas conversion.

## DAB Job Definition

```yaml
train_xgb_ray_d16:
  name: "[${var.env}] Train XGBoost Ray (D16s_v5 x 4)"
  parameters:
    - name: data_size
      default: medium
    - name: node_type
      default: D16sv5
    - name: run_mode
      default: full
    - name: num_workers
      default: "0"
    - name: cpus_per_worker
      default: "0"
    - name: warehouse_id
      default: "148ccb90800933a1"
    - name: table_name
      default: ""
  tasks:
    - task_key: train
      notebook_task:
        notebook_path: ./notebooks/train_xgb_ray.ipynb
        base_parameters:
          data_size: "{{job.parameters.data_size}}"
          node_type: "{{job.parameters.node_type}}"
          run_mode: "{{job.parameters.run_mode}}"
          num_workers: "{{job.parameters.num_workers}}"
          cpus_per_worker: "{{job.parameters.cpus_per_worker}}"
          warehouse_id: "{{job.parameters.warehouse_id}}"
          table_name: "{{job.parameters.table_name}}"
          catalog: ${var.catalog}
          schema: ${var.schema}
        source: WORKSPACE
      new_cluster:
        spark_version: "17.3.x-cpu-ml-scala2.13"
        node_type_id: "Standard_D16s_v5"
        num_workers: 4
        data_security_mode: SINGLE_USER
        spark_conf:
          spark.executorEnv.OMP_NUM_THREADS: "15"
        custom_tags:
          ResourceClass: default
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

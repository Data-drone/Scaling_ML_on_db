# Error Diagnosis Reference

How to retrieve and interpret errors from failed Databricks job runs. The Databricks Jobs API wraps all notebook failures as `RUN_EXECUTION_ERROR` with `"Workload failed, see run output for details"`. Getting the actual traceback requires a specific 4-step API call pattern.

---

## 4-Step Error Retrieval

### Step 1: Get the parent run ID

From `databricks bundle run` output, or from the Databricks UI (Jobs > click job > click failed run > copy run ID from URL).

For debug-level deploy logs:
```bash
databricks bundle deploy -t dev --debug --log-file /tmp/bundle.log
grep -E "RUN_EXECUTION_ERROR|run_id|job_id" /tmp/bundle.log | tail -20
```

### Step 2: Get task-level run IDs

**Critical:** `get-run-output` does NOT work on multi-task parent runs. You MUST get the task-level `run_id` first.

```bash
# CLI
databricks jobs get-run <PARENT_RUN_ID> --output json | \
  jq -r '.tasks[] | "\(.task_key)\trun_id=\(.run_id)\tstate=\(.state.result_state // "?")"'
```

```bash
# REST API (for automation)
curl -s -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  "${DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id=${PARENT_RUN_ID}" | \
  jq '.tasks[] | {task_key, run_id, result: .state.result_state}'
```

### Step 3: Get actual error and stack trace

```bash
# CLI
databricks jobs get-run-output <TASK_RUN_ID> --output json | jq '{error, error_trace}'
```

```bash
# REST API
curl -s -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  "${DATABRICKS_HOST}/api/2.1/jobs/runs/get-output?run_id=${TASK_RUN_ID}" | \
  jq '{error, error_trace}'
```

**Key fields returned:**
- `error` -- the exception message (e.g., `RuntimeError: ...`)
- `error_trace` -- full Python traceback with line numbers
- `notebook_output.result` -- the `dbutils.notebook.exit()` value (success only)

### Step 4: If error_trace is empty -- check infrastructure

Some failures do not produce notebook-level errors (cluster crash, OOM, spot preemption). Check cluster events:

```bash
# Get cluster_id from the run details
CLUSTER_ID=$(curl -s -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  "${DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id=${TASK_RUN_ID}" | \
  jq -r '.tasks[0].cluster_instance.cluster_id // .cluster_instance.cluster_id')

# Get cluster events
curl -s -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/clusters/events" \
  -d "{\"cluster_id\": \"${CLUSTER_ID}\", \"limit\": 20}" | \
  jq '.events[] | {type, timestamp, details}'
```

If cluster events are also empty, check the Spark UI driver logs manually (Jobs UI > failed task > "View driver logs").

---

## API Gotchas

1. **API version 2.0 vs 2.1**: `GET /api/2.0/jobs/list` returns `INVALID_PARAMETER_VALUE`. Always use `GET /api/2.1/jobs/list` (API 2.1+).

2. **Multi-task parent runs**: Calling `get-run-output` with a parent run_id (a multi-task job) returns `INVALID_PARAMETER_VALUE`. You MUST use the task-level `run_id` from `tasks[].run_id`.

3. **ANSI escape codes**: `error_trace` contains ANSI color codes. Strip them for clean output:
   ```python
   import re
   clean_trace = re.sub(r'\x1b\[[0-9;]*m', '', raw_trace)
   ```

4. **Cluster events API**: Uses `POST /api/2.0/clusters/events` (not GET). The body must be JSON with `cluster_id`.

---

## Programmatic Error Retrieval (Python)

Simplified pattern from `src/crash_retriever.py` for agent use:

```python
import re
import requests

HOST = os.environ["DATABRICKS_HOST"]  # or hardcode your workspace URL
TOKEN = os.environ["DATABRICKS_TOKEN"]  # or use dbutils.secrets
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def get_run_errors(parent_run_id: int) -> list[dict]:
    """Get error details for all failed tasks in a job run."""
    # Step 1: Get task run IDs from parent
    resp = requests.get(
        f"{HOST}/api/2.1/jobs/runs/get",
        headers=HEADERS,
        params={"run_id": parent_run_id},
    ).json()

    errors = []
    for task in resp.get("tasks", []):
        if task.get("state", {}).get("result_state") != "FAILED":
            continue

        # Step 2: Get error output for each failed task
        task_out = requests.get(
            f"{HOST}/api/2.1/jobs/runs/get-output",
            headers=HEADERS,
            params={"run_id": task["run_id"]},
        ).json()

        # Strip ANSI codes from trace
        raw_trace = task_out.get("error_trace", "")
        clean_trace = re.sub(r"\x1b\[[0-9;]*m", "", raw_trace)

        errors.append({
            "task_key": task.get("task_key"),
            "task_run_id": task["run_id"],
            "error": task_out.get("error", ""),
            "error_trace": clean_trace,
            "cluster_id": task.get("cluster_instance", {}).get("cluster_id"),
        })

    return errors


def get_cluster_events(cluster_id: str) -> list[dict]:
    """Get cluster events for infrastructure failure diagnosis."""
    resp = requests.post(
        f"{HOST}/api/2.0/clusters/events",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={
            "cluster_id": cluster_id,
            "limit": 20,
            "event_types": [
                "TERMINATING", "TERMINATED", "DRIVER_NOT_RESPONDING",
                "NODES_LOST",
            ],
        },
    ).json()
    return resp.get("events", [])
```

For the full-featured crash retriever with report generation, use `src/crash_retriever.py`:
```python
from src.crash_retriever import CrashRetriever

cr = CrashRetriever(host=HOST, token=TOKEN)
report = cr.diagnose_run(run_id=12345)
print(report.summary())
print(f"Category: {report.crash_category}")  # SPOT_PREEMPTION, OUT_OF_MEMORY, CONFIG_ERROR, etc.
```

---

## Known RuntimeError Patterns (RE1--RE5)

These are the actual errors observed across 113 failed runs in this project. When you see one, apply the fix directly.

### RE1: Ray Data SQL query failure

**Signature:**
```
RuntimeError: Query 'SELECT * FROM brian_gen_ai.xgb_scaling.imbalanced_100m' execution failed.
```

**Where:** `ray.data.read_databricks_tables()` in `train_xgb_ray.ipynb`

**Root cause:** The SQL warehouse hit a limit (concurrency, timeout, or size). `DatabricksUCDatasource` only checks `state != "SUCCEEDED"` and does not surface the inner SQL error.

**Fix checklist:**
1. Verify warehouse `148ccb90800933a1` is running: Databricks UI > SQL Warehouses
2. Check if the table exists: `spark.sql("SELECT COUNT(*) FROM brian_gen_ai.xgb_scaling.imbalanced_100m")`
3. For 100M+ rows, the SQL warehouse query may timeout. Switch to Spark-based loading:
   ```python
   # Instead of ray.data.read_databricks_tables(warehouse_id=..., query=...)
   spark_df = spark.read.table(input_table)
   ray_ds = ray.data.from_spark(spark_df)
   ```
4. If you must use the SQL warehouse, increase warehouse size to Medium/Large

**Frequency:** 5 failures (Mar 5-6, all on 100M Plasma jobs)

---

### RE2: MLflow model signature missing (Unity Catalog)

**Signature:**
```
MlflowException: Model passed for registration did not contain any signature metadata.
All models in the Unity Catalog must be logged with a model signature.
```

**Where:** `mlflow.xgboost.log_model()` in `train_xgb_ray.ipynb`

**Root cause:** Unity Catalog requires explicit model signatures. The `log_model()` call is missing `signature=`.

**Fix:**
```python
from mlflow.models import infer_signature
import xgboost as xgb

# After training, before logging:
sig = infer_signature(X_test_eval, booster.predict(xgb.DMatrix(X_test_eval)))
mlflow.xgboost.log_model(
    xgb_model=booster,
    artifact_path="model",
    signature=sig,                          # <-- required for UC
    registered_model_name=uc_model_name,
)
```

**Frequency:** 2 failures (Mar 5, Ray 1M 2W D8)

---

### RE3: Ray not initialized on GPU ML Runtime

**Signature:**
```
RaySystemError: System error: Ray has not been started yet.
You can start Ray with 'ray.init()'.
```

**Where:** `ray.cluster_resources()` call after `setup_ray_cluster()` in GPU notebook

**Root cause:** `setup_ray_cluster()` silently fails or does not connect on GPU ML Runtime (`17.3.x-gpu-ml-scala2.13`). The standard `ray.shutdown() + ray.init(runtime_env=...)` pattern also hangs on GPU clusters.

**Fix:**
```python
from ray.util.spark import setup_ray_cluster
setup_ray_cluster(...)

# Add explicit check before using Ray:
import ray
if not ray.is_initialized():
    print("WARNING: setup_ray_cluster failed silently, attempting fallback...")
    ray.init(ignore_reinit_error=True)
    if not ray.is_initialized():
        raise RuntimeError("Ray failed to initialize on GPU cluster")

cluster_resources = ray.cluster_resources()  # Now safe
```

**Note:** On GPU clusters, you may need `num_gpus_per_worker=1` in `setup_ray_cluster()`.

**Frequency:** 11 failures (Feb 13, all GPU jobs)

---

### RE4: GTE serving endpoint deployment failures

**Signature (multiple sub-patterns):**

| Error | Cause |
|-------|-------|
| `RuntimeError: Endpoint deployment failed: UPDATE_FAILED` | Model artifact or dependency issue in serving environment |
| `RuntimeError: Endpoint was deleted -- deployment failed` | Endpoint cleaned up during deploy (race condition or manual delete) |
| `RuntimeError: Pyfunc endpoint failed: 404` | Endpoint not found after deployment (deleted or failed to create) |
| `ImportError: attempted relative import with no known parent package` | Custom model code uses relative imports that don't resolve in serving env |
| `ModuleNotFoundError: No module named 'torch'` | `torch` not in `pip_requirements` of `mlflow.pyfunc.log_model()` |
| `FileNotFoundError: modeling_new.py` | Model artifacts incomplete -- custom model files missing |

**General fix for pyfunc model serving:**
1. Always specify `pip_requirements=` explicitly (do not rely on auto-detection)
2. Use absolute imports in custom model code
3. Verify artifacts before deploying: `mlflow.artifacts.list_artifacts(run_id, "model/")`
4. Consider using `mlflow.transformers.log_model()` instead of raw `pyfunc` for HuggingFace models

**Frequency:** 23 failures (Mar 2-4)

---

### RE5: Empty error_trace (infrastructure or internal failure)

**Signature:** `error_trace` field is empty or missing in `get-run-output` response.

| Symptom | Likely cause |
|---------|-------------|
| Empty trace + short execution_duration (<30s) | Cluster startup failure or spot preemption |
| Empty trace + execution ~60-120s + cluster healthy | Notebook internal crash — likely env var loss after `ray.init(runtime_env=...)` causing `read_databricks_tables` to fail (see G9 in gotchas). The crash produces no trace because the error occurs in a native C extension. |
| Empty trace + cluster terminated message | User cancellation or timeout |
| Empty trace + "unable to access notebook" message | Notebook path wrong or workspace drift (L15) |

**Debug approach:**
1. Check cluster events API (see Step 4 above)
2. If cluster events show `JOB_FINISHED` (cluster was healthy), the issue is inside the notebook
3. Create a diagnostic notebook with per-cell checkpoints and `dbutils.notebook.exit()` to isolate which cell crashes
4. Common culprit for Ray notebooks: `DATABRICKS_TOKEN` env var lost after `ray.shutdown()` + `ray.init(runtime_env=...)`. Fix: re-assert `os.environ["DATABRICKS_TOKEN"]` before `ray.data.read_databricks_tables()`
5. Check if notebook path exists: `databricks workspace get-status <NOTEBOOK_PATH>`

**Frequency:** ~41 failures (Feb, mostly older scaling experiments) + 2 failures (Mar 12, Ray env var loss)

---

## Decision Tree: Which Error Am I Seeing?

```
error_trace is not empty?
  |
  +-- YES --> Read the traceback:
  |     |
  |     +-- Contains "Query '...' execution failed"  --> RE1 (SQL warehouse)
  |     +-- Contains "signature metadata"             --> RE2 (MLflow signature)
  |     +-- Contains "Ray has not been started"       --> RE3 (Ray init)
  |     +-- Contains "UPDATE_FAILED" or "pyfunc"      --> RE4 (serving endpoint)
  |     +-- Contains "NotebookImportException"        --> L13 (__init__.py comment)
  |     +-- Other Python traceback                    --> Read traceback for root cause
  |
  +-- NO --> Infrastructure failure (RE5):
        |
        +-- Check cluster events for TERMINATING/NODES_LOST
        +-- Check if short duration (<30s) --> likely startup failure
        +-- Re-deploy bundle to fix workspace drift
```

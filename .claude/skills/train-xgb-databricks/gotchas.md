# Databricks Gotchas for XGBoost Training

## G1: OMP_NUM_THREADS=1 (Critical — 3.4x performance impact)

Databricks silently sets `OMP_NUM_THREADS=1` on Spark executors. XGBoost's C++ layer caps threads at `min(nthread, omp_get_max_threads())`, so your `nthread=14` silently becomes 1.

**Fix (3 layers):**
1. `spark.executorEnv.OMP_NUM_THREADS: "15"` in cluster Spark config (most important)
2. `ray.init(runtime_env={"env_vars": {"OMP_NUM_THREADS": "15"}})`
3. Worker-level: `os.environ["OMP_NUM_THREADS"]` + ctypes BEFORE `import xgboost`

Layer 1 is most important — layers 2 and 3 may be too late if libgomp is already loaded.

**Root cause chain:**
```
Databricks sets OMP_NUM_THREADS=1 on Spark executors
  → Ray workers inherit OMP_NUM_THREADS=1
    → XGBoost loads vendored libgomp
      → libgomp reads OMP_NUM_THREADS=1 at init
        → omp_get_max_threads() returns 1
          → XGBoost C++: nthread = min(param, 1) = 1
```

## G2: ML Runtime suffix required

Use `17.3.x-cpu-ml-scala2.13` (CPU) or `17.3.x-gpu-ml-scala2.13` (GPU).
The plain `-scala2.13` runtime does NOT include XGBoost, Ray, or MLflow.

## G3: src/__init__.py must start with docstring, not comment

If `src/__init__.py` starts with `# comment`, Databricks DBR 17.3 misidentifies it as a notebook → `NotebookImportException` blocks all imports.

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

## G7: Azure Spot preemption on large VMs

E32s_v5 spot instances get preempted frequently. Use `SPOT_WITH_FALLBACK_AZURE` or pin to on-demand for long experiments (> 30 min).

## G8: Model signature required for Unity Catalog

`mlflow.*.log_model()` to Unity Catalog MUST include `signature=`. Without it:
```
MlflowException: Model passed for registration did not contain any signature metadata.
```

Fix:
```python
from mlflow.models import infer_signature
sig = infer_signature(X_sample, model.predict(X_sample))
mlflow.sklearn.log_model(model, "model", signature=sig, registered_model_name=uc_name)
```

## G9: DATABRICKS_HOST format for ray.data.read_databricks_tables

`ray.data.read_databricks_tables()` reads `DATABRICKS_HOST` from env vars to construct API URLs. If `DATABRICKS_HOST` includes the `https://` prefix (as Databricks contexts return it), Ray constructs `https://https://adb-...` which resolves `host='https'` and fails with:
```
HTTPSConnectionPool(host='https', port=443): Max retries exceeded
```

Also: after `ray.shutdown()` + `ray.init(runtime_env=...)`, the driver process may lose `DATABRICKS_TOKEN` and `DATABRICKS_HOST`. The `runtime_env.env_vars` only propagate to *workers*, not the driver.

The error trace from the Jobs API will be *empty* (RE5 pattern) because the notebook crashes internally.

**Fix:** Re-assert env vars immediately before `read_databricks_tables()`, and strip the protocol prefix from HOST:
```python
host_for_ray = databricks_host_url.replace("https://", "").replace("http://", "")
os.environ["DATABRICKS_HOST"] = host_for_ray
os.environ["DATABRICKS_TOKEN"] = databricks_token
```

## G10: XGBoost 2.0+ deprecates gpu_hist and gpu_id

Starting with XGBoost 2.0 (Oct 2023), `tree_method="gpu_hist"` and `gpu_id` are deprecated. Use instead:
```python
xgb_params = {
    "tree_method": "hist",            # NOT "gpu_hist"
    "device": f"cuda:{selected_gpu}", # NOT "gpu_id"
}
```
The old params still work via a compatibility shim in the sklearn wrapper on DBR 17.3, but emit deprecation warnings and may break in future XGBoost releases.

## G11: Executor registration delay on new job clusters

When a Databricks job spins up a new cluster, executors may not be registered with the Spark context immediately. Code that reads `sc._jsc.sc().getExecutorMemoryStatus().size()` right at notebook start may see 0 executors.

**Fix:** Retry with backoff:
```python
for attempt in range(5):
    num_executors = sc._jsc.sc().getExecutorMemoryStatus().size() - 1
    if num_executors >= 1:
        break
    time.sleep(10)
```

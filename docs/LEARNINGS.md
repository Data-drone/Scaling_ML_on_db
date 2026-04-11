# Learnings: Scaling XGBoost on Databricks

Accumulated learnings from benchmarking and debugging distributed XGBoost training on Databricks with Ray on Spark. This is a living document — add new findings as experiments progress.

---

## Critical Findings

### L1: OMP_NUM_THREADS=1 is silently set by Databricks (3.4x speedup when fixed)

**Severity:** Critical
**Track:** Ray Scaling, Single-Node Scaling
**Date discovered:** During initial Ray distributed experiments
**Status:** RESOLVED — fix documented below

**Problem:**
Databricks/Spark silently sets `OMP_NUM_THREADS=1` on executor JVMs. Ray workers inherit this environment variable. Because XGBoost's C++ layer caps the effective thread count at `min(nthread_param, omp_get_max_threads())`, setting `nthread=14` in XGBoost params has NO effect — it silently uses 1 thread per worker.

**Root Cause Chain:**
```
Databricks sets OMP_NUM_THREADS=1 on Spark executors
    → Ray on Spark workers inherit OMP_NUM_THREADS=1
        → XGBoost loads its vendored libgomp-25c89faf.so.1.0.0
            → libgomp reads OMP_NUM_THREADS=1 at initialization
                → omp_get_max_threads() returns 1
                    → XGBoost C++: nthread = min(nthread_param, omp_get_max_threads()) = 1
```

**Impact:**
| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| CPU utilization per worker | 6.5-7.2% | 68-72% | 10x |
| Train time (10M rows, 4 workers) | 272-298s | 79-83s | 3.4x faster |
| Cores used per worker | 1 of 15 | 14 of 15 | 14x |

**Fix (3 layers of defence):**
1. **Spark config** (most important): `spark.executorEnv.OMP_NUM_THREADS: "15"` in spark_conf
2. **Ray runtime_env**: `runtime_env={"env_vars": {"OMP_NUM_THREADS": "15"}}`
3. **Worker-level**: `os.environ["OMP_NUM_THREADS"]` + `ctypes.CDLL("libgomp.so.1").omp_set_num_threads()` BEFORE `import xgboost`

**Why layer 1 is most important:** Layers 2 and 3 may be too late if libgomp is already initialised by a transitive dependency (e.g., numpy, scipy). The Spark config sets the env var at JVM startup — before any Python process.

---

### L2: XGBoost bundles its own OpenMP runtime

**Track:** Ray Scaling
**Status:** Informational — explains why OMP_NUM_THREADS is the only reliable control

XGBoost ships a vendored `libgomp` inside `xgboost.libs/` (e.g., `libgomp-25c89faf.so.1.0.0`). The system `libgomp.so.1` and XGBoost's bundled copy are separate shared libraries with separate internal state.

**Implication:** You cannot use `omp_set_num_threads()` on the system libgomp and expect XGBoost's bundled copy to pick it up. `OMP_NUM_THREADS` is read at library load time by ALL libgomp instances, making it the only reliable cross-library control.

**To verify:** Check `omp_w{rank}_xgb_omp_linked_libs` MLflow params — they show the actual library path XGBoost is using.

---

### L3: Plasma object store tuning has minimal impact at 10M scale

**Track:** Ray Plasma Tuning
**Status:** Confirmed across 34 experiments

Across 34 plasma tuning experiments varying object store memory (0-40 GB), heap memory (0-60 GB), and spill settings, training time was essentially flat once OMP was fixed.

| Object Store | Heap Memory | Allow Slow | Train Time |
|-------------|-------------|-----------|-----------|
| Default (~19 GB) | Default | No | 80s |
| 8 GB | Default | No | 80s |
| 12 GB | Default | No | 80s |
| 24 GB | Default | Yes | 80s |
| 24 GB | 20 GB | Yes | 80s |
| 40 GB (E16) | Default | Yes | 80s |

**Hypothesis to test:** Object store tuning may become critical at 100M+ rows where data exceeds available memory. Need to test with `large` and `xlarge` presets.

---

### L4: Worker scaling shows super-linear speedup (2 → 4 workers)

**Track:** Ray Scaling
**Status:** Observed but not yet fully explained

| Workers | Node Type | Train Time (10M) | Speedup vs 2W |
|---------|-----------|-------------------|---------------|
| 2 | D16s_v5 | ~510s | 1.0x |
| 4 | D16s_v5 | ~80s | 6.4x |
| 4 | E16s_v5 | ~79s | 6.5x |

6.4x speedup from 2x workers is super-linear. Likely explanation: each worker's data shard halves, improving L2/L3 cache locality for XGBoost's histogram-based tree method. The histogram bins fit in cache at 4 workers but not at 2.

**Learning:** Don't model distributed XGBoost scaling as purely compute-bound. Cache effects dominate at the boundary between "fits in cache" and "doesn't fit."

---

### L5: Worker CPU ~70% is expected, not a bottleneck

**Track:** Ray Scaling, Single-Node Scaling
**Status:** Understood

After the OMP fix, workers average ~70% CPU utilization, not 100%. This is expected overhead:
- Ray Data materialisation (converting distributed shards to Pandas/DMatrix)
- Allreduce synchronisation between workers after each boosting round
- Memory allocation for DMatrix construction
- XGBoost's histogram quantile sketch phase (I/O-bound, not CPU-bound)

**Not worth optimising:** The ~30% overhead is structural. Focus on scaling experiments, not squeezing more CPU.

---

### L6: Per-worker system metrics are essential for distributed debugging

**Track:** Ray Scaling
**Status:** Working — implemented as Ray actors

Standard MLflow system metrics only capture the **driver node**. For distributed training, you need per-worker CPU/memory/disk metrics to diagnose issues like:
- OMP_NUM_THREADS=1 (only visible as low worker CPU)
- Memory pressure causing spilling
- Network bottlenecks during allreduce

**Implementation:** `WorkerMetricsMonitor` Ray actors (one per worker node) running `mlflow.system_metrics.SystemMetricsMonitor` with a per-node prefix:
- `system/worker_0/cpu_utilization_percentage`
- `system/worker_0/system_memory_usage_megabytes`
- etc.

**Key learning:** Without these metrics, we would not have detected the OMP issue. Always enable per-worker metrics in distributed experiments.

---

### L7: `OmpDiagnosticsCollector` pattern for collecting worker state

**Track:** Ray Scaling
**Status:** Working — zero-CPU Ray actor

Since Ray worker stdout is not accessible via the Databricks REST API, we use a zero-CPU Ray actor to collect OMP state from each worker:

```python
@ray.remote(num_cpus=0)
class OmpDiagnosticsCollector:
    def __init__(self):
        self._results = {}
    def report(self, worker_rank, diagnostics):
        self._results[worker_rank] = diagnostics
    def get_all(self):
        return dict(self._results)
```

Logged as MLflow params: `omp_w{rank}_omp_env_at_start`, `omp_w{rank}_ctypes_libgomp.so.1`, etc.

**Pattern is reusable:** Any per-worker diagnostic can be collected this way — GPU info, library versions, memory layout, etc.

---

## Data Generation Learnings

### L8: Batched column generation prevents Spark lineage explosion

**Track:** Data Generation
**Status:** Working

When generating 250+ columns of synthetic data, adding all columns in a single `select()` causes Spark to build an enormous logical plan. This leads to:
- StackOverflowError in the Catalyst optimizer
- Very slow plan compilation (minutes)
- OOM on the driver

**Fix:** Batch `select()` calls (50 columns at a time) with `localCheckpoint()` every 200 features. `localCheckpoint()` materialises the DataFrame and cuts the lineage, giving Spark a clean starting point.

**Lesson:** Any wide-table generation on Spark needs lineage management. Plan for batched column creation from the start.

---

### L9: Synthetic data class imbalance ratio matters for benchmark validity

**Track:** Data Generation
**Status:** Informational

All datasets use 5% minority class (95/5 split). This is deliberate:
- Forces models to learn from limited positive examples
- AUC-PR is more informative than AUC-ROC at this imbalance
- Matches real-world fraud/anomaly detection use cases

**Note:** Perfect AUC-PR=1.0 on 10M rows suggests the informative features may be too strong. Consider reducing `n_informative_num` for more realistic benchmarks.

---

## Infrastructure Learnings

### L10: Databricks Asset Bundles require specific ML runtime versions

**Track:** All
**Status:** Working

- ML Runtime: `17.3.x-cpu-ml-scala2.13` (includes Ray 2.37.0, XGBoost 2.1.x)
- Non-ML runtime: `17.3.x-scala2.13` (for data generation only)
- GPU runtime: `16.2.x-gpu-ml-scala2.12` (tested — required for V100/T4 GPU clusters)
- Unity Catalog: Requires `data_security_mode: SINGLE_USER`

**Gotcha:** The `-cpu-ml-` suffix is required for the ML Runtime. Using the plain `-scala2.13` runtime will NOT have XGBoost, Ray, or MLflow pre-installed.

---

### L11: Azure Spot instances fail frequently on large VMs

**Track:** All
**Status:** Use SPOT_WITH_FALLBACK_AZURE

E32s_v5 spot instances get preempted more frequently than D8s_v5. For long-running experiments (>30 min), consider:
- Using `SPOT_WITH_FALLBACK_AZURE` (current setting) — falls back to on-demand
- Pinning to on-demand for critical benchmark runs
- Using smaller nodes with more workers (D8 x 8 vs E32 x 2)

---

### L12: Ray on Spark worker stdout is not accessible via REST API

**Track:** Ray Scaling
**Status:** Workaround in place (OmpDiagnosticsCollector)

Databricks REST API (`/api/2.0/clusters/get-output`, DBFS driver logs) only returns the driver's stdout/stderr. Ray worker stdout is lost unless explicitly collected via:
1. Ray actors reporting back to driver (current approach)
2. Worker-side file writes to DBFS/Unity Catalog
3. Structured logging to a shared sink

**Impact on crash retrieval:** When a Ray worker crashes, the crash traceback may only appear in Spark executor logs (accessible via Spark UI but NOT via REST API programmatically). Need to use cluster events API + executor log paths for post-mortem analysis.

---

### L13: `src/__init__.py` starting with `#` comment triggers NotebookImportException

**Severity:** Critical — blocks ALL notebook imports of `src` module
**Track:** All (Single-Node, Ray, GPU)
**Date discovered:** 2026-02-22 smoke test failures
**Status:** RESOLVED

**Problem:**
When `src/__init__.py` is deployed via DAB to the Databricks workspace and starts with a `#` comment line (e.g., `# Scaling XGBoost on Databricks — shared utilities`), Databricks DBR 17.3's workspace import machinery misidentifies it as a notebook file. Any `from src.xxx import yyy` then fails with:

```
NotebookImportException: Unable to import module `src`.
The following file appears to be a notebook:
/Workspace/Users/.../src/__init__.py
Importing notebooks directly is not supported.
```

**Fix:** Use a triple-quoted docstring instead of a comment:
```python
# BAD — triggers NotebookImportException
# Scaling XGBoost on Databricks — shared utilities

# GOOD — Databricks recognises this as a plain Python file
"""Scaling XGBoost on Databricks - shared utilities."""
```

**Root cause:** Databricks uses heuristics to distinguish notebooks from plain `.py` files. A file starting with `#` followed by text looks like a `# Databricks notebook source` header line. Using a docstring avoids this ambiguity.

---

### L14: Databricks Jobs API requires task-level run_id for error details

**Severity:** High — without this, all you get is "Workload failed, see run output for details"
**Track:** All (CI/CD, agent automation)
**Date discovered:** 2026-02-22
**Status:** Documented

**Problem:**
`GET /api/2.1/jobs/runs/get` only returns a generic error message. The actual stack trace requires a separate `GET /api/2.1/jobs/runs/get-output` call with the **task-level** `run_id` (not the parent job run_id).

**The two-step pattern:**
```bash
# Step 1: Get task-level run IDs
curl -s "${HOST}/api/2.1/jobs/runs/get?run_id=${PARENT_RUN_ID}" \
  -H "Authorization: Bearer ${TOKEN}" | jq '.tasks[].run_id'

# Step 2: Get actual error for each task
curl -s "${HOST}/api/2.1/jobs/runs/get-output?run_id=${TASK_RUN_ID}" \
  -H "Authorization: Bearer ${TOKEN}" | jq '{error, error_trace}'
```

**Key fields returned by get-output:**
- `error`: The error message string (e.g., `NotebookImportException: ...`)
- `error_trace`: Full Python stack trace
- `notebook_output.result`: The notebook's `dbutils.notebook.exit()` value (on success)

**Gotcha:** Calling `get-output` with a multi-task parent run_id returns `INVALID_PARAMETER_VALUE`. You MUST use the task-level run_id found in `tasks[].run_id`.

**If `error_trace` is empty:** Check cluster events API (`GET /api/2.0/clusters/events?cluster_id=...`) for infrastructure failures (OOM, spot preemption, driver crash).

---

### L15: Local repo vs workspace code drift is a silent killer

**Track:** All (CI/CD)
**Date discovered:** 2026-02-22
**Status:** Ongoing risk

When multiple agents (Codex, Claude, manual edits) push code to the workspace independently, the deployed code can diverge from the local git repo. DAB `databricks bundle deploy` syncs local → workspace, but direct workspace API writes (`workspace/import`) bypass the bundle entirely.

**Symptoms:**
- Notebook imports modules that don't exist in the local repo
- Config APIs have different function signatures
- Tests pass locally but notebooks fail on Databricks

**Best practices:**
1. Always deploy via `databricks bundle deploy` (single source of truth)
2. Use `workspace/list` + `workspace/export` APIs to verify deployed state
3. Add a version hash check in notebook cell 1 to detect drift
4. Never edit workspace files directly — always commit to git first

---

### L16: vCPU detection regex must handle NC-series GPU node types

**Severity:** High — causes immediate crash on GPU clusters
**Track:** Ray GPU
**Date discovered:** 2026-04-13
**Status:** RESOLVED

**Problem:**
The notebook's vCPU auto-detection regex `re.search(r"[de](\d+)", node_type)` only matches D-series (D16sv5 → 16) and E-series (E16sv5 → 16) node types. NC-series GPU nodes (NC6sv3, NC8asT4v3) don't have a `d` or `e` prefix before the digit count. The regex returns None, falling back to a default of 8 vCPUs.

For NC6s_v3 (6 vCPUs), this means `allocatable_cpus_per_node = 8 - 1 = 7`, and `setup_ray_cluster(num_cpus_worker_node=7)` requests 7 CPUs on a 6-core node → immediate crash: `ValueError: cpu number per Ray worker node should be <= spark worker node CPU cores`.

**Fix:** Chain a second regex for NC-series:
```python
node_vcpus_match = re.search(r"[de](\d+)", node_type_lower) or re.search(r"nc(\d+)", node_type_lower)
```

---

### L17: Azure GPU VM quota limits determine available GPU families

**Track:** GPU
**Date discovered:** 2026-04-13
**Status:** Informational

Azure subscriptions have per-family GPU core quotas. On this workspace:
- **NCsv3 (V100):** 100 cores — enough for 9× NC6s_v3 (54 cores) or 8× NC12s_v3 (96 cores)
- **NCASv3_T4 (T4):** 0 cores allocated — all T4 runs fail with AZURE_QUOTA_EXCEEDED_EXCEPTION
- **NC_A100_v4:** No quota reported (likely 0)
- **NC_H100_v5:** 0 cores confirmed

**Impact:** You can only use V100 for multi-node GPU training. The 100-core V100 quota supports one 9-node cluster at a time (54 cores); a second concurrent GPU cluster would need ≤46 cores.

**To check quota:** Use the Databricks `clusters/list-node-types` API — `node_info.available_core_quota` shows remaining cores per family.

### L18: Concurrent GPU clusters cause Azure quota contention (partial executor registration)

**Severity:** High — silently degrades training performance
**Track:** Ray GPU
**Date discovered:** 2026-04-15
**Status:** RESOLVED — root cause identified

**Problem:**
GPU benchmark runs showed only 4-5 of 8 requested workers registering. This appeared to be a Spark/Ray executor registration issue on GPU ML Runtime, but was actually Azure quota contention.

**Root Cause:**
Two GPU benchmark jobs (10M and 30M) were submitted simultaneously, each requesting 8 workers + 1 driver = 9× NC6s_v3 = 54 vCPUs. Combined need: 108 cores. Available NCsv3 quota: 100 cores.

**Evidence from cluster event logs:**
- **10M cluster:** Started with 5 workers (quota allowed 36 cores initially). AUTORECOVERY eventually scaled to 8 after 30M cluster stabilized.
- **30M cluster:** Started with 4 workers. Resize to 8 failed: `AZURE_QUOTA_EXCEEDED_EXCEPTION` — "Current Usage: 96, Additional Required: 18, New Limit Required: 114". Stuck at 4-5 workers for entire run.

**Fix:** Run GPU benchmarks sequentially, not concurrently. Alternatively, request ≤6 workers per cluster (7 nodes × 6 vCPUs = 42 cores; two clusters = 84 < 100 quota).

**Key insight:** Databricks does NOT fail the job when Azure can't provision all requested workers. Instead, it starts the cluster with partial workers, begins the notebook, then attempts auto-recovery in the background. The notebook's `sc._jsc.sc().getExecutorMemoryStatus().size() - 1` captures whatever partial count is available at that moment.

---

### L19: Autoresearch Ray pipeline produced near-random models (AUC-PR ~0.02)

**Severity:** Critical — silently produced garbage results for 10M and 30M datasets
**Track:** Autoresearch Skill, Ray Scaling
**Date discovered:** 2026-04-11
**Status:** RESOLVED — sanity check added to Phase 4, experiments re-run via dedicated notebook

**Problem:**
The autoresearch-xgb skill generated Ray distributed training code for the 10M and 30M datasets that produced AUC-PR ~0.02-0.03 (near random chance for a ~2% minority class). The dedicated `train_xgb_ray.ipynb` notebook produces AUC-PR=1.0 on the same data with the same Ray pipeline.

**Impact:**
| Pipeline | Dataset | AUC-PR | AUC-ROC | Status |
|----------|---------|--------|---------|--------|
| Autoresearch (broken) | 10M | 0.024-0.032 | 0.55-0.65 | Near random |
| Autoresearch (broken) | 30M | 0.023-0.029 | 0.54-0.63 | Near random |
| Dedicated notebook (correct) | 10M | 1.0 | 1.0 | Perfect |
| Dedicated notebook (correct) | 30M | 1.0 | 1.0 | Perfect |

**Root cause:**
The autoresearch agent dynamically generates training code at runtime via the Databricks Command Execution API. The generated code diverged from the tested `train_xgb_ray.ipynb` in ways that broke model quality. Most likely causes (the exact generated code is not preserved):

1. **API parameter mismatch:** Using sklearn-style `learning_rate` instead of native API `eta` in the `xgboost.train()` call, causing the parameter to be silently ignored (defaults to 0.3)
2. **Feature column ordering mismatch:** Using `sorted()` in `shard_to_dmatrix()` but schema-order in the eval path, making predictions appear random
3. **DMatrix construction bug:** Feature/label columns swapped or feature matrix built incorrectly in the generated `shard_to_dmatrix()` function

**Fix (3 parts):**
1. **Sanity check in Phase 4:** After baseline training, verify AUC-PR > 0.1 for binary tasks. If near random chance, halt and flag as pipeline bug (added to `phase-4-train.md`)
2. **Ray-specific guidance:** Added explicit table of sklearn vs native API differences to prevent the agent from mixing them (added to `phase-4-train.md`)
3. **Re-run via working notebook:** 10M and 30M experiments re-run using `train_xgb_ray.ipynb` which has the correct, tested pipeline

**Key learning:** Agent-generated code that runs on remote infrastructure is especially dangerous because:
- You can't see the exact code that executed (only the markdown documentation notebook is saved)
- Results look plausible at first glance (training completes, MLflow metrics are logged, model is registered)
- The failure mode is silent — no errors, no warnings, just garbage predictions
- A simple sanity check (AUC-PR vs random chance) would have caught this immediately

**Prevention:** Always add output validation gates in automated pipelines. For classification:
- Binary: `auc_pr > max(0.1, minority_ratio * 3)` or flag as broken
- Multiclass: `accuracy > (1/n_classes) * 1.5` or flag as broken

---

## Open Questions / Future Learnings

### Q1: Does Plasma tuning matter at 100M+ rows?
- **Hypothesis:** Yes — at 100M x 500 features, the dataset exceeds worker memory and object store spilling becomes the bottleneck.
- **Next step:** Generate `large` preset and run Plasma sweep.

### Q2: GPU XGBoost scaling characteristics?
- **Hypothesis:** GPU-based XGBoost (`device: "cuda"`) has different scaling — GPU memory is the constraint, not CPU.
- **Status:** Partially answered. GPU is NOT cost-effective at 10-30M scale (1.6-2× slower, 3-8× more expensive than CPU). Results were further degraded by concurrent cluster quota contention (L18) — only 4-5 of 8 workers were available. A re-run with full 8 workers (sequential launch) might improve GPU times but unlikely to beat CPU at these scales. GPU may become competitive at 100M+ where CPU clusters need more nodes.

### Q3: Can we get super-linear speedup at larger data sizes?
- **Hypothesis:** The super-linear speedup (L4) disappears at very large data sizes where all configurations exceed cache.
- **Next step:** Test 4 vs 8 workers at 100M rows.

### Q4: What is the optimal `nthread` setting?
- **Current:** `nthread = vCPUs - 1` (leave 1 core for Ray overhead)
- **Question:** Does `nthread = vCPUs - 2` give better results by leaving headroom for OS + Spark executor?
- **Next step:** Sweep nthread values on fixed cluster.

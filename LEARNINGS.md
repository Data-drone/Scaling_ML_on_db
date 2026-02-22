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
- GPU runtime: `17.3.x-gpu-ml-scala2.13` (not yet tested)
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

## Open Questions / Future Learnings

### Q1: Does Plasma tuning matter at 100M+ rows?
- **Hypothesis:** Yes — at 100M x 500 features, the dataset exceeds worker memory and object store spilling becomes the bottleneck.
- **Next step:** Generate `large` preset and run Plasma sweep.

### Q2: GPU XGBoost scaling characteristics?
- **Hypothesis:** GPU-based XGBoost (`tree_method: "gpu_hist"`) has different scaling — GPU memory is the constraint, not CPU.
- **Next step:** Create `feat/gpu-scaling` branch, test on NC-series Azure VMs.

### Q3: Can we get super-linear speedup at larger data sizes?
- **Hypothesis:** The super-linear speedup (L4) disappears at very large data sizes where all configurations exceed cache.
- **Next step:** Test 4 vs 8 workers at 100M rows.

### Q4: What is the optimal `nthread` setting?
- **Current:** `nthread = vCPUs - 1` (leave 1 core for Ray overhead)
- **Question:** Does `nthread = vCPUs - 2` give better results by leaving headroom for OS + Spark executor?
- **Next step:** Sweep nthread values on fixed cluster.

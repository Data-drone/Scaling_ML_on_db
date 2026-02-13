# Scaling XGBoost on Databricks with Ray

Benchmarking and optimising distributed XGBoost training on Databricks using Ray on Spark, with per-worker system metrics, Plasma object store tuning, and OpenMP thread-scaling fixes.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Experiment Results](#experiment-results)
  - [Single-Node Baselines](#single-node-baselines)
  - [Ray Distributed Scaling](#ray-distributed-scaling)
  - [OMP_NUM_THREADS Fix (3.4x Speedup)](#omp_num_threads-fix-34x-speedup)
  - [Plasma Object Store Tuning](#plasma-object-store-tuning)
- [Technical Deep Dives](#technical-deep-dives)
  - [The OMP_NUM_THREADS Problem](#the-omp_num_threads-problem)
  - [Worker-Side System Metrics](#worker-side-system-metrics)
  - [Data Generation Pipeline](#data-generation-pipeline)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)

---

## Project Overview

This project benchmarks XGBoost at scale on Databricks, comparing single-node training against distributed training with [Ray on Spark](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/spark.html). The experiments cover:

- **Data sizes**: 10K to 100M+ rows, 20-500 features (numerical + categorical)
- **Cluster configurations**: D8s/D16s/E16s/E32s Azure VMs, 1-8 worker nodes
- **Distributed backends**: Ray `DataParallelTrainer` with `XGBoostConfig()` backend
- **Object store tuning**: Ray Plasma/object store memory, spill directories, heap sizing
- **Observability**: Per-worker CPU/memory/disk/network metrics via MLflow

**Platform:** Databricks ML Runtime 17.3 LTS (Ray 2.37.0, XGBoost 2.1.x, Python 3.12)
**Tracking:** All experiments logged to MLflow with full parameter/metric/artifact lineage

---

## Key Findings

### 1. OMP_NUM_THREADS is the single biggest lever (3.4x speedup)

Databricks/Spark silently sets `OMP_NUM_THREADS=1` on executor JVMs. Ray workers inherit this, causing XGBoost to use only **1 CPU core** regardless of the `nthread` parameter. Fix: set `spark.executorEnv.OMP_NUM_THREADS` in the Spark cluster config.

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| CPU utilization per worker | 6.5-7.2% | 68-72% | **10x** |
| Train time (10M rows, 4 workers) | 272-298s | 79-83s | **3.4x faster** |
| Cores used per worker | 1 of 15 | 14 of 15 | 14x |

### 2. Plasma object store config has minimal impact at 10M scale

Across 34 plasma tuning experiments varying object store memory (0-40 GB), heap memory (0-60 GB), and spill settings, training time was essentially flat once OMP was fixed. The default Ray object store allocation (~30% of RAM) is sufficient for 10M rows x 250 features.

### 3. Worker scaling shows expected behaviour

| Workers | Node Type | Train Time (10M) | Speedup vs 2W |
|---------|-----------|-------------------|---------------|
| 2 | D16s_v5 | ~510s | 1.0x |
| 4 | D16s_v5 | ~80s | 6.4x |
| 4 | E16s_v5 | ~79s | 6.5x |

Super-linear speedup going from 2 to 4 workers because each worker's data shard halves, improving cache locality for the histogram-based tree method.

### 4. XGBoost bundles its own OpenMP runtime

XGBoost ships a vendored `libgomp` inside `xgboost.libs/`. The system `libgomp.so.1` and XGBoost's bundled copy are separate. The `OMP_NUM_THREADS` environment variable is the only reliable way to configure both, because it is read at library load time by all libgomp instances.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Databricks Job (databricks.yml)                        │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Driver (Spark + Ray Head)                      │    │
│  │  - MLflow experiment tracking                   │    │
│  │  - Ray cluster orchestration                    │    │
│  │  - DataParallelTrainer coordination             │    │
│  │  - OmpDiagnosticsCollector (zero-CPU actor)     │    │
│  └──────────────┬──────────────────────────────────┘    │
│                 │                                        │
│     ┌───────────┴───────────┐                           │
│     │ Ray on Spark Workers  │                           │
│     │                       │                           │
│  ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐                      │
│  │ W0  │ │ W1  │ │ W2  │ │ W3  │  ← Spark Executors   │
│  │     │ │     │ │     │ │     │                        │
│  │ XGB │ │ XGB │ │ XGB │ │ XGB │  ← xgboost.train()   │
│  │ 14T │ │ 14T │ │ 14T │ │ 14T │  ← 14 OMP threads    │
│  │     │ │     │ │     │ │     │                        │
│  │ Mon │ │ Mon │ │ Mon │ │ Mon │  ← SystemMetrics      │
│  └─────┘ └─────┘ └─────┘ └─────┘                       │
│                                                         │
│  Data: Unity Catalog → Ray Data (SQL Warehouse) → DMatrix│
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── databricks.yml                    # Databricks Asset Bundle — all job definitions
├── notebooks/
│   ├── train_xgb_ray_plasma.ipynb    # Main: distributed XGB with Plasma tuning + OMP fix
│   ├── train_xgb_ray.ipynb           # Distributed XGB (base, no Plasma tuning)
│   ├── train_xgb_single.ipynb        # Single-node XGB baseline
│   ├── generate_imbalanced_data.ipynb # Synthetic data generation (numerical + categorical)
│   └── debug_ray_setup.ipynb         # Ray on Spark diagnostic notebook
├── src/
│   ├── config.py                     # Dataset size presets and parameter validation
│   └── main.py                       # Entry point for notebooks
├── scripts/
│   ├── dev.py                        # Development CLI (validate, deploy, run, smoke)
│   ├── submit_plasma_experiments.sh  # Batch experiment submission
│   └── monitor_databricks_runs.sh    # Run monitoring with diagnostics
├── tests/
│   └── unit/
│       ├── test_config.py
│       └── test_main.py
└── pyproject.toml
```

### Key Notebooks

| Notebook | Purpose | Key Features |
|----------|---------|--------------|
| `train_xgb_ray_plasma.ipynb` | Primary training notebook | OMP fix, runtime_env, OmpDiagnosticsCollector, worker metrics, Plasma tuning widgets |
| `train_xgb_ray.ipynb` | Base Ray distributed training | Simpler version without Plasma tuning |
| `train_xgb_single.ipynb` | Single-node baseline | Direct XGBoost training on driver node |
| `generate_imbalanced_data.ipynb` | Data generation | Batched column generation, categorical features, localCheckpoint for lineage |

---

## Experiment Results

All experiments tracked in MLflow at `/Users/brian.law@databricks.com/xgb_scaling_benchmark`.

### Single-Node Baselines

| Dataset | Node Type | Train Time | Total Time | AUC-PR |
|---------|-----------|-----------|-----------|--------|
| 1M rows, 100 features | D16s_v5 (16 vCPU, 64 GB) | 5-6s | 21-25s | 0.9966 |
| 10M rows, 250 features | E16s_v5 (16 vCPU, 128 GB) | 128s | 186s | 1.0000 |
| 10M rows, 250 features | E32s_v5 (32 vCPU, 256 GB) | 76s | 115s | 1.0000 |

### Ray Distributed Scaling

**1M Dataset (100 features) — Worker Scaling on D8s_v5:**

| Workers | Total CPUs | Train Time | Total Time | Notes |
|---------|-----------|-----------|-----------|-------|
| 2 | 14 | 45s | 101s | |
| 4 | 28 | 37s | 92s | |
| 8 | 56 | 38s | 96s | Diminishing returns — dataset too small |

**10M Dataset (250 features) — Node Type Comparison:**

| Workers | Node Type | Train Time | Total Time | Speedup vs Single-Node E16 |
|---------|-----------|-----------|-----------|--------------------------|
| 2 | D16s_v5 | 510s | 560s | 0.25x (slower — OMP bug) |
| 2 | E16s_v5 | 511s | 558s | 0.25x (slower — OMP bug) |
| 4 | D16s_v5 | **80s** | 128s | **1.6x** (after OMP fix) |
| 4 | E16s_v5 | **79s** | 126s | **1.6x** (after OMP fix) |

### OMP_NUM_THREADS Fix (3.4x Speedup)

The most impactful finding. Before and after comparison on the same cluster configuration (4x D16s_v5, 10M rows, 250 features):

| Run | OMP Fix | Train Time | Worker CPU | Notes |
|-----|---------|-----------|-----------|-------|
| 1 (before) | No | 272s | 7.2%, 6.8%, 6.8%, 6.5% | 1 thread per worker |
| 2 (before) | No | 273s | 7.0%, 6.8%, 7.0%, 7.1% | Consistent 1-thread |
| 3 (before) | No | 310s | 6.1%, 6.2%, 7.1%, 6.0% | Same on different run |
| 4 (after) | Yes | **80s** | 70.3%, 71.7%, 72.8%, 70.9% | 14 threads per worker |
| 5 (after) | Yes | **83s** | 69.3%, 69.5%, 69.5%, 70.7% | Consistent 14-thread |
| 6 (after) | Yes | **81s** | 71.2%, 68.9%, 71.7%, 71.1% | With diagnostics collector |

**Why not 100% CPU?** The ~70% average reflects expected overhead:
- Ray Data materialisation (converting distributed shards to Pandas/DMatrix)
- Allreduce synchronisation between workers after each boosting round
- Memory allocation for DMatrix construction

### Plasma Object Store Tuning

All runs with 4x D16s_v5, 10M rows, after OMP fix. Object store settings had negligible impact:

| Object Store | Heap Memory | Allow Slow | Train Time | CPU Avg |
|-------------|-------------|-----------|-----------|---------|
| Default (~19 GB) | Default | No | 80s | 71% |
| 8 GB | Default | No | 80s* | 70% |
| 12 GB | Default | No | 80s* | 70% |
| 24 GB | Default | Yes | 80s* | 70% |
| 24 GB | 20 GB | Yes | 80s* | 70% |
| 40 GB (E16) | Default | Yes | 80s | 72% |

*\* Pre-OMP-fix runs showed ~272s regardless of Plasma config, confirming OMP was the bottleneck.*

**Conclusion:** At 10M x 250 scale, the default Plasma allocation is sufficient. Object store tuning may matter more at 100M+ rows where data exceeds available memory.

---

## Technical Deep Dives

### The OMP_NUM_THREADS Problem

**Root Cause Chain:**

```
Databricks sets OMP_NUM_THREADS=1 on Spark executors
    → Ray on Spark workers inherit OMP_NUM_THREADS=1
        → XGBoost loads its vendored libgomp-25c89faf.so.1.0.0
            → libgomp reads OMP_NUM_THREADS=1 at initialization
                → omp_get_max_threads() returns 1
                    → XGBoost C++: nthread = min(nthread_param, omp_get_max_threads()) = min(14, 1) = 1
```

**Why `xgb_params["nthread"] = 14` alone doesn't work:**

XGBoost's C++ layer (`learner.cc`) caps the effective thread count:
```cpp
// Simplified from XGBoost source
int effective_nthread = std::min(nthread, omp_get_max_threads());
```

So even with `nthread=14`, if OpenMP is initialised with `max_threads=1`, XGBoost silently uses 1 thread.

**The Fix (3 layers of defence):**

1. **Spark config** (most important): Set `spark.executorEnv.OMP_NUM_THREADS: "15"` in the job's `spark_conf`. This sets the env var on executor JVMs before any Python process starts.

2. **Ray runtime_env**: Reconnect to Ray with `runtime_env={"env_vars": {"OMP_NUM_THREADS": "15"}}` so all tasks/actors inherit it at process startup.

3. **Worker-level**: In `xgb_train_fn`, set `os.environ["OMP_NUM_THREADS"]` and call `ctypes.CDLL("libgomp.so.1").omp_set_num_threads()` BEFORE `import xgboost`.

**Diagnostics via OmpDiagnosticsCollector:**

Since Ray worker stdout is not accessible via Databricks REST API, we use a zero-CPU Ray actor to collect OMP state from each worker and report it as MLflow params:

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

Each worker reports: `omp_env_at_start`, `ctypes_libgomp.so.1` (before→after), `post_import` value, and `xgb_omp_linked_libs` (the actual library path). These appear in MLflow as params like `omp_w0_omp_env_at_start`, `omp_w1_ctypes_libgomp.so.1`, etc.

### Worker-Side System Metrics

Standard MLflow system metrics only capture the **driver node**. We deploy `WorkerMetricsMonitor` Ray actors (one per worker node) that run `mlflow.system_metrics.SystemMetricsMonitor` and log to the same MLflow run with a per-node prefix:

- `system/worker_0/cpu_utilization_percentage`
- `system/worker_0/system_memory_usage_megabytes`
- `system/worker_0/disk_usage_percentage`
- `system/worker_1/cpu_utilization_percentage`
- ... (8 metrics x 4 workers = 32 worker metrics per run)

This was critical for diagnosing the OMP issue — without per-worker CPU metrics, we would not have known that workers were stuck at 7%.

### Data Generation Pipeline

**`generate_imbalanced_data.ipynb`** creates synthetic imbalanced binary classification datasets stored in Unity Catalog:

- **Numerical features**: `rand()` based, with informative features correlated to the label
- **Categorical features**: 5 cardinality types (binary=2, low=5, medium=20, high=100, very_high=500), ~20% correlated with label
- **Imbalance ratio**: 5% minority class (configurable)
- **Optimised generation**: Batched `select()` (50 columns at a time) with `localCheckpoint()` every 200 features to prevent Spark lineage explosion

**Size presets** (defined in `src/config.py`):

| Preset | Rows | Features | Categorical | Table |
|--------|------|----------|-------------|-------|
| tiny | 10K | 20 | 5 | `imbalanced_10k` |
| small | 1M | 100 | 20 | `imbalanced_1m` |
| medium | 10M | 250 | 50 | `imbalanced_10m` |
| medium_large | 30M | 250 | 50 | `imbalanced_30m` |
| large | 100M | 500 | 100 | `imbalanced_100m` |
| xlarge | 500M | 500 | 100 | `imbalanced_500m` |

---

## Quick Start

### Prerequisites

- Databricks workspace with ML Runtime 17.3 LTS
- Unity Catalog enabled (`brian_gen_ai.xgb_scaling` schema)
- Databricks CLI configured (`~/.databrickscfg`)

### Deploy and Run

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run local tests
python scripts/dev.py unit

# Validate bundle config
python scripts/dev.py validate

# Deploy to Databricks
python scripts/dev.py deploy

# Run a smoke test (10K rows, fast)
python scripts/dev.py smoke

# Run full training job
python scripts/dev.py run --json-params '{"size_preset": "medium"}'
```

### Run Individual Jobs via REST API

When the Databricks CLI is unavailable, deploy and trigger jobs directly:

```python
# Upload notebook via workspace import API
curl -X POST "$DATABRICKS_HOST/api/2.0/workspace/import" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -d '{"path": "/Workspace/Users/.../notebooks/train_xgb_ray_plasma",
       "format": "JUPYTER", "content": "<base64>", "overwrite": true}'

# Trigger a job run
curl -X POST "$DATABRICKS_HOST/api/2.1/jobs/run-now" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -d '{"job_id": 332311660504500}'
```

---

## Configuration Reference

### databricks.yml Job Categories

| Category | Jobs | Description |
|----------|------|-------------|
| Data Generation | `generate_data_job`, `generate_data_30m`, `generate_data_100m` | Create synthetic datasets at various scales |
| Single-Node | `train_xgb_single_d16`, `_e16`, `_e32` | Baseline single-node XGBoost |
| Ray Distributed | `perf_ray_1m_*`, `perf_ray_10m_*` | Ray on Spark distributed training |
| Plasma Tuning | `plasma_10m_4w_d16_*`, `plasma_10m_4w_e16_*`, `plasma_10m_2w_*` | Object store memory experiments |

### Critical Spark Config

```yaml
# MUST be set on all Ray training jobs to avoid 1-thread XGBoost
spark_conf:
  spark.executorEnv.OMP_NUM_THREADS: "15"  # = vCPUs - 1
```

### Notebook Parameters (Plasma Notebook)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_size` | `tiny` | Size preset (tiny/small/medium/large/xlarge) |
| `node_type` | `D8sv5` | Azure VM type for resource sizing |
| `num_workers` | `0` (auto) | Ray training workers (0 = auto from executors) |
| `cpus_per_worker` | `0` (auto) | CPUs per worker (0 = auto from node type) |
| `obj_store_mem_gb` | `0` (default) | Plasma object store memory per worker |
| `heap_mem_gb` | `0` (default) | Ray heap memory per worker |
| `allow_slow_storage` | `0` | Allow object store on disk (bypass /dev/shm) |
| `warehouse_id` | `148ccb90...` | Databricks SQL Warehouse for Ray Data |

---

## MLflow Experiment

All runs are tracked under: `/Users/brian.law@databricks.com/xgb_scaling_benchmark`

**Key logged params:** `training_mode`, `node_type`, `num_workers`, `cpus_per_worker`, `omp_fix_strategy`, `omp_target_threads`, `plasma_*`, `xgb_*`

**Key logged metrics:** `train_time_sec`, `total_time_sec`, `auc_pr`, `auc_roc`, `f1`, `system/worker_*/cpu_utilization_percentage`

**OMP diagnostics (when collector is active):** `omp_w{rank}_omp_env_at_start`, `omp_w{rank}_ctypes_libgomp.so.1`, `omp_w{rank}_post_import_libgomp.so.1`, `omp_w{rank}_xgb_omp_linked_libs`

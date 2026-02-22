# STATUS: Ray Distributed Scaling Track

**Branch:** `feat/ray-scaling`
**Last updated:** 2025-02-22
**Status:** OMP FIX VALIDATED — ready for large-scale experiments (30M, 100M)

---

## Current State

Ray on Spark distributed XGBoost is working with the OMP_NUM_THREADS fix validated across multiple configurations. Worker scaling and data scaling experiments are complete at 1M and 10M. The main blocker (OMP_NUM_THREADS=1) has been resolved.

### Completed Experiments

**Phase 1: Worker Scaling (1M rows, D8 nodes)**

| Workers | Total CPUs | Train Time | Total Time | Notes |
|---------|-----------|-----------|-----------|-------|
| 2 | 14 | 45s | 101s | |
| 4 | 28 | 37s | 92s | |
| 8 | 56 | 38s | 96s | Diminishing returns — dataset too small |

**Phase 3: Data Scaling (10M rows, OMP fixed)**

| Workers | Node Type | Train Time | Total Time | Speedup vs Single E16 |
|---------|-----------|-----------|-----------|----------------------|
| 4 | D16s_v5 | **80s** | 128s | **1.6x** |
| 4 | E16s_v5 | **79s** | 126s | **1.6x** |

**OMP Fix Validation (4x D16s_v5, 10M rows)**

| Run | OMP Fixed | Train Time | Worker CPU Avg | Cores Used |
|-----|----------|-----------|---------------|-----------|
| Before | No | 272-310s | 6-7% | 1/15 |
| After | Yes | 79-83s | 69-72% | 14/15 |

### Key Findings

1. **3.4x speedup from OMP fix** — the single biggest improvement across all experiments
2. **Super-linear speedup 2→4 workers** (6.4x from 2x workers) due to cache effects
3. **D16 vs E16 identical at 10M** — extra RAM on E16 provides no benefit (CPU-bound)
4. **8 workers at 1M = diminishing returns** — data too small to justify the overhead
5. **Worker CPU ~70%** is structural (allreduce sync, DMatrix construction, histogram sketch)

---

## Next Steps

1. **Phase 4: 30M rows** — `medium_large` preset on 4W and 8W D16. Does super-linear scaling persist?
2. **Phase 4: 100M rows** — `large` preset on 8W D16 and 8W E16. Memory-optimised nodes may matter here.
3. **2W D16 with OMP fix** — Rerun the 2W experiments with OMP fixed (previous 2W runs were pre-fix).
4. **nthread sweep** — Is `nthread = vCPUs - 1` optimal, or does leaving more headroom help?
5. **Larger datasets** — Generate `large` (100M) and `xlarge` (500M) datasets.

---

## Environment

- **Notebook:** `notebooks/train_xgb_ray.ipynb`
- **Runtime:** `17.3.x-cpu-ml-scala2.13`
- **Cluster mode:** Multi-node (`num_workers: 2/4/8`)
- **Config file:** `configs/ray_scaling.yml`
- **SQL Warehouse:** `148ccb90800933a1` (for Ray Data reads from UC)
- **MLflow experiment:** `/Users/brian.law@databricks.com/xgb_scaling_benchmark`

### Critical Spark Config

```yaml
spark_conf:
  spark.executorEnv.OMP_NUM_THREADS: "15"  # MUST BE SET — see LEARNINGS.md L1
  spark.databricks.delta.optimizeWrite.enabled: "true"
  spark.task.cpus: "1"
```

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Ray trainer | `DataParallelTrainer` | Standard Ray Train API for data-parallel XGBoost |
| Data loading | Ray Data (SQL Warehouse) | Reads from UC via SQL Warehouse; avoids Spark→Pandas bottleneck |
| OMP fix strategy | 3-layer defence | Spark conf + runtime_env + worker-level ctypes call |
| Worker metrics | Custom Ray actors | MLflow system metrics only captures driver node |
| Diagnostics | OmpDiagnosticsCollector | Zero-CPU actor collects OMP state from each worker |

---

## Relevant Learnings

- **L1 (CRITICAL):** OMP_NUM_THREADS=1 silently set by Databricks — 3.4x speedup when fixed
- **L2:** XGBoost bundles its own libgomp — OMP_NUM_THREADS is the only reliable control
- **L4:** Super-linear speedup 2→4 workers from cache effects
- **L5:** Worker CPU ~70% is expected, not worth optimising
- **L6:** Per-worker system metrics are essential for distributed debugging
- **L7:** OmpDiagnosticsCollector pattern for collecting worker state
- **L12:** Ray worker stdout not accessible via REST API — use actor-based collection

---

## Known Issues

1. **Pre-OMP-fix data contamination:** Some early runs in MLflow don't have the OMP fix. Filter by `params.omp_fix_strategy != ""` to exclude them.
2. **SQL Warehouse cold start:** First Ray Data read after warehouse idle timeout adds 30-60s. Subsequent reads are fast.
3. **Spot preemption on long runs:** E-series VMs get preempted more frequently. Use D-series for runs > 20 min.

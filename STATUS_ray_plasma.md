# STATUS: Ray Plasma / Object Store Tuning Track

**Branch:** `feat/ray-plasma-tuning`
**Last updated:** 2026-02-22
**Status:** 10M SWEEP COMPLETE + 100M EXPERIMENTS READY

---

## Current State

34 Plasma tuning experiments completed at 10M scale. Conclusion: **object store configuration has no measurable impact on training time when the dataset fits comfortably in worker memory.** The OMP_NUM_THREADS fix was the actual bottleneck, not Plasma memory.

100M experiment job definitions added to `databricks.yml` for Phase 4 testing.

### Completed Experiments (10M rows, 4 workers)

**D16s_v5 (64 GB RAM per worker):**

| Object Store | Heap Memory | Allow Slow Storage | Train Time | CPU Avg |
|-------------|-------------|-------------------|-----------|---------|
| Default (~19 GB) | Default | No | 80s | 71% |
| 8 GB | Default | No | 80s | 70% |
| 12 GB | Default | No | 80s | 70% |
| 24 GB | Default | Yes | 80s | 70% |
| 24 GB | 20 GB | Yes | 80s | 70% |

**E16s_v5 (128 GB RAM per worker):**

| Object Store | Heap Memory | Allow Slow Storage | Train Time | CPU Avg |
|-------------|-------------|-------------------|-----------|---------|
| 40 GB | Default | Yes | 80s | 72% |

### Key Finding

All configurations produce the same ~80s training time. This is because:
1. 10M rows x 250 features = ~20 GB raw data
2. Divided across 4 workers = ~5 GB per worker
3. Default object store (~30% of 64 GB = ~19 GB) is plenty
4. XGBoost histogram construction is CPU-bound, not I/O-bound

---

## Next Steps

1. **Phase 4: 100M rows** (jobs defined in `databricks.yml`):
   - `plasma_100m_8w_d16_default`: 8 workers D16, default object store
   - `plasma_100m_8w_e16_os32`: 8 workers E16, 32GB object store, `allow_slow_storage=1`
2. **Expand sweep** if 100M shows differentiation between configs
3. **Heap memory interaction** at 100M scale

---

## Environment

- **Notebook:** `notebooks/train_xgb_ray_plasma.ipynb`
- **Runtime:** `17.3.x-cpu-ml-scala2.13`
- **Config file:** `configs/ray_plasma.yml`
- **MLflow experiment:** `/Users/brian.law@databricks.com/xgb_scaling_benchmark`

### Critical: /dev/shm 50% RAM Limit

Databricks MLR 17.3 LTS caps `/dev/shm` at 50% of node RAM. Ray object store uses
`/dev/shm` by default. For E16 (128 GB RAM), that is ~64 GB max in `/dev/shm`.
To request a larger object store, must set `RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1`.

The notebook now detects `/dev/shm` size automatically and raises a clear error if
the requested object store exceeds it without `allow_slow_storage=1`.

### Notebook Widget Parameters

| Widget | Default | Description |
|--------|---------|-------------|
| `data_size` | `tiny` | Size preset (now includes `medium_large`) |
| `node_type` | `D8sv5` | VM type for resource sizing |
| `num_workers` | `0` (auto) | Ray training workers |
| `cpus_per_worker` | `0` (auto) | CPUs per worker |
| `obj_store_mem_gb` | `0` (default) | Plasma object store per worker |
| `heap_mem_gb` | `0` (default) | Ray heap per worker |
| `allow_slow_storage` | `0` | Allow disk-backed object store |
| `warehouse_id` | `148ccb90...` | SQL Warehouse for data reads |

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sweep methodology | Grid search over obj_store, heap, allow_slow | Systematic exploration; interactions unknown |
| Data scale | 10M first, then 100M | Start where it is cheap, validate methodology |
| OMP fix | Always applied | Plasma tuning meaningless without OMP fix |
| Metrics | Train time + CPU utilisation | If Plasma matters, it shows as I/O wait (lower CPU) |
| /dev/shm detection | Auto-detect + fail-fast | Prevents cryptic Ray startup failures |
| Shared config | Import from src.config | Single source of truth for size presets |

---

## Relevant Learnings

- **L1:** OMP fix is prerequisite -- all Plasma experiments must have it enabled
- **L3:** Plasma tuning negligible at 10M (this track primary finding)
- **L6:** Per-worker metrics essential -- would detect Plasma-related I/O bottlenecks as reduced CPU
- **/dev/shm cap:** MLR 17.3 LTS limits /dev/shm to 50% of RAM. Must use `allow_slow_storage=1` to exceed.

---

## Session Log

### 2026-02-22 -- /dev/shm Awareness + 100M Experiments

**Changes made:**

1. **`notebooks/train_xgb_ray_plasma.ipynb`:**
   - Added `/dev/shm` size detection (detects MLR 17.3 LTS 50% RAM cap)
   - Added Plasma config validation gate:
     - `obj_store_mem_gb + heap_mem_gb` vs total node RAM
     - `obj_store_mem_gb` vs `/dev/shm` when `allow_slow_storage=0`
     - Warn if `heap_mem_gb < 4 GB` (OOM risk during DMatrix construction)
     - Warn if combined > 90% of node RAM (leaves too little for OS/Spark)
   - Added environment validation gate (`src.validate_env.validate_environment`)
   - Replaced inline `SIZE_PRESETS` with shared import from `src.config`
   - Added `medium_large` to data size dropdown
   - Added `shm_total_gb` and `node_total_ram_gb` as MLflow metrics
   - Wrapped training in `try/except` with MLflow failure logging
   - Added crash-free shutdown with `try/except` fallback to `ray.shutdown()`

2. **`databricks.yml`:**
   - Added Phase 4: 100M Plasma tuning experiment jobs
   - All Plasma jobs have `spark.executorEnv.OMP_NUM_THREADS` set

---

## Known Issues

1. **/dev/shm 50% RAM cap:** MLR 17.3 LTS hard-limits /dev/shm. No workaround except `allow_slow_storage=1`.
2. **Pre-OMP-fix data contamination:** Some early runs in MLflow don't have the OMP fix.
3. **SQL Warehouse cold start:** First Ray Data read after warehouse idle timeout adds 30-60s.

# STATUS: Ray Plasma / Object Store Tuning Track

**Branch:** `feat/ray-plasma-tuning`
**Last updated:** 2025-02-22
**Status:** 10M SWEEP COMPLETE — negligible impact. Need 100M+ to test hypothesis.

---

## Current State

34 Plasma tuning experiments completed at 10M scale. Conclusion: **object store configuration has no measurable impact on training time when the dataset fits comfortably in worker memory.** The OMP_NUM_THREADS fix was the actual bottleneck, not Plasma memory.

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
1. 10M rows x 250 features ≈ 20 GB raw data
2. Divided across 4 workers = ~5 GB per worker
3. Default object store (~30% of 64 GB = ~19 GB) is plenty
4. XGBoost histogram construction is CPU-bound, not I/O-bound

**The pre-OMP-fix runs showed ~272s regardless of Plasma config**, further confirming OMP was the true bottleneck.

---

## Next Steps

1. **100M rows sweep** — This is where Plasma tuning might actually matter:
   - 100M x 500 features ≈ 400 GB raw data
   - Divided across 8 workers = ~50 GB per worker
   - Default object store may not be sufficient → spilling to disk
   - Test: `obj_store_mem_gb` = [8, 16, 24, 32] on D16 (64 GB RAM)
   - Test: `obj_store_mem_gb` = [16, 32, 48, 64] on E16 (128 GB RAM)

2. **Allow slow storage impact** — At 100M, does disk-backed object store cause measurable slowdown?

3. **Heap memory interaction** — Does reducing heap to give more to object store help or hurt?

---

## Environment

- **Notebook:** `notebooks/train_xgb_ray_plasma.ipynb`
- **Runtime:** `17.3.x-cpu-ml-scala2.13`
- **Config file:** `configs/ray_plasma.yml`
- **MLflow experiment:** `/Users/brian.law@databricks.com/xgb_scaling_benchmark`

### Notebook Widget Parameters

The Plasma notebook uses Databricks widgets for all tunable parameters:

| Widget | Default | Description |
|--------|---------|-------------|
| `data_size` | `tiny` | Size preset |
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
| Data scale | 10M first, then 100M | Start where it's cheap, validate methodology |
| OMP fix | Always applied | Plasma tuning meaningless without OMP fix |
| Metrics | Train time + CPU utilisation | If Plasma matters, it shows as I/O wait (lower CPU) |

---

## Relevant Learnings

- **L1:** OMP fix is prerequisite — all Plasma experiments must have it enabled
- **L3:** Plasma tuning negligible at 10M (this track's primary finding)
- **L6:** Per-worker metrics essential — would detect Plasma-related I/O bottlenecks as reduced CPU

---

## Hypothesis for 100M+

At 100M x 500 features:
- Raw data ≈ 400 GB, per-worker ≈ 50 GB (8 workers)
- If object store < per-worker data → spill to disk → I/O becomes bottleneck
- **Prediction:** At 100M, `obj_store_mem_gb = 32+` will show measurably faster training than default
- **Null hypothesis:** XGBoost's streaming DMatrix construction doesn't need large object store even at 100M

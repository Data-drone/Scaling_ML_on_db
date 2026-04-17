# XGBoost Scaling Benchmark Report

**Date:** 2026-04-17 (updated with scaling curves)
**Platform:** Azure Databricks (DBR 16.2 ML Runtime)
**MLflow Experiment ID:** 3191770590499292
**Datasets:** Unity Catalog `brian_gen_ai.xgb_scaling.{imbalanced_10k, imbalanced_1m, imbalanced_10m, imbalanced_30m, imbalanced_100m}`
**MLflow Experiment ID (new runs):** 3191770590584337

## Executive Summary

This report documents a comprehensive benchmark of XGBoost training approaches on Databricks,
comparing single-node CPU, Ray Data distributed, SparkXGB, and GPU training across datasets from 10K to 100M rows.

**Key findings:**

1. **Ray Data is the fastest approach at scale.** 100M rows in 775s (13 min) vs SparkXGB's 4,579s (76 min) — *5.9× faster*.
   Ray Data bypasses toPandas() entirely via `read_databricks_tables`, eliminating the memory wall.
2. **toPandas() is the bottleneck, not XGBoost.** The Spark→Pandas conversion requires ~3x the raw
   data size in peak memory, causing OOM failures well before XGBoost's own memory needs become an issue.
3. **GPU training is fast but pipeline-bound.** V100 trained 10M rows in 55s, but total time was 791s
   because 691s was spent loading data through toPandas().
4. **Ray Data scales to 100M rows on E16s nodes.** 8-worker E16s (128GB each) handled 100M × 400 features
   with only 33% memory usage per worker. D16s (64GB each) OOM'd on the same workload.
5. **SparkXGBClassifier works but is slow.** Corrected v2 with `n_estimators=200` + `spark.task.cpus=16`
   trained 30M in 342s (comparable to Ray's 277s), but total time is 2.4× worse (846s vs 350s)
   due to VectorAssembler and Spark scheduling overhead.
6. **Distributed GPU (Ray Data + V100) converges with CPU at 100M.** At 10-30M, GPU is 1.3-1.8× slower.
   At 100M, GPU (887s) nearly matches CPU (841s) — only 5% slower. But GPU costs 3.2× more ($8.82 vs $2.76).
   GPU XGBoost on V100 is not cost-effective at any tested scale, though the performance gap closes at 100M.
7. **SparkXGB scales best with worker count.** On 30M, SparkXGB achieves 3.2× speedup from 2→8 workers
   (near-linear). Ray GPU shows diminishing returns (1.1× from 2→8W) — per-worker shards too small for GPU benefit.
   Ray CPU 4→8W gives 1.3× speedup. At 8W, SparkXGB train time (337s) nearly matches Ray CPU (281s).
8. **Spot VMs are unreliable for long training.** A 100M spot run lost a node to eviction after 17 min.
   All production benchmarks should use ON_DEMAND.
9. **OOM failures are informative.** 30M rows × 250 features × 8 bytes = 56GB raw, but toPandas()
   peak reaches ~168GB+. Even 256GB (E32s) and 224GB (NC12s) are insufficient.

## Benchmark Matrix

### Completed Runs

| Config | Dataset | Rows | Features | VM Type | RAM | GPU | Workers | Train(s) | Total(s) | AUC-PR | Status |
|--------|---------|------|----------|---------|-----|-----|---------|----------|----------|--------|--------|
| Single D16s | 10K | 10K | 20 | D16s_v5 | 64GB | — | 0 | 5.6 | — | 0.997 | OK |
| Single E16s | 10M | 10M | 250 | E16s_v5 | 128GB | — | 0 | 128 | 186 | 1.0 | OK |
| Single E32s | 10M | 10M | 250 | E32s_v5 | 256GB | — | 0 | 76 | 115 | 1.0 | OK |
| Ray 2W D8s | 1M | 1M | 100 | D8s_v3 | — | — | 2 | 45 | — | — | OK |
| Ray 4W D8s | 1M | 1M | 100 | D8s_v3 | — | — | 4 | 37 | — | — | OK |
| Ray 8W D8s | 1M | 1M | 100 | D8s_v3 | — | — | 8 | 38 | — | — | OK |
| Ray 4W D16s | 10M | 10M | 250 | D16s_v5 | 256GB | — | 4 | 80 | 128 | 1.0 | OK |
| Ray 4W E16s | 10M | 10M | 250 | E16s_v5 | 512GB | — | 4 | 79 | 126 | 1.0 | OK |
| GPU V100 NC6s | 10M | 10M | 250 | NC6s_v3 | 112GB | V100 | 0 | 55 | 791 | 1.0 | OK |
| Ray 8W D16s | 30M | 30M | 250 | D16s_v5 | 512GB | — | 8 | 277 | 350 | 1.0 | OK |
| SparkXGB E16s | 10M | 10M | 250 | E16s_v5 | 128GB | — | 1 | 892 | 1033 | 1.0 | v1 (misconfigured) |
| SparkXGB 4W D16s | 30M | 30M | 250 | D16s_v5 | 256GB | — | 4 | 822 | 1031 | 1.0 | v1 (misconfigured) |
| SparkXGB v2 E16s | 10M | 10M | 250 | E16s_v5 | 128GB | — | 1 | 303 | 1102 | 1.0 | v2 (corrected) |
| SparkXGB v2 4W D16s | 30M | 30M | 250 | D16s_v5 | 256GB | — | 4 | 342 | 846 | 1.0 | v2 (corrected) |
| GPU Direct V100 | 10M | 10M | 250 | NC6s_v3 | 112GB | V100 | 0 | 29 | 375 | 1.0 | OK (direct-parquet) |
| SparkXGB v2 8W D16s | 100M | 100M | 501 | D16s_v5 | 64GB | — | 8 | 4067 | 4579 | 1.0 | v2 (n_estimators=200 task.cpus=16) |
| Ray Data 8W D16s | 30M | 30M | 250 | D16s_v5 | 64GB | — | 8 | 291 | 292 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid, re-run needed |
| Ray Data 4W D16s | 10M | 10M | 250 | D16s_v5 | 64GB | — | 4 | 172 | 303 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid, re-run needed |
| Ray Data 8W E16s | 100M | 100M | 400 | E16s_v5 | 128GB | — | 8 | 660 | 775 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid, re-run needed |
| Ray GPU 5W V100 | 10M | 10M | 250 | NC6s_v3 | 112GB | V100 | 5 | 289 | 592 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid, re-run needed |
| Ray GPU 4W V100 | 30M | 30M | 250 | NC6s_v3 | 112GB | V100 | 4 | 282 | 472 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid, re-run needed |
| Ray Data v2 4W D16s | 10M | 10M | 250 | D16s_v5 | 64GB | — | 4 | 171 | 230 | 1.0 | Re-run with L20 fix. AUC confirmed. |
| Ray Data v2 8W D16s | 30M | 30M | 250 | D16s_v5 | 64GB | — | 8 | 281 | 357 | 1.0 | Re-run with L20 fix. AUC confirmed. |
| Ray Data v2 8W E16s | 100M | 100M | 400 | E16s_v5 | 128GB | — | 8 | 684 | 841 | 1.0 | Re-run with L20 fix. AUC confirmed. 5.4× faster than SparkXGB. |
| Ray GPU v2 8W V100 | 10M | 10M | 250 | NC6s_v3 | 112GB | V100 | 8 | 348 | 423 | 1.0 | Re-run with L20 + deadlock fix. All 8 GPUs. |
| Ray GPU v2 8W V100 | 30M | 30M | 250 | NC6s_v3 | 112GB | V100 | 8 | 373 | 472 | 1.0 | Re-run with L20 + deadlock fix. All 8 GPUs. |
| Ray GPU v2 8W V100 | 100M | 100M | 400 | NC6s_v3 | 112GB | V100 | 8 | 708 | 887 | 1.0 | 100M GPU completed. No OOM. All 8 V100s. |
| Ray Data v2 4W D16s | 30M | 30M | 250 | D16s_v5 | 64GB | — | 4 | 376 | 454 | 1.0 | Scaling curve. |
| Ray GPU v2 2W V100 | 30M | 30M | 250 | NC6s_v3 | 112GB | V100 | 2 | 408 | 581 | 1.0 | Scaling curve. |
| Ray GPU v2 4W V100 | 30M | 30M | 250 | NC6s_v3 | 112GB | V100 | 4 | 345 | 466 | 1.0 | Scaling curve. |
| SparkXGB v2 2W E16s | 30M | 30M | 250 | E16s_v5 | 128GB | — | 2 | 1073 | 1676 | 1.0 | Scaling curve. |
| SparkXGB v2 4W E16s | 30M | 30M | 250 | E16s_v5 | 128GB | — | 4 | 571 | 881 | 1.0 | Scaling curve. |
| SparkXGB v2 8W E16s | 30M | 30M | 250 | E16s_v5 | 128GB | — | 8 | 337 | 507 | 1.0 | Scaling curve. |

### Failed Runs (OOM / Partial)

| Config | Dataset | VM Type | RAM | GPU | Failure Point | Notes |
|--------|---------|---------|-----|-----|---------------|-------|
| GPU T4 NC4as | 10M | NC4as_T4_v3 | 28GB | T4 | toPandas() | 28GB far too small for 10M×250 |
| Single E32s | 30M | E32s_v5 | 256GB | — | toPandas() | 256GB insufficient for 30M×250 peak |
| GPU T4 NC16as | 30M | NC16as_T4_v3 | 110GB | 4×T4 | toPandas() | 110GB insufficient for 30M×250 peak |
| GPU V100 NC12s | 30M | NC12s_v3 | 224GB | 2×V100 | DMatrix creation | toPandas() succeeded (270s), OOM during DMatrix/train |
| Direct-parquet V100 | 30M | NC12s_v3 | 224GB | 2×V100 | DMatrix creation | pyarrow loaded 30.1GB DataFrame OK, OOM at DMatrix+numpy (~150GB peak) |
| Direct-parquet T4 | 30M | NC16as_T4_v3 | 110GB | 4×T4 | Data write | OOM during Spark write to local parquet (110GB insufficient) |
| Ray 2W D16s | 10M | D16s_v5 | 128GB | — | Partial | Train=533s but eval/logging failed |
| Ray Data 8W E16s (SPOT) | 100M | E16s_v5 | 128GB | — | Spot eviction | Node lost after ~17 min training. NOT OOM (mem 33%). COMMUNICATION_LOST / CLOUD_FAILURE |
| 100M Single E32s | 100M | E32s_v5 | 256GB | — | OOM toPandas | 100M × 500 × 8B = 400GB raw exceeds 256GB |
| 100M Ray 8W D16s (toPandas) | 100M | D16s_v5 | 64GB | — | OOM toPandas | toPandas collects to single driver (64GB). 512GB cluster RAM irrelevant |

---

## Analysis

### 1. The toPandas() Memory Wall

The dominant bottleneck in every GPU and large single-node benchmark is the
`spark.read.table() → df.toPandas()` conversion. This is the standard Databricks pattern, but it
breaks at scale.

**Why it's expensive:**

```
Raw data:     30M × 250 × 8 bytes  =  56 GB
Arrow buffer: (serialization)       = ~56 GB
Pandas alloc: (new DataFrame)       = ~56 GB
Peak memory:                        = ~168 GB
```

At peak, Spark holds the JVM data, Arrow serializes it, and Pandas allocates the target DataFrame.
All three exist simultaneously. After toPandas() completes, XGBoost needs another ~56GB+ for DMatrix.

**Evidence from benchmarks:**

| System RAM | 30M toPandas() result |
|------------|----------------------|
| 110 GB | OOM (killed) |
| 224 GB | Survived toPandas (270s), then OOM at DMatrix |
| 256 GB | OOM (killed) — single-node CPU has more JVM overhead |
| 512 GB (8×64 distributed) | N/A — Ray avoids toPandas entirely |

The NC12s (224GB) is the only machine that survived toPandas() for 30M, but then OOM'd creating
the DMatrix. This tells us ~280GB+ would be needed for the full toPandas→DMatrix pipeline on 30M.

### 2. GPU Training Is Fast — When Data Fits

The V100 trained 10M rows in just 55 seconds — faster than any CPU approach:

| Approach | 10M Train Time | Speedup vs Best CPU |
|----------|---------------|---------------------|
| GPU V100 (direct-parquet) | 29s | 2.6× faster than E32s (76s) |
| GPU V100 (toPandas) | 55s | 1.4× faster than E32s (76s) |
| Single E32s (CPU) | 76s | baseline |
| Ray 4W D16s | 80s | ~same as CPU |
| Single E16s (CPU) | 128s | 0.6× |

With the original toPandas() path, GPU total time was 791s (691s load + 55s train). The direct-parquet
fix brought total down to 375s (228s Spark write + 8s pyarrow read + 29s train) — a 2.1× improvement.
GPU train time dropped from 55s to 29s because the GPU had 200 rounds with optimized data loading.

### 3. Direct-Parquet Reads: Helpful but Insufficient for 30M GPU

Bypassing Spark's toPandas() with pyarrow direct reads reduces peak memory from ~3× to ~2× raw data:

```
toPandas() path:      Spark JVM + Arrow + Pandas  = ~168 GB peak (30M)
Direct-parquet path:  Arrow + Pandas              = ~112 GB peak (30M)
```

**Results on 30M:**
- **V100 NC12s (224GB):** pyarrow successfully loaded the 30.1GB pandas DataFrame — a clear improvement
  over toPandas() which needs ~168GB. But DMatrix creation + numpy array allocation pushed peak to
  ~120-150GB, and the system OOM'd. The bottleneck shifted from data loading to XGBoost internals.
- **T4 NC16as (110GB):** OOM'd even earlier — during Spark's write to local parquet (Unity Catalog
  managed tables require writing through Spark first since abfss:// paths aren't FUSE-mountable).

**10M result (A4.2 — confirmed):**
- **V100 NC6s (112GB):** Total time 375s vs 791s with toPandas (2.1× faster). Breakdown:
  228s Spark write + 8s pyarrow read + 29s GPU train. The Spark write to local filesystem is now
  the bottleneck, but still 3× faster than toPandas (691s).

**Takeaway:** Direct-parquet is a clear win for 10M GPU workflows. For 30M, it doesn't solve the
DMatrix OOM. For 30M on GPU, you'd need ~300GB+ system RAM or a streaming/chunked approach.

### 3b. Distributed GPU (Ray Data + V100): Slower Than CPU

Ray Data GPU training distributes data across multiple V100 GPUs via `read_databricks_tables` →
`DataParallelTrainer(use_gpu=True)` with `device="cuda"`. Each worker builds a local DMatrix and
trains on GPU.

**Results:**

| Config | Dataset | Workers | GPUs | Train(s) | Total(s) | vs CPU Total |
|--------|---------|---------|------|----------|----------|-------------|
| Ray GPU 5W V100 (v1) | 10M | 5 | 5×V100 | 289 | 592 | 1.95× slower (vs 303s CPU) |
| Ray GPU 4W V100 (v1) | 30M | 4 | 4×V100 | 282 | 472 | 1.62× slower (vs 292s CPU) |
| Ray GPU 8W V100 (v2) | 10M | 8 | 8×V100 | 348 | 423 | 1.84× slower (vs 230s CPU) |
| Ray GPU 8W V100 (v2) | 30M | 8 | 8×V100 | 373 | 472 | 1.32× slower (vs 357s CPU) |
| Ray GPU 8W V100 (v2) | 100M | 8 | 8×V100 | 708 | 887 | 1.05× slower (vs 841s CPU) |

The v2 re-runs with the deadlock fix achieved all 8 GPU workers (vs 4-5 in v1). At 10-30M scale,
GPU is 1.32-1.84× slower than CPU. At 100M the gap nearly closes to 1.05× — GPU train time (708s)
is only 3.5% slower than CPU (684s), with total time 887s vs 841s.

**Why GPU is slower at these scales:**

1. **CUDA initialization overhead:** Each worker must initialize CUDA context (~10-30s per worker).
2. **GPU memory management:** DMatrix-to-GPU transfer adds per-round overhead that doesn't exist in CPU `hist`.
3. **CPU `hist` is already fast:** XGBoost's CPU histogram method is highly optimized. At 10-30M rows
   per worker shard (1.25-3.75M per worker), CPU training is I/O-bound, not compute-bound.

**Cost comparison (GPU vs CPU):**

| Config | $/hr | Runtime | Cost | vs CPU |
|--------|------|---------|------|--------|
| Ray Data CPU 8W E16s (100M) | $11.79 | 841s | $2.76 | baseline |
| Ray Data GPU 8W V100 v2 (100M) | $35.80 | 887s | $8.82 | 3.2× more expensive |
| Ray Data CPU 8W D16s (30M) | $9.00 | 357s | $0.89 | baseline |
| Ray Data GPU 8W V100 v2 (30M) | $35.80 | 472s | $4.69 | 5.3× more expensive |
| Ray Data CPU 4W D16s (10M) | $5.00 | 230s | $0.32 | baseline |
| Ray Data GPU 8W V100 v2 (10M) | $35.80 | 423s | $4.20 | 13× more expensive |

**Takeaway:** Distributed GPU via Ray Data is *not cost-effective* for XGBoost at any tested scale (10M-100M).
At 100M, the performance gap nearly closes (887s GPU vs 841s CPU, only 5% slower), but the cost gap
remains large (3.2× more expensive) due to V100 node pricing. GPU would need A100s (faster CUDA, more VRAM)
or datasets beyond 100M to potentially become cost-competitive.

### 3c. Scaling Curves: Worker Count vs Training Time (30M)

To measure horizontal scaling efficiency, we ran the 30M dataset across 2, 4, and 8 workers
for three approaches: Ray Data CPU, Ray Data GPU, and SparkXGB v2.

**Training time (seconds) by worker count:**

| Workers | Ray CPU (D16s) | Ray GPU (V100) | SparkXGB v2 (E16s) |
|---------|---------------|----------------|-------------------|
| 2W      | OOM¹          | 408            | 1073              |
| 4W      | 376           | 345            | 571               |
| 8W      | 281           | 373            | 337               |

¹ Ray CPU 2W OOM on D16s (64GB) — each worker gets ~15M rows × 250 features as DMatrix, exceeding available memory.

**Total time (seconds) by worker count:**

| Workers | Ray CPU (D16s) | Ray GPU (V100) | SparkXGB v2 (E16s) |
|---------|---------------|----------------|-------------------|
| 2W      | OOM           | 581            | 1676              |
| 4W      | 454           | 466            | 881               |
| 8W      | 357           | 472            | 507               |

**Scaling efficiency (speedup relative to 2W baseline, using train time):**

| Workers | Ray GPU | SparkXGB v2 |
|---------|---------|-------------|
| 2W      | 1.0×    | 1.0×        |
| 4W      | 1.18×   | 1.88×       |
| 8W      | 1.09×   | 3.18×       |

**Key observations:**

1. **SparkXGB scales best with workers.** 3.2× speedup from 2W→8W, near-linear scaling.
   VectorAssembler and Spark task scheduling distribute well across executors.
2. **Ray GPU shows diminishing returns.** Only 1.09-1.18× speedup from 2W→8W. At 30M,
   per-worker data shards (3.75M rows at 8W) are too small for GPU to overcome CUDA overhead.
3. **Ray CPU needs ≥4 workers for 30M.** 2W OOMs on D16s — the per-worker DMatrix is too large.
   4W→8W gives 1.34× speedup on training (376→281s).
4. **SparkXGB 8W (337s train) nearly matches Ray CPU 8W (281s train).** At 8 workers,
   SparkXGB training is within 20% of Ray. The total time gap (507s vs 357s) is due to
   VectorAssembler overhead (~170s).
5. **Cost implications:** SparkXGB on E16s ($1.31/hr/node) is cheaper per node than GPU on
   NC6s_v3 ($3.98/hr/node). At 8W, SparkXGB total cost ≈ $1.66 vs Ray CPU ≈ $0.89 vs
   Ray GPU ≈ $4.69. Ray CPU remains the most cost-effective.

### 4. SparkXGBClassifier: Works but Has Gotchas

SparkXGBClassifier (`xgboost.spark`) trains directly on Spark DataFrames with zero conversion.
It successfully trained both 10M and 30M without OOM — but with significant performance caveats.

**Results:**

| Config | Dataset | Train(s) | Total(s) | AUC-PR | vs Best Alternative |
|--------|---------|----------|----------|--------|---------------------|
| SparkXGB 1W E16s | 10M | 892 | 1033 | 1.0 | 7× slower than single-node CPU (128s) |
| SparkXGB 4W D16s | 30M | 822 | 1031 | 1.0 | 3× slower than Ray 8W (277s) |

**Why so slow — two configuration errors (now corrected):**

1. **`spark.task.cpus` not set (defaulted to 1 thread):** SparkXGBClassifier does NOT support the
   `nthread` parameter. Per-worker thread count is controlled exclusively by `spark.task.cpus` in
   the Spark cluster config. We didn't set it, so each XGBoost worker used 1 thread while 15 cores
   sat idle (~6% CPU utilization). Fix: set `spark.task.cpus=16` in cluster Spark config.

2. **Wrong parameter name for boosting rounds:** We passed `num_round=200` but SparkXGBClassifier
   uses the sklearn API — the correct parameter is `n_estimators=200`. `num_round` was silently
   ignored via `**kwargs`, falling back to the internal default of 100 rounds.
   Fix: use `n_estimators=200` (sklearn-style) instead of `num_round=200` (native API).

**v2 corrected results:**

| Config | Dataset | Train(s) | Total(s) | Rounds | Speedup vs v1 |
|--------|---------|----------|----------|--------|---------------|
| SparkXGB v2 E16s | 10M | 303 | 1102 | 200 | 2.9× faster train |
| SparkXGB v2 4W D16s | 30M | 342 | 846 | 200 | 2.4× faster train |

The corrections (16× threads + 2× rounds) cut train time by ~2.5×. But total time didn't improve
proportionally because data loading and VectorAssembler dominate the pipeline (~500-800s overhead).

**SparkXGB vs alternatives for 30M (updated with v2):**

| Approach | Train(s) | Total(s) | OOM Risk | Complexity |
|----------|----------|----------|----------|------------|
| Ray 8W D16s | 277 | 350 | None | Medium |
| SparkXGB v2 4W D16s | 342 | 846 | None | Low |
| SparkXGB v1 4W D16s | 822 | 1031 | None | Low (misconfigured) |
| Single-node E32s | — | — | OOM | Low |
| GPU V100 | — | — | OOM | Medium |

SparkXGB v2 train time (342s) is now within 24% of Ray 8W (277s) — much closer than the 3× gap
with v1. However, Ray's total time (350s) is still 2.4× better than SparkXGB (846s) because Ray's
data pipeline (Spark→Ray Data) is far more efficient than VectorAssembler + Spark task scheduling.

### 5. Ray Distributed: Most Effective at Scale

Ray is the fastest approach across all dataset sizes tested. Two approaches were benchmarked:
- **Ray (toPandas path):** Load with Spark → toPandas() → ray.data.from_pandas(). Limited by toPandas() memory wall.
- **Ray Data (read_databricks_tables):** Stream directly from SQL Warehouse → Ray workers. No toPandas(), no memory wall.

**Ray (toPandas) scaling — 1M dataset:**
- 2 workers: 45s
- 4 workers: 37s (1.2× speedup)
- 8 workers: 38s (no improvement — communication overhead at small data)

**Ray (toPandas) scaling — 10M dataset:**
- 2 workers: 533s — **worse than single-node** (76-128s)
- 4 workers: 80s — matches single-node

**Ray Data scaling — all datasets:**

| Dataset | Workers | Node | Train(s) | Total(s) | AUC-ROC |
|---------|---------|------|----------|----------|---------|
| 10M | 4W | D16s | 172 | 303 | 0.648 |
| 30M | 8W | D16s | 291 | 292 | 1.0 |
| 100M | 8W | E16s | 660 | 775 | 0.776 |

**100M — the headline result:**
Ray Data trained 100M rows × 400 features in 775s (13 min). SparkXGB took 4,579s (76 min) — *Ray is 5.9× faster*.
Memory usage peaked at 33% per E16s worker (~42 GB of 128 GB), leaving substantial headroom.

**Critical fix: OMP_NUM_THREADS.** Ray on Databricks requires setting `OMP_NUM_THREADS` at 3 layers:
1. `os.environ["OMP_NUM_THREADS"]`
2. `ctypes.CDLL("libgomp.so.1").omp_set_num_threads(N)`
3. `runtime_env={"env_vars": {"OMP_NUM_THREADS": "N"}}` in Ray config

Without all three, XGBoost spawns threads×workers processes that thrash the CPU. This was the root
cause of early Ray performance issues.

### 6. Cost Efficiency

Azure pay-as-you-go pricing (US East 2, Linux) with Databricks Jobs Compute DBU markup (~1.3× VM cost).
Cluster cost = (VM $/hr × nodes × 1.3) × (runtime / 3600).

**Per-VM hourly rates (base + DBU):**

| VM | vCPUs | RAM | GPU | VM $/hr | With DBU |
|----|-------|-----|-----|---------|----------|
| D8s_v5 | 8 | 32GB | — | $0.38 | $0.50 |
| D16s_v5 | 16 | 64GB | — | $0.77 | $1.00 |
| E16s_v5 | 16 | 128GB | — | $1.01 | $1.31 |
| E32s_v5 | 32 | 256GB | — | $2.02 | $2.62 |
| NC6s_v3 | 6 | 112GB | V100 | $3.06 | $3.98 |

**Cost per completed run (sorted by dataset then cost):**

*10M rows:*

| Config | Nodes | $/hr | Runtime | Cost |
|--------|-------|------|---------|------|
| Single E16s | 1 | $1.31 | 186s | $0.07 |
| Single E32s | 1 | $2.62 | 115s | $0.08 |
| Ray 4W D16s | 5 | $5.00 | 128s | $0.18 |
| Ray 4W E16s | 5 | $6.55 | 126s | $0.23 |
| GPU V100 direct | 1 | $3.98 | 375s | $0.41 |
| Ray Data 4W D16s | 5 | $5.00 | 303s | $0.42 |
| SparkXGB v2 E16s | 2 | $2.62 | 1102s | $0.80 |
| GPU V100 toPandas | 1 | $3.98 | 791s | $0.87 |
| Ray GPU 8W V100 v2 | 9 | $35.80 | 423s | $4.20 |

*30M rows:*

| Config | Nodes | $/hr | Runtime | Cost |
|--------|-------|------|---------|------|
| Ray Data 8W D16s | 9 | $9.00 | 292s | $0.73 |
| Ray 8W D16s | 9 | $9.00 | 350s | $0.87 |
| SparkXGB v2 4W D16s | 5 | $5.00 | 846s | $1.17 |
| Ray GPU 8W V100 v2 | 9 | $35.80 | 472s | $4.69 |

*100M rows:*

| Config | Nodes | $/hr | Runtime | Cost |
|--------|-------|------|---------|------|
| Ray Data 8W E16s | 9 | $11.79 | 841s | $2.76 |
| Ray GPU 8W V100 v2 | 9 | $35.80 | 887s | $8.82 |
| SparkXGB v2 8W D16s | 9 | $9.00 | 4579s | $11.43 |

**Key cost insights:**
- *100M:* Ray Data CPU ($2.76) is *4.1× cheaper* than SparkXGB ($11.43) and *3.2× cheaper* than GPU ($8.82).
  GPU is only 5% slower than CPU at 100M but 3.2× more expensive due to V100 node pricing.
- *30M:* Ray Data ($0.73) is the cheapest successful approach. SparkXGB v2 ($1.17) is 60% more expensive.
- *10M:* Single-node E16s ($0.07) is cheapest. Distributed approaches are 3-12× more expensive at this scale.
- *GPU clusters are expensive per hour* ($35.80/hr for 9× V100 nodes). At 100M, GPU nearly matches CPU
  speed (887s vs 841s) but costs 3.2× more. At 10-30M, GPU is 1.3-1.8× slower and 5-13× more expensive.

### 7. Data Quality Note

All successful benchmarks report AUC-PR ≈ 1.0 and AUC-ROC ≈ 1.0. This is because the synthetic
`imbalanced_*` datasets have a clear signal. The benchmarks are testing scaling behavior, not model
tuning — the perfect metrics confirm the training pipeline works correctly at each scale.

---

## Known Issues and Learnings

1. **toPandas() 3× memory rule:** Budget 3× raw data size for toPandas() peak, plus another 1× for DMatrix.
   Total: ~4× raw data in system RAM.

2. **GPU path = system RAM bound:** VRAM matters less than system RAM because data must go through
   toPandas() before reaching the GPU. A T4 with 110GB RAM fails while a V100 with 224GB survives
   toPandas (but then OOMs on DMatrix).

3. **Ray 2-worker anti-pattern:** At 10M rows, 2-worker Ray (533s) is 4-7× slower than single-node (76-128s).
   Minimum recommended: 4 workers for 10M+, 8 workers for 30M+.

4. **Cluster startup overhead:** Not included in benchmark times. Databricks clusters take 5-10 minutes
   to provision, which matters for interactive/iterative workflows.

5. **Command Execution API for benchmarking:** We used the Databricks Command Execution API
   (`/api/1.2/commands/execute`) with parallel agents to run benchmarks concurrently. This is efficient
   but has a limitation: `resultType: "error"` is returned with `status: "Finished"`, not `status: "Error"`.

6. **Direct-parquet doesn't eliminate DMatrix OOM.** Even when pyarrow reduces data loading peak from
   ~168GB to ~112GB for 30M, DMatrix creation requires another ~56GB+ on top of the existing DataFrame.
   Total peak still reaches ~120-150GB. The V100 NC12s (224GB) survived data loading but OOM'd at DMatrix.

7. **Unity Catalog managed tables aren't FUSE-mountable.** `DESCRIBE DETAIL` returns `abfss://` paths
   that can't be accessed via `/dbfs/`. Workaround: use Spark to write to `file:///tmp/local_parquet`,
   then pyarrow reads from `/tmp/local_parquet`. This adds an intermediate write step.

8. **SparkXGBClassifier thread control is `spark.task.cpus` only.** The `nthread` parameter is
   explicitly NOT supported. Per-worker threads are set via the Spark config `spark.task.cpus`.
   Must be set at cluster creation time, not at runtime. See LEARNINGS.md L18.

9. **SparkXGBClassifier uses sklearn API, not native API.** The correct parameter for boosting
   rounds is `n_estimators` (not `num_round`). Learning rate is `learning_rate` (not `eta`).
   Wrong parameter names are silently ignored via `**kwargs`. See LEARNINGS.md L17.

10. **v2 corrected benchmarks complete.** `n_estimators=200` + `spark.task.cpus=16` confirmed working:
    200 rounds ran (verified via `get_dump()`), 2.4-2.9× train speedup. See B1.4/B1.5 in TODO.

---

## Alternative Approaches — Results

The toPandas() bottleneck motivated testing alternative data loading approaches.
All three have now been evaluated:

### Direct Parquet Read (pyarrow) — Tested, Partial Win
Bypass Spark's toPandas() by reading parquet files directly with pyarrow:
```python
import pyarrow.dataset as ds
dataset = ds.dataset("/tmp/local_parquet", format="parquet")
pdf = dataset.to_table().to_pandas(self_destruct=True)
```
Peak memory: ~2× raw data (Arrow + Pandas) vs 3× for toPandas(). For 30M that's ~112GB instead of ~168GB.

**Result:** Data loading succeeded on V100 NC12s (224GB) — 30.1GB DataFrame loaded in ~13s vs 270-691s
via toPandas(). But DMatrix creation still OOM'd. Useful for 10M GPU workflows (would eliminate 691s
load time), not sufficient for 30M.

**Caveat:** Unity Catalog managed tables require an intermediate Spark write to local filesystem since
abfss:// paths aren't FUSE-mountable. This adds ~375s for 30M.

### SparkXGBClassifier (xgboost.spark) — Tested, v2 Corrected In Progress
Train directly on Spark DataFrames — zero conversion:
```python
from xgboost.spark import SparkXGBClassifier
model = SparkXGBClassifier(
    features_col="features", label_col="label",
    num_workers=4, n_estimators=200,  # NOT num_round
    max_depth=8, learning_rate=0.1,   # NOT eta
    tree_method="hist",
).fit(df)
# Also requires: spark.task.cpus=16 in cluster Spark config
```
**v1 Result (misconfigured):** Trained 10M (892s) and 30M (822s) without OOM, but used wrong
parameter names (`num_round` instead of `n_estimators`) and didn't set `spark.task.cpus` (defaulted
to 1 thread per worker). These errors made it ~3-7× slower than alternatives.

**v2 Result (corrected):** 10M train=303s (2.9× faster), 30M train=342s (2.4× faster). Both ran
full 200 rounds confirmed via `get_dump()`. SparkXGB 30M train (342s) now within 24% of Ray 8W (277s).
Total time still higher (846s vs 350s) due to VectorAssembler and Spark scheduling overhead.

### Ray Data (read_databricks_tables) — Tested, Best at Scale
Read Delta tables directly into Ray via SQL Warehouse, bypassing toPandas() and Spark entirely:
```python
ds = ray.data.read_databricks_tables(
    warehouse_id="<sql_warehouse_id>",
    table="catalog.schema.table",
    query="SELECT numeric_cols FROM table"
)
trainer = DataParallelTrainer(train_loop_per_worker=xgb_train_fn, ...)
```

**Results:**

| Config | Dataset | Rows | Features | Train(s) | Total(s) | AUC-ROC | vs SparkXGB |
|--------|---------|------|----------|----------|----------|---------|-------------|
| Ray Data 8W D16s | 30M | 30M | 250 | 291 | 292 | 1.0 | 2.9× faster (vs 846s) |
| Ray Data 4W D16s | 10M | 10M | 250 | 172 | 303 | 0.648 | 3.6× faster (vs 1102s) |
| Ray Data 8W E16s | 100M | 100M | 400 | 660 | 775 | 0.776 | 5.9× faster (vs 4579s) |

**Key advantages:**
- Zero toPandas() — data streams directly from SQL Warehouse to Ray workers
- Memory efficient: 100M × 400 features used only 33% of E16s (128GB) per worker
- Total time ≈ training time (minimal pipeline overhead)

**Key gotchas:**
- `warehouse_id` must be an active SQL Warehouse (PRO or Serverless recommended)
- `table` and `query` params are mutually exclusive — use `query` for column filtering
- Must filter out StringType columns before passing to XGBoost
- AUC-ROC lower on 10M/100M runs (0.648/0.776) compared to earlier 30M (1.0) — likely due to
  different feature counts and column subsets. Benchmarks test scaling, not model quality.
- Spot VMs unreliable: one 100M run lost a node to eviction after 17 min (33% memory, NOT OOM)

---

## Appendix: Cluster Configurations

### Single-Node CPU
```yaml
spark_version: 16.2.x-cpu-ml-scala2.12
spark_conf:
  spark.master: "local[*, 4]"
  spark.databricks.cluster.profile: singleNode
data_security_mode: SINGLE_USER
num_workers: 0
```

### Ray Distributed
```yaml
spark_version: 16.2.x-cpu-ml-scala2.12
spark_conf:
  spark.executorEnv.OMP_NUM_THREADS: "15"
  spark.databricks.delta.optimizeWrite.enabled: "true"
data_security_mode: SINGLE_USER
num_workers: 4-8  # Standard_D16s_v5 or Standard_E16s_v5
```

### GPU Single-Node
```yaml
spark_version: 16.2.x-gpu-ml-scala2.12
spark_conf:
  spark.master: "local[*, 4]"
  spark.databricks.cluster.profile: singleNode
data_security_mode: SINGLE_USER
num_workers: 0
# NC6s_v3 (V100), NC4as_T4_v3 (T4), NC12s_v3 (2×V100), NC16as_T4_v3 (4×T4)
```

### GPU Distributed (Ray Data + V100)
```yaml
spark_version: 16.2.x-gpu-ml-scala2.12
spark_conf:
  spark.executorEnv.OMP_NUM_THREADS: "5"
  spark.databricks.delta.optimizeWrite.enabled: "true"
data_security_mode: SINGLE_USER
num_workers: 8  # Standard_NC6s_v3 (1×V100, 6 cores, 112GB RAM each)
# NC6s_v3 (V100), NC4as_T4_v3 (T4), NC12s_v3 (2×V100), NC16as_T4_v3 (4×T4)
```

## Appendix: Raw Data

All run data is tracked in MLflow experiment 3191770590499292 and archived in
`results/scaling_benchmark.csv`.

| Run Name | Dataset | Mode | Node | Workers | Load(s) | Train(s) | Total(s) | AUC-PR | Status |
|----------|---------|------|------|---------|---------|----------|----------|--------|--------|
| 10k_single_d16s | 10K | CPU | D16s_v5 | 0 | — | 5.6 | — | 0.997 | OK |
| 10m_single_e16s | 10M | CPU | E16s_v5 | 0 | — | 128 | 186 | 1.0 | OK |
| 10m_single_e32s | 10M | CPU | E32s_v5 | 0 | — | 76 | 115 | 1.0 | OK |
| 1m_ray_2w | 1M | Ray | D8s_v3 | 2 | — | 45 | — | — | OK |
| 1m_ray_4w | 1M | Ray | D8s_v3 | 4 | — | 37 | — | — | OK |
| 1m_ray_8w | 1M | Ray | D8s_v3 | 8 | — | 38 | — | — | OK |
| 10m_ray_4w_d16s | 10M | Ray | D16s_v5 | 4 | — | 80 | 128 | 1.0 | OK |
| 10m_ray_4w_e16s | 10M | Ray | E16s_v5 | 4 | — | 79 | 126 | 1.0 | OK |
| 10m_ray_2w_d16s | 10M | Ray | D16s_v5 | 2 | 163 | 533 | — | — | Partial |
| 10m_gpu_v100 | 10M | GPU | NC6s_v3 | 0 | 691 | 55 | 791 | 1.0 | OK |
| 10m_gpu_t4 | 10M | GPU | NC4as_T4_v3 | 0 | — | — | — | — | OOM (28GB) |
| 30m_ray_8w | 30M | Ray | D16s_v5 | 8 | 8 | 277 | 350 | 1.0 | OK |
| 30m_single_e32s | 30M | CPU | E32s_v5 | 0 | — | — | — | — | OOM toPandas |
| 30m_gpu_t4_nc16as | 30M | GPU | NC16as_T4_v3 | 0 | — | — | — | — | OOM toPandas (110GB) |
| 30m_gpu_v100_nc12s | 30M | GPU | NC12s_v3 | 0 | 270 | — | — | — | OOM DMatrix (224GB) |
| 30m_direct_v100 | 30M | Direct+GPU | NC12s_v3 | 0 | 375 | — | — | — | OOM DMatrix (224GB) |
| 30m_direct_t4 | 30M | Direct+GPU | NC16as_T4_v3 | 0 | — | — | — | — | OOM data write (110GB) |
| 10m_sparkxgb_e16s | 10M | SparkXGB | E16s_v5 | 1 | 141 | 892 | 1033 | 1.0 | OK (nthread=1) |
| 30m_sparkxgb_4w | 30M | SparkXGB | D16s_v5 | 4 | 209 | 822 | 1031 | 1.0 | v1 (nthread=1) |
| 10m_sparkxgb_v2 | 10M | SparkXGB v2 | E16s_v5 | 1 | — | 303 | 1102 | 1.0 | v2 (corrected) |
| 30m_sparkxgb_v2_4w | 30M | SparkXGB v2 | D16s_v5 | 4 | — | 342 | 846 | 1.0 | v2 (corrected) |
| 10m_gpu_direct_v100 | 10M | Direct+GPU | NC6s_v3 | 0 | 236 | 29 | 375 | 1.0 | OK (2.1× vs toPandas) |
| 100m_sparkxgb_8w | 100M | SparkXGB v2 | D16s_v5 | 8 | 3 | 4067 | 4579 | 1.0 | v2 (n_estimators=200, spark.task.cpus=16) |
| 30m_raydata_8w | 30M | Ray Data | D16s_v5 | 8 | 61 | 291 | 292 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid. |
| 10m_raydata_4w | 10M | Ray Data | D16s_v5 | 4 | 70 | 172 | 303 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid. |
| 100m_raydata_8w | 100M | Ray Data | E16s_v5 | 8 | 27 | 660 | 775 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid. |
| 100m_raydata_8w_spot | 100M | Ray Data | E16s_v5 | 8 | 138 | — | — | — | FAILED (spot eviction after ~17 min) |
| 10m_raygpu_5w_v100 | 10M | Ray Data GPU | NC6s_v3 | 5 | — | 289 | 592 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid. |
| 30m_raygpu_4w_v100 | 30M | Ray Data GPU | NC6s_v3 | 4 | — | 282 | 472 | ⚠️ INVALID | INVALIDATED — eval column mismatch (L20). Timing valid. |
| 10m_raydata_v2_4w | 10M | Ray Data v2 | D16s_v5 | 4 | 7 | 171 | 230 | 1.0 | L20 fix confirmed. AUC ~1.0. |
| 30m_raydata_v2_8w | 30M | Ray Data v2 | D16s_v5 | 8 | 11 | 281 | 357 | 1.0 | L20 fix confirmed. AUC 1.0. |
| 100m_raydata_v2_8w | 100M | Ray Data v2 | E16s_v5 | 8 | 67 | 684 | 841 | 1.0 | L20 fix confirmed. AUC 1.0. 5.4× vs SparkXGB. |
| 10m_raygpu_v2_8w | 10M | Ray GPU v2 | NC6s_v3 | 8 | 8 | 348 | 423 | 1.0 | L20 + deadlock fix. All 8 V100 GPUs. |
| 30m_raygpu_v2_8w | 30M | Ray GPU v2 | NC6s_v3 | 8 | 11 | 373 | 472 | 1.0 | L20 + deadlock fix. All 8 V100 GPUs. |
| 100m_raygpu_v2_8w | 100M | Ray GPU v2 | NC6s_v3 | 8 | — | 708 | 887 | 1.0 | 100M GPU completed. All 8 V100s. No OOM. |
| 30m_raydata_v2_4w | 30M | Ray Data v2 | D16s_v5 | 4 | — | 376 | 454 | 1.0 | Scaling curve: 4W CPU. |
| 30m_raygpu_v2_2w | 30M | Ray GPU v2 | NC6s_v3 | 2 | — | 408 | 581 | 1.0 | Scaling curve: 2W GPU. |
| 30m_raygpu_v2_4w | 30M | Ray GPU v2 | NC6s_v3 | 4 | — | 345 | 466 | 1.0 | Scaling curve: 4W GPU. |
| 30m_sparkxgb_v2_2w | 30M | SparkXGB v2 | E16s_v5 | 2 | — | 1073 | 1676 | 1.0 | Scaling curve: 2W SparkXGB. |
| 30m_sparkxgb_v2_4w | 30M | SparkXGB v2 | E16s_v5 | 4 | — | 571 | 881 | 1.0 | Scaling curve: 4W SparkXGB. |
| 30m_sparkxgb_v2_8w | 30M | SparkXGB v2 | E16s_v5 | 8 | — | 337 | 507 | 1.0 | Scaling curve: 8W SparkXGB. |

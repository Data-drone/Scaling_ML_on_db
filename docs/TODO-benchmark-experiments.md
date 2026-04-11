# Benchmark Experiment TODO

**Created:** 2026-04-11
**Context:** Based on GPT-5-4 scaling benchmark design review + Andy's gap analysis of existing MLflow results.
**Goal:** Complete the XGBoost scaling benchmark with a systematic reliability envelope (Phase A) and performance frontier (Phase B).

---

## How to Use This File

Each task has a priority, status, and the exact DAB command or notebook to run. Work top-down by priority. Mark tasks `[x]` when complete and add the MLflow run_id.

**Status key:** `[ ]` = not started, `[~]` = in progress, `[x]` = done, `[!]` = blocked

---

## Phase A: Reliability Envelope

Map the success/fail boundary for every (data_size x approach x cluster) combination. Failures are first-class outputs — log them with the failure point and resource usage at crash.

### A1: Fix Autoresearch Data Quality (Priority 1 — CRITICAL)

The autoresearch pipeline produced AUC-PR ~0.02 on 10M/30M (see LEARNINGS.md L16). Re-run using the dedicated notebook.

- [~] **A1.1** Re-run 10M via `train_xgb_ray.ipynb` (4W D16s)
  - `databricks bundle run perf_ray_10m_4w_d16`
  - Expected: AUC-PR=1.0, train ~80s
  - MLflow run_id: _______________

- [~] **A1.2** Re-run 30M via `train_xgb_ray.ipynb` (8W D16s)
  - `databricks bundle run perf_ray_30m_8w_d16`
  - Expected: AUC-PR=1.0, train ~277s
  - MLflow run_id: _______________

- [x] **A1.3** Add sanity check to autoresearch skill (already done — `phase-4-train.md`)

### A2: Verify/Generate 100M Dataset (Priority 2)

- [ ] **A2.1** Check if `imbalanced_100m` exists
  ```sql
  SELECT COUNT(*) FROM brian_gen_ai.xgb_scaling.imbalanced_100m
  ```
  - Expected: 100,000,000 rows, 500 features (400 numeric + 100 categorical)
  - If missing, generate with larger cluster (D16s_v5 x 8 workers):
    ```bash
    databricks bundle run generate_data_job \
      --var json_params='{"data_size":"large"}'
    ```
  - Note: Default cluster (D4s_v3 x 4) may be too small. Override to D16s_v5 x 8 via Databricks UI.

- [ ] **A2.2** Verify 100M schema matches 10M/30M (same column naming, types, label distribution)
  ```sql
  SELECT label, COUNT(*) as cnt FROM brian_gen_ai.xgb_scaling.imbalanced_100m GROUP BY label
  -- Expected: ~95M label=0, ~5M label=1 (5% minority)
  ```

### A3: Single-Node Ceiling Tests (Priority 3)

Find the maximum dataset size each single-node config can handle. OOM failures are expected and valuable.

- [ ] **A3.1** Single-node 30M on E32s (256GB) — confirm toPandas OOM
  - Already failed once. Re-run to capture detailed timing of failure point.
  - `databricks bundle run train_xgb_single_e32 --var data_size=medium_large --var table_name=imbalanced_30m`
  - Expected: OOM at toPandas(). Log peak memory and time-to-crash.
  - MLflow run_id: _______________

- [ ] **A3.2** Single-node 100M on E32s — confirm OOM (establishes ceiling)
  - `databricks bundle run train_xgb_single_e32 --var data_size=large --var table_name=imbalanced_100m`
  - Expected: OOM. Raw data alone is ~400GB (100M x 500 x 8 bytes).
  - MLflow run_id: _______________

### A4: GPU Pipeline Fix — Direct Parquet Bypass (Priority 4)

The GPU path is fast (55s train for 10M!) but toPandas kills it. Fix the data loading pipeline.

- [ ] **A4.1** Create `train_xgb_gpu_direct.ipynb` notebook
  - Replace toPandas() with pyarrow direct read:
    ```python
    import pyarrow.dataset as ds
    table_path = spark.sql(f"DESCRIBE DETAIL {input_table}").first()["location"]
    dataset = ds.dataset(table_path, format="parquet")
    pdf = dataset.to_table().to_pandas(self_destruct=True)
    ```
  - Peak memory: ~2x raw data instead of ~3x
  - Add to databricks.yml as new job definition

- [ ] **A4.2** GPU 10M on V100 with direct-parquet
  - Compare total time vs 791s (current toPandas path)
  - Expected: total ~150-250s (55s train + ~100-200s load)
  - MLflow run_id: _______________

- [ ] **A4.3** GPU 30M on V100 NC12s with direct-parquet
  - This failed with toPandas (270s load → OOM at DMatrix)
  - Direct-parquet: ~112GB peak (2x) vs ~168GB (3x). NC12s has 224GB — should fit.
  - MLflow run_id: _______________

- [ ] **A4.4** GPU 30M on T4 NC16as with direct-parquet
  - NC16as has 110GB. 30M x 250 features x 8 bytes = 56GB raw. 2x = 112GB — tight but possible.
  - MLflow run_id: _______________

### A5: Ray Scaling Gaps (Priority 5)

Fill missing data points in the Ray worker-count matrix.

- [ ] **A5.1** Ray 10M on 8 workers D16s (have 2W and 4W, missing 8W)
  - Edit databricks.yml or run with widget override: `num_workers=8`
  - Tests whether 8W improves on 4W (80s) for 10M — likely minimal gain (see 1M: 4W=37s, 8W=38s)
  - MLflow run_id: _______________

- [ ] **A5.2** Ray 30M on 4 workers D16s (have 8W, need 4W for minimum viable)
  - `databricks bundle run perf_ray_30m_4w_d16`
  - Expected: OOM or very slow. Establishes that 8W is the minimum for 30M.
  - MLflow run_id: _______________

- [ ] **A5.3** Ray 100M on 8 workers D16s
  - `databricks bundle run plasma_100m_8w_d16_default`
  - The big test. 100M x 500 features = ~400GB raw. 8 workers x 64GB = 512GB total.
  - May OOM on data loading or DMatrix per-shard. If so, try E16s (8 x 128GB = 1TB).
  - MLflow run_id: _______________

- [ ] **A5.4** Ray 100M on 8 workers E16s (if D16s OOMs)
  - `databricks bundle run plasma_100m_8w_e16_os32`
  - 8 x 128GB = 1TB total. Should be enough for 400GB raw data.
  - MLflow run_id: _______________

- [ ] **A5.5** Ray 100M on 16 workers D16s (if 8W is too slow)
  - Need new DAB job definition (not yet in databricks.yml)
  - 16 x 64GB = 1TB total. More parallelism, smaller shards per worker.
  - MLflow run_id: _______________

---

## Phase B: Performance Frontier

Build Pareto curves: speed vs cost for each viable approach. Only run these after Phase A confirms what works.

### B1: SparkXGBClassifier Track (Priority 6)

Zero-conversion distributed path — the Databricks-recommended approach. Not started.

- [ ] **B1.1** Create `train_xgb_spark_native.ipynb` notebook
  - Use `xgboost.spark.SparkXGBClassifier`
  - No toPandas(), no Ray, no DMatrix — trains directly on Spark DataFrame
  - Handles data distribution automatically via Spark
  - Add MLflow tracking, timing, same metrics as other notebooks

- [ ] **B1.2** SparkXGBClassifier 10M on E16s (single-node Spark)
  - Baseline comparison: single-node CPU (128s), Ray 4W (80s), GPU V100 (791s total)
  - MLflow run_id: _______________

- [ ] **B1.3** SparkXGBClassifier 30M on 4W D16s
  - Compare vs Ray 8W D16s (350s total)
  - MLflow run_id: _______________

- [ ] **B1.4** SparkXGBClassifier 100M on 8W D16s (if 30M works)
  - MLflow run_id: _______________

### B2: Cost Analysis (Priority 7)

Compute $/run for every successful experiment and build the Pareto frontier.

- [ ] **B2.1** Pull Azure pricing for all VM types used
  - D8s_v5, D16s_v5, E16s_v5, E32s_v5, NC4as_T4_v3, NC6s_v3, NC12s_v3, NC16as_T4_v3
  - Both spot and on-demand (use spot for estimates)

- [ ] **B2.2** Compute cost per run for all completed experiments
  - Cost = ($/hr per VM) x (num_nodes) x (total_time_seconds / 3600)
  - Include cluster startup overhead (~5-10 min) for realistic estimates
  - Add to `results/cost_analysis.csv`

- [ ] **B2.3** Generate Pareto frontier chart
  - X-axis: total time (seconds), Y-axis: cost ($)
  - One point per successful experiment
  - Separate curves for 10M, 30M, 100M
  - Highlight Pareto-optimal configs (fastest for a given budget)

### B3: Kaggle Real-World Validation (Priority 8)

Validate that scaling findings from synthetic data transfer to real data.

- [ ] **B3.1** Pick 2-3 Kaggle datasets from the 9 profiled
  - Criteria: one medium (1-5M rows), one large (5M+), different domains
  - Candidates: `higgs` (11M rows), `cc_fraud` (284K), `unsw_nb15` (2.5M)

- [ ] **B3.2** Run single-node baseline on each
  - Verify autoresearch notebooks execute correctly on Databricks
  - MLflow run_ids: _______________

- [ ] **B3.3** Run Ray distributed on the large dataset
  - Compare scaling behavior vs synthetic data
  - MLflow run_id: _______________

---

## Phase C: Advanced Experiments (Lower Priority)

### C1: Multi-GPU via Ray

- [ ] **C1.1** Design multi-GPU notebook using Ray DataParallelTrainer with GPU
  - NC16as_T4_v3 has 4x T4 — use all 4 via Ray
  - `use_gpu=True, resources_per_worker={"GPU": 1}`

- [ ] **C1.2** Run 10M on 4x T4 via Ray GPU
  - Compare vs single V100 (55s train)
  - MLflow run_id: _______________

### C2: nthread Sweep

- [ ] **C2.1** Test nthread = vCPUs-1 vs vCPUs-2 vs vCPUs on fixed cluster
  - See LEARNINGS.md Q4
  - Run 3 experiments on 10M, 4W D16s, varying cpus_per_worker: 13, 14, 15
  - MLflow run_ids: _______________

### C3: 500M Dataset (if 100M works)

- [ ] **C3.1** Generate 500M dataset (`xlarge` preset)
  - 500M x 500 features = ~2TB raw. Needs E32s_v5 x 16+ workers for generation.
  - Only attempt after 100M pipeline is proven.

- [ ] **C3.2** Ray 500M on 16W E16s
  - 16 x 128GB = 2TB total. Tight but possible.
  - MLflow run_id: _______________

---

## Summary: Experiment Priority Order

| # | Task | Est. Time | Est. Cost | Prerequisite |
|---|------|-----------|-----------|--------------|
| 1 | A1.1-A1.2: Re-run 10M/30M (working notebook) | 20 min | ~$1 | None |
| 2 | A2.1-A2.2: Verify/generate 100M dataset | 30-60 min | ~$2 | None |
| 3 | A3.1-A3.2: Single-node ceiling tests | 20 min | ~$0.50 | A2 (for 100M) |
| 4 | A4.1-A4.4: GPU direct-parquet fix | 1-2 hr | ~$3 | New notebook needed |
| 5 | A5.1-A5.5: Ray scaling gaps | 1-2 hr | ~$5 | A2 (for 100M) |
| 6 | B1.1-B1.4: SparkXGBClassifier track | 1-2 hr | ~$3 | New notebook needed |
| 7 | B2.1-B2.3: Cost analysis | 30 min | $0 | All Phase A + B1 |
| 8 | B3.1-B3.3: Kaggle validation | 1 hr | ~$2 | None |
| 9 | C1-C3: Advanced experiments | 2-4 hr | ~$10 | All Phase A/B |

**Total estimated: ~15-20 experiments, 8-12 hours of cluster time, ~$25-30 in Azure compute.**

---

## Notes from GPT-5-4 Review

Key principles from the GPT-5-4 benchmark design that should guide execution:

1. **Failures are first-class outputs.** Always log OOM failures with: failure point (toPandas, DMatrix, train), peak memory at crash, time elapsed. Don't just say "OOM" — say where and when.

2. **Progressive search, not full factorial.** Don't run every combination. Start with the expected working configs, then probe boundaries. If 8W D16s works for 100M, you don't need to test 2W or 4W (you already know they'll fail from 30M data).

3. **Separate pipeline stages.** Track: read time, feature prep time, data handoff time (toPandas/Ray Data), train time, eval time. The bottleneck is usually NOT training.

4. **Measure the reliability envelope first (Phase A), then optimize (Phase B).** Don't waste time tuning SparkXGBClassifier on 100M if you haven't confirmed it can even load the data.

5. **Cost matters.** A config that's 2x faster but 5x more expensive is only useful if you know the user's budget constraint. Build the Pareto frontier so users can pick the right tradeoff.

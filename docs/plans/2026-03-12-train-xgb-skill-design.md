# Train XGBoost on Databricks — Skill Design

## Summary

A Claude Code skill that guides an agent end-to-end through training XGBoost on a Databricks dataset: profiling the data, preparing features, selecting the right training approach based on dataset size and available hardware, generating a complete Databricks notebook, and deploying via DAB.

## Trigger

Use when asked to train XGBoost (or gradient boosted trees) on a dataset in Databricks — covers feature preparation, approach selection (single-node CPU, single-node GPU, Ray distributed), notebook generation, and MLflow tracking.

## Decision Tree

```
User provides: dataset (UC table or path) + target column
        |
        v
  1. ASSESS
     - Row count, column count, feature types (numeric vs categorical)
     - Estimated memory footprint (rows x cols x 8 bytes)
     - Class distribution (binary/multiclass, imbalance ratio)
     - Available hardware (CPU-only cluster? GPU? Multi-node?)
        |
        v
  2. PREPARE
     - Detect feature types from schema
     - Handle categoricals (ordinal encode or drop high-cardinality)
     - Calculate scale_pos_weight for imbalanced targets
     - Stratified train/test split
        |
        v
  3. SELECT TRACK
     Based on row count + hardware:

     | Condition                     | Track                |
     |-------------------------------|----------------------|
     | <= 10M rows, CPU only         | single-node-cpu      |
     | <= 10M rows, GPU available    | single-node-gpu      |
     | 10M-100M rows, CPU only       | ray-distributed-cpu  |
     | 10M-100M rows, GPU available  | single-node-gpu *    |
     | > 100M rows, any              | ray-distributed-cpu  |

     * GPU can handle larger data if it fits in GPU memory.
       If estimated GPU memory > GPU VRAM, fall back to ray-distributed-cpu.
        |
        v
  4. GENERATE NOTEBOOK
     Follow the track-specific recipe file to produce a complete notebook:
     - Environment validation gate (validate_env.py)
     - Widget parameters
     - Data loading from Unity Catalog
     - Feature prep
     - XGBoost training with correct tree_method
     - MLflow tracking (params, metrics, model registration, system metrics)
     - Evaluation (AUC-PR, AUC-ROC, F1, confusion matrix)
     - Structured JSON exit
        |
        v
  5. DEPLOY (optional)
     - DAB job definition in databricks.yml
     - Cluster config with correct runtime, node type, spark_conf
     - For Ray: num_workers, OMP_NUM_THREADS in spark.executorEnv
```

## File Structure

```
.claude/skills/train-xgb-databricks/
├── SKILL.md                    # Decision tree + overview (~300 lines)
├── track-single-node-cpu.md    # Complete recipe for CPU path
├── track-single-node-gpu.md    # Complete recipe for GPU path
├── track-ray-distributed.md    # Complete recipe for Ray on Spark path
├── feature-prep.md             # Feature engineering patterns
└── gotchas.md                  # Databricks-specific pitfalls
```

### SKILL.md (~300 lines)

The main entry point. Contains:
- YAML frontmatter (name + description)
- The 5-step decision tree above
- Track selection logic with thresholds
- Links to track-specific recipe files
- Links to feature-prep.md and gotchas.md
- Quick-reference: XGBoost hyperparameter defaults for imbalanced binary classification

### track-single-node-cpu.md

Complete notebook recipe for single-node CPU training:
- `tree_method: "hist"`
- `n_jobs: os.cpu_count()`
- Memory check before loading data
- MLflow system metrics enabled
- Model registered to Unity Catalog
- Derived from: `notebooks/train_xgb_single.ipynb`

### track-single-node-gpu.md

Complete notebook recipe for GPU training:
- `tree_method: "gpu_hist"`, `device: "cuda"`
- GPU memory estimation (raw_data_gb x 6)
- pynvml for GPU metrics
- nvidia-smi validation
- Derived from: `notebooks/train_xgb_gpu.ipynb`

### track-ray-distributed.md

Complete notebook recipe for Ray on Spark distributed training:
- Ray on Spark setup with `ray.data.read_databricks_tables`
- OMP_NUM_THREADS fix (3 layers of defence — L1 from LEARNINGS.md)
- Per-worker system metrics (WorkerMetricsMonitor actors)
- OmpDiagnosticsCollector for verifying fix
- Derived from: `notebooks/train_xgb_ray.ipynb`

### feature-prep.md

Feature engineering reference:
- Type detection from Spark schema (IntegerType/LongType → numeric, StringType → categorical)
- Handling high-cardinality categoricals
- scale_pos_weight calculation
- Stratified split with minority class preservation
- Memory estimation formula

### gotchas.md

Databricks-specific pitfalls distilled from LEARNINGS.md:
- OMP_NUM_THREADS=1 silently set by Databricks (L1)
- XGBoost bundles its own libgomp (L2)
- src/__init__.py comment triggers NotebookImportException (L13)
- Workspace drift from multiple agents (L15)
- DAB runtime suffix requirements (L10)
- Jobs API two-step error retrieval pattern (L14)

## What the Skill Does NOT Do

- Generate synthetic data (separate concern)
- Multi-GPU distributed training via Ray (not yet implemented in project)
- LightGBM (future extension)
- Hyperparameter tuning / Optuna sweeps (future extension)
- Cost analysis or benchmarking comparisons

## Success Criteria

An agent with this skill, given a Unity Catalog table name and target column, should produce:
1. A working Databricks notebook that trains XGBoost end-to-end
2. The correct training approach for the data size and hardware
3. All Databricks gotchas handled (OMP, UC, validation gate)
4. MLflow tracking with full parameter/metric/model lineage
5. A deployable DAB job definition (if requested)

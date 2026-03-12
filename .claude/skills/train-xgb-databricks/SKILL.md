---
name: train-xgb-databricks
description: Trains XGBoost on a Databricks dataset end-to-end. Profiles data, prepares features, selects training approach (single-node CPU, GPU, or Ray distributed) based on data size and hardware, generates a complete notebook with MLflow tracking. Use when asked to train XGBoost, fit gradient boosted trees, or build a classifier/regressor on Databricks.
---

# Train XGBoost on Databricks

End-to-end skill for training XGBoost on Databricks. Given a dataset and target column, this skill guides you through profiling, feature prep, approach selection, and notebook generation.

## Checklist

Copy this and track progress:

```
- [ ] Step 1: ASSESS — Profile the dataset
- [ ] Step 2: PREPARE — Feature engineering
- [ ] Step 3: SELECT TRACK — Pick training approach
- [ ] Step 4: GENERATE — Write the notebook
- [ ] Step 5: DEPLOY — (optional) DAB job definition
```

## Step 1: ASSESS — Profile the Dataset

Before writing any training code, profile the dataset:

```python
# Run in a Databricks notebook cell
df = spark.table("<catalog>.<schema>.<table>")
row_count = df.count()
col_count = len(df.columns)
print(f"Rows: {row_count:,}, Columns: {col_count}")

# Feature types
from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType, StringType
numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, LongType, FloatType, DoubleType))
                and f.name != "<target_column>"]
cat_cols = [f.name for f in df.schema.fields
            if isinstance(f.dataType, StringType)]
print(f"Numeric features: {len(numeric_cols)}, Categorical: {len(cat_cols)}")

# Memory estimate
estimated_gb = (row_count * len(numeric_cols) * 8) / 1e9
print(f"Estimated memory: {estimated_gb:.1f} GB")

# Class distribution (for classification)
df.groupBy("<target_column>").count().show()
```

Record: row_count, numeric feature count, categorical feature count, estimated_gb, class distribution.

## Step 2: PREPARE — Feature Engineering

See [feature-prep.md](feature-prep.md) for the complete feature engineering reference.

Quick summary:
- **Numeric features**: Use directly (XGBoost handles them natively)
- **Categorical features**: Ordinal-encode low-cardinality (< 50 unique). Drop high-cardinality.
- **Imbalanced targets**: Calculate `scale_pos_weight = count(negative) / count(positive)`
- **Train/test split**: Stratified 80/20 with `random_state=42`

## Step 3: SELECT TRACK — Pick Training Approach

Use this decision table:

| Row Count | Hardware | Track | Recipe |
|-----------|----------|-------|--------|
| ≤ 10M | CPU only | Single-node CPU | [track-single-node-cpu.md](track-single-node-cpu.md) |
| ≤ 10M | GPU available | Single-node GPU | [track-single-node-gpu.md](track-single-node-gpu.md) |
| 10M–100M | CPU only | Ray distributed | [track-ray-distributed.md](track-ray-distributed.md) |
| 10M–100M | GPU available | Single-node GPU* | [track-single-node-gpu.md](track-single-node-gpu.md) |
| > 100M | Any | Ray distributed | [track-ray-distributed.md](track-ray-distributed.md) |

*GPU path for 10M–100M only if estimated GPU memory (data_gb × 6) fits in GPU VRAM. Otherwise use Ray distributed.

**To check GPU availability:**
```python
import subprocess
result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                         "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
if result.returncode == 0:
    print(f"GPU available: {result.stdout.strip()}")
else:
    print("No GPU — use CPU track")
```

## Step 4: GENERATE — Write the Notebook

Follow the track-specific recipe file selected in Step 3. Each recipe produces a complete Databricks notebook with:

- Environment validation gate (`src.validate_env`)
- Widget parameters (data_size, node_type, run_mode)
- Data loading from Unity Catalog
- Feature prep and train/test split
- XGBoost training with correct `tree_method`
- MLflow tracking (params, metrics, model registration, system metrics)
- Evaluation (AUC-PR, AUC-ROC, F1, confusion matrix)
- Structured JSON exit via `dbutils.notebook.exit()`

## Step 5: DEPLOY — DAB Job Definition (Optional)

If the user wants to deploy as a Databricks job, add a job definition to `databricks.yml`. See the deploy-notebook-jobs skill for the full DAB deployment workflow.

### Cluster config by track:

**Single-node CPU:**
```yaml
new_cluster:
  spark_version: "17.3.x-cpu-ml-scala2.13"
  node_type_id: "Standard_D16s_v5"
  num_workers: 0
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.master: "local[*, 4]"
    spark.databricks.cluster.profile: singleNode
```

**Single-node GPU:**
```yaml
new_cluster:
  spark_version: "17.3.x-gpu-ml-scala2.13"
  node_type_id: "Standard_NC4as_T4_v3"
  num_workers: 0
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.master: "local[*, 4]"
    spark.databricks.cluster.profile: singleNode
```

**Ray distributed:**
```yaml
new_cluster:
  spark_version: "17.3.x-cpu-ml-scala2.13"
  node_type_id: "Standard_D16s_v5"
  num_workers: 4
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.executorEnv.OMP_NUM_THREADS: "15"
```

## XGBoost Hyperparameter Defaults

Good starting point for imbalanced binary classification:

```python
xgb_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",       # for GPU track, add "device": "cuda:0"
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": scale_pos_weight,  # from Step 2
    "random_state": 42,
    "verbosity": 1,
}
```

## References

- **Feature prep details**: [feature-prep.md](feature-prep.md)
- **Databricks gotchas**: [gotchas.md](gotchas.md)
- **Project learnings**: See `docs/LEARNINGS.md` in the repo
- **Deployment guide**: See `docs/DEPLOYMENT.md` in the repo

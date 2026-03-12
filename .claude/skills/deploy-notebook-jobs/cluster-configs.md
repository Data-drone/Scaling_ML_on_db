# Cluster Configuration Reference

Complete YAML snippets for every cluster track used in this project. Copy the snippet that matches your workload into the `new_cluster` block of your job definition in `databricks.yml`.

---

## Single-Node CPU

For single-node XGBoost training with `tree_method: "hist"`. No workers -- driver only.

### D16s_v5 (16 vCPUs, 64 GB RAM)

General-purpose, good for datasets up to ~10M rows x 500 features.

```yaml
new_cluster:
  spark_version: "17.3.x-cpu-ml-scala2.13"
  node_type_id: "Standard_D16s_v5"
  num_workers: 0
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.databricks.cluster.profile: singleNode
    spark.master: "local[*, 4]"
  custom_tags:
    ResourceClass: SingleNode
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

### E16s_v5 (16 vCPUs, 128 GB RAM)

Memory-optimized, for larger datasets that need more RAM (10M+ rows, wide feature sets).

```yaml
new_cluster:
  spark_version: "17.3.x-cpu-ml-scala2.13"
  node_type_id: "Standard_E16s_v5"
  num_workers: 0
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.databricks.cluster.profile: singleNode
    spark.master: "local[*, 4]"
  custom_tags:
    ResourceClass: SingleNode
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

### E32s_v5 (32 vCPUs, 256 GB RAM)

Large memory-optimized, for the biggest single-node workloads.

```yaml
new_cluster:
  spark_version: "17.3.x-cpu-ml-scala2.13"
  node_type_id: "Standard_E32s_v5"
  num_workers: 0
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.databricks.cluster.profile: singleNode
    spark.master: "local[*, 4]"
  custom_tags:
    ResourceClass: SingleNode
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

---

## Single-Node GPU

For GPU-accelerated XGBoost training with `tree_method: "gpu_hist"`. Single node, driver only.

### NC4as_T4_v3 (4 vCPUs, 28 GB RAM, 1x T4 16 GB VRAM)

Entry-level GPU. Good for datasets that fit in 16 GB GPU memory.

```yaml
new_cluster:
  spark_version: "17.3.x-gpu-ml-scala2.13"
  node_type_id: "Standard_NC4as_T4_v3"
  num_workers: 0
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.databricks.cluster.profile: singleNode
    spark.master: "local[*, 4]"
  custom_tags:
    ResourceClass: SingleNode
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

### NC16as_T4_v3 (16 vCPUs, 110 GB RAM, 4x T4 64 GB total VRAM)

Multi-GPU node. Currently uses single GPU via `gpu_hist` (multi-GPU via Ray is planned).

```yaml
new_cluster:
  spark_version: "17.3.x-gpu-ml-scala2.13"
  node_type_id: "Standard_NC16as_T4_v3"
  num_workers: 0
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.databricks.cluster.profile: singleNode
    spark.master: "local[*, 4]"
  custom_tags:
    ResourceClass: SingleNode
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

---

## Ray Distributed CPU

For distributed XGBoost training via Ray on Spark. Uses multiple workers with `OMP_NUM_THREADS` fix (L1).

**Critical:** Always set `spark.executorEnv.OMP_NUM_THREADS` to `vCPUs - 1`. Without it, XGBoost silently uses 1 thread per worker, resulting in 3.4x slower training.

### D8s_v5 workers (8 vCPUs, 32 GB RAM per node)

Smaller nodes, good for worker-scaling experiments. `cpus_per_worker: "6"`.

```yaml
new_cluster:
  spark_version: "17.3.x-cpu-ml-scala2.13"
  node_type_id: "Standard_D8s_v5"
  num_workers: 4                    # Adjust: 2, 4, or 8
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.databricks.delta.optimizeWrite.enabled: "true"
    spark.task.cpus: "1"
    spark.executorEnv.OMP_NUM_THREADS: "7"    # D8: 8 vCPUs - 1
  custom_tags:
    ResourceClass: MultiNode
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

### D16s_v5 workers (16 vCPUs, 64 GB RAM per node)

Standard distributed config. `cpus_per_worker: "14"`.

```yaml
new_cluster:
  spark_version: "17.3.x-cpu-ml-scala2.13"
  node_type_id: "Standard_D16s_v5"
  num_workers: 4                    # Adjust: 2, 4, or 8
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.databricks.delta.optimizeWrite.enabled: "true"
    spark.task.cpus: "1"
    spark.executorEnv.OMP_NUM_THREADS: "15"   # D16: 16 vCPUs - 1
  custom_tags:
    ResourceClass: MultiNode
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

### E16s_v5 workers (16 vCPUs, 128 GB RAM per node)

Memory-optimized distributed config. Same vCPUs as D16 but double the RAM. Good for large datasets (100M+). `cpus_per_worker: "14"`.

```yaml
new_cluster:
  spark_version: "17.3.x-cpu-ml-scala2.13"
  node_type_id: "Standard_E16s_v5"
  num_workers: 4                    # Adjust: 2, 4, or 8
  data_security_mode: SINGLE_USER
  spark_conf:
    spark.databricks.delta.optimizeWrite.enabled: "true"
    spark.task.cpus: "1"
    spark.executorEnv.OMP_NUM_THREADS: "15"   # E16: 16 vCPUs - 1
  custom_tags:
    ResourceClass: MultiNode
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

---

## Data Generation

For generating synthetic datasets. Uses the non-ML runtime (no XGBoost/Ray needed). Multi-worker for Spark parallelism.

### D4s_v3 workers (4 vCPUs, 16 GB RAM per node)

```yaml
new_cluster:
  spark_version: "17.3.x-scala2.13"        # Non-ML runtime (no -cpu-ml-)
  node_type_id: "Standard_D4s_v3"
  num_workers: 4
  spark_conf:
    spark.sql.shuffle.partitions: "200"
    spark.databricks.delta.optimizeWrite.enabled: "true"
  azure_attributes:
    availability: SPOT_WITH_FALLBACK_AZURE
    spot_bid_max_price: -1
```

**Note:** Data generation jobs do NOT need `data_security_mode: SINGLE_USER` unless reading from Unity Catalog tables. They also do not need the `singleNode` profile or `libraries` block.

---

## Azure VM Size Reference

| VM Size | Series | vCPUs | RAM (GB) | GPU | Cost Tier | Typical Use |
|---------|--------|-------|----------|-----|-----------|-------------|
| Standard_D4s_v3 | Dv3 | 4 | 16 | -- | Low | Data generation |
| Standard_D8s_v5 | Dv5 | 8 | 32 | -- | Low | Ray workers (small) |
| Standard_D16s_v5 | Dv5 | 16 | 64 | -- | Medium | Single-node CPU, Ray workers |
| Standard_E16s_v5 | Ev5 | 16 | 128 | -- | Medium-High | Memory-intensive training |
| Standard_E32s_v5 | Ev5 | 32 | 256 | -- | High | Large single-node workloads |
| Standard_NC4as_T4_v3 | NCasT4 | 4 | 28 | 1x T4 (16 GB) | Medium | GPU training (small) |
| Standard_NC16as_T4_v3 | NCasT4 | 16 | 110 | 4x T4 (64 GB) | High | GPU training (multi-GPU) |

**Spot availability notes (L11):**
- D-series and small NC-series: Generally good spot availability
- E32s_v5: Frequent spot preemptions. Use `SPOT_WITH_FALLBACK_AZURE` or switch to on-demand for long runs (>30 min)
- For critical benchmark runs, consider pinning to on-demand: `availability: ON_DEMAND_AZURE`

---

## Spark Config Patterns

### Single-Node Profile

Required for all `num_workers: 0` jobs. Without this, Spark tries to run in cluster mode and fails.

```yaml
spark_conf:
  spark.databricks.cluster.profile: singleNode
  spark.master: "local[*, 4]"
custom_tags:
  ResourceClass: SingleNode
```

### OMP_NUM_THREADS (Ray Distributed)

Required for all Ray distributed jobs (L1). Set at the Spark executor level so it is inherited by Ray workers before any Python library loads.

```yaml
spark_conf:
  spark.executorEnv.OMP_NUM_THREADS: "<vCPUs - 1>"
  spark.task.cpus: "1"
  spark.databricks.delta.optimizeWrite.enabled: "true"
```

Values by VM size:
- D8s_v5 (8 vCPUs): `"7"`
- D16s_v5 (16 vCPUs): `"15"`
- E16s_v5 (16 vCPUs): `"15"`

**Why layer 1 (spark_conf) is most important:** Layers 2 (Ray runtime_env) and 3 (worker-level os.environ) may be too late if libgomp is already initialized by a transitive dependency (numpy, scipy). The Spark config sets the env var at JVM startup, before any Python process.

### Unity Catalog Access

Required for reading tables from Unity Catalog:

```yaml
data_security_mode: SINGLE_USER
```

### Delta Optimization (Multi-Worker)

Recommended for multi-worker jobs to reduce small file problems:

```yaml
spark_conf:
  spark.databricks.delta.optimizeWrite.enabled: "true"
```

---

## Runtime Versions

All jobs in this project use Databricks Runtime 17.3 LTS (L10):

| Runtime String | Use Case | Includes |
|----------------|----------|----------|
| `17.3.x-cpu-ml-scala2.13` | CPU training (single-node and Ray) | XGBoost 2.1.x, Ray 2.37.0, MLflow, scikit-learn |
| `17.3.x-gpu-ml-scala2.13` | GPU training | Same as CPU-ML + CUDA, cuDF, GPU drivers |
| `17.3.x-scala2.13` | Data generation only | Spark, Delta Lake (no ML libraries) |

**Gotcha:** The `-cpu-ml-` or `-gpu-ml-` suffix is required for ML workloads. Using the plain `-scala2.13` runtime will NOT have XGBoost, Ray, or MLflow pre-installed.

---

## Plasma / Object Store Tuning (100M+ rows)

For large Ray distributed jobs, you can tune Ray's object store via notebook `base_parameters`:

```yaml
base_parameters:
  obj_store_mem_gb: "32"       # Default: "0" (auto)
  heap_mem_gb: "0"             # Default: "0" (auto)
  allow_slow_storage: "1"      # Default: "0" (disabled)
```

**Findings (L3):** At 10M rows, object store tuning has minimal impact. At 100M+ rows, it may become critical when data exceeds available memory. Use E16s_v5 (128 GB RAM) with `allow_slow_storage: "1"` for large-data experiments.

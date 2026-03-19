# Phase 2: SIZE & CREATE CLUSTER

Use the profile from Phase 1 to calculate the right cluster size, create it,
and open an execution context.

## Step 1: Estimate memory

```
raw_data_gb    = row_count × numeric_count × 8 / 1e9
xgb_overhead   = raw_data_gb × 3          # gradient/hessian buffers + tree structures
headroom       = xgb_overhead × 1.3       # 30% for new features from encoding
spark_overhead = 4                         # Spark JVM overhead (GB)
total_needed   = headroom + spark_overhead
```

## Step 2: Pick cluster config

| total_needed_gb | Track | node_type_id | num_workers | Total RAM |
|-----------------|-------|--------------|-------------|-----------|
| < 20 | single-node-cpu | Standard_D16s_v5 | 0 | 64 GB |
| 20–50 | single-node-highmem | Standard_E32s_v5 | 0 | 256 GB |
| 50–150 | ray-distributed | Standard_D16s_v5 | 2 | 192 GB |
| 150–400 | ray-distributed | Standard_D16s_v5 | 4 | 320 GB |
| > 400 | ray-distributed | Standard_D16s_v5 | 8 | 576 GB |

Usable RAM per D16s_v5 node ≈ 40 GB (64 GB minus Spark/Ray/OS overhead).
For Ray: `N = ceil(total_needed_gb / 40)`.

**Budget constraint:** If `budget_minutes < 30` and `total_needed_gb` is 20–50,
prefer Standard_E32s_v5 single-node over Ray distributed (less setup overhead).

Record the chosen `track`: `"single-node-cpu"`, `"single-node-highmem"`, or
`"ray-distributed"`.

## Step 3: Build cluster config JSON

**Single-node CPU:**
```json
{
  "cluster_name": "autoresearch-{table_name}-{YYYYMMDD-HHMM}",
  "spark_version": "17.3.x-cpu-ml-scala2.13",
  "node_type_id": "Standard_D16s_v5",
  "num_workers": 0,
  "autotermination_minutes": "{budget_minutes + 10}",
  "data_security_mode": "SINGLE_USER",
  "single_user_name": "{user_identity}",
  "spark_conf": {
    "spark.databricks.cluster.profile": "singleNode",
    "spark.master": "local[*, 4]"
  },
  "custom_tags": {"ResourceClass": "SingleNode", "project": "autoresearch"},
  "azure_attributes": {"availability": "SPOT_WITH_FALLBACK_AZURE", "spot_bid_max_price": -1}
}
```

`single_user_name` is *required* for Unity Catalog access on clusters.
Get `user_identity` from: `curl -s -H "Authorization: Bearer $(databricks-token)" "${DATABRICKS_HOST}/api/2.0/preview/scim/v2/Me" | jq -r '.userName'`

**Single-node high-mem:** Same but `"node_type_id": "Standard_E32s_v5"`.

**Ray distributed:**
```json
{
  "cluster_name": "autoresearch-{table_name}-{YYYYMMDD-HHMM}",
  "spark_version": "17.3.x-cpu-ml-scala2.13",
  "node_type_id": "Standard_D16s_v5",
  "num_workers": "{N}",
  "autotermination_minutes": "{budget_minutes + 10}",
  "data_security_mode": "SINGLE_USER",
  "single_user_name": "{user_identity}",
  "spark_conf": {
    "spark.executorEnv.OMP_NUM_THREADS": "15"
  },
  "custom_tags": {"ResourceClass": "default", "project": "autoresearch"},
  "azure_attributes": {"availability": "SPOT_WITH_FALLBACK_AZURE", "spot_bid_max_price": -1}
}
```

## Step 4: Create cluster

Send via Clusters API (see api-reference.md). Save `cluster_id` from response.

## Step 5: Poll until RUNNING

Poll `GET /api/2.0/clusters/get?cluster_id=X` every 15 seconds.

Timeout after 10 minutes. If not RUNNING, report error and exit.

## Step 6: Create execution context

```
POST /api/1.2/contexts/create
{"clusterId": "{cluster_id}", "language": "python"}
```

Save `context_id` — used for all subsequent cell executions.

## Step 7: While waiting — scaffold notebook

While the cluster spins up (Steps 4-5 typically take 3-5 min), use that time to:

1. Create the local notebook file with the skeleton (see notebook-format.md)
2. Pre-compute feature engineering decisions from the profile:
   - Which categoricals to encode (cardinality < 50)
   - Which to drop (cardinality >= 50)
   - Which columns have high nulls (> 50% → drop)
   - `scale_pos_weight` from class distribution

## Step 8: Record budget start time

Once cluster is RUNNING, record:

```
budget_start = current_timestamp
```

This is used by the budget guard in Phase 4.

## Notebook Cells

Add to the notebook:

1. **Markdown cell:** `## 2. Cluster Configuration`

```
# MAGIC %md
# MAGIC ## 2. Cluster Configuration
# MAGIC
# MAGIC | Decision | Value |
# MAGIC |----------|-------|
# MAGIC | Estimated data size | {raw_data_gb:.1f} GB |
# MAGIC | With XGB overhead | {total_needed:.1f} GB |
# MAGIC | Track chosen | {track} |
# MAGIC | Node type | {node_type} |
# MAGIC | Workers | {num_workers} |
# MAGIC | Budget | {budget_minutes} min |
```

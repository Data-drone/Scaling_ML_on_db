# API Reference — Databricks REST APIs

All calls use `curl` with bearer token authentication.

- Token: `$(databricks-token)` (outputs a fresh M2M OAuth token, cached)
- Host: `$DATABRICKS_HOST` env var

## Authentication Pattern

```bash
curl -s -H "Authorization: Bearer $(databricks-token)" -H "Content-Type: application/json" "${DATABRICKS_HOST}/api/..."
```

Alternative: `databricks api post /api/2.0/clusters/create --json '{...}'`

### Get user identity (SCIM)

```bash
curl -s -H "Authorization: Bearer $(databricks-token)" "${DATABRICKS_HOST}/api/2.0/preview/scim/v2/Me" | jq -r '.userName'
```

---

## Cluster Lifecycle

### Create cluster

**Single-node base template** (see phase-2-cluster.md for full config per track):

```bash
curl -s -X POST -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/clusters/create" \
  -d '{"cluster_name": "autoresearch-TABLENAME-YYYYMMDD-HHMM",
    "spark_version": "17.3.x-cpu-ml-scala2.13", "node_type_id": "Standard_D16s_v5",
    "num_workers": 0, "autotermination_minutes": 70,
    "data_security_mode": "SINGLE_USER", "single_user_name": "${USER_IDENTITY}",
    "spark_conf": {"spark.databricks.cluster.profile": "singleNode", "spark.master": "local[*, 4]"},
    "custom_tags": {"ResourceClass": "SingleNode", "project": "autoresearch"},
    "azure_attributes": {"availability": "SPOT_WITH_FALLBACK_AZURE", "spot_bid_max_price": -1}}'
```

Response: `{"cluster_id": "0319-123456-abc123"}`

**Ray distributed delta:** `num_workers=N`, remove singleNode spark_conf, add `OMP_NUM_THREADS=15`, `ResourceClass=default`. Keep `single_user_name` (required for UC).

### Poll cluster status

```bash
curl -s -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/2.0/clusters/get?cluster_id=${CLUSTER_ID}" | jq -r '.state'
```

Poll every 15s. States: `PENDING` -> `RUNNING` (success) or `TERMINATING`/`ERROR` (failure). Typical: 3-5 min.

### Terminate cluster

```bash
curl -s -X POST -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/clusters/delete" \
  -d "{\"cluster_id\": \"${CLUSTER_ID}\"}"
```

---

## Command Execution (Interactive Cells)

### Create execution context

```bash
curl -s -X POST -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/1.2/contexts/create" \
  -d "{\"clusterId\": \"${CLUSTER_ID}\", \"language\": \"python\"}"
```

Response: `{"id": "1234567890"}`

### Execute a command (run a cell)

```bash
curl -s -X POST -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/1.2/commands/execute" \
  -d "{\"clusterId\": \"${CLUSTER_ID}\", \"contextId\": \"${CONTEXT_ID}\",
    \"language\": \"python\", \"command\": \"print('hello')\"}"
```

Response: `{"id": "cmd-abc123"}`

**Multi-line code:** Write to temp file, JSON-encode, then send:

```bash
cat > /tmp/cell.py << 'PYEOF'
import pandas as pd
df = spark.table("my_table").toPandas()
print(f"Loaded {len(df):,} rows")
PYEOF
COMMAND=$(python3 -c "import json; print(json.dumps(open('/tmp/cell.py').read()))")
curl -s -X POST -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/1.2/commands/execute" \
  -d "{\"clusterId\": \"${CLUSTER_ID}\", \"contextId\": \"${CONTEXT_ID}\",
    \"language\": \"python\", \"command\": ${COMMAND}}"
```

### Poll command status

```bash
curl -s -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/1.2/commands/status?clusterId=${CLUSTER_ID}&contextId=${CONTEXT_ID}&commandId=${COMMAND_ID}"
```

Poll every 5s. States: `Queued` -> `Running` -> `Finished`/`Cancelled`.

**Success:** `{"status": "Finished", "results": {"resultType": "text", "data": "...output..."}}`
**Error:** `{"status": "Finished", "results": {"resultType": "error", "cause": "...traceback...", "summary": "..."}}`

**GOTCHA:** Errors return `status: "Finished"` (NOT `"Error"`), with `resultType: "error"`. Always check `results.resultType`. The `cause` field contains ANSI codes — strip with `re.sub(r'\x1b\[[0-9;]*m', '', cause)`.

### Destroy execution context

```bash
curl -s -X POST -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/1.2/contexts/destroy" \
  -d "{\"clusterId\": \"${CLUSTER_ID}\", \"contextId\": \"${CONTEXT_ID}\"}"
```

---

## SQL Statement API (for profiling — Phase 1)

### Execute SQL

```bash
curl -s -X POST -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/sql/statements/" \
  -d '{"warehouse_id": "148ccb90800933a1",
    "statement": "SELECT COUNT(*) as cnt FROM catalog.schema.table",
    "wait_timeout": "30s", "disposition": "INLINE", "format": "JSON_ARRAY"}'
```

**Response:** `status.state` = `SUCCEEDED` → data in `result.data_array`. Column metadata in `manifest.columns`.
If `state` is `PENDING`, poll: `GET /api/2.0/sql/statements/${STATEMENT_ID}`.

---

## Workspace Import (upload notebook)

### Create parent directory (required before first upload)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/workspace/mkdirs" \
  -d "{\"path\": \"/Users/${USER_IDENTITY}/autoresearch\"}"
```

`USER_IDENTITY` = user email or SP ID from SCIM API (see Authentication / SCIM below).

### Upload/overwrite notebook

```bash
NB_CONTENT=$(base64 -w0 /path/to/notebook.py)
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/workspace/import" \
  -d "{\"path\": \"/Users/${USER_IDENTITY}/autoresearch/table_name_20260319\", \"format\": \"SOURCE\", \"language\": \"PYTHON\", \"content\": \"${NB_CONTENT}\", \"overwrite\": true}"
```

**Note:** `content` must be base64-encoded. `path` should NOT include `.py` — Databricks adds it.

---

## MLflow API (read metrics — source of truth)

Use MLflow API to read metric values (not cell output). Source of truth for experiment comparison.

### Get run metrics

```bash
curl -s -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/2.0/mlflow/runs/get?run_id=${RUN_ID}" | jq '.run.data.metrics'
```

### Search runs in experiment

```bash
curl -s -X POST -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/mlflow/runs/search" \
  -d "{\"experiment_ids\": [\"${EXPERIMENT_ID}\"], \"max_results\": 10, \"order_by\": [\"metrics.auc_pr DESC\"]}"
```

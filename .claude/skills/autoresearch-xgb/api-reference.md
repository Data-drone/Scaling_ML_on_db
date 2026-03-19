# API Reference — Databricks REST APIs

All calls use `curl` with bearer token authentication.

- Token: `$(databricks-token)` (outputs a fresh M2M OAuth token, cached)
- Host: `$DATABRICKS_HOST` env var

## Authentication Pattern

```bash
curl -s \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/..."
```

Or via `databricks api`:

```bash
databricks api post /api/2.0/clusters/create --json '{...}'
```

---

## Cluster Lifecycle

### Create cluster

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/clusters/create" \
  -d '{
    "cluster_name": "autoresearch-TABLENAME-YYYYMMDD-HHMM",
    "spark_version": "17.3.x-cpu-ml-scala2.13",
    "node_type_id": "Standard_D16s_v5",
    "num_workers": 0,
    "autotermination_minutes": 70,
    "data_security_mode": "SINGLE_USER",
    "spark_conf": {
      "spark.databricks.cluster.profile": "singleNode",
      "spark.master": "local[*, 4]"
    },
    "custom_tags": {
      "ResourceClass": "SingleNode",
      "project": "autoresearch"
    },
    "azure_attributes": {
      "availability": "SPOT_WITH_FALLBACK_AZURE",
      "spot_bid_max_price": -1
    }
  }'
```

Response: `{"cluster_id": "0319-123456-abc123"}`

**For Ray distributed (multi-node),** change:
- `"num_workers": 4` (or calculated N)
- Remove `singleNode` profile and `local[*,4]` from spark_conf
- Add `"spark.executorEnv.OMP_NUM_THREADS": "15"` to spark_conf
- Change `ResourceClass` to `"default"`

### Poll cluster status

```bash
curl -s \
  -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/2.0/clusters/get?cluster_id=${CLUSTER_ID}" \
  | jq -r '.state'
```

Poll every 15 seconds until `RUNNING`. Typical: 3-5 minutes.
States: `PENDING` → `RUNNING` (success) or `TERMINATING`/`ERROR` (failure).

### Terminate cluster

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/clusters/delete" \
  -d "{\"cluster_id\": \"${CLUSTER_ID}\"}"
```

---

## Command Execution (Interactive Cells)

### Create execution context

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/1.2/contexts/create" \
  -d "{\"clusterId\": \"${CLUSTER_ID}\", \"language\": \"python\"}"
```

Response: `{"id": "1234567890"}`

### Execute a command (run a cell)

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/1.2/commands/execute" \
  -d "{
    \"clusterId\": \"${CLUSTER_ID}\",
    \"contextId\": \"${CONTEXT_ID}\",
    \"language\": \"python\",
    \"command\": \"print('hello from autoresearch')\"
  }"
```

Response: `{"id": "cmd-abc123"}`

**Important:** The `command` field is a JSON string. For multi-line code, write
the Python to a temp file, read it, and JSON-encode it:

```bash
# Write Python code to temp file
cat > /tmp/cell.py << 'PYEOF'
import pandas as pd
df = spark.table("my_table").toPandas()
print(f"Loaded {len(df):,} rows")
PYEOF

# JSON-encode and send
COMMAND=$(python3 -c "import json; print(json.dumps(open('/tmp/cell.py').read()))")
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/1.2/commands/execute" \
  -d "{
    \"clusterId\": \"${CLUSTER_ID}\",
    \"contextId\": \"${CONTEXT_ID}\",
    \"language\": \"python\",
    \"command\": ${COMMAND}
  }"
```

### Poll command status

```bash
curl -s \
  -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/1.2/commands/status?clusterId=${CLUSTER_ID}&contextId=${CONTEXT_ID}&commandId=${COMMAND_ID}"
```

Poll every 5 seconds. Status values: `Queued` → `Running` → `Finished`/`Error`/`Cancelled`.

**Response on success:**
```json
{
  "id": "cmd-abc123",
  "status": "Finished",
  "results": {
    "resultType": "text",
    "data": "hello from autoresearch\n"
  }
}
```

**Response on error:**
```json
{
  "id": "cmd-abc123",
  "status": "Error",
  "results": {
    "resultType": "error",
    "cause": "NameError: name 'undefined_var' is not defined",
    "summary": "NameError: ..."
  }
}
```

### Destroy execution context

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/1.2/contexts/destroy" \
  -d "{\"clusterId\": \"${CLUSTER_ID}\", \"contextId\": \"${CONTEXT_ID}\"}"
```

---

## SQL Statement API (for profiling — Phase 1)

### Execute SQL

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/sql/statements/" \
  -d '{
    "warehouse_id": "148ccb90800933a1",
    "statement": "SELECT COUNT(*) as cnt FROM catalog.schema.table",
    "wait_timeout": "30s",
    "disposition": "INLINE",
    "format": "JSON_ARRAY"
  }'
```

**Response (inline, success):**
```json
{
  "status": {"state": "SUCCEEDED"},
  "result": {
    "data_array": [["10000000"]],
    "row_count": 1
  },
  "manifest": {
    "columns": [{"name": "cnt", "type_name": "BIGINT"}]
  }
}
```

If `state` is `PENDING`, poll with:

```bash
curl -s \
  -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/2.0/sql/statements/${STATEMENT_ID}"
```

---

## Workspace Import (upload notebook)

### Upload/overwrite notebook

```bash
# Base64-encode the notebook content
NB_CONTENT=$(base64 -w0 /path/to/notebook.py)

curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/workspace/import" \
  -d "{
    \"path\": \"/Users/user@example.com/autoresearch/table_name_20260319\",
    \"format\": \"SOURCE\",
    \"language\": \"PYTHON\",
    \"content\": \"${NB_CONTENT}\",
    \"overwrite\": true
  }"
```

**Note:** The `content` field must be base64-encoded. The `path` should not
include the `.py` extension — Databricks adds it automatically.

---

## MLflow API (read metrics — source of truth)

### Get run metrics

```bash
curl -s \
  -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/2.0/mlflow/runs/get?run_id=${RUN_ID}" \
  | jq '.run.data.metrics'
```

Use this to read metric values rather than parsing cell output. The MLflow API
is the source of truth for experiment comparison.

### Search runs in experiment

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/mlflow/runs/search" \
  -d "{
    \"experiment_ids\": [\"${EXPERIMENT_ID}\"],
    \"max_results\": 10,
    \"order_by\": [\"metrics.auc_pr DESC\"]
  }"
```

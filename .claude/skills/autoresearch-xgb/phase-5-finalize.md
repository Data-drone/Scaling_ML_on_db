# Phase 5: FINALIZE

Compare results, register the best model, finalize the notebook, and tear down.

## Step 1: Pick the winner

From the experiments tracked in Phase 4, identify the run with the highest
AUC-PR. Verify via MLflow API (source of truth):

```bash
curl -s -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/2.0/mlflow/runs/get?run_id=${BEST_RUN_ID}" \
  | jq '.run.data.metrics'
```

## Step 2: Register best model to Unity Catalog

Run on cluster:

```python
uc_model_name = "{catalog}.{schema}.autoresearch_{table_name}"

# Model was already logged in Phase 4 training runs.
# Register from the best run:
result = mlflow.register_model(
    f"runs:/{best_run_id}/model",
    uc_model_name,
)
print(f"Registered model: {uc_model_name}")
print(f"Version: {result.version}")
```

Add to notebook: `## 11. Model Registration` with UC model path and version.

## Step 3: Update notebook summary

Use the Edit tool to replace the placeholder summary at the top of the local
notebook file. Replace the line:

    _Summary will be added after experiments complete._

With the full results summary:

```
# MAGIC ## Results Summary
# MAGIC
# MAGIC | Experiment | AUC-PR | AUC-ROC | F1 | Train Time |
# MAGIC |------------|--------|---------|-----|------------|
# MAGIC | Baseline   | {baseline_auc_pr:.4f} | {baseline_auc_roc:.4f} | {baseline_f1:.4f} | {baseline_time:.1f}s |
# MAGIC | Exp 2      | {exp2_auc_pr:.4f} | ... | ... | ... |
# MAGIC | ...        | ... | ... | ... | ... |
# MAGIC
# MAGIC **Best model:** {best_experiment} (AUC-PR: {best_auc_pr:.4f})
# MAGIC **Registered as:** `{uc_model_name}` v{version}
# MAGIC **MLflow run:** `{best_run_id}`
# MAGIC
# MAGIC ## Decisions Made
# MAGIC
# MAGIC - **Track:** {track} — {reason}
# MAGIC - **Cluster:** {node_type} × {num_workers} workers ({total_ram} GB RAM)
# MAGIC - **Features:** {n_final_features} used (dropped {n_high_card} high-cardinality, {n_high_null} high-null)
# MAGIC - **Encoded:** {n_encoded} categorical columns via ordinal encoding
# MAGIC - **Imbalance:** scale_pos_weight = {spw:.2f}
# MAGIC - **Total time:** {total_minutes:.1f} minutes (budget: {budget_minutes} min)
```

## Step 4: Upload final notebook

Upload the completed notebook to the workspace:

```bash
NB_CONTENT=$(base64 -w0 /workspace/group/scaling_xgb_work/notebooks/autoresearch/{notebook_file})

curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/workspace/import" \
  -d "{
    \"path\": \"/Users/{user_email}/autoresearch/{table_name}_{YYYYMMDD}\",
    \"format\": \"SOURCE\",
    \"language\": \"PYTHON\",
    \"content\": \"${NB_CONTENT}\",
    \"overwrite\": true
  }"
```

## Step 5: Terminate cluster

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(databricks-token)" \
  -H "Content-Type: application/json" \
  "${DATABRICKS_HOST}/api/2.0/clusters/delete" \
  -d "{\"cluster_id\": \"${CLUSTER_ID}\"}"
```

Verify termination:

```bash
curl -s -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/2.0/clusters/get?cluster_id=${CLUSTER_ID}" \
  | jq -r '.state'
```

Expected: `TERMINATING` or `TERMINATED`.

## Step 6: Report to user

Send a summary message with:

- Best model: experiment name, AUC-PR, AUC-ROC, F1
- Notebook location: workspace path (linkable in Databricks UI)
- MLflow experiment: path and run ID
- UC model: registered model name and version
- Cluster config: track, node type, workers
- Total time: minutes elapsed
- Number of experiments run
- Key feature engineering decisions

# Phase 5: FINALIZE

Compare results, register the best model, finalize the notebook, and tear down.

## Step 1: Pick the winner

From the experiments tracked in Phase 4, identify the run with the highest primary metric. Verify via MLflow API (api-reference.md).

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

Use the Edit tool to replace `_Summary will be added after experiments complete._` at the top of the notebook with (see notebook-format.md for `# MAGIC` syntax):

- **Results Summary** table: Experiment | primary metric | secondary metrics | Train Time
- **Best model:** name, metric value, UC model name + version, MLflow run ID
- **Decisions Made:** track + reason, cluster config, feature counts/drops, encoding, imbalance handling, total time vs budget

## Step 4: Upload final notebook

Upload via Workspace Import API (api-reference.md). Path: `/Users/{user_email}/autoresearch/{table_name}_{YYYYMMDD}`.

## Step 5: Terminate cluster

Terminate via Clusters API (api-reference.md). Verify state is `TERMINATING` or `TERMINATED`.

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

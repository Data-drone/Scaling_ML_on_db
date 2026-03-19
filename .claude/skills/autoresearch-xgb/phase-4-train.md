# Phase 4: TRAIN & EVALUATE

Run XGBoost training experiments on the live cluster. Start with a baseline,
then try variations if budget allows.

## Budget Guard

Before each experiment, check remaining time:

```python
import time
elapsed_minutes = (time.time() - budget_start) / 60
remaining = budget_minutes - elapsed_minutes
phase5_reserve = budget_minutes * 0.2  # 20% for finalize

if remaining < phase5_reserve:
    print(f"Budget guard: {remaining:.1f} min left, need {phase5_reserve:.1f} for finalize")
    print("Skipping remaining experiments — moving to Phase 5")
    # Go to Phase 5
```

Run this check as a cell before each experiment.

## Progress Guard

Track best metric and stop if stagnant:

```
best_auc_pr = 0.0
best_run_id = None
best_experiment = None
experiments_without_improvement = 0

# After each experiment:
if new_auc_pr > best_auc_pr:
    best_auc_pr = new_auc_pr
    best_run_id = run_id
    best_experiment = experiment_name
    experiments_without_improvement = 0
else:
    experiments_without_improvement += 1
    if experiments_without_improvement >= 2:
        print("Progress guard: no improvement in 2 experiments — stopping")
        # Go to Phase 5
```

## Experiment 1: Baseline (always runs)

### Cell: Baseline training (single-node)

```python
import xgboost as xgb
import time, os
from sklearn.metrics import (average_precision_score, roc_auc_score,
    f1_score, confusion_matrix, classification_report)
from mlflow.models import infer_signature

xgb_params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": os.cpu_count(),
    "random_state": 42,
    "verbosity": 1,
}

with mlflow.start_run(run_name="baseline", log_system_metrics=True) as run:
    mlflow.log_params({
        "experiment": "baseline",
        "table": "{table}",
        "n_rows": len(X_train) + len(X_test),
        "n_features": X_train.shape[1],
        **{f"xgb_{k}": v for k, v in xgb_params.items()},
    })

    train_start = time.time()
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    train_time = time.time() - train_start
    mlflow.log_metric("train_time_sec", train_time)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_pr = average_precision_score(y_test, y_proba)
    auc_roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metrics({"auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1})

    sig = infer_signature(X_test.head(100), model.predict_proba(X_test.head(100)))
    mlflow.sklearn.log_model(model, "model", signature=sig)

    baseline_run_id = run.info.run_id
    print(f"Baseline: AUC-PR={auc_pr:.4f} | AUC-ROC={auc_roc:.4f} | F1={f1:.4f}")
    print(f"Train time: {train_time:.1f}s | Run ID: {baseline_run_id}")
    print(classification_report(y_test, y_pred))
```

After this cell, **read metrics from MLflow API** (source of truth):

```bash
curl -s -H "Authorization: Bearer $(databricks-token)" \
  "${DATABRICKS_HOST}/api/2.0/mlflow/runs/get?run_id=${BASELINE_RUN_ID}" \
  | jq '.run.data.metrics[] | select(.key == "auc_pr") | .value'
```

Update best tracking: `best_auc_pr`, `best_run_id`, `best_experiment`.

Add to notebook with markdown: `## 6. Training — Baseline` with config table
and results summary.

**Upload notebook checkpoint** after baseline completes.

## Experiment 2: Shallower trees (if budget allows)

Check budget guard first.

Same as baseline but:
- `"max_depth": 4`
- `run_name="exp2_depth4"`
- `mlflow.log_param("experiment", "depth4")`

Compare AUC-PR to best. Update tracking.

Add to notebook: `## 7. Experiment 2 — Shallower Trees (depth=4)`

## Experiment 3: Deeper trees + more rounds (if budget allows)

Check budget guard and progress guard.

Same as baseline but:
- `"max_depth": 8, "n_estimators": 200`
- `run_name="exp3_depth8_200r"`
- `mlflow.log_param("experiment", "depth8_200r")`

Add to notebook: `## 8. Experiment 3 — Deeper Trees (depth=8, 200 rounds)`

## Experiment 4: Feature importance pruning (if budget allows)

Check budget guard and progress guard.

```python
# Get feature importances from baseline
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X_train.columns.tolist(),
    "importance": importances
}).sort_values("importance", ascending=True)

n_drop = int(len(X_train.columns) * 0.2)
drop_cols = importance_df.head(n_drop)["feature"].tolist()
X_train_pruned = X_train.drop(columns=drop_cols)
X_test_pruned = X_test.drop(columns=drop_cols)

# Retrain with pruned features
# run_name="exp4_pruned", experiment="pruned_80pct"
```

Add to notebook: `## 9. Experiment 4 — Feature Pruning (top 80%)`

## Results Comparison Cell

After all experiments:

```python
experiment = mlflow.get_experiment_by_name(experiment_path)
runs_df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.auc_pr DESC"],
    max_results=10,
)
cols = ["run_id", "params.experiment", "metrics.auc_pr",
        "metrics.auc_roc", "metrics.f1", "metrics.train_time_sec"]
print(runs_df[[c for c in cols if c in runs_df.columns]].to_string())
```

Add to notebook: `## 10. Results Comparison` with a markdown table.

## For Ray Distributed Track

When `track == "ray-distributed"`, the training is fundamentally different.
Use the pattern from `.claude/skills/train-xgb-databricks/track-ray-distributed.md`:

- Data loaded via `ray.data.read_databricks_tables()` (not pandas)
- Training uses `DataParallelTrainer` with `XGBoostConfig`
- OMP 3-layer fix must be applied
- Model extracted from Ray checkpoint
- **Critical:** Strip `https://` from `DATABRICKS_HOST` before passing to Ray
  (see gotchas.md G9)

The same experiment structure applies (baseline + variations), but the cell
code is the Ray recipe instead of single-node sklearn-style.

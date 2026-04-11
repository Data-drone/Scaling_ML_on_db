# Phase 4: TRAIN & EVALUATE

Run XGBoost training experiments on the live cluster. Start with a baseline,
then try variations if budget allows.

## Budget Guard

Run before each experiment. Reserve 20% of `budget_minutes` for Phase 5:

```python
elapsed = (time.time() - budget_start) / 60
remaining = budget_minutes - elapsed
if remaining < budget_minutes * 0.2:
    # Skip remaining experiments → Phase 5
```

## Progress Guard

Primary metric: `auc_pr` (binary) or `accuracy` (multiclass). Track `best_metric_value`, `best_run_id`, `best_experiment`, `experiments_without_improvement`.

```
After each experiment:
  if new_metric > best_metric_value → update best, reset counter
  else → increment counter; if >= 2 → stop, go to Phase 5
```

## Experiment 1: Baseline (always runs)

### Cell: Baseline training (single-node)

Build dynamically based on `task_type` from Phase 3.

**Baseline params:** `tree_method=hist`, `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `n_jobs=os.cpu_count()`, `random_state=42`.
- Binary: `objective=binary:logistic`, `scale_pos_weight`, `eval_metric=aucpr`
- Multiclass: `objective=multi:softprob`, `num_class=n_classes`, `eval_metric=mlogloss`

In `mlflow.start_run(run_name="baseline", log_system_metrics=True)`:
- Log params: experiment, task_type, table, n_rows, n_features, all xgb_params
- Train XGBClassifier, time it
- Binary metrics: `average_precision_score`, `roc_auc_score`, `f1_score` (using `predict_proba`)
- Multiclass metrics: `accuracy_score`, `f1_score(average='macro')`
- Log model with `infer_signature`, print `classification_report`

Verify metrics via MLflow API (api-reference.md). Update `best_metric_value`, `best_run_id`, `best_experiment`.

Add to notebook with markdown: `## 6. Training — Baseline` with config table
and results summary.

**Upload notebook checkpoint** after baseline completes.

### Sanity Check (REQUIRED after baseline — see LEARNINGS.md L16)

After the baseline experiment completes, verify that the model actually learned:

```python
# Sanity check: detect pipeline bugs that produce random-chance models
if task_type == "binary":
    minority_ratio = y_test.sum() / len(y_test) if hasattr(y_test, 'sum') else 0.02
    random_chance_aucpr = minority_ratio  # AUC-PR at random = class proportion
    if metrics["auc_pr"] < max(0.1, random_chance_aucpr * 3):
        print(f"SANITY CHECK FAILED: AUC-PR={metrics['auc_pr']:.4f} is near random chance ({random_chance_aucpr:.4f})")
        print("This likely indicates a pipeline bug, not a hard dataset.")
        print("Common causes:")
        print("  - Feature column ordering mismatch between train and eval")
        print("  - sklearn-style params passed to native xgboost.train() API")
        print("  - Labels shuffled during Ray Data split")
        print("  - DMatrix built with wrong feature/label mapping")
        mlflow.log_param("sanity_check", "FAILED_near_random")
        mlflow.log_param("sanity_check_detail", f"auc_pr={metrics['auc_pr']:.4f} < threshold={max(0.1, random_chance_aucpr * 3):.4f}")
        # DO NOT continue to experiments 2-4 — fix the pipeline first
        # Skip to Phase 5 with a warning
elif task_type == "multiclass":
    random_chance_acc = 1.0 / n_classes
    if metrics.get("accuracy", 0) < random_chance_acc * 1.5:
        print(f"SANITY CHECK FAILED: accuracy={metrics['accuracy']:.4f} is near random chance ({random_chance_acc:.4f})")
        mlflow.log_param("sanity_check", "FAILED_near_random")
```

**If sanity check fails:** Do NOT run experiments 2-4. Log the failure, add a
markdown cell documenting the issue, and go directly to Phase 5. The model
produced garbage predictions — further hyperparameter variations will also
produce garbage. The fix is in the data pipeline, not the model config.

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

Same as baseline but: drop bottom 20% features by `model.feature_importances_` from baseline. `run_name="exp4_pruned"`, `experiment="pruned_80pct"`.

Add to notebook: `## 9. Experiment 4 — Feature Pruning (top 80%)`

## Results Comparison Cell

Use `mlflow.search_runs()` ordered by `primary_metric DESC`. Print comparison table with run_id, experiment name, metrics, and train time.

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

### Ray Track: Critical Differences from Single-Node (see LEARNINGS.md L16)

**DO NOT copy single-node code patterns into Ray training.** The APIs are
fundamentally different and mixing them produces silently broken models:

| Aspect | Single-Node (sklearn) | Ray Distributed (native) |
|--------|-----------------------|--------------------------|
| API | `XGBClassifier(**params)` | `xgboost.train(params, dtrain, ...)` |
| Learning rate param | `learning_rate=0.1` | `eta=0.1` (NOT `learning_rate`) |
| Rounds param | `n_estimators=100` | `num_boost_round=100` (separate arg) |
| eval_metric | `eval_metric="aucpr"` | `eval_metric=["aucpr"]` (in params dict) |
| Feature columns | Preserved by DataFrame | Must be consistent across `shard_to_dmatrix()` calls |
| Prediction | `model.predict_proba(X)[:, 1]` | `booster.predict(DMatrix(X))` (returns probabilities directly) |

**Column ordering in `shard_to_dmatrix()`:**
Use `sorted(c for c in batch.keys() if c != label_col)` to ensure consistent
column ordering across all shards AND the eval DMatrix. If train uses sorted
columns but eval uses schema-order columns, the model appears to predict randomly.

**The eval DMatrix MUST use the same column ordering as train:**
```python
# CORRECT: Same sorted column ordering for eval
feature_columns_sorted = sorted(feature_columns)
X_test_eval = np.column_stack([batch[c] for c in feature_columns_sorted])
```

**The sanity check from the baseline section applies equally to Ray runs.**
If AUC-PR < 0.1 on a binary task after Ray baseline, the pipeline is broken.

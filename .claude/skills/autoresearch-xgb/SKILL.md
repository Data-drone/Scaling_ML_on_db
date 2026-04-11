---
name: autoresearch-xgb
description: Autonomous XGBoost research agent. Given a Unity Catalog table and target column, profiles the data, creates a right-sized cluster, runs iterative experiments via Command Execution API, and produces a polished Databricks notebook. Use when asked to "research", "auto-train", "find the best model", or "explore and train" on a dataset.
---

# AutoResearch: XGBoost on Databricks

Autonomous agent that explores a dataset and produces the best XGBoost model
it can find within a time budget. The output is a well-documented Databricks
notebook uploaded to the workspace.

## Inputs

Extract from the user's request:

| Input | Required | Example | Default |
|-------|----------|---------|---------|
| `table` | yes | `brian_gen_ai.xgb_scaling.imbalanced_10m` | — |
| `target_col` | yes | `label` | — |
| `budget_minutes` | no | `45` | `60` |

Parse `catalog`, `schema`, `table_name` from the three-part table identifier.

## Checklist

Copy this and track progress:

```
- [ ] Phase 1: PROFILE — Profile the dataset via SQL warehouse
- [ ] Phase 2: SIZE & CREATE — Calculate cluster size, create cluster + context
- [ ] Phase 3: FEATURE ENG — Run feature engineering cells on live cluster
- [ ] Phase 4: TRAIN — Run baseline + experiments, compare results
- [ ] Phase 5: FINALIZE — Clean up notebook, register model, teardown
```

## Phase 1: PROFILE — See [phase-1-profile.md](phase-1-profile.md). SQL profiling (no cluster needed).
## Phase 2: SIZE & CREATE — See [phase-2-cluster.md](phase-2-cluster.md). Memory estimate → cluster config → create & wait.
## Phase 3: FEATURE ENG — See [phase-3-features.md](phase-3-features.md). Load, auto-detect types, encode, nulls, split. Upload checkpoint.
## Phase 4: TRAIN — See [phase-4-train.md](phase-4-train.md). Baseline + variations with budget/progress guards. MLflow tracking.
## Phase 5: FINALIZE — See [phase-5-finalize.md](phase-5-finalize.md). Register model, summary notebook, terminate cluster, report.

## References

- **API calls:** [api-reference.md](api-reference.md)
- **Notebook format:** [notebook-format.md](notebook-format.md)
- **Design doc:** `docs/plans/2026-03-19-autoresearch-agent-design.md`
- **Existing XGB skill:** `.claude/skills/train-xgb-databricks/` (for Ray patterns)
- **Deploy skill:** `.claude/skills/deploy-notebook-jobs/` (for error diagnosis)

## Critical Configuration Notes

- **Unity Catalog access:** Clusters MUST include `data_security_mode: "SINGLE_USER"`
  AND `single_user_name: "<SP_ID_or_email>"`. Without these, `spark.read.table()` fails
  with `UC_NOT_ENABLED`. Get user identity from SCIM API: `GET /api/2.0/preview/scim/v2/Me`.
- **SQL Statement API `wait_timeout`:** Max is `50s`, not `60s`. Values > 50s return 400.
- **Command execution error status:** Errors return `status: "Finished"` with
  `results.resultType: "error"`, NOT `status: "Error"`. Always check `resultType`.

## Error Handling

- Cell error → read error, one retry with fix, if still fails log in markdown
- Cluster dies → save local notebook, upload partial, report to user
- OOM → log error, try fewer features or smaller sample
- Budget exceeded → skip remaining experiments, go to Phase 5
- No metric improvement after 2 experiments → stop early, go to Phase 5

# STATUS: Single-Node Scaling Track

**Branch:** `feat/single-node-scaling`
**Last updated:** 2025-02-22
**Status:** BASELINES ESTABLISHED — ready for larger data experiments

---

## Current State

Single-node XGBoost baselines are established on 3 VM sizes. These serve as the reference points for all distributed scaling experiments.

### Completed Experiments

| Dataset | Node Type | vCPUs | RAM | Train Time | Total Time | AUC-PR |
|---------|-----------|-------|-----|-----------|-----------|--------|
| 1M rows, 100 features | D16s_v5 | 16 | 64 GB | 5-6s | 21-25s | 0.9966 |
| 10M rows, 250 features | E16s_v5 | 16 | 128 GB | 128s | 186s | 1.0000 |
| 10M rows, 250 features | E32s_v5 | 32 | 256 GB | 76s | 115s | 1.0000 |

### Key Observations

- **D16 → E16 at 10M:** Same vCPUs, 2x RAM → no speedup. CPU-bound at this scale.
- **E16 → E32 at 10M:** 2x vCPUs → 1.68x speedup (128s → 76s). Near-linear CPU scaling.
- **1M dataset:** Too small for meaningful single-node benchmarking (5s train time).
- **AUC-PR = 1.0:** Synthetic data is too easy at 10M rows. Consider harder generation params.

---

## Next Steps

1. **D16 at 10M** — Fill the gap: `medium` data on `D16sv5`. Expected: ~128s (same as E16, CPU-limited).
2. **30M rows on E32** — Push single-node limits. `medium_large` preset. Expected: ~230s train, may fit in 256 GB.
3. **100M rows on E32** — Expected to OOM. This establishes the ceiling where distributed training is required.
4. **Harder synthetic data** — Reduce `n_informative_num` to get AUC-PR < 1.0.

---

## Environment

- **Notebook:** `notebooks/train_xgb_single.ipynb`
- **Runtime:** `17.3.x-cpu-ml-scala2.13`
- **Cluster mode:** Single-node (`spark.databricks.cluster.profile: singleNode`)
- **Config file:** `configs/single_node.yml`
- **MLflow experiment:** `/Users/brian.law@databricks.com/xgb_scaling_benchmark`

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| XGBoost tree_method | `hist` | Histogram-based is fastest for large data; `approx` and `exact` not competitive |
| Eval metric | `aucpr` primary | Better than AUC-ROC for imbalanced data (5% minority class) |
| n_estimators | 100 | Enough to converge; increasing to 200+ adds time without improving AUC |
| max_depth | 8 | Good balance; 6 slightly underfits, 10 overfits at 1M |

---

## Relevant Learnings

- **L5:** Worker CPU ~70% is expected (not a bottleneck) — also applies to single-node
- **L9:** AUC-PR = 1.0 suggests synthetic data is too easy
- **L10:** ML Runtime version (`-cpu-ml-`) is required for pre-installed XGBoost
- **L11:** Azure Spot instances fail more on larger VMs

# STATUS: GPU Scaling Track

**Branch:** `feat/gpu-scaling`
**Last updated:** 2025-02-22
**Status:** PLANNED — no experiments run yet

---

## Current State

This track has not started yet. It depends on:
1. GPU ML Runtime availability (`17.3.x-gpu-ml-scala2.13`)
2. Azure GPU VM quota (NC-series)
3. Baseline CPU results from `feat/single-node-scaling` and `feat/ray-scaling`

---

## Planned Experiments

### Phase 1: Single-GPU Baselines

| # | Data | Node | GPU | Goal |
|---|------|------|-----|------|
| 1 | 1M | NC6s_v3 | 1x V100 (16 GB) | Baseline — expect ~2-3s train |
| 2 | 10M | NC6s_v3 | 1x V100 (16 GB) | Does 16 GB GPU memory fit 10M x 250? |
| 3 | 10M | NC4as_T4_v3 | 1x T4 (16 GB) | Cost comparison — T4 is much cheaper |
| 4 | 30M | NC12s_v3 | 2x V100 (32 GB) | Multi-GPU single-node |

### Phase 2: Multi-GPU via Ray

| # | Data | Node | GPUs | Workers | Goal |
|---|------|------|------|---------|------|
| 5 | 10M | NC16as_T4_v3 | 4x T4 | 4 | Multi-GPU distributed via Ray |
| 6 | 100M | NC12s_v3 | 2x V100 | 4 | 8 GPUs total for large-scale |

### Phase 3: GPU vs CPU Comparison

| # | Data | GPU Config | CPU Config | Goal |
|---|------|-----------|-----------|------|
| 7 | 10M | 1x V100 | 4x D16 (56 CPUs) | Cost-per-second comparison |
| 8 | 100M | 4x T4 | 8x D16 (112 CPUs) | At scale, which wins? |

---

## Prerequisites

Before starting this track:

- [ ] Confirm GPU ML Runtime supports Ray on Spark
- [ ] Check Azure GPU VM quotas (NC-series may need quota increase)
- [ ] Verify XGBoost GPU histogram is available in ML Runtime 17.3
- [ ] Create `notebooks/train_xgb_gpu.ipynb` (new notebook)
- [ ] Add GPU jobs to `databricks.yml`
- [ ] Test that `src/validate_env.py` correctly detects GPU presence

---

## Open Questions

1. **Does OMP_NUM_THREADS affect GPU XGBoost?**
   - `tree_method=gpu_hist` does most work on GPU, but data preprocessing may use CPU
   - Hypothesis: OMP_NUM_THREADS matters less for GPU but still affects data loading

2. **GPU memory limits for DMatrix**
   - XGBoost GPU builds DMatrix in GPU memory
   - V100 = 16 GB, T4 = 16 GB
   - 10M x 250 features ≈ 20 GB raw → may not fit without compression
   - Need to check if `external_memory` mode works with GPU

3. **Multi-GPU via Ray vs XGBoost native**
   - XGBoost has native multi-GPU support via `gpu_id` parameter
   - Ray DataParallelTrainer assigns one GPU per worker
   - Which approach is better for multi-GPU?

4. **T4 vs V100 for XGBoost**
   - V100 has higher memory bandwidth (900 GB/s vs 320 GB/s)
   - T4 is ~4x cheaper per hour
   - XGBoost histogram is memory-bandwidth bound → V100 may be disproportionately fast

---

## Environment (Planned)

- **Notebook:** `notebooks/train_xgb_gpu.ipynb` (to be created)
- **Runtime:** `17.3.x-gpu-ml-scala2.13`
- **Config file:** `configs/gpu_scaling.yml`
- **MLflow experiment:** `/Users/brian.law@databricks.com/xgb_scaling_benchmark`

---

## Relevant Learnings

- **L1:** OMP_NUM_THREADS — verify if this affects GPU training path
- **L10:** Must use GPU ML Runtime (`-gpu-ml-`) for CUDA support
- **L11:** Azure spot instances — GPU VMs may be even harder to get as spot

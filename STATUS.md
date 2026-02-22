# STATUS: GPU Scaling Track

**Branch:** `feat/gpu-scaling`
**Last updated:** 2025-02-22
**Status:** READY FOR TESTING -- notebook and job definitions created

---

## Current State

GPU-accelerated XGBoost training notebook and Databricks job definitions are in place.
No experiments have been run yet. The track is ready for first smoke tests on NCT4 VMs.

### Artifacts Created

| Artifact | Path | Description |
|----------|------|-------------|
| GPU Training Notebook | `notebooks/train_xgb_gpu.ipynb` | XGBoost with `tree_method=gpu_hist`, MLflow tracking, GPU memory estimation |
| GPU Config | `configs/gpu_scaling.yml` | Node type specs, planned experiments, XGBoost GPU params |
| Job Definitions | `databricks.yml` | 4 GPU jobs: 2 configurable + 2 perf tests |

### Job Definitions

| Job Key | VM Type | GPUs | Description |
|---------|---------|------|-------------|
| `train_xgb_gpu_nc4_t4` | NC4as_T4_v3 | 1x T4 (16 GB) | Single T4 GPU, 4 cores, 28 GB RAM |
| `train_xgb_gpu_nc16_t4` | NC16as_T4_v3 | 4x T4 (64 GB) | Multi-GPU, 16 cores, 110 GB RAM |
| `perf_gpu_1m_nc4_t4` | NC4as_T4_v3 | 1x T4 | 1M rows baseline on single T4 |
| `perf_gpu_10m_nc4_t4` | NC4as_T4_v3 | 1x T4 | 10M rows scaling test on single T4 |

### Notebook Features

- **GPU detection** via `nvidia-smi` with error handling
- **GPU memory estimation** before training (raw data x6 for DMatrix overhead)
- **Environment validation** using `src/validate_env.py` with `require_gpu=True`
- **MLflow logging** of GPU-specific params: gpu_type, gpu_memory_gb, tree_method
- **Error tracking** with graceful exit on failure
- All jobs use **MLR 17.3 LTS GPU** (`17.3.x-gpu-ml-scala2.13`)

---

## Planned Experiments

### Phase 1: Single-GPU Baselines (NEXT)

| # | Data | Node | GPU | Goal |
|---|------|------|-----|------|
| 1 | 1M | NC4as_T4_v3 | 1x T4 (16 GB) | Baseline -- compare to CPU D16 single-node |
| 2 | 10M | NC4as_T4_v3 | 1x T4 (16 GB) | Does 16 GB GPU memory fit 10M x 250? |
| 3 | 10M | NC16as_T4_v3 | 4x T4 (64 GB) | Multi-GPU single-node |

### Phase 2: GPU vs CPU Comparison

| # | Data | GPU Config | CPU Config | Goal |
|---|------|-----------|-----------|------|
| 4 | 1M | 1x T4 | D16 (16 CPUs) | Cost-per-second comparison at small scale |
| 5 | 10M | 1x T4 | E32 (32 CPUs) | At medium scale, which wins? |

---

## Prerequisites

Before running experiments:

- [x] Create `notebooks/train_xgb_gpu.ipynb`
- [x] Add GPU jobs to `databricks.yml`
- [x] Verify `src/validate_env.py` supports GPU detection (`require_gpu=True`)
- [ ] Confirm GPU ML Runtime supports XGBoost `gpu_hist` in MLR 17.3
- [ ] Check Azure GPU VM quotas (NC-series may need quota increase)
- [ ] Run smoke test on NC4as_T4_v3
- [ ] Verify GPU memory estimation accuracy

---

## Open Questions

1. **Does OMP_NUM_THREADS affect GPU XGBoost?**
   - `tree_method=gpu_hist` does most work on GPU, but data preprocessing may use CPU
   - Hypothesis: OMP_NUM_THREADS matters less for GPU but still affects data loading

2. **GPU memory limits for DMatrix**
   - XGBoost GPU builds DMatrix in GPU memory
   - T4 = 16 GB
   - 10M x 250 features = ~20 GB raw -> may not fit without compression
   - Need to check if `external_memory` mode works with GPU

3. **Multi-GPU via Ray vs XGBoost native**
   - XGBoost has native multi-GPU support via `gpu_id` parameter
   - Ray DataParallelTrainer assigns one GPU per worker
   - Which approach is better for multi-GPU?

4. **T4 cost-effectiveness for XGBoost**
   - T4 has 320 GB/s memory bandwidth (vs V100 900 GB/s)
   - T4 is ~4x cheaper per hour
   - XGBoost histogram is memory-bandwidth bound -> T4 may be surprisingly competitive

---

## Environment

- **Notebook:** `notebooks/train_xgb_gpu.ipynb`
- **Runtime:** `17.3.x-gpu-ml-scala2.13`
- **Cluster mode:** Single-node (`spark.databricks.cluster.profile: singleNode`)
- **Config file:** `configs/gpu_scaling.yml`
- **MLflow experiment:** `/Users/brian.law@databricks.com/xgb_scaling_benchmark`

---

## Relevant Learnings

- **L1:** OMP_NUM_THREADS -- verify if this affects GPU training path
- **L10:** Must use GPU ML Runtime (`-gpu-ml-`) for CUDA support
- **L11:** Azure spot instances -- GPU VMs may be even harder to get as spot

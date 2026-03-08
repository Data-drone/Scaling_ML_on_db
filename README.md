# Scaling XGBoost on Databricks

Comparing XGBoost training efficiency across single-node CPU, distributed CPU (Ray on Spark), and GPU approaches as datasets scale from 1M to 500M+ rows on Databricks.

## Objective

Determine the most efficient way to train XGBoost at scale on Databricks by measuring training time, resource utilisation, and cost across three approaches at increasing data sizes (1M → 10M → 30M → 100M → 500M rows).

## Approaches

### 1. Single-Node CPU
Standard python XGBoost (`tree_method=hist`) running on the Spark driver node. No distribution framework. Serves as the baseline — simple to run, limited by single-machine memory and CPU.

- **Notebook:** `notebooks/train_xgb_single.ipynb`
- **Hardware:** D16s/E16s/E32s Azure VMs (16–32 vCPU, 64–256 GB RAM)

### 2. Multi-Node CPU (Ray on Spark)
Distributed XGBoost via Ray `DataParallelTrainer` running across Spark executors. Data loaded through Ray Data from Unity Catalog. Includes OMP thread-scaling fix and per-worker system metrics.

- **Notebook:** `notebooks/train_xgb_ray.ipynb`
- **Hardware:** 2–8 worker nodes (D8s/D16s/E16s), Ray on Spark

### 3. GPU (Single-Node + Multi-Node via Ray)
XGBoost with `device=cuda` on NVIDIA T4 GPUs. Single-node uses native XGBoost GPU. Multi-node planned via Ray with 1 GPU per actor.

- **Notebook:** `notebooks/train_xgb_gpu.ipynb`
- **Hardware:** NC4as_T4_v3 (1×T4), NC16as_T4_v3 (4×T4) Azure VMs

## Repo Structure

```
.
├── notebooks/
│   ├── train_xgb_single.ipynb        # Single-node CPU baseline
│   ├── train_xgb_ray.ipynb           # Distributed CPU via Ray on Spark
│   ├── train_xgb_gpu.ipynb           # GPU training (single + multi-node)
│   └── generate_imbalanced_data.ipynb # Synthetic dataset generation
├── src/                               # Shared utilities (config, validation, benchmark)
├── configs/                           # Per-track experiment configs (YAML)
├── databricks.yml                     # Databricks Asset Bundle job definitions
├── docs/
│   ├── PROJECT_PLAN.md               # Experiment roadmap and status
│   ├── LEARNINGS.md                  # Accumulated findings and debugging notes
│   └── DEPLOYMENT.md                 # Deploy and troubleshoot guide
├── AGENTS.md                          # Per-notebook coding guidelines
└── pyproject.toml
```

## Quick Start

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run local tests
pytest tests/

# Validate Databricks bundle config
databricks bundle validate -t dev

# Deploy to Databricks
databricks bundle deploy -t dev

# Run a job
databricks bundle run -t dev train_xgb_single_e16
```

## Docs

- [Project Plan](docs/PROJECT_PLAN.md) — experiment roadmap, status, what's next
- [Learnings](docs/LEARNINGS.md) — accumulated findings from experiments
- [Deployment](docs/DEPLOYMENT.md) — deploy commands and error debugging
- [Agent Guidelines](AGENTS.md) — per-notebook coding conventions

# Project Plan: Scaling XGBoost on Databricks

## Goals

Answer three questions:

1. **At what data size does single-node CPU stop being practical?** (memory limits, training time)
2. **How does Ray distributed CPU scale with workers and data size?** (scaling efficiency, overhead, cost/benefit)
3. **When does GPU beat distributed CPU?** (crossover point, cost per training run)

### Success Criteria

- Complete experiment matrix below with training time + total time for each cell
- Identify the recommended approach for each data size tier (1M, 10M, 30M, 100M)
- Document cost estimates (DBU-hours) for each approach at 10M and 100M scale

## Experiment Matrix

### Single-Node CPU

| Data Size | Rows | Features | D16s (16c/64G) | E16s (16c/128G) | E32s (32c/256G) |
|-----------|------|----------|-----------------|------------------|------------------|
| small     | 1M   | 100      | ✅ 5-6s / 21-25s | ⬜ | ⬜ |
| medium    | 10M  | 250      | ⬜ | ✅ 128s / 186s | ✅ 76s / 115s |
| medium_large | 30M | 250   | ⬜ | ⬜ | ⬜ planned |
| large     | 100M | 500      | — | — | ⬜ expected OOM |

Format: `train_time / total_time` or status

### Ray on Spark (Distributed CPU)

| Data Size | Workers × Node | CPUs/Worker | OMP Fix | Train Time | Total Time | Notes |
|-----------|----------------|-------------|---------|------------|------------|-------|
| 1M | 2 × D8s | 6 | Yes | 45s | 101s | |
| 1M | 4 × D8s | 6 | Yes | 37s | 92s | |
| 1M | 8 × D8s | 6 | Yes | 38s | 96s | Diminishing returns |
| 1M | 4 × D16s | 14 | Yes | ⬜ | ⬜ | D16 vs D8 comparison |
| 10M | 2 × D16s | 14 | No | 510s | 560s | Pre-OMP fix |
| 10M | 2 × E16s | 14 | No | 511s | 558s | Pre-OMP fix |
| 10M | 4 × D16s | 14 | Yes | 80s | 128s | 3.4× speedup from OMP fix |
| 10M | 4 × E16s | 14 | Yes | 79s | 126s | |
| 10M | 2 × D16s | 14 | Yes | ⬜ | ⬜ | OMP validation |
| 30M | 4 × D16s | 14 | Yes | ⬜ | ⬜ | planned |
| 30M | 8 × D16s | 14 | Yes | ⬜ | ⬜ | planned |
| 100M | 8 × D16s | 14 | Yes | ⬜ | ⬜ | planned — Plasma tuning may matter here |
| 100M | 8 × E16s | 14 | Yes | ⬜ | ⬜ | planned — 32GB object store |

### GPU

| Data Size | Node Type | GPUs | Train Time | Total Time | Notes |
|-----------|-----------|------|------------|------------|-------|
| 1M | NC4as_T4 (1×T4, 4c/28G) | 1 | ⬜ | ⬜ | planned — baseline |
| 10M | NC4as_T4 (1×T4, 4c/28G) | 1 | ⬜ | ⬜ | planned — fits in 16GB GPU mem? |
| 10M | NC16as_T4 (4×T4, 16c/110G) | 1 | ⬜ | ⬜ | planned — single GPU on multi-GPU node |
| 30M | NC16as_T4 (4×T4, 16c/110G) | 4 | ⬜ | ⬜ | planned — multi-GPU via Ray |
| 100M | NC16as_T4 (4×T4, 16c/110G) | 4 | ⬜ | ⬜ | planned — GPU memory limit test |

## Current State (March 2026)

### What's working
- **Single-node CPU:** Notebook runs end-to-end, tested at 1M and 10M. Results logged to MLflow.
- **Ray distributed CPU:** Notebook runs end-to-end with OMP fix, tested at 1M (2/4/8 workers) and 10M (2/4 workers). Worker-level system metrics and OMP diagnostics working.
- **GPU:** Notebook exists and runs on NC4as_T4. Not yet benchmarked systematically.
- **Data generation:** Datasets exist in Unity Catalog for tiny/small/medium. 30M and 100M need generating.
- **Infrastructure:** All jobs defined in `databricks.yml` for dev target. Deploy via `databricks bundle deploy`.

### Key findings so far
- **OMP_NUM_THREADS=1** is silently set by Databricks on Spark executors → 3.4× speedup when fixed (see [LEARNINGS.md L1](LEARNINGS.md))
- **Plasma tuning** has no impact at 10M scale — default Ray object store is sufficient
- **Super-linear scaling** observed going from 2→4 workers (6.4× speedup from 2× workers), likely cache effects

## What's Next

Priority order:

1. **Generate 30M and 100M datasets** — need `medium_large` and `large` presets in Unity Catalog
2. **Run GPU baselines** — 1M and 10M on NC4as_T4 to get first GPU numbers for comparison
3. **Run 30M experiments** — Ray 4W and 8W on D16s to see how scaling holds at larger data
4. **Run single-node 30M on E32s** — find the single-node ceiling
5. **Run 100M experiments** — Ray 8W on D16s and E16s, test whether Plasma tuning matters at this scale
6. **Multi-GPU via Ray** — 10M and 30M on NC16as_T4 with 4 GPUs via Ray DataParallelTrainer
7. **Cost analysis** — compute DBU-hours for each completed experiment, build cost comparison table

## Open Questions

1. **Does Plasma tuning matter at 100M+ rows?** At 10M it made no difference. Hypothesis: yes, because the dataset will exceed worker memory.
2. **What's the GPU memory limit for XGBoost DMatrix?** T4 has 16GB — how many rows × features fit before OOM?
3. **Does multi-GPU Ray scaling work like multi-CPU?** Same `DataParallelTrainer` pattern, but GPU memory and PCIe bandwidth are the constraints instead of CPU/RAM.
4. **What is the optimal `nthread` setting?** Currently vCPUs−1. Does vCPUs−2 help by leaving headroom for OS + Spark?
5. **Does the OMP_NUM_THREADS issue affect GPU training?** GPU training uses CUDA cores, not CPU OpenMP threads — but the data loading pipeline is still CPU-bound.

## Key Decisions Made

| Decision | Rationale | Reference |
|----------|-----------|-----------|
| OMP fix via 3 layers (Spark conf + Ray runtime_env + worker-level) | Spark conf alone is most important, but all 3 for defence in depth | [LEARNINGS.md L1](LEARNINGS.md) |
| Deprioritised Plasma tuning | No impact at 10M. Revisit at 100M+ | [LEARNINGS.md L3](LEARNINGS.md) |
| Use `src/__init__.py` docstring not comment | Databricks misidentifies comment-starting files as notebooks | [LEARNINGS.md L13](LEARNINGS.md) |
| Per-worker metrics via Ray actors | MLflow system metrics only captures driver node | [LEARNINGS.md L6](LEARNINGS.md) |
| Deploy only via `databricks bundle deploy` | Avoids workspace code drift from direct API writes | [LEARNINGS.md L15](LEARNINGS.md) |

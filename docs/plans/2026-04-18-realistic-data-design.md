# Realistic Synthetic Data — Design Document

**Date:** 2026-04-18
**Status:** Approved
**Branch:** `agent/andy/realistic-data`

## Problem

Current synthetic datasets (`imbalanced_*`) are trivially linearly separable.
Every benchmark run achieves AUC=1.0, making accuracy metrics meaningless.
The data uses `randn(seed) + label * weight` — a simple mean shift that any
linear classifier can separate perfectly.

## Goals

1. Produce datasets that yield realistic AUC-ROC (~0.80–0.85) with XGBoost at depth=8, 200 rounds
2. Feature distributions and relationships that mimic real-world tabular data
3. Same row counts and feature counts as existing presets for direct timing comparison
4. Fully deterministic and reproducible via seed
5. Scalable to 100M+ rows using Spark-native generation (no pandas)
6. Old `imbalanced_*` tables remain untouched for reproducibility

## Track 1: Realistic Synthetic Generator (primary)

### Naming

- Tables: `realistic_10k`, `realistic_1m`, `realistic_10m`, `realistic_30m`, `realistic_100m`
- Notebook: `notebooks/generate_realistic_data.ipynb`
- Config presets: added to `src/config.py` with `realistic_` prefix

### Feature Generation

**1. Latent factor structure**

5–8 latent factors drawn from Normal(0,1) per row. Each latent factor drives
a cluster of 5–15 observed features:

    observed_i = loading_i * latent_k + (1 - loading_i) * noise_i

Loading strengths vary (0.3–0.8) to create realistic multicollinearity.

**2. Non-linear label generation**

Label probability from a non-linear combination of latent factors:

    logit = β₁·z₁ + β₂·z₁² + β₃·z₁·z₃ + β₄·max(z₄, 0) + β₅·sin(z₅) + ε
    p = sigmoid(logit + bias)
    label = Bernoulli(p)

- Interaction terms (z₁·z₃) require tree splits to capture
- Quadratic and ReLU terms add non-linearity
- ε ~ Normal(0, noise_scale) controls difficulty
- bias calibrated to achieve target minority_ratio (5%)

**3. Noise features**

~60% of numeric features are pure noise (uncorrelated with latent factors or label).

**4. Realistic distributions**

Transform observed features to non-Gaussian shapes:
- Log-normal (right-skewed, like transaction amounts)
- Zero-inflated (many zeros + continuous tail)
- Bimodal (mixture of two Gaussians)
- Heavy-tailed (Student-t)

Applied as element-wise transforms on the Gaussian base.

**5. Categorical features**

- ~20% informative (distribution depends on latent factor + label)
- Rest are noise (uniform random)
- Mixed cardinalities: binary (2), low (5), medium (20), high (100), very high (500)
- Some categorical-numeric interactions influence label probability

**6. Feature counts match existing presets**

| Preset | Rows | Num Features | Cat Features | Total |
|--------|------|-------------|-------------|-------|
| tiny | 10K | 15 | 5 | 20 |
| small | 1M | 80 | 20 | 100 |
| medium | 10M | 200 | 50 | 250 |
| medium_large | 30M | 200 | 50 | 250 |
| large | 100M | 400 | 100 | 500 |

### Implementation

- All generation in Spark (Column expressions + randn/rand)
- Latent factors as intermediate columns, dropped before final write
- Batched select() to avoid massive logical plans (same pattern as current generator)
- `noise_scale` widget parameter (default 2.0) for difficulty tuning
- Deterministic via seed parameter

### Calibration

Target: AUC-ROC ~0.82 with XGBoost depth=8, 200 rounds on 10K dataset.
Method: Generate 10K, train locally, adjust noise_scale until AUC lands in 0.78–0.85 range.

## Track 2: CTGAN Experimental (separate)

### Naming

- Tables: `ctgan_fraud_10k`, `ctgan_fraud_1m`, `ctgan_fraud_10m`, `ctgan_fraud_30m`
- Notebook: `notebooks/experimental/generate_ctgan_data.ipynb`

### Approach

1. Train CTGAN (from SDV library) on `kaggle_fraud_detection` table (500K rows, 8 features)
2. Sample synthetic rows at target sizes
3. Add synthetic noise features to reach target feature counts
4. Write to Delta tables

### Limitations

- CTGAN training is slow and may not scale beyond 30M rows practically
- Only 8 real features — bulk of columns will be synthetic noise anyway
- Generated data quality varies with CTGAN hyperparameters
- Adds sdv/ctgan dependency to the project

### Naming convention

Tables prefixed with `ctgan_fraud_` to clearly mark as experimental and
distinguish from the primary `realistic_` tables.

## Files to Create/Modify

1. `src/config.py` — add realistic presets
2. `notebooks/generate_realistic_data.ipynb` — main generator
3. `notebooks/experimental/generate_ctgan_data.ipynb` — CTGAN experimental
4. `docs/plans/2026-04-18-realistic-data-design.md` — this document

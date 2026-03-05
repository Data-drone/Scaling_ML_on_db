# Scaling Gradient Boosting Project

# Project-wide Guidelines

## General Do's
- Use descriptive variable names
- build in ipynbs
- use `%pip install -U mlflow` as a base and add library dependencies as required followed by `%restart_python`

## General Don'ts
- Don't build and deploy python scripts
- Don't use `dbutils.library.restartPython()`

## General Environment
- **Primary platform**: Databricks
- **Portability**: Should work in other environments aside from Unity Catalog integrations

# Project Sections

## Data Generation Notebook
**Applies to: ** `notebooks/generate_imbalanced_data.ipynb`
**Purpose:** Generate Synthetic Datasets for training models with

**Key Steps:**
1. have configuration for number of columns and column types
2. generate dataset in a distributed fashion
3. persist dataset to delta table

**Do's:**
- save output to delta table

## Single Node CPU Training for XGB
**Applies to: ** `notebooks/train_xgb_single.ipynb`
**Purpose:** Single Node xgboost training on cpu 

**Key Steps:**
1. Load Dataset with spark and convert to pandas
2. Format dataset for use with python xgboost
3. Train model instrumented with mlflow
4. log trained model to unity catalog

**Do's:**
- Set os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

**Don'ts:**
- use distributed xgboost

## Single Node GPU Training for XGB
**Applies to: ** `notebooks/train_xgb_gpu.ipynb`
**Purpose:** Single Node xgboost training on gpu

**Key Steps:**
1. Load Dataset with spark and convert to pandas
2. Format dataset for use with python xgboost
3. Train model instrumented with mlflow
4. log trained model to unity catalog

**Do's:**
- `%pip install pynvml` for gpu logging
- Set os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

**Don'ts:**
- use distributed xgboost

## Multi Node training with Ray
**Applies to: ** `notebooks/train_xgb_ray.ipynb`
**Purpose:** Multi Node xgboost training on Ray cpu

**Key Steps:**
1. Load Dataset with spark
2. Transform dataset to `ray.data` format
3. Train model instrumented with mlflow on ray
4. Save model to unity catalog

**Do's:**
- Use Ray with python xgb running on the works
- Load data using `ray.data.read_databricks_tables`
- Use `ray.data` apis to manipulate data before loading to xgboost

**Don'ts:**
- Use pandas for data manipulation

**Notes**
- `/dev/shm` is restricted to 50% of node ram on Databricks classic compute

# Deployment
- See: `docs/DEPLOYMENT.md`
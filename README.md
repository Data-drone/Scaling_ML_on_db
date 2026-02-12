# Scaling XGB

Large-scale XGBoost training experiments using Databricks and PySpark.

## Project Structure

```
.
├── databricks.yml          # Databricks Asset Bundle configuration
├── scripts/
│   └── dev.py              # Development CLI (validate, deploy, run, etc.)
├── src/
│   ├── __init__.py
│   ├── config.py           # Parameter parsing and validation
│   └── main.py             # Entry point for notebooks
├── notebooks/
│   └── generate_imbalanced_data.ipynb
├── tests/
│   └── unit/
│       ├── test_config.py
│       └── test_main.py
├── pyproject.toml          # Python project config
└── README.md
```

## Quick Start

```bash
# 1. Install dev dependencies
pip install -e ".[dev]"

# 2. Run local tests
python scripts/dev.py unit

# 3. Check prerequisites and auth
python scripts/dev.py setup

# 4. Validate bundle config
python scripts/dev.py validate

# 5. Deploy to Databricks
python scripts/dev.py deploy

# 6. Run a smoke test (fast, cheap)
python scripts/dev.py smoke

# 7. Run full job
python scripts/dev.py run
```

## Authentication (Zero Hardcoded Secrets)

This project uses **Databricks unified authentication**. No tokens or secrets are committed to the repository.

### How It Works

1. **Databricks CLI profiles** are stored in `~/.databrickscfg`
2. The VS Code/Cursor Databricks extension uses OAuth and creates profiles automatically
3. `scripts/dev.py` reads auth from your local profile—never from repo files

### Managing Profiles

```bash
# List available profiles
databricks auth profiles

# See current auth details
databricks auth describe

# Login to a workspace (creates/updates profile)
databricks auth login --host https://your-workspace.cloud.databricks.com

# Login with a specific profile name
databricks auth login --host https://your-workspace.cloud.databricks.com --profile my-profile
```

### Using Profiles with dev.py

```bash
# Option 1: Explicit --profile flag
python scripts/dev.py deploy --profile my-profile
python scripts/dev.py run --profile my-profile

# Option 2: Environment variable
export DATABRICKS_CONFIG_PROFILE=my-profile
python scripts/dev.py deploy
python scripts/dev.py run

# Option 3: Default profile (no flag needed if DEFAULT profile is set)
python scripts/dev.py deploy
```

### VS Code Extension Note

The Databricks VS Code extension OAuth login works for the extension UI, but **bundle CLI commands** still rely on the Databricks CLI unified auth visible in your terminal session. After logging in via the extension, verify with:

```bash
databricks auth describe
```

## Development Commands

All commands use `scripts/dev.py`:

| Command | Description |
|---------|-------------|
| `setup` | Check CLI installed, auth configured |
| `unit` | Run local pytest suite |
| `validate` | Validate bundle config (`databricks bundle validate`) |
| `deploy` | Deploy bundle to workspace |
| `run` | Run the job (full mode) |
| `smoke` | Run the job (smoke mode - tiny data, fast) |
| `destroy` | Tear down deployed resources |

### Common Options

```bash
# Use a specific profile
python scripts/dev.py <command> --profile <PROFILE_NAME>

# Target a different environment
python scripts/dev.py <command> --target prod
```

### Running with Parameters

```bash
# Smoke test (tiny dataset, fast)
python scripts/dev.py smoke

# Full run with size preset
python scripts/dev.py run --json-params '{"size_preset": "medium"}'

# Custom parameters
python scripts/dev.py run --json-params '{"total_rows": 5000000, "n_features": 200}'
```

### Available Size Presets

| Preset | Rows | Features | Use Case |
|--------|------|----------|----------|
| `tiny` | 10K | 20 | Smoke tests |
| `small` | 1M | 100 | Development |
| `medium` | 10M | 250 | Testing |
| `large` | 64M | 500 | Production |
| `xlarge` | 200M | 500 | Scale testing |

## Machine-Readable Output

`scripts/dev.py run` outputs markers for automation:

```
RUN_ID=123456789
RUN_URL=https://your-workspace.cloud.databricks.com/jobs/...
RESULT_JSON={"status":"ok","row_count":10000,...}
FINAL_STATE=SUCCEEDED
```

## Testing Strategy

### Local Unit Tests

Core logic in `src/` is pure Python with no Spark dependencies:

```bash
python scripts/dev.py unit
python scripts/dev.py unit --coverage
```

### Remote Smoke Tests

Smoke mode runs on Databricks with tiny synthetic data:

```bash
python scripts/dev.py smoke
```

This validates the end-to-end pipeline cheaply before running expensive full jobs.

## Configuration

### databricks.yml

Update these TODOs for your workspace:

1. **Workspace URL** (or rely on default profile)
2. **Unity Catalog / Schema names**
3. **Cluster settings** (`spark_version`, `node_type_id`, `num_workers`)
4. **Optional**: Cluster policy ID, spot instances, notifications

### Notebook Parameters

The notebook accepts these parameters (via widgets or job params):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `env` | Environment name | `dev` |
| `run_mode` | `full` or `smoke` | `full` |
| `json_params` | JSON string with overrides | `{}` |

## Iteration Loop

Typical development workflow:

```bash
# 1. Make changes to src/ or notebook

# 2. Run local tests
python scripts/dev.py unit

# 3. Validate config
python scripts/dev.py validate

# 4. Deploy changes
python scripts/dev.py deploy

# 5. Quick smoke test
python scripts/dev.py smoke

# 6. Check output, iterate...

# 7. Full run when ready
python scripts/dev.py run --json-params '{"size_preset": "medium"}'
```

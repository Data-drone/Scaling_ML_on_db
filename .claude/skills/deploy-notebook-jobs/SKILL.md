---
name: deploy-notebook-jobs
description: Deploys a Databricks notebook as a job via DAB (Databricks Asset Bundles). Generates the correct databricks.yml job definition, picks cluster config, deploys, runs the job, and diagnoses failures using known error patterns. Use when asked to deploy a notebook, create a job, run a training job, or troubleshoot a failed Databricks run.
---

# Deploy Notebook as Databricks Job (DAB)

End-to-end skill for deploying a notebook as a Databricks job via Asset Bundles. Handles job definition, cluster selection, deployment, execution, and failure diagnosis.

## Checklist

Copy this and track progress:

```
- [ ] Step 1: CONFIGURE — Generate job definition in databricks.yml
- [ ] Step 2: DEPLOY — Push bundle to workspace
- [ ] Step 3: RUN — Execute the job
- [ ] Step 4: DIAGNOSE — Handle failures (if any)
```

---

## Step 1: CONFIGURE -- Generate Job Definition

Add the job definition to `databricks.yml` under `resources.jobs`. Use the template below, filling in values from the cluster config reference.

### Job Definition Template

```yaml
resources:
  jobs:
    <job_key>:
      name: "[${var.env}] <Human-Readable Job Name>"
      description: "<What this job does>"

      parameters:
        - name: data_size
          default: small
        - name: node_type
          default: <NODE_TYPE_TAG>
        - name: run_mode
          default: full
        - name: table_name
          default: ""

      tasks:
        - task_key: <task_key>
          notebook_task:
            notebook_path: ./notebooks/<notebook_name>.ipynb
            base_parameters:
              data_size: "{{job.parameters.data_size}}"
              node_type: "{{job.parameters.node_type}}"
              run_mode: "{{job.parameters.run_mode}}"
              table_name: "{{job.parameters.table_name}}"
              catalog: ${var.catalog}
              schema: ${var.schema}
            source: WORKSPACE

          new_cluster:
            spark_version: "<RUNTIME>"
            node_type_id: "<AZURE_VM_SIZE>"
            num_workers: <NUM_WORKERS>
            data_security_mode: SINGLE_USER
            spark_conf:
              <SPARK_CONF_ENTRIES>
            custom_tags:
              ResourceClass: <SingleNode|MultiNode>
            azure_attributes:
              availability: SPOT_WITH_FALLBACK_AZURE
              spot_bid_max_price: -1

          libraries:
            - pypi:
                package: psutil

      tags:
        project: scaling_xgb
        environment: ${var.env}
        node_type: <NODE_TYPE_TAG>
```

### Quick Cluster Config Reference

Pick the track that matches your workload:

| Track | Runtime | VM Size | Workers | Key spark_conf |
|-------|---------|---------|---------|----------------|
| Single-node CPU | `17.3.x-cpu-ml-scala2.13` | `Standard_D16s_v5` | 0 | `singleNode` profile, `local[*, 4]` |
| Single-node CPU (high-mem) | `17.3.x-cpu-ml-scala2.13` | `Standard_E16s_v5` or `E32s_v5` | 0 | `singleNode` profile, `local[*, 4]` |
| Single-node GPU | `17.3.x-gpu-ml-scala2.13` | `Standard_NC4as_T4_v3` | 0 | `singleNode` profile, `local[*, 4]` |
| Single-node GPU (multi-GPU) | `17.3.x-gpu-ml-scala2.13` | `Standard_NC16as_T4_v3` | 0 | `singleNode` profile, `local[*, 4]` |
| Ray distributed (D8) | `17.3.x-cpu-ml-scala2.13` | `Standard_D8s_v5` | 2/4/8 | `OMP_NUM_THREADS`, `optimizeWrite` |
| Ray distributed (D16) | `17.3.x-cpu-ml-scala2.13` | `Standard_D16s_v5` | 2/4/8 | `OMP_NUM_THREADS`, `optimizeWrite` |
| Data generation | `17.3.x-scala2.13` | `Standard_D4s_v3` | 4 | `shuffle.partitions`, `optimizeWrite` |

For full YAML snippets and VM details, see [cluster-configs.md](cluster-configs.md).

### Important Configuration Rules

1. **Runtime selection** (L10): Use `-cpu-ml-` for CPU training, `-gpu-ml-` for GPU, plain `-scala2.13` for data generation only. The `-ml-` suffix is required for XGBoost, Ray, and MLflow.

2. **Unity Catalog**: Always set `data_security_mode: SINGLE_USER`. Required for accessing catalog tables.

3. **Single-node profile**: All `num_workers: 0` jobs MUST include:
   ```yaml
   spark_conf:
     spark.databricks.cluster.profile: singleNode
     spark.master: "local[*, 4]"
   custom_tags:
     ResourceClass: SingleNode
   ```

4. **OMP_NUM_THREADS** (L1): All Ray distributed jobs MUST set `spark.executorEnv.OMP_NUM_THREADS` in spark_conf. Without this, XGBoost silently uses 1 thread per worker (3.4x slower). Set to `vCPUs - 1`:
   - D8s_v5 (8 vCPUs): `"7"`
   - D16s_v5 (16 vCPUs): `"15"`
   - E16s_v5 (16 vCPUs): `"15"`

5. **__init__.py** (L13): If `src/__init__.py` starts with a `#` comment, Databricks misidentifies it as a notebook. Use a triple-quoted docstring instead.

6. **Source of truth** (L15): Always deploy via `databricks bundle deploy`. Never edit workspace files directly.

### Ray Distributed Extra Parameters

Ray jobs need additional `base_parameters`:

```yaml
base_parameters:
  num_workers: "<NUM_WORKERS>"
  cpus_per_worker: "<CPUS_PER_WORKER>"
  warehouse_id: "148ccb90800933a1"
  # Plus the standard params: data_size, node_type, run_mode, table_name, catalog, schema
```

Where `cpus_per_worker`:
- D8s_v5: `"6"` (8 vCPUs minus 2 for overhead)
- D16s_v5: `"14"` (16 vCPUs minus 2 for overhead)
- E16s_v5: `"14"` (16 vCPUs minus 2 for overhead)

---

## Step 2: DEPLOY -- Push Bundle to Workspace

```bash
# Ensure you are in the repo root
cd /workspace/group/scaling_xgb_work

# Validate the bundle config first
databricks bundle validate -t dev

# Deploy to dev (default target)
databricks bundle deploy -t dev

# For verbose output on failures:
databricks bundle deploy -t dev --debug --log-file /tmp/bundle.log
```

**Expected output:** Lists of created/updated resources. If any job definition has YAML errors, the validate step will catch them.

**Deploy to prod:**
```bash
databricks bundle deploy -t prod
```

---

## Step 3: RUN -- Execute the Job

```bash
# Run a specific job by its key name in databricks.yml
databricks bundle run -t dev <job_key>

# Example:
databricks bundle run -t dev perf_single_1m_d16

# Override parameters at run time:
databricks bundle run -t dev train_xgb_single_e16 \
  --params data_size=medium,table_name=imbalanced_10m
```

The command outputs a run URL. Copy the run ID for monitoring:

```bash
# Check run state
databricks jobs get-run <RUN_ID> --output json | jq '.state'

# Watch until completion (poll every 30s)
while true; do
  STATE=$(databricks jobs get-run <RUN_ID> --output json | jq -r '.state.life_cycle_state')
  echo "$(date +%H:%M:%S) $STATE"
  [[ "$STATE" == "TERMINATED" || "$STATE" == "INTERNAL_ERROR" ]] && break
  sleep 30
done
```

---

## Step 4: DIAGNOSE -- Handle Failures

If the run fails, follow the error diagnosis procedure. The #1 problem: Databricks wraps all notebook failures as `RUN_EXECUTION_ERROR` with the useless message `"Workload failed, see run output for details"`. The actual error requires a specific API call pattern.

For the full 4-step error retrieval procedure and the RE1-RE5 known error patterns, see [error-diagnosis.md](error-diagnosis.md).

### Quick Diagnosis Flow

```
Run failed
    |
    v
Get parent run ID from bundle output or UI
    |
    v
Get task-level run IDs (parent run has no error details)
    |
    v
Get error + error_trace from task run
    |
    +-- error_trace has content? --> Match against RE1-RE5 patterns
    |
    +-- error_trace is empty? --> Check cluster events API for infra failures
```

### Quick RE Pattern Match

| Pattern | Signature | Quick Fix |
|---------|-----------|-----------|
| RE1 | `Query '...' execution failed` | SQL warehouse down or table missing. Check warehouse + table. |
| RE2 | `Model passed for registration did not contain any signature` | Add `signature=infer_signature(...)` to `log_model()` |
| RE3 | `Ray has not been started yet` | Add `ray.is_initialized()` check after `setup_ray_cluster()` |
| RE4 | `UPDATE_FAILED` / `ImportError` / `ModuleNotFoundError` in serving | Fix pyfunc deps and imports |
| RE5 | Empty error_trace | Infrastructure failure -- check cluster events API |

---

## References

- **Cluster config details**: [cluster-configs.md](cluster-configs.md)
- **Error diagnosis details**: [error-diagnosis.md](error-diagnosis.md)
- **Project learnings**: `docs/LEARNINGS.md` in the repo
- **Deployment guide**: `docs/DEPLOYMENT.md` in the repo
- **Crash retriever code**: `src/crash_retriever.py` in the repo

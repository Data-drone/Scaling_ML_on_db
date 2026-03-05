# Deployment instructions

- Deploy:
  databricks bundle deploy -t <target>

## Diagnose `databricks bundle deploy` failures (e.g., RUN_EXECUTION_ERROR)

# Note: RUN_EXECUTION_ERROR is usually a wrapper; the real cause is in the failed TASK output / driver logs.

# 1) Re-run with debug logging (capture IDs)
databricks bundle deploy -t <target> --debug --log-file /tmp/bundle.log

# 2) Extract identifiers from the log (job/run/pipeline)
grep -E "RUN_EXECUTION_ERROR|run_id|job_id|pipeline_id" /tmp/bundle.log | tail -n 50

# 3) If this is a Jobs run (run_id)
# SINGLE-TASK job:
databricks jobs get-run-output <RUN_ID> --output json

# MULTI-TASK job (get-run-output on parent run is NOT supported):
databricks jobs get-run <PARENT_RUN_ID> --output json > /tmp/run.json
jq -r '.tasks[] | "\(.task_key)\trun_id=\(.run_id)\tstate=\(.state.life_cycle_state)/\(.state.result_state // "NA")"' /tmp/run.json
databricks jobs get-run-output <TASK_RUN_ID> --output json

# 4) If output is sparse (spark-submit / some task types), use the Jobs UI:
# Open the failed task and read driver stdout/stderr + event logs.

# 5) If this is a DLT pipeline (pipeline_id), inspect pipeline update/event logs instead of jobs runs.

## File Hygiene
- Do not commit generated artifacts (`__pycache__`, `.pyc`, local env files).
- Keep `.gitignore` updated for local/dev-only files.
- Preserve existing user changes in unrelated files.

## Safety
- Never expose or commit secrets/tokens.
- Redact sensitive values in logs, examples, and docs.
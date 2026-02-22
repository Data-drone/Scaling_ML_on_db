"""
CrashRetriever: Programmatic crash diagnosis from Databricks Jobs API.

Retrieves and formats failure information from Databricks job runs:
  - Job run state and error messages
  - Cluster events (spot preemption, OOM, startup failures)
  - Driver log excerpts (last N lines of stderr/stdout)
  - Structured crash report for debugging

Usage:
    from src.crash_retriever import CrashRetriever

    cr = CrashRetriever(host=WORKSPACE_URL, token=TOKEN)
    report = cr.diagnose_run(run_id=12345)
    print(report.summary())

    # Or diagnose the most recent failure for a job
    report = cr.diagnose_latest_failure(job_id=67890)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    requests = None  # Will fail at runtime with clear message


@dataclass
class CrashReport:
    """Structured crash diagnosis report."""
    run_id: int
    job_id: Optional[int] = None
    job_name: Optional[str] = None
    cluster_id: Optional[str] = None

    # State
    life_cycle_state: str = ""
    result_state: str = ""
    state_message: str = ""

    # Timing
    start_time_ms: int = 0
    end_time_ms: int = 0
    setup_duration_ms: int = 0

    # Error details
    error_code: str = ""
    error_message: str = ""

    # Cluster events (e.g., spot preemption, OOM)
    cluster_events: list = field(default_factory=list)

    # Driver log excerpts
    driver_stdout_tail: str = ""
    driver_stderr_tail: str = ""

    # Task-level failures
    task_failures: list = field(default_factory=list)

    # Raw API responses for further analysis
    raw_run: dict = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        if self.end_time_ms and self.start_time_ms:
            return (self.end_time_ms - self.start_time_ms) / 1000
        return 0

    @property
    def is_spot_preemption(self) -> bool:
        """Check if failure was due to Azure spot instance preemption."""
        for event in self.cluster_events:
            event_type = event.get("type", "")
            details = event.get("details", {})
            reason = details.get("reason", {})
            if event_type == "TERMINATING" and "SPOT" in str(reason).upper():
                return True
        return "SPOT" in self.state_message.upper() or "preempt" in self.state_message.lower()

    @property
    def is_oom(self) -> bool:
        """Check if failure was due to out-of-memory."""
        markers = ["OutOfMemoryError", "oom-killer", "Cannot allocate memory", "MemoryError"]
        combined = f"{self.state_message} {self.error_message} {self.driver_stderr_tail}"
        return any(m.lower() in combined.lower() for m in markers)

    @property
    def is_config_error(self) -> bool:
        """Check if failure was due to misconfiguration."""
        markers = [
            "INVALID_PARAMETER_VALUE", "InvalidParameterValue",
            "ClusterNotFoundException", "RESOURCE_DOES_NOT_EXIST",
            "spark_version", "node_type_id",
        ]
        combined = f"{self.state_message} {self.error_message} {self.error_code}"
        return any(m.lower() in combined.lower() for m in markers)

    @property
    def crash_category(self) -> str:
        """Classify the crash into a category."""
        if self.result_state in ("SUCCESS", "SUCCEEDED"):
            return "SUCCESS"
        if self.is_spot_preemption:
            return "SPOT_PREEMPTION"
        if self.is_oom:
            return "OUT_OF_MEMORY"
        if self.is_config_error:
            return "CONFIG_ERROR"
        if self.life_cycle_state == "INTERNAL_ERROR":
            return "INTERNAL_ERROR"
        if self.result_state == "CANCELED":
            return "CANCELED"
        if self.result_state == "TIMEDOUT":
            return "TIMEOUT"
        return "UNKNOWN_FAILURE"

    def summary(self) -> str:
        """Human-readable crash summary."""
        lines = [
            "=" * 70,
            f"CRASH REPORT — Run {self.run_id}",
            "=" * 70,
            f"Job:           {self.job_name or 'N/A'} (ID: {self.job_id or 'N/A'})",
            f"Cluster:       {self.cluster_id or 'N/A'}",
            f"State:         {self.life_cycle_state} / {self.result_state}",
            f"Category:      {self.crash_category}",
            f"Duration:      {self.duration_sec:.1f}s (setup: {self.setup_duration_ms / 1000:.1f}s)",
            f"Error Code:    {self.error_code or 'N/A'}",
            "",
            "STATE MESSAGE:",
            self.state_message or "(none)",
            "",
        ]

        if self.error_message:
            lines.extend(["ERROR MESSAGE:", self.error_message, ""])

        if self.task_failures:
            lines.append("TASK FAILURES:")
            for task in self.task_failures:
                lines.append(f"  - {task.get('task_key', 'unknown')}: {task.get('state', {}).get('result_state', 'N/A')}")
                if task.get("state", {}).get("state_message"):
                    lines.append(f"    {task['state']['state_message'][:200]}")
            lines.append("")

        if self.cluster_events:
            lines.append(f"CLUSTER EVENTS ({len(self.cluster_events)} relevant):")
            for event in self.cluster_events[-5:]:  # Last 5 events
                ts = event.get("timestamp", 0)
                t = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if ts else "??:??:??"
                etype = event.get("type", "UNKNOWN")
                details = json.dumps(event.get("details", {}), indent=None)[:150]
                lines.append(f"  [{t}] {etype}: {details}")
            lines.append("")

        if self.driver_stderr_tail:
            lines.extend([
                "DRIVER STDERR (last lines):",
                self.driver_stderr_tail[-2000:],
                "",
            ])

        # Recommendations
        lines.append("RECOMMENDATIONS:")
        cat = self.crash_category
        if cat == "SPOT_PREEMPTION":
            lines.append("  - Switch to ON_DEMAND or SPOT_WITH_FALLBACK_AZURE")
            lines.append("  - Use smaller, more available VM sizes (D8 instead of E32)")
        elif cat == "OUT_OF_MEMORY":
            lines.append("  - Use a larger VM (more RAM) or add more workers to distribute data")
            lines.append("  - Reduce data_size preset (e.g., medium instead of large)")
            lines.append("  - For Ray: increase obj_store_mem_gb or enable allow_slow_storage")
        elif cat == "CONFIG_ERROR":
            lines.append("  - Run src/validate_env.py to check environment")
            lines.append("  - Verify databricks.yml cluster config (runtime, node_type, spark_conf)")
        elif cat == "INTERNAL_ERROR":
            lines.append("  - This is usually a Databricks platform issue — retry the job")
            lines.append("  - If persistent, check Databricks status page")
        else:
            lines.append("  - Check driver stderr logs for Python tracebacks")
            lines.append("  - Run src/validate_env.py to check environment")

        lines.append("=" * 70)
        return "\n".join(lines)


class CrashRetriever:
    """
    Retrieve and diagnose failures from Databricks Jobs API.

    Args:
        host: Databricks workspace URL (e.g., "https://adb-xxx.azuredatabricks.net")
        token: Databricks personal access token
    """

    def __init__(self, host: str, token: str):
        if requests is None:
            raise ImportError("'requests' library required. pip install requests")

        self.host = host.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def _api(self, method: str, path: str, **kwargs) -> dict:
        """Make authenticated API call."""
        url = urljoin(self.host + "/", path.lstrip("/"))
        resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _get_run(self, run_id: int) -> dict:
        """Get run details via Jobs API 2.1."""
        return self._api("GET", "/api/2.1/jobs/runs/get", params={"run_id": run_id})

    def _get_run_output(self, run_id: int) -> dict:
        """Get run output (notebook result, error trace)."""
        try:
            return self._api("GET", "/api/2.1/jobs/runs/get-output", params={"run_id": run_id})
        except Exception:
            return {}

    def _get_cluster_events(self, cluster_id: str, limit: int = 50) -> list:
        """Get cluster events (termination reasons, spot preemption, etc.)."""
        try:
            data = self._api("POST", "/api/2.0/clusters/events", json={
                "cluster_id": cluster_id,
                "limit": limit,
                "event_types": [
                    "TERMINATING", "TERMINATED", "DRIVER_NOT_RESPONDING",
                    "NODES_LOST", "AUTOSCALING_STATS_REPORT",
                ],
            })
            return data.get("events", [])
        except Exception:
            return []

    def _get_driver_logs(self, cluster_id: str, max_bytes: int = 50_000) -> tuple:
        """
        Retrieve driver log excerpts from DBFS.

        Returns (stdout_tail, stderr_tail).
        Note: Logs may not be available if cluster is already terminated.
        """
        stdout = ""
        stderr = ""

        for log_type in ["stdout", "stderr"]:
            try:
                path = f"dbfs:/cluster-logs/{cluster_id}/driver/{log_type}"
                data = self._api("GET", "/api/2.0/dbfs/read", params={
                    "path": path,
                    "offset": 0,
                    "length": max_bytes,
                })
                import base64
                content = base64.b64decode(data.get("data", "")).decode("utf-8", errors="replace")
                if log_type == "stdout":
                    stdout = content
                else:
                    stderr = content
            except Exception:
                pass  # Logs often not available for terminated clusters

        return stdout, stderr

    def diagnose_run(self, run_id: int) -> CrashReport:
        """
        Diagnose a specific job run.

        Args:
            run_id: Databricks job run ID

        Returns:
            CrashReport with structured failure information
        """
        run = self._get_run(run_id)
        state = run.get("state", {})

        report = CrashReport(
            run_id=run_id,
            job_id=run.get("job_id"),
            life_cycle_state=state.get("life_cycle_state", ""),
            result_state=state.get("result_state", ""),
            state_message=state.get("state_message", ""),
            start_time_ms=run.get("start_time", 0),
            end_time_ms=run.get("end_time", 0),
            setup_duration_ms=run.get("setup_duration", 0),
            raw_run=run,
        )

        # Extract cluster ID from tasks
        tasks = run.get("tasks", [])
        for task in tasks:
            cluster_instance = task.get("cluster_instance", {})
            if cluster_instance.get("cluster_id"):
                report.cluster_id = cluster_instance["cluster_id"]
                break

        # Get job name
        try:
            job = self._api("GET", "/api/2.1/jobs/get", params={"job_id": report.job_id})
            report.job_name = job.get("settings", {}).get("name", "")
        except Exception:
            pass

        # Get run output (error trace)
        output = self._get_run_output(run_id)
        if output.get("error"):
            report.error_message = output["error"]
        if output.get("error_trace"):
            report.error_message = output["error_trace"]

        # Get cluster events
        if report.cluster_id:
            report.cluster_events = self._get_cluster_events(report.cluster_id)

        # Get driver logs
        if report.cluster_id:
            stdout, stderr = self._get_driver_logs(report.cluster_id)
            report.driver_stdout_tail = stdout[-3000:] if stdout else ""
            report.driver_stderr_tail = stderr[-3000:] if stderr else ""

        # Extract task-level failures
        for task in tasks:
            task_state = task.get("state", {})
            if task_state.get("result_state") not in (None, "", "SUCCESS"):
                report.task_failures.append({
                    "task_key": task.get("task_key", "unknown"),
                    "state": task_state,
                })

        return report

    def diagnose_latest_failure(self, job_id: int, look_back: int = 10) -> Optional[CrashReport]:
        """
        Find and diagnose the most recent failed run for a job.

        Args:
            job_id: Databricks job ID
            look_back: Number of recent runs to check

        Returns:
            CrashReport for the most recent failure, or None if no failures found
        """
        runs = self._api("GET", "/api/2.1/jobs/runs/list", params={
            "job_id": job_id,
            "limit": look_back,
            "expand_tasks": True,
        })

        for run in runs.get("runs", []):
            state = run.get("state", {})
            result = state.get("result_state", "")
            if result not in ("SUCCESS", "SUCCEEDED", ""):
                return self.diagnose_run(run["run_id"])

        return None

    def list_recent_failures(self, job_id: int, limit: int = 20) -> list:
        """
        List recent failed runs for a job (summary only, without full diagnosis).

        Returns list of dicts with run_id, result_state, state_message, duration.
        """
        runs = self._api("GET", "/api/2.1/jobs/runs/list", params={
            "job_id": job_id,
            "limit": limit,
        })

        failures = []
        for run in runs.get("runs", []):
            state = run.get("state", {})
            result = state.get("result_state", "")
            if result not in ("SUCCESS", "SUCCEEDED", ""):
                failures.append({
                    "run_id": run["run_id"],
                    "result_state": result,
                    "life_cycle_state": state.get("life_cycle_state", ""),
                    "state_message": state.get("state_message", "")[:200],
                    "start_time_ms": run.get("start_time", 0),
                    "end_time_ms": run.get("end_time", 0),
                    "duration_sec": (run.get("end_time", 0) - run.get("start_time", 0)) / 1000,
                })
        return failures

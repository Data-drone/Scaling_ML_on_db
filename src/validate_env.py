"""
Environment validation gate for Databricks scaling experiments.

Run this as the FIRST cell in any experiment notebook to catch misconfiguration
before burning cluster time. Validates:
  - Databricks runtime version
  - Cluster type (single-node vs multi-node)
  - Required Spark config (e.g., OMP_NUM_THREADS)
  - Library availability (XGBoost, Ray, MLflow, psutil)
  - Unity Catalog access
  - GPU availability (for GPU track)

Usage in notebook:
    from src.validate_env import validate_environment
    validate_environment(
        track="ray-scaling",
        expected_workers=4,
        require_gpu=False,
    )
"""

import os
import sys
import importlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info

    def __str__(self):
        icon = "PASS" if self.passed else ("WARN" if self.severity == "warning" else "FAIL")
        return f"[{icon}] {self.name}: {self.message}"


@dataclass
class EnvironmentReport:
    """Aggregate validation report."""
    checks: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed or c.severity == "warning" for c in self.checks)

    @property
    def errors(self) -> list:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def summary(self) -> str:
        lines = ["=" * 60, "Environment Validation Report", "=" * 60]
        for check in self.checks:
            lines.append(str(check))
        lines.append("-" * 60)
        if self.passed:
            lines.append(f"RESULT: PASSED ({len(self.errors)} errors, {len(self.warnings)} warnings)")
        else:
            lines.append(f"RESULT: FAILED ({len(self.errors)} errors, {len(self.warnings)} warnings)")
            lines.append("Fix the errors above before running the experiment.")
        lines.append("=" * 60)
        return "\n".join(lines)

    def add(self, name: str, passed: bool, message: str, severity: str = "error"):
        self.checks.append(ValidationResult(name, passed, message, severity))


def _check_dbr_version(report: EnvironmentReport, min_version: str = "17.3"):
    """Check Databricks Runtime version."""
    dbr = os.environ.get("DATABRICKS_RUNTIME_VERSION", "")
    if not dbr:
        report.add("DBR Version", False, "Not running on Databricks (DATABRICKS_RUNTIME_VERSION not set)", "warning")
        return

    # Parse major.minor from strings like "17.3.x-cpu-ml-scala2.13"
    parts = dbr.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
        min_parts = min_version.split(".")
        min_major, min_minor = int(min_parts[0]), int(min_parts[1])

        if (major, minor) >= (min_major, min_minor):
            report.add("DBR Version", True, f"{dbr} >= {min_version}")
        else:
            report.add("DBR Version", False, f"{dbr} < {min_version} — upgrade cluster runtime")
    except (ValueError, IndexError):
        report.add("DBR Version", False, f"Could not parse DBR version: '{dbr}'", "warning")


def _check_ml_runtime(report: EnvironmentReport, require_gpu: bool = False):
    """Check that ML Runtime is being used."""
    dbr = os.environ.get("DATABRICKS_RUNTIME_VERSION", "")
    if not dbr:
        report.add("ML Runtime", False, "Cannot verify — not on Databricks", "warning")
        return

    if require_gpu:
        if "gpu-ml" in dbr:
            report.add("ML Runtime", True, f"GPU ML Runtime detected: {dbr}")
        else:
            report.add("ML Runtime", False, f"GPU ML Runtime required but got: {dbr}. Use '17.3.x-gpu-ml-scala2.13'")
    else:
        if "ml" in dbr:
            report.add("ML Runtime", True, f"ML Runtime detected: {dbr}")
        else:
            report.add("ML Runtime", False, f"ML Runtime required but got: {dbr}. Use '17.3.x-cpu-ml-scala2.13'")


def _check_omp_config(report: EnvironmentReport, track: str):
    """Check OMP_NUM_THREADS configuration."""
    if track not in ("ray-scaling", "ray-plasma-tuning"):
        report.add("OMP Config", True, f"OMP check not required for track '{track}'", "info")
        return

    omp = os.environ.get("OMP_NUM_THREADS", "")
    if omp and omp != "1":
        report.add("OMP Config", True, f"OMP_NUM_THREADS={omp}")
    elif omp == "1":
        report.add(
            "OMP Config", False,
            "OMP_NUM_THREADS=1 — XGBoost will use only 1 core! "
            "Set spark.executorEnv.OMP_NUM_THREADS in cluster Spark config. "
            "See LEARNINGS.md L1."
        )
    else:
        report.add(
            "OMP Config", False,
            "OMP_NUM_THREADS not set. On Databricks executors, Spark sets it to 1 by default. "
            "Ensure spark.executorEnv.OMP_NUM_THREADS is set in the job cluster config.",
            "warning"
        )


def _check_libraries(report: EnvironmentReport, track: str, require_gpu: bool = False):
    """Check required Python libraries are available."""
    required = ["xgboost", "mlflow", "psutil"]

    if track in ("ray-scaling", "ray-plasma-tuning"):
        required.extend(["ray", "ray.train"])

    if require_gpu:
        required.append("cupy")  # Optional but useful for GPU debugging

    for lib in required:
        try:
            importlib.import_module(lib)
            report.add(f"Library: {lib}", True, "Available")
        except ImportError:
            severity = "error" if lib in ("xgboost", "mlflow", "ray") else "warning"
            report.add(f"Library: {lib}", False, f"Not installed — pip install {lib}", severity)


def _check_spark_cluster(report: EnvironmentReport, expected_workers: Optional[int] = None):
    """Check Spark cluster configuration."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            report.add("Spark Cluster", False, "No active SparkSession", "warning")
            return

        sc = spark.sparkContext
        # Get executor count (subtract 1 for driver)
        executors = sc._jsc.sc().getExecutorMemoryStatus().size() - 1
        report.add("Spark Cluster", True, f"{executors} executors detected")

        if expected_workers is not None and executors != expected_workers:
            report.add(
                "Worker Count", False,
                f"Expected {expected_workers} workers but found {executors}. "
                "Check cluster num_workers in databricks.yml.",
                "warning"
            )
    except Exception as e:
        report.add("Spark Cluster", False, f"Could not inspect cluster: {e}", "warning")


def _check_unity_catalog(report: EnvironmentReport, catalog: str = "brian_gen_ai", schema: str = "xgb_scaling"):
    """Check Unity Catalog access."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            report.add("Unity Catalog", False, "No SparkSession — cannot verify UC access", "warning")
            return

        # Try to list tables in the schema
        spark.sql(f"USE CATALOG {catalog}")
        spark.sql(f"USE SCHEMA {schema}")
        tables = spark.sql("SHOW TABLES").collect()
        table_names = [row.tableName for row in tables]
        report.add("Unity Catalog", True, f"Access OK — {len(table_names)} tables in {catalog}.{schema}")
    except Exception as e:
        report.add("Unity Catalog", False, f"Cannot access {catalog}.{schema}: {e}", "warning")


def _check_gpu(report: EnvironmentReport):
    """Check GPU availability."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split("\n")
            report.add("GPU", True, f"{len(gpus)} GPU(s): {'; '.join(gpus)}")
        else:
            report.add("GPU", False, "nvidia-smi failed — no GPU available")
    except FileNotFoundError:
        report.add("GPU", False, "nvidia-smi not found — no GPU runtime")
    except Exception as e:
        report.add("GPU", False, f"GPU check failed: {e}")


def validate_environment(
    track: str = "single-node-scaling",
    expected_workers: Optional[int] = None,
    require_gpu: bool = False,
    catalog: str = "brian_gen_ai",
    schema: str = "xgb_scaling",
    min_dbr_version: str = "17.3",
    raise_on_failure: bool = True,
) -> EnvironmentReport:
    """
    Run all environment validation checks.

    Args:
        track: Scaling track name (single-node-scaling, ray-scaling, ray-plasma-tuning, gpu-scaling)
        expected_workers: Expected number of Spark workers (None = don't check)
        require_gpu: Whether GPU is required
        catalog: Unity Catalog name
        schema: Unity Catalog schema
        min_dbr_version: Minimum Databricks Runtime version
        raise_on_failure: Whether to raise RuntimeError on validation failure

    Returns:
        EnvironmentReport with all check results

    Raises:
        RuntimeError: If raise_on_failure=True and any error-severity check fails
    """
    report = EnvironmentReport()

    _check_dbr_version(report, min_dbr_version)
    _check_ml_runtime(report, require_gpu)
    _check_omp_config(report, track)
    _check_libraries(report, track, require_gpu)
    _check_spark_cluster(report, expected_workers)
    _check_unity_catalog(report, catalog, schema)

    if require_gpu:
        _check_gpu(report)

    print(report.summary())

    if raise_on_failure and not report.passed:
        raise RuntimeError(
            f"Environment validation failed with {len(report.errors)} error(s). "
            "Fix the issues above before running the experiment."
        )

    return report

"""
Benchmark result collection and comparison utilities.

Standardises how experiment results are recorded, compared, and reported
across all scaling tracks. Integrates with MLflow for storage and retrieval.

Usage:
    from src.benchmark import BenchmarkResult, compare_results

    # Record a result
    result = BenchmarkResult(
        track="ray-scaling",
        data_size="medium",
        node_type="D16sv5",
        num_workers=4,
        train_time_sec=80.2,
        total_time_sec=128.5,
        auc_pr=1.0,
        cpu_avg_pct=71.2,
    )
    result.log_to_mlflow(run_id="abc123")

    # Compare results across experiments
    compare_results([result1, result2, result3])
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class BenchmarkResult:
    """Standardised benchmark result for any scaling track."""

    # Identity
    track: str                      # single-node-scaling, ray-scaling, etc.
    data_size: str                  # tiny, small, medium, etc.
    node_type: str                  # D16sv5, E16sv5, NC6sv3, etc.
    num_workers: int = 1            # 1 for single-node, N for distributed
    cpus_per_worker: int = 0        # 0 = all available
    gpus_per_worker: int = 0        # 0 for CPU tracks

    # Timing
    train_time_sec: float = 0.0     # XGBoost training wall time
    total_time_sec: float = 0.0     # Total job time (includes data load, setup)
    setup_time_sec: float = 0.0     # Cluster + env setup time
    data_load_time_sec: float = 0.0 # Data loading from UC

    # Model quality
    auc_pr: float = 0.0
    auc_roc: float = 0.0
    f1: float = 0.0

    # Resource utilisation
    cpu_avg_pct: float = 0.0        # Average CPU across workers
    cpu_per_worker: list = field(default_factory=list)  # Per-worker CPU %
    memory_peak_gb: float = 0.0     # Peak memory usage

    # Config
    omp_threads: int = 0            # OMP_NUM_THREADS setting (0 = not set)
    xgb_nthread: int = 0            # XGBoost nthread parameter
    obj_store_mem_gb: float = 0.0   # Ray object store (0 = default)
    heap_mem_gb: float = 0.0        # Ray heap memory (0 = default)

    # Metadata
    mlflow_run_id: str = ""
    databricks_run_id: str = ""
    timestamp: float = field(default_factory=time.time)
    notes: str = ""

    @property
    def speedup_vs(self) -> dict:
        """Placeholder for speedup calculations — populated by compare_results."""
        return {}

    @property
    def cost_efficiency(self) -> float:
        """
        Rough cost efficiency metric: lower is better.
        Approximation: train_time * num_workers * vcpus_per_hour_cost.
        """
        # Approximate Azure VM costs per hour (spot pricing)
        cost_per_vcpu_hour = {
            "D8sv5": 0.384 / 8,     # ~$0.048/vCPU/h
            "D16sv5": 0.768 / 16,   # ~$0.048/vCPU/h
            "E16sv5": 1.008 / 16,   # ~$0.063/vCPU/h
            "E32sv5": 2.016 / 32,   # ~$0.063/vCPU/h
        }
        rate = cost_per_vcpu_hour.get(self.node_type, 0.05)
        vcpus = self.cpus_per_worker or 8
        cost = (self.total_time_sec / 3600) * self.num_workers * vcpus * rate
        return round(cost, 4)

    def to_dict(self) -> dict:
        """Convert to dict for serialisation."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def log_to_mlflow(self, run_id: Optional[str] = None):
        """
        Log benchmark result to an MLflow run.

        Args:
            run_id: Existing MLflow run ID. If None, logs to active run.
        """
        try:
            import mlflow

            if run_id:
                client = mlflow.tracking.MlflowClient()
                client.log_param(run_id, "benchmark_track", self.track)
                client.log_param(run_id, "benchmark_data_size", self.data_size)
                client.log_param(run_id, "benchmark_node_type", self.node_type)
                client.log_param(run_id, "benchmark_num_workers", self.num_workers)
                client.log_metric(run_id, "benchmark_train_time_sec", self.train_time_sec)
                client.log_metric(run_id, "benchmark_total_time_sec", self.total_time_sec)
                client.log_metric(run_id, "benchmark_auc_pr", self.auc_pr)
                client.log_metric(run_id, "benchmark_cpu_avg_pct", self.cpu_avg_pct)
                client.log_metric(run_id, "benchmark_cost_efficiency", self.cost_efficiency)
            else:
                mlflow.log_params({
                    "benchmark_track": self.track,
                    "benchmark_data_size": self.data_size,
                    "benchmark_node_type": self.node_type,
                    "benchmark_num_workers": self.num_workers,
                })
                mlflow.log_metrics({
                    "benchmark_train_time_sec": self.train_time_sec,
                    "benchmark_total_time_sec": self.total_time_sec,
                    "benchmark_auc_pr": self.auc_pr,
                    "benchmark_cpu_avg_pct": self.cpu_avg_pct,
                    "benchmark_cost_efficiency": self.cost_efficiency,
                })
        except ImportError:
            print("Warning: mlflow not available — benchmark not logged")


def compare_results(results: list, sort_by: str = "train_time_sec") -> str:
    """
    Generate a comparison table across benchmark results.

    Args:
        results: List of BenchmarkResult objects
        sort_by: Field to sort by (default: train_time_sec)

    Returns:
        Formatted comparison table string
    """
    if not results:
        return "No results to compare."

    sorted_results = sorted(results, key=lambda r: getattr(r, sort_by, 0))
    baseline = sorted_results[-1]  # Slowest as baseline for speedup

    lines = [
        "=" * 100,
        "BENCHMARK COMPARISON",
        "=" * 100,
        f"{'Track':<20} {'Data':<10} {'Node':<12} {'Workers':>8} {'Train(s)':>10} {'Total(s)':>10} {'Speedup':>8} {'CPU%':>6} {'AUC-PR':>8} {'Cost':>8}",
        "-" * 100,
    ]

    for r in sorted_results:
        speedup = baseline.train_time_sec / r.train_time_sec if r.train_time_sec > 0 else 0
        lines.append(
            f"{r.track:<20} {r.data_size:<10} {r.node_type:<12} {r.num_workers:>8} "
            f"{r.train_time_sec:>10.1f} {r.total_time_sec:>10.1f} {speedup:>7.1f}x "
            f"{r.cpu_avg_pct:>5.0f}% {r.auc_pr:>8.4f} ${r.cost_efficiency:>7.4f}"
        )

    lines.extend([
        "-" * 100,
        f"Baseline (slowest): {baseline.track} / {baseline.data_size} / {baseline.node_type} / {baseline.num_workers}W",
        "=" * 100,
    ])

    return "\n".join(lines)


def load_results_from_mlflow(
    experiment_path: str = "/Users/brian.law@databricks.com/xgb_scaling_benchmark",
    filter_track: Optional[str] = None,
) -> list:
    """
    Load benchmark results from MLflow experiment.

    Args:
        experiment_path: MLflow experiment path
        filter_track: Optional track name to filter by

    Returns:
        List of BenchmarkResult objects
    """
    try:
        import mlflow

        experiment = mlflow.get_experiment_by_name(experiment_path)
        if experiment is None:
            print(f"Experiment not found: {experiment_path}")
            return []

        filter_str = ""
        if filter_track:
            filter_str = f"params.benchmark_track = '{filter_track}'"

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_str,
        )

        results = []
        for _, row in runs.iterrows():
            try:
                result = BenchmarkResult(
                    track=row.get("params.benchmark_track", "unknown"),
                    data_size=row.get("params.benchmark_data_size", "unknown"),
                    node_type=row.get("params.benchmark_node_type", "unknown"),
                    num_workers=int(row.get("params.benchmark_num_workers", 1)),
                    train_time_sec=float(row.get("metrics.benchmark_train_time_sec", 0)),
                    total_time_sec=float(row.get("metrics.benchmark_total_time_sec", 0)),
                    auc_pr=float(row.get("metrics.benchmark_auc_pr", 0)),
                    cpu_avg_pct=float(row.get("metrics.benchmark_cpu_avg_pct", 0)),
                    mlflow_run_id=row.get("run_id", ""),
                )
                results.append(result)
            except (ValueError, TypeError):
                continue

        return results

    except ImportError:
        print("Warning: mlflow not available — cannot load results")
        return []

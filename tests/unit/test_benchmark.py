"""Tests for src/benchmark.py — benchmark result collection and comparison."""

import json
import pytest
from src.benchmark import BenchmarkResult, compare_results


class TestBenchmarkResult:
    def test_basic_creation(self):
        result = BenchmarkResult(
            track="ray-scaling",
            data_size="medium",
            node_type="D16sv5",
            num_workers=4,
            train_time_sec=80.2,
            total_time_sec=128.5,
            auc_pr=1.0,
        )
        assert result.track == "ray-scaling"
        assert result.train_time_sec == 80.2

    def test_to_dict(self):
        result = BenchmarkResult(
            track="single-node-scaling",
            data_size="small",
            node_type="D16sv5",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["track"] == "single-node-scaling"

    def test_to_json(self):
        result = BenchmarkResult(
            track="gpu-scaling",
            data_size="medium",
            node_type="NC6sv3",
        )
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["track"] == "gpu-scaling"

    def test_cost_efficiency(self):
        result = BenchmarkResult(
            track="ray-scaling",
            data_size="medium",
            node_type="D16sv5",
            num_workers=4,
            cpus_per_worker=14,
            total_time_sec=128.5,
        )
        cost = result.cost_efficiency
        assert cost > 0
        assert isinstance(cost, float)


class TestCompareResults:
    def test_compare_empty(self):
        output = compare_results([])
        assert "No results" in output

    def test_compare_two_results(self):
        r1 = BenchmarkResult(
            track="single-node", data_size="medium", node_type="D16sv5",
            num_workers=1, train_time_sec=128, total_time_sec=186, auc_pr=0.9966,
        )
        r2 = BenchmarkResult(
            track="ray-scaling", data_size="medium", node_type="D16sv5",
            num_workers=4, train_time_sec=80, total_time_sec=128, auc_pr=1.0,
            cpu_avg_pct=71.0,
        )
        output = compare_results([r1, r2])
        assert "BENCHMARK COMPARISON" in output
        assert "single-node" in output
        assert "ray-scaling" in output

    def test_compare_sorted_by_train_time(self):
        r1 = BenchmarkResult(
            track="slow", data_size="medium", node_type="D16sv5",
            train_time_sec=300, total_time_sec=350,
        )
        r2 = BenchmarkResult(
            track="fast", data_size="medium", node_type="D16sv5",
            train_time_sec=80, total_time_sec=128,
        )
        output = compare_results([r1, r2])
        # Fast should appear before slow
        fast_pos = output.index("fast")
        slow_pos = output.index("slow")
        assert fast_pos < slow_pos

"""Tests for GPU benchmark eval_core — no GPU required."""

import math
import types
from unittest.mock import MagicMock

import pytest

# We can't import eval_core directly because it imports torch at module level.
# Instead, test the logic by constructing the objects manually.


def _make_mock_reference(test_cases=None, benchmark_cases=None, score_scale=3000.0):
    """Create a mock reference module with the expected attributes."""
    ref = types.ModuleType("mock_reference")
    ref.SCORE_SCALE = score_scale
    ref.BENCH_USE_CUDA_EVENTS = False  # Use wall-clock for tests
    ref.BENCH_REL_ERROR = 0.001
    ref.BENCH_WALL_TIMEOUT_NS = None
    ref.BENCH_NO_GRAD = True
    ref.BENCH_MAX_REPEATS = 5
    ref.BENCH_MAX_TIME_NS = 1e9
    ref.BENCH_WARMUP_STYLE = "tiny_benchmark"
    ref.TEST_CASES = test_cases or [{"size": 10}]
    ref.BENCHMARK_CASES = benchmark_cases or [{"size": 100}]
    return ref


class TestBenchConfig:
    """Test BenchConfig extracts reference module attributes correctly."""

    def test_defaults(self):
        """BenchConfig should use defaults when reference has no overrides."""
        ref = types.ModuleType("bare_ref")
        # Import conditionally — skip if torch not available
        try:
            from benchmarks.gpu_mode.eval_core import BenchConfig
        except ImportError:
            pytest.skip("torch not available")

        config = BenchConfig(ref)
        assert config.score_scale == 3000.0
        assert config.use_cuda_events is True
        assert config.max_repeats == 100

    def test_overrides(self):
        """BenchConfig should read explicit values from reference."""
        try:
            from benchmarks.gpu_mode.eval_core import BenchConfig
        except ImportError:
            pytest.skip("torch not available")

        ref = _make_mock_reference(score_scale=5000.0)
        config = BenchConfig(ref)
        assert config.score_scale == 5000.0
        assert config.use_cuda_events is False
        assert config.max_repeats == 5


class TestCloneData:
    """Test clone_data handles various types."""

    def test_clone_dict(self):
        try:
            from benchmarks.gpu_mode.eval_core import clone_data
        except ImportError:
            pytest.skip("torch not available")

        original = {"a": 1, "b": [2, 3]}
        cloned = clone_data(original)
        assert cloned == original
        assert cloned is not original
        assert cloned["b"] is not original["b"]

    def test_clone_tuple(self):
        try:
            from benchmarks.gpu_mode.eval_core import clone_data
        except ImportError:
            pytest.skip("torch not available")

        original = (1, {"x": 2}, [3])
        cloned = clone_data(original)
        assert cloned == original
        assert cloned[1] is not original[1]


class TestComputeStats:
    """Test compute_stats math."""

    def test_single_value(self):
        try:
            from benchmarks.gpu_mode.eval_core import compute_stats
        except ImportError:
            pytest.skip("torch not available")

        result = compute_stats([100.0])
        assert result["runs"] == 1
        assert result["mean"] == 100.0
        assert result["std"] == 0.0
        assert result["err"] == 0.0

    def test_multiple_values(self):
        try:
            from benchmarks.gpu_mode.eval_core import compute_stats
        except ImportError:
            pytest.skip("torch not available")

        result = compute_stats([100.0, 200.0])
        assert result["runs"] == 2
        assert result["mean"] == 150.0
        assert result["std"] > 0
        assert result["err"] > 0

    def test_identical_values(self):
        try:
            from benchmarks.gpu_mode.eval_core import compute_stats
        except ImportError:
            pytest.skip("torch not available")

        result = compute_stats([50.0, 50.0, 50.0])
        assert result["mean"] == 50.0
        assert result["std"] == 0.0


class TestScoringFormula:
    """Test that the scoring formula is correct: score = scale / geom_mean_us."""

    def test_scoring_math(self):
        """Verify the scoring formula matches the documented behavior."""
        score_scale = 3000.0
        # If geom_mean is 100 us, score should be 3000/100 = 30
        bench_means_ns = [100_000.0]  # 100 us in nanoseconds
        means_seconds = [ns / 1e9 for ns in bench_means_ns]
        geom_mean_s = math.pow(math.prod(means_seconds), 1.0 / len(means_seconds))
        geom_mean_us = geom_mean_s * 1e6
        score = score_scale / geom_mean_us
        assert abs(score - 30.0) < 0.001

    def test_geometric_mean_multiple_cases(self):
        """Verify geometric mean across multiple benchmark cases."""
        bench_means_ns = [100_000.0, 400_000.0]  # 100us, 400us
        means_seconds = [ns / 1e9 for ns in bench_means_ns]
        geom_mean_s = math.pow(math.prod(means_seconds), 1.0 / len(means_seconds))
        geom_mean_us = geom_mean_s * 1e6
        # geom_mean of 100 and 400 = 200
        assert abs(geom_mean_us - 200.0) < 0.001

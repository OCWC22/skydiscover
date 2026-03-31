"""
Shared GPU benchmark core — backend-agnostic correctness and benchmarking logic.

This module provides the reusable evaluation primitives that both the Python/Triton
evaluator (shared_eval.py) and future HIP evaluator (hip_backend.py) share:
  - Data cloning for correctness checks
  - Statistical timing with convergence detection
  - Kernel warmup strategies
  - Single-case benchmarking with correctness gating
  - Full evaluation pipeline: correctness → warmup → benchmark → score
  - Standardized artifact formatting

Backends provide a `KernelAdapter` that wraps the submission's callable and
handles backend-specific loading, while this module handles the rest.
"""

import copy
import contextlib
import dataclasses
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.cuda

from skydiscover.evaluation.evaluation_result import EvaluationResult


# ---------------------------------------------------------------------------
# Data cloning (used by correctness checks to preserve original inputs)
# ---------------------------------------------------------------------------

def clone_data(data):
    """Recursively clone data, handling tensors, dataclasses, and nn.Modules."""
    if isinstance(data, tuple):
        return tuple(clone_data(x) for x in data)
    if isinstance(data, list):
        return [clone_data(x) for x in data]
    if isinstance(data, dict):
        return {k: clone_data(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        fields = {f.name: clone_data(getattr(data, f.name)) for f in dataclasses.fields(data)}
        return type(data)(**fields)
    if isinstance(data, torch.nn.Module):
        cloned = copy.deepcopy(data)
        if hasattr(data, 'seq_len'):
            cloned.seq_len = data.seq_len
        return cloned
    return data


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(durations: List[float]) -> Dict[str, float]:
    """Compute statistics from a list of durations (in nanoseconds).

    Returns dict with keys: runs, mean, std, err.
    """
    n = len(durations)
    avg = sum(durations) / n
    if n > 1:
        var = sum((x - avg) ** 2 for x in durations) / (n - 1)
        std = math.sqrt(var)
        err = std / math.sqrt(n)
    else:
        std, err = 0.0, 0.0
    return {"runs": n, "mean": avg, "std": std, "err": err}


# ---------------------------------------------------------------------------
# Benchmark configuration (read from reference module)
# ---------------------------------------------------------------------------

class BenchConfig:
    """Benchmark configuration extracted from a reference module."""

    def __init__(self, reference_module):
        self.score_scale = getattr(reference_module, 'SCORE_SCALE', 3000.0)
        self.use_cuda_events = getattr(reference_module, 'BENCH_USE_CUDA_EVENTS', True)
        self.rel_error = getattr(reference_module, 'BENCH_REL_ERROR', 0.001)
        self.wall_timeout_ns = getattr(reference_module, 'BENCH_WALL_TIMEOUT_NS', 120e9)
        self.no_grad = getattr(reference_module, 'BENCH_NO_GRAD', False)
        self.max_repeats = getattr(reference_module, 'BENCH_MAX_REPEATS', 100)
        self.max_time_ns = getattr(reference_module, 'BENCH_MAX_TIME_NS', 10e9)
        self.warmup_style = getattr(reference_module, 'BENCH_WARMUP_STYLE', 'tiny_benchmark')
        self.test_cases = getattr(reference_module, 'TEST_CASES', [])
        self.benchmark_cases = getattr(reference_module, 'BENCHMARK_CASES', [])


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup_kernel(kernel_fn: Callable, bench_args: dict, bench_config: BenchConfig,
                  generate_input_fn: Callable) -> None:
    """Warmup the kernel to trigger JIT compilation."""
    if bench_config.warmup_style == 'timed_calls':
        data = generate_input_fn(**bench_args)
        start = time.perf_counter()
        while time.perf_counter() - start < 0.2:
            kernel_fn(data)
            torch.cuda.synchronize()
    else:
        bench_single(kernel_fn, bench_args, bench_config, generate_input_fn,
                     max_time_ns=10e7)


# ---------------------------------------------------------------------------
# Single-case benchmark
# ---------------------------------------------------------------------------

def bench_single(
    kernel_fn: Callable,
    bench_args: dict,
    bench_config: BenchConfig,
    generate_input_fn: Callable,
    check_fn: Optional[Callable] = None,
    max_time_ns: Optional[float] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Benchmark a kernel on a single case.

    Args:
        kernel_fn: The kernel callable.
        bench_args: kwargs passed to generate_input_fn.
        bench_config: Benchmark configuration.
        generate_input_fn: reference.generate_input function.
        check_fn: Optional correctness check (data_copy, output) -> (bool, str).
        max_time_ns: Override for max benchmark time.

    Returns:
        (stats_dict_or_None, error_str_or_None).
        Stats dict has durations in nanoseconds.
    """
    if max_time_ns is None:
        max_time_ns = bench_config.max_time_ns

    data = generate_input_fn(**bench_args)
    data_copy = clone_data(data)

    ctx = torch.no_grad() if bench_config.no_grad else contextlib.nullcontext()
    with ctx:
        output = kernel_fn(data)
        torch.cuda.synchronize()
        if check_fn is not None:
            passed, msg = check_fn(data_copy, output)
            if not passed:
                return None, f"Benchmark correctness: {msg}"
    del output

    # Timed runs
    durations_ns = []
    bm_start = time.perf_counter_ns()

    with ctx:
        for i in range(bench_config.max_repeats):
            torch.cuda.synchronize()

            if bench_config.use_cuda_events:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                output = kernel_fn(data)
                e.record()
                torch.cuda.synchronize()
                duration_ns = s.elapsed_time(e) * 1e6
            else:
                start_ns = time.perf_counter_ns()
                output = kernel_fn(data)
                torch.cuda.synchronize()
                duration_ns = time.perf_counter_ns() - start_ns

            del output
            durations_ns.append(duration_ns)

            if i > 1:
                st = compute_stats(durations_ns)
                if st["mean"] > 0 and st["err"] / st["mean"] < bench_config.rel_error:
                    break
                if st["mean"] * st["runs"] > max_time_ns:
                    break
                if bench_config.wall_timeout_ns is not None and \
                   (time.perf_counter_ns() - bm_start) > bench_config.wall_timeout_ns:
                    break

    return compute_stats(durations_ns), None


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_correctness(
    kernel_fn: Callable,
    test_cases: List[dict],
    generate_input_fn: Callable,
    check_fn: Callable,
    no_grad: bool = False,
) -> Optional[EvaluationResult]:
    """Run correctness checks. Returns EvaluationResult on failure, None on success."""
    import traceback as tb_module

    for i, tc in enumerate(test_cases):
        try:
            data = generate_input_fn(**tc)
            data_copy = clone_data(data)
            torch.cuda.synchronize()
            ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
            with ctx:
                output = kernel_fn(data)
            torch.cuda.synchronize()
            passed, msg = check_fn(data_copy, output)
            if not passed:
                return EvaluationResult(
                    metrics={"combined_score": 0.0, "correctness": 0.0},
                    artifacts={
                        "error": f"Test {i} failed: {msg}",
                        "failure_stage": "correctness",
                        "test_index": str(i),
                        "backend": "unknown",
                    },
                )
        except Exception as exc:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "correctness": 0.0},
                artifacts={
                    "error": f"Test {i} error: {exc}",
                    "traceback": tb_module.format_exc(),
                    "failure_stage": "correctness",
                    "test_index": str(i),
                    "backend": "unknown",
                },
            )
    return None  # All passed


def run_benchmarks(
    kernel_fn: Callable,
    bench_config: BenchConfig,
    generate_input_fn: Callable,
    check_fn: Callable,
    backend_name: str = "local",
) -> EvaluationResult:
    """Run warmup + benchmarks + scoring. Assumes correctness already passed.

    Returns EvaluationResult with metrics and artifacts.
    """
    # Warmup
    warmup_kernel(kernel_fn, bench_config.benchmark_cases[0], bench_config, generate_input_fn)

    # Benchmark each case
    bench_means_ns = []
    for bench_args in bench_config.benchmark_cases:
        st, err = bench_single(kernel_fn, bench_args, bench_config,
                               generate_input_fn, check_fn)
        if err:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "correctness": 1.0},
                artifacts={
                    "error": err,
                    "failure_stage": "benchmark",
                    "backend": backend_name,
                },
            )
        bench_means_ns.append(st["mean"])

    # Score: geometric mean → microseconds → score
    means_seconds = [ns / 1e9 for ns in bench_means_ns]
    geom_mean_s = math.pow(math.prod(means_seconds), 1.0 / len(means_seconds))
    geom_mean_us = geom_mean_s * 1e6
    score = bench_config.score_scale / geom_mean_us

    metrics = {
        "combined_score": score,
        "correctness": 1.0,
        "geom_mean_us": geom_mean_us,
    }
    artifacts = {
        "hardware": backend_name,
        "backend": backend_name,
    }
    for i, ns in enumerate(bench_means_ns):
        artifacts[f"bench_{i}_mean_us"] = f"{ns / 1e3:.2f}"

    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def evaluate_kernel(
    kernel_fn: Callable,
    bench_config: BenchConfig,
    generate_input_fn: Callable,
    check_fn: Callable,
    backend_name: str = "local",
) -> EvaluationResult:
    """Full evaluation: correctness → warmup → benchmark → score.

    This is the main entry point for backend adapters.
    """
    # Correctness
    fail = run_correctness(
        kernel_fn, bench_config.test_cases, generate_input_fn,
        check_fn, no_grad=bench_config.no_grad,
    )
    if fail is not None:
        fail.artifacts["backend"] = backend_name
        return fail

    # Benchmark
    return run_benchmarks(
        kernel_fn, bench_config, generate_input_fn, check_fn,
        backend_name=backend_name,
    )

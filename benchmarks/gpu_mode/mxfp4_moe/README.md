# GPU Mode: MXFP4 Mixture-of-Experts (MoE) Kernel

Evolve a high-performance fused MoE kernel using SkyDiscover. The benchmark implements a sparse Mixture-of-Experts FFN forward pass with expert weights stored in OCP Microscaling FP4 (MXFP4) format, targeting AMD Instinct MI355X (cDNA4) GPUs.

## Task

Given input activations `x` in bfloat16 and MXFP4-packed expert weights:

1. Route tokens to top-k experts via a learned router
2. Dequantize selected experts' MXFP4 weights (FP4 E2M1 + per-block E8M0 scales)
3. Compute `SiLU(x @ W1) @ W2` for each selected expert
4. Combine with softmax gate scores → output

## Hardware Target

| Spec | Value |
|------|-------|
| GPU | AMD Instinct MI355X (cDNA4) |
| Compute | 9.2 PFLOPS FP4, 2.3 PFLOPS FP16 |
| Memory | 288 GB HBM3E, 8 TB/s |
| Wavefront | 64 lanes (NOT 32) |
| Matrix Cores | MFMA with block exponent scaling |
| LDS | 64 KB per CU |

## Scoring

Same formula as all GPU benchmarks:

```
combined_score = 3000.0 / geom_mean_us
```

Higher score = faster kernel. `geom_mean_us` is the geometric mean of kernel runtimes in microseconds across benchmark cases.

## Benchmark Cases

| Case | Tokens | Hidden | Intermediate | Experts | Top-K | Purpose |
|------|--------|--------|-------------|---------|-------|---------|
| Test 1 | 16 | 1024 | 4096 | 8 | 2 | Correctness |
| Test 2 | 32 | 2048 | 8192 | 8 | 2 | Correctness |
| Test 3 | 64 | 4096 | 14336 | 16 | 2 | Correctness |
| Test 4 | 64 | 4096 | 14336 | 16 | 4 | Correctness |
| Bench 1 | 128 | 4096 | 14336 | 16 | 2 | **Performance** |
| Bench 2 | 256 | 4096 | 14336 | 16 | 2 | **Performance** |

## Quick Start

```bash
# Run locally on MI355X with ROCm
uv run skydiscover-run \
  benchmarks/gpu_mode/mxfp4_moe/initial_program.py \
  benchmarks/gpu_mode/mxfp4_moe/evaluator.py \
  -c benchmarks/gpu_mode/mxfp4_moe/config.yaml \
  -s adaevolve \
  -i 100

# Or with EvoX
uv run skydiscover-run \
  benchmarks/gpu_mode/mxfp4_moe/initial_program.py \
  benchmarks/gpu_mode/mxfp4_moe/evaluator.py \
  -c benchmarks/gpu_mode/mxfp4_moe/config.yaml \
  -s evox \
  -i 100
```

## Requirements

- **ROCm 7.x** with PyTorch ROCm build
- **Triton** with ROCm backend (`triton-rocm` or upstream Triton with AMD support)
- AMD Instinct MI355X GPU (or compatible cDNA4 accelerator)

> **Note**: `shared_eval.py`'s local evaluation path works on ROCm because PyTorch exposes the `torch.cuda` API surface through HIP compatibility. The `modal_eval.py` remote backend is **NVIDIA-only** — `GPUMODE_USE_MODAL=true` is not supported for this benchmark.

## MXFP4 Format

OCP Microscaling FP4 (E2M1):
- Two 4-bit codes packed per `uint8` byte (low nibble = even index, high nibble = odd index)
- 32-element blocks with shared E8M0 scaling factor (power-of-two)
- Value range: ±0.5 (subnormal) to ±6.0 (normal)

## Future: Remote AMD Evaluation

To add MI355X remote evaluation support:
1. Create `benchmarks/gpu_mode/rocm_eval.py` with the same `**kwargs → dict` interface as `modal_eval._eval_triton_impl`
2. Extend `shared_eval._evaluate_modal()` dispatch to recognize `MI355X` 
3. See `MODAL_REFERENCE_CODE` in `reference.py` for the self-contained evaluation contract

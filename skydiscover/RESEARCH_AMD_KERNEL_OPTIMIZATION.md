# Deep Research: AMD Kernel Optimization Ecosystem for MXFP4 MOE on cDNA4

## Executive Summary

AMD has built an extensive, layered ecosystem for GPU kernel optimization — from low-level HIP/ASM to high-level AI agents. For our MXFP4 MOE kernel benchmark in SkyDiscover, the most directly relevant tools are:

1. **AITER** — AMD's production kernel library with **ready-to-use MXFP4 MoE kernels**
2. **GEAK** — AMD's own AI agent for Triton/HIP kernel generation (directly comparable to SkyDiscover)
3. **FlyDSL** — Python-native kernel DSL that enabled production MoE kernels in days instead of weeks
4. **CK-Tile** — Composable Kernel framework for hand-tuned GEMM/MoE/Attention kernels
5. **DRTriton** — RL-trained model that generates optimized Triton kernels from PyTorch code

---

## 1. AITER — AI Tensor Engine for ROCm

**Source**: [ROCm/aiter on GitHub](https://github.com/ROCm/aiter) | [AITER Blog Post](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)

### What It Is
AITER is AMD's centralized, open-source library of pre-optimized AI operators for ROCm GPUs. It provides JIT-compiled implementations of the exact operations our benchmark targets.

### Key Operators Relevant to Us

| Operator | Performance Gain | Notes |
|----------|-----------------|-------|
| **Fused MoE (block-scale MXFP4)** | **Up to 3x boost** | Automatically selects optimal MoE kernel by quantization method |
| Fused MLA decode | Up to 17x boost | Production-used in DeepSeek-R1 serving |
| Fused MHA prefill | Up to 14x boost | Multi-head attention for prefill |
| Block-scaled GEMM | Native MXFP4 support | Uses MFMA scale instructions on cDNA4 |

### Why This Matters for SkyDiscover
- AITER's `fused_moe` API is the **production reference** for MXFP4 MoE on MI355X
- ROCm 7.0 ships MXFP4 MoE kernels pre-integrated in AITER
- Our benchmark's `initial_program.py` could use AITER as a strong baseline seed
- The LLM system prompt in `config.yaml` should reference AITER patterns as optimization targets

### Integration with vLLM/SGLang
- [vLLM PR #14982](https://github.com/vllm-project/vllm/pull/14982) integrates AITER fused MoE
- AITER's `fused_moe` API auto-selects kernel type based on quantization method
- ROCM_AITER_FA delivers **2.8–4.6x faster** TPOT vs legacy attention backends

### Also: JAX-AITER
- [JAX-AITER Blog](https://rocm.blogs.amd.com/software-tools-optimization/jax-aiter/README.html) — brings the same optimized kernels to JAX on ROCm

---

## 2. GEAK — Generating Efficient AI-centric GPU Kernels

**Source**: [AMD-AGI/GEAK on GitHub](https://github.com/AMD-AGI/GEAK) | [GEAK Paper (arXiv:2507.23194)](https://arxiv.org/abs/2507.23194) | [GEAK v2 Blog](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html)

### What It Is
GEAK is AMD's own AI agent framework for automated kernel generation — **directly comparable to SkyDiscover**. It has three sub-agents:

#### GEAK-OptimAgentv2 (Instruction → Triton)
- Generates Triton kernels from natural language descriptions
- Multi-offspring evolution with LLM-based evaluator
- **Hardware-aware feedback loop** — profiles on actual AMD hardware
- Used for instruction-to-kernel generation

#### GEAK-OpenEvolve (Triton → Optimized Triton)
- Built on AlphaEvolve/OpenEvolve evolutionary framework
- **Quality-Diversity (MAP-Elites)** search — maintains diverse, high-quality kernel population
- Results: **3.42x speedup** on TritonBench, **7.02x speedup** on ROCm benchmark
- Directly analogous to SkyDiscover's AdaEvolve/EvoX

#### GEAK-HIP (HIP → Optimized HIP)
- [GEAK HIP Blog](https://rocm.blogs.amd.com/software-tools-optimization/geak-hip-optimizations/README.html)
- Optimizes existing HIP C++ kernels
- **2.07x speedup** on Voxelization, **1.68x on SwiGLU** — beating hand-optimized versions
- Extends GEAK to lower-level HIP code, not just Triton

### Authors
- **Jianghui Wang**, Vinay Joshi, Saptarshi Majumder, Xu Chao, **Bin Ding**, Ziqiong Liu, Pratik Prabhanjan Brahma, **Dong Li**, **Zicheng Liu**, Emad Barsoum (AMD)
- **Sharon Zhou** acknowledged for constructive feedback

### Key Differences: GEAK vs SkyDiscover

| Feature | GEAK | SkyDiscover |
|---------|------|-------------|
| Primary target | AMD Instinct GPUs specifically | Hardware-agnostic (any evaluator) |
| Search method | Reflexion + MAP-Elites (OpenEvolve) | AdaEvolve (adaptive UCB) + EvoX (co-evolution) |
| Hardware feedback | Profiles on AMD hardware directly | Evaluator-provided (configurable) |
| Scope | GPU kernels only | 200+ tasks (math, systems, algorithms, kernels) |
| Open source | Yes (GitHub) | Yes (GitHub) |
| HIP support | Yes (GEAK-HIP) | No (Triton/PyTorch only) |

### Implication for Our Work
We could reference GEAK's optimization patterns in our system prompt — the LLM should know about the specific AMD hardware optimizations GEAK discovered.

---

## 3. FlyDSL — Python-Native Kernel DSL

**Source**: [FlyDSL + Kimi-K2.5 Blog](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)

### What It Is
FlyDSL is a Python-native kernel DSL that compiles through MLIR passes to optimized binaries for AMD GPUs. Its key advantage is **development speed** — production-quality kernels in a fraction of the time of CK or assembly.

### Why It Matters
- Enabled a **mixed-precision (W4A16 + BF16) fused MoE kernel** for Kimi-K2.5 in days
- Profiling showed fused_moe was **88–90% of total GPU time** — making MoE the #1 optimization target
- CK couldn't handle the Kimi-K2.5 shapes (E=384 experts), Triton needed extensive tuning
- FlyDSL compiled the solution targeting both **gfx942 (MI300X)** and **gfx950 (MI350/MI355X)**

### Technical Details
- Written entirely in Python
- Compiles via MLIR pipeline → optimized ISA
- Adopted as AMD's **default** for Kimi-K2.5 MoE inference
- Supports mixed-precision quantization paths that CK doesn't yet cover

### For Our Benchmark
FlyDSL represents a potential alternative to Triton for the initial_program.py if we want maximum performance. However, Triton is the better choice for SkyDiscover's LLM-driven evolution since LLMs understand Triton syntax well.

---

## 4. CK-Tile — Composable Kernel Framework

**Source**: [CK-Tile GEMM Blog](https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html) | [CK Docs](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/)

### What It Is
CK-Tile is AMD's C++ template library for building portable, high-performance kernels. It's the layer beneath AITER — many AITER kernels are built on CK-Tile.

### Key Features
- **Vendor-optimized GEMM pipelines** with tunable tile sizes, pipeline stages, and wave counts
- **LDS bank conflict avoidance** via XOR-based swizzle transformations
- **TilePartitioner API**: maps problem dimensions to GPU hierarchy (workgroup → wave → thread)
- APIs for fused-MHA, fused-MoE, SmoothQuant, element-wise kernels

### LDS Bank Conflict Blog
[Avoiding LDS Bank Conflicts on AMD GPUs Using CK-Tile](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
- LDS organized into 32 or 64 banks × 4 bytes each
- Padding reduces bank conflicts but costs occupancy
- XOR swizzle can achieve conflict-free access patterns
- Critical for our MXFP4 dequant kernel — 32-element blocks must be tiled to avoid LDS conflicts

### For Our Benchmark
CK-Tile patterns should be referenced in the system prompt — the LLM should know about:
- Optimal tile sizes for cDNA4 (256×256 tiles work well per FP8 GEMM blog)
- LDS bank conflict avoidance via swizzling
- Pipeline staging for fused operations

---

## 5. DRTriton — RL-Trained Triton Kernel Generator

**Source**: [DRTriton Paper (arXiv:2603.21465)](https://arxiv.org/abs/2603.21465)

### What It Is
A 7B model (based on Qwen-2.5-Coder) trained via curriculum reinforcement learning on synthetic data to convert PyTorch code → optimized Triton kernels.

### Key Results
- **DRTriton-7B achieves speedup on 92% of KernelBench Level 2**
- Compare: GPT-5.2 achieves speedup on only 23%, Claude-Sonnet-4.5 on 19%
- Trained on just 2,026 synthetic pairs + 100K synthetic RL programs
- **Generalizes to real-world kernels** despite training on synthetic data only

### Three-Stage Pipeline
1. **CSP-DAG**: Synthetic data generation with guaranteed operator coverage
2. **Curriculum RL**: Decoupled reward for correctness + speed simultaneously
3. **Test-time search**: Further optimizes generated kernels at inference time

### For Our Benchmark
DRTriton could potentially be used as one of the LLM backends in SkyDiscover's multi-model pool — it's specifically trained for Triton kernel generation and dramatically outperforms general-purpose models.

---

## 6. Triton on ROCm — Key Optimization Knowledge

### AMD-Specific Triton Internals
**Source**: [Optimizing Triton Kernels on ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-triton-kernel.html) | [Peak Performance Blog](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)

Key Triton → AMD mapping:
- `tl.arange` → MFMA layout (`amd_mfma`) for matrix ops
- Shared memory → LDS (Local Data Share)
- `num_warps` → controls wavefronts per workgroup (but wavefront=64, NOT 32)
- `num_stages` parameter: 0 for single GEMM, 1 for two fused GEMMs (like Flash Attention)

### cDNA4-Specific MFMA Instructions
**Source**: [Matrix Core Programming on cDNA3/cDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)

- `V_MFMA_SCALE_F32_16X16X128_F8F6F4` — block-scaled MFMA for mixed FP4/FP6/FP8
- `V_MFMA_SCALE_F32_32X32X64_F8F6F4` — larger tile variant
- `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4` — HIP compiler intrinsic (256-bit wide args)
- MicroScaling (MX) formats: **hardware-accelerated dequant + matmul in single instruction**
- 64 lanes cooperate per MFMA instruction (wavefront64)

### FP8 GEMM on cDNA4 (Directly Applicable to FP4)
**Source**: [FP8 GEMM Optimization on cDNA4](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)

Key optimization findings:
- **LDS**: 160 KB per CU (up from cDNA3), 256 B/clk read bandwidth
- **GLOBAL_LOAD_LDS**: 128-bit per-lane transfer width (vs 32-bit on cDNA3) — 4x wider loads
- **Best GEMM config**: 256×256 tile with 512 threads (not smaller tiles)
- **Swizzling** eliminates LDS bank conflicts
- `mfma_16x16` typically outperforms `mfma_32x32` even for large tiles

---

## 7. The People

### Key AMD Researchers in This Space

#### GEAK Team (AMD AGI)
- **Jianghui Wang** — Lead author of GEAK paper
- **Bin Ding** — GEAK author, ROCm blog author
- **Dong Li** — GEAK author, ROCm blog posts
- **Zicheng Liu** — GEAK author
- **Emad Barsoum** — AMD Senior VP, GEAK team lead
- **Sharon Zhou** — Acknowledged in GEAK for constructive feedback; panelist at AMD Open Source AI Week 2025

#### Triton on ROCm
- **Ning Zhang** — "Unlock Peak Performance on AMD GPUs with Triton Kernel Optimizations" blog author (April 2025)
- **Lei Zhang** — Triton-Distributed blog author
- **George Wang** — Multiple ROCm optimization blog posts, CK-Tile work

#### AITER / Production Kernels
- **SageMoore** — vLLM AITER fused MoE integration (GitHub PRs)
- **David Li** — CK-Tile hands-on GEMM blog author

### Note on "Sharon Zhang/Chang"
I searched extensively but could not find a specific "Sharon Zhang" or "Sharon Chang" at AMD publishing kernel optimization work. The closest match is **Sharon Zhou**, who is acknowledged in the GEAK-Triton v2 work and appeared at AMD's Open Source AI Week. The name may also refer to **Ning Zhang** (Triton optimization blog) or **Lei Zhang** (Triton-Distributed). If you can recall more context about where you saw the name, I can search more specifically.

---

## 8. How This Maps to Our SkyDiscover Benchmark

### What We Should Incorporate into config.yaml System Prompt

Our current system prompt is good but should be enriched with these AMD-specific optimization patterns:

1. **MFMA Scale Instructions**: Reference `V_MFMA_SCALE_F32_16X16X128_F8F6F4` — the hardware does dequant+matmul in ONE instruction. This is the ultimate optimization target.

2. **LDS Sizing on cDNA4**: 160 KB per CU (not the generic "64 KB" in our current prompt). 256 B/clk read bandwidth. Tell the LLM about 128-bit GLOBAL_LOAD_LDS.

3. **Optimal Tile Sizes**: 256×256 tiles with 512 threads per AMD's FP8 GEMM blog. `mfma_16x16` beats `mfma_32x32`.

4. **LDS Swizzling**: XOR-based swizzle patterns from CK-Tile to avoid bank conflicts.

5. **num_stages Tuning**: 0 for single GEMM ops, 1 for fused two-GEMM sequences.

6. **AITER Patterns**: Reference the `fused_moe` API pattern — batched expert execution, block-scale dequant.

### Potential SkyDiscover Enhancements

1. **DRTriton as LLM backend**: Add DRTriton-7B to the model pool — it achieves 92% speedup rate on kernel benchmarks vs 23% for GPT-5.2

2. **GEAK-style hardware feedback**: Add AMD profiling data (wavefront occupancy, LDS utilization, MFMA efficiency) as evaluator artifacts fed back to the LLM

3. **Multi-level optimization**: Start with Triton, then evolve to HIP/CK-Tile patterns for maximum performance

---

## 9. Complete AMD Kernel Optimization Stack

```
Layer 5: AI Agents
├── GEAK-OptimAgentv2  (instruction → Triton)
├── GEAK-OpenEvolve    (Triton → optimized Triton, evolutionary)
├── GEAK-HIP           (HIP → optimized HIP)
├── DRTriton           (PyTorch → Triton, RL-trained 7B model)
└── SkyDiscover        (evaluator-driven evolution, our tool)

Layer 4: High-Level DSLs
├── Triton ROCm        (Python-like GPU kernel DSL)
└── FlyDSL             (Python-native, MLIR → ISA, used for prod MoE)

Layer 3: Kernel Libraries
├── AITER              (pre-optimized fused operators: MoE, MLA, GEMM)
├── CK-Tile            (C++ template framework for portable kernels)
└── hipBLASLt          (BLAS-level GEMM operations)

Layer 2: Runtime
├── ROCm 7.x           (AMD GPU compute stack)
├── HIP                (CUDA-compatible programming model)
└── Composable Kernel  (low-level kernel building blocks)

Layer 1: Hardware ISA
├── MFMA instructions  (matrix core operations)
├── MFMA Scale         (block-scaled FP4/FP6/FP8 with dequant)
├── LDS                (160 KB shared memory, 256 B/clk)
└── Wavefront64        (64-lane SIMD execution)
```

---

## Sources

### AMD Official
- [AITER GitHub](https://github.com/ROCm/aiter)
- [AITER Blog](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)
- [AITER MLA Blog](https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html)
- [GEAK GitHub](https://github.com/AMD-AGI/GEAK)
- [GEAK-Triton v2 Blog](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html)
- [GEAK v1 Blog](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html)
- [GEAK HIP Blog](https://rocm.blogs.amd.com/software-tools-optimization/geak-hip-optimizations/README.html)
- [FlyDSL + Kimi-K2.5 MoE Blog](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)
- [Matrix Core Programming on cDNA3/cDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [FP8 GEMM Optimization on cDNA4](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
- [Triton Kernel Optimization on ROCm](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)
- [CK-Tile GEMM Blog](https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html)
- [LDS Bank Conflicts on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
- [MXFP4/MXFP6 Quantization on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [ROCm 7.0 Release Blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.0-blog/README.html)
- [Triton-Distributed Blog](https://rocm.blogs.amd.com/software-tools-optimization/triton-distributed-c/README.html)
- [Optimizing Triton Kernels (ROCm Docs)](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-triton-kernel.html)
- [Block Scaled MatMul (Triton Tutorial)](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
- [cDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [vLLM AITER MoE Integration PR](https://github.com/vllm-project/vllm/pull/14982)
- [vLLM ROCm Attention Backend Blog](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html)

### Papers
- [GEAK Paper (arXiv:2507.23194)](https://arxiv.org/abs/2507.23194)
- [DRTriton Paper (arXiv:2603.21465)](https://arxiv.org/abs/2603.21465)
- [Dr. Kernel Paper (arXiv:2602.05885)](https://arxiv.org/abs/2602.05885)
- [AutoTriton Paper (arXiv:2507.05687)](https://arxiv.org/abs/2507.05687)

"""
Reference implementation for MXFP4 Mixture-of-Experts (MoE) fused kernel.

Target hardware: AMD Instinct MI355X (cDNA4 architecture).
Weights are stored in OCP Microscaling FP4 (MXFP4) format:
  - Two 4-bit E2M1 codes packed per uint8 byte
  - One E8M0 shared exponent (power-of-two scale) per 32-element block
  - Activations in bfloat16

The MoE forward pass:
  1. Router: compute expert affinities, select top-k experts per token
  2. Gate: softmax over selected expert scores
  3. Expert FFN: for each selected expert, dequantize MXFP4 weights then
     compute SiLU(x @ W1) @ W2
  4. Combine: weighted sum of expert outputs by gate scores
"""

import math
import gc
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Scoring and benchmark configuration (read by shared_eval.py)
# ---------------------------------------------------------------------------

SCORE_SCALE = 3000.0

# Wall-clock timing (includes routing + grouping + dequant + matmul orchestration)
BENCH_USE_CUDA_EVENTS = False
BENCH_REL_ERROR = 0.01
BENCH_WALL_TIMEOUT_NS = None
BENCH_NO_GRAD = True
BENCH_MAX_REPEATS = 100
BENCH_MAX_TIME_NS = 10e9
BENCH_WARMUP_STYLE = 'timed_calls'


# ---------------------------------------------------------------------------
# OCP MXFP4 (E2M1) lookup table
# ---------------------------------------------------------------------------
# 4-bit E2M1 format: 1 sign + 2 exponent + 1 mantissa, bias=1
# Code | S EE M | Value
# 0000 | 0 00 0 |  0.0    (positive zero / subnormal)
# 0001 | 0 00 1 |  0.5    (subnormal: 0.mantissa * 2^(1-bias) = 0.1 * 2^0)
# 0010 | 0 01 0 |  1.0
# 0011 | 0 01 1 |  1.5
# 0100 | 0 10 0 |  2.0
# 0101 | 0 10 1 |  3.0
# 0110 | 0 11 0 |  4.0
# 0111 | 0 11 1 |  6.0
# 1000 | 1 00 0 | -0.0    (negative zero, treated as 0.0)
# 1001 | 1 00 1 | -0.5
# 1010 | 1 01 0 | -1.0
# 1011 | 1 01 1 | -1.5
# 1100 | 1 10 0 | -2.0
# 1101 | 1 10 1 | -3.0
# 1110 | 1 11 0 | -4.0
# 1111 | 1 11 1 | -6.0

_FP4_E2M1_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _fp4_lut(device: torch.device) -> torch.Tensor:
    """Return the 16-entry FP4 E2M1 dequantization lookup table."""
    return torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# MXFP4 packing / unpacking helpers
# ---------------------------------------------------------------------------

def _unpack_nibbles(packed: torch.Tensor, out_dim: int) -> torch.Tensor:
    """Unpack uint8 tensor with two FP4 codes per byte into int codes.

    Args:
        packed: [..., out_dim // 2] uint8 tensor
        out_dim: number of FP4 elements in the last dimension

    Returns:
        [..., out_dim] int32 tensor of 4-bit code indices (0..15)
    """
    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    # Interleave: element 2i = low nibble, element 2i+1 = high nibble
    shape = packed.shape[:-1] + (out_dim,)
    out = torch.empty(shape, dtype=torch.int32, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _dequantize_matrix(packed: torch.Tensor, scales: torch.Tensor,
                       block_size: int, out_dim: int) -> torch.Tensor:
    """Dequantize an MXFP4-packed weight matrix to float32.

    Args:
        packed: [..., out_dim // 2] uint8 — packed FP4 codes
        scales: [..., out_dim // block_size] float32 — per-block E8M0 scales
        block_size: number of elements sharing one scale (32 for OCP MX)
        out_dim: number of output elements in last dimension

    Returns:
        [..., out_dim] float32 dequantized matrix
    """
    device = packed.device
    lut = _fp4_lut(device)
    codes = _unpack_nibbles(packed, out_dim)  # [..., out_dim]
    values = lut[codes]  # [..., out_dim] float32

    # Expand scales to match: [..., out_dim // block_size] -> [..., out_dim]
    num_blocks = out_dim // block_size
    # Reshape values into blocks, multiply by scale, flatten back
    batch_shape = values.shape[:-1]
    values = values.view(*batch_shape, num_blocks, block_size)
    scales_expanded = scales.unsqueeze(-1)  # [..., num_blocks, 1]
    values = values * scales_expanded
    values = values.view(*batch_shape, out_dim)
    return values


def _pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack int code tensor [..., N] into uint8 [..., N//2] (low/high nibble)."""
    assert codes.shape[-1] % 2 == 0
    low = codes[..., 0::2].to(torch.uint8)
    high = codes[..., 1::2].to(torch.uint8)
    return low | (high << 4)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class MoEConfig:
    """Configuration and weights for the MXFP4 MoE benchmark.

    Weight storage:
      - w1_packed: [num_experts, hidden_dim, intermediate_dim // 2] uint8
        (gate+up projection, MXFP4-packed)
      - w1_scales: [num_experts, hidden_dim, intermediate_dim // block_size] float32
      - w2_packed: [num_experts, intermediate_dim, hidden_dim // 2] uint8
        (down projection, MXFP4-packed)
      - w2_scales: [num_experts, intermediate_dim, hidden_dim // block_size] float32
      - router_weight: [num_experts, hidden_dim] bfloat16
    """
    num_tokens: int
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    top_k: int
    block_size: int  # fixed at 32 (OCP MX standard)
    router_weight: torch.Tensor   # [E, H] bf16
    w1_packed: torch.Tensor       # [E, H, I//2] uint8
    w1_scales: torch.Tensor       # [E, H, I//block_size] float32
    w2_packed: torch.Tensor       # [E, I, H//2] uint8
    w2_scales: torch.Tensor       # [E, I, H//block_size] float32


# ---------------------------------------------------------------------------
# Test / benchmark cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {"num_tokens": 16,  "hidden_dim": 1024, "intermediate_dim": 4096,
     "num_experts": 8,  "top_k": 2, "seed": 9247},
    {"num_tokens": 32,  "hidden_dim": 2048, "intermediate_dim": 8192,
     "num_experts": 8,  "top_k": 2, "seed": 2197},
    {"num_tokens": 64,  "hidden_dim": 4096, "intermediate_dim": 14336,
     "num_experts": 16, "top_k": 2, "seed": 9107},
    {"num_tokens": 64,  "hidden_dim": 4096, "intermediate_dim": 14336,
     "num_experts": 16, "top_k": 4, "seed": 5291},
]

BENCHMARK_CASES = [
    {"num_tokens": 128, "hidden_dim": 4096, "intermediate_dim": 14336,
     "num_experts": 16, "top_k": 2, "seed": 9817},
    {"num_tokens": 256, "hidden_dim": 4096, "intermediate_dim": 14336,
     "num_experts": 16, "top_k": 2, "seed": 5292},
]


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def generate_input(num_tokens, hidden_dim, intermediate_dim, num_experts,
                   top_k, seed):
    """Generate benchmark input with pre-packed MXFP4 expert weights.

    Returns:
        Tuple of (MoEConfig, x) where x is [num_tokens, hidden_dim] bf16.
    """
    block_size = 32
    assert intermediate_dim % block_size == 0
    assert hidden_dim % block_size == 0
    assert intermediate_dim % 2 == 0
    assert hidden_dim % 2 == 0

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    # Router weight: bf16 [E, H]
    router_weight = (torch.randn(
        (num_experts, hidden_dim), dtype=torch.float32,
        generator=gen, device='cuda'
    ) / math.sqrt(hidden_dim)).to(torch.bfloat16)

    # W1: [E, H, I] packed as MXFP4
    # Generate random FP4 codes (0..15) then pack
    w1_codes = torch.randint(
        0, 16, (num_experts, hidden_dim, intermediate_dim),
        generator=gen, device='cuda', dtype=torch.int32
    )
    w1_packed = _pack_fp4_codes(w1_codes)
    w1_scales = (2.0 ** torch.randint(
        -4, 5, (num_experts, hidden_dim, intermediate_dim // block_size),
        generator=gen, device='cuda', dtype=torch.int32
    ).float())

    # W2: [E, I, H] packed as MXFP4
    w2_codes = torch.randint(
        0, 16, (num_experts, intermediate_dim, hidden_dim),
        generator=gen, device='cuda', dtype=torch.int32
    )
    w2_packed = _pack_fp4_codes(w2_codes)
    w2_scales = (2.0 ** torch.randint(
        -4, 5, (num_experts, intermediate_dim, hidden_dim // block_size),
        generator=gen, device='cuda', dtype=torch.int32
    ).float())

    config = MoEConfig(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=top_k,
        block_size=block_size,
        router_weight=router_weight,
        w1_packed=w1_packed,
        w1_scales=w1_scales,
        w2_packed=w2_packed,
        w2_scales=w2_scales,
    )

    # Input activations: bf16 [T, H]
    x = (torch.randn(
        (num_tokens, hidden_dim), dtype=torch.float32,
        generator=gen, device='cuda'
    ) / math.sqrt(hidden_dim)).to(torch.bfloat16)

    return config, x


# ---------------------------------------------------------------------------
# Reference kernel
# ---------------------------------------------------------------------------

@torch.no_grad()
def ref_kernel(data):
    """Reference MoE forward pass with MXFP4 weight dequantization.

    Args:
        data: Tuple of (MoEConfig, x) where x is [T, H] bf16.

    Returns:
        output: [T, H] bf16 tensor.
    """
    config, x = data
    T = config.num_tokens
    H = config.hidden_dim
    I = config.intermediate_dim
    E = config.num_experts
    K = config.top_k
    BS = config.block_size

    # Step 1: Router — compute expert scores and select top-k
    router_logits = x.float() @ config.router_weight.float().T  # [T, E]
    topk_scores, topk_experts = torch.topk(router_logits, k=K, dim=-1)  # [T, K]
    gates = torch.softmax(topk_scores, dim=-1).float()  # [T, K]

    # Step 2: Group tokens by expert and compute FFN
    output = torch.zeros(T, H, dtype=torch.float32, device=x.device)

    for expert_idx in range(E):
        # Find which (token, slot) pairs selected this expert
        mask = (topk_experts == expert_idx)  # [T, K]
        if not mask.any():
            continue

        token_indices, slot_indices = torch.where(mask)  # both [num_selected]

        # Dequantize this expert's weights
        w1 = _dequantize_matrix(
            config.w1_packed[expert_idx], config.w1_scales[expert_idx],
            BS, I
        )  # [H, I]
        w2 = _dequantize_matrix(
            config.w2_packed[expert_idx], config.w2_scales[expert_idx],
            BS, H
        )  # [I, H]

        # FFN: SiLU(x @ W1) @ W2
        x_sel = x[token_indices].float()  # [num_selected, H]
        hidden = F.silu(x_sel @ w1)  # [num_selected, I]
        y = hidden @ w2  # [num_selected, H]

        # Weighted accumulate
        gate_vals = gates[token_indices, slot_indices].unsqueeze(-1)  # [num_selected, 1]
        output.index_add_(0, token_indices, y * gate_vals)

    return output.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------

@torch.no_grad()
def _verbose_allclose(received, expected, rtol=1e-05, atol=1e-08, max_print=5):
    if received.shape != expected.shape:
        return False, [f"SIZE MISMATCH. received: {received.shape}, expected: {expected.shape}"]

    diff = torch.abs(received.to(torch.float32) - expected.to(torch.float32))
    tolerance = atol + rtol * torch.abs(expected.to(torch.float32))
    tol_mismatched = diff > tolerance
    nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))
    posinf_mismatched = torch.logical_xor(torch.isposinf(received), torch.isposinf(expected))
    neginf_mismatched = torch.logical_xor(torch.isneginf(received), torch.isneginf(expected))
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched)
    num_mismatched = mismatched.count_nonzero().item()

    if num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        for index in mismatched_indices[:max_print]:
            i = tuple(index.tolist())
            mismatch_details.append(f"ERROR at {i}: received={received[i]} expected={expected[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched.")
        return False, mismatch_details

    return True, [f"Maximum error: {torch.max(diff):.6f}"]


def check_implementation(data, submission_output, rtol=3e-2, atol=3e-2):
    """Check submission output against reference.

    Returns:
        (passed: bool, msg: str)
    """
    if not isinstance(submission_output, torch.Tensor):
        return False, f"Expected torch.Tensor, got {type(submission_output).__name__}"

    config, x = data
    expected_shape = (config.num_tokens, config.hidden_dim)
    if submission_output.shape != expected_shape:
        return False, (f"Shape mismatch: got {tuple(submission_output.shape)}, "
                       f"expected {expected_shape}")

    # Move submission output to CPU and free GPU memory before running reference
    sub_cpu = submission_output.cpu()
    del submission_output
    gc.collect()
    torch.cuda.empty_cache()

    expected = ref_kernel(data)
    exp_cpu = expected.cpu()
    del expected
    gc.collect()
    torch.cuda.empty_cache()

    good, reasons = _verbose_allclose(sub_cpu, exp_cpu, rtol=rtol, atol=atol)
    if not good:
        return False, "Output mismatch: " + " ".join(reasons)

    return True, "Match"


# ---------------------------------------------------------------------------
# Self-contained reference code for remote execution
# ---------------------------------------------------------------------------

MODAL_REFERENCE_CODE = r'''
import math
import gc
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn.functional as F

_FP4_E2M1_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def _fp4_lut(device):
    return torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=device)

def _unpack_nibbles(packed, out_dim):
    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    shape = packed.shape[:-1] + (out_dim,)
    out = torch.empty(shape, dtype=torch.int32, device=packed.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out

def _dequantize_matrix(packed, scales, block_size, out_dim):
    lut = _fp4_lut(packed.device)
    codes = _unpack_nibbles(packed, out_dim)
    values = lut[codes]
    batch_shape = values.shape[:-1]
    num_blocks = out_dim // block_size
    values = values.view(*batch_shape, num_blocks, block_size)
    values = values * scales.unsqueeze(-1)
    return values.view(*batch_shape, out_dim)

def _pack_fp4_codes(codes):
    low = codes[..., 0::2].to(torch.uint8)
    high = codes[..., 1::2].to(torch.uint8)
    return low | (high << 4)

@dataclass
class MoEConfig:
    num_tokens: int
    hidden_dim: int
    intermediate_dim: int
    num_experts: int
    top_k: int
    block_size: int
    router_weight: torch.Tensor
    w1_packed: torch.Tensor
    w1_scales: torch.Tensor
    w2_packed: torch.Tensor
    w2_scales: torch.Tensor

def generate_input(num_tokens, hidden_dim, intermediate_dim, num_experts, top_k, seed):
    block_size = 32
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    router_weight = (torch.randn(
        (num_experts, hidden_dim), dtype=torch.float32, generator=gen, device='cuda'
    ) / math.sqrt(hidden_dim)).to(torch.bfloat16)
    w1_codes = torch.randint(0, 16, (num_experts, hidden_dim, intermediate_dim),
                             generator=gen, device='cuda', dtype=torch.int32)
    w1_packed = _pack_fp4_codes(w1_codes)
    w1_scales = (2.0 ** torch.randint(-4, 5, (num_experts, hidden_dim, intermediate_dim // block_size),
                                       generator=gen, device='cuda', dtype=torch.int32).float())
    w2_codes = torch.randint(0, 16, (num_experts, intermediate_dim, hidden_dim),
                             generator=gen, device='cuda', dtype=torch.int32)
    w2_packed = _pack_fp4_codes(w2_codes)
    w2_scales = (2.0 ** torch.randint(-4, 5, (num_experts, intermediate_dim, hidden_dim // block_size),
                                       generator=gen, device='cuda', dtype=torch.int32).float())
    config = MoEConfig(
        num_tokens=num_tokens, hidden_dim=hidden_dim, intermediate_dim=intermediate_dim,
        num_experts=num_experts, top_k=top_k, block_size=block_size,
        router_weight=router_weight, w1_packed=w1_packed, w1_scales=w1_scales,
        w2_packed=w2_packed, w2_scales=w2_scales,
    )
    x = (torch.randn((num_tokens, hidden_dim), dtype=torch.float32,
                      generator=gen, device='cuda') / math.sqrt(hidden_dim)).to(torch.bfloat16)
    return config, x

@torch.no_grad()
def ref_kernel(data):
    config, x = data
    T, H, I, E, K, BS = (config.num_tokens, config.hidden_dim, config.intermediate_dim,
                          config.num_experts, config.top_k, config.block_size)
    router_logits = x.float() @ config.router_weight.float().T
    topk_scores, topk_experts = torch.topk(router_logits, k=K, dim=-1)
    gates = torch.softmax(topk_scores, dim=-1).float()
    output = torch.zeros(T, H, dtype=torch.float32, device=x.device)
    for expert_idx in range(E):
        mask = (topk_experts == expert_idx)
        if not mask.any():
            continue
        token_indices, slot_indices = torch.where(mask)
        w1 = _dequantize_matrix(config.w1_packed[expert_idx], config.w1_scales[expert_idx], BS, I)
        w2 = _dequantize_matrix(config.w2_packed[expert_idx], config.w2_scales[expert_idx], BS, H)
        x_sel = x[token_indices].float()
        hidden = F.silu(x_sel @ w1)
        y = hidden @ w2
        gate_vals = gates[token_indices, slot_indices].unsqueeze(-1)
        output.index_add_(0, token_indices, y * gate_vals)
    return output.to(torch.bfloat16)

@torch.no_grad()
def _verbose_allclose(received, expected, rtol=1e-05, atol=1e-08, max_print=5):
    if received.shape != expected.shape:
        return False, [f"SIZE MISMATCH. received: {received.shape}, expected: {expected.shape}"]
    diff = torch.abs(received.to(torch.float32) - expected.to(torch.float32))
    tolerance = atol + rtol * torch.abs(expected.to(torch.float32))
    tol_mismatched = diff > tolerance
    nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))
    posinf_mismatched = torch.logical_xor(torch.isposinf(received), torch.isposinf(expected))
    neginf_mismatched = torch.logical_xor(torch.isneginf(received), torch.isneginf(expected))
    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )
    mismatched_indices = torch.nonzero(mismatched)
    num_mismatched = mismatched.count_nonzero().item()
    if num_mismatched >= 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        for index in mismatched_indices[:max_print]:
            i = tuple(index.tolist())
            mismatch_details.append(f"ERROR at {i}: received={received[i]} expected={expected[i]}")
        if num_mismatched > max_print:
            mismatch_details.append(f"... and {num_mismatched - max_print} more mismatched.")
        return False, mismatch_details
    return True, [f"Maximum error: {torch.max(diff):.6f}"]

def check_implementation(data, submission_output, rtol=3e-2, atol=3e-2):
    if not isinstance(submission_output, torch.Tensor):
        return False, f"Expected torch.Tensor, got {type(submission_output).__name__}"
    config, x = data
    expected_shape = (config.num_tokens, config.hidden_dim)
    if submission_output.shape != expected_shape:
        return False, f"Shape mismatch: got {tuple(submission_output.shape)}, expected {expected_shape}"
    sub_cpu = submission_output.cpu()
    del submission_output
    gc.collect()
    torch.cuda.empty_cache()
    expected = ref_kernel(data)
    exp_cpu = expected.cpu()
    del expected
    gc.collect()
    torch.cuda.empty_cache()
    good, reasons = _verbose_allclose(sub_cpu, exp_cpu, rtol=rtol, atol=atol)
    if not good:
        return False, "Output mismatch: " + " ".join(reasons)
    return True, "Match"
'''

# EVOLVE-BLOCK-START
"""
Initial MXFP4 MoE submission — correct but unfused baseline.

Dequantizes expert weights per-expert using a Triton kernel, then runs the MoE
forward pass with standard PyTorch matmuls. Leaves substantial room for:
  - Fused dequant+GEMM kernels
  - Expert-parallel batched execution
  - Tiled LDS-optimized memory access
  - Wavefront64-aware scheduling (AMD cDNA4)
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from reference import MoEConfig

# ---------------------------------------------------------------------------
# OCP MXFP4 (E2M1) lookup table — must match reference.py exactly
# ---------------------------------------------------------------------------
_FP4_E2M1_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

_lut_cache = {}


def _get_lut(device: torch.device) -> torch.Tensor:
    if device not in _lut_cache:
        _lut_cache[device] = torch.tensor(
            _FP4_E2M1_VALUES, dtype=torch.float32, device=device
        )
    return _lut_cache[device]


# ---------------------------------------------------------------------------
# Triton dequantization kernel
# ---------------------------------------------------------------------------

@triton.jit
def _dequant_mxfp4_kernel(
    packed_ptr,   # [rows, cols_packed] uint8
    scale_ptr,    # [rows, num_blocks] float32
    lut_ptr,      # [16] float32
    out_ptr,      # [rows, out_dim] bfloat16
    rows,
    cols_packed,  # = out_dim // 2
    out_dim,
    num_blocks,   # = out_dim // BLOCK_SIZE_FP4
    stride_packed_row,
    stride_packed_col,
    stride_scale_row,
    stride_scale_col,
    stride_out_row,
    stride_out_col,
    BLOCK_SIZE_FP4: tl.constexpr,  # 32 (OCP MX block size)
    BLOCK_M: tl.constexpr,         # rows per program
    BLOCK_N: tl.constexpr,         # output cols per program (must be multiple of BLOCK_SIZE_FP4)
):
    """Unpack MXFP4 codes, look up values, apply per-block scales, store bf16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row range for this tile
    row_start = pid_m * BLOCK_M
    row_offs = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offs < rows

    # Column range in output space
    col_start = pid_n * BLOCK_N
    col_offs = col_start + tl.arange(0, BLOCK_N)
    col_mask = col_offs < out_dim

    # Corresponding packed byte columns (2 FP4 values per byte)
    # Element 2i -> byte i low nibble, element 2i+1 -> byte i high nibble
    byte_cols = col_offs // 2  # [BLOCK_N]
    is_high = (col_offs % 2)   # 0 = low nibble, 1 = high nibble

    # Load packed bytes
    packed_offs = row_offs[:, None] * stride_packed_row + byte_cols[None, :] * stride_packed_col
    combined_mask = row_mask[:, None] & col_mask[None, :]
    packed_bytes = tl.load(packed_ptr + packed_offs, mask=combined_mask, other=0)
    packed_int = packed_bytes.to(tl.int32)

    # Extract nibble codes
    low_codes = packed_int & 0x0F
    high_codes = (packed_int >> 4) & 0x0F
    codes = tl.where(is_high[None, :] == 1, high_codes, low_codes)

    # Lookup FP4 values
    values = tl.load(lut_ptr + codes)  # [BLOCK_M, BLOCK_N] float32

    # Load per-block scales
    block_idx = col_offs // BLOCK_SIZE_FP4  # [BLOCK_N]
    scale_offs = row_offs[:, None] * stride_scale_row + block_idx[None, :] * stride_scale_col
    block_mask = row_mask[:, None] & (block_idx[None, :] < num_blocks)
    scales = tl.load(scale_ptr + scale_offs, mask=block_mask, other=1.0)

    # Dequantized = value * scale
    result = values * scales

    # Store as bfloat16
    out_offs = row_offs[:, None] * stride_out_row + col_offs[None, :] * stride_out_col
    tl.store(out_ptr + out_offs, result.to(tl.bfloat16), mask=combined_mask)


def _dequantize_expert(packed: torch.Tensor, scales: torch.Tensor,
                       out_dim: int, block_size: int = 32) -> torch.Tensor:
    """Dequantize a single expert's weight matrix using Triton.

    Args:
        packed: [rows, out_dim // 2] uint8
        scales: [rows, out_dim // block_size] float32
        out_dim: number of output columns
        block_size: MX block size (32)

    Returns:
        [rows, out_dim] bfloat16
    """
    rows = packed.shape[0]
    cols_packed = packed.shape[1]
    num_blocks = out_dim // block_size
    device = packed.device

    out = torch.empty((rows, out_dim), dtype=torch.bfloat16, device=device)
    lut = _get_lut(device)

    BLOCK_M = min(16, rows)
    BLOCK_N = min(128, out_dim)
    # Ensure BLOCK_N is a power of 2 and multiple of block_size
    BLOCK_N = max(block_size, 1 << (BLOCK_N - 1).bit_length())
    BLOCK_N = min(BLOCK_N, out_dim)

    grid = (
        (rows + BLOCK_M - 1) // BLOCK_M,
        (out_dim + BLOCK_N - 1) // BLOCK_N,
    )

    _dequant_mxfp4_kernel[grid](
        packed, scales, lut, out,
        rows, cols_packed, out_dim, num_blocks,
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_FP4=block_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def custom_kernel(data: Tuple[MoEConfig, torch.Tensor]) -> torch.Tensor:
    """MXFP4 MoE forward pass — baseline implementation.

    Args:
        data: Tuple of (config: MoEConfig, x: torch.Tensor)
              x is [num_tokens, hidden_dim] bfloat16

    Returns:
        output: [num_tokens, hidden_dim] bfloat16
    """
    config, x = data

    T = config.num_tokens
    H = config.hidden_dim
    I = config.intermediate_dim
    E = config.num_experts
    K = config.top_k
    BS = config.block_size

    # Step 1: Router — compute expert affinities, select top-k
    router_logits = x.float() @ config.router_weight.float().T  # [T, E]
    topk_scores, topk_experts = torch.topk(router_logits, k=K, dim=-1)  # [T, K]
    gates = torch.softmax(topk_scores, dim=-1).float()  # [T, K]

    # Step 2: Group tokens by expert, run dequant + FFN
    output = torch.zeros(T, H, dtype=torch.float32, device=x.device)

    for expert_idx in range(E):
        # Find which (token, slot) pairs selected this expert
        mask = (topk_experts == expert_idx)  # [T, K]
        if not mask.any():
            continue

        token_indices, slot_indices = torch.where(mask)

        # Dequantize this expert's weights using Triton kernel
        w1 = _dequantize_expert(
            config.w1_packed[expert_idx], config.w1_scales[expert_idx], I, BS
        )  # [H, I] bf16
        w2 = _dequantize_expert(
            config.w2_packed[expert_idx], config.w2_scales[expert_idx], H, BS
        )  # [I, H] bf16

        # FFN: SiLU(x @ W1) @ W2
        x_sel = x[token_indices]  # [num_selected, H] bf16
        hidden = F.silu(x_sel.float() @ w1.float())  # [num_selected, I]
        y = hidden @ w2.float()  # [num_selected, H]

        # Weighted accumulate
        gate_vals = gates[token_indices, slot_indices].unsqueeze(-1)
        output.index_add_(0, token_indices, y * gate_vals)

    return output.to(torch.bfloat16)
# EVOLVE-BLOCK-END

"""Utilities for handling TPU FP4 packed weights and quantization metadata."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

from .fp4_utils import (
    FP4_LUT,
    fp4_indices_from_values,
    pack_fp4_from_fp32,
    unpack_fp4,
)
from .mxfp4_utils import MXFP4_BLOCK_SIZE
from vllm.model_executor.layers.quantization import (  # type: ignore
    QUANTIZATION_METHODS,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (  # type: ignore
    QuantizationConfig,
)

# TPU FP4 constants
TPU_FP4_SUBCHANNEL_SIZE: int = 256
TPU_FP4_BYTES_PER_SUBCHANNEL: int = TPU_FP4_SUBCHANNEL_SIZE // 2
TPU_FP4_QUANT_METHOD: str = "tpu_fp4"
TPU_FP4_BLOCKS_PER_SUBCHANNEL: int = TPU_FP4_SUBCHANNEL_SIZE // MXFP4_BLOCK_SIZE


FP4_MIN_VALUE: float = FP4_LUT.min().item()
FP4_MAX_VALUE: float = FP4_LUT.max().item()

LOGGER = logging.getLogger(__name__)

class TpuFp4Config(QuantizationConfig):
    """Minimal QuantizationConfig so vLLM recognizes TPU FP4 weights."""

    @classmethod
    def get_name(cls) -> str:
        return TPU_FP4_QUANT_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TpuFp4Config":
        inst = cls()
        inst.fmt = str(config.get("fmt", "")).lower()
        inst.weight_block_size = config.get("weight_block_size")
        return inst

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return None


def _validate_blocks(blocks_u8: torch.Tensor) -> None:
    if blocks_u8.dtype != torch.uint8:
        raise ValueError(
            f"Expected TPU FP4 blocks to be uint8, got {blocks_u8.dtype}")
    if blocks_u8.shape[-1] % 2 != 0:
        raise ValueError("TPU FP4 blocks must have an even number of packed nibbles; "
                         f"got last dimension {blocks_u8.shape[-1]}")


def _validate_scales(scales: torch.Tensor) -> None:
    if scales.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            "TPU FP4 scales must be bf16 or fp32; got " f"{scales.dtype}")


def dequant_tpu_fp4_to_bf16(blocks_u8: torch.Tensor,
                            scales: torch.Tensor) -> torch.Tensor:
    """Dequantize TPU FP4 packed weights into bfloat16 tensors.

    TPU checkpoints keep the logical axis flattened instead of reshaping to
    explicit 256-wide subchannels (unlike MXFP4's 32). This avoids padding
    overhead when the reducation dim is not divisible by 256. Since 32 is much
    smaller than 256, this situation occurs with more models than MXFP4.
    """
    _validate_blocks(blocks_u8)
    _validate_scales(scales)

    fp4_vals = unpack_fp4(blocks_u8)
    float_scales = scales.to(torch.float32)

    if fp4_vals.ndim != float_scales.ndim:
        raise ValueError(
            "TPU FP4 blocks and scales must have the same rank: blocks "
            f"{fp4_vals.shape} vs scales {float_scales.shape}")

    # Ensure scale prefix dims can broadcast to the decoded codes prefix dims.
    for idx, (code_dim, scale_dim) in enumerate(
            zip(fp4_vals.shape[:-1], float_scales.shape[:-1])):
        if scale_dim not in (1, code_dim):
            raise ValueError(
                "TPU FP4 scale prefix dim is not broadcastable to codes: "
                f"dim {idx}, blocks {code_dim}, scales {scale_dim}")

    # Expand scales to match the decoded prefix dims before repeating along the
    # implicit 256-wide subchannel axis that remains flattened in checkpoints.
    expanded_prefix = (*fp4_vals.shape[:-1], float_scales.shape[-1])
    float_scales = float_scales.expand(expanded_prefix)

    expanded_scales = torch.repeat_interleave(float_scales,
                                              TPU_FP4_SUBCHANNEL_SIZE,
                                              dim=-1)
    if expanded_scales.shape[-1] < fp4_vals.shape[-1]:
        raise ValueError(
            "TPU FP4 scales do not cover the decoded values: "
            f"{expanded_scales.shape[-1]} < {fp4_vals.shape[-1]}")

    expanded_scales = expanded_scales[..., :fp4_vals.shape[-1]]
    return (fp4_vals * expanded_scales).to(torch.bfloat16)


def unpack_tpu_fp4_to_fp32(
    blocks_u8: torch.Tensor,
    scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode TPU FP4 blocks and scales to float32 tensors suitable for QArray."""
    _validate_blocks(blocks_u8)
    _validate_scales(scales)

    fp4_vals = unpack_fp4(blocks_u8)
    float_scales = scales.to(torch.float32)

    if fp4_vals.ndim != float_scales.ndim:
        raise ValueError(
            "TPU FP4 blocks and scales must have the same rank: blocks "
            f"{fp4_vals.shape} vs scales {float_scales.shape}")


    return fp4_vals, float_scales


def pack_tpu_fp4_from_fp32(codes_fp32: torch.Tensor) -> torch.Tensor:
    """Pack float32 FP4 codes into TPU FP4 uint8 blocks."""
    return pack_fp4_from_fp32(codes_fp32)


def merge_fp4_exponents(
    fp4_values: torch.Tensor,
    block_scales: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge FP4 blocks into wider subchannels row by row.

    Args:
        fp4_values: Tensor of FP4 LUT values, shape (rows, num_blocks, block_size).
        block_scales: Per-block scales, shape (rows, num_blocks).
        block_size: Elements per original FP4 block (must equal MXFP4_BLOCK_SIZE).

    Returns:
        merged_fp4: Quantized FP4 values with shared subchannel scales, same shape as fp4_values.
        merged_block_scales: Block-wise scales after merging, same shape as block_scales.
        merged_subchannel_scales: Per-subchannel scales (rows, num_blocks // blocks_per_subchannel).
    """

    rows, num_blocks, last_dim = fp4_values.shape
    if last_dim != block_size:
        raise ValueError("Last dimension must equal block_size")
    if block_scales.shape != (rows, num_blocks):
        raise ValueError("block_scales must match fp4_values on prefix dims")

    if block_size != MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"TPU FP4 merge expects block_size={MXFP4_BLOCK_SIZE}; got {block_size}")
    blocks_per_subchannel = TPU_FP4_BLOCKS_PER_SUBCHANNEL
    if num_blocks % blocks_per_subchannel != 0:
        raise ValueError("num_blocks must be divisible by blocks_per_subchannel")

    num_subchannels = num_blocks // blocks_per_subchannel

    values_grouped = fp4_values.view(rows, num_subchannels, blocks_per_subchannel, block_size)
    scales_grouped = block_scales.view(rows, num_subchannels, blocks_per_subchannel)

    ratio = (
        (scales_grouped.unsqueeze(-1) / scales_grouped.unsqueeze(-2))
        .unsqueeze(-1)
    )
    adjusted = values_grouped.unsqueeze(-2) * ratio

    indices = fp4_indices_from_values(adjusted.reshape(-1).to(torch.float32))
    quantized = FP4_LUT[indices.long()].reshape_as(adjusted)
    recon = quantized * scales_grouped.unsqueeze(-2).unsqueeze(-1)

    original = values_grouped * scales_grouped.unsqueeze(-1)
    original = original.unsqueeze(-2)

    diff = recon - original
    block_dim = blocks_per_subchannel * block_size
    err = diff.square().sum(dim=(2, 4)) / block_dim
    best_idx = torch.argmin(err, dim=-1)

    best_scales = torch.take_along_dim(
        scales_grouped,
        best_idx.unsqueeze(-1),
        dim=-1,
    ).squeeze(-1)

    gather_idx = (
        best_idx.unsqueeze(-1)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, blocks_per_subchannel, 1, block_size)
    )
    best_quantized = torch.take_along_dim(quantized, gather_idx, dim=3).squeeze(3)

    block_scales_out = best_scales.unsqueeze(-1).expand(-1, -1, blocks_per_subchannel)

    merged_fp4 = best_quantized.reshape(rows, num_blocks, block_size)
    merged_block_scales = block_scales_out.reshape(rows, num_blocks)
    merged_subchannel_scales = best_scales
    return merged_fp4, merged_block_scales, merged_subchannel_scales


def ensure_tpu_fp4_registered() -> None:
    """Register the TPU FP4 quantization config with vLLM if available."""
    if TPU_FP4_QUANT_METHOD in QUANTIZATION_METHODS:
        _extend_tpu_supported_quantization()
        return
    register_quantization_config(TPU_FP4_QUANT_METHOD)(TpuFp4Config)
    _extend_tpu_supported_quantization()


def _extend_tpu_supported_quantization() -> None:
    from tpu_inference.platforms import TpuPlatform  # type: ignore

    methods = getattr(TpuPlatform, "supported_quantization", None)
    if isinstance(methods, list) and TPU_FP4_QUANT_METHOD not in methods:
        methods.append(TPU_FP4_QUANT_METHOD)

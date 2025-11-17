"""Quantize a GPT-OSS checkpoint to TPU FP4 and export packed safetensors."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.config import ModelConfig, VllmConfig
from tqdm import tqdm

from huggingface_hub import snapshot_download

from tpu_inference.models.common.model_loader import apply_qwix_quantization
from tpu_inference.models.jax.gpt_oss import GptOss
from tpu_inference.models.jax.utils.quantization.mxfp4_utils import (
    MXFP4_BLOCK_SIZE,
    MXFP4_QUANT_METHOD,
    dequant_mxfp4_to_bf16,
    unpack_mxfp4_to_fp32,
)
from tpu_inference.models.jax.utils.quantization.quantization_utils import (
    DEFAULT_GPT_OSS_TPU_FP4_CONFIG,
    update_vllm_config_for_qwix_quantization,
)
from tpu_inference.models.jax.utils.quantization.tpu_fp4_utils import (
    TPU_FP4_SUBCHANNEL_SIZE,
    dequant_tpu_fp4_to_bf16,
    merge_fp4_exponents,
    pack_tpu_fp4_from_fp32,
)
from tpu_inference.models.jax.utils.weight_utils import get_param
from tpu_inference.models.jax.utils import file_utils

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s")

COMPARISON_TOLERANCE = 1e-6

# JAX path to HF tensor mapping for packed weights.
_PACK_TARGETS = {
    "layers.{layer}.custom_module.mlp1_weight_EDF2":
    "model.layers.{layer}.mlp.experts.gate_up_proj",
    "layers.{layer}.custom_module.mlp2_weight_EFD":
    "model.layers.{layer}.mlp.experts.down_proj",
}


class QuantizationError(RuntimeError):
    pass


def _log_merge_comparison(base_key: str, original_blocks: torch.Tensor,
                          original_scales: torch.Tensor,
                          merged_blocks: torch.Tensor,
                          merged_scales: torch.Tensor) -> None:
    """Dequantize tensors and log comparison metrics."""
    original = dequant_mxfp4_to_bf16(original_blocks, original_scales)
    merged = dequant_tpu_fp4_to_bf16(merged_blocks, merged_scales)

    if original.ndim >= 2:
        original = original.swapaxes(-1, -2)
    if merged.ndim >= 2:
        merged = merged.swapaxes(-1, -2)

    if original.shape != merged.shape:
        LOGGER.warning("Skipping diff for %s due to shape mismatch %s vs %s",
                       base_key, original.shape, merged.shape)
        return

    if original.numel() == 0:
        LOGGER.warning("Skipping diff for %s due to empty tensor", base_key)
        return

    original32 = original.to(torch.float32)
    merged32 = merged.to(torch.float32)
    diff = merged32 - original32
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    max_abs = diff.abs().max().item()
    changed = torch.count_nonzero(diff.abs() > COMPARISON_TOLERANCE).item()
    total = diff.numel()
    pct_changed = 100.0 * changed / total
    message = (
        f"Merge diff {base_key}: rmse={rmse:.6e} max_abs={max_abs:.6e} "
        f"changed={changed}/{total} ({pct_changed:.4f}%)"
    )
    tqdm.write(message)
    LOGGER.info(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize GPT-OSS experts to TPU FP4 and emit safetensors")
    parser.add_argument("source",
                        type=Path,
                        help="Path to the source Hugging Face checkpoint")
    parser.add_argument("output",
                        type=Path,
                        help="Directory to write the quantized checkpoint")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="PRNG seed for deterministic operations")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Allow overwriting an existing output directory")
    parser.add_argument("--merge-exponents",
                        action="store_true",
                        help="Merge MXFP4 exponent groups instead of running Qwix quantization")
    return parser.parse_args()


def _prepare_output_dir(source: Path, output: Path, overwrite: bool) -> Path:
    if source.exists():
        source_root = source
    else:
        repo_id = str(source)
        if file_utils.is_hf_repo(repo_id):
            LOGGER.info("Downloading model snapshot for %s", repo_id)
            source_root = Path(snapshot_download(repo_id))
        else:
            raise FileNotFoundError(f"Source checkpoint not found: {source}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output} already exists; use --overwrite")
        shutil.rmtree(output)
    shutil.copytree(source_root, output)
    return output


def _build_vllm_config(model_dir: Path) -> VllmConfig:
    model_config = ModelConfig(model=str(model_dir),
                               trust_remote_code=True,
                               dtype="bfloat16")
    hf_config = model_config.hf_config
    quant_method = None
    if hasattr(hf_config, "quantization_config") and \
            hf_config.quantization_config is not None:  # type: ignore[attr-defined]
        quant_method = hf_config.quantization_config.get(  # type: ignore[attr-defined]
            "quant_method")
    if quant_method:
        model_config.quantization = quant_method  # type: ignore[assignment]
    model_config.hf_config = hf_config

    vllm_config = VllmConfig(model_config=model_config)
    vllm_config.load_config.download_dir = str(model_dir)
    update_vllm_config_for_qwix_quantization(vllm_config)
    if "quantization" not in vllm_config.additional_config:
        vllm_config.additional_config["quantization"] = (
            DEFAULT_GPT_OSS_TPU_FP4_CONFIG)
    return vllm_config


def _create_mesh() -> Mesh:
    devices = jax.devices()
    if not devices:
        raise QuantizationError("No JAX devices available for quantization")
    mesh_array = np.array(devices).reshape(1, len(devices))
    return Mesh(mesh_array, ("data", "model"))


def _apply_quantization(vllm_config: VllmConfig, mesh: Mesh,
                        seed: int) -> GptOss:
    rng = jax.random.PRNGKey(seed)
    model = GptOss(vllm_config, rng, mesh)
    with mesh:
        model.load_weights(rng)
        model = apply_qwix_quantization(vllm_config,
                                        model,
                                        rng,
                                        mesh,
                                        apply_to_abstract_model=False)
    return model


def _to_host_float32(array: jax.Array) -> np.ndarray:
    host = jnp.asarray(array, dtype=jnp.float32)
    return np.asarray(jax.device_get(host))


def _collect_packed_weights(model: GptOss) -> Dict[str, Tuple[torch.Tensor,
                                                             torch.Tensor]]:
    params = nnx.state(model)
    replacements: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    num_layers = model.hf_config.num_hidden_layers
    layer_iter = tqdm(range(num_layers),
                      desc="Collecting quantized layers",
                      leave=False)
    for layer in layer_iter:
        for jax_path_tpl, hf_tpl in _PACK_TARGETS.items():
            jax_path = jax_path_tpl.format(layer=layer)
            hf_key = hf_tpl.format(layer=layer)
            param = get_param(params, jax_path)
            if not hasattr(param, "array"):
                raise QuantizationError(f"Expected quantized array at {jax_path}")
            q_array = param.array
            codes_fp32 = np.array(_to_host_float32(q_array.qvalue.value),
                                   copy=True)
            scales_fp32 = np.array(_to_host_float32(q_array.scale.value),
                                    copy=True)

            codes_tensor = torch.from_numpy(codes_fp32).to(torch.float32)
            scales_tensor = torch.from_numpy(scales_fp32).to(torch.bfloat16)

            # Reorder to (experts, hidden, channels)
            codes_tensor = codes_tensor.permute(0, 2, 1).contiguous()
            scales_tensor = scales_tensor.permute(0, 2, 1).contiguous()

            blocks_tensor = pack_tpu_fp4_from_fp32(codes_tensor).contiguous()
            replacements[hf_key] = (blocks_tensor, scales_tensor)
    return replacements


def _load_mxfp4_tensor(model_dir: Path,
                       base_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
    blocks_name = f"{base_key}_blocks"
    scales_name = f"{base_key}_scales"
    for st_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(st_path, framework="pt") as handle:
            keys = set(handle.keys())
            if blocks_name not in keys or scales_name not in keys:
                continue
            LOGGER.debug("Loaded MXFP4 tensor %s from %s", base_key, st_path.name)
            return handle.get_tensor(blocks_name), handle.get_tensor(scales_name)
    raise QuantizationError(
        f"Failed to locate MXFP4 tensors for {base_key} in {model_dir}")


def _merge_mxfp4_tensor(blocks: torch.Tensor,
                        scales: torch.Tensor) -> Tuple[torch.Tensor,
                                                       torch.Tensor]:
    fp4_values, scales_fp32 = unpack_mxfp4_to_fp32(blocks, scales)
    prefix_shape = fp4_values.shape[:-2]
    num_blocks = fp4_values.shape[-2]
    block_size = fp4_values.shape[-1]

    blocks_per_subchannel = TPU_FP4_SUBCHANNEL_SIZE // MXFP4_BLOCK_SIZE
    pad_blocks = (-num_blocks) % blocks_per_subchannel
    if pad_blocks > 0:
        pad_fp4 = torch.zeros((*prefix_shape, pad_blocks, block_size),
                              dtype=fp4_values.dtype,
                              device=fp4_values.device)
        pad_scales = torch.ones((*prefix_shape, pad_blocks),
                                dtype=scales_fp32.dtype,
                                device=scales_fp32.device)
        fp4_values = torch.cat([fp4_values, pad_fp4], dim=-2)
        scales_fp32 = torch.cat([scales_fp32, pad_scales], dim=-1)

    num_blocks_padded = fp4_values.shape[-2]
    flat_values = fp4_values.reshape(-1, num_blocks_padded, block_size)
    flat_scales = scales_fp32.reshape(-1, num_blocks_padded)

    merged_fp4, merged_block_scales, merged_subchannel_scales = (
        merge_fp4_exponents(flat_values, flat_scales, MXFP4_BLOCK_SIZE))

    merged_fp4 = merged_fp4.view(*prefix_shape, num_blocks_padded, block_size)
    merged_block_scales = merged_block_scales.view(*prefix_shape,
                                                   num_blocks_padded)
    num_subchannels = ((num_blocks + blocks_per_subchannel - 1) //
                       blocks_per_subchannel)
    merged_subchannel_scales = merged_subchannel_scales.view(
        *prefix_shape, num_subchannels)

    if pad_blocks > 0:
        merged_fp4 = merged_fp4[..., :-pad_blocks, :]
        merged_block_scales = merged_block_scales[..., :-pad_blocks]

    codes_tensor = merged_fp4.reshape(*prefix_shape,
                                      num_blocks * MXFP4_BLOCK_SIZE)
    blocks_tensor = pack_tpu_fp4_from_fp32(codes_tensor.to(torch.float32))
    scales_tensor = merged_subchannel_scales.to(torch.bfloat16)
    return blocks_tensor.contiguous(), scales_tensor.contiguous()


def _collect_merged_weights(model_dir: Path) -> Dict[str, Tuple[torch.Tensor,
                                                                torch.Tensor]]:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    quant_method = (config.get("quantization_config", {}) or {}).get(
        "quant_method")
    if quant_method != MXFP4_QUANT_METHOD:
        raise QuantizationError(
            "Merge mode requires an MXFP4 checkpoint; found "
            f"quant_method={quant_method!r}")
    num_layers = config.get("num_hidden_layers")
    if num_layers is None:
        raise QuantizationError("config.json missing num_hidden_layers")

    LOGGER.info("Merging MXFP4 exponent groups into TPU FP4 subchannels")
    replacements: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    layer_iter = tqdm(range(num_layers),
                      desc="Merging layers",
                      leave=False)
    for layer in layer_iter:
        for hf_tpl in _PACK_TARGETS.values():
            base_key = hf_tpl.format(layer=layer)
            LOGGER.debug("Merging tensor %s", base_key)
            blocks, scales = _load_mxfp4_tensor(model_dir, base_key)
            merged_blocks, merged_scales = _merge_mxfp4_tensor(blocks, scales)
            _log_merge_comparison(base_key, blocks, scales, merged_blocks,
                                   merged_scales)
            replacements[base_key] = (merged_blocks, merged_scales)
    return replacements


def _rewrite_safetensors(model_dir: Path,
                         replacements: Mapping[str, Tuple[torch.Tensor,
                                                          torch.Tensor]]) -> None:
    pending = set(replacements.keys())
    for st_path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(st_path, framework="pt") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata()
            tensors = {name: handle.get_tensor(name) for name in keys}

        local_updates: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        new_tensors = {}
        for name, tensor in tensors.items():
            base_key = None
            if name in replacements:
                base_key = name
            elif name.endswith("_blocks") or name.endswith("_scales"):
                candidate = name[:-7]
                if candidate in replacements:
                    base_key = candidate
            if base_key is None:
                new_tensors[name] = tensor
            else:
                local_updates[base_key] = replacements[base_key]
        if not local_updates:
            continue
        for base_key, bundle in local_updates.items():
            blocks, scales = bundle
            new_tensors[f"{base_key}_blocks"] = blocks.cpu()
            new_tensors[f"{base_key}_scales"] = scales.cpu()
            pending.discard(base_key)
        save_file(new_tensors, st_path, metadata=metadata)
        LOGGER.info("Updated %s", st_path)
    if pending:
        missing = ", ".join(sorted(pending))
        raise QuantizationError(
            f"Failed to rewrite tensors for keys: {missing}")


def _update_config_json(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    quant_config = config.get("quantization_config", {})
    quant_config.update({
        "quant_method": "tpu_fp4",
        "weight_block_size": [1, TPU_FP4_SUBCHANNEL_SIZE],
    })
    config["quantization_config"] = quant_config
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = _parse_args()
    output_dir = _prepare_output_dir(args.source, args.output, args.overwrite)
    if args.merge_exponents:
        replacements = _collect_merged_weights(output_dir)
    else:
        vllm_config = _build_vllm_config(output_dir)
        mesh = _create_mesh()
        model = _apply_quantization(vllm_config, mesh, args.seed)
        replacements = _collect_packed_weights(model)
    _rewrite_safetensors(output_dir, replacements)
    _update_config_json(output_dir)
    LOGGER.info("Quantized checkpoint written to %s", output_dir)


if __name__ == "__main__":
    main()

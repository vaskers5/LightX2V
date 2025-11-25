from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.utils.envs import GET_DTYPE

FP8_SCALE_SUFFIX = "__fp8_scale"


def _reshape_scale(scale: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.view(*([1] * len(target_shape)))
    if scale.numel() == target_shape[0]:
        broadcast_shape = (target_shape[0],) + (1,) * (len(target_shape) - 1)
        return scale.view(broadcast_shape)
    if scale.shape == target_shape:
        return scale
    raise ValueError(
        f"Unable to broadcast scale with shape {tuple(scale.shape)} to target shape {tuple(target_shape)}"
    )


def _dequantize_fp8_tensor(quant_tensor: torch.Tensor, scale_tensor: torch.Tensor) -> torch.Tensor:
    if quant_tensor.dtype != torch.uint8:
        logger.warning(
            f"Expected uint8 tensor for FP8 payload, got {quant_tensor.dtype}. Returning original tensor."
        )
        return quant_tensor

    if not torch.cuda.is_available():
        raise RuntimeError("FP8 LoRA checkpoint detected but CUDA is not available for dequantization.")

    device = torch.device("cuda")
    fp8_tensor = quant_tensor.to(device, non_blocking=True).view(torch.float8_e4m3fn)
    scale_tensor = scale_tensor.to(device, dtype=torch.float32, non_blocking=True)
    scale_tensor = _reshape_scale(scale_tensor, fp8_tensor.shape)
    dequant = fp8_tensor.to(torch.float32) * scale_tensor
    return dequant.cpu()


def load_lora_safetensors(file_path: str | Path, target_dtype: torch.dtype | None = None) -> Dict[str, torch.Tensor]:
    """Load LoRA tensors, transparently handling FP8-quantized entries."""

    tensors: Dict[str, Dict[str, torch.Tensor]] = {}
    with safe_open(file_path, framework="pt") as handle:
        keys = list(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            if key.endswith(FP8_SCALE_SUFFIX):
                base_key = key[: -len(FP8_SCALE_SUFFIX)]
                tensors.setdefault(base_key, {})["scale"] = tensor
            else:
                tensors.setdefault(key, {})["payload"] = tensor

    output: Dict[str, torch.Tensor] = {}
    for key, bundle in tensors.items():
        payload = bundle.get("payload")
        scale = bundle.get("scale")
        if payload is None:
            logger.warning(f"Missing payload tensor for {key} in {file_path}, skipping entry")
            continue

        if scale is None:
            tensor = payload
        else:
            tensor = _dequantize_fp8_tensor(payload, scale)

        desired_dtype = target_dtype or GET_DTYPE()
        output[key] = tensor.to(desired_dtype)

    return output

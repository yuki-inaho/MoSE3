"""Precision helpers: pick a dtype that fits the GPU and cast the model accordingly.

The released weights are float32 (about 6 GB), which does not fit on 8 GB GPUs such as the
RTX 2070. On those GPUs ``--dtype auto`` casts the model to float16 (Turing has native fp16 but
no flash-attention and no native bf16) and runs the forward pass under fp16 autocast.
"""
from __future__ import annotations

import torch

_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

# the small per-point heads kept in fp32: the SE(3) fit downstream is sensitive to their output
_FP32_HEADS = {
    "base": ("camera_head", "point_head", "conf_head"),
    "": ("track_head", "vis_head", "rigid_head"),
}


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """``auto`` -> bf16 autocast + fp32 weights on Ampere+ (as upstream), fp16 weights on older
    CUDA GPUs such as the RTX 2070, fp32 on CPU."""
    if name == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    if name not in _ALIASES:
        raise ValueError(f"unknown dtype {name!r}; choose from ['auto', {', '.join(sorted(_ALIASES))}]")
    return _ALIASES[name]


def prepare_model(model, device: torch.device, dtype_name: str = "auto"):
    """Move the model to ``device`` and cast it to the resolved dtype (fp32 = untouched).

    Returns ``(model, autocast_dtype)``: the dtype to autocast the forward pass with.
    """
    dtype = resolve_dtype(dtype_name, device)
    if dtype == torch.float32:
        return model.to(device), torch.bfloat16  # keep the upstream autocast policy for fp32 weights
    model = model.to(dtype).to(device)  # cast on CPU first: fp32 + fp16 never coexist on the GPU
    for owner_name, head_names in _FP32_HEADS.items():
        owner = model if not owner_name else getattr(model, owner_name)
        for head_name in head_names:
            head = getattr(owner, head_name, None)
            if head is not None:
                head.to(torch.float32)
    return model, dtype

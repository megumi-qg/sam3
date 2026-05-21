# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
"""
LoRA (Low-Rank Adaptation) for efficient fine-tuning of SAM3.

Component -> module path prefix in Sam3Image:
- Vision Encoder (ViT): backbone.vision_backbone.trunk
- Text Encoder: backbone.language_backbone
- Geometry Encoder: geometry_encoder
- DETR Encoder: transformer.encoder
- DETR Decoder: transformer.decoder
- Mask Decoder: segmentation_head
"""

from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default target prefixes for each component (relative to Sam3Image root)
LORA_COMPONENT_PREFIXES = {
    "vision_encoder": "backbone.vision_backbone.trunk",
    "text_encoder": "backbone.language_backbone",
    "geometry_encoder": "geometry_encoder",
    "detr_encoder": "transformer.encoder",
    "detr_decoder": "transformer.decoder",
    "mask_decoder": "segmentation_head",
    "dot_prod_scoring": "dot_prod_scoring",
}


class LoRALinear(nn.Module):
    """
    Wraps nn.Linear with low-rank adaptation:
    delta_W = (lora_alpha/r) * B @ A, output = linear(x) + x @ delta_W^T.
    A: (in_features, r), B: (r, out_features).
    
    Exposes weight and bias properties for compatibility with modules that
    directly access these attributes (e.g., MultiheadAttention).
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        in_features = linear.in_features
        out_features = linear.out_features
        self.linear = linear
        self.linear.requires_grad_(False)

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # A (in_features, r), B (r, out_features) => x @ A @ B = (x @ A) @ B
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        """Expose original linear weight for compatibility."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose original linear bias for compatibility."""
        return self.linear.bias

    @property
    def in_features(self):
        """Expose in_features for compatibility."""
        return self.linear.in_features

    @property
    def out_features(self):
        """Expose out_features for compatibility."""
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        compute_dtype = self.linear.weight.dtype
        if x.dtype != compute_dtype:
            x = x.to(compute_dtype)

        out = self.linear(x)
        # x @ A @ B  with A (in,r), B (r,out) => (x @ A) @ B
        lora_a = self.lora_A if self.lora_A.dtype == x.dtype else self.lora_A.to(x.dtype)
        lora_b = self.lora_B if self.lora_B.dtype == x.dtype else self.lora_B.to(x.dtype)
        lora_out = (x @ lora_a @ lora_b) * self.scaling
        out = out + lora_out

        if out.dtype != input_dtype:
            out = out.to(input_dtype)
        return out


class DTypeSafeLinear(nn.Module):
    """
    Wrap nn.Linear and preserve the old LoRA wrapper's dtype behavior.

    Some SAM3 inference paths feed bfloat16 activations into modules whose
    weights stay in float32. A bare nn.Linear would then raise a dtype mismatch.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        weight_dtype = self.linear.weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(weight_dtype)
        out = self.linear(x)
        if out.dtype != input_dtype:
            out = out.to(input_dtype)
        return out


def _apply_lora_to_module(
    module: nn.Module,
    prefix: str,
    target_prefixes: Set[str],
    r: int,
    lora_alpha: float,
    replacements: List[tuple],
) -> None:
    """Recursively replace nn.Linear under target_prefixes with LoRALinear."""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if any(full_name.startswith(p) for p in target_prefixes):
                lora_linear = LoRALinear(child, r=r, lora_alpha=lora_alpha)
                setattr(module, name, lora_linear)
                replacements.append((full_name, lora_linear))
        else:
            _apply_lora_to_module(
                child, full_name, target_prefixes, r, lora_alpha, replacements
            )


def apply_lora_to_sam3(
    model: nn.Module,
    target_components: Optional[List[str]] = None,
    target_prefixes: Optional[List[str]] = None,
    r: int = 8,
    lora_alpha: float = 16.0,
    freeze_non_lora: bool = True,
    unfreeze_components: Optional[List[str]] = None,
) -> List[str]:
    """
    Inject LoRA into the given SAM3 (or Sam3Image) model.

    Args:
        model: The Sam3Image (or detector) model.
        target_components: High-level component names, e.g.
            ["vision_encoder", "text_encoder", "geometry_encoder",
             "detr_encoder", "detr_decoder", "mask_decoder"].
            Used only when target_prefixes is None.
        target_prefixes: Explicit module name prefixes (e.g. "backbone.vision_backbone.trunk").
            If provided, target_components is ignored.
        r: LoRA rank.
        lora_alpha: LoRA scaling (effective scale = lora_alpha / r).
        freeze_non_lora: If True, set requires_grad=False for all non-LoRA parameters
            so that only LoRA params are trained.
        unfreeze_components: List of component names that should NOT be frozen even when
            freeze_non_lora=True. These components will be fully fine-tuned.
            E.g., ["mask_decoder"] to fully fine-tune mask_decoder while using LoRA for others.

    Returns:
        List of replaced module path names (for logging / param group selection).
    """
    if target_prefixes is None:
        if target_components is None:
            target_components = list(LORA_COMPONENT_PREFIXES.keys())
        target_prefixes = [
            LORA_COMPONENT_PREFIXES[c] for c in target_components if c in LORA_COMPONENT_PREFIXES
        ]
    if not target_prefixes:
        return []
    targets = set(target_prefixes)
    replacements = []
    _apply_lora_to_module(model, "", targets, r, lora_alpha, replacements)
    replaced_paths = [path for path, _ in replacements]

    if freeze_non_lora:
        # Get prefixes for components that should not be frozen
        unfreeze_prefixes = set()
        if unfreeze_components:
            unfreeze_prefixes = {
                LORA_COMPONENT_PREFIXES[c]
                for c in unfreeze_components
                if c in LORA_COMPONENT_PREFIXES
            }
        
        for n, p in model.named_parameters():
            if "lora_A" not in n and "lora_B" not in n:
                # Check if this parameter belongs to an unfreeze component
                should_unfreeze = any(n.startswith(prefix) for prefix in unfreeze_prefixes)
                if not should_unfreeze:
                    p.requires_grad = False

    return replaced_paths


def merge_lora_into_sam3(model: nn.Module) -> List[str]:
    """
    Fold LoRA delta into base Linear weights and replace LoRALinear with nn.Linear for inference.

    LoRA adds (x @ A @ B) * scaling; merged weight is W + ((B.T @ A.T) * scaling).

    Returns:
        Module paths where LoRA layers were merged.
    """
    merged_paths = []

    def _merge_recursive(module: nn.Module, prefix: str) -> None:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                w = child.linear.weight.data
                b = child.linear.bias.data.clone() if child.linear.bias is not None else None
                scaling = child.scaling
                delta = (child.lora_B.data.T @ child.lora_A.data.T) * scaling
                w_merged = w + delta
                linear_new = nn.Linear(
                    child.linear.in_features,
                    child.linear.out_features,
                    bias=child.linear.bias is not None,
                )
                linear_new = linear_new.to(
                    device=child.linear.weight.device, dtype=child.linear.weight.dtype
                )
                linear_new.weight.data = w_merged.to(
                    device=child.linear.weight.device, dtype=child.linear.weight.dtype
                )
                if b is not None:
                    linear_new.bias.data = b.to(
                        device=child.linear.weight.device,
                        dtype=child.linear.weight.dtype,
                    )
                setattr(module, name, DTypeSafeLinear(linear_new))
                merged_paths.append(full_name)
            else:
                _merge_recursive(child, full_name)

    _merge_recursive(model, "")
    return merged_paths

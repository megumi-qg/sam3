# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.train.loss.loss_fns import CORE_LOSS_KEY, dice_loss, sigmoid_focal_loss


class Sam3TrackerLossWrapper(nn.Module):
    """
    Tracker loss v2 for the single-object SAM3 training adapter.

    The loss is intentionally lightweight but goes beyond the v1 smoke-test
    objective by supervising three complementary signals:
    1. Dense mask quality on every sampled stage.
    2. Target presence / absence via `object_score_logits`.
    3. Lightweight temporal consistency on `obj_ptr` for adjacent visible stages.
    """

    def __init__(
        self,
        mask_weight: float = 20.0,
        dice_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_high_res_masks: bool = True,
        presence_weight: float = 1.0,
        presence_pos_weight: float = 5.0,
        temporal_ptr_weight: float = 0.1,
        temporal_area_weight: float = 0.0,
        temporal_centroid_weight: float = 0.0,
        temporal_min_area: float = 1e-4,
        temporal_area_delta: float = 0.05,
        temporal_centroid_delta: float = 0.02,
    ) -> None:
        super().__init__()
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_high_res_masks = use_high_res_masks
        self.presence_weight = presence_weight
        self.presence_pos_weight = presence_pos_weight
        self.temporal_ptr_weight = temporal_ptr_weight
        self.temporal_area_weight = temporal_area_weight
        self.temporal_centroid_weight = temporal_centroid_weight
        self.temporal_min_area = temporal_min_area
        self.temporal_area_delta = temporal_area_delta
        self.temporal_centroid_delta = temporal_centroid_delta

    def _get_pred_key(self) -> str:
        return "pred_masks_high_res" if self.use_high_res_masks else "pred_masks"

    def _extract_target_masks(
        self, targets: Dict[str, torch.Tensor], pred_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_masks = targets.get("masks")
        target_missing = (
            target_masks is None
            or target_masks.numel() == 0
            or ("num_boxes" in targets and targets["num_boxes"].sum().item() == 0)
        )
        if target_missing:
            target_masks = torch.zeros_like(pred_masks[:1])
            valid_mask = torch.ones(
                target_masks.shape[0], device=target_masks.device, dtype=torch.bool
            )
            return target_masks, valid_mask

        target_masks = target_masks.to(dtype=pred_masks.dtype)
        if target_masks.dim() == 2:
            target_masks = target_masks.unsqueeze(0)

        valid_mask = targets.get("is_valid_mask", None)
        if valid_mask is None or valid_mask.numel() == 0:
            valid_mask = torch.ones(
                target_masks.shape[0], device=target_masks.device, dtype=torch.bool
            )
        else:
            valid_mask = valid_mask.to(device=target_masks.device, dtype=torch.bool)
        return target_masks, valid_mask

    def _get_target_presence(
        self,
        targets: Dict[str, torch.Tensor],
        *,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        target_masks = targets.get("masks")
        if target_masks is None or target_masks.numel() == 0:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)

        if target_masks.dim() == 2:
            target_masks = target_masks.unsqueeze(0)
        target_masks = target_masks.to(device=device)
        target_presence = (target_masks.flatten(1) > 0).any(dim=1).to(torch.float32)

        if target_presence.numel() < batch_size:
            padded = torch.zeros(batch_size, device=device, dtype=torch.float32)
            padded[: target_presence.numel()] = target_presence
            target_presence = padded
        else:
            target_presence = target_presence[:batch_size]

        if "num_boxes" in targets and targets["num_boxes"] is not None:
            num_boxes = targets["num_boxes"].to(device=device).view(-1).to(torch.float32)
            if num_boxes.numel() < batch_size:
                padded = torch.zeros(batch_size, device=device, dtype=torch.float32)
                padded[: num_boxes.numel()] = num_boxes
                num_boxes = padded
            else:
                num_boxes = num_boxes[:batch_size]
            target_presence = target_presence * (num_boxes > 0).to(torch.float32)

        return target_presence

    def _compute_mask_loss(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred_masks = outputs[self._get_pred_key()]
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks[:, 0]

        target_masks, valid_mask = self._extract_target_masks(targets, pred_masks)

        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(1),
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        max_instances = min(
            pred_masks.shape[0], target_masks.shape[0], valid_mask.shape[0]
        )
        pred_masks = pred_masks[:max_instances]
        target_masks = target_masks[:max_instances]
        valid_mask = valid_mask[:max_instances]

        if valid_mask.sum() == 0:
            zero = pred_masks.sum() * 0.0
            return {
                "loss_mask": zero,
                "loss_dice": zero,
                "num_valid_masks": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        pred_masks = pred_masks[valid_mask]
        target_masks = target_masks[valid_mask]
        num_masks = torch.clamp(valid_mask.sum().float(), min=1.0)

        loss_mask = sigmoid_focal_loss(
            pred_masks.flatten(1),
            target_masks.flatten(1),
            num_masks,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            triton=False,
        )
        loss_dice = dice_loss(
            pred_masks.flatten(1),
            target_masks.flatten(1),
            num_masks,
        )
        core_loss = self.mask_weight * loss_mask + self.dice_weight * loss_dice
        return {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "num_valid_masks": num_masks.detach(),
            CORE_LOSS_KEY: core_loss,
        }

    def _compute_presence_loss(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        object_score_logits = outputs.get("object_score_logits")
        pred_masks = outputs[self._get_pred_key()]
        zero = pred_masks.sum() * 0.0
        if self.presence_weight <= 0.0 or object_score_logits is None:
            return {
                "loss_presence": zero,
                "num_presence_labels": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        logits = object_score_logits.float().view(object_score_logits.shape[0], -1)
        if logits.shape[1] != 1:
            logits = logits[:, :1]
        logits = logits[:, 0]

        target_presence = self._get_target_presence(
            targets, device=logits.device, batch_size=logits.shape[0]
        )
        pos_weight = torch.tensor(
            self.presence_pos_weight, device=logits.device, dtype=logits.dtype
        )
        loss_presence = F.binary_cross_entropy_with_logits(
            logits,
            target_presence,
            pos_weight=pos_weight,
        )
        return {
            "loss_presence": loss_presence,
            "num_presence_labels": torch.tensor(
                float(logits.shape[0]), device=logits.device
            ),
            CORE_LOSS_KEY: self.presence_weight * loss_presence,
        }

    def _compute_temporal_ptr_loss(
        self,
        prev_outputs: Optional[Dict[str, torch.Tensor]],
        prev_targets: Optional[Dict[str, torch.Tensor]],
        curr_outputs: Dict[str, torch.Tensor],
        curr_targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        curr_pred_masks = curr_outputs[self._get_pred_key()]
        zero = curr_pred_masks.sum() * 0.0
        if (
            self.temporal_ptr_weight <= 0.0
            or prev_outputs is None
            or prev_targets is None
            or "obj_ptr" not in prev_outputs
            or "obj_ptr" not in curr_outputs
        ):
            return {
                "loss_temporal_ptr": zero,
                "num_temporal_pairs": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        prev_ptr = prev_outputs["obj_ptr"].float()
        curr_ptr = curr_outputs["obj_ptr"].float()
        batch_size = min(prev_ptr.shape[0], curr_ptr.shape[0])
        if batch_size == 0:
            return {
                "loss_temporal_ptr": zero,
                "num_temporal_pairs": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        prev_presence = self._get_target_presence(
            prev_targets, device=prev_ptr.device, batch_size=batch_size
        ).bool()
        curr_presence = self._get_target_presence(
            curr_targets, device=curr_ptr.device, batch_size=batch_size
        ).bool()
        valid_pairs = prev_presence & curr_presence

        if valid_pairs.sum() == 0:
            return {
                "loss_temporal_ptr": zero,
                "num_temporal_pairs": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        prev_ptr = F.normalize(prev_ptr[:batch_size][valid_pairs], dim=-1)
        curr_ptr = F.normalize(curr_ptr[:batch_size][valid_pairs], dim=-1)
        loss_temporal_ptr = 1.0 - (prev_ptr * curr_ptr).sum(dim=-1)
        loss_temporal_ptr = loss_temporal_ptr.mean()
        return {
            "loss_temporal_ptr": loss_temporal_ptr,
            "num_temporal_pairs": valid_pairs.sum().detach().float(),
            CORE_LOSS_KEY: self.temporal_ptr_weight * loss_temporal_ptr,
        }

    def _compute_temporal_mask_consistency_loss(
        self,
        prev_outputs: Optional[Dict[str, torch.Tensor]],
        curr_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        curr_pred_masks = curr_outputs[self._get_pred_key()]
        zero = curr_pred_masks.sum() * 0.0
        if (
            (self.temporal_area_weight <= 0.0 and self.temporal_centroid_weight <= 0.0)
            or prev_outputs is None
            or self._get_pred_key() not in prev_outputs
        ):
            return {
                "loss_temporal_area": zero,
                "loss_temporal_centroid": zero,
                "num_temporal_mask_pairs": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        prev_pred_masks = prev_outputs[self._get_pred_key()]
        if prev_pred_masks.dim() == 4 and prev_pred_masks.shape[1] == 1:
            prev_pred_masks = prev_pred_masks[:, 0]
        if curr_pred_masks.dim() == 4 and curr_pred_masks.shape[1] == 1:
            curr_pred_masks = curr_pred_masks[:, 0]

        if prev_pred_masks.shape[-2:] != curr_pred_masks.shape[-2:]:
            prev_pred_masks = F.interpolate(
                prev_pred_masks.unsqueeze(1),
                size=curr_pred_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        batch_size = min(prev_pred_masks.shape[0], curr_pred_masks.shape[0])
        if batch_size == 0:
            return {
                "loss_temporal_area": zero,
                "loss_temporal_centroid": zero,
                "num_temporal_mask_pairs": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        prev_prob = torch.sigmoid(torch.clamp(prev_pred_masks[:batch_size].float(), -50.0, 50.0))
        curr_prob = torch.sigmoid(torch.clamp(curr_pred_masks[:batch_size].float(), -50.0, 50.0))
        h, w = curr_prob.shape[-2:]
        area_scale = float(h * w)
        prev_area = prev_prob.flatten(1).sum(dim=1) / area_scale
        curr_area = curr_prob.flatten(1).sum(dim=1) / area_scale
        valid_pairs = (prev_area > self.temporal_min_area) & (
            curr_area > self.temporal_min_area
        )

        if valid_pairs.sum() == 0:
            return {
                "loss_temporal_area": zero,
                "loss_temporal_centroid": zero,
                "num_temporal_mask_pairs": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        area_diff = torch.log(prev_area[valid_pairs] + 1e-6) - torch.log(
            curr_area[valid_pairs] + 1e-6
        )
        loss_temporal_area = F.smooth_l1_loss(
            area_diff,
            torch.zeros_like(area_diff),
            beta=self.temporal_area_delta,
        )

        ys = torch.linspace(0.0, 1.0, h, device=curr_prob.device, dtype=curr_prob.dtype)
        xs = torch.linspace(0.0, 1.0, w, device=curr_prob.device, dtype=curr_prob.dtype)
        y_grid = ys.view(1, h, 1)
        x_grid = xs.view(1, 1, w)
        prev_mass = prev_prob.flatten(1).sum(dim=1).clamp(min=1e-6)
        curr_mass = curr_prob.flatten(1).sum(dim=1).clamp(min=1e-6)
        prev_cx = (prev_prob * x_grid).flatten(1).sum(dim=1) / prev_mass
        curr_cx = (curr_prob * x_grid).flatten(1).sum(dim=1) / curr_mass
        prev_cy = (prev_prob * y_grid).flatten(1).sum(dim=1) / prev_mass
        curr_cy = (curr_prob * y_grid).flatten(1).sum(dim=1) / curr_mass
        centroid_diff = torch.stack(
            [
                prev_cx[valid_pairs] - curr_cx[valid_pairs],
                prev_cy[valid_pairs] - curr_cy[valid_pairs],
            ],
            dim=1,
        )
        loss_temporal_centroid = F.smooth_l1_loss(
            centroid_diff,
            torch.zeros_like(centroid_diff),
            beta=self.temporal_centroid_delta,
        )

        core_loss = (
            self.temporal_area_weight * loss_temporal_area
            + self.temporal_centroid_weight * loss_temporal_centroid
        )
        return {
            "loss_temporal_area": loss_temporal_area,
            "loss_temporal_centroid": loss_temporal_centroid,
            "num_temporal_mask_pairs": valid_pairs.sum().detach().float(),
            CORE_LOSS_KEY: core_loss,
        }

    def _merge_losses(
        self, total_losses: Dict[str, torch.Tensor], stage_losses: Dict[str, torch.Tensor]
    ) -> None:
        for key, value in stage_losses.items():
            total_losses[key] = total_losses.get(key, 0) + value

    def forward(self, find_stages, find_targets):
        total_losses: Dict[str, torch.Tensor] = {}
        assert len(find_stages) == len(find_targets)

        prev_outputs = None
        prev_targets = None
        for stage_outputs, stage_targets in zip(find_stages, find_targets):
            outputs = stage_outputs[-1]

            self._merge_losses(total_losses, self._compute_mask_loss(outputs, stage_targets))
            self._merge_losses(
                total_losses, self._compute_presence_loss(outputs, stage_targets)
            )
            self._merge_losses(
                total_losses,
                self._compute_temporal_ptr_loss(
                    prev_outputs, prev_targets, outputs, stage_targets
                ),
            )
            self._merge_losses(
                total_losses,
                self._compute_temporal_mask_consistency_loss(prev_outputs, outputs),
            )

            prev_outputs = outputs
            prev_targets = stage_targets

        return total_losses


class Sam3WeakTrackerLossWrapper(Sam3TrackerLossWrapper):
    """
    Weak tracker loss for scribble-supervised tracking.

    Targets follow the project-wide tri-state convention:
    1 = foreground scribble / accepted pseudo foreground, 0 = valid background,
    255 = ignore. The mask loss is computed only on valid pixels so pseudo masks
    can be used as conditioning while scribbles remain the supervision gate.
    """

    def __init__(self, *args, ignore_index: int = 255, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ignore_index = int(ignore_index)

    def _get_target_presence(
        self,
        targets: Dict[str, torch.Tensor],
        *,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        target_masks = targets.get("masks")
        if target_masks is None or target_masks.numel() == 0:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)

        if target_masks.dim() == 2:
            target_masks = target_masks.unsqueeze(0)
        target_masks = target_masks.to(device=device)
        target_presence = (target_masks.flatten(1) == 1).any(dim=1).to(torch.float32)

        if target_presence.numel() < batch_size:
            padded = torch.zeros(batch_size, device=device, dtype=torch.float32)
            padded[: target_presence.numel()] = target_presence
            target_presence = padded
        else:
            target_presence = target_presence[:batch_size]

        return target_presence

    def _compute_mask_loss(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred_masks = outputs[self._get_pred_key()]
        if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
            pred_masks = pred_masks[:, 0]

        target_masks = targets.get("masks")
        target_missing = (
            target_masks is None
            or target_masks.numel() == 0
            or ("num_boxes" in targets and targets["num_boxes"].sum().item() == 0)
        )
        if target_missing:
            zero = pred_masks.sum() * 0.0
            return {
                "loss_mask": zero,
                "loss_dice": zero,
                "num_valid_masks": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        if target_masks.dim() == 2:
            target_masks = target_masks.unsqueeze(0)
        target_masks = target_masks.to(device=pred_masks.device)

        instance_valid = targets.get("is_valid_mask", None)
        if instance_valid is None or instance_valid.numel() == 0:
            instance_valid = torch.ones(
                target_masks.shape[0], device=pred_masks.device, dtype=torch.bool
            )
        else:
            instance_valid = instance_valid.to(
                device=pred_masks.device, dtype=torch.bool
            )

        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks.unsqueeze(1),
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        max_instances = min(
            pred_masks.shape[0], target_masks.shape[0], instance_valid.shape[0]
        )
        pred_masks = pred_masks[:max_instances]
        target_masks = target_masks[:max_instances]
        instance_valid = instance_valid[:max_instances]

        if instance_valid.sum() == 0:
            zero = pred_masks.sum() * 0.0
            return {
                "loss_mask": zero,
                "loss_dice": zero,
                "num_valid_masks": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        pred_masks = pred_masks[instance_valid]
        target_masks = target_masks[instance_valid]

        valid_region = target_masks != self.ignore_index
        if valid_region.sum() == 0:
            zero = pred_masks.sum() * 0.0
            return {
                "loss_mask": zero,
                "loss_dice": zero,
                "num_valid_masks": zero.detach(),
                CORE_LOSS_KEY: zero,
            }

        pred_masks = torch.clamp(pred_masks.float(), min=-50.0, max=50.0)
        target_binary = (target_masks == 1).to(dtype=pred_masks.dtype)
        valid_region = valid_region.to(dtype=pred_masks.dtype)

        prob = pred_masks.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, target_binary, reduction="none"
        )
        p_t = prob * target_binary + (1.0 - prob) * (1.0 - target_binary)
        focal = ce_loss * ((1.0 - p_t) ** self.focal_gamma)
        if self.focal_alpha >= 0:
            alpha_t = (
                self.focal_alpha * target_binary
                + (1.0 - self.focal_alpha) * (1.0 - target_binary)
            )
            focal = alpha_t * focal

        focal = focal * valid_region
        valid_pixels = valid_region.flatten(1).sum(dim=1).clamp(min=1.0)
        loss_mask = (focal.flatten(1).sum(dim=1) / valid_pixels).mean()

        prob_valid = prob * valid_region
        target_valid = target_binary * valid_region
        numerator = 2.0 * (prob_valid * target_valid).flatten(1).sum(dim=1)
        denominator = (
            prob_valid.flatten(1).sum(dim=1)
            + target_valid.flatten(1).sum(dim=1)
        )
        loss_dice = (1.0 - (numerator + 1.0) / (denominator + 1.0)).mean()

        core_loss = self.mask_weight * loss_mask + self.dice_weight * loss_dice
        return {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
            "num_valid_masks": instance_valid.sum().float().detach(),
            CORE_LOSS_KEY: core_loss,
        }

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import logging
import pickle
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam3.model.data_misc import BatchedDatapoint
from sam3.model.sam3_tracker_base import Sam3TrackerBase

logger = logging.getLogger(__name__)


class Sam3TrackerTrainAdapter(Sam3TrackerBase):
    """
    Training adapter for the original SAM3 single-object tracker.

    Design goals:
    - treat each (volume, category) pair as one single-object tracking sample
    - start from a single conditioning frame
    - train propagation on later slices with dense mask supervision
    - optionally randomize the conditioning slice and corrupt the seed mask to
      reduce train-test gap under auto-seed inference
    """

    def __init__(
        self,
        *,
        backbone,
        transformer,
        maskmem_backbone,
        checkpoint_path: Optional[str] = None,
        image_backbone_checkpoint_path: Optional[str] = None,
        image_backbone_lora_alpha: float = 16.0,
        image_backbone_lora_r: int = 8,
        strict_state_dict_loading: bool = False,
        freeze_image_encoder: bool = False,
        use_memory_selection: bool = False,
        init_frame_strategy: Literal["earliest", "random_visible", "mixed"] = "earliest",
        earliest_init_frame_prob: float = 0.5,
        use_noisy_seed_mask: bool = False,
        clean_seed_prob: float = 1.0,
        light_noise_prob: float = 0.0,
        medium_noise_prob: float = 0.0,
        seed_shift_max_px: int = 8,
        seed_erode_dilate_max_kernel: int = 7,
        seed_dropout_max_ratio: float = 0.15,
    ) -> None:
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            maskmem_backbone=maskmem_backbone,
            image_size=1008,
            num_maskmem=7,
            backbone_stride=14,
            multimask_output_in_sam=True,
            forward_backbone_per_frame_for_eval=True,
            trim_past_non_cond_mem_for_eval=False,
            multimask_output_for_tracking=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            non_overlap_masks_for_mem_enc=False,
            max_cond_frames_in_attn=4,
            offload_output_to_cpu_for_eval=False,
            sam_mask_decoder_extra_args={
                "dynamic_multimask_via_stability": True,
                "dynamic_multimask_stability_delta": 0.05,
                "dynamic_multimask_stability_thresh": 0.98,
            },
            use_memory_selection=use_memory_selection,
        )
        self.iter_use_prev_mask_pred = False
        self.add_all_frames_to_correct_as_cond = False
        self.teacher_force_obj_scores_for_mem = False
        self.prob_to_dropout_spatial_mem = 0.0
        self.init_frame_strategy = init_frame_strategy
        self.earliest_init_frame_prob = float(earliest_init_frame_prob)
        self.use_noisy_seed_mask = use_noisy_seed_mask
        self.clean_seed_prob = float(clean_seed_prob)
        self.light_noise_prob = float(light_noise_prob)
        self.medium_noise_prob = float(medium_noise_prob)
        self.seed_shift_max_px = max(0, int(seed_shift_max_px))
        self.seed_erode_dilate_max_kernel = max(1, int(seed_erode_dilate_max_kernel))
        self.seed_dropout_max_ratio = max(0.0, float(seed_dropout_max_ratio))

        if self.init_frame_strategy not in {"earliest", "random_visible", "mixed"}:
            raise ValueError(
                f"Unsupported init_frame_strategy={self.init_frame_strategy!r}"
            )

        if checkpoint_path is not None:
            self._load_full_sam3_checkpoint(
                checkpoint_path,
                strict_state_dict_loading=strict_state_dict_loading,
            )
        if image_backbone_checkpoint_path is not None:
            self._load_image_backbone_checkpoint(
                image_backbone_checkpoint_path,
                lora_alpha=image_backbone_lora_alpha,
                lora_r=image_backbone_lora_r,
            )

        if freeze_image_encoder and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze SAM3 tracker image encoder backbone")

    def _load_checkpoint_state_dict(self, checkpoint_path: str, *, allow_pickle: bool):
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            try:
                ckpt = torch.load(f, map_location="cpu", weights_only=not allow_pickle)
            except pickle.UnpicklingError:
                if not allow_pickle:
                    raise
                logger.info(
                    "Falling back to weights_only=False for checkpoint %s", checkpoint_path
                )
                f.seek(0)
                ckpt = torch.load(f, map_location="cpu", weights_only=False)
        return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    def _load_full_sam3_checkpoint(
        self, checkpoint_path: str, *, strict_state_dict_loading: bool
    ) -> None:
        state_dict = self._load_checkpoint_state_dict(
            checkpoint_path, allow_pickle=False
        )

        tracker_state = {}
        for key, value in state_dict.items():
            if key.startswith("tracker."):
                tracker_state[key[len("tracker."):]] = value
            elif key.startswith("detector.backbone.vision_backbone."):
                # The released SAM3 tracker checkpoint path does not contain a tracker backbone,
                # so we initialize it from detector visual weights.
                mapped_key = "backbone." + key[len("detector.backbone.") :]
                tracker_state[mapped_key] = value

        missing_keys, unexpected_keys = self.load_state_dict(
            tracker_state, strict=strict_state_dict_loading
        )
        if missing_keys:
            logger.info("Single-object tracker adapter missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.info(
                "Single-object tracker adapter unexpected keys: %s", unexpected_keys
            )

    def _merge_lora_weight(
        self,
        state_dict,
        weight_key: str,
        *,
        lora_alpha: float,
        lora_r: int,
    ):
        weight = state_dict[weight_key]
        a_key = weight_key.replace(".linear.weight", ".lora_A")
        b_key = weight_key.replace(".linear.weight", ".lora_B")
        if a_key not in state_dict or b_key not in state_dict:
            return weight

        scaling = lora_alpha / lora_r
        delta = (state_dict[b_key].T @ state_dict[a_key].T).to(weight.dtype) * scaling
        return weight + delta

    def _load_image_backbone_checkpoint(
        self,
        checkpoint_path: str,
        *,
        lora_alpha: float,
        lora_r: int,
    ) -> None:
        state_dict = self._load_checkpoint_state_dict(
            checkpoint_path, allow_pickle=True
        )

        image_backbone_state = {}
        for key, value in state_dict.items():
            if not key.startswith("backbone.vision_backbone."):
                continue
            if key.endswith(".lora_A") or key.endswith(".lora_B"):
                continue
            mapped_key = key.replace(".linear.", ".")
            if key.endswith(".linear.weight"):
                value = self._merge_lora_weight(
                    state_dict,
                    key,
                    lora_alpha=lora_alpha,
                    lora_r=lora_r,
                )
            image_backbone_state[mapped_key] = value

        missing_keys, unexpected_keys = self.load_state_dict(
            image_backbone_state, strict=False
        )
        if missing_keys:
            logger.info(
                "Image backbone init missing keys (expected for non-backbone modules): %s",
                missing_keys,
            )
        if unexpected_keys:
            logger.info(
                "Image backbone init unexpected keys: %s", unexpected_keys
            )
        logger.info(
            "Loaded tracker visual backbone init from %s with %d tensors",
            checkpoint_path,
            len(image_backbone_state),
        )

    def _rand_float(self, device):
        return float(torch.rand((), device=device).item())

    def _sample_index(self, upper: int, device):
        return int(torch.randint(upper, size=(), device=device).item())

    def _stage_mask_from_target(self, target, image_hw, device, *, use_seed=False):
        h, w = image_hw
        batch_size = (
            int(target.num_boxes.shape[0])
            if target is not None and hasattr(target, "num_boxes")
            else 1
        )
        zero_mask = torch.zeros(batch_size, 1, h, w, device=device, dtype=torch.float32)
        segments = getattr(target, "seed_segments", None) if use_seed else target.segments
        if target is None or segments is None:
            return zero_mask, False
        if segments.numel() == 0 or (
            hasattr(target, "num_boxes") and target.num_boxes.sum().item() == 0
        ):
            return zero_mask, False

        if segments.dim() == 2:
            segments = segments.unsqueeze(0)
        segments = segments.to(device=device, dtype=torch.float32)

        stage_masks = []
        cursor = 0
        visible_any = False
        num_boxes = target.num_boxes.tolist()
        for count in num_boxes:
            count = int(count)
            if count > 0:
                # Weak scribble masks use 255 for ignore.  The tracker seed must
                # contain only confident foreground pixels, never the ignored area.
                sample_mask = segments[cursor]
                sample_mask = ((sample_mask > 0) & (sample_mask < 255)).to(
                    dtype=torch.float32
                )
                cursor += count
                visible_any = visible_any or bool((sample_mask > 0).any().item())
            else:
                sample_mask = torch.zeros(h, w, device=device, dtype=torch.float32)
            stage_masks.append(sample_mask)

        stage_masks = torch.stack(stage_masks, dim=0).unsqueeze(1)
        return stage_masks, visible_any

    def _sample_init_frame(self, visible_frames, *, start_frame_idx, device):
        if not visible_frames:
            return start_frame_idx
        if self.init_frame_strategy == "earliest":
            return visible_frames[0]
        if self.init_frame_strategy == "random_visible":
            return visible_frames[self._sample_index(len(visible_frames), device)]
        if self._rand_float(device) < self.earliest_init_frame_prob:
            return visible_frames[0]
        return visible_frames[self._sample_index(len(visible_frames), device)]

    def _sample_seed_mode(self, device):
        if not self.use_noisy_seed_mask:
            return "clean"

        probs = {
            "clean": max(0.0, self.clean_seed_prob),
            "light_noise": max(0.0, self.light_noise_prob),
            "medium_noise": max(0.0, self.medium_noise_prob),
        }
        total_prob = sum(probs.values())
        if total_prob <= 0.0:
            return "clean"

        threshold = self._rand_float(device) * total_prob
        running = 0.0
        for mode, prob in probs.items():
            running += prob
            if threshold <= running:
                return mode
        return "clean"

    def _shift_mask(self, mask, shift_y: int, shift_x: int):
        shifted = torch.roll(mask, shifts=(shift_y, shift_x), dims=(-2, -1))
        if shift_y > 0:
            shifted[..., :shift_y, :] = 0
        elif shift_y < 0:
            shifted[..., shift_y:, :] = 0
        if shift_x > 0:
            shifted[..., :, :shift_x] = 0
        elif shift_x < 0:
            shifted[..., :, shift_x:] = 0
        return shifted

    def _morph_mask(self, mask, kernel_size: int, *, mode: Literal["erode", "dilate"]):
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        if mode == "dilate":
            return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        if mode == "erode":
            return -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
        raise ValueError(f"Unsupported morphology mode: {mode}")

    def _dropout_patch(self, mask, *, max_ratio: float, device):
        if max_ratio <= 0.0:
            return mask
        _, _, h, w = mask.shape
        drop_h = max(1, int(round(h * self._rand_float(device) * max_ratio)))
        drop_w = max(1, int(round(w * self._rand_float(device) * max_ratio)))
        top = self._sample_index(max(1, h - drop_h + 1), device)
        left = self._sample_index(max(1, w - drop_w + 1), device)
        out = mask.clone()
        out[..., top : top + drop_h, left : left + drop_w] = 0
        return out

    def _perturb_single_seed_mask(self, mask, *, severity, device):
        if severity == "light_noise":
            shift_limit = max(1, self.seed_shift_max_px // 2)
            morph_limit = max(1, self.seed_erode_dilate_max_kernel // 2)
            dropout_ratio = self.seed_dropout_max_ratio * 0.5
            shift_prob, morph_prob, dropout_prob = 0.5, 0.5, 0.3
        elif severity == "medium_noise":
            shift_limit = max(1, self.seed_shift_max_px)
            morph_limit = max(1, self.seed_erode_dilate_max_kernel)
            dropout_ratio = self.seed_dropout_max_ratio
            shift_prob, morph_prob, dropout_prob = 0.8, 0.8, 0.6
        else:
            return mask

        original_mask = (mask > 0.5).to(dtype=torch.float32)
        perturbed = original_mask.clone()

        if shift_limit > 0 and self._rand_float(device) < shift_prob:
            shift_y = self._sample_index(2 * shift_limit + 1, device) - shift_limit
            shift_x = self._sample_index(2 * shift_limit + 1, device) - shift_limit
            perturbed = self._shift_mask(perturbed, shift_y=shift_y, shift_x=shift_x)

        if morph_limit > 1 and self._rand_float(device) < morph_prob:
            kernel_candidates = max(1, (morph_limit + 1) // 2)
            kernel_size = 2 * self._sample_index(kernel_candidates, device) + 1
            morph_mode = "dilate" if self._rand_float(device) < 0.5 else "erode"
            perturbed = self._morph_mask(perturbed, kernel_size, mode=morph_mode)

        if dropout_ratio > 0.0 and self._rand_float(device) < dropout_prob:
            perturbed = self._dropout_patch(
                perturbed, max_ratio=dropout_ratio, device=device
            )

        perturbed = (perturbed > 0.5).to(dtype=torch.float32)
        if not bool((perturbed > 0.5).any().item()):
            return original_mask
        return perturbed

    def _build_seed_mask(self, gt_mask, *, device):
        mode = self._sample_seed_mode(device)
        if mode == "clean":
            return gt_mask, mode

        seed_mask = gt_mask.clone()
        for sample_idx in range(seed_mask.shape[0]):
            seed_mask[sample_idx : sample_idx + 1] = self._perturb_single_seed_mask(
                seed_mask[sample_idx : sample_idx + 1],
                severity=mode,
                device=device,
            )
        return seed_mask, mode

    def prepare_prompt_inputs(self, backbone_out, input: BatchedDatapoint, start_frame_idx=0):
        num_frames = len(input.find_targets)
        device = input.img_batch.device
        image_hw = tuple(input.img_batch.shape[-2:])

        gt_masks_per_frame = {}
        seed_masks_per_frame = {}
        visible_frames = []
        seed_visible_frames = []
        for stage_id, targets in enumerate(input.find_targets):
            stage_mask, is_visible = self._stage_mask_from_target(
                targets, image_hw=image_hw, device=device
            )
            seed_stage_mask, seed_is_visible = self._stage_mask_from_target(
                targets, image_hw=image_hw, device=device, use_seed=True
            )
            gt_masks_per_frame[stage_id] = stage_mask
            seed_masks_per_frame[stage_id] = seed_stage_mask
            if is_visible:
                visible_frames.append(stage_id)
            if seed_is_visible:
                seed_visible_frames.append(stage_id)

        init_frame = self._sample_init_frame(
            seed_visible_frames or visible_frames,
            start_frame_idx=start_frame_idx,
            device=device,
        )
        seed_mask, seed_mode = self._build_seed_mask(
            seed_masks_per_frame.get(init_frame, gt_masks_per_frame[init_frame]),
            device=device,
        )
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        backbone_out["num_frames"] = num_frames
        backbone_out["init_cond_frames"] = [init_frame]
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t != init_frame
        ]
        backbone_out["frames_to_add_correction_pt"] = []
        backbone_out["mask_inputs_per_frame"] = {init_frame: seed_mask}
        backbone_out["point_inputs_per_frame"] = {}
        backbone_out["seed_metadata"] = {
            "init_frame": init_frame,
            "seed_mode": seed_mode,
            "visible_frames": visible_frames,
            "seed_visible_frames": seed_visible_frames,
        }
        return backbone_out

    def forward(self, input: BatchedDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            backbone_out = self.forward_image(input.img_batch)
        else:
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        stage_outputs = self.forward_tracking(backbone_out, input)
        return [[stage_out] for stage_out in stage_outputs]

    def back_convert(self, targets):
        return {
            "masks": targets.segments,
            "is_valid_mask": targets.is_valid_segment,
            "num_boxes": targets.num_boxes,
            "object_ids_padded": targets.object_ids_padded,
        }

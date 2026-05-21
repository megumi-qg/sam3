# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import fnmatch
import logging
from typing import Iterable, Optional

import torch
from iopath.common.file_io import g_pathmgr

from sam3.model.data_misc import BatchedDatapoint, NestedTensor
from sam3.model.video_tracking_multiplex import (
    VideoTrackingDynamicMultiplex,
    VideoTrackingMultiplex,
)

logger = logging.getLogger(__name__)


class Sam3MultiplexTrainAdapter(VideoTrackingDynamicMultiplex):
    """
    Thin training adapter for SAM 3.1 multiplex tracker.

    It keeps the tracker architecture unchanged while adapting our current
    `BatchedDatapoint` training batch and `Trainer` expectations:
    - wraps plain image tensors as `NestedTensor`
    - returns `SAM3Output` instead of the raw tracker tuple
    - exposes a lightweight `back_convert` for the tracker loss wrapper
    """

    def __init__(
        self,
        *,
        backbone,
        transformer,
        maskmem_backbone,
        multiplex_controller,
        checkpoint_path: Optional[str] = None,
        strict_state_dict_loading: bool = False,
        freeze_image_encoder: bool = False,
        freeze_patterns: Optional[list[str]] = None,
        forward_backbone_per_frame_for_eval: bool = False,
    ) -> None:
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            maskmem_backbone=maskmem_backbone,
            multiplex_controller=multiplex_controller,
            image_size=1008,
            backbone_stride=14,
            num_maskmem=7,
            use_high_res_features_in_sam=True,
            use_obj_ptrs_in_encoder=True,
            max_obj_ptrs_in_encoder=16,
            add_tpos_enc_to_obj_ptrs=True,
            proj_tpos_enc_in_obj_ptrs=True,
            use_mlp_for_obj_ptr_proj=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            use_no_obj_ptr=True,
            use_linear_no_obj_ptr=True,
            no_obj_embed_spatial=True,
            sincos_tpos_enc=True,
            multimask_output_in_sam=True,
            multimask_output_for_tracking=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_multimask_token_for_obj_ptr=True,
            num_multimask_outputs=3,
            apply_sigmoid_to_mask_logits_for_mem_enc=True,
            sigmoid_scale_for_mem_enc=2.0,
            sigmoid_bias_for_mem_enc=-1.0,
            non_overlap_masks_for_mem_enc=False,
            add_output_suppression_embeddings=True,
            add_object_conditional_embeddings=False,
            condition_as_mask_input=True,
            condition_as_mask_input_fg=1.0,
            condition_as_mask_input_bg=0.0,
            use_maskmem_tpos_v2=True,
            save_image_features=True,
            randomness_fix=True,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            iou_prediction_use_sigmoid=False,
            forward_backbone_per_frame_for_eval=forward_backbone_per_frame_for_eval,
            offload_output_to_cpu_for_eval=False,
            trim_past_non_cond_mem_for_eval=False,
            max_cond_frames_in_attn=4,
            is_dynamic_model=True,
            sam_mask_decoder_extra_args={
                "dynamic_multimask_via_stability": True,
                "dynamic_multimask_stability_delta": 0.05,
                "dynamic_multimask_stability_thresh": 0.98,
            },
            compile_all_components=False,
            use_memory_selection=False,
        )

        if checkpoint_path is not None:
            self._load_checkpoint(
                checkpoint_path,
                strict_state_dict_loading=strict_state_dict_loading,
            )

        if freeze_image_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze multiplex image encoder backbone")

        if freeze_patterns:
            self._apply_freeze_patterns(freeze_patterns)

    def _load_checkpoint(
        self, checkpoint_path: str, *, strict_state_dict_loading: bool
    ) -> None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = self.load_state_dict(
            ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            logger.info("Multiplex train adapter missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.info("Multiplex train adapter unexpected keys: %s", unexpected_keys)

    def _apply_freeze_patterns(self, freeze_patterns: Iterable[str]) -> None:
        frozen_names = []
        for name, param in self.named_parameters():
            if any(fnmatch.fnmatch(name, pattern) for pattern in freeze_patterns):
                param.requires_grad = False
                frozen_names.append(name)
        logger.info("Froze %s params via freeze_patterns", len(frozen_names))

    def _normalize_input_batch(self, input: BatchedDatapoint) -> BatchedDatapoint:
        if isinstance(input.img_batch, NestedTensor):
            return input
        return BatchedDatapoint(
            img_batch=NestedTensor(tensors=input.img_batch, mask=None),
            find_text_batch=input.find_text_batch,
            find_inputs=input.find_inputs,
            find_targets=input.find_targets,
            find_metadatas=input.find_metadatas,
            raw_images=input.raw_images,
            get_queries=input.get_queries,
        )

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        # ACDC full-supervision uses a fixed object set per slice window, so we
        # deliberately bypass the dynamic-object bookkeeping path here.
        return VideoTrackingMultiplex.prepare_prompt_inputs(
            self, backbone_out, input, start_frame_idx=start_frame_idx
        )

    def forward_tracking(self, backbone_out, input, *args, **kwargs):
        return VideoTrackingMultiplex.forward_tracking(
            self, backbone_out, input, *args, **kwargs
        )

    def forward(self, input: BatchedDatapoint):
        input = self._normalize_input_batch(input)
        stage_outputs, _ = super().forward(input, is_inference=False)
        return [[stage_out] for stage_out in stage_outputs]

    def back_convert(self, targets):
        return {
            "masks": targets.segments,
            "is_valid_mask": targets.is_valid_segment,
            "num_boxes": targets.num_boxes,
            "object_ids_padded": targets.object_ids_padded,
        }

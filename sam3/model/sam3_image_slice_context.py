import logging
from typing import Dict, Optional

import torch
from sam3.model.model_misc import SAM3Output
from sam3.train.data.collator import BatchedDatapoint

from .geometry_encoders import Prompt
from .sam3_image import Sam3ImageOnVideoMultiGPU

logger = logging.getLogger(__name__)


class Sam3ImageSliceContext(Sam3ImageOnVideoMultiGPU):
    """
    Slice-context V1 model.

    This model predicts only the center slice in a sampled window and injects
    neighboring slice features as visual prompt tokens.
    """

    def __init__(
        self,
        *args,
        slice_context_adapter=None,
        center_frame_strategy: str = "middle",
        context_feature_level: int = -1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.slice_context_adapter = slice_context_adapter
        self.center_frame_strategy = center_frame_strategy
        self.context_feature_level = context_feature_level

    def _get_center_frame_idx(self, num_frames: int) -> int:
        if self.center_frame_strategy != "middle":
            raise ValueError(
                f"Unsupported center_frame_strategy: {self.center_frame_strategy}"
            )
        return num_frames // 2

    def _infer_num_prompt_instances(self, find_input) -> int:
        text_ids = getattr(find_input, "text_ids", None)
        if text_ids is not None and text_ids.numel() > 0:
            return int(text_ids.numel())

        input_boxes_mask = getattr(find_input, "input_boxes_mask", None)
        if input_boxes_mask is not None and input_boxes_mask.ndim > 0:
            return int(input_boxes_mask.shape[0])

        return 1

    def _build_context_prompt(
        self,
        backbone_out: Dict,
        center_frame_idx: int,
        num_prompt_instances: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.slice_context_adapter is None:
            return None, None

        frame_features = backbone_out.get("backbone_fpn", None)
        if frame_features is None:
            logger.warning(
                "Slice context adapter skipped because backbone_fpn is missing."
            )
            return None, None

        frame_features = frame_features[self.context_feature_level]
        context_tokens, neighbor_indices = self.slice_context_adapter(
            frame_features, center_idx=center_frame_idx
        )
        if context_tokens.numel() == 0:
            return None, None

        logger.debug(
            "Slice context uses center=%s with neighbors=%s",
            center_frame_idx,
            neighbor_indices,
        )

        visual_prompt_embed = context_tokens.unsqueeze(1).expand(
            -1, num_prompt_instances, -1
        )
        visual_prompt_mask = torch.zeros(
            (num_prompt_instances, context_tokens.shape[0]),
            dtype=torch.bool,
            device=context_tokens.device,
        )
        return visual_prompt_embed, visual_prompt_mask

    def forward_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt: Prompt,
        visual_prompt_embed: Optional[torch.Tensor] = None,
        visual_prompt_mask: Optional[torch.Tensor] = None,
    ):
        with torch.profiler.record_function("SAM3ImageSliceContext._encode_prompt"):
            prompt, prompt_mask, backbone_out = self._encode_prompt(
                backbone_out,
                find_input,
                geometric_prompt,
                visual_prompt_embed=visual_prompt_embed,
                visual_prompt_mask=visual_prompt_mask,
            )
        with torch.profiler.record_function("SAM3ImageSliceContext._run_encoder"):
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out, find_input, prompt, prompt_mask
            )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        with torch.profiler.record_function("SAM3ImageSliceContext._run_decoder"):
            out, hs = self._run_decoder(
                memory=out["encoder_hidden_states"],
                pos_embed=encoder_out["pos_embed"],
                src_mask=encoder_out["padding_mask"],
                out=out,
                prompt=prompt,
                prompt_mask=prompt_mask,
                encoder_out=encoder_out,
            )

        with torch.profiler.record_function(
            "SAM3ImageSliceContext._run_segmentation_heads"
        ):
            self._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=find_input.img_ids,
                vis_feat_sizes=encoder_out["vis_feat_sizes"],
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        if self.training or self.num_interactive_steps_val > 0:
            self._compute_matching(out, self.back_convert(find_target))
        return out

    def forward(self, input: BatchedDatapoint):
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        backbone_out.update(self.backbone.forward_image(input.img_batch))

        num_frames = len(input.find_inputs)
        if num_frames < 1:
            raise ValueError("Expected at least one sampled slice in the input window")

        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)

        if self.training:
            frame_indices = [self._get_center_frame_idx(num_frames)]
            loss_stages = frame_indices
        else:
            frame_indices = list(range(num_frames))
            loss_stages = None

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE,
            loss_stages=loss_stages,
        )

        num_interactive_steps = 0 if self.training else self.num_interactive_steps_val
        for frame_idx in frame_indices:
            find_input = input.find_inputs[frame_idx]
            find_target = input.find_targets[frame_idx]

            if (
                find_input.input_points is not None
                and find_input.input_points.numel() > 0
            ):
                logger.warning("Point prompts are ignored in slice-context V1.")

            geometric_prompt = Prompt(
                box_embeddings=find_input.input_boxes,
                box_mask=find_input.input_boxes_mask,
                box_labels=find_input.input_boxes_label,
            )

            num_prompt_instances = self._infer_num_prompt_instances(find_input)
            visual_prompt_embed, visual_prompt_mask = self._build_context_prompt(
                backbone_out=backbone_out,
                center_frame_idx=frame_idx,
                num_prompt_instances=num_prompt_instances,
            )

            stage_outs = []
            for cur_step in range(num_interactive_steps + 1):
                if cur_step > 0:
                    geometric_prompt, _ = self.interactive_prompt_sampler.sample(
                        geo_prompt=geometric_prompt,
                        find_target=find_target,
                        previous_out=stage_outs[-1],
                    )

                out = self.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=find_target,
                    geometric_prompt=geometric_prompt.clone(),
                    visual_prompt_embed=visual_prompt_embed,
                    visual_prompt_mask=visual_prompt_mask,
                )
                out["center_frame_idx"] = frame_idx
                stage_outs.append(out)

            previous_stages_out.append(stage_outs)
        return previous_stages_out

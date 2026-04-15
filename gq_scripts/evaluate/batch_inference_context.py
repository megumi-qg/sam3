"""
SAM3 slice-context V1 批量推理（滑窗上下文 → 中心切片预测 → 按 volume 聚合）

流程简述
--------
1. 读取测试目录下的 COCO JSON（默认 image_annotations.coco.json）与 images/。
2. 按病例与帧将切片分组为 volume。
3. 对每个 volume 的每张中心切片，构建一个局部滑窗（默认 5 张）；
   邻近切片用于构造 visual prompt tokens，模型只输出中心切片的 mask。
4. 将所有 volume 的结果写入 predictions.pkl，供 batch_evaluate.py 离线算指标。

当前脚本与 `batch_evaluate.py` 输出格式保持一致，因此评估链可直接复用。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2

import sam3
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from sam3.model_builder import build_sam3_image_video_context_model

# 保证以 `python gq_scripts/evaluate/batch_inference_context.py` 方式运行时，
# 能导入同目录下的 batch_inference 工具函数。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from batch_inference import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_COMPONENTS,
    DEFAULT_LORA_UNFREEZE_COMPONENTS,
    _is_lora_checkpoint,
    _safe_category_dir,
    build_volume_groups,
    save_volume_as_nii,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _default_test_dir() -> str:
    return "/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100/test"


def _default_checkpoint() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/full_video_lora_100_context_v1/"
        "checkpoints/checkpoint.pt"
    )


def _default_output_dir(test_dir: str) -> str:
    parent = os.path.dirname(test_dir.rstrip(os.sep))
    return os.path.join(parent, "inference_predictions_context")


def _build_transform(resolution: int):
    return v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _normalize_checkpoint_state_dict(ckpt_obj: Dict[str, Any]) -> Dict[str, Any]:
    if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        state_dict = ckpt_obj["model"]
    else:
        state_dict = ckpt_obj

    keys = list(state_dict.keys())
    if any(k.startswith("detector.") for k in keys):
        return OrderedDict(
            (k.replace("detector.", "", 1), v)
            for k, v in state_dict.items()
            if k.startswith("detector.")
        )
    return state_dict


def load_context_checkpoint_and_model(
    checkpoint_path: str,
    bpe_path: str,
    device: str,
    use_lora: Optional[bool] = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: float = DEFAULT_LORA_ALPHA,
    lora_target_components: Optional[List[str]] = None,
    lora_unfreeze_components: Optional[List[str]] = None,
    context_pool_size: int = 2,
    context_max_context_distance: int = 4,
    context_max_neighbor_frames: Optional[int] = 4,
    context_output_dim: int = 256,
    context_dropout: float = 0.1,
    center_frame_strategy: str = "middle",
    context_feature_level: int = -1,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _normalize_checkpoint_state_dict(ckpt)

    if use_lora is None:
        use_lora = _is_lora_checkpoint(state_dict)
        if use_lora:
            logger.info("Checkpoint 含 LoRA 参数，按 LoRA 结构加载。")

    if lora_target_components is None:
        lora_target_components = DEFAULT_LORA_TARGET_COMPONENTS
    if lora_unfreeze_components is None:
        lora_unfreeze_components = (
            DEFAULT_LORA_UNFREEZE_COMPONENTS if use_lora else None
        )

    model = build_sam3_image_video_context_model(
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
        async_all_gather=False,
        gather_backbone_out=False,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_target_components=lora_target_components if use_lora else None,
        lora_freeze_non_lora=False,
        lora_unfreeze_components=lora_unfreeze_components,
        context_pool_size=context_pool_size,
        context_max_context_distance=context_max_context_distance,
        context_max_neighbor_frames=context_max_neighbor_frames,
        context_output_dim=context_output_dim,
        context_dropout=context_dropout,
        center_frame_strategy=center_frame_strategy,
        context_feature_level=context_feature_level,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("load_state_dict 缺失键（前 20 个）: %s", list(missing)[:20])
    else:
        logger.info("Checkpoint 权重已加载。")
    if unexpected:
        logger.info("Checkpoint 中未使用的键（已忽略）: %d 个", len(unexpected))

    model.eval()
    return model


def _make_window_indices(center_pos: int, length: int, window_size: int) -> List[int]:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            f"context_window_size must be a positive odd integer, got {window_size}"
        )
    radius = window_size // 2
    return [min(max(center_pos + offset, 0), length - 1) for offset in range(-radius, radius + 1)]


def _prepare_image_batch(
    images: Sequence[Image.Image],
    transform,
    device: torch.device,
) -> torch.Tensor:
    tensors = [transform(v2.functional.to_image(image).to(device)) for image in images]
    return torch.stack(tensors, dim=0)


def _predict_center_mask(
    model,
    image_batch: torch.Tensor,
    center_idx: int,
    prompt: str,
    img_height: int,
    img_width: int,
    confidence_threshold: float,
) -> np.ndarray:
    device = image_batch.device
    with torch.inference_mode():
        backbone_out = model.backbone.forward_image(image_batch)
        text_outputs = model.backbone.forward_text([prompt], device=str(device))
        backbone_out.update(text_outputs)

        visual_prompt_embed, visual_prompt_mask = model._build_context_prompt(
            backbone_out=backbone_out,
            center_frame_idx=center_idx,
            num_prompt_instances=1,
        )
        geometric_prompt = model._get_dummy_prompt(num_prompts=1)
        find_stage = FindStage(
            img_ids=torch.tensor([center_idx], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        outputs = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            find_target=None,
            geometric_prompt=geometric_prompt,
            visual_prompt_embed=visual_prompt_embed,
            visual_prompt_mask=visual_prompt_mask,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        if "presence_logit_dec" in outputs:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = out_probs * presence_score
        out_probs = out_probs.squeeze(-1)

        keep = out_probs > confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        if out_masks.shape[0] == 0:
            return np.zeros((img_height, img_width), dtype=bool)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor(
            [img_width, img_height, img_width, img_height], device=device
        )
        _ = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_height, img_width),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()[:, 0]

        best_idx = out_probs.argmax().item()
        pred_mask_2d = (out_masks[best_idx] > 0.5).cpu().numpy().astype(bool)
        return pred_mask_2d


@torch.inference_mode()
def run_inference_for_volume_context(
    slice_list,
    volume_categories,
    test_dir,
    images_dict,
    categories_dict,
    categories_names_dict,
    model,
    transform,
    device: torch.device,
    confidence_threshold: float,
    context_window_size: int,
    save_png_dir=None,
    patient_name=None,
):
    result_categories = []

    ordered_slices = []
    for img_id, slice_idx, view in slice_list:
        img_info = images_dict[img_id]
        img_path = os.path.join(test_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        ordered_slices.append(
            {
                "img_id": img_id,
                "slice_idx": slice_idx,
                "view": view,
                "image": Image.open(img_path).convert("RGB"),
                "height": img_info["height"],
                "width": img_info["width"],
            }
        )

    if not ordered_slices:
        return result_categories

    for category_id in volume_categories:
        category_name = categories_dict[category_id]
        if category_id in categories_names_dict:
            text_prompt = np.random.choice(categories_names_dict[category_id]).lower()
        else:
            text_prompt = category_name.lower()

        pred_masks_2d = []
        slice_indices = []

        for center_pos, center_slice in enumerate(ordered_slices):
            window_indices = _make_window_indices(
                center_pos=center_pos,
                length=len(ordered_slices),
                window_size=context_window_size,
            )
            window_images = [ordered_slices[idx]["image"] for idx in window_indices]
            image_batch = _prepare_image_batch(window_images, transform, device)
            center_idx_in_window = context_window_size // 2

            pred_mask_2d = _predict_center_mask(
                model=model,
                image_batch=image_batch,
                center_idx=center_idx_in_window,
                prompt=text_prompt,
                img_height=center_slice["height"],
                img_width=center_slice["width"],
                confidence_threshold=confidence_threshold,
            )

            pred_masks_2d.append(pred_mask_2d)
            slice_indices.append(center_slice["slice_idx"])

            if save_png_dir and patient_name:
                png_subdir = os.path.join(
                    save_png_dir, patient_name, _safe_category_dir(category_name)
                )
                os.makedirs(png_subdir, exist_ok=True)
                png_path = os.path.join(
                    png_subdir, f"slice_{center_slice['slice_idx']:03d}.png"
                )
                Image.fromarray(pred_mask_2d.astype(np.uint8) * 255).save(png_path)

        result_categories.append(
            {
                "category_id": category_id,
                "category_name": category_name,
                "slice_indices": slice_indices,
                "masks": pred_masks_2d,
            }
        )

    return result_categories


def _resolve_patient_name(patient_id, frame_id, slice_list, images_dict) -> str:
    first_view = slice_list[0][2]
    if frame_id == "default":
        first_img_info = images_dict[slice_list[0][0]]
        first_filename = first_img_info["file_name"]
        if first_filename.startswith("Case"):
            return f"Case{patient_id}"
        return patient_id
    if first_view is None:
        if isinstance(frame_id, str) and not frame_id.isdigit():
            return f"subject{patient_id}_{frame_id}"
        return f"patient{patient_id}_frame{frame_id}"
    if first_view in ("SA", "LA"):
        return f"{patient_id}_{frame_id}"
    return f"patient{patient_id}_{frame_id}"


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 slice-context V1 批量推理：滑窗上下文预测中心切片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test_dir", type=str, default=_default_test_dir())
    parser.add_argument("--checkpoint_path", type=str, default=_default_checkpoint())
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="image_annotations.coco.json",
        help="COCO JSON 文件名（位于 test_dir 下）",
    )
    parser.add_argument("--resize_size", type=int, default=1008)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument(
        "--context_window_size",
        type=int,
        default=5,
        help="滑窗大小；必须为正奇数，建议与训练的 num_stages_sample 一致",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="predictions.pkl")
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--png_dir", type=str, default=None)
    parser.add_argument("--save_nii", action="store_true")
    parser.add_argument("--nii_dir", type=str, default=None)
    parser.add_argument(
        "--limit_volumes",
        type=int,
        default=None,
        help="仅推理前 N 个 volumes，用于 smoke test",
    )
    parser.add_argument(
        "--use_lora",
        type=lambda x: None if x.lower() == "auto" else x.lower() == "true",
        default=None,
        metavar="auto|true|false",
    )
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=float, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--context_pool_size", type=int, default=2)
    parser.add_argument("--context_max_context_distance", type=int, default=4)
    parser.add_argument("--context_max_neighbor_frames", type=int, default=4)
    parser.add_argument("--context_output_dim", type=int, default=256)
    parser.add_argument("--context_dropout", type=float, default=0.1)
    parser.add_argument("--center_frame_strategy", type=str, default="middle")
    parser.add_argument("--context_feature_level", type=int, default=-1)
    args = parser.parse_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(
        sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
    )
    json_path = os.path.join(args.test_dir, args.annotation_file)

    if args.output_dir is None:
        args.output_dir = _default_output_dir(args.test_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)

    save_png_dir = None
    if args.save_png:
        save_png_dir = args.png_dir or os.path.join(args.output_dir, "pngs")
        os.makedirs(save_png_dir, exist_ok=True)
        logger.info("PNG 输出: %s", save_png_dir)

    save_nii_dir = None
    if args.save_nii:
        save_nii_dir = args.nii_dir or os.path.join(args.output_dir, "nii")
        os.makedirs(save_nii_dir, exist_ok=True)
        logger.info("NIfTI 输出: %s", save_nii_dir)

    logger.info("加载 slice-context 模型 …")
    model = load_context_checkpoint_and_model(
        checkpoint_path=args.checkpoint_path,
        bpe_path=bpe_path,
        device=device_str,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        context_pool_size=args.context_pool_size,
        context_max_context_distance=args.context_max_context_distance,
        context_max_neighbor_frames=args.context_max_neighbor_frames,
        context_output_dim=args.context_output_dim,
        context_dropout=args.context_dropout,
        center_frame_strategy=args.center_frame_strategy,
        context_feature_level=args.context_feature_level,
    )
    transform = _build_transform(args.resize_size)

    logger.info("读取 COCO: %s", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    data = build_volume_groups(coco_data, args.test_dir)
    images_dict = data["images_dict"]
    categories_dict = data["categories_dict"]
    categories_names_dict = data["categories_names_dict"]
    annotations_by_image = data["annotations_by_image"]
    images_by_volume = data["images_by_volume"]

    volume_keys = sorted(images_by_volume.keys())
    if args.limit_volumes is not None:
        volume_keys = volume_keys[: args.limit_volumes]

    volumes_out = []
    for patient_id, frame_id in tqdm(volume_keys, desc="Volumes"):
        slice_list = images_by_volume[(patient_id, frame_id)]
        if not slice_list:
            continue

        patient_name = _resolve_patient_name(patient_id, frame_id, slice_list, images_dict)

        volume_categories = set()
        for img_id, _, _ in slice_list:
            for ann in annotations_by_image[img_id]:
                volume_categories.add(ann["category_id"])
        if not volume_categories:
            continue

        categories_result = run_inference_for_volume_context(
            slice_list=slice_list,
            volume_categories=volume_categories,
            test_dir=args.test_dir,
            images_dict=images_dict,
            categories_dict=categories_dict,
            categories_names_dict=categories_names_dict,
            model=model,
            transform=transform,
            device=device,
            confidence_threshold=args.confidence_threshold,
            context_window_size=args.context_window_size,
            save_png_dir=save_png_dir,
            patient_name=patient_name if save_png_dir else None,
        )

        if save_nii_dir and categories_result:
            try:
                save_volume_as_nii(categories_result, patient_name, save_nii_dir)
            except Exception as e:
                tqdm.write(f"Warning: nii save failed for {patient_name}: {e}")

        volumes_out.append(
            {
                "patient_id": patient_id,
                "frame_id": frame_id,
                "patient_name": patient_name,
                "categories": categories_result,
            }
        )

    result = {
        "volumes": volumes_out,
        "config": {
            "test_dir": args.test_dir,
            "annotation_file": args.annotation_file,
            "checkpoint_path": args.checkpoint_path,
            "context_window_size": args.context_window_size,
        },
    }
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    logger.info("已写入 %s（%d volumes）", out_path, len(volumes_out))


if __name__ == "__main__":
    main()

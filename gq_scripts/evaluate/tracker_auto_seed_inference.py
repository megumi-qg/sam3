"""
ACDC detector-first + tracker-refine 联合推理脚本。

目标：
1. 不使用真实 GT mask 作为 tracker 的 init seed / cond mask
2. 先用 image model 在整套 volume 上对每个类别逐切片预测
3. 从高置信 detector 切片中选出多张 conditioning frames 送给 tracker
4. tracker 负责跨切片传播与补全
5. 最终输出采用“detector 优先、tracker 补空”的混合策略

说明：
- 推理阶段不会读取 GT segmentation 来决定 detector / tracker 条件帧
- 为了与现有离线评测工具兼容，脚本会读取 `frame_annotations.coco.json`
  中的 `images` 元数据来恢复 `image_id / height / width / original_slice_idx`
  映射，但不会消费其中的 `annotations` 字段参与推理
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

import sam3
from sam3.model_builder import build_tracker
from sam3.train.masks_ops import rle_encode

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from batch_inference import load_checkpoint_and_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ACDC_CATEGORY_ID_TO_PROMPT = {
    1: "right ventricle",
    2: "myocardium",
    3: "left ventricle",
}


def _default_test_dir() -> str:
    return "/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/test"


def _default_image_checkpoint() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/full_video_lora_100/checkpoints/"
        "val_acdc_segmentation_coco_eval_segm_AP.pt"
    )


def _default_tracker_checkpoint() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init/checkpoints/"
        "val_acdc_segmentation_coco_eval_segm_AP.pt"
    )


def _default_output_dir() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/"
        "full_sam3_tracker_image_init_auto_seed_test"
    )


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _category_prompts_from_coco(frame_coco: dict) -> Dict[int, str]:
    categories = frame_coco.get("categories") or []
    if categories:
        prompts = {}
        for category in categories:
            category_id = int(category["id"])
            name = category.get("name")
            if not name and isinstance(category.get("names"), list) and category["names"]:
                name = category["names"][0]
            prompts[category_id] = str(name if name else category_id)
        return prompts
    return dict(ACDC_CATEGORY_ID_TO_PROMPT)


def _numpy_slice_to_pil(img_array: np.ndarray) -> Image.Image:
    if img_array.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape={img_array.shape}")

    if img_array.dtype != np.uint8:
        arr = img_array.astype(np.float32)
        arr_min = float(arr.min()) if arr.size else 0.0
        arr_max = float(arr.max()) if arr.size else 0.0
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        img_array = (arr * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(img_array).convert("RGB")


def _build_frame_metadata(frame_coco_json: dict) -> Dict[str, List[dict]]:
    volume_to_frames = defaultdict(list)
    for img in frame_coco_json.get("images", []):
        basename = os.path.basename(img["file_name"])
        volume_name = os.path.splitext(basename)[0]
        volume_to_frames[volume_name].append(
            {
                "image_id": int(img["id"]),
                "height": int(img["height"]),
                "width": int(img["width"]),
                "frame_idx": int(img.get("frame_idx", 0)),
                "original_slice_idx": int(
                    img.get("original_slice_idx", img.get("frame_idx", 0))
                ),
            }
        )
    for volume_name in volume_to_frames:
        volume_to_frames[volume_name].sort(key=lambda x: x["original_slice_idx"])
    return dict(volume_to_frames)


def _tracker_state_from_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    tracker_state = {}
    for key, value in state_dict.items():
        if key.startswith("tracker."):
            tracker_state[key[len("tracker."):]] = value
        elif key.startswith("detector.backbone.vision_backbone."):
            mapped_key = "backbone." + key[len("detector.backbone.") :]
            tracker_state[mapped_key] = value
        else:
            tracker_state[key] = value

    return tracker_state


def load_tracker_predictor(checkpoint_path: str, device: str):
    tracker = build_tracker(
        apply_temporal_disambiguation=False,
        with_backbone=True,
    )
    state_dict = _tracker_state_from_checkpoint(checkpoint_path)
    missing, unexpected = tracker.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("tracker load_state_dict 缺失键（前 20 个）: %s", list(missing)[:20])
    if unexpected:
        logger.warning(
            "tracker load_state_dict 未使用键（前 20 个）: %s",
            list(unexpected)[:20],
        )
    # Sam3TrackerPredictor 默认会在 __init__ 里常驻开启一个全局 bf16 autocast，
    # 这会污染同进程中的 image-model seed 搜索。这里显式退出，后续由本脚本
    # 自己决定在哪些 tracker 计算段使用局部 autocast。
    if hasattr(tracker, "bf16_context"):
        try:
            tracker.bf16_context.__exit__(None, None, None)
        except Exception:
            pass
    tracker = tracker.to(device)
    tracker.eval()
    return tracker


def build_tracker_transform(resolution: int):
    return v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def prepare_volume_inputs(
    npz_path: str,
    frame_meta: List[dict],
    device: str,
    resolution: int,
) -> Tuple[List[Image.Image], torch.Tensor, np.ndarray]:
    with np.load(npz_path) as data:
        volume = data["volume"]
        slice_indices = data.get("slice_indices")
        if slice_indices is None:
            slice_indices = np.arange(volume.shape[0], dtype=np.int32)

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume in {npz_path}, got shape={volume.shape}")

    if len(frame_meta) != volume.shape[0]:
        raise ValueError(
            f"Slice count mismatch for {npz_path}: volume has {volume.shape[0]} slices, "
            f"but frame metadata has {len(frame_meta)} entries."
        )

    expected_slice_indices = [m["original_slice_idx"] for m in frame_meta]
    if list(map(int, slice_indices.tolist())) != expected_slice_indices:
        raise ValueError(
            f"Slice index mismatch for {npz_path}: npz slice_indices={slice_indices.tolist()} "
            f"vs frame_annotations={expected_slice_indices}"
        )

    pil_slices = [_numpy_slice_to_pil(volume[i]) for i in range(volume.shape[0])]

    transform = build_tracker_transform(resolution)
    image_tensors = [
        transform(v2.functional.to_image(img).to(device)) for img in pil_slices
    ]
    image_batch = torch.stack(image_tensors, dim=0)
    return pil_slices, image_batch, slice_indices


def load_volume_pil_slices(
    npz_path: str,
    frame_meta: List[dict],
) -> Tuple[List[Image.Image], np.ndarray]:
    with np.load(npz_path) as data:
        volume = data["volume"]
        slice_indices = data.get("slice_indices")
        if slice_indices is None:
            slice_indices = np.arange(volume.shape[0], dtype=np.int32)

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume in {npz_path}, got shape={volume.shape}")

    if len(frame_meta) != volume.shape[0]:
        raise ValueError(
            f"Slice count mismatch for {npz_path}: volume has {volume.shape[0]} slices, "
            f"but frame metadata has {len(frame_meta)} entries."
        )

    expected_slice_indices = [m["original_slice_idx"] for m in frame_meta]
    if list(map(int, slice_indices.tolist())) != expected_slice_indices:
        raise ValueError(
            f"Slice index mismatch for {npz_path}: npz slice_indices={slice_indices.tolist()} "
            f"vs frame_annotations={expected_slice_indices}"
        )

    pil_slices = [_numpy_slice_to_pil(volume[i]) for i in range(volume.shape[0])]
    return pil_slices, slice_indices


@torch.inference_mode()
def compute_tracker_cached_features(tracker_model, image_batch: torch.Tensor):
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if image_batch.device.type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        backbone_out = tracker_model.forward_image(image_batch)
    cached_features = {}
    num_frames = image_batch.shape[0]
    for frame_idx in range(num_frames):
        cached_features[frame_idx] = (
            image_batch[frame_idx : frame_idx + 1],
            {
                "backbone_fpn": [
                    feat[frame_idx : frame_idx + 1]
                    for feat in backbone_out["backbone_fpn"]
                ],
                "vision_pos_enc": [
                    pos[frame_idx : frame_idx + 1]
                    for pos in backbone_out["vision_pos_enc"]
                ],
            },
        )
    return cached_features


@torch.inference_mode()
def run_image_prompt(processor, image: Image.Image, prompt: str) -> dict:
    if torch.cuda.is_available():
        autocast_ctx = torch.autocast(device_type="cuda", enabled=False)
    else:
        autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

    with autocast_ctx:
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)

    masks = state["masks"]
    scores = state["scores"]
    if len(masks) == 0:
        h, w = image.height, image.width
        return {
            "mask": np.zeros((h, w), dtype=bool),
            "score": 0.0,
            "area_px": 0,
            "area_ratio": 0.0,
        }

    best_idx = int(scores.argmax().item())
    score = float(scores[best_idx].item())
    pred_mask_tensor = masks[best_idx]
    if pred_mask_tensor.dim() == 3:
        pred_mask_tensor = pred_mask_tensor.squeeze(0)
    pred_mask = pred_mask_tensor.detach().cpu().numpy().astype(bool)
    if pred_mask.shape != (image.height, image.width):
        pred_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                (image.width, image.height), Image.NEAREST
            )
        ) > 0

    area_px = int(pred_mask.sum())
    area_ratio = float(area_px / (image.height * image.width))
    return {
        "mask": pred_mask,
        "score": score,
        "area_px": area_px,
        "area_ratio": area_ratio,
    }


def collect_detector_predictions_for_category(
    processor,
    pil_slices: List[Image.Image],
    prompt: str,
):
    predictions = []
    for frame_idx, image in enumerate(pil_slices):
        result = run_image_prompt(processor, image, prompt)
        predictions.append(
            {
                "frame_idx": frame_idx,
                "score": result["score"],
                "area_px": result["area_px"],
                "area_ratio": result["area_ratio"],
                "mask": result["mask"],
            }
        )
    return predictions


def select_conditioning_frames(
    detector_predictions: List[dict],
    *,
    detector_condition_threshold: float,
    min_mask_area_px: int,
    min_mask_area_ratio: float,
    max_cond_frames: int,
    min_cond_frame_gap: int,
    fallback_to_best_seed: bool,
    selection_strategy: str,
):
    valid = [
        pred
        for pred in detector_predictions
        if pred["score"] >= detector_condition_threshold
        and pred["area_px"] >= min_mask_area_px
        and pred["area_ratio"] >= min_mask_area_ratio
    ]
    if selection_strategy == "topk_score":
        valid.sort(key=lambda x: x["score"], reverse=True)
    elif selection_strategy == "earliest_above_threshold":
        valid.sort(key=lambda x: x["frame_idx"])
    elif selection_strategy == "best_single":
        valid.sort(key=lambda x: x["score"], reverse=True)
        if valid:
            valid = [valid[0]]
    elif selection_strategy == "largest_single":
        valid.sort(
            key=lambda x: (x["area_px"], x["score"], -abs(x["frame_idx"])),
            reverse=True,
        )
        if valid:
            valid = [valid[0]]
    else:
        raise ValueError(f"Unsupported selection_strategy: {selection_strategy}")

    selected = []
    for pred in valid:
        if any(
            abs(pred["frame_idx"] - kept["frame_idx"]) < min_cond_frame_gap
            for kept in selected
        ):
            continue
        selected.append(pred)
        if len(selected) >= max_cond_frames:
            break

    selection_reason = (
        "thresholded_single_cond"
        if selection_strategy in {"best_single", "largest_single"}
        else f"thresholded_{selection_strategy}"
    )
    if not selected and fallback_to_best_seed and detector_predictions:
        best_global = max(detector_predictions, key=lambda x: x["score"])
        if (
            best_global["area_px"] >= min_mask_area_px
            and best_global["area_ratio"] >= min_mask_area_ratio
        ):
            selected = [best_global]
            selection_reason = "fallback_best"

    selected.sort(key=lambda x: x["frame_idx"])
    return selected, selection_reason


def build_conditioning_report(
    prompt: str,
    detector_predictions: List[dict],
    cond_frames: List[dict],
    selection_reason: str,
):
    cond_indices = {item["frame_idx"] for item in cond_frames}
    per_slice_debug = []
    for pred in detector_predictions:
        per_slice_debug.append(
            {
                "frame_idx": pred["frame_idx"],
                "score": pred["score"],
                "area_px": pred["area_px"],
                "area_ratio": pred["area_ratio"],
                "is_conditioning_frame": pred["frame_idx"] in cond_indices,
            }
        )

    return {
        "prompt": prompt,
        "selected": len(cond_frames) > 0,
        "selection_reason": selection_reason,
        "num_conditioning_frames": len(cond_frames),
        "conditioning_frames": [
            {
                "frame_idx": int(item["frame_idx"]),
                "score": float(item["score"]),
                "area_px": int(item["area_px"]),
                "area_ratio": float(item["area_ratio"]),
            }
            for item in cond_frames
        ],
        "per_slice": per_slice_debug,
    }


@torch.inference_mode()
def run_tracker_single_direction(
    tracker_model,
    cached_features,
    video_height: int,
    video_width: int,
    num_frames: int,
    cond_frames: List[dict],
    reverse: bool,
):
    if not cond_frames:
        return {}

    start_frame_idx = (
        max(frame["frame_idx"] for frame in cond_frames)
        if reverse
        else min(frame["frame_idx"] for frame in cond_frames)
    )
    state = tracker_model.init_state(
        video_height=video_height,
        video_width=video_width,
        num_frames=num_frames,
        cached_features=dict(cached_features),
    )
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if state["device"].type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        for cond in cond_frames:
            tracker_model.add_new_mask(
                state,
                frame_idx=int(cond["frame_idx"]),
                obj_id=1,
                mask=torch.from_numpy(cond["mask"]),
            )
        tracker_model.propagate_in_video_preflight(state, run_mem_encoder=True)

        outputs = {}
        for frame_idx, obj_ids, _, video_res_masks, obj_scores in tracker_model.propagate_in_video(
            state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=None,
            reverse=reverse,
            tqdm_disable=True,
            obj_ids=None,
            run_mem_encoder=True,
            propagate_preflight=False,
        ):
            obj_pos = obj_ids.index(1)
            score_tensor = obj_scores[obj_pos]
            score = float(score_tensor.sigmoid().flatten()[0].item())
            mask = (video_res_masks[obj_pos, 0] > 0).detach().cpu().numpy().astype(bool)
            outputs[frame_idx] = {"score": score, "mask": mask}

    return outputs


def run_tracker_bidirectional(
    tracker_model,
    cached_features,
    video_height: int,
    video_width: int,
    num_frames: int,
    cond_frames: List[dict],
):
    if not cond_frames:
        return {}

    forward_outputs = run_tracker_single_direction(
        tracker_model=tracker_model,
        cached_features=cached_features,
        video_height=video_height,
        video_width=video_width,
        num_frames=num_frames,
        cond_frames=cond_frames,
        reverse=False,
    )
    reverse_outputs = run_tracker_single_direction(
        tracker_model=tracker_model,
        cached_features=cached_features,
        video_height=video_height,
        video_width=video_width,
        num_frames=num_frames,
        cond_frames=cond_frames,
        reverse=True,
    )
    merged = dict(forward_outputs)
    for frame_idx, out in reverse_outputs.items():
        prev = merged.get(frame_idx)
        if prev is None or out["score"] > prev["score"]:
            merged[frame_idx] = out
    return merged


def run_tracker_with_mode(
    tracker_model,
    cached_features,
    video_height: int,
    video_width: int,
    num_frames: int,
    cond_frames: List[dict],
    propagation_mode: str,
):
    if propagation_mode == "bidirectional":
        return run_tracker_bidirectional(
            tracker_model=tracker_model,
            cached_features=cached_features,
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            cond_frames=cond_frames,
        )
    if propagation_mode == "forward_only":
        return run_tracker_single_direction(
            tracker_model=tracker_model,
            cached_features=cached_features,
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            cond_frames=cond_frames,
            reverse=False,
        )
    if propagation_mode == "reverse_only":
        return run_tracker_single_direction(
            tracker_model=tracker_model,
            cached_features=cached_features,
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            cond_frames=cond_frames,
            reverse=True,
        )
    raise ValueError(f"Unsupported propagation_mode: {propagation_mode}")


def choose_final_output_for_frame(
    detector_pred: dict,
    tracker_pred: Optional[dict],
    *,
    final_output_mode: str,
    detector_output_threshold: float,
    tracker_detection_threshold: float,
    allow_low_score_detector_fallback: bool,
    allow_low_score_tracker_fallback: bool,
):
    detector_ok = (
        detector_pred is not None
        and detector_pred["score"] >= detector_output_threshold
        and detector_pred["area_px"] > 0
    )
    tracker_ok = (
        tracker_pred is not None
        and (
            tracker_detection_threshold <= 0
        or tracker_pred["score"] > tracker_detection_threshold
        )
    )

    if final_output_mode == "detector_only":
        if detector_ok:
            return {
                "source": "detector",
                "score": float(detector_pred["score"]),
                "mask": detector_pred["mask"],
            }
        if (
            allow_low_score_detector_fallback
            and detector_pred is not None
            and detector_pred["area_px"] > 0
        ):
            return {
                "source": "detector_fallback",
                "score": float(detector_pred["score"]),
                "mask": detector_pred["mask"],
            }
        return None

    if final_output_mode == "tracker_only":
        if tracker_ok:
            return {
                "source": "tracker",
                "score": float(tracker_pred["score"]),
                "mask": tracker_pred["mask"],
            }
        if allow_low_score_tracker_fallback and tracker_pred is not None:
            return {
                "source": "tracker_fallback",
                "score": float(tracker_pred["score"]),
                "mask": tracker_pred["mask"],
            }
        return None

    if final_output_mode == "tracker_first" and tracker_ok:
        return {
            "source": "tracker",
            "score": float(tracker_pred["score"]),
            "mask": tracker_pred["mask"],
        }
    if detector_ok:
        return {
            "source": "detector",
            "score": float(detector_pred["score"]),
            "mask": detector_pred["mask"],
        }
    if tracker_ok:
        return {
            "source": "tracker",
            "score": float(tracker_pred["score"]),
            "mask": tracker_pred["mask"],
        }
    if final_output_mode == "tracker_first" and allow_low_score_tracker_fallback and tracker_pred is not None:
        return {
            "source": "tracker_fallback",
            "score": float(tracker_pred["score"]),
            "mask": tracker_pred["mask"],
        }
    if (
        allow_low_score_detector_fallback
        and detector_pred is not None
        and detector_pred["area_px"] > 0
    ):
        return {
            "source": "detector_fallback",
            "score": float(detector_pred["score"]),
            "mask": detector_pred["mask"],
        }
    if allow_low_score_tracker_fallback and tracker_pred is not None:
        return {
            "source": "tracker_fallback",
            "score": float(tracker_pred["score"]),
            "mask": tracker_pred["mask"],
        }
    return None


def mask_to_coco_result(
    image_id: int,
    category_id: int,
    mask: np.ndarray,
    score: float,
) -> dict:
    mask_tensor = torch.from_numpy(mask[None]).to(torch.bool)
    rle = rle_encode(mask_tensor)[0]
    h, w = mask.shape
    area = float(mask.sum() / (h * w))
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": rle,
        "score": float(score),
        "area": area,
        "bbox": [0.0, 0.0, 0.0, 0.0],
    }


def main():
    parser = argparse.ArgumentParser(
        description="ACDC detector-first + tracker-refine 联合推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_dir", type=str, default=_default_test_dir())
    parser.add_argument(
        "--frame_annotation_file",
        type=str,
        default="frame_annotations.coco.json",
        help="仅使用其中 images 元数据映射 image_id，不用 annotations 做 seed",
    )
    parser.add_argument(
        "--image_checkpoint_path",
        type=str,
        default=_default_image_checkpoint(),
    )
    parser.add_argument(
        "--tracker_checkpoint_path",
        type=str,
        default=_default_tracker_checkpoint(),
    )
    parser.add_argument("--output_dir", type=str, default=_default_output_dir())
    parser.add_argument("--output_name", type=str, default="coco_predictions_segm.json")
    parser.add_argument("--resize_size", type=int, default=1008)
    parser.add_argument(
        "--detector_condition_threshold",
        type=float,
        default=0.7,
        help="detector mask 可作为 tracker conditioning frame 的阈值",
    )
    parser.add_argument(
        "--detector_output_threshold",
        type=float,
        default=0.7,
        help="最终输出中优先采用 detector mask 的阈值",
    )
    parser.add_argument(
        "--final_output_mode",
        type=str,
        default="detector_first",
        choices=["detector_first", "tracker_first", "tracker_only", "detector_only"],
        help=(
            "最终 mask 来源策略。detector_first 是原始混合策略；"
            "tracker_only 用于评估 tracker 本体贡献。"
        ),
    )
    parser.add_argument("--tracker_detection_threshold", type=float, default=0.7)
    parser.add_argument("--min_mask_area_px", type=int, default=32)
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.0)
    parser.add_argument("--max_cond_frames", type=int, default=4)
    parser.add_argument("--min_cond_frame_gap", type=int, default=1)
    parser.add_argument(
        "--conditioning_selection_strategy",
        type=str,
        default="topk_score",
        choices=[
            "topk_score",
            "earliest_above_threshold",
            "best_single",
            "largest_single",
        ],
        help=(
            "conditioning frame 选择策略。"
            "`earliest_above_threshold` 更接近当前 tracker 训练适配器的单起始帧范式；"
            "`best_single` 用单个最高分 seed 做传播；"
            "`largest_single` 用单个面积最大的高置信 seed 做传播。"
        ),
    )
    parser.add_argument(
        "--propagation_mode",
        type=str,
        default="bidirectional",
        choices=["bidirectional", "forward_only", "reverse_only"],
        help=(
            "tracker 传播方向。"
            "`forward_only` 更接近当前训练里从初始条件帧往后传播的范式。"
        ),
    )
    parser.add_argument("--fallback_to_best_seed", action="store_true")
    parser.add_argument(
        "--allow_low_score_detector_fallback",
        action="store_true",
        help=(
            "允许 detector 在 score < detector_output_threshold 时仍以 fallback 形式进入最终输出。"
            "默认关闭，以避免把低置信伪阳性直接写入结果。"
        ),
    )
    parser.add_argument(
        "--allow_low_score_tracker_fallback",
        action="store_true",
        help=(
            "允许 tracker 在 score < tracker_detection_threshold 时仍以 fallback 形式进入最终输出。"
        ),
    )
    parser.add_argument("--limit_volumes", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.device != "cuda":
        raise RuntimeError(
            "当前自动 seed tracker 推理脚本要求 CUDA。请在有 GPU 的环境下运行。"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False，无法运行 GPU 推理。")

    frame_json_path = os.path.join(args.test_dir, args.frame_annotation_file)
    if not os.path.isfile(frame_json_path):
        raise FileNotFoundError(f"找不到 frame annotation file: {frame_json_path}")

    frame_coco = _load_json(frame_json_path)
    volume_to_frames = _build_frame_metadata(frame_coco)
    category_id_to_prompt = _category_prompts_from_coco(frame_coco)
    volume_names = sorted(volume_to_frames.keys())
    if args.limit_volumes is not None:
        volume_names = volume_names[: args.limit_volumes]

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    logger.info("加载 image model: %s", args.image_checkpoint_path)
    _, image_processor = load_checkpoint_and_model(
        args.image_checkpoint_path,
        bpe_path,
        args.device,
        args.resize_size,
        confidence_threshold=0.0,
        use_lora=None,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    predictions = []
    detector_candidate_predictions = []
    tracker_candidate_predictions = []
    conditioning_report = {}
    detector_plan = {}
    final_selection_report = {}

    logger.info("第一阶段：运行 detector 并选择 conditioning frames ...")
    for volume_name in tqdm(volume_names, desc="Detector volumes"):
        npz_path = os.path.join(args.test_dir, "volumes", f"{volume_name}.npz")
        if not os.path.isfile(npz_path):
            logger.warning("跳过缺失 volume: %s", npz_path)
            continue

        frame_meta = volume_to_frames[volume_name]
        pil_slices, _ = load_volume_pil_slices(npz_path=npz_path, frame_meta=frame_meta)

        conditioning_report[volume_name] = {}
        detector_plan[volume_name] = {}
        for category_id, prompt in category_id_to_prompt.items():
            detector_predictions = collect_detector_predictions_for_category(
                processor=image_processor,
                pil_slices=pil_slices,
                prompt=prompt,
            )
            cond_frames, selection_reason = select_conditioning_frames(
                detector_predictions,
                detector_condition_threshold=args.detector_condition_threshold,
                min_mask_area_px=args.min_mask_area_px,
                min_mask_area_ratio=args.min_mask_area_ratio,
                max_cond_frames=args.max_cond_frames,
                min_cond_frame_gap=args.min_cond_frame_gap,
                fallback_to_best_seed=args.fallback_to_best_seed,
                selection_strategy=args.conditioning_selection_strategy,
            )

            detector_plan[volume_name][category_id] = detector_predictions
            conditioning_report[volume_name][str(category_id)] = build_conditioning_report(
                prompt=prompt,
                detector_predictions=detector_predictions,
                cond_frames=cond_frames,
                selection_reason=selection_reason,
            )

            if cond_frames:
                detector_plan[volume_name][category_id] = detector_predictions

    del image_processor
    torch.cuda.empty_cache()

    logger.info("第二阶段：加载 tracker 并做双向传播 ...")
    logger.info("加载 tracker model: %s", args.tracker_checkpoint_path)
    tracker_model = load_tracker_predictor(args.tracker_checkpoint_path, args.device)

    for volume_name in tqdm(volume_names, desc="Track volumes"):
        if not detector_plan.get(volume_name):
            continue

        npz_path = os.path.join(args.test_dir, "volumes", f"{volume_name}.npz")
        if not os.path.isfile(npz_path):
            continue

        frame_meta = volume_to_frames[volume_name]
        pil_slices, image_batch, _ = prepare_volume_inputs(
            npz_path=npz_path,
            frame_meta=frame_meta,
            device=args.device,
            resolution=args.resize_size,
        )
        del pil_slices
        cached_features = compute_tracker_cached_features(tracker_model, image_batch)

        video_height = int(frame_meta[0]["height"])
        video_width = int(frame_meta[0]["width"])
        num_frames = len(frame_meta)

        for category_id, prompt in category_id_to_prompt.items():
            detector_predictions = detector_plan[volume_name].get(category_id, [])
            cond_frames, _ = select_conditioning_frames(
                detector_predictions,
                detector_condition_threshold=args.detector_condition_threshold,
                min_mask_area_px=args.min_mask_area_px,
                min_mask_area_ratio=args.min_mask_area_ratio,
                max_cond_frames=args.max_cond_frames,
                min_cond_frame_gap=args.min_cond_frame_gap,
                fallback_to_best_seed=args.fallback_to_best_seed,
                selection_strategy=args.conditioning_selection_strategy,
            )

            tracker_outputs = run_tracker_with_mode(
                tracker_model=tracker_model,
                cached_features=cached_features,
                video_height=video_height,
                video_width=video_width,
                num_frames=num_frames,
                cond_frames=cond_frames,
                propagation_mode=args.propagation_mode,
            )
            class_report = final_selection_report.setdefault(volume_name, {}).setdefault(
                str(category_id),
                {
                    "prompt": prompt,
                    "source_counts": {
                        "detector": 0,
                        "tracker": 0,
                        "detector_fallback": 0,
                        "tracker_fallback": 0,
                        "none": 0,
                    },
                    "num_low_score_nonempty_detector_frames": 0,
                    "num_low_score_nonempty_tracker_frames": 0,
                },
            )

            for local_frame_idx, meta in enumerate(frame_meta):
                detector_pred = (
                    detector_predictions[local_frame_idx]
                    if local_frame_idx < len(detector_predictions)
                    else None
                )
                tracker_pred = tracker_outputs.get(local_frame_idx)
                final_out = choose_final_output_for_frame(
                    detector_pred,
                    tracker_pred,
                    final_output_mode=args.final_output_mode,
                    detector_output_threshold=args.detector_output_threshold,
                    tracker_detection_threshold=args.tracker_detection_threshold,
                    allow_low_score_detector_fallback=args.allow_low_score_detector_fallback,
                    allow_low_score_tracker_fallback=args.allow_low_score_tracker_fallback,
                )
                if (
                    detector_pred is not None
                    and detector_pred["area_px"] > 0
                ):
                    detector_candidate_predictions.append(
                        mask_to_coco_result(
                            image_id=meta["image_id"],
                            category_id=category_id,
                            mask=detector_pred["mask"],
                            score=detector_pred["score"],
                        )
                    )
                    if detector_pred["score"] < args.detector_output_threshold:
                        class_report["num_low_score_nonempty_detector_frames"] += 1
                if (
                    tracker_pred is not None
                    and tracker_pred["mask"].any()
                ):
                    tracker_candidate_predictions.append(
                        mask_to_coco_result(
                            image_id=meta["image_id"],
                            category_id=category_id,
                            mask=tracker_pred["mask"],
                            score=tracker_pred["score"],
                        )
                    )
                    if tracker_pred["score"] < args.tracker_detection_threshold:
                        class_report["num_low_score_nonempty_tracker_frames"] += 1
                if final_out is None:
                    class_report["source_counts"]["none"] += 1
                    continue
                class_report["source_counts"][final_out["source"]] += 1

                predictions.append(
                    mask_to_coco_result(
                        image_id=meta["image_id"],
                        category_id=category_id,
                        mask=final_out["mask"],
                        score=final_out["score"],
                    )
                )

    predictions.sort(key=lambda x: (x["image_id"], x["category_id"]))
    detector_candidate_predictions.sort(key=lambda x: (x["image_id"], x["category_id"]))
    tracker_candidate_predictions.sort(key=lambda x: (x["image_id"], x["category_id"]))

    pred_path = os.path.join(args.output_dir, args.output_name)
    with open(pred_path, "w") as f:
        json.dump(predictions, f)
    detector_pred_path = os.path.join(args.output_dir, "detector_predictions_segm.json")
    with open(detector_pred_path, "w") as f:
        json.dump(detector_candidate_predictions, f)
    tracker_pred_path = os.path.join(args.output_dir, "tracker_predictions_segm.json")
    with open(tracker_pred_path, "w") as f:
        json.dump(tracker_candidate_predictions, f)

    seed_report_path = os.path.join(args.output_dir, "seed_selection_report.json")
    with open(seed_report_path, "w") as f:
        json.dump(conditioning_report, f, indent=2)
    final_selection_report_path = os.path.join(
        args.output_dir, "final_selection_report.json"
    )
    with open(final_selection_report_path, "w") as f:
        json.dump(final_selection_report, f, indent=2)

    logger.info("detector-first + tracker-refine 推理完成")
    logger.info("predictions: %s", pred_path)
    logger.info("detector candidates: %s", detector_pred_path)
    logger.info("tracker candidates:  %s", tracker_pred_path)
    logger.info("seed report:  %s", seed_report_path)
    logger.info("final report: %s", final_selection_report_path)
    logger.info("prediction count: %d", len(predictions))


if __name__ == "__main__":
    main()

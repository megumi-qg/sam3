#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a high-precision pseudo-seed video COCO JSON for weak tracker training.

The output keeps scribble supervision and pseudo conditioning separate:
- `segmentations`: original target-class scribbles, used by weak loss.
- `valid_masks`: original scribble valid regions, used as supervision gate.
- `seed_segmentations`: high-confidence pseudo masks, used as tracker condition.

No full labels are read by this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "gq_scripts" / "evaluate"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from batch_inference import load_checkpoint_and_model  # noqa: E402
from tracker_auto_seed_inference import run_image_prompt  # noqa: E402


ACDC_CATEGORY_ID_TO_PROMPT = {
    1: "right ventricle",
    2: "myocardium",
    3: "left ventricle",
}


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _rle_to_mask(rle: Optional[dict], height: int, width: int) -> np.ndarray:
    if not rle:
        return np.zeros((height, width), dtype=bool)
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.sum(axis=2) > 0
    return mask.astype(bool)


def _mask_to_rle(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    mask = np.asfortranarray(mask.astype(np.uint8))
    if int(mask.sum()) == 0:
        return None
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _empty_rle(height: int, width: int) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _mask_to_bbox(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1),
    ]


def _numpy_slice_to_pil(img_array: np.ndarray) -> Image.Image:
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


def _load_volume_slices(split_dir: str, video: dict) -> List[Image.Image]:
    npz_path = os.path.join(split_dir, video["npz_path"])
    with np.load(npz_path) as data:
        volume = data["volume"]
    return [_numpy_slice_to_pil(volume[i]) for i in range(volume.shape[0])]


def _build_ann_lookup(annotations: List[dict]) -> Dict[tuple[int, int], dict]:
    return {(int(ann["video_id"]), int(ann["category_id"])): ann for ann in annotations}


def _passes_seed_filter(
    pred: dict,
    *,
    scribble_mask: np.ndarray,
    valid_mask: np.ndarray,
    score_threshold: float,
    min_area_px: int,
    max_area_ratio: float,
    min_scribble_recall: float,
    max_other_scribble_overlap_px: int,
) -> tuple[bool, str]:
    pred_mask = pred["mask"].astype(bool)
    area_px = int(pred_mask.sum())
    if pred["score"] < score_threshold:
        return False, "low_score"
    if area_px < min_area_px:
        return False, "small_area"
    if pred["area_ratio"] > max_area_ratio:
        return False, "large_area"

    scribble_px = int(scribble_mask.sum())
    if scribble_px <= 0:
        return False, "no_positive_scribble"
    recall = float((pred_mask & scribble_mask).sum() / max(1, scribble_px))
    if recall < min_scribble_recall:
        return False, "misses_positive_scribble"

    other_scribble = valid_mask & (~scribble_mask)
    other_overlap = int((pred_mask & other_scribble).sum())
    if other_overlap > max_other_scribble_overlap_px:
        return False, "overlaps_other_scribble"

    return True, "accepted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scribble-compatible pseudo seed bank for tracker training."
    )
    parser.add_argument(
        "--split_dir",
        default="/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/train",
    )
    parser.add_argument(
        "--scribble_video_json",
        default=(
            "/home/gaoqi/dataset/using/acdc/processed/"
            "sam3_video_npz_coco_fullframes_100/train/"
            "scribble_tmi_video_annotations.coco.json"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default=(
            "/home/gaoqi/sam3/gq_experiment/acdc/scribble_video_lora_100/"
            "checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt"
        ),
    )
    parser.add_argument(
        "--output_json",
        default=(
            "/home/gaoqi/sam3/gq_experiment/acdc/scribble_video_lora_100/"
            "pseudo_seed_bank/scribble_tmi_pseudo_seed_video_annotations.coco.json"
        ),
    )
    parser.add_argument("--resize_size", type=int, default=1008)
    parser.add_argument("--score_threshold", type=float, default=0.97)
    parser.add_argument("--min_area_px", type=int, default=32)
    parser.add_argument("--max_area_ratio", type=float, default=0.35)
    parser.add_argument("--min_scribble_recall", type=float, default=0.8)
    parser.add_argument("--max_other_scribble_overlap_px", type=int, default=0)
    parser.add_argument("--limit_volumes", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    scribble_coco = _load_json(args.scribble_video_json)
    videos = list(scribble_coco["videos"])
    if args.limit_volumes is not None:
        videos = videos[: args.limit_volumes]
    ann_lookup = _build_ann_lookup(scribble_coco["annotations"])

    bpe_path = _REPO_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    logging.info("Loading scribble image model: %s", args.checkpoint_path)
    _, processor = load_checkpoint_and_model(
        args.checkpoint_path,
        str(bpe_path),
        args.device,
        args.resize_size,
        confidence_threshold=0.0,
        use_lora=None,
    )

    output_annotations = []
    report = {
        "config": vars(args),
        "num_videos": len(videos),
        "num_input_annotations": len(scribble_coco["annotations"]),
        "num_output_annotations": 0,
        "num_seed_frames": 0,
        "reject_reasons": {},
        "per_annotation": [],
    }

    out_ann_id = 0
    for video in tqdm(videos, desc="Volumes"):
        pil_slices = _load_volume_slices(args.split_dir, video)
        video_id = int(video["id"])
        height = int(video["height"])
        width = int(video["width"])

        for category in scribble_coco["categories"]:
            category_id = int(category["id"])
            ann = ann_lookup.get((video_id, category_id))
            if ann is None:
                continue

            # Prefer the dataset-local category name. Falling back by id is only
            # valid for legacy ACDC JSONs without category names.
            prompt = str(
                category.get("name")
                or ACDC_CATEGORY_ID_TO_PROMPT.get(category_id, category_id)
            )
            frame_indices = [int(x) for x in ann.get("frame_indices", [])]
            if not frame_indices:
                continue

            # Run only on scribble-positive frames; these are the frames where
            # weak supervision is available for this category.
            pred_by_frame = {}
            for frame_idx in frame_indices:
                pred_by_frame[frame_idx] = run_image_prompt(
                    processor, pil_slices[frame_idx], prompt
                )

            seed_segmentations = []
            seed_scores = []
            accepted = 0
            reject_reasons = {}
            for list_idx, frame_idx in enumerate(frame_indices):
                scribble_mask = _rle_to_mask(
                    ann["segmentations"][list_idx], height, width
                )
                valid_mask = _rle_to_mask(ann["valid_masks"][list_idx], height, width)
                pred = pred_by_frame[frame_idx]
                keep, reason = _passes_seed_filter(
                    pred,
                    scribble_mask=scribble_mask,
                    valid_mask=valid_mask,
                    score_threshold=args.score_threshold,
                    min_area_px=args.min_area_px,
                    max_area_ratio=args.max_area_ratio,
                    min_scribble_recall=args.min_scribble_recall,
                    max_other_scribble_overlap_px=args.max_other_scribble_overlap_px,
                )
                if keep:
                    seed_rle = _mask_to_rle(pred["mask"])
                    seed_segmentations.append(seed_rle if seed_rle is not None else [])
                    seed_scores.append(float(pred["score"]))
                    accepted += 1
                else:
                    # Explicit empty seed: prevents weak tracker training from
                    # silently falling back to the scribble mask as condition.
                    seed_segmentations.append(_empty_rle(height, width))
                    seed_scores.append(float(pred["score"]))
                    reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                    report["reject_reasons"][reason] = (
                        report["reject_reasons"].get(reason, 0) + 1
                    )

            if accepted == 0:
                report["per_annotation"].append(
                    {
                        "video_id": video_id,
                        "video_name": video["video_name"],
                        "category_id": category_id,
                        "accepted": 0,
                        "num_frames": len(frame_indices),
                        "reject_reasons": reject_reasons,
                    }
                )
                continue

            out_ann = {
                "id": out_ann_id,
                "video_id": video_id,
                "category_id": category_id,
                "iscrowd": int(ann.get("iscrowd", 0)),
                "frame_indices": frame_indices,
                "bboxes": ann["bboxes"],
                "segmentations": ann["segmentations"],
                "valid_masks": ann["valid_masks"],
                "seed_segmentations": seed_segmentations,
                "seed_scores": seed_scores,
                "areas": ann["areas"],
            }
            output_annotations.append(out_ann)
            out_ann_id += 1
            report["num_seed_frames"] += accepted
            report["per_annotation"].append(
                {
                    "video_id": video_id,
                    "video_name": video["video_name"],
                    "category_id": category_id,
                    "accepted": accepted,
                    "num_frames": len(frame_indices),
                    "reject_reasons": reject_reasons,
                }
            )

    out_coco = {
        "info": {
            "description": "High-confidence scribble-model pseudo seeds for weak tracker training",
            "source_scribble_video_json": args.scribble_video_json,
            "source_checkpoint": args.checkpoint_path,
            "score_threshold": args.score_threshold,
        },
        "videos": scribble_coco["videos"],
        "annotations": output_annotations,
        "categories": scribble_coco["categories"],
    }
    report["num_output_annotations"] = len(output_annotations)
    _write_json(args.output_json, out_coco)
    report_path = os.path.splitext(args.output_json)[0] + "_report.json"
    _write_json(report_path, report)
    logging.info("Saved pseudo seed JSON: %s", args.output_json)
    logging.info("Saved report: %s", report_path)
    logging.info(
        "Accepted %d seed frames across %d annotations",
        report["num_seed_frames"],
        report["num_output_annotations"],
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from sam3.train.utils.distributed import is_main_process


def _decode_rle_mask(rle: dict, height: int, width: int) -> np.ndarray:
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.sum(axis=2) > 0
    mask = mask.astype(bool)
    if mask.shape == (height, width):
        return mask
    return (
        np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
        )
        > 0
    )


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = int(pred.sum() + gt.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def _category_name(category: dict) -> str:
    if isinstance(category.get("names"), list) and category["names"]:
        return str(category["names"][0])
    return str(category.get("name", category["id"]))


class VideoDiceEvaluator:
    """Compute lightweight 3D Dice/IoU from a dumped COCO result file.

    This mirrors the Dice aggregation used by
    ``gq_scripts/evaluate/evaluate_tracker_coco_predictions.py`` but returns
    training meters instead of writing a report.  It intentionally skips HD95
    and NSD so validation checkpointing stays cheap.
    """

    def __init__(
        self,
        gt_path: str,
        prefix: str = "",
        iou_type: str = "segm",
    ):
        if iou_type != "segm":
            raise ValueError("VideoDiceEvaluator currently supports only segm results")
        self.gt_path = gt_path
        self.prefix = prefix

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        logging.info("VideoDice evaluator: Loading predictions from %s", dumped_file)
        with open(dumped_file, "r") as f:
            predictions = json.load(f)
        with open(self.gt_path, "r") as f:
            coco_data = json.load(f)

        pred_index = self._build_prediction_index(predictions)
        data = self._build_video_volume_groups(coco_data)
        gt_index = self._build_gt_index(coco_data)

        category_dices = defaultdict(list)
        category_ious = defaultdict(list)
        case_dices = []
        case_ious = []

        for volume_id, slice_list in data["images_by_volume"].items():
            volume_dices = []
            volume_ious = []
            volume_categories = set()
            for img_id, _ in slice_list:
                volume_categories.update(
                    ann["category_id"]
                    for ann in data["annotations_by_image"].get(img_id, [])
                )

            for category_id in sorted(volume_categories):
                pred_slices = []
                gt_slices = []
                for img_id, _ in slice_list:
                    img = data["images_dict"][img_id]
                    height = int(img["height"])
                    width = int(img["width"])
                    gt_ann = gt_index.get((int(img_id), int(category_id)))
                    if gt_ann is None:
                        gt_mask = np.zeros((height, width), dtype=bool)
                    else:
                        gt_mask = _decode_rle_mask(gt_ann["segmentation"], height, width)

                    pred = pred_index.get((int(img_id), int(category_id)))
                    if pred is None:
                        pred_mask = np.zeros((height, width), dtype=bool)
                    else:
                        pred_mask = _decode_rle_mask(
                            pred["segmentation"], height, width
                        )
                    pred_slices.append(pred_mask)
                    gt_slices.append(gt_mask)

                pred_3d = np.stack(pred_slices, axis=0)
                gt_3d = np.stack(gt_slices, axis=0)
                dice = _dice(pred_3d, gt_3d)
                iou = _iou(pred_3d, gt_3d)
                class_name = data["categories_dict"][category_id]
                category_dices[class_name].append(dice)
                category_ious[class_name].append(iou)
                volume_dices.append(dice)
                volume_ious.append(iou)

            if volume_dices:
                case_dices.append(float(np.mean(volume_dices)))
                case_ious.append(float(np.mean(volume_ious)))

        prefix = f"{self.prefix}_" if self.prefix else ""
        outs = {
            f"{prefix}dice": float(np.nanmean(case_dices)) if case_dices else float("nan"),
            f"{prefix}iou": float(np.nanmean(case_ious)) if case_ious else float("nan"),
        }
        for class_name, values in category_dices.items():
            key = class_name.replace(" ", "_")
            outs[f"{prefix}{key}_dice"] = float(np.nanmean(values))
        return outs

    @staticmethod
    def _build_prediction_index(predictions):
        pred_index = {}
        for pred in predictions:
            key = (int(pred["image_id"]), int(pred["category_id"]))
            previous = pred_index.get(key)
            if previous is None or float(pred.get("score", 0.0)) > float(
                previous.get("score", 0.0)
            ):
                pred_index[key] = pred
        return pred_index

    @staticmethod
    def _build_gt_index(coco_data: dict) -> Dict[Tuple[int, int], dict]:
        return {
            (int(ann["image_id"]), int(ann["category_id"])): ann
            for ann in coco_data.get("annotations", [])
        }

    @staticmethod
    def _build_video_volume_groups(coco_data: dict):
        images_dict = {int(img["id"]): img for img in coco_data["images"]}
        categories_dict = {
            int(cat["id"]): _category_name(cat) for cat in coco_data["categories"]
        }
        annotations_by_image = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            annotations_by_image[int(ann["image_id"])].append(ann)

        images_by_volume = defaultdict(list)
        for img_id, img_info in images_dict.items():
            volume_id = int(img_info.get("video_id", img_id))
            slice_idx = int(img_info.get("original_slice_idx", img_info.get("frame_idx", 0)))
            images_by_volume[volume_id].append((img_id, slice_idx))
        for volume_id in images_by_volume:
            images_by_volume[volume_id].sort(key=lambda item: item[1])

        return {
            "images_dict": images_dict,
            "categories_dict": categories_dict,
            "annotations_by_image": annotations_by_image,
            "images_by_volume": images_by_volume,
        }

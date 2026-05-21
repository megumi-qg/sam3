"""
从 tracker 导出的 COCO 预测 JSON 计算 3D 指标。

适用场景：
- 使用 `sam3/train/train.py -c configs/acdc/full_sam3_tracker_image_init_test.yaml`
  跑完 test 后，会在 dump 目录生成 `coco_predictions_segm.json`
- 本脚本读取该 JSON，并按 volume / category / slice 重组，计算
  Dice、IoU、HD95、NSD

示例：

    python gq_scripts/evaluate/evaluate_tracker_coco_predictions.py \
        --predictions_json /home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_test/dumps/acdc/coco_predictions_segm.json \
        --test_dir /home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/test \
        --dataset_name ACDC
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_TOOL_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "tool"))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from batch_evaluate import _candidate_spacing_dirs
from batch_evaluate_utils import (
    build_spacing_map_from_nii_dir,
    compute_dice,
    compute_hd95,
    compute_iou,
    compute_nsd,
    decode_rle_mask,
    load_spacing_map,
)
import evaluation_json_to_excel_table as _ej2tsv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _default_predictions_json() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_test/"
        "dumps/acdc/coco_predictions_segm.json"
    )


def _default_test_dir() -> str:
    return "/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/test"


def _build_prediction_index(predictions):
    pred_index = {}
    for pred in predictions:
        key = (pred["image_id"], pred["category_id"])
        prev = pred_index.get(key)
        if prev is None or float(pred.get("score", 0.0)) > float(prev.get("score", 0.0)):
            pred_index[key] = pred
    return pred_index


def _resolve_spacing(volume_name, dataset_name, spacing_map):
    dataset_name = dataset_name.strip().lower()
    if dataset_name == "camus":
        return (1.0, 0.30799999833106995, 0.30799999833106995)

    if volume_name not in spacing_map:
        return None
    spacing_axes = spacing_map[volume_name]
    return (spacing_axes[2], spacing_axes[0], spacing_axes[1])


def _build_video_volume_groups(coco_data):
    images_dict = {img["id"]: img for img in coco_data["images"]}

    categories_dict = {}
    for cat in coco_data["categories"]:
        cid = cat["id"]
        if "names" in cat and isinstance(cat["names"], list):
            categories_dict[cid] = cat["names"][0]
        elif "name" in cat:
            categories_dict[cid] = cat["name"]
        else:
            raise ValueError(f"Category {cid} must have 'name' or 'names'")

    annotations_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    images_by_volume = defaultdict(list)
    volume_name_by_id = {}
    for img_id, img_info in images_dict.items():
        volume_id = img_info.get("video_id", img_id)
        slice_idx = img_info.get("original_slice_idx", img_info.get("frame_idx", 0))
        file_name = os.path.basename(img_info["file_name"])
        volume_name = os.path.splitext(file_name)[0]
        volume_name_by_id[volume_id] = volume_name
        images_by_volume[volume_id].append((img_id, int(slice_idx)))

    for volume_id in images_by_volume:
        images_by_volume[volume_id].sort(key=lambda item: item[1])

    return {
        "images_dict": images_dict,
        "categories_dict": categories_dict,
        "annotations_by_image": annotations_by_image,
        "images_by_volume": images_by_volume,
        "volume_name_by_id": volume_name_by_id,
    }


def _clean_dict(value):
    if isinstance(value, dict):
        return {k: _clean_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_dict(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(
        description="从 tracker 的 coco_predictions_segm.json 计算 3D Dice/IoU/HD95/NSD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions_json",
        type=str,
        default=_default_predictions_json(),
        help="tracker dump 输出的 coco_predictions_segm.json",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=_default_test_dir(),
        help="测试集目录（video_annotations/frame_annotations/spacing_map 所在目录）",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ACDC",
        help="ACDC | BTCV | Promise12 | MMs2 | CAMUS | MSCMR | ISBI",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="frame_annotations.coco.json",
        help="相对 test_dir 的 GT COCO 标注文件名",
    )
    parser.add_argument(
        "--spacing_file",
        type=str,
        default=None,
        help="spacing JSON；不指定时优先用 test_dir/spacing_map.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="evaluation_results_*.json 输出目录；默认与 predictions_json 同目录",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.predictions_json):
        logger.error("找不到 tracker predictions json: %s", args.predictions_json)
        return
    if not os.path.isdir(args.test_dir):
        logger.error("test_dir 无效: %s", args.test_dir)
        return

    dataset_name = args.dataset_name.strip().lower()
    allowed = {"camus", "acdc", "mms2", "btcv", "promise12", "mscmr", "isbi"}
    if dataset_name not in allowed:
        logger.error("dataset_name 须为 %s 之一", allowed)
        return

    if args.spacing_file is None and dataset_name != "isbi":
        auto_spacing = os.path.join(args.test_dir, "spacing_map.json")
        if os.path.isfile(auto_spacing):
            args.spacing_file = auto_spacing
            logger.info("使用 test_dir 内 spacing: %s", auto_spacing)

    spacing_map = {}
    if dataset_name != "camus":
        if args.spacing_file:
            spacing_map = load_spacing_map(args.spacing_file)
            logger.info("spacing 条目数（来自文件）: %d", len(spacing_map))
        elif dataset_name == "isbi":
            logger.info(
                "ISBI 默认不自动查找 spacing；未提供 --spacing_file，HD95/NSD 将为 nan。"
            )
        else:
            for spacing_dir in _candidate_spacing_dirs(args.test_dir, args.dataset_name):
                part = build_spacing_map_from_nii_dir(spacing_dir, args.dataset_name)
                if part:
                    spacing_map.update(part)
                    logger.info("spacing 条目数（来自 %s）: %d", spacing_dir, len(part))
            if spacing_map:
                logger.info("spacing 总条目数: %d", len(spacing_map))
            else:
                logger.warning(
                    "未提供 spacing 且候选目录无 nii：HD95/NSD 将为 nan（Dice/IoU 仍计算）"
                )

    json_path = os.path.join(args.test_dir, args.annotation_file)
    if not os.path.isfile(json_path):
        logger.error("找不到 GT 标注: %s", json_path)
        return

    logger.info("加载 tracker 预测: %s", args.predictions_json)
    with open(args.predictions_json, "r") as f:
        predictions = json.load(f)
    pred_index = _build_prediction_index(predictions)

    logger.info("加载 GT: %s", json_path)
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    data = _build_video_volume_groups(coco_data)
    images_dict = data["images_dict"]
    categories_dict = data["categories_dict"]
    annotations_by_image = data["annotations_by_image"]
    images_by_volume = data["images_by_volume"]
    volume_name_by_id = data["volume_name_by_id"]

    category_ious = defaultdict(list)
    category_dices = defaultdict(list)
    category_hd95s = defaultdict(list)
    category_nsds = defaultdict(list)
    per_patient = {}
    n_skipped_no_spacing = 0

    for volume_id, slice_list in tqdm(images_by_volume.items(), desc="Evaluating volumes"):
        volume_name = volume_name_by_id[volume_id]
        patient_name = volume_name
        slice_idx_to_img = {slice_idx: img_id for img_id, slice_idx in slice_list}

        spacing_3d = _resolve_spacing(
            volume_name, dataset_name, spacing_map
        )
        if spacing_3d is None and dataset_name not in {"camus", "isbi"}:
            n_skipped_no_spacing += 1

        volume_categories = set()
        for img_id, _ in slice_list:
            for ann in annotations_by_image[img_id]:
                volume_categories.add(ann["category_id"])
        if not volume_categories:
            continue

        volume_per_class = {}
        volume_overall_ious = []
        volume_overall_dices = []
        volume_overall_hd95s = []
        volume_overall_nsds = []

        for category_id in sorted(volume_categories):
            category_name = categories_dict[category_id]
            pred_masks_2d = []
            gt_masks_2d = []
            slice_indices = []

            for img_id, slice_idx in slice_list:
                img_info = images_dict[img_id]
                h, w = img_info["height"], img_info["width"]

                pred_entry = pred_index.get((img_id, category_id))
                if pred_entry is None:
                    pred_mask = np.zeros((h, w), dtype=bool)
                else:
                    pred_mask = decode_rle_mask(pred_entry["segmentation"], h, w)
                    if pred_mask.shape != (h, w):
                        pred_pil = Image.fromarray(pred_mask.astype(np.uint8) * 255)
                        pred_pil = pred_pil.resize((w, h), Image.NEAREST)
                        pred_mask = np.array(pred_pil) > 0

                anns = [
                    ann
                    for ann in annotations_by_image[img_id]
                    if ann["category_id"] == category_id
                ]
                if not anns:
                    gt_mask = np.zeros((h, w), dtype=bool)
                else:
                    gt_mask = decode_rle_mask(anns[0]["segmentation"], h, w)

                pred_masks_2d.append(pred_mask)
                gt_masks_2d.append(gt_mask)
                slice_indices.append(slice_idx)

            if not pred_masks_2d:
                continue

            num_slices = len(pred_masks_2d)
            h, w = pred_masks_2d[0].shape
            pred_mask_3d = np.zeros((num_slices, h, w), dtype=bool)
            gt_mask_3d = np.zeros((num_slices, h, w), dtype=bool)
            for i, (pred_mask, gt_mask) in enumerate(zip(pred_masks_2d, gt_masks_2d)):
                pred_mask_3d[i] = pred_mask
                gt_mask_3d[i] = gt_mask

            iou = compute_iou(pred_mask_3d, gt_mask_3d)
            dice = compute_dice(pred_mask_3d, gt_mask_3d)
            if spacing_3d is not None:
                try:
                    hd95 = compute_hd95(pred_mask_3d, gt_mask_3d, spacing=spacing_3d)
                except Exception:
                    hd95 = float("nan")
                try:
                    nsd = compute_nsd(
                        pred_mask_3d,
                        gt_mask_3d,
                        spacing=spacing_3d,
                        threshold_mm=2.0,
                    )
                except Exception:
                    nsd = float("nan")
            else:
                hd95 = float("nan")
                nsd = float("nan")

            category_ious[category_name].append(iou)
            category_dices[category_name].append(dice)
            category_hd95s[category_name].append(hd95)
            category_nsds[category_name].append(nsd)

            volume_per_class[category_name] = {
                "slice_indices": slice_indices,
                "dice": float(dice),
                "iou": float(iou),
                "hd95": None if (np.isnan(hd95) or np.isinf(hd95)) else float(hd95),
                "nsd": None if np.isnan(nsd) else float(nsd),
                "pred_sum": int(pred_mask_3d.sum()),
                "gt_sum": int(gt_mask_3d.sum()),
            }
            volume_overall_ious.append(iou)
            volume_overall_dices.append(dice)
            if not (np.isnan(hd95) or np.isinf(hd95)):
                volume_overall_hd95s.append(hd95)
            if not np.isnan(nsd):
                volume_overall_nsds.append(nsd)

        vol_dice = (
            float(np.mean(volume_overall_dices)) if volume_overall_dices else float("nan")
        )
        vol_iou = (
            float(np.mean(volume_overall_ious)) if volume_overall_ious else float("nan")
        )
        vol_hd95 = (
            float(np.nanmean(volume_overall_hd95s))
            if volume_overall_hd95s
            else float("nan")
        )
        vol_nsd = (
            float(np.nanmean(volume_overall_nsds))
            if volume_overall_nsds
            else float("nan")
        )

        per_patient[patient_name] = {
            "per_class": {str(name): volume_per_class[name] for name in volume_per_class},
            "overall": {
                "dice": vol_dice,
                "iou": vol_iou,
                "hd95": None if (np.isnan(vol_hd95) or np.isinf(vol_hd95)) else vol_hd95,
                "nsd": None if np.isnan(vol_nsd) else vol_nsd,
            },
        }

        hd95_str = (
            f", HD95={vol_hd95:.4f}"
            if not (np.isnan(vol_hd95) or np.isinf(vol_hd95))
            else ", HD95=N/A"
        )
        nsd_str = f", NSD={vol_nsd:.4f}" if not np.isnan(vol_nsd) else ", NSD=N/A"
        logger.info(
            "%s: Dice=%.4f, IoU=%.4f%s%s",
            patient_name,
            vol_dice,
            vol_iou,
            hd95_str,
            nsd_str,
        )

    if n_skipped_no_spacing > 0:
        logger.warning(
            "%d 个 volume 无 spacing，HD95/NSD 为 nan。可提供 --spacing_file 或在 test 下放 nii。",
            n_skipped_no_spacing,
        )

    case_dices = []
    case_ious = []
    case_hd95s = []
    case_nsds = []
    for patient_name in sorted(per_patient.keys()):
        overall = per_patient[patient_name]["overall"]
        case_dices.append(overall["dice"])
        case_ious.append(overall["iou"])
        case_hd95s.append(overall["hd95"] if overall["hd95"] is not None else float("nan"))
        case_nsds.append(overall["nsd"] if overall["nsd"] is not None else float("nan"))

    n_cases = len(case_dices)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print("\nPer-category metrics:")
    print("-" * 100)
    print(
        f"{'Category':<20} {'IoU':<12} {'Dice':<12} {'HD95(mm)':<12} {'NSD(2mm)':<12} {'Count':<10}"
    )
    print("-" * 100)

    per_class_summary = {}
    for category_name in sorted(category_ious.keys()):
        ious = category_ious[category_name]
        dices = category_dices[category_name]
        hd95s = category_hd95s[category_name]
        nsds = category_nsds[category_name]
        mean_iou = float(np.mean(ious)) if ious else float("nan")
        mean_dice = float(np.mean(dices)) if dices else float("nan")
        mean_hd95 = float(np.nanmean(hd95s)) if hd95s else float("nan")
        mean_nsd = float(np.nanmean(nsds)) if nsds else float("nan")
        print(
            f"{category_name:<20} {mean_iou:<12.4f} {mean_dice:<12.4f} {mean_hd95:<12.2f} {mean_nsd:<12.4f} {len(ious):<10}"
        )

        valid_hd95s = [h for h in hd95s if np.isfinite(h)]
        valid_nsds = [n for n in nsds if not np.isnan(n)]
        per_class_summary[category_name] = {
            "dice_mean": mean_dice,
            "dice_std": float(np.std(dices))
            if len(dices) > 1
            else (0.0 if dices else float("nan")),
            "iou_mean": mean_iou,
            "iou_std": float(np.std(ious))
            if len(ious) > 1
            else (0.0 if ious else float("nan")),
            "hd95_mean": float(np.mean(valid_hd95s)) if valid_hd95s else float("nan"),
            "hd95_std": float(np.std(valid_hd95s))
            if len(valid_hd95s) > 1
            else (0.0 if valid_hd95s else float("nan")),
            "nsd_mean": float(np.mean(valid_nsds)) if valid_nsds else float("nan"),
            "nsd_std": float(np.std(valid_nsds))
            if len(valid_nsds) > 1
            else (0.0 if valid_nsds else float("nan")),
            "count": len(ious),
        }

    print("-" * 100)
    if n_cases > 0:
        overall_iou = float(np.nanmean(case_ious))
        overall_dice = float(np.nanmean(case_dices))
        overall_iou_std = float(np.nanstd(case_ious)) if n_cases > 1 else 0.0
        overall_dice_std = float(np.nanstd(case_dices)) if n_cases > 1 else 0.0
        valid_hd95 = [x for x in case_hd95s if np.isfinite(x)]
        valid_nsd = [x for x in case_nsds if not np.isnan(x)]
        overall_hd95 = float(np.mean(valid_hd95)) if valid_hd95 else float("nan")
        overall_nsd = float(np.mean(valid_nsd)) if valid_nsd else float("nan")
        overall_hd95_std = (
            float(np.std(valid_hd95))
            if len(valid_hd95) > 1
            else (0.0 if valid_hd95 else float("nan"))
        )
        overall_nsd_std = (
            float(np.std(valid_nsd))
            if len(valid_nsd) > 1
            else (0.0 if valid_nsd else float("nan"))
        )
    else:
        overall_iou = overall_dice = overall_hd95 = overall_nsd = float("nan")
        overall_iou_std = overall_dice_std = overall_hd95_std = overall_nsd_std = float(
            "nan"
        )

    print(
        f"{'Overall (macro)':<20} {overall_iou:<12.4f} {overall_dice:<12.4f} {overall_hd95:<12.2f} {overall_nsd:<12.4f} {n_cases:<10}"
    )
    print(
        f"{'Overall (Mean±SD)':<20} IoU={overall_iou:.4f}±{overall_iou_std:.4f}  Dice={overall_dice:.4f}±{overall_dice_std:.4f}  HD95={overall_hd95:.2f}±{overall_hd95_std:.2f}  NSD={overall_nsd:.4f}±{overall_nsd_std:.4f}"
    )
    print("=" * 100)

    overall_summary = {
        "dice_mean": float(np.nanmean(case_dices)) if case_dices else float("nan"),
        "dice_std": float(np.nanstd(case_dices)) if len(case_dices) > 1 else (0.0 if case_dices else float("nan")),
        "iou_mean": float(np.nanmean(case_ious)) if case_ious else float("nan"),
        "iou_std": float(np.nanstd(case_ious)) if len(case_ious) > 1 else (0.0 if case_ious else float("nan")),
        "hd95_mean": overall_hd95,
        "hd95_std": overall_hd95_std,
        "nsd_mean": overall_nsd,
        "nsd_std": overall_nsd_std,
    }

    output_dir = args.output_dir or os.path.dirname(args.predictions_json)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"evaluation_results_{dataset_name}.json")
    output_dict = {
        "per_class": _clean_dict(per_class_summary),
        "overall": _clean_dict(overall_summary),
        "per_patient": _clean_dict(per_patient),
        "n_cases": len(per_patient),
    }
    with open(output_file, "w") as f:
        json.dump(output_dict, f, indent=2)

    logger.info("结果已写入: %s", output_file)
    try:
        rows = _ej2tsv.extract_rows(output_dict)
        tsv_text = _ej2tsv.to_tsv(rows)
        tsv_path = _ej2tsv.default_output_path(Path(output_file).resolve())
        tsv_path.write_text(tsv_text, encoding="utf-8")
        logger.info("Excel 可复制 TSV 已写入: %s", tsv_path)
    except Exception as exc:
        logger.warning(
            "未能从评估结果生成 TSV（可手动运行: python gq_scripts/tool/evaluation_json_to_excel_table.py %s）: %s",
            output_file,
            exc,
        )


if __name__ == "__main__":
    main()

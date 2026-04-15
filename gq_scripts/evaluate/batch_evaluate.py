"""
从 ``batch_inference.py`` 生成的 ``predictions.pkl`` 计算 3D 指标（不加载模型）。

评估范围与推理脚本一致：仅包含有标注的 volume / 类别；无 GT 的切片不参与。

指标：IoU、Dice、HD95、NSD（需体素 spacing；可从 ``--spacing_file``、
``test_dir/spacing_map.json`` 或 ``test_dir`` 下 ``.nii.gz`` 自动推断）。

若 Dice/IoU 全为 0：多为推理 ``confidence_threshold`` 过高，请用 ``0.0`` 重跑推理。

数据集：``--dataset_name`` 为 ``ACDC`` / ``BTCV`` / ``Promise12`` / ``MMs2`` / ``CAMUS`` / ``MSCMR``（大小写不敏感）。

示例（ACDC + 与 ``scribble_lora`` 一致的测试目录）::

    python gq_scripts/evaluate/batch_evaluate.py \
        --predictions_file /home/gaoqi/sam3/gq_experiment/acdc/scribble_tmi_lora/inference_test/predictions.pkl \
        --test_dir /home/gaoqi/dataset/using/acdc/processed/png_coco_sam3_fullframes_weak/test \
        --dataset_name ACDC

一键推理+评估请用同目录下的 ``run_inference_and_eval.sh``。

写入 ``evaluation_results_*.json`` 后，会同时在同目录生成
``evaluation_results_*_excel_table.tsv``（制表符分隔，便于粘贴到 Excel / WPS），
逻辑与 ``gq_scripts/tool/evaluation_json_to_excel_table.py`` 一致。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# 保证以 ``python gq_scripts/evaluate/batch_evaluate.py`` 方式运行时能导入同目录工具模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_TOOL_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "tool"))
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from batch_evaluate_utils import (
    compute_iou,
    compute_dice,
    compute_hd95,
    compute_nsd,
    decode_rle_mask,
    load_spacing_map,
    build_spacing_map_from_nii_dir,
)

import evaluation_json_to_excel_table as _ej2tsv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_patient_frame(img_file: str):
    """解析文件名 → patient_id, frame_id, slice_idx, view（与 batch_inference 一致）。"""
    basename = os.path.basename(img_file)

    match_acdc = re.match(r"patient(\d+)_frame(\d+)_slice(\d+)\.png", basename)
    if match_acdc:
        return (
            match_acdc.group(1),
            match_acdc.group(2),
            int(match_acdc.group(3)),
            None,
        )

    match_mms2 = re.match(r"(\d+)_(SA|LA)_(ED|ES)_slice(\d+)\.png", basename)
    if match_mms2:
        patient_id = match_mms2.group(1)
        view = match_mms2.group(2)
        phase = match_mms2.group(3)
        slice_idx = int(match_mms2.group(4))
        return patient_id, f"{view}_{phase}", slice_idx, view

    match_camus = re.match(
        r"patient(\d+)_([A-Z0-9]+)_(ED|ES)(?:_slice(\d+))?\.png", basename
    )
    if match_camus:
        patient_id = match_camus.group(1)
        view = match_camus.group(2)
        phase = match_camus.group(3)
        slice_idx_str = match_camus.group(4)
        slice_idx = int(slice_idx_str) if slice_idx_str else 0
        return patient_id, f"{view}_{phase}", slice_idx, view

    match_btcv = re.match(r"([\d]+)-Image_slice(\d+)\.png", basename)
    if match_btcv:
        return match_btcv.group(1), "default", int(match_btcv.group(2)), None

    match_promise12 = re.match(r"Case(\d+)_slice(\d+)\.png", basename)
    if match_promise12:
        return match_promise12.group(1), "default", int(match_promise12.group(2)), None

    match_isbi = re.match(r"patient(\d+)_slice(\d+)\.png", basename)
    if match_isbi:
        return match_isbi.group(1), "default", int(match_isbi.group(2)), None

    match_mscmr = re.match(r"subject(\d+)_([A-Za-z0-9]+)_slice(\d+)\.png", basename)
    if match_mscmr:
        patient_id = match_mscmr.group(1)
        phase = match_mscmr.group(2)
        slice_idx = int(match_mscmr.group(3))
        return patient_id, phase, slice_idx, None

    return None, None, None, None


def build_volume_groups_and_annotations(coco_data, test_dir):
    """COCO → 按 (patient_id, frame_id) 分组的切片与标注索引。"""
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
    for img_id, img_info in images_dict.items():
        patient_id, frame_id, slice_idx, view = extract_patient_frame(
            img_info["file_name"]
        )
        if patient_id is not None and frame_id is not None:
            images_by_volume[(patient_id, frame_id)].append((img_id, slice_idx, view))

    has_view = False
    is_mms2 = False
    for img_list in images_by_volume.values():
        if img_list and len(img_list[0]) >= 3 and img_list[0][2] is not None:
            has_view = True
            if img_list[0][2] in ("SA", "LA"):
                is_mms2 = True
            break

    if has_view and is_mms2:
        filtered = defaultdict(list)
        for key, img_list in images_by_volume.items():
            if img_list and len(img_list[0]) >= 3 and img_list[0][2] == "SA":
                filtered[key] = img_list
        images_by_volume = filtered

    for key in images_by_volume:
        images_by_volume[key].sort(key=lambda x: x[1])

    return {
        "images_dict": images_dict,
        "categories_dict": categories_dict,
        "annotations_by_image": annotations_by_image,
        "images_by_volume": images_by_volume,
    }


def _default_predictions() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/scribble_tmi_lora/inference_test/"
        "predictions.pkl"
    )


def _default_test_dir() -> str:
    return (
        "/home/gaoqi/dataset/using/acdc/processed/png_coco_sam3_fullframes_weak/test"
    )


def _candidate_spacing_dirs(test_dir: str, dataset_name: str):
    """返回用于自动推断 spacing 的候选目录（优先 test_dir）。"""
    dirs = [test_dir]
    dataset_key = (dataset_name or "").strip().lower()
    if dataset_key == "mscmr":
        norm_test = os.path.normpath(test_dir)
        if "/processed/" in norm_test:
            root = norm_test.split("/processed/")[0]
            raw_test = os.path.join(root, "raw", "test")
            dirs.extend([raw_test, os.path.join(raw_test, "images")])
    else:
        return dirs

    dedup = []
    seen = set()
    for d in dirs:
        nd = os.path.normpath(d)
        if nd not in seen:
            dedup.append(nd)
            seen.add(nd)
    return dedup


def main():
    parser = argparse.ArgumentParser(
        description="从 batch_inference 的 predictions.pkl 计算 3D Dice/IoU/HD95/NSD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=_default_predictions(),
        help="batch_inference 输出的 predictions.pkl",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="测试集目录；默认用 pkl 内 config.test_dir 或下方默认 ACDC test",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ACDC",
        help="ACDC | BTCV | Promise12 | MMs2 | CAMUS | MSCMR",
    )
    parser.add_argument(
        "--spacing_file",
        type=str,
        default=None,
        help="spacing JSON；不指定时若存在 test_dir/spacing_map.json 则自动使用，否则尝试从 .nii.gz 推断",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="image_annotations.coco.json",
        help="相对 test_dir 的 COCO 标注文件名",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="evaluation_results_*.json 输出目录；默认与 predictions_file 同目录",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.predictions_file):
        logger.error("找不到 predictions 文件: %s", args.predictions_file)
        return

    with open(args.predictions_file, "rb") as f:
        pred_data = pickle.load(f)

    volumes_pred = pred_data["volumes"]
    config = pred_data.get("config", {})
    test_dir = args.test_dir or config.get("test_dir") or _default_test_dir()
    if not test_dir or not os.path.isdir(test_dir):
        logger.error("test_dir 无效，请设置 --test_dir")
        return

    if args.spacing_file is None and (args.dataset_name or "").strip().lower() != "isbi":
        auto_spacing = os.path.join(test_dir, "spacing_map.json")
        if os.path.isfile(auto_spacing):
            args.spacing_file = auto_spacing
            logger.info("使用 test_dir 内 spacing: %s", auto_spacing)

    json_path = os.path.join(test_dir, args.annotation_file)
    if not os.path.isfile(json_path):
        logger.error("找不到标注: %s", json_path)
        return

    dataset_name = args.dataset_name.strip().lower()
    allowed = {"camus", "acdc", "mms2", "btcv", "promise12", "mscmr", "isbi"}
    if dataset_name not in allowed:
        logger.error("dataset_name 须为 %s 之一", allowed)
        return

    is_camus = dataset_name == "camus"
    is_mms2 = dataset_name == "mms2"
    is_acdc = dataset_name == "acdc"
    is_btcv = dataset_name == "btcv"
    is_promise12 = dataset_name == "promise12"
    is_mscmr = dataset_name == "mscmr"
    is_isbi = dataset_name == "isbi"

    spacing_map = {}
    if not is_camus:
        if args.spacing_file:
            spacing_map = load_spacing_map(args.spacing_file)
            logger.info("spacing 条目数（来自文件）: %d", len(spacing_map))
        elif is_isbi:
            logger.info(
                "ISBI 默认不自动查找 spacing；未提供 --spacing_file，HD95/NSD 将为 nan。"
            )
        else:
            for spacing_dir in _candidate_spacing_dirs(test_dir, args.dataset_name):
                part = build_spacing_map_from_nii_dir(spacing_dir, args.dataset_name)
                if part:
                    spacing_map.update(part)
                    logger.info(
                        "spacing 条目数（来自 %s）: %d", spacing_dir, len(part)
                    )
            if spacing_map:
                logger.info("spacing 总条目数: %d", len(spacing_map))
            else:
                logger.warning(
                    "未提供 spacing 且候选目录无 nii：HD95/NSD 将为 nan（Dice/IoU 仍计算）"
                )

    logger.info("加载 GT: %s", json_path)
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    data = build_volume_groups_and_annotations(coco_data, test_dir)
    images_dict = data["images_dict"]
    categories_dict = data["categories_dict"]
    annotations_by_image = data["annotations_by_image"]
    images_by_volume = data["images_by_volume"]

    category_ious = defaultdict(list)
    category_dices = defaultdict(list)
    category_hd95s = defaultdict(list)
    category_nsds = defaultdict(list)
    all_ious, all_dices, all_hd95s, all_nsds = [], [], [], []
    per_patient = {}
    n_skipped_no_volume = 0
    n_skipped_no_spacing = 0

    for vol in tqdm(volumes_pred, desc="Evaluating volumes"):
        patient_id = vol["patient_id"]
        frame_id = vol["frame_id"]
        patient_name = vol["patient_name"]
        volume_key = (patient_id, frame_id)

        if volume_key not in images_by_volume:
            n_skipped_no_volume += 1
            continue

        slice_list = images_by_volume[volume_key]
        slice_idx_to_img = {s[1]: s[0] for s in slice_list}
        first_view = slice_list[0][2] if slice_list else None

        if is_camus:
            spacing_3d = (1.0, 0.30799999833106995, 0.30799999833106995)
        else:
            if is_acdc:
                key = f"patient{patient_id}_frame{frame_id}"
            elif is_mms2:
                key = f"{patient_id}_{frame_id}"
            elif is_btcv:
                key = patient_id
            elif is_promise12:
                key = f"Case{patient_id}"
            elif is_mscmr:
                key = f"subject{patient_id}_{frame_id}"
            elif is_isbi:
                key = f"patient{patient_id}"
            else:
                key = (
                    f"patient{patient_id}_{frame_id}"
                    if first_view is None
                    else f"patient{patient_id}_{frame_id}"
                )
            if key in spacing_map:
                spacing_axes = spacing_map[key]
                spacing_3d = (spacing_axes[2], spacing_axes[0], spacing_axes[1])
            else:
                n_skipped_no_spacing += 1
                spacing_3d = None

        volume_per_class = {}
        volume_overall_ious = []
        volume_overall_dices = []
        volume_overall_hd95s = []
        volume_overall_nsds = []

        for cat_data in vol["categories"]:
            category_id = cat_data["category_id"]
            category_name = cat_data["category_name"]
            slice_indices = cat_data["slice_indices"]
            pred_masks_2d = cat_data["masks"]

            if len(pred_masks_2d) == 0:
                continue

            gt_masks_2d = []
            for si in slice_indices:
                img_id = slice_idx_to_img.get(si)
                if img_id is None:
                    gt_masks_2d.append(np.zeros_like(pred_masks_2d[0]))
                    continue
                img_info = images_dict[img_id]
                h, w = img_info["height"], img_info["width"]
                anns = [
                    a
                    for a in annotations_by_image[img_id]
                    if a["category_id"] == category_id
                ]
                if not anns:
                    gt_masks_2d.append(np.zeros((h, w), dtype=bool))
                else:
                    gt_masks_2d.append(
                        decode_rle_mask(anns[0]["segmentation"], h, w)
                    )

            num_slices = len(pred_masks_2d)
            h, w = pred_masks_2d[0].shape
            pred_mask_3d = np.zeros((num_slices, h, w), dtype=bool)
            gt_mask_3d = np.zeros((num_slices, h, w), dtype=bool)
            for i in range(num_slices):
                pred_mask_3d[i] = pred_masks_2d[i]
                if i < len(gt_masks_2d):
                    g = gt_masks_2d[i]
                    if g.shape != (h, w):
                        g_pil = Image.fromarray(g.astype(np.uint8) * 255)
                        g_pil = g_pil.resize((w, h), Image.NEAREST)
                        g = np.array(g_pil) > 0
                    gt_mask_3d[i] = g

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
            all_ious.append(iou)
            all_dices.append(dice)
            all_hd95s.append(hd95)
            all_nsds.append(nsd)

            volume_per_class[category_name] = {
                "dice": float(dice),
                "iou": float(iou),
                "hd95": None
                if (np.isnan(hd95) or np.isinf(hd95))
                else float(hd95),
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
            float(np.mean(volume_overall_dices))
            if volume_overall_dices
            else float("nan")
        )
        vol_iou = (
            float(np.mean(volume_overall_ious))
            if volume_overall_ious
            else float("nan")
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
            "per_class": {str(c): volume_per_class[c] for c in volume_per_class},
            "overall": {
                "dice": vol_dice,
                "iou": vol_iou,
                "hd95": None
                if (np.isnan(vol_hd95) or np.isinf(vol_hd95))
                else vol_hd95,
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
    if n_skipped_no_volume > 0:
        logger.warning(
            "%d 个 volume 在 COCO 中无对应切片，已跳过。", n_skipped_no_volume
        )

    case_dices = []
    case_ious = []
    case_hd95s = []
    case_nsds = []
    for p in sorted(per_patient.keys()):
        o = per_patient[p]["overall"]
        case_dices.append(o["dice"])
        case_ious.append(o["iou"])
        case_hd95s.append(o["hd95"] if o["hd95"] is not None else float("nan"))
        case_nsds.append(o["nsd"] if o["nsd"] is not None else float("nan"))
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

    for category_name in sorted(category_ious.keys()):
        ious = category_ious[category_name]
        dices = category_dices[category_name]
        hd95s = category_hd95s[category_name]
        nsds = category_nsds[category_name]
        mean_iou = np.mean(ious) if ious else float("nan")
        mean_dice = np.mean(dices) if dices else float("nan")
        mean_hd95 = float(np.nanmean(hd95s)) if hd95s else float("nan")
        mean_nsd = float(np.nanmean(nsds)) if nsds else float("nan")
        print(
            f"{category_name:<20} {mean_iou:<12.4f} {mean_dice:<12.4f} {mean_hd95:<12.2f} {mean_nsd:<12.4f} {len(ious):<10}"
        )

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

    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        if isinstance(d, list):
            return [clean_dict(x) for x in d]
        if isinstance(d, float) and (np.isnan(d) or np.isinf(d)):
            return None
        return d

    per_class_summary = {}
    for category_name in sorted(category_ious.keys()):
        ious = category_ious[category_name]
        dices = category_dices[category_name]
        hd95s = category_hd95s[category_name]
        nsds = category_nsds[category_name]
        valid_hd95s = [h for h in hd95s if np.isfinite(h)]
        valid_nsds = [n for n in nsds if not np.isnan(n)]
        per_class_summary[category_name] = {
            "dice_mean": float(np.mean(dices)) if dices else float("nan"),
            "dice_std": float(np.std(dices))
            if len(dices) > 1
            else (0.0 if dices else float("nan")),
            "iou_mean": float(np.mean(ious)) if ious else float("nan"),
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

    if n_cases > 0:
        overall_dice_mean = float(np.nanmean(case_dices))
        overall_dice_std = (
            float(np.nanstd(case_dices))
            if n_cases > 1
            else (0.0 if n_cases else float("nan"))
        )
        overall_iou_mean = float(np.nanmean(case_ious))
        overall_iou_std = (
            float(np.nanstd(case_ious))
            if n_cases > 1
            else (0.0 if n_cases else float("nan"))
        )
        valid_hd95_for_std = [x for x in case_hd95s if np.isfinite(x)]
        valid_nsd_for_std = [x for x in case_nsds if not np.isnan(x)]
        overall_hd95_mean = (
            float(np.mean(valid_hd95_for_std)) if valid_hd95_for_std else float("nan")
        )
        overall_hd95_std = (
            float(np.std(valid_hd95_for_std))
            if len(valid_hd95_for_std) > 1
            else (0.0 if valid_hd95_for_std else float("nan"))
        )
        overall_nsd_mean = (
            float(np.mean(valid_nsd_for_std)) if valid_nsd_for_std else float("nan")
        )
        overall_nsd_std = (
            float(np.std(valid_nsd_for_std))
            if len(valid_nsd_for_std) > 1
            else (0.0 if valid_nsd_for_std else float("nan"))
        )
    else:
        overall_dice_mean = overall_dice_std = overall_iou_mean = overall_iou_std = float(
            "nan"
        )
        overall_hd95_mean = overall_hd95_std = overall_nsd_mean = overall_nsd_std = float(
            "nan"
        )
    overall_summary = {
        "dice_mean": overall_dice_mean,
        "dice_std": overall_dice_std,
        "iou_mean": overall_iou_mean,
        "iou_std": overall_iou_std,
        "hd95_mean": overall_hd95_mean,
        "hd95_std": overall_hd95_std,
        "nsd_mean": overall_nsd_mean,
        "nsd_std": overall_nsd_std,
    }

    output_dir = args.output_dir or os.path.dirname(args.predictions_file)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"evaluation_results_{dataset_name}.json")
    output_dict = {
        "per_class": clean_dict(per_class_summary),
        "overall": clean_dict(overall_summary),
        "per_patient": clean_dict(per_patient),
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
    logger.info(
        "Per-class: %d 类, Per-patient: %d 例",
        len(per_class_summary),
        len(per_patient),
    )


if __name__ == "__main__":
    main()

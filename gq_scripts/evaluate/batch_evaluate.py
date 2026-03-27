"""
SAM3 模型批量评估脚本 - 3D 医学图像分割评估

基于 batch_inference.py 生成的预测结果文件，对 3D volume 计算评估指标，
不加载模型、不运行推理。

评估范围（与 batch_inference 一致）：
- 仅对「有至少一条标注」的 volume（patient+frame）做评估；预测文件里没有的 volume 不会参与。
- 对每个 volume，仅评估「在该 volume 内至少有一条标注」的类别；无标注的类别在该 volume 下不计算指标。
- 因此：测试集中完全没有标注的图像/volume 不会出现在预测文件里，也不会被评估；每个 volume 只评估其 GT 出现过的类别。

若评估结果 Dice/IoU 全为 0、HD95 很大：通常是推理时 confidence_threshold 过高，
导致预测被过滤成全零。请用 --confidence_threshold 0.0 重新跑 batch_inference 再评估。

Spacing：可不提供 --spacing_file；脚本会从 test_dir（及子目录 img/images/nii）内的 .nii.gz 自动读取 spacing 以计算 HD95/NSD。若既无 spacing 文件也无 nii，仍会计算 Dice/IoU，HD95/NSD 为 nan。

功能：
1. 从 .pkl 预测结果和 COCO JSON 加载预测与 GT，按 volume 组合成 3D
2. 评估指标：IoU、Dice、HD95、NSD（支持 3D spacing）
3. 输出：控制台汇总、per-patient 与 per-class 结果，保存为 JSON
4. 支持数据集：ACDC、CAMUS、MMs2、BTCV、Promise12

常用命令示例（复制到终端执行）：
  test_dir 内若无 .nii.gz，必须用 --spacing_file 才能算 HD95/NSD；每行末尾的 \\ 不能少，否则下一行参数不会生效。
------------------------------------------------------------
# ACDC（需 --spacing_file，如从 acdc2 等含原始 nii 的目录生成的 spacing_map.json）
python gq_scripts/evaluate/batch_evaluate.py \
    --predictions_file /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_acdc_0.7/predictions.pkl \
    --test_dir /home/gaoqi/dataset/using/acdc4/test \
    --dataset_name ACDC \
    --spacing_file /home/gaoqi/dataset/using/acdc2/test/spacing_map.json

# BTCV（需 --spacing_file，如从 btcv_1 等含原始 nii 的目录生成的 spacing_map.json）
python gq_scripts/evaluate/batch_evaluate.py \
    --predictions_file /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_btcv_0.7/predictions.pkl \
    --test_dir /home/gaoqi/dataset/using/btcv_2/test \
    --dataset_name BTCV \
    --spacing_file /home/gaoqi/dataset/using/btcv_1/test/spacing_map.json

# Promise12（需 --spacing_file，如从 promise12_2 等含原始 nii 的目录生成的 spacing_map.json）
python gq_scripts/evaluate/batch_evaluate.py \
    --predictions_file /home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_promise12_0.7/predictions.pkl \
    --test_dir /home/gaoqi/dataset/using/promise12_3/test \
    --dataset_name Promise12 \
    --spacing_file /home/gaoqi/dataset/using/promise12_2/test/spacing_map.json
------------------------------------------------------------
"""

import os
import re
import json
import argparse
import pickle
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

from batch_evaluate_util import (
    compute_iou,
    compute_dice,
    compute_hd95,
    compute_nsd,
    decode_rle_mask,
    load_spacing_map,
    build_spacing_map_from_nii_dir,
)


def extract_patient_frame(img_file):
    """从文件名提取 patient_id、frame_id、slice_idx、view。
    
    支持：
    - ACDC: patient{数字}_frame{数字}_slice{数字}.png
    - MMs2: {patient_id}_{view}_{phase}_slice{数字}.png (view: SA/LA, phase: ED/ES)
    - CAMUS: patient{数字}_{view}_{phase}_slice{数字}.png 或 无 _slice 后缀
    - BTCV: {patient_id}-Image_slice{数字}.png (例如: 0507688-Image_slice000.png)
    - Promise12: Case{patient_id}_slice{数字}.png (例如: Case00_slice000.png)
    """
    basename = os.path.basename(img_file)

    match_acdc = re.match(r"patient(\d+)_frame(\d+)_slice(\d+)\.png", basename)
    if match_acdc:
        patient_id = match_acdc.group(1)
        frame_id = match_acdc.group(2)
        slice_idx = int(match_acdc.group(3))
        return patient_id, frame_id, slice_idx, None

    match_mms2 = re.match(r"(\d+)_(SA|LA)_(ED|ES)_slice(\d+)\.png", basename)
    if match_mms2:
        patient_id = match_mms2.group(1)
        view = match_mms2.group(2)
        phase = match_mms2.group(3)
        slice_idx = int(match_mms2.group(4))
        frame_id = f"{view}_{phase}"
        return patient_id, frame_id, slice_idx, view

    match_camus = re.match(
        r"patient(\d+)_([A-Z0-9]+)_(ED|ES)(?:_slice(\d+))?\.png", basename
    )
    if match_camus:
        patient_id = match_camus.group(1)
        view = match_camus.group(2)
        phase = match_camus.group(3)
        slice_idx_str = match_camus.group(4)
        slice_idx = int(slice_idx_str) if slice_idx_str else 0
        frame_id = f"{view}_{phase}"
        return patient_id, frame_id, slice_idx, view

    # BTCV: {patient_id}-Image_slice{slice_idx}.png (例如: 0507688-Image_slice000.png)
    match_btcv = re.match(r"([\d]+)-Image_slice(\d+)\.png", basename)
    if match_btcv:
        patient_id = match_btcv.group(1)
        slice_idx = int(match_btcv.group(2))
        frame_id = "default"  # BTCV 没有 frame 概念，使用默认值
        return patient_id, frame_id, slice_idx, None

    # Promise12: Case{patient_id}_slice{slice_idx}.png (例如: Case00_slice000.png)
    match_promise12 = re.match(r"Case(\d+)_slice(\d+)\.png", basename)
    if match_promise12:
        patient_id = match_promise12.group(1)
        slice_idx = int(match_promise12.group(2))
        frame_id = "default"  # Promise12 没有 frame 概念，使用默认值
        return patient_id, frame_id, slice_idx, None

    return None, None, None, None


def build_volume_groups_and_annotations(coco_data, test_dir):
    """从 COCO JSON 构建按 (patient_id, frame_id) 分组的切片列表及标注索引。"""
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


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 批量评估：从推理结果文件计算 3D 指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default="/home/gaoqi/sam3/gq_experiment/final/lora_acdc_btcv_promise12/inference_btcv_0.7/predictions.pkl",
        help="batch_inference.py 生成的预测结果 .pkl 路径",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="测试集目录（若未指定则从 predictions 的 config 读取）",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="BTCV",
        help="数据集名称: CAMUS, ACDC, MMs2, BTCV, Promise12",
    )
    parser.add_argument(
        "--spacing_file",
        type=str,
        default=None,
        help="spacing 映射 JSON 路径；不指定时从 test_dir 内的 .nii.gz 自动读取 spacing",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="image_annotations.coco.json",
        help="COCO 标注 JSON 文件名（相对于 test_dir）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估结果 JSON 输出目录；默认与 predictions_file 同目录",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.predictions_file):
        print(f"Error: predictions file not found: {args.predictions_file}")
        return

    with open(args.predictions_file, "rb") as f:
        pred_data = pickle.load(f)

    volumes_pred = pred_data["volumes"]
    config = pred_data.get("config", {})
    test_dir = args.test_dir or config.get("test_dir")
    if not test_dir or not os.path.isdir(test_dir):
        print("Error: test_dir 未指定或目录不存在，请提供 --test_dir")
        return

    json_path = os.path.join(test_dir, args.annotation_file)
    if not os.path.isfile(json_path):
        print(f"Error: annotation file not found: {json_path}")
        return

    dataset_name = args.dataset_name.strip().lower()
    allowed = {"camus", "acdc", "mms2", "btcv", "promise12"}
    if dataset_name not in allowed:
        print(f"Error: dataset_name 必须是 {allowed} 之一")
        return

    is_camus = dataset_name == "camus"
    is_mms2 = dataset_name == "mms2"
    is_acdc = dataset_name == "acdc"
    is_btcv = dataset_name == "btcv"
    is_promise12 = dataset_name == "promise12"

    spacing_map = {}
    if not is_camus:
        if args.spacing_file:
            spacing_map = load_spacing_map(args.spacing_file)
            print(f"Loaded {len(spacing_map)} spacing entries from --spacing_file")
        else:
            spacing_map = build_spacing_map_from_nii_dir(test_dir, args.dataset_name)
            if spacing_map:
                print(f"Loaded {len(spacing_map)} spacing entries from .nii.gz under test_dir")
            else:
                print("Warning: 未提供 --spacing_file 且 test_dir 下未找到 .nii.gz，将跳过 HD95/NSD（Dice/IoU 仍会计算）")

    print("Loading COCO JSON for GT ...")
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
            # CAMUS 固定 spacing，已按 (z, y, x) 与 (num_slices, h, w) 对齐
            spacing_3d = (1.0, 0.30799999833106995, 0.30799999833106995)  # (z, y, x)
        else:
            if is_acdc:
                key = f"patient{patient_id}_frame{frame_id}"
            elif is_mms2:
                key = f"{patient_id}_{frame_id}"
            elif is_btcv:
                key = patient_id
            elif is_promise12:
                key = f"Case{patient_id}"
            else:
                key = f"patient{patient_id}_{frame_id}" if first_view is None else f"patient{patient_id}_{frame_id}"
            if key in spacing_map:
                # spacing_map 来自 nii get_zooms()，顺序为 (axis0, axis1, axis2)；nii 通常为 (H,W,Z) 或 (W,H,Z)，即 axis2=slice
                # pred_mask_3d 形状为 (num_slices, h, w) = (Z, H, W)，故 spacing = (spacing_Z, spacing_H, spacing_W) = (json[2], json[0], json[1])
                spacing_axes = spacing_map[key]
                spacing_3d = (spacing_axes[2], spacing_axes[0], spacing_axes[1])  # (z, y, x) 对应 (num_slices, h, w)
            else:
                n_skipped_no_spacing += 1
                spacing_3d = None  # 无 spacing 时仍计算 Dice/IoU，HD95/NSD 置为 nan

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
                anns = [a for a in annotations_by_image[img_id] if a["category_id"] == category_id]
                if not anns:
                    gt_masks_2d.append(np.zeros((h, w), dtype=bool))
                else:
                    gt_masks_2d.append(decode_rle_mask(anns[0]["segmentation"], h, w))

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
                    nsd = compute_nsd(pred_mask_3d, gt_mask_3d, spacing=spacing_3d, threshold_mm=2.0)
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

        vol_dice = float(np.mean(volume_overall_dices)) if volume_overall_dices else float("nan")
        vol_iou = float(np.mean(volume_overall_ious)) if volume_overall_ious else float("nan")
        vol_hd95 = float(np.nanmean(volume_overall_hd95s)) if volume_overall_hd95s else float("nan")
        vol_nsd = float(np.nanmean(volume_overall_nsds)) if volume_overall_nsds else float("nan")

        per_patient[patient_name] = {
            "per_class": {str(c): volume_per_class[c] for c in volume_per_class},
            "overall": {
                "dice": vol_dice,
                "iou": vol_iou,
                "hd95": None if (np.isnan(vol_hd95) or np.isinf(vol_hd95)) else vol_hd95,
                "nsd": None if np.isnan(vol_nsd) else vol_nsd,
            },
        }
        hd95_str = f", HD95={vol_hd95:.4f}" if not (np.isnan(vol_hd95) or np.isinf(vol_hd95)) else ", HD95=N/A"
        nsd_str = f", NSD={vol_nsd:.4f}" if not np.isnan(vol_nsd) else ", NSD=N/A"
        print(f"{patient_name}: Dice={vol_dice:.4f}, IoU={vol_iou:.4f}{hd95_str}{nsd_str}")

    if n_skipped_no_spacing > 0:
        print(f"\nWarning: {n_skipped_no_spacing} volume(s) 未找到对应 spacing，已计算 Dice/IoU，HD95/NSD 为 nan。"
              f"若需 HD95/NSD，请确保 test_dir（或子目录 img/images/nii）内有对应 .nii.gz，或提供 --spacing_file。")
    if n_skipped_no_volume > 0:
        print(f"\nWarning: {n_skipped_no_volume} volume(s) 在 COCO JSON 中找不到对应切片，已跳过。")

    # 每病例宏平均 (Per-case Mean)，用于 Overall 的样本间变异性 (Inter-patient Mean ± SD)
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
    print(f"{'Category':<20} {'IoU':<12} {'Dice':<12} {'HD95(mm)':<12} {'NSD(2mm)':<12} {'Count':<10}")
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
        print(f"{category_name:<20} {mean_iou:<12.4f} {mean_dice:<12.4f} {mean_hd95:<12.2f} {mean_nsd:<12.4f} {len(ious):<10}")

    print("-" * 100)
    # Overall: 样本间变异性 (Inter-patient)。Step1 每病例宏平均 Metric_Case_i；Step2 Overall_Mean=mean(Metric_Case_i)；Step3 Overall_SD=std(Metric_Case_i)
    if n_cases > 0:
        overall_iou = float(np.nanmean(case_ious))
        overall_dice = float(np.nanmean(case_dices))
        overall_iou_std = float(np.nanstd(case_ious)) if n_cases > 1 else 0.0
        overall_dice_std = float(np.nanstd(case_dices)) if n_cases > 1 else 0.0
        valid_hd95 = [x for x in case_hd95s if np.isfinite(x)]
        valid_nsd = [x for x in case_nsds if not np.isnan(x)]
        overall_hd95 = float(np.mean(valid_hd95)) if valid_hd95 else float("nan")
        overall_nsd = float(np.mean(valid_nsd)) if valid_nsd else float("nan")
        overall_hd95_std = float(np.std(valid_hd95)) if len(valid_hd95) > 1 else (0.0 if valid_hd95 else float("nan"))
        overall_nsd_std = float(np.std(valid_nsd)) if len(valid_nsd) > 1 else (0.0 if valid_nsd else float("nan"))
    else:
        overall_iou = overall_dice = overall_hd95 = overall_nsd = float("nan")
        overall_iou_std = overall_dice_std = overall_hd95_std = overall_nsd_std = float("nan")
    print(f"{'Overall (macro)':<20} {overall_iou:<12.4f} {overall_dice:<12.4f} {overall_hd95:<12.2f} {overall_nsd:<12.4f} {n_cases:<10}")
    print(f"{'Overall (Mean±SD)':<20} IoU={overall_iou:.4f}±{overall_iou_std:.4f}  Dice={overall_dice:.4f}±{overall_dice_std:.4f}  HD95={overall_hd95:.2f}±{overall_hd95_std:.2f}  NSD={overall_nsd:.4f}±{overall_nsd_std:.4f}")
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
            "dice_std": float(np.std(dices)) if len(dices) > 1 else (0.0 if dices else float("nan")),
            "iou_mean": float(np.mean(ious)) if ious else float("nan"),
            "iou_std": float(np.std(ious)) if len(ious) > 1 else (0.0 if ious else float("nan")),
            "hd95_mean": float(np.mean(valid_hd95s)) if valid_hd95s else float("nan"),
            "hd95_std": float(np.std(valid_hd95s)) if len(valid_hd95s) > 1 else (0.0 if valid_hd95s else float("nan")),
            "nsd_mean": float(np.mean(valid_nsds)) if valid_nsds else float("nan"),
            "nsd_std": float(np.std(valid_nsds)) if len(valid_nsds) > 1 else (0.0 if valid_nsds else float("nan")),
            "count": len(ious),
        }

    # Overall: 样本间变异性 (Inter-patient)。Mean/SD 来自 N 个病例的「每病例宏平均」Metric_Case_i
    if n_cases > 0:
        overall_dice_mean = float(np.nanmean(case_dices))
        overall_dice_std = float(np.nanstd(case_dices)) if n_cases > 1 else (0.0 if n_cases else float("nan"))
        overall_iou_mean = float(np.nanmean(case_ious))
        overall_iou_std = float(np.nanstd(case_ious)) if n_cases > 1 else (0.0 if n_cases else float("nan"))
        valid_hd95_for_std = [x for x in case_hd95s if np.isfinite(x)]
        valid_nsd_for_std = [x for x in case_nsds if not np.isnan(x)]
        overall_hd95_mean = float(np.mean(valid_hd95_for_std)) if valid_hd95_for_std else float("nan")
        overall_hd95_std = float(np.std(valid_hd95_for_std)) if len(valid_hd95_for_std) > 1 else (0.0 if valid_hd95_for_std else float("nan"))
        overall_nsd_mean = float(np.mean(valid_nsd_for_std)) if valid_nsd_for_std else float("nan")
        overall_nsd_std = float(np.std(valid_nsd_for_std)) if len(valid_nsd_for_std) > 1 else (0.0 if valid_nsd_for_std else float("nan"))
    else:
        overall_dice_mean = overall_dice_std = overall_iou_mean = overall_iou_std = float("nan")
        overall_hd95_mean = overall_hd95_std = overall_nsd_mean = overall_nsd_std = float("nan")
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

    print(f"\nResults saved to: {output_file}")
    print(f"  Per-class: {len(per_class_summary)} categories, Per-patient: {len(per_patient)} cases")


if __name__ == "__main__":
    main()

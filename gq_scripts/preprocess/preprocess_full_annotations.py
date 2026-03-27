#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 ACDC / PROMISE12 / BTCV 的 3D 医学影像体数据切片并保存为单张图像，同时输出图像级别的 COCO 格式 JSON：
 - `image_annotations.coco.json`：包含 `images` 和按图像的 `annotations`

支持数据集
----------
- acdc：心脏 MRI，gt 命名 patientXXX_frame01_gt.nii.gz
- promise12：前列腺 MRI，gt 命名 CaseXX_segmentation.nii.gz
- btcv_cervix：腹部/宫颈 CT，gt 命名 XXXXXX-Mask.nii.gz（对应 img: XXXXXX-Image.nii.gz）

数据预处理方式
--------------
MRI 模态 (--modality mri)：
1. 百分位裁剪 (percent_clip_to_u8)：仅对非零体素计算，默认 p_low=0.5, p_high=99.5
2. 掩码：仅考虑 vol > 0 的体素，背景 0 不参与百分位计算
3. 裁剪后 Min-Max 归一化，线性映射到 [0, 255] 并转为 uint8

CT 模态 (--modality ct)：
1. 窗宽窗位 (window/level)：根据 --window_level 和 --window_width 将 HU 值映射到 [0, 255]
2. 公式：low = WL - WW/2, high = WL + WW/2；output = (HU - low) / (high - low) * 255，再 clip 到 [0, 255]

标注 (GT) 处理：
   - 直接从 nii.gz 读取，不做强度变换，转为 uint8

切片策略 (slice_policy)：
   - nonempty：仅保存含前景标注的切片
   - all：保存所有切片

输出格式：
   - 图像：PNG
   - 掩码：PNG (mode='L')
   - COCO JSON：含 RLE 分割、bbox、area 等

输出目录结构
------------
  output_dir/
    images/
    masks/
    image_annotations.coco.json

用法示例
--------
# ACDC (MRI)
 python gq_scripts/preprocess/acdc/preprocess_image_annotations.py \
     --dataset acdc \
     --modality mri \
     --img_dir /home/gaoqi/dataset/using/acdc1/train/img \
     --gt_dir /home/gaoqi/dataset/using/acdc1/train/gt \
     --output_dir /home/gaoqi/dataset/using/acdc5/train \
     --slice_policy all

# PROMISE12
 python gq_scripts/preprocess/acdc/preprocess_image_annotations.py \
     --dataset promise12 \
     --modality mri \
     --img_dir /home/gaoqi/dataset/using/promise12_2/train/img \
     --gt_dir /home/gaoqi/dataset/using/promise12_2/train/gt \
     --output_dir /home/gaoqi/dataset/using/promise12_3/train \
     --slice_policy all

# BTCV-cervix (CT)
 python gq_scripts/preprocess/acdc/preprocess_image_annotations.py \
     --dataset btcv_cervix \
     --modality ct \
     --window_level 40 \
     --window_width 400 \
     --img_dir /home/gaoqi/dataset/using/btcv_1/test/img \
     --gt_dir /home/gaoqi/dataset/using/btcv_1/test/gt \
     --output_dir /home/gaoqi/dataset/using/btcv_2/test \
     --slice_policy all

slice_policy: nonempty/all
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils


# 数据集配置：gt 文件名映射、类别映射
# gt_replace: (from_str, to_str) 用于 img_path.name.replace(from_str, to_str) 得到 gt 文件名
DATASET_CONFIG = {
    "acdc": {
        "gt_replace": (".nii.gz", "_gt.nii.gz"),
        "categories": {1: "Right Ventricle", 2: "Myocardium", 3: "Left Ventricle"},
    },
    "promise12": {
        "gt_replace": (".nii.gz", "_segmentation.nii.gz"),
        "categories": {1: "Prostate"},
    },
    "btcv_cervix": {
        "gt_replace": ("-Image.nii.gz", "-Mask.nii.gz"),
        "categories": {1: "Structure_1", 2: "Structure_2", 3: "Structure_3", 4: "Structure_4"},
    },
}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def percent_clip_to_u8(volume, p_low=0.5, p_high=99.5):
    """MRI 预处理：百分位裁剪 + Min-Max 归一化到 [0, 255]。"""
    vol = np.asarray(volume).astype(np.float32)
    mask = vol > 0
    out = np.zeros_like(vol, dtype=np.float32)
    if mask.any():
        lo = np.percentile(vol[mask], p_low)
        hi = np.percentile(vol[mask], p_high)
        vol = np.clip(vol, lo, hi)
        vmin, vmax = vol[mask].min(), vol[mask].max()
        if vmax > vmin:
            out[mask] = (vol[mask] - vmin) / (vmax - vmin) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def window_level_to_u8(volume, window_level=40, window_width=400):
    """CT 预处理：窗宽窗位将 HU 值映射到 [0, 255]。"""
    vol = np.asarray(volume).astype(np.float32)
    low = window_level - window_width / 2
    high = window_level + window_width / 2
    out = (vol - low) / (high - low) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return [float(x_min), float(y_min), float(width), float(height)]


def rle_encode(mask: np.ndarray):
    if not np.any(mask):
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle


def process(img_dir: Path, gt_dir: Path, output_dir: Path, slice_policy: str = 'nonempty',
            dataset: str = 'acdc', modality: str = 'mri', window_level: float = 40, window_width: float = 400):
    """从指定的 img/gt 文件夹读取 nii.gz，切片并写入 output_dir（含 images/、masks/、image_annotations.coco.json）。"""
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"不支持的 dataset: {dataset}，可选: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset]
    category_map = config["categories"]
    gt_from, gt_to = config["gt_replace"]

    if not img_dir.exists():
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"标注目录不存在: {gt_dir}")

    output_dir = output_dir.resolve()
    out_images_dir = output_dir / 'images'
    out_masks_dir = output_dir / 'masks'
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    # Image-level COCO
    image_coco: Dict[str, Any] = {
        'info': {'description': f'{dataset.upper()} (images)'},
        'images': [],
        'annotations': [],
        'categories': [{'id': k, 'name': v} for k, v in category_map.items()],
    }

    image_id = 0
    ann_id = 0

    img_files = sorted(list(img_dir.glob('*.nii.gz')))
    logging.info(f'Processing {len(img_files)} volumes from {img_dir} (dataset={dataset}) ...')

    file_pairs = []
    for img_path in img_files:
        gt_filename = img_path.name.replace(gt_from, gt_to)
        gt_path = gt_dir / gt_filename
        if not gt_path.exists():
            logging.warning(f'Ground truth not found for {img_path.name}, expected {gt_filename}, skipping.')
            continue
        file_pairs.append((img_path, gt_path))

    for img_path, gt_path in tqdm(file_pairs, desc='Volumes'):
        base_name = img_path.stem.replace('.nii', '')

        img_sitk = sitk.ReadImage(str(img_path))
        gt_sitk = sitk.ReadImage(str(gt_path))
        img_vol = sitk.GetArrayFromImage(img_sitk)  # (D, H, W)
        gt_vol = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)

        if modality == 'ct':
            img_vol_u8 = window_level_to_u8(img_vol, window_level, window_width)
        else:
            img_vol_u8 = percent_clip_to_u8(img_vol)

        depth, height, width = img_vol.shape

        for z in range(depth):
            img_slice = img_vol_u8[z]
            gt_slice = gt_vol[z]

            has_fg = np.any(np.isin(gt_slice, list(category_map.keys())))
            if slice_policy == 'nonempty' and not has_fg:
                continue

            slice_fname = f"{base_name}_slice{z:03d}.png"
            save_img = out_images_dir / slice_fname
            save_mask = out_masks_dir / slice_fname

            Image.fromarray(img_slice).save(save_img)
            Image.fromarray(gt_slice.astype(np.uint8), mode='L').save(save_mask)

            # Image-level entry
            image_entry = {
                'id': image_id,
                'file_name': str(Path('images') / slice_fname),
                'mask_file_name': str(Path('masks') / slice_fname),
                'height': int(height),
                'width': int(width),
            }
            image_coco['images'].append(image_entry)

            # Image-level annotations: one annotation per object (category) present in this slice
            unique_labels = np.unique(gt_slice)
            for label in unique_labels:
                if label == 0 or label not in category_map:
                    continue
                binary_mask = (gt_slice == label).astype(np.uint8)
                bbox = mask_to_bbox(binary_mask)
                if bbox is None:
                    continue
                area = float(np.sum(binary_mask))
                rle = rle_encode(binary_mask)
                if rle is None:
                    continue
                ann = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': bbox,
                    'area': area,
                    'segmentation': rle,
                    'iscrowd': 0,
                }
                image_coco['annotations'].append(ann)
                ann_id += 1

            image_id += 1

    # Save JSON file
    image_json_path = output_dir / 'image_annotations.coco.json'

    # Ensure serializable: pycocotools RLE already has 'counts' as str when encoded above
    with open(image_json_path, 'w') as f:
        json.dump(image_coco, f, indent=2)

    logging.info(f"Saved image COCO: {image_json_path} ({len(image_coco['images'])} images, {len(image_coco['annotations'])} anns)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ACDC/PROMISE12/BTCV 3D nii.gz 切片为图像并生成 COCO JSON')
    parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc', 'promise12', 'btcv_cervix'],
                        help='数据集类型')
    parser.add_argument('--modality', type=str, default='mri', choices=['mri', 'ct'],
                        help='影像模态：mri 用百分位裁剪，ct 用窗宽窗位')
    parser.add_argument('--window_level', type=float, default=40,
                        help='CT 窗位 (WL)，仅 modality=ct 时有效')
    parser.add_argument('--window_width', type=float, default=400,
                        help='CT 窗宽 (WW)，仅 modality=ct 时有效')
    parser.add_argument('--img_dir', type=str, required=True, help='包含 .nii.gz 图像的文件夹路径')
    parser.add_argument('--gt_dir', type=str, required=True, help='包含标注 nii.gz 的文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录，将生成 images/、masks/、image_annotations.coco.json')
    parser.add_argument('--slice_policy', type=str, default='nonempty', choices=['all', 'nonempty'])
    args = parser.parse_args()

    setup_logging()
    img_dir = Path(args.img_dir).expanduser().resolve()
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    process(img_dir, gt_dir, output_dir, args.slice_policy, args.dataset,
            modality=args.modality, window_level=args.window_level, window_width=args.window_width)


if __name__ == '__main__':
    main()

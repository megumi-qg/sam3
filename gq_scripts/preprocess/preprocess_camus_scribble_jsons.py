#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 CAMUS 的 scribble 标注处理并保存为 COCO 格式的 JSON（按图像的 annotations）。

输出路径：
 - `dataset/CAMUS/train/image_scribble_annotations.coco.json`
 - `dataset/CAMUS/val/image_scribble_annotations.coco.json`

数据组织假设（仓库内已有结构）：
 - 原始影像： `dataset/CAMUS/ori_train`, `dataset/CAMUS/ori_val`, `dataset/CAMUS/ori_test`
     文件如 `patient0001_2CH_ED.nii.gz`, `patient0001_2CH_ES.nii.gz` 等
 - scribble GT（弱监督 scribble）：位于 `dataset/CAMUS/gt_scribble_train`,
     `dataset/CAMUS/gt_scribble_val`, `dataset/CAMUS/gt_scribble_test`，
     典型命名如 `patient0001_2CH_ED_gt.nii.gz`。

脚本行为：
 - 遍历 `ori_{split}` 中的 `_ED` 与 `_ES` 体数据（例如 `patient0001_2CH_ED.nii.gz`），
     在 `gt_scribble_{split}` 中寻找对应的 scribble 文件（通常 `{base}_gt.nii.gz`），
     若找不到则跳过并记录警告。
 - 为每个切片生成一条 image 记录，并为每个前景 scribble 类（1/2/3）生成 annotation；
     annotation 包含 `segmentation`（该类的 RLE）和 `valid_mask`（切片上所有已知 scribble 区域的并集，忽略 ignore 标签）。

关于 scribble 的标签定义（CAMUS）：
 - `0`: Background scribble（有背景的 scribble 线条）
 - `1`: Left Ventricle
 - `2`: Left Ventricle Myocardium
 - `3`: Left Atrium
 - `4`: ignore label（未标注 / 屏蔽区域）

可视化颜色映射（脚本会导出彩色的 `scribble_masks`）：
 - `0` (Background scribble): 黄色 `(255,255,0)` （突出显示）
 - `1` (Left Ventricle): 红色 `(255,0,0)`
 - `2` (Left Ventricle Myocardium): 绿色 `(0,255,0)`
 - `3` (Left Atrium): 蓝色 `(0,0,255)`
 - `4` (ignore): 黑 `(0,0,0)`（表示被忽略的区域）

注意：脚本不再导出灰度图像（假定你已用全监督预处理生成了 `images/...`），
只会导出彩色 `scribble_masks`，并在 JSON 中保留 `file_name` 指向 `images/...`。

用法举例：
 python aaa_scripts/preprocess/preprocess_camus_scribble_jsons.py --dataset_root dataset/CAMUS
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Set

import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils


IGNORE_LABEL = 4


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def percent_clip_to_u8(volume, p_low=0.5, p_high=99.5):
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


def rle_encode(mask: np.ndarray):
    """Encodes a binary mask (0/1) to COCO RLE format."""
    if not np.any(mask):
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle


def infer_category_name(cat_id: int) -> str:
    # CAMUS 标签语义:
    # 1 -> Left Ventricle
    # 2 -> Left Ventricle Myocardium
    # 3 -> Left Atrium
    if cat_id == 1:
        return 'Left Ventricle'
    if cat_id == 2:
        return 'Left Ventricle Myocardium'
    if cat_id == 3:
        return 'Left Atrium'
    return f'class_{cat_id}'


def process_split(dataset_root: Path, split: str, slice_policy: str = 'nonempty'):
    ori_dir = dataset_root / f'ori_{split}'
    scribble_dir = dataset_root / f'gt_scribble_{split}'

    if not ori_dir.exists():
        logging.warning(f'ori directory not found: {ori_dir}, skipping {split}.')
        return
    if not scribble_dir.exists():
        logging.warning(f'scribble directory not found: {scribble_dir}, skipping {split}.')
        return

    out_split = dataset_root / split
    out_images_dir = out_split / 'images'
    out_scribble_masks_dir = out_split / 'scribble_masks'
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_scribble_masks_dir.mkdir(parents=True, exist_ok=True)

    image_coco: Dict[str, Any] = {
        'info': {'description': f'CAMUS {split} scribble annotations (images)'},
        'images': [],
        'annotations': [],
        'categories': [],
    }

    image_id = 0
    ann_id = 0
    observed_cats: Set[int] = set()

    # Collect all volume files (ED and ES) recursively (handle patient subfolders)
    vol_files = sorted(list(ori_dir.rglob('*.nii.gz')))
    logging.info(f'Found {len(vol_files)} volumes under {ori_dir} (including subfolders) (split={split})')

    # We'll process files that contain _ED or _ES in their stem (common CAMUS naming)
    # Exclude _gt files as they are annotation files, not original images
    candidates = [p for p in vol_files if ('_ED' in p.stem or '_ES' in p.stem) and '_gt' not in p.stem]

    for vol_path in tqdm(candidates, desc=f'Volumes ({split})'):
        # Handle .nii.gz double extension: stem only removes .gz, need to also remove .nii
        base = vol_path.stem
        if base.endswith('.nii'):
            base = base[:-4]
        # Remove '_gt' suffix if present to avoid double '_gt'
        if base.endswith('_gt'):
            base = base[:-3]
        
        # Try multiple naming patterns
        scribble_candidates = [
            scribble_dir / f'{base}_gt.nii.gz',  # Standard: base_gt.nii.gz
            scribble_dir / f'{base}.nii.gz',     # Fallback: base.nii.gz (if already has _gt in name)
        ]
        
        # Also try recursive search in case files are in subdirectories
        scribble_candidate = None
        for candidate in scribble_candidates:
            if candidate.exists():
                scribble_candidate = candidate
                break
        
        # If not found, try recursive search
        if scribble_candidate is None:
            # Search recursively for matching scribble file
            pattern = f'{base}_gt.nii.gz'
            matches = list(scribble_dir.rglob(pattern))
            if matches:
                scribble_candidate = matches[0]
            else:
                # Try without _gt suffix
                pattern2 = f'{base}.nii.gz'
                matches2 = list(scribble_dir.rglob(pattern2))
                if matches2:
                    scribble_candidate = matches2[0]
        
        if scribble_candidate is None or not scribble_candidate.exists():
            searched_paths = [str(c) for c in scribble_candidates]
            logging.warning(f'Scribble file not found for {vol_path.name} (base={base}, searched: {searched_paths}), skipping.')
            continue

        img_sitk = sitk.ReadImage(str(vol_path))
        scribble_sitk = sitk.ReadImage(str(scribble_candidate))
        img_vol = sitk.GetArrayFromImage(img_sitk)  # (D, H, W) or (H, W) for 2D
        scribble_vol = sitk.GetArrayFromImage(scribble_sitk).astype(np.int32)

        # Handle both 2D (single slice) and 3D volumes
        # Ensure img_vol and scribble_vol have the same dimensions
        if img_vol.ndim == 2:
            # 2D image: add a dimension to make it (1, H, W)
            img_vol = img_vol[np.newaxis, ...]
        elif img_vol.ndim != 3:
            logging.warning(f'Unexpected image dimensions {img_vol.shape} for {vol_path.name}, skipping.')
            continue
        
        if scribble_vol.ndim == 2:
            # 2D scribble: add a dimension to make it (1, H, W)
            scribble_vol = scribble_vol[np.newaxis, ...]
        elif scribble_vol.ndim != 3:
            logging.warning(f'Unexpected scribble dimensions {scribble_vol.shape} for {scribble_candidate.name}, skipping.')
            continue
        
        # Ensure dimensions match
        if img_vol.shape != scribble_vol.shape:
            logging.warning(f'Image shape {img_vol.shape} does not match scribble shape {scribble_vol.shape} for {vol_path.name}, skipping.')
            continue

        img_vol_u8 = percent_clip_to_u8(img_vol)

        depth, height, width = img_vol.shape

        # Warn if multiple slices will generate the same filename
        if depth > 1:
            logging.warning(f'Volume {vol_path.name} has {depth} slices, but will generate files without slice index. '
                          f'Only the last processed slice will be saved for each volume.')

        for z in range(depth):
            img_slice = img_vol_u8[z]
            scribble_slice = scribble_vol[z]

            # Check foreground labels (exclude background 0 and ignore label)
            unique_labels = np.unique(scribble_slice)
            valid_fg_labels = [int(l) for l in unique_labels if l != 0 and l != IGNORE_LABEL]
            has_fg = len(valid_fg_labels) > 0

            if slice_policy == 'nonempty' and not has_fg:
                continue

            slice_fname = f"{base}.png"
            save_img = out_images_dir / slice_fname
            save_scribble_mask = out_scribble_masks_dir / slice_fname

            # 图像已由全监督预处理导出，跳过再次导出。

            # 保存彩色 scribble mask（不同类别用不同颜色）
            if not save_scribble_mask.exists():
                m = scribble_slice.copy()
                # 保留 ignore label 为独立类别（不转换为 0），可视化时用黑色表示

                # 颜色映射：0 背景 -> 黑, 1 -> 红, 2 -> 绿, 3 -> 蓝, 其他按哈希生成颜色
                h, w = m.shape
                rgb = np.zeros((h, w, 3), dtype=np.uint8)

                def label_to_color(lbl: int):
                    # 0 背景 -> 黄色（突出）
                    if lbl == 0:
                        return (255, 255, 0)
                    if lbl == 1:
                        return (255, 0, 0)
                    if lbl == 2:
                        return (0, 255, 0)
                    if lbl == 3:
                        return (0, 0, 255)
                    # 4 ignore -> 黑色
                    if lbl == 4:
                        return (0, 0, 0)
                    # simple deterministic color for other labels
                    np.random.seed(int(lbl))
                    return tuple((np.random.randint(50, 256, size=3)).tolist())

                unique = np.unique(m)
                for lbl in unique:
                    color = label_to_color(int(lbl))
                    mask_l = (m == lbl)
                    if mask_l.any():
                        rgb[mask_l] = color

                Image.fromarray(rgb, mode='RGB').save(save_scribble_mask)

            image_entry = {
                'id': image_id,
                'file_name': str(Path('images') / slice_fname),
                'mask_file_name': str(Path('scribble_masks') / slice_fname),
                'height': int(height),
                'width': int(width),
            }
            image_coco['images'].append(image_entry)

            union_mask = (scribble_slice != IGNORE_LABEL).astype(np.uint8)
            union_rle = rle_encode(union_mask)

            for label in valid_fg_labels:
                binary_mask = (scribble_slice == label).astype(np.uint8)
                if not np.any(binary_mask):
                    continue
                area = float(np.sum(binary_mask))
                rle = rle_encode(binary_mask)
                if rle is None:
                    continue

                ann = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': int(label),
                    'area': area,
                    'segmentation': rle,
                    'valid_mask': union_rle,
                    'iscrowd': 0,
                }
                image_coco['annotations'].append(ann)
                ann_id += 1
                observed_cats.add(int(label))

            image_id += 1

    # Build categories list from observed labels
    cats = sorted(list(observed_cats))
    image_coco['categories'] = [{'id': c, 'name': infer_category_name(c)} for c in cats]

    image_json_path = out_split / 'image_scribble_annotations.coco.json'
    with open(image_json_path, 'w') as f:
        json.dump(image_coco, f, indent=2)

    logging.info(f"Saved scribble COCO: {image_json_path} ({len(image_coco['images'])} images, {len(image_coco['annotations'])} anns)")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='dataset/CAMUS')
    parser.add_argument('--slice_policy', type=str, default='nonempty', choices=['all', 'nonempty'])
    args = parser.parse_args()

    setup_logging()
    dataset_root = Path(args.dataset_root)

    for split in ['train', 'val']:
        process_split(dataset_root, split, args.slice_policy)


if __name__ == '__main__':
    main()

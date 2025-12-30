#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 ACDC 的 3D MRI 体数据切片并保存为单张图像，同时输出两份 COCO 格式的 JSON：
 - `image_annotations.coco.json`：包含 `images` 和按图像的 `annotations`
 - `video_annotations.coco.json`：包含 `videos` 和按 video 的 `annotations`（每个 annotation 含每帧的 segmentations/bboxes/areas 列表）

输出目录结构（默认）：
 dataset/sam3/ACDC_new/train/
   images/
   masks/
   image_annotations.coco.json
   video_annotations.coco.json
 dataset/sam3/ACDC_new/test/ ...

用法示例：
 python scripts/preprocess_acdc_split_jsons.py \
     --dataset_root dataset/ACDC \
     --output_root dataset/sam3/ACDC_new \
     --slice_policy nonempty

注：视频级别的 `segmentations` 使用 COCO RLE（pycocotools），每帧若无目标会以 `null`/`None` 占位。
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


# ACDC 类映射
CATEGORY_MAP = {
    1: "Right Ventricle",
    2: "Myocardium",
    3: "Left Ventricle",
}


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


def process_split(dataset_root: Path, output_root: Path, split: str, slice_policy: str = 'nonempty'):
    split_dir_name = f"pre_{split}"
    img_dir = dataset_root / split_dir_name / "img"
    gt_dir = dataset_root / split_dir_name / "gt"

    if not img_dir.exists() or not gt_dir.exists():
        logging.warning(f"Preprocessed directories not found: {img_dir} or {gt_dir}, skipping {split}.")
        return

    out_split = output_root / split
    out_images_dir = out_split / 'images'
    out_masks_dir = out_split / 'masks'
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    # Image-level COCO
    image_coco: Dict[str, Any] = {
        'info': {'description': f'ACDC {split} (images)'},
        'images': [],
        'annotations': [],
        'categories': [{'id': k, 'name': v} for k, v in CATEGORY_MAP.items()],
    }

    # Video-level COCO
    video_coco: Dict[str, Any] = {
        'info': {'description': f'ACDC {split} (videos)'},
        'videos': [],
        'annotations': [],
        'categories': [{'id': k, 'name': v} for k, v in CATEGORY_MAP.items()],
    }

    image_id = 0
    ann_id = 0
    video_id = 0

    img_files = sorted(list(img_dir.glob('*.nii.gz')))
    logging.info(f'Processing {len(img_files)} volumes in {split_dir_name}...')

    file_pairs = []
    for img_path in img_files:
        gt_filename = img_path.name.replace('.nii.gz', '_gt.nii.gz')
        gt_path = gt_dir / gt_filename
        if not gt_path.exists():
            logging.warning(f'Ground truth not found for {img_path.name}, expected {gt_filename}, skipping.')
            continue
        file_pairs.append((img_path, gt_path))

    for img_path, gt_path in tqdm(file_pairs, desc=f'Volumes ({split})'):
        vid = video_id
        video_id += 1
        base_name = img_path.stem.replace('.nii', '')

        img_sitk = sitk.ReadImage(str(img_path))
        gt_sitk = sitk.ReadImage(str(gt_path))
        img_vol = sitk.GetArrayFromImage(img_sitk)  # (D, H, W)
        gt_vol = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)

        img_vol_u8 = percent_clip_to_u8(img_vol)

        depth, height, width = img_vol.shape

        # Prepare video entry
        video_entry = {
            'id': vid,
            'video_name': base_name,
            'file_names': [],
            'height': int(height),
            'width': int(width),
            'length': 0,
        }

        # Per-video accumulator for object tracks: {category_id: {'segmentations': [], 'bboxes': [], 'areas': []}}
        per_object: Dict[int, Dict[str, List[Any]]] = {k: {'segmentations': [], 'bboxes': [], 'areas': []} for k in CATEGORY_MAP.keys()}

        for z in range(depth):
            img_slice = img_vol_u8[z]
            gt_slice = gt_vol[z]

            has_fg = np.any(np.isin(gt_slice, list(CATEGORY_MAP.keys())))
            if slice_policy == 'nonempty' and not has_fg:
                # For video-level consistency we must still keep frame ordering; but to reduce size we skip entirely.
                # We will NOT include skipped frames in video file_names; thus segmentations/bboxes lists will align to included frames only.
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
                if label == 0 or label not in CATEGORY_MAP:
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

            # Update per-video accumulators for each category (for this included frame)
            video_entry['file_names'].append(str(Path('images') / slice_fname))
            for cat in CATEGORY_MAP.keys():
                binary_mask = (gt_slice == cat).astype(np.uint8)
                if np.any(binary_mask):
                    per_object[cat]['segmentations'].append(rle_encode(binary_mask))
                    per_object[cat]['bboxes'].append(mask_to_bbox(binary_mask))
                    per_object[cat]['areas'].append(float(np.sum(binary_mask)))
                else:
                    per_object[cat]['segmentations'].append(None)
                    per_object[cat]['bboxes'].append(None)
                    per_object[cat]['areas'].append(0.0)

            image_id += 1

        # finalize video entry
        video_entry['length'] = len(video_entry['file_names'])
        video_coco['videos'].append(video_entry)

        # Build video-level annotations: one per category that appears at least once
        for cat, accum in per_object.items():
            # Check if any frame had the object
            if any([s is not None for s in accum['segmentations']]):
                video_ann = {
                    'id': len(video_coco['annotations']),
                    'video_id': vid,
                    'category_id': int(cat),
                    'segmentations': accum['segmentations'],  # list of RLE dicts or None
                    'bboxes': accum['bboxes'],
                    'areas': accum['areas'],
                    'iscrowd': 0,
                    'height': int(height),
                    'width': int(width),
                }
                video_coco['annotations'].append(video_ann)

    # Save JSON files
    image_json_path = out_split / 'image_annotations.coco.json'
    video_json_path = out_split / 'video_annotations.coco.json'

    # Ensure serializable: pycocotools RLE already has 'counts' as str when encoded above
    with open(image_json_path, 'w') as f:
        json.dump(image_coco, f, indent=2)

    with open(video_json_path, 'w') as f:
        json.dump(video_coco, f, indent=2)

    logging.info(f"Saved image COCO: {image_json_path} ({len(image_coco['images'])} images, {len(image_coco['annotations'])} anns)")
    logging.info(f"Saved video COCO: {video_json_path} ({len(video_coco['videos'])} videos, {len(video_coco['annotations'])} anns)")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='dataset/ACDC')
    parser.add_argument('--output_root', type=str, default='dataset/sam3/ACDC_new')
    parser.add_argument('--slice_policy', type=str, default='nonempty', choices=['all', 'nonempty'])
    args = parser.parse_args()

    setup_logging()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    for split in ['train', 'test']:
        process_split(dataset_root, output_root, split, args.slice_policy)


if __name__ == '__main__':
    main()

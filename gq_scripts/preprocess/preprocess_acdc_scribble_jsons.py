#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 ACDC 的 scribble 标注处理并保存为 COCO 格式的 JSON：
 - `image_scribble_annotations.coco.json`：包含 `images` 和按图像的 `annotations`（不包含 bbox）

scribble_bench 文件夹中的标签：
 - 0: 背景 (Background Scribble)
 - 1: 右心室 (Right Ventricle)
 - 2: 心肌 (Myocardium)
 - 3: 左心室 (Left Ventricle)
 - 4: ignore label（没有 scribble 标注的地方，需要忽略）

修改说明 (方案一)：
 在每个 annotation 中增加 'valid_mask' 字段。
 - segmentation: 当前类别的 scribble (1, 2, or 3)
 - valid_mask: 该切片上所有有效 scribble 的并集 (0 + 1 + 2 + 3)，即排除 label 4 的所有区域。
   这样在训练时：
     - segmentation=1 -> Positive (1)
     - valid_mask=1 & segmentation=0 -> Negative (0)
     - valid_mask=0 -> Ignore (255)

用法示例：
 python aaa_scripts/preprocess/preprocess_acdc_scribble_jsons.py \
     --input_img_dir /home/gaoqi/dataset/using/acdc2/train/img \
     --input_scribble_dir /home/gaoqi/dataset/using/acdc2/train/scribble_bench \
     --output_images_dir /home/gaoqi/dataset/using/acdc3/train/images1 \
     --output_scribble_masks_dir /home/gaoqi/dataset/using/acdc3/train/scribble_masks \
     --output_json_path /home/gaoqi/dataset/using/acdc3/train/image_scribble_annotations1.coco.json \
     --slice_policy nonempty
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


# ACDC 类映射（忽略 label 4，也不为 label 0 生成前景框，但 0 会被包含在 valid_mask 中）
CATEGORY_MAP = {
    1: "Right Ventricle",
    2: "Myocardium",
    3: "Left Ventricle",
}
IGNORE_LABEL = 4  # ignore label，需要忽略 (255)

# 可视化颜色映射：将 label 值映射到灰度值以便可视化
# 0 (背景) -> 64 (较暗的灰)
# 1 (RV) -> 128 (中等灰)
# 2 (Myo) -> 192 (较亮的灰)
# 3 (LV) -> 255 (白色)
# 4 (Ignore) -> 0 (黑色)
VISUALIZATION_MAP = {
    0: 64,     # 背景 -> 较暗的灰
    1: 128,    # RV -> 中等灰
    2: 192,    # Myo -> 较亮的灰
    3: 255,    # LV -> 白色
    4: 0,      # Ignore -> 黑色
}


def apply_visualization_map(mask: np.ndarray) -> np.ndarray:
    """将 label mask 映射到可视化灰度值"""
    result = np.zeros_like(mask, dtype=np.uint8)
    for label, gray_value in VISUALIZATION_MAP.items():
        result[mask == label] = gray_value
    return result


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


def process_split(
    img_dir: Path,
    scribble_dir: Path,
    out_images_dir: Path,
    out_scribble_masks_dir: Path,
    output_json_path: Path,
    split: str = 'train',
    slice_policy: str = 'nonempty'
):
    """
    处理单个split的数据
    
    Args:
        img_dir: 输入图像目录（包含 .nii.gz 文件）
        scribble_dir: 输入scribble标注目录（包含 .nii.gz 文件）
        out_images_dir: 输出图像PNG目录
        out_scribble_masks_dir: 输出scribble mask PNG目录
        output_json_path: 输出JSON文件路径
        split: split名称（用于日志）
        slice_policy: 切片策略 ('all' 或 'nonempty')
    """
    if not img_dir.exists() or not scribble_dir.exists():
        logging.warning(f"Input directories not found: {img_dir} or {scribble_dir}, skipping.")
        return

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_scribble_masks_dir.mkdir(parents=True, exist_ok=True)

    # Image-level COCO for scribble annotations
    image_coco: Dict[str, Any] = {
        'info': {'description': f'ACDC {split} scribble annotations (images)'},
        'images': [],
        'annotations': [],
        'categories': [{'id': k, 'name': v} for k, v in CATEGORY_MAP.items()],
    }

    image_id = 0
    ann_id = 0

    img_files = sorted(list(img_dir.glob('*.nii.gz')))
    logging.info(f'Processing {len(img_files)} volumes in {split}...')

    file_pairs = []
    for img_path in img_files:
        base_name_without_ext = img_path.stem.replace('.nii', '')
        scribble_filename_1 = f"{base_name_without_ext}_scribble.nii.gz"
        scribble_filename_2 = f"{base_name_without_ext}_gt.nii.gz"
        scribble_filename_3 = img_path.name  
        
        scribble_path = None
        for scribble_filename in [scribble_filename_1, scribble_filename_2, scribble_filename_3]:
            candidate_path = scribble_dir / scribble_filename
            if candidate_path.exists():
                scribble_path = candidate_path
                break
        
        if scribble_path is None:
            logging.warning(f'Scribble file not found for {img_path.name}, skipping.')
            continue
        
        file_pairs.append((img_path, scribble_path))

    for img_path, scribble_path in tqdm(file_pairs, desc=f'Volumes ({split})'):
        base_name = img_path.stem.replace('.nii', '')

        img_sitk = sitk.ReadImage(str(img_path))
        scribble_sitk = sitk.ReadImage(str(scribble_path))
        img_vol = sitk.GetArrayFromImage(img_sitk)  # (D, H, W)
        scribble_vol = sitk.GetArrayFromImage(scribble_sitk).astype(np.uint8)

        img_vol_u8 = percent_clip_to_u8(img_vol)

        depth, height, width = img_vol.shape

        for z in range(depth):
            img_slice = img_vol_u8[z]
            scribble_slice = scribble_vol[z]

            # 检查是否有有效的 Foreground 标注 (1, 2, 3)
            # 背景(0) 和 Ignore(4) 不算 Foreground
            valid_fg_labels = [l for l in np.unique(scribble_slice) if l in CATEGORY_MAP]
            has_fg = len(valid_fg_labels) > 0
            
            if slice_policy == 'nonempty' and not has_fg:
                continue

            slice_fname = f"{base_name}_slice{z:03d}.png"
            save_img = out_images_dir / slice_fname
            save_scribble_mask = out_scribble_masks_dir / slice_fname

            # 保存图像
            if not save_img.exists():
                Image.fromarray(img_slice).save(save_img)
            
            # 保存可视化 Mask (可选，仅用于肉眼检查)
            # 将 label 值映射到可视化灰度值：
            # 背景(0) -> 较暗的灰(64)，RV(1) -> 中等灰(128)，Myo(2) -> 较亮的灰(192)，LV(3) -> 白色(255)，Ignore(4) -> 黑色(0)
            if not save_scribble_mask.exists():
                scribble_mask_visualized = apply_visualization_map(scribble_slice)
                Image.fromarray(scribble_mask_visualized, mode='L').save(save_scribble_mask)

            # Image-level entry
            image_entry = {
                'id': image_id,
                'file_name': str(Path('images') / slice_fname),
                'mask_file_name': str(Path('scribble_masks') / slice_fname),
                'height': int(height),
                'width': int(width),
            }
            image_coco['images'].append(image_entry)

            # === 计算 Valid Mask (Union of all known regions) ===
            # 有效区域 = 所有不等于 IGNORE_LABEL(4) 的区域
            # 这包括：Background(0), RV(1), Myo(2), LV(3)
            # 这些区域在 Loss 计算中是 "已知" 的 (要么正，要么负)
            union_mask = (scribble_slice != IGNORE_LABEL).astype(np.uint8)
            union_rle = rle_encode(union_mask)

            # Image-level annotations
            unique_labels = np.unique(scribble_slice)
            for label in unique_labels:
                # 只为前景物体 (1, 2, 3) 创建 Annotation
                if label not in CATEGORY_MAP:
                    continue
                
                # 创建该类别的二值 mask（Positive Sample）
                # 注意：binary_mask 只有当前类别是 1，其他都是 0
                binary_mask = (scribble_slice == label).astype(np.uint8)
                
                if not np.any(binary_mask):
                    continue
                
                area = float(np.sum(binary_mask))
                rle = rle_encode(binary_mask)
                if rle is None:
                    continue
                
                # 创建 Annotation
                ann = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': int(label),
                    'area': area,
                    'segmentation': rle,    # 正样本 Mask (仅当前类)
                    'valid_mask': union_rle, # 【新增】有效区域 Mask (当前类+其他类+背景)
                    'iscrowd': 0,
                }
                image_coco['annotations'].append(ann)
                ann_id += 1

            image_id += 1

    # Save JSON file
    # Ensure serializable
    with open(output_json_path, 'w') as f:
        json.dump(image_coco, f, indent=2)

    logging.info(f"Saved scribble COCO: {output_json_path} ({len(image_coco['images'])} images, {len(image_coco['annotations'])} anns)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='处理 ACDC scribble 标注并保存为 COCO 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python %(prog)s \\
      --input_img_dir /home/gaoqi/dataset/using/acdc2/train/img \\
      --input_scribble_dir /home/gaoqi/dataset/using/acdc2/train/scribble_bench \\
      --output_images_dir /home/gaoqi/dataset/using/acdc3/train/images \\
      --output_scribble_masks_dir /home/gaoqi/dataset/using/acdc3/train/scribble_masks \\
      --output_json_path /home/gaoqi/dataset/using/acdc3/train/image_scribble_annotations.coco.json \\
      --slice_policy nonempty
        """
    )
    
    parser.add_argument('--input_img_dir', type=str, required=True,
                        help='输入图像目录（包含 .nii.gz 文件）')
    parser.add_argument('--input_scribble_dir', type=str, required=True,
                        help='输入scribble标注目录（包含 .nii.gz 文件）')
    parser.add_argument('--output_images_dir', type=str, required=True,
                        help='输出图像PNG目录')
    parser.add_argument('--output_scribble_masks_dir', type=str, required=True,
                        help='输出scribble mask PNG目录')
    parser.add_argument('--output_json_path', type=str, required=True,
                        help='输出JSON文件路径')
    parser.add_argument('--slice_policy', type=str, default='nonempty', choices=['all', 'nonempty'],
                        help='切片策略：all=所有切片，nonempty=仅包含前景标注的切片')
    
    args = parser.parse_args()

    setup_logging()

    img_dir = Path(args.input_img_dir)
    scribble_dir = Path(args.input_scribble_dir)
    out_images_dir = Path(args.output_images_dir)
    out_scribble_masks_dir = Path(args.output_scribble_masks_dir)
    output_json_path = Path(args.output_json_path)
    
    # 从路径中推断split名称（用于日志）
    split = 'train'  # 默认值
    if 'train' in str(img_dir):
        split = 'train'
    elif 'test' in str(img_dir):
        split = 'test'
    
    process_split(
        img_dir=img_dir,
        scribble_dir=scribble_dir,
        out_images_dir=out_images_dir,
        out_scribble_masks_dir=out_scribble_masks_dir,
        output_json_path=output_json_path,
        split=split,
        slice_policy=args.slice_policy
    )


if __name__ == '__main__':
    main()
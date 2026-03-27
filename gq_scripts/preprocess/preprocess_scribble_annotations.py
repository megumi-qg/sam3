#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 scribble 标注处理并保存为 COCO 格式的 JSON。
支持 ACDC、BTCV、PROMISE 数据集。仅处理 scribble 并生成 annotation，不输出图像。

annotation 字段：
 - segmentation: 当前类别的 scribble (RLE)
 - valid_mask: 有效区域并集，排除 ignore label

ACDC
python gq_scripts/preprocess/preprocess_scribble_annotations.py \
    --dataset acdc \
    --input_img_dir /home/gaoqi/dataset/using/acdc1/train/img \
    --input_scribble_dir /home/gaoqi/dataset/using/acdc1/train/scribble_bench \
    --output_json_path /home/gaoqi/dataset/using/acdc4/train/scribble_annotations.coco.json \
    --slice_policy all

BTCV
python gq_scripts/preprocess/preprocess_scribble_annotations.py \
    --dataset btcv \
    --input_img_dir /home/gaoqi/dataset/using/btcv_1/train/img \
    --input_scribble_dir /home/gaoqi/dataset/using/btcv_1/train/scribble_bench \
    --output_json_path /home/gaoqi/dataset/using/btcv_2/train/scribble_annotations.coco.json \
    --slice_policy all

PROMISE
python gq_scripts/preprocess/preprocess_scribble_annotations.py \
    --dataset promise \
    --input_img_dir /home/gaoqi/dataset/using/promise12_2/train/img \
    --input_scribble_dir /home/gaoqi/dataset/using/promise12_2/train/scribble_bench \
    --output_json_path /home/gaoqi/dataset/using/promise12_3/train/scribble_annotations.coco.json \
    --slice_policy all

"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from pycocotools import mask as mask_utils

# 各数据集的类别映射、ignore label、scribble 文件命名规则
# scribble 命名: 传入 (base_name, img_name) -> 返回候选 scribble 文件名列表
DATASET_CONFIG = {
    'acdc': (
        {1: "right ventricle", 2: "myocardium", 3: "left ventricle"},
        4,
        lambda b, n: [f"{b}_scribble.nii.gz", f"{b}_gt.nii.gz", n],
    ),
    'btcv': (
        {1: "bladder", 2: "uterus", 3: "rectum", 4: "small bowel"},
        5,
        lambda b, n: [n.replace("-Image", "-Mask")],  # 0759564-Image.nii.gz -> 0759564-Mask.nii.gz
    ),
    'promise': (
        {1: "prostate"},
        2,
        lambda b, n: [f"{b}_segmentation.nii.gz"],  # Case00.nii.gz -> Case00_segmentation.nii.gz
    ),
}


def rle_encode(mask: np.ndarray):
    """将二值 mask 编码为 COCO RLE 格式"""
    if not np.any(mask):
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle


def process_split(
    img_dir: Path,
    scribble_dir: Path,
    output_json_path: Path,
    category_map: Dict[int, str],
    ignore_label: int,
    get_scribble_candidates: Callable[[str, str], List[str]],
    dataset_name: str = '',
    split: str = 'train',
    slice_policy: str = 'nonempty'
):
    """处理 scribble 标注并生成 COCO 格式 JSON"""
    if not img_dir.exists() or not scribble_dir.exists():
        logging.warning(f"Input directories not found: {img_dir} or {scribble_dir}, skipping.")
        return

    image_coco: Dict[str, Any] = {
        'info': {'description': f'{dataset_name} {split} scribble annotations'},
        'images': [],
        'annotations': [],
        'categories': [{'id': k, 'name': v} for k, v in category_map.items()],
    }

    image_id = 0
    ann_id = 0

    img_files = sorted(img_dir.glob('*.nii.gz'))
    file_pairs = []
    for img_path in img_files:
        base_name = img_path.stem.replace('.nii', '')
        img_name = img_path.name
        for scribble_filename in get_scribble_candidates(base_name, img_name):
            candidate = scribble_dir / scribble_filename
            if candidate.exists():
                file_pairs.append((img_path, candidate))
                break
        else:
            logging.warning(f'Scribble file not found for {img_path.name}, skipping.')

    logging.info(f'Processing {len(file_pairs)} volumes in {split}...')

    for img_path, scribble_path in tqdm(file_pairs, desc=f'Volumes ({split})'):
        base_name = img_path.stem.replace('.nii', '')
        scribble_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(scribble_path))).astype(np.uint8)
        depth, height, width = scribble_vol.shape

        for z in range(depth):
            scribble_slice = scribble_vol[z]
            has_fg = any(l in category_map for l in np.unique(scribble_slice))

            if slice_policy == 'nonempty' and not has_fg:
                continue

            slice_fname = f"{base_name}_slice{z:03d}.png"
            image_coco['images'].append({
                'id': image_id,
                'file_name': str(Path('images') / slice_fname),
                'height': int(height),
                'width': int(width),
            })

            union_mask = (scribble_slice != ignore_label).astype(np.uint8)
            union_rle = rle_encode(union_mask)

            for label in np.unique(scribble_slice):
                if label not in category_map:
                    continue
                binary_mask = (scribble_slice == label).astype(np.uint8)
                if not np.any(binary_mask):
                    continue
                rle = rle_encode(binary_mask)
                if rle is None:
                    continue

                image_coco['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': int(label),
                    'area': float(np.sum(binary_mask)),
                    'segmentation': rle,
                    'valid_mask': union_rle,
                    'iscrowd': 0,
                })
                ann_id += 1

            image_id += 1

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(image_coco, f, indent=2)

    logging.info(f"Saved: {output_json_path} ({len(image_coco['images'])} images, {len(image_coco['annotations'])} anns)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='处理 ACDC scribble 标注并保存为 COCO 格式 JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  ACDC:  python %(prog)s --dataset acdc --input_img_dir ... --input_scribble_dir ... --output_json_path ...
  BTCV:  python %(prog)s --dataset btcv --input_img_dir ... --input_scribble_dir ... --output_json_path ...
  PROMISE: python %(prog)s --dataset promise --input_img_dir ... --input_scribble_dir ... --output_json_path ...
        """
    )
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'btcv', 'promise'],
                        help='数据集类型：acdc / btcv / promise')
    parser.add_argument('--input_img_dir', type=str, required=True,
                        help='输入图像目录（.nii.gz，用于匹配 scribble 文件）')
    parser.add_argument('--input_scribble_dir', type=str, required=True,
                        help='输入 scribble 标注目录')
    parser.add_argument('--output_json_path', type=str, required=True,
                        help='输出 JSON 文件路径')
    parser.add_argument('--slice_policy', type=str, default='nonempty', choices=['all', 'nonempty'],
                        help='all=所有切片, nonempty=仅含前景标注的切片')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    category_map, ignore_label, get_scribble_candidates = DATASET_CONFIG[args.dataset]
    img_dir = Path(args.input_img_dir)
    scribble_dir = Path(args.input_scribble_dir)
    output_json_path = Path(args.output_json_path)
    split = 'test' if 'test' in str(img_dir) else 'train'

    process_split(
        img_dir=img_dir,
        scribble_dir=scribble_dir,
        output_json_path=output_json_path,
        category_map=category_map,
        ignore_label=ignore_label,
        get_scribble_candidates=get_scribble_candidates,
        dataset_name=args.dataset.upper(),
        split=split,
        slice_policy=args.slice_policy
    )


if __name__ == '__main__':
    main()

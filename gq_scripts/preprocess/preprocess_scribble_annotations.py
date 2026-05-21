#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
将 3D 医学体数据的 scribble 标注导出为 COCO 风格 JSON（仅元数据与 RLE，不写 PNG）。

支持的数据集
------------
脚本内置 **acdc、btcv、promise、isbi、mscmr**。``--dataset`` 决定类别、ignore 规则与 scribble 文件名匹配。

**mscmr（仅 train）**：只应对 **train** 的 ``images`` / ``labels`` 生成 **一份** scribble JSON；**不要**对 val/test 调用本脚本。
标签 **1=right ventricle，2=myocardium，3=left ventricle**；**4 为 ignore**，不参与类别与 RLE 实例，``valid_mask`` 为 ``(slice != 4)``（与 ACDC scribble 一致）。
文件：``subjectXX_DE.nii.gz`` ↔ ``subjectXX_DE_scribble.nii.gz``。

输入
----
- --dataset：acdc / btcv / promise / mscmr（决定类别名、ignore 标签、scribble 文件名规则）。
- --input_img_dir：含 *.nii.gz 的图像目录；脚本按文件名匹配同病例的 scribble 文件。
- --input_scribble_dir：与图像一一对应的 scribble 体数据（NIfTI，整型标签）。
- --output_json_path：输出的 .json 路径（父目录不存在会自动创建）。
- --slice_policy：nonempty（默认，跳过无任何前景类别的切片）或 all（每层切片都生成一条 image 记录）。

处理流程
--------
1. 遍历图像目录下每个 *.nii.gz，按数据集规则在 scribble 目录中查找第一个存在的候选文件。
2. 读入整卷 scribble，按 z 维切片；按 slice_policy 决定是否保留该层。
3. 对每层：写入一条 images 记录（file_name 为虚拟路径 images/{病例}_slice{zzz}.png，便于与下游 2D 管线对齐；不导出真实 PNG）。
4. 对该层每个出现在 category_map 中的标签：segmentation 为该类的二值 mask 的 COCO RLE；valid_mask 为「非 ignore 标签」区域的 RLE（并集），用于约束有效监督区域。
   对 ISBI，这意味着 `0=ignore`，而 `1/2/3` 都属于有效 scribble 区域，其中 `3` 仅作为背景监督证据进入 valid_mask，
   不会被导出为前景实例。

输出
----
一个 JSON 对象，含 info、images、annotations、categories；每条 annotation 含 segmentation、valid_mask、
area、iscrowd 等 COCO 常用字段。

示例:

    python gq_scripts/preprocess_scribble_annotations.py \
        --dataset acdc \
        --input_img_dir /home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_variants/train/img \
        --input_scribble_dir /home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_variants/train/acdc_scribbles_TMI \
        --output_json_path /home/gaoqi/dataset/using/acdc/processed/png_coco_sam3_fullframes_weak/train/scribble_tmi_annotations.coco.json \
        --slice_policy all

    MSCMR train::

        python gq_scripts/preprocess/preprocess_scribble_annotations.py \
            --dataset mscmr \
            --input_img_dir /home/gaoqi/dataset/using/mscmr/raw/train/images \
            --input_scribble_dir /home/gaoqi/dataset/using/mscmr/raw/train/labels \
            --output_json_path /home/gaoqi/dataset/using/mscmr/processed/png_coco_sam3_fullframes_weak/train/scribble_annotations.coco.json \
            --slice_policy all
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from pycocotools import mask as mask_utils

# 类别映射、ignore 标签、scribble 文件名候选（base_name 为去掉 .nii.gz 后的 stem）
# ignore_label: int 表示该标签像素不计入 valid_mask；None 表示 valid_mask = (scribble_slice > 0)
DatasetCfg = Tuple[Dict[int, str], Optional[int], Callable[[str, str], List[str]]]

DATASET_CONFIG: Dict[str, DatasetCfg] = {
    'acdc': (
        {1: "right ventricle", 2: "myocardium", 3: "left ventricle"},
        4,
        lambda b, n: [f"{b}_scribble.nii.gz", f"{b}_gt.nii.gz", n],
    ),
    'btcv': (
        {1: "bladder", 2: "uterus", 3: "rectum", 4: "small bowel"},
        5,
        lambda b, n: [n.replace("-Image", "-Mask")],
    ),
    'promise': (
        {1: "prostate"},
        2,
        lambda b, n: [f"{b}_segmentation.nii.gz"],
    ),
    'isbi': (
        {1: "peripheral zone", 2: "central gland"},
        0,
        lambda b, n: [n],
    ),
    'mscmr': (
        {1: "right ventricle", 2: "myocardium", 3: "left ventricle"},
        4,  # ignore：不参与类别，valid_mask 排除
        lambda b, n: [f"{b}_scribble.nii.gz"],
    ),
}


def _assert_mscmr_train_paths_only(img_dir: Path, scribble_dir: Path, output_json_path: Path) -> None:
    """MSCMR 仅处理 train：输入须在 .../train/images 与 .../train/labels；不得使用 val/test 路径。"""
    for p in (img_dir, scribble_dir, output_json_path):
        parts = p.resolve().parts
        if "val" in parts or "test" in parts:
            raise ValueError(
                "MSCMR：本脚本只生成 train 的 scribble JSON，路径中不得包含 val 或 test："
                f" {p}"
            )
    for p in (img_dir, scribble_dir):
        parts = p.resolve().parts
        if "train" not in parts:
            raise ValueError(
                "MSCMR：请将 --input_img_dir / --input_scribble_dir 设为 raw 下 train 的 images 与 labels 目录："
                f" {p}"
            )


def rle_encode(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """将二值 mask 编码为 COCO RLE（空 mask 返回 None）。"""
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
    ignore_label: Optional[int],
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
        'categories': [{'id': k, 'name': v} for k, v in sorted(category_map.items())],
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

    category_ids = np.array(list(category_map.keys()), dtype=np.int64)

    for img_path, scribble_path in tqdm(file_pairs, desc=f'Volumes ({split})'):
        base_name = img_path.stem.replace('.nii', '')
        scribble_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(scribble_path))).astype(np.uint8)
        depth, height, width = scribble_vol.shape

        for z in range(depth):
            scribble_slice = scribble_vol[z]
            has_fg = bool(np.isin(scribble_slice, category_ids).any())

            if slice_policy == 'nonempty' and not has_fg:
                continue

            slice_fname = f"{base_name}_slice{z:03d}.png"
            image_coco['images'].append({
                'id': image_id,
                'file_name': str(Path('images') / slice_fname),
                'height': int(height),
                'width': int(width),
            })

            if ignore_label is None:
                union_mask = (scribble_slice > 0).astype(np.uint8)
            else:
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
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(image_coco, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved: {output_json_path} ({len(image_coco['images'])} images, {len(image_coco['annotations'])} anns)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='将 ACDC / BTCV / PROMISE / ISBI / MSCMR 的 scribble 体数据导出为 COCO 风格 JSON（RLE，不写图像文件）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  ACDC:  python %(prog)s --dataset acdc --input_img_dir ... --input_scribble_dir ... --output_json_path ...
  BTCV:  python %(prog)s --dataset btcv --input_img_dir ... --input_scribble_dir ... --output_json_path ...
  PROMISE: python %(prog)s --dataset promise --input_img_dir ... --input_scribble_dir ... --output_json_path ...
  ISBI:  python %(prog)s --dataset isbi --input_img_dir .../train/image --input_scribble_dir .../train/scribble --output_json_path .../train/scribble_annotations.coco.json
  MSCMR:  python %(prog)s --dataset mscmr --input_img_dir .../raw/train/images --input_scribble_dir .../raw/train/labels --output_json_path .../train/scribble_annotations.coco.json
        """
    )
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'btcv', 'promise', 'isbi', 'mscmr'],
                        help='数据集类型：acdc / btcv / promise / isbi / mscmr')
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
    if args.dataset == "mscmr":
        _assert_mscmr_train_paths_only(img_dir, scribble_dir, output_json_path)
    path_parts = set(img_dir.resolve().parts)
    if 'val' in path_parts:
        split = 'val'
    elif 'test' in path_parts:
        split = 'test'
    else:
        split = 'train'

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

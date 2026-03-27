#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从npz格式文件生成video_annotations.coco.json

该脚本读取已转换为npz格式的ACDC数据集，生成视频级别的COCO格式JSON文件。
每个npz文件包含：
- imgs: 图像数组，形状为 (D, H, W)，dtype=uint8
- gts: 标注数组，形状为 (D, H, W)，dtype=int32
- spacing: 体素间距元组

输出格式：
- videos: 包含视频基本信息（id, name, npz_path, height, width, length）
- annotations: 每个类别在每个视频中的标注，包含frame_indices、segmentations和bboxes
- categories: 类别定义

使用示例：
    python gq_scripts/preprocess/acdc/generate_video_annotations_from_npz.py \
        --npz-dir /home/gaoqi/dataset/using/acdc4/test/data \
        --output-json /home/gaoqi/dataset/using/acdc4/test/video_annotations.coco.json \
        --npz-prefix data/
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm


# ACDC 类别映射
CATEGORY_MAP = {
    1: "right ventricle",
    2: "myocardium",
    3: "left ventricle",
}


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def mask_to_bbox(mask: np.ndarray) -> Optional[List[float]]:
    """
    从mask计算边界框 [x_min, y_min, width, height]
    
    参数:
        mask: 二值mask数组
    
    返回:
        边界框列表，如果mask为空则返回None
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return [float(x_min), float(y_min), float(width), float(height)]


def rle_encode(mask: np.ndarray) -> Optional[Dict]:
    """
    将mask编码为COCO RLE格式
    
    参数:
        mask: 二值mask数组
    
    返回:
        RLE字典，包含'size'和'counts'，如果mask为空则返回None
    """
    if not np.any(mask):
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle


def process_npz_file(npz_path: Path, npz_prefix: str = "") -> Dict[str, Any]:
    """
    处理单个npz文件，提取视频信息和标注
    
    参数:
        npz_path: npz文件路径
        npz_prefix: npz路径在JSON中的前缀（用于相对路径）
    
    返回:
        包含video信息和annotations的字典
    """
    # 加载npz文件
    data = np.load(npz_path)
    imgs = data['imgs']  # (D, H, W)
    gts = data['gts']    # (D, H, W)
    spacing = data.get('spacing', None)  # 可选
    
    depth, height, width = imgs.shape
    
    # 获取文件名（不含扩展名）
    name = npz_path.stem
    
    # 构建npz路径（用于JSON中）
    if npz_prefix:
        npz_relative_path = f"{npz_prefix.rstrip('/')}/{npz_path.name}"
    else:
        npz_relative_path = str(npz_path)
    
    # 视频信息
    video_info = {
        'id': 0,  # 将在主函数中分配
        'name': name,
        'npz_path': npz_relative_path,
        'height': int(height),
        'width': int(width),
        'length': int(depth),
    }
    
    # 为每个类别收集标注
    annotations = []
    
    for category_id, category_name in CATEGORY_MAP.items():
        # 找到该类别出现的所有帧
        frame_indices = []
        segmentations = []
        bboxes = []
        
        for frame_idx in range(depth):
            gt_slice = gts[frame_idx]  # (H, W)
            binary_mask = (gt_slice == category_id).astype(np.uint8)
            
            if np.any(binary_mask):
                # 该帧包含此类别
                frame_indices.append(frame_idx)
                
                # 编码RLE
                rle = rle_encode(binary_mask)
                if rle is not None:
                    segmentations.append(rle)
                else:
                    segmentations.append(None)
                
                # 计算bbox
                bbox = mask_to_bbox(binary_mask)
                bboxes.append(bbox)
        
        # 如果该类别至少在一帧中出现，创建annotation
        if len(frame_indices) > 0:
            annotation = {
                'id': 0,  # 将在主函数中分配
                'video_id': 0,  # 将在主函数中分配
                'category_id': int(category_id),
                'iscrowd': 0,
                'frame_indices': frame_indices,
                'segmentations': segmentations,
                'bboxes': bboxes,
            }
            annotations.append(annotation)
    
    return {
        'video': video_info,
        'annotations': annotations,
    }


def generate_video_annotations(npz_dir: Path, output_json: Path, npz_prefix: str = ""):
    """
    从npz目录生成video_annotations.coco.json
    
    参数:
        npz_dir: npz文件所在目录
        output_json: 输出JSON文件路径
        npz_prefix: npz路径在JSON中的前缀
    """
    # 查找所有npz文件
    npz_files = sorted(list(npz_dir.glob("*.npz")))
    
    if len(npz_files) == 0:
        logging.warning(f"在 {npz_dir} 中未找到npz文件")
        return
    
    logging.info(f"找到 {len(npz_files)} 个npz文件，开始处理...")
    
    # 初始化COCO格式字典
    coco_data = {
        'info': {
            'description': 'ACDC video annotations (full supervision)',
        },
        'videos': [],
        'annotations': [],
        'categories': [
            {'id': k, 'name': v} for k, v in CATEGORY_MAP.items()
        ],
    }
    
    video_id = 0
    annotation_id = 0
    
    # 处理每个npz文件
    for npz_path in tqdm(npz_files, desc="处理npz文件"):
        try:
            result = process_npz_file(npz_path, npz_prefix)
            
            # 设置video_id
            video_info = result['video']
            video_info['id'] = video_id
            
            # 添加视频信息
            coco_data['videos'].append(video_info)
            
            # 处理annotations
            for ann in result['annotations']:
                ann['id'] = annotation_id
                ann['video_id'] = video_id
                coco_data['annotations'].append(ann)
                annotation_id += 1
            
            video_id += 1
            
        except Exception as e:
            logging.error(f"处理文件 {npz_path} 时出错: {e}")
            continue
    
    # 保存JSON文件
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    logging.info(f"成功生成 {output_json}")
    logging.info(f"统计信息: {len(coco_data['videos'])} 个视频, {len(coco_data['annotations'])} 个标注, {len(coco_data['categories'])} 个类别")


def main():
    parser = argparse.ArgumentParser(
        description="从npz格式文件生成video_annotations.coco.json"
    )
    parser.add_argument(
        '--npz-dir',
        type=str,
        required=True,
        help='npz文件所在目录路径'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        required=True,
        help='输出的JSON文件路径'
    )
    parser.add_argument(
        '--npz-prefix',
        type=str,
        default='',
        help='npz路径在JSON中的前缀（例如: data/），用于生成相对路径'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    npz_dir = Path(args.npz_dir)
    output_json = Path(args.output_json)
    
    if not npz_dir.exists():
        logging.error(f"npz目录不存在: {npz_dir}")
        return
    
    generate_video_annotations(npz_dir, output_json, args.npz_prefix)


if __name__ == '__main__':
    main()

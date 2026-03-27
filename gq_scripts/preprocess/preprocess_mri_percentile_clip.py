#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI图像百分位裁剪和归一化预处理脚本

功能：
1. 读取MRI图像（nii.gz格式）
2. 进行百分位裁剪（p_low=0.5, p_high=99.5），仅对非零体素计算
3. 裁剪后进行Min-Max归一化
4. 线性映射到 [0, 255] 并转为 uint8
5. 保存为nii.gz文件，保留原始的空间信息（spacing, origin, direction等）

用法示例：
    python gq_scripts/preprocess/preprocess_mri_percentile_clip.py \
        --input /home/gaoqi/dataset/using/promise12_2/test/img/Case15.nii.gz \
        --output /home/gaoqi/dataset/using/promise12_2/test/img/Case15_processed.nii.gz \
        --p_low 0.5 \
        --p_high 99.5
    python gq_scripts/preprocess/preprocess_mri_percentile_clip.py \
        --input /home/gaoqi/SAT/gq_dataset/PROMISE12/test/results_pro/PROMISE12_original_vis/img_Case15_original.nii.gz \
        --output /home/gaoqi/SAT/gq_dataset/PROMISE12/test/results_pro/PROMISE12_original_vis/img_Case15_original_processed.nii.gz \
        --p_low 0.5 \
        --p_high 99.5
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def percent_clip_and_normalize(volume, p_low=0.5, p_high=99.5):
    """
    MRI预处理：百分位裁剪 + Min-Max归一化到 [0, 255]。
    
    Parameters
    ----------
    volume : np.ndarray
        输入的3D体积数据
    p_low : float
        低百分位阈值，默认0.5
    p_high : float
        高百分位阈值，默认99.5
    
    Returns
    -------
    np.ndarray
        处理后的uint8数组，值域[0, 255]
    """
    vol = np.asarray(volume).astype(np.float32)
    mask = vol > 0  # 仅考虑非零体素
    
    out = np.zeros_like(vol, dtype=np.float32)
    
    if mask.any():
        # 计算非零体素的百分位值
        lo = np.percentile(vol[mask], p_low)
        hi = np.percentile(vol[mask], p_high)
        
        # 裁剪到百分位范围
        vol = np.clip(vol, lo, hi)
        
        # Min-Max归一化
        vmin, vmax = vol[mask].min(), vol[mask].max()
        if vmax > vmin:
            out[mask] = (vol[mask] - vmin) / (vmax - vmin) * 255.0
        else:
            # 如果所有值相同，设为0
            out[mask] = 0.0
    
    return out.clip(0, 255).astype(np.uint8)


def process_mri_file(input_path: Path, output_path: Path, p_low: float = 0.5, p_high: float = 99.5):
    """
    处理单个MRI文件。
    
    Parameters
    ----------
    input_path : Path
        输入nii.gz文件路径
    output_path : Path
        输出nii.gz文件路径
    p_low : float
        低百分位阈值
    p_high : float
        高百分位阈值
    """
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    logging.info(f"读取文件: {input_path}")
    
    # 读取图像
    sitk_image = sitk.ReadImage(str(input_path))
    volume = sitk.GetArrayFromImage(sitk_image)  # (D, H, W)
    
    logging.info(f"图像形状: {volume.shape}")
    logging.info(f"数据类型: {volume.dtype}")
    logging.info(f"值域: [{volume.min():.2f}, {volume.max():.2f}]")
    
    # 处理图像
    logging.info(f"进行百分位裁剪 (p_low={p_low}, p_high={p_high}) 和归一化...")
    processed_volume = percent_clip_and_normalize(volume, p_low=p_low, p_high=p_high)
    
    logging.info(f"处理后值域: [{processed_volume.min()}, {processed_volume.max()}]")
    
    # 创建新的SimpleITK图像，保留原始的空间信息
    processed_sitk = sitk.GetImageFromArray(processed_volume)
    processed_sitk.SetSpacing(sitk_image.GetSpacing())
    processed_sitk.SetOrigin(sitk_image.GetOrigin())
    processed_sitk.SetDirection(sitk_image.GetDirection())
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存处理后的图像
    logging.info(f"保存到: {output_path}")
    sitk.WriteImage(processed_sitk, str(output_path))
    
    logging.info("处理完成！")


def main():
    parser = argparse.ArgumentParser(description='MRI图像百分位裁剪和归一化预处理')
    parser.add_argument('--input', type=str, required=True,
                        help='输入MRI图像路径（nii.gz格式）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出图像路径（nii.gz格式）')
    parser.add_argument('--p_low', type=float, default=0.5,
                        help='低百分位阈值，默认0.5')
    parser.add_argument('--p_high', type=float, default=99.5,
                        help='高百分位阈值，默认99.5')
    
    args = parser.parse_args()
    
    setup_logging()
    
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    
    process_mri_file(input_path, output_path, p_low=args.p_low, p_high=args.p_high)


if __name__ == '__main__':
    main()

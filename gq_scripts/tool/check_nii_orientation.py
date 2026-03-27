#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历文件夹中的所有 3D NIfTI (.nii.gz) 文件，统计这些数据的方向（例如 RAS）。

方向标记说明：
- RAS: Right-Anterior-Superior (右-前-上)
- LPI: Left-Posterior-Inferior (左-后-下)
- 等等

使用示例：
    python check_nii_orientation.py --folder /path/to/nii/folder
    python check_nii_orientation.py --folder /path/to/nii/folder --recursive
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk


def direction_matrix_to_orientation(direction_matrix: Tuple[float, ...]) -> str:
    """
    将方向矩阵转换为方向字符串（如 RAS, LPI 等）。
    
    Args:
        direction_matrix: SimpleITK 的方向矩阵（9个元素的元组，表示3x3矩阵）
    
    Returns:
        方向字符串，如 "RAS", "LPI" 等
    """
    # 将元组转换为3x3矩阵
    matrix = np.array(direction_matrix).reshape(3, 3)
    
    # 找到每列中绝对值最大的元素，确定该轴的方向
    # 列0对应x轴，列1对应y轴，列2对应z轴
    orientation = []
    
    for col_idx in range(3):
        col = matrix[:, col_idx]
        abs_col = np.abs(col)
        max_idx = np.argmax(abs_col)
        sign = np.sign(col[max_idx])
        
        # 映射到方向字符
        if max_idx == 0:  # x轴
            orientation.append('R' if sign > 0 else 'L')
        elif max_idx == 1:  # y轴
            orientation.append('A' if sign > 0 else 'P')
        elif max_idx == 2:  # z轴
            orientation.append('S' if sign > 0 else 'I')
    
    return ''.join(orientation)


def get_nii_orientation(nii_path: Path) -> Tuple[str, np.ndarray]:
    """
    获取 NIfTI 文件的方向信息。
    
    Args:
        nii_path: NIfTI 文件路径
    
    Returns:
        (方向字符串, 方向矩阵)
    """
    try:
        img = sitk.ReadImage(str(nii_path))
        direction_matrix = img.GetDirection()
        orientation = direction_matrix_to_orientation(direction_matrix)
        direction_array = np.array(direction_matrix).reshape(3, 3)
        return orientation, direction_array
    except Exception as e:
        print(f"警告: 无法读取文件 {nii_path}: {e}")
        return None, None


def scan_folder(folder_path: Path, recursive: bool = False) -> List[Path]:
    """
    扫描文件夹中的所有 .nii.gz 文件。
    
    Args:
        folder_path: 文件夹路径
        recursive: 是否递归搜索子文件夹
    
    Returns:
        .nii.gz 文件路径列表
    """
    if recursive:
        nii_files = list(folder_path.rglob('*.nii.gz'))
    else:
        nii_files = list(folder_path.glob('*.nii.gz'))
    
    return sorted(nii_files)


def main():
    parser = argparse.ArgumentParser(
        description="遍历文件夹中的所有 3D NIfTI 文件，统计数据方向"
    )
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='包含 .nii.gz 文件的文件夹路径'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='是否递归搜索子文件夹'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出统计结果到文件（可选）'
    )
    
    args = parser.parse_args()
    
    # 关闭 SimpleITK 警告
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    
    folder_path = Path(args.folder).expanduser().resolve()
    
    if not folder_path.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return
    
    if not folder_path.is_dir():
        print(f"错误: 路径不是文件夹: {folder_path}")
        return
    
    # 扫描所有 .nii.gz 文件
    print(f"正在扫描文件夹: {folder_path}")
    print(f"递归模式: {'开启' if args.recursive else '关闭'}")
    nii_files = scan_folder(folder_path, args.recursive)
    
    if not nii_files:
        print(f"未找到任何 .nii.gz 文件")
        return
    
    print(f"找到 {len(nii_files)} 个 .nii.gz 文件\n")
    
    # 统计方向
    orientation_counter = Counter()
    file_orientations = {}
    failed_files = []
    
    print("正在分析文件方向...")
    for nii_file in nii_files:
        orientation, direction_matrix = get_nii_orientation(nii_file)
        if orientation is not None:
            orientation_counter[orientation] += 1
            file_orientations[nii_file] = (orientation, direction_matrix)
        else:
            failed_files.append(nii_file)
    
    # 打印统计结果
    print("\n" + "="*80)
    print("方向统计结果")
    print("="*80)
    print(f"\n总文件数: {len(nii_files)}")
    print(f"成功分析: {len(file_orientations)}")
    print(f"失败文件: {len(failed_files)}")
    
    if orientation_counter:
        print("\n方向分布:")
        print("-" * 80)
        for orientation, count in orientation_counter.most_common():
            percentage = count / len(file_orientations) * 100
            print(f"  {orientation:6s}: {count:4d} 个文件 ({percentage:5.2f}%)")
    
    # 打印详细文件列表
    print("\n详细文件列表:")
    print("-" * 80)
    for nii_file, (orientation, direction_matrix) in sorted(file_orientations.items()):
        rel_path = nii_file.relative_to(folder_path)
        print(f"  {orientation:6s}  |  {rel_path}")
    
    # 打印失败文件
    if failed_files:
        print("\n失败文件列表:")
        print("-" * 80)
        for failed_file in failed_files:
            rel_path = failed_file.relative_to(folder_path)
            print(f"  {rel_path}")
    
    # 保存结果到文件
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("NIfTI 文件方向统计结果\n")
            f.write("="*80 + "\n\n")
            f.write(f"扫描文件夹: {folder_path}\n")
            f.write(f"递归模式: {'开启' if args.recursive else '关闭'}\n")
            f.write(f"总文件数: {len(nii_files)}\n")
            f.write(f"成功分析: {len(file_orientations)}\n")
            f.write(f"失败文件: {len(failed_files)}\n\n")
            
            if orientation_counter:
                f.write("方向分布:\n")
                f.write("-" * 80 + "\n")
                for orientation, count in orientation_counter.most_common():
                    percentage = count / len(file_orientations) * 100
                    f.write(f"  {orientation:6s}: {count:4d} 个文件 ({percentage:5.2f}%)\n")
            
            f.write("\n详细文件列表:\n")
            f.write("-" * 80 + "\n")
            for nii_file, (orientation, direction_matrix) in sorted(file_orientations.items()):
                rel_path = nii_file.relative_to(folder_path)
                f.write(f"  {orientation:6s}  |  {rel_path}\n")
            
            if failed_files:
                f.write("\n失败文件列表:\n")
                f.write("-" * 80 + "\n")
                for failed_file in failed_files:
                    rel_path = failed_file.relative_to(folder_path)
                    f.write(f"  {rel_path}\n")
        
        print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()

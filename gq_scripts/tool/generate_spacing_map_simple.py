#!/usr/bin/env python3
"""
从指定 ori_images 目录读取所有 .nii.gz 的 spacing 并生成 JSON 映射。
键为去掉后缀的文件名，例如: patient0276_4CH_ED -> [spacing_x, spacing_y, spacing_z]

支持的数据集：
- btcv: 从 0507688-Image.nii.gz 提取 key 为 0507688
- promise12: 从 Case00.nii.gz 提取 key 为 Case00
- 其他: 直接使用去掉 .nii.gz 的文件名作为 key
"""
import os
import json
import argparse
import re

try:
    import nibabel as nib
except Exception:
    nib = None


def extract_key_from_filename(filename, dataset=None):
    """根据数据集类型从文件名提取 key。
    
    Args:
        filename: 文件名（包含 .nii.gz 后缀）
        dataset: 数据集类型，可选 'btcv', 'promise12'，或其他
    
    Returns:
        str: 提取的 key
    """
    base_name = filename[:-7] if filename.endswith('.nii.gz') else filename  # remove .nii.gz
    
    if dataset == 'btcv':
        # BTCV: 0507688-Image.nii.gz -> 0507688
        match = re.match(r'([\d]+)-Image', base_name)
        if match:
            return match.group(1)
        return base_name
    elif dataset == 'promise12':
        # Promise12: Case00.nii.gz -> Case00 (已经是正确格式)
        return base_name
    else:
        # 默认：直接使用去掉后缀的文件名
        return base_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', 
        default="/home/gaoqi/dataset/using/mms2_2/test/img", 
        help='ori_images 目录')
    parser.add_argument('--output', 
        default="/home/gaoqi/sam3/dataset/MMs2/test/spacing_map.json",
        help='输出 JSON 文件路径')
    parser.add_argument('--dataset',
        type=str,
        default=None,
        choices=['btcv', 'promise12'],
        help='数据集类型，用于确定 key 的提取方式')
    args = parser.parse_args()

    if nib is None:
        print('Error: nibabel not installed in this environment.')
        return

    spacing_map = {}
    for fn in sorted(os.listdir(args.input_dir)):
        if not fn.lower().endswith('.nii.gz'):
            continue
        path = os.path.join(args.input_dir, fn)
        try:
            img = nib.load(path)
            zooms = img.header.get_zooms()
            if len(zooms) >= 3:
                spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
            elif len(zooms) == 2:
                spacing = (float(zooms[0]), float(zooms[1]), 1.0)
            else:
                spacing = (1.0, 1.0, 1.0)
        except Exception as e:
            print(f'Warning: failed to read {path}: {e}')
            continue

        key = extract_key_from_filename(fn, args.dataset)
        spacing_map[key] = [spacing[0], spacing[1], spacing[2]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(spacing_map, f, indent=2)
    print(f'Wrote {len(spacing_map)} entries to {args.output}')


if __name__ == '__main__':
    main()

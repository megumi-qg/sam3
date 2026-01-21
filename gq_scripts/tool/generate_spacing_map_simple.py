#!/usr/bin/env python3
"""
从指定 ori_images 目录读取所有 .nii.gz 的 spacing 并生成 JSON 映射。
键为去掉后缀的文件名，例如: patient0276_4CH_ED -> [spacing_x, spacing_y, spacing_z]
"""
import os
import json
import argparse

try:
    import nibabel as nib
except Exception:
    nib = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', 
        default="/home/gaoqi/dataset/using/mms2_2/test/img", 
        help='ori_images 目录')
    parser.add_argument('--output', 
        default="/home/gaoqi/sam3/dataset/MMs2/test/spacing_map.json",
        help='输出 JSON 文件路径')
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

        key = fn[:-7]  # remove .nii.gz
        spacing_map[key] = [spacing[0], spacing[1], spacing[2]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(spacing_map, f, indent=2)
    print(f'Wrote {len(spacing_map)} entries to {args.output}')


if __name__ == '__main__':
    main()

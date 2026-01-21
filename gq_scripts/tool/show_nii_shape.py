#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单查看 NIfTI (.nii / .nii.gz) 文件的体素尺寸、体素间距等信息。

使用示例：
    python show_nii_shape.py --path /path/to/amos_0001.nii.gz
"""

import argparse
from pathlib import Path

import SimpleITK as sitk


def describe_nii(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在")

    img = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(img)  # numpy array (z, y, x)

    shape = array.shape  # (depth, height, width)
    spacing = img.GetSpacing()[::-1]  # sitk spacing order is (x, y, z)
    origin = img.GetOrigin()
    direction = img.GetDirection()

    print(f"文件: {path}")
    print(f"数据类型: {array.dtype}")
    print(f"Shape (z, y, x): {shape}")
    print(f"体素间距 spacing (z, y, x): {spacing}")
    print(f"原点 origin: {origin}")
    print(f"方向矩阵 direction (flattened): {direction}")


def parse_args():
    ap = argparse.ArgumentParser(description="查看 NIfTI 文件的 shape")
    ap.add_argument(
        "--path",
        type=str,
        default="/home/gaoqi/weaksam/data/ACDC/pre_test/gts_nii/patient101_frame01.nii.gz",
        required=False,
        help="NIfTI 文件路径 (.nii 或 .nii.gz)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    args = parse_args()
    describe_nii(Path(args.path).expanduser().resolve())


#!/usr/bin/env python3
"""
将 batch_inference.py 生成的 predictions.pkl 转换为每个 volume 一个的 .nii.gz 预测 mask 文件。

标签规则：0=背景，1/2/3/...= 按 category_id 升序的类别（与 batch_inference 保存的 nii 一致）。
体数据形状：(n_slices, H, W)。

用法:
    python gq_scripts/evaluate/pkl_to_nii.py --pkl /path/to/predictions.pkl [--output_dir /path/to/nii]
    # 不指定 output_dir 时，默认保存到 pkl 所在目录下的 nii 子目录
"""

import os
import argparse
import pickle

import numpy as np

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def _safe_name(name):
    """将 patient_name 转为安全文件名（替换空格、斜杠）。"""
    return name.replace(" ", "_").replace("/", "_").strip() or "unknown"


def save_volume_as_nii(categories_result, patient_name, nii_dir, affine=None):
    """将单个 volume 的按类别预测合并为 3D 标签体并保存为 .nii.gz。

    保存时转为 (W, H, n_slices) 以与常见 NIfTI/ITK-SNAP 主图维度一致，便于叠加显示。
    """
    if not categories_result:
        return
    sorted_cats = sorted(categories_result, key=lambda x: x["category_id"])
    n_slices = len(sorted_cats[0]["masks"])
    h, w = sorted_cats[0]["masks"][0].shape
    vol = np.zeros((n_slices, h, w), dtype=np.uint8)
    for label_val, cat in enumerate(sorted_cats, start=1):
        for i, mask_2d in enumerate(cat["masks"]):
            if i < vol.shape[0] and mask_2d.shape[0] == h and mask_2d.shape[1] == w:
                vol[i][mask_2d] = label_val
    # (n_slices, H, W) -> (W, H, n_slices)，与主图 (x, y, z) 一致
    vol = np.transpose(vol, (2, 1, 0))
    if affine is None:
        affine = np.eye(4)
    nii = nib.Nifti1Image(vol, affine, nib.Nifti1Header())
    safe = _safe_name(patient_name)
    out_path = os.path.join(nii_dir, f"{safe}.nii.gz")
    os.makedirs(nii_dir, exist_ok=True)
    nib.save(nii, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="将 predictions.pkl 转为每个 volume 一个的 .nii.gz 文件",
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default="/home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_promise12_0.7/predictions.pkl",
        help="predictions.pkl 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/gaoqi/sam3/gq_experiment/final/weak_lora_acdc_btcv_promise12/inference_promise12_0.7/nii",
        help="nii.gz 输出目录；默认使用 pkl 所在目录下的 nii 子目录",
    )
    args = parser.parse_args()

    if not HAS_NIBABEL:
        raise RuntimeError("需要安装 nibabel: pip install nibabel")

    pkl_path = os.path.abspath(args.pkl)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"未找到 pkl 文件: {pkl_path}")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(pkl_path), "nii")

    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    volumes = result.get("volumes", [])
    if not volumes:
        print("pkl 中无 volumes 数据")
        return 0

    for vol in volumes:
        patient_name = vol.get("patient_name", "unknown")
        categories = vol.get("categories", [])
        if not categories:
            continue
        save_volume_as_nii(categories, patient_name, args.output_dir)

    print(f"已保存 {len([v for v in volumes if v.get('categories')])} 个 nii.gz 到: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 3D 医学影像（NIfTI）按切片导出为 PNG，并生成图像级 COCO JSON（dense / full 监督）。

默认输出 `image_annotations.coco.json`。**MSCMR 一键模式**（`--mscmr_all`）依次处理 **train / val / test**：
**train** 使用 scribble 体数据，导出 `images/`、`masks/`（掩码含 scribble 标签）及 `image_annotations.coco.json`；
**val / test** 使用 manual，导出同目录结构及 `full_annotations.coco.json`。
弱监督用的 **仅 RLE、无 PNG** 的 scribble JSON 仍由 ``preprocess_scribble_annotations.py`` 单独生成。

支持数据集
----------
- **acdc**：心脏 MRI；GT：`patientXXX_frame01.nii.gz` → `patientXXX_frame01_gt.nii.gz`
- **promise12**：前列腺 MRI；GT：`CaseXX.nii.gz` → `CaseXX_segmentation.nii.gz`
- **isbi**：前列腺 MRI；图像与 GT 同名：`patientXXX.nii.gz` ↔ `patientXXX.nii.gz`
- **btcv_cervix**：腹部/宫颈 CT；GT：`XXXXXX-Image.nii.gz` → `XXXXXX-Mask.nii.gz`
- **mscmr_train**：MSCMR **train**，scribble GT：`subjectXX_DE.nii.gz` → `subjectXX_DE_scribble.nii.gz`。
- **mscmr_eval**：MSCMR **val / test**，manual GT：`subjectXX_DE.nii.gz` → `subjectXX_DE_manual.nii.gz`。
  类别 **1–3** 为三类结构；**4** 仅在 scribble 中为 ignore（不参与 COCO 类别与实例）。

MSCMR raw 布局::
    <raw_root>/
      train/images  train/labels  (*_scribble.nii.gz)
      val/images    val/labels    (*_manual.nii.gz)
      test/images   test/labels   (*_manual.nii.gz)

影像预处理
----------
**MRI**（`--modality mri`）：
1. 仅在体素值 > 0 的区域计算百分位（默认 p_low=0.5, p_high=99.5）
2. 裁剪后对该区域内做 Min-Max，线性映射到 [0, 255]，uint8

**CT**（`--modality ct`）：窗宽窗位映射到 [0, 255]（见 `--window_level` / `--window_width`）

**标注**：自 nii.gz 读取，不做强度变换，存为 uint8 掩码 PNG。

**切片策略**（`--slice_policy`）：`nonempty` 仅保留含前景类别的切片；`all` 保留全部轴向切片。

输出
----
**默认**::

  <output_dir>/
    images/
    masks/
    image_annotations.coco.json

**MSCMR --mscmr_all**::

  <mscmr_output_root>/train/
    images/  masks/  image_annotations.coco.json
  <mscmr_output_root>/val/  与  test/
    images/  masks/  full_annotations.coco.json

用法示例
--------
# ACDC（MRI）
python gq_scripts/preprocess/preprocess_full_annotations.py \\
    --dataset acdc --modality mri \\
    --img_dir .../img --gt_dir .../gt --output_dir .../out \\
    --slice_policy all

# MSCMR：train（scribble PNG）+ val/test（manual PNG）
python gq_scripts/preprocess/preprocess_full_annotations.py \
    --mscmr_all \
    --mscmr_raw_root /home/gaoqi/dataset/using/mscmr/raw \
    --mscmr_output_root /home/gaoqi/dataset/using/mscmr/processed/png_coco_sam3_fullframes_weak \
    --modality mri --slice_policy all

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


# 数据集：gt 文件名由 img 文件名经 gt_replace 替换得到；categories 为类别 id -> 名称
DATASET_CONFIG = {
    "acdc": {
        "gt_replace": (".nii.gz", "_gt.nii.gz"),
        "categories": {1: "right ventricle", 2: "myocardium", 3: "left ventricle"},
    },
    "promise12": {
        "gt_replace": (".nii.gz", "_segmentation.nii.gz"),
        "categories": {1: "prostate"},
    },
    "isbi": {
        "gt_replace": (".nii.gz", ".nii.gz"),
        "categories": {1: "peripheral zone", 2: "central gland"},
    },
    "btcv_cervix": {
        "gt_replace": ("-Image.nii.gz", "-Mask.nii.gz"),
        "categories": {
            1: "Structure_1",
            2: "Structure_2",
            3: "Structure_3",
            4: "Structure_4",
        },
    },
    # MSCMR train：scribble 体数据切片 → masks 为 uint8 标签图（含 4=ignore 像素，COCO 仅 1–3）
    "mscmr_train": {
        "gt_replace": (".nii.gz", "_scribble.nii.gz"),
        "categories": {
            1: "right ventricle",
            2: "myocardium",
            3: "left ventricle",
        },
    },
    "mscmr_eval": {
        "gt_replace": (".nii.gz", "_manual.nii.gz"),
        "categories": {
            1: "right ventricle",
            2: "myocardium",
            3: "left ventricle",
        },
    },
}

# MSCMR --mscmr_all：split -> (dataset 配置键, 输出 COCO 文件名)
MSCMR_ALL_SPLITS: Tuple[Tuple[str, str, str], ...] = (
    ("train", "mscmr_train", "image_annotations.coco.json"),
    ("val", "mscmr_eval", "full_annotations.coco.json"),
    ("test", "mscmr_eval", "full_annotations.coco.json"),
)

DEFAULT_MSCMR_RAW = Path("/home/gaoqi/dataset/using/mscmr/raw")
# 与 ACDC 的 processed/png_coco_sam3_fullframes_weak 目录约定一致
DEFAULT_MSCMR_OUT = Path("/home/gaoqi/dataset/using/mscmr/processed/png_coco_sam3_fullframes_weak")


def _assert_mscmr_eval_not_train_paths(*paths: Path) -> None:
    """mscmr_eval（manual）仅用于 val/test；路径中不得出现 train 目录段。"""
    for p in paths:
        if "train" in p.resolve().parts:
            raise ValueError(
                "MSCMR manual（mscmr_eval）仅用于 val/test。路径中不得包含 train 目录："
                f" {p}"
            )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def percent_clip_to_u8(volume: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """MRI：百分位裁剪 + Min-Max 到 [0, 255] uint8（背景 0 不参与统计）。"""
    vol = np.asarray(volume).astype(np.float32)
    mask = vol > 0
    out = np.zeros_like(vol, dtype=np.float32)
    if mask.any():
        lo = float(np.percentile(vol[mask], p_low))
        hi = float(np.percentile(vol[mask], p_high))
        clipped = np.clip(vol, lo, hi)
        vmin = float(clipped[mask].min())
        vmax = float(clipped[mask].max())
        if vmax > vmin:
            out[mask] = (clipped[mask] - vmin) / (vmax - vmin) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def window_level_to_u8(
    volume: np.ndarray,
    window_level: float = 40,
    window_width: float = 400,
) -> np.ndarray:
    """CT：窗宽窗位映射到 [0, 255] uint8。"""
    vol = np.asarray(volume).astype(np.float32)
    low = window_level - window_width / 2
    high = window_level + window_width / 2
    out = (vol - low) / (high - low) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def mask_to_bbox(mask: np.ndarray) -> List[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def rle_encode(mask: np.ndarray):
    if not np.any(mask):
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def normalize_gt_labels(dataset: str, gt_vol: np.ndarray) -> np.ndarray:
    """Normalize dataset-specific raw labels before PNG/COCO export."""
    gt_vol = np.asarray(gt_vol).astype(np.uint8, copy=True)

    if dataset == "isbi":
        # ISBI dense labels may encode background as 3; normalize to the usual 0 background.
        gt_vol[gt_vol == 3] = 0

    return gt_vol


def append_anns_for_gt_slice(
    gt_slice: np.ndarray,
    category_map: Dict[int, str],
    image_id: int,
    ann_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """由单张语义掩码切片生成 COCO annotations（跳过背景与不存在的类别）。"""
    anns: List[Dict[str, Any]] = []
    for label in np.unique(gt_slice):
        lab = int(label)
        if lab == 0 or lab not in category_map:
            continue
        binary_mask = (gt_slice == lab).astype(np.uint8)
        bbox = mask_to_bbox(binary_mask)
        if bbox is None:
            continue
        area = float(np.sum(binary_mask))
        rle = rle_encode(binary_mask)
        if rle is None:
            continue
        anns.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": lab,
                "bbox": bbox,
                "area": area,
                "segmentation": rle,
                "iscrowd": 0,
            }
        )
        ann_id += 1
    return anns, ann_id


def process(
    img_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    slice_policy: str = "nonempty",
    dataset: str = "acdc",
    modality: str = "mri",
    window_level: float = 40,
    window_width: float = 400,
    coco_json_name: str = "image_annotations.coco.json",
) -> None:
    """读取 img/gt 目录下成对 nii.gz，切片写入 output_dir。"""
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"不支持的 dataset: {dataset}，可选: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset]
    category_map = config["categories"]
    gt_from, gt_to = config["gt_replace"]

    if not img_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在或不是目录: {img_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在或不是目录: {gt_dir}")

    if dataset == "mscmr_eval":
        _assert_mscmr_eval_not_train_paths(img_dir, gt_dir, output_dir)

    output_dir = output_dir.resolve()
    out_images_dir = output_dir / "images"
    out_masks_dir = output_dir / "masks"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    image_coco: Dict[str, Any] = {
        "info": {"description": f"{dataset.upper()} (sliced NIfTI → COCO)"},
        "images": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in sorted(category_map.items())],
    }

    image_id = 0
    ann_id = 0

    img_files = sorted(img_dir.glob("*.nii.gz"))
    logging.info("Processing %d volumes from %s (dataset=%s) ...", len(img_files), img_dir, dataset)

    file_pairs: List[Tuple[Path, Path]] = []
    for img_path in img_files:
        gt_filename = img_path.name.replace(gt_from, gt_to)
        gt_path = gt_dir / gt_filename
        if not gt_path.is_file():
            logging.warning(
                "Ground truth not found for %s, expected %s — skipped.",
                img_path.name,
                gt_filename,
            )
            continue
        file_pairs.append((img_path, gt_path))

    if not file_pairs:
        logging.warning("No valid image–label pairs under %s / %s.", img_dir, gt_dir)

    category_ids = list(category_map.keys())

    for img_path, gt_path in tqdm(file_pairs, desc="Volumes"):
        base_name = img_path.name[: -len(".nii.gz")]

        img_sitk = sitk.ReadImage(str(img_path))
        gt_sitk = sitk.ReadImage(str(gt_path))
        img_vol = sitk.GetArrayFromImage(img_sitk)
        gt_vol = sitk.GetArrayFromImage(gt_sitk)

        if img_vol.shape != gt_vol.shape:
            logging.warning(
                "Shape mismatch %s vs %s: img %s gt %s — skipped.",
                img_path.name,
                gt_path.name,
                img_vol.shape,
                gt_vol.shape,
            )
            continue

        gt_vol = normalize_gt_labels(dataset, gt_vol)

        if modality == "ct":
            img_vol_u8 = window_level_to_u8(img_vol, window_level, window_width)
        else:
            img_vol_u8 = percent_clip_to_u8(img_vol)

        depth, height, width = img_vol.shape

        for z in range(depth):
            img_slice = img_vol_u8[z]
            gt_slice = gt_vol[z]

            has_fg = np.any(np.isin(gt_slice, category_ids))
            if slice_policy == "nonempty" and not has_fg:
                continue

            slice_fname = f"{base_name}_slice{z:03d}.png"
            save_img = out_images_dir / slice_fname
            save_mask = out_masks_dir / slice_fname

            Image.fromarray(img_slice).save(save_img)
            Image.fromarray(gt_slice, mode="L").save(save_mask)

            image_coco["images"].append(
                {
                    "id": image_id,
                    "file_name": str(Path("images") / slice_fname),
                    "mask_file_name": str(Path("masks") / slice_fname),
                    "height": int(height),
                    "width": int(width),
                }
            )

            new_anns, ann_id = append_anns_for_gt_slice(gt_slice, category_map, image_id, ann_id)
            image_coco["annotations"].extend(new_anns)

            image_id += 1

    image_json_path = output_dir / coco_json_name
    with open(image_json_path, "w", encoding="utf-8") as f:
        json.dump(image_coco, f, indent=2, ensure_ascii=False)

    logging.info(
        "Saved COCO: %s (%d images, %d anns)",
        image_json_path,
        len(image_coco["images"]),
        len(image_coco["annotations"]),
    )


def run_mscmr_all(
    raw_root: Path,
    output_root: Path,
    slice_policy: str,
    modality: str,
    window_level: float,
    window_width: float,
) -> None:
    """MSCMR：train（scribble）→ image_annotations.coco.json；val/test（manual）→ full_annotations.coco.json。"""
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"MSCMR raw 根目录不存在: {raw_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    for split, ds_key, coco_name in MSCMR_ALL_SPLITS:
        img_dir = raw_root / split / "images"
        gt_dir = raw_root / split / "labels"
        out_dir = output_root / split
        if not img_dir.is_dir() or not gt_dir.is_dir():
            logging.warning("跳过 split %s：缺少 images/ 或 labels/（%s / %s）", split, img_dir, gt_dir)
            continue
        logging.info("=== MSCMR [%s] dataset=%s -> %s ===", split, ds_key, out_dir)
        process(
            img_dir,
            gt_dir,
            out_dir,
            slice_policy=slice_policy,
            dataset=ds_key,
            modality=modality,
            window_level=window_level,
            window_width=window_width,
            coco_json_name=coco_name,
        )


def main() -> None:
    import argparse

    ds_choices = list(DATASET_CONFIG.keys())
    parser = argparse.ArgumentParser(
        description="3D NIfTI 切片为 PNG + COCO JSON（ACDC / PROMISE12 / ISBI / BTCV / MSCMR）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="acdc",
        choices=ds_choices,
        help="数据集类型；MSCMR 一键模式请用 --mscmr_all",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="mri",
        choices=["mri", "ct"],
        help="mri：百分位归一；ct：窗宽窗位",
    )
    parser.add_argument("--window_level", type=float, default=40, help="CT 窗位 WL")
    parser.add_argument("--window_width", type=float, default=400, help="CT 窗宽 WW")
    parser.add_argument("--img_dir", type=str, default=None, help="含体数据 .nii.gz 的目录")
    parser.add_argument("--gt_dir", type=str, default=None, help="含标注 .nii.gz 的目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出根目录（生成 images/、masks/、json）")
    parser.add_argument(
        "--coco_json_name",
        type=str,
        default="image_annotations.coco.json",
        help="输出 COCO 文件名；MSCMR manual 建议 full_annotations.coco.json",
    )
    parser.add_argument(
        "--slice_policy",
        type=str,
        default="nonempty",
        choices=["all", "nonempty"],
        help="nonempty：仅含前景的切片；all：全部轴向切片",
    )
    parser.add_argument(
        "--mscmr_all",
        action="store_true",
        help="MSCMR：train→scribble PNG+image_annotations；val/test→manual PNG+full_annotations",
    )
    parser.add_argument(
        "--mscmr_raw_root",
        type=str,
        default=str(DEFAULT_MSCMR_RAW),
        help="MSCMR 原始数据根目录（内含 train/val/test）",
    )
    parser.add_argument(
        "--mscmr_output_root",
        type=str,
        default=str(DEFAULT_MSCMR_OUT),
        help="MSCMR 输出根目录（其下 train/val/test 各有 images、masks 及对应 COCO JSON）",
    )

    args = parser.parse_args()
    setup_logging()

    if args.mscmr_all:
        run_mscmr_all(
            Path(args.mscmr_raw_root),
            Path(args.mscmr_output_root),
            args.slice_policy,
            args.modality,
            args.window_level,
            args.window_width,
        )
        return

    if args.img_dir is None or args.gt_dir is None or args.output_dir is None:
        parser.error("非 --mscmr_all 时必须提供 --img_dir、--gt_dir、--output_dir")

    process(
        Path(args.img_dir).expanduser().resolve(),
        Path(args.gt_dir).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        args.slice_policy,
        args.dataset,
        modality=args.modality,
        window_level=args.window_level,
        window_width=args.window_width,
        coco_json_name=args.coco_json_name,
    )


if __name__ == "__main__":
    main()

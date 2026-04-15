#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
将 3D 医学体数据导出为适配当前 SAM3 `VideoGroundingDataset` 的
`NPZ volume + video_annotations.coco.json` 格式。

设计目标
--------
这个脚本不是为了替代现有的 2D PNG + COCO 预处理，而是为了给
“一个 3D volume 作为一个 sample 输入模型”的实验提供稳定的数据入口。

输出结构
--------
默认输出目录结构如下::

  <output_dir>/
    volumes/
      patientXXX_frameYY.npz
    video_annotations.coco.json
    spacing_map.json

其中：

- `volumes/*.npz`
  - `volume`: 归一化后的 uint8 体数据，shape = [D, H, W]
  - `mask`: 原始整型标签体，shape = [D, H, W]
  - `slice_indices`: 当前 volume 中每一层对应原始 NIfTI 的 z 索引
  - `spacing_xyz`: 原始 SimpleITK spacing，顺序与 `GetSpacing()` 一致，即 `(x, y, z)`

- `video_annotations.coco.json`
  - `videos`: 每个 volume 一条记录
  - `annotations`: 每个 volume / category 一条稀疏 track 记录，
    只保存该类别真正出现过的切片

注意事项
--------
1. 该脚本默认使用 **volume-level normalization**，而不是逐切片独立归一化。
   这样更适合后续做切片上下文学习。
2. 当前仓库里的 `COCO_VIDEO_FROM_JSON` + `VideoGroundingDataset` 仍然主要根据
   “有 query 的 frame” 采样训练帧，因此仅准备数据并不等价于已经实现了真正的
   跨切片建模；它只是为后续研究提供了合适的数据格式。
3. 默认 `--slice_policy all`，因为做 3D / 邻近切片上下文学习时，通常不应在
   预处理阶段丢掉无前景切片。

用法示例
--------
ACDC::

    python gq_scripts/preprocess/preprocess_video_annotations.py \
        --dataset acdc --modality mri \
        --img_dir /home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_100/train/img \
        --gt_dir /home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_100/train/gt \
        --output_dir /home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/train \
        --slice_policy all
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from pycocotools import mask as mask_utils
from tqdm import tqdm

FRAME_ID_STRIDE = 1_000_000


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
    "mscmr_eval": {
        "gt_replace": (".nii.gz", "_manual.nii.gz"),
        "categories": {
            1: "right ventricle",
            2: "myocardium",
            3: "left ventricle",
        },
    },
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def percent_clip_to_u8(
    volume: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> np.ndarray:
    """MRI：按整卷非零区域做百分位裁剪，再线性映射到 [0, 255]。"""
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
    """CT：按窗宽窗位映射到 [0, 255]。"""
    vol = np.asarray(volume).astype(np.float32)
    low = window_level - window_width / 2
    high = window_level + window_width / 2
    out = (vol - low) / (high - low) * 255.0
    return out.clip(0, 255).astype(np.uint8)


def normalize_gt_labels(dataset: str, gt_vol: np.ndarray) -> np.ndarray:
    """对数据集特定标签做标准化，和现有 2D 预处理保持一致。"""
    gt_vol = np.asarray(gt_vol).astype(np.uint8, copy=True)

    if dataset == "isbi":
        gt_vol[gt_vol == 3] = 0

    return gt_vol


def rle_encode(mask: np.ndarray) -> Dict[str, Any] | None:
    if not np.any(mask):
        return None
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def mask_to_bbox(mask: np.ndarray) -> List[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def make_frame_image_id(video_id: int, frame_idx: int) -> int:
    """
    Build a globally unique frame id from (video_id, frame_idx).

    We keep this deterministic formula in sync with the video JSON loader so that
    validation predictions can be matched against frame-level COCO GT without
    needing an extra lookup table.
    """
    return int(video_id) * FRAME_ID_STRIDE + int(frame_idx)


def build_sparse_track(
    gt_vol: np.ndarray,
    category_id: int,
    kept_slice_indices: List[int],
) -> Dict[str, List[Any]]:
    """
    为单个类别构建一个稀疏的 video annotation track。

    这里按 volume / category 聚合，而不是做真正的 instance tracking。
    对当前医学语义分割场景，这通常更符合“类别 query + 分割 mask”的训练方式。
    """
    frame_indices: List[int] = []
    bboxes: List[List[float]] = []
    segmentations: List[Dict[str, Any]] = []
    areas: List[float] = []

    for local_idx, orig_idx in enumerate(kept_slice_indices):
        gt_slice = gt_vol[orig_idx]
        binary_mask = (gt_slice == category_id).astype(np.uint8)
        if not np.any(binary_mask):
            continue

        bbox = mask_to_bbox(binary_mask)
        rle = rle_encode(binary_mask)
        if bbox is None or rle is None:
            continue

        frame_indices.append(int(local_idx))
        bboxes.append(bbox)
        segmentations.append(rle)
        areas.append(float(np.sum(binary_mask)))

    return {
        "frame_indices": frame_indices,
        "bboxes": bboxes,
        "segmentations": segmentations,
        "areas": areas,
    }


def process(
    img_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    dataset: str,
    modality: str,
    window_level: float,
    window_width: float,
    slice_policy: str,
    coco_json_name: str,
    max_volumes: int | None,
) -> None:
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset}. Choices: {list(DATASET_CONFIG.keys())}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    output_dir = output_dir.expanduser().resolve()
    volumes_dir = output_dir / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)

    config = DATASET_CONFIG[dataset]
    category_map = config["categories"]
    gt_from, gt_to = config["gt_replace"]

    img_files = sorted(img_dir.glob("*.nii.gz"))
    if max_volumes is not None:
        img_files = img_files[:max_volumes]

    file_pairs: List[Tuple[Path, Path]] = []
    for img_path in img_files:
        gt_filename = img_path.name.replace(gt_from, gt_to)
        gt_path = gt_dir / gt_filename
        if not gt_path.is_file():
            logging.warning(
                "Ground truth not found for %s, expected %s - skipped.",
                img_path.name,
                gt_filename,
            )
            continue
        file_pairs.append((img_path, gt_path))

    video_coco: Dict[str, Any] = {
        "info": {
            "description": (
                f"{dataset.upper()} video-style NPZ dataset for SAM3 3D/context experiments"
            )
        },
        "videos": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in sorted(category_map.items())],
    }
    frame_coco: Dict[str, Any] = {
        "info": {
            "description": (
                f"{dataset.upper()} frame-level COCO GT derived from video-style NPZ dataset"
            )
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in sorted(category_map.items())],
    }
    spacing_map: Dict[str, List[float]] = {}

    video_id = 0
    ann_id = 0
    frame_ann_id = 0
    category_ids = list(category_map.keys())

    logging.info(
        "Processing %d volumes from %s (dataset=%s, slice_policy=%s) ...",
        len(file_pairs),
        img_dir,
        dataset,
        slice_policy,
    )

    for img_path, gt_path in tqdm(file_pairs, desc="Volumes"):
        base_name = img_path.name[: -len(".nii.gz")]

        img_sitk = sitk.ReadImage(str(img_path))
        gt_sitk = sitk.ReadImage(str(gt_path))
        img_vol = sitk.GetArrayFromImage(img_sitk)
        gt_vol = sitk.GetArrayFromImage(gt_sitk)

        if img_vol.shape != gt_vol.shape:
            logging.warning(
                "Shape mismatch for %s vs %s: img %s gt %s - skipped.",
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
        if slice_policy == "nonempty":
            keep_mask = np.any(np.isin(gt_vol, category_ids), axis=(1, 2))
            kept_slice_indices = np.where(keep_mask)[0].astype(np.int32).tolist()
        else:
            kept_slice_indices = list(range(depth))

        if not kept_slice_indices:
            logging.warning("No kept slices for %s - skipped.", base_name)
            continue

        volume_rel_path = Path("volumes") / f"{base_name}.npz"
        volume_save_path = output_dir / volume_rel_path
        np.savez_compressed(
            volume_save_path,
            volume=img_vol_u8[kept_slice_indices].astype(np.uint8),
            mask=gt_vol[kept_slice_indices].astype(np.uint8),
            slice_indices=np.asarray(kept_slice_indices, dtype=np.int32),
            spacing_xyz=np.asarray(img_sitk.GetSpacing(), dtype=np.float32),
        )

        spacing_xyz = [float(v) for v in img_sitk.GetSpacing()]
        spacing_map[base_name] = spacing_xyz

        video_coco["videos"].append(
            {
                "id": video_id,
                "video_name": base_name,
                "npz_path": str(volume_rel_path),
                "length": len(kept_slice_indices),
                "height": int(height),
                "width": int(width),
                "slice_indices": kept_slice_indices,
                "spacing_xyz": spacing_xyz,
                "original_length": int(depth),
            }
        )

        for local_idx, orig_idx in enumerate(kept_slice_indices):
            frame_coco["images"].append(
                {
                    "id": make_frame_image_id(video_id, local_idx),
                    "file_name": str(volume_rel_path),
                    "height": int(height),
                    "width": int(width),
                    "video_id": int(video_id),
                    "frame_idx": int(local_idx),
                    "original_slice_idx": int(orig_idx),
                    "is_npz": True,
                }
            )

        for category_id in sorted(category_map):
            track = build_sparse_track(gt_vol, category_id, kept_slice_indices)
            if not track["frame_indices"]:
                continue
            video_coco["annotations"].append(
                {
                    "id": ann_id,
                    "video_id": video_id,
                    "category_id": category_id,
                    "iscrowd": 0,
                    **track,
                }
            )
            ann_id += 1

            for frame_pos, local_idx in enumerate(track["frame_indices"]):
                frame_coco["annotations"].append(
                    {
                        "id": frame_ann_id,
                        "image_id": make_frame_image_id(video_id, local_idx),
                        "category_id": category_id,
                        "bbox": track["bboxes"][frame_pos],
                        "segmentation": track["segmentations"][frame_pos],
                        "area": track["areas"][frame_pos],
                        "iscrowd": 0,
                        "video_id": int(video_id),
                        "frame_idx": int(local_idx),
                    }
                )
                frame_ann_id += 1

        video_id += 1

    coco_path = output_dir / coco_json_name
    if "video_annotations" in coco_json_name:
        frame_coco_name = coco_json_name.replace("video_annotations", "frame_annotations")
    else:
        frame_coco_name = f"frame_{coco_json_name}"
    frame_coco_path = output_dir / frame_coco_name
    spacing_map_path = output_dir / "spacing_map.json"
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(video_coco, f, indent=2, ensure_ascii=False)
    with open(frame_coco_path, "w", encoding="utf-8") as f:
        json.dump(frame_coco, f, indent=2, ensure_ascii=False)
    with open(spacing_map_path, "w", encoding="utf-8") as f:
        json.dump(spacing_map, f, indent=2, ensure_ascii=False)

    logging.info(
        "Saved %s (%d videos, %d annotations)",
        coco_path,
        len(video_coco["videos"]),
        len(video_coco["annotations"]),
    )
    logging.info(
        "Saved %s (%d images, %d annotations)",
        frame_coco_path,
        len(frame_coco["images"]),
        len(frame_coco["annotations"]),
    )
    logging.info("Saved %s", spacing_map_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export 3D NIfTI volumes as SAM3 video-style NPZ + COCO JSON.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="acdc",
        choices=list(DATASET_CONFIG.keys()),
        help="Dataset type.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="mri",
        choices=["mri", "ct"],
        help="MRI uses percentile clipping; CT uses window/level.",
    )
    parser.add_argument("--window_level", type=float, default=40, help="CT window level")
    parser.add_argument("--window_width", type=float, default=400, help="CT window width")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with input image NIfTI files")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with GT NIfTI files")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory containing volumes/ and video_annotations.coco.json",
    )
    parser.add_argument(
        "--slice_policy",
        type=str,
        default="all",
        choices=["all", "nonempty"],
        help="all keeps full depth; nonempty trims the saved volume to slices with foreground labels.",
    )
    parser.add_argument(
        "--coco_json_name",
        type=str,
        default="video_annotations.coco.json",
        help="Output COCO JSON file name.",
    )
    parser.add_argument(
        "--max_volumes",
        type=int,
        default=None,
        help="Optional debug cap on the number of volumes to process.",
    )

    args = parser.parse_args()
    setup_logging()
    process(
        img_dir=Path(args.img_dir).expanduser().resolve(),
        gt_dir=Path(args.gt_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        dataset=args.dataset,
        modality=args.modality,
        window_level=args.window_level,
        window_width=args.window_width,
        slice_policy=args.slice_policy,
        coco_json_name=args.coco_json_name,
        max_volumes=args.max_volumes,
    )


if __name__ == "__main__":
    main()

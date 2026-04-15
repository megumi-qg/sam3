#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
将 3D 医学体数据的 scribble 标注导出为适配当前 SAM3 video-like 管线的
COCO JSON（仅生成弱监督标注 JSON，复用已有 `volumes/*.npz`）。

设计原则
--------
这个脚本严格对齐现有 2D 弱监督语义：

- `segmentation` 只保存目标类别的 scribble；
- `valid_mask` 保存当前切片的有效监督区域；
- 下游 dataset 会据此构造 tri-state mask：
  - `1`: 正样本 scribble
  - `0`: valid 区域内的背景
  - `255`: ignore

输出
----
输出文件是一个 `video_annotations.coco.json` 风格的弱监督版本，例如：

- `scribble_tmi_video_annotations.coco.json`

它与已有 full 版 JSON 共享同一套 `volumes/*.npz`，因此默认要求输出目录下已经存在：

- `volumes/<case>.npz`

典型用法
--------
ACDC::

    python gq_scripts/preprocess/preprocess_video_scribble_annotations.py \
        --dataset acdc \
        --input_img_dir /home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_100/train/img \
        --input_scribble_dir /home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_100/train/acdc_scribbles_TMI \
        --output_json_path /home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/train/scribble_tmi_video_annotations.coco.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from pycocotools import mask as mask_utils
from tqdm import tqdm


DatasetCfg = Tuple[Dict[int, str], Optional[int], Callable[[str, str], List[str]]]


DATASET_CONFIG: Dict[str, DatasetCfg] = {
    "acdc": (
        {1: "right ventricle", 2: "myocardium", 3: "left ventricle"},
        4,
        lambda b, n: [f"{b}_scribble.nii.gz", f"{b}_gt.nii.gz", n],
    ),
    "btcv": (
        {1: "bladder", 2: "uterus", 3: "rectum", 4: "small bowel"},
        5,
        lambda b, n: [n.replace("-Image", "-Mask")],
    ),
    "promise": (
        {1: "prostate"},
        2,
        lambda b, n: [f"{b}_segmentation.nii.gz"],
    ),
    "isbi": (
        {1: "peripheral zone", 2: "central gland"},
        0,
        lambda b, n: [n],
    ),
    "mscmr": (
        {1: "right ventricle", 2: "myocardium", 3: "left ventricle"},
        4,
        lambda b, n: [f"{b}_scribble.nii.gz"],
    ),
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def rle_encode(mask: np.ndarray) -> Optional[Dict[str, Any]]:
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


def build_sparse_scribble_track(
    scribble_vol: np.ndarray,
    category_id: int,
    kept_slice_indices: List[int],
    ignore_label: Optional[int],
) -> Dict[str, List[Any]]:
    frame_indices: List[int] = []
    bboxes: List[List[float]] = []
    segmentations: List[Dict[str, Any]] = []
    valid_masks: List[Dict[str, Any]] = []
    areas: List[float] = []

    for local_idx, orig_idx in enumerate(kept_slice_indices):
        scribble_slice = scribble_vol[orig_idx]
        binary_mask = (scribble_slice == category_id).astype(np.uint8)
        if not np.any(binary_mask):
            continue

        bbox = mask_to_bbox(binary_mask)
        seg_rle = rle_encode(binary_mask)
        if bbox is None or seg_rle is None:
            continue

        if ignore_label is None:
            union_mask = (scribble_slice > 0).astype(np.uint8)
        else:
            union_mask = (scribble_slice != ignore_label).astype(np.uint8)
        valid_rle = rle_encode(union_mask)
        if valid_rle is None:
            continue

        frame_indices.append(int(local_idx))
        bboxes.append(bbox)
        segmentations.append(seg_rle)
        valid_masks.append(valid_rle)
        areas.append(float(np.sum(binary_mask)))

    return {
        "frame_indices": frame_indices,
        "bboxes": bboxes,
        "segmentations": segmentations,
        "valid_masks": valid_masks,
        "areas": areas,
    }


def _load_existing_volume_meta(npz_path: Path) -> Tuple[List[int], List[float], Tuple[int, int], int]:
    with np.load(npz_path) as data:
        if "slice_indices" not in data:
            raise KeyError(f"Missing 'slice_indices' in {npz_path}")
        kept_slice_indices = data["slice_indices"].astype(np.int32).tolist()
        spacing_xyz = (
            data["spacing_xyz"].astype(np.float32).tolist()
            if "spacing_xyz" in data
            else None
        )
        volume = data["volume"]
        height, width = int(volume.shape[1]), int(volume.shape[2])
        original_length = max(kept_slice_indices) + 1 if kept_slice_indices else int(volume.shape[0])
    return kept_slice_indices, spacing_xyz, (height, width), original_length


def process_split(
    img_dir: Path,
    scribble_dir: Path,
    output_json_path: Path,
    category_map: Dict[int, str],
    ignore_label: Optional[int],
    get_scribble_candidates: Callable[[str, str], List[str]],
    dataset_name: str,
    max_volumes: Optional[int],
) -> None:
    if not img_dir.exists() or not scribble_dir.exists():
        raise FileNotFoundError(f"Missing input directories: {img_dir} or {scribble_dir}")

    output_json_path = output_json_path.expanduser().resolve()
    output_root = output_json_path.parent
    volumes_dir = output_root / "volumes"
    if not volumes_dir.is_dir():
        raise FileNotFoundError(
            f"Expected existing volumes directory at {volumes_dir}. "
            "Please generate the full 3D dataset first."
        )

    video_coco: Dict[str, Any] = {
        "info": {"description": f"{dataset_name} scribble video annotations"},
        "videos": [],
        "annotations": [],
        "categories": [{"id": k, "name": v} for k, v in sorted(category_map.items())],
    }

    image_files = sorted(img_dir.glob("*.nii.gz"))
    if max_volumes is not None:
        image_files = image_files[:max_volumes]

    file_pairs: List[Tuple[Path, Path]] = []
    for img_path in image_files:
        base_name = img_path.stem.replace(".nii", "")
        img_name = img_path.name
        for scribble_filename in get_scribble_candidates(base_name, img_name):
            candidate = scribble_dir / scribble_filename
            if candidate.exists():
                file_pairs.append((img_path, candidate))
                break
        else:
            logging.warning("Scribble file not found for %s, skipped.", img_path.name)

    logging.info("Processing %d volumes for weak video annotations ...", len(file_pairs))

    video_id = 0
    ann_id = 0

    for img_path, scribble_path in tqdm(file_pairs, desc="Volumes"):
        base_name = img_path.stem.replace(".nii", "")
        npz_rel_path = Path("volumes") / f"{base_name}.npz"
        npz_path = output_root / npz_rel_path
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"Expected existing volume NPZ at {npz_path}. "
                "Weak video annotations are designed to reuse the full dataset volumes."
            )

        kept_slice_indices, spacing_xyz, (height, width), original_length = _load_existing_volume_meta(npz_path)

        scribble_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(scribble_path))).astype(np.uint8)
        depth = int(scribble_vol.shape[0])
        if depth < original_length:
            raise ValueError(
                f"Scribble depth {depth} is smaller than referenced original_length "
                f"{original_length} for {base_name}"
            )

        if spacing_xyz is None:
            img_sitk = sitk.ReadImage(str(img_path))
            spacing_xyz = [float(v) for v in img_sitk.GetSpacing()]

        video_coco["videos"].append(
            {
                "id": video_id,
                "video_name": base_name,
                "npz_path": str(npz_rel_path),
                "length": len(kept_slice_indices),
                "height": int(height),
                "width": int(width),
                "slice_indices": kept_slice_indices,
                "spacing_xyz": [float(v) for v in spacing_xyz],
                "original_length": int(original_length),
            }
        )

        for category_id in sorted(category_map):
            track = build_sparse_scribble_track(
                scribble_vol=scribble_vol,
                category_id=category_id,
                kept_slice_indices=kept_slice_indices,
                ignore_label=ignore_label,
            )
            if not track["frame_indices"]:
                continue

            video_coco["annotations"].append(
                {
                    "id": ann_id,
                    "video_id": video_id,
                    "category_id": int(category_id),
                    "iscrowd": 0,
                    **track,
                }
            )
            ann_id += 1

        video_id += 1

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(video_coco, f, indent=2, ensure_ascii=False)

    logging.info(
        "Saved %s (%d videos, %d annotations)",
        output_json_path,
        len(video_coco["videos"]),
        len(video_coco["annotations"]),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export 3D scribble annotations as SAM3 video-style weak-supervision COCO JSON.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset type: acdc / btcv / promise / isbi / mscmr",
    )
    parser.add_argument(
        "--input_img_dir",
        type=str,
        required=True,
        help="Input image directory (.nii.gz), used to match cases and recover metadata.",
    )
    parser.add_argument(
        "--input_scribble_dir",
        type=str,
        required=True,
        help="Input scribble directory (.nii.gz).",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        required=True,
        help="Output weak video JSON path. Its parent directory must already contain volumes/.",
    )
    parser.add_argument(
        "--max_volumes",
        type=int,
        default=None,
        help="Optional debug cap on the number of volumes to process.",
    )

    args = parser.parse_args()
    setup_logging()

    category_map, ignore_label, get_scribble_candidates = DATASET_CONFIG[args.dataset]
    process_split(
        img_dir=Path(args.input_img_dir).expanduser().resolve(),
        scribble_dir=Path(args.input_scribble_dir).expanduser().resolve(),
        output_json_path=Path(args.output_json_path).expanduser().resolve(),
        category_map=category_map,
        ignore_label=ignore_label,
        get_scribble_candidates=get_scribble_candidates,
        dataset_name=args.dataset,
        max_volumes=args.max_volumes,
    )


if __name__ == "__main__":
    main()

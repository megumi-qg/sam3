#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 NCI-ISBI (PROSTATE) 的 DICOM 影像 + NRRD 标注转换为 NIfTI。

输入目录默认结构:
  <raw_root>/
    Training/
      image/.../<CaseID>/.../<SeriesUID>/*.dcm
      label/*.nrrd
    Test/
      image/.../<CaseID>/.../<SeriesUID>/*.dcm
      label/*.nrrd
    Leaderboard/
      image/.../<CaseID>/.../<SeriesUID>/*.dcm
      label/*.nrrd

输出目录:
  <output_root>/
    training/
      images/<CaseID>.nii.gz
      labels/<CaseID>.nii.gz
    test/
      images/<CaseID>.nii.gz
      labels/<CaseID>.nii.gz
    leaderboard/
      images/<CaseID>.nii.gz
      labels/<CaseID>.nii.gz

说明:
- 自动处理标签命名差异:
  - ProstateDx-xx-xxxx.nrrd
  - ProstateDx-xx-xxxx_correctedLabels.nrrd
  - ProstateDx-xx-xxxx_truth.nrrd
- 若同一病例出现多份标签，优先级:
  correctedLabels > truth > plain
- 标签会按影像几何信息进行最近邻重采样对齐（仅在不一致时）。
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk


CASE_RE = re.compile(r"^Prostate(?:Dx|3T)-\d{2}-\d{4}$")
SPLIT_NAME_MAP = {
    "Training": "training",
    "Test": "test",
    "Leaderboard": "leaderboard",
}


@dataclass
class LabelChoice:
    case_id: str
    path: Path
    priority: int


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_label_case(label_path: Path) -> Optional[LabelChoice]:
    name = label_path.name
    if not name.endswith(".nrrd"):
        return None

    if name.endswith("_correctedLabels.nrrd"):
        case_id = name[: -len("_correctedLabels.nrrd")]
        priority = 3
    elif name.endswith("_truth.nrrd"):
        case_id = name[: -len("_truth.nrrd")]
        priority = 2
    else:
        case_id = name[: -len(".nrrd")]
        priority = 1

    if not CASE_RE.match(case_id):
        return None
    return LabelChoice(case_id=case_id, path=label_path, priority=priority)


def build_label_map(label_dir: Path) -> Dict[str, Path]:
    selected: Dict[str, LabelChoice] = {}
    for p in sorted(label_dir.glob("*.nrrd")):
        parsed = parse_label_case(p)
        if parsed is None:
            continue
        prev = selected.get(parsed.case_id)
        if prev is None or parsed.priority > prev.priority:
            selected[parsed.case_id] = parsed
    return {k: v.path for k, v in selected.items()}


def index_case_dirs(image_root: Path) -> Dict[str, Path]:
    case_dirs: Dict[str, Path] = {}
    for p in image_root.rglob("*"):
        if not p.is_dir():
            continue
        name = p.name
        if CASE_RE.match(name):
            case_dirs[name] = p
    return case_dirs


def find_series_dir(case_dir: Path) -> Tuple[Optional[Path], int, int]:
    """
    返回 (series_dir, dicom_count, candidate_count)。
    若存在多个包含 dcm 的候选目录，选择 dcm 数量最多者。
    """
    candidates: List[Tuple[Path, int]] = []
    for p in case_dir.rglob("*"):
        if not p.is_dir():
            continue
        files = list(p.iterdir())
        cnt = sum(1 for f in files if f.is_file() and f.suffix.lower() == ".dcm")
        if cnt > 0:
            candidates.append((p, cnt))

    if not candidates:
        return None, 0, 0

    candidates.sort(key=lambda x: (-x[1], str(x[0])))
    best_dir, best_cnt = candidates[0]
    return best_dir, best_cnt, len(candidates)


def read_dicom_volume(series_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
    if series_ids:
        series_files = reader.GetGDCMSeriesFileNames(str(series_dir), series_ids[0])
    else:
        series_files = sorted(str(p) for p in series_dir.glob("*.dcm"))
    if not series_files:
        raise RuntimeError(f"No DICOM files found in {series_dir}")
    reader.SetFileNames(series_files)
    return reader.Execute()


def resample_label_if_needed(label: sitk.Image, image: sitk.Image) -> sitk.Image:
    same = (
        label.GetSize() == image.GetSize()
        and label.GetSpacing() == image.GetSpacing()
        and label.GetOrigin() == image.GetOrigin()
        and label.GetDirection() == image.GetDirection()
    )
    if same:
        return label
    return sitk.Resample(
        label,
        image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label.GetPixelID(),
    )


def process_split(raw_root: Path, out_root: Path, split_name: str) -> Dict[str, int]:
    split_dir = raw_root / split_name
    image_root = split_dir / "image"
    label_root = split_dir / "label"
    out_split = out_root / SPLIT_NAME_MAP[split_name]
    out_img_dir = out_split / "images"
    out_lab_dir = out_split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"split 目录缺失 image/label: {split_dir}")

    label_map = build_label_map(label_root)
    case_dir_map = index_case_dirs(image_root)

    converted = 0
    skipped_missing_image = 0
    skipped_missing_label = 0
    multi_series_cases = 0
    failed = 0

    all_cases = sorted(set(label_map.keys()) | set(case_dir_map.keys()))
    logging.info(
        "[%s] cases total=%d, with_label=%d, with_image=%d",
        split_name,
        len(all_cases),
        len(label_map),
        len(case_dir_map),
    )

    for case_id in all_cases:
        case_dir = case_dir_map.get(case_id)
        label_path = label_map.get(case_id)

        if case_dir is None:
            skipped_missing_image += 1
            logging.warning("[%s] missing image case dir: %s", split_name, case_id)
            continue
        if label_path is None:
            skipped_missing_label += 1
            logging.warning("[%s] missing label file: %s", split_name, case_id)
            continue

        series_dir, dcm_count, candidate_count = find_series_dir(case_dir)
        if series_dir is None:
            failed += 1
            logging.error("[%s] no dcm found under case: %s", split_name, case_id)
            continue
        if candidate_count > 1:
            multi_series_cases += 1
            logging.warning(
                "[%s] %s has %d series candidates, selected %s (dcm=%d)",
                split_name,
                case_id,
                candidate_count,
                series_dir,
                dcm_count,
            )

        try:
            img = read_dicom_volume(series_dir)
            lab = sitk.ReadImage(str(label_path))
            lab = resample_label_if_needed(lab, img)

            out_img = out_img_dir / f"{case_id}.nii.gz"
            out_lab = out_lab_dir / f"{case_id}.nii.gz"
            sitk.WriteImage(img, str(out_img))
            sitk.WriteImage(lab, str(out_lab))
            converted += 1
        except Exception as e:
            failed += 1
            logging.exception("[%s] failed case %s: %s", split_name, case_id, e)

    stats = {
        "split": split_name,
        "total_cases_union": len(all_cases),
        "converted": converted,
        "skipped_missing_image": skipped_missing_image,
        "skipped_missing_label": skipped_missing_label,
        "multi_series_cases": multi_series_cases,
        "failed": failed,
    }
    logging.info("[%s] done: %s", split_name, stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ISBI DICOM+NRRD to NIfTI")
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=Path("/home/gaoqi/dataset/using/isbi/raw"),
        help="ISBI raw root",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/gaoqi/dataset/using/isbi/processed/nifti"),
        help="Output root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Training", "Test", "Leaderboard"],
        choices=["Training", "Test", "Leaderboard"],
        help="Splits to process",
    )
    args = parser.parse_args()

    setup_logging()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_stats: List[Dict[str, int]] = []
    for split in args.splits:
        stats = process_split(raw_root, output_root, split)
        all_stats.append(stats)

    summary = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "splits": args.splits,
        "stats": all_stats,
    }
    summary_path = output_root / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info("summary saved: %s", summary_path)


if __name__ == "__main__":
    main()

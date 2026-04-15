#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 WSL4MIS 的 Prostate_training_volumes/*.h5 转换为 NIfTI，并按编号切分 train/test。

划分规则（固定）:
- patient001 ~ patient060 -> train
- patient061 ~ patient080 -> test

输出目录结构:
  <output_root>/
    train/
      image/
      label/
      scribble/
    test/
      image/
      label/
      scribble/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import h5py
import SimpleITK as sitk


PATTERN = re.compile(r"^patient(\d{3})\.h5$")


def decide_split(patient_idx: int) -> str:
    if 1 <= patient_idx <= 60:
        return "train"
    if 61 <= patient_idx <= 80:
        return "test"
    raise ValueError(f"patient index out of expected range [1, 80]: {patient_idx}")


def to_nifti(arr, out_path: Path) -> None:
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ISBI scribble h5 volumes to split NIfTI")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/gaoqi/dataset/using/isbi/processed/scribble/Prostate_training_volumes"),
        help="Directory containing patientXXX.h5 files",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/gaoqi/dataset/using/isbi/processed/scribble_nifti_split"),
        help="Output root directory",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {"train": 0, "test": 0}
    converted = 0
    skipped = 0

    for h5_path in sorted(input_dir.glob("patient*.h5")):
        m = PATTERN.match(h5_path.name)
        if m is None:
            skipped += 1
            continue

        patient_idx = int(m.group(1))
        split = decide_split(patient_idx)
        case_id = h5_path.stem  # patient001

        split_dir = output_root / split
        image_dir = split_dir / "image"
        label_dir = split_dir / "label"
        scribble_dir = split_dir / "scribble"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        scribble_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(h5_path, "r") as f:
            image = f["image"][:]
            label = f["label"][:]
            scribble = f["scribble"][:]

        to_nifti(image, image_dir / f"{case_id}.nii.gz")
        to_nifti(label, label_dir / f"{case_id}.nii.gz")
        to_nifti(scribble, scribble_dir / f"{case_id}.nii.gz")

        counts[split] += 1
        converted += 1

    summary = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "converted": converted,
        "skipped": skipped,
        "split_counts": counts,
        "split_rule": {
            "train": "patient001-patient060",
            "test": "patient061-patient080",
        },
    }
    summary_path = output_root / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

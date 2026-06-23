#!/usr/bin/env python3
"""Resize SAT-Pro CMPB masks to the CMPB NIfTI grid for visualization.

SAT-Pro stores its exported images and segmentations in its own intermediate
grid. ITK-SNAP requires the main image and segmentation to have the same array
shape, so this script nearest-neighbor resizes the combined SAT label maps to
the corresponding CMPB reference images and copies the reference affine/header.

These outputs are for qualitative visualization only, not metric computation.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


OUT_ROOT = Path("/home/gaoqi/sam3/gq_experiment/cmpb/sat_pro_resampled_to_cmpb_grid")

DATASETS = {
    "acdc": {
        "sat_dir": Path("/home/gaoqi/SAT/gq_dataset/ACDC/test/results_pro_cmpb/ACDC"),
        "ref_dir": Path("/home/gaoqi/dataset/nnU-Net/raw/Dataset120_ACDCS_CMPB/imagesTs"),
        "prefix": "seg_",
        "suffix": "",
        "ref_name": lambda case: f"{case}_0000.nii.gz",
        "out_name": lambda case: f"{case}.nii.gz",
    },
    "btcv": {
        "sat_dir": Path("/home/gaoqi/SAT/gq_dataset/BTCV/test/results_pro_cmpb/BTCV"),
        "ref_dir": Path("/home/gaoqi/dataset/nnU-Net/raw/Dataset121_BTCVS_CMPB/imagesTs"),
        "prefix": "seg_",
        "suffix": "-Image",
        "flip_axes": (0, 1),
        "ref_name": lambda case: f"{case}_0000.nii.gz",
        "out_name": lambda case: f"{case}.nii.gz",
    },
    "promise12": {
        "sat_dir": Path("/home/gaoqi/SAT/gq_dataset/PROMISE12/test/results_pro_cmpb/PROMISE12"),
        "ref_dir": Path("/home/gaoqi/dataset/nnU-Net/raw/Dataset122_PROMISES_CMPB/imagesTs"),
        "prefix": "seg_",
        "suffix": "",
        "ref_name": lambda case: f"{case}_0000.nii.gz",
        "out_name": lambda case: f"{case}.nii.gz",
    },
}


def case_id_from_sat_path(path: Path, prefix: str, suffix: str) -> str:
    name = path.name.removesuffix(".nii.gz")
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected SAT mask name: {path.name}")
    case = name[len(prefix):]
    if suffix and case.endswith(suffix):
        case = case[: -len(suffix)]
    return case


def resize_nearest(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    factors = [target / source for target, source in zip(target_shape, mask.shape)]
    resized = zoom(mask, factors, order=0, mode="nearest", prefilter=False)
    if resized.shape != target_shape:
        slices = tuple(slice(0, size) for size in target_shape)
        fixed = np.zeros(target_shape, dtype=resized.dtype)
        src_slices = tuple(slice(0, min(a, b)) for a, b in zip(resized.shape, target_shape))
        fixed[slices] = resized[src_slices]
        resized = fixed
    return resized.astype(np.uint8)


def main() -> int:
    total = 0
    for dataset, cfg in DATASETS.items():
        out_dir = OUT_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        sat_masks = sorted(cfg["sat_dir"].glob(f"{cfg['prefix']}*.nii.gz"))
        for sat_path in sat_masks:
            case = case_id_from_sat_path(sat_path, cfg["prefix"], cfg["suffix"])
            ref_path = cfg["ref_dir"] / cfg["ref_name"](case)
            if not ref_path.exists():
                print(f"SKIP missing reference for {dataset} {case}: {ref_path}")
                continue

            sat_img = nib.load(str(sat_path))
            ref_img = nib.load(str(ref_path))
            mask = np.asarray(sat_img.dataobj)
            resized = resize_nearest(mask, ref_img.shape)
            for axis in cfg.get("flip_axes", ()):
                resized = np.flip(resized, axis=axis)

            header = ref_img.header.copy()
            header.set_data_dtype(np.uint8)
            out_img = nib.Nifti1Image(resized, ref_img.affine, header)
            out_path = out_dir / cfg["out_name"](case)
            nib.save(out_img, str(out_path))
            total += 1
        print(f"{dataset}: wrote {len(list(out_dir.glob('*.nii.gz')))} masks to {out_dir}")

    print(f"Total converted masks: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

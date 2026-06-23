#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build CMPB clean train/val/test datasets for SAM3-Scribble.

This script fixes patient-level splits and creates new clean processed datasets
without modifying the original processed directories.

ACDC:
  Uses the existing expert-scribble processed split.

BTCV:
  Source: /home/gaoqi/dataset/using/btcv/processed/nifti_train_test_img_gt_scribble
  Split: train 20 / val 4 from the original train source, test 6 unchanged.
  Modality: CT, window level 40, width 400.

PROMISE12:
  Source: /home/gaoqi/dataset/using/promise12/processed/nifti_split_img_gt_scribble
  Split: train 40 / val 10 from the train source, test 30 unchanged from test source.
  The train-source and test-source case IDs may overlap but refer to different
  volumes in this local dataset; split files use source prefixes for clarity.
  Modality: MRI, percentile clipping.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gq_scripts.preprocess import preprocess_full_annotations
from gq_scripts.preprocess import preprocess_scribble_annotations

SPLIT_DIR = REPO_ROOT / "gq_paper" / "cmpb" / "splits"

ACDC_PROCESSED = Path(
    "/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100"
)
ACDC_OUT = Path(
    "/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100_cmpb_clean"
)

BTCV_NIFTI = Path(
    "/home/gaoqi/dataset/using/btcv/processed/nifti_train_test_img_gt_scribble"
)
BTCV_OUT = Path(
    "/home/gaoqi/dataset/using/btcv/processed/png_coco_sam3_slices_cmpb_clean"
)

PROMISE_NIFTI = Path(
    "/home/gaoqi/dataset/using/promise12/processed/nifti_split_img_gt_scribble"
)
PROMISE_OUT = Path(
    "/home/gaoqi/dataset/using/promise12/processed/png_coco_sam3_cmpb_clean"
)


BTCV_SPLITS = {
    "train": [
        "0759564",
        "0773652",
        "1411226",
        "1565722",
        "2477092",
        "2609008",
        "2780380",
        "3089528",
        "3463338",
        "3744998",
        "4526856",
        "5502532",
        "5664630",
        "6171298",
        "6339208",
        "6682806",
        "6798630",
        "7657884",
        "8745574",
        "9570942",
    ],
    "val": ["1577656", "2088692", "2469782", "3388252"],
    "test": ["0507688", "0763890", "1578068", "5176452", "5458334", "7742556"],
}

PROMISE_SPLITS = {
    "train": [
        "Case00",
        "Case01",
        "Case02",
        "Case03",
        "Case06",
        "Case07",
        "Case08",
        "Case10",
        "Case11",
        "Case12",
        "Case13",
        "Case14",
        "Case15",
        "Case16",
        "Case17",
        "Case19",
        "Case20",
        "Case22",
        "Case23",
        "Case24",
        "Case25",
        "Case26",
        "Case27",
        "Case30",
        "Case31",
        "Case32",
        "Case33",
        "Case37",
        "Case38",
        "Case39",
        "Case40",
        "Case41",
        "Case42",
        "Case43",
        "Case44",
        "Case45",
        "Case46",
        "Case47",
        "Case48",
        "Case49",
    ],
    "val": [
        "Case04",
        "Case05",
        "Case09",
        "Case18",
        "Case21",
        "Case28",
        "Case29",
        "Case34",
        "Case35",
        "Case36",
    ],
    "test": [
        "Case00",
        "Case01",
        "Case02",
        "Case03",
        "Case04",
        "Case05",
        "Case06",
        "Case07",
        "Case08",
        "Case09",
        "Case10",
        "Case11",
        "Case12",
        "Case13",
        "Case14",
        "Case15",
        "Case16",
        "Case17",
        "Case18",
        "Case19",
        "Case20",
        "Case21",
        "Case22",
        "Case23",
        "Case24",
        "Case25",
        "Case26",
        "Case27",
        "Case28",
        "Case29",
    ],
}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def write_split_file(path: Path, ids: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")
    logging.info("Wrote %s", path)


def list_cases_from_images(split_dir: Path, dataset: str) -> list[str]:
    ids = set()
    image_dir = split_dir / "images"
    for p in image_dir.glob("*.png"):
        name = p.name
        if dataset == "acdc":
            ids.add(name.split("_frame")[0])
        elif dataset == "btcv":
            ids.add(name.split("-Image_slice")[0])
        elif dataset == "promise":
            ids.add(name.split("_slice")[0])
    return sorted(ids)


def symlink_or_replace(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def symlink_dir_or_replace(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    dst.symlink_to(src, target_is_directory=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_acdc_clean() -> None:
    logging.info("Building acdc clean aliases from expert-scribble processed split")
    ACDC_OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        src = ACDC_PROCESSED / split
        dst = ACDC_OUT / split
        dst.mkdir(parents=True, exist_ok=True)
        symlink_dir_or_replace(src / "images", dst / "images")
        symlink_dir_or_replace(src / "masks", dst / "masks")

        full_src = src / "full_annotations.coco.json"
        image_src = src / "image_annotations.coco.json"
        if full_src.exists():
            copy_if_exists(full_src, dst / "full_annotations.coco.json")
            copy_if_exists(full_src, dst / "image_annotations.coco.json")
        elif image_src.exists():
            copy_if_exists(image_src, dst / "image_annotations.coco.json")
            copy_if_exists(image_src, dst / "full_annotations.coco.json")

        scribble_src = src / "scribble_tmi_annotations.coco.json"
        copy_if_exists(scribble_src, dst / "scribble_tmi_annotations.coco.json")
        copy_if_exists(scribble_src, dst / "scribble_annotations.coco.json")
        copy_if_exists(src / "spacing_map.json", dst / "spacing_map.json")


def prepare_nifti_split(
    source_root: Path,
    staging_root: Path,
    dataset: str,
    split: str,
    case_ids: list[str],
    source_split: str,
) -> tuple[Path, Path, Path | None]:
    img_out = staging_root / split / "img"
    gt_out = staging_root / split / "gt"
    scribble_out = staging_root / split / "scribble_bench"
    for d in [img_out, gt_out, scribble_out]:
        d.mkdir(parents=True, exist_ok=True)

    for case_id in case_ids:
        if dataset == "btcv":
            img_name = f"{case_id}-Image.nii.gz"
            gt_name = f"{case_id}-Mask.nii.gz"
            scribble_name = f"{case_id}-Mask.nii.gz"
        elif dataset == "promise":
            img_name = f"{case_id}.nii.gz"
            gt_name = f"{case_id}_segmentation.nii.gz"
            scribble_name = f"{case_id}_segmentation.nii.gz"
        else:
            raise ValueError(dataset)

        symlink_or_replace(source_root / source_split / "img" / img_name, img_out / img_name)
        symlink_or_replace(source_root / source_split / "gt" / gt_name, gt_out / gt_name)

        scribble_src = source_root / source_split / "scribble_bench" / scribble_name
        if scribble_src.exists():
            symlink_or_replace(scribble_src, scribble_out / scribble_name)

    spacing_src = source_root / source_split / "spacing_map.json"
    if spacing_src.exists():
        shutil.copy2(spacing_src, staging_root / split / "spacing_map.json")

    return img_out, gt_out, scribble_out if any(scribble_out.iterdir()) else None


def alias_full_annotations(split_out: Path) -> None:
    image_json = split_out / "image_annotations.coco.json"
    full_json = split_out / "full_annotations.coco.json"
    if image_json.exists():
        shutil.copy2(image_json, full_json)


def build_dataset(
    dataset: str,
    source_root: Path,
    out_root: Path,
    splits: dict[str, list[str]],
    source_split_for: dict[str, str],
    full_dataset_arg: str,
    scribble_dataset_arg: str,
    modality: str,
) -> None:
    staging_root = out_root / "_nifti_splits"
    staging_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    for split, ids in splits.items():
        logging.info("Building %s %s (%d cases)", dataset, split, len(ids))
        source_split = source_split_for[split]
        img_dir, gt_dir, scribble_dir = prepare_nifti_split(
            source_root, staging_root, dataset, split, ids, source_split
        )

        split_out = out_root / split
        preprocess_full_annotations.process(
            img_dir=img_dir,
            gt_dir=gt_dir,
            output_dir=split_out,
            slice_policy="all",
            dataset=full_dataset_arg,
            modality=modality,
            window_level=40,
            window_width=400,
        )
        alias_full_annotations(split_out)

        if scribble_dir is not None:
            category_map, ignore_label, get_candidates = (
                preprocess_scribble_annotations.DATASET_CONFIG[scribble_dataset_arg]
            )
            preprocess_scribble_annotations.process_split(
                img_dir=img_dir,
                scribble_dir=scribble_dir,
                output_json_path=split_out / "scribble_annotations.coco.json",
                category_map=category_map,
                ignore_label=ignore_label,
                get_scribble_candidates=get_candidates,
                dataset_name=dataset.upper(),
                split=split,
                slice_policy="all",
            )


def write_all_split_files() -> None:
    for split in ["train", "val", "test"]:
        acdc_ids = list_cases_from_images(ACDC_PROCESSED / split, "acdc")
        write_split_file(SPLIT_DIR / f"acdc_{split}.txt", acdc_ids)

    for split, ids in BTCV_SPLITS.items():
        write_split_file(SPLIT_DIR / f"btcv_{split}.txt", ids)

    for split, ids in PROMISE_SPLITS.items():
        prefix = "test_source/" if split == "test" else "train_source/"
        write_split_file(SPLIT_DIR / f"promise12_{split}.txt", [prefix + x for x in ids])

    manifest = {
        "acdc": {
            "source_processed_dir": str(ACDC_PROCESSED),
            "processed_dir": str(ACDC_OUT),
            "scribble": "expert scribble_tmi_annotations.coco.json",
            "split": {s: list_cases_from_images(ACDC_PROCESSED / s, "acdc") for s in ["train", "val", "test"]},
            "modality": "mri",
        },
        "btcv": {
            "source_dir": str(BTCV_NIFTI),
            "processed_dir": str(BTCV_OUT),
            "scribble": "ScribbleBench-generated",
            "split": BTCV_SPLITS,
            "modality": "ct",
        },
        "promise12": {
            "source_dir": str(PROMISE_NIFTI),
            "processed_dir": str(PROMISE_OUT),
            "scribble": "ScribbleBench-generated",
            "split": {
                "train": ["train_source/" + x for x in PROMISE_SPLITS["train"]],
                "val": ["train_source/" + x for x in PROMISE_SPLITS["val"]],
                "test": ["test_source/" + x for x in PROMISE_SPLITS["test"]],
            },
            "modality": "mri",
            "note": "train_source and test_source reuse case IDs but are content-distinct local subsets.",
        },
    }
    (SPLIT_DIR / "cmpb_clean_split_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


def main() -> None:
    setup_logging()
    write_all_split_files()
    build_acdc_clean()

    build_dataset(
        dataset="btcv",
        source_root=BTCV_NIFTI,
        out_root=BTCV_OUT,
        splits=BTCV_SPLITS,
        source_split_for={"train": "train", "val": "train", "test": "test"},
        full_dataset_arg="btcv_cervix",
        scribble_dataset_arg="btcv",
        modality="ct",
    )

    build_dataset(
        dataset="promise",
        source_root=PROMISE_NIFTI,
        out_root=PROMISE_OUT,
        splits=PROMISE_SPLITS,
        source_split_for={"train": "train", "val": "train", "test": "test"},
        full_dataset_arg="promise12",
        scribble_dataset_arg="promise",
        modality="mri",
    )


if __name__ == "__main__":
    main()

import importlib.util
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_preprocess_registers_isbi():
    module = _load_module(
        "preprocess_full_annotations",
        "gq_scripts/preprocess/preprocess_full_annotations.py",
    )

    assert "isbi" in module.DATASET_CONFIG
    assert module.DATASET_CONFIG["isbi"]["gt_replace"] == (".nii.gz", ".nii.gz")
    assert module.DATASET_CONFIG["isbi"]["categories"] == {
        1: "peripheral zone",
        2: "central gland",
    }


def test_scribble_preprocess_registers_isbi():
    module = _load_module(
        "preprocess_scribble_annotations",
        "gq_scripts/preprocess/preprocess_scribble_annotations.py",
    )

    category_map, ignore_label, get_candidates = module.DATASET_CONFIG["isbi"]

    assert category_map == {1: "peripheral zone", 2: "central gland"}
    assert ignore_label == 0
    assert get_candidates("patient001", "patient001.nii.gz") == ["patient001.nii.gz"]


def test_full_preprocess_normalizes_isbi_background_label(tmp_path):
    module = _load_module(
        "preprocess_full_annotations",
        "gq_scripts/preprocess/preprocess_full_annotations.py",
    )

    img_dir = tmp_path / "images"
    gt_dir = tmp_path / "labels"
    out_dir = tmp_path / "out"
    img_dir.mkdir()
    gt_dir.mkdir()

    img_vol = np.array([[[0.0, 1.0], [2.0, 3.0]]], dtype=np.float32)
    gt_vol = np.array([[[1, 2], [3, 0]]], dtype=np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(img_vol), str(img_dir / "patient001.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(gt_vol), str(gt_dir / "patient001.nii.gz"))

    module.process(
        img_dir=img_dir,
        gt_dir=gt_dir,
        output_dir=out_dir,
        slice_policy="all",
        dataset="isbi",
        modality="mri",
        coco_json_name="full_annotations.coco.json",
    )

    mask = np.array(Image.open(out_dir / "masks" / "patient001_slice000.png"))
    assert set(np.unique(mask).tolist()) == {0, 1, 2}

    coco = json.loads((out_dir / "full_annotations.coco.json").read_text())
    assert coco["categories"] == [
        {"id": 1, "name": "peripheral zone"},
        {"id": 2, "name": "central gland"},
    ]
    assert {ann["category_id"] for ann in coco["annotations"]} == {1, 2}


def test_scribble_preprocess_uses_nonzero_isbi_labels_for_valid_mask(tmp_path):
    module = _load_module(
        "preprocess_scribble_annotations",
        "gq_scripts/preprocess/preprocess_scribble_annotations.py",
    )

    img_dir = tmp_path / "images"
    scribble_dir = tmp_path / "scribble"
    out_json = tmp_path / "scribble_annotations.coco.json"
    img_dir.mkdir()
    scribble_dir.mkdir()

    img_vol = np.zeros((1, 2, 3), dtype=np.float32)
    scribble_vol = np.array([[[0, 1, 3], [2, 0, 3]]], dtype=np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(img_vol), str(img_dir / "patient001.nii.gz"))
    sitk.WriteImage(
        sitk.GetImageFromArray(scribble_vol),
        str(scribble_dir / "patient001.nii.gz"),
    )

    category_map, ignore_label, get_candidates = module.DATASET_CONFIG["isbi"]
    module.process_split(
        img_dir=img_dir,
        scribble_dir=scribble_dir,
        output_json_path=out_json,
        category_map=category_map,
        ignore_label=ignore_label,
        get_scribble_candidates=get_candidates,
        dataset_name="ISBI",
        split="train",
        slice_policy="all",
    )

    coco = json.loads(out_json.read_text())
    assert coco["categories"] == [
        {"id": 1, "name": "peripheral zone"},
        {"id": 2, "name": "central gland"},
    ]
    assert {ann["category_id"] for ann in coco["annotations"]} == {1, 2}

    expected_valid = (scribble_vol[0] > 0).astype(np.uint8)
    for ann in coco["annotations"]:
        valid_mask = mask_utils.decode(ann["valid_mask"])
        assert np.array_equal(valid_mask, expected_valid)

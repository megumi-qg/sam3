"""
Convert ``batch_inference.py`` predictions.pkl to COCO result JSON.

The detector-only evaluation pipeline stores final binary masks in a pickle
grouped by volume/category/slice.  Tracker merge scripts consume COCO result
JSON keyed by the video/frame COCO image ids.  This utility bridges the two
formats so the official detector-only masks can be used as the detector branch
in post-hoc tracker merges.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def _load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def _mask_to_coco_result(
    *,
    image_id: int,
    category_id: int,
    mask: np.ndarray,
    score: float,
) -> dict:
    mask_u8 = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_u8)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    height, width = mask.shape
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": rle,
        "score": float(score),
        "area": float(mask.sum() / max(1, height * width)),
        "bbox": [0.0, 0.0, 0.0, 0.0],
    }


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)
    return (
        np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
        )
        > 0
    )


def _frame_index(frame_coco: dict) -> dict[tuple[str, int], dict]:
    index = {}
    for image in frame_coco["images"]:
        file_name = os.path.basename(image["file_name"])
        volume_name = os.path.splitext(file_name)[0]
        slice_idx = int(image.get("original_slice_idx", image.get("frame_idx", 0)))
        index[(volume_name, slice_idx)] = image
    return index


def _lookup_frame(frames: dict[tuple[str, int], dict], volume_name: str, slice_idx: int):
    image = frames.get((volume_name, slice_idx))
    if image is not None:
        return image
    if volume_name.isdigit():
        image = frames.get((f"patient{volume_name}", slice_idx))
        if image is not None:
            return image
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert detector-only predictions.pkl into COCO result JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--frame_annotations_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--score",
        type=float,
        default=1.0,
        help="Score assigned to each non-empty final detector mask.",
    )
    parser.add_argument(
        "--include_empty",
        action="store_true",
        help="Write empty masks as zero-score predictions. Merge scripts do not need this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.predictions_file, "rb") as f:
        pred_data = pickle.load(f)

    frame_coco = _load_json(args.frame_annotations_json)
    frames = _frame_index(frame_coco)

    results = []
    missing = []
    for volume in pred_data["volumes"]:
        volume_name = str(volume["patient_name"])
        for category in volume["categories"]:
            category_id = int(category["category_id"])
            for slice_idx, mask in zip(category["slice_indices"], category["masks"]):
                image = _lookup_frame(frames, volume_name, int(slice_idx))
                if image is None:
                    missing.append((volume_name, int(slice_idx)))
                    continue
                height = int(image["height"])
                width = int(image["width"])
                mask = _resize_mask(np.asarray(mask).astype(bool), height, width)
                if not mask.any() and not args.include_empty:
                    continue
                score = args.score if mask.any() else 0.0
                results.append(
                    _mask_to_coco_result(
                        image_id=int(image["id"]),
                        category_id=category_id,
                        mask=mask,
                        score=score,
                    )
                )

    if missing:
        preview = ", ".join(f"{name}:{idx}" for name, idx in missing[:10])
        raise RuntimeError(
            f"{len(missing)} detector slices were not found in frame annotations. "
            f"First missing: {preview}"
        )

    results.sort(key=lambda item: (item["image_id"], item["category_id"]))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Wrote {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()

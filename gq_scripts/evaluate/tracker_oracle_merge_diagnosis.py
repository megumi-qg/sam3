"""
Oracle-merge and visualization diagnosis for ACDC SAM3 detector+tracker outputs.

This script intentionally uses GT only for analysis/oracle selection.  It should
not be used as a real inference protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import sam3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from batch_evaluate_utils import compute_dice, compute_iou, decode_rle_mask
from batch_inference import load_checkpoint_and_model
from tracker_auto_seed_inference import (
    _category_prompts_from_coco,
    _build_frame_metadata,
    _load_json,
    collect_detector_predictions_for_category,
    compute_tracker_cached_features,
    load_tracker_predictor,
    load_volume_pil_slices,
    mask_to_coco_result,
    prepare_volume_inputs,
    run_tracker_with_mode,
    select_conditioning_frames,
)


ACDC_CATEGORY_ID_TO_NAME = {
    1: "right ventricle",
    2: "myocardium",
    3: "left ventricle",
}


def _default_test_dir() -> str:
    return "/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/test"


def _default_image_checkpoint() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/scribble_video_lora_100/checkpoints/"
        "val_acdc_segmentation_coco_eval_segm_AP.pt"
    )


def _default_tracker_checkpoint() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/"
        "checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt"
    )


def _default_output_dir() -> str:
    return (
        "/home/gaoqi/sam3/gq_experiment/acdc/"
        "scribble_sam3_tracker_image_init_v1_oracle_diagnosis"
    )


def _build_gt_index(frame_coco: dict) -> Dict[Tuple[int, int], dict]:
    gt_index = {}
    for ann in frame_coco.get("annotations", []):
        gt_index[(int(ann["image_id"]), int(ann["category_id"]))] = ann
    return gt_index


def _decode_gt_mask(
    gt_index: Dict[Tuple[int, int], dict],
    image_id: int,
    category_id: int,
    height: int,
    width: int,
) -> np.ndarray:
    ann = gt_index.get((int(image_id), int(category_id)))
    if ann is None:
        return np.zeros((height, width), dtype=bool)
    mask = decode_rle_mask(ann["segmentation"], height, width)
    if mask.shape != (height, width):
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
        ) > 0
    return mask.astype(bool)


def _safe_pred(pred: Optional[dict], height: int, width: int) -> dict:
    if pred is None:
        return {
            "mask": np.zeros((height, width), dtype=bool),
            "score": 0.0,
            "area_px": 0,
            "area_ratio": 0.0,
        }
    mask = pred["mask"].astype(bool)
    if mask.shape != (height, width):
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
        ) > 0
    return {
        "mask": mask,
        "score": float(pred.get("score", 0.0)),
        "area_px": int(mask.sum()),
        "area_ratio": float(mask.sum() / max(1, height * width)),
    }


def _threshold_pred(pred: dict, *, threshold: float, strict: bool = False) -> dict:
    score = float(pred["score"])
    ok = score > threshold if strict else score >= threshold
    if ok and int(pred["area_px"]) > 0:
        return pred
    mask = np.zeros_like(pred["mask"], dtype=bool)
    return {
        **pred,
        "mask": mask,
        "area_px": 0,
        "area_ratio": 0.0,
    }


def _append_coco_prediction(
    predictions: List[dict],
    *,
    image_id: int,
    category_id: int,
    mask: np.ndarray,
    score: float,
) -> None:
    if not mask.any():
        return
    predictions.append(
        mask_to_coco_result(
            image_id=image_id,
            category_id=category_id,
            mask=mask,
            score=score,
        )
    )


def _overlay_mask(base: Image.Image, mask: np.ndarray, color: Tuple[int, int, int], alpha=95):
    img = base.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    mask_img = Image.fromarray(mask.astype(np.uint8) * int(alpha), mode="L").resize(
        img.size, Image.NEAREST
    )
    color_img = Image.new("RGBA", img.size, color + (0,))
    color_img.putalpha(mask_img)
    return Image.alpha_composite(img, color_img).convert("RGB")


def _make_tile(
    image: Image.Image,
    mask: np.ndarray,
    title: str,
    *,
    color: Tuple[int, int, int],
    size: int = 180,
) -> Image.Image:
    thumb = image.convert("RGB").resize((size, size), Image.BILINEAR)
    mask_resized = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.NEAREST)
    ) > 0
    vis = _overlay_mask(thumb, mask_resized, color=color)
    canvas = Image.new("RGB", (size, size + 34), "white")
    canvas.paste(vis, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, size + 8), title, fill=(0, 0, 0))
    return canvas


def _save_case_visualization(
    case: dict,
    pil_image: Image.Image,
    out_path: Path,
) -> None:
    size = 180
    tiles = [
        _make_tile(pil_image, case["gt_mask"], "GT", color=(0, 210, 70), size=size),
        _make_tile(
            pil_image,
            case["detector_mask"],
            f"Detector D={case['detector_dice']:.3f}",
            color=(40, 90, 255),
            size=size,
        ),
        _make_tile(
            pil_image,
            case["tracker_mask"],
            f"Tracker D={case['tracker_dice']:.3f}",
            color=(255, 120, 0),
            size=size,
        ),
        _make_tile(
            pil_image,
            case["oracle_mask"],
            f"Oracle: {case['oracle_source']}",
            color=(190, 40, 220),
            size=size,
        ),
    ]
    header_h = 54
    gap = 10
    width = len(tiles) * size + (len(tiles) - 1) * gap
    height = header_h + tiles[0].height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{case['volume']} | {case['class_name']} | slice {case['frame_idx']} | "
        f"Δ={case['delta_tracker_minus_detector']:+.3f}"
    )
    draw.text((6, 8), title, fill=(0, 0, 0))
    subtitle = (
        f"det_score={case['detector_score']:.3f}, "
        f"trk_score={case['tracker_score']:.3f}, seed={case['seed_frames']}"
    )
    draw.text((6, 30), subtitle, fill=(70, 70, 70))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, header_h))
        x += size + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _summarize_rows(rows: List[dict]) -> dict:
    if not rows:
        return {}
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class_name"]].append(row)

    def summarise(group):
        detector = np.array([float(r["detector_dice"]) for r in group], dtype=np.float32)
        tracker = np.array([float(r["tracker_dice"]) for r in group], dtype=np.float32)
        oracle = np.array([float(r["oracle_dice"]) for r in group], dtype=np.float32)
        sources = Counter(r["oracle_source"] for r in group)
        return {
            "num_slices": len(group),
            "mean_detector_2d_dice": float(detector.mean()),
            "mean_tracker_2d_dice": float(tracker.mean()),
            "mean_oracle_2d_dice": float(oracle.mean()),
            "mean_oracle_gain_over_detector": float((oracle - detector).mean()),
            "tracker_better_count": int((tracker > detector).sum()),
            "tracker_better_ratio": float((tracker > detector).mean()),
            "oracle_source_counts": dict(sources),
        }

    return {
        "overall": summarise(rows),
        "per_class": {name: summarise(group) for name, group in by_class.items()},
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run detector/tracker/oracle-merge diagnosis with visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_dir", default=_default_test_dir())
    parser.add_argument("--frame_annotation_file", default="frame_annotations.coco.json")
    parser.add_argument("--image_checkpoint_path", default=_default_image_checkpoint())
    parser.add_argument("--tracker_checkpoint_path", default=_default_tracker_checkpoint())
    parser.add_argument("--output_dir", default=_default_output_dir())
    parser.add_argument("--resize_size", type=int, default=1008)
    parser.add_argument("--detector_condition_threshold", type=float, default=0.7)
    parser.add_argument("--detector_output_threshold", type=float, default=0.7)
    parser.add_argument("--tracker_detection_threshold", type=float, default=0.7)
    parser.add_argument("--min_mask_area_px", type=int, default=32)
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.0)
    parser.add_argument("--max_cond_frames", type=int, default=1)
    parser.add_argument("--min_cond_frame_gap", type=int, default=1)
    parser.add_argument(
        "--conditioning_selection_strategy",
        default="earliest_above_threshold",
        choices=["topk_score", "earliest_above_threshold", "best_single", "largest_single"],
    )
    parser.add_argument(
        "--propagation_mode",
        default="forward_only",
        choices=["bidirectional", "forward_only", "reverse_only"],
    )
    parser.add_argument("--fallback_to_best_seed", action="store_true")
    parser.add_argument("--limit_volumes", type=int, default=None)
    parser.add_argument("--max_visualizations", type=int, default=36)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("oracle diagnosis currently requires CUDA.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    frame_json_path = os.path.join(args.test_dir, args.frame_annotation_file)
    frame_coco = _load_json(frame_json_path)
    volume_to_frames = _build_frame_metadata(frame_coco)
    gt_index = _build_gt_index(frame_coco)
    category_id_to_prompt = _category_prompts_from_coco(frame_coco)
    category_id_to_name = dict(category_id_to_prompt)
    volume_names = sorted(volume_to_frames.keys())
    if args.limit_volumes is not None:
        volume_names = volume_names[: args.limit_volumes]

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    print(f"Loading image model: {args.image_checkpoint_path}")
    _, image_processor = load_checkpoint_and_model(
        args.image_checkpoint_path,
        bpe_path,
        args.device,
        args.resize_size,
        confidence_threshold=0.0,
        use_lora=None,
    )

    detector_plan = {}
    seed_report = {}
    for volume_name in tqdm(volume_names, desc="Detector"):
        npz_path = os.path.join(args.test_dir, "volumes", f"{volume_name}.npz")
        frame_meta = volume_to_frames[volume_name]
        pil_slices, _ = load_volume_pil_slices(npz_path=npz_path, frame_meta=frame_meta)
        detector_plan[volume_name] = {}
        seed_report[volume_name] = {}
        for category_id, prompt in category_id_to_prompt.items():
            detector_predictions = collect_detector_predictions_for_category(
                processor=image_processor,
                pil_slices=pil_slices,
                prompt=prompt,
            )
            cond_frames, selection_reason = select_conditioning_frames(
                detector_predictions,
                detector_condition_threshold=args.detector_condition_threshold,
                min_mask_area_px=args.min_mask_area_px,
                min_mask_area_ratio=args.min_mask_area_ratio,
                max_cond_frames=args.max_cond_frames,
                min_cond_frame_gap=args.min_cond_frame_gap,
                fallback_to_best_seed=args.fallback_to_best_seed,
                selection_strategy=args.conditioning_selection_strategy,
            )
            detector_plan[volume_name][category_id] = detector_predictions
            seed_report[volume_name][str(category_id)] = {
                "prompt": prompt,
                "selection_reason": selection_reason,
                "conditioning_frames": [
                    {
                        "frame_idx": int(c["frame_idx"]),
                        "score": float(c["score"]),
                        "area_px": int(c["area_px"]),
                    }
                    for c in cond_frames
                ],
            }

    del image_processor
    torch.cuda.empty_cache()

    print(f"Loading tracker model: {args.tracker_checkpoint_path}")
    tracker_model = load_tracker_predictor(args.tracker_checkpoint_path, args.device)

    detector_predictions_json = []
    tracker_predictions_json = []
    oracle_predictions_json = []
    rows = []
    visual_cases = []

    for volume_name in tqdm(volume_names, desc="Tracker + oracle"):
        npz_path = os.path.join(args.test_dir, "volumes", f"{volume_name}.npz")
        frame_meta = volume_to_frames[volume_name]
        pil_slices, image_batch, _ = prepare_volume_inputs(
            npz_path=npz_path,
            frame_meta=frame_meta,
            device=args.device,
            resolution=args.resize_size,
        )
        cached_features = compute_tracker_cached_features(tracker_model, image_batch)

        video_height = int(frame_meta[0]["height"])
        video_width = int(frame_meta[0]["width"])
        num_frames = len(frame_meta)

        for category_id, prompt in category_id_to_prompt.items():
            detector_predictions = detector_plan[volume_name].get(category_id, [])
            cond_frames, _ = select_conditioning_frames(
                detector_predictions,
                detector_condition_threshold=args.detector_condition_threshold,
                min_mask_area_px=args.min_mask_area_px,
                min_mask_area_ratio=args.min_mask_area_ratio,
                max_cond_frames=args.max_cond_frames,
                min_cond_frame_gap=args.min_cond_frame_gap,
                fallback_to_best_seed=args.fallback_to_best_seed,
                selection_strategy=args.conditioning_selection_strategy,
            )
            seed_frames = [int(c["frame_idx"]) for c in cond_frames]
            tracker_outputs = run_tracker_with_mode(
                tracker_model=tracker_model,
                cached_features=cached_features,
                video_height=video_height,
                video_width=video_width,
                num_frames=num_frames,
                cond_frames=cond_frames,
                propagation_mode=args.propagation_mode,
            )

            for local_frame_idx, meta in enumerate(frame_meta):
                image_id = int(meta["image_id"])
                height = int(meta["height"])
                width = int(meta["width"])
                gt_mask = _decode_gt_mask(
                    gt_index, image_id, category_id, height, width
                )
                detector_pred_raw = _safe_pred(
                    detector_predictions[local_frame_idx]
                    if local_frame_idx < len(detector_predictions)
                    else None,
                    height,
                    width,
                )
                tracker_pred_raw = _safe_pred(
                    tracker_outputs.get(local_frame_idx),
                    height,
                    width,
                )
                detector_pred = _threshold_pred(
                    detector_pred_raw,
                    threshold=args.detector_output_threshold,
                    strict=False,
                )
                tracker_pred = _threshold_pred(
                    tracker_pred_raw,
                    threshold=args.tracker_detection_threshold,
                    strict=True,
                )

                detector_dice = float(compute_dice(detector_pred["mask"], gt_mask))
                tracker_dice = float(compute_dice(tracker_pred["mask"], gt_mask))
                detector_raw_dice = float(compute_dice(detector_pred_raw["mask"], gt_mask))
                tracker_raw_dice = float(compute_dice(tracker_pred_raw["mask"], gt_mask))
                detector_iou = float(compute_iou(detector_pred["mask"], gt_mask))
                tracker_iou = float(compute_iou(tracker_pred["mask"], gt_mask))

                if tracker_dice > detector_dice:
                    oracle_source = "tracker"
                    oracle_mask = tracker_pred["mask"]
                    oracle_score = tracker_pred["score"]
                    oracle_dice = tracker_dice
                    oracle_iou = tracker_iou
                else:
                    oracle_source = "detector"
                    oracle_mask = detector_pred["mask"]
                    oracle_score = detector_pred["score"]
                    oracle_dice = detector_dice
                    oracle_iou = detector_iou

                _append_coco_prediction(
                    detector_predictions_json,
                    image_id=image_id,
                    category_id=category_id,
                    mask=detector_pred["mask"],
                    score=detector_pred["score"],
                )
                _append_coco_prediction(
                    tracker_predictions_json,
                    image_id=image_id,
                    category_id=category_id,
                    mask=tracker_pred["mask"],
                    score=tracker_pred["score"],
                )
                _append_coco_prediction(
                    oracle_predictions_json,
                    image_id=image_id,
                    category_id=category_id,
                    mask=oracle_mask,
                    score=oracle_score,
                )

                row = {
                    "volume": volume_name,
                    "image_id": image_id,
                    "frame_idx": int(meta["frame_idx"]),
                    "original_slice_idx": int(meta["original_slice_idx"]),
                    "category_id": int(category_id),
                    "class_name": category_id_to_name[category_id],
                    "detector_score": float(detector_pred["score"]),
                    "tracker_score": float(tracker_pred["score"]),
                    "detector_raw_score": float(detector_pred_raw["score"]),
                    "tracker_raw_score": float(tracker_pred_raw["score"]),
                    "detector_area_px": int(detector_pred["area_px"]),
                    "tracker_area_px": int(tracker_pred["area_px"]),
                    "detector_raw_area_px": int(detector_pred_raw["area_px"]),
                    "tracker_raw_area_px": int(tracker_pred_raw["area_px"]),
                    "gt_area_px": int(gt_mask.sum()),
                    "detector_dice": detector_dice,
                    "tracker_dice": tracker_dice,
                    "detector_raw_dice": detector_raw_dice,
                    "tracker_raw_dice": tracker_raw_dice,
                    "oracle_dice": float(oracle_dice),
                    "detector_iou": detector_iou,
                    "tracker_iou": tracker_iou,
                    "oracle_iou": float(oracle_iou),
                    "oracle_source": oracle_source,
                    "delta_tracker_minus_detector": tracker_dice - detector_dice,
                    "seed_frames": ",".join(map(str, seed_frames)),
                }
                rows.append(row)

                if gt_mask.any() or detector_pred["mask"].any() or tracker_pred["mask"].any():
                    visual_cases.append(
                        {
                            **row,
                            "gt_mask": gt_mask,
                            "detector_mask": detector_pred["mask"],
                            "tracker_mask": tracker_pred["mask"],
                            "oracle_mask": oracle_mask,
                            "pil_image": pil_slices[local_frame_idx],
                        }
                    )

    for name, predictions in [
        ("detector_predictions_segm.json", detector_predictions_json),
        ("tracker_predictions_segm.json", tracker_predictions_json),
        ("oracle_merge_predictions_segm.json", oracle_predictions_json),
    ]:
        predictions.sort(key=lambda x: (x["image_id"], x["category_id"]))
        with open(output_dir / name, "w") as f:
            json.dump(predictions, f)

    csv_path = output_dir / "slice_oracle_diagnosis.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize_rows(rows)
    summary.update(
        {
            "config": {
                "image_checkpoint_path": args.image_checkpoint_path,
                "tracker_checkpoint_path": args.tracker_checkpoint_path,
                "detector_condition_threshold": args.detector_condition_threshold,
                "detector_output_threshold": args.detector_output_threshold,
                "tracker_detection_threshold": args.tracker_detection_threshold,
                "max_cond_frames": args.max_cond_frames,
                "conditioning_selection_strategy": args.conditioning_selection_strategy,
                "propagation_mode": args.propagation_mode,
            }
        }
    )
    with open(output_dir / "oracle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "seed_selection_report.json", "w") as f:
        json.dump(seed_report, f, indent=2)

    tracker_wins = sorted(
        [c for c in visual_cases if c["delta_tracker_minus_detector"] > 0],
        key=lambda x: x["delta_tracker_minus_detector"],
        reverse=True,
    )
    detector_wins = sorted(
        [c for c in visual_cases if c["delta_tracker_minus_detector"] < 0],
        key=lambda x: x["delta_tracker_minus_detector"],
    )
    n_each = max(1, args.max_visualizations // 2)
    for idx, case in enumerate(tracker_wins[:n_each]):
        _save_case_visualization(
            case,
            case["pil_image"],
            vis_dir / "tracker_better" / f"{idx:03d}_{case['volume']}_{case['class_name'].replace(' ', '_')}_s{case['frame_idx']}.png",
        )
    for idx, case in enumerate(detector_wins[:n_each]):
        _save_case_visualization(
            case,
            case["pil_image"],
            vis_dir / "detector_better" / f"{idx:03d}_{case['volume']}_{case['class_name'].replace(' ', '_')}_s{case['frame_idx']}.png",
        )

    print(f"Done. Output dir: {output_dir}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()

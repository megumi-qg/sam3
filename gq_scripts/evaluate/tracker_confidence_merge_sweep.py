"""
Parameter sweep for a simple confidence-aware detector/tracker merge rule.

The script runs detector and tracker once, then evaluates many deterministic
merge rules:

    S_det = detector_score
    S_trk = tracker_score - lambda * distance_to_seed

When both masks pass their own thresholds, tracker is selected only if
S_trk > S_det + margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
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
        "scribble_sam3_tracker_image_init_v1_confidence_merge_sweep"
    )


def _float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _build_gt_index(frame_coco: dict) -> Dict[tuple[int, int], dict]:
    return {
        (int(ann["image_id"]), int(ann["category_id"])): ann
        for ann in frame_coco.get("annotations", [])
    }


def _decode_gt_mask(gt_index, image_id, category_id, height, width) -> np.ndarray:
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
        mask = np.zeros((height, width), dtype=bool)
        return {"mask": mask, "score": 0.0, "area_px": 0}
    mask = pred["mask"].astype(bool)
    if mask.shape != (height, width):
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), Image.NEAREST
            )
        ) > 0
    return {"mask": mask, "score": float(pred.get("score", 0.0)), "area_px": int(mask.sum())}


def _distance_to_seed(frame_idx: int, seed_frames: List[int]) -> int:
    if not seed_frames:
        return 10**6
    return int(min(abs(int(frame_idx) - int(seed)) for seed in seed_frames))


def _choose_candidate(
    sample: dict,
    *,
    detector_threshold: float,
    tracker_threshold: float,
    distance_penalty: float,
    margin: float,
) -> tuple[str, np.ndarray, float]:
    det_ok = (
        sample["detector_area_px"] > 0
        and sample["detector_score"] >= detector_threshold
    )
    trk_ok = (
        sample["tracker_area_px"] > 0
        and sample["tracker_score"] > tracker_threshold
    )

    if det_ok and trk_ok:
        s_det = sample["detector_score"]
        s_trk = sample["tracker_score"] - distance_penalty * sample["seed_distance"]
        if s_trk > s_det + margin:
            return "tracker", sample["tracker_mask"], sample["tracker_score"]
        return "detector", sample["detector_mask"], sample["detector_score"]
    if trk_ok:
        return "tracker", sample["tracker_mask"], sample["tracker_score"]
    if det_ok:
        return "detector", sample["detector_mask"], sample["detector_score"]
    return "none", np.zeros_like(sample["gt_mask"], dtype=bool), 0.0


def _evaluate_samples(samples: List[dict], config: dict) -> dict:
    grouped = defaultdict(list)
    source_counts = Counter()
    for sample in samples:
        source, mask, _ = _choose_candidate(sample, **config)
        source_counts[source] += 1
        grouped[(sample["volume"], sample["category_id"], sample["class_name"])].append(
            (sample["frame_idx"], mask, sample["gt_mask"])
        )

    per_class = defaultdict(lambda: {"dice": [], "iou": []})
    for (_, _, class_name), items in grouped.items():
        items.sort(key=lambda x: x[0])
        pred_3d = np.stack([item[1] for item in items], axis=0)
        gt_3d = np.stack([item[2] for item in items], axis=0)
        per_class[class_name]["dice"].append(float(compute_dice(pred_3d, gt_3d)))
        per_class[class_name]["iou"].append(float(compute_iou(pred_3d, gt_3d)))

    class_rows = {}
    class_dice_means = []
    class_iou_means = []
    for class_name, values in per_class.items():
        dice_values = np.array(values["dice"], dtype=np.float32)
        iou_values = np.array(values["iou"], dtype=np.float32)
        class_rows[class_name] = {
            "dice_mean": float(dice_values.mean()),
            "dice_std": float(dice_values.std()),
            "iou_mean": float(iou_values.mean()),
            "iou_std": float(iou_values.std()),
        }
        class_dice_means.append(class_rows[class_name]["dice_mean"])
        class_iou_means.append(class_rows[class_name]["iou_mean"])

    return {
        "overall_dice": float(np.mean(class_dice_means)),
        "overall_iou": float(np.mean(class_iou_means)),
        "source_counts": dict(source_counts),
        "per_class": class_rows,
    }


def _write_predictions(samples: List[dict], config: dict, path: Path) -> None:
    predictions = []
    for sample in samples:
        source, mask, score = _choose_candidate(sample, **config)
        if source == "none" or not mask.any():
            continue
        predictions.append(
            mask_to_coco_result(
                image_id=sample["image_id"],
                category_id=sample["category_id"],
                mask=mask,
                score=score,
            )
        )
    predictions.sort(key=lambda x: (x["image_id"], x["category_id"]))
    with open(path, "w") as f:
        json.dump(predictions, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep simple confidence-aware tracker merge parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test_dir", default=_default_test_dir())
    parser.add_argument("--frame_annotation_file", default="frame_annotations.coco.json")
    parser.add_argument("--image_checkpoint_path", default=_default_image_checkpoint())
    parser.add_argument("--tracker_checkpoint_path", default=_default_tracker_checkpoint())
    parser.add_argument("--output_dir", default=_default_output_dir())
    parser.add_argument("--resize_size", type=int, default=1008)
    parser.add_argument("--detector_condition_threshold", type=float, default=0.7)
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
    parser.add_argument("--detector_thresholds", default="0.6,0.7,0.8")
    parser.add_argument("--tracker_thresholds", default="0.6,0.7,0.8")
    parser.add_argument("--distance_penalties", default="0,0.01,0.02,0.05")
    parser.add_argument("--margins", default="0,0.03,0.05,0.1")
    parser.add_argument("--limit_volumes", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("confidence merge sweep currently requires CUDA.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_json_path = os.path.join(args.test_dir, args.frame_annotation_file)
    frame_coco = _load_json(frame_json_path)
    volume_to_frames = _build_frame_metadata(frame_coco)
    gt_index = _build_gt_index(frame_coco)
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
    category_id_to_prompt = _category_prompts_from_coco(frame_coco)
    category_id_to_name = dict(category_id_to_prompt)

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
                    {"frame_idx": int(c["frame_idx"]), "score": float(c["score"])}
                    for c in cond_frames
                ],
            }

    del image_processor
    torch.cuda.empty_cache()

    print(f"Loading tracker model: {args.tracker_checkpoint_path}")
    tracker_model = load_tracker_predictor(args.tracker_checkpoint_path, args.device)

    samples = []
    for volume_name in tqdm(volume_names, desc="Tracker candidates"):
        npz_path = os.path.join(args.test_dir, "volumes", f"{volume_name}.npz")
        frame_meta = volume_to_frames[volume_name]
        _, image_batch, _ = prepare_volume_inputs(
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
            detector_predictions = detector_plan[volume_name][category_id]
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
                det = _safe_pred(detector_predictions[local_frame_idx], height, width)
                trk = _safe_pred(tracker_outputs.get(local_frame_idx), height, width)
                samples.append(
                    {
                        "volume": volume_name,
                        "image_id": image_id,
                        "frame_idx": int(meta["frame_idx"]),
                        "category_id": int(category_id),
                        "class_name": category_id_to_name[category_id],
                        "gt_mask": _decode_gt_mask(
                            gt_index, image_id, category_id, height, width
                        ),
                        "detector_mask": det["mask"],
                        "detector_score": det["score"],
                        "detector_area_px": det["area_px"],
                        "tracker_mask": trk["mask"],
                        "tracker_score": trk["score"],
                        "tracker_area_px": trk["area_px"],
                        "seed_distance": _distance_to_seed(local_frame_idx, seed_frames),
                    }
                )

    sweep_rows = []
    best = None
    for detector_threshold in _float_list(args.detector_thresholds):
        for tracker_threshold in _float_list(args.tracker_thresholds):
            for distance_penalty in _float_list(args.distance_penalties):
                for margin in _float_list(args.margins):
                    config = {
                        "detector_threshold": detector_threshold,
                        "tracker_threshold": tracker_threshold,
                        "distance_penalty": distance_penalty,
                        "margin": margin,
                    }
                    metrics = _evaluate_samples(samples, config)
                    row = {
                        **config,
                        "overall_dice": metrics["overall_dice"],
                        "overall_iou": metrics["overall_iou"],
                        "num_detector": metrics["source_counts"].get("detector", 0),
                        "num_tracker": metrics["source_counts"].get("tracker", 0),
                        "num_none": metrics["source_counts"].get("none", 0),
                    }
                    for class_name, vals in metrics["per_class"].items():
                        key = class_name.replace(" ", "_")
                        row[f"{key}_dice"] = vals["dice_mean"]
                    sweep_rows.append(row)
                    if best is None or metrics["overall_dice"] > best["metrics"]["overall_dice"]:
                        best = {"config": config, "metrics": metrics, "row": row}

    sweep_rows.sort(key=lambda x: x["overall_dice"], reverse=True)
    csv_path = output_dir / "confidence_merge_sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best, f, indent=2)
    with open(output_dir / "seed_selection_report.json", "w") as f:
        json.dump(seed_report, f, indent=2)
    _write_predictions(
        samples,
        best["config"],
        output_dir / "best_confidence_merge_predictions_segm.json",
    )

    print(f"Done. Output dir: {output_dir}")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run CMPB inference confidence-threshold sweep for SAM3-Scribble and SAM3-Full."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path("/home/gaoqi/sam3")
PYTHON = sys.executable


@dataclass(frozen=True)
class ModelSpec:
    name: str
    exp_dir: Path
    checkpoint: Path
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: float = 16.0


MODELS = [
    ModelSpec(
        "SAM3-Scribble",
        REPO / "gq_experiment/cmpb/weak_lora_acdc_btcv_promise12",
        REPO / "gq_experiment/cmpb/weak_lora_acdc_btcv_promise12/checkpoints/val_mean_segmentation_coco_eval_segm_Dice.pt",
    ),
    ModelSpec(
        "SAM3-Full",
        REPO / "gq_experiment/cmpb/full_lora_acdc_btcv_promise12",
        REPO / "gq_experiment/cmpb/full_lora_acdc_btcv_promise12/checkpoints/val_mean_segmentation_coco_eval_segm_Dice.pt",
    ),
]


DATASETS = [
    {
        "key": "acdc",
        "name": "ACDC",
        "test_dir": "/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100_cmpb_clean/test",
        "spacing_file": "/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100_cmpb_clean/test/spacing_map.json",
    },
    {
        "key": "btcv",
        "name": "BTCV",
        "test_dir": "/home/gaoqi/dataset/using/btcv/processed/png_coco_sam3_slices_cmpb_clean/test",
        "spacing_file": "/home/gaoqi/dataset/using/btcv/processed/png_coco_sam3_slices_cmpb_clean/_nifti_splits/test/spacing_map.json",
    },
    {
        "key": "promise12",
        "name": "Promise12",
        "test_dir": "/home/gaoqi/dataset/using/promise12/processed/png_coco_sam3_cmpb_clean/test",
        "spacing_file": "/home/gaoqi/dataset/using/promise12/processed/png_coco_sam3_cmpb_clean/_nifti_splits/test/spacing_map.json",
    },
]


def log(msg: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def query_gpus() -> list[tuple[int, int]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in out.strip().splitlines():
        if line.strip():
            idx_s, used_s = [x.strip() for x in line.split(",")]
            gpus.append((int(idx_s), int(used_s)))
    return gpus


def choose_gpu(max_used_mb: int, allowed: set[int] | None) -> int | None:
    try:
        gpus = query_gpus()
    except Exception as exc:
        log(f"nvidia-smi failed: {exc}")
        return None
    candidates = [
        (idx, used)
        for idx, used in gpus
        if used <= max_used_mb and (allowed is None or idx in allowed)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def run(cmd: list[str], log_path: Path, gpu: int | None = None) -> int:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + " ".join(cmd) + "\n")
        if gpu is not None:
            f.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
        return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).wait()


def paths(model: ModelSpec, dataset: dict, threshold: float) -> tuple[Path, Path, Path]:
    tag = str(threshold)
    out_dir = model.exp_dir / f"threshold_sweep_{dataset['key']}_{tag}"
    pred = out_dir / "predictions.pkl"
    eval_json = out_dir / f"evaluation_results_{dataset['key']}.json"
    return out_dir, pred, eval_json


def run_sweep(args) -> None:
    allowed = set(args.gpu_ids) if args.gpu_ids else None
    for model in MODELS:
        if not model.checkpoint.is_file():
            log(f"SKIP {model.name}: missing checkpoint {model.checkpoint}")
            continue
        for threshold in args.thresholds:
            for dataset in DATASETS:
                out_dir, pred_path, eval_path = paths(model, dataset, threshold)
                if eval_path.is_file() and not args.force:
                    log(f"EXISTS {model.name} {dataset['key']} th={threshold}: {eval_path}")
                    continue

                if not pred_path.is_file() or args.force:
                    gpu = None
                    while gpu is None:
                        gpu = choose_gpu(args.max_used_mb, allowed)
                        if gpu is None:
                            log(f"Waiting for free GPU for {model.name} {dataset['key']} th={threshold}")
                            time.sleep(args.poll_sec)
                    cmd = [
                        PYTHON,
                        "gq_scripts/evaluate/batch_inference.py",
                        "--test_dir",
                        dataset["test_dir"],
                        "--checkpoint_path",
                        str(model.checkpoint),
                        "--confidence_threshold",
                        str(threshold),
                        "--output_dir",
                        str(out_dir),
                        "--lora_r",
                        str(model.lora_r),
                        "--lora_alpha",
                        str(model.lora_alpha),
                    ]
                    log(f"INFER {model.name} {dataset['key']} threshold={threshold} on GPU {gpu}")
                    ret = run(cmd, model.exp_dir / f"threshold_sweep_infer_{dataset['key']}_{threshold}.log", gpu)
                    if ret != 0:
                        log(f"INFER failed: {model.name} {dataset['key']} threshold={threshold}")
                        continue

                cmd = [
                    PYTHON,
                    "gq_scripts/evaluate/batch_evaluate.py",
                    "--predictions_file",
                    str(pred_path),
                    "--test_dir",
                    dataset["test_dir"],
                    "--dataset_name",
                    dataset["name"],
                    "--spacing_file",
                    dataset["spacing_file"],
                    "--output_dir",
                    str(out_dir),
                ]
                log(f"EVAL {model.name} {dataset['key']} threshold={threshold}")
                ret = run(cmd, model.exp_dir / f"threshold_sweep_eval_{dataset['key']}_{threshold}.log")
                if ret != 0:
                    log(f"EVAL failed: {model.name} {dataset['key']} threshold={threshold}")


def write_summary(thresholds: list[float]) -> None:
    rows = []
    for model in MODELS:
        for threshold in thresholds:
            vals = []
            for dataset in DATASETS:
                _out, _pred, eval_path = paths(model, dataset, threshold)
                if not eval_path.is_file():
                    continue
                data = json.loads(eval_path.read_text())
                o = data["overall"]
                rows.append({
                    "model": model.name,
                    "threshold": threshold,
                    "dataset": dataset["key"],
                    "iou_mean": o["iou_mean"],
                    "dice_mean": o["dice_mean"],
                    "hd95_mean": o["hd95_mean"],
                    "nsd_mean": o["nsd_mean"],
                })
                vals.append(o)
            if len(vals) == len(DATASETS):
                rows.append({
                    "model": model.name,
                    "threshold": threshold,
                    "dataset": "macro_avg",
                    "iou_mean": sum(v["iou_mean"] for v in vals) / len(vals),
                    "dice_mean": sum(v["dice_mean"] for v in vals) / len(vals),
                    "hd95_mean": sum(v["hd95_mean"] for v in vals) / len(vals),
                    "nsd_mean": sum(v["nsd_mean"] for v in vals) / len(vals),
                })

    out_csv = REPO / "gq_paper/cmpb/results_summary/threshold_sweep_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        log("No rows found; summary not written.")
        return
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log(f"Summary written: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("--max_used_mb", type=int, default=2000)
    parser.add_argument("--poll_sec", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_sweep(args)
    write_summary(args.thresholds)


if __name__ == "__main__":
    main()

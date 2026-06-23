#!/usr/bin/env python3
"""
Run CMPB ablation training and 3D evaluation as a resumable queue.

The script:
1. Checks GPU memory with nvidia-smi.
2. Launches each missing training job on two currently idle GPUs.
3. Waits for all training jobs to finish.
4. Runs ACDC/BTCV/PROMISE12 batch inference and 3D evaluation for each finished
   experiment.
5. Writes a compact CSV summary.

It is safe to re-run: existing Dice-best checkpoints, predictions, and
evaluation JSON files are skipped unless --force_eval is used.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO = Path("/home/gaoqi/sam3")
PYTHON = sys.executable


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    config: str
    exp_dir: Path
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_targets: Optional[str] = None


EXPERIMENTS = [
    Experiment(
        "R2_reduced_box_giou_loss",
        "configs/final/weak_lora_reg_r2_reduced_geo_loss.yaml",
        REPO / "gq_experiment/cmpb/ablation_reg_r2_reduced_geo_loss",
    ),
    Experiment(
        "L1_rank_r4",
        "configs/final/weak_lora_rank_l1_r4.yaml",
        REPO / "gq_experiment/cmpb/ablation_lora_rank_l1_r4",
        lora_r=4,
        lora_alpha=8.0,
    ),
    Experiment(
        "L2_rank_r16",
        "configs/final/weak_lora_rank_l2_r16.yaml",
        REPO / "gq_experiment/cmpb/ablation_lora_rank_l2_r16",
        lora_r=16,
        lora_alpha=32.0,
    ),
    Experiment(
        "S1_vision_only",
        "configs/final/weak_lora_scope_s1_vision_only.yaml",
        REPO / "gq_experiment/cmpb/ablation_lora_scope_s1_vision_only",
        lora_targets="vision_encoder",
    ),
    Experiment(
        "S2_detr_only",
        "configs/final/weak_lora_scope_s2_detr_only.yaml",
        REPO / "gq_experiment/cmpb/ablation_lora_scope_s2_detr_only",
        lora_targets="detr_encoder,detr_decoder",
    ),
    Experiment(
        "S3_heads_only",
        "configs/final/weak_lora_scope_s3_heads_only.yaml",
        REPO / "gq_experiment/cmpb/ablation_lora_scope_s3_heads_only",
        lora_targets="none",
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


def checkpoint_path(exp: Experiment) -> Path:
    return exp.exp_dir / "checkpoints/val_mean_segmentation_coco_eval_segm_Dice.pt"


def query_gpus() -> list[tuple[int, int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    gpus = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx_s, total_s, used_s = [x.strip() for x in line.split(",")]
        gpus.append((int(idx_s), int(total_s), int(used_s)))
    return gpus


def choose_gpus(
    count: int,
    max_used_mb: int,
    reserved: set[int],
    allowed_gpu_ids: Optional[set[int]] = None,
) -> Optional[list[int]]:
    try:
        gpus = query_gpus()
    except Exception as exc:
        log(f"nvidia-smi failed: {exc}")
        return None
    free = [
        (idx, used)
        for idx, _total, used in gpus
        if idx not in reserved
        and used <= max_used_mb
        and (allowed_gpu_ids is None or idx in allowed_gpu_ids)
    ]
    free.sort(key=lambda x: x[1])
    if len(free) < count:
        return None
    return [idx for idx, _used in free[:count]]


def run_subprocess(cmd: list[str], log_path: Path, cuda_gpus: list[int]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cuda_gpus)
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("a", buffering=1) as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + " ".join(cmd) + "\n")
        f.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        return proc.wait()


def launch_training(exp: Experiment, gpus: list[int]) -> subprocess.Popen:
    exp.exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp.exp_dir / "auto_train.log"
    cmd = [
        PYTHON,
        "sam3/train/train.py",
        "-c",
        exp.config,
        "--use-cluster",
        "0",
        "--num-gpus",
        str(len(gpus)),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env.setdefault("PYTHONUNBUFFERED", "1")
    f = log_path.open("a", buffering=1)
    f.write("\n" + "=" * 100 + "\n")
    f.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + " ".join(cmd) + "\n")
    f.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
    f.flush()
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    proc._log_file = f  # keep file alive until process exits
    return proc


def train_missing(args) -> list[Experiment]:
    pending = [e for e in EXPERIMENTS if args.only is None or e.exp_id in args.only]
    if args.skip_train:
        return pending

    pending = [e for e in pending if not checkpoint_path(e).is_file()]
    if not pending:
        log("All selected training checkpoints already exist.")
        return [e for e in EXPERIMENTS if args.only is None or e.exp_id in args.only]

    running: dict[Experiment, tuple[subprocess.Popen, list[int]]] = {}
    log(f"Training queue: {', '.join(e.exp_id for e in pending)}")
    while pending or running:
        finished = []
        for exp, (proc, gpus) in running.items():
            ret = proc.poll()
            if ret is None:
                continue
            proc._log_file.close()
            finished.append(exp)
            status = "finished" if ret == 0 else f"failed with code {ret}"
            log(f"TRAIN {exp.exp_id} {status} on GPUs {gpus}")
        for exp in finished:
            running.pop(exp)

        reserved = {g for _exp, (_proc, gs) in running.items() for g in gs}
        while pending and len(running) < args.max_train_jobs:
            gpus = choose_gpus(2, args.max_used_mb, reserved, args.allowed_gpu_ids)
            if gpus is None:
                break
            exp = pending.pop(0)
            proc = launch_training(exp, gpus)
            running[exp] = (proc, gpus)
            reserved.update(gpus)
            log(f"TRAIN started {exp.exp_id} on GPUs {gpus}; log={exp.exp_dir / 'auto_train.log'}")
            time.sleep(args.launch_gap_sec)

        if pending or running:
            time.sleep(args.poll_sec)

    return [e for e in EXPERIMENTS if args.only is None or e.exp_id in args.only]


def inference_args(exp: Experiment, dataset: dict, threshold: float) -> tuple[Path, Path, Path]:
    out_dir = exp.exp_dir / f"inference_{dataset['key']}_{threshold}"
    pred_path = out_dir / "predictions.pkl"
    eval_path = out_dir / f"evaluation_results_{dataset['key']}.json"
    return out_dir, pred_path, eval_path


def evaluate_experiment(exp: Experiment, args) -> None:
    ckpt = checkpoint_path(exp)
    if not ckpt.is_file():
        log(f"SKIP eval {exp.exp_id}: missing checkpoint {ckpt}")
        return
    for dataset in DATASETS:
        out_dir, pred_path, eval_path = inference_args(exp, dataset, args.threshold)
        if eval_path.is_file() and not args.force_eval:
            log(f"EVAL exists {exp.exp_id} {dataset['key']}: {eval_path}")
            continue

        if not pred_path.is_file() or args.force_eval:
            gpus = None
            while gpus is None:
                gpus = choose_gpus(1, args.max_used_mb, reserved=set(), allowed_gpu_ids=args.allowed_gpu_ids)
                if gpus is None:
                    log(f"Waiting for one free GPU before inference {exp.exp_id} {dataset['key']}")
                    time.sleep(args.poll_sec)
            cmd = [
                PYTHON,
                "gq_scripts/evaluate/batch_inference.py",
                "--test_dir",
                dataset["test_dir"],
                "--checkpoint_path",
                str(ckpt),
                "--confidence_threshold",
                str(args.threshold),
                "--output_dir",
                str(out_dir),
                "--lora_r",
                str(exp.lora_r),
                "--lora_alpha",
                str(exp.lora_alpha),
            ]
            if exp.lora_targets is not None:
                cmd += ["--lora_target_components", exp.lora_targets]
            log(f"INFER {exp.exp_id} {dataset['key']} on GPU {gpus[0]}")
            ret = run_subprocess(cmd, exp.exp_dir / f"auto_infer_{dataset['key']}.log", gpus)
            if ret != 0:
                log(f"INFER failed {exp.exp_id} {dataset['key']} with code {ret}")
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
        log(f"EVAL {exp.exp_id} {dataset['key']}")
        ret = run_subprocess(cmd, exp.exp_dir / f"auto_eval_{dataset['key']}.log", [])
        if ret != 0:
            log(f"EVAL failed {exp.exp_id} {dataset['key']} with code {ret}")


def write_summary(experiments: list[Experiment], threshold: float) -> None:
    out_csv = REPO / "gq_paper/cmpb/results_summary/ablation_auto_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp in experiments:
        vals = []
        for dataset in DATASETS:
            _out_dir, _pred_path, eval_path = inference_args(exp, dataset, threshold)
            if not eval_path.is_file():
                continue
            data = json.loads(eval_path.read_text())
            overall = data["overall"]
            rows.append({
                "experiment": exp.exp_id,
                "dataset": dataset["key"],
                "iou_mean": overall["iou_mean"],
                "iou_std": overall["iou_std"],
                "dice_mean": overall["dice_mean"],
                "dice_std": overall["dice_std"],
                "hd95_mean": overall["hd95_mean"],
                "hd95_std": overall["hd95_std"],
                "nsd_mean": overall["nsd_mean"],
                "nsd_std": overall["nsd_std"],
            })
            vals.append(overall)
        if len(vals) == len(DATASETS):
            rows.append({
                "experiment": exp.exp_id,
                "dataset": "macro_avg",
                "iou_mean": sum(v["iou_mean"] for v in vals) / len(vals),
                "iou_std": "",
                "dice_mean": sum(v["dice_mean"] for v in vals) / len(vals),
                "dice_std": "",
                "hd95_mean": sum(v["hd95_mean"] for v in vals) / len(vals),
                "hd95_std": "",
                "nsd_mean": sum(v["nsd_mean"] for v in vals) / len(vals),
                "nsd_std": "",
            })
    if not rows:
        log("No evaluation JSON files found; summary not written.")
        return
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log(f"Summary written: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_used_mb", type=int, default=2000)
    parser.add_argument("--poll_sec", type=int, default=300)
    parser.add_argument("--launch_gap_sec", type=int, default=30)
    parser.add_argument("--max_train_jobs", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None, help="Optional candidate GPU ids.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Run only selected experiment IDs.")
    args = parser.parse_args()
    args.allowed_gpu_ids = set(args.gpu_ids) if args.gpu_ids else None

    selected = train_missing(args)
    if not args.skip_eval:
        for exp in selected:
            evaluate_experiment(exp, args)
    write_summary(EXPERIMENTS, args.threshold)


if __name__ == "__main__":
    main()

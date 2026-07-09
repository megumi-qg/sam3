#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_PYTHON="${CONDA_PYTHON:-/home/gaoqi/anaconda3/envs/sam3/bin/python}"

MSCMR_DETECTOR_CKPT="${MSCMR_DETECTOR_CKPT:-$REPO_ROOT/gq_experiment/mscmr/expert_scribble_lora/checkpoints/val_mscmr_segmentation_coco_eval_segm_AP.pt}"
ISBI_DETECTOR_CKPT="${ISBI_DETECTOR_CKPT:-$REPO_ROOT/gq_experiment/isbi/scribble_lora/checkpoints/val_isbi_segmentation_coco_eval_segm_AP.pt}"

"$CONDA_PYTHON" gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py \
  --split_dir /home/gaoqi/dataset/using/mscmr/processed/sam3_video_npz_coco_fullframes/train \
  --scribble_video_json /home/gaoqi/dataset/using/mscmr/processed/sam3_video_npz_coco_fullframes/train/scribble_video_annotations.coco.json \
  --checkpoint_path "$MSCMR_DETECTOR_CKPT" \
  --output_json "$REPO_ROOT/gq_experiment/mscmr/expert_scribble_lora/pseudo_seed_bank/scribble_pseudo_seed_video_annotations.coco.json" \
  --score_threshold 0.97 \
  --min_scribble_recall 0.8

"$CONDA_PYTHON" gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py \
  --split_dir /home/gaoqi/dataset/using/isbi/processed/sam3_video_npz_coco_fullframes_train_val_test/train \
  --scribble_video_json /home/gaoqi/dataset/using/isbi/processed/sam3_video_npz_coco_fullframes_train_val_test/train/scribble_video_annotations.coco.json \
  --checkpoint_path "$ISBI_DETECTOR_CKPT" \
  --output_json "$REPO_ROOT/gq_experiment/isbi/scribble_lora/pseudo_seed_bank/scribble_pseudo_seed_video_annotations.coco.json" \
  --score_threshold 0.0 \
  --min_scribble_recall 0.1

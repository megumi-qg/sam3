#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DETECTOR_CKPT="${DETECTOR_CKPT:-$REPO_ROOT/gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/checkpoints/val_macro_segmentation_coco_eval_segm_AP.pt}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/pseudo_seed_bank_val_macro}"
CONDA_PYTHON="${CONDA_PYTHON:-/home/gaoqi/anaconda3/envs/sam3/bin/python}"

mkdir -p "$OUT_ROOT"/{acdc,mscmr,isbi}

"$CONDA_PYTHON" gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py \
  --split_dir /home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/train \
  --scribble_video_json /home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/train/scribble_tmi_video_annotations.coco.json \
  --checkpoint_path "$DETECTOR_CKPT" \
  --output_json "$OUT_ROOT/acdc/scribble_tmi_pseudo_seed_video_annotations.coco.json" \
  --score_threshold 0.97 \
  --min_scribble_recall 0.8

"$CONDA_PYTHON" gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py \
  --split_dir /home/gaoqi/dataset/using/mscmr/processed/sam3_video_npz_coco_fullframes/train \
  --scribble_video_json /home/gaoqi/dataset/using/mscmr/processed/sam3_video_npz_coco_fullframes/train/scribble_video_annotations.coco.json \
  --checkpoint_path "$DETECTOR_CKPT" \
  --output_json "$OUT_ROOT/mscmr/scribble_pseudo_seed_video_annotations.coco.json" \
  --score_threshold 0.97 \
  --min_scribble_recall 0.8

# ISBI detector scores are lower calibrated than ACDC/MSCMR. Keep the weak
# supervision closed-loop by filtering with scribble consistency, but do not
# hard-gate on score.
"$CONDA_PYTHON" gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py \
  --split_dir /home/gaoqi/dataset/using/isbi/processed/sam3_video_npz_coco_fullframes_train_val_test/train \
  --scribble_video_json /home/gaoqi/dataset/using/isbi/processed/sam3_video_npz_coco_fullframes_train_val_test/train/scribble_video_annotations.coco.json \
  --checkpoint_path "$DETECTOR_CKPT" \
  --output_json "$OUT_ROOT/isbi/scribble_pseudo_seed_video_annotations.coco.json" \
  --score_threshold 0.0 \
  --min_scribble_recall 0.1

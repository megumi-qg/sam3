#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/gq_experiment/joint/tracker_eval/checkpoint_val_dice_sweep}"
IMAGE_CHECKPOINT_PATH="${IMAGE_CHECKPOINT_PATH:-$REPO_ROOT/gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/checkpoints/checkpoint_20.pt}"
TRACKER_CKPT_DIR="${TRACKER_CKPT_DIR:-$REPO_ROOT/gq_experiment/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1/checkpoints}"
CKPT_NAMES="${CKPT_NAMES:-checkpoint_5.pt checkpoint_10.pt checkpoint_15.pt checkpoint_20.pt checkpoint_25.pt checkpoint_30.pt checkpoint_35.pt checkpoint_40.pt checkpoint_45.pt checkpoint_50.pt val_macro_segmentation_coco_eval_segm_AP.pt}"

declare -A DATASET_DIRS=(
  [acdc]="/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/val"
  [mscmr]="/home/gaoqi/dataset/using/mscmr/processed/sam3_video_npz_coco_fullframes/val"
  [isbi]="/home/gaoqi/dataset/using/isbi/processed/sam3_video_npz_coco_fullframes_train_val_test/val"
)

declare -A DATASET_NAMES=(
  [acdc]="ACDC"
  [mscmr]="MSCMR"
  [isbi]="ISBI"
)

mkdir -p "$OUT_ROOT"

for ckpt_name in $CKPT_NAMES; do
  ckpt_path="$TRACKER_CKPT_DIR/$ckpt_name"
  ckpt_tag="${ckpt_name%.pt}"
  if [[ ! -f "$ckpt_path" ]]; then
    echo "[missing] $ckpt_path" >&2
    exit 1
  fi

  for dataset in acdc mscmr isbi; do
    test_dir="${DATASET_DIRS[$dataset]}"
    dataset_name="${DATASET_NAMES[$dataset]}"
    out_dir="$OUT_ROOT/$ckpt_tag/$dataset"
    eval_dir="$out_dir/tracker_only_eval_3d"
    eval_json="$eval_dir/evaluation_results_${dataset}.json"
    log_file="$out_dir/run.log"
    mkdir -p "$out_dir" "$eval_dir"

    if [[ -f "$eval_json" ]]; then
      echo "[skip] $ckpt_tag $dataset"
      continue
    fi

    echo "[run] gpu=$GPU ckpt=$ckpt_tag dataset=$dataset"
    CUDA_VISIBLE_DEVICES="$GPU" /home/gaoqi/anaconda3/envs/sam3/bin/python \
      gq_scripts/evaluate/tracker_auto_seed_inference.py \
      --test_dir "$test_dir" \
      --image_checkpoint_path "$IMAGE_CHECKPOINT_PATH" \
      --tracker_checkpoint_path "$ckpt_path" \
      --output_dir "$out_dir" \
      --detector_condition_threshold 0.7 \
      --detector_output_threshold 0.7 \
      --tracker_detection_threshold 0.7 \
      --final_output_mode detector_first \
      --min_mask_area_px 32 \
      --min_mask_area_ratio 0.0 \
      --max_cond_frames 4 \
      --min_cond_frame_gap 1 \
      --conditioning_selection_strategy topk_score \
      --propagation_mode bidirectional \
      > "$log_file" 2>&1

    /home/gaoqi/anaconda3/envs/sam3/bin/python \
      gq_scripts/evaluate/evaluate_tracker_coco_predictions.py \
      --predictions_json "$out_dir/tracker_predictions_segm.json" \
      --test_dir "$test_dir" \
      --dataset_name "$dataset_name" \
      --annotation_file frame_annotations.coco.json \
      --output_dir "$eval_dir" \
      >> "$log_file" 2>&1
  done
done

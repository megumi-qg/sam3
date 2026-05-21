#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

EXP_DIR="${EXP_DIR:-$REPO_ROOT/gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced}"
CKPT_DIR="${CKPT_DIR:-$EXP_DIR/checkpoints}"
OUT_ROOT="${OUT_ROOT:-$EXP_DIR/dice_sweep_test_two_ckpts}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.7}"

declare -a CKPTS=(
  "val_macro_segmentation_coco_eval_segm_AP.pt"
  "checkpoint_20.pt"
)

declare -A DATASET_DIRS=(
  [acdc]="/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100/test"
  [mscmr]="/home/gaoqi/dataset/using/mscmr/processed/png_coco_sam3_fullframes/test"
  [isbi]="/home/gaoqi/dataset/using/isbi/processed/png_coco_sam3_fullframes_train_val_test/test"
)

declare -A ANN_FILES=(
  [acdc]="image_annotations.coco.json"
  [mscmr]="full_annotations.coco.json"
  [isbi]="image_annotations.coco.json"
)

mkdir -p "$OUT_ROOT"

for ckpt_name in "${CKPTS[@]}"; do
  ckpt_path="$CKPT_DIR/$ckpt_name"
  ckpt_tag="${ckpt_name%.pt}"
  for dataset in acdc mscmr isbi; do
    test_dir="${DATASET_DIRS[$dataset]}"
    ann_file="${ANN_FILES[$dataset]}"
    out_dir="$OUT_ROOT/$ckpt_tag/$dataset"
    eval_json="$out_dir/evaluation_results_${dataset}.json"
    if [[ -f "$eval_json" ]]; then
      echo "[skip] exists: $eval_json"
      continue
    fi

    echo "[run] ckpt=$ckpt_name dataset=$dataset"
    python gq_scripts/evaluate/batch_inference.py \
      --test_dir "$test_dir" \
      --annotation_file "$ann_file" \
      --checkpoint_path "$ckpt_path" \
      --output_dir "$out_dir" \
      --confidence_threshold "$CONFIDENCE_THRESHOLD"
    python gq_scripts/evaluate/batch_evaluate.py \
      --predictions_file "$out_dir/predictions.pkl" \
      --test_dir "$test_dir" \
      --annotation_file "$ann_file" \
      --dataset_name "$dataset" \
      --output_dir "$out_dir"
  done
done

python gq_scripts/evaluate/summarize_joint_dice_sweep.py "$OUT_ROOT"

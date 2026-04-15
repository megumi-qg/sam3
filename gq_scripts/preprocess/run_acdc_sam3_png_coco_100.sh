#!/usr/bin/env bash
# 基于 nifti_split_img_gt_scribble_100 的 train/val/test，生成 SAM3 用 PNG+COCO。
# full：img + gt → preprocess_full_annotations.py
# weak：img + acdc_scribbles_TMI → preprocess_scribble_annotations.py
set -euo pipefail

NIFTI_ROOT="${NIFTI_ROOT:-/home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_100}"
OUT="${OUT:-/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL="${SCRIPT_DIR}/preprocess_full_annotations.py"
SCR="${SCRIPT_DIR}/preprocess_scribble_annotations.py"

for split in train val test; do
  if [[ "$split" == "test" ]]; then
    COCO_NAME="image_annotations.coco.json"
  else
    COCO_NAME="full_annotations.coco.json"
  fi
  echo "=== full_annotations [$split] -> $COCO_NAME ==="
  python3 "$FULL" \
    --dataset acdc --modality mri \
    --img_dir "$NIFTI_ROOT/$split/img" \
    --gt_dir "$NIFTI_ROOT/$split/gt" \
    --output_dir "$OUT/$split" \
    --slice_policy all \
    --coco_json_name "$COCO_NAME"
done

for split in train val test; do
  echo "=== scribble_tmi [$split] ==="
  python3 "$SCR" \
    --dataset acdc \
    --input_img_dir "$NIFTI_ROOT/$split/img" \
    --input_scribble_dir "$NIFTI_ROOT/$split/acdc_scribbles_TMI" \
    --output_json_path "$OUT/$split/scribble_tmi_annotations.coco.json" \
    --slice_policy all
done

echo "Done. Output: $OUT"

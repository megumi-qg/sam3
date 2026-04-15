#!/usr/bin/env bash
# 基于 nifti_split_img_gt_scribble_100 的 train/val/test，
# 生成适配当前 SAM3 VideoGroundingDataset 的 NPZ + video COCO 数据。
set -euo pipefail

NIFTI_ROOT="${NIFTI_ROOT:-/home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_100}"
OUT="${OUT:-/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="${SCRIPT_DIR}/preprocess_video_annotations.py"

for split in train val test; do
  echo "=== video_annotations [$split] ==="
  python3 "$VIDEO" \
    --dataset acdc --modality mri \
    --img_dir "$NIFTI_ROOT/$split/img" \
    --gt_dir "$NIFTI_ROOT/$split/gt" \
    --output_dir "$OUT/$split" \
    --slice_policy all
done

echo "Done. Output: $OUT"

#!/usr/bin/env bash
# 基于现有 sam3_video_npz_coco_fullframes_100/{split}/volumes，
# 生成 ACDC 的 3D scribble video 弱监督 JSON。
set -euo pipefail

NIFTI_ROOT="${NIFTI_ROOT:-/home/gaoqi/dataset/using/acdc/processed/nifti_split_img_gt_scribble_100}"
OUT="${OUT:-/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCR="${SCRIPT_DIR}/preprocess_video_scribble_annotations.py"

for split in train val test; do
  echo "=== scribble_tmi_video [$split] ==="
  python3 "$SCR" \
    --dataset acdc \
    --input_img_dir "$NIFTI_ROOT/$split/img" \
    --input_scribble_dir "$NIFTI_ROOT/$split/acdc_scribbles_TMI" \
    --output_json_path "$OUT/$split/scribble_tmi_video_annotations.coco.json"
done

echo "Done. Output JSONs are under: $OUT/{train,val,test}"

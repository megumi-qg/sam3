#!/usr/bin/env bash
set -euo pipefail

cd /home/gaoqi/sam3

EXP=ablation_obj_o3_reduced_geo_matcher
CKPT="/home/gaoqi/sam3/gq_experiment/cmpb/${EXP}/checkpoints/val_mean_segmentation_coco_eval_segm_Dice.pt"
LOGDIR="/home/gaoqi/sam3/gq_experiment/cmpb/${EXP}/eval_logs"
mkdir -p "${LOGDIR}"

run_one() {
  local ds="$1"
  local test_dir="$2"
  local ds_name="$3"
  local spacing="$4"
  local out="/home/gaoqi/sam3/gq_experiment/cmpb/${EXP}/inference_${ds}_0.7"
  mkdir -p "${out}"

  echo "infer ${ds}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python gq_scripts/evaluate/batch_inference.py \
    --test_dir "${test_dir}" \
    --checkpoint_path "${CKPT}" \
    --confidence_threshold 0.7 \
    --output_dir "${out}" \
    > "${LOGDIR}/inference_${ds}_0.7.log" 2>&1

  echo "eval ${ds}"
  python gq_scripts/evaluate/batch_evaluate.py \
    --predictions_file "${out}/predictions.pkl" \
    --test_dir "${test_dir}" \
    --dataset_name "${ds_name}" \
    --spacing_file "${spacing}" \
    --output_dir "${out}" \
    > "${LOGDIR}/evaluate_${ds}_0.7.log" 2>&1
}

run_one acdc \
  /home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100_cmpb_clean/test \
  ACDC \
  /home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100_cmpb_clean/test/spacing_map.json

run_one btcv \
  /home/gaoqi/dataset/using/btcv/processed/png_coco_sam3_slices_cmpb_clean/test \
  BTCV \
  /home/gaoqi/dataset/using/btcv/processed/png_coco_sam3_slices_cmpb_clean/_nifti_splits/test/spacing_map.json

run_one promise12 \
  /home/gaoqi/dataset/using/promise12/processed/png_coco_sam3_cmpb_clean/test \
  Promise12 \
  /home/gaoqi/dataset/using/promise12/processed/png_coco_sam3_cmpb_clean/_nifti_splits/test/spacing_map.json

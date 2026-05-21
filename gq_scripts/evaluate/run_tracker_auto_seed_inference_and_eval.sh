#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /home/gaoqi/anaconda3/etc/profile.d/conda.sh
conda activate sam3

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TEST_DIR="${TEST_DIR:-/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100/test}"
IMAGE_CHECKPOINT_PATH="${IMAGE_CHECKPOINT_PATH:-/home/gaoqi/sam3/gq_experiment/acdc/full_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt}"
TRACKER_CHECKPOINT_PATH="${TRACKER_CHECKPOINT_PATH:-/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_auto_seed_test}"

DETECTOR_CONDITION_THRESHOLD="${DETECTOR_CONDITION_THRESHOLD:-0.7}"
DETECTOR_OUTPUT_THRESHOLD="${DETECTOR_OUTPUT_THRESHOLD:-0.7}"
TRACKER_DETECTION_THRESHOLD="${TRACKER_DETECTION_THRESHOLD:-0.7}"
FINAL_OUTPUT_MODE="${FINAL_OUTPUT_MODE:-detector_first}"
MIN_MASK_AREA_PX="${MIN_MASK_AREA_PX:-32}"
MIN_MASK_AREA_RATIO="${MIN_MASK_AREA_RATIO:-0.0}"
MAX_COND_FRAMES="${MAX_COND_FRAMES:-4}"
MIN_COND_FRAME_GAP="${MIN_COND_FRAME_GAP:-1}"
CONDITIONING_SELECTION_STRATEGY="${CONDITIONING_SELECTION_STRATEGY:-topk_score}"
PROPAGATION_MODE="${PROPAGATION_MODE:-bidirectional}"
FALLBACK_TO_BEST_SEED="${FALLBACK_TO_BEST_SEED:-0}"
ALLOW_LOW_SCORE_DETECTOR_FALLBACK="${ALLOW_LOW_SCORE_DETECTOR_FALLBACK:-0}"
ALLOW_LOW_SCORE_TRACKER_FALLBACK="${ALLOW_LOW_SCORE_TRACKER_FALLBACK:-0}"
LIMIT_VOLUMES="${LIMIT_VOLUMES:-}"

# 默认行为：
# 1) inference 阶段不读取 GT segmentation 做 seed
# 2) evaluate 阶段读取 test split 的 GT 计算 3D 指标
RUN_EVAL="${RUN_EVAL:-1}"
DATASET_NAME="${DATASET_NAME:-ACDC}"
PREDICTIONS_JSON="${PREDICTIONS_JSON:-$OUTPUT_DIR/coco_predictions_segm.json}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$OUTPUT_DIR/eval_3d}"

AUTO_SEED_ARGS=(
  --test_dir "$TEST_DIR"
  --image_checkpoint_path "$IMAGE_CHECKPOINT_PATH"
  --tracker_checkpoint_path "$TRACKER_CHECKPOINT_PATH"
  --output_dir "$OUTPUT_DIR"
  --detector_condition_threshold "$DETECTOR_CONDITION_THRESHOLD"
  --detector_output_threshold "$DETECTOR_OUTPUT_THRESHOLD"
  --tracker_detection_threshold "$TRACKER_DETECTION_THRESHOLD"
  --final_output_mode "$FINAL_OUTPUT_MODE"
  --min_mask_area_px "$MIN_MASK_AREA_PX"
  --min_mask_area_ratio "$MIN_MASK_AREA_RATIO"
  --max_cond_frames "$MAX_COND_FRAMES"
  --min_cond_frame_gap "$MIN_COND_FRAME_GAP"
  --conditioning_selection_strategy "$CONDITIONING_SELECTION_STRATEGY"
  --propagation_mode "$PROPAGATION_MODE"
)

if [[ "$FALLBACK_TO_BEST_SEED" == "1" ]]; then
  AUTO_SEED_ARGS+=(--fallback_to_best_seed)
fi

if [[ "$ALLOW_LOW_SCORE_DETECTOR_FALLBACK" == "1" ]]; then
  AUTO_SEED_ARGS+=(--allow_low_score_detector_fallback)
fi

if [[ "$ALLOW_LOW_SCORE_TRACKER_FALLBACK" == "1" ]]; then
  AUTO_SEED_ARGS+=(--allow_low_score_tracker_fallback)
fi

if [[ -n "$LIMIT_VOLUMES" ]]; then
  AUTO_SEED_ARGS+=(--limit_volumes "$LIMIT_VOLUMES")
fi

echo "[run_tracker_auto_seed_inference_and_eval] 运行自动 seed 推理 ..."
/bin/bash -lc "source /home/gaoqi/anaconda3/etc/profile.d/conda.sh && conda activate sam3 && CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python gq_scripts/evaluate/tracker_auto_seed_inference.py ${AUTO_SEED_ARGS[*]}"

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ ! -f "$PREDICTIONS_JSON" ]]; then
    echo "错误: 找不到 $PREDICTIONS_JSON" >&2
    exit 1
  fi

  echo "[run_tracker_auto_seed_inference_and_eval] 运行 3D 指标评测 ..."
  python gq_scripts/evaluate/evaluate_tracker_coco_predictions.py \
    --predictions_json "$PREDICTIONS_JSON" \
    --test_dir "$TEST_DIR" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$EVAL_OUTPUT_DIR"
fi

echo "[run_tracker_auto_seed_inference_and_eval] 完成。"
echo "  predictions: $PREDICTIONS_JSON"
echo "  seed report: $OUTPUT_DIR/seed_selection_report.json"
echo "  final report: $OUTPUT_DIR/final_selection_report.json"
if [[ "$RUN_EVAL" == "1" ]]; then
  echo "  eval json:   $EVAL_OUTPUT_DIR/evaluation_results_${DATASET_NAME,,}.json"
  echo "  eval tsv:    $EVAL_OUTPUT_DIR/evaluation_results_${DATASET_NAME,,}_excel_table.tsv"
fi

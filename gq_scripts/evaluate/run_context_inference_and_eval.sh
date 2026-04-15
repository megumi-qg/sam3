#!/usr/bin/env bash
# 运行 slice-context V1 推理，再复用 batch_evaluate.py 做 3D 评估。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_TEST="/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100/test"
DEFAULT_CKPT="/home/gaoqi/sam3/gq_experiment/acdc/full_video_lora_100_context_v1/checkpoints/checkpoint.pt"

TEST_DIR="${TEST_DIR:-$DEFAULT_TEST}"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CKPT}"
DATASET_NAME="${DATASET_NAME:-ACDC}"
SKIP_INFERENCE="${SKIP_INFERENCE:-0}"

EXTRA_INF=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_INF=("$@")
fi

INF_CMD=(
  python gq_scripts/evaluate/batch_inference_context.py
  --test_dir "$TEST_DIR"
  --checkpoint_path "$CHECKPOINT"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  INF_CMD+=(--output_dir "$OUTPUT_DIR")
fi
INF_CMD+=("${EXTRA_INF[@]}")

if [[ "$SKIP_INFERENCE" != "1" ]]; then
  echo "[run_context_inference_and_eval] 运行推理 …"
  "${INF_CMD[@]}"
else
  echo "[run_context_inference_and_eval] SKIP_INFERENCE=1，跳过推理"
fi

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  _OUT="$OUTPUT_DIR"
else
  _PARENT="$(dirname "${TEST_DIR%/}")"
  _OUT="${_PARENT}/inference_predictions_context"
fi
PREDICTIONS="${PREDICTIONS:-$_OUT/predictions.pkl}"

if [[ ! -f "$PREDICTIONS" ]]; then
  echo "错误: 找不到 $PREDICTIONS（请先成功运行推理或设置 PREDICTIONS）" >&2
  exit 1
fi

echo "[run_context_inference_and_eval] 运行评估 …"
python gq_scripts/evaluate/batch_evaluate.py \
  --predictions_file "$PREDICTIONS" \
  --test_dir "$TEST_DIR" \
  --dataset_name "$DATASET_NAME" \
  --output_dir "$_OUT"

EVAL_JSON="${_OUT}/evaluation_results_${DATASET_NAME,,}.json"
EVAL_TSV="${_OUT}/evaluation_results_${DATASET_NAME,,}_excel_table.tsv"
echo "[run_context_inference_and_eval] 完成。JSON: ${EVAL_JSON}；TSV: ${EVAL_TSV}"

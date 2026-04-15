#!/usr/bin/env bash
# 先运行 batch_inference.py，再运行 batch_evaluate.py（同一套 TEST_DIR / OUTPUT_DIR）。
#
# 用法示例（均在仓库 sam3 根目录下执行）：
#
#   1) 全部用脚本里写死的默认路径（测试集、checkpoint 等）：
#        ./gq_scripts/evaluate/run_inference_and_eval.sh
#
#   2) 只改这一次运行要用的目录/权重（不改你 shell 里长期的环境变量）：

# TEST_DIR=/home/gaoqi/dataset/using/mscmr/processed/png_coco_sam3_fullframes_weak/test \
# CHECKPOINT=/home/gaoqi/sam3/gq_experiment/mscmr/expert_scribble_lora/checkpoints/val_mscmr_segmentation_coco_eval_segm_AP.pt \
# OUTPUT_DIR=/home/gaoqi/sam3/gq_experiment/mscmr/expert_scribble_lora/inference_test \
# DATASET_NAME=MSCMR \
# ./gq_scripts/evaluate/run_inference_and_eval.sh -- --confidence_threshold 0.7
#      含义：等号左边是变量名，右边是值；整行最末尾必须是「要执行的脚本」。
#      这些变量只影响这一条命令，关掉终端不会残留。
#
#   3) 给「推理」多传参数（例如把置信度改成 0）：
#        ./gq_scripts/evaluate/run_inference_and_eval.sh -- --confidence_threshold 0.0
#      必须先写 --，再写要传给 batch_inference.py 的参数；中间的 -- 表示「后面都交给推理脚本」。
#      评估脚本 batch_evaluate 不会收到这些参数。
#
# 环境变量（均可选，有默认值）：
#   TEST_DIR       测试集目录（含 image_annotations.coco.json）
#   CHECKPOINT     模型 .pt
#   OUTPUT_DIR     推理输出目录（内含 predictions.pkl）；不设置则用 inference 脚本默认
#   DATASET_NAME   ACDC | BTCV | ...（传给 batch_evaluate）
#   PREDICTIONS    评估用的 pkl 路径；默认 $OUTPUT_DIR/predictions.pkl
#   SKIP_INFERENCE 设为 1 则跳过推理，只跑评估（需已有 pkl）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_TEST="/home/gaoqi/dataset/using/acdc/processed/png_coco_sam3_fullframes_weak/test"
DEFAULT_CKPT="/home/gaoqi/sam3/gq_experiment/acdc/scribble_tmi_lora/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt"

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
  python gq_scripts/evaluate/batch_inference.py
  --test_dir "$TEST_DIR"
  --checkpoint_path "$CHECKPOINT"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  INF_CMD+=(--output_dir "$OUTPUT_DIR")
fi
INF_CMD+=("${EXTRA_INF[@]}")

if [[ "$SKIP_INFERENCE" != "1" ]]; then
  echo "[run_inference_and_eval] 运行推理 …"
  "${INF_CMD[@]}"
else
  echo "[run_inference_and_eval] SKIP_INFERENCE=1，跳过推理"
fi

# 解析最终 output_dir 与 pkl 路径（与 batch_inference 默认逻辑一致）
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  _OUT="$OUTPUT_DIR"
else
  _PARENT="$(dirname "${TEST_DIR%/}")"
  _OUT="${_PARENT}/inference_predictions"
fi
PREDICTIONS="${PREDICTIONS:-$_OUT/predictions.pkl}"

if [[ ! -f "$PREDICTIONS" ]]; then
  echo "错误: 找不到 $PREDICTIONS（请先成功运行推理或设置 PREDICTIONS）" >&2
  exit 1
fi

echo "[run_inference_and_eval] 运行评估 …"
python gq_scripts/evaluate/batch_evaluate.py \
  --predictions_file "$PREDICTIONS" \
  --test_dir "$TEST_DIR" \
  --dataset_name "$DATASET_NAME" \
  --output_dir "$_OUT"

# batch_evaluate 写入 evaluation_results_<dataset 小写>.json，并在同目录生成 *_excel_table.tsv
EVAL_JSON="${_OUT}/evaluation_results_${DATASET_NAME,,}.json"
EVAL_TSV="${_OUT}/evaluation_results_${DATASET_NAME,,}_excel_table.tsv"
echo "[run_inference_and_eval] 完成。JSON: ${EVAL_JSON}；TSV: ${EVAL_TSV}"

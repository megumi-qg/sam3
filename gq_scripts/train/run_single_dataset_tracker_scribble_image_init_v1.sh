#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-mscmr}"
CONDA_PYTHON="${CONDA_PYTHON:-/home/gaoqi/anaconda3/envs/sam3/bin/python}"
GPUS="${GPUS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

case "$DATASET" in
  mscmr)
    CONFIG="configs/mscmr/scribble_tracker_image_init_v1.yaml"
    LOG_FILE="${LOG_FILE:-mscmr_scribble_tracker_image_init_v1.log}"
    ;;
  isbi)
    CONFIG="configs/isbi/scribble_tracker_image_init_v1.yaml"
    LOG_FILE="${LOG_FILE:-isbi_scribble_tracker_image_init_v1.log}"
    ;;
  *)
    echo "Unsupported DATASET=$DATASET; expected mscmr or isbi" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES="$GPUS" nohup "$CONDA_PYTHON" sam3/train/train.py \
  -c "$CONFIG" \
  --use-cluster 0 \
  --num-gpus "$NUM_GPUS" \
  > "$LOG_FILE" 2>&1 &

echo "Started $DATASET tracker training. PID=$! LOG=$LOG_FILE GPUS=$GPUS CONFIG=$CONFIG"

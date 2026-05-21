#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_PYTHON="${CONDA_PYTHON:-/home/gaoqi/anaconda3/envs/sam3/bin/python}"
GPUS="${GPUS:-2,3}"
NUM_GPUS="${NUM_GPUS:-2}"
LOG_FILE="${LOG_FILE:-joint_acdc_mscmr_isbi_scribble_tracker_image_init_v1.log}"

CUDA_VISIBLE_DEVICES="$GPUS" nohup "$CONDA_PYTHON" sam3/train/train.py \
  -c configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1.yaml \
  --use-cluster 0 \
  --num-gpus "$NUM_GPUS" \
  > "$LOG_FILE" 2>&1 &

echo "Started joint tracker training. PID=$! LOG=$LOG_FILE GPUS=$GPUS"

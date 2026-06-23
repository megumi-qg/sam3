# Practical Commands

Run commands from `/home/gaoqi/sam3` unless noted otherwise.

## Main Training

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python sam3/train/train.py \
  -c configs/final/full_lora.yaml --use-cluster 0 --num-gpus 2 \
  < /dev/null > full_lora_cmpb.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 nohup python sam3/train/train.py \
  -c configs/final/weak_lora.yaml --use-cluster 0 --num-gpus 2 \
  < /dev/null > weak_lora_cmpb.log 2>&1 &
```

## 2D Test

```bash
CUDA_VISIBLE_DEVICES=0,1 nohup python sam3/train/train.py \
  -c configs/final/full_lora_test.yaml --use-cluster 0 --num-gpus 2 \
  < /dev/null > full_lora_cmpb_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 nohup python sam3/train/train.py \
  -c configs/final/weak_lora_test.yaml --use-cluster 0 --num-gpus 2 \
  < /dev/null > weak_lora_cmpb_test.log 2>&1 &
```

## Ablation Suite

```bash
nohup python gq_scripts/evaluate/run_cmpb_ablation_suite.py \
  --gpu_ids 0 1 2 3 4 5 6 7 \
  --max_used_mb 2000 \
  > cmpb_ablation_suite.log 2>&1 &
```

Runner behavior:

- one training uses 2 GPUs
- default `--max_train_jobs=2`, so at most 4 GPUs are occupied
- skips existing Dice-best checkpoints and existing evaluation JSONs
- writes `gq_paper/cmpb/results_summary/ablation_auto_summary.csv`

## Threshold Sweep

```bash
nohup python gq_scripts/evaluate/run_cmpb_threshold_sweep.py \
  --gpu_ids 0 1 2 3 4 5 6 7 \
  --max_used_mb 2000 \
  > cmpb_threshold_sweep.log 2>&1 &
```

Threshold sweep covers SAM3-Scribble and SAM3-Full at 0.3, 0.5, 0.7, and 0.9.
It writes:

- `gq_paper/cmpb/results_summary/threshold_sweep_summary.csv`

## LaTeX Compilation

Run from `gq_paper/cmpb/elsarticle/` inside the `sam3` conda environment:

```bash
tectonic main.tex
tectonic supplementary.tex
```

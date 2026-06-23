# CMPB Ablation Experiments

This file is the experiment ledger and paper-number lookup for CMPB ablations.
The broader submission plan lives in `revision_plan.md`.

## Common Protocol

All weak-supervision ablations fix:

- Training data: ACDC, BTCV_cervix, and PROMISE12 joint training.
- Training supervision: train split uses scribble annotations.
- Val/test supervision: dense image annotations for evaluation.
- Mask loss: `PartialMasks(loss_mask=200, loss_dice=10, ignore_index=255)`.
- Main proposed R0: `weak_lora.yaml`, with box/GIoU regression loss=0,
  matcher geometry cost=0, hybrid LoRA scope, rank=8/alpha=16, threshold=0.7.
- Checkpoint selection:
  `checkpoints/val_mean_segmentation_coco_eval_segm_Dice.pt`.
- Final metrics: 3D IoU, Dice, HD95, and NSD.

## 1. Box/GIoU Regression Loss

Purpose: test whether scribble-derived boxes are suitable regression targets.
Only box/GIoU regression loss changes; matcher geometry cost stays 0/0.

| ID | Config | loss_bbox/loss_giou | Matcher bbox/GIoU | Experiment dir |
|---|---|---:|---:|---|
| R0 Proposed | `weak_lora.yaml` | 0/0 | 0/0 | `gq_experiment/cmpb/weak_lora_acdc_btcv_promise12` |
| R1 Full geo loss | `weak_lora_reg_r1_full_geo_loss.yaml` | 5/2 | 0/0 | `gq_experiment/cmpb/ablation_reg_r1_full_geo_loss` |
| R2 Reduced geo loss | `weak_lora_reg_r2_reduced_geo_loss.yaml` | 1/1 | 0/0 | `gq_experiment/cmpb/ablation_reg_r2_reduced_geo_loss` |

| Variant | Avg Dice (%) | Avg IoU (%) | Avg HD95 | Avg NSD (%) | Interpretation |
|---|---:|---:|---:|---:|---|
| R0 Proposed | 86.27 | 77.35 | 10.48 | 82.15 | Best overall reference |
| R1 Full box/GIoU loss | 85.86 | 76.75 | 10.27 | 81.65 | Restoring original regression loss lowers Dice/NSD |
| R2 Reduced box/GIoU loss | 85.95 | 76.99 | 11.32 | 81.87 | Reduced geometry loss still below R0 |

Conclusion: removing box/GIoU regression losses is supported. The effect size
is moderate; the paper should say this improves robustness and overall
Dice/NSD, not that it dramatically changes performance.

## 2. Matcher Geometry Cost

Purpose: test whether Hungarian matching should use scribble-derived geometry
costs. This group fixes `PartialMasks`, box/GIoU regression loss=0, LoRA
scope/rank, and threshold=0.7; only matcher costs differ.

| ID | Config | Matcher cls/bbox/GIoU | Experiment dir |
|---|---|---:|---|
| O1 Original matcher | `weak_lora_obj_o1_original_matcher.yaml` | 2/5/2 | `gq_experiment/cmpb/ablation_obj_o1_original_matcher` |
| O2 Proposed | `weak_lora.yaml` | 2/0/0 | `gq_experiment/cmpb/weak_lora_acdc_btcv_promise12` |
| O3 Reduced geometry | `weak_lora_obj_o3_reduced_geo_matcher.yaml` | 2/1/1 | `gq_experiment/cmpb/ablation_obj_o3_reduced_geo_matcher` |
| O4 Rebalanced | existing ablation config | 5/1/1 | `gq_experiment/cmpb/ablation_obj_o4_rebalanced_matcher` |

Result file:

- `gq_paper/cmpb/results_summary/objective_ablation_o1_o2_o3_o4.csv`

| Variant | Matcher cls/bbox/GIoU | Avg Dice (%) | Avg IoU (%) | Avg HD95 | Avg NSD (%) |
|---|---:|---:|---:|---:|---:|
| O1 Original matcher | 2/5/2 | 86.19 | 77.24 | 10.55 | 81.74 |
| O2 Proposed | 2/0/0 | 86.27 | 77.35 | 10.48 | 82.15 |
| O3 Reduced geometry | 2/1/1 | 86.12 | 77.18 | 10.65 | 81.83 |
| O4 Rebalanced | 5/1/1 | 85.83 | 76.80 | 10.90 | 81.69 |

Conclusion: removing geometry matching cost gives the best macro-average
result, but the margin over O1/O3 is small. Frame this as a simple and stable
scribble-compatible assignment choice, not a large performance gain.

## 3. LoRA Rank

Purpose: verify that performance does not simply come from larger LoRA
capacity. All rank ablations use the same hybrid LoRA scope:
`vision_encoder`, `text_encoder`, `geometry_encoder`, `detr_encoder`, and
`detr_decoder` use LoRA; `mask_decoder` and `dot_prod_scoring` are fully
trainable.

| ID | Config | rank/alpha | Experiment dir |
|---|---|---:|---|
| L1 Low rank | `weak_lora_rank_l1_r4.yaml` | 4/8 | `gq_experiment/cmpb/ablation_lora_rank_l1_r4` |
| R0 Proposed | `weak_lora.yaml` | 8/16 | `gq_experiment/cmpb/weak_lora_acdc_btcv_promise12` |
| L2 High rank | `weak_lora_rank_l2_r16.yaml` | 16/32 | `gq_experiment/cmpb/ablation_lora_rank_l2_r16` |

| Variant | Rank/alpha | Avg Dice (%) | Avg IoU (%) | Avg HD95 | Avg NSD (%) |
|---|---:|---:|---:|---:|---:|
| L1 Low rank | 4/8 | 86.33 | 77.35 | 10.45 | 81.70 |
| R0 Proposed | 8/16 | 86.27 | 77.35 | 10.48 | 82.15 |
| L2 High rank | 16/32 | 85.66 | 76.58 | 11.38 | 81.19 |

Conclusion: rank 4 is competitive and slightly higher in Dice, while rank 8
has better NSD and is the balanced default. Increasing rank to 16 does not
help. Do not claim rank 8 is strictly best on every metric.

## 4. LoRA Scope

Purpose: test whether hybrid LoRA is more stable than single-module
adaptation. All scope ablations fix rank=8/alpha=16 and the R0 objective.

| ID | Config | Trainable scope | Experiment dir |
|---|---|---|---|
| S1 Vision-only | `weak_lora_scope_s1_vision_only.yaml` | LoRA on `vision_encoder`; full heads | `gq_experiment/cmpb/ablation_lora_scope_s1_vision_only` |
| S2 DETR-only | `weak_lora_scope_s2_detr_only.yaml` | LoRA on `detr_encoder,detr_decoder`; full heads | `gq_experiment/cmpb/ablation_lora_scope_s2_detr_only` |
| S3 Heads-only | `weak_lora_scope_s3_heads_only.yaml` | no LoRA; full heads only | `gq_experiment/cmpb/ablation_lora_scope_s3_heads_only` |
| R0 Proposed | `weak_lora.yaml` | hybrid LoRA + full heads | `gq_experiment/cmpb/weak_lora_acdc_btcv_promise12` |

S3 heads-only requires a special optimizer config because there are no
`lora_A/lora_B` parameters. The config optimizes only `segmentation_head.*` and
`dot_prod_scoring.*`.

| Variant | Avg Dice (%) | Avg IoU (%) | Avg HD95 | Avg NSD (%) | Status |
|---|---:|---:|---:|---:|---|
| R0 Hybrid LoRA | 86.27 | 77.35 | 10.48 | 82.15 | complete |
| S1 Vision-only | 85.83 | 76.78 | 10.47 | 81.01 | complete |
| S2 DETR-only | 82.82 | 73.36 | 12.79 | 78.08 | complete |
| S3 Heads-only | 62.91 | 51.24 | 39.60 | 53.88 | complete |

Conclusion: hybrid LoRA is clearly stronger than single-side or heads-only
adaptation. Vision-only remains competitive but lower in Dice/NSD; DETR-only
drops substantially; heads-only fails badly.

## 5. Inference Confidence Threshold

Purpose: verify that threshold=0.7 is not cherry-picking and observe loose vs
strict filtering.

This group does not retrain. It runs inference/evaluation from Dice-best
checkpoints.

| Model | Experiment dir | Thresholds |
|---|---|---|
| SAM3-Scribble | `gq_experiment/cmpb/weak_lora_acdc_btcv_promise12` | 0.3, 0.5, 0.7, 0.9 |
| SAM3-Full | `gq_experiment/cmpb/full_lora_acdc_btcv_promise12` | 0.3, 0.5, 0.7, 0.9 |

Script:

- `gq_scripts/evaluate/run_cmpb_threshold_sweep.py`

Result file:

- `gq_paper/cmpb/results_summary/threshold_sweep_summary.csv`

| Model | Threshold | Avg Dice (%) | Avg IoU (%) | Avg HD95 | Avg NSD (%) |
|---|---:|---:|---:|---:|---:|
| SAM3-Scribble | 0.3 | 85.79 | 76.61 | 11.74 | 80.95 |
| SAM3-Scribble | 0.5 | 86.14 | 77.12 | 10.99 | 81.61 |
| SAM3-Scribble | 0.7 | 86.27 | 77.35 | 10.48 | 82.15 |
| SAM3-Scribble | 0.9 | 85.91 | 77.03 | 9.99 | 81.89 |
| SAM3-Full | 0.3 | 86.32 | 77.77 | 12.63 | 83.06 |
| SAM3-Full | 0.5 | 86.53 | 78.12 | 11.28 | 83.36 |
| SAM3-Full | 0.7 | 86.50 | 78.22 | 10.49 | 83.40 |
| SAM3-Full | 0.9 | 66.27 | 57.03 | 20.94 | 60.80 |

Conclusion: threshold 0.7 is a robust operating point. For SAM3-Scribble, 0.7
has the best macro Dice/IoU/NSD among tested thresholds, while 0.9 improves
HD95 but lowers Dice/NSD. For SAM3-Full, 0.5 and 0.7 are close, but 0.9 is too
strict and causes severe missed detections.

# Project Context

This repository is currently on the older `scribble-weaksam3` project line,
not the later `sam3_scribble++` tracker extension.

The active goal is to revise a rejected MICCAI 2026 SAM3-Scribble paper for
Computer Methods and Programs in Biomedicine (CMPB).

## Method Framing

The project adapts the SAM3 image model for medical image segmentation using
text prompts and weak scribble supervision.

Use this framing:

- text-prompted medical image segmentation
- weakly supervised SAM3 adaptation
- zero-click text-prompted workflow

Avoid making the core claim sound like standard iterative interactive
segmentation. The method is text-prompted and trained with sparse scribble
labels, not an interactive click-by-click segmentation pipeline.

## Main Repository Areas

- `sam3/`: model, training, evaluation, and LoRA code.
- `sam3/train/configs/final/`: current CMPB training and test configs.
- `gq_scripts/preprocess/`: dataset preprocessing and clean CMPB split
  construction.
- `gq_scripts/evaluate/`: batch inference, evaluation, ablation runners, and
  threshold sweeps.
- `gq_experiment/cmpb/`: CMPB experiment outputs.
- `gq_paper/cmpb/`: CMPB manuscript, aggregation scripts, splits, and result
  summaries.

## Key Constraints

- Weak-supervision ablations should keep `PartialMasks`; do not use full mask
  loss on scribble labels.
- Final paper tables should prefer 3D `batch_evaluate.py` results over 2D COCO
  AP.
- BTCV small bowel is the hardest category; weak BTCV Dice can be reasonable
  while HD95/NSD remain worse because the anatomy is complex and discontinuous.

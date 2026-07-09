# Project Overview

This repository is a SAM3 customization project for medical image segmentation.

The primary training path is still SAM3 image-model fine-tuning, not the official video training recipe. Current work covers:

- full-supervised fine-tuning
- scribble-based weak supervision
- LoRA parameter-efficient fine-tuning
- 3D volumes organized as video-like samples
- tracker propagation/refinement experiments
- slice-context experiments
- single-dataset follow-up checks for ACDC, MSCMR, and ISBI

## Default Assumptions

- The core task is medical image segmentation with SAM3 image-model fine-tuning.
- Hydra configs are the first control surface for training behavior.
- ACDC is the main reference dataset for understanding local implementation choices.
- Weak supervision must preserve the `1/0/255` partial-supervision semantics.
- LoRA is the default parameter-efficient fine-tuning strategy.
- 3D/video-like data organization is connected, but it is not automatically true 3D modeling.
- Tracker and context experiments are active research directions, not default replacements for the image-model baseline.

## Main Boundaries

- Do not treat scribbles as dense masks.
- Do not assume `train_batch_size` in video-like configs is equivalent to old 2D image batch size.
- Do not judge final medical results only by 2D COCO AP; use 3D Dice/IoU/HD95/NSD when available.
- Do not use full labels or full-supervised tracker checkpoints as the main method for strict scribble experiments.
- `gq_experiment/cmpb/` is from another branch of experiments and is not evidence for the current branch's ACDC/MSCMR/ISBI line.

## Related Files

- Data and paths: `environment-and-data.md`
- Weak supervision contract: `weak-supervision.md`
- Training and evaluation: `training-and-eval.md`
- 3D/context: `video-context.md`
- ACDC experiment map: `acdc-experiments.md`
- Non-ACDC single-dataset experiments: `single-dataset-experiments.md`
- Tracker: `tracker/README.md`
- Current stage: `current-state.md`

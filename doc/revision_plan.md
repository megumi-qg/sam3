# CMPB Revision Plan

This file records the remaining work before submitting the SAM3-Scribble paper
to Computer Methods and Programs in Biomedicine (CMPB). Detailed ablation
numbers live in `ablation_experiments.md`.

## Current Judgment

The paper direction matches CMPB, but the submission should be strengthened
from "model runs with results" into a complete, reproducible, and sufficiently
validated biomedical computing method.

Main remaining risks:

- MICCAI reviewers' sparse-supervision and ablation concerns need fuller
  responses.
- Scribble vs point / sparse supervision comparison is still missing.
- Annotation burden analysis is still thin.
- Statistical testing, failure cases, and a fuller discussion are still needed.

## Completed Or Mostly Completed

- Elsevier `elsarticle` migration.
- Introduction rewrite.
- Independent Related Work section.
- Main tables by ACDC / BTCV_cervix / PROMISE12.
- Per-structure tables moved to supplementary.
- Matcher objective ablation O1-O4 completed.
- Proposed SAM3-Scribble switched to O2 no geometric matcher.
- Box/GIoU regression loss ablation completed.
- LoRA rank/scope ablations completed.
- Threshold sensitivity sweep completed.

## Highest-Priority Missing Experiment

### Scribble vs Point / Sparse Supervision

MICCAI reviewer 3 questioned whether scribble is just a point prompt / sparse
points variant. Add a direct comparison under the same three-dataset joint
training protocol.

Recommended variants:

- sample equal-number foreground points from scribbles
- foreground scribble only, without negative/valid scribble
- complete scribble + `valid_mask`, current method
- optional: different scribble sparsity ratios

Goal: show that the method uses scribble structure and negative/valid
information, not just sparse foreground points.

## Annotation Cost / Burden Analysis

Current manuscript only has scribble pixel ratio, which is not enough.

Recommended additions:

- scribble pixels / full mask pixels ratio per dataset
- boundary scribble length / full boundary length ratio
- if feasible, manual or semi-manual timing for 3-5 dense masks vs scribbles
- otherwise, cite ScribbleBench or medical scribble literature for reduced
  annotation burden

Goal: support "substantially reduced annotation requirements" without
overclaiming.

## Per-Class And Statistical Results

Per-structure tables are already in supplementary, but statistical testing is
still needed.

Recommended additions:

- per-dataset per-class Dice / HD95
- per-case mean ± std
- paired tests for:
  - SAM3-Scribble vs ScribbleBench
  - SAM3-Scribble vs SAT-Pro
  - SAM3-Scribble vs SAM3-Full
- optional boxplot or violin plot for per-case Dice distribution

## Failure Case / Qualitative Analysis

Current qualitative results mostly show successful examples.

Add:

- failure cases
- BTCV_cervix cases with high HD95
- leakage, over-segmentation, or missed-boundary examples

Goal: make the discussion credible and not only success-driven.

## Dataset And Protocol Details

Add or expand a dataset/protocol table with:

- dataset name
- modality
- organs/classes
- train/test volumes
- image or slice counts
- split source
- scribble generation protocol
- spacing source for HD95

Important details:

- ACDC and PROMISE12 use official/source-preserving splits.
- BTCV_cervix uses an internal SAT split because original test targets are not
  public.
- All datasets use slice-wise 2D inference followed by stacked 3D evaluation.

## Implementation And Reproducibility Details

Add complete implementation details:

- SAM3 checkpoint source
- input resolution
- optimizer
- learning rate
- scheduler
- epochs
- batch size
- GPU type and count
- LoRA rank/alpha
- trainable parameter count calculation
- checkpoint selection metric
- fixed confidence threshold 0.7
- HD95/NSD spacing handling
- code/data availability or release plan

## Discussion Expansion

The discussion should become more journal-style:

- why Partial Mask Loss is suitable for scribble supervision
- why box/GIoU targets from scribbles are unreliable
- why removing geometry matching cost is a stable choice
- why BTCV_cervix HD95 can lag behind the full model
- value of a text-prompted zero-click workflow
- strengths and limits of 2D slice-wise formulation
- relation to static WSL methods, SAM-style prompt models, and medical
  foundation models
- low-annotation-cost deployment implications
- future work: iterative prompts, point/box/click prompts, 3D context, tracker
  propagation

## Minimum Submission Bar

Before CMPB submission, complete at least:

1. Ablation study table.
2. Scribble vs point / sparse supervision comparison.
3. Per-class and mean ± std result table.
4. Dataset/protocol table.
5. Expanded Related Work and Discussion.
6. More complete implementation details.

If only the writing is expanded without the missing experiments, the paper is
still likely to be considered under-validated.

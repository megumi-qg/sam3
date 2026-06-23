# Agent Notes: SAM3-Scribble CMPB Resubmission

This file is the entry index for AI agents working in this repository. Do not
put detailed project state here. Detailed context lives under `doc/`.

## Current Task Context

- Repository line: older `scribble-weaksam3`, not the later
  `sam3_scribble++` tracker extension.
- Current paper goal: revise a rejected MICCAI 2026 SAM3-Scribble paper for
  Computer Methods and Programs in Biomedicine (CMPB).
- Core method framing: text-prompted SAM3 adaptation for medical image
  segmentation under weak scribble supervision.
- Important wording constraint: do not present the method as a standard
  iterative interactive segmentation method.
- Important training constraint: weak-supervision ablations must use
  `PartialMasks`; do not train with dense/full mask loss on scribble labels,
  because unlabeled pixels in scribble masks are unknown, not background.

## Documentation Map

Start from `doc/README.md`.

- Project and repository overview: `doc/project_context.md`
- Paper files and manuscript state: `doc/paper_context.md`
- Dataset paths, splits, preprocessing, spacing: `doc/dataset_protocol.md`
- Training configs and code-level protocol: `doc/training_protocol.md`
- Main full-vs-weak results: `doc/experiment_results.md`
- Ablation ledger and paper-number lookup: `doc/ablation_experiments.md`
- CMPB revision priorities and remaining work: `doc/revision_plan.md`
- Common commands for training, evaluation, ablations, and LaTeX:
  `doc/commands.md`
- Safe paper wording and reviewer-risk notes: `doc/writing_guidance.md`

## Fast Routing

- If editing the CMPB manuscript, read `doc/paper_context.md`,
  `doc/revision_plan.md`, and `doc/writing_guidance.md`.
- If running or debugging experiments, read `doc/dataset_protocol.md`,
  `doc/training_protocol.md`, and `doc/commands.md`.
- If checking reported numbers, read `doc/experiment_results.md` and
  `doc/ablation_experiments.md`.
- If updating project background, update the relevant file under `doc/` and
  keep this file as an index.

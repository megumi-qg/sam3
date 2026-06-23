# Project Documentation Index

This directory stores the long-form project context for AI agents. `agent.md`
is intentionally kept short and should point here instead of duplicating
details.

## Which File To Read

- `project_context.md`: repository line, project goal, major directories, and
  high-level method framing.
- `paper_context.md`: paper files, manuscript status, compilation notes, and
  CMPB-specific framing.
- `dataset_protocol.md`: clean dataset paths, train/val/test splits,
  preprocessing, scribble sources, and spacing files for 3D metrics.
- `training_protocol.md`: main configs, LoRA setup, PartialMasks protocol,
  checkpoint selection, early stopping, and relevant code changes.
- `experiment_results.md`: main full-supervision vs SAM3-Scribble results and
  evaluation file locations.
- `ablation_experiments.md`: ablation design, experiment directories, current
  results, and concise paper interpretations.
- `revision_plan.md`: remaining CMPB submission work, priority experiments,
  paper-structure improvements, and minimum submission bar.
- `commands.md`: practical training, testing, ablation, threshold-sweep,
  evaluation, and LaTeX commands.
- `writing_guidance.md`: safe wording, claims to avoid, and reviewer-risk
  notes.

## Update Rule

Keep each fact in one primary location:

- Stable paths and protocol facts belong in dataset/training/paper context
  files.
- Numeric results belong in `experiment_results.md` or
  `ablation_experiments.md`.
- Future work and priorities belong in `revision_plan.md`.
- Reusable commands belong in `commands.md`.
- Claim wording belongs in `writing_guidance.md`.

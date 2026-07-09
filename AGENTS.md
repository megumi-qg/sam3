# AGENTS.md

This repository uses `docs/agent/` as the project context directory.

Before changing code or running experiments, read the narrowest relevant files:

- `docs/agent/README.md` for the document map and update rules.
- `docs/agent/project-overview.md` for project scope and default assumptions.
- `docs/agent/environment-and-data.md` for local environment, paths, and dataset semantics.
- `docs/agent/weak-supervision.md` for scribble supervision, `valid_mask`, and COCO JSON contracts.
- `docs/agent/training-and-eval.md` for Hydra configs, LoRA, inference, and evaluation.
- `docs/agent/video-context.md` for 3D/video-like samples and slice context experiments.
- `docs/agent/acdc-experiments.md` for old single-ACDC experiment directories and which ones are still useful.
- `docs/agent/tracker/README.md` for tracker-specific work.
- `docs/agent/current-state.md` for the current experimental stage, latest results, and next steps.
- `docs/agent/upstream-sam3.md` for local notes about official SAM3/SAM3.1 docs.

Update rules:

- Stable project mechanisms belong in the topic file that owns them.
- Current results, active checkpoints, and next actions belong only in `docs/agent/current-state.md`.
- Tracker details belong under `docs/agent/tracker/`; do not duplicate them in general files.
- Historical tracker or context results should move to an archive section/file instead of staying in current guidance.
- `gq_experiment/cmpb/` belongs to another branch of experiments; do not use it as evidence for the current branch unless the user explicitly asks.
- Keep this root file short. It is an index, not the project memory.

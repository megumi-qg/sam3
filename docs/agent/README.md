# Agent Documentation Map

This directory is the maintained project context for Codex and other AI agents.

## Entry Point

Codex reads the repository-level `AGENTS.md` by default. That file should stay short and route the agent here. Do not put full project history in `AGENTS.md`.

## Files

- `project-overview.md`: stable project scope, main assumptions, and boundaries.
- `environment-and-data.md`: conda environment, local paths, datasets, split semantics, and metric caveats.
- `weak-supervision.md`: scribble supervision, `valid_mask`, partial-mask loss, and COCO JSON contracts.
- `training-and-eval.md`: primary Hydra configs, LoRA policy, inference scripts, and evaluation conventions.
- `video-context.md`: 3D/video-like data organization, context v1 result, and context direction.
- `acdc-experiments.md`: ACDC-only experiment directory map and cleanup guidance.
- `single-dataset-experiments.md`: non-ACDC single-dataset experiment map and detector-only checks.
- `upstream-sam3.md`: local summary of official SAM3/SAM3.1 docs relevant to this fork.
- `current-state.md`: current experiment stage, strongest baselines, latest tracker status, and next steps.
- `tracker/README.md`: tracker-specific index.
- `tracker/method.md`: tracker training, seed bank, adapter/loss, and inference semantics.
- `tracker/results-current.md`: joint ACDC+MSCMR+ISBI tracker results.
- `tracker/results-archive.md`: ACDC-only and previous tracker/context results.

## Update Rules

- Put stable mechanisms in the topic file that owns them.
- Put active results and next steps only in `current-state.md`.
- Put tracker-specific details only under `tracker/`.
- If a result is superseded, move it to an archive file and state why it is historical.
- Avoid maintaining the same conclusion in two files.
- Prefer paths to exact scripts/configs over prose descriptions when possible.
- Treat `gq_experiment/cmpb/` as another branch's experiment area. It is not part of the current branch's main evidence chain unless the user explicitly asks for CMPB work.

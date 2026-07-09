# Upstream SAM3 Notes

This file summarizes the official SAM3 documents only where they matter for this fork. Keep the upstream files themselves unchanged:

- `README.md`
- `README_TRAIN.md`
- `RELEASE_SAM3p1.md`

## Official Training Docs

`README_TRAIN.md` describes general SAM3 fine-tuning with Hydra, local execution, cluster execution, monitoring, and evaluation examples.

Local project caveat:

- The official recipes do not directly encode this repository's medical scribble supervision, `valid_mask`, joint ACDC+MSCMR+ISBI setup, or tracker seed-bank workflow.

## SAM 3.1

`RELEASE_SAM3p1.md` introduces SAM 3.1 and Object Multiplex for faster multi-object video tracking/inference.

Local status:

- The repository has SAM 3.1/Object Multiplex code paths.
- Local model-builder entries include SAM3.1 predictor and multiplex video predictor/model builders.
- Default checkpoint source is `facebook/sam3.1`.
- User's local SAM3.1 checkpoint path is `/home/gaoqi/official_ckpt/sam3.1_hf`.

Important caveat:

- SAM3.1 improves the official video inference/tracking side, but it does not directly replace this project's current medical image-model fine-tuning, scribble supervision, joint tracker training, or context experiments.
- If trying SAM3.1 tracker training, start from a minimal full-supervised closed loop before adapting strict scribble semantics.

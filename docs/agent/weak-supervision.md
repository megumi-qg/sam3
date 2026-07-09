# Weak Supervision Contract

This project's scribble supervision is partial-region supervision. It must not be converted mentally or programmatically into dense-mask supervision.

## Preprocessing

Full-supervised preprocessing:

- `gq_scripts/preprocess/preprocess_full_annotations.py`
- `gq_scripts/preprocess/preprocess_video_annotations.py`

Scribble weak-supervised preprocessing:

- `gq_scripts/preprocess/preprocess_scribble_annotations.py`
- `gq_scripts/preprocess/preprocess_video_scribble_annotations.py`

Scribble preprocessing produces COCO-style annotations where:

- `segmentation` stores target scribble pixels.
- `valid_mask` defines the trusted supervision region.

## `valid_mask`

`valid_mask` is the core weak-supervision mechanism.

Mask values:

- `1`: positive scribble pixels.
- `0`: valid-region background.
- `255`: ignored region.

In the `scribble1` setting, other objects' scribbles enter the current query's `valid_mask` and act as background constraints.

## COCO JSON Loading

Key files:

- `sam3/train/data/coco_json_loaders.py`
- `sam3/train/data/sam3_image_dataset.py`
- `sam3/train/data/sam3_video_dataset.py`

Responsibilities:

- `coco_json_loaders.py` converts COCO JSON into query-based training structures.
- `sam3_image_dataset.py` converts full or weak annotations into training masks.
- `sam3_video_dataset.py` handles video-like volume samples.

Weak-supervised dataset behavior:

- Build three-value masks with `1/0/255`.
- Preserve `ignore_index=255`.
- Bbox priority is inferred bbox, JSON bbox, then pseudo bbox computed from scribble.

## Loss Contract

Full-supervised configs use dense `Masks`.

Scribble configs use `PartialMasks`.

Important files:

- `sam3/train/loss/loss_fns.py`
- `sam3/train/loss/sam3_tracker_loss.py`

Rules:

- `PartialMasks` computes loss only on valid regions.
- `ignore_index=255` must be preserved.
- Scribble training should not introduce full labels unless the experiment is explicitly no longer strict weak supervision.

# Tracker Documentation

Tracker-specific context lives here.

Read order:

1. `method.md` for training/inference semantics.
2. `results-current.md` for joint ACDC+MSCMR+ISBI results.
3. `results-archive.md` when ACDC-only tracker context matters.
4. `../current-state.md` for the newest active next steps.

## Core Principle

The tracker is not the default replacement for the image detector. Its intended role is propagation, refinement, temporal consistency, and diagnosis of detector/tracker complementarity.

For strict scribble experiments:

- Do not train tracker with full labels.
- Do not use full-supervised tracker checkpoints as the main method.
- Conditioning seeds should come from weak detector predictions or explicitly documented pseudo-seed banks.

## Key Files

Training adapter and loss:

- `sam3/model/sam3_tracker_train_adapter.py`
- `sam3/train/loss/sam3_tracker_loss.py`

Inference and evaluation:

- `gq_scripts/evaluate/tracker_auto_seed_inference.py`
- `gq_scripts/evaluate/evaluate_tracker_coco_predictions.py`
- `gq_scripts/evaluate/run_tracker_auto_seed_inference_and_eval.sh`

Pseudo seed bank:

- `gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py`

Joint configs:

- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v2_val_macro.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v3_temporal_area_centroid_val_macro.yaml`

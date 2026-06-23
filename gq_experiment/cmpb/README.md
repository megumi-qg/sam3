# CMPB Experiments

This directory stores experiments for the CMPB resubmission protocol.

## Current clean experiments

- `full_lora_acdc_btcv_promise12`: full-supervision hybrid LoRA training on the CMPB clean splits. Uses `train/full_annotations.coco.json` for training and `val/image_annotations.coco.json` for model selection.
- `weak_lora_acdc_btcv_promise12`: proposed weak-supervision hybrid LoRA training on the CMPB clean splits. This is the O2 no-geometric-matcher run (`cls/bbox/giou = 2/0/0`). Uses `train/scribble_annotations.coco.json` for training and `val/image_annotations.coco.json` for model selection.

Both current configs use balanced training sampling:

- ACDC multiplier: 3
- BTCV multiplier: 1
- PROMISE12 multiplier: 3

This gives approximately balanced effective training image counts: ACDC 3966, BTCV 3768, PROMISE12 3246. Validation uses multiplier 1.

Both current configs save unified best checkpoints by three-dataset mean segmentation AP and Dice, and use early stopping on `val_mean_segmentation_coco_eval_segm_Dice`.

## Archived experiments

- `ablation_obj_o1_original_matcher`: matcher ablation with original SAM3 matching costs (`2/5/2`).
- `ablation_obj_o3_reduced_geo_matcher`: matcher ablation with reduced geometric costs (`2/1/1`).
- `ablation_obj_o4_rebalanced_matcher`: matcher ablation for the previous rebalanced matcher (`5/1/1`). This was previously named `weak_lora_acdc_btcv_promise12` before O2 was adopted as the proposed method.

Checkpoint cleanup note: each retained training directory keeps only
`checkpoints/val_mean_segmentation_coco_eval_segm_Dice.pt`, because final CMPB
3D inference/evaluation uses the Dice-best checkpoint. Intermediate
`checkpoint*.pt` files and AP-best checkpoint copies were deleted to save
storage.

Removed stale runs:

- `full_lora_acdc_btcv_promise12_old_before_btcv_prompt_fix`: stale run before
  the BTCV prompt fix.
- `full_lora_acdc_btcv_promise12_test_dice`: unreferenced duplicate/test-dice
  run.

Note: previous BTCV clean annotations had inconsistent category prompts between `scribble_annotations.coco.json` and `image_annotations.coco.json` / `full_annotations.coco.json` (`bladder/uterus/rectum/small bowel` vs `Structure_1..4`). The clean BTCV JSON files and preprocessing script have been fixed to use the anatomical category names consistently.

# Training Protocol

## Main Configs

- Full supervision: `sam3/train/configs/final/full_lora.yaml`
- Weak supervision / proposed SAM3-Scribble:
  `sam3/train/configs/final/weak_lora.yaml`
- Test configs:
  - `sam3/train/configs/final/full_lora_test.yaml`
  - `sam3/train/configs/final/weak_lora_test.yaml`

## Weak-Supervision Control Conditions

All weak-supervision ablations should keep:

- Joint training on ACDC, BTCV_cervix, and PROMISE12.
- Train split uses scribble annotations.
- Val/test use dense image annotations for evaluation.
- Mask loss: `PartialMasks(loss_mask=200, loss_dice=10, ignore_index=255)`.
- Checkpoint selection: three-dataset validation mean Dice best checkpoint.
- Main metrics: 3D IoU, Dice, HD95, and NSD.

Do not use full mask loss on scribble labels. Scribble labels leave many pixels
unlabeled; treating those pixels as background trains the model to predict
scribble lines and gives an uninterpretable ablation.

## Proposed SAM3-Scribble Setup

- Config: `weak_lora.yaml`
- Box/GIoU regression loss: 0/0
- Hungarian matcher geometry cost: bbox/GIoU = 0/0
- Inference confidence threshold: 0.7
- LoRA rank/alpha: r=8, alpha=16
- LoRA targets:
  - `vision_encoder`
  - `text_encoder`
  - `geometry_encoder`
  - `detr_encoder`
  - `detr_decoder`
- Fully trainable heads:
  - `mask_decoder`
  - `dot_prod_scoring`

Balanced sampling:

- ACDC multiplier: 3
- BTCV multiplier: 1
- PROMISE12 multiplier: 3
- Validation/test multiplier: 1

Validation metrics:

- segmentation AP
- Dice

Best checkpoints:

- unified three-dataset mean AP
- unified three-dataset mean Dice

Early stopping monitors `val_mean_segmentation_coco_eval_segm_Dice` with
`min_epochs=12`, `patience=6`, and `min_delta=0.002`.

## Important Code Changes

- `sam3/train/data/sam3_image_dataset.py`: dataset multiplier is actually
  applied.
- `sam3/eval/coco_eval_offline.py`: supports AP-only output and mean Dice.
- `sam3/train/trainer.py`: supports unified combined best checkpoints and early
  stopping.
- `gq_scripts/evaluate/batch_inference.py`: LoRA inference targets were fixed
  to match hybrid LoRA training; `mask_decoder` is not a LoRA target.
- `gq_scripts/evaluate/batch_inference.py`: accepts
  `--lora_target_components` for scope-ablation checkpoints.
- `sam3/model/lora.py`: empty `lora_target_components: []` can freeze the
  backbone and unfreeze only heads for S3 heads-only.

## Completed Main Runs

Experiment directories:

- Full: `gq_experiment/cmpb/full_lora_acdc_btcv_promise12`
- Weak / proposed SAM3-Scribble:
  `gq_experiment/cmpb/weak_lora_acdc_btcv_promise12`

Best validation checkpoints:

- Full AP-best and Dice-best both selected at epoch 4.
- Weak O2 AP-best and Dice-best both selected at epoch 2.
- Use `checkpoints/val_mean_segmentation_coco_eval_segm_Dice.pt` for final 3D
  inference and evaluation.

Checkpoint cleanup note:

- Each retained CMPB training directory keeps only
  `checkpoints/val_mean_segmentation_coco_eval_segm_Dice.pt`.
- Intermediate `checkpoint*.pt` files and AP-best checkpoint copies were
  deleted because final CMPB 3D inference/evaluation uses the Dice-best
  checkpoint.
- The stale `full_lora_acdc_btcv_promise12_old_before_btcv_prompt_fix` and
  unreferenced `full_lora_acdc_btcv_promise12_test_dice` directories were
  removed to save storage.

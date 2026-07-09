# ACDC Tracker Results

This file keeps ACDC-only tracker context. These results are not the joint ACDC+MSCMR+ISBI conclusion, but they are still important for explaining why tracker merge was tried.

## ACDC Full Tracker

ACDC-only full tracker experiments showed that tracker propagation can work under cleaner settings.

Historical notes:

- Full image baseline path: `gq_experiment/acdc/full_video_lora_100/`
- Full tracker v1 path: `gq_experiment/acdc/full_sam3_tracker_image_init/`
- Full tracker v2 path: `gq_experiment/acdc/full_sam3_tracker_image_init_v2/`
- Full tracker v2 train-like hybrid was the strongest ACDC-only full tracker setting.

Full tracker v2 used:

- detector/image backbone: `gq_experiment/acdc/full_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- tracker checkpoint: `gq_experiment/acdc/full_sam3_tracker_image_init_v2/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`

Archived test outputs:

- `gq_experiment/acdc/full_sam3_tracker_image_init_v2/tests/merge_train_like/`: Dice 0.9324
- `gq_experiment/acdc/full_sam3_tracker_image_init_v2/tests/tracker_only_train_like/`: Dice 0.9319
- `gq_experiment/acdc/full_sam3_tracker_image_init_v2/tests/tracker_only_best_single_bidir/`: Dice 0.9297

These results are not directly comparable to the current joint ACDC+MSCMR+ISBI scribble tracker setting.

## ACDC Scribble Tracker V1

ACDC-only scribble tracker v1 found:

- Tracker-only with thresholding could show useful propagation signal.
- Naive detector-first or simple confidence-aware merge did not reliably turn that signal into final improvement.
- Tracker-only propagation improved over detector-only in this ACDC-only setting.
- The old reliability-aware temporal merge improvement was from a test-split sweep and should be treated as oracle-style, not as a strict validation-selected merge result.
- This positive ACDC-only result motivated promotion to joint ACDC+MSCMR+ISBI, where the same style of merge did not hold up.

Historical paths:

- `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/tests/scribble_sam3_tracker_image_init_v1_reliability_temporal_merge_sweep_with_tracker_only/`
- `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/tests/scribble_sam3_tracker_image_init_v1_oracle_diagnosis/`

Key ACDC-only scribble results under the strict validation-selected merge protocol:

| Setting | Dice | Path |
| --- | ---: | --- |
| detector-only | 0.9130 | `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/tests/test_official_detector/eval_3d/` |
| tracker-only | 0.9153 | `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/tests/test_candidates_val_config_protocol/eval_3d_tracker/` |
| validation-selected reliability merge | 0.9093 | `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/tests/test_merge_fixed_val_config/eval_3d/` |

Historical non-strict result:

- The removed test-sweep merge directory previously gave 0.9205 Dice, but it selected merge hyperparameters on the test split. Do not use it as strict paper evidence.

## Scope Boundary

The joint setting has a stronger detector baseline and different dataset mixture. In that setting, tracker v1/v2/v3 and reliability-aware merge do not exceed the strong `checkpoint_20.pt` detector baseline.

Use this file for ACDC-only tracker motivation and ablation discussion. Use `results-current.md` for joint tracker claims.

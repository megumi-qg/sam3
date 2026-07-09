# Single-Dataset Experiments

This file maps non-ACDC single-dataset experiments under `gq_experiment/mscmr/` and `gq_experiment/isbi/`.

These experiments are not automatically old. They were run after joint ACDC+MSCMR+ISBI tracker merge degraded, to diagnose whether MSCMR and ISBI behave differently when trained and evaluated as separate datasets.

## Current Interpretation

MSCMR/ISBI now have single-dataset detector and tracker diagnostic runs. The detector runs remain the official detector baselines. The tracker runs are useful for diagnosing whether propagation adds value per dataset. Strict merge results should use validation-selected hyperparameters and fixed test application.

Use them to answer:

- whether single-dataset detector fine-tuning is stronger or weaker than the joint detector on that dataset
- whether the joint setting introduces dataset-interaction issues
- whether MSCMR/ISBI have exploitable detector/tracker complementarity under per-dataset tracker training

Current outcome under validation-selected merge hyperparameters:

- MSCMR shows a small positive tracker-only gain over the single-dataset detector, but fixed validation-selected merge does not beat the detector.
- ISBI tracker-only is substantially weaker than detector-only, and fixed validation-selected merge does not beat the detector.
- Do not claim a broad MSCMR+ISBI merge improvement from this evidence.

## Directory Map

| Directory | Meaning | Current relevance |
| --- | --- | --- |
| `gq_experiment/mscmr/expert_scribble_lora/` | MSCMR expert-scribble detector LoRA run. | Detector-only diagnostic after joint merge degradation. |
| `gq_experiment/mscmr/scribble_tracker_image_init_v1/` | MSCMR single-dataset scribble tracker run. | Tracker checkpoint source for per-dataset merge diagnostics. |
| `gq_experiment/mscmr/full_lora/` | MSCMR full-supervised detector LoRA run. | Baseline/debug context; no test eval found in the current scan. |
| `gq_experiment/isbi/full_lora/` | ISBI full-supervised detector LoRA run. | Full-label detector reference. |
| `gq_experiment/isbi/scribble_lora/` | ISBI scribble detector LoRA run. | Detector-only diagnostic after joint merge degradation. |
| `gq_experiment/isbi/scribble_tracker_image_init_v1/` | ISBI single-dataset scribble tracker run. | Tracker checkpoint source for per-dataset merge diagnostics. |

## Known Test Results

| Experiment | Eval path | Dice | Note |
| --- | --- | ---: | --- |
| `mscmr/expert_scribble_lora` | `inference_test_0.7/evaluation_results_mscmr.json` | 0.8816 | detector-only result |
| `isbi/full_lora` | `inference_test/evaluation_results_isbi.json` | 0.8670 | full-label detector result; HD95/NSD are null in this output |
| `isbi/scribble_lora` | `inference_test/evaluation_results_isbi.json` | 0.8378 | detector-only scribble result |

## Tracker Merge Diagnostics

Main test outputs, using best validation-Dice tracker checkpoints. Merge hyperparameters are selected on `val_merge_sweep` and fixed on `test_merge_fixed_val_config`:

| Dataset | Method | Eval path | Dice | IoU | HD95 | NSD | Note |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| MSCMR | detector | `gq_experiment/mscmr/tracker_eval/single_dataset_v1_val_dice/eval_3d_official_detector/evaluation_results_mscmr.json` | 0.8816 | 0.7933 | 7.91 | 0.8336 | official detector `predictions.pkl` converted to COCO results |
| MSCMR | tracker-only | `gq_experiment/mscmr/tracker_eval/single_dataset_v1_val_dice/eval_3d_tracker/evaluation_results_mscmr.json` | 0.8855 | 0.7990 | 7.09 | 0.8425 | small gain over detector |
| MSCMR | val-config merge | `gq_experiment/mscmr/tracker_eval/single_dataset_v1_val_dice/test_merge_fixed_val_config/eval_3d/evaluation_results_mscmr.json` | 0.8804 | 0.7913 | 10.45 | 0.8289 | validation-selected merge underperforms detector |
| ISBI | detector | `gq_experiment/isbi/tracker_eval/single_dataset_v1_val_dice/eval_3d_official_detector/evaluation_results_isbi.json` | 0.8378 | 0.7282 | nan | nan | official detector `predictions.pkl` converted to COCO results |
| ISBI | tracker-only | `gq_experiment/isbi/tracker_eval/single_dataset_v1_val_dice/eval_3d_tracker/evaluation_results_isbi.json` | 0.7496 | 0.6049 | nan | nan | substantially below detector |
| ISBI | val-config merge | `gq_experiment/isbi/tracker_eval/single_dataset_v1_val_dice/test_merge_fixed_val_config/eval_3d/evaluation_results_isbi.json` | 0.8312 | 0.7182 | nan | nan | recovers toward detector but does not beat it |

Validation-selected merge configs:

- MSCMR: detector threshold 0.7, tracker threshold 0.7, tracker bias 0.0, temporal weight 0.2, disagreement penalty 0.05, margin 0.03; validation source counts `detector=204`, `tracker=11`, `none=10`.
- ISBI: detector threshold 0.7, tracker threshold 0.7, tracker bias 0.0, temporal weight 0.1, distance penalty 0.01, disagreement penalty 0.02, margin 0.0; validation source counts `detector=280`, `tracker=124`, `none=222`.

Caveat:

- The earlier test-sweep merge directories were removed. Use `val_merge_sweep/best_config.json` for selected hyperparameters and `test_merge_fixed_val_config/` for strict test results.
- ISBI HD95/NSD are `nan` because the run did not provide spacing to the evaluator.

## Tracker Training Preparation

Single-dataset tracker training should use the corresponding single-dataset detector checkpoint to build pseudo seeds:

- MSCMR detector checkpoint: `gq_experiment/mscmr/expert_scribble_lora/checkpoints/val_mscmr_segmentation_coco_eval_segm_AP.pt`
- ISBI detector checkpoint: `gq_experiment/isbi/scribble_lora/checkpoints/val_isbi_segmentation_coco_eval_segm_AP.pt`

Important ISBI caveat:

- ISBI category ids are `1=peripheral zone` and `2=central gland`.
- Pseudo-seed generation must use COCO category names, not ACDC's category-id prompt map.
- If revisiting joint tracker training, regenerate the ISBI pseudo seed bank after this fix before retraining.

Training command convention:

- Long-running tracker training commands should be launched with `nohup` and a named log file.

Prepared tracker configs:

- `sam3/train/configs/mscmr/scribble_tracker_image_init_v1.yaml`
- `sam3/train/configs/isbi/scribble_tracker_image_init_v1.yaml`

Important config fix:

- Add `trainer.loss.mscmr: ${tracker_train.loss}` in the MSCMR tracker config.
- Add `trainer.loss.isbi: ${tracker_train.loss}` in the ISBI tracker config.
- Without these dataset-specific keys, the trainer routes batches to `default` and uses `DummyLoss`, causing the first backward pass to fail because the loss has no grad function.

Prepared helper scripts:

- `gq_scripts/preprocess/run_single_dataset_tracker_pseudo_seed_banks.sh`
- `gq_scripts/train/run_single_dataset_tracker_scribble_image_init_v1.sh`

Generated pseudo seed banks:

- `gq_experiment/mscmr/expert_scribble_lora/pseudo_seed_bank/scribble_pseudo_seed_video_annotations.coco.json`
- `gq_experiment/isbi/scribble_lora/pseudo_seed_bank/scribble_pseudo_seed_video_annotations.coco.json`

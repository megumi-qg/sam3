# Current State

Last reorganized: 2026-07-08.

This file is the rolling project memory. Update this file for active results, current checkpoints, and next steps. Do not duplicate these current conclusions in topic files.

## Current Research Stage

The current work is diagnosing when tracker propagation and detector/tracker merge help or hurt scribble-supervised segmentation.

The research path so far:

1. ACDC-only scribble detector was extended with a tracker and merge rules.
2. ACDC-only tracker-only propagation improved over detector-only, but the old ACDC merge improvement came from test-split sweep and is not a strict validation-selected merge result.
3. The same idea was promoted to joint ACDC+MSCMR+ISBI fine-tuning.
4. In the joint setting, tracker-only results are weaker than detector-only results, and fair reliability-aware merge is also weaker than the strongest detector-only baseline.
5. Because joint merge degraded, single-dataset MSCMR and ISBI detector/tracker merge runs were used to diagnose whether the non-ACDC datasets behave differently when trained separately.

Current high-level tasks:

- Strong joint scribble detector baseline.
- Pseudo-seed tracker trained from detector predictions.
- Fair detector/tracker merge evaluation.
- Diagnosis of whether tracker propagation adds value beyond a strong detector.
- Single-dataset MSCMR/ISBI tracker-merge diagnostics.

Branch boundary:

- `gq_experiment/cmpb/` belongs to another branch of experiments and should not be mixed into the current branch's ACDC/MSCMR/ISBI evidence chain.
- ACDC-only experiments under `gq_experiment/acdc/` are documented in `docs/agent/acdc-experiments.md`.
- Non-ACDC single-dataset experiments under `gq_experiment/mscmr/` and `gq_experiment/isbi/` are documented in `docs/agent/single-dataset-experiments.md`.

## Strong Detector Baseline

Experiment root:

- `gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/`

Important checkpoints:

- `checkpoints/checkpoint_20.pt`
- `checkpoints/val_macro_segmentation_coco_eval_segm_AP.pt`

Current strongest detector baseline:

| Detector ckpt | Macro | ACDC | MSCMR | ISBI | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `checkpoint_20.pt` | 0.8940 | 0.9200 | 0.9016 | 0.8602 | strongest current detector Dice baseline |
| `val_macro_segmentation_coco_eval_segm_AP.pt` | 0.8887 | 0.9122 | 0.8870 | 0.8668 | used as v2/v3 tracker backbone |

Source summary:

- `gq_experiment/joint/tracker_eval/v3_temporal_area_centroid/summary_test.tsv`

## Pseudo Seed Banks

Builder:

- `gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py`

Joint scripts:

- `gq_scripts/preprocess/run_joint_tracker_pseudo_seed_banks.sh`
- `gq_scripts/preprocess/run_joint_tracker_pseudo_seed_banks_val_macro.sh`

Seed banks are detector-derived pseudo masks. Full labels are used for evaluation, not for strict scribble tracker seed generation.

Important distinction:

- `tracker_auto_seed_inference.py` writes `detector_predictions_segm.json` as raw detector candidates inside the tracker workflow.
- That file is not the same as the official detector-only baseline output.
- Fair merge must use the official detector pipeline's final masks as the detector branch.

## Tracker Status

The current trained tracker line is joint ACDC+MSCMR+ISBI.

Relevant configs:

- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v2_val_macro.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v3_temporal_area_centroid_val_macro.yaml`

Experiment roots:

- `gq_experiment/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1/`
- `gq_experiment/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v2_val_macro/`
- `gq_experiment/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v3_temporal_area_centroid_val_macro/`

Current v3 checkpoints include:

- `checkpoints/val_macro_segmentation_coco_eval_segm_AP.pt`
- `checkpoints/val_macro_segmentation_dice.pt`

V3 validation Dice sweep found those two best checkpoints tied:

| Rank | Checkpoint | Macro Dice | ACDC | MSCMR | ISBI |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `val_macro_segmentation_dice` | 0.8730 | 0.8947 | 0.8958 | 0.8286 |
| 2 | `val_macro_segmentation_coco_eval_segm_AP` | 0.8730 | 0.8947 | 0.8958 | 0.8286 |

Source:

- `gq_experiment/joint/tracker_eval/v3_temporal_area_centroid/checkpoint_val_dice_sweep/summary_two_val_macro_ckpts.tsv`

## Latest Test Summary

Source:

- `gq_experiment/joint/tracker_eval/v3_temporal_area_centroid/summary_test.tsv`

| Method | Macro | ACDC | MSCMR | ISBI |
| --- | ---: | ---: | ---: | ---: |
| `detector_checkpoint_20` | 0.8940 | 0.9200 | 0.9016 | 0.8602 |
| `detector_val_macro` | 0.8887 | 0.9122 | 0.8870 | 0.8668 |
| `v2_tracker_only` | 0.8770 | 0.9065 | 0.8888 | 0.8356 |
| `v2_fair_reliability_merge` | 0.8794 | 0.9067 | 0.8906 | 0.8408 |
| `v3_tracker_only_val_dice_ckpt` | 0.8755 | 0.9032 | 0.8901 | 0.8333 |
| `v3_fair_reliability_merge` | 0.8787 | 0.9060 | 0.8877 | 0.8422 |

Current conclusion:

- V2/V3 tracker and reliability-aware merge do not exceed the strong `checkpoint_20.pt` detector baseline.
- Even on the ACDC subset inside joint evaluation, merge does not beat the joint detector's ACDC result.
- V3 temporal area/centroid losses did not produce a clear improvement over V2.
- The old ACDC-only scribble merge result reached 0.9205 Dice only under a test-split sweep. Under the strict validation-selected protocol, ACDC tracker-only reaches 0.9153 versus detector-only 0.9130, but merge drops to 0.9093.
- Current joint tracker work is useful for diagnosis, ablation, temporal consistency analysis, and future method design, but it is not the main performance-improving result.

## Single-Dataset Follow-Up

After joint tracker merge degraded, MSCMR and ISBI single-dataset runs were used as diagnostics. Per-dataset tracker configs initially routed losses to `default`, which used `DummyLoss`; the fixed configs explicitly map `trainer.loss.mscmr` and `trainer.loss.isbi` to `tracker_train.loss`.

Relevant paths:

- `gq_experiment/mscmr/expert_scribble_lora/`
- `gq_experiment/isbi/full_lora/`
- `gq_experiment/isbi/scribble_lora/`
- `gq_experiment/mscmr/scribble_tracker_image_init_v1/`
- `gq_experiment/isbi/scribble_tracker_image_init_v1/`

Single-dataset and ACDC-only tracker merge test summary, using official detector final masks converted from detector `predictions.pkl` as the detector branch. Merge hyperparameters are selected on validation and then fixed on test:

| Dataset | Detector Dice | Tracker Dice | Merge Dice | Notes |
| --- | ---: | ---: | ---: | --- |
| ACDC | 0.9130 | 0.9153 | 0.9093 | tracker-only improves, but validation-selected merge underperforms detector |
| MSCMR | 0.8816 | 0.8855 | 0.8804 | tracker-only improves, but validation-selected merge underperforms detector |
| ISBI | 0.8378 | 0.7496 | 0.8312 | tracker-only is weak; validation-selected merge recovers toward detector but does not beat it |

MSCMR 3D metrics include HD95/NSD: detector HD95 7.91, NSD 0.8336; tracker HD95 7.09, NSD 0.8425; validation-selected merge HD95 10.45, NSD 0.8289. ISBI HD95/NSD are unavailable in this run because no spacing file was provided and the evaluator records them as `nan`.

Important correction:

- Previous ACDC test-sweep merge result `0.9205` and the initial MSCMR/ISBI test-sweep results were oracle-style because merge hyperparameters were selected on the evaluated test split.
- The current strict protocol is `val_merge_sweep` for hyperparameter selection and `test_merge_fixed_val_config` for test evaluation.
- Under this strict protocol, current merge rules do not improve over detector-only; tracker-only still shows useful signal on ACDC and MSCMR.

See `docs/agent/single-dataset-experiments.md` for the directory map and known detector-only results.

## Next Steps

Recommended next steps:

1. Analyze failure cases where tracker replaces or merges with detector output and hurts Dice.
2. Run oracle or per-slice complementarity analysis to quantify whether detector/tracker have exploitable disagreement.
3. Replace the current merge selector with a validation-stable uncertainty/quality estimator before making a positive merge claim.
4. For ISBI, diagnose why tracker propagation is much weaker than detector-only before making a positive merge claim.
5. For paper writing, separate tracker-only temporal propagation gains from merge-selection results; do not cite test-sweep merge as strict evidence.
6. Keep `checkpoint_20.pt` as the main joint detector baseline unless a newer detector run beats it under the same evaluation protocol.

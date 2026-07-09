# Tracker Results

This file records current scribble tracker results. It includes the joint ACDC+MSCMR+ISBI line and the newer MSCMR/ISBI single-dataset diagnostics. It should not overwrite the ACDC-only positive merge result in `results-archive.md`.

## Configs

- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v2_val_macro.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v3_temporal_area_centroid_val_macro.yaml`

## Test Summary

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

## Conclusion

- The strongest detector remains `checkpoint_20.pt`.
- V2/V3 tracker-only results do not beat the detector baseline.
- V2/V3 fair reliability-aware merge does not beat the detector baseline.
- The joint ACDC subset also does not reproduce the ACDC-only reliability-aware merge improvement.
- V3 temporal area/centroid losses did not clearly improve over V2.
- Joint tracker value is diagnostic and methodological unless a later selection method closes the gap.

## Validation-Selected Merge Diagnostics

Source roots:

- `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/tests/`
- `gq_experiment/mscmr/tracker_eval/single_dataset_v1_val_dice/`
- `gq_experiment/isbi/tracker_eval/single_dataset_v1_val_dice/`

Main test summary:

| Dataset | Detector Dice | Tracker Dice | Merge Dice | Interpretation |
| --- | ---: | ---: | ---: | --- |
| ACDC | 0.9130 | 0.9153 | 0.9093 | tracker-only improves, merge selector hurts |
| MSCMR | 0.8816 | 0.8855 | 0.8804 | tracker-only improves, merge selector hurts |
| ISBI | 0.8378 | 0.7496 | 0.8312 | tracker-only weak, merge does not beat detector |

Notes:

- Detector branch uses official detector final masks converted from detector `predictions.pkl`, not `detector_predictions_segm.json` from tracker inference.
- Merge hyperparameters are selected on `val_merge_sweep/best_config.json` and applied on test via `test_merge_fixed_val_config/`.
- ACDC validation merge source counts: `detector=706`, `tracker=177`, `none=59`.
- MSCMR validation merge source counts: `detector=204`, `tracker=11`, `none=10`.
- ISBI validation merge source counts: `detector=280`, `tracker=124`, `none=222`.
- ISBI HD95/NSD are unavailable in this run because no spacing file was supplied.
- Earlier test-sweep merge directories were removed from the active result paths because they selected hyperparameters on test.

## Useful Result Paths

- `gq_experiment/joint/tracker_eval/v2_val_macro/summary_test.tsv`
- `gq_experiment/joint/tracker_eval/v2_val_macro_checkpoint_val_dice_sweep/summary_two_val_macro_ckpts.tsv`
- `gq_experiment/joint/tracker_eval/v3_temporal_area_centroid/summary_test.tsv`
- `gq_experiment/joint/tracker_eval/v3_temporal_area_centroid/checkpoint_val_dice_sweep/summary_two_val_macro_ckpts.tsv`
- `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/tests/test_merge_fixed_val_config/eval_3d/evaluation_results_acdc.json`
- `gq_experiment/mscmr/tracker_eval/single_dataset_v1_val_dice/test_merge_fixed_val_config/eval_3d/evaluation_results_mscmr.json`
- `gq_experiment/isbi/tracker_eval/single_dataset_v1_val_dice/test_merge_fixed_val_config/eval_3d/evaluation_results_isbi.json`

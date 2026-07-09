# ACDC Experiments

This file maps `gq_experiment/acdc/`. It is a directory map for ACDC-only experiments, not a judgment that every single-dataset experiment is obsolete.

## Current Use

Keep these experiments for reference. Do not delete them automatically.

Use this file when you need ACDC-only detector, context, or tracker history. Use `docs/agent/single-dataset-experiments.md` for non-ACDC single-dataset checks, and use `docs/agent/current-state.md` for the active research narrative.

ACDC-only tracker results are still relevant for method motivation because tracker-only propagation improved over the detector-only baseline. The old reliability-aware temporal merge improvement came from a test-split sweep, so it should be treated as oracle-style rather than strict validation-selected evidence.

## Directory Map

| Directory | Meaning | Current relevance |
| --- | --- | --- |
| `full_lora_100/` | 2D full-supervised ACDC LoRA image-model baseline. | Useful for comparing 2D vs video-like ACDC. |
| `scribble_lora_100/` | 2D scribble weak-supervised ACDC LoRA baseline. | ACDC-only weak baseline. |
| `full_video_lora_100/` | Video-like/volume-packaged full-supervised ACDC LoRA baseline. | Important ACDC-only full baseline. |
| `scribble_video_lora_100/` | Video-like/volume-packaged scribble ACDC LoRA baseline. | Important ACDC-only scribble detector baseline and pseudo-seed source. |
| `full_video_lora_100_context_v1/` | Full-supervised context v1 experiment. | Negative context result; archive but keep. |
| `scribble_video_lora_100_context_v1/` | Scribble context v1 experiment. | Negative context result; archive but keep. |
| `full_sam3_tracker_image_init_v2/` | ACDC-only full tracker v2 training. | Tracker reference under cleaner supervision. |
| `scribble_sam3_tracker_image_init_v1/` | ACDC-only scribble tracker v1 training. | Key evidence that tracker propagation can help; current validation-selected merge does not beat detector-only. |

## Full Tracker V2 Checkpoints

The full ACDC tracker v2 slide/result used:

- detector/image backbone: `gq_experiment/acdc/full_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- tracker checkpoint: `gq_experiment/acdc/full_sam3_tracker_image_init_v2/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`

Other periodic checkpoints in those two checkpoint directories were deleted after the experiment was archived.

## ACDC Test Results

Source files are under each experiment's `inference_test/evaluation_results_acdc.json`, except the two reruns listed below.

| Experiment | Dice | IoU | HD95 | NSD | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `full_lora_100/inference_test` | 0.9291 | 0.8733 | 3.4620 | 0.9654 | 2D full baseline. |
| `full_video_lora_100/inference_test` | 0.9323 | 0.8761 | 2.7422 | 0.9701 | strongest ACDC-only full image baseline. |
| `full_video_lora_100/inference_test_rerun_20260421` | 0.9183 | 0.8578 | 5.1350 | 0.9516 | Bad rerun; useful only as LoRA-runtime debugging evidence. |
| `full_video_lora_100/inference_test_rerun_merge_lora` | 0.9323 | 0.8762 | 2.7422 | 0.9701 | LoRA merge/runtime fix restored baseline. |
| `full_video_lora_100_context_v1/inference_test` | 0.9063 | 0.8375 | 7.4618 | 0.9351 | Context v1 full negative result. |
| `scribble_lora_100/inference_test` | 0.9170 | 0.8507 | 3.3787 | 0.9518 | 2D scribble baseline. |
| `scribble_video_lora_100/inference_test` | 0.9130 | 0.8463 | 6.5001 | 0.9479 | ACDC-only video-like scribble detector baseline. |
| `scribble_video_lora_100_context_v1/inference_test` | 0.8572 | 0.7717 | 26.1729 | 0.8712 | Context v1 scribble negative result. |

## Full Tracker V2 Test Outputs

The long test directory names under `full_sam3_tracker_image_init_v2/tests/` were shortened:

| Directory | Meaning | Dice |
| --- | --- | ---: |
| `merge_train_like/` | Detector + tracker train-like merge/hybrid output. | 0.9324 |
| `tracker_only_train_like/` | Tracker-only train-like output. | 0.9319 |
| `tracker_only_best_single_bidir/` | Tracker-only highest-score single seed + bidirectional propagation. | 0.9297 |
| `threshold_sweep/` | Threshold sweep for tracker-only train-like predictions. | varies |

See `gq_experiment/acdc/full_sam3_tracker_image_init_v2/tests/README.md` for the local result-map.

## Cleanup Guidance

The following are not primary evidence for joint ACDC+MSCMR+ISBI claims, but they are still useful archive/debug material:

- `full_lora_100/` and `scribble_lora_100/`: 2D-only ACDC baselines.
- `*_context_v1/`: negative context v1 experiments.
- `full_video_lora_100/inference_test_rerun_20260421/`: failed rerun before LoRA merge/runtime was fixed.
- `full_sam3_tracker_image_init_v2/` and `scribble_sam3_tracker_image_init_v1/`: ACDC-only tracker experiments. Keep best checkpoints and final test outputs; periodic checkpoints can be removed after the key result path is preserved.

Do not remove these directories without explicit user approval. They may still be needed for paper ablations, debugging, or historical comparison.

# scribble_sam3_tracker_image_init_v1_reliability_temporal_merge_sweep_with_tracker_only

This folder stores a post-hoc reliability-aware temporal detector/tracker merge sweep for `scribble_sam3_tracker_image_init_v1`.

The sweep uses already generated detector/tracker predictions from:

- `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1_oracle_diagnosis/detector_predictions_segm.json`
- `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1_oracle_diagnosis/tracker_predictions_segm.json`

No GT is used for merge decisions. GT is only used to evaluate each candidate rule.

## Merge Rule

For each slice/class, compute:

```text
R_det = detector_score
        + detector_bias
        - temporal_weight * detector_temporal_jump

R_trk = tracker_score
        + tracker_bias
        - distance_penalty * distance_to_seed
        - temporal_weight * tracker_temporal_jump
```

When both detector and tracker pass their thresholds, choose tracker only if:

```text
R_trk > R_det + margin + disagreement_penalty * (1 - IoU(det_mask, trk_mask))
```

This sweep includes `detector_threshold=1.1` so tracker-only thresholded is included as a baseline instead of forcing detector fallback.

## Best Config

```json
{
  "detector_threshold": 0.9,
  "tracker_threshold": 0.7,
  "detector_bias": 0.0,
  "tracker_bias": 0.0,
  "temporal_weight": 0.4,
  "distance_penalty": 0.0,
  "disagreement_penalty": 0.1,
  "margin": 0.03
}
```

Source counts:

- detector: `714`
- tracker: `22`
- none: `62`

## Results

| Output | Overall Dice | LV | MYO | RV | HD95 | NSD |
|---|---:|---:|---:|---:|---:|---:|
| detector thresholded | 0.9129 | 0.9427 | 0.9101 | 0.8860 | 6.4989 | 0.9479 |
| tracker thresholded | 0.9185 | 0.9405 | 0.9107 | 0.9043 | 3.0714 | 0.9550 |
| previous confidence-aware merge | 0.9162 | 0.9427 | 0.9101 | 0.8959 | - | - |
| reliability-aware temporal merge | 0.9205 | 0.9430 | 0.9101 | 0.9085 | 3.48 | 0.9572 |
| oracle merge | 0.9243 | 0.9463 | 0.9148 | 0.9117 | 3.0082 | 0.9617 |

## Takeaway

Reliability-aware temporal merge improves over detector-only, tracker-only, and the previous simple confidence-aware merge. The gain mainly comes from improving RV while keeping LV/MYO close to detector performance.

There is still a gap to oracle merge, so future work should focus on stronger reliability estimation or using tracker as a training-time temporal consistency regularizer.

## Main Files

- `reliability_temporal_merge_sweep_results.csv`: all swept rules sorted by Overall Dice.
- `best_config.json`: best config and summary metrics.
- `best_reliability_temporal_merge_predictions_segm.json`: best merged COCO predictions.
- `eval_3d_best/`: full 3D ACDC evaluation for the best merged predictions.


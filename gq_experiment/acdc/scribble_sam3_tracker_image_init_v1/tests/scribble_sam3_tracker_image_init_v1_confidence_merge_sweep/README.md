# scribble_sam3_tracker_image_init_v1_confidence_merge_sweep

这是 `scribble_sam3_tracker_image_init_v1` 的第一版规则式 confidence-aware merge 参数搜索。

## 规则

对每个 slice / class，同时考虑 detector mask 和 tracker mask。

```text
S_det = detector_score
S_trk = tracker_score - distance_penalty * distance_to_seed
```

如果 detector 和 tracker 都超过各自阈值，则只有当：

```text
S_trk > S_det + margin
```

才使用 tracker，否则使用 detector。

如果只有一方超过阈值，就使用该分支；两者都不过阈值则输出空 mask。

## 搜索空间

- detector threshold: `0.6, 0.7, 0.8`
- tracker threshold: `0.6, 0.7, 0.8`
- distance penalty: `0, 0.01, 0.02, 0.05`
- margin: `0, 0.03, 0.05, 0.1`
- seed selection: `earliest_above_threshold`
- max conditioning frames: `1`
- propagation mode: `forward_only`

## Best Config

```json
{
  "detector_threshold": 0.8,
  "tracker_threshold": 0.7,
  "distance_penalty": 0.02,
  "margin": 0.0
}
```

Best source counts:

- detector: `640`
- tracker: `98`
- none: `60`

## 3D ACDC Result

| Output | Overall Dice | LV | MYO | RV |
|---|---:|---:|---:|---:|
| confidence-aware merge best | 0.9162 | 0.9427 | 0.9101 | 0.8959 |

## Interpretation

这一版简单 score-based merge 比 detector thresholded 结果更好，但仍低于 tracker thresholded 与 oracle merge。

关键原因是 detector score 和 tracker score 没有充分校准，单纯比较 score 很难识别所有 tracker 更优的 slice。下一步应加入 continuity / uncertainty 信号，例如 area jump、centroid jump、detector-tracker agreement。

## Files

- `confidence_merge_sweep_results.csv`: 全部参数组合结果。
- `best_config.json`: 最优参数及内部评估结果。
- `best_confidence_merge_predictions_segm.json`: 最优参数对应的 COCO 预测。
- `eval_3d_best/`: 最优参数的正式 3D ACDC 评估结果。

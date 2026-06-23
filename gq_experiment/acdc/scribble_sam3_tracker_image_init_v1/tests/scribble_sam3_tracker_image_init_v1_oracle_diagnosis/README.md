# scribble_sam3_tracker_image_init_v1_oracle_diagnosis

这是针对 `scribble_sam3_tracker_image_init_v1` 的 detector / tracker / oracle merge 诊断结果。

## 推理设置

- image checkpoint: `gq_experiment/acdc/scribble_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- tracker checkpoint: `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- seed selection: `earliest_above_threshold`
- `detector_condition_threshold=0.7`
- `detector_output_threshold=0.7`
- `tracker_detection_threshold=0.7`
- `max_cond_frames=1`
- `propagation_mode=forward_only`

注意：oracle merge 只用于诊断上限，使用 GT 在 detector / tracker 中逐 slice 选择 Dice 更高者，不能作为真实推理方法。

## 逐 slice 诊断

- 总 slice-class 数：`798`
- tracker 优于 detector：`240 / 798 = 30.1%`
- detector 优于或等于 tracker：`558 / 798 = 69.9%`
- mean detector 2D Dice：`0.8901`
- mean tracker 2D Dice：`0.8943`
- mean oracle 2D Dice：`0.9039`
- oracle 相对 detector 的 2D Dice gain：`+0.0139`

按类别：

| Class | tracker better | tracker better ratio | oracle 2D gain |
|---|---:|---:|---:|
| LV | 60 / 266 | 22.6% | +0.0066 |
| MYO | 115 / 266 | 43.2% | +0.0050 |
| RV | 65 / 266 | 24.4% | +0.0300 |

## 3D ACDC 指标

| Output | Overall Dice | LV | MYO | RV |
|---|---:|---:|---:|---:|
| detector thresholded | 0.9129 | 0.9427 | 0.9101 | 0.8860 |
| tracker thresholded | 0.9185 | 0.9405 | 0.9107 | 0.9043 |
| oracle merge | 0.9243 | 0.9463 | 0.9148 | 0.9117 |

## 结论

Tracker 确实存在互补价值，尤其对 RV 的潜在帮助最大。当前 detector-first hybrid 没有明显提升，并不是因为 tracker 完全无效，而是因为固定阈值 + detector-first 规则没有识别出 tracker 更好的那些 slice。

下一步更值得做的是 confidence-aware / uncertainty-aware selection，或者把 tracker 用作训练时的 slice consistency regularizer，而不是继续使用简单 detector-first merge。

## 主要文件

- `slice_oracle_diagnosis.csv`: 逐 slice / class 的 detector、tracker、oracle Dice 诊断。
- `oracle_summary.json`: 汇总统计。
- `detector_predictions_segm.json`: thresholded detector COCO 预测。
- `tracker_predictions_segm.json`: thresholded tracker COCO 预测。
- `oracle_merge_predictions_segm.json`: oracle merge COCO 预测。
- `eval_3d_detector/`: detector 3D 评估。
- `eval_3d_tracker/`: tracker 3D 评估。
- `eval_3d_oracle/`: oracle merge 3D 评估。
- `visualizations/`: tracker better / detector better 的可视化样例。

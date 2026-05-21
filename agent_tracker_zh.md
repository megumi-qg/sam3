# SAM3 Tracker Agent Gene

面向后续 AI agent 的高密度控制文档。用户可读的长说明见 `agent_tracker_user_zh.md`。

## Keywords

- 项目：SAM3 tracker, ACDC, 3D volume as video, single-object propagation
- 目标：`image detector + tracker propagation/refinement`，不是 tracker 替代 image model
- Full-supervised 主线：`full_video_lora_100` image baseline + tracker v1/v2
- Scribble 主线：不能用 full label 训练 tracker；只能用 scribble model pseudo seed + scribble valid region loss
- 关键类别：RV / MYO / LV，每个 `(volume, category)` 是一个单目标 tracking task

## Project Summary

- 当前 image-only full baseline：`Overall Dice = 0.9323`
- 当前最稳 full tracker 结果：tracker v2 train-like hybrid，`Overall Dice = 0.9324`
- tracker 的价值目前主要是传播、补全、时序一致性；不要默认 tracker-only 会超过 detector。
- full tracker v2 有进步：`best_single + bidirectional` 从 v1 的 `0.9185` 提升到 v2 的 `0.9297`
- scribble tracker v1 是负结果：可训练、可推理，但没有增强 scribble image baseline。

## Core Paths

环境：

```bash
source /home/gaoqi/anaconda3/etc/profile.d/conda.sh
conda activate sam3
```

数据：

- ACDC video data：`/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100`
- ACDC 2D PNG test：`/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100/test`

关键 checkpoint：

- full image baseline：`/home/gaoqi/sam3/gq_experiment/acdc/full_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- full tracker v1：`/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- full tracker v2：`/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_v2/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- scribble image baseline：`/home/gaoqi/sam3/gq_experiment/acdc/scribble_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- scribble tracker v1：`/home/gaoqi/sam3/gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`

关键代码：

- tracker train adapter：`sam3/model/sam3_tracker_train_adapter.py`
- tracker loss：`sam3/train/loss/sam3_tracker_loss.py`
- data object / seed mask：`sam3/train/data/sam3_image_dataset.py`
- collator：`sam3/train/data/collator.py`
- geometry transforms：`sam3/train/transforms/basic_for_api.py`
- auto seed inference：`gq_scripts/evaluate/tracker_auto_seed_inference.py`
- eval wrapper：`gq_scripts/evaluate/run_tracker_auto_seed_inference_and_eval.sh`

## Full Tracker Results

v1 config：`sam3/train/configs/acdc/full_sam3_tracker_image_init.yaml`

- 训练：`full_video_lora_100` 初始化 backbone，earliest visible clean GT seed，forward-like 分布。
- train-like hybrid：`Overall Dice = 0.9324`
- tracker-only train-like：`Overall Dice = 0.9314`
- best-single bidirectional：`Overall Dice = 0.9185`
- 结论：v1 证明 tracker 能传播，但复杂推理策略 train-test gap 很大。

v2 config：`sam3/train/configs/acdc/full_sam3_tracker_image_init_v2.yaml`

- 训练：mixed init frame, noisy seed mask, random reverse time axis。
- best val AP：`0.7347` at epoch `46`
- train-like hybrid：`Overall Dice = 0.9324`
- tracker-only train-like：`Overall Dice = 0.9319`
- best-single bidirectional：`Overall Dice = 0.9297`
- 结论：v2 对中间 seed / 双向传播更鲁棒，但仍没有稳定明显超过 image-only baseline。

默认 full tracker 推理优先用：

```bash
CUDA_VISIBLE_DEVICES=0 \
TRACKER_CHECKPOINT_PATH=/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_v2/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt \
OUTPUT_DIR=/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_v2_auto_seed_test_train_like \
MAX_COND_FRAMES=1 \
CONDITIONING_SELECTION_STRATEGY=earliest_above_threshold \
PROPAGATION_MODE=forward_only \
bash gq_scripts/evaluate/run_tracker_auto_seed_inference_and_eval.sh
```

## Scribble Tracker Results

scribble image baseline：

- checkpoint：`gq_experiment/acdc/scribble_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- test：`Overall Dice = 0.9130`, `IoU = 0.8463`, `HD95 = 6.5001`, `NSD = 0.9479`

scribble tracker v1 config：`sam3/train/configs/acdc/scribble_sam3_tracker_image_init_v1.yaml`

- strict weak supervision：不得用 full label / full tracker checkpoint。
- image backbone 初始化来自 `scribble_video_lora_100`。
- conditioning seed 来自离线 high-confidence pseudo mask。
- supervision 仍来自 scribble `valid_mask`；`Sam3WeakTrackerLossWrapper` 只在非 ignore 区域算 loss。
- tri-state target：`1` foreground, `0` valid background, `255` ignore。
- `presence_weight = 0.0`，因为没有 scribble 不等于目标不存在。

pseudo seed bank：

- JSON：`/home/gaoqi/sam3/gq_experiment/acdc/scribble_video_lora_100/pseudo_seed_bank/scribble_tmi_pseudo_seed_video_annotations.coco.json`
- report：同目录 `_report.json`
- 生成脚本：`gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py`
- 已知统计：`num_seed_frames = 3358`，每个 annotation 至少一个 seed。

生成命令：

```bash
CUDA_VISIBLE_DEVICES=0 python gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py \
  --score_threshold 0.97 \
  --min_scribble_recall 0.8 \
  --max_other_scribble_overlap_px 0
```

scribble tracker v1 test：

- train-like hybrid：`Overall Dice = 0.9130`, `IoU = 0.8463`, `HD95 = 6.5149`, `NSD = 0.9480`
- tracker-only 原始无阈值：`Overall Dice = 0.9076`, `IoU = 0.8367`, `HD95 = 4.1681`, `NSD = 0.9405`
- tracker-only + `tracker_detection_threshold=0.7`：`Overall Dice = 0.9185`, `LV = 0.9405`, `MYO = 0.9107`, `RV = 0.9043`
- detector thresholded：`Overall Dice = 0.9129`; oracle merge：`Overall Dice = 0.9243`
- 规则版 confidence-aware merge 参数搜索最佳约 `0.9162`，低于 tracker-only thresholded 和 oracle merge；说明简单 score/threshold 规则没有可靠识别 tracker 更优 slice。
- 结论：v1 tracker 本身有可用传播信号，threshold 后超过 detector-only；当前问题主要在真实 merge / selection 规则，而不是 tracker 完全无效。

## Known Fixes

- LoRA 推理曾导致 image baseline 假性退化；若 `full_video_lora_100` 推理突然掉分，优先检查 LoRA merge / runtime 路径。
- scribble tracker 多卡曾报 `stack expects each tensor to be equal size`。
- 根因：`seed_segment` 没有跟 `segment` 一起经过 `crop / hflip / resize / pad`。
- 修复位置：`sam3/train/transforms/basic_for_api.py`，`seed_segment` 现在走同一套几何变换；scribble tri-state mask padding 保持 `255 ignore`。

## Strategy

- full tracker：默认以 v2 train-like hybrid 作为最稳主结果。
- full tracker 继续推进时，优先尝试 detector pseudo seed 参与训练、多 seed conditioning、image backbone LoRA 联合微调。
- scribble tracker：v1 已证明 tracker-only thresholded 有互补价值，但 naive detector-first hybrid 没有稳定转化为最终增益。
- scribble 主线更有希望的方向：confidence/uncertainty-aware merge、tracker 输出反哺 image model、跨切片一致性正则。
- 写论文时可将 scribble tracker v1 表述为：高置信 pseudo seed + scribble-gated propagation loss 能学到传播能力；关键瓶颈是如何可靠选择 detector vs tracker。

## AVOID

- 不要用 full label 训练 scribble tracker。
- 不要用 full-supervised tracker checkpoint 作为 scribble 主方法。
- 不要把 pseudo mask 当 dense GT；pseudo seed 只用于 conditioning 或高置信筛选。
- 不要把没有 scribble 的 slice 当作 absence label。
- 不要默认 `best_single + bidirectional` 优于 train-like forward-only。
- 不要只看 2D COCO AP 判断最终好坏；ACDC 最终看 3D Dice / IoU / HD95 / NSD。
- 不要在未检查 LoRA merge 路径时相信突然退化的 image baseline。
- 不要在修改 mask 数据结构后忘记同步 transforms、collator、loss、inference。

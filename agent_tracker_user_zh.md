# SAM3 Tracker 研究说明

本文档是给用户阅读的版本，用来保留实验脉络、方法解释和阶段性结论。给后续 AI agent 快速读取的高密度版本见 `agent_tracker_zh.md`。

## 研究定位

当前 tracker 主线不是为了让 tracker 单独替代 image model，而是为了构建：

- image model 负责逐切片检测和自动 seed
- tracker 负责跨切片传播、补全和时序一致性
- 最终希望 `image detector + tracker propagation/refinement` 优于单独的 image-only baseline

在 ACDC 任务中，一个 3D volume 被看作按切片排列的视频。当前 tracker 训练语义是单目标传播：一个样本对应一个 `(volume, category)`，类别包括 RV / MYO / LV。

## 默认路径

环境：

```bash
source /home/gaoqi/anaconda3/etc/profile.d/conda.sh
conda activate sam3
```

ACDC video-like 数据：

- `/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100`

主要 checkpoint：

- full image baseline：
  `/home/gaoqi/sam3/gq_experiment/acdc/full_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- full tracker v2：
  `/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_v2/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- scribble image baseline：
  `/home/gaoqi/sam3/gq_experiment/acdc/scribble_video_lora_100/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`
- scribble tracker v1：
  `/home/gaoqi/sam3/gq_experiment/acdc/scribble_sam3_tracker_image_init_v1/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt`

关键代码：

- `sam3/model/sam3_tracker_train_adapter.py`
- `sam3/train/loss/sam3_tracker_loss.py`
- `sam3/train/configs/acdc/full_sam3_tracker_image_init_v2.yaml`
- `sam3/train/configs/acdc/scribble_sam3_tracker_image_init_v1.yaml`
- `gq_scripts/evaluate/tracker_auto_seed_inference.py`
- `gq_scripts/evaluate/run_tracker_auto_seed_inference_and_eval.sh`

## Full-Supervised Tracker 结果

full image-only baseline：

- `Overall Dice = 0.9323`

tracker v1：

- 配置：`sam3/train/configs/acdc/full_sam3_tracker_image_init.yaml`
- 使用 `full_video_lora_100` image model 初始化 tracker backbone
- conditioning frame 固定为 earliest visible frame
- conditioning mask 使用干净 GT mask
- train-like hybrid：`Overall Dice = 0.9324`
- tracker-only train-like：`Overall Dice = 0.9314`
- best-single bidirectional：`Overall Dice = 0.9185`

v1 说明 tracker 并不是不会传播，但它只见过干净 earliest GT seed，所以面对 best-single、largest-single、bidirectional 这类更复杂测试策略时 train-test gap 很明显。

tracker v2：

- 配置：`sam3/train/configs/acdc/full_sam3_tracker_image_init_v2.yaml`
- 训练策略：mixed init frame、noisy seed mask、random reverse time axis
- best val COCO AP：`0.7347`
- train-like hybrid：`Overall Dice = 0.9324`
- tracker-only train-like：`Overall Dice = 0.9319`
- best-single bidirectional：`Overall Dice = 0.9297`

v2 的意义在于分布对齐更好：`best_single + bidirectional` 从 v1 的 `0.9185` 提升到 `0.9297`。但 v2 仍然没有稳定显著超过 full image-only baseline，因此 full tracker 主线目前的表述应谨慎：它接近并局部增强 baseline，而不是已经形成强增益。

## Scribble Tracker 弱监督方案

scribble 主线的目标是减少医生 dense 标注量，因此不能用 full label 训练 tracker，也不能把 full-supervised tracker checkpoint 当作主方法。

当前 strict weak-supervised tracker v1 的设计是：

- 固定已经训练好的 scribble image model：`scribble_video_lora_100`
- 在 train set 上离线生成 pseudo masks 和 confidence scores
- 只保留极高置信、且满足 scribble consistency 的 pseudo mask 作为 tracker conditioning seed
- 训练 tracker 时，conditioning 来自 pseudo seed
- 监督仍由 scribble `valid_mask` 控制
- 不把 pseudo mask 当 dense GT

对应方法可以写成：

- High-Precision Scribble-Compatible Seed Selection
- Pseudo-Conditioned Weak Tracker
- Scribble-Gated Propagation Loss
- Tracker-to-Image Pseudo Label Loop

## Pseudo Seed Bank

生成脚本：

- `gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py`

推荐命令：

```bash
source /home/gaoqi/anaconda3/etc/profile.d/conda.sh
conda activate sam3
CUDA_VISIBLE_DEVICES=0 python gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py \
  --score_threshold 0.97 \
  --min_scribble_recall 0.8 \
  --max_other_scribble_overlap_px 0
```

输出：

- `/home/gaoqi/sam3/gq_experiment/acdc/scribble_video_lora_100/pseudo_seed_bank/scribble_tmi_pseudo_seed_video_annotations.coco.json`
- `/home/gaoqi/sam3/gq_experiment/acdc/scribble_video_lora_100/pseudo_seed_bank/scribble_tmi_pseudo_seed_video_annotations.coco_report.json`

已知统计：

- `num_seed_frames = 3358`
- 所有 420 个 annotations 都至少有一个 seed
- 平均每个 annotation 约 8 个 seed frames

## Scribble Tracker v1 训练与修复

训练命令：

```bash
CUDA_VISIBLE_DEVICES=2,3 nohup python sam3/train/train.py \
  -c configs/acdc/scribble_sam3_tracker_image_init_v1.yaml \
  --use-cluster 0 \
  --num-gpus 2 \
  > scribble_sam3_tracker_image_init_v1.log 2>&1 &
```

曾出现的问题：

- 多卡 DataLoader collate 报错：`stack expects each tensor to be equal size`
- 根因：新增的 `seed_segment` 没有跟 `segment` 一起经过 `crop / hflip / resize / pad`
- 修复：`sam3/train/transforms/basic_for_api.py` 中已经让 `seed_segment` 走同一套几何变换
- 额外修复：scribble tri-state mask 的 padding 保持 `255 ignore`

修复后 2-GPU smoke 已经能完整跑过 train / val / evaluator。

## Scribble Tracker v1 结果

scribble image baseline：

- `Overall Dice = 0.9130`
- `IoU = 0.8463`
- `HD95 = 6.5001`
- `NSD = 0.9479`

scribble tracker v1 train-like hybrid：

- `Overall Dice = 0.9130`
- `IoU = 0.8463`
- `HD95 = 6.5149`
- `NSD = 0.9480`

scribble tracker v1 tracker-only：

- 原始无阈值：`Overall Dice = 0.9076`, `IoU = 0.8367`, `HD95 = 4.1681`, `NSD = 0.9405`
- 加 `tracker_detection_threshold=0.7` 后：`Overall Dice = 0.9185`
- thresholded per-class：`LV Dice = 0.9405`, `MYO Dice = 0.9107`, `RV Dice = 0.9043`

thresholded 诊断：

- detector thresholded：`Overall Dice = 0.9129`
- tracker thresholded：`Overall Dice = 0.9185`
- 规则版 confidence-aware merge 最优：`Overall Dice ≈ 0.9162`
- oracle merge：`Overall Dice = 0.9243`

当前结论：

- `scribble_sam3_tracker_image_init_v1` 不是完全负结果：tracker-only 加阈值后超过 detector thresholded。
- train-like hybrid 基本等于 scribble image baseline，没有增强。
- 当前瓶颈是 detector / tracker 的真实 merge 或 selection 规则，简单 detector-first 和规则版 confidence-aware merge 都没有把 tracker 的互补价值转化为最终增益。
- 这版方法适合作为 strict weak-supervised tracker 的第一版 attempt、消融基线，以及 confidence-aware merge / temporal consistency 的动机实验。

## 推理命令

full tracker v2 默认主结果：

```bash
CUDA_VISIBLE_DEVICES=0 \
TRACKER_CHECKPOINT_PATH=/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_v2/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt \
OUTPUT_DIR=/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_v2_auto_seed_test_train_like \
MAX_COND_FRAMES=1 \
CONDITIONING_SELECTION_STRATEGY=earliest_above_threshold \
PROPAGATION_MODE=forward_only \
bash gq_scripts/evaluate/run_tracker_auto_seed_inference_and_eval.sh
```

tracker-only 能力测试：

```bash
CUDA_VISIBLE_DEVICES=0 \
TRACKER_CHECKPOINT_PATH=/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_v2/checkpoints/val_acdc_segmentation_coco_eval_segm_AP.pt \
OUTPUT_DIR=/home/gaoqi/sam3/gq_experiment/acdc/full_sam3_tracker_image_init_v2_auto_seed_test_tracker_only_train_like \
MAX_COND_FRAMES=1 \
CONDITIONING_SELECTION_STRATEGY=earliest_above_threshold \
PROPAGATION_MODE=forward_only \
DETECTOR_OUTPUT_THRESHOLD=1.1 \
TRACKER_DETECTION_THRESHOLD=0.7 \
bash gq_scripts/evaluate/run_tracker_auto_seed_inference_and_eval.sh
```

## 下一步建议

full tracker：

- 引入 detector pseudo seed 参与训练
- 尝试多 seed conditioning
- 尝试 image backbone LoRA 联合微调
- 继续观察 RV 的 Dice 与 HD95 尾部异常

scribble tracker：

- 不建议继续期待当前 tracker-only 直接超过 scribble image baseline
- 更有价值的方向是用 tracker 做跨切片一致性约束
- 可以尝试把 tracker 输出作为 pseudo label 反哺 image model
- 可以设计 uncertainty-aware fusion，只在 tracker 高置信且 detector 不稳定区域使用 tracker
- 论文中当前 v1 更适合写成负结果和动机：naive pseudo-conditioned weak tracker 能学习传播，但还不足以提升 strong scribble image baseline

## 重要注意事项

- scribble 主线不要使用 full label 训练 tracker。
- scribble 主线不要使用 full-supervised tracker checkpoint 作为主方法。
- pseudo mask 不能直接当 dense GT，只能作为 conditioning seed 或高置信候选。
- 没有 scribble 的 slice 不能直接当 absence label。
- ACDC 最终主指标看 3D Dice / IoU / HD95 / NSD，不要只看 2D COCO AP。
- LoRA checkpoint 如果推理突然退化，优先检查 LoRA merge / runtime 路径。

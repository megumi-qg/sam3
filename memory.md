# Session Memory

给下一个 AI 接手用。系统性背景见 `agent_zh.md`、`agent_tracker_zh.md`、`agent_tracker_user_zh.md`；这里只保留当前阶段最关键的实验口径、结果、路径和下一步。

## Project Context

- 任务：SAM3 医学图像弱监督分割，当前从 ACDC 扩展到 ACDC + MSCMR/MACMR + ISBI。
- 两阶段框架：
  - Stage 1: 训练 weak detector。
  - Stage 2: 从 detector 初始化 tracker，只训练 tracker，image encoder frozen。
- 关键口径：tracker 的 conditioning seed 来自同一个弱监督 detector 的预测，不使用 full label 生成 seed；full label 只用于 val/test evaluation。
- 重要提醒：`tracker_auto_seed_inference.py` 导出的 `detector_predictions_segm.json` 是 tracker 流程中的 raw detector candidates，不等于 official detector-only baseline。公平 merge 必须使用 official detector pipeline 的最终 masks 作为 detector branch。

## Detector Checkpoints

配置：

- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_lora.yaml`
- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_lora_balanced.yaml`

输出目录：

- `gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/`

重点 checkpoint：

- `checkpoints/checkpoint_20.pt`
- `checkpoints/val_macro_segmentation_coco_eval_segm_AP.pt`

official detector-only test Dice：

| Detector ckpt | Macro | ACDC | MSCMR | ISBI | 备注 |
|---|---:|---:|---:|---:|---|
| `checkpoint_20.pt` | 0.8940 | 0.9200 | 0.9016 | 0.8602 | 当前最强 detector Dice baseline |
| `val_macro_segmentation_coco_eval_segm_AP.pt` | 0.8887 | 0.9122 | 0.8870 | 0.8668 | AP best，不等于 Dice best；ISBI 更好 |

结果位置：

- `gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/dice_sweep_test_two_ckpts/summary.tsv`
- `gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/dice_sweep_test_two_ckpts/checkpoint_20/*/evaluation_results_*.json`
- `gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/dice_sweep_test_two_ckpts/val_macro_segmentation_coco_eval_segm_AP/*/evaluation_results_*.json`

结论：

- detector 按 val AP 保存的 `val_macro...pt` 不一定是最终 Dice 最好的模型。
- `checkpoint_20.pt` 是当前主表里应保留的强 detector baseline。
- `val_macro...pt` 适合作为新 tracker v2 的 detector backbone，用于验证“tracker 是否能提升自己的 detector backbone”。

## Pseudo Seed Banks

生成脚本：

- `gq_scripts/preprocess/build_scribble_tracker_pseudo_seed_bank.py`
- `gq_scripts/preprocess/run_joint_tracker_pseudo_seed_banks.sh`
- `gq_scripts/preprocess/run_joint_tracker_pseudo_seed_banks_val_macro.sh`

`checkpoint_20.pt` seed bank：

- `gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/pseudo_seed_bank/acdc/scribble_tmi_pseudo_seed_video_annotations.coco.json`
- `gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/pseudo_seed_bank/mscmr/scribble_pseudo_seed_video_annotations.coco.json`
- `gq_experiment/joint/acdc_mscmr_isbi_scribble_lora_balanced/pseudo_seed_bank/isbi/scribble_pseudo_seed_video_annotations.coco.json`

统计：

| Seed bank | ACDC | MSCMR | ISBI |
|---|---:|---:|---:|
| `checkpoint_20.pt` | videos 140, anns 420, seed_frames 3384 | videos 25, anns 73, seed_frames 492 | videos 48, anns 49, seed_frames 86 |
| `val_macro...pt` | videos 140, anns 420, seed_frames 3606 | videos 25, anns 70, seed_frames 989 | videos 48, anns 56, seed_frames 652 |

ISBI seed 在 `checkpoint_20.pt` 下明显稀疏，曾放宽阈值：`score_threshold=0.0`, `min_scribble_recall=0.1`。

## Tracker V1 Status

配置：

- `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1.yaml`

输出：

- `gq_experiment/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v1/`

训练特点：

- `image_backbone_checkpoint=.../checkpoint_20.pt`
- 只训练 tracker，image encoder frozen。
- 采样近似 balanced：ACDC x1、MSCMR x5、ISBI x4。
- 每 5 epoch 保存 `checkpoint_*.pt`。
- 原始 global best 按三数据集 val AP mean 保存：`val_macro_segmentation_coco_eval_segm_AP.pt`。
- v1 训练日志中的 best AP 约 `val_macro/segmentation/coco_eval_segm_AP=0.528485`。

启动脚本：

- `gq_scripts/train/run_joint_tracker_scribble_image_init_v1.sh`

## Fair Merge Results

同口径 fair merge 已完成：detector branch 使用 official detector-only pipeline 输出，tracker branch 使用 `tracker_auto_seed_inference.py` 的 `tracker_predictions_segm.json`；各数据集独立 val sweep，固定 best config 到 test。

转换脚本：

- `gq_scripts/evaluate/convert_batch_predictions_to_coco_results.py`
- 该脚本把 `batch_inference.py` 的 `predictions.pkl` 转为 merge 可用 COCO result JSON，并支持 ISBI `061`/`patient061` 命名差异。

三数据集 test 汇总：

| Dataset | Detector-only `checkpoint_20` | Tracker-only AP-best | Fair simple merge | Fair reliability-aware merge |
|---|---:|---:|---:|---:|
| ACDC Dice | 0.9200 | 0.9041 | 0.9146 | 0.9145 |
| MSCMR Dice | 0.9016 | 0.8886 | 0.9014 | 0.9013 |
| ISBI Dice | 0.8602 | 0.8122 | 0.8316 | 0.8321 |
| Macro Dice | 0.8940 | 0.8683 | 0.8825 | 0.8826 |

结论：

- v1 fair merge 三个数据集都没有超过 official detector-only `checkpoint_20.pt`。
- MSCMR merge 与 detector-only 基本持平但略低；ISBI merge 比 tracker-only 高，但仍低于 detector-only。
- 当前 v1 tracker/merge 更适合作为诊断或消融，不应作为主结果提升。

主要结果路径：

- ACDC:
  - `gq_experiment/joint/tracker_eval/acdc/test_official_detector/eval_3d/evaluation_results_acdc.json`
  - `gq_experiment/joint/tracker_eval/acdc/test_fair_simple_merge/eval_3d/evaluation_results_acdc.json`
  - `gq_experiment/joint/tracker_eval/acdc/test_fair_full_merge/eval_3d/evaluation_results_acdc.json`
- MSCMR:
  - `gq_experiment/joint/tracker_eval/mscmr/test_official_detector/eval_3d/evaluation_results_mscmr.json`
  - `gq_experiment/joint/tracker_eval/mscmr/test_fair_simple_merge/eval_3d/evaluation_results_mscmr.json`
  - `gq_experiment/joint/tracker_eval/mscmr/test_fair_full_merge/eval_3d/evaluation_results_mscmr.json`
- ISBI:
  - `gq_experiment/joint/tracker_eval/isbi/test_official_detector/eval_3d/evaluation_results_isbi.json`
  - `gq_experiment/joint/tracker_eval/isbi/test_fair_simple_merge/eval_3d/evaluation_results_isbi.json`
  - `gq_experiment/joint/tracker_eval/isbi/test_fair_full_merge/eval_3d/evaluation_results_isbi.json`

## Tracker Checkpoint Dice Sweep

用户怀疑 tracker 按 AP 保存不一定对应 Dice 最优，因此已跑 v1 tracker 各 checkpoint 的 val tracker-only 3D Dice sweep。

脚本与结果：

- `gq_scripts/evaluate/run_joint_tracker_checkpoint_val_dice_sweep.sh`
- `gq_experiment/joint/tracker_eval/checkpoint_val_dice_sweep/summary_tracker_only_val_dice.tsv`

top 结果：

| Rank | Checkpoint | Macro Dice | ACDC | MSCMR | ISBI |
|---:|---|---:|---:|---:|---:|
| 1 | `checkpoint_20` | 0.8625 | 0.8936 | 0.8894 | 0.8044 |
| 2 | `checkpoint_15` | 0.8619 | 0.8930 | 0.8866 | 0.8059 |
| 3 | `checkpoint_5` | 0.8616 | 0.8895 | 0.8841 | 0.8111 |
| 5 | `val_macro_segmentation_coco_eval_segm_AP` | 0.8583 | 0.8914 | 0.8900 | 0.7934 |

结论：

- v1 下游 val Dice 最优是 `checkpoint_20.pt`，不是 AP-best `val_macro...pt`。
- 以后训练 tracker 最好同时保存 AP best 和 Dice best，不要只依赖 AP。

## Tracker V1 Checkpoint_20 Test Merge

已额外评估 v1 `checkpoint_20.pt` tracker 的 test merge：

| Dataset | Detector-only `checkpoint_20` | Tracker-only ckpt20 | Merge ckpt20 |
|---|---:|---:|---:|
| ACDC Dice | 0.9200 | 0.9072 | 0.9160 |
| MSCMR Dice | 0.9016 | 0.8865 | 0.9020 |
| ISBI Dice | 0.8602 | 0.8157 | 0.8390 |
| Macro Dice | 0.8940 | 0.8698 | 0.8856 |

结论：

- `checkpoint_20` tracker merge 比 AP-best tracker merge 稍好，MSCMR 略高于 detector-only，但 macro 仍低于 detector-only。
- tracker-only 推理阶段用过 `--tracker_detection_threshold 0.7` 生成 `tracker_predictions_segm.json`；`evaluate_tracker_coco_predictions.py` 本身不再额外扫阈值，而是对每个 image/category 取最高分预测算 3D Dice。

结果位置：

- `gq_experiment/joint/tracker_eval/checkpoint_20_fair_merge/*/test_tracker_only/eval_3d/`
- `gq_experiment/joint/tracker_eval/checkpoint_20_fair_merge/*/test_full_merge/eval_3d/`

## Tracker V2 Val-Macro Backbone Plan

由于 `checkpoint_20.pt` detector 太强，v1 merge 很难超过 detector-only；用户决定用 `val_macro_segmentation_coco_eval_segm_AP.pt` 作为 detector backbone 再训练一版 tracker v2。

实验口径：

- 用 `val_macro...pt` 重新生成 pseudo seed bank。
- 训练 tracker v2：配置主要改 `image_backbone_checkpoint` 和 seed bank 路径。
- fair merge 时 detector branch 也必须使用 `val_macro...pt` official detector outputs。
- 报告时同时保留：
  - `checkpoint_20 detector-only`
  - `val_macro detector-only`
  - `val_macro tracker-only`
  - `val_macro fair merge`

已完成的 v2 训练前准备：

- 新增 Dice evaluator: `sam3/eval/video_dice_eval.py`
- 修改 trainer 支持多个 global best checkpoint: `sam3/train/trainer.py`
- 新建 v2 配置: `sam3/train/configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v2_val_macro.yaml`
- 新建 val-macro seed bank 脚本: `gq_scripts/preprocess/run_joint_tracker_pseudo_seed_banks_val_macro.sh`
- 新建 v2 训练脚本: `gq_scripts/train/run_joint_tracker_scribble_image_init_v2_val_macro.sh`
- 已用 `val_macro...pt` 生成新 pseudo seed bank，统计见上方 seed bank 表。

v2 checkpoint 保存策略：

- 每 5 epoch 保存 `checkpoint_*.pt`。
- 按 val AP mean 保存：`val_macro_segmentation_coco_eval_segm_AP.pt`。
- 按 val Dice mean 保存：`val_macro_segmentation_dice.pt`。

验证过：

- `python -m py_compile sam3/eval/video_dice_eval.py sam3/train/trainer.py` 通过。
- v2 YAML 可以被 Hydra compose。
- `VideoDiceEvaluator` 在 ACDC dump 上返回 Dice/IoU 正常。
- 两个 shell 脚本 `bash -n` 通过。

最终 v2 nohup 训练命令：

```bash
cd /home/gaoqi/sam3

CUDA_VISIBLE_DEVICES=2,3 nohup /home/gaoqi/anaconda3/envs/sam3/bin/python sam3/train/train.py \
  -c configs/joint/acdc_mscmr_isbi_scribble_tracker_image_init_v2_val_macro.yaml \
  --use-cluster 0 \
  --num-gpus 2 \
  > joint_acdc_mscmr_isbi_scribble_tracker_image_init_v2_val_macro.log 2>&1 &
```

或：

```bash
cd /home/gaoqi/sam3
GPUS=2,3 NUM_GPUS=2 bash gq_scripts/train/run_joint_tracker_scribble_image_init_v2_val_macro.sh
```

## Temporal Consistency Loss Idea

用户导师建议在 tracker 中加入连续一致性损失。建议方案要克制、可实现、风险低，不建议一开始引入 optical flow/3D registration。

推荐实验：`Temporal Anatomical Consistency Loss` / `Adjacent Slice Consistency Loss`。

核心 loss：

- `L_area`: 相邻 slice 预测概率 mask 的面积连续性，使用 Huber。
- `L_centroid`: soft centroid 的位置连续性，使用 Huber。
- `L_shape`: 可选弱 shape consistency，先按质心偏移对齐后做 soft Dice consistency，权重要小。

总损失：

```text
L_total = L_tracker_weak
        + lambda_area * L_area
        + lambda_centroid * L_centroid
        + lambda_shape * L_shape
```

建议初始权重：

- `lambda_area=0.05`
- `lambda_centroid=0.05`
- `lambda_shape=0.01` 或先不加

必要 gate：

- 只对同一 video、同一 category/object 的相邻 slice 计算。
- 若 slice index 间隔 > 2，跳过。
- 若两帧预测面积都太小，跳过。
- 对 apex/base 器官出现/消失边界弱化或跳过。
- 先做 area + centroid，再加 weak shape。

建议实验矩阵：

| Exp | Loss |
|---|---|
| v2 baseline | 无 temporal consistency |
| v2 + area | 只加面积连续 |
| v2 + area + centroid | 加位置连续 |
| v2 + area + centroid + weak shape | 加轻量形状一致 |

预期可能改善：

- ISBI PZ/CG 跨 slice 断裂。
- ACDC/MSCMR RV 边界或跳动。
- HD95/NSD、tracker-only 稳定性、merge 中 tracker 替换 detector 时的失败率。

相关前沿脉络：

- SAM 2: streaming memory / video segmentation。
- MedSAM-2: 把 2D/3D 医学图像当作 video 分割。
- Cutie: object-level memory reading。
- 一些 RVOS 工作使用 temporal consistency / mask consistency score。

## Older ACDC Tracker Context

旧的单 ACDC scribble tracker 结果仍有参考价值，但不要和当前 joint tracker 混淆：

- 单 ACDC detector-only Dice 约 `0.9130`。
- 单 ACDC reliability-aware temporal merge 曾达到约 `Dice=0.9205`。
- oracle merge 约 `0.9243`。
- 旧结果目录：
  - `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1_reliability_temporal_merge_sweep_with_tracker_only/`
  - `gq_experiment/acdc/scribble_sam3_tracker_image_init_v1_oracle_diagnosis/`

解释：

- 旧实验能 merge 提升，可能因为单 ACDC detector baseline 较弱；当前 joint `checkpoint_20.pt` detector 已很强，尤其 ACDC Dice 0.9200，merge 超过它更难。

## Recommended Next Steps

1. 启动 tracker v2 val-macro backbone 训练，观察 AP-best 与 Dice-best checkpoint 是否分离。
2. v2 训练完成后，用 `val_macro...pt` official detector branch 做 fair merge，报告 `val_macro detector-only/tracker-only/merge`，并与 `checkpoint_20 detector-only` 同表展示。
3. 若 v2 仍不能给出足够清晰提升，再做 oracle upper bound 和失败样本诊断，确认 detector/tracker 逐 slice 互补空间。
4. 若要做新方法，优先实现 area + centroid temporal consistency loss；先小规模 ablation，再考虑 weak shape。

# SAM3 微调项目背景说明

本仓库是一个面向医学图像分割的 SAM3 定制化微调项目，主路径基于 **SAM3 image model**，而不是官方 video 训练路径。当前工作同时覆盖：

- 全监督微调
- 基于 scribble 的弱监督微调
- 基于 LoRA 的参数高效微调
- 将 3D volume 组织成 video-like sample 的实验路径
- 第一版切片上下文关联实验 `context v1`

## 项目主线

当前项目最重要的事实是：

- 主训练范式仍然是 **image model fine-tuning**
- 3D 医学数据经常先被切成 2D 切片，转成 COCO 风格 JSON 后训练
- 即使引入 3D / video-like 数据组织，也不代表模型已经天然学会跨切片上下文

默认应把本仓库看成：

- 一个以 ACDC 为主参考数据集的 SAM3 医学分割微调工程
- 2D 全监督 / 弱监督链路已经成熟
- 3D / 切片上下文研究仍处于实验推进阶段

## 环境与默认路径

数据根目录：

- `/home/gaoqi/dataset/using`

推荐 Conda 环境：

- `sam3`

后续 AI 在运行训练、评估、预处理脚本前，默认都应先激活：

- `source /home/gaoqi/anaconda3/etc/profile.d/conda.sh`
- `conda activate sam3`

除非用户明确说明，否则默认使用各数据集 `processed` 目录，而不是 `raw`。

## 重点数据集

当前最相关的数据集：

- `acdc`
- `isbi`
- `mscmr`

### ACDC

ACDC 的默认 2D processed 路径：

- `/home/gaoqi/dataset/using/acdc/processed/sam3_png_coco_fullframes_100`

ACDC 的默认 3D / video-like processed 路径：

- `/home/gaoqi/dataset/using/acdc/processed/sam3_video_npz_coco_fullframes_100`

ACDC 是当前理解本仓库实现的主参考数据集。

### ISBI

ISBI 当前已完成预处理适配。

关键路径：

- 源 split：`/home/gaoqi/dataset/using/isbi/processed/scribble_nifti_split`
- 输出数据：`/home/gaoqi/dataset/using/isbi/processed/png_coco_sam3_fullframes`

重要注意事项：

- 当前只有 `train` 和 `test`，没有 `val`
- 现有 H5 -> NIfTI 数据不包含可信物理 spacing
- 如果用户没有显式提供 `--spacing_file`，默认只把 `Dice/IoU` 当可靠指标
- `HD95/NSD` 在 ISBI 上默认可能无意义或为 `nan`

### MSCMR

MSCMR 默认 processed 路径：

- `/home/gaoqi/dataset/using/mscmr/processed/png_coco_sam3_fullframes`

重要事实：

- `raw/train` 当前只有 scribble 标签，没有完整 dense/manual 标签
- `raw/val` 和 `raw/test` 才有完整标签

因此：

- 如果后续 AI 发现 MSCMR 的 `train/full_annotations.coco.json` 被用于训练，必须先确认其 full 标签来源
- 不能默认假设 MSCMR 和 ACDC 一样天然同时具备完整 full train/val/test 标注

## 最重要的配置文件

理解 ACDC 2D 路径时，优先看：

- `sam3/train/configs/acdc/full_lora_100.yaml`
- `sam3/train/configs/acdc/scribble_lora_100.yaml`

理解 ACDC 3D / video-like 路径时，优先看：

- `sam3/train/configs/acdc/full_video_lora_100.yaml`
- `sam3/train/configs/acdc/scribble_video_lora_100.yaml`

理解切片上下文实验时，优先看：

- `sam3/train/configs/acdc/full_video_lora_100_context_v1.yaml`
- `sam3/train/configs/acdc/scribble_video_lora_100_context_v1.yaml`

## 推理与评估

主要脚本位于：

- `gq_scripts/evaluate/batch_inference.py`
- `gq_scripts/evaluate/batch_inference_context.py`
- `gq_scripts/evaluate/batch_evaluate.py`
- `gq_scripts/evaluate/batch_evaluate_utils.py`
- `gq_scripts/evaluate/run_inference_and_eval.sh`
- `gq_scripts/evaluate/run_context_inference_and_eval.sh`

当前工作流中的默认推理阈值：

- `confidence_threshold = 0.7`

原因：

- ACDC 中有些切片并不存在目标结构
- 阈值偏高可以减少空白切片上的伪阳性

但要记住：

- `0.7` 是当前常用工作阈值，不是理论常数
- 如果结果几乎全为 0，可用更低阈值如 `0.0` 排查

评估指标：

- Dice
- IoU
- HD95
- NSD

其中：

- `HD95` 和 `NSD` 依赖 spacing
- 评估时优先读取 `test_dir/spacing_map.json`
- 某些场景下也可从原始 NIfTI 推断

## 预处理与 JSON 语义

### 全监督预处理

主要文件：

- `gq_scripts/preprocess/preprocess_full_annotations.py`
- `gq_scripts/preprocess/preprocess_video_annotations.py`

作用：

- 将 3D 医学数据切成 2D PNG 或组织成 volume `.npz`
- 生成 full COCO JSON
- 导出可直接喂给 SAM3 训练的数据目录

### Scribble 弱监督预处理

主要文件：

- `gq_scripts/preprocess/preprocess_scribble_annotations.py`
- `gq_scripts/preprocess/preprocess_video_scribble_annotations.py`

作用：

- 生成 scribble 风格 COCO JSON
- 用 `segmentation` 存目标 scribble
- 用 `valid_mask` 定义可信监督区域

### `valid_mask` 的含义

`valid_mask` 是本项目弱监督设计中的核心机制，不是附带字段。

语义上：

- `1` 表示正样本 scribble 像素
- `0` 表示 valid 区域内背景
- `255` 表示忽略区域

在 `scribble1` 设定下：

- 其他对象的 scribble 会进入当前 query 的 `valid_mask`
- 从而作为背景约束，而不是简单忽略

因此后续 AI 必须记住：

- 本项目不会把 scribble 假装成 dense mask
- 弱监督是 **部分区域监督**，不是稠密监督

## COCO JSON 如何接入 SAM3

关键文件：

- `sam3/train/data/coco_json_loaders.py`
- `sam3/train/data/sam3_image_dataset.py`

其中：

- `coco_json_loaders.py` 负责把 COCO JSON 整理成 query-based 训练所需结构
- `sam3_image_dataset.py` 负责把 full / weak annotation 转成训练 mask

弱监督时：

- dataset 会构造三值 mask `1/0/255`
- bbox 优先级为：推断 bbox > JSON bbox > 从 scribble 计算 pseudo bbox

## 全监督与弱监督配置差异

### `full_lora_100.yaml`

- 使用 full annotations
- segmentation loss 使用 `Masks`
- bbox / giou loss 开启
- matcher 更依赖 bbox 相关 cost

### `scribble_lora_100.yaml`

- 训练使用 `scribble_tmi_annotations.coco.json`
- 验证仍使用 full annotations
- segmentation loss 使用 `PartialMasks`
- `loss_bbox` / `loss_giou` 关闭
- matcher 更偏向分类项

对应损失主要在：

- `sam3/train/loss/loss_fns.py`

关键原则：

- `PartialMasks` 只在 valid 区域内计算损失
- `ignore_index=255` 必须保留

## LoRA 微调策略

LoRA 逻辑位于：

- `sam3/model/lora.py`

模型构造入口位于：

- `sam3/model_builder.py`

当前 ACDC LoRA 路径的核心约定：

- 大部分主干组件走 LoRA
- `mask_decoder` / `segmentation_head` 与 `dot_prod_scoring` 保持全量可训练
- LoRA 注入发生在加载预训练 checkpoint 之后

后续 AI 修改 LoRA 行为时，应优先联动检查：

- `sam3/model/lora.py`
- `sam3/model_builder.py`
- 当前使用的 Hydra 配置

## 3D / video-like 路径应如何理解

当前仓库已经支持把一个 3D volume 组织成一个 video-like sample。

关键文件：

- `sam3/train/data/coco_json_loaders.py`
- `sam3/train/data/sam3_video_dataset.py`
- `sam3/model/sam3_image.py`
- `sam3/model_builder.py`

需要记住的事实：

- `VideoGroundingDataset` 和 `COCO_VIDEO_FROM_JSON` 已支持 volume-as-sample
- 当前实现可以从 `npz` 逐 slice 读取图像
- `Sam3ImageOnVideoMultiGPU` 能接收多帧输入
- 但 `forward` 仍然主要是 **按 frame 逐张做 `forward_grounding`**

因此当前的 “3D 输入” 本质上是 **多切片打包输入**，还不等于真正的跨切片 attention / memory / feature fusion。

### 3D 弱监督路径

当前仓库已支持在 video annotation 中携带 `valid_mask`。

因此：

- 3D scribble video JSON 会继续复用 2D 弱监督的三值语义
- weak mask 值域应为 `0/1/255`
- full mask 值域应为 `0/1`

### 3D 验证闭环

本地代码已修正两个关键问题：

- 为每个 frame 保持固定类别数 query，避免 stage 对齐失败
- 在验证阶段给每个 `(video_id, frame_idx)` 分配全局唯一 `frame_id`

因此现在可以正常走：

- `PredictionDumper`
- `frame_annotations.coco.json`
- 标准 COCO evaluator

ACDC 3D 路径默认会导出：

- `video_annotations.coco.json`
- `frame_annotations.coco.json`
- `scribble_tmi_video_annotations.coco.json`

## 当前 3D batch 语义

不要把 3D 配置里的 `train_batch_size=2` 机械理解成旧 2D 训练的 `batch_size=18`。

更准确地说：

- `train_batch_size=2` 表示一个 step 有 `2` 个 3D volume 样本
- 但训练时通常不会把 volume 全部切片都送进模型
- 例如 ACDC 3D 配置曾使用 `num_stages_sample=4`
- 那么一个 step 实际参与训练的 slice 数更接近 `2 x 4 = 8`

同时它也不应简单等同于“2D batch size = 8”，因为：

- video-like 路径还有额外的 stage / query 组织开销
- 显存行为与纯 2D 训练并不完全一致

## 当前 3D 路径的限制

即使数据已经按 volume 组织，仍需区分三个层次：

1. 数据是否按 volume 组织
2. 训练采样时是否真的采到邻近切片 / 空白切片
3. 模型前向时是否真的发生跨切片信息交互

当前 3D 路径主要解决的是第 1 点和部分第 2 点，不是第 3 点。

## 切片上下文研究背景

用户当前的研究动机是：

- SAM3 image model 的主训练路径是 2D
- 官方 video 相关模块并不直接适配当前想做的医学训练
- 用户尝试过把预训练权重里的 video 模块接回自己微调后的 image model，但效果不理想
- 因此转向更现实的路线：
  - 先把 3D volume 作为一个 sample 输入现有训练管线
  - 再在 image model 主干上单独研究切片上下文学习

与这个方向相关的工程启发来自：

- `https://github.com/facebookresearch/sam3/issues/318`

这个 issue 更重要的启发是：

- 3D 医学数据可以先被组织成 video-like sample
- 从而为后续切片上下文建模打下工程基础

## Slice Context V1

仓库中已经接入第一版切片上下文实验骨架，目标是：

- 输入一个连续切片窗口
- 训练时只预测中心切片
- 把邻近切片特征压缩成 visual prompt tokens
- 通过 image-model 路径注入上下文

关键文件：

- `sam3/model/slice_context_adapter.py`
- `sam3/model/sam3_image_slice_context.py`
- `sam3/model_builder.py`
- `gq_scripts/evaluate/batch_inference_context.py`
- `gq_scripts/evaluate/run_context_inference_and_eval.sh`

关键配置：

- `sam3/train/configs/acdc/full_video_lora_100_context_v1.yaml`
- `sam3/train/configs/acdc/scribble_video_lora_100_context_v1.yaml`

### `context v1` 的真实机制

更准确地说，`context v1` 不是“5 张图联合做一个真正的 3D 编码器”，而是：

- 输入一个连续窗口，当前 ACDC 配置为 `window size = 5`
- 中心切片仍是实际要分割的目标
- 邻近切片先经过 backbone 提特征
- 再通过 `AdaptiveAvgPool2d((2,2)) + Linear` 压缩成 context tokens
- 每张邻居切片产生 `4` 个 token
- 4 张邻居切片一共得到 `16 x 256` 的 visual prompt tokens
- 再按相对中心切片的位置加上 slice-level relative position embedding
- 最后作为 `visual_prompt_embed` 注入 `_encode_prompt(...)`

因此它本质上是 **neighbor-slice feature prompting**，而不是显式 slice-to-slice 对齐建模。

### 当前配置与行为

当前 context 配置里：

- `num_stages_sample=5`
- `stage_stride_min=1`
- `stage_stride_max=1`

也就是严格连续窗口。

训练 / 验证行为：

- 训练时：只监督中心切片
- 验证时：对整套 volume 逐帧滑窗输出

因此当前 context 配置已经支持：

- 训练期自动验证
- 保存 best checkpoint
- 离线 test 推理与评估

### 已完成的工程接通

截至目前，以下链路已跑通：

- context 训练 smoke
- context 自动 val + best checkpoint
- context 离线推理 smoke
- ACDC test 上 full / scribble context 模型的正式推理评估

### 当前结果结论

ACDC 上，`context v1` 相比无上下文 baseline **出现退化**。

对比结果：

- Full baseline：`Dice 0.9323`
- Full context v1：`Dice 0.9063`
- Scribble baseline：`Dice 0.9130`
- Scribble context v1：`Dice 0.8572`

因此当前结论应写成：

- `context v1` 已经完成工程闭环
- 但其“邻切片特征 -> visual prompt token”的直接注入方式，在 ACDC 上暂未带来收益
- 弱监督下退化更明显

这不代表切片上下文方向错误，更可能意味着：

- 当前注入方式过于粗糙
- prompt-level 强注入扰乱了原本稳定的 2D 表征
- 还没有形成真正有效的切片关联机制

### 后续更推荐的方向

比起继续强化 `context v1`，当前更推荐探索更保守的 `context v2`：

- 只看最小邻域，如 `center±1`
- 使用带门控的残差融合，门控初值设为 0
- 尽量把上下文融合放在 image feature 路径，而不是强行塞入 prompt 路径
- 目标先从“不伤 baseline”开始，而不是一开始追求大提升

## 推荐优先查看的文件

快速理解项目：

- `sam3/train/configs/acdc/full_lora_100.yaml`
- `sam3/train/configs/acdc/scribble_lora_100.yaml`
- `sam3/train/configs/acdc/full_video_lora_100.yaml`
- `sam3/train/configs/acdc/scribble_video_lora_100.yaml`
- `sam3/train/configs/acdc/full_video_lora_100_context_v1.yaml`
- `sam3/train/configs/acdc/scribble_video_lora_100_context_v1.yaml`
- `sam3/model_builder.py`

修改弱监督行为：

- `gq_scripts/preprocess/preprocess_scribble_annotations.py`
- `gq_scripts/preprocess/preprocess_video_scribble_annotations.py`
- `sam3/train/data/coco_json_loaders.py`
- `sam3/train/data/sam3_image_dataset.py`
- `sam3/train/loss/loss_fns.py`

修改 3D / context 行为：

- `gq_scripts/preprocess/preprocess_video_annotations.py`
- `sam3/train/data/sam3_video_dataset.py`
- `sam3/model/sam3_image.py`
- `sam3/model/sam3_image_slice_context.py`
- `sam3/model/slice_context_adapter.py`
- `gq_scripts/evaluate/batch_inference_context.py`

## 给后续 AI 的默认假设

除非用户另行说明，默认前提如下：

- 项目核心仍然是医学图像分割下的 SAM3 image model 微调
- Hydra 配置是第一控制入口
- ACDC 是主参考数据集
- `valid_mask` 是弱监督核心机制
- 弱监督必须保留 `1/0/255` 三值语义
- LoRA 是默认参数高效微调策略
- 3D / video-like 路径已经接通，但不应误认为已经实现真正 3D 建模
- `context v1` 已经完成工程验证，但当前结果不如 baseline

如果后续要修改弱监督或 3D / context 行为，务必联动检查：

- 预处理
- JSON loader
- dataset
- loss
- Hydra 配置
- 推理脚本

因为这些部分在本项目中是强耦合的。

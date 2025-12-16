### roboflow_train
`num_images: 100`: 设定每个类别仅使用 100 张图片进行训练
`supercategory`: 动态选择数据集类别。它使用 `submitit.job_array.task_index`，这意味着这是一个任务数组（Job Array），可以一次性提交 100 个任务，每个任务对应 RF100 中的一个子数据集（如 `zebrasatasturias`, `blood-cell` 等）。

train_transforms:
- **`FilterCrowds`**: 过滤掉标记为“拥挤”的目标。
- **`RandomizeInputBbox`**: 对输入的边界框（BBox）添加噪声（标准差 0.1，最大 20像素），增加模型鲁棒性。
- **`RandomResizeAPI`**: 随机调整图像大小。
    - `sizes`: 目标分辨率（由 `scratch.resolution` 定义，即 1008）。
    - `square: true`: 保持正方形。
- **`PadToSizeAPI`**: 将图像填充（Padding）到固定分辨率。
- **`NormalizeAPI`**: 标准化图像（均值和方差在 `scratch` 中定义）。
- **`FilterEmptyTargets`**: 过滤掉没有目标对象的样本。

### loss
使用的是 `Sam3LossWrapper`，这里有一个重要细节：
- `loss_fn_semantic_seg: null`: 语义分割损失被禁用。
- 主要损失 (`loss_fns_find`): 侧重于目标检测（Box Detection）。
    - `Boxes`: 边界框回归损失。
        - `loss_bbox`: L1 损失权重 5.0。
        - `loss_giou`: GIoU 损失权重 2.0（衡量框的重叠度）。
    - `IABCEMdetr`: 分类损失（类似 DETR 的匹配机制）。
        - `loss_ce`: 交叉熵分类损失权重 20.0。
        - `presence_loss`: 目标存在性损失。
- Matcher: 使用 `BinaryOneToManyMatcher`，这表明模型在训练时将预测框与真实框进行“一对多”的匹配，有助于加速收敛。

### scratch
这是全局超参数的集中定义区，控制模型架构和训练细节。
- 开关设置:
    - `enable_segmentation: False`: 注意，虽然是 SAM 模型，但此配置关闭了分割掩码（Mask）生成，仅进行边界框检测（Object Detection）。
- 模型架构:
    - `d_model: 256`: Transformer 的隐藏层维度。
    - `pos_embed`: 使用正弦位置编码（Sine Position Embedding）。
- 图像参数:
    - `resolution: 1008`: 输入图像分辨率（非常大，适合精细检测）。
- 训练超参数:
    - `train_batch_size: 1`: 单卡 Batch Size 为 1。
    - `gradient_accumulation_steps: 1`: 梯度累积步数。
    - `max_data_epochs: 20`: 训练 20 个 Epoch。
    - `lr_scale: 0.1`: 学习率缩放因子。
    - `lr_transformer`: Transformer 部分学习率 `8e-4 * 0.1`。
    - `lr_vision_backbone`: 视觉主干学习率 `2.5e-4 * 0.1`。
    - `lr_language_backbone`: 语言主干学习率 `5e-5 * 0.1`（语言部分学习率最低，通常为了保持预训练知识）。

### trainer
定义了 PyTorch Lightning 或类似的训练循环逻辑。
- 基本设置:
    - `max_epochs: 20`: 总训练轮数。
    - `accelerator: cuda`: 使用 GPU。
    - `skip_saving_ckpts: true`: **不保存中间检查点**，只在最后保存（或完全不存，需结合 `checkpoint` 字段看）。
- 数据加载器 (`data`):
    - 使用 `Sam3ImageDataset`。
    - Train set 读取 `_annotations.coco.json`。
    - Val set 读取 `test` 目录的数据。
- 优化器 (`optim`):
    - AMP: 开启混合精度训练 (`bfloat16`)，节省显存并加速。
    - Optimizer: `AdamW`。
    - Scheduler: 使用 `InverseSquareRootParamScheduler`（Transformer 常用），带有 20 步的 Warmup。
    - Layer Decay: 对视觉主干网络使用了层级学习率衰减 (`0.9`)，这意味着越靠近输入层的参数更新越慢，保留底层特征。
- 评估 (`meters`):
    - 使用 COCO 标准进行评估 (`iou_type: "bbox"`).
    - 结果会 dump 到日志目录中。
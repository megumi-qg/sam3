# SAM3 Tracker 学习笔记

先只讨论普通 `tracker`，不讨论 `multiplex tracker`。

## 0. `Sam3TrackerTrainAdapter` 是什么？

`Sam3TrackerTrainAdapter` 不是 SAM3 官方发布时强调的那个“完整视频推理模型”名字，而是这个项目里为了**训练普通单目标 tracker**额外封装出来的一个训练适配器。

它的定义在：

- [sam3/model/sam3_tracker_train_adapter.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_train_adapter.py:18)

从类定义可以直接看出，它：

1. 继承自 `Sam3TrackerBase`
2. 目标是给“original SAM3 single-object tracker”提供一个最小但可训练的封装
3. 适配当前医学场景：把每个 `(volume, category)` 当成一个单目标 tracking 样本

对应注释在：

- [sam3/model/sam3_tracker_train_adapter.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_train_adapter.py:20)

更准确地说，它的作用不是“重新发明一个 tracker”，而是：

`把原始 SAM3 tracker 改造成一个适合本项目 Trainer / Dataset / Loss 直接调用的训练模型`

也就是说，它站在两个世界中间：

- 一边是底层的 `Sam3TrackerBase`，负责真正的 tracking / memory / mask decoder 逻辑
- 另一边是本项目的训练框架，负责 dataloader、loss、batch、验证和 checkpoint 保存

所以你可以把它理解为一层“训练适配壳”。

## 0.1 它在训练链路中的位置

本项目训练 tracker 时，配置文件并不是去构建完整的 `detector + tracker` 视频系统，而是直接构建：

- `build_sam3_tracker_train_model(...)`

位置在：

- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1377)

这个构建函数会：

1. 创建 tracker 需要的 `maskmem_backbone`
2. 创建 tracker 自己的 `transformer`
3. 创建视觉 `backbone`
4. 最后实例化 `Sam3TrackerTrainAdapter`

对应代码：

- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1397)
- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1405)

所以训练时真正进 `Trainer` 的模型，不是官方最外层 demo model，而是这个 adapter。

这也是为什么你训练好的权重第一层前缀是：

- `backbone`
- `transformer`
- `maskmem_backbone`
- `sam_mask_decoder`
- `sam_prompt_encoder`

而不是官方整模型里的：

- `detector`
- `tracker`

因为保存的是 adapter 自己的 `state_dict`，不是外层组合模型的 `state_dict`。

## 0.2 它具体帮训练做了什么？

`Sam3TrackerTrainAdapter` 最重要的额外工作，是把“医学分割训练样本”整理成 tracker 能理解的 prompt / memory 输入格式。

其中最关键的函数是：

- [sam3/model/sam3_tracker_train_adapter.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_train_adapter.py:228)

`prepare_prompt_inputs(...)` 做的事情可以概括成：

1. 遍历一个样本里的多帧 GT mask
2. 找到第一个目标可见的切片，作为初始 conditioning frame
3. 把这帧 GT mask 放入 `mask_inputs_per_frame`
4. 其余切片作为后续传播监督目标

换句话说，它把一个医学 volume 的多切片样本转成了 tracker 熟悉的训练范式：

`第一帧给真值提示，后续帧做传播预测`

这正是你这个项目里 tracker 训练能成立的关键。

## 0.3 它和官方权重的关系

`Sam3TrackerTrainAdapter` 还负责把官方权重映射到当前训练模型里。

相关逻辑在：

- [sam3/model/sam3_tracker_train_adapter.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_train_adapter.py:101)

它会做两件事：

1. 如果官方 checkpoint 里有 `tracker.*`，就去掉这个前缀，加载到 adapter 内部
2. 如果官方 checkpoint 里只有 `detector.backbone.vision_backbone.*`，也会把这些视觉权重映射成 adapter 里的 `backbone.*`

另外，在你的 `image_init` 设定下，它还会额外从你训练好的 image backbone checkpoint 初始化视觉 backbone：

- [sam3/model/sam3_tracker_train_adapter.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_train_adapter.py:146)

所以可以把它看成：

- 下层继承原始 tracker 能力
- 上层适配本项目训练接口
- 中间负责做 checkpoint 权重映射与初始化

## 0.4 一句话理解

一句话总结：

`Sam3TrackerTrainAdapter = 为本项目医学数据训练而包装出来的“可训练版 SAM3 单目标 tracker”`

## 1. SAM3 的 tracker 模块原理是什么？

`sam3` 的 tracker 本质上是一个“带记忆的分割器”，而不是传统意义上单独依赖光流的追踪器。

它在每一帧上会重复做两件事：

1. 读取历史帧留下来的 memory，把当前帧特征和这些 memory 融合。
2. 用融合后的特征，通过 SAM 风格的 mask decoder 输出当前帧的 mask。

所以它的核心思路可以概括成：

`当前帧分割 = 当前帧视觉信息 + 历史目标记忆`

在代码里，最核心的一步是 `track_step(...)`，位于：

- [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:929)

这个函数内部的逻辑是：

1. 先调用 `_prepare_memory_conditioned_features(...)` 读取历史记忆。
2. 再调用 `_forward_sam_heads(...)` 用 SAM 风格头部预测当前帧 mask。
3. 最后调用 `_encode_new_memory(...)` 把当前帧的预测结果重新写回 memory，供后续帧使用。

因此，tracker 不是“拿上一帧 mask 直接平移”，而是“不断读写 memory 的递推式分割模块”。

## 2. 为什么它能实现前后传播？

因为它的“传播”本质上不是一个只能朝前运行的特殊网络，而是：

`按照某个时间顺序，重复调用同一个 track_step`

在推理接口里：

- [sam3/model/sam3_tracking_predictor.py](/home/gaoqi/sam3/sam3/model/sam3_tracking_predictor.py:790)

`propagate_in_video(...)` 会根据 `reverse` 参数决定遍历顺序：

- `reverse=False`：从前往后传播
- `reverse=True`：从后往前传播

更关键的是，tracker 在读取 memory 时，会根据传播方向改变“从哪一侧取邻近帧记忆”，以及对应的时间位置编码符号。这个逻辑在：

- [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:560)

也就是说：

- 正向传播时，“历史”是前面的帧
- 反向传播时，“历史”变成后面的帧

所以前向和后向并不是两套不同模型，而是同一个 tracker，只是换了时间遍历方向和 memory 读取方向。

## 3. tracker 模块位于整个 SAM3 模型的哪个部分？

如果看完整视频系统，tracker 是一个独立子模块。

它由下面这个函数构建：

- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:452)

在完整视频模型中，整体结构大致是：

1. 构建 `tracker`
2. 构建 `detector`
3. 把它们组合成完整视频推理模型

对应代码位置：

- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:733)

如果只看 tracker 自身，它在整个模型中的位置可以理解为：

`图像 backbone 特征` 和 `最终 mask 输出` 之间

也就是：

`backbone feature -> 时序记忆融合 -> SAM mask decoder -> 当前帧 mask`

更细一点说，tracker 主要由三部分组成：

1. `transformer`
作用：把当前帧特征和历史 memory 做 cross-attention 融合。

2. `maskmem_backbone`
作用：把历史 mask 和视觉特征编码成可存入 memory bank 的表示。

3. `SAM prompt/mask heads`
作用：根据融合后的特征输出当前帧 mask，并抽取 `obj_ptr`。

对应构建位置：

- memory encoder： [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:351)
- tracker transformer： [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:387)

## 4. SAM3 的 memory bank 是什么？和 tracker 有什么关系？

`memory bank` 可以理解成 tracker 的“短时工作记忆”。

它保存的不是原始图像，而是历史帧中和目标相关的压缩表示，主要包括：

1. `maskmem_features`
这是空间级的记忆特征，来源于“历史帧视觉特征 + 该帧目标 mask”的融合结果。

2. `maskmem_pos_enc`
这是 memory 对应的位置编码，告诉模型这些记忆的空间位置和时间顺序。

3. `obj_ptr`
这是目标级别的紧凑向量，可以理解成“这个目标是谁”的一个摘要表示。

真正把 mask 和视觉特征编码成 memory 的模块是：

- [sam3/model/memory.py](/home/gaoqi/sam3/sam3/model/memory.py:166)

其中 `SimpleMaskEncoder` 做的事情很直接：

1. 先把 mask 下采样
2. 再和像素级视觉特征相加融合
3. 再输出 memory feature 和位置编码

写 memory 的地方在：

- [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:1023)
- [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:796)

读 memory 的地方在：

- [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:585)

一句话概括两者关系：

`tracker` 是“如何利用历史信息完成当前帧分割”的主体，`memory bank` 是它反复读写的历史缓存。

如果没有 memory bank，tracker 就会退化成“逐帧独立分割”，跨帧稳定性会明显下降。

## 5. 一个简洁的数据流图

```text
输入提示（点 / 框 / 初始 mask）
        |
        v
当前帧图像 -> Backbone 提取当前帧视觉特征
        |
        v
读取 memory bank
  - 历史 maskmem_features
  - 历史 maskmem_pos_enc
  - 历史 obj_ptr
        |
        v
Tracker Transformer 做时序记忆融合
        |
        v
SAM 风格 Mask Decoder 输出当前帧 mask + obj_ptr
        |
        v
Memory Encoder 把“当前帧特征 + 当前 mask”编码成新 memory
        |
        v
写回 memory bank，供下一帧或反向传播时使用
```

## 6. 一个更口语化的理解

可以把 tracker 想成一个“边看当前帧、边翻历史笔记”的分割器。

- `memory bank` 像历史笔记
- `tracker transformer` 像查阅和整合笔记的过程
- `mask decoder` 像在当前帧上做最终判断
- `_encode_new_memory(...)` 像把当前帧的新结论再记回笔记本

因此它能够持续传播，是因为它每处理完一帧，就会更新一次自己的记忆；而它能够反向传播，是因为这个“查历史笔记”的过程本身并不限定方向，顺着时间读可以，倒着时间读也可以。

## 7. 关键代码位置

- tracker 主入口： [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:929)
- 读取 memory： [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:560)
- 写入 memory： [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:796)
- 推理传播接口： [sam3/model/sam3_tracking_predictor.py](/home/gaoqi/sam3/sam3/model/sam3_tracking_predictor.py:790)
- memory encoder 实现： [sam3/model/memory.py](/home/gaoqi/sam3/sam3/model/memory.py:166)
- tracker 构建： [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:452)
- 完整视频模型中接入 tracker： [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:733)

## 8. 用 tracker 推理时会不会有误差累积？

会，这类基于 memory 的 tracker 天然就有“误差累积”风险。

原因是：

1. 当前帧的预测 mask 会被重新编码进 memory bank。
2. 后续帧又会继续读取这些 memory 来做推理。

也就是说，后续帧并不只是看原图，还会受到前面预测结果的影响。

这条链路在代码里很清楚：

- 当前帧预测完后写回 memory： [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:1023)
- 后续帧读取这些 memory： [sam3/model/sam3_tracker_base.py](/home/gaoqi/sam3/sam3/model/sam3_tracker_base.py:560)

所以如果某一帧开始偏了，比如：

- mask 边界慢慢漂
- 遮挡时把背景也记成前景
- 目标消失后仍然“硬跟”

这些误差就可能通过 memory 继续传到后面的帧，形成 drift。

不过，SAM3 也做了一些缓解：

1. 不是无限读取所有历史帧，而是只取部分 conditioning frames 和最近若干 memory。
2. 用 `object_score_logits`、`no_obj_ptr`、`no_obj_embed_spatial` 这类机制降低“目标不存在时还强行传播”的风险。
3. 在开启 `use_memory_selection` 时，会优先保留质量更好的 memory，减少差记忆污染后续推理。
4. 用户如果在中间帧重新加点或加 mask，相当于重新给 tracker 一个更可靠的锚点。

一句话总结：

`tracker` 的强项是能跨帧传播信息，但代价就是“预测结果会反过来影响后续预测”，因此误差累积是可能发生的，只是可以通过 memory 筛选、重新交互、缩短传播链等方式缓解。

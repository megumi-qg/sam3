# SAM3.1 Multiplex 学习笔记

这份笔记只讲 `SAM 3.1` 新增的 `Object Multiplex`，尽量用专业但容易理解的话来说明。

## 1. Multiplex 想解决什么问题？

原始的 SAM3 视频跟踪流程，基本上是：

`一个 object 跑一遍 tracker`

如果有很多 object，就要重复很多次，计算量会随着目标数近似线性增长。

官方在发布说明里写得很明确：

- SAM3 原始视频管线会独立处理每个 object
- SAM3.1 的 `Object Multiplex` 会把多个 object 分组后联合处理

对应说明：

- [RELEASE_SAM3p1.md](/home/gaoqi/sam3/RELEASE_SAM3p1.md:5)
- [RELEASE_SAM3p1.md](/home/gaoqi/sam3/RELEASE_SAM3p1.md:9)

所以 multiplex 的核心目标不是提升“单个目标”的理论能力，而是：

`让多目标跟踪更高效，减少重复计算`

## 2. Multiplex 是怎么做到“一次分割多个 object”的？

一句话版本：

`把多个 object 打包成一个 bucket，在共享的特征和共享的 memory 上联合推理，再把结果拆回各个 object。`

这里有三个关键词：

1. `bucket`
把多个 object 放进一个固定容量的小组里。

2. `mux`
把原来按 object 排列的数据，重新排成 bucket 形式的张量。

3. `demux`
推理结束后，再把 bucket 里的结果拆回每个 object。

这套数据空间变换由 `MultiplexState` 管理：

- [sam3/model/multiplex_utils.py](/home/gaoqi/sam3/sam3/model/multiplex_utils.py:20)

它的说明写得很直白：

- data space：按 object 排列
- multiplex space：按 `(num_buckets, multiplex_count, ...)` 排列

所以 multiplex 不是“模型真的失去 object 区分能力了”，而是：

`先把多个 object 规整地打包到同一个计算批次，再在结果端拆开`

## 3. bucket 是什么？

bucket 可以理解成“一个小批次 object 容器”。

每个 bucket 有固定槽位数 `multiplex_count`，比如默认常见是 `16`。  
如果有 30 个 object，就可能被分成 2 个 bucket：

- bucket 1: 16 个 object
- bucket 2: 14 个 object + 若干 padding 槽位

这件事在 `MultiplexState` 里管理得很清楚：

- 每个 bucket 都有固定长度
- 不够的槽位用 padding 占位
- 新 object 可以填已有空槽，也可以新开 bucket

对应代码：

- [sam3/model/multiplex_utils.py](/home/gaoqi/sam3/sam3/model/multiplex_utils.py:27)
- [sam3/model/multiplex_utils.py](/home/gaoqi/sam3/sam3/model/multiplex_utils.py:149)

所以你可以把 bucket 想成：

`把很多 object 分成若干组，每组一起送进模型处理`

## 4. 为什么这样会更快？

因为多目标之间有很多计算其实是可以共享的。

比如：

1. 同一帧的图像 backbone 特征，本来就不该为每个 object 重复算一遍。
2. 多个 object 的 memory 读取、mask 解码、后处理，很多操作都可以批量化。
3. GPU 更擅长处理一个较大的规则张量，而不是频繁地为很多小 object 单独启动一遍计算。

官方说明把它称为：

`shared-memory approach for joint multi-object tracking`

见：

- [RELEASE_SAM3p1.md](/home/gaoqi/sam3/RELEASE_SAM3p1.md:5)

从代码角度看，multiplex 版专门引入了：

1. multiplex memory encoder
2. multiplex transformer
3. multiplex controller / state
4. multiplex mask decoder

也就是说它不是只在外面套一层 batch，而是把“memory 读写和 mask 解码”都按多目标联合推理重写了一遍。

## 5. multiplex 和普通 tracker 的本质区别

普通 tracker 的思路更像：

`一个 object -> 一套 memory -> 一次 track_step`

multiplex tracker 的思路更像：

`多个 object -> 打包进 bucket -> 在共享计算图里一起做 track_step`

它仍然保留了 tracker 的核心思想：

- 读 memory
- 输出当前帧 mask
- 把当前帧结果再写回 memory

但不同点在于：

1. memory 不再是纯“单 object”视角，而是以 bucket 为组织单位进行联合编码。
2. object 的张量要不断在 `data space` 和 `multiplex space` 之间切换。
3. 输出的 `obj_ptr`、`maskmem_features` 等都可以是 bucket 形式，而不是简单的 `[num_obj, ...]`。

比如在 multiplex 跟踪代码里，`obj_ptr` 的注释就是：

- [sam3/model/video_tracking_multiplex.py](/home/gaoqi/sam3/sam3/model/video_tracking_multiplex.py:74)

它是：

`[num_buckets, multiplex_count, C]`

而不是普通 tracker 那种按 object 直接排开的形式。

## 6. multiplex 的 memory 是怎么处理的？

这是 multiplex 的关键点。

普通 tracker 里，memory encoder 往往是对“单 object mask + 当前帧特征”做融合。  
而在 multiplex 里，会先把多个 object 的 mask 通过 `mux(...)` 打包，再一起送去 memory encoder。

你可以看到：

- multiplex memory backbone 是专门创建的，不是直接复用普通版
- 它在构建时显式接收 `multiplex_count`

对应代码：

- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1069)

写 memory 时也会显式把 mask 做 `mux`：

- [sam3/model/video_tracking_multiplex.py](/home/gaoqi/sam3/sam3/model/video_tracking_multiplex.py:1674)

最后再在需要时 `demux` 回到按 object 的视角。

所以可以把 multiplex 的 memory 理解成：

`共享骨架 + 按 slot 区分 object 的联合 memory 表示`

它不是简单把很多 object 的 memory 拼接起来，而是为“多目标一起读写”专门设计了张量布局。

## 7. multiplex 架构就是“用 multiplex tracker 代替原有 tracker”吗？

可以说“基本上是”，但要稍微精确一点。

如果从视频跟踪子系统的角度看：

- 是的，SAM3.1 的 multiplex 版本确实是用一套新的 multiplex tracking stack 来替代原来“每个 object 独立跑 tracker”的方式。

因为在构建 SAM3.1 multiplex 视频模型时，走的是专门的 builder：

- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1183)

而不是旧的 `build_tracker(...)`。

再往上封装成 predictor 时，也是先构建 multiplex tracker model，再包成 predictor wrapper：

- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1379)
- [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1429)

但如果更严格地说，它不是“仅仅把旧 tracker 类名替换一下”。

因为 multiplex 版同时换掉了很多关键组件：

1. memory backbone 是 multiplex 专用版
2. transformer 是 decoupled / multiplex 专用版
3. mask decoder 也是 multiplex 相关实现
4. 还额外引入了 `MultiplexController` 和 `MultiplexState`

所以更准确的说法是：

`SAM 3.1 用一整套 multiplex tracking 架构，替代了原来逐 object 独立 tracking 的实现路径。`

## 8. multiplex 的核心数据流

```text
多个 object
   |
   v
MultiplexController / MultiplexState
把 object 分配到若干 buckets
   |
   v
mux: 按 bucket 重新组织 mask / pointer / memory 张量
   |
   v
共享的 backbone / transformer / memory encoder / mask decoder
在 bucket 维度上联合推理
   |
   v
得到 bucket 形式的 mask、obj_ptr、memory
   |
   v
demux: 拆回每个 object 的结果
   |
   v
继续写回 memory，并进入下一帧传播
```

## 9. 一句最重要的理解

普通 tracker 是：

`很多 object，很多次独立跟踪`

multiplex tracker 是：

`很多 object，先打包，再共享计算，一起跟踪`

它的本质不是“让一个 mask 同时代表多个 object”，而是：

`让多个 object 在同一套高效张量布局里联合完成 tracking`

## 10. 关键代码位置

- 发布说明： [RELEASE_SAM3p1.md](/home/gaoqi/sam3/RELEASE_SAM3p1.md:5)
- multiplex 状态管理： [sam3/model/multiplex_utils.py](/home/gaoqi/sam3/sam3/model/multiplex_utils.py:20)
- multiplex 跟踪主实现： [sam3/model/video_tracking_multiplex.py](/home/gaoqi/sam3/sam3/model/video_tracking_multiplex.py:1)
- multiplex memory backbone 构建： [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1069)
- multiplex 视频模型构建： [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1183)
- multiplex 视频 predictor 构建： [sam3/model_builder.py](/home/gaoqi/sam3/sam3/model_builder.py:1379)

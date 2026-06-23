# Query 速记

在当前项目里，`query` 这个词有两层含义，不能混在一起理解。

## 1. 数据层 query

这是训练数据组织里的 `find query`。

可以近似理解为：

- 一张图像或一帧 slice
- 配一个 prompt，例如某个解剖类别
- 形成一条“当前要分割什么”的任务请求

在 ACDC 这类单类别分割场景里，可以把它简单理解为：

- 一条 query ≈ “这张 slice 上，分割这个类别”

tracker 路径里常说的：

- `stage 间 query 数变化`

本质上说的是：

- 不同 3D volume 长度不同
- 到后面的 stage，有些样本已经没有对应 slice 了
- 所以当前 stage 参与计算的任务条目数变了

这次 tracker `val` 报错里说的 query，主要是这一层。

## 2. 模型内部 query

这是 transformer decoder 里的 learnable queries，也可以叫 object queries / query slots。

它们更像是一组可学习向量：

- 和图像特征、prompt 条件一起进入 decoder
- 每个 query slot 输出一个候选目标
- 后续再得到 `pred_logits / pred_boxes / pred_masks`

这就是之前“query 是一个可学习向量”的那种理解，对 `sam3 image model` 来说是对的。

## 3. `pad_n_queries = 200` 是什么

在 2D image model 配置里，例如：

- `sam3/train/configs/acdc/full_lora_100.yaml`

里面的：

- `pad_n_queries: 200`

更接近于：

- loss 端按最多 200 个 decoder queries 来对齐 / 归一化

它对应的是“模型内部 query”的数量设定，不是“数据层 query”的数量。

## 4. 一句话区分

- 数据层 query：这张图 / 这张 slice 上要分割什么
- 模型内部 query：decoder 里的一组可学习 query 向量

因此：

- 之前 2D 配置里说的 `200 个 query`，主要是模型内部 query
- tracker `val` 出错时说的 `stage 间 query 数变化`，主要是数据层任务条目数变化

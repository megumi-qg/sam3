# Weakly Supervised Medical Segmentation Paper Notes

这个文件用于整理 scribble / weak supervision 相关论文，目标不是做完整文献综述，而是沉淀成后续改进当前 SAM3 scribble segmentation / tracker 实验时可直接参考的方法库。

## Paper Index

| Paper | Venue / Year | Supervision | Domain | Key Idea | 对当前项目的用途 |
|---|---|---|---|---|---|
| Scribbles for All: Benchmarking Scribble Supervised Segmentation Across Datasets | NeurIPS Datasets and Benchmarks 2024 | Synthetic scribble from dense masks | Natural image segmentation | 自动从 dense mask 生成近似人工 scribble，并系统 benchmark scribble 方法 | 提供 scribble 生成、scribble 长度消融、benchmark 设计思路 |
| Addressing Inconsistent Labeling With Cross Image Matching for Scribble-Based Medical Image Segmentation | IEEE TIP 2025 | Scribble | Cardiac MRI | 用跨图像 reference set 做 pixel-level feature matching，缓解 scribble 不一致和信息缺失 | 可借鉴 cross-image feature bank、class token、query matching 来增强 SAM3 scribble 微调 |
| Revisiting 3D Medical Scribble Supervision: Benchmarking Beyond Cardiac Segmentation | arXiv 2025 | Synthetic scribble from dense masks | 3D multi-dataset medical segmentation | 提出 ScribbleBench，指出许多 cardiac scribble 方法跨任务泛化差，强推 nnU-Net + partial loss 强基线 | 提醒当前方法要做强基线、跨数据集/跨设置验证、充分 ablation，避免只在 ACDC 上过拟合 |
| DMSPS: Dynamically Mixed Soft Pseudo-label Supervision for Scribble-supervised Medical Image Segmentation | Medical Image Analysis 2024 | Scribble | ACDC / WORD / BraTS2020 | 双分支网络动态混合 soft pseudo-label，并用 uncertainty-filtered annotation expansion 做二阶段训练 | 可借鉴 soft pseudo seed、uncertainty expansion、双预测分支/ensemble confidence 来增强 SAM3 scribble tracker |
| ScribFormer: Transformer Makes CNN Work Better for Scribble-based Medical Image Segmentation | IEEE TMI 2024 | Scribble | Cardiac MRI / cardiac CT | CNN-Transformer 双分支融合局部细节与全局形状，并用 ACAM consistency 扩展弱监督信号 | 可借鉴 detector/tracker 双源互补、全局形状约束、activation/attention consistency |
| MedCL: Learning Consistent Anatomy Distribution for Scribble-supervised Medical Image Segmentation | MIDL 2025 | Scribble + image-level label | MSCMRseg / BTCV / MyoPS | 通过 feature mixing + prototype clustering 学习 anatomy distribution prior，支持 SAM 和 UNet | 可借鉴 anatomy prototype、局部/全局解剖一致性、SAM weak supervision adaptation |
| ModelMix: A New Model-Mixup Strategy to Minimize Vicinal Risk Across Tasks for Few-Scribble Based Cardiac Segmentation | MICCAI 2024 | Few scribbles + unlabeled images | Cardiac MRI / pathology CMR | 通过跨任务 encoder 参数插值构造 virtual models，并用 vicinal regularization 学习互补任务知识 | 可借鉴跨任务/跨模型 regularization，把 detector、tracker 或不同数据集模型连接起来 |
| ZScribbleSeg: A Comprehensive Segmentation Framework with Modeling of Efficient Annotation and Maximization of Scribble Supervision | Medical Image Analysis 2026 | Scribble | Multiple 2D/3D medical datasets | 建模高效 scribble，使用 mixup/occlusion 做 supervision augmentation，并用 EM 估计 class prior + spatial/shape prior 修正预测 | 可借鉴 scribble efficiency、class proportion prior、spatial prior 和 shape/continuity regularization |
| EFFDNet: A Scribble-Supervised Medical Image Segmentation Method with Enhanced Foreground Feature Discrimination | MICCAI 2025 / LNCS 2026 | Scribble | ACDC / NCI-ISBI | 利用 scribble 隐含的前景-背景语义，做 FBSL 特征对比约束和 FADC 前景上下文增强 | 可借鉴 foreground-background feature separation、scribble-box 前景增强、前景敏感 tracker 训练 |
| Soft Self-labeling and Potts Relaxations for Weakly-supervised Segmentation | Preprint / CV paper | Scribble | Pascal VOC / Cityscapes / ADE20K | 用 soft pseudo-label 作为优化变量，结合 Potts 空间正则和 collision CE 表达不确定性 | 可借鉴 uncertainty-aware pseudo label、soft tracker seed、空间连续性/边界一致性 loss |

---

## 1. Scribbles for All: Benchmarking Scribble Supervised Segmentation Across Datasets

- Authors: Wolfgang Boettcher, Lukas Hoyer, Ozan Unal, Jan Eric Lenssen, Bernt Schiele
- Venue / Year: NeurIPS 2024 Datasets and Benchmarks
- Paper file: `paper/Boettcher 等 - Scribbles for All Benchmarking Scribble Supervised Segmentation Across Datasets.pdf`
- Task: semantic segmentation with scribble supervision
- Domain: natural image / autonomous driving / many-class scene segmentation
- Supervision type: automatically generated scribble labels
- Main contribution type: dataset + scribble generation algorithm + benchmark

### Core Idea

这篇不是提出一个新的医学分割模型，而是提出 `Scribbles for All`：给定已有 dense segmentation mask，自动生成接近人工标注风格的 scribble label。它的核心价值是让 scribble-supervised segmentation 不再只依赖 PascalVOC / ScribbleSup 这类较简单 benchmark，而是扩展到 Cityscapes、KITTI360、ADE20K 等更复杂数据集。

### Motivation

作者认为现有 scribble segmentation 研究有两个问题：

- 可用 scribble 数据集太少，主要集中在 PascalVOC / ScribbleSup。
- PascalVOC 对 scribble 方法来说已经偏简单，无法充分暴露复杂场景中的小目标、多类别、物体边界等问题。

这个观点对我们有启发：如果论文只在 ACDC 上报告结果，容易被质疑 benchmark 单一。后续如果时间允许，可以考虑在 MSCMR、MSCMRSeg、CAMUS 或其它医学数据集上补充验证。

### Scribble Generation Method

输入是 dense segmentation mask，输出是每个 object / connected component 的一条 scribble。设计目标包括：

- mimic human annotations：scribble 大致穿过目标中心，避免长时间贴近边界；大而简单的目标 scribble 可以更粗，复杂目标需要更细。
- probabilistic generation：加入随机性，避免相似形状产生完全相同的 scribble。
- no boundary violation：硬约束 scribble 不能跨越类别边界。

生成流程大致是：

1. 按类别分离 dense mask。
2. 对每个类别做 connected component analysis。
3. 根据目标面积做 size-dependent erosion，避免 scribble 靠近边界。
4. 对复杂或非凸目标，必要时使用 skeleton / center-of-mass 替代点。
5. 在目标边缘采样点，寻找近似最远点对，得到目标主方向。
6. 沿主方向拟合曲线，并加入额外采样点，得到最终 scribble。

### Datasets

作者生成并发布了多个 scribble 数据集：

| Dataset | Source | Classes | Train / Val | Labeled pixels |
|---|---|---:|---:|---:|
| s4Pascal | PascalVOC | 21 | 10,582 / 1,449 | 2.25% |
| s4Cityscapes | Cityscapes | 19 | 2,975 / 500 | 2.36% |
| s4KITTI360 | KITTI360 | 16 | 49,000 / 12,000 | 2.49% |
| s4ADE20K | ADE20K | 150 | 25,574 / 2,000 | 4.71% |

它们还统计了 scribble 距离边界的比例、每张图平均 scribble 数等。这类统计对我们有价值：可以用来描述 ACDC scribble 标注有多稀疏，以及 pseudo seed / valid region 的覆盖率。

### Experiments

Benchmark 方法包括：

- TEL
- AGMM-SASS
- SASformer
- 一个简单 EMA / mean-teacher baseline，使用 SegFormer-B4

主要观察：

- s4Pascal 和人工 ScribbleSup 的训练效果非常接近，说明自动 scribble 生成具有合理性。
- PascalVOC 上 scribble 方法相对 fully-supervised 的 performance 大约可到 90% 以上，说明该 benchmark 已接近饱和。
- 在 s4Cityscapes、s4KITTI360、s4ADE20K 上，相对 performance 通常下降到约 80%，更能区分方法能力。
- scribble length ablation 很重要：不同方法对 scribble 长度缩短的鲁棒性不同。TEL 对缩短较稳健，SASformer 更敏感。

### Important Takeaways

- Scribble 方法的评估不能只看一个容易数据集；复杂数据集会暴露更多问题。
- 需要报告 scribble label 的稀疏度，例如 labeled pixel ratio、scribble 数量、scribble 长度。
- Scribble length / annotation sparsity ablation 是很好的论文实验，可以证明方法在更少标注下是否仍稳健。
- 自动 scribble 生成可以作为一种 controlled annotation simulation，用来分析标注质量、标注长度、标注偏差。

### Limitations

- 自动生成 scribble 依赖已有 dense mask，因此更适合 benchmark 构建，不适合真实低成本标注场景。
- 论文主要是自然图像语义分割，不是医学图像。
- 它没有提出新的医学弱监督 loss 或 3D continuity 方法。

### Relevance to My Project

这篇对当前项目的直接方法启发有限，但对论文实验设计很有帮助：

- 可以在 ACDC scribble 上报告 `labeled pixel ratio`，突出医生标注量降低。
- 可以做 scribble sparsity / scribble length ablation：例如 100%、50%、25% scribble 有效区域。
- 可以把 pseudo seed bank 的覆盖率、confidence 分布、seed slice 数量作为方法分析。
- 如果后续生成 synthetic scribble，可以用于 controlled experiment：比较人工 scribble、synthetic scribble、pseudo seed 的影响。

### Possible Ideas to Borrow

- 加入 `annotation efficiency analysis`：标注像素比例 vs Dice。
- 加入 `scribble length ablation`：减少 scribble supervision 看 SAM3 是否仍稳健。
- 加入 `label distribution statistics`：不同类别 LV/MYO/RV 的 scribble 覆盖是否不均衡。
- 在论文中强调：你的方法不仅追求 Dice，还要关注 sparse annotation 下的 robustness。

---

## 2. Addressing Inconsistent Labeling With Cross Image Matching for Scribble-Based Medical Image Segmentation

- Authors: Jingkun Chen, Wenjian Huang, Jianguo Zhang, Kurt Debattista, Jungong Han
- Venue / Year: IEEE Transactions on Image Processing, 2025
- DOI: 10.1109/TIP.2025.3530787
- Paper file: `paper/Chen 等 - 2025 - Addressing Inconsistent Labeling With Cross Image Matching for Scribble-Based Medical Image Segmenta.pdf`
- Task: scribble-based medical image segmentation
- Domain: cardiac MRI
- Supervision type: scribble
- Main contribution type: feature matching / representation learning module

### Core Idea

这篇的核心问题是：scribble 标注不仅稀疏，而且存在明显不一致性。不同图像、不同 annotator、不同心动周期下，同一类结构的 scribble 大小、形状、宽度、位置可能不同，导致网络从 scribble 学到的监督信号不稳定。

作者提出 cross-image matching：从不同图像的 scribble 位置提取 feature，构建一个 reference set。训练时，当前图像中的 pixel feature 会与 reference set 中的 pixel-level queries 和 class-tokens 做匹配，从而得到更平滑、更一致的表示。

### Motivation

医学 scribble 标注的难点包括：

- incomplete：scribble 只覆盖很少一部分像素。
- subjective：不同人画法不同。
- inconsistent：同类结构的 scribble 形状、大小、位置可能变化很大。
- cardiac motion：舒张期 / 收缩期图像差异导致结构外观变化明显。

这和我们当前项目很相关：仅用 valid-region loss 只能解决“在哪里监督”的问题，但没有解决“scribble 标注不一致”和“未标注像素如何借助其它图像获得稳定类别信息”的问题。

### Method Overview

网络包含 encoder、decoder，以及连接二者的 representation learning module。encoder 输出 feature 后经过两个 MLP head：

- contextual projection MLP：通道降到一半，保留上下文信息。
- comparison projection MLP：通道降到 1/8，用于降低匹配计算成本。

方法包含两个关键组件：

- reference set construction
- representation learning module

### Reference Set

Reference set 来自 scribble 位置的 feature，包含两类信息：

| Component | 来源 | 作用 |
|---|---|---|
| pixel-level queries | 所有 scribble 位置的 pixel features | 提供局部、多样化的 pixel-level reference |
| semantic-level class-tokens | 每个类别的 scribble feature 均值 / prototype | 提供全局类别语义 guidance |

Pixel-level queries 使用 FIFO queue 存储和更新。它保存来自不同训练 iteration / image 的局部 scribble feature，使当前图像能和跨图像的 reference pixels 匹配。

Class-tokens 每个类别只维护一个 token，并用 momentum update 更新：

```text
m_{t+1} = 0.99 * m_t + 0.01 * theta_t
```

其中 `theta_t` 是当前 batch 中该类别 scribble feature 的均值。作者认为单个 momentum class-token 比存多个 class-token 更稳，因为 scribble 太稀疏，多 token 会引入同类内部的不一致 guidance。

### Representation Learning Module

该模块包含两个单元：

- representation smoothing unit
- class regression unit

Representation smoothing：

- 当前 training pixels 与 reference set 中的 pixel-level queries 做 similarity matching。
- 使用 softmax similarity 得到权重。
- 用 reference queries 的 contextual representations 修正当前 pixel feature。
- 作用是平滑表示，减少噪声和无关信息。

Class regression：

- 当前 training pixels 与 semantic-level class-tokens 做 matching。
- 根据 class-token guidance 得到 class-aware regressed feature。
- 作用是引入类别级语义约束，缓解 scribble 信息缺失。

最终修正后的 feature 由三部分组合：

- smoothed pixel-level feature
- class-token regressed feature
- 原始 contextual feature

然后送入 decoder 做分割。

### Loss Design

使用两个 loss：

```text
L = L_pCE + alpha * L_Dice
```

- `L_pCE`: partial cross entropy，只在 scribble 标注像素上计算。
- `L_Dice`: pseudo-label based Dice loss。
- `alpha = 0.5`。

这个 loss 本身并不复杂，真正创新点在 feature matching / reference set，而不是 loss 公式。

### Datasets

作者在三个 cardiac dataset 上实验：

| Dataset | Data | Classes | Annotation |
|---|---|---|---|
| ACDC Scribble | 100 patients / 200 3D scans | RV, MYO, LV | scribble from prior work |
| MS-CMRSeg Scribbles | 25 3D LGE MS-CMR images | BG, RV, MYO, LV | manual scribble |
| MS-CMRSeg Challenge | 90 3D images from LGE + bSSFP | RV, MYO, LV | generated scribble, fewer pixels |

实验使用 80% / 20% split，five-fold cross-validation。指标为 DSC 和 95HD，并使用 Wilcoxon rank-sum test 统计显著性。

### Results

主要结果：

- ACDC Scribble：mean DSC 达到约 `87.9%`，比第二名高 `0.7%`；MYO 提升 `1.3%`，LV 提升 `1.0%`，RV 排第二。
- MS-CMRSeg Scribbles：mean DSC `85.3%`，比最近竞争方法高 `1.5%`。
- MS-CMRSeg Challenge：mean DSC `73.5%`，比之前最佳方法高 `2.4%`；RV `72.7%`，MYO `65.2%`，LV `82.6%`。
- 作者指出 95HD 对 scribble weak supervision 可能不稳定，小错误会对 95HD 造成过大影响；Dice 更能反映整体 overlap。

### Ablation

关键消融：

- Pixel-level queries：ACDC DSC 从 `87.1%` 提升到 `87.9%`；MS-CMRSeg Challenge 从 `71.6%` 到 `73.5%`。
- Semantic-level class-tokens：ACDC DSC 从 `87.2%` 提升到 `87.9%`；MS-CMRSeg Challenge 从 `71.0%` 到 `73.5%`。
- Class-token 更新策略：单 token + momentum update 优于 class-token queue；原因是 scribble 稀疏时，多 token 会加重同类不一致 guidance。
- Soft similarity 优于 hard similarity；说明软匹配比只选择最相似 feature 更稳定。
- 增加 scribble bias 后，该方法仍比 MPLS 更稳，说明 cross-image reference set 能缓解不一致标注。

### Important Takeaways

- 只改 partial CE / valid region loss 可能不够，因为它没有利用跨图像同类结构的共同表示。
- Scribble supervision 的核心问题不仅是 sparse，还有 inconsistent。
- Feature bank / reference set 是一个很适合 weak supervision 的设计：它把少量 scribble pixels 聚合成跨图像的类别知识。
- 对于医学图像，class-level prototype 可能比每张图单独学习更稳定。

### Limitations

- 需要维护 feature queue / class token，训练实现更复杂。
- 方法主要针对 CNN / encoder-decoder 网络，直接迁移到 SAM3 需要决定 reference set 接在哪个 feature 层。
- 它仍依赖 pseudo-label Dice，pseudo-label 质量会影响训练。
- 它没有直接利用 3D slice continuity 或 tracker propagation。

### Relevance to My Project

这篇和当前项目非常相关。你现在的 SAM3 scribble baseline 主要是：

- SAM3 detector fine-tuning
- valid-region / partial mask loss
- pseudo-label / presence 相关训练逻辑

但它缺少“跨图像类别一致性建模”。Chen 这篇正好补这个空缺。

当前项目可借鉴方向：

1. 在 SAM3 image encoder / DETR hidden states 上建立 class prototype bank。
2. 从 scribble valid pixels 或高置信 pseudo mask 中提取 class-specific features。
3. 对 unlabeled pixels 或 detector queries 做 cross-image feature matching。
4. 使用 momentum class token 缓解 scribble 标注不一致。
5. 将 tracker memory bank 与 cross-image reference bank 区分开：tracker memory 是 intra-volume temporal memory，reference bank 是 inter-image / inter-volume class memory。

### Possible Ideas to Borrow

#### Idea A: SAM3 Scribble Class Prototype Bank

为 LV / MYO / RV 各维护一个 momentum prototype：

```text
proto_c = 0.99 * proto_c + 0.01 * mean(feature at scribble/pseudo-positive pixels)
```

训练时增加 prototype consistency：

```text
feature_i should be close to proto_class
```

这可以作为比单纯 valid-region loss 更 fancy 的 scribble-specific 模块。

#### Idea B: Cross-Image Query Bank for Pseudo Label Refinement

把高置信 pseudo mask 内的 features 存入 queue，作为 pixel-level queries。当前 slice 的不确定区域与 query bank 做相似度匹配，得到 refined pseudo label 或 consistency target。

这和你的 pseudo seed bank 思路天然兼容。

#### Idea C: Tracker Memory + Class Reference Dual Memory

可以设计两个 memory：

- tracker memory：同一个 volume 内跨 slice 传播。
- class reference memory：跨 patient / cross-image 的类别 prototype。

论文表述可以是：

> intra-volume tracker memory captures anatomical continuity, while inter-volume class reference memory mitigates inconsistent scribble annotations.

这会比单独 tracker 更有论文创新感。

#### Idea D: Consistency-aware Merge with Prototype Confidence

当前 tracker / detector merge 只看 score 不够。可以额外看预测 mask 内 feature 与 class prototype 的相似度：

```text
Q(mask) = model_score + prototype_similarity - continuity_penalty
```

如果 tracker mask 的 prototype similarity 更高，则更可信。这可能比单纯 `tracker_score > detector_score` 更有解释力。

### How It May Guide Next Experiments

短期可做：

- 在当前 inference diagnosis 中，离线提取 detector/tracker mask 区域的 image feature，比较它们与类别 prototype 的相似度，看 prototype similarity 是否能帮助判断 oracle source。

中期可做：

- 在 SAM3 detector fine-tuning 中加入 class prototype consistency loss。

长期可做：

- 设计 `Scribble-guided Dual Memory Adaptation`：inter-image class memory + intra-volume tracker memory。

---

## 3. Revisiting 3D Medical Scribble Supervision: Benchmarking Beyond Cardiac Segmentation

- Authors: Karol Gotkowski, Klaus H. Maier-Hein, Fabian Isensee
- Venue / Year: arXiv 2025
- Paper file: `paper/Gotkowski 等 - 2025 - Revisiting 3D Medical Scribble Supervision Benchmarking Beyond Cardiac Segmentation.pdf`
- Task: 3D medical image segmentation with scribble supervision
- Domain: multiple medical segmentation datasets, not only cardiac MRI
- Supervision type: generated scribble labels from dense masks
- Main contribution type: benchmark + strong baseline + validation critique

### Core Idea

这篇论文的核心不是提出一个很复杂的新 scribble 模型，而是重新审视 3D 医学 scribble supervision 领域的评价方式。作者认为，很多方法主要在 ACDC / MSCMR 这类 cardiac benchmark 上做实验，因此容易形成局部最优：在 cardiac 上看起来很强，但换到其它器官、其它模态、其它分割任务时泛化明显下降。

作者提出 `ScribbleBench`，覆盖 7 个医学 3D segmentation 数据集，并指出一个很重要的结论：很多看起来 fancy 的 scribble-specific novelty 在 broader benchmark 上反而会降低泛化，而简单的 `nnU-Net + partial loss` 是一个被低估的强基线。

### Motivation

作者认为 scribble supervision 要成为真实可用的方法，需要满足几个要求：

- `R1`: 能跨任务、跨器官、跨模态泛化，而不是只在 ACDC / MSCMR 上好。
- `R2`: 需要系统 benchmark，避免用单一 cardiac benchmark 得出过强结论。
- `R3`: 方法不能过度绑定某个架构或数据集，应尽量容易接入不同 segmentation backbone。
- `R4`: 应该使用成熟实践，例如 3D architecture 通常优于 2D slice-wise 方法。
- `R5`: 需要开源实现保证可复现。

这个观点对当前项目非常重要：如果我们在论文里加入 tracker 模块，必须证明它不是只在某个 seed / merge 规则下偶然有效，而是要通过清晰 ablation 和 failure analysis 说明它到底贡献在哪里。

### ScribbleBench

ScribbleBench 包含 7 个医学 3D segmentation 数据集：

| Dataset | Domain |
|---|---|
| ACDC | cardiac cine MRI |
| MSCMR | cardiac MRI |
| WORD | abdominal multi-organ CT |
| LiTS | liver / tumor CT |
| BraTS2020 | brain tumor MRI |
| AMOS2022 | abdominal multi-organ CT/MRI |
| KiTS2023 | kidney tumor CT |

作者从 dense mask 自动生成 scribble。每个 slice、每个类别生成两类 scribble：

- interior scribble：位于目标内部，用 NURBS 生成，模拟人工在目标内部画线。
- border scribble：覆盖一小段边界，并加入随机 offset，补充边界信息。

这个生成方式和前面的 `Scribbles for All` 有点类似，但这里重点是 3D 医学 benchmark 和跨任务泛化。

### Main Findings

论文总结了三个主要 validation pitfalls：

- `P1`: limited evaluation hides lack of generalization。只看 cardiac benchmark 会让方法显得接近 solved，但换到 ScribbleBench 后性能大幅下降。
- `P2`: superficial novelties may degrade performance。很多复杂模块在 cardiac 上提升明显，但跨任务后可能没有收益甚至负收益。
- `P3`: simple generalizing methods are neglected。简单方法在 cardiac 上不一定最高，但跨任务更稳。

代表性结果如下：

| Method | Cardiac ACDC | Cardiac MSCMR | ScribbleBench Mean |
|---|---:|---:|---:|
| ShapePU | 0.850 | 0.844 | 0.369 |
| ScribFormer | 0.881 | 0.840 | 0.548 |
| CycleMix | 0.884 | 0.863 | 0.559 |
| DMSPS | 0.891 | 0.874 | 0.697 |
| nnU-Net + DenseCRF | 0.741 | 0.732 | 0.738 |
| nnU-Net + WORD-style simple method | 0.519 | 0.645 | 0.755 |

这里最刺眼的是：一些 cardiac 上高分的方法，ScribbleBench mean 反而不如简单方法。作者想说明，单数据集上的新模块很容易过拟合 benchmark 习惯。

### Strong Baseline: nnU-Net + Partial Loss

作者提出一个非常强的基线：`nnU-Net + partial loss`。这里的 partial loss 不只限于 partial CE，而是把 nnU-Net 常用的 CE + Dice 都改成只在 scribble-labeled voxels / valid regions 上计算。

关键结果：

| Method | ScribbleBench Mean |
|---|---:|
| nnU-Net + pCE 2D | 0.718 |
| nnU-Net + pL 2D | 0.752 |
| nnU-Net + pCE 3D | 0.770 |
| nnU-Net + pL 3D | 0.813 |
| nnU-Net dense supervision | 0.856 |

作者的结论是：`nnU-Net + pL` 虽然在 cardiac benchmark 上不是最高，但在 ScribbleBench 上泛化最好，甚至超过很多专门设计的 scribble 方法。

### Important Takeaways

- 只在 ACDC 上证明 tracker 或 weak-supervision 模块有效是不够稳的，论文里至少要有强 baseline 和系统 ablation。
- 简单 valid-region / partial loss 不是弱点，它可能是一个非常强、可泛化的基础策略。
- 复杂模块必须证明“跨设置仍有用”，否则容易被认为是 cardiac-specific tuning。
- 3D continuity / 3D architecture 对医学 scribble 很关键。即使 SAM3 是 slice/video 形式，也需要强调如何利用 volume continuity。
- 如果 tracker 加入后没有稳定提升 detector baseline，就不应该强行包装成主贡献；更合理的定位是诊断其在低置信 slice、稀疏 scribble、跨 slice consistency 中是否有独立价值。

### Limitations

- ScribbleBench 的 scribble 是从 dense mask 自动生成的，不等同于真实医生标注。
- 论文重点是 benchmark 和 generalization critique，不提供复杂的 SAM / foundation model adaptation 方案。
- 它强调 nnU-Net 强基线，因此对 SAM3 这类 foundation model 的迁移需要我们自己设计实验。

### Relevance to My Project

这篇对当前项目像一个“审稿人视角提醒”：

- 你的 SAM3 scribble detector 目前只改 valid-region loss，但如果效果很强，这本身可以被合理解释为 strong foundation model + simple partial loss baseline。
- Tracker 模块如果不能提升 overall Dice，需要更谨慎地定位：它可以作为 anatomical continuity / uncertainty-aware correction，而不是简单声称整体超过 detector。
- 论文实验应该加入 `strong baseline comparison`：scribble SAM3 detector、scribble SAM3 detector + tracker、tracker-only、confidence-aware merge、oracle merge。
- 需要报告 failure case：哪些 slice detector 更好，哪些 slice tracker 更好，tracker 是否主要帮助低置信或断裂区域。
- 如果时间允许，最好在 ACDC 之外补一个数据集，哪怕规模小，也能回应 Gotkowski 对 cardiac overfitting 的担忧。

### Possible Ideas to Borrow

#### Idea A: Treat Simple Partial Loss as a Strong Baseline, Not a Weak Contribution

论文写作上可以把当前 detector baseline 表述成：

```text
SAM3 foundation representation + scribble-valid partial supervision already forms a strong weakly supervised baseline.
```

然后 tracker 的作用不是替代 detector，而是尝试补充 3D / temporal continuity。

#### Idea B: Add Generalization-oriented Ablation

针对 tracker，不只报告一个 Dice：

- seed threshold ablation
- merge policy ablation
- tracker contribution ratio
- oracle source upper bound
- low-confidence detector slice subset performance

这样可以避免“模块很复杂但整体没提升”的尴尬。

#### Idea C: Benchmark-style Reporting

在论文中加入一个小表：

| Setting | Purpose |
|---|---|
| detector only | strong SAM3 scribble baseline |
| tracker only | tracker independent ability |
| detector-first merge | SAM3 default-like inference |
| confidence-aware merge | proposed uncertainty-aware integration |
| oracle merge | upper bound / room for better selection |

这会让 tracker 实验更像科学分析，而不是盲目堆模块。

---

## 4. DMSPS: Dynamically Mixed Soft Pseudo-label Supervision for Scribble-supervised Medical Image Segmentation

- Authors: Meng Han, Xiangde Luo, Xiangjiang Xie, Wenjun Liao, Shichuan Zhang, Tao Song, Guotai Wang, Shaoting Zhang
- Venue / Year: Medical Image Analysis, 2024
- Paper file: `paper/Han 等 - 2024 - DMSPS Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segment.pdf`
- Task: scribble-supervised medical image segmentation
- Domain: cardiac MRI, abdominal CT, brain tumor MRI
- Supervision type: scribble
- Main contribution type: dual-branch soft pseudo-label learning + uncertainty-based annotation expansion

### Core Idea

DMSPS 的核心是：scribble 只标注少量像素，如果直接用 hard pseudo-label，很容易过度自信并传播错误。作者提出一个双分支网络 `DB-Net`，两个 decoder 共享 encoder，但输出略有差异。训练时动态混合两个 decoder 的 softmax 概率，生成 soft pseudo-label 来监督两个分支。

然后做二阶段训练：第一阶段用 raw scribble 训练；第二阶段用第一阶段模型产生低不确定性的 pseudo labels，把 sparse scribble 扩展成更大的 high-confidence annotation 区域，再重新训练。

### Motivation

作者认为 pseudo-label 方法有两个问题：

- single-model bias：模型用自己的预测监督自己，容易强化自身偏差。
- hard pseudo-label overconfidence：错误 pseudo label 一旦变成 one-hot，会给模型很强的错误监督。

DMSPS 对应地提出：

- 用两个 decoder 的动态 mixture 减少单分支 bias。
- 用 soft pseudo-label 保留类别不确定性，避免 hard label 的过强错误梯度。
- 用 uncertainty filtering 只把可靠区域扩展为第二阶段 annotation。

### Method Overview

网络结构是 shared encoder + two decoders：

| Component | 作用 |
|---|---|
| shared encoder | 提取公共图像特征 |
| main decoder | 输出预测 `p1` |
| auxiliary decoder | 输出预测 `p2`，带 perturbation / dropout |
| dynamically mixed soft pseudo-label | 用 `p1` 和 `p2` 随机混合得到 |
| uncertainty-based expansion | 从第一阶段预测中筛选可靠区域作为第二阶段扩展标注 |

动态 soft pseudo-label：

```text
p_hat = alpha * p1 + (1 - alpha) * p2
alpha ~ U(0, 1)
```

训练时 `p_hat` 会 detach，只作为 pseudo target，不让梯度反向更新 pseudo-label 本身。作者强调不使用 `argmax(p_hat)`，因为 argmax 会丢掉不确定性，并把错误标签变成过度自信的 hard label。

### Loss Design

第一阶段包含两个 loss：

```text
L = L_pCE + lambda * L_SPS
```

其中：

- `L_pCE`: partial cross entropy，只在 scribble labeled pixels 上计算，两个 decoder 都参与。
- `L_SPS`: soft pseudo-label supervision，让两个 decoder 分别学习 detached `p_hat`。
- 论文最终采用 CE 作为 soft pseudo-label loss，比 MSE / Dice / KL 更好。

第二阶段先用第一阶段模型生成扩展标注：

```text
p_bar = 0.5 * p1 + 0.5 * p2
y_bar = argmax(p_bar)
U(i) = normalized_entropy(p_bar_i)
M = U < tau
s_tilde = largest_connected_component(y_bar * M)
```

然后把 `s_tilde` 作为 uncertainty-filtered high-confidence expanded annotation，在扩展区域上计算 partial CE：

```text
L' = L'_pCE + lambda * L_SPS
```

这里的关键是 `M = U < tau`：只有低不确定性区域才会被当作可靠 pseudo annotation。

### Datasets

作者在三个数据集上验证：

| Dataset | Task | Setting |
|---|---|---|
| ACDC | RV / MYO / LV cardiac MRI segmentation | 2D UNet, raw scribbles from prior work |
| WORD | abdominal multi-organ CT segmentation | 3D UNet |
| BraTS2020 | brain tumor MRI segmentation | 3D UNet |

这点和 Gotkowski 的批评形成呼应：DMSPS 至少不是只在 ACDC 上验证。

### Results

核心结果：

| Dataset | pCE baseline | DMSPS stage 1 | DMSPS stage 2 |
|---|---:|---:|---:|
| ACDC | 50.46% DSC | 88.94% DSC | 89.51% DSC |
| WORD | 75.46% DSC | 86.79% DSC | 87.56% DSC |
| BraTS2020 | about 53.61% DSC | 74.28% DSC | 76.53% DSC |

ACDC raw scribble 下，DMSPS stage 2 达到 `89.51%` mean DSC，接近 fully supervised upper bound `90.31%`。当 scribble 长度缩短到 `1/4` 时，GCL 从 `87.89%` 降到 `84.75%`，DMSPS 只从 `89.51%` 降到 `87.59%`，说明 soft pseudo-label + uncertainty expansion 对更稀疏 scribble 更稳。

### Ablation

ACDC 上关键消融：

| Method | Mean DSC |
|---|---:|
| pCE baseline | 50.46% |
| DB-Net only | 59.77% |
| DB-Net + consistency regularization | 87.71% |
| DB-Net + cross pseudo supervision | 87.11% |
| fixed alpha = 0.5 | 87.81% |
| hard pseudo-label | 87.83% |
| DMSPS stage 1 | 88.94% |
| DMSPS stage 2 | 89.51% |
| fully supervised | 90.31% |

soft pseudo-label loss 对比：

| Loss for soft pseudo-label | ACDC mean DSC |
|---|---:|
| MSE | 86.86% |
| Dice | 87.97% |
| KL | 88.69% |
| CE | 88.94% |

作者还分析了两个关键超参：

- `lambda`: soft pseudo-label supervision 权重，论文中 `1.0` 到 `10.0` 相对稳定，最佳约为 `8.0`。
- `tau`: uncertainty threshold，ACDC 最佳约为 `0.1`，更复杂的 WORD / BraTS2020 需要更大的 `tau`。

### Important Takeaways

- 对 scribble weak supervision，pseudo-label 不应该只用 hard mask；soft probability distribution 本身包含有价值的不确定性。
- 双分支 / ensemble-style prediction 可以给 uncertainty estimation 提供更可靠依据。
- 二阶段 annotation expansion 是一个很强的 scribble-specific 设计：先从 sparse scribble 学一个初始模型，再把低不确定区域扩展成训练监督。
- 对 extremely sparse scribble，soft pseudo-label 比 hard pseudo-label 更稳。
- 只靠 partial CE 会浪费大量 unlabeled pixels；DMSPS 的核心价值是把 unlabeled pixels 以 soft / confidence-aware 的方式重新纳入训练。

### Limitations

- 需要两个 decoder 和二阶段训练，训练成本高于单模型 partial CE。
- 主要基于 U-Net / 3D U-Net，迁移到 SAM3 时不能直接复制 decoder 结构。
- 二阶段 expanded annotation 依赖第一阶段模型质量；如果第一阶段校准差，扩展标注仍可能带噪声。
- 它没有显式利用 SAM3 tracker / memory bank，也没有处理 detector-tracker merge。

### Relevance to My Project

这篇和你当前 scribble SAM3 tracker 非常相关，因为你现在已经有：

- scribble image model baseline
- pseudo seed bank
- confidence score / threshold
- tracker propagation
- confidence-aware merge 诊断

DMSPS 给出的启发是：不要只把 pseudo mask 当作 hard seed，而要保留 soft score / uncertainty，并让它参与训练或 merge。

当前项目可借鉴方向：

1. 用 scribble SAM3 detector 生成 `soft pseudo mask`，不要只保存二值 mask。
2. 为每个 pseudo seed 保存 uncertainty，例如 entropy、top1-top2 margin、mask stability、score head confidence。
3. 训练 tracker 时，只用高置信区域初始化 memory；低置信区域不作为监督或仅作为 soft consistency。
4. 让 detector 和 tracker 形成双预测源，类似 DMSPS 的两个 decoder：二者不一定谁绝对正确，而是可以用 agreement / disagreement 估计 uncertainty。
5. 第二阶段可以训练一个 `SAM3 scribble detector-tracker stage2`：用 stage1 detector + tracker agreement 生成 expanded annotations，再继续微调。

### Possible Ideas to Borrow

#### Idea A: Detector-Tracker Soft Pseudo Supervision

把 detector prediction 和 tracker prediction 看成两个 pseudo-label sources：

```text
p_mix = alpha * p_detector + (1 - alpha) * p_tracker
```

然后在 unlabeled pixels 上加入 soft consistency loss。这里 `alpha` 可以随机采样，也可以由 confidence / temporal distance 控制。

这会比简单 tracker-first / detector-first merge 更像一个 scribble-specific 训练方法。

#### Idea B: Agreement-based Pseudo Seed Expansion

只在 detector 和 tracker 都高置信、且 mask overlap 较高的区域生成 expanded pseudo annotation：

```text
M_reliable = high_conf(detector) AND high_conf(tracker) AND high_iou(det, trk)
```

这可以自然解释为：

> detector provides semantic confidence, tracker provides anatomical continuity, and their agreement defines reliable weak supervision.

#### Idea C: Soft Seed Memory

当前 tracker seed 如果是 binary mask，可能会把 detector 错误变成强记忆。可以借鉴 DMSPS，把 seed mask 改成 soft / confidence-weighted memory：

```text
memory_weight = confidence * (1 - uncertainty)
```

高置信区域写入强 memory，低置信区域弱写入或不写入。

#### Idea D: Stage-2 SAM3 Scribble Adaptation

训练路线可以设计成：

1. Stage 1: train scribble SAM3 detector with partial loss.
2. Generate detector pseudo masks and uncertainty on train set.
3. Run tracker propagation from high-confidence seeds.
4. Select detector-tracker agreement regions as expanded annotations.
5. Stage 2: fine-tune detector + tracker with partial loss + soft pseudo consistency.

这个方案比“单纯加 tracker”更有论文方法感，也更符合 scribble weak supervision 的主题。

---

## 5. ScribFormer: Transformer Makes CNN Work Better for Scribble-based Medical Image Segmentation

- Authors: Zihan Li, Yuan Zheng, Dandan Shan, Shuzhou Yang, Qingde Li, Beizhan Wang, Yuanting Zhang, Qingqi Hong, Dinggang Shen
- Venue / Year: IEEE Transactions on Medical Imaging, 2024
- Paper file: `paper/Li 等 - 2024 - ScribFormer Transformer Makes CNN Work Better for Scribble-Based Medical Image Segmentation.pdf`
- Task: scribble-supervised medical image segmentation
- Domain: cardiac MRI / cardiac CT
- Supervision type: scribble
- Main contribution type: CNN-Transformer hybrid architecture + attention-guided CAM consistency

### Core Idea

ScribFormer 的核心问题是：scribble 只监督少量像素，普通 CNN 受局部感受野限制，很难从稀疏 scribble 中学到全局形状信息。作者提出一个 triple-branch framework，把 CNN 的局部细节、Transformer 的全局语义，以及 attention-guided class activation map 结合起来。

它的结构包括：

- CNN branch：偏局部纹理和边界细节。
- Transformer branch：偏全局依赖和整体形状。
- ACAM branch：用深层 attention-guided CAM 约束浅层 ACAM，缓解 scribble 未标注区域无法训练的问题。

### Motivation

作者认为 scribble-supervised medical segmentation 的难点不是只在“标注少”，还在于：

- sparse scribble 无法直接描述完整目标区域。
- CNN 容易只根据局部信息产生噪声区域。
- CAM / pseudo-label 方法容易只激活最有判别性的局部区域，而不是完整器官。
- Transformer 有全局建模能力，但纯 Transformer 可能缺少局部细节。

因此 ScribFormer 选择 CNN + Transformer 协同，而不是单纯用 CNN 或单纯用 ViT。

### Method Overview

ScribFormer 使用 hybrid CNN-Transformer encoder。CNN feature map 和 Transformer patch embedding 通过 `Feature Coupling Units` 对齐通道与空间维度后融合。

训练时有两个 segmentation output：

| Output | 来源 | 作用 |
|---|---|---|
| `y_CNN` | CNN decoder | 保留局部细节 |
| `y_Trans` | Transformer decoder | 建模全局形状 |

然后用两类监督：

- scribble partial CE：`y_CNN` 和 `y_Trans` 都只在 scribble labeled pixels 上计算。
- dynamic pseudo-label：动态混合 `y_CNN` 和 `y_Trans` 生成 hard pseudo-label，再用 Dice loss 监督两个 branch。

动态 pseudo-label 形式为：

```text
Y = argmax(alpha * y_CNN + (1 - alpha) * y_Trans)
```

其中 `alpha` 每个 iteration 随机采样。

### ACAM Branch

ACAM 是这篇论文比较有特色的部分。作者用 CNN branch 不同层的 activation map 表示模型关注区域，然后让浅层 ACAM 向深层 ACAM 对齐。

直觉是：

- 深层 feature 融合了 Transformer 的 global context，更接近目标完整区域。
- 浅层 feature 容易只关注局部纹理。
- 让浅层 ACAM 学深层 ACAM，可以把监督从 scribble pixels 扩展到更大的 attention 区域。

这不是直接生成 dense pseudo mask，而是通过 feature / attention consistency 给未标注区域更多约束。

### Loss Design

总损失为：

```text
L_total = lambda_1 * L_ss + lambda_2 * L_pl + lambda_3 * L_acam
```

其中：

- `L_ss`: scribble-supervised partial cross entropy，只在 scribble labeled pixels 上计算。
- `L_pl`: pseudo-label Dice loss，监督 CNN / Transformer 两个分支。
- `L_acam`: ACAM consistency loss，让浅层 attention map 对齐深层 attention map。

论文中使用：

```text
(lambda_1, lambda_2, lambda_3) = (1, 0.5, 0.1)
```

### Datasets

| Dataset | Modality | Classes | Split / Setting |
|---|---|---|---|
| ACDC | cine MRI | LV / RV / MYO | 70 / 15 / 15，通常用 35 scribble training images |
| MSCMRseg | LGE MRI | LV / RV / MYO | 25 / 5 / 15 |
| HeartUII | cardiac CT | LV / LA / RV / RA / AO / MYO | private dataset, 53 / 13 / 16 |

### Results

核心结果：

| Dataset | Best prior scribble method | ScribFormer | Fully supervised reference |
|---|---:|---:|---:|
| ACDC | CycleMix 0.848 | 0.888 | CycleMix-F 0.886, nnU-Net 0.920 |
| MSCMRseg | CycleMix 0.800 | 0.839 | CycleMix-F 0.810, nnU-Net 0.902 |
| HeartUII | CycleMix 0.810 | 0.833 | UNet++-F 0.833, nnU-Net 0.914 |

作者强调 ScribFormer 在某些设置下甚至超过普通 fully supervised CNN baseline，但仍低于 nnU-Net dense upper bound。

### Ablation

关键消融：

| Model | CNN | Transformer | ACAM | ACDC Avg Dice |
|---|---|---|---|---:|
| #1 | yes | no | no | 0.678 |
| #2 | no | yes | no | 0.672 |
| #3 | yes | no | yes | 0.713 |
| #4 | yes | yes | no | 0.872 |
| #5 | yes | yes | yes | 0.888 |

Decoder 消融：

| Decoder | ACDC Avg Dice |
|---|---:|
| CNN decoder | 0.692 |
| Transformer decoder | 0.830 |
| CNN + Transformer decoder | 0.888 |

Loss 消融：

| Loss setting | ACDC Avg Dice |
|---|---:|
| `L_ss` only | 0.780 |
| `L_ss + L_acam` | 0.806 |
| `L_ss + L_pl` | 0.866 |
| `L_ss + L_pl + L_acam` | 0.888 |

说明 pseudo-label loss 提升最大，ACAM consistency 进一步补充类别稳定性和 attention 覆盖。

### Important Takeaways

- Scribble weak supervision 很需要全局形状信息，仅靠局部 CNN 容易在未标注区域产生噪声。
- 双分支互补是一个高价值设计：一个分支提供局部细节，另一个分支提供全局形状。
- Attention / activation consistency 可以作为比直接 hard pseudo-label 更温和的弱监督扩展方式。
- 对你的 SAM3 项目来说，SAM3 detector 本身已经有强全局语义，tracker 则提供 volume continuity；二者可以类比 ScribFormer 的两个互补分支。

### Limitations

- 它的 pseudo-label 是 hard argmax，和 DMSPS 的 soft pseudo-label 相比可能更容易过度自信。
- 结构比较复杂，直接迁移到 SAM3 不现实。
- 主要仍是 cardiac 数据集，和 Gotkowski 提到的 cardiac benchmark over-specialization 有一定重叠风险。
- 它没有显式利用 3D temporal / slice continuity。

### Relevance to My Project

这篇对你当前 tracker 项目的启发是：不要把 detector 和 tracker 只看成 inference merge 的两个结果，而可以把它们解释为两个互补信息源：

- detector：强 semantic / global representation，负责单 slice 语义识别。
- tracker：强 temporal / anatomical continuity，负责跨 slice 传播一致性。

如果训练上引入 detector-tracker consistency，可以借鉴 ScribFormer 的思路：两个分支各自输出，动态混合或互相监督，然后再用 attention / feature-level consistency 约束未标注区域。

### Possible Ideas to Borrow

#### Idea A: Detector-Tracker Complementary Branches

把 detector 和 tracker 写成两个互补分支：

```text
detector = semantic branch
tracker = continuity branch
```

论文表达可以是：

> The detector branch captures strong slice-wise semantic cues, while the tracker branch provides anatomical continuity across adjacent slices.

#### Idea B: Attention / Feature Consistency Instead of Only Mask Loss

如果 mask-level tracker merge 不提升，可以尝试 feature/attention-level consistency：

```text
feature(detector mask region) should be consistent with feature(tracker propagated region)
```

这会比单纯改 merge rule 更像训练方法。

#### Idea C: Dynamic Detector-Tracker Pseudo Target

类似 ScribFormer 动态混合 CNN / Transformer：

```text
p_mix = alpha * p_detector + (1 - alpha) * p_tracker
```

但建议采用 DMSPS 的 soft target，而不是 ScribFormer 的 hard argmax target。

---

## 6. MedCL: Learning Consistent Anatomy Distribution for Scribble-supervised Medical Image Segmentation

- Authors: Ke Zhang, Vishal M. Patel
- Venue / Year: MIDL 2025
- Paper file: `paper/Zhang和Patel - 2025 - MedCL Learning Consistent Anatomy Distribution for Scribble-supervised Medical Image Segmentation.pdf`
- Task: scribble-supervised medical image segmentation
- Domain: cardiac structure, abdominal multi-organ, myocardial pathology
- Supervision type: scribble + image-level category labels
- Main contribution type: anatomy distribution learning via feature mixing and clustering

### Core Idea

MedCL 的核心观点是：医学分割标签不是任意分布的，它们具有 anatomy distribution prior。比如不同心脏结构之间的位置关系、形状分布、类别组合关系是有规律的。作者希望在 scribble 很少的情况下，通过 feature mixing 和 prototype clustering 学到这种解剖分布，而不是只依赖稀疏 scribble pixels。

这篇特别值得注意的是：它同时做了 `SAM-based` 和 `UNet-based` 实现，因此和你当前 SAM3 weak supervision 项目非常接近。

### Motivation

作者认为现有 scribble 方法有几个问题：

- 需要较多 scribble annotation 才能稳定。
- 对规则器官比较有效，但对 long-tailed / irregular pathology 不一定有效。
- pseudo-label 方法容易受错误 mask 噪声影响。
- 普通 mixup 常常破坏医学结构形状，但如果设计得好，feature mixing 可以用来学习 anatomy distribution。

因此 MedCL 不是直接扩展 pseudo labels，而是学习“解剖 prototype 的分布关系”。

### Method Overview

MedCL 包含两个步骤：

1. `feature mixing`: 在 intra-image 和 inter-image 层面混合图像、box、text prompt / prompt embedding，生成更丰富的 image-prompt pairs。
2. `feature clustering`: 将预测映射到 anatomy prototypes，并用 cluster regularization 学习 compact、discriminative、consistent 的解剖类别分布。

Intra-mix：

- 把图像和小角度旋转版本混合。
- 用 bounding box mask 保留 ROI 内结构，主要在 ROI 外做 mix，避免破坏目标结构。
- 采样单类别 prompt 和多类别 prompt 组合，让模型学习单结构和组合结构。

Inter-mix：

- 混合两张图像。
- 合并 bounding boxes。
- 插值 text prompt embeddings。
- 用 mix consistency loss 约束混合图像预测和混合预测一致。

### Anatomy Prototype Clustering

MedCL 维护一组 anatomy prototypes。它假设好的 prototype cluster 应该满足：

- compactness：同一类别 prototype 聚集。
- discriminability：不同类别 prototype 分开。
- anatomy consistency：类别组合的分布在局部到全局尺度上保持一致。

Cluster loss 大致鼓励：

```text
same-class prototypes close
different-class prototypes separated
multi-class anatomy distribution consistent
```

Anatomy consistency 用 prompt 组合构造。例如单个 LV / MYO / RV 的预测之和，应该和组合 prompt 对应的结构预测一致。这对 SAM 类模型尤其有启发，因为 SAM / SAM3 本身有 prompt-conditioned 输出。

### Loss Design

总目标由 unsupervised 和 weakly supervised 两部分组成：

```text
L = L_mix + L_cluster + L_ac + L_scribble + L_category
```

其中：

- `L_mix`: mixed image / mixed prediction consistency。
- `L_cluster`: anatomy prototype compactness + discriminability。
- `L_ac`: anatomy consistency，约束单类别和多类别组合的预测 / prototype 分布一致。
- `L_scribble`: scribble pixels 上的 CE + Dice。
- `L_category`: image-level category label loss，抑制不存在类别。

这里的关键创新不是 partial loss，而是 anatomy prototype / distribution consistency。

### Datasets

| Dataset | Task | Setting |
|---|---|---|
| MSCMRseg | regular cardiac structures | 45 LGE MRI, 25 / 5 / 15 split, 5 training scribbles |
| MyoPS | myocardial pathology | scar / edema irregular pathology, 20 / 5 / 20 split |
| BTCV | abdominal multi-organ CT | 13 organs, nnU-Net based implementation |

### Results

MSCMRseg 5 scribbles：

| Method | Backbone | Avg Dice |
|---|---|---:|
| PCE | UNet | 0.157 |
| WSL4 | UNet | 0.687 |
| ModelMix | UNet | 0.784 |
| MedCL | SAM | 0.805 |
| MedCL | UNet | 0.832 |
| FullSup-nnUNet | nnUNet | 0.799 |

MyoPS irregular pathology 5 scribbles：

| Method | Backbone | Avg Dice |
|---|---|---:|
| PCE | UNet | 0.182 |
| ModelMix | UNet | 0.440 |
| MedCL | SAM | 0.486 |
| MedCL | UNet | 0.497 |
| FullSup-nnUNet | nnUNet | 0.529 |

BTCV multi-organ：

- `MedCL-nnUNet` from scratch achieves `85.21` average Dice。
- 作者指出它甚至超过了使用大规模 CT 预训练的 VoCo `83.85`。

### Ablation

MSCMRseg validation 上关键消融：

| Setting | SAM Avg Dice | UNet Avg Dice |
|---|---:|---:|
| baseline | 0.350 | 0.222 |
| Mix / `L_mix` only | 0.521 | 0.563 |
| `L_cluster` only | 0.558 | 0.638 |
| Mix + `L_cluster` | 0.707 | 0.744 |
| Full MedCL | 0.833 | 0.828 |

结论很清楚：feature mixing、cluster loss、anatomy consistency 都有贡献，最终 anatomy consistency 把性能推到最高。

### Important Takeaways

- Scribble supervision 不一定只能围绕 pseudo-label 展开，也可以围绕 anatomy prior / class distribution 展开。
- SAM 类模型很适合做 prompt combination consistency，因为不同 text prompt / class prompt 可以自然生成单类和组合类预测。
- Prototype cluster 可以作为更 fancy 的 scribble-specific 模块，而且比简单 valid-region loss 更容易写成方法贡献。
- 对 irregular pathology，解剖分布和类别关系可能比单纯 mask pseudo-label 更稳。

### Limitations

- 论文是 MIDL 2025 submission，可能还不是最终正式发表版本。
- 方法较复杂，需要 prompt combination、feature mixing、prototype clustering 和 Sinkhorn-style online mapping。
- MSCMRseg 中出现 MedCL-UNet 高于 FullSup-nnUNet 的结果，需要谨慎引用和解释，避免被审稿人质疑 split / supervision setting 不一致。
- 它使用 SAM / MedSAM，不是 SAM3 detector-tracker，迁移时需要重新设计接入点。

### Relevance to My Project

MedCL 对你的项目非常有价值，因为它给了一个“弱监督 specific”的方向：不是再堆一个 tracker，而是让 tracker / detector 的输出符合 anatomy distribution。

你可以把当前项目升级成：

```text
Scribble-supervised SAM3 adaptation with anatomical consistency and temporal memory.
```

其中：

- detector 提供 prompt-conditioned semantic prediction。
- tracker 提供 intra-volume continuity。
- anatomy prototypes 提供 inter-sample / class-distribution prior。

这样 tracker 不再是孤立模块，而是和 scribble weak supervision 目标绑定起来。

### Possible Ideas to Borrow

#### Idea A: SAM3 Anatomy Prototype Bank

为 LV / MYO / RV 建立 prototype：

```text
proto_c = EMA(mean(feature inside confident mask or scribble pixels))
```

训练时约束 prediction region 的 feature 接近对应 prototype，不同类 prototype 分离。

#### Idea B: Prompt Combination Consistency

如果 SAM3 支持逐类 prompt，可以比较：

```text
prediction("left ventricle") + prediction("myocardium")
≈ prediction("left ventricle and myocardium")
```

对 ACDC 来说，可以构造 LV、MYO、RV 的单类和组合类 consistency。

#### Idea C: Detector-Tracker Anatomy Agreement

把 detector mask 和 tracker mask 都映射到 anatomy prototype space：

```text
Q(mask) = score + prototype_similarity + temporal_consistency
```

用于 confidence-aware merge。这样 merge 不只是规则阈值，而有 anatomical confidence。

#### Idea D: Weak-supervised Stage-2 Expansion with Anatomy Prior

在 pseudo seed bank 生成时，不只看 score：

```text
reliable_seed = high_score AND high_prototype_similarity AND detector_tracker_agreement
```

这比单纯 `score > threshold` 更像一个完整方法。

---

## 7. ModelMix: A New Model-Mixup Strategy to Minimize Vicinal Risk Across Tasks for Few-Scribble Based Cardiac Segmentation

- Authors: Ke Zhang, Vishal M. Patel
- Venue / Year: MICCAI 2024
- Paper file: `paper/Zhang和Patel - 2024 - ModelMix A New Model-Mixup Strategy to Minimize Vicinal Risk Across Tasks for Few-Scribble Based Ca.pdf`
- Task: few-scribble supervised cardiac segmentation
- Domain: ACDC, MSCMRseg, MyoPS
- Supervision type: 5 scribble-annotated volumes + unlabeled images
- Main contribution type: model-parameter mixup across correlated tasks

### Core Idea

ModelMix 的核心不是混合图像或标签，而是混合不同任务模型的 encoder 参数。作者认为相关医学分割任务之间存在共享知识，例如 cardiac structure segmentation 和 myocardial pathology segmentation 之间有解剖相关性。直接训练一个共享 encoder 容易在极少 scribble 下欠稳，而完全独立训练又无法利用互补任务。

因此他们先为每个任务训练独立模型，然后随机选择 encoder 中的卷积层，对不同任务模型的 convolution kernels / bias 做线性插值，构造 virtual model。接着用 supervised 和 unsupervised vicinal regularization 约束 virtual model 与 individual model 的输出一致。

### Motivation

Few-scribble setting 比普通 scribble 更难，因为每个任务只有少量 scribble-annotated volumes，其余训练图像未标注。已有方法容易：

- 只利用 scribble pixels，监督太稀疏。
- 对少量标注样本过拟合。
- 直接 mix 不同任务图像会产生不真实图像，也无法处理类别不一致。
- 共享 encoder 对多个任务做 generalization 时，在小数据下容易学不好 task-specific 信息。

ModelMix 的折中是：保留每个任务的独立 encoder，同时通过 virtual model 建立任务之间的连续空间。

### Method Overview

ModelMix 包含三步：

1. 分别训练每个任务的模型，每个任务有独立 encoder 和 decoder。
2. 在两个相关任务的 encoder 中随机选择卷积层，对参数做线性插值。
3. 对 virtual model 施加 vicinal regularization，使它在不同任务上和原模型保持一致。

参数混合直觉：

```text
mixed_kernel = lambda * kernel_i + (1 - lambda) * kernel_j
mixed_bias = lambda * bias_i + (1 - lambda) * bias_j
```

这里 `lambda` 从 beta distribution 采样。

### Loss Design

训练目标由三部分组成：

```text
L = L_inv + L_vicinal-reg + L_vicinal-sup
```

其中：

- `L_inv`: image-level mix invariant loss，对同一任务内部的 mixed image 做一致性正则。
- `L_vicinal-reg`: unsupervised vicinal regularization，让 virtual model 输出和 individual model 输出相似。
- `L_vicinal-sup`: supervised vicinal loss，在 scribble annotated pixels 上对 virtual model 和 individual model 计算 CE + Dice。

`L_vicinal-reg` 用 cosine similarity 约束输出。`L_vicinal-sup` 是 scribble pixels 上的 CE + Dice，因此仍然不需要 full mask。

### Datasets

| Dataset | Task | Setting |
|---|---|---|
| ACDC | cardiac structure segmentation | RV / LV / MYO, 5 scribble volumes |
| MSCMRseg | LGE cardiac structure segmentation | LV / MYO / RV, 5 scribble volumes |
| MyoPS | myocardial pathology segmentation | scar / edema, 5 scribble volumes |

作者使用 5 个 randomly selected scribble-annotated volumes，加上其它 unlabeled training images。

### Results

MSCMRseg 5 scribbles：

| Method | Avg Dice |
|---|---:|
| PCE | 0.304 |
| CycleMix | 0.315 |
| ShapePU | 0.461 |
| WSL4 | 0.687 |
| ModelMix w/ MyoPS | 0.784 |
| FullSup-UNet | 0.651 |
| FullSup-nnUNet | 0.799 |

MyoPS 5 scribbles：

| Method | Extra task model | Avg Dice |
|---|---|---:|
| PCE | none | 0.182 |
| CVIR | ratio prior | 0.186 |
| nnPU | ratio prior | 0.263 |
| ModelMix | MSCMRseg | 0.487 |
| ModelMix | ACDC | 0.532 |
| ModelMix | ACDC + MSCMRseg | 0.509 |
| FullSup-UNet | dense | 0.434 |
| FullSup-nnUNet | dense | 0.529 |

值得注意的是，ModelMix 在 MyoPS 上接近甚至超过普通 FullSup-UNet，并接近 FullSup-nnUNet。

### Ablation

在 MyoPS + MSCMRseg / ACDC 组合上，关键观察包括：

- `L_inv` 明显提升基础模型，说明同任务内 mix consistency 有用。
- 加入 `L_vicinal-sup` 后大幅提升，说明跨任务 virtual model 在 scribble pixels 上学习很有效。
- 加入 `L_vicinal-reg` 后进一步提升，说明 unlabeled pixels 的一致性约束提供额外信息。
- shared encoder baseline 明显差于 ModelMix，说明少量 scribble 下“一把 encoder 同时学多个任务”不如“独立模型 + virtual interpolation”稳。

作者还观察到：MyoPS 与 MSCMRseg 的互补性略强于 MyoPS 与 ACDC，因为二者都包含 enhanced pathological imaging 信息。

### Important Takeaways

- Few-scribble 场景下，跨任务信息非常有价值，尤其是医学任务之间存在解剖相关性。
- 不一定要直接混合图像或标签；混合模型参数也可以构造任务间的 vicinal space。
- 用 virtual model 做 consistency，可以把 unlabeled images 纳入训练。
- 共享 encoder 不一定是最佳选择，特别是在小标注场景下，保留 task-specific model 再做 regularization 可能更稳。

### Limitations

- 需要多个相关任务 / 数据集。如果只有 ACDC 单数据集，ModelMix 的优势不容易发挥。
- 方法主要针对 CNN encoder 的卷积参数插值，直接套到 SAM3 transformer / tracker memory 不简单。
- 它依赖任务相关性；如果两个任务差异太大，mixing 可能无益。
- 只在 few-scribble cardiac/pathology setting 中验证，泛化到 SAM3 detector-tracker 需要重新设计。

### Relevance to My Project

ModelMix 给你的项目一个很有趣的方向：不要只把 detector 和 tracker 当成两个 inference source，也可以把它们当成两个“相关任务模型”或“相关路径”，在训练中构造中间状态并做 regularization。

可能的对应关系：

- detector model：slice-wise semantic segmentation task。
- tracker model：volume propagation / continuity task。
- virtual model：detector-tracker adapter / mixed LoRA / mixed head。

如果你之后有 MSCMR / MyoPS 数据，也可以把 ACDC scribble SAM3 和其它 cardiac dataset 的 scribble SAM3 做 cross-task regularization。

### Possible Ideas to Borrow

#### Idea A: Detector-Tracker Vicinal Consistency

构造 detector 和 tracker 输出之间的中间 pseudo target：

```text
p_virtual = lambda * p_detector + (1 - lambda) * p_tracker
```

然后要求 model 对相邻 slice / mixed feature 的输出与 `p_virtual` 保持一致。

#### Idea B: LoRA / Adapter Mix

如果 detector LoRA 和 tracker LoRA 分别训练，可以尝试在 adapter 参数上做 interpolation：

```text
adapter_mix = lambda * adapter_detector + (1 - lambda) * adapter_tracker
```

再用 consistency loss 约束 mixed adapter 的输出。这是 ModelMix 在 SAM3 LoRA 场景下的自然变体。

#### Idea C: Cross-dataset Cardiac Regularization

如果后续加入 MSCMR 或 MyoPS，可以训练 task-specific SAM3 adapters，然后通过 ModelMix 思路让它们互相正则：

```text
ACDC adapter <-> MSCMR adapter <-> MyoPS adapter
```

这会比只在 ACDC 上加 tracker 更能回应泛化性质疑。

---

## 8. ZScribbleSeg: A Comprehensive Segmentation Framework with Modeling of Efficient Annotation and Maximization of Scribble Supervision

- Authors: Ke Zhang, Bomin Wang, Hangqi Zhou, Xiahai Zhuang
- Venue / Year: Medical Image Analysis, 2026
- Paper file: `paper/Zhang 等 - 2026 - ZScribbleSeg A comprehensive segmentation framework with modeling of efficient annotation and maxim.pdf`
- Task: scribble-supervised medical image segmentation
- Domain: ACDC, MSCMRseg, BTCV, MyoPS, Decathlon-BrainTumor, Decathlon-Prostate
- Supervision type: scribble
- Main contribution type: efficient scribble modeling + supervision augmentation + spatial / shape prior regularization

### Core Idea

ZScribbleSeg 是一个非常系统的 scribble-supervised segmentation 框架。作者认为现有方法主要在 annotated pixels 上算 loss，或者把 scribble 向邻近区域传播成 pseudo label，但这些做法容易出现监督不足、under-segmentation 和形状不真实。

论文分两条线解决：

- efficient scribble modeling：研究什么样的 scribble 更有效，并用 mixup + occlusion 模拟更高效的 scribble supervision。
- prior regularization：估计类别比例 `pi prior`，再结合 spatial prior / shape constraints 修正未标注区域预测。

### Motivation

作者提出两个关于高效 scribble 的原则：

- annotated pixel proportion 更大时，训练梯度更充分，模型更接近 full annotation 上界。
- scribble 分布越随机、越广，越能保留全局结构信息，而不是只在局部区域提供监督。

简单加粗 scribble 只能增加局部像素数量，不能扩大全局覆盖。因此作者用 supervision augmentation 来模拟“更大比例 + 更随机分布”的 scribble。

### Method Overview

ZScribbleSeg 包含三部分：

1. `efficient scribble modeling`: 用 mixup 最大化 supervision，用 occlusion 模拟随机分布。
2. `prior computation`: 用 EM 估计 label class proportion，也就是 `pi prior`。
3. `prior regularization`: 用 spatial prior loss 和 shape regularization 修正 under-segmentation / fragmented predictions。

最终 ZScribbleNet 的训练包含：

```text
L_pce + L_global + L_spatial + L_shape
```

其中：

- `L_pce`: scribble pixels 上的 partial CE。
- `L_global`: mixup / occlusion 后的 global consistency。
- `L_spatial`: 用 spatial energy + pi prior 选择 class-specific pixel subsets 做空间先验约束。
- `L_shape`: 对规则结构促进连通性，减少碎片化预测。

### Supervision Augmentation

作者用 PuzzleMix-style mixup 生成 mixed image-scribble pairs，并用 saliency 最大化来寻找更有监督价值的混合区域。

然后引入 occlusion：

- 随机遮挡 mixed image 的一块区域。
- 遮挡区域 label 设置为 background。
- 这样可以增加背景 scribble 的比例，缓解 foreground/background labeled pixel imbalance。

为了避免 mixup 破坏形状，作者增加 global consistency loss，要求增强前后预测在整体上保持一致。

### Prior Regularization

ZScribbleSeg 的另一个核心是 `pi prior` 和 spatial prior。

直觉是：scribble 标注比例通常不等于真实类别比例。例如某类结构可能只画了一小段，但真实 mask 占比更大。如果直接相信 scribble distribution，模型容易 under-segment。

作者使用 EM algorithm 估计每个类别的 mixture ratio，也就是 `pi prior`。再结合 spatial energy map，对未标注像素进行 class-specific ranking 和 top-pi selection，用来构造 spatial prior loss。

对于规则结构，作者还加入 shape regularization 促进结构连通性；对于 MyoPS / BrainTumor 这类不规则病灶，则去掉 `L_shape`，避免强行施加不合适的形状约束。

### Datasets

| Dataset | Task |
|---|---|
| ACDC | cardiac LV / MYO / RV segmentation |
| MSCMRseg | LGE cardiac structure segmentation |
| BTCV | abdominal organ segmentation |
| MyoPS | myocardial pathology scar / edema segmentation |
| Decathlon-BrainTumor | tumor / edema segmentation |
| Decathlon-Prostate | 3D prostate central gland / peripheral zone segmentation |

这是目前笔记里覆盖面最广的一篇，和 Gotkowski 对“不能只看 cardiac benchmark”的提醒很一致。

### Results

代表性结果：

| Dataset | PCE Avg Dice | ZScribbleSeg Avg Dice | FullSup-nnUNet Avg Dice |
|---|---:|---:|---:|
| ACDC | 0.770 | 0.862 | 0.874 |
| MSCMRseg | 0.385 | 0.870 | 0.907 |
| BTCV subset | 0.655 | 0.856 | 0.893 |
| MyoPS | 0.281 | 0.636 | 0.630 |
| BrainTumor | 0.626 | 0.763 | 0.823 |
| Prostate 3D | 0.659 | 0.706 | 0.726 |

几个重要观察：

- ACDC 上 ZScribbleSeg 不是最高，HELPNet / TIP25 更强，但它接近 FullSup-nnUNet。
- MSCMRseg 上 ZScribbleSeg 达到 0.870，显著强于 PCE。
- MyoPS 上 ZScribbleSeg 0.636，略高于 FullSup-nnUNet 0.630，说明 prior regularization 对不规则病灶很有帮助。
- 3D Prostate 上 ZScribbleSeg 0.706，接近 FullSup-nnUNet 0.726。

### Ablation

论文消融显示：

- efficient scribble modeling 带来明显提升。
- 仅使用 `pi prior` 已能提升，因为它修正了类别比例偏差。
- 加入 spatial energy 后，Dice 从约 0.881 提升到 0.894，HD 从 44.25 mm 降到 27.08 mm。
- 最终 ZScribbleSeg 达到约 0.899 Dice 和 8.70 mm HD，说明 efficient scribble + prior regularization 是互补的。

运行成本方面，`L_spatial` 只增加训练开销，不影响推理时间；例如 ACDC 训练从 0.1423 s/iter 增到 0.1689 s/iter，推理时间不变。

### Important Takeaways

- Scribble 标注的“形态”很重要，不能只把 scribble 当作一组 valid pixels。
- 随机、广覆盖的 scribble 更能表达全局结构；这对医学结构尤其关键。
- Class proportion prior 可以直接针对 under-segmentation，这是 scribble weak supervision 常见问题。
- Spatial prior / shape regularization 是比简单 pseudo-label 更可解释的弱监督机制。
- 不同任务应使用不同先验：规则器官可以用连通性/形状约束，不规则病灶应谨慎使用。

### Limitations

- 方法组件较多，包括 mixup、occlusion、EM prior estimation、spatial energy、shape regularization，实现复杂度高。
- 部分结果与其它论文结果来自不同训练设置，跨论文比较需要谨慎。
- 先验设计和任务类型相关，规则器官与不规则病灶不能共用同一套 shape constraint。
- 不是 SAM / SAM3 框架，需要重新决定 prior loss 接在 mask logits、query features 还是 final mask 上。

### Relevance to My Project

ZScribbleSeg 对你当前项目非常有用，因为它直接说明：scribble-specific innovation 可以不只是 partial loss，也可以围绕“标注效率建模”和“未标注区域先验修正”展开。

对 SAM3 scribble tracker 来说，最相关的是：

- `pi prior`: 用 detector / tracker 的 volume-level prediction 估计每类结构比例，防止 tracker 传播导致过小或过大。
- `spatial prior`: 用图像边界、距离、feature similarity 或 tracker consistency 构造 spatial energy，帮助选择可靠未标注区域。
- `shape / connectivity`: 对 LV / MYO / RV 这类规则结构，约束跨 slice 连续性和 mask 连通性。
- `efficient scribble`: 可以在论文中分析 scribble 覆盖率、分布范围、随机性，并说明 SAM3 如何利用稀疏但全局有效的 scribble。

### Possible Ideas to Borrow

#### Idea A: Volume-level Class Proportion Prior

对每个 volume / class 估计一个结构比例先验：

```text
pi_c = estimated volume ratio of class c
```

训练或 merge 时惩罚明显偏离比例先验的预测，尤其避免 tracker propagated masks 过度 shrink。

#### Idea B: Tracker-aware Spatial Prior

把 tracker propagation 当作一种 spatial energy：

```text
spatial_energy = image_boundary_score + tracker_consistency + detector_confidence
```

然后在未标注区域中选择 top-pi 的 pixels 作为 soft reliable region。

#### Idea C: Connectivity / Shape Regularization for Cardiac Structures

对 LV / MYO / RV，可以加入轻量后处理或训练正则：

```text
penalize fragmented components
encourage smooth cross-slice area change
encourage ring-like MYO topology
```

这比单纯让 tracker 传播 mask 更贴近医学结构先验。

#### Idea D: Scribble Efficiency Analysis

在论文实验中加入：

- scribble pixel ratio
- scribble slice coverage
- per-class scribble imbalance
- pseudo seed spatial coverage
- detector/tracker contribution by low-coverage slices

这样可以把你的工作从“改 loss”提升到“分析并最大化 scribble supervision efficiency”。

---

## 9. EFFDNet: A Scribble-Supervised Medical Image Segmentation Method with Enhanced Foreground Feature Discrimination

- Authors: Jinhua Liu, Shu Yun Tan, Xulei Yang, Yanwu Xu, Si Yong Yeo
- Venue / Year: MICCAI 2025, LNCS 15975, published 2026
- Paper file: `paper/Liu 等 - 2026 - EFFDNet A Scribble-Supervised Medical Image Segmentation Method with Enhanced Foreground Feature Di.pdf`
- Task: scribble-supervised medical image segmentation
- Domain: ACDC cardiac MRI / NCI-ISBI prostate MRI
- Supervision type: scribble
- Main contribution type: feature discrimination loss + foreground augmentation
- Code: `https://github.com/Aurora-003-web/EFFDNet`

### Core Idea

EFFDNet 的核心观点是：scribble 不只是 sparse pixel label，它还隐含了 foreground-background semantics。也就是说，有 foreground-class scribble 的局部区域更可能包含目标解剖结构；没有 foreground scribble 的区域更偏 background / non-target。

基于这个观察，作者提出两个模块：

- `FBSL`: Foreground-Background Separation Loss，在特征空间拉近同类 foreground/background 区域，推远 foreground 与 background。
- `FADC`: Foreground Augmentation with Diverse Context，把一个样本的 foreground scribble-box 区域裁剪到另一个样本的背景中，增强模型对前景结构的敏感性。

### Motivation

作者认为现有 scribble 方法通常有两类：

- 把 scribble 当作 seed region，然后做 label propagation。
- 把 scribble 当作 sparse annotation，只在标注像素上做 partial CE。

这两类方法都没有充分利用医生 scribble 中隐含的“前景在哪里、背景在哪里”的结构先验，因此容易出现 foreground/background 混淆、类别间混淆和形态不完整。

### Basic Framework

EFFDNet 使用 Mean Teacher 风格的 student-teacher 框架：

- Student network 接收 scribble label 和 teacher 生成的 pseudo-label 监督。
- Teacher network 不通过梯度更新，而是 student 权重的 EMA。
- Teacher 生成 hard pseudo-label，用于补充未标注区域监督。

基础 loss 包含：

```text
L_S: scribble-supervised partial CE，只在 scribble pixels 上计算
L_P: pseudo-label supervised CE，在 dense pseudo-label 上计算
```

### FBSL: Foreground-Background Separation Loss

FBSL 的实现逻辑：

1. 从 segmentation head 前一层提取 feature map。
2. 把 feature map 划分成 `K x K` 个局部区域，并对每个区域做 average pooling。
3. 根据该区域内是否包含 foreground scribble，将区域标为 foreground 或 background。
4. 对聚合后的区域特征做 contrastive-style loss：
   - 同为 foreground 的区域互为 positive。
   - 同为 background 的区域互为 positive。
   - foreground 与 background 互为 negative。

最终目标是让模型在 feature space 中更清楚地区分目标解剖结构和背景区域。

整体 loss 形式可以概括为：

```text
L1 = L_S^1 + lambda * (L_P^1 + delta * L_FBSL)
```

### FADC: Foreground Augmentation with Diverse Context

FADC 的核心是用 scribble 的 bounding box 近似定位 foreground：

1. 根据 foreground scribble 得到最小 scribble bounding box。
2. 从样本 `Xp` 中 crop foreground scribble-box 区域。
3. 把这个 foreground crop 替换到另一个 batch 样本 `Xo` 的对应区域。
4. 同步替换 scribble label 和 pseudo-label。

这样做得到“同一个 foreground 出现在不同 background context 中”的训练样本，目的是：

- 增强前景敏感性。
- 减少模型对背景上下文的过拟合。
- 提升 foreground morphology 的识别能力。

FADC 对应的增强样本 loss：

```text
L2 = L_S^2 + lambda * L_P^2
L = L1 + L2
```

### Datasets and Results

作者在 ACDC 和 NCI-ISBI 上做 5-fold cross-validation，2D slice-wise 训练，测试时重组成 3D volume 评估 Dice。

| Dataset | Classes | EFFDNet Result |
|---|---|---:|
| ACDC | RV / Myo / LV | 86.54 / 85.67 / 92.15 DSC |
| NCI-ISBI | PZ / CG | 73.00 / 86.37 DSC |

ACDC 上 EFFDNet 接近 full supervision，但仍低于 FullSup：

| Method | RV | Myo | LV |
|---|---:|---:|---:|
| ScribFormer | 86.24 | 84.01 | 91.07 |
| EFFDNet | 86.54 | 85.67 | 92.15 |
| FullSup | 89.49 | 89.07 | 93.95 |

### Ablation

论文消融显示两个模块都有效：

| Method | ACDC RV | ACDC Myo | ACDC LV | NCI PZ | NCI CG |
|---|---:|---:|---:|---:|---:|
| Baseline | 83.13 | 80.16 | 87.71 | 65.91 | 65.92 |
| + FBSL | 85.90 | 84.60 | 92.17 | 72.39 | 85.90 |
| + FADC | 85.88 | 82.55 | 89.49 | 71.15 | 82.18 |
| Full EFFDNet | 86.54 | 85.67 | 92.15 | 73.00 | 86.37 |

重要观察：

- FBSL 对 Myo / LV / CG 提升很明显，说明 foreground-background feature separation 对结构边界和类别区分有帮助。
- FADC 单独也有明显提升，尤其能缓解背景过分割和类别混淆。
- 两者结合最好，说明 feature discrimination 和 foreground augmentation 是互补的。

### Important Takeaways

- Scribble-specific innovation 不一定只写在 loss 的 valid region 上，也可以挖掘 scribble 隐含的 foreground/background semantics。
- 对医学结构来说，foreground-background separation 比单纯 partial CE 更能解释为什么模型学到了结构。
- 用 scribble bounding box 做 foreground crop 是一种很轻量的弱监督增强，不需要 full mask。
- 这篇对“我的方法不够 fancy”的问题很有启发：可以把医生 scribble 解释为 anatomical foreground cue，而不是普通 sparse label。

### Limitations

- FBSL 只区分 foreground vs background，没有显式区分不同 foreground class 之间的混淆。
- FADC 使用 scribble bounding box，可能会把部分 background 一起 crop 进去；但这也使它不依赖 full mask。
- 论文仍然依赖 teacher pseudo-label，pseudo-label 错误可能被放大。
- 方法是 2D slice-wise，对 3D / temporal consistency 没有直接建模。

### Relevance to My Project

EFFDNet 对当前 SAM3 scribble detector/tracker 项目很有借鉴价值：

- 可以在 SAM3 detector 或 tracker feature 上加入 foreground-background contrastive loss，而不是只在 mask logits 上做 partial CE。
- 可以用 scribble 的 bounding box 或高置信 detector pseudo mask 生成 foreground crop，做 foreground context augmentation。
- 对 tracker 来说，FBSL 可以改成 `tracker memory feature` 的 foreground/background separation，让 memory bank 更关注目标结构而不是背景。
- 对 merge 来说，可以把 foreground discrimination score 作为 detector/tracker 选择的辅助依据。

### Possible Ideas to Borrow

#### Idea A: SAM3 Feature-level FBSL

在 SAM3 的 image feature、DETR hidden states 或 tracker memory feature 上做局部区域聚合：

```text
region has foreground scribble -> foreground region
region has no foreground scribble -> background / unlabeled candidate
```

然后做 foreground-background contrastive loss，增强 SAM3 对 LV/MYO/RV 前景的判别能力。

#### Idea B: Scribble-box Foreground Augmentation

用 scribble bounding box 而不是 full mask 做 foreground crop：

```text
crop foreground box from volume A
paste into compatible background region of volume B
supervise only scribble pixels + pseudo labels
```

这可以作为比“只改 partial loss”更明确的 scribble-specific 模块。

#### Idea C: Tracker Memory Foreground Discrimination

让 tracker 写入 memory bank 的特征满足：

```text
same-class foreground memory closer
foreground memory farther from background memory
```

这样 tracker 不只是传播 mask，而是在 memory feature 层面学习“什么是目标结构”。

---

## 10. Soft Self-labeling and Potts Relaxations for Weakly-supervised Segmentation

- Authors: Zhongwen Zhang, Yuri Boykov
- Venue / Year: preprint / CV paper
- Paper file: `paper/Zhang和Boykov - Soft Self-labeling and Potts Relaxations for Weakly-supervised Segmentation.pdf`
- Task: weakly supervised semantic segmentation
- Domain: Pascal VOC 2012 / Cityscapes / ADE20K
- Supervision type: scribble / block-wise weak annotation
- Main contribution type: principled soft pseudo-label optimization + Potts spatial regularization

### Core Idea

这篇论文的核心观点是：弱监督分割中的 pseudo-label 不应该总是 hard label。对于未标注像素，模型本来就不确定，hard pseudo-label 会过早把不确定区域压成某一类，容易造成错误强化。

作者提出 soft self-labeling：把 pseudo-label 也作为一个可优化的 soft categorical distribution，并与网络预测一起优化一个明确的 weakly-supervised loss。这样 pseudo-label 可以表达不确定性，同时通过 Potts regularization 保持空间一致性。

### Motivation

传统 self-labeling 常见问题：

- hard pseudo-label 不能表达边界和未标注区域的不确定性。
- 很多 pseudo-label generation 是 heuristic，与训练 loss 没有严格对应关系。
- 多阶段 pseudo-label 训练可能缺少收敛保证，难复现。

作者希望建立一个更“数学上干净”的框架：pseudo-labeling sub-problem 和 network training sub-problem 都来自同一个 joint loss。

### Method Overview

作者从弱监督 Potts loss 出发：

```text
scribble NLL + entropy on unlabeled pixels + Potts pairwise regularization
```

然后引入 soft pseudo-label `y_i` 作为辅助变量：

```text
y_i in Delta^K
```

其中 scribble pixels 上要求：

```text
y_i = ground-truth scribble label
```

未标注像素上的 `y_i` 则由优化过程估计。最终 self-labeling loss 可概括为：

```text
L = - sum_{i in S} log sigma_i^{y_i}
    + eta * sum_{i not in S} H(sigma_i, y_i)
    + lambda * sum_{ij in N} P(y_i, y_j)
```

含义是：

- scribble pixels 上用真实 scribble label 监督预测。
- unlabeled pixels 上让 network prediction `sigma_i` 和 soft pseudo-label `y_i` 接近。
- 相邻像素的 soft pseudo-label 通过 Potts relaxation 保持空间一致。

### Soft Pseudo-label vs Hard Pseudo-label

这篇最值得借鉴的是 soft pseudo-label 的态度：

- 如果某个像素类别不确定，`y = (0.5, 0.5)` 不应该强迫网络输出同样不确定。
- 但也不应该直接把它硬分配成某一类。
- 更合理的是：用特殊的 cross-entropy 形式让网络保持 decisive，同时让 soft label 表达 uncertainty。

作者比较了三种 cross-entropy：

| Loss | 形式直觉 | 作用 |
|---|---|---|
| Standard CE | `H(y, sigma)` | 容易让 prediction 模仿 soft label 的不确定性 |
| Reverse CE | `H(sigma, y)` | 对 label uncertainty 更鲁棒 |
| Collision CE | 基于预测与 soft label 的 probability collision | 对不确定 soft label 更稳，论文中效果最好 |

### Potts Relaxation

Potts regularization 用来约束相邻像素的 label 一致性。作者比较了多种 relaxation：

| Relaxation | Intuition |
|---|---|
| Bilinear `P_BL` | 接近 graph cut，但优化时容易有 local minima |
| Quadratic `P_Q` | 接近 random walker，可产生 soft solution |
| Normalized quadratic `P_NQ` | 结合 bilinear 和 quadratic 的优点 |
| Collision CE `P_CCE` | log-based pairwise term |
| Collision divergence `P_CD` | normalized + log-based，实验表现最好 |
| Log-quadratic `P_LQ` | 缓解 vanishing gradient |

实验结论是：

- log-based Potts relaxation 通常更好，因为可以缓解 flat region / vanishing gradient。
- normalized relaxation 更好，因为可以减少 bilinear/quadratic 的局部极小问题。
- nearest-neighbor neighborhood 比 dense neighborhood 更适合 self-labeling，因为 dense neighborhood 容易退化成类别体积/比例约束，边界质量变差。

### Datasets and Results

主要实验在 Pascal VOC 2012 scribble supervision 上：

- 训练集：augmented 10,582 images
- 验证集：1,449 images
- metric：mIoU
- backbone：DeepLabv3+ with MobileNetV2 / ResNet101，另有 ViT-linear

Potts relaxation 对比中，full-length scribble 时：

| Relaxation | mIoU |
|---|---:|
| PBL | 67.24 |
| PNQ | 71.12 |
| PQ | 71.05 |
| PCCE | 67.41 |
| PCD | 71.22 |
| PLQ | 71.21 |

最终 SOTA 对比中：

| Method | Architecture | mIoU |
|---|---|---:|
| TEL | DeepLabV3+ | 77.1 |
| AGMM | ViT-linear | 78.7 |
| HCCE + PCD | DeepLabV3+ batch 16 | 78.1 |
| HCCE + PCD no pretrain | ViT-linear batch 16 | 80.94 |
| Full supervision ViT-linear | ViT-linear | 81.4 |

论文强调：在标准架构上，soft self-labeling 可以接近甚至在部分设置下超过 full supervision baseline。

### Important Takeaways

- 弱监督 pseudo-label 不一定要 hard；soft label 更适合表达未标注像素和边界的不确定性。
- pseudo-label 生成最好和训练 loss 有明确关系，而不是“先生成，再硬塞给网络”。
- 空间连续性可以通过 Potts / CRF 风格正则表达，而且不一定需要复杂新网络结构。
- 对不确定 label，standard CE 可能不是最优；reverse CE / collision CE 更值得尝试。

### Limitations

- 论文主要是自然图像语义分割，不是医学图像。
- Potts optimization 和 soft pseudo-label solver 实现复杂度高。
- 结果依赖较多超参数，例如 `eta`、`lambda`、neighborhood、pairwise potential。
- 没有直接讨论 3D volume 或 cardiac slice continuity。

### Relevance to My Project

这篇对你的 SAM3 scribble tracker 很关键，因为你现在遇到的问题正是：

- detector pseudo mask 不是 full label，不能直接当真值。
- tracker seed mask 有 confidence，高置信可以用，低置信不能硬用。
- merge 时 detector / tracker 都有不确定性，简单 detector-first 或 tracker-first 都太粗。

可以借鉴这篇，把 tracker 的 pseudo seed / propagated mask 设计成 soft supervision：

- 高置信区域：作为较硬的 pseudo-label。
- 边界和低置信区域：保留 soft probability。
- 相邻 slice / 相邻 pixel：用 Potts-style consistency 约束。

### Possible Ideas to Borrow

#### Idea A: Soft Tracker Seed

不要把 detector seed mask 二值化后直接喂 tracker，而是保留 probability / confidence：

```text
seed_soft_mask = detector_probability * confidence_weight
```

这样 tracker 初始化可以知道哪些区域可靠，哪些区域只是候选。

#### Idea B: Uncertainty-aware Tracker Loss

对 tracker propagated mask 使用 soft pseudo-label loss：

```text
high confidence pixels: strong CE / Dice
low confidence pixels: weak consistency / entropy regularization
boundary pixels: allow uncertainty
```

这比“只筛高置信 mask 作为 seed”更细粒度。

#### Idea C: Potts-style Cross-slice Consistency

把 2D Potts neighborhood 扩展到 cardiac volume：

```text
within-slice neighbor consistency
adjacent-slice mask probability consistency
edge-aware penalty near image boundary
```

对 tracker 来说，这可以成为更 scribble-specific 的连续性损失。

#### Idea D: Confidence-aware Merge as Soft Selection

当前 detector-first / tracker-first 都是硬规则。可以改成：

```text
final_prob = w_det * prob_det + w_trk * prob_trk
w_det, w_trk depend on confidence, temporal distance, entropy, shape prior
```

这与 soft self-labeling 的思想一致：不确定时不要过早做 hard decision。

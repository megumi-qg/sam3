# <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">数据集</span></span>

<span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">ACDC(100例), WORD, BraTS2020</span></span>

<span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">ACDC 的 scribble 来自于</span></span><span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F15388841%2Fitems%2FT5NHQY2F%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/T5NHQY2F">Valvano 等, 2021</a></span>)</span>.\ <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">WORD 的 scribble 来自于</span></span><span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F15388841%2Fitems%2FGT7BHZJC%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/GT7BHZJC">Luo 等, 2022</a></span>)</span>, 脚本生成\ <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">BraTS2020 的 scribble 来自于WSL4MIS 脚本生成</span></span>

<span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">Considering the large inter-slice spacing (mostly 10 mm), we used 2D networks for slice-by-slice segmentation, and stacked the results into a volume for 3D evaluation.</span></span>

<span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">To investigate the effectiveness of different methods with fewer scribbles, we cut the length of each scribble to 1/2, 1/4, 1/8 and 1/16 respectively, leading to different degrees of sparse annotations.</span></span>

***

# 模型

2D 分割任务（ACDC）采用**<span style="color: rgb(0, 0, 0);"><span style="">UNet</span></span>**作为主干网络，3D 分割任务 (<span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">WORD, BraTS2020</span></span>) 采用**<span style="color: rgb(0, 0, 0);"><span style="">3D UNet</span></span>**作为主干；\
同时论文验证了该方法可泛化到 UNet++、SwinUNet、Attention UNet、UNETR 等多种 CNN 类与 Transformer 类分割主干。

## **DB-Net**

整体为**<span style="color: rgb(0, 0, 0);"><span style="">共享编码器 + 两个独立解码器</span></span>**的架构：

1.  <span style="color: rgb(0, 0, 0);"><span style="">共享编码器：由多个级联的卷积 - 下采样单元组成，负责提取输入图像的多尺度特征；</span></span>
2.  <span style="color: rgb(0, 0, 0);"><span style="">两个独立解码器：主解码器与辅助解码器，均由级联的卷积 - 上采样单元构成，各自输出独立的分割预测；</span></span>
3.  <span style="color: rgb(0, 0, 0);"><span style="">扰动设计：在编码器瓶颈（bottleneck）的输出特征送入辅助解码器前，加入 dropout 层（比率为 0.5）引入特征扰动，使两个解码器产生不同的预测结果与决策边界，实现互相监督，同时缓解单分支预测的固有偏差。</span></span>

## 软伪标签生成方式

软伪标签通过**双分支预测动态加权混合**的方式生成，全程保留概率分布而非取硬标签：

1.  先获取两个解码器的输出：共享编码器提取特征后，两个解码器分别输出经 softmax 归一化的像素级类别概率图  $p_1$  和  $p_2$ ；

2.  引入混合系数  $\alpha$ ，该系数在每轮训练迭代中从均匀分布 U(0,1) 中随机采样，按照下式融合两个分支的概率图： $\hat{p}=\alpha \times p_1 + (1.0-\alpha) \times p_2$

3.  不对融合结果执行 argmax 操作，直接将融合后的概率分布  $\hat{p}$  作为软伪标签。这种动态混合的软标签既缓解了单分支预测的偏差，也保留了类别间的相似性与预测不确定性，对噪声的鲁棒性更强。

## 损失函数

论文的整体损失由 **部分交叉熵损失（pCE）** 和**软伪标签监督损失（SPS）** 加权构成，公式为：\
$\mathcal{L}=\mathcal{L}_{pCE} + \lambda \mathcal{L}_{SPS}$

其中 $\lambda$ 为权重平衡因子，三个数据集均设置为 8.0。

*   部分交叉熵损失  $\mathcal{L}_{pCE}$  ：仅在 scribble 标注的像素区域计算交叉熵，完全忽略未标注像素的梯度，对两个解码器的预测分别计算后取平均，负责保证标注区域的分割准确性。

*   软伪标签监督损失  $\mathcal{L}_{SPS}$ ：将动态混合得到的软伪标签作为监督信号，分别计算两个解码器的预测与软伪标签之间的交叉熵后取平均；计算时<u>软伪标签会被截断梯度（detach）</u>，避免梯度回传污染伪标签。该损失为大量未标注像素提供辅助监督，提升模型的泛化能力。

在第二阶段训练中，标注区域替换为经过不确定性筛选的扩展标注，部分交叉熵损失替换为在扩展标注区域计算的 $\mathcal{L}_{pCE}'$ ，整体损失变为 $\mathcal{L}'=\mathcal{L}_{pCE}' + \lambda \mathcal{L}_{SPS}$，软伪标签损失的计算逻辑保持不变。

**软伪标签截断梯度**：计算软伪标签监督损失时，把混合得到的软伪标签 $\hat{p}$ 当作固定不变的监督目标（类似真值），损失的梯度只会反向传播给两个解码器的预测 $p_1$ 、$p_2$ ，不会继续回传给 $\hat{p}$ 本身，也就不会更新生成 $\hat{p}$ 的那部分网络参数。

**不确定性筛选**

用预测熵衡量模型对每个像素的把握程度，只保留模型 “高置信” 的区域作为扩展标注，排除边界、模糊等易出错的不确定区域。

*   **获取平均预测** 用第一阶段训练好的最优模型对训练图像推理，将两个解码器的输出概率图  $p_1$ 、 $p_2$  取平均，得到平均概率图  $\bar{p}$ ；再对  $\bar{p}$  取 argmax，得到硬预测结果  $\bar{y}$ 。

*   **计算像素级不确定性（信息熵）** 用归一化的信息熵计算每个像素的不确定性，得到不确定性图 U：  $U(i)=-\frac{1}{log (C)} \sum_{c=0}^{C-1} \overline{p}_{i}^{c} log \left(\overline{p}_{i}^{c}\right)$  其中 C 是分割类别数。熵值越高，代表模型对该像素的类别判断越犹豫、不确定性越高；熵值越低，代表模型越确信。

*   **阈值筛选得到置信区域** 设定不确定性阈值  $\tau$  ，保留所有不确定性低于阈值的像素，生成二值置信掩码 M：  $M=U<\tau$  掩码内的区域就是模型认为 “预测可靠” 的区域。不同任务阈值不同：简单的心脏分割 ACDC 设为 0.1，腹部多器官 WORD 设为 0.3，更复杂的脑肿瘤 BraTS2020 设为 0.4，任务难度越高，阈值相应放宽。

*   **后处理生成最终扩展标注** 将硬预测  $\bar{y}$  和置信掩码 M 相乘，只保留高置信区域的类别；再通过形态学操作保留每个类别最大的连通分量，去除零散噪声点，最终得到扩展标注  $\tilde{s}$ ，用于第二阶段模型的训练。

***

# **<span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">结果</span></span>**

*   <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">ACDC 数据集（心脏）：平均DSC达 89.51%，显著高于仅基于scribble训练的UNet模型。</span></span>

*   <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">WORD 数据集（腹部器官）：平均DSC提升至 87.56%，优于一致性正则化（CR）与交叉伪监督（CPS）策略。</span></span>

*   <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">BraTS2020 数据集（脑肿瘤）：平均DSC提升至 76.53%，接近完全监督结果。</span></span>

*   <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">消融实验表明：</span></span>

    *   <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">双分支结构显著增强了特征提取与泛化能力；</span></span>
    *   <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">动态混合软伪标签优于CR和CPS策略；</span></span>
    *   <span style="color: rgb(34, 34, 34);"><span style="background-color: rgb(255, 255, 255);">第二阶段的扩展标注机制对最终性能贡献最大</span></span>

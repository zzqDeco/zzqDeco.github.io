---
title: "LaMa 论文精读：傅里叶卷积、大掩码修复与跨分辨率泛化"
description: "精读 LaMa 如何通过 Fast Fourier Convolution、高感受野感知损失和大掩码训练，在单阶段图像修复中恢复周期结构并泛化到训练未见的高分辨率"
pubDate: "2026-07-20T11:23:18+08:00"
updatedDate: "2026-07-20T11:23:18+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "Image Inpainting"
  - "Computer Vision"
  - "Fourier Convolution"
  - "Generative Models"
  - "Code Reading"
draft: false
---

图像修复（Image Inpainting）在演示中看似直观：用户在画面上涂抹掉某个路人或杂物，模型自动把空白区域补齐。然而在底层建模中，高质量补全需要同时满足三重约束——生成区域必须与孔洞外的可见上下文语义协调，必须在全图尺度上延续几何结构与透视线条，还必须在像素级接缝处消除色彩与纹理突变。当掩码孔洞变宽、周期性结构（如建筑窗格、栅栏、百叶窗）变长，或者测试分辨率显著超出训练裁剪尺寸时，依赖局部卷积核与短程纹理匹配的传统网络便会迅速退化为模糊色块或断裂伪影。

**Resolution-robust Large Mask Inpainting with Fourier Convolutions** 提出的 **LaMa**（Large Mask Inpainting），正是围绕“如何在全链路中建立大范围有效感受野”这一核心命题展开的系统设计。它放弃了繁琐的多阶段结构预测管线（如先预测边缘或语义图再渲染纹理），以单阶段全卷积网络同步改造了三个关键环节：在生成器残差块中引入快速傅里叶卷积（Fast Fourier Convolution, FFC），使部分通道通过频域变换在网络浅层即获得全图感受野；在训练监督中引入基于扩张卷积语义分割主干的高感受野感知损失（HRFPL）；在数据采样中主动构造更宽、更具侵略性的大掩码训练分布。这三项机制协同发力，共同成就了 LaMa 的修复效果。

本文以 arXiv [`2109.07161v2`](https://arxiv.org/abs/2109.07161v2) 与 WACV 2022 正式发表版本（含官方补充材料）为主线，结合官方代码仓库固定提交 [`advimman/lama@786f5936`](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7)（Apache-2.0 许可）进行静态源码走读。文中所有实验指标与图表均引自原论文及补充材料（CC BY 4.0）。

> **核心判断**：LaMa 的方法论价值在于证明了大区域修复必须将“全局上下文感知”同时落实到特征算子、感知监督与训练样本难度之中。频域卷积对周期结构延续和跨分辨率迁移展现出极佳的归纳偏置，但作为确定性的单阶段前馈模型，它同样存在强透视网格失真、多解语义平滑以及依赖历史软件栈等现实边界。

![LaMa teaser：大区域对象移除与结构恢复](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig01-teaser.webp)

*大掩码对象移除、高分辨率建筑修复和规则结构恢复示例。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 1, CC BY 4.0.*

## 1. 任务定义、大掩码困境与核心贡献

### 论文元数据与四大设计边界

| 项目 | 内容 |
| --- | --- |
| 题名 | Resolution-robust Large Mask Inpainting with Fourier Convolutions |
| 作者 | Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, Victor Lempitsky |
| 机构 | Samsung AI Center-Moscow 等 |
| 发表与版本 | arXiv `2109.07161v2`（2021-11-11）；WACV 2022, pp. 2149–2159（CC BY 4.0） |
| 官方代码 | [advimman/lama](https://github.com/advimman/lama)，本文固定源码基线为 commit [`786f5936`](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7)（Apache-2.0） |

概括而言，LaMa 通过结合具有全图感受野的 FFC 生成器、大感受野语义感知损失以及宽孔洞掩码训练策略，构建了一个单阶段全卷积图像修复框架。在深入技术细节前，有必要先明确该框架的四个基本界定：
1. **单阶段前馈架构**：输入被掩码遮挡的 RGB 图像与二值掩码，生成器单次前向传播即可直接输出完整 RGB 图像，无需显式预测中间边缘图或分割图（仓库中提供的多尺度特征精化 `refinement.py` 属于后期可选的测试时优化扩展，并未用于论文主表评测）。
2. **算子复用与系统集成**：核心算子 Fast Fourier Convolution（FFC）源自 Chi 等人在 NeurIPS 2020 提出的通用视觉主干模块；LaMa 的核心贡献在于将其引入图像修复生成器，并通过与感知损失及大掩码策略的联合消融，阐明了频域混合为何契合大孔洞与周期结构恢复。
3. **跨分辨率泛化的经验属性**：标准模型仅在从约 $512\times 512$ 图像裁剪出的 $256\times 256$ 图块上训练，却能在推理时直接作用于 $1536\times 1920$ 等更高分辨率输入且性能退化显著慢于普通卷积基线。这种稳健性源于全卷积结构与频域参数共享的经验组合，而非严格的数学尺度等变保证。
4. **专注于给定掩码的内容补全**：模型接收外部给定的缺失区域掩码完成像素合成，至于“如何分割出待删除目标”（如结合 SAM 交互分割）属于上游管线职责。

### 输入输出数学契约与大孔洞修复的五重难点

设完整 RGB 图像为 $\mathbf{x}\in[0,1]^{H\times W\times3}$，二值缺失掩码为 $\mathbf{m}\in\{0,1\}^{H\times W}$。遵循公开代码的约定，令 $m_{ij}=1$ 代表待合成的缺失区域（孔洞），$m_{ij}=0$ 代表已知背景区域。生成器的输入 $\mathbf{x}'$ 由掩码清零后的可见图像与单通道掩码在通道维拼接而成（共 4 个通道），经全卷积网络 $f_\theta$ 预测完整图像 $\hat{\mathbf x}$，最终在输出端将已知区域的原始像素无损贴回：

$$
\mathbf{x}'=\operatorname{stack}\left(\mathbf{x}\odot(1-\mathbf m),\mathbf m\right),\qquad
\hat{\mathbf x}=f_\theta(\mathbf{x}'),\qquad
\mathbf{x}_{out}=\mathbf{x}\odot(1-\mathbf m)+\hat{\mathbf x}\odot\mathbf m.
$$

在工程集成中，掩码的极性约定（`mask == 1` 表示待生成区域）、输入图像尺寸对下采样倍率的对齐填充（padding），以及仅在掩码内部采用预测像素的合成契约（compositing），是防止已知背景色彩漂移和边缘伪影的基础保障：

```text
image: float32, shape [B, 3, H, W], normalized to [0, 1]
mask:  float32, shape [B, 1, H, W], values in {0, 1} (1 = region to synthesize)
generator_input = concat(image * (1 - mask), mask, dim=1)  # shape [B, 4, H, W]
predicted_image = generator(generator_input)
output = image * (1 - mask) + predicted_image * mask
```

相比仅几像素宽的细小划痕修复，大面积掩码修复（Large Mask Inpainting）之所以困难重重，源于五个相互交织的物理挑战。首先，孔洞中心距离最近的已知边界像素往往相隔数十乃至上百像素，局部卷积需要堆叠极深层数才能间接传递边界信息，且长路径梯度衰减严重。其次，结构约束往往横跨全图，例如被抹除的行人遮挡了一段铁艺栏杆，唯有同时观测孔洞左右两侧的栏杆间距、倾角与相位才能实现无缝对接，仅靠单侧纹理外推必然造成栅格错位。再者，大孔洞内部具有高度的语义多解性（沙发背后的墙面既可以是白墙也可以是挂画），单一真值下的逐像素 $L_1/L_2$ 损失极易驱使网络输出模糊的平均色块。此外，建筑立面等场景还涉及低频透视收缩与高频纹理细节的耦合；而当测试分辨率提升至 $1536$ 甚至更高时，网络固定的像素级感受野相对于整幅画面的覆盖比例被大幅压缩，训练期微小的局部偏差会被放大为刺眼的条带伪影（如 Fig. 1 中规则建筑与大面积遮挡所示）。LaMa 正是为了打破这一“局部盲视”困局而设计的。

## 2. 有效感受野与快速傅里叶卷积（FFC）生成器

### 理论感受野的局限与频域全局通信

在传统卷积神经网络中，堆叠多层 $3\times 3$ 卷积配合步长下采样，虽然在纸面计算上能让深层特征的**理论感受野（Theoretical Receptive Field）**覆盖 $256\times 256$ 画幅，但 Luo 等人的经典研究表明，神经网络的**有效感受野（Effective Receptive Field, ERF）**仅占理论范围的一小部分，输出响应对输入像素的梯度贡献围绕中心呈近似高斯衰减。对于图像修复而言，孔洞内部原本就没有有效输入信号，如果网络在早期层缺乏跨越孔洞的直接信息通道，后续层就只能在被掩码污染的局部激活上打转。

为了在网络的每一个残差块内直接建立全图通信，LaMa 引入了 **Fast Fourier Convolution（FFC）**。与显式构造 $HW\times HW$ 全局注意力矩阵的 Transformer 相比，二维离散傅里叶变换（2D FFT）的时间复杂度仅为 $\mathcal O(HW\log(HW))$，且由于每一个频谱系数都是全图所有空间像素的线性组合，在频域执行一次简单的逐频率 $1\times 1$ 卷积再逆变换回空间域，即可让输出特征图的任意空间位置直接汇聚全图信息。更重要的是，自然场景中的周期性结构（窗格、砖墙、百叶、栅栏）在空间域分散各处，转换到频域后却会凝聚为特定频率上的显著能量峰，使网络极易捕捉并延续全局重复节律。

### Fourier Unit 与 Local / Global 四路双分支残差块

设送入频域单元（Fourier Unit）的特征张量为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$。由于输入为实数张量，二维实数快速傅里叶变换 $\operatorname{rFFT2}$ 具有共轭对称性，最后一维只需保留约一半频率分量 $\lfloor W/2\rfloor + 1$。公开源码 [`saicinpainting/training/modules/ffc.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/modules/ffc.py) 中的 `FourierUnit` 依次执行实数 FFT、将复数频谱的实部与虚部沿通道维拼接为 $2C$ 维实张量、施加可学习的 $1\times 1$ 卷积与批归一化及 ReLU 激活，最后重组为复数并通过逆变换 $\operatorname{irFFT2}$ 还原回空间域尺寸：

$$
\mathbf{X}=\operatorname{rFFT2}(\mathbf{x}),\qquad
\mathbf{X}_{ri}=\operatorname{concat}(\Re(\mathbf X),\Im(\mathbf X)),
$$

$$
\tilde{\mathbf X}_{ri}=\operatorname{ReLU}\left(
\operatorname{BN}\left(W_{1\times1}*\mathbf X_{ri}\right)
\right),\qquad
\mathbf{y}=\operatorname{irFFT2}(\tilde{\mathbf X}).
$$

在此基础上，完整的 **FFC 模块**并未将全部通道一股脑送入频域，而是按输入比例 $\alpha_{in}$ 和输出比例 $\alpha_{out}$ 将特征通道划分为**局部分支（Local branch，$\mathbf x_l$）**与**全局分支（Global branch，$\mathbf x_g$）**，形成 $\mathbf{x}=(\mathbf{x}_l,\mathbf{x}_g)$。两路分支之间通过四条子路径实现局部细节与全局上下文的双向交互：

$$
\mathbf y_l=f_{l\to l}(\mathbf x_l)+f_{g\to l}(\mathbf x_g),\qquad
\mathbf y_g=f_{l\to g}(\mathbf x_l)+f_{g\to g}(\mathbf x_g).
$$

其中 $f_{l\to l}$、$f_{l\to g}$ 与 $f_{g\to l}$ 均为常规的 $3\times 3$ 空间卷积，分别负责局部特征提取与跨分支信息交换；而全局到全局的主路径 $f_{g\to g}$ 则采用基于上述 `FourierUnit` 的谱变换模块（**Spectral Transform**）。在官方主配置 `ffc_resnet_075.yaml` 中，中间残差块的全局通道占比被设定为 $\alpha_{in}=\alpha_{out}=0.75$（即 75% 通道走频域全局支路，25% 通道保留空间局部卷积）。阅读源码时有一个值得留意的配置事实：虽然原始 FFC 论文在 Spectral Transform 中还设计了一个分块处理的局部频率单元（Local Fourier Unit, LFU），但 LaMa 的官方配置文件中显式设置了 `enable_lfu: false`——这说明 LaMa 的性能增益完全来自全图 `FourierUnit` 与双分支残差架构，并未启用额外的分块 LFU。

![LaMa generator、FFC 与 Spectral Transform](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig02-architecture-ffc.webp)

*左侧是 LaMa 训练与生成主链，中间是 FFC 四路信息流，右侧是 Spectral Transform。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 2, CC BY 4.0.*

Fig. 2 汇总了生成器的整体拓扑。标准 **LaMa-Fourier** 生成器采用经典的 ResNet 式编码-残差-解码骨架：输入 4 通道张量经初始卷积与 **3 次空间下采样**后，穿过 **9 个 FFC 残差块**（内部始终维护元组形式的 `(x_l, x_g)` 双分支张量流），再经 **3 次转置卷积上采样**输出 3 通道 RGB 图像，总参数量约为 **27M**（扩展版 Big LaMa 则将残差块加深至 18 个，参数量约 51M）。

![周期结构修复对比](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig04-periodic-structure-comparison.webp)

*窗户与链式栅栏上的对比。LaMa-Fourier 用较少参数保持了更连续的重复结构。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 4, CC BY 4.0.*

Fig. 4 直观展示了这种频域归纳偏置在周期结构上的威力。考虑一维周期信号 $x[n]=\cos(2\pi f n+\phi)$，小窗口卷积在孔洞边缘只能窥见不足半个周期的片段，难以估计全局频率 $f$ 与对侧相位 $\phi$；而 `FourierUnit` 直接在频域对齐整幅画面的主频能量。因此在建筑立面窗格与铁丝网栅栏修复中，常规卷积基线（LaMa-Regular）在孔洞中央迅速退化为模糊色块或杂乱断线，而仅 27M 参数的 LaMa-Fourier 却能清晰、连贯地将网格结构从孔洞一侧桥接到另一侧。

## 3. 高感受野感知损失与掩码对齐对抗训练

### 高感受野感知损失（HRFPL）与预训练主干消融

如果仅有看得远的生成器，而损失函数只考核局部像素误差，网络依然缺乏恢复大尺度连贯结构的优化动力。考虑到大孔洞修复具有天然的多解性（例如抹去前景后的背景未必与原图唯一的 ground truth 逐像素重合），LaMa 完全放弃了逐像素 $L_1$ 重建损失，转而提出 **高感受野感知损失（High Receptive Field Perceptual Loss, HRFPL）**：

$$
\mathcal L_{\mathrm{HRFPL}}(\hat{\mathbf x},\mathbf x)
=\sum_k w_k\left\|\phi_k(\hat{\mathbf x})-\phi_k(\mathbf x)\right\|_2^2,
$$

其中 $\phi_k(\cdot)$ 表示冻结的预训练特征提取网络第 $k$ 个残差阶段的特征图（代码实现在 [`training/losses/perceptual.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/perceptual.py) 的 `ResNetPL` 中）。与传统感知损失普遍采用 ImageNet 分类预训练的 VGG19 不同，HRFPL 做出了两项针对性改进：一是将主干替换为在后层引入**扩张卷积**（dilated convolutions）的 ResNet50，使高层特征在不损失空间分辨率的前提下覆盖更广的感受野；二是采用 **ADE20K 密集语义分割**作为预训练任务（pretext task），迫使特征空间关注全图场景布局与物体部件边界，而非仅仅聚焦于分类显著局部。

![感知损失权重](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table05-perceptual-loss-weights.webp)

*补充材料给出的各感知损失最优权重。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 5, CC BY 4.0.*

![感知损失消融](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-table03-perceptual-loss-ablation.webp)

*在 segmentation masks 上比较不同 perceptual feature extractor。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Table 3, CC BY 4.0.*

| 感知损失变体 | 特征主干 | 预训练任务（Pretext） | 是否含扩张卷积 | 最优损失权重 | Places Seg. FID ↓ | Places Seg. LPIPS ↓ |
| --- | --- | --- | :---: | ---: | ---: | ---: |
| **HRFPL（本文采用）** | ResNet50 | Semantic Segmentation | 是 | 30.0 | **5.69** | **0.059** |
| ClfPL-Dilated | ResNet50 | ImageNet Classification | 是 | 1.0 | 5.87 | 0.059 |
| ClfPL-Regular | ResNet50 | ImageNet Classification | 否 | 1.0 | 6.00 | 0.061 |
| ClfPL-VGG19 | VGG19 | ImageNet Classification | 否 | 0.1 | 6.29 | 0.063 |
| Perceptual Baseline | - | - | - | - | 6.46 | 0.065 |

如补充材料 Table 5 与主文 Table 3 所示，在各自通过网格搜索选定最优损失权重后（权重数值差异源于不同网络特征激活的绝对量级不同），从 VGG19（FID `6.29`）切换到标准分类 ResNet50（`6.00`）、再引入扩张卷积（`5.87`）、最终叠加语义分割预训练得到 HRFPL（`5.69`），FID 与 LPIPS 呈现出清晰的阶梯式改善，证实了“扩张卷积扩大感受野”与“密集分割预训练强化空间语义”两项因素的互补增益。

### 掩码对齐 PatchGAN、特征匹配与 R1 正则化

为了在感知损失之上补足逼真的高频局部纹理，LaMa 引入了 PatchGAN 风格的判别器 $D_\xi$（结构见补充材料 Listing 2：由 5 层 $4\times 4$ 卷积、BatchNorm 与 LeakyReLU 组成，输出二维空间 patch 评分图）：

```python
NLayerDiscriminator(
    Conv2d(3, 64, kernel_size=4, stride=2, padding=2), LeakyReLU(0.2),
    Conv2d(64, 128, kernel_size=4, stride=2, padding=2), BatchNorm2d(128), LeakyReLU(0.2),
    Conv2d(128, 256, kernel_size=4, stride=2, padding=2), BatchNorm2d(256), LeakyReLU(0.2),
    Conv2d(256, 512, kernel_size=4, stride=1, padding=2), BatchNorm2d(512), LeakyReLU(0.2),
    Conv2d(512, 1, kernel_size=4, stride=1, padding=2),
)
```

在非饱和对抗损失（Non-saturating adversarial loss）的基础上，LaMa 设计了一个关键的**掩码对齐局部判别策略**：由于生成图在已知区域（$m=0$）会被原图像素直接覆盖，判别器只应将**感受野与掩码区域（$m=1$）相交的局部图块**判定为伪造样本（Fake），而将完全落在已知区域内的图块同样视为真实样本（Real）。这样既防止了生成器浪费容量去拟合已知像素，又利用跨越掩码边界的判别器感受野自然约束了孔洞接缝处的连续性。

同时，为抑制对抗训练的震荡与高频棋盘格伪影，系统在 [`adversarial.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/adversarial.py) 与 [`feature_matching.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/feature_matching.py) 中进一步集成了判别器中间层特征匹配损失 $\mathcal L_{\mathrm{DiscPL}}$（即 Feature Matching Loss）与真实样本梯度惩罚 $R_1$ 正则项：

$$
\mathcal L_{\mathrm{DiscPL}}
=\sum_k \left\|D_\xi^{(k)}(\hat{\mathbf x})-D_\xi^{(k)}(\mathbf x)\right\|_1,\qquad
R_1=\mathbb E_{\mathbf x}\left[\left\|\nabla_{\mathbf x}D_\xi(\mathbf x)\right\|_2^2\right].
$$

最终，整个系统的总训练目标函数由四项加权求和构成（具体调度逻辑位于 [`training/trainers/default.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/trainers/default.py)）：

$$
\mathcal L_{final}
=\kappa\mathcal L_{Adv}
+\alpha\mathcal L_{\mathrm{HRFPL}}
+\beta\mathcal L_{\mathrm{DiscPL}}
+\gamma R_1,\qquad
(\kappa=10,\ \alpha=30,\ \beta=100,\ \gamma=0.001).
$$

## 4. 大掩码采样策略与几何宽度统计

### 激进的混合掩码生成器（Aggressive Mask Generation）

![LaMa mask policies](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig03-mask-policies.webp)

*DeepFillv2 风格、窄 irregular、宽 irregular 与 box masks 的视觉差异。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 3, CC BY 4.0.*

LaMa 的第三块基石是将训练掩码的生成分布提升为与网络架构同等重要的核心超参数。如 Fig. 3 所示，传统方法（如 DeepFillv2）使用的随机笔画掩码虽然也可能覆盖一定面积，但由于笔刷较细，几乎每个缺失像素都在几像素之内紧挨着可见背景，网络只需学会局部插值就能轻松过关。为了彻底堵死这种“局部抄近道”的可能，LaMa 在 [`saicinpainting/training/data/masks.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/data/masks.py) 中通过 `MixedMaskGenerator` 以 **50% : 50%** 的概率混合采样两类激进掩码（核心逻辑见补充材料 Listing 1）：

```python
def mixed_mask(height, width, rng):
    if rng.random() < 0.5:
        mask = random_irregular_mask(
            height, width, max_angle=4, max_len=200, max_width=100, max_times=5
        )
    else:
        mask = random_box_mask(
            height, width, margin=10, bbox_min_size=30, bbox_max_size=150, max_times=4
        )
    return mask.astype("float32")
```

第一类是具有极大笔刷宽度（在 256 分辨率下最大笔宽可达 100 像素、最大折线长度 200 像素）的随机多边形折线（`RandomIrregularMaskGenerator`），第二类是最多叠加 4 个大尺寸矩形框的盒式掩码（`RandomRectangleMaskGenerator`）。

![随机 mask 参数](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table04-mask-generator-parameters.webp)

*256 与 512 分辨率下 narrow、medium、wide、train 的 irregular/box 参数。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 4, CC BY 4.0.*

### 掩码几何宽度（Mask Width）与多尺度测试集分布

为了定量刻画“孔洞内部像素距离已知边界到底有多远”，论文定义了比单纯面积占比（masked area ratio）更具几何意义的 **掩码宽度（Mask Width）** 指标。对掩码区域 $\Omega_{mask}$ 内的任意缺失像素 $p$，先计算其到已知像素集合 $\Omega_{known}$ 的最短欧氏距离 $d(p)$，再对全体缺失像素求平均：

$$
d(p)=\min_{q\in\Omega_{known}}\|p-q\|_2,\qquad
w(\mathbf m)=\frac{1}{|\Omega_{mask}|}
\sum_{p\in\Omega_{mask}} d(p).
$$

一条横贯全图但仅 3 像素宽的细长划痕，其平均距离 $w(\mathbf m)$ 接近于 1；而一个同样面积的实心大圆盘，其内部像素的平均距离则大得多。

![256 分辨率 mask 统计](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig04-mask-statistics-256.webp)

*256 分辨率下 DeepFillv2、LaMa training 和 narrow/medium/wide test masks 的面积与 width 分布。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 4, CC BY 4.0.*

![512 分辨率 mask 统计](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig05-mask-statistics-512.webp)

*512 分辨率下 Places narrow、medium、wide 与 segmentation masks 的统计。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 5, CC BY 4.0.*

![Segmentation mask examples](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig03-segmentation-masks.webp)

*Places segmentation test set 的对象区域、目标 hole 与背景限制示例。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 3, CC BY 4.0.*

结合补充材料 Table 4 与 Fig. 4、Fig. 5、Fig. 3 可以看到，DeepFillv2 训练掩码的几何宽度集中在极窄区间，而 LaMa 训练掩码广泛覆盖了从窄到宽的距离谱系。在测试端，作者除了按笔刷粗细构建 **Narrow**、**Medium**、**Wide** 三档随机掩码测试集（每档各 30,000 个图像-掩码对）外，还专门利用 Detectron2 提取自然场景中的真实前景物体轮廓，构建了约 4,000 对 **Segmentation Masks** 测试集（限制单个掩码面积在整图的 4%～40% 之间），以贴近真实世界中“点击移除特定物体”的产品形态。

## 5. 训练配置、评测基准与主实验对比

### 训练超参数配置与评测指标体系

标准 LaMa 模型与扩展版 Big LaMa 的核心训练契约对比如下：

| 配置维度 | 标准 LaMa（LaMa-Fourier） | 扩展版 Big LaMa |
| --- | --- | --- |
| **训练集与图像规模** | Places-Standard 高分辨率集；CelebA-HQ | Places-Challenge 全集（约 **4.5M** 张图像） |
| **训练裁剪尺寸** | 从约 $512\times 512$ 图像随机裁剪 **$256\times 256$** | 同样在 **$256\times 256$** 裁剪块上训练 |
| **生成器规模** | 9 个 FFC 残差块（`ratio_g=0.75`，`enable_lfu=false`），**27M** 参数 | **18 个** FFC 残差块，**51M** 参数 |
| **批次大小与迭代数** | Batch size = **30**，训练 **1,000,000** 步 | Batch size = **120**（4 倍吞吐），训练更充分 |
| **优化器与学习率** | Adam；生成器 $\text{LR}=10^{-3}$，判别器 $\text{LR}=10^{-4}$ | 同左；在 8×V100 GPU 上训练约 **240 小时** |

在定量评测中，论文同时报告集合分布距离 **FID**（Fréchet Inception Distance）与成对感知距离 **LPIPS**：

$$
\operatorname{FID}
=\|\mu_r-\mu_g\|_2^2
+\operatorname{Tr}\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right).
$$

为了防止全测试集的平均指标被相对简单的小面积掩码稀释，补充材料 Table 2 与 Table 3 还将 Places（$512\times 512$）与 CelebA-HQ（$256\times 256$）中**掩码遮挡面积高达 40%～50% 的最困难子集**单独剥离出来报告。

![Places 完整指标](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table02-places-full-metrics.webp)

*Places 上 narrow/medium/wide/segmentation masks 的完整 FID 与 LPIPS。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 2, CC BY 4.0.*

![CelebA-HQ 完整指标](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table03-celebahq-full-metrics.webp)

*CelebA-HQ 上 40%-50% masked 困难子集与全部样本指标。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 3, CC BY 4.0.*

此外，针对自动指标无法完全反映人类对修复接缝敏感度的问题，论文还实施了两项盲测用户研究（User Study，界面见补充材料 Fig. 1 与 Fig. 2）：一是 **双图并排相对偏好（Side-by-side Relative Preference, RP，越高越好）**，以 LaMa-Fourier 为固定基准（定义为 50%），让受试者在 LaMa 与对比模型之间挑选更真实的一张；二是 **单图盲找掩码准确率（Spot-the-mask Accuracy, Acc，越低越好）**，仅向受试者展示单张修复结果并要求点击最可疑的伪造位置，未点中掩码区域即说明修复达到了“肉眼难辨”的融合效果。

![Side-by-side 用户研究界面](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig01-user-study-side-by-side.webp)

*原始界面左、右为两个模型结果，中间显示 masked input。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 1, CC BY 4.0.*

![Spot-the-mask 用户研究界面](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig02-user-study-spot-mask.webp)

*评估者只看到修复图并点击最可疑位置。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 2, CC BY 4.0.*

### Table 1 主结果与盲测用户偏好分析

![LaMa 主结果](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-table01-main-results.webp)

*Places 与 CelebA-HQ 的主结果。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Table 1, CC BY 4.0.*

![LaMa 用户研究结果](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table01-user-study.webp)

*Places 512 上 narrow/wide masks 的相对偏好与 spot-the-mask 准确率。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 1, CC BY 4.0.*

汇总主文 Table 1 与补充材料 Table 1，LaMa-Fourier 在各数据集切分上的核心指标如下：

| 评测维度 | Places Narrow | Places Wide | Places Segmentation | CelebA-HQ Narrow | CelebA-HQ Wide |
| --- | ---: | ---: | ---: | ---: | ---: |
| **LaMa-Fourier FID ↓** | **0.63** | 2.21 | **5.35** | **7.26** | **6.96** |
| **LaMa-Fourier LPIPS ↓** | **0.090** | **0.135** | **0.058** | **0.085** | **0.098** |
| **Spot-the-mask Acc (%) ↓** | **34 ± 1.7** | **54 ± 1.7** | - | - | - |

横向对比当时最强的基线模型，仅 **27M** 参数的确定性模型 LaMa-Fourier 在绝大多数指标上全面超越了参数量高达 **109M** 的随机生成对抗模型 CoModGAN 与 **85M** 参数的多阶段模型 MADF。唯一值得留意的细微例外出现在 Places Wide 的 FID 单项上：CoModGAN 取得 `1.82`（优于 LaMa 的 `2.21`），但在衡量与原图上下文结构一致性的 LPIPS 上，LaMa（`0.135`）依然显著优于 CoModGAN（`0.147`）。在人类盲测中，受试者在 Narrow 掩码下仅有 `34%` 的概率能找准 LaMa 的修补区域，在困难的 Wide 掩码下也仅有 `54%`；并排偏好（RP）对比中，除 CoModGAN 在极宽掩码下与 LaMa 旗鼓相当外，其余方法均被 LaMa 大幅拉开差距。

## 6. 架构消融、掩码迁移与跨分辨率泛化分析

### Table 2 架构消融：验证频域全局感受野的真实贡献

为了排除“性能提升仅仅是因为参数量变化或网络深度改变”的干扰假设，论文在 Table 2 中系统对比了 5 种生成器变体：

![LaMa 架构消融](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-table02-architecture-ablation.webp)

*Fourier、dilated、regular、shallow/deep 生成器对比。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Table 2, CC BY 4.0.*

| 生成器变体 | 卷积算子类型 | 参数量 | 残差块数 | Places Narrow<br>FID / LPIPS | Places Wide<br>FID / LPIPS |
| --- | --- | ---: | ---: | :---: | :---: |
| **Base Fourier（标准 LaMa）** | Fast Fourier Conv | **27M** | 9 | 0.63 / 0.090 | **2.21 / 0.135** |
| **Base Dilated** | Multi-Dilated Conv | 46M | 9 | 0.66 / **0.089** | 2.30 / 0.136 |
| **Base Regular** | 普通 $3\times 3$ Conv | 46M | 9 | **0.60 / 0.089** | 3.51 / 0.139 |
| **Shallow Fourier** | Fast Fourier Conv | 19M | 6 | 0.72 / 0.094 | 2.31 / 0.138 |
| **Deep Regular** | 普通 $3\times 3$ Conv | 74M | 15 | 0.63 / 0.090 | 2.62 / 0.137 |

这组对照实验揭示了极其清晰的物理规律：
1. **窄洞靠局部，宽洞靠全局**：在孔洞很窄的 `Narrow` 测试集上，局部邻域已包含充足线索，46M 参数的普通卷积 `Base Regular` 甚至拿到了略好于 Fourier 的 FID（`0.60` vs `0.63`）。然而一旦切换到宽孔洞 `Wide` 测试集，`Base Regular` 的 FID 立刻恶化至 `3.51`，而参数少约 40% 的 `Base Fourier` 稳稳守在 **`2.21`**。
2. **单纯堆叠深度与参数无法替代全局算子**：将普通卷积加深至 15 个残差块、参数量膨胀至 **74M**（约为标准 LaMa 的 2.7 倍）得到 `Deep Regular`，其 Wide FID 虽改善至 `2.62`，却依然不及仅 19M 参数、6 个残差块的 `Shallow Fourier`（`2.31`）与 27M 的 `Base Fourier`（`2.21`）。同时，采用多分支扩张卷积扩大感受野的 `Base Dilated`（`2.30`）也显著优于普通卷积，共同证实了**早期且宽广的有效感受野才是攻克大孔洞的核心机制**。

### Table 4 掩码训练消融：大掩码策略的跨模型迁移性

![宽 mask 训练消融](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-table04-mask-training-ablation.webp)

*同一方法分别使用 narrow 与 wide mask 训练后的结果。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Table 4, CC BY 4.0.*

| 修复模型 | 训练期掩码分布 | Places Narrow<br>FID / LPIPS | Places Wide<br>FID / LPIPS | 掩码策略迁移表现 |
| --- | --- | :---: | :---: | --- |
| **LaMa-Regular** | Narrow Training | 0.68 / 0.091 | 5.41 / 0.144 | 切换 Wide 训练后，宽洞与窄洞指标**同步全面提升** |
| | **Wide Training** | **0.60 / 0.089** | **3.51 / 0.139** | |
| **RegionWise** | Narrow Training | 0.90 / 0.102 | 4.75 / 0.149 | 切换 Wide 训练后，宽洞与窄洞指标**同步全面提升** |
| | **Wide Training** | **0.74 / 0.095** | **3.56 / 0.144** | |
| **DeepFill v2** | Narrow Training | **1.06 / 0.104** | 5.20 / 0.155 | 宽洞显著改善，但窄洞指标出现一定程度退化 |
| | **Wide Training** | 1.35 / 0.107 | **4.34 / 0.148** | |
| **EdgeConnect** | Narrow Training | **1.33 / 0.111** | 8.37 / 0.160 | 边缘预测中间瓶颈受宽洞冲击，窄洞明显退化 |
| | **Wide Training** | 2.78 / 0.141 | **7.94 / 0.160** | |

Table 4 将 LaMa 提出的宽掩码训练策略（Wide Training）横向移植给四个不同架构的模型。结果显示，**所有四个模型在切换宽掩码训练后，Wide 测试集上的 FID 均获得大幅改善**（例如 LaMa-Regular 从 `5.41` 降至 `3.51`，RegionWise 从 `4.75` 降至 `3.56`）。更有趣的是，容量充足的单阶段模型（LaMa-Regular 与 RegionWise）在宽洞训练后连带 Narrow 测试集也变得更好，因为困难样本迫使网络放弃了脆弱的局部插值捷径；而受制于两阶段边缘预测瓶颈的 EdgeConnect 在面对过宽孔洞时中间边缘图崩溃，反而拖累了窄洞表现。这说明激进的训练分布必须与具备长程建模潜力的网络架构相匹配。

### Fig. 5 & Fig. 6：从 256 训练块到 1920×1536 的跨分辨率泛化

![LaMa 跨分辨率修复](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig05-high-resolution-transfer.webp)

*标准模型用 256 crop 训练，在 640×512 与 1920×1536 输入上的直接前向结果。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 5, CC BY 4.0.*

![FID 与 LPIPS 随分辨率变化](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig06-resolution-metrics.webp)

*不同模型的 FID/LPIPS 随测试分辨率变化。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 6, CC BY 4.0.*

Fig. 5 与 Fig. 6 展示了论文最引人注目的实验发现：所有 LaMa 变体均仅在 $256\times 256$ 图块上训练，测试时不做分块切片（tiling），直接将 $640\times 512$ 乃至 $1920\times 1536$ 的高分辨率大图整张送入全卷积生成器。在 $640\times 512$ 分辨率下，普通卷积与傅里叶卷积尚能维持相近的视觉质量；但当分辨率提升至 $1920\times 1536$（边长扩大约 3 倍）时，LaMa-Regular 与 Big Regular 在孔洞内部出现了严重的竖向条带伪影与色彩崩塌，即便直接在 $512\times 512$ 上训练的 MADF 也产生了明显斑块，唯独 **LaMa-Fourier 与 Big LaMa-Fourier 依然平滑、连贯地补全了高分辨率墙面与室内结构**。Fig. 6 的定量曲线同样印证了这一趋势：随着测试分辨率从 256 逐步升高至 2048，虽然所有模型的 FID 与 LPIPS 都在上升，但 FFC 系列曲线的退化斜率显著平缓于常规卷积与扩张卷积基线。

这种跨分辨率稳健性的物理根源在于：当输入图像尺寸扩大 3 倍而掩码占整图的比例保持不变时，孔洞的绝对像素跨度也扩大了 3 倍。固定卷积核尺寸的常规卷积或固定扩张率的多分支扩张卷积，其相对整幅画面的感受野覆盖比例缩小为原来的 $1/3$，再次陷入“孔洞中心看不到边界”的窘境；而 FFC 的全局分支在任意特征图尺寸下始终对整张特征图执行 2D FFT，其全图覆盖比例不随分辨率升高而缩水，且自然图像的低频布局能量与频域 $1\times 1$ 卷积权重在不同网格尺寸间具备良好的经验迁移性。

当然，在工程实践中看待这一特性需要保持理性：拥有全图感受野只意味着算子能够读取远处像素，面对存在多个周期相位歧义的复杂立面时网络仍可能选错对齐相位；频域对低频布局的稳定保持并不能凭空合成超出训练分布的高频传感器噪点或微纹理；而在系统层面，8K 级超大图或极端长宽比全景图仍会受制于 FFT 显存工作区（workspace）开销、BatchNorm 统计量偏移以及缺乏语义理解的固有局限，因此实际部署必须明确标定经过验证的安全分辨率区间。

## 7. Big LaMa 扩展、失败模式与替代架构对照

### Big LaMa 的正反例剖析与跨领域探索

通过将残差块翻倍至 18 个（51M 参数）、训练数据扩展至 450 万张 Places-Challenge 图像并将批次大小扩大到 120，**Big LaMa** 进一步推高了单阶段修复的视觉上限。

![Big LaMa 正例](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig06-big-lama-positive.webp)

*建筑、街景、游乐场和百叶背景中的成功对象移除。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 6, CC BY 4.0.*

![Big LaMa 失败案例](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig07-big-lama-negative.webp)

*链网、长椅、工业结构和街景中的透视或复杂背景失败。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 7, CC BY 4.0.*

对照补充材料 Fig. 6 的成功案例与 Fig. 7 的典型失败案例，可以非常清楚地摸清频域卷积修复的能力边界：当背景是由近乎平行于成像平面的建筑窗户、百叶窗、摩天轮辐条或规则路面构成时（Fig. 6），Big LaMa 能够在大面积移除前景人物后完美续接背景线条。反之，当场景出现以下五类复杂几何与光学耦合时（Fig. 7），模型便容易暴露出伪影：
1. **强透视收缩的非平稳网格**：如近大远小的倾斜链式铁丝网，其空间频率随深度连续变化，全局 FFT 倾向于混合单一平稳频率，导致补全区域出现网格扭曲；
2. **多层深度交错的复杂背景**：当被删除人物背后同时重叠着近处栏杆、中景立柱与远处建筑时，缺乏显式三维深度与平面先验的纯 RGB 网络容易将不同深度层的结构粘连混淆；
3. **关键结构锚点被完全遮挡**：当掩码恰好吞没了物体的转角或几何交汇点，孔洞外部缺乏可外推的唯一拓扑线索；
4. **浅景深与散焦光斑（Bokeh）不一致**：当背景处于镜头大光圈虚焦状态时，生成器有时会补出过度锐利或模糊半径失配的纹理块；
5. **长周期相位的半格错位**：虽然频域抓住了重复周期的步长，但在孔洞跨度过大时仍可能在接缝处出现错位半格的几何阶梯。

![频谱图、病理、遥感、绘画和游戏画面的跨领域修复](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig08-domain-transfer.webp)

*Big LaMa 在 Places 之外的音乐频谱、组织切片、鸟瞰图、绘画和游戏画面上展示的案例。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 8, CC BY 4.0.*

![Outpainting 与 MRI 示例](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig09-more-domain-transfer.webp)

*店面 outpainting 和 MRI 图像修复示例。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 9, CC BY 4.0.*

补充材料 Fig. 8 与 Fig. 9 还展示了将 Places 上训练的 Big LaMa 直接零样本应用于音乐声谱图、病理组织切片、卫星遥感鸟瞰图、艺术油画、游戏画面、图像外扩（Outpainting）以及脑部 MRI 切片的探索案例。这些示例说明自然图像中学到的低层连续性与频率延续先验具备一定的跨模态迁移潜力；但必须强调，在医学影像（MRI/病理）或科学遥感等严肃领域，视觉上平滑好看的纹理修补绝不等于真实的解剖或地理事实，绝不能未经临床验证直接用于辅助诊断或定量科学测量。

### LaMa-Dilated 替代路线与推理吞吐耗时剖析

![LaMa-Dilated 架构](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig10-dilated-architecture.webp)

*每个 residual block 将通道分为 dilation 1/2/4/8 的四组并求和。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 10, CC BY 4.0.*

考虑到部分移动端芯片、嵌入式 NPU 或专用推理引擎对复数算子 `rFFT / irFFT` 的硬件加速支持不够完善，论文在补充材料 Fig. 10 中给出了纯空间域的替代架构 **LaMa-Dilated**：将每个残差块的输入通道均匀切分为 4 组，分别通过扩张率（dilation rate）为 $1, 2, 4, 8$ 的 $3\times 3$ 卷积核提取多尺度上下文，再将四路结果求和融合。从前文 Table 2 可知，LaMa-Dilated（46M 参数）在 Places Wide 上取得了 `2.30` 的 FID（非常接近 LaMa-Fourier 的 `2.21`），在 Narrow 上 LPIPS 甚至达到 `0.089`。虽然它在超高分辨率下的退化速度仍快于 FFT 且大扩张率偶有网格效应（gridding artifacts），但在算子受限的端侧部署环境中，LaMa-Dilated 是一条极具实用价值的工程备选方案。

![LaMa 推理时间曲线](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig11-inference-time.webp)

*Nvidia 1080Ti、batch size 100、100 次运行平均的 sec/image 曲线。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 11, CC BY 4.0.*

补充材料 Fig. 11 对比了各模型变体在单张 NVIDIA GTX 1080Ti 上（设定 `batch_size = 100` 跑满 GPU 吞吐，取 100 次运行均值）随分辨率升高的单图平均耗时曲线。数据显示，尽管标准 LaMa-Fourier 的参数量（27M）比 LaMa-Regular（46M）少约 40%，但由于 FFT 频域变换与实虚部重排涉及额外的显存读写开销，**LaMa-Fourier 的平均推理耗时反而比 LaMa-Regular 慢约 20%**（而 46M 的 LaMa-Dilated 与 51M 的 Big LaMa-Fourier 则处于更高耗时区间）。这提醒我们在评估模型效率时，必须将参数规模与实际硬件上的算子执行延迟、Batch-1 交互延迟及 FFT 工作区显存峰值区分开来。

## 8. 官方源码架构、Hydra 配置与可选精化走读

### 代码模块映射与 Hydra 配置体系

对照固定提交 [`advimman/lama@786f5936`](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7)，官方仓库基于 PyTorch Lightning 与 Hydra 配置框架组织，训练与推理的核心代码链路一一对应如下：

| 核心职责 | 固定提交源码路径 | 关键类 / 函数与实现要点 |
| --- | --- | --- |
| **FFC 与生成器** | [`saicinpainting/training/modules/ffc.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/modules/ffc.py) | `FourierUnit`（调用 `torch.fft.rfftn/irfftn`）、`SpectralTransform`、`FFC`、`FFCResnetBlock`、`FFCResNetGenerator` |
| **掩码生成器** | [`saicinpainting/training/data/masks.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/data/masks.py) | `RandomIrregularMaskGenerator`、`RandomRectangleMaskGenerator`、`MixedMaskGenerator` |
| **高感受野感知损失** | [`saicinpainting/training/losses/perceptual.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/perceptual.py) | `ResNetPL`：封装冻结的 ADE20K 扩张卷积 ResNet50 提取四阶段特征并计算距离 |
| **对抗与特征匹配损失** | [`saicinpainting/training/losses/adversarial.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/adversarial.py)<br>[`saicinpainting/training/losses/feature_matching.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/feature_matching.py) | `NonSaturatingWithR1`（掩码对齐 PatchGAN 与 $R_1$ 梯度惩罚）及判别器中间层 $L_1$ 特征匹配 |
| **训练器与入口脚本** | [`saicinpainting/training/trainers/default.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/trainers/default.py)<br>[`bin/train.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/bin/train.py) / [`bin/predict.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/bin/predict.py) | `DefaultInpaintingTrainingModule` 串联前向、合成与四项损失；`predict.py` 负责批量推理 |
| **多尺度测试精化** | [`saicinpainting/evaluation/refinement.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/evaluation/refinement.py) | `refine_predict`：测试阶段可选的多尺度潜特征迭代优化扩展 |

在实验配置层面，LaMa 大量使用 Hydra 的层级组合机制。例如顶层训练配置 [`configs/training/lama-fourier.yaml`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/lama-fourier.yaml) 与 [`configs/training/big-lama.yaml`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/big-lama.yaml) 会引用生成器子配置 [`configs/training/generator/ffc_resnet_075.yaml`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/generator/ffc_resnet_075.yaml)，解析合并后的核心超参数结构如下：

```yaml
generator:
  kind: ffc_resnet
  input_nc: 4
  output_nc: 3
  n_downsampling: 3
  n_blocks: 9          # Big LaMa 中覆盖为 18
  resnet_conv_kwargs:
    ratio_gin: 0.75
    ratio_gout: 0.75
    enable_lfu: false

losses:
  adversarial: { weight: 10 }
  resnet_pl: { weight: 30 }
  feature_matching: { weight: 100 }
  r1: { weight: 0.001 }
```

### 预测脚本细节与可选的多尺度精化（Refinement）

调用官方 [`bin/predict.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/bin/predict.py) 执行批量推理时，标准命令行形式为：

```bash
python3 bin/predict.py \
  model.path=/absolute/path/to/big-lama \
  indir=/absolute/path/to/input \
  outdir=/absolute/path/to/output
```

虽然模型本身是一次前向传播，但预测管线中包含了几个直接决定输出质量的工程细节：首先，由于生成器包含 3 次步长为 2 的下采样，输入图像的宽高必须向上填充（modulo padding，默认采用镜像填充 reflection padding）至 8 的整数倍，前向完成后再裁剪回原始尺寸；其次，模型加载时不仅需要 `.ckpt` 权重文件，还必须读取同目录下的 `config.yaml` 以还原正确的残差块数量与通道比例；最后，输出图像通过 `image * (1 - mask) + predicted * mask` 严格保留掩码外的原始像素，且输入掩码应保持二值化（若将抗锯齿羽化后的半透明灰度掩码直接送入仅见过 `{0, 1}` 二值掩码的网络，容易在孔洞边缘诱发异常色带）。

此外，仓库在 [`saicinpainting/evaluation/refinement.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/evaluation/refinement.py) 中还实现了一个可选的测试时多尺度精化模块（Multi-scale Refinement）：对高分辨率输入先构建图像金字塔，由粗到细运行生成器，并在每个尺度上冻结网络权重、利用 $L_1$ 边界一致性损失对生成器编码特征执行若干步梯度下降优化。在研读或复现时需要严格区分这一模块与标准 LaMa 的边界：

| 对比维度 | 标准 LaMa 推理（论文主表口径） | 多尺度精化（Refinement，仓库可选扩展） |
| --- | --- | --- |
| **计算范式** | 纯前馈单次生成器前向传播（1-pass forward） | 多尺度金字塔 + 测试时反向传播迭代优化潜特征 |
| **延迟与显存** | 毫秒至百毫秒级，仅占用单次前向激活显存 | 耗时增加数倍至数十倍，需保存计算图与梯度状态 |
| **适用场景** | 实时交互式涂抹消除、高吞吐批量处理与基准对比 | 离线单张超高分辨率精修、对耗时不敏感的导出模式 |

## 9. 方法演进对比、工程落地管线与局限性审视

### 在图像修复技术演进谱系中的坐标

将 LaMa 置于 2019 年至今的图像修复技术演进脉络中，可以清晰地把握不同范式的取舍：
- **空间多尺度与 Transformer 修复模型（AOT-GAN, MAT, ZITS）**：AOT-GAN 通过聚合多扩张率上下文变换提升大洞修复；CVPR 2022 的 MAT（Mask-Aware Transformer）利用掩码感知动态注意力建模长程依赖，在非周期语义内容生成上比固定频域混合更具灵活性；ZITS 则显式引入线框与边缘结构引导加强建筑透视控制。它们都继承了 LaMa“扩大有效感受野 + 宽掩码训练”的核心思想。
- **扩散模型修复（RePaint, Latent Diffusion / SD Inpainting）**：以 Stable Diffusion Inpainting 为代表的扩散模型拥有强大的开放世界语义先验与文本引导能力，当掩码覆盖整张人脸或需要凭空生成全新物体时，扩散模型远超确定性回归网络；但扩散模型需要数十步迭代去噪，延迟较高且存在随机幻觉风险。在 **“无需文本提示、快速且确定性地抹除前景杂物并无缝延续背景纹理”** 这一高频刚需场景下，仅 27M/51M 参数、单步前馈的 LaMa 至今仍是工业界最常用的轻量级基线与混合管线首选（例如先用 SAM 点击获取目标掩码，交由 LaMa 毫秒级预览或直接消除背景，仅在超大语义缺失时路由给扩散模型）。

### 生产级对象消除服务落地指南

若将 LaMa 部署为线上生产服务，建议围绕输入预处理、分辨率路由、合成质检与合规审计构建完整闭环：
1. **掩码膨胀与羽化职责分离**：分割模型（如 SAM）或用户涂抹生成的原始掩码往往紧贴物体边缘，直接修复极易残留物体边缘的暗影或色差光晕。线上预处理应先根据目标尺寸自适应执行形态学膨胀（dilation），将硬边二值掩码送入 LaMa 生成器；拿到预测图后，仅在最终合成阶段使用轻微羽化（feathering）的软掩码做边缘 Alpha 混合，切勿把灰度软掩码直接作为模型第四通道输入。
2. **避免盲目分块（Tiling）与构建生产评测矩阵**：由于 FFC 依赖全图频谱上下文，简单的网格切块推理会破坏跨块相位一致性并产生拼接缝。对于超大图像，优先采用“整图等比缩放至验证过的安全最大边长（如 1024～1536）推理，或围绕掩码外扩充足上下文裁剪框（crop with large context margin）推理后再贴回原图”的策略。线上验收不应只看单一平均 FID，而应按**掩码面积占比**（$<5\%, 5\%\sim 20\%, 20\%\sim 40\%, >40\%$）、**几何宽度**、**是否触碰图像外边界**（触边修复实质退化为单侧无约束的 Outpainting）、**长宽比**及**场景类型**（人像、文字招牌、强透视建筑）建立分层测试矩阵，并同步监控 Batch-1 的 P50/P95/P99 延迟与显存峰值。
3. **编辑溯源与安全合规边界**：修复模型生成的像素本质上是基于统计先验的视觉猜测。在新闻纪实、司法取证、金融票据、临床医疗读片及科学遥感测量等对事实真实性有严格要求的场景中，必须禁止无标识的静默篡改，系统应完整留存原始图像、操作掩码与模型版本日志，并在导出文件的元数据（如 C2PA 溯源标准）中明确标记生成式编辑区域。

### 局限性总结与可复现性状态评估

综合论文实验与代码审计，LaMa 的核心局限可归纳为四点：其一，**缺乏显式三维几何与语义理解**，面对强透视缩放网格、多层深度遮挡或复杂前景语义补全时容易产生几何扭曲；其二，**确定性单输出掩盖了多解分布**，给定输入只能给出一个固定结果，无法像生成式扩散模型那样提供多样化候选供用户挑选；其三，**跨分辨率泛化存在经验上界**，当测试尺寸继续推高至 4K/8K 时，固定网络容量与 FFT 显存开销依然会成为瓶颈；其四，**开源环境存在年代断层**——官方仓库固定在 2021 年的 PyTorch `1.8.0`、PyTorch Lightning `1.2.9` 与旧版 Hydra 组合上，且早期托管在 Yandex Disk 上的部分数据集与权重链接已发生迁移，在现代 PyTorch 2.x 环境中复现时，建议按“算子单测（验证 `FourierUnit` 形状与梯度）$\rightarrow$ 官方权重固定样例回归比对 $\rightarrow$ 测试集指标复算 $\rightarrow$ 完整训练复现”的阶梯逐级核验。

## 10. 总结与参考资料

LaMa 之所以能成为大掩码图像修复领域的里程碑式工作，在于它清晰地揭示并解决了局部卷积网络在大孔洞修复中的根本症结：**当缺失区域变宽时，不能指望靠堆叠局部卷积层去缓慢拼凑远方线索，而必须让网络在早期残差块内就具备全图通信能力（Fast Fourier Convolution），让损失函数在深层语义特征上考核大范围结构一致性（HRF Perceptual Loss），并让训练数据分布主动暴露宽孔洞难题（Aggressive Large Mask Training）**。这三层设计的紧密咬合，使一个仅 27M 参数的单阶段全卷积网络在周期性结构恢复与跨分辨率泛化上展现出惊人的生命力，至今仍是理解频域视觉算子与工程级图像编辑管线的必读经典。

### 参考资料

1. Roman Suvorov et al., [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161v2), arXiv:`2109.07161v2` / [WACV 2022 Open Access](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html), pp. 2149–2159.
2. Official Repository: [advimman/lama](https://github.com/advimman/lama)（Apache-2.0），本文源码走读固定于 commit [`786f5936`](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7).
3. Lu Chi et al., [Fast Fourier Convolution](https://proceedings.neurips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html), NeurIPS 2020.
4. Wenjie Luo et al., [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html), NeurIPS 2016.
5. Richard Zhang et al., [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924), CVPR 2018.
6. Martin Heusel et al., [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), NeurIPS 2017.
7. Jiahui Yu et al., [Free-Form Image Inpainting with Gated Convolution](https://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Free-Form_Image_Inpainting_With_Gated_Convolution_ICCV_2019_paper.html), ICCV 2019.
8. Kamyar Nazeri et al., [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://arxiv.org/abs/1901.00212), ICCV Workshops 2019.
9. Shengyu Zhao et al., [Large Scale Image Completion via Co-Modulated Generative Adversarial Networks](https://openreview.net/forum?id=sSjqmfsk95O), ICLR 2021.
10. Yanhong Zeng et al., [Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/abs/2104.01431), ICCV 2021.
11. Wenbo Li et al., [MAT: Mask-Aware Transformer for Large Hole Image Inpainting](https://openaccess.thecvf.com/content/CVPR2022/html/Li_MAT_Mask-Aware_Transformer_for_Large_Hole_Image_Inpainting_CVPR_2022_paper.html), CVPR 2022.
12. Qiaole Dong et al., [Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Incremental_Transformer_Structure_Enhanced_Image_Inpainting_With_Masking_Positional_Encoding_CVPR_2022_paper.html), CVPR 2022.
13. Andreas Lugmayr et al., [RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html), CVPR 2022.
14. Robin Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html), CVPR 2022.

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

图像修复常被演示成一个很直观的任务：在图片上涂掉某个对象，模型把空白补上。但真正困难的问题从来不是“把洞填满”，而是让补出的内容同时满足三类约束：它应当与洞外可见证据一致，应当在全图尺度上延续结构与透视，还应当在像素尺度上看不出明显接缝。洞越大、周期结构越长、测试分辨率越高，局部卷积和短程纹理匹配就越容易失效。

**Resolution-robust Large Mask Inpainting with Fourier Convolutions** 提出的 LaMa，正是围绕这个矛盾设计的。它没有增加一个复杂的多阶段结构预测器，而是从“有效感受野”出发，同时改造生成器、训练损失和 mask 分布：生成器在 residual blocks 中引入 Fast Fourier Convolution（FFC），让一部分通道通过频域变换获得图像级感受野；感知损失使用带扩张卷积、在语义分割任务上训练的 ResNet50 特征；训练时主动生成更宽、更不规则的孔洞。三者共同工作，才构成 LaMa，而不是“把普通卷积替换成 FFT”这一句口号。

本文以 arXiv `2109.07161v2` 和 WACV 2022 正式论文为主线，对照官方仓库 [`advimman/lama@786f5936`](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7) 做静态源码阅读。论文和补充材料的图表按 CC BY 4.0 等比例引用；代码仓库为 Apache-2.0。本文没有下载权重，没有启动旧版 PyTorch/Lightning 环境，也没有复现 Places 或 CelebA-HQ 训练、推理和用户研究。

先给出结论：

> LaMa 的长期价值不只是频域模块，而是证明了大区域修复必须把全局上下文能力贯穿到表示、监督和训练样本难度中。FFC 对周期结构和跨分辨率迁移很有效，但它不理解被删除对象的语义，也不能保证唯一正确、事实真实或无限分辨率下稳定。

![LaMa teaser：大区域对象移除与结构恢复](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig01-teaser.webp)

*大掩码对象移除、高分辨率建筑修复和规则结构恢复示例。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 1, CC BY 4.0.*

## 1. 一句话贡献：把“看得远”落实到整条训练链路

LaMa 可以用一句话概括：**用具有图像级感受野的 FFC 生成器、同样强调大感受野的语义感知损失，以及覆盖宽孔洞的大 mask 训练分布，训练一个单阶段、全卷积的修复模型。**

这句话中有四个限定。

第一，LaMa 是**单阶段**模型。输入 masked RGB 和二值 mask，生成器一次前向输出完整 RGB。它没有先显式预测边缘、语义分割、深度或结构图，再交给第二阶段渲染。补充仓库中的 latent refinement 是后来可选的推理扩展，不是论文主表所依赖的标准路径。

第二，LaMa 使用的是 **Fast Fourier Convolution**，但 FFC 不是 LaMa 发明的。FFC 来自 Chi 等人在 NeurIPS 2020 的工作。LaMa 的贡献是把 FFC 放入图像修复生成器，并通过损失、mask 策略和系统实验说明它为何适合大洞和周期结构。

第三，“resolution-robust”是一个有实验边界的经验结论。标准模型用从约 $512\times512$ 图片裁出的 $256\times256$ crop 训练，论文把同一全卷积模型直接用于更高分辨率，并观察到 FFC 模型比 regular/dilated baseline 退化更慢。它不是对任意分辨率、任意域和任意 mask 的尺度等变证明。

第四，论文解决的是**给定 mask 的图像补全**。它不负责识别要删除什么，不负责生成可审计的事实，也不判断某个编辑是否允许。SAM 一类分割模型、交互式涂抹工具或业务审核系统属于 LaMa 之外的产品层。

## 2. 论文信息、版本与开放状态

| 项目 | 内容 |
| --- | --- |
| 题名 | Resolution-robust Large Mask Inpainting with Fourier Convolutions |
| 作者 | Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, Victor Lempitsky |
| arXiv | `2109.07161v2` |
| v1 / v2 | 2021-09-15 / 2021-11-11 |
| 正式发表 | WACV 2022, pp. 2149-2159 |
| 主论文 | 11 pages, Fig. 1-6, Table 1-4 |
| 论文许可 | CC BY 4.0 |
| 官方代码 | [advimman/lama](https://github.com/advimman/lama) |
| 本文源码基线 | commit [`786f5936`](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7) |
| 代码许可 | Apache-2.0 |
| 官方 release | 仓库没有正式 tag/release |

论文作者来自 Samsung AI Center-Moscow 等机构。WACV 正式版本与 arXiv v2 的主方法一致，补充材料进一步给出完整指标、用户研究、mask 参数、Big LaMa 正反例、跨领域案例、感知损失权重、LaMa-Dilated 架构和推理时间。

代码状态需要单独说明。仓库当前仍可阅读和运行，但 README 的主要环境是 PyTorch `1.8.0`、torchvision `0.9.0`、PyTorch Lightning `1.2.9` 和 Hydra 时代的依赖组合。原始 Yandex Disk 权重与数据链接已有迁移或失效情况，社区通常通过 Google Drive、Hugging Face 镜像或封装项目取得模型。这不影响论文方法成立，却会显著增加 2026 年重新搭建原环境的成本。

本文固定到 commit 而不是漂移的 `main` 行号。后文提到 `ffc.py`、mask generator、loss、trainer、predict 和 refinement 时，都指向这一代码快照。源码静态阅读能回答“公开实现如何表达论文机制”，不能替代原训练数据、随机种子、硬件环境和 checkpoint 的完整实验复现。

## 3. 任务边界：Inpainting 不等于所有生成式编辑

给定完整图像 $\mathbf{x}\in[0,1]^{H\times W\times3}$ 和二值缺失 mask $\mathbf{m}\in\{0,1\}^{H\times W}$，本文统一采用代码中的约定：$m_{ij}=1$ 表示待修复区域，$m_{ij}=0$ 表示已知区域。生成器输入为 masked image 与 mask 的通道拼接：

$$
\mathbf{x}'=\operatorname{stack}\left(\mathbf{x}\odot(1-\mathbf m),\mathbf m\right),
$$

因此 $\mathbf{x}'$ 有 4 个通道。生成器预测完整图像：

$$
\hat{\mathbf x}=f_\theta(\mathbf{x}').
$$

产品输出通常还会把已知区域原样合回去：

$$
\mathbf{x}_{out}=\mathbf{x}\odot(1-\mathbf m)+\hat{\mathbf x}\odot\mathbf m.
$$

最后这个合成操作非常重要。模型即使对全图都输出 RGB，也不意味着产品应允许它改写 mask 外的像素。已知区域保留既减少不必要漂移，也让编辑合同更清晰。

这个任务与几个相邻概念不同：

| 任务 | 额外输入 | 主要目标 |
| --- | --- | --- |
| Image inpainting | 图像 + 内部 mask | 补全图像内部缺失区域 |
| Object removal | 图像 + 对象 mask | 删除指定对象并修复背景 |
| Outpainting | 图像 + 画布外扩区域 | 向原图边界外生成内容 |
| Text-guided editing | 图像 + 文本/区域 | 按语义指令修改内容 |
| Conditional generation | 文本、布局、边缘等 | 从条件生成整幅图像 |

LaMa 原生处理第一行；对象移除是它最常见的应用，但“对象 mask 从哪里来”不属于模型；outpainting 可以通过特殊 mask 形式尝试，补充材料也展示了案例，但它不是主训练协议；文本指令、风格控制和可控重绘更不是原论文能力。

## 4. 为什么大洞比小洞难得多

如果 mask 只是几像素宽的划痕，局部邻域通常已经包含足够信息。颜色插值、纹理复制或小感受野卷积就能得到合理结果。大洞改变了问题性质。

**可见证据距离变远。** 洞中心到最近已知像素可能有几十甚至几百像素。局部卷积必须经过很多层才能间接获得边界信息，实际梯度贡献又可能集中在近邻。

**结构约束跨越全图。** 被删除的人挡住了一段栏杆，正确补全需要在左右两侧栏杆之间延续间距、方向和相位。只从一侧复制纹理很容易造成格子错位。

**多解性增强。** 一张沙发后面的墙可以是空白、挂画或窗户。数据中只有一个 ground truth，但符合上下文的合理答案可能很多。逐像素 L1/L2 会倾向平均化，单张图的 LPIPS 也无法覆盖全部合理解。

**语义和几何耦合。** 建筑窗格不仅是周期纹理，还受透视收缩、遮挡边界和立面分区影响。模型需要同时处理低频布局与高频细节。

**高分辨率放大局部错误。** 训练时一个像素级的轻微纹理偏差，在 1536 分辨率上可能变成明显条带、重复块或接缝。模型的名义感受野不变时，相对于图像尺寸的覆盖比例还会下降。

因此，大 mask inpainting 的核心不只是提高生成质量，而是让模型在每个输出位置都能够有效利用远距离证据。LaMa 把这个需求概括为 high receptive field，并分别放进 generator、perceptual loss 和 mask distribution。

## 5. Fig. 1 Teaser 应该怎样读

Fig. 1 展示了人物移除、建筑遮挡和高分辨率修复。它最有说服力的部分是规律结构：窗户、百叶、立面边缘和重复图案跨越了较大的缺失区，输出仍能延续某种全局一致性。这正对应 FFC 的设计动机。

但 teaser 不能证明三件事。

第一，它不能证明所有输入都成功。补充材料专门展示了透视扭曲、复杂背景和虚焦场景的失败案例，说明作者并未把这些结果当成无条件能力。

第二，它不能证明语义真实性。被遮挡区域没有唯一可观察真值；模型只生成一个视觉上可接受的猜测。对新闻、医疗、司法取证或科学记录，视觉合理不能替代事实可靠。

第三，它不能单独证明 FFC 的因果贡献。真正的证据来自 Table 2 的架构消融、Table 3 的感知损失消融、Table 4 的 mask 训练消融，以及 Fig. 5/6 的跨分辨率比较。漂亮案例是问题入口，不是结论终点。

## 6. 输入输出合同：mask 约定比公式更重要

图像修复代码最常见的集成错误之一，是不同库对 mask 的黑白约定相反。有的项目令白色表示保留，有的令白色表示删除。LaMa 公开代码在预测阶段使用 `image * (1 - mask)`，也就是 mask 为 1 的区域会被清空，然后与 mask 拼成四通道输入。

工程上至少要固定以下合同：

```text
image: float32, shape [B, 3, H, W], normalized to configured range
mask:  float32, shape [B, 1, H, W], values in {0, 1}
mask == 1: region to synthesize
masked_image = image * (1 - mask)
generator_input = concat(masked_image, mask, dim=1)
predicted_image = generator(generator_input)
output = image * (1 - mask) + predicted_image * mask
```

这里还有几个隐藏条件。

- 输入宽高通常需要满足网络下采样倍率或配置中的 padding 规则。
- mask 边界是否抗锯齿、是否膨胀，会影响生成器看到的已知像素以及最终接缝。
- 如果 JPEG 压缩、alpha 通道或颜色空间处理发生在 mask 合成前后，边缘可能出现色带。
- 批处理时每张图的宽高、mask 比例和显存成本不同，不能只按图片张数估算容量。

论文公式关注模型，生产系统则必须把 mask 约定、归一化、padding、合成和输出编码一起版本化。

## 7. 理论感受野与有效感受野不是一回事

堆叠很多 $3\times3$ 卷积后，理论感受野会逐层扩大。若每一层 stride 为 1，增加一层可让边长多覆盖 2 个像素；下采样还会让后续卷积映射回输入时覆盖更大区域。从纸面计算看，较深 ResNet 很快就能“覆盖整张 256 图”。

问题在于，**理论上可连接**不等于**实际有显著影响**。有效感受野研究表明，输出对远处输入的梯度贡献往往集中在中心附近，分布近似高斯。网络可以通过很多层传递远距离信息，但这条路径长、衰减强，也容易被局部纹理捷径取代。

对 inpainting，洞内输出没有对应输入像素，真正有用的证据位于洞外。一个模型即便理论感受野跨过洞，如果优化过程中主要依赖邻近边界，也会在大洞中心失去方向。周期结构尤其暴露这个问题：局部窗口里每一块砖都相似，只有观察更远位置才能确定相位和整体排列。

LaMa 的策略不是无限加深网络，而是让一部分通道通过 Fourier transform 在单个 block 中建立全图信息交换。与此同时，感知损失也选用更大感受野的特征网络，训练 mask 又主动扩大“必须看远处”的样本比例。三处设计共同减少局部捷径。

## 8. FFC 的来源：LaMa 使用它，而不是发明它

Fast Fourier Convolution 最早作为通用视觉 backbone 模块提出。它把 feature channels 划分为 local 与 global 两组：local 分支保留常规卷积的空间局部性，global 分支用频域变换实现非局部更新，分支之间还允许信息交换。

LaMa 选择 FFC 有两个实际理由。

一是它不需要显式构建 $HW\times HW$ 的注意力矩阵。二维 FFT 的复杂度约为 $O(HW\log(HW))$，频域中的 $1\times1$ 卷积在所有频率位置共享参数。与全局 self-attention 相比，尤其在 2021 年硬件和分辨率条件下，这是一条较直接的全局信息路径。

二是频域表示天然暴露周期模式。规则窗格、栅栏和条纹会在频谱中形成相对集中的响应。模型并不会因此“自动懂建筑”，但它更容易在全局范围协调重复频率和相位相关信息。

必须保留的边界是：FFT 只是线性变换，语义能力仍来自数据、非线性网络与损失。高频噪声、透视变化和非平稳纹理不会因为进入频域就自动变简单。FFC 也不是严格的尺度等变模块；论文对跨分辨率的解释是机制假设加实验观察，而不是数学保证。

## 9. Fourier Unit：从空间特征到全图频率混合

设输入特征为 $\mathbf{x}\in\mathbb{R}^{B\times C\times H\times W}$。Fourier Unit 的核心流程可以写成：

$$
\mathbf{X}=\operatorname{rFFT2}(\mathbf{x}),
$$

其中实输入的频谱只需保留最后一维的一半频率。$\mathbf{X}$ 是复数。公开实现把实部与虚部展开到 channel 维：

$$
\mathbf{X}_{ri}=\operatorname{concat}(\Re(\mathbf X),\Im(\mathbf X)).
$$

然后应用可学习的 $1\times1$ convolution、BatchNorm 和 ReLU：

$$
\tilde{\mathbf X}_{ri}=\operatorname{ReLU}\left(
\operatorname{BN}\left(W_{1\times1}*\mathbf X_{ri}\right)
\right).
$$

最后把通道重新组装为复数并做逆变换：

$$
\mathbf{y}=\operatorname{irFFT2}(\tilde{\mathbf X}).
$$

为什么一个频域 $1\times1$ 卷积具有全图感受野？因为每个 Fourier coefficient 都由整个空间平面的像素共同决定。修改频率系数后再逆变换，输出任一空间位置都可能受输入全图影响。这条路径比堆叠局部卷积更短。

源码 [`saicinpainting/training/modules/ffc.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/modules/ffc.py) 中的 `FourierUnit` 具体处理了 `torch.fft.rfftn`、实虚部排列、卷积、归一化、激活和 `irfftn`。代码使用现代 `torch.fft` API；这只是 PyTorch 表达方式更新，不改变论文中的核心算法。

## 10. Local / Global 双分支与四条信息流

FFC 不把所有 channel 都送进频域。设输入按照比例 $\alpha_{in}$ 分成 local 与 global：

$$
\mathbf{x}=(\mathbf{x}_l,\mathbf{x}_g).
$$

输出也按照 $\alpha_{out}$ 分成两组。四条变换路径为：

$$
\mathbf y_l=f_{l\to l}(\mathbf x_l)+f_{g\to l}(\mathbf x_g),
$$

$$
\mathbf y_g=f_{l\to g}(\mathbf x_l)+f_{g\to g}(\mathbf x_g).
$$

$f_{l\to l}$ 是常规局部卷积；$f_{g\to g}$ 使用 Spectral Transform；另外两条路径负责把局部信息注入全局分支，并把全局上下文反馈到局部分支。这样做避免两个极端：全用局部卷积会缺少远程通信；全用频域又可能削弱位置敏感的细节建模。

官方 `ffc_resnet_075.yaml` 把 residual blocks 的 global 输入/输出通道比例设置为 `0.75`。这意味着 block 中大部分通道参与全局路径，但仍保留局部分支。这个 `0.75` 是论文与配置验证过的选择，不是所有任务的通用常数。

`FFC` 类负责计算 local/global channel 数与四条路径；`FFC_BN_ACT` 在两路输出上分别加归一化和激活；`FFCResnetBlock` 再把它们组织为残差块。阅读这些类比只看论文框图更容易理解一个关键事实：FFC 输出在网络内部仍然是两组 tensor，后续 block 持续维护 local/global 结构，而不是每层都立即压回普通 tensor。

## 11. Spectral Transform 与 LFU

Spectral Transform 是 global-to-global 路径的主体。它通常先用 $1\times1$ 卷积调整通道，经过 Fourier Unit，再与捷径或可选局部频率单元（LFU）结果相加，最后再用 $1\times1$ 卷积投影。

LFU 的出发点是补充局部频率信息：把 feature map 分块后重排，让 Fourier Unit 在较小空间范围处理局部重复模式。理论上它可以同时利用全局和局部频率结构。但 LaMa 官方的主要 FFC generator 配置明确写着：

```yaml
resnet_conv_kwargs:
  ratio_gin: 0.75
  ratio_gout: 0.75
  enable_lfu: false
```

因此，介绍 FFC 原始模块时可以解释 LFU，描述 LaMa 主配置时却不能说“模型依赖 LFU”。关闭 LFU 是公开配置事实。LaMa 的主要增益来自 global Fourier Unit、双分支 residual blocks、损失与 mask 训练的组合，而不是必须启用局部频率单元。

代码还支持 `spatial_scale_factor`、spectral positional encoding、SE 等可选项。它们体现实现的通用性，不代表论文主模型全部开启。精读源码时要避免“代码里存在一个参数，所以论文一定使用了它”的反向推断。

## 12. Fig. 2：单阶段 ResNet-like Generator

![LaMa generator、FFC 与 Spectral Transform](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig02-architecture-ffc.webp)

*左侧是 LaMa 训练与生成主链，中间是 FFC 四路信息流，右侧是 Spectral Transform。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 2, CC BY 4.0.*

Fig. 2 把方法压缩为三层视图。最左侧是完整生成器与损失；中间放大 FFC；最右侧再放大 Spectral Transform。标准 LaMa-Fourier 生成器采用：

1. 输入 stem 与 3 次 downsampling；
2. 9 个 FFC residual blocks；
3. 3 次 upsampling；
4. 输出 3 通道 RGB。

标准模型约 27M 参数。补充材料中的 Big LaMa 把 residual blocks 增到 18 个，参数约 51M。Table 2 还比较了 shallow Fourier、deep regular 和 dilated 变体，用来分离“只是参数更多”“只是网络更深”和“确实需要更大有效感受野”这几种解释。

这里的单阶段很有工程意义。它让预测路径保持简单：无需缓存边缘图、语义图或多阶段中间结果；模型可以全卷积地直接作用于不同尺寸；失败排查也更容易把问题定位到输入 mask、生成器或后处理。但单阶段也意味着所有结构推断都隐含在 feature 中，不能像显式几何 pipeline 那样单独检查中间语义是否正确。

## 13. 为什么 FFC 特别适合周期结构

周期结构是 LaMa 最有辨识度的实验。设一维简化信号为 $x[n]=\cos(2\pi f n+\phi)$。局部窗口也许只能看到半个周期，难以确定 $f$ 和 $\phi$；完整频谱会在对应频率附近产生集中能量，给网络一条直接比较远处重复模式的路径。

在二维图片中，窗格、砖墙、百叶和链式栅栏具有方向、间距和相位。洞两侧出现的纹理局部外观相近，但正确连接需要知道：

- 重复单元的主方向；
- 水平和垂直频率；
- 透视导致的非均匀间距；
- 洞两侧相位如何对齐；
- 哪些边缘属于前景遮挡，哪些属于背景结构。

FFC 最直接帮助前三项和部分相位协调。透视变化、遮挡层次和语义归属仍要由网络从数据中学习。这也解释了为何补充失败案例中，规则但强透视的网格仍可能扭曲：频率并非全图平稳，单一周期在不同位置发生变化。

![周期结构修复对比](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig04-periodic-structure-comparison.webp)

*窗户与链式栅栏上的对比。LaMa-Fourier 用较少参数保持了更连续的重复结构。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 4, CC BY 4.0.*

图中 LaMa-Regular 和更深的 regular baseline 在大洞内部出现模糊、断裂或无规则纹理；LaMa-Fourier 更好地延续窗格和栅栏。CoModGAN、MADF 的部分结果也有局部结构，但参数量更高且平均指标不一定更稳。这个图支持“FFC 对重复结构有效”，不能支持“FFC 在所有语义对象上都优于所有生成模型”。

## 14. HRF Perceptual Loss：监督网络也必须看得远

只改生成器还不够。如果训练损失主要比较局部纹理，生成器仍可能学到局部最优。LaMa 使用 High Receptive Field Perceptual Loss（HRFPL），其核心是用具有较大感受野、在语义分割任务上训练的网络提取多层 feature，再比较预测图与 ground truth。

设固定特征网络第 $k$ 层输出为 $\phi_k(\cdot)$，一个概念化写法是：

$$
\mathcal L_{\mathrm{HRFPL}}(\hat{\mathbf x},\mathbf x)
=\sum_k w_k\left\|\phi_k(\hat{\mathbf x})-\phi_k(\mathbf x)\right\|_2^2.
$$

实际实现会考虑输入归一化、各层尺度和 mask 相关处理。重要的不是某个符号，而是 feature extractor 的选择：ResNet50 后层使用 dilation 扩大感受野，并在 ADE20K 语义分割 pretext task 上训练。相比 ImageNet 分类网络，分割模型保留更密集的空间语义；相比普通 ResNet，dilation 让高层 feature 在不继续降低分辨率的情况下覆盖更广区域。

为什么这对 inpainting 有用？生成器可能补出局部锐利、但整体不连贯的窗格；普通像素或小感受野特征会奖励每一小块“像窗户”，HRF feature 更可能感知整段立面是否形成一致结构。换句话说，损失函数为全局一致性提供了训练信号，而 FFC 为生成器提供了实现这项约束的路径。

补充材料说明感知损失使用所有四个 residual stages 的输出。不同 backbone 的最优权重经过单独搜索：HRFPL 的最终权重为 30，classification ResNet 变体通常为 1，VGG19 为 0.1。数值差异来自特征尺度，不能据此说 HRFPL “重要 30 倍”。

![感知损失权重](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table05-perceptual-loss-weights.webp)

*补充材料给出的各感知损失最优权重。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 5, CC BY 4.0.*

## 15. Segmentation Backbone：大感受野之外还需要什么

论文没有只比较 VGG 与 ResNet 名称，而是把 HRFPL 拆成两个可检验因素：**dilation 是否有帮助，semantic segmentation pretext 是否有帮助。** Table 3 的四个主要变体是：

- ResNet50，segmentation pretext，带 dilation，即 HRFPL；
- ResNet50，classification pretext，带 dilation；
- ResNet50，classification pretext，不带 dilation；
- VGG19，classification pretext。

主表结果如下：

![感知损失消融](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-table03-perceptual-loss-ablation.webp)

*在 segmentation masks 上比较不同 perceptual feature extractor。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Table 3, CC BY 4.0.*

| 感知损失 | Backbone | Pretext | Dilation | FID | LPIPS |
| --- | --- | --- | --- | ---: | ---: |
| HRFPL | RN50 | Segmentation | 是 | **5.69** | **0.059** |
| ClfPL | RN50 | Classification | 是 | 5.87 | 0.059 |
| ClfPL | RN50 | Classification | 否 | 6.00 | 0.061 |
| ClfPL | VGG19 | Classification | 否 | 6.29 | 0.063 |
| PL baseline | - | - | - | 6.46 | 0.065 |

从第一、二行看，固定 dilation 后，segmentation pretext 把 FID 从 5.87 改善到 5.69，LPIPS 持平。从第二、三行看，classification ResNet 加 dilation 把 FID 从 6.00 改善到 5.87。两个因素都有贡献，但幅度不是数量级变化。

这组消融的正确读法是：语义密集预训练和大感受野 feature 对该设置有互补收益。它不能证明 ADE20K segmentation backbone 是所有 inpainting loss 的唯一最优选择，也没有覆盖现代自监督视觉模型、CLIP、DINO 或扩散模型 feature。2021 年的设计在当时合理，今天迁移到新任务仍应重新验证。

源码 [`training/losses/perceptual.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/perceptual.py) 中的 `ResNetPL` 封装固定特征网络与 feature distance。训练时这个网络不随生成器共同学习。否则 loss feature 自身会追随生成器漂移，无法稳定充当度量空间。

## 16. Patch Adversarial Loss：只在需要生成的区域追究真假

感知损失强调结构和语义，但它往往不能独立生成足够锐利的局部纹理。LaMa 同时使用 PatchGAN 风格判别器。判别器输出的不是单个全图标量，而是一张空间 score map；每个位置对应输入图的一块 receptive field。

论文使用 non-saturating adversarial objective。用 $D_\xi$ 表示判别器，概念上生成器项为：

$$
\mathcal L_G^{adv}=-\mathbb E\left[\log\sigma(D_\xi(\hat{\mathbf x}))\right],
$$

判别器项为：

$$
\mathcal L_D^{adv}
=-\mathbb E[\log\sigma(D_\xi(\mathbf x))]
-\mathbb E[\log(1-\sigma(D_\xi(\hat{\mathbf x})))] .
$$

LaMa 的关键细节是 patch label 与 mask 对齐。生成图中与 mask 相交的 patch 是模型真正合成的区域，应作为 fake 监督；已知区域在最终输出中本来就可直接复制，不应让判别器主要学习区分这些位置。实现会把 mask 缩放到 discriminator feature map 的空间尺度，并用它控制损失区域。

这比“整幅生成图全部判 fake”更符合任务合同。否则 discriminator 可以在已知区域寻找微小重建差异，生成器被迫浪费容量重建本应原样保留的像素。反过来，只监督 mask 内部也不能忽略边界，因为判别器 patch 的 receptive field 会跨过 mask 边缘，接缝仍会进入真假判断。

补充材料 Listing 2 给出判别器结构：多层 $4\times4$ convolution，stride 主要为 2，LeakyReLU 和部分 BatchNorm，最终输出单通道 patch logits。用可复制形式概括如下：

```python
NLayerDiscriminator(
    Conv2d(3, 64, kernel_size=4, stride=2, padding=2),
    LeakyReLU(0.2),
    Conv2d(64, 128, kernel_size=4, stride=2, padding=2),
    BatchNorm2d(128),
    LeakyReLU(0.2),
    Conv2d(128, 256, kernel_size=4, stride=2, padding=2),
    BatchNorm2d(256),
    LeakyReLU(0.2),
    Conv2d(256, 512, kernel_size=4, stride=1, padding=2),
    BatchNorm2d(512),
    LeakyReLU(0.2),
    Conv2d(512, 1, kernel_size=4, stride=1, padding=2),
)
```

这段结构是论文补充材料的语义化转录，不是复制当前仓库构造器的逐行代码。实际源码应以固定 commit 中的 discriminator 配置和工厂函数为准。

## 17. Feature Matching 与 R1：让 GAN 训练不只追逐判别边界

GAN 对抗项能增加锐度，也容易产生不稳定、棋盘纹理或局部高频 artifact。LaMa 加入两种稳定器。

**Feature matching loss** 比较真实图和生成图在 discriminator 中间层的 activation。若第 $k$ 层为 $D_\xi^{(k)}$，可写为：

$$
\mathcal L_{FM}
=\sum_k \left\|D_\xi^{(k)}(\hat{\mathbf x})-D_\xi^{(k)}(\mathbf x)\right\|_1.
$$

它不只要求最终真假 logit 过关，还要求中间多尺度统计接近真实图。对纹理和边界，这是比单个 adversarial score 更平滑的监督。

**R1 regularization** 在真实样本上惩罚判别器输入梯度：

$$
R_1=\mathbb E_{\mathbf x}\left[\left\|\nabla_{\mathbf x}D_\xi(\mathbf x)\right\|_2^2\right].
$$

它限制判别器在真实数据附近过度尖锐，减少 generator 与 discriminator 相互放大的震荡。代码 [`training/losses/adversarial.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/adversarial.py) 中的 `NonSaturatingWithR1` 负责这部分；[`feature_matching.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/feature_matching.py) 则计算中间 feature matching。

在产品迁移中，不能只复制 generator 权重或结构，然后任意换掉训练损失，仍期待论文同样行为。LaMa 的模型质量来自训练目标组合；HRFPL 提供结构度量，adversarial loss 提供局部真实感，feature matching 与 R1 抑制不稳定。

## 18. Final Loss：四种目标如何配合

论文把总目标写成：

$$
\mathcal L_{final}
=\kappa\mathcal L_{Adv}
+\alpha\mathcal L_{HRFPL}
+\beta\mathcal L_{DiscPL}
+\gamma R_1,
$$

其中最终选择为：

$$
\kappa=10,\qquad
\alpha=30,\qquad
\beta=100,\qquad
\gamma=0.001.
$$

论文中的 `DiscPL` 对应 discriminator feature matching/perceptual 路径。权重经 validation 上的 coordinate-wise beam search 选择。权重绝对值不能直接比较“哪项更重要”，因为不同 loss 的原始数值尺度不同。`100` 不意味着 feature matching 在概念上比 R1 重要十万倍。

更合理的工程检查是记录每项 loss 的实际量级、梯度范数和训练趋势：

| 监控项 | 异常信号 |
| --- | --- |
| HRFPL | 持续下降但视觉仍模糊，可能语义 feature 忽略高频 |
| Generator adversarial | 剧烈震荡，可能判别器过强或数据/mask 分布不稳 |
| Discriminator loss | 很快趋近极端，可能真假任务过于容易 |
| Feature matching | 居高不下，可能生成纹理统计与真实域偏离 |
| R1 | 梯度峰值频繁，可能判别器局部过尖 |

当前仓库的 [`training/trainers/default.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/trainers/default.py) 把 generator、discriminator、perceptual 和 feature matching 连接到训练 step。Hydra 配置决定具体权重。理解这一点有助于排查“代码类存在，但当前实验配置是否启用”的问题。

## 19. Fig. 3：大掩码不是面积更大这么简单

![LaMa mask policies](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig03-mask-policies.webp)

*DeepFillv2 风格、窄 irregular、宽 irregular 与 box masks 的视觉差异。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 3, CC BY 4.0.*

LaMa 把训练 mask 视为方法组成，而不是数据加载器中的无关随机增强。图中可以看到四类难度来源：

1. 窄线条虽然可能覆盖较大面积，但每个缺失像素离已知区域很近；
2. 宽 irregular strokes 会形成厚重的连通区域；
3. 大矩形让洞中心与边界距离显著增加；
4. 多个 shape 的叠加会产生复杂边界和不规则拓扑。

训练时 irregular polygonal chains 与 rectangles 以大致 50/50 的策略混合。相比只模拟细笔刷，LaMa 的 mask generator 更常创建长路径、较大 brush width 和大 box。这样，生成器不能只学习从最近边缘延伸纹理，而必须使用更远上下文。

这个设计可以看成一种 task curriculum，但它不是从易到难的时间课程，而是通过采样分布直接提高困难样本占比。风险也很明确：如果训练 mask 远大于产品实际 mask，模型可能牺牲窄洞细节；如果产品 mask 具有完全不同的边界统计，例如毛发级 alpha、文字轮廓或医学器官边缘，论文参数未必最优。

## 20. Mask Width：比 masked area 更接近“需要看多远”

论文和补充材料不仅统计 mask area，还使用 mask width。对每个缺失像素 $p$，计算它到最近已知像素集合 $\Omega_{known}$ 的欧氏距离：

$$
d(p)=\min_{q\in\Omega_{known}}\|p-q\|_2.
$$

mask width 可理解为对所有缺失像素距离的平均：

$$
w(\mathbf m)=\frac{1}{|\Omega_{mask}|}
\sum_{p\in\Omega_{mask}} d(p).
$$

这比面积更能反映修复对远程上下文的依赖。举例：一条贯穿全图但只有 3 像素宽的曲线可能覆盖很多像素，平均距离仍很小；一个居中的正方形即使面积相同，内部像素离边界更远，困难明显更高。

mask width 仍不是完美难度指标。它没有考虑洞外纹理是否规则、被遮挡对象是否语义复杂、洞是否跨越关键透视线，也没有区分多个离散洞与单个连通洞。但作为训练 mask 和测试集的几何统计，它比单一 area ratio 更有解释力。

## 21. 源码中的 Mask Generator

固定 commit 的 [`saicinpainting/training/data/masks.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/data/masks.py) 将论文策略拆为可组合生成器：

- `RandomIrregularMaskGenerator`：随机起点、角度、长度、brush width 和多段折线；
- `RandomRectangleMaskGenerator`：随机数量、尺寸与 margin 的 box；
- `RandomSegmentationMaskGenerator`：基于对象 silhouette 构造更语义化的测试 mask；
- `MixedMaskGenerator`：按概率组合多个 generator。

补充材料 Listing 1 的逻辑可压缩为以下伪代码：

```python
def mixed_mask(height, width, rng):
    if rng.random() < 0.5:
        mask = random_irregular_mask(
            height, width,
            max_angle=4,
            max_len=200,
            max_width=100,
            max_times=5,
        )
    else:
        mask = random_box_mask(
            height, width,
            margin=10,
            bbox_min_size=30,
            bbox_max_size=150,
            max_times=4,
        )
    return mask.astype("float32")
```

参数随图像分辨率和 narrow/medium/wide 设置变化，不能把这段伪代码当成完整官方实现。真正复现应读取 Hydra data config，并记录随机种子、mask generator 版本、各 generator 概率和 augmentation 顺序。

![随机 mask 参数](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table04-mask-generator-parameters.webp)

*256 与 512 分辨率下 narrow、medium、wide、train 的 irregular/box 参数。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 4, CC BY 4.0.*

源码与论文之间还有一个容易忽略的区别：代码是一套通用训练框架，包含更多 mask 配置和数据路径；论文表格只报告特定实验组合。看到 `masks.py` 支持某种 generator，不代表主结果使用了所有选项。

## 22. Supplement Mask Statistics：训练分布到底有多激进

![256 分辨率 mask 统计](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig04-mask-statistics-256.webp)

*256 分辨率下 DeepFillv2、LaMa training 和 narrow/medium/wide test masks 的面积与 width 分布。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 4, CC BY 4.0.*

256 图的统计最直接揭示论文 mask 策略。DeepFillv2 风格训练 mask 的面积分布相对集中，width 也偏小；LaMa training mask 覆盖更宽的面积和距离范围。测试集 narrow、medium、wide 则刻意形成不同难度层级。

![512 分辨率 mask 统计](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig05-mask-statistics-512.webp)

*512 分辨率下 Places narrow、medium、wide 与 segmentation masks 的统计。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 5, CC BY 4.0.*

512 图的 segmentation masks 与随机 irregular masks 分布不同。对象 silhouette 可能具有细长部件、孔洞和复杂轮廓；它们更接近真实对象移除，但又受检测器类别和筛选流程影响。补充材料从 Places 的结构场景中构造了这套评测集，并使用 Detectron2 前景对象 silhouette 帮助生成 mask。

![Segmentation mask examples](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig03-segmentation-masks.webp)

*Places segmentation test set 的对象区域、目标 hole 与背景限制示例。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 3, CC BY 4.0.*

这套数据构造比随机笔刷更接近对象删除，却仍不等于真实产品流量。产品中常见的手工 mask 会有漏分、过分、软边、细发丝和触边对象；自动分割又会携带模型偏差。上线前需要在真实 mask 日志上重新统计 area、width、连通分量、边界复杂度和对象类别。

## 23. 训练设置：数字背后的计算合同

标准模型的主要训练设置是：

| 项目 | 标准 LaMa |
| --- | --- |
| 输入 crop | $256\times256$ |
| 原始图近似尺寸 | $512\times512$ |
| Generator blocks | 9 |
| LaMa-Fourier 参数量 | 27M |
| 训练迭代 | 1,000,000 |
| Batch size | 30 |
| Generator optimizer | Adam, LR `0.001` |
| Discriminator optimizer | Adam, LR `0.0001` |
| FFC global ratio | `0.75` |
| LFU | 主配置关闭 |

“1M iterations”不能脱离 batch、crop 和数据采样解释。每个 iteration 看到 30 个 crop，理论样本曝光量约 3000 万，但图片和随机 mask 会重复组合；这不等于 3000 万独立图像。数据增强、分布式 sampler、validation checkpoint 策略也会影响最终模型。

生成器与判别器不同学习率是典型 GAN 稳定性选择。重新训练时，如果改成更大 batch、混合精度、不同 Adam beta 或现代编译器，loss 动态可能变化，不能只保留 LR 数字。

官方训练入口 [`bin/train.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/bin/train.py) 通过 Hydra 组合模型、data、loss 和 trainer 配置。论文所说“同一方法”实际上对应一组配置文件与代码版本。复现实验必须存档 resolved config，而不只是保存命令行。

## 24. 数据集与评测协议

论文主要使用 Places 和 CelebA-HQ。

**Places** 提供大量室内外场景，适合评估建筑、街景、房间和自然背景。标准训练使用 Places-Standard 的高分辨率图像。最终评测包含 narrow、medium、wide random masks，以及专门构造的 segmentation masks。

**CelebA-HQ** 主要是人脸。人脸结构先验很强，局部错误容易被察觉；但数据内容单一，无法代表一般场景。论文按 DeepFill 相关 split 构造训练、验证和测试，并在 narrow、medium、wide masks 上报告结果。

主文 Table 1 汇总 Places 512 与 CelebA-HQ 256 的核心结果；补充 Table 2/3 进一步把“所有样本”与 masked area 40%-50% 的困难子集分开。这个细分很有价值，因为平均指标可能被大量小洞稀释。

![Places 完整指标](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table02-places-full-metrics.webp)

*Places 上 narrow/medium/wide/segmentation masks 的完整 FID 与 LPIPS。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 2, CC BY 4.0.*

![CelebA-HQ 完整指标](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table03-celebahq-full-metrics.webp)

*CelebA-HQ 上 40%-50% masked 困难子集与全部样本指标。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 3, CC BY 4.0.*

评测集图片与 mask 的配对数也要分清。Places random-mask test sets 各包含 30,000 对，segmentation set 约 4,000 对；validation 另有 2,000 对用于 checkpoint 选择。不能把 image-mask pair 数量误写为独立原始图片数量。

## 25. FID、LPIPS 与用户偏好各自回答什么

FID 比较真实图集合与生成图集合在 Inception feature 空间中的均值和协方差：

$$
\operatorname{FID}
=\|\mu_r-\mu_g\|_2^2
+\operatorname{Tr}\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right).
$$

它是集合级分布指标。FID 低说明生成集合在某种 feature 统计上更接近真实集合，不告诉我们某一张图的洞是否与上下文语义一致，也不告诉我们 mask 边界是否准确。

LPIPS 比较两张图在预训练网络多层 feature 上的距离。它是成对指标，能比像素 L2 更容忍小位移和感知等价变化；但 inpainting 有多解，一个合理生成若与唯一 ground truth 内容不同，也可能得到较差 LPIPS。

论文补充用户研究两个协议：

- **side-by-side relative preference (RP)**：在 LaMa 与某个 baseline 的两个结果中选择更真实者；
- **spot the mask accuracy (Acc)**：只看修复后的图，点击最可疑区域，若定位到修复区域则记为识别成功。

![Side-by-side 用户研究界面](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig01-user-study-side-by-side.webp)

*原始界面左、右为两个模型结果，中间显示 masked input。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 1, CC BY 4.0.*

![Spot-the-mask 用户研究界面](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig02-user-study-spot-mask.webp)

*评估者只看到修复图并点击最可疑位置。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 2, CC BY 4.0.*

RP 越高越好，Acc 越低越好。Acc 低表示用户更难找到修复位置，不是分类准确率差。论文通过 bootstrap 报告标准差，并限制每位评估者页面数以减轻适应效应。用户研究弥补了自动指标的一部分盲区，但仍受样本选择、显示尺寸、标注人群和任务说明影响。

## 26. Table 1 主结果：LaMa 赢在哪里

![LaMa 主结果](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-table01-main-results.webp)

*Places 与 CelebA-HQ 的主结果。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Table 1, CC BY 4.0.*

LaMa-Fourier 的核心数值为：

| 数据集 / Mask | FID | LPIPS |
| --- | ---: | ---: |
| Places narrow | 0.63 | 0.090 |
| Places wide | 2.21 | 0.135 |
| Places segmentation | 5.35 | 0.058 |
| CelebA-HQ narrow | 7.26 | 0.085 |
| CelebA-HQ wide | 6.96 | 0.098 |

在 Places wide masks 上，CoModGAN 的 FID `1.82` 优于 LaMa 的 `2.21`，但 LPIPS `0.147` 差于 LaMa 的 `0.135`；这提醒我们不能把“consistent outperform”简化为每个单元格都第一。论文更强的整体主张是：LaMa 在不同 mask 与两个数据集上保持较稳定的指标，同时只有 27M 参数；CoModGAN 约 109M，MADF 约 85M。

参数量不是速度的充分代理。FFT 有额外算子和内存访问，论文明确指出 LaMa-Fourier 平均比 LaMa-Regular 慢约 20%，尽管参数少约 40%。因此，“更小”不等于“必然更快”。

主表也存在比较边界。baseline 使用公开预训练模型，并尽量按各自推荐设置评测，但训练数据、mask 分布和输入尺寸未必完全一致。主表证明 LaMa 在这套统一评测中很强，不是对所有后续方法和所有产品 mask 的永久排名。

## 27. 用户研究结果：更难被发现不等于事实正确

![LaMa 用户研究结果](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-table01-user-study.webp)

*Places 512 上 narrow/wide masks 的相对偏好与 spot-the-mask 准确率。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Table 1, CC BY 4.0.*

LaMa-Fourier 作为 RP 基准被定义为 50。其 spot-the-mask Acc 在 narrow masks 为约 `34±1.7`，wide masks 为约 `54±1.7`。宽洞更容易被定位，符合任务难度直觉。CoModGAN 在 wide masks 的 RP 接近或略高于基准，但在 narrow masks 不占优势；MADF 在 narrow masks 接近，wide masks 明显下降。

三点值得注意。

第一，表格中的 LaMa RP=50 是比较基准定义，不是“50% 的绝对用户喜欢率”。每个 baseline 与 LaMa 成对比较，跨行不一定形成全局传递排序。

第二，spot-the-mask 衡量视觉可检测性。一个模型可能补出不存在的窗户，用户却难以察觉；它在这个指标上很好，但事实层面仍错。

第三，用户研究使用固定分辨率、页面和任务说明。手机小屏、压缩后图片或专业摄影师评审可能得到不同结果。产品验收需要把“用户是否能看出编辑”“编辑是否符合意图”“是否保留身份/品牌/文字”等维度拆开。

## 28. Table 2 架构消融：不是单纯参数规模效应

![LaMa 架构消融](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-table02-architecture-ablation.webp)

*Fourier、dilated、regular、shallow/deep 生成器对比。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Table 2, CC BY 4.0.*

| 模型 | 卷积 | 参数量 | Blocks | Narrow FID / LPIPS | Wide FID / LPIPS |
| --- | --- | ---: | ---: | --- | --- |
| Base | Fourier | 27M | 9 | 0.63 / 0.090 | **2.21 / 0.135** |
| Base | Dilated | 46M | 9 | 0.66 / **0.089** | 2.30 / 0.136 |
| Base | Regular | 46M | 9 | **0.60 / 0.089** | 3.51 / 0.139 |
| Shallow | Fourier | 19M | 6 | 0.72 / 0.094 | 2.31 / 0.138 |
| Deep | Regular | 74M | 15 | 0.63 / 0.090 | 2.62 / 0.137 |

在 narrow masks 上，Base Regular 的 FID `0.60` 略好于 Fourier `0.63`；到了 wide masks，Regular 退化到 `3.51`，Fourier 保持 `2.21`。这正是论文要强调的难度分界：局部模型对小洞足够，洞变宽后全局路径的价值才明显。

Deep Regular 有 74M 参数、15 blocks，wide FID 改善到 `2.62`，仍不及 27M Fourier 的 `2.21`。因此性能不是“加参数就能完全解释”。Dilated 46M 达到 `2.30`，说明扩大感受野本身确实关键；Fourier 则在参数效率和跨分辨率上更有优势。

这个消融也反驳一种过度宣传：Fourier 不是每个场景都绝对更好。窄洞指标里 regular/dilated 可以相当甚至略优。LaMa 的优势集中在大 mask、复杂结构与更高测试分辨率，而不是所有像素补全条件下的统一支配。

## 29. Table 4 Mask 消融：困难训练样本能否迁移到其他模型

![宽 mask 训练消融](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-table04-mask-training-ablation.webp)

*同一方法分别使用 narrow 与 wide mask 训练后的结果。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Table 4, CC BY 4.0.*

Table 4 不只比较 LaMa，而是把 wide-mask training 应用于 LaMa-Regular、DeepFill v2、EdgeConnect 和 RegionWise。核心数字如下：

| 方法 | 训练 mask | Narrow FID / LPIPS | Wide FID / LPIPS |
| --- | --- | --- | --- |
| LaMa-Regular | Narrow | 0.68 / 0.091 | 5.41 / 0.144 |
| LaMa-Regular | Wide | **0.60 / 0.089** | **3.51 / 0.139** |
| DeepFill v2 | Narrow | 1.06 / 0.104 | 5.20 / 0.155 |
| DeepFill v2 | Wide | 1.35 / 0.107 | **4.34 / 0.148** |
| EdgeConnect | Narrow | 1.33 / 0.111 | 8.37 / 0.160 |
| EdgeConnect | Wide | 2.78 / 0.141 | **7.94 / 0.160** |
| RegionWise | Narrow | 0.90 / 0.102 | 4.75 / 0.149 |
| RegionWise | Wide | **0.74 / 0.095** | **3.56 / 0.144** |

LaMa-Regular 和 RegionWise 在 wide-mask training 后，不仅宽洞变好，窄洞也改善；这说明更激进的 mask 分布可以减少模型对局部捷径的依赖。DeepFill v2、EdgeConnect 则出现明显取舍：宽洞改善，窄洞 FID/LPIPS 变差。训练数据策略的收益依赖模型容量、结构先验和 loss，不是免费增强。

这张表支持两个结论。其一，LaMa 的 mask 策略具有一定可迁移性，至少不只对 FFC 有效。其二，生成器必须有能力利用困难样本；如果结构仍偏向局部或中间任务形成瓶颈，扩大 mask 可能只是让训练更难。

在现代工程中，可以把 mask 分布作为显式超参数族，而不是单个固定生成器。建议按生产日志划分窄洞、宽洞、触边洞、多连通洞、细结构洞和语义对象洞，分别报告指标。只有总体 FID 时，很难知道一次 mask 策略修改改善了哪类请求，又损害了哪类请求。

## 30. Fig. 5：从 256 训练到 1536 测试

![LaMa 跨分辨率修复](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig05-high-resolution-transfer.webp)

*标准模型用 256 crop 训练，在 640×512 与 1920×1536 输入上的直接前向结果。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 5, CC BY 4.0.*

Fig. 5 的两行使用相同场景与 mask，但分辨率不同。所有 LaMa 变体都在 $256\times256$ crop 上训练；MADF 则直接以 $512\times512$ 训练。测试时不是切 tile 再拼接，而是把全图一次送入全卷积生成器。

在 640×512 行，regular、Fourier 和 Big LaMa 差异相对有限。到了 1920×1536，LaMa-Regular 出现大面积颜色条带或模糊竖带，Big Regular 仍有明显结构错误；LaMa-Fourier 更稳定地恢复墙面和室内对象。MADF 也出现可见 artifact。

为什么分辨率提高会放大 regular convolution 的问题？标准模型在低分辨率 feature map 上虽然有较大理论感受野，但输入尺寸扩大三倍后，同样的卷积路径只覆盖更小的相对区域。洞中心需要跨越更多 feature positions 才能接触边界。FFC 的全局分支每次仍对整个当前 feature map 做 Fourier transform，因此相对覆盖不会以同样方式收缩。

不过，输入越大，FFT 的频率网格也改变，模型在训练中从未见过这些尺寸。它能工作，是全卷积结构、频率参数共享、低频结构保留与训练先验共同产生的经验泛化。不能把它解释成严格的连续尺度不变性。

## 31. Fig. 6：分辨率提升时质量如何退化

![FID 与 LPIPS 随分辨率变化](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-fig06-resolution-metrics.webp)

*不同模型的 FID/LPIPS 随测试分辨率变化。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Fig. 6, CC BY 4.0.*

Fig. 6 把定性观察变成曲线。随着分辨率从低到高增加，所有模型的 FID 和 LPIPS 都变差；真正的差异是退化斜率。LaMa-Regular、LaMa-Dilated 和 Big Regular 在高分辨率处增长更快，LaMa-Fourier 与 Big LaMa-Fourier 更平缓。

这张图最重要的事实不是“FFC 指标不变”，因为它同样退化；而是“FFC 的退化相对更小”。因此 `resolution-robust` 更准确的中文是“对分辨率变化更稳健”，不是“分辨率无关”。

图中的 Big LaMa-Fourier 带星号，仅作参考，因为它的训练条件不同：更多 block、更大数据集和更大 batch。不能把 Big 与标准 baseline 的差距全部归因于 FFC。标准 LaMa-Fourier 与 LaMa-Regular/Dilated 的比较更适合判断结构贡献。

曲线还缺少产品常见的几个维度：不同 mask area/width 在每个分辨率下的分层、GPU 显存峰值、实际 wall-clock、频域算子在不同硬件上的效率，以及极端长宽比。跨分辨率上线应补做这些测试，不能只看方形图上的 FID。

## 32. 跨分辨率解释的边界

论文给出三种直觉：FFC 的 image-wide receptive field、尺度变化后低频保留、频域 $1\times1$ convolution 的某种尺度共享。它们解释了观察，但都不是无限泛化的保证。

**全图感受野不等于全图语义正确。** 模型能看到远处，不代表它知道哪些远处证据应该对应当前洞。重复建筑立面可能存在多个相位选择，模型仍会猜错。

**低频稳定不等于高频可恢复。** 放大图像后，纹理细节和压缩噪声的频率分布会变化。FFC 可以维持布局，输出却可能缺少真实传感器细节。

**全卷积不等于任意尺寸都可运行。** 下采样倍率、padding、FFT 内存、BatchNorm 统计和 GPU workspace 都可能限制尺寸。输入 8K 图即使数学上可前向，也可能超显存或非常慢。

**同域放大不等于跨域。** Places 训练图放大到高分辨率，与 MRI、遥感、漫画或文档图片是不同泛化轴。补充材料展示了一些跨域正例，但没有给出系统性跨域 benchmark。

**指标平缓不等于视觉无缺陷。** FID 是集合统计，局部文本、标志、人脸身份和几何线条错误可能被平均掉。高分辨率 QA 必须按局部 crop 放大检查。

因此，工程文档应把“支持更高分辨率”写成经过验证的尺寸区间、显存上限、典型延迟和失败类型，而不是一个没有上界的布尔能力。

## 33. Big LaMa：扩容了什么

Big LaMa 是补充材料和 teaser 中的 51M 模型。它相对标准 LaMa-Fourier 有三项主要变化：

| 维度 | 标准 LaMa | Big LaMa |
| --- | ---: | ---: |
| FFC residual blocks | 9 | 18 |
| 参数量 | 27M | 51M |
| 训练图片来源 | Places-Standard | Places-Challenge 子集 |
| 图片规模 | 标准集 | 约 4.5M |
| Batch size | 30 | 120 |
| 硬件 / 时间 | 论文未以同口径详列 | 8×V100，约 240 小时 |

Big LaMa 仍以低分辨率 $256\times256$ crop 训练，原图近似 $512\times512$。它不是高分辨率专门微调模型，而是通过更大容量、更多数据和更大 batch 检验方法是否可以扩展。

![Big LaMa 正例](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig06-big-lama-positive.webp)

*建筑、街景、游乐场和百叶背景中的成功对象移除。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 6, CC BY 4.0.*

正例展示了大面积人物和前景移除后，建筑纹理、街道、摩天轮、店面与百叶线条的延续。它们很好地说明 LaMa 适合结构性背景。但这些图并非随机抽样成功率统计；作者将其明确称为 positive examples。评价模型时，必须同时看下一节的 negative examples。

Big LaMa 的 240 小时训练成本是在 8 张 V100 上报告的历史结果。它不能直接映射到今天的 GPU 时间，更不能简单换算云成本：数据 I/O、软件版本、混合精度、通信和实际利用率都未知。该数字的价值是展示训练规模，而不是提供采购报价。

## 34. 失败案例：透视、复杂背景与虚焦

![Big LaMa 失败案例](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig07-big-lama-negative.webp)

*链网、长椅、工业结构和街景中的透视或复杂背景失败。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 7, CC BY 4.0.*

失败图比正例更能解释方法边界。

**强透视周期。** 链式围栏在图像不同位置的网格大小和方向变化。全局频谱倾向描述整体频率，但透视结构是空间非平稳的。模型可能生成一块统计上像网格、几何上却无法接续的纹理。

**复杂背景层次。** 人物后面同时有栏杆、立柱、建筑和光照。删除后需要恢复多个深度层。LaMa 只有 RGB 监督，没有显式深度或平面约束，容易把层次混合。

**被遮挡信息过多。** 当对象覆盖关键结构节点，洞外没有足够证据。模型只能依据数据先验猜测，视觉上可能合理，却与真实背景不同。

**Bokeh 与景深。** 背景虚焦程度随深度和镜头变化。生成器若补出过锐或模糊尺度不一致的纹理，用户会立即注意到编辑区。

**重复结构的相位错误。** FFC 擅长发现频率，不保证精确边缘连接。局部相位错一格，FID 可能变化不大，但视觉上明显。

这些错误没有简单的后处理万能修复。产品可以通过减小 mask、保留更多边界上下文、提供多候选结果、检测结构线和人工复核来降低风险。对关键场景，最好允许用户调整 mask 或回退到内容感知填充，而不是强制接受单次输出。

## 35. 跨领域案例：正例是能力线索，不是 benchmark

![频谱图、病理、遥感、绘画和游戏画面的跨领域修复](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig08-domain-transfer.webp)

*Big LaMa 在 Places 之外的音乐频谱、组织切片、鸟瞰图、绘画和游戏画面上展示的案例。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 8, CC BY 4.0.*

![Outpainting 与 MRI 示例](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig09-more-domain-transfer.webp)

*店面 outpainting 和 MRI 图像修复示例。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 9, CC BY 4.0.*

这些图说明模型学到的某些低层结构先验可以迁移：连续纹理、重复谱线、组织纹理、道路模式、绘画笔触和游戏背景在频率层面与自然图像共享部分规律。它们是很有价值的探索线索。

但不能据此声称 LaMa 适合医疗修复。MRI 和病理图像中的小结构可能携带诊断信息，视觉平滑的补全可能制造或删除病灶。论文没有医学 ground truth 评测、临床读片、灵敏度/特异度、失效检测或监管流程。示例只说明网络能输出“看起来连续”的图，不说明医学正确。

类似地，遥感补全可能改变建筑、道路或事件证据；频谱图补全可能改变信号分析；绘画修复涉及艺术史和作者风格。跨域上线必须重新定义 ground truth、风险和审计。一个通用自然图像模型的好看结果，不自动拥有专业领域许可。

## 36. LaMa-Dilated：不用 FFT 的替代路线

![LaMa-Dilated 架构](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig10-dilated-architecture.webp)

*每个 residual block 将通道分为 dilation 1/2/4/8 的四组并求和。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 10, CC BY 4.0.*

论文并没有把 Fourier 描述成唯一扩大感受野的方法。LaMa-Dilated 把 regular residual block 替换为 Multi-Dilated Convolution Block：输入沿 channel 分成四组，分别使用 dilation 1、2、4、8 的 $3\times3$ convolution，再把结果相加或融合。

这种结构有几个优势：

- 只依赖常规 convolution，部署框架支持更成熟；
- 在移动端或不擅长 FFT 的加速器上可能更容易优化；
- Table 2 的 wide FID `2.30` 接近 Fourier 的 `2.21`；
- narrow LPIPS `0.089` 还略优于 Fourier 的 `0.090`。

局限是 dilation 采样位置仍是固定离散网格，感受野虽大但不是每个 block 的全图覆盖；大 dilation 还可能产生 gridding artifact。分辨率继续上升时，固定 dilation 相对于图像尺寸仍会缩小，Fig. 6 中的退化速度快于 Fourier。

这组对照体现了很实用的工程态度：若目标硬件没有高效 `rFFT/irFFT`，LaMa-Dilated 可能是可接受替代，而不是因为论文题名包含 Fourier 就必须不计成本地使用 FFT。模型选择应结合质量、延迟、算子支持和部署维护。

## 37. 推理效率：参数量、算子速度和分辨率必须分开

![LaMa 推理时间曲线](/images/blog/lama-resolution-robust-large-mask-inpainting/lama-supp-fig11-inference-time.webp)

*Nvidia 1080Ti、batch size 100、100 次运行平均的 sec/image 曲线。Source: Suvorov et al., arXiv:2109.07161v2 / WACV 2022, Supplementary Fig. 11, CC BY 4.0.*

补充 Fig. 11 比较 LaMa-Regular、LaMa-Fourier、LaMa-Dilated、Big LaMa-Fourier 和 Big LaMa-Regular。测试在 Nvidia 1080Ti 上，以 batch 100 让 GPU 充分负载，并对 100 次运行取平均。

曲线显示，推理时间随分辨率显著上升。标准 LaMa-Regular 最快，LaMa-Fourier 稍慢；LaMa-Dilated 与 Big Fourier 在高分辨率更慢；Big Regular 的位置反映参数量和卷积效率的另一种组合。论文正文概括 LaMa-Fourier 平均比 Regular 慢约 20%，同时参数少约 40%。

这份测量不能直接代表在线单图延迟。batch 100 更接近吞吐 benchmark，交互式编辑通常 batch 1；1080Ti 的 FFT kernel、内存带宽和现代 GPU 不同；预处理、mask 读取、padding、模型加载、输出编码和可选 refinement 都未纳入单纯 forward 曲线。

部署时至少要分别记录：

| 指标 | 原因 |
| --- | --- |
| Batch 1 p50/p95 延迟 | 交互式编辑体验 |
| 固定 GPU 的 images/s | 离线批处理吞吐 |
| 峰值显存 | 决定最大分辨率与并发 |
| FFT workspace | 可能随尺寸跳变 |
| 冷启动 / 权重加载 | serverless 或弹性扩容 |
| 预处理与编码 | PNG/JPEG、mask resize 可能占显著比例 |
| 失败与 OOM 比例 | 仅平均延迟会掩盖不可服务请求 |

“27M 模型”只能说明权重规模，不能回答这些问题。

## 38. 论文到源码的完整主链

官方实现的训练数据流可以概括为：

```text
image dataset
  -> random crop / augmentation
  -> mask generator
  -> masked RGB + mask
  -> FFCResNetGenerator
  -> predicted full image
  -> composite image
  -> HRF perceptual loss
  -> PatchGAN adversarial loss
  -> discriminator feature matching
  -> R1 regularization
  -> generator/discriminator optimizers
  -> checkpoint
```

预测数据流更短：

```text
image + mask files
  -> dataset loader
  -> padding to supported shape
  -> checkpoint/config restoration
  -> one generator forward
  -> known-region compositing
  -> crop padding
  -> output image
```

主要代码位置如下：

| 职责 | 固定 commit 路径 |
| --- | --- |
| FFC / Fourier Unit / Generator | [`training/modules/ffc.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/modules/ffc.py) |
| Mask generators | [`training/data/masks.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/data/masks.py) |
| Perceptual loss | [`training/losses/perceptual.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/perceptual.py) |
| Adversarial / R1 | [`training/losses/adversarial.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/adversarial.py) |
| Feature matching | [`training/losses/feature_matching.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/losses/feature_matching.py) |
| Trainer | [`training/trainers/default.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/training/trainers/default.py) |
| 训练入口 | [`bin/train.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/bin/train.py) |
| 预测入口 | [`bin/predict.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/bin/predict.py) |
| 可选 refinement | [`evaluation/refinement.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/evaluation/refinement.py) |

源码组织说明 LaMa 不是一个孤立的 `nn.Module`。训练复现依赖数据注册、Hydra config、Lightning module、loss factory 和 checkpoint 元数据。只把 `FFCResNetGenerator` 抄进新项目，可能跑出图，却不等于复现论文。

## 39. 配置系统：真正的模型定义散落在哪里

LaMa 使用 Hydra 把实验拆成多层配置。网络结构不是只在 Python 构造器里硬编码；training config 引用 generator、discriminator、loss、data、optimizer 和 trainer 子配置，命令行还可以覆盖任意字段。

[`configs/training/lama-fourier.yaml`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/lama-fourier.yaml) 描述标准 Fourier 训练组合；[`configs/training/big-lama.yaml`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/big-lama.yaml) 对应更大的模型设置；[`configs/training/generator/ffc_resnet_075.yaml`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/configs/training/generator/ffc_resnet_075.yaml) 固定了 4 通道输入、3 通道输出、3 次 downsampling、9 个 residual blocks、global ratio `0.75` 与 `enable_lfu: false` 等关键结构。

概念上，一个 resolved config 至少包含：

```yaml
generator:
  kind: ffc_resnet
  input_nc: 4
  output_nc: 3
  n_downsampling: 3
  n_blocks: 9
  resnet_conv_kwargs:
    ratio_gin: 0.75
    ratio_gout: 0.75
    enable_lfu: false

losses:
  adversarial:
    weight: 10
  perceptual:
    weight: 30
  feature_matching:
    weight: 100
  r1:
    weight: 0.001
```

这只是便于阅读的摘要，不应直接替换仓库配置。Hydra 的 defaults 合并、插值、命令行覆盖和运行目录会改变最终值。复现时应保存 `.hydra/config.yaml`、`.hydra/overrides.yaml`、Git commit、Python lock/conda env、CUDA/cuDNN 版本和输入数据清单。

配置分层的好处是能系统做消融：只切换 generator 为 regular/dilated，保持 loss/data 不变；或只切换 mask generator，保持模型不变。风险则是实验漂移隐藏在多个 YAML 中。论文数字与某个 checkpoint 对不上时，先比较 resolved config，而不是只读顶层文件。

Big LaMa 的“18 blocks、51M、batch 120”也不是单个类名表达的。它来自结构与 training config 的组合。源码阅读必须沿配置引用向下追踪，否则容易把默认 9 blocks 当成所有模型，或者把代码支持的 LFU 误写成论文实际启用。

## 40. 预测入口：一次前向并不意味着没有工程细节

官方 [`bin/predict.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/bin/predict.py) 负责加载预测配置、数据集和训练目录中的 checkpoint。README 的典型用法大致是：

```bash
python3 bin/predict.py \
  model.path=/absolute/path/to/big-lama \
  indir=/absolute/path/to/input \
  outdir=/absolute/path/to/output
```

输入目录通常要求图片与对应 mask 按命名约定配对。实际命令和文件后缀应以固定 commit README 与 prediction dataset 实现为准。不要把任意 alpha PNG 丢进去并假定它会自动识别哪一部分需要删除。

预测过程还有几项容易影响结果：

**Padding。** 下采样网络要求尺寸适配一定倍率。实现会 pad 到可处理大小，前向后再裁回原尺寸。若使用 reflection padding，靠近边缘的洞会看到镜像上下文；若产品改成 zero padding，边缘行为会不同。

**已知区域合成。** 生成器输出完整图，但最终通常只在 mask 内采用预测。若直接保存 raw prediction，mask 外可能有轻微颜色漂移，指标和用户体验都会变化。

**颜色与动态范围。** PIL/OpenCV 的 RGB/BGR、整数/浮点归一化、PNG gamma 和 JPEG 编码可能造成“模型输出颜色不对”的假象。必须在模型前后各保存一个校验样本。

**mask 二值化。** 软 mask 是用于边缘混合，还是直接作为模型第四通道，必须明确。论文训练主要使用二值 mask；产品把抗锯齿 alpha 直接输入可能形成未验证分布。

**模型加载。** checkpoint 与 `config.yaml` 是一对。仅复制权重文件而丢失 generator 配置，会导致结构不匹配或使用错误默认值。

“单阶段一次前向”意味着模型计算图简单，不代表整个文件到文件链路没有失败点。工程验收要分别检查预处理、模型、合成和编码。

## 41. Refinement：可选扩展不能倒写成论文主模型

仓库 [`saicinpainting/evaluation/refinement.py`](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/saicinpainting/evaluation/refinement.py) 提供可选 multi-scale refinement。它通常先运行 LaMa 得到初始结果，再在图像金字塔或中间 feature 空间做优化，使高分辨率输出更贴合感知特征和边界。

这与标准预测有本质差别：

| 维度 | 标准 LaMa | Refinement |
| --- | --- | --- |
| 计算 | 单次 generator forward | 多尺度、多步优化 |
| 状态 | 固定权重推理 | 每张图优化中间表示 |
| 延迟 | 相对低且稳定 | 显著更高，随步数与尺寸变化 |
| 显存 | 主要是一次前向 activation | 需要保留优化图与多尺度状态 |
| 复现口径 | 主论文 Table/Fig 核心路径 | 仓库可选增强 |

因此，看到某个社区 demo 标注“LaMa refinement”时，不能把它的质量或延迟归入论文标准 27M 模型。反过来，部署标准 LaMa 时，也不应把 refinement 当成必需依赖。

refinement 适合离线、少量高价值图片或对细节要求高且允许等待的工作流。交互式对象删除通常更看重稳定的 batch-1 延迟。合理产品可以提供“快速预览”和“高质量导出”两种模式，但必须分别压测、分别标注，并允许用户取消高质量任务。

多 GPU 分段与 feature optimization 还增加故障恢复复杂度。若中途 OOM 或超时，系统应保留标准前向结果作为回退，而不是让整个编辑失败。论文没有为这条可选路径提供生产 SLA，工程团队需要自行定义。

## 42. 可复现性审计：公开代码解决了什么，没解决什么

LaMa 的公开程度在 2021 年图像生成论文中较好：论文、补充材料、训练代码、预测代码、配置和预训练模型路径都曾提供。固定代码仍能清晰映射 FFC、loss 与 mask generator。

但 2026 年重新复现会遇到现实障碍：

1. **依赖老化。** PyTorch Lightning 1.2.9、Hydra 和当时的 PyTorch/CUDA 组合与现代环境差异大。
2. **链接迁移。** 原始 Yandex 权重或数据资源存在失效/访问变化，镜像不一定带原始校验值。
3. **数据规模。** Places 与 Big LaMa 的 4.5M 图像准备、存储和 I/O 不是普通本地实验。
4. **训练成本。** Big LaMa 报告 8×V100、约 240 小时；标准模型也有 1M iterations。
5. **随机性。** 随机 crop、随机 mask、GAN 训练和分布式顺序都会影响结果；公开材料未把所有 seed 与确定性设置固化为一个现代一键环境。
6. **评测资产。** segmentation test set、用户研究页面和 baseline checkpoint 需要按论文协议重建。
7. **软件修复。** 将旧 API 迁移到现代 `torch.fft`、Lightning 或纯 PyTorch 时，细微 padding、normalization 和 checkpoint loading 差异会改变输出。

一次严谨复现应分层验收，而不是直接追 Table 1：

```text
Level 0: 配置可解析，单个 batch 可前向
Level 1: FourierUnit / FFC shape 与梯度单测通过
Level 2: 官方 checkpoint 在固定样例上与参考输出一致
Level 3: 官方测试集 FID/LPIPS 可重算
Level 4: 训练小模型能复现趋势
Level 5: 完整 Places/Big LaMa 训练复现绝对数字
Level 6: 用户研究协议复现
```

本文完成的是论文内容核对与固定 commit 静态阅读，不属于上述运行级复现。文章中的实验数值来自论文，不是本机产出。

## 43. 与后续修复方法的关系

LaMa 之后的图像修复大致沿几条路线演进。

**AOT-GAN** 使用 aggregated contextual transformations，在多 dilation/上下文聚合上提高大洞结构能力。它和 LaMa 都重视长程上下文，但机制不同：AOT 更偏空间多尺度聚合，LaMa 用 local/global FFC。

**MAT（Mask-Aware Transformer）** 通过 mask-aware attention 和 transformer 表达远距离依赖。注意力能更灵活地关联非周期语义区域，但计算和部署特征与 FFT 不同。两者都在回答“洞中心如何利用远处证据”，不是简单的新旧替代。

**ZITS** 等结构增强方法显式引入边缘、线段或 transformer 结构先验，对建筑和透视线可能更有控制力；代价是 pipeline 更复杂，中间结构预测错误会级联。

**扩散式 inpainting**，包括 RePaint、latent diffusion 与 Stable Diffusion inpainting，提供更强语义生成、多样性和文本控制。它们能对巨大缺失区生成新的语义内容，而 LaMa 更擅长快速、确定性地延续可见背景。扩散模型通常需要多步采样，延迟和一致性策略不同。

**SAM-assisted object removal** 解决的是 mask 获取。SAM/SAM2 可以把点击或框转换为对象 mask，再交给 LaMa 或扩散模型修复。SAM 不负责生成背景；LaMa 不负责理解用户点的是哪个对象。把两者连接起来是产品 pipeline，而不是原论文能力。

从 2026 年视角看，LaMa 仍有三个现实位置：

- 低延迟、本地或批量对象删除；
- 作为扩散编辑前的快速预览或 baseline；
- 在结构性背景、无需文本控制的场景中提供稳定起点。

它不再代表所有语义修复的最高上限，但“全局感受野 + 困难 mask 训练 + 结构感知损失”的设计原则仍被后续方法继承。

## 44. 工程落地：从输入 mask 到可回滚输出

一个可靠的对象删除服务不应只有 `model(image, mask)`。建议把链路拆成以下阶段。

### 44.1 Mask 预处理

记录原始 mask，不要直接覆盖。对分割 mask 可做小幅 dilation，确保对象边缘、阴影或光晕被覆盖；dilation 半径应随分辨率或对象尺寸变化，而不是固定像素。过度膨胀会删除更多真实背景，增加生成难度。

硬边二值 mask 用于模型输入，软边 feather mask 可用于最终合成。两者用途不同。把 feather 后的灰度 mask直接输入一个只见过二值 mask 的模型，可能导致边界行为异常。

对于触边对象，普通对象移除会变成部分 outpainting。应单独标记并使用不同阈值、模型或人工提示。极细结构、多连通组件和巨大 mask 也应进入专门路由。

### 44.2 分辨率策略

优先整图缩放到验证过的最大边长，再一次前向。简单 tile 会破坏 FFC 的全图上下文：每块看到不同频谱和边界，拼接处容易出现颜色/相位不一致。如果必须 tile，应增加大 overlap、统一低分辨率全局引导，并在重叠区做平滑融合，但这已超出论文标准方法。

不要只按总像素决定 OOM。FFT 对尺寸因子、workspace 和后端 kernel 可能敏感。压测矩阵应覆盖常见宽高比和非 2 的幂尺寸。

### 44.3 结果合成

模型输出后，把已知区域从原图精确复制；只在 mask 边缘使用可控 feather。保存原图、原始 mask、处理后 mask、raw prediction 和 final composite，便于问题审计。

若图片带 alpha、ICC profile、EXIF orientation 或 HDR 信息，需在预处理前归一化，并在输出时明确哪些元数据保留。否则模型像素正确，导出仍可能旋转、偏色或丢透明度。

### 44.4 失败检测

可以结合规则和模型：

- mask 内外边界的颜色/梯度突变；
- 重复结构频率在洞两侧与洞内是否一致；
- 人脸、文字、logo、车牌等高风险区域检测；
- 输出局部的过度平滑、棋盘格和重复 patch；
- 多次轻微 mask 扰动后的输出稳定性；
- 用户撤销、重试和手工修改率。

失败检测不能证明修复真实，但能把明显 artifact 路由到第二模型或人工处理。

### 44.5 回滚与可观测性

每次请求至少记录模型版本、checkpoint hash、resolved inference config、输入尺寸、mask 统计、延迟、显存/OOM、输出 hash 与用户操作。产品必须保留原图，支持无损撤销。对于批处理，单张失败不应中断整批。

## 45. 生产测试矩阵：不要只复算 FID

论文 benchmark 是方法比较起点，生产验收应覆盖真实工作流。一个可执行矩阵如下：

| 维度 | 分桶建议 | 主要检查 |
| --- | --- | --- |
| Mask area | 0-5%, 5-20%, 20-40%, >40% | 质量退化与 OOM |
| Mask width | narrow / medium / wide | 远程上下文能力 |
| 连通分量 | 1, 2-5, >5 | 多洞一致性 |
| 边界位置 | 内部 / 触边 / 跨角 | outpainting 风险 |
| 图像尺寸 | 512, 1K, 2K, 4K | 延迟、显存、结构 |
| 长宽比 | 1:1, 4:3, 16:9, panorama | FFT 与 padding |
| 内容 | 人像、建筑、自然、文字、商品 | 域偏差 |
| 结构 | 周期、强透视、虚焦、反射 | 典型失败 |
| Mask 来源 | 手画、SAM、检测框、alpha | 边界分布 |
| 编码 | PNG、JPEG、透明、广色域 | 前后处理一致性 |

指标也应分层：

**自动像素/特征指标**：LPIPS、PSNR/SSIM（仅作辅助）、边界梯度差、局部频率一致性。

**集合指标**：FID 或更适合当前数据域的 distribution metric。样本量不足时 FID 方差很大，应报告置信区间。

**任务指标**：对象是否完全删除，周围主体是否保留，文字/logo 是否被意外修改，是否出现新对象。

**人评**：视觉真实感、意图符合度、边界可见性、是否愿意直接导出。评审应看到足够分辨率并允许放大。

**系统指标**：batch-1 p50/p95/p99、吞吐、OOM、重试、冷启动、GPU 利用率、单请求成本。

**行为指标**：撤销率、再次涂抹率、切换高质量模式比例、人工修图时间。它们往往比离线 FID 更接近产品价值。

论文的 narrow/wide 分层值得直接继承。把所有请求混成一个平均分，会掩盖模型对真正困难大洞的退化。

## 46. 安全、事实与产品边界

LaMa 会生成原图中不存在的像素。这个事实决定了它不应在没有标识和审计的情况下修改证据型图像。

**新闻与司法。** 删除人物、车辆、标识或时间线索会改变事件叙述。系统应保留原件、记录编辑操作，并在导出或元数据中标明内容已被生成式修改。

**医疗与科学。** 补出的组织、病灶、云层、遥感地物或频谱信号可能看似连续但不真实。不能把论文跨域展示当作专业安全验证。科学数据修补至少要同时保留 mask、模型版本和不确定性说明，并禁止把生成区域用于定量测量。

**身份与隐私。** 对象删除可能暴露此前被遮挡的猜测内容，也可能错误重建私人信息。模型输出不是真实“恢复”，不应被描述成看到了被遮挡背景。

**版权与艺术修复。** 对画作或摄影作品的补全会引入模型先验。商业使用还需核对训练数据、模型权重和输出政策；论文/代码许可不自动解决训练数据和内容权利。

**滥用与透明度。** 产品应限制批量证据篡改场景，提供审计日志和原图对比。若用于普通创意编辑，至少要让用户知道结果是生成内容，而不是真实还原。

技术上，LaMa 没有 uncertainty head，也不生成多候选概率。输出像素很确定，不代表 epistemic uncertainty 低。若业务风险高，应通过多模型一致性、mask 扰动、人工复核或直接拒绝处理来管理不确定性。

## 47. 局限性与批判

把论文贡献与边界放在一起，可以得到更完整的判断。

### 47.1 结构先验仍是隐式的

FFC 提供全局通信，但没有显式透视、深度、平面、对称或对象拓扑。周期结构在频域中更显著，强透视和多层遮挡仍可能失败。若任务要求精确建筑线条，显式 line/depth guidance 可能更稳。

### 47.2 单输出掩盖多解性

LaMa generator 通常给定输入后输出一个结果，没有像扩散采样那样自然提供多候选。大洞本来存在多种合理答案，单输出使用户无法比较，也让 LPIPS ground-truth 评测把合理差异当成错误。

### 47.3 GAN artifact 与训练稳定性

对抗训练可以提高锐度，也可能引入重复纹理、局部噪声和训练不稳定。论文使用 R1 与 feature matching 缓解，但没有消除。重新训练到新域需要认真调 discriminator 和 loss，而不是只冻结论文权重。

### 47.4 训练域偏差

Places 强调自然场景，CelebA-HQ 强调人脸。文档、UI 截图、工业缺陷、医学和遥感并未系统覆盖。补充案例是定性证据，不是域内评测。

### 47.5 分辨率泛化有上界

模型在 512-2048 范围退化更慢，不代表 4K/8K 或超长 panorama 仍可靠。FFT 内存、特征分辨率和未见纹理都会形成上界。论文也未提供无限尺寸策略。

### 47.6 指标不完整

FID/LPIPS 不能判断生成事实、文字正确性、身份一致性或用户意图。用户研究也偏向视觉可检测性。论文缺少结构线指标、mask 边界专门指标和失败置信度。

### 47.7 依赖环境已老化

代码开放是优势，但旧 PyTorch/Lightning/Hydra 组合和迁移后的权重链接增加复现成本。现代重写若没有参考输出与单元测试，很容易产生“能跑但不等价”的实现。

### 47.8 FFC 不是免费全局上下文

FFT 有复杂内存访问和尺寸敏感性。LaMa-Fourier 参数更少，却比 Regular 慢约 20%。在某些 NPU、移动端或推理引擎中，dilated convolution 可能更实际。

这些局限不削弱 LaMa 的历史贡献。它们说明正确定位是“强大且高效的大 mask baseline”，而不是“所有图像编辑的最终方案”。

## 48. 复现与源码阅读清单

如果要把静态阅读推进到可运行复现，建议按以下顺序：

### 环境与资产

- 固定仓库 commit `786f5936b27fb3dacd2b1ad799e4de968ea697e7`；
- 保存 Python、PyTorch、torchvision、Lightning、Hydra、CUDA 和 cuDNN 版本；
- 对模型权重记录下载来源、文件大小和 SHA-256；
- 确认论文 checkpoint 与 config 目录完整；
- 只在拥有合法访问权限时准备 Places/CelebA-HQ；
- 保存 evaluation masks，而不是每次随机重生成。

### 模块单测

- `FourierUnit` 输入输出 shape、dtype 与有限值；
- `rFFT -> real/imag -> irFFT` 的维度顺序；
- local/global channel split 在 ratio 0、0.75、1 附近的边界；
- `FFCResnetBlock` residual shape；
- mask 为全 0、全 1、单像素和大矩形时的数据路径；
- 不同宽高、奇偶尺寸与 padding；
- 已知区域最终合成必须逐像素保持原图。

### 参考输出

- 选择 5-10 张公开样例和固定 mask；
- 保存 raw generator output 与 final composite；
- 比较 CPU/GPU、FP32/AMP 和新旧 PyTorch；
- 允许浮点微差，但对结构、颜色和边界做图像回归；
- 将参考输出与 checkpoint hash 绑定。

### 指标复算

- narrow/wide/segmentation masks 分开；
- 明确 FID 实现、Inception 权重与样本量；
- 明确 LPIPS backbone 和输入归一化；
- 对 40%-50% masked 困难子集单独报告；
- 用 bootstrap 或多 seed 报告不确定性。

### 训练复现

- 先用小数据确认 loss 曲线和可视结果；
- 再复现 Regular vs Fourier 的相对趋势；
- 再做 narrow vs wide mask 消融；
- 最后才考虑 1M iterations 或 Big LaMa；
- 所有实验保存 resolved Hydra config、代码状态和日志。

只有完成参考输出或指标复算，才能说“运行了官方模型”；只有完整训练和协议一致，才能说“复现了论文实验”。本文没有做这些运行步骤。

## 49. 推荐阅读路径

时间有限时，建议按下面顺序阅读：

1. Abstract 与 Fig. 1：理解作者要解决的大洞、周期结构和高分辨率问题；
2. Fig. 2 与 Section 2：看清 FFC generator、HRFPL、adversarial 和 mask strategy；
3. Table 2/3/4：分别验证 generator、loss 和 mask 的贡献；
4. Fig. 4/5/6：理解周期结构与跨分辨率主张；
5. Supplement Fig. 4/5 与 Table 4：看 mask 分布和参数；
6. Supplement Fig. 6/7：同时看成功和失败；
7. `ffc.py`：从 `FourierUnit` 读到 `FFCResNetGenerator`；
8. `masks.py`、`perceptual.py`、`adversarial.py`、`default.py`；
9. `lama-fourier.yaml`、`ffc_resnet_075.yaml`、`bin/predict.py`；
10. 最后再读 `evaluation/refinement.py`，避免把可选路径误当主模型。

若要补背景，先读 Fast Fourier Convolution 原论文和 Effective Receptive Field，再对照 DeepFillv2、EdgeConnect、CoModGAN。若关心现代方法，再读 AOT-GAN、MAT、ZITS 和扩散式 inpainting。

## 50. 结论

LaMa 的方法表面上很简洁：四通道输入、ResNet-like generator、若干 FFC blocks 和一组 loss。但论文真正完整的论证是三层协同。

**表示层**用 local/global channel split 和 Fourier Unit 缩短远程信息路径，使洞内位置能直接利用全图上下文；这对窗格、栅栏和长结构尤其有效。

**监督层**用带 dilation 的 segmentation ResNet feature 构造 HRF perceptual loss，让训练目标不只奖励局部纹理，还更关注大尺度语义结构；PatchGAN、feature matching 和 R1 再补足锐度与稳定性。

**数据层**主动生成更宽、更复杂的 irregular 与 box masks，让模型在训练时不得不使用远距离证据。Table 4 说明这条经验还可迁移到部分其他模型，但不同架构存在窄洞与宽洞的取舍。

实验最强的结论不是“FFT 总是最好”，而是：当 mask 变宽、周期结构变长、测试分辨率超过训练分辨率时，拥有全局有效感受野的模型退化更慢。Regular convolution 在窄洞上可以相当甚至略优；dilated convolution 是有竞争力的替代；Big LaMa 的收益还混合了容量、数据和 batch 扩展。

从今天看，扩散模型提供了更强的语义生成和文本控制，结构引导方法提供更显式的几何约束，SAM 让对象 mask 获取更方便。但 LaMa 仍然是一条重要基线：单阶段、无需文本、推理相对快、擅长延续背景结构，并且公开实现足以让工程师检查每个核心机制。

最应保留的工程原则是：

> 大区域修复不能只在网络中“加一个全局模块”。生成器要看得远，损失要评价远程一致性，训练 mask 也要迫使模型使用远处证据；三者缺一，模型都可能重新退回局部捷径。

同时必须保留产品边界：LaMa 生成的是视觉猜测，不是真实恢复；高分辨率稳健不是无限尺寸保证；跨领域正例不是专业安全认证；optional refinement 不是论文主模型；论文指标也不能替代真实 mask、真实硬件和真实用户流程的验收。

## 参考资料

1. Suvorov et al., [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161v2), arXiv `2109.07161v2`.
2. Suvorov et al., [WACV 2022 open-access paper](https://openaccess.thecvf.com/content/WACV2022/html/Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_Convolutions_WACV_2022_paper.html), pp. 2149-2159.
3. Official repository, [advimman/lama](https://github.com/advimman/lama), 本文固定 commit [`786f5936`](https://github.com/advimman/lama/tree/786f5936b27fb3dacd2b1ad799e4de968ea697e7).
4. Chi et al., [Fast Fourier Convolution](https://proceedings.neurips.cc/paper/2020/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html), NeurIPS 2020.
5. Luo et al., [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html), NeurIPS 2016.
6. Zhang et al., [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924), CVPR 2018.
7. Heusel et al., [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), NeurIPS 2017.
8. Yu et al., [Free-Form Image Inpainting with Gated Convolution](https://openaccess.thecvf.com/content_ICCV_2019/html/Yu_Free-Form_Image_Inpainting_With_Gated_Convolution_ICCV_2019_paper.html), ICCV 2019.
9. Nazeri et al., [EdgeConnect: Generative Image Inpainting with Adversarial Edge Learning](https://arxiv.org/abs/1901.00212), 2019.
10. Zhao et al., [Large Scale Image Completion via Co-Modulated Generative Adversarial Networks](https://openreview.net/forum?id=sSjqmfsk95O), ICLR 2021.
11. Zeng et al., [Aggregated Contextual Transformations for High-Resolution Image Inpainting](https://arxiv.org/abs/2104.01431), ICCV 2021。
12. Li et al., [MAT: Mask-Aware Transformer for Large Hole Image Inpainting](https://openaccess.thecvf.com/content/CVPR2022/html/Li_MAT_Mask-Aware_Transformer_for_Large_Hole_Image_Inpainting_CVPR_2022_paper.html), CVPR 2022.
13. Dong et al., [Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Incremental_Transformer_Structure_Enhanced_Image_Inpainting_With_Masking_Positional_Encoding_CVPR_2022_paper.html), CVPR 2022.
14. Lugmayr et al., [RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html), CVPR 2022.
15. Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html), CVPR 2022.

## 复现状态声明

本文完成了 arXiv v2、WACV 主论文、官方补充材料和固定 commit 源码的静态交叉阅读；图表数字由最终 PDF 人工核对。本文没有下载 LaMa/Big LaMa 权重，没有安装旧版训练环境，没有运行训练、预测、refinement、FID/LPIPS 或用户研究，也没有声称复现论文结果。文中所有论文实验数字均归属于 Suvorov 等人的原始报告。

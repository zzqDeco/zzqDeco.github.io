---
title: "TrustMark 论文精读：任意分辨率图像水印、频谱损失与可重水印工作流"
description: "精读 TrustMark 如何通过 MUNIT 嵌入器、ResNet50 提取器、可微扰动、焦点频率损失、分辨率缩放和 TrustMark-RM 实现高质量鲁棒图像水印"
pubDate: "2026-08-24T11:22:25+08:00"
updatedDate: "2026-08-24T11:22:25+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "Image Watermarking"
  - "Content Provenance"
  - "Computer Vision"
  - "Generative AI"
  - "Code Reading"
draft: false
---

生成式图像进入日常创作之后，内容溯源逐渐演变为一项跨链路系统工程。文件头元数据可以记录作者、编辑历史和密码学签名，却极易在社交平台转码、截图或普通导出时被剥离；被动图像分类器能够评估像素分布是否接近某类生成模型，却无法将判断绑定到可审计的资产标识；传统频域或最低有效位（LSB）水印则往往在大幅裁切、有损压缩、缩放与多轮编辑后迅速失效。更现实的制约在于，生产工作流中的图像很少停留在 $256\times256$ 的标准尺寸：相机原片、印刷海报、数字绘画与生成式资产横跨不同的分辨率、长宽比与纹理统计。

Adobe Research 与萨里大学提出的 **TrustMark: Universal Watermarking for Arbitrary Resolution Images**，尝试在同一个端到端系统中回应对抗分发扰动、保持高视觉质量与适配任意分辨率这三项需求。它在 MUNIT 风格的编码器与 ResNet50 提取器之间插入可微噪声模拟层，并联合优化 YUV 像素误差、LPIPS 感知特征、焦点频率损失（Focal Frequency Loss, FFL）与 WGAN-GP 对抗损失；在推理端，它通过“在 $256^2$ 分辨率上预测残差、再双线性插值回原始尺寸”的残差缩放策略（Residual Resolution Scaling）绕开固定输入尺寸限制；同时，论文还配套训练了基于 KBNet 的水印移除网络 **TrustMark-RM**，支持“先擦除旧水印、再写入新水印”的多轮重水印（re-watermarking）工作流。

> **核心判断**：TrustMark 的核心工程价值，在于把深度不可见水印从固定小图上的编解码实验推进到了兼顾分发扰动、任意分辨率残差搬运、多轮重写、BCH 纠错与 C2PA 内容凭证软绑定（soft binding）的完整资产接口；而在安全边界上，它的鲁棒性依然受限于训练噪声分布，公开解码器与移除器客观上扩大了攻防面，且像素内恢复出的比特流本身只是耐久检索线索，必须配合外部密码学签名与托管注册表才能构成完整的来源证明。

## 1. 论文基线、代码版本与问题边界

本文以 arXiv [`2311.18297v1`](https://arxiv.org/abs/2311.18297v1) 为论文与实验分析基线（后续 ICCV 2025 录用版将题名扩展为 *TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images*），源码静态走读锁定在官方仓库 [`adobe/trustmark`](https://github.com/adobe/trustmark/tree/0ed40cbe8188f664fd9cbbeacd969807de27440a) 的固定提交 `0ed40cbe`。

| 维度 | 内容与口径说明 |
| --- | --- |
| 题名与作者 | *TrustMark: Universal Watermarking for Arbitrary Resolution Images*（Tu Bui, Shruti Agarwal, John Collomosse） |
| 论文版本 | arXiv `2311.18297v1`（2023-11-30，14 页含补充材料，arXiv non-exclusive distribution license） |
| 源码基线 | [`adobe/trustmark`](https://github.com/adobe/trustmark/tree/0ed40cbe8188f664fd9cbbeacd969807de27440a) commit `0ed40cbe`（MIT 许可，源码内 `pyproject.toml` 标 `0.9.0`，PyPI 已发至 `0.9.1`） |
| 论文与仓库范围差异 | 论文 v1 仅实验 `B`、`Q` 两款模型，评测无纠错码（ECC）的原始比特准确率；当前开源库新增了 `C`、`P` 变体、BCH 纠错、中心区域写入、边缘羽化、局部检测器、JS/Rust 跨语言运行时及 C2PA 软绑定示例，但未开源完整训练流水线与 MIR-Flickr 训练清单 |

在进入架构细节前，需要先厘清不可见水印在内容溯源体系中的确切位置及其四元工程权衡：

1. **产品定位与相近技术边界**：不可见数字水印将短比特载荷主动嵌入像素纹理，与追求威慑效果的**可见水印**、隐藏通信存在性的**隐写术（Steganography）**、需要抗合谋追踪的**数字指纹（Fingerprinting）**以及做被动概率判别的 **AI 图像检测器**目标各异。在与 **C2PA 元数据**配合时，TrustMark 充当的是耐久的**软绑定标识符通道（soft-binding identifier）**：当外部签名清单（manifest）因平台转码丢失时，系统可凭像素中恢复的短随机 ID 去可信数据库中查回原始签名声明，而信任本身仍由外部密码学验证承担。
2. **威胁模型区分**：论文核心解决的是良性分发链路中的**经验鲁棒性**（缩放、裁切、JPEG 压缩、色彩调整、模糊与噪声）。在恶意对抗场景下，攻击目标可细分为**移除（Removal）**、**伪造（Spoofing）**、**跨图转移（Transfer）**与**载荷替换（Substitution）**。论文仅探索了 I-FGSM 白盒移除与专用恢复网络 TrustMark-RM；由于编码器与解码器均公开且无私钥门控，任何拥有模型的攻击者均可写入任意公开载荷，因此安全防伪必须依赖服务端发行记录与签名绑定。
3. **四元工程权衡**：水印系统始终在**载荷容量（Payload Capacity）**、**不可见性（Imperceptibility，以 PSNR/SSIM/LPIPS 度量）**、**抗扰动鲁棒性（Robustness）**与**推理部署成本（Compute & Deployability）**四个维度之间折中。论文在训练期通过质量损失权重 $\alpha_{\max}$ 调节工作点（衍生出偏鲁棒的 `B` 与偏画质的 `Q`），而在推理期通过残差缩放系数 $\lambda$（对应代码中的 `WM_STRENGTH`）提供单图微调旋钮。

## 2. 整体架构与端到端形式化

![TrustMark 嵌入、扰动、提取与移除架构](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig01-architecture.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 1, arXiv non-exclusive distribution license.*

如 Fig. 1 所示，TrustMark 将“水印写入与提取”（左半部）和“水印移除”（右半部）拆分为两个独立训练的阶段，但二者共享同一对预训练好的嵌入器 $E$ 与提取器 $X$。

在内部固定分辨率 $256\times256$ 下，设原始宿主图像（cover image）为 $\mathbf x\in\mathbb R^{256\times256\times3}$，待嵌入的长度为 $l$ 的二进制水印向量为 $\mathbf w\in\{0,1\}^{l}$。前向训练链路依次执行以下变换：

$$
\mathbf y=E(\mathbf x,\mathbf w), \qquad \tilde{\mathbf y}=N(\mathbf y), \qquad \bar{\mathbf w}=X(\tilde{\mathbf y})\in[0,1]^l.
$$

其中，嵌入器 $E$ 输出带水印图像 $\mathbf y$，可微噪声层 $N$ 在训练期模拟几何与光度扰动，提取器 $X$ 通过末端的 Sigmoid 头预测每一位为 1 的概率 $\bar{\mathbf w}$。评测时以 $0.5$ 为阈值二值化并计算平均比特准确率（bit accuracy）。需要特别注意的是，平均比特准确率 $\operatorname{Acc}_{\mathrm{bit}}=p$ 并不等于整条消息无错恢复的概率：即便各位独立且 $p=0.97$，在无纠错码（ECC）时 100 位载荷全部正确的概率也仅有 $0.97^{100}\approx 4.8\%$。因此，论文报告无 ECC 的原始比特准确率是为了纯粹衡量物理信道质量，而工业部署必须叠加 BCH 等纠错编码。

### 2.1 早期图文融合与 MUNIT 风格嵌入器

为了让二维连续图像在每个空间位置都能充分感知离散比特，预处理模块 $E_{\mathrm{pre}}$ 先将一维向量 $\mathbf w$ 空间投射并上采样至与图像相同的 $256\times256$ 分辨率，与 RGB 通道拼接后经 $3\times3$ 卷积生成 $d$ 通道融合特征：

$$
E_{\mathrm{pre}}(\mathbf x,\mathbf w) \in\mathbb R^{256\times256\times d}.
$$

这种早期融合（Early Fusion）使网络从第一层起就能根据局部纹理复杂度动态分配残差强度，并在全图范围内散布冗余编码以抵御局部裁切。

嵌入器主体采用经过定制改造的 MUNIT U-Net 架构（激活函数为 SiLU）：一方面**彻底移除了通道归一化层（Channel/Instance Normalization）**，避免归一化操作抹平水印赖以生存的微弱色彩与频谱偏移；另一方面加深网络并缩减内部通道数，将参数规模压缩约一半。网络主体输出的多通道特征随后进入后处理模块 $E_{\mathrm{post}}$——由多层 $1\times1$ 卷积、SiLU 与末端 $\tanh$ 组成的逐像素通道重组头，将像素值约束回 $[-1,1]$。由于 $1\times1$ 卷积不混合邻域像素，它能够在将局部纹理、全局上下文与水印特征映射回 RGB 空间的同时，避免末端大卷积核带来高频模糊（Table 4 消融表明，单独引入 $E_{\mathrm{post}}$ 即可将 PSNR 从 36.30 dB 提升至 37.91 dB）。

### 2.2 ResNet50 提取器与可微噪声模拟层

提取器 $X$ 采用带有残差连接的标准 ResNet50 骨干，将末端分类头替换为 $l$ 维 Sigmoid 输出。残差快捷连接不仅稳定了深层梯度回传，也使微弱的水印统计信号得以跨越多级下采样并在全局池化前汇聚多处局部证据。

为了迫使嵌入器将信号写入耐受分发变换的频段与空间结构，训练时在 $E$ 与 $X$ 之间串联了可微噪声模块 $N$。每张生成图像先依次通过 **3 类基础几何变换**（随机水平翻转、随机裁切、随机缩放），再从 **15 类可选光度与滤波扰动**（JPEG 可微近似、亮度、色相、对比度、锐化、色彩抖动、RGB 偏移、饱和度、灰度化、高斯模糊、中值模糊、方框模糊、运动模糊、高斯噪声、色调分离 Posterize）中随机抽取 **2 类**叠加。由于整条扰动链均可导，恢复损失的梯度能够直接穿透 $X$ 与 $N$ 指导 $E$ 的空间-频域分配。

## 3. 联合损失函数与四阶段课程学习调度

### 3.1 比特恢复损失与四重视觉质量约束

TrustMark 的端到端训练目标在图像视觉质量 $\mathcal L_{\mathrm{quality}}$ 与水印恢复交叉熵 $\mathcal L_{\mathrm{recovery}}$ 之间加权平衡：

$$
\mathcal L_{\mathrm{total}} = \alpha\mathcal L_{\mathrm{quality}}(\mathbf x,\mathbf y) + \mathcal L_{\mathrm{recovery}}(\mathbf w,\bar{\mathbf w}),
$$

其中恢复损失为逐比特二元交叉熵：

$$
\mathcal L_{\mathrm{recovery}} = -\frac1l\sum_{j=1}^{l} \left[w_j\log \bar w_j+(1-w_j)\log(1-\bar w_j)\right], \qquad \operatorname{Acc}_{\mathrm{bit}} = \frac1l\sum_{j=1}^{l} \mathbb 1\left[(\bar w_j>0.5)=w_j\right].
$$

视觉质量损失则由四个互补项线性组合而成（实验默认权重为 $\beta_{\mathrm{YUV}}=1.5,\ \beta_{\mathrm{LPIPS}}=1,\ \beta_{\mathrm{FFL}}=1.5,\ \beta_{\mathrm{GAN}}=1$）：

$$
\mathcal L_{\mathrm{quality}} = \beta_{\mathrm{YUV}}\mathcal L_{\mathrm{YUV}} + \beta_{\mathrm{LPIPS}}\mathcal L_{\mathrm{LPIPS}} + \beta_{\mathrm{FFL}}\mathcal L_{\mathrm{FFL}} + \beta_{\mathrm{GAN}}\mathcal L_{\mathrm{GAN+GP}}.
$$

这四项损失分别锁住了不同维度的视觉伪影：

1. **YUV 空间均方误差**：$\mathcal L_{\mathrm{YUV}} = \left\|\operatorname{YUV}(\mathbf x)-\operatorname{YUV}(\mathbf y)\right\|_2^2$。将亮度（Y）与色度（UV）分离度量，契合人类视觉系统对亮度失真与色偏的不同敏感度。
2. **LPIPS 感知特征距离**：利用预训练视觉网络的多层特征距离约束局部结构与语义纹理一致性，允许人眼不敏感的高纹理区容纳略大的像素改动。
3. **焦点频率损失（Focal Frequency Loss, FFL）**：直接约束二维傅里叶频谱差异

   $$
   \mathcal L_{\mathrm{FFL}}(\mathbf x,\mathbf y) = \rho_{f(\mathbf x),f(\mathbf y)} \left\|f(\mathbf y)-f(\mathbf x)\right\|_2, \qquad \rho_{f(\mathbf x),f(\mathbf y)} \in\mathbb R^{256\times256\times3},
   $$

   其中动态权重矩阵 $\rho$ 自动放大当前最难对齐的频率分量，抑制嵌入器在特定方向频带上留下刺眼的周期性亮纹（单独引入可提升约 1.7 dB PSNR）。
4. **WGAN-GP 对抗损失**：通过带梯度惩罚的判别器约束生成图像的整体统计分布贴近真实自然图像：

   $$
   \mathcal L_{\mathrm{GAN+GP}}(\mathbf x,\mathbf y) = \mathbb E_{\mathbf y\sim\mathbb P_E}[D(\mathbf y)] - \mathbb E_{\mathbf x\sim\mathbb P_{\mathrm{real}}}[D(\mathbf x)] + \lambda_{\mathrm{GP}} \mathbb E\left[\left(\left\|\nabla D(\cdot)\right\|_2-1\right)^2\right].
   $$

### 3.2 四阶段自动触发训练状态机

在移除归一化层并引入强噪声与对抗损失后，若从第 0 步起就同时开启全部约束，提取器尚未学会识别微弱信号，高权重的质量损失便会迫使嵌入器退化为恒等映射，导致恢复梯度彻底消失。为此，补充材料 Table 5 设计了基于验证集比特准确率自动跃迁的四阶段课程学习（Curriculum Learning）：

![TrustMark 四阶段训练调度](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-table05-training-stages.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Table 5, arXiv non-exclusive distribution license.*

| 阶段 | Fixed Batch | Random Batch | 可微噪声 $N$ | $\alpha$ 线性爬升 | WGAN-GP | 跃迁触发阈值（Bit Acc.） |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| Stage 0 | 开启 | 关闭 | 关闭 | 关闭（固定 $\alpha=0.05$） | 关闭 | 初始冷启动阶段 |
| Stage 1 | 关闭 | 开启 | 关闭 | 关闭（固定 $\alpha=0.05$） | 关闭 | 突破 $0.90$ 时进入 |
| Stage 2 | 关闭 | 开启 | 开启 | 关闭（固定 $\alpha=0.05$） | 关闭 | 突破 $0.95$ 时进入 |
| Stage 3 | 关闭 | 开启 | 开启 | 开启（10k 步爬升至 $\alpha_{\max}$） | 开启 | 突破 $0.98$ 时进入 |

整个状态机遵循“先打通信道、再抗噪、最后压低可见度”的两时间尺度逻辑：Stage 0 固定同一批宿主图、仅随机采样水印向量，让提取器在数百步内快速破除冷启动；准确率越过 $0.90$ 后切入全量随机图像（Stage 1）；越过 $0.95$ 后接入 3+2 可微噪声训练抗干扰能力（Stage 2）；当含噪准确率达到 $0.98$ 后（典型 TrustMark-B 约在 800 步左右即可依次走完前三阶段），正式进入 Stage 3，开启 WGAN-GP 并在 10,000 步内将 $\alpha$ 从 $0.05$ 线性拉升至目标上限 $\alpha_{\max}$（TrustMark-B 取 $20$，TrustMark-Q 取 $27.5$）。

## 4. 任意分辨率残差缩放与 TrustMark-RM 重水印机制

### 4.1 Algorithm 1：Residual Resolution Scaling

如果直接将 $256\times256$ 网络输出的图像插值放大到 2K 或 4K，原始高分辨率图像的高频纹理、文字边缘与发丝细节会在下采样瓶颈中永久丢失。Algorithm 1 提出的残差分辨率缩放（Residual Resolution Scaling）巧妙地绕开了这一瓶颈：

![TrustMark 任意分辨率残差缩放算法](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-algorithm01-resolution-scaling.webp)

*Source: Bui et al., arXiv:2311.18297v1, Algorithm 1, arXiv non-exclusive distribution license.*

给定任意尺寸的原始图像 $\mathbf x_{H,W}$，算法先将其归一化至 $[-1,1]$ 并下采样为固定尺寸输入 $\bar{\mathbf x}=\operatorname{Resize}(\mathbf x,256,256)$；接着在 $256^2$ 分辨率上运行嵌入器（或移除器 $R$）提取低分辨率残差：

$$
\mathbf r_{\mathrm{enc}} = E(\bar{\mathbf x},\mathbf w)-\bar{\mathbf x}, \qquad \mathbf r_{\mathrm{rm}} = R(\bar{\mathbf x})-\bar{\mathbf x}.
$$

随后，仅将该低幅残差双线性/双三次插值回原始分辨率 $\mathbf r_{H,W}=\operatorname{Resize}(\mathbf r,H,W)$，并叠加到未经任何降质的原始高分辨率图像上（支持引入运行时强度旋钮 $\lambda\in[0.75,1.50]$）：

$$
\mathbf y = \operatorname{clamp}\left(\mathbf x + \lambda\mathbf r_{H,W},\ -1,\ 1\right).
$$

由于原始高分辨率像素全程作为基底保留，网络预测的只是平滑分布的微弱增量，从而在固定计算开销下保住了高分辨率画质。当然，这一黑盒包装也决定了水印信号的空间频率上限仍受 $256^2$ 网格制约：面对极端细长比（如长条横幅）或超大尺寸画布时，直接全局缩放会导致信号过度拉伸或扭曲，因此后续开源代码额外补充了长宽比裁剪、中心方形区域写入与边缘羽化策略。

### 4.2 TrustMark-RM 与多轮重水印（Re-watermarking）

当数字资产在不同创作者或平台间流转时，往往需要更新水印标识。如果直接在带水印图像上反复调用 $E$ 写入新载荷，历次残差会不断叠加，导致 PSNR 迅速恶化。为此，论文基于图像恢复骨干网络 **KBNet** 训练了专用的水印移除器 **TrustMark-RM**（记为 $R$）。

在训练 $R$ 时，嵌入器 $E$ 与提取器 $X$ 全程冻结；每张宿主图像随机采样 $n=3$ 个不同的水印向量生成带水印样本 $\mathbf y_i=E(\mathbf x,\mathbf w_i)$，迫使 $R$ 学习消除该嵌入器族的共性残差分布，而非记忆单一载荷。移除器的优化目标由三部分组成：

$$
\mathcal L_{\mathrm{MSE}}^R = \left\|R(\mathbf y)-\mathbf x\right\|_2^2, \qquad \mathcal L_{\mathrm{GAN+GP}}^R, \qquad \mathcal L_X^R = \left\|X(R(\mathbf y))-X(\mathbf x)\right\|_2^2.
$$

其中，像素重建项 $\mathcal L_{\mathrm{MSE}}^R$ 与对抗项 $\mathcal L_{\mathrm{GAN+GP}}^R$ 保证恢复后的图像在像素与统计分布上逼近原图，而提取器响应对齐项 $\mathcal L_X^R$ 则直接压迫恢复图在冻结提取器 $X$ 上的输出退化为与无水印原图一致的随机响应。借此，多轮重水印得以按“先擦除、再重编码”的迭代公式运行：

$$
\mathbf y^{(k+1)} = E\left(R(\mathbf y^{(k)}),\ \mathbf w_{k+1}\right).
$$

由于 $E$ 包含非线性变换与截断操作，$R$ 学习的只是条件期望意义下的近似逆映射，每轮擦除仍会残留微小的恢复偏差（restoration bias）。因此，在有条件保存无水印母版（master asset）的生产系统中，始终应优先从母版派生新版本；仅在无法获取母版的流转环节才启用 `remove-then-encode` 链路。

## 5. 主实验、多维鲁棒性与重水印评估

论文使用 MIRFlickr-1M 中的 10 万张图像训练（1,000 张验证），在单张 RTX 3090 上训练 TrustMark 150 个 epoch 约需 48 小时（推理平均编码 125 ms、解码 25 ms），而在 A100 上训练 TrustMark-RM 100 个 epoch 约需两周。评测集涵盖 CLIC、DIV2K（800 张训练 + 100 张验证，分辨率高、纹理最复杂）与 MetFace（人脸方形图）。

### 5.1 三大基准主结果与残差可视化（Table 1、Fig. 2、Table 2）

![TrustMark 与五个 baseline 的主结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table01-main-benchmarks.webp)

*Source: Bui et al., arXiv:2311.18297v1, Table 1, arXiv non-exclusive distribution license.*

Table 1 在原始输出分辨率下对比了 TrustMark-Q/B 与五种代表性基线（为公平起见，固定分辨率基线同样套用了 Resolution Scaling）：

| 方法 | CLIC PSNR (dB) | DIV2K PSNR (dB) | MetFace PSNR (dB) | CLIC Noised Acc. | DIV2K Noised Acc. | MetFace Noised Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TrustMark-Q** | **43.26** | 42.39 | **45.34** | 0.95 | 0.95 | 0.96 |
| **TrustMark-B** | 41.53 | 40.20 | 43.87 | **0.97** | **0.97** | **0.97** |
| RoSteALS | 30.03 | 27.95 | 33.77 | 0.94 | 0.93 | 0.93 |
| RivaGAN | 41.04 | 41.06 | 40.98 | 0.79 | 0.78 | 0.82 |
| SSL Watermarking | 42.74 | **42.73** | 42.84 | 0.60 | 0.57 | 0.70 |
| StegaStamp | 37.48 | 35.87 | 39.35 | 0.72 | 0.70 | 0.72 |
| dwtDctSvd | 39.13 | 38.02 | 41.14 | 0.52 | 0.51 | 0.52 |

从表中可以清晰看到质量与鲁棒性的帕累托边界：传统频域方法 `dwtDctSvd` 与自监督特征水印 `SSL` 虽然能拿到 38～42 dB 的高 PSNR，但在复合几何与光度扰动下比特准确率分别跌至 $0.51\sim0.52$（等同随机猜测）和 $0.57\sim0.70$；`RoSteALS` 虽具备 $0.93\sim0.94$ 的抗噪恢复率，却受限于冻结自编码器的重建瓶颈，PSNR 停留在 28～33 dB。相比之下，`TrustMark-B` 在维持 40～43.8 dB 画质的同时将三组含噪准确率推高至 **0.97**，`TrustMark-Q` 则以约 1～2% 的准确率让渡换取了 42.4～45.3 dB 的高保真画质。

![TrustMark 与 baseline 的定性水印和残差比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig02-qualitative-comparison.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 2, arXiv non-exclusive distribution license.*

Fig. 2 的 20 倍残差放大图直观解释了数值背后的视觉差异：部分基线在平滑背景或物体边缘处留下了明显的色块与轮廓伪影，而联合了 $E_{\mathrm{post}}$、FFL 与感知损失的 TrustMark 残差呈现出均匀细密的自适应纹理分布。

![TrustMark 在 ImageNet-C 噪声配置下的对比](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table02-imagenet-c.webp)

*Source: Bui et al., arXiv:2311.18297v1, Table 2, arXiv non-exclusive distribution license.*

为了排除“自定义 18 类训练噪声对本方法更有利”的潜在偏差，Table 2 切换至 RoSteALS 所用的 ImageNet-C 噪声协议并关闭 Resolution Scaling（取 $\alpha_{\max}=15$）。在该外部协议下，TrustMark 依然取得 **38.87 dB PSNR** 与 **0.95 含噪准确率**，显著优于同协议下的 RoSteALS（32.68 dB / 0.94）与 StegaStamp（31.26 dB / 0.88）。

### 5.2 容量扩展、抗攻击成本与分项扰动剖析（Fig. 4 & Fig. 5）

![TrustMark payload 长度、噪声强度和对抗攻击三面板](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig04-payload-noise-adversarial.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 4, arXiv non-exclusive distribution license.*

Fig. 4 沿着三个关键压力轴揭示了模型的退化规律：

1. **载荷容量（左图）**：当嵌入位数从 32 bits 扩展约 6 倍至 200 bits 时，受制于有限的图像冗余空间，PSNR 下降约 7.5 dB，含噪比特准确率下降约 11%。
2. **训练噪声强度（中图）**：完全关闭噪声模拟时模型可达到 53.2 dB 的极高 PSNR，但毫无抗干扰能力；将训练噪声提升至 High 档位后，模型主动牺牲约 13 dB 画质（降至 40.2 dB）以换取稳健的跨扰动恢复，比特准确率仅衰减约 3%。
3. **I-FGSM 白盒梯度攻击成本（右图）**：在 $\epsilon=8/255$ 的白盒迭代攻击下，无噪声训练的模型在 50 步内几乎全军覆没；而经过高强度可微噪声训练的模型，有 32% 的样本需要超过 3,000 次梯度迭代才能将准确率压至阈值以下。

![TrustMark 对十五类单独扰动的恢复率](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig05-individual-perturbations.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 5, arXiv non-exclusive distribution license.*

Fig. 5 进一步拆解了 15 类单项扰动下的恢复表现。除高斯噪声与方框模糊（Box Blur）两项略逊于 RoSteALS 外，TrustMark-Q 在其余所有几何与色彩变换上均保持领先，避免了像 `dwtDctSvd` 遇上 RGB 偏移即跌至 0.5 随机水平那样的单点崩溃。在实际生产验收中，同样应当建立此类分项与复合渠道回放矩阵，而非仅凭单一平均准确率签发 SLA。

### 5.3 水印移除与 20 轮重水印实验（Table 3、Fig. 3、Fig. 7）

![TrustMark-RM 水印移除结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table03-removal.webp)

*Source: Bui et al., arXiv:2311.18297v1, Table 3, arXiv non-exclusive distribution license.*

Table 3 对比了专用恢复网络 TrustMark-RM 与 I-FGSM 白盒优化攻击在 DIV2K（针对 TrustMark-B）上的移除效果：I-FGSM 虽然能把解码准确率压到 $0.629$，但代价是将画质破坏至 23.48 dB PSNR / 0.613 SSIM；而 **TrustMark-RM 在将比特准确率压低至 0.553（接近 0.5 随机水平）的同时，交出了 48.48 dB PSNR 与 0.997 SSIM 的高保真恢复图像**（残余的 0.053 偏离主要源于高分辨率残差插值带来的微弱残留）。

![有无 TrustMark-RM 时多轮重水印曲线](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig03-rewatermarking-curves.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 3, arXiv non-exclusive distribution license.*

![TrustMark 一次、五次和十次重水印示例](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig07-rewatermarking-example.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 7, arXiv non-exclusive distribution license.*

Fig. 3 与 Fig. 7 追踪了连续 20 轮写入不同随机载荷的累积效应：无论是否使用移除器，最新一轮写入的载荷都能以接近 1.0 的准确率被读出；但若直接覆盖写入，图像 PSNR 会随轮数增加急剧滑坡，而在每次重编码前先经过 TrustMark-RM 擦除旧信号，则能在 5～10 轮迭代内大幅延缓画质劣化。

## 6. 消融实验、频谱分析与补充材料深读

### 6.1 模块消融、自适应强度权衡曲线与高纹理局限（Table 4、Fig. 6、Fig. 8）

![TrustMark 架构与损失消融](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table04-ablation.webp)

*Source: Bui et al., arXiv:2311.18297v1, Table 4, arXiv non-exclusive distribution license.*

Table 4（在 DIV2K、$\alpha_{\max}=20$ 下评测）验证了三大质量组件与提取器骨干的协同作用：当 GAN、FFL 与 $E_{\mathrm{post}}$ 全部关闭时，基线 PSNR 为 36.30 dB、含噪准确率为 0.955；单独启用 GAN、FFL、$E_{\mathrm{post}}$ 分别将 PSNR 提升至 36.99 dB、38.16 dB 与 37.91 dB；**三者联合启用时 PSNR 跃升至 40.30 dB，含噪准确率同步升至 0.977**。提取器一侧的消融则证实，无残差连接的普通 CNN 无法在四阶段课程中稳定收敛，而 ResNet50 在模型容量与提取精度之间取得了最佳平衡。

![TrustMark 训练损失权重对 PSNR 与恢复率的影响](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig06-quality-robustness-tradeoff.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 6, arXiv non-exclusive distribution license.*

![TrustMark 在密集纹理和杂乱图像上的局限](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig08-limitations-clutter.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 8, arXiv non-exclusive distribution license.*

Fig. 6 扫描了训练期权重上限 $\alpha_{\max}\in(0,30)$ 的连续工作曲线：随着 $\alpha_{\max}$ 增大至 30，PSNR 提升超过 8 dB，而含噪比特准确率仅回落不到 5%，`B`（20）与 `Q`（27.5）正是该曲线上选取的两个代表性锚点。Fig. 8 则坦诚展示了方法在毛发、密集细线与杂乱植被等高频杂波图像（cluttered images）上的退化：密集高频纹理虽然有利于人眼视觉掩蔽，但也会与水印信号争夺高频通道，导致局部残差放大并拉低提取准确率。

### 6.2 频谱伪影抑制与跨分辨率缩放实证（Supp. Fig. 9–11 & Fig. 16）

![TrustMark 加入 FFL 前后的平均频谱残差](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig09-frequency-ffl.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 9, arXiv non-exclusive distribution license.*

补充材料 Fig. 9 展示了 MetFace 上的平均傅里叶频谱残差：未加 FFL 时，残差在十字轴四个方向频带上呈现出明显的规律性亮斑；引入 FFL 动态频域加权后，能量分布显著均匀化，从频域机理上解释了 PSNR 的提升。

![TrustMark 水印在不同目标分辨率下的插值比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig10-resolution-scaling-watermarking.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 10, arXiv non-exclusive distribution license.*

![TrustMark-RM 在不同目标分辨率下的插值比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig11-resolution-scaling-removal.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 11, arXiv non-exclusive distribution license.*

![水印图像 PSNR 随评测分辨率变化](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig16-psnr-resolution-dependence.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 16, arXiv non-exclusive distribution license.*

补充 Fig. 10 与 Fig. 11 提供了支撑“任意分辨率残差缩放”最核心的定量曲线：随着目标分辨率向 DIV2K 原尺寸（2K）逼近，若直接将网络输出的 $256^2$ 整图做双线性或双三次放大，PSNR 会因高频细节丢失而断崖式暴跌；而仅插值放大残差的 Resolution Scaling 在整个分辨率跨度上 PSNR 波动不足 0.1 dB（比特准确率仅比双三次整图插值低约 1 个百分点），且这一保真优势在水印移除器 TrustMark-RM（Fig. 11）上同样成立。此外，补充 Fig. 16 揭示了一个常被忽视的评测陷阱：对带水印图像做下采样会平滑掉高频残差，导致同一张 `dwtDctSvd` 水印图在不同评测分辨率下算出的 PSNR 相差高达 **4.7 dB**，印证了在原始高分辨率下统一计算指标的必要性。

### 6.3 内部激活图、极简控制图与运行时强度（Supp. Fig. 12–15 & Table 6）

![TrustMark 嵌入器预处理与后处理激活图](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig12-activation-maps.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 12, arXiv non-exclusive distribution license.*

![TrustMark 在纯色、几何纹理和条纹控制图上的结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig13-control-images.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 13, arXiv non-exclusive distribution license.*

补充 Fig. 12 的通道激活图表明，网络在早期融合后同时编码了物体边缘轮廓与全局低幅载波模式，再由 $E_{\mathrm{post}}$ 聚合成自适应 RGB 残差；而补充 Fig. 13 在纯色块、棋盘格与规则条纹等极简合成图（control images）上依然实现了 100% 的无噪比特恢复，说明提取器并非依赖自然图像语义先验才能解码。

![TrustMark 低中高三档噪声模拟参数](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-table06-noise-severity.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Table 6, arXiv non-exclusive distribution license.*

![TrustMark 训练时不同噪声来源的视觉示例](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig14-noise-examples.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 14, arXiv non-exclusive distribution license.*

![TrustMark 运行时残差系数对 PSNR 与 bit accuracy 的影响](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig15-inference-strength.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 15, arXiv non-exclusive distribution license.*

补充 Table 6 与 Fig. 14 完整列出了 Low / Medium / High 三档可微噪声的参数采样区间与视觉效果（训练期对每张图固定执行 3 项几何基础变换 + 随机 2 项可选扰动）。补充 Fig. 15 则给出了推理期残差缩放系数 $\lambda$ 的单图调节曲线：在不重训模型的前提下，在 $\lambda\in[0.75,1.50]$ 区间内微调残差幅值，即可平滑交换 PSNR 与解码准确率，构成了开源库中 `WM_STRENGTH` 参数的算法依据。

## 7. 官方源码架构与工程演进走读

对比 2023 年的 arXiv v1 论文，官方仓库 [`adobe/trustmark`](https://github.com/adobe/trustmark/tree/0ed40cbe8188f664fd9cbbeacd969807de27440a) 在工程化过程中扩展了变体矩阵、纠错编码、区域裁剪与多语言运行时：

```text
TrustMark(...)
  -> DataLayer(secret_len=100, encoding_type=...)
  -> check_and_download(): 拉取 YAML/ckpt 并校验 MD5
  -> encode(): 业务载荷 -> 100-bit BCH 数据包 -> U-Net 编码 -> 零均值校正与残差缩放 -> 边缘羽化贴回 PIL
  -> decode(): 区域裁剪/可选旋转或 bbox 定位 -> ResNet 解码 logits -> BCH 纠错 -> 还原业务载荷
  -> remove_watermark(): KBNet 预测反向残差 -> 残差缩放 -> 贴回 PIL
```

### 7.1 核心推理主链与网络实现（`trustmark.py`、`model.py`、`unet.py`）

1. **端到端 API 与残差零均值化（[`trustmark.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L73-L527)）**：构造器负责初始化 `DataLayer` 并按需拉取编码器、解码器、移除器与边界框检测器权重（通过 [`check_and_download()`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L28-L67) 校验远端模型 MD5）。在 [`encode()`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L465-L507) 中，代码在论文 Algorithm 1 的基础上增加了一步工程细节：**按 RGB 通道减去残差均值**（zero-mean adjustment）以消除全图整体色偏，随后按 `WM_MERGE` 插值放大、乘上 `WM_STRENGTH` 并在长宽比超限时配合 1% 带宽（上限 50 像素）的边缘羽化（feather paste）贴回中心区域。
2. **Lightning 训练容器与按需裁剪加载（[`model.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/model.py#L31-L133)）**：`TrustMark_Arch` 继承自 PyTorch Lightning，内部保留了 `fixed_input`、`bit_acc_thresholds=[0.9, 0.95, 0.98]` 与手动双优化器更新逻辑，印证了补充 Table 5 的四阶段状态机设计。在推理加载时，`load_model(..., part='encoder')` 会将解码器、判别器、损失与噪声层全部替换为 `Identity`（加载解码器时反之），以便从同一个联合训练 checkpoint 中剥离出轻量推理子图。
3. **空间投影、无归一化 U-Net 与多骨干解码器（[`unet.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/unet.py#L239-L461)）**：[`Secret2Image`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/unet.py#L239-L251) 先用全连接层将 100-bit 向量映射为 $16\times16\times3$ 特征图再上采样至图像尺寸；[`Unet1`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/unet.py#L254-L308) 将 RGB 图像与 3 通道水印特征拼接为 6 通道，经四级 `norm='none'` 的编解码与跳跃连接后，在最后一级再次拼入原始 6 通道输入，并经 `3x3 -> 1x1 SiLU -> 1x1 tanh` 输出编码图；移除器则封装在 `python/trustmark/KBNet/` 与 `denoise.py` 中。

### 7.2 四种模型变体（B / Q / C / P）与 BCH 数据层契约（`datalayer.py`）

开源仓库在论文 `B`、`Q` 的基础上扩展了面向轻量部署的 `C` 与面向高感知质量的 `P`（四个变体权重互不兼容，解码端必须使用对应模型族）：

| 模型变体 | 论文 v1 覆盖 | 典型 PSNR（源码注释口径） | 架构与预处理特性 |
| --- | :---: | :---: | --- |
| **B** (Balanced) | 是 | ~41 dB | ResNet50 解码器（输入 245），$\alpha_{\max}=20$，偏抗噪鲁棒性 |
| **Q** (Quality) | 是 | ~43 dB | ResNet50 解码器（输入 245），$\alpha_{\max}=27.5$，仓库默认变体 |
| **C** (Compact) | 否（后续新增） | ~39 dB | ResNet18 轻量解码器，面向低算力与端侧部署 |
| **P** (Perceptual) | 否（后续新增） | ~48 dB | 解码器输入 224，强化感知损失，强制中心方形区域裁剪且默认强度乘 $1.25$ |

在数据层（[`datalayer.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/datalayer.py#L14-L100)），模型底层固定的 100 bits 被划分为 **4 bits 版本与模式头**（2 位保留 + 2 位 Schema 标识）和 **96 bits 数据+BCH 校验区**：

| BCH Schema | 有效业务比特数 | 7-bit ASCII 字符数 | BCH 校验比特数 | 最大可纠正翻转位数 ($t$) | 版本头取值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BCH_SUPER` | 40 bits | 5 字符 | 56 bits | 8 bits | `0` |
| `BCH_5`（默认） | 61 bits | 8 字符 | 35 bits | 5 bits | `1` |
| `BCH_4` | 68 bits | 9 字符 | 28 bits | 4 bits | `2` |
| `BCH_3` | 75 bits | 10 字符 | 21 bits | 3 bits | `3` |

由于文本模式通过 `ord(t) & 127` 强制按 7-bit ASCII 截断打包，生产系统应优先使用二进制模式写入高熵随机生成的非透明标识符（opaque ID），避免直接写入 UTF-8 字符或明文用户隐私（PII）。

### 7.3 跨语言运行时（Python / JS / Rust）与 C2PA 软绑定集成

官方仓库提供了三套能力边界各异的语言实现，并通过 [`c2pa_watermark_example.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/c2pa/c2pa_watermark_example.py) 演示了与 C2PA 内容凭证的衔接：

| 能力维度 | Python 参考实现 | JavaScript (`onnxruntime-web`) | Rust Crate (`ort`) |
| --- | --- | --- | --- |
| 水印写入（Encode） | 完整支持（含文本/二进制与区域羽化） | 不支持 | 支持（仅二进制子集） |
| 水印提取（Decode） | 完整支持（含旋转重试与 bbox 定位） | 支持（WebGPU / WASM） | 支持 |
| 水印移除（Remove） | 完整支持（KBNet） | 不支持 | 不支持 |
| 模型格式与依赖 | PyTorch `.ckpt`（运行时按需下载） | ONNX | ONNX |

在跨语言协同（例如服务端 Rust/Python 写入、浏览器端 JS 解码）时，必须使用标准测试图（Golden Images）严格对齐 Resize 插值核、色彩通道顺序、P 变体的 224 方形裁剪规则、BCH 比特序以及边界载荷。而在接入 C2PA 体系时，水印的信任传递链条为：

```text
图像像素 -> TrustMark 解码器 -> 61-bit 随机标识符 -> 检索可信 Manifest 仓库 -> 验证密码学签名与证书链 -> 确认溯源声明
```

像素中的水印仅承担前两步的“抗剥离索引”职责；一旦脱离外部签名验证与资产哈希比对，单纯解出一段合法 BCH 比特流绝不构成来源真实性证明。

## 8. 生产落地指南、安全防御与总结

将 TrustMark 纳入生产级内容溯源管线时，需要在工程、安全与合规层面落实以下规范：

1. **供应链固化与图像预处理契约**：在构建容器镜像阶段预先拉取模型配置与权重并锁定 SHA-256 摘要，禁止线上请求触发首次 `urlretrieve` 网络阻塞；在 API 外层统一处理 EXIF 旋转方向、CMYK/16-bit/RGBA 色彩空间转换与 ICC 配置文件，并在最终落盘的 JPEG/WebP 字节流上执行回读解码校验。
2. **对抗安全与多解码器假阳性控制**：公开解码器意味着不能将算法视为保密信道。对外检测接口应仅返回最小化的状态与溯源查询结果，严禁向未授权调用者暴露逐比特原始置信度（防止被用作黑盒梯度擦除的 Oracle）；当系统串联尝试多个模型变体（B/Q/C/P）、多个旋转角度与多个局部检测框时，多重假设检验会显著推高随机图像的 BCH 碰撞假阳性率，必须建立基于置信度阈值与内容哈希二次核验的过滤机制。
3. **隐私保护与生命周期治理**：切勿将用户 ID、邮箱或设备号直接编码进不可见水印。应当在像素中仅写入高熵随机索引，在服务端映射表中实施多租户隔离、访问审计与过期撤销（Revocation）——当用户发起数据删除或撤销请求时，直接注销服务端映射记录与 C2PA 清单状态，而无需依赖有损的 `TrustMark-RM` 去修改已分发到公网的历史图片副本。
4. **方法局限性审视**：
   - **空间频率与极端画幅受限**：内部网络固定在 $256^2$，极窄长图、超小局部目标或多图层拼贴场景下的鲁棒性仍依赖启发式裁剪与检测补丁；
   - **跨域分布泛化未充分验证**：训练集中于 MIR-Flickr 自然照片，面对高频杂波图像（Fig. 8）已有退化，扩展至医学影像、线稿动漫、UI 截图或广色域 HDR 资产前需独立标定；
   - **缺乏生成式重绘鲁棒性**：作为像素级后处理水印，它主要抵御常规信号处理变换，若图像经过扩散模型图生图（Img2Img）重绘或专用恢复网络擦除，水印信号仍会被大幅削弱。

综上所述，TrustMark 通过“无归一化 MUNIT 编码 + 频谱/感知/对抗联合约束 + 四阶段课程学习 + 任意分辨率残差缩放”，为高分辨率数字资产提供了一套兼顾画质与工程可用性的软绑定水印底座；将其准确地定位于 C2PA 密码学凭证体系下的**耐久检索层**，方能最大化发挥其工程价值。

## 参考资料

- [TrustMark arXiv abstract, v1](https://arxiv.org/abs/2311.18297v1)
- [TrustMark arXiv PDF, v1](https://arxiv.org/pdf/2311.18297v1)
- [TrustMark arXiv TeX source, v1](https://arxiv.org/e-print/2311.18297v1)
- [TrustMark ICCV 2025 paper page](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html)
- [adobe/trustmark official repository](https://github.com/adobe/trustmark)
- [Pinned source commit `0ed40cbe`](https://github.com/adobe/trustmark/tree/0ed40cbe8188f664fd9cbbeacd969807de27440a)
- [TrustMark on PyPI](https://pypi.org/project/trustmark/)
- [C2PA specification and ecosystem](https://c2pa.org/)
- [HiDDeN: Hiding Data With Deep Networks](https://arxiv.org/abs/1807.09937)
- [StegaStamp: Invisible Hyperlinks in Physical Photographs](https://arxiv.org/abs/1904.05343)
- [RoSteALS: Robust Steganography using Autoencoder Latent Space](https://arxiv.org/abs/2304.03400)
- [Focal Frequency Loss for Image Reconstruction and Synthesis](https://arxiv.org/abs/2012.12821)
- [KBNet: Kernel Basis Network for Image Restoration](https://arxiv.org/abs/2303.02881)

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

生成式图像进入日常创作与商业发行之后，内容溯源迅速演变为一项跨链路系统工程。文件头元数据（如 EXIF 或 C2PA JUMBF 清单）虽然能完整记录创作者身份、编辑历史与密码学签名，却极易在社交平台转码、即时通讯转发或屏幕截图中被彻底剥离；被动式 AI 图像检测器只能给出分布层面的概率判别，无法将一幅具体图像绑定到可审计的资产记录；而传统的最低有效位（LSB）或频域变换水印（DWT/DCT/SVD）又往往在裁切、缩放、色彩抖动与有损压缩的复合冲击下迅速失效。更棘手的工程矛盾在于：深度学习水印模型通常只能在 $256\times256$ 等固定低分辨率下训练，而摄影原片、商业海报与数字艺术资产动辄达到 2K 甚至 4K 分辨率，且横跨千差万别的长宽比与纹理统计。

Adobe Research 与萨里大学提出的 **TrustMark**（[arXiv:2311.18297](https://arxiv.org/abs/2311.18297v1)，后续扩展录用于 ICCV 2025；官方实现见 [`adobe/trustmark`](https://github.com/adobe/trustmark/tree/0ed40cbe8188f664fd9cbbeacd969807de27440a)），将不可见数字水印从固定小图上的学术实验推进到了面向生产管线的完整资产接口。通读全篇论文与开源代码，TrustMark 的核心突破可以提炼为以下四项**核心设计哲学与工程权衡**：

1. **信号保真与高频伪影抑制：去归一化 MUNIT、逐像素重组头与焦点频率损失**（FFL）：水印信号本质上是叠加在自然图像上的微弱残差。TrustMark 在 MUNIT 风格 U-Net 中**彻底移除了全部通道与实例归一化层**，防止归一化操作抹平微弱的跨通道编码偏移；在输出端接上由 $1\times1$ 卷积构成的逐像素后处理头 $E_{\mathrm{post}}$，避免 $3\times3$ 卷积核混合邻域像素造成边缘模糊；同时在训练目标中引入**焦点频率损失**（Focal Frequency Loss, FFL），动态惩罚二维傅里叶频谱中最难对齐的频率分量，消除了传统深度水印在水平与垂直频带上留下的周期性十字亮纹。
2. **破解端到端对抗训练冷启动：基于准确率的四阶段课程状态机**：如果从训练第 0 步起就同时开启 18 类可微噪声、WGAN-GP 对抗判别器与高权重的画质约束，尚未学会识别水印的提取器无法提供有效梯度，嵌入器会瞬间退化为恒等映射（直接输出原图以骗取零画质损失）。TrustMark 设计了由验证集比特准确率自动触发的四阶段课程状态机（阈值依次为 $0.90\to 0.95\to 0.98$），严格遵循“先固定样本打通信道、再放开随机图像、随后注入可微噪声、最后线性拉升画质与对抗权重”的两时间尺度演进逻辑。
3. **固定算力与任意高分辨率解耦：低分辨率残差缩放与可逆重水印**：为了不在 $256\times256$ 网络瓶颈中折损 4K 原图的高频发丝与文字边缘，TrustMark 提出了**残差分辨率缩放**（Residual Resolution Scaling）——网络仅在 $256^2$ 下预测低幅残差并执行按通道零均值校正，随后只将残差双线性插值回原始画幅并叠加到无损原图上，使跨分辨率 PSNR 波动控制在 0.1 dB 以内。针对资产跨平台流转时的多轮更新需求，论文配套训练了基于 KBNet 的水印移除器 **TrustMark-RM**，以“先擦除旧残差、再写入新载荷”的工作流抑制多轮叠加引起的画质崩塌。
4. **物理信道与密码学信任解耦：BCH 纠错封装与 C2PA 软绑定安全边界**：单比特准确率达到 $97\%$ 时，100 位无保护载荷整包无错恢复的概率仅有 $0.97^{100}\approx 4.8\%$。工程实现通过 4 档 BCH 纠错码（容忍 $3\sim 8$ 位翻转）将底层物理信道转化为可靠的数据包接口。更重要的是，在公开编解码器的威胁模型下，像素内解出的 61-bit 标识符只充当 C2PA 内容凭证体系中的**软绑定检索键**（Soft-Binding Lookup Key），真正的防伪信任依然锚定在外部密码学签名与托管清单仓库之上。

---

## 1. 内容溯源的四元工程权衡与安全边界

在深入网络结构之前，我们需要先厘清不可见数字水印在内容溯源体系中的确切位置。工程落地中的图像水印始终受制于四个维度的帕累托权衡：**载荷容量**（Payload Capacity，单图可承载的比特数）、**视觉不可见性**（Imperceptibility，以 PSNR、SSIM 与 LPIPS 衡量）、**抗扰动鲁棒性**（Robustness，经历几何裁切与光度滤波后的比特恢复率）以及**推理部署成本**（Compute & Deployability，能否以固定算力处理任意尺寸画布并运行于浏览器或边缘端）。

1. **软绑定标识符与相近技术的边界**：不可见水印将短比特载荷主动织入像素纹理，与旨在制造视觉威慑的**可见水印**、追求隐蔽大容量通信的**隐写术**（Steganography）、面向多用户抗合谋追踪的**数字指纹**（Fingerprinting）以及做被动分布判别的 **AI 生成检测器**各司其职。在与 [C2PA](https://c2pa.org/) 标准配合时，TrustMark 扮演的是耐久的**软绑定标识符通道**（soft-binding identifier）：当文件头的签名清单（manifest）在社交分发中被剥离后，验证端可凭像素中幸存的短随机 ID 去可信注册表中查回原始签名声明，而防篡改验证本身仍交由外部非对称密码学完成。
2. **良性分发退化与恶意对抗威胁模型**：论文主攻的是良性分发链路中的**经验鲁棒性**（缩放、裁切、翻转、JPEG 压缩、色彩调整、模糊与噪声）。而在恶意对抗视角下，攻击面可细分为**移除**（Removal，抹去水印使提取失败）、**伪造**（Spoofing，向无关图像写入合法水印）、**跨图转移**（Transfer，从带水印图中剥离残差贴入另一幅图）与**载荷替换**（Substitution，覆盖原有标识）。由于 TrustMark 的编码器、解码器与移除器权重完全公开且无私钥门控，任何持有模型的人都能向任意图像写入合法格式的水印包；因此，单纯从像素中解出一段合法 BCH 比特流绝不等于来源真实，必须结合服务端发行记录与资产内容哈希二次核验才能闭环。
3. **训练期与推理期的双层工作点调节**：为了适配不同业务对画质与鲁棒性的偏好，TrustMark 在训练期通过视觉质量损失上限权重 $\alpha_{\max}$ 锚定宏观工作点（衍生出偏重抗噪的 `TrustMark-B`、偏重高画质的 `TrustMark-Q`、轻量化的 `TrustMark-C` 以及极致感知质量的 `TrustMark-P`）；在推理期则暴露残差缩放系数 $\lambda$（对应源码中的 `WM_STRENGTH`），允许在不重训模型的前提下按单张图像的纹理复杂度微调嵌入强度。

---

## 2. 整体架构：早期图文融合与去归一化设计

![TrustMark 嵌入、扰动、提取与移除架构](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig01-architecture.webp)

如架构总览图所示，TrustMark 将整个系统拆分为两个分阶段训练的子任务：左半部是端到端联合训练的**水印嵌入器 $E$、可微噪声模拟层 $N$ 与提取器 $X$**；右半部则是固定 $E$ 与 $X$ 后独立训练的**水印移除网络 $R$**（即 TrustMark-RM）。

在内部固定的 $256\times256$ 工作分辨率下，设归一化至 $[-1,1]$ 的原始宿主图像（cover image）为 $\mathbf x\in\mathbb R^{256\times256\times3}$，待嵌入的长度为 $l$（默认 $l=100$）的二进制水印向量为 $\mathbf w\in\{0,1\}^{l}$。训练期前向链路依次执行三步变换：

$$
\mathbf y=E(\mathbf x,\mathbf w), \qquad \tilde{\mathbf y}=N(\mathbf y), \qquad \bar{\mathbf w}=X(\tilde{\mathbf y})\in[0,1]^l.
$$

其中，嵌入器 $E$ 输出带水印图像 $\mathbf y$，可微噪声层 $N$ 模拟分发链路中的几何与光度退化，提取器 $X$ 通过末端的 Sigmoid 激活预测每一位取 1 的后验概率 $\bar{\mathbf w}$。评测时以 $0.5$ 为阈值二值化并统计平均比特准确率（bit accuracy）。需要格外留心的是，单比特准确率 $\operatorname{Acc}_{\mathrm{bit}}=p$ 与整包无错恢复率之间存在指数鸿沟：即便各位相互独立且 $p=0.97$，在无纠错码（ECC）保护时 100 位载荷全部命中真值的概率也仅有 $0.97^{100}\approx 4.8\%$。论文主表汇报无 ECC 的原始比特准确率是为了纯粹度量底层物理信道容量，而生产部署必须叠加后文所述的 BCH 纠错层。

### 2.1 为什么采用早期融合、移除归一化并追加逐像素后处理头

如何把一维离散比特向量 $\mathbf w\in\{0,1\}^l$ 无痕地织入二维连续像素 $\mathbf x$，是深度水印编码器的第一个结构难题。如果只在 U-Net 的最深层瓶颈处注入比特向量，空间分辨率已经缩小了十余倍，网络很难根据局部边缘和纹理精细调节每个像素的残差幅值。

TrustMark 采取了**早期空间融合**（Early Fusion）策略。在源码 [`unet.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/unet.py#L239-L251) 的 `Secret2Image` 模块中，预处理网络 $E_{\mathrm{pre}}$ 先通过全连接层与 Reshape 将 100-bit 向量投射为 $16\times16\times3$ 的低维特征图，再经上采样放大到与图像完全对齐的 $256\times256\times3$ 空间网格，与原始 RGB 通道拼接为 6 通道张量后送入首层 $3\times3$ 卷积：

$$
E_{\mathrm{pre}}(\mathbf x,\mathbf w) \in\mathbb R^{256\times256\times d}.
$$

这样一来，编码器从第一层卷积起就能同时感知每个空间位置的原始像素与目标水印信号，从而在平坦背景区压低残差、在复杂纹理区加大冗余，并在全图范围内分散编码以抵御局部裁切。

在主干网络 [`Unet1`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/unet.py#L254-L308) 的设计上，作者对经典的 MUNIT 图像翻译架构做了三项极具针对性的改造：

1. **彻底移除通道与实例归一化**（`norm='none'`）：在风格迁移任务中，Instance Normalization 有助于抹除原图色彩统计以注入新风格；但在不可见水印任务中，水印信息恰恰隐藏在极其微弱的像素幅值与跨通道统计偏移里。实验发现，保留归一化层会直接洗掉早期融合注入的微弱比特信号，并引发全局色偏。因此，TrustMark 将四级编解码器中的归一化全部移除，统一改用平滑的 SiLU 激活函数，并加深网络层数、缩减内部通道宽度，使参数量比原始 MUNIT 减小约一半。
2. **末端原始输入重注入**：在四级 U-Net 解码回到 $256\times256$ 分辨率后，网络在最后一级再次将最初始的 6 通道图文拼接张量拼入特征图，确保深层上采样不会遗忘精细的宿主图纹理基底。
3. **逐像素通道重组后处理头 $E_{\mathrm{post}}$**：如果直接用一层 $3\times3$ 卷积输出最终的 RGB 图像，卷积核在空间邻域内的加权求和不可避免地会带来轻微的高频模糊，拉低 PSNR。为此，作者在主干末端专门追加了后处理模块 $E_{\mathrm{post}}$——由两层 $1\times1$ 卷积、SiLU 激活与末端 $\tanh$ 组成的逐像素通道重组头，将像素值约束回 $[-1,1]$。由于 $1\times1$ 卷积感受野严格限制在单个像素内部，它只负责在通道维度上将局部纹理、全局上下文与水印特征线性/非线性重组为 RGB 残差，绝不向邻近像素扩散模糊。后文表 4 的消融实验证实，仅凭 $E_{\mathrm{post}}$ 一项改动就能将 PSNR 从 36.30 dB 显著拉升至 37.91 dB。

### 2.2 ResNet50 提取器与 3+2 可微噪声模拟层

在解码一侧，提取器 $X$ 采用带有残差快捷连接（skip connections）的标准 ResNet50 骨干，仅将末端的全连接分类头替换为 $l$ 维 Sigmoid 输出。后文表 4 对比了无残差连接的普通 CNN、VGG16 与 ResNet18/50/101：普通 CNN 在四阶段课程中根本无法稳定收敛，因为经过高强度画质打压后的水印残差幅值极小，多层下采样后梯度极易消失；而 ResNet50 的残差通路让微弱的高频与中频统计证据得以跨层汇聚，在模型体积与解码精度之间取得了最佳平衡。

为了让嵌入器学会把信号写进真正耐受真实世界分发变换的频段，训练时在 $E$ 与 $X$ 之间串联了可微噪声模块 $N$。每张生成的带水印图像 $\mathbf y$ 在送入提取器前，都会实时经历一套随机组合的“**3 项基础几何变换 + 2 项随机光度/滤波扰动**”：

- **必选的 3 项几何变换**：随机水平翻转（训练提取器对镜像翻转保持不变性）、随机裁切与随机缩放（迫使水印信号在全图多尺度冗余分布）。
- **15 选 2 的光度与信号退化**：每次前向传播从 15 类可微算子中随机抽取 2 类串联叠加，涵盖可微 JPEG 压缩近似（训练期采用可导近似以传导梯度，测试期则切换为真实的 `PIL` 不可微 JPEG 编解码以防自欺欺人）、亮度、色相、对比度、锐化、色彩抖动、RGB 通道偏移、饱和度、灰度化、高斯模糊、中值模糊、方框模糊、运动模糊、高斯噪声与色调分离（Posterize）。

由于整条退化链路完全由可微张量算子拼装而成，提取端的交叉熵梯度能够无损穿透 $N$ 回传给嵌入器 $E$，逼迫 $E$ 自动避开那些一经 JPEG 量化或模糊滤波就会被抹除的脆弱高频角落。

---

## 3. 联合损失函数与四阶段课程学习状态机

### 3.1 比特恢复损失与四重互补视觉约束

TrustMark 的端到端训练目标在图像视觉质量损失 $\mathcal L_{\mathrm{quality}}$ 与水印恢复交叉熵 $\mathcal L_{\mathrm{recovery}}$ 之间寻求动态平衡：

$$
\mathcal L_{\mathrm{total}} = \alpha\mathcal L_{\mathrm{quality}}(\mathbf x,\mathbf y) + \mathcal L_{\mathrm{recovery}}(\mathbf w,\bar{\mathbf w}),
$$

其中恢复损失为标准的逐比特二元交叉熵（Binary Cross-Entropy）：

$$
\mathcal L_{\mathrm{recovery}} = -\frac1l\sum_{j=1}^{l} \left[w_j\log \bar w_j+(1-w_j)\log(1-\bar w_j)\right], \qquad \operatorname{Acc}_{\mathrm{bit}} = \frac1l\sum_{j=1}^{l} \mathbb 1\left[(\bar w_j>0.5)=w_j\right].
$$

如果仅仅用 RGB 空间的均方误差（MSE）作为画质约束，嵌入器往往会在平坦天空或人脸皮肤上留下人眼极易察觉的彩色斑块或周期性网格条纹。为此，TrustMark 将 $\mathcal L_{\mathrm{quality}}$ 拆解为四个分别镇守不同感知维度的互补损失项（默认权重配比为 $\beta_{\mathrm{YUV}}=1.5,\ \beta_{\mathrm{LPIPS}}=1,\ \beta_{\mathrm{FFL}}=1.5,\ \beta_{\mathrm{GAN}}=1$）：

$$
\mathcal L_{\mathrm{quality}} = \beta_{\mathrm{YUV}}\mathcal L_{\mathrm{YUV}} + \beta_{\mathrm{LPIPS}}\mathcal L_{\mathrm{LPIPS}} + \beta_{\mathrm{FFL}}\mathcal L_{\mathrm{FFL}} + \beta_{\mathrm{GAN}}\mathcal L_{\mathrm{GAN+GP}}.
$$

为什么这四项缺一不可？我们可以从人类视觉系统（HVS）与信号频域的弱点逐一剖析：

1. **YUV 空间均方误差 $\mathcal L_{\mathrm{YUV}} = \left\|\operatorname{YUV}(\mathbf x)-\operatorname{YUV}(\mathbf y)\right\|_2^2$**：人眼视网膜的视杆细胞对亮度（Y 通道）边缘极其敏锐，而视锥细胞对色度（U/V 通道）的空间分辨率则相对迟钝。在 YUV 色彩空间度量像素误差，能引导网络将能量优先藏匿于人眼不敏感的色度与复杂纹理分量中。
2. **LPIPS 感知特征距离 $\mathcal L_{\mathrm{LPIPS}}$**：像素级 $L_2$ 损失对全图一视同仁，但相同幅值的残差落在纯白背景上会非常刺眼，落在茂密树叶或毛发纹理上却完全不可见。LPIPS 借助预训练视觉骨干的多层特征图距离，实质上为嵌入器注入了**视觉掩蔽效应**（Visual Masking）先验，允许高语义复杂度区域容纳更大的水印能量。
3. **焦点频率损失**（Focal Frequency Loss, FFL）$\mathcal L_{\mathrm{FFL}}$：卷积上采样与周期性载波极易在二维傅里叶频谱的特定方向频带上激发出尖锐的能量峰，反映在空域就是贯穿全图的细密十字网格伪影。FFL 直接将宿主图与编码图变换到二维频域 $f(\mathbf x), f(\mathbf y)$，并计算动态加权的频谱欧氏距离：

   $$
   \mathcal L_{\mathrm{FFL}}(\mathbf x,\mathbf y) = \rho_{f(\mathbf x),f(\mathbf y)} \left\|f(\mathbf y)-f(\mathbf x)\right\|_2, \qquad \rho_{f(\mathbf x),f(\mathbf y)} \in\mathbb R^{256\times256\times3}.
   $$

   这里的频谱权重矩阵 $\rho$ 与当前频域误差幅值成正比——哪个频率分量偏离原图最严重，$\rho$ 就会自动放大该频率处的梯度惩罚，从而精准削平频谱尖峰（单独引入 FFL 即可使 PSNR 跃升 1.86 dB）。
4. **WGAN-GP 对抗损失 $\mathcal L_{\mathrm{GAN+GP}}$**：利用带梯度惩罚（Gradient Penalty）的判别器 $D$ 约束生成图像的全局自然统计分布：

   $$
   \mathcal L_{\mathrm{GAN+GP}}(\mathbf x,\mathbf y) = \mathbb E_{\mathbf y\sim\mathbb P_E}[D(\mathbf y)] - \mathbb E_{\mathbf x\sim\mathbb P_{\mathrm{real}}}[D(\mathbf x)] + \lambda_{\mathrm{GP}} \mathbb E\left[\left(\left\|\nabla D(\cdot)\right\|_2-1\right)^2\right].
   $$

### 3.2 为什么需要四阶段自动触发训练状态机

在移除了 U-Net 的所有归一化层、又叠加了 18 类强噪声与四重画质约束后，端到端训练面临一个致命的**冷启动死锁**：在训练刚开始的第 0 步，提取器 $X$ 的权重完全随机，根本无法从图像中读出任何比特信息，因此它回传给嵌入器 $E$ 的恢复梯度 $\nabla \mathcal L_{\mathrm{recovery}}$ 近似于漫无目的的随机游走；此时如果画质权重 $\alpha$ 很大或者判别器 $D$ 已经介入，嵌入器 $E$ 会立刻发现一条“捷径”——直接令输出等于原图 $\mathbf y = \mathbf x$，即可瞬间拿到完美的零画质损失。一旦 $E$ 坍缩为恒等映射，水印信号彻底归零，$X$ 就更不可能学到任何特征，整个系统永久卡死在 50% 的随机猜测水平。

![TrustMark 四阶段训练调度](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-table05-training-stages.webp)

为了解开这一死锁，补充材料表 5（以及源码 [`model.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/model.py#L31-L133) 中的 `bit_acc_thresholds=[0.9, 0.95, 0.98]`）设计了一套由验证集比特准确率自动驱动的**四阶段课程学习状态机**（Curriculum Learning）：

1. **Stage 0**（固定图像批次破冰，目标 $\operatorname{Acc}_{\mathrm{bit}} > 0.90$）：开启 `Fixed Batch` 模式，反复使用同一批固定的宿主图像（仅随机重采样水印向量 $\mathbf w$），关闭可微噪声层 $N$ 与 WGAN-GP，并将画质权重锁定在极低的 $\alpha=0.05$。在没有图像内容变化和噪声干扰的温室环境下，嵌入器被允许在图像上留下非常显眼的编码图案，提取器得以在短短几百步内迅速对齐特征并打通梯度信道。
2. **Stage 1**（泛化至全量随机图像，目标 $\operatorname{Acc}_{\mathrm{bit}} > 0.95$）：一旦验证集准确率突破 $0.90$，状态机自动关闭 `Fixed Batch`、切入全量随机训练图像（`Random Batch`），迫使编解码器摆脱对固定样本的死记硬背，学会在任意自然图像纹理上读写信号。
3. **Stage 2**（接入 3+2 可微噪声淬炼鲁棒性，目标 $\operatorname{Acc}_{\mathrm{bit}} > 0.98$）：当无噪准确率越过 $0.95$ 后，正式激活可微噪声层 $N$，同时仍保持 $\alpha=0.05$ 的低画质惩罚。此时网络专注于在几何裁切与光度滤波下保护信号存活。由于前三阶段任务目标纯粹，典型的 `TrustMark-B` 仅需约 800 个训练步就能走完 Stage 0 至 Stage 2。
4. **Stage 3**（开启对抗判别器并线性拉升画质权重）：当含噪准确率达到 $0.98$ 的高水位后，通信信道已足够强健。状态机切入最终阶段：激活 WGAN-GP 判别器，并在接下来的 10,000 步内将画质权重 $\alpha$ 从 $0.05$ 线性爬升至目标上限 $\alpha_{\max}$（`TrustMark-B` 取 $20$，`TrustMark-Q` 取 $27.5$）。在漫长的剩余训练中，强画质约束像一台液压机一样，在不破坏已建立的提取通道的前提下，将可见的水印残差一点点挤压进人眼不可察觉的高频纹理与频谱间隙中。

---

## 4. 任意分辨率残差缩放与可逆重水印机制

### 4.1 残差分辨率缩放：如何用 256 分辨率网络无损处理 4K 原图

既然编码器 $E$ 与解码器 $X$ 的内部张量尺寸被固定在 $256\times256$，推理时面对一张 $3840\times2160$ 的 4K 商业摄影原片该怎么办？最朴素的做法是把 4K 原图缩放到 $256^2$ 送入 $E(\bar{\mathbf x},\mathbf w)$，再把输出的带水印小图双线性放大回 4K——但这会造成灾难性的画质崩塌，因为原图中所有高于 $256^2$ 奈奎斯特采样率的发丝、织物纹理与锐利文字边缘都在下采样时永久丢失了。

![TrustMark 任意分辨率残差缩放算法](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-algorithm01-resolution-scaling.webp)

论文算法 1 提出的**残差分辨率缩放**（Residual Resolution Scaling）用极其优雅的代数拆解绕开了这一瓶颈。其核心洞察在于：**宿主图像本身富含不可压缩的高频细节，但水印嵌入器对图像所做的修改（即水印残差 $\mathbf r$）本质上是一组平滑的低幅空间调制信号**。因此，我们完全不需要重建高分辨率原图，只需在低分辨率下估算残差，再将残差插值放大叠加回无损原图即可：

1. **下采样与低分辨率残差估算**：给定任意尺寸 $H\times W$ 的原始图像 $\mathbf x\in[-1,1]^{H\times W\times 3}$，先将其下采样为固定网格输入 $\bar{\mathbf x}=\operatorname{Resize}(\mathbf x,256,256)$，运行嵌入器（或移除器 $R$）并作差得到 $256^2$ 分辨率下的残差张量：

   $$
   \mathbf r_{\mathrm{enc}} = E(\bar{\mathbf x},\mathbf w)-\bar{\mathbf x}, \qquad \mathbf r_{\mathrm{rm}} = R(\bar{\mathbf x})-\bar{\mathbf x}.
   $$

2. **通道零均值校正与插值叠加**：在源码 [`trustmark.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L465-L507) 的 `encode()` 实现中，为了彻底杜绝残差叠加后引发全图轻微偏色，代码在算法 1 之上额外执行了一步**按 RGB 通道减去残差空间均值**（zero-mean adjustment）的操作；随后仅将该低幅残差插值回原始画幅 $\mathbf r_{H,W}=\operatorname{Resize}(\mathbf r,H,W)$，乘上运行时强度旋钮 $\lambda\in[0.75,1.50]$（即 `WM_STRENGTH`）后加回未经任何降质的原始高分辨率图像 $\mathbf x$：

   $$
   \mathbf y = \operatorname{clamp}\left(\mathbf x + \lambda\mathbf r_{H,W},\ -1,\ 1\right).
   $$

这一设计带来了三重工程红利：第一，无论输入是 512 像素还是 8K 巨幅海报，神经网络前向推理的 FLOPs 与显存开销完全恒定；第二，原始高分辨率图像的高频细节 100% 原样保留，跨分辨率 PSNR 几乎恒定；第三，解码时只需将高分辨率带水印图下采样回 $256\times256$（或按变体要求缩放到 $245^2$/$224^2$）送入提取器 $X$，插值放大的残差恰好在逆向缩放中还原回训练期的空间频率分布。

当然，把 $256^2$ 方形网格的残差直接拉伸到极端长宽比画布（如 $4000\times 400$ 的网页长横幅）会导致水印波形在一个方向上被过度拉长。为此，开源库在工程上补充了长宽比保护逻辑：当长宽比超限时（或使用 `P` 变体时），自动截取中心方形/合规矩形区域嵌入水印，并在拼回原图时沿边界施加 1% 宽度（上限 50 像素）的线性阿尔法羽化（feather paste），消除接缝痕迹。

### 4.2 TrustMark-RM 与多轮可逆重水印工作流

在真实的创意供应链中，一件数字资产往往会依次流经摄影师、修图师、广告代理商与发行平台，每个环节都需要写入自己的溯源标识。如果直接在已经带有水印的图像上反复调用 $E$ 写入新载荷，历次残差就会像滚雪球一样层层累加，导致图像 PSNR 随重写轮数急剧滑坡。

为了支持可持续的多轮重水印（Re-watermarking），论文基于先进的图像恢复骨干网络 [KBNet](https://arxiv.org/abs/2303.02881) 训练了专用的水印移除器 **TrustMark-RM**（记为 $R$）。在训练 $R$ 时，预训练好的嵌入器 $E$ 与提取器 $X$ 全程冻结；每张宿主图像 $\mathbf x$ 会随机采样 $n=3$ 个截然不同的水印向量生成带水印样本 $\mathbf y_i=E(\mathbf x,\mathbf w_i)$，迫使 $R$ 捕捉该嵌入器族的共性残差统计，而非针对单一固定载荷过拟合。移除器的联合优化目标包含三项：

$$
\mathcal L_{\mathrm{MSE}}^R = \left\|R(\mathbf y)-\mathbf x\right\|_2^2, \qquad \mathcal L_{\mathrm{GAN+GP}}^R, \qquad \mathcal L_X^R = \left\|X(R(\mathbf y))-X(\mathbf x)\right\|_2^2.
$$

其中，像素重建项 $\mathcal L_{\mathrm{MSE}}^R$ 与对抗项 $\mathcal L_{\mathrm{GAN+GP}}^R$ 负责将画质还原至原图水平；而极具巧思的**提取器响应对齐项** $\mathcal L_X^R$ 则直接要求净化后图像在冻结提取器 $X$ 上的输出 logits 退化为与无水印原图 $X(\mathbf x)$ 一致的随机响应（而非强行优化交叉熵去翻转比特），从而在不引入额外反向伪影的前提下彻底抹除旧水印信号。借此，多轮资产流转得以按“先擦除旧水印、再编码新载荷”的迭代公式运行：

$$
\mathbf y^{(k+1)} = E\left(R(\mathbf y^{(k)}),\ \mathbf w_{k+1}\right).
$$

需要从数学上指出的是，由于嵌入过程包含截断与非线性激活，移除器 $R$ 学习的只是条件期望意义下的近似逆映射，每轮擦除仍会残留极其微小的恢复偏差（restoration bias）。因此，在能够访问无水印母版（master asset）的集中式媒资系统中，始终应优先从母版派生新水印副本；仅在下游合作方无法获取母版的分布式流转环节，才启用 `remove-then-encode` 链路。

---

## 5. 主实验、多维鲁棒性与重水印评估

论文使用 MIRFlickr-1M 中的 10 万张图像训练（1,000 张验证），在单张 RTX 3090 上训练 TrustMark 150 个 epoch 约耗时 48 小时（单图平均编码耗时 125 ms、解码仅需 25 ms），而在 A100 上训练基于 KBNet 的 TrustMark-RM 100 个 epoch 约需两周。评测横跨三大基准数据集：**CLIC**（专业摄影压缩基准）、**DIV2K**（800 张训练 + 100 张验证，2K 高分辨率且高频纹理最复杂）与 **MetFace**（艺术人像方形图）。

### 5.1 三大基准主结果与残差分布可视化

![TrustMark 与五个 baseline 的主结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table01-main-benchmarks.webp)

表 1 在原始高分辨率下系统对比了 `TrustMark-Q`、`TrustMark-B` 与五种代表性基线（为保证公平，所有固定分辨率深度学习基线均统一套用了本文的残差分辨率缩放策略）。仔细审视表 1 的数字，可以清晰看到此前方法难以兼顾画质与鲁棒性的根源：

- **高画质但脆弱的基线**：传统频域方法 `dwtDctSvd` 与自监督特征水印 `SSL Watermarking` 虽然能在 CLIC/DIV2K 上拿到 $38.02\sim 42.74\text{ dB}$ 的高 PSNR，但一旦遭遇表中的复合几何与光度扰动（Noised Acc.），`dwtDctSvd` 的比特准确率瞬间跌至 $0.51\sim 0.52$（完全等同于抛硬币瞎猜），`SSL` 也跌至 $0.57\sim 0.70$，无法支撑实际分发。
- **高鲁棒但画质受损的基线**：基于冻结自编码器潜空间的 `RoSteALS` 虽然在复合噪声下保住了 $0.93\sim 0.94$ 的比特准确率，但受制于自编码器本身的重建瓶颈，其 PSNR 暴跌至 $27.95\sim 33.77\text{ dB}$，肉眼可见明显失真；`StegaStamp`（$35.87\sim 39.35\text{ dB}$，含噪准确率 $0.70\sim 0.72$）与 `RivaGAN`（约 $41.0\text{ dB}$，含噪准确率 $0.78\sim 0.82$）则在两端均存在明显短板。
- **TrustMark 的帕累托推进**：在统一承载 100 bits 载荷的条件下，`TrustMark-B` 在维持高保真画质（CLIC **41.53 dB**、DIV2K **40.20 dB**、MetFace **43.87 dB**，SSIM 全线 $\ge 0.99$）的同时，将三大数据集上的含噪比特准确率推高至全表最高的 **0.97**；而偏重画质的 `TrustMark-Q` 则仅以让渡 $1\%\sim 2\%$ 含噪准确率（$0.95\sim 0.96$）的微小代价，将 PSNR 进一步推升至 **43.26 dB**（CLIC）、**42.39 dB**（DIV2K）与 **45.34 dB**（MetFace）。

![TrustMark 与 baseline 的定性水印和残差比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig02-qualitative-comparison.webp)

图 2 的 20 倍残差放大图直观揭示了数值领先背后的空域机理：`RoSteALS` 与 `StegaStamp` 在平滑背景与轮廓处留下了刺眼的低频色块和光晕，而得益于早期融合、$E_{\mathrm{post}}$ 逐像素重组头与 FFL 频域约束的协同，TrustMark 的残差精准贴合着前景物体的高频边缘纹理分布，在平坦区域几乎隐形。

![TrustMark 在 ImageNet-C 噪声配置下的对比](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table02-imagenet-c.webp)

读者很自然会追问：表 1 的评测噪声与 TrustMark 训练期的可微噪声同源，这是否会对基线方法不公平？为了消除这一疑虑，表 2 彻底切换到外部基线 `RoSteALS` 所采用的 **ImageNet-C** 独立噪声协议，并关闭残差缩放直接在 $256^2$ 分辨率下进行同基准对比（取 $\alpha_{\max}=15$）。结果证明，即便在别人定义的噪声主场下，TrustMark 依然交出了 **38.87 dB PSNR** 与 **0.95 含噪准确率**，大幅领先同协议下的 `RoSteALS`（32.68 dB / 0.94）与 `StegaStamp`（31.26 dB / 0.88）。

### 5.2 容量扩展、白盒梯度擦除成本与 15 类单项扰动剖析

![TrustMark payload 长度、噪声强度和对抗攻击三面板](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig04-payload-noise-adversarial.webp)

图 4 沿着三个极端压力轴探明了系统的容量与安全边界：

1. **载荷容量扩展**（左图）：当嵌入位数从 32 bits 扩展约 6 倍至 200 bits 时，受制于自然图像有限的感知掩蔽冗余，PSNR 从约 44.5 dB 平滑回落约 7.5 dB，含噪比特准确率下降约 11%。这印证了将默认载荷定在 100 bits（配合 BCH 纠错承载 61-bit 随机 ID）的工程合理性。
2. **训练噪声强度档位**（中图）：如果训练期完全关闭噪声模拟（None），模型能考出 53.2 dB 的惊人 PSNR，但面对真实分发扰动毫无招架之力；将训练噪声推至 High 档位后，模型主动牺牲约 13 dB 画质（稳定在 40.2 dB）以加固信道，使高强度扰动下的准确率仅衰减约 3%。
3. **I-FGSM 白盒梯度攻击成本**（右图）：当攻击者拥有完整提取器权重并发起步长受限（$\epsilon=8/255$）的 I-FGSM 白盒迭代擦除攻击时，未经噪声训练的模型在 50 步内全军覆没；而经历过 High 档可微噪声淬炼的 TrustMark，有 **32% 的样本需要超过 3,000 次梯度迭代**才能将准确率压低至失效阈值以下，显著抬高了批量白盒擦除的算力门槛。

![TrustMark 对十五类单独扰动的恢复率](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig05-individual-perturbations.webp)

图 5 进一步拆解了 15 类单项扰动下的恢复表现。除高斯噪声与方框模糊（Box Blur）两项略逊于牺牲了逾 12 dB 画质的 `RoSteALS` 外，`TrustMark-Q` 在其余所有几何与色彩变换上均稳居第一，彻底消除了像 `dwtDctSvd` 遇上 RGB 通道偏移即跌至 0.50 随机水平那样的单点雪崩。这也给工业界提了一个醒：验收水印系统时绝不能只看单一平均准确率，必须建立覆盖目标业务真实转码参数（如微信/抖音/小红书压缩链路）的分项回放矩阵。

### 5.3 水印移除与 20 轮连续重水印实验

![TrustMark-RM 水印移除结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table03-removal.webp)

表 3 评估了专用恢复网络 TrustMark-RM 与 I-FGSM 白盒攻击在 DIV2K（针对 `TrustMark-B` 编码图）上的擦除表现：I-FGSM 虽然能把解码准确率压低到 $0.629$，但代价是将图像破坏得面目全非（PSNR 跌至 23.48 dB，SSIM 仅 0.613）；相比之下，**TrustMark-RM 在将比特准确率压低至 0.553（逼近 0.50 的完全擦除极限）的同时，交出了 48.48 dB PSNR 与 0.997 SSIM 的超高清修复画质**。至于为什么准确率停留在 0.553 而非严格的 0.500，作者指出这是因为高分辨率残差插值在全图留下了极其微弱的低频统计痕迹。

![有无 TrustMark-RM 时多轮重水印曲线](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig03-rewatermarking-curves.webp)

![TrustMark 一次、五次和十次重水印示例](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig07-rewatermarking-example.webp)

图 3 与图 7 展示了连续 20 轮写入不同随机载荷的累积效应：无论中间是否穿插 TrustMark-RM，最新一轮写入的水印都能以接近 1.0 的准确率被提取器读出（新残差自然压倒旧残差）；但如果直接暴力覆盖写入（绿线），图像 PSNR 会在 20 轮内从 43 dB 暴跌至 32 dB 以下；而在每次重编码前先用 TrustMark-RM 擦除上一轮残差（红线），则能在 5 至 10 轮资产流转内将画质衰减斜率压平一半以上，使第 10 轮重水印图像在肉眼观察下依然纯净无瑕（图 7）。

---

## 6. 消融实验、频谱机理与补充材料深读

### 6.1 核心组件消融、质量-鲁棒性连续工作曲线与高频杂波局限

![TrustMark 架构与损失消融](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table04-ablation.webp)

表 4（在 DIV2K、$\alpha_{\max}=20$ 下评测）用详实的数据证明了三大质量组件与提取器骨干的互补性：

- **嵌入端组件协同**：当 WGAN-GP、FFL 与 $E_{\mathrm{post}}$ 全部关闭时，基线 PSNR 仅为 36.30 dB、含噪准确率为 0.955；单独启用 GAN、FFL、$E_{\mathrm{post}}$ 分别将 PSNR 提升至 36.99 dB（+0.69 dB）、38.16 dB（+1.86 dB）与 37.91 dB（+1.61 dB）；两两组合继续稳步攀升，而在**三者全部启用时，PSNR 跃升至 40.30 dB（净增 4.0 dB），同时含噪比特准确率不仅没有因画质变严而下降，反而同步升至 0.977**。这说明抑制空域模糊与频谱伪影同样有助于提取器更干净地分离出有效编码特征。
- **提取端骨干选型**：无残差连接的普通 CNN 在四阶段课程中训练发散（Failed）；VGG16（35.69 dB / 0.934）与轻量化的 ResNet18（38.58 dB / 0.956）表现次之；ResNet50（40.30 dB / 0.977）与更重的 ResNet101（40.35 dB / 0.976）几乎打平，确立了 ResNet50 作为主力骨干（以及 ResNet18 作为后续开源库 `C` 紧凑变体）的依据。

![TrustMark 训练损失权重对 PSNR 与恢复率的影响](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig06-quality-robustness-tradeoff.webp)

![TrustMark 在密集纹理和杂乱图像上的局限](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig08-limitations-clutter.webp)

图 6 扫描了训练期权重上限 $\alpha_{\max}\in(0,30)$ 的连续帕累托曲线：随着 $\alpha_{\max}$ 从接近 0 增大至 30，图像 PSNR 大幅提升超过 8 dB，而含噪比特准确率仅缓慢回落不到 5%，`TrustMark-B`（$\alpha_{\max}=20$）与 `TrustMark-Q`（$\alpha_{\max}=27.5$）正是这条平滑曲线上选取的两个代表性锚点。

图 8 则坦诚揭示了方法在**高频杂波图像**（cluttered images，如浓密动物毛发、繁茂植被与密集细线）上的退化案例。直觉上，密集纹理提供了极佳的人眼视觉掩蔽，但从通信信道视角看，宿主图像自身剧烈起伏的高频能量也构成了强烈的背景干扰（与水印信号争夺高频频段），迫使嵌入器注入更大幅值的残差，并导致局部提取准确率下滑。

### 6.2 频谱伪影消除与跨分辨率缩放实证

![TrustMark 加入 FFL 前后的平均频谱残差](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig09-frequency-ffl.webp)

为什么焦点频率损失（FFL）能单独带来 1.86 dB 的 PSNR 增益？补充材料图 9 展示了 MetFace 数据集上的平均二维傅里叶频谱残差对比：在未加 FFL 的左图中，频谱残差在水平与垂直十字轴的四个方向频带上呈现出极其刺眼的周期性亮斑——这正是卷积算子在固定步长网格上激发出方向性谐波的直接证据；而引入 FFL 动态频域加权后（右图），十字轴上的能量尖峰被彻底抹平，残差均匀弥散到全频段背景中。

![TrustMark 水印在不同目标分辨率下的插值比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig10-resolution-scaling-watermarking.webp)

![TrustMark-RM 在不同目标分辨率下的插值比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig11-resolution-scaling-removal.webp)

![水印图像 PSNR 随评测分辨率变化](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig16-psnr-resolution-dependence.webp)

补充材料图 10 与图 11 给出了支撑“残差分辨率缩放”最有力的定量曲线。随着目标图像分辨率从 $256^2$ 一路放大至 DIV2K 原尺寸（2K）：

- 如果直接将网络输出的 $256^2$ 整图做双线性（Bilinear）或双三次（Bicubic）插值放大，图像 PSNR 会因为高频细节丢失而从 40 dB 大幅滑落至 26 dB 左右；
- 反观仅插值放大残差的 **Resolution Scaling**（蓝线），在整个分辨率跨度上 **PSNR 波动不足 0.1 dB**，而比特准确率仅比双三次整图插值低约 1 个百分点；且这一跨分辨率保真特性在水印移除器 TrustMark-RM（图 11）上同样完美成立。

补充材料图 16 还顺带揭穿了一个在以往论文中屡见不鲜的“评测水分”：对高分辨率带水印图像做下采样会充当低通滤波器平滑掉高频水印残差，导致同一张 `dwtDctSvd` 水印图在 $256^2$ 下测出的 PSNR 比在原尺寸下测出的虚高了整整 **4.7 dB**。因此，任何面向生产的水印基准都必须在原始输出分辨率下直接计算画质指标。

### 6.3 内部激活图、极简控制图与推理期强度旋钮

![TrustMark 嵌入器预处理与后处理激活图](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig12-activation-maps.webp)

![TrustMark 在纯色、几何纹理和条纹控制图上的结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig13-control-images.webp)

打开网络黑盒，补充材料图 12 可视化了预处理模块 $E_{\mathrm{pre}}$ 与后处理头 $E_{\mathrm{post}}$ 的通道激活图：网络在早期融合阶段同时提取了宿主图的显著物体边缘轮廓与全局低幅载波模式，最终在 $E_{\mathrm{post}}$ 中将二者相乘调制为自适应残差。补充材料图 13 则在纯色块、棋盘格与规则条纹等毫无自然语义的**极简合成控制图**（control images）上进行了测试——TrustMark 依然实现了 100% 的无噪比特恢复，证明提取器学到的是真正的底层频空域调制规律，绝非依赖自然图像分类特征的投机取巧。

![TrustMark 低中高三档噪声模拟参数](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-table06-noise-severity.webp)

![TrustMark 训练时不同噪声来源的视觉示例](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig14-noise-examples.webp)

![TrustMark 运行时残差系数对 PSNR 与 bit accuracy 的影响](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig15-inference-strength.webp)

补充材料表 6 与图 14 完整公开了 Low / Medium / High 三档可微噪声的参数采样区间与视觉退化效果。补充材料图 15 则验证了推理期残差缩放系数 $\lambda$ 的单图调节能力：在不重新训练模型的前提下，只需在 $\lambda\in[0.75,1.50]$ 区间内线性缩放残差幅值，就能在单张图片上平滑交换 PSNR 与解码准确率——这直接演化为了开源代码中面向用户的 `WM_STRENGTH` 参数。

---

## 7. 官方源码架构与工程演进走读

对比 2023 年底发布的 arXiv v1 论文，官方开源仓库 [`adobe/trustmark`](https://github.com/adobe/trustmark/tree/0ed40cbe8188f664fd9cbbeacd969807de27440a)（固定 commit `0ed40cbe`）在迈向工业级 SDK 的过程中，补齐了多变体模型族、BCH 纠错打包、长宽比保护与跨语言运行时。其核心推理调用链可概括为：

```text
TrustMark(...)
  -> DataLayer(secret_len=100, encoding_type=...)
  -> check_and_download(): 按需拉取 YAML/ckpt 并校验 MD5 摘要
  -> encode(): 业务载荷 -> 100-bit BCH 数据包 -> U-Net 编码 -> 通道零均值校正与残差缩放 -> 边缘羽化贴回 PIL
  -> decode(): 区域裁剪/可选旋转或 bbox 定位 -> ResNet 解码 logits -> BCH 纠错 -> 还原业务载荷
  -> remove_watermark(): KBNet 预测反向残差 -> 残差缩放 -> 贴回 PIL
```

### 7.1 推理主链与按需子图剥离

1. **端到端 API 与零均值羽化融合（[`trustmark.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L73-L527)）**：构造器负责初始化 `DataLayer` 并通过 [`check_and_download()`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L28-L67) 拉取且校验远端权重的 MD5。在 [`encode()`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L465-L507) 中，代码计算出 $256^2$ 残差后，先沿空间维度减去每个 RGB 通道的均值以消除整图色偏，再按 `WM_MERGE` 插值放大、乘上 `WM_STRENGTH`，并在开启区域裁剪时配合 1% 带宽（上限 50 像素）的边缘羽化贴回中心区域。
2. **Lightning 容器与轻量推理子图剥离（[`model.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/model.py#L31-L133)）**：`TrustMark_Arch` 继承自 PyTorch Lightning，内部保留了 `fixed_input`、`bit_acc_thresholds=[0.9, 0.95, 0.98]` 与双优化器交替更新逻辑。在生产推理加载时，`load_model(..., part='encoder')` 会将解码器、判别器、损失函数与噪声层全部替换为零开销的 `Identity` 占位符（仅加载解码器时同理），从而直接从联合训练的 `.ckpt` 中剥离出极简推理子图。

### 7.2 四种模型变体与 BCH 纠错数据层契约

开源仓库在论文 `B`、`Q` 两款模型之上，新增了面向端侧轻量部署的 `C` 与追求极致视觉质量的 `P`（注意：四个变体的编解码权重互不兼容，提取时必须指定对应模型族）：

| 模型变体 | 论文 v1 覆盖 | 典型 PSNR（源码注释口径） | 架构与预处理特性 |
| --- | :---: | :---: | --- |
| **B** (Balanced) | 是 | ~41 dB | ResNet50 解码器（输入 $245^2$），$\alpha_{\max}=20$，偏重抗噪鲁棒性 |
| **Q** (Quality) | 是 | ~43 dB | ResNet50 解码器（输入 $245^2$），$\alpha_{\max}=27.5$，开源仓库默认变体 |
| **C** (Compact) | 否（后续新增） | ~39 dB | ResNet18 轻量解码器，面向低算力与边缘端快速解码 |
| **P** (Perceptual) | 否（后续新增） | ~48 dB | 解码器输入 $224^2$，强化感知损失，强制中心方形区域裁剪且默认强度乘 $1.25$ |

在数据打包层（[`datalayer.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/datalayer.py#L14-L100)），模型底层固定的 100 bits 物理信道被严格划分为 **4 bits 版本与模式头**（2 位保留位 + 2 位 Schema 标识位）和 **96 bits 数据+BCH 校验区**：

| BCH Schema | 有效业务比特数 | 7-bit ASCII 字符数 | BCH 校验比特数 | 最大可纠正翻转位数 ($t$) | 版本头取值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `BCH_SUPER` | 40 bits | 5 字符 | 56 bits | 8 bits | `0` |
| `BCH_5`（默认） | 61 bits | 8 字符 | 35 bits | 5 bits | `1` |
| `BCH_4` | 68 bits | 9 字符 | 28 bits | 4 bits | `2` |
| `BCH_3` | 75 bits | 10 字符 | 21 bits | 3 bits | `3` |

由于头部自带了 2 bits 的 Schema 标识，解码端无需事先知晓编码方选用了哪一档纠错强度，即可根据读出的头部自动初始化对应的 BCH 解码器。同时需要注意，源码中的文本模式通过 `ord(t) & 127` 强制按 7-bit ASCII 截断打包，因此在生产系统中应统一使用二进制模式写入高熵随机生成的非透明标识符（opaque ID），切勿直接塞入 UTF-8 中文字符或明文用户隐私（PII）。

### 7.3 跨语言运行时与 C2PA 软绑定集成

官方仓库提供了 Python、JavaScript 与 Rust 三套语言实现，并通过 [`c2pa_watermark_example.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/c2pa/c2pa_watermark_example.py) 演示了与 C2PA 内容凭证的完整衔接：

| 能力维度 | Python 参考实现 | JavaScript (`onnxruntime-web`) | Rust Crate (`ort`) |
| --- | --- | --- | --- |
| 水印写入（Encode） | 完整支持（含文本/二进制与区域羽化） | 不支持 | 支持（仅二进制子集） |
| 水印提取（Decode） | 完整支持（含旋转重试与 bbox 定位） | 支持（WebGPU / WASM） | 支持 |
| 水印移除（Remove） | 完整支持（KBNet） | 不支持 | 不支持 |
| 模型格式与依赖 | PyTorch `.ckpt`（运行时按需下载） | ONNX | ONNX |

在构建“服务端 Rust/Python 写入、浏览器端 JS 提取”的跨语言链路时，工程团队必须使用标准黄金测试图（Golden Images）严格对齐三端预处理的 Resize 插值核、RGB 通道顺序、`P` 变体的 $224^2$ 中心方形裁剪规则以及 BCH 比特序。而在接入 C2PA 体系时，水印的信任传递链条为：

```text
图像像素 -> TrustMark 解码器 -> 61-bit 随机标识符 -> 检索可信 Manifest 仓库 -> 验证密码学签名与证书链 -> 确认溯源声明
```

像素中的不可见水印仅承担前两步的“**抗剥离索引**”职责；一旦脱离外部密码学签名验证与图像内容感知哈希比对，单纯从像素中解出一段合法的 BCH 比特流绝不构成来源真实性证明。

---

## 8. 生产落地指南、安全防御与方法局限

将 TrustMark 纳入生产级内容溯源管线时，需要在工程、安全与方法边界上落实以下准则：

1. **模型供应链固化与色彩空间契约**：在构建容器镜像阶段预先拉取模型配置与权重并锁定 SHA-256 摘要，严禁线上高并发请求触发首次 `urlretrieve` 远端下载阻塞；在调用 `encode()` 前统一处理 EXIF 方向旋转、CMYK/16-bit/RGBA 到 8-bit sRGB 的色彩转换，并在最终落盘的 JPEG/WebP 字节流上执行一次回读解码校验，确保有损压缩未击穿当前图片的 BCH 纠错余量。
2. **防黑盒 Oracle 擦除与多解码器假阳性控制**：公开解码器意味着绝不能将算法视为保密信道。对外开放的溯源查询接口应仅返回最终查验状态与绑定的 Manifest 摘要，严禁向未授权调用方返回逐比特原始 Sigmoid 置信度（防止攻击者利用置信度变化构造黑盒梯度擦除器）。此外，当检测端串联尝试多个模型变体（B/Q/C/P）、多个旋转角度与多个局部检测框时，多重假设检验会显著推高无水印自然图像偶然撞上合法 BCH 校验码的**假阳性率**（False Positive Rate），必须结合预测 logits 置信度阈值与内容哈希二次比对加以过滤。
3. **隐私保护与可撤销生命周期治理**：切勿将用户 ID、邮箱或设备序列号直接编码进不可见水印。正确做法是在像素中仅写入无意义的随机 UUID/雪花 ID，在服务端映射表中实施多租户隔离与过期撤销（Revocation）——当用户发起隐私合规删除或撤销授权时，直接注销服务端映射记录即可使已分发图片中的水印索引失效，无需依赖有损的 `TrustMark-RM` 去召回全网历史图片。
4. **方法局限性审视**：
   - **空间频率受限于 $256^2$ 网格**：内部编码网格固定在 $256^2$，面对极端细长比画布、超小局部贴图或多图层拼贴合成场景，仍需依赖外围裁剪与边界框检测补丁；
   - **极端高频杂波与非自然图像退化**：训练数据集中于 MIR-Flickr 自然照片，面对高频杂波图像（图 8）已有明显退化，迁移至医学影像、线稿动漫、UI 截图或广色域 HDR 资产前必须独立标定；
   - **难抗扩散模型重绘再生攻击**：作为像素级后处理水印，TrustMark 擅长抵御传统信号处理退化，但若图像经过扩散模型图生图（Img2Img）重绘或 VAE 重编码，像素级残差仍会被大幅洗刷。

回望全文，TrustMark 通过“去归一化 MUNIT 编码 + 频谱/感知/对抗四重约束 + 四阶段课程学习 + 任意分辨率残差缩放”，漂亮地解开了固定算力深度学习网络服务任意高分辨率创意资产的工程死结；而将其准确地安放在 C2PA 密码学凭证体系的**耐久软绑定检索层**上，则是其走向大规模工业部署的关键所在。

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

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

生成式图像进入日常创作之后，“这张图从哪里来”逐渐变成一个系统问题。元数据可以记录作者、编辑历史和签名，却可能在截图、社交平台转码或普通导出时丢失；图像分类器可以猜测内容是否由某类模型生成，却很难把判断绑定到一个可审计的资产标识；传统水印可以在频域或像素低位写入数据，又常在大幅裁切、压缩、缩放和多轮编辑后失效。更棘手的是，真实工作流里的图像并不统一停留在 $256\times256$：相机照片、海报、数字绘画和生成式内容具有不同分辨率、长宽比与纹理统计。

**TrustMark: Universal Watermarking for Arbitrary Resolution Images** 尝试把这些约束放入同一个可训练系统。它用 MUNIT 风格的嵌入器把比特写进图像，用 ResNet50 提取器恢复比特，在两者之间插入可微噪声模拟；又同时在 YUV 像素、LPIPS 感知特征、频谱和对抗分布上约束图像质量。为了绕过固定输入尺寸，它不直接在高分辨率图像上运行大网络，而是在 $256^2$ 图像上预测残差，再把残差插值回原分辨率。论文还训练了一个独立的 TrustMark-RM，把水印视为可学习去除的弱噪声，以支持“先移除旧标识、再写入新标识”的重水印流程。

这条路线很实用，也容易被过度解读。TrustMark 能恢复一段隐藏比特，不等于它证明图像真实；TrustMark-RM 能让原提取器接近随机响应，不等于它提供对抗安全；“任意分辨率”是残差缩放策略，不是一个原生理解任意尺寸的网络；论文中的 bit accuracy 是原始比特平均正确率，不等于加入纠错后整个业务标识一定成功恢复。本文会持续区分这些概念。

本文严格以 arXiv [`2311.18297v1`](https://arxiv.org/abs/2311.18297v1) 为论文基线。后续 ICCV 2025 版本把题名扩展为 **TrustMark: Robust Watermarking and Watermark Removal for Arbitrary Resolution Images**，这里只在版本状态中说明，不混入其后续实验。源码静态阅读固定到官方仓库 commit [`0ed40cbe8188f664fd9cbbeacd969807de27440a`](https://github.com/adobe/trustmark/tree/0ed40cbe8188f664fd9cbbeacd969807de27440a)。该仓库没有 tag 或 GitHub Release；固定提交中的 `pyproject.toml` 仍为 `0.9.0`，而 PyPI 已发布 `0.9.1`。论文、Git commit 和 PyPI 包是三条不同版本轴。

先给出全文判断：

> TrustMark 的长期价值，是把不可见水印从“固定小图上的编码器实验”推进到一个考虑扰动、分辨率、移除、纠错和内容凭证连接方式的工程接口；它的核心风险，则是鲁棒性仍取决于有限扰动分布，公开解码器与移除器扩大了攻防面，而水印标识本身从来不是内容真实性的密码学证明。

## 1. 论文、作者、版本与开放状态

| 项目 | 内容 |
| --- | --- |
| 题名 | TrustMark: Universal Watermarking for Arbitrary Resolution Images |
| 作者 | Tu Bui, Shruti Agarwal, John Collomosse |
| 机构 | Adobe Research；University of Surrey |
| arXiv | `2311.18297v1` |
| 提交日期 | 2023-11-30 |
| PDF | 14 pages，主文与补充材料合并 |
| 论文许可 | arXiv non-exclusive distribution license |
| 后续状态 | ICCV 2025 版本改题为 Robust Watermarking and Watermark Removal for Arbitrary Resolution Images |
| 官方代码 | [adobe/trustmark](https://github.com/adobe/trustmark) |
| 源码基线 | commit `0ed40cbe8188f664fd9cbbeacd969807de27440a` |
| 代码许可 | MIT |
| Python 包 | 固定源码写 `0.9.0`；PyPI 当前发布线已到 `0.9.1` |

论文 v1 的研究对象只有 TrustMark-B 与 TrustMark-Q。当前仓库还提供 C、P 两个变体、BCH 纠错 schema、区域定位、中心区域嵌入、feathering、JavaScript 解码、Rust 子集和 C2PA soft-binding 示例。这些能力能说明项目在论文后继续工程化，却不能被倒写成 2023 年论文的实验条件。反过来，仓库公开了推理接口和模型加载逻辑，并没有提供 MIR-Flickr 训练数据、完整训练配置和一键重现实验的流水线；“源码公开”不等于“论文训练可完全复现”。

许可也需要拆开。论文图表随 arXiv non-exclusive distribution license 发布，本文只在学术评论中等比例引用并保留作者归属，不把它标成 CC。官方代码仓库是 MIT，模型文件由首次运行时从 Adobe 端点下载，实际部署还需核对模型文件、软件包与业务数据各自的许可。后续 ICCV 页面、PyPI 页面和仓库 README 是版本状态证据，不改变 v1 图表的来源口径。

## 2. 任务与术语：水印不是一个单一问题

**不可见数字水印**把短消息嵌入图像内容，希望人眼难以察觉，但机器可以恢复。它通常同时优化视觉质量、容量和经过分发变换后的恢复率。

**可见水印**把文字或 Logo 直接叠加在画面上。它的目标包括威慑和显式归属，技术问题与 TrustMark 不同。

**Steganography** 强调通信的存在本身不易被发现，通常关心载荷和隐蔽性；水印更常强调内容归属、鲁棒识别或完整性信号。两者模型和指标有重叠，但安全目标并不相同。

**Fingerprinting** 往往为不同接收者嵌入不同标识，用于泄漏追踪。它还需要碰撞抵抗、用户映射和合谋攻击分析，不能只看平均 bit accuracy。

**C2PA metadata** 是签名 manifest，记录内容与编辑声明。它依赖密码学签名、证书与可信声明链；水印可以携带一个 soft-binding identifier，在元数据丢失后帮助查回 manifest，但不能代替签名验证。

**AI image detector** 根据像素统计判断“像不像某类生成器输出”，通常是概率分类。TrustMark 解码的是主动写入的 payload；没有水印的真实图、没有采用 TrustMark 的 AI 图，以及被成功攻击的图都不应由它推断来源。

因此，TrustMark 的最准确产品角色是：**一个主动嵌入、可机器检索的耐久标识通道**。通道中可以放索引、版本或策略代码，语义和信任则由通道之外的签名、数据库、访问控制和审计系统承担。

## 3. 威胁模型：鲁棒、移除与伪造要分开

论文主要研究普通分发扰动：缩放、裁切、JPEG、颜色与亮度变化、模糊、噪声、posterize 等。训练时把这些操作做成可微模块，让嵌入器学会把信号放在更能穿过扰动的位置。这是一种**经验鲁棒性**：只对训练和评测覆盖的变换族有证据。

恶意攻击者的目标至少有四种。其一是 **removal**，让解码器再也识别不出 payload；其二是 **spoofing**，让没有授权的图像被识别为某个标识；其三是 **watermark transfer**，把一张图的信号搬到另一张图；其四是 **payload substitution**，把旧标识替换成新标识。论文用 I-FGSM 探索了白盒移除，也训练了 TrustMark-RM，却没有系统覆盖伪造、转移、密钥泄露和多解码器攻击。

还有一个常被忽视的层次：**来源声明本身是否可信**。即使 100 个原始比特全部恢复，也只证明“当前像素与某个解码器约定相匹配”。如果 payload 是公开、可预测或可复制的，攻击者可能把同一标识写入别的内容。只有把 payload 设计成随机、不可直接解释的索引，并在服务端绑定签名 manifest、资产哈希、发行主体和撤销状态，才能把检测结果纳入更强的来源判断。

TrustMark-RM 更不是安全防护。它是论文主动提供的恢复与重水印工具，同时也证明模型水印可以被专门网络削弱。开放实现有利于互操作、研究和审计，也意味着安全不能依赖算法保密。合理的威胁模型必须假设攻击者知道架构、拥有解码器，甚至可以收集水印前后样本训练代理移除器。

## 4. 四元权衡：容量、不可见性、鲁棒性与部署

水印系统可以看成四个互相牵制的目标。

**Payload capacity** 是可传输的原始或受保护比特数。载荷越长，每个比特能分配到的图像冗余越少，恢复更难；若使用纠错码，有效业务容量还会进一步缩小。

**Imperceptibility** 要求水印图 $\mathbf y$ 与 cover $\mathbf x$ 接近。PSNR 衡量像素误差，SSIM 衡量局部结构，LPIPS 近似感知差异；这些指标仍不能替代大规模主观研究。

**Robustness** 要求解码在 $\tilde{\mathbf y}=N(\mathbf y)$ 上仍正确。它受扰动组合、强度、顺序和图像内容影响，单一平均值会掩盖某些变换几乎失败的情况。

**Compute and deployability** 包括内部分辨率、模型尺寸、CPU/GPU 延迟、模型下载、浏览器兼容、色彩与格式处理，以及图像是否要被重新缩放。TrustMark 用固定 $256^2$ 网络和 residual scaling 控制成本，但这也形成对细小局部区域、极端长宽比和插值过程的结构性约束。

论文里的 $\alpha_{\max}$ 主要在不可见性与恢复率之间移动工作点：增大图像质量损失权重，PSNR 上升但 bit accuracy 下降。当前代码又提供运行时 `WM_STRENGTH`，直接缩放残差。二者都能改变可见性与鲁棒性，却发生在不同阶段，不能把运行时 strength 当作复现论文 $\alpha_{\max}$ 的等价参数。

## 5. Fig. 1：TrustMark 与 TrustMark-RM 的整体架构

![TrustMark 嵌入、扰动、提取与移除架构](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig01-architecture.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 1, arXiv non-exclusive distribution license.*

**图解决的问题。** Fig. 1 把“写入水印”和“移除水印”分成两个网络，却让它们共享固定的 embedder $E$ 与 extractor $X$。左半部分回答鲁棒水印怎样端到端训练；右半部分回答怎样合成 removal training pairs。

**图中结构。** cover image 与 watermark 进入 $E$ 得到 encoded image；训练时 encoded image 通过 differentiable noise $N$，再由 $X$ 恢复 watermark。判别器与 YUV、LPIPS、FFL 等损失约束输出质量。TrustMark-RM 则随机生成多个 payload，经冻结的 $E$ 产生带水印样本，由 restoration network $R$ 恢复 cover，同时用冻结 $X$ 检查输出不应继续稳定响应原水印。

**支持的结论。** 论文没有试图用同一网络既添加又移除水印，而是承认两者目标不同：嵌入是在固定图像容量中加入信息，移除更接近去噪。这种职责分离让 remover 可以选择 KBNet 一类图像恢复骨干，而不扭曲 embedder 的训练目标。

**不能证明的内容。** 架构图不能证明水印不可伪造，也不能说明任意编辑都能通过。右侧 remover 的存在反而提示：恢复率是模型与攻击之间的经验结果，不是密码学不可删除性。图中也没有展示 ECC、C2PA、区域定位或跨语言运行时，那些属于后续代码层。

## 6. 输入输出形式化：内部始终是 256 方图

设 cover image 为：

$$
\mathbf x\in\mathbb R^{256\times256\times3},
\qquad
\mathbf w\in\{0,1\}^{l}.
$$

嵌入器产生：

$$
\mathbf y=E(\mathbf x,\mathbf w),
$$

训练时施加可微扰动：

$$
\tilde{\mathbf y}=N(\mathbf y),
$$

提取器预测每一位为 1 的概率：

$$
\bar{\mathbf w}=X(\tilde{\mathbf y})\in[0,1]^l.
$$

论文文字有时把 $\bar{\mathbf w}$ 写成二值向量，但网络最后是 sigmoid，训练和阈值前实际输出是概率。bit accuracy 通常在阈值 0.5 后比较每一位。若各位独立且准确率为 $p$，整个 $l$ 位消息无错的概率并不是 $p$，而是近似 $p^l$；例如 $p=0.97,l=100$ 时只有约 $4.8\%$。实际错误存在相关性，纠错码也会改变结果，但这个估算足以说明：平均 bit accuracy 不能替代完整 payload 成功率。

高分辨率输入 $\mathbf x_{H,W}$ 不直接进入网络。Algorithm 1 先把它缩到 $256^2$，让网络产生输出，再取输出与缩小图的残差，把残差插值到 $(H,W)$ 后叠加在原图。因而“任意分辨率”的可计算接口成立，但隐藏信号的空间频率仍由 $256^2$ 预测和插值决定。

## 7. Early Image-Watermark Fusion

水印向量只有 $l$ 个离散比特，图像却是二维连续结构。TrustMark 的预处理模块 $E_{\mathrm{pre}}$ 先把 watermark 扩展到与图像相同的空间分辨率，再与 RGB 通道拼接，随后用 $d$ 个 $3\times3$ 卷积产生 $d$ 通道特征：

$$
E_{\mathrm{pre}}(\mathbf x,\mathbf w)
\in\mathbb R^{256\times256\times d}.
$$

这种 early fusion 让每个空间位置从网络第一层就能访问完整 payload，而不是等到瓶颈层才注入一段全局向量。理论上，嵌入器可以根据局部纹理决定某些 bit 的残差强度，又能在全图复制冗余信息以抵抗裁切。它不是简单把每个比特固定映射到某个像素，因为后续 MUNIT 风格网络会把图像内容与 payload 联合变换。

early fusion 也带来两个工程问题。第一，水印 feature map 的生成规则必须在训练和推理完全一致；当前包把业务字符串先编码成 100-bit 数据层，再送入模型。第二，模型变体之间即使 raw payload 同为 100 bits，编码器和解码器权重也不同，不能假设 Q 写入的信号由 P 解码。数据 schema 只解决“比特如何解释”，不解决“像素信号由哪个模型族生成”。

## 8. MUNIT 风格 Embedder

论文称嵌入器主体基于 MUNIT，但做了针对水印的修改：移除 channel normalization、加深网络、减小内部通道，从而把模型规模约减半。下采样路径聚合更大范围的图像上下文，上采样路径恢复空间细节，skip connection 保留局部纹理。非线性使用 SiLU。

移除 normalization 有合理动机。Instance/channel normalization 会重写通道统计，而不可见水印依赖微小的颜色、亮度和频谱差异；过强归一化可能抹掉正需传递的低幅信号。另一方面，没有归一化也使训练对初始化、学习率和阶段调度更敏感，这与论文专门设计四阶段训练相呼应。

论文强调 $E$ 直接输出 encoded image，而不是像 StegaStamp、RivaGAN 那样显式预测要相加的残差：

$$
\mathbf y=E(\mathbf x,\mathbf w).
$$

但在高分辨率部署时，Algorithm 1 又从 $E(\bar{\mathbf x},\mathbf w)-\bar{\mathbf x}$ 计算残差。也就是说，“训练网络直接输出图像”和“推理时把差值当残差搬运”并不矛盾；后者是包装 fixed-resolution model 的黑盒策略。

## 9. Post-process Network：为什么不是一层 RGB 投影

MUNIT 主体输出 $n$ 通道 feature，最后要变回 RGB。最简单的做法是一层卷积，但论文发现这会限制高频细节保真。TrustMark 的 $E_{\mathrm{post}}$ 使用多层 $1\times1$ convolution，中间插入 SiLU，末端使用 `tanh` 把值约束在 $[-1,1]$。

$1\times1$ convolution 不混合相邻像素，只在同一空间位置重组通道。它适合把不同尺度路径累积的局部、全局、payload feature 重新投影成 RGB，同时避免末端再引入较大空间核造成纹理平滑。多层非线性使映射能力高于单一线性投影。

Table 4 的消融显示，单独加入 $E_{\mathrm{post}}$ 可把 PSNR 从约 36.30 dB 提高到 37.91 dB；与 GAN 或 FFL 组合时提升更大。这是“post-process 不只是实现细节”的实验证据。不过，PSNR 增益不能直接证明输出在所有显示设备与色彩管理链路中不可见。论文在标准 RGB/YUV 张量上训练，没有覆盖 HDR、广色域、调色板 PNG 或复杂印刷链路。

## 10. ResNet50 Extractor 与检测逻辑

提取器 $X$ 使用标准 ResNet50，把最后分类层替换为 $l$ 维 sigmoid head。每一维对应一个 bit probability。论文报告，只有 ResNet 风格骨干配合特定训练阶段能稳定拟合；Table 4 也比较了不带残差连接的普通卷积堆叠和多个 ResNet 深度。

残差网络适合这个任务，不代表它“看见”语义水印。水印信号幅度很小，网络需要在多层下采样中保留细微统计；residual connection 缓解深层优化和微弱梯度传播。最终全局表示还要在图像不同位置聚合重复编码，才能抵抗局部 crop。

当前代码把模型 logits 与零比较，而不是对 sigmoid 后概率与 0.5 比较，两者在数学上等价。随后 `DataLayer.decode_bitstream()` 解析版本位、选择 BCH decoder、尝试纠错并返回 `(payload, detected, version)`。这里的 `detected` 已经是数据层判断，不是论文所报告的简单原始 bit accuracy。论文实验明确没有施加 ECC，因此当前 API 的“检测成功”不能直接与论文表格一列对齐。

还应区分 **decoder confidence** 与 **payload validity**。一个模型可以给每位很高置信度却形成无效 BCH code，也可以有少量低置信位但被纠错成功。生产监控应同时保留 raw logits/bit confidence、纠错翻转数、schema 版本和最终 payload 成功率，而不能只统计一个布尔值。

## 11. Differentiable Noise Model

TrustMark 在 $E$ 和 $X$ 之间插入 $N$，让训练图像经历三类基础几何变换和十五类可选扰动。基础变换是 random flip、crop、resize；可选扰动包括 JPEG、brightness、hue、contrast、sharpness、color jitter、RGB shift、saturation、grayscale、Gaussian blur、median blur、box blur、motion blur、Gaussian noise 和 posterize。

每个 encoded image 都经过 3 个 base transforms，再随机选择 2 个 optional transforms。所有变换都设计为可微，恢复损失能穿过 $X$ 与 $N$ 回传到 $E$。这使嵌入器不只是学会在 clean image 上骗过 decoder，而是主动寻找经过常见分发链仍能保留的信号。

可微近似与真实系统仍有差异。JPEG 的 differentiable approximation 不一定复刻浏览器、手机相册、社交平台和 GPU codec 的 rounding、chroma subsampling 与 metadata 行为；多个变换的顺序也会改变结果。训练时“3+2”是一种采样分布，而真实用户可能先裁切、锐化、截图、再压缩多次。论文的鲁棒性主张只能覆盖实验分布附近，不能概括所有图像处理链。

一个实用的部署方法，是把线上真实变换日志转成定期回放集：按平台、导出格式、质量参数和编辑链统计 decode rate，再决定是否扩充训练噪声或选择更鲁棒模型。仅在 ImageNet-C 或固定扰动表上保持高平均准确率，无法替代这种分布监控。

## 12. Total Objective：alpha 控制什么

TrustMark 的总目标是：

$$
\mathcal L_{\mathrm{total}}
=\alpha\mathcal L_{\mathrm{quality}}(\mathbf x,\mathbf y)
+\mathcal L_{\mathrm{recovery}}(\mathbf w,\bar{\mathbf w}).
$$

$\mathcal L_{\mathrm{recovery}}$ 是 bit-wise binary cross-entropy：

$$
\mathcal L_{\mathrm{recovery}}
=-\frac1l\sum_{j=1}^{l}
\left[w_j\log \bar w_j+(1-w_j)\log(1-\bar w_j)\right].
$$

质量损失由四部分构成：

$$
\begin{aligned}
\mathcal L_{\mathrm{quality}}
=&\ \beta_{\mathrm{YUV}}\mathcal L_{\mathrm{YUV}}
+\beta_{\mathrm{LPIPS}}\mathcal L_{\mathrm{LPIPS}}\\
&+\beta_{\mathrm{FFL}}\mathcal L_{\mathrm{FFL}}
+\beta_{\mathrm{GAN}}\mathcal L_{\mathrm{GAN+GP}}.
\end{aligned}
$$

实验设置为 $\beta_{\mathrm{LPIPS}}=1$、$\beta_{\mathrm{YUV}}=1.5$、$\beta_{\mathrm{FFL}}=1.5$、$\beta_{\mathrm{GAN}}=1$。$\alpha$ 不是直接乘水印残差的“强度”，而是训练中图像质量相对于恢复损失的权重。训练初期设 $\alpha=0.05$，先让 extractor 学会读出；后期才线性增加到 $\alpha_{\max}$。TrustMark-B 使用 20，TrustMark-Q 使用 27.5，Q 因更重视图像质量而获得更高 PSNR、略低 noisy accuracy。

这些权重有强烈的训练依赖。改变 bit length、backbone、噪声强度或数据域，损失尺度都会变化，不能把 `27.5` 当作通用最佳值。当前 API 的 `WM_STRENGTH` 是推理时对残差乘系数；它可以移动单张图的工作点，却不会重新训练 extractor 对这种强度分布的校准。

## 13. Recovery BCE 与“97% 准确率”的正确读法

bit accuracy 定义为：

$$
\operatorname{Acc}_{\mathrm{bit}}
=\frac1l\sum_{j=1}^{l}
\mathbb 1\left[(\bar w_j>0.5)=w_j\right].
$$

随机猜测期望是 0.5。论文 Table 1 的 `0.97` 表示平均 100 位中约 97 位正确，而不是 97% 样本的 100-bit payload 完全正确。论文在实验中没有应用 ECC，以便直接比较水印信号本身；当前软件包默认 `use_ECC=True`，业务层看到的是另一种指标。

对于带纠错的数据层，应至少报告四个量：raw bit accuracy、完整 payload 成功率、BCH correction count、false detection rate。只报告成功样本上的平均纠错数会遗漏失败；只报告 bit accuracy 会掩盖整包错误；只报告 detected ratio 又可能遗漏随机 bit 恰好构成有效 schema 的假阳性。

payload 分布也重要。论文每张图配随机 bit vector，使 0/1 大致平衡。实际系统若把短整数直接零填充，某些位高度固定，decoder 可能把先验当信号。当前 DataLayer 通过版本位、BCH 和 7-bit ASCII 建立结构，但业务仍应优先嵌入随机 opaque identifier，而不是邮箱、姓名、订单号等 PII。

## 14. YUV MSE 与 LPIPS：像素和感知空间互补

YUV loss 在亮度与色度空间比较 cover 和 encoded image：

$$
\mathcal L_{\mathrm{YUV}}
=\left\|\operatorname{YUV}(\mathbf x)-
\operatorname{YUV}(\mathbf y)\right\|_2^2.
$$

相比直接 RGB MSE，YUV 把亮度与颜色差异分开，能更贴近传统图像编码和人眼对亮度/色度的不同敏感性。它仍只是固定色彩变换后的均方误差，不包含显示设备、观看距离和局部视觉掩蔽的完整模型。

LPIPS 使用预训练网络的多层 feature distance，约束语义与纹理感知差异。它可容忍某些像素级小变化，又对结构性伪影更敏感。水印任务中的理想状态不是让 LPIPS 取代 PSNR，而是让局部像素、感知结构与频谱分布共同约束残差。

两种损失也可能冲突。把水印藏进高纹理区，像素误差未必更小，却可能更难被人看到；LPIPS 对极细高频彩噪的敏感度也不总与人一致。论文缺少用户研究，因此“imperceptible”主要由 PSNR/SSIM/定性图支持。生产上线前应在代表性显示设备、缩放比例和图像类别上做盲测，而不是把 40 dB 当作绝对不可见阈值。

## 15. Focal Frequency Loss

论文把 Focal Frequency Loss 写成：

$$
\mathcal L_{\mathrm{FFL}}(\mathbf x,\mathbf y)
=\rho_{f(\mathbf x),f(\mathbf y)}
\left\|f(\mathbf y)-f(\mathbf x)\right\|_2,
$$

其中 $f(\cdot)$ 是二维 Fourier transform，动态权重矩阵：

$$
\rho_{f(\mathbf x),f(\mathbf y)}
\in\mathbb R^{256\times256\times3}
$$

会强调当前难以重建的频率分量。普通像素损失把空间误差逐点平均，GAN 关心输出分布，FFL 则直接检查 cover 与 watermarked image 的频谱差异。对于水印这种低幅、可能在频谱上形成规律峰值的信号，这是很自然的约束。

“频谱更接近”不等于“水印只写在高频”。网络仍可在多个频段分配信号，动态权重也随样本变化。FFL 的作用是减少可被频域统计轻易识别的伪影，并提升 PSNR，而不是提供隐写安全证明。一个攻击者可以训练专门的 steganalyzer，利用相位、局部残差或神经特征，而不只看平均 magnitude spectrum。

补充 Fig. 9 会显示：没有 FFL 时，残差频谱在四个方向频带上更亮；加入 FFL 后这些结构减弱。Table 4 还表明 FFL 单独带来约 1.7 dB PSNR 增益。两者共同支持“FFL 对输出质量有贡献”，但没有独立安全实验说明它降低了水印可检测性。

## 16. WGAN-GP：让输出更像真实图像分布

论文的对抗项为：

$$
\begin{aligned}
\mathcal L_{\mathrm{GAN+GP}}(\mathbf x,\mathbf y)
=&\ \mathbb E_{\mathbf y\sim\mathbb P_E}[D(\mathbf y)]
-\mathbb E_{\mathbf x\sim\mathbb P_{\mathrm{real}}}[D(\mathbf x)]\\
&+\lambda_{\mathrm{GP}}
\mathbb E\left[
\left(\left\|\nabla D(\cdot)\right\|_2-1\right)^2
\right].
\end{aligned}
$$

判别器迫使 encoded image 在训练分布上接近真实 cover，gradient penalty 约束 critic 的梯度范数，缓解 WGAN 训练不稳定。对水印来说，GAN 不必“生成新内容”，而是惩罚由嵌入器产生、可被数据驱动判别器识别的统计伪影。

Table 4 显示单独加入 GAN 约提升 0.7 dB PSNR，与 $E_{\mathrm{post}}$、FFL 同时启用时总提升更大。这种协同意味着质量并非由一个指标决定：post-process 提供表达能力，FFL 修正频谱，GAN 修正数据分布。

GAN 也会增加训练成本与不确定性。判别器可能只覆盖 MIR-Flickr 的视觉统计，遇到医学图、线稿、UI 截图或科学图时，不一定给出合适梯度。论文的 control images 与跨域案例很有价值，但不是完整域泛化保证。若业务图像明显偏离自然照片，应独立验证，必要时重新训练或禁用自动嵌入。

## 17. 四阶段训练：先让系统学会“读”，再要求“看不见”

TrustMark 没有从第一步就同时打开随机数据、强噪声、GAN 和高质量权重。补充材料把训练拆成四个自动触发阶段：

![TrustMark 四阶段训练调度](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-table05-training-stages.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Table 5, arXiv non-exclusive distribution license.*

| Stage | Fixed batch | Random batch | Noise | $\alpha$ ramp | GAN | 触发 bit accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 开 | 关 | 关 | 关 | 关 | 初始状态 |
| 1 | 关 | 开 | 关 | 关 | 关 | $0.90$ |
| 2 | 关 | 开 | 开 | 关 | 关 | $0.95$ |
| 3 | 关 | 开 | 开 | 开 | 开 | $0.98$ |

**表解决的问题。** 端到端目标在训练初期存在明显的鸡生蛋问题：如果 extractor 还不会识别低幅信号，quality loss 和 noise 会逼 embedder 输出几乎原图，恢复梯度就可能消失；若先让 embedder 生成很强信号，又可能进入难以恢复高质量的局部最优。

**表中结构。** Stage 0 重复同一批 cover，只随机 payload，关闭 $N$ 与 GAN，并令 $\alpha=0.05$。准确率过 90% 后换成全数据随机 batch；过 95% 后加入噪声；过 98% 后开启 GAN，并在后续 10,000 iterations 把 $\alpha$ 线性拉到 20 或 27.5。论文称典型 TrustMark-B 约在 500、600、800 iteration 进入后三阶段。

**支持的结论。** 这张表说明论文性能不只来自网络模块，curriculum 本身是可训练性的关键。Table 4 中不同 backbone 的差异也要在相同阶段策略下解释，不能把训练失败简单归因于容量不足。

**不能证明的内容。** 固定的 90/95/98% 阈值并非普适。batch 统计会抖动，bit length 与噪声强度改变后阈值对应的任务难度也不同。公开仓库没有完整训练循环，读者不能仅凭推理代码重建这个状态机。

这种两时间尺度思路值得迁移到其他隐蔽信号任务：先建立可读通道，再逐步施加视觉与鲁棒约束。但工程实现应保存当前 stage、触发指标的滑动统计和 $\alpha$ 进度到 checkpoint，避免恢复训练后阶段错位。

## 18. Algorithm 1：Residual Resolution Scaling 逐行精读

![TrustMark 任意分辨率残差缩放算法](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-algorithm01-resolution-scaling.webp)

*Source: Bui et al., arXiv:2311.18297v1, Algorithm 1, arXiv non-exclusive distribution license.*

令原图为 $\mathbf x_{H,W}$。算法先把像素从 $[0,255]$ 归一化到 $[-1,1]$：

$$
\mathbf x\leftarrow \mathbf x/127.5-1.
$$

然后插值得到固定模型输入：

$$
\bar{\mathbf x}=\operatorname{Resize}(\mathbf x,256,256).
$$

嵌入时在小图上计算：

$$
\mathbf r=E(\bar{\mathbf x},\mathbf w)-\bar{\mathbf x};
$$

移除时则计算：

$$
\mathbf r=R(\bar{\mathbf x})-\bar{\mathbf x}.
$$

残差被放回目标分辨率：

$$
\mathbf r_{H,W}=\operatorname{Resize}(\mathbf r,H,W),
$$

最终输出：

$$
\mathbf y=\operatorname{clamp}
(\mathbf x+\mathbf r_{H,W},-1,1).
$$

**算法解决的问题。** 如果直接把网络输出的 $256^2$ 图像整体放大，高分辨率原图细节会被下采样永久破坏。残差缩放只搬运模型预测的低幅变化，把原始高分辨率像素作为基底，因此主体纹理、文字与边缘不必经过低分辨率瓶颈。

**算法中的关键假设。** 水印残差在空间上足够平滑或冗余，插值后仍可被把高分辨率图缩回解码分辨率的 extractor 识别；remover 在小图预测的反向残差也应在放大后削弱信号。训练中的 resize 噪声正是在支撑这个假设。

**支持的结论。** 补充 Fig. 10/11 显示，Resolution Scaling 在不同输出尺寸上的 PSNR 下降远小于“先生成整张 $256^2$ 图再双线性/双三次放大”。这说明保留原图、只放大 residual 的策略有效。

**不能外推的内容。** 网络没有在原生高分辨率上观察细小对象。对于极窄长图，统一压到方形会严重扭曲；当前代码因此加入长宽比阈值、中心区域策略和 P 模型强制方形 crop，这些都是论文后的补丁。算法也不能保证无限尺寸：当 $H,W$ 极大，同一 $256^2$ residual 被扩散到更大面积，局部强度、频率和解码预处理都会变化。

补充材料还引入运行时系数 $\lambda$：

$$
\mathbf y\leftarrow
\operatorname{clamp}(\mathbf x+\lambda\mathbf r,-1,1).
$$

$\lambda<1$ 提高图像质量但降低恢复率，$\lambda>1$ 则增强水印。论文建议较合理区间约为 $[0.75,1.50]$。这相当于当前 API 的 strength 思路，但部署仍应按模型与数据集标定，不能把上界视为安全范围。

## 19. TrustMark-RM：把水印当作可学习噪声

TrustMark-RM 的骨干来自 KBNet。其 kernel basis attention 与多轴特征融合面向图像恢复，目标不是预测 payload，而是把带水印图还原为 cover。训练时冻结 $E$ 和 $X$，对每张 cover 采样 $n=3$ 个随机 watermark：

$$
\mathbf y_i=E(\mathbf x,\mathbf w_i),\qquad i=1,2,3,
$$

然后优化：

$$
\hat{\mathbf x}_i=R(\mathbf y_i).
$$

使用多个随机 payload 很重要。若每张 cover 只有一个固定 watermark，remover 可能记忆某种 payload-specific residual；随机化迫使它学习 embedder 产生的信号分布。冻结 $E/X$ 则固定了攻击目标，使 $R$ 的训练不会通过共同适应让解码器变得更容易或更困难。

把 removal 称为“逆过程”只能是直觉。$E$ 是内容相关、非线性且经过 clamp 的映射，不保证可逆；watermarked image 也可能丢掉少量 cover 信息。$R$ 实际学习条件期望意义下的恢复，因此输出会包含 restoration bias。论文在多轮重水印中观察到 remover noise 累积，正是这种非精确逆映射的结果。

TrustMark-RM 还有双重产品含义。正面看，它让创作工具可以在版本迁移或所有权变更时移除旧标识后写入新标识，减少残差叠加。安全看，它展示只要能生成训练对，专用恢复网络就能把 decode accuracy 压向随机。一个水印方案的威胁模型不能假设攻击者只会用 JPEG，而不会训练神经移除器。

## 20. Removal Loss：恢复视觉与破坏解码是两个目标

论文给 TrustMark-RM 三个约束。首先是像素重建：

$$
\mathcal L_{\mathrm{MSE}}^R
=\left\|R(\mathbf y)-\mathbf x\right\|_2^2,
$$

直接提高 PSNR。其次是与 TrustMark 类似的 $\mathcal L_{\mathrm{GAN+GP}}^R$，让恢复图落回真实图像分布。第三是 extractor response loss：

$$
\mathcal L_X^R
=\left\|X(R(\mathbf y))-X(\mathbf x)\right\|_2^2.
$$

原始 cover 对一个没有嵌入信号的 extractor 应接近随机响应，所以让恢复图与 cover 的响应一致，可以专门消除模型仍能读取的残留。单用 MSE 得到高 PSNR，不一定足以破坏 watermark；反过来，I-FGSM 可以降低 bit accuracy，却明显损害图像。三项分别对应“像原图”“像自然图”“不再像有效水印”。

这里仍没有密码学意义的删除证明。bit accuracy 降到 0.553 只是针对指定 extractor 和评测预处理；攻击者可能使用不同 decoder、集成模型或残差信号检测器发现水印痕迹。对数据删除合规而言，模型响应变随机也不等于关联数据库中的记录、C2PA manifest 或缓存已被删除。

## 21. Re-watermarking：覆盖旧信号为什么不够

直接反复调用 $E$ 写入不同 payload，会把多个低幅残差叠加在同一图像上。新 payload 仍能恢复，因为最后一次信号占主导，但 PSNR 会持续下降。理想工作流是：

$$
\mathbf y^{(k+1)}
=E\left(R(\mathbf y^{(k)}),\mathbf w_{k+1}\right).
$$

TrustMark-RM 先削弱旧信号，再写新信号，能明显减缓质量劣化。然而 $R$ 本身不是恒等映射，每轮会加入恢复偏差；多轮后这份偏差也会积累。论文因此没有声称 re-watermarking 可无限次执行。

业务上更稳妥的做法，是始终保留无水印 master asset，并从 master 派生每个发行版本。只有拿不到 master 时才运行 remove-and-reencode。若水印代表不可变的发行事件，还应给每次写入建立 manifest 版本、旧标识撤销关系和像素文件哈希，而不是让图像内最后一个 payload 成为唯一历史。

## 22. 实验设置与复现成本

TrustMark 使用 MIRFlickr 1M 中的 101K 张图训练，其中 1K 用作 validation。评测数据包括 CLIC、DIV2K 和 MetFace。DIV2K 使用公开的 800 张训练图与 100 张验证图，因为官方 test ground truth 不公开；它分辨率高、内容更丰富，也是三者中最难的一组。MetFace 领域窄且图像均为方形，若某 baseline 对非等比例 resize 不鲁棒，可能在 MetFace 上显得异常好。

训练 TrustMark 150 epochs，batch size 32，AdamW 初始学习率按“每张图 $4\times10^{-6}$”描述，并使用 cosine annealing。单张 GeForce RTX 3090 加标准 Intel i7 约需 48 小时。TrustMark-RM 使用 $n=3$、batch size 8、100 epochs，在 A100 上约两周。论文报告 RTX 3090 平均 encode/decode 为 125/25 ms。

这些数字有三个口径。第一，encode/decode 时间不等于端到端资产流水线时间，未含模型首次下载、ECC、文件读写、色彩转换和网络服务。第二，RM 训练耗时远高于 embedder，说明 restoration backbone 与多水印数据合成成本不可忽略。第三，论文没有给完整 random seed、数据划分文件、环境锁定和训练 checkpoint 生成脚本；当前仓库以推理为主，不能直接复现实验。

论文与 baseline 的公平性也需细读。作者在其噪声设置下重训 baseline，前提是重训能改善性能；推理时对多数 fixed-resolution 方法统一应用 Resolution Scaling，但 RivaGAN 与 dwtDctSvd 原生支持可变分辨率，不套该步骤。这比直接抄原论文数字更合理，却仍受不同模型容量、payload 长度和训练 recipe 影响。

## 23. 指标精读：PSNR、SSIM 与 Bit Accuracy

PSNR 基于均方误差：

$$
\operatorname{PSNR}
=10\log_{10}\frac{L^2}{\operatorname{MSE}(\mathbf x,\mathbf y)},
$$

其中 $L$ 为像素动态范围。它对全图平均微小误差敏感，却可能掩盖局部明显条纹。SSIM 比较局部亮度、对比度和结构，仍不是完整感知模型。

bit accuracy 已在前文定义。`clean` 在无额外扰动的 watermarked image 上测，`noised` 经过随机噪声组合后测。均值旁的标准差反映跨样本变化；某些方法平均值接近随机且方差小，说明几乎系统性失败，而不是少数极端样本拉低结果。

论文补充材料还指出 PSNR 对评测分辨率敏感。对两张很不同的图，插值前后 PSNR 变化较小；对 cover 与其微小水印版本，重采样会平滑残差，PSNR 可变化数 dB。因此，如果两个固定分辨率模型各自在自己的设计分辨率上报告 PSNR，数字并不直接可比。Resolution Scaling 的另一价值，是让所有结果在原始输出尺寸与原图比较。

这些指标没有覆盖 false positive。一个没有水印的图被误判为有效 payload，在 provenance 系统里可能比偶尔漏检更严重。论文也没有大规模用户研究、stealability、payload collision 和 calibrated confidence。工程验收必须补这些维度。

## 24. Table 1：三数据集主结果

![TrustMark 与五个 baseline 的主结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table01-main-benchmarks.webp)

*Source: Bui et al., arXiv:2311.18297v1, Table 1, arXiv non-exclusive distribution license.*

图像质量的关键数字如下：

| Method | CLIC PSNR | DIV2K PSNR | MetFace PSNR |
| --- | ---: | ---: | ---: |
| TrustMark-Q | **43.26** | 42.39 | **45.34** |
| TrustMark-B | 41.53 | 40.20 | 43.87 |
| RoSteALS | 30.03 | 27.95 | 33.77 |
| RivaGAN | 41.04 | 41.06 | 40.98 |
| SSL | 42.74 | **42.73** | 42.84 |
| StegaStamp | 37.48 | 35.87 | 39.35 |
| dwtDctSvd | 39.13 | 38.02 | 41.14 |

扰动后的 bit accuracy：

| Method | CLIC noised | DIV2K noised | MetFace noised |
| --- | ---: | ---: | ---: |
| TrustMark-Q | 0.95 | 0.95 | 0.96 |
| TrustMark-B | **0.97** | **0.97** | **0.97** |
| RoSteALS | 0.94 | 0.93 | 0.93 |
| RivaGAN | 0.79 | 0.78 | 0.82 |
| SSL | 0.60 | 0.57 | 0.70 |
| StegaStamp | 0.72 | 0.70 | 0.72 |
| dwtDctSvd | 0.52 | 0.51 | 0.52 |

**表解决的问题。** Table 1 同时评估不可见性与鲁棒性，避免只用 PSNR 或只用恢复率宣布胜出。

**表中数字。** Q 在 CLIC 与 MetFace 的 PSNR 最好，DIV2K 与 SSL 接近；B 比 Q 低约 2 dB，却在三组 noised accuracy 上都达到 0.97。clean 状态下 TrustMark 几乎 1.00。dwtDctSvd clean 很高，noised 却接近随机，说明“能写入”与“能经分发保留”是两件事。

**支持的结论。** TrustMark 提供了两个可选工作点，且在作者统一噪声协议下呈现更好的联合 Pareto 位置。DIV2K 的 PSNR 和准确率略低，也支持高质量、多样内容更难嵌入的判断。

**不能证明的内容。** 表格没有 ECC、完整 payload 成功率、false positive 或主动攻击。baseline 的架构年代、容量和原生分辨率不同；统一 Resolution Scaling 降低了比较偏差，但没有消除训练 recipe 差异。数字不能直接变成“任何平台 97% 可恢复”的 SLA。

## 25. Fig. 2：定性结果与 20x Residual

![TrustMark 与 baseline 的定性水印和残差比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig02-qualitative-comparison.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 2, arXiv non-exclusive distribution license.*

**图解决的问题。** 平均 PSNR 无法展示误差是否形成可见条纹、颜色偏移或对象轮廓。Fig. 2 对 CLIC、DIV2K、MetFace 的 cover、watermarked image 和放大 20 倍 residual 做并排比较。

**图中现象。** TrustMark residual 更接近灰色细噪声，较少直接复现对象轮廓；部分 baseline 在平坦区域、边缘或颜色通道上形成更明显结构。原尺度 watermarked images 肉眼接近 cover，而放大残差揭示网络仍在全图分配信号。

**支持的结论。** 图像说明 post-process、FFL 与多损失联合训练确实改变了残差统计，不只是提高一个数值。内容自适应嵌入也比固定频域模板更能利用纹理掩蔽。

**不能证明的内容。** “看不见对象轮廓”不等于不可检测；深度 steganalysis 可以利用人眼不可见统计。论文选取的是代表性图而非盲随机网格，缺少主观 A/B。残差乘 20 的可视化还会受显示缩放、颜色映射和浏览器压缩影响。

## 26. Table 2：统一 ImageNet-C 协议

![TrustMark 在 ImageNet-C 噪声配置下的对比](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table02-imagenet-c.webp)

*Source: Bui et al., arXiv:2311.18297v1, Table 2, arXiv non-exclusive distribution license.*

| Method | PSNR | Noised bit accuracy |
| --- | ---: | ---: |
| TrustMark, $\alpha_{\max}=15$ | 38.87 | **0.95** |
| RoSteALS | 32.68 | 0.94 |
| StegaStamp | 31.26 | 0.88 |
| SSL | **41.84** | 0.62 |
| RivaGAN | 40.32 | 0.77 |
| dwtDctSvd | 38.96 | 0.61 |

**表解决的问题。** 主实验采用 TrustMark 自己的 18 类噪声，可能被质疑对本方法更有利。Table 2 改用 RoSteALS/ImageNet-C 配置并关闭 Resolution Scaling，给出另一个统一协议。

**表中数字。** SSL 的 PSNR 最高，却只有 0.62 noised accuracy；TrustMark PSNR 为 38.87，恢复率 0.95。RoSteALS 恢复率接近 0.94，但 PSNR 被其 frozen autoencoder 质量上限限制在约 33 dB。

**支持的结论。** TrustMark 的优势不只来自自定义几何噪声；在另一套噪声族上仍保持较强联合表现。表格也再次说明单看 PSNR 会偏向几乎不写强信号的方法。

**不能证明的内容。** ImageNet-C 本是分类鲁棒性扰动集合，并非完整社交平台处理链，而且缺少几何变换。关闭 Resolution Scaling 后的数字不能与主 Table 1 直接混合。$\alpha_{\max}=15$ 也不是 B/Q 工作点，说明模型重新按协议选择了质量权衡。

## 27. Fig. 4：Bit Length、Noise Severity 与白盒攻击

![TrustMark payload 长度、噪声强度和对抗攻击三面板](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig04-payload-noise-adversarial.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 4, arXiv non-exclusive distribution license.*

**图解决的问题。** 主表固定 payload 与噪声，只给一个工作点。Fig. 4 同时扫描容量、训练噪声强度和 I-FGSM 攻击迭代数，揭示系统如何在三个压力轴上退化。

**图中数字。** payload 从 32 增至 200 bits，约 6 倍容量使 PSNR 下降约 7.5 dB，bit accuracy 下降约 11%。不使用 noise simulation 时 PSNR 可到 53.2 dB；提高到 high severity 后约为 40.2 dB，而 bit accuracy 只下降约 3%。白盒攻击中，无噪声训练模型几乎都能在 50 iteration 内被打到阈值；high-severity 模型有 32% 样本需超过 3000 iteration。

**支持的结论。** 更强噪声训练会主动牺牲视觉质量，把水印放得更稳；容量不是免费参数；经验鲁棒训练对特定梯度攻击也能提高成本。这些结果解释了为何 B/Q 要提供不同工作点。

**不能证明的内容。** 迭代数不等于通用安全强度，依赖步长、$\epsilon=8/255$、停止条件和攻击者模型。对 transfer attack、替代 decoder、remover network 或几何搜索，趋势可能不同。200-bit 实验也不代表当前软件包的 100-bit checkpoint 能任意切换长度。

## 28. Fig. 5：逐扰动鲁棒性不能被平均值替代

![TrustMark 对十五类单独扰动的恢复率](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig05-individual-perturbations.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 5, arXiv non-exclusive distribution license.*

**图解决的问题。** 一个模型可能在多数轻微颜色变化上很好，却被一种常见 blur 或 RGB shift 完全击穿；平均 noised accuracy 会掩盖这种单点失败。Fig. 5 分别施加每类扰动。

**图中现象。** TrustMark-Q 在大多数来源上优于最接近的 RoSteALS，例外是 Gaussian noise 与 box blur。dwtDctSvd 对 RGB shift 接近随机，SSL、StegaStamp、RivaGAN 各有明显弱项。几何变换和色彩变换对不同模型的影响模式并不一致。

**支持的结论。** 覆盖更广的 differentiable noise family 提高了鲁棒性分布的均衡程度，而不仅是对 JPEG 特化。图也为工程测试矩阵提供模板：应分项记录而非只看汇总。

**不能证明的内容。** 每一柱仍是固定强度范围下的单变换。现实操作经常复合，例如 crop 后 resize 再 JPEG；组合效应可能非线性。论文没有给平台特定转码、截图、打印扫描和滤镜链，因此不能从这张图推出实际渠道保证。

## 29. Table 3：TrustMark-RM 与 I-FGSM

![TrustMark-RM 水印移除结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table03-removal.webp)

*Source: Bui et al., arXiv:2311.18297v1, Table 3, arXiv non-exclusive distribution license.*

| Method | PSNR | SSIM | Bit accuracy |
| --- | ---: | ---: | ---: |
| TrustMark-RM | **48.48** | **0.997** | **0.553** |
| I-FGSM | 23.48 | 0.613 | 0.629 |

**表解决的问题。** 白盒 adversarial attack 可以降低 decoder accuracy，却可能把图像改坏。Table 3 比较“专门恢复图像并取消信号”和“只优化攻击目标”的质量差异。

**表中数字。** TrustMark-RM 在 DIV2K/TrustMark-B 上达到 48.48 dB 与 0.997 SSIM，同时把 bit accuracy 降到 0.553；I-FGSM 只有 23.48 dB、0.613 SSIM，且 bit accuracy 仍为 0.629。

**支持的结论。** 训练一个 restoration network 比逐图梯度攻击更适合重水印工作流：既能削弱原 extractor，又能保留视觉质量。它也实证了水印信号可被当作结构化弱噪声学习。

**不能证明的内容。** 0.553 仍高于 0.5，论文认为 Resolution Scaling 削弱了 remover 效力。实验只针对同族 B 模型，不能说明对所有 watermark detector 都“删除干净”。I-FGSM 参数也不必然是最佳攻击，所以表格不是防御强度排行榜。

## 30. Fig. 3 与 Fig. 7：多轮重水印的质量账本

![有无 TrustMark-RM 时多轮重水印曲线](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig03-rewatermarking-curves.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 3, arXiv non-exclusive distribution license.*

![TrustMark 一次、五次和十次重水印示例](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig07-rewatermarking-example.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 7, arXiv non-exclusive distribution license.*

**两图解决的问题。** Fig. 3 给出迭代次数与 PSNR/bit accuracy 曲线，Fig. 7 把 1、5、10 次重水印的真实图与放大残差摆出来；一个提供整体趋势，一个展示误差形态。

**图中现象。** 无 remover 时 PSNR 随次数快速下降；每次先用 TrustMark-RM 可以维持更高质量。两条路径的最新 payload bit accuracy 都大致稳定，说明旧信号叠加主要伤害视觉质量，而不立即阻止最后一个 payload 被读出。定性图中多轮后残差扩大，RM 路径仍不是完全无损。

**支持的结论。** remove-then-encode 比直接覆盖更适合资产迭代，且 remover 的价值主要体现在质量而非当前 payload recovery。论文没有只展示一次理想重写，而是追踪到 20 次，暴露长期累积。

**不能证明的内容。** 测试不等于真实编辑链。多轮之间没有加入重新裁切、调色或生成式编辑；remover 也只针对 TrustMark。同一图片被不同组织、不同模型族反复写入时，残差和解码器干扰可能不同。生产系统仍应保留 master 并限制重写次数。

## 31. Table 4：GAN、FFL、E_post 与 Backbone 消融

![TrustMark 架构与损失消融](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-table04-ablation.webp)

*Source: Bui et al., arXiv:2311.18297v1, Table 4, arXiv non-exclusive distribution license.*

质量模块的精简转录：

| GAN | FFL | $E_{\mathrm{post}}$ | PSNR | Clean acc. | Noised acc. |
| --- | --- | --- | ---: | ---: | ---: |
| 关 | 关 | 关 | 36.30 | 0.991 | 0.955 |
| 开 | 关 | 关 | 36.99 | 0.990 | 0.958 |
| 关 | 开 | 关 | 38.16 | 0.989 | 0.973 |
| 关 | 关 | 开 | 37.91 | 1.000 | 0.970 |
| 开 | 开 | 开 | **40.30** | 0.999 | **0.977** |

Extractor backbone 部分表明无 residual connection 的普通 convolutional 结构难以达到同等水平，而 ResNet18/34/50/101 之间的差距并非随深度单调增大。论文最终选择 ResNet50，是容量、收敛与准确率之间的经验点。

**表解决的问题。** 主模型包含多个同时变化的模块，Table 4 通过组合开关证明增益不是单一 GAN 或更深 decoder 带来的。

**表中数字。** 三项全关时约 36.30 dB；单开 GAN、FFL、post-process 分别约增加 0.7、1.7、1.6 dB；三项全开到 40.30 dB，并把 noised accuracy 从 0.955 提到 0.977。$E_{\mathrm{post}}$ 与 FFL/GAN 有协同。

**支持的结论。** 频谱、通道重映射和分布监督确实互补。阶段训练使多数变体 clean accuracy 都接近 1，质量指标更能区分结构设计。

**不能证明的内容。** 消融只在 DIV2K、$\alpha_{\max}=20$ 和固定训练预算上进行。移除模块后若重新调权重或加训练时长，结果可能变化。表格也没有参数量、FLOPs 和 latency 的逐项开销，无法判断每 1 dB 的工程成本。

## 32. Fig. 6：训练时 alpha_max 的质量-鲁棒曲线

![TrustMark 训练损失权重对 PSNR 与恢复率的影响](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig06-quality-robustness-tradeoff.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 6, arXiv non-exclusive distribution license.*

**图解决的问题。** B 与 Q 只是两个离散名称，Fig. 6 扫描 $\alpha_{\max}$，展示完整工作曲线。

**图中现象。** 当 $\alpha_{\max}$ 增到约 30，PSNR 提升超过 8 dB，而 bit accuracy 下降不到 5%。论文认为稳定训练的有效范围约为 $(0,30)$。曲线阴影表明不同样本的变化并不完全一致。

**支持的结论。** TrustMark 的质量/鲁棒性不是单一最优，而是可通过训练配置选择。B 偏 balanced，Q 偏 quality；部署应按渠道扰动和视觉要求选择，而不是默认 Q 永远更好。

**不能证明的内容。** 横轴是训练时损失权重，不是运行时残差强度。模型需要重新训练才能位于不同点；补充 $\lambda$ 才是单模型运行时控制。图也没有完整 payload success、ECC 和人眼阈值，不能仅靠曲线自动确定产品配置。

## 33. Fig. 8：密集纹理既能掩蔽，也会干扰恢复

![TrustMark 在密集纹理和杂乱图像上的局限](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-fig08-limitations-clutter.webp)

*Source: Bui et al., arXiv:2311.18297v1, Fig. 8, arXiv non-exclusive distribution license.*

**图解决的问题。** 很多水印展示偏爱平滑、构图清楚的自然图。Fig. 8 专门给出毛发、重复细线和密集植被等 cluttered images。

**图中现象。** 高纹理提供视觉掩蔽空间，却也与低幅水印共享高频带，使 embedder 更难维持高 PSNR、extractor 更难把信号从内容中分离。放大残差在复杂边缘附近更明显，准确率也略低。

**支持的结论。** 水印难度是内容相关的。固定全局 strength 不是所有图片的最优策略，应该建立 per-image quality 与 confidence 检查。

**不能证明的内容。** 三个案例不足以描述失败率。论文没有按纹理、颜色、边缘密度或语义类别分桶，也没有给拒绝嵌入阈值。生产系统若只汇总总体 decode rate，会让低频出现但高风险的内容类型被平均掉。

## 34. 补充 Fig. 9-11：频谱与跨分辨率证据

### 34.1 FFL 的频谱效果

![TrustMark 加入 FFL 前后的平均频谱残差](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig09-frequency-ffl.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 9, arXiv non-exclusive distribution license.*

**问题。** Table 4 只显示 FFL 提高 PSNR，不说明频域结构怎样变化。

**结构。** 左图是 MetFace cover 的平均频谱，中间和右边是无/有 FFL 的放大残差频谱。加入 FFL 后四个方向的亮带减弱，误差更均匀。

**结论。** 动态频率损失确实作用到设计目标，而不只是作为额外正则偶然提高像素分数。

**边界。** 平均频谱会消除个体差异，也不覆盖相位和局部频谱。它不能证明 steganalysis 失败。

### 34.2 水印的 Resolution Scaling

![TrustMark 水印在不同目标分辨率下的插值比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig10-resolution-scaling-watermarking.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 10, arXiv non-exclusive distribution license.*

**问题。** 比较 residual scaling 与直接对完整 $256^2$ 输出做 bilinear/bicubic interpolation。

**结构。** 随输出接近 DIV2K 原始分辨率，直接插值的 PSNR 明显下降；Resolution Scaling 只下降约 0.1 dB。bit accuracy 稍低于 bicubic 约 1 个百分点。

**结论。** 保留原图像素、只迁移 residual 能显著减少高分辨率细节损失；它是“任意分辨率”主张最直接的定量支撑。

**边界。** 结果是从同一 2K 数据向下采样，不包含超大图、极端长宽比或局部 crop embedding。bit accuracy 的 1% 损失仍可能显著影响整包成功率。

### 34.3 移除器的 Resolution Scaling

![TrustMark-RM 在不同目标分辨率下的插值比较](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig11-resolution-scaling-removal.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 11, arXiv non-exclusive distribution license.*

**问题。** 验证同一黑盒 residual 方法是否也适用于 removal，而不是只对 embedder 有效。

**结构。** Resolution Scaling 的恢复 PSNR 同样相对稳定；直接放大完整低分辨率恢复图会丢失原图细节。removal bit accuracy 的方向与嵌入相反，越接近 0.5 越好。

**结论。** 算法对“输出与输入之差是目标修改”的图像到图像模型具有一定通用性，因而能复用到 $R$。

**边界。** remover 的低分辨率预测未必包含高分辨率水印的全部局部残差；论文自身已观察 scaling 后准确率不能完全到 0.5。把它扩展到其他水印算法需要对 resize 鲁棒性重新验证。

## 35. 补充 Fig. 12-16：模型在看哪里，运行时怎样变化

### 35.1 Pre/Post Activation Maps

![TrustMark 嵌入器预处理与后处理激活图](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig12-activation-maps.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 12, arXiv non-exclusive distribution license.*

**图解决的问题。** 网络是否只是把固定噪声模板叠在所有图上，还是会根据图像内容分配信号？补充激活图展示 early fusion 后与 post-process 前后的中间通道。

**图中现象。** 一些通道响应对象边缘、纹理和亮度区域，另一些更像全局低幅模式；post-process 将这些通道混合为 RGB residual。激活不是可直接解释的“某 bit 对应某物体”，而是内容与 payload 的联合表征。

**支持的结论。** 多通道 $1\times1$ pooling 有真实可用的输入，不只是增加参数；嵌入残差具有内容自适应性。

**不能证明的内容。** 激活可视化没有因果性，挑选通道也可能强化想看到的模式。它不说明攻击者无法从中训练检测器，也不证明语义对象信息已被彻底去除。

### 35.2 Control Images

![TrustMark 在纯色、几何纹理和条纹控制图上的结果](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig13-control-images.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 13, arXiv non-exclusive distribution license.*

**图解决的问题。** 自然图像有丰富纹理可以掩蔽水印，极简纯色和规则几何图更能暴露结构性伪影。

**图中结构。** 第一行是 host，第二行写入同一随机 watermark，第三行把 residual 放大 20 倍。论文报告这些 control images 的 bit accuracy 均为 100%。

**支持的结论。** 模型不是只在 MIR-Flickr 风格自然照片上才能传递信号；即使输入缺少语义纹理，decoder 也能恢复。

**不能证明的内容。** 规则图是少量手工案例，100% 是这些样本的 clean recovery，不代表视觉质量或扰动后鲁棒性。纯色区域上低幅条纹反而可能更容易被人察觉，图中缺少盲测。

### 35.3 Noise Severity 参数表与实例

![TrustMark 低中高三档噪声模拟参数](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-table06-noise-severity.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Table 6, arXiv non-exclusive distribution license.*

| 扰动族 | Low | Medium | High |
| --- | --- | --- | --- |
| JPEG | 高质量区间 | 更宽质量区间 | 可到更低质量 |
| Resize / Crop | 小幅变化 | 中等变化 | 更强尺寸与区域变化 |
| Blur | 小 kernel / sigma | 中等 kernel / sigma | 更大 kernel / sigma |
| Color | 轻微亮度色相 | 扩大抖动 | 更强亮度、对比度、饱和度 |
| Noise / Posterize | 低幅、较多 bit | 中等幅度 | 高幅、较少 bit |

表格中的具体上下界通过 uniform random sampling 抽取；JPEG 的 $q$ 表示质量因子，blur 的 $k$ 表示 kernel，motion blur 的角度和方向另行采样。这里用精简表表达族级变化，完整数字保留在原表图中。

![TrustMark 训练时不同噪声来源的视觉示例](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig14-noise-examples.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 14, arXiv non-exclusive distribution license.*

**图表解决的问题。** “高噪声”若没有参数和视觉例子就不可复核。Table 6 定义采样范围，Fig. 14 展示 crop、resize、JPEG、颜色、blur、noise 等实际强度。

**结构或数字。** 三档不仅改变某一个 sigma，而是共同扩大多个变换的范围。训练样本仍只抽两个 optional transforms，因而“High”不等于一次把 15 类全部叠加。

**支持的结论。** Fig. 4 的 quality/robustness 曲线对应明确的数据分布变化，而不是抽象的强弱标签。它也为后续复现者提供了参数基线。

**不能证明的内容。** 参数范围并不代表真实平台频率。若生产中 90% 是 JPEG、10% 是截图，uniform sampling 可能浪费容量；不同图像格式的 JPEG 实现也不相同。最佳训练分布应由业务渠道数据决定。

### 35.4 运行时 Strength

![TrustMark 运行时残差系数对 PSNR 与 bit accuracy 的影响](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig15-inference-strength.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 15, arXiv non-exclusive distribution license.*

**图解决的问题。** 训练一个 checkpoint 后，是否还能按单图调整工作点？

**图中现象。** $\lambda$ 从小到大时 PSNR 下降、bit accuracy 上升，并逐渐饱和。论文认为约 $0.75$ 到 $1.50$ 兼顾两者。

**支持的结论。** Residual Scaling 暴露了一个便宜、无需重训的样本级控制旋钮。当前代码的 `WM_STRENGTH` 正是这一工程方向，并对 P 变体额外乘 1.25。

**不能证明的内容。** 同一 $\lambda$ 对平坦图与复杂纹理图的可见性不同；曲线是 DIV2K 平均值，不是 per-image 安全阈值。系数过大还会触发 clamp，改变信号线性关系。

### 35.5 PSNR 的分辨率依赖

![水印图像 PSNR 随评测分辨率变化](/images/blog/trustmark-universal-watermarking-arbitrary-resolution/trustmark-supp-fig16-psnr-resolution-dependence.webp)

*Source: Bui et al., arXiv:2311.18297v1, Supplementary Fig. 16, arXiv non-exclusive distribution license.*

**图解决的问题。** 为什么 fixed-resolution 水印论文之间的 PSNR 可能不可直接比较？

**图中数字。** 两张完全不同图像的 PSNR 随 resize 最大变化约 0.35 dB；cover 与其 dwtDctSvd watermarked version 的 PSNR 可变化约 4.7 dB。微小水印残差很容易被插值平滑，所以在缩小图上评测会人为抬高质量。

**支持的结论。** 在原始分辨率计算指标是必要的，Resolution Scaling 不只是输出漂亮大图，也统一了评测坐标。

**不能证明的内容。** 实验只用了一个水印方法和一对示例图。它说明偏差方向和量级可能很大，不提供对所有数据集的校正公式。

## 36. 补充表格的复现意义

主文给出模型模块，补充 Table 5/6 才提供训练状态机和噪声范围。对于 TrustMark，这两类信息比再多一张定性图更接近“可复现合同”：

1. stage transition 决定哪些损失在什么时候开始产生梯度；
2. $\alpha$ ramp 的起点、终点和持续 iteration 决定 B/Q 工作点；
3. noise family 的参数上界决定模型学到哪类鲁棒性；
4. 每样本只抽两个 optional transforms，决定复合扰动覆盖；
5. fixed batch warmup 决定 extractor 冷启动是否稳定。

但它们仍不是完整 recipe。论文没有发布训练集文件列表与随机种子，也没有提供全部 optimizer group、augmentation 顺序、checkpoint selection 和失败 run 统计。当前仓库的 [`TrustMark_Arch`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/model.py#L31-L87) 仍保留 `bit_acc_thresholds=[0.9,0.95,0.98]`、fixed input 与手动优化等痕迹，说明论文训练逻辑部分留在代码中；但没有数据与完整入口，静态对照只能验证设计一致性，不能证明复现。

## 37. 论文-源码主链：从 API 到模型再回到 PIL

当前 Python 推理路径可以概括为：

```text
TrustMark(...)
  -> DataLayer(secret_len=100, encoding_type=...)
  -> load config/checkpoint and verify MD5
  -> encode(): payload -> 100 bits -> encoder -> residual scaling -> PIL
  -> decode(): crop/rotate/detect -> decoder logits -> BCH -> payload
  -> remove_watermark(): remover -> residual scaling -> PIL
  -> localize(): optional detector -> normalized boxes
```

构造器在 [`trustmark.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L73-L139) 中选择 `C/Q/B/P`、device、ECC schema、区域比例、是否加载 remover 和 bbox detector。默认会加载 encoder、decoder 与 remover；这意味着“只想 decode”仍可能承担不必要的模型下载，调用者应显式检查初始化参数和当前包版本。

`encode()` 在 [`L465-L507`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L465-L507) 执行数据层编码、区域裁切、$256^2$ resize、模型前向和残差放大。当前实现比论文 Algorithm 1 多了一步：按 RGB channel 去掉 residual mean，以减小全局 color shift。然后按 `WM_MERGE` 插值、乘 `WM_STRENGTH`，再把结果贴回原图。

`decode()` 在 [`L400-L462`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L400-L462) 可选尝试四个旋转方向，也可先运行 bbox detector，再对候选区域逐个解码。普通路径仍使用与 encode 相同的中心区域规则，resize 到 decoder resolution 后以 logit $>0$ 得到 raw bits。

`remove_watermark()` 在 [`L509-L527`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L509-L527) 复刻 Algorithm 1：在 256 方图运行 remover，取输出减输入的残差，插值回处理区域，再贴回原图。代码结构证明论文所谓黑盒 residual scaling 已成为 encode/remove 共用的部署模式。

这条主链仍没有处理所有图像语义。PIL 图像若是 CMYK、16-bit、带 alpha 或带 ICC profile，需要调用者确认转换与保存。示例覆盖 JPEG、GenAI JPEG 和 RGBA PNG，但一个示例不能建立完整格式保证。生产系统应在 API 外固定输入色彩空间、alpha 策略、EXIF orientation 和输出编码参数。

### 37.1 `model.py`：训练容器与推理拆分

[`TrustMark_Arch`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/model.py#L31-L133) 是一个 PyTorch Lightning module，内部持有 encoder、decoder、loss、discriminator 与可选 noise。它把 `automatic_optimization` 关闭，表明训练时需要手工控制生成器和判别器更新；`fixed_input`、`update_gen`、`bit_acc_thresholds` 则对应补充 Table 5 的阶段状态。`get_input()` 在 warmup 时缓存第一批图像，却继续接收新的 secret，这正是“固定 cover、随机 payload”的源码实现。

`forward()` 返回 `(stego,residual)`。若具体 encoder 声明 `return_residual`，容器会做 `cover + enc_out`；否则把 encoder 输出视为整张 stego，再计算 `enc_out-cover`。这层兼容说明仓库曾支持不同输出语义，而论文 v1 描述的主嵌入器是直接输出图像。静态阅读时不能看到 `residual` 字样就反推论文训练始终是显式残差网络。

推理加载时，`load_model(..., part='encoder')` 会把 decoder、discriminator、loss、noise 都替换成 `Identity`；加载 decoder 时则反向替换 encoder 等模块。这减少内存和初始化依赖，也说明当前 checkpoint 仍来自一个联合训练 Lightning state dict。`strict=False` 加载提高版本兼容性，却可能让遗漏 key 悄悄通过，生产启动应记录 `misses/ignores` 而不是只检查进程没有报错。

### 37.2 `unet.py`：论文架构在当前代码中的形状

[`Secret2Image`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/unet.py#L239-L251) 先用 linear layer 把 payload 映射成 $16\times16\times3$，再上采样到目标分辨率。这比论文“把水印插值到图像尺寸”的文字更具体：它不是简单逐 bit repeat，而是学习一个低分辨率空间投影。

[`Unet1`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/unet.py#L254-L308) 将 RGB 与三通道 secret image 拼成 6 通道，经过四级 stride-2 encoder。decoder 每级先上采样，再与 skip feature 拼接；最后一级还重新拼入原始 6 通道输入，让末端输出直接访问 cover 与 payload。post head 是 `3x3 -> 1x1 SiLU -> 1x1 tanh`，与论文“复杂 $E_{post}$ 保留高频”的论述对应。

代码里的 `norm='none'` 在 encoder/decoder 多处出现，对应论文去除 channel normalization。模型还要求 resolution 是 16 的倍数，因为 `Secret2Image` 以 $16^2$ 为基底。这里的结构证据比 README 描述更强，但 checkpoint 下载的 YAML 并不在 Git 仓库里；具体 width、middle blocks 与 variant 配置来自远端文件，离线源码本身不足以重建每个模型。

[`SecretDecoder`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/unet.py#L409-L461) 支持 ResNet18/50/101、ResNeXt、GoogLeNet、DenseNet、EfficientNet、RegNet、ConvNeXt、VGG 和简单 CNN 等多种 arch。论文 Table 4 只报告其中一部分消融；这些分支说明研究代码保留了更多试验面，不代表官方每个组合都有发布 checkpoint 或相同质量保证。

### 37.3 KBNet、定位器与论文后模块

`python/trustmark/KBNet/` vendored 了 KBNet-S/L 及基础组件，`denoise.py` 把它包装成 TrustMark remover。论文明确把 KBNet 作为当前图像恢复骨干，并非 TrustMark 原创。复用成熟 restoration architecture 是务实选择，但 vendored third-party code 的许可、上游修复和数值差异也需要单独治理。

当前 `localize()` 使用 detector 输出一个或多个 normalized boxes，`decode(DETECTFIRST=True)` 再对候选 crop 逐个尝试。这个能力解决“水印只集中在图像某一区域或经过拼贴后位于局部”的后续需求。论文 v1 的 architecture、Fig. 1 和实验都没有 bbox detector，不能据当前 API 声称 2023 模型已经做区域水印定位。

当前区域写入还增加 `concentrate_wm_region` 与 feather paste。长宽比超过 2 时默认取中心方形区域，否则取按比例缩放的中心矩形；边缘最多用 50 像素、约区域尺寸 1% 的带宽混合。这缓解 patch 边界接缝，却引入新的鲁棒性问题：用户裁掉中心、内容主体不在中心或多次拼贴时，信号可能整体丢失。区域策略必须与产品编辑模型一起选择。

### 37.4 论文与当前实现差异清单

| 主题 | arXiv v1 | 固定 commit 当前实现 |
| --- | --- | --- |
| Variant | B、Q | C、Q、B、P |
| Payload 评测 | raw bits，无 ECC | 默认 100-bit packet + BCH |
| 分辨率 | 整图 residual scaling | 长宽比裁切、区域集中、feathering |
| 解码 | extractor | 可选 rotation 与 bbox localization |
| 语言 | 研究实现 | Python、JS decode、Rust subset |
| 来源连接 | 未来 C2PA 应用 | 提供 soft-binding 示例 |
| 训练 | 四阶段、完整损失 | 留有模块与状态，但无完整公开 recipe |

这张差异表的意义不是评价哪一版“更正确”，而是防止把软件当前能力当作论文证据。论文回答方法是否有效，当前代码回答使用者今天能调用什么；二者通过相同核心网络联系，却处于不同时间点和验证范围。

## 38. 模型下载、MD5 与离线部署

模型权重和 YAML 不随 Python wheel 或 Git 仓库打包。固定 commit 在 [`trustmark.py#L28-L67`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/trustmark.py#L28-L67) 定义 Adobe 模型端点与每个文件的 MD5。`check_and_download()` 发现本地文件缺失或 checksum 不符时调用 `urllib.request.urlretrieve()`。

这套设计减小包体积并支持更新托管位置，但给生产带来几个明确责任：

- 首次请求可能阻塞在网络下载，不能把冷启动延迟当成模型推理延迟；
- 运行环境必须允许访问远端，否则初始化返回 `None` 后可能在更深处失败；
- MD5 在这里主要用于传输完整性和文件对应关系，不是强抗碰撞供应链签名；
- 端点迁移、文件替换和代理缓存都应纳入发布管理；
- 离线镜像需要预取精确文件并验证 hash，不能依赖运行时联网；
- 多进程同时首次下载时需防止写同一目标文件的竞态。

更稳妥的服务部署会把模型作为镜像构建输入，记录 artifact digest、软件 commit、model type 与 data schema，在启动时只做只读校验。若业务使用 C2PA，manifest 里记录的算法版本也应能映射回这组模型标识。仅记录 `trustmark 0.9.1` 不足以区分 wheel、Git commit、checkpoint 和配置。

## 39. B、Q、C、P 四种 Variant 的边界

论文只报告 B 与 Q。B 使用 $\alpha_{\max}=20$，更偏 robustness；Q 使用 27.5，更偏 quality。当前源码注释称 Q 是常用默认，典型 PSNR 约 43；B 被保留用于论文结果。

C 是后续 compact variant，使用 ResNet18 decoder，注释给出的典型 PSNR 约 39，面向资源受限部署。P 是后续高感知质量变体，训练时提高 perceptual loss 权重并使用更丰富数据，注释给出典型 PSNR 约 48。P 的 encoder resolution 仍为 256，但 decoder 使用 224，并且始终强制中心方形 crop；当前代码还把 P 的运行时 strength 乘 1.25。

| Variant | 论文 v1 | 当前代码定位 | Decoder/预处理要点 |
| --- | --- | --- | --- |
| B | 是 | robustness/quality balanced | ResNet50 系，decoder 245 |
| Q | 是 | 默认质量型 | ResNet50 系，decoder 245 |
| C | 否，后续 | compact | ResNet18，资源更低 |
| P | 否，后续 | perceptual quality | decoder 224，强制方形中心区域 |

四个 variant 共享 100-bit 接口，却不是互操作编码。一个 Q decoder 不能被当作通用 TrustMark detector。系统必须把 model type 与 payload identifier 一同纳入协议，或按已知候选顺序运行多个 decoder。多 decoder 会提高延迟，也会增加 false positive 的多重检验问题。

论文的 B/Q 曲线也不能用 README 的“typical PSNR”替代。后者来自当前工程模型与示例口径，可能包含 ECC、区域处理和后来权重。本文所有论文结果仍只引用 v1 表格。

## 40. ECC 与有效容量：100 Bits 不等于 100 Bits 业务数据

固定代码的 [`DataLayer`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/python/trustmark/datalayer.py#L14-L100) 把 raw model payload 固定为 100 bits，其中 4 bits 是版本字段：2 位保留，2 位选择 schema。其余部分由业务数据与 BCH parity 共享。

| Schema | 业务 payload | ECC bits | 可纠正 bit flips | Version value |
| --- | ---: | ---: | ---: | ---: |
| BCH_SUPER | 40 | 56 | 8 | 0 |
| BCH_5 | 61 | 35 | 5 | 1 |
| BCH_4 | 68 | 28 | 4 | 2 |
| BCH_3 | 75 | 21 | 3 | 3 |

默认 `BCH_5` 因而只能承载 61 protected bits，而不是任意 100-bit 字符串。文本路径使用 7-bit ASCII，再打包成字节并补零；Unicode 字符会被 `ord(t) & 127` 截断，不能把它当 UTF-8 任意文本通道。更推荐 binary mode 与服务端生成的随机标识。

纠错能力的“5 bit flips”也有代码语义：BCH 针对整个 100-bit packet 的指定数据/校验布局工作，错误若高度突发、版本位损坏或超过能力，decode 会失败。图像噪声造成的 bit error 可能相关，不一定符合独立随机翻转。上线前应在实际渠道上统计 packet success 与 correction distribution，而不是只根据理论 $t$ 值估算。

版本位让 decoder 可以自动选择 schema，是协议演进的好基础，却没有编码 model variant、issuer 或 key version。若未来继续扩充，必须避免在只有 2 个实际 schema bits 的字段里无序增加含义。一个更完整的外部 envelope 可把模型族放入 C2PA 或服务端索引，水印内部保持短随机 ID。

## 41. 跨语言实现：Python、JavaScript 与 Rust 不是同一能力面

官方仓库同时包含三条执行路径。

**Python** 是功能最完整的参考实现：可 encode、decode、remove、可选 localize，支持文本/二进制、ECC、四种 model type、rotation 尝试和区域控制。它依赖 PyTorch、Pillow、OmegaConf 等，并使用 `.ckpt`。

**JavaScript** 示例基于 ONNX Runtime Web，README 明确只做 decode。浏览器若可用会走 WebGPU，也可以使用 WASM。它适合扩展程序或客户端检查，但不包含 encode/remove；Q 与 P 还使用不同输入分辨率和 square-crop 规则。

**Rust** crate 提供所有 variant 的 binary encode/decode 子集与相同 ECC 等级，使用 ONNX 模型。README 明确称其为 Python 功能子集，没有文本 convenience API 与 remover/localizer。它更适合命令行、桌面或服务端低依赖集成，但模型预处理和图像保存参数仍需与 Python 对齐。

| 能力 | Python | JavaScript | Rust |
| --- | --- | --- | --- |
| Encode | 是 | 否 | 是，binary subset |
| Decode | 是 | 是 | 是 |
| Remove | 是 | 否 | 否 |
| Localize | 可选 | 否 | 否 |
| BCH schema | 是 | 是 | 是 |
| Model format | PyTorch ckpt | ONNX | ONNX |

跨语言互操作不能只测试“同一张示例能解码”。还要覆盖 resize kernel、rounding、RGB channel order、alpha、JPEG decode、crop rule、P variant 的 224 输入、BCH bit order和边界 payload。仓库提供代码不等于这些组合都有正式兼容性承诺；版本升级应使用 golden images 做双向回归。

## 42. TrustMark 与 C2PA：Soft Binding 只是查找线索

C2PA manifest 可以携带创作与编辑声明，并由签名保护。但很多平台会删除 metadata。TrustMark 可以在像素中嵌入一个短随机 identifier；若组织保留 identifier 到 manifest 的可信映射，接收方即使拿到无 metadata 的拷贝，也可用 decoder 找回 ID，再查询原 manifest。这就是 durable content credential 中的 soft binding。

当前仓库的 [`c2pa_watermark_example.py`](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/c2pa/c2pa_watermark_example.py) 展示把 identifier 反映到 C2PA soft-binding assertion。这个组合的信任链是：

```text
pixels
  -> TrustMark decoder
  -> opaque identifier
  -> trusted manifest store or embedded assertion
  -> signature/certificate verification
  -> provenance claims
```

水印层只负责第一、二步之间的耐久信号。若数据库被篡改、签名无效、证书撤销或 ID 被复制到另一张图，水印本身无法判断。反过来，C2PA manifest 完整存在时，应优先验证签名和 asset binding，不必把水印结果当更高权威。

仓库还提到 signpost 模式：开放 TrustMark 可以指示图像中共存另一种水印，并通过受控 registry 指明解码算法。此时建议使用更强 BCH_SUPER，把有限 payload 用于算法标识。它是后续工程用法，不属于 2023 v1 的主实验。

## 43. 安全批判：公开 Decoder、Removal 与 Spoofing

不可见水印系统至少需要回答以下安全问题。

**False positive。** 对海量无水印图运行 decoder，总会出现某些 bit 模式。BCH validity 降低随机通过概率，但多个 model variant、多个 rotation、多个 bbox 候选会放大试验次数。应计算系统级假阳性而不是单次 decoder 数字。

**Removal。** TrustMark-RM 已证明专用网络可削弱信号。攻击者还可用压缩、扩散重绘、局部修复和白盒优化。鲁棒性应以攻击成本曲线报告，而非“不可移除”。

**Spoofing。** 开放 encoder 允许任何人写入任意公开 payload。若 payload 本身被当作品牌证明，攻击者可伪造。服务端必须验证 payload 是否由授权发行记录绑定，并核对签名、时间、文件或感知哈希。

**Transfer。** 一个水印残差可能从源图估计后叠加到目标图，尤其当 residual scaling 形成全图低频模板时。内容自适应可增加难度，但论文没有专门 transfer benchmark。

**Oracle abuse。** 若公开 API 返回逐 bit confidence、纠错位置或详细失败原因，攻击者可用它优化移除；若只返回布尔值，又不利于诊断。生产接口应区分内部观测与外部最小响应，并限流、记录异常查询。

**Keylessness。** TrustMark 模型本身没有秘密密钥控制写入。它适合作为开放软绑定通道，而不是依靠密钥不可伪造的传统签名水印。如果业务需要发行者认证，必须把签名与授权系统放在 payload 映射层。

这些限制不否定 TrustMark 的实用性。它们决定了正确系统形态：水印是多层 provenance 中一个可损坏、可攻击的信号源，检测结论要与签名 manifest、数据库状态、内容相似度和策略一起评估。

## 44. 与相关工作的关系

**HiDDeN** 建立了 encoder-noise-decoder 的可微鲁棒水印范式。TrustMark 延续这条主线，但增加更丰富噪声、频谱损失和高分辨率残差部署。

**StegaStamp** 面向物理世界与透视、拍摄扰动，强调从打印或相机图恢复。其固定分辨率和可见残差是 TrustMark 试图改善的比较点；两者威胁模型不完全相同。

**RivaGAN** 原本面向视频，具有原生分辨率处理路径。把它作为静态图 baseline 有参考价值，但时间冗余与视频一致性不是 TrustMark 的研究对象。

**SSL watermarking** 借助自监督特征，往往具有高 PSNR，但论文实验显示在噪声下恢复较弱。它代表“弱信号、好质量”的另一端。

**RoSteALS** 使用预训练 latent autoencoder，并引入训练调度与鲁棒思路。TrustMark 明确借用其从低 $\alpha$ 开始的策略，同时摆脱 frozen autoencoder 带来的约 33.9 dB 质量上限。

**传统 DWT/DCT/SVD** 计算简单、可解释、无需训练，却可能对特定色彩或几何变换脆弱。它们仍适合资源受限、威胁较弱或标准兼容场景，不能因神经方法平均分更高就一概淘汰。

后续生成式水印路线把标识注入生成模型权重、latent 或采样过程，适合“生成时即带标”。TrustMark 是后处理式，能覆盖相机图、历史资产和不同生成器，代价是任何人都可运行 encoder，且图像经历再生成时更易失效。两类技术可以共存，不必争夺同一层。

## 45. 工程落地清单

一个可审计的 TrustMark 接入至少应完成以下决策。

**协议。** 固定 model variant、ECC schema、binary/text mode、payload namespace、issuer、版本和撤销规则。payload 使用随机 opaque ID，不放 PII。

**模型供应链。** 在镜像构建时预取配置与 checkpoint，记录 SHA-256/MD5、仓库 commit、wheel version 和 ONNX/PyTorch format；生产不依赖冷启动下载。

**图像预处理。** 统一 EXIF orientation、RGB/alpha/ICC 处理、长宽比策略和最大尺寸。明确中心区域嵌入是否允许，以及 crop 后能否保持业务可用。

**输出编码。** 固定 PNG/JPEG/WebP 质量、chroma subsampling 与 metadata 保存。编码后立即以最终文件字节重新读取并 decode，而不是只验证内存张量。

**强度选择。** 按图像类别与分发渠道标定 variant/strength。平坦图、文字图和密集纹理图应有独立阈值；低 confidence 时允许拒绝发布或使用其他 provenance 手段。

**性能。** 分离模型冷加载、encode、decode、ECC、文件 I/O 和远端查询。CPU/GPU、浏览器与 Rust 路径分别压测，不能直接套 RTX 3090 的 125/25 ms。

**回退。** decode 失败不应自动声明“没有来源”；应返回 `not detected / corrupt / unsupported version / unavailable model` 等可区分状态。重水印优先从 master 重建，remover 只作次选。

**审计。** 保存发行记录、payload、模型版本、文件 hash、C2PA manifest ID、时间和操作主体。删除与撤销要同时覆盖像素资产、映射服务与缓存。

## 46. 生产监控：从平均准确率转向分层 SLO

基础监控应至少包括：

- 每个 model variant 与 schema 的 decode attempt、detected、ECC-corrected、failed；
- raw bit confidence 分布、BCH correction count 和完整 payload 成功率；
- 按输入尺寸、长宽比、格式、质量、内容类型与渠道分桶；
- encode 后的 PSNR/LPIPS 抽样，以及人工可见性审核；
- JPEG、resize、crop、blur、color 等回放集的逐项恢复率；
- 首次模型下载失败、checksum mismatch、加载时间和显存；
- localize 候选数、rotation retry 次数和多模型尝试次数；
- payload 查询命中、撤销、冲突与跨资产重复；
- remove/re-watermark 次数和 master 是否可用。

可把端到端成功事件定义为：最终发布文件能被指定 decoder 恢复有效 schema，payload 查到未撤销发行记录，manifest 签名有效，并且内容绑定检查通过。这个事件比单纯 bit accuracy 更接近业务目标。

告警也应双向。恢复率突然下降可能是平台改了转码；异常升高的假阳性或同一 payload 出现在大量不相似图像上，可能是 spoofing/transfer；模型下载 hash 变化可能是供应链问题；某类输入 PSNR 下降可能是色彩或 alpha 处理回归。

## 47. 隐私、数据治理与标识生命周期

不可见不等于无隐私风险。水印 payload 能跨平台、跨截图被机器读取，如果直接嵌入用户 ID、邮箱、设备号或订单号，就形成难以察觉的追踪标识。即使当前 payload 只有 40-75 个受保护 bits，也足以编码稳定标识。

更合理的设计是高熵随机 ID，服务端按权限解析。映射表应有最小化字段、访问审计、保留期和删除机制；公开 decoder 可以返回“存在某 schema”，但查询具体来源需要认证。若资产面向公开互联网，应评估这种 durable identifier 是否符合用户告知与地区隐私法规。

撤销不是修改历史图像。系统可以把 ID 标成 revoked，并在查询时返回状态；像素中的 bit 仍可能存在。若用户要求删除，需要处理原文件、派生文件、CDN、备份与索引。TrustMark-RM 只影响像素信号，不会自动删除外部记录。

多租户环境还要防 payload collision 与越权查询。随机空间应足够大，创建时做唯一性检查；映射服务要按 issuer/tenant 分区；同一个短 ID 不应在不同租户中被无上下文解析。C2PA 签名主体、TrustMark issuer 与资产所有者之间的关系需要显式建模。

## 48. 局限性、推荐阅读路径与结论

TrustMark 的局限可以分为六层。

第一，**分辨率**。内部网络仍是 $256^2$，任意分辨率由 residual interpolation 获得。极端长宽比、局部超细内容、超大图和多区域编辑没有充分证据。

第二，**内容分布**。训练以 MIR-Flickr 自然图为主，密集纹理已出现退化。科学图、医学图、文档、UI、HDR 与广色域需要独立评测。

第三，**安全**。I-FGSM 只是有限白盒攻击；removal、spoofing、transfer、替代 decoder 和生成式再编码都没有完整覆盖。开放模型必须配合签名与服务端验证。

第四，**指标**。PSNR/SSIM/bit accuracy 缺少主观用户研究、完整 payload success、false positive 和 calibrated confidence。高平均值不能直接形成 provenance SLA。

第五，**复现**。论文公开方法与推理代码，但训练数据清单、完整一键 recipe、所有 checkpoint 生成过程和随机性证据不足。RM 的两周 A100 成本也提高了独立验证门槛。

第六，**工作流**。re-watermarking 仍累积 remover noise；水印不能代替 master asset、版本控制、C2PA manifest、权限和审计。

推荐阅读顺序是：先看 Abstract、Fig. 1、Algorithm 1 和 Table 1，建立系统主线；再读损失与 Fig. 4/5，理解鲁棒性的代价；随后读 Table 3/4 与 Fig. 3/7，理解 remover 和重水印；最后读补充 Table 5/6、Resolution Scaling 曲线，再进入固定 commit 的 `trustmark.py`、`datalayer.py`、`model.py`、JavaScript/Rust 与 C2PA 示例。

最终判断是：

> TrustMark 最重要的贡献，不是某个 PSNR 数字，而是把固定分辨率神经水印包装成可进入真实资产流的残差接口，并把噪声训练、频谱质量、移除和重写放进同一问题框架。它适合作为内容来源系统的耐久信号层；一旦被当成“图像真实证明”或“不可移除安全标签”，就超出了论文和代码能够支持的边界。

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

本文只进行了论文、补充材料与官方源码的静态阅读，没有下载 checkpoint，没有运行 encode/decode/remove，没有训练 TrustMark/TrustMark-RM，也没有复现论文 benchmark。文中性能数字均属于作者给定数据、硬件和协议，不能直接外推为当前 PyPI 包或任意生产图像流水线的保证。

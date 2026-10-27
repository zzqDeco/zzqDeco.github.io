---
title: "Qwen3-VL-Seg 论文精读：开放世界指代分割、Box-Guided Decoder 与 SA1B-ORS"
description: "精读 Qwen3-VL-Seg 如何将 Qwen3-VL 的开放世界框定位先验转成像素级掩码，并通过轻量解码器、SA1B-ORS 和 ORS-Bench 扩展开放世界指代分割"
pubDate: "2026-07-10T10:26:59+08:00"
updatedDate: "2026-07-10T10:26:59+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "Computer Vision"
  - "Multimodal LLM"
  - "Referring Segmentation"
  - "Visual Grounding"
  - "Qwen"
draft: false
---

Qwen3-VL-Seg 最核心的建模切入点，在于重新定义了多模态大语言模型（MLLM）输出的边界框（bounding box）：在传统视觉定位任务中，预测框往往被视为推理的终点；而在本文的架构设计里，边界框进一步充当了像素级掩码解码的结构先验。基础模型 Qwen3-VL 已经能够依托开放词汇、视觉属性、空间方位和上下文关系准确框定目标，但二维矩形框只能回答“目标大致在哪个区域”；Qwen3-VL-Seg 则通过在模型内部挂载一个轻量级掩码解码器（box-guided mask decoder），将这种高阶语义定位能力平滑传导至连续、边界敏感的二值分割掩码。

这一设计处于当前多模态分割两条主流路线的交汇处。第一条路线采用 `MLLM -> box/point -> SAM` 的双模型串联范式（如 LISA、SAM4MLLM 等），由 MLLM 负责语言理解与提示生成、外部 SAM 负责边界恢复，虽然掩码质量稳健，但在线推理需同时维护两个基础模型，显存开销与链路延迟较重；第二条路线让 MLLM 直接以自回归方式输出多边形顶点或离散掩码 token，虽然接口高度统一，却受制于长序列生成延迟、坐标量化误差与空间分辨率瓶颈。Qwen3-VL-Seg 探索了第三条原生密集预测（MLLM-native dense prediction）路径：完整保留 Qwen3-VL 的对话生成与框定位接口，在主干旁新增一个 `17M` 参数的框引导解码器，联合消费多层 ViT 视觉特征、语言分割 token、预测边界框以及原图浅层纹理，从而在单栈架构内完成端到端指代分割。

阅读本文时，需要对三个核心概念的适用边界建立准确认知：其一，`17M` 指的是新增解码器的架构参数规模（约占 4B 基座模型的 0.4%），但两阶段训练的第二阶段仍需对 4B 语言主干做全参数微调；其二，`SAM-free` 针对的是在线部署与推理链路不再调用外部 SAM，而离线构建 SA1B-CoRS 训练数据时依然借助了 SAM2 生成初始粗掩码；其三，“开放世界（Open-World）”能力主要由百万级长尾训练数据与分布外评测基准（ORS-Bench）支撑，在面对训练分布极度匮乏的高风险专业领域（如医疗诊断与自动驾驶）时，模型仍有明显能力边界。

本文精读对象为 Tongyi Lab（Alibaba Group）发表的预印本 **Qwen3-VL-Seg: Unlocking Open-World Referring Segmentation with Vision-Language Grounding**（arXiv: `2605.07141v1`，提交于 2026-05-08）。截至写作时，官方尚未开放代码仓库、模型权重、SA1B-ORS 数据集及 ORS-Bench 评测集下载链接。本文基于论文正文、TeX 源码与实验图表进行系统剖析。

> **一句话总结**：Qwen3-VL-Seg 将 MLLM 自回归生成的语义边界框转化为掩码解码的强空间先验，配合多尺度特征注入、软边界框门控与两轮掩码感知查询精化，构建了由 `17M` 原生解码器、`2.99M` 样本的 SA1B-ORS 数据集以及 ORS-Bench 组成的开放世界指代分割体系。

## 1. 任务谱系、核心贡献与技术路线定位

### 论文元数据与三大支柱贡献

| 项目 | 内容 |
| --- | --- |
| 题名 | Qwen3-VL-Seg: Unlocking Open-World Referring Segmentation with Vision-Language Grounding |
| 作者 | Yuan Yao, Qiushi Yang, Humen Zhong, Jiangning Wei, Yifang Men, Shuai Bai, Miaomiao Cui, Zhibo Yang |
| 机构 | Tongyi Lab, Alibaba Group |
| arXiv / DOI | [`2605.07141v1`](https://arxiv.org/abs/2605.07141)（2026-05-08，22 pages）/ `10.48550/arXiv.2605.07141` |
| 基础模型与新增模块 | Qwen3-VL-4B + 17M 参数 Box-Guided Mask Decoder |
| 开源状态 | 论文与 TeX 源码公开；官方代码、权重与数据集暂未发布 |

论文的系统贡献由三个相互咬合的部分组成：
1. **模型架构**：在 Qwen3-VL-4B 上扩展 `17M` 参数的框引导掩码解码器，在推理期摆脱对外部 SAM 模型的依赖；
2. **训练数据（SA1B-ORS）**：从 200 万张 SA-1B 原始图像出发，通过自动化多模态数据引擎蒸馏出 `1.05M` 类别导向样本（SA1B-CoRS）与 `1.94M` 描述性实例消歧样本（SA1B-DeRS），合计约 299 万样本；
3. **评测基准（ORS-Bench）**：构建包含 `9,055` 个样本的分布内多格式评测集（ORS-ID-Bench）与涵盖 6 类分布偏移、共约 `1,200` 个样本的分布外压力测试集（ORS-OOD-Bench）。

这三项工作共同解决了传统指代分割的瓶颈：仅有轻量解码器而缺乏大规模长尾监督，模型仍会局限于 RefCOCO 的封闭日常词汇；仅有自动数据引擎而缺乏高分辨率解码支路，MLLM 仍停留于粗粒度画框；仅有训练集而缺乏分层基准，则无法区分简单类别匹配与复杂关系推理的真实增益。

### 指代视觉任务谱系与三条技术路线对比

为了理清后续实验指标的物理意义，首先需要将几类相近的指代与分割任务区分开来：

| 任务名称 | 输入形式 | 输出形式 | 核心挑战 |
| --- | --- | --- | --- |
| **REC**（Referring Expression Comprehension） | 图像 + 指代表达式 | 单个二维边界框 | 语言消歧、粗粒度空间定位（通常 $\text{IoU}>0.5$ 即算成功） |
| **RES**（Referring Expression Segmentation） | 图像 + 指代表达式 | 单个像素级二值掩码 | 语义消歧 + 精细边缘、细长部件与孔洞拓扑恢复 |
| **GRES**（Generalized RES） | 图像 + 表达式 | 零个、一个或多个掩码 | 无目标拒识（no-target）、多目标枚举与计数 |
| **Open-Vocabulary Segmentation** | 图像 + 开放类别名 | 类别对应的全部区域 | 超出封闭词表的长尾类别泛化 |
| **ORS**（Open-World Referring Segmentation） | 图像 + 不受限的类别/属性/关系描述 | 一个或多个实例掩码 | 开放词汇、同类实例消歧、复杂空间关系推理与 OOD 泛化 |

在传统 REC 任务中，边界框只要框住主体即可得分；而到了 RES 与 ORS 任务中，自行车辐条、椅子腿空隙、衣物褶皱边缘或树枝分叉处的任何背景混入都会直接拉低像素级 IoU。同时，MLLM 内部的视觉表征天然面临语义与分辨率的张力：为了压缩上下文长度并对齐语言空间，送入 LLM 的视觉 token 通常经历了 patch 合并与高层抽象，越靠近 ViT 顶层语义越稳定，但高频边缘细节流失也越严重。因此，将边界框 $(x_1,y_1,x_2,y_2)$ 转化为高质量掩码，必须同时解决**找回中间层局部结构**、**将框几何与语言语义绑定到同一查询**、**引入原图浅层纹理并屏蔽框外背景噪声**、以及**利用初始粗掩码反馈修正查询**这四个问题。

横向对比当前多模态分割的五类主流技术路线，可以更清楚地看到 Qwen3-VL-Seg 的生态位：

| 技术路线与代表工作 | 语言开放度 | 边界精细度 | 在线推理依赖 | 核心优势 | 潜在风险与代价 |
| --- | --- | --- | --- | --- | --- |
| **MLLM + 外部分割器**（LISA, GSVA, SegAgent+SAM） | 高 | 高 | MLLM + SAM 双大模型 | 直接复用强分割基座，边缘鲁棒 | 显存与延迟翻倍，定位与分割难联合优化 |
| **自回归多边形 / 离散掩码 Token**（部分原生 MLLM） | 高 | 中 | 单一 MLLM | 架构完全统一，无需额外解码头 | 序列过长、坐标量化阶梯感、细碎孔洞难表达 |
| **MLLM + 轻量原生解码器**（PixelLM, UFO, **Qwen3-VL-Seg**） | 高 | 中至高 | MLLM + 轻量解码头 | 端到端联合训练，单栈紧凑部署 | 需侵入式读取内部多层特征，高度依赖训练数据质量 |
| **分割基础模型**（SAM, SAM2, SAM3） | 中至高 | 极高 | 专用视觉分割大模型 | 类别/概念级多实例覆盖极强 | 缺乏复杂长句语法、否定与多跳空间关系推理能力 |
| **视觉通才模型**（Florence-2, Grounding DINO） | 中 | 任务相关 | 统一视觉编码解码器 | 检测与短语定位精度高、速度快 | 缺乏开放式多轮对话与复杂多模态推理接口 |

Qwen3-VL-Seg 的核心洞察在于：既然现代 MLLM 已经具备很强的框定位（box grounding）能力，就不必让掩码解码器仅凭一个抽象的 `<mask_token>` 在全图盲目搜索，直接把预测框作为空间锚点注入查询构建与像素特征门控，就能以极小的参数增量撑起高质量的原生分割。

## 2. 总体架构与 Box-Guided Mask Decoder 设计

### 系统总体信息流与数学形式化

![Qwen3-VL-Seg architecture](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig01-architecture.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 1.*

如 Fig. 1 所示，系统由 **Qwen3-VL 主干**与 **Box-Guided Mask Decoder** 两大部分构成。给定输入图像 $I$ 与自然语言指令，图像首先经过 ViT 编码器提取视觉特征，与文本指令一同送入 Qwen3-VL 语言模型。语言模型以自回归方式生成结构化的 JSON 响应，其中包含每个目标的二维边界框 `bbox_2d`、类别标签 `label` 以及特殊的掩码占位符 `<mask_token>`。此时，预训练 MLLM 向掩码解码器暴露四组内部表征：

$$
\left\{F_{\mathrm{vis}}^l\right\}_{l=1}^{L},\quad
T_{\mathrm{mm}},\quad
T_{\mathrm{seg}},\quad
B_{\mathrm{box}}.
$$

其中 $\left\{F_{\mathrm{vis}}^l\right\}_{l=1}^{L}$ 为从 ViT 不同深度层级抽取的多尺度视觉特征图，$T_{\mathrm{mm}}$ 为经过多模态对齐的视觉嵌入，$T_{\mathrm{seg}}$ 为对应 `<mask_token>` 位置的语言隐藏状态（承载目标指代语义），$B_{\mathrm{box}}=(x_1,y_1,x_2,y_2)$ 为模型预测的归一化边界框。新增的掩码解码器 $\mathcal D$ 结合原始图像 $I$ 的浅层特征生成最终预测掩码：

$$
\hat M=\mathcal D\!\left(
\{F_{\mathrm{vis}}^l\}_{l=1}^{L},
T_{\mathrm{mm}},
T_{\mathrm{seg}},
B_{\mathrm{box}},
I
\right).
$$

这一接口形式说明，Qwen3-VL-Seg 并非一个挂载在黑盒文本 API 之后的独立后处理模块，它必须在模型前向传播过程中读取多层 ViT 特征图与特定 token 的隐藏状态，并保持二维空间网格坐标的严格对齐。

![Qwen3-VL-Seg mask decoder](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig02-mask-decoder.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 2.*

Fig. 2 详细展示了 `17M` 解码器内部的数据流向，其计算过程可归纳为四条紧密配合的支路：

```text
multi-level ViT features ──> spatial adapters ──┐
                                                ├─> decoder memory F_mem
multimodal visual embeddings ──────────────────┘

seg token T_seg + Fourier(B_box) ──> object query Q^(0) ──> transformer decoder ──> Q^(1)

raw image I ──> shallow CNN ──> soft box gate M(x,y) ──┐
                                                       ├─> pixel features F_pixel
fused ViT features F_fuse ──> PixelShuffle upsample ───┘

Q^(1) + F_pixel ──> mask pass 1 ──> mask-aware pooling F_tar ──> refined query Q^(2) ──> mask pass 2
```

为了控制计算复杂度，解码器并未在所有高分辨率多尺度特征上直接执行昂贵的交叉注意力，而是先在压缩后的多模态记忆 $F_{\mathrm{mem}}$ 上更新目标查询，再将高分辨率浅层特征留到动态卷积核掩码预测阶段融合。下面逐一拆解这四个核心子模块。

### 模块一：多尺度空间特征注入与解码器记忆构建

由于 ViT 顶层特征丢失了局部几何边缘，模型从多个中间 ViT 层抽取特征 $F_{\mathrm{vis}}^l$，先用 $1\times 1$ 卷积统一投影至解码器隐藏维度 $X_0^{(l)}=\operatorname{Conv}_{1\times1}(F_{\mathrm{vis}}^l)$，随后通过深度可分离卷积（DWConv）分支注入局部空间归纳偏置：

$$
\tilde F_l=X_0^{(l)}+
s\cdot\operatorname{GELU}\!\left(
\operatorname{DWConv}\!\left(
\operatorname{GroupNorm}(X_0^{(l)})
\right)\right).
$$

此处可学习标量缩放因子 $s$ 初始化为极小的 $10^{-3}$。这种近零初始化策略保证了训练初期空间适配器近似于恒等映射，避免随机初始化的卷积核破坏预训练 ViT 的表征流形，随着训练推进再逐步引入多尺度边缘特征。接着，各层适配后的特征在通道维拼接（$\oplus$）并融合为 $F_{\mathrm{fuse}}$：

$$
F_{\mathrm{fuse}}=
\phi_{\mathrm{fuse}}\!\left(
\tilde F_1\oplus\cdots\oplus\tilde F_{L-1}\oplus\tilde F_L
\right).
$$

与此同时，多模态视觉嵌入 $T_{\mathrm{mm}}$ 经线性投影后重排回二维特征图 $T'_{\mathrm{mm}}=\operatorname{Reshape}(W_{\mathrm{mm}}T_{\mathrm{mm}})$，与多尺度融合特征 $F_{\mathrm{fuse}}$ 及可学习的二维位置编码 $P_{\mathrm{mem}}$ 相加，构成 Transformer 解码器的键值记忆（Memory）：

$$
F_{\mathrm{mem}}=
T'_{\mathrm{mm}}+F_{\mathrm{fuse}}+P_{\mathrm{mem}}.
$$

这三个加项各司其职：$T'_{\mathrm{mm}}$ 提供与语言指令充分交互过的高层语义，$F_{\mathrm{fuse}}$ 补齐中间层局部空间结构，$P_{\mathrm{mem}}$ 则在特征展平为序列后保留二维坐标拓扑。

### 模块二：框编码融合的空间-语义查询构建

如果仅使用语言侧的 `<mask_token>` 特征 $T_{\mathrm{seg}}$ 作为初始查询，当面对“分割图中的每一只狗”等多实例指令或存在多个相似干扰物的场景时，查询向量缺乏明确的空间指向。Qwen3-VL-Seg 因此将预测框 $B_{\mathrm{box}}=(x_1,y_1,x_2,y_2)$ 显式编码进初始查询中。设框的宽高分别为 $w$ 和 $h$，模型利用傅里叶位置编码函数 $\gamma(\cdot)$ 构造框几何嵌入：

$$
E_{\mathrm{box}}=
\gamma(x_1)\oplus\gamma(y_1)
\oplus\gamma(0.2\log w+0.5)
\oplus\gamma(0.2\log h+0.5).
$$

将宽度 $w$ 和高度 $h$ 先映射到对数域再做傅里叶编码，是为了让相对尺度变化更加平滑——从 10 像素增至 20 像素的相对变化幅度，远大于从 500 像素增至 510 像素。随后，框几何嵌入与分割语义 token 经线性变换相加并归一化，形成同时知晓“目标是谁”和“目标在哪”的初始查询 $Q_{\mathrm{seg}}^{(0)}$，再通过 Transformer 解码器与全局记忆 $F_{\mathrm{mem}}$ 交互得到更新后的查询 $Q_{\mathrm{seg}}^{(1)}$：

$$
Q_{\mathrm{seg}}^{(0)}=
\operatorname{LayerNorm}\!\left(
\operatorname{MLP}_{\mathrm{box}}(E_{\mathrm{box}})
+W_{\mathrm{seg}}T_{\mathrm{seg}}
\right),\qquad
Q_{\mathrm{seg}}^{(1)}=
\operatorname{Decoder}(Q_{\mathrm{seg}}^{(0)},F_{\mathrm{mem}}).
$$

### 模块三：软边界框门控（Soft Box Gate）与高分辨率像素融合

为了恢复发丝、细杆或物体轮廓等高频细节，模型引入一个轻量级卷积主干从原图直接提取高分辨率浅层特征 $F_{\mathrm{cnn}}=\operatorname{Stem}(I)$。然而，原图浅层特征充斥着大量与目标无关的背景边缘和杂乱纹理，若无差别地注入全图，高频背景噪声反而会干扰目标边界判定。反之，若直接按预测框做硬裁剪（hard crop），一旦 MLLM 预测的边界框稍微偏紧切掉了动物尾巴或车轮边缘，解码器就永远无法找回框外丢失的像素。

为此，论文先将预测框在宽度和高度方向各自向外扩展 `15%` 得到扩展框 $(x'_1,y'_1,x'_2,y'_2)$，再构造一个可微分的二维软边界框门控函数 $M(x,y)$：

$$
M(x,y)=
\sigma\!\left(\alpha(x-x'_1)\right)
\sigma\!\left(\alpha(x'_2-x)\right)
\sigma\!\left(\alpha(y-y'_1)\right)
\sigma\!\left(\alpha(y'_2-y)\right),
$$

其中 $\sigma(\cdot)$ 为 Sigmoid 函数，锐度系数固定为 $\alpha=20$。在扩展框内部，四个 Sigmoid 因子均趋近于 1；在框外区域，至少有一个因子迅速衰减至 0；而在框边界附近则形成平滑的可微过渡带（当单条类别指令对应多个预测框时，对多个框的门控图取逐像素空间最大值 $\max$ 以构成并集区域）。随后，多尺度特征 $F_{\mathrm{fuse}}$ 经两级 PixelShuffle 上采样为 $F_{\mathrm{up}}=\operatorname{Upsample}(F_{\mathrm{fuse}})$，与经过软门控过滤的浅层图像特征拼接，生成高分辨率像素表征 $F_{\mathrm{pixel}}$：

$$
F_{\mathrm{pixel}}=
F_{\mathrm{up}}\oplus
\left(M(x,y)\odot F_{\mathrm{cnn}}\right).
$$

这里软门控仅作用于容易引入高频噪声的浅层分支 $F_{\mathrm{cnn}}$，而承载全局语义上下文的主分支 $F_{\mathrm{up}}$ 则保持全图畅通，兼顾了抗噪性与框定位容错率。

### 模块四：两轮掩码感知查询精化与 IoU 置信度头

获得查询 $Q_{\mathrm{seg}}^{(1)}$ 与像素特征 $F_{\mathrm{pixel}}$ 后，动态掩码预测头 $\Psi$ 首先生成第一轮粗掩码对数几率图 $M_{\mathrm{logit}}^{(1)}=\Psi(Q_{\mathrm{seg}}^{(1)},F_{\mathrm{pixel}})$。为了进一步消除遮挡边界或紧邻同类干扰物的歧义，模型将第一轮预测的软掩码 $\sigma(M_{\mathrm{logit}}^{(1)})$ 作为空间注意力权重，从高分辨率像素特征 $F_{\mathrm{pixel}}$ 中池化出目标区域的专属表征 $F_{\mathrm{tar}}$：

$$
F_{\mathrm{tar}}=
\frac{
\sum_{h,w}
\left(
\sigma(M_{\mathrm{logit}}^{(1)})
\odot F_{\mathrm{pixel}}
\right)
}{
\sum_{h,w}\sigma(M_{\mathrm{logit}}^{(1)})+\epsilon
},
$$

其中 $\epsilon=10^{-6}$ 用于保证数值稳定。该目标感知表征经投影层 $\phi_{\mathrm{ref}}$ 以残差形式回流更新查询向量，驱动第二轮动态掩码预测并插值回原图分辨率：

$$
Q_{\mathrm{seg}}^{(2)}=
\operatorname{LayerNorm}\!\left(
Q_{\mathrm{seg}}^{(1)}+
\phi_{\mathrm{ref}}(F_{\mathrm{tar}})
\right),\qquad
M_{\mathrm{logit}}^{(2)}=\Psi(Q_{\mathrm{seg}}^{(2)},F_{\mathrm{pixel}}),\qquad
\hat M=\operatorname{Interp}(M_{\mathrm{logit}}^{(2)}).
$$

这种“查询 $\rightarrow$ 初始掩码 $\rightarrow$ 掩码加权像素池化 $\rightarrow$ 精化查询 $\rightarrow$ 最终掩码”的两阶段闭环，以极低的计算开销实现了查询向量与真实像素证据的双向对齐。此外，精化后的查询还会送入一个线性辅助头 $\hat s_{\mathrm{IoU}}=W_{\mathrm{IoU}}Q_{\mathrm{seg}}^{(2)}$，用于预测整张掩码的期望 IoU 质量得分。

## 3. 结构化输出协议、两阶段训练与评测指标

### ChatML 结构化 JSON 协议与联合损失函数

在交互协议上，Qwen3-VL-Seg 完全复用基座模型的 ChatML 上下文格式。以多实例类别分割为例，系统提示词格式为：

```text
Locate and segment every instance that belongs to the following categories
"dog", report bbox coordinates and masks in JSON format.
```

语言模型的文本输出被严格约束为确定性的 JSON 数组结构：

```json
[
  {
    "bbox_2d": [210, 45, 890, 520],
    "label": "dog",
    "mask": "<mask_start><mask_token><mask_end>"
  }
]
```

其中 `bbox_2d` 为归一化二维坐标，`label` 为目标类别或指代标签，`mask` 字段则输出 `<mask_start><mask_token><mask_end>` 占位符序列。这种将“结构化符号输出”与“密集像素解码”分离的协议，既避免了把成千上万个掩码像素塞进自回归序列，又让每个预测实例拥有一一对应的文本标签、边界框与 `<mask_token>` 隐藏状态，非常便于下游工具解析或在解码失败时降级使用边界框。

训练阶段，模型联合优化自回归文本生成损失 $\mathcal L_{\mathrm{text}}$ 与像素级分割损失 $\mathcal L_{\mathrm{seg}}$：

$$
\mathcal L=
\mathcal L_{\mathrm{text}}+
\lambda_{\mathrm{seg}}\mathcal L_{\mathrm{seg}},
\qquad
\mathcal L_{\mathrm{seg}}=
\lambda_{\mathrm{BCE}}\mathcal L_{\mathrm{BCE}}+
\lambda_{\mathrm{Dice}}\mathcal L_{\mathrm{Dice}}.
$$

其中逐像素二元交叉熵损失 $\mathcal L_{\mathrm{BCE}}$ 与抵抗前景背景面积失衡的 $\mathcal L_{\mathrm{Dice}}$ 分别定义为：

$$
\mathcal L_{\mathrm{BCE}}=
-\frac{1}{HW}\sum_{p=1}^{HW}
\left[
y_p\log \hat y_p+(1-y_p)\log(1-\hat y_p)
\right],\qquad
\mathcal L_{\mathrm{Dice}}=
1-
\frac{2\sum_p \hat y_p y_p+\epsilon}
{\sum_p \hat y_p+\sum_p y_p+\epsilon}.
$$

### 两阶段协同训练（Two-Stage Training Recipe）

作者在实验中发现，如果直接在高强度分割数据上微调视觉编码器与语言模型，虽然密集分割与框定位指标会迅速攀升，但基座模型原有的通用视觉问答（VQA）、OCR 识别与数学逻辑推理能力会出现明显的灾难性遗忘。为此，论文设计了“先专精感知、再协同重平衡”的两阶段训练策略：

| 训练阶段 | 模块更新策略 | 训练步数与学习率调度 | 训练数据配比与设计目的 |
| --- | --- | --- | --- |
| **Stage 1：分割中心适配**<br>（Segmentation-Centric Adaptation） | 冻结 LLM 主干并挂载 `rank=32` 的 LoRA；**放开 ViT 视觉编码器**与 `17M` 掩码解码器 | `10,000` 步；主学习率 $1\times 10^{-4}$（余弦退火），ViT 采用 `0.01x` 差分学习率（$1\times 10^{-6}$） | 混合公开 RES 数据集与自建 SA1B-ORS；用小步长温和调整 ViT 高层特征以适应精细像素边界，快速建立框到掩码的映射能力 |
| **Stage 2：协同能力增强**<br>（Synergistic Enhancement） | 将 Stage 1 的 LoRA 合并回 LLM，**全量微调（Full Fine-tuning）4B LLM 主干**与掩码解码器；**冻结 ViT 编码器** | `5,000` 步；采用极低学习率 $7\times 10^{-7}$（余弦衰减） | 按 `3 : 1 : 2` 混合分割数据、通用多模态理解数据与多模态推理数据（含由原始 Qwen3-VL-Instruct 生成的 STEM 离策略蒸馏样本）；锁定已成型的视觉密集表征，修复 LLM 的通用推理与 OCR 能力 |

这一两阶段设计清晰地体现了多任务微调中的解耦思想：Stage 1 给 ViT 施加 `0.01x` 差分学习率，是因为完全冻结 ViT 会导致严格边界指标 P@0.9 大幅下滑（见后文 Table 5），而步长过大则会彻底冲毁预训练视觉流形；Stage 2 反过来冻结已经适应了密集分割的 ViT，仅以 $7\times 10^{-7}$ 的微调步长全量更新 LLM，从而在不牺牲分割精度的前提下找回通用多模态智能。

### 评测指标体系：mIoU、cIoU、P@t 与匈牙利匹配

在实验评测中，设第 $i$ 个样本的预测掩码为 $P_i$、真值掩码为 $G_i$，单样本交并比为 $\operatorname{IoU}_i=\frac{|P_i\cap G_i|}{|P_i\cup G_i|}$。论文综合采用以下五项指标：
- **平均交并比（mIoU）**：$\operatorname{mIoU}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{IoU}_i$，对所有样本赋予相同权重，能敏感反映小目标的分割成败；
- **累计交并比（cIoU）**：$\operatorname{cIoU}=\frac{\sum_i |P_i\cap G_i|}{\sum_i |P_i\cup G_i|}$，先累加全数据集的交集与并集像素总数再求比值，容易受大面积目标主导，因此需与 mIoU 及小目标分桶配合阅读；
- **阈值精度（$\text{P@}t$）**：$\text{P@}t=\frac{1}{N}\sum_{i=1}^{N}\mathbf 1[\operatorname{IoU}_i>t]$（取 $t\in\{0.5, 0.7, 0.9\}$），其中 **P@0.9** 对轮廓贴合度要求极高，是检验多尺度特征与浅层图像分支是否真正改善精细边界的试金石；
- **定位精度（REC Prec@0.5）**：衡量预测边界框与真值框 IoU 大于 0.5 的比例，用于检验像素级监督是否反向增强了 MLLM 的框定位能力；
- **多实例匈牙利匹配（Hungarian Matching）**：针对一条类别指令对应多个目标的样本，先计算全部预测实例与真值实例之间的成对 IoU 矩阵，通过匈牙利算法求解最优二分图匹配，再在配对成功的实例上统计上述分割指标。

## 4. SA1B-ORS 数据引擎：从碎片掩码到开放世界指代监督

![SA1B-ORS examples](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig03-sa1b-ors-examples.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 3.*

经典的 RefCOCO/+/g 数据集虽拥有高质量人工指代短语，但图像仅源自早期 COCO，类别词表与场景分布相对封闭；而 Meta 发布的 SA-1B 数据集虽包含十亿级高质量自动掩码，却完全没有类别标签与自然语言描述，且同一个完整物体往往被切碎成数十个局部部件掩码（fragments）。为此，作者从 SA-1B 中抽取 200 万张原始图像，搭建了全自动多模态数据引擎，构建出包含 **SA1B-CoRS**（`1.05M` 类别导向样本）与 **SA1B-DeRS**（`1.94M` 描述性实例消歧样本）的 **SA1B-ORS** 数据集（见 Fig. 3）。

### SA1B-CoRS：五步碎片融合与类别级监督构建

![SA1B-CoRS pipeline](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig04-sa1b-cors-pipeline.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 4.*

如 Fig. 4 所示，SA1B-CoRS 的核心任务是将 SA-1B 中无语义的零散碎片重组为语义完整的对象级掩码，共分为五个步骤：

1. **实例语义蒸馏（Instance Distillation）**：先用开放词汇图像标签模型 RAM++ 提取候选词，再执行混合语义过滤——仅保留落在人工整理的高优先级词表中，或能沿着 WordNet 上位词路径（hypernym path）归入 `person`、`vehicle`、`animal`、`artifact`、`clothing`、`food`、`structure` 等实体根节点的标签，剔除场景词、纯颜色词与抽象事件词。随后调用开放词汇定位模型生成候选框，并通过兼顾语义上下位关系的层级化 NMS（Hierarchical NMS，例如在空间高度重叠的 `animal / dog / golden retriever` 中优先保留最具体的细粒度标签），获得实体集合 $\mathcal O=\{(l_i,b_i)\}_{i=1}^{N}$。
2. **粗掩码获取（Coarse Mask Acquisition）**：针对每个候选实体，利用 Qwen3-VL-Plus 预测定位框，并送入 **SAM2** 生成初始粗掩码 $m_p$。
3. **基于 IoF 的精细碎片合并（Fine Mask Merging）**：粗掩码 $m_p$ 具备良好的整体语义完整性，但边缘精细度不及 SA-1B 原生掩码碎片 $\{m_f^k\}$。为了判断一个小的局部碎片（如自行车的单个车轮或椅子的扶手）是否属于目标主体 $m_p$，如果计算标准交并比 IoU，小碎片与整个物体粗掩码的面积比极小，IoU 会非常低。因此，论文改用 **碎片重叠率（Intersection over Fragment, IoF）** 作为合并准则：

$$
\operatorname{IoF}(m_p,m_f^k)=
\frac{|m_p\cap m_f^k|}{|m_f^k|}.
$$

只要小碎片绝大部分面积落在粗掩码内部（$\operatorname{IoF}$ 接近 1），即可判定其为目标的组成部分。将所有高 IoF 碎片求并集后，流水线依次执行粗掩码近邻补齐、孔洞填充、形态学去噪与连通域过滤（若精细合并后面积异常萎缩则安全回退至 SAM2 粗掩码），最后基于框与掩码重叠度去除同类重复实例。
4. **MLLM 视觉验证（MLLM Verification）** 与 **5. 指代指令生成（Caption Generation）**：将原图、蓝色高亮叠加候选掩码的图像以及类别标签组成三元组，交由 Qwen3-VL-Plus 审验“类别是否匹配、主体是否完整覆盖、是否粘连邻近背景、掩码是否严重碎裂、是否漏掉画面中同样显著的同类实例”。仅通过审验的高质量标注被封装为单实例或多实例类别指代训练样本。

### SA1B-DeRS：生成-反向定位闭环的描述性消歧

![SA1B-DeRS pipeline](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig05-sa1b-ders-pipeline.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 5.*

类别名称能解决“把所有船分出来”，但无法解决“分出绿橙相间小船上戴米色帽子的那个人”。SA1B-DeRS 专门面向复杂场景下的单实例精细消歧，其三步流水线（Fig. 5）设计了一个严密的自洽验证闭环：

1. **五维指令生成（Instruction Curation）**：将原图、目标掩码高亮图与真值框输入教师模型 Qwen3-VL-Plus，要求其综合五个维度生成具有唯一指向性的自然语言描述：**类别名称**（是什么）、**目标属性**（颜色、材质、穿着）、**目标状态**（姿态、动作、开合状态）、**相对空间位置**（左右、前后、序数、相对大小）以及**上下文交互关系**（与周围人或物的互动）。
2. **认知反向验证（Cognitive Verification）**：生成的描述是否真的消除了歧义？流水线抹去高亮提示，仅将原始图像与刚刚生成的描述文本重新喂给 Qwen3-VL-Plus，要求其反向做视觉定位输出边界框。**唯有当反向预测框与真实目标框的 $\text{IoU}\ge 0.8$ 时，该描述样本才被采纳**，否则直接丢弃。这一闭环有力确保了训练指令的视觉可定位性。
3. **显著性面积筛选（Saliency Selection）**：过滤掉占整图面积比例过小的极碎微目标，降低伪标签噪声（但这也在一定程度上减少了极小目标的训练曝光，与后文 Area OOD 的表现直接相关）。

![SA1B-ORS statistics](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig06-sa1b-ors-statistics.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 6.*

Fig. 6 的统计分布显示，SA1B-ORS 的样本规模与开放词汇覆盖度远超 RefCOCO 系列与 LVIS，高频词之后的类别呈现平滑的长尾分布；在全部训练样本中，复杂的描述性指令（DeRS）占比达 `64.8%`，词云中密集出现了 `left`、`foreground`、`above` 等空间方位词以及丰富外观属性词，为模型处理复杂语言指代奠定了数据基础。

## 5. ORS-Bench：分布内多指令格式与六类 OOD 压力测试

为了克服传统 RefCOCO 评测集指令形式单一、场景封闭的局限，论文配套构建了 **ORS-Bench**，由分布内评测集 **ORS-ID-Bench**（`9,055` 样本）与分布外评测集 **ORS-OOD-Bench**（约 `1,200` 样本）组成，全部样本均经过人工质检以确认掩码边界与语义指向的准确性。

![ORS-OOD-Bench examples](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig07-ors-ood-bench.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 7.*

两套子基准的构成与考察重点如下表所示：

| 基准子集 | 细分维度 / 桶（Bucket） | 样本规模 | 数据来源与核心考察能力 |
| --- | --- | ---: | --- |
| **ORS-ID-Bench**<br>（共 9,055 样本） | Single-instance category | 2,465 | 源自 COCO、LVIS、SA1B-CoRS；考察单目标开放词汇类别定位与边界分割 |
| | Multiple-instance category | 1,823 | 单一类别词对应图中多个同类实例；考察多目标完备枚举、防粘连与计数能力 |
| | Phrasal | 2,946 | 源自 RefCOCO/+/g 验证集；考察经典短语级指代与空间方位消歧 |
| | Descriptive | 1,821 | 源自自建 DeRS；考察长句多属性、状态与复杂上下文关系推理 |
| **ORS-OOD-Bench**<br>（共约 1,200 样本） | Category OOD | ~200 | 训练集中从未出现或出现频率低于 $10^{-4}$ 的罕见长尾类别 |
| | Area OOD | ~200 | 目标面积占比极小（$\text{ratio}<0.002$）或极大（$\text{ratio}>0.7$）的极端尺度样本 |
| | Instruction OOD | ~200 | 隐去直接类别名、包含否定/排除逻辑、疑问句式、多跳关系或长背景文本埋藏指令 |
| | Occlusion OOD | ~200 | 目标主体遭受严重遮挡、仅局部可见或被前景切割成多个不连通区域的样本 |
| | Lighting OOD | ~200 | 夜间低照度、强逆光、过曝或极端阴影等恶劣光学条件 |
| | Risk-sensitive scene | ~200 | 自动驾驶复杂街景、临床医学影像等训练分布几乎空白的高风险专业领域 |

如 Fig. 7 所示，ORS-OOD-Bench 的设计非常契合真实落地痛点：尤其是 `Instruction OOD` 刻意打破了训练模板，用排除法（“不是左边那只”）、功能隐喻或长段落上下文来指代目标，直接检验模型是真的理解了语言逻辑，还是仅仅记住了表面短语模式。

## 6. 经典指代分割与视觉定位基准评估

![Referring Expression Segmentation results](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-table01-res.webp)

*Source: Yao et al., arXiv:2605.07141v1, Table 1.*

在经典的 RefCOCO、RefCOCO+ 与 RefCOCOg 指代分割基准（Table 1，共 8 个评测切分，指标为 cIoU）上，Qwen3-VL-Seg 取得了极具统治力的表现：

| 模型 / 方法类型 | RefCOCO Val | RefCOCO TestA | RefCOCO TestB | RefCOCO+ Val | RefCOCO+ TestA | RefCOCO+ TestB | RefCOCOg Val | RefCOCOg Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LISA-7B（MLLM + SAM） | 74.9 | 79.1 | 72.3 | 65.1 | 70.8 | 58.1 | 67.9 | 70.6 |
| SegAgent+SAM | 79.2 | 81.4 | 75.7 | 71.5 | 76.7 | 65.4 | 74.8 | 75.6 |
| SAM3（分割基座） | 79.6 | 82.0 | 76.5 | 71.0 | 76.8 | 63.4 | 74.1 | 74.0 |
| Youtu-VL-4B | 80.5 | 82.6 | 77.9 | 74.3 | 78.5 | **71.4** | 76.2 | 76.8 |
| MLLMSeg | 81.2 | 83.1 | 78.4 | 75.0 | 79.3 | 69.8 | 77.5 | **78.4** |
| **Qwen3-VL-Seg (4B)** | **82.3** | **83.7** | **79.1** | **76.2** | **80.2** | 70.8 | **78.2** | 78.1 |

在全部 8 个切分中，Qwen3-VL-Seg 拿下 **6 项第一**，仅在 RefCOCO+ TestB（`70.8` vs Youtu-VL `71.4`）和 RefCOCOg Test（`78.1` vs MLLMSeg `78.4`）上以微弱差距位列次席。相比早期串联外部 SAM 的代表作 LISA，Qwen3-VL-Seg 在 RefCOCO Val 上高出 `7.4` 个百分点，在禁止使用位置词、纯靠外观属性消歧的 RefCOCO+ TestB 上更是大幅领先 `12.7` 个百分点；相比最新的 SegAgent+SAM 与 SAM3，也在全部切分上实现全面超越。

![Referring Expression Comprehension results](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-table02-rec.webp)

*Source: Yao et al., arXiv:2605.07141v1, Table 2.*

更有趣的现象出现在 Table 2 的指代框定位（REC，指标为 Prec@0.5）评测中。将 Qwen3-VL-Seg 与其未经分割微调的基座模型 Qwen3-VL-4B 直接对比：
- **RefCOCO Val / TestB**：从基座的 `90.7% / 86.7%` 提升至 `92.7% / 89.8%`；
- **RefCOCO+ TestB**：从基座的 `75.6%` 跃升至 `82.2%`，净增 **`+6.6`** 个百分点；
- **RefCOCOg Val**：从基座的 `87.3%` 提升至 `89.1%`。

这一结果有力地证明了**密集像素监督对框定位具有显著的反向促进作用（synergistic effect）**：在 Stage 1 联合训练过程中，逐像素掩码损失迫使 ViT 编码器与多模态表征更精细地剥离前景目标与紧邻背景，这种更清晰的物体轮廓感知反过来帮助语言模型输出了更准确的二维边界框，使其在通用对话接口下达到了逼近专用视觉通才模型（如 Florence-2）的定位水准。

## 7. 开放世界与分布外（OOD）泛化实验分析

### ORS-ID-Bench：语言密集型指令的显著优势

![ORS-ID-Bench results](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-table03-ors-id-bench.webp)

*Source: Yao et al., arXiv:2605.07141v1, Table 3.*

| Method | Single Cat.<br>cIoU / P@0.5 | Multiple Cat.<br>cIoU / P@0.5 | Phrasal<br>cIoU / P@0.5 | Descriptive<br>cIoU / P@0.5 |
| --- | :---: | :---: | :---: | :---: |
| Gemini-2.5-pro | 77.7 / 85.1 | 75.9 / 83.8 | 58.7 / 73.1 | 44.6 / 45.4 |
| Gemini-3.0-flash | 67.3 / 71.1 | 62.1 / 58.5 | 51.6 / 61.2 | 57.2 / 70.9 |
| Seed-2.0-pro | 65.4 / 69.9 | 66.4 / 73.0 | 63.8 / 80.7 | 61.0 / 74.7 |
| Youtu-VL-4B | 77.6 / 86.4 | - | 53.2 / 69.0 | 69.3 / 76.0 |
| UFO-8B | 72.4 / 83.7 | - | 52.1 / 67.2 | 73.7 / 70.9 |
| SAM3 | 94.2 / 97.3 | **94.6** / 91.4 | 66.6 / 79.1 | 75.5 / 78.2 |
| **Qwen3-VL-Seg** | **96.0 / 97.4** | 93.2 / **91.6** | **82.8 / 94.3** | **91.0 / 91.3** |

Table 3 最清晰地揭示了不同范式的能力边界。在全部 8 项细分指标中，Qwen3-VL-Seg 拿下 7 项第一：
- **在概念清晰的类别级任务上与 SAM3 平分秋色**：对于 `Single-instance category`，Qwen3-VL-Seg 达到 `96.0 / 97.4`；对于 `Multiple-instance category`，专用分割基座 SAM3 的 cIoU 达到 `94.6%`，略高于 Qwen3-VL-Seg 的 `93.2%`（而 P@0.5 上 Qwen3-VL-Seg 以 `91.6%` 略胜 `91.4%`）。这说明当指令仅仅是一个名词概念、核心挑战在于无遗漏地分割画面中所有同类大大小小的实例时，专用分割基础模型依然具备极强的像素覆盖优势。
- **在短语与复杂描述任务上拉开超过 15 个百分点的代差**：一旦进入需要空间方位与多属性关系消歧的 `Phrasal` 与 `Descriptive` 桶，SAM3 的 cIoU 迅速回落至 `66.6%` 与 `75.5%`，闭源商业大模型（Gemini 系列、Seed-2.0-pro）与同类原生分割模型（UFO-8B、Youtu-VL-4B）也普遍在 `44%`～`75%` 之间徘徊；而 Qwen3-VL-Seg 分别取得了 **`82.8% / 94.3%`** 与 **`91.0% / 91.3%`** 的高分，比最强基线分别高出 `16.2`（较 SAM3）与 `15.5` 个百分点的 cIoU。这正是“MLLM 强语言关系定位 + 框引导像素解码”协同发力的直接体现。

### 定性可视化对比：单实例、多实例与复杂描述

![Single-instance category results](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig08-single-instance-results.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 8.*

![Multiple-instance category results](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig09-multiple-instance-results.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 9.*

![Phrasal and descriptive results](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig10-phrasal-descriptive-results.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 10.*

结合 Fig. 8（单实例类别）、Fig. 9（多实例类别，不同颜色代表不同预测实例，UFO 与 Youtu-VL 因不支持多实例输出未列入）与 Fig. 10（短语及复杂描述）的定性对比，可以从四个维度观察 Qwen3-VL-Seg 的行为特质：
1. **轮廓贴合与细长结构恢复**：通用商业 MLLM（如 Gemini 系列）虽然能找准大致方位，但在大象腿部、毛毯边缘等案例中输出的掩码呈现明显的多边形锯齿或粗块状膨胀；Qwen3-VL-Seg 依托浅层图像特征与软门控，能够紧密贴合复杂轮廓并保留内部镂空拓扑。
2. **多实例分离与防粘连**：在 Fig. 9 的密集人群或成群动物场景中，模型需要先在 JSON 中准确枚举出每一个实例的独立边界框，再由各自框锚定的查询分别解码，有效防止了相邻同类实例粘连成一个连通域。
3. **细粒度部件与序数逻辑消歧**：在 Fig. 10 的“从底部数第二个瓶子”“消防车上抬起的水炮”等高难样例中，对比方法常因无法解析序数关系或局部部件名称而框错相邻物体，Qwen3-VL-Seg 则准确完成了从语言逻辑解析、局部框定位到精细掩码生成的全链路闭环。
4. **中间可解释性优势**：由于模型显式输出 `label` 与 `bbox_2d`，当最终分割出错时，工程师可以直接核查是“语言理解找错了目标（Grounding Error）”还是“框完全正确但边缘分割失败（Decoder Error）”，诊断透明度远高于端到端黑盒掩码输出。

### ORS-OOD-Bench：六类分布偏移下的真实泛化边界

![ORS-OOD cIoU comparison](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-fig11-ood-ciou-comparison.webp)

*Source: Yao et al., arXiv:2605.07141v1, Fig. 11.*

Fig. 11 汇总了模型在 ORS-OOD-Bench 六个分布外测试桶上的 cIoU 表现：

| OOD 测试桶 | Instruction OOD | Occlusion OOD | Lighting OOD | Area OOD | Category OOD | Risk-sensitive Scene |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen3-VL-Seg cIoU (%)** | **86.22** | **83.45** | **78.90** | **59.30** | **53.49** | **8.64** |

这组数字需要分两面解读。一方面，横向对比其他前沿 MLLM 与统一感知模型，Qwen3-VL-Seg 在各个 OOD 维度上均保持领先，尤其是在考察间接指代、否定排除与疑问句式的 `Instruction OOD`（`86.22%`）以及重度遮挡 `Occlusion OOD`（`83.45%`）、恶劣光照 `Lighting OOD`（`78.90%`）上展现出强大的抗干扰能力，证明基座大模型的语言泛化能力成功迁移到了分割任务。

另一方面，纵向审视绝对分数可以发现明确的退化阶梯：当面对极端大小尺度的 `Area OOD`（`59.30%`，部分源于 DeRS 构建阶段主动过滤了极小碎片目标）和罕见长尾类别 `Category OOD`（`53.49%`）时，性能已出现明显回落；而在涵盖临床医学影像与自动驾驶复杂路况的 `Risk-sensitive scene` 中，cIoU 更是骤降至 **`8.64%`**。这是因为专业医疗与车载图像在成像模态、解剖/交通语义规范及尺度分布上与日常互联网照片存在巨大鸿沟，未经领域专门适配的通用视觉先验绝不能直接用于安全攸关场景的决策。

## 8. 通用多模态能力保持与解码器消融实验

### Table 4：两阶段训练对通用多模态能力的损伤与修复

![General multimodal benchmark results](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-table04-general-multimodal.webp)

*Source: Yao et al., arXiv:2605.07141v1, Table 4.*

Table 4 通过追踪基座模型 `Qwen3-VL-4B Instruct`、`Stage 1` 检出模型与最终 `Stage 2` 模型在十余项主流多模态基准上的表现，直观展示了密集感知微调与通用推理能力之间的博弈：

| 评测基准（节选） | 考察能力维度 | Qwen3-VL-4B Instruct (Base) | Stage 1（仅分割适配） | Stage 2（最终 Qwen3-VL-Seg） | 相对 Base 的最终变化 |
| --- | --- | ---: | ---: | ---: | :---: |
| **RefCOCO Val** | 指代定位（Grounding） | 91.6 | 91.8 | **92.3** | +0.7 |
| **MMBench-EN** | 综合多模态感知理解 | 83.9 | **86.2** | 85.4 | +1.5 |
| **RealWorldQA** | 真实世界空间与视觉问答 | 69.8 | 69.5 | **71.2** | +1.4 |
| **InfoVQA Val** | 复杂图表与信息图 OCR 问答 | 80.3 | 75.2 | **81.4** | +1.1 |
| **CharXiv-RQ** | 科学图表推理问答 | 43.1 | 41.0 | **45.2** | +2.1 |
| **MMMU Val** | 大学学科级多模态知识推理 | **67.4** | 63.4 | 66.8 | -0.6 |
| **MathVision / MathVista** | 视觉数学几何推理 | **51.6 / 73.7** | 47.9 / 70.9 | 50.8 / 73.1 | -0.8 / -0.6 |
| **MMStar / MMMU-Pro** | 严苛综合多模态评测 | **69.8 / 53.2** | 67.5 / 49.8 | 67.7 / 51.3 | -2.1 / -1.9 |
| **AI2D / DocVQA Val** | 科学示意图 / 文档理解 | **84.1 / 95.3** | 78.9 / 93.8 | 79.7 / 94.1 | -4.4 / -1.2 |

数据轨迹非常清晰：仅执行 Stage 1 的高强度分割微调后，虽然 RefCOCO 与 MMBench-EN 有所提升，但 MMMU（`-4.0`）、InfoVQA（`-5.1`）、MathVision（`-3.7`）与 AI2D（`-5.2`）均发生了显著退化。引入 Stage 2 的 `3:1:2` 混合重平衡与 STEM 蒸馏数据后，绝大部分受损能力得到了大幅修复，甚至在 InfoVQA（`81.4`）、CharXiv-RQ（`45.2`）和 RealWorldQA（`71.2`）上反超了原始基座模型；不过在 AI2D（`-4.4`）、MMStar（`-2.1`）等任务上仍留有轻微折损。因此，Stage 2 的核心价值在于将专用感知微调对通用智能的干扰控制在可接受的工程范围内。

### Table 5：掩码解码器核心组件消融分析

![Mask decoder ablation](/images/blog/qwen3-vl-seg-open-world-referring-segmentation/qwen3-vl-seg-table05-ablation.webp)

*Source: Yao et al., arXiv:2605.07141v1, Table 5.*

Table 5 在 RefCOCO、RefCOCO+ 与 RefCOCOg 验证集上对解码器设计与训练策略进行了逐项消融（报告 mIoU、cIoU 与严苛的边界指标 P@0.9）：

1. **对比串联外部 SAM（`Qwen box + SAM`）**：将同一个 Qwen3-VL 预测的边界框直接喂给外部 SAM 模型，在 RefCOCO Val 上仅取得 `74.3 / 70.3` 的 mIoU/cIoU 与 `39.6%` 的 P@0.9；而端到端联合训练的 Qwen3-VL-Seg 达到 **`82.8 / 82.3`** 与 **`50.2%`** 的 P@0.9（提升超过 `10` 个百分点）。这说明当框仅作为独立后处理提示时，SAM 缺乏语言语义（如“只分割人手里的包，不包括背带”）的细粒度引导，而原生解码器融合了多层语义特征与框门控，能更精准地处理局部指代。
2. **冻结视觉编码器的代价（`freeze vit`）**：若在 Stage 1 完全冻结 ViT 主干、仅训练解码器与 LLM LoRA，RefCOCO Val 上的严格边界指标 P@0.9 会从 `50.2%` 暴跌 `6.1` 个百分点至 `44.1%`（RefCOCO+ 上也从 `47.6%` 跌至 `42.5%`）。这证实了原生 MLLM 的顶层视觉特征偏向语义压缩，必须通过小学习率（`0.01x`）微调 ViT 才能唤醒像素级边界感知。
3. **多尺度 ViT 特征与原图浅层分支的互补贡献（`w/o multivit` & `w/o image`）**：移除多层 ViT 空间适配器或移除受软框门控约束的原图浅层 CNN 分支，均会导致三个数据集上的 cIoU 与 P@0.9 一致下滑（例如在 RefCOCOg 上，完整模型 P@0.9 为 `47.3%`，去掉多尺度特征降至 `47.1%`，去掉浅层图像分支降至 `46.1%`），验证了“中层恢复结构、浅层补充边缘”的设计初衷。

## 9. 复现路线、生产系统落地与局限性审视

### 论文已披露接口与未公开实现细节的边界

由于官方暂未开源代码与权重，若团队计划基于公开的 Qwen3-VL-4B 复现该系统，需首先厘清论文已明确给出的规范与尚需自行探索的超参数：
- **论文已明确定义的接口**：基座采用 Qwen3-VL-4B；解码器四个子模块的完整数学公式；预测框向外扩展 `15%`、软门控锐度 $\alpha=20$、掩码加权池化稳定项 $\epsilon=10^{-6}$；Stage 1（LoRA `r=32`、ViT 学习率 `0.01x`、10,000 步、$\text{lr}=10^{-4}$）与 Stage 2（全量微调 LLM、冻结 ViT、5,000 步、$\text{lr}=7\times 10^{-7}$、数据配比 `3:1:2`）的训练配方；以及 SA1B-CoRS/DeRS 的五步/三步构建逻辑与反向定位 $\text{IoU}\ge 0.8$ 阈值。
- **论文未披露、需工程补齐的细节**：具体抽取 ViT 的哪几个层索引（layer indices）及解码器内部隐藏通道数；PixelShuffle 的具体上采样倍率与动态掩码卷积核尺寸；总损失中 $\lambda_{\mathrm{seg}}$、$\lambda_{\mathrm{BCE}}$、$\lambda_{\mathrm{Dice}}$ 及辅助 IoU 损失的精确权重比例；训练批大小（batch size）、图像输入分辨率与总 GPU 算力消耗；以及 SA1B-ORS 各清洗阶段的样本留存率与推理阶段的延迟/显存实测数据。

### 渐进式工程复现与生产服务落地建议

在工程复现与生产部署时，切忌直接从百万级自动数据与全量微调盲目起步，建议遵循以下渐进式验证与服务治理路线：

1. **先测真值框解码上界（GT-Box Oracle），再测端到端闭环**：在复现初期，先固定提取 Qwen3-VL-4B 的中间层特征，直接将 RefCOCO 的**真实边界框**（Ground-Truth Box）喂给 `17M` 解码器训练。通过在测试时并行维护 **GT-box mask**（检验解码器本身的特征恢复上限）、**Predicted-box mask**（真实端到端表现）与 **Oracle-box selection**（多候选框下的排序上限）三条评测线，并对真值框人为施加平移、缩放与切边抖动（Box Noise Stress Test），可以精准定位性能瓶颈究竟出在特征接口还是语言定位。
2. **建立结构化全链路误差归因（Error Taxonomy）**：利用模型显式输出 JSON 与两轮掩码的优势，在生产日志与回归测试中将失败案例自动归入 7 类互斥桶：**`F0` 协议解析失败**（JSON 语法错误或框与 `<mask_token>` 数量失配）、**`F1` 语言理解失败**（否定或序数关系解析错误）、**`F2` 框定位失败**（Box IoU 过低或漏检多实例）、**`F3` 解码器边界失败**（框准但掩码粗糙）、**`F4` 二轮精化退化**（Pass 2 的 IoU 反而低于 Pass 1）、**`F5` 置信度失准**（IoU Head 高分误报）与 **`F6` 领域偏移失败**。
3. **生产级防护、置信度校准与混合路由架构**：在上线服务时，需对 JSON 解析异常设置重试或降级返回边界框策略，并对多实例指令设置单次最大解码实例数上限以防显存溢出；使用独立验证集对辅助 IoU Head 计算期望校准误差（ECE）并设定拒识门限。在成熟的生产系统中，最合理的定位是将 Qwen3-VL-Seg 作为**开放语言路由器与默认原生分割器**——常规场景直接返回其低延迟原生掩码；当 IoU 置信度偏低或遇到极精细发丝抠图需求时，将其预测框与粗掩码作为提示转发给外部专业分割器（如 SAM3 或垂直领域医疗分割模型）；而在交互式标注工具与图像编辑软件中，则同时向用户暴露边界框与掩码以支持低成本人工微调。

### 核心局限性审视

综合全文分析，Qwen3-VL-Seg 仍存在五点值得后续研究改进的局限：
- **边界框误差的二次放大风险（Box Error Propagation）**：预测框同时参与了初始查询构建 $E_{\mathrm{box}}$ 与高分辨率软门控 $M(x,y)$。一旦 MLLM 预测框发生较大偏移或严重漏掉细长部件（超出 `15%` 扩框范围），查询的空间锚点与浅层边缘特征会同时受损，且第一轮的错误掩码还可能通过掩码感知池化 $F_{\mathrm{tar}}$ 在第二轮自我强化。论文缺少系统的框抖动敏感度曲线与两轮精化消融。
- **教师模型自闭环偏差与小目标过滤张力**：SA1B-CoRS 与 SA1B-DeRS 高度依赖同家族的 Qwen3-VL-Plus 进行生成与反向定位验证，容易形成偏向本家族语言风格与视觉偏好的教师蒸馏分布；同时 DeRS 为保质量剔除了极小面积目标，直接制约了模型在 `Area OOD`（`59.30%`）极小物体上的泛化表现。
- **实验消融颗粒度有待细化**：Table 5 仅消融了 ViT 微调、多尺度特征与浅层图像分支，并未单独剥离傅里叶框查询编码、软门控（对比硬裁剪或无门控）、两轮查询精化（对比单轮解码）以及辅助 IoU 头的独立增益。
- **缺乏在线服务（Serving）效率实测**：虽然 `17M` 解码器省去了外部 SAM 的权重显存，但保存多层 ViT 特征图、执行 PixelShuffle 上采样以及运行高分辨率动态卷积核仍会产生不小的激活显存（activation memory）与计算开销，论文未报告端到端延迟、吞吐量与峰值显存对比。
- **数据、代码与基准尚未开放审计**：SA1B-ORS 与 ORS-Bench 的图像去重细节、闭源 API 评测提示词规范以及模型权重暂未公开，目前尚无法由外部社区独立复查实验数字。

## 10. 总结与参考资料

Qwen3-VL-Seg 的核心启示在于：当多模态大语言模型已经能够用开放语言稳健地输出二维边界框时，这个边界框不应止步于粗粒度定位，而应作为连接高层语言推理与底层像素预测的结构桥梁。通过将傅里叶框编码注入目标查询、用可微软框门控过滤原图浅层纹理，并配合多尺度 ViT 适配器与两轮掩码感知精化，仅 `17M` 参数的原生解码器便在单栈架构内实现了优异的分割精度；而配套构建的 299 万样本 SA1B-ORS 数据引擎与两阶段协同微调配方，则系统展示了如何在大幅拓展开放世界复杂指代分割能力的同时，稳住多模态大模型的通用理解与推理基座。

### 参考资料

- **目标论文与基座模型**：
  1. Yuan Yao et al. [Qwen3-VL-Seg: Unlocking Open-World Referring Segmentation with Vision-Language Grounding](https://arxiv.org/abs/2605.07141), arXiv:2605.07141v1, 2026.（配套 [PDF](https://arxiv.org/pdf/2605.07141)、[HTML](https://arxiv.org/html/2605.07141)、[TeX Source](https://arxiv.org/e-print/2605.07141)）
  2. Qwen Team. [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631), arXiv 2025.
- **指代分割、视觉定位与多模态分割模型**：
  3. Ronghang Hu et al. [Segmentation from Natural Language Expressions](https://arxiv.org/abs/1603.06180), ECCV 2016.
  4. Licheng Yu et al. [Modeling Context in Referring Expressions](https://arxiv.org/abs/1608.00272), ECCV 2016.
  5. Chenxi Liu et al. [GRES: Generalized Referring Expression Segmentation](https://arxiv.org/abs/2306.00968), CVPR 2023.
  6. Xin Lai et al. [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/abs/2308.00692), CVPR 2024.
  7. Zhuofan Xia et al. [GSVA: Generalized Segmentation via Multimodal Large Language Models](https://arxiv.org/abs/2312.10103), CVPR 2024.
  8. Chaoyou Fu et al. [Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic](https://arxiv.org/abs/2306.15195), arXiv 2023.
  9. Haotian You et al. [Ferret: Refer and Ground Anything Anywhere at Any Granularity](https://arxiv.org/abs/2310.07704), ICLR 2024.
  10. Shilong Liu et al. [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), ECCV 2024.
  11. Bin Xiao et al. [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242), CVPR 2024.
- **分割基础模型、数据集与训练方法**：
  12. Alexander Kirillov et al. [Segment Anything](https://arxiv.org/abs/2304.02643), ICCV 2023.
  13. Nikhila Ravi et al. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), arXiv 2024.
  14. Agrim Gupta et al. [LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/abs/1908.03195), CVPR 2019.
  15. Tsung-Yi Lin et al. [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312), ECCV 2014.
  16. Kevin Lin et al. [Recognize Anything: A Strong Image Tagging Model](https://arxiv.org/abs/2306.03514), CVPR 2024.
  17. Edward J. Hu et al. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
  18. Fausto Milletari et al. [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797), 3DV 2016.
  19. Hamid Rezatofighi et al. [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630), CVPR 2019.

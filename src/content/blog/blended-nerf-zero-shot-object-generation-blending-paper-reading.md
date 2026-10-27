---
title: "Blended-NeRF 论文精读：现有 NeRF 场景中的零样本对象生成、ROI 编辑与体渲染融合"
description: "精读 Blended-NeRF 如何通过 3D ROI、CLIP 引导、双 NeRF 场融合、距离平滑和官方代码实现现有神经辐射场中的对象插入、替换与材质编辑"
pubDate: "2026-06-30T11:58:46+08:00"
updatedDate: "2026-06-30T11:58:46+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "NeRF"
  - "3D Scene Editing"
  - "Text-to-3D"
  - "Computer Vision"
  - "Graphics"
  - "Code Reading"
draft: false
---

Ori Gordon、Omri Avrahami 和 Dani Lischinski 发表的 **Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields** 聚焦于一个极具代表性的 3D 场景编辑问题：

> 当手中已经有一个训练好的 NeRF 场景时，能否仅在用户指定的 3D 局部区域内生成、替换、融合或风格化某个对象，同时严格保持区域外的原场景几何与外观不受干扰？

这一任务设定与从零生成全新 3D 对象的 text-to-3D 存在本质区别。Blended-NeRF 面向的是 **existing Neural Radiance Field（已有神经辐射场）**。用户无需用长文本重新描述整个三维世界，只需提供一个文本提示词（或参考图像块）并在原场景中框定一个 3D ROI（Region of Interest）包围盒，模型便会在该包围盒内执行零样本语义编辑，并将生成内容无缝融合回原始辐射场中。

这篇论文的核心价值在于将多项关键工程机制整合为一套自洽的隐式场局部编辑流水线：利用原始 NeRF 权重初始化生成分支以继承场景先验，通过 3D ROI 包围盒将优化梯度严格约束在局部空间，借助 CLIP 或 BLIP 视觉语言模型提供零样本语义引导，在光线采样点（ray sample）层级同时查询双分支辐射场并执行带距离平滑的体渲染融合（volumetric blending），最后配合背景增强、相机位姿采样、透射率损失（transmittance loss）和深度方差损失（depth loss）抑制几何退化。

本文精读的版本基线为 [arXiv:2306.12760v2](https://arxiv.org/abs/2306.12760v2)（初版提交于 2023-06-22，v2 修订于 2023-09-07；arXiv DOI 为 `10.48550/arXiv.2306.12760`，关联 ICCV 2023 Workshop 论文 DOI 为 `10.1109/ICCVW60793.2023.00316`）。论文 PDF 共 16 页、14 张图，采用 **CC BY-NC-ND 4.0** 许可；文中所有图表均从原论文 PDF 等比例裁切并保留原始版权声明。官方项目页为 [vision.huji.ac.il/blended-nerf](https://www.vision.huji.ac.il/blended-nerf/)，官方开源仓库为 [orig333/Blended-NeRF](https://github.com/orig333/Blended-NeRF)（MIT License）。下文将结合论文推导与开源源码进行深度拆解。

## 1. 论文信息、任务定义与双分支架构总览

### 1.1 论文元数据与核心定位

| 项目 | 内容 |
| --- | --- |
| 题名 | Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields |
| 作者 | Ori Gordon, Omri Avrahami, Dani Lischinski（The Hebrew University of Jerusalem） |
| 版本基线 | arXiv:2306.12760v2（2023-09-07），ICCVW 2023（DOI: `10.1109/ICCVW60793.2023.00316`） |
| 许可证 | 论文 CC BY-NC-ND 4.0；官方代码 MIT License |
| 官方资源 | 项目页 [vision.huji.ac.il/blended-nerf](https://www.vision.huji.ac.il/blended-nerf/) / 代码 [orig333/Blended-NeRF](https://github.com/orig333/Blended-NeRF) |
| 核心任务 | 已有 NeRF 场景中的文本/图像引导 3D ROI 局部对象生成、替换、融合与材质转换 |

一句话概括论文的核心方法：

> Blended-NeRF 将隐式 3D 局部编辑建模为“克隆一份原始 NeRF 作为可训练生成分支、仅在 3D ROI 内部利用视觉语言先验优化该分支、最后沿每条光线在采样点层级将原始分支与生成分支进行体渲染融合”的双分支协同流程。

这一设计由三个支点共同支撑：首先是**已有场景先验（existing NeRF）**，编辑建立在已收敛的场景辐射场之上，贴近真实内容修图工作流（例如在已有花坛中加一朵花、把松果替换成菠萝，或将花瓶转换为玻璃材质）；其次是**显式空间边界（3D ROI）**，利用三维包围盒（代码中存储为 `box_points/*.pt`）直接界定三维编辑范围，避免了将 2D 掩码反投影回 3D 时的多视角歧义；最后是**体渲染级融合（volumetric blending）**，将新旧内容的混合下沉到光线采样点的密度与颜色积分层，从而保证多视角遮挡关系与边界过渡的自然连贯。

### 1.2 从 Fig. 1 与 Fig. 2 看训练与融合闭环

Fig. 1 给出了 Blended-NeRF 的系统总览，左侧为局部优化训练阶段，右侧为体渲染推理融合阶段。

![Fig. 1: Blended-NeRF overview](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig01-overview.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 1, CC BY-NC-ND 4.0.*

系统中并行维护两个结构相同的神经辐射场：一个是冻结参数的**原始分支 $F^O_\theta$（original NeRF）**，负责锁住输入场景的既有几何与外观；另一个是**生成分支 $F^G_\theta$（generator NeRF）**，直接从 $F^O_\theta$ 的权重初始化，并在训练期间根据文本或图像提示词在 ROI 内部接受梯度更新。

在训练阶段，系统仅对穿过 3D ROI 包围盒的光线采样点计算体渲染（ROI-only rendering），将包围盒外的采样点贡献置零，以此驱动 CLIP/BLIP 语义损失专注于局部编辑目标。而在推理渲染阶段，系统沿同一批全局光线同时查询 $F^O_\theta$ 与 $F^G_\theta$：包围盒外部完全采用原始分支输出；包围盒内部根据任务意图执行替换或密度混合；包围盒边界区域则通过距离平滑算子（distance smoothing）消除拼接断层。

Fig. 2 用 Blender 合成船只场景（ship scene）直观展示了大范围对象替换的效果。

![Fig. 2: Large object replacement](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig02-large-object-replacement.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 2, CC BY-NC-ND 4.0.*

在该案例中，ROI 包围盒覆盖了船体底部与周围海面区域，模型在文本提示引导下将原有的沙地与水面替换为冰雪或全新地貌。这说明 Blended-NeRF 不仅能往空白空间插入小物件，也能对大面积既有区域执行语义重构。当然，这也揭示了 ROI 尺寸设定的敏感性：若包围盒过小，新生成的几何会被生硬截断；若包围盒过大，生成分支会侵占过多原始背景，降低非编辑区的保真度。

## 2. 现有 NeRF 局部编辑的难点与相关工作坐标

### 2.1 连续辐射场编辑的四重阻碍

标准 NeRF 将三维场景表示为连续的隐式映射函数：

$$
F_\theta(\mathbf{x}, \mathbf{d}) = (\sigma, \mathbf{c}),
$$

其中 $\mathbf{x}$ 为三维空间坐标，$\mathbf{d}$ 为观察方向，$\sigma$ 为体密度（volume density），$\mathbf{c}$ 为视角相关的 RGB 颜色。沿相机光线 $\mathbf{r}$ 采样 $N$ 个离散点后，像素颜色通过体渲染公式加权累积：

$$
\mathbf{C}(\mathbf{r}) =
\sum_{i=1}^{N}
T_i \left(1 - \exp(-\sigma_i \delta_i)\right)\mathbf{c}_i,
\qquad
T_i = \exp\left(-\sum_{j<i}\sigma_j\delta_j\right).
$$

这种全场景耦合的连续场表示虽然极其擅长新视角合成，但在面向局部语义编辑时却面临四重结构性阻碍：其一，**缺乏显式对象边界**，松果、地面、阴影、反光与背景全部纠缠在同一个 MLP 权重中，网络内部并不存在类似 `object_id=pinecone` 的天然解耦句柄；其二，**局部改动与多视角渲染强耦合**，修改任意三维采样点的密度都会影响所有穿过该点的相机视线，若仅在单一视角下优化很容易导致换视角后几何崩塌；其三，**视觉语言先验的监督粒度过粗**，CLIP 只能评估渲染图像块与文本在全局特征空间中的余弦相似度，无法直接指明三维空间中哪些采样点应当成为物体表面、哪些应当保持透明；其四，**严苛的背景保真要求**，在已有场景上动刀必须确保 ROI 之外的建筑、叶片、桌面纹理与全局光照纹丝不动。

Blended-NeRF 的架构设计正是对上述四重阻碍的逐一回应：用 3D ROI 包围盒提供近似的局部空间边界，用原场景权重初始化赋予生成分支几何与色彩统计先验，用 ROI 局部渲染隔绝背景梯度干扰，再用采样点层级的体渲染融合保障多视角连续性。

### 2.2 相关工作演进与 CompoNeRF 横向对照

在 3D 生成与神经场景编辑的技术谱系中，Blended-NeRF 处于隐式场表示（NeRF）、视觉语言零样本引导（Dream Fields / CLIP-NeRF）以及神经场景局部操纵（EditNeRF / ObjectNeRF / Volumetric Disentanglement）的交叉点。与 DreamFusion、SJC 等采用 2D 扩散模型 SDS 损失从零生成 3D 对象的路线相比，Blended-NeRF 选择了 CLIP/BLIP 相似度损失配合几何正则项，将重点放在已有辐射场的局部修补上。

将其与本站精读过的多物体组合生成工作 **CompoNeRF** 放在一起对比，可以清晰看清两类 NeRF 场景任务的边界差异：

| 对比维度 | Blended-NeRF | CompoNeRF |
| --- | --- | --- |
| **输入前提** | 已训练好的 existing NeRF + 单个/局部 ROI box + 文本/图像提示 | 全局多物体文本 + 子文本列表 + 多对象 3D box 布局 |
| **核心任务** | 在已有场景的局部区域内插入、替换、融合对象或转换材质 | 从文本与布局合成全新的多物体可重组 NeRF 场景 |
| **网络表示** | 冻结的原始分支 $F^O_\theta$ + 可训练的生成分支 $F^G_\theta$ | $m$ 个独立的局部对象 NeRF + 全局残差校准 MLP |
| **语义先验** | CLIP / BLIP 图文相似度损失 + 透射率与深度方差先验 | Stable Diffusion v1-4 的局部/全局双层 SDS 损失 |
| **融合机制** | 光线采样点层级的距离平滑与激活函数内/外密度合成 | 全局光线与多包围盒求交后的深度重排与密度残差校准 |
| **代码开源状态** | 官方完整开源（PyTorch，含预置 box 与 notebooks） | 官方仓库仅保留占位说明 |

简言之，**Blended-NeRF 解决的是“如何在已有 3D 神经场景上做局部非破坏性修图”，而 CompoNeRF 解决的是“如何用多个对象级 NeRF 像搭积木一样拼出多物体场景”。**

## 3. 局部 ROI 约束渲染与零样本语义优化

### 3.1 位置编码与原场景权重初始化

Blended-NeRF 沿用了标准 NeRF 的高频位置编码（positional encoding），将三维坐标与方向向量映射至高维傅里叶特征空间：

$$
\gamma(x) =
\left[
\cos(2^l x), \sin(2^l x)
\right]_{l=0}^{L-1}.
$$

在官方源码 `BlendedNeRF/run_BlendedNeRF.py` 的 `create_nerf(args)` 函数中，系统同时实例化结构完全一致的 `fixed_model`（即原始分支 $F^O_\theta$）与 `model`（即生成分支 $F^G_\theta$），并统一加载预先训练好的原场景权重（存放在 `base_weights/` 目录下）。

为什么生成分支必须从原始场景权重初始化，而不能从随机初始化的 MLP 开始训练？因为 CLIP 损失本身缺乏显式的三维空间定位能力——如果从随机噪声场起步，优化器不仅要摸索目标物体的外观，还要在毫无约束的情况下重建相机坐标系、空间尺度与基础辐射分布，极易陷入退化局部最优。继承原场景权重相当于把场景的几何骨架与光照基调直接注入生成分支，使后续梯度更新只需集中精力学习“目标对象与原局部内容之间的残差变化”。论文核心概念与官方源码实现的对应关系如下表所示：

| 论文核心概念 | 官方源码位置 | 工程实现职责 |
| --- | --- | --- |
| 原始分支 $F^O_\theta$ | `fixed_model`, `fixed_model_fine` | 保持参数冻结，提供不可编辑区及融合参考的原始 density/color |
| 生成分支 $F^G_\theta$ | `model`, `model_fine` | 从 base checkpoint 初始化，在 ROI 内接受 CLIP/BLIP 与几何正则更新 |
| 3D ROI 包围盒 | `box_points_path`, `render_kwargs_train['box_points']` | 划定三维轴对齐编辑边界，筛选参与局部训练的光线采样点 |
| ROI 局部渲染 | `render_in_box_fn` | 仅积分包围盒内的采样点，输出局部 RGB、透射率与视差图供损失计算 |
| 体渲染融合 | `raw2outputsBlend`, `render_full_frame_fn` | 在采样点 raw 输出层级组合双分支密度与颜色，渲染完整场景 |

### 3.2 3D ROI 交互控制与局部光线积分（ROI-only Rendering）

为了防止语义损失篡改背景，训练阶段仅对穿过 3D ROI 包围盒 $B$ 的光线片段执行积分。论文将 ROI-only rendering 定义为：

$$
C(r)=
\begin{cases}
\sum_{\mathbf{x}_i \in B} T_i(1-\exp(-\sigma_i \delta_i))\mathbf{c}_i,
& \exists \mathbf{x}_i \in r \text{ such that } \mathbf{x}_i \in B,\\
0, & \text{otherwise}.
\end{cases}
$$

这一局部积分策略起到双重隔离作用：一方面，它斩断了背景区域对 CLIP 编码器的投机干扰（例如当提示词为 “a strawberry” 时，若渲染整张图，周围绿叶与花盆可能让 CLIP 误判语义已部分满足，从而削弱目标区域的生成驱动力）；另一方面，包围盒外的空间根本不产生梯度，天然守护了原场景的非编辑区。

在 `render_rays()` 的代码实现中，系统通过三维轴对齐边界检查（axis-aligned range check）高效筛选落在包围盒内的采样点：

```python
pt_is_in_box = (
    (xmin <= pts[..., 0]) & (pts[..., 0] <= xmax) &
    (ymin <= pts[..., 1]) & (pts[..., 1] <= ymax) &
    (zmin <= pts[..., 2]) & (pts[..., 2] <= zmax)
)
```

随后 `render_in_box_fn()` 仅对满足 `pt_is_in_box` 的采样点查询网络，并输出 `rgb_map_in_box`、`transmittance_in_box` 和 `depth_map_in_box`。在交互层面，官方仓库提供了三个交互式 Jupyter Notebook（`find_3d_box_blender.ipynb`、`find_3d_box_llff.ipynb`、`find_3d_box_llff_360.ipynb`），方便用户从多角度观察原场景并将选定的包围盒顶点保存为 `.pt` 文件。

### 3.3 CLIP/BLIP 语义损失与几何正则先验（Transmittance & Depth Loss）

生成分支的语义方向由视觉语言编码器提供。给定局部渲染图像 $I_{ROI}$ 与目标文本提示 $y$，基础余弦相似度损失写作：

$$
\mathcal{L}_{sim} =
-E_{img}(I_{ROI})^T E_{txt}(y),
$$

其中 $E_{img}$ 与 $E_{txt}$ 分别为图像与文本编码器（源码 `optimization/losses.py` 同时支持 `CLIPLoss` 默认 `ViT-B/32` 以及 `BLIPLoss`）。

然而，单纯最小化 $\mathcal{L}_{sim}$ 会遭遇严重的“对抗样本式几何退化”：网络往往投机生成仅在单一视角下神似目标文本的二维薄片（billboard effect）、半透明雾状噪点，或者利用背景填充色骗取高相似度分数。为此，Blended-NeRF 在 Dream Fields 与 CLIP-NeRF 的基础上组合了六项几何先验与训练增强机制：

| 先验 / 增强机制 | 核心作用 | 工程调参边界 |
| --- | --- | --- |
| **自适应相机距离与位姿采样（Pose Sampling）** | 围绕 ROI 中心多角度采样相机，打破单视角过拟合 | 受限于场景原始有效视角范围（如 LLFF 前向场景） |
| **随机背景增强（Background Augmentation）** | 利用透射率掩码在 ROI 后方合成随机背景图，杜绝靠背景色骗分 | 背景纹理过强可能干扰半透明边缘的色彩收敛 |
| **视角方向提示词（Directional Prompts）** | 按相机方位动态追加 front/side/top view 描述 | 对无明显正背面区分的物体增益有限 |
| **透射率损失（Transmittance Loss, $\mathcal{L}_T$）** | 惩罚过高的局部透射率，迫使 ROI 内部生长出实质性非透明密度 | 权重或阈值过高会导致整个包围盒被实心雾块填满 |
| **深度方差损失（Depth Variance Loss, $\mathcal{L}_D$）** | 最大化局部视差图的方差，迫使生成物体具备三维立体起伏而非平面贴纸 | 过强会导致物体过度膨胀变形或遮挡邻近结构 |
| **正则退火策略（Annealing）** | 训练初期与后期动态调节正则权重，避免初期几何约束压垮语义探索 | 退火周期需随迭代步数针对性微调 |

具体而言，为了使采样相机视野恰好匹配包围盒尺寸，论文根据包围盒最大边长 $e_{max}$ 与相机半视场角 $\gamma$ 计算基础采样距离：

$$
d = \frac{e_{max}}{2\tan(\gamma/2)}.
$$

在此相机分布下，**透射率损失 $\mathcal{L}_T$** 约束相机位姿 $P$ 下的局部平均透射率不超过上限阈值 $\tau$（论文默认取 $\tau = 0.88$）：

$$
\mathcal{L}_T
=
-\min(\tau, \operatorname{mean}(T(P))).
$$

**深度方差损失 $\mathcal{L}_D$** 则对局部视差图（disparity map）$D(P)$ 的空间方差 $\sigma^2(D(P))$ 施加截断最大化约束（上限阈值为 $\rho$），从而消除扁平纸片化结构：

$$
\mathcal{L}_D
=
-\min(\rho, \sigma^2(D(P))).
$$

最终，源码 `ObjectnessLoss` 封装的总体优化目标为三者的加权和：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{sim}
+ \lambda_T\mathcal{L}_T
+ \lambda_D\mathcal{L}_D.
$$

## 4. 体渲染级融合机制：距离平滑与双模式密度合成

### 4.1 距离平滑算子（Distance Smoothing Operator，Fig. 3）

训练完成后，若在渲染时对 ROI 包围盒内外采取非黑即白的硬切换（盒内全用 $F^G_\theta$、盒外全用 $F^O_\theta$），包围盒六个切面上极易暴露出刺眼的几何截断与色彩接缝。Fig. 3 展示了论文设计的距离平滑算子如何实现从编辑核心向原始背景的渐进过渡。

![Fig. 3: Distance Smoothing Operator](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig03-distance-smoothing.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 3, CC BY-NC-ND 4.0.*

给定空间采样点 $\mathbf{x}$，设其到 ROI 中心的欧氏距离为 $d(\mathbf{x})$，包围盒对角线长度为 $diag$，距离平滑权重函数定义为：

$$
f(\mathbf{x}) = 1 - \exp\left(-\frac{\alpha d(\mathbf{x})}{diag}\right),
$$

其中超参数 $\alpha$ 控制衰减速率。利用该几何启发式权重，盒内采样点的融合密度与颜色按距中心远近做凸组合：

$$
\sigma_{blend}(\mathbf{x}) =
f(\mathbf{x})\sigma_O(\mathbf{x})
+ (1-f(\mathbf{x}))\sigma_G(\mathbf{x}),
$$

$$
\mathbf{c}_{blend}(\mathbf{x}) =
f(\mathbf{x})\mathbf{c}_O(\mathbf{x})
+ (1-f(\mathbf{x}))\mathbf{c}_G(\mathbf{x}).
$$

当采样点位于 ROI 中心附近时，$d(\mathbf{x})\to 0, f(\mathbf{x})\to 0$，体渲染由生成分支 $F^G_\theta$ 主导；当采样点移向包围盒边缘时，$f(\mathbf{x})$ 单调递增，原始分支 $F^O_\theta$ 的比重平滑上升。从 Fig. 3 可以清晰观察到，随着 $\alpha$ 增大，原始松果的几何与色调保留得越来越多，边界过渡也更为柔和。

### 4.2 插入、替换与融合：激活函数内外的密度合成机制（Fig. 4 & Fig. 8）

在实际场景编辑中，用户的操作意图可归纳为三类：**新对象插入**（new object insertion）在原本空旷的 ROI 内无中生有地放置物体；**对象替换**（object replacement）要求抹除 ROI 内原有的物体（如树干或松果）并换上新物体（如草莓或菠萝）；**对象混合**（object blending）则希望原物体与新生成内容共生交织（例如让藤蔓与花朵缠绕着 Lego 推土机的车轮生长）。

Blended-NeRF 将这三类意图统一在采样点层级的双分支合成公式中。对于第 $i$ 个采样点 $\mathbf{x}_i$，先计算原始分支与生成分支各自经密度激活函数 $\phi(\cdot)$（如 ReLU 或 Softplus，确保密度非负）后的不透明度贡献：

$$
\alpha_O(\mathbf{x}_i)
=
1 - \exp(\phi(\sigma_O(\mathbf{x}_i))\delta_i),
\qquad
\alpha_G(\mathbf{x}_i)
=
1 - \exp(\phi(\sigma_G(\mathbf{x}_i))\delta_i).
$$

两分支的采样点颜色按照各自的不透明度权重做归一化混合（$S$ 为 Sigmoid 函数，$\epsilon$ 为防除零常数）：

$$
\mathbf{c}(\mathbf{x}_i)
=
S\left(
\frac{
\mathbf{c}_O(\mathbf{x}_i)\alpha_O(\mathbf{x}_i)
+ \mathbf{c}_G(\mathbf{x}_i)\alpha_G(\mathbf{x}_i)
}{
\epsilon + \alpha_O(\mathbf{x}_i)+\alpha_G(\mathbf{x}_i)
}
\right).
$$

而决定编辑行为是“纯增量叠加”还是“可擦除替换”的关键，在于**双分支密度究竟在激活函数 $\phi$ 内部相加，还是在激活函数外部相加**：

1. **激活函数内相加（Sum inside activation，论文 Eq. 8）**：
   $$
   \sigma(\mathbf{x}_i)
   =
   \phi(\sigma_O(\mathbf{x}_i)+\sigma_G(\mathbf{x}_i)).
   $$
   由于相加发生在非线性激活之前，生成分支可以输出足够大的**负值预激活密度** $\sigma_G(\mathbf{x}_i) < 0$ 来抵消原始密度 $\sigma_O(\mathbf{x}_i)$，使激活后的总密度归零。这意味着生成分支拥有了“擦除并改写原场景既有几何”的能力。
2. **激活函数外相加（Sum outside activation，论文 Eq. 9）**：
   $$
   \sigma(\mathbf{x}_i)
   =
   \phi(\sigma_O(\mathbf{x}_i))
   + \phi(\sigma_G(\mathbf{x}_i)).
   $$
   由于两项在通过 $\phi$ 后均已非负，生成分支只能向空间中叠加新密度，绝无可能削减原始分支的固有密度。因此原始物体会被完整保留，新生成内容只能包裹或依附在原物体周围。

Fig. 4 与 Fig. 8 直观展示了这两种密度合成模式在 Lego 车轮场景与 LLFF Fern 场景中的显著差异。

![Fig. 4: Blending Modes](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig04-blending-modes.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 4, CC BY-NC-ND 4.0.*

![Fig. 8: Blending densities inside activation](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig08-blending-densities-inside-activation.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 8, CC BY-NC-ND 4.0.*

在 Fig. 4 中，当采用激活外相加（Eq. 9）时，橡胶车轮本体结构完好无损，蓝白绿三色花朵沿着车轮外缘包裹生长；而切换为激活内相加（Eq. 8）及 Fig. 8 的 Fern 场景设置时，生成分支直接蚀刻并改写了部分原车轮与植物叶片的实体几何，使花朵、蘑菇与原场景发生了深度的结构侵蚀与融合。在官方源码 `raw2outputsBlend()` 中，这一核心分水岭由布尔参数 `sum_in_act` 直接控制。这也说明在构建智能 3D 编辑系统时，若要让视觉语言模型（VLM）、多模态大模型或交互式 UI 自动选择编辑模式，必须将用户自然语言中的“添加（add）”、“替换（replace）”、“缠绕混合（blend）”或“重绘材质（restyle）”准确映射到底层的密度激活合成开关与层冻结策略上。

## 5. 定量评估、基线对比与几何先验消融

### 5.1 与 Volumetric Disentanglement 的主实验对比（Fig. 5, Fig. 9 & Table 1）

论文选取 **Volumetric Disentanglement（Benaim et al., ECCV 2022）** 作为对象替换任务的核心对比基线。Fig. 5 展示了在 LLFF Fern 场景中将树干区域分别替换为“白杨树（aspen tree）”和“草莓（strawberry）”的正面对比，附录 Fig. 9 则补充了更多环绕视角的渲染结果。

![Fig. 5: Comparison to Volumetric Disentanglement for object replacement](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig05-volumetric-disentanglement-comparison.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 5, CC BY-NC-ND 4.0.*

![Fig. 9: Additional views for object replacement comparison](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig09-additional-object-replacement-views.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 9, CC BY-NC-ND 4.0.*

从 Fig. 5 与 Fig. 9 可以看出，Volumetric Disentanglement 侧重于无监督前景/背景体分解，在零样本替换复杂语义目标时容易出现前景残留、边界模糊与纹理失真；而 Blended-NeRF 借助显式 3D ROI 约束与双分支体融合，不仅在 ROI 内部生成了特征鲜明的白杨树干与草莓表面，还在多视角移动下完好保留了后方蕨类叶片与砖墙背景。

Table 1 给出了对应的定量评估结果，涵盖三项核心指标：**CLIP Direction Similarity**（衡量图像编辑方向与文本变化方向的一致性，越高越好）、**CLIP Direction Consistency**（衡量相邻渲染视角下编辑方向的一致性，越高越好）以及 **LPIPS**（评估非编辑背景区域的感知失真度，越低越好）。

![Table 1: Quantitative Evaluation](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-table01-quantitative-evaluation.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Table 1, CC BY-NC-ND 4.0.*

Blended-NeRF 在三项指标上全面领先于基线，尤其是 LPIPS 的显著降低验证了“ROI 局部优化 + 外部固定分支渲染”对原场景背景的高保真保护能力。

### 5.2 深度先验与方向提示词的消融分析（Fig. 6 & Table 2）

为了验证 3.3 节所述几何先验的必要性，Fig. 6 对比了在生成甜甜圈（donut）时有无深度方差损失 $\mathcal{L}_D$ 的彩色渲染图与视差图（disparity map）。

![Fig. 6: Depth Loss Impact](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig06-depth-loss-impact.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 6, CC BY-NC-ND 4.0.*

当移除 $\mathcal{L}_D$ 时（图左），虽然 RGB 视图在特定角度下勉强拼凑出了甜甜圈的色彩轮廓以迎合 CLIP，但下方的视差图暴露了其本质——一个几乎没有深度起伏的扁平贴片；引入 $\mathcal{L}_D$ 后（图右），视差图展现出饱满的环状立体厚度，多视角几何结构得以确立。

Table 2 进一步通过 CLIP 与 BLIP 的检索准确率（R-Precision，以真实图像数据集 COCO GT 作为参考上界）量化了各组件的消融贡献：

![Table 2: Ablation study](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-table02-ablation-study.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Table 2, CC BY-NC-ND 4.0.*

数据表明，完整流水线（Full pipeline）在 CLIP 与 BLIP R-Precision 上均取得最高分；移除视角方向提示词（W/O directional prompts）会导致跨视角语义轻微下滑；而移除深度先验（W/O depth prior）则会引发检索准确率的明显跌落，证明良好的三维几何形态本身也是维持多视角图文语义一致性的基础。

## 6. 多场景定性案例剖析：对象插入、替换与材质转换

为了全面检验算法在不同相机轨迹与场景复杂度下的泛化表现，论文在合成数据集（Blender）、前向真实场景（LLFF）以及 360 度环绕真实场景（Real 360，如 vasedeck 与 pinecone）上开展了系统的定性实验。

### 6.1 合成与真实 360 场景中的对象插入与替换（Fig. 10–12）

Fig. 10 展示了在多个既有场景中执行零样本新对象插入（object insertion）的双视角结果；Fig. 11 与 Fig. 12 则将测试推向更具挑战性的 Real 360 真实环绕场景，分别在花瓶甲板（vasedeck）上插入花瓣对象，以及将松果（pinecone）替换为菠萝。

![Fig. 10: Object Insertion](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig10-object-insertion.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 10, CC BY-NC-ND 4.0.*

![Fig. 11: Object insertion in vasedeck 360 scene](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig11-vasedeck-object-insertion.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 11, CC BY-NC-ND 4.0.*

![Fig. 12: Object replacement in 360 pinecone scene](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig12-pinecone-object-replacement.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 12, CC BY-NC-ND 4.0.*

把这三组实验放在一起审视，可以看出 Blended-NeRF 在处理真实 360 度场景时的独特优势与客观边界：真实采集场景存在复杂的自然光照、散景背景与非均匀相机分布，由于生成分支直接从原场景 NeRF 初始化并配合距离平滑算子，新插入的花瓣或替换出的菠萝能够较好地融入原有的石板、木桌与苔藓色调中，未破坏周围环境的几何连续性。但另一方面，插入对象与承托平面之间并未建立真正的物理接触或投射阴影（如 Fig. 10 中部分插入物体略显悬浮），且受限于 CLIP 的低频语义监督，生成的三维表面仍偏向卡通化或平滑色块。

### 6.2 冻结几何密度的材质与纹理编辑（Fig. 7, Fig. 13 & Fig. 14）

当用户希望保持场景中已有物体的三维形状不变、仅对其执行风格化或材质替换（texture editing / conversion）时，Blended-NeRF 提供了一种极其稳定的微调模式：**在生成分支中完全冻结影响体密度 $\sigma$ 的网络层，仅放开颜色预测层（color branch）的参数更新。** Fig. 7 展示了该模式在松果与花瓶上的代表性效果，附录 Fig. 13 与 Fig. 14 则给出了 Real 360 场景下更丰富的多视角材质转换矩阵。

![Fig. 7: Texture Editing](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig07-texture-editing.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 7, CC BY-NC-ND 4.0.*

![Fig. 13: Texture conversion on 360 pinecone scene](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig13-pinecone-texture-conversion.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 13, CC BY-NC-ND 4.0.*

![Fig. 14: Texture conversion on 360 vasedeck scene](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig14-vasedeck-texture-conversion.webp)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 14, CC BY-NC-ND 4.0.*

无论是将真实松果转化为“燃烧的炭火（burning）”、“冰封状态（frozen）”或“粉色毛线（pink wool）”（Fig. 13），还是将真实花瓶转换为玻璃、岩石或水彩质感（Fig. 14），其多视角几何稳定性都明显高于从零生成几何的对象替换任务。原因在于原始 NeRF 已经通过多视角重建锁定了精准的三维表面密度分布，冻结密度层后，CLIP 损失不会再诱发薄片化或漂浮块伪影，只需专注于表面辐射颜色的重新着色。当然，这种纯色彩层微调无法创造超出原几何轮廓的新拓扑结构，且渲染结果仍为隐式辐射场而非具备独立法线贴图与粗糙度通道的工业 PBR 网格材质。

## 7. 官方源码剖析与工程复现指南

### 7.1 代码架构与核心渲染融合链路

官方开源仓库 [orig333/Blended-NeRF](https://github.com/orig333/Blended-NeRF) 基于 PyTorch 实现，整体工程结构紧凑清晰。按照 README 给出的标准启动链路：

```bash
conda create --name blended-nerf python=3.9
conda activate blended-nerf
git clone https://github.com/orig333/Blended-NeRF.git
cd Blended-NeRF
pip install -r requirements.txt

bash download_data.sh nerf_synthetic
bash download_data.sh nerf_llff
bash download_data.sh nerf_real_360

python main.py --config ./configs/config.txt
```

每一次编辑实验均需准备三项显式输入资产：原始场景图像与相机参数、预训练好的原场景 NeRF 权重（放入 `base_weights/`），以及预定位的 3D ROI 顶点文件（放入 `box_points/`，如 `configs/lego.txt` 中的 `box_points/lego_exhaust_box.pt`）。入口脚本 `main.py` 直接调用 `BlendedNeRF/run_BlendedNeRF.py` 中的 `train_blended_nerf()`，其核心执行逻辑可拆解为六个关键函数模块：

| 核心函数 / 模块 | 对应论文机制 | 源码实现细节 |
| --- | --- | --- |
| `config_parser()` | 实验与超参配置 | 解析文本提示、ROI 路径、融合开关 `sum_in_act`、$\lambda_T/\lambda_D$ 权重及数据集坐标参数 |
| `create_nerf(args)` | 双分支网络初始化 | 构建 `fixed_model` 与 `model`（含 coarse/fine 两级），统一加载 base checkpoint |
| `render_rays()` | 采样点入盒判定 | 计算 ROI 的宽、高、深与对角线 `diag`，执行轴对齐范围判定并调度局部/全局渲染 |
| `render_in_box_fn()` | ROI 局部体渲染 | 仅对盒内采样点积分，返回 `rgb_map_in_box`、`transmittance_in_box` 与 `depth_map_in_box` |
| `raw2outputsBlend()` | 体渲染级融合（Eq. 6–9） | 计算距离平滑权重 $f(\mathbf{x})$，根据 `sum_in_act` 合成双分支 raw 密度与颜色后积分 |
| `ObjectnessLoss` | 语义与几何联合损失 | 位于 `optimization/losses.py`，汇总 CLIP/BLIP 相似度、透射率损失与视差方差损失 |

在 `render_rays()` 中，采样点的入盒测试直接决定了局部优化的空间掩码：

```python
pt_is_in_box = (
    (box_points[:, 0].min() <= pts[..., 0]) &
    (box_points[:, 0].max() >= pts[..., 0]) &
    (box_points[:, 1].min() <= pts[..., 1]) &
    (box_points[:, 1].max() >= pts[..., 1]) &
    (box_points[:, 2].min() <= pts[..., 2]) &
    (box_points[:, 2].max() >= pts[..., 2])
)
```

而在 `raw2outputsBlend()` 中，代码首先把 `fixed_model` 与 `model` 的原始预测转换为不透明度权重，再根据 `sum_in_act` 分支决定是在激活函数前相加（支持负值擦除替换）还是激活函数后相加（纯增量混合），最后调用标准体渲染公式汇总像素颜色与深度图。

### 7.2 本地复现核对清单与论文阅读路线

在本地复现或基于源码二次开发时，建议按照以下清单逐项核对，尤其需要留意不同数据集的坐标系转换（如 LLFF 的 NDC 坐标与世界坐标映射 `ndc2world`、`box_points_world`）：

| 复现检查项 | 关键验证内容 |
| --- | --- |
| **运行环境与依赖** | Python 3.9 + PyTorch，确保 CLIP（`ViT-B/32`）与 BLIP/LAVIS 权重可正常加载 |
| **底模权重与坐标对齐** | 确认 `base_weights/` 下的预训练模型与当前 `configs/*.txt` 的数据集类型、缩放比例及 NDC 设置严格匹配 |
| **3D ROI 包围盒验证** | 先通过 `notebooks/find_3d_box_*.ipynb` 可视化检验 `box_points/*.pt` 是否准确包裹目标区域且未过度侵占背景 |
| **训练监控与中间诊断** | 检查 `logs/<scene>/<expname>/plots/in_box` 下的局部渲染图、透射率与视差图，排查是否出现全透明或纸片化退化 |
| **融合模式与边界评估** | 对比不同距离平滑系数 $\alpha$ 与 `sum_in_act` 开关下的多视角环绕视频，确认边界接缝与背景 LPIPS 保真度 |

若希望在有限时间内高效吃透原文与代码，推荐按以下顺序推进：先读 Abstract、Fig. 1 与 Sec. 3，抓牢双分支初始化、ROI-only rendering、距离平滑（Fig. 3）与激活函数内外密度合成（Fig. 4、Eq. 6–9）的数学定义；接着对照 Fig. 5–6 与 Table 1–2，理解基线对比优势以及深度方差损失对抑制扁平几何的关键作用；随后浏览 Fig. 7–14 在插入、替换与材质转换上的定性表现；最后直接阅读 `BlendedNeRF/run_BlendedNeRF.py` 中的 `render_rays()` 与 `raw2outputsBlend()`，将公式推导与代码张量操作一一闭环。

## 8. 局限性审视与总结

### 8.1 方法局限性与工业落地差距

尽管 Blended-NeRF 构建了一套清晰可用的隐式场景局部编辑框架，但从研究原型走向工业级 3D 内容生产工具，它仍存在三方面的显著局限：

- **空间交互先验的刚性限制**：整个编辑流程高度依赖用户预先给定的 3D ROI 包围盒。在三维辐射场中手动调整包围盒对普通用户而言门槛较高，且轴对齐长方体（AABB）难以贴合轮毂、弯曲树干、细长瓶口或斜向构件等非规则几何区域——包围盒过紧会截断生成物体，过松则会把周围无关背景卷入修改范围。论文虽指出可扩展为圆柱体或球体 ROI，但本质上仍缺乏基于三维语义分割场的自动对象拾取能力。
- **粗粒度语义监督与几何退化风险**：以 CLIP 或 BLIP 为代表的二维图文对比损失并非三维几何生成器。虽然论文通过透射率损失 $\mathcal{L}_T$、深度方差损失 $\mathcal{L}_D$、随机背景增强和环绕相机采样缓解了“单视角贴纸”和“漂浮噪点”现象，但在复杂遮挡或大视角跨度下，生成物体的表面细节依然偏软，且缺乏与原场景承托平面之间的真实物理接触与阴影投射建模。
- **单次优化耗时与非标资产输出**：每一次局部编辑都需要在具体场景和提示词下重新迭代优化生成分支，无法实现毫秒级的实时交互反馈。同时，编辑产物依然是耦合在 MLP 中的神经辐射场渲染视图，无法直接导出为具备干净拓扑网格、独立 UV 与 PBR 材质通道的标准工业资产，且论文采用的 CLIP Direction 与 R-Precision 指标仅能作为语义代理指标（proxy metrics），无法直接衡量三维网格质量。

### 8.2 结论

Blended-NeRF 的核心贡献在于将零样本 3D 编辑从“全局盲目优化单个隐式场”推进到了 **“原场景权重初始化 + 3D ROI 局部渲染约束 + 生成分支靶向优化 + 采样点级体渲染融合”** 的结构化范式。它通过距离平滑算子与激活函数内外的双模式密度合成，在一个统一框架内优雅地兼容了对象插入、对象替换、几何缠绕融合与冻结密度的材质转换，并通过开源代码完整展示了工程落地细节。在 3D 场景编辑的技术演进历程中，它既验证了隐式场局部非破坏性编辑的可行性，也为后续结合扩散先验（SDS）、三维语义分割（如 LERF / SA3D）以及显式高保真表示（如 3D Gaussian Splatting）指明了清晰的改进方向。

## References

- Gordon, Ori, Omri Avrahami, and Dani Lischinski. [Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields](https://arxiv.org/abs/2306.12760). arXiv:2306.12760v2, 2023.
- arXiv PDF: [https://arxiv.org/pdf/2306.12760](https://arxiv.org/pdf/2306.12760)
- Project page: [https://www.vision.huji.ac.il/blended-nerf/](https://www.vision.huji.ac.il/blended-nerf/)
- Official code: [https://github.com/orig333/Blended-NeRF](https://github.com/orig333/Blended-NeRF)
- Mildenhall et al. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf), ECCV 2020.
- Jain et al. [Zero-Shot Text-Guided Object Generation with Dream Fields](https://arxiv.org/abs/2112.01455), CVPR 2022.
- Wang et al. [CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields](https://arxiv.org/abs/2112.05139), CVPR 2022.
- Benaim et al. [Volumetric Disentanglement for 3D Scene Manipulation](https://arxiv.org/abs/2206.02776), ECCV 2022.
- Poole et al. [DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988), ICLR 2023.
- Wang et al. [Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation](https://arxiv.org/abs/2212.00774), CVPR 2023.

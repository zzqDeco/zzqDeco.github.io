---
title: "CasLayout 论文精读：级联 3D 布局扩散、稀疏关系建模与室内场景生成"
description: "精读 CasLayout 如何通过四阶段条件扩散、建筑约束、稀疏关系图与 Relation VAE 实现可控 3D 室内布局生成"
pubDate: "2026-06-26T09:24:04+08:00"
updatedDate: "2026-06-26T09:24:04+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "3D Scene Synthesis"
  - "Diffusion Models"
  - "Computer Vision"
  - "Generative AI"
draft: false
---

[CasLayout: Cascaded 3D Layout Diffusion for Indoor Scene Synthesis with Implicit Relation Modeling](https://arxiv.org/abs/2604.27361) 由 Yingrui Wu、Youkang Kong、Mingyang Zhao、Weize Quan、Dong-Ming Yan、Yang Liu 撰写（arXiv 提交日期 2026 年 4 月 30 日，DOI `10.1145/3811347`）。该工作将复杂的室内家具布局拆解为四个渐进可控的条件扩散阶段：

```text
建筑结构
  -> 家具数量和类别
  -> 家具尺寸和几何检索特征
  -> 稀疏关系 latent
  -> 最终 OBB 布局
```

论文的核心设计围绕四个技术支点展开：

1. 将室内布局从高维联合分布学习改写为四阶段级联条件生成（Cascaded Diffusion）。
2. 将墙体（walls）、门（doors）、窗（windows）等建筑构件（building elements）显式纳入全流程条件约束，摆脱了仅在空盒子里摆放家具的简化假设。
3. 摒弃冗余的全连接稠密关系图（dense relation graph），提出符合人类空间描述习惯的稀疏功能关系图（sparse relation graph）。
4. 设计双向关系变分自编码器（Bidirectional Relation VAE），将稀疏有向图压缩为节点对齐的关系潜变量（node-aligned relation latent），交由扩散模型生成与控制。

CasLayout 将 LLM/VLM、场景图、扩散模型、CAD 检索、建筑约束与用户交互编辑整合在同一模块化管线中，对 3D 室内场景生成以及“高层语言/视觉意图如何落地为连续几何结构”的多模态系统设计均具参考价值。

> **代码与数据开放状态说明**：论文给出的官方仓库为 [YingruiWoo/CasLayout](https://github.com/YingruiWoo/CasLayout)，目前仅包含 README（显示 “Official code and data for CasLayout -- coming soon.”），尚未公开训练、数据处理、评估源码及 3D-Front-Relationship 数据集。下文基于论文正文与实验报告深入拆解其方法架构与设计边界。

## 1. 任务定义与核心动机：为什么室内布局需要级联生成

### 1.1 3D 室内布局生成的决策维度

3D 室内场景合成（3D indoor scene synthesis）旨在生成可被渲染、交互编辑、碰撞检测乃至供机器人与仿真系统直接调用的结构化 3D 场景。一套完整的家具布局系统需要协同决定以下多个维度的变量：

| 决策对象 | 例子 | 难点 |
| --- | --- | --- |
| 家具数量 | 一个卧室里有几张床、几个床头柜 | 数量和房间功能、面积、门窗位置相关 |
| 家具类别 | sofa、coffee table、wardrobe、dining chair | 类别组合要符合场景语义 |
| 家具尺寸 | 床、柜子、桌椅的长宽高 | 尺寸影响可放置性和关系定义 |
| 家具风格或几何特征 | 从 CAD 数据库检索哪个模型 | 影响视觉一致性和几何可用性 |
| 位置和朝向 | 每个物体的 OBB | 要避免碰撞、越界、挡门窗 |
| 物体关系 | nightstand attach to bed、chair front of table | 要符合功能逻辑和用户控制 |
| 建筑约束 | walls、doors、windows、floor plan | 布局不能违反真实房间结构 |

早期的布局生成方法往往只考虑房间外边界和家具间的防碰撞。然而真实室内设计包含更细粒度的空间逻辑：门后必须留出开合区域，窗户不能被高柜完全遮挡，餐桌与餐椅需组成功能区，沙发与茶几需保持合理间距，床头柜需贴近床侧且成对对称。

现有生成方法在应对上述需求时普遍面临三方面瓶颈：
- **高维联合生成难度大**：若让单一模型直接同时预测所有物体的类别、尺寸、形状特征、位置与旋转，模型需要拟合极高维的联合分布，而有限的专业 3D 室内设计数据集难以覆盖如此庞大的组合空间。
- **稠密关系图负担重**：许多基于场景图的方法为每对物体建立边连接，形成 $O(N^2)$ 的全连接关系空间。但室内并非任意两个物体间都存在功能关联（例如电视柜与餐椅之间往往只是偶然的相对位置），强行建模所有成对关系会引入大量高熵噪声与冲突约束。
- **LLM/VLM 难以直接回归连续几何**：大语言模型擅长列出家具清单并描述“床头柜在床两侧”，却难以直接输出无碰撞、符合门窗约束的精确连续坐标。CasLayout 因此让 LLM/VLM 承担高层语义规划，将连续几何执行交给条件扩散模型。

### 1.2 核心背景概念：从 3D-FRONT 到 OBB

在深入公式之前，先厘清论文涉及的几项基础设定：

- **3D-FRONT 与 3D-FUTURE**：3D-FRONT 是包含房间结构与家具摆放的 3D 室内场景数据集，CasLayout 使用其中的卧室、餐厅与客厅数据，并在此基础上抽取空间关系构建了 **3D-Front-Relationship** 数据集。3D-FUTURE 则是配套的家具 CAD 模型库；布局系统通过预测类别、尺寸与几何检索特征，从 3D-FUTURE 中检索匹配的 3D 模型资产，从而专注于空间布局与检索特征预测。
- **户型图（Floor plan）与建筑构件（Building elements）**：户型图在文中以二值图像（binary image）形式通过交叉注意力注入扩散网络；墙（wall）、门（door）、窗（window）等建筑构件则作为条件节点显式建模，并在各扩散阶段维持重建监督，构成布局生成必须遵守的硬性几何约束。
- **带朝向三维包围盒（OBB, Oriented Bounding Box）**：单个家具的空间布局用 OBB 表示为
  $$
  B = (s,t,r),
  $$
  其中 $s\in\mathbb{R}^3$ 为长宽高尺寸，$t\in\mathbb{R}^3$ 为相对房间中心的平移，$r=(\cos\theta,\sin\theta)$ 为绕竖直轴旋转角的连续向量表示（采用余弦与正弦分量可消除角度周期边界的不连续性）。
- **场景关系图与多模态控制接口**：用节点表示物体、边表示空间或语义关系。CasLayout 构建稀疏关系图并将其压缩为隐式潜变量。在文本生成场景（text-to-scene）与图像生成场景（image-to-scene）任务中，系统遵循如下链路：

```text
text / image
  -> LLM / VLM 抽取 object list 和 sparse relations
  -> CasLayout 生成尺寸、关系 latent 和 OBB
  -> CAD retrieval 得到最终 3D 场景
```

### 1.3 级联分解的概率形式化

专业室内设计遵循由粗到细的渐进决策过程：先理解房间建筑结构，再确定家具类别与数量，随后规划尺寸风格与功能关系，最后落实到具体几何摆放。

若直接建模给定建筑条件 $A$ 下的完整家具布局 $L$（涵盖类别、特征、尺寸、关系、平移与旋转），需要直接拟合高维分布 $p(L\mid A)$。CasLayout 将其分解为四阶段条件链：

$$
p(L\mid A)
\approx
p(C\mid A)
p(P\mid C,A)
p(R_z\mid C,P,A)
p(B\mid C,P,R_z,A),
$$

其中 $C$ 为家具类别与数量，$P$ 为家具属性（含尺寸与形状检索特征），$R_z$ 为关系潜变量，$B$ 为最终 OBB 布局，$A$ 为户型图与墙门窗等建筑条件。

这一分治建模策略使每个阶段只需学习维度较低、语义明确的条件分布，同时带来三方面工程收益：一是显著降低了单阶段模型的学习难度；二是增强了交互可控性，允许用户在任意阶段插入条件（如固定家具列表、给定局部布局、指定关系图或修改门窗位置）；三是天然适配 LLM/VLM 的输出形态，将语言模型擅长的家具清单与稀疏关系直接对接为 Stage 1 与 Stage 3 的中间控制接口。

整体级联流程如下：

```text
Input:
  floor plan image
  walls / doors / windows
  optional object list / partial layout / sparse relation graph / text / image

Stage 1: Furniture Type Diffusion
  building elements -> object count + furniture categories

Stage 2: Furniture Property Diffusion
  categories + building elements -> size + feature embedding
  feature embedding -> CAD model retrieval

Stage 3: Relation Latent Diffusion
  categories + sizes + building elements -> node-aligned relation latent
  sparse relation graph <-> Relation VAE <-> latent vectors

Stage 4: Box Layout Diffusion
  categories + sizes + relation latent + building elements -> OBBs

Output:
  furniture OBB layout + retrieved CAD models
```

## 2. 房间表示与符号体系：建筑约束与节点渐进补全

论文将带家具的 3D 房间表示为布局 $L$，由建筑元素集合 $A$ 与家具元素集合 $F$ 共同组成。

建筑元素包含户型图以及 $m$ 个墙、门、窗构件：

$$
A=\{b_i\}_{i=1}^{m}, \quad b_i=(c_i,B_i), \quad c_i\in\{\text{wall},\text{door},\text{window}\}.
$$

家具元素由 $n$ 个待生成物体构成：

$$
F=\{f_j\}_{j=1}^{n}, \quad f_j=(c_j, feat_j, B_j).
$$

每个家具包含类别 $c_j$、几何检索特征 $feat_j$ 以及三维包围盒 $B_j=(s_j,t_j,r_j)$，其中 $s_j\in\mathbb{R}^3$ 表示尺寸（size），$t_j\in\mathbb{R}^3$ 表示平移（translation），$r_j=(\cos\theta_j,\sin\theta_j)\in\mathbb{R}^2$ 表示旋转（rotation）。

论文为每个 OBB 定义了物体局部坐标系：规定 $+x$ 轴为左侧（left），$+y$ 轴为正前方（front），反方向分别为右侧（right）与后方（behind）。相比全局坐标系，在局部坐标系下定义空间关系不会随房间整体旋转或家具朝向改变而发生语义漂移，更契合“椅子位于桌子正前方”这类设计语言。

在扩散网络内部，每个节点编码为 token：

$$
e_i=(c_i,s_i,t_i,r_i,pe_i),
$$

其中 $pe_i$ 为区分同类物体的位置编码（positional encoding）。为支持固定长度批处理，系统设定最大节点数 $N_{\max}$，不足部分以 `None` 类别和零向量填充；Stage 1 预测完成后即滤除多余的空节点，仅保留有效家具节点与建筑构件节点进入后续阶段。

每个家具节点的属性在四个阶段中按因果依赖逐步填充：

| 阶段 | 节点已知信息 | 节点待生成信息 | 下游依赖 |
| --- | --- | --- | --- |
| Stage 1 输入前 | building elements、floor plan | 家具数量、类别 | 后续所有 furniture token 的存在性 |
| Stage 1 后 | 类别 $c_j$、存在 mask | 尺寸、feature、关系、OBB | property diffusion 的条件 |
| Stage 2 后 | 类别、尺寸 $s_j$、feature/code | 关系 latent、位置、朝向 | CAD retrieval 与 relation diffusion |
| Stage 3 后 | 类别、尺寸、feature、关系 latent $z_j$ | 平移 $t_j$、旋转 $r_j$ | box layout diffusion |
| Stage 4 后 | 完整 OBB 与检索特征 | mesh retrieval / rendering | 应用和评估 |

这种阶段顺序体现了变量间的几何依赖：空间关系依赖类别与尺寸（不知道物体是床还是衣柜、不知道椅子的具体尺寸，就难以定义贴合、对齐或合理间距），而具体坐标与朝向又依赖功能关系（先明确“餐椅围绕餐桌”“床头柜贴近床侧”，再求解连续几何位置）。

下表汇总了核心数学符号及其含义：

| 符号 | 含义 | 说明 |
| --- | --- | --- |
| $A$ | architectural condition | 房间边界、floor plan、building elements |
| $B_A$ | building element set | walls、doors、windows 等建筑节点 |
| $F$ | furniture set | 所有待生成家具节点 |
| $N$ | 家具数量 | 不同房间不同，训练时通常需要 padding |
| $c_j$ | 第 $j$ 个家具类别 | bed、sofa、table 等离散类型 |
| $s_j$ | 家具尺寸 | OBB 的长宽高 |
| $e_j$ | 家具几何 feature/code | 用于 CAD retrieval |
| $t_j$ | 家具平移 | 相对房间坐标系 |
| $r_j$ | 家具旋转 | 通常写成 $(\cos\theta_j,\sin\theta_j)$ |
| $g_{ij}$ | 物体 $i\to j$ 的显式关系 | direction、distance、alignment、symmetry 等 |
| $z_j$ | node-aligned relation latent | Relation VAE 编码后的隐式关系表示 |
| $R_z$ | relation latent set | 第三阶段 diffusion 生成的节点 latent 集合 |
| $L$ | 完整布局 | 类别、尺寸、feature、关系 latent、OBB 的组合 |

## 3. 四阶段级联扩散框架：从类别、属性、关系到包围盒

CasLayout 的四个扩散阶段均采用基于 Transformer 主干的 DDPM 去噪扩散机制，各阶段针对不同的目标变量加噪与去噪。

```text
输入 floor plan / building elements
  |
  |-- Type Diffusion:
  |     生成 furniture token 的存在性和类别
  |
  |-- Property Diffusion:
  |     在类别条件下生成 size 和 retrieval feature/code
  |
  |-- Relation Latent Diffusion:
  |     在类别、尺寸和建筑条件下生成每个节点的 relation latent
  |
  |-- Box Layout Diffusion:
        在所有上游条件下生成 translation 和 rotation
```

### 3.1 Stage 1：家具类别扩散（Furniture Type Diffusion）

给定建筑条件 $A$，第一阶段预测哪些家具节点存在及其对应类别，回答“房间需要哪些家具、每类家具需要几个、哪些填充空节点应当丢弃”。

训练时，模型将离散类别独热向量视为连续向量，施加高斯噪声：

$$
c_{i,t}
=
\sqrt{\gamma(t)}c_i
+
\sqrt{1-\gamma(t)}\epsilon,
$$

并训练去噪网络 $\epsilon_{\theta}$ 最小化噪声预测均方误差：

$$
\mathcal{L}_{diff}
=
\mathbb{E}_{\epsilon,t}
\left[
\lVert
\epsilon-\epsilon_{\theta}(X_t,F,t)
\rVert_2^2
\right],
$$

其中 $F$ 为户型图条件，$X_t$ 为加噪后的节点 token 集合。Stage 1 产出的家具清单直接决定后续阶段的节点数量与类别条件。

### 3.2 Stage 2：家具属性扩散（Furniture Property Diffusion）

在家具类别已知的条件下，第二阶段生成每个家具的尺寸 $s_i$ 与几何检索特征（feature embedding）。尺寸 $s_i$ 决定后续空间布局的距离与对齐计算；检索特征则用于从 3D-FUTURE 库中匹配同类别且几何特征最接近的 CAD 模型。

为了提升特征的判别力并降低生成难度，论文引入预训练 VQ-VAE 将高维连续形状特征离散化为码本索引（discrete feature index）。同时，作者在实验分析中发现：若将形状检索特征继续作为下游 Stage 3 和 Stage 4 的条件输入，模型容易对特定 CAD 几何过拟合，反而损害关系建模质量与摆放精度。因此，CasLayout 对中间变量的传递做了明确筛选：

| 信息 | 用途 | 是否进入后续 layout 条件 |
| --- | --- | --- |
| furniture type | 决定关系和布局语义 | 是 |
| size | 决定可放置性和距离 | 是 |
| feature embedding | 检索 CAD 模型 | 否 |
| building elements | 约束物理空间 | 是 |

将特征与尺寸置于具体坐标生成之前，确立了“先确定家具形态与尺度，再决定精确落位”的因果次序。

### 3.3 Stage 3：关系潜变量扩散（Relation Latent Diffusion）

第三阶段生成与每个节点对齐的关系潜变量集合：

$$
R_z=\{r^l_i\}_{i=1}^{N}.
$$

这里的 $r^l_i$ 来自双向 Relation VAE 的编码空间。即便采用稀疏关系图，若直接在显式 $N\times N$ 关系矩阵上做扩散，矩阵中仍充斥大量空边（`None`），维度与稀疏性依然制约生成效果。通过 Relation VAE 将成对关系压缩至节点级潜变量，Stage 3 在条件 $(C,P,A)$ 下只需合成 $O(N)$ 个紧凑的节点潜向量，既能表达“床头柜贴床”“餐椅围绕餐桌”等局部结构约束，又避免了直接操作二次方规模的边矩阵。

### 3.4 Stage 4：包围盒布局扩散（Box Layout Diffusion）

第四阶段在家具类别、家具尺寸、建筑构件向量、户型图以及关系潜变量的共同约束下，生成每个家具的平移 $t_i$ 与旋转 $r_i$，组装成完整 OBB $B_i=(s_i,t_i,r_i)$。

网络在此阶段引入两条交叉注意力路径：一条关注关系潜变量（relation latents）以保障物体间的功能组织一致，另一条关注户型图特征（floor plan image）以保障几何位置不越界或冲突。

### 3.5 级联架构的收益与误差传播分析

级联分解兼具模块化、强可控性与高样本效率，但也引入了串联系统的典型风险——上游误差传播。若 Stage 1 漏掉关键家具或 Stage 2 预测尺寸失真，下游阶段难以自行逆转上游条件错误。四个阶段的典型故障模式与潜在修复机制对照如下：

| 阶段 | 典型错误 | 用户感知 | 可能修复方向 |
| --- | --- | --- | --- |
| Type | 多放或少放关键家具 | 房间功能不完整，例如卧室没有床 | object-list verifier、room-type prior、LLM checklist |
| Property | 尺寸不合适或检索风格不协调 | 家具比例怪、资产不统一 | size prior、style embedding、CAD diversity reranking |
| Relation | 关系 latent 错误或冲突 | 桌椅、床柜等功能组不成立 | sparse graph consistency check、relation classifier feedback |
| Box | 位置/朝向不合理、碰撞、挡门窗 | 场景不可用或不美观 | collision guidance、door-window clearance verifier、local refinement |

级联设计将问题边界梳理清晰：当生成结果不理想时，调试者可以精准定位是物体清单、尺度特征、关系潜变量还是几何落位出了问题，并针对性地接入校验器（verifier）、重采样策略或人工编辑接口。

## 4. 稀疏关系捕获与 3D-Front-Relationship 构建

室内设计中的有效空间约束天然具有稀疏性。CasLayout 明确区分了家具间关系与家具-建筑关系，并通过功能分区与锚点机制构建了 **3D-Front-Relationship** 数据集。

### 4.1 家具间关系（Furniture-Furniture Relations）

家具之间定义了四类空间关系：

| 关系类别 | 子类或定义 | 例子 |
| --- | --- | --- |
| Direction | left、right、front、behind、under、above | nightstand left of bed |
| Distance | attach to、adjacent、distant | chair attach to table |
| 2D Alignment | edge-align、x-center-align、y-center-align | two nightstands edge-aligned |
| Symmetry | same type、same feature、same size，且镜像对称 | bed 两侧床头柜 |

方位关系（Direction）在参考物体的局部坐标系下根据 front/left 轴向判定；距离关系（Distance）根据两个 OBB 的最小欧氏距离划分为紧贴（attach）、相邻（adjacent）与远离（distant）；二维对齐（2D Alignment）检测边缘或中心轴对齐；对称关系（Symmetry）则要求两个物体类别、检索特征与尺寸一致，且在水平面上呈镜像对称。

### 4.2 家具-建筑关系（Furniture-Building Relations）

墙体、门与窗户被表示为线段或零厚度 OBB。系统针对每个家具仅计算其到最近墙体的距离，但会完整计算其到所有门和窗的距离。这一设计直接对应真实物理规范：家具不得侵入屋门的开合扇区，窗前需保留采光与通行空间，大型家具不得穿出户型边界。

### 4.3 基于功能分区与锚点的稀疏关系图

为了滤除无意义的偶然相对位置，CasLayout 将家具按语义划分为不同功能区（functional zones），包括休闲区（lounging zone）、就餐区（dining zone）、睡眠区（bedding zone）、照明区（lighting zone）及其他（others）：

```text
Dense graph:
  every object pair has a relation candidate
  O(N^2) edges
  many relations are accidental or near-uniform

Sparse functional graph:
  preserve intra-zone relations
  choose anchors for each zone
  connect anchors across zones
  fewer edges, higher information density
```

在同一功能区内部，保留物体间的细粒度关系（如餐桌与餐椅、床与床头柜、沙发与茶几）；在不同功能区之间，仅为每个功能区选定一个核心锚点物体（anchor item），只建立锚点间的跨区关系。

论文使用信息熵量化关系分布的集中程度。设某类关系在 $n$ 个物体对、$m$ 个子类别上的概率分布为 $p(x_{ij})$，关系熵定义为：

$$
H(X)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{j=1}^{m}
p(x_{ij})\log_2 p(x_{ij}).
$$

在客厅（living room）的方位关系上，稀疏图的平均熵从稠密图的 `1.62` 显著降至 `1.04`。这表明稀疏化处理滤除了大量接近均匀分布的背景噪声边，大幅提升了每条关系边的语义信息密度。基于这套规则从 3D-FRONT 抽取出的 **3D-Front-Relationship** 数据集，同时为 Relation VAE 的重构训练与 Stage 3 关系潜变量扩散提供了监督信号。

## 5. 双向 Relation VAE：从二次关系矩阵到线性节点隐变量

Relation VAE 负责将稀疏有向关系图压缩为节点对齐的连续潜变量，并支持从潜变量解码还原成对关系类别：

```text
sparse directed relation graph
  -> node-aligned latent vectors
  -> reconstruct relation categories
```

### 5.1 有向关系图与特征输入

在有向图表示中，节点对应家具或建筑元素，有向边 $v_i\to v_j$ 的特征表示为：

$$
(pe_i,pe_j,R_{ij},R^s_{ij}),
$$

其中 $pe_i,pe_j$ 为两端节点的位置编码，$R_{ij}$ 为关系大类（如 Distance、Alignment），$R^s_{ij}$ 为关系子类（如 left、right、attach to，无关系则记为 `None`）。编码家具节点时仅输入类别与位置编码，排除家具的绝对 OBB 坐标，以防止 VAE 退化为坐标记忆器；建筑构件节点则保留完整 OBB 作为物理边界条件。

### 5.2 入边-出边双向交叉注意力（In-Out Cross-Attention）

在有向空间关系中，“其他物体以我为参照”（入边，in-degree）与“我以其他物体为参照”（出边，out-degree）承担不对等的语义角色。Relation VAE 对每个节点 $v_i$ 分别构建入边子图与出边子图，并通过两步交叉注意力顺序聚合上下文：

```text
node feature
  -> cross-attend incoming edge features
  -> cross-attend outgoing edge features
  -> self-attention with other nodes
  -> latent mean and variance
```

分开建模入边与出边使编码器能够准确捕捉有向参照系下的局部结构角色。

### 5.3 训练目标与复杂度压缩

编码器输出每个节点潜变量的高斯分布参数 $q_{\phi}(z_i\mid G)=\mathcal{N}(\mu_i,\sigma_i^2I)$，重参数化采样后由解码器预测成对关系标签。训练损失结合了分类交叉熵与 KL 散度正则项：

$$
\mathcal{L}_{VAE}
=
\mathcal{L}_{CE}
+
\lambda_{KL}\mathcal{L}_{KL},
$$

其中 $\lambda_{KL}$ 设为较小的 `0.001`，优先保证关系重构精度与潜变量表征能力。

通过 Relation VAE，原本需要 $G\in\mathbb{R}^{N\times N\times d_r}$ 表示的显式成对关系矩阵被压缩为线性规模的节点潜变量集合 $Z=\{z_i\}_{i=1}^{N}$（$z_i\in\mathbb{R}^{d_z}$）。Stage 3 扩散模型只需生成 $N$ 个节点向量，在降低生成维度、滤除残余噪声的同时，提供了连续平滑的控制空间（代价是面对极端复杂的高阶多体关系时，固定维度的节点潜变量可能存在细粒度信息瓶颈）。

## 6. 扩散网络架构与联合训练机制

### 6.1 Transformer 去噪主干与损失函数

四个扩散阶段共享相似的 Transformer 主干架构：

```text
node attributes
  -> attribute encoders
  -> token embeddings
  -> self-attention among nodes
  -> cross-attention to floor plan / relation latent
  -> noise prediction head
```

属性编码器分别处理类别、OBB 参数、检索特征与关系潜变量，叠加位置编码后通过自注意力建模节点间交互，并利用交叉注意力注入户型图或关系潜变量条件。户型图从 Stage 1 起便全程参与条件控制。

各阶段对待生成属性 $x_0$ 执行前向高斯加噪 $x_t = \sqrt{\gamma(t)}x_0 + \sqrt{1-\gamma(t)}\epsilon$，并优化噪声预测损失 $\mathcal{L}_{diff} = \mathbb{E}_{\epsilon,t}\left[\lVert\epsilon-\epsilon_{\theta}(X_t,F,t)\rVert_2^2\right]$。四个阶段的扩散目标变量划分如下：

| Stage | 被加噪生成的主要变量 |
| --- | --- |
| Type Diffusion | furniture type vectors |
| Property Diffusion | size 和 feature embedding |
| Relation Latent Diffusion | relation latent vectors |
| Box Layout Diffusion | translation 和 rotation 等 OBB attributes |

### 6.2 建筑构件辅助重建与 VAE-Diffusion 联合训练

为促使去噪网络在各阶段持续感知墙体、门、窗的几何边界，所有扩散阶段均附加了建筑元素 OBB 的辅助重建损失：

$$
\mathcal{L}_{rc}
=
\frac{1}{m}
\sum_{i=1}^{m}
\lVert
B_i-\hat{B}_i
\rVert_2^2.
$$

此外，Relation VAE 与 Stage 4 的 Box Layout Diffusion 采用 `1:1` 的损失权重进行联合训练（co-training）。若 VAE 仅独立优化关系图重构，其潜空间未必契合下游几何扩散模型的去噪动态；联合训练使包围盒布局的扩散梯度反向塑造 VAE 潜空间，确保关系潜变量同时兼顾关系语义重构与下游几何落位指导。

## 7. 多任务复用接口：重排、补全、图控制与多模态引导

得益于四阶段解耦设计，CasLayout 无需为不同应用单独训练模型，只需按需执行、跳过或用外部条件替换特定阶段，即可支持丰富的下游任务：

- **完整生成（Holistic generation）与结构感知生成（Structure-aware generation）**：从建筑构件与户型图出发依次运行全部四个阶段。由于门窗操作空间在每一阶段均受约束，面对复杂非规则户型时能有效避免越界与挡门问题。
- **场景重排（Rearrangement）**：给定已有家具类别与尺寸，直接跳过前两个阶段，仅执行 `Relation Latent Diffusion -> Box Layout Diffusion` 重新规划关系与摆放位置。
- **场景补全（Completion）与交互编辑（Editing）**：给定部分已知布局，通过掩码与占位符固定已知家具属性，仅对缺失部分去噪。在 Stage 3 中，系统利用已知局部关系的重构损失进行梯度引导（gradient guidance），促使新生成的关系潜变量与既有布局协调一致；同理，修改单件家具属性、增删物体或移动门窗位置均可局部重算完成。
- **关系图条件生成（Graph-conditioned generation）**：用户只需提供物体节点与关键的稀疏关系边，系统直接通过 Relation VAE 编码得到潜变量并交由 Stage 4 生成几何布局。
- **LLM / VLM 引导的多模态场景生成（Text/Image-to-Scene）**：由 LLM 从文本提示、或由 VLM 从参考图像中抽取家具列表（替换 Stage 1）与稀疏空间关系图（经 Relation VAE 编码替换 Stage 3），再由 Stage 2 与 Stage 4 扩散模型补全尺寸、检索特征与连续 OBB 坐标。局部坐标系下的方向定义与稀疏图结构大幅减轻了 VLM 的空间关系抽取负担。

## 8. 实验评估与消融分析

### 8.1 实验设置、评价指标与对比基线

实验基于 3D-FRONT 数据集的三大房型展开（卧室 `4,041` 个、餐厅 `900` 个、客厅 `813` 个），所有扩散网络在单张 NVIDIA A6000 GPU 上使用 AdamW 优化器训练。文中同时汇报了标准训练设置下的 **CasLayout** 与引入数据增强的通用模型 **CasLayout-G**。

评估体系涵盖分布统计、类别分布、几何重叠、关系合规率与主观偏好：

| 指标 | 含义 | 注意点 |
| --- | --- | --- |
| FID / KID | 渲染布局分布与真实布局分布的特征距离 | 衡量统计相似性，不直接等同于物理可用性 |
| TKL | 生成与真实家具类别分布的 KL 散度 | 反映家具类别与数量分布的契合度 |
| SCA | 场景分类准确率（越接近 50% 越难区分真假） | 检验生成布局是否贴近真实数据分布 |
| OBB IoU | 预测包围盒与真值或门窗操作空间的交并比 | 需结合具体任务解读（如门窗重叠越低越好） |
| Relation consistency | 输出布局满足给定空间关系的比例 | 直接衡量关系控制精度 |
| User study / Time | 人类偏好率 / 单场景推理耗时 | 评估主观设计质量与采样效率 |

对比基线覆盖自回归模型（[ATISS](https://arxiv.org/abs/2110.03675)）、单阶段扩散模型（[DiffuScene](https://arxiv.org/abs/2303.14207)）、全局到局部 Transformer（GLTScene）、语义图指令生成（[InstructScene](https://openreview.net/forum?id=ELrEzO9DEq)）以及视觉语言规划模型（搭载 GPT-4o / LLaMA-3.1-8B 后端的 [LayoutGPT](https://arxiv.org/abs/2305.15393) 与 LayoutVLM）。

### 8.2 生成质量、可控性与推理速度（Table 1 – Table 7）

**1. 户型图条件生成（Table 1）与无户型图生成（Table 2）**

在有户型图约束（with floor plan）与无户型图约束（without floor plan）两类设置下，CasLayout 与 CasLayout-G 在三类房间上的 FID 均显著优于基线：

| 方法 | 有户型图 Bedroom FID | 有户型图 Livingroom FID | 有户型图 Diningroom FID | 无户型图 Bedroom FID | 无户型图 Livingroom FID | 无户型图 Diningroom FID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LayoutGPT-GPT-4o | 20.42 | 27.88 | N/A | N/A | N/A | N/A |
| ATISS | 18.77 | 20.12 | 21.07 | 20.71 | 25.54 | 24.72 |
| DiffuScene | 18.48 | 24.62 | 25.78 | 19.75 | 23.94 | 24.07 |
| InstructScene | N/A | N/A | N/A | 20.03 | 22.82 | 21.76 |
| GLTScene | 19.80 | 22.01 | 20.94 | N/A | N/A | N/A |
| **CasLayout** | 17.53 | **18.14** | **18.61** | **18.85** | **21.67** | **19.69** |
| **CasLayout-G** | **17.17** | 18.51 | 18.66 | 19.27 | 22.05 | 19.85 |

对比时需区分公平训练条件下的 CasLayout 与使用数据增强的 CasLayout-G。定性结果进一步表明，即使在无户型图约束时，稀疏关系建模依然能促使家具形成清晰的功能分区。

**2. 用户偏好研究（Table 3）与关系一致性（Table 4）**

在 36 名参与者、每人评估 15 组有户型图与 15 组无户型图案列的用户研究中，CasLayout-G 获得了压倒性的偏好率（有户型图 **64.07%** vs ATISS 19.07% / GLTScene 9.44% / DiffuScene 6.67% / LayoutGPT 0.74%；无户型图 **62.41%** vs InstructScene 17.96% / DiffuScene 10.00% / ATISS 9.63%）。

当给定物体列表与稀疏关系图时，Table 4 显示输出布局在各类关系上的合规率（Relation Compliance）普遍达到 `90%` 左右或更高：

| 关系类型 | Bedroom (with floor) | Livingroom (with floor) | Diningroom (with floor) |
| --- | ---: | ---: | ---: |
| Obj-to-Obj Direction | 89.57% | 85.98% | 86.53% |
| Obj-to-Obj Distance | 91.81% | 88.87% | 89.66% |
| Alignment | 93.02% | 90.11% | 91.03% |
| Symmetry | 94.92% | 92.93% | 92.56% |
| Obj-to-Door/Window Distance | 94.80% | 91.67% | 92.08% |

**3. 场景重排、补全（Table 5 & 6）与快速采样推理（Table 7）**

在带户型图的重排与补全任务（Table 5）中，CasLayout 相比 DiffuScene 大幅降低了分布距离与冲突重叠：

| Task | 方法 | FID ↓ | KID ↓ | SCA | IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| Rearrangement | DiffuScene | 28.60 | 7.23 | 72.87 | 1.69 |
| Rearrangement | **CasLayout** | **18.34** | **1.03** | **63.17** | **0.87** |
| Completion | DiffuScene | 23.04 | 5.85 | 76.93 | 0.85 |
| Completion | **CasLayout** | **16.50** | **1.10** | **63.80** | **0.50** |

在推理耗时（Table 7）方面，结合高阶快速采样器可将四阶段扩散耗时压缩至毫秒级：

| Sampler | Steps | Time | FID | KID |
| --- | ---: | ---: | ---: | ---: |
| DDPM | 1000 | 1562.50 ms | 18.51 | 2.38 |
| DPM-Solver++ | 25 | 40.95 ms | 18.72 | 2.56 |
| UniPC | 15 | 25.57 ms | 18.57 | 2.69 |

采用 15 步 UniPC 或 25 步 DPM-Solver++ 即可在保持生成质量的同时，将单场景扩散采样耗时降至 `25–41 ms`（实际系统端到端部署还需计入 CAD 检索与渲染后处理开销）。

### 8.3 核心模块消融分析（Table 8 – Table 11）

1. **级联阶段数、顺序、关系表征与联合训练（Table 8）**：

| 消融变体 | FID ↓ | KID ↓ | SCA | 核心结论 |
| --- | ---: | ---: | ---: | --- |
| I-Stage | 22.26 | 6.23 | 73.31 | 单阶段联合生成负担过重 |
| II-Stage | 21.03 | 4.48 | 74.94 | 拆分类型与后续属性后有所改善 |
| III-Stage（无关系潜变量） | 20.52 | 3.71 | 71.02 | 移除 Stage 3 明显退化，证实关系建模至关重要 |
| Dense-Relation | 20.03 | 4.01 | 69.77 | 全连接关系引入噪声与复杂依赖 |
| Exp-Relation | 19.61 | 3.54 | 70.19 | 显式稀疏矩阵条件不如节点对齐潜变量 |
| No Co-Training | 18.95 | 3.07 | 69.05 | 独立训练 VAE 弱于与 Box Diffusion 联合训练 |
| **CasLayout (Ours)** | **18.51** | **2.38** | **68.88** | 完整四阶段 + 隐式稀疏关系 + 联合训练最优 |

此外，论文对比了 `Object -> Property -> Relation -> OBBs` 与 `Object -> Relation -> Property -> OBBs` 两种级联顺序，证实先生成属性（尺寸）再生成关系效果更佳，因为紧贴、相邻与对齐等关系语义直接依赖物体的几何尺度。

2. **双向 Relation VAE 结构（Table 9）**：在关系重构准确率上，稠密图（Dense Graph）仅为 `91.07%`，单向入边（In-Degree Only）为 `97.38%`，单向出边（Out-Degree Only）为 `97.95%`，入边出边简单混合（In-Out Mixing）为 `98.19%`，而论文提出的两步双向交叉注意力达到 **99.39%**。
3. **建筑构件约束（Table 10）与离散特征码（Table 11）**：Table 10 评估家具与门窗操作空间的交并比（注：原表数值方向与正文叙述存在一定排版歧义，但定性结论明确证实显式引入建筑构件能有效防止家具阻挡门窗开合）。Table 11 对比检索特征表示：连续特征码（FID `18.52` / KID `2.85`）、离散特征码（FID `18.81` / KID `3.47`）与“将离散独热索引视为连续变量交由 DDPM 去噪”（FID **18.51** / KID **2.38**），表明连续松弛去噪再映射回离散码本检索的设计最为稳健。

## 9. 方法谱系、系统局限与工程复现要点

### 9.1 与相关工作的横向对比

CasLayout 处于自回归/扩散生成、场景图控制与大语言模型规划三条研究脉络的交汇点：

| 方法 | 生成范式 | 关系图建模 | 户型与建筑构件 | 语言/图像控制 | CasLayout 的核心差异 |
| --- | --- | --- | --- | --- | --- |
| [ATISS](https://arxiv.org/abs/2110.03675) | 自回归 Transformer | 无显式图 | 非核心约束 | 否 | 沿属性阶段级联生成，不依赖物体排列顺序 |
| [DiffuScene](https://arxiv.org/abs/2303.14207) | 单阶段扩散 | 非核心 | 含户型条件设置 | 否 | 拆分为四阶段扩散，引入关系潜变量与门窗约束 |
| GLTScene | 全局到局部 Transformer | 非核心 | 强调复杂房间边界 | 否 | 显式建模稀疏功能关系与多任务复用接口 |
| [InstructScene](https://openreview.net/forum?id=ELrEzO9DEq) | 指令 + 语义图先验 | 显式关系图 | 不以门窗约束为核心 | 是 | 用功能区稀疏图与节点对齐隐式潜变量替代显式图 |
| [LayoutGPT](https://arxiv.org/abs/2305.15393) | LLM 直接规划 | 文本隐式推理 | 几何边界遵守较弱 | 是 | LLM 仅负责物体与关系规划，扩散模型负责连续几何 |
| LayoutVLM | VLM + 可微优化 | 视觉语言约束 | 依赖后验优化 | 是 | VLM 抽取局部坐标系稀疏关系，交由扩散模型生成 |
| [Graph-to-3D](https://openaccess.thecvf.com/content_ICCV_2021/html/Dhamo_Graph-to-3D_End-to-End_Generation_and_Manipulation_of_3D_Scenes_Using_Scene_ICCV_2021_paper.html) | 场景图到 3D | 显式场景图 | 不聚焦复杂户型 | 可编辑图 | 通过双向 Relation VAE 将稀疏图压缩至线性节点潜空间 |
| **CasLayout** | **四阶段级联扩散** | **稀疏关系潜变量** | **显式墙/门/窗约束** | **支持 LLM/VLM 接入** | **设计工作流分解 + 隐式关系控制 + 多任务阶段复用** |

### 9.2 系统局限与改进方向

1. **级联误差传播与校验机制缺失**：当前管线缺乏跨阶段回溯纠错能力，上游一旦产生错误的家具列表或异常尺寸，下游只能在错误条件上拟合局部最优解。后续工作可在各阶段间引入轻量校验器（verifier，如类别合理性检查、碰撞与门窗净空检测），在校验未通过时触发局部重采样。
2. **成对关系对高阶设计逻辑的覆盖有限**：文中的方位、距离、对齐和对称仍属于成对（pairwise）关系，尚未显式建模成组餐椅围合、会客区视线通廊、动线通道宽度、采光与人机工程学等群组级（group-level）与程序化硬约束。
3. **固定 CAD 库限制资产多样性**：依赖 3D-FUTURE 检索保证了网格拓扑质量，但也限制了长尾或定制化家具风格的生成能力。未来可将 Stage 2 升级为“检索 + 神经形状生成/形变”混合模式。
4. **评估指标与多模态泛化边界**：FID/KID/IoU 难以全面衡量多解室内设计的实际居住可用性；同时，文中的 LLM/VLM 文本与图像引导实验主要作为管线接口可行性展示，面对复杂遮挡图像或整屋多房间联动规划仍需系统性验证。

### 9.3 工程实现核查要点

在构建或复现类似级联布局系统时，核心技术细节集中在六个切面：
- **几何与建筑预处理**：统一房间局部坐标系与 $(\cos\theta,\sin\theta)$ 旋转表示，从 3D-FRONT 准确提取二值户型图及墙、门、窗的零厚度或线段 OBB。
- **稀疏关系抽取阈值**：明确方向判定扇区、距离分级（attach / adjacent / distant）与对齐容差，制定功能区（functional zone）归类表与跨区锚点（anchor）选择规则，并验证关系熵下降幅度。
- **离散形状码本构建**：提取 OpenShape 等 3D 形状特征，训练 VQ-VAE 构建离散码本，并在 Stage 2 中将独热码索引作连续松弛扩散后再执行最近邻 CAD 检索。
- **双向 Relation VAE 与联合训练**：实现入边、出边两步交叉注意力与节点自注意力，设置 $\lambda_{KL}=0.001$，并打通 Stage 4 Box Layout Diffusion 到 VAE 编码器的联合梯度反传路径（权重比 `1:1`）。
- **四阶段 Token 掩码与辅助损失**：妥善处理最大节点数 $N_{\max}$ 的填充与 Stage 1 后的空节点剔除，在全部四个阶段维持建筑构件重建损失 $\mathcal{L}_{rc}$。

## 10. 总结与延伸阅读

### 10.1 快速精读与延伸文献路径

若希望快速掌握论文脉络，建议按以下顺序阅读原论文：
1. 阅读 **Introduction** 与 **Fig. 1 / Fig. 2**，建立四阶段级联扩散与条件传递的全局视图。
2. 精读 **Section 3.3 & 3.6**，掌握功能区稀疏关系图、双向 Relation VAE 以及节点对齐关系潜变量的设计原理。
3. 对照 **Table 1–4** 与 **Table 8–11**，验证各阶段拆解、生成顺序、隐式关系表征与联合训练的实证收益。

进一步探索室内场景生成脉络可参考：[3D-FRONT](https://arxiv.org/abs/2011.09127)（基础数据集）、[ATISS](https://arxiv.org/abs/2110.03675)（自回归生成）、[DiffuScene](https://arxiv.org/abs/2303.14207)（单阶段扩散生成）、[InstructScene](https://openreview.net/forum?id=ELrEzO9DEq) 与 [Graph-to-3D](https://openaccess.thecvf.com/content_ICCV_2021/html/Dhamo_Graph-to-3D_End-to-End_Generation_and_Manipulation_of_3D_Scenes_Using_Scene_ICCV_2021_paper.html)（场景图控制）、[LayoutGPT](https://arxiv.org/abs/2305.15393)（大语言模型布局规划）以及 [DDPM](https://arxiv.org/abs/2006.11239) 与 [VQ-VAE](https://arxiv.org/abs/1711.00937)。

### 10.2 核心结论

CasLayout 构建了一套面向真实室内设计需求的可控级联生成系统：

```text
设计流程分解
+ 建筑元素约束
+ 稀疏关系图
+ Relation VAE latent
+ 四阶段条件扩散
+ LLM/VLM 高层接口
```

通过将高维联合分布根据任务因果顺序拆解为“类别与数量 $\to$ 尺寸与检索特征 $\to$ 节点对齐稀疏关系潜变量 $\to$ 三维包围盒”四个条件扩散阶段，CasLayout 让户型结构、稀疏关系图、局部已知布局以及文本/图像提示都能自然接入同一生成框架，为复杂结构化生成任务提供了清晰的分解范式。

## 参考文献

- Wu et al., 2026. [CasLayout: Cascaded 3D Layout Diffusion for Indoor Scene Synthesis with Implicit Relation Modeling](https://arxiv.org/abs/2604.27361).
- Hugging Face Paper Page. [CasLayout metadata](https://huggingface.co/papers/2604.27361).
- YingruiWoo. [CasLayout official repository](https://github.com/YingruiWoo/CasLayout).
- Fu et al., 2021. [3D-FRONT: 3D Furnished Rooms with Layouts and Semantics](https://arxiv.org/abs/2011.09127).
- Paschalidou et al., 2021. [ATISS: Autoregressive Transformers for Indoor Scene Synthesis](https://arxiv.org/abs/2110.03675).
- Tang et al., 2024. [DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis](https://arxiv.org/abs/2303.14207).
- Lin and Mu, 2024. [InstructScene: Instruction-Driven 3D Indoor Scene Synthesis with Semantic Graph Prior](https://openreview.net/forum?id=ELrEzO9DEq).
- Feng et al., 2023. [LayoutGPT: Compositional Visual Planning and Generation with Large Language Models](https://arxiv.org/abs/2305.15393).
- Dhamo et al., 2021. [Graph-to-3D: End-to-End Generation and Manipulation of 3D Scenes Using Scene Graphs](https://openaccess.thecvf.com/content_ICCV_2021/html/Dhamo_Graph-to-3D_End-to-End_Generation_and_Manipulation_of_3D_Scenes_Using_Scene_ICCV_2021_paper.html).
- Zhai et al., 2024. [CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs](https://arxiv.org/abs/2305.16283).
- Gao et al., 2023. [SceneHGN: Hierarchical Graph Networks for 3D Indoor Scene Generation with Fine-Grained Geometry](https://ieeexplore.ieee.org/document/10045799).
- Sun et al., 2025. [LayoutVLM: Differentiable Optimization of 3D Layout via Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_LayoutVLM_Differentiable_Optimization_of_3D_Layout_via_Vision-Language_Models_CVPR_2025_paper.html).
- Wang et al., 2024. [Chat2Layout: Interactive 3D Furniture Layout with a Multimodal LLM](https://arxiv.org/abs/2407.21333).
- Yang et al., 2024. [Holodeck: Language Guided Generation of 3D Embodied AI Environments](https://arxiv.org/abs/2312.09067).
- Ho et al., 2020. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Lu et al., 2025. [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095).
- Zhao et al., 2023. [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2302.04867).
- van den Oord et al., 2017. [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937).
- Liu et al., 2023. [OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding](https://arxiv.org/abs/2305.10764).

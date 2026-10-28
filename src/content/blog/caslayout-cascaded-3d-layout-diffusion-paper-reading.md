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

试想这样一个室内设计任务：给你一张带两扇门和一扇落地窗的异形客餐厅户型图，要求在里面合理摆放沙发、茶几、电视柜、餐桌和六把餐椅。人类设计师面对这个任务时，脑海中的推演有着严格的先后次序——先看房间面积与门窗位置能容纳哪些家具，再挑定家具的具体尺寸与款式，接着规划“餐椅围绕餐桌”“沙发正对电视柜且避开门扇开合区”的功能关系，最后才把每件家具推到精确的坐标与朝向上。

然而，主流的端到端 3D 场景生成模型却试图在一步之内同时猜出所有答案。无论是单阶段扩散模型（如 [DiffuScene](https://arxiv.org/abs/2303.14207)）还是直接用大语言模型回归三维坐标（如 [LayoutGPT](https://arxiv.org/abs/2305.15393)），往往要把家具数量、类别、长宽高、形状特征、位置与旋转角全部塞进同一个高维联合分布里一步预测。在仅有数千个房间的专业 3D 室内数据集（如 3D-FRONT）上，这种“一步登天”的联合建模极易导致家具穿墙挡门、餐椅悬空散落，或者对局部关系指令置若罔闻。与此同时，传统的场景图生成方法习惯在任意两个物体之间都连一条边，形成 $O(N^2)$ 的全连接稠密关系图——把毫不相干的“卧室台灯”与“玄关鞋柜”强行绑定空间约束，反而让模型淹没在高熵背景噪声里。

由中科院自动化所、北航与微软亚洲研究院团队（Yingrui Wu、Youkang Kong、Mingyang Zhao、Weize Quan、Dong-Ming Yan、Yang Liu）发表的 [CasLayout: Cascaded 3D Layout Diffusion for Indoor Scene Synthesis with Implicit Relation Modeling](https://arxiv.org/abs/2604.27361)（DOI `10.1145/3811347`，官方仓库 [YingruiWoo/CasLayout](https://github.com/YingruiWoo/CasLayout)），正面拆解了 3D 室内布局生成中的维度爆炸与关系冗余难题。针对三维室内布局中的联合分布高维耦合与场景图边冗余问题，CasLayout 提出了四项关键架构设计：

1. **按因果次序拆解高维联合分布**（Causal Cascaded Factorization over Joint Distribution）：放弃直接拟合完整布局的高维联合分布 $p(L\mid A)$，严格遵循人类设计的因果依赖，将其拆解为“家具类别与数量 $\to$ 尺寸与几何检索特征 $\to$ 节点对齐稀疏关系潜变量 $\to$ 三维包围盒坐标与朝向”四个级联条件扩散阶段。未知家具尺寸便无法定义“紧贴”与“对齐”，未知功能关系便无法求解连续落位；先定尺度、再定关系、最后落位，消除了变量间的耦合歧义。
2. **功能分区稀疏关系图滤除二次方背景噪声**（Functional-Zone Sparse Graph over Dense Pairwise Edges）：摒弃 $O(N^2)$ 的全连接场景图，按睡眠区、就餐区、休闲区等功能分区（Functional Zones）保留区内细粒度关系，跨区仅选定单一核心锚点物体（Anchor Item）建立连接。这一稀疏化策略将客厅方位关系的信息熵从稠密图的 `1.62` 大幅压降至 `1.04`，剔除了偶然相对位置带来的干扰。
3. **双向 Relation VAE 与扩散模型联合塑造线性节点潜空间**（Node-Aligned Relation Latent with Co-Training）：即便关系图已稀疏化，显式 $N\times N$ 边矩阵仍充斥大量 `None` 空边。论文设计双向关系变分自编码器（Bidirectional Relation VAE），利用入边-出边两步交叉注意力将二次方有向边压缩为 $O(N)$ 个节点对齐的关系潜变量，并与第四阶段包围盒扩散以 `1:1` 权重联合训练（Co-Training），让下游几何落位梯度反向塑造潜空间结构。
4. **建筑构件全程重建约束与中间特征选择性截断**（Explicit Architectural Constraints & Selective Feature Truncation）：将墙体、门、窗作为显式条件节点并在全部四个阶段施加辅助重建损失 $\mathcal{L}_{rc}$，防止去噪网络遗忘门窗净空边界；同时在第二阶段利用 VQ-VAE 离散码本完成 CAD 模型检索后，主动截断形状检索特征向后续阶段的传递，避免布局网络对特定 CAD 网格几何产生过拟合。

```text
建筑结构 (Floor Plan + Walls / Doors / Windows)
  -> Stage 1: 家具数量和类别 (Furniture Type Diffusion)
  -> Stage 2: 家具尺寸和几何检索特征 (Furniture Property Diffusion + VQ-VAE CAD Retrieval)
  -> Stage 3: 节点对齐稀疏关系潜变量 (Relation Latent Diffusion via Bidirectional Relation VAE)
  -> Stage 4: 最终三维包围盒布局 (Box Layout Diffusion)
```

## 1. 为什么单阶段生成与稠密场景图在室内布局中频频失效？

### 1.1 3D 室内布局生成的七维耦合困境

3D 室内场景合成（3D Indoor Scene Synthesis）的目标是生成可直接用于渲染、碰撞检测、交互编辑乃至机器人具身仿真的结构化三维场景。要让生成的房间既美观又实用，算法必须同时在七个相互制约的维度上做出正确决策：

| 决策维度 | 具体实例 | 耦合难点与物理约束 |
| --- | --- | --- |
| **家具数量** | 卧室放几张床、几个床头柜 | 数量受房间面积、功能定位与门窗开口位置共同制约 |
| **家具类别** | sofa、coffee table、wardrobe、dining chair | 类别组合需符合房间语义（如餐厅必须成套出现餐桌与餐椅） |
| **家具尺寸** | 床、衣柜、桌椅的长宽高 $s\in\mathbb{R}^3$ | 尺寸直接决定通道剩余宽度以及“紧贴”“对齐”的几何边界 |
| **几何检索特征** | 从 3D-FUTURE 库中匹配哪款 CAD 资产 | 影响全屋视觉风格统一性与网格拓扑可用性 |
| **位置与朝向** | 每个物体的带朝向包围盒（OBB） | 严禁穿墙越界、家具互穿或阻挡房门开合扇区 |
| **物体间关系** | nightstand attach to bed、chair front of table | 需满足局部功能成组逻辑与对称美学，支持用户显式指定 |
| **建筑硬约束** | walls、doors、windows、floor plan | 布局必须适应非规则多边形户型，窗前与门后需保留净空 |

早期的室内布局生成模型大多将房间简化为一个空旷的矩形外框，只关心家具不穿出墙面、彼此不发生严重碰撞。但在真实室内设计中，空间逻辑要精密得多：门后必须留出旋转开合扇区，落地窗前不宜摆放高大衣柜遮挡采光，餐椅需要围绕餐桌形成就餐区，床头柜需要成对贴合在床头两侧。

当现有生成范式试图应对上述复杂约束时，普遍撞上了三堵墙：
- **高维联合分布的维度灾难**：如果让单个扩散模型或自回归网络一次性输出所有物体的 $(c, s, feat, t, r)$，模型需要在极高维的混合（离散+连续）空间中寻找合法流形。而 3D-FRONT 等专业室内数据集仅有数千个有效房间样本，远不足以支撑如此庞大的联合组合空间学习。
- **全连接稠密图的噪声淹没**：许多基于场景图的方法（如 Graph-to-3D）默认在每对物体之间建立关系边，构成 $O(N^2)$ 的稠密图。然而，客厅里的吊灯与餐边柜之间并没有直接的设计对齐意图，它们的相对位置只是各自服从房间全局布局后的副产品。强行让神经网络拟合成百上千条偶然的成对边，只会引入剧烈的梯度冲突。
- **LLM/VLM 的连续几何短板**：大语言模型与视觉语言模型擅长高层语义规划（例如根据一句文本列出家具清单，或指出“两把扶手椅面对咖啡桌”），但让它们直接输出小数点后三位的三维坐标与旋转角时，几乎必然出现穿模、悬空或堵门。

### 1.2 从 3D-FRONT 到带朝向三维包围盒

为了将高层语义意图稳健地落地为三维连续几何，CasLayout 建立在以下核心数据与几何表征之上：

- **3D-FRONT、3D-FUTURE 与 3D-Front-Relationship**：系统以包含房间建筑结构与专业家具摆放的 [3D-FRONT](https://arxiv.org/abs/2011.09127) 数据集（涵盖卧室、客厅、餐厅三大主力房型）为基础，从中抽取稀疏空间关系构建了 **3D-Front-Relationship** 数据集；家具的三维网格则通过预测几何特征，从配套的 **3D-FUTURE** 工业 CAD 资产库中检索贴合。
- **户型图**（Floor Plan）与**建筑构件**（Building Elements）：房间的二值户型图（binary image）通过交叉注意力全程注入扩散主干；墙体（wall）、房门（door）与窗户（window）则作为显式条件节点参与自注意力计算，并在每个去噪阶段接受重建监督。
- **带朝向三维包围盒**（OBB, Oriented Bounding Box）：单件家具的空间几何由八维向量表示为
  $$
  B = (s,t,r),
  $$
  其中 $s\in\mathbb{R}^3$ 为长宽高尺寸，$t\in\mathbb{R}^3$ 为相对于房间中心的平移坐标，$r=(\cos\theta,\sin\theta)\in\mathbb{R}^2$ 为绕竖直轴（重力轴）旋转角 $\theta$ 的连续三角函数编码。为什么不直接回归角度标量 $\theta \in [-\pi, \pi)$？因为 $-\pi$ 与 $+\pi$ 在物理空间上完全重合，但在标量数值上却处于两端极值，直接计算均方误差会在周期边界处产生灾难性的梯度跳变；改用 $(\cos\theta, \sin\theta)$ 流形表示则保证了旋转空间的处处连续可导。

### 1.3 四阶段级联分解的概率形式化

既然一步生成完整布局 $L$ 过于困难，CasLayout 将给定建筑条件 $A$ 下的条件分布 $p(L\mid A)$ 按因果次序分解为四阶段马尔可夫链式条件乘积：

$$
p(L\mid A)
\approx
p(C\mid A)
p(P\mid C,A)
p(R_z\mid C,P,A)
p(B\mid C,P,R_z,A),
$$

式中各概率因子的物理含义与因果承接关系如下：
- $p(C\mid A)$（**Stage 1 家具类别扩散**）：在给定建筑条件 $A$（户型图与墙门窗节点）下，决定房间内存在哪些家具及其离散类别 $C$。
- $p(P\mid C,A)$（**Stage 2 家具属性扩散**）：在已知家具清单 $C$ 与建筑条件 $A$ 下，生成每件家具的三维尺寸与几何检索特征 $P$，并据此从 CAD 库检索匹配的三维网格。
- $p(R_z\mid C,P,A)$（**Stage 3 关系潜变量扩散**）：在已知家具类别、真实尺寸与建筑结构后，生成与每个节点对齐的隐式空间关系潜变量 $R_z$。
- $p(B\mid C,P,R_z,A)$（**Stage 4 包围盒布局扩散**）：汇总上游全部条件，求解每件家具的精确平移 $t$ 与朝向 $r$，输出最终三维包围盒集合 $B$。

```text
Input:
  floor plan image + walls / doors / windows
  optional: object list / partial layout / sparse relation graph / text / image

Stage 1: Furniture Type Diffusion
  building elements -> object count + furniture categories (C)

Stage 2: Furniture Property Diffusion
  categories + building elements -> size + VQ-VAE feature index (P)
  feature index -> 3D-FUTURE CAD model retrieval

Stage 3: Relation Latent Diffusion
  categories + sizes + building elements -> node-aligned relation latent (R_z)
  sparse relation graph <-> Bidirectional Relation VAE <-> latent vectors

Stage 4: Box Layout Diffusion
  categories + sizes + relation latent + building elements -> OBBs (B)

Output:
  collision-aware furniture OBB layout + retrieved CAD meshes
```

这种级联分解一举三得：第一，每个扩散模型只需专注于单一性质的目标变量，显著降低了训练样本需求；第二，中间阶段全部暴露为透明的可控接口，用户可以随时在 Stage 1 锁死家具清单，或在 Stage 3 注入自定义关系图；第三，它为 LLM 与 VLM 提供了理想的接驳锚点——让语言模型只负责抽取出家具列表（接管 Stage 1）和稀疏语义关系（经 VAE 编码后接管 Stage 3），而把最棘手的尺度匹配与防碰撞坐标求解留给扩散模型。

---

## 2. 房间几何表示、物体局部坐标系与节点渐进补全

在深入网络架构前，我们先厘清 CasLayout 是如何将复杂的房间结构与家具属性统一编码为 Transformer Token 的。

完整房间布局 $L$ 由建筑元素集合 $A$ 与家具元素集合 $F$ 共同构成。建筑元素包含二值户型图以及 $m$ 个墙、门、窗构件节点：

$$
A=\{b_i\}_{i=1}^{m}, \quad b_i=(c_i,B_i), \quad c_i\in\{\text{wall},\text{door},\text{window}\}.
$$

家具元素由 $n$ 个待生成物体构成：

$$
F=\{f_j\}_{j=1}^{n}, \quad f_j=(c_j, feat_j, B_j), \quad B_j=(s_j,t_j,r_j).
$$

### 2.1 为什么要定义物体局部坐标系？

在描述两个物体的空间方位（如“餐椅在餐桌前方”“床头柜在床的左侧”）时，如果直接使用房间的全局坐标系 $(X_{\text{room}}, Y_{\text{room}})$，就会遇到一个致命问题：一旦床或餐桌在房间里旋转了 $90^\circ$ 或 $180^\circ$，原本属于“床左侧”的床头柜在全局坐标轴下就会变成“后方”或“右侧”，导致空间关系标签随房间摆放朝向剧烈漂移。

为此，CasLayout 为每个家具 OBB 建立了以自身中心为原点的**物体局部坐标系**（Object-Centric Local Coordinate System）：硬性规定物体自身的 $+x$ 轴指向其左侧（left），$+y$ 轴指向其正前方（front），反方向则分别为右侧（right）与后方（behind）。如此一来，“椅子位于书桌正前方（front）”永远意味着椅子中心落在书桌局部坐标系的 $+y$ 轴扇区内，彻底解耦了局部功能组件关系与全局房间朝向。

### 2.2 节点 Token 填充与四阶段因果属性补全

在扩散 Transformer 内部，每个节点（无论建筑构件还是家具）被统一编码为属性 Token：

$$
e_i=(c_i,s_i,t_i,r_i,pe_i),
$$

其中 $pe_i$ 为位置编码（Positional Encoding），用于区分同一房间内的多个同类物体（例如六把完全相同的餐椅）。由于不同房间的家具数量 $n$ 动态变化，训练时系统设定最大节点数 $N_{\max}$，不足部分使用 `None` 类别与零向量做 Padding；当 Stage 1 预测出所有节点的类别后，系统立即滤除被判定为 `None` 的空节点，仅将真实存在的有效家具节点与建筑构件节点送入 Stage 2 至 Stage 4。

沿着四个级联阶段，每个家具 Token 的未知属性像拼图一样被逐步填满：

| 级联阶段 | 节点已知输入条件 | 节点当前待生成变量 | 对下游阶段的因果支撑 |
| --- | --- | --- | --- |
| **Stage 1 之前** | 建筑构件 $A$、二值户型图 | 有效家具数量 $n$、离散类别 $c_j$ | 决定后续所有家具 Token 的存在性与语义身份 |
| **Stage 1 之后** | 类别 $c_j$、有效节点掩码 | 尺寸 $s_j$、检索特征 $feat_j$、关系、位姿 | 为 Stage 2 属性生成提供类别先验 |
| **Stage 2 之后** | 类别 $c_j$、尺寸 $s_j$、检索特征码 | 节点关系潜变量 $z_j$、平移 $t_j$、旋转 $r_j$ | 检索 3D-FUTURE 网格；提供计算“紧贴/对齐”所需的真实尺寸 |
| **Stage 3 之后** | 类别 $c_j$、尺寸 $s_j$、关系潜变量 $z_j$ | 平移 $t_j$、旋转 $r_j$ | 为 Stage 4 提供局部功能区拓扑与门窗避让隐式约束 |
| **Stage 4 之后** | 完整 OBB $(s_j,t_j,r_j)$ 与 CAD 模型 | 场景三维网格组装与渲染 | 输出最终可交互 3D 室内场景 |

为方便对照阅读，下表汇总了全文核心数学符号的定义：

| 符号 | 物理含义 | 维度与编码说明 |
| --- | --- | --- |
| $A$ | 建筑条件集合（Architectural Condition） | 含二值户型图与 $m$ 个墙、门、窗构件节点 $b_i=(c_i,B_i)$ |
| $F$ | 家具节点集合（Furniture Set） | 房间内 $n$ 个待生成家具对象 $f_j=(c_j,feat_j,B_j)$ |
| $c_j$ | 第 $j$ 个节点的离散类别 | 家具类别（bed、sofa 等）或建筑类别（wall、door、window） |
| $s_j$ | 三维包围盒尺寸（Size） | $s_j\in\mathbb{R}^3$，表示长、宽、高 |
| $feat_j$ | 几何形状检索特征（Feature Code） | 经预训练 VQ-VAE 量化后的离散码本索引，用于 CAD 检索 |
| $t_j, r_j$ | 平移与旋转（Translation & Rotation） | $t_j\in\mathbb{R}^3$（相对房间中心），$r_j=(\cos\theta_j,\sin\theta_j)\in\mathbb{R}^2$ |
| $R_{ij}, R^s_{ij}$ | 有向边 $v_i\to v_j$ 的显式关系大类与子类 | 如大类 Distance、子类 attach to；无关系记为 `None` |
| $z_j / r^l_j$ | 节点对齐关系潜变量（Node-Aligned Latent） | Bidirectional Relation VAE 编码输出的连续节点隐向量 |
| $R_z$ | 房间关系潜变量集合 | $R_z=\{r^l_i\}_{i=1}^N$，作为 Stage 3 的生成目标与 Stage 4 的条件 |

---

## 3. 功能区稀疏关系图与双向 Relation VAE

在确定了家具类别与尺寸之后，为什么不能直接去预测每个家具的坐标，或者直接用传统的全连接场景图来引导布局？这正是 CasLayout 最核心的洞见所在。

### 3.1 为什么全连接稠密图会毒害布局生成？

假设一个客厅里有 $N=15$ 件家具与建筑构件，全连接有向图会包含 $15 \times 14 = 210$ 条边。在这 210 条边中，真正具有设计意义的只有“茶几在沙发正前方且紧贴”“电视柜正对沙发”“落地灯在沙发右侧”等十余条局部功能约束；其余近两百条边（例如“餐椅 A 在落地灯左前方且距离遥远”）纯粹是毫无设计关联的巧合。如果强迫模型去学习这些偶然边，不仅计算复杂度高达 $O(N^2)$，更会让这些近乎均匀分布的伪关系冲淡真正关键的局部结构信号。

为了提纯真正符合人类空间认知的设计约束，CasLayout 构建了 **3D-Front-Relationship** 数据集，精细定义了两类关系：

1. **家具间关系**（Furniture-Furniture Relations）：涵盖四大正交维度：

| 关系大类（$R_{ij}$） | 关系子类或几何判定准则（$R^s_{ij}$） | 典型室内设计实例 |
| --- | --- | --- |
| **Direction（方位）** | 在参考物体局部坐标系下划分为 `left`、`right`、`front`、`behind`、`under`、`above` | `nightstand` 位于 `bed` 的 `left` 侧 |
| **Distance（距离）** | 按两 OBB 最小欧氏距离划分为 `attach to`（紧贴）、`adjacent`（相邻）、`distant`（远离） | `dining chair` 与 `dining table` 保持 `attach to` |
| **2D Alignment（对齐）** | 水平投影面上的 `edge-align`（边缘对齐）、`x-center-align`、`y-center-align`（中心轴对齐） | 床头两侧的床头柜呈 `edge-align` |
| **Symmetry（对称）** | 两物体同类别、同检索特征、同尺寸，且在水平面上呈镜像对称 | 主床左右两侧对称摆放的一对床头柜 |

2. **家具-建筑关系**（Furniture-Building Relations）：墙体、门与窗户被建模为三维线段或零厚度 OBB。系统针对每件家具**仅计算其到最近一面墙体的距离**（保证大型家具靠墙摆放且不穿墙），但会**完整计算其到所有门和窗的距离**——因为无论家具摆在何处，都绝不能侵入房门的开合扇区或彻底堵死窗户。

在此基础上，CasLayout 提出了**基于功能分区与锚点的稀疏化算法**（Functional-Zone Sparse Graph）：
- 系统首先根据家具类别将其归入五大功能区（Functional Zones）：休闲区（lounging zone）、就餐区（dining zone）、睡眠区（bedding zone）、照明区（lighting zone）与其他区（others）。
- **区内稠密保留**：在同一个功能区内部，保留物体之间的细粒度关系（如餐桌与所有餐椅、床与床头柜）。
- **跨区锚点单连**：在不同功能区之间，砍掉所有普通成员间的跨区连线，仅为每个功能区选定一个核心**锚点物体**（Anchor Item，如就餐区的餐桌、休闲区的三人位主沙发、睡眠区的双人床），只建立锚点与锚点之间的跨区相对关系。

```text
Dense Graph (O(N^2) edges):
  every object pair has a candidate edge -> drowned in accidental spatial noise

Sparse Functional Graph:
  preserve fine-grained intra-zone edges (e.g., chairs <-> dining table)
  select 1 anchor per functional zone (e.g., main sofa, dining table, bed)
  connect only anchors across zones -> compact edges, high semantic density
```

论文用香农信息熵（Shannon Entropy）严格量化了这一稀疏化操作的收益。设某类关系在 $n$ 个物体对、$m$ 个关系子类上的经验概率分布为 $p(x_{ij})$，则平均关系熵定义为：

$$
H(X)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{j=1}^{m}
p(x_{ij})\log_2 p(x_{ij}).
$$

熵值越高，说明关系分布越接近随机均匀分布（即背景噪声越多）；熵值越低，说明关系分布具有越鲜明的设计规律。实验统计表明，在结构最复杂的客厅（Living Room）方位关系上，功能区稀疏图的平均熵从全连接稠密图的 **`1.62` 骤降至 `1.04`**。这意味着大量随机噪声边被精准剥离，留下的每一条边都承载着高密度的功能布局意图。

### 3.2 双向 Relation VAE：如何将二次方有向边压缩为线性节点潜变量？

拿到了高信息密度的稀疏有向关系图后，紧接着出现第二个架构难题：**如果直接在 Stage 3 中对 $N\times N$ 的稀疏关系矩阵做扩散生成，矩阵里绝大多数位置依然是代表“无直接关系”的 `None` 空边。** 对一个极度稀疏的离散二次方矩阵做高斯加噪与去噪，不仅计算开销随物体数平方增长，而且极易在解码时产生自相矛盾的环路关系（例如预测出 $A$ 在 $B$ 左边、$B$ 在 $C$ 左边，同时又预测出 $A$ 紧贴 $C$ 右边）。

为了彻底摆脱 $O(N^2)$ 矩阵扩散的泥潭，CasLayout 设计了**双向关系变分自编码器**（Bidirectional Relation VAE），将整张稀疏有向图压缩为与每个节点一一对应的连续潜变量向量 $Z=\{z_i\}_{i=1}^N$：

```text
Sparse Directed Relation Graph G (O(N^2) edges)
  -> NodeFeature + Incoming Edge Cross-Attention + Outgoing Edge Cross-Attention
  -> Node-Aligned Latent Vectors Z = {z_1, ..., z_N} (O(N) vectors)
  -> Pairwise Relation Classifier Decoder
```

这一模块的精妙之处体现在三个设计细节上：

1. **防作弊的输入特征屏蔽**：在有向关系图 $G$ 中，每条有向边 $v_i\to v_j$ 的特征编码为四元组
   $$
   (pe_i,pe_j,R_{ij},R^s_{ij}),
   $$
   即两端节点的位置编码、关系大类与关系子类。而在编码节点自身特征时，**家具节点仅输入离散类别 $c_i$ 与位置编码 $pe_i$，严禁输入家具的真实 OBB 坐标 $(t_i, r_i)$**；唯有建筑构件节点保留其完整 OBB 作为物理参照系。为什么要屏蔽家具坐标？因为一旦让 Relation VAE 看到家具的真实三维坐标，编码器就会走捷径退化为一个简单的“坐标记忆器”，从而丧失将纯拓扑关系映射为通用潜变量的能力。

2. **入边-出边两步双向交叉注意力**（In-Out Cross-Attention）：在有向空间关系中，“别人以我为参照物”（入边，Incoming Edges，例如多把餐椅都指向餐桌）与“我以别人为参照物”（出边，Outgoing Edges，例如餐椅指向餐桌）在几何层级上扮演着截然不同的角色——入边密集的节点往往是功能区的核心锚点，出边则定义了附属物体的依附关系。如果把入边和出边混为一谈，节点就会分不清自己是“支撑锚点”还是“附属挂件”。因此，Relation VAE 的编码器对每个节点 $v_i$ 分别收集其入边子图与出边子图，严格按顺序执行两步交叉注意力与一步全局自注意力：
   - 第一步：以节点特征为 Query，对所有**入边特征**做交叉注意力，聚合“有哪些物体依附于我”；
   - 第二步：以更新后的节点特征为 Query，对所有**出边特征**做交叉注意力，聚合“我需要依附或避让哪些物体与门窗”；
   - 第三步：在所有节点间执行自注意力交互，输出每个节点潜变量的高斯均值与方差 $q_{\phi}(z_i\mid G)=\mathcal{N}(\mu_i,\sigma_i^2I)$。

   这一设计的效果立竿见影：在关系重构准确率消融实验（Table 9）中，全连接稠密图的重构率仅为 `91.07%`；在稀疏图上，仅用入边为 `97.38%`，仅用出边为 `97.95%`，将入边与出边简单拼接混合（In-Out Mixing）为 `98.19%`，而采用**两步双向交叉注意力**则一举达到 **`99.39%`** 的近乎无损重构。

3. **轻正则化训练目标与线性复杂度降维**：解码器接收任意两个节点的潜变量对 $(z_i, z_j)$，直接分类预测它们之间的有向关系标签。VAE 的训练损失结合了交叉熵重构损失与 KL 散度正则项：
   $$
   \mathcal{L}_{VAE}
   =
   \mathcal{L}_{CE}
   +
   \lambda_{KL}\mathcal{L}_{KL},
   $$
   其中权重 $\lambda_{KL}$ 被刻意设为极小的 `0.001`。较小的 KL 惩罚避免了后验坍缩（Posterior Collapse），优先保证潜变量 $z_i$ 能够高保真地记住每一条稀疏关系约束。通过这一变换，原本需要 $G\in\mathbb{R}^{N\times N\times d_r}$ 表示的成对关系矩阵被压缩为线性规模的节点潜向量集合 $R_z=\{r^l_i\}_{i=1}^N$（$r^l_i\in\mathbb{R}^{d_z}$），让 Stage 3 扩散模型只需在 $O(N)$ 的紧凑连续空间中去噪即可生成全局协调的空间拓扑。

---

## 4. 四阶段级联扩散与 VAE-Diffusion 联合训练机制

有了节点 Token 表示与双向 Relation VAE，CasLayout 的四个生成阶段均采用基于 Transformer 主干的去噪扩散概率模型（DDPM）实现，但在目标变量、条件传递截断以及跨阶段联合训练上做出了针对性设计。

```text
Node Attributes (C / P / R_z / B + Building Elements A)
  -> Attribute Encoders + Positional Encoding pe_i
  -> Self-Attention among Furniture & Building Tokens
  -> Cross-Attention to Floor Plan Image F (and Relation Latents R_z in Stage 4)
  -> Noise Prediction Head + Building Element Reconstruction Head
```

### 4.1 四个扩散阶段的目标变量与特征截断哲学

在任意一个扩散阶段中，设当前阶段待生成的节点目标属性为 $x_0$，前向扩散过程按余弦或线性调度 $\gamma(t)$ 逐步注入标准高斯噪声 $\epsilon\sim\mathcal{N}(0,I)$：

$$
x_t = \sqrt{\gamma(t)}x_0 + \sqrt{1-\gamma(t)}\epsilon.
$$

去噪 Transformer 网络 $\epsilon_{\theta}$ 在户型图条件 $F$ 与上游已知节点属性包裹下，学习预测所加噪声：

$$
\mathcal{L}_{diff}
=
\mathbb{E}_{\epsilon,t}
\left[
\lVert
\epsilon-\epsilon_{\theta}(X_t,F,t)
\rVert_2^2
\right].
$$

四个阶段各自的职责与核心技术巧思如下：

1. **Stage 1：家具类别扩散**（Furniture Type Diffusion）：输入仅包含建筑构件节点与二值户型图，最大家具槽位数设为 $N_{\max}$。模型将每个家具槽位的离散类别独热向量（One-Hot Vector，含 `None` 类）视作连续向量施加高斯噪声：
   $$
   c_{i,t} = \sqrt{\gamma(t)}c_i + \sqrt{1-\gamma(t)}\epsilon.
   $$
   反向去噪结束后，对输出向量取 $\arg\max$ 即可同时确定每个槽位的家具类别并剔除所有预测为 `None` 的空槽位，干净利落地解决了动态物体数量预测问题。

2. **Stage 2：家具属性扩散与选择性特征截断**（Furniture Property Diffusion）：在类别 $C$ 已知的条件下，同时生成每件家具的三维尺寸 $s_i\in\mathbb{R}^3$ 与几何检索特征（Feature Embedding）。为了从 3D-FUTURE 库中检索出高匹配度的三维模型，作者首先提取 3D 资产的 [OpenShape](https://arxiv.org/abs/2305.10764) 形状特征，并训练一个 **VQ-VAE** 将高维连续形状特征量化为离散码本索引（Discrete Feature Index）。在扩散训练时，同样将该离散索引的独热向量做连续松弛并由 DDPM 去噪，推理后再映射回最近的离散码本执行 CAD 检索。实验（Table 11）证明，这种“离散码本 + 连续松弛 DDPM 去噪”的策略（FID `18.51` / KID `2.38`）显著优于直接扩散连续特征（FID `18.52` / KID `2.85`）或使用离散扩散模型（FID `18.81` / KID `3.47`）。

   **更关键的一步在于特征截断**（Feature Truncation）：Stage 2 生成了尺寸 $s_i$ 与形状检索特征 $feat_i$，但哪一个应该传给下游的 Stage 3（关系扩散）与 Stage 4（包围盒扩散）？作者在实验中发现了一个反直觉现象——**如果把形状检索特征 $feat_i$ 继续作为条件喂给 Stage 3 和 Stage 4，生成的布局质量反而会出现退化。** 原因在于：具体的 CAD 网格细节（比如沙发扶手是雕花木质还是现代布艺）与它在房间里的宏观几何落位并无本质关联；强行把细粒度形状特征传入下游，会导致关系与位置扩散网络对训练集中特定 CAD 资产的偶然摆放位置产生严重过拟合。因此，CasLayout 做出了一项极其克制的工程裁剪：**形状检索特征 $feat_i$ 仅在 Stage 2 内部用于检索 3D 网格，随后立即截断丢弃；唯有决定物理占位的三维尺寸 $s_i$ 与类别 $c_i$ 被允许向下游传递。**

| 节点属性信息 | 核心物理与视觉用途 | 是否作为条件传入 Stage 3 & Stage 4 | 设计权衡原因 |
| --- | --- | :---: | --- |
| **家具类别 $c_i$** | 决定功能分区归属与成组语义 | **是** | 类别是空间关系与落位的核心语义锚点 |
| **三维尺寸 $s_i$** | 决定物理占地面积、贴合间距与防碰撞边界 | **是** | 不知道真实长宽高就无法计算紧贴、对齐与避让 |
| **检索特征 $feat_i$** | 从 3D-FUTURE 检索匹配的 CAD 网格资产 | **否（截断）** | 防止下游几何扩散网络对特定 CAD 外观细节过拟合 |
| **建筑构件 $A$** | 约束墙体边界与门窗开合净空 | **是（全程）** | 硬性物理边界必须贯穿全链路 |

3. **Stage 3**（关系潜变量扩散）与 **Stage 4**（包围盒布局扩散）：Stage 3 在 $(C, s, A)$ 条件下生成每个节点的隐式关系向量 $R_z=\{r^l_i\}_{i=1}^N$。进入 Stage 4 后，去噪 Transformer 专门开辟了**两条独立的交叉注意力通道**：一条以节点 Token 为 Query 关注关系潜变量 $R_z$，确保生成的坐标满足“餐椅围桌”“床头柜对称贴床”等功能组织；另一条关注二值户型图的视觉特征 $F$，确保生成的平移 $t_i$ 与旋转 $r_i$ 严格落在异形房间轮廓内部。

### 4.2 建筑构件辅助重建损失与 VAE-Diffusion 联合训练

为了让四个阶段的 Transformer 始终对墙体、房门和窗户的位置保持高度敏感，避免在多步注意力传递中“遗忘”建筑边界，CasLayout 在全部四个扩散阶段中都为 $m$ 个建筑构件节点附加了**辅助包围盒重建损失**（Auxiliary Building Element Reconstruction Loss）：

$$
\mathcal{L}_{rc}
=
\frac{1}{m}
\sum_{i=1}^{m}
\lVert
B_i-\hat{B}_i
\rVert_2^2,
$$

其中 $B_i$ 为真实墙门窗的 OBB 参数，$\hat{B}_i$ 为去噪网络对应建筑 Token 输出的重建预测值。这一轻量级辅助任务迫使网络特征层时刻保留精确的门窗空间坐标。

与此同时，论文还解决了一个潜空间对齐问题：如果先单独把双向 Relation VAE 训练完毕并冻结，再用它的潜变量去训练 Stage 4 的 Box Layout Diffusion，那么 VAE 只关心“潜变量能不能还原出离散关系标签”，却丝毫不关心“这个潜变量对下游连续坐标去噪友不友好”。为此，CasLayout 将 **Relation VAE 与 Stage 4 的 Box Layout Diffusion 以 `1:1` 的损失权重进行端到端联合训练**（Co-Training）。在联合训练期间，Stage 4 预测家具坐标的扩散误差梯度会直接穿过关系交叉注意力层，反向传播回 Relation VAE 的编码器。这种双向塑形机制让节点潜变量 $z_i$ 既能以 `99.39%` 的精度编码离散拓扑关系，又天然具备引导连续几何去噪的平滑流形结构（消融实验证实，移除联合训练会导致 FID 从 `18.51` 恶化至 `18.95`，KID 从 `2.38` 退化至 `3.07`）。

---

## 5. 模块化阶段复用：一套权重驱动六类下游任务

得益于四阶段因果解耦架构，CasLayout 无需针对不同应用场景重新设计网络或微调权重，仅靠**阶段跳过**（Stage Skipping）、**中间替换**（Condition Injection）与**掩码梯度引导**（Masking & Gradient Guidance），就能用同一套模型无缝支撑六大核心任务：

- **全流程结构感知生成**（Holistic & Structure-Aware Generation）：给定户型图与墙门窗节点，从 Stage 1 到 Stage 4 顺序执行。由于门窗构件在每一阶段均受 $\mathcal{L}_{rc}$ 重建约束且在 Stage 3 显式编码了家具-门窗距离关系，面对带走廊、斜角或密集门窗的复杂户型时，能有效杜绝家具堵门与穿墙。
- **场景重排**（Scene Rearrangement）：当用户已经拥有一套选定的家具资产（即类别 $C$ 与尺寸 $s$ 已知），希望换一种摆放方案时，系统直接跳过 Stage 1 与 Stage 2，仅运行 `Stage 3 (Relation Latent Diffusion) -> Stage 4 (Box Layout Diffusion)`，在毫秒级时间内重新规划功能分区与几何落位。
- **场景补全与交互式局部编辑**（Completion & Interactive Editing）：当房间内已有部分固定家具（如已放好双人床和衣柜，要求自动补齐床头柜、书桌与椅子）时，系统通过掩码锁定已知家具的类别、尺寸与位姿，仅对未知槽位去噪。特别地，在 Stage 3 生成关系潜变量时，系统利用已知局部关系的分类重构损失计算**梯度引导**（Gradient Guidance），在去噪采样每一步将新生成物体的关系潜变量拉向与既有家具协调的方向；同理，移动单扇门窗或增删单件家具也只需局部重算对应阶段。
- **显式关系图条件生成**（Graph-Conditioned Generation）：当设计师直接给定家具清单与关键的稀疏关系边（如指定某把椅子必须正对书桌）时，系统无需运行 Stage 3 扩散，直接调用训练好的双向 Relation VAE 编码器将用户给定的稀疏图映射为潜变量 $R_z$，交由 Stage 4 求解满足关系的精确坐标。
- **LLM / VLM 驱动的多模态场景生成**（Text / Image-to-Scene）：面对一段自由文本描述或一张参考室内照片，系统调用大语言模型（LLM）或视觉语言模型（VLM）从中抽取出家具类别清单（直接替换 Stage 1 输出）以及物体局部坐标系下的稀疏关系图（经 Relation VAE 编码后替换 Stage 3 输出），随后由 Stage 2 补全合理尺寸并检索 CAD 模型，由 Stage 4 生成无碰撞的三维包围盒。由于只需输出稀疏的功能区内关系与局部方向（left/front），VLM 的空间推理负担被大幅减轻。

当然，任何级联系统都必须直面**上游误差传播**（Error Propagation）的工程挑战：一旦前置阶段产生异常条件，后续阶段只能在错误的前提上寻找局部最优。下表梳理了四个阶段的典型故障边界及工程上可挂载的校验修复机制（Verifier）：

| 级联阶段 | 典型长尾故障模式 | 最终三维场景表现 | 工程可挂载的轻量校验与自愈策略 |
| --- | --- | --- | --- |
| **Stage 1 (Type)** | 漏生核心主家具或重复生成过多大件 | 卧室没有床，或小客厅塞进三张三人沙发 | 引入房型必选清单校验器（Room-Type Checklist Verifier），不合规时触发重采样 |
| **Stage 2 (Property)** | 预测尺寸与房间面积失调，或检索资产风格割裂 | 超大衣柜塞不下墙角，现代沙发配古典雕花茶几 | 施加长宽高比例先验截断，并在同功能区内引入风格嵌入一致性重排序 |
| **Stage 3 (Relation)** | 关系潜变量解码出相互冲突的成对约束 | 餐椅未能成组围合餐桌，床头柜跑到床尾 | 利用 Relation VAE 解码器做自洽性筛查，结合已知关系梯度引导修正潜变量 |
| **Stage 4 (Box)** | 极限拥挤户型下发生轻微擦碰或侵入扇区边缘 | 两个包围盒边角局部重叠，或离门扇过近 | 在扩散采样末期叠加包围盒碰撞排斥梯度（Collision Guidance）或门窗净空后处理微调 |

---

## 6. 实验结果与消融洞察

### 6.1 实验设置与多维评测体系

实验在 3D-FRONT 数据集的三大房型上进行：包含 `4,041` 个卧室（Bedroom）、`900` 个餐厅（Dining room）与 `813` 个客厅（Living room）。所有扩散网络均在单张 NVIDIA A6000 GPU 上使用 AdamW 优化器完成训练。论文同时汇报了两个版本的模型：在与基线完全相同的标准训练集上训练的 **CasLayout**，以及引入旋转/镜像数据增强后跨房型联合训练的通用模型 **CasLayout-G**。

对比基线覆盖了室内场景生成的五大主流技术路线：自回归 Transformer 模型 [ATISS](https://arxiv.org/abs/2110.03675)、单阶段扩散模型 [DiffuScene](https://arxiv.org/abs/2303.14207)、全局到局部 Transformer 模型 GLTScene、基于语义图先验的指令生成模型 [InstructScene](https://openreview.net/forum?id=ELrEzO9DEq)，以及搭载 GPT-4o 或 LLaMA-3.1-8B 后端的视觉语言规划模型 [LayoutGPT](https://arxiv.org/abs/2305.15393) 与 [LayoutVLM](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_LayoutVLM_Differentiable_Optimization_of_3D_Layout_via_Vision-Language_Models_CVPR_2025_paper.html)。评估指标横跨俯视渲染分布距离（**FID / KID**，越低越好）、场景分类准确率（**SCA**，越接近 `50%` 说明判别器越难区分生成场景与真实场景）、家具类别分布 KL 散度（**TKL**）、包围盒重叠交并比（**OBB IoU**）、空间关系合规率（**Relation Compliance**）以及单场景采样耗时。

### 6.2 生成质量、关系可控性与毫秒级快速采样

**1. 户型图约束与无户型图约束下的全面领先**

在有户型图约束（with floor plan）与无户型图约束（without floor plan）两类核心基准下，CasLayout 与 CasLayout-G 在全部三类房型上均取得了最优的 FID 表现：

| 对比方法 | 有户型图 Bedroom FID ↓ | 有户型图 Livingroom FID ↓ | 有户型图 Diningroom FID ↓ | 无户型图 Bedroom FID ↓ | 无户型图 Livingroom FID ↓ | 无户型图 Diningroom FID ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LayoutGPT (GPT-4o) | 20.42 | 27.88 | N/A | N/A | N/A | N/A |
| ATISS | 18.77 | 20.12 | 21.07 | 20.71 | 25.54 | 24.72 |
| DiffuScene | 18.48 | 24.62 | 25.78 | 19.75 | 23.94 | 24.07 |
| InstructScene | N/A | N/A | N/A | 20.03 | 22.82 | 21.76 |
| GLTScene | 19.80 | 22.01 | 20.94 | N/A | N/A | N/A |
| **CasLayout** | 17.53 | **18.14** | **18.61** | **18.85** | **21.67** | **19.69** |
| **CasLayout-G** | **17.17** | 18.51 | 18.66 | 19.27 | 22.05 | 19.85 |

仔细审视上表可以发现一个规律：在家具数量较少、结构相对固定的卧室（Bedroom）中，单阶段扩散模型 DiffuScene（`18.48`）与 CasLayout（`17.53`）的差距尚在 1 个点左右；但一旦进入面积更大、包含多个功能区且家具数量激增的客厅（Livingroom）与餐厅（Diningroom），DiffuScene 的 FID 迅速恶化至 `24.62` 与 `25.78`，而 CasLayout 依然稳健地保持在 **`18.14`** 与 **`18.61`**。这有力印证了前文的核心论断：场景越复杂、物体越多，单阶段联合分布建模越容易崩溃，而四阶段级联分解与功能区稀疏关系潜变量的优势就越显著。

在邀请 36 名受试者（每人盲评 15 组有户型图与 15 组无户型图场景）的主观偏好研究中，CasLayout-G 同样赢得了压倒性的用户投票：有户型图条件下获得 **`64.07%`** 的首选率（远超 ATISS 的 `19.07%`、GLTScene 的 `9.44%`、DiffuScene 的 `6.67%` 与 LayoutGPT 的 `0.74%`）；无户型图条件下获得 **`62.41%`** 的首选率（大幅领先 InstructScene 的 `17.96%` 与 DiffuScene 的 `10.00%`）。

**2. 高达 90% 以上的显式空间关系合规率**

当给定家具列表与稀疏关系图时，生成的三维场景到底有没有严格听从关系指令？下表统计了输出布局在五大关系维度上的合规率（Relation Compliance）：

| 关系约束类型 | Bedroom 合规率 ↑ | Livingroom 合规率 ↑ | Diningroom 合规率 ↑ |
| --- | ---: | ---: | ---: |
| **物体间方位（Obj-to-Obj Direction）** | 89.57% | 85.98% | 86.53% |
| **物体间距离（Obj-to-Obj Distance）** | 91.81% | 88.87% | 89.66% |
| **二维对齐（2D Alignment）** | 93.02% | 90.11% | 91.03% |
| **镜像对称（Symmetry）** | 94.92% | 92.93% | 92.56% |
| **门窗避让距离（Obj-to-Door/Window Distance）** | 94.80% | 91.67% | 92.08% |

可以看到，无论是成对床头柜的镜像对称（逾 `92.5%`），还是家具对门窗净空区域的避让（逾 `91.6%`），被压缩进线性节点潜空间的稀疏关系都在 Stage 4 去噪过程中得到了忠实还原。

**3. 场景重排、局部补全与毫秒级快速采样**

在带户型图的场景重排（Rearrangement）与局部补全（Completion）任务上，CasLayout 相比单阶段扩散基线 DiffuScene 实现了质的飞跃——不仅将重排 FID 从 `28.60` 大幅压降至 **`18.34`**、KID 从 `7.23` 压降至 **`1.03`**，还将反映伪造痕迹的 SCA 从 `72.87%` 拉回至更接近真实分布的 **`63.17%`**，同时将包围盒重叠率 IoU 削减了近一半：

| 下游任务 | 方法 | FID ↓ | KID ↓ | SCA (越近 50% 越好) | OBB IoU ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| **场景重排（Rearrangement）** | DiffuScene | 28.60 | 7.23 | 72.87 | 1.69 |
| **场景重排（Rearrangement）** | **CasLayout** | **18.34** | **1.03** | **63.17** | **0.87** |
| **场景补全（Completion）** | DiffuScene | 23.04 | 5.85 | 76.93 | 0.85 |
| **场景补全（Completion）** | **CasLayout** | **16.50** | **1.10** | **63.80** | **0.50** |

有人可能会担心：把一个扩散模型拆成四个串联阶段，推理速度会不会变慢四倍？由于每个阶段操作的是紧凑的一维节点 Token 序列（而非高分辨率二维特征图），配合现代高阶常微分方程快速采样器（如 [DPM-Solver++](https://arxiv.org/abs/2211.01095) 或 [UniPC](https://arxiv.org/abs/2302.04867)），四阶段扩散的总采样耗时可以压缩到惊人的 **25–41 毫秒**：

| 扩散采样器（Sampler） | 采样步数（Steps） | 单场景扩散耗时 | FID ↓ | KID ↓ |
| --- | ---: | ---: | ---: | ---: |
| DDPM（标准采样） | 1000 | 1562.50 ms | 18.51 | 2.38 |
| **DPM-Solver++** | **25** | **40.95 ms** | 18.72 | 2.56 |
| **UniPC** | **15** | **25.57 ms** | 18.57 | 2.69 |

仅需 15 步 UniPC 采样，单场景生成耗时即可从 `1.56 秒` 降至 `25.57 毫秒`，而 FID 仅浮动 `0.06`，完全满足设计师在交互界面中实时拖拽门窗、即时预览新布局的流畅度要求。

### 6.3 关键消融实验：为什么每一环都不可或缺？

为了验证四阶段级联、生成次序、稀疏图与隐式潜变量的必要性，作者在客厅数据集上进行了详尽的控制变量消融：

| 消融变体 | FID ↓ | KID ↓ | SCA | 消融背后的核心机理洞察 |
| --- | ---: | ---: | ---: | --- |
| **I-Stage（单阶段联合生成）** | 22.26 | 6.23 | 73.31 | 将所有属性塞入同一阶段扩散，模型难以兼顾离散类别与连续几何 |
| **II-Stage（两阶段：类别 $\to$ 其余全部）** | 21.03 | 4.48 | 74.94 | 先确定家具存在性缓解了部分冲突，但尺寸、关系与坐标仍耦合过重 |
| **III-Stage（三阶段：移除关系潜变量 Stage 3）** | 20.52 | 3.71 | 71.02 | 跳过显式/隐式关系建模直接预测坐标，功能区成组结构明显退化 |
| **Dense-Relation（四阶段，但用全连接稠密图）** | 20.03 | 4.01 | 69.77 | $O(N^2)$ 偶然关系边引入高熵噪声，KID 甚至比不用关系的 III-Stage 更差 |
| **Exp-Relation（四阶段，直接扩散显式稀疏矩阵）** | 19.61 | 3.54 | 70.19 | 显式 $N\times N$ 稀疏矩阵中大量 `None` 边阻碍扩散收敛，弱于节点对齐潜变量 |
| **No Co-Training（独立训练 VAE 与 Stage 4）** | 18.95 | 3.07 | 69.05 | 冻结 VAE 潜空间缺乏下游几何落位梯度塑形，不如端到端联合训练 |
| **CasLayout（完整四阶段 + 稀疏潜变量 + 联合训练）** | **18.51** | **2.38** | **68.88** | 因果级联分解 + 功能区稀疏图 + 节点对齐潜变量 + 联合训练达到全局最优 |

这组消融数据揭示了两条极具启发性的结论：
1. **“稠密关系不如没有关系”**：对比 `III-Stage`（无关系，KID `3.71`）与 `Dense-Relation`（全连接稠密关系，KID `4.01`），强行塞入全连接关系图反而让分布距离变差了；唯有切换到功能区稀疏图并压缩为节点潜变量后，KID 才大幅降至 `2.38`。
2. **“尺寸必须先于关系生成”**：论文还专门对比了 `Object -> Property (Size) -> Relation -> OBBs` 与颠倒顺序的 `Object -> Relation -> Property (Size) -> OBBs`。实验证明**先生成尺寸再生成关系显著更优**——因为在物理世界中，一张床是 1.2 米单人床还是 2.0 米大床、一张餐桌是圆桌还是长桌，直接决定了周围能围合几把椅子以及“紧贴/对齐”的几何包络；在不知道物体尺度前去规划空间关系，无异于空中楼阁。

---

## 7. 3D 场景生成与感知重建的技术生态横向对比

把视野拉高到整个三维视觉与图形学社区，我们可以将 CasLayout 与本博客精读过的其他代表性三维场景生成与感知对齐工作放在同一坐标系下，审视各自的技术选型与适用边界：

| 代表工作 | 核心任务与输入条件 | 场景与几何表示范式 | 空间关系与物理约束机制 | 推理耗时与资产来源 | 最佳适用工程场景 |
| --- | --- | --- | --- | --- | --- |
| **CasLayout** | 户型图 / 建筑构件 / 文本 / 局部布局 $\to$ 3D 室内布局 | 家具 OBB $(s,t,r)$ + 墙门窗构件节点 | 功能区稀疏关系图 $\to$ 双向 Relation VAE 线性节点潜变量 + $\mathcal{L}_{rc}$ 建筑重建 | 扩散采样约 **25–41 ms**；从 3D-FUTURE 检索 CAD 网格 | 需严格遵守户型门窗边界、支持毫秒级交互重排/补全与显式关系控制的室内设计系统 |
| **Imaginarium** | 简短文本提示 $\to$ 微调 Flux 2D 引导图 $\to$ 高密度 3D 场景 | 含子空间标注的专业 3D 资产网格 + 预计算体素 | VLM 层级支撑树 / 靠墙 / 成套分组 + 体素模拟退火 + Blender 刚体物理下落 | 单场景约 **240 s**；自带 2,037 个专业精模资产库（单场景均 **31.86** 个物体） | 追求影视/游戏级视觉丰富度、包含大量桌面摆件与书架隔层陈设的高精度内容生产 |
| **Layout2Scene** | 3D 语义布局框 + 文本提示 $\to$ 可漫游 3D 场景渲染 | 前景物体规范空间 **2D Gaussian Splatting** + 背景墙地显式多边形与哈希纹理场 | 3D 布局透视投影 2D 语义图 + Stage 1 几何扩散（GSDS）与 Stage 2 外观扩散（ISD）解耦蒸馏 | 单场景优化约 **1.5 h**，渲染达 **30 FPS**；无需 CAD 库，从零生成神经高斯与纹理 | 已知粗略三维包围盒布局、无需检索现成 CAD 库即可生成 30 FPS 实时漫游与多通道几何缓冲的概念探索 |
| **Diorama** | 单张真实/合成 RGB 图像 $\to$ 零样本开放世界 3D 场景重建 | 开放词汇分割 + Trellis 单图生成 3D 资产 + 分段平面背景 | 局部接地场景图（支撑层级 + 5 类相对成对约束）+ 渲染与几何对齐 + SDFS 碰撞惩罚 | 无需训练（Zero-shot），单场景分钟级；由单图 3D 生成器动态合成资产 | 缺乏预定义 CAD 库、需从单张野外真实照片恢复带支撑层级与可交互物理属性的机器人仿真场景 |
| **SUFLECA** | 单张 RGB 图像 $\to$ 端到端 CAD 检索与 9-DoF 姿态对齐 | 像素对齐三维点云特征 + CAD 模型渲染特征空间 | 跨模态特征预对齐损失 $\mathcal{L}_{\text{align}}$（点云几何编码器向冻结 DINOv2 蒸馏）+ 端到端旋转/平移/缩放头 | 端到端前向推理；从 ScanNet / CAD 库检索 | 已知单视角图像、追求高精度的实例级 CAD 资产检索与 9 自由度空间位姿恢复 |

通过横向对比可以清晰看到各条路线的取舍权衡：
- **先验来源的权衡**：`CasLayout` 直接在 3D 布局空间内训练四个轻量条件扩散模型，因此在遵守户型边界、门窗净空以及毫秒级重排补全上拥有无可比拟的速度与结构严谨性；但它受限于 3D-FRONT 训练分布，主要摆放床、沙发、桌椅等大件主家具，难以像借助 2D Flux 视觉引导的 `Imaginarium` 那样在书架隔层、桌面上自然铺满三十余件精致小摆件。
- **资产检索 vs. 神经生成的权衡**：`CasLayout` 与 `Imaginarium`、`SUFLECA` 均采用“从预定义高质量 CAD 库检索网格”的路线，确保导出的每一个物体都具备干净的工业级三角网格拓扑；而 `Layout2Scene` 与 `Diorama` 则分别引入 2D 高斯泼溅（2DGS）蒸馏与单图 3D 生成器（Trellis），突破了固定 CAD 库的品类限制，代价则是优化耗时增加或单物体背面几何细节受限。
- **CasLayout 自身的改进空间**：首先，四阶段串联管线目前缺乏跨阶段回溯纠错机制，未来可在各阶段间挂载前文讨论的碰撞与门窗净空校验器（Verifier）实现闭环重采样；其次，现有的方位、距离、对齐与对称仍属成对（pairwise）关系，尚未显式建模人体工学动线通道宽度、视线遮挡与自然采光等全局程序化约束；最后，将 Stage 2 的纯 CAD 检索升级为“检索 + 局部神经形变生成”，有望在保持拓扑干净的同时突破 3D-FUTURE 资产库的风格边界。

---

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

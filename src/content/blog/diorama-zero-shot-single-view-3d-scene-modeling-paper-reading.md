---
title: "Diorama 论文精读：零样本单图 3D 场景建模、CAD 检索与语义布局优化"
description: "精读 Diorama 如何通过开放世界感知、PlainRecon、场景图、CAD 检索、9-DoF 位姿估计和分阶段语义布局优化，从单张图像构建可编辑 3D 场景"
pubDate: "2026-07-15T16:20:26+08:00"
updatedDate: "2026-07-15T16:20:26+08:00"
tags:
  - "Paper Reading"
  - "3D Scene Modeling"
  - "Single-view Reconstruction"
  - "CAD Retrieval"
  - "Computer Vision"
  - "Code Reading"
draft: false
---

从一张室内照片恢复 3D 场景，表面上像一个“把 2D 变成 3D”的生成问题，真正落到工程里却包含一串彼此耦合的决定：图中有哪些对象，它们被谁支撑，房间的墙、地面和开口在哪里，应该从哪个资产库选出什么 CAD，CAD 的旋转、平移和尺度是多少，多个对象拼在一起后是否悬空、相撞或穿墙。任何一个模块出错，都会沿管线传播到最终场景。

**Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling** 的选择不是训练一个端到端网络吞掉全部问题，而是反过来：把任务拆成开放词汇感知、建筑平面恢复、场景图推理、多模态资产检索、零样本 9-DoF 位姿估计和语义布局优化，再用现成 foundation models 与几何算法组合出完整场景。最终产物不是一张只能观看的新视角图片，也不是一个无法拆分的神经场，而是一组带类别、CAD 身份、位姿和支撑关系的可编辑对象。

这条路线最值得精读的地方，不是“用了 GPT-4o、DINOv2 或 SAM”这一事实，而是它重新定义了单图场景建模的输出合同：**先接受单图无法唯一决定真实世界，再输出若干语义合理、结构清晰、可编辑的 CAD 场景假设。** CAD 检索牺牲逐像素几何和纹理一致性，换来资产完整性、组合性和后续编辑能力；模块化设计牺牲了端到端联合最优，换来开放类别和组件可替换性。

本文严格以 arXiv [`2411.19492v1`](https://arxiv.org/abs/2411.19492v1) 为论文基线，方法、图表和数字均来自 v1。论文后来更新为 v2，并以 **Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling** 被 ICCV 2025 接收为 Highlight；这些后续状态会单独说明，但不会倒灌到 v1 实验。源码阅读固定在官方仓库 commit [`ec54d826`](https://github.com/3dlg-hcvc/diorama/tree/ec54d826cf6b39c6daf88947da897ff9626e5d61)。该提交晚于论文，代表当前公开实现，不是 v1 的精确历史快照。

先给出全文结论：

> Diorama 的长期价值，是把开放世界视觉基础模型的语义能力接到传统 CAD 场景表示和几何优化上；它的主要风险，则是长管线的误差级联、对外部模型和资产库的依赖，以及“语义合理的可编辑场景”与“精确、物理可靠的真实场景复原”之间仍然很大的距离。

## 1. 论文信息、版本与开放状态

| 项目 | 内容 |
| --- | --- |
| v1 题名 | Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling |
| 作者 | Qirui Wu, Denys Iliash, Daniel Ritchie, Manolis Savva, Angel X. Chang |
| 机构 | Simon Fraser University, Brown University, Alberta Machine Intelligence Institute (Amii) |
| arXiv | `2411.19492v1` |
| v1 提交日期 | 2024-11-29 |
| v1 PDF | 20 pages |
| DOI | `10.48550/arXiv.2411.19492` |
| 论文许可 | arXiv perpetual non-exclusive distribution license |
| 项目页 | [3dlg-hcvc.github.io/diorama](https://3dlg-hcvc.github.io/diorama/) |
| 官方代码 | [3dlg-hcvc/diorama](https://github.com/3dlg-hcvc/diorama)，MIT |
| 本文代码基线 | `ec54d826cf6b39c6daf88947da897ff9626e5d61` |
| 后续状态 | v2；ICCV 2025 Highlight，正式题名增加 `Indoor` |

版本边界需要特别强调。v1 的题名没有 `Indoor`，但系统和实验本来就集中于室内场景。后续正式版把这一适用范围写进题名，项目页和当前 README 也按 ICCV 版本描述。本文在解释 v1 时保留其原始题名、60K OOD 资产集合和 v1 表格；当前代码中的模型版本、阈值和数据路径只放在源码栏目。

论文称系统为 `zero-shot`、`open-world` 和 `training-free`。三者都需要加限定：

- `zero-shot` 指 Diorama 没有针对目标场景类别训练一套端到端 image-to-CAD 系统，不代表每个组件都没受过训练。
- `open-world` 指对象类别和 CAD 检索不被一个小型封闭类别表严格限制，不代表任意稀有对象、室外领域或极端遮挡都可靠。
- `training-free` 指论文主系统不进行端到端场景训练；它仍依赖 GPT-4o、OWLv2、SAM、Metric3DV2、DINOv2、DuoDuoCLIP、GigaPose scale network 和图像修复模型等预训练组件。

这些不是文字游戏，而是复现成本和结论强度的边界。一个不训练 Diorama 参数的用户，仍然要准备多套权重、CAD 数据、预渲染、多模态 embedding 和 GPT API。

## 2. 一句话贡献：把单图复原改写为结构化场景装配

Diorama 的输入是一张 RGB 图像 $\mathbf I$，输出不是单一 mesh，而是近似如下的结构：

```json
{
  "architecture": [
    {"id": "floor", "plane": [0, 1, 0, -1.2], "polygon": "..."},
    {"id": "wall_1", "plane": [0, 0, 1, -3.8], "polygon": "..."}
  ],
  "objects": [
    {
      "id": 7,
      "category": "stack of books",
      "caption": "three books beside the monitor",
      "cad_asset": "asset_id",
      "rotation": "3x3 matrix",
      "translation": [0.3, 1.1, 2.4],
      "scale": [0.24, 0.08, 0.19],
      "supported_by": 4
    }
  ]
}
```

真实实现的数据结构比这个例子复杂，但输出合同大致如此：房间建筑由平面和边界表示，对象由 CAD 与 9-DoF 变换表示，场景图维护对象间的支撑层级。所谓 9-DoF，通常指三维旋转、三维平移和三维尺度；论文的初始 Umeyama 讨论中也出现统一尺度，但系统最终场景对象要处理更一般的尺度与 OBB。

这种表示的优势是明确的：对象可以替换、移动、删除，支持物可以和被支持物一起调整，渲染器或仿真系统能直接消费标准 CAD。代价也同样明确：检索到的 CAD 可能只在语义上相似，轮廓、纹理、材质和局部几何不一定忠实于图片。

## 3. 任务谱系：Diorama 不是什么

理解论文之前，先把几个经常混用的任务分开。

**Single-view 3D reconstruction** 通常追求从单张图像恢复可渲染表面，输出可能是 mesh、NeRF、3D Gaussian 或隐式场。它更关注输入视图和新视角的外观一致性。

**3D scene understanding** 关注对象类别、实例、深度、姿态和关系，可以只输出结构化标签，不一定构造完整可渲染场景。

**CAD-based scene modeling** 使用离散资产库近似图中对象，再估计每个资产的姿态和场景关系。它更像 analysis-by-synthesis：先识别和检索，再装配。

**Scene generation** 可以从文本、布局或随机变量生成一个新场景，不要求对应某张真实输入图片。

**Novel-view synthesis** 只要求从新相机看起来合理，未必提供对象边界、支撑关系或可编辑资产。

Diorama 位于 scene understanding 与 CAD-based scene modeling 的交界：它从真实或合成图片读出场景，再用检索资产构造一个语义相符的 3D 假设。它可以接在 Flux-1 生成图像之后形成 text-to-scene，但 Flux 不是 Diorama 的训练目标，文本到场景也不是其原生输入接口。

## 4. Fig. 1：Teaser 展示了哪三种使用方式

![Diorama teaser：合成图、网络图与文本到场景](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig01-teaser.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 1, arXiv non-exclusive distribution license.*

Fig. 1 展示三类输入：SSDB 风格的合成室内图、互联网真实图片，以及由文本 prompt 经 Flux-1 生成的图片。每个案例都从不同相机渲染输出场景，强调结果不只是输入视角的一张重绘。

图中最重要的信号是**对象完整性与结构可见性**。即使输入只看到桌子或沙发的一部分，输出使用完整 CAD；墙、地面和对象可从新视角查看。它支持“结构化场景”这一主张。

但 teaser 不能证明四件事：

1. 不能证明被遮挡区域恢复的是唯一真实几何；那些部分来自检索资产和系统先验。
2. 不能证明纹理忠实。输出大量对象使用简单或资产自带材质，论文目标本来就不是 photometric reconstruction。
3. 不能证明物理可执行。视觉上接触不等于稳定、可碰撞或适合机器人仿真。
4. 不能证明系统在开放世界中的失败率。teaser 是挑选后的定性展示，真实适用范围要看 SSDB、ScanNet 和附录。

## 5. Fig. 2：两段式系统与多条中间证据

![Diorama 两段式系统管线](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig02-pipeline.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 2, arXiv non-exclusive distribution license.*

Fig. 2 把系统分成橙色的 **Open-world Perception** 和绿色的 **CAD-based Scene Modeling**。

开放世界感知阶段产生：

- 对象类别、2D box 和 instance mask；
- metric depth、normal 和对象点云；
- 房间建筑平面；
- 对象 caption 与 support hierarchy 场景图。

CAD 场景建模阶段消费这些证据：

- 文本类别与对象 crop 用于分层 CAD 检索；
- query crop、CAD 多视图、深度和 patch feature 用于位姿估计；
- 场景图与建筑平面给出布局优化约束；
- 多个候选 CAD 可形成多种合理 arrangement。

这张图也暴露了系统真正的工程性质：它不是一个 `forward(image) -> scene` 的单函数，而是一条带大量可缓存中间产物的 DAG。对象列表错了，后面不会凭空补回；mask 错了，对象点云会污染；depth 错了，尺度和平移会偏；scene graph 错了，优化器会认真地满足错误支撑关系。

因此模块化的优势和风险来自同一件事：**边界清楚使替换组件和定位故障变容易，也使错误在边界之间显式传播。**

## 6. 开放世界场景解析：从图像到对象集合

论文将输入图像记为 $\mathbf I$，识别到的开放词汇类别为 $\mathbf C$，检测框为 $\mathbf B$，实例 mask 为 $\mathbf M$。第 $i$ 个对象写作：

$$
\mathcal O=\left\{o_i=\{c_i,b_i,m_i\}\right\}_{i=1}^{N}.
$$

这里 $c_i$ 是类别或语言描述，$b_i$ 是 box，$m_i$ 是 mask。Metric3DV2 估计 metric depth $\mathbf D$ 与 normal $\mathbf N$，再通过相机内参把像素反投影到点云 $\mathbf P$。对象点云由 mask 选取：

$$
p_i=\left\{\Pi^{-1}(u,v,D_{uv})\mid m_i(u,v)=1\right\},
$$

其中 $\Pi^{-1}$ 表示从像素和深度回到相机坐标。论文没有把这条式子单独列出，但它准确描述了 mask、depth 和 intrinsics 如何共同决定对象 3D 证据。

开放词汇对象解析在论文与当前代码间有轻微演进。v1 主文写 OWLv2 + Segment Anything；当前 `VisionAgent` 支持 OWLv2，并配置 SAM2.1 Hiera Large。当前 `VLMAgent.understand_scene()` 先用 GPT-4o 生成场景类别与描述，再让视觉检测器落到具体实例。这种组合利用 LMM 的语义覆盖和专用 detector/segmenter 的空间精度。

### 6.1 为什么不让 GPT-4o 直接输出所有坐标

LMM 擅长描述“桌上有书、杯子和显示器”，却不稳定地给出精确像素边界、对象计数和遮挡轮廓。Diorama 让 GPT-4o 提供语言类别与关系，再让 OWLv2 和 segmentation model 处理定位，体现了一个实用原则：**让 foundation model 做它擅长的开放语义，让几何或专用模型做精确空间测量。**

### 6.2 检测后的去重

当前源码包含 class-aware NMS 与 multi-instance suppression。论文附录说明：同类别重复 box 通过最大交并相关规则过滤，接着对检测框运行 SAM，保留高分 mask。这里存在两类相反风险：阈值太松会把同一对象重复建模，阈值太紧会合并相邻同类对象。重复对象进入场景图后，还可能制造错误支撑边。

### 6.3 深度不是附属特征

在 Diorama 中，深度同时影响建筑平面、对象 3D 点云、位姿平移和尺度。Table 1 把 GT depth 与 estimated depth 分列，正是为了展示深度误差如何穿过整个系统。estimated depth 下总体 Acc 从相应 GT-depth 条件的 `0.33` 降至 `0.25`；这不是简单的深度 RMSE 变化，而是场景装配质量下降。

## 7. 场景图：把语言关系变成几何约束

![GPT-4o 与 Set-of-Mark 场景图生成](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig10-scene-graph.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 10, arXiv non-exclusive distribution license.*

对象检测完成后，Diorama 用 Set-of-Mark 思路把编号覆盖在对象区域上，再把增强图送给 GPT-4o。输出场景图定义为：

$$
G=\langle V,E\rangle,
$$

其中顶点 $v_i\in V$ 对应对象实例及其 caption，边 $e=\langle v_i,v_j\rangle$ 表示对象 $o_i$ 被 $o_j$ 支撑。附录 prompt 还让模型选择 `placed on`、`mounted on` 等类型，并约束每个被支撑对象只选择一个直接 parent。

场景图不是用于展示的装饰。它决定：

- 哪个 OBB 提供 support surface；
- 被支撑对象应该使用哪个 contact surface；
- 优化次序如何沿支撑层级传播；
- 书架内部对象是否进入 supporting volume；
- floor object 是否需要进一步贴近 wall。

### 7.1 Set-of-Mark 解决了什么

如果只把原图和对象名字交给 GPT-4o，“左边第二个杯子”很容易发生 referring error。编号 mask 把语言引用绑定到视觉区域，模型只需输出 ID。它降低了 grounding 歧义，却不保证关系正确：遮挡、透视和不可见接触仍可能让模型误判“放在桌上”与“挂在墙上”。

### 7.2 当前 `SceneSpec` 的职责

固定提交中的 [`diorama/scenespec.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/scenespec.py) 把场景图、对象列表、建筑元素与后续识别结果放进 `SceneSpec`。`parse_scene_graph()` 把语言模型输出解析为可遍历关系，`add_support_surf()` 等方法再写入几何优化需要的面与方向。

这个设计说明论文中的 $G$ 不是一次性 prompt 输出，而是贯穿管线的中间表示。生产化时应给它 schema version、验证器和 provenance：每条关系来自 GPT、启发式还是人工修改，必须可追踪。

## 8. PlainRecon：为什么要先恢复“空房间”

![PlainRecon 建筑恢复流程](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig03-plainrecon.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 3, arXiv non-exclusive distribution license.*

如果只恢复对象，不恢复墙、地板、天花板和凹凸房间结构，最终场景会缺少最重要的支撑与边界。许多室内布局方法假设 Manhattan room 或固定 cuboid，复杂房间、斜视角和开口会破坏这种简化。PlainRecon 选择把建筑表示为若干有边界的 3D planes。

流程分三步：

1. 检出对象并把对象区域 inpaint 掉，得到尽量完整的“empty room”；
2. 对修复图估计 depth 和 normal，反投影建筑点云；
3. 按 normal 聚类，拟合平面方程并恢复每个平面的二维边界。

先去家具再做平面拟合有直接动机。家具点云会与墙、地面混在一起；如果直接对原图聚类，桌面、柜门和墙面可能共享法线方向，RANSAC 很难知道哪个平面属于建筑。inpainting 不是为了生成漂亮背景，而是为几何估计提供较干净的输入。

## 9. PlainRecon 的每一步与失败面

### 9.1 对象 mask 与 dichotomous segmentation

附录说明作者结合 BiRefNet、MVANet 和 SEEM 等分割方案，目标是尽量覆盖前景对象的完整区域。这里 recall 比精细边界更关键：漏掉一块家具，会把家具纹理和深度当成墙；mask 稍微膨胀，虽然丢失一些墙面像素，却可能由 inpainting 补上。

当前仓库把这一阶段拆为 [`scripts/compute_dichotomous_segmentation.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/scripts/compute_dichotomous_segmentation.py) 与第三方 Inpaint-Anything 流程。README 要求用户配置路径并按阶段运行，说明 PlainRecon 还不是完全封装的一键模块。

### 9.2 Inpainting 产生的是几何提示，不是真实观测

图像修复会虚构被遮挡墙面、门窗或角落。Diorama 依赖通用 monocular depth/normal 模型在修复图上的稳健性，让平滑但可能不真实的纹理仍产生可用大平面。这个策略对墙和地面有合理性，因为建筑面通常低频；对复杂内嵌柜、楼梯或曲面背景就不一定成立。

### 9.3 深度与法线估计

论文比较 Depth Anything V2 与 Metric3DV2。Metric3DV2 的 metric scale 对后续场景尺寸尤其重要。作者观察 normal 对 inpainting 模糊更稳，因此聚类主要依据 normal，而不是只按 depth discontinuity 分段。

### 9.4 Normal clustering 的参数

附录给出一组具体但不应神化的经验参数：

- 先做 voxel downsampling 与 outlier removal；
- 用 12 个 K-means seed normals；
- 迭代吸收与 seed 夹角小于 $10^\circ$ 的点；
- 当剩余未聚类点少于 200 时停止；
- 使用 DBSCAN 分离同法线方向的不同墙面；
- 用 KNN，$k_n=1$，回填未标记点。

这些参数适合论文分辨率和数据，但不等于所有相机、房间和尺度上的最优值。法线噪声、点云密度与图像分辨率变化时，角度和点数阈值都应重新验证。

### 9.5 平面与边界

对每个聚类点集，PlainRecon 用 RANSAC 拟合：

$$
\pi_k:\quad a_kx+b_ky+c_kz+d_k=0.
$$

随后把点投到平面局部坐标，计算 convex hull，并用 rotating calipers 估计边界方向和包围矩形。结果不是无限平面，而是带 polygon bounds 的建筑元素，才能参与碰撞、支撑与渲染。

## 10. Fig. 12 与 Table 4：建筑恢复的证据

![PlainRecon 建筑恢复定性结果](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig12-architecture-results.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 12, arXiv non-exclusive distribution license.*

Fig. 12 把完整房间图与恢复出的彩色建筑平面对照。它展示 PlainRecon 能处理不止一个正交盒子：墙面转折、不同深度房间和开口可以由多块 plane 表示。但图中对象全部移除，无法从该图判断对象与建筑边界接触是否准确。

![SSDB 建筑平面恢复定量结果](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-table04-architecture-reconstruction.webp)

*Source: Wu et al., arXiv:2411.19492v1, Table 4, arXiv non-exclusive distribution license.*

关键数字转录如下：

| Method | Depth | Success | IoU ↑ | PE ↓ | EE ↓ | RMSE ↓ | CDb ↓ | Time ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RaC | DAv2 | 236 | 40.3 (58.8) | 39.8 (12.2) | 23.2 | **0.67** | 0.645 | 43s |
| RaC | M3D | 258 | 45.2 (60.3) | 33.7 (11.6) | 21.0 | 1.23 | 0.908 | 49s |
| PlainRecon | DAv2 | **344** | 47.8 | 13.3 | 27.5 | 1.26 | 0.503 | **23s** |
| PlainRecon | M3D | **344** | **58.6** | **9.6** | **18.9** | 1.37 | **0.447** | 29s |

PlainRecon 的最强信号是可靠性：两种深度下都对 344 张图输出结果，而 RaC 只有 236 或 258 个成功案例。括号中的 RaC 数字只统计成功案例，因此不能直接拿括号 IoU 与 PlainRecon 全集数字比较而忽略失败。

结果也不是单向胜利。RaC + DAv2 的 RMSE `0.67` 明显好于 PlainRecon，说明一个只看单指标的结论会误导。作者强调的是完整输出率、IoU、边界与 CDb 的综合表现。对于后续布局，无法输出建筑比一个中等 RMSE 更致命，这也是 PlainRecon 的工程优势。

## 11. 多模态 CAD 检索：语义正确优先于像素复制

Diorama 的资产池组合了 ShapeNet、ABO、3D-FUTURE、Objaverse 和 HSSD 等来源。不同数据集的分布差异很大：ShapeNet 类别相对经典且形状规整，3D-FUTURE 偏家具，ABO 来自电商资产，Objaverse 类别开放但质量不一，HSSD 面向室内仿真。

论文使用 DuoDuoCLIP 把文本、图像和多视图 3D 资产投到共同 embedding space。检索分两层：

1. 文本查询先找语义类别相近的候选，确保“显示器”不会因为轮廓相似检到一扇窗；
2. 对象 crop 再在语义候选中按视觉 embedding 重排，选择形态更接近的 CAD。

若记文本编码为 $f_t(c_i)$、对象图像编码为 $f_v(I_i)$、CAD 多视图编码为 $f_s(s_j)$，一个简化的 hybrid score 可以写成：

$$
S_{ij}=\lambda_t\,\cos(f_t(c_i),f_s(s_j))+
\lambda_v\,\cos(f_v(I_i),f_s(s_j)).
$$

论文并未用这条简式定义完整实现，但它说明为什么 text-only 与 visual-only 各有缺陷：文本保类别，视觉保形态；hybrid 把两者组合。

### 11.1 为什么检索而不是单图生成 3D 对象

检索资产通常具有封闭表面、完整背面、可用拓扑和可复用材质，适合快速场景装配。单图生成模型可以更贴近输入轮廓，却可能产生破损 mesh、不可编辑神经表示或昂贵后处理。Diorama 选择检索，符合其“紧凑、组合、可交互”的目标。

这也是目标函数的主动收缩：Diorama 不追求物体的精确纹理和细节重建，而追求语义与大体形状匹配。读定性图时，不能用输入沙发颜色和输出 CAD 颜色不一致直接判定方法失败；但如果应用要求品牌级商品复刻，这种不一致就是硬限制。

## 12. Table 3：DuoDuoCLIP hybrid 到底改善了什么

![不同检索表征的 Chamfer Distance](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-table03-retrieval-similarity.webp)

*Source: Wu et al., arXiv:2411.19492v1, Table 3, arXiv non-exclusive distribution license.*

Table 3 按 Household/Furniture、Occluded/Complete、Supported/Supporting 分组，在 SSDB 内部资产与 OOD 资产池上比较 CLIP、OpenShape 和 DuoDuoCLIP。数值是 L1 Chamfer Distance 乘以 $10^3$，越低越好。

DD-H 在多数列最好，例如 Household 的 `5.5/9.9`、Furniture 的 `3.2/7.6`、Supporting 的 `3.0/6.1`。Complete + SS 列的 visual-only DD-V 为 `4.1`，略好于 DD-H 的 `4.4`，说明 hybrid 不是每个子组绝对最优。更重要的是，表中采用 top-5 候选后选择与 GT Chamfer 最小者，这是一种 oracle-style retrieval evaluation：它评价候选集合是否包含几何近似项，不等于线上系统总能知道哪个候选与不可见 GT 最接近。

OOD 列普遍比 SS 列更难，符合跨数据集资产的 scale、朝向、网格和类别噪声差异。OpenShape 在部分 OOD 列出现很大 CD，作者据此认为多视图表征更适合 open-world retrieval；但单张表不足以判断所有开放 3D embedding 模型的通用排名。

## 13. 零样本 9-DoF 位姿：从 patch 对应到 3D 变换

![Diorama 零样本对象位姿估计](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig04-zero-shot-pose.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 4, arXiv non-exclusive distribution license.*

检索到 CAD 只是拿到形状，仍需决定它在房间中的朝向、位置和尺寸。Diorama 预先为每个 CAD 渲染 $T$ 个视图和对应 depth，把 query object crop 与 reference views 都编码为 DINOv2 patch features。

管线分为四层：

1. 在每个 reference view 上建立 query patch 到 reference patch 的语义对应；
2. 用循环一致性和特征相似度选最佳 CAD view，得到粗旋转；
3. 利用 query depth 与 CAD render depth 把 2D 对应提升到 3D；
4. 用 Umeyama + RANSAC 求旋转和平移，并用 GigaPose 小网络增强尺度稳定性。

它面对的难题比 instance-level pose 更强：query 中的椅子和检索 CAD 不是同一个实例，局部几何可能不同。DINOv2 patch 对应依赖“语义部件在同类对象上大致同构”，例如椅背对椅背、桌腿对桌腿，而不是追求严格像素复制。

## 14. 循环对应与最佳视角

设 query patch feature 为 $f_q^i$，reference view 的 patch feature 为 $f_r^j$。对 query patch $i$，先在 reference mask 内找最相似位置：

$$
j=\arg\max_{k:m_r^k>0}S(f_q^i,f_r^k),
$$

再从 $j$ 反向回 query：

$$
i'=\arg\max_{w:m_q^w>0}S(f_q^w,f_r^j).
$$

循环距离可写为：

$$
D_i=-\lVert i-i'\rVert_2.
$$

循环一致的点更可能是稳定语义对应。过滤后，对一个 reference view 的平均对应相似度为：

$$
\operatorname{sim}(I_q,I_r)=
\frac{1}{|\mathcal N|}\sum_{(i,j)\in\mathcal N}
S(f_q^i,f_r^j).
$$

系统选最高分视角作为粗 pose hypothesis，再取 top-K correspondences。v1 附录写 fine-tuned ViT-L、14×14 patch，渲染 180 个 224×224 灰度视图。当前配置同样设置 `n_ref: 180`、`patch_size: 14`、`num_correspondences: 50`，但 similarity threshold 为 `0.5`；v1 附录的对应过滤阈值是 `0.7`。这类差异说明当前仓库不能直接作为 v1 参数表。

### 14.1 最佳视角不是完整旋转真值

离散多视图提供粗旋转，精度受视角采样密度限制。物体对称时，多个视角可能等价；遮挡严重时，query 只包含局部。后续 3D correspondence 和布局优化需要继续修正。把最佳 render index 直接当最终 9-DoF pose，会退化为 Table 1 的 `Best-matching multiview` 基线。

## 15. 3D 对应、Umeyama 与 RANSAC

query patch 通过 estimated depth 得到 3D 点 $q_k$，reference patch 通过 CAD render depth 得到 $p_k$。理想的相似变换满足：

$$
q_k\approx sRp_k+t.
$$

Umeyama 求解：

$$
(R^*,t^*,s^*)=
\arg\min_{R,t,s}
\sum_{k=1}^{K}\left\lVert q_k-(sRp_k+t)\right\rVert_2^2,
$$

其中 $R\in SO(3)$。真实 correspondences 含有误匹配，RANSAC 反复采样最小集合、估计变换并计算内点，降低 outlier 影响。固定源码中 `GigaZSP.solve_umeyama_ransac()` 对应这条路径。

### 15.1 为什么尺度单独困难

单图 metric depth 误差、对象截断和遮挡都会影响点云外接尺度。若只用 Umeyama，少量错误 3D 对应可能把整个 CAD 放大数倍或压缩成很小。论文采用 GigaPose 的小 scale prediction network，利用 query/reference 表征估计更稳的相对尺度，再与几何求解结合。

这个做法再次说明 `training-free` 的边界：Diorama 自身不训练端到端模型，但尺度模块是训练过的。当前代码中 [`diorama/model/gigazsp/ist_net.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/model/gigazsp/ist_net.py) 包含对应的 `ISTNet`/regressor，权重由项目另行提供。

### 15.2 RANSAC 的系统代价

每个对象可能有多个检索 CAD，每个 CAD 有 180 个 render views，每个视角要比较 patch features，随后还要 RANSAC。论文在局限性中明确指出，这种过程可能低效且对 noisy correspondences、遮挡和词汇尺度估计敏感。它适合离线或交互式建模验证，不应在没有端到端 profile 的情况下声称实时。

## 16. Table 1：位姿结果与深度级联

![SSDB 零样本 9-DoF 对齐结果](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-table01-zero-shot-9dof.webp)

*Source: Wu et al., arXiv:2411.19492v1, Table 1, arXiv non-exclusive distribution license.*

| Method | GT rAcc | GT tAcc | GT sAcc | GT Acc | Est. rAcc | Est. tAcc | Est. sAcc | Est. Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best multiview | 0.34 | 0.93 | 0.52 | 0.19 | 0.34 | 0.84 | 0.44 | 0.15 |
| ZSP + DINOv2 | 0.37 | 0.93 | 0.57 | 0.23 | **0.42** | **0.85** | 0.55 | **0.26** |
| ZSP + ft DINOv2 | **0.45** | 0.94 | 0.66 | **0.34** | 0.41 | **0.85** | 0.54 | 0.24 |
| GigaPose | 0.36 | **0.95** | **0.71** | 0.27 | 0.36 | 0.84 | **0.64** | 0.22 |
| Ours | **0.45** | **0.95** | **0.71** | 0.33 | 0.41 | 0.84 | **0.64** | 0.25 |

论文的 `Ours` 在 GT depth 下没有拿到最高 overall Acc：ZSP + fine-tuned DINOv2 为 `0.34`，Ours 为 `0.33`。Diorama 的优势体现在多指标组合：旋转、平移、尺度、碰撞和关系。estimated depth 下 ZSP + DINOv2 overall `0.26` 也略高于 Ours `0.25`。所以准确表述是“Diorama 在多项 9-DoF、碰撞和关系指标上有竞争力并提供完整系统”，而不是“所有位姿指标全面最好”。

GT depth 到 estimated depth 时，Ours 的 tAcc `0.95 -> 0.84`，Acc `0.33 -> 0.25`，relation `0.61 -> 0.25`。relation 的大幅下降提醒我们：场景关系是否在 3D 中正确实现，强依赖对象位置和尺度，不是 GPT-4o 输出一条正确边就结束。

## 17. 多个检索假设：承认单图欠约束

![不同 CAD 检索数量对 9-DoF 对齐的影响](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-table07-retrieval-hypotheses.webp)

*Source: Wu et al., arXiv:2411.19492v1, Table 7, arXiv non-exclusive distribution license.*

| Retrievals | GT Acc | Estimated Acc |
| ---: | ---: | ---: |
| 1 | 0.09 | 0.08 |
| 4 | 0.18 | 0.15 |
| 8 | **0.21** | **0.18** |

从 1 个候选增加到 8 个，GT depth overall Acc 从 `0.09` 到 `0.21`，estimated depth 从 `0.08` 到 `0.18`。这说明检索排序第一名不一定是最适合建立 correspondence 的 CAD；增加候选扩大了几何兼容性。

但“多个 plausible hypotheses”不是校准后的概率分布。论文没有给每个完整 scene arrangement 的 posterior、置信区间或用户偏好模型。多个结果只是离散候选，仍需要启发式、指标或人工选择。生产产品若要展示多方案，应记录每个对象的检索分数、pose inlier ratio、collision 和 relation score，而不是把候选顺序伪装成可信概率。

## 18. 布局优化：从对象级对齐到场景级合理

![语义感知的分阶段场景布局优化](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig05-layout-optimization.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 5, arXiv non-exclusive distribution license.*

对象逐个估计的 pose 不会自动形成合理场景。一本书可能离桌面几厘米，椅子可能穿进桌腿，花瓶可能悬空，书架里的书可能突出侧板。Diorama 将每个对象近似为 oriented bounding box，并从场景图构造 contact surface、support surface 与 support direction。

优化被拆为四阶段：

1. **Orientation**：接触面与支撑面法线对齐，同时保留对象语义正面；
2. **Placement**：把接触面中心移到支撑面，并尽量保持对象相对位置；
3. **Space**：减少对象/建筑碰撞，限制被支撑对象不超出支撑面或支撑体积；
4. **Refinement**：再次执行 placement，并对靠墙对象建立 adherence 关系。

分阶段不是写作顺序，而是优化策略。部分变量在某阶段 detach，意味着该阶段只允许某些参数更新。当前 [`layout_optim.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/model/layout_optim.py) 中可以看到 rotation、translation、scale 在不同 loss 路径被有意冻结或开放。

## 19. Orientation：法线对齐与语义正面

对被支撑对象 $o_i$ 和支撑对象 $o_j$，记接触面法线为 $\mathbf n_i^c$，支撑面法线为 $\mathbf n_j^s$。论文写出：

$$
e_{\text{align}}=
\sum_{(i,j)\in G}
\left\lVert \mathbf n_i^c-\mathbf n_j^s\right\rVert_2.
$$

若一本书平放在桌上，书底法线应和桌面支撑方向一致；若壁画挂在墙上，背面接触方向应与墙面方向对应。当前源码还对 adherence surface 给予额外权重，处理贴墙对象。

仅对齐接触法线会产生绕法线自由旋转。椅子可以背对桌子，显示器可以朝墙。论文因此维护初始 front direction 在水平面上的投影，定义语义朝向损失：

$$
e_{\text{sem}}=
\sum_i\left\lVert \mathbf v_i^*-\mathbf v_i\right\rVert_2,
$$

$\mathbf v_i$ 是初始正面方向，$\mathbf v_i^*$ 是优化后方向。它不是从物理求出的，而来自 CAD canonical orientation 与前序语义判断。因此 CAD 坐标不统一时，语义朝向也会错。

Orientation stage 的优点是先把离散的大旋转错误消掉，再处理位置。若同时优化位置、旋转和尺度，碰撞梯度可能推动对象以奇怪角度逃离，而不是先落到正确支撑方向。

## 20. Placement：接触、支撑与相对位置

设接触面中心为 $c_i$，支撑面为 $S_j$，Placement 希望 $c_i$ 落在 $S_j$ 上。对平面 $S_j$ 上一点 $x_j$ 和法线 $n_j$，点到面的距离为：

$$
e_{\text{place}}=
\sum_{(i,j)\in G}
\left|\left(c_i-x_j\right)^Tn_j\right|.
$$

当前实现的 `place_on_loss()` 正是把优化后的 contact center 投到 support plane normal 上计算绝对距离。对于对象支撑对象，support surface 随支撑对象的 rotation、scale 和 translation 一起变换；对于地面、墙等建筑元素，则使用固定建筑 plane。

只把对象压到支撑面可能破坏原有布局：桌上的杯子、键盘和显示器会向同一位置聚集。论文引入相对位置项：

$$
e_{\text{rel}}=
\sum_{(i,j)}
\left\lVert
(c_i^*-c_j^*)-(c_i-c_j)
\right\rVert_2,
$$

用初始相对位移约束优化后布局。这个项保留了图像投影和初始点云给出的空间顺序，代价是初始误差也可能被保留。它是一种 regularizer，不是从真实 3D 标注得到的绝对真值。

## 21. Space：碰撞、支撑面积与容纳体积

对象都有 3D OBB。对两个凸 cuboid，Separating Axis Theorem 判断是否存在分离轴；若所有候选轴都重叠，就可计算最小 penetration depth。论文把各对象与建筑的穿透累计成 $e_{\text{col}}$。

碰撞损失有两个边界。第一，它使用 OBB 而不是精细 mesh，椅腿之间本可容纳物体，但 OBB 可能把整个区域视为实心。第二，碰撞最小不等于布局正确：把所有对象推得很远也能消除碰撞，所以它必须和 placement、relative location 一起使用。

普通 support surface 约束只要求接触中心落在平面上，没有保证整个对象底面不伸出桌边。`support_space_loss()` 把 contact corners 投到支撑面的两个局部轴，惩罚超过边界的部分。对于“书放进书架”“物体放进柜子”等 case，`support_volume_loss()` 还估计支撑体积，惩罚对象 corners 超出容纳区域。

可以把空间阶段写成：

$$
E_{\text{space}}=
\lambda_{\text{col}}e_{\text{col}}+
\lambda_{\text{surf}}e_{\text{surf}}+
\lambda_{\text{vol}}e_{\text{vol}}+
\lambda_{\text{rel}}e_{\text{rel}}.
$$

v1 没有以一张统一权重表完整公开所有实现常数，因此正文不补造具体 $\lambda$。当前代码中的系数属于后续实现，可用于理解，不应写成 v1 唯一配置。

## 22. Refinement：为什么最后再贴墙和落地

Space stage 可能为了减少碰撞而让对象重新离开支撑面。Refinement 再运行 placement，把对象拉回支撑关系；对于 floor-supported object，还用距离启发式识别潜在 wall adherence，例如书柜、电视柜或床头靠墙。

这种“先满足一组约束，再处理另一组，最后回补”的策略，和经典 block coordinate optimization 很接近。它不保证全局最优，却提供更稳定的优化路径。论文附录使用不同 SGD optimizer：基础学习率 `0.01`，momentum `0.9`；空间尺度优化学习率 `0.001`；每阶段 200 steps，每 50 steps 将学习率乘 `0.1`。

优化器本身没有学习房间布局分布。它只处理当前场景图、初始 pose、OBB 和建筑平面的局部几何约束。若 GPT 把“椅子被桌子支撑”判断错，优化器会尽力把椅子放到桌面上。这是可解释系统的优点也是危险：错误目标能沿 loss trace 被定位，但不会被模型常识自动修复。

## 23. Table 2：逐阶段消融的非单调性

![Diorama 布局优化阶段消融](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-table02-layout-ablation.webp)

*Source: Wu et al., arXiv:2411.19492v1, Table 2, arXiv non-exclusive distribution license.*

| Ablation | Collision ↓ | Orientation ↑ | Placement ↑ | Overall ↑ |
| --- | ---: | ---: | ---: | ---: |
| w/o optimization | 7.26 | 0.18 | 0.01 | 0.00 |
| + Orientation | 6.34 | 0.97 | 0.01 | 0.01 |
| + O, Placement | 7.61 | 0.97 | 0.94 | 0.91 |
| + O, P, Space | 5.90 | 0.97 | 0.54 | 0.53 |
| + O, P, S, Refinement | **4.43** | **0.97** | **0.95** | **0.92** |
| w/ GT scene graph | 3.28 | 0.97 | 0.97 | 0.95 |
| w/ GT architecture | 4.15 | 0.97 | 0.97 | 0.95 |

这张表最值得注意的不是最后的 `0.92`，而是中间指标非单调：Placement 后 collision 从 `6.34` 升到 `7.61`；Space 后 placement 从 `0.94` 降到 `0.54`。每个阶段修一个约束时都可能破坏另一个约束，最后 Refinement 才把 placement 恢复到 `0.95`。

GT scene graph 和 GT architecture 的 overall 都是 `0.95`，比完整预测的 `0.92` 高，说明上游结构信息仍是误差源。GT scene graph 的 collision `3.28` 优于 GT architecture 的 `4.15`，但不能由此简单断言 scene graph 永远比 architecture 更重要；两行分别替换一个组件，其他误差状态不同。

## 24. Table 8：为什么不能 all-in-one

![all-in-one 与 stage-wise 优化对比](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-table08-stagewise-optimization.webp)

*Source: Wu et al., arXiv:2411.19492v1, Table 8, arXiv non-exclusive distribution license.*

| Strategy | Collision ↓ | Orientation ↑ | Placement ↑ | Overall ↑ |
| --- | ---: | ---: | ---: | ---: |
| all-in-one | 12.49 | 0.18 | 0.84 | 0.15 |
| stage-wise | **4.43** | **0.97** | **0.95** | **0.92** |

all-in-one 把全部 terms 同时优化，结果不仅 overall 差，collision 甚至高于未优化初值。可能原因包括：loss 尺度不同、旋转/尺度/平移梯度相互干扰、初始 pose 离可行区域较远、OBB 碰撞项不光滑。stage-wise 通过人为 curriculum，把难问题拆成更局部的子问题。

这是一条可迁移的工程经验：当场景优化包含离散语义、强几何约束和多个尺度不同的目标时，先确定 orientation 和 support topology，再调整位置/尺度，通常比同时开放全部自由度更可控。它仍需要逐阶段失败检测，否则某阶段的坏结果会成为下一阶段的初值。

## 25. Fig. 6：SSDB 输出为何同时展示三种 arrangement

![SSDB 场景、支撑图与多个 CAD arrangement](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig06-ssdb-results.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 6, arXiv non-exclusive distribution license.*

每行依次展示 input image、ground truth、scene graph、使用 GT shapes 的 Arrangement 1，以及采用不同检索 CAD 的 Arrangement 2/3。这个排版把三种误差源分开：

- Arrangement 1 主要检验 pose、scene graph 和 layout；
- Arrangement 2/3 同时加入 retrieval mismatch；
- scene graph 列暴露 support hierarchy 的复杂度。

图中实验室台面和书架场景有大量小对象，说明论文不只处理床、桌、沙发等大件家具。另一方面，输出材质和几何与 GT 有明显差异，恰好体现 CAD-based modeling 的目标不是 exact reconstruction。

多个 arrangement 还揭示单图歧义：输入只约束可见表面和投影，许多 CAD 能解释相近轮廓。一个诚实系统应保留候选，而不是把 top-1 伪装成唯一真相。Diorama 做到了候选生成，但没有提供完整的场景级排序或人类偏好校准，这仍是后续空间。

## 26. 数据集：SSDB、OOD 资产库与 ScanNet

### 26.1 SSDB

Stanford Scene Database 包含约 130 个场景和 1700 个独特 CAD，覆盖办公桌、实验室、书架等密集排列。论文用 Blender 渲染 344 张图，平均每个场景约 3 个视角。SSDB 提供完整 CAD、建筑和支撑 hierarchy，适合做系统消融。

SSDB 的优点也是局限：图像是合成渲染，mask、对象和 CAD 对应可得，纹理和噪声分布与真实照片不同。论文可以精确计算 9-DoF 与 support relation，但不能据此证明互联网图片上的同等精度。

### 26.2 OOD retrieval collection

v1 将 HSSD 和 Objaverse LVIS 资产加入库，形成约 60K 的 OOD collection。所谓 OOD 是相对于 SSDB 资产和分布，不等于对任意 3D 数据源都 out-of-domain。资产预处理、canonical orientation、多视图渲染和 embedding 计算仍是重要前置。

### 26.3 ScanNet 子集

论文选 600 张 ScanNet 图，每张至少包含 3 个不同对象类别或 5 个实例，总计约 3000 instances，覆盖 24 categories。作者与 Scan2CAD、ROCA、DiffCAD 对比，但为了公平对齐，定性比较共享 mask、depth 和 3D shape inputs，主要隔离 pose alignment 模块。

这意味着 ScanNet 结果不是“完整 Diorama 从原始 RGB 对端到端方法”的完全统一比较。论文有意做 component-level comparison，读者也应按这个口径解释。

## 27. 指标：对象对齐与场景合理是两回事

传统 Scan2CAD 风格指标把 CAD 判为正确对齐，需要同时满足：

$$
e_t\le 20\text{ cm},\qquad
e_R\le 20^\circ,\qquad
e_s\le 20\%.
$$

分别得到 translation accuracy `tAcc`、rotation accuracy `rAcc`、scale accuracy `sAcc` 和联合 `Acc`。

论文认为平均对象指标不能捕获“少数极坏对象毁掉整个场景”，因此增加 scene-aware alignment。一个对象不仅要接近自身 GT pose，还要相对场景中其他对象满足位置、旋转和尺度阈值。简化写法为：

$$
\forall j\ne i:\quad
\Delta t_{ij}\le \tau_t,
\quad \Delta R_{ij}\le \tau_R,
\quad \Delta s_{ij}\le \tau_s.
$$

场景质量还包括：

- average mesh collision，越低越好；
- support relation accuracy，parent 和 support surface 都要正确；
- supportness/placement，检查对象是否实际落在支撑表面；
- retrieval Chamfer Distance，比较检索 CAD 与 GT shape。

这些指标比只看 input-view RGB 更贴合可编辑场景，但仍不包含动力学稳定性、可通行性、关节、材质、光照或用户编辑成本。

## 28. Table 5：小物体、遮挡和支撑角色

![不同 SSDB 对象组的平均对齐结果](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-table05-object-groups.webp)

*Source: Wu et al., arXiv:2411.19492v1, Table 5, arXiv non-exclusive distribution license.*

| Group | Categories / instances | rAcc | tAcc | sAcc | Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Household | 466 / 5577 | 0.44 | 0.84 | **0.71** | 0.29 |
| Furniture | 24 / 1137 | **0.53** | 0.84 | 0.69 | **0.34** |
| Occluded | 369 / 3175 | 0.43 | 0.82 | 0.64 | 0.25 |
| Complete | 401 / 3539 | **0.48** | **0.86** | **0.77** | **0.34** |
| Supported | 468 / 5715 | 0.46 | **0.85** | **0.71** | **0.30** |
| Supporting | 79 / 1634 | 0.46 | 0.82 | 0.70 | 0.28 |

遮挡对象 overall `0.25`，完整对象 `0.34`，符合 patch correspondence 和点云尺度对可见区域的依赖。Furniture rotation `0.53` 高于 Household `0.44`，很可能因为大件家具的 canonical orientation、可见结构和点云更稳定。

Supported 与 Supporting 差异并不大，说明“作为支撑物”不自动更容易。支撑家具通常体积大，但也更容易被多个对象遮挡；小被支撑物虽然点少，却可能轮廓完整。分组不是因果实验，只能帮助定位误差模式。

## 29. Fig. 7 与 Table 6：ScanNet 比较的正确读法

![Diorama、ROCA 与 DiffCAD 的 ScanNet 定性比较](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig07-scannet-comparison.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 7, arXiv non-exclusive distribution license.*

Fig. 7 中 Diorama、ROCA、DiffCAD 使用相同 mask、depth 和 3D shape input，减少检测与检索差异。Diorama 能处理 laptop、backpack 等对手封闭类别之外的对象，这是开放世界路线的重要优势。

![Scan2CAD object-focused alignment accuracy](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-table06-scannet-alignment.webp)

*Source: Wu et al., arXiv:2411.19492v1, Table 6, arXiv non-exclusive distribution license.*

Table 6 中 Diorama class average `3.55`、instance average `5.87`。ROCA 分别是 `8.72/12.87`，在许多已知类别明显更强；DiffCAD 5 hypotheses 是 `4.34/6.84`，也略高于 Diorama 平均。Diorama 的优势是无需 Scan2CAD 监督并能给 `others` 类别 `4.76`，而不是已知类别上的绝对 SOTA。

这张表尤其适合纠正摘要中的宽泛“significantly outperform baselines”表述。该结论在 SSDB、系统完整性或特定模块上成立，但 ScanNet 对齐表显示受监督 ROCA 仍强。高质量精读应保留这种张力，而不是只摘最有利数字。

## 30. Fig. 13：评价集合不同会怎样改变结论

![DiffCAD、ROCA 与 Diorama 使用的 Scan2CAD 评价集合差异](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig13-evaluation-set-differences.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 13, arXiv non-exclusive distribution license.*

Fig. 13 可视化不同工作使用的 GT subset。某些 benchmark 行只覆盖常见家具，无法评价 laptop、backpack 等额外对象。若一个开放世界方法把未覆盖对象也建模出来，传统 closed-set metric 既不给奖励，也可能在汇总时被排除。

这不意味着开放世界方法可以绕过标准 benchmark。正确做法是同时报告：

- 与既有工作重叠类别的可比指标；
- 新增类别的覆盖和质量；
- 完整系统在真实图片上的失败率；
- 使用不同 GT subset 时的分母。

Diorama 对 evaluation-set difference 的展示很有价值，但 v1 仍缺少一个大规模、开放类别、真实 RGB、带完整场景结构 GT 的数据集。这是领域数据稀缺问题，不是单篇论文能轻易解决的。

## 31. Fig. 8/15：互联网图片证明了什么

![Diorama 在网络室内图片上的结果](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig08-in-the-wild.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 8, arXiv non-exclusive distribution license.*

![更多互联网图片结果](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig15-more-real-world.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 15, arXiv non-exclusive distribution license.*

真实照片包含曝光变化、复杂材质、不同相机内参和 SSDB 未见的摆设。输出捕获了沙发、桌、灯、墙饰、植物和窗帘等大小对象，并能表达 wall-mounted 与 object-on-object 关系。这说明开放词汇感知、CAD 库和模块化几何在 domain shift 下没有立即失效。

但这些结果只有定性展示，没有 per-image GT、失败分母或人工偏好统计。可见的误差包括 CAD 形态不匹配、对象遗漏、墙面布局简化和尺度偏差。论文没有声称 photorealistic reconstruction，读者也不应把“看起来像一个房间”转换成精确数字孪生。

互联网图片来源于 Pixabay、Pexels 和 Unsplash。若做产品复现，还需处理输入图片版权、隐私、API 数据保留和生成资产许可证，论文方法没有自动解决这些治理问题。

## 32. Fig. 9/16：文本到场景其实是两阶段组合

![Flux-1 图像生成后接 Diorama 的文本到场景结果](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig09-text-to-scene.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 9, arXiv non-exclusive distribution license.*

![更多文本到场景案例](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig16-more-text-to-scene.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 16, arXiv non-exclusive distribution license.*

文本先交给 Flux-1 生成单张室内图，再由 Diorama 建模：

```text
text prompt
  -> Flux-1 image generation
  -> single RGB image
  -> Diorama perception / retrieval / pose / layout
  -> editable CAD scene
```

因此文本遵循度同时由 Flux 和 Diorama 决定。若 prompt 里的物体没有出现在生成图中，Diorama 不会凭原始文本补回；若 Flux 画出不符合透视或不可能的家具，Diorama 会试图用 CAD 近似。它不是直接对文本布局进行联合 3D 推理。

这条组合路线的优势是可以复用强图像模型的构图和风格先验，再把结果“资产化”。它和 Layout2Scene、CompoNeRF 等直接把 layout/text 作为 3D 优化条件的方法有根本区别：Diorama 的唯一几何观测仍是一张生成图片。

## 33. Fig. 11：结构化表示如何支持编辑

![Diorama 的对象重排、插入、删除和替换](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig11-editing.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 11, arXiv non-exclusive distribution license.*

Fig. 11 展示 rearrangement、insertion、removal 和 replacement。由于对象是独立 CAD，编辑不必重新优化一个整体神经场。一个产品可以把 `SceneSpec` 暴露给编辑器：点击对象后更换 asset、修改 transform 或更新 support parent。

真正的编辑系统仍需要额外逻辑：

- 删除 supporting object 时，如何处理其 children；
- 插入对象后如何选 contact surface；
- 替换为不同尺寸 CAD 后是否重新布局；
- 用户移动对象时，哪些关系应保留、哪些应解除；
- 建筑 opening、门窗和可通行区域如何维护。

论文只展示潜力，没有实现完整交互 UI、事务模型或物理校验。因此“可编辑”应理解为数据表示允许编辑，不等于已交付生产级场景编辑器。

## 34. Fig. 14：更多 SSDB 结果与长尾复杂度

![更多 SSDB 场景、场景图与替代 CAD arrangement](/images/blog/diorama-zero-shot-single-view-3d-scene-modeling/diorama-fig14-more-ssdb.webp)

*Source: Wu et al., arXiv:2411.19492v1, Fig. 14, arXiv non-exclusive distribution license.*

附录的大图覆盖卧室、储物间、客厅、厨房和餐桌，支撑图规模从少量大件到几十个小对象。它说明系统输出复杂度随对象数快速上升：每个对象都需要 detection、mask、point cloud、retrieval、pose 和 relation；支撑关系与碰撞近似还会引入对象对或层级计算。

图中 Arrangement 2/3 证明不同 CAD 仍能保留场景关系，也暴露检索资产的风格不统一。对于仿真数据生成，语义和结构可能比美术一致性重要；对于游戏或数字内容生产，用户还需要材质重建、风格统一、LOD、UV 和 license metadata。

## 35. 当前官方代码：不是一条单命令，而是一组阶段

当前 README 的运行顺序大致为：

```text
1. GPT-4o scene understanding
2. Metric3D depth estimation
3. OWLv2 + SAM2 object perception
4. DuoDuoCLIP CAD retrieval
5. DINOv2/GigaPose-based pose estimation
6. layout optimization
7. separate PlainRecon architecture scripts
```

入口 [`run.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/run.py) 使用 Hydra 读取 `config/cfg_wss.yaml`，创建 `VisionAgent` 和 `VLMAgent`，再根据 flags 选择阶段。配置中的 `load_depth_model`、`load_perception_model`、`load_retrieval_model`、`load_pose_model` 和 `run_optimization` 默认都为 false，体现阶段式执行和中间结果复用。

这并非负面评价。重模型很多时，阶段缓存可以避免每次修改 scene graph 都重新跑 depth 和 segmentation；也方便人工检查。问题在于复现者必须正确管理路径、模型版本、CAD embedding 与中间 JSON，否则“代码公开”不等于开箱即用。

## 36. `VisionAgent`、`VLMAgent` 与 `SceneSpec`

### 36.1 `VLMAgent`

[`diorama/vlm_agent.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/vlm_agent.py) 的 `understand_scene()` 读取图像并请求 GPT-4o，`generate_scene_graph_from_augment()` 消费带 marks 的图像，`parse_gpt4v_response()` 处理结构化输出。当前配置 `n_trials: 6`，说明代码预期 LMM 响应可能失败，需要重试和解析防护。

### 36.2 `VisionAgent`

[`diorama/vis_agent.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/vis_agent.py) 是视觉与几何 orchestrator：

- `_init_models()` 按配置加载 Metric3D、perception、retrieval 和 pose 模型；
- `detect_objects_w_owlv2()`、`segment_objects()` 完成 2D perception；
- `segment_object_crops_and_pcd()` 生成 crop 与对象点云；
- `augment_image_w_marks()` 生成 SoM 输入；
- `retrieve_shapes()` 写入 CAD candidates；
- `estimate_poses()` 运行 GigaZSP；
- `optimize_layout()` 进入 OBB 优化。

### 36.3 `SceneSpec`

`SceneSpec` 是跨阶段合同。它把 LMM 的对象/关系、视觉模块的 box/mask、检索结果和 support surface 汇合起来。一个更成熟的实现应对每个字段附加：source module、model version、confidence、timestamp、manual override 和 validation status。论文代码主要面向研究复现，未完整实现这套生产 provenance。

## 37. 当前配置揭示的运行成本

固定提交的 README 需要 Python 3.10、PyTorch 2.5.1、CUDA 12.1，并安装 SAM2、GroundingDINO、xformers、PyTorch3D、FAISS-GPU 和修改后的 OpenCLIP。权重至少包括：

- DINOv2 ViT-L；
- OWLv2 Large；
- SAM2.1 Hiera Large；
- BiRefNet；
- Metric3DV2；
- DuoDuoCLIP；
- GigaPose-derived scale model。

还要准备多个 CAD pool 的 metadata 和 embedding。当前配置中 text KNN 为 50，再取 visual top-5；每个 CAD 有 180 个 reference views；pose 每次保留 50 correspondences。GPT-4o 需要 API key 和网络调用。

这些信息足以推翻一种常见误解：training-free 不等于部署轻量。相反，模块式 foundation-model pipeline 往往显存、磁盘、网络和冷启动成本都很高，只是省去了目标任务端到端训练。

## 38. 论文与当前代码的差异审计

| 维度 | v1 论文 | 固定代码提交 |
| --- | --- | --- |
| 论文阶段 | 2024 v1 | 2026-01 后续实现 |
| segmentation 表述 | Segment Anything | 配置包含 SAM 与 SAM2.1，主路径可用 SAM2 |
| pose similarity threshold | 补充材料 `0.7` | `0.5` |
| reference views | 180 | `n_ref: 180` |
| patch model | fine-tuned DINOv2 ViT-L, patch 14 | `dinov2_vitl14`, patch 14, layer 17 |
| retrieval | 60K OOD collection 的论文实验 | 本地 `cad_pool` 配置与数据打包决定实际集合 |
| LMM | GPT-4o checkpoint dated in supplement | `gpt-4o` API model name，服务端版本可漂移 |
| architecture | PlainRecon 方法与评测 | 多脚本、第三方子模块和手工路径配置 |

版本差异不代表代码错误。论文发布后作者可能升级组件、修复阈值或重构数据。正确做法是：论文数字由 v1 PDF 支撑，源码行为由固定 commit 支撑；任何复现实验同时记录二者，不把“当前 README 跑通”写成“复现 v1 表格”。

## 39. 与 ROCA、DiffCAD 和传统 CAD 对齐的关系

ROCA 使用真实 Scan2CAD in-domain supervision，学习对象检测、形状和 pose；DiffCAD 用合成数据与概率模型产生 CAD 对齐假设。它们在已知 ScanNet 类别和 benchmark protocol 上具备强性能，但类别和训练域受限。

Diorama 的改变有三点：

1. 检测和语义改用开放词汇 foundation models；
2. 位姿依赖跨实例 patch correspondences，而不是每类 pose regressor；
3. 输出包含 architecture 与 support hierarchy，并在场景级优化。

代价是推理管线更长、外部依赖更多，且真实数据上的精确对齐不如受监督方法。两类路线不是简单替代关系：已知类别、固定设备、强标注环境可以继续选择专用模型；类别长尾、场景编辑和快速原型更适合 Diorama 式组合。

## 40. 与本站其他 3D 论文的关系

| 方法 | 输入与目标 | 核心表示 | Diorama 的区别 |
| --- | --- | --- | --- |
| CasLayout | 生成 3D layout | 级联布局扩散 | Diorama 从图片感知布局，不学习布局生成 prior |
| Imaginarium | 图片引导场景生成 | 资产检索 + 物理/视觉优化 | 同样检索资产，但 Diorama 更强调开放感知和 support graph |
| Layout2Scene | 语义 layout 到完整场景 | Gaussian object + polygon background | Layout2Scene 输入已有 layout，Diorama 负责从图像得到结构 |
| CompoNeRF | 文本多物体生成 | 多个局部 NeRF + 全局组合 | Diorama 用离散 CAD，不做 SDS/NeRF 优化 |
| Blended-NeRF | 编辑已有 NeRF | ROI 内双 NeRF 融合 | Diorama 从零装配整个 CAD 场景 |
| TRELLIS.2 | 生成高质量 3D 资产 | O-Voxel / sparse latent | TRELLIS.2 解决单资产表示，Diorama 解决场景理解与装配 |
| AssetGen / SF3D | 单图到可部署资产 | Mesh + UV + texture | 更关注单资产几何/贴图，Diorama 更关注多对象关系 |
| TripoSG | 单图高保真 shape | SDF VAE + Rectified Flow | 可作为未来检索失败时的对象生成组件 |

这张表也指向一个潜在融合方向：Diorama 的 scene graph 和 layout 提供对象级控制，AssetGen/TRELLIS.2/TripoSG 提供更贴近输入的单资产。未来系统可以先检索，检索置信度不足时再生成资产，并统一进入布局优化。论文 v1 没有实现这条混合路线。

## 41. 可迁移的工程经验

### 41.1 先定义输出合同，再选模型

Diorama 的成功来自对输出的选择。若目标是可编辑场景，就需要对象 ID、资产、pose 和关系；只优化 novel-view RGB 不够。工程团队应先决定下游需要 mesh、CAD、物理 collision、材质还是图像，再确定感知和生成管线。

### 41.2 中间结果必须可视化

至少保存：对象 marks、mask overlay、depth、normal、建筑 planes、检索 top-K、patch correspondences、RANSAC inliers、初始/最终 OBB、support graph 和每阶段 loss。否则最终场景失败时无法知道是 GPT、depth、retrieval 还是 optimizer。

### 41.3 语义和几何要分别验收

一个 CAD 可以语义正确但几何错误；一个 pose 可以局部对齐但违反支撑关系。检索、对象对齐和场景结构需要不同指标，不能用一张渲染图统一打分。

### 41.4 保留多假设，但控制组合爆炸

每个对象 8 个 CAD，场景有 20 个对象时不能枚举 $8^{20}$ 个组合。需要 per-object pruning、共享 scene score、beam search 或用户交互。Diorama 展示多个 arrangement，但没有给大规模组合选择算法。

### 41.5 优化器需要可行性门禁

每阶段后检查 NaN、尺度范围、建筑外比例、collision、supportness 和相机投影误差。若 orientation stage 已失败，不应继续运行数百步 placement。

## 42. 工程复现路线

一个诚实的复现应分层推进，而不是第一天就追完整 Fig. 6。

### Stage 0：环境与资产准备

1. 固定 Git commit、Python/PyTorch/CUDA 版本；
2. 下载并校验所有权重哈希；
3. 准备至少一个 CAD pool 的 metadata、canonical transform、多视图 RGB/depth 和 DuoDuoCLIP embedding；
4. 配置 GPT API，但不要把密钥写进 YAML 或日志；
5. 用单张图验证各模型可独立加载。

### Stage 1：2D 感知与场景图

1. 运行 GPT scene understanding；
2. 运行 OWLv2 和 SAM2；
3. 检查重复对象、漏检和 mask 边界；
4. 生成 marks，保存 GPT 原始响应和解析结果；
5. 验证 support graph 是 DAG，所有 target ID 存在。

### Stage 2：深度、点云与 PlainRecon

1. 明确相机内参来源；
2. 估计 metric depth/normal；
3. 检查单位和坐标约定；
4. 运行 object removal/inpainting；
5. 恢复 plane，确认 floor normal 与重力方向一致；
6. 记录每个 plane 的 inlier 数、面积和边界。

### Stage 3：检索与位姿

1. 先在一个对象上检查 text top-50 与 visual top-5；
2. 可视化 180 个 reference views 中的最佳视角；
3. 检查 cyclic correspondences 与相似度；
4. 记录 RANSAC inlier ratio、residual、scale network 输出；
5. 将 CAD 投影回输入图，检查 silhouette 与关键点。

### Stage 4：场景优化

1. 从 initial OBB 与 support graph 构造 surfaces；
2. 分阶段运行，每阶段保存 checkpoint；
3. 记录各 loss、collision 和 supportness；
4. 对尺度、位置和旋转设置合理范围；
5. 导出最终场景和多个渲染视角。

### Stage 5：评测

完整复现 Table 1-8 需要 SSDB、对应 GT、ScanNet subset、CAD pool 和作者评测协议。若只运行网络图片，只能报告定性结果、运行时间、人工失败分类和中间指标，不能声称复现论文表格。

## 43. 建议的数据契约

论文代码使用多个 JSON 和目录。若将其工程化，可以定义如下中间记录：

```json
{
  "scene_id": "scene_001",
  "paper_version": "2411.19492v1",
  "pipeline_commit": "ec54d826...",
  "input": {
    "image_sha256": "...",
    "intrinsics": [[1000, 0, 512], [0, 1000, 392], [0, 0, 1]]
  },
  "objects": [{
    "object_id": 7,
    "category": "chair",
    "bbox_xyxy": [120, 90, 410, 650],
    "mask_path": "masks/7.png",
    "depth_path": "depth.npy",
    "support_parent": "floor",
    "support_relation_source": "gpt-4o",
    "retrieval_candidates": [{"asset_id": "...", "text_score": 0.8, "image_score": 0.7}],
    "pose": {
      "policy": "dino-gigazsp-umeyama-ransac",
      "reference_view": 42,
      "correspondence_count": 50,
      "inlier_ratio": 0.62,
      "rotation": "...",
      "translation_m": [0.2, 1.0, 2.5],
      "scale": [0.8, 0.8, 0.8]
    }
  }]
}
```

关键不是字段名字，而是保留 provenance。GPT 的关系、detector 的类别、retrieval score 和 pose solver 证据不应被压扁成一个无法解释的最终 transform。

## 44. 监控与失败分类

### 感知层

- object count、重复率、mask coverage、平均 confidence；
- GPT 类别与 detector 类别不一致率；
- scene graph 无效 ID、环、缺 parent 比例；
- depth scale、normal consistency 与 invalid pixel 比例。

### 检索层

- top-1/top-5 score margin；
- 类别过滤后候选数；
- CAD 许可证和资产文件可用率；
- query 与最佳 render 的 patch similarity。

### 位姿层

- correspondence 数、RANSAC inlier ratio；
- rotation/translation/scale 异常值；
- CAD 投影与 mask IoU；
- 同一对象多个候选 pose 的分散程度。

### 场景层

- support graph depth；
- unsupported object、floating object 数；
- collision volume；
- 建筑外对象比例；
- 每阶段 loss 变化与 early failure；
- 导出场景的对象数、三角面数和资源缺失。

最终失败可以分类为 `missed-object`、`duplicate-object`、`bad-depth`、`bad-architecture`、`wrong-parent`、`retrieval-mismatch`、`pose-outlier`、`optimization-divergence`、`asset-license-missing`。这种分类比一个总分更能指导模块替换。

## 45. 局限性与批判

### 45.1 单图歧义没有消失

不可见背面、被遮挡对象、房间深度和真实尺寸无法由单张 RGB 唯一确定。Diorama 用 CAD、metric depth、scene graph 和多假设缓解，却没有解决不确定性本身。输出是 plausible reconstruction，不是 ground-truth recovery。

### 45.2 模块误差会级联

错误类别导致错误 CAD pool；错误 mask 导致点云污染；错误 depth 导致尺度/平移偏差；错误 support edge 导致优化目标错误。模块可替换不代表误差独立，系统需要跨模块 consistency checks。

### 45.3 CAD 检索牺牲外观与局部几何

资产可编辑，却可能与输入的品牌、纹理、比例和细节不同。对仿真、场景草图和内容原型足够，对文物数字化、商品复刻和精确测量不够。

### 45.4 建筑模型仍然简化

PlainRecon 适合平面建筑。曲墙、楼梯、复杂吊顶、镜面、玻璃和大面积开放空间会挑战 normal clustering 与 planar representation。inpainting 还可能虚构不真实的墙体。

### 45.5 支撑和物理关系较粗

OBB、support surface 和 SAT collision 提供静态近似，不包含重心、摩擦、关节、软体、可打开容器或动力学。视觉上“放在桌上”不等于仿真中稳定。

### 45.6 计算成本缺少端到端审计

Table 4 给了 PlainRecon 时间，但论文没有完整报告每个场景的 GPT API、检测、分割、180-view matching、RANSAC、检索和布局优化总延迟，也没有显存、硬件和成本表。不能从 `training-free` 推断低成本或实时。

### 45.7 真实世界证据主要是定性

SSDB 提供精细 GT 但为合成数据；ScanNet 定量比较按特定 subset 和共享输入；互联网图与 text-to-scene 只有展示。开放世界鲁棒性还需要更大规模真实 benchmark、人工评价和失败分母。

### 45.8 外部服务会漂移

GPT-4o 是托管 API 名称，底层 checkpoint 可能更新。当前代码依赖第三方仓库和权重，未来版本兼容性、许可与下载可用性都可能变化。严格复现必须缓存响应或固定可审计模型。

## 46. 论文主张的证据强度

| 主张 | v1 证据 | 强度判断 |
| --- | --- | --- |
| 可以从单图构造完整 CAD 场景 | SSDB、ScanNet、网络图、text-to-scene | 有系统级定量与定性支持 |
| 零样本 pose 有竞争力 | Table 1、6、7 | 在 SSDB 较强；ScanNet 非全面领先 |
| PlainRecon 更可靠 | Table 4，344/344 success | 支持较强，但 RMSE 非全部最好 |
| DuoDuoCLIP hybrid 检索更好 | Table 3 分组 CD | 多数分组支持；存在 oracle top-5 选择 |
| stage-wise 优化必要 | Table 2、8 | 证据直接且差距大 |
| 泛化到真实图片 | Fig. 8、15 | 定性支持，缺少规模化精度 |
| 可用于 text-to-scene | Fig. 9、16 | 展示可行性，不是直接文本 3D 模型评测 |
| open-world | 额外类别、OOD pool、网络图 | 方向性支持，不等于无限域可靠 |

高质量论文阅读不能把所有主张压成“效果很好”。Diorama 最有说服力的是系统分解、SSDB 消融和 stage-wise optimization；最薄弱的是端到端真实世界量化、运行成本与不确定性校准。

## 47. 常见误读

| 误读 | 正确表述 |
| --- | --- |
| Diorama 没有用训练模型 | Diorama 不做端到端训练，但大量组件是预训练模型 |
| open-world 表示任何对象都能恢复 | 它放宽类别闭集，仍受检测器、语言模型和 CAD 库覆盖限制 |
| 输出多个 arrangement 就解决了歧义 | 它提供离散候选，没有校准完整后验 |
| CAD 输出与真实对象完全一致 | 论文明确优先语义与可编辑性，接受几何/纹理 mismatch |
| GPT-4o 直接给出完整 3D 场景 | GPT 主要给类别、caption 和支撑关系；几何来自专用模型与优化 |
| Table 1 所有指标都是 Diorama 最好 | 某些 overall Acc 由 ZSP 略高，Diorama 强在综合系统指标 |
| ScanNet 上全面超过 ROCA/DiffCAD | Table 6 中 ROCA 和部分 DiffCAD 平均更高，Diorama优势是零样本与类别覆盖 |
| 可编辑表示等于生产级编辑器 | 表示支持编辑，但 UI、事务、物理和资产治理尚未实现 |
| 当前 GitHub 就是 v1 完整代码快照 | 固定提交晚于论文，组件和阈值已有演进 |
| training-free 意味着实时或低成本 | 论文未给完整端到端 profile，管线依赖多个重模型和 API |

## 48. 推荐阅读路径

如果只有一小时，建议：

1. Abstract、Fig. 2：建立任务和两段式系统。
2. Sec. 3.1、Fig. 3：理解对象、场景图和 PlainRecon。
3. Sec. 3.2、Fig. 4/5：理解检索、pose 和布局优化。
4. Table 1/2/3：检查三个核心模块的主要数字。
5. Supplement A.3/A.5/A.6：读取平面、对应和优化细节。
6. Table 4/6/7/8：看可靠性、真实数据、候选数和阶段优化。
7. Fig. 8/15：检查真实图片边界。
8. 最后阅读当前代码 `run.py -> vis_agent.py -> gigazsp -> layout_optim.py`。

若要继续补课：

- CAD alignment：Scan2CAD、ROCA、DiffCAD；
- open-vocabulary detection：OWLv2；
- segmentation：SAM/SAM2；
- metric depth：Metric3D v2；
- semantic correspondence：DINOv2、ZSP、GigaPose；
- multimodal 3D retrieval：OpenShape、DuoDuoCLIP；
- visual grounding：Set-of-Mark；
- architecture reconstruction：RaC、PlaneRCNN；
- scene generation：ATISS、DiffuScene、AnyHome、LayoutGPT。

## 49. 结论

Diorama 回答了一个有价值的系统问题：不训练一个新的端到端场景网络，能否把现有 foundation models、CAD 数据和几何优化组织成完整的单图 3D 场景建模器？v1 的回答是“在受控合成数据上可以系统评测，在 ScanNet 和网络图片上可以展示有意义的零样本泛化”。

它真正推进的不是某个单独网络结构，而是三种接口：

1. 从开放词汇 2D perception 到对象点云和 scene graph；
2. 从多模态 CAD retrieval 到跨实例 patch correspondence 与 9-DoF pose；
3. 从对象级 pose 到包含 support、collision 和 architecture 的 scene-level optimization。

论文也明确选择了 trade-off：牺牲精确纹理和实例几何，换取完整、紧凑、可替换的 CAD；牺牲端到端联合训练，换取类别开放性和组件可替换性；接受多种 plausible scene hypotheses，而不是宣称从单图恢复唯一真相。

在这些边界内，Diorama 是一篇很有工程启发的工作。它提醒我们，3D 场景生成或重建的核心不只是模型规模，而是输出表示、模块合同、几何约束、资产库和评测协议共同构成的系统。要把它推进到生产，还需要端到端延迟、失败率、不确定性、物理验证、资产许可、材质恢复和人工编辑成本的完整闭环。

## 参考文献与一手资料

1. Qirui Wu et al. [Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling](https://arxiv.org/abs/2411.19492v1). arXiv:2411.19492v1, 2024.
2. Diorama v1 [PDF](https://arxiv.org/pdf/2411.19492v1), [HTML](https://arxiv.org/html/2411.19492v1), [TeX source](https://arxiv.org/e-print/2411.19492v1).
3. Diorama [project page](https://3dlg-hcvc.github.io/diorama/) and [official repository](https://github.com/3dlg-hcvc/diorama); 本文源码基线 [`ec54d826`](https://github.com/3dlg-hcvc/diorama/tree/ec54d826cf6b39c6daf88947da897ff9626e5d61).
4. Qirui Wu et al. [Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Diorama_Unleashing_Zero-shot_Single-view_3D_Indoor_Scene_Modeling_ICCV_2025_paper.html). ICCV 2025, Highlight. 仅用于说明后续发布状态。
5. Armen Avetisyan et al. [Scan2CAD: Learning CAD Model Alignment in RGB-D Scans](https://arxiv.org/abs/1811.11187). CVPR 2019.
6. Can Gümeli et al. [ROCA: Robust CAD Model Retrieval and Alignment from a Single Image](https://arxiv.org/abs/2112.00463). CVPR 2022.
7. Yang Gao et al. [DiffCAD: Weakly-Supervised Probabilistic CAD Model Retrieval and Alignment from an RGB Image](https://arxiv.org/abs/2305.13570). SIGGRAPH Asia 2023.
8. Maxime Oquab et al. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193). 2023.
9. Matthias Minderer et al. [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683). NeurIPS 2023.
10. Alexander Kirillov et al. [Segment Anything](https://arxiv.org/abs/2304.02643). ICCV 2023.
11. Nikhila Ravi et al. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714). 2024.
12. Mu Hu et al. [Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-depth and Surface Normal Estimation](https://arxiv.org/abs/2404.15506). 2024.
13. Jason Y. Zhang et al. [OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding](https://arxiv.org/abs/2305.10764). NeurIPS 2023.
14. Jaewoo Lee et al. [DuoDuo CLIP: Efficient 3D Understanding with Multi-view Images](https://arxiv.org/abs/2406.11579). 2024.
15. Nguyen et al. [GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence](https://arxiv.org/abs/2311.14155). CVPR 2024.
16. Jianwei Yang et al. [Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://arxiv.org/abs/2310.11441). 2023.

## 附录 A：代码阅读地图

| 模块 | 固定提交路径 | 关键问题 |
| --- | --- | --- |
| 总入口 | [`run.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/run.py) | 六阶段如何按 flags 运行 |
| 配置 | [`config/cfg_wss.yaml`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/config/cfg_wss.yaml) | 模型、阈值、CAD pools 和 optimizer 参数 |
| 视觉 orchestrator | [`diorama/vis_agent.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/vis_agent.py) | detection、segmentation、retrieval、pose、optimization |
| LMM | [`diorama/vlm_agent.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/vlm_agent.py) | GPT 请求、重试、解析和场景图 |
| 场景合同 | [`diorama/scenespec.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/scenespec.py) | 对象、关系和 support surface 如何持久化 |
| 检索 | [`duoduoclip.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/model/duoduoclip/duoduoclip.py) | text/image/multiview 编码 |
| Patch 对应 | [`correspondence.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/model/gigazsp/correspondence.py) | cosine similarity、mask、top pairs、threshold |
| 位姿主链 | [`gigazsp.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/model/gigazsp/gigazsp.py) | reference view、scale、3D correspondence、RANSAC |
| 尺度网络 | [`ist_net.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/model/gigazsp/ist_net.py) | GigaPose-derived scale estimator |
| 布局优化 | [`layout_optim.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/model/layout_optim.py) | OBB、support、collision、分阶段 optimizer |
| 建筑工具 | [`arch_util.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/diorama/utils/arch_util.py) | 平面载入、边界与建筑 mesh |
| 深度脚本 | [`scripts/estimate_depth.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/scripts/estimate_depth.py) | depth、intrinsics 与点云 |
| 平面脚本 | [`scripts/compute_plane_segmentation.py`](https://github.com/3dlg-hcvc/diorama/blob/ec54d826cf6b39c6daf88947da897ff9626e5d61/scripts/compute_plane_segmentation.py) | normal clusters 与 plane segmentation |

## 附录 B：复现验收清单

- [ ] arXiv 结论固定为 v1，不混用 ICCV 数字。
- [ ] 代码 commit、权重版本和第三方 submodule commit 已记录。
- [ ] 输入图 SHA、版权和相机内参来源可追溯。
- [ ] GPT 请求模型名、日期、prompt、原始响应和解析结果已保存。
- [ ] 对象列表无重复 ID，所有 mask 与图像尺寸一致。
- [ ] Scene graph 不含环，所有被支撑对象 parent 存在。
- [ ] Depth 单位明确，点云坐标系与 CAD 坐标系转换有测试。
- [ ] 建筑 floor normal 朝上，墙面面积和边界不过度异常。
- [ ] 每个 CAD 的 canonical orientation、unit scale 和 license 已记录。
- [ ] Retrieval top-K 同时保存 text/image score，不只保留 top-1。
- [ ] Reference render 数量、相机 pose 和 depth 与 embedding 对齐。
- [ ] Patch correspondence 在对象 mask 内，过滤阈值与论文/代码版本一致。
- [ ] RANSAC inlier ratio 和 residual 达到门槛，否则标记低置信度。
- [ ] Scale、translation、rotation 有异常值保护。
- [ ] Initial CAD 投影回输入图后经过 silhouette 或人工检查。
- [ ] 每个 support edge 都绑定 contact/support surface。
- [ ] Orientation、Placement、Space、Refinement 各自保存 checkpoint。
- [ ] 优化后 collision、supportness 和建筑外比例没有恶化。
- [ ] 删除或替换 supporting object 时，children 有明确处理策略。
- [ ] 导出场景不缺失 mesh、texture 或材质引用。
- [ ] 网络图片只报告定性结果，不冒充论文 benchmark 复现。
- [ ] 端到端耗时包括 GPT、模型加载、预渲染/检索、RANSAC 和优化。
- [ ] 失败样例按模块分类，不只保留成功 gallery。

## 附录 C：坐标系、单位与变换排错

单图 3D 管线最容易出现的并不是模型报错，而是一个模块输出“看起来合法”的坐标，另一个模块按不同约定解释。Diorama 同时接触图像像素、相机坐标、世界坐标、CAD canonical coordinates、render-view coordinates 和 OBB local coordinates，复现时应把每次变换写成显式矩阵并配单元测试。

### C.1 像素与相机坐标

给定内参：

$$
K=\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix},
$$

像素 $(u,v)$ 和 metric depth $d$ 反投影为：

$$
p_{\text{cam}}=dK^{-1}[u,v,1]^T.
$$

需要确认 depth 表示沿 optical axis 的 $z$，还是沿 ray 的欧氏距离；两者在图像边缘会不同。还要确认图像 resize/crop 后 $K$ 是否同步更新。一个常见静默错误是 detector 在缩放图上输出 box，depth 仍按原始分辨率采样，mask 与点云发生整体偏移。

### C.2 相机与世界坐标

外参可能写成 camera-to-world $T_{wc}$，也可能是 world-to-camera $T_{cw}$。二者满足：

$$
T_{cw}=T_{wc}^{-1}.
$$

在 Blender、OpenCV、PyTorch3D 和 CAD 工具间，前向轴与向上轴可能分别采用 `-Z/+Y`、`+Z/-Y` 等约定。不要靠肉眼修改符号；应准备一个已知点、一个已知相机和三根 RGB axes，验证渲染与反投影 round trip。

### C.3 CAD canonical transform

资产库中的 mesh 可能使用米、厘米或无单位；有的模型以底面中心为原点，有的以几何中心为原点；正面方向也不统一。预处理应给每个 asset 固化：

```json
{
  "asset_id": "...",
  "source": "objaverse",
  "source_unit_to_meter": 0.01,
  "canonical_up": "+Y",
  "canonical_front": "+Z",
  "canonical_transform": "4x4 matrix",
  "bbox_min": [-0.5, 0.0, -0.4],
  "bbox_max": [0.5, 1.0, 0.4]
}
```

若 canonical transform 在生成 reference renders 时应用，却在最终导出时遗漏，patch correspondence 可能看似正确，场景中的 CAD 仍会旋转 90°。`semantic_orient_loss` 也依赖 front direction，一处规范错误会被优化器放大。

### C.4 9-DoF 变换次序

应明确使用 column vector 还是 row vector。column-vector 约定下常见写法为：

$$
p_{\text{world}}=R(Sp_{\text{cad}})+t,
$$

对应齐次矩阵：

$$
T=
\begin{bmatrix}
RS&t\\
0&1
\end{bmatrix}.
$$

若先平移再缩放，translation 也会被 scale 改变；若把 non-uniform scale 与 rotation 顺序交换，OBB 方向和 mesh 位置会不一致。复现测试应覆盖：identity、纯平移、绕单轴 90°、uniform scale、non-uniform scale 和组合变换。

### C.5 Support surface 的方向

论文公式直接比较 contact normal 与 support normal，但不同 mesh winding 可能使两个法线指向相反。系统需要统一“从实体向外”还是“从接触面指向支撑物”的约定。若方向相反，正确接触也会得到大 loss。测试可以构造一个单位盒放在地面上，确认 orientation loss、place loss 和 collision loss 都接近预期。

### C.6 数值与可视化双重验证

只看数值不足。建议对每个对象导出六张诊断图：输入 mask、query point cloud、最佳 reference view、patch correspondence、初始 CAD 投影、优化后 CAD 投影；对场景导出 top/front/side 三个正交视图和 support graph。任何坐标错误通常会在这些图里呈现稳定模式：全部对象镜像、尺度统一偏大、垂直轴颠倒或投影随图像位置漂移。

## 附录 D：面向产品原型的验收层级

论文复现、研究 demo 和生产原型需要不同验收标准。可以把 Diorama 式系统分为四级，避免“能跑一张图”被误写成生产可用。

### Level 1：模块烟雾测试

- 一张受控 SSDB 图能完成对象解析、depth、检索、pose 和 layout；
- 所有中间文件存在，无 NaN 和缺失资产；
- 能从三个新视角渲染；
- 能移动或替换一个对象并重新导出。

这一级只能证明依赖安装和数据流正确。

### Level 2：论文协议复现

- 使用固定 SSDB split、相同 GT/estimated depth 条件；
- 重现 Table 1-8 的指标实现和分母；
- 报告多个 seed 或明确确定性；
- 记录与论文差异，包括模型权重、GPT 服务版本和 CAD pool；
- 对无法达到的数字提供模块级误差分析。

这一级才可以称为“论文复现尝试”，仍不代表真实图片可用。

### Level 3：真实场景产品验证

- 建立包含不同房型、光照、相机和遮挡的内部测试集；
- 每张图有对象覆盖、scene graph、可编辑性和人工质量评分；
- 报告成功率、P50/P95 延迟、API 成本、显存峰值和资产库命中率；
- 失败时能降级到部分场景、请求用户确认或返回多个候选；
- 用户修改类别、支撑关系或 CAD 后可以局部重算，不必从头执行。

### Level 4：生产治理

- 输入图片和 CAD 资产许可证可追踪；
- GPT/API 数据保留、隐私和区域合规经过审查；
- 模型、prompt、CAD 库和评测集都具备版本发布与回滚；
- 外部权重或服务不可用时有明确故障模式；
- 生成结果在下游引擎中完成 mesh、材质、碰撞、尺度和安全检查；
- 人工 QA 修改会回流为失败分类和组件评测，而不是被成功 gallery 掩盖。

这套分层强调一个事实：Diorama 是很好的系统研究原型，但从论文指标到产品合同之间还存在环境、可观测性、治理和交互设计工作。把这些缺口写清，不会削弱论文贡献，反而能更准确地说明它为后续工程提供了什么。

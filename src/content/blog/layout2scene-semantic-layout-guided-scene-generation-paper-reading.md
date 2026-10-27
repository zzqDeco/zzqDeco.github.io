---
title: "Layout2Scene 论文精读：3D 语义布局引导、混合场景表示与几何/外观扩散先验"
description: "精读 Layout2Scene 如何通过 3D semantic layout、对象 Gaussian、背景 polygon、语义几何扩散和语义几何外观扩散实现可控 3D 场景生成"
pubDate: "2026-06-30T13:58:31+08:00"
updatedDate: "2026-06-30T13:58:31+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "3D Scene Generation"
  - "Text-to-3D"
  - "Diffusion Models"
  - "3D Gaussian Splatting"
  - "Computer Vision"
draft: false
---

Minglin Chen、Longguang Wang、Sheng Ao、Ye Zhang、Kai Xu 和 Yulan Guo 发表的 **Layout2Scene: 3D Semantic Layout Guided Scene Generation via Geometry and Appearance Diffusion Priors** 直面了当前 text-to-scene 领域的典型痛点：

> 一句简单的文本提示词或许能概括“一间卧室”或“一个现代客厅”的整体氛围，但很难精确约束每个家具对象的三维坐标、尺度、朝向、数量与相对关系。如果完全依赖全局文本驱动 3D 场景生成，极易产生布局混乱、家具重复堆叠、几何拓扑破碎且无法拆解编辑的结果。

Layout2Scene 的解决思路是将输入控制条件从单一文本升级为 **3D semantic layout（三维语义布局）+ scene-level prompt（场景级提示词）**：前者由一组附带语义类别或文本描述的 3D 有向包围盒（bounding boxes）组成，负责精确锚定“什么物体在什么位置”；后者负责指定整体房间类型与风格基调。在此基础上，论文构建了一套对象与背景解耦的混合三维表示——前景对象采用 2D Gaussians 表达复杂局部几何，房间背景采用显式多边形（polygons）加可学习纹理场承载大面积墙地平面；随后利用在室内场景数据上微调的两个 2D 扩散先验模型，按照 **“先精炼三维几何、再生成表面外观”** 的两阶段策略完成场景优化。

本文精读的版本基线为 [arXiv:2501.02519v1](https://arxiv.org/abs/2501.02519v1)（提交于 2025-01-05，正文共 10 页、6 张图、1 张主定量对比表，遵循 **CC BY 4.0** 许可）。截至目前，arXiv 与 Hugging Face 页面均未披露官方开源仓库或项目主页，论文仅注明其系统基于 ThreeStudio 框架开发。因此，本文将围绕原论文的数学推导、混合渲染机制、双扩散先验架构、实验评估及从零复现的工程契约展开深度剖析。

## 1. 论文信息、任务动机与核心问题形式化

### 1.1 论文元数据与一句话贡献

| 项目 | 内容 |
| --- | --- |
| 题名 | Layout2Scene: 3D Semantic Layout Guided Scene Generation via Geometry and Appearance Diffusion Priors |
| 作者 | Minglin Chen, Longguang Wang, Sheng Ao, Ye Zhang, Kai Xu, Yulan Guo |
| 署名机构 | 中山大学（Sun Yat-sen University）、国防科技大学（National University of Defense Technology） |
| 版本基线 | arXiv:2501.02519v1（2025-01-05，10 pages，CC BY 4.0） |
| 实现基座与代码状态 | 论文声明基于 ThreeStudio 框架实现；官方代码与项目页暂未公开 |

一句话概括论文的核心贡献：

> Layout2Scene 利用显式 3D 语义布局消除纯文本场景描述的空间位置歧义，通过“前景对象 2D Gaussian + 背景多边形网格”的混合表示保障对象级可编辑性，并串接语义几何扩散先验（Stage 1）与语义-几何联合外观扩散先验（Stage 2）实现几何与纹理的解耦优化。

这一设计的核心洞见在于**分阶段解耦（decoupled two-stage optimization）**。许多早期的 text-to-3D 或 text-to-scene 方法将几何成形、表面着色、光照协调与语义绑定全部压缩进同一个 SDS 优化目标中：当三维几何尚未稳定时，2D 图像扩散先验往往倾向于用表面色彩和纹理伪影去掩盖几何缺陷（例如在平面上画出虚假的立体凹凸感），甚至反过来拉扯坏局部形状。Layout2Scene 明确将优化过程切分为两个正交阶段：第一阶段（Geometry refinement）仅利用语义引导的法线/深度扩散先验更新高斯几何参数；第二阶段（Appearance generation）在冻结主体几何的前提下，结合语义、法线与深度三路条件生成多视角一致的 RGB 外观。

### 1.2 从 Fig. 1 看 3D 语义布局引导的任务边界

Fig. 1 展示了 Layout2Scene 的标准输入与多模态渲染输出：左侧为输入的 3D 语义布局与简短场景提示词，右侧为沿着室内导航相机轨迹渲染出的 RGB 外观、表面法线（Normal）与深度图（Depth）。

![Fig. 1: Layout2Scene teaser](/images/blog/layout2scene-semantic-layout-guided-scene-generation/layout2scene-fig01-teaser.webp)

*Source: Chen et al., arXiv:2501.02519v1, Fig. 1, CC BY 4.0.*

在图左侧的输入中，3D semantic layout 由一组带颜色编码类别的 3D 包围盒构成（涵盖 sofa、table、lamp、chair、door、cabinet 等室内要素），而全局文本仅需提供 “a living room” 这类高层场景描述。右侧同时展示 RGB、Normal 与 Depth 三种通道具有重要意义：评估一个 3D 场景生成方法不能只看 RGB 视图是否讨喜，还必须检验 Normal 图反映的局部表面朝向是否平滑干净、Depth 图反映的空间前后层次是否真实无断裂。论文在图注中指出，在预训练好扩散先验的前提下，单个场景在一张 NVIDIA V100 GPU 上约需 1.5 小时完成优化，并支持以 30 FPS 的帧率实时渲染。

### 1.3 Text-to-Scene 的三类失效与相关工作技术图谱

论文在引言中将纯文本驱动 3D 场景生成的核心痛点归纳为三类，并给出了针对性的架构回应：

1. **空间可控性缺失（Lack of Controllability）**：自然语言擅长定性描述而拙于定量定位，即使在长提示词中写明空间方位，2D 扩散先验在 3D 蒸馏过程中也很难精确兑现每件家具的坐标与尺寸。Layout2Scene 通过直接引入 3D 语义包围盒作为刚性空间约束解决这一问题。
2. **全局文本与局部视角的语义歧义（Semantic Ambiguity）**：场景提示词描述的是整个房间，但在优化渲染时每次相机视角只能拍到房间的局部角落（例如只看到半张沙发或床角）。让局部视图强行对齐全局房间提示词，极易导致模型在房间各处重复生成多张床或多组沙发（如 Text2Room 的常见失效）。Layout2Scene 通过将三维布局实时渲染为视角对齐的 2D 语义分割图（semantic map）注入扩散模型，使每个局部视角都清楚知晓当前画面像素对应的准确物体类别。
3. **场景表示不可编辑（Non-editability）**：使用单一网格（Mesh）或单片神经辐射场（NeRF）表示整个房间时，家具与墙地面完全耦合，生成后无法单独移动沙发或更换墙纸。Layout2Scene 在表示层将前景对象（2D Gaussians）与房间背景（Polygons + 可学习纹理）彻底分离，天然保留了对象级操作句柄。

将 Layout2Scene 放入整个 3D 场景生成与编辑的技术谱系（包括本站精读过的 CompoNeRF、Blended-NeRF、CasLayout 与 Imaginarium）中进行横向对照，可以清晰定位其技术坐标：

| 研究方向 | 代表工作 | 核心输入 | 核心输出表示 | 与 Layout2Scene 的关系与边界 |
| --- | --- | --- | --- | --- |
| **单物体 Text-to-3D** | DreamFusion, Magic3D, ProlificDreamer | 单物体文本提示 | 单主体 NeRF / Mesh / 3DGS | 奠定了 SDS/VSD 蒸馏范式，但缺乏多物体房间布局控制能力 |
| **文本到房间网格** | Text2Room | 房间级文本提示 | 纹理化室内三维网格（Mesh） | 渐进式视角补全易产生几何破洞与家具重复，缺乏 3D 布局显式控制 |
| **布局引导 NeRF 场景** | Set-the-Scene, CompoNeRF | 3D 包围盒 / 布局 + 局部/全局文本 | 组合式多分支 NeRF 场景 | 同样利用 3D 布局解耦对象，但 NeRF 体渲染较慢且易出现模糊与漂浮伪影 |
| **已有场景局部编辑** | Blended-NeRF, Instruct-NeRF2NeRF | 已训练 NeRF + 3D ROI / 编辑指令 | 局部修改后的 NeRF 场景 | 专注于在已有辐射场上增删改局部物体，而非从布局生成完整新房间 |
| **三维布局生成模型** | CasLayout, DiffuScene, ATISS | 户型边界 / 关系图 | 家具 3D 有向包围盒（OBB）参数 | 负责决定“家具盒子怎么摆”，可直接作为 Layout2Scene 的上游输入生成器 |
| **资产检索式场景生成** | Imaginarium, Holodeck | 文本 / 2D 引导图 + 3D 资产库 | CAD/Mesh 资产实例与位姿列表 | 依托现成工业网格库与物理摆放优化，而 Layout2Scene 直接从零优化神经混合场 |

由此可见，Layout2Scene 与 CasLayout（自动规划室内布局）及 Imaginarium（检索摆放现成 CAD 模型库）各司其职，专注于 **“给定结构化 3D 语义布局后，如何高保真地优化出可实时渲染、几何干净且对象解耦的 3D 场景表示”**。

## 2. 系统总体架构与混合场景表示（Hybrid Representation）

### 2.1 两阶段优化流水线总览（Fig. 2）

给定三维语义布局 $\{\mathcal{B}, \mathcal{T}\}$ 与场景级文本提示 $y$，其中 $\mathcal{B}$ 为 3D 包围盒集合，$\mathcal{T}$ 为各包围盒对应的类别或文本描述集合。第 $i$ 个包围盒由三维旋转矩阵、平移向量与三轴尺度参数化定义：

$$
\mathcal{B}_i = (\mathbf{R}_i, \mathbf{t}_i, \mathbf{s}_i),
\qquad
\mathbf{R}_i \in \mathbb{R}^{3 \times 3}, \quad
\mathbf{t}_i \in \mathbb{R}^{3}, \quad
\mathbf{s}_i \in \mathbb{R}^{3}.
$$

Fig. 2 展示了 Layout2Scene 从布局输入到最终三维场景产出的完整流水线。

![Fig. 2: Overview of Layout2Scene](/images/blog/layout2scene-semantic-layout-guided-scene-generation/layout2scene-fig02-pipeline.webp)

*Source: Chen et al., arXiv:2501.02519v1, Fig. 2, CC BY 4.0.*

整个系统属于典型的 **per-scene optimization（单场景迭代优化）** 架构，其内部由九个功能模块紧密串联而成：

| 系统模块 | 输入信号 | 输出产物 | 核心工程职责 |
| --- | --- | --- | --- |
| **Layout Parser** | 3D boxes $\mathcal{B}$、语义描述 $\mathcal{T}$、场景提示 $y$ | 归一化场景坐标系与语义调色板映射 | 将结构化布局转换为可微渲染器可识别的空间与语义约束 |
| **Object Initializer** | 每个对象的文本描述 $\mathcal{T}_i$ 与包围盒 $\mathcal{B}_i$ | 包围盒内的初始 2D Gaussian 图元集合 | 调用预训练 text-to-3D 模型（如 Shap-E）生成初始点云，避免从纯随机噪声起步 |
| **Background Builder** | 房间边界与墙/地/顶语义标签 | 显式多边形网格（Polygons）+ 可学习纹理场 | 用平整几何图元承载大面积墙面、地板与天花板，杜绝背景破碎 |
| **Hybrid Renderer** | 相机位姿、对象 Gaussians、背景 Polygons | 多通道缓冲图（RGB、Semantic、Normal、Depth） | 执行前景与背景的深度感知融合，向 2D 扩散先验提供对齐条件图 |
| **Camera Sampler** | 布局占据空间 / TSDF 距离场 | 覆盖全屋且防穿模的训练相机位姿分布 | 确保房间每个角落的物体都能均匀获得有效监督梯度 |
| **Geometry Prior** | 语义图 $\mathcal{S}$、提示词 $y$、加噪法线/深度潜在表示 | 几何去噪残差方向 $\epsilon_g$ | 在 Stage 1 提供符合语义布局边界的二维法线与深度先验评分 |
| **Geometry Optimizer** | 渲染法线/深度图 + GSDS 梯度 | 精炼后的高斯几何参数 $\theta_g$ | 更新位置、旋转、尺度与不透明度，消除初始点云的粗糙畸变 |
| **Appearance Prior** | 语义图 $\mathcal{S}$、法线 $\mathcal{N}$、深度 $\mathcal{D}$、提示词 $y$、加噪 RGB | 外观去噪残差方向 $\epsilon_a$ | 在 Stage 2 提供同时契合语义类别与三维几何结构的纹理先验评分 |
| **Appearance Optimizer** | 渲染 RGB 图 + ISD 梯度 + 重建损失 | 高斯颜色参数与背景哈希纹理场 $\theta_a$ | 在锁定几何的前提下生成逼真的家具材质、环境光影与墙地纹理 |

### 2.2 前景对象 2D Gaussians 与规范空间变换

为什么不全盘采用单一 NeRF 或全场景 3DGS？在室内场景中，前景家具（沙发、椅子、台灯）具有复杂的非规则局部拓扑，且需要支持独立平移、旋转和缩放；而背景结构（墙壁、地板、天花板）几何上几乎全是平坦大平面，却承载着高频墙纸或木地板纹理。若全部用自由高斯点表示，大面积平整墙面会消耗海量高斯图元且极易在优化早期出现墙面凹凸破洞；若全部用多边形网格表示，复杂家具的拓扑变形又难以灵活优化。

因此，Layout2Scene 将每个 3D 包围盒内的前景对象表示为规范空间（canonical space）中的一组 **2D Gaussian primitives**：

$$
\mathcal{O}
:=
\left\{
(\mathbf{p}_i, \mathbf{\Sigma}_i, \mathbf{s}_i, \alpha_i, \mathbf{c}_i, \mathbf{t}_i)
\right\}_i,
$$

其中 $\mathbf{p}_i \in \mathbb{R}^3$ 为图元中心位置，$\mathbf{\Sigma}_i \in \mathbb{R}^{3 \times 3}$ 为决定切平面朝向的旋转协方差参数，$\mathbf{s}_i \in \mathbb{R}^2$ 为二维高斯尺度，$\alpha_i \in \mathbb{R}$ 为不透明度，$\mathbf{c}_i \in \mathbb{R}^3$ 为 RGB 颜色，$\mathbf{t}_i \in \mathbb{R}^3$ 则是由分割调色板协议（segmentation color protocol）预定义的固定语义颜色向量。

在初始化阶段，系统利用 Shap-E 等文本到 3D 先验模型根据子文本 $\mathcal{T}_i$ 在规范空间内生成初始三维点云并转化为 2D Gaussians，随后通过第 $i$ 个包围盒的变换参数 $(\mathbf{R}_i, \mathbf{t}_i, \mathbf{s}_i)$ 将局部坐标 $\mathbf{x}_{local}$ 映射到世界坐标系：

$$
\mathbf{x}_{world}
=
\mathbf{R}_i
\left(
\mathbf{s}_i \odot \mathbf{x}_{local}
\right)
+ \mathbf{t}_i,
$$

其中 $\odot$ 表示逐元素乘法。通过 Splatting 光栅化，前景分支可一次性高速渲染出对象的 RGB 图 $\mathcal{I}_o$、不透明度图 $\alpha$、语义图 $\mathcal{S}_o$、法线图 $\mathcal{N}_o$ 与深度图 $\mathcal{D}_o$。更重要的是，后续若要移动或旋转某件家具，只需直接修改对应的 $(\mathbf{R}_i, \mathbf{t}_i, \mathbf{s}_i)$ 即可完成无损空间变换。

### 2.3 背景多边形（Background Polygons）与深度感知混合渲染

针对墙壁（wall）、地面（ground）与天花板（ceiling），系统直接使用由房间边界构建的显式多边形网格（polygons）固定其宏观平面几何，并配合多分辨率哈希编码的可学习纹理场（learnable texture field）表达复杂表面外观。背景分支同样渲染输出对应的 RGB 图 $\mathcal{I}_b$、不透明度 $\alpha_b$、语义图 $\mathcal{S}_b$、法线图 $\mathcal{N}_b$ 与深度图 $\mathcal{D}_b$。

随后，混合渲染器根据每个像素处前景对象深度 $\mathcal{D}_o$ 与背景深度 $\mathcal{D}_b$ 的前后遮挡关系，将任意渲染通道 $\mathcal{R} \in \{\mathcal{I}, \mathcal{S}, \mathcal{N}, \mathcal{D}\}$ 统一合成为完整场景视图：

$$
\mathcal{R} =
\begin{cases}
\alpha \cdot \mathcal{R}_o + (1-\alpha)\cdot \mathcal{R}_b, & \mathcal{D}_o \leq \mathcal{D}_b, \\
\mathcal{R}_b, & \mathcal{D}_o > \mathcal{D}_b.
\end{cases}
$$

当对象位于背景前方（$\mathcal{D}_o \leq \mathcal{D}_b$）时，像素按对象不透明度 $\alpha$ 与背景做 Alpha 混合；当对象被背景墙面遮挡时，则直接取背景值。这一混合机制相当于在渲染管线中内置了一层轻量级的场景图（scene graph）结构：背景多边形锁定了房间尺度的曼哈顿边界与导航基准，前景高斯图元则在各自的包围盒槽位内专注刻画家具细节。

### 2.4 布局感知的相机采样策略（Layout-aware Camera Sampling）

在基于 2D 扩散先验的单场景优化中，三维表示只能从被采样到的相机视角接收梯度：

$$
\mathcal{L}
=
\mathbb{E}_{\pi \sim p(\pi \mid \mathcal{B})}
\left[
\ell(\mathrm{Render}_\theta(\pi), y, \mathcal{S}_\pi)
\right].
$$

单物体生成通常采用围绕世界原点的球面均匀相机采样，但室内场景既没有单一中心主体，又充满墙壁与大件家具——盲目随机采样要么导致相机陷入家具内部（穿模贴脸，渲染出毫无语义的纯色块），要么导致房间边角长期缺乏视线覆盖而欠优化。

为此，Layout2Scene 设计了**布局感知相机采样（layout-aware camera sampling）**，确保采样相机同时满足“覆盖全场景”、“位于所有物体外部且保持安全距离”以及“视线聚焦于物体密集区”三项条件。具体而言，系统首先根据 3D 布局包围盒计算房间空间的截断符号距离场（TSDF），并将其归一化作为相机位置 $\mathbf{x}$ 的采样概率分布：

$$
p(\mathbf{x}) = \mathrm{Norm}(\mathrm{TSDF}(\mathbf{x})).
$$

确定相机位置后，计算从相机指向场景内各个对象中心的单位方向向量，提取其俯仰角（elevation，$\theta_i$）与方位角（azimuth，$\phi_i$）的均值与方差，并按高斯分布采样最终的相机观察朝向：

$$
\theta \sim \mathcal{N}(\mathrm{mean}(\theta_i), \mathrm{var}(\theta_i)),
\qquad
\phi \sim \mathcal{N}(\mathrm{mean}(\phi_i), \mathrm{var}(\phi_i)).
$$

## 3. 第一阶段：语义引导几何扩散与 GSDS 几何精炼

### 3.1 为什么必须先优化几何？（Stage 1 设计动机）

由 Shap-E 等上游模型初始化的对象 2D Gaussians 往往只具备粗糙的体积轮廓，缺乏精细的表面法线与准确的几何结构。如果跳过几何修正直接用 RGB 扩散模型优化颜色，强大的纹理先验会直接在粗糙甚至错误的表面上“画”出逼真的阴影和细节来欺骗 2D 判别器，导致三维几何永远停留在劣质状态。

然而，若直接使用普通的文本条件法线/深度扩散模型（如原始 ND-Diffusion）进行几何蒸馏，模型只看得见全局提示词（如 “a bedroom”），完全不知道当前渲染视图中各个像素应当遵守怎样的 3D 布局边界，从而轻易把包围盒内的物体拉扯变形或移位。因此，Stage 1 的核心任务是构建并调用一个**受实时渲染语义图 $\mathcal{S}$ 强约束的几何扩散先验**。

### 3.2 语义引导几何扩散模型（Semantic-guided Geometry Diffusion）

Layout2Scene 以预训练的法线-深度联合扩散模型 **ND-Diffusion** 为基座（其输入输出为拼接后的表面法线图 $\hat{\mathcal{N}}$ 与逆深度图 $\hat{\mathcal{D}}$ 的 VAE 潜在表示 $z_t$）。为了注入布局语义控制，论文在其 U-Net 之上挂载了一个可训练的 ControlNet 式语义条件编码器 $\mathcal{E}_c$。给定渲染出的二维语义分割图 $\mathcal{S}$ 与场景提示词 $y$，模型首先分别提取语义多尺度特征与原始扩散编码器特征，并将二者相加后送入 U-Net 解码器 $\mathcal{D}_{nd}$ 预测噪声：

$$
\{f_i^c\}_i = \mathcal{E}_c(\mathcal{S}),
\qquad
\{f_i\}_i = \mathcal{E}_{nd}(\hat{\mathcal{N}}, \hat{\mathcal{D}}; y),
$$

$$
\epsilon_g(\hat{\mathcal{N}}, \hat{\mathcal{D}}; y, \mathcal{S})
=
\mathcal{D}_{nd}(\{f_i^c + f_i\}_i).
$$

在预训练该几何先验时，冻结 ND-Diffusion 主干，仅在室内 RGB-D 数据集（SunRGBD）上通过标准去噪均方误差目标训练语义编码器分支：

$$
\mathcal{L}_{GLDM}
:=
\mathbb{E}_{x,y,\epsilon,t}
\left[
\left\|
\epsilon -
\epsilon_g(z_t; y,t,\mathcal{S})
\right\|_2^2
\right].
$$

### 3.3 几何分数蒸馏采样（GSDS）梯度推导

获得训练完毕的语义几何扩散先验 $\epsilon_g$ 后，Stage 1 采用 **几何分数蒸馏采样（Geometry Score Distillation Sampling，GSDS）** 来优化场景的三维几何参数：

$$
\nabla_{\theta_g}\mathcal{L}_{GSDS}
:=
\mathbb{E}_{t,\epsilon}
\left[
\omega(t)
\left(
\epsilon_g(z_t; y,t,\mathcal{S}) - \epsilon
\right)
\frac{\partial x}{\partial \theta_g}
\right],
$$

其中 $x$ 为当前相机视角下渲染出的法线图与深度图经 VAE 编码后的潜在表示，$\omega(t)$ 为时间步加权函数。至关重要的是，Stage 1 被允许更新的参数集合 $\theta_g$ **仅限于对象高斯图元的纯几何属性**（不含任何颜色或纹理参数）：

$$
\theta_g =
\left\{
(\mathbf{p}_i, \mathbf{\Sigma}_i, \mathbf{s}_i, \alpha_i)
\right\}_i.
$$

将 GSDS 与经典的 DreamFusion SDS 对比可以看出其针对性改进：

| 对比维度 | 经典 DreamFusion SDS | Layout2Scene GSDS（Stage 1） |
| --- | --- | --- |
| **2D 蒸馏模态** | 渲染 RGB 彩色图像的加噪去噪残差 | 渲染 Normal + Depth 几何缓冲图的去噪残差 |
| **空间条件控制** | 仅依赖全局文本与粗略视角方位词 | 逐像素对齐的实时渲染语义图 $\mathcal{S}$ + 场景文本 $y$ |
| **3D 更新参数** | 几何密度与表面颜色同时耦合更新 | 严格冻结颜色，仅更新高斯位置、旋转、尺度与不透明度 $\theta_g$ |
| **核心收益** | 链路简单，但易发生几何/纹理纠缠与布局漂移 | 在着色前锁死符合语义包围盒的干净表面法线与空间深度层次 |

## 4. 第二阶段：语义-几何联合引导外观扩散与 ISD 优化

### 4.1 多条件外观控制与 Semantic-Geometry Guided Diffusion

经过 Stage 1 的 5,000 步几何精炼后，场景中各家具的三维形状、表面法线与空间深度均已收敛稳定。进入 Stage 2（Appearance generation）后，系统的目标是生成逼真且符合室内光照逻辑的 RGB 外观。此时如果只向 Stable Diffusion 输入文本提示词或单一语义图，生成的二维纹理很容易忽略底层已经确立的三维凹凸起伏与前后遮挡关系（例如在沙发转角处画出扁平错位的阴影）。

为此，Layout2Scene 提出了 **语义-几何联合引导扩散模型（Semantic-Geometry Guided Diffusion）**，同时将四路互补条件注入外观生成过程：

| 输入条件 | 提供的约束维度 | 缺失该条件时的退化后果 |
| --- | --- | --- |
| **场景文本提示 $y$** | 整体房间类型、装修风格与材质基调（如现代风卧室） | 外观缺乏统一的审美方向与全局色调协调 |
| **语义分割图 $\mathcal{S}$** | 每个像素所属的精确家具类别边界 | 跨物体纹理串位，或将家具画成错误的类别 |
| **表面法线图 $\mathcal{N}$** | 局部表面的朝向变化、折角与精细几何凹凸 | 表面材质纹理与真实三维形状脱节，光影方向错乱 |
| **空间深度图 $\mathcal{D}$** | 全局空间远近层次与物体间的前后遮挡关系 | 远近尺度感失真，交界处阴影与透视关系异常 |

在网络架构上，论文采用三个独立的 ControlNet 编码器 $\mathcal{E}_s, \mathcal{E}_n, \mathcal{E}_d$ 分别提取语义图 $\mathcal{S}$、法线图 $\mathcal{N}$ 与深度图 $\mathcal{D}$ 的多尺度条件特征：

$$
\{f_i^s\}_i = \mathcal{E}_s(\mathcal{S}), \qquad
\{f_i^n\}_i = \mathcal{E}_n(\mathcal{N}), \qquad
\{f_i^d\}_i = \mathcal{E}_d(\mathcal{D}).
$$

随后将这三路条件特征与冻结的 Stable Diffusion U-Net 编码器特征 $\{f_i\}_i$ 逐层相加，送入解码器 $\mathcal{D}_{sd}$ 预测 RGB 潜在空间的噪声：

$$
\epsilon_a(\hat{\mathcal{I}}; y,\mathcal{S},\mathcal{N},\mathcal{D})
=
\mathcal{D}_{sd}
\left(
\{f_i^s + f_i^n + f_i^d + f_i\}_i
\right).
$$

该多条件外观先验同样在 SunRGBD 数据集上通过标准扩散噪声预测损失进行微调，使其深度适配室内真实场景的光照与色彩分布：

$$
\mathcal{L}_{ALDM}
:=
\mathbb{E}_{x,y,\epsilon,t}
\left[
\left\|
\epsilon -
\epsilon_a(z_t; y,t,\mathcal{S},\mathcal{N},\mathcal{D})
\right\|_2^2
\right].
$$

### 4.2 不变分数蒸馏（ISD）与重建正则化

在 Stage 2 的三维外观参数优化中，为了克服传统 SDS 在高引导权重下容易出现的色彩过饱和与多视角闪烁伪影，论文引入了 **不变分数蒸馏（Invariant Score Distillation，ISD）**。给定待优化的外观参数集 $\theta_a$（仅包含前景高斯图元的颜色参数 $\{\mathbf{c}_i\}_i$ 与背景多边形的可学习哈希纹理场参数），ISD 的梯度定义为：

$$
\nabla_{\theta_a}\mathcal{L}_{ISD}
:=
\mathbb{E}_{t,\epsilon}
\left[
\omega(t)
\left(
\lambda(t)\delta_{inv} + \omega\delta_{cls}
\right)
\frac{\partial x}{\partial \theta_a}
\right],
$$

其中包含两个互补的残差分量——用于稳定跨时间步去噪轨迹的**不变残差项 $\delta_{inv}$**（利用 DDIM 确定性反演从 $z_t$ 估计出较低噪声状态 $z_{t-c}$ 后计算差分），以及提供无分类器引导方向的**类别条件项 $\delta_{cls}$**：

$$
\delta_{inv}
:=
\epsilon_a(z_{t-c}; y,t-c)
-
\epsilon_a(z_t; y,t),
\qquad
\delta_{cls}
:=
\epsilon_a(z_t; y,t)
-
\epsilon_a(z_t; \emptyset,t).
$$

此外，为了进一步抑制相邻迭代与多视角之间的颜色漂移，论文额外引入了图像空间的 $L_2$ 重建约束项 $\mathcal{L}_{recon} = \|I-\hat{I}\|_2^2$，构成 Stage 2 的最终优化目标：

$$
\mathcal{L}_A
=
\mathcal{L}_{ISD}
+ \gamma \mathcal{L}_{recon}.
$$

由于 Stage 2 彻底冻结了 $\theta_g$，外观扩散先验无论如何更新都不会破坏 Stage 1 已经建立好的三维法线与深度结构，从而实现了几何与外观的安全接力。

## 5. 先验训练数据集构建与两阶段实现细节

### 5.1 基于 SunRGBD、BLIP-2 与 StableNormal 的数据管线

训练上述两个室内领域扩散先验（$\epsilon_g$ 与 $\epsilon_a$）需要大规模配对的“文本 + 语义图 + 法线图 + 深度图 + RGB 图”数据集。论文没有依赖昂贵且稀缺的合成 3D 室内网格资产，而是巧妙地对包含超过 10,000 张真实室内 RGB-D 图像的 **SunRGBD** 数据集进行了多模态伪标签扩充：

1. **原始传感器真值**：直接读取 SunRGBD 提供的真实 RGB 图像、二维语义分割标注（semantic map）以及深度图（转换为逆深度 inverse depth）。
2. **场景文本生成（BLIP-2）**：将 RGB 图像输入视觉语言大模型 **BLIP-2**，通过提问模板 `"what is the type of the scene?"` 自动生成场景级文本描述 $y$。
3. **表面法线估计（StableNormal）**：利用单目法线估计模型 **StableNormal** 从 RGB 图像中提取高质量伪真值法线图（normal map）。

这一务实的数据构建管线大幅降低了训练门槛，但也在客观上引入了三层数据域边界：首先，BLIP-2 针对固定问题生成的文本多为粗粒度的房间类型描述（如 “a bedroom”），对细粒度材质或特定艺术风格的覆盖较弱；其次，StableNormal 的预测误差会作为监督信号进入几何先验 $\epsilon_g$；最后，用户输入的 3D 语义布局类别必须严格对齐 SunRGBD 的语义分割调色板协议，否则在条件注入时会出现类别语义失配。

### 5.2 训练开销拆解与 Gaussian 参数分组学习率

阅读论文的算力成本时，必须严格区分**“先验模型的底座微调成本”**与**“单场景的生成优化成本”**：

| 阶段划分 | 具体任务 | 数据集 / 输入 | 迭代步数 | 硬件配置与耗时 |
| --- | --- | --- | --- | --- |
| **底座预训练 1** | 训练语义几何扩散先验 $\epsilon_g$ | SunRGBD（10k+ 配对帧） | 120,000 steps | 4 $\times$ NVIDIA V100（Batch = 16/GPU） |
| **底座预训练 2** | 训练语义-几何外观先验 $\epsilon_a$ | SunRGBD（10k+ 配对帧） | 120,000 steps | 4 $\times$ NVIDIA V100（Batch = 2/GPU） |
| **单场景生成 Stage 0** | Shap-E 对象点云与混合场景初始化 | 单个 3D 语义布局 + 提示词 | 一次性前向初始化 | 单卡 NVIDIA V100（约数分钟） |
| **单场景生成 Stage 1** | GSDS 几何参数精炼（更新 $\theta_g$） | 采样视角渲染 $\mathcal{S},\mathcal{N},\mathcal{D}$ | 5,000 steps | 单卡 NVIDIA V100（约 0.5 小时） |
| **单场景生成 Stage 2** | ISD 外观与背景纹理生成（更新 $\theta_a$） | 采样视角渲染全通道缓冲 | 10,000 steps | 单卡 NVIDIA V100（约 1.0 小时） |

可以看到，论文宣称的“1.5 小时生成一个场景”建立在两个 120k 步 ControlNet 先验已经训练完毕的前提之上。此外，基于 ThreeStudio 框架并使用 Adam 优化器时，不同物理含义的高斯与背景参数对学习率极为敏感，必须按阶段与参数组精细隔离：

| 参数组 | 激活优化阶段 | 学习率敏感度与失控表现 |
| --- | --- | --- |
| **高斯中心位置 $\mathbf{p}_i$** | Stage 1（Geometry） | 学习率过高会导致点云剧烈漂移、穿墙或逃逸出所属的 3D 包围盒 |
| **旋转协方差 $\mathbf{\Sigma}_i$** | Stage 1（Geometry） | 决定 2D 高斯切面朝向，更新过猛会导致法线图破碎成杂乱刺状伪影 |
| **二维尺度 $\mathbf{s}_i$** | Stage 1（Geometry） | 过大会导致家具表面膨胀糊成一团，过小则会产生针孔状几何空洞 |
| **不透明度 $\alpha_i$** | Stage 1（Geometry） | 过高会形成实心悬浮色块阻挡视线，过低则导致物体半透明虚化 |
| **高斯颜色 $\mathbf{c}_i$** | Stage 2（Appearance） | 更新步长过大时，不同相机视角下的 ISD 梯度拉扯会引发多视角色彩闪烁 |
| **背景哈希纹理场** | Stage 2（Appearance） | 学习率过高极易在平整墙面与地板上刷出高频棋盘格噪点或色斑 |

## 6. 实验评估与消融分析

### 6.1 定性对比与多场景生成展示（Fig. 3 & Fig. 4）

Fig. 3 将 Layout2Scene 与四类代表性基线在卧室（bedroom）与客厅（living room）场景上进行了定性对比，涵盖三类纯文本驱动方法（**DreamFusion**、**ProlificDreamer**、**Text2Room**）以及一类同样采用布局引导的组合式 NeRF 方法（**Set-the-Scene**）。

![Fig. 3: Qualitative comparisons](/images/blog/layout2scene-semantic-layout-guided-scene-generation/layout2scene-fig03-qualitative-comparison.webp)

*Source: Chen et al., arXiv:2501.02519v1, Fig. 3, CC BY 4.0.*

从 Fig. 3 可以清晰观察到两类方法的分水岭：DreamFusion 与 ProlificDreamer 缺乏房间级空间结构约束，生成的场景几何扭曲、墙地边界模糊；Text2Room 虽能拼接出房间网格，但存在明显的几何破洞且家具布局失控；Set-the-Scene 虽引入了局部对象布局，但受限于 NeRF 表示与耦合优化，渲染画面遍布雾状模糊（blur）与漂浮伪影（floaters）。相比之下，Layout2Scene 不仅严格遵循了指定的家具空间排布，而且得益于背景多边形与 Stage 1 几何精炼，其 RGB 视图干净锐利，法线图与深度图也展现出规整的室内几何结构。

Fig. 4 进一步展示了 Layout2Scene 在更多客厅与浴室（bathroom）布局下的多视角生成结果，每组左侧给出俯视（bird-eye-view）三维语义布局，右侧联合呈现多视角 RGB、Normal 与 Depth 渲染图。

![Fig. 4: Various scenes](/images/blog/layout2scene-semantic-layout-guided-scene-generation/layout2scene-fig04-various-scenes.webp)

*Source: Chen et al., arXiv:2501.02519v1, Fig. 4, CC BY 4.0.*

这种“RGB + 法线 + 深度”三通道并列展示有力支撑了论文的核心论点：生成结果在视觉纹理之下真正具备了平滑一致的平面墙体法线、清晰的家具轮廓边界以及连续合理的前后空间深度。

### 6.2 定量结果解读与三维评测指标体系扩展（Table 1）

Table 1 汇总了各方法在随机渲染的 120 个相机视角下的 **CS（CLIP Score）**、**IS（Inception Score）**、单场景优化耗时（**Tr.Time**）与实时渲染帧率（**FPS**）。

![Table 1: Quantitative comparison](/images/blog/layout2scene-semantic-layout-guided-scene-generation/layout2scene-table01-quantitative-comparison.webp)

*Source: Chen et al., arXiv:2501.02519v1, Table 1, CC BY 4.0.*

| 方法 | CS (CLIP Score) $\uparrow$ | IS (Inception Score) $\uparrow$ | 单场景训练耗时 (Tr.Time) $\downarrow$ | 渲染帧率 (FPS) $\uparrow$ |
| --- | ---: | ---: | --- | ---: |
| **DreamFusion** | 18.92 | 2.31 | 1.0 hour | 1.4 |
| **ProlificDreamer** | 18.51 | 2.19 | 2.0 hours | 2.3 |
| **Text2Room** | 22.42 | 3.21 | 0.3 hours | - |
| **Set-the-Scene** | 19.24 | 2.77 | 4.0 hours | 0.3 |
| **Layout2Scene (Ours)** | **25.69** | **3.51** | **1.5 hours** | **30.0** |

数据显示，Layout2Scene 在语义对齐度（CS 达 25.69）和图像质量多样性（IS 达 3.51）上显著超越所有基线；同时相比同属布局引导路线的 Set-the-Scene，其优化时间从 4 小时缩短至 1.5 小时，渲染速度更是凭借 2D Gaussian 与多边形光栅化从 0.3 FPS 跃升至 30 FPS（提升 100 倍）。

不过，审视 Table 1 时也应当清醒认识到：**CS（CLIP Score）衡量的是二维渲染图与全局文本的语义余弦相似度，IS（Inception Score）衡量的是二维图像的分类置信度与多样性，二者本质上都是 2D 图像代理指标。** 如果在后续工程落地中要真正检验 3D 场景的可用性，还必须在评测管线中补齐三维空间专属指标——包括**布局贴合度（Layout Adherence，检查高斯图元是否溢出包围盒）**、**对象数量准确率（Object Count Accuracy）**、**穿模碰撞率（Collision Rate）**、**跨视角特征一致性（Multi-view Consistency）**以及**编辑隔离稳定性（Edit Stability，移动单个物体后非目标区域的像素变化量）**。

### 6.3 几何先验与外观先验的消融实验（Fig. 5 & Fig. 6）

Fig. 5 与 Fig. 6 分别验证了 Stage 1 语义几何扩散先验与 Stage 2 语义-几何外观扩散先验的不可替代性。

![Fig. 5: Geometry prior ablation](/images/blog/layout2scene-semantic-layout-guided-scene-generation/layout2scene-fig05-geometry-prior-ablation.webp)

*Source: Chen et al., arXiv:2501.02519v1, Fig. 5, CC BY 4.0.*

![Fig. 6: Appearance prior ablation](/images/blog/layout2scene-semantic-layout-guided-scene-generation/layout2scene-fig06-appearance-prior-ablation.webp)

*Source: Chen et al., arXiv:2501.02519v1, Fig. 6, CC BY 4.0.*

- **几何先验消融（Fig. 5）**：当移除 Stage 1 的语义引导几何扩散先验（w/o geometry diffusion prior）时，渲染出的法线图充满高频颗粒噪声，家具表面凹凸不平，深度图边缘也十分毛糙；引入几何先验后（Ours），沙发与背景墙面的法线变得干净平滑，深度层次边界清晰锐利。这证明仅靠 Shap-E 的粗略初始化远不足以支撑高质量室内几何，Stage 1 的 GSDS 精炼起到了决定性的整形作用。
- **外观先验消融（Fig. 6）**：在给定相同语义图、法线图与深度图的条件下，对比直接使用未微调的原始多路 ControlNet 与论文在 SunRGBD 上微调后的 Semantic-Geometry Guided Diffusion。原始 ControlNet 生成的室内图像色调昏暗、材质单调且存在明显伪影；而经过室内域微调的外观先验不仅光照自然明亮，还能根据同一套几何布局生成风格多样的逼真家具材质。

## 7. 工程复现架构设计与风险排查

### 7.1 最小复现契约、核心伪代码与分阶段路线图

由于官方暂未开源代码，若要基于 ThreeStudio 与 2D Gaussian Splatting 从零复现 Layout2Scene，首先需要定义一套严格的**场景布局输入数据契约（Schema）**，以确保上游布局在两阶段优化中不会被隐式篡改：

```json
{
  "scene_id": "layout2scene_demo_001",
  "scene_prompt": "a modern living room",
  "unit": "meter",
  "room": {
    "floor_polygon": [[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0]],
    "wall_height": 2.8
  },
  "objects": [
    {
      "id": "sofa_01",
      "category": "sofa",
      "text": "a gray fabric sofa",
      "translation": [2.5, 0.8, 0.45],
      "rotation_yaw": 0.0,
      "size": [2.2, 0.9, 0.9]
    },
    {
      "id": "table_01",
      "category": "coffee table",
      "text": "a small wooden coffee table",
      "translation": [2.5, 1.8, 0.25],
      "rotation_yaw": 0.0,
      "size": [1.0, 0.6, 0.5]
    }
  ]
}
```

基于该数据契约，整个单场景两阶段生成管线的核心伪代码闭环如下：

```python
# Stage 0: 混合场景初始化
for obj in layout.objects:
    point_cloud = shap_e_text_to_3d(obj.text)
    obj_gaussians[obj.id] = init_2d_gaussians(point_cloud, obj.box, obj.category_color)

background = build_room_polygons_and_hash_texture(layout.room)
scene = HybridScene(obj_gaussians, background)

# Stage 1: 语义引导几何精炼 (5,000 steps，仅更新几何参数 theta_g)
for step in range(5000):
    camera = sample_layout_aware_camera(layout)
    semantic, normal, depth = scene.render_geometry_buffers(camera)
    grad_g = compute_gsds_gradient(geometry_prior, scene_prompt, semantic, normal, depth)
    scene.step_geometry_parameters(grad_g)

# Stage 2: 语义-几何联合外观生成 (10,000 steps，冻结 theta_g，仅更新外观参数 theta_a)
for step in range(10000):
    camera = sample_layout_aware_camera(layout)
    rgb, semantic, normal, depth = scene.render_all_buffers(camera)
    loss_a = compute_isd_and_recon_loss(
        appearance_prior, scene_prompt, semantic, normal, depth, rgb
    )
    scene.step_appearance_parameters(loss_a)
```

在推进工程复现时，切忌试图一次性联调所有模块，建议拆解为三个递进里程碑实施：

1. **里程碑一：打通混合表示渲染与布局相机采样底座**。先完成 `SceneLayout` 解析、SunRGBD 语义调色板对齐、房间背景多边形构建以及 Shap-E 初始 2D Gaussians 的空间仿射变换（$\mathbf{R}_i, \mathbf{t}_i, \mathbf{s}_i$）；随后实现基于 TSDF 的防穿模相机采样器，并验证混合渲染器能否在多视角下稳定输出深度对齐的 RGB、Semantic、Normal 与 Depth 四通道缓冲图。
2. **里程碑二：完成 SunRGBD 双扩散先验微调与分阶段蒸馏闭环**。使用 BLIP-2 与 StableNormal 处理 SunRGBD 数据集，分别训练基于 ND-Diffusion 的语义几何 ControlNet（$\epsilon_g$）和基于 Stable Diffusion 的三路条件外观 ControlNet（$\epsilon_a$）；接着在 ThreeStudio 中接入 Stage 1 的 GSDS 梯度回传与 Stage 2 的 ISD + $L_2$ 重建优化，严格校验两阶段的参数冻结边界。
3. **里程碑三：建立多维评估、对象级编辑与回归测试集**。除复现 120 视角的 CS/IS 指标与 FPS 基准外，需针对单个包围盒执行平移、旋转、删除与替换测试，验证混合表示的非破坏性局部编辑能力，并按房间类型与遮挡密度建立失败案例回归库。

### 7.2 常见实现风险诊断与适用边界

在联调上述管线时，绝大多数异常表现都源于模块间的数据接口错位或学习率失衡，可通过下表快速定位排查：

| 典型异常现象 | 潜在根因分析 | 工程排查手段 |
| --- | --- | --- |
| **家具图元漂出所属 3D 包围盒** | 高斯中心 $\mathbf{p}_i$ 学习率过高、局部到世界坐标变换矩阵错误，或相机视角严重偏置 | 在渲染图中叠加绘制线框包围盒与高斯质心轨迹，对超出边界的图元施加截断或软惩罚 |
| **法线/深度正常但 RGB 类别画错** | 渲染语义图的调色板 RGB 值与训练先验时的 SunRGBD 类别颜色映射不一致 | 单独导出 $\mathcal{S}$ 缓冲图，逐像素核对颜色编码表与外观 ControlNet 输入通道 |
| **单视角纹理惊艳但转视角闪烁** | Stage 2 的 ISD 步长过大、重建损失权重 $\gamma$ 过小，或局部角落相机采样覆盖率不足 | 沿着固定环绕轨迹导出连续视频，检查高斯颜色更新幅度与 TSDF 相机采样热力图 |
| **背景墙面出现高频噪点或明暗异常** | 背景哈希纹理场学习率过高，或外观 ControlNet 的多路条件引导权重（guidance scale）失衡 | 屏蔽前景对象单独渲染背景多边形，参考 Fig. 6 逐一排查三路 ControlNet 的特征加权系数 |

由此也可以清晰划定 Layout2Scene 的适用边界：它非常适合**已知粗略室内三维布局、追求快速生成 30 FPS 可交互漫游视图与多通道几何缓冲的概念设计探索**；但若业务场景要求“仅凭一句话自动规划全屋户型布局”、“严格满足刚体物理碰撞与人体工学动线”，或者“直接导出拓扑干净的工业级带 UV 网格与 PBR 贴图资产”，则需结合上游布局规划器（如 CasLayout）或转向基于 CAD 资产库检索与物理引擎优化的管线（如 Imaginarium）。

## 8. 局限性批判与总结

### 8.1 核心局限性的四个维度审视

站在严谨的学术与工业审视角度，Layout2Scene 的局限性可归纳为以下四个核心维度：

- **开源缺失与高昂的底座先验预训练门槛**：论文暂未公开官方源码与预训练模型权重，且正文未完整披露 GSDS/ISD 的时间步调度函数 $\omega(t)$、DDIM 步幅 $c$ 及各参数组具体学习率数值。更重要的是，“单场景 1.5 小时生成”建立在预先使用 4 张 V100 训练两个 120k 步扩散先验的基础之上，从零复现整套底座的算力与数据清洗成本相当可观。
- **上游布局先验假设与物理功能约束缺失**：系统假定输入的 3D 语义布局天然合理，并未内置任何物理引擎或功能语义约束。如果上游给定的布局存在穿模、悬空，或者违背室内设计常识（例如椅子背对书桌、床体堵死房门、台灯悬浮在茶几上方），Layout2Scene 只能忠实地在错误的位置上渲染出对应的家具，无法自动纠正空间拓扑与支撑关系。
- **室内数据域偏置与曼哈顿背景表达受限**：两个核心扩散先验完全在 SunRGBD 室内数据集（辅以 BLIP-2 粗粒度描述与 StableNormal 伪法线）上微调，这使模型能力被锁定在卧室、客厅、浴室等常规室内场景中，难以泛化至室外街景、自然地貌或幻想风格场景。同时，采用简单多边形表示背景默认了房间符合平整的曼哈顿世界假设，面对曲面穹顶、旋转楼梯或复杂开放式建筑结构时表达能力受限。
- **上游单物体初始化依赖与二维评测盲区**：前景高斯图元的初始形态高度依赖 Shap-E 等 text-to-3D 模型。一旦上游模型对特定复杂家具生成的初始点云拓扑严重残缺，Stage 1 的二维法线/深度蒸馏很难将其彻底扭转。此外，论文实验仅采用 CS 与 IS 两项 2D 图像指标，缺乏对三维网格几何精度、布局边界贴合率以及多步连续编辑后场景退化程度的定量验证。

### 8.2 推荐阅读路径与结论

若希望快速抓牢本文脉络，建议按以下顺序精读原论文：先看 Abstract、Fig. 1 与 Fig. 2，建立“3D 语义布局输入 + 前景高斯/背景多边形混合表示 + 先几何后外观两阶段优化”的全局图景；接着依次阅读 Sec. 3.1（混合渲染与 TSDF 相机采样）、Sec. 3.2（语义几何扩散与 GSDS）与 Sec. 3.3（三路条件外观扩散与 ISD），将两阶段冻结与更新的参数集严格对齐；最后结合 Fig. 3–6 与 Table 1 审视其相对基线的视觉与帧率优势。

总结而言，Layout2Scene 的核心贡献在于打破了以往 text-to-scene 将所有控制信息塞入一段文本、将所有三维属性塞入单一损失函数的粗放范式。通过**以结构化 3D 语义布局接管空间定位、以对象/背景混合表示兼顾渲染效率与可编辑性、以语义-几何多条件双扩散先验实现几何与外观的分阶段解耦蒸馏**，它为可控 3D 场景生成提供了一套逻辑严密且切合三维创作规律的系统设计范本。

## References

- Chen, Minglin, Longguang Wang, Sheng Ao, Ye Zhang, Kai Xu, and Yulan Guo. [Layout2Scene: 3D Semantic Layout Guided Scene Generation via Geometry and Appearance Diffusion Priors](https://arxiv.org/abs/2501.02519v1). arXiv:2501.02519v1, 2025.
- arXiv HTML v1: [https://arxiv.org/html/2501.02519v1](https://arxiv.org/html/2501.02519v1)
- arXiv PDF: [https://arxiv.org/pdf/2501.02519](https://arxiv.org/pdf/2501.02519)
- Hugging Face paper metadata: [https://huggingface.co/papers/2501.02519](https://huggingface.co/papers/2501.02519)
- Guo et al. [ThreeStudio: A Unified Framework for 3D Content Generation](https://github.com/threestudio-project/threestudio).
- Zhang et al. [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543), ControlNet.
- Poole et al. [DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988), ICLR 2023.
- Wang et al. [ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation](https://arxiv.org/abs/2305.16213), NeurIPS 2023.
- Hollein et al. [Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models](https://arxiv.org/abs/2303.11989), ICCV 2023.
- Cohen-Bar et al. [Set-the-Scene: Global-Local Training for Generating Controllable NeRF Scenes](https://arxiv.org/abs/2303.13450), ICCV 2023.
- Song et al. [Sun RGB-D: A RGB-D Scene Understanding Benchmark Suite](https://rgbd.cs.princeton.edu/).
- Li et al. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597).

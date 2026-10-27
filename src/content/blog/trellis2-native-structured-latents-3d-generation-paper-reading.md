---
title: "TRELLIS.2 论文精读：O-Voxel、Sparse Compression VAE 与 4B 参数 3D 生成模型"
description: "精读 Native and Compact Structured Latents for 3D Generation 如何通过 O-Voxel、稀疏压缩 VAE、PBR 材质建模与大规模 flow matching 实现高质量 3D 资产生成"
pubDate: "2026-07-07T10:59:40+08:00"
updatedDate: "2026-07-07T10:59:40+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "3D Generation"
  - "Computer Vision"
  - "Generative AI"
  - "3D Assets"
  - "Code Reading"
draft: false
---

在三维内容生成领域，模型参数量的扩张往往受制于底层三维表示的保真度上限：当隐式场（SDF/Occupancy）强行将开放曲面封闭化、或将 PBR 材质拆解为事后多视角贴图烘焙时，生成主干再庞大也难以直接产出结构完整且带物理材质的原生三维资产。Jianfeng Xiang、Xiaoxue Chen、Sicheng Xu、Ruicheng Wang、Zelong Lv、Yu Deng、Hongyuan Zhu、Yue Dong、Hao Zhao、Nicholas Jing Yuan 与 Jiaolong Yang 在技术报告 **Native and Compact Structured Latents for 3D Generation** 中提出的 **TRELLIS.2**，选择从底层三维表示切入重构整套生成栈。

该工作提出了一种无需连续场假设（field-free）的稀疏体素结构 **O-Voxel**，在同一套稀疏空间坐标中联合编码任意拓扑几何与 PBR 材质属性；随后利用全稀疏卷积构建的 **Sparse Compression VAE（SC-VAE）** 将高分辨率资产压缩至紧凑的结构化潜空间（SLat）；最终在该潜空间上训练总计约 4B 参数的三阶段 Flow Matching 生成模型，依次生成稀疏结构骨架、几何潜变量与材质潜变量。本文将沿着“原生表示 → 稀疏压缩 → 流匹配生成 → 算子系统 → 实验验证 → 源码实现”的脉络展开精读，并结合官方开源仓库 `microsoft/TRELLIS.2` 进行静态代码对照（本文聚焦论文与源码架构解析，不涉及本地实机训练或推理复现）。

## 1. 问题背景与系统概览

### 1.1 论文信息与核心设计哲学

| 项目 | 内容 |
| --- | --- |
| 论文标题 | Native and Compact Structured Latents for 3D Generation |
| 项目名称 | TRELLIS.2 |
| 预印本链接 | [arXiv:2512.14692v1](https://arxiv.org/abs/2512.14692)（[PDF](https://arxiv.org/pdf/2512.14692)，共 24 页） |
| 作者团队 | Jianfeng Xiang 等（Microsoft Research 等） |
| 官方资源 | [项目主页](https://microsoft.github.io/TRELLIS.2/) / [GitHub: microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) / [HF 权重: TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) |
| 核心关键词 | O-Voxel, Sparse Compression VAE, PBR Materials, Flow Matching, Sparse Convolution |

TRELLIS.2 的核心突破在于打通了“原生资产双向转换 — 高倍率稀疏潜空间压缩 — 大规模流匹配生成”的闭环。传统管线常在几何拓扑与材质属性上做出妥协，而 TRELLIS.2 的设计将表示、压缩与生成紧密耦合：O-Voxel 摆脱了水密（watertight）流形假设与连续标量场水平集（level set）提取，直接在稀疏体素上记录局部对偶顶点（dual vertex）、边相交标志与四边形切分权重，同时在同源体素上存储 base color、metallic、roughness 与 opacity 四类 PBR 通道。在此基础上，SC-VAE 将高分辨率稀疏体素压缩为有限数量的稀疏 token，使得 4B 规模的生成模型得以直接学习接近原生三维资产的联合分布。

![Fig. 1: TRELLIS.2 teaser](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig01-teaser.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 1, arXiv non-exclusive distribution license.*

Fig. 1 浓缩展示了这套架构的三方面特性。左侧为 $1536^3$ 高分辨率涡轮引擎资产的重建效果，包含完整渲染、法线图与多级分辨率细节预览，直观体现了 O-Voxel 对薄片、孔洞、锐利边缘及复杂内部结构的保留能力。中间展示了复杂机械角色的生成结果及其对应的 alpha、roughness、metallic 和 base color 通道，这些 PBR 属性均由潜空间直接解码产出。右侧对比了潜空间的紧凑度（横轴为 token 数，纵轴为重建指标），由于三维生成的计算瓶颈往往取决于每个资产占用的潜变量 token 数量与特征维度，TRELLIS.2 在更少 token 预算下实现了更高的重建保真度。在运行效率上，官方 README 披露 H100 单卡生成 $512^3$、$1024^3$ 与 $1536^3$ 资产分别约需 3 秒、17 秒与 60 秒（论文实验表格则另行报告了 A100 上的解码耗时，评估硬件边界时需注意区分）。

### 1.2 为什么三维生成长期卡在表示层

在二维图像生成中，规则像素网格与 patch token 提供了天然统一的特征载体；而真实三维资产涵盖开放曲面、非流形网格、薄片结构、嵌套内部结构以及带透光或金属反射的多材质组合，生成系统必须同时兼顾几何、拓扑、UV 纹理、PBR 属性与标准格式导出。下表汇总了主流三维表示的工程权衡：

| 表示形式 | 核心优势 | 主要局限性 |
| --- | --- | --- |
| Mesh | 资产管线通用、光栅化渲染直接 | 拓扑离散，直接生成与压缩困难，难以直接进行稠密特征学习 |
| SDF / Occupancy | 连续场表达清晰，适合隐式曲面重建 | 常依赖水密假设，处理开放曲面、薄壳与非流形结构不自然 |
| Dense Voxel | 规则网格天然适配三维卷积 | 高分辨率下显存与算力呈立方级爆炸，表面与材质表达粗糙 |
| Triplane / Neural Field | 空间压缩率高，适合神经渲染 | 转换为显式网格与标准 PBR 资产时依赖复杂的后处理流程 |
| 3D Gaussian | 实时渲染极快，视角合成视觉质量高 | 缺乏显式表面拓扑，无法直接导出标准网格与 PBR 材质贴图 |
| FlexiCubes / Dual Contouring | 网格重建质量高，锐利边缘还原较好 | 高分辨率下的潜变量 token 规模控制与材质联合建模仍有挑战 |
| SLAT / Structured Latent | 面向生成模型友好，兼顾局部几何 | 需平衡极高空间压缩率与原生几何/材质信息的无损保留 |

当表示层被迫先将三维资产转化为水密 SDF 场或延后进行多视角纹理烘焙时，首先会导致拓扑在预处理阶段被简化，开放布料、植物叶片、链甲与机械孔洞等细微结构在场化（field conversion）过程中永久丢失；其次，几何生成后再做多视图重投影或 UV 烘焙，极易引发跨视角接缝、高光烘焙污染以及透光/金属材质失真；最后，若缺乏高压缩率的结构化潜空间，直接在 $1024^3$ 或 $1536^3$ 空间建模会导致 token 序列长度突破 Transformer 的算力上限。

![Fig. 2: TRELLIS.2 method overview](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig02-method-overview.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 2, arXiv non-exclusive distribution license.*

为此，Fig. 2 定义了 TRELLIS.2 的双向系统边界：

```text
3D assets <---> O-Voxel <---> Sparse Compression VAE <---> SLat <---> Generative Flow Models
```

在训练方向上，系统先将带纹理网格快速转换为 O-Voxel，经由 SC-VAE 编码为紧凑的结构化潜变量（SLat），供流匹配模型学习分布；在推理方向上，模型根据条件图像依次生成稀疏结构坐标、形状潜变量与材质潜变量，再经 SC-VAE 解码器与 O-Voxel 后处理还原为带 PBR 材质的网格（如 GLB 格式）。这种设计将“资产保真转换”、“空间特征压缩”与“概率分布生成”解耦为三个各司其职的模块。

## 2. O-Voxel：兼顾几何拓扑与 PBR 材质的原生稀疏表示

### 2.1 数学定义、Flexible Dual Grid 与 PBR 属性布局

论文将 O-Voxel 定义为定义在有效稀疏体素（active voxels）上的特征元组集合：

$$
f=\{(f_i^{\text{shape}}, f_i^{\text{mat}}, p_i)\}_{i=1}^{L},
$$

其中 $p_i \in \{0,1,\ldots,N-1\}^3$ 表示第 $i$ 个有效体素在 $N \times N \times N$ 空间网格中的离散坐标，$f_i^{\text{shape}}$ 与 $f_i^{\text{mat}}$ 分别记录该体素内的局部几何与局部材质特征，未与资产表面相交的空白体素不占用存储与计算。由于三维表面仅占据体素空间的稀疏壳层，这种定义在天然规避空体素冗余的同时，将几何与材质严格对齐在同一套空间索引下，使后续网络得以直接学习体素对齐的物理材质属性。

![Fig. 3: O-Voxel conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig03-ovoxel-conversion.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 3, arXiv non-exclusive distribution license.*

Fig. 3 展示了 O-Voxel 的内部结构拆解。在几何侧（上半部分），O-Voxel 采用 **Flexible Dual Grid** Formulation：它不在体素中心预测占用率或连续 SDF 值，而是在每个原始网格单元（primal cell）内维护一个对偶顶点（dual vertex），并通过相邻体素的有效相交边（active edge）连接局部曲面。其几何特征向量定义为：

$$
f^{\text{shape}} = (v, \delta, \gamma),
$$

其中 $v \in \mathbb{R}_{[0,1]}^3$ 为对偶顶点在当前体素内部的相对偏移坐标，$\delta \in \mathbb{B}^3$ 为沿三个坐标轴方向的网格边相交布尔标志，$\gamma \in \mathbb{R}_{>0}$ 则控制对偶四边形面片切分为两个三角形时的对角线分割权重（splitting weight）。

在材质侧（下半部分），O-Voxel 直接在相同稀疏体素坐标上存储体素化的表面 PBR 属性：

$$
f^{\text{mat}} = (c, m, r, \alpha),
$$

其中 $c$ 为基础反照率颜色（base color），$m$ 为金属度（metallic），$r$ 为粗糙度（roughness），$\alpha$ 为不透明度（opacity）。通过共享稀疏坐标、分离特征布局（feature layout），O-Voxel 确保了体素化后的特征无需神经场迭代优化即可确定性地还原出显式网格与材质贴图。

### 2.2 网格与 O-Voxel 的双向几何转换（Algorithm 1 & 2）

![Algorithm 1: Mesh-to-O-Voxel conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-algorithm01-mesh-to-ovoxel.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Algorithm 1, arXiv non-exclusive distribution license.*

从显式三角网格到 O-Voxel 几何特征的转换依赖二次误差度量（Quadratic Error Function, QEF）最小化，具体流程如 Algorithm 1 所示：
1. **有效体素初始化**：检测网格三角形与体素网格边的相交关系，仅将表面穿过的局部体素标记为 active voxel 并分配数据结构。
2. **表面平面 QEF 累积**：当三角形 $T$ 与体素边 $e$ 相交于点 $q$（对应法向为 $n$）时，算法为关联体素构建平面距离 QEF 约束，迫使对偶顶点尽可能落在局部切平面附近以保留锐利边角。
3. **开放边界 QEF 累积**：针对非封闭曲面的开放边界边（open mesh edge），额外构建边界距离 QEF 约束。这一步专门解决了传统对偶轮廓法（Dual Contouring）在非水密网格开口处顶点漂移失稳的难题。
4. **正则化与求解**：引入正则化 QEF 项防止病态矩阵或弱约束下的顶点越界，求解线性系统得到最优对偶顶点坐标 $v$，并结合边相交标志 $\delta$ 与四边形切分权重 $\gamma$ 组装出完整的 $f^{\text{shape}}$。

![Algorithm 2: O-Voxel-to-Mesh conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-algorithm02-ovoxel-to-mesh.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Algorithm 2, arXiv non-exclusive distribution license.*

反向转换（O-Voxel-to-Mesh，Algorithm 2）则更加直接高效：首先将每个有效体素内的局部对偶顶点 $v$ 映射为全局空间中的网格顶点；随后遍历所有相交标志 $\delta$ 为真的体素边，跨相邻四个体素连接对偶顶点构成对偶四边形；若四个邻接体素均有效，则依据切分权重 $\gamma$ 沿误差最小的对角线将四边形剖分为两个三角形，即时输出最终网格。官方 `o-voxel` 库将这一确定性过程封装为 `flexible_dual_grid_to_mesh` 等核心算子，彻底省去了神经隐式场提取网格时的昂贵优化开销。

### 2.3 纹理与 O-Voxel 的双向材质转换（Algorithm 3 & 4）

![Algorithm 3: Texture-to-O-Voxel conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-algorithm03-texture-to-ovoxel.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Algorithm 3, arXiv non-exclusive distribution license.*

为了解决训练数据中“二维 UV 贴图材质”与模型“三维稀疏体素潜变量”之间的空间对齐问题，Algorithm 3 定义了 Texture-to-O-Voxel 转换：针对每个有效体素，算法查询与其相交的网格三角形，将体素中心投影至三角形表面并插值计算对应的 UV 坐标，进而从原始 PBR 贴图中采样材质属性，并按投影距离加权融合多个采样点，生成体素对齐的 $f^{\text{mat}}$。

![Algorithm 4: O-Voxel-to-Texture conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-algorithm04-ovoxel-to-texture.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Algorithm 4, arXiv non-exclusive distribution license.*

在导出阶段（O-Voxel-to-Texture，Algorithm 4），系统支持两种模式：一是轻量化的 **顶点着色模式（vertex mode）**，直接对网格顶点在 O-Voxel 材质体中做三线性插值，适合快速预览与调试；二是面向工业管线的 **贴图烘焙模式（map mode）**，先对重建网格执行 UV 参数化展开，再将纹理贴图中的每个纹素（texel）反查至三维表面位置，从稀疏材质体中插值提取 $(c, m, r, \alpha)$ 并写入贴图通道。官方代码中的 `o_voxel.postprocess.to_glb` 正是基于该模式串联了重网格化（remesh）、UV 展开、纹理烘焙与 GLB 打包。

## 3. Sparse Compression VAE（SC-VAE）与稀疏算子后端

### 3.1 为什么在 O-Voxel 之上还需要 SC-VAE

尽管 O-Voxel 已经剔除了空白体素，但一尊 $1024^3$ 分辨率的高精细资产仍包含数十万量级的有效表面体素。若直接在原始 O-Voxel 空间训练扩散或流匹配 Transformer，自注意力的序列长度与显存消耗将难以承受。因此，SC-VAE 的任务是将高分辨率 O-Voxel 进一步压缩为高信息密度的结构化潜变量 $z$：

$$
z = E_\phi(f), \qquad \hat f = D_\theta(z).
$$

论文采用 **16 倍空间下采样（16x spatial downsampling）** 与 32 维潜通道配置（`f16c32`），使潜空间坐标网格比原始 O-Voxel 粗糙 16 倍，同时依靠解码器恢复局部子体素几何与高频材质。

![Fig. 4: SC-VAE architecture](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig04-scvae-architecture.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 4, arXiv non-exclusive distribution license.*

如 Fig. 4 所示，SC-VAE 将 $f^{\text{shape}}$ 与 $f^{\text{mat}}$ 分别送入结构对称的全稀疏卷积（fully sparse-convolutional）编码器与解码器，避免在空白区域浪费算力。为了在 16 倍强空间下采样下保留细微特征，架构右侧引入了 **稀疏残差自编码层（Sparse Residual Autoencoding Layer）**：在下采样与上采样分支中融入空间到通道（space-to-channel / channel-to-space）的残差捷径，防止高频几何与纹理信息在层级池化中坍缩。

![Table 4: SC-VAE architecture](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table04-scvae-architecture.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Table 4, arXiv non-exclusive distribution license.*

Table 4 给出了 SC-VAE 编码器从 `1x` 到 `16x` 的逐级下采样规格，每级由子流形稀疏卷积（SubMConv）、LayerNorm、Linear、SiLU 及 ResEnc 模块堆叠而成，解码器呈对称拓扑。附录披露整个 SC-VAE 参数量高达约 800M（编码器约 354M，解码器约 474M），可见 TRELLIS.2 的模型规模不仅体现在后端的生成式 DiT，其前端的稀疏表示学习器本身就是一个重量级网络。

### 3.2 两阶段训练目标

SC-VAE 的训练分为两个阶段推进。第一阶段在较低分辨率数据上直接回归 O-Voxel 特征并施加 KL 正则化：几何分支对对偶顶点偏移 $v$ 采用均方误差（MSE），对边相交标志 $\delta$ 与体素剪枝掩码（pruning mask）$\rho$ 采用二元交叉熵（BCE）；材质分支对 PBR 属性采用 $L_1$ 损失。第一阶段总损失写为：

$$
\mathcal{L}_{s1}
= \lambda_v \|\hat v - v\|_2^2
+ \lambda_\delta \operatorname{BCE}(\hat \delta,\delta)
+ \lambda_\rho \operatorname{BCE}(\hat \rho,\rho)
+ \lambda_{\text{mat}}\|\hat f^{\text{mat}}-f^{\text{mat}}\|_1
+ \lambda_{\text{KL}}\mathcal{L}_{\text{KL}}.
$$

第二阶段引入高分辨率可微渲染感知监督（rendering-based perceptual supervision），对解码出的网格与材质渲染蒙版（mask）、深度图（depth）、法线图（normal）与 PBR 着色图，结合 $L_1$、SSIM 与 LPIPS 损失构造渲染项 $\mathcal{L}_{\text{render}}$：

$$
\mathcal{L}_{s2}=\mathcal{L}_{s1}+\mathcal{L}_{\text{render}}.
$$

两阶段策略兼顾了底层特征空间的收敛稳定性与最终三维表面、法线及材质在人眼视觉空间中的感知锐度。

### 3.3 FlexGEMM 高性能稀疏卷积后端

![Fig. 9: FlexGEMM speed comparison](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig09-flexgemm-speed.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 9, arXiv non-exclusive distribution license.*

由于 SC-VAE 的编码与解码完全建立在三维稀疏卷积之上，现有算子库（如 SpConv、TorchSparse、fVDB、WarpConvNet）在高分辨率与大批量下的显存带宽及核函数启动开销成为系统瓶颈。论文在附录中专门设计了 **FlexGEMM** 稀疏卷积后端（见 Fig. 9 速度对比）：其核心为掩码隐式矩阵乘（masked implicit GEMM），将特征收集（feature gathering）与矩阵乘法融合进单一 CUDA kernel 中以减少全局显存读写；同时利用格雷码排序（Gray code ordering）提升空间邻域访问的缓存局部性，并借助 Split-K 并行策略提高 GPU 占用率。这也解释了为何官方环境配置脚本必须显式编译安装 `flex_gemm`、`o-voxel`、`cumesh`、`nvdiffrast` 与 `nvdiffrec` 等底层 CUDA 扩展。

## 4. 三阶段 Flow Matching 生成架构与测试期分辨率扩展

### 4.1 生成模型主干与三阶段级联设计

![Table 5: Generative model architecture](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table05-generative-model-architecture.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Table 5, arXiv non-exclusive distribution license.*

在紧凑的 SLat 空间之上，TRELLIS.2 采用纯编码器架构的稀疏 Transformer（DiT）作为流匹配主干。如 Table 5 所示，单个流模型通过 `Linear(32(+32), 1536)` 将输入潜变量映射至 1536 维隐藏空间，堆叠 30 层 Transformer Block（包含 12 头 128 维自注意力、12 头 128 维图像条件交叉注意力以及 `1536 -> 8192 -> 1536` 的前馈网络），最后经 `LayerNorm + Linear(1536, 32)` 输出速度场预测，并结合三维旋转位置编码（RoPE）与 QK-Norm 保障大规模训练稳定性。单个阶段 DiT 参数量约为 1.3B，三个生成阶段合计构成约 4B 参数的完整体系：

```text
Input Image
  --> Stage 1: Sparse Structure Flow  (预测粗粒度有效体素坐标骨架)
  --> Stage 2: Shape SLat Flow        (在稀疏坐标上采样几何潜变量 -> 解码几何网格)
  --> Stage 3: Material SLat Flow     (以几何潜变量为条件采样材质潜变量 -> 解码 PBR 属性)
```

这三个阶段顺应了三维资产的内在物理依赖：第一阶段（对应代码中的 `sample_sparse_structure()`）先确立资产在三维空间中的稀疏体素占据骨架；第二阶段（`sample_shape_slat()` 或级联版 `sample_shape_slat_cascade()`）在稀疏坐标上生成形状潜变量；第三阶段（`sample_tex_slat()`）将归一化后的形状潜变量作为 `concat_cond` 拼接到输入端，在已知准确几何表面的前提下生成对齐的 PBR 材质潜变量，从而根除了材质漂浮或几何错位问题。

### 4.2 条件流匹配（Conditional Flow Matching）与测试期扩展

模型训练采用 Rectified Flow / Conditional Flow Matching 范式。设目标数据潜变量为 $x_0$，标准高斯噪声为 $\epsilon \sim \mathcal{N}(0, I)$，时间步 $t \in [0, 1]$ 下的线性插值路径定义为：

$$
x(t) = (1-t)x_0 + t\epsilon,\qquad t\in[0,1].
$$

该路径对应的目标速度场（vector field）为常数方向 $v(x,t)=\nabla_t x=\epsilon-x_0$。神经网络 $v_\theta$ 在给定图像条件与时间步 $t$ 下直接回归该速度场：

$$
\mathcal{L}_{\text{CFM}}(\theta)
= \mathbb{E}_{t,x_0,\epsilon}
\left\|v_\theta(x(t),t)-(\epsilon-x_0)\right\|_2^2.
$$

针对高分辨率资产生成，论文进一步设计了测试期分辨率与算力扩展策略（Test-time Resolution Scaling）。

![Fig. 8: Test-time scaling](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig08-test-time-scaling.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 8, arXiv non-exclusive distribution license.*

如 Fig. 8 所示，系统既支持在固定架构下增加采样计算步数，也支持通过多分辨率级联（cascade）从 $512^3$ 扩展至 $1024^3$ 乃至 $1536^3$。在官方推理管线中，这体现为 `512`、`1024`、`1024_cascade` 与 `1536_cascade` 四种运行模式：`sample_shape_slat_cascade()` 先由 $512$ 流模型生成基础形状潜变量，随后调用解码器的 `upsample()` 算子生成高分辨率候选稀疏坐标，并通过 `max_num_tokens` 动态监控显存预算——当复杂资产的派生活跃 token 数超出阈值时自动回退目标分辨率，实现了生成精细度与显存安全的自适应平衡。

## 5. 数据构建管线与评估基准

![Table 6: Dataset composition](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table06-dataset.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Table 6, arXiv non-exclusive distribution license.*

高质量的原生三维表示与 PBR 材质生成极度依赖经过严格清洗的资产数据集。Table 6 详列了 TRELLIS.2 的训练与评估数据构成，汇总自 TexVerse、Objaverse-XL（Sketchfab 与 GitHub 子集）、ABO 与 HSSD，并以 Toys4K（及 Sketchfab Featured）作为评估基准：

| 数据来源 | 可用几何资产数 (Shape available) | 可用材质资产数 (Material available) |
| --- | ---: | ---: |
| TexVerse | 503,387 | 382,996 |
| ObjaverseXL (Sketchfab) | 168,307 | 141,623 |
| ObjaverseXL (GitHub) | 293,887 | 202,188 |
| ABO | 4,485 | 4,485 |
| HSSD | 6,670 | 6,670 |
| **SC-VAE 训练集小计** | **473,349** | **354,966** |
| **全量生成模型训练集** | **976,736** | **737,962** |
| **Toys4K 评估集** | **3,229** | **2,282** |

表中“几何可用”与“材质可用”的数量差异揭示了数据清洗的关键准则：并非所有三维模型都具备合格的 PBR 材质贴图（例如论文明确剔除了缺乏标准 PBR 材质的 3D-FUTURE 数据集）。在训练材质 VAE 与材质流模型时，作者使用 Blender 解析脚本深入检查原始资产的着色器节点树，仅保留遵循标准金属度-粗糙度（metallic-roughness）PBR 工作流的样本。对应到官方仓库的 `data_toolkit/`，完整的数据预处理流水线依次执行：`build_metadata`（元数据初始化）$\rightarrow$ `download`（多源数据拉取）$\rightarrow$ `dump_mesh` / `dump_pbr`（标准化网格与 PBR 通道解包）$\rightarrow$ `dual_grid` / `voxelize_pbr`（转换 O-Voxel 几何与材质体素）$\rightarrow$ `encode_shape_latent` / `encode_pbr_latent` / `encode_ss_latent`（预编码各级潜变量缓存）$\rightarrow$ `render_cond`（渲染 16 视角条件图像）。

## 6. 实验分析：重建保真度、生成质量与消融验证

### 6.1 几何重建对比（Table 1）与 SC-VAE 消融（Table 3）

![Table 1: Reconstruction comparison](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table01-reconstruction.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Table 1, arXiv non-exclusive distribution license.*

Table 1 在 Toys4K 与 Sketchfab Featured 两个测试集上全面评估了不同三维表示与自编码器的几何重建效率与保真度。评价指标涵盖三类：表面几何距离（Mesh Distance `MD` 与 Chamfer Distance `CD`，均按 $\times 10^6$ 缩放，越低越好）、严格距离阈值下的表面重合度（$\text{F1}_{1\text{e-}8}$ 与 $\text{F1}_{1\text{e-}6}$，越高越好），以及多视角法线图渲染质量（PSNR 越高越好、LPIPS 越低越好）。

对比结果呈现出清晰的量效关系：首先，对比 `Ours 1024` 与前代 `TRELLIS`，两者潜空间 token 数均约为 9.6K，但 TRELLIS.2 凭借 16 倍空间下采样的 O-Voxel SC-VAE（解码耗时 0.301 秒 vs. TRELLIS 的 0.108 秒），在表面距离、F1 分数与法线 PSNR/LPIPS 上取得了全方位大幅领先；其次，对比高保真稀疏基线 `SparseFlex 1024`，后者需要消耗多达 225K 个 token，而 `Ours 1024` 仅用约 1/23 的 token 数量（9.6K）便实现了更优的综合重建指标，有力验证了“紧凑结构化潜空间”的核心主张。

![Table 3: SC-VAE ablation](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table03-scvae-ablation.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Table 3, arXiv non-exclusive distribution license.*

Table 3 进一步对 SC-VAE（主线配置 `f16c32`）展开架构消融。实验表明，一旦移除稀疏残差自编码层（w/o Residual AE），16 倍下采样下的 MD 与 F1 指标出现显著退化，证明强空间压缩必须配合通道间的信息重排捷径才能避免高频几何丢失；同时，采用精简卷积层与逐点 MLP 构成的优化残差块（optimized ResBlock），在降低稀疏卷积计算开销的同时提升了稀疏特征的表达能力。

### 6.2 单图到 3D 资产生成与条件 PBR 纹理生成（Fig. 5–7 & Table 2）

![Fig. 5: Generated assets](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig05-generated-assets.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 5, arXiv non-exclusive distribution license.*

![Fig. 6: Generation comparison](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig06-generation-comparison.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 6, arXiv non-exclusive distribution license.*

在生成任务的定性评估中，Fig. 5 展示了 TRELLIS.2 生成的复杂植物叶片、镂空机械、角色盔甲与蒸汽朋克装置，展现了模型对开放薄曲面、内部嵌套结构以及半透明/高反射金属材质的建模能力。Fig. 6 将 TRELLIS.2 与 Hunyuan3D 2.1、Step1X-3D、TRELLIS、Direct3D-S2 及 Hi3DGen 进行同图对比，并同步列出法线图、最终渲染图以及拆解出的 base color、metallic、roughness 通道（部分基线因不支持原生材质输出被标注为 “No PBR”）。

![Table 2: Image-to-3D generation results](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table02-generation.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Table 2, arXiv non-exclusive distribution license.*

Table 2 从跨模态对齐度与感知偏好两个维度量化了单图生成性能：TRELLIS.2 在视觉语义相似度 CLIP（`0.894`）、法线几何相似度 CLIP-N（`0.758`）、三维感知嵌入相似度 ULIP-2（`0.477`）与 Uni3D（`0.436`）上全部位居榜首，并在双盲用户偏好测试中取得了 `66.5%` 的整体胜率（Pref%）与 `69.0%` 的几何法线胜率（Pref-N%）。

![Fig. 7: PBR texture generation](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig07-pbr-texture-generation.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 7, arXiv non-exclusive distribution license.*

此外，得益于第三阶段材质流模型以形状潜变量为条件的解耦设计，TRELLIS.2 天然支持给定现成白模网格与参考图像的 **形状条件 PBR 纹理生成（Shape-conditioned PBR Texture Generation）**。如 Fig. 7 所示，相比 Hunyuan3D-Paint 与 TEXGen 等基于多视图扩散再重投影贴图的方案，直接在三维稀疏潜空间生成 $(c, m, r, \alpha)$ 避免了多视角拼接伪影与高光粘连。

### 6.3 双盲用户研究细节（Fig. 10 & Table 7）

![Fig. 10: User study interface](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig10-user-study-interface.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 10, arXiv non-exclusive distribution license.*

![Table 7: User study statistics](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table07-user-study.webp)

*Source: Xiang et al., arXiv:2512.14692v1, Table 7, arXiv non-exclusive distribution license.*

为了弥补自动指标对三维结构与材质真实感评估的盲区，Fig. 10 与 Table 7 展示了包含约 40 名参与者、100 组 AI 生成图像提示词的用户研究设置与统计明细。评估界面要求受试者分别基于完整 PBR 渲染视图评价“整体资产质量（Overall Quality）”，以及基于纯法线渲染视图评价“几何形状质量（Shape Quality）”以排除色彩纹理的视觉干扰。在共计 203 次整体评估与 213 次几何评估的统计中，TRELLIS.2 分别赢得了 135 次（`66.5%`）与 147 次（`69.0%`）首选。

## 7. 官方源码剖析与推理训练链路

官方开源仓库 `microsoft/TRELLIS.2` 完整公开了推理管线、纹理生成、O-Voxel 核心库、数据准备工具链与训练配置。下表梳理了论文核心模块与源码路径的映射关系：

| 论文核心模块 | 源码路径 | 核心职责与实现说明 |
| --- | --- | --- |
| Image-to-3D 管线 | `trellis2/pipelines/trellis2_image_to_3d.py` | 封装 `Trellis2ImageTo3DPipeline`，串联三阶段流模型采样与解码 |
| 形状条件贴图管线 | `trellis2/pipelines/trellis2_texturing.py` | 封装 `Trellis2TexturingPipeline`，将输入网格编码后采样 PBR 材质 |
| O-Voxel 核心算子库 | `o-voxel/` | 独立子包，提供网格/O-Voxel 双向转换、Morton/Hilbert 序列化及 GLB 导出 |
| SC-VAE 模型定义 | `trellis2/models/sc_vaes/*` | 实现形状与材质的全稀疏卷积变分自编码器 |
| 流匹配生成网络 | `trellis2/models/sparse_structure_flow.py`, `structured_latent_flow.py` | 实现稀疏结构 DiT 与结构化潜变量 DiT |
| 稀疏张量与注意力 | `trellis2/modules/sparse/*` | 封装 `SparseTensor`、稀疏卷积及稀疏感知 Transformer 组件 |
| 数据构建工具集 | `data_toolkit/` | 实现多源数据下载、网格/PBR 解包、体素化、潜变量预编码与条件图渲染 |
| 训练入口与配置 | `train.py`, `configs/scvae/*`, `configs/gen/*` | 提供 SC-VAE 两阶段训练及 1.3B 级流模型训练/高分微调配置 |
| 交互式演示与示例 | `app.py`, `app_texturing.py`, `example.py`, `example_texturing.py` | Gradio 界面与最小化命令行推理脚本 |

在 `Trellis2ImageTo3DPipeline` 中，`model_names_to_load` 清晰列出了运行期加载的八个子模型权重：`sparse_structure_flow_model`、`sparse_structure_decoder`、`shape_slat_flow_model_512`、`shape_slat_flow_model_1024`、`shape_slat_decoder`、`tex_slat_flow_model_512`、`tex_slat_flow_model_1024` 与 `tex_slat_decoder`。调用 `pipeline.run(image)` 时，代码依次执行图像前处理（`preprocess_image`）、提取 512/1024 视觉条件特征（`get_cond`）、采样稀疏体素骨架（`sample_sparse_structure`）、采样或级联生成形状潜变量（`sample_shape_slat` / `sample_shape_slat_cascade`）、拼接归一化形状潜变量采样材质潜变量（`sample_tex_slat`），最后调用 `decode_latent` 还原出三维资产。而在 `Trellis2TexturingPipeline` 中，材质通道切片被显式定义为：

```python
{
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}
```

对于希望深入底层的读者，`o-voxel/` 子库中的 `mesh_to_flexible_dual_grid`、`flexible_dual_grid_to_mesh`、`textured_mesh_to_volumetric_attr` 与 `o_voxel.postprocess.to_glb`（参见 `o-voxel/examples/` 下的 `mesh2ovox.py`、`ovox2mesh.py`、`ovox2glb.py`）是最值得优先阅读的源码。而在训练配置侧，`configs/scvae/` 区分了基础训练（`*_f16c32_fp16.json`）与渲染感知微调（`*_ft_512.json`），`configs/gen/` 则对应 `ss_flow_img_dit_1_3B_64_bf16.json`、`slat_flow_img2shape_dit_1_3B_512_bf16(_ft1024).json` 与 `slat_flow_imgshape2tex_dit_1_3B_512_bf16(_ft1024).json`，完整映射了论文的分阶段训练方案。

在推理与导出的工程细节上，官方 `example.py` 展示了两个值得注意的后处理约束：一是调用 `mesh.simplify(16777216)` 将三角面数限制在 `nvdiffrast` 光栅化器的硬件上限之内；二是 GLB 导出默认采用 `OPAQUE`（不透明）材质模式——尽管预测的 `alpha` 通道已完整烘焙进纹理贴图，但在导入 Blender 或游戏引擎时需由用户手动将 alpha 通道挂载至材质节点的透明度输入端。

## 8. 方法横向对比与工程复现指南

### 8.1 与主流 3D 生成系统及本站相关论文的坐标对比

将 TRELLIS.2 置于当前单图 3D 生成与三维场景建模的技术谱系中，可以清晰界定其优势与适用边界：

| 对比维度 / 代表工作 | 核心技术路线 | 相较于 TRELLIS.2 的定位差异 |
| --- | --- | --- |
| **TRELLIS (v1)** | 结构化潜空间（SLAT）三维生成 | TRELLIS.2 引入无场 O-Voxel、16x 高压缩 SC-VAE 与原生体素级 PBR 材质建模 |
| **Hunyuan3D 2.1 / Step1X-3D** | 几何生成 + 多视角扩散纹理烘焙两阶段系统 | TRELLIS.2 在统一稀疏体素潜空间内直接生成 PBR 通道，规避多视图贴图拼接伪影 |
| **SparseFlex / Direct3D-S2** | 稀疏体素/网格高保真重建与高效生成 | TRELLIS.2 在仅约 9.6K token 的紧凑预算下兼顾开放拓扑重建与完整 PBR 输出 |
| **SF3D / AssetGen** | 面向低延迟或可部署管线的网格、UV 与材质优化 | SF3D 主打 0.5s 前馈重建，AssetGen 强调低面数预算与法线烘焙管线；TRELLIS.2 侧重高保真原生表示与生成上限 |
| **CompoNeRF / Blended-NeRF / Layout2Scene / CasLayout / Imaginarium** | 场景级 NeRF 编辑、三维语义布局扩散或资产检索摆放 | 这些工作聚焦多物体空间布局与场景级一致性，而 TRELLIS.2 专注于单体资产的高质量几何与材质生成底座 |

### 8.2 工程复现清单与客观局限性

若需在工程环境中部署或复现 TRELLIS.2，建议遵循以下核对清单：
1. **软硬件环境准备**：配置 Linux 系统与至少 24GB 显存的 NVIDIA GPU（官方基准基于 A100/H100 与 CUDA 12.4），通过 `. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm` 编译全套自定义算子。
2. **分步验证路径**：先运行 `o-voxel/examples/` 脚本验证网格与 O-Voxel/GLB 的无损互转；再下载 `microsoft/TRELLIS.2-4B` 运行 `example.py` 与 `example_texturing.py`，测试 `512`、`1024_cascade` 与 `1536_cascade` 模式并观察 `max_num_tokens` 触发的分辨率回退行为；最后在 Blender 中检查导出的 GLB 法线、金属度、粗糙度与透明通道连接。

同时，在将 TRELLIS.2 引入实际生产管线前，需清醒认识其工程局限性：
- **算力与编译门槛高昂**：全套系统重度依赖自定义 CUDA 扩展（FlexGEMM、CuMesh 等）与大显存 GPU，在高分辨率级联模式下仍受 `max_num_tokens` 预算制约，难以在消费级轻薄设备上低延迟运行。
- **训练高度依赖高纯度 PBR 数据**：SC-VAE 与材质流模型的表现建立在近百万级经过 Blender 节点严格筛选的标准化 PBR 资产之上，迁移至缺乏干净 PBR 标注的垂直领域时数据清洗成本极高。
- **感知指标与工业级拓扑规范仍存代沟**：CLIP、ULIP-2、PSNR 与用户偏好主要反映视觉渲染质量，而游戏与影视生产还要求艺术家友好的四边形布线（edge flow）、合理的 UV 岛利用率、多级 LOD、骨骼绑定（rigging）兼容性以及碰撞体规范，模型导出的高面数三角网格仍需下游资产化后处理。

## 9. 结论与参考资料

TRELLIS.2 的核心启示在于：三维生成大模型的规模化（Scaling）必须以高保真、高紧凑度的原生三维表示为基础。通过提出支持任意非流形/开放拓扑与体素级 PBR 属性的 **O-Voxel**，配合 16 倍空间压缩的 **SC-VAE** 与高性能 **FlexGEMM** 稀疏算子后端，TRELLIS.2 将 4B 参数的三阶段流匹配模型建立在真正能够无损还原三维资产的潜空间之上。对于研究者与工程师而言，建议先精读论文第 3 节（Fig. 2–4）与附录 A（Algorithm 1–4）掌握 O-Voxel 与 SC-VAE 的数学构造，再对照 `o-voxel/` 与 `trellis2/pipelines/` 源码理解其从稀疏潜变量到标准 GLB 资产的完整工程闭环。

### References

- [Native and Compact Structured Latents for 3D Generation, arXiv:2512.14692](https://arxiv.org/abs/2512.14692)
- [PDF: Native and Compact Structured Latents for 3D Generation](https://arxiv.org/pdf/2512.14692)
- [TRELLIS.2 Project Page](https://microsoft.github.io/TRELLIS.2/)
- [Official GitHub: microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
- [Hugging Face Model: microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)
- [Hugging Face Demo: TRELLIS.2](https://huggingface.co/spaces/microsoft/TRELLIS.2)

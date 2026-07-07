---
title: "TRELLIS.2 论文精读：O-Voxel、Sparse Compression VAE 与 4B 参数 3D 生成模型"
description: "精读 Native and Compact Structured Latents for 3D Generation 如何通过 O-Voxel、稀疏压缩 VAE、PBR 材质建模与大规模 flow matching 实现高质量 3D 资产生成"
pubDate: "2026-07-07T10:59:40+08:00"
updatedDate: "2026-07-07T10:59:40+08:00"
tags:
  - "Paper Reading"
  - "3D Generation"
  - "Computer Vision"
  - "Generative AI"
  - "3D Assets"
  - "Code Reading"
draft: false
---

Jianfeng Xiang、Xiaoxue Chen、Sicheng Xu、Ruicheng Wang、Zelong Lv、Yu Deng、Hongyuan Zhu、Yue Dong、Hao Zhao、Nicholas Jing Yuan 和 Jiaolong Yang 在技术报告 **Native and Compact Structured Latents for 3D Generation** 中提出了 TRELLIS.2。它看起来是一篇“大模型做 3D 生成”的论文，但真正值得精读的地方不是 4B 参数本身，而是作者试图回答一个更底层的问题：

> 如果 3D 生成的表示层本身会丢拓扑、丢细节、丢材质，继续堆更大的生成模型是否只是把错误放大？

TRELLIS.2 的回答是先重做表示。论文提出一种 field-free 的稀疏体素结构 **O-Voxel**，同时表达几何和 PBR 材质；再用 fully sparse-convolutional **Sparse Compression VAE** 将高分辨率 3D 资产压缩到紧凑结构化 latent；最后在 latent 空间训练三阶段 flow matching 生成模型，分别负责 sparse structure、shape 和 material。

这篇报告会沿着“表示 -> 压缩 -> 生成 -> 系统 -> 实验 -> 代码”的顺序展开，并把论文方法和官方 `microsoft/TRELLIS.2` 仓库做静态对照。本文不声称完成本地推理、训练或复现论文结果。官方 README 也明确当前代码主要在 Linux、NVIDIA A100/H100、CUDA 12.4 环境下验证，完整模型和数据管线对普通本地机器并不轻。

用户最初给出的 PDF URL 是 `https://arxiv.org/pdf/2512/14692`，这个斜杠格式无法打开。本文使用规范链接：[arXiv abs: 2512.14692](https://arxiv.org/abs/2512.14692) 和 [PDF: 2512.14692](https://arxiv.org/pdf/2512.14692)。论文图表来自 arXiv PDF 等比例裁切；图注标注为 arXiv non-exclusive distribution license，避免把论文图误写成 GitHub 代码的 MIT 许可。

## 1. 论文信息与一句话贡献

论文和官方材料如下。

| 项目 | 内容 |
| --- | --- |
| 论文 | Native and Compact Structured Latents for 3D Generation |
| 项目名 | TRELLIS.2 |
| arXiv | `2512.14692v1` |
| 页数 | 24 页 |
| 作者 | Jianfeng Xiang 等 |
| 项目页 | `https://microsoft.github.io/TRELLIS.2/` |
| 官方仓库 | `microsoft/TRELLIS.2` |
| 模型 | `microsoft/TRELLIS.2-4B` |
| Demo | Hugging Face Spaces |
| 主要关键词 | O-Voxel, Sparse Compression VAE, PBR Materials, Flow Matching, Sparse Convolution |

一句话概括：

> TRELLIS.2 的核心贡献是把 3D 资产生成的表示层改造成可双向转换、可压缩、可表达任意拓扑和 PBR 材质的 O-Voxel，再在这种结构化 latent 上训练大规模生成模型。

这句话有三层含义。

第一，论文不是只提出一个更大的 image-to-3D 模型。作者将性能提升归因于表示、压缩和生成三者的耦合。O-Voxel 让训练数据更接近原生 3D 资产；SC-VAE 让高分辨率资产可以被压缩成有限数量的 sparse tokens；flow model 再学习这些 tokens 的分布。

第二，论文不是传统 mesh 生成，也不是普通 voxel occupancy。O-Voxel 不把 3D 几何压成一个连续场的 level set，也不要求输入资产 watertight。它直接在 sparse voxel 上保存 dual vertex、edge intersection、splitting weight 等局部拓扑信息。

第三，论文把材质放进表示，而不是后处理。许多 3D 生成系统先生成几何，再通过多视图图像或 UV baking 补纹理。TRELLIS.2 的 O-Voxel 同时保存 base color、metallic、roughness、opacity 等 PBR 属性，因此它的目标更接近“可渲染资产”而不只是“看起来像的表面”。

## 2. Fig. 1 总览：论文卖点是什么

![Fig. 1: TRELLIS.2 teaser](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig01-teaser.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 1, arXiv non-exclusive distribution license.*

Fig. 1 把论文主张压缩成三组现象。

左侧是高分辨率资产重建。论文展示了一个 `1536^3` 分辨率的涡轮引擎资产，结果里有正常渲染、normal map 和多级分辨率预览。这里强调的是 O-Voxel 不是只能表达粗糙 occupancy，而是能够保留薄片、孔洞、边缘和内部结构。

中间是生成结果。论文展示一个复杂机械角色，旁边列出 alpha、roughness、metallic、base color 等材质通道。注意这些不是最终 RGB 渲染的隐含效果，而是模型输出资产中的 PBR 属性。

右侧是 latent compactness 对比。横轴是 token 数，纵轴是重建质量相关指标。论文想说明：Ours 在较少 token 下达到更好的 reconstruction fidelity。这个点很关键，因为 3D 生成的成本常常不是参数量，而是每个资产需要多少 latent token、每个 token 携带多少维度、解码时需要多少 sparse convolution。

官方 README 给出的速度数字更适合工程读者理解边界：在 H100 上，`512^3` 约 3 秒，`1024^3` 约 17 秒，`1536^3` 约 60 秒。论文实验表中又报告了 A100 上的 decode time。写工程方案时要把这两组环境区分开，不要把 H100 速度外推到普通消费级 GPU。

## 3. 背景：3D 生成为什么先卡在表示层

2D 图像生成中，像素网格、latent image、patch token 都是相对规则的表示。3D 资产则复杂得多：真实资产可能是开放曲面、非流形 mesh、薄片结构、嵌套内部结构、带透明或金属材质的多材质对象。生成系统必须同时处理几何、拓扑、纹理、PBR 属性和导出格式。

常见表示各有代价。

| 表示 | 优点 | 主要问题 |
| --- | --- | --- |
| Mesh | 资产管线通用、渲染直接 | 拓扑离散，生成和压缩难，难以直接做 dense learning |
| SDF / occupancy | 连续场表达清晰，适合隐式重建 | 常假设 watertight，开放曲面和非流形不自然 |
| Dense voxel | 规则网格适合卷积 | 分辨率高时代价极大，材质表达粗糙 |
| Triplane / neural field | 压缩率高，适合神经渲染 | 转 mesh、转 PBR asset 仍需后处理 |
| 3D Gaussian | 渲染速度快，视觉质量高 | 拓扑和标准资产导出不直接，材质管线不天然 |
| Flexicubes / dual contouring 相关结构 | 能重建 mesh，边界细节较好 | latent 数量、压缩和材质耦合仍是难点 |
| SLAT / structured latent | 面向生成更友好 | 需要解决 token 数和原生 3D 信息保留问题 |

TRELLIS.2 的立场是：如果表示层必须先把 3D 资产转成 SDF、隐式场或后处理纹理，生成模型学到的分布就不是“原生资产分布”。这会造成三个后果。

第一，拓扑被提前简化。开放布料、植物叶片、链甲、车架、机械孔洞这类结构会在字段化或简化过程中损失。

第二，材质被延后处理。几何模型生成后再做多视图纹理或 UV baking，容易出现视角不一致、接缝、反射/透明材料不稳定等问题。

第三，高分辨率训练成本失控。直接在 `1024^3` 或 `1536^3` 空间建模，如果没有足够紧凑的 structured latent，token 数会直接拖垮生成模型。

## 4. Fig. 2 Pipeline：O-Voxel、SC-VAE、Flow Models

![Fig. 2: TRELLIS.2 method overview](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig02-method-overview.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 2, arXiv non-exclusive distribution license.*

Fig. 2 很小，但它定义了全文的系统边界。

```text
3D assets
  -> instant bidirectional conversion
  -> O-Voxel
  -> Sparse Compression VAE
  -> SLat
  -> large-scale generative modeling
  -> generated 3D assets
```

这条链路有两个方向。

训练方向是从真实资产出发。先把 textured mesh 转为 O-Voxel，再通过 SC-VAE 编码成 compact structured latent，最后训练 flow models 学习 latent 分布。

推理方向是从条件图像出发。模型先生成 sparse structure，再生成 shape latent，再生成 material latent，最后由 decoder 和 O-Voxel 后处理恢复 mesh/PBR material，并导出为 GLB 或用于渲染。

这里最重要的不是三阶段 pipeline 本身，而是它把“3D 表示”和“生成模型”解耦了。O-Voxel 负责资产信息保真，SC-VAE 负责压缩，flow matching 负责分布建模。任何一层弱都会限制整体上限。

## 5. O-Voxel：原生 3D 表示的基本形式

论文定义 O-Voxel 为一组 sparse voxel 上的特征元组：

$$
f=\{(f_i^{\text{shape}}, f_i^{\text{mat}}, p_i)\}_{i=1}^{L},
$$

其中 $p_i \in \{0,1,\ldots,N-1\}^3$ 是第 $i$ 个 active voxel 在 $N \times N \times N$ 网格中的坐标，$f_i^{\text{shape}}$ 保存局部几何，$f_i^{\text{mat}}$ 保存局部材质。没有与资产相交的空 voxel 不存储。

这个定义有两个现实好处。

第一，它天然是 sparse 的。真实 3D 资产只占据体素网格中的一小部分，特别是表面资产，大量内部和外部 voxel 都没有必要建模。

第二，它把几何和材质放在同一空间坐标上。后续 SC-VAE 和 texture flow 不需要把材质当成渲染图像里的颜色残差，而是可以学习稀疏体素上的 PBR 属性。

## 6. Fig. 3：Flexible Dual Grid 与 PBR 属性

![Fig. 3: O-Voxel conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig03-ovoxel-conversion.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 3, arXiv non-exclusive distribution license.*

Fig. 3 是理解 O-Voxel 的核心图。上半部分是 shape，下面是 material。

Shape 侧使用 Flexible Dual Grid。它不是在 voxel center 上预测 occupancy，也不是预测一个连续 SDF 后抽 iso-surface。每个 primal cell 有一个 dual vertex，局部 surface 通过相邻 voxel 的 active edge 连接出来。论文中的 shape feature 可以理解为：

$$
f^{\text{shape}} = (v, \delta, \gamma),
$$

其中 $v \in \mathbb{R}_{[0,1]}^3$ 表示 dual vertex 在 voxel 内的位置，$\delta \in \mathbb{B}^3$ 表示三个方向上的 edge intersection flag，$\gamma \in \mathbb{R}_{>0}$ 表示 quad split 的权重或相关几何信息。

Material 侧则保存体素化的 surface attributes：

$$
f^{\text{mat}} = (c, m, r, \alpha),
$$

其中 $c$ 是 base color，$m$ 是 metallic，$r$ 是 roughness，$\alpha$ 是 opacity。它们都和 sparse voxel 坐标对齐。

O-Voxel 的关键不是“把 mesh 体素化”这么简单，而是体素化以后仍然保留足够信息恢复 surface mesh 和 material maps。图中右侧两个 feature box 表明，几何和材质虽然共享 sparse voxel 位置，但各自有不同的 feature layout。

## 7. Mesh-to-O-Voxel：QEF 不是细节，而是表示的根

![Algorithm 1: Mesh-to-O-Voxel conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-algorithm01-mesh-to-ovoxel.png)

*Source: Xiang et al., arXiv:2512.14692v1, Algorithm 1, arXiv non-exclusive distribution license.*

Algorithm 1 说明 mesh 如何转成 O-Voxel shape feature。流程可以拆成五步。

第一步，为每个 active voxel 初始化数据结构。active 的定义来自 mesh triangle 与 voxel edge 的交互。也就是说，不是整张 dense grid 都要参与计算，只有被表面穿过的局部体素需要保存信息。

第二步，从 triangle intersection 累积 plane-distance QEF。每当 triangle $T$ 与某条 voxel edge $e$ 相交，算法得到交点 $q$ 和法向 $n$，再为相邻 voxel 建立 plane QEF。直觉上，这个 QEF 要求 dual vertex 尽量落在能解释局部表面的平面附近。

第三步，从 open mesh edge 累积 boundary-distance QEF。这个设计很重要。许多真实资产不是封闭曲面，如果只使用普通 surface QEF，开放边界会变得不稳定。boundary QEF 为开边界提供额外几何约束。

第四步，加入 regularization QEF。它避免 dual vertex 被少量局部约束拉到不合理位置，使解更稳定。

第五步，求解 QEF 并生成 shape feature。输出的 $f^{\text{shape}}$ 包含 dual vertex、edge flag 和 split 参数。

这与传统 marching cubes 或 SDF extraction 的差异在于，O-Voxel 不需要先学习或构造连续标量场。它直接利用 mesh 与 voxel grid 的交点、法向和边界关系来构造可还原的局部几何特征。

## 8. O-Voxel-to-Mesh：从 sparse features 回到资产格式

![Algorithm 2: O-Voxel-to-Mesh conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-algorithm02-ovoxel-to-mesh.png)

*Source: Xiang et al., arXiv:2512.14692v1, Algorithm 2, arXiv non-exclusive distribution license.*

O-Voxel-to-Mesh 的逻辑比 mesh-to-O-Voxel 更直接。

第一，为每个 dual vertex 创建 mesh vertex。也就是把 $f^{\text{shape}}$ 里的局部 vertex 位置转换为全局 mesh 顶点。

第二，遍历 active edge，跨 voxel 连接顶点形成面片。如果相邻 quad coordinates 都存在，就根据 split 参数把 quadrilateral 分成两个 triangles。

第三，输出最终 mesh。

这个过程的工程意义很大：TRELLIS.2 不是只输出一个神经场，需要再用昂贵优化转换成 mesh。它的表示从一开始就是为了快速回到标准资产格式而设计的。官方 `o-voxel` README 也把 `flexible_dual_grid_to_mesh`、`to_glb` 等函数作为核心 API 暴露出来。

## 9. Texture-to-O-Voxel 与 O-Voxel-to-Texture

![Algorithm 3: Texture-to-O-Voxel conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-algorithm03-texture-to-ovoxel.png)

*Source: Xiang et al., arXiv:2512.14692v1, Algorithm 3, arXiv non-exclusive distribution license.*

Texture-to-O-Voxel 的输入是带 PBR texture 的 mesh 和已知 O-Voxel shape features。算法对每个 active voxel 查询 intersecting triangles，然后把 voxel center 投影到三角形表面，利用 UV 坐标采样 texture maps。多个 sample 通过距离权重融合，得到该 voxel 的 material feature。

这一步解决的是一个常被忽略的问题：如果训练数据是 textured mesh，材质存储在 UV texture 上，而模型 latent 又在 sparse voxel 上，必须有一个稳定的对齐方法。Algorithm 3 正是把 surface texture 转成 voxel-aligned material feature。

![Algorithm 4: O-Voxel-to-Texture conversion](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-algorithm04-ovoxel-to-texture.png)

*Source: Xiang et al., arXiv:2512.14692v1, Algorithm 4, arXiv non-exclusive distribution license.*

O-Voxel-to-Texture 有两种模式。

一种是 vertex 模式：对 mesh vertex 做 trilinear interpolation，直接生成 vertex materials。它简单、快，适合调试或快速可视化。

另一种是 map 模式：先对 mesh 做 UV parameterization，再对 texture map 中每个 texel 找到对应 surface point，从 O-Voxel material volume 中插值得到 PBR 属性。这个模式更接近标准 DCC/游戏资产管线，因为最终可以导出 texture maps。

官方 README 的 GLB export 示例也体现了这一点：`o_voxel.postprocess.to_glb` 会接收 `attr_volume`、`coords`、`attr_layout`、`voxel_size`、`aabb` 等信息，然后做 remesh、UV unwrap、texture baking 和 GLB export。

## 10. SC-VAE：为什么需要第二层 latent

O-Voxel 已经是 sparse 的，为什么还要 SC-VAE？

因为 O-Voxel 虽然比 dense voxel 紧凑，但对生成模型来说仍然太大。一个 `1024^3` 资产可能包含大量 active voxels。直接在 O-Voxel feature 空间训练 transformer/flow model，会遇到 token 数和上下文长度问题。

SC-VAE 的目标是学习一个更紧凑的 structured latent：

$$
z = E_\phi(f), \qquad \hat f = D_\theta(z),
$$

其中 $f$ 是 O-Voxel features，$z$ 是 compressed sparse latent，$\hat f$ 是重建的 O-Voxel。论文强调 16x spatial downsampling，即 latent 的空间分辨率比原始 O-Voxel 粗 16 倍，但 decoder 仍要恢复高保真的 shape 和 material。

![Fig. 4: SC-VAE architecture](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig04-scvae-architecture.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 4, arXiv non-exclusive distribution license.*

Fig. 4 里可以看到三个结构。

左侧是输入和输出：O-Voxel 的 $f^{\text{shape}}$、$f^{\text{mat}}$ 被编码成 shape SLat 和 material SLat。

中间是 shape/material SC encoder 和 decoder。它们是 fully sparse-convolutional，而不是普通 dense 3D CNN。这样做的原因是 active voxels 很稀疏，如果用 dense 3D 卷积，计算会浪费在空体素上。

右侧是 Sparse Residual Autoencoding layer。它既承担 down/up sampling，又通过 residual shortcut 缓解强空间压缩带来的信息丢失。

## 11. SC-VAE 的两阶段训练目标

论文把 SC-VAE 训练分成两个阶段。

第一阶段用低分辨率数据稳定学习 O-Voxel feature regression 和 KL regularization。几何侧对 dual vertex 位置使用 MSE，对 edge flags 使用 BCE；材质侧对 material attributes 使用 L1，对 pruning mask 使用 BCE。可以写成：

$$
\mathcal{L}_{s1}
= \lambda_v \|\hat v - v\|_2^2
+ \lambda_\delta \operatorname{BCE}(\hat \delta,\delta)
+ \lambda_\rho \operatorname{BCE}(\hat \rho,\rho)
+ \lambda_{\text{mat}}\|\hat f^{\text{mat}}-f^{\text{mat}}\|_1
+ \lambda_{\text{KL}}\mathcal{L}_{\text{KL}}.
$$

第二阶段加入 high-resolution rendering-based perceptual supervision。模型会渲染 mask、depth、normal，并使用 L1、SSIM、LPIPS 等损失增强视觉和几何细节。论文中总损失写成：

$$
\mathcal{L}_{s2}=\mathcal{L}_{s1}+\mathcal{L}_{\text{render}}.
$$

这两阶段设计对应一个实用判断：只在 feature space 做重建可能指标好，但渲染效果未必稳定；加入 rendering loss 可以把 surface quality、normal quality 和 PBR appearance 拉回人眼感知空间。

## 12. Table 4/5：架构并不复杂，但规模很大

![Table 4: SC-VAE architecture](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table04-scvae-architecture.png)

*Source: Xiang et al., arXiv:2512.14692v1, Table 4, arXiv non-exclusive distribution license.*

Table 4 展示 SC-VAE encoder。可以看到从 `1x` 到 `16x` 的下采样路径，每级由 SubMConv、LayerNorm、Linear、SiLU、ResEnc 等模块组成。论文说明 decoder 对称构造。附录中给出的规模是约 800M 参数，其中 encoder 约 354M，decoder 约 474M。

这解释了为什么本文说 TRELLIS.2 的“大”不只在最后生成模型。表示学习层本身已经是大型 sparse-convolutional autoencoder。

![Table 5: Generative model architecture](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table05-generative-model-architecture.png)

*Source: Xiang et al., arXiv:2512.14692v1, Table 5, arXiv non-exclusive distribution license.*

Table 5 展示生成模型。每个 flow model 是 encoder-only transformer，主要结构是：

- input projection：`Linear(32(+32), 1536)`；
- stem 重复 30 次；
- self-attention：12 heads，每头 128；
- cross-attention：12 heads，每头 128；
- FFN：`1536 -> 8192 -> 1536`；
- output projection：`LayerNorm + Linear(1536, 32)`。

论文还使用 RoPE 和 QK-Norm 稳定训练。每个 DiT 约 1.3B 参数，三阶段组合后约 4B。

## 13. Generative Modeling：三阶段生成不是装饰

TRELLIS.2 的生成不是一次性从图像输出完整资产，而是拆成三步。

```text
input image
  -> sparse structure flow
  -> shape SLat flow
  -> texture/material SLat flow
  -> SC-VAE decoders
  -> O-Voxel / mesh / PBR material
```

第一阶段生成 sparse structure。它决定大致哪些空间位置有 active voxels，类似资产的稀疏骨架。官方 pipeline 里对应 `sample_sparse_structure()`，输出 sparse coords。

第二阶段生成 shape SLat。它在 sparse coords 上采样 shape latent，再由 shape decoder 恢复 mesh geometry。官方代码里是 `sample_shape_slat()` 或 `sample_shape_slat_cascade()`，后者用于高分辨率 cascade。

第三阶段生成 texture/material SLat。它以 shape latent 为条件，生成材质 latent。官方代码中 `sample_tex_slat()` 会把 normalized shape SLat 拼接为 `concat_cond`，再采样 texture SLat。

这种顺序有明显工程意义：材质不应该先于几何生成。PBR material 需要和 shape surface 对齐，如果没有 shape condition，texture model 很容易生成漂浮、不匹配或无法导出的材料。

## 14. Flow Matching 目标

论文使用 rectified flow / conditional flow matching。可以把训练过程理解为在 data sample $x_0$ 和 noise sample $\epsilon$ 之间构造线性路径：

$$
x(t) = (1-t)x_0 + t\epsilon,\qquad t\in[0,1].
$$

目标 vector field 是：

$$
v(x,t)=\nabla_t x=\epsilon-x_0.
$$

模型 $v_\theta$ 在给定条件和时间步后预测这个向量场，训练目标为：

$$
\mathcal{L}_{\text{CFM}}(\theta)
= \mathbb{E}_{t,x_0,\epsilon}
\left\|v_\theta(x(t),t)-(\epsilon-x_0)\right\|_2^2.
$$

在 TRELLIS.2 中，$x_0$ 不是图像 latent，而是 sparse structure 或 structured latent。区别在于这些 latent 是不规则 sparse tokens，因此生成模型需要 sparse-aware transformer 和配套采样器。

## 15. Test-time Resolution Scaling

![Fig. 8: Test-time scaling](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig08-test-time-scaling.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 8, arXiv non-exclusive distribution license.*

Fig. 8 解释了 TRELLIS.2 的高分辨率策略。左侧是 scale up resolution，右侧是 scale up computation。

直觉上，低分辨率模型先生成稳定结构，再通过高分辨率 stage 补细节。官方 pipeline 的 `pipeline_type` 也反映了这种分层：

- `512`：直接生成 512；
- `1024`：直接使用 1024 flow；
- `1024_cascade`：先 512，再上采样到 1024；
- `1536_cascade`：先 512，再通过 cascade 生成更高分辨率。

代码里的 `sample_shape_slat_cascade()` 会先用低分辨率 flow 生成 latent，再调用 decoder 的 `upsample()` 产生高分辨率候选 coords，并根据 `max_num_tokens` 控制 token 数。如果 token 过多，代码会降低高分辨率目标。这是一个很实际的显存与质量折中。

## 16. FlexGEMM：系统性能不是附属品

![Fig. 9: FlexGEMM speed comparison](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig09-flexgemm-speed.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 9, arXiv non-exclusive distribution license.*

如果只读主文，很容易低估 FlexGEMM 的重要性。TRELLIS.2 的 SC-VAE 是 fully sparse-convolutional，训练和解码都高度依赖 sparse convolution。已有库如 Spconv、Torchsparse、fvdb、WarpConvNet 各有性能和平台边界。

论文附录把 FlexGEMM 描述为一个高性能 sparse convolution 后端，核心思路是 masked implicit GEMM。它把 feature gathering 和 GEMM 融合到更少 kernel 中，减少显存读写；还使用 Gray code ordering 增强邻域模式一致性，并使用 Split-K 增加并行度。

工程上这说明一个事实：TRELLIS.2 不是“把 PyTorch 模型下载下来就能轻松跑”的轻量方案。它同时依赖 `o-voxel`、`flex_gemm`、`cumesh`、`nvdiffrast`、`nvdiffrec` 等扩展。官方 README 的安装命令也显式包含这些组件：

```bash
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
```

## 17. 数据准备：不是随便抓一些 3D 模型

![Table 6: Dataset composition](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table06-dataset.png)

*Source: Xiang et al., arXiv:2512.14692v1, Table 6, arXiv non-exclusive distribution license.*

Table 6 展示训练和评估数据组成。论文使用 TexVerse、Objaverse-XL、ABO、HSSD 等来源，并使用 Toys4K 作为评估集之一。关键数字如下。

| 数据来源 | Shape available | Material available |
| --- | ---: | ---: |
| TexVerse | 503,387 | 382,996 |
| ObjaverseXL sketchfab | 168,307 | 141,623 |
| ObjaverseXL github | 293,887 | 202,188 |
| ABO | 4,485 | 4,485 |
| HSSD | 6,670 | 6,670 |
| SC-VAE training set | 473,349 | 354,966 |
| All training set | 976,736 | 737,962 |
| Toys4K evaluation set | 3,229 | 2,282 |

这里要注意 “shape available” 和 “material available” 的区别。不是每个 3D asset 都有可用 PBR 材质。论文还明确排除了缺少 PBR material 的 3D-FUTURE。对于 material VAE，作者用 Blender 脚本解析 raw assets，仅保留使用标准 metallic-roughness PBR workflow 的资产。

官方 `data_toolkit/README.md` 对应这条数据管线：

```text
build_metadata
  -> download
  -> dump_mesh / dump_pbr
  -> dual_grid / voxelize_pbr
  -> encode_shape_latent / encode_pbr_latent / encode_ss_latent
  -> render_cond
```

这个流程说明，TRELLIS.2 的训练准备不只是“下载 Objaverse 然后训练”。它需要先把 raw asset 标准化、抽取 mesh/PBR、转换 O-Voxel、编码 latent、渲染多视角条件图像，再训练 flow models。

## 18. Reconstruction 实验：看 Table 1 要看三类指标

![Table 1: Reconstruction comparison](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table01-reconstruction.png)

*Source: Xiang et al., arXiv:2512.14692v1, Table 1, arXiv non-exclusive distribution license.*

Table 1 是全文最重要的定量表之一。它比较 shape reconstruction 的效率和 fidelity，包含 Toys4K 与 Sketchfab Featured 两个评估集。

指标大致分三类。

第一类是 surface distance。表中的 MD 和 CD 都按 `x10^6` 报告，越低越好。它们衡量重建表面和真实表面的几何偏差。

第二类是 F-score。`F1_{1e-8}`、`F1_{1e-6}` 对应不同阈值，越高越好。它更强调表面是否在小距离阈值内匹配。

第三类是 normal rendering 质量。PSNR 越高越好，LPIPS 越低越好。它们用渲染视角衡量法线图的重建视觉质量。

表中最值得注意的是 Ours 1024 与 TRELLIS 的对比。二者 token 数都约 9.6K，但 Ours 的 latent 维度更高，分辨率下采样为 16x，解码时间为 0.301 秒；TRELLIS 的解码时间为 0.108 秒。也就是说，TRELLIS.2 并不是在所有维度都更轻，它用更复杂的 structured latent 换取更高的重建 fidelity 和 PBR 表达能力。

另一个对比是 SparseFlex 1024。SparseFlex 有 225K tokens，明显更重；Ours 1024 用 9.6K tokens 就达到更好的综合质量。这支撑了论文“compact structured latent”的主张。

## 19. Image-to-3D 生成：Table 2 与 Fig. 6

![Fig. 5: Generated assets](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig05-generated-assets.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 5, arXiv non-exclusive distribution license.*

Fig. 5 是 TRELLIS.2 的定性展示。它突出三类能力：复杂几何、开放曲面、半透明/反射材质。比如植物、机械、角色盔甲和 steampunk 设备，都不是简单 watertight toy shapes。

![Fig. 6: Generation comparison](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig06-generation-comparison.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 6, arXiv non-exclusive distribution license.*

Fig. 6 将 Ours 与 Hunyuan3D 2.1、Step1X-3D、TRELLIS、Direct3D-S2、Hi3DGen 做可视化对比。它不仅展示 normal，还展示最终 render、base color、metallic、roughness 等小图。图中 “No PBR” 的标记提醒我们，有些 baseline 并不输出完整 PBR 材质，因此不能只用 RGB 渲染图比较。

![Table 2: Image-to-3D generation results](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table02-generation.png)

*Source: Xiang et al., arXiv:2512.14692v1, Table 2, arXiv non-exclusive distribution license.*

Table 2 的指标分为 alignment、quality 和 user preference。

| 指标 | 含义 | 风险 |
| --- | --- | --- |
| CLIP | 输入图像与渲染图的视觉语义相似 | 可能偏向 2D 语义，不保证 3D 拓扑正确 |
| CLIP-N | 使用 normal map 的相似度 | 更关注几何，但仍是 proxy |
| ULIP-2 / Uni3D | 3D-aware 多模态模型相似度 | 依赖模型本身的训练偏差 |
| Pref% / Pref-N% | 用户研究偏好 | 样本量和展示协议会影响结果 |

Ours 在表中达到 CLIP `0.894`、CLIP-N `0.758`、ULIP-2 `0.477`、Uni3D `0.436`，用户偏好 `66.5%`，normal 相关偏好 `69.0%`。这些数字支持论文结论，但也要保守解读：它们不是生产资产质量的完整证明。真实生产还要检查 mesh topology、UV、材质层、LOD、物理尺度、动画兼容和 DCC 工具导入。

## 20. Shape-conditioned PBR Texture Generation

![Fig. 7: PBR texture generation](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig07-pbr-texture-generation.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 7, arXiv non-exclusive distribution license.*

Fig. 7 展示 shape-conditioned texture generation。输入给定几何和图像 prompt，模型生成对应 PBR 材质。对比方法包括 Hunyuan3D-Paint 和 TEXGen。

这部分的关键是“shape-conditioned”。TRELLIS.2 不是单独生成一张贴图，而是在 shape latent 条件下生成 material latent。官方 `Trellis2TexturingPipeline` 也对应这个思路：先用 `encode_shape_slat()` 把输入 mesh 编成 shape latent，然后用 texture flow 采样 material latent，最后通过 texture decoder 和 GLB 后处理得到 PBR 资产。

代码中 material layout 明确写成：

```python
{
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}
```

这说明材质生成不是单通道颜色补全，而是把 PBR 参数作为模型输出的一部分。

## 21. Ablation：SC-VAE 设计到底影响什么

![Table 3: SC-VAE ablation](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table03-scvae-ablation.png)

*Source: Xiang et al., arXiv:2512.14692v1, Table 3, arXiv non-exclusive distribution license.*

Table 3 比较 SC-VAE 架构设计。主线配置是 `SC-VAE f16c32`，代表 16x spatial downsampling 和 32 latent channels。消融包括去掉 Residual AE、去掉 optimized ResBlock，以及不同下采样/通道配置。

这张表支撑两个判断。

第一，16x 压缩不是简单平均池化能做到的。去掉 Residual AE 后，MD 和 F-score 变差，说明强空间压缩需要信息重排机制，否则细节会被下采样吃掉。

第二，optimized ResBlock 不是只为了省算力。它用更少卷积层和 pointwise MLP 代替传统残差块，在保持效率的同时改善稀疏数据上的特征表达。

工程上，这意味着如果要复刻一个简化版 TRELLIS.2，不能只写一个普通 sparse UNet VAE 然后期待相同压缩率。Residual autoencoding layer 和 sparse block 设计是重建质量的一部分。

## 22. User Study：主观偏好怎么读

![Fig. 10: User study interface](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-fig10-user-study-interface.png)

*Source: Xiang et al., arXiv:2512.14692v1, Fig. 10, arXiv non-exclusive distribution license.*

Fig. 10 展示用户研究界面。参与者面对两类问题：整体质量和形状质量。整体质量看 fully rendered asset，形状质量看 normal map，以减少材质对几何判断的干扰。

![Table 7: User study statistics](/images/blog/trellis2-native-structured-latents-3d-generation/trellis2-table07-user-study.png)

*Source: Xiang et al., arXiv:2512.14692v1, Table 7, arXiv non-exclusive distribution license.*

Table 7 给出详细统计。Ours 在 overall 上获得 135 次选择，占 `66.5%`；在 shape 上获得 147 次选择，占 `69.0%`。参与人数约 40，测试集为 100 个 AI-generated image prompts。

主观研究有价值，因为 3D 资产质量很难完全由自动指标覆盖。但它也有局限：展示界面、视角、渲染环境、样本选择和参与者背景都会影响偏好。对于工业落地，用户研究只能说明 perceptual quality，不等同于资产可生产使用。

## 23. 论文-代码对照：入口脚本与 pipeline

官方仓库 `microsoft/TRELLIS.2` 当前已经公开推理、纹理生成、训练和数据准备入口。核心文件关系如下。

| 论文模块 | 代码入口 | 说明 |
| --- | --- | --- |
| image-to-3D pipeline | `trellis2/pipelines/trellis2_image_to_3d.py` | 串联 sparse structure、shape SLat、texture SLat、decode |
| shape-conditioned texturing | `trellis2/pipelines/trellis2_texturing.py` | 输入 mesh 和图像，生成 PBR material |
| O-Voxel conversion | `o-voxel/` | mesh/O-Voxel/GLB 转换核心库 |
| SC-VAE models | `trellis2/models/sc_vaes/*` | shape/material VAE |
| flow models | `trellis2/models/sparse_structure_flow.py`, `structured_latent_flow.py` | sparse structure 和 SLat flow |
| sparse ops | `trellis2/modules/sparse/*` | SparseTensor、SparseConv、Sparse Transformer |
| data toolkit | `data_toolkit/` | metadata、mesh dump、PBR dump、O-Voxel、latent encode、render condition |
| training | `train.py`, `configs/scvae/*`, `configs/gen/*` | SC-VAE 和 flow model 训练配置 |
| demo | `app.py`, `app_texturing.py` | Gradio demo |
| minimal examples | `example.py`, `example_texturing.py` | 推理和 texture generation 示例 |

`Trellis2ImageTo3DPipeline` 的 `model_names_to_load` 直接体现论文三阶段：

```text
sparse_structure_flow_model
sparse_structure_decoder
shape_slat_flow_model_512
shape_slat_flow_model_1024
shape_slat_decoder
tex_slat_flow_model_512
tex_slat_flow_model_1024
tex_slat_decoder
```

`run()` 方法的执行顺序也很清楚：

```text
preprocess_image
  -> get_cond at 512/1024
  -> sample_sparse_structure
  -> sample_shape_slat or sample_shape_slat_cascade
  -> sample_tex_slat
  -> decode_latent
```

这和论文中的 “sparse structure -> shape -> material” 完全对应。

## 24. 代码对照：O-Voxel 库

`o-voxel/README.md` 对 O-Voxel 的定位与论文一致：field-free sparse voxel representation，支持 Flexible Dual Grid 和 volumetric PBR attributes，并提供 mesh/O-Voxel 双向转换。

关键 API 包括：

| API | 作用 |
| --- | --- |
| `mesh_to_flexible_dual_grid` | 从 mesh 得到 active voxels、dual vertices 和 intersected flags |
| `flexible_dual_grid_to_mesh` | 从 O-Voxel shape features 重建 mesh |
| `textured_mesh_to_volumetric_attr` | 把 PBR texture 采样到 sparse voxel 属性 |
| `o_voxel.io.read/write` | 读写 sparse voxel 文件 |
| `o_voxel.serialize.encode_seq` | 用 Morton/Hilbert 相关顺序编码 sparse coords |
| `o_voxel.postprocess.to_glb` | mesh cleaning、UV unwrap、texture baking、GLB export |

这说明官方实现不是只发布了模型推理脚本，而是把表示层也作为独立库暴露出来。对于想复现论文的读者，`o-voxel/examples/mesh2ovox.py`、`ovox2mesh.py`、`ovox2glb.py` 是比 `app.py` 更应该先读的代码。

## 25. 代码对照：数据准备

`data_toolkit/README.md` 给出的流程和论文 Appendix C 对齐：

1. `build_metadata.py` 初始化数据元信息。
2. `download.py` 下载 ObjaverseXL、ABO、HSSD、TexVerse、Toys4K 等子集。
3. `dump_mesh.py` 和 `dump_pbr.py` 抽取标准 mesh 和 PBR texture。
4. `dual_grid.py` 和 `voxelize_pbr.py` 转 O-Voxel。
5. `encode_shape_latent.py`、`encode_pbr_latent.py`、`encode_ss_latent.py` 编码 latent。
6. `render_cond.py` 渲染 image-conditioned flow 所需条件图。

这条链路解释了为什么论文训练不是“一行 train.py”。真正的成本在数据标准化和缓存生成。尤其 material pipeline 要过滤 PBR workflow，不是所有模型都能进入 material VAE 训练集。

## 26. 代码对照：训练配置

官方 README 将训练配置分为 SC-VAE 和 flow model。

SC-VAE：

```text
configs/scvae/shape_vae_next_dc_f16c32_fp16.json
configs/scvae/shape_vae_next_dc_f16c32_fp16_ft_512.json
configs/scvae/tex_vae_next_dc_f16c32_fp16.json
configs/scvae/tex_vae_next_dc_f16c32_fp16_ft_512.json
```

Flow models：

```text
configs/gen/ss_flow_img_dit_1_3B_64_bf16.json
configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json
configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json
configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json
configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json
```

这个命名很有信息量：

- `ss_flow` 表示 sparse structure flow；
- `img2shape` 表示 image-conditioned shape generation；
- `imgshape2tex` 表示 image + shape conditioned texture generation；
- `ft1024` 表示高分辨率 finetune；
- `1_3B` 对应约 1.3B 参数阶段模型；
- `bf16` 对应训练精度。

## 27. 代码对照：推理边界

官方 README 的 minimal example 使用：

```python
from trellis2.pipelines import Trellis2ImageTo3DPipeline

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
mesh = pipeline.run(image)[0]
```

随后示例会做 PBR video rendering 和 GLB export。值得注意的是 `mesh.simplify(16777216)`，README 注释说这是 nvdiffrast limit。这个细节说明即使模型输出很强，后处理和渲染工具仍有三角形数量、显存和纹理尺寸等工程上限。

官方 README 还特别说明 GLB 默认以 `OPAQUE` 模式导出，alpha channel 虽然保留在 texture map 中，但需要在 3D 软件里手动连接到 opacity 或 alpha input。也就是说，TRELLIS.2 输出 opacity 属性，不代表所有下游工具都会自动以透明材质加载。

## 28. 与 TRELLIS、Hunyuan3D、Step1X 等工作的关系

TRELLIS.2 与其他 image-to-3D 系统的差别可以从表示和资产管线看。

| 方法方向 | 主要特点 | TRELLIS.2 的不同点 |
| --- | --- | --- |
| TRELLIS | structured latent 3D generation | TRELLIS.2 强化 O-Voxel、PBR material 和更高压缩率 |
| Hunyuan3D / Step1X-3D | 多阶段几何/纹理生成系统 | TRELLIS.2 更强调 native 3D PBR 属性和 O-Voxel 双向转换 |
| Direct3D-S2 | 高效 3D 生成 | TRELLIS.2 在重建和 PBR 表达上更激进 |
| Hi3DGen | 高质量 3D 生成 | TRELLIS.2 强调开放/非流形/内部结构和 PBR 输出 |
| SparseFlex | 稀疏结构高保真重建 | TRELLIS.2 用更少 latent tokens 追求 compactness |
| Dora / GaussianCube | 其他 3D latent 或 sparse 表示 | TRELLIS.2 更关注 mesh/PBR 原生资产转换 |

这不是说 TRELLIS.2 已经解决所有 3D 生成问题，而是它把竞争点从“生成图像看起来像 3D”推进到“生成资产是否能以 mesh + PBR material 形式进入下游管线”。

## 29. 与本站其他 3D 场景论文的关系

本站此前几篇 3D 论文报告关注点不同：

| 文章 | 核心任务 | 与 TRELLIS.2 的关系 |
| --- | --- | --- |
| CompoNeRF | 文本引导多物体组合 NeRF | 更偏 NeRF 优化和布局组合，非资产级 PBR latent |
| Blended-NeRF | 现有 NeRF 场景局部 ROI 编辑 | 更偏已有场景编辑和 CLIP guidance |
| Layout2Scene | 3D semantic layout guided scene generation | 更偏场景级 semantic layout 与两阶段优化 |
| CasLayout | 3D indoor layout diffusion | 生成家具布局，不生成可用资产几何/材质 |
| Imaginarium | 图像引导资产检索与物理布局 | 用资产库检索和 Blender 优化，不从 latent 直接生成资产 |
| TRELLIS.2 | image-to-3D asset generation | 关注单体/资产级几何、PBR、导出和压缩表示 |

因此 TRELLIS.2 可以被看作资产生成底座，而不是场景布局系统。它生成的 GLB/PBR asset 未来可以作为 Imaginarium 或 CasLayout 这类场景系统的资产候选，但它本身不解决房间布局、物理摆放、碰撞和场景语义一致性。

## 30. 工程复现清单

如果要从零复现或深入跑 TRELLIS.2，需要按以下层次准备。

### 环境

- Linux；
- NVIDIA GPU，至少 24GB 显存；
- A100/H100 是官方验证环境；
- CUDA Toolkit 推荐 12.4；
- Python 3.8+；
- Conda；
- `flash-attn` 或在不支持时切到 `xformers`；
- `nvdiffrast`、`nvdiffrec`、`cumesh`、`o-voxel`、`flexgemm`。

### 推理

- 下载 `microsoft/TRELLIS.2-4B`；
- 运行 `example.py` 验证 image-to-3D；
- 检查 `pipeline_type`：`512`、`1024`、`1024_cascade`、`1536_cascade`；
- 关注 `max_num_tokens` 对显存和质量的影响；
- 渲染检查 normal、base color、metallic、roughness、alpha；
- 导出 GLB 后用 Blender 或其他 DCC 工具检查材质连接和透明度。

### 数据准备

- 构建 metadata；
- 下载原始 3D assets；
- dump mesh；
- dump PBR；
- 计算 asset stats；
- 转 dual grid 和 PBR voxel；
- encode shape latent、PBR latent、sparse structure latent；
- 渲染 16 view 条件图；
- 更新 metadata。

### 训练

- 先训练 shape SC-VAE；
- 再训练 texture SC-VAE；
- 训练 sparse structure flow；
- 训练 shape flow；
- 训练 texture flow；
- 如需高分辨率，做 1024 finetune；
- 使用多 GPU、bf16、分布式训练配置；
- 单独监控 reconstruction fidelity、rendering loss、material loss 和 sample quality。

## 31. 局限性与批判

TRELLIS.2 的贡献很强，但不能把它读成“3D 生成已解决”。

第一，复现成本高。官方代码公开不等于普通开发机可复现。Linux、A100/H100、CUDA 扩展、多数据集、多阶段训练都提高门槛。

第二，数据质量是核心瓶颈。论文花了大量工作筛选 PBR assets、渲染多视角条件图、构建 material available 子集。换一个数据域，O-Voxel 和 SC-VAE 未必自动保持同样质量。

第三，自动指标仍然不完整。CLIP、ULIP-2、Uni3D、PSNR、LPIPS 都是 proxy。真实资产生产还要看 topology、UV、材质节点、scale、LOD、动画绑定、碰撞体和引擎兼容。

第四，PBR 输出不等于完美材质。模型能生成 base color、metallic、roughness、alpha，但下游渲染器、颜色空间、环境光、透明模式和纹理压缩都会影响最终效果。

第五，高分辨率 cascade 有 token budget。代码里显式存在 `max_num_tokens` 和 resolution fallback，这说明极复杂资产在高分辨率下仍受显存和算力约束。

第六，单资产生成不等于场景生成。TRELLIS.2 对单体资产很强，但不负责多资产布局、物理关系、语义一致性、交互编辑和场景级导航。

## 32. 推荐阅读路径

如果只想快速理解论文，建议这样读：

1. 先读 Abstract、Fig. 1、Fig. 2，抓住表示层、SC-VAE 和 flow model 的主线。
2. 再读 Section 3.1 和 Fig. 3，理解 O-Voxel 为什么不是普通 sparse voxel。
3. 读 Section 3.2 和 Fig. 4，理解 16x downsampling 和 SC-VAE 结构。
4. 读 Table 1 和 Table 2，分清 reconstruction fidelity 与 generation quality。
5. 读 Appendix A 的 Algorithm 1-4，确认 mesh/PBR 双向转换的细节。
6. 读 Appendix B，理解 FlexGEMM 为什么影响系统可用性。
7. 最后看官方 README、`trellis2/pipelines/trellis2_image_to_3d.py`、`o-voxel/README.md`、`data_toolkit/README.md`。

如果要做工程验证，顺序应该反过来：

1. 先跑 `o-voxel` examples，确认 mesh/O-Voxel/GLB 转换；
2. 再跑 `example.py`，确认预训练 pipeline；
3. 再尝试 `example_texturing.py`；
4. 最后才看训练配置和数据准备。

## 33. 结论

TRELLIS.2 的长期价值在于它没有把 3D 生成问题简化为“更大的 diffusion/flow 模型”。它先提出 O-Voxel，把开放曲面、非流形结构、内部结构和 PBR 材质放进一个可双向转换的 sparse structured representation；再用 SC-VAE 将高分辨率 O-Voxel 压缩成 compact structured latent；最后用三阶段 flow matching 生成 sparse structure、shape 和 material。

从研究角度看，它把 3D 生成的瓶颈前移到表示学习：只有当 latent 本身足够原生、紧凑、可还原，生成模型的规模化才有意义。

从工程角度看，它给出了一个接近真实资产管线的方向：不是只生成看起来像 3D 的渲染图，而是生成可以还原为 mesh、PBR attributes 和 GLB 的资产。但它仍然是重型系统，依赖高质量数据、CUDA 扩展、高端 GPU 和后处理链路。读这篇论文时，最该学的是表示层设计和系统分解，而不是简单记住“4B 参数”和几个速度数字。

## References

- [Native and Compact Structured Latents for 3D Generation, arXiv:2512.14692](https://arxiv.org/abs/2512.14692)
- [PDF: Native and Compact Structured Latents for 3D Generation](https://arxiv.org/pdf/2512.14692)
- [TRELLIS.2 Project Page](https://microsoft.github.io/TRELLIS.2/)
- [Official GitHub: microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
- [Hugging Face Model: microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)
- [Hugging Face Demo: TRELLIS.2](https://huggingface.co/spaces/microsoft/TRELLIS.2)

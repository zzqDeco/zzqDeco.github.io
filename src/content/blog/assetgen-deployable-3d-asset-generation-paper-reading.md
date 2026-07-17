---
title: "AssetGen 论文精读：可部署 3D 资产生成、MeshGen/TextureGen 与交互式低延迟系统"
description: "精读 AssetGen 如何通过两阶段 MeshGen、GPU 几何处理、UV 展开、法线烘焙、多视角 TextureGen、蒸馏和流水线优化实现 30 秒级可部署 3D 资产生成"
pubDate: "2026-07-09T16:14:31+08:00"
updatedDate: "2026-07-09T16:14:31+08:00"
tags:
  - "Paper Reading"
  - "3D Generation"
  - "Image-to-3D"
  - "Computer Graphics"
  - "Generative AI"
  - "Systems"
draft: false
---

AssetGen 这篇论文真正值得精读的地方，不是它又做了一个 image-to-3D 模型，而是它把 3D 生成问题的评价标准往工程交付方向推了一步。

过去几年 image-to-3D、text-to-3D、multi-view-to-3D 的论文很多，常见目标是生成更高分辨率的几何、更漂亮的渲染图、更高的 CLIP 分数或更好的多视角一致性。但如果你要把生成结果放进游戏、AR/VR、移动端实时渲染、创作者工具或 agentic asset creation loop，单纯“看起来像”还不够。一个可部署 3D 资产至少要满足几个更朴素但更硬的条件：

- mesh 不能无限高面数，要有可控 polygon budget。
- 要有 UV coordinates 和 texture atlas，否则很难进入普通实时引擎管线。
- 高模细节需要通过 normal baking 转移到低模上。
- 纹理要能稳定贴回 UV atlas，而不是只在几张渲染图里看起来合理。
- 整个流程不能动辄几分钟，否则交互式创作和 agent 循环会断掉。

AssetGen 的核心贡献就在这里：它把 image-to-3D 重新定义为 **deployable 3D asset generation at interactive speed**。论文声称默认 AssetGen 在 H100 部署上约 30 秒输出一个 simplified、UV-unwrapped、normal-baked textured mesh；AssetGen Flash 则以约 14 秒生成 preview 级结果。这个目标把模型、几何处理、纹理融合和系统优化绑在一起，而不是只优化某一个 diffusion block。

本文精读对象是 arXiv `2605.26137v1`：**AssetGen: Deployable 3D Asset Generation at Interactive Speed**。用户原始链接写成了 `https://arxiv.org/pdf/2605/26137`，规范 PDF 地址应为 `https://arxiv.org/pdf/2605.26137`。arXiv 页面显示论文提交于 2026-05-22，subjects 为 `cs.GR`、`cs.AI`、`cs.CV`，DOI 为 `10.48550/arXiv.2605.26137`。论文 PDF 共 30 页，license 显示为 arXiv perpetual non-exclusive license。

实施时我检查了 arXiv 页面、HTML 版本和 TeX source。没有发现 AssetGen 官方 GitHub、project page、checkpoint 或 demo 链接；源文件中引用了 CuMesh、xatlas、DRTK、FlashAttention-3、PyTorch compile 等依赖或相关组件，但它们不是 AssetGen 完整开源实现。因此本文只做论文级精读、图表解读和工程复现清单，不声称完成代码复现。

## 1. 一句话贡献

AssetGen 的一句话贡献是：

> 从单张参考图生成一个能直接进入实时渲染工作流的显式 3D asset，并把几何生成、几何后处理、纹理生成、UV 回投、法线烘焙和系统延迟一起优化。

这句话里有三个关键词。

第一是 **asset**。论文关心的不是 NeRF、3DGS 或多视角图片本身，而是显式 mesh、UV、normal map、texture atlas 组成的可导出资产。

第二是 **deployable**。结果要控制面数、支持移动端和实时渲染，不是只在论文渲染器里展示。

第三是 **interactive speed**。30 秒和 14 秒不是普通 benchmark 小数点，而是论文想服务的产品体验：用户或 agent 可以快速试错、评估、修改、再生成。

## 2. 论文信息与版本边界

论文基本信息如下。

| 项目 | 内容 |
| --- | --- |
| Title | AssetGen: Deployable 3D Asset Generation at Interactive Speed |
| arXiv | `2605.26137v1` |
| Date | 2026-05-22 |
| Authors | Dilin Wang, Xiaoyu Xiang, Kihyuk Sohn, Tom Monnier, Yu-Ying Yeh, Thu Nguyen-Phuoc, Jiawen Zhang, Yuchen Fan, Antoine Toisoul, Hyunyoung Jung, Prithviraj Dhar, Michael Bunnell, Nikolaos Sarafianos, Chuhang Zou, Roman Shapovalov, Andrea Vedaldi, Rakesh Ranjan |
| Institution | Reality Labs, Meta |
| PDF | 30 pages |
| DOI | `10.48550/arXiv.2605.26137` |
| License | arXiv perpetual non-exclusive license |
| Code status | 未发现官方代码、模型权重或项目页 |

论文的摘要已经把系统定位讲得很清楚：给一张参考图，默认配置在约 30 秒生成高质量 mesh、baked normals、color texture 和 controlled polygon budget；Flash 变体进一步降到约 14 秒，服务于更快的交互和 agentic creation loop。

这里要先避免两个误读。

第一，30 秒和 14 秒是论文报告的 H100 部署结果，不是任何普通 GPU 或本地机器都能达到的延迟。

第二，AssetGen 输出的是 color texture 和 normal-baked mesh。它和强调 PBR material decomposition 的 Meta 3D AssetGen 2024 不是同一个侧重点。本文会在后面单独对比。

## 3. 为什么 3D 生成不等于可部署资产

一个生成模型如果输出了高分辨率 mesh，并不意味着它已经可部署。游戏和移动端实时渲染管线一般关心：

| 维度 | 研究 demo 常见做法 | 可部署资产需要 |
| --- | --- | --- |
| 几何 | 高面数 mesh、implicit field、3DGS、NeRF | 控制 polygon budget，能实时渲染 |
| 纹理 | 多视角渲染看起来好 | UV atlas，稳定贴图 |
| 细节 | 高模几何直接保留 | normal baking 到低模 |
| 拓扑 | marching cubes 任意三角网格 | 尽量轻量、干净、可导出 |
| 延迟 | 分钟级也可接受 | 交互式工作流希望几十秒内 |
| 后处理 | 手工或离线工具完成 | pipeline 内自动完成 |

AssetGen 的系统设计就是围绕这个表展开的。MeshGen 负责生成高质量几何先验；geometry processing 把它变成低面数、UV、normal-baked mesh；TextureGen 负责把 reference appearance 贴回 UV atlas；latency optimization 确保这些步骤不是串行堆加到几分钟。

## 4. Fig. 1：论文真正的目标对象

![AssetGen teaser](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig01-teaser.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 1, arXiv perpetual non-exclusive license.*

Fig. 1 展示了论文的输出目标：不是一组新视角渲染图，而是一组可作为资产查看的对象。caption 里几个词很关键：high-fidelity、UV-unwrapped、normal-baked、textured 3D meshes、real-time rendering、approximately 30 seconds on H100 GPUs。

这张图解决的问题是“AssetGen 到底交付什么”。很多 image-to-3D 论文在 teaser 中展示的是渲染质量，AssetGen 则强调 asset contract：

```text
input image
  -> simplified mesh
  -> UV coordinates
  -> baked normal map
  -> color texture atlas
  -> exported textured mesh
```

这个 contract 比“生成一个 3D 表示”更接近创作者工具和实时引擎的接口。

需要注意的是，teaser 图不能证明所有类别都稳定，也不能证明拓扑适合 rigging。论文自己的 limitation 也承认，marching-cubes extraction 产生的是任意三角网格，简化流程能减面并保持表面相似，但不保证 artist-friendly edge flow、skeleton rigging 或 animation-ready deformation。

## 5. Fig. 2：系统总览与两种运行模式

![AssetGen system overview](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig02-system-overview.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 2, arXiv perpetual non-exclusive license.*

Fig. 2 是全文最重要的流程图。它把 AssetGen 拆成三大阶段：

1. **MeshGen**：从单张输入图生成密集、高细节几何。
2. **Geometry processing**：mesh simplification、hidden-face removal、UV unwrapping、normal baking。
3. **TextureGen**：生成多视角纹理，回投到 UV atlas，做 texture inpainting，最后导出 textured mesh。

默认 AssetGen 和 AssetGen Flash 的差别也在这张图里：

| 模式 | 目标 | 关键差异 |
| --- | --- | --- |
| AssetGen | 质量优先 | 两阶段 MeshGen，TextureGen 1024px，inference-time CFG，per-view super-resolution，2K texture atlas |
| AssetGen Flash | 延迟优先 | 跳过 MeshGen refinement，TextureGen 768px，CFG distilled conditional-only model，省略 super-resolution，1K atlas |

这说明 AssetGen 不是单模型，而是 pipeline productization。Flash 不是另一篇论文意义上的“小模型 baseline”，而是同一系统里的更快 operating point。

## 6. 数据过滤：可部署质量从数据开始

![AssetGen data filtering](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig03-data-filtering.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 3, arXiv perpetual non-exclusive license.*

Fig. 3 展示了 semantic filtering 移除的资产类型：背景几何、3D scans、错误朝向、低质量 mesh。

这部分容易被读者跳过，但对可部署资产非常关键。训练数据如果混入大量背景、扫描残缺、姿态不一致或低质量 mesh，模型会学到错误先验。对于 image-to-3D 来说，这类错误会表现为：

- 生成对象带上地面、墙面或摄影背景。
- 正面/背面方向混乱，导致 reference fidelity 指标不稳定。
- mesh 拓扑破碎，后续简化、UV、normal baking 都变难。
- TextureGen 学到不一致的多视角纹理。

论文从 in-house licensed datasets 开始，做 duplicate removal、metadata/vertex count filtering、semantic filtering 和 geometric filtering。这提醒我们：AssetGen 的质量不是只靠大模型，更靠资产生产链路的数据清洗。

## 7. Flood-fill sign estimation：SDF 训练的地基

![Flood-fill sign estimation](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig04-flood-fill-sign-estimation.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 4, arXiv perpetual non-exclusive license.*

MeshGen 依赖 SDF 表示。SDF 的关键是空间点到表面的 signed distance：点在物体外还是物体内。如果 mesh 是 watertight 且 winding 正确，inside/outside 可以相对稳定地判断。但真实 3D 资产经常有开口、破洞、法线方向混乱和非流形结构。

Fig. 4 展示了论文的 flood-fill-based voxel sign estimation。直觉上：

1. 在体素网格上标记靠近表面的区域。
2. 从网格边界 flood fill，找到外部区域。
3. 对表面 band 做 dilation，封住小缝隙，避免 flood fill 漏进内部。
4. 最终得到 inside/outside 分区，给 SDF 训练提供符号。

这个步骤不是生成模型本身，但它决定了 MeshGen 训练数据是否可靠。很多 3D 生成论文把数据预处理一笔带过，AssetGen 把它放进正文，是因为它服务的是 asset pipeline，而不是只展示少量最佳样例。

## 8. MeshGen：从参考图到密集几何

MeshGen 的输入是一张参考图，输出是高细节 3D 几何。它不是直接输出最终低模，而是先生成 dense shape，再交给后处理压缩成可部署 mesh。

论文采用 VecSet 表示。VecSet VAE 把输入 shape 编码成一组 latent tokens：

$$
\mathbf{z}\in\mathbb{R}^{N\times D}
$$

decoder 则把 $\mathbf{z}$ 映射成可查询的 SDF function。生成时，MeshGen 在 VecSet latent space 上做 conditional diffusion。

给定 clean latent $\mathbf{z}_0$ 和噪声 $\epsilon$，noisy latent 写成：

$$
\mathbf{z}_t=\alpha_t\mathbf{z}_0+\sigma_t\epsilon
$$

论文使用 v-prediction：

$$
\mathbf{v}_t=\alpha_t\epsilon-\sigma_t\mathbf{z}_0
$$

训练目标为：

$$
\mathcal{L}_{DM}
=\mathbb{E}_{t,\epsilon,\mathbf{z}_0}
\left[
\left\|
\phi_\theta(\mathbf{z}_t,t,\mathbf{c})-\mathbf{v}_t
\right\|^2
\right]
$$

其中 $\mathbf{c}$ 是图像条件。论文使用 frozen DINOv2 ViT-G/14 with registers 编码输入图像，产生 patch tokens 作为 cross-attention 条件。

## 9. MeshGen 架构：2.3B DiT 与 VecSet tokens

MeshGen backbone 遵循 DiT 设计：self-attention 处理 latent tokens，cross-attention 注入图像条件，AdaLN 做 timestep modulation，FFN 使用 SwiGLU。

论文给出的 coarse stage 配置包括：

| 项目 | 配置 |
| --- | --- |
| 参数量 | 约 2.3B |
| blocks | 24 |
| hidden dim | 2048 |
| heads | 16 |
| latent tokens | $N=4096$ |
| token dim | $D=64$ |
| image encoder | DINOv2 ViT-G/14 with registers |
| image resolution | 518px |

推理时还会使用 classifier-free guidance：

$$
\tilde{\mathbf{v}}_t
=\mathbf{v}^{uncond}_t
+s\left(\mathbf{v}^{cond}_t-\mathbf{v}^{uncond}_t\right)
$$

这会让每个 denoising step 需要 conditional 和 unconditional 两次 forward。后面 MeshGen distillation 的一大价值，就是把 CFG folding 进 student model，减少推理时的双分支成本。

## 10. Coarse-to-refine MeshGen

![MeshGen dense meshes](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig05-meshgen-dense-meshes.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 5, arXiv perpetual non-exclusive license.*

Fig. 5 展示了 MeshGen 在几何处理和简化前预测出的 dense meshes。这些结果细节丰富，但显然不是最终移动端或实时渲染要用的低面数资产。

论文使用 coarse-to-refine 两阶段：

1. **Coarse stage** 用 $N=4096$ tokens 建立全局形状和拓扑。
2. **Refinement stage** 使用 $N_{refine}=16384$ tokens，在 coarse mesh 表面附近补充局部细节。

refinement tokens 的位置来自 coarse mesh surface sampling 和 voxelization，因此每个 token 有空间锚点。论文强调这和 coarse stage 不同：coarse tokens 没有先验空间关联，而 refinement tokens 被绑定到 coarse geometry 的局部区域，更容易恢复细节。

refinement stage 还用更高分辨率 image conditioning：DINOv2 encoder 在 1022px resolution 上运行，产生约 4 倍 patch tokens。这说明细节恢复不仅依赖更密的 3D tokens，也依赖更高分辨率视觉条件。

## 11. MeshGen Distillation：从 120 步到 30 步

默认 MeshGen 原始推理配置包含 100 步 coarse denoising 和 20 步 refinement denoising，并且都使用 CFG。考虑到 CFG 需要双 forward，这个成本非常高。

论文采用 progressive distillation：

- coarse stage 从 100 步降到 25 步。
- refinement stage 从 20 步降到 5 步。
- 同时把 CFG folding 到 student model，让 student 学习 guided teacher output。

训练 student 时，teacher 从随机 student timestep 开始跑多个 deterministic DDIM steps，得到 multi-step target；student 用单步 MSE 去匹配这个 target。

![MeshGen distillation human evaluation](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table01-meshgen-distillation-human-eval.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 1, arXiv perpetual non-exclusive license.*

Table 1 是一个很有用的 sanity check。Teacher 是 120 steps，A100 上每样本 28.20 秒；Student 是 30 steps，3.78 秒。人评结果里 teacher wins 32.9%，student wins 30.5%，similar 36.6%。这不是证明 student 永远等同 teacher，而是说明在论文采样的人评协议里，4 倍 step reduction 没有明显感知质量崩溃。

这也解释了 AssetGen 的系统思路：如果最终目标是交互式 asset pipeline，模型推理不能只追求最高质量，还要进入可接受的 quality-latency Pareto 区域。

## 12. Geometry Post-processing：dense mesh 变成 runtime asset

MeshGen 输出的是 dense SDF / dense mesh。要变成 runtime asset，需要一组几何处理步骤：

```text
dense mesh
  -> mesh simplification
  -> hidden face removal
  -> UV unwrapping
  -> tangent-space normal baking
  -> simplified mesh + UV + normal map
```

这部分是 AssetGen 与许多 3D generation 论文的关键差别。很多方法生成高质量几何后，后处理交给用户或离线工具；AssetGen 把它放进模型系统的 critical path，并对延迟做优化。

## 13. Mesh simplification：控制 polygon budget

![Mesh simplification table](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table02-mesh-simplification.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 2, arXiv perpetual non-exclusive license.*

Table 2 比较了 5K 和 10K face targets 下的简化质量。指标包括：

| 指标 | 含义 |
| --- | --- |
| Latency | 简化耗时 |
| Chamfer | 简化 mesh 与原 mesh 点集距离 |
| Hausdorff | 最大最小距离，更关注 worst-case deviation |
| Flipped Normals | 多视角下法线方向不一致的像素 |
| Baked Mean | 烘焙法线后的角度误差 |
| Geo Mean | 简化几何自身法线误差 |

论文基于 CuMesh 风格的 GPU parallel edge collapse 和 QEM。QEM 的直觉是，每次边收缩都计算几何误差，选择低误差收缩以减少面数。

重要的是，AssetGen 不只看 Chamfer。对于实时渲染资产，normal map 能否保留高频细节同样关键，所以 Baked Mean 也被纳入评价。这比单纯“几何距离更小”更接近资产交付质量。

## 14. Hidden face removal：不要把不可见内面带进游戏

生成式 3D mesh 常常包含内部面、重叠面或用户永远看不到的背面结构。对于离线展示，这可能不明显；对于实时渲染和移动端，这些面会浪费三角形预算、增加 UV 展开复杂度、干扰 normal baking 和纹理生成。

AssetGen 在 geometry processing 中加入 hidden face removal，目标是把有限 polygon budget 留给可见表面。

这一步的风险也很明显：如果移除过度，某些后续视角或动画变形下可能出现洞；如果移除不足，低模资产又不够轻。论文把它放在可部署资产 pipeline 中，说明它不是可选美化，而是 assetization 的核心环节。

## 15. UV unwrapping：TextureGen 的接口

AssetGen 使用 xatlas 做 UV parameterization，并进一步探索 parallel chart segmentation。UV 是 TextureGen 的关键接口：多视角生成的颜色最终要回投到一个 texture atlas 上，而不是停留在独立视角图片里。

论文实际有一张 `Table3`，PDF 文本中排版为无空格的 `Table3`。它比较 serial 和 parallel xatlas 的 UV unwrapping 人评质量与速度。

![UV unwrapping human evaluation](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table03-uv-unwrapping-human-eval.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 3, arXiv perpetual non-exclusive license.*

这张表的重点是：parallel 版本速度更快，同时人评上没有明显质量损失。论文把 processing speed 归一化到 serial baseline，结果显示 parallel 从 4.70 秒降到 2.56 秒。对于 30 秒端到端目标来说，这类“看似传统图形学小优化”非常重要。

## 16. Normal baking：把高模细节搬到低模

![Normal baking table](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table04-normal-baking.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 4, arXiv perpetual non-exclusive license.*

Normal baking 的目标是把 high-poly mesh 的局部表面细节编码到 low-poly mesh 的 tangent-space normal map 中。这样低模在实时渲染时仍能表现出更丰富的光照细节。

Table 4 比较了 GPU-resident implementation 和 Blender。GPU baker 平均 271ms，Blender 平均 2920ms；最坏情况 GPU 1147ms，Blender 25854ms。质量指标上，Baked Mean 也接近或略优。

这张表支撑了论文的一个核心判断：AssetGen 的速度不是只靠 diffusion distillation。传统资产处理步骤如果仍然调用慢速离线工具，端到端延迟仍然会被拖垮。GPU-resident normal baking 是系统性能的一部分。

## 17. TextureGen：从 processed mesh 到 texture atlas

Geometry pipeline 输出的是 simplified mesh、UV、baked normal map 和相关 metadata。TextureGen 接下来要做的事情是：基于参考图和几何条件，生成多视角彩色外观，然后融合到 UV atlas。

训练数据构造包括：

- 从随机相机视角渲染 reference image。
- 从 10 个固定视角渲染 target images：8 个 side views，加 top 和 bottom。
- 使用 orthographic projection，让目标视图覆盖更稳定。
- 渲染 normal maps 和 position maps 作为几何条件。
- 使用不同 HDRI lighting 增强对真实参考图光照变化的鲁棒性。

论文明确说 TextureGen 生成的是 color texture，而不是 PBR material。这是一个重要边界：它更适合移动端和 runtime rendering 的简单管线，但不像 Meta 3D AssetGen 2024 那样主打 albedo、metalness、roughness 等 PBR 分解。

## 18. TextureGen Architecture：多视角 diffusion transformer

TextureGen 的核心是多视角 diffusion transformer。设参考图为 $\mathbf{I}$，processed mesh 为 $\mathcal{M}$，第 $k$ 个目标视角的 normal map 和 position map 为 $\mathbf{N}_k,\mathbf{P}_k$，模型同时生成 $K$ 个 target views：

$$
\{\mathbf{V}_k\}_{k=1}^{K}
$$

论文使用 pretrained image VAE latent。每个 $1024\times1024$ view 被编码为 $64\times64$ latent。reference image 和 $K$ 个 target views 被组织成 $1\times(1+K)$ latent grid：

- 第一格是 pinned reference latent，在 denoising 中固定。
- 其余格是 noisy target-view latents。
- loss 只作用在 target-view slots。

模型条件包括：

| 条件 | 作用 |
| --- | --- |
| reference image latent | appearance anchor，提供颜色、材质、风格 |
| normal maps | 表面朝向，帮助纹理与几何一致 |
| position maps | world-space correspondence，帮助跨视角对齐 |
| text prompt | 补充语义信息 |
| T5 / CLIP embeddings | 通过 cross-attention 注入文本条件 |

## 19. TextureGen 的 flow matching

TextureGen 不只是图像回归。论文使用 optimal-transport conditional flow matching，训练模型预测 velocity field：

$$
\mathbf{v}_\theta(\mathbf{z}_t,t,\mathbf{c})
$$

其中 $\mathbf{z}_t$ 是 noisy multi-view latent，$\mathbf{c}$ 包含 reference、geometry renders 和 text conditioning。

这和 MeshGen 的 diffusion 形成了一个很清楚的系统分工：

| 模块 | 表示空间 | 生成对象 | 加速方式 |
| --- | --- | --- | --- |
| MeshGen | VecSet shape latent | 3D geometry | progressive distillation + CFG folding |
| TextureGen | image VAE multi-view latent | target color views | progressive distillation + CFG distillation |
| Postprocess | UV atlas | final texture | GPU backprojection + blending + inpainting |

## 20. Structured multi-view attention

TextureGen 需要同时处理 reference slot 和 10 个 target views。如果每个 view 是 $64\times64$ latent tokens，那么 $1+K=11$ 个 views 共 45,056 tokens。全 self-attention 成本很高。

论文使用与相机布局匹配的 sparse attention pattern：

- 每个 target view 都看 reference slot。
- cardinal side views 看相邻两个 views。
- diagonal views 看 reference 和相邻 cardinal views。
- top/bottom views 看 reference 和四个 cardinal side views。

这个设计的直觉是：纹理一致性主要来自 reference 和相邻重叠视角，不需要每个视角都全局关注所有视角。它把 attention context 约减少 4 倍，同时保留跨视角传播路径。

## 21. Texture backprojection、blending 和 inpainting

TextureGen 生成的是多张 target-view images，最终还要映射到 UV atlas。论文采用 backward projection：对 atlas 中每个 texel，恢复对应 3D surface position，再投影到每个生成视角读取颜色。

边界和深度不连续处容易采样错误颜色，因此先用 point coordinate maps 的邻域差异构造 edge mask。对于每个 texel，还要处理 grazing angle 和 UV seam 造成的采样 footprint 问题，所以论文使用 anisotropic filtering 和 mip chain。

每个 texel 可能从多个视角获得候选颜色。论文用 incidence-weighted blending：

$$
T
=
\frac{
\sum_{k=1}^{K} w_k I_k^\alpha \odot A_k
}{
\sum_{k=1}^{K} w_k I_k^\alpha+\epsilon
}
$$

其中：

- $A_k$ 是第 $k$ 个视角回投得到的 partial atlas。
- $I_k$ 是该 texel 在第 $k$ 个视角下的 incidence / visibility。
- $w_k$ 是 view prior，给正面、背面等更重要视角更高权重。
- $\alpha$ 控制 incidence 权重的 sharpness。

最后，对没有可靠观测的 texels，论文使用 3D-aware inpainting：基于 texel 的 3D position 和 normal，在已知 texel 中找几何邻近且法线相近的颜色进行传播。这比单纯在 2D UV 空间做邻近填补更合理，因为 UV 空间相邻不一定代表 3D 表面相邻。

## 22. Geometric condition rendering：从 Blender 到 DRTK

TextureGen 推理时需要渲染 10 个视角的 normal/position maps。论文训练时用 Blender，但推理时替换为 DRTK GPU rasterizer。

论文报告 Blender 渲染 10 views 约 5 秒，DRTK 约 500ms。在 30 秒 pipeline 中，这 4.5 秒差距很大。更关键的是，DRTK 输出 GPU-resident tensors，避免频繁 CPU/GPU 或磁盘 I/O。

这再次体现 AssetGen 的系统性：只优化 diffusion steps 不够，condition rendering、UV backprojection、normal baking、mesh simplification 都要纳入 latency budget。

## 23. Latency optimization：不是把表格相加

AssetGen 的 latency 部分非常值得细读。论文反复强调：表格里的 stage runtimes 不能简单相加，因为很多操作是并行或重叠执行的；报告的 total 是 measured critical-path latency。

两类优化最关键：

| 层次 | 方法 |
| --- | --- |
| Kernel / precision | FlashAttention-3、non-blocking transfer、torch compile、FP8/INT8 selective quantization |
| Pipeline scheduling | independent work overlap、worker process、MeshGen/TextureGen service split、precomputed geometry-dependent data |

论文还说 MeshGen 和 TextureGen 是 separate services。MeshGen 在一张 H100 上运行，并把 simplified mesh、UV、baked normal map 和 metadata 写到 shared storage；TextureGen 读取后生成 texture。默认 AssetGen 的 TextureGen 使用两张 H100 分 split CFG branches；Flash 由于 CFG distilled，只用一张。

这意味着论文报告的 30 秒/14 秒是一个系统部署结果，不是单模型 kernel benchmark。

## 24. Table 5：MeshGen + geometry critical path

![MeshGen latency breakdown](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table05-meshgen-latency.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 5, arXiv perpetual non-exclusive license.*

Table 5 把 MeshGen 和 geometry processing 拆成多个 stage。默认 AssetGen 包含 coarse 和 refine；Flash 跳过 refinement。表里的 total 是 MeshGen + geometry 的 measured critical path，不是所有行相加。

几个观察：

1. Coarse diffusion 和 mesh decoding 仍然是主要成本。
2. Refinement 带来额外质量，但也增加延迟。
3. Geometry processing 中 UV unwrapping、normal baking 等步骤已经被压到秒级或亚秒级。
4. Flash 的大幅提速来自跳过 refinement，同时 texture 端也降规格。

对工程读者来说，这张表比单个 FPS 数字更有价值，因为它告诉你瓶颈在哪里。

## 25. Table 6：TextureGen critical path

![TextureGen latency breakdown](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table06-texturegen-latency.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 6, arXiv perpetual non-exclusive license.*

Table 6 展示 TextureGen 的耗时。默认 AssetGen 使用 1024px diffusion、inference-time CFG、per-view super-resolution 和 2K atlas；Flash 使用 768px、guidance-distilled conditional-only model、省略 super-resolution 和 1K atlas。

需要注意，TextureGen 不是只跑一个 diffusion model。它还包括：

- pre-processing。
- condition rendering。
- diffusion。
- precompute backproject data。
- build KD tree。
- per-view super-resolution。
- backprojection and inpainting。
- export textured mesh。

这说明 TextureGen 是一个完整 texture synthesis and fusion pipeline。

## 26. Table 7：端到端延迟对比

![End-to-end latency](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table07-end-to-end-latency.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 7, arXiv perpetual non-exclusive license.*

Table 7 报告端到端 wall-clock latency：AssetGen 约 30 秒，AssetGen Flash 约 14 秒。商业模型 A-D 则在近一分钟到两分钟区间。

这张表必须谨慎解读。论文自己也说明：

- AssetGen 延迟包括 input image 到 exported textured mesh 的完整 pipeline，并包含 0.65s image segmentation。
- 商业 baselines 的延迟是 website submission 到 result availability，平均五次。
- 商业系统的硬件、队列、服务端负载、导出逻辑都不可控。

因此 Table 7 不是硬件公平 benchmark，更像用户可感知延迟对比。它支持的结论是：AssetGen 在论文部署和评估协议下，把 image-to-asset 时间拉到了更适合交互式创作的范围。

## 27. Table 8：优化 ladder

![Latency ablation](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table08-latency-ablation.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 8, arXiv perpetual non-exclusive license.*

Table 8 是系统工程价值最高的表之一。它用一个代表性 test asset 做 cumulative optimization ladder。

MeshGen + Geometry 从 baseline 103.15s 逐步降到 12.25s，关键步骤包括：

- CuMesh 替换 CPU-bound geometry processing。
- progressive distillation。
- FlashAttention-3 + compile。
- I/O optimization。
- parallel UV segmentation。
- DRTK normal baking。

TextureGen 从 117.68s 降到 17.85s，最大收益来自 progressive distillation，然后是 FA3/compile、precompute backproject data、rendering optimization、CFG branch split。

这张表说明 AssetGen 的速度不是某个 trick，而是系统性剥离瓶颈。对很多生成系统来说，真正上线慢的地方往往不是主模型 forward，而是数据准备、渲染、后处理、传输和导出。

## 28. AssetBench：普通物体生成评估

论文提出 AssetBench：101 个由 technical artists 手工挑选和审核的高质量 3D assets，覆盖 vehicles、daily-use objects、animals 和部分 characters。评估时渲染 isometric/frontal reference view 作为 image prompt。

![AssetBench quantitative results](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table09-assetbench.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 9, arXiv perpetual non-exclusive license.*

Table 9 指标包括：

| 类别 | 指标 | 含义 |
| --- | --- | --- |
| Shape Quality | IoU、Chamfer Distance | 与 ground-truth 3D 的几何接近程度 |
| Fidelity | VLM、CLIP | 与参考图/语义的一致性 |

论文指出，Commercial Model B 在 volumetric geometry metrics 上领先，AssetGen 在 VLM verification 和 GT-CLIP reference fidelity 上最好。这是一个很平衡的结论：AssetGen 不一定在所有几何指标上第一，但它的强项是 asset pipeline 和 reference fidelity。

这也提醒我们不要只看单一指标。对于 asset creation，几何 IoU、视觉一致性、纹理质量、可部署性和延迟是共同目标。

## 29. Fig. 6：商业 baseline 定性对比

![Commercial baseline comparison](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig06-commercial-comparison.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 6, arXiv perpetual non-exclusive license.*

Fig. 6 展示 AssetGen 与 Commercial Model A-D 的定性对比。论文匿名商业模型，这是合理但也限制了解读强度。我们不能把图写成“AssetGen 全面超过某具体产品”，只能说在论文选择的 matched input/rendering 条件下，AssetGen 在 texture richness、reference fidelity 和几何可用性之间表现有竞争力。

这张图支撑的是产品视角：用户关心最终资产是否像输入图、是否干净、是否能导出和使用，而不只是算法指标。

## 30. CharacterBench：角色比普通物体更难

角色生成比普通物体难，因为它同时考验：

- 正脸和侧脸质量。
- 身体比例、服饰、配件。
- 手指和局部细节。
- 身份和 reference fidelity。
- 是否适合后续 rigging 或动画。

![CharacterBench results](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table10-characterbench.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 10, arXiv perpetual non-exclusive license.*

Table 10 显示 AssetGen 在 frontal face quality、face CLIP、face ref、finger ref、overall fidelity 等指标上表现强。AssetGen Flash 在若干 reference fidelity 指标上接近默认版本，但 face quality 明显弱于默认版本。这符合 Flash 的定位：更快、更适合 preview 和快速迭代，但不是最佳质量配置。

需要注意，论文 limitation 明确说静态 textured mesh 只是交互角色的第一步。rigging、skin weight、blend shapes、motion synthesis 都不在 AssetGen 当前系统内。

## 31. Human evaluation：低延迟下达到可比质量

![Human evaluation](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-table11-human-eval.webp)

*Source: Wang et al., arXiv:2605.26137v1, Table 11, arXiv perpetual non-exclusive license.*

Table 11 是 195 assets 的 blind human evaluation，评分维度包括 general、geometry 和 texture，采用 1-5 绝对分。

论文的解释很克制：AssetGen 在三个类别都最高，但 margin 并非压倒性。正确读法是：在强商业 baseline 面前，AssetGen 的默认路径以明显更低的 end-to-end latency 达到 comparable-to-better inspected asset quality。Flash 接近 Commercial Model A，但整体略低于 full AssetGen。

这比“所有指标第一”更有工程意义。真实产品里，低延迟和可接受质量的 Pareto 前沿往往比单点最优质量更重要。

## 32. 与 Trellis 2 的对比

![Trellis 2 comparison](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig07-trellis2-comparison.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 7, arXiv perpetual non-exclusive license.*

Fig. 7 对比 AssetGen 和 Trellis 2。论文指出 Trellis 2 输出按推荐方式保留到约一百万 polygons；而 AssetGen 输出简化到约 23K polygons。Trellis 2 在高分辨率资产表示方面很强，但它的输出要进入 UV unwrap 和移动端实时渲染，还要面对高面数和后处理成本。

这组对比不能简化成“谁视觉更好”。两者优化目标不同：

| 方法 | 优先目标 |
| --- | --- |
| Trellis 2 | 高质量 3D 生成、structured latent、高分辨率资产 |
| AssetGen | image-to-deployable-asset，低面数、UV、normal baked、低延迟 |

如果你要做可交互资产管线，AssetGen 的 contract 更贴近最终使用；如果你要研究 3D latent 和高质量重建，Trellis 2 的侧重点不同。

## 33. AssetGen galleries

![AssetGen sand gallery](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig08-gallery-sand.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 8, arXiv perpetual non-exclusive license.*

Fig. 8 是默认 AssetGen 的 sand-themed gallery。它展示的是质量优先配置：two-stage MeshGen + high-fidelity TextureGen，约 30 秒输出。

![AssetGen ice gallery](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig09-gallery-ice.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 9, arXiv perpetual non-exclusive license.*

Fig. 9 展示 ice-themed assets。可以看到几何和材质风格的一致性，但也要记住这些是论文 gallery，不能替代大规模失败案例分析。

![AssetGen Flash gallery](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig10-flash-gallery.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 10, arXiv perpetual non-exclusive license.*

Fig. 10 是 AssetGen Flash 的 magic forest gallery。Flash 的价值是快速 preview 和 agent loop，而不是最终质量超过默认配置。

## 34. Real-world inputs

![Real-world input results](/images/blog/assetgen-deployable-3d-asset-generation/assetgen-fig11-real-world-inputs.webp)

*Source: Wang et al., arXiv:2605.26137v1, Fig. 11, arXiv perpetual non-exclusive license.*

Fig. 11 展示真实世界照片输入。论文指出模型主要在 synthetic 3D data 上训练，但对自然光照、背景杂乱、部分遮挡也有一定鲁棒性。例如某些被遮挡对象可以通过 shape prior 补全。

这里也正好对应 limitation：single-image 3D inherently underconstrained。输入只有一个视角时，背面、遮挡区域、不对称结构和复杂机械细节都需要模型猜。模型能猜得像，不等于它知道真实几何。

## 35. 与 Meta 3D AssetGen 2024 的关系

AssetGen 这个名字容易和 2024 年的 Meta 3D AssetGen 混淆。两者相关，但侧重点不同。

| 工作 | 重点 |
| --- | --- |
| Meta 3D AssetGen 2024 | text-to-mesh，geometry、texture、PBR materials，强调 albedo/metalness/roughness 和 relighting |
| AssetGen 2026 | image-to-deployable-asset，强调 30s/14s latency、mesh simplification、UV、normal baking、texture atlas、pipeline optimization |

2026 论文明确说 TextureGen 生成 color texture，而不是 PBR materials，因为这对 runtime rendering on mobile 更友好。这是一个取舍：PBR 分解对专业渲染和 relighting 很重要，但 color texture + normal map 更轻、更容易进入移动端实时管线。

## 36. 与本站 3D 文章的技术对照

本站已经有多篇 3D scene / asset / NeRF 相关精读，可以用下面的方式定位 AssetGen。

| 文章 | 主要问题 | 与 AssetGen 的差异 |
| --- | --- | --- |
| TRELLIS.2 | O-Voxel、SC-VAE、structured latent、大规模 3D asset generation | 更关注原生 3D latent 与生成模型；AssetGen 更关注 deployable asset pipeline 和低延迟 |
| Layout2Scene | 3D semantic layout-guided scene generation | 关注场景布局和几何/外观扩散先验；AssetGen 是单物体 image-to-asset |
| Blended-NeRF | 现有 NeRF 中 ROI 对象编辑 | 关注局部 NeRF 编辑和融合；AssetGen 输出显式 mesh asset |
| CompoNeRF | 多物体组合 NeRF 与布局编辑 | 关注 text-guided multi-object NeRF composition；AssetGen 关注 runtime asset contract |
| Imaginarium | 图像引导资产检索、姿态估计、Blender 物理优化 | 偏场景布局和资产库系统；AssetGen 是资产生成系统 |
| CasLayout | 3D layout diffusion 与关系建模 | 生成家具布局，不生成具体 deployable mesh asset |

AssetGen 的独特位置是：它不是场景生成，也不是 NeRF 编辑，而是把单个 3D asset 生成推进到工程可用性问题。

## 37. 与相关工作的关系

论文相关工作可以分成几条线：

1. **3D shape generation**：TripoSR、InstantMesh、CRM、Spar3D、Trellis、Hunyuan3D、Direct3D 等。
2. **3D latent / representation**：triplanes、wavelets、primitives、VecSet、structured coordinates、O-Voxel。
3. **Texture synthesis**：Text2Tex、Paint3D、SyncDreamer、MVDream、Wonder3D、Era3D、MVPaint。
4. **Mesh processing**：QEM simplification、xatlas UV unwrapping、normal baking、DRTK rasterization。
5. **Inference acceleration**：progressive distillation、CFG distillation、FlashAttention、torch compile、low precision、pipeline parallelism。

AssetGen 的贡献不是在某一条线上提出全新数学理论，而是把这些线合成了一个面向交付的系统。论文最像一篇“3D generative system paper”，而不是单纯的 model paper。

## 38. 工程复现清单

如果未来 AssetGen 代码开放，复现时至少要检查这些模块：

| 模块 | 需要核对的内容 |
| --- | --- |
| Data filtering | duplicate removal、orientation filtering、background/scan/low-quality filtering |
| Sign estimation | flood-fill、surface band、dilation、inside/outside label |
| VecSet VAE | latent token count、decoder SDF query、marching cubes |
| MeshGen DiT | DINOv2 conditioning、AdaLN、cross-attention、SwiGLU、4096 tokens |
| Refinement | 16384 tokens、spatial anchors、1022px conditioning |
| Distillation | teacher steps、student steps、CFG folding、DDIM target inversion |
| Simplification | CuMesh/QEM、target faces、normal preservation |
| UV | xatlas、parallel segmentation、seam handling |
| Normal baking | DRTK rasterization、tangent frame、UV seam dilation |
| TextureGen | 10 views、normal/position conditioning、structured attention、flow matching |
| Backprojection | visibility test、incidence weighting、anisotropic filtering |
| Inpainting | geometry-aware KD-tree propagation |
| Serving | H100 allocation、shared storage、two-service handoff、pipeline overlap |
| Export | final mesh, UVs, normal map, texture atlas, metadata |

没有这些模块，只复现 MeshGen 或 TextureGen 都不能叫完整 AssetGen。

## 39. 生产可用性批判

AssetGen 把可部署资产问题讲得很清楚，但仍有边界。

第一，**硬件依赖强**。30 秒和 14 秒是在 H100 部署上报告的。普通消费级 GPU、单卡环境或 CPU 后处理流程不应期待同等延迟。

第二，**官方实现未公开**。论文没有 AssetGen GitHub、模型权重或服务 demo。读者无法验证完整 pipeline，也无法确认许多工程优化的实现细节。

第三，**商业 baseline 匿名**。匿名商业模型 A-D 保护了公平呈现，但限制了可复查性。网页延迟也受排队、区域、负载、导出策略影响。

第四，**拓扑不等于动画可用**。论文自己承认 marching cubes 输出任意三角网格，简化后仍不保证 artist-friendly edge flow、rigging 或 animation-ready deformation。

第五，**单图歧义无法消除**。背面、遮挡、非对称结构和罕见类别只能靠 prior 猜，失败是基本限制。

第六，**PBR 材质不是主目标**。论文输出 color texture 和 normal map，对移动端友好，但如果生产需要 albedo/roughness/metalness 和 relighting，还需要额外模块。

第七，**资产 QA 仍不可省**。真实游戏和 AR/VR 资产还要检查碰撞、LOD、材质规范、贴图压缩、法线方向、UV seam、引擎导入、物理尺寸和版权风险。

## 40. 推荐阅读路径

如果只读论文，不建议按页顺序从头到尾啃。更高效的顺序是：

1. Abstract、Fig. 1、Fig. 2，先理解 asset contract。
2. Sec. 5 MeshGen，看 VecSet diffusion、coarse-to-refine 和 distillation。
3. Sec. 6 Geometry Post-Processing，看 simplification、UV、normal baking。
4. Sec. 7 TextureGen，看 multi-view conditioning、structured attention、backprojection。
5. Sec. 8 Latency Optimization，看 Table 5-8。
6. Sec. 9 Evaluation，看 AssetBench、CharacterBench、人评。
7. Sec. 11 Limitation，理解 single-image ambiguity、topology、rigging 边界。
8. 之后补 Meta 3D AssetGen 2024、TRELLIS.2、Hunyuan3D、MVPaint、CuMesh、xatlas、DRTK。

## 41. 结论

AssetGen 的长期价值，不在于它又把某个 image-to-3D 指标提高了一点，而在于它把研究问题改写成了一个更接近生产的交付标准：

> 给定一张参考图，在几十秒内交付一个可导出、可实时渲染、有可控面数、有 UV、有 baked normals、有 texture atlas 的 3D asset。

为了做到这一点，论文没有只优化生成模型，而是把 MeshGen、geometry processing、TextureGen、distillation、GPU kernels、UV、normal baking、backprojection、inpainting 和 serving pipeline 一起设计。

这也是它对 3D 生成领域最有价值的提醒：未来的 3D asset generation 不会只由更大的 diffusion model 决定，还会由整个资产化管线决定。模型负责想象和重建，图形学流程负责把它变成可用资产，系统工程负责让它足够快，产品 QA 负责让它真的能进入创作工作流。

## References

- [AssetGen arXiv abs](https://arxiv.org/abs/2605.26137)
- [AssetGen arXiv PDF](https://arxiv.org/pdf/2605.26137)
- [AssetGen arXiv HTML](https://arxiv.org/html/2605.26137v1)
- [Meta 3D AssetGen 2024](https://arxiv.org/abs/2407.02445)
- [TRELLIS.2 project page](https://microsoft.github.io/TRELLIS.2/)
- [CuMesh](https://github.com/visualbruno/CuMesh)
- [xatlas](https://github.com/jpcy/xatlas)
- [DRTK](https://github.com/facebookresearch/DRTK)
- [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf)

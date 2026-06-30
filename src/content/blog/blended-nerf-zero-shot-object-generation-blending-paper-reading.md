---
title: "Blended-NeRF 论文精读：现有 NeRF 场景中的零样本对象生成、ROI 编辑与体渲染融合"
description: "精读 Blended-NeRF 如何通过 3D ROI、CLIP 引导、双 NeRF 场融合、距离平滑和官方代码实现现有神经辐射场中的对象插入、替换与材质编辑"
pubDate: "2026-06-30T11:58:46+08:00"
updatedDate: "2026-06-30T11:58:46+08:00"
tags:
  - "Paper Reading"
  - "NeRF"
  - "3D Scene Editing"
  - "Text-to-3D"
  - "Computer Vision"
  - "Graphics"
  - "Code Reading"
draft: false
---

Ori Gordon、Omri Avrahami 和 Dani Lischinski 的 **Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields** 研究的是一个非常具体但很有代表性的 3D 编辑问题：

> 已经有一个训练好的 NeRF 场景时，能不能只在用户指定的 3D 区域里生成、替换、融合或改变一个对象，同时尽量保持区域外的原场景不动？

这和“从文本生成一个全新 3D 对象”不一样。Blended-NeRF 面对的是 **existing Neural Radiance Field**，也就是一个已经训练好的隐式场景表示。用户不是从零描述整个世界，而是给出一个文本 prompt 或图像 patch，再指定一个 3D ROI box。模型要在这个 box 内做编辑，并把新内容自然地融回原来的神经辐射场。

这篇论文的价值不在于提出了一个全新的基础生成模型，而在于把几个工程上很关键的问题组合成一个相对完整的 NeRF 编辑流程：

- 用原始 NeRF 权重初始化一个 generator NeRF，而不是从随机场开始优化。
- 用 3D ROI box 限定编辑区域，让优化目标只作用于局部。
- 用 CLIP 或 BLIP 这类视觉-语言模型提供零样本语义引导。
- 在 ray sample 级别同时查询原始 NeRF 和生成 NeRF，并用密度、颜色、距离平滑做 volumetric blending。
- 用背景增强、姿态采样、transmittance loss、depth loss 等先验缓解 CLIP-guided NeRF 常见的漂浮、扁平、视角不一致问题。

本文精读的是 arXiv 当前版本 [arXiv:2306.12760v2](https://arxiv.org/abs/2306.12760v2)。arXiv 页面显示原始提交时间为 2023-06-22，v2 修订于 2023-09-07；arXiv DOI 为 `10.48550/arXiv.2306.12760`，Related DOI 为 `10.1109/ICCVW60793.2023.00316`。论文 PDF 共 16 页、14 张图，license 为 **CC BY-NC-ND 4.0**。文中插入的图表来自论文 PDF，只做等比例裁切，不重绘、不改图内内容。

官方项目页是 [vision.huji.ac.il/blended-nerf](https://www.vision.huji.ac.il/blended-nerf/)，官方代码是 [orig333/Blended-NeRF](https://github.com/orig333/Blended-NeRF)。仓库 README 标注了 ICCV 2023 AI3DCC，并采用 MIT License。本文会做论文级方法精读和源码静态阅读，但不默认下载数据、训练 base NeRF 或复现实验。

## 1. 论文信息与一句话贡献

| 项目 | 内容 |
| --- | --- |
| 题名 | Blended-NeRF: Zero-Shot Object Generation and Blending in Existing Neural Radiance Fields |
| 作者 | Ori Gordon, Omri Avrahami, Dani Lischinski |
| 机构 | The Hebrew University |
| arXiv | 2306.12760v2 |
| 初始提交 | 2023-06-22 |
| v2 修订 | 2023-09-07 |
| arXiv DOI | 10.48550/arXiv.2306.12760 |
| Related DOI | 10.1109/ICCVW60793.2023.00316 |
| 论文 license | CC BY-NC-ND 4.0 |
| 官方代码 | https://github.com/orig333/Blended-NeRF |
| 代码 license | MIT |
| 主要任务 | text/image-guided ROI editing in existing NeRF scenes |

一句话概括：

> Blended-NeRF 将局部 3D 编辑写成“复制一个原始 NeRF、只在 ROI 内用视觉-语言先验优化生成分支、再沿每条光线把原始分支与生成分支做体渲染级融合”的流程。

这里有三个关键词。

第一是 **existing NeRF**。论文不是训练一个通用 text-to-3D 模型，也不是生成完整场景，而是假设输入已经是一个 NeRF。这个设定比从零生成更贴近编辑：用户已有一个场景，希望局部加一朵花、替换一个松果、把花瓶变成玻璃材质。

第二是 **3D ROI**。局部编辑必须有空间边界。Blended-NeRF 使用 3D box 作为用户控制接口，代码中具体体现为 `box_points/*.pt`。这比“在 2D 图像上画 mask 再投回 3D”更直接，但也意味着用户需要能定位一个合理的 3D box。

第三是 **volumetric blending**。如果只是把 ROI 内的 NeRF 输出直接替换掉，边界会很硬；如果只是图像后处理，无法保证多视角一致。论文把 blending 放在体渲染采样点层面，让原始 density/color 和生成 density/color 共同参与 alpha compositing。

## 2. 从 Fig. 1 看整体流程

Fig. 1 是全文最重要的概览图。左侧是训练阶段，右侧是 blending 阶段。

![Fig. 1: Blended-NeRF overview](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig01-overview.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 1, CC BY-NC-ND 4.0.*

图里有两个 NeRF：

- $F^O_\theta$：original NeRF，表示输入场景，训练时基本作为固定参考。
- $F^G_\theta$：generator NeRF，初始化自 $F^O_\theta$，然后在 ROI 内被 CLIP/BLIP 引导优化。

训练阶段不是渲染整张图去做 CLIP loss，而是只渲染 ROI 内的内容。具体做法是：从相机发出 rays，沿 ray 采样点，只保留落在 3D box 内的采样点参与 ROI rendering；box 外的样本设置成零，避免 loss 推动全场景漂移。

训练完成后，推理阶段不是简单输出 $F^G_\theta$。Blended-NeRF 沿同一批 rays 同时查询 $F^O_\theta$ 和 $F^G_\theta$：

- box 外部使用原始 NeRF。
- box 内部按任务选择 replacement 或 blending。
- box 边界附近使用距离平滑，让过渡更自然。

这就是论文标题里 “generation and blending” 的含义：生成分支负责让 ROI 内出现目标对象；融合模块负责让它成为原场景的一部分。

## 3. 为什么现有 NeRF 场景编辑困难

NeRF 将场景表示为一个连续函数：

$$
F_\theta(\mathbf{x}, \mathbf{d}) = (\sigma, \mathbf{c}),
$$

其中 $\mathbf{x}$ 是 3D 位置，$\mathbf{d}$ 是视线方向，$\sigma$ 是 volume density，$\mathbf{c}$ 是 RGB color。沿一条 ray 采样多个点后，体渲染公式把每个点的颜色按透明度加权累积：

$$
\mathbf{C}(\mathbf{r}) =
\sum_{i=1}^{N}
T_i \left(1 - \exp(-\sigma_i \delta_i)\right)\mathbf{c}_i,
$$

$$
T_i = \exp\left(-\sum_{j<i}\sigma_j\delta_j\right).
$$

这个表示很适合 novel view synthesis，但对编辑不友好，原因至少有四个。

第一，NeRF 没有显式对象边界。一个松果、地面、阴影、反射和背景都被同一个 MLP 编码。想“替换松果”时，模型里没有天然的 `object_id=pinecone`。

第二，局部几何和全局视角强耦合。一个 3D 点的 density 改动会影响多个视角；如果优化只看单张渲染图，很容易在一个视角看起来对、换视角就崩。

第三，文本监督本身弱。CLIP 能判断图像和文本是否语义接近，但不能精确告诉 NeRF 哪个点应该变成目标对象的表面、哪个点应该透明、哪个点应该保留原色。

第四，编辑要保真。text-to-3D 从零生成时，背景可以不重要；existing NeRF editing 则要求 ROI 外的建筑、树叶、桌面、光照尽量不变。

Blended-NeRF 的设计正是围绕这些问题展开：ROI box 提供对象边界近似；初始化自原始 NeRF 提供 scene prior；局部 rendering 限制优化范围；volumetric blending 处理边界和多视角一致性。

## 4. 大对象替换：Fig. 2 的直观例子

Fig. 2 展示了 Blender ship 场景中的大对象替换。论文将 ROI box 放在海面和船底区域，用文本 prompt 引导模型把沙地/海面替换成新的视觉内容。

![Fig. 2: Large object replacement](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig02-large-object-replacement.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 2, CC BY-NC-ND 4.0.*

这个例子说明 Blended-NeRF 并不只处理“往空白区域放一个新小物体”。它也可以把 ROI 内原来已有的内容改成另一个概念，例如把船周围区域改成冰雪风格。关键不在于 prompt 本身多复杂，而在于 ROI 选得足够覆盖待替换区域。

如果 ROI 太小，生成对象会被截断；如果 ROI 太大，模型会被允许改动过多原场景，保真性下降。官方代码把 ROI 作为 `box_points_path` 读入，训练配置中显式指定，例如 `box_points/lego_exhaust_box.pt`。这意味着这套方法的交互质量，很大程度取决于用户能否给出合适的 box。

## 5. Positional Encoding 与原始场景初始化

论文沿用 NeRF 的 positional encoding，把低维坐标映射到高频特征：

$$
\gamma(x) =
\left[
\cos(2^l x), \sin(2^l x)
\right]_{l=0}^{L-1}.
$$

这部分本身不是 Blended-NeRF 的创新，但它影响了编辑流程。官方代码中 `BlendedNeRF/run_BlendedNeRF.py` 的 `create_nerf(args)` 会创建 trainable model 和 fixed model。二者结构相同，区别在于：

- fixed model 对应原始场景，用于提供不可编辑区域的 density/color。
- trainable model 从已有 checkpoint 初始化，用于生成或改变 ROI 内内容。

从原始 NeRF 初始化 generator 分支很重要。如果从随机 MLP 开始，CLIP loss 既要学几何、又要学材质、还要学场景坐标，非常不稳定。初始化自原场景等于把相机坐标、背景尺度、已有几何分布、颜色统计都作为先验带入。这样训练可以更专注于“局部差异”而不是“重建整个世界”。

代码静态阅读中可以看到三个层面的对应关系：

| 论文概念 | 代码位置 | 实现含义 |
| --- | --- | --- |
| 原始 NeRF | `fixed_model`, `fixed_model_fine` | 作为 frozen 或 reference 分支提供原场景输出 |
| 生成 NeRF | `model`, `model_fine` | 用 CLIP/BLIP 和几何 loss 更新的分支 |
| ROI box | `box_points_path`, `render_kwargs_train['box_points']` | 控制哪些采样点进入编辑区域 |
| blending | `raw2outputsBlend`, `render_full_frame_fn` | 对两套 raw density/color 做体渲染级组合 |
| ROI rendering | `render_in_box_fn` | 只渲染 box 内样本，用于训练语义 loss |

## 6. CLIP Similarity Loss：语义监督从哪里来

Blended-NeRF 使用语言-图像模型给 ROI 内生成内容提供语义方向。论文把相似度损失写成：

$$
\mathcal{L}_{sim} =
-E_{img}(I_{ROI})^T E_{txt}(y),
$$

其中 $I_{ROI}$ 是 ROI 内渲染结果，$y$ 是用户给定 prompt，$E_{img}$ 和 $E_{txt}$ 分别是图像和文本编码器。

这个 loss 的含义很直接：让渲染出来的 ROI 图像在 CLIP embedding 空间中接近文本。它的好处是零样本，不需要特定对象的 3D 训练数据；坏处是监督很粗。CLIP 不知道 NeRF 的 density 是否合理，不知道物体是不是薄片，也不关心同一个对象在另一个视角是否仍然合理。

官方实现中 `optimization/losses.py` 有 `CLIPLoss` 和 `BLIPLoss`。默认 CLIP 加载 `ViT-B/32`，训练 loss 封装在 `ObjectnessLoss`，返回：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{clip/blip}
+ \lambda_T \mathcal{L}_T
+ \lambda_D \mathcal{L}_D.
$$

这说明代码实现并不是只靠 CLIP。CLIP/BLIP 给语义方向，transmittance/depth 负责限制几何退化，背景增强和姿态采样负责降低过拟合到单一视角的风险。

## 7. ROI-only Rendering：为什么训练时只看 box 内

论文将 ROI rendering 写成：

$$
C(r)=
\begin{cases}
\sum_{\mathbf{x}_i \in B} T_i(1-\exp(-\sigma_i \delta_i))\mathbf{c}_i,
& \exists \mathbf{x}_i \in r \text{ such that } \mathbf{x}_i \in B,\\
0, & \text{otherwise}.
\end{cases}
$$

直观地说，训练时不是把整张图喂给 CLIP，而是只把 box 内内容渲染出来，再和 prompt 对齐。这样做有两个作用。

第一，它减少了 CLIP loss 对背景的干扰。假如 prompt 是 “a strawberry”，整张图里有花盆、树叶、地面、墙壁，CLIP 可能根据背景也给出较高相似度，导致优化目标不聚焦。只渲染 ROI 后，模型必须让 box 内内容承担语义。

第二，它保护原场景。ROI 外没有参与生成分支训练，最终也主要由 fixed model 渲染。因此只要 blending 边界处理得当，背景保真会比直接优化整个 NeRF 好。

在代码中，`render_rays()` 会根据 `box_points` 判断采样点是否在 box 内：

```python
pt_is_in_box = (
    (xmin <= pts[..., 0]) & (pts[..., 0] <= xmax) &
    (ymin <= pts[..., 1]) & (pts[..., 1] <= ymax) &
    (zmin <= pts[..., 2]) & (pts[..., 2] <= zmax)
)
```

随后 `render_in_box_fn()` 只对 box 内采样点查询 trainable model 或 fixed model，并输出 `rgb_map_in_box`、`transmittance_in_box`、`depth_map_in_box` 等训练 loss 需要的中间量。

## 8. 3D ROI Box：控制接口与复现门槛

ROI box 是 Blended-NeRF 的交互核心。它既不是自动检测出来的对象 mask，也不是数据集提供的标注，而是用户给出的 3D bounding box。论文里强调可以通过 GUI 在已有 NeRF 中定位 box；官方仓库提供三个 notebook：

- `notebooks/find_3d_box_blender.ipynb`
- `notebooks/find_3d_box_llff.ipynb`
- `notebooks/find_3d_box_llff_360.ipynb`

README 描述的流程是：给定 config，从不同角度和距离观察原始 NeRF 场景，定位 3D box，然后把 box points 保存为 `.pt` 文件用于训练。

这是一种很务实的设计。它没有试图解决“自动从文本定位 3D 对象”这个更难的问题，而是把空间控制交给用户。这样系统能获得强控制能力，但代价也清楚：

- ROI box 太松会允许模型修改过多背景。
- ROI box 太紧会切掉目标对象或导致边界不自然。
- box 是长方体，不适合轮子、树干、瓶口这类细长或曲面区域。
- 对普通用户来说，在 NeRF 场景里准确定位 3D box 本身并不轻量。

这也是论文局限性里提到的一个关键点：有些场景用圆柱体、球体或更复杂 shape 作为 ROI 会更合适，但本文只实现了 box。

## 9. Distance Smoothing Operator：边界为什么不硬切

如果在 box 内完全用 $F^G_\theta$，box 外完全用 $F^O_\theta$，边界处容易出现明显接缝。Fig. 3 展示了论文提出的 distance smoothing operator。

![Fig. 3: Distance Smoothing Operator](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig03-distance-smoothing.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 3, CC BY-NC-ND 4.0.*

论文定义：

$$
f(\mathbf{x}) = 1 - \exp\left(-\frac{\alpha d(\mathbf{x})}{diag}\right),
$$

其中 $d(\mathbf{x})$ 是点 $\mathbf{x}$ 到 ROI center 的欧氏距离，$diag$ 是 box 对角线长度，$\alpha$ 控制平滑强度。随后对 density 和 color 做加权：

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

注意这里的 $f(\mathbf{x})$ 不是由神经网络学出来的，而是一个几何启发式。它表达的是：越靠近 ROI 中心，越相信生成分支；越靠近或远离编辑核心，越把权重交还给原始分支。图里 $\alpha$ 越大，原始场景的影响越强，松果颜色越接近原图。

工程上，这个设计很关键。Blended-NeRF 的生成分支不一定能在 box 边界精确生成和背景一致的纹理、阴影和 density。如果硬切，接缝会暴露；如果用几何平滑，至少能让过渡区域更连续。

## 10. 插入、替换与融合：三个编辑模式不是一回事

Blended-NeRF 的应用可以分成三类。

第一类是 **new object insertion**。ROI 原本可能是空的，目标是在里面放入新对象。此时原始 density 在 ROI 内通常不重要，生成分支主要负责产生新对象。

第二类是 **object replacement**。ROI 里已有对象，目标是把它替换成新对象。例如 fern 场景中把树干替换成草莓，pinecone 场景中把松果替换成菠萝。这要求生成分支不仅添加新 density，还能抑制或覆盖原有 density。

第三类是 **object blending**。目标不是完全替换原对象，而是让新对象和原对象同时存在、互相缠绕或贴合。例如在 Lego 轮子附近生成植物，让植物围绕轮子生长。这要求两套 density/color 都参与，而不是二选一。

论文把这三类统一到 volumetric blending 框架下，但具体 density 组合方式不同。

## 11. Object Blending：Fig. 4 与 Eq. 6-9

Fig. 4 展示了 object blending 的两种 density 设计。prompt 是让绿色、白色和蓝色花朵与 Lego 轮子融合。

![Fig. 4: Blending Modes](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig04-blending-modes.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 4, CC BY-NC-ND 4.0.*

先看每个采样点的 alpha：

$$
\alpha_O(\mathbf{x}_i)
=
1 - \exp(\phi(\sigma_O(\mathbf{x}_i))\delta_i),
$$

$$
\alpha_G(\mathbf{x}_i)
=
1 - \exp(\phi(\sigma_G(\mathbf{x}_i))\delta_i).
$$

论文用 $\phi$ 表示 density activation，使 density 非负。颜色融合写成：

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
\right),
$$

其中 $S$ 是 sigmoid，$\epsilon$ 是数值稳定项。也就是说，颜色不是简单平均，而是按各自的 alpha 贡献加权。

更关键的是 density 有两种写法：

$$
\sigma(\mathbf{x}_i)
=
\phi(\sigma_O(\mathbf{x}_i)+\sigma_G(\mathbf{x}_i)),
$$

或者：

$$
\sigma(\mathbf{x}_i)
=
\phi(\sigma_O(\mathbf{x}_i))
+ \phi(\sigma_G(\mathbf{x}_i)).
$$

两者差别在于 activation 的位置。

第一种是 **sum inside activation**。先把原始 density logit 和生成 density logit 相加，再过 activation。这允许生成分支用负值抵消原始 density，因此可以“改掉”原物体的一部分。Fig. 4 中轮子被更多改变，植物更像是在替换和融合原几何。

第二种是 **sum outside activation**。两边各自过 activation 后再相加。因为两个非负 density 相加，生成分支只能增加新密度，不能减少原始密度。因此植物会包裹轮子，但不会真正移除轮子。

官方代码中 `raw2outputsBlend()` 的 `sum_in_act` 参数正对应这个设计。文章里如果只说“把两个 NeRF 相加”会丢掉重点：**相加发生在 activation 前还是后，决定了编辑是 additive 还是 replacement-capable。**

## 12. 主实验：与 Volumetric Disentanglement 的对象替换对比

论文使用 Volumetric Disentanglement 作为对象替换 baseline。Fig. 5 展示 fern 场景中把树干替换成 “aspen tree” 和 “strawberry” 的结果。

![Fig. 5: Comparison to Volumetric Disentanglement for object replacement](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig05-volumetric-disentanglement-comparison.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 5, CC BY-NC-ND 4.0.*

这个对比要注意读法。Volumetric Disentanglement 更偏向从前景和背景分解角度处理场景，而 Blended-NeRF 明确引入 text prompt、ROI box 和生成分支优化。因此 Fig. 5 的优势主要说明：在给定 ROI 和 prompt 的场景编辑任务中，Blended-NeRF 能生成更贴近 prompt 的新对象，并在多视角上更自然。

论文的定量表如下。

![Table 1: Quantitative Evaluation](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-table01-quantitative-evaluation.png)

*Source: Gordon et al., arXiv:2306.12760v2, Table 1, CC BY-NC-ND 4.0.*

Table 1 使用了三类指标：

- CLIP Direction Similarity：衡量文本方向变化和图像方向变化是否一致。
- CLIP Direction Consistency：衡量不同视角或相邻帧的编辑方向是否一致。
- LPIPS：衡量 ROI 外或背景保真相关差异，数值越低越好。

表中 Blended-NeRF 在三项指标上都优于 baseline。但这不能被误读成“Blended-NeRF 生成的 3D 几何一定正确”。CLIP 相关指标主要衡量语义对齐和方向一致性；LPIPS 也不是几何评价。真正的工程判断仍需要看多视角渲染、边界接缝、深度形态和目标对象是否物理合理。

## 13. Geometry Priors：为什么只靠 CLIP 会坏

CLIP loss 有一个常见问题：只要某些视角图像能骗过 CLIP，NeRF 就可能学出人眼不希望的结构。例如：

- 只在一个视角上像目标对象，侧面是薄片。
- 生成密度散在空间中，形成漂浮噪点。
- 生成对象过于透明或体积不足。
- 背景色被模型利用，CLIP 分数高但对象不清楚。

Blended-NeRF 继承了 Dream Fields 和 CLIP-NeRF 中的一些优化技巧，并加入新的先验。核心包括：

| 先验 / 增强 | 作用 | 风险边界 |
| --- | --- | --- |
| pose sampling | 从不同相机角度渲染，防止单视角过拟合 | 视角采样仍有限，不能保证任意视角正确 |
| background augmentation | 用随机背景替换空白区域，防止模型靠背景颜色骗分 | 可能改变视觉统计，需要谨慎调权重 |
| directional dependent prompts | 根据视角附加方向描述，如 front/top/side view | 方向词本身依赖 prompt 工程 |
| transmittance loss | 鼓励 ROI 内出现足够非透明内容 | 过强可能产生不必要 density |
| depth loss | 鼓励有体积的 3D 形状，减少扁平结构 | 过强可能让对象膨胀或遮挡背景 |
| annealing | 训练早期/后期逐步调节正则强度 | 依赖任务和场景调参 |

这些先验不是论文的边角料，而是 CLIP-guided NeRF 能否稳定工作的关键部分。

## 14. Transmittance Loss 与 Depth Loss

论文先定义相机距离采样：

$$
d = \frac{e_{max}}{2\tan(\gamma/2)},
$$

其中 $e_{max}$ 是 box 最大边长，$\gamma$ 是相机 AFOV 的一半。这个公式用于围绕 ROI 采样相机位置，让训练视角和 box 尺度匹配。

Transmittance loss 写成：

$$
\mathcal{L}_T
=
-\min(\tau, mean(T(P))).
$$

这里 $T(P)$ 是从 pose $P$ 渲染得到的平均 transmittance。直觉上，若 ROI 过于透明，模型没有真正生成对象；loss 会鼓励生成更多 density。论文使用的默认 $\tau$ 是 `0.88`。

Depth loss 写成基于 disparity map variance 的约束：

$$
\mathcal{L}_D
=
-\min(\rho, \sigma^2(D(P))).
$$

如果生成对象像一张平面贴纸，depth/disparity 的变化会很弱。depth loss 鼓励对象有更强的体积变化。论文实现里还会用 annealing 防止一开始就强推几何正则导致训练崩。

总目标是：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{sim}
+ \lambda_T\mathcal{L}_T
+ \lambda_D\mathcal{L}_D.
$$

官方代码 `optimization/losses.py` 中，`ObjectnessLoss` 把 CLIP/BLIP loss、transmittance loss 和 depth loss 汇总。代码默认参数能看到 `max_trans`、`trans_loss_lambda`、`max_depth_var`、`depth_loss_lambda` 等配置项；`run_BlendedNeRF.py` 中训练循环会从 `extras` 里取 `transmittance_in_box` 和 `depth_map_in_box` 来计算这些项。

## 15. Depth Loss 消融：Fig. 6 和 Table 2

Fig. 6 对比了有无 depth loss 时的结果。

![Fig. 6: Depth Loss Impact](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig06-depth-loss-impact.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 6, CC BY-NC-ND 4.0.*

没有 depth loss 时，生成的 donut 视觉上能骗过部分文本语义，但 disparity map 更像扁平结构。加入 depth loss 后，目标对象的深度变化更明显，形状也更像真正占据 3D 空间的物体。

Table 2 从指标上补充了 priors 和 augmentations 的影响。

![Table 2: Ablation study](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-table02-ablation-study.png)

*Source: Gordon et al., arXiv:2306.12760v2, Table 2, CC BY-NC-ND 4.0.*

这张表用 COCO GT 作为上界参照，比较 full pipeline、去掉 directional prompts、去掉 depth prior 的结果。可读出的结论是：

- full pipeline 的 CLIP/BLIP R-Precision 高于两个 ablation。
- directional prompts 的影响相对较小，但仍有帮助。
- depth prior 对几何和语义指标都有贡献。

不过 R-Precision 是文本-图像匹配指标，不是 3D mesh quality 指标。它适合比较“渲染图是否像 prompt”，不能评价 NeRF 是否有可用几何、是否可导出资产、是否符合物理约束。

## 16. Texture Editing：只改颜色，不改密度

Fig. 7 展示了材质/纹理编辑。比如把 pinecone 改成燃烧、冰冻、粉色羊毛；把 vase 改成玻璃、石头、水彩风格。

![Fig. 7: Texture Editing](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig07-texture-editing.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 7, CC BY-NC-ND 4.0.*

纹理编辑和对象替换不同。对象替换需要修改 density，因为几何形状也要改变；纹理编辑更希望保留原始几何，只改变颜色相关层。论文说明其做法是冻结 density 相关层，只训练影响 color 的层。

这类任务更能体现 existing NeRF editing 的优势。如果用户已经有一个 pinecone 或 vase 的多视角 NeRF，那么只改外观比从零生成对象更稳定，也更容易保持多视角一致性。缺点是它不能创造新的几何拓扑；如果 prompt 要求“松果变成一只鸟”，仅改颜色层显然不够。

## 17. Blending Densities Inside Activation：Fig. 8

Fig. 8 是 Fig. 4 的补充，展示在 LLFF fern 场景中使用 Eq. 8 进行 blending 的效果。

![Fig. 8: Blending densities inside activation](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig08-blending-densities-inside-activation.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 8, CC BY-NC-ND 4.0.*

这张图的重点不是“花或蘑菇好不好看”，而是 density 组合方式对原始场景的影响。使用 sum inside activation 后，生成分支有能力改变原 density，因此新对象不是简单贴在原对象表面，而可以与原几何发生更强相互作用。

工程落地时，这个选项应该和编辑意图绑定：

- 如果用户要“在桌上加一个苹果”，更适合 additive 模式，避免破坏桌面。
- 如果用户要“把松果换成菠萝”，需要 replacement-capable 模式，让生成分支能压掉原松果 density。
- 如果用户要“让藤蔓缠绕椅子”，可能需要 blending 模式，保留原椅子并添加新 density。

这也是为什么让召回模型、文本模型或 UI 自动选择编辑模式并不简单。用户意图里的“add”、“replace”、“blend”、“restyle”需要映射到不同的 density/color 组合策略。

## 18. 补充结果：对象替换与对象插入

Fig. 9 给出了 fern 场景对象替换的更多视角。

![Fig. 9: Additional views for object replacement comparison](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig09-additional-object-replacement-views.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 9, CC BY-NC-ND 4.0.*

这张图用于检查两个问题。第一是不同视角下新对象是否仍然稳定。第二是原背景是否被保留。Blended-NeRF 的结果不是完美 3D asset，但相比只做 2D 层面的编辑，多视角一致性更强。

Fig. 10 展示 object insertion。每一列是同一个编辑场景的两个视角。

![Fig. 10: Object Insertion](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig10-object-insertion.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 10, CC BY-NC-ND 4.0.*

这里可以看到 insertion 任务的典型边界：新对象大体能出现在空白区域，但它和原场景的接触、阴影、遮挡和尺度仍依赖 ROI、loss 权重和 NeRF 优化。论文展示的是可视化质量和多视角一致性，而不是物理仿真或可编辑 mesh。

## 19. Real 360 场景：vasedeck 和 pinecone

Fig. 11 展示在 vasedeck 360 scene 中插入花瓣对象。

![Fig. 11: Object insertion in vasedeck 360 scene](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig11-vasedeck-object-insertion.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 11, CC BY-NC-ND 4.0.*

Real 360 scene 比 synthetic scene 更难，因为真实采集数据有遮挡、光照、背景复杂度和相机轨迹限制。论文中用 360 场景说明 Blended-NeRF 不只在 Blender 合成数据上工作。

Fig. 12 展示 pinecone 360 scene 中把松果替换成菠萝。

![Fig. 12: Object replacement in 360 pinecone scene](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig12-pinecone-object-replacement.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 12, CC BY-NC-ND 4.0.*

这类替换任务的难点在于，新对象要占据原对象位置，同时不要破坏周围石板、苔藓和背景。Blended-NeRF 的初始化和 fixed branch 保留了原场景统计，ROI 和 blending 决定了新旧场景的过渡。

## 20. Real 360 场景的材质转换

Fig. 13 展示 pinecone 的材质转换。

![Fig. 13: Texture conversion on 360 pinecone scene](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig13-pinecone-texture-conversion.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 13, CC BY-NC-ND 4.0.*

材质转换看起来比对象替换更稳定，因为几何已经存在，优化目标主要改变颜色或外观。但即便如此，CLIP-guided texture editing 也可能过度改变局部光照、把风格扩散到边界，或者在某些视角出现纹理不一致。

Fig. 14 展示 vasedeck 场景中的花瓶材质转换。

![Fig. 14: Texture conversion on 360 vasedeck scene](/images/blog/blended-nerf-zero-shot-object-generation-blending/blended-nerf-fig14-vasedeck-texture-conversion.png)

*Source: Gordon et al., arXiv:2306.12760v2, Fig. 14, CC BY-NC-ND 4.0.*

这个例子说明 Blended-NeRF 可以把同一个真实场景对象映射到多种材质描述。但也要注意，它并不是 CAD/mesh 材质编辑工具。输出仍是 NeRF 渲染结果，不能直接等价于一个可导出的、拓扑干净、PBR 参数可控的 3D asset。

## 21. 论文-代码对照：从 README 到训练循环

官方仓库的 README 给出的最小流程是：

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

这个流程对应论文中的三个外部输入：

- existing scene data：原始 NeRF 场景数据。
- base scene weights：已有 NeRF 权重，放在 `base_weights/`。
- ROI box points：3D box 坐标，放在 `box_points/`。

`main.py` 非常薄，只调用 `train_blended_nerf()`。真正逻辑在 `BlendedNeRF/run_BlendedNeRF.py`。静态阅读可以把它分成六段：

| 代码段 | 论文概念 | 说明 |
| --- | --- | --- |
| `config_parser()` | 实验配置 | prompt、box、blend、depth loss、trans loss、dataset、render 参数 |
| `create_nerf(args)` | fixed/generator NeRF | 构建两套 NeRF，并加载 base checkpoint |
| `render_rays()` | volume rendering | 沿 ray 采样、判断点是否在 ROI box 内 |
| `render_in_box_fn()` | ROI-only rendering | 输出 ROI 内图像和中间量供 CLIP/regularization |
| `raw2outputsBlend()` | volumetric blending | 对 fixed/raw 和 train/raw 做 alpha/color/density 组合 |
| training loop | optimization | 渲染 ROI、做背景增强、计算 loss、保存渲染和 checkpoint |

`optimization/losses.py` 中的 `ObjectnessLoss` 对应论文目标函数。它把 CLIP/BLIP similarity、transmittance loss、depth loss 汇总。`optimization/augmentations.py` 中的 `sample_background()` 对应背景增强，用 transmittance mask 将 ROI 渲染与随机背景结合。

配置文件如 `configs/lego.txt` 会指定：

```text
expname = lego_test
basedir = ./logs/blender_lego/
box_points_path = box_points/lego_exhaust_box.pt
```

这说明官方实现不是一个“输入 prompt 自动编辑任意场景”的端到端系统。它需要用户准备 scene、base weights、ROI box 和 config；训练过程再从这些显式资产出发。

## 22. 代码中的 ROI 与 blending 细节

在 `render_rays()` 中，代码会计算 box 的宽、高、深和对角线长度，并判断采样点是否落在 box 内。判断逻辑是 axis-aligned range check：

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

这和论文中的 3D ROI box 一致，但也说明了一个限制：默认实现是 axis-aligned box 范式。如果编辑对象本身是斜的、细长的、环状的或局部曲面，box 只能近似。

`raw2outputsBlend()` 则是代码中最贴近论文 Eq. 6-9 的部分。它先把 fixed model 和 trainable model 的 raw output 转成 alpha，再根据 `sum_in_act` 控制 density 是在 activation 前相加还是 activation 后相加。最后仍然回到 NeRF 标准的 transmittance / weights / rgb_map 计算。

这说明 Blended-NeRF 的 blending 不是简单把两张图叠起来，也不是在 RGB 层面 alpha blend，而是在 volume rendering 的 raw density/color 层面处理。因此它可以影响遮挡关系和 depth map，而不只是像素颜色。

## 23. 与 CompoNeRF 的技术对照

本站此前精读过 CompoNeRF。两篇论文都和 text-guided NeRF、多对象或场景编辑相关，但任务边界不同。

| 维度 | Blended-NeRF | CompoNeRF |
| --- | --- | --- |
| 输入 | 已训练好的 existing NeRF + ROI box + prompt | 多对象文本 + 可编辑 3D box layout |
| 目标 | 编辑已有场景的局部区域 | 生成并组合多对象场景 |
| 表示 | fixed original NeRF + trainable generator NeRF | 多个 object-level NeRF + global composition |
| 控制接口 | 单个或局部 ROI box | 多对象 box layout |
| 核心问题 | 局部编辑保真与边界融合 | 多对象语义绑定与组合一致性 |
| 融合机制 | sample-level volumetric blending | density-based composition module |
| 代码状态 | 官方代码公开，可静态阅读 | 当时 README 显示 coming soon |

如果用一句话区分：**Blended-NeRF 更像“对已有 NeRF 做局部修图”，CompoNeRF 更像“用对象级 NeRF 搭一个多物体场景”。**

这种差异也影响工程复现。Blended-NeRF 需要先有原始 NeRF 权重和 ROI box；CompoNeRF 需要多对象布局和每个对象 prompt。前者更适合已有扫描/重建场景的编辑，后者更适合从 layout 和文本合成场景。

## 24. 与相关工作的关系

Blended-NeRF 站在几条工作线的交叉点上。

**NeRF / implicit scene representation** 提供了基础表示和可微渲染能力。Mildenhall 等人的 NeRF 证明了 MLP 加 volume rendering 可以从多视角图像重建新视角，但原始 NeRF 不解决对象级编辑。

**CLIP-guided generation** 提供了零样本文本监督。Dream Fields、CLIP-NeRF 等工作展示了语言-图像模型可以引导 3D 或 NeRF 表示优化，但常见问题是几何不稳定、局部控制弱、结果容易被 prompt 和视角采样影响。

**NeRF editing / object manipulation** 关注在已有隐式场景中改动局部对象。EditNeRF、ObjectNeRF、Volumetric Disentanglement 等方向提供了不同的 disentanglement 或局部编辑路径，但往往依赖类别先验、分解假设或更受限的编辑形式。

**Text-to-3D / diffusion prior** 如 DreamFusion、SJC、Magic3D 把 2D diffusion model 作为 3D 优化信号。Blended-NeRF 没有直接使用 Stable Diffusion 的 SDS 主线，而是基于 CLIP/BLIP guidance 和 NeRF priors，但面对的问题相似：2D 语义先验如何变成稳定 3D。

Blended-NeRF 的定位可以概括为：它不是最强的 text-to-3D generator，也不是最通用的 3D editor；它提出的是一个针对 **existing NeRF + local ROI** 的零样本编辑框架。

## 25. 工程复现清单

如果要在本地复现或改造 Blended-NeRF，至少要检查以下内容。

| 模块 | 需要准备或验证 |
| --- | --- |
| Python 环境 | README 使用 Python 3.9，依赖 PyTorch、CLIP、BLIP2/LAVIS 等 |
| 数据 | `download_data.sh` 支持 `nerf_synthetic`、`nerf_llff`、`nerf_real_360` |
| base weights | 放入 `base_weights/`，必须和 config、dataset、scene 坐标一致 |
| ROI box | `box_points/*.pt`，可通过 notebooks 定位 |
| config | prompt、scene path、box path、loss 权重、render scale、blend mode |
| 训练入口 | `python main.py --config ./configs/config.txt` |
| 中间结果 | `logs/<scene>/<expname>/plots/in_box`、renderonly、testset、checkpoint |
| loss 诊断 | CLIP/BLIP loss、transmittance loss、depth loss、是否出现透明/扁平结构 |
| 视觉 QA | 多视角渲染、ROI 边界、背景保真、对象尺度、视角一致性 |
| 指标 | CLIP direction similarity/consistency、LPIPS、R-Precision，只能作 proxy |

特别要注意数据和坐标一致性。NeRF synthetic、LLFF 和 real 360 场景的相机坐标、NDC 处理、scale、near/far 不完全相同。代码里有 `ndc2world`、`box_points_world` 等逻辑，说明 box points 可能需要在不同坐标系之间转换。

## 26. 局限性与批判

Blended-NeRF 的贡献清楚，但不能把它过度外推成通用 3D 编辑系统。

第一，ROI box 是强人工先验。方法效果高度依赖 box 位置、大小和形状。真实产品中，用户未必愿意或有能力在 NeRF 场景里手动定位 3D box。

第二，box shape 有限制。论文自己也提到，用 box 表示某些对象并不理想。例如 ship scene 中圆形或圆柱形区域可能比 box 更合适。对于细长结构、环状结构、复杂遮挡对象，box 会包含大量无关空间。

第三，CLIP/BLIP 指标不等于几何正确。模型可能生成语义上像 prompt 的渲染图，但 3D density 是薄片、泡沫、漂浮结构或多面 artifact。Fig. 6 说明 depth prior 能缓解，但不能完全解决。

第四，训练仍是优化式流程。每个编辑都需要针对场景和 prompt 训练 generator NeRF，不是实时交互式编辑。相比资产库检索或显式 mesh 操作，这类方法的迭代成本较高。

第五，输出不是生产级资产。NeRF 渲染图可看，并不意味着能导出干净 mesh、PBR 材质或可编辑 CAD 对象。

第六，评估体系偏 proxy。CLIP direction、BLIP R-Precision、LPIPS 可以支撑论文比较，但很难评价“对象是否真实可用”“边界是否物理合理”“是否适合游戏/影视/AR 生产”。

因此，对这篇论文更合理的判断是：它提出了一个扎实的 NeRF 局部编辑范式，而不是解决了所有 3D 场景编辑问题。

## 27. 推荐阅读路径

如果只想快速理解论文，可以按这个顺序读：

1. Abstract 和 Fig. 1：确认任务边界是 existing NeRF local editing。
2. Sec. 3：重点读 ROI rendering、distance smoothing 和 blending equations。
3. Fig. 3 / Fig. 4：理解边界平滑和 density activation 位置。
4. Fig. 5 / Table 1：看与 baseline 的主比较。
5. Fig. 6 / Table 2：看 priors 和 depth loss 为什么重要。
6. Fig. 7-14：看应用范围和真实场景效果。
7. README、`run_BlendedNeRF.py`、`optimization/losses.py`：把论文概念映射到代码。

如果从工程复现角度读，建议先看 README 和 configs，再看 notebooks 里的 box localization。因为没有 ROI box，即使理解了 loss，也无法启动一次真实编辑实验。

## 28. 结论

Blended-NeRF 的长期价值在于，它把文本/图像引导的 3D 编辑从“全局优化一个隐式场”推进到一个更可控的流程：**已有 NeRF 场景、用户指定 3D ROI、生成分支局部优化、原始分支保真、体渲染级融合**。

这套流程的优点是局部性强、保真性相对好、能支持对象插入、替换、融合和材质转换；缺点是需要手动 ROI、训练成本高、CLIP 监督粗、几何可靠性有限。

如果把它放在 3D 生成与编辑的技术谱系里，Blended-NeRF 不是终点，而是一个很重要的中间形态：它证明了在已有神经场景中做零样本局部语义编辑是可行的，同时也清楚暴露了后续系统必须解决的问题，例如更好的 3D selection、对象分解、实时优化、物理约束和资产级输出。

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

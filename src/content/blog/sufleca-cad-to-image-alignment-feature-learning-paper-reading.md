---
title: "SUFLECA 论文精读：大规模 NOC 特征学习、几何一致匹配与零样本 CAD 对齐"
description: "精读 SUFLECA 如何通过 12 数据集 NOC 弱监督、DPT 几何特征、互为 Top-k 对应、各向异性尺度共识和 SupeRANSAC，实现高效的单图 CAD-to-image 9D 对齐"
pubDate: "2026-07-22T14:18:27+08:00"
updatedDate: "2026-07-22T14:18:27+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "3D Vision"
  - "CAD Alignment"
  - "Pose Estimation"
  - "Computer Vision"
  - "Code Reading"
draft: false
---

从一张室内照片中识别出“这是一把椅子”，和把一份具体椅子 CAD 准确放进相机坐标系，是两个难度完全不同的问题。后者不仅要判断朝向，还要恢复三维平移与三个轴上的尺度；它必须面对遮挡、单图深度误差、真实照片与干净 CAD 渲染之间的域差异、对称结构造成的前后歧义，以及“检索到的 CAD 本来就不是图中那个实例”这一更根本的问题。

**SUFLECA: Scaling Up Feature Learning for CAD-to-image Alignment** 没有把这些困难重新包装成一个端到端姿态回归器。它选择了一条更接近经典几何的路线：先学习适合真实图像与 CAD 渲染互相匹配的紧凑稠密描述子，再从特征空间提出对应关系，用各向异性尺度下的成对距离一致性剔除错误匹配，最后交给鲁棒 3D registration 求解旋转、平移和尺度。论文的模型创新与几何算法创新互相依赖：只有更好的特征，最近邻仍可能产生 many-to-one 错配；只有更复杂的过滤，外观特征跨越真实/合成域时仍缺少几何语义。

这篇论文最容易被误读的词是 `zero-shot`。论文在方法部分明确限定：**zero-shot 只描述 CAD alignment stage**。目标检测、实例分割、度量深度和 CAD retrieval 都是上游输入。ScanNet25k 主表进一步使用 ROCA 的检测框与监督式 CAD retrieval、SAM2 mask，以及在 ScanNet25k 训练切分上微调的单目度量深度。因此，`33.4% / 42.3%` 证明的是“在固定上游协议下，弱监督训练的对齐模块具有很强的跨域能力”，不是“输入任意图片就能以零监督完成端到端 CAD 建模”。

本文严格以 arXiv [`2607.15058v1`](https://arxiv.org/abs/2607.15058v1) 为论文基线。源码阅读固定在官方仓库 tag `v1` 对应的 commit [`9df942e22601648e8165b0cffd7432ad4dc50d33`](https://github.com/snt-arg/SUFLECA/tree/9df942e22601648e8165b0cffd7432ad4dc50d33)。GitHub Release 的展示名称是 `v1.0`，真实 tag 却是 `v1`。官方开放了推理、评估、三个 checkpoint、ScanNet25k 验证资源和 notebook，同时在 README 中明确说明：训练与数据准备代码不计划发布。本文因此只做论文与源码静态阅读，不声称完成权重下载、render pool 构建、推理、训练或实验复现。

先给出全文判断：

> SUFLECA 的长期价值，是把基础视觉模型的 patch feature 进一步塑造成紧凑、几何可用、可以交给经典 registration 的 correspondence interface；它的主要风险，则仍在接口两端：上游 CAD retrieval 与 metric depth 决定问题是否可解，未公开的训练数据管线决定结果是否可独立复现。

## 1. 论文、作者、版本与开放状态

| 项目 | 内容 |
| --- | --- |
| 题名 | SUFLECA: Scaling Up Feature Learning for CAD-to-image Alignment |
| 作者 | Saad Ejaz, Miguel Fernandez-Cortizas, Javier Civera, Holger Voos, Jose Luis Sanchez-Lopez |
| 机构 | University of Luxembourg / SnT；I3A, Universidad de Zaragoza |
| arXiv | `2607.15058v1` |
| 提交日期 | 2026-07-16 |
| PDF | 9 pages |
| 学科 | Computer Vision and Pattern Recognition；Robotics |
| 论文许可 | CC BY 4.0 |
| 官方代码 | [snt-arg/SUFLECA](https://github.com/snt-arg/SUFLECA) |
| 源码基线 | tag `v1`，commit `9df942e22601648e8165b0cffd7432ad4dc50d33` |
| 代码许可 | Apache-2.0；vendored SupeRANSAC 为 MIT |
| 公开范围 | 推理、评估、checkpoint、验证资源、demo；不含训练与数据准备代码 |

截至本文实施时，arXiv 只有 v1，也没有可确认的正式会议录用信息，所以本文称其为 arXiv v1 预印本，不根据 IEEE 风格排版推断投稿或录用状态。版本边界尤其重要：论文与仓库发布时间非常接近，但仓库已包含一个论文主文没有展开的 DINOv3 零样本 CAD retrieval demo，也包含与论文公式不同的 NMS score fusion 和 information-matrix 实现。源码是理解公开系统的证据，不是逐行复刻论文实验的证明。

代码仓库的版本也有一个小陷阱。Release 页面标题显示 `v1.0`，但下载 URL、Git tag 和 README 链接都使用 `v1`。固定源码链接时应使用 commit，而不是会漂移的 `main`。仓库 `LICENSE` 文本和源文件 SPDX header 声明 Apache-2.0；GitHub License API 因许可证文件不是标准全文而未正确识别，这不改变仓库自身的许可声明，但使用者仍应保留版权、NOTICE 和第三方 SupeRANSAC 的 MIT 文本。

## 2. 一句话贡献：让 Feature 真正进入几何求解器

论文可以压缩为两个互补贡献。

第一，SUFLECA 不满足于直接拿 DINO、DUNE 一类基础模型的高维 patch feature 做最近邻，而是用 Normalized Object Coordinates（NOC）监督训练一个 DPT head。训练覆盖 12 个真实与合成数据集、674,851 张 frame，平均每帧有 3.38 个已标注对象。冻结的 perception encoder 保留通用视觉先验，DPT 在 NOC 任务中学习空间连续、几何相关的紧凑 descriptor。

第二，SUFLECA 不把 feature nearest neighbor 当成最终对应。它先做 mutual top-$k$，再估计粗略各向异性尺度，构造 correspondence 之间的几何一致矩阵，用主特征向量找到互相支持的匹配集合，最后运行 SupeRANSAC。这样，语义特征负责提出可能对应，几何负责判断这些对应能否共同解释一个 9D transformation。

两部分缺一不可。只扩大数据而保留 naive nearest neighbor，Table IV 的结果仍会明显下降；只在旧特征上叠加几何过滤，又会因为真实照片和 CAD render 的 feature domain gap 得不到足够多正确初始匹配。论文的系统判断是：**CAD alignment 的瓶颈不只是 pose solver，而是 solver 之前的 correspondence distribution。**

## 3. 任务边界：Retrieval、Alignment、Reconstruction 不是一回事

SUFLECA 的正式输入不是“任意一张照片”，而是以下数据：

1. 一张 RGB 图像；
2. 相机内参；
3. 目标对象的实例 mask；
4. 对象区域的 metric depth；
5. 一个已经检索出的 candidate CAD；
6. 该 CAD 的若干预渲染视图、point map 与 feature cache。

它的输出是 candidate CAD 到相机坐标系的旋转 $R$、平移 $t$、各向异性尺度 $S$，外加一个拟合质量分数。也就是说，SUFLECA 回答的是：“**给定这个 CAD，它应该怎样放进照片中的对象位置？**”它本身不回答“该选哪个 CAD”，更不从像素生成一个新 mesh。

**CAD retrieval** 在资产库里找候选形状，重点是实例或类别相似性。错误检索意味着后续不存在完全正确的刚性加尺度变换。

**CAD-to-image alignment** 在候选形状给定后估计姿态与尺度，重点是 2D/3D 或 3D/3D correspondence 和 registration。

**Category-level pose estimation** 往往假设类别内共享 canonical frame，但输入实例可以没有精确 CAD 对应。

**Single-view reconstruction** 则尝试恢复输入实例的表面，输出可能是 mesh、SDF、NeRF 或 Gaussian，不要求来自离散 CAD 库。

**Scene modeling** 还要处理多个对象、建筑结构、支撑关系、碰撞与布局。Diorama 属于这一层，SUFLECA 更像其中一个对象 pose module。

把这些任务分开后，“零样本”才有准确含义。一个系统可以在 alignment 上不使用目标 benchmark 的 pose label，却在 retrieval、depth 或 segmentation 中使用目标域监督；也可以完全零样本检索到错误 CAD，再由很强的 alignment 模块给出视觉上勉强合理的姿态。评估结论必须跟随组件边界，而不能跟随产品演示的输入输出表象。

## 4. Zero-shot 的准确语义与主实验上游

论文在 Section III-B 写得很清楚：`zero-shot refers specifically to the alignment stage`。这句话需要贯穿整篇阅读，而不是只在脚注出现。

ScanNet25k 主协议中，各个被比较的 correspondence-based 方法共享：

- ROCA bounding boxes；
- ROCA CAD retrieval；
- SAM2 instance masks；
- 在 ScanNet25k training split 上微调的 monocular metric depth；
- Scan2CAD/ Vid2CAD 风格的 NMS 与 20cm/20°/20% 正确性阈值。

SUFLECA 为了让**特征学习本身**满足严格零样本，又从训练集合中移除了 ScanNet-derived images，使用 `sufleca-wo-scannet` checkpoint。这个控制避免 DPT head 在训练时直接看到主 benchmark 的图像分布。但上游 depth model 的 fine-tuning 仍然存在，ROCA retrieval 也仍是监督式的，所以更准确的描述是：**alignment descriptor 与 correspondence estimator 对 ScanNet25k pose supervision 零样本，而完整 pipeline 不是。**

论文也做了更接近完整零样本的检索实验：GroundedSAM 检测/分割加 OSCAR CAD retrieval。此时实例 retrieval accuracy 只有 `3.8%`，SUFLECA alignment 仍得到 `22.5% / 33.1%` 的 category/instance accuracy，但明显低于使用 ROCA retrieval 时的 `33.4% / 42.3%`。这组结果不是附属消融，它揭示了系统真正的下一瓶颈。

## 5. 9D 对齐的数学形式与坐标约定

论文把 CAD 坐标系中的点 $mathcal X_{mathrm{cad}}$ 映射到相机坐标系：

$$
\mathcal X_{\mathrm{cam}}=RS\mathcal X_{\mathrm{cad}}+t,
$$

其中：

$$
R\in SO(3),\qquad
S=\operatorname{diag}(s_x,s_y,s_z),\qquad
t\in\mathbb R^3.
$$

旋转有 3 个自由度，平移有 3 个，各向异性尺度有 3 个，所以称为 9D pose。严格来说，旋转矩阵有 9 个元素但只含 3 个自由度；“9D”指参数自由度而不是矩阵长度。

各向异性尺度非常关键。统一尺度只允许 CAD 整体变大或变小，不能把一个偏窄的检索椅子调整成照片里更宽的椅子。引入 $s_x,s_y,s_z$ 提升了对 inexact CAD 的容忍度，却也让几何过滤更困难：刚体变换保持点对距离，各向异性尺度不保持欧氏距离，不能直接用 $\lVert p_i-p_j\rVert=\lVert q_i-q_j\rVert$ 判断 correspondence consistency。

论文还输出拟合质量 $\mathcal S_{\mathrm{fit}}$，用于 NMS 排序或未来的多视图融合。它不是 pose 的概率，也没有经过概率校准，而是由 registration Jacobian 构造的信息矩阵的 log-determinant。后文会看到，当前代码甚至没有完全采用论文中的残差方差归一化，所以这个 score 应被理解为工程 ranking signal，而不是可跨场景比较的置信概率。

## 6. Fig. 1：Accuracy、Runtime 与 VRAM 的联合主张

![SUFLECA 在 ScanNet25k 上的准确率、速度与显存比较](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-fig01-accuracy-runtime-vram.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Fig. 1, CC BY 4.0.*

Fig. 1 把纵轴 accuracy、横轴 objects per second 与气泡大小 VRAM 放到一张图里。SUFLECA 位于右上角，论文借此同时提出三个判断：精度高、每实例运行快、特征维度和显存更小。虚线代表表中最强监督方法的平均水平，SUFLECA 的主结果高于这条线。

图解决的问题不是“谁的绝对速度最快”这么简单，而是提醒读者：CAD alignment 方法可能通过大量 render、迭代 refinement 或高维 descriptor 换取精度。SUFLECA 的 384 维特征和单次 registration 路径，避免了 ZeroCAD/FoundationPose 一类 refinement 成本。SUFLECA-S 用 256 维 feature 进一步压缩资源。

但这张图有三个不能外推的边界。第一，runtime 只计 alignment，不含 detection、segmentation、depth、CAD retrieval 和 render-pool preparation。第二，所有数字来自 RTX 5090 workstation，不能直接外推到机器人边缘 GPU。第三，ZeroCAD 没有开源，论文的时间和显存来自作者构造的 runtime-only proxy，而且作者明确称其为真实耗时的 lower bound。气泡图更适合表达同一实验环境下的相对系统取舍，不是完整产品 SLA。

## 7. 三类 CAD 对齐范式

论文相关工作可以按“姿态从哪里来”分成三类。

**Direct regression** 直接从图像 feature 回归 rotation、translation、scale。Mask2CAD、Total3D 一类系统可以端到端训练，推理路径短，但强依赖目标数据集的 pose annotation，跨类别和跨域泛化通常受限。回归器还需要隐式学习对称性、遮挡和尺度多解。

**Render-and-compare** 从初始 pose 出发，不断渲染 CAD，再根据 render 与输入的差异更新姿态。SPARC、MultiObj-SPARC、FoundationPose 的相关路径能利用细粒度对齐信号，但迭代次数、render 成本和初值质量决定延迟；错误 CAD 或不可见区域也会让 photometric/feature residual 误导优化。

**Correspondence plus registration** 先建立 CAD point 与 image observation 的对应，再用几何求解。ROCA 直接预测 NOC，ZeroCAD 用弱监督 adapter 改造基础视觉特征，Diorama 用 DINOv2 correspondence 与额外尺度模块。其优势是神经网络与几何模块职责清楚，RANSAC 能显式处理 outlier；其风险是 correspondence 一旦太稀疏、太集中或 many-to-one，后面的 solver 无法恢复信息。

SUFLECA 属于第三类，但它的贡献不是简单换一个 backbone。它把“特征空间应具有什么性质”和“对应集合应满足什么几何结构”一起设计：NOC loss 让 descriptor 对对象规范坐标敏感，mutual top-$k$ 限制 feature matching，尺度共识再施加变换模型约束。

## 8. 监督方法与零样本方法的比较口径

Table II 中的符号容易造成第二层误读：`✓` 表示 full supervision，`*` 表示 weak supervision，`–` 表示 unsupervised；论文把 `*` 和 `–` 都放入 zero-shot alignment 讨论。这里的 zero-shot 不是“完全没有训练”，而是“不使用目标任务的直接 9D pose supervision”。

ROCA、SPARC、MultiObj-SPARC、CosCAD 使用目标域 pose 监督。它们是重要参照，因为它们代表任务专用训练能达到的水平。

DINOv3-L、DUNE-B 直接用预训练 feature，没有针对 CAD alignment 的 NOC adaptation。它们测量基础模型原生 3D awareness 的上限。

ZeroCAD 与 SUFLECA 都使用 NOC 弱监督。区别在于 ZeroCAD 主要依赖 9 类、30 万张合成单对象图像，并用高维 feature 与 nearest-neighbor matching；SUFLECA 扩大到 12 个真实/合成数据集，并重写 correspondence estimator。

Diorama 面向开放世界场景装配，使用额外的 GigaPose scale predictor 与布局模块。将它的对象级 pose 结果放进相同表格有参考意义，但不能抹去任务范围和上游差异。

DiffCAD、SDFit 需要类别 prior 或较重的生成/优化过程。FoundationPose 原本处理已知对象 6D pose，表中采用扩展到 9D 的实验版本。所谓“SUFLECA 超过监督方法”只在具体 ScanNet25k protocol 和共同上游下成立，不能推成所有类别、所有 CAD 库和所有输入条件的总体排名。

## 9. NOC：把对象表面放进统一坐标系

Normalized Object Coordinates 为对象类别或实例建立 canonical coordinate frame。最常见的直觉是：把对象沿三个轴平移和缩放到规范立方体，每个可见表面点携带一个 $(x,y,z)$ 规范坐标。若图像像素能预测 NOC，就得到“这个像素对应对象规范空间的哪个位置”，再结合深度或 2D 几何恢复 pose。

NOC 同时提供了语义和几何信号。RGB 上两个椅腿可能纹理相同，但在 canonical frame 中位于左前与右后；只按外观最近邻容易互换，NOC supervision 迫使 feature 保留部件位置。对连续表面而言，相邻像素的 NOC 通常也连续，因此 DPT feature 会获得空间平滑性。

不过，NOC 并非天然真值。它依赖 CAD 的 canonical orientation、对象 pose、尺度定义、mask 和 depth。不同数据集可能使用不同轴约定，同一类别的对称对象也可能有多个等价 canonical frame。论文的数据工程重点之一，正是把异构 annotation 转成统一、经过验证的 NOC map，而不是把 12 个数据集直接拼接。

SUFLECA 训练时预测 NOC，测试时却丢弃 NOC head。这说明 NOC 在这里是 representation-shaping task：模型不依赖输出坐标值的绝对精度完成 pose，而把 DPT 中间 feature 当 descriptor。这样既吸收几何监督，又保留高维特征区分外观和部件的能力。

## 10. 为什么原生 Foundation Feature 不够

DINOv3、DUNE 等视觉基础模型的 patch feature 已包含一定几何意识，能在跨图像 matching 中发现语义相似部件。问题是，它们的主训练目标并不是 CAD-to-image registration。

首先，feature 容易被纹理、颜色和上下文支配。真实椅子带材质、阴影与遮挡，CAD render 可能是随机或单色材质；外观相似度并不稳定。其次，基础模型常追求语义不变性。同一类别不同部件在高层语义上都属于“椅子”，但 registration 需要区分左/右、前/后和具体表面坐标。再次，feature dimension 很高。DINOv3-L 为 1024 维、DUNE-B 为 768 维，像素级相似度矩阵会消耗显存和计算。

最后，nearest-neighbor matching 本身没有一对一约束。一个目标 patch 可能成为多个 source patch 的最近邻；重复结构会形成整片错误对应。即使每一对 feature similarity 都高，它们也未必由同一个 $R,S,t$ 同时解释。

SUFLECA 对这些问题的回答分别是：真实/合成混合 NOC 监督缩小域差异；DPT 输出压缩到 384 或 256 维；mutual top-$k$ 提高对应的双向可信度；geometric consensus 把局部相似度提升为集合级一致性。换言之，它不是宣称 foundation model 没有 3D 能力，而是把这份能力改造成 registration 所需的接口。

## 11. Fig. 2：12 个数据集如何汇合成几何监督

![SUFLECA 多数据集 RGB、NOC 与合成配对示例](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-fig02-multidataset-noc-examples.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Fig. 2, CC BY 4.0.*

Fig. 2 的三行分别是 RGB frame、叠加在 RGB 上的 NOC，以及合成 counterpart。第一列是室内视频，第二列是室内扫描/移动相机，第三列是 casual bicycle photo。它要解决的问题是：训练数据不能只覆盖干净、居中、单对象 CAD render，否则 feature 在真实 clutter 和 occlusion 中仍会失效。

中间一行强调 NOC 不一定由原数据直接提供。有些数据集提供完整对象 pose 与 CAD annotation，作者通过 render、SAM2 mask、depth backprojection 计算 NOC；有些使用 OmniNOCS 或数据集自带标注。底部合成图与真实图共享大致 pose、遮挡和 scene context，但会随机化背景、纹理，并偶尔替换为视觉相似对象。这种 paired distribution 直接模拟推理时“真实 observation 对 synthetic CAD render”的跨域 matching。

图支持“多样数据和 real-synthetic pairing 是性能来源之一”，但不能证明所有 674K frame 的 annotation 同样可靠。不同来源的 depth、pose、CAD fit 和 mask 噪声差异很大。论文使用 NOC overlay error、有效像素和相机位移阈值过滤，具体阈值和完整清洗脚本却没有公开，所以数据规模不能等同于可复现的训练集。

## 12. Table I：训练数据构成与监督来源

![SUFLECA 训练数据集统计](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-table01-training-datasets.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Table I, CC BY 4.0.*

| RGB 来源 | NOC/标注来源 | Frames | 平均对象/帧 |
| --- | --- | ---: | ---: |
| Pascal3D+ | Included | 6,108 | 1.00 |
| Objectron | OmniNOCS | 7,357 | 1.20 |
| ARKitScenes | OmniNOCS | 52,146 | 3.55 |
| REAL275 | Included | 7,072 | 5.00 |
| RealEstate10K | CAD-Estate | 127,214 | 2.16 |
| ScanNet | Scannotate | 117,284 | 2.58 |
| ScanNet++ | Scannotate++ | 88,198 | 2.96 |
| Pix3D | Included | 11,622 | 1.00 |
| ObjectNet3D | Included | 20,520 | 1.00 |
| 3D-FRONT | Rendering | 69,578 | 6.43 |
| Hypersim | OmniNOCS | 44,545 | 13.55 |
| ShapeNet | Rendering | 123,207 | 1.00 |
| **总计** |  | **674,851** | **3.38** |

横线以上主要是真实世界数据，以下是合成数据。组合方式覆盖了三种互补来源：单对象或 casual images 提供类目和外观多样性；室内视频/扫描提供 clutter、遮挡与相机运动；3D-FRONT、Hypersim、ShapeNet 提供可控几何和密集真值。

表中 `frames` 不等于独立对象样本。按论文源码注释汇总，约有 2,280,568 个对象实例监督，Hypersim 平均每帧 13.55 个对象，ShapeNet 每帧只有一个。高对象密度使网络在 occlusion 与上下文中学习，但也意味着一帧的 pose/depth 系统误差可能同时污染多个对象。

严格 ScanNet25k 零样本训练会删除 ScanNet 行，frame 数降到 `557,567`，平均对象数变为 `3.55`。注意这并不删除所有“室内”数据，也不删除 ARKitScenes、ScanNet++ 等相近域；它控制的是直接 ScanNet-derived overlap，而不是构造极端 domain shift。

## 13. 数据规模的正确读法：674K frame 不等于 674K 对象

论文最醒目的数据数字是 `674,851` 张 frame，但训练监督真正落在对象像素上。由 TeX 中的统计注释可得，这些 frame 约包含 `2,280,568` 个对象实例，平均每帧 `3.38` 个。两组数字回答的是不同问题：frame 数描述图像与相机上下文的多样性，对象实例数描述 NOC 监督的密度。

这种密集监督有三项好处。第一，同一帧内不同对象共享光照、遮挡和背景，模型必须在复杂 scene context 中保持对象内的几何一致。第二，Hypersim、3D-FRONT 这类高对象密度数据增加了桌椅、柜体等共现场景。第三，real/synthetic 数据大致平衡，使“真实图查询 synthetic CAD render”不再只是测试时才出现的 domain gap。

但对象数不能直接当成独立样本数。相邻视频帧高度相关，同一 CAD 会在多个视角重复出现，单帧的错误深度或 pose 还可能同时污染多个对象。论文通过固定间隔采样和最小相机位移降低冗余，却没有公开完整阈值与最终对象级清单。因此更准确的结论是：SUFLECA 显著扩大了弱监督的覆盖面，而不是已经提供了一个可逐项审计的 228 万独立真值数据集。

## 14. 真实图与合成 counterpart 为什么要配对

推理时，匹配两端天然不对称：一端是有遮挡、光照、材质和背景的真实对象 crop，另一端是姿态可控、背景干净的 CAD render。只在纯 CAD render 上训练，网络很容易把材质和轮廓风格当作捷径；只在真实图上训练，又缺少精确而密集的规范坐标监督。

作者在存在 CAD annotation 时，为真实 frame 渲染一个近似同位姿的 synthetic counterpart。合成侧随机背景与对象纹理，部分对象还会替换成视觉相似实例。这里“近似同位姿”很重要：两张图不必形成逐像素 photometric pair，监督目标仍是各自的 NOC；共享视角、遮挡和场景上下文只是帮助网络学会跨域保持几何身份。

可以把这套设计理解为三种变化被有意拆开：

- **应保留的信息**：对象规范坐标、局部几何、朝向与部件关系。
- **应忽略的信息**：背景、材质、纹理、真实或合成的渲染风格。
- **允许变化的信息**：同类或视觉相似 CAD 的具体实例形状。

它比普通 domain randomization 更有针对性，因为变化围绕 CAD alignment 的两端构造。不过论文没有给出“替换为相似对象”的检索规则、替换比例和 paired/unpaired 数据消融，所以不能单独量化这一操作贡献了多少性能。

## 15. NOC Derivation：从 pose annotation 到像素级规范坐标

Normalized Object Coordinate（NOC）把对象表面点映射到类别或实例的 canonical cube。若 CAD 点为 $\mathbf p_{\mathrm{cad}}$，先按 CAD 的中心与尺度做规范化，就得到与图像视角无关的 $\mathbf n\in[0,1]^3$（具体中心范围随数据源约定）。每个可见像素的 NOC 值回答：“这个像素对应 CAD 规范空间的哪个位置？”

论文的数据生产链为：

1. 根据数据集给出的 CAD pose 把真值对象投影到 RGB，得到近似二维框。
2. 以框提示 SAM2，提取更贴合图像边缘的对象 mask。
3. 用传感器 depth 或 monocular metric depth 将 mask 内像素反投影到三维相机空间。
4. 利用已知的 CAD-to-camera 变换，把三维点变换回对象规范空间，写成 NOC map。
5. 将生成的 NOC 与 CAD overlay 的可见部分比较，过滤明显错误实例。

若像素坐标为 $(u,v)$、深度为 $d$、相机内参为 $(f_x,f_y,c_x,c_y)$，反投影是：

$$
\mathbf q(u,v,d)=
\begin{bmatrix}
(u-c_x)d/f_x\\
(v-c_y)d/f_y\\
d
\end{bmatrix}.
$$

随后用真值 $R,S,t$ 的逆变换恢复 CAD 坐标。这个过程说明 NOC 并非“从 SAM2 自动得到”：SAM2 只给 mask，几何值依赖 pose、depth、intrinsics 和 CAD annotation。任何一环错误都可能形成看似平滑、实际错位的监督。

## 16. NOC Verification：弱监督并非无条件接收

SUFLECA 的监督被称为 weakly supervised，不代表它没有质量控制。作者将派生 NOC 与渲染的 CAD NOC overlay 比较，实例的平均误差超过阈值就删除；一帧内有效 NOC 像素总数不足时，整帧删除。对于 RealEstate10K、ScanNet、ScanNet++ 等高帧率轨迹，还按时间间隔下采样并要求最小相机位移，以减少近重复视角。

这套过滤机制针对四类常见污染：

- CAD retrieval 或 annotation 指向了不相符的模型；
- pose 略偏，使投影轮廓与真实对象错开；
- depth 在反光、边缘或遮挡处错误；
- SAM2 mask 吞入背景或漏掉对象区域。

它仍有选择偏差。容易对齐、可见面积大、深度完整的对象更容易通过；薄结构、重遮挡、小对象和非典型外观更容易被过滤。最终模型在 benchmark 上更稳定，可能部分来自训练分布被“清洁化”。如果复现，应同时记录原始实例数、各过滤原因计数、类别保留率和可见面积分布，而不只记录最终 frame 数。

## 17. 严格 ScanNet25k 切分与 checkpoint 语义

论文为了验证 alignment stage 的 zero-shot 性质，从训练混合物中删除 ScanNet-derived images，得到 `557,567` frames，平均 `3.55` 对象/帧。主结果使用这一严格设置。官方仓库却把默认 checkpoint alias 配成 `sufleca`，而不是 `sufleca-wo-scannet`，因此“直接运行 README 默认配置”与“复现论文严格零样本主表”不是同一件事。

三个公开别名应这样理解：

| Checkpoint | 主要语义 | 适合用途 |
| --- | --- | --- |
| `sufleca` | DUNE-B + 384 维 DPT feature，训练数据包含 ScanNet-derived 部分 | 默认演示、追求目标域性能 |
| `sufleca-small` | DUNE-S + 256 维紧凑 feature | 显存和延迟受限的 alignment |
| `sufleca-wo-scannet` | 移除 ScanNet-derived training images | 论文 ScanNet25k 严格 zero-shot 口径 |

这里也不能把 `wo-scannet` 写成“没有任何相似室内数据”。ARKitScenes、ScanNet++、3D-FRONT、Hypersim 等仍带来室内几何 prior。严格切分证明的是没有直接使用 ScanNet-derived frame，而不是模型完全脱离室内数据分布。

## 18. Fig. 3：冻结编码器上的 DPT 几何适配器

![SUFLECA 特征模型架构](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-fig03-feature-model-architecture.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Fig. 3, CC BY 4.0.*

Fig. 3 从左到右是输入图像、冻结的 perception encoder、多层 patch token、DPT reassembly/fusion、稠密 feature，以及只在训练时使用的 binned NOC head。它解决的不是直接 pose regression，而是怎样把已有 foundation feature 轻量地旋转到“几何对应更容易”的特征空间。

冻结 backbone 有两个含义。优化上，训练只改变 DPT 与 NOC decoder，降低 12 数据集联合训练时灾难性遗忘和显存成本。表示上，DUNE 原有的外观、语义和姿态知识被保留，NOC 监督负责塑形而不是从零学习视觉表征。多层 hook 则让 DPT 同时拿到浅层局部结构和深层语义。

图支持“浅层几何监督足以把预训练 feature 改造成 correspondence descriptor”，但不证明 backbone 无关。论文使用 DUNE-B/S；换成 DINOv3、DINOv2 或其他编码器时，token 网格、register token、层选择和通道统计都要重新适配。

## 19. DUNE 是预训练底座，不是 SUFLECA 从零训练的网络

DUNE 是论文选用的 perception encoder。SUFLECA-B 对应 DUNE-B，backbone 约 102M 参数；SUFLECA-S 对应更小的 DUNE-S，论文整体参数量约 29M。SUFLECA 的训练冻结编码器，只更新 DPT 与 NOC head。因此“SUFLECA 用 674K 图训练 102M 网络”并不准确，更准确的说法是：它在冻结的 DUNE feature 上训练稠密几何适配层。

冻结策略带来可重复的工程优势：

- 不需要维护 backbone 的低学习率和分布式梯度状态；
- 多数据源 annotation 噪声不会直接重写全部 foundation feature；
- CAD render cache 可以与 checkpoint alias 绑定，稳定生成 feature；
- 小模型与大模型共享同一几何求解后端。

代价是 feature 能力上限仍受 DUNE 决定。若原 backbone 对透明物、细杆、无纹理平面或 uncommon category 表示不佳，DPT 只能在已有 token 中重组信息。论文的 blend 变体正是对这类遗忘与覆盖不足的折中：将 NOC-aligned descriptor 与原始 DUNE-B output feature 拼接，而不是强迫单个 feature 同时承担所有语义与几何任务。

## 20. DPT 如何把 patch token 变成稠密对应特征

Dense Prediction Transformer（DPT）在这里不是深度估计器，而是一个通用多尺度重组 head。编码器不同层的 token 先移除 class/register token，再 reshape 成二维 patch grid；各层通过投影统一通道，按不同倍率 reassemble；随后自粗到细融合并上采样到接近像素空间。

若第 $l$ 层 token 为 $Z_l\in\mathbb R^{B\times P\times C_l}$，patch 网格为 $H_p\times W_p$，则核心重排是：

$$
Z_l\longrightarrow F_l\in\mathbb R^{B\times C_l\times H_p\times W_p}
\longrightarrow \tilde F_l\in\mathbb R^{B\times d\times H_l\times W_l}.
$$

多层 $\tilde F_l$ 经过 residual convolution unit 与 fusion block 后得到 $F_{\mathrm{DPT}}$。主模型的 descriptor dimension 为 `384`，小模型为 `256`。推理前对通道维做 $\ell_2$ 归一化，因此点积可直接解释为 cosine similarity：

$$
\bar{\mathbf f}=\frac{\mathbf f}{\lVert\mathbf f\rVert_2+\varepsilon},
\qquad
\bar{\mathbf f}_i^\top\bar{\mathbf f}_j=\cos(\mathbf f_i,\mathbf f_j).
$$

这一步是性能与几何之间的接口：后续没有再训练一个 pose network，所有 render selection 和 correspondence 都只消费稠密归一化特征。

## 21. Binned NOC Head：分类提供分布，回归提供连续坐标

NOC head 对 $x,y,z$ 三个轴分别预测离散概率。主模型每轴 `64` 个 bin，小模型每轴 `50` 个 bin。设某轴 bin 中心为 $c_j$、logit 为 $a_j$，期望坐标是：

$$
\hat n=\sum_{j=1}^{m}c_j\,\operatorname{softmax}(a)_j.
$$

离散分类比直接回归更容易表达边界附近的不确定性，也能提供更稳定的 early training signal；期望值又避免最终 NOC 被限制在 bin center。论文描述 head 为两层浅卷积，因此几何信息主要必须存在于 DPT feature 中，head 没有足够容量在最后一步凭空恢复复杂表面。

要注意，binning 是训练目标的参数化方式，不是推理时的 feature quantization。公开推理 checkpoint 只需加载 DPT head 的权重，输出仍是 256/384 维连续 descriptor。把“64 bins”解释成“SUFLECA feature 只有 64 维”是错误的。

## 22. NOC 联合损失：类别正确与连续误差互补

论文的训练目标是：

$$
\mathcal L_{\mathrm{NOC}}
=\mathcal L_{\mathrm{CE}}
+\lambda\left\lVert\hat{\mathcal N}-\mathcal N\right\rVert_1,
\qquad \lambda=0.33.
$$

$\mathcal L_{\mathrm{CE}}$ 对每个坐标轴的真值 bin 做交叉熵，强调粗粒度 canonical region；$\ell_1$ 项作用于期望 NOC，补回 bin 内连续精度。对 mask 外像素不应计算对象 NOC loss，否则背景会被迫映射到任意规范坐标。论文没有公开完整 loss masking、数据源采样权重和增强实现，这些是复现训练时必须补齐的细节。

主模型训练 `50` epochs，使用 AdamW，学习率 `1.1\times10^{-4}`。这些数字描述论文设置，不是从公开仓库可以直接执行的 recipe，因为训练入口、optimizer config、dataset loader 和 NOC production pipeline 均未发布。公开 checkpoint 能验证 inference API，却不能用于重建训练证据链。

## 23. 为什么测试时必须丢弃 NOC Head

如果直接把预测 NOC 当 correspondence，一种自然方案是把图像像素和 CAD 点按最近 NOC 匹配。但 SUFLECA 刻意丢弃 head，使用它前面的高维 DPT feature。这揭示了论文真正的建模判断：NOC 是表征学习的监督坐标系，不是最终描述子的容量上限。

离散 NOC 只有三维，类别内对称位置、重复部件和遮挡边界可能映射得过于接近；384 维 feature 则可保留局部形状、上下文和可见性。NOC head 迫使 feature 与几何相关，高维 feature 继续保留区分 correspondence 所需的信息。这个“训练时有 decoder、测试时取 decoder 前 feature”的结构类似用 auxiliary task 学 representation。

公开代码也印证了边界：[`featurizer.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/featurizer.py)加载 DPT feature extractor checkpoint 并输出归一化稠密特征，没有公开论文中的 binned classifier 训练实现。因而不能根据仓库目录缺少 NOC head 就断言论文未使用它；正确解释是它不属于发布的 inference graph。

## 24. SUFLECA、SUFLECA-S 与 SUFLECA-blend

三个变体在同一 alignment algorithm 上更换 descriptor：

| 变体 | Backbone / feature | 维度 | 参数量 | 主要取舍 |
| --- | --- | ---: | ---: | --- |
| SUFLECA | DUNE-B 的 NOC-aligned DPT | 384 | 102M | 主模型，几何与效率平衡 |
| SUFLECA-S | DUNE-S 的 NOC-aligned DPT | 256 | 29M | 最低 VRAM 和延迟 |
| SUFLECA-blend | SUFLECA + frozen DUNE-B output | 1152 | 102M 级 backbone | 未见类别更强，但匹配成本更高 |

blend 并不是模型 ensemble，也不是分别跑两个 pose solver。它在一次 backbone forward 后拼接原始与 NOC-aligned feature，再进入同一 render selection、mutual top-k 和 registration。论文在 ScanNet25k 常见室内类别上发现 blend 与主模型接近或略低；在 CO3D unseen categories 上通常更好。这支持“弱监督会提高几何性，也可能削弱超出监督类别的语义泛化”这一解释。

小模型则说明描述子维数并非越大越好。若 matching quality 足够，256 维 feature 能降低矩阵乘法、render cache、显存和传输成本。这个结论只在论文的 DUNE-S 配置和 benchmark 上成立，不能推广成所有 256 维 feature 都优于 foundation model 的 1024/2048 维表示。

## 25. 公开推理管线：特征只是入口，几何求解才是闭环

![SUFLECA 官方仓库推理管线](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-repo-inference-pipeline.webp)

*Source: snt-arg/SUFLECA, tag v1, Apache-2.0.*

这张图来自官方仓库而非论文，因此不能标为论文 Fig. 5。它补足了主文没有单独画出的 inference contract：输入是 RGB、对象 mask、metric depth、intrinsics 和 candidate CAD；CAD 侧预先渲染固定视角并缓存 point map 与 feature；在线阶段选择最接近的 render，匹配 correspondence，再由鲁棒配准输出 $R,S,t$ 和质量分数。

推理链可压缩成：

```text
RGB + mask + depth + K
  -> target RGB-D point cloud + SUFLECA features
candidate CAD render cache
  -> select one of six views
  -> mutual top-k feature correspondences
  -> anisotropic-scale geometric consensus
  -> SupeRANSAC / Procrustes
  -> R, S, t, information matrix, score
```

图支持“公开代码覆盖单对象 alignment”这一结论，但它不包含 candidate CAD 从何而来。论文主表把 ROCA 的 box 与 CAD retrieval 结果作为输入；仓库 demo 另用 Grounding DINO、SAM2、MoGe-2 和 DINOv3 retrieval。两条路径共享 SUFLECA alignment，却不是同一个端到端系统。

## 26. 为什么保留二维 Render Feature，而不先聚合到 Mesh

一些方法从多个 render 抽取二维 feature，再把它们聚合到 CAD mesh 顶点，形成一个三维 descriptor field。这样便于任意视角查询，但必须处理遮挡、同一点多视图冲突、可见性加权和 mesh sampling。聚合策略如果简单平均，会抹掉视角相关线索；如果只取最大相似，又容易被单个错误 view 支配。

SUFLECA 保留每个 render 的二维 feature map 和对应 point map，先选择一个最接近 query 的 view，再在二维 feature space 匹配。优点是：

- feature 仍处于编码器训练时熟悉的图像网格，不需要任意的 mesh aggregation；
- 每个 render pixel 有明确可见表面点，避免把背面点混入当前匹配；
- 缓存结构简单，可直接存 `clean_points` 与 `clean_features`；
- 在线只处理一个 render，matching matrix 的规模可控。

缺点也直接：六个固定视角之间仍有量化误差，选错 view 后后续 correspondence 会整体恶化；被选 view 看不到的表面无法贡献；对旋转对称对象，多个 view 的得分可能接近。论文没有联合优化 view distribution，也没有保留多视图 hypothesis 到 RANSAC，因而“二维保真”是以 view-selection bottleneck 为代价的。

## 27. Render Pool：每个 CAD 的离线资产合同

论文对每个 CAD 维护 $n=6$ 个固定视角。公开 [`render_cache.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/render_cache.py)把原始 render 处理成两类缓存：用于快速 view scoring 的 `scores.npz`，以及每个视角的 `clean_XX.npz`。后者至少包含清理后的三维点和归一化 feature。

缓存必须与三项元数据绑定：输入分辨率、checkpoint alias/标识、CAD 几何中心。更换 SUFLECA-S 或 blend 后，feature dimension 和数值分布变化，旧 cache 不能静默复用；更改 `image_size` 也会改变网格采样。仓库脚本因此会验证 meta，并提供生成 checkpoint-specific cache 的入口。

render pool 是性能数字中容易被忽略的离线成本。Table VI 的 `0.53s` 不包含为整个 ShapeNetCore.v2 资产库渲染 RGB、point map、提取 feature 和写 cache。生产系统还要解决 CAD 版本、材质渲染器、canonical orientation、坐标单位、损坏 cache 和增量资产更新。把 render pool 当成普通静态目录可以做 demo，但长期服务需要把它视为有 schema 和版本的派生数据集。

## 28. Metric Depth、Mask 与反投影：查询侧几何入口

[`sv_align.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/sv_align.py)要求 color、实例 mask、16-bit 毫米 depth 和相机内参。代码把内参缩放到方形 `image_size=448`，反投影 mask 内像素，然后依次过滤：

1. 深度必须大于零并小于 `max_depth=5m`。
2. mask 内深度按四分位距过滤，默认 `1.5 IQR`。
3. 点云按 `32^3` voxel 网格清理，低于两个点的 voxel 被去除。
4. 有效点少于阈值时直接返回 `None`。
5. 最多用 Farthest Point Sampling 选 `512` 个 target points 做 view scoring。

这些不是无关预处理。单目 depth 的全局尺度误差会进入平移和对象 scale；mask 吞入背景会制造稳定但错误的大平面；intrinsics 错误会让三维点产生位置相关畸变。SUFLECA 的鲁棒 matching 可以拒绝部分 outlier，却无法从错误相机模型恢复真实几何。

主表使用在 ScanNet25k 上微调的 metric depth，因此不能将结果直接外推到任意手机照片。仓库零样本 demo 改用 MoGe-2，是工程可运行性路径，不是论文主表的 depth 设置。

## 29. Render Selection：用平均最近邻相似度选视角

对查询点子集 $\hat{\mathcal Q}_I$ 和 CAD 视角集合 $\mathcal V_C$，论文选择：

$$
V^*=\arg\max_{V\in\mathcal V_C}
\frac{1}{|\hat{\mathcal Q}_I|}
\sum_{q\in\hat{\mathcal Q}_I}
\max_{p\in V}\mathcal F_I(q)^\top\mathcal F_V(p).
$$

内层最大值为每个 query point 找该 render 中最相似的 pixel，外层取平均。它不要求一对一，也不强制空间一致，因此适合粗筛而不适合最终 correspondence。FPS 先把 target 压到 $N=512$，使六个视角的矩阵乘法有固定上界。

这个得分有明显偏差：render 中一个高频重复部件可以成为许多 query points 的共同最近邻；query mask 中背景会拉低平均；未见 surface 只能匹配相似但错误的可见 surface。论文 Table IV 也表明，若直接把 render-selection score 用作 NMS 质量分数，准确率降至 `29.2/38.8`。它的职责是选粗视角，不是证明最终 pose 正确。

## 30. 单向 Nearest Neighbor 为什么会制造几何假对应

给定 source feature $F_s\in\mathbb R^{N_s\times d}$ 与 target feature $F_t\in\mathbb R^{N_t\times d}$，最简单匹配是：

$$
j^*(i)=\arg\max_j F_s(i)^\top F_t(j).
$$

这种规则允许多个 source point 指向同一个 target point。椅背的多根竖条、柜门的多个把手、圆桌的多个边缘都可能有近似 feature；一张遮挡图中只剩一个可见部件时，CAD 的许多部件会坍缩到该部件。后续 RANSAC 收到的不是少量随机噪声，而是一组结构化 many-to-one outlier。

Mutual nearest neighbor 要求 $i$ 同时也是 $j^*(i)$ 的反向最近邻，能减少坍缩，却太保守。真实与合成 feature 存在 domain gap，正确匹配未必刚好互为第一名。SUFLECA 在两者之间选择 mutual top-$k$，允许局部候选不确定，再交给几何阶段筛选。

## 31. Mutual Top-k：语义候选与一对一贪心消歧

设 $\mathcal N_k^t(i)$ 为 source descriptor $i$ 在 target 中的 top-$k$，$\mathcal N_k^s(j)$ 为 target descriptor $j$ 在 source 中的 top-$k$。候选对满足：

$$
(i,j)\in\mathcal C
\iff
j\in\mathcal N_k^t(i)
\land
i\in\mathcal N_k^s(j).
$$

论文和默认配置使用 $k=13$。候选按 cosine similarity 从高到低排序，贪心保留 source 与 target 都未被占用的 pair，从而得到一对一集合。若候选超过 $M=256$，公开代码优先按 source 三维位置做 FPS，而不是只取最高分；这样避免所有对应集中在一个高纹理角落。

[`find_correspondences_mknn()`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/geometry.py)用 dense matrix multiplication 计算相似度，再以 `argpartition` 取双向 top-k。其效果依赖 feature 已经归一化，否则点积会混入 feature norm。权重最后归一化供 PROSAC/MAGSAC 使用。

一对一贪心不是全局最优 assignment。它没有求 Hungarian matching，也没有显式最大化全局总相似度；高分错误 pair 可能先占用正确点。作者用后续几何共识补偿，而不是把所有复杂度都放进 feature matching。

## 32. 各向异性尺度为何破坏普通距离一致性

9D 变换为：

$$
\mathbf q_i=R\,\operatorname{diag}(s_x,s_y,s_z)\mathbf p_i+t.
$$

对 pair 做差，平移消失，旋转不改变长度：

$$
\|\mathbf q_i-\mathbf q_j\|^2
=\Delta_{ij}^\top\operatorname{diag}(\mathbf s^2)\Delta_{ij}
=(\mathbf s^2)^\top(\Delta_{ij}\odot\Delta_{ij}),
$$

其中 $\Delta_{ij}=\mathbf p_i-\mathbf p_j$。若 $s_x=s_y=s_z=s$，target/source pair distance ratio 应接近常数 $s$，传统 scale-consistency 足够。若三个轴尺度不同，相同对象中沿 x 的 pair 与沿 z 的 pair 本来就有不同 ratio；直接要求统一 ratio 会误删正确对应。

因此 SUFLECA 不把各向异性缩放当成 RANSAC 之后才估的附加参数。它在 correspondence filtering 前先估一个粗 $\mathbf s$，用该尺度预测每对 source points 应在 target 中相隔多远，然后判断 pair 是否一致。这是方法相对普通 mutual matching 最关键的几何适配。

## 33. 归一化线性系统与 Isotropic 初始化

将 pair 方程除以 source pair 的平方长度：

$$
\phi_{ij}=\mathbf f_{ij}^\top\mathbf s^2,
\qquad
\mathbf f_{ij}=\frac{\Delta_{ij}\odot\Delta_{ij}}
{\|\Delta_{ij}\|^2},
\qquad
\phi_{ij}=\frac{\|\mathbf q_i-\mathbf q_j\|^2}
{\|\Delta_{ij}\|^2}.
$$

$\mathbf 1^\top\mathbf f_{ij}=1$，所以 $\phi_{ij}$ 是三个轴平方尺度的方向加权组合。直接最小二乘会被错误 correspondence 支配，论文先对所有 $\binom M2$ pair 计算 isotropic log-scale：

$$
\psi_{ij}=\frac12\log\phi_{ij},
\qquad
\hat\ell_{\mathrm{iso}}=\operatorname{mode}\{\psi_{ij}\}.
$$

代码用 `60` 个 histogram bins 近似 mode，再以 $\exp(2\hat\ell_{\mathrm{iso}})\mathbf1$ 初始化平方尺度。隐含假设是正确 pair-pairs 在 log ratio 上形成最大密集簇。若 outlier 具有强重复结构，或正确对应太少，histogram mode 也会选错；它只是鲁棒初值，不是统计一致性保证。

## 34. IRLS 与 Shrinkage：在可辨识性不足时回到保守先验

有了初值后，公开实现对 $\mathbf s^2$ 做五轮 Iteratively Reweighted Least Squares。每轮先解带权线性系统，再在 log residual 上以 median absolute deviation 构造 Huber 风格权重。大残差 pair 的影响逐渐下降，尺度被约束为正数。

各向异性尺度的可辨识性取决于 correspondence 是否覆盖三个方向。若所有点近似落在一块桌面上，垂直轴尺度几乎没有证据。论文因此把估计的 log-scale 向 isotropic 初值收缩：

$$
\tilde{\boldsymbol\ell}
=\alpha\hat{\boldsymbol\ell}
+(1-\alpha)\hat\ell_{\mathrm{iso}}\mathbf1,
\qquad
\tilde{\mathbf s}^2=\exp(2\tilde{\boldsymbol\ell}),
$$

默认 $\alpha=0.5$。这不是简单平均尺度值，而是在 log space 平衡相对比例。收缩降低缺轴时的方差，也引入 toward-isotropic bias：真实非常扁或非常长的物体可能被拉回。工程上应监控各轴 point spread 与线性系统条件数，而不是只看最终 scale 是否为正。

## 35. Consensus Matrix：把每个 correspondence 放进一致性图

尺度校正后，任意两个 correspondence 都能互相“投票”。预测的 target 平方距离为：

$$
\tilde d_{ij}^2=(\tilde{\mathbf s}^2)^\top
(\Delta_{ij}\odot\Delta_{ij}).
$$

共识矩阵定义为：

$$
G_{ij}=\mathbf1\!\left[
\left|
\frac{\|\mathbf q_i-\mathbf q_j\|^2-\tilde d_{ij}^2}
{\tilde d_{ij}^2}
\right|<\beta
\right],\qquad G_{ii}=0,
$$

论文设置 $\beta=0.01$。$G$ 可以看作 correspondence graph：节点是候选 pair，边表示两者在当前尺度下几何相容。正确匹配应形成较稠密子图，随机错误匹配连接更少。作者取 $G$ 的 principal eigenvector $\mathbf e$ 作为全局一致性分数，公开实现用 50 次 power iteration 近似。

谱分数优于简单 degree 的直觉是：与许多高质量节点相连，比与许多边缘节点相连更可信。但重复几何可能形成另一个内部一致的错误簇；principal eigenvector 只偏向连接更强的簇，并不知道哪一簇语义正确。

## 36. Relative Gate：为何阈值相对于最大谱分数

最终 mask 为：

$$
\mathbf m=\mathbf1[\mathbf e\geq\delta\max(\mathbf e)],
\qquad \delta=0.05.
$$

使用相对阈值而非绝对阈值，避免矩阵规模和图密度变化时谱向量数值范围漂移。`0.05` 看起来宽松，因为它的目标不是完成最终 outlier rejection，而是删除明显不属于主共识结构的候选，保留足够点给 RANSAC。

公开 `geometric_consensus_mask()` 还有两个降级路径：候选少于 3 个或 `beta<=0` 时全保留；各向异性系统有效 pair 少于 6 个时回退到 isotropic consensus。`sv_align.py` 只有在过滤后至少还剩 7 个 correspondence 时才采用 mask，否则保留原集合。这些 guard 防止过滤器把数据删空，但也意味着某些困难实例实际上没有得到论文所强调的完整几何筛选。

## 37. SupeRANSAC：鲁棒求解而非普通三点随机采样

过滤后的 correspondence 进入 vendored SupeRANSAC。公开配置使用：

- sampler：PROSAC，优先采样 feature score 较高的对应；
- scoring：MAGSAC，以软鲁棒评分替代固定 inlier count；
- confidence：`0.999`；
- min iterations：`500`；
- max iterations：论文效率表为 `300,000`；
- local/final optimization：当前配置关闭。

inlier threshold 不是恒定 11cm。代码先估对象点云以 median 为中心的 96% 半径 $r$，再取：

$$
\tau=\max(0.30r,\ 0.11\ \mathrm m).
$$

因此大对象获得更宽容的阈值，小对象至少使用 11cm。这增强跨对象尺度稳定性，但对很小对象可能过宽，对极大物体又可能接受明显偏移。

Table VI 显示 RANSAC 单项约 `332ms`，占 0.53s 总 alignment 的大头。所谓“sub-second”不是 neural forward 一步完成，仍包含最多 30 万次鲁棒 hypothesis search。降低迭代数能直接换速度，但准确率曲线没有在论文中展开。

## 38. Procrustes Fallback 与尺度合法性检查

若 RANSAC inlier 少于 7 个，或分解出的尺度不合法，代码进入 anisotropic Procrustes fallback。它先尝试在已有 inlier 上拟合；仍不稳定时用全部 correspondence。求解步骤为：中心化 source/target，以线性映射的 SVD 恢复旋转，再逐轴回归对角 scale，最后计算平移。

公开尺度合法性条件是：

```text
all scales > 0
max scale <= 10
min scale >= 0.05
max/min aspect ratio < 2.5
```

这些 hard bounds 防止退化矩阵输出离谱 CAD，但也编码了对象形状 prior。对真实长杆、薄板或单位异常 CAD，`aspect_ratio < 2.5` 可能拒绝本来合理的变换。另一方面，RANSAC 路径如果尺度合法，就直接信任其估计；代码注释也写着 “trust the ransac estimate”。工程部署应把 fallback 原因、inlier 数、尺度条件数和拒绝率写入日志，而不是把所有 `None` 汇总成“未检测到”。

## 39. Alignment Quality：从残差 Jacobian 到信息矩阵

对 inlier correspondence，残差为：

$$
\mathbf r_i=\mathbf q_i-RS\mathbf p_i-t.
$$

参数增量 $\mathbf x=[\delta\boldsymbol\theta,\delta\mathbf t,\delta\mathbf s]$ 有 9 维。令 $\mathbf u_i=RS\mathbf p_i$，论文给出 $3\times9$ Jacobian：

$$
\mathbf J_i=
\begin{bmatrix}
[\mathbf u_i]_\times & -I_3 & -R\operatorname{diag}(\mathbf p_i)
\end{bmatrix}.
$$

若假设各残差是独立同方差高斯噪声，Gauss-Newton 近似的信息矩阵是：

$$
\Lambda=\frac1{\sigma^2}\sum_i\mathbf J_i^\top\mathbf J_i,
\qquad
\sigma^2=\frac{\sum_i\|\mathbf r_i\|^2}{3\hat M-9}.
$$

论文把 $\mathcal S_{\mathrm{fit}}=\log\det(\Lambda)$ 作为 alignment quality。直觉上，更多、空间分布更丰富、残差更小的对应会让参数更可观测，信息体积更大。相比 CAD retrieval 的语义分数，它更接近“这组几何数据能否约束 pose”。

## 40. 信息矩阵分数的实现差异与校准风险

[`compute_information_matrix()`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/geometry.py)确实构造 $\sum J_i^\top J_i$，但参数 `unbiased=False` 默认关闭 $1/\sigma^2$ 归一化；`sv_align.py` 调用时没有改成 `True`。因此当前公开推理的 `logdet_info` 更偏向 correspondence 数量与几何覆盖，不完全等于论文公式中的残差归一化信息矩阵。

另一个差异出现在 NMS。论文文字强调结合 ROCA retrieval score 与 $\mathcal S_{\mathrm{fit}}$；当前 [`evaluation/eval_sv.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/evaluation/eval_sv.py)先在场景内将两个分数转换成 percentile rank，再相乘。源码注释说明 raw score 量纲不一致，直接相乘并不合理。rank fusion 提升相对可比性，却丢失绝对 confidence，跨场景阈值也仍需校准。

$\log\det$ 本身还有三个工程风险：旋转弧度、平移米、尺度无量纲混在同一参数向量；点的坐标单位会改变 determinant；接近奇异时分数对微小数值误差敏感。它适合在固定协议下做排序，不应直接解释成可跨类别、跨相机、跨资产库比较的“成功概率”。

## 41. 源码主链：从 checkpoint 到结果字典

官方仓库的单图对齐主链可以按五个接口阅读：

```text
load_sufleca_model(checkpoint)
  -> extract_sufleca_features(image)
  -> align_single_view(...)
       -> load_scores / load_clean_view
       -> find_correspondences_mknn
       -> geometric_consensus_mask
       -> run_ransac_registration / solve_anisotropic_procrustes
       -> compute_information_matrix
  -> {R, S, t, info, logdet_info, render_view, matched_points}
```

[`load_sufleca_model()`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/featurizer.py)接受公开 alias 或显式 `.pt` 路径；若 CUDA 不可用会退到 CPU。checkpoint 保存 config 与 `dpt_head` 权重，DUNE backbone 通过 Torch Hub 单独加载。这个依赖关系意味着仅有 382MB release checkpoint 仍不足以完全离线运行，还要准备匹配版本的 DUNE 与 Python/CUDA 依赖。

`align_single_view()` 的输出合同值得保留。除 $R,S,t$ 外，它返回被选 render、source/target inlier points、$9\times9$ information matrix 和 `logdet_info`。这些字段使上层系统能做可视化、NMS 与诊断，而不是只得到一个不可解释的 4x4 matrix。失败则返回 `None`，并仅在 debug logger 中记录原因。

## 42. Feature Extractor：冻结不是一句配置，而是代码约束

[`extractor.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/model/extractor.py)的 `HookedViTEncoderWrapper` 对指定 transformer blocks 注册 forward hook，删除一个 class token 加若干 register tokens，再将所选层 token 交给 DPT。初始化时所有 backbone parameter 都设为 `requires_grad=False`；重写的 `train()` 还强制 ViT 保持 eval mode。这避免调用上层 `model.train()` 时意外打开 dropout 或更新 backbone 状态。

`extract_sufleca_features()` 先按 ImageNet mean/std 预处理，模型输出以通道维归一化；若插值到目标分辨率，插值后再次归一化；最终 shape 从 `(B,C,H,W)` 变成 `(B,H,W,C)`，便于用布尔 mask 直接索引像素 feature。

这段代码也澄清了两个边界。第一，公开 checkpoint 只加载 `dpt_head`，不是一份包含 DUNE 的完全自包含模型。第二，`TorchvisionImageProcessor` 是仓库当前实现，不等同于论文训练时完整 augmentation。静态阅读可以验证推理 shape 和归一化，不能反推出训练数据增强。

## 43. Geometry 模块：论文算法如何被拆成可测试函数

[`geometry.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/geometry.py)把几何路径拆成相对独立的纯 NumPy 函数：

- `voxel_clean_indices()`：删除稀疏离群 voxel。
- `find_correspondences_mknn()`：FPS、双向 top-k、贪心一对一、correspondence cap。
- `_axis_aligned_consensus_score()`：histogram、IRLS、shrinkage、共识矩阵、power iteration。
- `geometric_consensus_mask()`：按相对谱分数生成 mask。
- `run_ransac_registration()`：配置并调用 vendored SupeRANSAC。
- `solve_anisotropic_procrustes()`：fallback closed-form fit。
- `compute_information_matrix()`：构造 Jacobian normal matrix。

代码没有 CUDA 自定义 kernel、GRU 或 learned matcher。计算热点是 dense feature similarity、$M^2$ pairwise geometry 和 SupeRANSAC C++ binding。SupeRANSAC 以第三方目录 vendored，采用 MIT license；SUFLECA 仓库整体是 Apache-2.0。复现环境如果误加载系统中另一版 `pysuperansac`，代码会捕获四参数 API 的 `TypeError` 并提示重新安装 vendored binding。

这种分层也有测试价值：可以用合成点云分别验证 scale recovery、outlier filtering 与 information matrix，而不必先运行视觉模型。仓库公开的是 inference system，几何函数反而比训练部分更完整。

## 44. Render Cache：缓存命中也是正确性问题

[`render_cache.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/render_cache.py)采用两级布局：

```text
<pool>/<synset>/<model>/precomputed/
  scores.npz          # view ids, per-view FPS features, metadata
  clean_00.npz        # points, features, metadata
  ...
  clean_05.npz
```

`scores.npz` 一次载入六个 view 的轻量评分 feature；选出 view 后，只读取对应 `clean_XX.npz`。这是减少磁盘 I/O 与内存峰值的关键。meta 至少验证 `image_size` 和 checkpoint 的绝对路径，不匹配就返回 cache miss，而不是悄悄用错 descriptor。

绝对路径绑定也会降低可移植性：checkpoint 搬到另一台机器后，即使内容相同，meta 仍可能不匹配。生产实现更适合用模型内容 hash、config digest 与 schema version，而不是文件绝对路径。另一个风险是 `.npz` 并没有事务式写入；并发生成或中断时需要临时文件加原子 rename，避免读到半写 cache。

仓库脚本还允许删除 raw renders，只保留预计算数组。这节省磁盘，但会降低审计能力。若 alignment 失败，保留 render PNG 能快速判断是 canonical view、point map 还是 feature 出错；正式管线应在空间成本与可诊断性之间明确选择。

## 45. Evaluation：主表不是把每个实例独立算对错

[`evaluation/eval_sv.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/evaluation/eval_sv.py)先读取 ROCA detection/retrieval，按每个 scene/frame/instance slot 运行 alignment，写出预测 CSV；随后把 pose 转到 world space 做 NMS，再按 Scan2CAD 的类别真值数量做 count cap，最后与 scene-level GT matching。

NMS 的宽松冲突阈值是 `0.4m / 60° / 0.6 scale ratio`，它用于删除重复预测，不是 Table II 的正确性阈值。正确性仍要求 translation 小于 `20cm`、rotation 小于 `20°`、scale 小于 `20%`。混淆两组阈值会严重误读 benchmark。

count cap 使用每个场景的 GT category counts，把该类预测截到真值数量。它是 Scan2CAD protocol 的一部分，有助于和既有工作比较，却不是部署可用的步骤，因为真实系统不知道场景里每类有几个对象。DiffCAD split 不使用同样的 NMS 抑制，因此更直接暴露 alignment 错误，论文也将其作为严苛补充协议。

评测脚本还提供 category filter、skip/success 计数和 score mode。若只报告最终 accuracy 而不报告 attempted、succeeded、skipped，可能把“算法拒绝困难实例”与“输出错误 pose”混在一起。生产验收应把 coverage 单独列出。

## 46. 当前评分实现：Percentile Rank Fusion 不是原始分数相乘

当前 evaluator 将 detection confidence 与 ROCA retrieval confidence 相乘为 semantic score，再在每个场景内分别对 semantic score 与 `logdet_info` 计算 percentile rank：

$$
r_{\mathrm{sem}},r_{\mathrm{geo}}\in(0,1),
\qquad
s_{\mathrm{mixed}}=r_{\mathrm{sem}}\,r_{\mathrm{geo}}.
$$

只有两个原分数都有限时才保留 mixed score。这样做的原因在源码注释中写得很清楚：`logdet` 只是单调排序键，其符号和尺度并不与 ROCA confidence 可比。rank fusion 不要求拟合两个量纲的 calibration。

它也有代价。场景中只有少量候选时 percentile 很离散；场景内 rank 无法比较跨场景置信度；两个极差但相对最高的候选仍会得到高 rank。更稳健的生产方案应在独立验证集上校准 $P(\mathrm{correct}\mid s_{\mathrm{sem}},s_{\mathrm{geo}},n_{\mathrm{inlier}},\kappa)$，其中 $\kappa$ 可表示信息矩阵条件数，并保留 reject option。

这项实现差异应如实记录，而不是简单判断“代码与论文矛盾”。论文提供方法公式和评分直觉，release code 体现作者为量纲问题做的后续工程修正。复现实验时应固定 commit 和 scoring mode，避免不同代码时点的数值不可比。

## 47. Zero-shot Demo：完整上游由四个额外模型拼起来

仓库的 [`zero_shot.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/zero_shot.py)与 [`demo_utils.py`](https://github.com/snt-arg/SUFLECA/blob/9df942e22601648e8165b0cffd7432ad4dc50d33/sufleca/demo_utils.py)构造了一条更接近实际单图输入的路径：

1. Grounding DINO 根据文字提示检测二维 box。
2. SAM2.1 以 box 为 prompt 生成实例 mask。
3. MoGe-2 预测 metric depth、point map 和内参。
4. DINOv3 在标签允许的 ShapeNet synset 中做 coarse-to-fine CAD retrieval。
5. SUFLECA 对检索 CAD 做 9D alignment。

DINOv3 retrieval 先用 mask 内 patch mean embedding 和所有 template 的 pooled descriptor 做 coarse top-k，再读取候选的 dense patch feature，以 query-to-template 或 symmetric max-mean similarity重排。缓存上限为 `6000` 个 dense template；模型 ID `facebook/dinov3-vitl16-pretrain-lvd1689m` 是 gated 资源，需要单独接受许可并登录 Hugging Face。

这个 demo 很有工程价值，因为它暴露了完整系统的依赖和失败点。它也扩大了“能运行”与“可复现主表”之间的距离：每个上游模型都有版本、许可、阈值与缓存，端到端 latency 和准确率都未在 Table VI 中报告。

## 48. Demo 不等于论文主评测，更不等于端到端零样本已解决

主实验使用 ROCA 的 supervised CAD retrieval，实例 retrieval accuracy 为 `34.3%`；零样本 GroundedSAM + OSCAR 的 retrieval accuracy 只有 `3.8%`。仓库 demo 后来改成 DINOv3 coarse-to-fine retrieval，但论文没有报告这条路径在 ScanNet25k 的完整指标。因此不能用 demo 中几张成功截图覆盖 Table V 暴露的 retrieval bottleneck。

四个判断必须分别写：

- SUFLECA alignment 是否在给定 candidate CAD、mask、depth 时准确；
- detector/segmenter 是否找到了正确对象区域；
- retriever 是否选择了几何上足够接近的 CAD；
- depth 与 intrinsics 是否让 target point cloud 保持 metric consistency。

论文的强证据集中在第一项。Table V 对第二、三项做了上游敏感性分析，demo 证明第四方模型可以接入，却没有形成统一端到端 benchmark。把“zero-shot alignment”扩写成“只需一张图就能零样本可靠找到并对齐任意 CAD”会越过论文证据。

## 49. 实验协议：三个评测视角回答三个不同问题

论文没有用一个数字覆盖所有主张，而是组合三套协议：

- **ScanNet25k NMS**：九个室内类别，使用 ROCA box/CAD、SAM2 mask、目标域微调 depth；回答给定现实上游时，alignment 能否提高完整 scene 评测。
- **DiffCAD split**：ScanNet25k 中六类对象，不做 NMS，使用未微调 depth；回答错误 alignment 不被场景级抑制时，单实例结果是否仍可靠。
- **CO3D adaptation**：600 张对象中心图像，包含 seen/unseen categories、模拟遮挡和不精确 CAD；回答 feature 是否能泛化到训练弱监督未覆盖的类别。

ScanNet25k 的正确条件是三个误差同时通过：

$$
e_t<0.20\ \mathrm m,
\qquad e_R<20^\circ,
\qquad e_s<20\%.
$$

其中旋转误差通常按相对旋转角，尺度误差按各轴相对误差协议计算。category average 先对九类分别算准确率再平均，避免 chair 等多实例类别支配；instance average 对全部实例平均，更接近数据集总体命中率。二者都应报告，因为高频类别和低频类别会产生不同叙事。

所有实验在单台带 NVIDIA RTX 5090 的 workstation 上执行。论文没有给出 CPU、RAM、CUDA、驱动、精确软件版本或多次重复方差；因此性能表是该环境的测量，不是硬件无关复杂度证明。

## 50. Table II：ScanNet25k 与 DiffCAD 主结果

![SUFLECA ScanNet25k 与 DiffCAD 结果](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-table02-scannet25k-results.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Table II, CC BY 4.0.*

下面转录总体数字；原表还含九类/六类的逐类结果：

| Method | 监督口径 | ScanNet25k cat. | ScanNet25k inst. | DiffCAD cat. | DiffCAD inst. |
| --- | --- | ---: | ---: | ---: | ---: |
| ROCA | full | 18.4 | 25.0 | - | - |
| MultiObj-SPARC | full | 30.3 | 40.3 | - | - |
| CosCAD | full | 27.4 | 33.2 | - | - |
| DINOv3-L | none | 11.8 | 15.1 | 13.6 | 16.0 |
| DUNE-B | none | 11.4 | 16.6 | 14.0 | 17.0 |
| Diorama | none | 8.7 | 11.3 | 11.9 | 18.4 |
| FoundationPose 9D | weak / zero-shot alignment | 19.2 | 25.7 | - | - |
| ZeroCAD | weak / zero-shot alignment | 23.1 | 30.1 | 15.9 | 20.9 |
| **SUFLECA** | weak / zero-shot alignment | **33.4** | **42.3** | 36.1 | 44.8 |
| SUFLECA-S | weak / zero-shot alignment | 30.6 | 39.5 | 33.6 | 41.7 |
| SUFLECA-blend | weak / zero-shot alignment | 30.0 | 38.5 | **36.6** | **44.9** |

在 NMS 协议下，SUFLECA 相比 ZeroCAD 提高 `10.3/12.2` 个百分点；在 DiffCAD split 提高 `20.2/23.9` 个百分点。不能把后者写成“相对提升 20.2%”，那是绝对百分点。若换算相对提升，category 约为 $20.2/15.9\approx127\%$，instance 约为 $23.9/20.9\approx114\%$。

逐类表显示 SUFLECA 在 bed `38.6`、display `31.4`、table `30.7` 等类突出，但 bathtub `23.3` 并没有超过监督方法的最高值 `25.8`，bin `27.6` 低于 MultiObj-SPARC 的 `44.8`。所谓“超过监督方法”是总体平均与多数列的 benchmark 结论，不是每类别、每场景都胜出。

## 51. “超过监督方法”的正确边界

SUFLECA 的 ScanNet25k `33.4/42.3` 确实高于 MultiObj-SPARC 的 `30.3/40.3`，这是论文最强结果之一。但比较仍包含共同上游和协议选择：所有相关方法按既有设置使用 ROCA box/CAD、SAM2 mask 与相同或对应 depth；NMS 与 count cap 会抑制重复错误；SUFLECA 的 feature 在 11 个非 ScanNet 数据源中仍见过大量室内对象。

监督标签类型也不同。MultiObj-SPARC 使用 ScanNet 的 9D pose supervision；SUFLECA 使用更大规模的跨数据集 NOC weak supervision。它说明“多域稠密几何监督 + 鲁棒配准”在该协议下比“目标域 pose supervision + 既有模型”有效，不说明 weak supervision 天然优于 full supervision，更不说明训练标注成本更低。SUFLECA 的 NOC derivation 同样依赖 pose、CAD、depth、SAM2 和复杂清洗。

此外，论文没有统一重跑所有监督 baseline；部分数字来自原论文。软件、depth、mask 与检测版本可能不同。最稳妥的判断是：在公开的 ScanNet25k NMS 表中，SUFLECA 达到最高总体均值，并展示出强零样本 alignment transfer；跨系统绝对排名仍应等待统一实现和独立复测。

## 52. DiffCAD Split：移除 NMS 后为什么差距扩大

DiffCAD split 只覆盖 bed、bookshelf、cabinet、chair、sofa、table 六类，不使用 NMS，并采用未在 ScanNet25k 上微调的 depth。Table II 中 SUFLECA 为 `36.1/44.8`，blend 为 `36.6/44.9`，ZeroCAD 为 `15.9/20.9`。错误 pose 不再靠场景级 score 与重复抑制消失，因而 correspondence 和 registration 本身的差异更直接。

论文还列出 DiffCAD (GT) oracle `35.8/41.9`：该方法用 ground-truth pose 从八个 hypothesis 中选择最佳。SUFLECA 在表中超过该 oracle，不能解释为“比真值更准”；oracle 只是在 DiffCAD 自己生成的有限八个候选里用 GT 做选择，不是直接输出 ground-truth transform。候选生成能力不足时，oracle 上界也可能低于另一套方法。

无 NMS 并不等于完全端到端。输入 CAD 仍来自指定检索路径，mask/depth 仍由外部模块提供。这个 split 更强地评估单实例 alignment，但仍不是 raw image 到 CAD pose 的完整系统测量。

## 53. Table III：CO3D 的已见类别与未见类别

![SUFLECA CO3D 已见与未见类别结果](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-table03-co3d-seen-unseen.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Table III, CC BY 4.0.*

| Method | Seen 3D-IoU ↑ | Seen ICP-Rot ↓ | Seen ADD-S ↓ | Seen @0.1 ↑ | Unseen 3D-IoU ↑ | Unseen ICP-Rot ↓ | Unseen ADD-S ↓ | Unseen @0.1 ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv3-L | 36.21 | 27.32 | 6.54 | 71.50 | 8.77 | 40.44 | 10.31 | 46.50 |
| DUNE-B | 23.40 | 35.81 | 8.48 | 63.50 | 4.46 | 44.44 | 11.84 | 34.50 |
| Diorama | 39.41 | 21.06 | 5.79 | 82.00 | 39.12 | 14.54 | 6.33 | **81.50** |
| SUFLECA | 62.63 | **10.24** | 3.80 | 84.00 | 48.89 | 16.64 | 5.86 | 78.75 |
| SUFLECA-S | 62.50 | 10.31 | **3.71** | 85.50 | 46.42 | 16.44 | 6.49 | 74.00 |
| SUFLECA-blend | **62.71** | 10.86 | 3.77 | **86.00** | **49.50** | **12.93** | **5.56** | 81.25 |

seen 是 chair、couch，CAD retrieval accuracy `77.5%`；unseen 是 toaster、hairdryer、microwave、suitcase，retrieval accuracy `52.5%`。每类先对实例取 median，再对类别 median 求 mean，因此表并非普通全样本平均。3D-IoU 与 ADD-S 对对称性和 shape mismatch 的敏感性也不同。

blend 在 unseen 的 3D-IoU、rotation、ADD-S 最好，支持拼回原 DUNE feature 有助于未见语义类别；但 Diorama 的 unseen ADD-S@0.1 为 `81.50`，略高于 blend `81.25`。不能把“blend 未见类别最好”写成所有指标都第一。

CO3D 没有所需的统一 GT CAD pose，论文将 COLMAP point cloud 用 monocular depth 选尺度、以 SAM3D 生成 category-level CAD pool，再用 OSCAR retrieval，并人为放置矩形 occluder。这个适配协议有价值，却也引入 pseudo ground truth、canonical orientation 与 retrieval 的额外不确定性。

## 54. Fig. 4：不精确 CAD、遮挡与朝向歧义

![SUFLECA 在 CO3D 上的定性对齐比较](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-fig04-qualitative-alignment.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Fig. 4, CC BY 4.0.*

Fig. 4 的顶格是所有方法共享的 masked input 与检索 CAD，后续列展示不同 feature/solver 的 overlay。它重点不是纹理还原，而是前后、左右、尺度和中心是否与可见对象一致。CO3D 的 candidate CAD 并非 GT instance，shape 差异使“完美轮廓重合”本来就不可达。

SUFLECA 的优势主要体现在几何轴与整体比例更稳定：弱纹理、遮挡后仍能用 NOC-aligned feature找到相似部位；mutual matching 避免同一可见 patch 吸收大量 CAD 点；尺度共识减少错误 correspondences 进入 RANSAC。

图也暴露不可解歧义。完全对称或只见局部的对象可能有多个等价 pose；canonical orientation 若在检索 CAD 与 benchmark 中不一致，视觉上合理的旋转仍会被算错。单个 qualitative montage 不能量化失败频率，也不能证明所有成功例是随机抽取；应与 Table III 的聚合指标一起读。

## 55. Table IV：数据、匹配与评分的逐项消融

![SUFLECA 消融实验](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-table04-ablation.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Table IV, CC BY 4.0.*

| Variant | cat. | inst. | 相对完整模型变化 |
| --- | ---: | ---: | --- |
| SUFLECA | **33.4** | **42.3** | baseline |
| 训练包含 ScanNet | 34.4 | 42.8 | +1.0 / +0.5 |
| nearest neighbor | 26.3 | 34.2 | -7.1 / -8.1 |
| mutual nearest neighbor | 30.7 | 39.4 | -2.7 / -2.9 |
| mutual top-k，无几何过滤 | 32.0 | 41.0 | -1.4 / -1.3 |
| NMS 用 ROCA retrieval score | 30.4 | 39.3 | -3.0 / -3.0 |
| NMS 用 render-selection score | 29.2 | 38.8 | -4.2 / -3.5 |

消融最有说服力的不是 ScanNet inclusion 只增加 `1.0/0.5`，而是 matching ladder 连续改善：NN -> mutual NN -> mutual top-k -> 加几何过滤。它把 feature 与 solver 的贡献分开，说明单纯扩大数据不足以替代对应约束。

但每行只改变一个高层模块，并没有报告 seed 方差、$k$、$\alpha$、$\beta$、$\delta$、render view 数或 RANSAC iterations 的敏感性。`1.4/1.3` 的几何过滤增益是否稳定，需要重复实验或置信区间。NMS score 消融还混入 evaluator 的当前 rank fusion 实现差异，复现时必须固定相同代码版本。

## 56. Table V：CAD Retrieval 才是完整系统的最大瓶颈

![SUFLECA 对 CAD retrieval 质量的鲁棒性](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-table05-cad-retrieval-robustness.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Table V, CC BY 4.0.*

| Detection / retrieval | Retrieval acc. | Alignment cat. | Alignment inst. |
| --- | ---: | ---: | ---: |
| Scan2CAD annotations (GT) | 100.0 | 43.0 | 54.2 |
| ROCA | 34.3 | 33.4 | 42.3 |
| GroundedSAM + OSCAR | 3.8 | 22.5 | 33.1 |

从 GT CAD 到 ROCA，retrieval accuracy 降 `65.7` 个百分点，alignment 只降 `9.6/11.9`；从 ROCA 到 fully zero-shot retrieval，retrieval 再降 `30.5` 点，alignment 降 `10.9/9.2`。这说明 SUFLECA 能把“不完全相同但几何近似”的 CAD 对齐好，alignment metric 不要求 retrieval ID 完全正确。

同一张表也否定了端到端问题已经解决。3.8% 的 instance retrieval 意味着绝大多数 zero-shot candidate 不是 GT CAD；即使 alignment 仍有 `22.5/33.1`，可编辑场景里的资产身份、部件拓扑与语义可能不对。对数字孪生或机器人抓取，近似形状并不总能替代正确实例。

## 57. 完整 Zero-shot CAD Fitting 的真实差距

可以把端到端成功粗略分解为：

$$
P(\mathrm{success})=
P(D)\,P(M\mid D)\,P(Z\mid M)\,P(C\mid I)\,
P(A\mid C,M,Z),
$$

其中 $D$ 是检测，$M$ 是 mask，$Z$ 是 metric depth，$C$ 是 candidate CAD，$A$ 是 alignment。实际事件不独立，公式只是提醒每一项都可能成为瓶颈。SUFLECA 主要提高最后一项，并通过不精确 CAD 实验证明它对 $C$ 的错误有一定容忍。

完整系统仍需要：开放词汇类别到 CAD taxonomy 的映射、多个 CAD hypothesis、对称性集合、深度尺度校准、失败拒绝和 scene-level collision/support reasoning。Table V 只更换检索输入，没有同时测 detector、mask 与 depth 在开放世界数据上的联合失败率。

因此最合理的产品定位是“高质量 CAD alignment backend”，而不是“一键生成完整场景”。它可以嵌入 Diorama 类场景装配系统，或作为 AR placement 的 pose refinement，但上游资产搜索和下游 scene consistency 仍要单独设计。

## 58. Table VI：效率来自紧凑 Feature，也受 RANSAC 支配

![SUFLECA 单实例对齐效率](/images/blog/sufleca-cad-to-image-alignment-feature-learning/sufleca-table06-efficiency.webp)

*Source: Ejaz et al., arXiv:2607.15058v1, Table VI, CC BY 4.0.*

| Method | dim | Peak VRAM (MB) | 每实例时间 (s) |
| --- | ---: | ---: | ---: |
| DINOv3-L | 1024 | 3394 | 0.94 |
| DUNE-B | 768 | 2696 | 0.82 |
| Diorama | 1024 | 3426 | 2.39 |
| DiffCAD | - | 3146 | 13.1 |
| FoundationPose | - | 8544 | 0.61 |
| ZeroCAD proxy，无 refine | 2048 | 4988 | 1.30 |
| ZeroCAD proxy，含 refine | 2048 | 5158 | 3.77 |
| **SUFLECA** | 384 | **2178** | **0.53** |
| **SUFLECA-S** | 256 | **1556** | **0.49** |
| SUFLECA-blend | 1152 | 3972 | 0.76 |

| SUFLECA component | 时间 (ms) |
| --- | ---: |
| Featurizer，每图一次 | 86.6 |
| Target render selection | 43.1 |
| Correspondence + filtering | 68.7 |
| SupeRANSAC，300K iterations | 332.0 |

四项和为 `530.4ms`，与 0.53s 对齐。主模型相对 DINOv3-L 的 descriptor 维数小 $1024/384\approx2.67$ 倍，小模型小 4 倍；相对 ZeroCAD proxy 的 2048 维则小 5.33 或 8 倍。compact feature 确实降低 matching matrix 和 cache 成本，但最终总延迟只从 0.53 降到 0.49，因为 RANSAC 占约 $332/530.4\approx62.6\%$。

## 59. 0.53 秒不包含什么

论文明确排除了 detection、segmentation、depth estimation 和 CAD retrieval，理由是比较方法共享这些阶段。还排除了 render-pool preparation、模型冷启动、checkpoint 下载、DUNE Torch Hub 加载和多候选 CAD 的重复 alignment。因此 0.53 秒是 warmed-up、单实例、给定上游与 cache 的 alignment latency。

若一帧有 10 个对象，featurizer 可以全图复用一次，但 render selection、correspondence 与 RANSAC 通常按实例运行；若每对象保留 5 个 CAD hypothesis，RANSAC 次数又会放大。端到端延迟应测：

$$
T_{\mathrm{e2e}}=T_{\mathrm{detect}}+T_{\mathrm{segment}}+T_{\mathrm{depth}}
+T_{\mathrm{retrieval}}+T_{\mathrm{feature}}
+\sum_{o=1}^{O}\sum_{h=1}^{H_o}T_{\mathrm{align}}^{(o,h)}.
$$

实际可以并行多个 hypothesis、批量 feature extraction，或按 retrieval confidence 提前截断；这些是系统优化空间，不在论文计时证明内。RTX 5090 的结果也不能直接外推到移动 GPU、数据中心卡或 CPU。

## 60. ZeroCAD Proxy：只能视为作者实现的延迟下界

ZeroCAD 没有公开代码，作者实现了仅用于 runtime 的 proxy，并分别报告无 refine `1.30s`、含 refine `3.77s`。论文明确称其为 true runtime 的 lower bound。原因是 proxy 可能省略原实现的数据搬运、预处理、特定 refinement 细节和工程开销；即使算法描述相同，也无法保证 kernel、版本与优化程度一致。

所以 Table VI 能支持“SUFLECA 自己在给定硬件上约 0.53s、内存 2178MB”以及“作者的统一环境中 compact feature 较省资源”。它不能作为 ZeroCAD 官方速度记录，也不能据此宣称 SUFLECA 在所有环境下精确快 `7.1x`。对 closed-source baseline 的公平做法是同时报告 proxy 假设、公开命令、warm-up、重复次数和原作者可验证结果；论文只满足其中一部分。

## 61. 与 Diorama 的关系：一个是对象级 Pose 模块，一个是场景装配系统

[Diorama](/blog/diorama-zero-shot-single-view-3d-scene-modeling-paper-reading/)从单图解析对象、建筑结构和支撑关系，检索 CAD，估计 9-DoF pose，再做分阶段 layout optimization。SUFLECA 聚焦其中最窄也最难的一段：给定对象 crop、mask、depth 和 candidate CAD，如何形成可靠 correspondence 并恢复 9D transform。

二者技术上高度互补：

| 维度 | Diorama | SUFLECA |
| --- | --- | --- |
| 输入范围 | 完整单图场景 | 单对象 RGB-D 区域 + candidate CAD |
| 上游感知 | GPT-4o、OWLv2、SAM、Metric3D | 主评测复用 ROCA/SAM2/depth；demo 可接 Grounding DINO |
| CAD retrieval | 文本筛选 + DuoDuoCLIP | 主评测由 ROCA；demo 用 DINOv3 coarse-to-fine |
| Pose | DINOv2 correspondence + Umeyama/RANSAC | NOC-aligned DPT + mutual top-k + anisotropic consensus + SupeRANSAC |
| 场景约束 | support/collision/relative layout optimization | 无 scene-level constraint |
| 主要输出 | 可编辑 CAD scene | 单个 CAD 的 $R,S,t$ 与 confidence |

如果把 SUFLECA 替换进 Diorama pose stage，潜在收益是更紧凑的 feature、显式各向异性尺度筛选和更强 inexact-CAD robustness；但接口并非直接替换。Diorama 的 depth、render pool、CAD 坐标系、pose convention 与 score 都需适配，scene optimization 还要处理 SUFLECA 的多个 hypothesis 和 uncertainty。

## 62. 与本站其他 3D 文章的关系

SUFLECA 位于“识别已有对象并放置已有 CAD”的路线，和生成式资产/场景模型的目标不同：

- [AssetGen](/blog/assetgen-deployable-3d-asset-generation-paper-reading/)从单图生成可部署 mesh、UV、normal 与 texture；它解决资产不存在时的 assetization，SUFLECA 假设 candidate CAD 已存在。
- [TRELLIS.2](/blog/trellis2-native-structured-latents-3d-generation-paper-reading/)设计 O-Voxel 与大规模 flow model 生成高分辨率 3D 资产；SUFLECA 不生成几何，只估 CAD pose。
- [SF3D](/blog/sf3d-stable-fast-3d-mesh-reconstruction-paper-reading/)强调 0.5 秒 feed-forward mesh reconstruction、UV 和 delighting；它输出新 mesh，SUFLECA 输出已检索 mesh 的变换。
- [TripoSG](/blog/triposg-high-fidelity-3d-shape-synthesis-paper-reading/)用 Rectified Flow 和 SDF VAE 做 shape synthesis；SUFLECA 的随机性来自上游 candidate，而 alignment 本身是确定特征 + 鲁棒估计。
- [Layout2Scene](/blog/layout2scene-semantic-layout-guided-scene-generation-paper-reading/)从 3D semantic layout 优化 Gaussian/polygon scene；SUFLECA 可以为布局中的现有 CAD 提供观测 pose，但没有 diffusion prior。
- [CompoNeRF](/blog/componerf-text-guided-compositional-nerf-paper-reading/)组合多个对象 NeRF，[Blended-NeRF](/blog/blended-nerf-zero-shot-object-generation-blending-paper-reading/)编辑已有 NeRF ROI；两者的可编辑单元是 neural field，SUFLECA 的单元是显式 CAD。

选择路径取决于输出 contract。若目标是准确复用资产库、保留零件语义和低成本编辑，CAD alignment 更合适；若照片中对象没有合适 CAD，生成式 mesh 更有覆盖；若目标只是新视角渲染，NeRF/3DGS 又可能不需要资产级 topology。把所有方法按一张视觉质量图排序会忽略这些根本差异。

## 63. 工程复现清单：公开推理能做什么，缺口在哪里

官方 README 给出了推理与评估入口，但完整复现至少需要以下资产：

### 环境与模型

- Linux、兼容的 Python/PyTorch/CUDA、C++ build toolchain。
- vendored SupeRANSAC 编译成功，Python binding 指向仓库版本。
- DUNE-B 或 DUNE-S 通过 Torch Hub 可访问。
- release 中约 `382MB` 的 checkpoint archive，并核对三个 alias。
- 若跑 demo，还需要 Grounding DINO、SAM2.1、MoGe-2 和 gated DINOv3。

### CAD 与缓存

- ShapeNetCore.v2 或同坐标约定的 CAD pool，并处理其单独许可。
- 论文九类 taxonomy 与 synset mapping。
- 每个 CAD 的 canonical center、六视角 RGB/point map。
- 针对 `sufleca`、`sufleca-small` 或 `sufleca-wo-scannet` 单独预计算 render cache。
- 保存 cache metadata、生成脚本 commit、输入资产 hash 和失败列表。

### ScanNet25k 评测

- release 中约 `1.17GB` 的评测辅助资源，外加 ScanNet/Scan2CAD 相应授权数据。
- ROCA detection 与 retrieval 结果、SAM2 instance mask、论文口径 metric depth。
- 明确选择 NMS protocol 或 DiffCAD split，不混合 count cap。
- 评测 CSV 保留 `object_score`、`roca_score`、pose、category、model ID 和 skip reason。

### 论文不可复现部分

- 12 数据集下载、统一坐标、授权和清洗脚本。
- synthetic counterpart 渲染、相似 CAD 替换规则。
- NOC derivation/verification 的完整阈值与失败统计。
- binned NOC head、optimizer、sampler、augmentation 和训练入口。
- 论文全部 baseline 的统一运行环境与 ZeroCAD 真正实现。

因此本文的“源码精读”只验证公开 inference/evaluation 的逻辑，不声称完成训练复现。官方明确表示目前不计划开放 training 与 data-preparation code，这是可复现性最重要的限制之一。

## 64. 工程落地清单：把对齐器变成可靠服务

### 输入合同

- 统一相机坐标系、右手/左手系、米制单位和 CAD canonical frame。
- 校验内参是否与 resize/crop 后图像一致，禁止静默使用默认 focal length。
- 对 mask 记录来源、score、面积、边界质量和与 detection box 的一致性。
- 对 depth 记录模型版本、全局 scale、有效率、IQR 和对象内 discontinuity。

### Candidate 管理

- 每个对象保留多个 CAD hypothesis，不要只保留 retrieval top-1。
- CAD metadata 至少含 synset、实际尺寸范围、对称群、canonical front/up、许可证和资产版本。
- render cache 用内容 hash 绑定 checkpoint、分辨率、renderer 和 CAD 版本。
- 针对对称物体在评估和下游规划中使用等价 pose 集，而不是强行给唯一 yaw。

### Alignment 质量

- 监控候选数、mutual top-k 数、几何 gate 保留率、RANSAC inlier 数和 fallback 比例。
- 保存 $\log\det\Lambda$、条件数、residual、尺度与对象尺寸，而不是只输出一个 score。
- 用独立验证集做 score calibration，给出 success probability 和 reject threshold。
- 发现 scale 越界、点云共面或 view-score margin 太小时保留多解或请求人工确认。

### 性能与故障恢复

- 全图 feature 只计算一次，按实例复用；多个 CAD hypothesis 批量做 render scoring。
- 设置 RANSAC time budget 与 early termination，而不是无条件 300K iterations。
- cache miss、模型下载失败、空 mask、无 depth、少于 7 对应分别返回结构化错误。
- 记录端到端 p50/p95/p99、每阶段 GPU/CPU 时间、队列等待和 cache hit rate。

### 下游验收

- 可视化 projected CAD、correspondence、inlier/outlier 和三轴尺度。
- 场景级检查 collision、support、floor penetration、wall attachment 和对象尺寸 prior。
- 机器人任务需额外验证抓取面、动态遮挡和相机外参；AR 任务需验证 temporal jitter。
- 保存原始输入与模型版本，确保错误 pose 可以审计和回放。

## 65. 局限性与批判

### 1. Alignment 的强结果依赖强上游

主表的 ROCA box/CAD、SAM2 mask 和目标域微调 metric depth 不是次要实现细节。Table V 已显示 retrieval 从 34.3% 降到 3.8% 后总体准确率显著下降。论文把模块边界说得较清楚，但标题和 teaser 容易让读者误解成端到端 zero-shot CAD fitting。

### 2. 训练数据大，但不可独立重建

12 个数据源带来规模与覆盖，也带来许可证、坐标约定、annotation 噪声和 duplicate risk。训练/数据代码不开放，外部研究者无法确认每个过滤阈值、采样比例、random seed 和 NOC overlay 误差分布。开放 checkpoint 缓解了使用门槛，没有补齐训练可复现性。

### 3. 深度误差会同时污染 scale 与 translation

单目 metric depth 不是纯相对 cue。全局尺度偏差会被对象 $S$ 吸收，空间不均匀偏差会扭曲点云并破坏 pairwise consensus。ScanNet 微调 depth 与真实开放场景之间的 domain shift 没有被完整评估。

### 4. 对称性与 canonical frame 仍是语义问题

几何匹配可以发现一个稳定 transform，却不能决定圆桌的“正面”，也不能解决两个数据集对微波炉 front/up 定义不同。CO3D 使用的 SAM3D mesh 还有 inconsistent canonical orientation。若 benchmark 未按对称群计算误差，合理 pose 可能被惩罚；若下游需要门把手方向，几何对称又不代表功能对称。

### 5. Mutual top-k 和共识图仍有硬阈值

$k=13$、$M=256$、$\alpha=0.5$、$\beta=0.01$、$\delta=0.05$ 在所有实验固定，说明方法工程上简洁；论文却没有 sensitivity curve。重复结构可能形成自洽错误簇，硬 gate 会不可逆删除 candidate，RANSAC 无法恢复被删的真对应。

### 6. 信息矩阵不是已校准 uncertainty

论文的 Fisher/Gauss-Newton 直觉建立在局部线性、独立同方差残差上。feature correspondence 具有相关性，RANSAC inlier 是选择后的样本，旋转/平移/尺度单位不同。公开代码还默认不除 $\sigma^2$。因此 $\log\det\Lambda$ 更像 ranking heuristic，不是严格 posterior covariance。

### 7. 效率表只覆盖 alignment

0.49/0.53 秒没有 detection、mask、depth、retrieval 和离线 render pool；多对象、多 hypothesis 和冷启动成本未测。RANSAC 占 62.6%，说明 feature 再压缩的收益会迅速碰到几何后端上限。

### 8. Benchmark 与真实用途之间仍有距离

ScanNet25k 只有九类常见室内对象，CO3D adaptation 只有六类、600 张图。论文没有动态物体、透明物、极小对象、户外复杂背景、真实相机标定漂移或跨视频稳定性。准确 pose 也不等于 CAD topology、材质和实例身份正确。

### 9. 数据与 CAD 许可需要逐项治理

论文代码是 Apache-2.0，论文图是 CC BY 4.0，SupeRANSAC 是 MIT；DUNE、DINOv3、SAM2、MoGe-2、ShapeNet、ScanNet 和各训练数据集各有自己的使用条款。开源代码许可不自动覆盖模型权重、训练数据和 CAD 资产的商业使用。

## 66. 推荐阅读路径

如果只用 30 分钟，按以下顺序：

1. Abstract 与 Fig. 1，先掌握“feature scale-up + geometric matching”两项贡献。
2. Fig. 3 和 Sec. III-C，理解 NOC head 为什么只在训练存在。
3. Sec. III-D 的 mutual top-k、尺度估计、共识矩阵与信息矩阵公式。
4. Table II、IV、V、VI，分别检查主结果、组件贡献、上游瓶颈和计时边界。
5. README 的 checkpoint/数据声明，确认训练代码没有开放。
6. `sufleca/sv_align.py`，看完整 inference data flow 和 early exits。
7. `sufleca/geometry.py`，逐个核对 matching、consensus、RANSAC 和 information matrix。
8. `evaluation/eval_sv.py`，理解 NMS、count cap 与 current rank fusion，避免只看主表数字。

如果要集成，先用合成已知 transform 的点云测试 `geometry.py`，再准备一个 CAD 的六视角 cache，最后接真实 detector/depth。直接从完整 demo 开始，会把 DINOv3 许可、模型下载、render pool、SAM2 与 geometry 的错误混在一起，难以定位。

## 67. 结论：把 Foundation Feature 变成几何接口

SUFLECA 最值得保留的思想不是某个单一阈值，而是明确划分 representation 与 estimation：用大规模、多域 NOC 弱监督把冻结视觉特征塑造成紧凑的几何 descriptor；用 mutual top-k 保留跨域不确定性；用各向异性尺度共识把 feature candidate 转成几何可验证 correspondence；最后让鲁棒 estimator 输出 9D pose 与可排序质量信息。

这条路线比端到端 pose regressor更容易解释和替换：feature、render cache、matching、scale filter、RANSAC、score 都有独立合同。论文在 ScanNet25k 和 CO3D 上给出了有力证据，特别是 `33.4/42.3` 主结果、matching ladder 消融和 `0.53s/2178MB` alignment 成本。

核心风险同样明确。`zero-shot` 只覆盖 alignment，主结果依赖监督式 retrieval 与目标域 depth；训练和数据准备不开放；完整零样本 retrieval 只有 `3.8%` instance accuracy；$\log\det$ 尚未校准；RANSAC 占主要延迟。SUFLECA 推进的是“给定候选 CAD，如何更可靠地对齐”，而不是已经解决“从任意单图自动找到并装配正确 3D 资产”的全部问题。

从长期看，论文的价值在于提出一个可复用的 correspondence interface：视觉 foundation model 不必直接生成 pose，也不必输出最终 NOC；只要其稠密 feature 经几何监督后能在 real image 与 synthetic CAD render 间稳定对应，传统 registration 就能重新成为强而透明的求解器。对 CAD-based scene modeling、AR placement 和机器人对象建模，这种 learned feature 与 explicit geometry 的分工，比“再训练一个更大的黑盒 pose head”更容易审计、优化和组合。

## 参考资料

### 论文与官方实现

- Ejaz, S., Fernandez-Cortizas, M., Civera, J., Voos, H., & Sanchez-Lopez, J. L. [SUFLECA: Scaling Up Feature Learning for CAD-to-image Alignment](https://arxiv.org/abs/2607.15058v1). arXiv:2607.15058v1, 2026.
- [arXiv HTML v1](https://arxiv.org/html/2607.15058v1)、[PDF v1](https://arxiv.org/pdf/2607.15058v1)、[TeX source v1](https://arxiv.org/e-print/2607.15058v1)。
- [snt-arg/SUFLECA](https://github.com/snt-arg/SUFLECA)，Apache-2.0；本文源码链接固定到 commit [`9df942e`](https://github.com/snt-arg/SUFLECA/tree/9df942e22601648e8165b0cffd7432ad4dc50d33)。
- [SUFLECA release `v1`](https://github.com/snt-arg/SUFLECA/releases/tag/v1)。GitHub 页面展示名称为 `v1.0`，实际 Git tag 为 `v1`。

### 表示、特征与对应

- Wang et al. [Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation](https://arxiv.org/abs/1901.02970). CVPR 2019.
- Krishnan et al. [OmniNOCS: A Unified NOCS Dataset and Model for 3D Lifting of 2D Objects](https://arxiv.org/abs/2407.08711). 2024.
- Sariyildiz et al. [DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](https://arxiv.org/search/?query=DUNE+Distilling+a+Universal+Encoder&searchtype=all). 2025.
- Ranftl et al. [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413). ICCV 2021.
- Oquab et al. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193). 2023.

### CAD 对齐与场景建模

- Gümeli et al. [ROCA: Robust CAD Model Retrieval and Alignment from a Single Image](https://arxiv.org/abs/2112.12725). CVPR 2022.
- Avetisyan et al. [Scan2CAD: Learning CAD Model Alignment in RGB-D Scans](https://arxiv.org/abs/1811.11187). CVPR 2019.
- Nguyen et al. [GigaPose: Fast and Robust Novel Object Pose Estimation via One Correspondence](https://arxiv.org/abs/2311.14155). CVPR 2024.
- Wu et al. [Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling](https://arxiv.org/abs/2411.19492v1). 2024.
- [SupeRANSAC official repository](https://github.com/danini/superansac)，SUFLECA 仓库内 vendored binding 采用 MIT license。

### 上游模型与数据

- Ravi et al. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714). 2024.
- Liu et al. [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499). 2023.
- [ShapeNet](https://shapenet.org/)、[ScanNet](http://www.scan-net.org/)、[CO3D](https://ai.meta.com/datasets/co3d-downloads/)。使用数据与 CAD 前应分别核验许可。

> 本文依据论文 v1 与官方仓库 tag `v1` / commit `9df942e22601648e8165b0cffd7432ad4dc50d33` 做静态精读。没有下载 checkpoint、ScanNet25k 资源或 ShapeNet render pool，没有运行推理、训练或论文评测；所有实验数字均来自论文，源码结论限定于该固定提交。

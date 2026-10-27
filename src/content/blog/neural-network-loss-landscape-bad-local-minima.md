---
title: "神经网络损失景观中的坏局部极小值：从 Backpropagation 到过参数化理论"
description: "梳理 BP、深度线性网络、宽网络、NTK 与过参数化视角下坏局部极小值问题的理论脉络"
pubDate: "2026-06-10T12:46:13+08:00"
updatedDate: "2026-06-10T12:46:13+08:00"
tags:
  - "Deep Reading"
  - "Deep Learning"
  - "Optimization"
  - "Loss Landscape"
  - "Theory"
draft: false
---

神经网络训练常被笼统地归结为“高维非凸优化”。由于多层权重矩阵的连乘与非线性激活函数的复合，经验风险函数在参数空间中呈现出局部极小值、鞍点、平坦高原、病态曲率以及连续的等价参数流形。然而，“目标函数非凸”与“梯度下降必然困于次优局部极小值”之间隔着漫长的理论鸿沟。

本文沿着早期反向传播（Backpropagation, BP）文献、深度线性网络、高维鞍点几何、宽网络与残差结构、神经正切核（NTK）直至现代过参数化收敛理论的脉络，系统梳理一个核心问题：**神经网络损失景观中的坏局部极小值（bad/spurious local minima），究竟在什么条件下真实存在，又在什么结构与尺度机制下消失或退居次要地位？**

## 1. 反向传播、梯度下降与坏局部极小值的形式化定义

给定训练集 $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{n}$ 与样本级损失函数 $\ell(\cdot,\cdot)$（如平方损失或交叉熵损失），监督学习的经验风险定义为

$$
L(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell(f_{\theta}(x_i),y_i),
$$

其中 $L$ 层前馈神经网络 $f_\theta(x)$ 由权重与偏置参数 $\theta=\{W_\ell,b_\ell\}_{\ell=1}^L$ 参数化：

$$
f_\theta(x)=W_L\sigma(W_{L-1}\sigma(\cdots\sigma(W_1x+b_1)\cdots)+b_{L-1})+b_L.
$$

即便外部损失 $\ell(\hat y, y)$ 关于预测值 $\hat y$ 是严格凸函数，参数进入目标函数的方式依然经过了多层矩阵乘积与非线性激活的嵌套复合，使得 $L(\theta)$ 在参数空间中高度非凸。

在梳理早期理论文献时，首先需要厘清三个经常被混用的概念：

1. **反向传播（Backpropagation, BP）**：一种利用链式法则在计算图上高效求取精确梯度 $\nabla_\theta L(\theta)$ 的算法。前向传播逐层计算 $a^0=x,\ z^\ell=W^\ell a^{\ell-1}+b^\ell,\ a^\ell=\sigma(z^\ell)$；反向传播则从输出层回传误差信号 $\delta^\ell=(W^{\ell+1})^\top \delta^{\ell+1}\odot \sigma'(z^\ell)$，并给出各层权重梯度 $\frac{\partial L}{\partial W^\ell}=\delta^\ell(a^{\ell-1})^\top$。正如 [Baldi & Sadowski 2016][baldi-sadowski-2016] 所言，BP 提供了深层信用分配所需的逆向信息通道（learning channel），其本身只负责精确算梯度。
2. **基于梯度的优化（Gradient-based training）**：依赖一阶梯度信号更新参数的过程，如标准梯度下降 $\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)$ 或随机梯度下降（SGD）。
3. **早期文献中的“BP 学习（BP learning）”**：在 [Gori & Tesi 1992][gori-tesi-1992]、[Yu 1992][yu-1992]、[Yu & Chen 1995][yu-chen-1995] 等 20 世纪 90 年代的论文中，“BP 是否会陷入局部极小值”讨论的实际上是“使用 BP 梯度的局部一阶优化过程”在误差曲面上能否收敛到全局最优。

### 局部极小值、坏局部极小值与鞍点

在参数空间中，若 $\nabla L(\theta^\star)=0$，则称 $\theta^\star$ 为**临界点（critical point）**。若对任意参数 $\theta$ 均有 $L(\theta^\star)\le L(\theta)$，则 $\theta^\star$ 为**全局极小值（global minimum）**；若存在邻域 $\mathcal{N}(\theta^\star)$ 使得对所有 $\theta\in\mathcal{N}(\theta^\star)$ 均有 $L(\theta^\star)\le L(\theta)$，则 $\theta^\star$ 为**局部极小值（local minimum）**。

理论文献中所称的**坏局部极小值**（bad / poor / spurious local minimum），特指在局部邻域内无法继续下降、但函数值严格劣于全局最小值的点：

$$
\theta^\star\text{ is a local minimum}\quad\text{and}\quad L(\theta^\star)>L_{\min}=\inf_\theta L(\theta).
$$

相应地，当一篇论文证明某类网络“没有坏局部极小值（no bad/spurious local minima）”时，其数学含义是**所有局部极小值均为全局极小值**（every local minimum is global）。这一性质远弱于凸性：一个高度非凸、存在多个分离盆地或弯曲谷地的函数，完全可以做到每个局部极小值都达到相同的全局最低损失。

同时，临界点除了局部极小值与局部极大值外，还包括大量的**鞍点（saddle points）**——即在某些方向上局部极小、在另一些方向上局部极大的临界点。若 Hessian 矩阵 $H=\nabla^2 L(\theta)$ 存在严格负特征值，该点称为**严格鞍点（strict saddle）**，二阶泰勒展开直接给出了局部下降方向；若 Hessian 半正定（无负特征值）而高阶项仍存在下降方向，则构成更难逃离的**退化鞍点（degenerate / bad saddle point）**。因此，排除坏局部极小值仅仅消除了“掉进次优死胡同”这一种静态风险，训练过程依然可能受制于鞍点平台期、平坦方向、病态条件数、初始化尺度以及训练误差与测试泛化之间的背离。

## 2. 早期 BP 误差曲面争论：局部陷阱的构造与无极小值充分条件

20 世纪 90 年代初围绕 BP 误差曲面的研究，开启了“构造失败反例”与“寻找良性结构条件”两条交织的路线。

[Gori & Tesi 1992][gori-tesi-1992] 在 *On the Problem of Local Minima in Backpropagation* 中明确展示了局部一阶训练的局限：在多层前馈网络中，确实可以构造出使梯度下降被困在次优局部极小值、无法正确拟合全部训练样本的具体实例。这一结果打破了将 BP 视为万能黑盒的早期乐观预期，证明在缺乏结构与数据条件约束时，非凸误差曲面上的局部几何、初始点位置与数据可分性足以阻碍全局收敛。

几乎在同一时期，[Yu 1992][yu-1992]（*Can backpropagation error surface not have local minima?*）与 [Yu & Chen 1995][yu-chen-1995] 从正面探索了误差曲面不存在坏局部极小值的结构性充分条件。针对含 $P$ 个互不重合输入样本（noncoincident input patterns）的两层 Sigmoid 前馈网络，Yu & Chen 证明：当隐藏层配备 $P-1$ 个 Sigmoid 神经元并辅以一个特殊构造的辅助神经元时，误差曲面上的每个局部极小值都是全局极小值（该证明的若干技术细节后经 [1996 年勘误][yu-chen-correction] 修正）。

用现代眼光回看，Yu & Chen 要求隐藏层宽度与样本数 $P$ 同阶，实际上已经触及了**过参数化**（over-parameterization）的核心直觉：当网络的参数自由度跨越由样本规模决定的临界阈值后，原本在欠参数化下封闭的次优盆地会打开新的下降通道。早期争论由此留下了此后三十年持续深化的三个问题：

1. 深度与非线性激活在制造坏局部极小值时分别扮演什么角色？
2. 网络宽度、残差连接或特定参数化如何从几何上消除次优局部极小值？
3. 在坏局部极小值消失后，高维训练动态的真正瓶颈转移到了哪里？

## 3. 深度线性网络：剥离非线性后的纯粹深度效应

为了单独检验“层数加深”对损失景观的影响，理论界首先研究了去掉非线性激活函数、仅保留多层矩阵连乘的**深度线性网络（deep linear networks）**：

$$
f_\theta(x)=W_LW_{L-1}\cdots W_1x.
$$

尽管它在函数表达能力上退化为线性映射 $f(x)=Wx$（其中 $W=W_L\cdots W_1$），但经验风险 $L(W_1,\ldots,W_L)$ 关于各层权重参数依然是高度非凸的多项式函数。

这一方向的先声来自 [Baldi & Hornik 1989][baldi-hornik-1989] 对两层线性自关联网络与主成分分析（PCA）关系的研究：在平方损失下，该网络的全局极小值对应于输入协方差矩阵主特征向量张成的子空间，而其余所有临界点均对应次要特征方向上的正交投影，在几何上全部表现为**鞍点**。

到了深度学习复兴期，[Kawaguchi 2016][kawaguchi-2016]（*Deep Learning without Poor Local Minima*）与 [Lu & Kawaguchi 2017][lu-kawaguchi-2017]（*Depth Creates No Bad Local Minima*）将这一结论推广到了任意深度与任意各层宽度的深度线性网络。在平方损失与温和的数据满秩假设下，他们严格证明了三条核心性质：

- 损失函数 $L(W_1,\ldots,W_L)$ 是非凸且非凹的；
- **所有局部极小值均为全局极小值**（即完全不存在坏局部极小值）；
- 所有非全局最优的临界点均为鞍点（其中包含二阶导数退化的高阶坏鞍点）。

这一系列定理（以及 [Yun, Sra & Jadbabaie 2018][yun-2017] 给出的充要判别条件，和 [Haeffele & Vidal 2015][haeffele-vidal-2015] 从正齐次矩阵/张量分解视角建立的全局最优框架）清晰地拆解了因果链条：

$$
\text{Depth alone} \implies \text{Non-convexity}, \qquad \text{Depth alone} \centernot\implies \text{Bad local minima}.
$$

单纯堆叠矩阵乘法虽然引入了非凸性与参数重缩放对称性，但矩阵分解的代数结构保证了任何次优临界点附近都存在使目标函数下降的扰动方向。

与此同时，[Saxe, McClelland & Ganguli 2013][saxe-2013] 对深度线性网络梯度下降微分方程的精确求解揭示了硬币的另一面：即便静态景观中没有任何坏局部极小值，深层网络的训练轨迹依然会经历漫长的平坦平台期（plateaus），不同奇异值模态按时间尺度逐级解锁。这说明深度带来的优化困难主要体现在**鞍点逃逸与病态动力学耦合**上，而非静态的次优局部陷阱。

## 4. 高维损失景观的几何图景：鞍点、低能带与模态连通性

当视角从深度线性网络转向一般高维非凸函数与真实深度网络时，统计物理与高维几何提供了理解损失景观的第二块拼图。

### 4.1 高维鞍点的统治地位与一阶逃逸

[Dauphin et al. 2014][dauphin-2014] 从高维随机高斯场与海森矩阵特征值谱分布出发指出：在极高维参数空间 $D$ 中，一个临界点要成为局部极小值，必须要求 Hessian 矩阵的全部 $D$ 个特征值同时非负；随着维度 $D$ 增大，正负特征值混合出现的概率呈指数级占优，使得高于全局最小值的临界点绝大多数都是鞍点。随后，一般非凸优化理论证明了随机初始化的一阶方法能够规避这类结构：[Lee et al. 2016][lee-2016] 利用稳定流形定理证明梯度下降几乎必然不会收敛到任何严格鞍点（strict saddle）；[Jin et al. 2017][jin-2017] 进一步证明加入各向同性随机扰动的梯度下降（Perturbed GD）可以在多项式时间内高效逃离严格鞍点并收敛至二阶驻点。

### 4.2 自旋玻璃近似、线性路径插值与连通谷地

在统计物理近似方向，[Choromanska et al. 2015][choromanska-2015] 将特定假设下的多层 ReLU 网络映射为球面自旋玻璃（spherical spin-glass）模型，指出随着网络规模扩大，局部极小值的损失值会高度集中在一个紧邻全局最小值的狭窄“低能带（band of low-energy critical points）”内，高损失的次优局部极小值在数量上呈指数级衰减。

经验层面的几何探测同样印证了这一趋势：

- **线性路径插值**：[Goodfellow, Vinyals & Saxe 2015][goodfellow-2015] 沿着随机初始化参数 $\theta_0$ 到收敛参数 $\theta^\star$ 的一维直线段检测损失曲线，发现许多常规架构在整条路径上呈现出平滑单调下降的形态，并未遭遇需要翻越的高耸能量壁垒。
- **滤波器归一化可视化**：[Li et al. 2018][li-2018] 提出消除尺度伪影的 Filter Normalization 可视化方法，直观展示了残差连接（skip connections）和宽通道设计如何将原本崎岖破碎的二维切片拉平为宽阔平滑的盆地。
- **模态连通性（Mode Connectivity）**：[Garipov et al. 2018][garipov-2018] 与 [Draxler et al. 2018][draxler-2018] 独立发现，不同随机种子训练得到的两个独立局部最优点之间，往往可以用一条简单的折线或二次曲线相连，且整条连接路径上的训练损失与测试准确率几乎不发生退化。

这些理论与实验证据共同刷新了对高维损失景观的几何直觉：在现代过参数化网络中，良性解更像是一片在高维空间中彼此连通的弯曲山谷流形（connected manifold of valleys），而非散落在参数空间各处的孤立针孔。

## 5. 非线性网络的正面理论：宽度、残差与过参数化如何消除局部陷阱

一旦重新引入 ReLU、Sigmoid 等非线性激活函数，矩阵连乘的代数交换性被打破。要让含非线性的神经网络排除坏局部极小值，理论证明必须借助**层宽越界、恒等残差参数化、特殊跳连神经元或神经正切核（NTK）极限**等结构性机制。

### 5.1 宽隐藏层与金字塔结构：表示满秩打开下降通道

当网络中存在一层宽度超过训练样本数 $n$ 的隐藏层时，该层在全部训练样本上的激活矩阵更容易保持满秩，从而允许后续层在特征空间中自由插值：

- **分段线性网络的扰动分析**：[Soudry & Carmon 2016][soudry-carmon-2016] 采用平滑分析（smoothed analysis）证明，在轻度过参数化（最后一层隐藏层神经元数与样本数之积超过输入维度量级）及类似 Dropout 的乘性噪声扰动下，分段线性网络的所有可微局部极小值均可达到零训练误差。
- **宽层 + 金字塔结构**：[Nguyen & Hein 2017][nguyen-hein-2017] 针对平方损失与实解析激活函数证明，若全连接网络某一隐藏层的神经元数严格大于训练样本数（$n_k > N$），且从该层到输出层的宽度呈非递增的金字塔结构（pyramidal structure），则几乎所有（勒贝格测度意义下）局部极小值均为全局最优。此后 [Lederer 2020][lederer-2020] 也在宽深网络的经验风险最小化设定下给出了无伪局部极小值的统一界。

### 5.2 恒等参数化与特殊跳连神经元：构造性消除次优驻点

除了直接加宽隐藏层，改变层间拓扑连接同样可以从几何上消灭次优局部极小值：

- **残差网络的恒等参数化**：[Hardt & Ma 2016][hardt-ma-2016] 将 ResNet $h_{\ell+1}=h_\ell+F_\ell(h_\ell)$ 的成功归结为“每一层都能在零点附近自然表达恒等映射（identity parameterization）”，并严格证明任意深度的线性残差网络不存在任何伪局部极小值，同时参量超过样本数的 ReLU 残差网络具备有限样本精确表征能力。
- **单个特殊跳连神经元**：[Liang et al. 2018][liang-2018]（*Adding One Neuron Can Eliminate All Bad Local Minima*）在二分类任务中证明了一个极具启发性的构造性结论：向任意普通前馈网络中添加一个带有直接通往输出层跳跃连接（skip connection）的特殊指数/多项式激活神经元，只要当前参数 $\theta$ 的损失高于全局最小值（$L(\theta)>L_{\min}$），沿着该特殊神经元的参数方向就必然存在严格下降方向 $\Delta\theta$ 使得 $L(\theta+\Delta\theta)<L(\theta)$，从而在全局范围内彻底消除所有坏局部极小值。

### 5.3 NTK 极限与多项式宽度收敛理论：初始化附近的动态凸化

另一条极具影响力的路线不再试图证明整个参数空间静态无陷阱，而是分析**足够宽的网络在随机初始化附近的梯度下降轨迹**。

[Jacot, Gabriel & Hongler 2018][jacot-2018] 提出的**神经正切核**（Neural Tangent Kernel, NTK）指出，当隐藏层宽度趋于无穷且采用合适的缩放初始化时，训练过程中每个参数偏离初始值 $\theta_0$ 的幅度极小，网络输出关于参数的一阶泰勒展开在整个训练期内保持精确：

$$
f_{\theta}(x) \approx f_{\theta_0}(x) + \nabla_\theta f_{\theta_0}(x)^\top(\theta-\theta_0), \qquad K(x,x') = \nabla_\theta f_{\theta_0}(x)^\top \nabla_\theta f_{\theta_0}(x').
$$

在这一惰性训练（lazy training）状态下，非凸神经网络的训练动态退化为确定性核矩阵 $K(x,x')$ 驱动的核回归问题。[Nitta 2018/2022][nitta-2018] 据此证明，在正态初始化与无限宽 NTK 极限下，深层 ReLU 网络的梯度下降轨迹不会落入任何伪局部极小值。[Allen-Zhu, Li & Song 2018/2019][allen-zhu-2018] 等工作进一步将无限宽极限拉回有限网络，证明当每层宽度达到样本数 $n$、深度 $L$ 与数据分离度倒数的多项式量级时，随机初始化的梯度下降能以线性收敛率将训练损失压至零。

## 6. 负面定理与反例：当有限宽非线性网络遇上真实局部陷阱

正面理论的繁荣并不意味着非线性网络的次优局部极小值是个伪命题。当网络宽度有限、结构不满足金字塔或跳过连接假设时，严格的负面结果证明了**伪局部极小值依然广泛存在，且具有不可忽略的吸引域**。

### 6.1 Safran & Shamir 2018：两层 ReLU 网络中普遍存在的伪局部极小值

[Safran & Shamir 2018][safran-shamir-2018] 考察了最基础的两层无偏置 ReLU 网络在Teacher-Student 设定下的平方损失：

$$
x\mapsto \sum_{i=1}^{k}\max\{0,w_i^\top x\}, \qquad x\sim\mathcal{N}(0,I_d).
$$

通过严谨的计算机辅助证明（computer-assisted proof），他们证实：即便输入服从标准高斯分布、输入维度 $d$ 任意高、且真实标签由完全同构的 $k$ 神经元教师网络无噪声生成，当 $6\le k\le 20$ 时，损失曲面上依然存在局部海森矩阵严格正定的**坏局部极小值**；而且从标准随机初始化出发运行梯度下降，有相当高的经验概率直接掉进这些次优陷阱。与此同时，他们也在实验中观察到：只要将学生网络的神经元数从 $k$ 略微增加到轻度过参数化规模（如 $2k$），梯度下降陷入次优局部极小值的概率便急剧下降。这一结果精准勾勒出了“恰好参数化（exact parameterization）”与“过参数化”之间的相变边界。

### 6.2 Du et al. 2018：静态存在坏极小值，但动态训练仍能高概率成功

[Du et al. 2018][du-2018] 对单隐藏层非重叠卷积 ReLU 网络（one-hidden-layer non-overlapping CNN）的分析则揭示了**静态景观与动态轨迹的分离**。论文一方面严格证明了在高斯输入下，该网络的总体风险曲面确实存在非全局最优的伪局部极小值；但另一方面又证明，只要配合权重归一化（Weight Normalization）与对称随机初始化，普通梯度下降依然能以不低于 $1/4$ 的常数概率避开这些坏极小值的吸引域、精确恢复真实卷积核参数（通过少数几次随机重启即可将成功率推向 1）。

这一定理说明，静态损失曲面上是否存在坏局部极小值，与随机初始化的一阶算法在实际训练中是否会被其捕获，是两个必须区分的数学问题：**次优极小值的吸引域体积（basin volume）与初始化分布的几何相对位置**，往往比极小值本身的存在性更具决定意义。

## 7. 四类消解机制与代表性文献全景对照

综合上述正面与负面理论，神经网络中“坏局部极小值消失或失效”的根源可以归纳为四种截然不同的数学机制：

| 机制分类 | 核心数学原理 | 代表性工作 | 适用边界与未覆盖情形 |
| --- | --- | --- | --- |
| **代数矩阵分解机制** | 多层线性乘积虽非凸，但矩阵因子分解的代数冗余保证了任何非全局驻点均存在下降方向 | [Baldi & Hornik 1989][baldi-hornik-1989]、[Kawaguchi 2016][kawaguchi-2016]、[Lu & Kawaguchi 2017][lu-kawaguchi-2017]、[Haeffele & Vidal 2015][haeffele-vidal-2015] | 仅适用于深度线性网络或满足特定正齐次/独立性简化假设的模型，无法直接推广至一般有限宽 ReLU 网络 |
| **宽层特征满秩插值机制** | 某一隐藏层宽度超过样本数 $n$ 时，样本激活矩阵满秩，使下游层可在表示空间内无阻碍地降低训练误差 | [Yu & Chen 1995][yu-chen-1995]、[Soudry & Carmon 2016][soudry-carmon-2016]、[Nguyen & Hein 2017][nguyen-hein-2017]、[Lederer 2020][lederer-2020] | 依赖层宽大于样本数、解析激活或金字塔结构，且仅保证训练误差极小，不直接给出测试泛化界 |
| **拓扑跳连与恒等参数化** | 通过残差恒等路径或直连输出的特殊神经元，为所有次优参数点显式注入局部下降方向 | [Hardt & Ma 2016][hardt-ma-2016]、[Liang et al. 2018][liang-2018] | 依赖特定残差参数化或特殊构造的直连神经元，不等同于任意普通网络微调 |
| **初始化邻域动态核化（NTK）** | 极宽网络在随机初始化附近参数位移极小，一阶线性化近似成立，非凸训练轨迹退化为凸核回归动态 | [Jacot et al. 2018][jacot-2018]、[Allen-Zhu et al. 2018][allen-zhu-2018]、[Nitta 2018/2022][nitta-2018] | 要求多项式级超大宽度且局限于 Lazy/Kernel 训练区，未能刻画有限宽大模型中的显著特征学习（Feature Learning） |

### 延伸讨论：训练全局最优与测试泛化（平坦性 vs 尺度重参数化）

即便网络在过参数化下顺利绕开坏局部极小值、找到了训练损失为零的全局极小值流形，流形上不同全局极小值的**测试泛化能力**依然可能相差悬殊。

围绕“什么样的极小值泛化更好”，[Keskar et al. 2016][keskar-2016] 通过大 Batch 与小 Batch 训练的对比提出经验假说：小 Batch SGD 引入的梯度噪声有助于优化器逃离尖锐极小值（sharp minima），落入泛化性能更优的平坦极小值（flat minima）。然而，[Dinh et al. 2017][dinh-2017] 随后给出了一个尖锐的理论反例：在多层正齐次 ReLU 网络中，利用层间权重的尺度重参数化变换 $(W_1, W_2) \mapsto (\alpha W_1, \alpha^{-1} W_2)$，可以在保持网络输入输出函数 $f_\theta(x)$ 逐点完全不变的前提下，将同一个极小值点的 Hessian 迹与最大特征值放大到任意高（即变得任意“尖锐”）。这证明朴素的欧氏参数空间曲率并非重参数化不变量；要连接损失景观几何与泛化，必须同时考量参数范数约束、流形体积、归一化层与 SGD 的隐式正则化偏置。

## 8. 总结：从“有无局部陷阱”走向高维优化动力学

回望从 1989 年至过参数化时代的完整理论演进，关于神经网络坏局部极小值的认识已经完成了三次关键跃迁：

1. **从笼统的“非凸即困难”到精细的结构拆解**：深度线性网络与矩阵分解定理证明了单纯增加网络深度不会制造坏局部极小值；而非线性激活在欠参数化或恰参数化下（如两层与单隐藏层卷积 ReLU 网络）确实会产生真实的伪局部极小值。
2. **从“单点静态存在性”到“过参数化相变与连通流形”**：当网络宽度、残差连接或参数规模跨越临界阈值后，孤立的次优陷阱被打通，高维参数空间中的低损失解连成一片可通过简单曲线无障碍穿行的谷地（Mode Connectivity）。
3. **从“静态几何”到“算法动力学与隐式偏置”**：现代深度学习优化的核心追问，已经从早期的“误差曲面上是否存在局部极小值”，转向了更具操作性的动力学问题——随机初始化与归一化如何缩减坏极小值的有效吸引域？一阶优化器如何穿越高维退化鞍点与病态曲率平台？在无穷多个零训练误差的全局极小值中，有限步长 SGD 与特征学习机制又是如何挑选出具备良性泛化结构的那一个？

这正是连接经典反向传播理论与现代大模型优化理论的主线。

## 参考文献

- [Pierre Baldi and Kurt Hornik, 1989. *Neural networks and principal component analysis: Learning from examples without local minima*.][baldi-hornik-1989]
- [Marco Gori and Alberto Tesi, 1992. *On the Problem of Local Minima in Backpropagation*.][gori-tesi-1992]
- [Xiao-Hu Yu, 1992. *Can backpropagation error surface not have local minima?*][yu-1992]
- [Xiao-Hu Yu and Guo-An Chen, 1995. *On the local minima free condition of backpropagation learning*.][yu-chen-1995]
- [Yu and Chen, 1996. Correction note.][yu-chen-correction]
- [Andrew Saxe, James McClelland and Surya Ganguli, 2013. *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks*.][saxe-2013]
- [Yann Dauphin et al., 2014. *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization*.][dauphin-2014]
- [R. Haeffele and René Vidal, 2015. *Global Optimality in Tensor Factorization, Deep Learning, and Beyond*.][haeffele-vidal-2015]
- [Anna Choromanska et al., 2015. *The Loss Surfaces of Multilayer Networks*.][choromanska-2015]
- [Ian Goodfellow, Oriol Vinyals and Andrew Saxe, 2015. *Qualitatively characterizing neural network optimization problems*.][goodfellow-2015]
- [Pierre Baldi and Peter Sadowski, 2016. *A Theory of Local Learning, the Learning Channel, and the Optimality of Backpropagation*.][baldi-sadowski-2016]
- [Jason Lee et al., 2016. *Gradient Descent Only Converges to Minimizers*.][lee-2016]
- [Kenji Kawaguchi, 2016. *Deep Learning without Poor Local Minima*.][kawaguchi-2016]
- [Nitish Keskar et al., 2016. *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*.][keskar-2016]
- [Daniel Soudry and Yair Carmon, 2016. *No bad local minima: Data independent training error guarantees for multilayer neural networks*.][soudry-carmon-2016]
- [Moritz Hardt and Tengyu Ma, 2016. *Identity Matters in Deep Learning*.][hardt-ma-2016]
- [Haihao Lu and Kenji Kawaguchi, 2017. *Depth Creates No Bad Local Minima*.][lu-kawaguchi-2017]
- [Laurent Dinh et al., 2017. *Sharp Minima Can Generalize For Deep Nets*.][dinh-2017]
- [Chi Jin et al., 2017. *How to Escape Saddle Points Efficiently*.][jin-2017]
- [Quynh Nguyen and Matthias Hein, 2017. *The Loss Surface of Deep and Wide Neural Networks*.][nguyen-hein-2017]
- [Chulhee Yun, Suvrit Sra and Ali Jadbabaie, 2017/2018. *Global optimality conditions for deep neural networks*.][yun-2017]
- [Itay Safran and Ohad Shamir, 2018. *Spurious Local Minima are Common in Two-Layer ReLU Neural Networks*.][safran-shamir-2018]
- [Simon Du et al., 2018. *Gradient Descent Learns One-hidden-layer CNN: Don't be Afraid of Spurious Local Minima*.][du-2018]
- [Hao Li et al., 2018. *Visualizing the Loss Landscape of Neural Nets*.][li-2018]
- [Timur Garipov et al., 2018. *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*.][garipov-2018]
- [Felix Draxler et al., 2018. *Essentially No Barriers in Neural Network Energy Landscape*.][draxler-2018]
- [Arthur Jacot, Franck Gabriel and Clément Hongler, 2018. *Neural Tangent Kernel: Convergence and Generalization in Neural Networks*.][jacot-2018]
- [Shiyu Liang et al., 2018. *Adding One Neuron Can Eliminate All Bad Local Minima*.][liang-2018]
- [Zeyuan Allen-Zhu, Yuanzhi Li and Zhao Song, 2018/2019. *A Convergence Theory for Deep Learning via Over-Parameterization*.][allen-zhu-2018]
- [Tohru Nitta, 2018/2022. *Spurious Local Minima of Deep ReLU Neural Networks in the Neural Tangent Kernel Regime*.][nitta-2018]
- [Johannes Lederer, 2020. *No Spurious Local Minima: on the Optimization Landscapes of Wide and Deep Neural Networks*.][lederer-2020]

[baldi-hornik-1989]: https://www.sciencedirect.com/science/article/pii/0893608089900142
[gori-tesi-1992]: https://doi.org/10.1109/34.107014
[yu-1992]: https://doi.org/10.1109/72.165604
[yu-chen-1995]: https://doi.org/10.1109/72.410380
[yu-chen-correction]: https://dblp.dagstuhl.de/rec/journals/tnn/YuC96.html
[saxe-2013]: https://arxiv.org/abs/1312.6120
[dauphin-2014]: https://arxiv.org/abs/1406.2572
[haeffele-vidal-2015]: https://arxiv.org/abs/1506.07540
[choromanska-2015]: https://arxiv.org/abs/1412.0233
[goodfellow-2015]: https://arxiv.org/abs/1412.6544
[baldi-sadowski-2016]: https://arxiv.org/abs/1506.06472
[lee-2016]: https://proceedings.mlr.press/v49/lee16.html
[kawaguchi-2016]: https://arxiv.org/abs/1605.07110
[keskar-2016]: https://arxiv.org/abs/1609.04836
[soudry-carmon-2016]: https://arxiv.org/abs/1605.08361
[hardt-ma-2016]: https://arxiv.org/abs/1611.04231
[lu-kawaguchi-2017]: https://arxiv.org/abs/1702.08580
[dinh-2017]: https://arxiv.org/abs/1703.04933
[jin-2017]: https://proceedings.mlr.press/v70/jin17a.html
[nguyen-hein-2017]: https://proceedings.mlr.press/v70/nguyen17a.html
[yun-2017]: https://arxiv.org/abs/1707.02444
[safran-shamir-2018]: https://arxiv.org/abs/1712.08968
[du-2018]: https://proceedings.mlr.press/v80/du18b.html
[li-2018]: https://arxiv.org/abs/1712.09913
[garipov-2018]: https://arxiv.org/abs/1802.10026
[draxler-2018]: https://proceedings.mlr.press/v80/draxler18a.html
[jacot-2018]: https://arxiv.org/abs/1806.07572
[liang-2018]: https://arxiv.org/abs/1805.08671
[allen-zhu-2018]: https://arxiv.org/abs/1811.03962
[nitta-2018]: https://arxiv.org/abs/1806.04884
[lederer-2020]: https://openreview.net/forum?id=EZ8aZaCt9k

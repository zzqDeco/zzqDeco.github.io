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

在经典数值优化的直觉中，一旦目标函数失去凸性（Convexity），局部一阶梯度算法便极易跌入次优的局部极小值（Suboptimal Local Minima）而无法自拔。然而，现代深度学习的日常实践却日复一日地上演着反直觉的一幕：面对参数量动辄数亿乃至数千亿、由多层矩阵连乘与非线性激活函数嵌套而成的高度非凸神经网络，仅凭最朴素的反向传播（Backpropagation, BP）与随机梯度下降（SGD），就能稳定地将训练经验风险压至零附近。

为什么“目标函数高度非凸”没有导致“梯度下降必然困死于次优局部极小值”？跨越从 1989 年早期反向传播分析到现代过参数化理论的三十年文献脉络，我们可以将神经网络损失景观（Loss Landscape）的底层机制凝练为**四条核心理论洞见**：

1. **非凸性不等于坏局部极小值**（代数深度与非线性瓶颈的剥离）：多层权重矩阵连乘固然使经验风险变为高度非凸的多项式函数，但矩阵因子分解的代数冗余结构保证了**单纯增加网络深度绝不会创造任何坏局部极小值**（即所有局部极小值均为全局极小值）；真正制造孤立次优陷阱的根源，是“欠参数化宽度瓶颈”与“非线性激活截断”的耦合。
2. **高维空间的临界点维度相变**（鞍点统治与一阶逃逸）：在维度高达 $D$ 的参数空间中，一个临界点要成为局部极小值，必须要求海森矩阵（Hessian）的全部 $D$ 个特征值同时非负。随机矩阵与统计物理谱分析表明，高损失区域的临界点以指数级概率优势呈现为正负特征值混合的**严格鞍点**（Strict Saddles），而随机初始化或带噪一阶梯度下降几乎必然能沿着负曲率特征方向逃离鞍点。
3. **过参数化与跳连拓扑打通逃逸流形**（从孤立针孔到连通山谷）：当隐藏层宽度跨越样本数阈值、引入恒等残差连接（Skip Connections）或进入神经正切核（NTK）线性化区间时，满秩特征插值与核动力学彻底瓦解了次优盆地的封闭边界，使高维参数空间中的最优解连成一片无能量壁垒的**连通山谷流形**（Mode Connectivity）。
4. **静态景观存在性与动态吸引域体积的分离**：即便在有限宽两层或卷积 ReLU 网络中严格存在海森矩阵正定的坏局部极小值，只要配合权重归一化与恰当的初始化尺度，次优陷阱的**吸引域体积**（Basin Volume）便会被大幅压缩；优化的核心矛盾由此从“静态曲面上有无局部陷阱”转向“训练动力学如何快速穿越退化鞍点平台并挑选出高泛化解”。

下文沿着“形式化几何分类 → 早期 BP 争论 → 深度线性网络 → 高维鞍点与连通流形 → 宽网络与 NTK 收敛 → 负面反例与泛化曲率争议”的脉络，完整拆解这一核心理论问题。

## 反向传播、梯度下降与临界点的几何分类学

给定训练集 $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{n}$（记输入矩阵为 $X\in\mathbb{R}^{d_0\times n}$，标签矩阵为 $Y\in\mathbb{R}^{d_L\times n}$）与样本级凸损失函数 $\ell(\cdot,\cdot)$（如平方损失或交叉熵损失），监督学习的经验风险定义为：

$$
L(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell(f_{\theta}(x_i),y_i),
$$

其中 $L$ 层前馈神经网络 $f_\theta(x)$ 由权重与偏置参数 $\theta=\{W_\ell,b_\ell\}_{\ell=1}^L$ 参数化：

$$
f_\theta(x)=W_L\sigma(W_{L-1}\sigma(\cdots\sigma(W_1x+b_1)\cdots)+b_{L-1})+b_L.
$$

即便外部损失 $\ell(\hat y, y)$ 关于预测输出 $\hat y$ 是严格凸函数，参数 $\theta$ 进入目标函数的方式依然经过了多层矩阵乘积与非线性激活 $\sigma(\cdot)$ 的复合，使得 $L(\theta)$ 在参数空间 $\mathbb{R}^D$ 中高度非凸（即存在 $\theta_1, \theta_2$ 及 $\lambda\in(0,1)$ 使得 $L(\lambda\theta_1+(1-\lambda)\theta_2) > \lambda L(\theta_1) + (1-\lambda)L(\theta_2)$）。

在梳理文献前，首先需要严格区分三个在早期论文中常被混为一谈的概念：
1. **反向传播**（Backpropagation, BP）：一种利用链式法则在计算图上高效计算精确一阶梯度 $\nabla_\theta L(\theta)$ 的动态规划算法。前向传播逐层计算激活值 $a^0=x,\ z^\ell=W_\ell a^{\ell-1}+b_\ell,\ a^\ell=\sigma(z^\ell)$；反向传播则从输出层逆向回传误差敏感度信号 $\delta^\ell=(W_{\ell+1})^\top \delta^{\ell+1}\odot \sigma'(z^\ell)$，并直接给出各层权重梯度 $\frac{\partial L}{\partial W_\ell}=\delta^\ell(a^{\ell-1})^\top$。正如 [Baldi & Sadowski 2016][baldi-sadowski-2016] 所言，BP 提供了深层信用分配所需的逆向信息通道（Learning Channel），它本身只负责无偏、精确地算出梯度，并不决定参数如何移动。
2. **基于梯度的优化动力学**（Gradient-based Optimization）：依赖一阶梯度信号更新参数的过程，如离散时间梯度下降 $\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)$、连续时间梯度流（Gradient Flow）微分方程 $\frac{d\theta_t}{dt} = -\nabla_\theta L(\theta_t)$，或引入小批量采样噪声的随机梯度下降（SGD）。
3. **早期文献中的“BP 学习”**（BP Learning）：在 [Gori & Tesi 1992][gori-tesi-1992]、[Yu 1992][yu-1992]、[Yu & Chen 1995][yu-chen-1995] 等 20 世纪 90 年代的论文中，“BP 是否会陷入局部极小值”讨论的实质问题是：**由 BP 提供梯度的一阶局部搜索动力学，在非凸误差曲面上是否会被高于全局最小值的局部盆地永久捕获**。

### 局部极小值、坏局部极小值与鞍点的二阶判据

设 $L(\theta)$ 二阶连续可微，我们在任意参数点 $\theta^\star$ 附近对损失函数做二阶泰勒展开：

$$
L(\theta^\star + \Delta\theta)
= L(\theta^\star)
+ \nabla L(\theta^\star)^\top \Delta\theta
+ \frac{1}{2}\Delta\theta^\top \nabla^2 L(\theta^\star)\Delta\theta
+ O(\|\Delta\theta\|_2^3).
$$

当一阶梯度消失（即 $\nabla L(\theta^\star)=0$）时，$\theta^\star$ 称为**临界点**（Critical Point，或驻点 Stationary Point）。此时局部几何形态由海森矩阵 $H(\theta^\star) = \nabla^2 L(\theta^\star) \in \mathbb{R}^{D\times D}$ 的特征值谱 $\{\lambda_1, \lambda_2, \dots, \lambda_D\}$ 决定：

| 临界点类型 | 一阶与二阶数学条件 | 局部几何形态与一阶优化后果 |
| --- | --- | --- |
| **全局极小值**（Global Minimum） | $\nabla L(\theta^\star)=0$，且 $\forall \theta,\ L(\theta^\star) \le L(\theta)$ | 处于整个景观的最低能级 $L_{\min}$；过参数化下通常构成连续流形 |
| **坏局部极小值**（Bad / Spurious Local Minimum） | 存在邻域 $\mathcal{N}(\theta^\star)$ 内 $L(\theta^\star)\le L(\theta)$，且 $L(\theta^\star) > L_{\min}$ | 局部四周全为上坡（如 $\lambda_{\min}(H) > 0$），无噪梯度下降一旦落入即被永久困死 |
| **严格鞍点**（Strict Saddle Point） | $\nabla L(\theta^\star)=0$，且最小特征值 $\lambda_{\min}(H) < 0$ | 沿负特征值对应特征向量 $v_{\min}$ 方向二阶项严格下降，加噪或随机初始化 GD 可高效逃离 |
| **退化鞍点**（Degenerate Saddle Point） | $\nabla L(\theta^\star)=0,\ \lambda_{\min}(H) = 0$，但存在高阶下降方向 | 二阶曲率在逃逸方向上退化为零（如猴子鞍面 $x^3 - 3xy^2$），会导致极漫长的训练平台期 |
| **局部极大值**（Local Maximum） | $\nabla L(\theta^\star)=0$，且最大特征值 $\lambda_{\max}(H) \le 0$ | 局部四周全为下坡，正向梯度下降具有天然排斥性，绝不会收敛至此 |

理解这张分类表有两处至关重要的概念边界：
- 首先，当一篇理论论文证明某类神经网络**“没有坏局部极小值”**（No Bad / Spurious Local Minima）时，其严格的数学含义是**所有局部极小值均为全局极小值**：

$$
\forall \theta^\star \in \{\theta : \theta \text{ is a local minimum of } L\},\qquad L(\theta^\star) = \inf_{\theta} L(\theta).
$$

这一性质远弱于凸性——一个高度非凸、拥有无数个分离盆地与弯曲山谷的函数，完全可以满足“每一个盆地的谷底都处于同一个全局最低海拔 $L_{\min}$”。
- 其次，**“不存在坏局部极小值”绝不意味着“训练轻而易举”**。排除坏局部极小值仅仅消除了“掉进次优死胡同”这一种静态风险；在真实训练中，一阶优化器依然可能被退化鞍点（Degenerate Saddles）的平坦高原拖慢数万步，或受困于病态海森条件数（Ill-conditioned Curvature）与梯度消失/爆炸。

## 早期 BP 误差曲面争论：失败反例构造与满秩充分条件

20 世纪 90 年代初围绕 BP 误差曲面的早期探索，同时开启了“构造次优局部陷阱反例”与“寻找无坏极小值结构条件”两条交织的路线。

### Gori & Tesi 1992：欠参数化与激活饱和如何锁死局部盆地

在早期连接主义热潮中，许多研究者乐观地猜想多层感知机的误差曲面天然平滑无陷阱。[Gori & Tesi 1992][gori-tesi-1992] 在 *On the Problem of Local Minima in Backpropagation* 中通过严谨的反例构造打破了这一幻想。

他们的核心洞见在于揭示了**非线性激活函数的导数衰减**与**隐藏层表征线性不可分**如何共同制造坏局部极小值。在使用 Sigmoid 或 $\tanh$ 激活的两层网络中，权重梯度的反向传播信号必须乘上激活函数的逐元素导数 $\sigma'(z)$。当网络隐藏层宽度有限、且训练数据集中包含几何上相互冲突的样本簇时，一部分样本会将隐藏层神经元推向深度饱和区（$|z| \gg 1,\ \sigma'(z) \approx 0$），从而切断这些样本对底层权重的梯度拉力；而剩余未饱和样本的局部梯度又恰好在当前次优参数点达到平衡（$\sum_{i\in\text{active}} \nabla\ell_i = 0$）。此时，尽管总训练误差远高于零，但任何微小参数扰动要么无法唤醒饱和神经元，要么会立刻推高未饱和样本的损失，从而在非零损失处形成四周封闭的坏局部极小值。

### Yu & Chen 1995：隐藏层宽度匹配样本数时的满秩逃逸

几乎在同一时期，[Yu 1992][yu-1992]（*Can backpropagation error surface not have local minima?*）与 [Yu & Chen 1995][yu-chen-1995] 从正面探索了什么样的网络结构能够彻底消灭坏局部极小值。

针对包含 $P$ 个互不重合输入样本（Noncoincident Input Patterns）$\{x_1, \dots, x_P\}$ 的两层前馈网络，Yu & Chen 证明（其技术细节后经 [1996 年勘误][yu-chen-correction] 完善）：**当隐藏层拥有 $P-1$ 个非线性神经元（连同偏置项共提供 $P$ 个自由度）时，误差曲面上的每一个局部极小值都是零训练误差的全局极小值**。

用现代线性代数的眼光审视，这一证明的底层机制异常清晰：记隐藏层在全部 $P$ 个样本上的激活输出矩阵为 $H(W_1) \in \mathbb{R}^{P \times P}$，网络最终预测为 $\hat Y = W_2 H(W_1)$。对输出层权重 $W_2$ 求偏导可得驻点条件：

$$
\frac{\partial L}{\partial W_2} = (\hat Y - Y) H(W_1)^\top = 0.
$$

只要隐藏层宽度达到样本数 $P$，实解析激活函数（如 Sigmoid）作用在互异样本上生成的激活矩阵 $H(W_1)$ 在几乎所有参数 $W_1$ 处均是**满秩可逆矩阵**（$\operatorname{rank}(H) = P$）。一旦 $H(W_1)$ 满秩，上式两边右乘 $(H(W_1)^\top)^{-1}$ 立即迫使残差矩阵 $\hat Y - Y = 0$，即任何临界点只要使 $H(W_1)$ 保持满秩，就必然已经达到了全局零误差。这实际上在三十年前就触及了现代**过参数化**（Over-parameterization）理论的灵魂：当隐藏层宽度跨过由样本数决定的秩阈值后，特征空间的满秩张成直接抹平了所有次优局部盆地。

## 深度线性网络：剥离非线性后的纯粹深度效应

进入深度学习复兴期后，理论界为了厘清“层数加深（Depth）”与“非线性激活（Nonlinearity）”在制造坏局部极小值时各自的责任，首先对剥离了非线性激活、仅保留多层矩阵连乘的**深度线性网络**（Deep Linear Networks）展开了彻底解剖：

$$
f_\theta(x)=W_LW_{L-1}\cdots W_1x, \qquad L(W_1,\dots,W_L) = \frac{1}{2n}\|W_L W_{L-1}\cdots W_1 X - Y\|_F^2.
$$

### 为什么说“深度本身绝不制造坏局部极小值”

尽管深度线性网络在函数空间中只能表达线性映射 $W_{\text{prod}} = W_L W_{L-1}\cdots W_1$，但损失函数 $L(W_1,\dots,W_L)$ 在参数空间中是一个高达 $2L$ 次的多项式，不仅是高度非凸且非凹的，还包含无穷多连续的参数重缩放对称流形（对任意可逆矩阵 $G_\ell$，将 $W_{\ell+1}\mapsto W_{\ell+1}G_\ell^{-1},\ W_\ell \mapsto G_\ell W_\ell$ 完全不改变网络输出）。

这一方向的先声来自 [Baldi & Hornik 1989][baldi-hornik-1989] 对两层线性自编码器（$L(W_1, W_2) = \|W_2 W_1 X - X\|_F^2$，隐层宽度 $r < d_0$）的完整解析：他们证明该景观仅有一组全局极小值（对应输入协方差矩阵 $\Sigma_{XX}$ 的前 $r$ 个主成分特征向量张成的 PCA 子空间），而其余所有临界点（对应挑选了次要特征向量的投影子空间）无一例外全是**严格鞍点**——只要沿着未入选的更大特征值方向做无穷小扰动，损失函数就会严格下降。

2016 年，[Kawaguchi 2016][kawaguchi-2016]（*Deep Learning without Poor Local Minima*）与随后的 [Lu & Kawaguchi 2017][lu-kawaguchi-2017]（*Depth Creates No Bad Local Minima*）、[Yun, Sra & Jadbabaie 2018][yun-2017] 将这一结论一举推广到了**任意深度 $L$ 与任意各层宽度 $\{d_0, d_1, \dots, d_L\}$** 的深度线性网络。在平方损失与温和的数据协方差满秩假设（$\Sigma_{XX} = XX^\top \succ 0$）下，他们给出了三条铁律：
1. **所有局部极小值均为全局极小值**：无论层数 $L$ 有多深、中间是否存在窄瓶颈层（$d_k < \min(d_0, d_L)$），只要 $\theta^\star$ 是局部极小值，其连乘矩阵 $W_L^\star \cdots W_1^\star$ 必然达到当前秩约束 $r_{\max} = \min_\ell d_\ell$ 下的全局最小损失（由 Eckart-Young-Mirsky 奇异值截断定理唯一确定）。
2. **所有非全局最优的临界点皆为鞍点**：若某个临界点处的损失高于全局最小值，则必然是由于连乘矩阵的秩发生退化（$\operatorname{rank}(W_L\cdots W_1) < r_{\max}$）或对齐了次优奇异向量；此时利用矩阵乘法的双线性结构，总能在退化核空间中构造出使损失严格下降的局部扰动方向。
3. **当 $L \ge 3$ 时会出现高阶退化鞍点**：在两层线性网络（$L=2$）中，所有鞍点都是具有严格负特征值的二阶严格鞍点；但当深度 $L \ge 3$ 时，若存在连续多层权重同时出现秩亏（例如在原点 $W_1 = W_2 = \dots = W_L = 0$ 处，梯度一阶导与海森二阶导全部为零，直到第 $L$ 阶导数才非零），就会产生海森矩阵半正定（$\nabla^2 L \succeq 0$）但高阶泰勒项存在下降方向的**退化鞍点**（Bad / Degenerate Saddles）。

这一系列定理（连同 [Haeffele & Vidal 2015][haeffele-vidal-2015] 从正齐次矩阵/张量分解视角建立的全局最优准则）在数学上彻底斩断了“层数深”与“坏局部极小值”的虚假联系：

$$
\text{Depth alone} \implies \text{Algebraic Non-convexity \& Degenerate Saddles}, \qquad \text{Depth alone} \centernot\implies \text{Bad Local Minima}.
$$

### Saxe et al. 2013：无坏极小值景观上的漫长鞍点平台动力学

既然深度线性网络连一个坏局部极小值都没有，为什么训练一个没有归一化或残差连接的极深网络依然异常困难？[Saxe, McClelland & Ganguli 2013][saxe-2013] 通过精确求解深度线性网络的连续时间梯度流微分方程，给出了动力学层面的完美解答。

在输入白化（$\frac{1}{n}XX^\top = I$）且各层权重初始奇异向量对齐的条件下，设输入-输出交叉协方差矩阵 $\Sigma_{YX} = \frac{1}{n}YX^\top$ 的第 $\alpha$ 个奇异值为 $s_\alpha$，连乘矩阵 $W_L\cdots W_1$ 在对应模态上的有效奇异值为 $u_\alpha(t)$。Saxe 等人推导出不同奇异值模态的演化完全解耦，且每个模态独立遵循如下非线性常微分方程：

$$
\tau \frac{d u_\alpha}{dt} = L \, u_\alpha^{2 - 2/L}\left(s_\alpha - u_\alpha\right),
$$

其中 $\tau$ 为与学习率成反比的时间常数。观察方程右侧的乘子 $u_\alpha^{2 - 2/L}$ 可以立即看出深度的动力学代价：
- 当 $L=1$（浅层线性回归）时，方程退化为 $\tau \dot u_\alpha = s_\alpha - u_\alpha$，误差按指数规律 $\exp(-t/\tau)$ 迅速衰减，不存在任何平台期。
- 当 $L=2$ 时，方程变为逻辑斯蒂增长（Logistic Growth）$\tau \dot u_\alpha = 2 u_\alpha (s_\alpha - u_\alpha)$，从极小初始值 $u_\alpha(0) = \epsilon \ll s_\alpha$ 逃离零鞍点所需的时间为对数级 $O\big(\frac{1}{s_\alpha}\ln(s_\alpha/\epsilon)\big)$。
- 当 $L \ge 3$ 时，零点 $u_\alpha = 0$ 变成了 $L$ 阶退化鞍点。对微分方程积分可知，从微小随机初始化 $u_\alpha(0) = \epsilon$ 逃离零点附近平坦高原所需的潜伏时间飙升至**幂律发散级**：

$$
T_{\text{escape}}(L) \propto \frac{1}{s_\alpha \, \epsilon^{1 - 2/L}}.
$$

这揭示了一个极其深刻的图景：**静态景观中没有坏局部极小值，只保证了时间趋于无穷时训练不会卡死在次优解；但在有限训练步数内，深层矩阵连乘在零点附近制造的高阶退化鞍点高原（Plateaus），才是阻碍深层网络收敛的真正动力学元凶**——这也从第一性原理上解释了为何正交初始化（使初始奇异值 $u_\alpha(0)\approx 1$ 远离零鞍点）与残差连接对训练超深网络至关重要。

## 高维损失景观的统计物理与几何图景：鞍点、低能带与模态连通性

当我们将视线从线性网络移向一般的高维非凸函数与真实多层非线性网络时，随机矩阵理论、统计物理自旋玻璃模型与高维拓扑探测共同拼出了损失景观的第二块版图。

### 为什么高维空间里高损失临界点几乎全是鞍点？

在二维或三维玩具函数的直觉里，山峰（局部极大）、山谷（局部极小）与山口（鞍点）似乎平分秋色。但 [Dauphin et al. 2014][dauphin-2014] 与 [Choromanska et al. 2015][choromanska-2015] 借助统计物理中的高维高斯随机场（Gaussian Random Fields）与球面 $p$-自旋玻璃模型（Spherical $p$-Spin Glass Model）指出，高维空间 $D \gg 1$ 存在剧烈的**临界点指数分层现象**。

设想在 $D$ 维参数空间中任取一个临界点 $\theta^\star$（$\nabla L(\theta^\star)=0$），记其海森矩阵 $H(\theta^\star)$ 负特征值的个数占总维度 $D$ 的比例为临界点指标 $\alpha \in [0, 1]$（$\alpha=0$ 即全部 $D$ 个特征值非负，对应局部极小值；$\alpha > 0$ 则对应存在 $\alpha D$ 个下降维度的鞍点）。根据随机矩阵理论中的 Wigner 半圆律（Wigner Semicircle Law），大型对称随机矩阵的特征值谱密度分布在区间 $[-2\sqrt{D}\sigma, 2\sqrt{D}\sigma]$ 上呈半圆形，且谱分布的中心位置与该临界点所处的能量（损失值 $L(\theta^\star)$）呈单调负相关：
- 当临界点处于**高损失区域**（$L(\theta^\star) \gg L_{\min}$）时，海森特征值谱的中心位于零点附近甚至偏负。此时要求全部 $D$ 个特征值同时越过零点挤在正半轴（即成为坏局部极小值），属于极小概率的“大偏差事件（Large Deviation）”，其发生概率随维度 $D$ 呈高斯指数衰减：

$$
\Pr\big(\lambda_{\min}(H(\theta^\star)) \ge 0 \ \big|\ L(\theta^\star) > L_{\min} + \Delta\big) \sim \exp\left(-c(\Delta) D^2\right).
$$

- 唯有当损失值下降到紧贴全局最小值的狭窄**低能带**（Band of Low-Energy Critical Points）以内时，特征值谱才整体右移至正半轴，使临界点真正转化为局部极小值。

在此基础上，非凸优化理论严格证明了一阶梯度方法规避鞍点的能力：[Lee et al. 2016][lee-2016] 运用动力系统中的稳定流形定理（Stable Manifold Theorem）证明，对于任何严格鞍点（$\lambda_{\min}(H) < 0$），会收敛到该鞍点的初始参数集合在 $\mathbb{R}^D$ 中的勒贝格测度严格为零（Measure Zero），即随机初始化的梯度下降以概率 $1$ 不会收敛到任何严格鞍点；[Jin et al. 2017][jin-2017] 进一步证明，只需在梯度陷入停滞时注入微小的各向同性球面噪声（Perturbed Gradient Descent），一阶算法就能在仅依赖于维度多对数因子 $O(\operatorname{polylog}(D)/\epsilon^2)$ 的迭代步数内高效逃离所有严格鞍点，直达二阶驻点。

### 滤波器归一化可视化与模态连通性（Mode Connectivity）

既然高维空间中的低能极小值数量庞大，这些极小值在几何上究竟是一座座彼此隔绝的孤岛，还是相互贯通的盆地？2015 至 2018 年的三项里程碑式实证研究彻底刷新了人们的几何直觉：

1. **一维线性路径无壁垒**：[Goodfellow, Vinyals & Saxe 2015][goodfellow-2015] 沿着随机初始化参数 $\theta_0$ 到 SGD 收敛解 $\theta^\star$ 的一维直线段 $\theta(\alpha) = (1-\alpha)\theta_0 + \alpha\theta^\star$ 扫描损失值，发现曲线呈现平滑单调下降，沿途并未翻越任何高耸的能量屏障。
2. **滤波器归一化揭示残差平滑效应**（Filter Normalization）：[Li et al. 2018][li-2018] 指出，直接在参数空间取随机高斯方向绘制二维等高线会受层间权重尺度差异的严重扭曲；通过按卷积核范数归一化的方向切片 $d_{i,j} \leftarrow \frac{d_{i,j}}{\|d_{i,j}\|_F}\|w_{i,j}\|_F$，他们直观展示了极深普通网络（如无残差的 VGG-56）的损失景观确实充满破碎尖锐的沟壑，但一旦引入**残差跳跃连接**（ResNet-56）与加宽通道，二维损失切片便奇迹般地化作一口平滑宽阔的近凸大碗。
3. **模态连通性**（Mode Connectivity）：如果取两个从不同随机种子独立训练收敛的全局极小值 $\theta_1$ 与 $\theta_2$，直接连接它们的直线段 $\theta(t) = (1-t)\theta_1 + t\theta_2$ 中点处损失往往很高，这曾让人误以为不同极小值位于孤立盆地中。然而，[Garipov et al. 2018][garipov-2018] 与 [Draxler et al. 2018][draxler-2018] 独立发现，只需将直线放宽为含单个控制节点 $\theta_c$ 的简单二次贝塞尔曲线（Quadratic Bézier Curve）或两段折线：

$$
\phi_{\theta_c}(t) = (1-t)^2 \theta_1 + 2t(1-t)\theta_c + t^2 \theta_2, \qquad t\in[0,1],
$$

通过最小化路径积分损失 $\int_0^1 L(\phi_{\theta_c}(t))\,dt$ 寻找拐点 $\theta_c$，总能找到一条连接 $\theta_1$ 与 $\theta_2$ 的平坦通道，使得整条路径上的训练损失与测试错误率始终维持在最低谷底（几乎零能量壁垒）。这有力地证明：在现代过参数化网络中，所谓的“多个局部极小值”实际上共同编织成了一张在高维空间中处处连通的**低损耗山谷流形**（Connected Manifold of Low-Loss Valleys）。

## 非线性网络的正面理论：宽度、残差与过参数化如何消除局部陷阱

一旦在网络中重新引入 ReLU、Sigmoid 等逐元素非线性激活函数，深度线性网络所仰仗的矩阵结合律（$W_2(W_1 x) = (W_2 W_1)x$）便宣告失效。为了让含非线性的神经网络在理论上排除坏局部极小值或保证全局收敛，研究者们发展出了三条极具洞察力的结构证明路线。

### 宽隐藏层与金字塔拓扑：特征矩阵满秩打开下降通道

延续早期 Yu & Chen 的直觉，第一条路线着眼于**当网络中存在一层宽度超过训练样本数 $n$ 的隐藏层时，非线性激活矩阵的满秩性质如何消灭次优驻点**。

[Nguyen & Hein 2017][nguyen-hein-2017]（*The Loss Surface of Deep and Wide Neural Networks*）针对实解析严格递增激活函数与任意凸可微损失证明了一个里程碑式的定理：设 $L$ 层全连接网络在第 $k$ 层隐藏层的宽度满足 $d_k \ge n$（神经元数不少于训练样本数），且从第 $k+1$ 层直到输出层 $L$ 的各层宽度呈非递增的**金字塔结构**（Pyramidal Structure，$d_{k+1} \ge d_{k+2} \ge \dots \ge d_L$）。那么，对于任何满足以下两个非退化条件的临界点 $\theta^\star$：
1. 第 $k$ 层在全部 $n$ 个训练样本上的激活矩阵 $F_k \in \mathbb{R}^{n \times d_k}$ 保持行满秩（$\operatorname{rank}(F_k) = n$）；
2. 从第 $k+1$ 层到第 $L$ 层的权重矩阵均保持满秩（$\operatorname{rank}(W_\ell) = d_\ell,\ \forall \ell > k$）；

该临界点 $\theta^\star$ **必然是全局极小值**。由于违反上述满秩条件的参数集合在参数空间中的勒贝格测度为零，这意味着宽金字塔网络的几乎所有局部极小值都自动达到了全局最优。其几何机制非常直观：当第 $k$ 层宽度 $d_k \ge n$ 且激活满秩时，样本在该层的特征表征张成了整个 $\mathbb{R}^n$ 空间；后续非递增的满秩金字塔层不会丢失误差回传维度（雅可比算子满射），因此只要预测残差 $\hat Y - Y \neq 0$，梯度 $\nabla_{W_{k+1}} L$ 就绝不可能为零。[Soudry & Carmon 2016][soudry-carmon-2016] 在分段线性（Leaky ReLU）网络上利用乘性扰动平滑分析，以及 [Lederer 2020][lederer-2020] 在宽深网络经验风险上的统一界，也从不同角度印证了这一“宽层满秩消除坏极小值”的普适机制。

### 恒等残差参数化与单个跳连神经元的构造性逃逸

如果网络每一层都很窄（无法满足 $d_k \ge n$），能否仅靠改变层间拓扑连接来消灭坏局部极小值？答案是肯定的：

- **残差网络的恒等参数化**（Identity Parameterization）：[Hardt & Ma 2016][hardt-ma-2016]（*Identity Matters in Deep Learning*）剖析了 ResNet $h_{\ell} = h_{\ell-1} + W_\ell h_{\ell-1} = (I + W_\ell)h_{\ell-1}$ 的优化奥秘。在前馈网络 $\prod_\ell W_\ell$ 中，原点 $W_\ell = 0$ 是严重秩亏的高阶退化鞍点；而在恒等残差参数化 $\prod_\ell (I + W_\ell)$ 下，原点 $W_\ell = 0$ 恰恰对应满秩的恒等映射 $I$。Hardt & Ma 证明，只要各层残差矩阵的谱范数满足 $\|W_\ell\|_2 < 1$，连乘映射天然保持可逆且条件数有界，不仅彻底消除了零点附近的退化鞍点高原，而且当参数量超过样本数时，多层 ReLU 残差网络可以用极小的权重范数精确表征任意有限样本集。
- **单个特殊跳连神经元消除所有坏极小值**：[Liang et al. 2018][liang-2018]（*Adding One Neuron Can Eliminate All Bad Local Minima*）在二分类与回归任务中给出了一个极具启发性的构造性定理。设 $g_w(x)$ 为任意结构（任意深度、任意宽度、任意激活函数，哪怕内部充满次优陷阱）的普通神经网络；现在仅向网络中额外并联**一个**直连输出层的指数或多项式跳连神经元（Skip-Connection Neuron）：

$$
f_\theta(x) = g_w(x) + a \exp(b^\top x + c) \quad \text{或} \quad g_w(x) + \sum_{j=1}^m a_j (b_j^\top x + c_j)^p.
$$

Liang 等人严格证明：在加上这单个跳连神经元并配合关于 $(a,b,c)$ 的温和正则项后，**新损失函数的每一个局部极小值都是全局极小值**。其数学原理在于反证逃逸构造：假设当前参数点 $\theta^\star$ 的损失高于全局最小值，那么残差向量在样本空间中非零；利用指数函数族 $\{\exp(b^\top x_i)\}$ 对互异样本点的线性无关性（Vandermonde 行列式非零），总能通过微调跳连神经元的方向参数 $b$ 与幅值 $a$，在局部二阶或高阶泰勒展开中构造出与当前残差呈负内积的严格下降方向 $\Delta\theta$，从而像在封闭盆地底部凿开一条排水隧道一样，让所有次优局部极小值瞬间转化为可逃离的鞍点。

### 神经正切核（NTK）与多项式宽度全局线性收敛

第三条路线跳出了对整个参数空间静态临界点的全局排查，转而直接追踪**足够宽的神经网络在随机初始化 $\theta_0$ 附近的梯度下降轨迹**，代表作包括 [Jacot, Gabriel & Hongler 2018][jacot-2018] 提出的**神经正切核**（Neural Tangent Kernel, NTK）以及 [Allen-Zhu, Li & Song 2018/2019][allen-zhu-2018]、[Du et al. 2019] 建立的有限宽过参数化收敛定理。

让我们用最简明的微分方程推导 NTK 为何能让非凸训练呈现出强凸般的全局收敛性。考虑平方损失 $L(\theta) = \frac{1}{2}\|u(\theta) - y\|_2^2$，其中 $u(\theta) = [f_\theta(x_1), \dots, f_\theta(x_n)]^\top \in \mathbb{R}^n$ 为网络在 $n$ 个训练样本上的预测向量。记预测向量对参数 $\theta\in\mathbb{R}^D$ 的雅可比矩阵为 $J(\theta) = \nabla_\theta u(\theta) \in \mathbb{R}^{n \times D}$。在连续时间梯度流 $\frac{d\theta_t}{dt} = -\nabla_\theta L(\theta_t) = -J(\theta_t)^\top (u(\theta_t) - y)$ 下，利用链式法则写出预测向量 $u_t = u(\theta_t)$ 在函数空间中的演化方程：

$$
\frac{d u_t}{dt} = J(\theta_t)\frac{d\theta_t}{dt} = -\underbrace{J(\theta_t) J(\theta_t)^\top}_{\Theta_t \in \mathbb{R}^{n\times n}} (u_t - y) = -\Theta_t (u_t - y),
$$

其中半正定矩阵 $\Theta_t$ 的第 $(i,j)$ 个元素即为经验**神经正切核**：

$$
(\Theta_t)_{i,j} = \big\langle \nabla_\theta f_{\theta_t}(x_i),\ \nabla_\theta f_{\theta_t}(x_j) \big\rangle.
$$

对训练损失 $L(t) = \frac{1}{2}\|u_t - y\|_2^2$ 关于时间求导，立得：

$$
\frac{d L(t)}{dt} = (u_t - y)^\top \frac{d u_t}{dt} = -(u_t - y)^\top \Theta_t (u_t - y) \le -2\,\lambda_{\min}(\Theta_t)\,L(t).
$$

这一不等式揭示了一个惊人的事实：**只要在训练轨迹沿途，这个 $n\times n$ 的核矩阵 $\Theta_t$ 的最小特征值始终有正下界**（例如 $\lambda_{\min}(\Theta_t) \ge \frac{\lambda_0}{2} > 0$），那么微分不等式直接给出指数级全局收敛：

$$
L(t) \le \exp(-\lambda_0 t)\,L(0) \xrightarrow{t\to\infty} 0!
$$

为什么过参数化能保证 $\lambda_{\min}(\Theta_t) \ge \frac{\lambda_0}{2} > 0$？[Jacot et al. 2018][jacot-2018] 与 [Allen-Zhu et al. 2018][allen-zhu-2018] 证明了两个关键引理：
1. **初始核严格正定**：只要样本互不平行（$\|x_i - x_j\|_2 \ge \delta$）且网络宽度 $m$ 足够大，随机高斯初始化下的初始核 $\Theta_0$ 依高概率依谱范数收敛于确定性的严格正定极限核 $\Theta^\infty$，即 $\lambda_{\min}(\Theta_0) \ge \lambda_0 > 0$。
2. **惰性训练下的核稳定性**（Lazy Training）：当隐藏层宽度 $m \to \infty$（或 $m \ge \operatorname{poly}(n, L, \delta^{-1})$）且采用 $\frac{1}{\sqrt{m}}$ 缩放时，为了将 $n$ 个样本的残差从 $O(1)$ 降为零，分摊到海量参数上的单个权重位移极小（$\|\theta_t - \theta_0\|_2$ 相对于核变化尺度可忽略）。因此雅可比矩阵在整个训练期内几乎保持在初始切平面上（$J(\theta_t) \approx J(\theta_0)$），使得核矩阵稳定不退化（$\|\Theta_t - \Theta_0\|_2 \le \frac{\lambda_0}{4}$），从而确保 $\lambda_{\min}(\Theta_t) \ge \frac{\lambda_0}{2}$ 全程成立。[Nitta 2018/2022][nitta-2018] 也据此证明，在无限宽 NTK 极限下，深层 ReLU 网络的梯度下降轨迹绝不会落入任何伪局部极小值。

## 负面定理与反例：当有限宽非线性网络遇上真实局部陷阱

正面理论的辉煌并不意味着我们可以草率地宣称“神经网络里根本没有坏局部极小值”。恰恰相反，当网络跳出无限宽 NTK 极限、回到宽度有限且无特殊跳连的常规 ReLU 架构时，2018 年的两项重量级理论工作证实：**坏局部极小值在非线性网络中不仅真实存在，而且在恰好参数化（Exact Parameterization）时具有巨大的吸引域**。

### Safran & Shamir 2018：两层 ReLU 网络中普遍存在的伪局部极小值

为了排除“数据分布病态”或“网络容量不足无法拟合目标函数”的干扰，[Safran & Shamir 2018][safran-shamir-2018]（*Spurious Local Minima are Common in Two-Layer ReLU Neural Networks*）考察了最纯净、最理想的**教师-学生**（Teacher-Student）无噪实现设定：
- 输入服从最温和的 $d$ 维标准高斯分布 $x \sim \mathcal{N}(0, I_d)$；
- 真实标签由一个含 $k$ 个正交单位权重向量 $\{v_1, \dots, v_k\}$（$\langle v_i, v_j\rangle = \delta_{ij}$）的教师两层 ReLU 网络精确生成；
- 学生网络拥有完全相同的 $k$ 个隐藏层 ReLU 神经元，训练目标为最小化总体平方风险（Population Risk）：

$$
L(w_1,\dots,w_k) = \frac{1}{2}\mathbb{E}_{x\sim\mathcal{N}(0,I_d)}\left[\left(\sum_{i=1}^{k}\max\{0, w_i^\top x\} - \sum_{j=1}^{k}\max\{0, v_j^\top x\}\right)^2\right].
$$

由于总体期望下的高斯 ReLU 核具有显式解析闭式解：

$$
\mathbb{E}_{x\sim\mathcal{N}(0,I_d)}\big[\max\{0, u^\top x\}\max\{0, v^\top x\}\big] = \frac{\|u\|_2\|v\|_2}{2\pi}\big(\sin\theta_{uv} + (\pi - \theta_{uv})\cos\theta_{uv}\big),
$$

其中 $\theta_{uv} = \arccos\frac{u^\top v}{\|u\|_2\|v\|_2}$，显然当学生神经元精确匹配教师神经元（$\{w_i\} = \{v_j\}$）时，全局最小损失严格为 $L_{\min} = 0$。

然而，Safran & Shamir 结合区间算术与计算机辅助严格证明（Computer-Assisted Proof）证实：**只要神经元个数 $k \ge 6$，在任意输入维度 $d \ge k$ 下，该总体风险曲面上都必然存在局部海森矩阵严格正定且损失严格大于零的坏局部极小值**（即 $\lambda_{\min}(\nabla^2 L) > 0,\ L > 0$）。

这些坏局部极小值是怎么长出来的？其物理图像是**对称性破缺下的神经元错配与分裂**：在随机初始化后，假设 $k=6$ 个学生神经元在竞争吸附 $6$ 个教师方向时，有 $2$ 个学生神经元共同坍缩到了同一个教师向量 $v_1$ 上（各自承担 $\frac{1}{2}v_1$），而剩下的 $4$ 个学生神经元被迫去分摊其余 $5$ 个正交教师向量 $\{v_2, \dots, v_6\}$（形成某种对称折中方向）。此时，由于 ReLU 在零点处的非光滑半空间截断效应，任何单个神经元试图脱离当前局部平衡去修补未覆盖的教师方向，都会在局部二阶邻域内立刻招致更大的平方误差反弹，从而被海森正定势垒死死锁在次优构型中。

更值得深思的是他们的配套动力学实验：当学生网络宽度恰好等于教师宽度（$m = k=10$，恰好参数化）时，从标准随机初始化出发运行梯度下降，**陷入上述坏局部极小值的失败率高达 $90\%$ 以上**；然而，一旦将学生网络的神经元数略微放宽至轻度过参数化（如 $m = 2k = 20$），多余的自由度允许每个教师方向至少被一个学生神经元捕获，原本封闭的错配陷阱瞬间瓦解，梯度下降收敛至全局零误差的成功率立刻跃升至接近 $100\%$。

### Du et al. 2018：静态存在坏极小值，但归一化动力学仍能高概率绕开

如果不能无限加宽网络，静态坏局部极小值的存在是否就宣判了梯度下降的死刑？[Du et al. 2018][du-2018]（*Gradient Descent Learns One-hidden-layer CNN: Don't be Afraid of Spurious Local Minima*）给出了否定的回答，深刻阐明了**静态存在性与动态吸引域的分离**。

他们分析了带共享卷积核 $w \in \mathbb{R}^d$ 与输出池化权重 $a \in \mathbb{R}^k$ 的单隐藏层非重叠卷积 ReLU 网络 $f_{w,a}(x) = \sum_{j=1}^k a_j \max\{0, w^\top x_{[j]}\}$。论文证明了两面互补的结论：
1. **静态景观确实有坑**：在高斯输入与真实教师参数 $(w^\star, a^\star)$ 下，损失曲面上确实存在与真实卷积核方向相反（$w \approx -w^\star$ 的局部扭曲投影）、且局部没有下降方向的**坏局部极小值**。
2. **动态轨迹高概率绕坑**：当对卷积核引入**权重归一化**（Weight Normalization，将方向 $v = \frac{w}{\|w\|_2}$ 与模长解耦）并从对称随机初始化出发运行梯度下降时，参数空间被一条由初始残差符号与角度决定的分水岭一分为二——坏局部极小值的吸引域被压缩在特定象限内，而随机初始化有**严格不低于 $1/4$ 的常数概率**直接落在通往全局最优解 $(w^\star, a^\star)$ 的单调吸引盆地中（只需独立随机初始化运行寥寥数次取训练损失最低者，就能以 $1 - (3/4)^R \to 1$ 的概率完美恢复全局最优）。

把 Safran & Shamir (2018) 与 Du et al. (2018) 放在一起对比，我们对局部极小值的认知便从非黑即白的二元对立走向了定量几何：坏局部极小值在有限宽 ReLU 网络中客观存在，但**过参数化通过增加冗余神经元打通了下降维度，而归一化与良好初始化则大幅挤压了次优陷阱的吸引域体积**（Basin Volume）。

## 四类消解机制全景对照与平坦极小值的重参数化争议

综合前三十年正反两方面的理论突破，神经网络中“坏局部极小值消失或失效”的数学根源可以清晰归纳为以下四类机制：

| 机制分类 | 核心数学原理 | 代表性工作 | 适用边界与未覆盖情形 |
| --- | --- | --- | --- |
| **代数矩阵分解机制** | 多层线性乘积虽非凸，但矩阵因子分解的代数冗余保证了任何非全局驻点均存在下降方向 | [Baldi & Hornik 1989][baldi-hornik-1989]、[Kawaguchi 2016][kawaguchi-2016]、[Lu & Kawaguchi 2017][lu-kawaguchi-2017]、[Haeffele & Vidal 2015][haeffele-vidal-2015] | 仅适用于深度线性网络或满足特定正齐次/张量分解简化假设的模型，无法直接推广至一般有限宽 ReLU 网络 |
| **宽层特征满秩插值机制** | 某一隐藏层宽度超过样本数 $n$ 时，样本激活矩阵满秩，使下游金字塔层可在表示空间内无阻碍地将残差压至零 | [Yu & Chen 1995][yu-chen-1995]、[Soudry & Carmon 2016][soudry-carmon-2016]、[Nguyen & Hein 2017][nguyen-hein-2017]、[Lederer 2020][lederer-2020] | 依赖层宽 $d_k \ge n$、解析激活及后续层非递增满秩假设，且仅保证训练误差极小，不直接给出测试泛化界 |
| **拓扑跳连与恒等参数化** | 通过残差恒等路径消除零点退化鞍点，或借直连输出的特殊神经元为所有次优参数点构造性注入局部下降方向 | [Hardt & Ma 2016][hardt-ma-2016]、[Liang et al. 2018][liang-2018] | 依赖特定残差范数约束或特殊构造的直连指数/多项式神经元 |
| **初始化邻域动态核化**（NTK） | 极宽网络在随机初始化附近单个参数位移极小，一阶泰勒线性化全程成立，非凸训练轨迹退化为正定核矩阵驱动的线性收敛 | [Jacot et al. 2018][jacot-2018]、[Allen-Zhu et al. 2018][allen-zhu-2018]、[Nitta 2018/2022][nitta-2018] | 要求多项式级超大宽度且局限于惰性核训练区（Lazy Regime），未能刻画有限宽大模型中的显著特征表示演化（Feature Learning） |

### 训练全局最优之后：平坦极小值（Flat Minima）与尺度变换悖论

当过参数化帮助优化器轻松跨越次优陷阱、进入训练损失全部为零（$L(\theta^\star)=0$）的全局极小值连通山谷流形后，一个新的核心问题浮出水面：**既然流形上有无穷多个参数点都能让训练误差归零，为什么 SGD 找到的全局极小值测试泛化极好，而某些人工构造或大 Batch 优化找到的全局极小值却严重过拟合？**

围绕这一问题，[Keskar et al. 2016][keskar-2016] 提出了影响深远的经验假说：小批量（Small-Batch）SGD 中的内生梯度噪声与各向异性协方差，会驱动参数逃离曲率极大的**尖锐极小值**（Sharp Minima），滑入海森特征值平缓的**平坦极小值**（Flat Minima）；而平坦极小值对训练集与测试集之间的分布偏移更不敏感，因而具备更强的泛化能力。

然而，[Dinh et al. 2017][dinh-2017]（*Sharp Minima Can Generalize For Deep Nets*）旋即给出了一个直击要害的数学反例，证明**脱离参数范数与重参数化不变性去空谈欧氏空间的海森曲率是毫无意义的**。

考察一个两层无偏置 ReLU 网络 $f_{(W_1, W_2)}(x) = W_2 \max\{0, W_1 x\}$。由于 ReLU 激活函数满足正齐次性（Positive Homogeneity，即对任意正实数 $\alpha > 0$，有 $\max\{0, \alpha z\} = \alpha \max\{0, z\}$），我们可以对两层权重施加如下尺度重参数化变换（Scale Reparameterization）：

$$
T_\alpha : (W_1, W_2) \longmapsto (\tilde W_1, \tilde W_2) = (\alpha W_1,\ \alpha^{-1} W_2), \qquad \alpha > 0.
$$

对于输入空间中的任意测试样本 $x$，变换后的网络输出函数与原网络**逐点严格恒等**：

$$
f_{T_\alpha(W_1,W_2)}(x) = (\alpha^{-1} W_2)\max\{0, \alpha W_1 x\} = \alpha^{-1}\alpha W_2\max\{0, W_1 x\} = f_{(W_1,W_2)}(x).
$$

既然两者的输入-输出映射在全空间上分毫不差，它们的训练损失与**真实测试泛化误差自然也 $100\%$ 完全相同**。但是，如果我们计算变换后参数点 $\tilde\theta = T_\alpha(\theta^\star)$ 处的二阶海森矩阵，记对角缩放矩阵为 $D_\alpha = \operatorname{diag}(\alpha I_{d_1}, \alpha^{-1} I_{d_2})$，由链式法则立得海森矩阵的合同变换关系：

$$
\nabla^2_{\tilde\theta} L(\tilde\theta) = D_\alpha^{-1}\,\nabla^2_{\theta} L(\theta^\star)\,D_\alpha^{-1}.
$$

其直接后果是：原本关于 $W_1$ 的二阶偏导块被缩小为 $\alpha^{-2}\nabla^2_{W_1} L$，而关于 $W_2$ 的二阶偏导块则被放大为 $\alpha^2 \nabla^2_{W_2} L$。只要原极小值点处 $\nabla^2_{W_2} L \neq 0$，我们只需令 $\alpha \to +\infty$（或 $\alpha \to 0^+$），就能在**完全不改变网络函数与泛化性能**的前提下，让海森矩阵的迹 $\operatorname{Tr}(\nabla^2 L)$ 与最大特征值 $\lambda_{\max}(\nabla^2 L)$ 飙升至无穷大（把一个“平坦极小值”变成任意“尖锐”的极小值）。

Dinh 等人的这一推导深刻警醒学界：朴素的欧氏海森特征值受制于重参数化尺度伪影；要想用损失景观的几何平坦度解释泛化，必须采用具有重参数化不变性的相对曲率度量（如归一化锐度 $\frac{\|\theta\|_2^2 \lambda_{\max}(\nabla^2 L)}{L}$、Fisher 信息矩阵度规或 PAC-Bayes 边缘体积），或是直接剖析有限步长 SGD、BatchNorm 与权重衰减在等效函数空间中施加的**隐式正则化偏置**（Implicit Bias）。

## 总结：从静态局部陷阱走向高维优化动力学

回望从 1989 年至过参数化时代的完整理论演进，学界对神经网络坏局部极小值的认识已经完成了三次范式级的跃迁：

1. **从笼统的“非凸即困难”到精细的代数与非线性解耦**：深度线性网络定理证明了单纯堆叠矩阵乘法绝不产生坏局部极小值（仅引入高阶退化鞍点）；而非线性激活在欠参数化或恰参数化宽度下（如有限宽两层与卷积 ReLU 网络）确实会通过神经元对称错配制造出海森严格正定的伪局部极小值。
2. **从“单点孤立陷阱”到“高维相变与连通山谷流形”**：在高维参数空间中，高于全局最小值的临界点呈指数级压倒性比例表现为具有负曲率逃逸通道的鞍点；而当网络宽度跨越样本秩阈值、引入残差恒等跳连或进入过参数化区间后，孤立的次优盆地被打通，全局最优解连成一片可通过简单曲线无壁垒穿行的连通流形（Mode Connectivity）。
3. **从“静态景观存在性”到“算法动力学与隐式偏置”**：现代深度学习优化的核心追问，早已跨越了早期“误差曲面上是否存在局部极小值”的静态争论，全面转向更具操作性的动力学命题——权重归一化与初始化尺度如何将坏极小值的吸引域体积压缩至零测度？自适应优化器与残差结构如何消除高阶退化鞍点的潜伏平台期？在无穷多个零训练误差的全局极小值流形上，有限步长 SGD 与特征学习机制（Feature Learning beyond NTK）又是如何导航至具备重参数化不变平坦性的高泛化解？

这正是连接经典反向传播理论与当代大模型优化理论的演进主线。

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

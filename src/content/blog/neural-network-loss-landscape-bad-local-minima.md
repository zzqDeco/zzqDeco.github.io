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

神经网络训练经常被一句话概括为“非凸优化”。这句话没有错，但它太粗糙。非凸意味着损失函数可能有局部极小值、鞍点、平坦区域、病态曲率和大量等价参数流形；它不自动意味着梯度下降一定会掉进坏局部极小值，也不意味着训练成功就等价于找到了某个唯一的全局最优点。

这篇报告要讨论的问题更窄，也更关键：**神经网络损失景观中的坏局部极小值，到底是不是反向传播训练的主要障碍？**

从早期 Backpropagation 文献到深度线性网络，再到宽网络、残差结构、NTK regime 和过参数化理论，相关结论并不是一条简单的“有”或“没有”的线。更准确的图景是：非凸不必然推出坏局部极小值；反过来，某些模型没有坏局部极小值，也不代表一般神经网络训练没有困难。

坏局部极小值是损失景观理论中的一个核心对象，但现代深度学习优化的困难往往还来自鞍点、平坦谷、病态曲率、初始化依赖、随机梯度动态、宽度 regime 与泛化之间的差异。

## 1. BP、梯度训练与损失景观

给定训练集

$$
\mathcal{D}=\{(x_i,y_i)\}_{i=1}^{n},
$$

监督学习中的经验风险通常写作：

$$
L(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell(f_{\theta}(x_i),y_i),
$$

其中 $f_{\theta}$ 是由参数 $\theta$ 定义的神经网络，$\ell$ 是样本级损失，例如平方损失或交叉熵损失。一个 $L$ 层前馈网络可以抽象为：

$$
f_\theta(x)=W_L\sigma(W_{L-1}\sigma(\cdots\sigma(W_1x+b_1)\cdots)+b_{L-1})+b_L.
$$

即使 $\ell$ 对预测值是凸函数，$L(\theta)$ 对参数 $\theta$ 通常也不是凸函数。原因在于参数不是线性地进入目标函数，而是通过多层矩阵乘积和非线性激活复合在一起。多层矩阵乘积加非线性激活，通常会形成高度非凸的参数空间目标函数。

这里要先区分三个容易混在一起的概念。

第一，**Backpropagation**，简称 BP，是高效计算梯度的算法。它使用链式法则计算：

$$
\nabla_{\theta}L(\theta).
$$

第二，**gradient-based training** 是基于这些梯度更新参数的训练过程。例如普通梯度下降为：

$$
\theta_{t+1}=\theta_t-\eta\nabla_{\theta}L(\theta_t).
$$

第三，早期文献中的 **BP learning** 常常把“反向传播计算梯度”和“梯度下降更新参数”合在一起讨论。因此 Gori & Tesi、Yu、Yu & Chen 等论文讨论 BP 是否会陷入局部极小值时，讨论的通常不是自动微分本身，而是整个基于 BP 的梯度训练过程。

从计算图角度看，前向传播先逐层计算：

$$
a^0=x,\quad z^\ell=W^\ell a^{\ell-1}+b^\ell,\quad a^\ell=\sigma(z^\ell),
$$

最后得到 $\hat y=f_\theta(x)$。反向传播再从输出层开始传播误差信号：

$$
\delta^\ell=(W^{\ell+1})^\top \delta^{\ell+1}\odot \sigma'(z^\ell),
$$

并得到权重梯度：

$$
\frac{\partial L}{\partial W^\ell}
=
\delta^\ell(a^{\ell-1})^\top.
$$

所以 BP 的核心承诺是高效给出正确梯度，而不是保证梯度训练找到全局最优。

这也解释了为什么 [Baldi & Sadowski 2016][baldi-sadowski-2016] 从 learning channel 的角度理解 BP：深层参数距离输出很远，如果目标误差信息不能有效回传，深层网络就无法解决 credit assignment 问题。BP 是一种高效的 backward learning channel，但它本身不是全局优化定理。

## 2. 坏局部极小值到底是什么

如果

$$
L(\theta^\star)\le L(\theta),\quad \forall \theta,
$$

则 $\theta^\star$ 是全局极小值。如果存在某个邻域 $\mathcal{N}(\theta^\star)$，使得

$$
L(\theta^\star)\le L(\theta),\quad \forall \theta\in\mathcal{N}(\theta^\star),
$$

则 $\theta^\star$ 是局部极小值。

**坏局部极小值**指的是局部上不能下降、但全局上仍然不是最优的点：

$$
\theta^\star\text{ is local minimum},\quad L(\theta^\star)>L_{\min}.
$$

英文文献中常见的说法包括 bad local minimum、poor local minimum 和 spurious local minimum。中文可以译为坏局部极小值、次优局部极小值或伪局部极小值。很多论文中的 “no bad local minima” 并不是说损失函数没有任何局部极小值，而是说：

$$
\text{every local minimum is global}.
$$

这个性质不等价于凸性。一个函数可以高度非凸，但所有局部极小值仍然是全局极小值。例如一维函数可以有弯曲的非凸形状，却没有真正的次优陷阱。

还需要区分局部极小值和临界点。如果

$$
\nabla L(\theta)=0,
$$

则 $\theta$ 是临界点。临界点可以是局部极小值、局部极大值，也可以是鞍点。鞍点在某些方向上看似稳定，在另一些方向上仍然可以下降。对于 Hessian 矩阵 $H=\nabla^2 L(\theta)$，如果存在负特征值，二阶局部近似就给出了下降方向；如果 Hessian 没有负特征值但该点仍不是局部极小值，优化器反而可能更难逃离。

因此，“没有坏局部极小值”只排除了其中一种失败模式。它不排除：

- 梯度非常小的鞍点；
- 大量平坦方向；
- Hessian 条件数很差导致的慢收敛；
- SGD 噪声和学习率调度带来的动态效应；
- 训练误差低但测试误差差的泛化问题。

这也是后面所有结果都需要谨慎表述的原因。

## 3. 早期 BP 文献：失败模式与充分条件

早期关于 BP 的讨论有两条看似相反、实际互补的路线。一条路线强调 BP 作为梯度训练可能失败；另一条路线寻找让 error surface 没有坏局部极小值的特殊条件。

### 3.1 Gori & Tesi 1992：BP 可以被局部极小值困住

[Gori & Tesi 1992][gori-tesi-1992] 的论文题为 *On the Problem of Local Minima in Backpropagation*。这篇论文的价值在于明确指出：BP 训练不是全局优化黑盒，确实可以构造出梯度训练陷入局部极小值、而网络尚未学完整个训练集的情形。

这不是说 BP 总会失败，而是在纠正一种过度乐观理解：BP 可以训练神经网络，不等于 BP 总能到达全局最优。

梯度下降只沿局部梯度方向移动。在非凸 error surface 上，局部几何结构、学习率、初始化、网络结构和数据可分性都会影响训练路径。

### 3.2 Yu 1992 与 Yu & Chen 1995：能否没有局部极小值

[Yu 1992][yu-1992] 的题目直接提出问题：*Can backpropagation error surface not have local minima?* 这篇短文的重要性不在于给出一般网络的终极答案，而在于把问题转成了结构性问题：是否存在某些网络和数据条件，使 BP error surface 没有坏局部极小值？

[Yu & Chen 1995][yu-chen-1995] 进一步给出两层前馈网络中的 local-minima-free 条件。根据论文摘要，该结果考虑 $P$ 个 noncoincident input patterns，即输入样本互不重合，并使用 $P-1$ 个 sigmoid hidden neurons 加一个 dummy hidden neuron 来得到充分条件。后续还存在 [1996 年 correction][yu-chen-correction]，所以引用时必须把它理解为特定结构下的充分条件，而不是一般 BP 网络无坏局部极小值的定理。

这个结果从现代视角看很有意思：隐藏单元数量和样本数同阶，已经带有早期过参数化思想。它暗示网络自由度增加后，损失景观可能发生结构性改变。

### 3.3 早期争论留下的问题

Gori & Tesi 说明一般 BP 训练可以被局部结构困住；Yu 与 Yu & Chen 则说明在特定条件下可以寻找局部极小值自由的结构。两者不矛盾。它们共同留下了后续三十年一直在深化的三个问题：哪些网络结构会产生坏局部极小值，哪些结构或宽度条件会消除它们，以及即使没有坏局部极小值，梯度训练是否仍可能很慢。

## 4. 深度线性网络：非凸但没有坏局部极小值

深度线性网络去掉激活函数，只保留层级矩阵乘积：

$$
f_\theta(x)=W_LW_{L-1}\cdots W_1x.
$$

从输入输出函数看，它等价于一个线性模型 $f(x)=Wx$，其中 $W=W_L\cdots W_1$。但从参数空间看，它仍然是非凸的，因为 $L(W_1,\ldots,W_L)$ 依赖多层矩阵乘积。

这使深度线性网络成为一个干净的理论基准：如果只有“深度”和“矩阵分解”，没有非线性，损失景观是否还会出现坏局部极小值？

### 4.1 Baldi & Hornik 1989：线性网络与 PCA

[Baldi & Hornik 1989][baldi-hornik-1989] 研究了线性前馈网络与主成分分析之间的关系。在自关联线性网络中，二次误差函数的最优解对应由主特征向量张成的子空间，其他临界点对应较差特征方向上的投影，表现为鞍点而不是坏局部极小值。

这篇论文是后续深度线性网络结果的重要前史：即使模型以神经网络参数化呈现，只要结构足够特殊，error surface 可以具有“局部极小值良性、困难主要在鞍点”的性质。

### 4.2 Saxe et al. 2013：没有坏局部极小值不等于训练快

[Saxe, McClelland & Ganguli 2013][saxe-2013] 给出了深度线性网络学习动态的精确解，解释了深层线性系统中的阶段式学习、平台期和奇异模式逐步学习现象。

它提醒我们：即便损失景观没有坏局部极小值，训练轨迹也可能长时间处于缓慢阶段。原因不一定是局部陷阱，而可能是不同方向的学习速度差异、初始化尺度、深度导致的动态耦合和曲率病态。

### 4.3 Kawaguchi 2016：Deep Learning without Poor Local Minima

[Kawaguchi 2016][kawaguchi-2016] 是这条理论线中的关键节点。论文先在平方损失下研究任意深度、任意宽度的深度线性网络，并证明：

- $L(\theta)$ 是非凸且非凹的；
- every local minimum is a global minimum；
- every critical point that is not global is a saddle point。

同时，深层线性网络中可能存在 bad saddle points，即 Hessian 没有负特征值的鞍点。这一点很重要：坏局部极小值被排除后，困难并没有消失，而是转移到了鞍点、平坦区域和训练动态上。

Kawaguchi 还在额外 independence assumption 下把类似结论约化到某些非线性网络情形。但这个扩展不能被误读为“一般非线性深度网络都没有坏局部极小值”。论文自己也明确承认理论和实践之间仍有距离。

### 4.4 Lu & Kawaguchi 2017：深度本身不制造坏局部极小值

[Lu & Kawaguchi 2017][lu-kawaguchi-2017] 把问题表述得更尖锐：深度和非线性都会制造非凸损失面，那么 **depth alone** 是否会制造坏局部极小值？

他们的回答是：没有非线性时，深度本身不会制造坏局部极小值，虽然它确实会制造非凸性。可以把这个结论写成：

$$
\text{depth} \Rightarrow \text{non-convexity},
$$

但不能写成：

$$
\text{depth} \Rightarrow \text{bad local minima}.
$$

这改变了对“深度网络难训练”的粗糙解释。困难不能简单归因于层数多，而要分解为深度、非线性、有限宽度、数据几何、初始化、优化器和损失函数的共同作用。

[Yun, Sra & Jadbabaie 2018][yun-2017] 进一步研究深度线性和非线性网络的全局最优条件，为临界点是否全局最优提供了更细的判别视角。这类工作共同说明：深度线性网络不是现实网络的完整替代品，但它是拆分“深度”和“非线性”的必要基准。

还有一条相邻路线来自非凸分解问题。[Haeffele & Vidal 2015][haeffele-vidal-2015] 从矩阵分解、张量分解和深度学习训练的共同结构出发，研究什么时候局部最优可以推出全局最优。这类结果并不直接覆盖所有常规网络，但它把神经网络的 loss landscape 放进了更一般的“过参数化因子分解”框架中：当目标函数和正则项具有合适的正齐次或可分解结构时，非凸参数化未必破坏全局最优性。

## 5. 高维损失景观：鞍点、平坦谷和经验几何

深度线性网络给出的是理论基准。真实网络更复杂，尤其在高维参数空间中，直觉常常失效。

### 5.1 Dauphin et al. 2014：鞍点可能比局部极小值更重要

[Dauphin et al. 2014][dauphin-2014] 的核心观点是：在高维非凸优化中，局部极小值不是唯一也不一定是主要困难；鞍点可能更普遍，也更能解释训练缓慢。

高维空间中，一个临界点要成为严格局部极小值，需要 Hessian 的所有方向都非负；而成为鞍点只需要存在正负混合方向。这使得大量临界点表现为鞍点或近似鞍点。训练轨迹如果进入梯度很小、曲率复杂的区域，普通梯度法会明显变慢。

这和 Kawaguchi 的深度线性结论相互呼应：即使没有坏局部极小值，bad saddle points 仍可能是优化困难的来源。

后续一般非凸优化理论进一步把这个观点形式化。[Lee et al. 2016][lee-2016] 证明，在一定光滑性和步长条件下，随机初始化的梯度下降几乎必然不会收敛到 strict saddle，也就是 Hessian 存在负特征值的鞍点。[Jin et al. 2017][jin-2017] 则说明，加入适当随机扰动后，一阶方法可以高效逃离鞍点并找到二阶驻点。这里的关键限定是 strict saddle：如果鞍点退化、平坦方向很多，或者 Hessian 负曲率非常弱，实际训练仍可能出现很长的平台期。

### 5.2 Choromanska et al. 2015：随机矩阵与低能临界点

[Choromanska et al. 2015][choromanska-2015] 用 spin-glass 和随机矩阵理论近似分析多层网络损失面。论文依赖一些强假设，例如变量独立、参数冗余和均匀性，因此不能被当作真实网络的严格全局定理。

但它提出了一个影响很大的直觉：大规模网络中，低损失临界点可能形成一个靠近全局最小值的“低能带”，差局部极小值在大网络中不一定是主要训练障碍。这个图景强化了一个认识：现代大网络训练成功未必是因为损失面简单，而可能是因为高维冗余让大量可达解都足够好。

### 5.3 Goodfellow et al. 2015：线性插值实验

[Goodfellow, Vinyals & Saxe 2015][goodfellow-2015] 用一组简单实验考察神经网络优化路径。例如沿初始化参数和训练后参数之间的线性路径观察损失变化，发现很多设置下路径上的损失并没有出现想象中的巨大障碍。

这不是一个证明“没有坏局部极小值”的定理。它更像是一个经验信号：真实训练遇到的问题可能不总是离散的“深坑”，而可能是曲率、尺度、鞍点和动态路径问题。

### 5.4 可视化与 mode connectivity

[Li et al. 2018][li-2018] 提出 filter normalization 等方法来可视化神经网络损失景观，并展示架构和训练设置会显著改变局部几何，例如残差连接往往让可视化景观看起来更平滑。可视化不能完整还原高维空间，但可以帮助理解不同训练设置的几何差异。

另一个重要经验现象是 mode connectivity。[Garipov et al. 2018][garipov-2018] 发现不同训练得到的 optima 可以由简单曲线连接，并且曲线上的训练和测试性能几乎保持稳定。[Draxler et al. 2018][draxler-2018] 也报告了不同 minima 之间可以存在几乎无 barrier 的连续路径。

这些结果把“局部极小值”从孤立点改写成更像高维连通谷地的图景：在现代过参数化网络中，很多好解可能由低损失路径连接，而不是彼此隔离在不同盆地中。

它们仍然是经验和特定架构下的发现，不应替代严格定理；但它们解释了为什么多个随机种子得到的模型可能处在不同参数点，却有相近性能。

## 6. 非线性网络与过参数化正面结果

深度线性网络说明“深度本身不必然制造坏局部极小值”。真正困难的问题是：**含非线性激活的网络，在什么条件下也能排除坏局部极小值？**

这类正面结果通常依赖至少一种条件：足够宽的层、特殊结构、skip connection、特定损失函数、特定数据条件、特定初始化，或者 NTK 极限。

### 6.1 Soudry & Carmon 2016：分段线性网络的训练误差保证

[Soudry & Carmon 2016][soudry-carmon-2016] 使用 smoothed analysis 研究多层网络在可微局部极小值处的训练误差。论文考虑 piecewise linear activation、quadratic loss、single output 和 mild over-parameterization，并证明一层隐藏网络在几乎所有数据集和 dropout-like noise realization 下，每个可微局部极小值训练误差为零；随后扩展到多隐藏层情形。

这个结果的重点不是“任意 ReLU 网络都无坏局部极小值”，而是说明在轻度过参数化与扰动分析框架下，局部极小值可以具有很强的训练误差保证。

### 6.2 Nguyen & Hein 2017：深而宽网络的 almost all local minima

[Nguyen & Hein 2017][nguyen-hein-2017] 研究平方损失、解析激活函数的全连接网络。核心条件是某一隐藏层的神经元数量大于训练样本数，并且从这一层之后的网络结构是 pyramidal。论文结论是 almost all local minima are globally optimal。

这里的条件非常具体：某一隐藏层宽度大于样本数 $n$，这一层之后的网络是 pyramidal structure，并且分析使用平方损失和解析激活函数。

它属于宽网络理论的正面结果，但不是对任意宽网络、任意激活函数、任意损失函数的无条件保证。

### 6.3 Hardt & Ma 2016：identity parameterization 与残差结构

[Hardt & Ma 2016][hardt-ma-2016] 把残差网络背后的一个设计原则理论化：每一层应该能容易表达 identity transformation。论文证明任意深的线性残差网络没有 spurious local optima；同时展示 ReLU 残差网络在参数数超过样本数时具有有限样本上的 universal expressivity。

这和工程中的 ResNet 经验相互呼应。残差结构不仅改善梯度流，也改变参数化方式，使模型更容易从“接近恒等映射”的位置开始学习增量函数：

$$
h_{\ell+1}=h_\ell + F_\ell(h_\ell).
$$

但要注意，线性 residual 网络无 spurious local optima 不等于一般非线性 ResNet 全局无坏局部极小值。它说明的是 identity parameterization 这个结构原则为何有利于优化。

### 6.4 Liang et al. 2018：一个特殊神经元可以消除坏局部极小值

[Liang et al. 2018][liang-2018] 的题目很有冲击力：*Adding One Neuron Can Eliminate All Bad Local Minima*。论文研究二分类任务，并证明在温和假设下，添加一个带 skip connection 到输出的特殊神经元，或者每层添加一个特殊神经元后，每个局部极小值都是全局极小值。

关键点是“特殊神经元”，不是随便往工程模型里多加一个普通神经元。这个额外自由度提供了非全局点附近的下降方向：

$$
L(\theta)>L_{\min}
\quad\Longrightarrow\quad
\exists \Delta\theta,\ L(\theta+\Delta\theta)<L(\theta).
$$

只要这样的下降方向存在，该点就不可能是局部极小值。这条结果说明，坏局部极小值不是不可改变的拓扑宿命，结构改造可以直接改变损失景观。

### 6.5 NTK regime：线性化动态与无限宽网络

NTK，即 Neural Tangent Kernel，由 [Jacot, Gabriel & Hongler 2018][jacot-2018] 系统提出。在无限宽或极宽网络中，如果参数初始化合适、训练过程中参数移动很小，网络函数可以在初始化附近被一阶线性化：

$$
f_{\theta}(x)
\approx
f_{\theta_0}(x)
+
\nabla_\theta f_{\theta_0}(x)^\top(\theta-\theta_0).
$$

对应的核为：

$$
K(x,x')
=
\nabla_\theta f_{\theta_0}(x)^\top
\nabla_\theta f_{\theta_0}(x').
$$

在这种 regime 中，训练动态接近核回归或线性模型优化，因此更容易得到收敛和全局优化结论。

[Nitta 2018/2020/2022][nitta-2018] 的当前 arXiv 版本明确把结论限定在 NTK regime：参数正态初始化、隐藏层宽度趋于无穷、梯度下降训练动态下，deep ReLU networks do not lie in spurious local minima。这个限定非常重要。它不是说任意有限宽 ReLU 网络的整个损失面都没有坏局部极小值，而是说在 NTK 极限和特定动态条件下不会落入 spurious local minima。

### 6.6 过参数化收敛理论：有限宽但需要足够宽

NTK 不是只在无限宽极限中有意义。一个重要研究方向是证明有限但充分宽的网络，在随机初始化附近训练时也近似保持 NTK 行为。[Allen-Zhu, Li & Song 2018/2019][allen-zhu-2018] 的 *A Convergence Theory for Deep Learning via Over-Parameterization* 就属于这条路线：当网络宽度是样本数、深度和精度参数的多项式量级时，梯度下降可以把训练误差降到很低，并和 NTK 描述建立联系。

这类结果的价值在于把“无限宽直觉”拉回有限网络，但它的代价也很清楚：宽度要求通常远大于日常工程模型的实际规模，证明关注的也是初始化附近的 lazy training 或 kernel-like regime。也就是说，它解释了一类过参数化网络为什么可优化，却不完整解释现代大模型中显著的 feature learning。

因此，NTK/过参数化收敛理论提供的是一个可靠但有限的基线：足够宽的网络、合适初始化和小参数移动，可以让训练动态近似凸化或核化。

它不应被理解成“所有大网络都是核方法”，也不应被理解成“只要参数多就一定泛化好”。

### 6.7 Lederer 2020：宽深网络的无伪局部极小值

[Lederer 2020][lederer-2020] 分析宽深网络经验风险优化景观，并证明在其理论设定下，约束和非约束 empirical-risk minimization 都没有 spurious local minima。OpenReview 摘要也强调，这支持了一个常见经验观察：增加网络宽度不仅提升表达能力，也可能促进优化。

宽度的直觉可以概括为：更多参数自由度带来更多可用下降方向，从而让次优局部陷阱更难稳定存在。

不过，这仍然是特定理论设定下的结果。宽度有助于优化，不等于参数越多就自动解决所有训练和泛化问题。

## 7. 负面结果：非线性网络仍可能有 spurious local minima

如果只看上面的正面结果，容易得到错误印象：现代网络大概都没有坏局部极小值。负面结果正是为了阻止这种过度外推。

### 7.1 Safran & Shamir 2018：两层 ReLU 网络中 spurious local minima 很常见

[Safran & Shamir 2018][safran-shamir-2018] 研究简单两层 ReLU 网络：

$$
x\mapsto \sum_{i=1}^{k}\max\{0,w_i^\top x\},
$$

并在平方损失下给出计算辅助证明：即使输入分布是标准高斯，即使维度任意高，即使目标值由同类网络生成，当 $6\le k\le 20$ 时，问题仍可能有 spurious local minima。论文还指出，在高维中相关规模的 target networks 几乎都可能导致 spurious local minima；实验上，梯度法命中这些局部极小值的概率也不低。

但论文同时观察到，mild over-parameterization 会显著减少这类局部极小值。这和正面结果并不冲突，反而说明过参数化假设在某些 ReLU 场景中可能是必要的。

### 7.2 Du et al. 2018：有坏局部极小值，但梯度下降仍可能成功

[Du et al. 2018][du-2018] 研究一层隐藏的非重叠卷积 ReLU 网络。论文证明在 Gaussian input 下存在 spurious local minimizer；但同时，使用 weight normalization、随机初始化和梯度下降，仍能以常数概率恢复真实参数，多次重启可提高成功概率。

这给出了一个更细的观点：存在 spurious local minima 不代表梯度下降必然失败；反过来，没有坏局部极小值也不代表梯度下降一定快速成功。

损失景观的静态性质和优化算法的动态性质必须分开讨论。局部极小值是否存在、它们的吸引域有多大、随机初始化落入吸引域的概率、学习率和归一化如何改变轨迹，这些都是不同层面的问题。

## 8. 四类机制：为什么坏局部极小值会消失

把上述正面结果放在一起看，可以归纳出四类不同机制。它们经常被混为一谈，但技术含义并不相同。

**第一类是线性分解机制。** 深度线性网络的函数类本身是线性的，但参数化是多层矩阵乘积。坏局部极小值消失的核心并不是网络表达能力神奇，而是矩阵分解结构让非全局临界点暴露出可以降低损失的方向。Baldi & Hornik、Kawaguchi、Lu & Kawaguchi、Yun 等结果都属于这一类或与这一类紧密相关。

**第二类是宽层插值机制。** 如果某一隐藏层宽度超过样本数，网络可以在训练样本上形成足够丰富的表示矩阵。直观地说，后续层可以在这个表示空间中完成插值，局部最优如果没有达到全局训练最优，通常会有可调整的自由度继续下降。Yu & Chen、Nguyen & Hein、Soudry & Carmon 的一部分思想可以从这个角度理解。

**第三类是结构性下降方向机制。** Liang et al. 的特殊神经元和 skip connection、Hardt & Ma 的 identity parameterization 都是在改变网络参数化，使非全局点附近出现额外下降方向。它们的共同点不是“多一点参数”这么简单，而是新增自由度具有特定连接方式，能够绕过原损失面的局部稳定结构。

**第四类是动态线性化机制。** NTK 和过参数化收敛理论关注的是训练轨迹，而不是完整参数空间中的每个点。网络足够宽且初始化合适时，训练过程停留在初始化附近，函数变化近似一阶线性模型，于是优化动态接近核方法或凸问题。Nitta、Jacot、Allen-Zhu/Li/Song 等结果属于这类。

这四类机制回答的问题不同：

| 机制 | 改善的对象 | 典型结论 | 不能推出什么 |
| --- | --- | --- | --- |
| 线性分解 | 参数化后的静态景观 | 深度线性网络无坏局部极小值 | 一般非线性网络也无坏局部极小值 |
| 宽层插值 | 样本上的表示能力 | 某些宽网络 almost all local minima global | 任意宽网络、任意损失都成立 |
| 结构性下降方向 | 局部扰动方向 | 特殊神经元或残差结构消除次优陷阱 | 随便加参数就能消除陷阱 |
| 动态线性化 | 初始化附近训练轨迹 | GD 在 NTK regime 中收敛良好 | 真实训练一定没有 feature learning |

这个分类有助于避免把所有 “no bad local minima” 结果混成一个笼统口号。论文之间的差别往往就在这些机制的边界上。

## 9. 统一图景：各论文在说什么

可以把这条文献线整理成下面的对照表。

| 论文 | 模型或对象 | 核心结论 | 关键限制 |
| --- | --- | --- | --- |
| Gori & Tesi 1992 | 多层前馈网络与 BP | BP 训练可以陷入局部极小值，也存在某些收敛条件 | 早期 BP 理论，依赖具体结构和学习环境 |
| Yu 1992 | BP error surface | 提出误差曲面能否没有局部极小值的问题 | 不能理解成一般无局部极小值定理 |
| Yu & Chen 1995 | 两层 sigmoid 网络 | 在互不重合样本、特定隐藏层构造下给出 local-minima-free 条件 | 结构特定，且有后续 correction |
| Baldi & Hornik 1989 | 线性网络与 PCA | 线性自关联网络中非最优临界点对应鞍点 | 线性/PCA 设定 |
| Saxe et al. 2013 | 深度线性网络动态 | 给出精确学习动态，解释平台期和阶段式学习 | 线性网络，关注动态而非一般非线性景观 |
| Kawaguchi 2016 | 深度线性网络 | 非凸但所有局部极小值全局；非全局临界点为鞍点 | 非线性推广依赖 independence assumption |
| Haeffele & Vidal 2015 | 非凸分解与深度学习训练 | 给出一类可推出全局最优性的分解框架 | 需要特定结构和正则化条件 |
| Lu & Kawaguchi 2017 | 深度线性网络 | depth alone 不制造坏局部极小值 | 不覆盖一般非线性网络 |
| Dauphin et al. 2014 | 高维非凸优化 | 鞍点可能是高维训练困难的主要来源 | 更偏一般优化与经验分析 |
| Lee et al. 2016 / Jin et al. 2017 | strict saddle 优化理论 | GD 或扰动 GD 可以避免/逃离 strict saddle | 退化鞍点和平坦平台仍可能困难 |
| Choromanska et al. 2015 | 随机模型近似 | 大网络低能临界点可能集中在好损失带 | 依赖强近似假设 |
| Goodfellow et al. 2015 | 经验优化路径 | 训练路径线性插值常未出现巨大 barrier | 经验观察，不是全局定理 |
| Soudry & Carmon 2016 | 分段线性网络 | 轻度过参数化下可微局部极小值有训练误差保证 | 单输出、二次损失、扰动分析设定 |
| Nguyen & Hein 2017 | 深而宽全连接网络 | 某宽层大于样本数且后续 pyramidal 时，几乎所有局部极小值全局 | 平方损失、解析激活、结构条件 |
| Hardt & Ma 2016 | 线性残差网络 | 线性 residual 网络无 spurious local optima | 线性残差结论不能直接外推到任意 ResNet |
| Liang et al. 2018 | 特殊神经元与 skip | 加特殊神经元可消除所有坏局部极小值 | 特殊构造和二分类设定 |
| Nitta 2018/2022 | 深 ReLU + NTK regime | 无限宽、正态初始化、GD 动态下不落入 spurious local minima | NTK 极限和动态条件 |
| Allen-Zhu et al. 2018/2019 | 过参数化深网收敛 | 足够宽的有限网络可用 GD 收敛到低训练误差 | 宽度要求强，偏 lazy/NTK regime |
| Lederer 2020 | 宽深网络 ERM | 特定宽深网络 ERM 无 spurious local minima | 依赖论文中的宽网络设定 |
| Safran & Shamir 2018 | 两层 ReLU 网络 | spurious local minima 可以很常见 | 特定 teacher-student 和 $k$ 范围 |
| Du et al. 2018 | 一层隐藏 CNN | 存在 spurious local minimum，但 GD 仍可概率性成功 | 特定 CNN、Gaussian input、weight normalization |
| Garipov / Draxler 2018 | 经验 mode connectivity | 不同 minima 之间常有低损失路径 | 经验现象，依赖架构和训练设置 |
| Keskar et al. 2016 / Dinh et al. 2017 | sharp/flat minima 与泛化 | flatness 与泛化有关，但朴素 sharpness 定义会受重参数化影响 | 不能用单一 sharpness 指标解释泛化 |

这张表的核心信息是：这组文献没有给出一个“神经网络到底有没有坏局部极小值”的单句答案。它们给出的答案是条件化的。模型、损失函数、宽度、结构、数据和训练动态共同决定具体的景观性质。

## 10. 常见误区

**误区一：BP 会自动找到全局最优。** BP 只是计算梯度。是否到达好解取决于目标函数几何、优化器、初始化、学习率、batch 噪声和网络结构。

**误区二：非凸就一定有很多坏局部极小值。** 深度线性网络明确反驳了这个推理。非凸性可以来自参数化冗余和矩阵分解，但所有局部极小值仍可能是全局的。

**误区三：没有坏局部极小值就训练容易。** 鞍点、平坦区域和病态曲率仍会让训练慢。Saxe 的深度线性动态和 Kawaguchi 的 bad saddle points 都说明了这一点。

**误区四：宽网络结果等于所有大模型都无局部陷阱。** 宽度相关定理通常依赖损失函数、激活、数据、初始化或层宽条件。它们不能直接外推到所有 Transformer、CNN 或多模态模型。

**误区五：全局训练最优等于泛化最好。** no bad local minima 是训练损失层面的性质。泛化还受到数据分布、模型容量、隐式正则化、优化路径和平坦性等因素影响。

围绕 flatness 的讨论也需要谨慎。[Keskar et al. 2016][keskar-2016] 用 large-batch training 的泛化差距支持了一个经验观点：大 batch 更容易收敛到 sharp minimizers，而 small batch 的噪声更容易偏向 flat minimizers。但 [Dinh et al. 2017][dinh-2017] 指出，深度 ReLU 网络存在尺度重参数化对称性，同一个函数可以被重参数化成任意更 sharp 的参数点，因此朴素 sharpness 不是参数化不变的泛化解释。更可靠的说法是：flatness、volume、SGD 噪声、归一化、正则化和数据结构共同影响泛化，不能把“平坦极小值好”当作无条件定理。

**误区六：NTK 已经解释了现代大模型训练。** NTK 提供了强有力的可证明基线，但它主要描述初始化附近、小参数移动、kernel-like 的训练。许多现代网络的实际表现依赖 feature learning、表示变化和数据结构，这些正是纯 NTK 近似会弱化的部分。

## 11. 结论：坏局部极小值不是唯一主角

从这条文献线可以得到一个更稳健的结论。

一般神经网络训练是非凸优化，因此在原则上可能出现坏局部极小值。早期 BP 文献已经展示了这种失败模式，并且提醒我们不能把 BP 当作全局优化算法。

但深度线性网络、线性残差网络、宽网络、特殊神经元构造和 NTK regime 又说明：非凸性并不自动带来坏局部极小值。在足够特殊或足够宽的设定下，所有局部极小值可以是全局的，或者训练动态不会落入 spurious local minima。

与此同时，负面结果显示，即使是两层 ReLU 网络和一层隐藏 CNN，也可能存在真实的 spurious local minima。只是这些点是否会成为实践中的主要障碍，还要看吸引域、初始化、归一化、过参数化和随机梯度动态。

所以，现代理解不应该是“深度网络没有优化困难”，也不应该是“深度网络到处都是坏局部极小值，所以 BP 靠运气”。更准确的说法是：坏局部极小值是重要对象，但不是唯一障碍，也未必是现代过参数化网络的主要障碍。

真正的理论问题已经从“有没有局部极小值”推进到更细的层面：

- 哪些局部极小值是坏的？
- 它们的吸引域有多大？
- 哪些结构会提供额外下降方向？
- 宽度和深度分别改变了什么几何性质？
- SGD 为什么偏向某些低损失、可泛化的区域？
- 训练误差的全局最优和测试分布上的泛化之间如何连接？

这才是从 Backpropagation 到过参数化理论之间，神经网络损失景观研究真正形成的主线。

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

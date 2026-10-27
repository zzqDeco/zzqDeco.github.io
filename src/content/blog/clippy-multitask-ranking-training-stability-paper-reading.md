---
title: "Clippy 论文精读：多任务推荐排序模型的训练稳定性、损失发散与自适应更新裁剪"
description: "精读 KDD 2023 Clippy 如何分析 YouTube 多任务排序模型训练不稳定问题，并用 Adagrad 更新裁剪提升大规模推荐模型稳定性"
pubDate: "2026-06-29T11:13:49+08:00"
updatedDate: "2026-06-29T11:13:49+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "Recommendation System"
  - "Optimization"
  - "Multitask Learning"
  - "Ranking"
  - "YouTube"
draft: false
---

Jiaxi Tang、Yoel Drori、Daryl Chang、Maheswaran Sathiamoorthy、Justin Gilmer、Li Wei、Xinyang Yi、Lichan Hong 和 Ed H. Chi 在 KDD 2023 发表的 **Improving Training Stability for Multitask Ranking Models in Recommender Systems**，聚焦工业级推荐系统中一个直接阻塞模型迭代的工程痛点：**大规模多任务排序模型在训练中途突发损失发散（loss divergence），导致模型坍塌并进入不可恢复状态**。

围绕这一问题，论文提出了 **Clippy（Adagrad with Clippy）**。作者将训练失稳的根因归纳为局部损失曲率陡峭（steep loss curvature）与有效步长过大（excessive effective step size）的叠加：

```text
loss curvature is steep
  + effective step size is too large
  -> loss divergence
  -> AUC collapses
  -> model may become unrecoverable
```

针对 Adagrad 优化器在推荐模型中的广泛使用，Clippy 直接约束经过预条件处理且乘上学习率后的实际参数更新量，从而在训练踏入失稳边界前及时收紧步长。本文从多任务排序的失稳成因、局部曲率与裁剪机制、生产实验与消融图表、以及 TensorFlow Recommenders 官方 `ClippyAdagrad` 源码实现四个维度展开精读。文中图表截取自原论文 PDF（CC BY 4.0 许可），关于 YouTube 生产环境私有数据的实验结论均以论文报告为准，源码部分基于公开仓库做静态分析。

## 1. 论文概览与损失发散问题定义

| 项目 | 内容 |
| --- | --- |
| 题名 | Improving Training Stability for Multitask Ranking Models in Recommender Systems |
| 作者 | Jiaxi Tang, Yoel Drori, Daryl Chang, Maheswaran Sathiamoorthy, Justin Gilmer, Li Wei, Xinyang Yi, Lichan Hong, Ed H. Chi |
| 会议 | KDD 2023 |
| DOI | `10.1145/3580305.3599846` |
| 页数 | 12 页 |
| 任务场景 | YouTube recommendations multitask ranking model |
| 主要方法 | Adagrad with Clippy |
| 官方实现 | TensorFlow Recommenders `experimental/optimizers/clippy_adagrad.py` |

论文的核心思路可以概括为：在基于 Adagrad 训练的 YouTube 多任务排序模型中，当底层共享结构遇到突发高曲率区域时，结合参数自身规模与绝对安全下界对实际更新步长进行逐坐标自适应裁剪，在不牺牲收敛速度的前提下支持翻倍学习率与更复杂的交叉网络结构。

### 1.1 工业排序模型的稳定性代价

在工业推荐系统的日常研发中，模型能否稳定跑完训练往往决定了特征工程与架构探索的迭代上限。频繁发生的训练发散会在四个层面造成直接损失：

| 成本 | 具体表现 |
| --- | --- |
| 训练资源浪费 | 大模型训练到中后段崩溃，TPU/GPU 算力与数据管线开销直接作废 |
| 研发阻塞 | 新特征、新任务、新结构无法稳定对齐基线，离线实验结论失效 |
| 上线风险 | 若质量门禁存在盲区，退化模型可能流入线上 serving |
| 迭代保守 | 团队被迫调低学习率、缩减模型容量或砍掉高风险特征，拉低质量上限 |

随着排序模型引入更多模态输入与多目标任务，且持续在分布动态漂移的行为流数据上做流式训练（sequential training），训练稳定性已从单纯的优化器调参演变为生产基础设施的核心瓶颈。

### 1.2 Micro-divergence 与 Full Divergence 判据

为了定量讨论稳定性故障，论文根据训练受损后的恢复能力，将损失发散划分为 micro-divergence（微发散）与 full divergence（完全发散）两类：

![Fig. 1: Loss divergence examples](/images/blog/clippy-multitask-ranking-stability/clippy-fig1-loss-divergence.webp)

*Source: Tang et al., KDD 2023, Fig. 1, CC BY 4.0.*

Fig. 1 上半部分展示了训练损失（training loss），下半部分对应训练集 AUC（train AUC），对比了两个模型的典型轨迹：

| 模型 | 现象 | 训练后果 |
| --- | --- | --- |
| model-a | loss 突然跳升、AUC 瞬时下挫，但后续逐步恢复 | micro-divergence |
| model-b | loss 急剧飙升、AUC 跌至 0.5 附近且持续锁死 | full divergence |

两类现象的工程严重程度截然不同。micro-divergence 表现为一次短暂的数值冲击，模型在后续迭代中能够自行走出异常区域，最终收敛质量通常与未发散模型相当；而 full divergence 则是致命故障，二分类 AUC 跌至 0.5 意味着预测彻底退化为随机猜测，继续训练也无法修复。将日常波动与这两类发散放在一起对比，可以得到清晰的工程处置分级：

| 层级 | 表现 | 处理方式 |
| --- | --- | --- |
| 正常抖动 | loss 小幅波动，评估指标稳步上升 | 正常现象，无需干预 |
| micro-divergence | loss/AUC 瞬时异常后自行恢复 | 记录报警日志，纳入稳定性诊断 |
| full divergence | loss 暴涨，AUC 坍塌且无法恢复 | 判定训练失败，立即熔断并阻断上线 |

### 1.3 稳定性复现、检测与度量的工程挑战

在工业流水线中治理 full divergence 之所以棘手，源于三个维度的客观限制：

| 挑战 | 含义 | 工程影响 |
| --- | --- | --- |
| Reproducibility（复现难） | 发散受随机种子与硬件并发影响，具有强偶发性 | 难以精准复现并归因到特定 batch、特征或任务 |
| Detection（检测难） | 高频评估拖慢训练吞吐，低频评估容易漏检瞬时异常 | 需在评估开销与细粒度监控之间做权衡 |
| Measurement（度量难） | 训练启动前缺乏预测稳定性的先验定量指标 | 只能依赖多随机种子、多学习率与多模型容量的网格试跑 |

正因为单次训练无法全面反映稳定性边界，论文在后续实验设计中采用了“多档模型容量 × 多倍学习率 × 多随机种子”的压力测试矩阵，重点考察算法在激进超参数与复杂结构下能否避免发散并保持高收敛质量。

## 2. 多任务排序架构与陡峭曲率下的失稳机理

论文的研究对象聚焦于 YouTube 推荐漏斗中的多任务排序模型（multitask ranking model），其典型结构如 Fig. 2 所示：

![Fig. 2: Multitask ranking model](/images/blog/clippy-multitask-ranking-stability/clippy-fig2-multitask-ranking-model.webp)

*Source: Tang et al., KDD 2023, Fig. 2, CC BY 4.0.*

该架构遵循经典的 Shared-Bottom 多任务范式：

```text
input features and embeddings
  -> shared layers
  -> top shared layer
  -> task-specific layers
  -> task predictions
```

在实际生产中，排序模型需要同时预估点击、完播、点赞、负反馈以及观看时长等多个异构目标（论文实验使用的简化测试台共训练 6 个任务，重点汇报其中 1 个二分类任务的 AUC 与 1 个回归任务的 RMSE）：

| 任务类型 | 典型监督信号 |
| --- | --- |
| binary classification | click、watch、like、long view |
| regression | watch time、dwell time、expected utility |
| negative feedback | hide、not interested、report |
| calibration | source score、position-aware response |

### 2.1 多任务排序与召回模型的结构差异

沿着 YouTube DNN（Covington et al., 2016）确立的“候选生成（candidate generation / retrieval）+ 排序（ranking）”两阶段漏斗来看，排序模型在训练稳定性上面临的挑战远高于召回模型：

| 维度 | Candidate generation / retrieval | Ranking |
| --- | --- | --- |
| 目标 | 从百万级视频池快速召回百级候选 | 对百级候选做多目标精细打分与重排 |
| 常见 loss | softmax、sampled softmax、contrastive loss | 多任务二分类交叉熵与回归损失组合 |
| 特征 | 粗粒度用户历史、item embedding、基础上下文 | 丰富的 user-item 交叉特征、曝光统计、多源候选特征 |
| 模型容量 | 受限于线上 ANN 检索延迟与向量导出解耦要求 | 网络更深更宽，普遍引入 DCN 等高阶显式交叉模块 |
| 任务数 | 单目标或少量代理目标 | 数十个正负反馈目标联合训练 |
| 稳定性风险 | 大词表频次不均、负采样偏差 | 多任务梯度冲突、底层共享表征污染、流式数据分布突变 |

排序模型之所以更易失稳，直接源于其高维交叉特征、大容量共享层与多目标损失叠加后形成的复杂损失曲面。

### 2.2 Shared Bottom 的梯度耦合效应

多任务学习依赖底层共享表征来提升泛化能力与样本效率，但共享层同样也是异常梯度的放大器。设模型共享参数为 $\theta_s$，第 $k$ 个任务的专属塔参数为 $\theta_k$，加权总损失定义为：

$$
L(\theta_s,\theta_1,\ldots,\theta_K)
= \sum_{k=1}^{K} \omega_k L_k(\theta_s,\theta_k)
$$

对共享参数 $\theta_s$ 求导可得：

$$
\nabla_{\theta_s} L
= \sum_{k=1}^{K} \omega_k \nabla_{\theta_s} L_k
$$

一旦某个任务在特定 batch 上因样本分布突变或极端标签产生异常梯度，该梯度便会直接写入共享层参数 $\theta_s$，进而扭曲所有其他任务塔的输入表征，触发跨任务的连锁失稳。

### 2.3 稳定边界与流式训练的诱发因素

从优化动力学角度看，损失发散发生在局部损失曲率陡峭而参数更新步长跨过稳定边界的时刻。以基础梯度下降 $w_{t+1} = w_t - \eta \nabla L(w_t)$ 为例，二次近似模型下的收敛稳定条件要求学习率满足：

$$
\frac{2}{\eta} > \alpha^\ast
$$

其中 $\alpha^\ast$ 为损失函数 Hessian 矩阵的最大特征值，刻画了局部曲面的最陡峭曲率（sharpness）。一旦有效步长超出 $2/\alpha^\ast$，迭代轨迹就会沿着高曲率特征向量方向剧烈震荡并指数级发散。

![Fig. 3: Edge of stability in a quadratic model](/images/blog/clippy-multitask-ranking-stability/clippy-fig3-edge-of-stability.webp)

*Source: Tang et al., KDD 2023, Fig. 3, CC BY 4.0.*

Fig. 3 直观演示了最大特征值 $\alpha_1=20$（对应稳定边界 $2/\alpha_1 = 0.1$）时的二次曲面优化轨迹：左图步长取 $0.09 < 0.1$，参数虽然存在振荡但稳定收敛至极小值；右图步长取 $0.11 > 0.1$，参数沿陡峭方向不断放大震荡，最终发散。

在静态数据集上，模型尚可通过固定较小的学习率规避这一风险；但在工业推荐的流式训练（sequential training）中，以下三类因素会不断推高局部曲率 $\alpha^\ast$，将原本安全的步长推入失稳区：

| 诱发因素 | 推高失稳风险的作用机制 |
| --- | --- |
| Data distribution changes | 训练窗口按时间流式推进，突发热点或日志分布漂移会使局部曲率瞬间变陡 |
| Larger model size and complexity | 增加网络宽度、深度或引入 DCN 显式特征交叉，会显著加剧损失曲面的尖锐程度 |
| Multiple objectives | 多任务梯度在共享层叠加耦合，单一任务的曲率突变会传导至全局参数 |

这也解释了工业实践中一个常见困境：添加高价值特征、扩充任务头、加深共享底座、引入 DCN 结构、调高学习率或扩大 batch size，每一项都是提升离线指标的标准动作，但叠加在一起就会频繁将模型推向稳定边界（edge of stability）之外。

## 3. 从传统梯度裁剪到 Clippy 的设计动机

推荐系统包含海量稀疏类别特征与稠密统计特征，工业界普遍采用 Adagrad 作为主力优化器。在时间步 $t$，记随机梯度为 $g_t = \nabla L(w_t)$，Adagrad 的逐坐标累加器与更新规则写作：

$$
G_t = G_{t-1} + g_t^2, \qquad r_t = g_t \cdot G_t^{-1/2}, \qquad w_{t+1} = w_t - \eta_t r_t
$$

其中平方与幂运算均按元素（coordinate-wise）执行，$G_t$ 为历史梯度平方累加器，$r_t$ 为预条件化后的更新方向，$\eta_t r_t$ 则是该步施加在参数 $w_t$ 上的实际更新量。

### 3.1 Adagrad 基础更新与传统裁剪基线（GC / AGC / LAMB）

为了在曲率骤增时控制步长，已有文献提出了三类典型的裁剪或自适应缩放策略：

1. **Gradient Clipping（GC，标准梯度裁剪）**：直接限制梯度的全局或层级 $L_2$ 范数。当 $\lVert g_t\rVert_2 > \lambda$ 时，按比例缩放梯度 $g_t \leftarrow \sigma_t g_t$：
   $$
   \sigma_t = \min\left\{\frac{\lambda}{\lVert g_t\rVert_2}, 1\right\}
   $$
   GC 实现简单，但固定阈值 $\lambda$ 对网络层宽、模型结构与 batch size 极度敏感，跨层或跨模型迁移时调参成本高昂。
2. **Adaptive Gradient Clipping（AGC，自适应梯度裁剪）**：将裁剪阈值与参数自身范数 $\lVert w_t\rVert_2$ 挂钩，通过限制梯度范数与参数范数之比来消除对绝对尺度的依赖：
   $$
   \sigma_t = \min\left\{\frac{\lambda \lVert w_t\rVert_2}{\lVert g_t\rVert_2}, 1\right\}
   $$
   然而在 Adagrad 语境下，AGC 监控的依然是未经过累加器缩放的原始梯度 $g_t$。当某些坐标历史梯度较小（即 $G_t$ 较小）时，即便 $g_t$ 并不极端，预条件后的 $r_t$ 也可能产生巨大跳变；反之，$g_t$ 较大的高频特征经过大累加器抑制后，实际步长未必危险。
3. **LAMB**：受信任域（trust ratio）思想启发，利用参数范数与更新方向范数的比值动态缩放步长。作者将 LAMB 迁移至 Adagrad 基线后发现，虽然它能压制发散，但在训练初期因参数范数较小会导致更新过于保守，收敛速度与最终指标明显落后。

### 3.2 失稳临界窗口（step-b）的训练动态观察

为了查明 GC 与 AGC 在复杂排序模型中失效的具体原因，论文在 Fig. 4 中深入剖析了一次真实发散过程中的层内统计量演变：

![Fig. 4: Training dynamics and clipping factors](/images/blog/clippy-multitask-ranking-stability/clippy-fig4-training-dynamics.webp)

*Source: Tang et al., KDD 2023, Fig. 4, CC BY 4.0.*

Fig. 4a 标出了发散前后的三个关键时间切片：

| 时刻 | 训练状态 |
| --- | --- |
| step-a | 训练处于健康收敛期，loss 平稳下降，AUC 维持高位 |
| step-b | 失稳临界窗口：loss 刚出现抬头迹象，AUC 开始轻微下滑，模型尚未彻底损坏 |
| step-c | 完全发散状态：loss 飙升至高位，AUC 彻底坍塌至 0.5，激活值大面积死区（dying ReLU） |

观察 Fig. 4b 中顶层共享隐藏层（top hidden layer）的统计轨迹可以发现：等到 step-c 再做干预已经太晚，此时网络层由于先前的剧烈过冲已陷入梯度消失或死区，无法通过后续更新自我修复。**真正决定生死的干预窗口是 step-b**。

在 step-b 时刻，虽然梯度范数出现脉冲式上扬，但由于大宽层包含数以万计的参数坐标，少数坐标上的极端预条件更新被全局 $L_2$ 范数大幅摊薄，导致 GC 与 AGC 计算出的裁剪系数 $\sigma_t$ 下调幅度极小，未能阻止参数跨过稳定边界。相反，Clippy 放弃了迟钝的 $L_2$ 范数与原始梯度监控，直接针对实际更新量采用逐坐标上界约束（$L_\infty$ 范数风格），在 step-b 瞬间给出了比 GC/AGC 小两个数量级的强力裁剪系数，精准拦截了失稳过冲。

## 4. Adagrad with Clippy 算法机制与统一视角

基于上述诊断，论文在 Algorithm 1 中给出了 **Adagrad with Clippy** 的完整迭代流程：

![Algorithm 1: Adagrad with Clippy](/images/blog/clippy-multitask-ranking-stability/clippy-algorithm1-adagrad-with-clippy.webp)

*Source: Tang et al., KDD 2023, Algorithm 1, CC BY 4.0.*

### 4.1 Algorithm 1 与核心更新约束

在每一步迭代 $t = 1, 2, \ldots$ 中，优化器对每一层参数张量 $w_t$ 依次执行以下五步计算：

1. 计算当前 batch 的随机梯度：$g_t = \frac{\partial L(w_t)}{\partial w_t}$。
2. 累加逐坐标梯度平方：$G_t = G_{t-1} + g_t^2$。
3. 计算 Adagrad 预条件更新方向：$r_t = g_t \cdot G_t^{-1/2}$。
4. 计算层级标量裁剪系数 $\sigma_t \in (0, 1]$：
   $$
   \sigma_t
   = \min\left\{
   1,\,
   \min\left(
   \frac{\lambda_{rel}|w_t|+\lambda_{abs}}
   {\eta_t |r_t|}
   \right)
   \right\}
   $$
   其中内层 $\min(\cdot)$ 遍历该层张量的所有参数坐标，取满足所有坐标约束的最小缩放比例。
5. 执行裁剪后的参数更新：$w_{t+1} = w_t - \eta_t \sigma_t r_t$。

将第四步与第五步合并观察，Clippy 本质上是在每一层寻找最大的标量缩放因子 $\sigma_t \le 1$，使得更新后的每一维参数变化量严格受控于如下逐坐标不等式：

$$
\eta_t \sigma_t |r_{t,i}| \le \lambda_{rel}|w_{t,i}| + \lambda_{abs}, \qquad \forall i
$$

这一不等式集成了三项核心设计：
- **直接约束实际更新步长 $\eta_t r_t$**：分母中包含预条件更新 $|r_t|$ 与当前学习率 $\eta_t$，能够真实反映 Adagrad 在各个坐标上即将产生的参数位移。
- **逐坐标敏感的 $L_\infty$ 边界**：内层对所有坐标取最小值（等价于用无穷范数控制最大违规坐标），确保哪怕只有一个坐标出现异常激增，整层更新也会同步缩减，避免异常被高维层宽稀释。
- **保持层内更新方向不变**：虽然边界按坐标逐一核算，但最终作用在该层上的 $\sigma_t$ 是一个统一标量，因此只缩放该层的有效步长，不扭曲 Adagrad 预条件更新方向 $r_t$ 的几何指向。

### 4.2 相对阈值、绝对阈值与学习率耦合机制

分子中的双阈值设计 $\lambda_{rel}|w_t| + \lambda_{abs}$ 兼顾了不同训练阶段的尺度特性。若仅保留相对项 $\lambda_{rel}|w_t|$（类似 AGC），当偏置项（bias）或零初始化权重在训练初期接近 $0$ 时，允许的更新上界也会趋近于 $0$，导致参数被死锁而无法启动学习。引入绝对安全下界 $\lambda_{abs}$ 后，算法实现了平滑的阶段切换：
- **训练初期（$|w_t| \approx 0$）**：$\lambda_{abs}$ 占主导地位，算法表现接近基于绝对阈值的裁剪，允许小参数正常起步；
- **训练中后期（$|w_t|$ 增长成型）**：$\lambda_{rel}|w_t|$ 占主导地位，算法自动过渡为按参数自身模长比例放缩的自适应控制。

与此同时，将学习率 schedule $\eta_t$ 显式纳入分母，使得裁剪强度能够随 warmup 或翻倍学习率实验动态自适应：当外部调高 $\eta_t$ 试图加速收敛时，一旦某些步的 $\eta_t |r_t|$ 越界，$\sigma_t$ 会自动成比例压低，从而允许模型安全运行在原本必崩的 2 倍学习率下。

### 4.3 四类裁剪算法的形式化对比

如果将各类稳定性方法统一纳入预条件更新框架 $w_{t+1} = w_t - (\eta_t \sigma_t) r_t$，可以清晰看出四者在计算缩放因子 $\sigma_t$ 时的本质差异：

| 方法 | 缩放因子 $\sigma_t$ 表达式 | 度量范数与监控对象 | 核心局限或代价 |
| --- | --- | --- | --- |
| GC | $\min\left\{1, \frac{\lambda}{\lVert g_t\rVert_2}\right\}$ | 全局/层级 $L_2$ 范数，监控原始梯度 $g_t$ | 绝对阈值跨层难调，忽略参数尺度与 Adagrad 预条件效应 |
| AGC | $\min\left\{1, \frac{\lambda \lVert w_t\rVert_2}{\lVert g_t\rVert_2}\right\}$ | 层级 $L_2$ 范数，监控原始梯度与参数比值 | 未考虑累加器 $G_t^{-1/2}$ 缩放，大层宽 $L_2$ 范数稀释局部坐标突变 |
| LAMB | 基于 $\frac{\lVert w_t\rVert_2}{\lVert r_t\rVert_2}$ 的信任域缩放 | 层级 $L_2$ 范数，强制对齐参数范数 | 丢弃更新幅值信息，初期参数较小时步长过分保守，拖慢收敛 |
| Clippy | $\min\left\{1, \min_i \frac{\lambda_{rel}|w_{t,i}|+\lambda_{abs}}{\eta_t |r_{t,i}|}\right\}$ | 逐坐标 $L_\infty$ 边界，监控实际步长 $\eta_t r_t$ | 引入 $\lambda_{rel}, \lambda_{abs}$ 超参（源码进一步扩展了累加器参考项） |

## 5. YouTube 排序实验设置与主结果分析

为了系统评估不同模型复杂度与学习率压力下的训练表现，作者没有仅在单一固定配置上对比，而是基于 YouTube 生产数据集搭建了三档渐进扩容的多任务排序测试台（testbed）。

### 5.1 三档复杂度压力测试台（Small / Large / Large+DCN）

![Table 1: Model settings](/images/blog/clippy-multitask-ranking-stability/clippy-table1-model-settings.webp)

*Source: Tang et al., KDD 2023, Table 1, CC BY 4.0.*

如 Table 1 所示，三档配置在非 Embedding 参数量与底层共享结构上逐级加码：

| Model | Non-Embedding Parameters | Shared Bottom Architecture | 设计目的 |
| --- | ---: | --- | --- |
| Small | 7.5M | FFN: 512 x 2 | 基础多任务排序基线 |
| Large | 57.0M | FFN: 4096 x 4 | 隐藏层加宽加深，非稀疏参数量扩大约 7.6 倍 |
| Large+DCN | 68.0M | DCN + LN -> 4096 x 4 | 在输入端叠加 DCN-v2 显式特征交叉与 LayerNorm，构造高曲率高压场景 |

所有模型均联合训练 6 个生产任务，论文汇报其中最具代表性的一个二分类任务 AUC（越高越好）与一个回归任务 RMSE（越低越好）。每种方法均在基准学习率（1x）与翻倍学习率（2x）下搜索最优配置，并固定最优学习率重复 3 个随机种子，统计均值与标准差。

### 5.2 多学习率与多任务离线主实验对比

Table 2 汇总了 Naive（无任何裁剪保护）、GC、AGC、LAMB 与 Clippy 在三档模型上的完整评测结果：

![Table 2: Overall performance](/images/blog/clippy-multitask-ranking-stability/clippy-table2-overall-performance.webp)

*Source: Tang et al., KDD 2023, Table 2, CC BY 4.0.*

横向与纵向拆解 Table 2，可以得出五点关键结论：

1. **无保护基线（Naive）全线崩溃**：即便在最简单的 Small（7.5M）模型与 1x 学习率下，Naive Adagrad 也 100% 发生发散，证明多任务流式排序的失稳风险是系统性存在的。
2. **GC 在复杂交叉结构下触及天花板**：精心调优的 GC 在 Small 与 Large 模型上表现稳健且能跑通 2x 学习率；但一旦升级到 Large+DCN，GC 在 2x 学习率下发生发散，被迫退回 1x 学习率（AUC 停留在 `72.27%`）。
3. **AGC 稳定性受限且方差偏高**：AGC 在全部三档模型上都只能使用 1x 学习率；在 Large+DCN 上，其 AUC 标准差明显放大（`±0.11`），表明模型已处于失稳边缘。
4. **LAMB 保守求稳但损失模型质量**：LAMB 虽然在三档模型上均支持 2x 学习率且未发生发散，但最终收敛质量显著掉队（Large+DCN 上 AUC 仅为 `72.07%`，甚至低于 GC 在 1x 学习率下的表现）。
5. **Clippy 兼顾激进学习率与最优收敛指标**：Clippy 是唯一在三档模型上全部稳定跑通 2x 学习率、且全面刷新最优指标的方法。在最复杂的 Large+DCN 上，Clippy 取得了 `72.37%` 的 AUC 与 `0.8180` 的 RMSE，相比最强基线 GC 提升了 `+0.10%` AUC（论文指出在该生产排序场景中，`0.1%` 的离线 AUC 提升已属于能够转化为显著线上收益的大幅改进）。

### 5.3 训练收敛轨迹与高学习率容忍度

Fig. 5 进一步展示了训练全周期的 AUC 动态演化曲线：

![Fig. 5: AUC curves](/images/blog/clippy-multitask-ranking-stability/clippy-fig5-auc-training-curves.webp)

*Source: Tang et al., KDD 2023, Fig. 5, CC BY 4.0.*

Fig. 5a 对比了 Large+DCN 模型在 1x 与 2x 学习率下的训练过程：在 1x 学习率下，除 LAMB 明显偏慢外，GC、AGC 与 Clippy 曲线几乎重合，验证了 Clippy 在安全区域不会过度干预正常收敛；而在 2x 学习率下，GC 与 AGC 先后崩溃或剧烈震荡，唯有 Clippy 全程平滑爬升并兑现了大步长带来的加速与质量红利。Fig. 5b 则将 Clippy 与最强基线 GC 在三档模型上的表现并置，清晰展示出**模型结构越复杂（从 Small 到 Large 再到 Large+DCN），Clippy 相对传统梯度裁剪的优势越显著**。

## 6. 层间裁剪行为、诱因消融与跨领域验证

除了主实验指标，论文通过层间裁剪轨迹、补充材料消融实验以及跨领域 Transformer 实验，进一步验证了前文的机理分析。

### 6.1 不同网络层的裁剪强度分布

Fig. 6 记录了 Large+DCN 模型在训练过程中五类参数层的裁剪系数 $\sigma_t \in (0, 1]$ 分布（数值越小代表裁剪越强）：

![Fig. 6: Clipping factors by layer](/images/blog/clippy-multitask-ranking-stability/clippy-fig6-layer-clipping-factors.webp)

*Source: Tang et al., KDD 2023, Fig. 6, CC BY 4.0.*

对比图中的 DCN weights、bottom hidden weights、top hidden weights、classification task weights 与 regression task weights，可以发现**越靠近底层的共享网络（如 DCN 层与底层隐藏层），触发的裁剪频率越高、缩放幅度也越大**。这一现象与多任务架构的梯度耦合机理高度吻合：底层参数直接承接全量任务的回传梯度，且其微小扰动会在前向传播中被高层网络逐级放大，因此 Clippy 自动将主要的步长抑制集中在了最脆弱的底层共享模块上。

### 6.2 失稳诱因消融实验（Table 3）

为了验证第 2 节提出的失稳诱因，作者在补充材料 Table 3 中以无裁剪保护的 Large+DCN 为基准，将学习率下调至临近失稳边界的 `0.4x`，逐一做减法消融并统计发散比例：

![Table 3: Divergence ratio ablation](/images/blog/clippy-multitask-ranking-stability/clippy-table3-divergence-ratio.webp)

*Source: Tang et al., KDD 2023, Table 3, CC BY 4.0.*

| 实验改动（Change） | 发散次数 / 尝试次数 | 发散率（Divergence Ratio） | 印证的风险维度 |
| --- | ---: | ---: | --- |
| Clean（原始 Large+DCN） | 5 / 5 | 100% | 基准高压状态必崩 |
| Smaller model size | 1 / 3 | 33% | 模型容量越大，曲率越陡峭 |
| Remove DCN | 2 / 3 | 66% | 显式高阶交叉结构加剧不稳定性 |
| Remove subset of input features | 2 / 3 | 66% | 输入特征越多，分布漂移与异常源越多 |
| Remove one task | 2 / 3 | 66% | 减少任务可缓解共享层梯度冲突 |
| Remove two tasks | 0 / 3 | 0% | 砍掉两个任务后 3 次试验全部稳定收敛 |

尽管受限于生产模型训练成本，每组消融仅运行了 3～5 次试验，但趋势非常明确：缩减模型规模、移除 DCN、裁剪输入特征、减少联合训练任务数，均能实质性降低发散概率；尤其是移除两个任务后发散率从 100% 降至 0%，直接印证了多任务耦合对训练稳定性的冲击。

### 6.3 隐藏层与任务头的跨层统计分析（Fig. 8）

补充材料 Fig. 8 补全了 Fig. 4 未展开的其余网络层（bottom hidden layer、classification task layer、regression task layer）在失稳窗口 step-b 前后的统计量表现：

![Fig. 8: Statistics from other layers](/images/blog/clippy-multitask-ranking-stability/clippy-fig8-other-layer-stats.webp)

*Source: Tang et al., KDD 2023, Fig. 8, CC BY 4.0.*

数据表明，底层隐藏层与回归任务头在 step-b 同样出现了明显的梯度跃升；而在二分类任务头（classification task layer）中，全局梯度范数变化极为微弱，导致 GC 与 AGC 在 step-b 完全未触发裁剪（$\sigma_t = 1$），唯独 Clippy 捕捉到了局部坐标的预条件步长异常并及时介入。

### 6.4 Transformer 机器翻译附加实验（Fig. 7）

为了检验 Clippy 在推荐系统之外的通用性，作者基于 `init2winit` 框架在 WMT 英德翻译任务（`translate_wmt` 数据集 + `xformer_translate` 模型）上对比了 AdamW、标准 Adagrad 与 Adagrad with Clippy：

![Fig. 7: Transformer translation experiment](/images/blog/clippy-multitask-ranking-stability/clippy-fig7-transformer-translation.webp)

*Source: Tang et al., KDD 2023, Fig. 7, CC BY 4.0.*

| 优化器 | 最佳验证集错误率（Error Rate） | 学习率容忍度与稳定性表现 |
| --- | ---: | --- |
| AdamW | `31.6%` | 标准 NLP 强基线，整体最优 |
| Adagrad | `36.9%` | 高学习率下极易发散，可用学习率窗口窄 |
| Adagrad with Clippy | `33.4%` | 显著拓宽 Adagrad 的稳定学习率区间，大幅缩小与 AdamW 的差距 |

这一结果清晰界定了方法的适用边界：在纯稠密 Transformer 任务上，Clippy 并未超越原生适配该领域的 AdamW；但当工程系统因为超大规模稀疏 Embedding 而必须采用 Adagrad 时，Clippy 能有效弥补 Adagrad 在深层交叉与注意力结构上的稳定性短板。

## 7. TensorFlow Recommenders 源码解析：ClippyAdagrad

论文开源的官方实现位于 TensorFlow Recommenders 仓库的 `tensorflow_recommenders/experimental/optimizers/clippy_adagrad.py`。对照源码与论文 Algorithm 1，可以看到工业级优化器在落地时所做的多项工程扩展：

| 源码符号 / 参数 | 对应论文概念 | 工程职责与说明 |
| --- | --- | --- |
| `shrink_by_references()` | Algorithm 1 第 4 步裁剪计算 | 给定待更新张量 `delta`、参考张量列表与对应阈值，计算全局缩放因子 |
| `ClippyAdagrad` | Adagrad with Clippy 优化器 | 继承自 Keras Optimizer，管理 slot 变量与稀疏/稠密更新分发 |
| `variable_relative_threshold` | 相对阈值 $\lambda_{rel}$ | 控制允许更新量相对于当前参数绝对值 $|w_t|$ 的比例（默认 `0.1`） |
| `absolute_threshold` | 绝对阈值 $\lambda_{abs}$ | 控制允许更新量的绝对安全下界（默认 `1e-2`） |
| `accumulator_relative_threshold` | 论文未提及（源码新增扩展） | 引入累加器逆平方根 $G_t^{-1/2}$ 作为额外参考尺度（默认 `0.0`） |
| `use_standard_accumulator_update` | 累加器更新时序控制 | `True` 对应论文先更 $G_t$ 再算步长；默认 `False` 采用延迟更新 |
| `clip_accumulator_update` | 异常梯度防污染开关 | 延迟更新模式下，可选用裁剪后的梯度更新累加器 $G_t$ |
| `export_clipping_factors` | 层级裁剪监控导出 | 将每步每层的 `clipping_factor` 写入优化器状态，供 TensorBoard 监控 |

### 7.1 核心接口与 shrink_by_references 实现

源码将逐坐标边界检查封装在纯函数 `shrink_by_references(tensor, references, relative_factors, absolute_factor)` 中。设待裁剪的实际更新张量为 $\Delta = \eta_t r_t$，参考张量集合为 $\{R^{(j)}\}$，对应的相对系数为 $\{\lambda_j\}$，绝对阈值为 $\lambda_{abs}$，该函数计算的逐坐标容许上界与全局缩放因子为：

$$
\text{max\_delta}_i = \sum_{j} \lambda_j \left|R^{(j)}_i\right| + \lambda_{abs}, \qquad
\sigma = \min\left(1.0,\, \min_i \frac{\text{max\_delta}_i}{|\Delta_i|}\right)
$$

当 $|\Delta_i| = 0$ 时，源码通过 `tf.where` 安全回退为 $1.0$ 以避免除零异常，最终返回缩放后的张量 $\sigma \cdot \Delta$ 与标量 $\sigma$。在 `ClippyAdagrad.update_step()` 中，各核心变量与论文公式严格对齐：

| 论文符号 | `update_step()` 源码实现语义 |
| --- | --- |
| $G_t$ | 优化器状态槽 `accumulator` |
| $G_t^{-1/2}$ | `precondition = tf.math.rsqrt(accumulator_values + self.epsilon)` |
| $\eta_t r_t$ | `delta = lr * grad_values * precondition` |
| $\sigma_t, \eta_t \sigma_t r_t$ | `clipped_delta, clipping_factor = shrink_by_references(delta, ...)` |
| $w_{t+1} = w_t - \eta_t \sigma_t r_t$ | 稠密变量调用 `variable.assign_sub(clipped_delta)`，稀疏变量调用 `scatter_sub` |

### 7.2 源码对论文 Algorithm 1 的工程扩展

相比论文中的精简公式，TFRS 源码在两个关键细节上做了针对生产环境的增强：

第一，**引入累加器参考项（`accumulator_relative_threshold`）**。在调用 `shrink_by_references` 时，源码传入了两个参考张量：`references = [variable_values, precondition]`。这意味着源码实际执行的裁剪边界为：

$$
|\Delta_i| \cdot \sigma_t \le \lambda_{var}|w_{t,i}| + \lambda_{acc}\left(G_{t,i} + \epsilon\right)^{-1/2} + \lambda_{abs}
$$

当训练进入中后期，累加器 $G_t$ 随步数单调递增，$\left(G_t + \epsilon\right)^{-1/2}$ 逐渐衰减。源码注释指出，开启 $\lambda_{acc} > 0$ 可以在训练初期放宽约束、而在后期随累加器增长自动收紧裁剪边界。当保持默认值 `accumulator_relative_threshold = 0.0` 时，该式退化回论文公式。

第二，**延迟累加器更新与异常梯度隔离（`use_standard_accumulator_update` & `clip_accumulator_update`）**。论文 Algorithm 1 先将 $g_t^2$ 累加进 $G_t$，再计算预条件更新；但如果某一步出现极端异常梯度（outlier gradient），直接累加会导致 $G_t$ 被永久撑大，使该坐标在后续训练中丧失有效学习率。为此，源码默认将 `use_standard_accumulator_update` 设为 `False`：先基于旧累加器 $G_{t-1}$ 计算预条件步长与裁剪因子 $\sigma_t$，再根据 `clip_accumulator_update` 决定是将原始梯度 $g_t^2$ 还是裁剪后的有效梯度 $(\sigma_t g_t)^2$ 写入 $G_t$。

### 7.3 IndexedSlices 稀疏梯度支持与监控指标导出

在推荐模型中，底层 Embedding 表的梯度以 `tf.IndexedSlices` 稀疏切片形式传递。若优化器直接将其转为稠密张量计算裁剪，会导致显存爆炸并破坏稀疏更新性能。`ClippyAdagrad` 在 `update_step` 中专门针对 `tf.IndexedSlices` 实现了稀疏分支：先通过 `tf.gather` 仅取出当前 batch 命中的稀疏行参数与对应累加器切片，在活跃子集上调用 `shrink_by_references` 计算缩放因子，最后通过 `scatter_add` 与 `scatter_sub` 原地写回。

此外，开启 `export_clipping_factors=True` 后，优化器会为每个受训变量维护一个非训练标量槽，实时记录每一步的 $\sigma_t$。在工程诊断中，围绕该指标可构建完整的稳定性监控面板：

| 监控指标 | 工程诊断用途 |
| --- | --- |
| per-layer min clipping factor | 定位全网最脆弱、受裁剪最剧烈的瓶颈层（如底层 DCN 或特定任务头） |
| clipping factor distribution | 观察 $\sigma_t$ 分位数，判断模型是否长期处于过度裁剪（$\sigma_t \ll 1$）导致欠拟合 |
| clipping trigger frequency | 区分裁剪是仅在异常 batch 偶发介入，还是全程高频激活 |
| clipping vs feature/task rollout | 新特征或新任务上线试跑时，对比 $\sigma_t$ 突变定位引发曲率恶化的变更源 |

## 8. 工业级多任务排序接入指南与方法局限性

将 Clippy 引入自有多任务排序系统时，建议遵循渐进式验证流程，而非简单替换优化器了事。

### 8.1 渐进式实验设计与监控体系搭建

1. **明确发散分级与熔断标准**：在训练平台中建立自动化判据，区分瞬时跳升后自愈的 micro-divergence、核心指标坍塌锁死的 full divergence、随机种子间高方差的 unstable convergence，以及因 $\sigma_t$ 长期过小导致收敛停滞的 over-clipping。
2. **搭建压力测试矩阵**：参照论文 Table 2，同时在基准学习率（1x）与激进学习率（1.5x / 2x）、基础模型与叠加高阶交叉模块（如 DCN / Multi-head Attention）的复杂模型上跑多随机种子对照，重点检验算法能否解锁更大步长与更深结构。
3. **优先作用于非 Embedding 稠密参数**：与论文实验一致，建议第一阶段仅对 shared bottom、特征交叉层与各任务塔等稠密层启用 Clippy，保持稀疏 Embedding 层的原有优化策略；待稠密层稳定收益明确后，再单独评估稀疏 ID 频次差异下的 `IndexedSlices` 裁剪影响。
4. **建立分层的更新量与任务级监控**：除了监控全局 loss，应按层记录 `gradient_norm`、`update_norm`、`max_abs_update`、`parameter_norm` 与 `clipping_factor`，并按任务拆分监控各自的 loss、AUC/RMSE、校准度（calibration）与标签分布漂移，以便在 step-b 窗口及时捕获失稳前兆。
5. **坚守线上发布门禁**：优化器层面的更新裁剪只能保障数值优化过程不崩，无法防御样本穿越、曝光偏差（exposure bias）、多目标权重失衡或线上特征不一致。模型推全前仍需严格执行离线门禁、Shadow 流量比对、小流量 A/B 与实时回滚机制。

### 8.2 方法边界、适用前提与总结

客观评估这篇论文，需要注意以下方法边界：首先，论文的核心实验建立在未公开的 YouTube 生产数据集与内部测试台上，Table 3 的归因消融样本量也较小（每组 3～5 次 trial），其具体的 AUC 提升幅度不能直接作为其他业务场景的定量预期；其次，Clippy 并非完全免调参，相对阈值 $\lambda_{rel}$、绝对阈值 $\lambda_{abs}$ 以及源码扩展的累加器阈值仍需结合业务模型尺度做校准，过严的阈值会重蹈 LAMB 欠拟合的覆辙；最后，它专注于解决 Adagrad 在多任务高曲率区域的步长失控问题，并不替代 AdamW 在稠密模型上的地位，也不解决推荐系统中的目标设计与样本偏差问题。

回到推荐系统演进的全局视角，当工业排序模型走过基础闭环，开始持续叠加流式数据、高维交叉特征、大容量共享底座与多目标任务时，下述失稳链条便会成为阻碍模型扩容的常态瓶颈：

```text
recommendation ranking model
  -> sequential data distribution changes
  -> many features and multiple tasks
  -> large shared bottom
  -> large batch and high learning rate
  -> steep curvature at some step
  -> actual Adagrad update too large
  -> loss divergence
```

Clippy 的核心贡献，正是精准切中了这条链路的末端环节——在失稳刚露头的临界窗口（step-b），通过逐坐标的相对与绝对混合边界控制 Adagrad 的实际参数更新步长，并以层级裁剪系数为工程团队提供了透明的稳定性诊断探针。对于正在遭遇“加特征、加任务或调高学习率后模型偶发崩溃”的推荐排序团队而言，这是一套逻辑闭环且极具实用价值的工程方案。

## 参考资料

1. Jiaxi Tang et al. [Improving Training Stability for Multitask Ranking Models in Recommender Systems](https://dl.acm.org/doi/10.1145/3580305.3599846). KDD 2023.
2. arXiv. [Improving Training Stability for Multitask Ranking Models in Recommender Systems](https://arxiv.org/abs/2302.09178).
3. Google Research. [Improving Training Stability for Multitask Ranking Models in Recommender Systems](https://research.google/pubs/improving-training-stability-for-multitask-ranking-models-in-recommender-systems/).
4. TensorFlow Recommenders. [Experimental optimizers directory](https://github.com/tensorflow/recommenders/tree/main/tensorflow_recommenders/experimental/optimizers).
5. TensorFlow Recommenders. [ClippyAdagrad raw source](https://raw.githubusercontent.com/tensorflow/recommenders/main/tensorflow_recommenders/experimental/optimizers/clippy_adagrad.py).
6. Jeremy Cohen et al. [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065).
7. Justin Gilmer et al. [A Loss Curvature Perspective on Training Instability in Deep Learning](https://arxiv.org/abs/2110.04369).
8. Andy Brock et al. [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171).

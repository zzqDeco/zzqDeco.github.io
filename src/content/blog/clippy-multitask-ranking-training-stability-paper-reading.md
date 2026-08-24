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

Jiaxi Tang、Yoel Drori、Daryl Chang、Maheswaran Sathiamoorthy、Justin Gilmer、Li Wei、Xinyang Yi、Lichan Hong 和 Ed H. Chi 在 KDD 2023 发表的 **Improving Training Stability for Multitask Ranking Models in Recommender Systems**，是一篇非常工程化的推荐系统论文。它讨论的不是“推荐模型如何更准”，而是一个经常被论文摘要轻描淡写、但在真实工业系统里会直接阻塞迭代的问题：**多任务排序模型训练到一半突然 loss divergence，模型变成不可用状态**。

这篇论文的主角是 **Clippy**，也就是论文提出的 Adagrad with Clippy。它不是普通意义上的“梯度裁剪小技巧”，而是一个围绕推荐排序模型训练稳定性重新组织的更新裁剪方法。论文把问题定位为：

```text
loss curvature is steep
  + effective step size is too large
  -> loss divergence
  -> AUC collapses
  -> model may become unrecoverable
```

它的核心动作不是直接裁剪 raw gradient，而是在 Adagrad 的 preconditioned update $r_t$ 层面做裁剪，并且把参数大小、绝对阈值和学习率都放进 clipping factor 的计算中。换句话说，Clippy 关心的是“这一步真正会让参数移动多少”，而不是只看“梯度范数有多大”。

本文是一篇中文论文精读，会覆盖四条线：

1. **问题线**：YouTube 多任务 ranking model 为什么比 retrieval model 更容易训练不稳定。
2. **优化线**：step size、loss curvature、Hessian 最大特征值、GC、AGC、LAMB 和 Clippy 的关系。
3. **实验线**：Fig. 1-8、Algorithm 1、Table 1-3 的逐图表精读。
4. **代码线**：官方 TensorFlow Recommenders 当前 `ClippyAdagrad` 源码和论文 Algorithm 1 的实现差异。

本文插入的论文图表来自用户提供的本地 ACM PDF，只做等比例裁切展示；论文为 CC BY 4.0 许可。本文不复现 YouTube production dataset，不声称跑通论文实验，只做论文级方法精读、源码静态阅读和工程落地复盘。

## 1. 论文信息与一句话贡献

论文基本信息如下。

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

一句话概括这篇论文：

> Clippy 把推荐多任务排序模型的 loss divergence 问题，从“梯度太大所以裁剪梯度”推进到“Adagrad 预条件更新在某些坐标上导致实际步长失控，所以按参数和 accumulator 参考值自适应裁剪更新”。

这句话里有三个关键词。

第一是 **multitask ranking model**。论文不是讨论所有模型的训练稳定性，而是从 YouTube 推荐排序模型出发。排序模型通常比召回模型更大，特征更多，目标更多，底层共享更多，因此一个任务的异常梯度可能通过 shared bottom 影响其他任务。

第二是 **loss divergence**。这里的问题不是普通训练震荡，而是训练 loss 突然抬升、AUC 突然下跌，严重时 AUC 变成接近 0.5，模型相当于随机预测，而且继续训练也无法恢复。

第三是 **update clipping**。Clippy 不是只看 raw gradient 的 $L_2$ norm，而是看 Adagrad 预条件后的实际 update，并使用类似 $L_\infty$ 的坐标敏感约束，对少数坐标的剧烈变化更敏感。

## 2. 为什么推荐排序模型的训练稳定性值得单独研究

推荐系统论文里常见的问题是：召回怎么做、排序怎么做、特征怎么建、目标怎么设、冷启动怎么处理。但真实工业训练链路里，还有一个更基础的问题：模型能不能稳定训练完。

如果一个模型的训练经常 diverge，会带来几类成本。

| 成本 | 具体表现 |
| --- | --- |
| 训练资源浪费 | 大模型训练到中后段崩掉，TPU/GPU 和数据管线成本被浪费 |
| 研发阻塞 | 新特征、新任务、新结构无法稳定比较，实验结论不可用 |
| 上线风险 | 如果质量门禁不充分，坏模型可能进入 serving |
| 迭代保守 | 团队被迫降低学习率、缩小模型、减少特征或任务，牺牲质量上限 |

论文很明确地把这件事和推荐系统规模联系起来：推荐模型越来越大，包含更多输入模态和任务；推荐数据也不是固定静态数据集，而是随时间变化的行为流。在这种环境里，训练稳定性不是“optimizer 调参小事”，而是生产模型能否持续迭代的基础设施问题。

这篇论文的价值也在这里。它没有提出一个更复杂的 ranker 架构，而是讨论模型变复杂之后如何不崩。

## 3. Loss divergence 症状：micro vs full

论文先把 loss divergence 分成两类：micro-divergence 和 full divergence。

![Fig. 1: Loss divergence examples](/images/blog/clippy-multitask-ranking-stability/clippy-fig1-loss-divergence.webp)

*Source: Tang et al., KDD 2023, Fig. 1, CC BY 4.0.*

Fig. 1 上半部分是 training loss，下半部分是 train AUC。图里有两个模型：

| 模型 | 现象 | 训练后果 |
| --- | --- | --- |
| model-a | loss 突然跳起，AUC 突然掉下去，但后续恢复 | micro-divergence |
| model-b | loss 迅速变高，AUC 掉到 0.5 附近且不能恢复 | full divergence |

micro-divergence 并不一定致命。它像一次短暂训练异常，模型可能在后续步骤恢复到正常质量。论文说这类模型最终质量可以和未发散模型相当。

full divergence 则是严重故障。AUC 掉到 0.5 意味着二分类任务接近随机预测；如果继续训练也回不来，这个模型就不能用于 serving。

### 3.1 为什么 Fig. 1 很重要

Fig. 1 的关键不是曲线形状，而是它给了一个工程判据：训练稳定性不只是 loss 是否单调下降，而是模型是否会进入不可恢复的坏状态。

很多训练异常有三种层级：

| 层级 | 表现 | 处理方式 |
| --- | --- | --- |
| 正常抖动 | loss 小幅波动，指标继续上升 | 可接受 |
| micro-divergence | loss/AUC 瞬间异常，后续恢复 | 需要记录和诊断 |
| full divergence | loss 暴涨，AUC 坍塌，无法恢复 | 训练失败，必须阻断上线 |

在工业推荐系统中，最危险的是 full divergence。因为它可能不是一开始就出现，而是在训练到几十万步之后突然发生。训练越贵，越需要提前规避这类风险。

### 3.2 为什么检测也很难

论文指出训练稳定性面临三类挑战：reproducibility、detection、measurement。

| 挑战 | 含义 | 工程影响 |
| --- | --- | --- |
| Reproducibility | divergence 可能随机发生，不总是可复现 | 很难定位具体 batch、特征或任务 |
| Detection | 高频 evaluation 会拖慢训练，低频 evaluation 可能漏掉 micro-divergence | 需要额外训练监控 |
| Measurement | 训练前缺少稳定性定量指标 | 通常只能多 seed、多配置试跑 |

这也是为什么论文后面的实验选择了“多个模型容量 + 多个学习率 + 多个随机种子”的评估方式。它不是只比最终 AUC，而是在比较哪些方法能让更复杂模型和更大学习率不崩。

## 4. 多任务 ranking model 定义

论文研究的是 YouTube 推荐系统里的 ranking model，而不是 candidate generation model。

![Fig. 2: Multitask ranking model](/images/blog/clippy-multitask-ranking-stability/clippy-fig2-multitask-ranking-model.webp)

*Source: Tang et al., KDD 2023, Fig. 2, CC BY 4.0.*

Fig. 2 是一个典型多任务排序模型：

```text
input features and embeddings
  -> shared layers
  -> top shared layer
  -> task-specific layers
  -> task predictions
```

在推荐系统里，ranking model 通常要预测多个目标。例如：

| 任务类型 | 可能标签 |
| --- | --- |
| binary classification | click、watch、like、long view |
| regression | watch time、dwell time、expected utility |
| negative feedback | hide、not interested、report |
| calibration | source score、position-aware response |

论文没有披露 YouTube production model 的完整任务列表，但实验里提到简化模型训练了六个任务，只展示了一个二分类任务的 AUC 和一个回归任务的 RMSE。

### 4.1 Ranking model 与 retrieval model 的差异

这篇论文和 YouTube DNN 2016 那篇论文可以接起来读。YouTube DNN 把推荐系统拆成 candidate generation 和 ranking；Clippy 这篇主要聚焦后者。

| 维度 | Candidate generation / retrieval | Ranking |
| --- | --- | --- |
| 目标 | 从海量 item 中召回几百个候选 | 对候选做精细排序 |
| 常见 loss | softmax、sampled softmax、contrastive loss | 多任务分类/回归 loss |
| 特征 | 粗粒度用户历史、item embedding、上下文 | 更丰富的 user-item cross、曝光、候选源、统计特征 |
| 模型容量 | 通常受 ANN 导出和低延迟约束 | 容量更大，特征更多 |
| 任务数 | 常见单目标或少目标 | 多目标更常见 |
| 稳定性风险 | 大词表/负采样/embedding 更新 | 多任务梯度、shared bottom、复杂特征、数据漂移 |

论文的一个核心判断是：ranking model 通常比 retrieval model 更容易训练不稳定。原因不是排序模型“天生差”，而是它具备更多导致 steep curvature 和异常梯度传播的条件。

### 4.2 Shared bottom 是双刃剑

多任务学习的基本动机是：多个任务共享底层特征表示，可以互相迁移，提高泛化和数据效率。但 shared layer 也会让异常传播更快。

假设模型有共享参数 $\theta_s$ 和任务参数 $\theta_k$，总 loss 是：

$$
L(\theta_s,\theta_1,\ldots,\theta_K)
= \sum_{k=1}^{K} \omega_k L_k(\theta_s,\theta_k)
$$

共享参数的梯度是：

$$
\nabla_{\theta_s} L
= \sum_{k=1}^{K} \omega_k \nabla_{\theta_s} L_k
$$

如果某个任务在某个 batch 上产生异常梯度，它不只影响自己的 tower，还会进入 shared bottom。随后其他任务的输入表示也会改变，产生连锁反应。论文把这描述为 shared layers 和 shared embeddings 的双刃剑效应。

## 5. Root cause：steep curvature 下 step size 过大

论文把 loss divergence 的内在原因概括为：

```text
step size being too large when loss curvature is steep
```

这里的 step size 不是单纯学习率 $\eta$，而是实际参数更新幅度。对于普通 gradient descent：

$$
w_{t+1} = w_t - \eta \nabla L(w_t)
$$

如果损失曲面局部很陡，学习率又过大，更新可能跨过稳定区域，使 loss 上升甚至发散。

论文引用了二次模型中的稳定条件：

$$
\frac{2}{\eta} > \alpha^\ast
$$

其中 $\alpha^\ast$ 是 loss Hessian 的最大特征值，也可以理解为局部曲率或 sharpness 的代表。如果 $\eta > 2/\alpha^\ast$，训练就可能进入不稳定状态。

![Fig. 3: Edge of stability in a quadratic model](/images/blog/clippy-multitask-ranking-stability/clippy-fig3-edge-of-stability.webp)

*Source: Tang et al., KDD 2023, Fig. 3, CC BY 4.0.*

Fig. 3 展示了一个二次模型。当最大特征值 $\alpha_1=20$ 时，稳定边界是：

$$
\frac{2}{\alpha_1} = 0.1
$$

左图 step size 为 0.09，低于 0.1，轨迹在最优点附近收敛；右图 step size 为 0.11，高于 0.1，轨迹沿高曲率方向被放大，出现训练不稳定。

### 5.1 为什么这对推荐系统尤其麻烦

在固定数据集上训练模型时，局部曲率虽然会变化，但数据分布相对稳定。推荐排序模型不同，它常常采用 sequential training：训练窗口按时间推进，输入特征和标签分布持续变化。

论文列出三个让推荐 ranking model 更危险的因素。

| 因素 | 为什么增加不稳定性 |
| --- | --- |
| Data distribution changes | 推荐数据按时间流入，特征和标签分布会突然变化，局部曲率可能变陡 |
| Larger model size and complexity | 排序模型容量更大、结构更复杂，经验上更容易出现尖锐曲率 |
| Multiple objectives | 多任务梯度通过共享层传播，一个任务异常会影响其他任务 |

这解释了为什么一些“质量提升”的改动会诱发训练不稳定：

1. 添加新特征。
2. 添加新任务。
3. 增大 shared bottom。
4. 引入 DCN 这类 cross 结构。
5. 提高学习率加速收敛。
6. 扩大 batch size，使优化更 curvature-driven。

这些改动本身不是错的；它们很多时候确实能提升模型质量。但它们也会把模型推到更靠近 edge of stability 的区域。

## 6. Baseline：Adagrad、GC、AGC、LAMB

论文的方法建立在 Adagrad 上。先看 Adagrad 的基础更新。

$$
G_t = G_{t-1} + g_t^2
$$

$$
r_t = g_t \cdot G_t^{-1/2}
$$

$$
w_{t+1} = w_t - \eta_t r_t
$$

这里 $g_t$ 是当前梯度，$G_t$ 是 accumulator，$r_t$ 是 Adagrad 预条件后的 update direction。所有平方和幂操作都是逐坐标的。

推荐系统里 Adagrad 很常见，尤其在大量 sparse feature 和 embedding 参与训练时，它对不同坐标自适应调节学习率，实践表现通常很强。

### 6.1 Gradient Clipping

Gradient Clipping 直接限制梯度范数：

$$
g \rightarrow \sigma g
$$

$$
\sigma = \min\left\{\frac{\lambda}{\lVert g\rVert}, 1\right\}
$$

如果梯度范数超过阈值 $\lambda$，就把它缩小到阈值以内。

GC 的优点是简单；缺点是阈值很敏感。不同层、不同模型结构、不同 batch size、不同学习率都可能需要重新调 $\lambda$。在复杂 ranking model 中，单靠 raw gradient norm 也不一定能捕捉实际参数更新风险。

### 6.2 Adaptive Gradient Clipping

AGC 的核心是看梯度范数和参数范数的比例：

$$
\sigma = \min\left\{
\frac{\lambda \lVert w\rVert}{\lVert g\rVert},
1
\right\}
$$

如果 $\lVert g\rVert / \lVert w\rVert$ 太大，就裁剪梯度。相比 GC，AGC 的阈值随参数范数变化，不需要为每层手动设置绝对梯度阈值。

但 AGC 仍然看的是 raw gradient，而不是 Adagrad 预条件后的实际 update。对于 Adagrad，某些坐标即使 raw gradient 不极端，也可能因为 accumulator 较小而产生较大的 $r_t$。反过来，某些 raw gradient 大的坐标，预条件后实际 update 可能并不大。

### 6.3 LAMB

LAMB 常用于大 batch 训练。它使用类似 trust ratio 的思想，让 update 的整体尺度和参数范数相关。论文把 LAMB 改造到 Adagrad 语境中作为 baseline。

LAMB 的问题是它更关注 update direction，对 update magnitude 的使用方式和 Clippy 不同。论文实验里 LAMB 能避免 divergence，但收敛速度和最终指标较差，作者推测原因是初期参数范数小，导致更新过保守。

### 6.4 为什么 baseline 不够

这几类方法都有一个共同目标：当局部曲率变陡时，缩小有效步长。

但论文在 YouTube ranking model 上观察到：

1. GC 和 AGC 的 measurement 对少数坐标的剧烈变化不够敏感。
2. $L_2$ norm 可能被大层宽摊薄，少数坐标异常不会明显改变整体范数。
3. Adagrad 的实际参数变化由 $r_t$ 决定，不是 raw $g_t$。
4. LAMB 稳定但可能牺牲收敛。

Clippy 就是在这些限制上设计的。

## 7. Fig. 4：训练动态与 Clippy 的关键观察

Fig. 4 是整篇论文最关键的图。

![Fig. 4: Training dynamics and clipping factors](/images/blog/clippy-multitask-ranking-stability/clippy-fig4-training-dynamics.webp)

*Source: Tang et al., KDD 2023, Fig. 4, CC BY 4.0.*

图分为两部分。

Fig. 4a 看三个时刻：

| 时刻 | 状态 |
| --- | --- |
| step-a | 模型仍健康，loss 正常，AUC 高 |
| step-b | loss 开始抬升，AUC 开始下降，但还不明显 |
| step-c | full divergence，loss 很高，AUC 掉到 0.5 |

Fig. 4b 看 top hidden layer 的统计量和不同方法给出的 clipping factor。作者发现：

1. step-b 时 gradient norm 突然变大，说明 loss curvature 可能变陡。
2. GC 和 AGC 的 measurement 虽然变大，但给出的 clipping factor 仍然不够小。
3. Clippy 的 measurement 对 step-b 的变化更敏感，clipping factor 可以比 GC/AGC 小两个数量级。

### 7.1 为什么 step-b 是关键窗口

step-c 已经太晚了。论文观察到 step-c 时某些层已经进入梯度非常小的区域，例如类似 dying ReLU 的状态。此时 loss 已经 full-diverged，继续训练也难恢复。

真正应该干预的是 step-b：loss 刚开始异常、AUC 刚开始变差、参数还没有完全进入不可恢复区。

因此，一个稳定性方法不能只在梯度已经爆炸到非常大时才介入，而要在“实际 update 开始不合理”时就缩小步长。Clippy 的设计正是面向这个时间点。

### 7.2 为什么 $L_\infty$ 风格更敏感

GC 和 AGC 常用 $L_2$ norm。如果一个层有很多参数，少数坐标剧烈变化可能被整体范数稀释。

举个简化例子。一个向量有 10000 个坐标，其中 9999 个坐标变化很小，1 个坐标变化极大。$L_2$ norm 当然会增加，但变化可能没有“最大坐标变化”那么显著。Clippy 更接近逐坐标检查：

```text
for each coordinate i:
  actual_update_i should not exceed a bound based on w_i and absolute threshold
take the smallest scale that satisfies all coordinates
```

这让它对少数坐标的突变更敏感。推荐模型里这很重要，因为某些 sparse feature、task head 或 shared layer 坐标可能在特定 batch 上发生异常。

## 8. Algorithm 1：Adagrad with Clippy

论文给出的 Algorithm 1 是 Clippy 的核心。

![Algorithm 1: Adagrad with Clippy](/images/blog/clippy-multitask-ranking-stability/clippy-algorithm1-adagrad-with-clippy.webp)

*Source: Tang et al., KDD 2023, Algorithm 1, CC BY 4.0.*

算法逐步如下。

第一步，计算 stochastic gradient：

$$
g_t = \frac{\partial L(w_t)}{\partial w_t}
$$

第二步，更新 Adagrad accumulator：

$$
G_t = G_{t-1}+g_t^2
$$

第三步，计算 Adagrad preconditioned update：

$$
r_t = g_t \cdot G_t^{-1/2}
$$

第四步，计算 clipping factor：

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

第五步，应用裁剪后的更新：

$$
w_{t+1}=w_t-\eta_t\sigma_t r_t
$$

### 8.1 Clippy 到底裁剪什么

Clippy 裁剪的是 $\eta_t r_t$ 这个实际参数更新，而不是 $g_t$ 本身。换句话说，Clippy 约束：

$$
\eta_t \sigma_t |r_t|
\le
\lambda_{rel}|w_t|+\lambda_{abs}
$$

这可以解释为：每个坐标的参数变化，不应该超过这个坐标自身规模的一定比例，再加一个绝对安全阈值。

如果 $|w_t|$ 很大，允许的 update 也可以更大；如果 $|w_t|$ 接近 0，仍然有 $\lambda_{abs}$ 防止阈值变成 0。

### 8.2 为什么有 absolute threshold

如果只有相对阈值：

$$
\lambda_{rel}|w_t|
$$

当某些参数初始化为 0 或非常小，例如 bias，允许 update 会接近 0，训练初期可能被过度裁剪。

加入 $\lambda_{abs}$ 后：

$$
\lambda_{rel}|w_t|+\lambda_{abs}
$$

训练初期参数很小时，absolute threshold 主导，行为更像 GC；训练后期参数变大时，相对阈值主导，行为更像 AGC。

这也是论文说 Clippy 可以从 GC-style 切换到 AGC-style 的原因。

### 8.3 为什么分母里有 learning rate

Clippy 的 clipping factor 分母是：

$$
\eta_t |r_t|
$$

这说明它约束的是实际更新幅度，而不是未乘学习率的 $r_t$。如果学习率 schedule 变化，例如 warmup 或 decay，Clippy 会相应调整。学习率越大，同样的 $r_t$ 会产生更大更新，因此 clipping factor 应该更小。

这点对推荐模型很重要。论文实验里 Clippy 的优势之一，就是能让复杂模型使用 2x learning rate 稳定训练。

## 9. Clippy、GC、AGC、LAMB 的统一视角

论文把这些方法统一写成：

$$
r_t = g_t \cdot G_t^{-1/2}
$$

$$
w_{t+1}=w_t-(\eta_t\sigma_t)r_t
$$

不同方法的差别在于 $\sigma_t$ 怎么算。

| 方法 | 裁剪因子直觉 | 主要参考量 | 可能问题 |
| --- | --- | --- | --- |
| GC | 梯度范数超过阈值就缩小 | $\lVert g\rVert_2$ | 阈值难调，和参数规模无关 |
| AGC | 梯度相对参数过大就缩小 | $\lVert w\rVert_2 / \lVert g\rVert_2$ | 仍看 raw gradient，$L_2$ 不够坐标敏感 |
| LAMB | 用参数范数调节 update | trust ratio | 可能过度忽略 update magnitude |
| Clippy | 实际 update 超出逐坐标边界就缩小 | $|w_t|$、$|r_t|$、$\eta_t$、$\lambda_{abs}$ | 仍需超参数与工程验证 |

Clippy 有三个显著特征。

第一，使用 update 而不是 raw gradient。对于 Adagrad，真正改变参数的是 $r_t$，不是 $g_t$。

第二，使用逐坐标边界，近似 $L_\infty$ 风格控制。这样对少数坐标突变更敏感。

第三，考虑 learning rate。相同 update direction，在不同学习率下风险不同。

## 10. Table 1：实验模型配置

论文实验不是直接拿完整 production model 做所有对比，而是构建了简化但足够有代表性的 ranking model testbed。

![Table 1: Model settings](/images/blog/clippy-multitask-ranking-stability/clippy-table1-model-settings.webp)

*Source: Tang et al., KDD 2023, Table 1, CC BY 4.0.*

Table 1 有三个模型配置：

| Model | Non-Embedding Parameters | Shared Bottom Architecture |
| --- | ---: | --- |
| Small | 7.5M | FFN: 512 x 2 |
| Large | 57.0M | FFN: 4096 x 4 |
| Large+DCN | 68.0M | DCN + LN -> 4096 x 4 |

这里的重点不只是参数量。

Small 到 Large 增加了 shared bottom 的容量；Large+DCN 在输入上加入 DCN-v2 和 LayerNorm，进一步增加模型复杂度。论文用这三档模型构造了稳定性压力测试：

```text
Small: basic setting
Large: bigger shared bottom
Large+DCN: bigger + cross structure
```

如果一个方法只在 Small 上稳定，在 Large+DCN 上崩，那它对真实 production ranking model 的帮助就有限。

## 11. Table 2：主实验结果

Table 2 是论文最重要的结果表。

![Table 2: Overall performance](/images/blog/clippy-multitask-ranking-stability/clippy-table2-overall-performance.webp)

*Source: Tang et al., KDD 2023, Table 2, CC BY 4.0.*

实验设置要先读清楚：

1. 数据来自 YouTube production dataset。
2. 模型训练六个任务，但论文只展示两个代表性任务：一个二分类 AUC，一个回归 RMSE。
3. 每个方法先搜索 1x 或 2x learning rate。
4. 选定最佳学习率后，重复 3 个随机种子，报告均值和标准差。
5. Naive 表示没有任何训练稳定性 treatment。

### 11.1 Naive 全部 diverged

Table 2 最左侧 Naive 在 Small、Large、Large+DCN 上全部 diverged。这个结果很强：即使 Small 模型，如果没有稳定性处理，也可能发散。

这说明论文研究的不是“超大模型才偶尔遇到”的边缘问题，而是这个训练设置下系统性存在的风险。

### 11.2 GC 是很强 baseline，但复杂模型受限

GC 在 Small 和 Large 上表现不错，并且能用 2x learning rate。但在 Large+DCN 上，GC 的 best learning rate 只能是 1x，否则会出现 divergence。

这说明手动调好的 gradient clipping 可以有效，但在模型复杂度继续上升时会触到边界。

### 11.3 AGC 合理但高方差

AGC 在 Small 和 Large 上能训练，但 best learning rate 是 1x。Large+DCN 上 AGC 的结果方差明显更高，说明它已经接近稳定性极限。

AGC 的理念是好的：把梯度规模和参数规模相对化。但在这篇论文的排序模型中，它对突发 update 风险仍不够敏感。

### 11.4 LAMB 稳定但收敛差

LAMB 能避免 divergence，但 AUC 和 RMSE 不如其他强 baseline。论文认为这可能是因为 LAMB 在初期过度受参数范数影响，导致收敛慢。

这提醒我们：稳定性方法不能只是“别崩”。如果它通过过度缩小更新换来稳定，最终质量和迭代速度都会受损。

### 11.5 Clippy 的关键优势

Clippy 在三个模型上都能使用 2x learning rate，并且不牺牲收敛。尤其在 Large+DCN 上，它的 AUC 和 RMSE 都优于 GC。

论文特别强调，在他们的模型中 0.1% AUC improvement 已经是非常显著、可能带来 live metric gain 的改进。Table 2 中 Large+DCN 下 Clippy 的 AUC 为 72.37，而 GC 为 72.27，差距正好是 0.10。

这不是单纯离线表格好看，而是说明 Clippy 对复杂模型和大步长训练更有价值。

## 12. Fig. 5：AUC 曲线精读

Table 2 给最终结果，Fig. 5 展示训练过程。

![Fig. 5: AUC curves](/images/blog/clippy-multitask-ranking-stability/clippy-fig5-auc-training-curves.webp)

*Source: Tang et al., KDD 2023, Fig. 5, CC BY 4.0.*

Fig. 5a 比较 Large+DCN 上不同方法在 1x 和 2x learning rate 下的 AUC 曲线。

关键现象：

1. 1x learning rate 下，多数方法都能训练，但 LAMB 曲线偏低。
2. 2x learning rate 下，GC/AGC/LAMB 更容易出现不稳定或收敛不足。
3. Clippy 在 2x learning rate 下仍能稳定上升，并且末端 AUC 更高。

Fig. 5b 比较 Clippy 和最强 baseline GC 在不同模型设置下的 AUC。模型越复杂，Clippy 和 GC 的差距越大。

这支持论文的核心主张：Clippy 的收益不是来自简单模型，而是在模型更大、更复杂、学习率更激进时更明显。

## 13. Fig. 6：不同层的 clipping factor

Fig. 6 看 Clippy 在不同层上的 clipping factor。

![Fig. 6: Clipping factors by layer](/images/blog/clippy-multitask-ranking-stability/clippy-fig6-layer-clipping-factors.webp)

*Source: Tang et al., KDD 2023, Fig. 6, CC BY 4.0.*

图中展示了 Large+DCN 模型训练过程中几类参数的 clipping factor：

1. DCN weights
2. bottom hidden weights
3. top hidden weights
4. classification task weights
5. regression task weights

clipping factor $\sigma \in (0,1]$，越小表示裁剪越强。

论文观察到 bottom layers 被裁剪得更多。直觉上这合理：

1. bottom layer 参数范数可能更小，因此相对阈值更小。
2. bottom layer 的小变化会影响后续所有 shared 和 task-specific 层，放大到模型输出。
3. 多任务模型里底层异常更容易传播到多个任务。

这给工程实践一个启发：训练稳定性监控不能只看最后一层或总梯度 norm。shared bottom、embedding、cross layer、task head 都应该分别记录 update norm 和 clipping factor。

## 14. Supplementary：Table 3 的因果线索

论文补充材料 Table 3 用 ablation 支持 Section 3.2 的原因分析。

![Table 3: Divergence ratio ablation](/images/blog/clippy-multitask-ranking-stability/clippy-table3-divergence-ratio.webp)

*Source: Tang et al., KDD 2023, Table 3, CC BY 4.0.*

实验使用 Large+DCN，不加任何训练稳定性 treatment，并把 learning rate 设为 0.4x，让模型处在 edge of instability 附近。

结果：

| Change | Diverged / Tried | Divergence Ratio |
| --- | ---: | ---: |
| Clean | 5 / 5 | 100% |
| Smaller model size | 1 / 3 | 33% |
| Remove DCN | 2 / 3 | 66% |
| Remove subset of input features | 2 / 3 | 66% |
| Remove one task | 2 / 3 | 66% |
| Remove two tasks | 0 / 3 | 0% |

这张表不能被解读成严格因果证明，因为每组 trial 数很少，且 production dataset 不公开。但它提供了很有价值的工程证据：

1. 模型越大，越容易不稳定。
2. DCN 这类增加交叉表达的结构会增加风险。
3. 输入特征越多，数据分布突变和梯度异常的来源越多。
4. 输出任务越多，多任务梯度冲突和异常传播越严重。

尤其是 remove two tasks 后 0/3 diverged，和论文关于多任务 shared layer 风险的解释是对齐的。

## 15. Fig. 8：其他层的统计

Fig. 4 只展示 top hidden layer。补充材料 Fig. 8 展示其他层。

![Fig. 8: Statistics from other layers](/images/blog/clippy-multitask-ranking-stability/clippy-fig8-other-layer-stats.webp)

*Source: Tang et al., KDD 2023, Fig. 8, CC BY 4.0.*

Fig. 8 包括：

1. bottom hidden layer weights
2. classification task layer weights
3. regression task layer weights

论文指出，除了 binary classification layer 外，其他层的行为和 top hidden layer 类似。更关键的是，在 classification task layer 上，GC/AGC 用的 measurement 甚至没有捕捉到 step-b 的突变，导致 step-b 没有应用裁剪。

这进一步支持 Clippy 的设计：只看 gradient norm 或 gradient/weight norm 可能漏掉特定层、特定坐标上的风险；而基于 actual update 的逐坐标约束更敏感。

## 16. Fig. 7：Transformer 附加实验

论文还在一个 Transformer 翻译任务上做了附加实验。

![Fig. 7: Transformer translation experiment](/images/blog/clippy-multitask-ranking-stability/clippy-fig7-transformer-translation.webp)

*Source: Tang et al., KDD 2023, Fig. 7, CC BY 4.0.*

这个实验基于 init2winit 的 `translate_wmt` 数据集和 `xformer_translate` 模型，任务是 English to German translation。作者比较了 AdamW、Adagrad 和 Adagrad with Clippy。

结果要谨慎解读：

| 方法 | 论文描述 |
| --- | --- |
| AdamW | 最佳验证 error rate 为 31.6%，仍是该任务强 baseline |
| Adagrad | 最佳验证 error rate 为 36.9%，且较容易 diverge |
| Adagrad with Clippy | 最佳验证 error rate 为 33.4%，没有超过 AdamW，但明显好于 Adagrad |

这说明 Clippy 的思想不只限于推荐模型，但论文的核心证据仍然来自 YouTube ranking model。不能把这个附加实验外推为“Clippy 优于 AdamW”。

更合理的结论是：当你因为 sparse features、推荐场景或历史原因使用 Adagrad 时，Clippy 可以显著改善 Adagrad 的稳定性和可用学习率范围。

## 17. 论文-代码对照：TensorFlow Recommenders 的 ClippyAdagrad

论文摘要里写明开源实现位于 TensorFlow Recommenders。当前官方源码路径是：

```text
tensorflow_recommenders/experimental/optimizers/clippy_adagrad.py
```

官方 raw source 可见于：

```text
https://raw.githubusercontent.com/tensorflow/recommenders/main/tensorflow_recommenders/experimental/optimizers/clippy_adagrad.py
```

当前源码的核心对象包括：

| 符号/函数 | 作用 |
| --- | --- |
| `shrink_by_references()` | 给定 tensor、references、relative factors 和 absolute factor，计算统一缩放系数 |
| `ClippyAdagrad` | Keras optimizer，实现 Adagrad variant with adaptive clipping |
| `variable_relative_threshold` | 对应变量相对阈值，接近论文 $\lambda_{rel}$ |
| `absolute_threshold` | 对应绝对阈值，接近论文 $\lambda_{abs}$ |
| `accumulator_relative_threshold` | 源码扩展，允许参考 Adagrad accumulator 的 inverse sqrt |
| `export_clipping_factors` | 导出每个变量的 clipping factor，方便诊断 |
| `clip_accumulator_update` | 源码扩展，可选择用 clipped gradient 更新 accumulator |
| `use_standard_accumulator_update` | 是否采用标准 Adagrad 的先更新 accumulator 再算 step |

### 17.1 shrink_by_references()

`shrink_by_references()` 的语义可以概括为：

```text
given tensor delta
given references ref_j
given relative_factors factor_j
given absolute_factor abs

find largest scale in [0, 1]
such that for every coordinate i:
  |delta_i| * scale
    <= sum_j |ref_j_i| * factor_j + abs
```

这正是 Clippy 的核心边界思想。对于每个坐标，允许变化量由参考值和阈值决定；最终选择所有坐标都满足条件的最大全局 scale。

源码返回两个东西：

1. scaled tensor
2. scalar scaling factor

这和论文中每层一个 clipping factor 的设定一致。

### 17.2 update_step() 与论文公式

源码里 `update_step()` 的核心逻辑可以对应到论文公式：

| 论文 | 源码语义 |
| --- | --- |
| $G_t$ | `accumulator` |
| $G_t^{-1/2}$ | `precondition = rsqrt(accumulator_values + epsilon)` |
| $r_t = g_t G_t^{-1/2}$ | `grad_values * precondition` |
| $\eta_t r_t$ | `delta = lr * grad_values * precondition` |
| clipping factor | `shrink_by_references(delta, ...)` 返回的 `clipping_factor` |
| $w_{t+1}=w_t-\eta_t\sigma_t r_t$ | `variable.assign_sub(clipped_delta)` 或 `scatter_sub` |

这说明源码实现裁剪的是 `delta`，也就是包含 learning rate 的实际参数更新，而不是 raw gradient。

### 17.3 references 的差异：源码比论文更工程化

论文 Algorithm 1 的阈值是：

$$
\lambda_{rel}|w_t|+\lambda_{abs}
$$

当前源码在 `shrink_by_references()` 中传入：

```text
references = [variable_values, precondition]
relative_factors = [
  variable_relative_threshold,
  accumulator_relative_threshold
]
absolute_factor = absolute_threshold
```

因此源码的边界更接近：

$$
|delta_i| \cdot scale
\le
\lambda_{var}|w_i|
+ \lambda_{acc}|G_i^{-1/2}|
+ \lambda_{abs}
$$

其中 `accumulator_relative_threshold` 默认是 0.0，但如果设为正数，就会加入 accumulator 相关参考项。源码注释说这可以在训练后期收紧 clipping threshold。

这点必须在文章里写清：**当前 TFRS 源码不是论文 Algorithm 1 的逐字实现，而是工程化扩展版本**。

### 17.4 accumulator update 的工程选项

论文 Algorithm 1 先更新 accumulator，再计算 $r_t$。但源码默认支持 delayed accumulator update：

```text
use_standard_accumulator_update = False
```

源码注释解释了动机：如果先不知道 clipped gradient，就无法用 clipped value 更新 accumulator。延迟 accumulator update 可以让实现选择：

1. 用原始 gradient 更新 accumulator。
2. 用 clipped gradient 更新 accumulator。

这由 `clip_accumulator_update` 控制。

这也是源码比论文更工程化的地方。真实优化器需要处理 outlier gradient、IndexedSlices、Keras serialization、变量级 clipping factor export 等问题。

### 17.5 IndexedSlices 支持

源码显式处理 `tf.IndexedSlices`。这对推荐系统很重要，因为 embedding lookup 的梯度常常是稀疏的。

如果不支持 IndexedSlices，优化器可能在推荐模型里性能很差，或者把稀疏更新转成 dense 更新导致内存和计算不可接受。

TFRS 的实现对稀疏梯度使用 `gather`、`scatter_add` 和 `scatter_sub`，说明它不是只为 dense toy model 写的。

### 17.6 export_clipping_factors 的诊断价值

论文 Fig. 6 展示了不同层的 clipping factor。源码里的 `export_clipping_factors` 正好服务这类诊断。

工程实践里，建议把 clipping factor 作为训练监控指标：

| 指标 | 用途 |
| --- | --- |
| per-layer min clipping factor | 找到被强裁剪层 |
| clipping factor distribution | 判断是否整体过度裁剪 |
| clipping frequency | 看裁剪是否只在异常 batch 发生 |
| clipping vs AUC/loss | 判断裁剪是否阻止 divergence |
| clipping vs feature/task changes | 定位新增特征或任务引发的问题 |

如果一个模型长期所有层 $\sigma \ll 1$，说明更新被持续强行压小，可能会影响收敛；如果几乎从不裁剪，又不能阻止 divergence，阈值可能过松。

## 18. 如何在自有多任务 ranker 中接入 Clippy

Clippy 不是“把优化器换掉就完事”。更稳的接入方式是把它放进训练稳定性实验框架。

### 18.1 先定义训练失败

不要只说“loss 爆了”。建议定义：

| 失败类型 | 判定 |
| --- | --- |
| micro-divergence | loss 瞬时跳升或关键任务 AUC 瞬时下降，但后续恢复 |
| full divergence | loss 持续异常，核心任务 AUC 接近随机或 RMSE 极端恶化 |
| unstable convergence | 没有 full divergence，但方差大、seed 间差异大 |
| over-clipping | 指标稳定但收敛明显变慢，clipping factor 长期过小 |

训练平台应记录 divergence 的 step、batch 时间、样本窗口、模型版本、特征版本、任务权重和 optimizer 配置。

### 18.2 先做 baseline 矩阵

至少比较：

1. Naive Adagrad
2. GC
3. AGC
4. LAMB 或现有 trust-ratio 方法
5. Clippy

每个方法至少跑：

1. 当前学习率
2. 更高学习率，例如 1.5x 或 2x
3. 多个随机种子
4. 一个小模型和一个复杂模型

不要只看单次最终指标。Clippy 的价值在于“让更激进、更复杂设置稳定训练”，所以实验要覆盖压力场景。

### 18.3 先在非 embedding 参数上试

论文实验把 Clippy 应用于 non-embedding model parameters。工程上也建议先这么做，因为 embedding 参数数量大、梯度稀疏、更新模式特殊。先稳定 shared bottom、DCN、task head 等 dense 参数，风险更小。

后续如果要扩展到 embedding，需要单独评估：

1. 稀疏梯度的 IndexedSlices 行为。
2. 热门 ID 与长尾 ID 的 update 差异。
3. 是否影响冷启动 item/user 的学习速度。
4. accumulator 初始化和阈值对低频特征的影响。

### 18.4 监控 update 而不是只监控 gradient

Clippy 的思想提醒我们：真正改变模型的是 update。

推荐记录：

```text
per layer:
  gradient_norm
  update_norm
  max_abs_update
  parameter_norm
  accumulator_stats
  clipping_factor
  clipping_frequency
```

同时按任务记录：

```text
per task:
  loss
  AUC / RMSE / calibration metric
  gradient contribution
  task weight
  label distribution drift
```

如果只看全局 loss，很可能错过 step-b 这类早期信号。

### 18.5 不能替代上线门禁

Clippy 解决训练稳定性，不解决所有推荐系统风险。即使训练不 diverge，也可能因为数据问题、目标错配、标签漂移或特征穿越导致线上效果差。

上线仍需要：

1. 离线指标门禁。
2. 训练过程异常检测。
3. 模型 sanity check。
4. Shadow serving。
5. 小流量 A/B。
6. Guardrail 指标。
7. Kill switch 和 rollback。

这点非常重要：优化器稳定性是训练链路的一环，不是推荐系统质量的总保证。

## 19. 与成熟推荐训练流程的关系

这篇论文可以和两个方向连起来。

第一个方向是 YouTube DNN。YouTube DNN 2016 讲的是 candidate generation 和 ranking 的系统拆分；Clippy 讲的是 ranking model 变大、变多任务之后，训练如何保持稳定。

第二个方向是冷启动到反馈闭环。冷启动文章讨论的是样本语义、曝光日志、负样本边界、ANN 低权重上线。Clippy 解决的是闭环建立之后，大规模 ranker 训练过程里的优化稳定性。

可以把成熟推荐系统的演进写成：

```text
logging and exposure contract
  -> candidate generation
  -> ranking model
  -> multitask objectives
  -> larger model and richer features
  -> training stability bottleneck
  -> update clipping / optimizer engineering
  -> safe model publishing and A/B
```

Clippy 位于这条链的后半段。没有可靠曝光日志和样本契约，训练再稳定也可能学错目标；没有稳定训练，模型容量和多任务目标又难以持续迭代。

## 20. 局限性与批判

### 20.1 数据和模型不可完整复现

论文实验基于 YouTube production dataset，数据不公开，模型也只是描述了简化 testbed。读者无法完整复现 Table 2 的结果。

这不削弱论文的工程价值，但意味着我们不能把数值结果当作通用 benchmark。

### 20.2 原因分析含 conjecture

论文对 root cause 的总结非常合理，也和 edge-of-stability 相关工作一致。但其中关于推荐模型为什么更不稳定的部分，仍包含 conjecture 和经验归因。Table 3 提供支持证据，但样本数有限。

### 20.3 Clippy 仍有超参数

Clippy 有 $\lambda_{rel}$ 和 $\lambda_{abs}$。当前 TFRS 源码还引入了 `accumulator_relative_threshold`、`clip_accumulator_update` 等选项。它减少了某些手动调阈值负担，但不是完全免调。

### 20.4 稳定性和收敛仍需权衡

过强裁剪会让训练变慢。论文中 LAMB 的例子说明“稳定但收敛差”不是好结果。Clippy 在论文设置里取得了更好 trade-off，但其他模型仍需要验证。

### 20.5 对 AdamW 等优化器的结论有限

论文附加实验显示 Adagrad with Clippy 在 Transformer 翻译任务上好于 Adagrad，但没有超过 AdamW。Clippy 可以适配其他 optimizer，但理论和大规模实证仍不足。

### 20.6 不解决推荐目标问题

Clippy 不解决：

1. selection bias
2. exposure bias
3. label leakage
4. 多目标权重错配
5. long-term satisfaction
6. 线上策略约束

它让模型更稳定地学习你给它的目标，但目标本身是否正确是另一个问题。

## 21. 推荐阅读路径

如果读原论文，建议顺序如下：

1. Abstract：先确认论文目标是 training stability，不是 ranking architecture。
2. Fig. 1 + Section 2.1：理解 micro/full divergence。
3. Fig. 2 + Section 3.1：理解多任务 ranking model。
4. Section 3.2 + Fig. 3：理解 root cause 和 edge of stability。
5. Fig. 4：重点读训练动态和 GC/AGC/Clippy 差异。
6. Algorithm 1：逐行读 Clippy 的 update clipping。
7. Section 4.3：读 GC/AGC/LAMB/Clippy 的统一形式。
8. Table 1 + Table 2：理解实验设置和主结果。
9. Fig. 5 + Fig. 6：看训练曲线和 clipping factor 诊断。
10. Appendix Table 3 + Fig. 8：看补充证据。
11. TFRS `clippy_adagrad.py`：最后看源码里的工程扩展。

## 22. 结论

Clippy 这篇论文的价值，不是告诉我们“梯度大就裁剪一下”，而是把推荐排序模型训练不稳定拆成了一个更精确的问题：

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

它的解决思路也很清楚：不要只看 raw gradient norm，而要看实际参数 update；不要只用全局 $L_2$ norm，而要对少数坐标的突变敏感；不要忽略学习率和参数自身尺度，而要把它们都放进 clipping factor。

对于真实推荐系统，这篇论文给出的启发是：

1. 大模型训练稳定性应该作为一等工程指标。
2. 多任务 shared bottom 的异常传播要被单独监控。
3. 新特征、新任务、新结构上线前要做稳定性压力测试。
4. 优化器实验要同时看稳定性、收敛速度和最终质量。
5. Clipping factor 本身是重要诊断信号，不只是优化器内部变量。
6. 稳定训练不等于推荐系统正确，还需要日志、样本、目标和上线保护。

如果你正在做多任务推荐 ranker，并且已经遇到“模型扩容或加任务后偶发 loss blow-up”的问题，Clippy 值得作为一个严肃 baseline。它最适合的位置不是替代所有训练治理，而是成为训练稳定性工具箱里的一个强组件。

## 参考资料

1. Jiaxi Tang et al. [Improving Training Stability for Multitask Ranking Models in Recommender Systems](https://dl.acm.org/doi/10.1145/3580305.3599846). KDD 2023.
2. arXiv. [Improving Training Stability for Multitask Ranking Models in Recommender Systems](https://arxiv.org/abs/2302.09178).
3. Google Research. [Improving Training Stability for Multitask Ranking Models in Recommender Systems](https://research.google/pubs/improving-training-stability-for-multitask-ranking-models-in-recommender-systems/).
4. TensorFlow Recommenders. [Experimental optimizers directory](https://github.com/tensorflow/recommenders/tree/main/tensorflow_recommenders/experimental/optimizers).
5. TensorFlow Recommenders. [ClippyAdagrad raw source](https://raw.githubusercontent.com/tensorflow/recommenders/main/tensorflow_recommenders/experimental/optimizers/clippy_adagrad.py).
6. Jeremy Cohen et al. [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065).
7. Justin Gilmer et al. [A Loss Curvature Perspective on Training Instability in Deep Learning](https://arxiv.org/abs/2110.04369).
8. Andy Brock et al. [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171).

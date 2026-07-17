---
title: "Generative Recommenders 论文精读：HSTU、M-FALCON 与万亿参数推荐序列建模"
description: "精读 Meta ICML 2024 Generative Recommenders 如何将推荐系统重构为序列生成问题，并通过 HSTU、Stochastic Length、M-FALCON 和 scaling law 支撑工业级部署"
pubDate: "2026-06-29T11:57:06+08:00"
updatedDate: "2026-06-29T11:57:06+08:00"
tags:
  - "Paper Reading"
  - "Recommendation System"
  - "Generative Recommenders"
  - "Sequential Recommendation"
  - "HSTU"
  - "Meta"
draft: false
---

Jiaqi Zhai、Lucy Liao、Xing Liu、Yueming Wang、Rui Li、Xuan Cao、Leon Gao、Zhaojie Gong、Fangda Gu、Michael He、Yinghai Lu 和 Yu Shi 在 ICML 2024 发表的 **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations**，是一篇野心很大的推荐系统论文。它试图回答一个工业界长期存在、但学术基准很难完整暴露的问题：

> 推荐系统是否也存在类似语言模型和视觉模型那样的 compute scaling law？如果有，传统 DLRM 范式为什么没有明显吃到这个红利？

论文给出的答案是 **Generative Recommenders**，简称 GRs。它不是把一个 Transformer 模块塞进 ranker，也不是把用户历史做成一个更长的 DIN 序列特征，而是把推荐任务重新表述为 **sequential transduction**：把内容 token、动作 token、目标候选 token 放在同一个序列建模框架里，让 ranking 和 retrieval 都可以被看作 target-aware 的自回归生成问题。

为了让这个范式能真的部署在十亿用户级系统中，论文又提出了三个关键工程组件：

1. **HSTU**：Hierarchical Sequential Transduction Unit，一个面向高基数、非平稳、流式推荐数据的自注意力 encoder。
2. **Stochastic Length**：训练时人为制造长序列稀疏性，让模型在有限训练成本下适应更长历史。
3. **M-FALCON**：Microbatched-Fast Attention Leveraging Cacheable OperatioNs，在 target-aware ranking 推理中通过 microbatch 与 cache 摊销候选评分成本。

这篇论文的摘要数字很醒目：HSTU 在 synthetic 和公开数据集上最高提升 `65.8%` NDCG；在 8192 长度序列上比 FlashAttention2-based Transformers 快 `5.3x` 到 `15.2x`；HSTU-based GRs 扩展到 `1.5 trillion` 参数，并在生产线上 A/B 中带来 `12.4%` 指标提升。本文会逐项拆开这些数字背后的任务定义、实验边界、模型设计和代码实现。

本文插入的论文图表来自 arXiv/PMLR PDF，只做等比例裁切展示，并在图注中标注来源。官方代码仓库 `meta-recsys/generative-recommenders` 已公开，本文会做源码静态阅读，但不会声称复现 Meta production dataset、1.5T 参数训练或线上 A/B。

## 1. 论文信息与一句话贡献

论文基本信息如下。

| 项目 | 内容 |
| --- | --- |
| 题名 | Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations |
| 作者 | Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, Yinghai Lu, Yu Shi |
| 会议 | ICML 2024, PMLR 235 |
| arXiv | `2402.17152` |
| 页数 | 26 页 |
| 官方代码 | `meta-recsys/generative-recommenders` |
| 主要关键词 | Generative Recommenders, HSTU, Stochastic Length, M-FALCON, scaling law |

一句话概括这篇论文：

> 它把工业推荐从“异构特征工程 + 多塔/多任务 DLRM”重新表述成“内容和动作交错的序列生成问题”，再用 HSTU 和 M-FALCON 解决长序列训练与 target-aware 候选评分的计算瓶颈。

这句话里有三层含义。

第一，论文批评的不是某个 DLRM 结构，而是 DLRM 的整体扩展方式。传统 DLRM 能处理大量离散特征、稠密特征、交叉特征和统计特征，但它通常依赖人工设计的特征组织方式，模型结构也被业务源、任务、特征簇切成很多模块。这样的系统很强，但不一定能像语言模型那样通过持续加大训练 compute 获得可预测的质量提升。

第二，论文提出的 GRs 不是传统 sequential recommender 的简单放大版。SASRec、BERT4Rec、GRU4Rec 主要把用户历史 item 当作序列；GRs 则把内容表示 $\Phi_i$ 和用户动作 $a_i$ 交错起来，并把 ranking 的候选 $\Phi_{i+1}$ 显式放进条件里。

第三，论文很重视推理成本。target-aware ranking 的难点是每个候选都要和用户历史交互，如果候选数是 $m$、历史长度是 $n$，朴素做法会把同一段历史重复算很多次。M-FALCON 的价值就在于把这部分共享计算缓存和摊销掉。

## 2. 从 Fig. 1 看论文想解决的尺度问题

Fig. 1 是全文立场最鲜明的一张图：论文把 DLRM、GR 和语言/视觉模型放在同一张训练 compute 坐标里。

![Fig. 1: DLRM 与 GR 训练 compute 趋势](/images/blog/generative-recommenders-hstu/hstu-fig01-training-compute.webp)

*Source: Zhai et al., ICML 2024, Fig. 1, CC BY 4.0.*

图中红点是语言和视觉模型，例如 AlexNet、ResNet、BERT、AlphaZero、GPT-3、LLaMA-2；蓝色星号是 DLRM 与 GR。论文想表达的不是“推荐模型也很大”，而是两个更具体的判断。

第一，过去工业 DLRM 的训练 compute 并没有沿着语言模型那条曲线快速上升。推荐业务每天处理的用户行为 token 数量极大，理论上并不缺数据，但 DLRM 的结构和训练流程使得它很难直接通过扩大模型和训练计算获得可预测收益。

第二，GR-23 和 GR-24 在图中被画到接近 GPT-3/LLaMA-2 的训练 compute 量级，这对应论文声称的生产部署规模。这里的关键不是模型参数一定要大到万亿，而是推荐系统开始被论文放进“scaling system”的视角里讨论。

对工程读者来说，Fig. 1 的启发是：推荐系统的下一阶段瓶颈可能不只是更好的特征、更好的双塔、更好的 ranker，而是 **训练范式是否能稳定消耗更多数据和 compute**。如果范式本身不能 scale，继续叠局部模块的边际收益会变低。

## 3. 背景：为什么工业 DLRM 难以继续 scale

论文对 DLRM 的批评可以拆成四类。

第一是 **特征空间缺少统一结构**。工业推荐使用的特征包括 creator id、user id、item id、上下文、计数、比率、cross feature、用户历史 embedding、候选内容 embedding 等。这些特征并不像文本 token 那样天然排成一条序列。DLRM 往往用 feature extraction、feature interaction、task tower 组合它们，结构复杂且高度工程化。

第二是 **vocabulary 高基数且动态变化**。语言模型面对的词表可能是 10 万量级静态 token，而推荐系统的 item、creator、content、ad、product 可能是十亿级，而且每天新增和下架。高基数动态 vocabulary 会带来 embedding、负采样、检索、冷启动和线上 serving 的连锁问题。

第三是 **训练数据是流式的**。论文反复强调 large-scale recommendation systems 不是离线 full-shuffle 小基准。真实系统持续接收用户行为，模型也往往持续训练。数据分布、候选池、业务目标都会随时间变化。

第四是 **候选评分需要 target-aware**。ranking 不是只输出一个用户向量再 ANN 检索，它要判断“这个用户在这个上下文下是否会对这个候选采取某种动作”。候选本身必须进入模型交互路径。这个要求提高了质量，也提高了推理成本。

这些问题共同解释了为什么“用普通 Transformer 直接替换 DLRM”并不够。普通 self-attention 的计算和内存成本在长序列下很高，softmax attention 对推荐数据的稀疏性和非平稳性也不一定最合适。

## 4. DLRM vs Generative Recommenders

Fig. 2 展示了论文最核心的范式转换。

![Fig. 2: DLRM 与 GR 的特征组织和训练流程](/images/blog/generative-recommenders-hstu/hstu-fig02-dlrm-vs-gr-features.webp)

*Source: Zhai et al., ICML 2024, Fig. 2, CC BY 4.0.*

左侧是传统 DLRM：不同特征簇通过各自的 feature extraction 和 interaction 模块进入任务预测层。图里的 `E,F,G,H` 表示 categorical features，不同颜色代表不同特征源。一个典型 DLRM 会有底层 embedding、特征交叉、序列 attention、dense MLP、任务塔等多个子模块。

右侧是 Generative Recommenders：它把内容、动作、辅助时间序列、主时间序列等信息组织成一条统一序列。内容表示记为 $\Phi_i$，动作记为 $a_i$。模型训练时不再只是把用户历史编码成一个向量，而是在目标感知的序列上学习下一个 token 或动作。

可以把 DLRM 到 GR 的变化理解为：

| 维度 | DLRM | Generative Recommender |
| --- | --- | --- |
| 输入组织 | 多个特征簇、多个模块 | 内容 token 与动作 token 交错序列 |
| 候选处理 | ranking 阶段候选进入特征交互 | 候选作为 target-aware 条件进入序列 |
| 训练方式 | 多任务或点式监督 | causal autoregressive / sequential transduction |
| 扩展方式 | 加特征、加 tower、加 cross | 加序列长度、模型容量、训练 compute |
| 主要瓶颈 | 特征和模块复杂度 | 长序列 attention 与候选评分成本 |

论文不是否定 DLRM 的有效性。相反，它承认 DLRM 在工业推荐中非常强，Fig. 9 还展示了一个强 DLRM baseline。但论文认为，如果目标是跨数量级地扩大训练 compute，GR 更接近语言模型式的统一建模范式。

## 5. Ranking 和 retrieval 如何变成 sequential transduction

Table 1 是 GR 形式化的入口。

![Table 1: Ranking 与 retrieval 作为 sequential transduction tasks](/images/blog/generative-recommenders-hstu/hstu-table01-sequential-transduction-tasks.webp)

*Source: Zhai et al., ICML 2024, Table 1, CC BY 4.0.*

传统 ranking 可以写成一个动作预测问题。设用户已经消费或看到过内容 $\Phi_0,\Phi_1,\ldots,\Phi_i$，并产生动作 $a_0,a_1,\ldots,a_i$。如果不显式考虑新候选，动作预测近似为：

$$
p(a_{i+1}\mid \Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_i,a_i).
$$

但 ranking 的实际问题是：给定候选 $\Phi_{i+1}$，预测用户是否会对这个候选采取动作 $a_{i+1}$。因此 GR 的 target-aware ranking 更接近：

$$
p(a_{i+1}\mid \Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_i,a_i,\Phi_{i+1}).
$$

这里的差异非常关键。$\Phi_{i+1}$ 被放进条件以后，模型可以直接学习用户历史和当前候选之间的交互，而不是把用户表示和候选表示分开编码后再用简单相似度比较。

Retrieval 则更像内容 token 预测：

$$
p(\Phi_{i+1}\mid \Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_i,a_i).
$$

这对应“下一个用户可能会接触的内容是什么”。它更适合召回或生成候选，而 ranking 更关注给定候选下的动作概率。

这也解释了论文标题里的 “Actions Speak Louder than Words”。在推荐里，用户动作 $a_i$ 不是附加标签，而是建模主语义的一部分。点击、停留、点赞、购买、跳过、隐藏等动作本身构成了偏好序列。

## 6. Table 10 和 Table 11：GR 与传统 sequential recommender 的边界

附录 Table 10 对比了相关工作。

![Table 10: sequential recommender 与 GR 相关工作对比](/images/blog/generative-recommenders-hstu/hstu-table10-related-work-ranking.webp)

*Source: Zhai et al., ICML 2024, Table 10, CC BY 4.0.*

这张表的重要性在于，它避免了一个常见误解：GR 不是 SASRec 的简单放大。SASRec、GRU4Rec、BERT4Rec 等方法通常预测下一个 item token；GR 的 ranking 设置要处理目标候选和动作，训练过程也更接近 target-aware 的 causal autoregressive formulation。

Table 11 则把内容和动作交错的生成目标写得更明确。

![Table 11: 对内容与动作序列的生成建模](/images/blog/generative-recommenders-hstu/hstu-table11-generative-modeling.webp)

*Source: Zhai et al., ICML 2024, Table 11, CC BY 4.0.*

表中核心问题是：模型到底生成什么？

一种方式是生成下一个 action token：

$$
p(a_i \mid \Phi_0,a_0,\ldots,\Phi_i).
$$

另一种方式是生成下一个 content token：

$$
p(\Phi_i \mid \Phi_0,a_0,\ldots,a_{i-1}).
$$

两者对应不同业务问题。Ranking 关注动作；retrieval 关注内容。GR 的优势在于它试图用同一个序列建模框架统一这两类任务，而不是为每个任务单独堆一套系统。

## 7. HSTU：不是普通 Transformer block

Fig. 3 给出了 DLRM 与 GR/HSTU 的组件对比。

![Fig. 3: DLRM 与 GR/HSTU 的关键组件](/images/blog/generative-recommenders-hstu/hstu-fig03-model-components.webp)

*Source: Zhai et al., ICML 2024, Fig. 3, CC BY 4.0.*

HSTU 的全称是 **Hierarchical Sequential Transduction Unit**。论文设计它的动机是：推荐数据具有高基数、强稀疏、非平稳、长序列、target-aware 候选交互等特性，直接用标准 Transformer 会遇到质量和效率问题。

一个 HSTU 层的核心公式是：

$$
U(X),V(X),Q(X),K(X)=\mathrm{Split}(\phi_1(f_1(X))).
$$

这里 $X$ 是输入序列表示，$f_1$ 是 pointwise transformation，$\phi_1$ 是非线性函数。与标准 Transformer 不同，HSTU 同时产生 $U,V,Q,K$，其中 $U(X)$ 是 gating weights。

注意力权重写作：

$$
A(X)=\phi_2(Q(X)K(X)^T+r_{ab}^{p,t}).
$$

$r_{ab}^{p,t}$ 是 relative attention bias，融合相对位置和相对时间。推荐序列不像自然语言那样只有 token 位置，事件之间的时间差也很重要。一个用户昨天点击和三个月前点击同一个类目的意义不同。

输出写作：

$$
Y(X)=f_2(\mathrm{Norm}(A(X)V(X))\odot U(X)).
$$

这里的关键是逐元素 gating：attention-pooled values 先经过 normalization，再和 $U(X)$ 做 elementwise product。论文认为这让 HSTU 能把 DLRM 中常见的 feature interaction、feature transformations 和 representation learning 统一到单个模块里。

从工程角度看，HSTU 的重点有四个。

第一，它使用 pointwise transformed attention，而不是标准 softmax attention。论文的 ablation 显示，替换成 softmax 会显著影响质量和训练表现。

第二，它通过 relative position/time bias 引入推荐序列中的时间结构。代码中对应 `RelativePositionalBias` 和 `RelativeBucketedTimeAndPositionBasedBias`。

第三，它用 jagged tensor path 处理变长历史序列，避免把大量 padding 当作真实 token 计算。官方代码中大量使用 `torch.ops.fbgemm.jagged_to_padded_dense` 和 `dense_to_jagged`。

第四，它支持 cache path，为 M-FALCON 这种推理摊销算法准备基础。

## 8. HSTU 的公开 ablation：Table 2 和 Table 5

论文先用 synthetic streaming setting 验证 HSTU 的 inductive hypotheses。

![Table 2: synthetic one-pass streaming setting 结果](/images/blog/generative-recommenders-hstu/hstu-table02-synthetic-streaming.webp)

*Source: Zhai et al., ICML 2024, Table 2, CC BY 4.0.*

Table 2 中 HSTU 明显优于 Transformers。更重要的是，`HSTU (-rabp,t, Softmax)` 比 `HSTU (-rabp,t)` 差很多，这说明论文不是只靠模型容量获益，而是 HSTU 的 attention/gating 结构本身在这个 synthetic streaming setting 里有价值。

Table 5 是另一组 ablation。

![Table 5: HSTU、HSTU ablation 与 Transformers 对比](/images/blog/generative-recommenders-hstu/hstu-table05-hstu-ablation.webp)

*Source: Zhai et al., ICML 2024, Table 5, CC BY 4.0.*

这张表的读法是：

1. `Transformer` 是普通基线。
2. `HSTU (-rabp,t, Softmax)` 移除了相对 bias 并使用 softmax 风格。
3. `HSTU (-rabp,t)` 保留 HSTU 的非 softmax attention/gating，但不使用 relative attention bias。
4. `HSTU (original rab)` 和 `HSTU` 加入更完整的 relative bias 设计。

结果支持两个结论：HSTU 的 pointwise attention 不是可有可无；relative bias 对推荐序列质量也有贡献。需要注意的是，这些结论仍然是在论文构造的数据和实验设置里成立，不能简单外推为“所有推荐场景都应替换成 HSTU”。

## 9. Stochastic Length：为什么要训练时人为制造稀疏性

真实推荐历史可能很长。用户一年内的曝光、点击、播放、购买、跳过可能达到很长序列。直接对完整序列训练代价很高；只训练短序列又会让模型在长上下文下泛化不好。

论文提出 Stochastic Length，核心思想是训练时对用户历史序列做随机长度采样，制造类似长序列中的稀疏性，让模型习惯在更稀疏、更长的上下文里工作。

Table 3 展示了不同 $\alpha$ 与最大序列长度下的 sparsity。

![Table 3: Stochastic Length 对序列稀疏性的影响](/images/blog/generative-recommenders-hstu/hstu-table03-sl-sparsity-main.webp)

*Source: Zhai et al., ICML 2024, Table 3, CC BY 4.0.*

可以看到，最大序列长度越长，sparsity 越高；$\alpha$ 越大，sparsity 越低。论文把 $\alpha=2.0$ 视为接近不制造额外稀疏性的情形，而较小 $\alpha$ 会显著增加稀疏性。

Fig. 4 展示 Stochastic Length 对指标的影响。

![Fig. 4: Stochastic Length 对指标的影响](/images/blog/generative-recommenders-hstu/hstu-fig04-stochastic-length-metrics.webp)

*Source: Zhai et al., ICML 2024, Fig. 4, CC BY 4.0.*

图中的柱状表示 NE difference，虚线表示 sparsity。它说明一个很实际的 tradeoff：稀疏性不是越高越好。合适的 Stochastic Length 能改善长序列泛化，但过强的稀疏会损害训练信号。

这和推荐系统里的常见经验一致。过短历史会丢失长期偏好，过长历史会引入噪声和计算压力；训练时随机化历史窗口可以让模型更稳健，但必须调参。

附录还给了 60 天和 90 天历史的 sparsity 表。

![Table 14: 60 天用户历史下 Stochastic Length 对稀疏性的影响](/images/blog/generative-recommenders-hstu/hstu-table14-sl-sparsity-60d.webp)

*Source: Zhai et al., ICML 2024, Table 14, CC BY 4.0.*

![Table 15: 90 天用户历史下 Stochastic Length 对稀疏性的影响](/images/blog/generative-recommenders-hstu/hstu-table15-sl-sparsity-90d.webp)

*Source: Zhai et al., ICML 2024, Table 15, CC BY 4.0.*

90 天历史比 60 天历史更稀疏。这进一步说明 Stochastic Length 本质上是为了让训练时看到“长历史但稀疏有效事件”的情况。

## 10. Stochastic Length 与 length extrapolation 的区别

附录 Table 16 对比了 Stochastic Length 和已有 length extrapolation 技术。

![Table 16: Stochastic Length 与 sequence length extrapolation 方法对比](/images/blog/generative-recommenders-hstu/hstu-table16-sl-vs-length-extrapolation.webp)

*Source: Zhai et al., ICML 2024, Table 16, CC BY 4.0.*

论文把 zero-shot、fine-tune、RoPE/NTK-style extrapolation 和 Stochastic Length 放在一起比较。它想表达的是：推荐场景的问题不只是位置编码能不能外推到更长长度，而是训练数据本身是否让模型习惯稀疏长历史。

Fig. 10 展示了 ranking metrics 上的附加结果。

![Fig. 10: Stochastic Length 对 ranking metrics 的影响](/images/blog/generative-recommenders-hstu/hstu-fig10-sl-ranking-metrics.webp)

*Source: Zhai et al., ICML 2024, Fig. 10, CC BY 4.0.*

对工程落地来说，Stochastic Length 更像一种训练数据构造策略，而不是模型结构技巧。它会影响：

1. 每个样本使用多少历史 token。
2. 训练时的 attention sparsity。
3. 推理时长历史的泛化。
4. 长期行为和近期行为的相对权重。

如果要把它落到自己的推荐系统里，需要把“历史截断策略”从数据管道里的随手实现，提升为可实验、可记录、可回滚的训练超参数。

## 11. M-FALCON：target-aware ranking 的部署关键

GR 的 ranking 模型把候选 $\Phi_{i+1}$ 放进条件里，这会带来一个推理问题：如果每个请求要评分 $m$ 个候选，朴素做法会对每个候选都重复计算用户历史上下文。

M-FALCON 解决的是这个问题。

![Fig. 11: M-FALCON 训练与推理示意](/images/blog/generative-recommenders-hstu/hstu-fig11-mfalcon-illustration.webp)

*Source: Zhai et al., ICML 2024, Fig. 11, CC BY 4.0.*

上半部分是训练：内容 token 和动作 token 交错进入 causal autoregressive 模型。下半部分是推理：同一用户历史后面接多个候选 $\Phi'_0,\ldots,\Phi'_{m-1}$，候选被分成多个 microbatch，每个 microbatch 一次性评分一组候选。

M-FALCON 的三个关键思想是：

1. **Batched inference for causal autoregressive settings**：把多个候选拼进同一个计算图，但通过 attention mask 防止不该发生的信息泄漏。
2. **Microbatching scales batched inference to large candidate sets**：当候选数很大时，分 microbatch 处理，控制显存和延迟。
3. **Encoder-level caching**：缓存历史部分的 $K(X)$、$V(X)$ 和中间状态，只对新增候选部分做必要计算。

Algorithm 1 给出伪代码。

![Algorithm 1: M-FALCON pseudocode](/images/blog/generative-recommenders-hstu/hstu-algorithm01-mfalcon.webp)

*Source: Zhai et al., ICML 2024, Algorithm 1, CC BY 4.0.*

用更工程化的语言说，M-FALCON 把 target-aware ranker 的候选评分从：

```text
for candidate in candidates:
  encode(history + candidate)
```

改成近似：

```text
cache = encode_shared_history(history)
for microbatch in chunk(candidates):
  score(history_cache + microbatch_candidates)
```

这不是普通 batching，因为 causal mask 和候选之间的信息隔离非常重要。一个候选的 token 不能看见另一个候选的目标输出，否则 ranking 分数会被污染。

## 12. M-FALCON 的吞吐结果

Fig. 6 是主文中的推理吞吐结果。

![Fig. 6: 不同候选评分设置下的 inference throughput](/images/blog/generative-recommenders-hstu/hstu-fig06-inference-throughput-main.webp)

*Source: Zhai et al., ICML 2024, Fig. 6, CC BY 4.0.*

图中横轴是 M-FALCON 中评分候选数 $m$，纵轴是 QPS。GR 模型的 FLOPs 远大于 DLRM，但通过 M-FALCON 和 HSTU 的效率设计，在某些候选数区间可以达到高于 DLRM 的吞吐。

附录 Fig. 12 展示端到端吞吐。

![Fig. 12: GR w/ M-FALCON 与 DLRM 的端到端吞吐](/images/blog/generative-recommenders-hstu/hstu-fig12-end-to-end-throughput.webp)

*Source: Zhai et al., ICML 2024, Fig. 12, CC BY 4.0.*

Fig. 13 则进一步看 microbatch scaling。

![Fig. 13: M-FALCON microbatch throughput scaling](/images/blog/generative-recommenders-hstu/hstu-fig13-mfalcon-throughput-scaling.webp)

*Source: Zhai et al., ICML 2024, Fig. 13, CC BY 4.0.*

这两张图的工程含义是：GR 的部署可行性不只取决于模型 FLOPs。只看单个 forward 的 FLOPs 会低估 batching、caching、microbatching 和 kernel 优化带来的吞吐收益。

但也要注意，吞吐图不是“任何团队都能直接部署 285x FLOPs GR”的证明。它依赖具体硬件、kernel、batching 方式、候选数、延迟预算、业务流量形态和工程实现。本文应把它读作“论文证明存在一条可行工程路径”，而不是“GR 一定比 DLRM 更便宜”。

## 13. HSTU encoder-level efficiency

Fig. 5 对比 HSTU 与 FlashAttention2-based Transformers 的 encoder-level efficiency。

![Fig. 5: HSTU vs FlashAttention2 Transformer 的 encoder-level efficiency](/images/blog/generative-recommenders-hstu/hstu-fig05-encoder-efficiency.webp)

*Source: Zhai et al., ICML 2024, Fig. 5, CC BY 4.0.*

这张图包含训练 NE、训练速度和推理速度。论文摘要中 `5.3x` 到 `15.2x` 的速度优势来自 8192 长度序列上的比较。

HSTU 更快的原因包括：

1. 它用更适合推荐序列的 attention/gating 设计，减少一部分标准 Transformer 中昂贵的中间结构。
2. 它不依赖普通 softmax attention 的完整流程。
3. 它通过 fused relative attention bias、sparse grouped GEMMs、jagged tensor path 等工程优化减少内存与计算开销。

在官方代码里，效率相关实现散落在 `ops/triton`、`ops/cpp/hstu_attention` 和 FBGEMM 相关调用中。本文不会编译这些 kernel，但会在代码对照里指出它们的位置和职责。

## 14. 公开数据集实验：Table 4 和 Table 12

Table 4 是主文公开数据集结果。

![Table 4: 公开数据集多轮 full-shuffle 设置结果](/images/blog/generative-recommenders-hstu/hstu-table04-public-datasets.webp)

*Source: Zhai et al., ICML 2024, Table 4, CC BY 4.0.*

公开数据集包括 ML-1M、ML-20M 和 Books。论文用 HR@10、HR@50、HR@200、NDCG@10、NDCG@50、NDCG@200 评价。HSTU 和 HSTU-large 相对 SASRec 有明显提升，在 Books 上提升尤其大。

官方 README 也给出了可复现表格。例如：

| Dataset | HSTU-large HR@10 | HSTU-large NDCG@10 | 相对 SASRec HR@10 |
| --- | ---: | ---: | ---: |
| ML-1M | 0.3294 | 0.1893 | +15.5% |
| ML-20M | 0.3556 | 0.2098 | +23.1% |
| Amazon Books | 0.0478 | 0.0262 | +56.7% |

附录 Table 12 加入 GRU4Rec 和 BERT4Rec 作为补充比较。

![Table 12: traditional sequential recommender 设置下补充比较](/images/blog/generative-recommenders-hstu/hstu-table12-traditional-sequential-results.webp)

*Source: Zhai et al., ICML 2024, Table 12, CC BY 4.0.*

这里需要读得谨慎。README 明确说明 BERT4Rec 和 GRU4Rec 行来自另一篇论文报告结果，并且使用 full negatives；HSTU/SASRec 行使用 sampled negatives。也就是说，这个比较不完全是同一训练管线下的端到端公平复现。论文把它作为完整背景，但不应把每个百分比都当成绝对公平排序。

对工程读者来说，公开数据集实验的价值主要是：HSTU 在传统 sequential recommendation benchmark 上没有退化，并且在相同配置下相对 SASRec 有增益。它不是 1.5T 生产部署结论的直接复现。

## 15. 工业 retrieval 和 ranking：Table 6、Table 7

论文真正的强主张来自工业实验。

Table 6 是 retrieval 模型对比。

![Table 6: retrieval model offline/online 对比](/images/blog/generative-recommenders-hstu/hstu-table06-retrieval-offline-online.webp)

*Source: Zhai et al., ICML 2024, Table 6, CC BY 4.0.*

Table 7 是 ranking 模型对比。

![Table 7: ranking model offline/online 对比](/images/blog/generative-recommenders-hstu/hstu-table07-ranking-offline-online.webp)

*Source: Zhai et al., ICML 2024, Table 7, CC BY 4.0.*

这些表格包含 offline NE、HR@K 以及 online metrics。论文报告 GR 在 retrieval 和 ranking 上都能带来线上提升，并在 ranking 中报告了 `+12.4%` 的主要线上指标提升。

读这部分时要把三个层次分开。

第一，`DLRM` 是强 baseline，不是随便搭的 MLP。附录还说明 baseline 反映了多年工业迭代经验。

第二，`GR (interactions only)`、`GR (new source)`、`GR (replace source)` 是不同接入方式。它们分别对应只用交互历史、新增一个 GR source、替换主要 DLRM source 等不同工程策略。

第三，online metric 是相对业务指标，不是公开 benchmark。我们无法从论文复现它，也无法知道所有 guardrail、traffic split、业务场景细节。因此正确表述是：论文报告了在 Meta 生产环境中的线上收益，而不是公开可验证的通用定理。

## 16. Scaling law：Fig. 7 的意义与边界

Fig. 7 是全文最有野心的结果。

![Fig. 7: DLRM 与 GR 在大规模工业设置下的 scaling behavior](/images/blog/generative-recommenders-hstu/hstu-fig07-industrial-scaling.webp)

*Source: Zhai et al., ICML 2024, Fig. 7, CC BY 4.0.*

图中 DLRM 和 GR 在训练 compute 增加时表现不同：DLRM 更容易出现质量饱和，而 GR 质量随 compute 呈现更接近 power-law 的改善趋势。

这张图如果成立，意义非常大。它意味着推荐系统也可能出现类似 foundation model 的扩展路径：先建立能吸收 compute 的统一建模范式，然后持续扩展数据、序列长度、模型容量和训练计算。

但边界也必须明确。

1. 这是工业内部数据和系统上的经验 scaling law，不是公开 benchmark 上可独立验证的普适规律。
2. 横轴训练 compute、模型参数、序列长度、特征范围和工程优化并不完全独立。
3. DLRM baseline 的饱和不代表所有 DLRM 变体都必然饱和。
4. GR 的 scaling 需要 HSTU、Stochastic Length、M-FALCON、kernel 优化、训练系统和线上 pipeline 共同支撑。

因此，Fig. 7 最适合被读作一个研究方向信号：如果推荐系统要进入 foundation model 式扩展，GR 是一条已经被生产系统验证过的候选路径。

## 17. DLRM baseline：Fig. 9 说明论文不是拿弱基线比较

附录 Fig. 9 展示了一个 DLRM ranking baseline。

![Fig. 9: DLRM DIN+DCN baseline architecture](/images/blog/generative-recommenders-hstu/hstu-fig09-dlrm-din-dcn-baseline.webp)

*Source: Zhai et al., ICML 2024, Fig. 9, CC BY 4.0.*

这个 baseline 包含 sparse sequence embedding、pooled embedding、dense embedding、float features、sequence architecture、sparse architecture、embedding architecture、dense architecture，以及 DIN 和 DCN 风格交互。

这说明论文并不是用一个过时的浅层模型做对照。它对比的是工业推荐中真实存在的复杂 DLRM 系统。也正因为 baseline 很强，GR 的优势才不能只解释为“用了更深的网络”。

不过 Fig. 9 也提醒我们：GR 的落地不是替换一个函数。传统 DLRM 系统里大量特征、任务、召回源、排序策略、业务规则都需要迁移或重新表达。论文中的 `GR (new source)` 和 `GR (replace source)` 实验，其实就是不同迁移策略。

## 18. Subsequence selection：Table 13

Stochastic Length 依赖 subsequence selection。Table 13 比较了不同选择方法。

![Table 13: Stochastic Length 中 subsequence selection 方法对比](/images/blog/generative-recommenders-hstu/hstu-table13-subsequence-selection.webp)

*Source: Zhai et al., ICML 2024, Table 13, CC BY 4.0.*

表中比较了 greedy、weighted、random。论文报告 feature-weighted selection 在主要指标上更好。

这部分对工程落地很关键，因为“随机截断历史”常常被实现成一个数据管道细节，但它会影响模型学到的长期偏好。一个可控的 subsequence selection 策略至少需要记录：

1. 选择方法版本。
2. 最大长度和采样长度分布。
3. 是否偏向近期行为。
4. 是否按行为类型或特征重要性加权。
5. 训练和推理的历史窗口是否一致。

否则，模型质量变化时很难判断是模型结构带来的，还是历史采样策略改变带来的。

## 19. 论文-代码对照：官方仓库整体入口

官方仓库 `meta-recsys/generative-recommenders` 的 README 说明，它包含论文代码以及相关实验代码。README 重点暴露了三条线：

1. **Public Experiments**：ML-1M、ML-20M、Amazon Books 上复现实验。
2. **Synthetic Dataset / MovieLens-3B**：用 fractal expansion 生成更大 synthetic dataset。
3. **DLRM-v3 / efficiency experiments**：训练/推理 benchmark、Triton/CUDA kernel、HSTU attention。

公开实验的典型命令是：

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
  --master_port=12345
```

这说明官方代码更像“研究复现 + 工程组件展示”，不是 Meta 内部生产系统的完整开源版。

## 20. `main.py`：训练入口与 DDP spawn

`main.py` 很短，主要职责是读取 `gin_config_file`，再用 `torch.multiprocessing.spawn` 启动多 GPU 训练。

关键流程可以概括为：

```text
main.py
  -> parse gin_config_file
  -> world_size = torch.cuda.device_count()
  -> mp.spawn(mp_train_fn)
  -> train_fn(rank, world_size, master_port)
```

这里有两个实现含义。

第一，论文公开实验默认是 PyTorch + gin-config 的研究工程风格，而不是一个完整线上训练平台。

第二，所有核心选择都下沉到 `train_fn` 和 `.gin` 配置里：数据集、序列长度、模型类型、loss、负采样、TopK 方法、学习率、batch size 等。

## 21. `train.py`：公开实验训练管线

`generative_recommenders/research/trainer/train.py` 是公开实验主线。

它的关键配置项包括：

| 参数 | 含义 |
| --- | --- |
| `dataset_name` | `ml-1m`、`ml-20m`、Amazon Books 等数据集 |
| `max_sequence_length` | 用户历史最大长度 |
| `main_module` | `SASRec` 或 `HSTU` |
| `loss_module` | `SampledSoftmaxLoss` 或 BCE |
| `num_negatives` | sampled softmax 负样本数 |
| `top_k_method` | `MIPSBruteForceTopK` 等评估检索方法 |
| `gr_output_length` | GR 输出长度 |
| `item_embedding_dim` | item embedding 维度 |

训练流程大致是：

```text
get_reco_dataset(...)
  -> create_data_loader(...)
  -> LocalEmbeddingModule(...)
  -> get_similarity_function(...)
  -> LearnablePositionalEmbeddingInputFeaturesPreprocessor(...)
  -> get_sequential_encoder(main_module="HSTU")
  -> SampledSoftmaxLoss(...)
  -> DDP train/eval loop
  -> get_top_k_module(...)
```

这条管线对应的是公开 sequential recommendation 实验。它和论文的工业级 GR ranking/retrieval 系统不是同一个完整系统，但能展示 HSTU 在公开数据集上的训练方式。

## 22. `features.py`：`SequentialFeatures` 是代码中的最小数据契约

`features.py` 定义了 `SequentialFeatures`：

```python
class SequentialFeatures(NamedTuple):
    past_lengths: torch.Tensor
    past_ids: torch.Tensor
    past_embeddings: Optional[torch.Tensor]
    past_payloads: Dict[str, torch.Tensor]
```

这对应论文里的“把用户历史组织成序列”。在 MovieLens 公开实验里，`movielens_seq_features_from_row` 会把：

1. `history_lengths`
2. `historical_ids`
3. `historical_ratings`
4. `historical_timestamps`
5. `target_ids`
6. `target_ratings`
7. `target_timestamps`

转换成 HSTU/SASRec 可以消费的序列特征和 supervision target。

这里有一个重要区别：公开 MovieLens 数据只有 item、rating、timestamp 这种相对简单的结构；论文生产系统里的 $\Phi_i$ 更复杂，包含内容、动作、上下文、可能还有多源异构特征。因此代码中的 `SequentialFeatures` 是最小公开接口，不是完整工业 schema。

## 23. `hstu.py`：HSTU 的实现结构

`hstu.py` 是官方代码中最值得精读的文件。

它的主要模块如下：

| 代码对象 | 对应论文概念 |
| --- | --- |
| `RelativePositionalBias` | 相对位置 bias |
| `RelativeBucketedTimeAndPositionBasedBias` | 相对时间 + 位置 bucket bias |
| `_hstu_attention_maybe_from_cache` | 带 cache 的 HSTU attention 计算 |
| `SequentialTransductionUnitJagged` | 单层 HSTU/STU，jagged tensor 路径 |
| `HSTUJagged` | 多层 HSTU 堆叠 |
| `HSTU` | 对外 sequential encoder，接入 embedding、preprocessor、postprocessor、similarity |

代码中的 cache 状态类型是：

```python
HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

它保存的内容包括 value、padded query、padded key、output 等。这个 cache path 与 M-FALCON 的推理摊销思路一致：历史部分可以复用，候选 microbatch 部分增量计算。

`RelativeBucketedTimeAndPositionBasedBias` 里有一个细节：它 bucketize 的是相邻时间差或事件时间跨度，而不是只看位置。这与论文强调的非平稳推荐数据一致。

## 24. `hstu-ml1m.gin`：公开 HSTU-large 配置长什么样

`configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin` 是 README 推荐的 ML-1M HSTU-large 配置之一。关键值包括：

| 配置 | 值 |
| --- | --- |
| `train_fn.dataset_name` | `"ml-1m"` |
| `train_fn.max_sequence_length` | `200` |
| `train_fn.local_batch_size` | `128` |
| `train_fn.main_module` | `"HSTU"` |
| `train_fn.item_embedding_dim` | `50` |
| `hstu_encoder.num_blocks` | `8` |
| `hstu_encoder.num_heads` | `2` |
| `hstu_encoder.dqk` | `25` |
| `hstu_encoder.dv` | `25` |
| `train_fn.loss_module` | `"SampledSoftmaxLoss"` |
| `train_fn.num_negatives` | `128` |
| `train_fn.top_k_method` | `"MIPSBruteForceTopK"` |

这组配置说明公开实验并不是万亿参数规模，而是一个可在单卡或小规模 GPU 上运行的研究配置。它的价值是验证 HSTU 相对 SASRec 的建模优势，而不是复现生产 scaling law。

## 25. `SampledSoftmaxLoss` 与负采样

官方代码里的 `SampledSoftmaxLoss` 是公开实验的重要训练目标。它接收 output embeddings、supervision ids、supervision embeddings、supervision weights，并使用 sampled negatives 计算自回归损失。

简化理解：

```text
positive = target item embedding
negatives = sampled item embeddings
logits = similarity(output, [positive, negatives])
loss = -log_softmax(logits)[positive]
```

README 明确说明，HSTU/SASRec 行使用 sampled softmax；某些外部 BERT4Rec/GRU4Rec 结果使用 full negatives，因此比较时要注意训练目标和负样本设置差异。

这也和 YouTube DNN 的候选生成训练有相似之处：在超大 item vocabulary 下，完整 softmax 通常不可行，sampled softmax 或类似近似是必要工程选择。

## 26. TopK 和 MIPS：公开实验的检索评估

`indexing/utils.py` 中的 `get_top_k_module` 根据 `top_k_method` 返回 `MIPSBruteForceTopK` 或 `MoLBruteForceTopK`。

在公开实验里，这更多用于评估 HR/NDCG 等指标，而不是生产 ANN serving。配置文件中 `train_fn.top_k_method = "MIPSBruteForceTopK"`，说明公开数据集规模足以用 brute-force MIPS 做评估。

这和生产推荐不同。生产系统需要多级召回、ANN、cache、在线特征、候选过滤、业务规则和 ranker。论文中的 M-FALCON 主要针对 target-aware ranking 的候选评分成本，而不是替代所有召回检索基础设施。

## 27. 与 YouTube DNN、Clippy、冷启动闭环的关系

这篇论文可以和前几篇推荐系统文章放在一条线上理解。

YouTube DNN 2016 的核心是 **candidate generation -> ranking** 两阶段架构。候选生成用大规模分类/ANN 把百万级视频缩小到几百个候选，ranking 再用更丰富特征优化 watch time。

Generative Recommenders 没有否定两阶段架构，而是重新思考 ranking/retrieval 的建模范式。它让 retrieval 变成下一个 content token 预测，让 ranking 变成给定 target content 后的 action token 预测。

Clippy 论文讨论的是大规模多任务 ranker 的训练稳定性。GR/HSTU 论文讨论的是推荐系统如何吸收更大 compute。两者互补：当模型变大、序列变长、任务更多时，训练稳定性问题只会更重要。

冷启动闭环文章强调曝光日志、真实反馈、弱监督退场和 ANN 低权重上线。GR/HSTU 论文则是成熟系统之后的 scaling 路径。没有可靠曝光和反馈闭环，GR 这种大模型也只是对噪声做更昂贵的拟合。

可以把推荐系统成熟度分成四层：

| 阶段 | 核心问题 | 代表方法 |
| --- | --- | --- |
| 冷启动 | 没有可靠训练样本 | 内容先验、规则、探索、曝光日志 |
| 基础闭环 | 有曝光反馈但模型简单 | 双塔、ANN、GBDT/MLP ranker |
| 工业 ranker | 多任务、多源特征、线上 A/B | DLRM、DIN、DCN、MMoE、watch-time target |
| scaling 推荐 | 统一序列建模和大 compute | GR、HSTU、M-FALCON、streaming training |

GR/HSTU 更接近第四层。把它提前用于没有闭环的系统，通常不会得到论文里的收益。

## 28. 工程落地清单：从普通序列推荐到 GR/HSTU

如果一个团队想把这篇论文变成可落地项目，不应该第一步就追 1.5T 参数，而应该按风险递增推进。

第一步，整理日志契约。至少需要：

1. 用户行为序列：曝光、点击、停留、点赞、购买、隐藏等动作。
2. 内容 token：item/content/product 的稳定 ID 和 embedding。
3. 时间戳：用于 relative time bias 和采样。
4. 候选上下文：ranking 时每个候选来自哪个 source。
5. 训练快照：防止未来信息泄漏。

第二步，做 traditional sequential recommender baseline。用 SASRec 或 HSTU 在公开/内部离线数据上建立 HR/NDCG 或业务指标离线基线。先证明序列建模有效，不要直接切线上。

第三步，引入 action token。把用户行为类型显式放进序列，而不是只训练下一个 item。这一步决定 GR 是否真的比传统 item-sequence 模型表达力更强。

第四步，引入 target-aware ranking。让候选 $\Phi_{i+1}$ 进入模型条件，预测动作 $a_{i+1}$。这是质量提升和推理成本同时上升的关键点。

第五步，实现候选评分摊销。先做小规模 batching，再做 cache，再做 microbatch。M-FALCON 的思想可以作为设计目标，但工程实现必须结合本地 serving 框架。

第六步，做线上 shadow 和 A/B。GR/HSTU 这种模型不能只看离线 HR/NDCG。需要检查延迟、吞吐、曝光分布、长尾覆盖、负反馈、业务 guardrail 和用户长期指标。

## 29. 常见误读

**误读 1：GR 就是推荐系统版 GPT。**

不准确。GR 借用了生成建模和自回归序列思想，但推荐数据的 token、动作、候选、时间、业务目标和 serving 约束都不同。推荐的 target-aware ranking 不能直接等同语言模型 next-token prediction。

**误读 2：HSTU 可以无脑替换 Transformer。**

不准确。HSTU 是面向论文场景设计的 encoder，在公开实验和生产实验中表现好，但迁移到新业务仍要重新验证。尤其是数据规模、候选组织、行为类型和序列长度不同，收益可能变化。

**误读 3：1.5T 参数意味着所有推荐系统都该做万亿模型。**

不准确。1.5T 是论文生产系统规模，不是推荐系统起步配置。大多数团队更应该先建立日志闭环、强 baseline 和线上评估，再考虑扩容。

**误读 4：公开代码能复现论文所有结论。**

不准确。官方代码能复现公开数据集实验，并提供 HSTU、训练配置、部分效率相关实现。但生产数据、线上 A/B、1.5T 训练和 Meta 内部 serving 系统不可复现。

**误读 5：GR 解决了推荐系统所有 bias。**

不准确。GR 是建模范式，不自动解决 selection bias、position bias、曝光偏差、反馈延迟、冷启动、公平性、多目标冲突和业务安全问题。这些仍然需要样本构造、实验设计和监控体系。

## 30. 局限性与批判

第一，生产数据不可复现。论文最强的 scaling law 和 online A/B 结果来自内部系统。外部读者只能复现公开 MovieLens/Amazon 设置，无法独立验证 1.5T 生产结论。

第二，公开数据集和工业任务差异很大。MovieLens/Amazon Books 的 item、行为、上下文、候选池复杂度远低于真实内容平台。公开结果证明 HSTU 在标准 sequential recommendation 上有效，但不等价于工业 ranking 结论。

第三，成本门槛极高。GR 的目标不是轻量化小模型，而是让推荐模型吸收更大 compute。没有足够数据、训练平台、serving 优化和线上实验能力，直接追随该路线成本很高。

第四，模型范式不能替代产品治理。推荐系统的目标不只是点击或停留，还包括用户满意度、多样性、安全、创作者生态、商业目标和长期健康。GR 可以提供更强表示，但目标函数仍需要业务定义。

第五，论文对失败案例讨论有限。我们能看到 GR 的优势，但看不到足够多负例：哪些场景 HSTU 不如 DLRM？哪些行为类型不适合序列化？哪些业务目标不适合 action token modeling？这些仍是落地时需要探索的问题。

## 31. 推荐阅读路径

如果只想快速读论文，建议顺序如下：

1. Abstract：抓住 `1.5 trillion`、`12.4%`、`65.8%`、`5.3x-15.2x` 这些主张。
2. Fig. 2 和 Table 1：理解 DLRM 到 GR 的范式转换。
3. Section 3：读 HSTU、Stochastic Length、M-FALCON。
4. Table 4、Table 5：看公开实验和 HSTU ablation。
5. Table 6、Table 7、Fig. 7：看工业实验与 scaling law。
6. Appendix F：补 Stochastic Length 的细节。
7. Appendix H：补 M-FALCON 的算法和吞吐实验。
8. 官方代码 README、`main.py`、`train.py`、`features.py`、`hstu.py`、`.gin` 配置。

如果从工程实现角度读代码，建议顺序如下：

1. `README.md`：确认公开实验能跑什么，不能跑什么。
2. `configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin`：看具体实验配置。
3. `main.py`：理解入口和 DDP。
4. `train.py`：理解数据、模型、loss、eval 管线。
5. `features.py`：理解公开数据如何变成序列特征。
6. `hstu.py`：理解 HSTU、relative bias、jagged path、cache path。
7. `sampled_softmax.py` 和 `autoregressive_losses.py`：理解训练目标。
8. `ops/triton` 和 `ops/cpp/hstu_attention`：只在需要效率复现时深入。

## 32. 结论

这篇论文的长期价值不在于“推荐系统又多了一个 attention block”，而在于它把推荐系统放进了生成建模和 scaling law 的框架里。

传统 DLRM 的强项是异构特征工程和任务定制，弱项是很难像语言模型那样统一、稳定、可预测地吃掉更多 compute。Generative Recommenders 的核心主张是：把内容、动作和候选统一成序列，ranking 和 retrieval 都可以作为 sequential transduction 来建模。

HSTU 解决的是 encoder 结构和效率问题；Stochastic Length 解决的是长序列训练稀疏性问题；M-FALCON 解决的是 target-aware ranking 推理成本问题。三者合在一起，才构成论文声称的工业级 GR 路线。

对真实工程团队来说，这篇论文最值得带走的不是“马上上万亿参数推荐模型”，而是三个更实际的判断：

1. 推荐系统的长期竞争可能越来越依赖能否吸收更大数据和 compute。
2. 用户动作序列本身是核心建模对象，不只是 ranker label。
3. 大模型推荐不是单点模型升级，而是日志、序列构造、训练、候选评分、缓存、在线 A/B 和治理体系的整体重构。

这也是标题 “Actions Speak Louder than Words” 真正有意思的地方：在推荐系统里，用户的动作就是语言；如果能把这些动作组织成可扩展的生成建模问题，推荐系统可能进入一个新的 scaling 阶段。

## 参考资料

1. Jiaqi Zhai et al., [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152), arXiv:2402.17152.
2. Jiaqi Zhai et al., [PMLR ICML 2024 paper page](https://proceedings.mlr.press/v235/zhai24a.html).
3. Meta RecSys, [generative-recommenders official GitHub repository](https://github.com/meta-recsys/generative-recommenders).
4. Wang-Cheng Kang and Julian McAuley, [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781), ICDM 2018.
5. Fei Sun et al., [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690), CIKM 2019.
6. Alexandros Karatzoglou et al., [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939), ICLR 2016 workshop.
7. Maxim Naumov et al., [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091), 2019.
8. Paul Covington, Jay Adams, Emre Sargin, [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/), RecSys 2016.
9. Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
10. Xinyang Yi et al., [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039), 2023.
11. Klenitskiy and Vasilev, [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602), 2023.

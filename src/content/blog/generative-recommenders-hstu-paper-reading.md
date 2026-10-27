---
title: "Generative Recommenders 论文精读：HSTU、M-FALCON 与万亿参数推荐序列建模"
description: "精读 Meta ICML 2024 Generative Recommenders 如何将推荐系统重构为序列生成问题，并通过 HSTU、Stochastic Length、M-FALCON 和 scaling law 支撑工业级部署"
pubDate: "2026-06-29T11:57:06+08:00"
updatedDate: "2026-06-29T11:57:06+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "Recommendation System"
  - "Generative Recommenders"
  - "Sequential Recommendation"
  - "HSTU"
  - "Meta"
draft: false
---

Jiaqi Zhai、Lucy Liao、Xing Liu、Yueming Wang、Rui Li、Xuan Cao、Leon Gao、Zhaojie Gong、Fangda Gu、Michael He、Yinghai Lu 和 Yu Shi 在 ICML 2024 发表的 **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations**，直面了工业推荐领域长期存在的一个核心疑问：

> 推荐系统是否也存在类似语言模型和视觉模型那样的算力扩展定律（compute scaling law）？如果存在，为什么传统的深度学习推荐模型（DLRM）范式未能显著兑现这一红利？

论文给出的系统性回答是 **Generative Recommenders（生成式推荐系统，简称 GRs）**。GRs 将异构特征彻底重构为**序列转导**（sequential transduction）任务：把内容表征 token、用户动作 token 与目标候选 token 交错编排在同一条时间序列中，从而将召回（retrieval）与排序（ranking）统一纳入目标感知（target-aware）的自回归生成框架。为了支撑这一范式在十亿级日活生产环境中的训练与在线服务，论文配套提出了三项核心工程组件：

1. **HSTU（Hierarchical Sequential Transduction Unit）**：专为高基数、非平稳、流式推荐数据设计的层级序列转导编码器。
2. **Stochastic Length（随机序列长度采样）**：在训练阶段按幂律分布动态稀疏化长历史序列，以可控计算成本保持模型对超长历史的建模能力。
3. **M-FALCON（Microbatched-Fast Attention Leveraging Cacheable OperatioNs）**：在目标感知的排序推理阶段，利用微批次（microbatching）与编码器键值缓存摊销海量候选的打分开销。

论文汇报的关键指标非常亮眼：HSTU 在合成与公开基准上最高取得 `65.8%` 的 NDCG 提升，在 8192 序列长度下比基于 FlashAttention-2 的 Transformer 快 `5.3x`～`15.2x`；基于 HSTU 的 GRs 参数规模最高扩展至 `1.5 trillion`（1.5 万亿），并在 Meta 线上 A/B 实验的排序主指标上取得 `+12.4%` 的业务增益。

本文围绕范式重构、HSTU 与 M-FALCON 算法细节、Scaling Law 实验证据以及官方开源仓库 `meta-recsys/generative-recommenders` 的核心代码实现展开精读。文中图表截取自 arXiv/PMLR 论文 PDF（CC BY 4.0 许可）；关于工业生产数据集、1.5T 模型训练与线上 A/B 收益的分析均以论文报告为准，代码走读部分基于官方公开的复现仓库做静态拆解。

## 1. 论文概览与工业 DLRM 的扩展瓶颈

| 项目 | 内容 |
| --- | --- |
| 题名 | Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations |
| 作者 | Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, Yinghai Lu, Yu Shi |
| 会议 | ICML 2024, PMLR 235 |
| arXiv | `2402.17152` |
| 页数 | 26 页 |
| 官方代码 | `meta-recsys/generative-recommenders` |
| 主要关键词 | Generative Recommenders, HSTU, Stochastic Length, M-FALCON, scaling law |

论文的整体脉络可以浓缩为一句话：**将工业推荐从“手工异构特征工程 + 多塔/多任务 DLRM”重构为“内容与动作交错的自回归序列生成问题”，并依靠 HSTU、Stochastic Length 与 M-FALCON 协同突破超长序列训练与目标感知推理的算力瓶颈**。

这一重构触及三个层面的转变：在架构层面，它改变了传统 DLRM 依赖人工堆叠特征交叉模块与业务子塔的扩展方式；在序列建模层面，它超越了 SASRec、BERT4Rec 等仅对历史物品 ID 建模的传统序列推荐，将内容特征 $\Phi_i$、用户反馈动作 $a_i$ 与待评估候选 $\Phi_{i+1}$ 显式融合；在推理系统层面，它通过 M-FALCON 将单个用户请求下 $m$ 个候选对长度 $n$ 历史的重复注意力计算转化为一次缓存共享与微批次增量计算。

### 1.1 训练算力趋势对比（Fig. 1）

Fig. 1 将工业推荐模型（DLRM 与 GR）与主流语言及视觉模型放在同一张训练算力（PFlops/s-days）时间轴上进行对照：

![Fig. 1: DLRM 与 GR 训练 compute 趋势](/images/blog/generative-recommenders-hstu/hstu-fig01-training-compute.webp)

*Source: Zhai et al., ICML 2024, Fig. 1, CC BY 4.0.*

图中红色数据点代表 AlexNet、ResNet、BERT、AlphaZero、GPT-3 与 LLaMA-2 等里程碑模型，蓝色星号则对应工业 DLRM 及论文提出的 GR-23、GR-24。尽管大型推荐平台每天沉淀的用户交互行为远超自然语言语料规模，但传统 DLRM 的训练算力曲线长期处于平缓状态，未能像大语言模型那样通过跨数量级扩充算力换取可预测的指标跃升。直到切换为统一序列转导架构后，GR-23 与 GR-24 的有效训练算力才逼近 GPT-3 与 LLaMA-2 的量级，开启了推荐系统的基础模型化扩展路径。

### 1.2 传统工业 DLRM 难以持续扩展的四类症结

为什么在 DLRM 框架下单纯堆叠参数和算力容易遭遇收益饱和？论文将其归结为推荐场景特有的四项结构性约束：

1. **异构特征空间缺乏统一序列结构**：工业排序特征涵盖用户画像、创作者 ID、物品 ID、上下文统计计数、滑动窗口交叉比率、历史行为序列以及多模态内容表征。DLRM 依靠定制化的特征提取器、显式交叉网络（如 DCN）与多任务塔将这些异构模块拼装在一起，导致计算图碎片化，难以通过统一增加深度或宽度来平滑扩展容量。
2. **十亿级动态高基数词表（High-Cardinality Dynamic Vocabulary）**：语言模型的词表通常固定在十万量级，而推荐系统的物品、广告与创作者规模高达十亿级且时刻处于新增与过期下架的动态变化中，给底层 Embedding 维护、负采样与冷启动泛化带来巨大负担。
3. **非平稳流式训练分布**：工业推荐面对的是随时间持续演化的用户行为流，候选池分布、季节性热点与业务目标时刻漂移，无法像静态离线文本语料那样做多轮全量随机洗牌（full-shuffle）。
4. **目标感知（Target-Aware）的严苛推理预算**：排序阶段必须精细刻画当前候选物品与用户历史偏好的交叉反应，候选特征必须尽早进入深层交互路径。如果在长序列自注意力模型中直接为每个候选重跑一遍完整历史编码，线上延迟与 GPU 算力开销将彻底失控。

上述四点共同表明，简单把标准 Transformer 模块塞进 DLRM 并不能解决问题；要释放序列模型的扩展潜力，必须从输入表征、注意力算子到在线推理调度进行全栈重构。

## 2. 生成式推荐（GR）范式：从特征拼装到序列转导

### 2.1 DLRM 与 GR 的特征组织对比（Fig. 2）

Fig. 2 直观展示了从传统 DLRM 到生成式推荐系统（GR）的特征重组过程：

![Fig. 2: DLRM 与 GR 的特征组织和训练流程](/images/blog/generative-recommenders-hstu/hstu-fig02-dlrm-vs-gr-features.webp)

*Source: Zhai et al., ICML 2024, Fig. 2, CC BY 4.0.*

在左侧的传统 DLRM 中，来自不同特征源的类别特征（`E, F, G, H`）与稠密数值特征分别经过独立的提取、池化与交叉子网，最后汇总至顶部 MLP。而在右侧的 GR 中，所有的用户历史交互、多源辅助时间序列、稀疏与稠密属性被统一序列化并压缩为交错排列的**内容表征 token $\Phi_i$** 与**用户动作 token $a_i$**。两套范式的核心差异对照如下：

| 维度 | 传统工业 DLRM | Generative Recommender (GR) |
| --- | --- | --- |
| 输入特征组织 | 异构特征簇分模块提取与拼接 | 内容 token $\Phi_i$ 与动作 token $a_i$ 交错成单一序列 |
| 候选物品接入方式 | 在排序塔中与池化后的用户向量做交叉 | 作为目标感知条件 $\Phi_{i+1}$ 直接追加至序列末端 |
| 监督与训练方式 | 针对单次曝光做点式（pointwise）多任务监督 | 沿用户时间轴做因果自回归（causal autoregressive）序列转导 |
| 容量扩展主轴 | 堆叠手工特征、扩充子塔与显式交叉层 | 扩展序列上下文长度 $n$、模型层数/维度与训练算力 |
| 核心工程瓶颈 | 特征管道维护成本高、算力利用率低 | 超长序列注意力开销与多候选并发打分延迟 |

### 2.2 召回与排序的统一序列转导形式化（Table 1）

Table 1 给出了将召回（Retrieval）与排序（Ranking）统一抽象为序列转导任务的数学形式：

![Table 1: Ranking 与 retrieval 作为 sequential transduction tasks](/images/blog/generative-recommenders-hstu/hstu-table01-sequential-transduction-tasks.webp)

*Source: Zhai et al., ICML 2024, Table 1, CC BY 4.0.*

设用户按时间顺序接触过的内容序列为 $\Phi_0, \Phi_1, \ldots, \Phi_i$，对应的反馈动作序列为 $a_0, a_1, \ldots, a_i$。若不区分具体待评候选，纯粹的动作预测可写为 $p(a_{i+1}\mid \Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_i,a_i)$；但真实排序任务需要评估“当系统向用户展示特定候选内容 $\Phi_{i+1}$ 时，用户产生多目标动作 $a_{i+1}$ 的概率分布”。因此，GR 将**目标感知排序**（Target-Aware Ranking）形式化为：

$$
p(a_{i+1}\mid \Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_i,a_i,\Phi_{i+1})
$$

通过将候选内容 $\Phi_{i+1}$ 显式置于条件序列末端，模型能够在深层自注意力网络中直接完成候选内容与全部历史交互的细粒度交叉。相应地，**召回任务**（Retrieval）则对应于在观测到动作 $a_i$ 之后，自回归地生成或检索用户下一个将要正向交互的内容表征：

$$
p(\Phi_{i+1}\mid \Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_i,a_i)
$$

这一划分也紧扣了论文题名“Actions Speak Louder than Words”的核心洞察：在推荐系统中，点击、完播、点赞、购买乃至跳过、隐藏等动作 $a_i$ 本身承载了最关键的用户偏好演进语义，必须作为一等序列 token 参与上下文建模。

### 2.3 与传统序列推荐模型的边界划分（Table 10 & Table 11）

为了厘清 GR 与早期学术界序列推荐工作（如 GRU4Rec、SASRec、BERT4Rec）的差异，论文在附录 Table 10 与 Table 11 中做了系统梳理：

![Table 10: sequential recommender 与 GR 相关工作对比](/images/blog/generative-recommenders-hstu/hstu-table10-related-work-ranking.webp)

*Source: Zhai et al., ICML 2024, Table 10, CC BY 4.0.*

![Table 11: 对内容与动作序列的生成建模](/images/blog/generative-recommenders-hstu/hstu-table11-generative-modeling.webp)

*Source: Zhai et al., ICML 2024, Table 11, CC BY 4.0.*

综合这两张对照表可以看出，传统序列推荐通常仅在正向交互的物品 ID 序列上预测下一个物品，既不区分曝光未点击等负向动作，也缺乏在序列末端接入任意候选 $\Phi_{i+1}$ 进行多任务动作预估的机制。而 GR 通过交错编排 $(\Phi_0, a_0, \Phi_1, a_1, \ldots)$，在同一套因果转导骨架上同时支持了面向动作生成的精排打分 $p(a_i \mid \Phi_0,a_0,\ldots,\Phi_i)$ 与面向内容生成的候选召回 $p(\Phi_i \mid \Phi_0,a_0,\ldots,a_{i-1})$。

## 3. HSTU 架构设计与结构消融实验

在明确了序列转导的任务形式后，核心问题随之转向：什么样的序列编码器能够高效处理推荐场景下的非平稳长序列与动态词表？论文为此提出了 **HSTU（Hierarchical Sequential Transduction Unit）**。

### 3.1 逐点变换注意力、相对时空偏置与门控机制（Fig. 3）

Fig. 3 将传统 DLRM 的异构子模块与 GR/HSTU 的统一层级结构进行了并置对比：

![Fig. 3: DLRM 与 GR/HSTU 的关键组件](/images/blog/generative-recommenders-hstu/hstu-fig03-model-components.webp)

*Source: Zhai et al., ICML 2024, Fig. 3, CC BY 4.0.*

单个 HSTU 层由三个紧密衔接的数学步骤构成。首先，输入序列表示 $X$ 经过一次线性投影 $f_1$ 与非线性激活 $\phi_1$（如 SiLU），直接切分出门控权重 $U(X)$、值 $V(X)$、查询 $Q(X)$ 与键 $K(X)$ 四个张量：

$$
U(X),V(X),Q(X),K(X)=\mathrm{Split}(\phi_1(f_1(X)))
$$

其次，在计算注意力权重矩阵 $A(X)$ 时，HSTU 融入了同时刻画序列相对位置（position）与事件相对时间差（time delta）的联合偏置项 $r_{ab}^{p,t}$，并采用逐点激活函数 $\phi_2$（pointwise transformation，如 SiLU）替代了标准 Transformer 中的行归一化 Softmax：

$$
A(X)=\phi_2\left(Q(X)K(X)^T+r_{ab}^{p,t}\right)
$$

最后，经过注意力聚合的表示通过归一化层（Norm），与门控分支 $U(X)$ 做逐元素哈达玛积（$\odot$），再经线性变换 $f_2$ 输出：

$$
Y(X)=f_2\left(\mathrm{Norm}(A(X)V(X))\odot U(X)\right)
$$

这三步设计针对推荐系统的物理特性做了深度定制：
- **弃用 Softmax 改用逐点激活注意力（Pointwise Aggregated Attention）**：标准 Softmax 强制令所有历史位置的注意力权重之和为 $1$，无法自然表达用户对某个候选类目的交互强度与历史活跃度（例如重度活跃用户与冷启动用户的总匹配度本不相同）。移除 Softmax 归一化不仅保留了交互频次与强度的幅值信息，还规避了长序列下昂贵的指数行归约开销。
- **相对时间与位置融合偏置 $r_{ab}^{p,t}$**：自然语言仅需关注词元间的相对位置，而推荐行为高度依赖发生时刻的物理间隔（昨天点击某品类与半年前点击同一品类的意图权重截然不同），显式引入分桶时间差偏置能有效捕捉非平稳兴趣衰减。
- **显式门控与消除独立 FFN**：通过 $\mathrm{Norm}(A(X)V(X))\odot U(X)$ 的子层门控设计，HSTU 在单一算子内同时完成了序列上下文聚合与高阶特征交叉（发挥类似 DLRM 中 DCN/门控网络的作用），省去了标准 Transformer 中参数与算力占比极高的独立前馈网络（FFN）。
- **原生适配变长 Jagged Tensor 与增量缓存**：真实用户的历史行为长度跨度极大，HSTU 底层围绕 FBGEMM 的不规则张量（Jagged Tensor）设计，完全消除了无意义的 Padding 计算，并暴露了供 M-FALCON 复用的中间状态缓存接口。

### 3.2 流式合成数据与公开基准上的结构消融（Table 2 & Table 5）

为了验证上述结构假设，论文首先在单遍流式合成数据集（synthetic one-pass streaming setting）上进行了受控对比（Table 2），随后在公开序列推荐基准上展开了逐模块消融（Table 5）：

![Table 2: synthetic one-pass streaming setting 结果](/images/blog/generative-recommenders-hstu/hstu-table02-synthetic-streaming.webp)

*Source: Zhai et al., ICML 2024, Table 2, CC BY 4.0.*

![Table 5: HSTU、HSTU ablation 与 Transformers 对比](/images/blog/generative-recommenders-hstu/hstu-table05-hstu-ablation.webp)

*Source: Zhai et al., ICML 2024, Table 5, CC BY 4.0.*

联合解读 Table 2 与 Table 5 的四档配置——标准 `Transformer` 基线、移除时空偏置并退回 Softmax 的 `HSTU (-rabp,t, Softmax)`、保留逐点激活但移除时空偏置的 `HSTU (-rabp,t)`，以及完整形态的 `HSTU`——可以清晰剥离出各组件的贡献：在合成流式设置与真实数据集上，将逐点激活强行换回 Softmax 会导致指标大幅回落，证实了非归一化逐点注意力对捕获推荐偏好强度的关键作用；而在保留逐点注意力的基础上叠加相对时间与位置偏置 $r_{ab}^{p,t}$，则进一步稳定提升了各数据集上的 HR 与 NDCG 表现。

## 4. Stochastic Length：面向超长行为序列的稀疏训练策略

工业推荐系统中，活跃用户在数月内积累的曝光、点击、完播与交互序列动辄长达数万步。如果在每个训练步都对全长历史执行 $O(n^2)$ 复杂度（或次二次复杂度）的注意力计算，训练吞吐将难以为继；反之，若在训练时硬性截断为短序列，模型在推理阶段面对长历史上下文时又会出现严重的泛化退化。为此，论文提出了 **Stochastic Length（SL，随机序列长度采样）**。

### 4.1 随机长度采样的稀疏度控制与离线收益（Table 3, Fig. 4, Table 14-15）

Stochastic Length 的核心思路是：考虑到用户的历史行为序列具有显著的时间冗余与重复模式，训练时以幂律分布随机采样保留子序列长度（由超参数 $\alpha$ 控制采样偏斜度），让模型在绝大部分训练样本上仅处理压缩后的稀疏子序列，仅以一定概率接触完整长序列。Table 3 统计了不同最大序列长度与参数 $\alpha$ 组合下的序列注意力稀疏度（Sparsity）：

![Table 3: Stochastic Length 对序列稀疏性的影响](/images/blog/generative-recommenders-hstu/hstu-table03-sl-sparsity-main.webp)

*Source: Zhai et al., ICML 2024, Table 3, CC BY 4.0.*

当 $\alpha=2.0$ 时近似于不施加额外的长度压缩，而调小 $\alpha$（如 $1.6$ 或 $1.2$）以及拉长最大历史窗口都会大幅提升计算的稀疏比例。Fig. 4 进一步展示了在工业数据上不同稀疏度配置对归一化熵（NE difference，柱状图）与计算稀疏度（虚线）的权衡影响：

![Fig. 4: Stochastic Length 对指标的影响](/images/blog/generative-recommenders-hstu/hstu-fig04-stochastic-length-metrics.webp)

*Source: Zhai et al., ICML 2024, Fig. 4, CC BY 4.0.*

可以看到，适度的 Stochastic Length 不仅大幅削减了训练算力消耗，甚至因为起到了序列级的正则化与去噪作用而略微改善了离线 NE 指标；只有当 $\alpha$ 过小导致历史信息被过度丢弃时，训练质量才会出现实质性受损。附录 Table 14 与 Table 15 进一步补充了 60 天与 90 天用户历史窗口下的稀疏度演进数据，验证了该策略在跨月度长周期序列上的稳定性：

![Table 14: 60 天用户历史下 Stochastic Length 对稀疏性的影响](/images/blog/generative-recommenders-hstu/hstu-table14-sl-sparsity-60d.webp)

*Source: Zhai et al., ICML 2024, Table 14, CC BY 4.0.*

![Table 15: 90 天用户历史下 Stochastic Length 对稀疏性的影响](/images/blog/generative-recommenders-hstu/hstu-table15-sl-sparsity-90d.webp)

*Source: Zhai et al., ICML 2024, Table 15, CC BY 4.0.*

### 4.2 与常规长度外推方法的对比（Table 16 & Fig. 10）

附录 Table 16 与 Fig. 10 将 Stochastic Length 与大语言模型中常见的长度外推（length extrapolation）方案——包括直接零样本外推（zero-shot）、长序列微调（fine-tune）以及 RoPE/NTK 风格位置插值——进行了横向评测：

![Table 16: Stochastic Length 与 sequence length extrapolation 方法对比](/images/blog/generative-recommenders-hstu/hstu-table16-sl-vs-length-extrapolation.webp)

*Source: Zhai et al., ICML 2024, Table 16, CC BY 4.0.*

![Fig. 10: Stochastic Length 对 ranking metrics 的影响](/images/blog/generative-recommenders-hstu/hstu-fig10-sl-ranking-metrics.webp)

*Source: Zhai et al., ICML 2024, Fig. 10, CC BY 4.0.*

实验表明，推荐序列的长度扩展难题不仅仅在于位置编码能否泛化到更大索引，更在于模型必须在训练期亲历长跨度、高稀疏的历史交互分布。Stochastic Length 通过在数据构造层直接注入多尺度长度分布，在训练速度与长上下文排序指标之间取得了显著优于纯位置编码外推的折中。

## 5. M-FALCON 推理算法与算子级效率分析

在排序服务（Ranking Serving）阶段，生成式推荐面临着严峻的推理算力挑战：假设用户历史序列长度为 $n$，单次请求需要对召回层送来的 $m$ 个候选物品 $\Phi'_0, \ldots, \Phi'_{m-1}$ 分别预估动作概率。若按朴素方式对每个候选单独拼接历史并执行前向传播，计算复杂度将高达 $O(m \cdot n^2)$，线上延迟完全无法接受。论文为此设计了 **M-FALCON（Microbatched-Fast Attention Leveraging Cacheable OperatioNs）**。

### 5.1 微批次与编码器缓存摊销机制（Fig. 11 & Algorithm 1）

Fig. 11 与 Algorithm 1 完整展示了 M-FALCON 的训练与推理计算图改造：

![Fig. 11: M-FALCON 训练与推理示意](/images/blog/generative-recommenders-hstu/hstu-fig11-mfalcon-illustration.webp)

*Source: Zhai et al., ICML 2024, Fig. 11, CC BY 4.0.*

![Algorithm 1: M-FALCON pseudocode](/images/blog/generative-recommenders-hstu/hstu-algorithm01-mfalcon.webp)

*Source: Zhai et al., ICML 2024, Algorithm 1, CC BY 4.0.*

M-FALCON 通过三层递进机制将 $m$ 个候选的目标感知打分成本压降至接近单次序列前向传播的量级：

1. **带掩码隔离的候选合批推理（Batched Target-Aware Inference）**：将多个待评候选 $\Phi'_0, \ldots, \Phi'_{b_m-1}$ 直接拼接在同一用户历史序列之后，并修改因果注意力掩码（attention mask），使得每个候选 token 都能完整关注到前序的 $n$ 个用户历史 token，但候选与候选之间相互不可见，从而在单次矩阵乘法中并行完成多候选交互，且严格杜绝候选间的信息泄漏。
2. **微批次切分（Microbatching）**：当单次请求的候选总数 $m$ 较大（例如数千个）时，直接拼成超长尾部会导致注意力矩阵膨胀并撑爆显存。M-FALCON 将 $m$ 个候选切分为大小为 $b_m$ 的 $\lceil m / b_m \rceil$ 个微批次依次处理，精准控制单步显存峰值与尾延迟。
3. **跨微批次的编码器级缓存（Encoder-Level Caching）**：由于同一个请求的各个候选微批次共享完全相同的用户历史前缀，M-FALCON 在首个微批次前向计算时，将每一层 HSTU 中用户历史部分的中间表征（$K(X)$、$V(X)$ 及层输出状态）持久化到缓存中；后续微批次仅需计算新增的 $b_m$ 个候选 token 对历史缓存的交叉注意力。

用伪代码直观对比，M-FALCON 将朴素的逐候选重算循环：

```text
for candidate in candidates:
  encode(history + candidate)
```

重构为一次历史编码加多轮轻量微批次增量打分：

```text
cache = encode_shared_history(history)
for microbatch in chunk(candidates, size=b_m):
  score(history_cache, microbatch)
```

### 5.2 候选评分吞吐扩展性（Fig. 6, Fig. 12, Fig. 13）

得益于上述缓存与微批次摊销，尽管单个 GR 模型的参数量与单序列 FLOPs 远高于传统 DLRM，其在多候选打分场景下的实际在线吞吐（QPS）依然极具竞争力。Fig. 6、Fig. 12 与 Fig. 13 分别从不同候选规模、端到端系统吞吐以及微批次大小扩展性三个维度验证了 M-FALCON 的线上性能：

![Fig. 6: 不同候选评分设置下的 inference throughput](/images/blog/generative-recommenders-hstu/hstu-fig06-inference-throughput-main.webp)

*Source: Zhai et al., ICML 2024, Fig. 6, CC BY 4.0.*

![Fig. 12: GR w/ M-FALCON 与 DLRM 的端到端吞吐](/images/blog/generative-recommenders-hstu/hstu-fig12-end-to-end-throughput.webp)

*Source: Zhai et al., ICML 2024, Fig. 12, CC BY 4.0.*

![Fig. 13: M-FALCON microbatch throughput scaling](/images/blog/generative-recommenders-hstu/hstu-fig13-mfalcon-throughput-scaling.webp)

*Source: Zhai et al., ICML 2024, Fig. 13, CC BY 4.0.*

这三组曲线揭示了一个重要的系统规律：在目标感知排序中，单纯对比单次前向的理论 FLOPs 会严重高估序列模型的部署成本。由于 DLRM 的许多交叉模块需要随候选数 $m$ 线性重复执行，而 GR 借助 M-FALCON 将重头的用户历史计算锁定在 $O(1)$ 次缓存构建上，因此在特定候选数量与微批次配置区间内，GR 甚至能跑出高于复杂 DLRM 的端到端吞吐。当然，这一高吞吐高度依赖底层 GPU 硬件特性、算子融合质量以及线上候选批大小分布。

### 5.3 HSTU 与 FlashAttention-2 的编码器效率对比（Fig. 5）

除了推理侧的 M-FALCON，HSTU 算子本身的执行效率也是支撑长序列训练与服务的基础。Fig. 5 对比了 HSTU 与基于 FlashAttention-2 的标准 Transformer 在不同序列长度与模型宽度下的训练与推理耗时：

![Fig. 5: HSTU vs FlashAttention2 Transformer 的 encoder-level efficiency](/images/blog/generative-recommenders-hstu/hstu-fig05-encoder-efficiency.webp)

*Source: Zhai et al., ICML 2024, Fig. 5, CC BY 4.0.*

在长达 8192 的序列上，HSTU 实现了相对 FlashAttention-2 Transformer **`5.3x` 至 `15.2x` 的速度提升**。这一加速来源于三方面协同：其一，HSTU 省去了独立的 FFN 子层，大幅削减了每层的矩阵乘总计算量；其二，逐点激活注意力免去了 Softmax 中的逐行最大值归约与指数求和同步；其三，底层结合 Jagged Tensor 与融合相对偏置的定制算子（开源仓库中的 `ops/triton` 与 `ops/cpp/hstu_attention`）显著降低了显存读写带宽开销。

## 6. 公开基准与工业级 Scaling Law 实验分析

### 6.1 公开数据集评测结果（Table 4 & Table 12）

在传统的公开序列推荐基准（MovieLens-1M、MovieLens-20M 与 Amazon Books）上，论文采用多轮全量洗牌（full-shuffle）设置对比了 SASRec 与 HSTU（Table 4），并在附录 Table 12 中补充了与 GRU4Rec、BERT4Rec 的对照：

![Table 4: 公开数据集多轮 full-shuffle 设置结果](/images/blog/generative-recommenders-hstu/hstu-table04-public-datasets.webp)

*Source: Zhai et al., ICML 2024, Table 4, CC BY 4.0.*

![Table 12: traditional sequential recommender 设置下补充比较](/images/blog/generative-recommenders-hstu/hstu-table12-traditional-sequential-results.webp)

*Source: Zhai et al., ICML 2024, Table 12, CC BY 4.0.*

结合官方仓库 README 公布的复现基准，HSTU-large 相比同管线下的 SASRec 取得了全面领先：

| Dataset | HSTU-large HR@10 | HSTU-large NDCG@10 | 相对 SASRec HR@10 提升 |
| --- | ---: | ---: | ---: |
| ML-1M | `0.3294` | `0.1893` | `+15.5%` |
| ML-20M | `0.3556` | `0.2098` | `+23.1%` |
| Amazon Books | `0.0478` | `0.0262` | `+56.7%` |

需要注意一处评测细节：官方 README 注明 Table 12 中 GRU4Rec 与 BERT4Rec 的数字直接引用自过往文献，且在训练时使用了全量负样本（full negatives）；而 SASRec 与 HSTU 则在同一套代码管线下使用采样负样本（sampled negatives，如 128 或 512 个负例）训练。因此，公开实验最核心的实证价值在于**严格控制变量下 HSTU 对 SASRec 的一致性超越**，证明了 HSTU 注意力与门控结构在标准物品序列预测上的建模优势。

### 6.2 工业召回与排序的离线与线上 A/B 表现（Table 6 & Table 7）

论文最具分量的实验来自大规模工业流式生产环境下的召回（Table 6）与排序（Table 7）离线/在线评测：

![Table 6: retrieval model offline/online 对比](/images/blog/generative-recommenders-hstu/hstu-table06-retrieval-offline-online.webp)

*Source: Zhai et al., ICML 2024, Table 6, CC BY 4.0.*

![Table 7: ranking model offline/online 对比](/images/blog/generative-recommenders-hstu/hstu-table07-ranking-offline-online.webp)

*Source: Zhai et al., ICML 2024, Table 7, CC BY 4.0.*

在这两组工业对比中，论文系统展示了极具工程参考价值的三种渐进接入形态：
- **`GR (interactions only)`**：仅输入纯用户交互历史序列，不依赖传统海量的手工统计与交叉特征，验证纯序列转导的基线能力；
- **`GR (new source)`**：将 GR 作为一路新增的特征源或打分源接入现有系统，与传统 DLRM 协同工作；
- **`GR (replace source)`**：直接用 GR 替换原有的核心 DLRM 模型路径。

在召回与排序两条链路上，GR 均显著改善了离线归一化熵（NE）与 Hit Rate，并在生产线上 A/B 测试中兑现了高达 **`+12.4%`** 的核心业务指标提升。

### 6.3 万亿参数工业级 Scaling Law 与强基线对照（Fig. 7, Fig. 9, Table 13）

Fig. 7 呈现了全文最核心的工业级发现——生成式推荐系统的算力扩展定律（Scaling Law）：

![Fig. 7: DLRM 与 GR 在大规模工业设置下的 scaling behavior](/images/blog/generative-recommenders-hstu/hstu-fig07-industrial-scaling.webp)

*Source: Zhai et al., ICML 2024, Fig. 7, CC BY 4.0.*

当横轴的训练算力跨数量级增长（综合扩增序列长度至 8192、加深网络层数、加宽隐藏维度并扩展至 1.5 万亿参数）时，传统工业 DLRM 的质量改善迅速趋于平缓饱和，而 GR 的评估指标则呈现出清晰的幂律（power-law）持续优化趋势。为了证明这里的 DLRM 对照组并非人为简化的弱基线，论文在附录 Fig. 9 中给出了生产级 DLRM 排序基线的完整拓扑：

![Fig. 9: DLRM DIN+DCN baseline architecture](/images/blog/generative-recommenders-hstu/hstu-fig09-dlrm-din-dcn-baseline.webp)

*Source: Zhai et al., ICML 2024, Fig. 9, CC BY 4.0.*

可以看到，该基线已经集成了稀疏序列 Embedding、池化表征、稠密浮点特征、DIN 风格的目标注意力序列模块以及 DCN 显式高阶交叉网络，代表了工业界历经多年打磨的成熟架构。此外，针对 Stochastic Length 在工业落地中的子序列筛选细节，附录 Table 13 进一步对比了贪心选择（greedy）、特征加权采样（feature-weighted）与纯随机采样（random）的效果：

![Table 13: Stochastic Length 中 subsequence selection 方法对比](/images/blog/generative-recommenders-hstu/hstu-table13-subsequence-selection.webp)

*Source: Zhai et al., ICML 2024, Table 13, CC BY 4.0.*

其中基于特征重要性加权的子序列采样（weighted）在保持高稀疏度的同时取得了最佳指标，提示工程团队在落地历史序列截断与采样时，应结合交互行为类型与时间权重设计可控的采样策略。

## 7. 官方代码走读：meta-recsys/generative-recommenders 核心链路

官方开源仓库 `meta-recsys/generative-recommenders` 提供了公开数据集复现（ML-1M、ML-20M、Amazon Books）、基于分形扩展（fractal expansion）的合成数据集 MovieLens-3B，以及 HSTU 的 PyTorch / Triton / C++ CUDA 算子实现。以下沿着训练启动、特征封装、HSTU 编码器到损失评估的核心调用链进行拆解。

### 7.1 训练主链路与 Gin 配置体系（main.py / train.py / .gin）

公开实验采用 PyTorch DDP 配合 Google `gin-config` 进行超参数注入。复现 ML-1M 上 HSTU-large 的标准命令如下：

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
  --master_port=12345
```

顶层入口 `main.py` 非常精简，负责解析 `.gin` 配置文件并通过 `torch.multiprocessing.spawn` 拉起分布式训练进程，直接进入 `generative_recommenders/research/trainer/train.py` 中的 `train_fn` 主循环：

```text
main.py (parse gin_config_file & mp.spawn)
  -> train.py: train_fn(rank, world_size, master_port)
       -> get_reco_dataset(...) & create_data_loader(...)
       -> LocalEmbeddingModule(...)
       -> LearnablePositionalEmbeddingInputFeaturesPreprocessor(...)
       -> get_sequential_encoder(main_module="HSTU")
       -> SampledSoftmaxLoss(...)
       -> DDP train / eval loop with get_top_k_module(...)
```

对照 `configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin`，可以清晰看到公开基准上 HSTU-large 的具体超参数契约：

| Gin 配置项 | 设定值 | 在训练链路中的职责 |
| --- | --- | --- |
| `train_fn.dataset_name` | `"ml-1m"` | 指定加载预处理后的 MovieLens-1M 序列数据 |
| `train_fn.max_sequence_length` | `200` | 用户历史交互序列的最大截断长度 |
| `train_fn.local_batch_size` | `128` | 单卡训练 batch size |
| `train_fn.main_module` | `"HSTU"` | 选择序列编码器架构（可切换为 `"SASRec"` 做基线对比） |
| `train_fn.item_embedding_dim` | `50` | 物品 ID Embedding 向量维度 |
| `hstu_encoder.num_blocks` | `8` | 堆叠 8 层 HSTU 转导单元 |
| `hstu_encoder.num_heads` | `2` | 注意力头数 |
| `hstu_encoder.dqk` / `dv` | `25` / `25` | 每个注意力头的 Query/Key 与 Value 维度 |
| `train_fn.loss_module` | `"SampledSoftmaxLoss"` | 采用负采样 Softmax 自回归训练目标 |
| `train_fn.num_negatives` | `128` | 每个正样本配对采样的负样本数量 |
| `train_fn.top_k_method` | `"MIPSBruteForceTopK"` | 评估阶段使用暴力最大内积搜索计算 HR@K 与 NDCG@K |

### 7.2 序列特征契约与 HSTU 核心实现（features.py / hstu.py）

在数据与模型的交界处，`generative_recommenders/research/data/features.py` 定义了贯穿全仓库的最小序列数据契约 `SequentialFeatures`：

```python
class SequentialFeatures(NamedTuple):
    past_lengths: torch.Tensor
    past_ids: torch.Tensor
    past_embeddings: Optional[torch.Tensor]
    past_payloads: Dict[str, torch.Tensor]
```

其中 `past_lengths` 记录每个样本的实际有效历史长度（供底层构造 Jagged Tensor），`past_ids` 与 `past_embeddings` 承载历史物品序列，而 `past_payloads` 字典则用于挂载时间戳（`timestamps`）、评分或动作类型（`ratings`）等附加序列信号。在公开 MovieLens 实验中，转换函数 `movielens_seq_features_from_row` 将原始行数据中的历史序列与目标监督项拆解组装为该结构。

进入核心模型文件 `generative_recommenders/research/modeling/sequential/hstu.py`，论文第 3 节的公式被映射为以下核心类与函数：

| 源码符号 | 对应论文概念 | 核心实现逻辑 |
| --- | --- | --- |
| `RelativePositionalBias` | 相对位置偏置 $r_{ab}^p$ | 构建可学习的相对位置偏移查找表，注入 $Q K^T$ 矩阵 |
| `RelativeBucketedTimeAndPositionBasedBias` | 相对时空融合偏置 $r_{ab}^{p,t}$ | 对相邻或成对事件的时间戳差值做对数分桶（bucketization），与位置偏置叠加 |
| `_hstu_attention_maybe_from_cache` | 带缓存的逐点激活注意力 | 计算 $\phi_2(QK^T + r_{ab}^{p,t})V$，支持从缓存状态增量拼接键值 |
| `SequentialTransductionUnitJagged` | 单层 HSTU 算子（Jagged 路径） | 执行线性投影切分 $U,V,Q,K$，调用注意力算子，完成 LayerNorm 与 $\odot U(X)$ 门控 |
| `HSTUJagged` / `HSTU` | 多层 HSTU 堆叠与对外编码器封装 | 管理多层残差堆叠，对接输入预处理器、Embedding 模块与输出后处理器 |

特别值得注意的是，`hstu.py` 中显式定义了缓存状态元组：

```python
HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

用于持久化跨步的 `v`、`padded_q`、`padded_k` 与层输出张量；同时在 `SequentialTransductionUnitJagged` 中高频调用 `torch.ops.fbgemm.jagged_to_padded_dense` 与 `dense_to_jagged`，将变长序列压缩在紧凑的一维拼接内存中运算，从代码层面印证了 HSTU 对长尾变长用户历史以及 M-FALCON 缓存复用的工程支持。

### 7.3 采样 Softmax 损失与 Brute-Force MIPS 评估模块

由于推荐系统的物品词表规模庞大，自回归生成训练通过 `SampledSoftmaxLoss`（位于 `autoregressive_losses.py`）近似全词表交叉熵。在每个有效序列位置上，模型将 HSTU 输出的上下文表征与真实下一个物品的正例 Embedding、以及随机抽样的 `num_negatives` 个负例 Embedding 计算相似度得分（点积或余弦相似度），进而最小化正例的负对数似然：

```text
positive = target_item_embedding
negatives = sampled_item_embeddings
logits = similarity(hstu_output, [positive, negatives])
loss = -log_softmax(logits)[positive]
```

在验证与测试阶段，`indexing/utils.py` 中的 `get_top_k_module` 根据配置实例化 `MIPSBruteForceTopK`（或混合内积模块 `MoLBruteForceTopK`），在全量候选物品库上执行精确的最大内积检索以统计 HR@K 与 NDCG@K。在工业生产环境中，这一评估检索层则会对接分布式 ANN 索引（召回阶段）或 M-FALCON 微批次打分引擎（排序阶段）。

## 8. 工程落地演进路径、常见误区与总结

将 Generative Recommenders 置于推荐系统演进史中审视，它与经典的 **YouTube DNN**（确立召回+排序两阶段漏斗）、**Clippy**（保障大规模多任务排序在流式训练中的数值稳定性）以及**曝光日志闭环治理**形成了清晰的递进关系：没有可靠的曝光日志与真实反馈样本，再庞大的生成式序列模型也只会对偏差与噪声做昂贵拟合；而当系统已经建立起成熟的工业级排序基线后，GR/HSTU 则指明了如何突破异构特征工程的边际递减，通过统一序列建模持续吸收海量数据与算力。

| 推荐系统演进阶段 | 核心工程矛盾 | 代表性架构与技术手段 |
| --- | --- | --- |
| 1. 冷启动与探索期 | 缺乏可靠曝光与正负反馈监督信号 | 内容先验表征、业务规则、探索策略、规范化曝光日志 |
| 2. 基础双塔与排序闭环 | 建立端到端个性化召回与点击率预估 | 双塔召回、ANN 向量检索、GBDT / 基础 MLP 排序 |
| 3. 成熟工业级 DLRM | 挖掘高阶异构特征交叉与多目标协同 | DLRM、DIN/DIEN、DCN-v2、MMoE/PLE、Clippy 稳定性优化 |
| 4. 生成式 Scaling 推荐 | 突破手工特征瓶颈，可预测地吸收算力红利 | GR 序列转导、HSTU 编码器、Stochastic Length、M-FALCON 推理 |

### 8.1 从传统推荐架构向 GR/HSTU 的渐进迁移步骤

对于希望在实际业务中借鉴 GR/HSTU 的工程团队，切忌盲目追求万亿参数规模，建议按照风险由低到高的六个阶段渐进演进：

1. **对齐时间序列日志契约**：梳理用户历史交互流，确保曝光、点击、完播、购买、跳过、隐藏等正负向动作均带有精确时间戳、稳定物品 ID/内容表征以及召回源上下文，并在离线构造样本时严格按曝光时刻截断以防未来信息穿越。
2. **建立同管线的序列推荐基线**：先在内部离线数据集上跑通基于采样 Softmax 的 SASRec 与基础版 HSTU，验证逐点激活注意力与相对时空偏置 $r_{ab}^{p,t}$ 在自有业务数据上的离线增益。
3. **引入交错的动作 Token（Action Token）**：打破仅用正向点击物品构建序列的惯例，将多类型用户反馈动作 $a_i$ 显式编码进序列，检验动作语义对捕捉细粒度偏好的贡献。
4. **改造目标感知排序（Target-Aware Ranking）**：在精排阶段将候选物品 $\Phi_{i+1}$ 追加至用户序列末端预测多任务动作概率，先作为原 DLRM 的一路增量特征或打分源（即论文中的 `GR (new source)` 模式）小流量上线验证。
5. **落地历史缓存与微批次推理摊销**：针对目标感知排序带来的延迟上升，参照 M-FALCON 依次实施多候选掩码合批、用户历史前缀 KV/状态缓存以及微批次（microbatching）切分，压降单请求 GPU 耗时。
6. **引入 Stochastic Length 扩展历史窗口**：在推理效率达标后，逐步拉长用户历史时间跨度（如从数百步扩展至数千步），配合特征加权的 Stochastic Length 采样控制训练成本，并通过线上 A/B 与长期留存护栏指标完成验收。

### 8.2 常见认知误区、方法边界与总结

在研读与转化这篇论文时，需要避免几类常见的简单化解读：
- **误区一：“GR 就是把语言模型 GPT 直接搬来算推荐”**。虽然 GR 借用了因果自回归的形式，但推荐序列的输入是高基数动态内容 $\Phi_i$ 与反馈动作 $a_i$ 的交错流，排序阶段必须做显式条件于候选 $\Phi_{i+1}$ 的目标感知打分，且注意力算子（HSTU 的非 Softmax 逐点激活与相对时间偏置）针对推荐分布做了根本改造。
- **误区二：“HSTU 可以无条件替代所有推荐模型或 Transformer”**。论文的万亿参数 Scaling Law 建立在海量流式交互数据、定制 Triton/CUDA 算子（Jagged Tensor、融合偏置）以及 M-FALCON 推理基础设施的协同之上；在特征稀疏但交互序列极短、或缺乏在线缓存改造能力的轻量系统中，传统 DLRM 依然具有极高的性价比。
- **误区三：“生成式序列建模自动消除了推荐偏差与多目标治理难题”**。GR 提供了强大的序列表示与算力扩展范式，但位置偏差（position bias）、曝光选择偏差（selection bias）、多目标商业价值权衡、内容生态多样性以及安全合规护栏，依然需要依赖严谨的样本工程、去偏策略与在线策略层共同保障。

总而言之，**Actions Speak Louder than Words** 的真正突破，在于它跳出了在 DLRM 内部局部修补交叉算子的思维定势，证明了**只要将用户的多维历史内容与反馈动作统一组织为因果序列转导问题，并配套解决长序列稀疏训练（Stochastic Length）、非平稳高效编码（HSTU）与多候选缓存摊销推理（M-FALCON），推荐系统同样能够走上随训练算力幂律扩展的 Scaling Law 轨道**。

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

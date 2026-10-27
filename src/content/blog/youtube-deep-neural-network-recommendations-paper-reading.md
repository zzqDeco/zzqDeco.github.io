---
title: "YouTube DNN 推荐系统论文精读：双阶段召回排序、候选生成与观看时长优化"
description: "精读 RecSys 2016 YouTube DNN 推荐系统如何用深度候选生成、排序模型、样本构造、特征工程和观看时长目标支撑大规模推荐"
pubDate: "2026-06-26T17:26:13+08:00"
updatedDate: "2026-06-26T17:26:13+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "Recommendation System"
  - "Deep Learning"
  - "Two-Tower"
  - "Ranking"
  - "YouTube"
draft: false
---

Paul Covington、Jay Adams 与 Emre Sargin 在 ACM RecSys 2016 发表的 **Deep Neural Networks for YouTube Recommendations** 篇幅仅有 8 页，却是工业界推荐系统领域最具影响力的经典文献之一。这篇论文的持久生命力源于它将超大规模视频推荐拆解为两个在计算预算、特征粒度与优化目标上各司其职的协同阶段：

```text
user history and context
  -> candidate generation: millions -> hundreds
  -> ranking: hundreds -> dozens
  -> impression
  -> watch / no watch / watch time
  -> offline training + online A/B
```

围绕这一漏斗闭环，论文系统阐述了深度候选生成、精排网络、防穿越样本构造、隐式反馈处理、负采样近似、近似最近邻（ANN）检索、内容时效性（freshness）建模、连续特征分位数归一化以及预期观看时长（expected watch time）加权优化方案。本文结合原论文图表（Fig. 1–7 与 Table 1）与数学推导，深入拆解以下核心工程问题：

1. 超大规模推荐系统为何必须划分为候选生成（candidate generation）与精排（ranking）两个阶段？
2. 候选生成为何被形式化为极大规模多分类（extreme multiclass classification）问题？
3. 训练阶段通过 Sampled Softmax 学习的分类分布，在线服务时如何转化为无需求解配分函数的 ANN 向量检索？
4. 为何训练样本的时序回滚构造与特征裁剪往往比单纯堆叠网络层数更能决定线上成败？
5. Example Age 特征如何修正滑动训练窗口带来的历史流行度时间平均偏置？
6. 精排阶段为何放弃单纯的点击率（CTR）目标，转而采用以观看时长加权的逻辑回归（Weighted Logistic Regression）？
7. 这篇发表于 2016 年的工业界名篇，与当今主流的双塔召回、序列注意力精排及多兴趣检索存在怎样的演进脉络？

## 1. 论文定位与核心三大挑战：Scale、Freshness、Noise

论文的基础档案如下表所示：

| 项目 | 内容 |
| --- | --- |
| 题名 | Deep Neural Networks for YouTube Recommendations |
| 作者 | Paul Covington, Jay Adams, Emre Sargin（Google） |
| 会议与 DOI | ACM RecSys 2016，`10.1145/2959100.2959190`（共 8 页） |
| 原始文献链接 | [ACM DOI](https://dl.acm.org/doi/10.1145/2959100.2959190)、[Google Research 页面](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)、[Google PDF](https://research.google.com/pubs/archive/45530.pdf) |

将这篇论文放入推荐系统的技术演进史中，可以清晰看到它在四条技术主线上的承前启后作用：

| 技术主线 | 论文发表前的主流范式 | 论文的核心突破与贡献 | 后续工业界演进方向 |
| --- | --- | --- | --- |
| 协同过滤表征 | 线性矩阵分解（MF）、WSABIE、成对排序损失 | 将深度候选生成定义为融合多源异构特征的非线性泛化协同过滤 | 双塔向量召回、DSSM 扩展、MIND/ComiRec 多兴趣召回 |
| 工业推荐架构 | 两阶段或多阶段漏斗已在内部使用，但公开文献缺乏端到端细节 | 完整公开从百万级语料到百级候选再到十级展示的双阶段漏斗与特征边界 | 多路召回融合、粗排（Pre-ranking）、精排、重排（Re-ranking）四级漏斗 |
| 优化目标设计 | 显式评分预测、点击率（CTR）二分类 | 指明 CTR 易受标题党诱导，提出基于加权逻辑回归直接逼近预期观看时长 | 多目标联合优化（MMoE/PLE）、长期用户留存、创作者生态与安全护栏 |
| 深度推荐工程 | 局部模块尝试小型神经网络 | 打通高基数 Embedding、Sampled Softmax 训练、离线向量导出与在线 ANN 检索全链路 | Wide & Deep、DeepFM、DIN/DIEN 序列注意力、生成式与大模型推荐 |

![Figure 1: YouTube mobile home recommendations](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig1-mobile-home.webp)

*Source: Covington et al., RecSys 2016, Fig. 1, CC BY-NC-ND 4.0.*

Fig. 1 展示了 YouTube 移动端首页（Home）的推荐界面。这一产品形态直接界定了整个推荐系统的物理边界：系统需要在特定的请求时刻、移动屏幕视口、卡片排列位次与缩略图上下文中，为用户实时组装出一列视频，并依靠用户随后的点击与真实观看时长完成反馈闭环。论文开篇将 YouTube 推荐面临的工程挑战高度凝练为 **Scale（规模）**、**Freshness（时效性）** 与 **Noise（反馈噪声）** 三个维度。

### 1.1 Scale：十亿级用户与百万级视频库下的算力约束

YouTube 拥有十亿量级的活跃用户与数以百万计的在线视频库，每日产生千亿规模的用户行为样本。在这一量级下，要求精排模型在数十毫秒的在线响应预算内对全库百万个视频逐一计算复杂交叉特征是完全不可行的。为此，系统必须通过两级漏斗分配算力预算：

```text
large corpus: millions of videos
  -> cheap but personalized retrieval (Candidate Generation)
  -> hundreds of candidate videos
  -> expensive feature-rich scoring (Ranking)
  -> dozens of displayed videos
```

在这套分工体系中，**候选生成（召回）阶段对召回覆盖率（Coverage / Recall）负责，精排阶段对头部列表的排序精度与业务目标校准负责**。一旦候选生成阶段漏掉了用户真正感兴趣的视频，后续精排模型即便表达能力再强也无从挽回；反之，若试图在候选生成阶段塞入高开销的候选级实时交叉特征，则会立即击穿在线延迟红线。

### 1.2 Freshness：高频动态内容流的非平稳分布挑战

与曲库或电影库相对静态的推荐场景不同，YouTube 每秒都有海量新视频上传。用户对突发新闻、热点事件与创作者最新动态存在强烈的消费诉求，视频的流行度分布随时间剧烈演化。然而，深度模型的离线训练数据通常截取自过去数周或数月的滑动时间窗口；若不对时间维度做显式建模，神经网络自然倾向于拟合整个训练窗口内的**时间平均流行度**，从而系统性偏爱已经积累了大量历史观看的老视频，对刚上传或正处于爆发初期的新视频反应迟钝。论文为此提出了极为巧妙的 **Example Age** 特征：在训练期显式告知模型每条样本距离训练窗口末端的时长，在线服务时将该特征统一置为 $0$（或微小的负数），直接引导模型输出“当前最新时刻”的即时热度分布。

### 1.3 Noise：隐式反馈的偏置与真实满意度鸿沟

在真实视频平台上，用户主动留下的显式点赞、点踩或满意度问卷极为稀疏，大规模模型训练只能依赖海量的隐式交互行为。然而，不同类型的用户反馈在数据丰度与噪声结构上存在显著差异：

| 反馈信号类型 | 数据可用性与规模 | 潜在偏差与噪声来源 |
| --- | --- | --- |
| Click（点击） | 数量极庞大、实时性最高 | 极易受夸张标题党（Clickbait）、诱导性缩略图与屏幕误触污染 |
| Watch（完成一定观看） | 规模大、比单纯点击更贴近真实兴趣 | 受自动连播（Autoplay）、后台播放与默认列表位置影响 |
| Watch Time（观看时长） | 连续信号，能有效反映用户沉浸度（Engagement） | 天然偏爱长视频，且娱乐消遣类与工具教学类视频的合理时长基准不同 |
| Like / Dislike（赞踩） | 语义极明确 | 极度稀疏，且主动表达群体的行为分布存在幸存者偏差 |
| Survey（满意度问卷） | 最接近长期主观满意度 | 采集成本高昂、覆盖率极低，仅适合做离线校准与护栏评估 |

针对上述噪声特征，论文在两阶段采取了差异化的监督策略：在候选生成阶段以“用户实际完成了视频观看（Watch）”作为正样本标签，过滤单纯的误点噪声；在精排阶段则抛弃传统的点击率（CTR）目标，将每个正样本按其真实观看时长（Watch Time）加权，从根本上遏制靠标题党骗取点击却被用户迅速退出的低质视频。

## 2. 双阶段漏斗总体架构：候选生成与精排的职责解耦

Fig. 2 描绘了 YouTube 推荐系统的端到端双阶段漏斗架构。

![Figure 2: Two-stage recommendation funnel](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig2-two-stage-funnel.webp)

*Source: Covington et al., RecSys 2016, Fig. 2, CC BY-NC-ND 4.0.*

图中六个核心模块的数据流转与工程职责归纳如下：

| 架构模块 | 输入信号 | 输出产物 | 系统角色定位 |
| --- | --- | --- | --- |
| Video Corpus | 全量在线可播放视频库 | 百万级（Millions）候选空间 | 底层检索语料池 |
| User History and Context | 历史观看序列、历史搜索词、地理区域、设备、人口统计学属性 | 用户侧多模态上下文特征 | 个性化意图来源 |
| Candidate Generation | 用户上下文表征 + 全量视频库输出向量 | 百级（Hundreds）高相关候选视频 | 低延迟粗粒度个性化召回 |
| Other Candidate Sources | 订阅频道更新、热门趋势、相似视频扩展、运营规则等 | 补充候选视频集合 | 多路召回融合（保障覆盖与新颖性） |
| Video Features | 候选视频细粒度属性、曝光历史统计、缩略图特征、用户-视频交叉关系 | 百级候选的高维排序特征向量 | 精排决策信号供给 |
| Ranking | 百级合并候选 + 丰富交叉与上下文特征 | 十级（Dozens）最终首页展示列表 | 精确校准预期观看时长与界面呈现 |

剖析这一双阶段漏斗，可以提炼出四条对现代工业推荐系统至关重要的设计准则。首先，**召回与精排的评估体系各司其职**：候选生成关注召回率（Recall@K）、平均精度均值（MAP）、候选多样性与数十毫秒内的检索延迟；精排则聚焦于具体曝光上下文下的预期观看时长、相对排序损失与 CPU 算力预算。其次，**深度候选生成作为多路召回生态的主力一环协同运作**：Fig. 2 中明确标出了 `Other candidate sources`，表明工业系统在深度向量召回之外，仍会并列保留订阅流、实时热点、同频道扩召与新内容探索等通道，共同汇入候选池。

再者，**跨阶段特征传递是精排校准的前提**：由于不同召回通道（Source）产出的原始分数量纲与分布互不可比，精排网络不仅接收候选视频本身，还显式将“哪些召回通道提名了该视频”以及“各通道给出的原始分数与排名”作为输入特征，由精排网络统一学习跨通道校准。最后，**线上 A/B 实验拥有最终一票否决权**：论文多次强调，离线的精确率、召回率与排序损失仅用于指导研究迭代方向，任何架构、特征或样本构造变更最终都必须通过线上真实流量的长周期 A/B 测试裁定。

## 3. 候选生成形式化：极大规模多分类与 Sampled Softmax

在候选生成阶段，论文摒弃了传统的矩阵分解内积拟合视角，创造性地将推荐召回建模为**极大规模多分类问题（Extreme Multiclass Classification）**：给定用户 $U$ 及其在时刻 $t$ 的上下文 $C$，模型需要在包含数百万个视频的语料库 $V$ 中，预测用户在时刻 $t$ 实际观看的视频类别 $w_t = i$。在标准 Softmax 形式下，类别 $i$ 的条件概率定义为：

$$
P(w_t=i \mid U,C)
= \frac{\exp(v_i^\top u)}
{\sum_{j\in V}\exp(v_j^\top u)},
$$

其中 $u \in \mathbb{R}^N$ 表示用户历史与上下文经深度神经网络编码后输出的 $N$ 维稠密向量（User/Context Embedding），$v_j \in \mathbb{R}^N$ 表示第 $j$ 个候选视频在 Softmax 输出层对应的类别权重向量（Output Video Embedding）。

这一建模方式巧妙实现了**离线分类训练与在线向量检索的数学同构**：在离线训练期，模型通过最小化真实观看视频类别 $i$ 的交叉熵损失，迫使具有相似观看偏好的用户向量 $u$ 与其常看的视频向量 $v_i$ 在内积空间中相互靠拢，本质上构成了一种深度非线性的协同过滤；而在在线服务（Serving）期，由于单次请求中用户向量 $u$ 固定且排序仅关心相对大小，分母上的全库配分函数 $\sum_{j\in V}\exp(v_j^\top u)$ 是一个与候选下标 $i$ 无关的正数常数，因此寻找条件概率 $P(w_t=i\mid U,C)$ 最高的 Top-$N$ 视频，在数学上严格等价于在向量空间 $\mathbb{R}^N$ 中求解最大内积检索（Maximum Inner Product Search, MIPS）：

$$
\operatorname{TopN}_{j\in V} \ v_j^\top u.
$$

### 3.1 训练加速：Candidate Sampling 与重要性加权

在离线训练中，由于全集 $V$ 的基数高达百万级，每次梯度更新都精确遍历分母 $\sum_{j\in V}\exp(v_j^\top u)$ 会带来难以承受的计算开销。论文未采用分层 Softmax（Hierarchical Softmax，因树结构上相近节点常将不相关的跨领域视频强行绑定，实测效果不佳），而是采用了**候选负采样（Candidate Sampling / Sampled Softmax）**：对每条正样本，从预设背景分布中随机采样数千个负例视频，仅在“1 个真实正类 + 数千个采样负类”的子集上计算交叉熵损失，并通过重要性加权（Importance Weighting）修正采样分布与真实全集分布之间的偏差。论文指出，这一采样近似方案在保证精度的同时，将训练速度提升了 100 倍以上。

需要特别注意这里的样本语义：候选生成阶段通过背景分布采出的数千个负类，仅用于近似 Softmax 归一化分母以撑开向量表征空间，并不代表用户真实看到并厌恶这些视频。

### 3.2 在线服务：近似最近邻（ANN）与四级评估解耦

在线服务时，为了在数十毫秒（tens of milliseconds）的严苛延迟上限内从百万级视频向量 $\{v_j\}_{j\in V}$ 中检索出与实时计算的用户向量 $u$ 内积最大的数百个候选，系统放弃了遍历精确打分，转而通过基于局部敏感哈希（LSH）或树/图索引的近似最近邻（ANN）服务完成次线性（sublinear）检索。

这种“训练用 Sampled Softmax 拟合分类概率、服务用 ANN 检索最大内积”的形态差异，要求工程上必须建立分层解耦的四级评估指标：

| 评估层级 | 核心观测指标 | 检验的工程目标 |
| --- | --- | --- |
| 离线模型训练层 | Sampled Softmax 交叉熵损失、梯度范数、Embedding 模长与奇异值分布 | 监控网络收敛稳定性与向量空间是否发生坍缩 |
| 离线召回评测层 | 留出集 Recall@K、MAP@K、品类覆盖率、新上传视频召回占比 | 评估表征空间对未来真实观看视频的捕获能力 |
| 在线 ANN 服务层 | 检索 P99 延迟、QPS 吞吐、近似检索相对于精确暴力检索的 Recall 折损率 | 平衡在线算力延迟预算与向量量化/剪枝精度损失 |
| 线上产品 A/B 层 | 人均观看时长、有效点击率、会话深度、新视频消费占比与留存护栏 | 验证召回候选与后续精排模型配合后的真实产品收益 |

## 4. 候选生成网络架构：异构序列池化与 ReLU 塔

Fig. 3 给出了候选生成深度神经网络的完整内部结构。

![Figure 3: Candidate generation architecture](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig3-candidate-generation.webp)

*Source: Covington et al., RecSys 2016, Fig. 3, CC BY-NC-ND 4.0.*

沿着自底向上的数据流向，该网络由五个核心表征模块串联而成：

### 4.1 观看历史与搜索历史的变长序列平均池化

用户的历史观看记录表现为变长的视频 ID 序列 $H_u = \{w_1,w_2,\ldots,w_m\}$。网络首先通过共享的大规模 Embedding 表将每个离散视频 ID 映射为稠密向量 $e_w \in \mathbb{R}^d$（这些底层输入 Embedding 与前文 Softmax 输出层的视频向量 $v_j$ 一同通过端到端反向传播联合学习）。为了将不同长度的历史序列转化为全连接层所需的定长输入，论文对比了求和（sum）、逐维取最大值（component-wise max）与平均池化，实证发现简单的**算术平均池化**（Average Pooling）效果最为稳健：

$$
h_{\text{watch}} =
\frac{1}{|H_u|}
\sum_{w\in H_u} e_w.
$$

用户的历史搜索查询（Search History）采用完全对称的方式处理：系统将历史搜索词切分为一元词（unigrams）与二元词组（bigrams）词元集合，查表映射为 Token Embeddings 后同样经平均池化聚合为定长的 `search vector`（$h_{\text{search}}$）。平均池化虽然舍弃了序列内部的严格先后次序，但对不同活跃度用户的序列长度变化天然免疫，在分布式训练与在线特征组装中具有极高的工程稳定性；同时，搜索词表征直接注入了用户主动表达的强显式意图，有效弥补了被动观看历史的滞后性。

### 4.2 人口统计学与多源异构上下文特征融合

若仅输入观看与搜索历史，模型对于刚注册或未登录的冷启动新用户将束手无策。为此，候选生成网络在底部广泛接入了非历史类的异构上下文信号（Heterogeneous Signals）：地理位置（Geographic Region）与设备类型（Device）作为离散类别特征映射为 Embedding；性别（Gender）、登录状态（Logged-in State）与用户年龄（Age）等二值或连续属性则归一化至 $[0,1]$ 区间后直接拼接。这些特征一方面为缺乏行为记录的新用户提供了可靠的群体与地域先验（Prior），另一方面也允许深层网络学习“同一观看历史在不同设备或不同国家下的差异化推荐偏好”。

### 4.3 由宽到窄的漏斗形 ReLU 多层感知机

底部所有池化向量与稠密特征拼接为宽输入向量 $z_0$：

$$
z_0 = [h_{\text{watch}}, h_{\text{search}}, x_{\text{geo}}, x_{\text{device}}, x_{\text{age}}, x_{\text{example\_age}}, \ldots],
$$

随后送入由宽逐层减半的塔式全连接 ReLU 网络（例如 `1024 ReLU -> 512 ReLU -> 256 ReLU`），通过多层非线性变换 $z_{\ell+1} = \operatorname{ReLU}(W_\ell z_\ell + b_\ell)$ 充分交叉历史兴趣与当前上下文，最终在顶层输出 256 维的用户表征向量 $u$。Fig. 3 右上方清晰对比了训练与服务的双头分流：离线训练接入 Softmax 层计算类别概率，在线服务则导出 $v_j$ 构建最近邻索引并实时检索 Top-$N$。

## 5. 时效性建模与样本构造：Example Age 与时间回滚机制

本篇论文最富洞察力的工程贡献，集中在第 3.3 节与第 3.4 节对**时间偏置修正（Example Age）**与**代理任务样本构造（Label and Context Selection）**的深度剖析。

### 5.1 Example Age：消除滑动训练窗口的时间平均偏置

Fig. 4 直观展示了视频流行度随上传后天数的剧烈衰减，以及引入 Example Age 前后的模型预测分布对比。

![Figure 4: Example age feature for freshness](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig4-example-age.webp)

*Source: Covington et al., RecSys 2016, Fig. 4, CC BY-NC-ND 4.0.*

设离线训练数据集截取自长度跨度为 $T$（如过去数周）的时间窗口。一条新视频上传后往往在最初几天内迎来观看峰值，随后热度迅速回落（如图中绿色经验曲线所示）。如果模型输入中不包含样本发生时刻的时间坐标，神经网络在最小化整个训练窗口的总交叉熵损失时，对于给定视频 $v$ 只能学到其在窗口期 $[1, T]$ 内的**时间平均观看概率**：

$$
\bar{p}(v)
= \frac{1}{T}
\sum_{t=1}^{T} p_t(v),
$$

这正是图中蓝色基线（Baseline）曲线平坦低矮、严重低估新视频刚上传时爆发峰值的根源。然而，线上推荐服务永远发生在训练窗口结束之后的最新时刻 $T+\epsilon$，系统真正需要预测的是窗口最右端的即时分布 $p_{T+\epsilon}(v)$。

为了解开这一时间混叠，论文为每条训练样本引入了一个标量特征 **Example Age**：

$$
a_{\text{example}} = t_{\max} - t_{\text{example}},
$$

其中 $t_{\max}$ 为该次训练数据集覆盖的最大截止时间戳，$t_{\text{example}}$ 为当前训练样本实际发生的时间戳（论文中还同时传入了该特征的平方项与平方根项以增强非线性拟合能力）。在离线训练时，模型通过 $a_{\text{example}}$ 清晰获知每条观看记录发生在距离当前多久以前，从而精准还原出视频随时间演化的真实热度曲线（图中红色曲线与绿色真实分布高度吻合）；到了在线服务（Serving）时刻，系统将所有请求的 `example age` 统一硬编码为 $0$（或略小于 $0$ 的微小数），直接指令神经网络输出处于时间轴最末端（即当下时刻）的概率预测值，无需依赖粗糙的人工规则提权便优雅解决了新视频与热点内容的时效性校准。

### 5.2 样本标签与上下文构造：防止未来信息穿越与代理任务走捷径

Fig. 5 对比了两种截然不同的训练样本构造方式：随机留出（Random Holdout）与时序回滚预测未来观看（Rollback History to Predict Future Watch）。

![Figure 5: Label and context selection](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig5-label-context-selection.webp)

*Source: Covington et al., RecSys 2016, Fig. 5, CC BY-NC-ND 4.0.*

在传统学术界协同过滤基准中，常见做法是从用户的全部观看集合中随机抽出一项作为预测标签（Fig. 5a），将其余所有观看记录（包括发生在标签时刻之后的未来行为）一并作为输入特征。论文一针见血地指出，这种做法在真实时间序列中会导致严重的**未来信息泄漏（Future Information Leakage）**与因果倒置。用户消费视频具有强烈的单向序列不对称性：例如用户通常按第一集、第二集、第三集的顺序追剧，或者先浏览宽泛的大众流行音乐、再逐步深入某个小众独立音乐人。若拿周三观看的第三集去“预测”周一观看的第二集，离线验证指标虽然极高，但线上服务时系统根本不可能提前获知用户的未来行为。因此，正确的样本构造必须严格执行**时间回滚（Rollback，Fig. 5b）**：选定用户在时刻 $t$ 的某次真实观看 $w_t$ 作为正样本 Label，仅将严格发生在时刻 $t$ 之前的历史观看与搜索记录 $\{a_i: t_i < t\}$ 组装为输入上下文，时刻 $t$ 之后的行为全部屏蔽。

在此基础上，论文还披露了另外三项直接决定线上 A/B 成败的样本工程细节：
- **主动裁剪强相关信号以防过拟合代理任务**：如果用户刚刚搜索了“Taylor Swift”并点击观看了搜索结果页的第一个视频，随后将该视频作为正样本标签，且在输入特征中原封不动地保留该次刚发生的搜索词与来源上下文，神经网络会迅速找到一条偷懒捷径——直接把首页推荐变成“复刻上一次搜索结果页”。这虽然能让离线 Next-Watch 预测准确率飙升，但用户回到首页时绝不希望满屏重复刚搜过的旧结果。为此，论文故意抹去了搜索历史的精细序列次序与紧邻搜索页结构，将其降格为无序词袋（Unordered Bag of Tokens），迫使模型挖掘更广泛的延伸兴趣。
- **每个用户采样固定上限的训练样本（Per-User Sample Capping）**：平台用户的活跃度服从极度偏斜的幂律分布。若直接按原始观看事件流生成样本，少数超重度活跃用户将贡献成千上万条样本并彻底主导梯度方向。论文对每个用户强制限制生成固定数量 $K$ 的训练样本（$\mathcal{D} = \bigcup_{u\in \mathcal{U}} \operatorname{SampleK}(\mathcal{E}_u, K)$），确保所有用户在损失函数中享有平等的权重，防止模型过拟合极客群体的怪癖。
- **采集全渠道观看而非仅限首页推荐曝光**：候选生成模型的训练正样本覆盖了用户在全网（包括外部网站嵌入播放器、搜索页、频道页等所有路径）触发的 YouTube 观看事件。假使训练仅局限在既有首页推荐系统曝光产生的点击上，新上传或未被旧推荐策略覆盖的视频将永远无法进入协同过滤传播链路，系统会迅速陷入自我强化的信息茧房。

## 6. 候选生成离线实验：异构特征叠加与网络深度收益

Fig. 6 汇报了候选生成模型在离线留出集上的平均精度均值（MAP@K）实验结果。

![Figure 6: Candidate feature and depth experiment](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig6-feature-depth-map.webp)

*Source: Covington et al., RecSys 2016, Fig. 6, CC BY-NC-ND 4.0.*

该实验的底层词表规模、序列截断长度与五档网络深度（Depth 0–4）配置整理如下：

| 实验配置项 / 网络深度 | 具体参数与结构设定 |
| --- | --- |
| 视频与搜索词表规模 | `1M` 视频类别（Video Vocabulary）、`1M` 搜索词元（Search Tokens） |
| Embedding 维度与截断窗口 | 向量维度统一为 `256 floats`；截取最近 `50` 次观看记录与最近 `50` 次搜索查询 |
| **Depth 0**（线性基线） | 无隐藏层，输入特征直接线性变换映射至 `256` 维 Softmax 空间（近似于线性矩阵分解） |
| **Depth 1** | `256 ReLU` |
| **Depth 2** | `512 ReLU -> 256 ReLU` |
| **Depth 3** | `1024 ReLU -> 512 ReLU -> 256 ReLU` |
| **Depth 4** | `2048 ReLU -> 1024 ReLU -> 512 ReLU -> 256 ReLU` |

综合剖析 Fig. 6 中四组特征组合（`Watches Only`、`Watches & Searches`、`Watches, Searches & Example Age` 与 `All Features`）随深度变化的曲线走势，可以得出两条清晰的结论。一方面，**多源异构特征的引入带来的性能跃升远超单纯加深网络**：在任意固定深度下，从仅使用观看历史到叠加搜索词意图、再叠加 Example Age 与人口统计学特征，MAP 均呈现显著阶跃；尤其是 Depth 0 线性模型在接入异构特征后难以消化跨模态交互，而多层 ReLU 塔则为 `{历史观看 × 搜索词 × Example Age × 地域设备}` 提供了高阶非线性交叉通道，使 All Features 曲线在深层结构下持续拉开优势。另一方面，**网络深度的边际收益递减规律显著**：从 Depth 0 到 Depth 3，各特征组合的 MAP 稳步提升；但从 Depth 3 增至 Depth 4 时增益已明显收窄，继续加宽加深不仅收敛愈发困难，还会推高在线 User Tower 的推理延迟，因此工程选型需在表征容量与在线服务预算之间适度止步。

## 7. 精排网络架构与特征工程：Impression 粒度与分位数归一化

经过候选生成与多路通道合并，候选规模从百万级收敛至数百个（Hundreds），精排（Ranking）模块正式接棒。与候选生成阶段仅能访问粗粒度用户特征不同，精排阶段由于打分候选数量少了四个数量级，得以基于**具体曝光记录**（Impression Data）引入高维度的候选视频特征以及精细的“用户—候选视频—召回源—界面上下文”交叉特征。Fig. 7 展示了精排深度网络的整体架构。

![Figure 7: Ranking network architecture](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig7-ranking-network.webp)

*Source: Covington et al., RecSys 2016, Fig. 7, CC BY-NC-ND 4.0.*

### 7.1 精排特征体系：计算频率解耦、跨槽位共享 Embedding 与交互统计

论文指出，尽管深度学习减轻了人工构造高阶特征交叉的负担，但将数十个原始业务信号转化为神经网络易于消化的输入表征，依然耗费了最主要的工程精力。精排特征按照基数与在线计算频率可分为四个正交维度：

| 特征分类维度 | 子类型 | 典型特征示例 | 编码与在线计算策略 |
| --- | --- | --- | --- |
| **按取值基数划分** | 单值类别特征（Univalent） | 当前曝光视频 ID（`impression video ID`） | 直接查表映射为单个稠密 Embedding |
| **按取值基数划分** | 多值类别特征（Multivalent） | 用户最近观看的 $N$ 个视频 ID 集合 | 查表后执行算术平均池化 $e_{\text{bag}} = \frac{1}{|B|}\sum_{i\in B} e_i$ |
| **按计算频率划分** | 请求级特征（Query Features） | 用户设备、语言、地理区域、近期全局活跃度 | **每次请求仅计算一次**，向所有候选广播复用以节省算力 |
| **按计算频率划分** | 曝光级特征（Impression Features） | 候选视频 ID、候选召回源得分、用户在该频道历史观看数 | **随候选数量线性增长**，需对数百个候选逐一计算与拼接 |

围绕高基数 ID 特征与用户-物品交互关系，论文重点强调了两项工程实践：
1. **同语义域跨特征槽位共享底层 Embedding 表与词表截断**：“视频 ID”这一实体空间会同时出现在当前曝光视频（`impression video ID`）、用户最近观看视频（`watched video ID`）、触发推荐的种子视频（`seed video ID`）等多个不同特征槽位（Feature Slots）中。精排网络让所有同属 Video ID 域的槽位**共享同一张底层大 Embedding 表**，但在拼接进入第一层全连接层时各自占据独立的输入通道；这既大幅削减了显存参数并加速长尾泛化，又允许上层网络针对不同槽位学习差异化权重。同时，针对百万级视频与频道空间，系统仅按点击曝光频次截取高频 Top-$N$ 纳入词表（Embedding 维度大致按唯一值数量的对数 $\log_2(|V|)$ 缩放），词表外（OOV）长尾 ID 统一映射至全零向量（Zero Embedding）。
2. **强区分度的用户-物品历史交互与疲劳度特征**：精排阶段最具预测力的信号来自刻画用户与当前候选视频（或同频道、同主题）过往互动深度的连续统计特征，包括“用户此前在该频道累计观看了多少个视频”、“用户距离上一次观看该主题视频过去了多久（Time since last watch）”，以及至关重要的**历史曝光未点击疲劳度特征（Number of previous impressions）**——如果某个视频在过去几次首页刷新中已连续向该用户曝光却均未被点击，模型会据此敏锐下调其预测得分，自然实现在线曝光降权与列表常看常新。此外，将候选生成阶段各通道提名该视频的标志位与原始得分传入精排，也是赋能多路召回校准的核心桥梁。

### 7.2 连续特征工程：经验累积分布分位数归一化与幂次展开

决策树模型（如 GBDT）对输入特征的单调非线性尺度变换天然不敏感，但基于梯度下降的深度神经网络对连续特征的数值尺度与长尾偏态分布高度敏感。推荐系统中的曝光次数、观看间隔时长、历史点击数等连续特征普遍呈现极端的幂律长尾与零膨胀分布，若直接输入原始数值（Raw Value），极易引发梯度爆炸或令激活函数饱和。

为此，论文提出在预处理阶段对每个连续特征 $x$（设其概率密度分布为 $f$）执行基于**经验累积分布函数**（Empirical CDF / Quantile Normalization）的分位数归一化，将其保序映射至 $[0,1)$ 区间内的均匀分布：

$$
\tilde{x}
= \int_{-\infty}^{x} d f.
$$

在工程实现上，离线管线只需单次扫描训练数据统计出一组百级分位点（Quantiles），在线与离线推理时在相邻分位点之间执行高速线性插值即可获得 $\tilde{x}$。更进一步，为了让浅层 ReLU 网络无需消耗过多隐藏层神经元就能轻松拟合变量的超线性与次线性响应曲线，论文将归一化后的特征 $\tilde{x}$ 与其**平方项和平方根项**一同作为三个独立通道输入网络底部：

$$
\left[\tilde{x},\quad \tilde{x}^2,\quad \sqrt{\tilde{x}}\right].
$$

后续消融实验证明，这一极低成本的多项式基展开能够稳定带来可观的离线与在线排序增益。

## 8. 预期观看时长目标与精排实验：加权逻辑回归解析

精排模型的优化目标是决定整个推荐系统价值取向的指挥棒。本节详细推导论文如何通过加权逻辑回归（Weighted Logistic Regression）与指数激活函数，在不改变分类训练框架的前提下直接预测预期观看时长。

### 8.1 为什么用 Expected Watch Time 替代单纯 CTR

如果精排模型仅以点击与否（0/1 二分类）训练标准逻辑回归并按点击率（CTR）排序，系统会不可避免地沦为标题党（Clickbait）的帮凶：一个封面耸人听闻但内容空洞的 10 分钟视频，可能诱骗大量用户点击并在观看 5 秒后愤怒关闭；在纯 CTR 指标下它会被排在榜首，而一个封面朴实但用户点击后会沉浸观看 8 分钟的优质深度视频反而排在后面：

| 候选视频特征 | 点击率（CTR） | 点击后平均观看时长 | 纯 CTR 排序位次 | 预期观看时长（Expected Watch Time）排序位次 |
| --- | --- | --- | --- | --- |
| 视频 A（标题党短退） | 极高（如 $10\%$） | 极短（如 $6$ 秒） | 榜首（预期点击高） | 靠后（每次曝光期望观看仅 $0.6$ 秒） |
| 视频 B（优质深度内容） | 中等（如 $3\%$） | 很长（如 $300$ 秒） | 靠后（点击率偏低） | 榜首（每次曝光期望观看达 $9.0$ 秒） |

因此，将精排目标切换为**每次曝光的预期观看时长（Expected Watch Time per Impression）**，能够将点击概率与点击后的消费深度完美统一。

### 8.2 加权逻辑回归的几率（Odds）数学推导

为了直接从分类网络输出预期观看时长，论文对标准交叉熵逻辑回归做了一处精妙改造：**所有发生点击的正样本（Positive Impressions）在损失函数中按其实测观看时长 $T_i$ 进行样本加权，所有曝光未点击的负样本（Negative Impressions）保持单位权重 $w=1$**。

我们来严格推导这一加权方式为何能让网络学出的对数几率（Logit）直接逼近预期观看时长。设某个特征邻域内的总曝光样本数为 $N$，其中发生点击的正样本数为 $k$（故未点击负样本数为 $N-k$），第 $i$ 个正样本的真实观看时长为 $T_i$。在标准无加权逻辑回归中，正类概率为 $p = \frac{k}{N}$，模型线性输出层 $z = f_\theta(x)$ 学习的是正负样本计数的对数几率（Log-Odds），即 $\exp(z) = \text{Odds} = \frac{p}{1-p} = \frac{k}{N-k}$。当我们将第 $i$ 个正样本的权重设为 $T_i$（等价于将该正样本在训练集中复制了 $T_i$ 份），而每个负样本权重仍为 $1$ 时，加权逻辑回归学到的新几率（Weighted Odds）自然变为正样本总权重与负样本总权重之比：

$$
\operatorname{Odds}_{\text{weighted}}
= \exp(f_\theta(x))
= \frac{\sum_{i=1}^{k} T_i}{N - k}.
$$

记该曝光条件下的真实点击概率为 $P = \frac{k}{N}$，点击条件下的期望观看时长为 $\mathbb{E}[T_{\text{click}}] = \frac{1}{k}\sum_{i=1}^k T_i$，则单次曝光的无条件预期观看时长（Expected Watch Time per Impression）为 $\mathbb{E}[T] = P \cdot \mathbb{E}[T_{\text{click}}] = \frac{\sum_{i=1}^k T_i}{N}$。将其代入上面的加权几率表达式，分子分母同除以 $N$ 可得：

$$
\operatorname{Odds}_{\text{weighted}}
= \frac{\frac{1}{N}\sum_{i=1}^{k} T_i}{1 - \frac{k}{N}}
= \frac{\mathbb{E}[T]}{1 - P}.
$$

在推荐列表曝光场景中，单次曝光的点击率 $P$ 通常是一个较小的值（例如 $P \ll 1$，几百分之几），利用一阶泰勒展开 $\frac{1}{1-P} \approx 1 + P \approx 1$，上式立即化简为：

$$
\operatorname{Odds}_{\text{weighted}}
= \frac{\mathbb{E}[T]}{1 - P}
\approx \mathbb{E}[T](1 + P)
\approx \mathbb{E}[T].
$$

这一推导揭示了一个极为优美的工程结论：在训练阶段，只需使用带样本权重的标准交叉熵损失（正样本权重 $w_i = T_i$，负样本权重 $w_j = 1$）训练逻辑回归；到了在线服务（Serving）阶段，只需将最后一层激活函数从 Sigmoid $\sigma(z)$ 替换为**指数函数**：

$$
\operatorname{score}(x) = \exp(f_\theta(x)),
$$

模型输出的标量分数即直接近似于该候选视频在当前曝光下的预期观看时长 $\mathbb{E}[T]$！

### 8.3 精排网络深度、宽度与消融实验解析

Table 1 给出了精排网络在次日留出集（Next-day Holdout Data）上不同隐藏层配置对应的**按用户加权逆序损失（Weighted Per-User Loss）**。

![Table 1: Ranking hidden layer depth and weighted per-user loss](/images/blog/youtube-dnn-recommendations/youtube-dnn-table1-ranking-depth-loss.webp)

*Source: Covington et al., RecSys 2016, Table 1, CC BY-NC-ND 4.0.*

论文定义的评估指标 Weighted Per-User Loss 紧扣观看时长目标：在同一用户的同一次推荐页面曝光中，任取一对正样本（被点击观看，时长为 $T_{\text{pos}}$）与负样本（曝光未点击）；若精排模型给负样本打出的分数高于正样本（发生逆序错误），则将该正样本的观看时长 $T_{\text{pos}}$ 计入“误排序时长损失”，最终指标即为全用户误排序观看时长占总观看时长的百分比（数值越低代表排序越精准）。实验结果整理如下：

| 精排隐藏层配置（Hidden Layers） | 按用户加权逆序损失（Weighted Per-User Loss） |
| --- | ---: |
| None（纯线性逻辑回归基线） | $41.6\%$ |
| `256 ReLU` | $36.9\%$ |
| `512 ReLU` | $36.7\%$ |
| `1024 ReLU` | $35.8\%$ |
| `512 ReLU -> 256 ReLU` | $35.2\%$ |
| `1024 ReLU -> 512 ReLU` | $34.7\%$ |
| `1024 ReLU -> 512 ReLU -> 256 ReLU` | $34.6\%$ |

审视 Table 1 及正文中配套的消融实验，可以提炼出三项极具价值的工业规律：
1. **非线性特征交叉与多层深度的权衡**：从无隐藏层（`None`，损失高达 $41.6\%$）引入单层 `256 ReLU`（损失骤降至 $36.9\%$），减损幅度达 $4.7$ 个百分点，证明在丰富的 Query 与 Impression 特征之上引入非线性变换收益巨大；随着网络继续加宽加深至三层塔式结构 `1024 -> 512 -> 256`，损失进一步降至 $34.6\%$，但相比两层结构 `1024 -> 512`（$34.7\%$）仅微降 $0.1$ 个百分点。由于精排需要对数百个候选逐一前向传播，最终线上选型必须严格受控于单次请求的在线 CPU 算力预算。
2. **连续特征幂次展开的边际贡献**：在 `1024 -> 512 -> 256` 架构下，若仅输入分位数归一化后的 $\tilde{x}$ 而去掉 $\tilde{x}^2$ 与 $\sqrt{\tilde{x}}$，加权逆序损失会上升 $0.2\%$（即退化至 $34.8\%$），说明在输入端显式提供超线性与次线性基函数对深层网络依然大有裨益。
3. **目标函数加权远比多堆一层网络更关键**：同样在 `1024 -> 512 -> 256` 架构下，若取消训练阶段的正样本观看时长加权（退化为正负样本等权重的普通 CTR 逻辑回归），加权观看时长逆序损失将剧烈恶化 $4.1\%$（飙升至 $38.7\%$，甚至不如最简单的单层 `256 ReLU` 加权模型）。这一对比有力印证了推荐系统的黄金定律：**样本标签与损失函数对业务目标的对齐精度，决定了模型效果的上限**。

## 9. 召回与精排全维度对照及现代推荐演进脉络

将候选生成与精排两个阶段置于同一表格下对比，可以一目了然地把握整个 YouTube DNN 架构的工程精髓：

| 对比维度 | 候选生成（Candidate Generation / 召回） | 精排（Ranking / 排序） |
| --- | --- | --- |
| **候选空间规模** | 百万级（Millions）全量视频库 | 百级（Hundreds）多路合并候选池 |
| **核心阶段使命** | 高召回覆盖率、低延迟粗筛个性化相关视频 | 高精度区分头部列表、校准多通道得分与业务目标 |
| **训练样本粒度与构造** | 全渠道观看事件；时序回滚历史 $\rightarrow$ 预测未来下一次观看 | 具体推荐页面曝光（Impression）$\rightarrow$ 点击与观看时长 |
| **主力输入特征** | 观看/搜索历史 ID 序列池化、人口统计学、Example Age | 曝光视频与历史交互统计、疲劳度计数、候选源与得分、分位数归一化连续特征 |
| **离线训练目标** | 极大规模多分类 + Sampled Softmax（重要性加权负采样） | 观看时长加权逻辑回归（正样本权重 $T_i$，负样本权重 $1$） |
| **在线服务（Serving）形态** | 导出视频向量 $v_j$，对用户向量 $u$ 执行近似最近邻（ANN）检索 Top-$N$ | 对每个候选视频前向传播 MLP，通过 $\exp(\text{logit})$ 预测预期观看时长 |
| **输出规模与硬约束** | 输出百级候选；受数十毫秒延迟与全局向量索引刷新约束 | 输出十级展示列表；受单次请求 CPU 算力预算与特征拉取延迟约束 |

站在 2016 年之后的十年技术演进视角回望，现代工业推荐系统的诸多前沿进展，均可视为对 YouTube DNN 特定简化假设的针对性升级：
- **从 Softmax 类别向量到显式双塔（Two-Tower）内容编码**：YouTube 2016 论文在候选生成阶段的视频侧向量 $v_j$ 本质上是 Softmax 输出层的独立 ID 类别参数，对毫无历史观看记录的纯新上传视频缺乏基于视频内容特征的泛化编码能力。现代双塔召回架构将视频/物品侧显式扩展为独立的 **Item Tower** $g_\phi(\text{item features})$，把标题文本、封面视觉、类目属性与统计特征实时编码为物品向量 $v = g_\phi(y)$，与用户塔 $u = f_\theta(x)$ 进行内积匹配，从而彻底打通了新物品的内容冷启动通道。
- **从无序平均池化到目标注意力与序列模型（DIN / DIEN / Transformer）**：YouTube DNN 采用简单的算术平均池化（Average Pooling）聚合历史观看与搜索序列，不仅抹平了行为的先后次序与时间间隔，而且面对不同候选视频时用户历史表征始终固定不变。随后阿里提出的 DIN（Deep Interest Network）引入目标候选注意力机制（Target Attention），根据当前待打分的候选视频动态加权用户的历史观看记录；DIEN、SASRec 以及近年兴起的生成式序列推荐（如 HSTU）则进一步引入自回归 Transformer 架构，深度建模长周期兴趣演化轨迹。
- **从单一用户向量到多兴趣向量召回（MIND / ComiRec）**：在候选生成阶段将一位用户数十次跨领域的观看历史强行压缩为单一 256 维向量 $u$，容易导致多峰兴趣（例如同一用户既看硬核编程教程、又看古典音乐与健身视频）在求平均后落入几个领域的几何中间地带。后续的 MIND（Multi-Interest Network with Dynamic Routing）与 ComiRec 通过胶囊网络或自注意力机制为每位用户提取多个兴趣胶囊向量 $\{u_1, u_2, \dots, u_K\}$，分别并发检索 ANN 索引再合并候选，显著改善了多兴趣用户的召回覆盖度。
- **从单一预期观看时长到多目标优化与安全生态治理**：论文通过加权逻辑回归将优化目标从 CTR 推进至 Expected Watch Time，成功遏制了浅层标题党；但在长期运营中，单纯追求观看时长仍会产生偏爱超长视频、诱导沉迷与信息窄化等副作用。现代大型推荐平台普遍演进为基于 MMoE 或 PLE 的多任务学习框架，同时预测点击、完播率、长观看时长、显式点赞/收藏、分享、负向点踩/不感兴趣以及随机抽样问卷满意度（Satisfaction Survey），并通过带约束的多目标融合公式兼顾创作者公平曝光与平台内容生态健康。

## 10. 论文局限性审视、核心工程启示与精读路径

客观审视这篇工业界经典文献，读者在借鉴其设计时应当留意四个历史与披露边界：其一，作为一篇工业架构概述论文，文中未披露完整的特征词表细节、分布式训练超参数、在线流式特征更新架构与线上 A/B 的具体提升百分比，不可将其当作开箱即用的完整复现手册；其二，所有实验结论均建立在 YouTube 超高密度的视频消费场景之上，对于低频高客单价的电商或重度依赖静态内容的垂类平台，特征重要性排序可能存在显著差异；其三，如第 9 节所述，候选生成侧缺乏显式的物品内容编码塔，且平均池化舍弃了序列顺序；其四，这篇 2016 年的论文仅代表 YouTube 转向深度学习初期的架构快照，如今的线上系统早已融入了深层序列建模、实时流式反馈与多目标强化学习。

尽管如此，这篇论文浓缩的五条工程方法论至今仍是构建任何推荐系统的必修课：
1. **严格区分训练目标与在线服务形态**：候选生成在离线是带采样的极大规模多分类，在线则是免配分函数的 ANN 最大内积检索；精排在离线是样本加权的交叉熵二分类，在线则是经 $\exp(\text{logit})$ 转换的预期时长打分。
2. **用防穿越的时间回滚构造替代随机留出**：离线样本必须严守在线服务时的信息边界（只用时刻 $t$ 之前的历史预测时刻 $t$ 的未来观看），必要时甚至需主动阉割过强的近期搜索结构特征，防止模型在代理任务上走捷径而损害真实首页体验。
3. **把时间非平稳性显式编码进模型**：通过 Example Age 在训练期学习流行度随时间的演化曲面，并在服务期置零输出当下时刻分布，比手工拍脑袋给新视频加固定偏置分更为精准优雅。
4. **召回与精排必须通过召回源特征紧密咬合**：ANN 召回仅提供一路候选池，精排模型必须显式消费候选来源 ID 与原始通道分数，并在曝光级（Impression）特征空间内融合疲劳度惩罚完成最终校准。
5. **模型收益高度依赖连续特征变换与目标函数对齐**：经验 CDF 分位数归一化配合 $[\tilde{x}, \tilde{x}^2, \sqrt{\tilde{x}}]$ 幂次展开保障了梯度平稳，而观看时长样本加权带来的 $4.1\%$ 逆序损失降幅更是远超多层网络堆叠。（关于如何在缺乏初始样本的业务中从零落地曝光日志、防穿越样本契约、双塔召回与四阶段上线闭环，可进一步参阅站内姊妹篇[《简单推荐算法实施指南：从冷启动弱监督到真实反馈闭环》](/blog/simple-recommender-cold-start-to-feedback-loop)。）

对于希望对照英文原版深入研读的读者，建议按以下十步线索串联阅读，并结合延伸文献拓展视野：
1. **Abstract & Section 1**：把握 Scale、Freshness、Noise 三大核心挑战。
2. **Section 2 & Fig. 2**：厘清 Candidate Generation 与 Ranking 两阶段漏斗的算力与职责边界。
3. **Section 3.1**：精读极大规模多分类形式化、Sampled Softmax 与在线 ANN 检索的衔接。
4. **Section 3.2 & Fig. 3**：分析观看/搜索历史平均池化与塔式 ReLU 结构。
5. **Section 3.3 & Fig. 4**：重点吃透异构特征融合与 Example Age 的训练/服务一致性设计。
6. **Section 3.4 & Fig. 5**：重点理解时序回滚（Rollback）、搜索特征去结构化与每用户定额采样。
7. **Section 3.5 & Fig. 6**：对照四条特征曲线与深度变化分析离线 MAP 收益。
8. **Section 4.1 & Fig. 7**：掌握单值/多值特征、Query/Impression 特征解耦、跨槽位共享 Embedding 与分位数归一化。
9. **Section 4.2**：推导加权逻辑回归的 Odds 公式与在线 $\exp(z)$ 激活。
10. **Section 4.3 & Table 1**：对比不同深宽度的加权逆序损失与两项关键消融实验。

| 拓展研究主题 | 经典参考文献 |
| --- | --- |
| YouTube 早期图与关联规则推荐 | Davidson et al., *The YouTube Video Recommendation System*, RecSys 2010 |
| 大词表线性排序与采样训练 | Weston et al., *WSABIE: Scaling Up To Large Vocabulary Image Annotation*, IJCAI 2011; Jean et al., ACL 2015 |
| 隐式反馈与停留时长建模 | Hu, Koren, Volinsky, ICDM 2008; Yi et al., *Beyond Clicks: Dwell Time for Personalization*, RecSys 2014 |
| 记忆与泛化联合精排架构 | Cheng et al., *Wide & Deep Learning for Recommender Systems*, DLRS 2016; DeepFM, IJCAI 2017; DCN, KDD 2017 |
| 序列注意力与多兴趣推荐 | Zhou et al., *DIN*, KDD 2018; *DIEN*, AAAI 2019; *MIND*, CIKM 2019; *SASRec*, ICDM 2018 |

## 11. 结语

回望 **Deep Neural Networks for YouTube Recommendations**，它之所以能在发表多年后依然被奉为工业推荐系统设计的圭臬，是因为作者没有将视野局限于某个精巧的局部网络算子，而是完整呈现了超大规模推荐系统从问题拆解、代理任务定义、防穿越样本构造、时效性偏差校准、特征归一化到商业目标函数对齐的系统工程全貌。

对于今天的算法工程师而言，研读这篇论文最大的收获在于习得其**面向生产约束的问题分解思维**：在面对海量候选与严苛延迟时，用“轻量向量召回 + 重度交叉精排”解耦算力；在面对历史滑动窗口与动态内容流冲突时，用 Example Age 显式对齐训练与服务的时间相位；在面对离线指标与线上体验背离时，毫不犹豫地通过时序回滚与特征裁剪修正样本构造；在面对标题党侵蚀用户体验时，用数学上自洽的观看时长加权逻辑回归重塑优化目标。无论底层神经网络从当年的多层 ReLU 演进为如今的深层 Transformer 还是生成式大模型，这套将算法模型严丝合缝嵌入数据闭环与产品界面的工程哲学，始终是工业界推荐系统迭代演进的基石。

## 参考资料

1. Paul Covington, Jay Adams, Emre Sargin. [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190). RecSys 2016.
2. Google Research. [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/).
3. Google Research PDF. [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/archive/45530.pdf).
4. James Davidson et al. The YouTube Video Recommendation System. RecSys 2010.
5. Jason Weston, Samy Bengio, Nicolas Usunier. WSABIE: Scaling Up To Large Vocabulary Image Annotation. IJCAI 2011.
6. Yi et al. Beyond Clicks: Dwell Time for Personalization. RecSys 2014.
7. Steffen Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
8. Yifan Hu, Yehuda Koren, Chris Volinsky. Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008.

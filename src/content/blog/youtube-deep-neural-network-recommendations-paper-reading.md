---
title: "YouTube DNN 推荐系统论文精读：双阶段召回排序、候选生成与观看时长优化"
description: "精读 RecSys 2016 YouTube DNN 推荐系统如何用深度候选生成、排序模型、样本构造、特征工程和观看时长目标支撑大规模推荐"
pubDate: "2026-06-26T17:26:13+08:00"
updatedDate: "2026-06-26T17:26:13+08:00"
tags:
  - "Paper Reading"
  - "Recommendation System"
  - "Deep Learning"
  - "Two-Tower"
  - "Ranking"
  - "YouTube"
draft: false
---

Paul Covington、Jay Adams 和 Emre Sargin 在 RecSys 2016 发表的 **Deep Neural Networks for YouTube Recommendations**，只有 8 页，却是工业推荐系统里非常耐读的一篇论文。它常被概括为“YouTube 用 DNN 做推荐”，但如果只看到 DNN，就会错过这篇文章真正值得反复读的部分。

它的核心贡献不是发明某个复杂神经网络结构，而是把超大规模视频推荐拆成了两个可服务化、可迭代、可评估的问题：

```text
user history and context
  -> candidate generation: millions -> hundreds
  -> ranking: hundreds -> dozens
  -> impression
  -> watch / no watch / watch time
  -> offline training + online A/B
```

这篇论文同时讨论了召回模型、排序模型、样本构造、隐式反馈、负采样、近似最近邻、freshness、特征工程、观看时长目标和线上 A/B。它没有给出一套可以直接下载运行的官方代码，也没有披露今天 YouTube 的内部架构；但它给出的系统分解方式，仍然是理解现代工业推荐系统的基础材料。

本文是一篇中文论文精读。重点不是把论文逐段翻译，而是回答几个工程问题：

1. 为什么 YouTube 要把推荐拆成 candidate generation 和 ranking 两个阶段？
2. candidate generation 为什么被写成 extreme multiclass classification？
3. sampled softmax 训练出来的概率，为什么线上 serving 时会变成 ANN 向量检索？
4. 为什么样本构造比模型结构更关键？
5. example age 为什么能缓解训练窗口带来的过时偏置？
6. ranking 为什么不直接优化 CTR，而要用 watch time 加权 logistic regression？
7. 这篇 2016 年论文对今天的双塔召回、冷启动、ANN、反馈闭环和 A/B 实验还有什么参考价值？

本文插入的 Fig. 1-7 和 Table 1 来自用户提供的本地 ACM PDF，仅做等比例裁切展示；版权和许可标注见每张图下方说明。本文只做论文级方法精读和工程复盘，不声称完成模型复现。

## 1. 论文信息与历史位置

论文基本信息如下。

| 项目 | 内容 |
| --- | --- |
| 题名 | Deep Neural Networks for YouTube Recommendations |
| 作者 | Paul Covington, Jay Adams, Emre Sargin |
| 机构 | Google |
| 会议 | ACM RecSys 2016 |
| DOI | `10.1145/2959100.2959190` |
| 页数 | 8 页 |
| 代码 | 无官方可运行代码 |
| 主要材料 | [ACM DOI](https://dl.acm.org/doi/10.1145/2959100.2959190)、[Google Research 页面](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)、[Google PDF](https://research.google.com/pubs/archive/45530.pdf) |

它的历史位置可以放在四条线之间看：

| 线索 | 论文之前 | 论文贡献 | 论文之后 |
| --- | --- | --- | --- |
| 协同过滤 | 矩阵分解、WSABIE、rank loss | 把候选生成视为非线性协同过滤 | 双塔召回、DSSM 类召回、MIND、多兴趣召回 |
| 工业推荐架构 | 召回和排序已经存在，但公开细节少 | 清楚展示 millions -> hundreds -> dozens 的漏斗 | 多路召回、粗排、精排、重排、多目标系统 |
| 目标函数 | CTR、点击、评分预测 | 强调 expected watch time，弱化 clickbait | 多目标优化、长期满意度、安全与创作者生态 |
| 深度学习推荐 | 早期 DNN 推荐尝试 | 把大规模 embedding、sampled softmax、ANN 串起来 | Wide & Deep、DeepFM、DIN/DIEN、Transformer 推荐 |

这篇论文的“新”不在今天看起来普通的 ReLU MLP，而在于它把 DNN 放进了真实系统闭环：如何采样、如何上线、如何控制延迟、如何处理新鲜内容、如何用 A/B 校准离线指标。这也是很多论文读起来像模型文章，而这篇更像工程文章的原因。

![Figure 1: YouTube mobile home recommendations](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig1-mobile-home.webp)

*Source: Covington et al., RecSys 2016, Fig. 1, CC BY-NC-ND 4.0.*

Fig. 1 本身很简单：YouTube 移动端首页展示一列推荐视频。但这个截图非常关键，因为后文所有模型讨论最终都要回到这个产品界面：用户并不是在一个离线 benchmark 上给视频打分，而是在某个请求、某个屏幕、某个位置、某个缩略图和上下文里选择是否观看。

因此，论文从一开始就不是在做“用户喜欢哪个视频”的抽象排序，而是在做“给定这个用户此刻进入 YouTube 首页，应该展示哪些视频，顺序是什么，并且后续如何用观看行为反馈训练系统”。

## 2. YouTube 推荐的三大约束

论文把 YouTube 推荐的挑战概括为 scale、freshness、noise。这三个词看起来普通，但每一个都会直接改变模型设计。

### 2.1 Scale: 不能全量精排

YouTube 的候选视频是百万级甚至更大规模，用户也是十亿级，行为样本是千亿级。这个规模下，如果想对每个请求把所有视频都送入复杂排序模型打分，在线延迟和计算成本都会不可接受。

所以系统必须先做 candidate generation：

```text
large corpus: millions of videos
  -> cheap but personalized retrieval
  -> hundreds of candidate videos
  -> expensive ranking model
  -> dozens of displayed videos
```

这个结构背后的工程原则是：**召回阶段负责覆盖，排序阶段负责精细区分**。

如果召回阶段漏掉了一个本应展示的视频，排序阶段再强也没有机会修复；如果召回阶段返回太多低质量候选，排序阶段需要浪费大量计算过滤噪声。因此两者不是谁替代谁，而是不同计算预算下的分工。

### 2.2 Freshness: 推荐库是动态分布

视频推荐和电影评分推荐不同。电影库相对稳定，而 YouTube 每秒都有大量新视频上传。用户对新鲜内容有明显需求，热门视频的流行度还会随时间快速变化。

这会带来一个很实际的问题：训练数据覆盖的是过去几周或几个月，模型学到的自然是训练窗口内的平均行为。如果没有额外机制，模型容易偏向过去已经有大量反馈的视频，对刚上传或刚开始传播的视频反应迟缓。

Freshness 不只是“给新视频加一点权重”。论文提出的 example age 特征更精细：在训练时告诉模型样本距离训练窗口结束还有多远，在 serving 时把这个特征设为 0 或略负，让模型预测“训练窗口末端”的分布，而不是“训练窗口平均”的分布。

### 2.3 Noise: 没有干净的满意度标签

YouTube 有点赞、点踩、问卷等显式反馈，但这些反馈稀疏。大规模训练主要依赖隐式反馈，尤其是 watch、click、watch time。

这和推荐系统里的一个基本事实一致：多数用户不会主动标注满意度，系统只能从行为中推断偏好。但行为不是偏好的纯标签：

| 行为 | 可用性 | 噪声来源 |
| --- | --- | --- |
| click | 数量多，实时性好 | 标题党、缩略图诱导、误触 |
| watch | 比 click 更接近兴趣 | 自动播放、背景播放、外部上下文 |
| watch time | 能反映 engagement | 长视频天然优势、学习/娱乐目标差异 |
| like/dislike | 语义强 | 极稀疏，用户群体偏 |
| survey | 更接近满意度 | 成本高，覆盖低 |

论文选择用 watch 作为 candidate generation 的正样本，用 expected watch time 作为 ranking 的主要目标。这并不意味着 watch time 是完美目标，而是说明在当时 YouTube 场景里，它比 CTR 更能抑制 clickbait。

## 3. 总体系统: candidate generation -> ranking

论文最重要的一张图是 Fig. 2。

![Figure 2: Two-stage recommendation funnel](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig2-two-stage-funnel.webp)

*Source: Covington et al., RecSys 2016, Fig. 2, CC BY-NC-ND 4.0.*

图里有几个关键模块。

| 模块 | 输入 | 输出 | 角色 |
| --- | --- | --- | --- |
| video corpus | 全量视频库 | 百万级候选空间 | 被检索对象 |
| user history and context | 观看历史、搜索历史、地理、设备、人口统计等 | 用户上下文表示 | 个性化来源 |
| candidate generation | 用户表示 + 视频库 | 几百个候选 | 大规模召回 |
| other candidate sources | 其他召回通道 | 候选补充 | 多路召回融合 |
| video features | 视频属性、曝光上下文、历史统计 | 排序特征 | 精排信号 |
| ranking | 几百个候选 + 丰富特征 | 几十个最终展示 | 目标函数优化 |

两个阶段的差异不是“一个浅一个深”，而是问题定义不同。

Candidate generation 面对的是百万级视频类目，需要在几十毫秒级延迟里找到大致相关的几百个视频。因此它只能使用较粗的用户和内容信号，并且必须能转成高效近似检索。

Ranking 面对的是几百个候选，可以使用更重的特征：视频缩略图相关信号、用户是否最近看过同频道、候选来自哪个召回源、这个候选在不同源里的分数、过去曝光但未点击次数等。排序阶段的模型可以更昂贵，因为打分对象少得多。

这解释了为什么“召回模型 ANN 分数很高”不能直接等于“最终排序应该靠前”。ANN 分数只说明在召回表示空间里相似或相关，排序还要判断当前界面、当前请求、候选来源、用户短期疲劳、展示频次、观看时长预期等因素。

### 3.1 双阶段架构的工程好处

这张漏斗图给今天的推荐系统仍然有几个直接启发。

第一，召回和排序的优化指标可以不同。召回阶段关心 coverage、recall、候选质量和延迟；排序阶段关心最终列表的 watch time、CTR、满意度、多样性和业务约束。

第二，召回可以多路。图里明确有 `other candidate sources`。这说明 YouTube DNN candidate generation 不是唯一入口，而是候选池的一路来源。工业系统通常会保留热门、订阅、同频道、相似视频、运营策略、新内容探索等多种候选源。

第三，排序需要知道候选从哪里来。论文在 ranking 部分强调，把 candidate generation 产生的信息传给 ranking 很关键，例如哪个 source 提名了视频、source 给了什么分数。否则 ranker 无法校准不同召回源之间不可比较的分数。

第四，线上 A/B 是最终裁判。论文明确说离线 precision、recall、ranking loss 会用于迭代，但最终效果依赖 live experiment。因为推荐系统的真实目标是用户在产品里的行为，而离线指标经常和线上结果不完全一致。

## 4. Candidate generation 形式化: 推荐作为极大规模分类

论文把 candidate generation 写成 extreme multiclass classification。给定用户 $U$ 和上下文 $C$，预测用户在时刻 $t$ 观看的视频 $w_t$ 属于视频库 $V$ 中哪一个类别：

$$
P(w_t=i \mid U,C)
= \frac{\exp(v_i^\top u)}
{\sum_{j\in V}\exp(v_j^\top u)}
$$

这里：

| 符号 | 含义 |
| --- | --- |
| $u \in \mathbb{R}^N$ | 用户和上下文经过 DNN 得到的 dense embedding |
| $v_j \in \mathbb{R}^N$ | 第 $j$ 个视频的 output embedding |
| $V$ | 全量视频类别集合，规模百万级 |
| $w_t$ | 用户在时刻 $t$ 实际观看的视频 |

这个写法有两层含义。

第一，训练时它是一个分类问题。用户的历史和上下文经过神经网络变成 $u$，输出层里每个视频有一个向量 $v_j$，softmax 给出“下一个观看视频是 $j$”的概率。

第二，serving 时它天然变成向量检索问题。因为对于某个用户请求，$u$ 已经确定，所有视频的 $v_j$ 可以预先导出到向量库。排序全量视频等价于找内积 $v_j^\top u$ 最大的 top $N$ 视频。

也就是说，论文里的 candidate generation 已经非常接近今天常说的双塔召回：

```text
training:
  user/context tower -> u
  video output embeddings -> v_j
  sampled softmax over video classes

serving:
  user/context tower -> u
  ANN maximum inner product search over v_j
  return top N candidates
```

它不是严格意义上的“两个独立 tower 同时编码 user 和 item feature”的现代双塔，因为视频侧 output embedding 更像 softmax 类别向量，而不是由视频内容 tower 实时编码。但从“用户向量 + 物品向量 + ANN 召回”的工程形态看，它已经是现代向量召回架构的重要先例。

### 4.1 为什么不用全 softmax

视频类别是百万级，全 softmax 的分母需要遍历所有视频：

$$
\sum_{j\in V}\exp(v_j^\top u)
$$

每个训练样本都算全量分母会非常昂贵。论文使用 candidate sampling：从背景分布采样几千个负类，再用 importance weighting 修正采样偏差。论文提到这种方式相比传统 softmax 有超过 100 倍加速。

训练过程可以抽象成：

```text
positive: actually watched video i
negatives: sampled videos from background distribution
loss: cross entropy over positive + sampled negatives
correction: importance weighting for sampled classes
```

这和今天推荐系统里的 sampled softmax、in-batch negatives、sampled negatives、two-tower contrastive training 有相通之处。不过要注意，sampled negative 在这里不是“用户真实讨厌的视频”。它只是为了训练分类器而采样的对比类别。

这和冷启动推荐里的负样本边界一致：未曝光 item 不是真实负反馈，只能作为采样 denominator 或弱负样本参与训练。

### 4.2 为什么线上变成 ANN

训练时 softmax 会输出校准概率，但线上 candidate generation 并不需要精确概率。召回阶段只需要找到得分最高的若干视频，也就是：

$$
\operatorname{TopN}_{j\in V} \ v_j^\top u
$$

如果对百万级视频逐个打分，延迟不可接受。论文说 serving time 要在 tens of milliseconds 约束下得到 top $N$，所以使用近似最近邻或类似 hashing 的 sublinear 检索。

这一步非常关键：**训练目标是分类概率，在线系统用的是向量检索近似 topN**。二者不是一回事。

因此，工程实现中至少要区分四套指标：

| 层级 | 指标 |
| --- | --- |
| 训练 | sampled softmax loss、训练收敛、embedding 分布 |
| 离线召回 | Recall@K、MAP、覆盖率、新视频覆盖、长尾覆盖 |
| ANN 服务 | latency、QPS、recall loss、index freshness、embedding 版本 |
| 线上 | watch time、CTR、会话深度、满意度、A/B guardrail |

只看训练 loss 不能证明召回系统上线有效，只看 ANN 分数也不能证明最终排序有效。

## 5. Candidate model architecture

Fig. 3 展示了 candidate generation 的模型结构。

![Figure 3: Candidate generation architecture](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig3-candidate-generation.webp)

*Source: Covington et al., RecSys 2016, Fig. 3, CC BY-NC-ND 4.0.*

这张图可以拆成六层。

### 5.1 Watch history embedding average

用户观看历史是一个变长视频 ID 序列：

$$
H_u = \{w_1,w_2,\ldots,w_m\}
$$

每个 watched video ID 查一个 embedding，然后把这些 embedding 聚合成固定长度 watch vector。论文尝试了 sum、component-wise max 等方式，最终发现简单 average 表现最好：

$$
h_{\text{watch}} =
\frac{1}{|H_u|}
\sum_{w\in H_u} e_w
$$

这个设计有优点也有损失。

优点是工程上非常稳定。无论用户历史长度是多少，输入维度固定；embedding lookup 和平均池化易于分布式训练；serving 也容易维护。

损失是序列顺序被丢弃。用户刚看过的视频和很久以前看过的视频在简单 average 里没有天然区分，除非另外加入时间、近期窗口或行为统计特征。后来的 DIN、DIEN、Transformer 推荐模型，很大程度上是在补足这种序列建模能力。

### 5.2 Search history embedding average

搜索历史类似处理。搜索 query 被切成 unigram 和 bigram token，每个 token 查 embedding，再做平均，得到 search vector。

这一步非常重要，因为搜索行为比观看行为更主动。用户搜索“Taylor Swift”通常说明短期意图很强。但论文后面也会强调，不能把最近一次搜索页面的结构直接泄漏给首页推荐模型，否则模型会学成“复现搜索结果页”。

### 5.3 Dense context features

图里还有 geographic embedding、example age、gender 等上下文特征。

论文明确说，地理区域、设备等可以 embedding 后拼接；gender、logged-in state、age 等简单 binary 或 continuous feature 直接归一化到 $[0,1]$ 后输入。

这说明 YouTube DNN 不是纯协同过滤模型。它在候选生成阶段就接入了 heterogeneous signals，包括：

| 特征组 | 例子 | 作用 |
| --- | --- | --- |
| watch history | 最近观看视频 ID | 协同过滤主信号 |
| search history | 搜索 query token | 显式短期意图 |
| demographic | 地区、设备、年龄、登录状态 | 新用户 prior 和分群 |
| freshness | example age | 修正训练窗口偏置 |

这对冷启动很重要。新用户没有观看历史时，模型仍然能从地理、设备、语言、默认 segment、上下文等特征给出合理 prior。只是这些 prior 不能替代真实行为历史。

### 5.4 Wide first layer + ReLU tower

所有特征拼接成一个宽输入层，然后经过几层 fully connected ReLU。论文使用了 bottom wide、upper layer progressively smaller 的 tower pattern。

可以抽象为：

$$
z_0 = [h_{\text{watch}}, h_{\text{search}}, x_{\text{geo}}, x_{\text{device}}, x_{\text{age}}, \ldots]
$$

$$
z_{\ell+1} = \operatorname{ReLU}(W_\ell z_\ell + b_\ell)
$$

最终得到 user vector $u$，送入 softmax 或 ANN。

今天看这个结构不复杂，但在 2016 年的工业推荐语境里，它相当于把矩阵分解升级成可以吸收任意 dense/categorical feature 的非线性模型。

### 5.5 Training softmax vs serving ANN

图上右侧有两个头：

```text
training: softmax -> class probabilities
serving: nearest neighbor index -> approx. top N
```

这点非常值得强调。很多人会把训练图理解成“线上也跑 softmax 得到概率”，但论文明确说 serving 时不需要校准概率，问题转成 dot-product nearest neighbor search。

这也是推荐系统工程常见的“训练-服务形态差异”：

| 阶段 | 目标 | 实现 |
| --- | --- | --- |
| 训练 | 学到区分 watched video 的表示 | sampled softmax |
| 导出 | 把视频 output embedding 写入索引 | embedding export |
| 服务 | 给用户向量找 topN 视频 | ANN / hashing / MIPS |
| 排序 | 在候选上重打分 | ranker |

模型文件、embedding 导出、向量索引、索引版本、在线请求版本都需要严格记录。否则训练出来的模型和线上使用的索引不一致，很难排查问题。

## 6. Heterogeneous signals 与 example age

Fig. 4 是论文里最有工程味道的一张图。

![Figure 4: Example age feature for freshness](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig4-example-age.webp)

*Source: Covington et al., RecSys 2016, Fig. 4, CC BY-NC-ND 4.0.*

图展示了某个视频随上传后天数变化的 class probability。绿色是经验分布，蓝色 baseline 模型更像训练窗口平均，红色加入 example age 后能更好拟合上传时间附近的峰值。

### 6.1 训练窗口的隐性偏置

假设训练数据覆盖最近几周。对于一个新上传后快速爆发的视频，训练窗口内的不同时刻观看概率差异很大。但如果模型只看到“用户历史 -> 视频 label”，它学到的往往是整个窗口的平均概率：

$$
\bar{p}(v)
= \frac{1}{T}
\sum_{t=1}^{T} p_t(v)
$$

而 serving 发生在训练窗口的末端或未来，系统真正想要的是：

$$
p_{T+\epsilon}(v)
$$

如果 $p_t(v)$ 非平稳，$\bar{p}(v)$ 和 $p_{T+\epsilon}(v)$ 可能差很多。新视频、病毒式传播、新闻事件和热点内容都会出现这个问题。

### 6.2 Example age 的做法

论文的做法是在训练样本里加入 example age：

$$
a = t_{\max} - t_{\text{example}}
$$

其中 $t_{\max}$ 是训练数据中最大的观测时间，$t_{\text{example}}$ 是当前训练样本的时间。训练时模型能学习“这个样本发生在训练窗口末端前多少时间”。Serving 时把这个特征设为 0 或略负，表示“现在就是训练窗口末端之后”。

直觉上，模型不再被迫把一个视频在整个训练窗口内的平均流行度压成单一值，而能学习它随时间变化的曲线。

### 6.3 Freshness 不是简单新视频 boost

很多系统会用规则：

```text
score = model_score + alpha * freshness_score
```

这当然能让新内容有机会曝光，但它的问题是粗糙。它无法区分“刚上传但没人喜欢”和“刚上传且正在传播”的内容，也难以让模型学习不同主题、不同用户群体对新鲜内容的差异。

Example age 更像是把时间相位交给模型：

```text
model_input = user_history + content_signals + example_age
model learns time-dependent likelihood
```

它不是替代探索机制，而是修正训练窗口造成的静态偏置。新内容真正获得反馈仍然需要曝光机会、探索预算和实时日志。

### 6.4 对今天系统的启发

今天做推荐系统，freshness 通常会拆成多层：

| 层级 | 方法 |
| --- | --- |
| 训练样本 | example age、time decay、按时间窗口采样 |
| 特征 | item age、publish time、trend velocity、recent CTR/CVR |
| 召回 | 新内容召回、热点召回、订阅/关注召回 |
| 排序 | freshness calibration、多目标权重 |
| 探索 | guided exploration for new items |
| 监控 | new item coverage、time-to-first-exposure、fresh watch time |

YouTube DNN 的 example age 是其中一个经典切入点：把时间非平稳性显式暴露给模型，而不是假设过去窗口和当前 serving 分布一致。

## 7. Label and context selection: 样本构造比结构更重要

Fig. 5 讲的是 label 和 input context 怎么选。

![Figure 5: Label and context selection](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig5-label-context-selection.webp)

*Source: Covington et al., RecSys 2016, Fig. 5, CC BY-NC-ND 4.0.*

这张图对推荐系统非常重要，因为它回答了一个容易被忽略的问题：训练样本到底是什么？

### 7.1 不要预测随机 held-out watch

传统协同过滤常见做法是：从用户历史里随机 hold out 一个 item，拿剩余 item 去预测它。Fig. 5a 就是这种形式。

问题是，这会泄漏未来信息。假设用户观看顺序是：

```text
Taylor Swift hit song
Taylor Swift interview
smaller niche artist
concert live clip
```

如果随机 hold out 第二个视频，用后面发生的观看行为去预测前面的视频，模型会获得线上 serving 时不可能拥有的信息。离线指标可能看起来好，但线上不一定转化。

### 7.2 Rollback history 预测 future watch

Fig. 5b 的做法更接近真实线上场景：选择某个观看事件作为 label，只输入它之前发生的行为，之后行为全部排除。

可以写成：

$$
\text{input}_t = \{a_i: t_i < t\}
$$

$$
\text{label}_t = w_t
$$

这就是 rollback。它避免了未来行为泄漏，也能保留消费序列的非对称性。

论文强调，预测 future watch 在 A/B testing 中表现更好。这是一个非常强的工程信号：离线样本构造方式会深刻影响线上结果。

### 7.3 为什么要丢掉一些强信号

论文还提到一个反直觉点：有些信号必须故意 withheld。例子是用户刚搜索了 “Taylor Swift”。如果模型知道这个 search query 的结构和 label 来源，它可能学会把搜索结果页复现在首页推荐里。这样离线预测下一个观看会很好，但首页体验很差。

因此，他们把搜索 query 表示为 unordered bag of tokens，丢弃部分序列和来源结构，让模型不要过拟合 surrogate task。

这说明推荐模型训练不是“特征越多越好”。如果某个特征只在训练 label 构造里强相关，但在目标展示场景里会让模型走捷径，就会造成训练目标和产品目标脱节。

### 7.4 固定每用户样本数

论文还有一个重要细节：每个用户生成固定数量训练样本，等价于在 loss 里让用户权重更均衡，避免少数高活跃用户主导训练。

如果不这么做，重度用户贡献大量 watch 事件，模型会更偏向重度用户的消费模式。对于首页推荐，平台通常希望对所有用户都有稳定体验，而不是只优化高活跃用户群体。

可以把用户级采样写成：

$$
\mathcal{D}
= \bigcup_{u\in \mathcal{U}}
\operatorname{SampleK}(\mathcal{E}_u, K)
$$

其中 $\mathcal{E}_u$ 是用户 $u$ 的观看事件集合，每个用户最多采样 $K$ 个训练样本。

### 7.5 从所有 YouTube watches 生成样本

论文说训练样本来自所有 YouTube watches，包括嵌入在其他站点上的观看，而不只是推荐系统产生的观看。原因是，如果只用推荐曝光产生的观看，系统会过度 exploitation，新内容难以进入协同过滤传播链路。

这点和现代推荐里“logging policy”和“exposure bias”的讨论有关。只用旧推荐策略产生的数据训练新模型，会强化旧策略的可见性偏差。YouTube 在 candidate generation 阶段利用更广泛的 watch 行为，试图让用户从其他路径发现的内容也能被传播到推荐系统里。

不过这也带来一个边界：所有 watches 并不等同于所有 homepage impressions。Candidate generation 学的是更广义的“用户可能观看什么”，ranking 才学习具体 impression 上会不会点击、会看多久。

## 8. Candidate experiments: features and depth

Fig. 6 展示了 candidate generation 的离线实验。

![Figure 6: Candidate feature and depth experiment](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig6-feature-depth-map.webp)

*Source: Covington et al., RecSys 2016, Fig. 6, CC BY-NC-ND 4.0.*

实验设置包括：

| 项目 | 论文描述 |
| --- | --- |
| 视频词表 | 1M videos |
| 搜索 token 词表 | 1M search tokens |
| embedding 维度 | 256 floats |
| watch bag size | 最近 50 个 watches |
| search bag size | 最近 50 个 searches |
| softmax 类别 | 1M video classes |
| depth 0 | 近似线性 factorization |
| deeper tower | 逐层减半的 ReLU 网络 |

图里有四条曲线：

1. Watches Only
2. Watches & Searches
3. Watches, Searches & Example Age
4. All Features

可以读出几个结论。

第一，只有 watch history 的 depth 0 模型类似线性矩阵分解，是一个强 baseline，但 MAP 明显低。

第二，加入 searches 后，模型捕获用户主动意图，MAP 提升明显。

第三，加入 example age 后，freshness 相关建模进一步改善离线 MAP。

第四，all features 与更深网络叠加提升最好，说明 DNN 的价值不只是 embedding，而是能建模多类异构特征之间的非线性交互。

### 8.1 深度为什么有用

如果只有用户 watch embedding 和视频 embedding，线性模型已经能做协同过滤：

$$
\operatorname{score}(u,v) = p_u^\top q_v
$$

但当输入里加入 search、geo、device、age、example age、gender 等特征后，模型需要学习交互：

```text
same watch history + different country -> different recommendations
same search token + different device -> different short-term intent
same video age + different topic -> different freshness curve
```

ReLU tower 提供了这种非线性组合能力。Fig. 6 的价值在于说明：深度不是孤立有效，而是在 heterogeneous signals 进入模型后更有效。

### 8.2 深度不是越深越好

论文列出的 candidate depth 配置是：

| Depth | Architecture |
| --- | --- |
| 0 | linear layer to 256 softmax dimension |
| 1 | 256 ReLU |
| 2 | 512 ReLU -> 256 ReLU |
| 3 | 1024 ReLU -> 512 ReLU -> 256 ReLU |
| 4 | 2048 ReLU -> 1024 ReLU -> 512 ReLU -> 256 ReLU |

Fig. 6 中深度提升收益逐渐变小。论文也提到继续增加宽度和深度会让增量收益变小、收敛更困难。

这和今天大规模推荐系统仍然一致：召回模型需要服务低延迟和可更新索引，不会无限堆复杂度。很多时候一个稳定可导出 embedding、可高频重训、可低延迟 ANN 的模型，比一个离线指标略高但服务复杂的模型更有价值。

## 9. Ranking 的输入边界

Candidate generation 返回几百个候选后，ranking 才开始工作。论文对 ranking 的描述非常清楚：它使用 impression data，把候选预测专门化和校准到具体用户界面。

这句话可以拆成三层。

第一，ranker 的样本单位是 impression，而不是全量用户-视频对。它知道“这个视频在这个请求、这个页面、这个位置、这个缩略图下是否被点击和观看”。

第二，ranker 能使用更丰富的特征，因为候选只有几百个。它可以接入很多 candidate generation 阶段无法承受的特征。

第三，ranker 要融合不同候选源。不同 source 的分数不可比较，ranker 需要知道 source ID 和 source score，学习如何校准。

## 10. Ranking network architecture

Fig. 7 展示了 ranking 网络。

![Figure 7: Ranking network architecture](/images/blog/youtube-dnn-recommendations/youtube-dnn-fig7-ranking-network.webp)

*Source: Covington et al., RecSys 2016, Fig. 7, CC BY-NC-ND 4.0.*

Ranking 模型仍然是 MLP tower，但输入特征更复杂。

### 10.1 Categorical feature: univalent vs multivalent

论文把 categorical feature 分成单值和多值。

| 类型 | 例子 | 表示方式 |
| --- | --- | --- |
| univalent | impression video ID | 查 embedding |
| multivalent | 用户最近看过的 N 个 video IDs | 多个 embedding 平均 |

和 candidate generation 一样，多值类别特征通过平均 embedding 变成固定宽度向量：

$$
e_{\text{bag}}
= \frac{1}{|B|}
\sum_{i\in B} e_i
$$

这让 feedforward network 能处理变长历史。

### 10.2 Query feature vs impression feature

论文还区分 query feature 和 impression feature。

| 类型 | 计算频率 | 例子 |
| --- | --- | --- |
| query feature | 每个请求计算一次 | user context、语言、设备、近期整体行为 |
| impression feature | 每个候选 item 计算一次 | 视频 ID、候选源、候选分数、用户与该视频关系 |

这个区分在工程上非常重要。Query feature 可以缓存或一次性计算，impression feature 会随候选数量线性增加。Ranker 的在线延迟往往受 impression feature 计算影响很大。

### 10.3 Shared embeddings

论文提到，同一个 ID space 的 categorical features 共享底层 embedding。例如 video ID 可以出现在：

1. 当前 impression video ID
2. 用户最近 watched video ID
3. seed recommendation video ID
4. candidate source 相关 video ID

底层 embedding 共享，但每个 feature 仍然作为独立输入喂给上层网络。这样既能减少参数和提升泛化，又允许上层网络学习不同 feature slot 的语义差异。

这和今天推荐模型里的 feature slot、shared embedding table、field-aware representation 很接近。

### 10.4 OOV zero embedding 与词表截断

论文说大基数 ID 空间会按 clicked impressions 频率截断，只保留 top $N$；OOV 映射到 zero embedding。

这体现了工程取舍：

| 问题 | 取舍 |
| --- | --- |
| 视频 ID 极多 | 只保留高频 ID |
| 长尾 ID 稀疏 | 用 zero embedding 或其他 fallback |
| embedding 参数巨大 | 维度随唯一值数量大致按 log 增长 |
| serving 内存有限 | 高频词表优先 |

今天的系统可能会用 hashing、动态 embedding、频繁重建词表、item tower 内容编码等方式处理长尾，但核心问题没变：大基数类别特征是推荐模型参数和服务复杂度的主要来源。

## 11. Continuous feature normalization

Ranking 里还有大量 continuous feature，例如 time since last watch、number of previous impressions 等。论文强调，连续特征归一化对收敛非常关键。

给定连续特征 $x$，分布为 $f$，论文用累计分布把它映射到 $[0,1)$：

$$
\tilde{x}
= \int_{-\infty}^{x} d f
$$

工程上用单次遍历数据得到分位数，再做线性插值近似。也可以理解为 quantile/CDF normalization：

```text
raw feature x
  -> empirical CDF F(x)
  -> normalized feature x_tilde in [0,1)
```

除了 $\tilde{x}$，论文还输入：

$$
\tilde{x}^2,\quad \sqrt{\tilde{x}}
$$

这样网络更容易形成超线性和次线性函数。

### 11.1 为什么这在推荐里重要

推荐 ranker 的连续特征分布通常很偏：

| 特征 | 分布问题 |
| --- | --- |
| 视频曝光次数 | 长尾，极少数 item 极高 |
| 最近观看间隔 | 多尺度，从秒到月 |
| 历史点击次数 | 零膨胀，长尾 |
| 频道观看次数 | 高活跃用户偏大 |
| item age | 新内容密集，旧内容长尾 |

如果直接把 raw value 输入 MLP，梯度尺度和激活分布会很不稳定。树模型对单调变换不敏感，但神经网络很敏感。因此归一化不是“预处理细节”，而是模型能否收敛和泛化的前提。

今天很多系统会进一步使用：

1. log1p transform
2. bucketization
3. quantile normalization
4. z-score by segment
5. missing indicator
6. feature clipping
7. feature freshness timestamp

但 YouTube DNN 论文给出的 CDF normalization 已经抓住了核心：连续特征必须被改造成神经网络容易学习的分布。

## 12. Modeling expected watch time

论文的 ranking objective 是 expected watch time per impression，而不是单纯 CTR。

### 12.1 为什么不是 CTR

CTR 容易被 clickbait 影响。用户点击了一个视频，不代表满意；如果很快退出，长期看不是好推荐。论文认为 watch time 更能反映 engagement。

这可以用一个简单对比说明：

| 视频 | CTR | 平均观看时长 | CTR 排序 | Watch time 排序 |
| --- | --- | --- | --- | --- |
| A: 标题党短退出 | 高 | 低 | 靠前 | 靠后 |
| B: 稳定观看 | 中 | 高 | 中等 | 靠前 |

Expected watch time 不是完美满意度，但比 click probability 更能抑制“骗点”。

### 12.2 Weighted logistic regression

论文使用 logistic regression under cross-entropy loss，但对 clicked positive examples 按 observed watch time 加权，对 unclicked negatives 权重为 1。

设训练样本总数为 $N$，正样本数为 $k$，第 $i$ 个正样本 watch time 为 $T_i$。论文推导说 logistic regression 学到的 odds 近似：

$$
\frac{\sum_i T_i}{N-k}
$$

如果点击概率 $P$ 很小，则 learned odds 近似：

$$
E[T](1+P) \approx E[T]
$$

serving 时用指数函数：

$$
\operatorname{score}(x) = \exp(f_\theta(x))
$$

把模型输出转成接近 expected watch time 的 odds。

### 12.3 如何理解这个目标

这个目标有几个非常工程化的优点。

第一，保留了 logistic classification 的训练框架。样本仍然是 clicked vs unclicked impression，只是 positive weight 变成 watch time。

第二，它把 watch time 融入 pairwise/listwise 排序之前的 pointwise 模型里，便于大规模训练和 serving。

第三，它让模型自然偏好“被点击且观看长”的视频，而不是“容易被点击但马上退出”的视频。

第四，它仍然需要 guardrail。Watch time 可能偏向长视频、沉浸式内容或某些消费模式，不一定等同于长期满意度。今天的推荐系统通常会把 watch time、satisfaction survey、like/dislike、not interested、diversity、creator health、安全策略等放进多目标框架。

## 13. Ranking experiments: depth, width and CPU budget

Table 1 是 ranking 部分的主要实验结果。

![Table 1: Ranking hidden layer depth and weighted per-user loss](/images/blog/youtube-dnn-recommendations/youtube-dnn-table1-ranking-depth-loss.webp)

*Source: Covington et al., RecSys 2016, Table 1, CC BY-NC-ND 4.0.*

表格列出不同 hidden layer 配置下的 weighted per-user loss：

| Hidden layers | Weighted per-user loss |
| --- | ---: |
| None | 41.6% |
| 256 ReLU | 36.9% |
| 512 ReLU | 36.7% |
| 1024 ReLU | 35.8% |
| 512 ReLU -> 256 ReLU | 35.2% |
| 1024 ReLU -> 512 ReLU | 34.7% |
| 1024 ReLU -> 512 ReLU -> 256 ReLU | 34.6% |

论文对 weighted per-user loss 的定义是：在 next-day holdout data 上，对一个用户同一页里正负 impression pair 进行比较。如果 negative impression 得分高于 positive impression，就把 positive impression 的 watch time 视为误预测观看时长。最终 loss 是误预测观看时长占总观看时长的比例。

这个指标比普通 accuracy 更贴近 watch time 目标：错把一个长观看正样本排低，比错把短观看正样本排低代价更大。

### 13.1 表格怎么读

从 None 到 256 ReLU，loss 从 41.6% 降到 36.9%，说明非线性隐藏层显著有效。

从 256 到 512，提升很小，说明单层加宽收益有限。

1024 单层更好，说明宽度能增加表达能力。

两层和三层继续提升，但边际收益下降。最佳表格项是 1024 -> 512 -> 256，loss 为 34.6%，只比两层 1024 -> 512 的 34.7% 低 0.1 个百分点。

论文强调最终选择还受 serving CPU budget 约束。即使三层略好，如果在线 CPU 成本过高，也不能无限扩大。

### 13.2 额外 ablation

论文还给了两个很有用的 ablation：

1. 对 1024 -> 512 -> 256 模型，只输入 normalized continuous features，不输入 powers，会让 loss 增加 0.2%。
2. 同样结构下，如果 positive 和 negative examples 等权，而不是按 watch time 加权，watch-time weighted loss 增加 4.1%。

第二点尤其关键：目标函数设计比多加一层网络更重要。模型结构从两层到三层只提升 0.1%，但错误的样本权重会损失 4.1%。这也是推荐系统里常见的规律：样本、标签、权重、曝光语义，往往比网络结构更决定成败。

## 14. Candidate generation 与 ranking 的对照

把两部分放在一起，可以得到一张更清楚的对照表。

| 维度 | Candidate generation | Ranking |
| --- | --- | --- |
| 输入规模 | 百万级视频库 | 几百个候选 |
| 目标 | 找到可能相关的视频 | 给候选精确排序 |
| 样本 | 用户历史 -> future watch | impression -> click/watch time |
| 主要信号 | watch/search history、demographic、example age | user-item 关系、候选源、曝光历史、连续特征 |
| 训练 | sampled softmax classification | watch-time weighted logistic regression |
| Serving | ANN / nearest neighbor topN | MLP score per impression |
| 输出 | hundreds candidates | dozens displayed videos |
| 约束 | 超低延迟、大索引、召回覆盖 | CPU budget、特征计算、目标校准 |
| 风险 | 漏召回、新内容覆盖不足 | clickbait、过拟合曝光位置、候选源校准不准 |

这张表也解释了为什么推荐系统不能只做一个统一大模型来排序全量 item。理论上可以，工程上很难承受。

## 15. 这篇论文的工业经验

### 15.1 召回不是排序

Candidate generation 返回的是“可能相关”的候选，不是最终列表。它的分数也不一定可直接跨来源比较。Ranking 必须对候选重新校准。

这对工程落地很重要：第一版 ANN 召回上线时，不应该让 ANN score 直接接管最终排序。更稳妥做法是低权重进入候选源，然后让 rank policy 或 ranker 融合：

```text
final_candidates =
  ann_candidates
  + rule_candidates
  + hot_candidates
  + profile_candidates
  + exploration_candidates

ranker_features include:
  candidate_source
  source_score
  source_rank
  model_version
  index_version
```

### 15.2 样本构造决定线上迁移

论文反复强调 surrogate problem 的选择。预测随机 held-out watch 离线可能好，但线上不如预测 future watch。保留太多站内结构信息会让模型复现搜索页而不是做首页推荐。

这说明推荐训练样本必须模拟 serving 时的信息边界：

| 问题 | 正确做法 |
| --- | --- |
| 用户特征是否包含曝光后行为 | 不包含，必须用 snapshot at time |
| label 是否来自未来 | 可以是 future watch，但 input 只能 rollback 到 label 前 |
| 未曝光 item 是否是负样本 | 不是真实负反馈 |
| 曝光未点击是否是负样本 | 可以是低权重负反馈，但需有曝光上下文 |
| 旧策略偏差如何处理 | 记录 logging policy，必要时做探索和校正 |

### 15.3 Freshness 要进模型和系统

Example age 是模型层面的 freshness；但系统还需要索引 freshness、内容上线到首曝时延、新内容探索预算、实时反馈更新。

如果 item embedding 每天导出一次，新视频入库后没有 embedding 或没有索引更新，再好的 example age 也无法召回它。因此 freshness 是一条链：

```text
content ingestion
  -> feature extraction
  -> candidate embedding
  -> ANN index update
  -> exploration exposure
  -> feedback collection
  -> training sample
  -> retraining / continual update
```

YouTube DNN 论文没有展开这整条链，但 Fig. 4 提醒我们：推荐系统面对的是非平稳分布。

### 15.4 Watch time 目标优于 CTR，但不是终点

论文对 CTR 的批判非常清楚：点击容易奖励 clickbait。Watch time 更接近 engagement。

但今天看，watch time 也不是最终答案。它可能鼓励沉迷、长视频偏置、重复消费、低质量长停留。成熟系统通常需要多目标：

| 目标 | 可能指标 |
| --- | --- |
| 短期 engagement | CTR、watch time、long view |
| 满意度 | survey、like/dislike、not interested |
| 长期留存 | next-day return、session quality |
| 多样性 | category diversity、creator diversity |
| 安全 | policy violation risk、borderline content |
| 生态 | creator exposure fairness、new creator coverage |

因此，论文里的 expected watch time 是理解工业目标函数的起点，不是今天所有推荐系统的终点。

## 16. 与现代推荐系统的关系

### 16.1 与双塔召回

今天的双塔召回常写成：

$$
u = f_\theta(\text{user features})
$$

$$
v = g_\phi(\text{item features})
$$

$$
s(u,v)=u^\top v
$$

YouTube DNN candidate generation 是一种早期形态：用户侧由 watch/search/context features 经过 DNN 得到 $u$，视频侧是 softmax output embedding $v_j$，线上通过 nearest neighbor 检索。

区别在于现代双塔通常会显式建 item tower，把 item 的文本、图像、类目、作者、统计特征编码成向量，方便冷启动 item。YouTube 2016 论文里的视频 output embedding 更依赖 ID 类别和观看反馈，对全新视频的内容冷启动披露较少。

### 16.2 与 Wide & Deep / DeepFM

Google 同期另一篇重要工作是 Wide & Deep。Wide 部分记忆稀疏共现，Deep 部分泛化。YouTube DNN 里没有显式 wide linear cross，但它同样面对“记忆和泛化”的问题：

| 问题 | YouTube DNN 的处理 |
| --- | --- |
| 记忆历史协同信号 | video ID embedding |
| 泛化到新组合 | ReLU tower + heterogeneous signals |
| 高基数类别 | shared embeddings + vocabulary truncation |
| 连续特征 | quantile normalization + powers |

DeepFM、xDeepFM、DCN 等后续模型更系统地建模 feature crossing，而 YouTube DNN 更强调大规模系统里的可服务化。

### 16.3 与 DIN/DIEN/Transformer 推荐

YouTube DNN 使用 average pooling 处理历史。后来的 DIN 使用 target attention，根据候选 item 对用户历史加权；DIEN 建模兴趣演化；Transformer 模型进一步建模序列顺序和长程依赖。

可以把它们看作在补足 YouTube DNN 的一个简化：

```text
YouTube DNN:
  history embeddings -> average -> fixed vector

DIN:
  history embeddings + target item -> attention pooling

Transformer:
  ordered behavior sequence -> contextual sequence representation
```

但这些模型通常更重，在线成本更高。是否值得使用，取决于候选规模、特征系统、延迟预算和业务收益。

### 16.4 与多兴趣召回

YouTube DNN 把用户压成一个向量 $u$。但用户兴趣常常是多峰的：同一个人可能同时喜欢音乐、编程、健身和新闻。单向量召回可能混合兴趣，导致某些兴趣被平均掉。

后来的 MIND、ComiRec、多兴趣 capsule 等方法会为一个用户生成多个兴趣向量：

$$
\{u_1,u_2,\ldots,u_K\}
$$

然后每个兴趣向量召回一批候选。这可以看作对 YouTube DNN 单用户向量召回的扩展。

### 16.5 与在线学习和近实时特征

论文提到 up-to-the-second impression and watch history 对 responsive recommendations 很重要，但细节超出论文范围。今天的推荐系统会更强调：

1. 实时曝光去重
2. session 内短期行为更新
3. feature store 的在线/离线一致性
4. streaming training examples
5. nearline model update
6. ANN index incremental update
7. bandit exploration 和 OPE

YouTube DNN 论文是一个静态快照，但它已经指出了实时反馈的重要性。

## 17. 与冷启动 Blog 的衔接

这篇论文对“简单推荐算法从冷启动到稳定闭环”的启发非常直接。

### 17.1 未曝光不是负样本

Candidate generation 的 sampled negatives 是为了近似 softmax，不代表用户真实不喜欢。Ranking 的 negatives 是 unclicked impressions，前提是视频确实展示给用户。

因此：

```text
unseen item != negative feedback
exposed but no click = weak negative with context
sampled negative = training denominator / contrastive negative
```

这条边界对冷启动系统非常关键。

### 17.2 召回是一路候选源

Fig. 2 里有 other candidate sources。第一版 ANN 召回不应该接管全部推荐，而应该作为一路候选源，在低权重或 shadow 模式下收集反馈。

### 17.3 Freshness 需要显式建模

如果新 item 没有曝光机会、没有 embedding、没有索引刷新，模型会天然偏旧。冷启动商品同样如此。需要 new item exploration、content prior、index refresh 和反馈闭环共同工作。

### 17.4 线上 A/B 是最终裁判

论文明确说离线指标不一定和线上 A/B 相关。冷启动系统上线时也不能只看离线 Recall@K。至少要看：

1. exposure log 写入成功率
2. interaction join rate
3. candidate valid rate
4. new item coverage
5. CTR / long dwell / conversion
6. hide / dislike / report
7. session-level guardrail

### 17.5 样本快照必须防止数据穿越

Fig. 5 的 rollback 思想可以直接映射到训练样本契约：

```text
TrainingExampleV1:
  user_snapshot_at_t: only features before exposure/label time
  positive_item: watched/clicked item
  negative_items: exposed but no action, sampled negatives with source
  exposure_context: request_id, position, page, source, model_version
  label_time: event time
```

如果 user snapshot 包含 label 之后行为，离线指标会虚高，线上效果会崩。

## 18. 简化复刻清单

如果要做一个教学版或中小规模业务版 YouTube DNN，不需要一开始复刻 YouTube 的全部复杂性。可以按下面顺序做。

### 18.1 数据与日志

先定义日志。

```text
ExposureLog:
  request_id
  user_id
  item_id
  position
  candidate_source
  source_score
  rank_score
  model_version
  index_version
  policy_version
  exposed_at

InteractionLog:
  request_id
  user_id
  item_id
  event_type
  watch_time_or_dwell_time
  occurred_at
```

没有曝光日志，ranking 的负样本语义不成立。

### 18.2 Candidate generation 样本

构造 rollback future watch 样本：

```text
for each user:
  sort watch events by time
  choose label event at t
  input = events before t
  label = watched item at t
```

对每个用户控制样本数量，避免高活跃用户主导训练。

### 18.3 Candidate generation 模型

最小结构：

```text
watch_history_item_ids -> item embeddings -> mean pooling
search_or_query_tokens -> token embeddings -> mean pooling
context features -> embeddings / normalized dense values
concat -> MLP -> user embedding
sampled softmax over item IDs
```

训练后导出 item output embedding，建立 ANN index。

### 18.4 ANN serving

```text
request user features
  -> user tower
  -> user embedding
  -> ANN topN
  -> hard filters
  -> merge with rule/hot/profile candidates
  -> ranker
  -> exposure log
```

ANN index 需要记录版本：

```text
index_version = model_version + item_embedding_snapshot + build_time
```

### 18.5 Ranking 样本

Ranking 样本必须来自曝光：

```text
positive:
  exposed and clicked/watched
  weight = watch_time or dwell_time

negative:
  exposed and not clicked after attribution window
  weight = 1 or low weight
```

推荐系统第一版如果没有足够曝光样本，不要假装能训练成熟 ranker。可以先用规则 rank policy，把日志积累起来。

### 18.6 Ranking 模型

最小 ranker 输入：

1. user features
2. item features
3. user-item cross features
4. candidate source
5. source score
6. recent exposure count
7. time since last interaction
8. normalized continuous features and powers

输出可以先做 expected dwell/watch time：

```text
clicked positive weight = dwell/watch time
unclicked negative weight = 1
activation = exp(logit)
score approximates expected time
```

### 18.7 上线流程

```text
shadow:
  produce ANN candidates but not affect ranking
  log overlap, valid rate, latency

low weight:
  ANN candidates enter candidate pool
  keep rule/hot/profile fallback
  monitor guardrails

ramp-up:
  increase ANN source quota or rank weight
  A/B test by user bucket

stable:
  periodic retraining
  index publishing
  rollback model/index/policy independently
```

## 19. 局限性与批判

这篇论文影响很大，但不能过度外推。

### 19.1 论文披露有限

论文只给出高层架构和关键经验，没有完整数据 schema、训练 pipeline、特征列表、ANN 实现细节、在线基础设施、A/B 配置或代码。读者不能把文中模型图当作完整可复现系统。

### 19.2 实验基于内部数据

实验结果来自 YouTube 内部数据和在线 A/B，没有公开数据集复现。Fig. 6 和 Table 1 只能说明在当时 YouTube 场景下这些设计有效，不能直接证明所有推荐业务都应采用同样结构。

### 19.3 Watch time 不是满意度等价物

Watch time 能抑制一部分 clickbait，但它可能鼓励长时间停留而非长期满意。今天需要多目标、长期指标和安全策略共同约束。

### 19.4 Candidate generation 对内容冷启动披露不足

论文主要讲 watch/search/demographic 和视频 ID embedding，没有详细说明全新视频如何基于内容特征进入召回。对于商品推荐、UGC 新内容推荐或资产库推荐，item cold start 需要文本、图像、类目、作者、质量分、探索机制等额外设计。

### 19.5 Average pooling 丢失序列结构

简单平均 watch/search embedding 稳定高效，但会丢失顺序、时间间隔、session 内上下文和目标 item 相关性。现代系统常用 attention、序列模型或多兴趣召回来补足。

### 19.6 2016 架构不等于今天的 YouTube

这篇论文是 2016 年公开快照，不代表今天 YouTube 的推荐系统。今天的系统必然更复杂，包括更强的序列建模、多目标、安全约束、创作者生态、实时反馈和更重的基础设施。

## 20. 推荐阅读路径

如果只读论文原文，建议按下面顺序：

1. Abstract: 把握两阶段问题拆分。
2. Fig. 2 + Section 2: 理解 candidate generation 和 ranking 的系统边界。
3. Section 3.1: 理解 recommendation as classification 和 sampled softmax。
4. Fig. 3: 看 training softmax 与 serving ANN 的差异。
5. Fig. 4: 重点读 example age 如何处理 freshness。
6. Fig. 5: 重点读 label/context selection 和 rollback。
7. Fig. 6: 看 features 与 depth 的离线效果。
8. Section 4.1: 读 ranking feature representation。
9. Section 4.2: 读 expected watch time 的 weighted logistic regression。
10. Table 1: 看 ranking 深度、宽度和 CPU budget 的取舍。

延伸阅读可以按主题分：

| 主题 | 推荐材料 |
| --- | --- |
| YouTube 早期推荐 | Davidson et al., The YouTube Video Recommendation System, RecSys 2010 |
| 大规模排序 | WSABIE, Weston et al., IJCAI 2011 |
| 隐式反馈 | Oard and Kim 1998; Hu, Koren, Volinsky 2008 |
| Dwell time | Yi et al., Beyond Clicks, RecSys 2014 |
| 大词表训练 | Jean et al., candidate sampling for large vocabulary |
| 双塔召回 | YouTube DNN candidate generation 后续工业实践 |
| Wide & Deep | Cheng et al., 2016 |
| 特征交叉 | DeepFM、DCN、xDeepFM |
| 序列推荐 | DIN、DIEN、SASRec、BERT4Rec |
| 冷启动 | DropoutNet、content-based retrieval、bandit exploration |

## 21. 结论

Deep Neural Networks for YouTube Recommendations 的长期价值，不在于“用 ReLU MLP 替代矩阵分解”这个表层结论，而在于它把推荐系统拆成了可以工程化迭代的完整链路。

它告诉我们：

1. 超大规模推荐必须先做候选生成，再做排序。
2. Candidate generation 可以写成 extreme multiclass classification，训练用 sampled softmax，服务用 ANN。
3. Ranking 面对的是 impression，不是抽象用户-视频对，因此需要更丰富的上下文和候选源特征。
4. 样本构造是核心。Rollback future watch 比随机 held-out 更接近线上场景。
5. Freshness 需要显式建模，example age 是一个经典方法。
6. CTR 不是足够好的目标，watch time 加权 logistic regression 是当时 YouTube 抑制 clickbait 的有效工程选择。
7. 离线指标只是迭代工具，最终仍要靠线上 A/B 和产品指标验证。

如果把这篇论文用于今天的工程实践，最重要的不是照抄网络结构，而是继承它的问题分解方式：先把召回、排序、样本、日志、目标函数、索引、A/B 和反馈闭环分清楚，再决定模型复杂度。

一个简单推荐系统也应该从这个原则出发。不要一开始就幻想一个模型解决全部问题；先让系统能展示、能记录、能 join、能训练、能回滚、能 A/B。模型会不断变化，但这个闭环才是推荐系统真正的基础设施。

## 参考资料

1. Paul Covington, Jay Adams, Emre Sargin. [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190). RecSys 2016.
2. Google Research. [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/).
3. Google Research PDF. [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/archive/45530.pdf).
4. James Davidson et al. The YouTube Video Recommendation System. RecSys 2010.
5. Jason Weston, Samy Bengio, Nicolas Usunier. WSABIE: Scaling Up To Large Vocabulary Image Annotation. IJCAI 2011.
6. Yi et al. Beyond Clicks: Dwell Time for Personalization. RecSys 2014.
7. Steffen Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
8. Yifan Hu, Yehuda Koren, Chris Volinsky. Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008.

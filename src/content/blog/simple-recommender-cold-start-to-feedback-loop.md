---
title: "简单推荐算法实施指南：从冷启动弱监督到真实反馈闭环"
description: "系统梳理推荐系统在缺少样本时如何通过内容先验、规则蒸馏、受控探索、曝光日志和双塔召回逐步建立稳定训练闭环"
pubDate: 2026-06-26
updatedDate: 2026-06-26
tags:
  - "Recommendation System"
  - "Cold Start"
  - "Two-Tower"
  - "ANN"
  - "Machine Learning"
  - "Engineering"
draft: false
---

很多团队第一次做推荐系统时，会把问题描述成“我要实现一个简单推荐算法”。但真实工程里，最难的通常不是写出一个相似度公式，也不是训练一个双塔模型，而是让系统从 **没有可靠样本** 的状态开始，逐步形成：

```text
曝光
  -> 行为
  -> 样本
  -> 训练
  -> 向量库
  -> 召回
  -> 再曝光
```

这条闭环。

本文讨论的不是电商、内容流、广告或 3D 资产平台里的某一个私有系统，而是一个公开泛化的工程路线：当你只有商品库、规则排序、少量用户画像，甚至还没有曝光日志时，如何启动一个简单但可持续演进的推荐系统。文章会以商品推荐为主例，核心实体写作 `item` 或 `SKU`，但方法同样适用于内容、视频、课程、素材、模型资产等推荐场景。

一句话概括本文观点：

> 冷启动阶段不能假装已经有可靠偏好监督。正确做法是用内容和规则弱监督初始化召回空间，用低权重 ANN 和受控探索制造可学习曝光，用严格日志把 unknown 逐步变成 observed，最后让真实反馈成为主监督。

这句话里有几个关键词：**弱监督**、**曝光**、**unknown**、**observed**、**主监督**。如果把它们混在一起，推荐系统很容易走偏。例如，把没有展示过的商品当作用户不喜欢的负样本；把规则系统蒸馏当成真实效果提升；或者第一版双塔一上线就替代规则召回。本文会尽量把这些边界讲清楚。

阅读本文时可以把它当成一份启动手册，而不是模型论文精读。第 1 到第 5 节先建立样本和冷启动边界；第 6 到第 13 节说明双塔、训练样本、ANN 和排序策略如何分工；第 14 到第 17 节讨论上线、探索、日志和阶段路线图；第 18 到第 19 节补充闭环后的重训、续训、发布和监控；最后两节是排错清单和最小落地配方。真正开工时，不建议跳过前面的日志和负样本边界直接看模型结构，因为后者只有在数据语义正确时才有价值。

如果只能选择一个最小起点，优先顺序应该是：规则 baseline、曝光日志、商品特征快照、ANN shadow、训练样本生成、最小双塔、低权重上线。这个顺序看起来保守，但它能让每一步都有可验证产物，也能让后续模型迭代建立在真实观测上，而不是建立在不可解释的伪标签上。

## 1. 为什么推荐系统不是先训练模型

推荐系统的第一个误区是：先拿一个模型训练起来。这个思路在监督学习任务里很自然，因为你通常已有一批明确的输入输出样本。但推荐系统，尤其是冷启动推荐，不是这样的。

推荐日志有一个特殊问题：用户只会对系统展示过的东西产生反馈。没有展示过的商品，用户没有点，不代表用户不喜欢；甚至用户完全不知道它存在。推荐数据天然是由旧策略选择展示什么之后产生的，这会带来 selection bias 和 exposure bias。Schnabel 等人在 “Recommendations as Treatments” 中讨论过这类推荐数据的选择偏差问题：训练和评估数据并不是从全量用户物品空间中均匀观测到的，而是由某种推荐或展示机制选择之后才被观测。

因此，推荐系统启动阶段最大的缺口不是模型，而是以下四件事：

| 缺口 | 具体表现 | 如果忽视会怎样 |
| --- | --- | --- |
| 曝光日志 | 不知道用户实际看到了什么 | 无法定义未点击、无法构造训练样本 |
| 负样本语义 | 不知道“不点击”是否代表不喜欢 | 把 unknown 当 negative，训练偏掉 |
| 策略版本 | 不知道样本来自哪个模型/规则/排序策略 | 后续无法解释模型效果变化 |
| 探索机制 | 系统只展示旧规则认为好的东西 | 新商品和长尾商品永远没有反馈 |

所以，一个简单推荐算法的第一阶段，不应该是“训练最强模型”，而应该是：

1. 有一个不会明显出错的 baseline 推荐策略。
2. 每次展示都写入可 join 的曝光日志。
3. 每个行为都能回连到当时的曝光、位置、候选来源和策略版本。
4. 能在低风险位置做受控探索。
5. 后续可以从日志里生成训练样本。

这也是本文反复强调的原则：**先启动闭环，再优化模型**。

## 2. 总体架构：商品单池召回

本文默认推荐主实体是商品或 SKU，召回池只有一个：商品向量库。也就是说，不先引入多子库路由，不把图文库、关键词库、品牌库、类目库拆成多个独立召回空间。商品内部的类目、品牌、标签、价格、热度、库存、文本 embedding 都是特征，不是单独的召回库。

整体链路可以先定成：

```text
用户请求
  -> 读取用户画像、近期行为、当前上下文
  -> 构造 UserFeatureV1
  -> UserTower / 规则画像生成 user_emb
  -> 在商品向量库 ANN TopN
  -> 读取 ItemFeatureV1
  -> 硬过滤、合并规则/热门/画像候选
  -> RankPolicyV1 统一排序
  -> 返回 TopK
  -> 写 ExposureLogV1
```

商品侧离线链路是：

```text
商品数据库
  -> ItemFeatureV1 快照
  -> ItemTower 离线推理
  -> ItemEmbeddingV1
  -> ANN index
  -> IndexManifest
```

这个结构有几个好处。

第一，语义清楚。线上目标就是 `user_emb -> item_emb TopK`，不需要先判断“去哪个子库”。如果商品有很多属性，它们进入 item tower，而不是成为路由分支。

第二，工程闭环短。第一版可以先用 deterministic embedding 或内容 embedding 写入 Qdrant、Milvus、FAISS/HNSWLib 之类的向量索引，验证召回链路，再替换成真实双塔 embedding。

第三，容易回滚。ANN 只是一路候选源，不直接接管最终排序。即使 ANN 结果不稳定，规则、热门、画像召回仍然可以兜底。

为了保持简单，本文默认向量维度为 `128`，打分为点积或余弦相似度：

$$
score(u,i)=u_{emb}^{\top}i_{emb}.
$$

如果使用 L2-normalized embedding，点积和 cosine similarity 在排序上等价：

$$
\cos(u,i)=
\frac{u_{emb}^{\top}i_{emb}}
{\|u_{emb}\|_2\|i_{emb}\|_2}.
$$

这些公式本身很简单。难点在于：`u_{emb}` 和 `i_{emb}` 应该从哪些可信信号里学出来。

### 2.1 推荐系统的五层边界

为了让工程实现不混乱，建议从第一天就把推荐系统拆成五层。

| 层 | 主要职责 | 不应该做什么 |
| --- | --- | --- |
| Candidate source | 产生候选，例如 ANN、热门、画像标签、规则候选 | 不直接决定最终排序 |
| Guardrail | 下架、删除、库存、黑名单、举报、合规过滤 | 不用模型分数覆盖硬规则 |
| Rank policy | 融合多路候选和多种分数 | 不把所有逻辑写死在模型里 |
| Logging | 记录曝光、行为、版本、来源、探索策略 | 不只记录最终 response |
| Training | 从日志和快照构造样本、训练、导出 embedding | 不直接读取线上实时可变状态 |

很多推荐系统早期混乱，根源是边界错位。例如让双塔模型同时承担召回、业务规则、库存过滤、已曝光惩罚和最终排序。这样看似简单，实际上会让模型训练目标不清楚，也会让线上回滚非常困难。

更稳妥的做法是让每层产出明确的中间对象：

```text
Candidate:
  item_id
  source
  source_score
  source_rank
  model_version

RankedItem:
  item_id
  final_score
  score_breakdown
  rank_policy_version

Exposure:
  request_id
  item_id
  rank
  source list
  score list
  model/index/policy version
```

这样后续任何问题都可以倒查：某个商品为什么出现，是哪一路召回来的，经过了哪些过滤，最终排序分数由哪些部分组成，用户看过之后发生了什么。

### 2.2 第一版不要追求完整智能，只追求闭环可验

一个能启动真实工作的 MVP，不需要一开始就具备完整个性化。它应该先满足下面这些条件：

1. 请求能返回 TopK 商品。
2. 返回商品都合法、在线、可展示。
3. 每个商品能解释候选来源和分数组成。
4. 每次展示都有曝光日志。
5. 用户行为能通过 `request_id + item_id` 回连曝光。
6. 能从日志里生成训练样本。
7. 能离线导出 item embedding 并构建 ANN index。
8. ANN 可以 shadow 或低权重接入。
9. 有 fallback 和 kill switch。

这九条看起来偏工程，但它们比第一版模型结构更重要。模型以后可以换，从规则到双塔，从双塔到多兴趣召回，从 MLP 到 Transformer，都可以逐步替换。日志、样本和版本闭环如果没建好，后面所有模型实验都没有可信依据。

## 3. 三类冷启动必须分开

“冷启动”这个词经常被混用。实际上至少有三类问题。

### 3.1 系统冷启动

系统冷启动是指推荐系统刚开始运行，没有可靠曝光日志，没有真实点击、收藏、加购、购买、隐藏、举报等反馈。此时你不能指望用真实 reward 训练模型，因为你还没有观测到 reward。

系统冷启动的目标不是立刻做出很强个性化，而是建立：

- 商品特征快照。
- 规则或热门 baseline。
- 曝光日志。
- 行为日志。
- 版本化策略。
- 受控探索。
- 训练样本生成链路。

这时第一版模型最多是一个不随机的初始化模型。它应该做到内容相关、类目合理、价格区间不离谱、新商品可被召回，但不能承诺已经真正理解用户偏好。

### 3.2 用户冷启动

用户冷启动是指新用户没有历史行为，或者匿名用户只有 session 上下文。此时 user tower 不能主要依赖 `user_id_embedding`，因为这个 embedding 没有足够行为支持。

可用信号包括：

- 当前场景，例如首页、详情页、搜索后推荐、相关推荐。
- 设备、地区、时间、星期。
- onboarding 选择的偏好。
- 默认人群 segment。
- session 内的点击、停留、跳过、隐藏。
- 热门和高质量候选兜底。

用户冷启动的目标是：不犯错、不过窄、能快速吸收 session feedback。

### 3.3 商品冷启动

商品冷启动是指新商品没有 CTR、CVR、销量、收藏、购买等历史统计，也没有被足够曝光。此时 item tower 不能主要依赖 `sku_id_embedding` 或历史热度。

可用信号包括：

- 类目、品牌、店铺。
- 标签、属性、材质、颜色、风格、房间、用途。
- 标题、搜索文本、描述 embedding。
- 图片/视频 embedding。
- 价格、折扣、库存、上下架状态。
- 同 SPU、同模型、同系列关系。

商品冷启动的关键是：内容 embedding 可用，新商品有探索预算，曝光和反馈能回流。

三类冷启动需要不同策略。系统冷启动重点是日志和 baseline；用户冷启动重点是上下文和 session；商品冷启动重点是内容特征和探索曝光。把它们混成一个“冷启动模型”会让文章和工程都变得模糊。

## 4. 监督信号可信度分层

在缺少样本时，最重要的是给信号分层。不是所有能进入训练的东西都是同等可信的标签。

| 层级 | 信号来源 | 能学到什么 | 可信度 | 主要用途 |
| --- | --- | --- | --- | --- |
| L0 | 商品状态、上下架、删除、库存、价格、类目、品牌、属性 | 商品事实和过滤条件 | 高 | guardrail、item feature |
| L1 | purchase、cart、like、favorite、hide、dislike、report，且有曝光上下文 | 用户对已展示 item 的明确响应 | 高 | 真实正负标签 |
| L2 | click、long dwell、detail view，且有曝光上下文 | 兴趣、注意力或弱意图 | 中高 | 正样本、排序目标 |
| L3 | 曝光未点击、曝光无动作，且曝光可验证、窗口结束 | 弱负反馈 | 中低 | 低权重负样本 |
| L4 | 热度、销量、CTR、CVR、运营权重 | 群体偏好或业务偏好 | 中低 | rank prior、fallback |
| L5 | 同 SPU、同模型、内容相似、文本/图片相似 | item-item 语义邻近 | 中 | item tower 预训练 |
| L6 | 从商品属性反推 synthetic user profile | 画像字段和商品字段兼容 | 低 | user tower 初始化 |
| L7 | 规则系统输出、规则排序分数 | 当前策略的近似 | 低到中 | 蒸馏、规则压缩 |

这张表里有三个关键边界。

第一，L0 很可靠，但不是用户偏好。`category=sofa`、`price=599`、`status=online` 都是事实，不能推出“用户喜欢这个商品”。它们适合做过滤、特征、内容空间建模。

第二，L1/L2/L3 必须有曝光上下文。一个用户购买了某商品当然是强信号，但如果无法知道它来自哪个场景、哪个 rank、哪个模型版本和哪个策略版本，那么后续训练和归因都会变差。

第三，L6/L7 可以训练模型，但不能证明用户喜欢。Synthetic user 和规则蒸馏是冷启动初始化手段，不是真实反馈替代品。

Hu、Koren、Volinsky 在 implicit feedback 经典工作中把隐式行为看成 preference signal 加 confidence，而不是干净的正负标签。这个视角对工程很有用：推荐日志里的行为有强弱置信度之分，未行为不等于明确负反馈。

## 5. 负样本边界：未曝光 item 不是负样本

推荐系统里最常见也最危险的错误，是把“用户没点某个商品”直接当成负样本。

错误逻辑是：

```text
用户没有点击 item_x
  -> item_x 是负样本
```

正确逻辑是：

```text
用户是否看到过 item_x？
  否 -> unknown
  是 -> 等待窗口内是否有动作？
       有正向动作 -> positive
       有明确负向动作 -> explicit negative
       无动作 -> weak negative
```

未曝光 item 只能作为 sampled negative、in-batch negative 或 contrastive denominator。它在语义上是 unknown，不是 negative。

### 5.1 随机负样本

随机负样本从全量商品里采样，主要作用是提供 contrastive loss 的分母，避免 embedding 坍缩。它可以帮助模型做粗粒度区分，例如沙发和灯具、耳机和衣服。

但随机负样本不表示用户不喜欢。很多随机采到的 item 只是未观察。它应该低权重使用，并过滤明显可能是正样本的情况，例如同 SPU、用户曾经正反馈过、属性高度相似。

### 5.2 同类 hard negative

同类 hard negative 更有训练价值，因为它让模型学习细粒度边界。

例如：

```text
用户画像：sofa + nordic + living_room + fabric
正样本：北欧布艺沙发
hard negative：工业风皮质沙发
```

但 hard negative 的风险也更高。同类目但不同属性，不代表用户不喜欢；相似 SKU 可能是替代品，也可能是潜在正样本。它适合作为低到中权重的边界样本，而不是强负反馈。

### 5.3 曝光未点击

曝光未点击是最接近真实负样本的数据，但仍然只是弱负。

成立条件至少包括：

1. 商品确实进入用户可见区域。
2. 记录了 `request_id`、`rank`、`position`、`scene`。
3. 等待窗口已经结束。
4. 窗口内没有 click、like、cart、purchase、hide、dislike 等行为。
5. 去重后不是重复曝光噪声。

即使满足这些条件，也不能把曝光未点击和 hide/dislike 等明确负反馈等价。用户可能没注意到，可能当时没有需求，也可能位置太低。

一个务实权重示例是：

| 行为 | 示例权重 | 解释 |
| --- | ---: | --- |
| purchase | `5.0` | 强正反馈 |
| cart | `4.0` | 强正反馈 |
| like/favorite | `3.0` | 明确正反馈 |
| click | `1.0` | 中等正反馈 |
| long dwell | `1.5` | 中高正反馈 |
| exposed_no_action | `0.1-0.3` | 弱负反馈 |
| hide/dislike/report | `4.0-5.0` | 明确负反馈或安全信号 |

### 5.4 推荐负样本策略

负样本策略应该随阶段演进：

| 阶段 | 负样本来源 | 语义 |
| --- | --- | --- |
| 0 样本 | item-item contrastive random negatives、同类 hard negatives | 内容空间分离，不是用户负反馈 |
| 少量样本 | explicit negative、exposed_no_action 低权重、in-batch negative | 开始引入用户反馈 |
| 可训练样本 | exposed_no_action、上一版 ANN/ranker 召回未点击、hard negative | 学真实排序边界 |
| 稳定闭环 | explicit negative、exposed_no_action、ranker hard negative、in-batch negative | 真实反馈主导 |

这里的底线是：**未曝光 item 不能当真实负样本**。如果博客只讲清这一个点，也能避免很多推荐系统初期训练错误。

## 6. Item Tower：先学商品内容空间

在系统冷启动和商品冷启动阶段，item tower 比 user tower 更容易先做起来，因为商品元数据通常已经存在。你可以先学习一个相对稳定的商品内容空间。

第一版 `ItemFeatureV1` 可以包含：

```json
{
  "schema_version": "item_feature_v1",
  "item_id": "sku:123",
  "sku_id": 123,
  "spu_id": 1001,
  "category_l1": "furniture",
  "category_l2": "sofa",
  "category_l3": "single_sofa",
  "brand_id": "brand_88",
  "shop_id": "shop_1024",
  "tag_ids": ["nordic", "living_room", "fabric"],
  "attr_value_ids": [
    "style=nordic",
    "room=living_room",
    "material=fabric",
    "color=white"
  ],
  "price": 599,
  "discount_rate": 0.82,
  "stock_status": "in_stock",
  "publish_age_hours": 72,
  "title_embedding": [0.01, -0.02],
  "image_embedding": [0.03, 0.04],
  "ctr_7d": 0.0,
  "ctr_7d_missing": true,
  "sales_7d": 0,
  "sales_7d_missing": true
}
```

注意两个细节。

第一，动态属性统一写成 `key=value` token，例如 `style=nordic`、`room=living_room`、`material=fabric`。这样不同类目的商品都能进入统一模型，不需要为每个类目定制结构。

第二，历史统计缺失要加 missing indicator。`ctr_7d=0` 可能表示商品曝光很多但没人点，也可能表示新商品没有曝光。没有 missing indicator，模型会把“没机会表现”和“表现差”混在一起。

### 6.1 内容预训练的正样本

item-item 内容预训练可以用以下正样本：

- 同一个 SKU 的两个增强视图。
- 同 SPU 下不同 SKU。
- 同 model 或同系列商品。
- 同类目且 tag/attribute 高重合。
- 文本/图片 embedding 接近的商品。

同 SKU 的增强视图可以通过 feature dropout 构造：

```text
view_1 = 去掉一部分 tag / attr / title token
view_2 = 去掉另一部分 tag / attr / image feature
```

训练目标是让两个 view 的 embedding 靠近。

### 6.2 内容预训练的负样本

负样本可以来自：

- 不同一级类目。
- 不同核心使用场景。
- 不同风格。
- 不同关键属性。
- 随机商品。

但同 SPU、同 model、高属性重合的商品不应作为强负样本，因为它们很可能是替代品或同系列商品。

### 6.3 Item Tower 学到了什么

item tower 内容预训练学到的是：

- 商品之间的语义邻近关系。
- 类目、标签、属性、文本、图片之间的融合。
- 对字段缺失和噪声的鲁棒性。
- 新商品的内容表达能力。

它没有学到：

- 某个用户喜欢什么。
- 某个场景下应该推什么。
- 点击率、购买意图、位置偏差。
- 业务目标。

所以它回答的是“这个商品和哪些商品内容相似”，不是“这个用户现在最可能喜欢哪个商品”。

### 6.4 商品特征质量检查比模型结构更早

在训练 item tower 之前，要先检查商品特征本身。很多“模型召回差”的问题，其实是商品特征脏、缺、乱。

建议至少做这些检查：

| 检查项 | 目的 | 异常信号 |
| --- | --- | --- |
| `item_id` 唯一性 | 确保推荐主键稳定 | 同一个商品多个 ID，或 ID 随快照变化 |
| 上下架状态 | 保证硬过滤可靠 | offline/deleted 商品进入训练正样本 |
| 类目覆盖 | 确认类目树可用 | 大量商品类目为空或落在 unknown |
| tag/attr 缺失率 | 确认可学习字段可用 | 新商品大面积缺 tag/attr |
| price 分布 | 防止极端值污染 | 价格为 0、负数、异常大值 |
| 文本 embedding 覆盖 | 确认语义特征可用 | 标题为空或 embedding 缺失 |
| 历史统计 missing indicator | 区分新商品和低表现商品 | 新商品被误认为 CTR/CVR 为 0 的差商品 |

商品特征检查要输出报告，而不是只在训练脚本里报错。原因是推荐系统通常每天或每小时刷新商品快照，特征质量会随上游数据变化而漂移。今天类目覆盖正常，不代表明天新接入的商品也是正常的。

可以固定生成一份 `item_feature_quality_report`：

```json
{
  "snapshot_id": "item_feature_snapshot_20260626_001",
  "row_count": 123456,
  "online_item_count": 118000,
  "unknown_category_rate": 0.003,
  "missing_tag_rate": 0.081,
  "missing_text_embedding_rate": 0.012,
  "price_p99": 2999,
  "offline_in_training_positive_count": 0,
  "created_at_ms": 1760000000000
}
```

这份报告可以作为训练前置门槛。例如 `offline_in_training_positive_count > 0` 时直接阻断训练，`missing_text_embedding_rate` 突然上升时暂停发布新 embedding。这样做比训练完再看指标更省时间。

还要做近邻检查。每次导出 item embedding 后，随机抽一些商品，查看它们的 TopK 近邻是否合理。近邻检查不需要代替线上指标，但能快速发现 embedding 空间是否崩坏。例如一个沙发的近邻全是灯具，通常说明类目或文本特征处理出错；一个新商品完全没有内容相似邻居，可能是属性缺失或 embedding 没写入。

## 7. User Tower：不要把 user_id 当冷启动支柱

用户侧更难，因为用户偏好来自行为，而冷启动恰恰缺行为。第一版 user tower 应该把 `user_id_embedding` 视作可选低权重特征，而不是核心。

一个公开泛化的 `UserFeatureV1` 可以是：

```json
{
  "schema_version": "user_feature_v1",
  "user_id": "u_456",
  "segment_id": "new_user_default",
  "context": {
    "scene": "home_feed",
    "device": "ios",
    "hour_bucket": 21,
    "weekday": 5,
    "city_id": "330100"
  },
  "long_term": {
    "category_ids": ["sofa", "chair"],
    "tag_ids": ["nordic", "living_room"],
    "style_ids": ["nordic"],
    "price_bucket": "500_800"
  },
  "short_term": {
    "positive_items": [
      {
        "item_id": "sku:123",
        "action": "click",
        "dwell_seconds": 20,
        "delta_hours": 1.5
      }
    ],
    "negative_items": [
      {
        "item_id": "sku:789",
        "action": "hide",
        "delta_hours": 2.0
      }
    ]
  },
  "dense_stats": {
    "active_days_30d": 3,
    "avg_click_price_30d": 580,
    "user_ctr_7d": 0.08
  }
}
```

行为序列可以先不用复杂 Transformer，第一版用加权池化即可。每个行为 token 可以拼接：

```text
item embedding
action type embedding
time bucket embedding
dwell feature
price bucket embedding
```

时间衰减权重可以写成：

$$
w_i=a_{type(i)}\exp(-\Delta t_i/\tau),
$$

其中 $a_{type(i)}$ 是行为类型权重，$\Delta t_i$ 是距离当前请求的时间差，$\tau$ 是衰减时间常数。

正反馈和负反馈可以分别池化：

$$
p_u=
\frac{\sum_{i \in \mathcal{P}_u} w_i e_i}
{\sum_{i \in \mathcal{P}_u} w_i+\epsilon},
\quad
n_u=
\frac{\sum_{i \in \mathcal{N}_u} w_i e_i}
{\sum_{i \in \mathcal{N}_u} w_i+\epsilon}.
$$

不要简单做：

```text
user_emb = positive_emb - negative_emb
```

更稳妥的是让 MLP 学融合：

```text
user_emb = MLP(concat(
  profile_emb,
  context_emb,
  positive_behavior_emb,
  negative_behavior_emb,
  dense_user_features
))
```

对于新用户，`positive_behavior_emb` 和 `negative_behavior_emb` 可以是 zero/default，`profile_emb` 来自默认 segment 或 onboarding，`context_emb` 来自真实请求。

### 7.1 第一版双塔结构可以很朴素

第一版不要急着上复杂序列模型。一个足够可落地的结构是：

```text
ItemTower:
  item_id_emb, optional with dropout
  category_emb
  brand_emb
  tag_embedding_bag
  attr_embedding_bag
  price_bucket_emb
  dense_item_mlp
  title_text_proj
  image_proj, optional
  -> concat
  -> fusion_mlp
  -> L2Norm
  -> item_emb R128

UserTower:
  user_id_emb, optional / default for cold users
  segment_emb
  context_emb
  long_term_profile_bag
  positive_behavior_pooling
  negative_behavior_pooling
  dense_user_mlp
  -> concat
  -> fusion_mlp
  -> L2Norm
  -> user_emb R128
```

其中 `item_id_emb` 和 `user_id_emb` 都要谨慎。它们在数据充分时有用，但冷启动阶段很容易让模型记住老商品、老用户，而不是学习可泛化的内容和画像关系。可用的工程手段包括：

- 降低 ID embedding 维度。
- 训练时对 ID embedding 做 feature dropout。
- 新用户、新商品使用 default/unknown embedding。
- 单独监控新用户、新商品分桶效果。

对于数值特征，不要直接裸喂：

| 原始特征 | 建议处理 |
| --- | --- |
| price | `log1p(price)` 或 price bucket |
| sales_7d | `log1p(sales_7d)` + missing indicator |
| ctr_7d/cvr_7d | clip + standardize + missing indicator |
| publish_age_hours | `log1p(age_hours)` 或 freshness bucket |
| dwell_seconds | clip + bucket + standardize |
| active_days_30d | clip + normalize |

数值特征的关键不是追求复杂，而是避免尺度失控和缺失语义混淆。新商品 `ctr_7d=0` 与老商品 `ctr_7d=0` 的含义不同，必须通过 missing indicator 或 exposure count 区分。

### 7.2 高频行为不一定每次请求全量回放

如果用户行为频繁，线上每次请求都从最近几百条行为重新编码会增加延迟和存储压力。第一版可以维护两个实时兴趣向量：

```text
user_pos_interest_vec
user_neg_interest_vec
```

每次发生行为时增量更新：

$$
p_u^{new}
=
\lambda p_u^{old}
+a_{type} i_{emb},
$$

其中 $\lambda$ 是衰减系数，$a_{type}$ 是行为权重，$i_{emb}$ 是被操作商品的 embedding。

负反馈向量同理：

$$
n_u^{new}
=
\lambda n_u^{old}
+b_{type} i_{emb}.
$$

线上 user tower 可以直接吃这两个实时向量，再融合画像和上下文：

```text
user_emb = MLP(concat(
  profile_emb,
  context_emb,
  user_pos_interest_vec,
  user_neg_interest_vec,
  dense_user_features
))
```

这种做法牺牲了一部分序列细节，但能显著降低在线计算复杂度。后续如果系统稳定，再把简单池化替换为 DIN、SASRec 或 Transformer 类行为编码器，也不会破坏前面的日志和样本契约。

## 8. Synthetic User-SKU Matching 的真实作用

没有真实用户行为时，可以从商品反推伪用户画像。这个方法有用，但必须说清楚它到底学到了什么。

假设一个商品是：

```json
{
  "item_id": "sku:123",
  "category": "sofa",
  "style": "nordic",
  "room": "living_room",
  "material": "fabric",
  "color": "white",
  "price_bucket": "500_800"
}
```

可以构造一个 synthetic user：

```json
{
  "schema_version": "user_feature_v1",
  "user_id": "synthetic",
  "long_term": {
    "category_ids": ["sofa"],
    "style_ids": ["nordic"],
    "room_ids": ["living_room"],
    "material_ids": ["fabric"],
    "color_ids": ["white"],
    "price_bucket": "500_800"
  },
  "short_term": {
    "positive_items": [],
    "negative_items": []
  },
  "context": {
    "scene": "home_feed"
  }
}
```

然后训练：

```text
synthetic user profile -> matching SKU
```

这一步学到的是字段兼容性：喜欢北欧风、客厅、沙发、白色、布艺的人，内容上应该匹配北欧白色布艺沙发。

它没有学到真实用户是否会点击，也没有学到价格、品牌、图片质量、库存、热度在真实行为中的权重。它的风险是把推荐退化为过硬的属性匹配：

```text
画像有 nordic -> 只推 nordic
画像有 sofa -> 只推 sofa
```

所以 synthetic 样本必须低权重，并标注来源：

```json
{
  "source": "synthetic_profile_match",
  "confidence_weight": 0.2
}
```

这类样本适合帮助 user tower 和 item tower 对齐到同一个向量空间，但不应该和真实 click、cart、purchase 平权。

## 9. 规则系统蒸馏的价值和风险

很多团队在做模型前已经有规则推荐：热门、类目匹配、标签匹配、价格偏好、业务加权、已曝光惩罚。这套系统虽然不是深度模型，但包含工程经验，可以作为 teacher。

蒸馏过程可以写成：

```text
UserFeature / profile / context
  -> 当前 rule ranker
  -> teacher scored item list
  -> two-tower 学习 teacher 排序偏好
```

训练方式可以是：

- top score item 作为 soft positive。
- low score item 作为 soft negative。
- pairwise ranking distillation。
- listwise KL distillation。

如果使用 softmax 蒸馏：

$$
p_i^{teacher}
=
\frac{\exp(s_i^{teacher}/T)}
{\sum_j \exp(s_j^{teacher}/T)},
\quad
p_i^{student}
=
\frac{\exp(s_i^{student}/T)}
{\sum_j \exp(s_j^{student}/T)}.
$$

蒸馏 loss 可以写成：

$$
\mathcal{L}_{distill}
=
\mathrm{KL}(p^{teacher}\|p^{student}).
$$

这里的温度 $T$ 很重要。$T$ 太小会让模型只学习 top item，放大规则系统的热门偏置；$T$ 稍大一些能让 near-boundary candidates 也参与学习。

### 9.1 蒸馏会复制偏见

规则蒸馏默认会复制：

- 热门商品偏置。
- 旧规则权重偏置。
- 候选池覆盖不足。
- 新商品低曝光问题。
- 类目或业务规则的主观偏好。

所以蒸馏不是效果提升本身，而是 warmup 和规则压缩。它的成功标准不是“student 和 teacher 完全一致”，而是“student 在保留基本相关性的同时，能提供更好的覆盖和后续学习入口”。

### 9.2 采样不能只取 teacher topK

蒸馏样本应该分桶采样：

| 桶 | 作用 |
| --- | --- |
| top bucket | 学 teacher 明确认为好的候选 |
| middle bucket | 学排序中段差异 |
| near-boundary bucket | 学决策边界 |
| low bucket | 学明显不合适候选 |
| new item bucket | 避免新商品完全无训练机会 |
| long-tail bucket | 防止模型只学热门 |

如果只取 teacher topK，模型会学成“热门规则系统的向量化版本”。

### 9.3 蒸馏必须有退出机制

当真实曝光和反馈稳定后，蒸馏权重要下降。退出条件可以包括：

- 曝光日志覆盖稳定。
- `exposure -> interaction` join rate 稳定。
- 每个主要 scene 有足够正负样本。
- ANN 分数分桶和真实反馈呈单调关系。
- A/B 不劣于规则 baseline。
- hide/dislike/report 没有明显上升。

满足这些条件后，蒸馏最多保留为冷启动 regularizer 或新模型 warmup，不应该继续和真实反馈平权。

## 10. Loss 设计和权重退场机制

双塔召回常用 in-batch softmax。一个 batch 里有 $B$ 个用户和 $B$ 个正样本商品：

$$
U=[u_1,\dots,u_B]^\top,
\quad
V=[v_1,\dots,v_B]^\top.
$$

logits 为：

$$
Z=\frac{UV^\top}{\tau}.
$$

第 $b$ 个用户的正样本是第 $b$ 个 item，则 loss 为：

$$
\mathcal{L}_{inbatch}
=
-\frac{1}{B}
\sum_{b=1}^{B}
\log
\frac{\exp(u_b^\top v_b/\tau)}
{\sum_{j=1}^{B}\exp(u_b^\top v_j/\tau)}.
$$

冷启动阶段还需要混合多个目标：

$$
\mathcal{L}
=
w_{real}\mathcal{L}_{real}
+w_{noact}\mathcal{L}_{exposed\_no\_action}
+w_{syn}\mathcal{L}_{synthetic}
+w_{distill}\mathcal{L}_{distill}
+w_{item}\mathcal{L}_{item}.
$$

权重应该随着数据成熟度变化：

| 阶段 | `w_real` | `w_noact` | `w_syn` | `w_distill` | `w_item` | 含义 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 样本 | 0.00 | 0.00 | 0.35 | 0.35 | 0.30 | 只做初始化 |
| 少量样本 | 0.20 | 0.10 | 0.25 | 0.25 | 0.20 | 真实反馈小权重加入 |
| 可训练样本 | 0.50 | 0.20 | 0.10 | 0.10 | 0.10 | 真实反馈开始主导 |
| 稳定闭环 | 0.70+ | 0.15-0.25 | 0.00-0.05 | 0.00-0.05 | 0.05-0.10 | synthetic/distill 退出或辅助 |

这些数值不是定理，而是工程起点。真正上线后要根据样本量、分桶指标、A/B 和负反馈调整。

关键是有方向性：真实反馈越来越重，synthetic 和 distillation 越来越轻。否则系统会永远停留在规则系统和内容匹配的近似版本。

## 11. 训练样本契约

训练样本必须能回答两个问题：

1. 用户在曝光时刻看到了什么？
2. 模型训练时是否使用了曝光之后的信息？

一个示例 `TrainingExampleV1`：

```json
{
  "schema_version": "training_example_v1",
  "example_id": "ex_20260626_000001",
  "source": "real_click",
  "label": 1.0,
  "weight": 1.0,
  "user_snapshot_at_t": {
    "user_id": "u_456",
    "snapshot_time_ms": 1760000000000,
    "long_term": {
      "category_ids": ["sofa", "chair"],
      "tag_ids": ["nordic", "living_room"]
    },
    "short_term": {
      "positive_items": ["sku:123"],
      "negative_items": []
    },
    "context": {
      "scene": "home_feed",
      "device": "ios",
      "hour_bucket": 21
    }
  },
  "positive_item": {
    "item_id": "sku:789",
    "feature_snapshot_id": "item_feature_snapshot_20260626_001"
  },
  "negative_items": [
    {
      "item_id": "sku:456",
      "negative_type": "exposed_no_action",
      "weight": 0.2
    }
  ],
  "exposure_context": {
    "request_id": "req_001",
    "rank": 7,
    "position": "feed_slot_7",
    "candidate_source": "ann_two_tower",
    "is_exploration": false,
    "model_version": "two_tower_v1",
    "rank_policy_version": "rank_policy_v1"
  }
}
```

这里有几个硬要求。

第一，`user_snapshot_at_t` 只能包含曝光之前的信息。曝光之后的 click、cart、purchase 只能作为 label，不能进入用户特征。否则就是数据穿越。

第二，`source` 必须存在。`real_click`、`exposed_no_action`、`synthetic_profile_match`、`rule_distillation`、`item_contrastive` 这些样本的可信度完全不同，训练和评估必须分开看。

第三，`model_version` 和 `rank_policy_version` 必须记录。没有版本，后面无法知道样本来自哪个策略，也无法分析策略变化带来的样本分布变化。

### 11.1 快照和 manifest 必须版本化

训练样本不能直接从线上当前表里临时拼。推荐训练要可复现，至少需要三个 manifest。

`ItemFeatureSnapshotManifestV1`：

```json
{
  "schema_version": "item_feature_snapshot_manifest_v1",
  "snapshot_id": "item_feature_snapshot_20260626_001",
  "feature_schema_version": "item_feature_v1",
  "source": "warehouse_snapshot",
  "row_count": 123456,
  "path": "artifacts/item_features/dt=2026-06-26/items.v1.jsonl",
  "created_at_ms": 1760000000000
}
```

`TrainingDatasetManifestV1`：

```json
{
  "schema_version": "training_dataset_manifest_v1",
  "dataset_id": "two_tower_dataset_20260626_001",
  "example_schema_version": "training_example_v1",
  "item_feature_snapshot_id": "item_feature_snapshot_20260626_001",
  "user_feature_snapshot_id": "user_feature_snapshot_20260626_001",
  "row_count": 1000000,
  "positive_count": 220000,
  "negative_count": 780000,
  "time_range": {
    "start_ms": 1759000000000,
    "end_ms": 1760000000000
  }
}
```

`ANNIndexManifestV1`：

```json
{
  "schema_version": "ann_index_manifest_v1",
  "index_id": "ann_index_20260626_001",
  "embedding_schema_version": "item_embedding_v1",
  "model_version": "two_tower_v1",
  "item_feature_snapshot_id": "item_feature_snapshot_20260626_001",
  "vector_dim": 128,
  "metric": "cosine",
  "item_count": 123456,
  "created_at_ms": 1760000000000
}
```

这些 manifest 不是形式主义。它们解决的是训练事故排查问题：

- 某次模型效果变差，是商品特征变了，还是训练样本变了？
- 线上 ANN index 里有多少商品？
- 这个 index 用的是哪个模型版本？
- 某条曝光对应的 item embedding 是否还能复现？
- 某个训练样本有没有数据穿越？

如果没有 manifest，推荐系统会很快进入“谁也说不清楚这版模型怎么来的”的状态。

### 11.2 样本生成要做时间窗约束

一个常见错误是训练样本构造时直接读取用户当前画像。这样会把曝光之后的行为混入曝光之前的特征。

正确方式是：

```text
for each exposure at time t:
  user_snapshot = user features only before t
  item_snapshot = item features valid at t
  label = interactions after t within window
  negatives = exposed items with no action after window
```

不同事件可以使用不同窗口：

| 事件 | 建议窗口 | 说明 |
| --- | --- | --- |
| click | 分钟级到小时级 | 反馈快 |
| long dwell | 分钟级 | 依赖页面停留 |
| favorite/cart | 小时级到天级 | 行为可能滞后 |
| purchase | 天级 | 转化链路长 |
| exposed_no_action | 至少等待一个合理窗口 | 不能刚曝光就判负 |

窗口不是越长越好。窗口太短会漏掉延迟行为；窗口太长会把 unrelated 行为错误归因给曝光。第一版可以按业务场景保守设置，再通过数据分析调整。

### 11.3 样本要保留 source，而不是合并成一个 label

训练时可以把多种来源混合，但数据上不要把它们压扁成一个 `label=1/0`。至少保留：

```text
source = real_click
source = real_cart
source = real_purchase
source = explicit_negative
source = exposed_no_action
source = synthetic_profile_match
source = rule_distillation
source = item_contrastive
```

这样评估时可以分开看。例如模型在 synthetic 样本上表现很好，但在 real click 上没有提升，说明它只学会了属性匹配；模型在 teacher distillation 上很一致，但 long-tail coverage 下降，说明它复制并放大了规则偏置。

## 12. ANN 检索和在线链路

ANN 不是推荐系统的全部，它只负责从大规模商品库里快速取出一批候选。最终排序仍然应该由 rank policy 统一完成。

第一版向量库可以选择 Qdrant、Milvus、pgvector 或 FAISS/HNSWLib。工程上可以这样取舍：

| 选型 | 第一版适合度 | 说明 |
| --- | --- | --- |
| Qdrant | 高 | 部署和 API 相对直接，payload filter 友好，适合验证线上 ANN 链路 |
| Milvus | 中 | 更偏大规模向量平台，第一阶段可能偏重 |
| FAISS/HNSWLib | 中 | 适合离线 benchmark 或内嵌服务，不适合作为完整在线服务起点 |
| pgvector | 中 | 适合数据量小、已有 PostgreSQL 的场景 |
| OpenSearch/Elasticsearch vector | 中 | 适合已有搜索基础设施，但推荐向量服务能力要单独评估 |

在线推荐链路可以写成：

```text
RecommendRequest
  -> read profile store
  -> build UserFeatureV1
  -> UserTower inference
  -> ANN topN, e.g. 200-500
  -> ItemFeature lookup
  -> hard filters
  -> merge rule/profile/hot/business recalls
  -> dedup by item_id / spu_id
  -> RankPolicyV1 scoring
  -> response topK
  -> write ExposureLogV1
```

ANN 返回的候选应该带上：

```json
{
  "item_id": "sku:123",
  "ann_score": 0.73,
  "source": "ann_two_tower",
  "embedding_version": "item_embedding_v1",
  "ann_index_version": "ann_index_20260626_001",
  "rank": 12
}
```

这几个字段后面都会进入曝光日志。否则你只能知道用户看到了商品，无法知道它为什么被召回。

## 13. RankPolicyV1：不要让召回模型承担所有业务逻辑

双塔召回负责“找候选”，不应该承担所有业务逻辑。库存、下架、重复、已购买、已隐藏、业务加权、探索位、同 SPU 去重，都应该由排序和策略层控制。

一个示例 `RankPolicyV1`：

```json
{
  "schema_version": "rank_policy_v1",
  "policy_id": "default_rank_policy_v1",
  "weights": {
    "ann_score": 0.20,
    "profile_tag_score": 0.35,
    "hot_score": 0.25,
    "business_score": 0.10,
    "rule_score": 0.10
  },
  "penalties": {
    "seen": -0.35,
    "purchased": -0.50,
    "same_spu_duplicate": -0.20,
    "recent_hide": -1.00
  },
  "hard_filters": {
    "filter_deleted": true,
    "filter_offline": true,
    "filter_unsellable": true,
    "filter_blocked": true,
    "filter_reported": true
  },
  "limits": {
    "ann_top_n": 500,
    "response_top_k": 20,
    "max_same_spu": 1
  }
}
```

最终分数可以写成：

$$
s_{final}
=
\alpha s_{ann}
+\beta s_{profile}
+\gamma s_{hot}
+\delta s_{business}
+\eta s_{rule}
-p.
$$

其中 $p$ 是各种惩罚项。第一版 `ann_score` 权重不应太高，可以从 `0.10-0.20` 开始。等真实反馈证明 ANN 分数有价值，再逐步提高。

## 14. 上线策略：shadow 到低权重

弱监督模型上线最重要的是控制风险。推荐三步。

### 14.1 Shadow

Shadow 阶段线上计算 ANN 结果，但不影响最终推荐。

记录：

- ANN topN。
- ann_score。
- item valid rate。
- 与规则候选的 overlap。
- 新商品覆盖率。
- 长尾覆盖率。
- 被 hard filter 过滤的比例。

Shadow 阶段的目标不是提升 CTR，而是验证 ANN 链路和候选质量。

### 14.2 Low Weight Recall

第二阶段 ANN 作为一路候选源进入推荐，但排序权重低。例如：

```text
ann_score weight = 0.10 ~ 0.20
rule/profile/hot 仍然主导
```

此时必须有兜底：

- ANN timeout -> 规则/热门/画像召回。
- user tower 失败 -> default profile recall。
- 向量库失败 -> static/rule fallback。
- 候选数不足 -> hot/profile 补齐。

### 14.3 Controlled Ramp-up

只有当分桶指标证明 ANN 分数和真实反馈有正相关，且 A/B 没有增加负反馈，才逐步提高 ANN 权重：

```text
0.20 -> 0.35 -> 0.50+
```

同时保留 kill switch：

- 关闭 ANN recall。
- 关闭 exploration。
- 回滚 `model_version`。
- 回滚 `rank_policy_version`。

第一版模型默认应该可关闭。这样团队才敢上线试错。

## 15. 探索机制：冷启动必须主动制造可学习曝光

如果系统永远只展示规则系统认为最好的商品，就永远只能学到规则系统已经选择展示过的分布。新商品、长尾商品、边界候选没有曝光，也就没有反馈。

探索不是全随机。推荐：

```text
90%-95% exploitation:
  当前稳定 rank policy

5%-10% guided exploration:
  在 guardrail 过滤后的高质量候选中做小幅扰动
```

探索候选可以来自：

- 内容匹配但低曝光商品。
- 新商品。
- ANN top 100-300 中排名较低但质量合格的商品。
- 同类目不同风格商品。
- 规则系统低估但内容相关商品。

探索位置也要控制。不要一开始放在最核心坑位。可以：

```text
top 3 保守
rank 4-20 轻度探索
长列表中部更多探索
```

探索日志必须包含：

```json
{
  "is_exploration": true,
  "exploration_policy_id": "guided_exploration_v1",
  "logging_policy_id": "rank_policy_v1",
  "candidate_source": "ann_two_tower",
  "rank_before_exploration": 83,
  "rank_after_exploration": 12
}
```

如果没有这些字段，后续无法区分“正常排序带来的曝光”和“探索策略带来的曝光”。训练样本也会混在一起。

Contextual bandit 文献把推荐看作顺序决策问题：系统基于上下文选择动作，同时从反馈里学习。本文不建议第一版直接上复杂 bandit 算法，但受控探索和 logging policy 这两个思想必须尽早建立。

## 16. 曝光日志：比第一版模型更重要

第一天就应该写曝光日志。没有曝光日志，后面没有可靠负样本，也无法评估模型。

`ExposureLogV1` 至少包含：

```json
{
  "schema_version": "exposure_log_v1",
  "request_id": "req_001",
  "user_id": "u_456",
  "item_id": "sku:123",
  "scene": "home_feed",
  "page": 1,
  "rank": 7,
  "position": "feed_slot_7",
  "candidate_sources": ["ann_two_tower", "hot_recall"],
  "scores": {
    "ann_score": 0.73,
    "profile_tag_score": 0.41,
    "hot_score": 0.63,
    "business_score": 0.10,
    "rule_score": 0.35,
    "final_score": 0.58
  },
  "model_version": "two_tower_v1",
  "ann_index_version": "ann_index_20260626_001",
  "rank_policy_version": "rank_policy_v1",
  "is_exploration": false,
  "exposed_at_ms": 1760000000000
}
```

`InteractionEventV1` 至少包含：

```json
{
  "schema_version": "interaction_event_v1",
  "event_id": "evt_001",
  "request_id": "req_001",
  "user_id": "u_456",
  "item_id": "sku:123",
  "event_type": "click",
  "event_at_ms": 1760000002000,
  "dwell_ms": 18000
}
```

训练样本从以下 join 得来：

```text
ExposureLogV1
  + InteractionEventV1
  + UserFeatureSnapshot
  + ItemFeatureSnapshot
  -> TrainingExampleV1
```

监控里必须有：

- 曝光日志写入成功率。
- 行为日志写入成功率。
- `request_id` 缺失率。
- `exposure -> interaction` join rate。
- item feature lookup 成功率。
- model/index/policy version 覆盖率。

这些指标不达标时，先修数据闭环，不要急着调模型。

## 17. 四阶段路线图

下面是一条从 0 到稳定的执行路线。

### 17.1 阶段 0：0 样本

状态：

- 没有曝光日志。
- 没有真实点击、收藏、购买。
- 只有商品 metadata。
- 可能有规则系统。

训练目标：

```text
0.35 * synthetic user-SKU matching
0.35 * rule distillation
0.30 * item-item contrastive
```

如果没有规则系统：

```text
0.60 * synthetic user-SKU matching
0.40 * item-item contrastive
```

上线方式：

- ANN 只 shadow，或 `ann_score <= 0.10`。
- 规则/热门/画像召回主导。
- 先写曝光日志。

验收标准：

- `ItemFeatureV1` 稳定导出。
- `ItemEmbeddingV1` 可生成。
- ANN index 可构建。
- ANN 返回商品全部合法。
- embedding neighborhood 人工检查合理。
- 推荐服务能写 `ExposureLogV1`。

不要做：

- 不要宣称个性化已经可用。
- 不要把未曝光 item 当负样本。
- 不要让 ANN 接管排序。
- 不要优化 CTR，因为还没有可靠 CTR。

### 17.2 阶段 1：少量样本

状态：

- 曝光日志刚接入。
- 点击、收藏、购买很少。
- hide/dislike 也很少。
- 样本分布很不稳定。

训练目标：

```text
0.20 * real feedback
0.10 * exposed_no_action
0.25 * synthetic
0.25 * distillation
0.20 * item contrastive
```

上线方式：

- `ann_score = 0.10-0.20`。
- exploration `5%` 左右。
- rule/profile/hot 仍然主导。

验收标准：

- exposure join rate 达标。
- no-action negative 可稳定生成。
- explicit negative 能立即影响过滤/惩罚。
- ANN candidate valid rate 接近 `100%`。
- 下架、删除、不可售泄漏为 `0`。
- ANN treatment 不显著增加 hide/dislike/report。

不要做：

- 不要用少量 click 训练大模型。
- 不要让 exposed_no_action 权重大于 click。
- 不要只看整体 CTR。
- 不要把规则蒸馏当成真实效果提升。

### 17.3 阶段 2：可训练样本

状态：

- 曝光日志稳定。
- 主要 scene 有足够曝光。
- click/like/cart/purchase 形成规模。
- exposed_no_action 样本稳定。

训练目标：

```text
0.50 * real feedback
0.20 * exposed_no_action
0.10 * hard negative
0.10 * item contrastive
0.05 * synthetic
0.05 * distillation
```

上线方式：

- `ann_score = 0.25-0.40`。
- exploration `5%-10%`。
- rule/profile/hot 保留兜底。

验收标准：

- ANN 分数分桶和真实反馈正相关。
- A/B CTR 或 deeper action 不劣于 baseline。
- hide/dislike/report 不升高。
- new item coverage 提升。
- long-tail coverage 提升。
- 重复 SPU rate 可控。
- 主要 scene/category 没有明显劣化。

不要做：

- 不要完全移除规则召回。
- 不要用 teacher agreement 当核心成功指标。
- 不要忽视 position bias 和 exposure bias。

### 17.4 阶段 3：稳定闭环

状态：

```text
曝光 -> 行为 -> 样本 -> 训练 -> embedding -> ANN -> 曝光
```

闭环稳定，模型定期重训，日志和特征版本可回溯。

训练目标：

```text
0.70 * real feedback
0.15 * exposed_no_action
0.10 * hard negative
0.05 * item regularization
```

synthetic 和 distillation 可以退出，或者只保留在新模型 warmup、冷启动 segment、规则 fallback 对齐里。

上线方式：

- ANN 是重要召回源，但仍然不是最终 ranker。
- 排序继续融合画像、热门、业务规则、惩罚和 guardrail。
- 支持自动化重训、index 构建、shadow validation、灰度发布、回滚。

验收标准：

- 周期性训练可复现。
- `model_version`、`feature_version`、`index_version` 可追踪。
- 线上 A/B 稳定正向。
- score calibration 稳定。
- 长尾和新商品有曝光机会。
- 负反馈率稳定。
- fallback 和 kill switch 可用。

不要做：

- 不要让模型无限放大历史曝光偏置。
- 不要只优化短期 CTR。
- 不要忽视新商品探索。
- 不要删除 guardrail。

### 17.5 从第一周到第六周的开工节奏

如果把上面的阶段拆成真实工作安排，可以按下面推进。

**第 1 周：基线和日志优先。**

目标不是训练模型，而是让推荐请求和曝光日志成型。

交付物：

- `RecommendRequest` / `RecommendResponse` 示例。
- `ExposureLogV1` 和 `InteractionEventV1` JSON fixture。
- 规则、热门、画像 baseline。
- `RankPolicyV1` 最小权重配置。
- 本地或测试环境日志写入验证。

验收方式：

- 任意一次推荐 response 都能找到对应曝光日志。
- 任意一次点击都能通过 `request_id + item_id` 回连曝光。
- 下架、删除、不可售商品不会出现在 response。

**第 2 周：商品特征快照。**

目标是让商品库变成可训练、可索引的数据。

交付物：

- `ItemFeatureV1` JSONL。
- `ItemFeatureSnapshotManifestV1`。
- 商品特征质量检查脚本。
- 基础统计：类目分布、价格分布、缺失率、上下架比例。

验收方式：

- `item_id` 唯一且稳定。
- 核心字段缺失率可解释。
- 新商品有内容特征，不依赖 CTR/CVR。

**第 3 周：确定性 embedding 和 ANN shadow。**

目标是先验证 ANN 链路，不急着训练双塔。

可以先用 deterministic embedding：

```text
category/tag/attr/text feature
  -> fixed hashing / simple projection
  -> item_emb
  -> ANN index
```

交付物：

- `ItemEmbeddingV1` JSONL。
- `ANNIndexManifestV1`。
- ANN topN 查询接口。
- Shadow 日志：ANN candidates 不影响最终排序。

验收方式：

- ANN 查询延迟可接受。
- ANN 返回商品合法率接近 `100%`。
- 人工抽查 embedding 近邻内容相关。
- Shadow 结果可写入日志。

**第 4 周：训练样本生成。**

目标是从曝光和行为生成第一批可审计样本。

交付物：

- `TrainingExampleV1` JSONL。
- `TrainingDatasetManifestV1`。
- 样本 source 分布。
- 正负样本比例。
- 数据穿越检查。

验收方式：

- 每条真实样本能追溯到曝光。
- `user_snapshot_at_t` 不包含曝光后行为。
- no-action negative 都有等待窗口。
- 未曝光商品不作为真实负样本。

**第 5 周：最小双塔 baseline。**

目标是训练一个不复杂但可替换 deterministic embedding 的模型。

交付物：

- `ItemTower` 和 `UserTower` baseline。
- in-batch softmax 训练。
- synthetic/distill/item contrastive 混合 loss。
- model manifest。
- item embedding 导出。

验收方式：

- 离线 loss 正常下降。
- 近邻人工检查不崩。
- 冷启动商品可被召回。
- ANN shadow 分布不比 deterministic embedding 更差。

**第 6 周：低权重上线和 A/B。**

目标是把 ANN 当一路候选源，而不是直接接管。

交付物：

- `ann_score` 低权重进入 `RankPolicyV1`。
- Exploration policy。
- A/B 指标面板。
- kill switch。

验收方式：

- hide/dislike/report 不升高。
- 下架、删除、不可售泄漏为 `0`。
- ann_score 分桶与真实反馈有初步正相关。
- 低曝光新商品开始有受控曝光。

这六周不是固定周期，而是一个优先级排序。小团队可以压缩，大团队可以并行，但不要反过来：先做复杂模型、GPU 服务和大数据平台，再补曝光日志。

### 17.6 每次模型发布前的检查清单

每次发布新模型或新 ANN index 前，建议固定检查：

```text
[ ] 训练数据 manifest 已生成
[ ] item feature snapshot 可追溯
[ ] user snapshot 无数据穿越
[ ] positive / negative / source 分布正常
[ ] 新旧模型 embedding 维度一致
[ ] item embedding 覆盖率达标
[ ] ANN index item_count 与 embedding 行数一致
[ ] ANN valid rate 达标
[ ] 下架/删除/不可售泄漏为 0
[ ] 近邻人工 case review 通过
[ ] shadow 指标无明显异常
[ ] rank_policy_version 已更新
[ ] rollback model/index/policy 可用
```

这份清单比“训练 loss 下降”更重要。推荐系统的失败经常不是 loss 不下降，而是某个版本、快照、过滤或日志字段对不上。

## 18. 闭环后的重训、续训与模型发布

前面的路线图讲的是如何从 0 样本走到稳定闭环。但闭环稳定之后，推荐系统还会进入另一个阶段：**模型如何持续变新，又不把线上系统变得不可控**。

很多团队会把这个问题简化成“每天重训一次”。这个说法太粗。成熟推荐系统里的持续训练至少包含五个不同层次：

| 层次 | 更新对象 | 常见频率 | 目标 | 主要风险 |
| --- | --- | --- | --- | --- |
| 特征快照更新 | `ItemFeatureV1`、统计特征、用户画像 | 小时级到天级 | 让模型看到新商品、新价格、新库存、新行为 | 快照和训练样本时间不一致 |
| ANN index 更新 | item embedding、向量索引、过滤字段 | 天级或更高 | 让召回池包含最新商品和最新 item 表达 | user tower 和 item index 不匹配 |
| batch retraining | 双塔、召回模型、排序模型 | 天级、周级 | 用最近闭环数据修正模型 | 历史偏置放大、窗口选择错误 |
| warm-start / continued training | 从上一版模型继续训练 | 小时级到天级 | 降低训练成本，保留已有 embedding 空间 | 继承旧模型错误，遗忘长尾 |
| nearline / online update | 高频用户兴趣、部分 embedding、部分参数 | 秒级到小时级 | 捕捉新鲜兴趣和热点变化 | 反馈回路放大，系统复杂度激增 |

[Google TFX 的 continuous training 论文](https://www.usenix.org/conference/opml19/presentation/baylor)和 [Google Cloud MLOps 文档](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)都把这类系统描述为持续训练流水线：训练不只是一个 cron job，而是带数据验证、模型验证、元数据记录、发布门禁和回滚机制的自动化流程。[Pinterest 的 learned retrieval 工程文章](https://medium.com/pinterest-engineering/establishing-a-large-scale-learned-retrieval-system-at-pinterest-eb0eaf7b92c5)也强调，双塔召回一旦进入生产，就必须自动重训模型、导出 item embedding、构建向量索引，并让线上 user tower 与对应 index 同步。[ByteDance 的 Monolith](https://arxiv.org/abs/2209.07663)和 [Meta 的 QuickUpdate](https://www.usenix.org/conference/nsdi24/presentation/matam)则代表更激进的方向：为推荐系统做实时或近实时更新，让模型更快吸收新行为。但这种路线需要很强的基础设施和风控能力，不应该作为第一版目标。

所以本文建议把闭环后的更新体系拆成三种模式：**全量重训、续训、增量更新**。三者不是替代关系，而是从稳到快的三层能力。

### 18.1 全量重训：推荐系统的基准刷新

全量重训是最容易审计、最容易回滚的模式。它从一个固定时间点的数据快照开始，重新构造训练样本、重新训练模型、重新导出 item embedding、重新构建 ANN index。

典型流程是：

```text
freeze data window
  -> build feature snapshots
  -> build training examples
  -> train model from scratch or stable initialization
  -> validate offline metrics and case review
  -> export item embeddings
  -> build ANN index
  -> run shadow validation
  -> canary / A/B
  -> ramp up or rollback
```

全量重训的关键不是“每次从零开始”，而是每次都有完整 manifest。至少要能回答：

```text
这版模型用了哪个曝光日志窗口？
这版模型用了哪个商品特征快照？
这版模型用了哪个用户特征快照？
这版模型用了哪些 source 的样本，各自权重是多少？
这版 item embedding 对应哪个 item schema？
这版 ANN index 包含多少 item，过滤字段来自哪个快照？
线上 user tower 是否和这个 index 来自同一个 embedding space？
```

如果团队还在早期，第一版重训可以按天或按周手动触发，但流程要按自动化流水线设计。不要把“手动跑脚本”写死成长期方案。成熟之后，触发方式可以分三类：

| 触发方式 | 示例 | 适合阶段 |
| --- | --- | --- |
| 时间触发 | 每天凌晨、每周固定重训 | 闭环早期，数据规模稳定增长 |
| 数据触发 | 新增曝光超过阈值、新品数量超过阈值 | 商品变化明显、冷启动压力大 |
| 质量触发 | CTR 分桶单调性下降、负反馈上升、漂移报警 | 稳定生产阶段 |

全量重训适合做“干净刷新”。它的缺点是反馈不够新鲜。例如，用户今天刚表现出新的兴趣，如果只等明天 batch model 才吸收，推荐会迟钝。因此，全量重训通常要搭配 session feature、nearline profile 或用户侧短期行为池化。

### 18.2 续训：用上一版模型 warm-start，但不要无条件相信上一版

续训指从上一版模型参数开始，继续用新窗口数据训练。它的好处是训练更快，embedding 空间更稳定，ANN 近邻变化不会过于剧烈。对于双塔召回，这一点很重要，因为 item embedding 的空间如果每次重训都大幅旋转，线上分数分布、近邻关系和 rank policy 权重都会跟着波动。

但续训也有明显风险：它会继承上一版模型的偏差。如果上一版模型过度依赖热门商品、过度放大某些类目，续训可能只是在这个偏差上继续优化。

因此，续训不应该只喂最近一天的新样本。更稳妥的训练集通常由三部分组成：

```text
recent window: 最近 1-7 天，高权重，吸收新鲜反馈
replay window: 最近 30-90 天，中权重，防止遗忘
anchor set: 固定审计样本或长期分布样本，低权重，稳定 embedding 空间
```

示例权重可以是：

```text
0.55 * recent_real_feedback
0.25 * replay_real_feedback
0.10 * recent_exposed_no_action
0.05 * item_contrastive_or_profile_regularizer
0.05 * anchor_set
```

这里的数字只是起点，真正要看样本量和业务变化速度。更重要的是，续训必须保留固定评估集。这个评估集不能只包含整体点击，还应该包含：

- 新用户请求。
- 新商品候选。
- 长尾商品候选。
- 负反馈 case。
- ANN 高分但规则低分 case。
- 规则高分但 ANN 低分 case。
- 过去线上出现过问题的回归 case。

如果续训后整体 CTR 可能变好，但新商品覆盖、长尾覆盖或负反馈 case 明显变差，这版模型不应该直接发布。

### 18.3 增量更新：先更新用户兴趣，再考虑在线训练模型参数

推荐系统想变新，最简单的方式不是立刻做在线训练，而是把用户短期兴趣做成在线或 nearline 特征。

例如 user tower 可以把长期画像和短期行为分开：

```text
user_emb = f(
  long_term_profile_snapshot,
  session_positive_pool,
  session_negative_pool,
  nearline_recent_behavior,
  request_context
)
```

其中 session 行为可以在请求链路里直接维护，nearline 行为可以通过分钟级或小时级聚合更新。这样即使模型参数一天只更新一次，用户向量仍然能随最近行为变化。

这种做法比直接在线更新模型参数更适合作为第一阶段。它有几个优点：

- 不改变 item embedding 空间。
- 不需要频繁重建 ANN index。
- 回滚简单，可以关闭短期行为特征或降低权重。
- 更容易解释某次推荐为什么变化。

真正的在线训练或近实时参数更新要晚一些再做。Monolith 这类系统面向的是极高频、大规模、时间敏感的推荐场景；QuickUpdate 这类方案则尝试在完整模型发布之间，高频发布部分重要参数。它们的共同前提是：日志、特征、训练、发布、监控都已经高度平台化。普通团队不应该把这当作冷启动后的下一步，而应该先把 batch retraining 和 nearline profile 做稳定。

### 18.4 双塔召回的特殊问题：user tower 和 item index 必须成对发布

双塔召回和普通排序模型不一样。item tower 导出的 item embedding 会被写进 ANN index；线上请求时 user tower 生成 user embedding 去查这个 index。也就是说，线上存在一个隐含契约：

```text
online_user_tower.embedding_space_version
  == ann_index.item_embedding_space_version
```

如果 user tower 是新模型，ANN index 还是旧 item embedding，分数会失真；如果 ANN index 已经换新，但线上 user tower 没换，召回也会失真。这个问题在双塔系统里很常见，必须用发布元数据约束。

可以定义一个发布 manifest：

```json
{
  "schema_version": "candidate_model_release_v1",
  "release_id": "candidate_model_2026_06_26_001",
  "model_version": "two_tower_v17",
  "embedding_space_version": "emb_space_v17",
  "user_tower_artifact": "s3://model/two_tower_v17/user_tower",
  "item_tower_artifact": "s3://model/two_tower_v17/item_tower",
  "item_feature_snapshot_id": "item_feature_snapshot_2026_06_26",
  "training_dataset_id": "trainset_2026_06_01_2026_06_25",
  "item_embedding_export_id": "item_emb_2026_06_26_v17",
  "ann_index_version": "qdrant_index_2026_06_26_v17",
  "rank_policy_versions_allowed": [
    "rank_policy_v11",
    "rank_policy_v12"
  ],
  "status": "shadow"
}
```

线上服务读取时，不要只配置 `model_version`。更稳妥的是配置 `release_id`，由 release manifest 决定 user tower、item embedding、ANN index、rank policy 的组合关系。

发布顺序也要固定：

```text
1. 训练并导出 user tower / item tower
2. 离线导出 item embedding
3. 构建新 ANN index，但不接线上流量
4. 用新 user tower 查询新 index 做 shadow
5. 检查 valid rate、coverage、近邻 case、score 分布
6. 灰度发布 release_id
7. A/B 或 canary 通过后提高流量
8. 保留上一版 model/index/policy 作为 rollback
```

这就是为什么“闭环后重训”不能只写成“重新 train 一下”。推荐召回模型的发布对象不是单个模型文件，而是一组必须一致的 artifact。

### 18.5 训练窗口、延迟标签和反馈污染

闭环稳定后，另一个难点是标签延迟。点击很快发生，购买、退款、留存、投诉可能几小时、几天甚至更久才出现。如果每天重训时只看短期点击，模型会偏向短期刺激；如果等待所有深层反馈，模型又不够新鲜。

一个常见折中是分层使用反馈：

| 反馈 | 可进入训练的延迟 | 权重方式 |
| --- | --- | --- |
| click/detail view | 分钟级到小时级 | 可较快进入，但要考虑位置偏置 |
| long dwell/like/cart | 小时级 | 权重高于 click |
| purchase/conversion | 天级 | 高权重，但样本延迟 |
| hide/dislike/report | 分钟级到小时级 | 作为 guardrail 和负反馈优先进入 |
| refund/complaint/retention | 天级到周级 | 用于评估和长期校正 |

训练样本生成时要给每种 label 设定成熟窗口。例如：

```text
click label window: 30 minutes
cart label window: 24 hours
purchase label window: 7 days
no_action negative window: 12-24 hours
```

这些窗口不是通用标准，只是说明应该显式配置。不要今天用 30 分钟 no-action，明天改成 24 小时，却不更新 `training_dataset_manifest`。窗口变化会直接改变 label 语义。

还要警惕反馈污染。模型上线后产生的曝光会进入下一轮训练，如果没有探索和去偏，模型会越来越相信自己过去展示过的东西。这也是 [Eugene Yan 关于 production ML 维护的文章](https://eugeneyan.com/writing/practical-guide-to-maintaining-machine-learning/)反复强调的风险：模型不是在静态数据集上运行，而是在改变未来数据分布。

降低反馈污染的做法包括：

- 保留 guided exploration，让系统持续观测非最优候选。
- 记录 `logging_policy_id`，区分样本来自 baseline、ANN、探索还是实验策略。
- 做 per-policy 样本分布统计，避免某个策略的数据完全支配训练。
- 在训练中限制热门商品、头部类目和高曝光用户的采样权重。
- 用固定 holdout 和人工 case review 观察模型是否越来越窄。
- 对重要发布做 A/B，而不是只看下一天 replay 指标。

### 18.6 从成熟方案里能学到什么

下面这张表把几类成熟资料和本文建议对应起来。

| 来源 | 真实系统关注点 | 对本文路线的启发 |
| --- | --- | --- |
| Google TFX / Google Cloud MLOps | continuous training、数据验证、模型验证、metadata、自动发布 | 重训要做成流水线，不是手工脚本 |
| Google Rules of ML | 模型新鲜度、训练/服务一致性、上线前检测、可解释的系统演进 | 先建立可观测 baseline，再逐步增加 ML 复杂度 |
| Pinterest learned retrieval | 双塔召回、自动模型训练、item embedding、ANN index、线上同步 | user tower 和 item index 必须按 release 成对管理 |
| ByteDance Monolith | 实时训练、collisionless embedding、推荐反馈快速吸收 | 在线训练是高阶能力，适合高频大规模场景 |
| Meta QuickUpdate | 在完整模型发布之间快速更新重要参数 | 高频部分发布可以降低新鲜度问题，但要求强发布系统 |
| Netflix RecSysOps | 问题发现、预测、诊断、修复的推荐系统运维流程 | 推荐系统需要面向故障和质量退化设计运维闭环 |
| Eugene Yan production ML articles | 反馈回路、数据漂移、监控、重训事故、系统性排查 | 重训前先确认数据、特征、日志和策略版本是否可信 |

这些资料共同指向一个结论：成熟推荐系统不是“模型越来越复杂”，而是“训练、发布、监控、回滚越来越制度化”。对小团队来说，最实际的路线是：

```text
batch retraining first
  -> warm-start with replay and fixed evaluation
  -> nearline user features
  -> automated release manifest
  -> canary / A/B / rollback
  -> only then consider online training or partial parameter update
```

如果没有稳定 manifest、shadow validation、A/B 和 rollback，在线训练只会让问题更难定位。相反，如果这些基础都做好了，即使模型仍然只是简单双塔，也已经具备持续演进能力。

## 19. 监控与验收指标

推荐系统稳定性要分层监控。

### 19.1 数据闭环指标

| 指标 | 意义 |
| --- | --- |
| `exposure_log_write_success_rate` | 曝光日志是否可靠写入 |
| `interaction_log_write_success_rate` | 行为日志是否可靠写入 |
| `exposure_interaction_join_rate` | 行为能否回连曝光 |
| `request_id_missing_rate` | 是否缺关键 join key |
| `item_feature_lookup_success_rate` | 排序时能否取到商品特征 |
| `model_version_coverage` | 样本是否可追踪模型版本 |

这些指标是训练前置条件。

### 19.2 召回质量指标

| 指标 | 意义 |
| --- | --- |
| `ann_candidate_valid_rate` | ANN 返回商品是否合法 |
| `offline_leakage_rate` | 下架/删除/不可售是否泄漏 |
| `coverage@K` | 候选覆盖面 |
| `category_coverage@K` | 类目覆盖 |
| `new_item_coverage@K` | 新商品曝光机会 |
| `long_tail_coverage@K` | 长尾覆盖 |
| `duplicate_spu_rate` | 同 SPU 重复 |

早期先保证召回出来的东西合法、覆盖足够、不是全热门。

### 19.3 用户反馈指标

| 指标 | 解释 |
| --- | --- |
| CTR | 点击率 |
| long dwell rate | 长停留率 |
| like/favorite rate | 正向反馈 |
| cart/purchase rate | 深层转化 |
| hide/dislike/report rate | 负反馈和安全信号 |
| bounce rate | 会话级负面结果 |

弱监督模型上线最常见的问题不一定是 CTR 低，而是看起来相关但用户不舒服，导致 hide/dislike/report 上升。

### 19.4 模型迁移指标

| 指标 | 意义 |
| --- | --- |
| `real_feedback_sample_count` | 真实反馈样本是否足够 |
| `positive_negative_ratio` | 正负比例是否稳定 |
| `per_scene_sample_coverage` | 主要场景是否有样本 |
| `per_category_sample_coverage` | 主要类目是否有样本 |
| `ann_score_bucket_ctr` | ANN 分数是否和反馈单调 |
| `teacher_student_agreement` | 蒸馏是否学到旧规则 |
| `coverage_delta_vs_teacher` | 是否比旧规则覆盖更好 |

特别要看 `ann_score` 分桶。如果高分桶没有更高 click/like/cart，说明 ANN 分数还不能提高权重。

### 19.5 离线评估不要伪装成线上效果

冷启动阶段的离线评估很容易误导。常见做法是把历史点击作为正样本，随机采未点击商品当负样本，然后算 AUC、Recall@K、NDCG。这个评估可以做，但不能被解释成真实线上效果。

原因有三点。

第一，历史点击来自旧策略。旧策略没有展示过的商品，即使模型能找出来，离线日志里也没有机会证明它好。

第二，随机负样本多数是 unknown，不是用户明确不喜欢。

第三，位置、页面、场景和曝光次数会影响行为。rank 靠前的商品更容易被点击，这不一定代表它更符合偏好。

因此离线评估要分成几类：

| 评估 | 可回答的问题 | 不能回答的问题 |
| --- | --- | --- |
| item 近邻人工审查 | 商品空间是否合理 | 用户是否喜欢 |
| teacher agreement | 是否学到规则系统 | 是否超过规则系统 |
| replay hit rate | 是否召回历史点击 | 新策略线上会不会更好 |
| score bucket analysis | 分数是否和已有反馈相关 | 是否消除了 selection bias |
| coverage analysis | 是否覆盖更多类目、长尾、新品 | 用户是否会转化 |

真正的线上效果仍然需要 shadow、低权重上线、A/B 和探索日志。离线评估是安全门，不是最终裁判。

### 19.6 Case Review 应该怎么做

推荐系统早期非常需要人工 case review。不是为了代替指标，而是为了定位系统性错误。

可以抽四类 case：

1. ANN 高分但规则低分。
2. 规则高分但 ANN 低分。
3. 新商品被 ANN 召回。
4. 用户产生 hide/dislike/report 的曝光。

每个 case 看这些字段：

```text
user profile
context
recent positive items
recent negative items
candidate item features
candidate sources
ann_score
rule_score
final_score
rank_policy_version
model_version
exposure result
```

审查时不要只问“这个商品看起来对不对”，还要问：

- 是内容相关但场景不合适吗？
- 是价格区间错了吗？
- 是重复推荐太多了吗？
- 是新商品探索合理但位置太高了吗？
- 是规则兜底救回了模型错误，还是规则压制了模型发现？
- 是负反馈被 user tower 吸收了，还是只在 rank policy 被惩罚？

这些问题会直接指导下一轮改特征、改 loss、改探索或改排序。

## 20. 常见错误清单

**错误 1：把未曝光 item 当负样本。**  
这是最严重的样本构造错误。未曝光 item 是 unknown，只能低权重作为采样负例或分母。

**错误 2：synthetic 与真实反馈平权。**  
Synthetic user-SKU matching 学的是字段兼容性，不是真实偏好。它必须低权重，并逐步退出。

**错误 3：只蒸馏 teacher topK。**  
这会复制热门偏置和规则偏置。蒸馏应包含 near-boundary、tail、新商品、低分样本。

**错误 4：ANN 直接接管最终排序。**  
ANN 是一路召回，不是 ranker。库存、去重、业务规则、负反馈惩罚都应在策略层控制。

**错误 5：没有曝光日志就训练模型。**  
没有曝光日志就没有可靠 no-action negative，也无法解释样本来自哪个策略。

**错误 6：只看 CTR。**  
早期还要看 hide、dislike、report、coverage、new item exposure、join rate、valid rate。

**错误 7：过早引入重基础设施。**  
Milvus、Kafka、Debezium、ClickHouse、Triton、复杂精排可以后置。第一阶段先跑通闭环。

**错误 8：忽视版本。**  
没有 `model_version`、`feature_snapshot_id`、`ann_index_version`、`rank_policy_version`，模型迭代无法追踪。

### 20.1 故障排查表

当线上出现异常时，可以按症状倒查。

| 症状 | 优先排查 | 常见原因 |
| --- | --- | --- |
| 推荐结果大量为空 | hard filter、候选数量、ANN 超时 | 过滤太严、向量库不可用、候选源没兜底 |
| 下架商品出现 | guardrail、item feature snapshot | 快照延迟、过滤字段缺失、线上状态未二次校验 |
| 全是热门商品 | rank policy、蒸馏样本、hot_score | 热门权重过高、teacher topK 蒸馏过强 |
| 新商品没有曝光 | item tower 特征、exploration policy | 过度依赖历史统计、没有新商品探索桶 |
| hide/dislike 上升 | ANN 高分 case、负反馈处理 | 内容相关但意图不符、负反馈没有进入过滤/惩罚 |
| ANN 分数无单调性 | 训练样本、loss 权重、负样本 | synthetic/distill 过重、真实反馈不足、false negative 多 |
| A/B 波动大 | 样本量、scene 分桶、探索比例 | 总量太小、场景混杂、探索位影响核心坑位 |
| 离线好线上差 | selection bias、position bias、日志缺失 | 离线负样本不真实、历史策略偏置 |

排查顺序要从数据和版本开始，不要直接改模型。推荐系统里“模型效果差”经常是日志、特征、过滤、版本或策略问题。

### 20.2 什么情况下该暂停模型迭代

如果出现下面情况，应先暂停模型迭代，修工程闭环：

- 曝光日志写入不稳定。
- 行为无法可靠 join 回曝光。
- `request_id` 缺失率高。
- 商品特征快照不可复现。
- ANN index 和 item embedding 行数不一致。
- 下架商品泄漏。
- 负反馈率明显上升但无法定位来源。
- A/B 分桶无法解释。

这些问题不解决，继续训练新模型只会制造更多不确定性。

## 21. 最小可落地配方

如果只保留一页执行摘要，可以这样定。

模型：

```text
embedding_dim = 128
similarity = cosine / dot product with L2Norm
```

ItemTower：

```text
category
brand/shop
tag bag
attribute bag
title/search text embedding
image embedding optional
price bucket
numeric features with missing indicators
sku_id embedding with dropout
-> item_emb R128
```

UserTower：

```text
context
default segment / onboarding profile
long-term category/tag/style preference
short-term positive behavior
short-term negative behavior
price preference
-> user_emb R128
```

0 样本 loss：

```text
35% synthetic user-SKU matching
35% rule distillation
30% item-item contrastive
```

少量样本 loss：

```text
20% real feedback
10% exposed_no_action
25% synthetic
25% distillation
20% item contrastive
```

上线：

```text
ANN shadow first
then ann_score weight <= 0.20
rule/profile/hot fallback remains
guided exploration 5%-10%
write ExposureLogV1 from day one
```

核心边界：

```text
未曝光 item != 负样本
synthetic user != 真实偏好
规则蒸馏 != 效果提升
item-item 预训练 != 个性化
ANN recall != 最终排序
曝光日志 > 第一版模型
```

这套流程的价值不是让冷启动第一天就有最优推荐，而是让系统不随机、不失控、可观测、可回滚，并且能开始积累真正可训练的数据。

稳定之后，团队的工作节奏也应该固定下来。每天看数据闭环和线上反馈，每周做一次训练样本质量复盘，每次模型发布前做 manifest 和近邻检查，每次策略调整后观察分场景、分用户冷启动、分商品冷启动指标。推荐系统不是一次上线后结束的功能，而是一个持续校准的反馈系统。只要曝光、行为、样本、训练、索引和排序这几件事都有版本、有日志、有回滚，后续无论换更复杂的召回模型、引入多兴趣建模、增加精排模型，还是接入更大的向量平台，都不会推倒重来。

另外，稳定闭环并不意味着模型可以脱离人工审查。推荐系统早期仍应保留固定的抽样复盘机制，尤其关注新用户、新商品、长尾商品、负反馈曝光和探索位曝光。只要这些分桶里出现异常，就应该先检查日志、特征和策略版本，再讨论模型结构。这样团队才能把推荐迭代从“凭感觉调权重”变成“按证据定位问题”：先判断用户是否真的看见了候选，再判断行为是否能正确 join 回曝光，最后才判断 embedding、ANN 和 rank policy 是否需要调整。

## 参考文献与延伸阅读

- [Yifan Hu, Yehuda Koren, Chris Volinsky. Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008.](https://yifanhu.net/PUB/cf.pdf)  
  隐式反馈推荐的经典工作。对本文最重要的启发是：隐式行为不是干净正负标签，而是带不同置信度的 preference signal。

- [Steffen Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.](https://arxiv.org/abs/1205.2618)  
  个性化排序损失的经典基线。本文没有直接采用 BPR 作为唯一 loss，但它说明隐式反馈推荐需要优化排序目标，而不是普通分类目标。

- [Tobias Schnabel et al. Recommendations as Treatments: Debiasing Learning and Evaluation. ICML 2016.](https://proceedings.mlr.press/v48/schnabel16.html)  
  讨论推荐训练和评估里的 selection bias。对本文的关键启发是：历史推荐日志来自旧策略选择展示之后的观测，不能当成无偏全量样本。

- [Paul Covington, Jay Adams, Emre Sargin. Deep Neural Networks for YouTube Recommendations. RecSys 2016.](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)  
  工业推荐中 candidate generation 和 ranking 两阶段架构的代表性论文。本文采用“ANN 召回只是候选源，最终排序另做”的同类工程思想。

- [Denis Baylor et al. Continuous Training for Production ML in the TensorFlow Extended TFX Platform. USENIX OpML 2019.](https://www.usenix.org/conference/opml19/presentation/baylor)  
  生产 ML 持续训练流水线的代表性资料。本文用它支撑“重训不是 cron job，而是带数据验证、模型验证、metadata 和发布门禁的流水线”这一观点。

- [Google Cloud. MLOps: Continuous delivery and automation pipelines in machine learning.](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)  
  工程视角解释 ML pipeline 自动化、持续训练和部署成熟度。本文参考其思想，把推荐重训拆成触发、验证、发布和回滚。

- [Martin Zinkevich. Rules of Machine Learning: Best Practices for ML Engineering.](https://developers.google.com/machine-learning/guides/rules-of-ml)  
  Google 的 ML 工程实践指南。本文借鉴其“先建立可观测 baseline，再逐步增加复杂度”的系统演进思路。

- [Pinterest Engineering. Establishing a Large-Scale Learned Retrieval System at Pinterest.](https://medium.com/pinterest-engineering/establishing-a-large-scale-learned-retrieval-system-at-pinterest-eb0eaf7b92c5)  
  双塔 learned retrieval 的工业落地文章。本文参考其自动训练、item embedding 导出、ANN index 构建和线上同步的工程思路。

- [Jian Zhu et al. Monolith: Real Time Recommendation System With Collisionless Embedding Table. arXiv 2022.](https://arxiv.org/abs/2209.07663)  
  大规模实时推荐训练系统。本文把它作为“在线训练是高阶能力，不应作为第一版目标”的代表案例。

- [Dan Matam et al. QuickUpdate: Fast and Efficient Model Updates for Online Recommender Systems. NSDI 2024.](https://www.usenix.org/conference/nsdi24/presentation/matam)  
  研究如何在完整模型发布之间快速更新推荐模型参数。本文用它说明部分参数高频发布可以提升新鲜度，但需要成熟发布基础设施。

- [Mohammad Saberian, Justin Basilico. RecSysOps: Best Practices for Operating a Large-Scale Recommender System. RecSys 2021.](https://doi.org/10.1145/3460231.3474620)  
  Netflix 推荐系统运维实践。本文借鉴其把问题发现、预测、诊断、修复纳入推荐系统生命周期的思路。

- [Eugene Yan. A Practical Guide to Maintaining Machine Learning Systems in Production.](https://eugeneyan.com/writing/practical-guide-to-maintaining-machine-learning/)  
  生产 ML 维护指南。本文参考其对数据漂移、反馈回路、监控和系统性排查的讨论。

- [Maksims Volkovs, Guangwei Yu, Tomi Poutanen. DropoutNet: Addressing Cold Start in Recommender Systems. NeurIPS 2017.](https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems)  
  通过 dropout 显式训练冷启动能力。本文借鉴其核心思想：冷启动模型需要在训练时模拟缺失 ID 或缺历史行为的情况，不能只在上线时临时补救。

- [Lihong Li et al. A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010.](https://arxiv.org/abs/1003.0146)  
  推荐探索和利用的经典 contextual bandit 工作。本文不建议第一版直接上复杂 bandit，但强调受控探索和 logging policy 是冷启动闭环的基础。

- [Yuta Saito et al. Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation. 2020.](https://arxiv.org/abs/2008.07146)  
  讨论 logged bandit feedback 和 off-policy evaluation 的数据与工具。对本文的启发是：如果想离线评估新策略，日志必须记录策略、动作、反馈和上下文。

- [Negative Sampling in Recommendation: A Survey and Future Directions.](https://arxiv.org/html/2409.07237v1)  
  系统梳理推荐里的负采样问题。本文只取其中一个工程结论：负采样有 false negative 风险，不能把采样负例等价于用户明确负反馈。

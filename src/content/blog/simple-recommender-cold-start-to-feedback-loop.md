---
title: "简单推荐算法实施指南：从冷启动弱监督到真实反馈闭环"
description: "系统梳理推荐系统在缺少样本时如何通过内容先验、规则蒸馏、受控探索、曝光日志和双塔召回逐步建立稳定训练闭环"
pubDate: "2026-06-26T15:20:05+08:00"
updatedDate: "2026-06-26T15:20:05+08:00"
tags:
  - "Deep Reading"
  - "Recommendation System"
  - "Cold Start"
  - "Two-Tower"
  - "ANN"
  - "Machine Learning"
  - "Engineering"
draft: false
---

工程团队在从零构建推荐系统时，最容易踩入的误区是把“冷启动”当成一个纯粹的神经网络结构问题——试图在第一周就直接照搬大厂的复杂双塔或序列注意力模型，却对线上日志缺失、历史规则偏置与特征时间穿越视而不见。在真实生产环境中，启动期最致命的瓶颈在于系统处于**缺乏无偏偏好样本的真空期**，首要工程任务是搭建起从受控曝光到真实反馈回流的因果数据闭环：

```text
曝光 (Exposure)
  -> 用户行为 (Interaction)
  -> 时间点快照样本 (Point-in-Time Sample)
  -> 模型训练 (Training)
  -> 向量索引构建 (Vector Index)
  -> 在线候选召回 (Retrieval)
  -> 新一轮受控曝光 (Controlled Exposure)
```

面向电商商品、内容流、视频、课程与 3D 资产等通用推荐场景（文中以商品 `item` 或 `SKU` 为主例），当我们手头仅有商品元数据、基础规则排序与少量用户画像，甚至尚未建立完整曝光日志时，整套推荐系统的工程设计与演进浓缩为**四条核心设计哲学**：

1. **观测边界先于模型训练**（Unknown $\neq$ Negative 与防穿越快照）：在推荐系统中，用户未点击某件未曾曝光的商品仅代表**未观测状态**（Unknown），绝不等同于**真实负反馈**（Negative）。系统上线的第一资产是携带策略版本号的曝光日志（`ExposureLogV1`）与严格回滚至曝光时刻 $t$ 之前的点快照样本（`TrainingExampleV1`）。唯有严守“先真实曝光、再等待完整归因窗口、最后判定弱负样本（`exposed_no_action`）”的因果铁律，模型才不会把长尾商品与新上架商品系统性压死。
2. **表征学习与业务规则正交解耦**（单池双塔召回 + 声明式 `RankPolicyV1`）：拒绝让双塔神经网络同时背负库存过滤、同 SPU 去重、已曝光疲劳度惩罚与运营加权等异构职责。双塔模型只专注在 $128$ 维单位超球面上度量“用户—商品”的语义与偏好内积，而将一票否决硬护栏（Guardrail）、多路通道分数融合与实时点踩重罚封装进可秒级热更新的声明式策略层 `RankPolicyV1`。
3. **弱监督充当启动脚手架、受控探索破除信息茧房**：在零真实反馈的启动期，联合运用商品多视图内容对比、逆向合成伪用户画像对齐（Synthetic Profile Matching）与全谱系六分桶规则蒸馏（Rule Distillation）撑开初始双塔度量空间；上线后在核心位维持稳健利用（Exploitation）的同时，固定划拨 $5\%\sim 10\%$ 中腰部坑位开展**受控引导探索**（Guided Exploration），主动将冷启动新品与模型分歧候选从 Unknown 转化为 Observed。
4. **双塔与向量索引的几何同构绑定与四阶段退火演进**：在线实时推理的 User Tower 与离线批量构建的 Item ANN Index 必须处于完全相同的版本化内积空间，任何一方单独升级都会造成内积失真，必须通过 `CandidateModelReleaseV1` 实现原子级成对发布与一键回滚；随着真实反馈样本在四个演进阶段中逐步成熟，多目标混合损失中的弱监督权重按预设门禁有序退火，最终完成向真实用户行为自演进闭环的平稳交接。

> 冷启动阶段的核心工程原则：用内容特征与既有规则的弱监督信号初始化双塔向量空间，用低权重 ANN 召回与受控探索制造可学习曝光，用严格的时间点快照日志将 unknown 逐步转化为 observed，最终让真实用户反馈接管主监督。

## 冷启动的核心矛盾与单池召回总体架构

在标准监督学习任务中，工程师通常先获取一批带有明确输入输出标签的静态数据集，再着手设计模型结构。推荐系统则面临截然不同的观测机制：用户只能对系统已经展示过的候选商品产生反馈。正如 Schnabel 等人在 *Recommendations as Treatments* 中所指出的，推荐系统的历史日志是由既有推荐策略（Logging Policy）筛选展示后的有偏观测（异于从全量“用户—物品”笛卡尔积空间中均匀采样），天然伴随严重的选择偏差（Selection Bias）与曝光偏差（Exposure Bias）。

在系统启动初期，制约推荐效果的四大核心缺口均集中在数据与机制层面：

| 核心缺口 | 线上具体表现 | 忽视该缺口的工程后果 |
| --- | --- | --- |
| 曝光日志缺失 | 仅保存接口返回列表或只记点击，不知用户实际看见了哪些商品 | 无法定义“曝光未点击”弱负样本，训练与归因失去分母 |
| 负样本语义混淆 | 将全库未点击商品无差别视为用户不喜欢的负样本 | 把未知（Unknown）当成负反馈（Negative），长尾与新品被系统性压制 |
| 策略版本断层 | 样本未记录产生该次曝光的模型版本、索引版本与排序策略版本 | 无法区分分布变化源于用户兴趣迁移还是策略自身调整 |
| 探索机制缺位 | 线上永远只展示旧规则认为高分的头部商品 | 新品与边界候选得不到曝光机会，反馈闭环锁死在历史局部分布 |

因此，构建推荐系统的第一原则是**先跑通可追溯的数据闭环，再迭代复杂模型**。在架构设计上，本文采用语义清晰、链路最短的**商品单池向量召回**方案：以商品（SKU）作为唯一召回实体，仅维护一个统一的商品向量库；商品所属的类目、品牌、标签、价格、库存以及文本与图像表征全部作为输入特征汇入 Item Tower，无需在召回前引入多子库路由分支。在线服务链路为：

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

与之配合的商品侧离线索引构建链路为：

```text
商品数据库
  -> ItemFeatureV1 快照与质量校验
  -> ItemTower 离线批量推理
  -> ItemEmbeddingV1
  -> ANN Index 构建
  -> ANNIndexManifestV1
```

这一单池架构在工程上具备三重优势：一是召回语义纯粹，线上检索统一收敛为 `user_emb -> item_emb TopN`；二是演进路径平滑，首版可先用确定性特征投影或内容向量写入 Qdrant、Milvus、FAISS/HNSWLib 验证在线检索链路，随后无缝替换为双塔模型导出的向量；三是回滚边界清晰，ANN 检索仅作为一路候选来源接入系统，即便初期向量质量波动，仍有规则、热门与画像通道兜底。

在数学形式上，本文默认双塔输出向量维度为 $128$，通过点积计算用户 $u$ 与商品 $i$ 的匹配得分：

$$
\operatorname{score}(u,i)=u_{\text{emb}}^{\top}i_{\text{emb}}.
$$

当用户向量与商品向量均经过 $L_2$ 归一化处理时，点积得分在排序意义上与余弦相似度完全等价：

$$
\cos(u,i)=
\frac{u_{\text{emb}}^{\top}i_{\text{emb}}}
{\|u_{\text{emb}}\|_2\|i_{\text{emb}}\|_2}.
$$

### 推荐系统的五层职责边界与数据契约

许多早期推荐系统陷入混乱的根源在于职责越界——例如试图让双塔召回模型同时包揽业务加权、库存过滤、已曝光去重与最终列表排序，既污染了表征学习目标，又导致线上规则微调必须重新训练模型。稳健的工程架构应当从第一天起将系统拆解为职责正交的五层：

| 系统层次 | 主要职责 | 严禁越界的行为 |
| --- | --- | --- |
| Candidate Source（候选源层） | 通过双塔 ANN、热门榜单、画像标签、规则检索等多通道生成候选 | 不直接决定最终展示排序 |
| Guardrail（安全护栏层） | 执行下架、删除、零库存、黑名单、举报、合规等一票否决硬过滤 | 不用模型预测分数覆盖硬性业务规则 |
| Rank Policy（排序策略层） | 融合多路候选与多维分数，施加已曝光、已购买与同款去重惩罚 | 不把所有动态运营逻辑写死在神经网络权重里 |
| Logging（日志采集层） | 记录真实曝光、交互行为、策略版本、候选来源与探索标记 | 不只记录最终 HTTP 响应而丢失展示上下文 |
| Training（离线训练层） | 从历史日志与特征快照构造样本、训练模型、导出向量与 Manifest | 不直接读取线上实时可变数据库状态去拼接历史样本 |

上述五层在运行期通过三类标准化的中间数据结构完成交接：

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

依靠这三层中间对象，线上任何一次推荐结果都能实现端到端倒查：某个商品为何出现在第 $k$ 位、由哪几路通道提名、经历了哪些硬过滤、各项子分数如何加权、用户看到后产生了何种反馈。第一版最小可行系统（MVP）只需优先达成九项工程闭环指标：请求稳定返回 TopK 商品、返回商品 $100\%$ 在售合法、每个候选可解释来源与得分构成、每次展示落盘曝光日志、用户行为可通过 `request_id + item_id` 精确回连曝光、离线能从日志生成防穿越训练样本、支持批量导出商品向量并构建 ANN 索引、支持 ANN 以 Shadow 或低权重模式接入，以及配备完整的降级 Fallback 与一键熔断开关（Kill Switch）。

## 三类冷启动解耦与监督信号可信度分层

在工程方案设计中，“冷启动”包含三类性质迥异的子问题，必须分别制定应对策略：

1. **系统冷启动**（System Cold Start）：推荐系统刚启动运行，既无历史曝光日志，也无真实点击、收藏、加购、购买、隐藏或举报反馈。此时不存在可学习的真实偏好监督，首要工程目标是落地商品特征快照、规则或热门基线、曝光与行为日志、版本化排序策略、受控探索机制以及样本生成链路。这一阶段产出的初版模型本质上是基于内容与规则先验的初始化模型，目标是做到内容相关、类目合理、价格区间得当且新商品可被召回。
2. **用户冷启动**（User Cold Start）：新注册用户尚无历史交互，或匿名访客仅有当前会话上下文。此时 User Tower 不能依赖 `user_id` 独立向量，而应充分挖掘当前请求场景（首页、商详页、搜索后推荐）、设备与地域时间上下文、注册引导（Onboarding）选择的兴趣偏好、默认人群分群（Segment）以及会话内的实时点击、停留、跳过与隐藏信号，在热门与高质候选兜底的基础上实现会话内快速适应。
3. **商品冷启动**（Item Cold Start）：新上架商品缺乏 CTR、CVR、销量、收藏与购买等后验统计，且累计曝光极低。此时 Item Tower 必须摆脱对 `sku_id` 记忆表征与历史热度的依赖，依靠三级类目、品牌店铺、标签与结构化属性（材质、颜色、风格、房间、用途）、标题文本向量、主图视觉向量、价格折扣及同 SPU/同系列关系生成内容向量，并借助线上探索配额获取初始曝光反馈。

### 监督信号可信度分层（L0–L7）

在样本积累过程中，不同来源的数据蕴含的偏好可信度天差地别。Hu、Koren 与 Volinsky 在隐式反馈协同过滤经典论文中指出，隐式行为应被建模为偏好方向（Preference）与置信度权重（Confidence）的结合。据此，工程上可将全部可用信号划分为 L0 至 L7 八个层级：

| 层级 | 信号来源 | 模型能学到什么 | 可信度 | 主要工程用途 |
| --- | --- | --- | --- | --- |
| **L0** | 商品状态、上下架、删除、库存、价格、类目、品牌、结构化属性 | 商品客观事实与合法过滤边界 | 高 | Guardrail 硬过滤、Item Tower 基础特征 |
| **L1** | 具备曝光上下文的 `purchase`、`cart`、`like`、`favorite`、`hide`、`dislike`、`report` | 用户对已展示商品的强正/负响应 | 高 | 核心真实正负标签、实时负向惩罚与过滤 |
| **L2** | 具备曝光上下文的 `click`、`long_dwell`、`detail_view` | 短期注意力与中等强度兴趣意图 | 中高 | 主力训练正样本、排序阶段时长加权目标 |
| **L3** | 曝光未点击、曝光无动作（曝光可验证且归因等待窗口已结束） | 特定上下文与位置下的弱负反馈 | 中低 | 低权重真实负样本（`exposed_no_action`） |
| **L4** | 历史热度、销量、统计 CTR/CVR、运营加权权重 | 群体平均偏好或业务导向偏好 | 中低 | 排序策略先验项、异常降级兜底 |
| **L5** | 同 SPU、同系列模型、高属性重合、文本/图像向量近邻 | 商品客观内容空间中的语义邻近关系 | 中 | Item Tower 自监督/对比学习预训练 |
| **L6** | 从商品结构化属性逆向合成的伪用户画像（Synthetic User Profile） | 用户画像字段与商品属性字段的字面兼容 | 低 | 零样本阶段 User Tower 与 Item Tower 空间对齐 |
| **L7** | 既有规则系统输出列表与规则综合打分 | 存量启发式规则策略的函数近似 | 低至中 | 规则蒸馏预热（Warmup）、多规则特征压缩 |

解读这张分层表需把握三条核心边界：第一，L0 是高可靠客观事实，但事实本身不代表用户偏好（`category=sofa` 仅说明它是沙发，用于过滤与内容表征）；第二，L1、L2、L3 必须绑定曝光上下文（脱离具体场景、展示位次与策略版本的孤立交互会降低归因质量）；第三，L6 与 L7 属于冷启动弱监督脚手架，仅用于初始化双塔向量空间，绝不能长期替代真实用户反馈。

### 负样本语义边界：未曝光商品绝非真实负反馈

推荐系统样本工程中最具破坏性的错误，是将“用户没有点击某件商品”直接等价为“用户讨厌该商品”。正确的样本状态判定必须以**是否真实曝光**作为分水岭：

```text
用户是否真实看到过 item_x？
  ├── 否 -> Unknown（未观测状态，仅可作低权重采样分母，绝非真实负反馈）
  └── 是 -> 归因等待窗口内是否产生交互动作？
        ├── 有正向交互（click / like / cart / purchase） -> Positive（真实正样本）
        ├── 有显式负向交互（hide / dislike / report）    -> Explicit Negative（强负样本）
        └── 窗口结束且无任何交互动作                      -> Weak Negative（曝光未动弱负样本）
```

围绕这一边界，训练中使用的三类负样本各自具备明确的适用范围与权重上限：
1. **全库随机负样本与批内负样本**（Random / In-Batch Negatives）：充当对比学习（Contrastive Loss）的分母项以防止向量坍缩，帮助模型拉开“沙发与灯具、耳机与服饰”等粗粒度类目距离。由于随机采到的商品绝大多数处于未观测（Unknown）状态，必须赋予极低权重，并在采样时剔除同 SPU、用户历史交互过或属性高度重合的潜在正例。
2. **同类目难负样本**（Hard Negatives）：例如当用户画像指向“北欧风客厅布艺沙发（正样本）”时，采样“工业风客厅皮质沙发”作为同类目难负例，迫使模型学习材质与风格等细粒度决策边界。但同类目不同属性的商品往往也是潜在替代品，存在假阴性风险，因此仅宜赋予低至中等权重。
3. **曝光未点击弱负样本**（`exposed_no_action`）：最贴近线上真实分布的负反馈来源，但其成立必须同时满足五项条件：商品确实进入用户可见视口、完整记录了 `request_id / rank / position / scene`、预设归因等待窗口已结束、窗口内无任何正负向动作，且经去重排除了高频重复曝光噪声。考虑到用户未点击可能是由于位置靠后或临时分心，其样本权重必须显著低于正向点击与显式点踩。

不同交互事件在训练中的参考样本权重配置如下：

| 行为或负样本类型 | 示例权重 | 语义定位与工程解释 |
| --- | ---: | --- |
| `purchase` | `5.0` | 最高置信度成交正反馈 |
| `cart` | `4.0` | 强购买意图正反馈 |
| `like / favorite` | `3.0` | 明确偏好正反馈 |
| `long_dwell` | `1.5` | 深度浏览正反馈 |
| `click` | `1.0` | 基础正向注意力信号 |
| `exposed_no_action` | `0.1 ~ 0.3` | 经过完整等待窗口确认的曝光未点击弱负反馈 |
| `hide / dislike / report` | `4.0 ~ 5.0` | 显式强负反馈或合规安全信号 |

随着系统从零样本推进至稳定闭环，负样本的组合策略也应同步演进：

| 演进阶段 | 主力负样本来源 | 训练语义定位 |
| --- | --- | --- |
| 0 样本阶段 | Item-Item 对比随机负例、同类目属性互斥难负例 | 撑开商品内容空间，不代表真实用户厌恶 |
| 少量样本阶段 | 显式负反馈（hide/dislike）、低权重 `exposed_no_action`、批内负例 | 开始引入真实曝光下的用户响应边界 |
| 可训练样本阶段 | `exposed_no_action`、上一版 ANN/策略召回未点击、同类目难负例 | 学习真实展示环境下的细粒度排序决策边界 |
| 稳定闭环阶段 | 显式负反馈、`exposed_no_action`、排序层难负例、批内负例 | 由真实曝光反馈全面主导负向边界学习 |

## Item Tower：构建稳定的商品内容表征空间

在系统冷启动与商品冷启动阶段，商品侧元数据完备度最高，因此应率先构建 Item Tower 学习稳定的商品内容表征空间。第一版标准化商品特征 Schema `ItemFeatureV1` 定义如下：

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

这里有两项决定冷启动成败的特征工程设计：
1. **跨类目动态属性扁平化编码**：不同品类的商品属性键（Key）差异极大（家具看材质与房间，服饰看尺码与版型）。将所有结构化属性统一序列化为 `key=value` 词元（如 `style=nordic`、`room=living_room`、`material=fabric`），能够让全库所有品类共享同一套 `EmbeddingBag` 编码器，避免了稀疏宽表带来的维度灾难。
2. **后验统计字段与缺失指示器配对解耦**：所有历史统计特征（如 `ctr_7d`、`sales_7d`）必须同步传入布尔型缺失指示器（`ctr_7d_missing`）。新上架商品因尚无曝光而呈现 `ctr_7d = 0.0, ctr_7d_missing = true`，其物理语义与“经历上万次曝光却无人问津的老商品（`ctr_7d = 0.0, ctr_7d_missing = false`）”截然相反。显式拆分这两个通道，能彻底防止神经网络把缺乏曝光机会的新品误判为劣质品。

### 商品内容对比预训练的正负样本构造

在零用户反馈阶段，可利用 L5 级商品语义邻近信号对 Item Tower 开展对比预训练。正样本对可采自：同一 SKU 经特征掩码（Feature Dropout）生成的两个增强视图、同 SPU 下的不同 SKU、同系列商品，或同类目下属性高度重合且文本/图像向量相近的商品对。其中同 SKU 的多视图增强通过随机丢弃部分字段构造：

```text
view_1 = 随机掩码一部分 tag / attr / title token 后的商品特征视图
view_2 = 随机掩码另一部分 tag / attr / image feature 后的商品特征视图
```

对比学习的目标是拉近 `view_1` 与 `view_2`（以及同 SPU 近邻）的归一化输出向量，同时推开采自不同一级类目、不同核心使用场景或互斥材质风格的负样本商品。通过这一预训练，Item Tower 能够习得跨字段语义融合与抗属性缺失的鲁棒性，赋予新上架商品合理的内容初始坐标。

### 商品特征质量门禁与近邻巡检

许多线上“模型召回质量差”的根源其实在于上游商品快照脏乱或字段大面积缺失。在启动 Item Tower 训练前，数据管线必须针对每日或每小时刷新的商品快照执行七项质量检查：

| 检查项 | 校验目的 | 触发阻断或告警的异常信号 |
| --- | --- | --- |
| `item_id` 唯一性 | 确保全链路推荐主键稳定 | 同一物理商品对应多个 ID，或主键随快照批次漂移 |
| 上下架状态一致性 | 保证硬过滤与训练样本合法 | `offline` 或 `deleted` 状态商品混入训练正样本 |
| 类目树覆盖率 | 确认骨干分类特征可用 | 大量商品类目为空或落入 `unknown` 兜底桶 |
| `tag / attr` 缺失率 | 确认细粒度内容字段完备 | 新接入批次商品大面积缺失标签与结构化属性 |
| `price` 分位数分布 | 防止极端录入错误污染网络 | 价格出现 0、负数或 99 分位值剧烈跳变 |
| 文本/图像向量覆盖率 | 保障多模态冷启动通道畅通 | 标题为空或预提取 `title_embedding` 缺失率激增 |
| 后验统计缺失指示器 | 解耦新品零值与差品零值 | 新品未正确标记 `ctr_7d_missing = true` |

工程管线应将检查结果固化为标准化的 `item_feature_quality_report` JSON 工件，并设置自动化准入门槛（例如当 `offline_in_training_positive_count > 0` 或 `missing_text_embedding_rate` 超标时直接阻断训练与索引发布）：

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

每次导出 `ItemEmbeddingV1` 后，还应自动抽样一批种子商品与纯冷启动新品，打印其向量空间 TopK 近邻列表。若发现沙发的近邻中混入灯具，或新品完全找不到同风格近邻，即可在离线阶段提前拦截特征编码故障。

## User Tower：多源特征融合与实时兴趣池化

用户侧建模的核心挑战在于冷热分布极度不均，因此首版 User Tower 应将 `user_id_embedding` 视为可选低维辅助特征，将表征重心放在上下文、长期偏好与短期正负行为序列上。标准化用户特征契约 `UserFeatureV1` 如下：

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

对于短期交互行为序列，第一阶段采用**时间衰减加权池化**即可兼顾表达力与低延迟。每个交互行为词元融合目标商品向量 $e_i$、行为类型、时间分桶、停留时长与价格带特征，其时间衰减权重定义为：

$$
w_i=a_{\text{type}(i)}\exp(-\Delta t_i/\tau),
$$

其中 $a_{\text{type}(i)}$ 为行为类型权重，$\Delta t_i$ 为该行为距当前请求的时间间隔，$\tau$ 为衰减时间常数。用户的近期正向反馈集合 $\mathcal{P}_u$ 与显式负向反馈集合 $\mathcal{N}_u$ 分别通过加权池化聚合为定长向量：

$$
p_u=
\frac{\sum_{i \in \mathcal{P}_u} w_i e_i}
{\sum_{i \in \mathcal{P}_u} w_i+\epsilon},
\quad
n_u=
\frac{\sum_{i \in \mathcal{N}_u} w_i e_i}
{\sum_{i \in \mathcal{N}_u} w_i+\epsilon}.
$$

在融合正负兴趣时，切忌直接做手工向量相减（`user_emb = p_u - n_u`），而应将正、负行为池化向量与长期画像、请求上下文拼接后交由 MLP 学习非线性交互。第一版双塔模型的完整组件结构如下：

```text
ItemTower:
  item_id_emb (optional with feature dropout)
  category_emb
  brand_emb
  tag_embedding_bag
  attr_embedding_bag
  price_bucket_emb
  dense_item_mlp
  title_text_proj
  image_proj (optional)
  -> concat
  -> fusion_mlp
  -> L2Norm
  -> item_emb R128

UserTower:
  user_id_emb (optional / default for cold users)
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

为了防止模型死记老用户与老商品 ID（遵循 Volkovs 等人在 DropoutNet 中验证的冷启动泛化机制），训练期应对 `item_id_emb` 与 `user_id_emb` 施加高比例 Feature Dropout，并为新用户与新商品绑定统一的 `default / unknown` 表征；同时对所有连续数值特征执行对数压缩、极值截断与缺失解耦处理：

| 原始数值特征 | 建议预处理方式 |
| --- | --- |
| `price` | `log1p(price)` 或离散 `price_bucket` |
| `sales_7d` | `log1p(sales_7d)` + `sales_7d_missing` 指示器 |
| `ctr_7d / cvr_7d` | 极值截断（Clip）+ 标准化 + 缺失指示器 |
| `publish_age_hours` | `log1p(age_hours)` 或新鲜度分桶 |
| `dwell_seconds` | 极值截断 + 分桶 + 标准化 |
| `active_days_30d` | 极值截断 + 归一化 |

针对高频活跃用户，为避免每次在线请求都实时回放数百条历史事件，工程上可在特征存储中为每位用户维护两个增量更新的实时兴趣向量 `user_pos_interest_vec` 与 `user_neg_interest_vec`。每当用户产生新交互时，流式任务直接按指数衰减公式刷新状态向量：

$$
p_u^{\text{new}}
=
\lambda p_u^{\text{old}}
+a_{\text{type}} i_{\text{emb}},
\quad
n_u^{\text{new}}
=
\lambda n_u^{\text{old}}
+b_{\text{type}} i_{\text{emb}},
$$

在线 User Tower 推理时只需直接读取这两个定长状态向量并拼接画像与上下文特征，即可将在线特征组装延迟压缩至数毫秒以内。

## 冷启动弱监督：伪用户属性对齐与规则系统蒸馏

在真实曝光反馈尚未形成的启动期，如果仅对 Item Tower 做自监督预训练，User Tower 的参数依然处于随机初始化状态，两个塔输出的向量根本不在同一个度量坐标系内。为了让 User Tower 与 Item Tower 在零样本期就能输出几何对齐的向量，工程上可联合采用**伪用户属性匹配**（Synthetic User-SKU Matching）与**规则系统蒸馏**（Rule Distillation）两项弱监督手段进行跨塔预热。

### 伪用户属性对齐：跨塔字段兼容性初始化

给定一件结构化属性明确的商品：

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

离线脚本可提取其属性标签逆向合成一个虚拟用户画像 `UserFeatureV1`：

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

以 `synthetic user profile -> matching SKU` 作为配对样本训练双塔，能够快速教会网络将带有“北欧风、客厅、布艺、500–800元”画像特征的用户向量拉向对应属性的沙发商品向量。需要注意的是，这类合成样本仅反映了字段字面兼容，不包含真实用户的点击转化偏好；若权重过高，推荐会退化为僵硬的属性过滤。因此合成样本必须显式标注来源并赋予低置信度权重（如 `{"source": "synthetic_profile_match", "confidence_weight": 0.2}`）。

### 规则系统蒸馏：平移存量启发式经验

若业务在引入双塔前已运行着一套融合热门、类目匹配、标签重合、价格偏好与已曝光惩罚的规则排序系统（Rule Ranker），则可将其作为 Teacher 开展知识蒸馏：

```text
UserFeature / profile / context
  -> 当前线上 Rule Ranker
  -> Teacher 打分排序后的候选商品列表
  -> Student 双塔模型学习 Teacher 的相对排序分布
```

设 Teacher 与 Student 对候选商品 $i$ 的得分分别为 $s_i^{\text{teacher}}$ 与 $s_i^{\text{student}}$，引入蒸馏温度 $T$ 计算软化概率分布：

$$
p_i^{\text{teacher}}
=
\frac{\exp(s_i^{\text{teacher}}/T)}
{\sum_j \exp(s_j^{\text{teacher}}/T)},
\quad
p_i^{\text{student}}
=
\frac{\exp(s_i^{\text{student}}/T)}
{\sum_j \exp(s_j^{\text{student}}/T)},
$$

再通过最小化列表级 KL 散度优化双塔参数：

$$
\mathcal{L}_{\text{distill}}
=
\mathrm{KL}(p^{\text{teacher}}\|p^{\text{student}}).
$$

由于规则系统天然带有热门偏置、候选覆盖窄与新品低曝光等历史局限，实施规则蒸馏必须严守三条工程纪律：
1. **合理设置温度 $T$**：避免 $T$ 过小导致模型只死记 Top 1 爆款，适度调大 $T$ 让排序中段与边界候选共同参与梯度更新。
2. **坚持全谱系分桶采样**：构建蒸馏候选列表时切忌只取 Teacher TopK，必须按以下六个分桶均衡采样：

| 蒸馏候选分桶 | 采样目的与训练作用 |
| --- | --- |
| `top bucket` | 学习 Teacher 明确判定高相关的头部候选 |
| `middle bucket` | 学习排序中腰部梯度的相对差异 |
| `near-boundary bucket` | 强化入选边界附近的细粒度区分能力 |
| `low bucket` | 明确推开明显不匹配的劣质候选 |
| `new item bucket` | 强制纳入新上架商品，避免新品在蒸馏期完全失声 |
| `long-tail bucket` | 纳入长尾冷门商品，抑制模型退化为纯热门榜单复刻机 |

3. **严格执行蒸馏退场条件**：一旦曝光日志覆盖与交互回连率稳定、各主场景积累足够真实正负样本、`ann_score` 分桶与真实反馈呈单调正相关、且线上 A/B 不劣于规则基线同时未推高负反馈（hide/dislike/report），就必须按计划下调直至移除蒸馏权重。

## 曝光日志、防穿越样本契约与版本化 Manifest

推荐系统的训练闭环完全建立在曝光日志与时间点特征快照之上。从系统上线的第一天起，服务端或客户端就必须按以下标准契约落盘 `ExposureLogV1` 与 `InteractionEventV1`：

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

交互事件日志通过 `request_id + user_id + item_id` 与上述曝光记录精确关联：

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

### 时间点快照与防数据穿越样本契约

离线训练样本由 `ExposureLogV1 + InteractionEventV1 + UserFeatureSnapshot + ItemFeatureSnapshot` 关联生成。在拼接过程中，严禁直接读取用户当前的最新画像表（那会把用户在时刻 $t$ 点击商品后才新增的偏好标签泄漏回时刻 $t$ 的输入特征，制造虚假的离线高准确率），必须严格按曝光发生时刻 $t$ 回滚特征窗口：

```text
for each exposure at time t:
  user_snapshot = 仅包含严格发生于时刻 t 之前的用户历史特征
  item_snapshot = 时刻 t 生效的商品特征快照版本
  label         = 时刻 t 之后在归因等待窗口内发生的交互事件
  negatives     = 同一请求或窗口内已曝光但等待窗口结束后仍无动作的商品
```

生成的标准训练样本契约 `TrainingExampleV1` 如下：

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

不同反馈事件应根据业务转化链路配置差异化的归因等待窗口，同时在样本中完整保留细粒度来源标记 `source`（涵盖 `real_click`、`real_cart`、`real_purchase`、`explicit_negative`、`exposed_no_action`、`synthetic_profile_match`、`rule_distillation`、`item_contrastive`），切勿将不同可信度的样本粗暴压扁为单一的二元标签：

| 交互事件 | 建议归因等待窗口 | 窗口设定说明 |
| --- | --- | --- |
| `click` | 分钟级到小时级 | 实时响应最快 |
| `long_dwell` | 分钟级 | 依赖页面停留时长回传 |
| `favorite / cart` | 小时级到天级 | 用户可能跨会话比价后收藏加购 |
| `purchase` | 天级 | 决策转化链路长，存在明显延迟反馈 |
| `exposed_no_action` | 至少等待一个完整合理窗口（如 12–24 小时） | 严禁刚下发曝光数分钟即草率判为负样本 |

### 保障全链路可复现的三类 Manifest

为了确保任何一次模型训练与线上索引都能精确复现并快速定位事故根因，特征快照、训练数据集与向量索引必须分别生成版本化 Manifest。

第一份是商品特征快照清单 `ItemFeatureSnapshotManifestV1`：

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

第二份是训练数据集清单 `TrainingDatasetManifestV1`：

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

第三份是在线向量索引清单 `ANNIndexManifestV1`：

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

## 在线 ANN 检索、策略层排序与受控探索

在向量检索组件选型上，第一阶段应优先考虑部署与过滤友好的轻量方案：

| 向量库选型 | 首版适配度 | 工程选型说明 |
| --- | --- | --- |
| **Qdrant** | 高 | 部署与 API 直接，Payload 属性过滤友好，非常适合快速打通线上 ANN 召回链路 |
| **Milvus** | 中 | 面向超大规模向量检索的分布式平台，冷启动首版引入运维成本偏重 |
| **FAISS / HNSWLib** | 中 | 适合离线基准测试或进程内嵌检索，若作独立在线服务需自行补齐索引热更新与过滤 |
| **pgvector** | 中 | 适合商品规模不大且技术栈已深度绑定 PostgreSQL 的团队 |
| **OpenSearch / ES Vector** | 中 | 适合复用现有搜索基础设施，但需独立评估高并发向量检索延迟 |

在线推荐服务的完整处理流水线如下：

```text
RecommendRequest
  -> 读取用户画像与实时状态存储 (Profile Store)
  -> 组装标准化特征 UserFeatureV1
  -> UserTower 实时推理生成 user_emb
  -> 查询向量库获取 ANN TopN（如 200~500 个候选）
  -> 批量查询候选商品特征 ItemFeatureV1
  -> 执行硬过滤 (Hard Filters)
  -> 合并规则、画像、热门与业务召回通道候选
  -> 按 item_id / spu_id 去重
  -> RankPolicyV1 综合打分与重排
  -> 返回最终 TopK 响应并落盘 ExposureLogV1
```

其中 ANN 召回通道返回的每个候选对象均需携带完整的版本与位次元数据，供后续排序与日志记录消费：

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

### RankPolicyV1：策略层多路融合与硬过滤

为了防止双塔召回模型背负库存、去重、疲劳度与运营加权等非表征任务，所有多路候选统一汇入声明式排序策略 `RankPolicyV1`：

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

每个合法候选的最终排序得分按如下线性组合与惩罚公式计算：

$$
s_{\text{final}}
=
\alpha s_{\text{ann}}
+\beta s_{\text{profile}}
+\gamma s_{\text{hot}}
+\delta s_{\text{business}}
+\eta s_{\text{rule}}
-p.
$$

新模型上线严格遵循**三步渐进放量策略**：
- 第一步为 **影子验证阶段**（Shadow Mode）：线上实时计算 ANN TopN 并写入影子日志，核验合法率、覆盖率、延迟及与规则候选的重合度，完全不干预前端展示；
- 第二步为 **低权重接入阶段**（Low-Weight Recall）：将 `ann_score` 权重 $\alpha$ 设为 $0.10\sim 0.20$，由规则、画像与热门主导排序，并配齐超时或故障自动降级（Fallback）；
- 第三步为 **受控提权阶段**（Controlled Ramp-up）：待分桶数据证明 `ann_score` 与真实转化单调正相关且负反馈未上升后，逐步将 $\alpha$ 提升至 $0.35\sim 0.50$ 以上，同时全程保留可一键关闭 ANN 或回滚版本的热开关（Kill Switch）。

### 受控引导探索：制造可学习的边界曝光

为了打破旧规则与初版模型的曝光偏差，线上服务应在 $90\%\sim 95\%$ 坑位执行常规最优排序（Exploitation）的同时，预留 $5\%\sim 10\%$ 预算开展**受控引导探索**（Guided Exploration）。探索候选必须在通过全部安全护栏（`hard_filters`）的前提下，从内容相关但低曝光的新上架商品、ANN Top $100\sim 300$ 的中腰部边界商品、同类目跨风格商品以及双塔高分但旧规则低分的分歧候选中抽取。展示位置上坚持“Top 3 核心位保守稳健、Rank $4\sim 20$ 中腰部适度穿插探索”的原则，并在曝光日志中完整记录探索前后的位次变化：

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

## 多目标混合损失与四阶段演进路线图

双塔模型训练的核心损失采用批内对比学习（In-Batch Softmax）。设批次大小为 $B$，用户向量矩阵与正样本商品向量矩阵分别为：

$$
U=[u_1,\dots,u_B]^\top,
\quad
V=[v_1,\dots,v_B]^\top.
$$

经温度系数 $\tau$ 缩放的对数几率矩阵为 $Z=UV^\top/\tau$，对应的批内交叉熵损失定义为：

$$
\mathcal{L}_{\text{inbatch}}
=
-\frac{1}{B}
\sum_{b=1}^{B}
\log
\frac{\exp(u_b^\top v_b/\tau)}
{\sum_{j=1}^{B}\exp(u_b^\top v_j/\tau)}.
$$

为了支撑系统从冷启动弱监督平滑过渡至真实反馈主导，总训练损失由真实正反馈损失、曝光未动弱负例（及难负例）损失、伪用户匹配损失、规则蒸馏损失与商品内容对比损失动态加权构成：

$$
\mathcal{L}
=
w_{\text{real}}\mathcal{L}_{\text{real}}
+w_{\text{noact}}\mathcal{L}_{\text{exposed\_no\_action}}
+w_{\text{syn}}\mathcal{L}_{\text{synthetic}}
+w_{\text{distill}}\mathcal{L}_{\text{distill}}
+w_{\text{item}}\mathcal{L}_{\text{item}}.
$$

各损失项权重与线上策略随数据成熟度按以下四阶段演进表动态退火迁移：

| 演进阶段 | `w_real` | `w_noact (+hard)` | `w_syn` | `w_distill` | `w_item` | 在线 ANN 模式与核心目标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **阶段 0：零样本启动期** | `0.00` | `0.00` | `0.35` | `0.35` | `0.30` | Shadow 模式（或权重 $\le 0.10$），跑通日志与向量空间初始化（无规则系统时取 `0.60 syn + 0.40 item`） |
| **阶段 1：少量样本破冰期** | `0.20` | `0.10` | `0.25` | `0.25` | `0.20` | 低权重召回（`ann_score = 0.10~0.20`）+ `5%` 引导探索，开始引入真实反馈与弱负例 |
| **阶段 2：规模化可训练期** | `0.50` | `0.30` | `0.05` | `0.05` | `0.10` | 受控提权（`ann_score = 0.25~0.40`）+ `5%~10%` 探索，真实反馈与难负例（`0.20 noact + 0.10 hard`）接管主导 |
| **阶段 3：稳定自演进闭环** | `0.70+` | `0.15~0.25` | `0.00~0.05` | `0.00~0.05` | `0.05~0.10` | 主力个性化召回源，弱监督正式退出主训练（仅留作新模型冷启辅助），进入常态化重训与 A/B 迭代 |

将上述四阶段映射为约 6 周的工程推进节奏，每个阶段的交付里程碑与准出验收标准如下：

### 阶段 0：零样本启动期（第 1–3 周）

- **第 1 周**（基线与日志就位）：落地 `RecommendRequest / RecommendResponse` 接口契约、规则/热门/画像基线排序、`RankPolicyV1` 初始配置以及 `ExposureLogV1` 与 `InteractionEventV1` 埋点，确保任意推荐响应均可关联曝光日志、任意点击均可通过 `request_id + item_id` 回连曝光，且下架或不可售商品零泄漏。
- **第 2 周**（商品特征快照与质量门禁）：构建 `ItemFeatureV1` 离线导出管线、`ItemFeatureSnapshotManifestV1` 与自动化质量检查报告，确保主键唯一、缺失率可解释且新品具备完整内容特征。
- **第 3 周**（初版向量导出与 ANN Shadow）：先通过确定性特征投影或阶段 0 弱监督双塔导出 `ItemEmbeddingV1` 并构建 `ANNIndexManifestV1`，上线 ANN Shadow 查询链路，验证检索延迟、$100\%$ 候选合法率与近邻语义合理性。

### 阶段 1：少量样本破冰期（第 4–5 周）

- **第 4 周**（防穿越样本管线）：上线基于时间点快照的样本生成流水线，产出首批 `TrainingExampleV1` 与 `TrainingDatasetManifestV1`，完成零数据穿越审计与归因等待窗口校验。
- **第 5 周**（首版真实反馈混合双塔）：按阶段 1 混合权重训练 `ItemTower` 与 `UserTower`，替换确定性向量并以 `ann_score = 0.10~0.20` 低权重及 `5%` 探索比例灰度上线。准出要求为：`exposure -> interaction` 回连率达标、显式负反馈（hide/dislike）实时触发过滤或重罚、灰度期间负反馈率无抬升。

### 阶段 2：规模化可训练期（第 6 周及之后）

当主场景曝光与点击、收藏、加购、成交样本形成稳定规模后，引入同类目与跨通道难负例挖掘，将真实正负反馈总权重提升至 `0.80`（`0.50 real + 0.20 noact + 0.10 hard`），并将线上 `ann_score` 权重逐步上调至 `0.25~0.40`。准出要求为：`ann_score` 分桶与真实 CTR/转化率呈现严格单调正相关，线上 A/B 实验核心转化指标不劣于基线，新品与长尾覆盖率显著提升，且同 SPU 重复率可控。

### 阶段 3：稳定自演进闭环（常态化运营）

当 `曝光 -> 行为 -> 样本 -> 训练 -> 向量索引 -> ANN 召回 -> 再曝光` 闭环完全跑通后，伪用户合成与规则蒸馏退出主训练目标，系统进入周期性自动重训、成对索引发布与多路策略协同阶段。每次发布新版模型或索引前，必须逐项通过以下**十三项发布前检查清单**：

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

## 闭环后的持续训练、成对发布与反馈防污染

进入稳定闭环后，推荐系统的核心工程课题转向**如何在持续吸收新鲜反馈的同时维持线上服务的可控性**。成熟生产环境中的持续更新由五个不同频率的层次协同组成：

| 更新层次 | 更新对象 | 常见频率 | 核心目标 | 主要风险 |
| --- | --- | --- | --- | --- |
| **特征快照更新** | `ItemFeatureV1`、统计特征、用户画像 | 小时级到天级 | 让系统及时感知新商品、新价格、新库存与新行为 | 快照时间戳与训练样本时间窗口不一致 |
| **ANN Index 更新** | `ItemEmbeddingV1`、向量索引、过滤字段 | 天级或更高 | 将最新上架商品与最新表征纳入可召回池 | 在线 User Tower 与 Item Index 向量空间失配 |
| **全量重训**（Batch Retraining） | 双塔召回模型、排序模型 | 天级、周级 | 用最新闭环窗口数据整体刷新模型参数 | 历史曝光偏置累积放大、时间窗口配置漂移 |
| **热启动续训**（Continued Training） | 基于上一版模型检查点继续微调 | 小时级到天级 | 节省训练算力，保持跨批次向量空间方位相对稳定 | 继承旧模型局部偏差、遗忘长尾与冷门分布 |
| **近线/在线更新**（Nearline Update） | 用户短期兴趣状态向量、局部热点参数 | 秒级到小时级 | 即时捕捉会话内兴趣转移与突发热点 | 反馈回路急速放大，工程与风控复杂度陡增 |

### 全量重训与热启动续训流水线

全量重训（Full Retraining）是推荐系统的基准刷新手段，可按时间触发（每日凌晨/每周固定）、数据量触发（新增曝光或新品数跨过阈值）或质量触发（分桶单调性下降、漂移报警），其标准自动化流水线如下：

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

在日常高频迭代中，若采用基于上一版模型参数的热启动续训（Warm-Start Continued Training）以维持向量空间平滑，切忌仅喂入最近一天的新样本。为防止模型顺着旧偏置越走越窄或发生灾难性遗忘，续训数据集应由近期窗口、历史回放窗口与固定锚点集按比例混合：

```text
0.55 * recent_real_feedback（最近 1~7 天高权重样本，吸收新鲜反馈）
0.25 * replay_real_feedback（过去 30~90 天中权重回放样本，防止遗忘历史分布）
0.10 * recent_exposed_no_action（近期确认无动作的弱负例）
0.05 * item_contrastive_or_profile_regularizer（内容对比正则，保护新品冷启表征）
0.05 * anchor_set（固定审计与回归样本集，稳定向量空间坐标系）
```

同时，续训必须绑定一套包含新用户请求、新商品候选、长尾候选、历史负反馈样本以及“ANN 与规则高度分歧样本”的固定离线评估集；若整体命中率微升但新品覆盖或负反馈回归样本恶化，则严禁放行发布。

### 双塔与向量索引的原子级成对发布

双塔召回与普通单塔排序模型有一个本质区别：线上 User Tower 实时生成的用户向量，必须与 ANN 索引中预存的商品向量处于严格相同的几何空间（`online_user_tower.embedding_space_version == ann_index.item_embedding_space_version`）。任何一方单独升级都会导致内积分布彻底失真。因此，线上服务严禁零散配置模型路径或索引名，必须通过统一的发布清单 `CandidateModelReleaseV1` 将双塔权重、特征快照、商品向量导出批次、ANN 索引版本与允许配合的排序策略版本封装为原子发布单元：

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

每次发布严格遵循以下八步原子流水线：

```text
1. 离线训练并导出新版 User Tower 与 Item Tower 权重
2. 使用新版 Item Tower 批量导出全量商品向量 ItemEmbeddingV1
3. 构建新版本 ANN Index（此时不切入线上生产流量）
4. 在推理服务加载新 User Tower 并指向新 Index 运行 Shadow 流量验证
5. 核验候选合法率、覆盖率、近邻人工巡检与分数分位数分布
6. 按 release_id 执行小流量灰度（Canary）发布
7. A/B 实验护栏与核心指标验证通过后逐步全量放开
8. 热备保留上一版 model / index / policy 实例以支持秒级一键回滚
```

### 延迟标签窗口与反馈回路防污染

在持续重训中，不同深度行为的成熟周期差异显著。样本生成管线需在 `TrainingDatasetManifestV1` 中显式固化各类标签的成熟窗口：

```text
click label window: 30 minutes
cart label window: 24 hours
purchase label window: 7 days
no_action negative window: 12-24 hours
```

为了防止模型上线后陷入“只推自己历史推过的商品、越训越窄”的反馈回路污染，工程上需筑牢五道防线：常驻 $5\%\sim 10\%$ 引导探索流量以持续观测边界候选；完整记录 `logging_policy_id` 并监控各策略来源的样本占比；对头部爆款商品与超高频活跃用户施加训练降采样限幅；保留商品内容对比正则项防止长尾表征坍缩；以及坚持用多维分桶指标与长期多样性护栏替代单一的次日离线回放指标。

将工业界成熟系统的工程实践与本文方案对照，可以清晰界定各阶段的能力演进优先级：

| 工业界参考文献与系统 | 真实系统关注焦点 | 对本文落地路线的启发 |
| --- | --- | --- |
| Google TFX / Cloud MLOps | 持续训练（CT）、数据验证、模型验证、元数据谱系、门禁发布 | 重训必须固化为带 Manifest 与门禁的自动化流水线 |
| Google Rules of ML | 模型新鲜度、训练/服务特征一致性、上线前检测、渐进演进 | 先建立可观测的规则基线与端到端日志，再提升模型复杂度 |
| Pinterest Learned Retrieval | 双塔向量召回、自动重训、全量向量导出、ANN 索引线上同步 | User Tower 与 Item ANN Index 必须按 `release_id` 成对管理 |
| ByteDance Monolith | 秒级流式训练、无碰撞 Embedding 表、极速吸收实时反馈 | 在线流式更新参数属高阶能力，首版宜用近线兴趣池化替代 |
| Meta QuickUpdate | 在完整模型发布周期之间高频增量更新关键子集参数 | 部分参数高频发布可改善新鲜度，但强依赖成熟的发布风控底座 |
| Netflix RecSysOps | 围绕异常发现、预测、诊断与修复构建推荐系统运维闭环 | 必须将监控看板、Case Review 与熔断回滚纳入日常工程制度 |
| Eugene Yan Production ML | 反馈回路自我强化、数据漂移、监控盲区与系统性排障 | 重训前先核验数据、特征、日志与策略版本可信度 |

对于绝大多数工程团队而言，最稳健的演进次序是：

```text
Batch Retraining（批式重训）
  -> Warm-Start + Replay + 固定评估集续训
  -> Nearline 用户短期兴趣特征实时刷新
  -> CandidateModelReleaseV1 自动化成对发布
  -> Canary / A/B 实验与秒级回滚
  -> 最后再评估是否需要在线流式训练或高频部分参数更新
```

## 分层监控体系、离线评估边界与故障排查

生产环境推荐系统的稳定性由四层监控指标共同守护：

### 四层核心监控面板

1. **数据闭环监控**（训练前置底线）：

| 指标名称 | 监控意义 |
| --- | --- |
| `exposure_log_write_success_rate` | 曝光日志是否可靠落盘 |
| `interaction_log_write_success_rate` | 用户交互行为日志是否可靠落盘 |
| `exposure_interaction_join_rate` | 交互行为能否通过 `request_id + item_id` 准确回连曝光 |
| `request_id_missing_rate` | 核心关联主键是否缺失 |
| `item_feature_lookup_success_rate` | 在线排序时商品特征查得率 |
| `model_version_coverage` | 曝光与样本中模型/索引/策略三类版本号的覆盖率 |

2. **召回质量与内容分布监控**：

| 指标名称 | 监控意义 |
| --- | --- |
| `ann_candidate_valid_rate` | ANN 返回候选的在线可售合法率 |
| `offline_leakage_rate` | 下架、删除或不可售商品的泄漏率（必须为 $0$） |
| `coverage@K` / `category_coverage@K` | 推荐列表对全库商品与类目树的覆盖广度 |
| `new_item_coverage@K` | 新上架冷启动商品获得的曝光占比 |
| `long_tail_coverage@K` | 中长尾商品获得的曝光占比 |
| `duplicate_spu_rate` | 同一响应列表中同 SPU 商品重复率 |

3. **用户正负向反馈监控**：弱监督模型上线初期的典型隐患是“字面属性相关但真实意图错配”，导致负向反馈率上升，因此必须协同监控正向转化与负向护栏：

| 指标名称 | 监控意义 |
| --- | --- |
| `CTR` | 基础点击率 |
| `long_dwell_rate` | 商详页长停留率（过滤误触与标题党） |
| `like / favorite_rate` | 显式收藏与喜爱率 |
| `cart / purchase_rate` | 深层加购与成交转化率 |
| `hide / dislike / report_rate` | 显式负反馈与合规举报率（核心安全护栏） |
| `bounce_rate` | 会话级跳出率 |

4. **模型迁移与校准监控**：用于指导何时上调 `ann_score` 排序权重、何时退出冷启动弱监督：

| 指标名称 | 监控意义 |
| --- | --- |
| `real_feedback_sample_count` | 真实正负反馈样本累计规模是否充足 |
| `positive_negative_ratio` | 训练样本正负例比例是否稳定 |
| `per_scene_sample_coverage` | 各核心业务场景是否均有足够样本覆盖 |
| `per_category_sample_coverage` | 各主营一级/二级类目是否均有足够样本覆盖 |
| `ann_score_bucket_ctr` | ANN 得分分桶与真实点击/转化率是否呈严格单调正相关 |
| `teacher_student_agreement` | 蒸馏阶段是否成功吸收旧规则系统的基准能力 |
| `coverage_delta_vs_teacher` | 相比旧规则系统是否拓展了有效商品覆盖面 |

### 离线评估的能力边界与人工 Case Review 规范

由于历史日志受旧策略选择偏差与展示位次偏置影响，且随机负样本多为未曝光未知项，各项离线评估方法均存在明确的解释边界：

| 离线评估方法 | 可回答的问题 | 无法回答的问题 |
| --- | --- | --- |
| Item 近邻人工审查 | 商品内容表征空间是否语义合理 | 用户在当前场景下是否喜欢 |
| Teacher Agreement | 是否成功拟合旧规则系统的打分逻辑 | 线上表现是否超越旧规则系统 |
| Replay Hit Rate | 能否召回旧策略曾展示并获点击的商品 | 新发掘的未曝光商品在线上是否表现更好 |
| Score Bucket Analysis | 已曝光样本上分数是否与真实反馈单调相关 | 是否消除了历史策略的选择偏差 |
| Coverage Analysis | 是否覆盖了更多类目、长尾与新商品 | 新增覆盖的候选能否带来真实转化 |

在指标监控之外，团队应定期针对四类典型样本（① ANN 高分但规则低分、② 规则高分但 ANN 低分、③ 被 ANN 召回的新上架商品、④ 触发用户 `hide/dislike/report` 的曝光）抽取完整上下文字段开展人工巡检（Case Review）：

```text
user profile / context
recent positive items / recent negative items
candidate item features / candidate sources
ann_score / rule_score / final_score
rank_policy_version / model_version
exposure result (click / dwell / hide / no_action)
```

通过逐条审视“是否存在场景错配、价格带越界、同款重复刷屏、探索位次过高，或负反馈仅在策略层被惩罚而未被 User Tower 吸收”，能够直接指导下一轮特征与损失迭代。

### 线上异常症状排查表与迭代熔断红线

当线上出现质量或稳定性异常时，应遵循“先查过滤与日志、再查版本与策略、最后查模型”的顺序对照下表排查：

| 线上异常症状 | 优先排查模块 | 高频根因 |
| --- | --- | --- |
| 推荐结果大量为空 | `hard_filters`、候选池规模、ANN 超时率 | 过滤条件过严、向量库不可用、多路候选源未配兜底 |
| 下架商品出现在前端 | Guardrail、商品特征快照时效 | 快照延迟、过滤字段缺失、返回前未做实时状态二次校验 |
| 满屏皆为热门商品 | `RankPolicyV1`、蒸馏采样、ID 特征 | `hot_score` 权重过高、只蒸馏了 Teacher TopK、模型过度记忆 ID |
| 新商品完全无曝光 | Item Tower 特征、缺失指示器、探索策略 | 新品缺标签/文本向量、未加 `ctr_7d_missing`、无新品探索配额 |
| `hide / dislike` 率上升 | ANN 高分 Case、实时负反馈处理链路 | 内容相关但意图不符、用户点踩后未立即进入过滤与重罚 |
| `ann_score` 分桶无单调性 | 模型与索引版本号、训练样本与 Loss 权重 | User Tower 与 Index 版本错配、弱监督过重、假阴性负例过多 |
| A/B 实验指标剧烈波动 | 样本量、场景分桶、探索坑位配置 | 流量太小、多场景混杂统计、探索候选侵入了核心 Top 3 坑位 |
| 离线指标极好但线上溃败 | 样本生成时间戳边界、离线负采样方式 | `user_snapshot_at_t` 混入了曝光后行为（数据穿越）、离线负样本失真 |

一旦发生曝光日志写入异常、交互回连率暴跌、`request_id` 大面积缺失、商品快照不可复现、ANN 索引与导出向量行数不符、下架商品泄漏或负反馈率激增且无法定位来源等情况，必须**立即熔断新模型放量并暂停模型迭代**，待工程数据闭环彻底修复后再恢复算法实验。

## 结语

从零构建并演进一套简单可靠的推荐系统，核心在于把握“数据因果边界先于模型复杂度、系统职责解耦先于参数调优”的工程哲学。通过单池召回与五层架构解耦，系统将双塔模型聚焦于 $128$ 维语义与偏好匹配，将硬过滤、多路融合与去重惩罚交由可声明式热更新的 `RankPolicyV1` 管控；通过严格区分未曝光未知项（Unknown）与真实负反馈（Negative），配合内容对比预训练、伪用户属性对齐与全谱系分桶规则蒸馏，系统得以在零样本期安全起步；再经由 Shadow 验证、低权重召回、受控引导探索、防穿越时间点样本快照以及 `CandidateModelReleaseV1` 成对发布契约，推荐系统便能稳步完成从冷启动弱监督向真实反馈自演进闭环的跨越。

## 参考文献与延伸阅读

- [Yifan Hu, Yehuda Koren, Chris Volinsky. Collaborative Filtering for Implicit Feedback Datasets. ICDM 2008.](https://yifanhu.net/PUB/cf.pdf)  
  隐式反馈推荐的奠基工作，阐明了隐式行为应当拆解为偏好方向（Preference）与置信度权重（Confidence），不可直接等同于干净的显式正负标签。

- [Steffen Rendle et al. BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.](https://arxiv.org/abs/1205.2618)  
  个性化排序损失的经典基线，确立了隐式反馈推荐直接优化相对排序序关系的建模范式。

- [Tobias Schnabel et al. Recommendations as Treatments: Debiasing Learning and Evaluation. ICML 2016.](https://proceedings.mlr.press/v48/schnabel16.html)  
  从因果推断视角剖析推荐训练与评估中的选择偏差（Selection Bias），指出历史推荐日志源于旧策略筛选展示后的有偏观测。

- [Paul Covington, Jay Adams, Emre Sargin. Deep Neural Networks for YouTube Recommendations. RecSys 2016.](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)  
  工业推荐系统中候选生成（Candidate Generation）与精排（Ranking）两阶段漏斗架构的代表性论文。

- [Denis Baylor et al. Continuous Training for Production ML in the TensorFlow Extended TFX Platform. USENIX OpML 2019.](https://www.usenix.org/conference/opml19/presentation/baylor)  
  生产级机器学习持续训练（Continuous Training）流水线的代表工作，展示了如何通过数据验证、模型验证、元数据谱系与发布门禁保障重训可靠性。

- [Google Cloud. MLOps: Continuous delivery and automation pipelines in machine learning.](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)  
  从系统工程视角拆解机器学习流水线自动化、持续训练触发机制、模型评估门禁与灰度回滚体系。

- [Martin Zinkevich. Rules of Machine Learning: Best Practices for ML Engineering.](https://developers.google.com/machine-learning/guides/rules-of-ml)  
  Google 机器学习工程实践指南，强调先打通端到端数据管线与可观测的简单基线，再逐步提升模型复杂度。

- [Pinterest Engineering. Establishing a Large-Scale Learned Retrieval System at Pinterest.](https://medium.com/pinterest-engineering/establishing-a-large-scale-learned-retrieval-system-at-pinterest-eb0eaf7b92c5)  
  双塔向量召回的工业界落地实践，详述了自动重训、商品向量批量导出、ANN 索引构建及线上成对同步机制。

- [Jian Zhu et al. Monolith: Real Time Recommendation System With Collisionless Embedding Table. arXiv 2022.](https://arxiv.org/abs/2209.07663)  
  大规模实时流式推荐训练系统代表作，适合在批式闭环与近线兴趣池化成熟后作为高阶演进参考。

- [Dan Matam et al. QuickUpdate: Fast and Efficient Model Updates for Online Recommender Systems. NSDI 2024.](https://www.usenix.org/conference/nsdi24/presentation/matam)  
  探讨如何在完整模型发布周期之间高频增量更新关键子集参数，以兼顾模型新鲜度与发布稳定性。

- [Mohammad Saberian, Justin Basilico. RecSysOps: Best Practices for Operating a Large-Scale Recommender System. RecSys 2021.](https://doi.org/10.1145/3460231.3474620)  
  Netflix 推荐系统运维实践，将异常发现、诊断、巡检与自动降级修复纳入推荐系统全生命周期管理。

- [Eugene Yan. A Practical Guide to Maintaining Machine Learning Systems in Production.](https://eugeneyan.com/writing/practical-guide-to-maintaining-machine-learning/)  
  生产环境机器学习维护指南，深入讨论了反馈回路自我强化、数据漂移、监控盲区与系统性排障流程。

- [Maksims Volkovs, Guangwei Yu, Tomi Poutanen. DropoutNet: Addressing Cold Start in Recommender Systems. NeurIPS 2017.](https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems)  
  通过在训练期显式施加特征与 ID Dropout 模拟冷启动缺失状态，迫使深度网络充分利用内容特征泛化。

- [Lihong Li et al. A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010.](https://arxiv.org/abs/1003.0146)  
  推荐系统中探索与利用（Exploration vs. Exploitation）的经典上下文赌博机工作，奠定了受控探索与策略日志记录的理论基础。

- [Yuta Saito et al. Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation. 2020.](https://arxiv.org/abs/2008.07146)  
  系统阐述带策略日志反馈（Logged Bandit Feedback）与离策略评估（OPE）的数据规范。

- [Negative Sampling in Recommendation: A Survey and Future Directions.](https://arxiv.org/html/2409.07237v1)  
  全面梳理推荐系统中的负采样策略与假阴性（False Negative）风险控制。

---
title: "Graphiti / Zep 论文精读：双时态知识图谱、Agent 长期记忆与混合检索"
description: "从 Zep 论文与 Graphiti v0.29.2 源码双线精读三层记忆图、双时态事实、增量知识抽取、混合检索、重排策略和 Agent 长期记忆工程"
pubDate: "2026-07-16T18:33:25+08:00"
updatedDate: "2026-07-16T18:33:25+08:00"
tags:
  - "Paper Reading"
  - "Agent Memory"
  - "Knowledge Graph"
  - "Graphiti"
  - "Zep"
  - "Code Reading"
draft: false
---

大语言模型的上下文窗口越来越长，但“能放进去”不等于“值得每次都放进去”，更不等于模型已经拥有可靠的长期记忆。一个持续数月运行的客服 Agent 会反复接触客户偏好、合同状态、产品迁移、故障处置和人员变化；一个个人助理需要知道用户过去喜欢什么，也要知道某些偏好后来已经改变；一个业务 Agent 不仅要记住聊天，还要吸收工单、CRM 记录、JSON 事件和外部数据库更新。把所有历史原文重新塞进 prompt，成本、延迟和噪声都会随时间增长。只保存向量 chunk，又很难表达“谁与谁是什么关系”“这条事实什么时候有效”“后来为何失效”。

**Zep: A Temporal Knowledge Graph Architecture for Agent Memory** 试图把这个问题改写为动态知识维护问题。论文中的核心组件 Graphiti 将输入保留为 episode 证据层，同时抽取实体和事实，维护社区摘要，并为事实记录现实世界有效时间与系统写入时间。查询时，它不只做向量近邻，而是组合语义检索、全文检索、图遍历和多种重排策略，再把日期范围、事实、实体摘要压缩为 Agent 可消费的上下文。

这篇论文值得精读的原因，不在于“知识图谱比向量库高级”这种过度简化的口号，而在于它把几个经常被分开讨论的问题放在了同一条数据链路上：原始证据如何保留，事实如何增量抽取，重复实体如何合并，矛盾事实如何失效，历史如何审计，候选如何召回，最终上下文如何控制长度。官方开源项目 [Graphiti](https://github.com/getzep/graphiti) 又让我们能够检查这些概念在稳定版代码中如何落地，以及论文发布后新增了哪些能力。

精读后的结论先写在前面：

> Graphiti 的真正价值不是“把聊天记录写进 Neo4j”，而是把 Agent 记忆变成具有证据来源、事实生命周期和可配置检索路径的持续知识系统。它能降低全量上下文的成本并改善部分长期记忆任务，但不会自动消除 LLM 抽取错误、时间歧义、实体误合并、图膨胀和权限治理问题。

本文严格区分三类对象：论文评测的是 2025 年初的 **Zep 系统**；本文源码阅读固定在 Apache-2.0 的 **Graphiti `v0.29.2`**；当前商业产品 **Zep Context Lake** 则包含托管基础设施和专有能力。三者相关，但不是同一份部署物。本文没有启动 Neo4j、FalkorDB 或 Neptune，没有调用模型，也没有复现论文远程 Zep API 实验。

![Zep paper title and abstract](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-paper-title-abstract.png)

*论文题名、作者与摘要。Source: Rasmussen et al., arXiv:2501.13956v1, paper excerpt, CC BY-NC-SA 4.0.*

## 1. 一句话贡献：从“存消息”到“维护事实历史”

如果只用一句话概括论文，可以写成：**Zep 通过 Graphiti 把连续到来的对话与业务事件组织成带双时态、证据溯源和混合检索能力的知识图，使 Agent 能在有限上下文中访问长期、动态且可更新的记忆。**

这句话包含五个不能省略的限定。

第一，系统处理的是**连续到来的 episode**，不是一次性离线导入一批静态文档。episode 可以是消息、文本或 JSON；当前 Graphiti 还扩展了直接事实三元组。增量性意味着每次写入都必须面对已有实体、已有事实和已有社区，而不是重新构建整张图。

第二，图中保留**原始证据层**。语义实体和事实不是脱离来源的“真理”，而是从某个 episode 抽取出的断言。出现错误时，系统需要能够回到原文，而不是只留下一个无法解释的 embedding 或摘要。

第三，事实具有**生命周期**。例如“Alice 在 Acme 工作”可能从 2024 年 3 月有效，到 2025 年 2 月失效；系统却可能到 2025 年 3 月才收到离职消息。现实时间和写入时间不同，单个 `updated_at` 无法同时表达二者。

第四，查询不是单路向量检索。人名、工单号和产品编号需要词面匹配，语义改写适合向量相似，关系问题需要图邻域，多路候选还要经过融合和重排。

第五，输出不是整张图，而是**为当前问题构造的有限上下文**。真正进入 LLM 的仍然是文本。图负责维护和选择，语言模型负责基于选出的事实生成答案。

因此，Graphiti 既不是数据库本身，也不是 Agent 框架的全部。它处在业务事件与模型上下文之间，是一个持续演化的记忆索引和事实维护层。

## 2. 论文信息、版本与开放边界

| 项目 | 内容 |
| --- | --- |
| 题名 | Zep: A Temporal Knowledge Graph Architecture for Agent Memory |
| 作者 | Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef |
| 机构 | Zep AI |
| arXiv | `2501.13956v1` |
| 提交日期 | 2025-01-20 |
| PDF | 12 pages, 3 tables |
| DOI | `10.48550/arXiv.2501.13956` |
| 论文许可 | CC BY-NC-SA 4.0 |
| 后续版本 | 截至本文写作时没有 v2 |
| 开源引擎 | [getzep/graphiti](https://github.com/getzep/graphiti)，Apache-2.0 |
| 源码基线 | [`v0.29.2`](https://github.com/getzep/graphiti/tree/v0.29.2)，commit `ff7e29ccd127d8d9721b5cbb2163a6407ef915fe` |
| 稳定版发布时间 | 2026-06-08 |
| 预发布边界 | `0.30.0pre5` 与后续 `main` 仅作演进提示，不作为稳定 API |

论文元数据明确写着“12 pages, 3 tables”。最终 PDF 也确实没有正式编号的 Figure。网上常见的 Graphiti 架构动画、简单图示和官网 Context Graph 图片都来自论文之后的仓库或产品文档，不能倒写成“论文 Fig. 1”。本文使用三类视觉材料：论文标题、上下文模板和三张表的原排版裁切；`v0.29.2` 标签下的官方仓库素材；以及明确标注“本文整理”的结构图。

许可也必须分开。论文采用 CC BY-NC-SA 4.0，本文的论文裁切按该许可注明作者和来源。Graphiti 仓库是 Apache-2.0，仓库中的 SVG/GIF 依据代码仓库许可标注。当前商业 Zep 网站上的产品材料不自动继承这两种许可，因此本文不复制商业站点图片。

论文发布时描述的是一个可通过远程 API 使用的 Zep 记忆服务，并把 Graphiti 称为核心的时态知识图引擎。开源仓库后来加入多个图后端、ontology、自定义实体与边类型、bulk ingestion、saga、MCP server、tracing 等能力。阅读代码可以帮助我们理解工程落点，但不能证明 2025 年 1 月论文实验已经使用 2026 年 6 月稳定版的全部实现。

## 3. Zep、Graphiti 与 Context Lake：三个名称，三种边界

官方当前文档把 Graphiti 和 Zep 分得很清楚：Graphiti 是开源的时态上下文图构建引擎；Zep 是托管的企业级 Context Lake，基于 Graphiti 思路和专有后端提供产品化能力。论文中的“Zep”则是当时被评测的记忆系统。为了避免同名漂移，可以按三层理解。

**论文中的 Zep 系统**包含 ingestion、Graphiti 图维护、检索、上下文构造和远程服务。论文报告的 DMR、LongMemEval 准确率、延迟和 token 数都属于这套服务的实验结果。它不是一个只下载 Python 包就能复刻的单进程程序。

**开源 Graphiti 引擎**提供图构建和检索核心：`Graphiti` 类负责 orchestrate episode 写入、实体与事实抽取、resolution、时态失效、搜索和社区维护；driver 负责图数据库；LLM、embedder 和 cross encoder 通过适配器注入。MCP 和 FastAPI server 是可选集成壳。开源代码给了机制透明度，但没有自动提供商业服务的可用区、SLA、权限模型、数据治理和运维团队。

**当前 Zep Context Lake**是托管产品。它可能提供更成熟的伸缩、隔离、监控、身份治理和产品 API，也可能运行与开源 Graphiti 不同的专有存储或优化。除非官方文档明确说明，不能把商业产品的 benchmark、SLA 或功能视作本地 Graphiti 默认能力；反过来，也不能用开源仓库中的某个实验性 server 代表 Zep 的生产架构。

这种边界直接影响工程决策。选择 Graphiti 意味着团队要自己承担模型供应商、图数据库、队列、重试、备份、扩缩容、鉴权和数据删除。选择托管 Zep 则把部分责任交给服务商，但要评估数据驻留、供应商依赖、成本和 API 边界。论文并没有替你做这个选择。

## 4. Agent Memory 到底在解决什么

“记忆”容易被拟人化。对工程系统来说，它至少包含四个不同问题：保存历史、从历史中找出相关内容、识别历史已经改变、把相关内容以受控形式提供给模型。只完成第一项，不等于拥有长期记忆。

### 4.1 长上下文不是免费数据库

把全部历史拼到 prompt 有三个显性成本。token 数会持续增长，推理延迟和费用随之增加；大量无关内容稀释注意力，模型可能在“needle in a haystack”场景中漏掉关键事实；历史中互相矛盾的陈述没有显式生命周期，模型只能自己猜哪一条更新。

更隐蔽的问题是治理。全量历史可能包含已经过期的个人信息、需要删除的敏感字段、系统内部指令和恶意 prompt injection。把原始上下文无差别送给生成模型，扩大了暴露面。

### 4.2 静态 RAG 不理解事实变化

传统文档 RAG 通常假设文档相对稳定：切 chunk、嵌入、入库、查询。对“Acme 的退款政策是什么”这类文档问题已经有效，但持续业务事件会不断改变实体状态。一个旧 chunk 和一个新 chunk可能同时被召回，向量距离不能表达哪一个在问题所指时间有效。

### 4.3 摘要会累积信息损失

递归摘要压缩成本低，却会把细粒度证据折叠成不可逆的概括。摘要器没有预先知道未来会问什么，今天看似无关的日期、否定词或关系，在下个月可能成为关键。多轮摘要还会放大早期误差。Graphiti 的 episode 层试图保留原始输入，把实体摘要和社区摘要视为检索辅助，而不是唯一事实存储。

### 4.4 记忆必须能承认“曾经正确”

现实系统不能把更新简单理解为覆盖。用户曾经居住在巴黎、后来搬到柏林，这两个事实都可能对不同时间的问题有用。合同曾经处于草案、后来签署，也需要保留转换历史。双时态图的目标不是让所有事实永远“当前有效”，而是保存何时有效、系统何时知道以及何时不再采用。

## 5. 知识图形式化：节点、边与 incidence function

论文把整个图写为：

$$
\mathcal{G} = (\mathcal{N}, \mathcal{E}, \phi),
$$

其中 $\mathcal{N}$ 是节点集合，$\mathcal{E}$ 是边集合，$\phi: \mathcal{E}\rightarrow \mathcal{N}\times\mathcal{N}$ 是 incidence function，给每条边指定源节点和目标节点。这个定义看似基础，却提醒我们：事实并不是一段漂浮的文本，而是连接具体实体的有向关系；episode 与实体之间的来源关系、实体与社区之间的成员关系，也都需要显式边。

Zep 将图拆成三个互相连接的子图：

$$
\mathcal{G}_e \subseteq \mathcal{G},\qquad
\mathcal{G}_s \subseteq \mathcal{G},\qquad
\mathcal{G}_c \subseteq \mathcal{G},
$$

分别对应 episode、semantic entity 和 community。三层不是三个彼此独立的数据库。episode 提供证据，semantic entity 提供可计算的关系与时态事实，community 提供主题级抽象。查询可能从一个实体出发沿图遍历，也可能先命中 episode，再通过来源边回到事实。

![Three-tier temporal memory graph](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-diagram01-three-tier-graph.svg)

*本文根据论文 Sec. 2 的三层图定义整理，非论文原图。三层之间的虚线表示 provenance 与 community membership 等跨层连接。*

这个分层的关键价值是把“信息保真”和“查询效率”分开。episode 层可以保留完整输入，semantic 层用结构化事实支持精确关系查询，community 层在需要全局背景时提供摘要。如果系统只保留语义层，一次错误抽取就可能永久替代证据；如果只保留 episode 层，关系和时间查询又退化为每次让 LLM重新阅读大量文本。

它也有代价。同一条输入会以原文、实体、事实、embedding、社区摘要多种形式存在，存储和维护成本增加。三层之间任何链接错误都会影响检索：episode 关联错实体，事实就难以审计；实体错误归入社区，主题摘要会被污染。因此，图分层不是消除错误，而是让错误有机会被定位和修复。

## 6. Episode Subgraph：非损失证据层的准确含义

episode subgraph $\mathcal{G}_e$ 保存进入系统的原始事件。论文列出三种输入：对话消息、任意文本和结构化 JSON。当前 `v0.29.2` 的 `EpisodeType` 仍有 `message`、`text`、`json`，并增加了 `fact_triple`。每个 episode 至少需要名字、正文、来源描述、reference time 和分组信息；当前代码还支持 episode metadata。

“non-lossy”不能被理解成“系统永远保留一切且绝不变化”。更准确的说法是：**语义抽取不会替代原始 episode，图中保留从实体和事实回到来源 episode 的路径。** 这样做有三个直接收益。

第一，答案可以附带 provenance。面对“为什么系统认为 Alice 已离职”，运维或用户可以回到导致失效判断的消息，而不是只看到一个模型生成的结论。

第二，抽取逻辑升级后可以重放。新 prompt、新实体 schema 或新模型可能从同一原文抽出不同事实。只要 episode 仍在，就有机会离线重建语义层。

第三，检索可以返回原文。某些问题需要措辞、上下文或未结构化的细节，实体摘要和事实三元组不足以回答。episode 是结构化图与原始语料之间的双向索引。

但 episode 不是无限日志的免责条款。生产系统仍要定义数据保留周期、删除和合规流程。若用户要求删除个人信息，不能因为“provenance 很重要”就忽略删除义务；相反，需要设计 episode、派生实体、事实、embedding 和社区摘要的级联清理。Graphiti 提供 `remove_episode()` 等基础操作，不等于已经替应用完成监管语义。

当前代码中的 `group_id` 常用于把一组 episode 和实体限制在同一图分区。它有助于查询过滤和数据库选择，但不是权限系统。知道某个 `group_id` 的调用方是否能访问该组，仍应由 API gateway、租户映射和授权层决定。

## 7. Semantic Entity Subgraph：实体、事实与来源

semantic entity subgraph $\mathcal{G}_s$ 是最像传统知识图的部分。实体节点代表人物、组织、地点、产品、事件等对象；实体边代表事实，例如 `Alice --works_at--> Acme`。节点通常有名称、摘要和 embedding，边有自然语言 fact、关系名、时间字段、embedding 以及来源 episode 列表。

论文刻意把事实表示为带文本内容的边，而不是要求所有关系都来自固定 ontology。这让系统能够快速接收开放域对话：LLM 可以生成“prefers decaf coffee”“blocked by ticket INC-42”“plans to visit Kyoto”之类关系，不必先由知识工程师定义完整 schema。当前 Graphiti 又支持 Pydantic entity/edge types，把开放抽取和领域约束放在同一框架里。

开放关系的优势是覆盖面，风险是规范化不足。同一个意思可能出现 `works_at`、`employed_by`、`is an employee of`；不同事实可能被错误合并；谓词文本过长会降低图查询可预测性。对高风险业务，完全 schema-free 通常不够，需要至少固定核心实体类型、标识符和关键关系，再允许长尾属性保持开放。

论文中的事实主要连接一对实体。现实陈述可能涉及三元以上关系，例如“供应商 A 在合同 C 下向客户 B 提供产品 P”。把它拆成若干二元边可能丢失事件级绑定。可行方案是把合同或事件本身建成实体节点，再让参与者分别连接到它。Graphiti 的灵活实体类型允许这样做，但建模质量仍取决于调用者的 ontology 和抽取 prompt。

语义层必须保留 source episode。否则所谓“知识”只是模型断言。Graphiti 的边记录 episode UUID 列表，允许一个事实得到多个来源支持。多来源不自动等于真实：相同错误可能被重复复制，两个来源也可能互相引用。生产检索可以把来源数量作为证据特征，却不应把它当作无条件置信度。

## 8. Community Subgraph：主题摘要不是事实替代品

community subgraph $\mathcal{G}_c$ 在实体图上构建更粗粒度的主题结构。论文采用 label propagation 一类社区发现方法，把连接紧密的实体聚成组，再由 LLM 为社区生成名称和摘要。一个社区可能对应“Acme 迁移项目”“客户退款事件”“Alice 的旅行计划”等主题。

社区层解决的是全局背景问题。直接返回若干局部事实可能无法解释一个长期项目的总体脉络；把整个邻域都交给模型又太长。社区摘要提供一个压缩入口，并能作为检索候选或上下文补充。

论文提出增量维护：新实体加入后，社区标签和摘要可以局部更新；同时周期性执行更完整的刷新，减少长期增量更新造成的漂移。这个设计承认了一个现实问题：社区不是稳定真相。图的连边变化、误抽取和重复实体都会改变聚类，摘要也可能随着成员变化而过时。

社区摘要尤其容易被误当作事实。它是对一组节点的语言压缩，可能省略时间范围、例外和否定。安全做法是将社区摘要用于导航和背景，而把关键结论锚定到事实边和 episode。若答案来自社区摘要，系统也应保留可追溯的成员节点，而不是只展示一段不可验证的概括。

从成本看，社区维护会引入额外图算法和 LLM 调用。小规模个人记忆未必需要每次写入都更新社区；百万实体业务图则需要考虑全量 label propagation、摘要刷新和并发写入的资源。Graphiti 提供 `build_communities()` 等入口，但调度频率、增量策略和失败恢复仍是部署方的责任。

## 9. 双时态模型：事实有两条时间轴

Zep 论文最重要的设计之一，是将时态知识图中的 valid time 和 transaction time 引入 Agent memory。用论文符号表示，$T$ 是事实在现实世界中有效的时间线，$T'$ 是事实在系统中的事务时间线。

假设用户在 2025 年 2 月 1 日从巴黎搬到柏林，但到 2025 年 2 月 10 日才在对话中告诉 Agent。对于“用户住在柏林”这条事实：

- `valid_at` 可以是 2025-02-01，即现实状态开始成立的时间；
- `created_at` 是 2025-02-10，即系统创建该事实边的时间；
- 如果后来得知用户于 2025-06-01 搬到罗马，柏林事实的 `invalid_at` 是 2025-06-01；
- 系统在 2025-06-05 收到消息并关闭旧边时，旧边的 `expired_at` 是 2025-06-05。

论文将这四类时间写为 $t_{valid}$、$t_{invalid}$、$t'_{created}$ 和 $t'_{expired}$。当前 `EntityEdge` 中可以看到对应字段 `valid_at`、`invalid_at`、`created_at`、`expired_at`，还保留用于时间解析的 `reference_time`。它们回答的是不同问题：

| 问题 | 应看哪条时间轴 |
| --- | --- |
| 2025 年 2 月 3 日用户住在哪里？ | valid time $T$ |
| 系统在 2025 年 2 月 8 日知道什么？ | transaction time $T'$ |
| 哪条消息让系统修改了判断？ | provenance + transaction time |
| 当前应采用哪条事实？ | 未失效的 valid interval 与最新事务状态 |

![Bitemporal fact lifecycle](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-diagram02-bitemporal-lifecycle.svg)

*本文根据论文的双时态字段整理，非论文原图。现实有效区间与系统持有区间可以错位。*

双时态的价值是支持“当时世界如何”和“当时系统知道什么”两类查询。后者对审计、回放和事故分析尤其重要：如果 Agent 在某天给出错误建议，我们需要知道它当时可见的是哪些事实，而不是只看今天已经修正的图。

它的困难在于自然语言时间并不总是明确。“我上周换了工作”“从下个月开始”“最近不喝咖啡了”都依赖 episode 的 reference time、时区和上下文。LLM 解析可能产生伪精确日期。生产系统应保存原始时间表达、解析置信度或精度等级，并在不确定时使用开放区间，而不是强行填满四个时间戳。

## 10. 事实失效：历史关闭，不是物理删除

当新 episode 中出现与现有事实矛盾的断言，Graphiti 会尝试找出相关旧边，设置其失效时间，并写入新事实。这里最容易被误写成“Graphiti 删除了旧事实”。实际上，论文强调的是时态失效：旧边仍然存在，仍可通过历史时间和来源 episode 审计，只是不再作为当前有效事实。

考虑三种更新：

1. “Alice 在 Acme 工作”之后出现“Alice 已离开 Acme”。这是明确的终止。
2. “Alice 喜欢拿铁”之后出现“Alice 最近更喜欢手冲”。两者可能并存，也可能是偏好替换。
3. “项目发布日是 5 月 1 日”之后出现“发布推迟到 5 月 15 日”。这是同一属性的新值覆盖旧值。

LLM 必须判断新事实是否使旧事实失效、失效从何时开始、是否只在某个条件下成立。这个任务比实体抽取更难，因为矛盾是语义和时间共同决定的。“Alice 在 Acme 任顾问”和“Alice 离开 Acme 全职岗位”可能不冲突；“不再住在巴黎”不代表从未住过巴黎。

因此，事实失效应该被视为一种**模型生成的变更建议**，而非不可审计的数据库 mutation。高风险应用可以记录：触发 episode、被关闭边、候选新边、模型版本、prompt 版本和置信信号；对合同、医疗或安全事实还可加入人工审核。

物理删除属于另一条治理流程。若 episode 因隐私请求被删除，关联事实是否只失效还是彻底清除，取决于法律与产品政策。时态历史保留不能凌驾于数据删除要求。Graphiti 的历史模型提供表达能力，不提供完整的数据保留制度。

## 11. 实体抽取与反思：更多上下文也会带来更多误差

论文的增量抽取不是只看当前消息。它将当前 episode 与之前四条消息放入实体抽取 prompt，并加入 speaker entity，帮助模型解析代词和省略。例如：

```text
User: I met Maya yesterday.
Assistant: How did it go?
User: She agreed to lead the Atlas migration.
```

只看最后一句无法确定 `She` 和 `Atlas` 的类型。短窗口可以恢复局部共指，同时避免每次读取整段会话。论文还使用 reflection，让模型检查第一次抽取是否遗漏实体，再补充结果。

这种设计有三个边界。

**窗口是经验参数。** `n=4` 在论文数据上可行，不代表所有对话都够用。长距离共指、跨会话身份和业务文档引用可能超出窗口。扩大窗口会增加 token 和干扰，也可能让模型把更早的同名实体错误合并。

**reflection 增加召回，也增加幻觉机会。** 第二次询问“是否遗漏实体”会鼓励模型补全。对抽取任务，补全一个不存在实体往往比漏掉低价值实体更危险。可以通过 schema、最小文本证据、offset 或 quote span 约束降低风险。

**speaker entity 是强先验。** 消息中的“I”通常指 speaker，但转述和引用会打破这个规则。系统必须区分 author、quoted speaker 和 mentioned person，不能把所有第一人称都挂到会话用户。

当前 `v0.29.2` 的 prompt 和 maintenance 模块把抽取步骤拆开，便于替换模型和类型约束。静态阅读能看到结构，却不能证明某个模型在你的语料上有足够准确率。上线前必须对实体 precision、recall、同名误合并和敏感属性抽取进行独立评测。

## 12. 实体消歧：先召回候选，再让模型做 resolution

增量系统最先遇到的图膨胀问题，是同一实体被重复创建。`Acme`、`Acme Corp.`、`ACME` 和“客户公司”可能指向同一组织；两个都叫 Alex 的人也可能完全不同。论文采用候选召回加 LLM resolution 的两阶段路径。

候选召回同时使用向量相似和全文检索。向量适合处理别名、改写和上下文语义，全文检索适合精确名称、缩写和标识符。论文实体 embedding 为 1024 维。召回后，LLM 比较新实体与候选的名称、摘要和上下文，决定复用已有节点还是创建新节点，并可能更新节点名称与摘要。

这个流程比单纯 embedding threshold 更灵活，但它仍然有典型失败模式：

- **误合并**：两个同名客户被合成一个节点，后续事实互相污染；
- **漏合并**：同一组织因别名创建多个节点，检索结果分散；
- **摘要自增强**：错误事实进入摘要后，后续 resolution 以错误摘要作为证据，形成反馈回路；
- **跨租户碰撞**：若查询缺少可靠分区，两个租户的同名实体可能成为候选；
- **身份漂移**：人员改名、公司并购和产品改版既可能是同一实体演变，也可能需要新节点与关系。

因此，领域系统应优先使用稳定业务 ID。客户 ID、工单号、仓库 URL 和合同 ID 比名称 embedding 更可靠。LLM resolution 应处理缺少 ID 的长尾，而不应替代已有主键。自定义 Pydantic entity types 可以把业务标识设为属性，但 Graphiti 不会自动知道哪个字段是你的唯一键。

误合并通常比漏合并更难修。漏合并可以后续聚合，误合并会把事实、社区和时间线混在一起。对高风险实体，阈值和 prompt 应倾向保守，并提供人工拆分和重建派生边的工具。

## 13. 事实抽取与去重：关系文本不是数据库约束

实体 resolution 完成后，系统从 episode 中抽取实体间事实。论文事实包括自然语言描述、源实体、目标实体、时间信息和 episode 来源。抽取后还要与同一实体对上的现有事实比较，以避免重复并识别矛盾。

“同一实体对”是重要的计算约束。若 `Alice -> Acme` 已有十条边，新事实优先在这组边中做 resolution，而不是与全图事实逐一比较。这降低候选规模，却也假设事实的两个端点已经正确。如果实体先被误合并，事实 resolution 会在错误候选集里继续做决定。

事实去重不应只比较 predicate 字符串。例如：

```text
Alice is employed by Acme.
Alice works at Acme.
Alice joined Acme as a contractor.
```

前两句可能等价，第三句可能是更具体的角色，也可能与前两句同时成立。模型要判断是 duplicate、update、contradiction 还是 independent fact。Graphiti 用 LLM 和 embedding 辅助，而不是靠字符串规则。

论文的二元边模型还会把复合事件压缩。例如“Acme 于 3 月 1 日任命 Alice 为 Project X 的负责人”至少涉及组织、人物、项目、角色和日期。如果只写 `Alice leads Project X`，任命主体可能丢失。领域 ontology 可将任命建成 Event 实体：`Acme --issued--> Appointment`、`Appointment --appoints--> Alice`、`Appointment --for--> Project X`。这增加节点数，却保留 n-ary 语义。

事实 resolution 的工程验收不能只看图是否“看起来合理”。需要构造带已知更新链的样本，分别测 duplicate precision、contradiction precision、时间边界准确率和 provenance 完整率。尤其要检查否定、计划、假设和转述：“Alice 可能加入 Acme”不能当作已发生事实，“Bob 说 Alice 已离职”也应保留信息来源语义。

## 14. 检索形式化：Search、Reranker、Constructor

论文把查询流程写成三个函数的复合：

$$
f(\alpha)=\chi\bigl(\rho(\varphi(\alpha))\bigr)=\beta.
$$

$\alpha$ 是输入查询；$\varphi$ 是 Search，从图和索引中找候选；$\rho$ 是 Reranker，对候选融合、排序或去冗余；$\chi$ 是 Constructor，把选中的事实、实体和社区组织成最终上下文 $\beta$。这个拆分比“调用一个 search API”更有解释力，因为三阶段的失败含义不同。

Search 失败意味着关键证据没有进入候选集，后续再强的重排也救不回来。Reranker 失败意味着证据在候选中，却被低质量项挤出 top-k。Constructor 失败则可能是证据已经选中，但日期、来源或实体摘要的组织方式让 LLM误解。

这种函数分解也给评测提供了层次：

1. 候选层测 fact recall、episode recall 和来源覆盖；
2. 重排层测 nDCG、MRR、top-k precision、多样性和重复率；
3. 构造层测 token 数、事实冲突、日期完整率和答案正确率；
4. 端到端层再测任务准确率、延迟、成本和无答案行为。

仅报告最终问答准确率，很难判断改进来自图、检索、prompt 还是更强答案模型。论文在 LongMemEval 上展示端到端收益，却没有公开足够的中间召回快照和完整评测脚本。这是后续复现需要补的证据。

当前 Graphiti 的 `search()` 是方便入口，默认走预设 recipe；更底层的 `search_()` 接收 `SearchConfig` 和 `SearchFilters`，允许分别控制边、节点、episode 和社区搜索。源码边界说明了一个关键事实：Graphiti 不是只有一种固定“混合检索算法”，而是一组可组合策略。

## 15. 候选召回：语义、词面和图结构各管一类问题

论文候选检索组合三种信号。

### 15.1 Cosine similarity

对查询 embedding $q$ 和候选 embedding $x$，余弦相似度为：

$$
\operatorname{cos}(q,x)=\frac{q\cdot x}{\lVert q\rVert_2\lVert x\rVert_2}.
$$

它适合“用户偏好的饮品是什么”与“他通常喝哪种咖啡”这类语义改写，也能找名称不同但描述接近的实体。它不擅长严格日期、编号和否定，并且依赖 embedding 模型的语言和领域覆盖。

### 15.2 BM25 / full-text search

全文检索对 `INC-2048`、`Project Atlas`、邮箱、产品 SKU 和人名更可靠。词面召回还能在 embedding 将两个相似实体混淆时提供精确候选。Graphiti 的具体全文能力依赖 driver：Neo4j、FalkorDB 和 Neptune/OpenSearch 的索引语义、分析器和排序并不完全相同。

### 15.3 Breadth-first search

BFS 从中心节点沿图边扩展，适合“与 Alice 同项目的人”“某工单关联的客户和产品”这类结构问题。图距离不是语义相关性的同义词：一个高度连接的 hub 节点可能把大量噪声带入邻域；方向、边类型和时间有效性都应参与过滤。

多路召回的目标不是让每种方法都返回同一批结果，而是覆盖不同失败面。一个实用 query plan 可以先用全文锁定实体，再以该实体为中心做 BFS，同时用向量找语义相关事实。若直接把三路 top-k 无限制合并，候选数量和重排成本会快速增长。

Graphiti 的过滤器必须在检索前后都考虑。`group_id`、实体类型、边类型、时间范围和 source scope 若只在最终结果阶段过滤，可能浪费大量图遍历和向量查询，也可能造成跨分区候选暴露。`v0.28.2` 曾强化 search filter 的 Cypher 注入防护，说明动态查询拼接本身就是安全面，而不只是相关性问题。

## 16. 重排：RRF、MMR、图距离和 Cross Encoder

候选来自不同分数空间，不能直接把 BM25 分数、cosine 和 BFS 距离相加。论文和当前实现提供多种重排方法。

**Reciprocal Rank Fusion** 只使用各列表名次。其直觉形式为：

$$
\operatorname{RRF}(d)=\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)},
$$

其中 $R$ 是检索器集合，$k$ 是平滑常数。RRF 对不同分数量纲鲁棒，适合作为默认混合融合。它不理解候选内容，只奖励在多路列表中排名靠前的项。

**Maximal Marginal Relevance** 在相关性与多样性之间平衡。直觉上：

$$
\operatorname{MMR}(d)=\lambda\operatorname{Rel}(d,q)-(1-\lambda)\max_{d'\in S}\operatorname{Sim}(d,d').
$$

它能避免 top-k 全是同一事实的近义复述，但若 $\lambda$ 太低，也可能为了“多样”引入低相关结果。

**Episode mentions** 按事实被多少 episode 支持或提及排序。它能奖励反复出现的稳定信息，却会偏向高频旧事实，压低刚发生但非常重要的更新。营销文案重复出现十次也不比一次签署合同更真实。

**Node distance** 从指定中心节点计算图距离，适合已知主体的关系查询。它依赖正确中心和图结构，无法替代语义匹配。

**Cross encoder** 将查询与候选成对输入模型，通常能获得更精细相关性，但推理成本随候选数增加。当前 Graphiti 默认可注入 cross encoder；简单 `search()` recipe 也可以只使用 RRF，不需要每次让生成式 LLM参与排序。把“Graphiti 搜索没有 LLM in the loop”理解为所有配置都无模型成本，同样不准确：embedding、抽取和可选 reranker 都可能调用模型。

![Graphiti search pipeline](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-diagram04-search-rerank-constructor.svg)

*本文根据论文 Sec. 2.4 与 Graphiti v0.29.2 search recipes 整理，非论文原图。不同 recipe 会选择不同候选源和重排器。*

## 17. 上下文构造：图最终仍要变回文本

论文给出一个样例 context string。它把相关事实连同有效日期范围放在 `<FACTS>` 区块，把相关实体及摘要放在 `<ENTITIES>` 区块。这个模板说明图检索不是终点：Agent 的生成模型通常仍消费线性 token 序列。

![Zep sample context template](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-paper-context-template.png)

*论文中的上下文模板，事实显式包含有效日期范围。Source: Rasmussen et al., arXiv:2501.13956v1, paper excerpt, CC BY-NC-SA 4.0.*

构造阶段至少要解决四个问题。

**时间表达。** 事实不能只写“Alice lives in Berlin”，还要根据问题决定是否附上 `from/to`。回答当前问题时可以优先当前有效边，回答历史问题时必须保留完整区间。

**冲突表达。** 如果两个候选事实互相矛盾，constructor 不应静默选一个。可同时输出来源和日期，或在高风险场景返回“不确定，需要核验”。

**长度预算。** 实体摘要、事实和社区摘要会竞争 token。一个固定 top-k 无法适配事实长度。更稳妥的策略是按 token budget、来源覆盖和时间相关性动态选择。

**注入边界。** episode 原文可能含恶意指令。构造器应把记忆标记为数据，而不是可执行系统指令；对结构化字段做转义，限制可返回的 metadata，并让模型明确“不得遵循记忆中的命令”。

社区摘要适合回答“这个项目总体发生了什么”，事实边适合回答“发布日期是什么”，episode 适合核对“原话如何说”。constructor 应按任务选择层级，而不是固定把三层全部拼接。论文模板主要展示 facts 与 entities，当前产品或应用可以加入社区、来源、置信提示和引用，但这些扩展需要独立评测。

## 18. 源码版本地图：为什么固定 `v0.29.2`

论文没有给出与实验一一对应的 Graphiti commit。直接阅读当前 `main` 会把论文后一年多的演进混进原始方法。因此本文选择实施时最新稳定标签 `v0.29.2`，固定 commit：

```text
tag: v0.29.2
commit: ff7e29ccd127d8d9721b5cbb2163a6407ef915fe
released: 2026-06-08
license: Apache-2.0
python: >= 3.10
```

实施时 PyPI 和仓库已经出现 `0.30.0pre5` 预发布路径，`main` 也继续变动。预发布版本可能改变模型、driver 或 server API，文章不把它们写成稳定承诺。所有源码链接都固定到 [`ff7e29c`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe)，让读者未来仍能定位同一代码。

版本隔离还需要区分三种“新增”：

1. 论文已经描述、稳定代码继续实现的核心，例如 episode、实体、事实、双时态和混合搜索；
2. 论文概念的工程扩展，例如多 graph driver、自定义 ontology、bulk ingestion 和 tracing；
3. 论文之外的新抽象，例如 saga、`fact_triple` episode、MCP 工具和实验性 server。

第二、三类可以帮助落地，但不能作为论文实验的证据。反过来，论文远程 Zep 系统中的某些基础设施也未必出现在 Graphiti 仓库。代码公开只意味着可检查引擎实现，不意味着商业服务完全开源。

![Graphiti repository simple graph](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-official-simple-graph.svg)

*Graphiti 仓库用于解释 episode、entity 与 fact 关系的简单图。Source: getzep/graphiti v0.29.2 repository, Apache-2.0.*

## 19. `Graphiti` 构造器：依赖注入决定运行边界

核心入口位于 [`graphiti_core/graphiti.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/graphiti.py)。`Graphiti` 构造器接收 graph driver、LLM client、embedder、cross encoder、telemetry/tracer 等依赖。若调用者不显式提供，默认路径会使用 OpenAI 相关实现。

这意味着“安装 Graphiti”与“拥有可运行的本地记忆系统”之间还有一段距离。至少需要：

- 一个受支持的图数据库和连接配置；
- 能执行结构化抽取与 resolution 的 LLM；
- 与索引维度一致的 embedding 模型；
- 若 recipe 使用 cross encoder，还需要重排模型；
- 索引与约束初始化；
- 生产场景中的队列、鉴权、监控和备份。

依赖注入的优点是可替换。LLM、embedding 和 graph backend 不被硬编码在算法中，应用可以使用 Azure OpenAI、Anthropic、Gemini、Groq、Voyage、BGE 或 OpenAI-compatible 服务。缺点是组合空间很大：某套 prompt 在 GPT-4o-mini 上可用，不代表在一个小型本地模型上能稳定输出结构化 JSON；embedding 维度变化会影响索引；cross encoder 的 token 限制也会截断长事实。

`Graphiti` 的 public methods 包括索引构建、episode 写入、bulk 写入、三元组写入、搜索、episode 检索、社区构建、episode 删除和 saga 摘要。它是一层 orchestration API，不是一个长期运行的 worker。应用可以直接调用，也可以通过 MCP/FastAPI 封装。

构造器还暴露 tracer 和 telemetry 入口。对多阶段 LLM 管线，trace 比一条总耗时更有价值：应能看到实体抽取、候选召回、resolution、事实失效、embedding、写图和搜索各自耗时与 token。默认可观测能力仍需接入你的后端和采样策略，不能只因为源码有 tracing hook 就认定已经满足生产审计。

## 20. `add_episode()`：一次写入背后的完整主链

[`add_episode()`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/graphiti.py) 是理解 Graphiti 的最佳入口。其主链可以抽象为：

```text
load recent episodes
  -> create episodic node
  -> extract entity nodes
  -> resolve entities against graph candidates
  -> extract facts between resolved entities
  -> resolve duplicate and contradictory facts
  -> derive temporal invalidation operations
  -> generate embeddings and summaries
  -> persist episode, entity nodes, fact edges, provenance edges
  -> optionally update communities
```

![Graphiti episode ingestion pipeline](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-diagram03-ingestion-pipeline.svg)

*本文根据论文抽取流程和 Graphiti v0.29.2 `add_episode()` 整理，非论文原图。核心 API 直接协调写入；生产后台队列属于调用层或 MCP server。*

这里有几个工程细节值得强调。

**顺序依赖很强。** 实体未 resolution，事实端点就不稳定；事实未 resolution，不能可靠判断旧边是否失效；时态更新未完成，当前搜索可能同时看到互相冲突的边。一次写入不是若干可任意并发的小任务。

**写入可能部分失败。** LLM 抽取成功但图数据库写入失败、实体写入成功但 edge 批量失败、社区更新超时，都可能留下中间状态。生产层应为 episode 分配稳定 ID，记录 ingestion status，并设计重试幂等语义。

**LLM 调用数不是常数。** 输入实体数、候选数、reflection、事实数和时间解析都会影响调用量。一个 2KB episode 与一个包含几十个实体的 JSON 事件，成本差异很大。容量规划不能只按 episode QPS。

**reference time 是必要字段。** 相对时间表达依赖它。客户端若传错时区或把处理时间当事件时间，会让双时态从入口开始失真。

源码注释建议将 ingestion 放入 background queue，例如 Celery。核心 `add_episode()` 自身不是 durable queue；MCP server 另有异步队列服务。这两个事实必须同时写清，不能把 server 的队列能力误认为核心库自动持久化每次调用。

## 21. Bulk ingestion 与 `fact_triple`：更快入口，不是免费一致性

稳定版提供 `add_episode_bulk()`，用于批量处理 episode；也提供 `add_triplet()`，允许调用方直接写入 source、relation、target。`EpisodeType.fact_triple` 则把结构化三元组纳入 episode 类型。这些能力适合从 CRM、工单系统或已有知识库导入数据。

bulk 的吞吐优势来自减少往返和并行处理，但它改变了增量语义。若一批 episode 相互引用或包含顺序更新，简单并行可能让 resolution 看不到同批次尚未提交的事实。离线导入应明确排序键、reference time、批次大小和冲突策略，并在完成后运行一致性审计。

直接三元组写入绕过自然语言抽取，适合上游已经有可信 schema 的事件：

```python
await graphiti.add_triplet(
    source_node=customer,
    edge=subscription_relation,
    target_node=plan,
)
```

它不会自动把不可靠业务数据变可靠。上游如果把临时状态当永久事实、遗漏事件时间或使用错误实体 ID，Graphiti 只会更高效地写入错误。结构化入口仍需验证 schema、唯一键和时态字段。

`fact_triple` 是论文后演进属性。论文原始输入列举 message、text 和 JSON；把当前 enum 倒写成论文方法会造成版本错置。更合理的表述是：稳定代码将论文的动态图维护能力扩展到了显式事实流。

对于大规模迁移，可以先将原始记录保留为 episode，再以事实三元组作为派生层。这样既获得确定性结构，也保留审计来源。若只导入三元组且没有 source metadata，后续很难解释事实从哪里来。

## 22. 自定义类型系统：开放抽取与领域 ontology 的折中

Graphiti 允许用 Pydantic 模型定义 entity types 和 edge types，并为类型声明额外属性。这是从通用记忆引擎走向业务知识图的关键能力。一个 SaaS 客服场景可以定义：

```python
class Customer(BaseModel):
    customer_id: str
    segment: str | None = None

class SupportTicket(BaseModel):
    ticket_id: str
    severity: str | None = None

class Affects(BaseModel):
    environment: str | None = None
    observed_at: datetime | None = None
```

类型约束能提高输出一致性，帮助过滤和查询，也能要求模型抽取稳定业务字段。但它不是数据库 schema migration 的替代。新增属性、重命名类型、改变枚举和删除字段都会影响既有节点、prompt、索引和下游查询。

`excluded_entity_types` 等配置允许在某些任务中限制抽取。限制过少会导致图中出现大量无价值名词，限制过多又会漏掉长尾信息。一个合理流程是先用离线样本分析实体分布，再定义核心类型和开放 fallback，而不是直接让模型“抽取所有重要实体”。

ontology 还会影响事实 resolution。若 `Company` 与 `Product` 类型混淆，同名节点可能被错误候选召回；若一个实体在不同 episode 中被赋予不同类型，系统需要决定合并、升级为更一般类型还是保留两个节点。

在高合规场景，类型系统还可作为敏感字段控制面。比如禁止把身份证号放入自由文本 summary，只允许加密后的 reference；对健康数据设置独立 group 和保留周期。但 Graphiti 不会自动识别 PII，也不提供完整字段级授权。类型定义只给应用提供落点。

## 23. 节点与边模型：代码中比论文多了什么

[`nodes.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/nodes.py) 和 [`edges.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/edges.py) 定义稳定版图数据模型。主要节点包括 `EpisodicNode`、`EntityNode`、`CommunityNode`，以及后续加入的 `SagaNode`。边则覆盖 entity-to-entity 事实、episode-to-entity 来源、community membership 和 saga chronology 等关系。

`EpisodicNode` 保存 episode 内容、source、source description、reference time、created time、group、related entity edges 和 metadata。metadata 对业务集成很实用，可以记录外部事件 ID、渠道、语言或数据分类，但也带来 schema 漂移和敏感数据泄漏风险。

`EntityNode` 保存名称、labels、summary、attributes 和 embedding。labels 可用于 ontology 与过滤。summary 是模型生成的聚合描述，不能视作权威字段；稳定业务属性最好放在明确的 typed attributes 中。

`EntityEdge` 是双时态核心。除 source/target、name 和 fact 外，还能看到 `episodes`、`valid_at`、`invalid_at`、`created_at`、`expired_at` 和 `reference_time`。`episodes` 让多个来源共同支撑一条边，时间字段允许历史查询。

`SagaNode` 和相关 chronology 是论文三层图之外的新抽象，用于把一组 episode 组织成更长的故事或过程。它对长项目和跨会话总结有价值，但不能把 saga 等同于论文中的 community：community 来自实体图聚类，saga 更强调 episode 顺序与叙事边界。

代码模型也揭示“图不是纯三元组集合”。节点和边都携带文本、embedding、时间、分组、来源和类型。任何迁移或备份策略都必须保留这些属性，否则恢复出来的只是拓扑，不是可用的 Agent memory。

## 24. 搜索实现：`search()`、`search_()` 与 recipes

搜索代码主要位于 [`graphiti_core/search/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/search)。`search()` 是面向常用事实查询的高层入口，接受 query、group IDs、中心节点等参数，并使用预定义配置。`search_()` 更通用，返回 `SearchResults`，调用者可为 edges、nodes、episodes 和 communities 分别设置检索与 reranker。

`SearchConfig` 描述每种资源是否启用、候选数量和重排方法；`SearchFilters` 限定类型、时间和属性；`search_config_recipes.py` 提供组合好的 recipe。源码结构表达了两个设计原则。

第一，**候选源和返回对象是两回事**。查询可以利用节点和图距离帮助排事实边，也可以直接返回 episode 或 community。应用应根据回答任务选择输出对象，而不是默认所有搜索都返回 fact string。

第二，**配置是相关性逻辑的一部分**。RRF、MMR、cross encoder、distance 和 mentions 的选择会改变结果，应该像模型版本一样被版本化和 A/B 测试。把 recipe 埋在代码默认值中，会让线上答案变化难以解释。

一个生产 query contract 可以显式记录：

```json
{
  "query": "When did the customer migrate to Project X?",
  "group_ids": ["tenant-acme"],
  "center_node_uuid": "customer-42",
  "recipe": "edge_hybrid_rrf_cross_encoder",
  "time_scope": {"as_of": "2025-05-01T00:00:00Z"},
  "limit": 12,
  "token_budget": 1800
}
```

Graphiti 并不原生定义这里所有业务字段，尤其是 `as_of` 和 `token_budget` 的完整产品语义。示例的意义是让调用层把搜索策略、时间视角和上下文预算显式化。

安全方面，搜索过滤器最终会进入 driver 查询。不能把用户输入直接拼成 Cypher labels、属性名或过滤表达式。稳定版已经包含相关硬化与测试，但应用自定义 query layer 仍需白名单、参数化查询和权限过滤。

## 25. Graph Drivers：同一 API 不代表相同运行特性

`v0.29.2` 支持 Neo4j、FalkorDB 和 Amazon Neptune/OpenSearch 等 driver，Kuzu 路径已经标记 deprecated。driver 负责节点/边 CRUD、索引、全文查询、向量搜索、BFS 和数据库生命周期。统一接口降低上层耦合，却不能抹平后端差异。

**Neo4j** 生态成熟，Cypher、全文索引和向量能力较完整，适合从单机到集群的多种部署。它的授权、备份和集群能力取决于版本与许可证，不能从 Graphiti 的 Apache-2.0 推导 Neo4j 全部功能也同样开放。

**FalkorDB** 强调高性能图查询和 Redis 风格部署。它的索引、事务和持久化行为与 Neo4j 不完全一致，容量测试需要使用真实 graph shape，而不是只比较空图 QPS。

**Amazon Neptune + OpenSearch** 将图与搜索能力组合在 AWS 托管服务中。网络拓扑、IAM、索引同步、区域费用和最终一致性都进入系统设计。Graphiti adapter 只是访问层，不替代 AWS 运维和权限配置。

**Kuzu** 在稳定版仍可见，但已弃用。新项目不应把 deprecated driver 当长期承诺；已有部署需要评估迁移和数据导出。

后端选择至少要测：episode 写入 P50/P95、实体 resolution 候选查询、事实 invalidation 的事务行为、混合搜索延迟、BFS 在高连接节点上的爆炸、索引重建时间、备份恢复和跨租户过滤。论文的远程 Zep 服务延迟不能作为任一开源 driver 的 SLA。

![Graphiti temporal graph walkthrough](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-official-temporal-walkthrough.gif)

*Graphiti 仓库中的时态图 walkthrough。Source: getzep/graphiti v0.29.2 repository, Apache-2.0.*

## 26. 模型提供方：可替换不等于可互换

`graphiti_core/llm_client/`、`embedder/` 和 `cross_encoder/` 提供模型适配。稳定版可以连接 OpenAI、Azure OpenAI、Anthropic、Gemini、Groq 等 LLM 路径，embedding 可使用 OpenAI、Voyage、BGE 等实现，也支持若干 OpenAI-compatible 本地服务。接口兼容只是第一层，行为兼容更难。

结构化抽取依赖模型稳定遵循 schema。不同提供方对 JSON schema、tool calling、重试和 token 限制的实现不同。同一个 prompt 在模型 A 上返回合法 JSON，在模型 B 上可能添加解释文本、截断数组或把日期格式写错。Graphiti 的 client 会处理部分解析与重试，但不能保证任意兼容接口都达到相同准确率。

embedding 切换也不是替换一个 URL。需要考虑：

- 向量维度是否与已有索引一致；
- 归一化和距离度量是否匹配；
- 多语言与领域术语能力；
- 节点、事实和查询是否使用同一模型版本；
- 模型升级时是否重嵌入全图；
- 批量速率、失败重试和数据合规。

若新旧 embedding 混在同一索引，cosine 分数通常失去意义。更稳妥的迁移是建立新索引版本、后台重嵌入、双读比较，再切换 recipe。

cross encoder 需要逐对评分，成本与候选数大致线性相关。它可能提高 top-k 精度，却增加查询延迟和模型依赖。低延迟 Agent 可以先 RRF 返回，高价值复杂查询再使用 cross encoder；或者离线评估哪些 query class 真正受益。

本地模型能降低外部数据传输，却不自动等于更安全或更便宜。团队要负责权重来源、推理服务、并发、补丁、日志脱敏和 prompt 注入防护。所谓“OpenAI-compatible”只描述协议外形，不描述输出质量。

## 27. Namespaces 与 Saga：论文之后的两个重要抽象

当前 Graphiti 文档与代码出现 namespaces 和 saga 等能力。它们有助于组织大型图，但不属于论文原始三层模型，必须独立理解。

namespace 可用于直接 CRUD、隔离不同类型的图内容或映射不同数据库选择。它解决的是命名和存储组织问题。它并不验证调用方身份，也不实现“用户只能访问自己 namespace”的授权规则。若 API 根据用户传入字符串选择 namespace，却没有服务端租户映射，就会形成越权入口。

`group_id` 类似地用于分区和过滤。一个安全请求链至少应当是：认证层得到 principal，授权层计算允许的 tenant/group，服务端将其注入 Graphiti 调用；客户端不能自由指定任意 group。审计日志需要记录 principal、group、query recipe 和返回资源 ID。

saga 则面向长时间 episode 序列。`summarize_saga()` 可以把一段相关事件组织成摘要，chronology 帮助保持顺序。它适合工单处理、项目迁移、销售机会等有明确生命周期的过程。

saga 与 community 的差异可以用一句话概括：community 回答“哪些实体在结构上属于同一主题”，saga 回答“哪些 episode 构成同一个演进过程”。一个项目社区可能存在多年，里面包含多个上线 saga、故障 saga 和复盘 saga。

saga 摘要仍是损失压缩。它应链接回 episode，并记录摘要模型与生成时间。业务过程更新后，摘要需要失效或重建。若下游把 saga summary 当作唯一历史，Graphiti 原本保留 provenance 的优势会被应用层重新丢掉。

## 28. `group_id` 的安全边界：分区键不是授权策略

许多示例会为同一用户、会话或租户使用一个 `group_id`。这很方便，但要明确它解决的是**图查询范围**，不是“谁有权访问”。从攻击者视角，只要能控制请求中的 group 字符串，就可能尝试读取其他组。

一个生产服务应至少有四层控制：

1. API 层验证身份和服务凭据；
2. 授权层根据 principal、organization 和资源关系计算可访问组；
3. Graphiti adapter 只接受服务端注入的 group，拒绝客户端直传；
4. 图数据库凭据按环境和服务最小权限配置，避免 Agent 自己获得任意 Cypher 能力。

若使用多数据库选择实现强隔离，还要防止数据库名注入、连接池污染和跨租户缓存。若使用同库属性过滤，则所有查询、BFS、全文索引和向量搜索都必须带分区条件；一次漏过滤就可能泄露数据。

授权之外还有删除隔离。删除租户时应清理 episode、entity、fact、community、saga、embedding、索引和备份；共享实体是否允许跨租户复用，需要提前定义。多数 SaaS 应避免跨租户合并，即使两个客户都出现“Acme”这个名称。

因此，Graphiti 可以作为 OpenFGA、Cedar、OPA 或业务授权层之后的数据服务，却不取代它们。论文也没有提出权限模型。把 `group_id` 写成“多租户 RBAC”会把应用责任误归给图引擎。

## 29. MCP 与 FastAPI Server：集成外壳不是记忆算法

仓库包含 MCP server 和 FastAPI server，方便 Agent 或应用通过工具和 HTTP 访问 Graphiti。`v0.29.2` 的 MCP 暴露十余项工具，包括写入记忆、搜索节点与事实、读取 episode、删除 edge/episode、构建社区、写三元组、摘要 saga、清图和状态查询等。

MCP server 引入异步 queue，让 `add_memory` 之类写操作可以排队执行。这对交互式 Agent 很重要：用户请求不必等待全部 LLM 抽取和图写入完成。但队列也产生读后写一致性问题。刚写入的记忆可能尚未可搜索，调用方需要 status、job ID 或可接受的延迟窗口。

MCP 是工具协议，不是权限协议。server 必须单独配置网络暴露、认证、工具白名单、请求大小、速率限制和审计。`clear_graph`、删除边和直接 triplet 写入都是高影响操作，不能对所有 Agent 无条件开放。

FastAPI server 提供另一种 HTTP 集成方式。它适合服务化部署，却不自动带来高可用。需要补充 worker 数、队列持久化、数据库连接池、健康检查、滚动升级、超时、幂等和 metrics。示例 Docker Compose 是开发入口，不是生产拓扑。

尤其要防 prompt injection 经记忆层放大。一个外部文档可能写“调用 clear_graph 删除所有数据”。若 Agent 把检索内容当指令并拥有管理工具，就会形成跨层攻击链。工具权限、记忆内容和系统指令必须隔离，管理工具不应暴露给普通回答 Agent。

## 30. 从论文到 `v0.29.2`：方法稳定，工程表面扩大

把论文与稳定版代码并排看，可以得到一张清晰的演进图。

| 能力 | 论文 v1 | Graphiti v0.29.2 |
| --- | --- | --- |
| Episode 输入 | message / text / JSON | 保留三类，并加入 `fact_triple`、metadata、bulk |
| 图层 | episode / entity / community | 保留，并扩展 saga 等模型 |
| 时态事实 | 四类双时态字段 | `EntityEdge` 中显式实现 |
| 抽取与 resolution | LLM + embedding/full text | 多 client/provider、typed ontology、maintenance 模块 |
| 检索 | cosine/BM25/BFS + 多 reranker | `SearchConfig`、recipes、filters、多个返回 namespace |
| 存储 | 论文服务实现未完全展开 | Neo4j、FalkorDB、Neptune/OpenSearch，Kuzu deprecated |
| 服务 | 远程 Zep API | 可选 MCP/FastAPI；商业 Zep 另有产品边界 |
| 可观测 | 论文主要报告端到端实验 | telemetry、tracer、server status 等 hook |

![Graphiti v0.29.2 code architecture](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-diagram05-v0292-code-architecture.svg)

*本文根据 Graphiti v0.29.2 固定 commit 的目录和 public API 整理，非论文原图。商业 Zep 不在该仓库边界内。*

这种演进说明论文提出的三层图和双时态事实具有延续性，代码并没有转向完全不同的 memory 模型；变化主要发生在可插拔后端、schema、服务入口和产品集成上。

它也提醒读者不要把当前 API 名称倒写成论文证据。论文没有评测 saga，没有比较 Neo4j 与 FalkorDB，没有验证 MCP 工具安全，也没有给自定义 Pydantic ontology 做消融。它们是工程能力，不是 Table 1-3 的因果来源。

![Structured and unstructured ingestion in Graphiti](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-official-structured-unstructured.gif)

*Graphiti 仓库对结构化与非结构化 ingestion 的演示。Source: getzep/graphiti v0.29.2 repository, Apache-2.0.*

## 31. 实验设置：先看模型、检索量与网络位置

论文用 Deep Memory Retrieval（DMR）和 LongMemEval 评估长期记忆问答。图构建使用 `gpt-4o-mini-2024-07-18`，embedding 和 reranking 使用 BGE-M3。答案生成比较 `gpt-4o-mini` 与 `gpt-4o-2024-11-20`；DMR 还引用 `gpt-4-turbo-2024-04-09` 的基线结果。

这个配置意味着实验不是“Graphiti 开源库 + 本地图数据库”的单组件 benchmark。结果由至少四部分共同决定：图构建模型、检索与重排、上下文模板、答案模型。更强答案模型可能更善于利用日期事实，也可能更能处理 full-context。要把收益归因到时态图，需要保持其他部分一致。

论文对检索数量的描述存在一个需要保留的歧义：实验设置章节写 Zep 检索 top 20 结果，DMR 小节又写 top 10。二者可能对应不同实验或不同对象，但论文没有给出足够细的逐表配置。本文不擅自统一，复现者应把 top-k 当作待核验参数。

LongMemEval 比较的是：

- **Full-context**：将完整历史放入模型上下文；
- **Zep**：通过图检索构造约 1.6K token 的相关上下文。

评分使用 LLM judge。LLM judge 能处理开放式答案，却带来模型偏好、prompt 敏感和重复运行方差。论文报告 latency IQR，但没有给出多 seed 准确率置信区间。

实验执行于 2024 年 12 月至 2025 年 1 月，客户端是波士顿的一台消费级笔记本，远程访问 AWS `us-west-2` 的 Zep API。网络跨区域延迟计入 Zep，而 full-context baseline 主要是模型 API。这个设置对 Zep 延迟并不占便宜，但也使数字难以外推到同区域生产部署或本地 Graphiti。

论文未完整公开远程服务配置、图后端、并发、缓存、模型调用成本、失败重试和评测快照。因此，表格适合证明“这套 Zep 服务在该协议下有效”，不适合直接作为自托管容量规划。

## 32. Table 1：DMR 的 `94.8%` 到底说明什么

![DMR results](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-table01-dmr.png)

*Deep Memory Retrieval 结果。Source: Rasmussen et al., arXiv:2501.13956v1, Table 1, CC BY-NC-SA 4.0.*

论文 Table 1 转录如下：

| Memory | Model | Score |
| --- | --- | ---: |
| Recursive Summarization | gpt-4-turbo | 35.3% |
| Conversation Summaries | gpt-4-turbo | 78.6% |
| MemGPT | gpt-4-turbo | 93.4% |
| Full-conversation | gpt-4-turbo | 94.4% |
| Zep | gpt-4-turbo | **94.8%** |
| Conversation Summaries | gpt-4o-mini | 88.0% |
| Full-conversation | gpt-4o-mini | 98.0% |
| Zep | gpt-4o-mini | **98.2%** |

Zep 相比 MemGPT 的 `94.8% vs 93.4%` 是 1.4 个百分点优势，说明其检索上下文至少没有因压缩而显著损失信息。但与同一模型 full-conversation 的比较更关键：`94.8 vs 94.4` 只高 0.4 个百分点，`98.2 vs 98.0` 只高 0.2 个百分点。这个差距不足以支持“图检索在 DMR 上大幅超过完整上下文”的说法。

Zep 的真正优势可能在效率和可扩展性，而 DMR Table 1 没有给该表的 token、成本和延迟。DMR 会话只有 60 条消息，完整历史可以放入模型窗口。论文也明确承认 DMR 对长期记忆的挑战有限。因此，Table 1 更像正确性 sanity check：Graphiti 没有因检索丢掉大量答案；它不是长期企业记忆的决定性证据。

递归摘要只有 35.3%，说明反复压缩容易损失细节；Conversation Summaries 明显更好，却仍落后于 full conversation 和 Zep。这个结果支持“保留证据比只留摘要稳健”，但不同方法的 prompt、模型版本和实现细节并未全部统一，不能把差异完全归因于数据结构。

## 33. DMR 批判：能装进窗口的数据，验证不了无限历史

DMR 的任务目标是从一段对话中找出某个早期事实。它适合测试基本 memory retrieval，却有四个限制。

第一，历史短。60 条消息对现代长上下文模型不构成硬容量压力。full-conversation 达到 94.4% 或 98.0%，证明直接读取已经很强。若基线不需要截断，图系统的主要价值只能通过成本、延迟和持续更新体现。

第二，更新复杂度有限。真正的时态记忆难题包括多次状态变化、回溯修正、迟到事件、同名实体和跨会话矛盾。单个 needle retrieval 不充分测试 `valid_at/invalid_at`。

第三，评测粒度粗。最终准确率无法告诉我们错误来自实体抽取、事实去重、搜索、重排还是答案模型。一个图可能构建错误，但答案模型从 episode 原文碰巧答对。

第四，数据域与生产风险不同。DMR 没有测试权限隔离、删除请求、prompt injection、批量写入和高并发更新。

因此，工程评测不应只复刻 DMR。至少应加入：同一属性多次更新、乱序事件、跨租户同名实体、否定与假设、来源冲突、删除后不可召回、无答案和 adversarial episode。长期记忆的质量是整个生命周期属性，不是一次 top-k 命中率。

## 34. Table 2：LongMemEval 的准确率、延迟和 token 三重结果

![LongMemEval overview](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-table02-longmemeval-overview.png)

*LongMemEval 总体结果。Source: Rasmussen et al., arXiv:2501.13956v1, Table 2, CC BY-NC-SA 4.0.*

| Memory | Model | Score | Latency | Latency IQR | Avg Context Tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Full-context | gpt-4o-mini | 55.4% | 31.3 s | 8.76 s | 115k |
| Zep | gpt-4o-mini | **63.8%** | **3.20 s** | **1.31 s** | **1.6k** |
| Full-context | gpt-4o | 60.2% | 28.9 s | 6.01 s | 115k |
| Zep | gpt-4o | **71.2%** | **2.58 s** | **0.684 s** | **1.6k** |

这张表比 DMR 更能支持论文主张，因为它同时显示质量和效率。完整历史平均约 115K tokens，Zep 上下文约 1.6K，减少约：

$$
1-\frac{1.6}{115}\approx 98.61\%.
$$

在 `gpt-4o-mini` 上，准确率从 55.4% 到 63.8%，绝对增加 8.4 个百分点，相对提升：

$$
\frac{63.8-55.4}{55.4}\approx 15.16\%.
$$

在 `gpt-4o` 上，准确率从 60.2% 到 71.2%，绝对增加 11.0 个百分点，相对提升：

$$
\frac{71.2-60.2}{60.2}\approx 18.27\%.
$$

摘要写“up to 18.5%”属于四舍五入或计算口径差异；严谨写法应同时给出原始分数和绝对百分点。把 `18.5%` 写成“提升 18.5 个百分点”是错误的。

延迟从 31.3 秒降到 3.20 秒，下降约 89.8%；从 28.9 秒降到 2.58 秒，下降约 91.1%。这与论文“约 90%”一致。IQR 也显著下降，说明长上下文不仅慢，波动也更大。

但这些 latency 包含跨区域网络和模型 API，不是图查询微基准。Zep 路径也可能利用服务缓存。表格不能回答图数据库占多少、embedding 检索占多少、cross encoder 占多少、答案模型占多少。落地时必须重新拆分。

## 35. 数字审计：相对提升、百分点和延迟下降不能混写

论文传播中最常见的数字误读有三类。

**把相对提升写成百分点。** `55.4 -> 63.8` 是 +8.4 个百分点，或相对 +15.2%；`60.2 -> 71.2` 是 +11.0 个百分点，或相对约 +18.3%。两种说法都对，但不能混用单位。

**把 context reduction 当答案 token reduction。** `115k -> 1.6k` 指输入上下文平均 token，不包含系统 prompt、问题、输出 token、图构建阶段 token，也不等于总成本下降 98.6%。Graphiti ingestion 已经提前支付 LLM 抽取和 embedding 成本。读多写少场景更容易摊薄，写多读少未必。

**把延迟下降当本地图查询速度。** Table 2 是端到端回答延迟。在 full-context 路径，115K prompt 的模型 prefill 是主要成本；Zep 将 prompt 大幅缩短，自然能降低生成调用延迟。图检索本身快不快，需要单独测。

一个完整成本模型可以写为：

$$
C_{total}=C_{ingest}+N_q(C_{retrieve}+C_{rerank}+C_{answer}),
$$

其中 $N_q$ 是后续查询次数。若一个 episode 被查询很多次，预处理成本被复用；若数据写入后几乎不查，复杂图构建可能得不偿失。论文没有提供 break-even 分析。

性能报告也应区分 cold/warm cache、同区域/跨区域、并发数、图规模和候选数。只复述 `3.20 s` 不足以给自托管系统定 SLA。

## 36. Table 3：偏好与时态提升明显，但存在真实回退

![LongMemEval breakdown](/images/blog/graphiti-zep-temporal-knowledge-graph-agent-memory/graphiti-zep-table03-longmemeval-breakdown.png)

*LongMemEval 分类型结果。Source: Rasmussen et al., arXiv:2501.13956v1, Table 3, CC BY-NC-SA 4.0.*

| Question Type | Model | Full-context | Zep | 绝对变化 |
| --- | --- | ---: | ---: | ---: |
| single-session-preference | gpt-4o-mini | 30.0% | 53.3% | +23.3 pp |
| single-session-assistant | gpt-4o-mini | 81.8% | 75.0% | -6.8 pp |
| temporal-reasoning | gpt-4o-mini | 36.5% | 54.1% | +17.6 pp |
| multi-session | gpt-4o-mini | 40.6% | 47.4% | +6.8 pp |
| knowledge-update | gpt-4o-mini | 76.9% | 74.4% | -2.5 pp |
| single-session-user | gpt-4o-mini | 81.4% | 92.9% | +11.5 pp |
| single-session-preference | gpt-4o | 20.0% | 56.7% | +36.7 pp |
| single-session-assistant | gpt-4o | 94.6% | 80.4% | -14.2 pp |
| temporal-reasoning | gpt-4o | 45.1% | 62.4% | +17.3 pp |
| multi-session | gpt-4o | 44.3% | 57.9% | +13.6 pp |
| knowledge-update | gpt-4o | 78.2% | 83.3% | +5.1 pp |
| single-session-user | gpt-4o | 81.4% | 92.9% | +11.5 pp |

偏好、时态推理、多会话和用户信息大多提升，符合时态实体图的设计目标。尤其 `gpt-4o` preference 从 20.0% 到 56.7%，说明全量上下文并不保证模型能在大量历史中找到偏好；结构化事实和检索可能更有效。

但 assistant 类明显回退：mini 下降 6.8 个百分点，4o 下降 14.2 个百分点。knowledge-update 在 mini 上也下降 2.5 个百分点。这些反例很重要。它们可能来自检索漏召回、事实抽取损失、构造器未保留 assistant 言论，或 full-context 对局部问题本来就更强。

表中的 `Delta` 是相对变化，例如 `30.0 -> 53.3` 写 77.7% 上升，而不是 23.3 pp。对基线 20.0%，到 56.7% 的相对增幅达到约 184%，看起来巨大，但样本量和方差未展示。应该优先看原始分数和每类样本数。

工程上可采用 hybrid fallback：图检索置信不足、问题明确指向某一最近 episode、或 query class 在离线评测中表现回退时，追加局部原文或使用 full-context/windowed context。Graph memory 不必成为唯一检索路径。

## 37. 实验边界：这不是可直接复制的生产 benchmark

综合三张表，论文证据可以支持三个有限结论：Zep 在 DMR 上与 full conversation 接近；在 LongMemEval 上显著减少上下文并提高总体准确率；收益主要集中在偏好、时态和多会话问题。它不能支持“Graphiti 在所有记忆任务上都优于长上下文”。

复现难点包括：

- 论文使用远程 Zep API，未给可对齐的服务快照；
- 图构建模型和答案模型使用带日期的商业模型版本；
- top-10/top-20 描述不完全一致；
- 没有公开完整 prompt、缓存状态、重试和失败样本；
- 没有 seed、置信区间和每类样本量；
- 没有公开图规模、节点边数量、构建 token 与费用；
- LongMemEval 答案由 LLM judge 评分；
- 远程网络位置与 API 端点影响延迟。

因此，本文没有声称使用 `v0.29.2` 复现 Table 1-3。开源代码可以帮助搭建类似流程，不能保证得到同样图、同样检索结果和同样分数。

对企业决策，更有价值的是在自有数据上构造 lifecycle benchmark：从真实工单和会话中匿名抽样，人工标注实体、事实、时间和更新关系；离线比较 full-context、recent-window、vector RAG、Graphiti；再以真实读写比例计算总成本。只有这样，才能判断图构建投入是否值得。

## 38. 相关工作：Graphiti 位于哪些路线之间

Agent memory 并不是单一技术类别。Graphiti 与几条路线都有交集，但解决重点不同。

### 38.1 MemGPT / Letta：上下文管理与虚拟记忆

MemGPT 将上下文窗口类比为有限内存，通过模型驱动的 memory management 在 core memory、archival memory 等层级间移动信息。它强调 Agent 自己决定何时写入和检索。Graphiti 更强调外部持续摄取、实体事实图和双时态更新。两者可以组合：Letta 管理 Agent 工作记忆，Graphiti 提供可审计的长期事实层。

区别也带来不同失败面。模型自治写记忆容易漏写或写入主观摘要；自动图抽取会产生实体与事实 resolution 错误。选择哪条路线取决于记忆是否主要来自 Agent 内部对话，还是来自多源业务事件。

### 38.2 GraphRAG：离线语料的社区层次

Microsoft GraphRAG 从文档构建实体关系图和社区摘要，擅长对大型静态语料做 global/local search。Graphiti 的 community 设计与它有家族相似性，但重点是增量 episode、事实有效期和实时更新。GraphRAG 常在语料变化后离线重建，Graphiti 试图在每次 episode 到来时局部维护。

### 38.3 LightRAG、RAPTOR 与层次检索

LightRAG 结合图结构和向量检索，RAPTOR 用递归聚类摘要构建树。它们都在解决“单层 chunk 召回不足”。Graphiti 的独特部分不是有图，而是 episode provenance 和双时态事实生命周期。若数据没有更新语义，复杂时态维护未必比轻量层次索引更划算。

### 38.4 AriGraph 与环境世界模型

AriGraph 等工作让 Agent 从环境交互中构建知识图，用于规划和探索。Graphiti 更像通用记忆基础设施，不限定某个 embodied 环境或策略学习。它能存环境事件，却不自动提供规划算法、奖励或策略更新。

### 38.5 传统时态知识图谱

数据库与知识图谱领域早已研究 valid time、transaction time、event sourcing、ontology 和 entity resolution。Zep 的贡献不是发明双时态，而是把这些思想集成到 LLM Agent 的持续抽取与检索链路中。工程团队应借鉴成熟时态数据库的 interval 约束、迟到事件和审计语义，而不是只从 prompt 角度思考。

### 38.6 普通向量记忆

向量库方案最轻：按消息或摘要切块，嵌入后 top-k。对于短历史、低更新和以语义回忆为主的应用，它可能已经足够。Graphiti 的成本只有在关系查询、更新历史、跨源实体和审计真正重要时才有回报。技术选型不应以“图更高级”为理由，而应以失败模式和读写经济性为理由。

## 39. 与 OpenSPG KAG 的关系：记忆维护与知识求解是两件事

本站另一篇 OpenSPG KAG 精读关注专业领域问答。KAG 通过 LLMFriSPG、知识-文本互索引、语义对齐和逻辑形式执行，把问题分解为检索、排序、数学和推理步骤。Graphiti 和 KAG 都使用知识图与文本证据，但系统中心不同。

| 维度 | Graphiti / Zep | OpenSPG KAG |
| --- | --- | --- |
| 首要问题 | Agent 长期记忆持续更新 | 专业问题的知识求解 |
| 数据单位 | episode、entity、fact、community | 文档、chunk、SPG schema、concept、instance |
| 时间重点 | 双时态事实生命周期 | 论文主线不是双时态记忆 |
| 推理重点 | 混合检索与上下文构造 | logical form planning 与 hybrid reasoning |
| schema | 可开放抽取，也可自定义类型 | 更强调专业 schema 与语义对齐 |
| 典型输出 | 与当前 query 相关的记忆事实 | 可执行推理过程和专业答案证据 |

两者可以在架构上衔接。Graphiti 接收用户与业务系统的实时事件，维护“这个客户最近发生了什么”；KAG 或其他 solver 使用经过治理的专业知识库回答“按政策应该如何处理”。前者是动态 context plane，后者是知识求解 plane。

不能把二者粗暴合并成一张无限大图。用户记忆通常含 PII、租户边界和短期状态，专业知识库则强调共享规则、版本和权威来源。它们的保留策略、权限和更新节奏不同。更稳妥的方案是在查询编排层并行获取个人/业务 context 与专业证据，再由回答策略明确标注来源。

Graphiti 也没有 KAG 论文中的逻辑形式执行器。BFS、RRF 和 cross encoder 是检索手段，不等于多步符号推理。若问题需要规则、数学或约束求解，应增加专用工具，而不是期待实体图本身自动完成 reasoning。

## 40. 工程落地指南：从 episode contract 开始，而不是先装 Neo4j

落地 Graphiti 最常见的错误，是先启动图数据库和 quickstart，再让所有业务消息直接进入模型。正确顺序应从数据合同、风险和评测开始。

### 40.1 定义 episode contract

每个事件至少需要稳定 ID、租户/主体、source、正文、reference time、ingestion time、数据分类和幂等键。示例：

```json
{
  "episode_id": "crm-event-01JY8Q...",
  "tenant_id": "tenant-acme",
  "subject_id": "customer-42",
  "source": "crm.timeline",
  "source_record_id": "evt-8891",
  "reference_time": "2026-07-15T09:30:00+08:00",
  "ingested_at": "2026-07-15T09:31:12+08:00",
  "content_type": "json",
  "content": {"plan": "enterprise", "status": "activated"},
  "classification": "confidential",
  "schema_version": 3
}
```

reference time 与 ingestion time 必须分开。事件重放使用原 reference time，不能因为重试而把事实起点改成今天。

### 40.2 先做小型金标准

选取几百条真实匿名事件，人工标注实体、业务 ID、事实、有效区间、重复和矛盾。评测抽取 precision/recall、entity resolution、temporal invalidation 和 search recall。没有这组数据，换 prompt 或模型只能凭感觉。

### 40.3 区分同步提交与异步可见

交互请求写入 durable queue，worker 调用 `add_episode()`，成功后记录可搜索版本。API 返回 job ID，而不是假装内存 enqueue 等于持久成功。需要定义 read-your-writes：用户刚提交偏好后，下一个请求是否必须立刻可见；若必须，可能要在 Graphiti 搜索外增加短期 pending memory。

### 40.4 建立幂等与重放

Graphiti UUID 与业务事件 ID 应有确定映射或幂等表。worker 重试不能重复创建 episode。模型、prompt、ontology 和 embedding 版本都要记录，以便升级时批量重建派生图。

### 40.5 为不同查询选择 recipe

人名和编号查询优先全文，关系查询提供中心节点和 BFS，开放语义问题用 embedding，最终再 RRF 或 cross encoder。不要为所有请求固定最昂贵 recipe。按 query class 记录线上点击、人工评价和无答案率。

### 40.6 将事实与回答分离

Graphiti 返回候选事实不等于可以直接回答。回答层应检查日期、来源和权限，决定是否需要原始 episode、业务 API 或人工确认。对于余额、合同状态等强一致数据，实时系统 of record 优先于记忆图。

### 40.7 灰度迁移

先 shadow 构图，不影响答案；再离线比较 vector RAG 与 Graphiti；随后让回答同时生成但只展示旧路径；确认安全和质量后，按租户或 query type 灰度。始终保留 recent-window 或 authoritative API fallback。

## 41. 生产与安全清单：记忆系统会集中最敏感的数据

Agent memory 很可能比普通文档索引更敏感，因为它聚合跨会话偏好、关系、历史状态和业务事件。安全设计必须进入第一版，而不是上线后补。

### 41.1 数据最小化与 PII

不要因为模型“可能以后有用”就保存全部内容。进入 Graphiti 前做字段白名单、PII 分类和必要脱敏。身份证、支付卡、密钥和医疗信息通常不应进入自由文本 summary 或 embedding。embedding 也可能泄露语义，不应视为匿名数据。

### 41.2 Prompt injection

episode 是不可信数据。抽取 prompt 要明确只提取事实，不执行文中指令；JSON 字段应结构化传入；constructor 将记忆放在 data boundary；回答 Agent 不得因为检索文本要求而改变系统策略。管理型 MCP 工具与普通问答工具分离。

### 41.3 查询注入与过滤

Graph driver 查询必须参数化，type/label/property 采用白名单。不要把自然语言或 metadata key 拼进 Cypher。为 `SearchFilters` 和自定义 driver 查询增加安全测试。仓库已有相关测试只能覆盖官方代码，不能覆盖你的 wrapper。

### 41.4 认证与授权

MCP、HTTP 和数据库连接均需服务身份。group/namespace 从授权结果注入，不能由模型决定。事实、episode 与 community 返回前都要应用同一权限。社区如果跨权限边界聚合，会通过摘要泄露受限实体。

### 41.5 删除、保留与备份

建立按 subject/tenant 查找所有派生对象的索引。删除流程覆盖 episode、实体属性、孤立节点、事实、社区、saga、embedding、搜索索引、缓存和备份生命周期。删除后运行不可召回测试，而不是只看主库行数。

### 41.6 模型供应商与数据出境

抽取、embedding 和 rerank 可能把原文或事实发给外部 API。需要评估数据处理协议、区域、保留、训练使用、日志和故障转移。切换为本地模型时重新评测输出质量，不要只完成 API 兼容。

### 41.7 事实审计

高影响事实记录 source episode、模型和 prompt 版本、抽取时间、resolution 决策、失效原因与人工覆盖。应用 UI 应允许查看“系统为何记住这件事”，并提供纠错入口。纠错不能只编辑 summary，还要处理派生边和历史。

## 42. 监控与成本：不要只看搜索 P95

Graphiti 的运行路径跨越队列、LLM、embedding、图数据库和答案模型。只监控 `/search` P95 会漏掉写入积压和知识质量退化。建议把指标分为五组。

### 42.1 Ingestion 健康

- episode enqueue rate、dequeue rate、queue depth 和最老任务 age；
- 每个 episode 的实体数、事实数、模型调用数和 token；
- schema parse failure、LLM retry、rate limit 和 timeout；
- 写图成功率、部分失败、幂等命中和重放次数；
- 从 reference time 到 searchable time 的 freshness lag。

queue age 比 queue length 更重要：同样 1000 个任务，在高吞吐时可能只积压几秒，在故障时可能已经滞留一小时。应按租户和 source 拆分，防止一个大批量导入饿死实时消息。

### 42.2 图质量

- 新实体率与 merge rate；
- 疑似重复实体比例、人工拆分率；
- 每个 episode 的事实去重率、矛盾率和 invalidation 率；
- 缺失 provenance 的边数；
- 无 `valid_at` 或时间解析失败的事实比例；
- 节点度分布、超级节点、孤立实体和社区漂移。

这些指标需要基线。新实体率突然升高，可能是业务增长，也可能是 resolution 模型失效；merge rate 过高可能是别名处理变好，也可能是误合并。

### 42.3 搜索质量

记录每个结果来自 semantic、BM25 还是 BFS，各路候选数量、RRF 名次、cross encoder 分数和最终选中来源。离线标注 query 计算 recall@k、MRR 和 nDCG；线上记录答案引用率、用户纠错、fallback 和无答案率。

搜索延迟按 embedding、driver 查询、BFS、rerank、constructor 分段。高连接实体会让 BFS 尾延迟突增，cross encoder 候选数会让模型成本线性上升。统一总耗时无法定位。

### 42.4 模型成本

分别统计实体抽取、reflection、entity resolution、fact extraction、fact resolution、temporal extraction、summary、embedding、rerank 和 answer token。写入成本与查询成本分开，按 episode 类型、租户和 source 归因。

当模型升级时，建立 canary：对固定样本同时运行新旧模型，比较实体/事实 diff、时间字段、图增长和检索答案。只看模型单价下降可能掩盖重试增加和质量下降。

### 42.5 业务可靠性

最终指标包括正确回答率、过期事实命中率、未经授权结果数、删除后召回数、人工纠错恢复时间和 memory-caused incident。系统应能一键停用图上下文、回退 recent window 或 authoritative API。记忆层故障不应阻断所有 Agent 请求。

## 43. 局限性与批判：时态图没有消除不确定性

论文展示了有吸引力的结果，但方法存在结构性限制。

**LLM 抽取误差会级联。** 实体漏抽导致事实缺失，实体误合并导致跨主体污染，事实 resolution 错误导致不当失效，社区摘要再把错误放大。图把错误结构化后，错误看起来可能比原文更“确定”。

**矛盾判断没有可靠真值。** 语言中的时间、条件、转述和观点很复杂。模型无法仅凭两条句子总是判断是否矛盾。对高风险事实，系统需要 authoritative source 和人工审查。

**双时态字段可能伪精确。** “最近”“曾经”“下季度”不能总被安全转换为精确时间点。论文没有系统评测时间解析误差，也没有展示区间不确定性表示。

**图会持续膨胀。** 保留 episode、旧事实、embedding、社区和 saga 使审计成为可能，也增加存储、索引和维护成本。事实失效不等于回收。需要 retention、compaction 和归档策略。

**社区会漂移。** 增量 label propagation 与摘要更新可能让社区标签随局部写入变化。周期性全量刷新又昂贵，且会改变检索结果。论文没有给大规模社区维护成本。

**检索正确不等于答案正确。** constructor 可能丢失来源或日期，答案模型可能忽略证据、混淆冲突或过度推断。必须分别评测 retrieval 与 generation。

**供应商与模型版本依赖明显。** 论文依赖 OpenAI 模型和 BGE-M3。商业模型更新、下线、限流或数据政策变化都会影响系统。开源多 provider adapter 降低协议锁定，却不消除质量迁移成本。

**实验可复现性有限。** 远程 Zep 服务快照、完整配置、seed、成本和中间图未公开。当前 Graphiti 代码不是论文实验快照。Table 1-3 是重要证据，不是完全可重复的基准包。

**生产安全不在论文核心范围。** 多租户授权、PII、删除、prompt injection、MCP 工具权限、备份和灾难恢复都需要额外系统。`group_id` 和 namespace 不能替代这些能力。

**不是每个应用都需要图。** 如果历史短、更新少、关系简单且 full-context 成本可接受，recent-window 或向量 RAG 更易维护。Graphiti 的收益应通过自有 benchmark 证明，而不是根据论文标题预设。

## 44. 推荐阅读路径与最终判断

第一次阅读不必从仓库目录开始。建议按以下顺序：

1. 先读论文摘要与 Sec. 2，建立 episode/entity/community 三层图；
2. 精读双时态字段和事实失效，确认 valid time 与 transaction time 的区别；
3. 读 Sec. 2.4 的 search/rerank/constructor，再看论文上下文模板；
4. 读 Table 2 和 Table 3，自己复算百分点、相对提升与回退类型；
5. 读 Appendix 的 entity extraction、resolution、fact extraction 和 temporal prompts，理解系统对 LLM 的依赖；
6. 再进入 `graphiti.py` 的 `add_episode()`，沿 maintenance 调用链阅读；
7. 最后看 search recipes、models、drivers、MCP server 和 tests，区分核心算法与集成外壳。

最终判断可以分为贡献、证据和风险三层。

**贡献层面**，Zep 把时态知识图、LLM 增量抽取和 Agent memory 连接得很完整。episode provenance 解决“事实从哪里来”，双时态解决“何时有效与何时知道”，混合检索解决“不同问题需要不同信号”。这些设计比“存摘要”或“存向量”更接近持续业务系统。

**证据层面**，LongMemEval 的 `115k -> 1.6k` token、总体准确率提升和约 90% 延迟下降有说服力，尤其支持长历史场景。DMR 的优势很小，Table 3 也存在 assistant 与部分 knowledge-update 回退。论文证明了潜力，没有证明所有 query class 和部署环境都受益。

**工程层面**，Graphiti `v0.29.2` 提供了可检查的 ingestion、resolution、temporal edge、search recipe 和多后端抽象。它是一块扎实的基础设施，不是完整产品。队列、权限、数据治理、成本控制、事实审核和故障恢复仍由部署方负责。

因此，Graphiti 最适合这样的场景：历史持续增长，事实会更新，实体关系有价值，同一记忆会被多次查询，且业务愿意为可追溯性投入治理。若只是给一个短对话加几条回忆，复杂图管线可能不是最经济的答案。

> Graphiti 的长期意义，在于让 Agent 记忆从“不可解释的历史片段集合”升级为“带时间、来源和生命周期的知识状态”；它的核心风险，则是把概率模型抽取出的断言误当成权威事实，并低估围绕这些断言所需的安全与数据治理。

## 附录 A：`v0.29.2` 源码阅读地图

以下路径都固定到 `ff7e29ccd127d8d9721b5cbb2163a6407ef915fe`：

| 主题 | 路径 | 阅读问题 |
| --- | --- | --- |
| 主编排 | [`graphiti_core/graphiti.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/graphiti.py) | `add_episode()` 先后调用哪些阶段，何时写图 |
| 节点 | [`graphiti_core/nodes.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/nodes.py) | episode/entity/community/saga 保存哪些属性 |
| 边 | [`graphiti_core/edges.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/edges.py) | provenance 与四类时间字段如何表示 |
| Prompt | [`graphiti_core/prompts/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/prompts) | 抽取、resolution、时间解析对模型提出什么要求 |
| 维护 | [`graphiti_core/utils/maintenance/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/utils/maintenance) | 去重、失效、社区维护如何拆分 |
| Ontology | [`graphiti_core/utils/ontology_utils/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/utils/ontology_utils) | Pydantic 类型如何进入抽取和验证 |
| Search | [`graphiti_core/search/search.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/search/search.py) | 多路候选如何执行和合并 |
| Search config | [`graphiti_core/search/search_config.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/search/search_config.py) | 哪些资源和 reranker 可配置 |
| Recipes | [`graphiti_core/search/search_config_recipes.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/search/search_config_recipes.py) | 默认 search 具体选择什么策略 |
| Filters | [`graphiti_core/search/search_filters.py`](https://github.com/getzep/graphiti/blob/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/search/search_filters.py) | 类型和属性过滤如何表达 |
| Drivers | [`graphiti_core/driver/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/driver) | 后端能力差异如何被抽象 |
| LLM clients | [`graphiti_core/llm_client/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/llm_client) | structured output、重试和 provider 差异 |
| Embedders | [`graphiti_core/embedder/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/embedder) | 维度、批量和模型迁移风险 |
| Rerankers | [`graphiti_core/cross_encoder/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/graphiti_core/cross_encoder) | cross encoder 何时进入查询 |
| MCP | [`mcp_server/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/mcp_server) | 工具、队列、HTTP/stdio 与权限缺口 |
| Server | [`server/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/server) | API 壳与核心库边界 |
| Tests | [`tests/`](https://github.com/getzep/graphiti/tree/ff7e29ccd127d8d9721b5cbb2163a6407ef915fe/tests) | driver、search、安全和 eval 覆盖到哪里 |

阅读时先画调用图，再看 provider 实现。否则很容易在某个 Cypher 查询或 prompt 模板中迷失，而忽略端到端数据状态。特别要比较 tests 中的理想输入与业务数据中的乱序、重复、空字段和超长内容。

## 附录 B：建议的最小验收矩阵

| 测试类 | 最小用例 | 期望 |
| --- | --- | --- |
| Episode 幂等 | 同一外部事件重放 3 次 | 只产生一个逻辑 episode，不重复事实 |
| 实体别名 | Acme / Acme Corp. / ACME | 合并到同一业务实体 |
| 同名隔离 | 两租户各有 Alex | 不跨租户合并或召回 |
| 事实更新 | 巴黎 -> 柏林 -> 罗马 | 三段 valid interval 可查询，历史不丢失 |
| 迟到事件 | 先收到 6 月事件，再收到 5 月事件 | valid time 正确插入，transaction history 可审计 |
| 非矛盾共存 | 全职离职 + 继续任顾问 | 不错误关闭顾问事实 |
| 否定与计划 | “可能加入”/“没有加入” | 不把计划当既成事实 |
| 来源审计 | 一个事实由两个 episode 支持 | 返回两个 provenance，删除一个后仍有正确来源 |
| Search exact | 工单 `INC-2048` | BM25/full text 稳定命中 |
| Search semantic | 同义改写问题 | 向量候选进入 top-k |
| Graph relation | 与客户同项目的负责人 | BFS 在权限与类型过滤内返回 |
| Temporal query | “2025-02-03 当时住哪里” | 使用历史 valid interval，不返回今天状态 |
| 无答案 | 图中无相关证据 | 明确返回不足，不编造事实 |
| Prompt injection | episode 要求忽略系统规则 | 不执行记忆中的命令 |
| 权限 | 客户 A 请求客户 B 的 group | 请求被授权层拒绝，图查询不执行 |
| 删除 | 删除 subject 全部数据 | 主库、索引、缓存和派生摘要均不可召回 |
| 队列恢复 | worker 写到一半崩溃 | 重试幂等，状态可观测，无半成品污染 |
| 模型升级 | 新旧抽取模型双跑 | diff 可审阅，图增长与错误率不过门禁 |

验收不能只在空图进行。至少准备小图、中等历史和高连接压力图；对每类用例运行多次，记录模型非确定性。使用外部模型时固定 model version 和 temperature，仍要允许供应商后端存在变化。

## 参考资料

- Rasmussen et al., [Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956v1), arXiv:2501.13956v1.
- [论文 PDF](https://arxiv.org/pdf/2501.13956v1) 与 [TeX source](https://arxiv.org/e-print/2501.13956v1).
- [Graphiti GitHub](https://github.com/getzep/graphiti) 与固定稳定标签 [`v0.29.2`](https://github.com/getzep/graphiti/tree/v0.29.2).
- [Graphiti 官方概览](https://help.getzep.com/graphiti/getting-started/overview).
- [Zep vs Graphiti 官方边界说明](https://help.getzep.com/zep-vs-graphiti).
- Packer et al., [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560).
- Edge et al., [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130).
- Maharana et al., [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753).

本文完成的是论文精读、稳定版源码静态阅读和工程边界分析，没有运行 Graphiti、图数据库、MCP server 或任何 LLM，也没有复现论文的远程 Zep 服务与实验。

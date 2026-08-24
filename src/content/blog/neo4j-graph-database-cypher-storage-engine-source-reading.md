---
title: "Neo4j 源码精读：属性图、Cypher 查询引擎、事务存储与图数据库工程"
description: "从 Neo4j 2026.06.0 Community 源码与 Cypher 论文双线精读属性图模型、查询编译执行、事务日志、页缓存、索引、Bolt 协议和 Community/Enterprise 边界"
pubDate: "2026-07-20T15:05:56+08:00"
updatedDate: "2026-07-20T15:05:56+08:00"
tags:
  - "Deep Reading"
  - "Neo4j"
  - "Graph Database"
  - "Cypher"
  - "Database Internals"
  - "Knowledge Graph"
  - "Code Reading"
draft: false
---

Neo4j 经常被压缩成一句话：“把关系作为一等公民的图数据库。”这句话没有错，却远远不够。一个真正的数据库系统不能只会把节点和边画出来：它必须接收来自驱动的并发请求，解析声明式查询，检查变量和类型，基于统计信息挑选执行计划，在事务隔离下读取和修改存储，先写事务日志，再让脏页安全落盘；它还要管理索引、锁、缓存、恢复、约束、导入、协议兼容和版本迁移。Neo4j 的工程价值，恰恰存在于这些“图模型之外”的数据库职责中。

本文采用三条互相校验、但不能混为一谈的证据线。

第一条是 [Neo4j Community `2026.06.0` 源码](https://github.com/neo4j/neo4j/tree/2026.06.0)，固定到 tag 解引用后的 commit [`eccd584a64d468af3daeab421478fe78567c518f`](https://github.com/neo4j/neo4j/tree/eccd584a64d468af3daeab421478fe78567c518f)。它告诉我们开源 Community Edition 中真实存在什么模块、接口如何分层、事务和存储如何连接。

第二条是 [Neo4j 2026.06 官方文档](https://neo4j.com/docs/)。它覆盖当前 Cypher 25、运行时、索引、存储格式和运维能力，同时也描述了不少只属于 Enterprise Edition 或 Aura 的功能。阅读时必须把“产品文档里存在”与“Community 仓库里有实现”分开。

第三条是 SIGMOD 2018 的论文 [**Cypher: An Evolving Query Language for Property Graphs**](https://doi.org/10.1145/3183713.3190657)。论文讨论的是 Cypher 9 的核心读取语义，它为属性图、模式、变量绑定、表和子句组合提供了形式化锚点，却不是 Neo4j 2026.06 的物理执行说明书。

先给出本文最终判断：

> Neo4j 的长期价值不是“边查找比 SQL JOIN 神奇地快”，而是把属性图数据模型、声明式模式匹配、成本规划、事务内核、原生图存储和多种辅助索引组织成一个可运维的数据库系统。它适合关系本身决定查询形状的业务，但不会自动解决知识本体、推理正确性、超级节点、无界路径、数据治理或查询建模问题。

本文只做论文级精读与源码静态阅读。没有完整编译 Neo4j，没有启动服务器、Enterprise 集群或 Aura，没有运行性能基准，也不会把官方示意图、文档建议或源码结构外推成生产 SLA。

![Neo4j property graph model](/images/blog/neo4j-graph-database-cypher-storage-engine/neo4j-doc-fig01-property-graph-model.webp)

*Neo4j 属性图中的节点、标签、关系类型和属性。Source: Neo4j Documentation 2026.06, CC BY 4.0.*

## 1. 一句话贡献：从“图形数据”到“图数据库系统”

如果只用一句话概括 Neo4j 的系统贡献，可以写成：**Neo4j 把属性图的邻接关系变成存储与查询规划的核心抽象，并在它周围构建了完整的 ACID 数据库执行链。**

这句话包含四层不能省略的含义。

第一，Neo4j 使用的是 **labeled property graph**，即节点可以带多个标签，关系有方向和单一类型，节点与关系都可以携带键值属性。它不同于只保存无类型边的图算法输入，也不同于 RDF 的三元组语义。

第二，用户通常通过 Cypher 描述“想匹配什么形状”，而不是手写每一步遍历。数据库需要把 ASCII-art 风格的 pattern 转成逻辑计划，再决定从哪个标签、约束或索引开始，如何扩展关系，何时过滤、聚合和投影。

第三，“原生图”并不意味着查询绕过数据库基础设施。一次 `MATCH` 仍要经过事务快照、锁或版本可见性、页缓存、记录读取、内存跟踪和结果流控。写查询还要生成存储命令、追加事务日志、提交并在崩溃后恢复。

第四，图查询性能来自数据模型、起点选择、基数控制、局部性和执行计划的共同作用，不来自一句“遍历是 $O(1)$”的口号。单条关系定位可以接近常数级的记录跳转，不代表可变长路径、超级节点或全图分析具有常数复杂度。

因此，本文不会把 Neo4j 写成“知识图谱可视化工具”，也不会把它写成“用边替代 JOIN 的万能数据库”。真正值得精读的是：图语言和数据库内核在哪里接上，哪些层负责语义，哪些层负责性能，哪些能力又超出开源仓库。

## 2. 项目、论文、版本与许可边界

| 项目 | 本文采用的口径 |
| --- | --- |
| 开源仓库 | [`neo4j/neo4j`](https://github.com/neo4j/neo4j) |
| 源码版本 | `2026.06.0` |
| 固定 commit | `eccd584a64d468af3daeab421478fe78567c518f` |
| 源码许可证 | GPL-3.0 |
| 当前语言主线 | Cypher 25；Cypher 5 已冻结 |
| LTS 边界 | 5.26 LTS 与 CalVer 版本线并存 |
| 语言论文 | Cypher 9，SIGMOD 2018，13 页 |
| Java 基线 | 根 `pom.xml` 的 `vm.target.version=21` |
| Maven 基线 | 根 `pom.xml` 要求 `3.9.11` |
| 关键依赖 | Scala `2.13.17`、Lucene `10.4.0`、Netty `4.2.15.Final` |
| 复现状态 | 静态阅读；未构建、未运行、未压测 |

Neo4j 从 2025 年开始使用 CalVer 发布线，同时保留 5.26 LTS。两条版本线意味着“当前文档”不一定适用于仍在 5.26 上运行的系统。本文以 2026.06 为主，涉及 5.26 时明确标记，不把新 `SEARCH` 子句、新 vector provider、Cypher 25 默认行为或存储迁移规则倒写到旧版本。

源码根 [`pom.xml`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/pom.xml) 声明项目版本 `2026.06.0`，要求 Maven `3.9.11`，并把 Java target 设为 21。仓库 [`README.asciidoc`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/README.asciidoc) 仍写着 Maven `3.8.2`。这不是语言细节，而是复现风险：源码构建时应优先相信构建系统实际执行的约束，而不是历史 README 文本。

许可边界更关键。README 明确写着 Community Edition 使用 GPLv3；Enterprise Edition 包含“不在此仓库中的额外闭源组件”，需要商业许可。因此，官方文档中的 `block` store、集群、在线备份、细粒度角色、pipelined/parallel runtime 等能力，即使可以阅读，也不能被描述成当前目标仓库已经开放。

Cypher 论文的版权声明允许在保留声明和完整引用的前提下，为个人或课堂用途复制部分内容。本文只在学术评论语境中引用必要的 Fig. 1-7 和 Table 1，不将论文误标为 CC 开放许可。Neo4j 官方文档图则依据文档页脚标注为 CC BY 4.0。源码、论文和文档是三套不同许可，不能用其中最宽松的一套覆盖另外两套。

## 3. Community、Enterprise 与 Aura 不是三个安装选项那么简单

“Neo4j”作为品牌同时指向开源数据库、商业发行版和托管云服务。若不先拆边界，后面的源码阅读会不断产生错误归因。

**Community Edition** 是本文实际能在 `neo4j/neo4j` 仓库中检查的主体。当前文档说明 Community 默认使用 `aligned` store，标准部署只支持一个标准数据库；Cypher 默认运行时是 slotted；认证和用户能力也比 Enterprise 简化。

**Enterprise Edition** 在 Community 基础上增加闭源模块。当前文档把 `block` 标为 Enterprise 推荐和默认格式，提供多数据库、集群、在线备份、细粒度角色、更多运行时和运维能力。文档是有效的产品事实，但闭源实现不能从本文固定 commit 推导出来。

**Aura** 是 Neo4j 托管服务。它负责基础设施、升级、备份、部分自动调优、云端身份和产品 SLA。Aura 中可用的 parallel runtime、扩缩容方式或监控接口，不应被写成本地 Community 默认行为。

还要把生态项目拆开：APOC 是单独的扩展项目；Graph Data Science 是图算法与机器学习产品；Browser、Bloom 是交互和可视化工具；各种语言 Driver 是独立仓库；GraphQL Library、Kafka Connector 也有自己的版本和许可。核心仓库提供数据库引擎与部分内置 procedure，不等于整个 Neo4j 产品矩阵都在一个 Maven reactor 中。

这种拆分不是吹毛求疵。若团队依据 Community 源码设计恢复流程，却在生产上选择 Enterprise block store，底层文件、迁移和容量假设会不同；若开发环境使用 Community slotted runtime，生产 Aura 的执行计划和内存表现也可能不同。精读必须把“共同接口”和“版本/版本型专有实现”同时写清。

## 4. 图数据库适合什么，不适合什么

图数据库最适合的不是“数据里有关系”，因为几乎所有业务数据都有关系。更有区分度的条件是：**查询经常从实体出发，沿不同类型关系做多跳扩展，路径形状本身就是业务语义，而且连接深度和方向在运行时变化。**

典型场景包括身份与访问网络、欺诈团伙、供应链依赖、服务拓扑、主数据关联、推荐候选、血缘与影响分析。比如“从这个账户出发，两跳内是否到达同一设备，再连接到近期拒付账户”，用图 pattern 表达比把多张关联表手工拼接更直接。

关系数据库并不会因为出现五个 JOIN 就失效。固定、规则、以聚合和批处理为主的关系查询，成熟 RDBMS 往往有更强的列式执行、物化视图和生态工具。把所有外键迁到图数据库，不会自动产生收益；若查询主要按单表过滤、排序和报表，图模型反而增加运维成本。

文档数据库适合把聚合根作为完整对象读写。它可以在文档内部嵌套关系，但跨文档多跳通常需要额外索引或应用层拼接。Neo4j 的关系是一等存储对象，适合不断改变遍历方向和深度的查询。

RDF 三元组库强调 URI、开放世界语义、标准词汇和推理；属性图强调工程友好的节点/关系属性与 pattern query。Neo4j 可以承载知识图谱数据，却不会因为使用了 `:Person`、`:WORKS_AT` 就自动拥有 OWL 推理、SHACL 验证、实体消歧或本体治理。图数据库是存储和查询底座，知识图谱是更上层的数据语义与治理体系。

最后，图数据库不等于图计算引擎。短路径、邻域检索和事务更新是 OLTP 图数据库的强项；全图 PageRank、社区发现和大规模矩阵运算属于分析工作负载，通常要借助 GDS 或其他计算系统。把全图算法直接写成无限扩展的 Cypher，可能让事务数据库承担不合适的资源压力。

## 5. 属性图形式化：节点、关系、标签、类型与属性

Cypher 论文把 property graph 定义为：

$$
G=(N,R,\operatorname{src},\operatorname{tgt},\iota,\lambda,\tau).
$$

其中：

- $N$ 是有限节点标识符集合；
- $R$ 是有限关系标识符集合；
- $\operatorname{src}:R\to N$ 和 $\operatorname{tgt}:R\to N$ 给出关系起点与终点；
- $\iota:(N\cup R)\times\mathcal K\rightharpoonup\mathcal V$ 是有限偏函数，把实体与属性键映射到值；
- $\lambda:N\to 2^{\mathcal L}$ 把节点映射到有限标签集合；
- $\tau:R\to\mathcal T$ 为每条关系指定一个关系类型。

这里有几个经常被忽略的语义。

一个节点可以没有标签，也可以有多个标签；标签更接近分类和访问入口，而不是 Java 类的唯一继承层级。一个关系只有一个 type，却可以和节点一样有属性。关系有方向，即使查询时允许无向匹配，存储事实仍有 source 和 target。

属性映射是偏函数：某个实体没有某个 key 是正常状态。Cypher 中缺失属性通常表现为 `null`，但 `null` 不是图中一个可索引的普通值。`WHERE n.email IS NULL` 与“属性值等于某个 null 常量”不是一回事。

属性图允许平行关系：同一对节点之间可以存在多条不同 ID、不同类型或不同属性的关系。比如用户与商品之间可以同时有多次 `VIEWED`，每条记录自己的时间和 session。若业务只需要“是否看过”，则应在建模和约束层决定是否聚合，不能假设数据库会自动去重。

路径是节点与关系交替组成的序列：

$$
p=n_1r_1n_2\cdots n_{m-1}r_{m-1}n_m,
$$

并满足每条 $r_i$ 的方向与相邻节点一致。路径是一次匹配结果，不一定是长期持久化实体。把“路径”作为节点存储是一种业务建模选择，而不是 Cypher 执行的必然结果。

![Cypher paper example property graph](/images/blog/neo4j-graph-database-cypher-storage-engine/cypher-paper-fig01-example-property-graph.webp)

*论文示例图同时包含研究者、学生、论文、作者关系、指导关系和引用关系。Source: Francis et al., SIGMOD 2018, Fig. 1.*

## 6. 图遍历的优势与“O(1)”神话

![Neo4j graph traversal example](/images/blog/neo4j-graph-database-cypher-storage-engine/neo4j-doc-fig02-graph-traversal.webp)

*从 Tom Hanks 经 `ACTED_IN` 到电影，再经 `DIRECTED` 到导演的局部遍历。Source: Neo4j Documentation 2026.06, CC BY 4.0.*

Neo4j 的经典优势是从已定位节点沿关系扩展。对于一个节点，存储记录可以指向相关关系结构；执行算子不必每跳都重新扫描一张全局边表。这个“index-free adjacency”直觉解释了局部多跳查询为何自然。

但它不意味着完整查询是 $O(1)$。设第 $i$ 层平均分支因子为 $b_i$，无过滤的 $d$ 跳候选数量近似为：

$$
|C_d|\approx\prod_{i=1}^{d}b_i.
$$

即使每次沿一条边的物理访问成本稳定，候选数量仍可能指数增长。超级节点的度数可以达到百万；无界或宽范围可变长路径会枚举大量中间状态；路径唯一性规则、过滤和聚合还需要额外内存。

性能还取决于起点。`MATCH (p:Person {id:$id})-[:KNOWS]->(f)` 若有唯一约束，可以先定位一个 `p` 再扩展。`MATCH (p:Person)-[:KNOWS]->(f) WHERE f.city=$city` 可能先扫大量 Person 和关系。声明式语言给优化器选择空间，但优化器依赖 schema、统计和可用索引，不会凭空知道业务中哪个节点最稀疏。

因此，更准确的结论是：Neo4j 为邻接扩展提供了直接的存储与执行原语，减少了关系重建成本；端到端复杂度仍由候选基数、路径长度、过滤位置、缓存命中、事务状态和返回规模决定。

## 7. Cypher by Example：模式是查询接口，不是执行计划

论文用下面的模式查询研究者、其指导学生和论文引用：

```cypher
MATCH (r:Researcher)
OPTIONAL MATCH (r)-[:SUPERVISES]->(s:Student)
WITH r, count(s) AS studentsSupervised
MATCH (r)-[:AUTHORS]->(p1:Publication)
OPTIONAL MATCH (p1)<-[:CITES*]-(p2:Publication)
RETURN r.name, studentsSupervised,
       count(DISTINCT p2) AS citedCount
```

这段代码同时展示了 pattern、可选匹配、变量作用域、聚合、可变长关系和结果投影。重要的是，语法顺序表达语义组合，不等于存储访问顺序。成本规划器可以选择合适起点和算子，只要保持查询语义。

`MATCH` 产生变量绑定。每一行可看成一个从变量名到值的记录，例如 $u=\{r\mapsto n_6,s\mapsto n_7\}$。多个匹配产生一个 bag of records；bag 允许重复行，这与集合不同。

![Cypher variable bindings](/images/blog/neo4j-graph-database-cypher-storage-engine/cypher-paper-fig02-variable-bindings.webp)

*同一图模式产生的变量绑定表，以及聚合后按研究者分组的结果。Source: Francis et al., SIGMOD 2018, Fig. 2.*

`WITH` 不只是“美化查询的中间 SELECT”。它决定下一阶段可见变量，可以执行投影、聚合、排序、分页和过滤。没有投影到 `WITH` 之后的变量会离开作用域。复杂 Cypher 的很多 bug 不是遍历错误，而是 bag cardinality 和变量作用域错误。

`OPTIONAL MATCH` 类似关系代数中的外连接：若整个可选 pattern 无匹配，它保留已有行，并把新引入变量设为 `null`。它不是“每条边都可选”。过滤放在 `OPTIONAL MATCH` 自己的 `WHERE` 还是后续全局 `WHERE`，会改变保留空行的语义。

`[:CITES*]` 表示可变长关系。论文用 rigid extension 形式化它：一个范围 pattern 对应若干固定长度 pattern 的集合。现代 Cypher 25 还引入更丰富的 quantified path patterns；不能用 2018 年语法图代替当前手册，但论文提供了理解变量绑定和路径枚举的基础。

## 8. Cypher 论文中的符号、模式和语义

![Cypher notation table](/images/blog/neo4j-graph-database-cypher-storage-engine/cypher-paper-table01-notation.webp)

*论文用于形式化 property key、节点、关系、标签、类型、值和 pattern 的符号。Source: Francis et al., SIGMOD 2018, Table 1.*

论文用 $\chi$ 表示 node pattern，用 $\rho$ 表示 relationship pattern，用 $\pi$ 表示 path pattern。一个节点 pattern 可以包含变量、标签集合和属性 map；一个关系 pattern 还包含方向、关系类型集合、长度范围和属性 map。

![Cypher pattern syntax](/images/blog/neo4j-graph-database-cypher-storage-engine/cypher-paper-fig03-pattern-syntax.webp)

*Cypher 9 核心 pattern 语法。它是历史语言形式化，不是 2026.06 全部语法清单。Source: Francis et al., SIGMOD 2018, Fig. 3.*

对给定路径 $p$、图 $G$ 和变量绑定 $u$，论文把 pattern 满足关系写为：

$$
(p,G,u)\models\pi.
$$

它表示路径 $p$ 在图 $G$ 中满足 pattern $\pi$，且 pattern 的自由变量按 $u$ 绑定。语义层只回答“哪些路径和绑定是合法结果”，不指定应使用标签扫描、索引 seek、关系扩展还是并行执行。

![Students and teachers property graph](/images/blog/neo4j-graph-database-cypher-storage-engine/cypher-paper-fig04-students-teachers-graph.webp)

*论文用于解释固定长度和可变长度 `KNOWS` pattern 的小图。Source: Francis et al., SIGMOD 2018, Fig. 4.*

查询以表到表的函数组合定义。设 $T_0$ 是只含一个空 tuple 的表，查询 $Q$ 在图 $G$ 上的输出为：

$$
\operatorname{output}(Q,G)=\llbracket Q\rrbracket_G(T_0).
$$

每个 clause 接收一张记录表，产生下一张记录表；多个 clause 的语义通过函数复合连接。这个视角解释了为什么 Cypher 既像图模式语言，又保留了关系式的行、投影、聚合与 UNION。

![Core Cypher syntax](/images/blog/neo4j-graph-database-cypher-storage-engine/cypher-paper-fig05-core-syntax.webp)

*论文形式化的表达式、查询和子句语法核心。Source: Francis et al., SIGMOD 2018, Fig. 5.*

对 bag 而言，`UNION ALL` 使用 bag union，重复记录的 multiplicity 相加；`UNION` 再做重复消除。设记录 $u$ 在 bag $T$ 中的重数为 $\operatorname{mult}_T(u)$，则：

$$
\operatorname{mult}_{T_1\uplus T_2}(u)
=\operatorname{mult}_{T_1}(u)+\operatorname{mult}_{T_2}(u).
$$

这不是纯理论细节。错误的多对多 pattern 会把行数放大，后续 `count()`、写入或内存占用都会被放大。用 `DISTINCT` 在末尾消除重复只能修饰结果，未必消除中间执行成本。

## 9. `OPTIONAL MATCH`、`null` 与聚合的语义陷阱

![Formal semantics of Cypher queries](/images/blog/neo4j-graph-database-cypher-storage-engine/cypher-paper-fig06-query-semantics.webp)

*Cypher 查询、RETURN、UNION 与 clause 复合的形式语义。Source: Francis et al., SIGMOD 2018, Fig. 6.*

![Formal semantics of Cypher clauses](/images/blog/neo4j-graph-database-cypher-storage-engine/cypher-paper-fig07-clause-semantics.webp)

*MATCH、OPTIONAL MATCH、WITH、WHERE 与 UNWIND 的形式语义。Source: Francis et al., SIGMOD 2018, Fig. 7.*

对每一条输入记录 $u\in T$，普通 `MATCH` 连接所有满足 pattern 的扩展绑定；若没有扩展，该输入行消失。`OPTIONAL MATCH` 则在没有匹配时保留 $u$，并把 pattern 新引入的自由变量映射到 `null`。可以把简化语义写成：

$$
\operatorname{optional}(u,\pi)=
\begin{cases}
\operatorname{match}(u,\pi), & \operatorname{match}(u,\pi)\neq\varnothing,\\
\{u\cup\{x\mapsto\text{null}\mid x\in\operatorname{free}(\pi)\}\}, & \text{otherwise}.
\end{cases}
$$

若随后执行 `WHERE x.prop = 1`，`x` 为 null 时表达式结果不是 true，该行会被过滤。若目标是保留无 `x` 的左侧实体，条件通常应附着在可选匹配内部，或显式处理 null。

聚合也受 bag 影响。`count(*)` 计算输入行数，`count(x)` 忽略 null，`count(DISTINCT x)` 还会去重。一个人有 3 个订单、每个订单有 4 个商品，如果同时匹配订单和商品，用户会出现 12 行。此时 `count(order)` 得到 12 而非 3，除非查询在正确阶段聚合或去重。

现代 Cypher 的类型、子查询、路径模式和 GQL 对齐能力已经超出论文核心。论文价值在于给出“结果应该是什么”的精确基础。源码中的 parser、rewriter、planner 和 runtime 可以演进，只要它们保持这些可观察语义。

## 10. 从 Cypher 9 到 Cypher 25：语言标准化不等于实现相同

Cypher 论文记录了 openCypher 时代的 Cypher 9。当前 [Cypher Manual](https://neo4j.com/docs/cypher-manual/current/) 以 Cypher 25 为主，Cypher 5 已冻结；Neo4j 2026.02 起，新数据库默认使用 Cypher 25。Cypher 25 加入后续特性，并继续向 ISO GQL 靠拢。

“向 GQL 对齐”不能写成“Cypher 25 就是完整 GQL”。语言可能实现某些 mandatory GQL features，也可能保留 Neo4j 扩展、不同版本开关和迁移限制。应用应通过显式 `CYPHER 25`、兼容性测试和 deprecation 文档管理升级，而不是依据语法相似度猜测。

同样，语言语义与执行实现是两个层次。一个 `MATCH` 可以由 Community slotted runtime 执行，也可以在 Enterprise 中使用 pipelined 或 parallel runtime；结果语义应一致，内存布局、调度、算子融合和并行度则不同。源码阅读必须先找到语义契约，再看某个 runtime 如何兑现它。

新版本 vector `SEARCH` 子句也说明了这种演进。2026.01 起官方优先推荐通过 `SEARCH` 查询向量索引，2026.06 的 `vector-2026.06` provider 又增加多标签/多类型、多属性过滤和量化配置。把旧 procedure 示例当作永久接口，或把 2026.06 preview 配置倒写进 5.26，都可能造成错误文档。

## 11. 一次请求的系统主链

从驱动发出查询，到记录返回客户端，可以用下面的抽象链路表示：

```text
Application / Driver
        |
        v
Bolt handshake + PackStream messages
        |
        v
Bolt connection state machine / transaction state
        |
        v
Cypher parse -> semantic analysis -> rewrite
        |
        v
Cost planner + query cache -> logical plan
        |
        v
Physical runtime -> operators / pipelines
        |
        v
Kernel transaction -> storage engine API
        |
        v
Record stores + indexes + transaction state
        |
        v
Muninn page cache -> files / transaction logs
```

这条链路最重要的认识是：层与层之间传递的不是同一种对象。客户端传输的是协议消息；parser 产生 AST；planner 产生逻辑计划；runtime 处理 rows、slots 或 morsels；kernel 暴露游标与事务 API；storage engine 读取记录并生成 commands；page cache 管理文件页；WAL 保存可重放的事务表示。

把这些层折叠成“Cypher 直接遍历磁盘上的边”会掩盖许多真实成本：连接流控、解析、规划、事务可见性、索引更新、内存分配、页缺失、日志刷盘都可能成为瓶颈。反过来，理解层次后才能正确定位慢查询究竟慢在候选爆炸、计划错误、page cache miss，还是写事务的 fsync。

![Cypher query lifecycle](/images/blog/neo4j-graph-database-cypher-storage-engine/neo4j-doc-fig03-cypher-query-lifecycle.webp)

*官方查询生命周期把 query string、数据库统计、optimizer、logical plan 与 physical plan 连接起来。Source: Neo4j Cypher Manual 2026.06, CC BY 4.0.*

## 12. Bolt 与 PackStream：数据库边界从协议开始

Neo4j 驱动通常通过 Bolt 连接服务器，默认端口是 `7687`。Bolt 是有状态的二进制协议，可运行在 TCP 或 WebSocket 上；值使用 PackStream 编码。它不是“把 Cypher 字符串套一层 HTTP”，而是管理版本协商、认证、事务、结果拉取、失败和 reset 状态。

Bolt 握手以魔数 `60 60 B0 17` 开始，客户端和服务端协商协议版本。之后消息在连接状态机中流转，例如 `HELLO`/`LOGON`、`RUN`、`PULL`、`BEGIN`、`COMMIT`、`ROLLBACK`、`RESET`。不同协议版本的消息集合会变化，驱动必须按协商结果行为。

固定源码中的 [`BoltServer`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/bolt/src/main/java/org/neo4j/bolt/BoltServer.java) 负责连接器和相关服务的生命周期；`community/bolt` 下还能看到 protocol、negotiation、fsm、transport、routing 和 testing 等子模块。Netty 提供网络事件循环和 channel pipeline，Bolt 自己定义数据库会话语义。

结果不是必须一次性全部返回。`RUN` 建立结果流，`PULL n` 控制拉取数量，驱动可以用 fetch size 形成背压。若应用把百万行结果全部 materialize，Bolt 流控无法替应用消除内存占用；正确做法仍是限制结果、分页或流式消费。

显式事务在协议上跨多条消息存在。连接状态、数据库事务和驱动 session 相关，但不能简单等同：驱动可能维护连接池，一个 session 在不同时间租用连接；服务端事务由具体连接上的消息启动和结束。网络中断后，客户端也不能仅凭“没收到成功响应”判断事务一定未提交，幂等设计仍然重要。

Bolt 层还决定错误如何传播。语法错误发生在 `RUN` 后的编译阶段；运行时约束冲突可能在消费结果或提交时出现；连接进入 FAILED 状态后通常需要 `RESET` 或关闭。只在应用中捕获一个通用 exception，会失去重试、回滚和用户错误之间的区分。

## 13. 解析、语义分析与重写：在规划之前淘汰不合法查询

Cypher query string 进入执行引擎后，首先要变成可检查的内部表示。解析阶段识别关键字、表达式、pattern、子句和参数，构造 AST；语义分析检查变量是否定义、作用域是否有效、函数参数和表达式类型是否兼容；重写阶段把多种表面语法规范化，为 planner 降低组合复杂度。

例如：

```cypher
MATCH (u:User {id: $userId})-[:PURCHASED]->(p:Product)
WHERE p.price > $minPrice
RETURN p.category, count(*) AS purchases
ORDER BY purchases DESC
```

parser 只知道这是 MATCH、WHERE、RETURN 和 ORDER BY 的语法树。语义分析还要确认 `u`、`p`、`p.price`、聚合表达式和别名作用域。planner 才会基于 `:User(id)` 是否有唯一约束、价格选择率和关系统计决定起点。

参数化在这里非常重要。把 `$userId` 作为参数，能让查询结构稳定，减少注入风险并提高计划缓存复用。把用户输入拼进 label、property 或完整 pattern，不仅带来安全风险，还会产生大量不同 query text，使缓存碎片化。不能参数化的结构标识符应经过白名单映射，而不是字符串转义幻想。

源码树中的 `community/cypher/front-end` 承担 parser、AST、rewriter 和 semantic analysis 的主要职责；`community/cypher/cypher` 负责更上层编译与执行协调。这里不复制大量 Scala 实现，因为类名和 phase 会随版本演进；更稳妥的阅读方法是从 [`ExecutionEngine.scala`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/cypher/cypher/src/main/scala/org/neo4j/cypher/internal/ExecutionEngine.scala) 逆向查找 compiler phase，再回到 front-end。

重写不是任意优化。它必须保持可观察语义，尤其是 null、重复行、短路、更新顺序和错误。某些看似等价的关系代数变换，在 bag semantics 或 OPTIONAL MATCH 下并不等价。这也是形式语义论文对工业优化器仍有价值的原因。

## 14. 逻辑规划：搜索的不是路径，而是执行计划空间

逻辑 planner 接收经过规范化和语义检查的查询表示，结合 schema、索引、约束和统计信息，构造候选逻辑计划并估算成本。固定源码中的 [`CypherPlanner.scala`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/cypher/cypher/src/main/scala/org/neo4j/cypher/internal/planning/CypherPlanner.scala) 是理解编译主链的重要入口。

设输入候选基数为 $|R|$，某过滤条件的选择率为 $s\in[0,1]$，最简单估计是：

$$
|R'|\approx |R|\cdot s.
$$

多个条件若被错误假设独立，估算误差会逐层放大。图数据中的标签、关系类型和属性往往高度相关：`(:Person)-[:WORKS_AT]->(:Company)` 与随机节点对完全不同；超级节点又让平均度数失去代表性。成本模型只能依据统计摘要近似，无法知道所有数据相关性。

规划器的核心选择包括：

- 从标签扫描、全节点扫描还是索引 seek 开始；
- 先匹配哪一段 pattern；
- 使用 expand、join、apply 还是 cartesian product 连接子计划；
- 过滤、投影、聚合和排序放在哪一层；
- 如何处理 OPTIONAL、子查询、可变长路径和更新；
- 哪些 operator 能由目标 runtime 支持。

一个常见误区是认为 Cypher 中写在前面的 pattern 一定先执行。声明式查询只定义结果，planner 可以重排独立 pattern。若业务依赖副作用顺序，应使用语言明确规定的 clause 和事务语义，而不是依赖某次 `PROFILE` 的算子顺序。

另一个误区是“有索引就一定用索引”。如果索引选择率差、返回数据过多、排序与索引不匹配，扫描可能更便宜。强制 hint 可以修复某些统计失真，却也把当前数据分布假设固化进查询；升级和数据增长后必须重新验证。

## 15. 计划缓存与重规划：缓存的是编译成果，不是查询结果

固定源码中的 [`QueryCache.scala`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/cypher/cypher/src/main/scala/org/neo4j/cypher/internal/QueryCache.scala) 展示了 Cypher 缓存体系的协调逻辑。不同层可能缓存已解析 query、逻辑计划或可执行计划，具体 key 还受语言版本、planner/runtime 选项、数据库 schema 和参数类型等因素影响。

缓存命中不意味着返回旧数据。计划缓存复用的是“如何执行”，每次运行仍在当前事务视图中读取数据。把它和 Redis 一类结果缓存混为一谈，会产生严重的一致性误解。

参数化查询更容易复用计划：

```cypher
MATCH (u:User {id: $id}) RETURN u
```

若应用把每个 ID 直接拼进字符串，结构相同的查询也可能产生不同 cache key。动态生成大量 label、relationship type 和 projection 组合同样会制造计划缓存压力。

另一方面，计划不能永久有效。创建或删除索引、约束改变、统计显著变化、配置或语言版本变化，都可能要求失效或重规划。即使 schema 不变，数据从一万增长到十亿，原本合适的 join order 也可能失效。Neo4j 使用统计差异阈值和时间策略管理重规划，但具体行为应以目标版本文档和配置为准。

生产监控不能只看 cache hit ratio。高命中但计划错误，仍会稳定地执行慢查询；低命中可能来自合理的 ad-hoc workload。应结合 planning time、execution time、db hits、estimated/actual rows 差异和 query fingerprint 判断。

## 16. 物理执行运行时：同一逻辑计划的不同机器

官方 2026.06 文档列出 slotted、pipelined 和 parallel 三类 runtime。它们共享 Cypher 语义和逻辑计划输入，却采用不同的物理布局与调度方式。版本和 edition 边界必须放在比较之前：**Community 默认并实际开放的是 slotted；pipelined 和 parallel 属于 Enterprise/Aura 产品能力，完整实现不在本文固定 Community 仓库中。**

### 16.1 Slotted runtime

slotted 为逻辑变量预分配 slot，把节点 ID、关系 ID、原始值和引用值放在结构化 execution context 中，避免所有值都经通用 map 查找。其执行模型接近 Volcano iterator：上层 operator 向子 operator 拉取一行，再继续向上生产。

源码中的 `community/cypher/slotted-runtime` 包含 pipe mapper、slot configuration 和算子实现；`community/cypher/interpreted-runtime` 仍保存大量 expression 与 query-context 基础设施。当前文档所说“原始 interpreted runtime 已退休”，并不意味着仓库里名为 `interpreted-runtime` 的所有内部代码都应该消失；模块名、公共运行时选项和代码复用是三个不同概念。

slotted 的优点是规划快、覆盖面完整、适合短事务查询。缺点是逐行 pull 与虚调用可能降低 CPU cache locality，复杂查询的 operator 边界也难以融合。

### 16.2 Pipelined runtime

pipelined runtime 把多个 operator 融合为 pipeline，以 morsel 为批次处理行，减少逐行函数调用并改善局部性。某些 pipeline breaker，例如排序、聚合或 eager materialization，会形成缓冲边界。它仍通常让单个查询在单线程推进，不等同于“有 pipeline 就自动多核并行”。

完整 pipelined runtime 属于 Enterprise 源码范围。Community 仓库可见逻辑计划、runtime-util 和接口，不足以复现官方文档描述的完整执行图。本文因此只解释公开文档中的概念，不给出不存在于固定仓库的伪源码路径。

### 16.3 Parallel runtime

parallel runtime 与 pipelined 架构相关，但允许多个 worker 执行 pipeline task 和分区算子，使一个读查询利用多核。2026.06 文档明确把它标为 Enterprise；它只支持读取，更新查询或已改变事务状态的场景不支持。并发查询很多时，单查询加速还可能以整体吞吐下降为代价。

“parallel 更快”不是无条件结论。图局部、几十毫秒完成的索引锚定查询可能被调度开销拖慢；全图、长时间、可分区读取更可能获益。是否使用应由真实 `PROFILE`、并发水平和 SLA 决定。

## 17. `EXPLAIN` 与 `PROFILE`：从猜测回到执行证据

`EXPLAIN` 只生成计划，不执行查询；`PROFILE` 执行查询并附加实际 rows、db hits、time、memory、page cache 等观测。写查询的第一轮性能审查，应先回答四个问题：从哪里起步、每层基数如何变化、哪里出现全扫描/笛卡尔积、估算与实际相差多少。

```cypher
PROFILE
MATCH (u:User {id: $userId})-[:PURCHASED]->(p:Product)
WHERE p.price > $minPrice
RETURN p.category, count(*) AS purchases
ORDER BY purchases DESC
```

![Neo4j query plan parts](/images/blog/neo4j-graph-database-cypher-storage-engine/neo4j-doc-fig04-query-plan-parts.webp)

*官方示例中的 planner/runtime 元数据、operator 表和查询摘要。Source: Neo4j Cypher Manual 2026.06, CC BY 4.0.*

常见字段应这样读：

| 字段 | 含义 | 常见误读 |
| --- | --- | --- |
| Estimated Rows | planner 对算子输出基数的估计 | 不是实际读取行数 |
| Rows | 执行时算子输出数量 | 不等于返回客户端的最终行数 |
| DB Hits | 通过 kernel 访问图数据的计数型指标 | 不是精确磁盘 I/O 次数 |
| Page Cache Hits/Misses | 对数据库页缓存的访问结果 | 不等于 OS block device 指标 |
| Memory | 算子跟踪的内存 | 不必等于 JVM 进程全部 RSS |
| Time | 运行时统计 | pipelined/parallel 中可能按 pipeline 汇总，不宜逐行机械相加 |

最有价值的信号通常是 Estimated Rows 与 Rows 的数量级差异。如果 planner 估计 10 行，实际产生 100 万行，它可能选择错误 join order、错误索引或过小内存策略。优化应先修正数据模型、统计、约束和 query shape，而不是只提高 JVM heap。

`DB Hits` 也不是“越低越快”的绝对尺度。一百万次命中若都在热 page cache 且顺序局部，可能比少量随机 page fault 更快；昂贵表达式、排序和网络传输也不完全反映在 hits 中。应把计划指标与事务日志、page cache、GC、CPU 和客户端消费速度共同观察。

写查询执行 `PROFILE` 会真的产生副作用，不能在生产数据上随意测试。安全方法是在只读副本、测试库或显式回滚事务中验证，并确认目标 runtime、语言版本和参数分布与生产一致。

## 18. `ExecutionEngine`：语言层与数据库事务的汇合点

[`ExecutionEngine.scala`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/cypher/cypher/src/main/scala/org/neo4j/cypher/internal/ExecutionEngine.scala) 和 Java compatibility wrapper [`ExecutionEngine.java`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/cypher/cypher/src/main/java/org/neo4j/cypher/internal/javacompat/ExecutionEngine.java) 是源码阅读的好入口。它们不承担所有 parser/planner/runtime 逻辑，而是协调查询执行所需的 compiler、cache、transactional context、monitor 和 result subscriber。

这里可以看到数据库系统常见的“控制面”职责：

- 接收 query text、参数、pre-parser 选项和事务上下文；
- 选择语言版本、planner 和 runtime；
- 调用 compilation pipeline 或命中 cache；
- 建立执行订阅与资源跟踪；
- 把 runtime result 转成 Java/协议层可消费结果；
- 把语法、规划、执行、终止和事务错误翻译到公共异常模型。

执行引擎不是全局无状态函数。计划依赖数据库 schema 和统计；执行依赖当前事务、security context、memory tracker、clock 和配置；结果消费可能持续到方法返回之后。若应用得到 lazy result 后长期不消费，它可能持续占用事务、游标、锁和网络连接。

从源码阅读角度，应该沿三条方向展开：向上跟 Java API、Bolt state machine 和 server；横向跟 compiler/cache/runtime；向下跟 `TransactionalContext`、kernel transaction 和 data read/write API。只读一个 `execute()` 方法无法理解真正的数据访问。

## 19. `GraphDatabaseFacade` 与 Kernel API：公共对象不是存储记录

[`GraphDatabaseFacade`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/kernel/src/main/java/org/neo4j/kernel/impl/factory/GraphDatabaseFacade.java) 实现面向嵌入式/内部调用者的数据库服务 facade。它负责数据库可用性、事务创建、查询执行和事件注册等公共入口，却不会直接解析 record store 文件。

公共 `Node`、`Relationship` 和 `Transaction` 对象是一层 API 抽象。真正数据访问经过 kernel transaction 暴露的 read、write、token、schema 和 cursor API。这样做有三点价值：

第一，语言层和 procedure 不绑定具体 store format。Community 的 `RecordStorageEngine` 可以通过统一 SPI 服务 kernel；Enterprise 可以提供不同实现。

第二，事务状态可以覆盖底层存储。在提交前，新建节点、删除关系或属性修改先存在 transaction state 中；读取必须把 store snapshot 与本事务未提交变更组合起来。

第三，访问控制、终止、内存跟踪、锁和统计可以在 kernel 边界统一实施。若 procedure 绕过 kernel 直接读取文件，就会破坏事务可见性和安全约束。

源码中的 [`StorageEngine`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/kernel-api/src/main/java/org/neo4j/storageengine/api/StorageEngine.java) 是理解这一边界的关键接口。它定义生命周期、store access、command creation/application、indexing、checkpoint/flush 协作和其他能力，但不规定 record 的具体布局。

## 20. 事务生命周期：一次查询可以短，事务不能含糊

[`KernelTransactionImplementation`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/kernel/src/main/java/org/neo4j/kernel/impl/api/KernelTransactionImplementation.java) 是 Community 内核中最值得耐心阅读的类之一。它连接事务状态、安全上下文、锁、游标、内存、超时、提交、回滚和终止。

一次事务大致经历：

```text
initialize
  -> acquire cursors / read snapshot
  -> execute reads and accumulate transaction state
  -> validate constraints and conflicts
  -> transform state into storage commands
  -> append transaction representation to WAL
  -> apply commands to stores and indexes
  -> close resources / release locks
```

只读事务可以跳过命令生成和写日志，却仍然需要可见性、游标和资源生命周期。写事务在提交前的很多操作只改变内存中的 transaction state；`COMMIT` 才把逻辑变化转成可恢复的持久化过程。

ACID 四个字母不能代替具体语义：

- **Atomicity**：一个事务的命令要么整体提交，要么不成为已提交状态；崩溃恢复不能留下半个业务事务。
- **Consistency**：数据库约束从一个合法状态过渡到另一个合法状态；业务规则仍需应用或约束表达。
- **Isolation**：并发事务通过锁和可见性规则隔离；默认行为不是任意工作负载下的自动 serializable。
- **Durability**：成功提交的事务在故障后可由日志和 store 恢复；这依赖文件系统、刷盘策略和运维配置。

事务有 timeout 和 termination。超时并不等于 Java 线程立刻被强杀，执行算子需要在检查点观察终止状态并释放资源。自定义 procedure 若执行不可中断的外部调用，可能延迟取消。生产上必须同时配置数据库事务超时、驱动超时和上游请求 deadline，并处理三者的竞态。

## 21. 并发控制、锁与死锁：图模型不会消灭竞争

当多个事务更新同一节点、关系、schema 或索引约束时，Neo4j 需要协调访问。锁可以在实体或更高层资源上取得；不同操作的锁粒度和顺序由内核决定。应用不应该依赖未文档化的锁顺序来保证业务逻辑。

一个经典 lost update 场景是两次并发“读余额、在应用中加一、再写回”。若更新表达式没有在同一事务中建立正确依赖，两次事务可能覆盖。更稳妥的 Cypher 让读写依赖在数据库内显式发生，并用约束或重试处理冲突。

死锁并不表示数据库损坏，而是两个事务形成等待环。数据库会选择一个事务终止，使另一个继续。客户端必须把 deadlock 和 transient failure 视为可重试类别，但重试前提是事务函数幂等，且外部副作用没有在数据库提交之前不可逆地发生。

长事务尤其危险：它持有锁和事务状态更久，增加日志、内存和冲突压力；批量写入几十万实体时，一个超大事务失败会浪费全部工作。应按业务原子性切分批次，而不是机械追求“事务越大越一致”。

读取也并非脱离并发语义。一次查询遍历多条关系时，要看到与目标版本一致的记录；page cache eviction 还要保留必要的 version context。源码的 transaction、cursor 和 page cache 版本协作，正是为了避免把不同时间状态随意拼接。

## 22. Storage Engine SPI：把逻辑变化翻译成可恢复命令

Kernel 的 transaction state 表示“业务上改变了什么”，storage engine 负责把它转换为“具体存储要写什么”。[`StorageEngine`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/kernel-api/src/main/java/org/neo4j/storageengine/api/StorageEngine.java) 把这一层抽象成 SPI。

对 Community [`RecordStorageEngine`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/record-storage-engine/src/main/java/org/neo4j/internal/recordstorage/RecordStorageEngine.java) 而言，提交要根据节点、关系、属性和 token 变化生成 record commands；命令先形成 transaction representation，再交给 log append 和 store apply。

这个两阶段表示有重要意义：事务日志不需要保存一份新数据库文件，它保存足以重放状态变化的命令；崩溃恢复可以按提交顺序重新应用；集群版也可以复制逻辑/物理命令流，但其具体协议属于 Enterprise 边界。

Storage engine 还与索引更新协作。属性或标签变化可能影响 range、text、full-text、vector 或 token lookup index。索引更新可以与 store command 一起纳入事务语义，部分 eventually consistent full-text 配置则有不同可见性。不能把“图记录提交成功”和“所有辅助索引在任何配置下立即一致”无条件画等号。

SPI 也解释了为何本文不能根据 RecordStorageEngine 推断 block store。两个 engine 可以共享 kernel 接口，内部 record layout、property inlining、关系组织和迁移能力却不同。开源实现是理解接口和 Community 行为的证据，不是闭源实现的替身。

## 23. Record Store 与 `aligned`：Community 实际在读写什么

当前官方 [Store formats](https://neo4j.com/docs/operations-manual/current/database-internals/store-formats/) 文档说明：Community 默认 `aligned`；Enterprise 推荐并默认 `block`；`standard` 和 `high_limit` 从 5.23 起弃用。本文固定仓库中的 `community/record-storage-engine` 对应 record-based 路线。

Record store 把不同实体拆成专门 store 文件和固定/动态记录。节点记录保存 ID、使用状态、标签信息和关系入口；关系记录保存 type、起止节点及连接结构；属性记录组成链，并在字符串或数组过大时引用 dynamic store；token store 保存 label、relationship type 和 property key 名称。

固定大小记录便于按 ID 计算文件偏移：

$$
\operatorname{offset}(id)=\operatorname{headerSize}+id\cdot\operatorname{recordSize}.
$$

这是理解“邻接访问不需要每跳全局索引”的基础之一：已知下一 record ID 后，可以定位对应文件页。不过记录仍通过 page cache 访问，可能发生 page fault；属性链和动态记录也可能导致多次读取。

标签和小属性是否内联、关系如何组织、记录大小和 ID 上限都由具体 format version 决定。应用不应解析 store file 作为稳定公共 API。离线工具必须与版本兼容，升级和迁移应使用 `neo4j-admin`，而不是自己改二进制文件。

## 24. Dense Node、关系组与超级节点

对低度节点，关系可以通过链式结构遍历。节点度数升高后，若所有关系仍混在一条长链中，按 type 和方向寻找子集会浪费大量扫描。Record storage 使用 dense-node/relationship-group 一类组织，把关系按类型以及 incoming/outgoing/loop 分组，提高目标扩展效率。

“dense”的阈值是内部配置和版本问题，不应在业务代码中硬编码。更重要的是，即使关系分组能快速找到 `:FOLLOWS` 的起点，遍历一个拥有千万粉丝的明星节点仍要处理大量结果。存储结构优化了入口，不会把输出基数变小。

超级节点常见于国家、热门商品、公共 tag、全局租户、默认权限组。缓解策略包括：

- 在关系上增加可选择过滤的时间或状态，并建立合适索引；
- 通过中间分桶节点按时间、地域或业务域拆分；
- 从更稀疏的一端开始匹配；
- 限制路径长度和返回数量；
- 把全局统计移到分析系统，而非每次在线展开；
- 使用 `PROFILE` 验证实际 rows，而不是只看语法简洁。

分桶也有代价：图模型更复杂，路径多一跳，写入需要选择 bucket，跨 bucket 查询要合并。它不是普遍最佳实践，而是针对已证实基数瓶颈的物化结构。

## 25. Muninn Page Cache：数据库页与 JVM heap 之外的核心内存

[`MuninnPageCache`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/io/src/main/java/org/neo4j/io/pagecache/impl/muninn/MuninnPageCache.java) 是 Community IO 层的关键实现。Neo4j 将 store 文件映射为 paged files，查询通过 page cursor pin 住页并读取记录；页不在 cache 时从文件 fault 进来，脏页随后 flush。

Page cache 与 JVM heap 职责不同。heap 保存查询对象、计划、事务状态和运行时结构；page cache 保存数据库文件页，通常使用堆外/本地内存。把所有可用内存都给 heap，反而会压缩图工作集的 page cache，增加 fault 和 GC 压力。

常用命中率可以写为：

$$
\operatorname{hitRatio}=\frac{H}{H+M},
$$

其中 $H$ 是 hits，$M$ 是 misses。但比率必须结合访问量和预热阶段。99% 命中率下每秒一百万次访问仍有一万次 miss；数据库重启后的冷缓存不能和稳定状态直接比较。

Page cache 之下还有操作系统文件缓存和存储设备。一次 Neo4j page-cache miss 不必然等于一次物理磁盘读，OS 可能仍有该页；一次 hit 也不等于查询一定快，CPU、锁、表达式和结果传输仍可能瓶颈。因此官方 `PROFILE` 的 page-cache 指标适合定位访问模式，主机层 I/O 指标负责验证设备压力。

Pin/unpin 是并发正确性的一部分。页被游标使用时不能随意回收；脏页 flush 要和 writer、checkpoint、版本上下文协调；eviction 还要避免让快照读取拼接不一致页面。阅读 `MuninnPageCache` 时，应把它视为事务系统的一部分，而不是普通 LRU map。

## 26. WAL：日志先于数据页，提交才有恢复基础

Write-Ahead Logging 的核心不变量是：**描述已提交修改的日志必须在相关数据页被认为持久之前，先达到所需的持久化边界。** 简化表示为：

$$
\operatorname{durable}(\text{log record})
\prec
\operatorname{durable}(\text{data page}).
$$

这不要求每次提交都把全部脏数据页写回磁盘。事务命令追加到日志并完成必要 flush 后，可以确认提交；store page 由后台或 checkpoint 后续刷盘。崩溃后，系统从日志重放尚未反映到 store 的已提交命令。

固定源码中的 [`TransactionLogWriter`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/kernel/src/main/java/org/neo4j/kernel/impl/transaction/log/TransactionLogWriter.java) 负责把事务表示追加为日志条目。周围包还包含 log files、channels、rotation、entry reader/writer、checkpoint 和 recovery 组件。

一次写事务的简化提交链是：

```text
transaction state
  -> storage engine creates commands
  -> transaction representation
  -> append to transaction log
  -> required force/flush boundary
  -> apply commands to stores and indexes
  -> publish committed transaction id
```

真实实现会处理 batch、并发提交、校验、checksum、内存跟踪和错误恢复，顺序也不应只凭这张抽象图推断。这里的价值是理解为什么“数据页还没全部写盘”仍可以安全提交，以及为什么损坏或丢失 transaction logs 会破坏恢复链。

## 27. Checkpoint、恢复与日志裁剪

Checkpoint 的目标是缩短恢复距离，并建立 store 与 transaction log 的一致位置。它会协调 page cache flush，把某个安全点之前的修改推进到 store，同时记录 checkpoint 信息。它不等于暂停所有业务、复制完整数据库或替代备份。

设最近 checkpoint 对应事务 $t_c$，崩溃前最后 durable commit 为 $t_d$，恢复主要需要处理区间：

$$
(t_c,t_d].
$$

Checkpoint 越频繁，恢复区间通常越短，但 flush 和 I/O 压力更高；越稀疏，正常运行写放大可能更低，故障恢复时间却增加。配置要结合写入率、设备能力和 RTO，而不是只追求一个方向。

恢复启动时需要识别完整日志条目、最后 checkpoint、已应用和未应用事务，重放可恢复命令，并处理崩溃中断的日志尾。校验和与版本字段帮助发现撕裂或不兼容条目。若底层存储违反 fsync 语义、日志文件被人工删除或复制快照不一致，数据库软件无法凭空恢复。

日志裁剪根据保留策略删除不再需要的历史日志。保留过少会影响 point-in-time 工具、增量备份或故障分析；保留过多会占满磁盘。Community 与 Enterprise 的备份/集群需求不同，不能照搬一个通用阈值。

Checkpoint 不是备份。它让同一 store 在崩溃后更快自恢复；备份要应对磁盘丢失、误删除、恶意破坏和机房故障，并且必须经过恢复演练。Community 缺少 Enterprise 在线备份能力时，团队更需要明确停机复制、文件系统快照一致性或升级版本的方案。

## 28. GB+Tree：图存储仍然需要有序索引

“原生图数据库”不代表不用树索引。图邻接适合从已知节点沿边扩展，但查询首先要找到锚点，约束要检查唯一性，token 要映射标签/类型，计数和 schema 也需要持久结构。固定源码中的 [`GBPTree`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/index/src/main/java/org/neo4j/index/internal/gbptree/GBPTree.java) 是 Neo4j 的通用持久 B+Tree 家族实现。

GB+Tree 名称中的 G 指 generation。树页使用 generation 信息和 recovery pointer 处理 crash-consistent 更新，配合 page cache、checkpoint 和 free-list 管理。它不是一个把 Java `TreeMap` 序列化到文件的简单容器。

B+Tree 查找的抽象复杂度为：

$$
O(\log_B N),
$$

其中 $B$ 是每个页节点容纳的分支数。数据库页通常能容纳很多 key，因此树高较低；实际性能仍取决于页缓存、key 编码、range scan 长度和并发写。

GB+Tree 被多个内部组件复用，不应把它和某一种公开 Cypher index 一一等同。公开 range index、token lookup、counts 或 id mapping 可能使用不同 provider 和 layout；理解 `GBPTree` 的价值在于掌握 Neo4j 如何在统一 page-cache 基础设施上构建 crash-aware 有序结构。

树索引与图扩展是互补关系。典型计划先通过索引把十亿节点缩成一个或几十个锚点，再沿关系扩展。如果没有索引，数据库可能先扫描所有 `:User`；如果只有索引却没有关系结构，多跳连接仍要重建关联。Neo4j 的性能来自两者组合，而不是二选一。

## 29. 基础索引类型：不同问题需要不同访问结构

Neo4j 2026.06 把索引分为 search-performance indexes 与 semantic indexes。前者包括 range、text、point、token lookup；后者包括 full-text 与 vector。名字相似不代表它们支持同一谓词。

| 索引类型 | 典型用途 | 不应期待的能力 |
| --- | --- | --- |
| Token lookup | 按 label 或 relationship type 找实体 | 按属性值排序或全文搜索 |
| Range | 等值、范围、部分排序、前缀等受支持谓词 | 任意 substring 或语义相似 |
| Text | 字符串特定搜索谓词 | 数值范围和向量相似 |
| Point | 空间点和距离/包围区域 | 通用 GIS 拓扑系统 |
| Full-text | 多属性文本、analyzer、BM25 风格相关性 | 强事务唯一性、向量语义 |
| Vector | embedding 近邻、过滤后 ANN | 精确图路径、关键词倒排 |

Token lookup 是图 pattern 最基础的入口之一。`MATCH (n:Person)` 要找到拥有 `Person` label 的节点；`MATCH ()-[r:KNOWS]->()` 要按关系类型定位候选。它不是用户通常显式设计的属性索引，却会影响 label/type scan。

Range index 适合 `n.email=$email`、时间区间、数值范围和部分排序。唯一约束通常由相应 backing index 支持。创建 index 后要经历 population，`CREATE INDEX` 命令返回不意味着它立刻 ONLINE；部署脚本应等待并检查状态。

Point index 针对 Neo4j point value 和坐标参考系，不等于完整地理信息系统。复杂 polygon、拓扑、地图投影和海量轨迹分析仍可能需要专用系统。

索引不是免费的副本。每次写入都要维护相关 index，population 会扫描已有数据，schema 变更会占用 I/O、CPU 和磁盘。只因为某属性出现在 `WHERE` 就创建索引，会增加写放大与规划空间；应根据查询频率、选择率和维护成本选择。

## 30. Full-text：Lucene 倒排索引是辅助结构，不是图遍历

Community 源码的 `community/lucene-index/.../fulltext` 包含 [`FulltextIndexProvider`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/lucene-index/src/main/java/org/neo4j/kernel/api/impl/schema/fulltext/FulltextIndexProvider.java)、reader、writer、analyzer loader、transaction state 和 eventually-consistent updater。

全文索引可以跨一个或多个 label/type、一个或多个字符串属性构建，并使用 analyzer 做分词、归一化、停用词和语言处理。标准 Cypher 示例是：

```cypher
CREATE FULLTEXT INDEX product_search
FOR (p:Product)
ON EACH [p.name, p.description]
```

查询结果包含实体和相关性 score。BM25 的核心直觉是：词在当前文档中频繁出现提高得分，词在全语料中过于常见降低区分度，同时用文档长度归一化。简化形式可写为：

$$
\operatorname{score}(D,Q)=
\sum_{q\in Q}\operatorname{IDF}(q)
\frac{f(q,D)(k_1+1)}
{f(q,D)+k_1\left(1-b+b\frac{|D|}{\operatorname{avgdl}}\right)}.
$$

这个公式解释相关性方向，不代表 Neo4j/Lucene 当前所有内部参数。Analyzer 选择同样重要：使用错误语言、停用词或 keyword analyzer，可能让看似正确的 index 完全不符合业务搜索。

Full-text 可以配置 eventually consistent 更新，把索引维护移出提交关键路径。收益是降低写事务延迟，代价是刚提交的数据短时间内可能搜索不到。应用若用全文搜索做授权、唯一性或强一致读后写检查，会犯层次错误。强约束应由事务 schema constraint 和精确属性访问承担。

全文命中只产生候选实体，后续仍可沿图关系过滤和扩展。例如先找语义相关商品，再限制它们属于用户有权限的目录。倒排索引不会执行关系路径；图扩展也不会替代自然语言文本检索。

## 31. Vector Index：HNSW、过滤与 2026.06 版本边界

Community 源码的 `community/lucene-index/.../vector` 包含 [`VectorIndexProvider`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/lucene-index/src/main/java/org/neo4j/kernel/api/impl/schema/vector/VectorIndexProvider.java)、[`VectorIndexReader`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/lucene-index/src/main/java/org/neo4j/kernel/api/impl/schema/vector/VectorIndexReader.java)、config、quantization 与 similarity 实现；[`VectorIndexProcedures`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/procedure/src/main/java/org/neo4j/procedure/builtin/VectorIndexProcedures.java) 保留 procedure 入口。

余弦相似度为：

$$
\operatorname{cos}(x,y)=\frac{x\cdot y}{\|x\|_2\|y\|_2},
$$

欧氏距离为：

$$
d_2(x,y)=\sqrt{\sum_i(x_i-y_i)^2}.
$$

HNSW 用多层近邻图近似搜索，不扫描全部向量。参数影响建图成本、内存、查询延迟和 recall。ANN 的“最近”是 embedding 空间中的近似，不等于业务相关、事实正确或图关系合法。

2026.06 当前文档的关键事实包括：

- 推荐通过 Cypher `SEARCH` 查询 vector index，旧 procedure 仍有兼容价值；
- 当前最丰富 provider 是 `vector-2026.06`；
- 支持 node 多标签、relationship 多 type 和额外标量属性过滤；
- dimension 范围为 1 到 4096，建议显式指定；
- similarity 支持 `cosine` 和 `euclidean`；
- quantization type 在 2026.06 演进，binary quantization 和 search expansion 等能力含 preview 边界；
- Community 可索引 `LIST<INTEGER|FLOAT>` embedding；原生 `VECTOR` property 依赖 Enterprise/Aura 的 block format。

一个概念示例：

```cypher
CREATE VECTOR INDEX product_embedding
FOR (p:Product) ON (p.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}}
```

实际查询语法必须按目标 2026.06 文档确认，尤其是 `SEARCH`、过滤和 provider 配置。本文不把 preview 选项写成长期稳定 contract。

Vector 与图的最佳组合通常是“语义候选 + 结构约束”：先用 embedding 找相似文档/商品/实体，再按租户、权限、时间、关系或路径过滤；或者先用图缩小业务邻域，再对邻域向量重排。把全库向量 top-k 直接喂给 LLM，仍可能跨租户、跨时间或跨实体类型召回错误内容。

## 32. Schema 与 Constraints：图自由不等于无契约

属性图允许不同节点拥有不同属性，这种 schema flexibility 有利于演进，却不等于生产数据不需要约束。Neo4j schema 可以声明索引和约束，planner 也会利用它们推断唯一性和基数。

Community 最常用的是属性唯一性约束：

```cypher
CREATE CONSTRAINT user_id_unique
FOR (u:User) REQUIRE u.id IS UNIQUE
```

唯一性不自动要求属性存在：没有 `id` 的 `:User` 不一定违反 uniqueness。属性存在约束、属性类型约束、node/relationship key 和更多结构化约束在当前文档中属于 Enterprise/Cypher 25 的不同能力边界，不能写成 Community 默认可用。

约束的价值不只是拒绝坏数据。唯一性 backing index 给 planner 一个高选择性锚点；`MERGE (u:User {id:$id})` 在并发写入时也需要约束避免重复实体。只靠应用先查后写，会产生竞态。

创建约束需要扫描已有数据，是原子但可能耗时的 schema operation。若历史数据有冲突，创建会失败。生产迁移应先离线审计、评估临时磁盘和锁影响，再创建并等待 ONLINE，不能把 DDL 混在无观测的应用启动脚本里。

约束仍不能表达所有业务规则。例如“订单总额等于明细之和”“只有项目成员能创建任务关系”“时间区间不能重叠”通常需要事务查询、procedure 或应用验证。数据库 consistency 是已声明约束的一致，不是业务真理自动成立。

## 33. Procedures、Functions 与扩展边界

Neo4j procedure 是在数据库进程中执行的扩展入口，function 则可嵌入表达式。`community/procedure-api` 定义注解和公共接口，`community/procedure` 包含内置 procedure，server 启动时负责发现、注册与权限控制。

一个 procedure 可以访问 transaction、log、database service 等注入上下文。它适合封装难以用纯 Cypher 表达的数据库内逻辑，却扩大了故障面：死循环、过量内存、阻塞网络调用和线程不安全代码都运行在数据库进程中。

生产扩展应做到：

- 固定兼容的 Neo4j API 和 JDK 版本；
- 对输入、label/type/property 名称做白名单验证；
- 使用当前事务，不绕过 kernel 读写文件；
- 响应 transaction termination；
- 设定内存、行数和外部调用上限；
- 用最小执行权限注册，不默认 boosted；
- 在升级前跑集成测试并准备移除回滚。

APOC 是独立项目，虽然很多 Neo4j 部署都会安装，但它不等于核心 Cypher，也不随 `neo4j/neo4j` 固定 commit 自动版本匹配。Graph Data Science 同样是独立产品，不应把 PageRank procedure 或投影图内存模型归入本文的 core storage engine。

内置 vector procedure 的存在也不能证明所有 vector API 都是 procedure-first。当前 Cypher `SEARCH` 正在成为主要声明式入口，procedure 更适合兼容和管理操作。阅读源码时应先看当前手册，再决定某个 class 是公共主路径还是遗留接口。

## 34. 数据导入：事务写入、批量导入和复制不是一回事

Neo4j 支持多种写入路径，适用前提不同。

**普通事务写入**通过 Driver、Bolt 和 Cypher 执行，保留完整约束、事务和在线可见性，适合增量业务流量。大量小事务会增加往返和提交开销，超大事务又占用过多内存和锁，因此通常按可重试批次提交。

**`LOAD CSV` 或批处理子查询**仍经过数据库事务，适合在线导入和数据转换。输入来源、并发数、错误处理和 `CALL { ... } IN TRANSACTIONS` 的批量大小需要压测。

**`neo4j-admin database import full`** 是离线全量构建路径，绕过普通在线事务流程，适合新数据库的大规模初始载入。它不是对已有活跃数据库随时执行的高吞吐写 API。Enterprise 的 incremental import 和 `block` 格式选项又有单独许可边界。

**database copy/store migration** 面向现有数据库文件和格式转换。当前文档警告格式迁移通常离线、I/O 密集，并可能重建索引；transaction ID 是物理存储语义的一部分，不能用文件复制拼出逻辑合并。

导入前最重要的是 ID 和幂等策略。CSV 行号不是稳定业务 ID；把人名当唯一键会错误合并；`MERGE` 没有唯一约束也可能并发重复。建议为实体设计稳定外部 key，为关系设计来源/时间/事件 ID，并记录导入批次。

## 35. 一致性检查、Store Copy 与升级

`community/consistency-check` 和 `community/import-tool` 暴露了数据库运维的另一面：一个 store 即使文件可打开，也可能存在记录引用、索引、counts 或 schema 不一致。离线 consistency check 用于检查结构，不等于业务数据验证。

一致性检查可能需要大量顺序 I/O、临时空间和时间。生产计划应明确在哪个副本或停机窗口运行、报告如何保存、发现错误后用什么备份恢复。不要在磁盘已接近满载时临时运行重型检查。

升级包括软件版本、Cypher 语言、store format、index provider、procedure 和 driver 多条兼容轴。一个稳健过程应区分：

1. 在当前版本完成备份与恢复演练；
2. 检查弃用和移除项；
3. 在副本上升级并等待 index population/migration；
4. 运行核心读写、约束、procedure 与计划回归；
5. 比较 store size、page cache、日志和延迟；
6. 准备真正可执行的回滚，而不是只保留旧二进制。

Store format 通常不能简单降级。当前文档已经给出 `standard`/`high_limit` 弃用和迁移期限，团队应在仍有受支持迁移工具时完成路线，而不是等到新版本启动硬阻断后再处理。

## 36. Server、配置与数据库生命周期

`community/server` 提供 HTTP server、routing、web resources 和 lifecycle 集成；`community/neo4j`、`community/dbms`、`community/kernel` 共同完成进程启动、数据库管理和依赖装配。Bolt 才是 Driver 查询主协议，HTTP 接口不应被误写成所有客户端的核心执行路径。

配置从 `neo4j.conf`、环境变量和命令行进入 typed settings。文件路径、listen/advertised address、page cache、heap、transaction logs、timeouts 和 connectors 都有独立语义。容器部署中最常见错误是只修改 listen address，却忘记 advertised address，导致客户端收到不可达路由地址。

Community 安装可以有且仅有一个标准数据库，另有 system database 承担管理元数据。Enterprise 才支持任意多个标准数据库、composite database 等高级数据库拓扑。应用用 label 模拟租户，与真正 database isolation 不是同一安全等级。

数据库 lifecycle 包括 create/open/start/stop/drop 和 availability。文件存在不代表数据库已经 online；index population、recovery 或迁移期间，状态和可服务能力会变化。健康检查应区分进程存活、Bolt 接受连接、目标数据库 online 和业务查询成功。

配置变更也分动态和需要重启。把任何 `neo4j.conf` 修改都当作热加载会产生错误预期；实施时应以 2026.06 setting reference 的 dynamic 标记为准，并在变更管理中记录旧值和回滚。

## 37. Community 安全：认证存在，细粒度授权不是默认能力

Community 支持用户认证和基础密码管理，但当前官方能力矩阵把角色管理、细粒度 privilege、LDAP/Active Directory、Kerberos、property/subgraph access control 等列为 Enterprise 能力。Community 用户具有隐含的广泛管理能力，不能用多个 Community user 模拟真正的 RBAC。

这带来直接的部署结论：

- 不应把 Community Bolt 端口直接暴露给不可信网络；
- 应使用 TLS、私网、防火墙和受控 API 层；
- 应用服务到数据库使用专用凭据，禁止共享默认密码；
- 业务授权必须在服务端强制，而不是只依赖客户端隐藏查询；
- 自定义 procedure 需要额外审查执行权限；
- 日志、query text 和参数可能包含 PII，需配置脱敏与保留周期。

参数化能防止“值位置”的 Cypher injection，却不能自动保护动态 label、type、property 名称和 procedure 名称。以下反模式仍然危险：

```text
"MATCH (n:" + userLabel + ") RETURN n"
```

结构化标识符应从受控枚举映射；若产品允许用户编写任意 Cypher，它本质上提供数据库查询权限，需要隔离数据库、资源限制和审计，而不是普通搜索框。

Graph 层的租户属性也不是访问控制。即使每个节点都有 `tenant_id`，漏写一个 `WHERE` 或 pattern 就可能跨租户。Enterprise subgraph privilege、独立数据库或应用层授权各有成本，但至少要有一层不可由普通查询遗漏的强制边界。

## 38. Enterprise 集群：文档可读，源码不可假装开放

![Neo4j Enterprise cluster operational view](/images/blog/neo4j-graph-database-cypher-storage-engine/neo4j-doc-fig05-enterprise-cluster-operational-view.webp)

*Enterprise 集群概念图，展示不同服务器承载数据库副本和 system database。该能力不在本文固定 Community 仓库中。Source: Neo4j Operations Manual 2026.06, CC BY 4.0.*

Enterprise 集群让数据库副本分布到多台 server，使用共识和复制机制维持可用性。不同数据库可以有不同 leader/follower 拓扑；system database 管理集群元数据；Driver 通过 routing table 将写请求发往适当 writer，把读请求分配给可用 reader。

![Neo4j Enterprise follower and writer routing](/images/blog/neo4j-graph-database-cypher-storage-engine/neo4j-doc-fig06-enterprise-follower-writer-routing.webp)

*Enterprise 概念图中的客户端、writer 与 follower。它说明路由角色，不证明任意延迟或故障切换时间。Source: Neo4j Operations Manual 2026.06, CC BY 4.0.*

集群不等于所有查询自动水平分片。一套常规 cluster 主要提供副本、故障切换和读扩展；单个图是否跨 shard、事务能否跨 database、分析查询如何并行，是不同产品/版本能力。2025.12 后文档出现 Infinigraph 和 sharded property database，更不能倒写进 Community 2026.06 源码。

读副本还引入 bookmark/causal consistency 问题。客户端写入后若立刻从落后 follower 读取，可能看不到刚提交变化。Driver bookmark 用于表达最低因果位置，但额外等待会影响延迟。应用必须决定哪些读需要 read-your-writes，哪些允许 eventual freshness。

官方图只证明架构角色，不证明故障恢复时间、写吞吐或网络分区行为。生产设计仍需验证副本数、跨可用区延迟、quorum、路由刷新、备份、证书、滚动升级和灾难恢复。本文没有 Enterprise 许可和集群实验，因此不报告性能结论。

## 39. 源码模块地图：从入口读，而不是从根目录顺序读

Neo4j 是大型 Maven 多模块项目。按业务链路阅读比从 `community/` 字母顺序翻文件有效：

| 层 | 主要模块 | 推荐入口 |
| --- | --- | --- |
| 协议 | `community/bolt` | `BoltServer`, protocol/FSM/transport |
| Cypher 前端 | `community/cypher/front-end` | parser、AST、semantic、rewriter |
| 规划协调 | `community/cypher/cypher` | `ExecutionEngine`, `CypherPlanner`, `QueryCache` |
| Community runtime | `slotted-runtime`, `interpreted-runtime`, `runtime-util` | slot、pipe mapper、query context |
| 公共图 API | `graphdb-api`, `kernel-api` | Transaction、Node、KernelTransaction、StorageEngine |
| 内核事务 | `community/kernel` | `GraphDatabaseFacade`, `KernelTransactionImplementation` |
| 存储 | `record-storage-engine`, `storage-engine-util` | `RecordStorageEngine`, stores、commands |
| I/O 与日志 | `community/io`, kernel transaction log | `MuninnPageCache`, `TransactionLogWriter` |
| 索引 | `community/index`, `lucene-index` | `GBPTree`, fulltext/vector providers |
| 服务与扩展 | `server`, `procedure`, `dbms` | connectors、procedure registry、database lifecycle |
| 工具 | `import-tool`, `consistency-check`, `cypher-shell` | offline import/check、CLI |

第一轮从请求主链读：`BoltServer -> ExecutionEngine -> planner/runtime -> KernelTransaction -> StorageEngine`。第二轮从提交主链读：transaction state -> commands -> log -> store apply -> checkpoint/recovery。第三轮再按一个功能纵切，例如 vector index 从 Cypher DDL、schema provider、Lucene reader 一直追到 procedure 和 tests。

测试是不可忽略的设计文档。类名告诉你“可能做什么”，integration test 才告诉你版本兼容、失败边界和预期输出。建议同时阅读 `community-it`、Bolt IT、Cypher acceptance tests、storage/recovery tests 和 index provider tests。

源码链接必须固定 commit。链接 `main` 或默认 `2026.06` branch 会随补丁变化，几个月后文章中的类和行为可能不再对应。本文所有核心链接固定到 `eccd584a...`，当前文档则明确标注为写作时的 2026.06 快照。

## 40. 构建系统审计：README 不是最终真相

根 `pom.xml` 不是简单的依赖列表，它定义 Java/Scala 编译、测试插件、license 检查、format、模块 reactor、平台依赖和 enforcer。当前可确认的关键值为：

```text
project.version      = 2026.06.0
vm.target.version    = 21
required.maven       = 3.9.11
scala.version        = 2.13.17
lucene.version       = 10.4.0
netty.version        = 4.2.15.Final
```

README 仍写 Maven 3.8.2，并建议给 Maven 至少 2 GiB heap、把 Linux open-files limit 提到约 40K。前者与 enforcer 不一致，应以 POM 为准；后两项说明完整构建本身是重型工作负载。

本文不执行完整 Maven 构建有三个原因。第一，目标是静态系统阅读，不是发布 Neo4j 二进制。第二，全 reactor 测试需要大量时间、磁盘、内存和系统资源。第三，用户计划明确排除运行数据库与基准。未构建必须写进复现状态，而不能用“源码已读”暗示编译通过。

如果实际复现，应从官方推荐 JDK、Maven wrapper/要求和单模块测试开始，固定 OS 与 native dependencies，再逐步扩大。直接在开发机执行 `mvn clean install` 既可能耗时，也会因文件描述符和内存限制产生与代码无关的失败。

## 41. 最小案例：把电影、用户和推荐路径连成一条执行链

下面用一个小型电影/商品混合图演示建模、索引、路径、聚合和查询计划。它是教学数据，不是本文运行过的 benchmark。

### 41.1 先用约束确定实体身份

```cypher
CREATE CONSTRAINT user_id_unique IF NOT EXISTS
FOR (u:User) REQUIRE u.id IS UNIQUE;

CREATE CONSTRAINT movie_id_unique IF NOT EXISTS
FOR (m:Movie) REQUIRE m.id IS UNIQUE;
```

在 Community 中，唯一性约束允许没有 `id` 的实体存在，因此应用仍应确保创建路径总是写入 ID。若需要强制存在和类型，必须评估 Enterprise constraint 或应用验证，不能只看这两条 DDL。

### 41.2 写入节点和关系

```cypher
MERGE (u:User {id: $userId})
  ON CREATE SET u.createdAt = datetime()

MERGE (m:Movie {id: $movieId})
SET m.title = $title,
    m.year = $year,
    m.embedding = $embedding

MERGE (u)-[r:RATED]->(m)
SET r.score = $score,
    r.updatedAt = datetime()
```

`MERGE` 的 identity pattern 必须与唯一约束对齐。若把经常变化的 `title` 也放进 `MERGE (m:Movie {id:$id,title:$title})`，标题变化可能尝试创建第二个节点并撞约束。正确做法是用稳定 ID 匹配，再 `SET` 可变属性。

关系是否 `MERGE` 取决于事件语义。用户只能有一个当前评分时，`MERGE (u)-[r:RATED]->(m)` 合理；若要保留每次评分事件，应 `CREATE` 带 event ID/timestamp 的关系或事件节点，不能让 `MERGE` 覆盖历史。

### 41.3 参数化图查询

```cypher
MATCH (u:User {id: $userId})-[r:RATED]->(m:Movie)
WHERE r.score >= $minScore
RETURN m.id, m.title, r.score
ORDER BY r.score DESC
LIMIT $limit
```

理想计划以 `NodeUniqueIndexSeek` 或等价唯一索引算子定位一个 User，再做 `Expand(All)` 找 `RATED`，过滤 score，排序并 limit。若计划从所有 Movie 扫起，应该检查约束是否 ONLINE、参数类型是否匹配和统计是否新鲜。

### 41.4 多跳候选与基数控制

```cypher
MATCH (u:User {id: $userId})-[:FOLLOWS]->(friend:User)
MATCH (friend)-[r:RATED]->(candidate:Movie)
WHERE r.score >= 4
  AND NOT EXISTS {
    MATCH (u)-[:RATED]->(candidate)
  }
RETURN candidate,
       count(DISTINCT friend) AS supportingFriends,
       avg(r.score) AS meanScore
ORDER BY supportingFriends DESC, meanScore DESC
LIMIT 20
```

这条查询表达“朋友喜欢但自己未评分”的候选。性能取决于用户关注人数、朋友评分度数和 anti-existence 子查询。若一个用户关注百万账户，中间行会爆炸；若先聚合每个 candidate，可以减少后续结果，但 `NOT EXISTS` 的位置和索引仍需 `PROFILE` 验证。

### 41.5 可变长路径必须有业务上限

```cypher
MATCH p = (a:User {id:$from})-[:FOLLOWS*1..4]->(b:User {id:$to})
RETURN length(p), [n IN nodes(p) | n.id]
ORDER BY length(p)
LIMIT 10
```

`1..4` 不是保守装饰，而是复杂度合同。无上界 path 在有环和高分支图中可能产生巨大搜索空间。若需求只是最短路径，应使用对应 shortest-path 语义，而不是枚举所有路径再 `ORDER BY length`。

### 41.6 Vector 候选再做图约束

概念上可以先从 movie vector index 找近邻，再只保留当前用户可见、未看过且属于允许目录的电影。2026.06 推荐 `SEARCH`，但语法和过滤能力应按目标 server 小版本测试。关键数据流是：

```text
query embedding
  -> vector ANN candidates
  -> tenant/catalog graph constraints
  -> exclude watched/rated relationships
  -> business score or reranker
  -> limited result
```

向量索引中的近邻 score 与图上的朋友支持数、时间衰减和业务质量不是同一尺度。简单相加前要做归一化或学习排序；更不能把低 score 当成“不允许访问”的授权信号。

## 42. 事务案例：重试、幂等和外部副作用

假设业务要创建订单节点、扣减配额并调用支付服务。错误顺序是：先调用支付，再尝试 Neo4j 事务；如果数据库 deadlock 回滚，支付已成功，系统状态分裂。另一个错误是数据库提交后同步调用支付，进程在两者之间崩溃，订单永远未支付。

更稳健的模式是在同一数据库事务中写订单和 outbox 事件：

```cypher
MATCH (u:User {id:$userId})
WHERE u.quota >= $amount
SET u.quota = u.quota - $amount
CREATE (o:Order {
  id:$orderId,
  amount:$amount,
  status:'PENDING',
  createdAt:datetime()
})
CREATE (u)-[:PLACED]->(o)
CREATE (:OutboxEvent {
  id:$eventId,
  kind:'ORDER_CREATED',
  aggregateId:$orderId,
  createdAt:datetime()
})
```

后台 worker 幂等消费 outbox，再更新订单状态。唯一约束保护 `orderId` 和 `eventId`；驱动遇到 transient failure 可以重试整个事务函数。这里 Neo4j 的 ACID 只覆盖数据库内部，outbox 模式负责跨系统一致性。

重试还有两个陷阱。第一，事务函数里不能生成每次不同的业务 ID，否则重试会创建多个逻辑订单；ID 应在重试闭包外生成。第二，procedure、日志或远程调用等外部副作用不能假设会随数据库回滚。

对于“写是否成功未知”的网络中断，客户端应使用业务 ID 查询最终状态，而不是盲目再创建。Bolt failure、driver retry 与业务幂等是三层责任，任何一层都不能替代另外两层。

## 43. 查询性能神话逐条拆解

### 43.1 “图遍历永远是 O(1)”

单次从记录到邻接结构的定位可以近似常数，但遍历要处理的关系数至少与输出/候选规模相关。深度 $d$、分支因子 $b$ 的无过滤树形搜索约为 $O(b^d)$。存储减少 JOIN 重建，不消灭组合爆炸。

### 43.2 “Cypher 比 SQL 慢，因为是高级语言”

声明式语言本身不决定速度。成本 planner 可以重排、下推和选择索引；SQL 也依赖 planner。真正差异来自数据模型、物理存储、统计质量、runtime 和工作负载。固定多表聚合可能适合关系数据库，动态多跳可能适合属性图。

### 43.3 “加内存就能修复慢查询”

更多 page cache 可以降低 miss，却不能修复笛卡尔积、错误 join order、返回百万行、无界 path 或锁冲突。先读 `PROFILE`，再区分访问模式和资源不足。

### 43.4 “索引越多越快”

索引增加 planner 选择和读入口，也增加写放大、population、磁盘和备份成本。低选择率属性、极少查询属性或高频变化 embedding 可能让维护成本大于收益。

### 43.5 “把结果 LIMIT 10 就只处理 10 行”

若 LIMIT 可以下推，执行可能提前停止；若前面有全局排序、聚合、distinct 或必须证明最优路径，仍可能处理全部候选。应看 plan 中 LIMIT 与 pipeline breaker 的位置。

### 43.6 “平均延迟足够说明性能”

图工作负载常有长尾：少量超级节点、深路径、冷 page fault、checkpoint、GC 和锁等待会拉高 P99。容量规划要按 query shape 分类，并单独监控读写、短路径、全文、向量和批量任务。

## 44. 与关系数据库、RDF 和文档数据库的技术对照

| 维度 | Neo4j 属性图 | 关系数据库 | RDF 三元组库 | 文档数据库 |
| --- | --- | --- | --- | --- |
| 主要结构 | 节点、关系、属性 | 表、行、键 | subject-predicate-object | JSON/BSON 文档 |
| 查询接口 | Cypher pattern | SQL relational algebra | SPARQL graph pattern | 文档查询/聚合管线 |
| 关系身份 | 关系可有 ID/属性 | 常用 join table/foreign key | predicate/triple | 引用或嵌套 |
| 多跳表达 | 直接 pattern/path | 多次 JOIN/递归 CTE | property path | 常需应用或聚合操作 |
| schema | 灵活 + index/constraints | 通常显式 schema | ontology/shape 可分离 | 灵活 schema |
| 推理 | 非默认 ontology reasoner | 非默认 | 可结合 RDFS/OWL | 非默认 |
| 强项 | 动态关系遍历、路径 | 固定关系、聚合、生态 | 语义互操作与推理 | 聚合根读写与水平扩展 |

这张表不是排名。许多系统会组合：交易事实留在关系数据库，关系投影同步到 Neo4j 做调查；文档正文放对象存储/搜索引擎，Neo4j 保存实体与来源关系；RDF ontology 负责标准语义，属性图服务在线路径查询。

系统组合会带来复制一致性。若 Neo4j 是 read model，需要定义 CDC offset、幂等 upsert、删除传播、重放和延迟指标；若 Neo4j 是 source of truth，则其他系统要订阅其变更。不能既把每个系统说成主库，又没有冲突解决规则。

## 45. Neo4j 与知识图谱：存储图不等于知识成立

知识图谱工程至少还需要实体身份、ontology/schema、来源、时间、置信、对齐、质量规则和治理。Neo4j 能表达这些结构，却不负责自动把原始文本变成正确知识。

一个可靠事实模型可能是：

```text
(subject:Entity)-[claim:CLAIMS]->(object:Entity)
claim.sourceEpisodeId
claim.validFrom
claim.validTo
claim.confidence
claim.extractorVersion
```

也可能把 Claim 建成节点，以容纳多参与者和多个来源。选择边还是节点取决于事实是否需要独立身份、生命周期和关系。Neo4j 不会替团队自动决定。

图上的 label 也不是 ontology class 的完整语义。`:Employee` 和 `:Person` 是否有继承、`:WORKS_AT` 的 domain/range、关系是否对称或传递，都要由应用、constraint、procedure 或外部推理层定义。

因此，Neo4j 是知识图谱的常见工程底座，而不是知识图谱方法论本身。把 LLM 抽出的三元组直接 `MERGE` 进去，只会快速构建一个可能重复、矛盾、无来源的图。

## 46. 与 Graphiti / Zep 和 OpenSPG KAG 的关系

本站此前精读的 Graphiti 和 OpenSPG KAG 都可能使用图数据库，但它们解决的是 Neo4j 之上的问题。

**Graphiti / Zep** 聚焦 Agent memory。它从 episode 中抽实体与事实，维护双时态有效期、来源、社区和混合检索。Neo4j/FalkorDB/Neptune 是可替换 graph driver。即使底层用 Neo4j，实体消歧、事实失效和 context constructor 仍由 Graphiti 负责。

**OpenSPG KAG** 聚焦专业知识服务。它定义 LLM-friendly schema、知识/原文互索引、语义对齐、逻辑形式规划和混合推理。底层 OpenSPG 负责图存储与服务，KAG Builder/Solver 负责知识构建和问题求解。

三者可以按层次理解：

```text
Application / Agent
  -> KAG logical solver or Graphiti memory orchestration
  -> domain schema, temporal facts, retrieval policies
  -> Cypher / graph API
  -> Neo4j query, transaction and storage engine
```

若 Neo4j 查询很快，但上层实体合并错了，答案仍然错；若上层知识正确，但 Cypher 产生候选爆炸，系统仍然不可服务。源码精读的价值是明确每层可观测、可测试和可替换的责任。

## 47. 生产落地路线：从模型到数据库，而不是反过来

### 47.1 先收集查询，再设计图

不要先画“所有实体关系大图”，再希望它支持任何问题。应列出核心 query shapes：锚点是什么、走哪些关系、最大深度、过滤属性、返回规模、更新频率和一致性需求。图 schema 为查询服务，同时保留业务可理解性。

### 47.2 定义稳定身份与来源

每类实体需要稳定业务 ID、来源系统和合并规则。姓名、标题、URL 都可能变化。跨源实体合并应保留 aliases 和 provenance，而不是不可逆覆盖。

### 47.3 把关系方向和粒度写成契约

`:EMPLOYED_BY` 与 `:EMPLOYS` 选择一个主方向；需要反向查询时 Cypher 可以反向匹配。关系是当前状态还是事件历史要提前决定。类型命名、时间字段和删除语义应进入 schema 文档与测试。

### 47.4 先建立最小约束

对业务 ID 创建唯一约束，检查历史冲突，再导入数据。根据真实 `WHERE` 和排序创建 index。全文和向量索引作为候选层，不承担唯一性和授权。

### 47.5 用真实分布压测 query shape

平均度数无法代表超级节点。压测数据要保留度数长尾、属性相关性、时间分布和冷热访问。分别测试冷缓存、预热后稳态、并发写、checkpoint 和恢复。

### 47.6 把迁移和回滚当成功能

版本升级、store migration、index rebuild、procedure compatibility 和 driver protocol 都要演练。备份只有恢复成功才有价值；回滚只有在 store format 仍兼容时才成立。

## 48. 监控清单：分层定位，而不是只看数据库 CPU

| 层 | 建议观测 |
| --- | --- |
| Client/Driver | pool usage、acquire wait、retry、timeout、bookmark wait |
| Bolt | active connections、handshake/auth failure、stream backpressure |
| Query | fingerprint、P50/P95/P99、planning/execution、rows、db hits |
| Runtime | memory、spill/eager、estimated vs actual rows、operator hotspots |
| Transaction | active/long-running、lock wait、deadlock、termination、commit rate |
| Page cache | hit/miss、fault、eviction、flush、working-set change |
| WAL/Checkpoint | append/force latency、rotation、checkpoint duration、log volume |
| Index | population state、size、read count、full-text lag、vector recall proxy |
| JVM/Host | heap、GC、native memory、CPU、file descriptors、disk latency/capacity |

慢查询日志应记录参数还是只记录结构，需要在可诊断性与隐私之间权衡。完整 embedding、用户查询和 PII 不应无期限进入日志；至少要有 hash/fingerprint、受控采样和权限。

告警必须关联业务影响。Page cache miss 增加但延迟稳定，可能是后台扫描；CPU 高但吞吐同步增长，可能是健康负载；最危险的是 queue/lock wait、P99、错误率和磁盘空间共同恶化。单指标阈值容易误报。

恢复指标同样重要：最近成功备份时间、最近恢复演练、checkpoint age、日志保留窗口、复制 lag。数据库“今天能查”不证明明天故障后能恢复。

## 49. 安全与治理检查清单

1. Bolt/HTTP 是否只在必要网络监听，advertised address 是否正确？
2. 是否启用 TLS，证书轮换是否演练？
3. 默认账号是否已更名/改密，凭据是否进入 secret manager？
4. Community 缺少细粒度 RBAC 时，是否有可信 API 层强制授权？
5. 动态 label/type/property 是否白名单化，还是字符串拼接？
6. Procedure 和插件是否固定版本、最小权限并经过安全审查？
7. Query log、transaction log、backup 中的 PII 保留多久？
8. 删除请求是否覆盖节点、关系、索引、备份和下游投影？
9. 全文/向量召回是否在返回前执行租户与权限过滤？
10. 数据导入、CDC 和管理员工具是否有独立审计身份？

“图上有权限关系”不等于数据库访问已经授权。OpenFGA、Cedar 或应用权限服务可以把关系授权建模在独立系统；Neo4j 也可以保存授权图，但最终 enforcement 必须在不可绕过的服务端路径发生。让任意客户端直接执行 Cypher，再要求每条查询自己记得权限 predicate，是脆弱设计。

## 50. 证据强度与本文没有证明的事

Cypher SIGMOD 2018 是语言设计与形式语义论文，不是 Neo4j 内核 benchmark。它证明核心读取语言可以被精确定义，并讨论历史与工业采用；它没有比较 page cache、store format、事务吞吐或当前 runtime。

Neo4j 官方文档说明受支持功能与配置，但架构图不是性能证据。Enterprise cluster 图不会给出故障切换时间；vector provider 表不会给出你数据上的 recall；“推荐 block format”也不等于 Community aligned 在任何工作负载都慢。

Community 源码提供机制证据：我们能看到 Bolt server、planner、slotted runtime、kernel transaction、record storage、page cache、transaction log、GB+Tree 和 Lucene provider。静态阅读不能证明代码在本文机器上构建通过，也不能证明生产配置正确。

本文尤其没有证明：

- Neo4j 比某个关系、文档、RDF 或向量数据库普遍更快；
- 某类图遍历具有端到端常数复杂度；
- Enterprise block store 和集群可以从 Community 源码复现；
- 2026.06 preview vector 配置会长期保持兼容；
- 任何图模型都会自动产生高质量知识图谱；
- 官方默认值适合目标数据规模与 SLA。

这些空白不是文章缺陷的掩饰，而是正确的证据边界。系统选型最终要回到可代表生产的数据、查询、硬件、并发和故障演练。

## 51. 局限性与批判

### 51.1 模型表达简单，治理复杂

节点和边很容易上手，却容易产生 label/type 失控、重复实体、关系方向不一致和事件/状态混用。图越容易写，越需要 schema review、约束和 provenance。

### 51.2 成本规划受统计近似限制

图关系具有强相关和长尾，平均统计很难捕获所有 pattern。估算错误会选错起点和连接顺序；hint 能短期修复，也会增加维护负担。

### 51.3 超级节点和路径爆炸是结构性问题

更大 page cache 或更快 SSD 只能缓解访问成本，不能改变候选组合数。业务上限、模型重构和分析/事务分离往往比硬件升级更重要。

### 51.4 Community 与 Enterprise 差距影响可复现性

官方最佳实践常围绕 block store、cluster、online backup、RBAC 和 pipelined runtime，但它们不在开源仓库。只读 Community 源码无法审计整个商业产品的数据路径。

### 51.5 大型 JVM/Scala/Java 代码库阅读成本高

模块多、生成代码、兼容层和内部 API 演进使源码导航困难。类名可能保留历史术语，例如 retired interpreted runtime 仍有内部模块。必须以测试和调用链校验，不能按目录名下结论。

### 51.6 新向量能力变化快

Provider、Cypher 语法、native VECTOR、量化和过滤在 CalVer 版本持续演进。把 Neo4j 当向量数据库使用时，升级测试和 recall/latency 评估不可省略。

### 51.7 图数据库不是语义真值机

Cypher 能精确执行写入的事实，不判断事实是否真实。LLM 抽取、实体对齐、时间解析和来源可信度需要上层系统治理。

## 52. 推荐阅读路径

如果只有两小时，建议按这个顺序：

1. 本文第 5-9 节和 Cypher 论文 Fig. 1、2、6、7，建立语义基础；
2. 官方 query lifecycle、`EXPLAIN/PROFILE` 和 runtime concepts，理解查询编译执行；
3. `ExecutionEngine.scala`、`CypherPlanner.scala`、`QueryCache.scala`；
4. `KernelTransactionImplementation.java` 与 `StorageEngine.java`；
5. `RecordStorageEngine.java`、`MuninnPageCache.java`、`TransactionLogWriter.java`；
6. `GBPTree.java`、full-text/vector provider；
7. Operations Manual 的 store formats、transaction logs、backup 和 security 边界。

如果负责生产运维，再补：driver manual、configuration reference、memory configuration、transaction management、upgrade/migration、consistency check、metrics 和目标 edition 的集群/备份文档。

如果负责知识图谱或 Agent memory，再补本站 Graphiti/Zep、OpenSPG KAG、OpenFGA 与推荐系统报告，把数据库、知识构建、检索推理和授权分层理解。

## 53. 结论：Neo4j 的核心是“图语义 + 数据库纪律”

回到开头的问题：Neo4j 为什么不只是把边存起来？因为真正可用的图数据库必须同时完成三类工作。

语言层把属性图 pattern、变量绑定、bag、null、聚合和路径语义定义清楚，使用户可以声明结果而不是编排磁盘访问。Cypher 论文提供了这一层的形式锚点。

执行层把 query string 编译成逻辑计划和物理运行时，利用 schema、统计、缓存和索引控制候选基数；Bolt、backpressure、内存与事务上下文让查询成为可服务请求，而不是离线脚本。

存储层把节点、关系和属性变成 record、page 和 command，用 page cache 提供局部访问，用 WAL、checkpoint 和 recovery 保证提交可恢复，用 GB+Tree 和 Lucene 支持锚定、全文和向量候选。

这三层共同构成 Neo4j 的价值，也共同构成复杂度。图模型错误会让查询语义偏离业务，统计错误会让计划失效，事务和存储配置错误会让数据在故障时不可恢复。Enterprise 与 Aura 能承担更多可用性和治理职责，却不能替团队决定实体身份、路径上限、授权边界和业务一致性。

因此，最可靠的采用方式不是先相信“图比表快”，而是先列出真实 query shapes，建立稳定 ID 和最小约束，用 `PROFILE` 和生产分布验证计划，再把事务、备份、升级、安全和故障恢复纳入同一设计。Neo4j 的长期价值，正在于它把图关系变成数据库的一等执行对象；它的长期风险，也在于团队可能只看到图形直觉，而忽略数据库纪律。

## 参考资料

### Neo4j 源码与版本

1. [neo4j/neo4j GitHub repository](https://github.com/neo4j/neo4j)
2. [Neo4j `2026.06.0` tag](https://github.com/neo4j/neo4j/tree/2026.06.0)
3. [Pinned source commit `eccd584a...`](https://github.com/neo4j/neo4j/tree/eccd584a64d468af3daeab421478fe78567c518f)
4. [Root `pom.xml`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/pom.xml)
5. [Repository README and license boundary](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/README.asciidoc)

### 官方文档

6. [Neo4j Documentation](https://neo4j.com/docs/)
7. [Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
8. [Cypher query lifecycle and runtimes](https://neo4j.com/docs/cypher-manual/current/planning-and-tuning/runtimes/concepts/)
9. [Understanding query plans](https://neo4j.com/docs/cypher-manual/current/planning-and-tuning/execution-plans/)
10. [Indexes for search performance](https://neo4j.com/docs/cypher-manual/current/indexes/search-performance-indexes/overview/)
11. [Full-text indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/full-text-indexes/)
12. [Vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)
13. [Create constraints](https://neo4j.com/docs/cypher-manual/current/schema/constraints/create-constraints/)
14. [Operations Manual](https://neo4j.com/docs/operations-manual/current/)
15. [Store formats](https://neo4j.com/docs/operations-manual/current/database-internals/store-formats/)
16. [Transaction management](https://neo4j.com/docs/operations-manual/current/database-internals/transaction-management/)
17. [Authentication and authorization](https://neo4j.com/docs/operations-manual/current/authentication-authorization/)

### 论文与语言语义

18. Nadime Francis et al. [Cypher: An Evolving Query Language for Property Graphs](https://homepages.inf.ed.ac.uk/pguaglia/papers/sigmod18.pdf). SIGMOD 2018. DOI: [10.1145/3183713.3190657](https://doi.org/10.1145/3183713.3190657).
19. Alastair Green et al. [Formal Semantics of the Language Cypher](https://arxiv.org/abs/1802.09984). arXiv:1802.09984.
20. openCypher. [openCypher resources](https://opencypher.org/).

### 源码阅读入口

21. [`BoltServer.java`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/bolt/src/main/java/org/neo4j/bolt/BoltServer.java)
22. [`ExecutionEngine.scala`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/cypher/cypher/src/main/scala/org/neo4j/cypher/internal/ExecutionEngine.scala)
23. [`CypherPlanner.scala`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/cypher/cypher/src/main/scala/org/neo4j/cypher/internal/planning/CypherPlanner.scala)
24. [`QueryCache.scala`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/cypher/cypher/src/main/scala/org/neo4j/cypher/internal/QueryCache.scala)
25. [`KernelTransactionImplementation.java`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/kernel/src/main/java/org/neo4j/kernel/impl/api/KernelTransactionImplementation.java)
26. [`StorageEngine.java`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/kernel-api/src/main/java/org/neo4j/storageengine/api/StorageEngine.java)
27. [`RecordStorageEngine.java`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/record-storage-engine/src/main/java/org/neo4j/internal/recordstorage/RecordStorageEngine.java)
28. [`MuninnPageCache.java`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/io/src/main/java/org/neo4j/io/pagecache/impl/muninn/MuninnPageCache.java)
29. [`TransactionLogWriter.java`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/kernel/src/main/java/org/neo4j/kernel/impl/transaction/log/TransactionLogWriter.java)
30. [`GBPTree.java`](https://github.com/neo4j/neo4j/blob/eccd584a64d468af3daeab421478fe78567c518f/community/index/src/main/java/org/neo4j/index/internal/gbptree/GBPTree.java)

---

**复现状态声明**：本文完成的是 Neo4j Community `2026.06.0`、官方 2026.06 文档与 Cypher 论文的静态交叉阅读。没有完整构建源码、启动 Neo4j、下载 Enterprise、连接 Aura、运行查询或复现任何性能结果。文中的 Cypher 和配置用于解释接口，实际采用前应在目标版本、edition、数据分布与硬件上验证。

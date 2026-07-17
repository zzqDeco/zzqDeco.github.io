---
title: "OpenFGA 精读：Zanzibar 关系授权模型、ReBAC 建模语言与生产级细粒度权限系统"
description: "从 Zanzibar 论文与 OpenFGA 官方实现双线精读关系元组、授权模型、Check/Expand/ListObjects、条件元组、CLI 测试、存储部署与工程落地边界"
pubDate: "2026-07-08T14:53:07+08:00"
updatedDate: "2026-07-08T14:53:07+08:00"
tags:
  - "OpenFGA"
  - "Authorization"
  - "Zanzibar"
  - "ReBAC"
  - "Security"
  - "System Design"
  - "Code Reading"
draft: false
---

OpenFGA 适合被误解，也适合被低估。

很多团队第一次看到它，会把它理解成“一个权限判断服务”：给它一个 `user`、一个 `relation`、一个 `object`，它返回 `allowed: true/false`。这个理解没有错，但太窄了。OpenFGA 真正想解决的问题，是把复杂应用里的授权关系从业务代码、SQL join、散落的角色枚举和接口 if/else 里抽离出来，变成一套可建模、可测试、可演进、可审计的关系授权系统。

更准确地说，OpenFGA 是一个受 Google Zanzibar 启发的开源细粒度授权引擎。Zanzibar 是 Google 在 2019 年 USENIX ATC 公开的全球授权系统论文，支撑了 Google 内部大量产品的 ACL 和协作权限。OpenFGA 不是 Google Zanzibar 的开源版本，也不能直接继承论文里的 Spanner 架构、Google 内部规模和可用性数字。它的价值在于把 Zanzibar 风格的核心模型用开源、可运行、面向开发者的方式重新实现出来：关系元组、授权模型 DSL、Check / Expand / ListObjects / ListUsers API、条件与上下文、CLI 测试、存储后端和服务化部署。

本文采用两条线精读：

1. **论文线**：精读 Zanzibar 的 tuple 模型、namespace 配置、一致性语义、API、架构和性能指标。
2. **工程线**：精读 OpenFGA 当前官方文档、OpenFGA server 仓库和 CLI 仓库，解释这些概念如何落到实际建模、测试、查询、存储和生产接入。

实施时核对的版本边界如下：`openfga/openfga` 最新 release 为 `v1.18.1`，发布时间是 `2026-06-29`；`openfga/cli` 最新 release 为 `v0.7.17`，发布时间是 `2026-06-24`。两者均为 Apache-2.0 许可证。版本会继续变化，本文的具体路径和命令以这个时间点的官方资料为准。

## 1. 一句话贡献

OpenFGA 的核心贡献不是提供一个“鉴权微服务模板”，而是把授权问题拆成三层：

| 层次 | 解决的问题 | 典型产物 |
| --- | --- | --- |
| 授权模型 | 你的业务里有哪些对象、关系和继承规则 | `model schema 1.1`、`type document`、`define viewer` |
| 关系数据 | 某个用户或用户集合和某个对象之间存在什么关系 | `document:roadmap#viewer@user:anne` |
| 查询执行 | 给定模型和关系数据，回答权限判断或枚举问题 | `Check`、`Expand`、`ListObjects`、`ListUsers` |

这三层分开以后，业务代码不再需要到处写“如果是 owner 或 admin 或父文件夹 viewer 或团队 member 就允许访问”的分支。业务服务把事实写成 tuple，把规则写成 model，然后调用 OpenFGA 查询。

这听起来简单，但复杂点在于：授权规则很容易出现继承、组合、例外、临时上下文和多租户生命周期。OpenFGA 的设计让这些复杂性显式化。显式化之后，才有可能测试、迁移、审计和灰度。

## 2. 论文与项目边界

Zanzibar 论文的题名是 **Zanzibar: Google's Consistent, Global Authorization System**，作者来自 Google，发表于 USENIX ATC 2019。论文报告的是 Google 内部生产系统：它使用 Spanner 存储关系元组，提供一致性 token，也就是论文里的 zookie，并支撑了非常高的请求量、可用性和规模。

OpenFGA 的定位不同。官方 README 把它描述为一个高性能、灵活的授权/权限引擎，受 Google Zanzibar 启发。它提供 HTTP 和 gRPC API，支持多种存储后端，包括 in-memory、PostgreSQL、MySQL 和 SQLite beta；还提供 CLI、Playground、SDK 和 Terraform provider 等生态工具。

这一区分非常重要：

| 项目 | 能引用的结论 | 不能外推的结论 |
| --- | --- | --- |
| Zanzibar 论文 | 关系授权模型、namespace 配置、一致性问题、分布式授权系统设计 | OpenFGA 一定具备 Google 内部规模、Spanner 语义和同等 SLO |
| OpenFGA | 开源实现、当前 DSL/API/CLI、可部署存储后端、工程接入流程 | 它是 Google Zanzibar 原代码，或自动解决所有业务权限治理问题 |

本文会多次强调这个边界。尤其是论文中的性能图、可用性图和 Spanner 架构，只能作为设计启发，不能写进 OpenFGA 的 SLA。

## 3. 授权问题背景

先把几个常被混在一起的词分开：

| 概念 | 回答的问题 | 例子 |
| --- | --- | --- |
| Authentication | 你是谁 | 登录、OIDC、JWT、session |
| Authorization | 你能不能做这件事 | `anne` 能不能读 `document:roadmap` |
| RBAC | 你有什么角色 | `admin`、`editor`、`viewer` |
| ABAC | 你和资源有什么属性 | 地区、部门、时间、套餐、IP |
| ReBAC | 你和资源之间有什么关系 | owner、member、parent folder viewer、team member |

很多早期系统用 RBAC 就够了。例如后台管理系统只有几个角色：超级管理员、运营、财务、客服。问题出现在协作型、多租户、资源层级复杂的产品里：

- 一个用户是组织成员，但只对某些项目有编辑权限。
- 文件继承文件夹权限，但可以单独分享给外部用户。
- 团队成员可以读团队仓库，但敏感分支需要额外批准。
- 用户有 `viewer` 权限，但如果被 block 或合同过期，需要排除。
- 一次请求中临时带入 IP、时间窗口或审批结果，影响是否允许访问。

这些规则如果全写在业务代码里，通常会形成三类债务：

1. **规则不可见**：权限逻辑散落在服务、SQL、前端和后台脚本中。
2. **规则不可测**：很难用固定输入验证“某个用户是否应该能访问某类资源”。
3. **规则不可迁移**：一旦角色或组织层级调整，历史数据和接口行为一起变成风险。

OpenFGA 的建模方式就是为了解决这类问题。

## 4. Zanzibar 的关系元组模型

Zanzibar 的基本数据结构是 relation tuple。论文使用的文本形式可以写成：

$$
\langle object \rangle \# \langle relation \rangle @ \langle user \rangle
$$

其中 `object` 通常由 namespace 和 object id 组成，`relation` 是对象上定义的关系，`user` 可以是真实用户，也可以是一个 userset。

例如：

```text
doc:readme#owner@10
group:eng#member@11
doc:readme#viewer@group:eng#member
doc:readme#parent@folder:A
```

这个模型的关键是：`user` 不一定是单个用户。它可以是 `group:eng#member` 这样的用户集合。于是“eng 组成员可以看 readme 文档”不需要展开成 N 条用户 tuple，而是可以用一条对象到用户集合的关系表达。

![Example relation tuples](/images/blog/openfga-zanzibar-fine-grained-authorization/zanzibar-table01-relation-tuples.webp)

*Source: Pang et al., USENIX ATC 2019, Table 1.*

Table 1 展示了 Zanzibar tuple 的三个重要能力：

第一，tuple 可以表示对象和用户的直接关系，例如 `doc:readme#owner@10`。

第二，tuple 可以表示组关系，例如 `group:eng#member@11`。这为组织、团队、域、部门等用户集合提供了基础。

第三，tuple 可以表示对象之间的继承关系，例如 `doc:readme#parent@folder:A`。这让“文档继承文件夹权限”成为模型的一部分，而不是业务代码里的特殊分支。

OpenFGA 继承了这个核心思想。它的关系元组同样由 `user`、`relation`、`object` 组成。只是 OpenFGA 把建模语言、API 和存储后端做成了面向开发者的开源产品。

## 5. Namespace 配置与授权模型

只有 tuple 还不够。系统还需要知道 relation 之间如何推导。

例如，如果一个用户是文档 owner，那么通常也应该是 editor 和 viewer。如果一个用户是父文件夹 viewer，那么可能也应该是子文档 viewer。这类规则不应该靠写入大量冗余 tuple 实现，而应该写进授权模型。

Zanzibar 论文用 namespace configuration 表达这些规则。

![Simple namespace configuration](/images/blog/openfga-zanzibar-fine-grained-authorization/zanzibar-fig01-namespace-config.webp)

*Source: Pang et al., USENIX ATC 2019, Fig. 1.*

这张图里的逻辑可以翻译成更接近 OpenFGA 的表达：

```text
model
  schema 1.1

type user

type folder
  relations
    define viewer: [user]

type document
  relations
    define owner: [user]
    define editor: [user] or owner
    define parent_folder: [folder]
    define viewer: [user] or editor or viewer from parent_folder
```

这里有几种核心构造：

| 构造 | 含义 |
| --- | --- |
| `[user]` | 允许直接写入 `user` 和当前 relation 的 tuple |
| `or owner` | 当前 relation 包含同对象上的另一个 relation |
| `viewer from parent_folder` | 从当前对象关联的 parent folder 继续取 viewer |
| `schema 1.1` | OpenFGA 当前建模语言版本声明 |

OpenFGA 的授权模型就是 Zanzibar namespace configuration 的开发者友好版本。它让权限规则成为一个独立文件，可以被版本化、校验、测试、发布和回滚。

## 6. OpenFGA 核心概念

OpenFGA 文档中最常见的概念有七个。

| 概念 | 解释 |
| --- | --- |
| Store | 一组授权模型和关系元组的隔离空间，通常对应一个应用或租户边界 |
| Authorization Model | 描述类型、关系和推导规则的模型 |
| Type | 一类对象，例如 `user`、`document`、`folder`、`organization` |
| Object | 某个具体对象，例如 `document:roadmap` |
| Relation | 对象上定义的关系，例如 `viewer`、`editor`、`parent` |
| User | 被授权的主体，可以是真实用户，也可以是 userset |
| Relationship Tuple | 存储的关系事实，由 `user`、`relation`、`object` 组成 |

可以把 OpenFGA 的运行过程抽象成：

```text
authorization model + relationship tuples + optional request context
  -> OpenFGA query engine
  -> allowed / userset / object list / user list
```

这里的重点是：OpenFGA 不是读取业务数据库后帮你猜权限。它只处理你写进去的授权事实和你定义好的授权模型。业务系统仍然要负责：

- 在创建组织、项目、文件、成员关系时写入 tuple。
- 在删除或迁移资源时清理 tuple。
- 在请求入口完成身份认证。
- 在服务端调用 OpenFGA 并按结果执行 fail-closed。

## 7. 建模语言精读

OpenFGA 当前建模语言使用 `model schema 1.1`。一个典型模型如下：

```text
model
  schema 1.1

type user

type team
  relations
    define member: [user]

type document
  relations
    define owner: [user]
    define editor: [user, team#member] or owner
    define viewer: [user, user:*, team#member] or editor
```

这个模型有几层含义。

第一，`type user` 声明了用户类型，但没有定义关系。用户通常作为主体使用。

第二，`team#member` 是 userset 类型限制。它表示 `document#editor` 可以直接授予一个团队的 member 集合，而不仅仅是单个 user。

第三，`user:*` 表示 public access。它很强，也很危险。它适合公开文档、公开资源这类明确场景，不应该被当作默认兜底。

第四，`viewer` 通过 `or editor` 继承 editor。也就是说，如果 Check 询问 `user:anne` 是否是 `document:roadmap` 的 viewer，OpenFGA 不只看直接 viewer tuple，也会检查她是否是 editor。

### 7.1 Direct relationship

Direct relationship 表示可以直接写入的关系。例如：

```text
define editor: [user, team#member]
```

这意味着可以写：

```text
document:roadmap#editor@user:anne
document:roadmap#editor@team:platform#member
```

但不能写：

```text
document:roadmap#editor@organization:acme#member
```

除非模型显式允许 `organization#member`。

这个限制非常重要。它让模型成为数据写入的类型约束，而不是只在查询时生效。没有类型限制的授权系统很容易积累脏 tuple，最后每次迁移都变成考古。

### 7.2 Computed userset

Computed userset 是“同一对象上另一个 relation 的用户集合”。例如：

```text
define viewer: [user] or editor
```

如果一个用户是 `document:roadmap#editor`，那么她也是 `document:roadmap#viewer`。这表达的是 relation 之间的包含关系。

### 7.3 Tuple-to-userset

Tuple-to-userset 是 Zanzibar / OpenFGA 建模中最有表达力的部分之一。典型写法是：

```text
type folder
  relations
    define viewer: [user, folder#viewer]

type document
  relations
    define parent_folder: [folder]
    define viewer: [user] or viewer from parent_folder
```

`viewer from parent_folder` 的意思是：先找到当前 document 的 `parent_folder` 关系指向哪个 folder，再检查用户是否在那个 folder 的 viewer 集合里。

这就是资源继承的基础。业务代码不需要自己查父文件夹，再查父文件夹权限，再判断继承；授权模型直接表达了这条路径。

### 7.4 Union / Intersection / Exclusion

OpenFGA 支持组合关系。常见的 union 是 `or`：

```text
define viewer: [user] or editor or viewer from parent
```

Intersection 用 `and` 表示：

```text
define viewer: ([user] or viewer from parent) and member from organization
```

这表示用户必须同时满足两类条件：一方面是直接 viewer 或继承 viewer，另一方面必须是 organization member。

Exclusion 用于“有权限但被排除”的场景。它很有用，但也容易把模型变成难以解释的黑箱。例如：

```text
define viewer: member but not blocked
```

这适合黑名单、禁用用户、文档 deny list。但如果系统大量依赖 exclusion，排查权限问题会变难，因为允许路径和拒绝路径都要解释清楚。

## 8. Tuple 设计与数据建模

OpenFGA 的模型语言回答“怎样推导权限”，tuple 则回答“当前事实是什么”。

例如文档协作系统可以写入：

```text
organization:acme#member@user:anne
folder:product#organization@organization:acme
folder:product#viewer@organization:acme#member
document:roadmap#parent_folder@folder:product
document:roadmap#editor@team:pm#member
team:pm#member@user:ben
```

如果模型定义了：

```text
type document
  relations
    define parent_folder: [folder]
    define editor: [user, team#member]
    define viewer: [user] or editor or viewer from parent_folder
```

那么：

- `user:anne` 可以从 `organization:acme#member` -> `folder:product#viewer` -> `document:roadmap#viewer` 这条路径获得权限。
- `user:ben` 可以从 `team:pm#member` -> `document:roadmap#editor` -> `document:roadmap#viewer` 这条路径获得权限。

这里有一个常见工程误区：把 tuple 当成业务状态数据库。OpenFGA 应该存授权关系事实，而不是所有业务事实。资源标题、价格、状态、正文、画像、搜索索引、审计日志都应该在业务系统或其他存储里。OpenFGA 只需要知道“这些事实中哪些会影响授权”。

## 9. Check API 精读

Check 是最核心的查询：给定用户、关系和对象，回答是否允许。

抽象成函数就是：

$$
\text{Check}(u, r, o, M, T, C) \rightarrow \{\text{allowed}: true/false\}
$$

其中：

- $u$ 是 user，例如 `user:anne`。
- $r$ 是 relation，例如 `viewer`。
- $o$ 是 object，例如 `document:roadmap`。
- $M$ 是 authorization model。
- $T$ 是已存储的 relationship tuples。
- $C$ 是请求时传入的 context、contextual tuples 或 condition context。

一个服务端调用大致像这样：

```bash
fga query check \
  --store-id "$FGA_STORE_ID" \
  --model-id "$FGA_MODEL_ID" \
  user:anne viewer document:roadmap
```

如果带 contextual tuple，可以把本次请求的临时事实放进去：

```bash
fga query check \
  --store-id "$FGA_STORE_ID" \
  --model-id "$FGA_MODEL_ID" \
  user:anne can_view transaction:A \
  --contextual-tuple "user:anne user ip-address-range:10.0.0.0/16" \
  --contextual-tuple "user:anne user timeslot:18_19"
```

Check 的工程原则很直接：

1. 权限检查必须在服务端执行，不能只放在前端。
2. OpenFGA 调用失败时默认 fail closed，除非业务明确设计了降级路径。
3. 每个关键接口都应该明确检查哪个 object、哪个 relation。
4. relation 命名要贴近业务动作，例如 `can_view`、`can_edit`、`can_approve`，避免把内部角色名暴露给调用方。

## 10. Expand、ListObjects 和 ListUsers

除了 Check，OpenFGA 还提供几类关系查询。

| API | 回答的问题 | 常见用途 |
| --- | --- | --- |
| Expand | 某个 object relation 最终包含哪些 userset | 调试权限路径、解释模型 |
| ListObjects | 某个 user 对哪些 objects 有某个 relation | 列出用户可访问文档、RAG 过滤候选 |
| ListUsers | 哪些 users 对某个 object 有某个 relation | 分享面板、权限审计、协作成员列表 |

这些 API 很方便，但也更容易被误用。

Check 是点查询，最适合在线请求路径。ListObjects 和 ListUsers 是枚举查询，成本和模型结构、tuple 分布、过滤条件强相关。不要把 OpenFGA 当成全文搜索或通用图数据库。如果你要在 1000 万文档里做关键词检索，再过滤用户权限，更常见的架构是：

```text
search / vector retrieval -> candidate documents -> batch Check / authorized filter -> rank / return
```

或者在某些场景下用 ListObjects 先拿授权对象集合，再交给搜索系统做过滤。哪种顺序更好，取决于候选规模、权限稀疏度和延迟预算。

## 11. Contextual Tuples 与 Conditions

Contextual tuples 是 OpenFGA 的请求级临时 tuple。它们不会被写入持久存储，只在一次 Check、Expand、ListObjects 或 ListUsers 请求中生效。

适合 contextual tuples 的场景包括：

- 本次请求经过了某个临时审批。
- 用户当前 IP 属于某个允许网段。
- 会话里临时选择了某个组织身份。
- 上游服务已经验证了某个外部关系，但不希望持久写入。

Conditions 则用于把结构化上下文纳入授权判断。例如“只允许工作时间访问”“只允许某个 IP 范围访问”“只允许资源处于 active 状态”。它们让 ABAC 风格的判断可以和 ReBAC 关系图结合。

不过这两个能力都要克制使用。经验上可以按下面的边界划分：

| 信息类型 | 建议位置 |
| --- | --- |
| 长期稳定关系，例如 membership、owner、parent | 持久 tuple |
| 请求级临时事实，例如本次审批通过 | contextual tuple |
| 可计算请求属性，例如时间、IP、套餐 | condition context |
| 大规模业务状态，例如订单、余额、全文内容 | 业务系统，不要塞进 OpenFGA |

如果大量权限都依赖复杂 context，模型会变得难测；如果大量临时事实都写成持久 tuple，存储会膨胀，清理也会困难。

## 12. Zanzibar 的一致性机制

Zanzibar 论文最重要的系统贡献之一，是把授权检查和一致性问题绑定在一起讨论。

论文提出了一个经典问题：new enemy problem。假设用户刚刚被移出某个共享文档的访问列表，但系统因为缓存、复制延迟或读取旧快照，仍然允许他访问最新内容，就会造成安全问题。

Zanzibar 用 zookie 表示一致性 token。客户端可以在后续请求中携带 zookie，要求授权检查至少读到某个变更点之后的状态。论文还区分 Safe 和 Recent 请求，用于解释不同新鲜度要求下的延迟和跨区域读取成本。

OpenFGA 也提供一致性相关能力，但不能把它简单等同于 Google Zanzibar 的 Spanner + zookie 语义。正确写法应该是：

- Zanzibar 论文展示了一种强一致授权系统设计。
- OpenFGA 当前 API/CLI 中有 consistency 相关参数和实现细节。
- 具体一致性语义要以 OpenFGA 当前版本文档、部署存储和服务配置为准。

这一区分在技术报告中必须写清，否则很容易把论文系统能力误写成开源项目承诺。

## 13. Zanzibar 架构精读

Zanzibar 的总体架构如下。

![Zanzibar architecture](/images/blog/openfga-zanzibar-fine-grained-authorization/zanzibar-fig02-architecture.webp)

*Source: Pang et al., USENIX ATC 2019, Fig. 2.*

图中最关键的模块有四类：

| 模块 | 作用 |
| --- | --- |
| aclserver | 接收 Check、Read、Expand、Write 等请求，并在集群内 fan out |
| relation tuple database | 存储每个 namespace 的关系元组，论文中基于 Spanner |
| changelog | 支撑 Watch API，向客户端推送 tuple 变更 |
| Leopard | 用于优化大规模集合计算和集合展开的索引系统 |

这张图给 OpenFGA 的启发是：授权服务不是一个简单 REST wrapper。它需要处理递归关系解析、缓存、存储读取、变更传播、枚举查询和延迟预算。

但这张图不能直接当作 OpenFGA 部署图。OpenFGA 的开源部署通常是：

```text
application service
  -> OpenFGA HTTP / gRPC server
  -> datastore: memory / PostgreSQL / MySQL / SQLite beta
```

如果使用 Postgres 或 MySQL，OpenFGA 的高可用性、备份、容量规划和延迟就和你的数据库部署质量相关。OpenFGA server 可以水平扩展，但存储层、网络、缓存和查询模型仍然决定实际表现。

## 14. OpenFGA 运行与存储

OpenFGA Docker 运行时常见端口：

| 端口 | 作用 |
| --- | --- |
| `3000` | Playground |
| `8080` | HTTP API |
| `8081` | gRPC API |

官方文档中 SQLite 方式可以这样运行：

```bash
docker run --rm --network=openfga \
  -v openfga:/home/nonroot \
  -u nonroot \
  openfga/openfga migrate \
  --datastore-engine sqlite \
  --datastore-uri 'file:/home/nonroot/openfga.db'

docker run --name openfga --network=openfga \
  -p 3000:3000 -p 8080:8080 -p 8081:8081 \
  -v openfga:/home/nonroot \
  -u nonroot \
  openfga/openfga run \
  --datastore-engine sqlite \
  --datastore-uri 'file:/home/nonroot/openfga.db'
```

Postgres 方式类似：

```bash
docker run --name openfga --network=openfga \
  -p 3000:3000 -p 8080:8080 -p 8081:8081 \
  openfga/openfga run \
  --datastore-engine postgres \
  --datastore-uri 'postgres://postgres:password@postgres:5432/postgres?sslmode=disable'
```

生产环境不要使用 in-memory datastore。Playground 适合开发和演示，不应该作为生产暴露面。PostgreSQL / MySQL 这类持久存储需要正常做迁移、备份、监控、连接池、慢查询分析和容量规划。

## 15. CLI 工作流

OpenFGA CLI 是把授权模型纳入工程流程的关键工具。典型开发路径不是“写完模型直接上线”，而是：

```bash
fga model validate --file model.fga
fga model test --tests store.fga.yaml
fga store import --file store.fga.yaml
fga query check user:anne viewer document:roadmap
fga query expand document:roadmap viewer
fga query list-objects user:anne viewer document
fga query list-users document:roadmap viewer
```

CLI 仓库的代码路径也体现了这个分工：

| 路径 | 对应能力 |
| --- | --- |
| `cmd/model/validate.go` | 模型校验 |
| `cmd/model/test.go` | 模型测试 |
| `cmd/store/import.go` / `export.go` | store 文件导入导出 |
| `cmd/tuple/write.go` / `read.go` / `delete.go` | tuple 管理 |
| `cmd/query/check.go` | Check 查询 |
| `cmd/query/expand.go` | Expand 查询 |
| `cmd/query/list-objects.go` | ListObjects 查询 |
| `cmd/query/list-users.go` | ListUsers 查询 |
| `internal/storetest/*` | 本地模型测试和 store 文件测试 |

工程上最应该引入的是 `fga model test`。授权模型没有测试就上线，相当于把权限系统当成了配置文件。更合理的做法是：每次模型变更都写一组正例和反例，CI 里运行模型测试，再发布新 model id。

## 16. 论文-代码对照：OpenFGA Server

静态阅读 `openfga/openfga` 当前 `main` 分支，可以把论文概念映射到几组代码路径。

| 论文/概念 | OpenFGA 代码路径 | 解读 |
| --- | --- | --- |
| 服务入口 | `cmd/openfga/main.go`、`cmd/run/run.go` | 启动 HTTP/gRPC server、加载配置 |
| 迁移 | `cmd/migrate/migrate.go` | 数据库 schema 迁移入口 |
| Check 解析 | `pkg/server/commands/check.go`、`internal/check/*`、`internal/graph/*` | 权限判断、递归解析、图遍历 |
| ListObjects | `pkg/server/commands/list_objects.go`、`internal/graph/object_providers.go` | 对象枚举 |
| Conditions | `internal/condition/*` | 条件解析、类型、求值 |
| 模型验证 | `internal/validation/*`、`internal/modelgraph/*` | 模型静态校验、关系图分析 |
| 授权保护 | `internal/authz/*` | OpenFGA 服务自身的访问控制相关逻辑 |
| 架构说明 | `docs/architecture/architecture.md`、`deployment.svg`、`internals.svg` | 项目内部架构文档 |

这说明 OpenFGA 的实现不是“查一张表然后返回 true/false”。Check 需要根据模型构建查询计划，遍历关系图，处理 computed userset、tuple-to-userset、conditions、contextual tuples、缓存、限制和错误。

在写技术报告时，代码对照的重点不是逐行解释 Go 代码，而是让读者知道：模型语言里的每个构造，最终都会影响查询图和执行成本。

## 17. 建模案例 1：文档协作系统

文档协作系统是 Zanzibar 和 OpenFGA 最容易理解的案例。

模型可以写成：

```text
model
  schema 1.1

type user

type group
  relations
    define member: [user]

type folder
  relations
    define parent: [folder]
    define viewer: [user, group#member]
    define editor: [user, group#member]
    define can_view: viewer or editor or can_view from parent
    define can_edit: editor or can_edit from parent

type document
  relations
    define parent: [folder]
    define owner: [user]
    define editor: [user, group#member] or owner
    define viewer: [user, group#member] or editor
    define can_view: viewer or can_view from parent
    define can_edit: editor or can_edit from parent
```

一些 tuple：

```text
folder:product#viewer@group:pm#member
group:pm#member@user:anne
document:roadmap#parent@folder:product
document:roadmap#owner@user:ben
```

那么：

- `user:anne` 通过 `group:pm#member` 获得 `folder:product#viewer`，进而获得 `document:roadmap#can_view`。
- `user:ben` 通过 owner 获得 editor，再获得 viewer 和 can_edit。

这个模型的好处是权限继承显式。坏处是如果文件夹层级很深，Check 成本会受递归深度影响。因此生产模型需要限制层级深度、监控查询耗时，并避免把所有业务树都无脑塞进授权模型。

## 18. 建模案例 2：SaaS 组织与项目

SaaS 系统里常见对象是 organization、project、resource。

```text
model
  schema 1.1

type user

type organization
  relations
    define member: [user]
    define admin: [user]
    define billing_admin: [user] or admin

type project
  relations
    define organization: [organization]
    define member: [user] or member from organization
    define admin: [user] or admin from organization
    define can_read: member or admin
    define can_write: admin

type report
  relations
    define project: [project]
    define viewer: [user]
    define can_view: viewer or can_read from project
    define can_edit: can_write from project
```

这个模型表达了几件事：

1. 组织 member 默认是项目 member。
2. 组织 admin 默认是项目 admin。
3. report 可以直接分享给 user，也可以继承 project 的 read/write。
4. billing 权限不应该自动等同于项目写权限，除非模型明确写出。

生产落地时要重点处理 tuple 生命周期。例如用户退出组织时，需要删除 `organization:acme#member@user:anne`；项目归属变化时，需要更新 `project:alpha#organization@organization:acme`；资源删除时，需要清理相关 tuple 或依赖后台清理任务。

## 19. 建模案例 3：GitHub 风格仓库权限

仓库权限常见关系包括 org member、team member、repo reader/writer/admin、branch protection、environment approver。

一个简化模型：

```text
model
  schema 1.1

type user

type organization
  relations
    define member: [user]
    define owner: [user]

type team
  relations
    define organization: [organization]
    define member: [user]

type repository
  relations
    define organization: [organization]
    define reader: [user, team#member] or member from organization
    define writer: [user, team#member]
    define admin: [user, team#member] or owner from organization
    define blocked: [user]
    define can_read: reader or writer or admin
    define can_write: (writer or admin) but not blocked
    define can_admin: admin
```

这里 `but not blocked` 看起来很自然，但要谨慎。Exclusion 会让解释权限更复杂：某个用户本来通过 team 获得 writer，但又因为 blocked 被排除。UI、审计和客服排查都要能说明这条拒绝路径。

这也是建模时的通用原则：允许路径可以多，拒绝路径要少而明确。

## 20. 工程接入路线

把已有系统迁移到 OpenFGA，不应该从“替换所有权限判断”开始。更稳妥的路线是：

| 阶段 | 目标 | 关键动作 |
| --- | --- | --- |
| 0. 盘点 | 看清现有权限事实 | 整理角色、资源、继承、例外、接口检查点 |
| 1. 建模 | 写出第一版 authorization model | 用 CLI validate/test 验证正反例 |
| 2. 双写 | 新业务变更同时写 legacy ACL 和 OpenFGA tuple | 不改变线上决策 |
| 3. Shadow | 在线请求同时调用 legacy 和 OpenFGA | 记录差异，不拦截 |
| 4. 灰度 | 小流量由 OpenFGA 决策 | 保留 legacy fallback 和 kill switch |
| 5. 收敛 | OpenFGA 成为主决策路径 | 建立模型版本、审计、监控和迁移流程 |

Shadow 阶段特别关键。权限系统最怕“看起来模型正确，但历史数据和边界场景不一致”。Shadow 差异日志应该至少包含：

- request id、user、relation、object。
- legacy result、OpenFGA result。
- model id、store id、tuple 写入版本。
- 差异原因分类：缺 tuple、模型规则错误、历史脏数据、业务例外未建模、调用方传错对象。

不要在差异归零前直接切主路径。

## 21. 性能与可观测性

Zanzibar 论文报告了非常高的生产规模和低延迟。

![Rate of Check Safe and Check Recent requests](/images/blog/openfga-zanzibar-fine-grained-authorization/zanzibar-fig03-check-safe-recent-rate.webp)

*Source: Pang et al., USENIX ATC 2019, Fig. 3.*

![Latency of Check Safe responses](/images/blog/openfga-zanzibar-fine-grained-authorization/zanzibar-fig04-check-safe-latency.webp)

*Source: Pang et al., USENIX ATC 2019, Fig. 4.*

![RPC response latency](/images/blog/openfga-zanzibar-fine-grained-authorization/zanzibar-table02-rpc-latency.webp)

*Source: Pang et al., USENIX ATC 2019, Table 2.*

这些数字只能说明 Zanzibar 论文系统的表现。OpenFGA 的性能取决于模型、tuple 规模、存储后端、网络、部署方式和查询类型。

实际生产中至少要监控：

| 指标 | 为什么重要 |
| --- | --- |
| Check latency p50/p95/p99 | 影响每个业务接口 |
| Check error rate | 权限服务失败通常要 fail closed |
| datastore latency | OpenFGA 很多慢查询最终来自存储 |
| recursive depth / dispatch count | 模型复杂度的运行时表现 |
| ListObjects/ListUsers latency | 枚举查询比 Check 更容易超预算 |
| tuple write rate | 组织、成员、资源变更高峰会影响存储 |
| model id 分布 | 确认调用方是否使用了预期版本 |
| allow / deny ratio | 发现异常放行或异常拒绝 |

Zanzibar 论文里的可用性图也要按边界理解。

![Zanzibar availability](/images/blog/openfga-zanzibar-fine-grained-authorization/zanzibar-fig05-availability.webp)

*Source: Pang et al., USENIX ATC 2019, Fig. 5.*

它说明 Google 的 Zanzibar 生产系统在论文统计窗口里的表现，不说明你本地 Docker、单库 Postgres 或普通 Kubernetes 部署也能自动达到这个水平。

## 22. 安全边界

OpenFGA 是授权系统，不是认证系统。它不会帮你证明请求里的 `user:anne` 真的是 Anne。

正确的请求链路通常是：

```text
client
  -> application gateway / identity provider
  -> application service parses authenticated subject
  -> application service calls OpenFGA Check
  -> application service enforces allow / deny
```

几条安全边界必须写进工程规范：

1. **用户身份来自认证系统**：不要让前端自己传 `user` 字符串决定身份。
2. **OpenFGA 服务自身要鉴权**：不能让任意内部服务随意写 tuple 或改模型。
3. **权限检查在服务端执行**：前端隐藏按钮只是体验优化，不是安全边界。
4. **tuple 不放敏感明文**：object id 和 user id 尽量使用不可逆内部 id，避免泄露 PII。
5. **写权限比读权限更敏感**：谁能创建 store、写 model、写 tuple，本身也需要权限治理。
6. **默认拒绝**：OpenFGA 错误、超时、模型缺失、对象 id 异常时，不要默认放行。

## 23. 常见错误清单

| 错误 | 后果 | 更好的做法 |
| --- | --- | --- |
| 把 OpenFGA 当认证系统 | 伪造 user 直接绕过权限 | 先做认证，再用认证主体调用 Check |
| 把 role 字符串直接搬进模型 | 模型仍然是硬编码 RBAC | 用 relation 表达资源关系和继承 |
| 未曝光所有历史例外 | Shadow 一切主路径后大量误拒 | 先收集 legacy 差异并分类 |
| 没有模型测试 | 小改动破坏关键权限 | 每个模型变更都跑正反例测试 |
| 过度使用 contextual tuples | 线上行为难复现 | 长期事实写 tuple，临时事实才 contextual |
| 把 ListObjects 当搜索引擎 | 查询成本不可控 | 搜索候选 + 授权过滤，或严格控制枚举规模 |
| 删除资源不清 tuple | 残留权限污染审计和枚举 | 建立 tuple 生命周期和后台清理 |
| 把 Zanzibar 指标当 OpenFGA SLA | 错误容量评估 | 对自己的模型和存储做压测 |

## 24. 与相关系统对比

OpenFGA 常和 OPA、Cedar、Casbin、SpiceDB/Authzed、Ory Keto 一起讨论。它们解决的是相邻问题，但心智模型不同。

| 系统 | 主要心智模型 | 更适合的场景 |
| --- | --- | --- |
| OpenFGA | Zanzibar 风格 ReBAC，关系元组 + 授权模型 | 资源关系复杂、共享/继承/组权限明显的应用 |
| SpiceDB/Authzed | Zanzibar 风格 ReBAC，强调生产级关系权限 | 类似 OpenFGA 的关系授权场景 |
| OPA | 通用策略引擎，Rego 规则 | 基础设施策略、Kubernetes、请求策略、合规规则 |
| Cedar | AWS 推出的策略语言 | 以主体、资源、动作、上下文为核心的策略授权 |
| Casbin | 多模型访问控制库 | 嵌入式 RBAC/ABAC/ACL 模型，语言生态丰富 |
| Ory Keto | Zanzibar 启发的权限服务 | 关系授权服务化场景 |

不要把这些系统简单排成“谁更强”。关键是你的主问题是什么：

- 如果你需要表达“用户与资源之间的关系图”，OpenFGA / SpiceDB 这类 ReBAC 系统更自然。
- 如果你需要表达“请求必须满足一组策略条件”，OPA / Cedar 可能更自然。
- 如果你的系统只需要简单角色和本地库，Casbin 可能更轻。

## 25. 生产落地清单

一套可上线的 OpenFGA 接入至少要覆盖下面这些项。

| 类别 | 检查项 |
| --- | --- |
| 模型 | 有版本号、有测试、有 review、有回滚 |
| 数据 | tuple 双写、清理、幂等、重放、备份 |
| 服务 | HTTP/gRPC 超时、重试、熔断、fail closed |
| 存储 | Postgres/MySQL HA、迁移、连接池、慢查询 |
| 观测 | latency、error、allow/deny、model id、store id、dispatch |
| 安全 | OpenFGA API 鉴权、写 model/write tuple 权限、审计日志 |
| 灰度 | shadow、差异日志、分租户切换、kill switch |
| 文档 | relation 语义、对象命名、tuple 生命周期、排障手册 |

尤其要注意模型版本。OpenFGA 的 authorization model 是版本化的。业务服务调用时要明确使用哪个 model id，否则“新模型已写入但线上仍在用旧模型”会变成非常隐蔽的问题。

## 26. 推荐阅读路径

如果只想快速理解，推荐顺序如下：

1. 先读 Zanzibar 论文 Abstract、Section 2、Fig. 1、Fig. 2。
2. 再读 OpenFGA Concepts，理解 store、model、tuple、relation、object、user。
3. 读 Configuration Language，重点看 direct relationship、userset、`viewer from parent`、intersection、exclusion。
4. 用 CLI 写一个 folder/document 模型，并跑 `fga model test`。
5. 读 OpenFGA API 的 Check、Expand、ListObjects、ListUsers。
6. 最后读 `openfga/openfga` 的 `internal/check`、`internal/graph`、`pkg/server/commands` 和 CLI 的 `cmd/query`、`cmd/model`。

不要一开始就研究全部部署参数。OpenFGA 入门最难的是模型，不是 Docker 命令。

## 27. 结论

OpenFGA 的长期价值在于把“谁能对什么资源做什么”这件事变成一个显式系统，而不是散落在业务代码里的隐式习惯。

它最适合的场景，是资源关系复杂、共享和继承明显、权限需要频繁演进的系统：文档协作、SaaS 组织项目、开发者平台、RAG 文档授权、企业知识库、数据权限、审批和工作流平台。

但 OpenFGA 不会自动解决所有授权问题。它不负责认证，不负责业务数据存储，不负责搜索排序，不负责把错误模型变正确，也不保证你的部署具备 Zanzibar 论文里的全球一致性和生产规模。它把授权问题变成了可建模、可测试、可查询的系统工程，后续的模型设计、tuple 生命周期、灰度迁移、可观测性和安全治理仍然需要工程团队认真完成。

一句话总结：

> OpenFGA 的核心不是 `allowed: true`，而是让复杂权限关系从业务代码里退出来，成为可以被版本化、测试、迁移和审计的授权模型。

## References

- [Zanzibar: Google's Consistent, Global Authorization System](https://www.usenix.org/conference/atc19/presentation/pang)
- [Zanzibar paper PDF](https://www.usenix.org/system/files/atc19-pang.pdf)
- [OpenFGA Concepts](https://openfga.dev/docs/concepts)
- [OpenFGA Configuration Language](https://openfga.dev/docs/configuration-language)
- [OpenFGA Relationship Queries](https://openfga.dev/docs/interacting/relationship-queries)
- [OpenFGA API](https://openfga.dev/api/service)
- [openfga/openfga](https://github.com/openfga/openfga)
- [openfga/cli](https://github.com/openfga/cli)

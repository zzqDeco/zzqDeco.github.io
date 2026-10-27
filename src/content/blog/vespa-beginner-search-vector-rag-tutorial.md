---
title: "Vespa 入门教程：从本地搜索应用到向量检索、混合排序与 RAG 基础"
description: "系统讲解 Vespa 的应用包、Schema、YQL、文档写入、BM25、向量近邻、混合检索、排序配置和本地/云端部署入门流程"
pubDate: "2026-07-02T18:20:51+08:00"
updatedDate: "2026-07-02T18:20:51+08:00"
tags:
  - "Deep Reading"
  - "Vespa"
  - "Search Engine"
  - "Vector Search"
  - "Hybrid Search"
  - "RAG"
  - "Tutorial"
draft: false
---

Vespa 这个名字容易让人先想到机车，但本文讲的是 **Vespa.ai / Vespa Engine**：一个开源的大规模数据 serving engine，用来构建搜索、推荐、个性化、向量检索、RAG 和机器学习排序系统。如果只把 Vespa 叫作“向量数据库”，会漏掉它最重要的部分。Vespa 可以存向量、做 ANN，但它真正的定位更像一个统一的 AI retrieval serving platform：在同一个系统里同时处理全文检索、结构化过滤、向量近邻、tensor 计算、多阶段排序、模型推理、实时更新和分布式低延迟服务。官方首页对它的描述很直接：Vespa 可以在 vectors、tensors、text 和 structured data 上查询、组织和推理，并服务搜索、RAG、推荐和半结构化导航等场景。

本文定位于一篇从零上手的 Vespa 工程实操教程。我们会用一个统一的 `docs-search` 示例，从本地 Docker 启动开始，编写 application package，定义 schema，写入（feed）文档，执行 YQL 查询，加入 BM25 与向量字段，完成 hybrid search 和 rank-profile 调试，最后将其扩展到生产级 RAG chunk schema。读完本文后，你应该能清晰回答三个核心问题：

1. Vespa 的 application package、schema、`services.xml`、YQL 和 rank-profile 分别承担什么职责。
2. 一个最小本地 Vespa 搜索应用怎样完成部署、写入、更新、查询和特征调试。
3. Vespa 的 BM25、向量近邻、多阶段混合排序以及在 RAG / 推荐系统中的后端角色应该如何理解与演进。

本文默认你有 Docker 或 Podman 基础，能在命令行里执行 `docker`、`curl`、`python3` 和 Vespa CLI。所有配置与代码均以完整代码块给出，实操时可直接复制到本地实验目录运行。

## 1. 为什么学 Vespa：定位、场景与核心心智模型

### 1.1 适用场景与定位：不止是向量数据库

Vespa 不是最轻量的工具。如果你的需求只是存 1 万条向量、快速搭一个个人 demo，那么 Milvus、Qdrant、Chroma、SQLite 向量扩展甚至直接调用 FAISS，上手都会更快。然而，当你的业务需求开始演进为下面这些复杂形态时，Vespa 就值得认真评估：

| 业务需求 | 为什么 Vespa 合适 |
| --- | --- |
| 搜索既要关键词匹配，又要语义向量召回 | Vespa 在同一引擎内原生支持 lexical search、vector search 和统一的 ranking pipeline |
| 检索结果需要复杂排序与实时特征 | rank-profile 可以组合 BM25、向量相似度、业务属性、模型特征和 tensor 表达式 |
| 数据高频变化且要求实时可见 | Vespa 强调 real-time updates 与在线局部字段更新，而非离线建好索引后长期不动 |
| RAG 需要过滤、分片、排序与证据返回 | 文档字段、structured filter、`nearestNeighbor` 和 summary 裁剪可在单次查询中闭环 |
| 推荐系统需要候选召回和在线多阶段排序 | Vespa 可同时承担向量召回、实时特征访问与 first-phase / second-phase / global-phase 排序 |
| 希望减少多系统拼接成本 | 设计目标是尽量让 retrieval、ranking、inference 靠近数据原地执行，避免跨服务搬运候选集 |

把 Vespa 放回大家熟悉的搜索与向量生态中横向对比，可以更直观地看清它的坐标（以下对比仅帮助理解系统边界，并非优劣排名）：

| 系统 | 直觉定位 | 典型适用场景 |
| --- | --- | --- |
| Elasticsearch / OpenSearch | 经典全文搜索与日志/文档分析生态 | 文本搜索、日志检索、多维聚合、已有 ELK/OpenSearch 生态 |
| Milvus | 面向大规模 embedding 的专用向量检索系统 | 亿级向量存储、ANN 召回、以纯向量相似为主的数据集 |
| Qdrant | 开发者友好的轻量向量数据库 | 快速搭建语义搜索、带 payload filter 的向量召回服务 |
| Vespa | 统一 AI retrieval serving engine | 文本 + 向量 + 结构化过滤 + 多阶段排序 + 在线模型推理 + 实时更新 |

从官方架构图中可以清楚看到 Vespa 在完整 AI retrieval workflow 中的位置：

![Vespa distributed architecture](https://vespa.ai/vespa-content/uploads/2026/06/Vespa-Distributed-Architecture-1.png)

*Source: Vespa.ai Architecture, official website.*

这张架构图将 Vespa 的核心能力归纳为 Retrieve、Rank、Infer、Update、Serve 五个环节，底层由 unified data & compute、tensor engine、query execution 和 distributed architecture 支撑。官方架构页传达的核心设计哲学是：不要把全文检索、向量检索、特征存储、模型推理和在线服务拆成多个彼此通过网络搬运大量候选数据的独立系统，而是尽量在统一的分布式架构中、靠近数据所在节点完成这些计算。

这也解释了 Vespa 与普通向量数据库的本质差异。普通向量库通常回答的问题是：

```text
给我一个 query vector，返回最相似的 topK vectors。
```

而 Vespa 在生产环境中更常回答的问题是：

```text
给我一个文本 query、query vector、用户上下文和结构化过滤条件；
先用关键词、向量和元数据过滤联合召回候选集；
再用 rank-profile 在数据节点就地组合 BM25、closeness、业务新鲜度与模型输出完成多阶段排序；
最后按指定 summary 裁剪返回结果并附带可解释的调试特征。
```

因此，本文将 Vespa 当作搜索、RAG 与推荐的在线 serving 基础设施来讲解。它不是 prompt orchestration 框架，也不是离线训练框架；你可以把 embedding 生成、LLM 调用、业务逻辑和前端 UI 放在外部服务中，由 Vespa 专门负责把最匹配的候选文档找出来、排好序、返回必要字段。

### 1.2 核心概念与读写链路

Vespa 的学习曲线之所以比单一向量库更陡，主要在于需要建立一套统一的配置与执行心智模型。入门阶段需要先掌握五个核心概念：

| 概念 | 职责与作用 | 工程类比 |
| --- | --- | --- |
| application package | 定义一个 Vespa 应用的完整声明式配置包 | 搜索服务的配置工程目录 |
| schema | 定义文档类型、字段、索引、summary 与 rank profiles | 数据表结构 + 索引策略 + 排序算子配置 |
| document | 被写入 Vespa 的单条数据记录 | 搜索文档、商品条目、文章或 RAG chunk |
| container cluster | 无状态请求处理层，暴露 HTTP 接口与自定义组件 | API Gateway + Query Planner + Document 路由层 |
| content cluster | 有状态数据存储、索引维护与检索计算层 | 分片存储引擎 + 倒排/向量索引 + 就地排序节点 |

官方文档中的 Vespa Overview 图清晰展示了容器层（Stateless Container Cluster）与内容层（Stateful Content Cluster）的协作关系：

![Vespa overview](https://docs.vespa.ai/assets/img/vespa-overview.svg)

*Source: Vespa Documentation, official docs.*

图中最值得关注的是无状态层与有状态层的职责边界：container cluster 负责接收 HTTP 查询与写入请求、解析 YQL、分发子查询并合并最终响应；content cluster 则负责文档持久化、属性更新、倒排与 HNSW 索引维护，以及在数据本地执行候选匹配和排序计算。这两条核心读写链路可以用下面的流程图概括：

```text
【查询链路 Query Path】
client
  |  HTTP / Vespa CLI query
  v
container cluster
  |  parse YQL, prepare query, fan out to content nodes
  v
content cluster
  |  match candidates by text / vector / filter
  |  compute first-phase & second-phase ranking close to data
  v
container cluster
  |  merge top hits, optional global-phase, fill summaries, render JSON
  v
client

【写入链路 Feed Path】
client / feed pipeline
  |  document JSON / JSONL (put / update / remove)
  v
container document-api
  |  validate & route document operation by document ID
  v
content cluster
  |  store fields, update attributes in memory, build inverted & HNSW indexes
  v
document becomes immediately visible to queries
```

查询链路和写入链路完全由 application package 驱动：`services.xml` 决定集群拓扑与节点分工，`schemas/*.sd` 决定文档如何存储、如何建索引、如何裁剪返回以及如何计算排序得分。

## 2. 本地环境准备与第一个 Application Package

### 2.1 Docker 容器启动与 Vespa CLI 配置

入门阶段使用本地 Docker 容器是最直接的方式。在开始前请准备以下本地环境：

- **Docker Desktop 或 Podman**：确保守护进程正在运行，且至少分配 **4GB 及以上可用内存**（官方 hybrid search 教程同样将 4GB dedicated memory 作为最低要求，内存不足会导致部署缓慢或容器 OOM 退出）。
- **Vespa CLI**：可通过 Homebrew（`brew install vespa-cli`）或官方 GitHub Releases 安装。
- **`curl` 与 `python3`**：用于验证 HTTP 端点和生成带 384 维向量的测试数据。

本教程使用 Vespa 容器的两个标准端口：`8080` 用于查询、写入和 Document API 等数据面 HTTP 请求；`19071` 用于 config server 和应用部署控制面。执行以下命令拉取镜像并启动本地容器：

```bash
docker pull vespaengine/vespa

docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
```

容器启动后，可以通过 `docker logs -f vespa` 观察启动过程，或使用容器内置的日志格式化工具查看可读性更好的服务日志：

```bash
docker exec vespa vespa-logfmt
```

接着将 Vespa CLI 的目标环境指向本地，并检查服务就绪状态：

```bash
vespa config set target local
vespa status
```

如果 `vespa status` 报错，请先不要急着编写 schema，优先检查 Docker 容器是否正常运行（`docker ps`）、`8080` 与 `19071` 端口是否被本机其他进程占用、Docker 内存配额是否充足，或者清理掉之前残留的同名容器（`docker rm -f vespa`）后重新启动。

### 2.2 最小工程结构与 `services.xml` 精读

我们创建一个名为 `docs-search` 的示例应用，用来检索技术文档片段，逐步支持全文检索、标签过滤、向量召回与混合排序。入门阶段只需要两个文件即可构成最小可用的 application package：

```text
docs-search/
  services.xml
  schemas/
    doc.sd
```

其中 `services.xml` 描述 Vespa 的服务集群拓扑，`schemas/doc.sd` 描述文档类型 `doc` 的字段、索引与排序规则。可以把整个 `docs-search/` 目录理解为搜索服务的声明式基础设施配置。先在本地创建该目录并进入工作区（后文所有 `vespa deploy` 命令默认均在 `docs-search/` 目录下执行）：

```bash
mkdir -p docs-search/schemas
cd docs-search
```

将以下配置写入 `docs-search/services.xml`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<services version="1.0">
  <container id="default" version="1.0">
    <search />
    <document-api />
    <nodes>
      <node hostalias="node1" />
    </nodes>
  </container>

  <content id="docs" version="1.0">
    <redundancy>1</redundancy>
    <documents>
      <document type="doc" mode="index" />
    </documents>
    <nodes>
      <node hostalias="node1" distribution-key="0" />
    </nodes>
  </content>
</services>
```

这份配置文件清晰对应了前文介绍的两层架构：

| 配置节点 | 核心含义与工程作用 |
| --- | --- |
| `<container id="default">` | 定义无状态容器集群，负责接管外部 HTTP 请求 |
| `<search />` | 在容器集群上启用查询处理与 `/search/` 端点 |
| `<document-api />` | 在容器集群上启用文档读写与 `/document/v1/` 端点 |
| `<content id="docs">` | 定义有状态内容集群，负责存储数据、维护索引与执行匹配计算 |
| `<document type="doc" mode="index" />` | 声明该内容集群存储 `doc` 类型文档（必须与 `doc.sd` 的 schema 名一致），并启用 `index` 模式 |
| `<redundancy>1</redundancy>` | 数据副本数设为 1，本地单节点实验无需多副本；生产环境需根据高可用要求调整 |
| `<node hostalias="node1">` | 将容器与内容服务绑定到本地单节点的默认主机别名 `node1` |

在本地开发阶段，我们只需要记住“container 接请求、content 存数据并算排序”；到了生产环境中，可以在这两个集群下扩展多个节点、配置多副本与分组分布，而无需改动上层查询代码。

## 3. Schema 设计：字段、索引、属性与演进规范

### 3.1 编写 `doc.sd`：`index`、`attribute` 与 `summary` 的边界

将以下内容写入 `docs-search/schemas/doc.sd`，这份 schema 一次性包含了后续实验所需的文本、结构化属性、向量索引和多套排序配置：

```text
schema doc {
  document doc {
    field id type string {
      indexing: attribute | summary
      match: word
    }

    field title type string {
      indexing: index | summary
      match: text
      index: enable-bm25
    }

    field body type string {
      indexing: index | summary
      match: text
      index: enable-bm25
    }

    field url type string {
      indexing: attribute | summary
      match: word
    }

    field tags type array<string> {
      indexing: attribute | summary
    }

    field published_at type long {
      indexing: attribute | summary
      attribute: fast-search
    }

    field embedding type tensor<float>(x[384]) {
      indexing: attribute | index
      attribute {
        distance-metric: angular
      }
      index {
        hnsw {
          max-links-per-node: 16
          neighbors-to-explore-at-insert: 100
        }
      }
    }
  }

  fieldset default {
    fields: title, body
  }

  document-summary compact {
    summary id {}
    summary title {}
    summary url {}
    summary tags {}
    summary published_at {}
  }

  rank-profile default {
    first-phase {
      expression: nativeRank(title, body)
    }
  }

  rank-profile bm25 inherits default {
    first-phase {
      expression: bm25(title) + bm25(body)
    }
    match-features {
      bm25(title)
      bm25(body)
    }
  }

  rank-profile semantic {
    inputs {
      query(q) tensor<float>(x[384])
    }
    first-phase {
      expression: closeness(field, embedding)
    }
    match-features {
      closeness(field, embedding)
      distance(field, embedding)
    }
  }

  rank-profile hybrid inherits semantic {
    first-phase {
      expression: bm25(title) + bm25(body) + 0.5 * closeness(field, embedding)
    }
    match-features {
      bm25(title)
      bm25(body)
      closeness(field, embedding)
      distance(field, embedding)
    }
  }
}
```

初学者阅读 Vespa schema 时，最容易混淆的是 `indexing:` 语句里的三个核心关键字：`index`、`attribute` 和 `summary`。它们分别决定了字段在底层的物理存储方式与可参与的计算阶段：

- **`index`（倒排/图索引）**：对于文本字段（如 `title`、`body`），`index` 表示分词后构建倒排索引，用于全文检索、词项匹配与 BM25 计算；只有显式声明了 `index: enable-bm25`，才能在 `rank-profile` 中调用 `bm25(field)`。对于 tensor 向量字段（如 `embedding`），`index` 表示构建 HNSW 近似近邻图索引。
- **`attribute`（内存列式属性）**：表示字段以内存属性结构驻留，适合结构化精确过滤、范围比较、分组聚合（grouping）、排序算子实时读取（如 `attribute(published_at)`）以及向量距离计算。对高频用于 `where` 过滤的标量或数组字段，可以加上 `attribute: fast-search` 为其构建额外的查找结构；但对仅偶尔展示、不参与高频过滤的字段，无需开启 `fast-search` 以免增加内存开销。
- **`summary`（结果返回摘要）**：表示该字段写入文档摘要存储，允许在查询命中后随结果返回给客户端。如果某个字段没有配置 `summary`，即使它在查询中被成功匹配或参与了排序，最终返回的 JSON `fields` 里也不会出现该字段。此外，通过定义 `document-summary compact`，可以在查询时按需只拉取轻量字段子集，避免每次都传输大段正文或高维向量。

在此基础上，`fieldset default` 指定了当查询使用 `userQuery()` 时默认搜索 `title` 和 `body` 两个文本字段；而 `rank-profile` 则定义了 `default`、`bm25`、`semantic` 和 `hybrid` 四套排序策略。我们可以把字段配置原则总结为下表：

| 字段用途 | 推荐 Schema 配置 | 本例对应字段 |
| --- | --- | --- |
| 全文检索与相关性打分 | `indexing: index \| summary` + `match: text` + `index: enable-bm25` | `title`, `body` |
| 结构化过滤与排序特征 | `indexing: attribute \| summary`（高频过滤加 `attribute: fast-search`） | `tags`, `published_at` |
| 精确标识与展示元数据 | `indexing: attribute \| summary` + `match: word` | `id`, `url` |
| 稠密向量 ANN 检索 | `tensor<float>(x[D])` + `indexing: attribute \| index` + `distance-metric` + `hnsw` | `embedding` |

### 3.2 Schema 演进风险与向量版本管理

在本地学习时，修改 `doc.sd` 后直接执行 `vespa deploy` 即可生效；但在生产环境中，Schema 变更会直接影响底层索引结构、内存占用、写入管线和在线查询兼容性。不同类型变更的风险等级差异很大：

| Schema 变更类型 | 线上影响与风险等级 | 工程处理建议 |
| --- | --- | --- |
| 新增 `summary` 字段或 `document-summary` | 低风险，不影响既有索引匹配 | 部署后新写入文档即生效，历史文档视需要补写 |
| 新增 `attribute` 或 `index` 字段 | 中等风险，需要重新 feed 或 partial update 数据才能命中有值记录 | 先部署 schema，再异步回填字段数据，最后切流查询 |
| 修改 `rank-profile` 表达式 | 排序分布变化，可能导致线上相关性波动 | 新增带版本号的 profile（如 `hybrid_v2`），经离线评估与灰度对比后再替换默认配置 |
| 修改字段类型或修改 `tensor` 维度 | 高风险破坏性变更，涉及底层存储重建或 `validation-overrides` | 避免原地强改，建议新增字段或新建集群双写迁移 |
| 升级 Embedding 模型版本 | 极高隐蔽风险，新旧向量空间不兼容会导致召回质量静默崩塌 | 严禁将新模型的 query vector 与旧模型的 document embedding 混搜 |

一个稳健的线上 Schema 演进流程应遵循：`schema 变更提案 -> 本地容器验证 -> Staging 部署并回放样本数据 -> 离线查询回归与延迟压测 -> 生产灰度发布 -> 监控指标与一键回滚预案`。

尤其需要注意的是向量字段的模型版本管理：一旦切换了 embedding 模型（哪怕维度同样是 384 维），新旧向量也不在同一个语义空间内。工程实践中建议在 schema 中显式增加版本属性字段：

```text
field embedding_version type string {
  indexing: attribute | summary
  match: word
  attribute: fast-search
}
```

并在查询时强制带上模型版本过滤条件，从而让模型升级、双写灰度和紧急回滚完全可控：

```bash
vespa query \
  'yql=select * from doc where embedding_version contains "e5-small-v1" and ({targetHits:50}nearestNeighbor(embedding, q))' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic'
```

## 4. 数据写入（Feed）、首次部署与 Document API

### 4.1 构造 JSONL 数据与 Document ID 规范

Vespa 支持通过 JSON 或 JSONL 格式批量写入文档，每一行代表一个独立的文档操作。Vespa 的全局文档唯一标识符（Document ID）遵循固定格式：

```text
id:<namespace>:<document-type>::<user-specific-id>
```

在我们的示例中，形如 `id:docs-search:doc::hybrid-search` 的 ID 里，`docs-search` 是业务命名空间（namespace），`doc` 必须严格对应 `doc.sd` 中声明的 document type，双冒号后面的 `hybrid-search` 则是业务侧主键。

为了让包含 384 维向量的 schema 可以直接在本地跑通，我们用一段简短的 Python 脚本生成 4 条测试文档并保存为 `documents.jsonl`。注意这里的 `embedding` 只是用正弦函数生成的确定性测试向量（toy embedding），仅用于打通数据与查询链路；真实生产系统应调用句向量模型生成：

```bash
python3 - <<'PY' > documents.jsonl
import json
import math

docs = [
    {
        "id": "vespa-basics",
        "title": "Vespa application package basics",
        "body": "Learn how services.xml, schema, document fields and rank profiles work together in Vespa.",
        "url": "https://docs.vespa.ai/en/basics/",
        "tags": ["vespa", "schema", "beginner"],
        "published_at": 1760000000,
        "seed": 1,
    },
    {
        "id": "hybrid-search",
        "title": "Hybrid search with BM25 and vector retrieval",
        "body": "Hybrid search combines lexical retrieval, vector nearest neighbor search and ranking profiles.",
        "url": "https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html",
        "tags": ["vespa", "hybrid-search", "vector-search"],
        "published_at": 1760000100,
        "seed": 2,
    },
    {
        "id": "rag-backend",
        "title": "Vespa as a retrieval backend for RAG",
        "body": "A RAG backend stores chunks, filters by metadata, retrieves candidates and ranks evidence.",
        "url": "https://docs.vespa.ai/en/rag/",
        "tags": ["rag", "retrieval", "ranking"],
        "published_at": 1760000200,
        "seed": 3,
    },
    {
        "id": "recommendation-serving",
        "title": "Real time recommendation serving",
        "body": "Recommendation systems need candidate generation, filtering, ranking and real time feature updates.",
        "url": "https://vespa.ai/",
        "tags": ["recommendation", "serving", "ranking"],
        "published_at": 1760000300,
        "seed": 4,
    },
]

def embedding(seed, dim=384):
    values = []
    for i in range(dim):
        values.append(round(math.sin(seed * 0.17 + i * 0.013), 6))
    return values

for d in docs:
    fields = {
        "id": d["id"],
        "title": d["title"],
        "body": d["body"],
        "url": d["url"],
        "tags": d["tags"],
        "published_at": d["published_at"],
        "embedding": embedding(d["seed"]),
    }
    print(json.dumps({
        "put": f"id:docs-search:doc::{d['id']}",
        "fields": fields,
    }))
PY
```

执行 `head -n 1 documents.jsonl` 可以查看生成的单行结构。除了全量写入（`put`）之外，Vespa 的 `<document-api />` 还支持单文档读取（`get`）、局部字段更新（`update`，例如只实时刷新某篇文档的 `tags` 或点击率特征而无需重传正文字段与向量）以及删除（`remove`），这些 CRUD 操作既可以通过 `vespa document` CLI 执行，也可以直接请求 `/document/v1/<namespace>/<doc-type>/docid/<id>` HTTP 接口。

### 4.2 首次部署、批量写入与基础连通性验证

确认当前终端位于 `docs-search/` 目录下（执行 `ls` 应能看到 `services.xml`、`schemas/` 和 `documents.jsonl`），然后依次执行应用部署与数据写入：

```bash
vespa config set target local
vespa deploy
vespa feed documents.jsonl
```

当控制台返回部署成功且 4 条文档全部写入完成后，我们先执行一条不带任何复杂排序的最小查询来验证端到端连通性：

```bash
vespa query 'yql=select * from doc where true limit 3'
```

如果返回如下结构的 JSON，且 `root.fields.totalCount` 为 `4`、`root.children` 中包含刚才写入的文档列表，说明容器层、内容层、Schema 索引与摘要返回链路已经全部跑通：

```json
{
  "root": {
    "id": "toplevel",
    "relevance": 1.0,
    "fields": {
      "totalCount": 4
    },
    "children": [
      {
        "id": "id:docs-search:doc::vespa-basics",
        "relevance": 0.0,
        "source": "docs",
        "fields": {
          "id": "vespa-basics",
          "title": "Vespa application package basics"
        }
      }
    ]
  }
}
```

## 5. YQL 查询、BM25 全文检索与结构化过滤

### 5.1 YQL 基础语法与结构化过滤

Vespa 使用 YQL（Vespa Query Language）描述“从哪个 schema 中、按什么条件取出哪些字段”。其最核心的职责边界是：**YQL 负责决定哪些文档进入候选集（Retrieval & Filtering），而 `rank-profile` 负责决定这些候选文档如何计算得分与排序（Ranking）**。尽管 YQL 也支持 `order by` 子句按属性字段直接排序，但在构建搜索和 RAG 系统时，应尽量把召回与过滤条件写在 YQL 中，把相关性打分逻辑收敛在 `rank-profile` 中。

下面是一组最常用的基础 YQL 与结构化过滤查询示例：

```bash
# 1. 只投影返回需要的字段（减少网络传输开销）
vespa query 'yql=select id,title,url from doc where true limit 2'

# 2. 使用 userQuery() 在默认 fieldset (title, body) 上执行文本检索
vespa query \
  'yql=select id,title,url from doc where userQuery() limit 5' \
  'query=hybrid search'

# 3. 使用 contains 指定在单个文本字段上匹配词项
vespa query \
  'yql=select id,title,url from doc where title contains "Vespa" limit 5'

# 4. 在数组属性字段 tags 上做结构化过滤
vespa query \
  'yql=select id,title,tags from doc where tags contains "vector-search" limit 5'

# 5. 在数值属性字段 published_at 上做时间范围过滤并按时间倒序排列
vespa query \
  'yql=select id,title,published_at from doc where published_at > 1760000000 order by published_at desc limit 10'

# 6. 组合结构化标签过滤与全文检索
vespa query \
  'yql=select id,title,tags from doc where tags contains "vespa" and userQuery() limit 10' \
  'query=rank profile' \
  'ranking=bm25'
```

在真实的电商、文档站或知识库搜索 UI 中，除了过滤之外，还经常需要在搜索结果侧边栏展示“按标签或分类聚合统计数量”的导航面板（例如 `tag: vespa (10)`、`tag: rag (4)`）。Vespa 在 YQL 中内置了强大的 Grouping / Faceting 聚合语法（形如 `select * from doc where ... | all(group(tags) each(output(count())))`），能够在一次检索过程中同时返回排序结果和分面统计计数。初学阶段了解这一能力边界即可，建议先将主链路的文本、向量与混合排序调通，再按需引入分面聚合。

### 5.2 BM25 文本相关性检索与特征验证

在前面的 `doc.sd` 中，我们已经为 `title` 和 `body` 开启了 `index: enable-bm25`，并定义了将两者 BM25 分数相加的 `bm25` 排序配置，同时用 `match-features` 暴露了各个字段的分项得分：

```text
rank-profile bm25 inherits default {
  first-phase {
    expression: bm25(title) + bm25(body)
  }
  match-features {
    bm25(title)
    bm25(body)
  }
}
```

现在我们可以显式指定 `ranking=bm25` 发起查询，并通过 `ranking.listFeatures=true` 打开特征观察开关：

```bash
vespa query \
  'yql=select id,title,url from doc where userQuery() limit 5' \
  'query=hybrid search tutorial' \
  'ranking=bm25' \
  'ranking.listFeatures=true'
```

在返回的每条结果中，除了总分 `relevance` 外，你还能在 `matchfeatures` 字段里清晰看到 `bm25(title)` 和 `bm25(body)` 各自贡献了多少分。BM25 的核心优势在于词面精确匹配能力强、计算高效且完全可解释，特别适合包含专有名词、代码标识符、型号缩写和明确关键词的查询；但它的局限性同样明显——无法跨越词面鸿沟理解同义表达（例如用户搜索 `semantic retrieval` 时，若文档里只写了 `vector nearest neighbor`，纯 BM25 就可能漏召回）。这正是我们需要引入向量检索的原因。

## 6. 向量检索与 Hybrid Search（混合检索）

### 6.1 Tensor 向量字段与 HNSW 索引配置

Vespa 使用统一的 `tensor` 类型系统来表达一维向量和更高维张量。回顾我们在 `doc.sd` 中为 `embedding` 字段编写的定义：

```text
field embedding type tensor<float>(x[384]) {
  indexing: attribute | index
  attribute {
    distance-metric: angular
  }
  index {
    hnsw {
      max-links-per-node: 16
      neighbors-to-explore-at-insert: 100
    }
  }
}
```

这段配置中的每一项都有明确的工程含义：

| 配置项 | 含义与调参取舍 |
| --- | --- |
| `tensor<float>(x[384])` | 声明一个名为 `x`、维度固定为 384 的单精度浮点稠密向量 |
| `indexing: attribute \| index` | 同时将向量存入内存属性并构建 ANN 索引，支持快速近似近邻检索与精确距离计算 |
| `distance-metric: angular` | 指定角度距离度量（常用于余弦相似场景，此外还支持 `euclidean`、`prenormalized-angular`、`dotproduct` 等） |
| `hnsw` | 启用 Hierarchical Navigable Small World 图索引加速近似最近邻搜索 |
| `max-links-per-node: 16` | HNSW 图中每个节点的最大边数（对应论文参数 $M$），增大可提升召回率但会增加内存占用 |
| `neighbors-to-explore-at-insert: 100` | 写入建图时的探索候选宽度（对应 $efConstruction$），增大可提升图质量但会降低写入吞吐 |

需要注意，Schema 中声明的维度 `x[384]` 是强约束：无论是写入文档时的字段数组，还是查询时传入的 query vector，维度都必须严格等于 384，少一维、多一维或格式错误都会直接触发请求失败。

### 6.2 `nearestNeighbor` 近邻查询与三处名称对齐

我们先用 Python 生成一个与 `seed=2`（即 `hybrid-search` 文档）相近的 384 维查询向量，保存到 `query-vector.txt`：

```bash
python3 - <<'PY' > query-vector.txt
import math
values = [round(math.sin(2 * 0.17 + i * 0.013), 6) for i in range(384)]
print("[" + ",".join(map(str, values)) + "]")
PY
```

随后使用 YQL 的 `nearestNeighbor(field, query_tensor)` 算子发起纯向量近邻检索，并使用 `semantic` 排序配置以及 `compact` 摘要格式：

```bash
vespa query \
  'yql=select * from doc where {targetHits:10}nearestNeighbor(embedding, q)' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic' \
  'presentation.summary=compact' \
  'hits=5'
```

在编写向量查询时，新手最容易踩坑的地方是**三处张量名称必须严格对齐**：

| 声明位置 | 配置写法 | 作用说明 |
| --- | --- | --- |
| YQL 查询条件 | `nearestNeighbor(embedding, q)` | 指定用文档字段 `embedding` 与查询输入张量 `q` 计算近邻 |
| HTTP / CLI 查询参数 | `input.query(q)=[...]` | 在请求时传入名为 `q` 的实际向量数值 |
| Schema `rank-profile` | `inputs { query(q) tensor<float>(x[384]) }` | 声明排序配置接收名为 `query(q)` 的 384 维浮点张量输入 |

此外，请特别区分查询中的两个数量参数：`{targetHits:10}` 是传给 `nearestNeighbor` 算子的**向量召回候选预算**（即让底层 HNSW 索引为每个内容节点找出至少 10 个最近邻候选交给后续排序阶段），而 `'hits=5'`（或 YQL 中的 `limit 5`）才是**最终返回给客户端的结果条数**。在结合结构化过滤进行向量检索时，也只需在 YQL 中用 `and` 连接条件，Vespa 会自动在满足过滤条件的文档集合上执行近邻搜索：

```bash
vespa query \
  'yql=select * from doc where tags contains "rag" and ({targetHits:20}nearestNeighbor(embedding, q))' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic' \
  'hits=5'
```

### 6.3 Hybrid Search：合并词面与语义召回

混合检索（Hybrid Search）的核心工程思想是：不要把召回质量押注在单一信号上。BM25 擅长捕捉精确术语、代码符号与字面重合，而向量检索擅长捕捉同义改写、跨词面意图与自然语言问答。在 Vespa 中，实现混合检索只需在 YQL 中用 `or` 合并两路候选集，并交由统一的 `hybrid` 排序配置打分：

```bash
vespa query \
  'yql=select id,title,url from doc where userQuery() or ({targetHits:20}nearestNeighbor(embedding, q))' \
  'query=hybrid search tutorial' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=hybrid' \
  'ranking.listFeatures=true' \
  'hits=5'
```

对应的 `hybrid` 排序配置继承自 `semantic`，在 `first-phase` 中将标题/正文的 BM25 得分与向量 `closeness` 线性组合：

```text
rank-profile hybrid inherits semantic {
  first-phase {
    expression: bm25(title) + bm25(body) + 0.5 * closeness(field, embedding)
  }
  match-features {
    bm25(title)
    bm25(body)
    closeness(field, embedding)
    distance(field, embedding)
  }
}
```

这里直接相加的线性表达式是为了直观展示机制，但在生产环境中需要格外谨慎：无界且随文档长度波动的 BM25 分数与分布在 `(0, 1]` 区间的 `closeness` 往往处于不同数值尺度，直接相加容易导致某一路信号完全支配最终排序。工程演进时建议遵循以下五步法：

1. **单路验证 BM25**：只跑文本检索，确认分词、`fieldset` 和词面相关性正常。
2. **单路验证 Semantic**：只跑 `nearestNeighbor`，确认向量维度、距离度量和语义召回正常。
3. **合并候选集**：用 `userQuery() or ({targetHits:K}nearestNeighbor(...))` 放宽召回池。
4. **观测特征分布**：开启 `match-features`，统计真实查询集上 BM25 与 `closeness` 的分值区间。
5. **精细化重排**：引入分值归一化、倒数排名融合（RRF）或多阶段排序（second-phase / global-phase）。

## 7. Ranking Profile 进阶、调试与性能质量调优

### 7.1 多阶段排序（First-Phase / Second-Phase）与业务特征组合

当候选集规模较大或排序模型较复杂时，不可能对所有召回文档都执行昂贵的深度打分。Vespa 的 `rank-profile` 原生支持多阶段级联排序架构，并在表达能力上支持自定义函数（`function`）、业务属性计算与多阶段重排：

- **`first-phase`（第一阶段粗排）**：在 content node 上对所有匹配 YQL 条件的候选文档逐一执行，要求表达式轻量高效（如 BM25、基础向量 `closeness`、简单时间衰减）。
- **`second-phase`（第二阶段精排）**：同样在各个 content node 本地执行，但仅对 `first-phase` 得分最高的 Top-N 文档（由 `rerank-count` 控制）运行更复杂的特征交叉、GBDT 树模型或轻量神经网络。
- **`global-phase`（全局重排）**：在 container node 合并完所有分片返回的优选结果后，对全局 Top-K 候选执行最终重排、跨分片归一化或交叉编码器（Cross-Encoder）打分。

下面的扩展示例展示了如何在 `rank-profile` 中编写自定义函数计算文档新鲜度加成（`freshness_boost`），并结合 `first-phase` 与 `second-phase` 构建两阶段排序及调试特征输出：

```text
rank-profile hybrid-multiphase inherits semantic {
  function age_days() {
    expression: max(0, (now() - attribute(published_at)) / 86400)
  }

  function freshness_boost() {
    expression: if(age_days < 30, 0.2, 0.0)
  }

  function lexical_score() {
    expression: bm25(title) + bm25(body)
  }

  first-phase {
    expression: lexical_score + 0.5 * closeness(field, embedding)
  }

  second-phase {
    rerank-count: 100
    expression: firstPhase + freshness_boost
  }

  match-features {
    bm25(title)
    bm25(body)
    closeness(field, embedding)
    age_days
    freshness_boost
    firstPhase
  }
}
```

排序逻辑越复杂，越要在 `match-features` 中保留中间函数值，否则当线上排序结果发生漂移时将难以定位究竟是词面分、向量分还是业务加权项导致的变化。

### 7.2 响应结构、Document Summary 与 Trace 调试

理解 Vespa 查询响应 JSON 的核心字段是日常排障的基础：

| 响应 JSON 路径 | 字段含义与调试价值 |
| --- | --- |
| `root.children[]` | 最终返回的命中文档数组，按 `relevance` 从高到低排列 |
| `children[].id` / `relevance` | 命中文档的完整 Vespa Document ID 及其最终排序得分 |
| `children[].fields` | 由当前 `document-summary` 决定的返回字段集合（开启调试时含 `matchfeatures`） |
| `root.fields.totalCount` | 匹配查询条件的文档总数（取决于查询形态，可能是精确值或估计值） |
| `root.coverage` | 查询覆盖率信息，用于排查是否有分片超时、降级或数据未完全就绪 |
| `timing` / `trace` | 开启调试参数后返回的各阶段耗时明细与内部执行计划 |

当你需要深入查看查询在 container 与 content 节点之间的解析、分发与耗时细节时，可以在查询中附加 `tracelevel=2`（注意输出信息量较大，仅用于开发排障，切勿在生产流量上默认开启）：

```bash
vespa query \
  'yql=select * from doc where userQuery() limit 3' \
  'query=vespa schema' \
  'tracelevel=2'
```

面对任何一条结果不符合预期的复杂查询，只要依次追问三个问题，Vespa 就永远不会变成黑盒：

1. **候选集从哪里来？** 是通过 `userQuery()`、`nearestNeighbor`、结构化过滤，还是它们的 `and`/`or` 组合召回的？
2. **排序由谁控制？** 请求实际命中的是哪个 `rank-profile`，其 `first-phase` / `second-phase` 表达式是什么？
3. **得分如何构成？** 打开 `ranking.listFeatures=true` 后，每条结果的 `match-features` 能否合理解释当前名次？

### 7.3 性能与质量调优的第一层旋钮

跑通基本功能后，工程下一步通常是权衡检索质量与服务延迟。在引入复杂深度学习重排模型之前，建议优先掌握以下第一层基础旋钮：

| 调优旋钮 | 直接影响 | 常见工程取舍 |
| --- | --- | --- |
| `hits` / `limit` | 最终返回给客户端的文档数量 | 数量越大，摘要填充与网络序列化响应体开销越大 |
| `targetHits` / `totalTargetHits` | `nearestNeighbor` 阶段的向量候选预算 | 预算越高，长尾召回率越好，但距离计算与粗排延迟随之上升 |
| HNSW `max-links-per-node` | 图索引每个节点的连接度 | 调高可改善高召回区间的精度，但会显著增加常驻内存占用 |
| HNSW `neighbors-to-explore-at-insert` | 写入时搜索近邻的深度 | 调高可提升图索引质量，但会降低文档写入（feed）吞吐 |
| `document-summary` 字段体积 | 摘要阶段读取磁盘/内存与传输的 payload 大小 | 列表页检索应使用轻量 summary，避免无意义返回大段正文或原始向量 |
| `rank-profile` 表达式与 `rerank-count` | 粗排与精排阶段的 CPU 计算量 | 将昂贵算子下沉到 `second-phase` 并限制 `rerank-count` |
| `match-features` 数量 | 附加特征的打包与传输成本 | 评估与排障阶段极有价值，但线上高并发正常请求不应默认全开 |
| 结构化 Filter 选择性 | 参与向量/文本匹配的文档子集范围 | 过滤条件过宽会增加计算量，过严则可能导致召回候选不足 |

在做质量调优时，切忌只盯着某一条 demo query 调权重，因为混合检索非常容易出现“改好一个 case 却让整体退化”的现象。建议先构建一个包含 50～200 条真实查询的离线回归集（人工标注或至少审查 Top-10），每次修改 `rank-profile` 后对比哪些查询变好、哪些变差、BM25 与 `closeness` 的分值范围是否失衡。在做性能调优时，则重点观测三条曲线：`targetHits -> P99 延迟`、`targetHits -> 离线相关性指标（如 NDCG@10）` 以及 `summary 大小 -> 响应延迟`。例如，当 `targetHits` 从 20 增加到 200 时若相关性几乎不再提升而延迟翻倍，就应果断收紧候选预算。

## 8. 从教程示例走向 RAG 与推荐系统后端

### 8.1 RAG 检索后端定位与生产级 Chunk Schema 扩展

在 RAG（Retrieval-Augmented Generation）架构中，Vespa 通常不直接负责编排 Prompt 或调用大语言模型，而是作为核心的 **Retrieval & Ranking Backend**。一条典型的生产级 RAG 链路如下：

```text
用户提问 (User Question)
  -> 查询改写 / 意图分类 / 生成 Query Embedding
  -> Vespa 混合检索 (BM25 + nearestNeighbor + 租户/权限/语言/时间过滤)
  -> Vespa 多阶段排序 (融合词面匹配、语义相似度、文档权威度与时效性)
  -> 按 citation summary 返回 Top-K 证据片段及来源元数据
  -> 上游应用或编排框架组织 Prompt 调用 LLM 生成带引用的回答
```

将 Vespa 与外部 AI 组件放在一起看，职责边界非常清晰：Embedding 服务负责将文档切块和查询转化为向量；LangChain / LlamaIndex 或自研网关负责 Agent 流程、工具调用与 Prompt 拼接；LLM 服务负责最终文本生成；而 **Vespa 专门负责在海量文档分片（chunks）上安全、快速、可解释地完成多路召回、权限过滤、混合重排与证据返回**。

本文前面的 `doc` schema 为了降低入门门槛只定义了少量字段。在真实的企业级 RAG 系统中，检索粒度通常是切分后的文档块（chunk），需要携带多租户隔离、ACL 权限、章节上下文与引用溯源信息。我们可以将教程示例平滑演进为如下生产级 `schemas/chunk.sd`：

```text
schema chunk {
  document chunk {
    field chunk_id type string {
      indexing: attribute | summary
      match: word
    }

    field doc_id type string {
      indexing: attribute | summary
      match: word
    }

    field source_uri type string {
      indexing: attribute | summary
      match: word
    }

    field title type string {
      indexing: index | summary
      match: text
      index: enable-bm25
    }

    field section type string {
      indexing: index | summary
      match: text
      index: enable-bm25
    }

    field text type string {
      indexing: index | summary
      match: text
      index: enable-bm25
    }

    field tenant_id type string {
      indexing: attribute | summary
      match: word
      attribute: fast-search
    }

    field language type string {
      indexing: attribute | summary
      match: word
      attribute: fast-search
    }

    field updated_at type long {
      indexing: attribute | summary
      attribute: fast-search
    }

    field embedding type tensor<float>(x[384]) {
      indexing: attribute | index
      attribute {
        distance-metric: angular
      }
      index {
        hnsw {
          max-links-per-node: 16
          neighbors-to-explore-at-insert: 100
        }
      }
    }
  }

  fieldset default {
    fields: title, section, text
  }

  document-summary citation {
    summary chunk_id {}
    summary doc_id {}
    summary source_uri {}
    summary title {}
    summary section {}
    summary text {}
    summary updated_at {}
  }

  rank-profile rag-hybrid {
    inputs {
      query(q) tensor<float>(x[384])
    }
    first-phase {
      expression: bm25(title) + bm25(section) + bm25(text) + closeness(field, embedding)
    }
    match-features {
      bm25(title)
      bm25(section)
      bm25(text)
      closeness(field, embedding)
      distance(field, embedding)
    }
  }
}
```

基于这份 `chunk` schema，业务后端可以在单次查询中同时完成租户隔离（`tenant_id`）、语种过滤（`language`）、三字段 BM25 检索、向量近邻召回以及面向 LLM 上下文窗口裁剪的 `citation` 摘要返回：

```bash
vespa query \
  'yql=select * from chunk where tenant_id contains "acme" and language contains "zh" and (userQuery() or ({targetHits:50}nearestNeighbor(embedding, q)))' \
  'query=Vespa 如何配置 rank profile' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=rag-hybrid' \
  'presentation.summary=citation' \
  'hits=8'
```

当然，也要清醒认识工程边界：RAG 的最终回答质量并非只由 Vespa 决定，文档解析与 chunk 切分策略、Embedding 模型表征能力、Query 改写、Prompt 约束以及评估集建设同样缺一不可。Vespa 解决的是检索与排序层的工程确定性。

### 8.2 推荐系统在线召回与排序落点

除了搜索与 RAG，Vespa 也是构建实时推荐与个性化系统的常见底座。推荐系统中的核心概念可以自然映射到我们已经学过的 Vespa 原语上：

| 推荐系统核心概念 | Vespa 中的对应实现落点 |
| --- | --- |
| 物品静态与实时特征（Item Features） | Document fields、`attribute` 内存属性（支持高频局部 `update`）与 tensors |
| 物品双塔向量（Item Embedding） | Document 中的 `tensor` 字段 + HNSW 索引 |
| 用户实时向量与上下文（User Embedding / Context） | 查询请求传入的 `input.query(user_emb)` 等输入张量 |
| 多路候选召回（Candidate Generation） | `nearestNeighbor` 向量召回 + 文本标签召回 + 库存/地域/已读过滤 |
| 在线粗排与精排策略（Ranking Policy） | `rank-profile` 的 `first-phase`、`second-phase` 与 ONNX / GBDT 模型推理 |
| 业务规则与多样性打散 | 排序表达式提权/降权、结构化过滤与 Grouping 分组采样 |
| 特征归因与线上调试 | `match-features` 与 `rank-features` |

同样需要明确的是，Vespa 作为在线 serving 引擎，并不会自动替你完成推荐系统的离线数据闭环：曝光与点击日志采集、正负样本拼接、双塔与排序模型训练、特征一致性校验以及 A/B 实验平台，仍然需要配套的数据与机器学习工程体系支撑。

## 9. 部署形态选择、故障排查与完整命令回放

### 9.1 本地 Docker、Vespa Cloud 与自托管选型及生产化路线

当你的原型准备走向团队协作或生产环境时，通常需要在三种运行形态之间做选择：

| 部署形态 | 适用阶段 | 核心优势 | 主要工程代价与风险 |
| --- | --- | --- | --- |
| 本地 Docker 单容器 | 个人学习、Schema 验证、功能原型 | 启动快、零成本、可完全离线试错 | 单节点无高可用，性能与运维方式不代表生产环境 |
| Vespa Cloud（官方托管） | 团队项目、试生产、追求少运维的线上服务 | 官方负责集群扩缩容、安全升级与多区高可用，CLI 体验一致 | 需注册云账号、配置证书权限并承担托管服务成本 |
| 自托管（Kubernetes / 裸金属） | 具备专职平台团队、有严格数据驻留或合规要求的企业 | 基础设施自主可控，可深度定制网络与存储硬件 | 集群编排、容量规划、滚动升级、监控告警与故障恢复均需自建 |

Vespa CLI 在本地与云端之间切换只需修改 `target` 配置：

```bash
# 本地 Docker 模式
vespa config set target local

# Vespa Cloud 托管模式
vespa config set target cloud
vespa auth login
vespa config set application mytenant.myapp.default
```

如果你打算用 1 小时到 1 周的时间系统掌握 Vespa 并推进到准生产状态，推荐遵循以下循序渐进的实验与生产化路线：

- **第 1 小时（跑通闭环）**：启动本地 Docker，完成 `vespa deploy`、`vespa feed`、`where true` 全量查询与 BM25 查询，能清晰解释 `services.xml` 与 `doc.sd` 的职责分工。
- **第 1～2 天（单路能力与特征直觉）**：准备 20 条以上真实查询，结合 `tags`/`published_at` 过滤观察 BM25 Top-10，开启 `match-features` 分析各字段得分；接入真实 Embedding 模型生成 384 或 768 维向量，对比纯 BM25 与纯 `nearestNeighbor` 在精确术语和语义改写查询上的优劣。
- **第 3 天（混合排序实验）**：用 `or` 合并文本与向量召回，编写 `hybrid` 乃至带 `second-phase` 的排序配置，调节权重并人工评估 Top-10 排序变化。
- **第 1 周及生产化准备（工程加固）**：固定 Document ID 命名规范与模型版本号字段；建立带重试与幂等保证的自动 feed 管线；设计轻量 `compact`/`citation` 摘要；构建离线查询评估集；针对 `targetHits`、HNSW 参数与并发 QPS 进行压测；补齐延迟监控（P95/P99）、节点健康检查、安全认证与多节点冗余（`redundancy`）规划。

### 9.2 常见错误清单与统一排查顺序

初学 Vespa 时遇到的绝大多数问题，都可以通过下表快速定位：

| 常见错误类型 | 典型报错或异常现象 | 排查与修复方法 |
| --- | --- | --- |
| 容器未就绪或内存不足 | `vespa status` 失败、部署卡死或容器自动退出 | 执行 `docker ps`、`docker logs vespa`，确保分配给 Docker 的内存 $\ge$ 4GB |
| 端口冲突 | 容器启动失败或 `8080`/`19071` 无法连接 | 检查宿主机端口占用，或清理旧容器 `docker rm -f vespa` |
| 执行目录错误 | `vespa deploy` 提示找不到应用文件 | 确认当前终端工作目录下直接包含 `services.xml` 和 `schemas/` |
| Schema 与服务配置名不一致 | 部署失败或写入/查询报 document type 错误 | 确保 `services.xml` 中 `<document type="doc">` 与 `schemas/doc.sd` 中的 `schema doc` 一致 |
| 查询命中但缺少某字段 | 返回的 `children[].fields` 中看不到目标字段 | 给该字段补上 `indexing: ... \| summary`，并检查是否指定了不含该字段的 `document-summary` |
| BM25 报错或分数恒为 0 | 部署报错或 `bm25(field)` 无效 | 确认目标文本字段配置了 `indexing: index` 且显式声明 `index: enable-bm25` |
| 向量写入或查询报 Tensor 错误 | 提示维度不匹配或格式解析失败 | 核对 schema `x[384]`、feed 文档中的数组长度与 `query-vector.txt` 的元素个数完全一致 |
| `nearestNeighbor` 报缺少输入 | 查询报错 missing query tensor | 严格对齐 YQL `nearestNeighbor(embedding, q)`、参数 `input.query(q)` 与 `inputs { query(q) ... }` |
| 混淆 `targetHits` 与 `hits` | 返回条数与预期不符 | 记住 `targetHits` 是单节点 ANN 召回候选预算，`hits` / `limit` 才是最终返回条数 |
| Hybrid 排序被单路信号垄断 | 排序结果完全等同于纯 BM25 或纯向量 | 开启 `ranking.listFeatures=true` 检查分值尺度，引入权重缩放、归一化或两阶段重排 |

遇到链路不通或检索结果异常时，请严格遵循以下**统一排查顺序**逐层验证，切忌在底层尚未打通时盲目修改上层参数：

```text
1. 容器与状态：docker ps / docker logs vespa -> vespa status 确认控制面与数据面就绪
2. 应用包部署：vespa deploy 确认 services.xml 与 schema 语法校验通过并激活
3. 数据写入验证：vespa feed -> 执行 select * from doc where true 确认 totalCount > 0 且 summary 字段齐全
4. 单路召回验证：先单独跑通 BM25 文本查询，再单独跑通 nearestNeighbor 向量查询
5. 混合与排序调试：合并 or 候选集 -> 指定 ranking profile -> 开启 ranking.listFeatures=true 核对 match-features
```

### 9.3 最小可运行验证、完整命令回放与核心术语速查

完成本教程全部实操后，你的本地环境应满足以下最小可运行验收标准：

- Docker 容器 `vespa` 正常运行，且 `docker exec vespa vespa-logfmt` 无持续报错。
- `vespa deploy` 与 `vespa feed documents.jsonl` 均成功执行，`where true` 查询能返回写入的文档及预期的 summary 字段。
- `ranking=bm25`、`ranking=semantic`（配合 `nearestNeighbor`）以及 `ranking=hybrid` 三种模式均能正常返回结果，且附加 `ranking.listFeatures=true` 时能看到 `bm25(title)`、`bm25(body)`、`closeness(field, embedding)` 等调试特征。

这也构成了 Vespa 日常开发最核心的迭代闭环：`编辑 schema / services.xml -> vespa deploy -> vespa feed -> vespa query -> 检查 match-features -> 调整 rank-profile`。下面将本文从零启动到混合检索的完整命令按顺序汇总，方便一键对照回放（其中生成测试数据的 Python 脚本见第 4.1 节与第 6.2 节）：

```bash
# 1. 启动本地 Vespa 容器
docker pull vespaengine/vespa
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa

# 2. 创建应用目录并编写 services.xml 与 schemas/doc.sd
mkdir -p docs-search/schemas
cd docs-search
# （按第 2.2 节和第 3.1 节写入配置文件，按第 4.1 节生成 documents.jsonl）

# 3. 部署应用并写入数据
vespa config set target local
vespa deploy
vespa feed documents.jsonl

# 4. 验证全量查询与 BM25 文本检索
vespa query 'yql=select * from doc where true limit 3'
vespa query \
  'yql=select id,title,url from doc where userQuery() limit 5' \
  'query=hybrid search tutorial' \
  'ranking=bm25'

# 5. 验证纯向量近邻查询（按第 6.2 节生成 query-vector.txt）
vespa query \
  'yql=select id,title,url from doc where {targetHits:10}nearestNeighbor(embedding, q)' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic' \
  'hits=5'

# 6. 验证 Hybrid Search 与特征调试输出
vespa query \
  'yql=select id,title,url from doc where userQuery() or ({targetHits:20}nearestNeighbor(embedding, q))' \
  'query=hybrid search tutorial' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=hybrid' \
  'ranking.listFeatures=true' \
  'hits=5'
```

为便于日常查阅，这里将 Vespa 最常出现的专有术语浓缩为速查表：

| 术语 | 一句话核心定义 |
| --- | --- |
| Application package | Vespa 应用声明式配置包，至少包含 `services.xml` 和 `schemas/*.sd` |
| `services.xml` | 定义 `container` 与 `content` 集群拓扑、端点开关与节点分布的配置文件 |
| Schema (`*.sd`) | 定义文档结构、字段类型、索引方式、`document-summary` 和 `rank-profile` 的核心文件 |
| Document ID | 全局文档标识，格式为 `id:<namespace>:<document-type>::<user-id>` |
| `index` / `attribute` / `summary` | 字段三大存储属性：分别对应倒排或图索引、内存列式属性（过滤/排序/距离计算）与结果返回摘要 |
| Fieldset | 一组文本字段的逻辑集合，供 `userQuery()` 作为默认全文检索范围 |
| YQL | Vespa 查询语言，决定从哪个 schema 按什么过滤与召回条件选出候选集 |
| `userQuery()` / `nearestNeighbor` | YQL 中的两大核心算子：分别用于解析请求文本查询与执行向量近似最近邻搜索 |
| `targetHits` vs `hits` | `targetHits` 是向量算子在单节点的候选召回预算；`hits` 是最终返回给客户端的结果数 |
| Tensor / HNSW | Tensor 是 Vespa 表达向量与高维张量的类型；HNSW 是构建在 Tensor 上的近似近邻图索引 |
| Rank profile | 排序策略配置，支持 `inputs`、自定义 `function`、`first-phase`、`second-phase` 与 `global-phase` |
| `closeness` / `bm25` | 常用排序特征：分别度量向量近邻相似度与文本词面相关性 |
| `match-features` | 随命中文档一并输出的排序子特征明细，是实现白盒可解释调优的关键工具 |

如果只想记住一个最精炼的公式化心智模型，请记住这四句话：

```text
schema 决定数据怎么被理解与索引
YQL 决定候选怎么被过滤与取出
rank-profile 决定候选怎么分阶段排序
summary 决定结果怎么裁剪与返回
```

## 10. 结论与参考资料

Vespa 入门的核心在于建立清晰的职责分层：`services.xml` 定义无状态容器层与有状态内容层的服务拓扑；Schema 定义文档字段、倒排/HNSW 索引、内存属性、返回摘要与排序模型；Feed 管线将业务数据与向量实时写入文档存储；YQL 负责组合全文、向量与结构化条件召回候选；`rank-profile` 则在靠近数据的节点上完成多阶段打分与特征输出。一旦这套心智模型建立起来，Vespa 就成为一个统一的 AI retrieval serving 平台——你可以从最简单的 BM25 文本搜索起步，平滑叠加向量近邻、元数据过滤、多阶段重排、RAG 证据裁剪与推荐系统在线服务，并且每一步排序结果都能通过 `match-features` 白盒解释。

如果你准备进一步深入阅读官方资料，推荐按以下顺序递进：**Vespa Basics -> News Tutorial -> Hybrid Search Tutorial -> Schema Reference -> YQL Reference -> Ranking & Phased Ranking -> Nearest Neighbor Search Guide -> RAG Documentation -> Vespa Cloud / pyvespa**。先用本文的最小应用在本地跑通完整闭环，再带着具体的业务问题查阅参考手册，学习效率会高得多。

### References

- [Vespa.ai official website](https://vespa.ai/)
- [Vespa AI Search Platform Architecture](https://vespa.ai/architecture/)
- [Vespa Documentation](https://docs.vespa.ai/)
- [Hybrid Text Search Tutorial](https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html)
- [News search and recommendation tutorial](https://docs.vespa.ai/en/learn/tutorials/news-2-basic-feeding-and-query.html)
- [Vespa CLI documentation](https://docs.vespa.ai/en/clients/vespa-cli.html)
- [Approximate nearest neighbor search using HNSW](https://docs.vespa.ai/en/querying/approximate-nn-hnsw.html)
- [Nearest neighbor search guide](https://docs.vespa.ai/en/querying/nearest-neighbor-search-guide.html)
- [Schemas](https://docs.vespa.ai/en/basics/schemas.html)
- [YQL query language](https://docs.vespa.ai/en/query-language.html)
- [Ranking introduction](https://docs.vespa.ai/en/ranking.html)

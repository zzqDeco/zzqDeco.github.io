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

Vespa 这个名字容易让人先想到机车，但本文讲的是 **Vespa.ai / Vespa Engine**：一个开源的大规模数据 serving engine，用来构建搜索、推荐、个性化、向量检索、RAG 和机器学习排序系统。

如果只把 Vespa 叫作“向量数据库”，会漏掉它最重要的部分。Vespa 可以存向量、做 ANN，但它真正的定位更像一个统一的 AI retrieval serving platform：同一个系统里同时处理全文检索、结构化过滤、向量近邻、tensor 计算、多阶段排序、模型推理、实时更新和分布式低延迟服务。官方首页对它的描述很直接：Vespa 可以在 vectors、tensors、text 和 structured data 上查询、组织和推理，并服务搜索、RAG、推荐和半结构化导航等场景。

这篇文章不是 Vespa 全量手册，而是一篇从零上手教程。我们会用一个统一的 `docs-search` 示例，从本地 Docker 启动开始，写 application package，定义 schema，feed 文档，跑 YQL 查询，加入 BM25，加入向量字段，再做 hybrid search 和 rank-profile 调试。读完后你应该能回答三个问题：

1. Vespa 的 application package、schema、services.xml、YQL 和 rank-profile 分别负责什么。
2. 一个最小本地 Vespa 搜索应用怎样部署、写入、查询和调试。
3. Vespa 的 BM25、向量近邻、混合检索和 RAG 后端角色应该怎么理解。

本文默认你有 Docker 或 Podman 基础，能在命令行里执行 `docker`、`curl`、`python3` 和 Vespa CLI。文章不新增仓库内 demo 目录，所有代码以教程代码块给出；真正落地时可以把这些文件复制到自己的实验目录。

## 1. 什么时候应该学 Vespa

Vespa 不是最轻的工具。如果你的需求只是存 1 万条向量、做一个个人 demo，Milvus、Qdrant、Chroma、SQLite 向量扩展，甚至直接 FAISS，都可能更快。但如果你的需求开始接近下面这些形态，Vespa 就值得认真看：

| 需求 | 为什么 Vespa 合适 |
| --- | --- |
| 搜索既要关键词匹配，又要语义向量召回 | Vespa 同时支持 lexical search、vector search 和同一 ranking pipeline |
| 检索结果需要复杂排序 | rank-profile 可以组合 BM25、向量相似、业务字段、模型特征和 tensor 表达式 |
| 数据经常变化 | Vespa 强调 real-time updates，不是离线建好索引后一周不动 |
| RAG 需要过滤、分片、排序、证据返回 | 文档字段、structured filter、nearestNeighbor 和 summary 可以放在同一查询里 |
| 推荐系统需要候选召回和在线排序 | Vespa 可以做向量召回、实时特征访问和多阶段 ranking |
| 希望减少“搜索引擎 + 向量库 + reranker 服务 + 特征服务”的拼接成本 | Vespa 的设计目标是尽量让 retrieval、ranking、inference 靠近数据执行 |

本文把 Vespa 当成一个搜索/RAG/推荐基础设施来讲。它不是 prompt orchestration 框架，也不是训练框架；它更像线上 serving 层。你可以把 embedding 生成、LLM 调用、业务 API、前端 UI 放在其他服务中，Vespa 负责把候选文档找出来、排好、返回必要字段。

## 2. Vespa 是什么：不是“另一个向量库”

先看官方架构图。

![Vespa distributed architecture](https://vespa.ai/vespa-content/uploads/2026/06/Vespa-Distributed-Architecture-1.png)

*Source: Vespa.ai Architecture, official website.*

这张图把 Vespa 放在完整 AI retrieval workflow 中：Retrieve、Rank、Infer、Update、Serve。它的核心技术包括 unified data & compute、tensor engine、query execution 和 distributed architecture。官方架构页的关键观点是：不要把检索、排序、模型推理和 serving 拆成多个彼此搬运数据的系统，而是尽量在统一分布式架构中、靠近数据执行这些步骤。

这就是 Vespa 和普通向量数据库的主要差别。

普通向量库通常回答的是：

```text
给我一个 query vector，返回最相似的 topK vectors。
```

Vespa 更常回答的是：

```text
给我一个 query、query vector、用户上下文和过滤条件；
先用关键词、向量和结构化条件取候选；
再用 rank-profile 组合 BM25、closeness、业务特征、模型输出；
最后返回某个 summary 下的结果和调试特征。
```

它不是“只有向量检索”，也不是“只有搜索引擎”。它的入门难点也因此不是某条命令，而是心智模型。下面我们先把这个心智模型建立起来。

## 3. 核心心智模型

Vespa 入门需要先理解五个词：

| 概念 | 作用 | 类比 |
| --- | --- | --- |
| application package | 定义一个 Vespa 应用的配置包 | 搜索服务的配置工程 |
| schema | 定义文档类型、字段、索引、summary、rank profiles | 数据表结构 + 检索/排序配置 |
| document | 被写入 Vespa 的数据项 | 搜索文档、商品、文章、chunk |
| container cluster | 无状态请求处理层 | query API / document API / custom components |
| content cluster | 有状态数据存储和检索层 | shard / index / document store |

一个入门查询链路可以画成：

```text
client
  |
  |  HTTP / Vespa CLI query
  v
container cluster
  |
  |  parse YQL, prepare query, fan out
  v
content cluster
  |
  |  match candidates by text/vector/filter
  |  compute first-phase ranking close to data
  v
container cluster
  |
  |  merge hits, fill summaries, render response
  v
client
```

官方文档里的 Vespa overview 图也能帮助建立这个模型：

![Vespa overview](https://docs.vespa.ai/assets/img/vespa-overview.svg)

*Source: Vespa Documentation, official docs.*

图里最值得注意的不是组件数量，而是职责边界：容器层负责 query/document API 和请求处理，内容层负责文档存储、索引、匹配和 ranking 计算。很多系统会把“向量库”“全文检索”“特征服务”“reranker 服务”拆成多个独立组件，再在业务层合并结果；Vespa 的默认思路是把这些能力尽量收进一个 application package 和查询执行流程里。

写入链路则是：

```text
client / feed pipeline
  |
  |  document JSON
  v
container document-api
  |
  |  route document operation
  v
content cluster
  |
  |  store fields, update attributes, build indexes
  v
query becomes visible
```

这两个链路都由 application package 控制。`services.xml` 决定有哪些 cluster，schema 决定文档如何存、如何索引、如何返回、如何排序。

## 4. 本地环境准备

入门阶段使用本地 Docker 最直接。你需要：

- Docker Desktop 或 Podman，容器运行中。
- 至少给 Docker/Podman 分配 4GB 内存。官方 hybrid tutorial 也把 4GB dedicated memory 当作最低提醒。
- Vespa CLI。官方文档提到可以用 Homebrew 或 GitHub releases 安装。
- `curl` 和 `python3`，用于验证 endpoint 和生成示例数据。

本教程使用两个端口：

| 端口 | 用途 |
| --- | --- |
| `8080` | 查询、写入、document API、HTTP endpoint |
| `19071` | config server / deploy endpoint |

启动本地 Vespa 容器：

```bash
docker pull vespaengine/vespa

docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
```

等待容器启动后，可以看日志：

```bash
docker logs -f vespa
```

也可以用 Vespa 自带日志格式化工具：

```bash
docker exec vespa vespa-logfmt
```

配置 Vespa CLI 指向本地：

```bash
vespa config set target local
```

验证本地服务：

```bash
vespa status
```

如果 `vespa status` 一直失败，先不要写 schema。优先确认：

- Docker 容器是否运行：`docker ps`
- 端口是否映射：`8080` 和 `19071`
- Docker 内存是否足够
- 本机是否已有同名容器：`docker rm -f vespa`

## 5. 创建第一个 application package

我们用一个叫 `docs-search` 的示例应用。它搜索技术文档片段，支持全文检索、标签过滤、向量召回和 hybrid ranking。

目录结构：

```text
docs-search/
  services.xml
  schemas/
    doc.sd
```

入门阶段只需要两个文件。

`services.xml` 描述 Vespa 服务拓扑：container 和 content cluster。

`schemas/doc.sd` 描述文档类型 `doc`：字段、索引、summary、rank profiles。

这两个文件就是最小 application package。你可以把它理解为“搜索服务的声明式配置”。它不是你的业务后端代码，也不是前端代码。

创建目录：

```bash
mkdir -p docs-search/schemas
cd docs-search
```

后面所有 `vespa deploy` 默认在 `docs-search/` 目录下执行。

## 6. `services.xml` 精读

写入 `docs-search/services.xml`：

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

逐块解释：

| 配置 | 含义 |
| --- | --- |
| `<container>` | 无状态请求处理层 |
| `<search />` | 启用查询 endpoint |
| `<document-api />` | 启用写入/访问 document endpoint |
| `<content>` | 有状态内容集群，负责存储、索引、匹配 |
| `<document type="doc" mode="index" />` | 指定 content cluster 存储 `doc` 类型，并使用 index mode |
| `<redundancy>1</redundancy>` | 本地单节点无需副本，生产环境另行规划 |
| `<node hostalias="node1">` | 本地单节点 host alias |

入门阶段只要理解：container 接请求，content 存数据和执行检索。生产环境可以有多个 container node、多个 content node、多副本和多集群，但不要一开始就复杂化。

## 7. Schema 入门：字段、index、attribute、summary

写入 `docs-search/schemas/doc.sd`：

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

这份 schema 覆盖了入门最常见的字段类型。

| 字段 | 作用 | 关键配置 |
| --- | --- | --- |
| `id` | 业务文档 ID | `attribute | summary` |
| `title` | 标题全文检索 | `index | summary`, `enable-bm25` |
| `body` | 正文全文检索 | `index | summary`, `enable-bm25` |
| `url` | 返回链接和精确字段 | `attribute | summary` |
| `tags` | 结构化标签 | `attribute | summary` |
| `published_at` | 排序/过滤时间戳 | `attribute: fast-search` |
| `embedding` | 语义向量 | `tensor<float>(x[384])`, `attribute | index`, HNSW |

这里最容易混的是 `index`、`attribute`、`summary`。

**`index`** 表示字段进入倒排索引，用于全文检索、BM25、文本 matching。`title` 和 `body` 需要 `index`。

**`attribute`** 表示字段作为 attribute 存储，适合过滤、排序、实时访问和向量检索。`published_at`、`tags` 和 `embedding` 需要 attribute。

**`summary`** 表示字段可以在结果里返回。如果字段没有 summary，查询命中了也可能不在返回结果里出现。

**`index: enable-bm25`** 是 BM25 的前提。你在 rank-profile 里写 `bm25(title)`，但 `title` 没启用 BM25，部署或查询会出问题。

**`fieldset default`** 控制 `userQuery()` 默认搜索哪些文本字段。这里默认搜索 `title` 和 `body`。

**`rank-profile`** 控制排序。本文先用三个 profile：

- `bm25`：词面相关性。
- `semantic`：向量相似性。
- `hybrid`：BM25 + 向量相似。

## 8. 准备 feed JSON

Vespa CLI 可以读取 JSONL。每一行是一个 document operation。最常见的是 `put`。

为了让 384 维向量示例可运行，我们用 Python 生成一个 `documents.jsonl`。向量只是固定 pattern 的 toy embedding，不代表真实语义质量；真实系统应该用 embedding 模型生成。

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

检查一行：

```bash
head -n 1 documents.jsonl
```

Vespa document id 形如：

```text
id:<namespace>:<document-type>::<user-specific-id>
```

本文使用：

```text
id:docs-search:doc::hybrid-search
```

其中 `docs-search` 是 namespace，`doc` 是 schema/document type，最后是业务 id。

## 9. 第一次部署和查询

确认在 `docs-search/` 目录下：

```bash
pwd
ls
```

应该看到：

```text
services.xml
schemas/
documents.jsonl
```

部署应用：

```bash
vespa config set target local
vespa deploy
```

写入文档：

```bash
vespa feed documents.jsonl
```

最简单查询：

```bash
vespa query 'yql=select * from doc where true limit 3'
```

如果结果里出现 `root.children`，说明基本链路已经通了。

一个典型返回结构大致是：

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

入门阶段先不要急着调 ranking。先确认四件事：

1. `vespa deploy` 成功。
2. `vespa feed` 成功。
3. `where true` 能查到文档。
4. 返回结果里有你想看的 fields。

如果第四点失败，通常是字段没有 `summary`，或者使用了不包含该字段的 `document-summary`。

## 10. YQL 入门

Vespa 查询常用 YQL。最小结构是：

```text
select <fields> from <schema> where <condition> limit <n>
```

返回所有字段：

```bash
vespa query 'yql=select * from doc where true limit 2'
```

只返回部分字段：

```bash
vespa query 'yql=select id,title,url from doc where true limit 2'
```

用 `userQuery()` 做文本查询：

```bash
vespa query \
  'yql=select id,title,url from doc where userQuery() limit 5' \
  'query=hybrid search'
```

用 `contains` 查询某个字段：

```bash
vespa query \
  'yql=select id,title,url from doc where title contains "Vespa" limit 5'
```

用结构化字段过滤：

```bash
vespa query \
  'yql=select id,title,tags from doc where tags contains "rag" limit 5'
```

用时间过滤：

```bash
vespa query \
  'yql=select id,title,published_at from doc where published_at > 1760000100 limit 5'
```

这里有一个重要边界：**YQL 决定候选怎么被取出来，rank-profile 决定候选怎么排序**。当然 YQL 里也可以写 order/sort，但不要把全部 relevance 逻辑散落在业务查询字符串里。入门时尽量让检索条件在 YQL 中，排序逻辑在 rank-profile 中。

## 11. BM25 文本搜索

我们的 schema 已经给 `title` 和 `body` 开了 BM25：

```text
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
```

也定义了 `bm25` rank-profile：

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

查询：

```bash
vespa query \
  'yql=select id,title,url from doc where userQuery() limit 5' \
  'query=hybrid search tutorial' \
  'ranking=bm25'
```

带调试特征：

```bash
vespa query \
  'yql=select id,title,url from doc where userQuery() limit 5' \
  'query=hybrid search tutorial' \
  'ranking=bm25' \
  'ranking.listFeatures=true'
```

这时可以看到 BM25 相关特征。调试 ranking 时，`match-features` 很重要，因为它让你知道每个结果为什么得分高。

BM25 的优点是词面匹配强、可解释、稳定。缺点是不理解语义。如果用户搜 “semantic retrieval”，文档里只写 “vector nearest neighbor”，BM25 未必能匹配好。这就是要引入向量检索的原因。

## 12. 结构化过滤和排序

搜索系统很少只有文本。真实业务经常需要：

- 只搜索某个 category。
- 只返回某些 tags。
- 过滤已下架商品。
- 按发布时间、价格、库存、权限过滤。
- 给新内容、热门内容、业务权重加分。

Vespa 里这类字段通常用 attribute。本文例子里：

```text
field tags type array<string> {
  indexing: attribute | summary
}

field published_at type long {
  indexing: attribute | summary
  attribute: fast-search
}
```

查询 `tags`：

```bash
vespa query \
  'yql=select id,title,tags from doc where tags contains "vector-search" limit 5'
```

按时间过滤：

```bash
vespa query \
  'yql=select id,title,published_at from doc where published_at > 1760000000 limit 10'
```

需要注意：不是所有字段都应该加 `attribute: fast-search`。fast-search 会为 attribute 建额外结构，适合经常用于过滤的字段。对偶尔返回、不用于过滤排序的字段，只做 `summary` 或普通 attribute 就够了。

入门原则：

| 字段用途 | 推荐配置 |
| --- | --- |
| 全文检索 | `index | summary`, `match: text`, 需要 BM25 时 `index: enable-bm25` |
| 精确过滤 | `attribute`, 高频过滤可考虑 `attribute: fast-search` |
| 结果展示 | `summary` |
| 排序特征 | `attribute` |
| 向量检索 | tensor field + `attribute | index` + distance metric |

## 13. 向量字段与 tensor

Vespa 的向量字段用 tensor 表达。本文 schema 中：

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

逐项解释：

| 配置 | 含义 |
| --- | --- |
| `tensor<float>(x[384])` | 384 维 dense float vector |
| `attribute` | 向量作为 attribute 存储，可用于距离计算 |
| `index` | 建向量索引，用于 approximate nearest neighbor |
| `distance-metric: angular` | 使用 angular distance |
| `hnsw` | 使用 HNSW 近似近邻索引 |
| `max-links-per-node` | HNSW 图中每个节点链接数，影响召回/内存 |
| `neighbors-to-explore-at-insert` | 插入时探索邻居数，影响建图质量/写入成本 |

官方文档也给出类似模式：tensor field 使用 `attribute | index`，attribute 中配置 distance metric，index 中配置 HNSW 参数。入门阶段不要过度调 HNSW。先用默认/示例参数跑通，再根据数据规模、延迟、召回率调参。

向量维度必须匹配。schema 是 `x[384]`，feed 文档和 query vector 都必须是 384 维。少一维、多一维、字符串格式不对，都会失败。

## 14. nearestNeighbor 查询

先生成一个 384 维 query vector。这里仍然用 toy vector：

```bash
python3 - <<'PY' > query-vector.txt
import math
values = [round(math.sin(2 * 0.17 + i * 0.013), 6) for i in range(384)]
print("[" + ",".join(map(str, values)) + "]")
PY
```

执行向量查询：

```bash
vespa query \
  'yql=select id,title,url from doc where {targetHits:10}nearestNeighbor(embedding, q)' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic' \
  'hits=5'
```

对应 rank-profile：

```text
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
```

这里有三个名字必须对齐：

| 位置 | 名称 |
| --- | --- |
| YQL | `nearestNeighbor(embedding, q)` |
| query 参数 | `input.query(q)=...` |
| rank-profile inputs | `query(q) tensor<float>(x[384])` |

`embedding` 是文档字段，`q` 是 query tensor 名称。很多入门错误来自这三个地方没对齐。

`targetHits` 表示 nearestNeighbor 期望产生多少候选。它不是最终返回数量。最终返回数量由 `hits` 或 YQL limit 等控制。可以粗略理解：

```text
targetHits: vector retrieval candidate budget
hits: final returned hits
```

真实系统里，向量查询的 query vector 应该来自同一个 embedding 模型。本文 toy vector 只能验证链路，不代表语义检索质量。

## 15. Hybrid Search：BM25 + Vector

混合检索的基本思想是：不要赌一种召回方式。

BM25 擅长：

- 精确词命中。
- 术语、代码、产品名、人名、缩写。
- 用户 query 和文档字段词面重合较强的场景。

向量检索擅长：

- 同义表达。
- query 和文档词面不同但语义接近。
- RAG chunk、问答、自然语言描述。

Hybrid search 通常先合并两路候选，再统一排序。YQL 可以这样写：

```bash
vespa query \
  'yql=select id,title,url from doc where userQuery() or ({targetHits:20}nearestNeighbor(embedding, q))' \
  'query=hybrid search tutorial' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=hybrid' \
  'hits=5'
```

对应 rank-profile：

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

这个入门版直接相加，足够说明机制，但生产上要谨慎。BM25 和 closeness 的尺度不一定匹配。官方 hybrid tutorial 也展示了更复杂的策略，例如归一化、RRF、global-phase reranking 等。入门时建议按这个顺序演进：

1. 只跑 BM25，确认文本检索可靠。
2. 只跑 semantic，确认向量链路可靠。
3. 用 `or` 合并候选。
4. 用 match-features 看每个结果的 BM25 和 closeness。
5. 再调权重、归一化、RRF 或二阶段排序。

不要一上来就把所有东西写成一个大 rank expression。先让每一路单独可解释，再合并。

## 16. Ranking Profile 深入一点点

Vespa 的 rank-profile 是入门到进阶的分水岭。简单查询可以只用默认排序；真正做搜索质量，就需要控制 rank-profile。

最简单 rank-profile：

```text
rank-profile bm25 {
  first-phase {
    expression: bm25(title) + bm25(body)
  }
}
```

带调试特征：

```text
rank-profile bm25 {
  first-phase {
    expression: bm25(title) + bm25(body)
  }
  match-features {
    bm25(title)
    bm25(body)
  }
}
```

带 query input：

```text
rank-profile semantic {
  inputs {
    query(q) tensor<float>(x[384])
  }
  first-phase {
    expression: closeness(field, embedding)
  }
}
```

带业务特征：

```text
rank-profile freshness inherits bm25 {
  function age_days() {
    expression: (now() - attribute(published_at)) / 86400
  }
  function freshness_boost() {
    expression: if(age_days < 30, 0.2, 0.0)
  }
  first-phase {
    expression: bm25(title) + bm25(body) + freshness_boost
  }
  match-features {
    bm25(title)
    bm25(body)
    age_days
    freshness_boost
  }
}
```

上面这段只是说明 rank expression 可以组合业务特征；实际是否要加 freshness，需要看业务目标。搜索系统里排序逻辑越强，越要保留调试特征，否则线上效果变化很难解释。

Vespa 还支持 second-phase、global-phase、模型文件、tensor 表达式等。入门阶段先掌握 first-phase 和 match-features 就够了。

## 17. 返回结果和 Debug

Vespa 查询返回 JSON。最常看的字段：

| 字段 | 含义 |
| --- | --- |
| `root.children` | 命中的结果列表 |
| `children[].id` | Vespa document id |
| `children[].relevance` | 排序得分 |
| `children[].fields` | 返回 summary fields |
| `root.fields.totalCount` | 估计或准确命中数量，取决于查询 |
| `coverage` | 查询覆盖信息 |
| `timing` | 延迟信息 |

常用 debug 参数：

```bash
vespa query \
  'yql=select id,title,url from doc where userQuery() or ({targetHits:20}nearestNeighbor(embedding, q))' \
  'query=hybrid search tutorial' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=hybrid' \
  'ranking.listFeatures=true' \
  'hits=5'
```

如果想控制返回字段，可以用 document summary：

```bash
vespa query \
  'yql=select * from doc where userQuery() limit 5' \
  'query=vespa schema' \
  'ranking=bm25' \
  'presentation.summary=compact'
```

如果要看更详细的查询处理，可以提高 trace level：

```bash
vespa query \
  'yql=select * from doc where userQuery() limit 3' \
  'query=vespa' \
  'tracelevel=2'
```

`tracelevel` 输出会变多，不建议默认在生产流量中开启。

## 18. Grouping / Faceting 入门

搜索 UI 经常需要 facet，例如按标签聚合：

```text
tag: vespa (10)
tag: rag (4)
tag: vector-search (3)
```

Vespa 有 grouping/aggregation 能力。入门时不需要深挖复杂表达式，只需要知道它解决的是“搜索结果旁边的导航统计”问题，而不是普通排序问题。

一个简化理解：

```text
query -> matched docs -> group by tag/category -> return counts
```

在 e-commerce、文档站、日志搜索、素材库中，facet 很重要，因为用户常常先搜一个词，再用 category、tag、price、date、author 收窄结果。

如果你刚开始学习 Vespa，可以把 grouping 放在 BM25、filter、vector、hybrid 之后再学。先把检索和排序跑通，再做结果导航。

## 19. RAG 入门连接

RAG 系统里，Vespa 通常不负责调用 LLM，而负责 retrieval backend。

一个典型 RAG 链路：

```text
user question
  -> rewrite / classify / embed
  -> Vespa hybrid retrieval
  -> metadata filtering
  -> rank-profile reranking
  -> return top passages with citations
  -> LLM answer generation
```

Vespa 在这里承担：

| 环节 | Vespa 作用 |
| --- | --- |
| chunk 存储 | document schema 存 chunk text、source、url、metadata、embedding |
| lexical retrieval | BM25 / text matching |
| semantic retrieval | nearestNeighbor over embedding |
| filter | tenant、permission、language、time、source |
| ranking | hybrid profile、freshness、authority、reranker features |
| evidence return | document summary 返回 snippet/source/url |

它和 LangChain、LlamaIndex 的关系可以这样理解：

| 系统 | 典型职责 |
| --- | --- |
| Vespa | 存储、检索、排序、过滤、返回证据 |
| LangChain / LlamaIndex | 编排 LLM、prompt、tool、agent、retriever wrapper |
| Embedding service | 生成 document/query embedding |
| LLM service | 生成最终回答 |

如果 RAG 只是一个小 demo，任何 vector store 都能跑。如果 RAG 需要权限过滤、实时更新、混合检索、排序可解释、业务权重、多租户或大规模 serving，Vespa 的优势会更明显。

## 20. 推荐系统连接

Vespa 也常用于推荐和个性化系统。它可以做候选召回，也可以做在线排序。

和本站推荐系统冷启动文章对应起来：

| 推荐系统概念 | Vespa 中的落点 |
| --- | --- |
| Item feature | document fields / attributes / tensors |
| Item embedding | tensor field + HNSW |
| User embedding | `input.query(q)` |
| Candidate retrieval | nearestNeighbor / lexical / filters |
| Rank policy | rank-profile |
| Business rule | rank expression / filter |
| Debug feature | match-features / rank-features |
| Online serving | container + content cluster |

但要明确：Vespa 不自动解决推荐闭环。曝光日志、点击/收藏/购买、负反馈、训练样本、模型训练、embedding 导出、index 发布和 A/B 实验仍然要由推荐系统工程处理。

Vespa 适合作为 serving backend，而不是替代整个推荐平台。

## 21. 本地 Docker、Vespa Cloud、自托管如何选

| 方式 | 适合阶段 | 优点 | 风险 |
| --- | --- | --- | --- |
| 本地 Docker | 学习、schema 验证、demo | 快、便宜、可离线试错 | 单节点、不代表生产运维 |
| Vespa Cloud | 试生产、托管部署、团队项目 | 少管运维，接近生产服务方式 | 需要账号、权限、成本和 Cloud 配置 |
| 自托管/Kubernetes | 有平台团队和合规要求 | 自主管控强 | 运维、升级、容量、监控都要自己承担 |

Vespa CLI 支持本地和 Cloud target：

```bash
# 本地
vespa config set target local

# Vespa Cloud
vespa config set target cloud
vespa auth login
vespa config set application mytenant.myapp.default
```

本教程默认只做本地 Docker。原因很简单：入门先学 application package、schema、YQL、ranking。Cloud、Kubernetes、多节点、证书、部署流水线是下一阶段问题。

## 22. 常见错误清单

| 错误 | 现象 | 处理 |
| --- | --- | --- |
| 容器没起来 | `vespa status` 失败 | `docker ps`、`docker logs vespa` |
| 端口冲突 | Docker 启动失败或 endpoint 不通 | 检查 8080/19071 是否被占用 |
| Docker 内存不足 | 部署慢、容器异常、OOM | 给 Docker/Podman 至少 4GB+ |
| 不在 application package 目录 | `vespa deploy` 找不到文件 | 确认当前目录有 `services.xml` |
| schema 名和 document type 不一致 | feed/query 类型错误 | `services.xml` 的 document type 必须和 schema 对齐 |
| 字段没 `summary` | 查询结果没有该字段 | 给展示字段加 `summary` |
| 忘记 `index: enable-bm25` | BM25 profile 出错或得分异常 | 对需要 BM25 的 index field 开启 |
| vector 维度不匹配 | feed/query 报 tensor 错误 | schema、document embedding、query vector 维度一致 |
| nearestNeighbor 名称不一致 | 查询报 input 缺失 | YQL `q`、`input.query(q)`、rank-profile `query(q)` 对齐 |
| 只跑向量，不看过滤 | 结果语义相似但业务不可用 | 加 metadata filter 和 rank policy |
| hybrid 直接相加 | 某一路分数支配排序 | 用 match-features 调试尺度，必要时归一化 |
| 把 `targetHits` 当最终 hits | 返回数量不符合预期 | `targetHits` 是候选预算，`hits` 是返回数量 |

排错顺序建议：

```text
docker/container
  -> vespa status
  -> vespa deploy
  -> vespa feed
  -> where true
  -> BM25 only
  -> vector only
  -> hybrid
  -> rank debug
```

不要在 `vespa deploy` 还没成功时调 query；不要在 `where true` 查不到文档时调 ranking；不要在 BM25 和 vector 单路都没验证时调 hybrid。

## 23. 最小可运行清单

完成本文后，你应该有以下结果：

- Docker 容器 `vespa` 正在运行。
- `vespa config set target local` 已设置。
- `docs-search/services.xml` 存在。
- `docs-search/schemas/doc.sd` 存在。
- `vespa deploy` 成功。
- `vespa feed documents.jsonl` 成功。
- `vespa query 'yql=select * from doc where true limit 3'` 能返回文档。
- `ranking=bm25` 能返回文本相关结果。
- `ranking=semantic` + `nearestNeighbor` 能返回向量结果。
- `ranking=hybrid` 能合并文本和向量候选。
- `ranking.listFeatures=true` 能看到 BM25/closeness 特征。
- `docker exec vespa vespa-logfmt` 没有持续错误。

这是入门闭环。做到这里，你已经理解了 Vespa 最核心的开发循环：

```text
edit schema/services
  -> deploy
  -> feed
  -> query
  -> inspect features
  -> adjust rank-profile
  -> repeat
```

## 24. 与 Elasticsearch / OpenSearch / Milvus / Qdrant 的直觉对比

这个对比不是排名，只是帮助定位。

| 系统 | 直觉定位 | 适合 |
| --- | --- | --- |
| Elasticsearch / OpenSearch | 经典全文搜索和日志/文档搜索生态 | 搜索、日志、聚合、已有 ES 生态 |
| Milvus | 大规模向量检索系统 | 向量库、ANN、embedding 数据集 |
| Qdrant | 开发者友好的向量数据库 | 快速搭向量搜索、payload filter |
| Vespa | AI retrieval serving engine | 文本 + 向量 + 结构化过滤 + 多阶段排序 + 实时更新 |

Vespa 的学习曲线更陡，因为它要求你理解 schema、services、YQL、rank-profile、tensor 和分布式 serving。但它的好处也在这里：当系统从“只找相似向量”变成“多路召回 + 复杂排序 + 实时特征 + 可解释调试”时，Vespa 的统一模型会减少很多系统拼接成本。

## 25. 生产化下一步

本文只覆盖本地入门。生产化还需要补很多东西：

| 方向 | 要做什么 |
| --- | --- |
| 数据管线 | 文档生成、embedding 生成、feed retry、幂等、删除 |
| Schema 演进 | 字段新增、类型修改、validation-overrides、回滚 |
| 多节点 | redundancy、content node 数量、container node 数量 |
| 安全 | 认证、TLS、权限、网络隔离 |
| 监控 | query latency、feed latency、coverage、node health、resource |
| 排序实验 | rank-profile 版本、A/B、offline eval、online metrics |
| RAG | chunking、metadata、citation、reranking、answer quality eval |
| 推荐 | user features、item embedding 更新、曝光日志、模型同步 |
| 性能 | targetHits、HNSW 参数、summary size、rank phases、cache |

如果要从本文示例走向生产，建议按这个顺序推进：

1. 固定 schema 和 document id 规范。
2. 建立可重复 feed pipeline。
3. 建立 query 集合和离线评估。
4. 为每个 rank-profile 保留 match-features。
5. 压测 QPS、P95/P99 和 feed 延迟。
6. 再考虑 Vespa Cloud 或自托管多节点。

## 26. 推荐阅读路径

建议按这个顺序读官方材料：

1. Vespa basics：先理解 deploy、schema、writing、querying、ranking。
2. News tutorial：学习 application package、feed、query 的完整入门流程。
3. Hybrid Search Tutorial：学习 BM25 + vector + ranking profile。
4. Schema docs：系统理解 field、indexing、attribute、summary。
5. YQL docs：理解 select、where、contains、userQuery、nearestNeighbor。
6. Ranking docs：理解 first-phase、second-phase、global-phase、features。
7. Nearest neighbor search guide：理解 tensor、distance、HNSW、targetHits。
8. RAG docs：理解 chunk、embedder、hybrid retrieval、evidence return。
9. Vespa Cloud docs：如果要部署托管环境，再看 Cloud。
10. pyvespa：如果你习惯 Python 原型，可以再学 Python API。

不要一开始就读所有 reference。先用一个最小应用跑通，再带着具体问题查 reference，会快很多。

## 27. 完整命令回放

最后把本文的最小命令串起来，方便对照。

```bash
# 1. Start Vespa
docker pull vespaengine/vespa
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa

# 2. Create app
mkdir -p docs-search/schemas
cd docs-search

# 3. Write services.xml and schemas/doc.sd
# Use the files shown in this article.

# 4. Generate feed file
python3 generate-documents.py > documents.jsonl

# 5. Deploy and feed
vespa config set target local
vespa deploy
vespa feed documents.jsonl

# 6. Query all
vespa query 'yql=select * from doc where true limit 3'

# 7. BM25
vespa query \
  'yql=select id,title,url from doc where userQuery() limit 5' \
  'query=hybrid search tutorial' \
  'ranking=bm25'

# 8. Vector
python3 generate-query-vector.py > query-vector.txt
vespa query \
  'yql=select id,title,url from doc where {targetHits:10}nearestNeighbor(embedding, q)' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic' \
  'hits=5'

# 9. Hybrid
vespa query \
  'yql=select id,title,url from doc where userQuery() or ({targetHits:20}nearestNeighbor(embedding, q))' \
  'query=hybrid search tutorial' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=hybrid' \
  'ranking.listFeatures=true' \
  'hits=5'
```

上面 `generate-documents.py` 和 `generate-query-vector.py` 在正文里是 heredoc 形式展示的临时命令。真实项目里建议把它们变成正式脚本，并把 embedding 模型、字段校验、feed retry、日志和指标补上。

## 28. 从教程示例扩展到 RAG chunk schema

本文为了入门，把文档类型叫作 `doc`，字段也很少。真实 RAG 后端通常需要更细的 chunk schema。一个更接近生产的 chunk 文档可能包含：

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| `chunk_id` | string | chunk 主键 |
| `doc_id` | string | 原始文档 ID |
| `source_uri` | string | 引用来源 |
| `title` | string | 标题检索和展示 |
| `section` | string | 章节标题 |
| `text` | string | chunk 正文 |
| `tenant_id` | string | 多租户/权限过滤 |
| `acl` | array/string set | 权限过滤 |
| `language` | string | 语言过滤和分词 |
| `updated_at` | long | 新鲜度和增量更新 |
| `embedding` | tensor | 语义检索 |

对应 schema 可以这样演进：

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

查询时加入租户和语言过滤：

```bash
vespa query \
  'yql=select * from chunk where tenant_id contains "acme" and language contains "zh" and (userQuery() or ({targetHits:50}nearestNeighbor(embedding, q)))' \
  'query=Vespa 如何配置 rank profile' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=rag-hybrid' \
  'presentation.summary=citation' \
  'hits=8'
```

这个例子体现了 Vespa 做 RAG 后端的价值：权限过滤、语言过滤、BM25、向量候选、hybrid rank 和 citation summary 在一个 query 里完成。业务层拿到结果后再组织 prompt 给 LLM，而不是自己在多个系统之间拼候选。

但这里也有一个重要边界：RAG 的回答质量不只由 Vespa 决定。chunk 切分、embedding 模型、query rewrite、reranker、prompt、LLM、引用格式、评估集都会影响最终效果。Vespa 解决的是 retrieval/ranking serving，不替代完整 RAG 工程。

## 29. YQL 查询配方 Cookbook

下面整理一些常用查询形态。它们不一定都适合直接复制到生产，但可以帮助理解 Vespa 的表达方式。

**返回最新文档：**

```bash
vespa query \
  'yql=select id,title,published_at from doc where true order by published_at desc limit 10'
```

**关键词检索 + 标签过滤：**

```bash
vespa query \
  'yql=select id,title,tags from doc where tags contains "vespa" and userQuery() limit 10' \
  'query=rank profile' \
  'ranking=bm25'
```

**只用向量召回，但返回 compact summary：**

```bash
vespa query \
  'yql=select * from doc where {targetHits:20}nearestNeighbor(embedding, q)' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic' \
  'presentation.summary=compact' \
  'hits=5'
```

**文本和向量任一命中都可进入候选：**

```bash
vespa query \
  'yql=select * from doc where userQuery() or ({targetHits:30}nearestNeighbor(embedding, q))' \
  'query=Vespa vector search' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=hybrid' \
  'hits=10'
```

**过滤后再向量检索：**

```bash
vespa query \
  'yql=select * from doc where tags contains "rag" and ({targetHits:20}nearestNeighbor(embedding, q))' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic' \
  'hits=5'
```

**打开 ranking 特征调试：**

```bash
vespa query \
  'yql=select * from doc where userQuery() or ({targetHits:20}nearestNeighbor(embedding, q))' \
  'query=hybrid ranking' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=hybrid' \
  'ranking.listFeatures=true' \
  'hits=5'
```

**查看 trace：**

```bash
vespa query \
  'yql=select * from doc where userQuery() limit 3' \
  'query=vespa schema' \
  'tracelevel=2'
```

这些查询背后有三个问题需要反复问：

1. 这个 query 的候选从哪里来？BM25、nearestNeighbor、filter，还是组合？
2. 最终排序由哪个 rank-profile 控制？
3. 如果结果不符合预期，我能不能从 match-features 解释每个结果为什么排在这里？

只要这三个问题能回答，Vespa 查询就不会变成黑盒。

## 30. 性能和质量调参的第一层旋钮

入门跑通以后，下一步通常是调质量和延迟。先不要急着上复杂模型，先看这些基础旋钮。

| 旋钮 | 影响 | 常见取舍 |
| --- | --- | --- |
| `hits` | 返回结果数量 | 数量越大，响应体越大 |
| `targetHits` / `totalTargetHits` | nearestNeighbor 候选预算 | 候选越多，召回可能更好，延迟更高 |
| HNSW `max-links-per-node` | 图连接度 | 更高可能提升召回，但内存更高 |
| HNSW `neighbors-to-explore-at-insert` | 插入时探索量 | 更高建图更好，写入更慢 |
| summary 字段数量 | 返回 payload 大小 | 返回越多，网络和填充成本越高 |
| rank expression 复杂度 | 排序耗时 | 表达式越复杂，延迟越高 |
| match-features 数量 | 调试信息 | 调试有用，但不要默认给所有线上请求开 |
| filter 选择性 | 候选范围 | 过滤太宽会慢，太窄会漏召回 |

质量调参建议从离线 query set 开始。准备 50-200 个真实 query，每个 query 人工标注或至少人工查看 top10。每次改 rank-profile 后，记录：

- 哪些 query 变好了。
- 哪些 query 变差了。
- BM25 和 closeness 的分数范围。
- 是否有某类字段过度支配排序。
- 是否有过滤导致候选不足。

不要只看一个 demo query。Hybrid search 很容易出现“某个 query 看起来很好，但整体退化”的情况。

性能调参建议从三条曲线开始：

```text
targetHits -> latency
targetHits -> judged relevance
summary size -> response latency
```

如果 targetHits 从 20 增加到 200，质量几乎不变但延迟明显上升，就没有必要浪费预算。如果 summary 返回了大段正文但 UI 只展示标题和链接，就应该改 document-summary。

## 31. Schema 演进和线上发布风险

本地学习时，改 schema 后直接 `vespa deploy` 就行。生产环境不能这么随意。Schema 变化会影响索引、字段存储、rank-profile、feed pipeline 和查询兼容性。

常见 schema 变更：

| 变更 | 风险 |
| --- | --- |
| 新增 summary 字段 | 通常较低 |
| 新增 attribute 字段 | 需要 feed 或 update 数据 |
| 新增 index 字段 | 需要建立索引 |
| 修改字段类型 | 高风险，可能需要重建数据 |
| 修改 tensor 维度 | 高风险，feed/query/model 全部要同步 |
| 修改 rank-profile | 线上排序变化，需要评估和回滚 |
| 修改 document type | 高风险，影响 feed 和 query |

一个稳妥的线上流程应该有：

```text
schema change proposal
  -> local deploy test
  -> staging deploy
  -> feed sample docs
  -> query regression
  -> latency test
  -> production rollout
  -> monitor
  -> rollback plan
```

对于向量字段，尤其要注意版本管理。embedding 模型变了，向量空间就变了。不要把新模型生成的 query vector 去搜旧模型生成的 document embedding，也不要让 user tower 和 item index 跨版本混用。推荐系统和 RAG 都容易踩这个坑。

建议在文档里加版本字段：

```text
field embedding_version type string {
  indexing: attribute | summary
  match: word
}
```

查询时过滤版本：

```bash
vespa query \
  'yql=select * from doc where embedding_version contains "e5-small-v1" and ({targetHits:50}nearestNeighbor(embedding, q))' \
  "input.query(q)=$(cat query-vector.txt)" \
  'ranking=semantic'
```

这会让灰度和回滚更可控。

## 32. 入门实验路线：从 1 小时到 1 周

如果你真要学 Vespa，不建议只读文章。可以按这个节奏做实验。

**第 1 小时：跑通本地链路**

- Docker 启动。
- `vespa deploy`。
- `vespa feed`。
- `where true` 查询。
- BM25 查询。

验收标准：能解释 `services.xml` 和 `doc.sd` 各自作用。

**第 1 天：建立搜索质量直觉**

- 准备 20 个真实 query。
- 每个 query 看 BM25 top10。
- 加 tags/time 过滤。
- 开 match-features 看 BM25(title) 和 BM25(body)。
- 调一次 rank-profile。

验收标准：能解释某个结果为什么排第一。

**第 2 天：加入向量**

- 用真实 embedding 模型生成 384 或 768 维 embedding。
- feed 文档向量。
- 跑 nearestNeighbor。
- 对比 BM25 和 semantic 的结果差异。
- 记录哪些 query 适合 BM25，哪些适合 vector。

验收标准：能说明向量检索解决了哪些词面不匹配问题，也能指出哪些精确词场景它不如 BM25。

**第 3 天：做 Hybrid**

- 用 `or` 合并 userQuery 和 nearestNeighbor。
- 建 `hybrid` rank-profile。
- 打开 match-features。
- 调整 BM25 和 closeness 权重。
- 人工评估 top10。

验收标准：能说明 hybrid 是否比单路更好，以及坏 case 是什么。

**第 1 周：接近真实服务**

- 设计稳定 document id。
- 建 feed retry 和删除流程。
- 增加 source、tenant、language、updated_at 等 metadata。
- 增加 compact/citation summaries。
- 做 query regression set。
- 压测 targetHits、hits、summary size。
- 决定是否试 Vespa Cloud 或自托管环境。

验收标准：这不再是 demo，而是一个可以接入业务原型的 retrieval backend。

这个路线比“直接上 RAG demo”慢一点，但更稳。很多 RAG demo 的问题不是 LLM，而是 retrieval 后端不可解释；Vespa 的学习价值正在于把 retrieval 和 ranking 解释清楚。

## 33. Vespa 入门术语表

最后把本文反复出现的术语集中整理一下。初学 Vespa 时，很多困惑来自同一个词在不同系统里含义相近但边界不同。

| 术语 | 入门解释 |
| --- | --- |
| Application package | Vespa 应用配置包，至少包含 `services.xml` 和 schema |
| `services.xml` | 定义 container/content cluster、document API、search endpoint 等服务拓扑 |
| Schema | 定义文档类型、字段、索引、summary、rank-profile |
| Document type | Vespa 中的一类文档，例如本文的 `doc` 或 RAG 的 `chunk` |
| Document id | Vespa 文档唯一标识，例如 `id:docs-search:doc::hybrid-search` |
| Field | 文档字段，例如 `title`、`body`、`embedding` |
| `index` | 字段进入索引，支持全文匹配和 BM25 等特征 |
| `attribute` | 字段作为 attribute 存储，支持过滤、排序、实时访问、tensor 距离计算 |
| `summary` | 字段可以返回给客户端 |
| Fieldset | 一组文本字段，常用于 `userQuery()` 默认搜索范围 |
| Rank profile | 排序配置，定义 first-phase、second-phase、特征和输入 |
| First phase | 第一阶段排序，通常在 content node 上对候选计算 |
| Match features | 命中阶段输出的调试特征，用于解释排序 |
| YQL | Vespa Query Language，用于表达从哪个 schema 中取哪些候选 |
| `userQuery()` | 使用请求里的 `query=` 参数在默认 fieldset 上做文本查询 |
| `nearestNeighbor` | Vespa 的向量近邻查询 operator |
| `targetHits` | nearestNeighbor 的候选预算，不等于最终返回数量 |
| Tensor | Vespa 表达向量和多维数据的类型系统 |
| HNSW | 常用 approximate nearest neighbor 图索引 |
| `closeness` | Vespa ranking 中用于向量近邻相似度的 rank feature |
| BM25 | 经典词面相关性排序函数 |
| Hybrid search | 通常指 lexical retrieval 和 vector retrieval 的组合 |
| Container cluster | 无状态请求处理层 |
| Content cluster | 有状态数据、索引、检索和 ranking 执行层 |
| Vespa CLI | 用于 deploy、feed、query、status 等开发操作的命令行工具 |
| Vespa Cloud | Vespa 官方托管服务 |

如果只记一个公式化心智模型，可以记这个：

```text
schema 决定数据怎么被理解
YQL 决定候选怎么被取出
rank-profile 决定候选怎么排序
summary 决定结果怎么返回
```

如果一个查询结果不对，也按这个顺序查：

1. 文档是否写入成功？
2. 字段是否被正确 indexing？
3. 查询 YQL 是否真的选中了候选？
4. rank-profile 是否被正确指定？
5. summary 是否包含你想看的字段？
6. match-features 是否解释得通？

这套排查顺序比盲目调权重有效得多。

## 34. 学完本文后的自测问题

可以用下面这些问题检验自己是否真的入门。

**概念类：**

1. 为什么 Vespa 不应该只被理解为向量数据库？
2. `container` 和 `content` 的职责有什么区别？
3. `index`、`attribute`、`summary` 三者分别解决什么问题？
4. 为什么 `index: enable-bm25` 对 BM25 rank-profile 很重要？
5. 为什么 query vector 的维度必须和 schema 中 tensor 维度一致？
6. `targetHits` 和 `hits` 有什么区别？
7. 为什么 hybrid search 不应该直接无脑把 BM25 和向量分数相加？

**操作类：**

1. 如何确认本地 Vespa 容器已经启动？
2. 如何把 Vespa CLI 指向本地？
3. `vespa deploy` 应该在哪个目录执行？
4. 如何 feed JSONL？
5. 如何查询所有文档确认写入成功？
6. 如何打开 ranking feature 调试？
7. 如何查看 Vespa 容器日志？

**工程类：**

1. 如果查询命中了文档但没有返回 `url`，你会先查什么？
2. 如果 BM25 查询没有结果，是 schema、feed、YQL、还是 rank-profile 的问题？怎么逐步排查？
3. 如果 nearestNeighbor 报 tensor 维度错误，最可能是哪几处不一致？
4. 如果 hybrid search 的结果全被向量分数支配，应该看哪些 match-features？
5. 如果要上线 RAG，为什么要加 `tenant_id`、`source_uri`、`updated_at` 这类 metadata？
6. 如果 embedding 模型换版本，为什么 document embedding 和 query embedding 必须成对切换？
7. 如果返回结果很慢，为什么应该先检查 summary size、targetHits 和 rank expression？

这些问题都能答出来，再去读官方 reference 会轻松很多。

## 35. 结论

Vespa 入门的关键不是背命令，而是建立正确分层：

- `services.xml` 定义服务拓扑。
- schema 定义文档、索引、属性、summary 和 rank profiles。
- feed 把业务数据写成 Vespa documents。
- YQL 负责检索候选。
- rank-profile 负责排序和特征计算。
- tensor / nearestNeighbor 负责向量检索。
- hybrid search 把 lexical 和 semantic 两种信号放到同一个 retrieval + ranking 流程里。

一旦这个心智模型建立起来，Vespa 就不再是“配置很多的搜索引擎”，而是一个统一的 retrieval serving 平台。你可以从 BM25 文本搜索开始，逐步加入向量、过滤、排序、RAG、推荐和模型推理；每一步都能通过 schema、YQL、rank-profile 和 match-features 解释清楚。

最稳妥的学习路径是：先跑通本地 Docker，写一个只有 `title/body` 的搜索应用；再加 BM25；再加 tensor 和 nearestNeighbor；最后做 hybrid ranking。不要把 Vespa 一开始就当成全功能黑盒，也不要把它简化成普通向量库。它的价值正是在两者之间：用一个系统承载大规模、实时、可排序、可调试的 AI retrieval。

## References

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

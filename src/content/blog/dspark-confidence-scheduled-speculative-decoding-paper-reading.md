---
title: "DSpark 论文精读：置信度调度推测解码、半自回归生成与大模型推理加速"
description: "精读 DeepSeek DSpark 如何通过半自回归 draft model、confidence-scheduled verification 和 DeepSpec 训练评估代码改进大模型 speculative decoding 的速度、接受率与吞吐边界"
pubDate: "2026-06-29T14:12:56+08:00"
updatedDate: "2026-06-29T14:12:56+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "LLM Inference"
  - "Speculative Decoding"
  - "Semi-Autoregressive Generation"
  - "DeepSeek"
  - "Systems"
draft: false
---

Xin Cheng、Xingkai Yu、Chenze Shao、Jiashi Li、Yunfan Xiong 等来自北京大学与 DeepSeek-AI 的研究者在 2026 年发布的 **DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation**，是一篇面向大规模在线推理服务（LLM Serving）的系统级技术报告。它直面了推测解码（speculative decoding）在工业界高并发部署中的一个核心矛盾：

> 当大模型服务处在严苛的交互延迟约束（TPS SLA）与有限的目标模型批处理容量（target batch capacity）下，推测解码应当生成怎样的候选序列，并把多少候选 token 送交目标模型验证？

传统推测解码往往默认“草稿模型（drafter）单次生成的候选块越长、每轮验证接受的 token 越多，推理就越快”。然而在真实的高并发服务中，**目标模型的验证批容量（verification batch capacity）是高度稀缺的共享资源**。如果一批并发请求中掺杂了大量低置信度的草稿后缀（draft suffix），盲目将它们送入目标模型验证会严重挤占有效算力，导致系统聚合吞吐下滑，甚至无法满足单用户的生成速率 SLA。

针对这一痛点，DSpark 从生成与验证两端协同破局：在草稿生成端采用**半自回归生成（semi-autoregressive generation）**，以高容量的并行主干网络保证 $O(1)$ 步生成长候选块的吞吐优势，再叠加极轻量的顺序头（sequential head）注入块内采样依赖，有效抑制并行草稿模型的后缀接受率衰减（suffix decay）；在目标验证端提出**置信度调度验证（confidence-scheduled verification）**，精准预估每个草稿位置被目标模型接受的概率，并结合推理引擎的硬件吞吐曲线 $\mathrm{SPS}(B)$，在批级别动态截断各请求的验证前缀长度。

本文结合论文与官方开源代码库 `deepseek-ai/DeepSpec` 展开精读，依次剖析半自回归草稿架构、硬件感知前缀调度算法、训练与评测代码实现，以及离线基准与 DeepSeek-V4 线上服务的实验表现。文中图表截取自官方技术报告 PDF。

## 1. 论文概览与推测解码的系统瓶颈

| 项目 | 内容 |
| --- | --- |
| 题名 | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation |
| 作者 | Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang |
| 机构 | Peking University, DeepSeek-AI |
| PDF | 33 页技术报告 |
| 官方代码 | `deepseek-ai/DeepSpec` |
| 开源内容 | DeepSpec 训练/评估代码、DSpark/DFlash/Eagle3 配置、公开基准评测脚本、Hugging Face 预训练权重 |

DSpark 的核心贡献可以凝练为：**将推测解码的优化目标从单纯拉长草稿生成长度，升级为提升候选前缀（prefix）的块内一致性，并依据校准后的存活概率与硬件吞吐曲线，将有限的目标模型验证预算精准分配给期望收益最高的候选 token**。

围绕这一目标，全文贯穿着三个紧密咬合的核心概念：**连续前缀存活（prefix survival）**——由于目标模型验证只接受从头开始的连续匹配前缀，一旦首个位置被拒，后续位置的草稿无论质量高低都会在本轮作废，因此越靠前的位置权重越高；**分布对齐置信度（confidence）**——置信度头预测的目标是草稿分布与目标模型分布的重叠率（即真实接受概率），而非草稿模型自身的词元最大概率；**硬件感知调度（hardware-aware scheduling）**——验证低置信度词元的机会成本随系统并发动态变化，调度器必须将推理引擎的每秒步数曲线 $\mathrm{SPS}(B)$ 纳入全局吞吐目标函数。

### 1.1 自回归解码延迟矛盾与推测解码基础

标准大语言模型遵循自回归解码（autoregressive generation）：给定上下文 $x_{1:t}$，模型计算条件分布 $p_\theta(x_{t+1}\mid x_{\le t})$ 并采样出 $x_{t+1}$，随后将其拼回序列继续计算 $p_\theta(x_{t+2}\mid x_{\le t+1})$。尽管 KV Cache 避免了历史上下文的重复编码，但每生成一个新 token 依然需要串行读取一遍目标模型的全部权重。在线上服务中，这一访存密集型特性引发了四个维度的指标博弈：

| 服务维度 | 核心诉求 | 系统代价与冲突 |
| --- | --- | --- |
| 单请求低延迟 | 尽快返回每个生成的 token | 被迫限制批大小（batch size），GPU 算力利用率低 |
| 系统高吞吐 | 大规模多请求合批以摊销权重访存 | 单步推理耗时增加，请求排队与交互等待拉长 |
| 模型高质量 | 部署参数量更大的目标模型（Target Model） | 单步前向传播的访存与计算开销成倍上升 |
| 强交互 SLA | 保障高的单用户生成速率（tok/s/user） | 单机可承载的并发连接数大幅萎缩 |

推测解码（Speculative Decoding）通过引入一个轻量的草稿模型（Draft Model，分布记为 $q$）来打破串行瓶颈：在每一轮解码中，先由草稿模型快速提议 $\gamma$ 个候选词元 $\hat{x}_{t+1:t+\gamma}$，再交由目标模型（分布记为 $p$）在单次前向传播中并行计算这 $\gamma$ 个位置的真实分布，并通过修正拒绝采样（rejection sampling）验证最长可接受前缀。这一机制在严格保证最终输出分布与目标模型 $p$ 完全一致（无损加速）的前提下，用每轮平均接受长度（accepted length，记作 $\tau$）成倍放大了单次目标模型前向传播的有效产出。

### 1.2 自回归与并行 Drafter 的两难：延迟与后缀衰减（Suffix Decay）

在设计草稿模型时，现有主流方案分化为自回归（Autoregressive）与并行（Parallel）两条路线，各自面临明显的结构瓶颈：

- **自回归草稿模型（如 Eagle3）**：按顺序逐个生成候选词元，联合分布严格拆解为条件连乘：
  $$
  q(\hat{x}_{t+1:t+\gamma}\mid x_{\le t}) = \prod_{j=1}^{\gamma} q(\hat{x}_{t+j}\mid x_{\le t},\hat{x}_{t+1:t+j-1})
  $$
  由于后序位置能够观测到前序实际采样出的词元，块内语义连贯性好。但代价是草稿阶段本身必须串行执行 $\gamma$ 次前向计算，草稿延迟呈 $O(\gamma)$ 线性增长；为了压低这部分开销，自回归草稿网络通常只能设计得很浅（例如单层），限制了首词元的预测容量。
- **并行草稿模型（如 DFlash）**：单次前向传播同时输出全部 $\gamma$ 个位置的条件独立分布：
  $$
  q(\hat{x}_{t+1:t+\gamma}\mid x_{\le t}) \approx \prod_{j=1}^{\gamma} q_j(\hat{x}_{t+j}\mid x_{\le t})
  $$
  并行生成的草稿延迟近似为 $O(1)$，允许使用更深的主干网络（如 5 层）来提升首词元预测质量。但由于位置 $j$ 无法感知位置 $1,\ldots,j-1$ 的具体采样结果，一旦上下文存在多种合理的后续分支（多模态分布），不同位置极易各自挑选互不兼容的分支词元，拼凑出语法或逻辑断裂的后缀，导致条件接受率随位置后移急剧滑坡——即论文重点指出的**后缀衰减（suffix decay）**。

DSpark 的出发点正是融合两者的优势：保留并行主干的高容量与 $O(1)$ 块生成效率，同时在输出端嵌入极轻量的半自回归顺序修正头来消除后缀冲突，并配合置信度调度器裁除低价值后缀。

### 1.3 DeepSpec 开源生态与复现边界

论文配套开源的 [DeepSeek-AI/DeepSpec](https://github.com/deepseek-ai/DeepSpec) 仓库提供了完整的数据准备、模型定义、训练器与基准评估框架，并在 Hugging Face 上发布了基于 Qwen3-4B、Qwen3-8B、Qwen3-14B 与 Gemma4-12B 的 Eagle3、DFlash 及 DSpark 预训练权重（例如 [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7)）。论文核心组件与开源代码模块的对应关系如下：

| 论文核心模块 | DeepSpec 源码路径 | 功能说明 |
| --- | --- | --- |
| 训练超参数配置 | `config/dspark/dspark_qwen3_4b.py` 等 | 定义块大小、草稿层数、目标层索引、Markov 头与损失权重 |
| 半自回归顺序头 | `deepspec/modeling/dspark/markov_head.py` | 实现 `VanillaMarkov`、`GatedMarkovHead` 与 `RNNHead` |
| DSpark 主模型 | `deepspec/modeling/dspark/{qwen3,gemma4}/modeling.py` | 封装并行主干、基础 logits、置信度头与目标隐藏层融合 |
| 联合训练损失 | `deepspec/modeling/dspark/loss.py` | 实现交叉熵、分布 L1 距离、置信度 BCE 与位置衰减加权 |
| 分布式训练器 | `deepspec/trainer/dspark_trainer.py` | 管理训练循环、模型前向与损失回传 |
| 离线提议与截断 | `deepspec/eval/dspark/draft_ops.py` | 执行并行主干前向、顺序头逐步采样与静态置信度阈值截断 |
| 评估与校准统计 | `deepspec/eval/dspark/{evaluator.py,confidence_head.py}` | 运行验证闭环，统计接受长度 $\tau$、ECE、AUROC、Brier 及可靠性图 |

关于代码与实验的边界在此统一说明：首先，完整复现从零训练需要预计算并存储目标模型的隐藏状态缓存（默认 `Qwen/Qwen3-4B` 配置下 target cache 约需 **38TB** 磁盘空间），普通研究环境更适合直接加载官方发布的预训练权重运行评估；其次，开源仓库的 `eval` 模块针对离线单样本评测设计，使用的是静态置信度阈值（`confidence_threshold`），而论文 Algorithm 1 所述的硬件感知批级全局调度器以及 Fig. 7/8 中基于 DeepSeek-V4 生产引擎与线上真实流量的服务曲线，属于内部生产推理系统范畴，未包含在公开评估脚本中。

## 2. DSpark 整体架构与半自回归生成机制

### 2.1 Draft Better + Verify Smarter 解码周期（Fig. 1）

Fig. 1 完整描绘了 DSpark 在单轮推测解码中的工作流：

![DSpark architecture and decoding cycle](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig1-architecture-cycle.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 1.*

整个解码周期由三个阶段串联而成：
1. **目标锚点生成（Anchor Generation）**：给定提示词序列 `A B C`，目标模型在上一轮验证结束时真实推进并产出一个锚点词元 `D`（连同其多层隐藏状态），作为当前轮草稿生成的可靠起点。
2. **半自回归草稿提议（Drafting Phase — Draft Better）**：锚点 `D` 与掩码占位符送入 DSpark 的并行 Transformer 主干（Parallel Block），一次性并行算出全部 $\gamma$ 个候选位置的隐藏表征与基础 logits；随后轻量顺序块（Sequential Block）依次在块内做局部条件采样，产出候选序列 `E F G H` 及对应的逐位置置信度分数 $c_1, c_2, c_3, c_4$。
3. **硬件感知前缀调度与验证（Verification Phase — Verify Smarter）**：调度器结合当前并发批次内所有请求的置信度序列与硬件吞吐曲线 $\mathrm{SPS}(B)$，判定末尾词元 `H` 的预期收益不足以抵消其占用的批处理容量，当即裁掉 `H` 并仅将高置信度前缀 `E F G` 送入目标模型并行验证；最终目标模型接受 `E` 和 `F`，在第三位拒绝 `G` 并重采样产出修正词元 `G*`，开启下一轮循环。

### 2.2 半自回归生成：并行主干与轻量顺序修正

从数学形式上看，DSpark 的半自回归生成将草稿预测解耦为“全局并行特征提取”与“局部低秩顺序修正”两步。首先，并行主干网络（例如 5 层非因果注意力层）在单次前向传播中并行产出 $\gamma$ 个候选位置的隐藏状态与基础词表对数几率（base logits）：

$$
h_1,\ldots,h_\gamma = \mathrm{ParallelBackbone}(x_{\le t}), \qquad U_1,\ldots,U_\gamma = W_{\mathrm{lm}}[h_1,\ldots,h_\gamma]
$$

若直接对每个 $U_j$ 独立采样，模型即退化为 DFlash。为了在不牺牲 $O(1)$ 主干吞吐的前提下恢复块内条件依赖，DSpark 引入了一个轻量偏置函数 $b(\hat{x}_{j-1}, h_j)$，利用前一个位置刚刚采样出的词元 $\hat{x}_{j-1}$ 动态修正当前位置的 logits：

$$
\tilde{U}_j = U_j + b(\hat{x}_{j-1}, h_j), \qquad \hat{x}_j \sim \mathrm{Softmax}(\tilde{U}_j)
$$

由于偏置函数 $b(\cdot)$ 仅涉及低秩矩阵向量乘或极小的循环单元，无需重新执行跨全序列的自注意力计算，因此顺序采样的额外延迟开销极低（实测仅占 0.6%～1.3%），却能有效锚定块内的局部语法与多词元搭配一致性。

### 2.3 Markov / RNN Head 的三种建模形态（markov_head.py）

在 `deepspec/modeling/dspark/markov_head.py` 中，官方代码实现了三种复杂度递增的顺序修正头：

| 源码类名 | 依赖范围与建模机制 | 核心数学形式 |
| --- | --- | --- |
| `VanillaMarkov` | 一阶马尔可夫转移：仅依赖前一个采样词元 $\hat{x}_{j-1}$ 的低秩词表偏置 | $b_j = W_2 W_1[\hat{x}_{j-1}], \quad \tilde{U}_j = U_j + b_j$ |
| `GatedMarkovHead` | 上下文门控一阶转移：结合当前位置隐藏状态 $h_j$ 动态门控转移特征 | $g_j=\sigma(W_g[h_j;W_1[\hat{x}_{j-1}]]), \quad b_j=W_2(g_j\odot W_1[\hat{x}_{j-1}])$ |
| `RNNHead` | 块内循环状态传递：维护隐状态 $s_j$，聚合整个已采样前缀 $\hat{x}_{1:j-1}$ 的信息 | $s_j, o_j = \mathrm{RNN}(s_{j-1}, W_1[\hat{x}_{j-1}], h_j), \quad b_j = W_2(o_j)$ |

在训练阶段，源码调用 `apply_block_logits()`，直接利用真实目标序列（teacher forcing）并行计算所有位置的修正后 logits $\tilde{U}_{1:\gamma}$；在推理阶段，源码调用 `sample_block_tokens()`，在一个长度为 $\gamma$ 的轻量循环中逐位置完成“查表计算偏置 $b_j \rightarrow$ 叠加 $U_j \rightarrow$ 采样 $\hat{x}_j$”，从而以极小代价实现了半自回归块生成。

## 3. 置信度预测与硬件感知前缀调度器

### 3.1 面向分布重叠度的 Confidence Head 与前缀存活率

在推测解码中，草稿模型自身输出的词元概率 $\max_v q(v)$ 往往存在严重的过度自信，并不能准确反映目标模型是否会接受该词元。从拒绝采样的理论性质出发，给定草稿分布 $q$ 与目标分布 $p$，该位置词元被目标模型接受的真实期望概率（accept rate）严格等于两个离散分布的总变差距离补集（即分布重叠面积）：

$$
\alpha = 1 - \frac{1}{2}\lVert p - q \rVert_1 = 1 - \frac{1}{2}\sum_{v \in \mathcal{V}} |p(v) - q(v)|
$$

在 `deepspec/modeling/dspark/loss.py` 的 `_compute_accept_rate_3d()` 函数中，代码正是通过对草稿 logits 与目标 logits 分别执行 Softmax 后按上式计算软标签 $\alpha \in [0, 1]$，并使用二元交叉熵（BCE）训练置信度预测头 $\hat{c}_j = \sigma(\text{ConfidenceHead}(h_j, \hat{x}_{j-1}, \hat{x}_j))$：

$$
\mathcal{L}_{\mathrm{conf}} = \mathrm{BCEWithLogits}(\text{logit}(\hat{c}_j),\, \alpha_j)
$$

考虑到推理调度器真正关心的是从块首到第 $j$ 个位置的**连续前缀存活概率（prefix survival probability）**：

$$
a_{r,j} = \prod_{i=1}^{j} c_{r,i}
$$

单步置信度的微小系统性高估会在连乘中呈指数级放大。为此，源码不仅统计单步绝对误差 `confidence_abs_error` 与均值偏差 `confidence_bias`，还专门监控累积乘积偏差 `confidence_cumprod_bias`，并在部署前通过温度缩放等后验校准（post-hoc calibration）手段消除连乘偏差。

### 3.2 硬件感知前缀调度算法（Algorithm 1）

拿到批内各请求经过校准的逐位置置信度后，Algorithm 1 给出了将验证预算分配问题与底层推理引擎硬件特性解耦结合的全局贪心调度器：

![Hardware-Aware Prefix Scheduler](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-algorithm1-hardware-aware-prefix-scheduler.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Algorithm 1.*

设当前并发批次共有 $R$ 个活跃请求，第 $r$ 个请求生成的草稿块长度为 $\gamma$，其第 $j$ 个位置的前缀存活概率为 $a_{r,j} = \prod_{i \le j} c_{r,i}$。若调度器为请求 $r$ 分配的验证前缀长度为 $\ell_r \in \{0, 1, \ldots, \gamma\}$，则目标模型在本轮前向验证中需要处理的**总词元批大小**（Total Verification Batch Size）为：

$$
B(\ell_1,\ldots,\ell_R) = \sum_{r=1}^{R}(1 + \ell_r)
$$

其中常数 $1$ 代表每个请求必定推进或验证的基础锚点位。与此同时，整个批次在本轮验证后预期产出的总有效词元数为：

$$
\tau(\ell_1,\ldots,\ell_R) = \sum_{r=1}^{R}\left(1 + \sum_{j=1}^{\ell_r} a_{r,j}\right)
$$

记推理引擎在批大小为 $B$ 时的每秒执行步数（Step-Per-Second，由引擎初始化时离线压测建表获得）为 $\mathrm{SPS}(B)$，则系统的**期望聚合输出吞吐**（Expected Aggregate Throughput）定义为：

$$
\Theta(\ell_1,\ldots,\ell_R) = \tau(\ell_1,\ldots,\ell_R) \cdot \mathrm{SPS}\!\left(B(\ell_1,\ldots,\ell_R)\right)
$$

调度器的目标是求解组合优化问题 $\max_{\ell_1,\ldots,\ell_R} \Theta$。由于对任意请求 $r$，前缀存活率 $a_{r,j} = \prod_{i \le j} c_{r,i}$ 随位置 $j$ 自然单调递减（$a_{r,1} \ge a_{r,2} \ge \cdots \ge a_{r,\gamma}$），这意味着在任意请求内部，靠前的位置永远比靠后的位置具备更高的边际收益。利用这一单调性质，Algorithm 1 无需做指数级搜索，直接将所有请求的全部候选位置 $(r, j)$ 按 $a_{r,j}$ 从大到小全局排序，每次贪心接纳当前存活率最高的一个词元（令 $B \leftarrow B+1,\ \tau \leftarrow \tau + a_{r,j}$），并在吸纳过程上精确评估 $\Theta = \tau \cdot \mathrm{SPS}(B)$ 的全局最大值点，从而在 $O(R\gamma \log(R\gamma))$ 时间内求得最优验证长度分配 $(\ell_1^\ast,\ldots,\ell_R^\ast)$。

### 3.3 动态负载调度对静态阈值的优势

对比常见的固定阈值截断策略（例如无视系统负载统一保留 $c_j > 0.7$ 的前缀），Algorithm 1 的核心突破在于实现了**负载自适应（Load-Adaptivity）**：

| 系统运行负载状态 | $\mathrm{SPS}(B)$ 局部斜率特征 | 验证低置信词元的机会成本 | Algorithm 1 的自适应调度行为 |
| --- | --- | --- | --- |
| 低并发（访存受限区） | 增大 $B$ 几乎不降低 $\mathrm{SPS}(B)$ | 极低（空闲算力近乎免费） | 放宽准入，分配长验证前缀以最大化单步接受长度 $\tau$ |
| 中等并发（过渡区） | $\mathrm{SPS}(B)$ 开始随 $B$ 缓步下滑 | 中等 | 优先保障高确定性前缀（如数学/代码请求），裁剪发散的闲聊后缀 |
| 高并发（算力饱和区） | 继续增大 $B$ 导致 $\mathrm{SPS}(B)$ 陡降 | 极高（直接拖慢全批所有请求） | 激进收缩验证预算，仅保留首尾极高把握词元，力保系统总吞吐与 SLA |

更重要的是，该调度器打破了请求间的隔离：在同一个并发批次内，来自结构化代码生成请求的第 5 个草稿词元（可能 $a_{r,5} = 0.85$）完全可以优先于另一个开放式创意写作请求的第 2 个草稿词元（可能 $a_{r',2} = 0.30$）获得目标模型的验证配额。

## 4. 训练目标设计与 DeepSpec 工程管线

### 4.1 联合损失函数：CE、分布 L1 对齐与位置衰减（loss.py）

在 `deepspec/modeling/dspark/loss.py` 的 `compute_dspark_loss()` 中，DSpark 的训练目标由词元级交叉熵 $\mathcal{L}_{\mathrm{CE}}$、草稿与目标分布的 $L_1$ 距离对齐损失 $\mathcal{L}_{L1}$ 以及置信度预测损失 $\mathcal{L}_{\mathrm{conf}}$ 三项加权构成，并引入了块内位置指数衰减权重 $w_j$：

$$
w_j = \exp\!\left(-\frac{j}{\gamma_{\mathrm{decay}}}\right), \qquad
\mathcal{L} = \sum_{j=1}^{\gamma} w_j \left( \lambda_{\mathrm{CE}} \mathcal{L}_{\mathrm{CE},j} + \lambda_{L1} \mathcal{L}_{L1,j} \right) + \lambda_{\mathrm{conf}} \mathcal{L}_{\mathrm{conf}}
$$

对照 `config/dspark/dspark_qwen3_4b.py` 中的默认超参数设定，各项权重的具体取值与物理意义如下：

| 配置参数 | 默认值 | 训练作用解析 |
| --- | ---: | --- |
| `ce_loss_alpha` ($\lambda_{\mathrm{CE}}$) | `0.1` | 针对目标真实词元 $x^\ast_j$ 的负对数似然 $-\log q_\phi(x^\ast_j\mid \cdot)$，提供基础监督锚点 |
| `l1_loss_alpha` ($\lambda_{L1}$) | `0.9` | 最小化 $\lVert q_\phi(\cdot) - p_\theta(\cdot) \rVert_1$，直接等价于最大化推测解码理论接受率 $\alpha = 1 - \frac{1}{2}\lVert p-q\rVert_1$ |
| `confidence_head_alpha` ($\lambda_{\mathrm{conf}}$) | `1.0` | 训练置信度头拟合脱离计算图的真实分布重叠率 `accept_rate_3d.detach()` |
| `loss_decay_gamma` ($\gamma_{\mathrm{decay}}$) | `4.0` | 赋予块首位置更高梯度权重，契合前缀存活对靠前位置极度敏感的物理特性 |

将 `l1_loss_alpha` 设为主导项（`0.9`）体现了推测解码与普通知识蒸馏的差异：由于拒绝采样直接取决于 $\lVert p - q \rVert_1$，直接优化 $L_1$ 距离比单纯优化 KL 散度或硬标签 CE 更契合下游验证目标。

### 4.2 数据生成、Target Cache 与训练配置（dspark_qwen3_4b.py）

DeepSpec 的训练数据流（见 `scripts/data/README.md`）分为三步：首先从源数据集（默认为 [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)）切分提示词；其次启动兼容 OpenAI `/v1` 接口的推理引擎（如 SGLang 或 vLLM），用目标模型自举生成全部回复文本，确保草稿模型的训练分布与目标模型线上解码分布严格对齐；最后提取目标模型在指定深浅层上的隐藏状态落盘构建 Target Cache。

在训练启动阶段，`scripts/train/train.sh` 调用 `train.py` 并根据可见 GPU 自动派生多卡进程。以 `config/dspark/dspark_qwen3_4b.py` 为例，核心模型与训练架构参数如下：

| 配置字段 | 设定值 | 架构与工程说明 |
| --- | --- | --- |
| `target_model_name_or_path` | `"Qwen/Qwen3-4B"` | 绑定的目标语言模型 |
| `block_size` ($\gamma$) | `7` | 单次提议的草稿块最大长度 |
| `num_draft_layers` | `5` | DSpark 并行主干包含 5 层 Transformer 块 |
| `target_layer_ids` | `[1, 9, 17, 25, 33]` | 从目标模型均匀抽取 5 个深浅层的隐藏状态供草稿主干融合 |
| `mask_token_id` | `151669` | 并行块内未知候选位置的掩码占位词元 ID |
| `markov_head_type` / `markov_rank` | `"vanilla"` / `256` | 采用低秩维度为 256 的一阶 Markov 顺序头 |
| `confidence_head_with_markov` | `True` | 置信度头同时融合主干隐藏状态与 Markov 顺序特征 |
| `precision` / `global_batch_size` / `torch_compile` | `"bf16"` / `512` / `True` | 采用 BF16 混合精度、全局批大小 512 并开启 `torch.compile` 算子融合 |

在 `deepspec/trainer/dspark_trainer.py` 中，`Qwen3DSparkTrainer.run_batch()` 执行单步前向传播产出草稿 logits、对齐后的目标 logits 与置信度预测值，随后交由 `compute_dspark_loss()` 完成损失聚合与反向传播。

### 4.3 离线评测基准与验证循环（evaluator.py & draft_ops.py）

离线评估通过 `scripts/eval/eval.sh` 启动，覆盖数学推理（GSM8K、MATH500、AIME25）、代码生成（MBPP、HumanEval、LiveCodeBench）与开放对话（MT-Bench、Alpaca、Arena-Hard-v2）三大类共 9 个主流基准。

在 `deepspec/eval/dspark/evaluator.py` 与 `draft_ops.py` 中，单步推测验证的代码执行流清晰对应了第 2 节所述的模型机制：
1. `forward_dspark_draft_block()` 接收当前锚点词元、位置编码、草稿 KV Cache 及目标模型多层隐藏状态，以 `is_causal=False` 的双向/块内并行模式运行 5 层草稿主干；
2. `build_dspark_proposal()` 先计算基础 logits，再调用 `sample_draft_tokens()` 触发 Markov/RNN 头的逐步条件采样，随后通过 `_predict_confidence_logits()` 与 `_confident_prefix_length()` 截断产出最终提议序列；
3. 目标模型对提议前缀执行并行验证，并由 `ConfidenceHeadRecorder`（位于 `confidence_head.py`）按候选位置分桶统计期望校准误差（ECE）、AUROC、Brier 分数以及用于绘制可靠性图（`reliability_diagram.png`）的分位点数据。

## 5. 离线实验精读：接受长度、后缀衰减与校准分析

### 5.1 多模型与多领域主实验对比（Table 1）

为了纯粹检验草稿模型本身的生成质量，论文在 Table 1 的离线主实验中关闭了动态置信度截断，令 Eagle3、DFlash 与 DSpark 在固定提议长度下直接比拼每轮平均接受长度 $\tau$（越高越好）：

![Main speculative decoding results](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-table1-main-results.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Table 1.*

横向对比四个目标模型（Qwen3-4B/8B/14B、Gemma4-12B）在 9 个基准上的表现，可以得出三点明确结论：
- **DSpark 全线刷新最高接受长度**：在所有目标模型与所有任务领域上，DSpark 均显著超越 Eagle3 与 DFlash。例如在 Qwen3-4B 的 GSM8K 上，DSpark 达到 `6.11`（DFlash 为 `5.40`，Eagle3 为 `5.14`）；在难度更高的开放对话 Alpaca 上，DSpark 达到 `3.54`（DFlash 为 `2.96`，Eagle3 为 `2.26`）。
- **任务结构化程度决定接受长度上限**：数学（Math）与代码（Code）任务的语法约束强、局部确定性高，各模型的平均接受长度普遍在 `4.5`～`6.5` 之间；而开放式闲聊（Chat）存在广泛的同义表述与多分支走向，接受长度普遍回落至 `2.2`～`3.8` 区间。
- **兼具高容量首词与连贯后缀的双重红利**：DSpark 对 DFlash 的稳定领先直接证明了轻量顺序头对修复后缀冲突的有效性；而它对自回归 Eagle3 的大幅领先，则得益于并行主干能够堆叠更深网络（5 层 vs 浅层自回归），从而大幅提高了决定全块命运的首个位置预测准确率。

### 5.2 逐位置条件接受率与后缀衰减实证（Fig. 2）

为了把“当前位置自身预测变差”与“被前序位置拒绝所连累”区分开来，Fig. 2 统计了更为精细的**逐位置条件接受率（position-wise conditional acceptance rate）**——即仅在前 $k-1$ 个位置全部通过验证的条件下，考察第 $k$ 个位置被目标模型接受的概率：

![Position-wise conditional acceptance](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig2-position-wise-conditional-acceptance.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 2.*

Fig. 2 直观揭示了三类模型的内在动力学差异：纯并行的 DFlash 凭借 5 层深主干，在位置 1（Position 1）的起点明显高于浅层自回归的 Eagle3，但随着位置后移（从位置 2 到位置 6），由于缺乏对已采样前缀的条件感知，其条件接受率出现陡峭的**后缀衰减（suffix decay）**；相反，Eagle3 虽在位置 1 起点偏低，但随着已采样前缀逐步消除局部歧义，其后序位置的条件接受率保持平稳甚至微幅上扬；**DSpark 的曲线则成功兼具了两者之长**：在位置 1 保持了与 DFlash 相同的高起点，而在后续位置通过半自回归顺序头显著拉平了衰减斜率，在全部位置上对 DFlash 形成包络优势。

### 5.3 Drafter 深度、提议长度与顺序头开销消融（Fig. 3 & Fig. 4）

Fig. 3 与 Fig. 4 进一步从模型深度与提议长度两个维度评估了半自回归设计的性价比：

![Effect of drafter depth](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig3-drafter-depth.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 3.*

![Effect of proposal length and latency overhead](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig4-proposal-length-latency.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 4.*

在 Fig. 3 的深度消融中（对比 1 层、2 层与 5 层主干），仅使用 **2 层主干的 DSpark 即可在平均接受长度上超越 5 层主干的纯并行 DFlash**。这说明块内顺序依赖缺失是无法单靠盲目加深并行网络来弥补的结构性缺陷，引入显式的局部条件修正远比堆叠并行层高效。

在 Fig. 4 的提议长度扩展实验中（将块大小从 4 逐步扩展至 16），随着候选块变长，DSpark 相对 DFlash 的接受长度优势持续扩大；同时最右侧的延迟对比面板表明，顺序修正头带来的额外推理耗时仅占 **`0.6%` 至 `1.3%`**，在硬件执行层面几乎可以忽略不计。

### 5.4 置信度阈值扫描与后验校准分析（Fig. 5 & Fig. 6）

在验证端，Fig. 5 与 Fig. 6 考察了置信度预测头的区分能力与概率校准质量：

![Confidence threshold sweep](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig5-confidence-threshold-sweep.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 5.*

![Reliability diagram on Alpaca](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig6-reliability-diagram-alpaca.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 6.*

观察 Fig. 5 的阈值扫描曲线可以发现，当置信度阈值从 `0`（全量送验）逐步上调时，橙色柱代表的被拒词元（rejected tokens）被大幅削减，而绿色柱代表的已接受词元（accepted tokens）仅轻微受损，使得有效验证命中率（acceptance rate 折线）单调攀升。这证明置信度头能够精准识别并剔除注定被拒的冗余后缀。

Fig. 6 则展示了 Alpaca 数据集上的可靠性校准图（Reliability Diagram）：未经校准的原始置信度头（Raw）虽具备良好的排序区分度（AUROC 达 `0.81`～`0.90`），但曲线位于对角线下方，存在一定程度的过度自信（ECE 在 `3%`～`8%` 之间）；经过简单的后验校准（Post-hoc Calibration）后，预测存活概率与真实经验接受率几乎完美贴合在对角线上，ECE 压降至 **`1%` 左右**，为 Algorithm 1 中的期望吞吐计算 $\tau = \sum (1 + \sum a_{r,j})$ 提供了可靠的数值基础。

## 6. DeepSeek-V4 线上服务前沿与负载自适应验证

### 6.1 吞吐与单用户 TPS 的帕累托前沿扩展（Fig. 7）

为了检验 DSpark 在真实工业级大模型服务中的端到端收益，论文在 **DeepSeek-V4-Flash** 与 **DeepSeek-V4-Pro**（预览版生产推理引擎）上接入了真实线上用户流量，对比了生产环境原有的单步多词元预测基线（MTP-1）与 5 层草稿主干的 DSpark-5：

![Throughput vs TPS](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig7-throughput-vs-tps.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 7.*

为什么原有生产基线只开启了单步预测（MTP-1）而非更长的静态多步预测？作者指出，在缺乏置信度调度的传统方案中，静态拉长草稿会在高并发下急剧膨胀目标模型的验证批大小，反而拖垮单卡聚合吞吐。而接入具备硬件感知调度的 DSpark-5 后，横轴单用户生成速率（TPS，tok/s/user）与纵轴单卡聚合吞吐（Throughput，tok/s/GPU）构成的帕累托前沿（Pareto Frontier）实现了大幅外扩：
- **在常规高交互 SLA 锚点（80 tok/s/user）下**：DSpark 将 DeepSeek-V4-Flash 的单卡聚合吞吐提升了 **`+51%`**，将 DeepSeek-V4-Pro 的聚合吞吐提升了 **`+52%`**。
- **在极严苛交互 SLA 锚点（120 tok/s/user）下**：此时 MTP-1 已逼近其物理延迟极限，只能维持极低的并发度，因此 DSpark 展现出高达 `661%`（V4-Flash）与 `406%`（V4-Pro）的吞吐倍率，实质上解锁了基线无法企及的高速交互服务区。
- **在匹配的实用吞吐水平下**：DSpark 为 DeepSeek-V4-Flash 带来了 **`60%`～`85%`** 的单用户生成速度提升，为 DeepSeek-V4-Pro 带来了 **`57%`～`78%`** 的单用户速度提升。

### 6.2 并发负载自适应的验证预算分配（Fig. 8）

Fig. 8 进一步拆解了不同并发请求数（Concurrency）下系统吞吐与每请求平均验证预算（Verification Budget）的动态联动，揭示了 Fig. 7 帕累托前沿外推的内在机理：

![Load-adaptive throughput and verification budgets](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig8-load-adaptive-throughput-budget.webp)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 8.*

观察下排的验证预算曲线可以看到：当系统处于低至中等并发时，目标模型算力充裕，Algorithm 1 自动为每个请求分配 `4`～`6` 个词元的宽裕验证预算，充分榨取推测解码的多词元跳进红利；而当并发请求数持续攀升、目标模型批容量趋于饱和时，调度器平滑地将平均验证预算压降收敛至 `1.5`～`2.5` 个高把握词元，避免低置信度后缀拖慢全批步进速率。正是这种“闲时激进多验、忙时保守精验”的自适应弹性，使得 DSpark 在全并发谱系上始终稳居吞吐曲线上方。

## 7. 相关工作谱系与推理系统工程启示

### 7.1 推测解码相关技术路线对比

将 DSpark 放入大模型推测解码的技术演进谱系中，可以清晰定位它与各流派的继承与差异关系：

| 技术流派 | 代表性工作 | 核心提议与验证机制 | DSpark 的演进定位 |
| --- | --- | --- | --- |
| 独立小模型草稿 | 标准 Speculative Decoding | 部署同族小模型逐词元自回归提议，大模型并行验证 | 共享无损拒绝采样框架，但用特征层轻量草稿网络替代独立小模型 |
| 内部多头预测 | Medusa / DeepSeek MTP | 在目标模型顶层挂载多个并行预测头 | 生产环境中以 MTP-1 作为核心线上对比基线 |
| 自回归特征草稿 | EAGLE / Eagle3 | 融合目标模型隐藏状态与词元嵌入，自回归逐词生成草稿 | 克服其 $O(\gamma)$ 串行草稿延迟与浅层网络容量受限问题 |
| 纯并行块草稿 | DFlash | 单次前向并行预测整个草稿块，保持 $O(1)$ 步生成开销 | 继承其并行主干高容量优势，引入半自回归头攻克后缀衰减 |
| 树状多分支验证 | SpecInfer / Medusa Tree | 构造候选词元树并通过树注意力掩码一次验证多条路径 | 聚焦线性前缀的高并发批容量分配，规避复杂树结构在饱和批下的算力膨胀 |

### 7.2 高并发 LLM 推理系统的工程设计原则

超越具体的模型结构，DSpark 为构建工业级大模型推理服务提供了四条普适的系统工程准则：
1. **将目标模型批容量视为首要稀缺预算**：在单请求离线评测中，多验证几个草稿词元似乎毫无代价；但在高并发连续批处理（continuous batching）中，每一个送入目标模型的低置信度草稿词元都在直接挤占同批其他请求的算力配额。推测解码必须具备批级预算意识。
2. **解码调度策略必须与硬件吞吐画像耦合**：最优的提议截断长度不存在脱离负载的全局常数，必须将推理引擎在具体 GPU 上的每秒步数曲线 $\mathrm{SPS}(B)$ 显式编码进在线调度目标。
3. **概率校准（Calibration）是期望收益调度的前提**：当置信度分数不再仅用于单调阈值截断，而是以连乘形式 $a_{r,j} = \prod_{i \le j} c_{r,i}$ 进入期望吞吐公式时，任何未校准的过度自信都会被指数级放大并误导全局资源分配，因此必须将 ECE 纳入模型上线门禁。
4. **以“吞吐-延迟”帕累托前沿取代单一加速比汇报**：大模型服务的真实效能取决于特定 tok/s/user SLA 锚点下的单卡可承载吞吐，使用完整的吞吐-交互速率前沿曲线评估解码算法，远比脱离并发背景的“单样本加速 X 倍”更具工程指导意义。

## 8. 工程落地清单、方法局限性与总结

### 8.1 从离线评测到在线调度的工程接入清单

若工程团队计划在自有推理栈中复刻或接入 DSpark 范式，建议按以下四个阶段分步推进：

1. **数据对齐与草稿模型训练**：使用线上目标模型对业务真实提示词分布自举生成回复文本，确保训练分布无偏移；针对目标模型架构选定均匀分布的深浅隐藏层索引（`target_layer_ids`），联合优化 $\mathcal{L}_{L1}$、$\mathcal{L}_{\mathrm{CE}}$ 与 $\mathcal{L}_{\mathrm{conf}}$（若受限于数十 TB 的全量 Target Cache 存储成本，可优先基于官方开源的预训练权重或采用在线实时提取隐藏状态的蒸馏流水线做领域微调）。
2. **离线多维基准与校准度验收**：在数学、代码、对话等分领域测试集上统计无截断平均接受长度 $\tau$ 及逐位置条件接受率曲线，确认半自回归头有效压制了后缀衰减；同时绘制可靠性图并拟合并部署后验校准参数，确保各位置 ECE 控制在 `1%`～`2%` 以内。
3. **推理引擎剖析与全局调度器集成**：在目标部署硬件上离线压测目标模型在不同验证批大小 $B$ 下的执行耗时，生成轻量查找表 $\mathrm{SPS}(B)$；在连续批处理调度器（如 SGLang / vLLM 调度循环）中接入 Algorithm 1 的贪心前缀准入逻辑，并配置异常熔断回退机制（当并发极端爆表或置信度异常时平滑退回单步解码）。
4. **分并发等级与 SLA 锚点的线上灰度验收**：在灰度流量上分并发桶监控首词元延迟（TTFT）、单用户 TPS 分布、单卡聚合吞吐以及实际触发的平均验证长度 $\bar{\ell}$，确认系统在低并发下充分放大单步接受长度、在高并发下自动收缩验证预算且不劣化尾延迟。

### 8.2 方法边界与核心结论

在借鉴 DSpark 时，同样需要客观把握其适用边界：一方面，草稿模型与目标模型高度绑定，一旦目标模型权重更新、或者从常规对话切换至长思维链（thinking mode）与特定垂直领域分布，草稿模型的接受率与置信度校准度均可能发生漂移，需要配套增量微调与重校准机制；另一方面，在高度发散、创造性极强或词表分布极度平坦的生成任务上，目标分布本身的熵极高，任何推测解码所能获得的平均接受长度上限都会自然收窄。

回顾全文，DSpark 的核心价值在于它跳出了单纯在草稿模型内部堆叠算子的局部视角，以 **“Draft Better（并行主干 + 半自回归顺序头缓解后缀衰减）”** 与 **“Verify Smarter（校准置信度 + 硬件感知贪心前缀调度）”** 的双轮驱动，打通了从词元级分布对齐到高并发集群吞吐帕累托前沿演进的完整闭环，为现代大模型在线推理加速树立了清晰的工程范式。

## 参考资料

- [DeepSeek-AI/DeepSpec GitHub](https://github.com/deepseek-ai/DeepSpec)
- [DSpark paper in DeepSpec](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf)
- [deepseek-ai/dspark_qwen3_4b_block7 checkpoint](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7)
- [DFlash: Efficient Speculative Decoding via Dynamic-Length Parallel Drafting](https://arxiv.org/abs/2602.06036)
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
- [SpecForge](https://github.com/sgl-project/SpecForge)
- [SGLang](https://github.com/sgl-project/sglang)
- [Qwen3](https://github.com/QwenLM/Qwen3)
- [Gemma](https://github.com/google-deepmind/gemma)

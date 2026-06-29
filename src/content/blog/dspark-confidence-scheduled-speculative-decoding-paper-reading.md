---
title: "DSpark 论文精读：置信度调度推测解码、半自回归生成与大模型推理加速"
description: "精读 DeepSeek DSpark 如何通过半自回归 draft model、confidence-scheduled verification 和 DeepSpec 训练评估代码改进大模型 speculative decoding 的速度、接受率与吞吐边界"
pubDate: "2026-06-29T14:12:56+08:00"
updatedDate: "2026-06-29T14:12:56+08:00"
tags:
  - "Paper Reading"
  - "LLM Inference"
  - "Speculative Decoding"
  - "Semi-Autoregressive Generation"
  - "DeepSeek"
  - "Systems"
draft: false
---

Xin Cheng、Xingkai Yu、Chenze Shao、Jiashi Li、Yunfan Xiong 等来自 Peking University 与 DeepSeek-AI 的作者在 2026 年发布的 **DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation**，是一篇非常系统工程取向的大模型推理加速报告。

它讨论的问题不是“怎样让小模型预测下一个 token 更准”这么窄，而是一个更接近真实线上 serving 的问题：

> 当大模型服务处在高并发、严格交互延迟和有限 batch capacity 下，speculative decoding 应该把多少 draft token 送给 target model 验证？

传统 speculative decoding 的直觉是：draft model 越快、一次 draft 越长，target model 每轮验证接受的 token 越多，整体速度越快。但 DSpark 指出，真实系统里还存在另一个约束：**target verification 的 batch 不是免费的**。如果一批请求中有很多低置信度 draft suffix，盲目把它们送去验证会消耗宝贵的 target batch capacity，导致吞吐下降，甚至让服务无法满足严格 TPS SLA。

DSpark 的核心答案可以压缩成两句话。

第一，draft 侧用 **semi-autoregressive generation**：用重一点的 parallel backbone 保持 $O(1)$ 生成长块的吞吐优势，再用轻量 sequential head 在块内引入依赖，缓解 parallel drafter 的 suffix decay。

第二，verification 侧用 **confidence-scheduled verification**：为每个 draft 位置预测被 target 接受的概率，再结合当前 serving engine 的吞吐曲线，动态选择每个请求要验证的 prefix 长度。

这篇报告最值得读的地方，是它把“模型结构”“接受率校准”“硬件吞吐曲线”“线上 SLA”放在同一个框架里讨论。本文会按论文和官方 DeepSpec 代码两条线精读：先拆 DSpark 的算法，再对照 `deepseek-ai/DeepSpec` 中的训练、评估、配置和核心实现。本文不声称复现 DSpark 训练，不构建默认约 38TB 的 Qwen3-4B target cache，也不复现 DeepSeek-V4 的生产 serving 结果。

本文插入的图表来自用户提供的本地 PDF 和官方 DeepSpec 仓库中的 `DSpark_paper.pdf`，仅做等比例裁切展示，并在图注标注来源。

## 1. 论文信息与一句话贡献

论文基本信息如下。

| 项目 | 内容 |
| --- | --- |
| 题名 | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation |
| 作者 | Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang |
| 机构 | Peking University, DeepSeek-AI |
| PDF | 33 页技术报告 |
| 官方代码 | `deepseek-ai/DeepSpec` |
| 开源内容 | DeepSpec 训练/评估代码、DSpark/DFlash/Eagle3 configs、公开 benchmark 数据、released checkpoints |
| 未开源或不可复现部分 | DeepSeek-V4 production traffic、线上 serving engine、完整硬件感知 scheduler 生产实现、SPS profiling 环境 |

一句话概括 DSpark：

> DSpark 把 speculative decoding 的优化目标从“让 drafter 一次吐出更多 token”推进到“让 drafter 生成更可接受的 prefix，并把 target verification 预算分配给最可能存活的 token”。

这句话里有三个关键词。

第一是 **prefix**。Speculative decoding 的 target verification 接受的是连续前缀。如果第一个 draft token 被拒绝，后面 token 再好也无法在这一轮直接贡献速度。因此 draft block 的位置越靠前，价值越高；越靠后的 token 越容易被前面 rejection 拖累。

第二是 **confidence**。DSpark 的 confidence head 不是预测 token probability 本身，而是估计“这个 draft token 会不会被 target 接受”。它关心 draft distribution 与 target distribution 的接近程度，也关心 prefix survival 的累积效果。

第三是 **hardware-aware**。如果系统空闲，验证稍低置信度 token 的机会成本不高；如果系统已经高并发，低置信度 token 会挤占 target batch capacity。DSpark 的 scheduler 把 serving engine 的 step-per-second 曲线 $\mathrm{SPS}(B)$ 放进目标函数，而不是只看模型侧接受率。

## 2. 论文与代码状态

DSpark 的官方代码位于 [DeepSeek-AI/DeepSpec](https://github.com/deepseek-ai/DeepSpec)。README 把 DeepSpec 定义为一个 speculative decoding draft model 的 full-stack codebase，覆盖数据准备、draft model 实现、训练代码和评估脚本。

DeepSpec 当前支持三类 draft model：

| Algorithm | 论文/来源 | DeepSpec 中的作用 |
| --- | --- | --- |
| DSpark | 本文主角 | 半自回归 drafter + confidence head |
| DFlash | parallel drafter baseline | DSpark 的 parallel backbone 设计来源之一 |
| Eagle3 | autoregressive drafter baseline | 用于对比 $O(\gamma)$ sequential drafting |

README 还发布了 Table 1 对应的 Hugging Face checkpoints，包括 Qwen3-4B、Qwen3-8B、Qwen3-14B 与 Gemma4-12B 目标模型下的 Eagle3、DFlash 和 DSpark checkpoints。以 DSpark/Qwen3-4B 为例，对应仓库是 [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7)。

这点很重要：本文不是只读 PDF。DeepSpec 让我们能看到论文里若干概念如何落到代码：

| 论文概念 | 代码位置 | 说明 |
| --- | --- | --- |
| DSpark training config | `config/dspark/dspark_qwen3_4b.py` 等 | block size、draft layers、target layer ids、Markov head、confidence loss 权重 |
| Semi-autoregressive head | `deepspec/modeling/dspark/markov_head.py` | `VanillaMarkov`、`GatedMarkovHead`、`RNNHead` |
| DSpark model | `deepspec/modeling/dspark/qwen3/modeling.py`、`gemma4/modeling.py` | parallel backbone、logits、confidence head、target hidden states |
| Training loss | `deepspec/modeling/dspark/loss.py` | CE、L1 distribution matching、confidence BCE、accept-rate metrics |
| Trainer | `deepspec/trainer/dspark_trainer.py` | 构建 DSpark draft model 并调用 loss |
| Eval proposal | `deepspec/eval/dspark/draft_ops.py` | 生成 draft block、采样 token、按 confidence threshold 截断 |
| Eval loop | `deepspec/eval/dspark/evaluator.py` | target/draft 模型加载、verification loop、confidence recorder |
| Confidence metrics | `deepspec/eval/dspark/confidence_head.py` | ECE、AUROC、Brier、reliability diagram |

但代码边界也必须讲清。论文 Algorithm 1 描述的是高并发 production serving 中的硬件感知全局 prefix scheduler；DeepSpec 开源评估代码主要暴露的是单请求 speculative decoding、静态 `confidence_threshold` 和 confidence metrics。也就是说，开源代码能帮助理解 DSpark 的模型和离线评估，但不等于完整复现 DeepSeek-V4 serving controller。

## 3. 背景：为什么自回归解码慢

大语言模型的标准解码是 autoregressive generation。给定上下文 $x_{1:t}$，模型计算：

$$
p_\theta(x_{t+1}\mid x_{\le t})
$$

采样或贪心选择 $x_{t+1}$ 之后，再把它拼回上下文，继续计算：

$$
p_\theta(x_{t+2}\mid x_{\le t+1})
$$

这个过程的核心瓶颈是 **每个输出 token 至少需要一次 target model forward**。即使 KV cache 已经复用了历史 key/value，新的 token 仍然要经过模型的每一层。对用户来说，体感速度近似由 token/s/user 决定；对服务商来说，成本还取决于 token/s/GPU、batching、并发、显存、调度和 SLA。

大模型线上服务里的矛盾通常是：

| 目标 | 需求 | 代价 |
| --- | --- | --- |
| 低延迟 | 单请求尽快返回 token | batch 变小，GPU 利用率下降 |
| 高吞吐 | 多请求合批，提高 GPU 利用率 | 单请求排队和等待增加 |
| 高质量 | 用更大的 target model | 每步 forward 更贵 |
| 强交互 | 提高 tok/s/user | 并发承载能力下降 |

Speculative decoding 试图绕开其中一部分矛盾：用便宜 draft model 先猜多个 token，再让 expensive target model 一次并行验证这些 token。如果多个 draft token 被接受，target model 每次 forward 就能推进多个输出 token。

## 4. Speculative Decoding 基础

经典 speculative decoding 有两个模型：

- **Target model**：用户真正想要的模型，分布记为 $p$。
- **Draft model**：便宜、快速的模型，分布记为 $q$。

在一轮 decoding 中，draft model 先生成 $\gamma$ 个候选 token：

$$
\hat{x}_{t+1:t+\gamma}
$$

随后 target model 对这段候选进行并行验证。直觉上，如果 draft model 和 target model 在这些 token 上的分布足够接近，就可以接受多个 token；一旦某个位置拒绝，后续 token 也不能继续作为本轮 accepted prefix 使用。

因此一个关键指标是 **accepted length**，常记为 $\tau$。如果每轮平均接受更多 token，那么 target model forward 次数减少，生成速度提高。

但 speculative decoding 不是随便让 draft model 猜，也不是用 draft 结果替代 target 结果。正确算法必须保持 target distribution 的一致性。也就是说，用户最终看到的采样分布仍然应该等价于 target model，而 draft model 只承担“提出候选、帮助加速”的角色。

这也解释了为什么 DSpark 关注的是 prefix survival，而不是普通 next-token accuracy。位置 $j$ 的 token 只有在 $1,\ldots,j-1$ 全部接受之后才有机会被验证成功。第一个 token 的错误会让整个 block 的后缀收益直接归零。

## 5. 现有 drafter 的两难

论文把现有 draft model 的设计困境讲得很清楚：autoregressive drafter 和 parallel drafter 各有问题。

### Autoregressive Drafter

Eagle3 这类 autoregressive drafter 按 token 顺序生成 draft：

$$
q(\hat{x}_{t+1:t+\gamma}\mid x_{\le t})
=\prod_{j=1}^{\gamma} q(\hat{x}_{t+j}\mid x_{\le t},\hat{x}_{t+1:t+j-1})
$$

优点是自然建模块内依赖，后面 token 可以条件于前面已经采样出来的 draft token。缺点是 draft latency 随 $\gamma$ 增长，近似是 $O(\gamma)$。如果 draft model 自身也要跑多次 forward，长 draft block 的收益会被 draft 开销吃掉。

### Parallel Drafter

DFlash 这类 parallel drafter 一次 forward 生成整个 block：

$$
q(\hat{x}_{t+1:t+\gamma}\mid x_{\le t})
\approx \prod_{j=1}^{\gamma} q_j(\hat{x}_{t+j}\mid x_{\le t})
$$

优点是吞吐高，block 生成近似 $O(1)$。缺点是每个位置的预测缺少对前面 sampled token 的条件依赖。遇到多模态上下文时，parallel drafter 可能在不同位置选择不兼容的 continuation。论文举的直觉是，前缀有多种合理补全时，独立位置可能混合不同路径，形成 incoherent suffix。

这种现象会表现为 **suffix decay**：第一个 token 的条件接受率可能很高，但越往后越容易下降。

### DSpark 的折中

DSpark 的设计目标是：

- 保留 parallel backbone 的高 capacity 和 $O(1)$ 长块生成优势。
- 用轻量 sequential head 补上块内依赖。
- 用 confidence head 估计每个位置的接受风险。
- 用 scheduler 避免低置信度 suffix 浪费 target verification。

这就是“draft better, verify smarter”的含义。

## 6. DSpark 总览：Draft Better + Verify Smarter

![DSpark architecture and decoding cycle](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig1-architecture-cycle.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 1.*

Fig. 1 展示了 DSpark 一轮 decoding 的主要流程。为了便于理解，可以把它拆成三步。

第一步，target model 先运行一步，得到 anchor token。图中 prompt tokens 是 `A B C`，target model 先生成 `D`。这个 `D` 不是 draft model 猜出来的，而是 target model 真实推进出来的 token，因此它可以作为后续 drafting 的可靠锚点。

第二步，DSpark 用 `D` 作为输入进入 drafting phase。Parallel block 先生成多个位置的 hidden states 和 base logits，Sequential block 再引入块内依赖，生成 `E F G H` 这类 draft tokens，并输出对应 confidence scores $c_1,\ldots,c_4$。

第三步，Hardware-Aware Prefix Scheduler 根据 confidence scores 和系统吞吐曲线，决定只验证 prefix `E F G`，丢弃低置信度 token `H`。target model 并行验证该 prefix，接受 `E` 和 `F`，拒绝 `G`，并生成纠正 token `G^\ast`。

这个流程背后的关键是：DSpark 不把“最长 draft block”当作目标，而是把 **target verification 的有效产出** 当作目标。低置信度 token 如果大概率会被拒绝，它不只是“没贡献”，还会挤占 batch capacity。

## 7. 半自回归生成：为什么一点点顺序依赖很值钱

论文的 semi-autoregressive generation 可以理解为“parallel backbone + lightweight sequential correction”。

Parallel stage 产生每个位置的 hidden state：

$$
h_1,\ldots,h_\gamma
$$

并得到 base logits：

$$
U_1,\ldots,U_\gamma
$$

如果直接从 $U_j$ 采样每个位置，就回到了普通 parallel drafter。DSpark 进一步加入 sequential head，让第 $j$ 个位置的 logits 可以依赖前一个 sampled token：

$$
\tilde{U}_j = U_j + b(\hat{x}_{j-1}, h_j)
$$

其中 $b(\cdot)$ 是一个轻量 bias function。它不需要像完整 transformer layer 那样重新计算整个上下文，只需要在 draft block 内提供一个局部条件信号。

这个设计的价值在于，speculative decoding 的收益对 prefix 非常敏感。只要 sequential head 能减少后缀不一致，哪怕它很轻，也可能显著提高 accepted length。论文 Fig. 3/4 的消融正是在证明这一点。

## 8. Markov / RNN Head 代码精读

DeepSpec 中 semi-autoregressive head 的核心文件是 `deepspec/modeling/dspark/markov_head.py`。它提供三类实现。

| 类 | 直觉 | 代码行为 |
| --- | --- | --- |
| `VanillaMarkov` | 用上一个 token 查 embedding，再投影成 vocabulary bias | `markov_w1` 把 token 映射到低秩 latent，`markov_w2` 投到 vocab |
| `GatedMarkovHead` | 根据当前位置 hidden state 调节 token bias | 拼接 hidden state 和 prev token embedding，生成 gate |
| `RNNHead` | 在 block 内维护 recurrent state | 用 GRU-like step 让后续位置访问更长的 sampled prefix |

`VanillaMarkov` 的基本形式可以写成：

$$
b_j = W_2 W_1[\hat{x}_{j-1}]
$$

然后修正 base logits：

$$
\tilde{U}_j = U_j + b_j
$$

`GatedMarkovHead` 多了一个 gate：

$$
g_j=\sigma(W_g[h_j;W_1[\hat{x}_{j-1}]])
$$

$$
b_j=W_2(g_j\odot W_1[\hat{x}_{j-1}])
$$

`RNNHead` 则维护状态 $s_j$：

$$
s_j = \mathrm{RNN}(s_{j-1}, W_1[\hat{x}_{j-1}], h_j)
$$

$$
b_j = W_2(o_j)
$$

代码里的 `sample_block_tokens()` 体现了 DSpark 的核心行为：它不是一次性独立采样所有位置，而是在一个 Python loop 里逐位置采样，下一位置把上一位置 sampled token 当作条件信号。不过这个 loop 只运行轻量 head，而不是每个位置都跑完整 draft transformer，因此 overhead 很小。

这一点也解释了论文标题里的 semi-autoregressive：它不是完全自回归 draft model，而是在 parallel backbone 之后加一个低成本顺序依赖层。

## 9. Confidence Head：预测的不是 token probability

DSpark 的 confidence head 目标容易被误解。它不是简单输出“这个 token 的 softmax probability 很高，所以值得验证”。真正要预测的是 **target model 是否会接受这个 draft token**。

在训练时，如果有 draft distribution $q$ 和 target distribution $p$，论文和代码使用的接受率近似可以写为：

$$
\alpha = 1-\frac{1}{2}\lVert p-q\rVert_1
$$

DeepSpec 的 `loss.py` 中，`_compute_accept_rate_3d()` 正是对 draft logits 和 target logits 做 softmax 后计算：

$$
\alpha = 1.0 - 0.5\sum_v |p(v)-q(v)|
$$

并 clamp 到 $[0,1]$。

confidence head 的训练目标是 BCE：

$$
\mathcal{L}_{conf}
=\mathrm{BCEWithLogits}(\hat{c}, \alpha)
$$

代码里 `outputs.confidence_pred` 对齐 `accept_rate_3d.detach()`，并记录三类诊断：

- `confidence_abs_error`：置信度绝对误差。
- `confidence_bias`：整体偏高还是偏低。
- `confidence_cumprod_bias`：prefix survival 累积概率的偏差。

这第三项尤其重要。单点 confidence 稍微偏高，累积到 prefix survival 后可能会被放大：

$$
a_{r,j}=\prod_{i\le j}c_{r,i}
$$

如果每个 $c_i$ 都过度自信，scheduler 会错误地把低质量 suffix 放进 verification batch。

## 10. Hardware-Aware Prefix Scheduler

![Hardware-Aware Prefix Scheduler](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-algorithm1-hardware-aware-prefix-scheduler.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Algorithm 1.*

Algorithm 1 是 DSpark 从“离线接受率提升”走向“线上 serving 可用”的关键。

设当前 batch 中有 $R$ 个 active requests。第 $r$ 个请求的 draft block 有 $\gamma$ 个位置，confidence sequence 是：

$$
c_{r,1},c_{r,2},\ldots,c_{r,\gamma}
$$

由于 speculative decoding 只能接受连续 prefix，第 $j$ 个 token 的 survival probability 是：

$$
a_{r,j}=\prod_{i\le j} c_{r,i}
$$

如果给请求 $r$ 分配 verification length $\ell_r$，那么 target model 这一轮看到的 token batch size 是：

$$
B=\sum_{r=1}^{R}(1+\ell_r)
$$

这里的 $1$ 对应每个请求至少要 target model 继续生成或验证的 anchor/next token，$\ell_r$ 是额外 draft prefix。

期望接受 token 数是：

$$
\tau=\sum_{r=1}^{R}\left(1+\sum_{j=1}^{\ell_r}a_{r,j}\right)
$$

如果 serving engine 在 batch size $B$ 下的 step throughput 是 $\mathrm{SPS}(B)$，系统级期望输出吞吐就是：

$$
\Theta = \tau \cdot \mathrm{SPS}(B)
$$

DSpark 的 scheduler 要做的事是：

$$
\max_{\ell_1,\ldots,\ell_R}\Theta
$$

表面上这像一个组合优化问题，但 prefix survival $a_{r,j}$ 对 $j$ 单调不增。Algorithm 1 因此把所有候选 token $(r,j)$ 按 $a_{r,j}$ 从高到低排序，沿着 greedy admission path 逐个加入 verification batch，并在 $\Theta$ 不再增加时停止。

这个设计有两个很强的工程含义。

第一，scheduler 的输入不只是模型 confidence，还包括 engine 的 $\mathrm{SPS}(B)$ 曲线。这个曲线可以在 engine 初始化时 profile 成轻量表。

第二，scheduler 做的是 batch-level 全局分配。某个请求的第三个 draft token 可能比另一个请求的第一个 draft token 更不值得验证，反之亦然。统一排序让高价值 token 先占用 target capacity。

## 11. 为什么不是固定阈值

很多 speculative decoding 方法会用固定 confidence threshold，例如只验证 $c_j>0.7$ 的 prefix。DSpark 认为这在高并发生产系统里不够。

原因是 threshold 的最优值依赖负载。

| 系统状态 | 验证低置信 token 的机会成本 | 更合理策略 |
| --- | --- | --- |
| 低并发、target capacity 空闲 | 较低 | 可以验证更长 prefix，争取更多 accepted tokens |
| 中等负载 | 中等 | 按 survival probability 和 SPS 曲线权衡 |
| 高并发、batch capacity 紧张 | 很高 | 严格裁掉低置信 suffix，避免拖垮吞吐 |

固定阈值把这些状态压成一个常数。DSpark 的 Algorithm 1 则把系统负载纳入优化目标。当并发上升时，$\mathrm{SPS}(B)$ 会改变，继续增加 verification batch size 可能让每步耗时变大，导致 $\Theta$ 下降。此时即使某些 token 的 confidence 还不错，也未必值得验证。

这也是 DSpark 与“离线接受率提升论文”的重要区别。它关心的是 serving frontier：在不同 throughput 和 tok/s/user 约束下，系统能不能维持更好的 Pareto frontier。

## 12. Training Objective：CE + L1 + Confidence

DeepSpec 的 DSpark loss 位于 `deepspec/modeling/dspark/loss.py`。配置文件 `config/dspark/dspark_qwen3_4b.py` 中默认权重是：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `ce_loss_alpha` | `0.1` | token-level CE |
| `l1_loss_alpha` | `0.9` | draft/target 分布 L1 对齐 |
| `confidence_head_alpha` | `1.0` | confidence BCE |
| `loss_decay_gamma` | `4.0` | position decay |

CE 项让 draft logits 学习目标 token：

$$
\mathcal{L}_{CE}
=-\log q_\phi(x^\ast_j\mid \cdot)
$$

L1 项让 draft distribution 靠近 target distribution：

$$
\mathcal{L}_{L1}
=\lVert q_\phi(\cdot)-p_\theta(\cdot)\rVert_1
$$

confidence 项预测 accept rate：

$$
\mathcal{L}_{conf}
=\mathrm{BCEWithLogits}(\hat{c}_j,\alpha_j)
$$

整体可以写成：

$$
\mathcal{L}
=\lambda_{CE}\mathcal{L}_{CE}
+\lambda_{L1}\mathcal{L}_{L1}
+\lambda_{conf}\mathcal{L}_{conf}
$$

代码还对 block 内位置加入衰减：

$$
w_j=\exp(-j/\gamma_{decay})
$$

这符合 speculative decoding 的 prefix 性质：越靠前的位置越重要。第一个 token 的质量直接决定整个 block 是否有机会贡献收益。

## 13. DeepSpec 数据管线

DeepSpec 的数据准备文档在 `scripts/data/README.md`，默认以 `Qwen/Qwen3-4B` 为 target model。流程分三步：

1. 下载并切分 prompt 数据。
2. 用 target model 重新生成 assistant answers。
3. 预计算 target cache，供训练读取。

默认 source dataset 是 [mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend)。第二步需要一个 OpenAI-compatible inference engine，README 示例使用 SGLang，但也说明可替换为 vLLM、TGI 等，只要暴露兼容 `/v1` endpoint。

第三步是复现门槛最高的地方：target cache 会存储训练集中 token 对应的 target hidden states。README 明确提示，默认 `Qwen/Qwen3-4B` 设置下 target cache 大约需要 **38TB** 存储。

这解释了为什么本文不跑训练。DSpark 不是一个“下载代码、单卡跑半小时”的教学项目。它开源了训练链路，但默认配置面向多 GPU、大存储和完整 target-cache pipeline。

## 14. 训练入口与配置

训练入口是 `scripts/train/train.sh`，它调用：

```bash
python train.py \
  --config config/dspark/dspark_qwen3_4b.py \
  --opts "data.target_cache_path=${target_cache_dir}"
```

脚本注释说明这里不是标准 `torchrun` 语义。`train.py` 会根据 visible GPUs 自己 spawn worker；`RANK/WORLD_SIZE` 表示 node rank/node count。默认 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，即单节点 8 GPU。

`config/dspark/dspark_qwen3_4b.py` 的关键配置如下。

| 字段 | 值 | 解释 |
| --- | --- | --- |
| `target_model_name_or_path` | `Qwen/Qwen3-4B` | target model |
| `block_size` | `7` | draft block 最大长度 |
| `num_draft_layers` | `5` | DSpark drafter transformer layers |
| `target_layer_ids` | `[1, 9, 17, 25, 33]` | 从 target model 抽取 hidden states 的层 |
| `mask_token_id` | `151669` | mask token |
| `markov_rank` | `256` | Markov head latent rank |
| `markov_head_type` | `vanilla` | 默认 Markov head |
| `confidence_head_with_markov` | `True` | confidence head 使用 Markov features |
| `precision` | `bf16` | 训练精度 |
| `global_batch_size` | `512` | 全局 batch |
| `torch_compile` | `True` | 开启 torch compile |

Trainer 入口在 `deepspec/trainer/dspark_trainer.py`。`Qwen3DSparkTrainer.run_batch()` 调用模型 forward，随后把 outputs 交给 `compute_dspark_loss()`。这说明 DSpark 的训练代码非常直接：模型负责产出 draft logits、target ids、eval mask、aligned target logits、confidence pred；loss 函数负责组合 CE/L1/confidence。

## 15. 评估入口与 Benchmark

评估入口是 `scripts/eval/eval.sh`：

```bash
python eval.py \
  --target_name_or_path ${target_name_or_path} \
  --draft_name_or_path ${draft_name_or_path}
```

README 列出的 benchmark 包括：

| Domain | Dataset |
| --- | --- |
| Math | GSM8K, MATH500, AIME25 |
| Code | MBPP, HumanEval, LiveCodeBench |
| Chat | MT-Bench, Alpaca, Arena-Hard-v2 |

评估实现的关键在 `deepspec/eval/dspark/evaluator.py`。流程可以概括为：

1. 加载 target model 和 DSpark draft model。
2. 初始化 target hidden states，供 draft model 使用。
3. `forward_dspark_draft_block()` 运行 DSpark backbone。
4. `build_dspark_proposal()` 采样 draft tokens，并按 confidence threshold 截断。
5. target model 执行 verification。
6. 记录 accepted tokens、confidence calibration 和 reliability metrics。

需要注意的是，开源 eval 的 `confidence_threshold` 是静态 threshold。它可以验证 confidence head 是否有区分度，也可以生成 reliability diagram，但不是论文 Algorithm 1 的生产全局 scheduler。

## 16. Table 1 主结果精读

![Main speculative decoding results](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-table1-main-results.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Table 1.*

Table 1 报告的是每轮 decoding 的 accepted length $\tau$，越高越好。目标模型包括 Qwen3-4B、Qwen3-8B、Qwen3-14B 和 Gemma4-12B；draft baselines 是 Eagle3、DFlash 和 DSpark。

论文为了隔离 raw draft quality，在 Table 1 离线评估中禁用了 confidence scheduler，让所有方法使用固定 block proposal。这意味着 Table 1 主要证明的是 **DSpark drafter 质量**，而不是 Algorithm 1 的系统调度收益。

几个结论很清楚。

第一，DSpark 在所有目标模型和所有 domain 上都是最佳。以 Qwen3-4B 为例，DSpark 在 GSM8K 上达到 `6.11`，DFlash 是 `5.40`，Eagle3 是 `5.14`；在 Alpaca 上 DSpark 是 `3.54`，DFlash 是 `2.96`，Eagle3 是 `2.26`。

第二，domain predictability 差异很大。Math 和 code 的 accepted length 普遍高于 chat。结构化任务的后续 token 更容易预测，因此 draft block 更容易被 target 接受；开放式 chat 的多模态性更强，接受长度自然更短。

第三，DSpark 对 DFlash 的优势说明半自回归 head 有实际价值。DFlash 已经是高吞吐 parallel drafter；DSpark 在它之上进一步提升 accepted length，说明 suffix dependency 不是小问题。

第四，DSpark 对 Eagle3 的优势也很有意思。直觉上 autoregressive drafter 应该更会建模块内依赖，但 Eagle3 为了控制 draft latency 通常不能太深。Parallel backbone 可以用更大 capacity 提高第一个位置质量，而第一个位置又对 prefix survival 极其关键。

## 17. Fig. 2：条件接受率揭示 Suffix Decay

![Position-wise conditional acceptance](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig2-position-wise-conditional-acceptance.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 2.*

Fig. 2 的指标是 position-wise conditional acceptance。它不是普通 prefix survival，而是在第 $k$ 个位置只统计前 $k-1$ 个位置已经成功接受的样本，然后看第 $k$ 个 token 是否也被接受。

这个指标隔离了“当前位置本身的预测质量”。如果直接看 prefix survival，后面位置会被前面 rejection 连带惩罚，很难看出到底是当前位置差，还是前面已经失败。

图里有三个现象。

第一，DFlash 在 position 1 很强。特别是在 Math 和 Chat 上，它起点高于 Eagle3。这说明 parallel drafter 因为可以用更深网络，第一 token 的预测质量可能超过浅层 autoregressive drafter。

第二，DFlash 往后出现 suffix decay。它的后续位置不能条件于实际 sampled prefix，因此越往后越容易混合不同模式。

第三，Eagle3 后续位置稳定甚至上升。因为它按顺序生成，已经采样的前缀会降低后续不确定性。

DSpark 的曲线介于两者优点之间：保留第一位置的高 capacity，同时用 sequential head 缓解 suffix decay。

## 18. Fig. 3/4：一点点自回归的性价比

![Effect of drafter depth](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig3-drafter-depth.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 3.*

Fig. 3 比较 drafter depth。绿色是 DSpark，蓝色是 DFlash；点大小或样式表示 1L/2L/5L。

关键结论是：DSpark 增加层数会提升 accepted length，而且 2-layer DSpark 就能超过 5-layer DFlash baseline。这说明顺序依赖不是只能靠堆更多 parallel layers 解决。一个轻量 sequential head 可以比单纯增加 backbone depth 更高效地修正 suffix。

![Effect of proposal length and latency overhead](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig4-proposal-length-latency.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 4.*

Fig. 4 研究 proposal length 和 latency overhead。左侧三个 panel 分别是 Math、Code、Chat，右侧是 latency。

随着 draft length 从 4 增加到 16，DFlash 和 DSpark 都能获得更长 accepted length，但 DSpark 始终高于 DFlash。右侧 latency panel 表明 sequential head 的 overhead 很小，论文标出的增幅大约在 0.6% 到 1.3%。

这张图支持 DSpark 的核心工程判断：如果只需要很少的额外 latency，就能显著改善 suffix consistency，那么 semi-autoregressive head 是值得加入 production drafter 的。

不过也要注意，Fig. 4 仍然是离线或受控实验。线上是否值得用更长 proposal length，还要看 target verification cost 和并发状态。这正是 Algorithm 1 要处理的问题。

## 19. Fig. 5/6：置信度、阈值与校准

![Confidence threshold sweep](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig5-confidence-threshold-sweep.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 5.*

Fig. 5 是 confidence threshold sweep。横轴是 threshold，柱子分为 accepted tokens 和 rejected tokens，线表示 acceptance rate。

当 threshold 为 0 时，相当于固定长度 verification：所有 draft positions 都送给 target model。随着 threshold 提高，低置信度 token 被裁掉，rejected tokens 减少，整体 acceptance rate 上升。

但这不是说 threshold 越高越好。threshold 提高会减少被验证的 token 数。如果太保守，即使 acceptance rate 高，也可能每轮推进 token 太少。正确目标不是最大化 acceptance rate，而是最大化系统吞吐和用户速度。

![Reliability diagram on Alpaca](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig6-reliability-diagram-alpaca.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 6.*

Fig. 6 是 Alpaca 上的 reliability diagram。它显示 raw confidence estimator 有不错的判别力，AUC 大约在 0.81 到 0.90，但存在 overconfidence。经过 post-hoc calibration 后，ECE 从约 3%-8% 降到约 1% 左右。

为什么 calibration 对 DSpark 特别重要？因为 scheduler 使用的是 prefix survival：

$$
a_{r,j}=\prod_{i\le j}c_{r,i}
$$

如果每个 $c_{r,i}$ 都偏高，乘积也会偏高，scheduler 会把本应裁掉的 suffix 放进 verification batch。对单请求来说这只是浪费一次验证；对高并发系统来说，它会污染整个 batch 的资源分配。

DeepSpec 中 `deepspec/eval/dspark/confidence_head.py` 实现了 ECE、AUROC、Brier 和 reliability diagram 的统计。代码中 `PerPositionConfidenceMetrics` 维护 coarse/fine bins，并按位置汇总校准质量。这说明 confidence head 在 DSpark 中不是一个只用于论文画图的附属模块，而是 scheduler 的核心输入。

## 20. DeepSeek-V4 线上部署结果

![Throughput vs TPS](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig7-throughput-vs-tps.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 7.*

Fig. 7 是 DSpark 最有系统价值的一张图。横轴是单用户生成速度 TPS，纵轴是聚合输出吞吐 token/s/GPU。蓝色是 MTP baseline，绿色是 DSpark。

论文在 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 预览版 serving engines 上对比 DSpark-5 和 MTP-1。这里的 MTP-1 是旧 production setup。论文解释说，历史上保留 single-token setup，是因为静态 multi-token drafter 在高并发下可能因为 verification overhead 反而降低 aggregate throughput。

图中 DSpark 的 frontier 明显向外移动。对于 V4-Flash，在 80 tok/s/user SLA anchor 处，DSpark 的 aggregate throughput 提升 51%；在更严格的 120 tok/s/user 处，图中标出 661% throughput。对于 V4-Pro，也有 52% 和 406% 的对应点。

但这些大百分比要谨慎解读。论文自己也提醒，高 SLA 点上 MTP-1 已经接近 operational boundary，只能维持很小并发。因此 661%/406% 更适合解读为“DSpark 扩展了可行 interactivity frontier”，而不是一般场景下稳定有 6 倍吞吐。

更稳健的读法是：在 matched practical throughput levels 下，DSpark 让 V4-Flash 的 per-user generation speed 提升约 60%-85%，让 V4-Pro 提升约 57%-78%。

## 21. Fig. 8：Load-Adaptive Verification Budget

![Load-adaptive throughput and verification budgets](/images/blog/dspark-confidence-scheduled-speculative-decoding/dspark-fig8-load-adaptive-throughput-budget.png)

*Source: Cheng et al., DSpark technical report, DeepSeek-AI / DeepSpec, 2026, Fig. 8.*

Fig. 8 展示了 DSpark 为什么能改善 Fig. 7 的 frontier。上排是吞吐随并发变化，下排是平均 verification budget 随并发变化。

低到中等并发时，DSpark 有足够 target capacity，可以给每个请求分配更长 verification budget，大约 4-6 个 token。这样每次 target forward 有机会接受更多 token，提高吞吐。

随着并发上升，target capacity 变紧张。DSpark 的 scheduler 会逐步降低 verification budget，避免把低置信度 suffix 送进 target batch。相比之下，MTP baseline 的 verification budget 是静态的，不能根据负载平滑调整。

这张图说明 Algorithm 1 的价值不是离线 accepted length，而是 **负载自适应**。系统空闲时多验证，系统忙时少浪费。这个策略特别适合线上 LLM 服务，因为请求长度、prompt 类型、输出不确定性、并发状态都高度动态。

## 22. 论文-代码对照：公开实现与生产算法的边界

这里把 DSpark 论文和 DeepSpec 代码做一次更直接的对照。

### 22.1 论文 Fig. 1 与 `draft_ops.py`

`deepspec/eval/dspark/draft_ops.py` 中的 `forward_dspark_draft_block()` 对应 Fig. 1 中 draft block 的 forward。它接收：

- `draft_input_ids`
- `position_ids`
- `past_key_values_draft`
- `target_hidden_states`
- `start`
- `block_size`

然后调用模型的 `_forward_backbone()`，并设置 `is_causal=False`，体现 parallel block 的特点。

`build_dspark_proposal()` 对应 draft token 采样和 confidence prefix 截断。它先用 `compute_logits()` 得到 base logits，再调用 `sample_draft_tokens()`，后者会进入 Markov/RNN head 的 sequential sampling。随后 `_predict_confidence_logits()` 生成每个位置的 confidence logits，`_confident_prefix_length()` 根据静态 threshold 决定 proposal token count。

这部分代码对 Fig. 1 的 draft better 有很好的对应；但它没有实现 production-level Algorithm 1 的 batch-global greedy scheduler。

### 22.2 论文 semi-autoregression 与 `markov_head.py`

`markov_head.py` 是理解 DSpark 的核心。它清楚展示了“轻量 sequential head”如何落地：

- Training 时 `apply_block_logits()` 使用 teacher-forced token ids 修正每个位置 logits。
- Inference 时 `sample_block_tokens()` 逐步采样，每一步把上一步 sampled token 作为下一步条件。
- RNN head 额外维护 state，使位置 $k$ 能访问更完整的 block prefix 历史。

这比只看论文公式更直观：DSpark 并没有把 parallel drafter 改成昂贵 autoregressive transformer，而是在 logits correction 层加入低秩、低成本依赖。

### 22.3 论文 confidence 与 `confidence_head.py`

论文强调 confidence 要可校准。开源代码中 `ConfidenceHeadRecorder` 和 `PerPositionConfidenceMetrics` 做了三类统计：

- 每个位置的 ECE。
- AUROC。
- Brier score。

这对应 Fig. 6 的 reliability diagram。实现中还有 `RELIABILITY_PLOT_FILENAME = "reliability_diagram.png"`，说明评估脚本可以产出类似论文图的 artifact。

### 22.4 论文训练目标与 `loss.py`

`loss.py` 把 target/draft distribution 的 L1 距离、CE、confidence BCE 放在一起。值得注意的是，confidence target 不是 hard label，而是 soft accept rate：

$$
\alpha = 1-\frac{1}{2}\lVert p-q\rVert_1
$$

这使 confidence head 学到的不是“采样 token 是否刚好匹配某个 ground truth token”，而是 draft distribution 与 target distribution 的接近程度。对 speculative decoding 来说，这比普通 next-token accuracy 更贴近 verification 行为。

### 22.5 论文 Algorithm 1 与开源代码差异

这一点必须明确写清。

论文 Algorithm 1 需要以下信息：

- 当前 batch 内所有 active requests。
- 每个 request 的 confidence sequence。
- serving engine 在不同 batch size 下的 $\mathrm{SPS}(B)$ 曲线。
- 全局 greedy admission path。

DeepSpec 开源 eval 代码则主要服务离线 benchmark：

- 单样本生成。
- 静态 `confidence_threshold`。
- confidence metrics 记录。
- accepted length 统计。

因此，DeepSpec 可以帮助研究者训练和评估 DSpark-style drafter，但不能直接复现 DeepSeek-V4 production scheduler。

## 23. 与相关工作的关系

DSpark 可以放在 speculative decoding 的几个方向中理解。

| 方向 | 代表方法 | 核心思路 | DSpark 的关系 |
| --- | --- | --- | --- |
| 小模型 draft | Speculative Decoding | 小模型顺序生成 draft，大模型验证 | DSpark 仍使用 target verification 框架 |
| 多头预测 | Medusa、MTP | target 模型内部预测多个未来 token | DSpark 对比 MTP-1 production baseline |
| Autoregressive drafter | EAGLE / Eagle3 | draft 逐 token 条件生成 | DSpark 用它作为强 baseline |
| Parallel drafter | DFlash | 一次 forward 生成 block | DSpark 继承 parallel backbone 优点 |
| 树状 speculative | SpecInfer 等 | 提出多分支候选树 | DSpark 关注 prefix scheduler，不走复杂树验证主线 |
| 查找式 draft | Prompt Lookup | 从 prompt/history 中复制候选 | 与 DSpark 训练式 drafter 路线不同 |
| 半自回归生成 | Semi-autoregressive NMT | 在 parallel 与 autoregressive 之间折中 | DSpark 把该思想迁移到 speculative draft |

DSpark 的独特位置在于，它不是单独优化 draft model，也不是单独优化 verification algorithm，而是把两者和 serving hardware curve 连起来。

## 24. 与推理系统工程的关系

从系统工程角度，DSpark 的启发比“accepted length 提升多少”更广。

### 24.1 Batch Capacity 是核心资源

在高并发 LLM serving 中，target model 的 batch capacity 是稀缺资源。一个低置信度 token 如果被送去验证，不只是自己可能被拒绝，还会让同一批里其他更高价值 token 等待。

这类似推荐系统里的曝光预算：不是所有候选都值得进入精排，不是所有 draft suffix 都值得进入 target verification。

### 24.2 解码策略应该看 Load

很多解码算法在单请求 benchmark 上比较速度，但线上系统的最优策略随并发变化。DSpark 的 Algorithm 1 把 $\mathrm{SPS}(B)$ 放进目标函数，说明解码策略应该和 serving engine profile 绑定。

### 24.3 Confidence 要能校准，不只是能排序

如果 confidence 只用于排序，AUC 高可能够用；如果 confidence 要进入期望吞吐公式，calibration 就很重要。Fig. 6 和 `confidence_head.py` 都说明 DSpark 把 calibration 当作可观测指标。

### 24.4 线上速度不是单个数字

Fig. 7 的 frontier 说明 LLM serving 速度应同时报告：

- per-user TPS。
- aggregate token throughput。
- SLA anchor。
- concurrency。
- target model 和 engine 配置。

单独说“加速 2 倍”很容易误导。DSpark 的线上结果更像是一个 Pareto frontier 移动，而不是固定倍率。

## 25. 复现与工程清单

如果要在工程环境里复刻 DSpark-style workflow，至少要检查以下内容。

### 25.1 数据与 Target Cache

- 确认 target model，例如 `Qwen/Qwen3-4B`。
- 准备 prompt dataset，例如 open-perfectblend。
- 用 target model 重新生成 answers，保证 draft 训练分布匹配 target deployment。
- 预计算 target hidden states cache。
- 评估存储成本；默认 Qwen3-4B setting 约 38TB，不适合轻量实验。

### 25.2 Draft Model 训练

- 选择配置：`config/dspark/dspark_qwen3_4b.py` 等。
- 确认 `block_size` 与目标验证长度一致。
- 确认 `target_layer_ids` 与目标模型结构匹配。
- 监控 CE、L1、confidence BCE、position-wise accept rate。
- 保存 checkpoints 到 `~/checkpoints/<project_name>/<exp_name>/step_*`。

### 25.3 离线评估

- 使用 `eval.py` 跑 GSM8K、MATH500、AIME25、MBPP、HumanEval、LiveCodeBench、MT-Bench、Alpaca、Arena-Hard-v2。
- 报告 accepted length $\tau$。
- 按 domain 拆分 math/code/chat。
- 记录 confidence ECE、AUC、Brier。
- 比较 Eagle3、DFlash、DSpark，而不是只报告单模型结果。

### 25.4 Serving 集成

- 实现 target engine 的 $\mathrm{SPS}(B)$ profiling。
- 把 confidence logits 转成 calibrated confidence。
- 实现 batch-level prefix scheduler。
- 记录每轮实际 verification length、accepted length、rejected tokens。
- 按并发分桶监控 throughput 和 TPS。
- 设置 fallback：当 confidence head、scheduler 或 drafter 异常时回退 MTP-1 或普通 autoregressive decoding。

### 25.5 上线验收

- 低并发：DSpark 不应显著增加首 token latency。
- 中并发：aggregate throughput 应优于 baseline。
- 高并发：verification budget 应自动下降，不能让 target batch 爆掉。
- SLA anchor：在不同 tok/s/user 目标下绘制 frontier，而不是只看平均速度。
- 质量一致性：确认 speculative decoding 不改变 target model 输出分布的正确性假设。

## 26. 局限性与批判

DSpark 的结果很强，但不能过度外推。

第一，生产 traffic 不可复现。Fig. 7/8 来自 DeepSeek-V4 live user traffic 和内部 serving engine。外部读者无法复现这些曲线，也无法确认 traffic mix、prompt length、输出长度、batching policy 对结果的影响。

第二，production scheduler 未完整开源。DeepSpec 开源了模型、训练、评估和 checkpoints，但 Algorithm 1 在真实 serving engine 中如何接入、如何 profile $\mathrm{SPS}(B)$、如何处理连续 batching 和 SLA policy，仍然是系统实现细节。

第三，target cache 成本极高。默认 Qwen3-4B cache 约 38TB，使得“从零训练 DSpark”对多数个人和小团队不现实。更现实的路径是使用 released checkpoints 或在小数据、小模型上做方法验证。

第四，domain shift 需要重新训练。README 提醒，如果目标模型运行在 thinking mode 或 domain-specific setting，应该重新 fine-tune draft model。Speculative drafter 对 target model 和 generation distribution 很敏感，不能把一个 checkpoint 当作通用加速器。

第五，accepted length 不是唯一指标。更长 accepted length 通常意味着更快，但真实用户体验还受首 token latency、tail latency、输出质量、服务稳定性和成本影响。DSpark 的线上结果用 frontier 方式报告，这是正确方向，但外部部署仍需重新评估。

第六，confidence calibration 是脆弱环节。Algorithm 1 依赖 prefix survival 的数值可靠性。如果 confidence head 在新 domain 上过度自信，scheduler 可能做出错误分配。

## 27. 推荐阅读路径

如果时间有限，建议按以下顺序读。

1. 先读 Abstract 和 Fig. 1，建立 DSpark 的整体结构。
2. 读 Section 3.1，理解 semi-autoregressive generation 为什么缓解 suffix decay。
3. 读 Section 3.2 和 Algorithm 1，理解 hardware-aware prefix scheduler。
4. 读 Table 1，确认离线 accepted length 的主结果。
5. 读 Fig. 2-4，理解 parallel capacity、suffix decay、sequential head 的消融。
6. 读 Fig. 5-6，理解 confidence threshold 和 calibration。
7. 读 Fig. 7-8，理解线上 serving frontier。
8. 看 DeepSpec README，确认代码能做什么、不能做什么。
9. 看 `config/dspark/dspark_qwen3_4b.py`、`markov_head.py`、`loss.py`、`draft_ops.py`。
10. 最后看 `scripts/data/README.md`，理解为什么完整训练复现成本很高。

## 28. 结论

DSpark 的贡献不只是提出一个新的 draft model，而是把 speculative decoding 的优化目标重新组织了一遍。

传统视角容易问：draft model 能不能一次生成更多 token？DSpark 的回答是：更长 draft block 只有在 target verification 资源不被浪费时才有价值。因此要同时解决两个问题：

- **Draft better**：用 parallel backbone + semi-autoregressive head 提高 prefix 质量，缓解 suffix decay。
- **Verify smarter**：用 calibrated confidence + hardware-aware scheduler，把 target batch capacity 分给最可能存活的 prefix token。

从论文结果看，DSpark 在离线 accepted length 上稳定超过 Eagle3 和 DFlash；从线上部署看，它能在 DeepSeek-V4 serving 中把吞吐-交互 frontier 往外推。更重要的是，它给推理系统提供了一个清晰范式：大模型解码加速不能只看单请求算法，也必须看 batch capacity、负载、校准和 SLA。

对工程实现者来说，DeepSpec 的价值在于把 DSpark 的训练与评估链路公开出来；但它也明确暴露了真实复现成本和 production gap。本文的实践建议是：如果只是学习方法，先用 released checkpoints 和 eval pipeline 理解 accepted length 与 confidence metrics；如果要做生产级改造，必须补上 engine profiling、global scheduler、线上监控和安全回退。

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

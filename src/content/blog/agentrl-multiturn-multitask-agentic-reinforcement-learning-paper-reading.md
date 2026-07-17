---
title: "AgentRL 论文精读：全异步 Rollout-Training、跨策略采样与多任务 Agent 强化学习"
description: "精读 AgentRL 如何通过全异步多轮训练、统一环境控制面、Cross-Policy Sampling、Task Advantage Normalization 和开源训练框架扩展多任务 Agentic RL"
pubDate: "2026-07-15T11:49:06+08:00"
updatedDate: "2026-07-15T11:49:06+08:00"
tags:
  - "Paper Reading"
  - "Reinforcement Learning"
  - "Agentic RL"
  - "Asynchronous RL"
  - "Multi-Task Learning"
  - "Code Reading"
draft: false
---

大语言模型强化学习正在从“对一个 prompt 生成一次答案”转向“让模型在真实环境里连续做事”。一个 WebShop Agent 需要搜索、筛选、比较并购买商品；一个数据库 Agent 要先理解问题，再多次生成 SQL、读取结果和修正查询；一个操作系统 Agent 会在命令行里观察文件、执行命令并处理错误。轨迹由多轮模型生成和环境反馈交替组成，时长可能相差数十倍，奖励分布、动作空间和失败模式也因任务而异。

这使 Agentic RL 同时成为算法问题和分布式系统问题。同步训练会被最慢轨迹拖住；不同环境缺少统一生命周期和函数调用协议；单一策略反复消费自己的采样数据，探索可能快速收缩；多个任务混在一起时，奖励尺度和学习速度差异又会让梯度相互干扰。只替换一个 loss，并不能自动解决这些约束。

**AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework** 的主要价值，正是把这些问题放在同一框架内处理。它在系统侧解耦 rollout 与 learner，用中央 Controller 管理容器化环境；在算法侧通过 Cross-Policy Sampling 引入来自不同策略版本的动作路径，再按任务归一化 Token 级 advantage。论文还公开了训练框架和环境部署框架，使我们可以检查“fully asynchronous”究竟落在哪个队列边界、陈旧策略如何进入采样，以及公开代码是否真的移除了所有等待。

精读后的结论可以先写在前面：

> AgentRL 的长期价值不是某个独立公式，而是把多轮采样、环境控制、参数同步、探索和多任务稳定性组织成一个可执行的 Agentic RL 系统；它的关键边界是，公开 GRPO 实现仍然要求同一 prompt 的完整采样组，Cross-Policy Sampling 也会主动引入 off-policy 数据，因此“全异步”和“跨策略”都必须按具体实现理解。

本文以 arXiv `2510.04206v1` 和官方仓库 commit [`6a73409d`](https://github.com/THUDM/AgentRL/tree/6a73409d31ba695d383b978a8ad3ef400d90c054) 为固定基线。论文方法、当前公开实现和论文之后的仓库扩展会分别标注。本文没有下载模型、启动 Controller、运行容器环境或使用 H800 复现训练，也不把论文关于 AutoGLM 的使用声明视为独立验证过的生产结果。

## 1. 一句话贡献

AgentRL 可以概括为四个相互依赖的部件：

1. **Fully-asynchronous rollout-training**：rollout engine 和训练 worker 使用独立 GPU 资源组并发工作，learner 从完成队列持续取数据，不等待整个大批次同时完成。
2. **Environment deployment framework**：统一函数调用接口、容器化任务 worker 和中央 Controller 管理 session、环境生命周期、路由、观测和奖励。
3. **Cross-Policy Sampling**：在一条多轮轨迹的不同环境 step 使用策略池中的不同模型产生动作，以扩大探索路径；训练时用更新较慢的 stale rollout engine 近似策略池。
4. **Task Advantage Normalization**：GRPO 得到 advantage 后，按任务汇总所有有效 action Token，再分别标准化，降低多任务之间尺度和学习速度的差异。

四者解决的问题不同。异步调度降低硬件等待，环境框架降低任务接入和故障隔离成本，跨策略采样改善探索，任务级归一化处理优化干扰。把其中任何一项单独称为 AgentRL 都会丢失论文的系统设计重点。

## 2. 论文信息、版本与开放边界

| 项目 | 内容 |
| --- | --- |
| 题名 | AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework |
| 作者 | Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, Zhenyu Hou, Zehan Qi, Hanyu Lai, Yifan Xu, Rui Lu, Hongning Wang, Jie Tang, Yuxiao Dong |
| 机构 | Tsinghua University, Z.AI |
| arXiv | `2510.04206v1` |
| 提交日期 | 2025-10-05 |
| PDF | 24 pages |
| DOI | `10.48550/arXiv.2510.04206` |
| 论文许可 | CC BY 4.0 |
| 论文状态 | arXiv v1 preprint；PDF 标注 `Preprint. Under review.` |
| 官方代码 | [THUDM/AgentRL](https://github.com/THUDM/AgentRL) |
| 本文代码基线 | `6a73409d31ba695d383b978a8ad3ef400d90c054`，2026-01-17 |
| 代码许可 | MIT；部分代码源自 Apache-2.0 的 AgentBench 与 VeRL |
| 论文主实验 | Qwen2.5 3B/7B/14B/32B、GLM-4-9B，五个 AgentBench-FC 环境 |

论文使用 NeurIPS 2025 preprint 模板，但模板和录用是两回事。arXiv 页面截至本文写作时只有 v1，也没有正式会议页面，因此本文不写成“NeurIPS 2025 已录用论文”。作者栏注明前两位作者贡献相同，若干作者在 Z.AI 实习期间完成工作；摘要称框架和算法用于构建 AutoGLM，但论文没有公开 AutoGLM 的生产拓扑、流量、成本或故障数据。

官方仓库在论文发布后继续演进。当前 `main` 已加入模块化 trainer、独立评估包、gRPC transport、dashboard、更多状态后端和 Kubernetes 相关代码。它们有助于理解工程方向，却不能倒写成 2025 年论文实验时已经使用的部件。本文引用源码时固定 commit，避免后续 `main` 改动让类名和行为漂移。

## 3. 从单轮 RL 到 Agentic RL，变化在哪里

单轮 RLVR 的典型结构是：给定一个 prompt，一次生成完整 response，校验器返回奖励，然后 learner 更新模型。生成长度仍会变化，但系统通常可以把请求组织为较规则的批次。Agentic RL 则在一条 episode 内反复经历：

```text
prompt / initial observation
  -> model action or tool call
  -> environment transition
  -> observation and reward signal
  -> append to context
  -> next model action
  -> ...
  -> terminal reward / timeout / task limit
```

这带来至少五类结构变化。

**第一，轨迹时长由环境和策略共同决定。** 有的 episode 一次工具调用就结束，有的会在错误路径上循环二十轮。同步批处理中，短轨迹完成后占用的生成资源很难自动转化为 learner 进度。

**第二，环境不是纯函数。** 数据库、操作系统和购物环境都有 session 状态，需要启动、路由、心跳、超时、回收和故障恢复。模型生成服务不能只拿一个静态 prompt。

**第三，模型 Token 与环境文本语义不同。** assistant action 是策略选择，应进入 policy loss；tool observation 由环境产生，虽然进入后续上下文，却不能当作模型行为优化。训练数据必须保留 Token 级 mask。

**第四，不同任务的奖励和难度不可直接比较。** 即使最终都映射到二值成功，成功率、轨迹长度和有效 Token 数也不同。把所有 advantage 混在一起做一次全局标准化，会让某些任务长期主导梯度。

**第五，异步会产生 policy lag。** trajectory 生成期间 learner 可能已经更新若干次。训练时的当前策略、真正生成 Token 的 rollout 策略和用于 KL 的 reference policy 是三个不同对象。

AgentRL 的设计可以看作对这五类变化的逐项响应。

## 4. Table 1/2：论文如何界定问题空间

![AgentRL 与其他强化学习框架和方法的能力对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table01-framework-comparison.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 1, CC BY 4.0.*

Table 1 把已有工作按 Multi-Turn、Multi-Task、Full-Async、Interactive Environments 和 Heterogeneous Environments 五个维度分类。它想表达的是：训练算法、异步基础设施和环境系统需要同时存在，而不是说所有被列为叉号的方法完全不能处理对应任务。表中的能力判断来自作者定义和论文比较，不是独立兼容性认证；不同项目在论文之后也可能增加新功能。

![单轮强化学习与 Agentic RL 的挑战对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table02-challenges.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 2, CC BY 4.0.*

Table 2 更直接地把挑战分成 infrastructure 与 algorithm 两列：

| 场景 | 基础设施问题 | 算法问题 |
| --- | --- | --- |
| Single-Turn | 同步 rollout 尚可管理 | 训练相对稳定、易扩展 |
| Multi-Turn | 同步计算浪费；大量交互环境难并发 | 状态空间更大，需要探索；训练中探索下降 |
| Multi-Task | 异构环境难统一 | 任务干扰与泛化不足 |

这张表的作用是建立工程分解，不是给出因果证明。例如，多轮训练不一定必须全异步，短 episode 或小规模实验仍可同步运行；多任务性能下降也不能只靠 advantage normalization 完全消除。论文提供的是在其目标规模下的一组设计选择。

## 5. 形式化基础：环境状态、语言状态与高层动作

论文把第 $i$ 个 Agent 任务写为 MDP：

$$
\mathcal{T}_i = \left(\mathcal{S}_i^{\mathrm{env}},\mathcal{A}_i,P_i,r_i,\rho_i\right),
$$

其中 $\mathcal{S}_i^{\mathrm{env}}$ 是环境状态，$\mathcal{A}_i$ 是高层动作空间，$P_i$ 是转移函数，$r_i$ 是奖励，$\rho_i$ 是初始状态分布。对 LLM Agent，仅有环境状态还不够，因为模型实际看到的是序列化上下文。论文定义组合状态：

$$
s_t = \left(s_t^{\mathrm{env}},s_t^{\mathrm{ctx}}\right),
$$

$s_t^{\mathrm{ctx}}\in\mathcal{V}^*$ 包含当前 episode 的对话、工具调用和 observation 前缀。一次高层动作不是单个离散编号，而是一串 Token：

$$
a_t=(y_{t,1},y_{t,2},\ldots,y_{t,L_t}),
$$

其概率分解为：

$$
\pi_\theta(a_t\mid s_t)
=\prod_{k=1}^{L_t}P_\theta\left(y_{t,k}\mid s_t^{\mathrm{ctx}},y_{t,<k}\right).
$$

这个分解非常重要。环境按完整函数调用执行动作，learner 却在 Token 级计算 log-prob、importance ratio、KL 和 advantage。也就是说，**环境 step 是交互粒度，Token 是优化粒度**。如果日志只保存完整 action 文本和最终奖励，就无法忠实还原公开实现中的 PPO/GRPO loss。

一条轨迹写成：

$$
\tau=\left(s^{(0)},a^{(0)},r^{(1)},s^{(1)},\ldots,a^{(T-1)},r^{(T)},s^{(T)}\right).
$$

多任务训练集合为 $\mathcal{T}=\{\mathcal{T}_1,\ldots,\mathcal{T}_{N_{\mathrm{task}}}\}$。每个样本 $x_{i,j}$ 可以生成 $K_{i,j}$ 条轨迹，组成 GRPO group：

$$
G_{i,j}=\{\tau_{i,j,1},\ldots,\tau_{i,j,K_{i,j}}\}.
$$

这里已经埋下后文的异步边界：只要使用组内相对 advantage，系统就必须知道同一 $x_{i,j}$ 的完整 group，至少在 advantage 计算前形成一个局部同步点。

## 6. PPO 与 GRPO：AgentRL 没有发明新的基础 policy loss

PPO 的 clipped surrogate objective 为：

$$
\mathcal{L}_{\mathrm{PPO}}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right],
$$

其中：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

GRPO 不训练 Critic，而是在同一 prompt 的 $K$ 条轨迹内比较 return：

$$
\hat A_{i,j,g}=
\frac{\hat R_{i,j,g}-\operatorname{mean}(\hat R_{i,j})}
{\operatorname{std}(\hat R_{i,j})}.
$$

论文实验以 GRPO 为基础，并增加 Cross-Policy Sampling 和 Task Advantage Normalization。它没有声称重新定义 PPO 的 ratio，也没有像 SAO 那样引入 Direct Double-Sided Importance Sampling。公开代码的 `ppo_loss()` 仍然读取 rollout 时保存的 `rollout_log_prob`，与当前 actor 的 Token log-prob 比较，再执行普通上下界 clipping；可选 KL 项使用冻结 reference model 的 log-prob。

GRPO 的好处是省掉与 actor 规模接近的 value model，代价是每个 prompt 需要多条样本。论文使用每个样本 8 次 rollout。对多轮环境而言，这不仅增加生成成本，还意味着同一 group 的快轨迹必须等最慢轨迹完成，才能计算组内均值和标准差。AgentRL 缩小了全局同步范围，却没有消除这个数学依赖。

## 7. Fig. 1：论文先展示了什么结果

![AgentRL 在五个环境上的增益和训练过程](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig01-overall-performance.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 1, CC BY 4.0.*

Fig. 1a 用 Qwen2.5-32B 展示 RL 前后五个环境的成功率变化。柱状图标出的绝对百分点提升分别约为：OS `+14.7`、WebShop `+31.1`、DB `+14.6`、KG `+43.2`、ALFWorld `+62.4`。这些数字说明任务收益差异很大：ALFWorld 和 KG 的 base model 留有巨大改进空间，而 OS 的提升相对有限。

Fig. 1b 把五任务平均成功率随训练样本数的变化画成曲线。起点是未做 SFT 的 `37.2%`，训练后达到约 `70.4%`。图中同时画了 Claude-Sonnet-4、GPT-5 和 DeepSeek-R1 的提示式评估水平。这里必须注意：AgentRL 模型经历了目标环境上的在线 RL，而 API 模型是 prompting baseline；这能证明目标任务训练后的结果较高，但不是同训练预算、同数据、同参数规模的纯模型能力比较。

曲线还不能单独证明 Cross-Policy Sampling、异步系统或任务归一化中的哪一项导致提升。组件贡献要结合 Table 6 与 Fig. 7 的消融阅读；硬件效率则要看 Fig. 4。

## 8. Fig. 2：一张图中的四条数据流

![AgentRL 总体框架](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig02-framework-overview.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 2, CC BY 4.0.*

Fig. 2 上半部分表示 rollout 和 training 的异步关系，下半部分同时放入环境框架、Cross-Policy Sampling 和 Task Advantage Normalization。可以把它拆成四条流：

1. **任务流**：训练数据样本进入异步 Task Manager，创建具体环境 session。
2. **交互流**：rollout engine 生成 action，经 Controller 路由到 task worker，返回 observation、status 和 reward。
3. **训练流**：完整轨迹进入 Buffer，满足取数条件后计算 advantage、reference log-prob 和 actor gradient。
4. **参数流**：actor 更新后的参数通过 NCCL 推送给当前 rollout engine；stale engine 仅隔若干 step 更新。

论文图里的“Cross-Policy Sampling”画成模型 $M_1/M_2$ 在同一轨迹 step 间切换，“Task Advantage Normalization”则画在 advantage estimation 之后。两者顺序不能颠倒：先要完成轨迹并得到 group advantage，才有按任务标准化的数值；Cross-Policy 发生在行为数据生成阶段。

公开仓库实现与图中概念基本对应，但把系统边界进一步具体化：Ray 管理 worker 和 object reference，SGLang 承担生成，FSDP 承担 actor/reference，Go Controller 管理环境路由，Python TaskWorker 承担任务语义。

## 9. 全异步 Rollout-Training 到底异步什么

论文把 rollout engine 放在独立资源组，训练模块则在另一组 GPU 上执行。训练每完成一次更新，就从已完成数据队列取可用轨迹，不等待“本轮发出的所有轨迹”一起返回。队列允许 batch size 在范围内浮动，使 learner 可以消费当前积累的完成数据。

同步流程近似为：

```text
roll out whole batch -> wait for slowest trajectory -> train -> sync weights -> next batch
```

AgentRL 流程近似为：

```text
rollout group:  continuously execute environment sessions ------>
learner group:       pull completed groups -> update -> pull -> update ------>
parameter stream:                         push latest actor weights ------>
```

这种设计减少的是 **rollout stage 与 training stage 之间的大屏障**。它让 learner 更新时 rollout GPU 仍能生成，让某些轨迹尚未完成时其他完整 group 可以先训练。论文还设置最大数据队列，并要求每个训练 step 尽可能把可用轨迹移动到训练侧，以限制数据陈旧程度。

但“fully asynchronous”不等于每条轨迹完成后立即单独更新。只要 `adv_estimator=grpo` 且 `n=8`，公开 Buffer 会把同一 `group_id` 的 8 条结果暂存在 `groups` 字典中，达到 `group_size` 后才一起加入可消费队列。异步存在两个层次：

- **跨 prompt / 跨任务异步**：不同 group 互不等待，哪个完整就先进入训练。
- **prompt 内局部同步**：同一 group 仍等待 8 条轨迹完成。

这一区分是理解 AgentRL 与 SAO 差异的关键。

## 10. Fig. 3：同步屏障如何形成 GPU 空泡

![同步与全异步训练架构对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig03-sync-vs-async.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 3, CC BY 4.0.*

Fig. 3a 中，多个 task rollout 长度不同，短轨迹完成后仍要等待长轨迹，随后所有 GPU 才一起进入 training。生成和训练在时间轴上交替，空白区域就是 pipeline bubble。

Fig. 3b 把 GPU 资源拆成 rollout 组和 training 组。任务协程持续填充 rollout 组，已完成数据则供 learner 消费。参数和数据异步传递，资源不再要求同一时间切换角色。代价也很明确：

- rollout 和 actor 各自需要常驻模型，增加总显存；
- 参数更新与数据生成不再严格同版本，引入 policy lag；
- 需要流量控制，否则完成队列可能无限增长；
- 故障恢复必须同时处理环境 session、队列数据和模型 checkpoint；
- 固定 rollout/actor GPU 比例不一定适合训练全过程。

因此，异步不是免费吞吐。它把同步等待变成了资源切分、数据新鲜度和控制面的复杂性。

## 11. Fig. 4：论文确实测了吞吐，但结论有范围

![AgentRL 异步训练吞吐与同步基线对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig04-throughput.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 4, CC BY 4.0.*

Fig. 4 以 Qwen2.5-14B 和 WebShop 为例，在 16、32、64 GPU 下比较同步 baseline 与 AgentRL。纵轴是千 Token 每秒，横纵轴都按对数呈现。图中标注大致为：

| GPU 数 | 同步 baseline | AgentRL |
| ---: | ---: | ---: |
| 16 | 9K tokens/s | 17K tokens/s |
| 32 | 16K tokens/s | 33K tokens/s |
| 64 | 29K tokens/s | 56K tokens/s |

在这一设置下，AgentRL 接近同步方案的两倍，并随 GPU 增加保持较好的增长趋势。这比只写“减少空泡”更有证据，因为论文给出了实际吞吐曲线。

仍需保留四个边界。第一，这是 WebShop 与 14B 模型，不代表环境延迟和轨迹分布完全不同的任务。第二，论文没有在图旁完整列出 rollout/actor GPU 划分、网络拓扑、batch 波动和队列年龄。第三，吞吐是生成与训练系统指标，不等于单位样本学习效率。第四，更多 Token 每秒只有在样本质量、policy lag 和奖励正确性可控时才有价值。

## 12. 源码审计：Buffer 暴露了真正的同步边界

官方 [`buffer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/components/buffer.py) 的核心逻辑很短：

```python
class Buffer:
    def __init__(self, max_size, group_size):
        self.strict_group = group_size > 1
        self.group_size = group_size
        self.groups = {}
        self.queue = deque()

    async def add(self, item):
        if self.strict_group:
            group_id = item["group_id"]
            self.groups.setdefault(group_id, []).append(item)
            if len(self.groups[group_id]) >= self.group_size:
                self.queue.extend(self.groups.pop(group_id))
        else:
            self.queue.append(item)
```

`get(minimum, multiple)` 又会把返回数量向 `multiple` 的整数倍取整；strict group 下，还会取 `group_size` 与调用方 `multiple` 的最小公倍数。其语义是：

1. 未完成的 group 不会暴露给 learner；
2. 不同 group 可以按完成顺序进入队列；
3. learner 一次可能取走超过 minimum 的所有可用完整 group；
4. 队列到达 `max_size` 时，生产者等待消费者释放空间；
5. `groups` 中尚未完整的轨迹不计入可消费 queue。

这是一种 **group-aware asynchronous batching**。它比整批同步灵活，但长尾 group 仍可能占用环境和 rollout 资源。公开配置中 `n: 8`，所以每个 prompt 仍需要八条 trajectory。若某条一直失败或 task worker 丢失，系统还需要重试、超时或清理机制，否则该 group 无法形成 advantage。

与 SAO 相比，AgentRL 的目标不同。SAO 直接把每个 prompt 的 rollout 数降为 1，重新引入 Critic 来替代组内 baseline；AgentRL 保留 GRPO 的无 Critic优势，因此也保留其局部 barrier。把两者都称作“异步 RL”没有错，但不能认为它们解决的是同一个同步点。

## 13. Fig. 5：训练框架与环境部署框架的边界

![AgentRL 训练框架与环境部署框架](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig05-environment-deployment-framework.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 5, CC BY 4.0.*

Fig. 5 把系统切成 Training Framework 和 Environment Deployment Framework。左侧负责 rollout、模型更新和异步 agent loop；右侧每个 host 上运行多个 task worker 与容器环境；中间的 AgentRL Controller 是统一入口。

这个分层值得保留，因为 RL 系统和任务环境的扩缩容规律不同：

- rollout engine 受模型大小、KV cache、batching 和生成长度约束；
- actor/reference 受参数、优化器状态、FSDP 通信和 microbatch 约束；
- task worker 受 CPU、网络、数据库、浏览器或模拟器资源约束；
- Controller 受 session 路由、心跳、状态存储和 API 请求量约束。

如果把所有组件塞进同一个进程，某个环境卡死会拖累生成服务，模型重启也会销毁全部 session。AgentRL 用 task worker 把任务语义封装起来，Controller 只维护通用控制接口，从而允许训练框架不了解 SQL、购物网站或 ALFWorld 的具体规则。

论文称 Controller 管理数千并发 episode。当前 README 则把 Go Controller 的目标写成最多 `10,000` concurrent sessions。后者是项目设计目标，不等于论文已经在同一实验中给出 10,000 session 的吞吐与可用性数据；本文只把它作为当前实现说明。

## 14. 统一 Function-Call API：环境如何变成训练接口

论文基于 AgentBench 的五个环境，把原有动作格式改造成 OpenAI Function Call 风格。统一后的任务 worker 至少需要提供三类操作：

```text
start(task sample)
  -> session id + initial messages + tool schemas

observation(session id, assistant message/tool call)
  -> new messages + reward + status + finish flag

end(session id, done)
  -> release or persist environment state
```

当前源码的 [`openai_chat_task()`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/components/task_manager.py) 把 Controller URL 分别绑定到 `openai_chat_start`、`openai_chat_obs` 和 `openai_chat_end`，然后创建异步 agent loop。具体任务只需要在 Controller/worker 一侧实现启动参数、工具定义、动作执行和奖励逻辑。

统一协议带来三个工程收益。

**训练代码不依赖任务类型。** 同一个 rollout loop 可以执行 WebShop search、数据库 SQL 或知识图谱查询，只要 task worker 返回统一消息结构。

**环境可以独立扩缩容。** DB worker、OS worker 和 ALFWorld worker 可以部署在不同 host，Controller 根据 task name 和 session 路由。

**轨迹日志具有统一语义。** assistant 生成、tool call、observation、reward、finish 和 status 可以使用相同字段进入数据管线，便于跨任务监控。

代价是任务适配层必须足够严格。工具 schema、参数解析、异常语义和 reward 都可能污染训练数据。一个把非法 tool call 错误判成成功的 worker，会让策略学习错误协议；一个在 observation 中泄漏答案的环境，则会让成功率失真。

## 15. 当前 Agent Loop：Token mask 比对话文本更重要

[`openai_chat_agent_loop()`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/agentic/loops.py) 展示了多轮轨迹如何变成训练张量。

启动环境后，worker 返回历史消息、工具列表和 session id。代码调用 tokenizer 的 chat template 构造初始 `input_ids`，这些 prompt Token 的 `loss_mask` 全部设为 0。每轮生成后：

```python
text, received_log_probs = await gen_fn(input_ids=ids)
new_ids = [token_id for log_prob, token_id in received_log_probs]
new_log_probs = [log_prob for log_prob, token_id in received_log_probs]

ids += new_ids
loss_mask += [1] * len(new_ids)
log_probs += new_log_probs
```

模型生成的 assistant Token 被标为 1，并保留 rollout 时真实 log-prob。随后 `FunctionCallParser` 把文本转换成结构化 tool call，发给环境。环境返回的新 message 被 chat template 编码，代码通过前后模板的差分取得 observation Token：

```python
ids += observation_ids
loss_mask += [0] * len(observation_ids)
log_probs += [0] * len(observation_ids)
```

因此，observation 会影响后续 action 的条件上下文，却不直接产生 policy gradient。这比仅保存 `history` 文本可靠：同一条对话里有系统提示、用户输入、assistant action 和环境反馈，只有 action 是策略输出。

loop 在 `done`、最大轮数或最大 Token 长度触发时结束。若环境状态不是 `completed`，最终 reward 被替换成 `incomplete_punishment`；公开论文配置使用 `-0.2`。返回数据包含 `input_ids`、`position_ids`、`loss_mask`、`rollout_log_prob`、最终 reward、Token 级 reward 张量、history 和环境 metrics。

这一实现也暴露复现风险：不同 tokenizer/chat template 版本可能改变 Token 边界；函数调用 parser 若无法识别输出，会把它当普通文本；截断可能切掉 action 或 observation；环境重试若不保持幂等，会得到不同状态。真实训练必须记录 tokenizer、模板、parser 和 task worker 版本。

## 16. 容器、Session 与故障隔离不是附属功能

多轮环境拥有生命周期。以数据库任务为例，一个 session 可能创建临时数据库状态；操作系统任务可能修改文件；WebShop 会维护页面和购物状态。Controller 至少需要回答：

- 哪个 worker 拥有某个 session？
- worker 心跳丢失后如何处理尚未完成的 episode？
- 同一训练样本重试是否复用状态？
- timeout 是环境失败、模型失败还是基础设施失败？
- episode 结束后如何回收容器与外部资源？
- Controller 重启后 session 路由是否可恢复？

当前仓库的 Controller 用 Go 实现，TaskWorker 是 Python/FastAPI 服务。HTTP transport 兼容 AgentBench 原有 API；gRPC transport 允许 worker 主动连接 Controller，避免每个远程 worker 都需要独立可入站 IP/端口。状态 provider 代码还覆盖本地、Redis、Consul 与 etcd 路线。

源码中可以找到 Docker、Kubernetes 和手工环境控制器，但当前任务文档的表述与源码能力并不完全同步。这属于论文后的工程演进和文档漂移，不能简单写成“论文系统原生支持 Kubernetes”。最稳妥的口径是：论文明确讨论容器化环境；固定 commit 的源码还包含 Kubernetes 相关实现；本文没有启动它们，也不评价生产成熟度。

在生产训练中，环境失败必须与策略失败分开计数。网络超时、容器启动失败和数据库不可用不应该默认映射成模型负奖励，否则 learner 会优化基础设施噪声。一个合理的数据契约至少需要 `termination_reason` 和 `infrastructure_error`，而不仅是单个 reward。

## 17. Cross-Policy Sampling：为什么单策略会探索收缩

Agent 在多轮环境中做出的早期动作会改变后续可达状态。一个模型如果在训练早期偏好某条路径，它采到的数据会进一步强化这条路径，其他有效策略难以出现。随着 policy 变得更尖锐，同 prompt 的多次采样也可能高度相似，GRPO group 内 reward 方差下降，训练信号变弱。

论文把这个问题称为探索不足，并借用模型自生成数据导致多样性下降的讨论。它的应对不是简单提高 temperature，而是让一条 trajectory 的不同环境 step 可以来自不同 policy。某个模型擅长规划但不愿调用工具，另一个模型规划一般却能打破工具调用循环；交替动作可能访问任何单一策略都很少触达的状态。

这里的“policy”有两种实现层次：

1. **推理验证中的异构模型池**：例如 Qwen 与 Llama，逐 step 随机选择模型。
2. **RL 训练中的同构时间版本池**：当前 actor 对应的 rollout engine 与延迟更新的 stale engine。

论文先用异构模型说明概念，再在训练系统里用旧策略近似。二者都增加行为分布多样性，但不保证产生同样的探索效果。

## 18. Cross-Policy 轨迹公式与支持集直觉

普通轨迹的动作来自固定模型 $m$；Cross-Policy 轨迹写成：

$$
\tau^c=\left(s^{(0)},a^{c,(0)},r^{(1)},s^{(1)},\ldots,s^{(T)}\right),
$$

其中：

$$
a^{c,(t)}\sim \operatorname{random}(\mathcal{M})(\cdot\mid s^{(t)}),
$$

$\mathcal{M}$ 是策略集合。换言之，每个环境 step 先随机选一个 policy，再由它基于完整上下文生成 action。

附录进一步引入 grounding 函数：

$$
\Gamma:\mathcal{L}_{\mathrm{valid}}\rightarrow
\Delta(\mathcal{S}^{\mathrm{env}}),
$$

把有效语言状态映射到环境状态分布。若成功环境状态集合为 $\mathcal{G}$，可到达成功状态的语言 preimage 定义为：

$$
\mathcal{L}_{\mathcal{G}}
=\Gamma^{-1}(\mathcal{G})\cap\mathcal{L}_{\mathrm{valid}}.
$$

论文用下式表达 Cross-Policy 的直觉：

$$
\operatorname{supp}(\tau^c)\cap\mathcal{L}_{\mathcal{G}}
\supsetneqq
\bigcup_m\left(
\operatorname{supp}(\tau^m)\cap\mathcal{L}_{\mathcal{G}}
\right).
$$

这个严格超集不应被当作普遍定理。若多个模型完全相同、随机切换破坏上下文一致性，或环境成功路径只依赖单一稳定策略，Cross-Policy 未必扩大成功支持集。更准确的解读是：**不同 policy 在相同语言协议下具有互补条件动作分布时，step-level 组合可能产生新的有效路径。** 论文的 Fig. 8/9 提供经验例子，而不是一般证明。

## 19. Fig. 6：Single、Mix 与 Cross 不能混用

![Single、Mix 与 Cross-Policy 三种采样策略](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig06-rollout-strategies.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 6, CC BY 4.0.*

Fig. 6 画出三种方案：

- **Single**：一条轨迹从头到尾由一个模型生成。
- **Mix**：数据集中一部分完整轨迹来自模型 1，另一部分完整轨迹来自模型 2。
- **Cross**：同一轨迹内部的不同 step 可以由模型 1 或模型 2 生成。

Mix 增加了训练集层面的策略多样性，却没有组合策略路径。若模型 1 总在第三步陷入循环，模型 2 总在第一步走错，那么把两者的失败轨迹混在一起不会自动产生“模型 1 规划前两步、模型 2 打破第三步循环”的路径。Cross 的目标正是让互补行为发生在 episode 内。

这种组合也可能失败。模型 2 不一定理解模型 1 留下的隐含计划，工具参数格式可能不同，后续模型会把前文视为自己生成的数据而过度信任。生产系统应记录每个 action 的 `behavior_policy_id`，才能分析切换位置、成功率和分布偏移；只记录整条 trajectory 的模型名会丢失核心因果线索。

## 20. Stale Engine：论文方法在训练代码中的近似

真实 RL 训练很难在同一个 actor update 中同时维护 Qwen 与 Llama 两种架构。论文改用同一模型的早期版本：一部分 rollout engine 每 step 接收最新参数，另一部分 stale engine 每隔多个 step 更新。当前 [`agentrl_trainer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/agentrl_trainer.py) 的 `cross_sampler` 为每次生成调用选择 engine：

```python
def cross_sampler(item, **kwargs):
    if rollout_stale is None:
        return rollout.dispatch_rank(...).generate(**kwargs)
    if random.random() < (1 - stale_ratio):
        return rollout.dispatch_rank(...).generate(**kwargs)
    return rollout_stale.dispatch_rank(...).generate(**kwargs)
```

公开五任务 14B 配置为：

```yaml
rollout_ratio: 0.5
rollout_stale_ratio: 0.25
stale_step: 25
task_adv_norm: true
```

`rollout_ratio=0.5` 先把总 GPU 的一半分给 rollout；`rollout_stale_ratio=0.25` 又从 rollout GPU 中切出四分之一给 stale pool。注意，`cross_sampler` 使用同一个 `stale_ratio` 作为选择 stale engine 的概率。actor 每 step 向当前 engine 推送权重，仅当 `global_step % stale_step == 0` 时才更新 stale engine。

因此，公开实现并不是显式维护任意大小的 model pool，而是“两级时间版本池”：current 与 stale。它是论文 Cross-Policy 思想的工程近似。`stale_step=25` 也不是理论常数；学习率、模型更新幅度、任务长度和队列年龄变化后，相同 step 差可能对应完全不同的 KL 距离。

## 21. Off-Policy 边界：探索多样性和分布偏移是一体两面

Cross-Policy 有意让数据来自非当前策略。异步队列又会额外增加版本滞后。训练时至少需要区分：

- $\pi_{\mathrm{behavior}}$：实际生成某个 Token 的 current 或 stale rollout engine；
- $\pi_\theta$：learner 更新时的当前 actor；
- $\pi_{\mathrm{ref}}$：用于 KL regularization 的冻结 reference model。

公开 agent loop 保存每个生成 Token 的 `rollout_log_prob`，`ppo_loss()` 重新计算当前 actor log-prob，因此可以形成 importance ratio。普通 PPO clipping 限制单次 policy update 的极端 ratio，但不能证明跨 25 step stale policy 的所有数据都近似 on-policy。公开配置还使用 `kl_loss_coef=1e-4` 约束 actor 与 reference 的漂移。

论文的系统性防线主要是：限制数据队列大小、每次更新尽量移动可用轨迹、保存行为 log-prob 并使用 PPO/GRPO clipping。它没有报告完整的 policy-version lag 分布，也没有像 SAO 那样在 ratio 超出双边区间时把 Token 梯度完全置零。

工程上应额外监控：

$$
\operatorname{lag}_{\mathrm{step}}
=v_{\mathrm{train}}-v_{\mathrm{behavior}},
$$

以及 Token ratio 的分位数、PPO clipped fraction、stale/current 样本比例和按 lag 分桶的成功率。若 stale 数据大量落在 clipping 区域，所谓探索可能只增加了无法有效学习的样本。

## 22. Task Advantage Normalization：为什么要在 GRPO 之后再归一化

对任务 $i$，论文收集当前 batch 内所有有效 action Token 的 advantage：

$$
\mathcal{A}_i^{\mathrm{tok}}
=\left\{
\hat A_{i,s,g,t,k}
\mid 1\le s\le S_i,
1\le g\le K_{i,s},
1\le t\le T_{i,s,g},
1\le k\le L_{i,s,g,t}
\right\}.
$$

然后按任务计算均值与标准差：

$$
\tilde A_{i,s,g,t,k}
=\frac{\hat A_{i,s,g,t,k}-\mu_i}{\sigma_i},
\quad
\mu_i=\operatorname{mean}(\mathcal{A}_i^{\mathrm{tok}}),
\quad
\sigma_i=\operatorname{std}(\mathcal{A}_i^{\mathrm{tok}}).
$$

这和 GRPO 的组内归一化不是重复操作。GRPO 在同一 prompt 的多条 trajectory 间比较 return，处理的是样本内相对好坏；Task Advantage Normalization 在任务 batch 内比较所有有效 Token，处理的是任务间尺度和学习速度。

源码 `adv_norm(items)` 先用 `loss_mask` 选择 assistant action Token，把同一 `data_source` 下的有效 advantage 拼接，计算 mean/std，再把该任务所有 item 的完整 advantage 张量标准化，分母加 `1e-6` 防止数值问题。

这种方法也有代价。任务 batch 太小时均值和方差噪声很大；极长轨迹贡献更多 Token，可能主导统计；标准化会抹掉任务整体 reward 水平差异；失败率接近 0 或 1 时 advantage 可能退化。它降低的是一类一阶尺度干扰，不是完整的多任务梯度冲突算法。

## 23. 多任务数据混合：平衡样本数不等于平衡优化

当前 trainer 为每个数据集建立独立 DataLoader，用 `repeat(..., n)` 为每个 prompt 生成 group，再通过 `interleave()` 和无限 `cycle()` 交错任务。它试图让五个环境都持续进入队列，而不是按原始数据集大小被动混合。

但多任务平衡至少有四种不同口径：

1. prompt 数量相同；
2. trajectory 数量相同；
3. action Token 数量相同；
4. GPU 时间或环境 wall-clock 相同。

一个 ALFWorld prompt 可能比 DB prompt 产生更多轮数和 Token。即使 prompt 轮询公平，Token-mean loss 仍可能让长轨迹占更多训练权重；异步完成顺序也会让快任务更早填满 Buffer。论文用 Task Advantage Normalization 缓解梯度尺度，但没有完整证明采样、Token 和计算预算同时公平。

生产系统应按任务同时监控 issued prompts、completed trajectories、valid action tokens、environment seconds、GPU rollout seconds 和 learner loss weight。只看最终 batch 的任务条数会隐藏异步系统中的速度偏置。

## 24. 当前训练源码：从 YAML 到一次 Actor Update

官方 README 指向 [`examples/training/agentrl_trainer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/agentrl_trainer.py) 作为论文结果复现入口。固定 commit 下的主链可以压缩为十步：

1. 读取 YAML，启动 Ray，计算 rollout、stale rollout 与 actor GPU 数量。
2. 创建三个 placement group：current rollout、stale rollout、actor/reference。
3. 用 `spawn()` 创建 `AsyncSglangWorker` 与 `FSDPWorker` collective handle。
4. 构建 SGLang engine、actor、optimizer、checkpoint manager 和 reference model。
5. 注册 actor rank 0 到 rollout worker 的 NCCL sender/receiver。
6. 创建多任务 DataLoader、`DistributedTaskManager`、Ray Queue 与 group-aware Buffer。
7. 异步执行 agent loop，收集完整 group；learner 取出可用数据。
8. 计算 GRPO advantage，并按 `data_source` 执行 Task Advantage Normalization。
9. reference forward 得到 `ref_log_prob`；actor forward/backward 计算 PPO/KL loss 并 step。
10. 每 step 同步 current rollout；按 `stale_step` 同步 stale rollout；按间隔验证和保存 checkpoint。

当前 14B 五任务示例配置还包含：

| 配置 | 值 | 含义 |
| --- | ---: | --- |
| `max_steps` | 1000 | 示例训练更新上限 |
| `train.batch_size` | 256 | trainer 中与 `n` 相乘形成 minimum trajectory 数 |
| `train.n` | 8 | 每个 prompt 的 group size |
| `train.concurrency` | 2048 | 分布式任务 worker 数配置 |
| `val.n` | 4 | 每个验证样本重复数 |
| `max_total_len` | 8192 | 总上下文上限 |
| `max_turns` | 20 | 环境交互轮数上限 |
| `actor.fsdp_size` | 16 | actor FSDP world size 配置 |
| `max_tokens_per_micro_batch` | 16384 | actor microbatch Token 上限 |
| `lr` | `1e-6` | actor 学习率 |
| `clip_ratio` | `0.2` | PPO clipping |
| `kl_loss_coef` | `1e-4` | reference KL 权重 |

代码令 `real_bsz=batch_size*n`，因此示例的 minimum 是 `2048` 条 trajectory，而不是 256。第一次取数还要求 `1.7x` minimum，以便预热 Buffer。这里的数字属于当前公开示例，不应倒写成论文正文明确披露的统一 batch size；论文只明确给出每样本 8 次 rollout、温度 0.8、H800、14B 至少 16 GPU 和多任务训练超过 1000 step。

## 25. Action/Observation 训练数据契约

论文没有给出完整 wire schema，但结合公式与源码，可以构造一个不混淆语义的最小契约：

```json
{
  "trajectory_id": "kg-train-1847-g3-v412",
  "prompt_id": "kg-train-1847",
  "group_id": "kg-train-1847-v412",
  "task": "kg-env_train-v2",
  "sample_index": 1847,
  "behavior_policy": {
    "default_version": 412,
    "action_versions": [412, 387, 412, 387],
    "stale_flags": [false, true, false, true]
  },
  "input_ids": [151644, 8948, 198, 2610],
  "position_ids": [0, 1, 2, 3],
  "loss_mask": [0, 0, 1, 0],
  "rollout_logprobs": [0.0, 0.0, -0.284, 0.0],
  "turns": [
    {
      "turn": 0,
      "assistant_action": {"name": "get_relation", "arguments": {}},
      "observation": "...",
      "behavior_policy_version": 412,
      "started_at": "...",
      "completed_at": "..."
    }
  ],
  "reward": 1.0,
  "status": "completed",
  "termination_reason": "environment_success",
  "infrastructure_error": null,
  "controller_version": "controller-v0.2.0",
  "task_worker_version": "kg-worker-sha",
  "tokenizer_revision": "model-revision"
}
```

公开代码返回的字段更精简，并依赖上游 dataset item 提供 `group_id`、`data_source` 等元数据。上述扩展不是官方 schema，而是根据生产审计需求补出的建议。

几个字段不能省略。`group_id` 决定 GRPO 比较范围；`loss_mask` 决定哪些 Token 是 action；`rollout_logprobs` 是行为策略证据；`behavior_policy_version` 用于计算 policy lag；`termination_reason` 区分成功、任务上限和基础设施失败；Controller/worker/tokenizer 版本用于回放。

Cross-Policy 下只保存轨迹级 policy id 不够，因为同一 episode 的不同 action 可能来自 current 或 stale engine。源码目前将 `cross_sampler` 的选择隐藏在生成函数里，返回数据没有显式持久化逐 action engine id。论文复现可以依靠整体概率和实验配置，生产诊断则应补充该字段。

## 26. 实验设置：五类环境分别在测什么

论文基于 AgentBench-FC 的五个多轮环境：

| 缩写 | 环境 | 主要能力 | 典型失败 |
| --- | --- | --- | --- |
| AF | ALFWorld | 常识规划、物体操作、根据反馈重规划 | 找错位置、动作循环、达到轮数上限 |
| DB | Database | 理解问题、生成 SQL、读取结果 | SQL 语法/语义错误、错误 join、未验证结果 |
| KG | Knowledge Graph | 关系检索、多跳组合、工具参数理解 | 推理闭环、工具名或参数错误 |
| OS | Operating System | 命令行规划、文件与进程操作 | 破坏状态、命令失败、错误路径 |
| WS | WebShop | 搜索、筛选、页面导航和属性匹配 | 购买错误商品、搜索循环、遗漏约束 |

这些任务都使用函数调用协议，但动作空间、反馈密度和 episode 成本不同。ALFWorld 的环境状态更接近具身规划，DB/KG 的工具返回结构化结果，OS 带有更强副作用，WebShop 则有长导航路径。把它们放在一个模型里，确实比同一类型的五个数据集更能检验环境异构性。

训练模型覆盖 Qwen2.5-3B/7B/14B/32B-Instruct 和 GLM-4-9B。Qwen 系列直接从基础 Instruct 模型进行 RL，没有额外 warm-up SFT；GLM-4-9B 因函数调用格式适配问题，先使用少量 SFT 数据冷启动。这个差异意味着 GLM 行和 Qwen 行不是完全相同的起点。

训练使用 H800 GPU，14B 的最低配置为 16 GPU；多任务混合训练超过 1000 step。rollout temperature 为 0.8，每个样本采 8 条轨迹；完整轨迹正确得到二值 reward，超过最大轮数或响应长度被惩罚 `-0.2`。SGLang 负责推理，FSDP 负责训练。评估同样使用 temperature 0.8，每个任务连续运行 4 次，报告均值与标准差。

论文没有完整公开随机 seed、全部优化器参数、各规模 GPU 分配、每次训练总 Token、wall-clock、训练成本和失败重跑次数。公开 YAML 补充了部分配置，但不能保证就是 Table 3 每个模型最终 run 的唯一配置。

## 27. Table 3：主结果应该怎样读

![AgentRL 五任务主实验结果](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table03-main-results.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 3, CC BY 4.0.*

Table 3 同时列出 API prompting、开源模型 prompting、已有 Agent training 方法和 AgentRL。最重要的行可以精简如下：

| 模型 | ALFWorld | DB | KG | OS | WebShop | AVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude-Sonnet-4 Thinking | 69.0 | 68.4 | 64.4 | 51.0 | 38.3 | 58.2 |
| GPT-5 | 65.4 | 63.2 | 64.1 | 34.5 | 33.7 | 52.2 |
| DeepSeek-R1 | 51.4 | 60.4 | 50.2 | 53.6 | 31.0 | 49.3 |
| Qwen2.5-14B-Instruct base | 8.7 | 48.4 | 35.3 | 26.0 | 17.6 | 27.2 |
| Qwen2.5-32B-Instruct base | 32.1 | 55.8 | 33.8 | 37.0 | 27.5 | 37.2 |
| AgentRL Qwen2.5-3B | 92.4 | 60.0 | 55.0 | 40.5 | 52.1 | 60.0 |
| AgentRL Qwen2.5-7B | 91.5 | 63.7 | 57.8 | 40.8 | 56.1 | 62.0 |
| AgentRL Qwen2.5-14B | 91.5 | 72.2 | 72.8 | 43.6 | 58.5 | 67.7 |
| AgentRL Qwen2.5-32B | 94.5 | 70.4 | 77.0 | 51.7 | 58.6 | **70.4** |
| AgentRL GLM-4-9B | 93.3 | 66.9 | 75.7 | 33.2 | 55.9 | 65.0 |

结果支持三条较稳健的判断。

**RL 训练收益很大。** 32B 平均值从 `37.2` 提升到 `70.4`，14B 从 `27.2` 提升到 `67.7`。提升集中在 ALFWorld、KG 和 WebShop，说明交互协议和任务反馈确实能让模型学会环境特定策略。

**模型规模仍有收益，但不是每项单调。** 32B 平均最好，14B 在 DB 上反而高于 32B，OS 也没有随每个规模严格增长。论文所谓 scaling trend 是总体趋势，不是所有任务、所有相邻规模都单调。

**训练后的开源模型在该协议下超过表中 API baseline。** AgentRL-32B 平均 `70.4`，高于 Claude-Sonnet-4 Thinking 的 `58.2`。但前者针对五个环境做了在线训练，后者没有；这更接近“专门训练模型与通用 prompting 模型”的比较，而非基础模型实力排名。

表中 Hephaestus 与 AgentLM 的 WebShop 数字带星号，表示直接来自原论文，而非 AgentRL 团队统一重跑。跨论文结果可能使用不同环境版本或评估细节，需要谨慎横向比较。

## 28. Table 4：一个多任务模型是否真的等于五个专家

![AgentRL 单任务训练与多任务训练对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table04-single-vs-multitask.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 4, CC BY 4.0.*

Table 4 训练五个 Qwen2.5-14B 单任务模型，再与一个五任务联合模型比较：

| 模型 | ALFWorld | DB | KG | OS | WebShop | AVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 五个专家逐列取最好 | 89.7 | 73.9 | 72.2 | 43.1 | 60.3 | **67.8** |
| 一个 AgentRL 多任务模型 | **91.5** | 72.2 | **72.8** | **43.6** | 58.5 | **67.7** |

多任务模型平均 `67.7`，几乎等于五个单任务专家逐列拼出的 `67.8`，而部署只需要一套权重。这是论文“generalist agent”主张中最有价值的证据之一。

但“matches the best results among all task-specific models”应按平均值理解。多任务模型在 DB 比专家最佳低 `1.7`，WebShop 低 `1.8`；它在 ALFWorld、KG 和 OS 又略高于专家。它不是每项都精确匹配，也不是证明加入任何新任务都不会发生负迁移。

单任务行还显示强烈的任务专化：例如 AgentRL-ALFWorld 在 ALFWorld 达到 `89.7`，但 WebShop 只有 `15.9`；AgentRL-WebShop 在 WebShop 达到 `60.3`，ALFWorld 为 `0.0`。这说明在线 RL 很容易把函数调用策略推向特定环境。多任务训练的意义不是平均分技巧，而是防止单环境优化彻底破坏其他交互协议。

## 29. Table 5：BFCL-v3 泛化提升存在，但幅度有限

![AgentRL 在 BFCL-v3 上的泛化结果](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table05-bfcl-generalization.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 5, CC BY 4.0.*

论文用未参与训练的 BFCL-v3 检查函数调用泛化：

| 模型 | single-turn nonlive | single-turn live | multi-turn | overall |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-32B-Instruct | 86.0 | 77.4 | 16.2 | 59.9 |
| AgentRL Qwen2.5-32B | 85.8 | 79.3 | 19.2 | 61.4 |
| 变化 | -0.2 | +1.9 | +3.0 | +1.5 |

最合理的结论是：五任务 Agentic RL 没有明显破坏 single-turn nonlive 能力，并对 live 与 multi-turn 函数调用带来小幅迁移，整体增加 `1.5`。它不能支持“获得了全面通用工具能力”的强结论。

特别是 multi-turn 从 `16.2` 到 `19.2`，相对增幅不小，绝对水平仍低。训练环境共享统一函数调用协议，迁移可能来自格式熟练度、工具调用倾向和长上下文交互习惯，而不一定是更强的未见工具语义推理。

BFCL 还只是一个 OOD benchmark。真正部署前，应额外检查 schema 长度、无效参数恢复、多工具并发、工具返回注入、权限边界和高风险副作用。

## 30. Table 6 与 Fig. 7：两个算法部件贡献了什么

![Cross-Policy Sampling 与 Task Advantage Normalization 消融表](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table06-ablation.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 6, CC BY 4.0.*

| 14B 设置 | AF | DB | KG | OS | WS | AVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentRL 完整模型 | 93.1 | 64.0 | 67.7 | 45.1 | 55.0 | **65.0** |
| 去掉 cross sampling | 91.9 | 61.6 | 55.7 | 39.7 | 54.5 | 60.7 |
| 去掉 task advantage norm | 91.1 | 62.6 | 54.7 | 38.0 | 50.6 | 59.4 |

去掉 Cross-Policy 后平均下降 `4.3`，KG 下降 `12.0`；去掉任务归一化后平均下降 `5.6`，KG、OS 和 WebShop 都明显回退。两者并非只影响同一任务。

![AgentRL Cross-Policy 与任务归一化训练曲线消融](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig07-ablation.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 7, CC BY 4.0.*

Fig. 7a 的 KG 曲线显示，无 Cross-Policy 的模型前期上升较快，却较早进入平台；带 Cross-Policy 的曲线后期继续提升。作者据此解释跨策略采样扩展了可探索状态。另一种可能是 stale 数据改变了训练正则化和更新速度，论文没有通过等 KL、等 entropy 对照完全排除。

Fig. 7b 中，去掉 Task Advantage Normalization 的 ALFWorld 曲线学习较慢且方差更大，支持它缓和任务间尺度干扰。Fig. 7c 的五任务平均结果表明两个部件组合最好。由于曲线未展示多个随机 seed 的独立轨迹和训练成本，仍不能精确量化稳定性方差。

值得注意的是，Table 6 的完整 14B 平均 `65.0`，而 Table 3 的 AgentRL-14B 是 `67.7`。它们显然不是同一个最终 run 或完全相同评估快照。文章不能把两张表中的数字混成一组，也不应拿消融表的 `65.0` 说主模型最终性能。

## 31. Fig. 8：先用异构模型验证，再用 stale policy 训练

![Cross-Policy Sampling 在 WebShop 推理和训练中的效果](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig08-cross-policy-webshop.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 8, CC BY 4.0.*

Fig. 8a 在推理阶段比较 Qwen2.5-14B、Llama-3.1-8B、Mix 和 Cross-Policy 的 pass@k。低 $k$ 时，Cross-Policy 不一定优于最佳单模型；随着 $k$ 增大，Cross 曲线逐渐超过两者和 Mix。这符合“组合路径增加覆盖”的目标，也说明它的优势更像搜索多样性，而不是单条轨迹质量无条件更高。

Mix 的 pass@k 上限受两个模型完整轨迹并集约束；Cross 能构造新的 step 组合。若成功路径需要模型能力互补，随着样本数增加，更容易碰到有效切换序列。

Fig. 8b 是 WebShop 上的初步训练实验。带 Cross-Policy 的模型从起始点和最终 pass@k 都高于无 Cross-Policy 版本。论文明确提醒，图中设置与主实验不完全相同，因此不能直接用这张图估计 Table 3 的收益占比。

推理实验是真正的异构模型 Cross-Policy，训练实验则使用同模型不同时间版本。它们共同支持“策略多样性有帮助”，但并不证明 stale engine 完全复制了 Qwen/Llama 的互补性。

## 32. Fig. 9 与 Table 7：案例和错误状态怎样限制结论

![KG 任务中 GLM、Llama 与 Cross-Policy 的案例](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig09-cross-policy-kg-case-study.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 9, CC BY 4.0.*

Fig. 9 的问题是“同时实践藏传佛教和道教的宗教实践数量”。GLM 能推理答案，却陷入重复验证且没有按协议完成工具交互；Llama 愿意调用工具，却错误理解 `count` 工具并达到任务上限。Cross-Policy 轨迹先利用 GLM 的关系规划，再切换到 Llama 触发工具尝试，最后回到 GLM 形成合理调用并成功。

这是一个有解释力的例子：失败不是知识完全缺失，而是规划、协议遵循和工具倾向的组合缺陷。策略切换改变了局部行为模式，打破单模型循环。

它仍只是精选案例，不能说明随机切换平均都会产生互补。实际系统需要统计：成功轨迹切换次数、切换发生的 turn、不同 policy 顺序、无效切换率和额外推理成本。

![基础模型与 AgentRL 的主要终止状态对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table07-failure-modes.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 7, CC BY 4.0.*

| 环境 | Base Completed | Base Task Limit | AgentRL Completed | AgentRL Task Limit |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 0.070 | 0.680 | 0.926 | 0.074 |
| DB | 0.957 | 0.043 | 0.993 | 0.007 |
| KG | 0.747 | 0.213 | 0.947 | 0.033 |
| OS | 0.548 | 0.444 | 0.847 | 0.118 |
| WebShop | 0.725 | 0.275 | 0.980 | 0.020 |

Table 7 显示 AgentRL 大幅减少 Task Limit Reached，特别是 ALFWorld。可是 caption 明确说 `Completed` 只表示 agent 提交了答案，不保证答案正确；两个状态也不穷尽所有终止原因，行内百分比不必相加为 100%。

因此，这张表证明模型更容易走到“提交”状态，不等于成功率本身。若只优化 completed rate，模型也可能更快提交错误答案。应把 termination status、环境 reward 和答案校验分开监控。

## 33. AgentRL 与 SAO：两种“异步 Agentic RL”并不重复

本站上一篇 [SAO 精读](/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning-paper-reading/) 研究的是另一个同步点。两者可以放在同一张表中理解：

| 维度 | AgentRL | SAO |
| --- | --- | --- |
| 核心目标 | 多轮、多任务训练系统与环境扩展 | 单轨迹异步优化的稳定性 |
| 基础 advantage | GRPO group-relative | Actor-Critic / Token-level GAE |
| 每 prompt rollout | 论文与配置为 8 | 1 |
| 异步边界 | rollout 与 learner 解耦；完整 group 入 Buffer | 单条 trajectory 完成即可入队 |
| prompt 内 group barrier | 保留 | 移除 |
| policy lag 处理 | 有界队列、rollout log-prob、PPO clipping、KL | Direct Double-Sided Importance Sampling 硬门控 |
| 探索机制 | current/stale Cross-Policy | 不是主要贡献 |
| 多任务稳定 | Task Advantage Normalization | 论文重点不在多任务归一化 |
| Critic | 默认 GRPO 路线不需要 | 需要 value model |
| observation 处理 | `loss_mask=0`，不参与 policy loss | Skip-Observation GAE 跨过 observation |
| 系统效率证据 | 给出 WebShop 14B 吞吐曲线 | 论文未给吞吐对照 |
| 环境控制面 | Controller + worker + container | 依赖外部 Agent 环境/训练系统 |

AgentRL 在 2025 年先把 rollout-training stage 解耦，并提供环境控制面；SAO 在 2026 年进一步指出，GRPO 完整 group 自身仍是异步瓶颈，于是牺牲无 Critic 的简洁性，换成 single rollout 和 value model。

两者也可以组合成未来路线：保留 AgentRL 的 Controller、TaskWorker、参数同步和多任务环境，把 `buffer_group_size` 调为 1，接入 value worker、Skip-Observation GAE 和 DIS。但这不是改一个配置即可完成：数据需要 value prediction，loss 需要 Critic，checkpoint 和 GPU 资源分配也要重构。

在 off-policy 处理上，两者哲学相反。AgentRL 主动保留 stale policy 以增加探索，SAO 则对偏离当前策略过远的 Token 严格屏蔽梯度。一个成熟系统可能需要同时回答：哪些 lag 是有益探索，哪些 lag 已经无法安全学习。仅以 policy version 差判断不够，最好结合 Token ratio、KL 和任务结果。

## 34. 与 VeRL、AReaL、ROLL、slime 和 AgentBench 的关系

AgentRL 论文的 Table 1 把若干 RL 框架列为对照，但工程比较应按职责拆分。

**VeRL / OpenRLHF** 主要提供大模型 RL 训练、分布式 worker 和 rollout/learner 编排。AgentRL 的论文实现也以 VeRL 为基础做全异步改造。AgentRL 的差异重点是多轮环境交互、异步 agent loop 和环境部署框架，而不是重新实现所有底层张量并行能力。

**AReaL** 同样关注异步 RL，强调 actor-learner 解耦与 stale trajectory 管理。论文表中把它标为 Full-Async，但不具备 AgentRL 定义的完整多任务、交互环境与异构环境集成。这个对比依赖论文发布时版本，不能视为永久功能差距。

**ROLL** 提供更广泛的后训练基础设施，后续版本包含 Agentic RL 路线。比较时应查看当前实现，而不是只复制 AgentRL Table 1 的 2025 快照。

**slime** 是 SGLang 生态中的 RL scaling 框架，强调灵活数据生成和高性能 rollout。它可以作为 AgentRL 训练侧的相关实现参考，却不自动提供 AgentRL 的 Controller/TaskWorker 协议。

**AgentBench** 更接近环境和评测底座。AgentRL 重构其五个环境为 function-call 接口并提供集中式部署。论文成绩仍依赖 AgentBench-FC 的任务定义和 reward；换成浏览器、移动端或真实软件工程环境，需要重新实现 worker、容器镜像和可验证奖励。

**OpenHands / E2B** 提供代码执行或沙箱环境，适合 Agent episode 隔离。它们解决环境安全与执行，不直接提供 GRPO advantage、actor 更新和模型参数同步。

因此，AgentRL 不是所有相关系统的替代品。它更像把分布式 RL 训练框架与异构 Agent 环境之间缺失的控制层和训练协议补齐。

## 35. 论文发布后，官方仓库增加了什么

固定 commit 的 README 已经不只描述论文原型。需要按时间与证据分层：

### 35.1 模块化 Trainer

当前 `trainer` 包把 placement group、collective handle、FSDP worker、SGLang worker、NCCL 参数传输、task manager、Buffer、advantage 和 loss 分成独立组件。`async_trainer.py` 提供通用异步 GRPO 示例，`agentrl_trainer.py` 则保留 stale rollout 与任务归一化，作为论文路线的复现入口。

算法枚举还支持 GAE、GRPO_PASSK、REINFORCE++、REMAX 和 RLOO。这些能力说明仓库正向通用训练框架演进，但论文实验不能因此写成“比较了所有这些算法”。

### 35.2 Controller 与 Dashboard

Go Controller 管理 worker、session 和路由，仓库带有 dashboard 前端与 snapshot 相关命令。当前 release 列表中的 `controller-v0.2.0` 是 Controller 子组件版本，不是 AgentRL 论文模型或整个 monorepo 的 `v0.2.0`。

### 35.3 HTTP 与 gRPC Transport

HTTP 保持 AgentBench 兼容；gRPC 允许 worker 从远端集群接入，无需 Controller 访问每个 worker 的公开地址。这是跨网络部署的重要改进，但需要额外处理 TLS、认证、服务发现和 backpressure，README 的可连接性说明不等于生产安全默认完成。

### 35.4 `agentrl-eval`

仓库增加实验性的评估包和 CLI，README 仍明确称其 experimental。论文早期使用的 `server_agent.py`/检查脚本与新评估包应分开描述；新 CLI 不能作为论文所有 Table 3 数字的原始评测入口证据。

### 35.5 更多环境与部署代码

源码出现 MobileRL、Kubernetes controller 和多个状态 provider。它们代表扩展方向。论文主实验仍是五个 AgentBench-FC 环境，明确的环境论证也以 containerized worker 为主。

### 35.6 CI 与可复现边界

仓库为 controller、worker、trainer 和 eval 配置了独立 CI workflow，能验证组件测试和构建。完整 14B/32B 多任务 H800 训练显然不适合普通 CI，因此“CI 通过”只能证明基础代码路径，不证明论文曲线可一键重现。

## 36. 参数同步与资源分配：为什么不是简单的生产者消费者

当前 trainer 先读取 Ray 集群 GPU 数量，再按 `rollout_ratio` 分配 rollout 与 actor。示例 `0.5` 意味着一半 GPU 常驻 SGLang，另一半用于 actor/reference。actor 和 reference 通过 `num_gpus=0.5` 共享 placement group bundle，具体显存是否足够取决于 dtype、FSDP shard 和模型大小。

参数同步不是通过磁盘 checkpoint。actor rank 0 创建 `NCCLTensorSender`，rollout worker 注册 `NCCLTensorReceiver`。初始模型就绪后先完整推送一次；每次 actor step 后再次发送 current 权重；stale receiver 只在指定周期接收。

这条参数流有几个生产问题：

- 权重传输会占用网络与 GPU 通信资源，过于频繁可能抵消异步收益；
- rollout engine 更新时需要处理在途请求和 KV cache，必须定义新旧权重切换语义；
- current pool 内多个 worker 要保持一致版本，否则“current”本身也是策略混合；
- actor checkpoint 成功不代表 rollout 已同步相同版本；
- stale pool 需要明确版本号，不能只用布尔标签。

训练数据最好记录 `actor_version_at_update`、`rollout_version_at_first_action`、逐 action version 和参数同步完成时间。这样才能区分队列等待、长 episode 和同步延迟分别贡献多少 lag。

资源比例也应动态调优。训练早期模型可能生成更长、更错误的轨迹，环境资源成为瓶颈；后期成功路径缩短后 learner 可能成为瓶颈。固定 50/50 是公开示例，不是普遍最优。可依据 queue depth、rollout GPU utilization 和 learner idle time调整下一次作业的资源配比；运行中动态重分配则更复杂。

## 37. 从论文复现到真实工作启动：分阶段实施路线

如果团队要采用 AgentRL 思想，不应从 14B 五任务 H800 训练直接起步。更稳妥的路线分为六个阶段。

### 阶段 0：环境协议验证

只部署 Controller 和一个最小 task worker，用固定模型 API 手工跑 episode。验收：start/observation/end 可重试，session 不串扰，timeout 可清理，reward 与人工结果一致，history 可以完整回放。

### 阶段 1：单任务离线轨迹与评估

使用 `agentrl-eval` 或等价脚本采集 base model 轨迹，不做 RL。建立成功率、Completed、Task Limit、非法工具调用、平均轮数、P95 时长和环境错误率基线。先修 reward 与环境，不要用 RL 掩盖协议问题。

### 阶段 2：同步小规模 GRPO

在一个可验证任务上使用 3B 级模型与小 group，验证 `group_id`、reward、loss mask、rollout log-prob、reference KL 和 checkpoint。确保训练后的提升可在独立验证集复现。

### 阶段 3：异步 Rollout-Training

拆分 rollout 与 actor 资源组，引入有界 queue。先关闭 stale pool，比较同步与异步在同 GPU 数、同样本预算下的吞吐和最终效果。记录 queue age 与 policy lag，而不是只看 tokens/s。

### 阶段 4：Cross-Policy 与任务归一化

先在一个探索瓶颈明确的任务上开启 stale engine，扫描 stale ratio 与 update interval。随后引入第二任务，比较全局归一化与按任务归一化。每次只改变一个部件，避免无法解释收益。

### 阶段 5：多任务灰度与故障演练

逐个加入环境，保持任务级 dashboard。演练 worker 丢失、Controller 重启、模型同步失败、环境超时、错误 reward 和 checkpoint 恢复。训练系统必须能暂停发新 episode、排空或丢弃陈旧队列、恢复明确的 policy version。

### 阶段 6：生产训练治理

固定模型、tokenizer、任务镜像、Controller、worker、reward 和配置版本；保存训练数据清单、代码 commit 和指标快照。建立 cost per successful trajectory、环境资源配额、敏感数据治理和回滚流程。

## 38. 生产监控：至少要覆盖四个平面

### 38.1 Rollout 平面

- issued/completed/failed trajectories per second；
- action Token、observation Token 与总序列长度分布；
- episode turns、wall-clock、timeout、task limit；
- current/stale engine 请求比例和逐任务成功率；
- SGLang batch、KV cache、取消请求和 OOM；
- invalid tool call、parser failure 和空 action。

### 38.2 Queue 与版本平面

- input queue、incomplete group、completed Buffer 深度；
- queue age P50/P95/P99；
- group completion time 与组内最慢/最快比；
- behavior-to-train policy version lag；
- stale pool 实际版本、参数同步耗时与失败次数；
- 因过期、截断或基础设施错误丢弃的 trajectory 数。

### 38.3 Learner 平面

- actor loss、entropy、approx KL、reference KL；
- PPO upper/lower clip fraction 与 ratio 分位数；
- gradient norm、optimizer step time、FSDP communication；
- 每任务 reward、advantage mean/std、有效 Token 数；
- task contribution to loss 和 task sampling/compute share；
- validation success rate 与 checkpoint regression。

### 38.4 Environment 平面

- Controller request latency、session 数与路由失败；
- worker heartbeat、container startup、cleanup 和 leaked session；
- task-specific tool latency、数据库/浏览器/模拟器错误；
- reward verifier mismatch 与人工抽检失败；
- 相同 sample 重试结果的一致性；
- environment version 与数据版本分布。

这些指标需要按 task、model version、behavior policy 和 termination reason 分桶。总体平均值会掩盖某个环境永久失败或 stale trajectory 集中在特定任务的问题。

## 39. 验收门槛：吞吐提升不能以训练质量为代价

一个可执行的异步上线门槛可以写成：

| 类别 | 最小验收条件 |
| --- | --- |
| 数据正确性 | action/observation mask 抽检 100% 正确；reward 与人工校验一致率达标 |
| 环境可靠性 | session 泄漏为 0；基础设施错误不计模型负奖励；超时可回收 |
| 异步效率 | 同硬件下 wall-clock throughput 明显优于同步基线；learner/rollout idle 降低 |
| 数据新鲜度 | queue age 和 policy lag 在预设上限内；超限数据有丢弃或降权策略 |
| 优化稳定性 | KL、clip fraction、entropy、gradient norm 无持续异常；多 seed 可重复 |
| 任务公平 | 各任务 Token、计算和 loss contribution 无长期失衡 |
| 模型质量 | 独立验证集不劣于同步 baseline；不能只比较训练 reward |
| 恢复能力 | checkpoint、Controller 重启、worker 丢失演练通过 |

Cross-Policy 应再增加专门门槛：stale trajectory 的成功率不能显著劣化；高 lag 数据不能全部被 PPO clip；开启后探索指标或最终成功率应有可重复收益；关闭 stale pool 的 kill switch 必须保留。

Task Advantage Normalization 也需要观测每任务标准差。若某任务有效 advantage 方差接近 0，简单除以很小标准差会放大噪声，尽管源码加了 `1e-6`。生产实现可以设置最小样本数、方差下限或退回未归一化，但这些属于论文外扩展，需单独实验。

## 40. 局限性与批判

### 40.1 “全异步”仍保留 group barrier

论文强调 rollout 和 training 解耦，这一点成立；公开 GRPO Buffer 又明确要求完整 group，这一点也成立。文章若只复述前者，会让读者误以为任意单条 trajectory 都可立即训练。对于极端长尾任务，prompt 内最慢样本仍可能成为主要瓶颈。

### 40.2 Cross-Policy 的理论论证较弱

支持集严格扩大的公式更像直觉假设，没有给出保证条件和证明。随机策略切换可能扩大有效路径，也可能破坏计划一致性。Fig. 8/9 提供经验支持，但任务和模型池有限。

### 40.3 Stale Policy 同时是探索源和偏差源

论文把旧策略视为多样性来源，异步 RL 文献通常又把 policy lag 视为需要控制的偏差。AgentRL 没有报告 version lag、Token ratio 分布或 stale interval 的系统消融，难以判断收益来自探索、正则化还是训练节奏变化。

### 40.4 实验任务仍然有限

五个环境覆盖具身文本、数据库、知识图谱、操作系统和购物，但都来自 AgentBench-FC 且使用统一 function-call 改造。它们不能代表真实浏览器、移动端、软件工程、研究 Agent 和有安全约束的企业系统。

### 40.5 API baseline 比较不完全对称

AgentRL 模型针对目标任务训练，GPT-5、Claude 和 DeepSeek 行主要是 prompting。表格能证明训练后的模型在该 benchmark 协议中表现更高，不能说明 32B 模型在通用能力上超过这些模型。

### 40.6 多任务结论基于五任务平均

一个模型平均匹配五个专家的逐列最佳值是强结果，但 DB 与 WebShop 仍有差距。新增更多冲突任务后是否维持，需要进一步实验。Task Advantage Normalization 也没有解决容量竞争和灾难性遗忘的全部机制。

### 40.7 复现信息仍不完整

论文公开硬件类型、最低 GPU 数、采样数和部分引擎，却缺少完整 seed、wall-clock、总 Token、优化器细节和论文 checkpoint。仓库示例很有价值，但路径是占位符，完整 AgentBench-FC 数据、镜像和大集群仍有较高门槛。

### 40.8 AutoGLM 是使用声明，不是公开消融

摘要称框架用于构建 AutoGLM，但没有给出生产流量、故障率、训练成本或独立对照。可以引用为作者声明，不能写成“AutoGLM 已公开证明 AgentRL 在生产稳定”。

### 40.9 安全和数据治理讨论不足

OS、数据库和 Web 任务可能执行有副作用的动作。论文重点在训练效率，没有展开容器逃逸、凭证隔离、网络出口、恶意 observation、敏感日志和工具权限。生产使用必须在环境层补齐最小权限和审计。

## 41. 复现清单

### 论文级复现

- 固定 arXiv v1、官方代码 commit 和 AgentBench-FC commit；
- 准备 Qwen2.5-Instruct 模型与兼容 tokenizer/chat template；
- 部署五个 task worker 和 Controller，逐任务验证 reward；
- 使用 SGLang、FSDP、Ray 和 NCCL；
- 记录 8-rollout GRPO、temperature 0.8、最大轮数与 `-0.2` 惩罚；
- 分别跑 base、完整 AgentRL、无 Cross-Policy、无 Task Advantage Normalization；
- 每任务至少按论文协议重复评估 4 次；
- 单独报告吞吐、成功率和基础设施错误。

### 源码级验证

- 检查 `Buffer.strict_group` 与 `group_id` 数量；
- 验证 observation Token 的 `loss_mask` 为 0；
- 验证 rollout log-prob 与 Token 对齐；
- 验证 current engine 每 step 更新、stale engine 按周期更新；
- 验证 `data_source` 正确区分五个任务；
- 验证 reference model 只 forward、不执行 optimizer step；
- 验证 checkpoint marker 与实际目录一致；
- 验证验证任务使用 current rollout 而非 stale pool。

### 不应声称完成的项目

- 只启动 simple-calculator 不能称为论文复现；
- 只跑 3B WebShop 不能称为五任务结果复现；
- 只通过 trainer unit test 不能称为 H800 吞吐复现；
- 只看到 reward 上升不能称为 Cross-Policy 理论得到验证；
- 没有逐任务独立评估不能称为 generalist agent；
- 没有生产证据不能称为复现 AutoGLM。

## 42. 代码阅读地图

| 层次 | 固定 commit 路径 | 阅读问题 |
| --- | --- | --- |
| 论文训练入口 | [`examples/training/agentrl_trainer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/agentrl_trainer.py) | current/stale worker、task norm、参数同步如何串联 |
| 示例配置 | [`configs/qw14b_waodk.yaml`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/configs/qw14b_waodk.yaml) | GPU 比例、group、并发、loss 和任务列表 |
| 异步任务 | [`components/task_manager.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/components/task_manager.py) | Ray Queue 如何转成环境任务 |
| 完成队列 | [`components/buffer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/components/buffer.py) | 完整 group barrier 在哪里 |
| Agent loop | [`agentic/loops.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/agentic/loops.py) | action、observation、mask 和 log-prob 如何生成 |
| Advantage | [`algorithms/advantage.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/algorithms/advantage.py) | GRPO 与后续新增 estimator 的边界 |
| Policy loss | [`algorithms/loss_funcs.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/algorithms/loss_funcs.py) | behavior ratio、PPO clip、entropy 和 KL |
| Rollout worker | `workers/async_sglang_worker.py` | SGLang engine 和异步生成 |
| Actor worker | `workers/fsdp_worker.py` | 模型、optimizer、microbatch 和 checkpoint |
| 参数传输 | `components/nccl_tensor_comm.py` | actor 到 rollout 的权重流 |
| Worker 聚合 | `workers/collective_handle.py` | Ray actor 与 rank 广播 |
| Controller | `controller/` | session、worker、路由、dashboard 和 snapshot |
| Task worker | `worker/` | Task 生命周期、HTTP/gRPC 和环境控制器 |

阅读顺序不应从底层 CUDA/NCCL 开始。先看 `agentrl_trainer.py` 建立数据流，再看 Buffer 和 Agent loop 确认语义，随后看 loss，最后才追 worker 与 Controller 的分布式细节。

## 43. 推荐阅读路径

若只用一小时理解论文，建议按以下顺序：

1. Abstract、Table 2、Fig. 2：先建立问题与系统分解。
2. Sec. 3.1 与 Fig. 3/4：理解异步边界和吞吐证据。
3. Sec. 3.2、Fig. 6、Appendix Cross-Policy formalization：理解跨策略的真实主张。
4. Sec. 3.3：理解任务级 Token advantage 标准化。
5. Table 3/4/5：区分目标任务收益、多任务专家对比和 OOD 泛化。
6. Table 6、Fig. 7/8/9：检查组件证据和案例边界。
7. `agentrl_trainer.py`、`buffer.py`、`loops.py`：验证论文术语在源码中的行为。
8. README 的 Controller/TaskWorker 部分：理解环境控制面。
9. 最后阅读 Appendix failure analysis 与 limitations，避免只记住 SOTA 数字。

延伸阅读可按问题选择：异步 actor-learner 看 IMPALA、A3C、AReaL；组相对优化看 GRPO、DAPO；长时域 credit assignment 看本站 [Long-Horizon Agentic RL 综述](/blog/long-horizon-reinforcement-learning-credit-assignment-agentic-rl/)；单轨迹异步稳定性看 [SAO 精读](/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning-paper-reading/)；环境协议可继续阅读 AgentBench、OpenHands 和 MobileRL。

## 44. 结论

AgentRL 解决的不是“如何写一个更复杂的 PPO 公式”，而是如何让多轮、多任务 Agent 的在线 RL 真正跑起来。它把 rollout GPU 与 learner GPU 解耦，用有界队列减少全局等待；用统一 function-call 与中央 Controller 管理异构环境；用 stale policy 构造跨策略探索；用任务级 advantage 标准化缓解联合训练干扰。

论文的证据也形成了较完整链条：Fig. 4 给出实际吞吐，Table 3 展示五任务收益，Table 4 检查一个多任务模型与五个专家，Table 5 检查未见函数调用任务，Table 6/Fig. 7 做组件消融，Fig. 8/9 解释 Cross-Policy 的行为机制。相比只报告最终 reward 的 Agentic RL 工作，这种系统与算法共同评估很有价值。

最需要保留的判断边界有四条：

1. 全异步指 rollout-training stage 解耦，不代表 GRPO prompt group barrier 消失。
2. stale policy 能增加探索，也会增加 off-policy drift；有界队列和 PPO clipping不是完整理论保证。
3. 多任务平均持平不等于每个任务都匹配专家，也不保证加入更多任务仍无负迁移。
4. 官方代码可读、可扩展，不等于论文 H800 训练、checkpoint 和 AutoGLM 生产系统已完整复现。

在这些边界内，AgentRL 的长期贡献相当清晰：**Agentic RL 的扩展能力来自训练调度、环境控制、数据语义和优化方法的共同契约。** 如果环境 session 不可靠、action mask 错误、行为策略证据缺失或任务采样失衡，再好的 RL loss 也只是在更快地学习错误数据。

## 参考文献与一手资料

1. Hanchen Zhang et al. [AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework](https://arxiv.org/abs/2510.04206). arXiv:2510.04206v1, 2025.
2. AgentRL paper [PDF](https://arxiv.org/pdf/2510.04206), [HTML](https://arxiv.org/html/2510.04206), [TeX source](https://arxiv.org/e-print/2510.04206).
3. THUDM. [AgentRL official repository](https://github.com/THUDM/AgentRL), 本文固定阅读 commit [`6a73409d`](https://github.com/THUDM/AgentRL/tree/6a73409d31ba695d383b978a8ad3ef400d90c054).
4. John Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
5. Zhihong Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). 2024.
6. Guangming Sheng et al. [HybridFlow / VeRL](https://arxiv.org/abs/2409.19256). 2024.
7. Jiaxuan Guo et al. [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298). 2025.
8. Xiao Liu et al. [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688). 2023.
9. Zhenyu Hou et al. [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508). 2026.
10. Shridhar et al. [ALFWorld: Aligning Text and Embodied Environments for Interactive Learning](https://arxiv.org/abs/2010.03768). 2021.
11. Yao et al. [WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://arxiv.org/abs/2207.01206). 2022.
12. Patil et al. [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html).

## 附录 A：实现审查清单

在提交 AgentRL 风格训练任务前，可以逐项检查：

- [ ] 每个 prompt 的 `group_id` 唯一且 8 条 rollout 不跨 prompt 混组。
- [ ] system/user/observation Token 的 `loss_mask` 为 0，assistant action 为 1。
- [ ] 每个 action Token 保存真实行为策略 log-prob，没有用 current actor 事后伪造。
- [ ] Token 序列、mask、position id 和 log-prob 长度完全相同。
- [ ] 超过长度时不会留下与 mask 错位的半个工具调用。
- [ ] 环境异常与模型失败使用不同 termination reason。
- [ ] reward verifier 对重复执行是确定或至少可审计的。
- [ ] incomplete group 有超时和清理策略。
- [ ] current/stale engine 都有明确 policy version。
- [ ] 参数同步失败不会继续把 worker 标成 current。
- [ ] queue 有上限，并监控 pending group 与 completed queue 两种容量。
- [ ] 按任务统计 prompt、trajectory、Token、wall-clock 和 loss weight。
- [ ] task advantage normalization 只使用对应任务的有效 action Token。
- [ ] 标准差接近 0 时有诊断，不只依赖 `1e-6` 静默放大。
- [ ] reference model 不更新参数，版本与训练起点可追溯。
- [ ] validation 不使用 stale engine，也不混入训练队列。
- [ ] checkpoint 包含 actor、optimizer、global step 和版本元数据。
- [ ] 恢复训练时 Controller/环境队列处理策略明确。
- [ ] W&B 或其他日志中不泄漏敏感 observation、凭证或用户数据。
- [ ] 训练完成后用固定环境版本做独立评估，不直接报告训练 reward。

## 附录 B：常见误读速查

| 误读 | 正确表述 |
| --- | --- |
| AgentRL 每条轨迹完成就立即训练 | 完整 GRPO group 完成后才能进入公开 Buffer；不同 group 可异步 |
| Cross-Policy 就是混合两个模型的数据 | Mix 混合完整轨迹；Cross 在同一 trajectory 的 step 间切换 policy |
| stale engine 是纯系统优化 | 它是训练中近似 Cross-Policy 的算法部件，也引入 off-policy drift |
| Task Advantage Normalization 替代 GRPO | 它在 GRPO advantage 之后按任务做第二层 Token 标准化 |
| 32B 超过 GPT-5 说明基础模型更强 | 32B 针对五任务做了 RL，GPT-5 是 prompting baseline |
| 多任务模型每项都达到专家最好 | 平均 `67.7` 接近专家拼接 `67.8`，DB/WebShop 仍略低 |
| Completed 就是任务成功 | Completed 只表示已提交答案，正确性由 reward 另行判断 |
| 仓库当前功能都是论文实现 | gRPC、K8s、eval、dashboard 等部分能力属于后续演进 |
| Controller `v0.2.0` 是论文模型版本 | 它是 Controller 子组件 release，不是统一模型/框架版本 |
| 官方代码存在就等于论文可一键复现 | 模型、数据、镜像、H800 集群和训练 checkpoint 仍是重要前置 |

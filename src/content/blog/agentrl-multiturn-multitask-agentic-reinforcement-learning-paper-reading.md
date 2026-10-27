---
title: "AgentRL 论文精读：全异步 Rollout-Training、跨策略采样与多任务 Agent 强化学习"
description: "精读 AgentRL 如何通过全异步多轮训练、统一环境控制面、Cross-Policy Sampling、Task Advantage Normalization 和开源训练框架扩展多任务 Agentic RL"
pubDate: "2026-07-15T11:49:06+08:00"
updatedDate: "2026-07-15T11:49:06+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "Reinforcement Learning"
  - "Agentic RL"
  - "Asynchronous RL"
  - "Multi-Task Learning"
  - "Code Reading"
draft: false
---

大语言模型强化学习正在从单轮文本生成拓展至复杂环境中的连续多轮决策。一个 WebShop 智能体需要搜索、筛选、比价并下单购买商品；一个数据库智能体需要先理解自然语言查询，再多次编写 SQL、读取执行结果并修正查询；一个操作系统智能体则要在终端里查看文件、运行命令并处理权限或路径报错。这类轨迹由多轮模型生成与环境反馈交替构成，单条 episode 的耗时可能相差数十倍，且奖励分布、动作空间与失败模式因任务而异。

这些特征使 Agentic RL 同时面临分布式系统与强化学习算法的双重挑战：同步批处理会被最慢的长尾轨迹拖住；异构环境缺少统一的会话生命周期与函数调用协议；单一策略反复在自身采样轨迹上强化，容易导致探索过早收缩；而将多个任务混合训练时，奖励尺度与收敛速度的差异又会引发跨任务梯度干扰。

**AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework** 将上述系统与算法瓶颈纳入同一框架协同求解：在系统侧解耦 rollout 与 learner 资源组，并通过中央 Controller 统一调度容器化环境；在算法侧提出 Cross-Policy Sampling（跨策略采样），在多轮轨迹的不同步骤引入不同策略版本的动作以拓宽探索覆盖面，再配合 Task Advantage Normalization（任务级优势归一化）平衡多任务梯度尺度。借助官方开源的训练与环境部署代码，我们可以深入审计其异步队列边界、陈旧策略采样机制以及 Token 级掩码实现。

本文以预印本 `arXiv:2510.04206v1` 与官方仓库固定提交 [`6a73409d`](https://github.com/THUDM/AgentRL/tree/6a73409d31ba695d383b978a8ad3ef400d90c054) 为分析基线，明确区分论文原始实验设定与仓库后续演进（如 gRPC 传输、Kubernetes 控制器及实验性评估套件）。全文聚焦于论文方法与源码架构拆解，不涉及 H800 集群重训或 AutoGLM 生产管线的独立验证。

## 1. 核心贡献、问题空间与论文概览

### 1.1 四大核心组件与论文信息

AgentRL 的核心架构由四个紧密协同的模块组成：

1. **Fully-asynchronous rollout-training**：推理生成（rollout engine）与参数更新（learner worker）部署在独立的 GPU 资源组上并发执行，learner 从完成队列中持续拉取可用样本组，消除全局大批次等待屏障。
2. **Environment deployment framework**：基于统一的 OpenAI Function Call 接口、容器化任务工作节点（TaskWorker）与中央控制器（Controller），集中管理会话路由、环境生命周期、观测返回与奖励结算。
3. **Cross-Policy Sampling**：在单条多轮轨迹的不同环境步骤（step）间交替调用策略池中的不同模型生成动作，扩展可达状态空间；在 RL 训练中，通过周期性延迟同步的陈旧推理引擎（stale rollout engine）高效近似策略池。
4. **Task Advantage Normalization**：在计算 GRPO 组内相对优势之后，按任务类别汇总批次内所有有效动作 Token 的优势值并分别执行标准化，缓解多任务联合训练中的尺度失衡与梯度干扰。

| 项目 | 内容 |
| --- | --- |
| 题名 | AgentRL: Scaling Agentic Reinforcement Learning with a Multi-Turn, Multi-Task Framework |
| 作者 | Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, Zhenyu Hou 等 |
| 机构 | Tsinghua University, Z.AI |
| arXiv | `2510.04206v1`（2025-10-05，CC BY 4.0，DOI: `10.48550/arXiv.2510.04206`） |
| 官方代码与基线 | [THUDM/AgentRL](https://github.com/THUDM/AgentRL)（本文代码审计固定 commit `6a73409d`，2026-01-17，MIT 许可） |
| 主实验配置 | Qwen2.5 3B/7B/14B/32B、GLM-4-9B，基于 AgentBench-FC 的五类多轮交互环境 |

### 1.2 从单轮 RLVR 到多轮多任务 Agentic RL 的结构性变化

单轮可验证奖励强化学习（RLVR）的数据流相对规整：给定 prompt，模型一次性生成完整回答，校验器打分后即可进入梯度更新。而在多轮 Agentic RL 中，单条 episode 需要在模型与外部环境之间反复循环：

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

这一交互范式在系统与算法层面引发了五方面深刻变化：

- **交互耗时受环境与策略双重影响**：简单样本可能一次工具调用即宣告完成，复杂或陷入死循环的样本则会持续交互至 20 轮上限；同步批处理下，提前完成的硬件通道只能闲置等待。
- **环境具备有状态会话（Stateful Session）**：数据库、操作系统与购物网站均维护内部状态，要求控制面支持容器启动、会话绑定、心跳检测、超时中断与沙箱回收。
- **动作 Token 与环境观测 Token 语义分离**：模型生成的 assistant action 属于策略决策，需计入策略梯度；而环境返回的 tool observation 虽作为后续轮次的条件上下文，但并非由策略采样产生，必须通过 Token 级 `loss_mask` 严格屏蔽。
- **跨任务奖励与轨迹结构高度异构**：不同任务的初始成功率、平均轮数与有效动作 Token 数量差异显著，若直接对全批次优势值做全局统一归一化，高方差或长序列任务将主导梯度方向。
- **异步流水线引入策略版本滞后（Policy Lag）**：在长轨迹与环境交互期间，learner 可能已完成多步参数更新，使得采样时刻的行为策略（rollout policy）、更新时刻的当前策略（current actor）与计算 KL 惩罚的参考策略（reference policy）分属不同版本。

### 1.3 问题空间界定（Table 1 & Table 2）与总体表现预览（Fig. 1 & Fig. 2）

![AgentRL 与其他强化学习框架和方法的能力对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table01-framework-comparison.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 1, CC BY 4.0.*

Table 1 从 Multi-Turn、Multi-Task、Full-Async、Interactive Environments 与 Heterogeneous Environments 五个维度将 AgentRL 与既有框架对照，强调大规模多任务智能体训练需要异步训练引擎与异构环境控制面同时就位。

![单轮强化学习与 Agentic RL 的挑战对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table02-challenges.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 2, CC BY 4.0.*

Table 2 进一步将单轮、多轮与多任务场景下的核心挑战拆解为基础设施（Infrastructure）与算法（Algorithm）两个正交维度：

| 场景 | 基础设施瓶颈 | 算法优化挑战 |
| --- | --- | --- |
| Single-Turn | 同步 rollout 易于管理 | 训练相对平稳，易于扩展 |
| Multi-Turn | 同步等待造成算力空泡；大规模并发交互环境管理复杂 | 状态空间剧增，训练过程中策略探索能力易退化 |
| Multi-Task | 异构环境难以统一部署与接入 | 跨任务梯度干扰与泛化能力受限 |

![AgentRL 在五个环境上的增益和训练过程](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig01-overall-performance.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 1, CC BY 4.0.*

Fig. 1 展示了 `Qwen2.5-32B` 在五类环境上经 AgentRL 训练前后的成功率跃升：OS（`+14.7`）、WebShop（`+31.1`）、DB（`+14.6`）、KG（`+43.2`）与 ALFWorld（`+62.4`），五任务平均成功率从未经 SFT 基座的 `37.2%` 提升至 `70.4%`（Fig. 1b）。图中虚线标注了 Claude-Sonnet-4、GPT-5 与 DeepSeek-R1 的零样本/提示式评测得分作为外部参考。

![AgentRL 总体框架](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig02-framework-overview.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 2, CC BY 4.0.*

Fig. 2 给出了涵盖四条核心数据流的总体架构图：

1. **任务调度流**：训练样本进入异步 Task Manager，在目标环境中初始化具体会话；
2. **多轮交互流**：rollout engine（含策略池 $M_1, M_2$ 跨策略切换）生成动作，经 Controller 路由至容器化 TaskWorker，获取环境观测与奖励；
3. **异步训练流**：完整采样组写入 Buffer，learner 拉取就绪批次，依次完成 GRPO 优势计算、Task Advantage Normalization、参考模型前向与策略梯度更新；
4. **参数同步流**：actor 更新后的权重通过 NCCL 实时广播至 current rollout engine，并按固定步数间隔同步至 stale rollout engine。

## 2. 强化学习形式化与全异步 Rollout-Training 架构

### 2.1 组合状态 MDP、Token 级优化粒度与 PPO/GRPO 基座

论文将第 $i$ 个智能体任务形式化为马尔可夫决策过程（MDP）：

$$
\mathcal{T}_i = \left(\mathcal{S}_i^{\mathrm{env}},\mathcal{A}_i,P_i,r_i,\rho_i\right),
$$

其中 $\mathcal{S}_i^{\mathrm{env}}$ 为外部环境状态，$\mathcal{A}_i$ 为高层动作空间，$P_i$ 为状态转移函数，$r_i$ 为奖励函数，$\rho_i$ 为初始状态分布。由于大语言模型只能观测文本序列，论文定义联合状态：

$$
s_t = \left(s_t^{\mathrm{env}},s_t^{\mathrm{ctx}}\right),
$$

其中语言上下文 $s_t^{\mathrm{ctx}}\in\mathcal{V}^*$ 保存当前 episode 截至第 $t$ 步的对话历史、工具调用及环境反馈。模型在第 $t$ 步输出的高层动作由 $L_t$ 个 Token 组成：

$$
a_t=(y_{t,1},y_{t,2},\ldots,y_{t,L_t}),
\qquad
\pi_\theta(a_t\mid s_t)
=\prod_{k=1}^{L_t}P_\theta\left(y_{t,k}\mid s_t^{\mathrm{ctx}},y_{t,<k}\right).
$$

这一分解确立了 **“环境以完整 step 交互，策略以单个 Token 优化”** 的双层粒度。单条完整轨迹记作 $\tau=\left(s^{(0)},a^{(0)},r^{(1)},s^{(1)},\ldots,a^{(T-1)},r^{(T)},s^{(T)}\right)$。在多任务集合 $\mathcal{T}=\{\mathcal{T}_1,\ldots,\mathcal{T}_{N_{\mathrm{task}}}\}$ 中，系统对每个训练样本 $x_{i,j}$ 生成 $K_{i,j}$ 条轨迹，构成 GRPO 采样组：

$$
G_{i,j}=\{\tau_{i,j,1},\ldots,\tau_{i,j,K_{i,j}}\}.
$$

在策略优化层，AgentRL 沿用标准 PPO 裁剪目标与 GRPO 组内相对优势估计：

$$
\mathcal{L}_{\mathrm{PPO}}(\theta)=
\mathbb{E}_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right],
\qquad
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)},
$$

$$
\hat A_{i,j,g}=
\frac{\hat R_{i,j,g}-\operatorname{mean}(\hat R_{i,j})}
{\operatorname{std}(\hat R_{i,j})}.
$$

公开源码的 [`ppo_loss()`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/algorithms/loss_funcs.py) 直接读取生成阶段保存的 `rollout_log_prob` 与当前 actor 的 Token 对数概率计算比率 $r_t(\theta)$，并附加基于冻结参考模型的 KL 正则项。采用 GRPO 省去了与策略模型同规模的 Critic 显存开销，但也决定了同一样本的 $K=8$ 条轨迹必须全部完成，才能算出组内均值与标准差。

### 2.2 全异步 Rollout-Training 流水线与吞吐验证（Fig. 3 & Fig. 4）

![同步与全异步训练架构对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig03-sync-vs-async.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 3, CC BY 4.0.*

Fig. 3 对比了同步流水线与 AgentRL 全异步架构的硬件调度：在同步架构（Fig. 3a）中，同一批次内的短轨迹必须等待多轮长轨迹结束，全部 GPU 才能切换至训练阶段，时间轴上存在大量空白空泡（pipeline bubble）；而在全异步架构（Fig. 3b）中，GPU 集群被物理切分为独立的 rollout 资源组与 training 资源组：

```text
rollout group:    continuously execute environment sessions ------>
learner group:         pull completed groups -> update -> pull -> update ------>
parameter stream:                           push latest actor weights ------>
```

通过双端解耦与有界数据队列，learner 在更新参数的同时，rollout GPU 仍在持续推进未完成的环境会话。

![AgentRL 异步训练吞吐与同步基线对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig04-throughput.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 4, CC BY 4.0.*

Fig. 4 在 `Qwen2.5-14B` 与 WebShop 环境上实测了 16、32、64 张 GPU 规模下的系统吞吐率（千 Token/秒，双对数坐标）：

| GPU 数量 | 同步基线吞吐 | AgentRL 异步吞吐 | 相对加速比 |
| ---: | ---: | ---: | ---: |
| 16 | ~9K tokens/s | ~17K tokens/s | ~1.89× |
| 32 | ~16K tokens/s | ~33K tokens/s | ~2.06× |
| 64 | ~29K tokens/s | ~56K tokens/s | ~1.93× |

实验表明，在该配置下 AgentRL 取得了接近同步基线 **2 倍** 的吞吐率，并随硬件规模扩展保持良好的近似线性加速。

### 2.3 源码审计：Group-Aware Buffer 的真实同步边界与参数同步机制

审视官方仓库的 [`buffer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/components/buffer.py)，可以精准定位 AgentRL 中“全异步”的真实粒度：

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

当启用 GRPO 且 `group_size = 8` 时，`strict_group` 为 `True`。此时单条已完成的轨迹首先暂存于 `self.groups[group_id]` 字典中，**只有当同一 `group_id` 凑齐全部 8 条轨迹后，整组样本才会一次性推入可消费队列 `self.queue`**。因此，AgentRL 的异步调度具有清晰的层次边界：

- **跨样本 / 跨任务级完全异步**：不同 prompt 的采样组互不阻塞，先凑齐 8 条轨迹的组优先进入 `self.queue` 供 learner 消费；
- **单样本组内局部同步**：同一 prompt 的 8 条 rollout 仍受组内最慢轨迹制约，以此换取 GRPO 无需维护 Critic 的无 Critic 优势。

在参数同步方面，trainer 按 `rollout_ratio`（示例配置为 `0.5`）将集群 GPU 划分为 SGLang 推理池与 FSDP 训练池。Actor 的 rank 0 节点通过 `NCCLTensorSender` 与各 rollout worker 的 `NCCLTensorReceiver` 建立直连通信：每完成一步梯度更新，最新权重即通过 NCCL 广播至 current rollout 池，而 stale rollout 池仅在满足 `global_step % stale_step == 0` 时更新。训练过程中，系统可根据队列积压深度、rollout GPU 利用率与 learner idle time 调整下一轮作业的推理/训练卡数配比。

## 3. 统一环境部署控制面与多轮轨迹数据契约

### 3.1 训练框架与环境部署框架的分层解耦（Fig. 5）

![AgentRL 训练框架与环境部署框架](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig05-environment-deployment-framework.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 5, CC BY 4.0.*

如 Fig. 5 所示，AgentRL 将整体系统解耦为左侧的 **Training Framework** 与右侧的 **Environment Deployment Framework**，中间通过 **AgentRL Controller** 桥接。这种分层设计源于大模型算力与环境沙箱的扩缩容规律截然不同：

- **模型推理与训练层**受限于显存容量、KV Cache、FSDP 梯度通信与微批次大小；
- **环境工作节点**（TaskWorker）受限于 CPU 核数、磁盘 I/O、数据库连接池、浏览器实例或容器生命周期；
- **中央控制器**（Controller）负责维护全局会话路由表、心跳探活与负载均衡，使训练侧无需感知底层究竟是 SQL 数据库还是 ALFWorld 模拟器。

### 3.2 统一 OpenAI Function-Call 接口与 Agent Loop 的 Token Mask 构造

AgentRL 将 AgentBench 中的五类异构环境统一封装为 OpenAI Function Call 协议，每个 TaskWorker 对外暴露三个核心生命周期原语：

```text
start(task sample)
  -> session_id + initial messages + tool schemas

observation(session_id, assistant message / tool call)
  -> new messages + reward + status + finish flag

end(session_id, done)
  -> release or recycle container session state
```

在训练侧，[`openai_chat_agent_loop()`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/agentic/loops.py) 负责将多轮交互实时转化为对齐的训练张量。会话启动时，初始系统提示词与用户问题的 `loss_mask` 全部置为 `0`。随后每轮模型生成动作时：

```python
text, received_log_probs = await gen_fn(input_ids=ids)
new_ids = [token_id for log_prob, token_id in received_log_probs]
new_log_probs = [log_prob for log_prob, token_id in received_log_probs]

ids += new_ids
loss_mask += [1] * len(new_ids)
log_probs += new_log_probs
```

模型输出的动作 Token 被标记为 `loss_mask = 1`，并同步记录采样时刻的 `rollout_log_prob`。接着，`FunctionCallParser` 解析工具调用并发送至环境，环境返回的观测消息经由 chat template 差分编码追加至序列末尾：

```python
ids += observation_ids
loss_mask += [0] * len(observation_ids)
log_probs += [0] * len(observation_ids)
```

环境观测 Token 的 `loss_mask` 恒为 `0`，确保它们仅作为后续动作的条件上下文而不直接产生策略梯度。当会话达到最大轮数或最大长度仍未处于 `completed` 状态时，循环自动终止并赋予 `incomplete_punishment`（公开配置设为 `-0.2`）。

### 3.3 Action/Observation 训练数据契约与状态转移审查要点

结合论文形式化定义与源码实现，一条完整的多轮训练样本需满足如下数据契约与异步状态转移不变量：

```json
{
  "trajectory_id": "kg-train-1847-g3-v412",
  "prompt_id": "kg-train-1847",
  "group_id": "kg-train-1847-v412",
  "task": "kg-env_train-v2",
  "input_ids": [151644, 8948, 198, 2610],
  "position_ids": [0, 1, 2, 3],
  "loss_mask": [0, 0, 1, 0],
  "rollout_logprobs": [0.0, 0.0, -0.284, 0.0],
  "reward": 1.0,
  "status": "completed",
  "termination_reason": "environment_success"
}
```

在工程实现与代码审查中，需重点保障以下核心不变量：

1. **序列张量等长与掩码隔离**：`input_ids`、`position_ids`、`loss_mask` 与 `rollout_logprobs` 长度严格一致；system/user/observation 位置的 `loss_mask` 必须为 `0`，仅 assistant action 位置为 `1` 且保存真实采样对数概率。
2. **组标识与超时回收**：同一 prompt 的 8 条 rollout 共享唯一 `group_id`；若某条子轨迹因容器崩溃或死锁长期未归，Task Manager 需触发超时清理或重试，避免残缺 group 永久滞留于 `Buffer.groups` 字典中造成内存泄漏。
3. **终止状态与奖励解耦**：明确区分策略逻辑导致的任务超限（赋予 `-0.2` 惩罚）与底层容器/网络故障（标记为基础设施异常并剔除，避免模型拟合系统噪声）。
4. **评估与训练引擎隔离**：验证集评估（validation）必须固定路由至最新权重的 current rollout engine，且评估轨迹不得混入训练 Buffer。

## 4. 跨策略采样（Cross-Policy Sampling）与离策边界

### 4.1 探索收缩成因、跨策略轨迹公式与支持集扩展直觉

在多轮环境中，智能体早期的动作选择决定了后续可访问的状态子图。若仅用单一策略自采样训练，随着策略分布逐渐尖锐（entropy 下降），同一 prompt 采出的 8 条轨迹往往陷入相同的局部套路，导致 GRPO 组内奖励方差趋近于零，探索能力迅速退化。

AgentRL 提出 **Cross-Policy Sampling**，不再由单一模型包揽整条轨迹，而是维护一个策略集合 $\mathcal{M}$，在轨迹的每个环境步骤 $t$ 随机抽取一个策略生成当前动作：

$$
\tau^c=\left(s^{(0)},a^{c,(0)},r^{(1)},s^{(1)},\ldots,s^{(T)}\right),
\qquad
a^{c,(t)}\sim \operatorname{random}(\mathcal{M})(\cdot\mid s^{(t)}).
$$

论文附录从可达状态支持集（support set）的角度解释了其直觉：设 grounding 函数 $\Gamma:\mathcal{L}_{\mathrm{valid}}\rightarrow\Delta(\mathcal{S}^{\mathrm{env}})$ 将有效语言上下文映射为环境状态分布，目标成功状态集合为 $\mathcal{G}$，其语言原像为 $\mathcal{L}_{\mathcal{G}}=\Gamma^{-1}(\mathcal{G})\cap\mathcal{L}_{\mathrm{valid}}$。当策略池中不同模型具备互补的条件动作分布（例如模型 A 擅长高层规划但容易在工具语法上卡壳，模型 B 擅长工具调用格式却缺乏全局分解能力）时，步骤级交替采样能够组合出任一单模型均难以独立走通的成功路径：

$$
\operatorname{supp}(\tau^c)\cap\mathcal{L}_{\mathcal{G}}
\supsetneqq
\bigcup_m\left(
\operatorname{supp}(\tau^m)\cap\mathcal{L}_{\mathcal{G}}
\right).
$$

这一包含关系成立的前提是各策略共享统一的上下文与工具调用协议，且局部动作切换不会彻底破坏多轮计划的语义连贯性。

### 4.2 Single、Mix 与 Cross 采样对比（Fig. 6）及 Stale Engine 工程近似

![Single、Mix 与 Cross-Policy 三种采样策略](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig06-rollout-strategies.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 6, CC BY 4.0.*

Fig. 6 区分了三种多模型采样方式：

- **Single**：每条轨迹从头至尾由单一模型生成；
- **Mix**：轨迹级混合——数据集中一部分完整轨迹由模型 1 生成，另一部分完整轨迹由模型 2 生成，其可达路径仍受限于各模型独立支持集的简单并集 $\bigcup_m \operatorname{supp}(\tau^m)$；
- **Cross**：步骤级混合——在同一条轨迹的不同交互轮次间动态切换模型 1 与模型 2，实现 episode 内部的行为组合。

在在线 RL 训练中，同时驻留多个异构大模型（如同时部署 Qwen 与 Llama）会带来高昂的显存与维护成本。因此，AgentRL 采用 **同构模型的时间版本池（Current + Stale Rollout Engines）** 来近似跨策略采样。在 [`agentrl_trainer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/agentrl_trainer.py) 中，每次单轮生成调用通过 `cross_sampler` 按概率路由：

```python
def cross_sampler(item, **kwargs):
    if rollout_stale is None:
        return rollout.dispatch_rank(...).generate(**kwargs)
    if random.random() < (1 - stale_ratio):
        return rollout.dispatch_rank(...).generate(**kwargs)
    return rollout_stale.dispatch_rank(...).generate(**kwargs)
```

在 14B 五任务公开配置（[`qw14b_waodk.yaml`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/configs/qw14b_waodk.yaml)）中：

```yaml
rollout_ratio: 0.5
rollout_stale_ratio: 0.25
stale_step: 25
task_adv_norm: true
```

系统将一半 GPU 划归 rollout，再从中切出 `25%` 作为 `rollout_stale` 引擎（每 25 个训练步同步一次权重），并在每个交互 step 以 `0.25` 的概率调用 stale 引擎生成动作。

### 4.3 探索多样性与 Off-Policy 偏移的权衡

引入 stale engine 意味着系统主动将离策（off-policy）动作注入当前轨迹。在训练更新时，每个动作 Token 的 `rollout_log_prob` 均由实际生成它的引擎（current 或滞后至多 25 步的 stale 引擎）记录，随后在 `ppo_loss()` 中与当前策略 $\pi_\theta$ 计算重要性比率 $r_t(\theta)$ 并执行标准 PPO 裁剪（`clip_ratio = 0.2`），同时配合 `kl_loss_coef = 1e-4` 的参考模型 KL 约束控制整体策略漂移。

这种设计利用适度的历史版本差异拓宽了动作多样性，但也要求 `stale_step` 不能过大：若陈旧策略与当前策略的分布距离过远，大量 stale 动作对应的比率 $r_t(\theta)$ 将落入 PPO 裁剪饱和区，从而丧失有效梯度。

## 5. 任务级优势归一化（Task Advantage Normalization）与多任务平衡

### 5.1 GRPO 组内归一化之后的任务级 Token 标准化

在多任务联合训练中，不同环境的奖励稀疏度、成功轮数与回复长度差异巨大。即便先经过 GRPO 组内归一化，各任务在一个训练批次内的有效 Token 优势分布仍可能存在明显的均值偏移与方差失衡。为此，AgentRL 在 GRPO 之后引入第二层归一化——**Task Advantage Normalization**。

具体而言，针对批次中的每个任务 $i$，收集该任务下所有有效动作 Token（即 `loss_mask == 1`）的组相对优势集合：

$$
\mathcal{A}_i^{\mathrm{tok}}
=\left\{
\hat A_{i,s,g,t,k}
\;\middle|\;
1\le s\le S_i,\ 
1\le g\le K_{i,s},\ 
1\le t\le T_{i,s,g},\ 
1\le k\le L_{i,s,g,t}
\right\},
$$

然后按任务独立计算均值与标准差，对每个 Token 的优势值做标准化：

$$
\tilde A_{i,s,g,t,k}
=\frac{\hat A_{i,s,g,t,k}-\mu_i}{\sigma_i},
\qquad
\mu_i=\operatorname{mean}(\mathcal{A}_i^{\mathrm{tok}}),
\qquad
\sigma_i=\operatorname{std}(\mathcal{A}_i^{\mathrm{tok}}).
$$

这两层归一化各司其职：**GRPO 组内归一化**在同一 prompt 的 8 条轨迹之间评判相对优劣；**Task Advantage Normalization** 则在任务维度拉齐不同环境在当前批次内的 Token 级梯度尺度（源码实现中分母附加 `1e-6` 防止零方差除零）。

### 5.2 多任务数据混合与采样平衡口径

在数据源构建上，trainer 为五个环境分别实例化独立的 DataLoader，对每个样本重复 `n = 8` 次以赋予相同的 `group_id`，再通过 `interleave()` 与无限 `cycle()` 均匀交错各任务的 prompt 流。

需要注意的是，“发出的 prompt 数量均衡”并不自动等同于“优化过程中的 Token 权重均衡”：交互轮数多、单步回复长的任务（如 ALFWorld 或 WebShop）天然产生更多有效动作 Token，而响应极快的环境又会在异步队列中更早凑齐完整 group。Task Advantage Normalization 对齐了各任务的单位 Token 优势方差，在工程监控中仍需按任务维度分别追踪完成组数与有效 Token 占比。

### 5.3 训练源码主链拆解：从配置解析到单步 Actor 更新

结合 [`examples/training/agentrl_trainer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/agentrl_trainer.py) 与 `qw14b_waodk.yaml`，AgentRL 的完整训练主链可归纳为以下核心步骤与参数设定：

1. **资源组初始化**：启动 Ray 集群，按 `rollout_ratio` 与 `rollout_stale_ratio` 创建三个 placement group（current rollout、stale rollout 以及共享 GPU 的 actor/reference FSDP 组）。
2. **通信建立与任务派发**：构建 SGLang 引擎与 FSDP 模型，注册 Actor rank 0 到全体 rollout worker 的 NCCL 权重广播通道；启动 `DistributedTaskManager` 消费交错的多任务数据流。
3. **异步采样与成组入队**：并发协程通过 `cross_sampler` 与 Controller 驱动多轮环境交互，生成的轨迹写入 `Buffer`，凑齐 8 条同 `group_id` 样本后进入就绪队列。
4. **双层优势计算与参数更新**：learner 从 `Buffer` 拉取至少 `real_bsz = batch_size * n` 条轨迹（首步预热要求 `1.7x`），依次执行 GRPO 组内优势计算与按 `data_source` 分组的 `adv_norm()`，随后完成 reference 前向、actor 前反向传播及优化器更新，并按步数策略同步两级 rollout 池。

| 配置参数 | 设定值 | 语义说明 |
| --- | ---: | --- |
| `max_steps` | 1000 | 训练迭代总步数上限 |
| `train.batch_size` / `train.n` | 256 / 8 | 单步基准 prompt 数与每 prompt 采样组大小（最小拉取轨迹数 $256\times 8 = 2048$） |
| `train.concurrency` | 2048 | 并发执行的异步环境会话协程上限 |
| `max_total_len` / `max_turns` | 8192 / 20 | 单条轨迹最大上下文 Token 数与最大环境交互轮数 |
| `actor.fsdp_size` / `max_tokens_per_micro_batch` | 16 / 16384 | Actor FSDP 分片规模与单微批次最大 Token 容量 |
| `lr` / `clip_ratio` / `kl_loss_coef` | `1e-6` / `0.2` / `1e-4` | 学习率、PPO 裁剪阈值与参考模型 KL 正则系数 |

## 6. 实验设置与多任务评测结果分析

### 6.1 五类异构环境（AgentBench-FC）与训练配置

论文实验基于改造为统一函数调用接口的 AgentBench-FC 五大环境：

| 缩写 | 环境名称 | 核心考察能力 | 典型失败模式 |
| --- | --- | --- | --- |
| AF | ALFWorld | 家居具身常识规划、多步物体搜寻与操作 | 找错容器位置、动作死循环、耗尽轮数上限 |
| DB | Database | 自然语言转 SQL、表结构探测与结果校验 | SQL 语法/连接错误、未根据空结果修正查询 |
| KG | Knowledge Graph | 实体关系检索、多跳图谱查询与工具参数构造 | 关系跳数闭合失败、工具名称或参数类型错误 |
| OS | Operating System | Bash 命令行规划、文件系统与进程状态操作 | 命令执行报错、路径或权限判断失误 |
| WS | WebShop | 电商搜索、属性筛选、页面跳转与商品匹配下单 | 遗漏用户属性约束、搜索词循环、提前买错商品 |

实验模型涵盖 `Qwen2.5-3B/7B/14B/32B-Instruct`（直接从 Instruct 模型启动 RL，无额外 SFT 预热）以及 `GLM-4-9B`（因原生函数调用格式差异，先用少量冷启动 SFT 数据对齐协议）。训练在 H800 GPU 集群上进行（14B 模型至少使用 16 张 H800），训练与评估采样温度均设为 `0.8`，每项任务重复评估 4 次并报告均值。

### 6.2 五任务主实验结果（Table 3）与多任务 vs. 单任务专家对比（Table 4）

![AgentRL 五任务主实验结果](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table03-main-results.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 3, CC BY 4.0.*

Table 3 核心实验结果汇总如下：

| 模型 | ALFWorld | DB | KG | OS | WebShop | AVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude-Sonnet-4 Thinking | 69.0 | 68.4 | 64.4 | 51.0 | 38.3 | 58.2 |
| GPT-5 | 65.4 | 63.2 | 64.1 | 34.5 | 33.7 | 52.2 |
| DeepSeek-R1 | 51.4 | 60.4 | 50.2 | 53.6 | 31.0 | 49.3 |
| Qwen2.5-14B-Instruct (Base) | 8.7 | 48.4 | 35.3 | 26.0 | 17.6 | 27.2 |
| Qwen2.5-32B-Instruct (Base) | 32.1 | 55.8 | 33.8 | 37.0 | 27.5 | 37.2 |
| AgentRL Qwen2.5-3B | 92.4 | 60.0 | 55.0 | 40.5 | 52.1 | 60.0 |
| AgentRL Qwen2.5-7B | 91.5 | 63.7 | 57.8 | 40.8 | 56.1 | 62.0 |
| AgentRL Qwen2.5-14B | 91.5 | 72.2 | 72.8 | 43.6 | 58.5 | 67.7 |
| **AgentRL Qwen2.5-32B** | **94.5** | **70.4** | **77.0** | **51.7** | **58.6** | **70.4** |
| AgentRL GLM-4-9B | 93.3 | 66.9 | 75.7 | 33.2 | 55.9 | 65.0 |

从 Table 3 可以得出三点关键结论：

1. **多轮环境在线 RL 增益显著**：`Qwen2.5-14B` 五任务平均分从 `27.2` 跃升至 `67.7`（`+40.5`），`32B` 从 `37.2` 跃升至 `70.4`（`+33.2`）；甚至训练后的 `3B` 小模型（平均 `60.0`）也超越了未经该环境 RL 训练的 `32B` 基座与表中通用闭源模型的 prompting 表现。
2. **随模型参数规模呈现清晰的扩展趋势（Scaling Trend）**：从 3B（`60.0`）、7B（`62.0`）、14B（`67.7`）到 32B（`70.4`），总体平均性能稳步上升，在状态空间复杂的 KG 与 OS 任务上大模型的容量优势尤为突出。
3. **跨基座模型架构有效**：`GLM-4-9B` 在接入该框架后同样取得 `65.0` 的平均分，验证了算法对不同模型家族的通用性。

![AgentRL 单任务训练与多任务训练对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table04-single-vs-multitask.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 4, CC BY 4.0.*

Table 4 进一步检验了“单一多任务通才模型（Generalist）”与“五个单任务专家模型（Specialists）”的表现差异（均基于 `Qwen2.5-14B`）：

| 训练模式 | ALFWorld | DB | KG | OS | WebShop | AVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 五个单任务专家逐列取最优 | 89.7 | **73.9** | 72.2 | 43.1 | **60.3** | **67.8** |
| **单个 AgentRL 多任务模型** | **91.5** | 72.2 | **72.8** | **43.6** | 58.5 | **67.7** |

观察 Table 4 的细项可以发现：单任务 RL 极易导致严重的领域过拟合与跨任务灾难性遗忘——例如仅在 ALFWorld 上训练的模型虽然本任务达到 `89.7`，但在 WebShop 上跌至 `15.9`；仅在 WebShop 上训练的模型在 ALFWorld 上甚至降为 `0.0`。相比之下，单个 AgentRL 多任务模型仅用一套参数便取得了 `67.7` 的平均分，与五个单任务专家拼出的理论上限（`67.8`）几乎持平，并在 ALFWorld、KG 与 OS 三项任务上反超单任务专家，体现了跨任务联合训练的正向迁移价值。

### 6.3 函数调用泛化评估（BFCL-v3, Table 5）

![AgentRL 在 BFCL-v3 上的泛化结果](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table05-bfcl-generalization.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 5, CC BY 4.0.*

为评估训练后的模型在分布外（OOD）工具调用基准上的泛化能力，论文在未参与训练的 BFCL-v3 上对比了 `Qwen2.5-32B`：

| 模型 | Single-Turn Non-Live | Single-Turn Live | Multi-Turn | Overall |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-32B-Instruct | 86.0 | 77.4 | 16.2 | 59.9 |
| AgentRL Qwen2.5-32B | 85.8 | 79.3 | 19.2 | **61.4** |
| 相对变化 | -0.2 | +1.9 | +3.0 | +1.5 |

结果表明，五任务 Agentic RL 并未损害基础的单轮函数调用精度，同时在 Single-Turn Live（`+1.9`）与 Multi-Turn（`+3.0`）子集上带来了温和的正向泛化。

## 7. 算法消融、跨策略案例与终止状态分析

### 7.1 Cross-Policy 与任务归一化消融实验（Table 6 & Fig. 7）

![Cross-Policy Sampling 与 Task Advantage Normalization 消融表](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table06-ablation.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 6, CC BY 4.0.*

Table 6 在 `Qwen2.5-14B` 上分别移除了 Cross-Policy Sampling 与 Task Advantage Normalization：

| 14B 消融配置 | AF | DB | KG | OS | WS | AVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **AgentRL 完整模型** | **93.1** | **64.0** | **67.7** | **45.1** | **55.0** | **65.0** |
| w/o Cross-Policy Sampling | 91.9 | 61.6 | 55.7 | 39.7 | 54.5 | 60.7 |
| w/o Task Advantage Norm | 91.1 | 62.6 | 54.7 | 38.0 | 50.6 | 59.4 |

移除跨策略采样使平均分下降 `4.3` 个百分点（其中多跳探索要求最高的 KG 暴跌 `12.0` 个百分点）；移除任务级优势归一化使平均分下降 `5.6` 个百分点，KG、OS 与 WebShop 同步回退。注意 Table 6 的 14B 消融实验与 Table 3 的主实验（平均 `67.7`）对应不同的训练步数或评估快照，因此解读时应将消融表的内部相对差值与主表最终峰值区分开来。

![AgentRL Cross-Policy 与任务归一化训练曲线消融](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig07-ablation.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 7, CC BY 4.0.*

Fig. 7 的训练动态进一步揭示了两项机制的作用阶段：在 Fig. 7a（KG）中，不含 Cross-Policy 的基线虽然前期爬升略快，但在约 300 步后即陷入平台期，而启用 Cross-Policy 的曲线在中后期持续突破上限；在 Fig. 7b（ALFWorld）中，缺少 Task Advantage Normalization 导致训练前期爬升迟缓且震荡剧烈；Fig. 7c 的五任务平均曲线则证明两者结合能够兼顾收敛速度与后期上限。

### 7.2 异构模型推理验证与 WebShop 训练曲线（Fig. 8）

![Cross-Policy Sampling 在 WebShop 推理和训练中的效果](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig08-cross-policy-webshop.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 8, CC BY 4.0.*

Fig. 8a 先在纯推理阶段测试了 `Qwen2.5-14B` 与 `Llama-3.1-8B` 组成的异构策略池在 WebShop 上的 `pass@k` 表现：当采样次数 $k$ 较小时，Cross-Policy 略低于单模型最优值；但随着 $k$ 增大，Cross-Policy 的 `pass@k` 曲线超越了两个单模型以及轨迹级混合（Mix），证实步骤级切换确实能够生成单一策略无法覆盖的成功路径。Fig. 8b 则展示了在 WebShop 训练中引入 stale engine 后，训练起点与收敛后的 `pass@k` 均获得一致提升。

### 7.3 知识图谱互补案例（Fig. 9）与终止状态统计（Table 7）

![KG 任务中 GLM、Llama 与 Cross-Policy 的案例](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-fig09-cross-policy-kg-case-study.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Fig. 9, CC BY 4.0.*

Fig. 9 给出了一个直观的 KG 互补案例（查询“同时包含藏传佛教与道教的宗教实践数量”）：单独使用 GLM 时，模型虽能规划关系路径，却陷入重复确认而未能按协议提交工具调用；单独使用 Llama 时，模型积极调用工具，却误用了 `count` 参数导致反复报错直至耗尽轮数上限。而在 Cross-Policy 轨迹中，前期由 GLM 完成关系分解，中途切换至 Llama 打破僵局触发关键工具调用，最后切回 GLM 汇总正确答案，直观体现了多策略在单条 episode 内协同破局的机制。

![基础模型与 AgentRL 的主要终止状态对比](/images/blog/agentrl-multiturn-multitask-agentic-reinforcement-learning/agentrl-table07-failure-modes.webp)

*Source: Zhang et al., arXiv:2510.04206v1, Table 7, CC BY 4.0.*

Table 7 统计了训练前后轨迹的终止状态分布（`Completed` 指模型主动提交答案，`Task Limit` 指耗尽最大交互轮数）：

| 环境 | Base Completed | Base Task Limit | AgentRL Completed | AgentRL Task Limit |
| --- | ---: | ---: | ---: | ---: |
| ALFWorld | 0.070 | 0.680 | 0.926 | 0.074 |
| DB | 0.957 | 0.043 | 0.993 | 0.007 |
| KG | 0.747 | 0.213 | 0.947 | 0.033 |
| OS | 0.548 | 0.444 | 0.847 | 0.118 |
| WebShop | 0.725 | 0.275 | 0.980 | 0.020 |

经过 AgentRL 训练（结合超轮数 `-0.2` 惩罚），各环境的 `Task Limit` 超时率大幅下降（如 ALFWorld 从 `68.0%` 降至 `7.4%`）。需要注意，`Completed` 仅代表模型按协议完成了提交动作，提交内容的正确性仍需结合 Table 3 的最终奖励综合评判。

## 8. 生态横向对比与开源仓库演进

### 8.1 AgentRL 与 SAO 的双层异步机制对比

将本文精读的 AgentRL 与姊妹篇 [SAO 论文精读](/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning-paper-reading/) 对照，可以清晰看到异步 Agentic RL 在不同层面的演进取舍：

| 对比维度 | AgentRL (`arXiv:2510.04206`) | SAO (`arXiv:2607.07508`) |
| --- | --- | --- |
| 核心定位 | 多轮、多任务训练系统与异构环境控制面扩展 | 长时域单轨迹异步优化的稳定性与价值估计 |
| 基础优势估计 | GRPO 组内相对优势 + Task Advantage Norm | Actor-Critic + Skip-Observation Token-level GAE |
| 每 Prompt 采样数 | $G = 8$ | $G = 1$（Single Rollout） |
| 异步队列边界 | 跨组异步，同一 prompt 需凑齐 8 条轨迹方可入队 | 单条轨迹完成即刻入队，彻底消除组内屏障 |
| 离策（Off-Policy）处理 | 主动引入 Stale Engine 促进探索 + 标准 PPO 软裁剪 | 基于 Rollout Log-Prob 的双边硬门控（DIS）清零越界梯度 |
| 额外显存开销 | 无需 Critic，但需驻留 Stale Rollout Pool 与 Reference | 需维护同规模 Critic 并每步更新 2 次（冻结 Attention） |
| 环境观测处理 | `loss_mask = 0` 屏蔽策略损失 | `loss_mask = 0` + GAE 递推显式跳过观测片段 |
| 系统吞吐实测 | 报告 WebShop 14B 约 2× 吞吐加速曲线 | 未报告系统吞吐对比，聚焦千步收敛准确率 |

简言之，AgentRL 优先保留了 GRPO 无 Critic 的架构简洁性，通过解耦 rollout/learner 资源组消除跨批次空泡，并把适度的策略陈旧性（stale policy）转化为探索动力；而 SAO 则针对单条轨迹耗时数十分钟的极端长尾场景，直接拆掉 GRPO 的组内同步墙，并用 DIS 硬门控与高频 Critic 解决随之而来的单样本高方差与严重离策问题。

### 8.2 与 VeRL、AReaL、ROLL、slime 及 AgentBench 的职责划分

在开源后训练生态中，AgentRL 的系统定位可概括为“连接底层分布式 RL 引擎与上层异构容器环境的中间控制层”：

- **对比 VeRL / OpenRLHF / slime**：这些框架提供了高效的 FSDP/Megatron 训练与 SGLang/vLLM 推理编排（AgentRL 的训练侧亦借鉴了 VeRL 的组件设计），但通常不内置管理数千并发有状态容器会话的中央 Controller 与多任务 Function-Call 路由协议。
- **对比 AReaL / Alibaba ROLL**：三者均致力于通过异步化消除长尾 rollout 空泡，而 AgentRL 的特色在于将异步训练与多任务优势归一化、跨策略采样及 AgentBench-FC 容器生态深度整合。
- **对比 AgentBench / OpenHands**：后者定义了评测基准与沙箱执行协议，AgentRL 则将其改造为支持高并发流式采样、超时惩罚与回放审计的在线强化学习数据源。

### 8.3 论文发布后的仓库演进与核心代码阅读地图

在审计固定提交 [`6a73409d`](https://github.com/THUDM/AgentRL/tree/6a73409d31ba695d383b978a8ad3ef400d90c054) 时，可以看到官方仓库已在 2025 年论文原型基础上进行了多项工程扩展：

1. **模块化 `trainer` 与算法扩展**：除复现论文的 `agentrl_trainer.py` 外，新增了通用 `async_trainer.py`，并在优势计算枚举中扩展了 GAE、GRPO_PASSK、REINFORCE++、REMAX 与 RLOO。
2. **Go Controller 与双传输协议**：Controller 采用 Go 语言重写以支撑更高并发（README 设计目标达 `10,000` 并发会话），除兼容原版 HTTP 接口外，新增了允许远端 TaskWorker 主动反向注册的 gRPC 传输层以及 Web Dashboard。
3. **独立评估包 `agentrl-eval` 与多后端支持**：新增实验性的评估 CLI，并在环境控制器中加入了 Docker、Kubernetes 以及 Redis/Consul/etcd 状态后端代码。

为便于读者深入源码，核心模块路径整理如下：

| 模块层次 | 固定 Commit 源码路径 | 核心实现逻辑 |
| --- | --- | --- |
| 论文训练主入口 | [`examples/training/agentrl_trainer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/agentrl_trainer.py) | Current/Stale 资源池划分、`cross_sampler`、`adv_norm` 与 NCCL 同步主循环 |
| 五任务示例配置 | [`examples/training/configs/qw14b_waodk.yaml`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/examples/training/configs/qw14b_waodk.yaml) | GPU 比例、`stale_step=25`、`n=8`、并发度与超参数定义 |
| 异步任务管理器 | [`trainer/.../components/task_manager.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/components/task_manager.py) | 从 Ray Queue 消费样本并拉起异步环境交互协程 |
| 成组完成队列 | [`trainer/.../components/buffer.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/components/buffer.py) | `strict_group` 组内屏障与动态批次提取 |
| 多轮交互循环 | [`trainer/.../agentic/loops.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/agentic/loops.py) | Action/Observation 编码、`loss_mask` 与 `rollout_log_prob` 构造 |
| 优势与策略损失 | [`algorithms/advantage.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/algorithms/advantage.py) / [`loss_funcs.py`](https://github.com/THUDM/AgentRL/blob/6a73409d31ba695d383b978a8ad3ef400d90c054/trainer/src/agentrl/trainer/algorithms/loss_funcs.py) | GRPO 组相对优势、重要性比率、PPO 裁剪与 KL 惩罚 |
| 环境控制面与节点 | `controller/` & `worker/` | Go 会话路由控制器、Python TaskWorker 与容器生命周期管理 |

## 9. 生产落地治理、局限性与结论

### 9.1 分阶段落地路线、生产监控平面与验收门槛

在实际工程中落地 AgentRL 架构，建议遵循由简入繁的四步演进路线，并配套建立覆盖四个平面的可观测性体系：

1. **分阶段验证路线**：
   - *阶段一（环境协议与离线基线）*：独立部署 Controller 与单类 TaskWorker，跑通 `start/observation/end` 闭环、超时回收与奖励校验，采集基础模型的完成率与错误分布；
   - *阶段二（同步小规模 GRPO）*：在单任务、3B 模型上验证 `group_id` 聚合、`loss_mask` 对齐与 `rollout_log_prob` 一致性；
   - *阶段三（异步解耦与吞吐对标）*：切分 rollout 与 learner 资源组，暂不启用 stale pool，对比同硬件下的吞吐加速比与队列滞后分布；
   - *阶段四（跨策略探索与多任务联合）*：接入多任务数据源并开启 `task_adv_norm` 与 `rollout_stale`，逐项验证探索增益与跨任务平衡。
2. **四大核心监控平面**：
   - *Rollout 平面*：追踪每秒完成轨迹数、动作/观测 Token 长度比、工具解析失败率及 current/stale 引擎请求占比；
   - *Queue 与版本平面*：监控 `Buffer.groups` 未完成组积压数、就绪队列等待延迟（P50/P95/P99）及采样至训练的策略版本差（policy lag）；
   - *Learner 平面*：按任务分桶记录有效动作 Token 占比、任务优势均值与标准差（防范方差趋零导致除数放大）、PPO 裁剪比例及参考模型 KL 散度；
   - *Environment 平面*：监控 Controller 路由延迟、容器启动失败率、孤儿会话（leaked sessions）回收情况及基础设施报错比例。

### 9.2 方法局限性与复现边界

在肯定 AgentRL 系统与算法贡献的同时，也应客观审视其边界条件：

- **组内同步屏障依然存在**：全异步架构消除了 rollout 与 training 阶段之间的全局等待，但在 `n = 8` 的 GRPO 设置下，同一 prompt 的 8 条轨迹仍需等待组内最慢样本完成才能进入可消费队列。
- **跨策略采样的理论条件与离策张力**：支持集严格扩展依赖各策略在相同工具协议下具备互补能力；训练中使用滞后 25 步的 stale engine 虽能增加动作多样性，但同时引入了离策偏移，论文未进一步量化不同 `stale_step` 下的有效样本留存率与策略滞后分布。
- **多任务均衡的覆盖范围**：实验在五个改写为统一 Function-Call 格式的 AgentBench-FC 环境中取得了接近单任务专家的平均性能（`67.7` vs. `67.8`，其中 DB 与 WebShop 仍略低于单任务专家约 `1.7`～`1.8` 个百分点）。面对任务数量更多、工具副作用更复杂的真实生产环境，仅靠一阶均值/方差标准化尚不足以完全消除底层表征的容量竞争。
- **复现与生产声明边界**：尽管官方开源了完整的训练与控制面框架，但完整复现论文中 14B/32B 五任务主表结果仍依赖大规模 H800 集群、全套容器镜像与环境数据；此外，摘要中提及的 AutoGLM 应用属于工业界落地声明，论文未附带对应的生产流量与线上消融数据。

### 9.3 总结

AgentRL 的核心价值在于证明了 **多轮、多任务 Agentic RL 的规模化扩展，必须依靠分布式调度、环境控制面、数据契约与强化学习算法的协同设计**。通过将 rollout 与 learner 物理分离并以中央 Controller 纳管容器化环境，系统在 14B WebShop 基准上实现了近两倍的吞吐提升；通过引入 Current/Stale 双池跨策略采样与任务级 Token 优势标准化，单个多任务模型在五类异构环境上的平均成功率大幅跃升并逼近五个单任务专家的组合上限。

对于构建智能体后训练基础设施的团队而言，AgentRL 不仅提供了一套可直接参考的开源代码基线，更明确了工程落地的核心准则：唯有先筑牢环境会话隔离、Action/Observation 掩码对齐与行为策略日志追溯的底层契约，上层的异步加速与多任务强化学习优化才能真正转化为稳健的模型能力增益。

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

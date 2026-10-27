---
title: "长期收益强化学习算法研究进展：从时间信用分配到 Agentic RL"
description: "系统梳理长时域、稀疏延迟奖励强化学习中的信用分配、时间抽象、离线到在线、世界模型与 LLM Agent RL 进展"
pubDate: "2026-06-10T18:53:09+08:00"
updatedDate: "2026-06-10T18:53:09+08:00"
tags:
  - "Deep Reading"
  - "Reinforcement Learning"
  - "Long-Horizon RL"
  - "Credit Assignment"
  - "Agentic RL"
  - "Survey"
draft: false
---

长期收益强化学习涵盖了一组相互交织的序列决策与优化难题。标准强化学习的目标函数本身即定义为长期累计回报的期望最大化：

$$
J(\pi)=\mathbb{E}_{\tau\sim \pi}
\left[
\sum_{t=0}^{T}\gamma^t r_t
\right],
$$

其中 $\tau=(s_0,a_0,r_0,\dots,s_T)$ 表示策略 $\pi$ 诱导出的轨迹，$\gamma\in[0,1)$ 为折扣因子。尽管理论形式已经囊括远期收益，但在实际训练中，当环境同时呈现奖励高度稀疏、反馈严重延迟、决策视野（horizon）漫长、状态部分可观测以及动作空间庞大等特征时，优化算法极难从有限样本中辨识出究竟是哪些早期决策促成了最终结果。

文献中讨论的“长期收益 RL”主要对应三类问题设定。第一类是**有限时域稀疏奖励任务（finite-horizon sparse-reward tasks）**，典型代表包括机器人精细操作、Atari 硬探索游戏、网页导航与 AppWorld 任务，这类场景具有明确的回合（episode）边界，通常仅在任务最终成功或失败时给出反馈。第二类是**长时域目标条件强化学习（long-horizon goal-conditioned RL）**，目标状态 $g$ 显式作为策略输入 $\pi(a\mid s,g)$，核心瓶颈在于目标空间体积庞大，无向随机探索极难触达成功轨迹。第三类是**持续性任务（continuing tasks）**，涵盖推荐系统、资源调度、长期用户留存优化与能源管理等无自然终止边界的场景；在此类任务中，折扣回报往往只是一种工程近似，平均奖励准则（average reward criterion）更能直接刻画系统的稳态表现：

$$
\rho(\pi)=
\lim_{T\to\infty}
\frac{1}{T}
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T-1} r_t
\right].
$$

上述三类设定在算法假设与评估方式上存在显著差异。若将三者混为一谈，容易产生两类常见偏差：一是将“价值传播缓慢”误判为所有长时域任务的唯一瓶颈；二是将特定机器人操作或网页智能体的实验结论盲目外推为通用强化学习规律。

## 1. 长期收益 RL 的核心症结与问题解耦

在马尔可夫决策过程（MDP）框架下，强化学习环境由状态空间 $\mathcal{S}$、动作空间 $\mathcal{A}$、转移分布 $P(s'\mid s,a)$、奖励函数 $r(s,a,s')$、初始状态分布 $\mu_0$ 以及折扣因子 $\gamma$ 共同定义。策略 $\pi(a\mid s)$ 致力于最大化状态价值函数

$$
V^{\pi}(s)
=
\mathbb{E}_{\pi}
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\mid s_t=s
\right],
$$

以及动作价值函数

$$
Q^{\pi}(s,a)
=
\mathbb{E}_{\pi}
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\mid s_t=s,a_t=a
\right].
$$

在短时域、稠密奖励任务中，局部动作与即时后续回报之间的统计关联清晰易学。一旦进入长时域稀疏奖励场景，训练困难便分化为四个相互耦合的维度：

- **时间信用分配（Temporal credit assignment）**：终局奖励应归因于哪些早期关键动作？当智能体在第 200 步成功开门时，真正决定成败的往往是第 20 步拾取钥匙、第 70 步记录路线信息以及第 130 步规避障碍，第 199 步的临门一脚仅是顺理成章的收尾。
- **价值传播（Value propagation）**：终局反馈如何高效回传至早期状态？一阶时序差分（TD）每次 Bellman 备份仅将奖励信号向前推进一个时间步，在数百步的延迟下，早期状态的价值估计需要消耗海量样本才能形成有效梯度。
- **稀疏探索（Exploration）**：首个成功轨迹从何而来？若初始策略在随机探索下触达目标状态的概率趋近于零，任何精密的信用分配机制都将因缺乏正样本而失效。
- **时间抽象（Temporal abstraction）**：如何将底层原子动作提炼为技能、子目标、选项（option）、动作块（action chunk）或高层计划？一旦有效决策频率从每步一次降低为每 $K$ 步一次，有效规划视野将大幅缩短。

Pignatelli 等人在 [Temporal Credit Assignment Survey](https://arxiv.org/abs/2312.01072) 中将信用分配严格定义为“从有限经验中推断过往动作对未来结果的影响”，并指出真实环境反馈往往伴随高噪声与长延迟，极少直接揭示因果责任（causal responsibility）。因此，解决长期收益问题需要在有限轨迹中重建跨时间步的因果影响链条。围绕这一目标，学术界逐渐演化出十条互补的技术路线：多步 TD 基础设施、显式信用分配、稀疏探索与课程学习、分层强化学习（HRL）、动作分块（action chunking）、世界模型与规划、离线到在线强化学习（offline-to-online RL）、LLM Agent RL、奖励塑形与过程奖励，以及平均奖励强化学习。

在深入具体算法前，有必要先厘清五个常被混用的概念维度：

1. **决策视野长（Long horizon）**：回合总步数或有效决策链条极长。即便每步都提供稠密奖励，只要策略需要在漫长序列中维持稳定控制（如长程导航、多工序装配、长期留存优化），依然属于长时域优化问题。
2. **奖励延迟大（Reward delay）**：关键决策与结果反馈之间相隔大量时间步。延迟奖励可以与稠密小奖励并存，但决定最终成败的往往是早期的准备工作。
3. **奖励稀疏度高（Reward sparsity）**：绝大多数时间步反馈为零，首要瓶颈在于如何首次采到非零奖励样本。
4. **部分可观测性强（Partial observability）**：当前观测 $o_t$ 不满足马尔可夫性质，智能体必须依赖历史记忆做决策，使得长程信用分配与记忆表征学习深度纠缠。
5. **持续性目标（Continuing objective）**：任务无自然终点，核心诉求在于优化稳态平均效用或长期约束满足率。

不同业务场景往往呈现出上述维度的不同组合：网页智能体通常兼具长视野、稀疏终局反馈、部分可观测性与复杂结构化动作空间；推荐系统较少面临单回合终局稀疏奖励，却受制于延迟业务指标与持续性目标；机器人拾取与装配则主要受困于稀疏探索与富含接触的动力学控制。在工程选型时，可以先通过以下诊断矩阵定位核心矛盾：

| 核心诊断问题 | 优先技术路线 |
| --- | --- |
| 随机或现有策略几乎采不到成功轨迹？ | 稀疏探索、自动课程学习、离线行为先验、专家示范 |
| 已有成功轨迹，但难以区分关键步骤与冗余动作？ | RUDDER、TVT、Synthetic Returns、COCOA、HCAPO、过程奖励 |
| 底层原子动作粒度过细，导致决策链条过长？ | 分层强化学习（HRL）、Options、子目标生成、动作分块（Action Chunking） |
| 在线试错成本高昂或存在物理/业务安全风险？ | 离线强化学习（Offline RL）、离线到在线微调、保守策略更新 |
| 环境动力学局部可预测且真实交互样本昂贵？ | 世界模型（World Model）、短程模型预测控制（MPC）、潜空间规划 |
| 任务无自然终止边界、关注稳态长期收益？ | 平均奖励强化学习、持续性任务优化、离策略评估（OPE） |
| 奖励依赖外部验证器或人类偏好模型？ | 势函数奖励塑形（PBRS）、过程奖励模型（PRM）、防奖励黑客约束 |

## 2. 多步 TD、资格迹与 GAE：价值传播的基础设施

最基础的改进路线保持任务结构不变，通过重塑回报或优势函数的估计器来加速远期信号回传。标准一阶 TD 学习采用 $y_t=r_t+\gamma V(s_{t+1})$ 作为自举（bootstrap）目标，具有低方差与高稳定性的优点，但在长延迟场景下传播极慢。若将自举跨度扩展为 $n$ 步，多步回报（$n$-step return）可一次性纳入未来 $n$ 步的真实奖励：

$$
G_t^{(n)}
=
\sum_{i=0}^{n-1}\gamma^i r_{t+i}
+
\gamma^n V(s_{t+n}).
$$

增大 $n$ 能够加快终局信号向前回传的速度，但也会推高回报估计的方差；在离策略（off-policy）训练中，若轨迹采自行为策略 $\mu$ 而评估目标为当前策略 $\pi$，多步累积还会放大分布偏移。TD($\lambda$) 与资格迹（eligibility traces）通过对不同步数的回报做指数加权平均来平衡这一矛盾：

$$
G_t^{\lambda}
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}G_t^{(n)}.
$$

在后向视图（backward view）中，资格迹通过向量 $e_t=\gamma\lambda e_{t-1}+\nabla_{\theta} V_{\theta}(s_t)$ 为近期访问过的状态或模型参数维持逐步衰减的“短期记忆”，并结合单步 TD 误差 $\delta_t=r_t+\gamma V_{\theta}(s_{t+1})-V_{\theta}(s_t)$ 将更新信号沿访问轨迹向后分配。

在策略梯度与 Actor-Critic 框架中，[GAE（Generalized Advantage Estimation）](https://arxiv.org/abs/1506.02438)将指数加权多步残差思想引入优势函数估计。给定价值函数的单步 TD 残差 $\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t)$，GAE 定义优势估计器为：

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l \delta_{t+l}^{V}.
$$

通过调节 $\lambda\in[0,1]$，GAE 在单步 TD 残差的低方差与蒙特卡洛回报的低偏差之间实现平滑过渡，并成为 [PPO](https://arxiv.org/abs/1707.06347) 等主流策略优化算法的标准组件。当训练进入分布式异步采样或重放缓冲区（replay buffer）场景时，离策略多步修正变得不可或缺：[Retrace($\lambda$)](https://arxiv.org/abs/1606.02647) 通过截断重要性采样比率保证离策略多步备份的收缩性质；[IMPALA](https://arxiv.org/abs/1802.01561) 则在 Actor-Learner 分布式架构中引入 V-trace，修正采样 Actor 滞后策略与 Learner 最新策略之间的分布偏差。2025 年提出的 [Deep Reinforcement Learning with Gradient Eligibility Traces](https://arxiv.org/abs/2507.09087) 进一步将梯度 TD 资格迹推广至深度网络，证明经典的多步迹机制在现代深度强化学习中依然具备挖掘潜力。

| 方法 | 核心机制 | 长期收益增益 | 能力边界 |
| --- | --- | --- | --- |
| 1-step TD | 单步自举更新 | 极低方差，训练稳定 | 远期奖励需经逐轮迭代缓慢回传 |
| $n$-step Return | 显式累加 $n$ 步奖励后自举 | 按跨度 $n$ 倍速回传终局信号 | 跨度增大时方差剧增，离策略修正困难 |
| TD($\lambda$) / 资格迹 | 多步回报的几何加权与后向迹分配 | 兼顾传播速度与方差控制 | 按时间距离均匀衰减，无法跳步归因 |
| GAE | 指数加权 TD 残差估计优势函数 | 显著改善策略梯度的偏差-方差折中 | 强依赖基线价值函数 $V_\theta$ 的表征精度 |
| Retrace / V-trace | 截断重要性采样修正多步残差 | 支撑大规模分布式异步多步训练 | 仅修正分布偏移，不涉及探索与任务解构 |

从优化机制剖析，上述方法本质上围绕三个核心旋钮展开权衡：
- **自举深度（Bootstrap depth）**：决定依赖价值函数预测与依赖真实采样轨迹的比例。跨度太浅会导致终局反馈无法及时抵达早期状态，跨度过深则会令累积环境噪声淹没梯度信号。
- **同策略与离策略权衡（On-policy vs. Off-policy）**：同策略直接利用当前策略轨迹做无偏多步估计，但样本复用率低；离策略能够聚合历史轨迹与海量异步 Actor 数据，但必须承受重要性采样截断带来的偏差。
- **价值网络的表征瓶颈**：在策略梯度公式 $\nabla_\theta J(\pi_\theta)\approx\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\hat A_t]$ 中，若价值网络 $V_\theta$ 无法捕捉早期关键状态与远期成功概率之间的非线性关联，早期时间步的优势估计 $\hat A_t$ 仍会退化为随机噪声。

总体而言，多步 TD、资格迹与 GAE 构成了长时域强化学习的底层算子，专注于提升已有奖励信号的沿途传播效率；然而由于它们严格按照时间距离做指数衰减，当关键动作与结果反馈之间夹杂数百步无关操作时，必须引入更具结构针对性的信用分配与时间抽象机制。

## 3. 显式时间信用分配：跨时间步的因果与关联归因

面对数百步延迟的终局奖励，显式时间信用分配（explicit temporal credit assignment）试图绕开按部就班的 Bellman 逐步回传，直接建立过去事件与远期结果之间的归因通路。从演进脉络看，这一方向经历了从“序列回报分解”到“记忆注意力传输”，再到“状态直接关联”、“反事实因果推断”以及“大语言模型事后归因”的深化过程。

### 3.1 轨迹回报重分配与记忆价值传输

[RUDDER（Return Decomposition for Delayed Rewards）](https://arxiv.org/abs/1806.07857) 开创了通过序列模型进行奖励重分配（reward redistribution）的典型范式。给定原始回合回报 $G_0=\sum_{t=0}^{T}\gamma^t r_t$，RUDDER 训练一个回归模型预测整条轨迹的最终回报，并利用序列特征变化将回合总回报分解到具体的历史时间步上，构造出满足回报等价性（return-equivalent）的重分配奖励 $r'_t$。在理想重分配下，所有远期奖励已被提前折算至触发该结果的早期关键动作处，使得后续状态的期望未来奖励趋近于零：

$$
\mathbb{E}\left[\sum_{k=1}^{\infty}\gamma^k r'_{t+k}\mid s_t,a_t\right]\approx 0.
$$

一旦满足该条件，$Q$ 值的学习便简化为对即时重分配奖励 $r'_t$ 的拟合，彻底消除了长链条自举偏差。不过，RUDDER 的成败高度取决于回报分解模型能否在有限样本中准确识别关键事件；在高维视觉输入或复杂多模态轨迹中，分解模型容易将偶然共现的统计相关性误判为真实贡献。

当长时域任务同时伴随部分可观测性时，早期事件往往通过“写入外部记忆并在远期读取”的方式影响最终收益。[Temporal Value Transport（TVT）](https://arxiv.org/abs/1810.06721) 针对此类需要跨期回忆（recall）的 POMDP 任务，利用记忆模块的注意力权重构建价值直通车：若智能体在时刻 $t$ 预测价值或执行动作时，通过注意力机制高度聚焦于过去时刻 $k$ 写入的记忆槽 $m_k$（注意力权重为 $w_{t,k}$），TVT 便将当前时刻的价值更新量按比例直接传输回历史事件 $k$：

$$
\Delta V(m_k) \propto w_{t,k}\,\Delta V(s_t).
$$

这种机制使“早期观察密码或钥匙位置、数十步后凭记忆开门”的任务得以绕过中间无关探索步，直接完成跨期价值绑定。其潜在局限在于注意力权重仅反映表征读取强度，读取某一历史记忆未必等同于该历史动作对最终奖励负有因果责任。

### 3.2 状态关联预测与反事实贡献分析

为了进一步摆脱对显式记忆读取结构的依赖，[Synthetic Returns](https://arxiv.org/abs/2102.12425) 提出了状态关联学习（state-associative learning）。传统 TD 依赖 $s_0 \leftarrow s_1 \leftarrow \cdots \leftarrow r_T$ 的链式回传，而 Synthetic Returns 直接训练一个辅助网络，预测任意状态 $s_k$ 对未来任意远期奖励的增量贡献（$s_k \rightarrow \text{future reward contribution}$），并将该预测值作为合成回报（synthetic return）注入 TD 目标。与 RUDDER 针对完整轨迹做全局事后分解不同，Synthetic Returns 侧重于学习状态特征与远期事件之间的点对点预测关联，使关键准备状态在训练早期即可获得密集梯度指导。

然而，无论是序列分解、记忆注意力还是状态关联，本质上仍在捕捉统计相关性。更严格的信用分配需要回答反事实问题：“假使当时在状态 $s_t$ 未执行动作 $a_t$，而是改选其他参考动作 $\tilde a_t$，最终结果分布 $Y$ 会发生多大改变？”建立在事后信用分配（Hindsight Credit Assignment, HCA）框架之上的 [COCOA](https://proceedings.neurips.cc/paper_files/paper/2023/file/d8bd445c2abe1343cce0e14b361b2fb3-Paper-Conference.pdf) 利用反事实贡献分析（counterfactual contribution analysis）将时间信用形式化为期望结果的反事实差分：

$$
C_t
\approx
\mathbb{E}[Y\mid s_t,a_t]
-
\mathbb{E}[Y\mid s_t,\tilde a_t].
$$

这种表征将信用分配从“时间邻近度”或“共现频率”提升到了“动作是否实质性改变了结果分布”的因果层面。代价在于反事实分支的结果无法在同一次采样中直接观测，必须借助逆动力学模型、后验动作分布估计器或结构化假设进行推断，工程实现复杂度显著高于前几类方法。

### 3.3 多轮 LLM Agent 中的细粒度事后信用分配

随着大语言模型智能体（LLM Agents）成为研究焦点，显式信用分配在文本与工具交互轨迹中展现出新的生命力。一次复杂的智能体回合往往跨越数十轮思维链推理、API 调用、网页操作与错误恢复，而环境通常仅在回合结束时给出一个二元成败信号。[HCAPO（Hindsight Credit Assignment Policy Optimization）](https://arxiv.org/abs/2603.08754) 指出，直接沿用 GRPO 等无价值网络（value-free）方法会将同一回合优势无差别地广播给所有时间步，导致中间冗余动作甚至错误尝试被错误强化；为此，HCAPO 引入事后推理模型（hindsight reasoning critic）对历史步骤的实际贡献做重打分，在 WebShop 与 ALFWorld 等多步基准上显著改善了训练稳定性。[Agentic Credit Assignment Survey](https://arxiv.org/abs/2604.09459) 进一步将 2024–2026 年间的智能体信用分配方法按作用粒度归纳为词元（token）、片段（segment）、步骤（step）、轮次（turn）与多智能体（multi-agent）五个层级。

| 方法代表 | 归因信号构造机制 | 作用粒度 | 核心归纳偏置 | 典型失效边界 |
| --- | --- | --- | --- | --- |
| RUDDER | 序列模型的总回报差分分解 | 单步 $(s_t, a_t)$ | 关键动作伴随预测回报的阶跃变化 | 高维轨迹下易将虚假共现当成因果贡献 |
| TVT | 外部记忆模块的跨期读取权重 | 记忆写入时间步 | 被远期高价值决策检索的记忆具有高贡献 | 注意力检索相关性与真实因果责任可能背离 |
| Synthetic Returns | 状态到远期奖励的直接回归预测 | 状态 $s_t$ | 特定瓶颈状态与远期回报存在稳定统计关联 | 策略剧烈演化时关联预测器易受分布偏移影响 |
| COCOA / HCA | 基于后验分布的反事实期望差分 | 单步动作 $a_t$ | 改变未来结果分布的动作才具备真实信用 | 反事实分布不可直接观测，逆模型估计方差高 |
| HCAPO | LLM 事后反演与步骤级重加权 | 交互轮次 / 工具调用 | 语义可解释的中间推理轨迹可由事后视角评判 | 依赖判别器或验证器质量，易受奖励黑客攻击 |

从工程架构审视，显式信用分配模块相当于在原始环境与策略优化器之间挂载了一层“奖励翻译器（reward translator）”：它接收粗粒度的终局结果与完整历史轨迹，输出细粒度的密集归因信号。当该翻译器准确捕捉任务因果结构时，长时域样本效率将成倍提升；反之，一旦翻译器存在系统性偏差，智能体会迅速利用翻译漏洞收敛到错误的局部捷径。

## 4. 稀疏奖励探索与课程学习：获取高价值初始轨迹

任何信用分配机制都以“经验池中已存在包含非零奖励的轨迹”为前提。在极度稀疏奖励的任务中，智能体面临的首要障碍是无法通过随机探索触达哪怕一次成功状态。围绕“如何从零生成高价值训练轨迹”，研究工作沿着事后目标重标注、状态归档回溯、内在好奇心驱动以及定向课程学习四个方向展开。

### 4.1 事后目标重标注与状态回溯探索

针对目标条件强化学习（GCRL），[Hindsight Experience Replay（HER）](https://arxiv.org/abs/1707.01495) 提出了一种极为优雅的数据复用机制：当智能体尝试达成目标 $g$ 失败并最终停留在某状态对应的新目标 $g'$ 时，虽然该轨迹在原目标 $g$ 下回报为零，却完整演示了如何到达 $g'$。HER 将原始转移样本 $(s_t,a_t,s_{t+1},g)$ 重新标注为 $(s_t,a_t,s_{t+1},g')$ 并赋予成功奖励，使海量失败轨迹瞬间转化为密集的有效监督信号。这一机制非常适合目标空间与状态空间存在自然映射的机械臂操作任务（如 reach、push、pick-and-place），但在缺乏显式目标状态表征的开放网页或纯代码任务中难以直接套用。

对于不具备连续目标重标注条件、且存在大量死胡同或不可逆陷阱的硬探索（hard exploration）任务，失败的根源往往在于智能体偶然到达某个有潜力的中间状态后，下一轮回合又退回起点从头随机游走，导致深层状态覆盖率呈指数衰减。[Go-Explore](https://arxiv.org/abs/1901.10995) 打破了“每回合必须从初始状态盲目采样”的常规，确立了先返回再探索（return-then-explore）的两阶段范式：系统维护一个已发现有趣状态的归档库（archive），每次采样时先从库中挑选一个高潜力边界状态，通过模拟器状态重置或目标条件控制策略直接返回该状态，再从该处开展新一轮探索，待拼出完整通关轨迹后再通过模仿学习进行鲁棒化（robustification）。在 Montezuma's Revenge 与 Pitfall 等经典硬探索基准上的突破表明，长时域探索的关键在于维持边界状态的可返回性与覆盖广度，单纯调大策略熵正则项（entropy bonus）收效甚微。

### 4.2 内在动机、状态覆盖与定向课程生成

当环境不支持状态重置时，内在动机（intrinsic motivation）方法通过构造自监督内部奖励来驱动智能体主动访问未知区域。[ICM（Intrinsic Curiosity Module）](https://arxiv.org/abs/1705.05363) 在结合逆动力学特征空间的基础上，以转移前向模型的预测误差作为好奇心奖励；[RND（Random Network Distillation）](https://arxiv.org/abs/1810.12894) 则进一步简化结构，训练一个预测网络去拟合随机固定目标网络的输出，直接将蒸馏误差作为状态新颖度（novelty）信号。沿着状态覆盖与无监督技能发现的思路，[Skew-Fit](https://arxiv.org/abs/1903.03698) 通过对已访问状态分布做偏置重加权（skewed sampling），引导目标生成器持续采样低密度边界状态以最大化目标熵；[DIAYN](https://arxiv.org/abs/1802.06070) 则在无外部奖励条件下最大化潜在技能变量与访问状态之间的互信息，预先习得一组行为可区分的底层技能供下游稀疏任务调用。

然而，在高维长时域空间中，无差别地追求状态新颖度或全空间覆盖容易陷入“看电视噪声（noisy-TV）”陷阱或在无关紧要的旁支区域消耗算力。[Automatic Curriculum Learning（ACL）](https://arxiv.org/abs/2003.04664) 将训练任务分布本身视为动态优化对象，依据智能体当前能力边界由浅入深地编排环境参数或子目标。2025 年提出的 [DISCOVER](https://arxiv.org/abs/2505.19850) 进一步针对超长时域稀疏奖励 GCRL 设计了定向探索机制：算法不再盲目铺开全空间覆盖，而是估计候选探索目标与最终目标任务之间的方向一致性，优先采样沿着任务求解流形延伸的前沿目标。

探索机制与信用分配在长时域训练闭环中扮演着相辅相成的角色——前者负责在荒漠中开辟出包含奖励的样本路径，后者负责从路径中提炼出关键决策梯度。工程实践中，可依据当前经验池的样本状态确定优化优先级：

| 经验池正样本丰度 | 关键步骤可辨识度 | 第一优先级技术动作 |
| --- | --- | --- |
| 几乎为零 | 低 | 引入专家示范、离线先验数据、状态重置回溯（Go-Explore）或定向课程学习 |
| 具备明确目标结构但正样本稀缺 | 中 | 部署事后目标重标注（HER）、偏置目标采样（Skew-Fit / DISCOVER） |
| 已采到少量成功轨迹 | 低 | 挂载奖励重分配（RUDDER）、事后信用归因（HCAPO）或过程奖励模型（PRM） |
| 成功样本充足 | 高 | 聚焦 Actor-Critic 稳定化、离策略多步修正、动作分块与世界模型规划 |

这一结构也揭示了长时域强化学习落地的工程铁律：当任务极度稀疏时，单纯微调 Critic 损失函数无异于巧妇难为无米之炊；优先通过示范采集、脚本策略、课程环境设计、可逆重置接口与失败轨迹重标注完成“数据破冰”，往往比更换优化算法更为紧迫。

## 5. 时间抽象：从分层强化学习（HRL）到动作分块（Action Chunking）

当一项任务需要连续执行上千个底层原子动作（primitive actions）才能完成时，在单步粒度上直接优化 $\pi(a_t\mid s_t)$ 会同时引爆探索空间与信用分配跨度。时间抽象（temporal abstraction）的核心洞见是将多步底层控制封装为在时间上延展的宏动作、闭环技能或连贯动作块。

### 5.1 Options 框架与经典分层架构演进

Sutton 等人提出的 [Options 框架](https://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf) 奠定了时间抽象的半马尔可夫决策过程（SMDP）理论基础。一个选项（option）由三元组定义为：

$$
o=(\mathcal{I}_o,\pi_o,\beta_o),
$$

其中 $\mathcal{I}_o\subseteq\mathcal{S}$ 为允许启动该选项的初始状态集，$\pi_o(a\mid s)$ 为选项内部的闭环执行策略，$\beta_o(s)\in[0,1]$ 为在各状态下的终止概率。若每个选项平均持续 $K$ 个底层时间步，高层策略的有效决策视野便可近似压缩为 $H_{\text{effective}}\approx H/K$。高层控制器只需在选项切换边界做宏观路由（如“走向房门”→“握住把手”→“推门进入”），局部扰动由选项内部闭环策略 $\pi_o$ 自行消化。

围绕“如何自动发现并稳定训练分层结构”，深度 HRL 发展出三条代表性脉络：
1. **端到端选项学习**：[Option-Critic](https://arxiv.org/abs/1609.05140) 推导了选项内部策略与终止函数的策略梯度定理，尝试无需人工先验地端到端学习选项；但在缺乏额外正则约束时，极易发生选项退化（option collapse）——要么单个选项包揽全过程，要么每个选项每步立即终止、退化回原子动作。
2. **多时间尺度表征解耦**：[FeUdal Networks（FuN）](https://arxiv.org/abs/1703.01161) 采用 Manager-Worker 双层架构，低频运行的 Manager 在潜表征空间输出方向性目标向量，高频运行的 Worker 通过最大化内在余弦相似度奖励来落实该方向，将不同时间尺度的信用分配显式解耦到两个网络模块中。
3. **离策略目标条件分层控制**：[HIRO](https://arxiv.org/abs/1805.08296) 将高低层接口直接定义为原始状态空间中的相对子目标（subgoal），以适配连续控制与离策略训练。针对“底层策略更新导致同一高层子目标对应的实际状态轨迹发生漂移”的非平稳耦合难题，HIRO 引入离策略子目标重标注（off-policy correction），将历史经验中的高层动作修正为能让当前底层策略复现相同状态转移的等效子目标。

### 5.2 离线分层学习与大语言模型引导的时序子目标

近年来，随着离线数据集与基础模型的发展，HRL 的重心正从“在线盲目试错发现技能”转向“从离线轨迹与先验知识中提炼可靠子目标”。在离线长时域稀疏奖励场景下，[Guider](https://www.ijcai.org/proceedings/2023/0469.pdf) 通过高层策略序列化生成子目标来引导底层离线策略拼接；[RD-HRL](https://openreview.net/forum?id=5E5sd3TWGD) 引入价值引导的子目标规划与可达性约束，防止高层规划器输出超出底层执行能力的分布外（OOD）目标；[OHIO](https://openreview.net/forum?id=dTPz4rEDok) 则专门解决离线数据由异构策略采集、高层隐变量不可直接观测时的分层策略反演问题。

与此同时，大语言模型为子目标分解提供了丰富的语义先验。2026 年提出的 [STO-RL（Subgoal Temporal Order RL）](https://arxiv.org/abs/2601.08107) 利用 LLM 为长时域稀疏任务生成具有严格偏序关系的子目标序列，通过状态-阶段映射函数追踪任务进度，并结合势函数奖励塑形（PBRS）合成时序一致的稠密奖励。这一设计直击 HRL 的核心痛点：子目标之间往往存在强时序依赖（例如必须先填写必填项、再点击提交按钮），若奖励塑形忽视先后顺序，智能体极易在局部子目标之间反复横跳以刷取中间分数。

| 架构范式 | 高低层通信接口 | 核心归纳偏置与机制 | 主要工程瓶颈 |
| --- | --- | --- | --- |
| Options / Option-Critic | 离散选项索引 $o$ 与终止概率 $\beta_o$ | SMDP 闭环子策略，端到端优化终止条件 | 易发生选项坍缩或语义碎片化 |
| FeUdal Networks | 潜空间方向向量 $g_t\in\mathbb{R}^d$ | 多时间尺度隐藏状态与方向性内在奖励 | 潜空间目标缺乏直观物理可解释性 |
| HIRO | 状态空间相对偏移子目标 | 离策略子目标重标注缓解高低层非平稳性 | 高维观测（如图像）下难以直接定义状态距离 |
| Guider / RD-HRL / OHIO | 离线分布内可达状态子目标 | 结合保守价值估计与离线轨迹拼接（stitching） | 受制于离线数据集的序列覆盖完备度 |
| STO-RL | 具偏序约束的语义子目标阶段 | LLM 常识先验拆解 + 势函数保序稠密塑形 | 依赖阶段映射器精度与先验子目标的物理可达性 |

综合来看，任何基于子目标的分层系统要想在工程中稳定奏效，其生成的中间子目标都必须同时满足四项硬性准则：**动力学可达性**（底层策略在合理步数内能够稳定抵达）、**完成度可验证性**（系统可通过状态断言、DOM 检查或验证器准确判定子目标是否达成）、**时序偏序一致性**（前序子目标为后续子目标创造前置条件而非制造冲突）以及**全局目标对齐性**（达成子目标序列能够单调逼近最终任务回报，避免落入局部死胡同）。

### 5.3 从单步备份到块级价值评估：Q-Chunking 的机制突破

传统分层强化学习虽然具备闭环技能表达力，但高低层联合训练的非平稳性与终止条件设计长期制约着工程落地。2025–2026 年间，源于机器人模仿学习（如 ACT、Diffusion Policy）的**动作分块**（Action Chunking）技术大举进入强化学习领域，成为长时域稀疏奖励与离线到在线（offline-to-online）微调中最具活力的轻量级时间抽象路线。

标准策略每步仅输出单个动作 $\pi(a_t\mid s_t)$，而动作分块策略直接以 $K$ 步动作序列作为基本输出单元：

$$
\pi(a_{t:t+K-1}\mid s_t).
$$

将动作空间从 $\mathcal{A}$ 提升至 $\mathcal{A}^K$ 带来了三重直接收益：一是将有效决策周期缩短为原来的 $1/K$；二是天然保证了多步探索动作的时序连贯性（temporal coherence），克服了单步高斯噪声在连续控制中相互抵消的抖动问题；三是无需额外学习选项终止条件或子目标语义。

[Reinforcement Learning with Action Chunking（Q-Chunking）](https://arxiv.org/abs/2507.07969) 敏锐地指出，要在强化学习中充分发挥动作分块的威力，必须将 Critic 的评估对象同步升级为整段动作块 $Q_K(s_t,a_{t:t+K-1})$，其 Bellman 目标定义为：

$$
y_t
=
\sum_{i=0}^{K-1}\gamma^i r_{t+i}
+
\gamma^K Q_{\text{target}}
\left(
s_{t+K},
\pi(s_{t+K})
\right).
$$

初学者容易将 Q-Chunking 与经典 $n$-step Q-learning 混为一谈，但二者在决策变量建模上存在本质差异。经典 $n$-step Q-learning 的更新式为：

$$
Q(s_t,a_t)
\leftarrow
\sum_{i=0}^{n-1}\gamma^i r_{t+i}
+
\gamma^n \max_{a'} Q(s_{t+n},a').
$$

尽管右侧目标跨越了 $n$ 步奖励，但左侧 Critic 的自变量依然仅包含单步动作 $a_t$；中间步骤 $a_{t+1:t+n-1}$ 隐式受控于重放缓冲区中的历史采样策略，Critic 无法显式评估或控制这 $n-1$ 步的动作组合。相比之下，Q-Chunking 的价值函数 $Q_K(s_t,a_t,a_{t+1},\dots,a_{t+K-1})$ 将整段动作序列显式纳入输入，使 Critic 能够精确区分“起点状态与首步动作相同、但后续连贯动作走势迥异”的轨迹质量，从而将离线示范数据中的时序连贯先验直接转化为在线探索优势。

### 5.4 连续块控制、序列 Critic 与自适应分块边界

尽管固定长度分块简洁高效，但也引入了新的结构性张力：动作空间维度随块长 $K$ 线性膨胀（$\dim(\mathcal{A}^K)=K\dim(\mathcal{A})$），且在块内 $K$ 步执行期间策略处于开环（open-loop）状态。在自由空间移动阶段，长块有助于快速跨越距离并加速价值回传；然而一旦进入富含接触的精细操作或环境突发扰动阶段，过长的开环块会导致智能体反应迟钝。围绕这一核心矛盾，近期研究衍生出四条改进分支：

- **面向稀疏操作任务的连续块 Actor-Critic**：[AC3（Actor-Critic for Continuous Action Chunks）](https://arxiv.org/abs/2508.11143) 针对长时域机器人稀疏奖励操作，将连续动作块融入 Actor-Critic 架构，通过仅在成功轨迹上更新 Actor、在 Critic 侧结合块内多步回报（intra-chunk $n$-step returns）以及引入自监督内在奖励，在 BiGym 与 RLBench 的 25 项长程操作任务上实现了稳定收敛。
- **仅在 Critic 侧引入序列感知**：[T-SAC（Chunking the Critic）](https://arxiv.org/abs/2503.03660) 另辟蹊径，保留 Actor 的单步闭环输出能力以维持高频反应性，同时采用 Transformer 架构构建序列条件 Critic，使其在评估多步目标时能够显式建模短轨迹片段内的动作交互。
- **因果 Transformer 驱动的自适应块长**：[ACSAC（Adaptive Chunk Size Actor-Critic）](https://arxiv.org/abs/2605.11009) 利用因果 Transformer Q 网络同时输出不同前缀长度 $k\in\{1,\dots,K_{\max}\}$ 的动作块价值，在每个分块边界根据当前环境状态动态裁定最优执行跨度 $K$，兼顾自由移动期的长程连贯性与接触交互期的闭环敏捷度。
- **消除跨尺度折扣偏差的优势分块选择**：[AQC（Adaptive Q-Chunking）](https://arxiv.org/abs/2605.05544) 揭示了自适应分块中的一个隐蔽数值陷阱——由于不同块长 $k$ 对应的 Bellman 目标具有不同的折扣缩放因子 $\gamma^k$，直接比较原始 $Q_k$ 值会产生系统性的尺度偏差（scale mismatch），导致策略倾向于短块。AQC 通过计算各视野跨度相对于其自身基线的优势值（horizon-normalized advantage）来裁定块长，在离线到在线微调中取得了更为稳健的自适应切换效果。
- **模型基离线强化学习中的块级转移**：[MAC（Model-Based RL with Action Chunks）](https://openreview.net/forum?id=WXGb9unEHo) 进一步将动作分块引入离线世界模型，直接学习块级动力学模型 $\hat s_{t+K}=f_\theta(s_t,a_{t:t+K-1})$。相比单步模型自回归滚动 $K$ 次，单次块级预测大幅削减了复合建模误差（compounding error），并配合富有表现力的行为分块策略（如流模型或扩散模型）做拒绝采样，有效遏制了规划器对分布外动作的恶意利用。

| 方法 | Actor 输出形式 | Critic 评估对象 | 块长 $K$ 策略 | 解决的核心痛点 |
| --- | --- | --- | --- | --- |
| Q-Chunking | 固定长度动作块 $a_{t:t+K-1}$ | 块级价值 $Q_K(s_t, a_{t:t+K-1})$ | 固定超参数 | 离线到在线转换中的连贯探索与多步价值备份 |
| AC3 | 连续动作块 $a_{t:t+K-1}$ | 结合块内多步回报的块级 Critic | 固定超参数 | 极稀疏奖励下长程机器人操作的训练崩溃 |
| T-SAC | 单步动作 $a_t$（保持闭环） | 短轨迹段 Transformer Critic | Critic 视野固定，Actor 单步 | 兼顾多步信用分配与单步高频闭环反馈 |
| ACSAC | 可变长度动作块 | 因果 Transformer 多前缀 Q 网络 | 按状态动态选择 | 平衡长块时序连贯性与接触阶段的反应敏捷度 |
| AQC | 可变长度动作块 | 消除折扣尺度偏差的多视野优势 | 基于相对优势自适应选择 | 修正不同视野 $Q$ 值直接比较导致的短视偏置 |
| MAC | 行为分块分布采样 | 块级世界模型 + 保守价值估计 | 固定块级跳跃 | 抑制离线长程模型滚动的复合误差与 OOD 利用 |

## 6. 世界模型与潜空间规划：短程动力学预测与长期价值自举

当真实环境交互代价高昂时，模型基强化学习（Model-Based RL）通过学习环境动力学模型并在想象轨迹（imagined trajectories）中优化策略或执行规划，提供了极具吸引力的样本效率优势。然而在长时域任务中，单步转移模型自回归滚动 $k$ 步生成的预测状态 $\hat s_{t+k}=f_{\theta}^{(k)}(s_t,a_{t:t+k-1})$ 不可避免地遭受复合误差累积——微小的单步预测偏差在数十步迭代后会被指数放大，导致远期想象轨迹彻底脱离真实物理流形。因此，现代长时域世界模型普遍摒弃了“依靠模型开环预测整个漫长未来”的幻想，转而确立了**短程潜空间规划 + 长期价值函数自举 + 时间抽象 + 分布外动作约束**的协同范式。

[DreamerV3](https://arxiv.org/abs/2301.04104) 展示了循环状态空间模型（RSSM）在跨领域长时域任务中的通用性：通过将高维观测压缩至紧凑的离散潜空间，并配合符号对数变换（symlog）、百分位回报归一化等鲁棒化设计，DreamerV3 用同一套超参数在百余项任务上超越专用算法，并在 Minecraft 中无需人工课程示范即从零学会收集钻石。其核心策略是在潜空间中仅展开有限步数的想象轨迹（通常 $H=15$），在想象视野末端挂载长期价值函数进行自举，从而将模型误差严格限制在短程窗口内。

[TD-MPC2](https://arxiv.org/abs/2310.16828) 则将隐式无解码器（decoder-free）世界模型与局部模型预测控制（MPC）深度融合。在每个决策时刻，系统在学习到的潜空间 $z_t$ 中求解带终端价值自举的有限视野轨迹优化问题：

$$
\max_{a_{t:t+H-1}}
\mathbb{E}
\left[
\sum_{i=0}^{H-1}\gamma^i r_{t+i}
+
\gamma^H V(z_{t+H})
\right].
$$

这种设计将长时域决策职责清晰拆分为三层架构：
1. **表征压缩层（Representation model）**：将非马尔可夫的历史观测与动作编码为紧凑潜状态 $z_t$，在过滤像素冗余噪声的同时满足局部可预测性。
2. **短程动力学预测层（Short-horizon predictor）**：仅负责预测未来数步（或如前文 MAC 所述预测单个动作块跨度 $\hat s_{t+K}=f_{\theta}(s_t,a_{t:t+K-1})$）内的潜状态转移、即时奖励与终止信号，确保局部推演高保真。
3. **规划与终端价值接口（Planner / Value interface）**：利用短程模型通过交叉熵方法（CEM）或采样评估候选动作序列的近期得失，而将视野 $H$ 之外的漫长远期收益交由终端价值函数 $V(z_{t+H})$ 兜底。

当这一架构应用于离线长时域数据时，规划器极易搜寻到模型预测虚高但偏离数据集支撑集的对抗性动作。因此，结合行为分块先验与拒绝采样（如 MAC）以封堵模型漏洞，已成为长时域模型基离线强化学习的标准配置。

## 7. 离线与离线到在线强化学习：长时域工程落地的主干道

在工业界真实场景中，让智能体从零开始在线随机试错往往是不可接受的：机械臂失控会损毁硬件，网页与企业智能体乱点会触发不可逆的业务副作用，推荐与金融系统的激进探索则会直接伤害用户体验与资金安全。因此，先利用历史日志或专家示范开展离线预训练（Offline RL），再在安全围栏内执行小规模在线微调（Offline-to-Online RL），构成了长时域强化学习工程落地的主干路线。

### 7.1 分布外动作抑制与离线轨迹拼接

[D4RL](https://arxiv.org/abs/2004.07219) 基准系统揭示了离线数据集的异构特性：历史数据可能源于手工控制器、人类遥操作、随机策略或次优混合策略。在标准 Bellman 最优算子中，最大化操作 $\max_a Q(s,a)$ 极易选出未包含在离线数据集 $\mathcal{D}$ 中的分布外（OOD）动作；由于缺乏真实样本纠偏，Critic 对这些 OOD 动作的过高估计会通过自举迅速传遍全图，引发价值崩溃。围绕这一核心瓶颈，主流离线算法形成了三种互补的约束机制：

- **保守价值惩罚**：[CQL（Conservative Q-Learning）](https://arxiv.org/abs/2006.04779) 在标准 Bellman 误差之上显式加入正则项，压低全动作空间上的高 $Q$ 值并推高数据集内动作的 $Q$ 值，优化目标为：

$$
\min_Q
\alpha
\left(
\log \sum_a \exp Q(s,a)
-
\mathbb{E}_{a\sim \mathcal{D}}Q(s,a)
\right)
+
\text{Bellman error}.
$$

这一约束保证学习到的价值函数成为真实策略价值的保守下界。
- **分布内隐式期望回归**：[IQL（Implicit Q-Learning）](https://arxiv.org/abs/2110.06169) 彻底规避了在训练中采样未见动作的操作，仅对数据集内动作 $(s,a)\sim\mathcal{D}$ 采用非对称期望分位数回归（expectile regression）拟合价值上限：

$$
L_V(\psi)
=
\mathbb{E}_{(s,a)\sim \mathcal{D}}
\left[
L_2^{\tau}
\left(
Q_{\theta}(s,a)-V_{\psi}(s)
\right)
\right],
$$

随后通过优势加权回归（AWR）提取策略：

$$
\max_{\pi}
\mathbb{E}_{(s,a)\sim \mathcal{D}}
\left[
\exp(\beta A(s,a))\log \pi(a\mid s)
\right].
$$

- **极简行为克隆正则**：[TD3+BC](https://arxiv.org/abs/2106.06860) 直接在确定性策略梯度目标上叠加均方行为克隆项并配合状态归一化，证明在许多连续控制任务中，稳健的分布内行为锚定比复杂的价值外推更为有效。

当任务视野拉长时，离线强化学习的考核重点从单步分布匹配升级为跨片段的**轨迹拼接**（trajectory stitching）与长程推理。近期发布的 [OGBench](https://arxiv.org/abs/2410.20092) 专门围绕离线目标条件强化学习设计了涵盖多段拼接、长程规划、高维视觉输入与随机动力学的基准测试，推动离线算法从“模仿单条完整好轨迹”迈向“将多条次优局部片段组合成全局最优长路径”。

### 7.2 从离线先验到在线微调：序列级数据覆盖观

离线预训练虽然消除了冷启动盲目探索的代价，但最终策略性能的天花板仍受制于历史数据集的覆盖边界。离线到在线（Offline-to-Online）强化学习旨在以离线策略为安全起点，通过少量在线交互突破离线数据的次优限制。[AWAC](https://arxiv.org/abs/2006.09359) 采用优势加权行为克隆格式的更新规则，使策略在从离线向在线过渡时避免因突发分布偏移而清洗掉预训练权重；前文介绍的 Q-Chunking、AQC 与 MAC 则进一步表明，在长时域稀疏奖励任务中，离线数据的真正财富在于**动作序列级（sequence-level）的时序连贯先验**。

在长时域视角下，评估一份离线数据集的质量必须同时审视三个层级的覆盖度：
1. **状态覆盖（State coverage）**：数据是否囊括了通往终局所必经的瓶颈中间状态（如“已拾取钥匙”状态）；若关键阶段完全缺失，任何离线算法都无从凭空捏造后续技能。
2. **动作覆盖（Action coverage）**：在关键决策点附近是否包含多样化的动作尝试，使 Critic 能够分辨优劣而非仅做盲目克隆。
3. **序列与拼接覆盖（Sequence & stitching coverage）**：数据集中是否包含可首尾相接的连贯动作片段。许多长时域离线训练的失败，皆源于单步状态看似均有覆盖，但多段子技能在交接边界处缺乏可达过渡，导致在线执行时在拼接点频繁坠崖。

因此，离线先验决定了初始探索的安全基线与可学行为边界，而在线微调则必须在严格控制价值漂移、物理安全与奖励黑客的前提下，重点填补跨阶段交接处的序列覆盖空白。

## 8. LLM Agent RL、奖励塑形与过程验证器设计

### 8.1 多轮软件交互中的 LLM Agent RL 演进

2024 年以来，大语言模型智能体（LLM Agents）将长期收益强化学习推向了一个全新的软件交互舞台。在这类任务中，智能体的动作空间涵盖自然语言思维链（thought tokens）、结构化工具调用（tool/API calls）、浏览器 DOM 交互、终端命令执行、代码库编辑与数据库查询；一个完整任务回合往往跨越数十轮环境反馈，直至最终通过单元测试或任务验证器判定成败。

将多轮交互智能体置于强化学习框架下，其轨迹可形式化为结构化事件序列：

$$
\tau =
(o_0,a_0,y_0,o_1,a_1,y_1,\dots,o_T,R),
$$

其中观测 $o_t$ 聚合了用户指令、多轮对话上下文与系统状态快照，动作 $a_t$ 兼具词元级自回归生成与工具调用两层结构，环境返回 $y_t$ 包含 API 响应、编译器报错、网页渲染结果或验证器反馈，$R$ 为终局奖励。与传统连续控制相比，LLM Agent RL 面临非平稳文本观测空间、高昂的外部环境执行延迟与副作用风险；但同时它也拥有独特的结构红利——语言空间天然支持语义子目标规划、代码与规则验证器（verifier）以及事后自然语言反思。构建一套可扩展的智能体强化学习系统，需要在工程上打通四大核心接口：

| 系统接口层 | 核心职责 | 工程挑战与痛点 |
| --- | --- | --- |
| 采样与轨迹接口（Rollout） | 并发调度多轮对话、工具执行、异常捕获与状态快照记录 | 异构动态工作流与外部 API 延迟极难对齐为标准张量批次 |
| 奖励与验证接口（Reward） | 集成单元测试、环境断言、规则检查器与偏好/过程奖励模型 | 开放任务缺乏精确验证器，代理奖励易遭长序列奖励黑客攻击 |
| 多粒度归因接口（Credit） | 将回合级反馈拆解至轮次（turn）、工具调用（step）或词元（token） | 数千词元与数十步工具调用中真正决定成败的决策点极度稀疏 |
| 安全与隔离接口（Safety） | 容器沙箱、文件系统快照回滚、细粒度权限控制与副作用拦截 | 在线探索可能污染真实数据库、触发外部写操作或耗尽 API 配额 |

围绕上述四个层面，近期代表性工作从算法显存效率、课程演化、系统解耦与过程归因多点突破：
- **有状态数字环境中的显存高效在线 RL**：Apple 提出的 [LOOP](https://arxiv.org/abs/2502.01600) 将交互式数字智能体建模为 POMDP，设计了无需独立价值网络（value-free）、支持多轮交互的留一法 PPO 变体，使强化学习微调显存开销接近普通 SFT；在 AppWorld 基准上，32B 参数的 LOOP 智能体超越了体量更大的 OpenAI o1 模型 9 个百分点，证实了在有状态软件环境中直接开展多轮 RL 优化的有效性。
- **网页智能体的自演化在线课程**：[WebRL](https://arxiv.org/abs/2411.02337) 针对网页导航任务中训练指令稀缺、结果奖励稀疏以及在线更新易引发策略遗忘三大难题，构建了根据智能体当前能力边界动态生成新任务的自演化在线课程（self-evolving curriculum），辅以结果监督奖励模型与 KL 锚定经验回放，使 Llama-3.1-8B 与 GLM-4-9B 在 WebArena-Lite 上的成功率实现大幅跃升。
- **智能体执行与强化学习训练管线解耦**：[AgentGym-RL](https://openreview.net/forum?id=ZgCCDwcGwn)（ICLR 2026 Oral）构建了覆盖多领域长时域决策的统一开源多轮 RL 训练框架，并通过 ScalingInter-RL 分阶段交互扩展策略稳定长程训练；[Agent Lightning](https://arxiv.org/abs/2508.03680) 则从系统架构层面彻底解耦“智能体工作流执行”与“底层 RL 训练”，通过统一的事件追踪协议捕获任意复杂智能体（含多智能体协作、动态路由与外部检索）的运行轨迹，再利用分层强化学习与信用分配模块将其自动转化为标准 RL 训练转移样本。
- **面向长轨迹的过程奖励与逐步进度归因**：为破解终局二元反馈过粗的困局，[AgentPRM](https://arxiv.org/abs/2502.10325) 利用蒙特卡洛滚动（Monte Carlo rollouts）估计中间状态达成终局目标的期望概率，训练轻量级智能体过程奖励模型（Process Reward Model）；[SPA（Stepwise Progress Attribution）](https://arxiv.org/abs/2505.20732) 则将长时域任务视为逐步累积任务进度（progress）的过程，通过将终局奖励按步骤进度增量进行重分配，为每轮工具交互提供稠密且对齐全局目标的推进信号。

### 8.2 势函数奖励塑形、过程奖励模型（PRM）与防黑客验证器

在长时域任务中，将稀疏终局反馈改写为稠密中间信号是最直观的工程加速手段，但粗暴的人工奖励塑形（reward shaping）极易扭曲最优策略——例如为足球机器人设置“靠近球即给正奖励”，可能导致机器人围着球原地打转而拒绝射门。

Ng、Harada 与 Russell 在经典论文 [Policy Invariance Under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) 中证明，若附加塑形奖励 $F(s,a,s')$ 严格取为状态势函数 $\Phi:\mathcal{S}\to\mathbb{R}$ 的折扣差分形式：

$$
F(s,a,s')=\gamma \Phi(s')-\Phi(s),
$$

则沿任意长度为 $T$ 的轨迹累加该塑形项会产生裂项相消（telescoping sum）：

$$
\sum_{t=0}^{T-1}\gamma^t
\left(
\gamma \Phi(s_{t+1})-\Phi(s_t)
\right)
=
-\Phi(s_0)+\gamma^T\Phi(s_T).
$$

在固定初始状态与适当的终止势约束下，轨迹总塑形回报仅相差一个与中间动作路径无关的边界项，从而严格保证了**最优策略不变性（policy invariance）**。这一理论揭示了安全奖励塑形的本质：势函数 $\Phi(s)$ 可以度量“向目标靠近的潜在进度”，但绝不能赋予某个中间状态可通过循环刷分无限累加的净收益。

在现代长时域与智能体强化学习中，稠密监督信号主要演化出三类实现形态：
1. **基于子目标偏序的势函数塑形**：如 STO-RL 将 LLM 规划出的有序子目标阶段映射为势函数 $\Phi(s)$，在享受语义先验引导的同时坚守策略不变性底线。
2. **基于人类偏好与轨迹对比的学习型奖励模型**：源自 [Christiano et al. (2017)](https://arxiv.org/abs/1706.03741) 的轨迹片段偏好学习（RLHF），通过成对比较训练奖励模型，适合难以用代码规则定义成败的主观质量评估。
3. **面向推理与交互步骤的过程奖励模型（PRM）**：如 AgentPRM 与 SPA，针对每个工具调用或环境状态转移评估中间进度。

需要注意的是，学习型过程奖励模型与启发式步骤打分通常不具备势函数差分的结构形式，因而在长时域训练中极易诱发**奖励黑客（reward hacking）**。视野越长，智能体可用于钻空子的动作组合空间就越大：网页智能体可能学会在输入框内反复拼写触发验证器高分的关键词而不提交表单，代码智能体可能通过删除测试断言来伪造单测全绿。为此，工业界普遍采用多层级复合奖励架构来平衡信号密度与抗攻击鲁棒性：

| 验证与奖励层级 | 信号粒度 | 核心优势 | 漏洞风险与防范重点 |
| --- | --- | --- | --- |
| 单元测试 / 精确状态断言 | 回合终局或关键检查点 | 客观确定、零幻觉、可全自动并发执行 | 测试覆盖不全时易被硬编码绕过，需保留隐藏测试集 |
| 环境成败标志（Success Flag） | 回合终局 | 与真实业务目标完全一致，无代理偏差 | 信号极度稀疏，无法单独提供早期梯度方向 |
| 规则式过程检查器 | 单步 / 工具调用 | 计算开销极低、语义透明可解释 | 静态规则易被高频重复调用刷分，需配额上限或势函数化 |
| 学习型过程奖励模型（PRM） | 步骤 / 轨迹片段 | 能够密集量化向终局目标的推进概率 | 在分布外状态易产生虚高外推，需不确定性惩罚与终局锚定 |
| LLM-as-a-Judge | 轮次 / 终局汇总 | 适配开放域无标准答案任务，泛化能力强 | 存在位置偏置、冗长偏置与提示词注入风险，需交叉校验 |

稳健的工程实践通常将硬性规则作为安全护栏（拦截非法格式与危险调用），以精确验证器作为终局胜负锚点，将过程奖励严格限制为辅助优势估计或势函数差分项，并通过定期离线抽样审查防范代理指标异化。

## 9. 平均奖励准则与持续性任务：无终止边界的长期收益

前述讨论大多建立在具有明确终止状态的回合制（episodic）任务之上。然而，推荐系统长期用户留存优化、数据中心动态能耗与算力调度、库存与供应链持续补货、常驻企业后台自动化等核心工业场景，均属于**无自然终点的持续性任务（continuing tasks）**。在这类系统中，强行设定人工截断窗口或套用较小的折扣因子 $\gamma$，往往会掩盖策略对系统稳态分布的深远影响。

平均奖励强化学习（Average-Reward RL）直接以单位时间步的稳态期望收益作为优化目标：

$$
\rho(\pi)
=
\lim_{T\to\infty}
\frac{1}{T}
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T-1}r_t
\right].
$$

在遍历性（ergodic）马尔可夫链假设下，$\rho(\pi)$ 与初始状态无关，直接刻画策略诱导出的平稳分布下的全局表现。为评估特定状态的瞬时优劣，算法定义差分价值函数（differential value function）来度量从状态 $s$ 出发相对于稳态平均收益的累计偏差：

$$
h^{\pi}(s)
=
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{\infty}
\left(
r_t-\rho(\pi)
\right)
\mid s_0=s
\right].
$$

设想两个竞争策略：策略 A 在前 100 步通过激进压榨攫取高额短期回报，却将系统推入难以恢复的低效稳态；策略 B 前期回报温和，但能将系统长期维持在高水平稳态。在短视野评估或较小折扣因子 $\gamma$ 下，折扣回报准则往往受初期高回报主导而错误优选策略 A；唯有平均奖励准则（或 $\gamma\to 1$ 的极限分析）能够剥离短期瞬态幻象，准确识别策略 B 的长期稳态优势。这种张力在真实业务中比比皆是：推荐系统依赖标题党诱导点击可在短期内拉升 CTR，却会加速用户信任耗竭与长期流失；运维调度过度超卖资源可压低当期成本，却会推高未来级联故障风险。

在算法层面，[Zhang and Ross (2021)](https://arxiv.org/abs/2106.07329) 指出折扣设定下的策略改进下界无法直接平移至平均奖励设定，并据此推导出面向持续性任务的平均奖励置信域策略优化（ATRPO/APO）；[Jin et al. (2024)](https://proceedings.mlr.press/v238/jin24b.html) 建立了更具可操作性的平均奖励 Q-learning 理论框架；[RVI-SAC](https://arxiv.org/abs/2408.01972) 则将相对价值迭代（Relative Value Iteration）与最大熵强化学习相结合，推动平均奖励离策略深度强化学习走向实用。在工程落地中，优化持续性长期收益不仅需要替换 Bellman 更新算子，还必须配套建设跨周期的离策略评估（OPE）、马尔可夫链混合时间（mixing time）监控、长期护栏指标（guardrail metrics）以及渐进式安全发布机制，使长期收益优化上升为完整的决策系统治理工程。

## 10. 统一技术图谱、前沿趋势与工程选型指南

### 10.1 十条技术路线对照与模块化架构组合

将上述十条技术路线置于同一坐标系下，可以清晰看到各方向的职责边界与适用象限：

| 技术路线 | 代表工作 | 解决的核心瓶颈 | 主要局限与代价 | 典型适用场景 |
| --- | --- | --- | --- | --- |
| 多步 TD / GAE | TD($\lambda$), GAE, Retrace, V-trace, IMPALA | 加速价值回传、平衡偏差与方差 | 按时间距离盲目衰减，长视野下仍嫌缓慢 | Actor-Critic 通用训练底座 |
| 显式信用分配 | RUDDER, TVT, Synthetic Returns, COCOA, HCAPO | 跨时间步直接归因关键历史事件 | 归因模型易混入虚假相关，反事实估计复杂 | 严重延迟奖励、记忆型任务、多轮智能体 |
| 稀疏探索与课程 | HER, Go-Explore, RND, Skew-Fit, DIAYN, DISCOVER | 从零获取包含正反馈的初始轨迹 | 仅解决样本生成，不直接负责精细归因 | 硬探索游戏、目标条件机械臂操作 |
| 分层强化学习（HRL） | Options, Option-Critic, FuN, HIRO, Guider, STO-RL | 通过子目标或闭环技能缩短有效视野 | 高低层联合训练非平稳，强依赖子目标质量 | 具明显阶段结构的复杂控制与规划 |
| 动作分块 | Q-Chunking, AC3, T-SAC, ACSAC, AQC, MAC | 轻量级时间抽象、连贯探索与块级备份 | 块内偏开环控制，长块降低突发反馈敏捷度 | 机器人连续操作、离线到在线微调 |
| 世界模型与规划 | DreamerV3, TD-MPC2, MAC | 利用潜空间短程推演大幅提升样本效率 | 长程滚动存在复合误差，易遭规划器钻漏洞 | 真实采样昂贵的连续控制与视觉控制 |
| 离线与离线到在线 | D4RL, CQL, IQL, TD3+BC, AWAC, OGBench | 利用历史先验规避冷启动探索风险 | 性能上限受制于数据集状态与序列覆盖度 | 工业机器人、推荐系统、企业级智能体 |
| LLM Agent RL | LOOP, WebRL, AgentGym-RL, Agent Lightning | 多轮工具调用、软件交互与执行-训练解耦 | 环境交互延迟高、验证器构建难、安全要求严 | 网页导航、代码修复、API 编排智能体 |
| 奖励塑形与 PRM | PBRS, RLHF, AgentPRM, SPA | 将终局稀疏监督转化为中间过程推进信号 | 非势函数塑形易改变最优策略并诱发奖励黑客 | 具备验证器或过程评估器的多步任务 |
| 平均奖励 RL | ATRPO/APO, Feasible Q-Learning, RVI-SAC | 优化无终止边界系统的稳态长期平均效用 | 深度函数逼近下差分价值估计方差较大 | 长期留存优化、持续资源调度与控制 |

长期收益强化学习的系统构建通常围绕六个正交工程维度进行模块化组合：
- **数据供给机制**：纯在线探索 / 离线预训练 / 离线到在线微调 / 专家示范引导；
- **时间与动作粒度**：底层原子动作 / 固定或自适应动作块 / 闭环选项与技能 / 高层子目标 / 工具与 API 调用；
- **奖励与归因粒度**：回合终局奖励 / 势函数塑形 / 步骤级过程奖励（PRM） / 事后反事实与注意力归因；
- **环境动力学利用**：无模型（Model-Free）直接优化 / 块级转移模型 / 短程潜空间 MPC 规划 + 终端价值自举；
- **优化目标准则**：有限视野成败率 / 折扣累计回报 / 持续性任务稳态平均奖励；
- **安全与验证护栏**：保守价值下界 / 容器沙箱与状态回滚 / 精确验证器锚定 / 离策略评估与线上护栏指标。

例如，一套生产级网页购物智能体通常会组合“离线示范预训练 + 礼貌在线微调”、“工具调用级动作粒度”、“终局结果验证器 + 过程奖励模型（AgentPRM）”、“无模型分组或留一法策略优化（如 LOOP/HCAPO）”、“有限视野任务成功率目标”以及“浏览器沙箱与写操作权限拦截”；而一套工业机器人精密装配系统则更倾向于采用“遥操作示范 + 离线到在线微调”、“自适应动作分块（AQC/ACSAC）+ 底层阻抗控制”、“稀疏成功标志 + 势函数接触引导”、“短程潜空间世界模型（TD-MPC2/MAC）”与“硬件力矩安全限幅”。

### 10.2 前沿研究趋势与五大开放性难题

纵观近两年的突破性进展，长期收益强化学习正在经历五项深刻的方法论演进：

1. **从原子动作优化全面转向时间扩展动作（Temporally Extended Actions）**：无论是经典 Options、分层子目标还是近期大放异彩的 Action Chunking，核心共识都是不再让高层策略在每个底层时间步重复决策。这四类时间抽象形态在闭环反馈能力与工程复杂度上各有取舍：

| 抽象形态 | 数学形式 | 块内闭环反馈能力 | 核心工程挑战 |
| --- | --- | --- | --- |
| Option（选项） | $(\mathcal{I}_o,\pi_o,\beta_o)$ | 强（内部闭环策略） | 自动发现语义选项与稳定训练终止条件极难 |
| Skill（无监督技能） | $\pi(a\mid s,z)$ | 强（潜变量条件闭环） | 无奖励预训练的技能未必对齐下游任务流形 |
| Subgoal（子目标） | $\pi_{\text{high}}(g\mid s),\pi_{\text{low}}(a\mid s,g)$ | 中（底层朝子目标闭环） | 子目标空间定义、动力学可达性与高低层非平稳性 |
| Action Chunk（动作块） | $\pi(a_{t:t+K-1}\mid s_t)$ | 弱至中（块内开环，边界闭环） | 固定块长 $K$ 在高频接触阶段反应滞后，需自适应边界 |

Action Chunking 之所以在 2025–2026 年迅速成为主流，正是因为它省去了选项终止函数与子目标可达性校验的沉重包袱，天然契合扩散策略、流匹配策略与离线示范轨迹中的连贯动作块结构。

2. **从回合级粗粒度反馈转向步骤/块/轮次级精细信用分配**：RUDDER、Synthetic Returns、HCAPO、AgentPRM 与 SPA 共同印证了“仅靠终局单一标量奖励更新整条长轨迹”对样本的巨大浪费。传统控制任务正将信用分配下沉至状态-动作对或动作块，而多轮智能体则将归因粒度细化至每次工具调用、每个交互轮次乃至关键推理片段。
3. **离线行为先验成为长时域训练的默认起点**：面对动辄数百步的稀疏任务，从零在线盲目探索已基本退出工业实践。由离线预训练提取行为分块先验、借助保守价值学习（CQL/IQL）筑牢分布内底线、再结合块级探索开展小规模在线改进，已成为高性价比的标准流水线。
4. **世界模型放弃超长开环滚动，收敛于“短程模型规划 + 长期价值自举”**：DreamerV3、TD-MPC2 与 MAC 的成功表明，世界模型的优势在于表征压缩与局部多步推演；通过限制规划视野或改用块级单步跳跃，并将远期收益交由价值函数自举，能够有效规避复合建模误差。
5. **大语言模型以双重身份深度融入长时域 RL**：一方面，**LLM 作为被训练的交互智能体（LLM as Agent）**，通过 LOOP、WebRL、AgentGym-RL 与 Agent Lightning 在有状态软件环境中直接优化多轮工具决策；另一方面，**LLM 作为结构先验与奖励协作者（LLM as Prior / Verifier）**，在 STO-RL、HCAPO 等工作中承担时序子目标拆解、事后反事实反思与过程奖励打分职责。明确区分这两种角色，有助于在工程中合理规划算力开销与验证边界。

与此同时，该领域仍面临五个亟待突破的开放性难题：
- **第一，长程时间信用分配缺乏统一的诊断与评测基准。** 现有工作各自在延迟奖励游戏、记忆回忆迷宫、机械臂长程操作、离线轨迹拼接或网页工具调用上验证算法，这些场景虽统称为“长时域”，但核心瓶颈迥异。正如 [Temporal Credit Assignment Survey](https://arxiv.org/abs/2312.01072) 所呼吁，学界亟需能够剥离探索干扰、定量测定“真实因果贡献识别精度”的标准化评测协议。
- **第二，时间抽象的动态边界划分尚无通用最优解。** 固定动作块长在自由移动与精细接触之间的顾此失彼，推动了 ACSAC 与 AQC 等自适应分块研究；然而如何在高维部分可观测环境中，以最小算力代价实时感知动力学突变并无缝切换闭环控制频率，仍需更优雅的理论与架构支撑。
- **第三，离线先验的“赋能”与“禁锢”难以两全。** 离线数据集赋予了长时域智能体宝贵的初始序列先验，却也将策略锁死在历史采集策略的动作流形附近。当在线环境出现分布偏移或需要拼接跨度极大的新技能组合时，如何既放开探索脚步又防止预训练表征灾难性遗忘，依然考验着离线到在线算法的平衡艺术。
- **第四，长视野下的过程奖励黑客（Reward Hacking）极难根除。** 除严格满足裂项相消条件的势函数塑形（PBRS）外，绝大多数学习型过程奖励模型（PRM）与 LLM 裁判都存在代理偏差。决策链条越长，策略搜寻并放大奖励模型盲区的自由度就越高，亟需发展具备形式化保序保证与不确定性感知的过程监督机制。
- **第五，LLM Agent RL 的训练成本控制与安全隔离机制尚处探索阶段。** 真实软件环境中的多轮滚动采样受限于网页加载、代码编译与外部 API 响应，吞吐量远低于传统张量物理模拟器；同时，带副作用的工具调用要求训练基础设施全面配备轻量级容器快照、事务级状态回滚与细粒度权限护栏。

### 10.3 场景选型矩阵与递进式文献阅读路径

针对不同业务场景的核心矛盾，可参照以下选型矩阵确定技术突破口：

| 业务与任务场景特征 | 首选技术组合 | 应当规避的误区 |
| --- | --- | --- |
| **机器人长时域稀疏奖励操作**（具备少量遥操作示范） | 离线行为分块预训练 + Q-Chunking / AQC / AC3 在线微调 + 短程世界模型（MAC） | 从零开展单步高斯噪声在线探索，或使用过长固定开环块处理精细接触 |
| **目标条件稀疏奖励任务**（状态与目标空间结构清晰） | 事后目标重标注（HER）+ 偏置/定向目标课程（Skew-Fit / DISCOVER） | 在未采到任何成功样本前耗费算力调整复杂 Critic 结构 |
| **强部分可观测与长延迟回忆任务**（早期关键事件决定终局） | 记忆增强架构 + 显式信用分配（RUDDER / TVT / Synthetic Returns / COCOA） | 仅依赖单步马尔可夫状态与浅层 TD($\lambda$) 期望信号自然回传 |
| **网页、代码与企业软件多轮智能体**（工具调用 + 终局验证器） | 执行-训练解耦框架（Agent Lightning / AgentGym-RL）+ 事后/过程归因（HCAPO / AgentPRM / SPA）+ 沙箱回滚 | 仅保存最终回复文本与二元成败分、忽略中间工具返回与轮次归因 |
| **环境交互极度昂贵或具高安全风险**（工业控制、医疗、金融） | 保守离线强化学习（CQL / IQL）+ 离线分层/块级规划（RD-HRL / MAC）+ 护栏内微调 | 未经离线分布内锚定与安全验证即直接部署在线探索策略 |
| **无自然终点的持续性业务系统**（推荐留存、长周期调度） | 平均奖励准则（ATRPO / RVI-SAC）+ 延迟指标因果归因 + 离策略评估（OPE）与线上护栏 | 将数天延迟的业务指标硬塞入短视野回合奖励，或仅监控短期瞬态点击率 |

对于希望系统深入该领域的研究者与工程师，建议按以下六个递进层次开展文献阅读：
1. **第一层（价值传播与优势估计基石）**：精读 [GAE](https://arxiv.org/abs/1506.02438)、[Retrace($\lambda$)](https://arxiv.org/abs/1606.02647)、[IMPALA](https://arxiv.org/abs/1802.01561)、[PPO](https://arxiv.org/abs/1707.06347) 以及综述 [A Survey of Temporal Credit Assignment in Deep RL](https://arxiv.org/abs/2312.01072)，吃透多步自举、偏差-方差权衡与离策略重要性采样修正。
2. **第二层（显式时间信用分配）**：依次研读 [RUDDER](https://arxiv.org/abs/1806.07857)、[TVT](https://arxiv.org/abs/1810.06721)、[Synthetic Returns](https://arxiv.org/abs/2102.12425) 与 [COCOA](https://proceedings.neurips.cc/paper_files/paper/2023/file/d8bd445c2abe1343cce0e14b361b2fb3-Paper-Conference.pdf)，理解如何通过序列分解、记忆注意力与反事实推断跨越数百步无关轨迹精准归因。
3. **第三层（稀疏探索与目标课程）**：阅读 [HER](https://arxiv.org/abs/1707.01495)、[Go-Explore](https://arxiv.org/abs/1901.10995)、[RND](https://arxiv.org/abs/1810.12894)、[Skew-Fit](https://arxiv.org/abs/1903.03698)、[DIAYN](https://arxiv.org/abs/1802.06070)、[ACL Survey](https://arxiv.org/abs/2003.04664) 与 [DISCOVER](https://arxiv.org/abs/2505.19850)，掌握如何从零破冰获取高价值初始轨迹。
4. **第四层（时间抽象与分层强化学习）**：从经典的 [Options 框架](https://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)、[Option-Critic](https://arxiv.org/abs/1609.05140)、[FeUdal Networks](https://arxiv.org/abs/1703.01161)、[HIRO](https://arxiv.org/abs/1805.08296) 读到近期的离线与 LLM 引导分层工作 [Guider](https://www.ijcai.org/proceedings/2023/0469.pdf)、[RD-HRL](https://openreview.net/forum?id=5E5sd3TWGD) 与 [STO-RL](https://arxiv.org/abs/2601.08107)，配合 [PBRS](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) 理解子目标分解与保序奖励塑形。
5. **第五层（动作分块、世界模型与离线到在线合流）**：重点研读 [D4RL](https://arxiv.org/abs/2004.07219)、[CQL](https://arxiv.org/abs/2006.04779)、[IQL](https://arxiv.org/abs/2110.06169)、[OGBench](https://arxiv.org/abs/2410.20092)、[DreamerV3](https://arxiv.org/abs/2301.04104)、[TD-MPC2](https://arxiv.org/abs/2310.16828) 以及 2025–2026 年的动作分块代表作 [Q-Chunking](https://arxiv.org/abs/2507.07969)、[AC3](https://arxiv.org/abs/2508.11143)、[T-SAC](https://arxiv.org/abs/2503.03660)、[ACSAC](https://arxiv.org/abs/2605.11009)、[AQC](https://arxiv.org/abs/2605.05544) 与 [MAC](https://openreview.net/forum?id=WXGb9unEHo)。
6. **第六层（Agentic RL 与持续性任务）**：跟进 [LOOP](https://arxiv.org/abs/2502.01600)、[WebRL](https://arxiv.org/abs/2411.02337)、[AgentGym-RL](https://openreview.net/forum?id=ZgCCDwcGwn)、[Agent Lightning](https://arxiv.org/abs/2508.03680)、[HCAPO](https://arxiv.org/abs/2603.08754)、[AgentPRM](https://arxiv.org/abs/2502.10325)、[SPA](https://arxiv.org/abs/2505.20732)、[Agentic Credit Assignment Survey](https://arxiv.org/abs/2604.09459) 以及平均奖励代表工作 [ATRPO](https://arxiv.org/abs/2106.07329) 与 [RVI-SAC](https://arxiv.org/abs/2408.01972)。

### 10.4 结语

回顾长期收益强化学习的演进历程，领域范式已经完成了从早期“单步原子动作 + 回合级稀疏奖励 + 一阶 TD 逐步备份”，向“多尺度时间抽象 + 多粒度显式信用分配 + 离线行为先验 + 定向课程探索 + 短程世界模型规划 + 验证器与过程奖励协同 + 安全在线微调”的系统性跃迁。前沿研究的焦点早已超越单一 Bellman 备份公式的局部修补，全面转向对轨迹结构、动作粒度、奖励拓扑与数据闭环的联合重构。

在物理机器人与连续控制领域，离线到在线微调、自适应动作分块与潜空间短程世界模型的深度耦合正在重塑长程操作的样本效率天花板；在多轮软件交互与大语言模型智能体领域，执行与训练解耦框架、事后反事实归因与防黑客过程奖励模型正将强化学习推向复杂工具链编排；而在无自然终点的工业决策系统中，平均奖励准则、离策略评估与安全发布护栏则为稳态长期价值提供了严谨的治理框架。归根结底，长期收益强化学习的本质是一套关于时间结构的工程与数学方法论——它将漫长混沌的未来拆解为可验证的阶段信号，将粗粒度的终局成败还原为可归因的步骤贡献，将底层原子动作封装为可复用的时序积木，并在离线先验、在线探索与安全边界的闭环交织中实现持续进化。

## 参考文献

- Sutton and Barto. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).
- Sutton, Precup and Singh, 1999. [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](https://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf).
- Ng, Harada and Russell, 1999. [Policy invariance under reward transformations: Theory and application to reward shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
- Schulman et al., 2015. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- Munos et al., 2016. [Safe and Efficient Off-Policy Reinforcement Learning](https://arxiv.org/abs/1606.02647).
- Bacon, Harb and Precup, 2016. [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140).
- Schulman et al., 2017. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Christiano et al., 2017. [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741).
- Pathak et al., 2017. [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363).
- Vezhnevets et al., 2017. [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161).
- Andrychowicz et al., 2017. [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495).
- Nachum et al., 2018. [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296).
- Espeholt et al., 2018. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561).
- Burda et al., 2018. [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894).
- Eysenbach et al., 2018. [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070).
- Arjona-Medina et al., 2018/2019. [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857).
- Hung et al., 2018/2019. [Optimizing Agent Behavior over Long Time Scales by Transporting Value](https://arxiv.org/abs/1810.06721).
- Ecoffet et al., 2019. [Go-Explore: a New Approach for Hard-Exploration Problems](https://arxiv.org/abs/1901.10995).
- Pong et al., 2019. [Skew-Fit: State-Covering Self-Supervised Reinforcement Learning](https://arxiv.org/abs/1903.03698).
- Portelas et al., 2020. [Automatic Curriculum Learning For Deep RL: A Short Survey](https://arxiv.org/abs/2003.04664).
- Fu et al., 2020. [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://arxiv.org/abs/2004.07219).
- Kumar et al., 2020. [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779).
- Nair et al., 2020. [AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359).
- Raposo et al., 2021. [Synthetic Returns for Long-Term Credit Assignment](https://arxiv.org/abs/2102.12425).
- Kostrikov et al., 2021. [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).
- Fujimoto and Gu, 2021. [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860).
- Zhang and Ross, 2021. [On-Policy Deep Reinforcement Learning for the Average-Reward Criterion](https://arxiv.org/abs/2106.07329).
- Hafner et al., 2023. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104).
- Yang et al., 2023. [Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis](https://proceedings.neurips.cc/paper_files/paper/2023/file/d8bd445c2abe1343cce0e14b361b2fb3-Paper-Conference.pdf).
- Li, Zhu and Zhang, 2023. [Offline Hierarchical Reinforcement Learning Using Subgoal Generation](https://www.ijcai.org/proceedings/2023/0469.pdf).
- Hansen et al., 2023. [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828).
- Pignatelli et al., 2023/2024. [A Survey of Temporal Credit Assignment in Deep Reinforcement Learning](https://arxiv.org/abs/2312.01072).
- Jin et al., 2024. [Feasible Q-Learning for Average Reward Reinforcement Learning](https://proceedings.mlr.press/v238/jin24b.html).
- Zhou et al., 2024. [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning](https://arxiv.org/abs/2408.01972).
- Park et al., 2024. [OGBench: Benchmarking Offline Goal-Conditioned RL](https://arxiv.org/abs/2410.20092).
- Gur et al., 2024. [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://arxiv.org/abs/2411.02337).
- Chen et al., 2025. [Reinforcement Learning for Long-Horizon Interactive LLM Agents](https://arxiv.org/abs/2502.01600).
- Choudhury, 2025. [Process Reward Models for LLM Agents: Practical Framework and Directions](https://arxiv.org/abs/2502.10325).
- Tian, Celik and Neumann, 2025. [Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns](https://arxiv.org/abs/2503.03660).
- Lu et al., 2025. [Stepwise Progress Attribution: A Step-level Reinforcement Learning Framework for LLM Agents](https://arxiv.org/abs/2505.20732).
- Chen et al., 2025. [DISCOVER: Directed Exploration for Sparse-Rewards Very Long-Horizon Goal-Conditioned Reinforcement Learning](https://arxiv.org/abs/2505.19850).
- Li, Zhou and Levine, 2025. [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969).
- Pignatelli et al., 2025. [Deep Reinforcement Learning with Gradient Eligibility Traces](https://arxiv.org/abs/2507.09087).
- Luo et al., 2025. [Agent Lightning: Train ANY AI Agents with Reinforcement Learning](https://arxiv.org/abs/2508.03680).
- Yang et al., 2025. [Actor-Critic for Continuous Action Chunks](https://arxiv.org/abs/2508.11143).
- Park et al., 2025/2026. [Scalable Offline Model-Based RL with Action Chunks](https://openreview.net/forum?id=WXGb9unEHo).
- Gu et al., 2026. [STO-RL: Offline RL under Sparse Rewards via LLM-Guided Subgoal Temporal Order](https://arxiv.org/abs/2601.08107).
- Xi et al., 2026. [AgentGym-RL: An Open-Source Framework to Train LLM Agents for Long-Horizon Decision Making via Multi-Turn RL](https://openreview.net/forum?id=ZgCCDwcGwn).
- Tan et al., 2026. [Hindsight Credit Assignment for Long-Horizon LLM Agents](https://arxiv.org/abs/2603.08754).
- Liu et al., 2026. [Agentic Credit Assignment: A Survey of Credit Assignment in LLM Agents](https://arxiv.org/abs/2604.09459).
- Chen et al., 2026. [ACSAC: Adaptive Chunk Size Actor-Critic with Causal Transformer Q-Network](https://arxiv.org/abs/2605.11009).
- Gireesh, Ju and Wang, 2026. [Adaptive Q-Chunking for Offline-to-Online Reinforcement Learning](https://arxiv.org/abs/2605.05544).

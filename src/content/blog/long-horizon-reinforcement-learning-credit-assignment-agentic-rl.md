---
title: "长期收益强化学习算法研究进展：从时间信用分配到 Agentic RL"
description: "系统梳理长时域、稀疏延迟奖励强化学习中的信用分配、时间抽象、离线到在线、世界模型与 LLM Agent RL 进展"
pubDate: 2026-06-10
updatedDate: 2026-06-10
tags:
  - "Reinforcement Learning"
  - "Long-Horizon RL"
  - "Credit Assignment"
  - "Agentic RL"
  - "Survey"
draft: false
---

长期收益强化学习听起来像一个单独的算法问题，但更准确地说，它是一组相互缠绕的训练困难。标准强化学习本来就最大化长期累计回报：

$$
J(\pi)=\mathbb{E}_{\tau\sim \pi}
\left[
\sum_{t=0}^{T}\gamma^t r_t
\right],
$$

其中 $\tau=(s_0,a_0,r_0,\dots,s_T)$ 是策略 $\pi$ 诱导出的轨迹，$\gamma\in[0,1)$ 是折扣因子。问题不在于目标函数是否“长期”，而在于当奖励非常稀疏、非常延迟、任务 horizon 很长、环境部分可观测、动作空间又很大时，训练系统很难判断哪些早期决策真正导致了最终结果。

这篇报告讨论的“长期收益 RL”主要覆盖三类场景。

第一类是 **finite-horizon sparse-reward tasks**：例如机器人操作、Atari hard exploration、网页任务、AppWorld 任务。它们通常有明确 episode，最终成功或失败才给奖励。

第二类是 **long-horizon goal-conditioned RL**：目标 $g$ 显式出现，策略写成 $\pi(a\mid s,g)$。这类问题的难点常常是目标空间大、随机探索很难碰到成功轨迹。

第三类是 **continuing tasks**：推荐、调度、长期用户价值、能源管理等任务没有自然 episode。此时 discounted return 未必是唯一合理目标，average reward criterion 可能更贴近真实业务：

$$
\rho(\pi)=
\lim_{T\to\infty}
\frac{1}{T}
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{T-1} r_t
\right].
$$

这三类问题都会被称为“长期收益”，但算法假设和评估方式并不相同。把它们混在一起，会导致两个常见误解：一是把“奖励传播慢”误认为所有长时域任务的唯一瓶颈；二是把某个机器人 manipulation 或网页 agent 结果外推成一般 RL 结论。

## 1. 长期收益 RL 到底难在哪里

用 Markov Decision Process 表示，强化学习任务由状态空间 $\mathcal{S}$、动作空间 $\mathcal{A}$、转移分布 $P(s'\mid s,a)$、奖励函数 $r(s,a,s')$、初始分布 $\mu_0$ 和折扣因子 $\gamma$ 组成。策略 $\pi(a\mid s)$ 的目标是最大化价值函数：

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

以及动作价值函数：

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

在短 horizon、dense reward 的任务里，局部动作和后续奖励之间的统计关系相对容易学习。长时域稀疏奖励任务则把问题拆成四个困难。

**Temporal credit assignment**：最终奖励应该分给哪些早期动作？如果一个 agent 在第 200 步成功开门，真正关键的可能是第 20 步拿钥匙、第 70 步保存路线信息、第 130 步避开障碍，而不是第 199 步的最后一个动作。

**Value propagation**：终局奖励如何快速传回早期状态？一阶 TD 每次 Bellman backup 只把信息往前传一步。在几百步延迟下，价值估计可能需要大量样本才能形成可用梯度。

**Exploration**：成功轨迹从哪里来？如果随机策略几乎不可能完成任务，再好的 credit assignment 也没有正样本可分配。

**Temporal abstraction**：是否应该把单步动作提升为技能、子目标、option、action chunk 或高层计划？如果有效决策频率从每步一次变为每 $K$ 步一次，优化问题会变得完全不同。

[Temporal Credit Assignment Survey](https://arxiv.org/abs/2312.01072) 把 credit assignment 定义为从有限经验中学习动作对后续结果的影响，并强调现实反馈往往 noisy、delayed，且几乎不说明 causal responsibility。这一点非常关键：长期收益 RL 不是简单把 return 里的 $\gamma$ 调大，而是要在有限轨迹中识别影响链条。

从这个视角看，长期收益 RL 的历史可以理解为十条互补路线：多步 TD 基础设施、显式 credit assignment、稀疏探索、HRL、action chunking、world model、offline/offline-to-online、LLM Agent RL、reward shaping、average reward。

为了避免后文混淆，先把几个经常被合并使用的术语拆开。

**Horizon 长** 指 episode 或有效决策链条很长。即使每步都有 dense reward，只要策略必须在很长时间内保持稳定，仍然是 long-horizon optimization。典型例子是长程导航、长序列机器人装配、长期用户留存优化。

**Reward delay 大** 指关键动作和回报之间相隔很多步。延迟奖励不一定稀疏，例如一个任务可能每步都有小 reward，但真正决定成功的是很早的准备动作。

**Reward sparsity 高** 指非零 reward 很少。它首先是探索问题：agent 必须先碰到奖励，才有信号可学习。

**Partial observability 强** 指当前 observation 不足以决定最优动作，agent 必须依赖记忆。此时长时域 credit assignment 往往和 memory learning 纠缠在一起。

**Continuing objective** 指任务没有自然终止。此时优化长期平均效用、长期约束违约率或稳定状态收益，和有限 episode 成功率不是同一个问题。

这几个维度可以同时出现，也可以分开出现。一个网页 agent 任务通常同时有长 horizon、稀疏最终 reward、部分可观测和复杂动作空间；一个推荐系统任务可能没有稀疏终局成功，却有 delayed business metric 和 continuing objective；一个机器人 pick-and-place 任务可能主要是 sparse exploration 和 contact-rich control，而不是 memory。

因此，选择算法前更有用的问题不是“哪个 long-horizon RL 算法最好”，而是：

| 诊断问题 | 指向的主要路线 |
| --- | --- |
| 成功轨迹几乎采不到吗？ | exploration、curriculum、offline prior、demonstration |
| 成功轨迹有了，但不知道哪些步骤关键吗？ | RUDDER、TVT、COCOA、HCAPO、process reward |
| 原子动作太细，决策链条太长吗？ | HRL、options、subgoal、action chunking |
| 在线交互昂贵或危险吗？ | offline RL、offline-to-online、安全在线改进 |
| 环境可预测且交互样本贵吗？ | world model、MPC、latent planning |
| 任务没有自然 episode 吗？ | average reward、continuing RL、OPE、safe policy improvement |
| reward 来自 verifier 或偏好模型吗？ | reward shaping、PRM、reward hacking control |

## 2. 多步 TD、eligibility trace 与 GAE：长期收益的基础设施

最基础的路线不改变任务结构，只改变 return 或 advantage 的估计方式。

普通一阶 TD 目标为：

$$
y_t=r_t+\gamma V(s_{t+1}).
$$

它的优点是方差低、更新稳定，缺点是在长延迟奖励中传播很慢。$n$-step return 把未来 $n$ 步奖励一次性纳入目标：

$$
G_t^{(n)}
=
\sum_{i=0}^{n-1}\gamma^i r_{t+i}
+
\gamma^n V(s_{t+n}).
$$

当 $n$ 变大时，奖励传播更快，但估计方差通常也更高；在 off-policy 场景中，如果轨迹来自行为策略 $\mu$ 而目标策略是 $\pi$，还会引入分布不匹配问题。

TD($\lambda$) 和 eligibility traces 用指数加权的多步 return 做折中：

$$
G_t^{\lambda}
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}G_t^{(n)}.
$$

等价地，eligibility trace 为最近访问过的状态或参数保留“资格”：

$$
e_t=\gamma\lambda e_{t-1}+\nabla_{\theta} V_{\theta}(s_t),
$$

然后用 TD error 更新：

$$
\delta_t=r_t+\gamma V_{\theta}(s_{t+1})-V_{\theta}(s_t).
$$

这让误差信号能够沿访问轨迹向后分配，而不是只更新当前状态。

在 policy gradient 和 actor-critic 中，[GAE](https://arxiv.org/abs/1506.02438) 把这种思想用于 advantage estimation。先定义 TD residual：

$$
\delta_t^V=r_t+\gamma V(s_{t+1})-V(s_t),
$$

再定义：

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l \delta_{t+l}^{V}.
$$

GAE 的价值在于降低 policy gradient 方差，并通过 $\lambda$ 控制 bias-variance trade-off。它不是长期 credit assignment 的完整答案，因为它仍依赖 value function 是否能表示远期影响。

off-policy 多步学习又进一步引出 Retrace、V-trace 和 IMPALA。[Retrace($\lambda$)](https://arxiv.org/abs/1606.02647) 通过截断重要性采样系数提高 off-policy 多步 return 的安全性。[IMPALA](https://arxiv.org/abs/1802.01561) 用 actor-learner 架构扩展采样吞吐，并用 V-trace 修正 actor 生成数据与 learner 当前策略之间的偏差。[PPO](https://arxiv.org/abs/1707.06347) 则通过 clipped surrogate objective 提供稳定 policy optimization，因此经常作为长时域任务的训练底座。

这些方法的共同作用是改善 **value propagation** 和优化稳定性。它们解决的不是“如何发现关键事件”，而是“已有奖励信号如何更有效地进入价值估计和策略梯度”。

| 方法 | 核心作用 | 对长期收益的帮助 | 主要限制 |
| --- | --- | --- | --- |
| 1-step TD | 低方差 bootstrapping | 稳定训练 | 延迟奖励传播慢 |
| $n$-step return | 一次传播多步奖励 | 更快回传终局信号 | 方差更高，off-policy 更难 |
| TD($\lambda$) / eligibility trace | 多步 return 加权折中 | 改善短到中等延迟 | 极长 horizon 仍不够 |
| GAE | actor-critic 中稳定优势估计 | 降低 policy gradient 方差 | 依赖 value function 质量 |
| Retrace / V-trace | off-policy 多步修正 | 分布式训练更稳定 | 不解决探索与任务结构 |

进一步看，这些方法实际在调三个旋钮。

第一个旋钮是 **bootstrap depth**。1-step TD 几乎完全依赖 bootstrap，Monte Carlo return 完全不 bootstrap，$n$-step return 和 $\lambda$-return 在中间取折中。长时域任务中，如果 bootstrap 太浅，终局 reward 传不回来；如果完全不用 bootstrap，方差又可能让 policy gradient 淹没在噪声里。

第二个旋钮是 **on-policy 还是 off-policy**。on-policy 方法可以更直接使用当前策略轨迹，但样本效率较低；off-policy 方法可以复用 replay buffer 或大规模 actor 采样，但多步 return 会遇到重要性采样系数爆炸、截断偏差和分布漂移。

第三个旋钮是 **value function 的表示能力**。即使 TD($\lambda$) 或 GAE 形式上能把长期 reward 写进 target，如果 $V_\theta$ 无法表示“早期关键状态”和“远期成功概率”之间的关系，更新仍然会退化为噪声。许多长时域失败不是 Bellman 方程错了，而是函数逼近、数据覆盖和探索共同不足。

这一点在 actor-critic 中尤其明显。策略梯度常写作：

$$
\nabla_\theta J(\pi_\theta)
\approx
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t\mid s_t)\hat A_t
\right].
$$

如果 $\hat A_t$ 对早期动作几乎全是零或噪声，策略更新就不会偏向真正关键的早期决策。GAE 改善的是 $\hat A_t$ 的估计质量，但它不能凭空生成“成功轨迹”，也不能保证 advantage 的因果解释正确。

2025 年的 [Deep Reinforcement Learning with Gradient Eligibility Traces](https://arxiv.org/abs/2507.09087) 也反映了一个趋势：eligibility trace 这类早期思想正在被重新带回深度 RL。动机很直接：一阶 TD 的 credit assignment 太慢，而更原则化的 gradient TD trace 在深度场景里仍有工程空间。

结论是：多步 TD、eligibility trace、GAE、Retrace 和 V-trace 是长期收益 RL 的基础设施，但不是终点。它们改善的是传播和估计，不能单独解决稀疏探索、因果贡献识别或时间抽象。

## 3. 显式时间信用分配：把远期奖励分给真正的过去事件

如果最终奖励隔了几百步才出现，另一条路线是不等 Bellman backup 慢慢传播，而是直接学习“哪些过去事件贡献了未来奖励”。

### 3.1 RUDDER：reward redistribution

[RUDDER](https://arxiv.org/abs/1806.07857) 是显式 credit assignment 的代表。它的核心思想是 reward redistribution：训练一个 return decomposition model，把 episode-level return 分解到轨迹中的关键事件上。

如果原始 return 是：

$$
G_0=\sum_{t=0}^{T}\gamma^t r_t,
$$

RUDDER 希望构造一个 redistributed reward $r'_t$，使新过程和原过程 return-equivalent，同时把远期奖励提前分配到真正有贡献的时间点。理想情况下，重分配后的 expected future reward 接近 0：

$$
\mathbb{E}\left[\sum_{k=1}^{\infty}\gamma^k r'_{t+k}\mid s_t,a_t\right]\approx 0.
$$

这样 $Q$-value 估计主要变成估计 immediate redistributed reward，而不是长链条的未来回报。

RUDDER 的直觉很强：如果第 20 步拿钥匙导致第 200 步开门成功，那么训练信号不应只在第 200 步出现，而应该部分回到第 20 步。它的主要难点也很清楚：return decomposition model 本身要能识别关键贡献事件。在高维视觉、多模态轨迹、复杂工具调用中，贡献识别可能和原任务一样难。

### 3.2 Temporal Value Transport：用记忆注意力传送价值

[Temporal Value Transport](https://arxiv.org/abs/1810.06721) 关注需要记忆的长时域任务。它使用 memory attention：如果 agent 当前为了预测 reward 或 value 注意到了很久以前的 memory，那么当前价值可以被 transport 回那个过去事件。

形式化地说，当前 value target 不只更新当前状态，还通过 attention weights $w_{t,k}$ 更新被访问的过去记忆 $m_k$：

$$
\Delta V(m_k) \propto w_{t,k}\,\Delta V(s_t).
$$

这类方法尤其适合 POMDP 或需要 recall 的任务。例如 agent 很早看到密码、钥匙位置或地图结构，几十步后才使用这些信息。TVT 的风险是 attention 相关不等于因果贡献。模型注意到某个 memory，不代表那个 memory 对奖励有真实 causal responsibility。

### 3.3 Synthetic Returns：学习状态和远期奖励的直接关联

[Synthetic Returns](https://arxiv.org/abs/2102.12425) 提出 state-associative learning，让 agent 学习某个状态对任意远未来奖励的贡献，并把这种预测作为 synthetic return 用于 TD learning。

普通 TD 依赖一步步传播：

$$
s_0 \leftarrow s_1 \leftarrow s_2 \leftarrow \cdots \leftarrow r_T.
$$

Synthetic Returns 试图直接学习：

$$
s_k \rightarrow \text{future reward contribution}.
$$

它和 RUDDER 的区别在于：RUDDER 分解整条轨迹 return 并重分配 reward；Synthetic Returns 学状态与远期 reward 的关联，让 TD 不必完全依赖逐步回传。论文中特别强调，长延迟和大量无关中间事件会让 TD 学习失败，而 synthetic return 可以让关键状态更早获得训练信号。

### 3.4 COCOA 与反事实信用分配

更接近因果定义的做法会问：

如果当时没有执行 $a_t$，后面还会得到这个 reward 吗？

[COCOA](https://proceedings.neurips.cc/paper_files/paper/2023/file/d8bd445c2abe1343cce0e14b361b2fb3-Paper-Conference.pdf) 建立在 Hindsight Credit Assignment 之上，用 counterfactual contribution analysis 衡量动作对后续 reward 的贡献。这类方法把 credit 从“时间距离近不近”推进到“动作是否改变了结果分布”。

如果记 $Y$ 为未来 outcome，credit 可以理解为某种反事实差异：

$$
C_t
\approx
\mathbb{E}[Y\mid s_t,a_t]
-
\mathbb{E}[Y\mid s_t,\tilde a_t],
$$

其中 $\tilde a_t$ 是反事实动作或参考动作。难点在于反事实 outcome 不可直接观测，需要模型、估计器或结构假设。工程复杂度明显高于多步 TD。

### 3.5 LLM Agent 中的 hindsight credit

显式 credit assignment 在 LLM agent 中重新变热。长时域 LLM agent 的一次 episode 可能包含几十个 thought、tool call、API invocation、网页操作和恢复动作，最终只有成功或失败信号。

[HCAPO](https://arxiv.org/abs/2603.08754) 针对 long-horizon LLM agents 的稀疏奖励训练，指出 GRPO 这类 value-free 方法在 step-level $Q$ 估计和 intermediate baseline 上存在问题，并用 hindsight reasoning 细化关键步骤的 credit。它的适用语境不是传统连续控制，而是 multi-step LLM agent benchmark，例如 WebShop 和 ALFWorld。

[Agentic Credit Assignment Survey](https://arxiv.org/abs/2604.09459) 则把 2024-2026 早期的 agentic credit assignment 方法按粒度整理为 token、segment、step、turn、multi-agent 等层级。这个分类说明，LLM agent 的 credit assignment 比传统控制多了一个维度：credit 可以分给 token、工具调用、自然语言中间步骤，也可以分给整个 turn。

显式 credit assignment 的核心价值是减少“奖励只出现在终点”的训练浪费。它的核心风险是贡献识别本身依赖模型和假设。如果模型把 credit 分错，训练会被导向错误策略。

把这一组方法放在一起，可以看到它们的差异不在“是否多步”，而在“credit 是怎么被构造出来的”。

| 方法 | credit 来源 | 适合的结构 | 主要失败模式 |
| --- | --- | --- | --- |
| RUDDER | return decomposition model | 轨迹里存在可识别关键事件 | 分解模型把相关性误当贡献 |
| TVT | attention over memory | 远期 reward 依赖早期观察或记忆 | attention 不等于 causality |
| Synthetic Returns | state-associative prediction | 状态和远期 reward 有直接统计关联 | 关联学习受分布外状态影响 |
| COCOA / HCA | counterfactual contribution | 能定义合理反事实 baseline | 反事实不可观测，估计器偏差 |
| HCAPO / Agentic credit | hindsight reasoning / critic | LLM agent 的 turn、tool call、step 轨迹 | verifier 或 hindsight critic 错配 |

工程上可以把显式 credit assignment 看成一层“reward translator”：输入原始轨迹和粗粒度 outcome，输出更细粒度的训练信号。这个 translator 可以是序列模型、记忆 attention、反事实模型、LLM judge、verifier 或 process reward model。它一旦可靠，长期训练会明显变简单；它一旦不可靠，错误信号会比 sparse reward 更危险，因为 agent 会更快学到错的捷径。

## 4. 稀疏奖励探索与 curriculum：成功轨迹从哪里来

credit assignment 假设已经有包含奖励的轨迹。稀疏奖励任务常常更早失败：agent 根本碰不到成功状态。

### 4.1 HER：把失败轨迹改写成成功经验

[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) 面向 goal-conditioned sparse reward。给定目标 $g$，轨迹没有达到 $g$，按原任务看是失败；但这条轨迹可能到达了另一个目标 $g'$。HER 把同一条经验重标注为：

$$
(s_t,a_t,s_{t+1},g')
$$

并把它当作达成 $g'$ 的成功样本。

它的关键不是直接让 agent 达成原目标，而是提高 sparse reward 下的数据利用率。HER 特别适合目标表示清晰的任务，例如机械臂 reach、push、pick-and-place。不适合没有自然 goal representation 的任务，例如开放网页任务或纯语言 agent。

### 4.2 Go-Explore：先回到有希望的状态，再探索

[Go-Explore](https://arxiv.org/abs/1901.10995) 指出 hard exploration 任务失败常常不是价值函数问题，而是 agent 无法稳定回到有希望的中间状态继续探索。它的核心流程是：

```text
1. 记住已经发现的 interesting states；
2. 选择一个 promising state；
3. 返回该状态；
4. 从那里继续探索；
5. 找到解后再 robustify。
```

在 Montezuma's Revenge 和 Pitfall 这类 Atari 任务中，这种“先返回，再探索”的思路比纯随机探索有效得多。它告诉我们：长期收益 RL 中的探索不能只靠更大的 entropy bonus。状态覆盖、可返回性和 archive 机制都很重要。

### 4.3 Intrinsic motivation：RND、ICM、Skew-Fit、DIAYN

[RND](https://arxiv.org/abs/1810.12894) 使用随机网络蒸馏，把预测误差作为 novelty signal，引导 agent 探索不熟悉状态。[ICM](https://arxiv.org/abs/1705.05363) 用 forward/inverse dynamics 的预测误差构造 curiosity。它们的共同思想是：外部 reward 稀疏时，用内部奖励提供探索梯度。

[Skew-Fit](https://arxiv.org/abs/1903.03698) 关注 goal distribution，通过最大化状态覆盖来训练 goal-reaching agent。它不是手工指定所有目标，而是让 agent 逐渐把目标分布推向当前不常见但可达的状态。

[DIAYN](https://arxiv.org/abs/1802.06070) 在没有外部奖励的情况下学习多样技能，通过最大化 skill 和状态之间的互信息，使不同 skill 对应可区分的行为模式。它解决的是“先学会做很多事情”，再服务下游稀疏奖励任务。

这些方法对长期收益的帮助主要是产生有价值数据，而不是解释终局 reward 的因果来源。它们和 RUDDER、TVT、HCAPO 这类 credit assignment 方法是互补关系。

### 4.4 自动课程学习与 directed exploration

[Automatic Curriculum Learning](https://arxiv.org/abs/2003.04664) 把训练任务分布作为可调对象：不让 agent 一开始就面对最难任务，而是根据能力动态选择难度、目标或环境参数。

2026 前后的 [DISCOVER](https://arxiv.org/abs/2505.19850) 更明确地面向 sparse-reward goal-conditioned very long-horizon RL：它选择“朝向目标任务方向”的 exploratory goals，而不是在所有可达目标里无方向地覆盖。这个方向重要，因为长时域任务空间的体积可能巨大，无差别探索会浪费样本。

探索类方法回答的问题是：成功轨迹从哪里来？credit assignment 回答的是：成功轨迹里的哪些动作重要？长期收益 RL 往往必须二者结合。

一个实用的判断方式是把任务放进下面的二维表。

| 轨迹里有成功样本吗？ | 能判断哪些步骤重要吗？ | 更优先的方向 |
| --- | --- | --- |
| 没有 | 否 | exploration、curriculum、demonstration、offline prior |
| 有少量 | 否 | reward redistribution、hindsight credit、process reward |
| 有较多 | 是 | actor-critic 稳定化、offline-to-online、model-based planning |
| 没有但有目标结构 | 部分可以 | HER、goal relabeling、goal curriculum、Skew-Fit |

这解释了为什么很多长期收益系统必须先做数据工程。对于极稀疏任务，算法论文里看起来核心是 critic 或 policy loss，实际落地时第一优先级常常是构造可学习数据：示范、脚本策略、回放轨迹、课程环境、可逆 reset、模拟器并行、失败轨迹重标注。没有这些，credit assignment 只能在全零 reward 上运行。

## 5. 时间抽象与 HRL：把长 horizon 分解成可执行结构

如果一个任务需要 1000 个 primitive actions，直接学习 $\pi(a_t\mid s_t)$ 会让 credit assignment 和 exploration 都非常困难。Hierarchical Reinforcement Learning 的直觉是把底层动作提升为 temporally extended actions。

### 5.1 Options framework

[Options framework](https://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf) 把普通动作扩展为 option。一个 option $o$ 包含三部分：

$$
o=(\mathcal{I}_o,\pi_o,\beta_o),
$$

其中 $\mathcal{I}_o$ 是 initiation set，$\pi_o(a\mid s)$ 是 option 内部策略，$\beta_o(s)$ 是终止概率。option 可以理解为一段 closed-loop policy，例如“走到门口”“抓取物体”“打开网页中的某个菜单”。

如果平均 option length 是 $K$，有效决策 horizon 可以粗略理解为：

$$
H_{\text{effective}}\approx \frac{H}{K}.
$$

这不是严格保证，因为 option 本身也可能失败或产生副作用，但它说明了时间抽象为什么有用：高层策略不必在每个 primitive timestep 上重新规划。

### 5.2 Option-Critic、FeUdal Networks、HIRO

[Option-Critic](https://arxiv.org/abs/1609.05140) 试图端到端学习 option 内部策略和 termination condition，不依赖人工定义 option。它推进了 HRL 的可学习性，但也带来 option collapse、语义不稳定和探索不足等问题。

[FeUdal Networks](https://arxiv.org/abs/1703.01161) 用 Manager/Worker 架构区分时间尺度：Manager 在较慢频率上输出抽象 goal，Worker 在环境步上执行动作。它把 long timescale credit assignment 显式写进架构设计。

[HIRO](https://arxiv.org/abs/1805.08296) 面向 continuous control 和 off-policy HRL。高层策略提出 subgoal，低层策略负责达到 subgoal。由于低层策略会变化，高层动作的语义也会变，因此 HIRO 提出 off-policy correction 来缓解高低层非平稳耦合。

| 方法 | 时间抽象形式 | 解决的问题 | 主要风险 |
| --- | --- | --- | --- |
| Options | closed-loop option | temporally extended actions | option discovery 难 |
| Option-Critic | 端到端 option 学习 | 减少人工 option 设计 | collapse、终止不稳定 |
| FeUdal Networks | Manager/Worker | 多时间尺度 credit | goal 表示依赖强 |
| HIRO | high-level subgoal | off-policy HRL 连续控制 | 高低层非平稳 |

### 5.3 Offline HRL 与 LLM-guided subgoal

随着 offline RL 重要性上升，HRL 也开始从在线探索转向离线数据中的子目标提取。[Guider](https://www.ijcai.org/proceedings/2023/0469.pdf) 用高层策略顺序生成 subgoal，引导 offline long-horizon sparse-reward task 学习。[RD-HRL](https://openreview.net/forum?id=5E5sd3TWGD) 强调 reliable sub-goals 和 value-guided sub-goal planning，目标是缩短高低层 planner 的有效 horizon。[OHIO](https://openreview.net/forum?id=dTPz4rEDok) 则处理 offline hierarchical policy learning 中高层动作不可观测、数据来自不同策略结构的问题。

2026 年的 [STO-RL](https://arxiv.org/abs/2601.08107) 把 LLM 用于生成有时间顺序的 subgoal sequence，再把状态映射到 subgoal stage，并用 potential-based reward shaping 生成 temporally consistent 的 dense reward。它的重要点不只是“LLM 生成子目标”，而是强调子目标之间有时间依赖：如果 shaping 忽略顺序，agent 可能被错误中间目标牵引。

HRL 的核心瓶颈始终是子目标质量。好的子目标缩短 horizon；坏的子目标制造错误优化方向。LLM 可以提供任务先验，但不能替代可验证的可达性和奖励一致性检查。

一个好的子目标通常需要同时满足四个条件。

**可达性**：低层策略在足够多初始状态下能达到子目标。如果子目标只是语言上合理，但在动力学上不可达，高层计划会变成空中楼阁。

**可验证性**：系统能判断子目标是否完成。机器人里可能是位姿、接触状态或视觉检测；网页 agent 里可能是 DOM 状态、API response 或 verifier；语言任务里可能是单元测试或结构化检查。

**时间顺序一致性**：子目标之间不能互相打架。例如“先提交表单，再补全必填字段”就是错误顺序。STO-RL 这类方法强调 temporal order，原因就在这里。

**对最终目标有贡献**：子目标不能只是容易完成，还必须推动最终 return。许多 curriculum 方法早期能提高中间成功率，但如果中间目标和最终目标错配，会形成局部策略。

从实现角度看，HRL 和 action chunking 的差别也很重要。HRL 通常希望得到 closed-loop skill：执行中能根据状态反馈调整；action chunking 更常得到 semi-open-loop action sequence：一段时间内按序执行。前者表达力更强但训练更复杂，后者工程更简单但对扰动更敏感。

## 6. Action Chunking：近两年的工程友好时间抽象

Action chunking 是 2025-2026 年长期收益 RL 中非常活跃的路线，尤其在机器人和 offline-to-online RL 中。

普通策略每步输出一个动作：

$$
\pi(a_t\mid s_t).
$$

Action chunking 让策略一次输出一段动作：

$$
\pi(a_{t:t+K-1}\mid s_t).
$$

它的直接好处有三个：降低有效决策频率；让探索更 temporally coherent；让 $K$-step backup 更自然。它和 HRL 相似，但更轻量。它通常不学习 initiation set 或 termination condition，而是直接把动作空间从 $\mathcal{A}$ 变成 $\mathcal{A}^K$。

### 6.1 Q-chunking：chunk-level critic

[Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969) 提出 Q-chunking，面向 long-horizon sparse-reward offline-to-online RL。它的关键不是简单使用 $n$-step return，而是把 critic 的评价对象改为整段 action chunk：

$$
Q_K(s_t,a_{t:t+K-1}).
$$

对应 target 可以写成：

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

普通 $n$-step TD 仍然把多步 reward 回填到单步 $Q(s_t,a_t)$；Q-chunking 明确评价整段 action sequence。因此它更能利用离线数据中的 temporally consistent behavior，也更适合 offline prior 到 online exploration 的转换。

这一区别可以写得更直接。普通 $n$-step Q-learning 学的是：

$$
Q(s_t,a_t)
\leftarrow
\sum_{i=0}^{n-1}\gamma^i r_{t+i}
+
\gamma^n \max_{a'} Q(s_{t+n},a').
$$

也就是说，训练目标跨了 $n$ 步，但 action variable 仍然只有 $a_t$。如果中间动作 $a_{t+1:t+n-1}$ 来自 replay buffer，它们只是 target 的一部分，不是 critic 要评价的决策对象。

Q-chunking 学的是：

$$
Q_K(s_t,a_t,a_{t+1},\dots,a_{t+K-1}),
$$

critic 的输入显式包含整个动作片段。这样一来，critic 可以区分“同一个起点和第一个动作，但后续动作序列不同”的情况。在长时域稀疏奖励里，这个差异很关键：单步动作往往没有足够信息，真正有意义的是一段连贯操作。

但是这个收益也有代价。动作空间从 $\mathcal{A}$ 变成 $\mathcal{A}^K$，维度随 $K$ 线性增加；如果每步动作是连续向量，chunk policy 的输出维度会明显变大。固定 $K$ 还会把“何时重新决策”写死。自由空间移动阶段适合长 chunk，接触、避障、网页状态变化或工具调用失败阶段则需要短 chunk 或闭环修正。

不过，chunked policy 常常更 open-loop。如果环境需要每步强反馈控制，过长 chunk 会降低反应性。这也是后续 adaptive chunk size 方法出现的原因。

### 6.2 AC3、T-SAC、ACSAC、AQC

[AC3](https://arxiv.org/abs/2508.11143) 把 continuous action chunk 放进 actor-critic，用于 long-horizon robotic manipulation with sparse reward。它的稳定机制包括：actor 只从成功轨迹更新，critic 使用 intra-chunk $n$-step returns，并引入自监督 intrinsic reward。论文在 BiGym 和 RLBench 的 25 个任务上实验，定位很明确：机器人长时域操作。

[T-SAC](https://arxiv.org/abs/2503.03660) 的角度不同。它不一定让 actor 输出 chunk，而是让 critic 看到短 trajectory segment，用 Transformer critic 和 $N$-step target 增强 long-horizon credit assignment，同时保持 SAC 风格的一步 policy update。它更像“chunking the critic”，而不是“chunking the actor”。

[ACSAC](https://arxiv.org/abs/2605.11009) 针对固定 chunk size 的局限，使用 causal Transformer critic 评估不同 chunk size 的 expected return，在 chunk boundary 自适应选择 $K$。它要平衡两个矛盾：长 chunk 有 temporal coherence 和快速 value backup，短 chunk 有更强 reactivity。

[AQC](https://arxiv.org/abs/2605.05544) 也解决固定 chunk size 问题，但强调 naive 比较不同 horizon 的 Q 值会因为 discount-scale mismatch 偏向短 chunk。它通过相对每个 horizon baseline 的 advantage 比较来做 chunk size selection。

| 方法 | 关键对象 | 主要贡献 | 主要限制 |
| --- | --- | --- | --- |
| Q-chunking | $Q_K(s,a_{t:t+K-1})$ | chunk-level TD 与 offline-to-online exploration | 固定 $K$，open-loop 反应性下降 |
| AC3 | continuous action chunks | 稳定机器人 sparse reward 操作 | 依赖成功轨迹和任务设置 |
| T-SAC | sequence-conditioned critic | critic 侧建模短轨迹段 | 不直接解决 actor chunk policy |
| ACSAC | adaptive chunk size | 根据状态选择 chunk size | 额外 critic/Transformer 复杂度 |
| AQC | advantage-based chunk selection | 缓解 horizon 比较偏差 | 仍依赖多 horizon 估计质量 |

### 6.3 MAC：action chunks 进入 model-based offline RL

[MAC](https://openreview.net/forum?id=WXGb9unEHo) 把 action chunking 引入 offline model-based RL。传统 model-based value expansion 用模型 rollout 若干步，rollout 长度越长，bootstrap bias 越小，但 compounding model error 越大。MAC 用 action-chunk model 从一段动作直接预测未来状态，以减少逐步模型误差累积；同时通过从 expressive behavioral action-chunk policy 中 rejection sampling，降低 model exploitation 的 OOD action 风险。

这说明 action chunking 不只是 actor-critic 技巧，也开始影响 world model 和 offline model-based pipeline。

Action chunking 的定位可以总结为：它比传统 HRL 更工程友好，因为不需要学习 option termination，也不需要显式 subgoal semantics；但它也比 option/skill 更少闭环结构，过长 chunk 可能在 contact-rich 或反馈敏感阶段出问题。

从算法设计上，chunk size $K$ 是最核心的超参数。$K$ 太小，退化回单步控制；$K$ 太大，策略反应慢、动作维度高、离线数据覆盖变稀。实际系统里常见的折中是：在行为克隆或离线阶段用 chunk 捕捉 demonstration 的连贯性，在在线阶段加入 adaptive boundary、replanning、MPC 或 low-level feedback，让 chunk 不至于完全 open-loop。

## 7. 世界模型与规划：短程模型 + 长期价值

长期收益任务中，model-based RL 的吸引力很明显：如果真实环境交互昂贵，可以先学习世界模型，再在 imagined trajectories 中训练策略或规划。

### 7.1 DreamerV3

[DreamerV3](https://arxiv.org/abs/2301.04104) 学习 latent world model，并在 imagined future scenarios 中优化 actor-critic。它报告用单一配置覆盖大量任务，并在 Minecraft 中从零开始采集 diamond。这个结果对长期收益 RL 很有启发：agent 不一定只在真实环境里等待稀疏奖励，可以通过模型想象提高样本效率。

但 world model 的限制也非常明确。长 rollout 会产生 compounding error：

$$
\hat s_{t+k}=f_{\theta}^{(k)}(s_t,a_{t:t+k-1}),
$$

当 $k$ 增大时，模型误差会不断累积，最终 imagined trajectory 可能偏离真实环境。

### 7.2 TD-MPC2

[TD-MPC2](https://arxiv.org/abs/2310.16828) 代表另一种实用路线：在 learned latent space 里做 local trajectory optimization，并结合 terminal value function。短期由模型做 MPC，长期由 value function bootstrap：

$$
\max_{a_{t:t+H-1}}
\mathbb{E}
\left[
\sum_{i=0}^{H-1}\gamma^i r_{t+i}
+
\gamma^H V(z_{t+H})
\right].
$$

这避免了精确预测整个长未来的要求。它更像“短程规划 + 长期价值估计”，而不是长程 open-loop rollout。

### 7.3 世界模型与 action chunking 合流

MAC 进一步说明，world model 和 temporal abstraction 正在合流。模型不一定只预测单步转移 $s_{t+1}$，也可以围绕 action chunks 预测较远未来：

$$
\hat s_{t+K}=f_{\theta}(s_t,a_{t:t+K-1}).
$$

这在长 horizon offline RL 中尤其有价值，因为离线数据里往往存在 temporally coherent behavior。模型如果只看单步动作，容易在长 rollout 中积累误差；如果建模 chunk-level transition，可能更稳定。

因此，长期收益任务中更可靠的 model-based 模式通常是：

```text
短程模型规划
+ 长期 value bootstrap
+ temporal abstraction
+ OOD action control
```

而不是盲目追求无限长世界模型预测。

可以把 world model 在长期收益里的角色分成三层。

第一层是 **representation model**：学习一个 latent state $z_t$，让历史 observation、动作和任务相关信息被压缩到可预测空间。POMDP 中这一步尤其重要，因为原始 observation 不是 Markov state。

第二层是 **short-horizon predictor**：预测接下来几步或一个 chunk 后的 latent state、reward、termination。这里追求的是局部可靠，而不是完美模拟整个未来。

第三层是 **planner / value interface**：用模型生成候选短轨迹，再用 value function 估计轨迹尾部长期价值。TD-MPC2 这类方法的实用性就在于它不要求模型承担所有长期预测。

这也解释了为什么 model-based RL 和 offline RL 容易互相吸引。离线数据足以训练模型，但 offline model-based 方法容易被 planner 利用模型漏洞。长期任务会放大这个问题：短期小误差经过多步规划后可能变成虚假的高价值轨迹。因此，MAC 这样的工作才会强调 behavioral chunk policy、rejection sampling 和 OOD action control。

## 8. Offline 与 Offline-to-Online RL：长期收益工程落地的主路线

真实长期收益任务通常不能让 agent 从零在线乱探索。机器人会损坏设备，网页 agent 会触发副作用，推荐系统会影响用户体验，企业 agent 可能修改真实数据。因此 offline 或 offline-to-online 往往是默认前提。

### 8.1 Offline RL 基础

[D4RL](https://arxiv.org/abs/2004.07219) 把 offline RL benchmark 系统化，强调离线数据可能来自 hand-designed controllers、human demonstrations、random policies、多任务数据或 mixture policies。它指出普通 online RL benchmark 不能充分评估 dataset coverage、distribution shift 和 stitching 能力。

[CQL](https://arxiv.org/abs/2006.04779) 的核心是 conservative Q-function。offline RL 中，policy 可能选择数据集外动作，critic 对这些动作的 value 会过估计。CQL 通过压低 unseen actions 的 Q 值，让 learned policy value 成为更保守的 lower bound。

[IQL](https://arxiv.org/abs/2110.06169) 避免直接评价 dataset 外动作，通过 expectile value regression 和 advantage-weighted behavioral cloning 做隐式策略改进。它的实用价值在于绕开 explicit out-of-distribution action maximization。

[TD3+BC](https://arxiv.org/abs/2106.06860) 展示了 minimalist route：在 TD3 上加入 behavior cloning 项和状态归一化，就能在 D4RL 上取得强结果。这说明 offline RL 的很多性能来自非常朴素但有效的 in-distribution regularization。

这几类方法可以用一个共同原则概括：offline RL 不敢无约束地做

$$
\max_a Q(s,a),
$$

因为 $\arg\max$ 很可能落在数据集没有覆盖的动作上。CQL 选择让 critic 对 OOD action 保守；IQL 选择尽量不显式查询 OOD action；TD3+BC 选择在 actor update 里把策略拉回行为数据附近。

CQL 的直觉可写成：

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

它惩罚所有动作上的高 Q，同时相对保留数据动作的 Q。连续动作中实现会用采样近似。

IQL 的核心是先用 expectile regression 学价值：

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

再用 advantage-weighted regression 更新策略：

$$
\max_{\pi}
\mathbb{E}_{(s,a)\sim \mathcal{D}}
\left[
\exp(\beta A(s,a))\log \pi(a\mid s)
\right].
$$

TD3+BC 则把 actor 目标写成 RL objective 加 BC regularization。它提醒我们：长期收益任务里，一个简单稳定的行为先验有时比复杂的价值外推更重要。

### 8.2 Offline-to-online：从历史数据到小规模在线改进

[AWAC](https://arxiv.org/abs/2006.09359) 关注先用离线数据降低探索成本，再在线微调策略。它用 advantage-weighted behavioral cloning 风格的更新，让策略偏向数据中 advantage 高的动作。

Q-chunking 也明确定位于 offline-to-online：离线数据不是只用来保守训练，而是要转化为好的 exploratory policy。长时域任务中，这一点尤其重要。因为离线数据的优势不只是状态覆盖，也包括 action sequence distribution。一个机器人 demonstration 的价值常常不在单个动作，而在动作序列的连贯性。

MAC 也强调了这个 sequence-level 视角：用 behavioral action-chunk policy 进行 rejection sampling，可以减少模型利用 OOD action 的风险。

### 8.3 OGBench 与 long-horizon reasoning benchmark

[OGBench](https://arxiv.org/abs/2410.20092) 是更近的 offline goal-conditioned RL benchmark，任务设计关注 stitching、long-horizon reasoning、高维输入和随机性。这标志着 benchmark 也在从普通 locomotion return 转向更能暴露长期推理能力的设置。

Offline/offline-to-online RL 的统一结论是：

```text
offline prior 能降低探索成本，
但 dataset coverage 决定了可学行为边界；
online fine-tuning 能突破离线数据限制，
但必须控制安全、分布偏移和 reward hacking。
```

对于长期收益 RL，数据不是简单的 replay buffer，而是行为先验、轨迹结构和探索边界。

更具体地说，长时域 offline 数据至少有三种覆盖概念。

**State coverage**：数据是否包含关键状态和中间阶段。如果从未出现“钥匙已拿到”的状态，后续开门策略很难被学到。

**Action coverage**：关键状态附近是否有足够动作变化。只有一个 demonstration 可能告诉 agent 怎么做，却不足以告诉 critic 其他动作为什么差。

**Sequence coverage**：数据是否包含完整或可拼接的成功片段。长时域任务中，单步覆盖不等于序列覆盖。很多失败不是某一步没见过，而是多段技能无法在正确顺序上拼起来。

这也是 OGBench、Q-chunking、MAC 这些工作重要的原因：它们把 benchmark 和算法关注点从单步 $(s,a)$ 分布推进到 goal-conditioned stitching 和 action sequence distribution。对于 offline-to-online，安全在线改进的核心不是“能否探索”，而是“能否在不破坏已有能力和安全边界的情况下探索”。

## 9. LLM Agent RL：长时域问题进入软件环境

2024 之后，长期收益 RL 的一个新场景快速变热：LLM agents。这里的动作不再只是连续控制或离散游戏动作，而是 token、tool call、API call、网页点击、代码修改、数据库查询、文件操作等。一个 episode 可能需要几十轮甚至上百步，最终才知道是否完成任务。

这和机器人任务非常相似：最终 reward 稀疏，中间动作多，状态部分可观测，失败路径复杂，训练成本高。但 LLM agent 也有特殊优势：可以引入 verifier、tool feedback、process reward、LLM-generated subgoal 和语言化 hindsight reasoning。

把 LLM agent 写成 RL 问题时，状态和动作通常不是单一张量，而是一组结构化事件。

observation $o_t$ 通常包含：用户请求、对话历史、工具输出、环境状态快照、错误信息、权限状态和任务上下文。动作 $a_t$ 则可能落在多个层级：token generation、tool call、browser action、API call、code edit 或 final answer。

轨迹可以写成：

$$
\tau =
(o_0,a_0,y_0,o_1,a_1,y_1,\dots,o_T,R),
$$

其中 $y_t$ 是工具返回、错误信息、网页 DOM、测试输出或 verifier feedback。这个形式说明了 LLM Agent RL 的额外难点：observation 是非平稳文本和工具状态，action 有离散 token 和结构化工具调用两种层次，reward 常常由外部 verifier 在终点给出。

因此，一个可训练的 agent RL 系统通常需要四个接口。

| 接口 | 作用 | 典型问题 |
| --- | --- | --- |
| rollout interface | 记录每个 turn、tool call、observation、latency、错误 | 动态 workflow 很难标准化 |
| reward interface | 把 outcome、verifier、human feedback 转成 reward | reward hacking 和 judge 偏差 |
| credit interface | 把 episode reward 分配到 step、turn 或 token | 长轨迹中关键步骤稀疏 |
| safety interface | sandbox、权限、回滚、side-effect control | 在线探索可能影响真实系统 |

### 9.1 LOOP：在 stateful digital environments 中训练 LLM agent

Apple 的 [LOOP](https://arxiv.org/abs/2502.01600) 把 interactive digital agents 形式化为 POMDP，并提出 memory-efficient PPO 变体。LOOP 不使用 value network，只保留一份 LLM，因此显存需求接近普通 fine-tuning。论文报告 32B LOOP agent 在 AppWorld 上超过更大的 OpenAI o1 agent 9 个百分点。

更重要的不是单个 benchmark 数字，而是它证明了一个方向：LLM agent 可以在 stateful digital environments 中通过 RL 直接优化交互行为，而不只是靠 SFT、prompt engineering 或 rejection sampling。

### 9.2 AgentGym-RL 与训练系统标准化

[AgentGym-RL](https://openreview.net/forum?id=ZgCCDwcGwn) 是 ICLR 2026 Oral，提出面向 LLM agents 的统一开源 RL 框架，并提出 ScalingInter-RL staged training 来稳定 long-horizon RL training。

这类工作的重要性在系统层面：LLM agent RL 的难点不是单个 loss，而是环境接口、rollout、失败恢复、reward、并行采样、trajectory logging、tool sandbox 和训练框架要同时成立。长期收益 RL 在这里变成了系统工程问题。

### 9.3 WebRL：网页 agent 的 online curriculum

[WebRL](https://arxiv.org/abs/2411.02337) 面向网页 agent，提出 self-evolving online curriculum RL。它处理三个问题：训练任务稀缺、反馈稀疏、online learning 中 policy distribution drift。报告中 Llama-3.1-8B 和 GLM-4-9B 在 WebArena-Lite 上的成功率显著提升。

WebRL 和前面的 curriculum/exploration 主题完全呼应。区别只是环境从机器人或游戏变成网页交互，reward 从物理目标变成任务完成 verifier。

### 9.4 Agent Lightning：解耦 agent 执行和 RL 训练

[Agent Lightning](https://arxiv.org/abs/2508.03680) 从系统角度提出“任何 agent 都可接入 RL”的框架。它把 agent execution 和 training 解耦，定义统一数据接口，并用 hierarchical RL 和 credit assignment module 把 trajectory-level return 分解成 transition。

它的意义在于把长期收益 RL 的核心问题工程化：不是要求 agent 框架重写成某个 RL loop，而是把已有 agent 的执行轨迹转化为可训练数据。这对真实工具调用 agent 很关键，因为真实 agent 往往包含动态 workflow、多 agent 协作、检索、代码执行和外部 API。

### 9.5 Process reward 与 stepwise progress

[AgentPRM](https://arxiv.org/abs/2502.10325) 提出 Agent Process Reward Models，用 Monte Carlo rollouts 计算 reward targets，并以轻量 actor-critic 方式训练 LLM agents。它关注的是 outcome reward 太稀疏时如何构造 process-level feedback。

[SPA](https://arxiv.org/abs/2505.20732) 提出 Stepwise Progress Attribution，把 agentic task 看成由多个步骤累积 progress 完成，并基于 stepwise progress 产生中间 reward。

LLM Agent RL 的核心问题和传统 RL 一样，但 credit 粒度更细：token、segment、step、turn、tool call 都可能成为训练单位。真正困难的是 verifier 与 reward model 是否可靠。长期任务给了 agent 更多步骤，也给了 reward hacking 更多空间。

从传统 RL 的角度看，LLM Agent RL 的新意不在“长期”二字，而在训练对象从固定环境动作扩展到复杂软件行为。网页点击失败、API 参数错、代码测试不过、文件状态被污染、工具超时，都会成为 trajectory 的一部分。一个 agent 训练框架如果只保存最终文本和最终 reward，就丢掉了最重要的 credit assignment 信息。

所以，Agent Lightning 这类“执行和训练解耦”的工作很关键。它们把现有 agent 框架产生的 execution trace 转换为 RL transition，使得不同 agent、不同工具、不同环境可以共享训练管线。这个方向和传统 distributed RL 的 actor-learner 分离类似，但多了工具协议、权限、安全和结构化日志。

## 10. Reward shaping、过程奖励与 verifier

长期收益任务里，最直接的工程手段是把 sparse terminal reward 变成 dense reward。但 reward shaping 有风险：错误 shaping 会改变最优策略。

[Ng, Harada and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) 的 potential-based reward shaping 结果说明，满足以下形式的 shaping term 可以保持 policy invariance：

$$
F(s,a,s')=\gamma \Phi(s')-\Phi(s).
$$

因为沿轨迹求和时会产生 telescoping effect，最优策略不被改变。直觉上，potential $\Phi$ 可以奖励“朝目标进展”，但不能偷偷改变任务偏好。

现代长期收益 RL 中，reward shaping 出现三种新形态。

第一是 **subgoal-based shaping**：根据子目标进展给奖励。例如 STO-RL 用 LLM 生成 temporally ordered subgoals，再用 potential-based shaping 把 sparse reward 转成有序 dense signal。

第二是 **learned reward model**：[Human Preferences RL](https://arxiv.org/abs/1706.03741) 用人类对 trajectory segment 的偏好训练 reward model，是 learned reward 的经典起点。现代 agent 任务中，偏好数据、轨迹比较、LLM-as-judge 和 verifier 都可以成为 reward model 来源。

第三是 **process reward**：给中间步骤、tool call、reasoning step 打分。例如 AgentPRM 和 SPA 都试图把 outcome-level supervision 分配到过程层。

这条线的最大风险是 reward hacking。长期任务的 action sequence 更长，agent 有更多机会利用 proxy reward 的漏洞。一个网页 agent 可能学会触发 verifier 的关键词而不完成任务；一个工具调用 agent 可能通过无意义调用制造 progress signal；一个机器人 agent 可能追逐 shaped reward 而不达成真实目标。

因此，process reward 不能只看密度，还要看三点：是否和最终任务一致；是否可以被简单 hack；是否在 OOD 状态下仍然可靠。

PBRS 的特殊地位在于它有明确的不改变最优策略条件。沿一条轨迹累加 shaping term：

$$
\sum_{t=0}^{T-1}\gamma^t
\left(
\gamma \Phi(s_{t+1})-\Phi(s_t)
\right)
=
-\Phi(s_0)+\gamma^T\Phi(s_T).
$$

在固定初始状态和适当终止条件下，它只改变回报的常数项或终止 potential，不改变动作排序。相比之下，learned process reward 往往没有这个 telescoping guarantee。它也许更强，因为可以学复杂过程质量；也更危险，因为它可能把“看起来像进展”的行为奖励成真正目标。

LLM agent 里常见的 verifier 也要分层理解。

| verifier 类型 | 信号粒度 | 优点 | 风险 |
| --- | --- | --- | --- |
| unit test / exact checker | episode 或 step | 客观、可自动化 | 覆盖不全，可能被过拟合 |
| environment success flag | episode | 与任务直接相关 | 稀疏，诊断信息少 |
| rule-based process checker | step / tool call | 便宜、可解释 | 规则错配，容易被钻空子 |
| learned reward model | segment / trajectory | 能覆盖主观质量 | 偏差、OOD 不稳定 |
| LLM-as-judge | step / final | 灵活，适合开放任务 | judge drift、提示敏感、可被诱导 |

长期收益训练中，最稳的做法通常不是只依赖一种 reward，而是分层组合：硬约束负责安全和格式，verifier 负责最终成功，process reward 负责中间进展，人工或离线评估负责抽样校准。

## 11. Average Reward 与 continuing tasks：另一种“长期收益”

很多真实长期收益问题没有自然 episode。例如推荐系统优化长期留存、网络调度优化长期吞吐、能源系统优化长期效率、企业 agent 持续处理任务流。此时 discounted episodic return 可能只是工程近似。

Average reward objective 写作：

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

与 discounted value 不同，average reward 更直接描述 steady-state performance。对应的 differential value function 为：

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

[On-Policy Deep Reinforcement Learning for the Average-Reward Criterion](https://arxiv.org/abs/2106.07329) 指出，discounted return 下的 policy improvement bound 不一定适合 average-reward setting，并提出 average-reward TRPO 类方法。[Feasible Q-Learning for Average Reward Reinforcement Learning](https://proceedings.mlr.press/v238/jin24b.html) 则关注更可行的 average-reward Q-learning 理论。

[RVI-SAC](https://arxiv.org/abs/2408.01972) 这类工作说明，现代 deep RL 正在重新关注 continuing tasks。原因很现实：很多业务长期指标不是 finite episode 成功率，而是长期平均效用。

所以，如果“长期收益”指长期用户价值或长期系统效用，就不能只套 sparse terminal reward 的工具箱。需要额外考虑 off-policy evaluation、stationarity、mixing、safe policy improvement 和 delayed business metrics。

discounted objective 和 average-reward objective 的差别可以用一个简单例子说明。如果一个策略前 100 步收益很高，但之后导致系统进入低收益稳态；另一个策略前期收益较低，但长期稳态更好。短 horizon 或较小 $\gamma$ 的 discounted return 可能偏好前者，而 average reward 会偏好后者。

在业务系统中，这类差异很常见：推荐系统中过度点击诱导可能短期提升 CTR，却长期降低留存；运维调度中过度压榨资源可能短期降低成本，却提高未来故障率；agent 自动化中过度激进的操作可能快速完成当前任务，却积累不可见风险。

continuing tasks 还要求更严格的评估协议。offline evaluation 不能只看单 episode return，需要估计策略改变后的长期分布；online A/B 不能只看短期指标，需要 guardrail metrics；策略更新不能只追求 expected return，还要控制 catastrophic side effects。换句话说，这里的“长期收益”已经从算法估计问题扩展成决策系统治理问题。

## 12. 统一技术图谱

下面这张表把主要路线压缩到同一视角。

| 方向 | 代表论文 / 方法 | 解决的核心问题 | 主要局限 | 适合场景 |
| --- | --- | --- | --- | --- |
| 多步 TD / GAE | TD($\lambda$), GAE, Retrace, V-trace, IMPALA | value propagation, bias-variance | 长 horizon 仍慢，探索无帮助 | actor-critic 基础训练 |
| 显式 credit assignment | RUDDER, TVT, Synthetic Returns, COCOA, HCAPO | 把远期 reward 分给关键动作 | 贡献识别难，依赖模型/记忆/反事实估计 | delayed reward, LLM agent |
| 稀疏奖励探索 | HER, Go-Explore, RND, ICM, Skew-Fit, DIAYN, DISCOVER | 找到成功轨迹或有用中间状态 | 不直接解决 reward 分配 | hard exploration, GCRL |
| HRL / subgoal | Options, Option-Critic, FuN, HIRO, Guider, RD-HRL, STO-RL | 缩短有效 horizon | 子目标质量和训练稳定性 | 结构化任务、机器人、offline HRL |
| Action chunking | Q-chunking, AC3, T-SAC, ACSAC, AQC, MAC | 时间抽象、coherent exploration、chunk-level backup | chunk size、open-loop 反应性 | manipulation, offline-to-online |
| World model | DreamerV3, TD-MPC2, MAC | 样本效率、短程规划 | compounding error, OOD exploitation | 连续控制、数据昂贵任务 |
| Offline / off2on RL | D4RL, CQL, IQL, TD3+BC, AWAC, OGBench | 利用历史数据，降低探索风险 | 数据覆盖不足、分布偏移 | 机器人、推荐、企业 agent |
| LLM Agent RL | LOOP, WebRL, AgentGym-RL, Agent Lightning | 多轮交互、工具调用、稀疏成功信号 | reward/verifier、环境成本、安全性 | 网页、API、软件 agent |
| Reward shaping / PRM | PBRS, preference RL, AgentPRM, SPA | 生成中间训练信号 | reward hacking、proxy 错配 | verifier 可用的任务 |
| Average reward RL | ATRPO/APO, RVI-SAC, feasible Q-learning | continuing long-run objective | 理论和 deep RL 工程仍不成熟 | 长期业务指标、持续系统 |

这张表也说明，长期收益 RL 不是一条线性阶梯，不存在“先用 TD，再换 RUDDER，再换 HRL，再换 Agentic RL”的固定路线。更合理的理解是模块组合：

```text
数据来源：online / offline / offline-to-online / human demonstration
动作粒度：primitive action / chunk / option / subgoal / tool call
奖励粒度：terminal / step / segment / process / verifier
环境模型：model-free / short-horizon model / latent planner
目标形式：discounted episodic / finite-horizon success / average reward
安全机制：conservative value / sandbox / rollback / guardrail / OPE
```

真实系统往往需要在每一行选一个配置。例如，一个网页购物 agent 可能是 offline-to-online 数据来源、tool-call 动作粒度、verifier + process reward、model-free PPO、finite-horizon success objective，并带 sandbox 和权限控制；一个机器人装配系统可能是 demonstration + online fine-tuning、action chunk + low-level controller、sparse success + shaped contact reward、短程 world model、finite horizon objective，并带物理安全约束。

## 13. 研究趋势

### 趋势一：从单步动作转向时间扩展动作

Options、skills、subgoals、macro-actions、action chunks 都属于这个趋势。区别在于抽象对象不同：

| 抽象 | 形式 | 闭环性 | 典型问题 |
| --- | --- | --- | --- |
| Option | 内部策略 + 终止条件 | 强 | discovery 和 termination 难 |
| Action chunk | 动作序列 $a_{t:t+K-1}$ | 弱到中等 | 固定 $K$ 与反应性 |
| Subgoal | 高层状态目标 | 中等 | 可达性与 reward 一致性 |
| Skill | 可复用行为 | 强 | 学习成本和组合性 |

Action chunking 火起来，是因为它工程上更轻：不需要显式 option discovery，不需要 termination condition，也贴合 imitation/offline data 中的动作序列。

### 趋势二：从 episode-level reward 转向 step/chunk/turn-level credit

RUDDER、Synthetic Returns、HCAPO、AgentPRM、SPA 都体现了同一方向：终局成功/失败信号太粗，必须把 reward 或 advantage 分配到更细粒度。

传统 RL 里这个粒度常是 state-action；机器人里可以是 chunk；LLM agent 里可以是 tool call、turn 或 reasoning segment。

### 趋势三：offline prior 成为长期收益 RL 的默认前提

长时域稀疏奖励任务从零在线探索太贵。因此 D4RL、OGBench、offline HRL、Q-chunking、AWAC、MAC 都强调历史数据、demonstration 或 behavior prior。

未来更常见的模式可能是：

```text
offline pretraining
+ conservative / in-distribution value learning
+ temporal abstraction
+ small-scale online improvement
+ verifier / safety gate
```

### 趋势四：world model 不再追求无限长 rollout

DreamerV3 和 TD-MPC2 表明 world model 是长期收益的重要路线，但稳定方法通常避免长程 open-loop prediction。更实用的是短程模型规划、latent dynamics、terminal value bootstrap 和 OOD action control。

### 趋势五：LLM 进入两个位置

LLM 在长期收益 RL 中有两种角色。

第一，**LLM as agent**：LOOP、WebRL、AgentGym-RL、Agent Lightning 训练 LLM 直接进行多轮工具调用、网页操作、API 操作。

第二，**LLM as prior / planner / reward helper**：STO-RL 用 LLM 生成 subgoal temporal order，AgentPRM/SPA 等方向用语言或 rollout 辅助过程奖励。

这两种角色不能混为一谈。前者训练的是 agent 本身，后者把 LLM 当作结构先验或 reward helper。

## 14. 仍未解决的问题

**第一，长期 credit assignment 缺少统一评价协议。** 不同论文评估 delayed reward、memory、hard exploration、offline stitching、LLM tool use，它们都叫 long-horizon，但失败模式不同。Temporal Credit Assignment Survey 特别强调 credit 定义和诊断协议仍不统一。

**第二，子目标和 chunk 边界难以自动确定。** 固定 chunk size 在自由空间运动中有效，但在接触丰富阶段会降低反应性。ACSAC 和 AQC 正是为了解决这个问题。HRL 中 option termination 和 subgoal boundary 也有同样困难。

**第三，offline data 既是优势，也是偏差来源。** 离线数据提供探索先验，但也限制 policy 的可达行为。长时域任务中，不只是单步 action distribution 重要，action sequence distribution 也重要。

**第四，process reward 容易引入错误目标。** PBRS 有 policy invariance 保证，但 learned PRM、LLM-generated reward、tool-call reward 不一定有。越长的 episode，agent 越可能利用 proxy 漏洞。

**第五，LLM Agent RL 的成本和安全边界仍然早期。** 环境交互昂贵，工具调用可能有副作用，reward/verifier 可能不可靠，agent 训练系统需要 sandbox、rollback、observability 和权限控制。

## 15. 场景选型建议

如果任务是 **机器人长时域 sparse reward manipulation**，优先考虑 offline demonstrations、action chunking、offline-to-online RL、world model 或 HRL。Q-chunking、AC3、AQC、MAC 这类方法更贴近工程现实。

如果任务是 **goal-conditioned sparse reward**，HER、curriculum、Skew-Fit、DISCOVER 和 offline GCRL benchmark 是更合适的阅读入口。核心问题通常是目标覆盖和可达性。

如果任务是 **需要记忆的 delayed reward**，RUDDER、TVT、Synthetic Returns、COCOA 更直接。它们关注的是关键事件识别，而不是单纯探索。

如果任务是 **网页、API、代码、企业流程 agent**，要把传统 long-horizon RL 映射到 agent 轨迹：turn-level credit、tool-call reward、verifier、process reward、环境 sandbox、rollout 成本和安全边界都要一起设计。LOOP、WebRL、AgentGym-RL、HCAPO、Agent Lightning、AgentPRM 是更相关的路线。

如果任务是 **长期业务指标**，不要默认使用 finite-horizon discounted return。average reward、off-policy evaluation、safe policy improvement 和 delayed metric attribution 可能更重要。

更落地的架构选择可以按下面思路做。

| 场景信号 | 首选组合 | 不建议优先做 |
| --- | --- | --- |
| 成功样本极少但可模拟 | curriculum + exploration + hindsight relabeling | 直接上复杂 critic |
| 有大量专家轨迹 | behavior cloning + offline RL + action chunking | 从零 online RL |
| 环境交互昂贵但可建模 | latent world model + short MPC + value bootstrap | 超长 open-loop rollout |
| 工具调用 agent 有 verifier | PPO/GRPO 类训练 + process reward + credit assignment trace | 只用最终成功率更新整段文本 |
| 任务强安全约束 | offline conservative learning + guarded online improvement | 无约束探索 |
| 长期业务指标延迟数天 | OPE + causal metric attribution + safe policy rollout | 把业务指标硬塞进短 episode reward |

## 16. 推荐阅读顺序

第一层先读基础：GAE、Retrace、IMPALA/V-trace、PPO，以及 Temporal Credit Assignment Survey。目标是理解 value propagation、advantage estimation 和 off-policy correction。

第二层读显式 credit assignment：RUDDER、TVT、Synthetic Returns、COCOA。目标是理解 reward redistribution、memory-based credit 和 counterfactual contribution。

第三层读探索和目标条件 RL：HER、Go-Explore、RND/ICM、Skew-Fit、DIAYN、Automatic Curriculum Learning、DISCOVER。目标是理解成功轨迹如何产生。

第四层读时间抽象：Options、Option-Critic、FeUdal Networks、HIRO、Guider、RD-HRL、STO-RL。目标是理解 subgoal、skill、option 和 offline HRL 的结构性取舍。

第五层读 2025-2026 前沿：Q-chunking、AC3、T-SAC、ACSAC、AQC、MAC、DreamerV3、TD-MPC2。目标是理解 action chunking、world model 和 offline-to-online 如何合流。

第六层读 Agentic RL：LOOP、WebRL、AgentGym-RL、HCAPO、Agent Lightning、AgentPRM、SPA。目标是理解长时域 credit assignment 如何进入 LLM agent、tool use 和多轮交互。

## 17. 结论

长期收益强化学习的进展，不是某一个算法突然解决了所有问题。更准确的图景是，领域共识正在从：

```text
单步动作
+ episode-level sparse reward
+ 1-step TD
```

转向：

```text
时间抽象
+ 多粒度 credit assignment
+ offline prior
+ curriculum exploration
+ world model / planning
+ process reward / verifier
+ safe online improvement
```

也就是说，长期收益 RL 的前沿已经从“如何写一个更好的 Bellman backup”扩展成“如何重新组织轨迹、奖励、动作粒度、数据来源和训练系统”。

对于传统机器人和控制任务，当前最活跃的是 offline-to-online、action chunking 和 world model 的结合。对于 LLM agent 和多轮软件交互任务，当前最活跃的是 process reward、hindsight credit assignment 和 agent RL framework。对于真实长期业务指标，不能只看 discounted episodic return，还要考虑 average reward、off-policy evaluation、safe online improvement 和 reward hacking 控制。

最终，长期收益 RL 不是一个单点算法栈，而是一套关于时间的建模方式：把长未来拆成可学习的信号，把粗粒度终局反馈拆成可归因的中间贡献，把单步动作组织成可复用的时间结构，再把离线数据、在线探索、验证器和安全约束放进同一个训练闭环。

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

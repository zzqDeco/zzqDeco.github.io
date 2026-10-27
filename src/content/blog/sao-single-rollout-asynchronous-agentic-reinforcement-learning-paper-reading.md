---
title: "SAO 论文精读：单轨迹异步优化、双边重要性采样与 Agentic RL 稳定训练"
description: "精读 SAO 如何通过单轨迹异步采样、双边 Token 级重要性门控、快速价值模型更新和 Skip-Observation GAE 稳定训练长时域推理与编码智能体"
pubDate: "2026-07-15T09:46:03+08:00"
updatedDate: "2026-07-15T09:46:03+08:00"
tags:
  - "Deep Reading"
  - "Paper Reading"
  - "Reinforcement Learning"
  - "Agentic RL"
  - "Asynchronous RL"
  - "LLM Post-Training"
  - "Training Systems"
draft: false
---

大语言模型强化学习正在从单轮问答生成转向长时域的环境交互。数学推理 Agent 会交替撰写自然语言、调用 Python 解释器、读取执行结果并继续推导；编码 Agent 会浏览仓库、修改文件、运行测试并分析报错，单条完整轨迹可能持续数十分钟甚至数小时。在此场景下，强化学习系统不仅需要设计合理的优化目标，还必须应对不同轨迹完成时间差异巨大的系统瓶颈。

同步训练通常先收集一整批 rollout，再统一计算奖励与更新参数。只要批次内存在一条耗时极长的慢轨迹（straggler），已完成的轨迹就只能原地等待。Group Relative Policy Optimization（GRPO）进一步要求同一个 prompt 的多条 response 组成完整 group，才能计算组内相对 advantage。这种设计在同步短序列推理中省去了 Critic 显存与计算开销，但在长时域异步 Agentic RL 中却构成了显著的组内同步屏障。

**Single-Rollout Asynchronous Optimization（SAO）** 从这一结构矛盾出发：面对反馈天然按单条轨迹异步到达的长时域任务，直接放弃 prompt 内的采样组，让每个 prompt 仅生成一条 rollout，轨迹一经完成即可进入训练队列。为了填补移除 GRPO 组内 baseline 后的方差控制空白，SAO 重新引入 value model，并通过高频 Critic 更新、冻结 Attention 参数的 Critic 训练以及跨环境观测的 Skip-Observation GAE，保障单轨迹 Actor-Critic 的稳定性；同时，利用 rollout 阶段记录的 Token log-probability 直接计算重要性权重，将偏离信任区间的 Token 从策略梯度中完全屏蔽。

SAO 的核心设计将 **异步调度、行为策略日志、优势估计、价值网络稳定性与 Agent 轨迹语义** 紧密耦合在同一训练闭环中。需要提前指出的是，本文精读基于预印本 `arXiv:2607.07508v1`，原论文重点验证了算法收敛性与任务准确率，但未披露端到端吞吐提升、GPU 利用率或 policy lag 分布等系统效率指标；文中关于“已成功部署至 GLM-5.2（750B-A40B）”的表述属于生产应用声明，并未附带 750B 规模下的公开消融实验。

## 1. 问题定义、核心贡献与论文概览

### 1.1 一句话贡献与五大核心部件

SAO 的核心思路可以概括为：

> 以“单 prompt 对应单条立即可训练的 rollout”解除 GRPO 的组内同步屏障，再结合基于 rollout log-prob 的双边 Token 门控与高频稳健的 value model，实现长时域异步 Agentic RL 的稳定优化。

该框架由五个相互支撑的算法部件组成：

1. **Single-rollout sampling**：每个 prompt 仅采样一条轨迹，消除等待同组其他轨迹完成的同步屏障。
2. **Direct Double-Sided Importance Sampling（DIS）**：直接对比当前策略与实际采样策略（rollout policy）的 Token 级对数概率，双边屏蔽越界样本。
3. **Faster Value Update**：每次策略更新匹配两次价值网络更新（$K=2$），使 Critic 更快追踪策略分布的漂移。
4. **Frozen-Attention Value Model**：在 RL 阶段冻结 value model 的 Attention 层，仅更新 MoE projection 参数，抑制 Critic 梯度波动。
5. **Skip-Observation Token-level GAE**：优势函数递推直接跨过环境 observation 片段，仅在模型生成的 action Token 之间传播信用。

五个部件缺一不可：仅采用 single rollout 会面临剧烈的梯度方差；仅采用 DIS 无法消除组内等待；而若只引入 Critic 却不持久化 rollout 阶段的行为概率，则无法对异步陈旧样本执行可靠的 off-policy 门控。

### 1.2 论文基本信息与适用条件

| 项目 | 内容 |
| --- | --- |
| 题名 | Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning |
| 作者 | Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong |
| 机构 | Tsinghua University（前两位作者于 Z.AI 实习期间完成） |
| arXiv | `2607.07508v1`（2026-07-08，CC BY 4.0，DOI: `10.48550/arXiv.2607.07508`） |
| 主实验模型 | Qwen3-30B-A3B-Thinking-2507 |
| 生产部署声明 | 用于开放模型 GLM-5.2（750B-A40B）的 Agentic RL 管线 |

SAO 针对的是具备以下特征的强化学习场景：

- 模型以多轮 Agent 方式与外部工具或容器环境持续交互，不同 rollout 的完成时间呈重尾分布；
- rollout 生成与参数训练在物理资源上解耦，learner 更新期间 rollout 引擎仍在使用旧版本权重生成轨迹；
- 真实交互环境通常只为单次任务执行返回一次反馈，不适合或难以对同一状态廉价克隆多个并行副本；
- 推理引擎能够准确记录每个生成 Token 在采样时刻的 log-probability，且硬件预算允许部署与维护额外的 value model。

若任务仅涉及生成长度均匀的短文本，且同一 prompt 可低成本并发采样多条响应，同步 GRPO 依然是简洁高效的方案；反之，在长尾效应显著的多轮工具调用与软件工程任务中，SAO 解决的正是“**如何让单轨迹异步 Agentic RL 稳定收敛**”这一核心问题。

### 1.3 Fig. 1：基准测试总体结果概览

![SAO benchmark results](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig01-benchmark-results.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 1, CC BY 4.0.*

Fig. 1 汇总了 AIME2025、BeyondAIME、HMMT Nov 2025、IMOAnswerBench 四项工具集成数学推理任务，以及基于 OpenHands 框架的 SWE-Bench Verified 编码任务表现。图中对比了基础/SFT 模型、GRPO 与 SAO 三组结果。

在全部五项基准上，SAO 均取得显著领先：AIME2025 从 SFT 的 `80.4` 提升至 `97.3`，BeyondAIME 从 `53.3` 提升至 `74.8`，SWE-Bench Verified 从基础模型的 `23.0%` 提升至 `29.8%`。这些数据验证了单轨迹异步 Actor-Critic 在长时域任务上的算法有效性。

## 2. 异步单轨迹动机与陈旧性来源

### 2.1 长尾轨迹（Straggler）与同步批处理屏障

假设一个训练批次包含 $B=128$ 条轨迹。在数学推理或代码修复中，简单样本可能在数十秒内结束，而复杂样本可能反复调用 Python 或执行测试长达数百轮、生成数万 Token。同步调度的单批完成时间取决于最慢的那条轨迹：

$$
T_{\text{batch}}^{\text{sync}}
\approx \max_{i\in\{1,\ldots,B\}} T_i.
$$

当轨迹耗时分布高度右偏、环境 I/O 波动剧烈或最大交互轮数（turn limit）较高时，尾部延迟会造成严重的算力空泡。异步 actor-learner 架构通过拆分批处理边界来消除等待：已完成的轨迹立即写入就绪队列（ready queue），learner 按可用样本持续更新参数，而长耗时轨迹在后台继续生成：

```text
rollout workers              learner
---------------              -------
trajectory A --done------->  replay / ready queue
trajectory B --------done->  sample -> optimize
trajectory C --done------->  publish new weights
trajectory D ------------->  continue generation
```

这种流水线重叠虽然提升了硬件利用率，但也带来了天然代价：trajectory D 由较旧的策略版本启动，当它最终完成并进入训练队列时，learner 可能已经完成了多轮梯度更新。异步训练本质上是用 **policy lag（策略滞后）与 off-policy drift（离策漂移）** 换取更低的同步等待。

### 2.2 PPO 与 GRPO：Group Baseline 的双刃剑

给定查询 $q$ 与模型生成的序列 $y=[y_1,\ldots,y_{|y|}]$，策略记作 $\pi_\theta(y\mid q)$。标准 PPO 使用当前策略与采样快照 $\pi_{\theta_{\text{old}}}$ 的 Token 概率比：

$$
r_t(\theta)=
\frac{\pi_\theta(y_t\mid q,y_{<t})}
{\pi_{\theta_{\text{old}}}(y_t\mid q,y_{<t})},
$$

并最大化裁剪代理目标（clipped surrogate objective）：

$$
\mathbb{E}\left[
\frac{1}{|y|}\sum_{t=1}^{|y|}
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

PPO 的优势估计 $\hat A_t$ 依赖价值网络 $V_\phi$。Critic 通过最小化均方误差拟合回报 $R$：

$$
\mathcal L_{\phi}^{\mathrm{VF}}
=\mathbb E\left[(V_\phi(q,y_{<t})-R)^2\right],
$$

并结合广义优势估计（GAE）平衡偏差与方差：

$$
\hat A_t^{\mathrm{GAE}}
=\sum_{l=0}^{|y|-t-1}(\gamma\lambda)^l\delta_{t+l},
\qquad
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

Critic 的优势在于其 baseline 仅依赖当前状态 $s_t$，无需同一 prompt 并发生成多条轨迹；代价则是额外的模型显存、前反向计算与 Critic 训练不稳定性。

GRPO 为省去 Critic，对每个 prompt $q$ 采样 $G$ 条输出 $\{y_1,\ldots,y_G\}$ 并获取奖励 $\{R_1,\ldots,R_G\}$，利用组内均值与标准差构造优势：

$$
\hat A_{i,t}
=\frac{R_i-\mu_R}{\sigma_R},
\qquad
\mu_R=\frac{1}{G}\sum_{j=1}^{G}R_j,
\qquad
\sigma_R=
\sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j-\mu_R)^2+\epsilon_{\text{stab}}}.
$$

问题在于，$\mu_R$ 与 $\sigma_R$ 必须等待组内全部 $G$ 条轨迹返回奖励后才能计算。哪怕一组 8 条轨迹中有 7 条早已结束，只要最后一条卡在工具执行或长链推理中，前 7 条样本就无法构成完整的 GRPO 训练组。这种由优势定义本身引入的 group dependency，与异步流式到达的轨迹语义直接冲突。SAO 因此选择回归 Actor-Critic 路线，以单轨迹采样彻底移除组内屏障。

### 2.3 三种策略版本（Current / Old / Rollout）与 Policy Lag

在异步强化学习中，必须严格区分三个策略概念：

| 名称 | 记号 | 作用 |
| --- | --- | --- |
| Current policy | $\pi_\theta$ | learner 当前正在执行梯度更新的目标模型 |
| Old policy | $\pi_{\theta_{\text{old}}}$ | 传统 PPO 优化周期开始时固定的参考快照 |
| Rollout policy | $\pi_{\text{rollout}}$ | 采样器生成具体某个 Token 时实际使用的行为策略 |

同步 PPO 中，$\pi_{\theta_{\text{old}}}$ 与 $\pi_{\text{rollout}}$ 通常重合；但在异步系统中，rollout worker 的权重同步存在延迟，一条长轨迹甚至可能跨越 learner 的多个更新步。若用版本差粗略度量滞后：

$$
\operatorname{lag}(\tau)=v_{\text{learner-at-consume}}-v_{\text{rollout}},
$$

版本号之差仅是外部代理指标。单步更新幅度、参数规模与具体上下文对概率分布的影响差异巨大，真正决定离策程度的是 Token 级的对数概率偏移：

$$
\Delta\ell_t
=\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{rollout}}(a_t\mid s_t).
$$

SAO 不再维护庞大的历史 $\pi_{\theta_{\text{old}}}$ 检查点集合，而是直接持久化生成时的 $\log\pi_{\text{rollout}}(a_t\mid s_t)$，从 Token 概率比直接判断样本是否可信。

### 2.4 Fig. 2：单轨迹异步采样与 Batch 聚合语义

![SAO single-rollout overview](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig02-single-rollout-overview.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 2, CC BY 4.0.*

Fig. 2 直观对比了 GRPO 与 SAO 的调度差异：上半部分的 GRPO 即使在不同轨迹按不同顺序完成时，仍需等待同属一个 prompt group 的兄弟样本凑齐；下半部分的 SAO 则允许轨迹严格按照 `9, 8, ..., 3, 2, 1` 的实际完成顺序流式进入就绪队列，右侧则通过 rollout policy 与 current policy 的比值信任区间过滤越界 Token。

需要强调的是，**Single Rollout（group size = 1）绝不等于训练 Batch Size 为 1**。论文主实验的配置为：

- **SAO**：policy batch size 为 `128`，group size 为 `1`（即每次更新聚合 128 个不同 prompt 的单条已完成轨迹）；
- **GRPO 基线**：每批包含 `16` 个 prompt，每个 prompt 采样 `8` 条轨迹，总样本数同样为 `128`。

因此，single rollout 的准确含义是 **one rollout per prompt**。learner 依然从就绪队列中提取 128 条来自不同 prompt 的轨迹组成全局大批次，以保障并行计算效率与梯度估计的稳定性：

```text
many prompts
  -> one rollout per prompt
  -> completion-ordered ready queue
  -> global training batch of 128 trajectories
  -> two critic updates + one policy update
```

## 3. 双边直接重要性采样（Direct Importance Sampling, DIS）

### 3.1 直接基于 Rollout Log-Prob 计算概率比

SAO 定义的 Token 级重要性比率为：

$$
r_t(\theta)
=\exp\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{rollout}}(a_t\mid s_t)
\right).
$$

其中 $\pi_{\text{rollout}}(a_t\mid s_t)$ 是推理引擎在采样动作 Token $a_t$ 时直接记录的行为概率。这种做法省去了事后加载历史模型快照重算旧概率的开销，但也对推理与训练引擎的一致性提出了严格要求：

1. `rollout_logprobs` 必须与训练侧重算时的温度（temperature）、top-$p$ 采样语义保持一致；
2. Tokenizer、chat template、工具输出拼接方式与 position ID 必须在两端严格对齐；
3. 环境返回的 observation Token 并非由策略采样生成，必须通过 mask 排除在比率计算之外。

### 3.2 DIS 双边硬门控目标函数与阈值设定

SAO 的策略优化目标写作：

$$
L(\theta)
=\hat{\mathbb E}_t\left[
f(r_t(\theta);\epsilon_l,\epsilon_h)
\hat A_t
\log\pi_\theta(a_t\mid s_t)
\right],
$$

其中双边校准函数（double-sided calibration function）定义为：

$$
f(x;\epsilon_l,\epsilon_h)=
\begin{cases}
x, & 1-\epsilon_l < x < 1+\epsilon_h,\\
0, & \text{otherwise}.
\end{cases}
$$

这与标准 PPO 的 `clip(x, 1 - eps, 1 + eps)` 存在本质区别。标准 PPO 仅将越界比率截断到边界常数，并根据优势正负决定单侧约束；而 SAO 的 $f(x;\epsilon_l,\epsilon_h)$ 对任何超出开区间 $(1-\epsilon_l, 1+\epsilon_h)$ 的 Token 直接输出 `0`。换言之，一旦某个 Token 的当前生成概率相对采样时刻偏移过大，无论其优势 $\hat A_t$ 符号如何，该 Token 的策略梯度都会被彻底置零（硬门控屏蔽）。

论文针对两类任务设定了非对称的经验阈值：

- **数学推理实验**：$\epsilon_{\text{low}}=0.3,\ \epsilon_{\text{high}}=5.0$，对应可信比率区间 $0.7 < r_t < 6.0$；
- **SWE-Bench 编码实验**：$\epsilon_{\text{low}}=0.8,\ \epsilon_{\text{high}}=3.0$，对应可信比率区间 $0.2 < r_t < 4.0$。

因此，论文所谓的“strict”指的是对上下两侧越界样本均执行梯度清零，而非信任区间本身比传统 PPO 的 $[0.8, 1.2]$ 更窄。上下界显著非对称，反映了作者对概率暴跌与概率激增采取了不同的容忍度，且具体边界需随任务领域调节。

### 3.3 偏差-方差权衡与 Log-Space 数值实现

DIS 的核心收益在于：无需保存多版本历史模型，即可在 Token 粒度精准剔除极端离策样本，防止异步陈旧轨迹引发破坏性梯度爆炸。其代价则是被屏蔽的 Token 不再提供学习信号，且由于门控条件依赖当前参数 $\theta$，训练分布会偏向“仍与当前策略接近”的局部动作。

在数值实现上，为避免低概率 Token 在 `bf16/fp16` 下直接相除发生下溢或指数爆炸，应在对数空间完成差值计算与门控判定：

$$
\Delta_t
=\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{rollout}}(a_t\mid s_t),
\qquad
\log(1-\epsilon_l) < \Delta_t < \log(1+\epsilon_h).
$$

仅对满足不等式的可信位置计算 $r_t = \exp(\Delta_t)$ 并 detach 权重后乘以 $\hat A_t \log\pi_\theta(a_t\mid s_t)$。由于推理引擎（如 SGLang/vLLM）与训练引擎（如 Megatron/FSDP）在算子融合、张量并行切分及浮点归约顺序上存在细微差异，工程落地前通常需要执行同版本重放测试（same-version replay test），确保同一权重下两端算出的 $\Delta_t$ 噪声远小于门控边界。

## 4. 价值网络稳定化与训练机制

### 4.1 单轨迹为何必须重新引入 Value Model

当每个 prompt 仅生成一条轨迹并获得单一标量奖励时，系统失去了同 prompt 内部的对照样本。若直接使用原始奖励或全局滑动均值构造优势，会把不同难度的 prompt 混为一谈：一道极难的竞赛题拿到中等分数本应属于正向激励，却可能因低于全局历史均值而被赋予负优势。

引入状态相关的价值网络 $V_\phi(s_t)$ 后，优势估计回归为：

$$
\hat A_t \approx R_t - V_\phi(s_t).
$$

Critic 能够基于当前上下文与环境反馈动态评估预期回报。然而，在失去组内统计兜底后，Actor 的更新方向完全仰赖 Critic 的预测质量；一旦 $V_\phi$ 失准或滞后，单轨迹策略梯度就会迅速崩溃。为此，SAO 从更新频率、可训练参数子集与冷启动三个维度对 Critic 进行了专门加固。

### 4.2 Faster Value Update：双时间尺度的 Critic 追踪

在异步训练中，策略 $\pi_\theta$ 每更新一步，其诱导的状态分布与回报分布都会随之改变。若 Critic 与 Actor 按 1:1 同步更新，价值网络往往滞后于当前策略，导致计算出的优势包含较大的追踪偏差。SAO 解耦了两者的更新频率：

$$
\text{for each policy update:}
\qquad
\underbrace{K\text{ critic updates}}_{K=2}
+\underbrace{1\text{ actor update}}_{1}.
$$

这种双时间尺度（two-timescale）设计让 Critic 以更快的速度逼近移动中的策略价值。在实验配置中，policy 学习率设为 $1\times10^{-6}$，value model 学习率设为 $5\times10^{-6}$，配合每步 $K=2$ 次更新，共同加快了 Critic 的收敛追踪。

### 4.3 Frozen-Attention Value Model 与 Critic 预训练冷启动

作者在实验中观察到，对 value model 进行全参数 RL 微调时，梯度范数显著偏高且剧烈震荡；进一步拆解发现，不稳定性主要源自 Full Attention 层，而 MoE 专家层相对平稳。基于此，SAO 在 RL 阶段**冻结 Critic 的全部 Attention 参数，仅优化 MoE projection 层**。

从表示学习的角度看，预训练模型的 Attention 层已具备读取长上下文与定位关键 Token 的路由能力，Critic 在 RL 阶段的核心任务是将已有隐状态映射为标量价值。冻结 Attention 相当于施加强结构正则化，防止高方差的稀疏奖励冲垮底层注意力表征。需要注意，这一设计深度绑定了主实验所用的 `Qwen3-30B-A3B` MoE 架构；若迁移至稠密（Dense）Transformer，则需探索仅微调 FFN 或低秩适配器等替代方案。

此外，论文指出 Critic 的冷启动质量至关重要：若 RL 初始阶段价值估计近乎随机，早期错误的优势信号会将 Actor 推向不可逆的退化区域。因此，SAO 强调在进行 RL 之前先扩展价值预训练语料（Scaling Value Pretraining）并配合 10 步 value warmup。不过，原论文并未公开该预训练语料的具体规模、构建方式与独立消融数据，这是复现时需自行补齐的环节。

## 5. Agent 轨迹语义与优势估计（Skip-Observation GAE）

### 5.1 Action 与 Observation 的 Token 级语义边界

在多轮工具交互中，一条完整的智能体轨迹可表示为动作与环境反馈的交替序列：

$$
\mathcal T=[a_0,o_0,a_1,o_1,\ldots],
$$

其中 $a_i$ 是模型生成的动作片段（自然语言推理、工具调用指令或代码补丁），$o_i$ 是外部环境返回的观测片段（Python 标准输出、测试报错日志或文件内容）。

尽管在送入 Transformer 时，$\mathcal T$ 被拼接为一条连续的 Token 序列，但在强化学习语义下，两类 Token 截然不同：$a_i$ 由策略采样产生，拥有明确的行为概率并承担策略梯度；$o_i$ 由外部环境决定，不应计入策略损失，也不应要求价值网络在环境输出内部逐 Token 预测状态转移。因此，训练张量必须维护严格的掩码：

```text
loss_mask:   1 for model action tokens, 0 for environment observations
value_mask:  define valid states receiving value supervision and TD backup
```

### 5.2 Skip-Observation Token-level GAE 与终止状态协议

标准 Token 级 GAE 沿着自回归序列的相邻位置逐一递推。但在 Agent 轨迹中，第 $i$ 轮动作的末尾 Token $a_{i,N}$ 之后紧跟着环境观测 $o_i$，直到 $o_i$ 结束才出现第 $i+1$ 轮动作的首个 Token $a_{i+1,0}$。如果机械地按相邻索引计算 TD 残差，就会把环境生成文本的内部转移混入优势估计。

SAO 提出 **Skip-Observation Token-level GAE**，在动作边界处直接跨过中间的 $o_i$：

$$
\hat A(a_{i,N})
=\delta_i+\gamma\lambda\hat A(a_{i+1,0}),
\qquad
\delta_i
=r_i+\gamma V(a_{i+1,0})-V(a_{i,N}).
$$

跳过 $o_i$ 并不意味着忽略环境反馈：当计算下一轮首动作的状态价值 $V(a_{i+1,0})$ 时，由于因果自注意力的前缀上下文已经包含了完整的 $o_i$，$V(a_{i+1,0})$ 自然融合了环境执行结果对后续成功率的修正。被跳过的仅仅是在 $o_i$ 内部非决策 Token 上的无效递推。

| Token 区间 | 生成主体 | 参与 Policy Loss | 参与 Value 回归 | GAE 下一递推位置 |
| --- | --- | --- | --- | --- |
| Assistant reasoning | Policy | 是 | 是 | 下一个 action Token |
| Tool call | Policy | 是 | 是 | 下一个 action Token 或下一轮 $a_{i+1,0}$ |
| Tool observation ($o_i$) | Environment | 否 | 屏蔽 | 直接跳过 |
| Final answer | Policy | 是 | 是 | Terminal（终止边界） |

在工程实现中，预处理器会为每个有效 action Token 预计算 `next_learnable_token_index`：动作片段内部指向相邻的下一个 Token，动作末尾 $a_{i,N}$ 直接指向 $a_{i+1,0}$。同时，必须区分真正的任务终止（terminal，bootstrap value 置 0）与因达到最大轮数/上下文上限或容器超时导致的截断（truncation，使用末状态价值自举或单独赋惩罚），避免将基础设施中断误标为策略逻辑失败。

### 5.3 Length-Adaptive GAE 与动作粒度对比（Token-level vs. Step-level）

考虑到长时域轨迹的长度跨度极大，SAO 沿用了 VAPO 的长度自适应 GAE（Length-Adaptive GAE），根据有效序列长度 $l$ 动态调节策略侧的衰减系数 $\lambda_{\text{policy}}$：

$$
\lambda_{\text{policy}}
=1-\frac{1}{\alpha l},
\qquad \alpha=1.5,
$$

而 Critic 训练侧则固定使用 $\lambda_{\text{critic}}=1$。轨迹越长，$\lambda_{\text{policy}}$ 越接近 1，使稀疏的终端奖励能够跨越更长的决策链条向前传播；短轨迹则采用较小的 $\lambda_{\text{policy}}$ 以抑制远期方差。

针对优势估计的粒度，论文在附录中对比了主线的 Token-level GAE 与两种按对话轮次（conversation turn）聚合的 Step-level 变体：

- **Step Average**：将第 $i$ 轮内所有 $n$ 个 Token 的价值预测取均值，即 $V(S_i)=\frac{1}{n}\sum_{j=1}^{n}v_{i,j}$；
- **Last-Token Prediction**：仅取该轮末尾 Token 的价值预测，即 $V(S_i)=v_{i,n}$。

两种 Step-level 变体均按轮次计算 $\delta_i=R_i+\gamma V(S_{i+1})-V(S_i)$，并将所得的标量优势 $\hat A_i$ 广播给该轮内的全部 Token。虽然轮次聚合平滑了局部噪声，但它抹平了同一轮长推理链内部早期假设、关键公式推导与工具调用之间的细粒度信用分配差异。后续消融实验表明，Token-level 粒度在长链数学推理中优势明显。

## 6. 核心实验结果与训练动态分析

### 6.1 实验设置与超参数配置

数学推理实验以 `Qwen3-30B-A3B-Thinking-2507` 为基座，先在由 GPT-OSS-120B 合成的 Tool-Integrated Reasoning（TIR，穿插自然语言与 Python 调用）数据上进行 3 个 epoch 的 SFT，再用该 SFT 模型同时初始化 Actor 与 Critic。SWE-Bench Verified 编码实验则直接从基座模型出发，在 OpenHands 框架下进行 RL 训练。

| 配置项 | 设定值 |
| --- | --- |
| 全局批次大小（Policy batch size） | 128 条轨迹 |
| SAO 采样组大小（Group size） | 1 rollout per prompt（共 128 个 prompt） |
| GRPO 对照组设置 | 16 prompts × 8 rollouts = 128 条轨迹 |
| 最大上下文长度 | 128K Tokens |
| 学习率配置 | Policy LR $= 1\times10^{-6}$，Value LR $= 5\times10^{-6}$（含 10 步 warmup） |
| Critic 更新频率 | 每次 Policy 更新对应 $K=2$ 次 Critic 更新 |
| DIS 门控阈值 | 推理任务 $\epsilon_l=0.3,\ \epsilon_h=5.0$；编码任务 $\epsilon_l=0.8,\ \epsilon_h=3.0$ |
| 最大交互轮数（Max turns） | 数学推理 50 轮；SWE-Bench 300 轮 |
| 采样参数 | Temperature $= 1.0$，top-$p = 1.0$ |

评估阶段，AIME2025、HMMT Nov 2025 与 IMOAnswerBench 报告 16 次独立采样评估的均值，BeyondAIME 报告 4 次评估均值。

### 6.2 数学推理（Table 1）与 SWE-Bench Verified（Table 2）主结果

![SAO math reasoning results](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table01-math-reasoning-results.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 1, CC BY 4.0.*

Table 1 核心对比行整理如下：

| 模型 / 训练策略 | AIME2025 | BeyondAIME | HMMT Nov 2025 | IMOAnswerBench |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B SFT（w/ Python） | 80.4 | 53.3 | 75.2 | 53.3 |
| GRPO（w/ Python） | 84.2 | 54.8 | 76.0 | 55.8 |
| GRPO + DIS | 93.5 | 70.8 | 84.0 | 70.0 |
| SAO（DIS only） | 94.2 | 71.5 | 86.7 | 71.3 |
| **SAO（完整版）** | **97.3** | **74.8** | **88.3** | **74.0** |

从表中可以提炼出清晰的增量证据链：

1. **标准异步 GRPO 提升有限**：由于在约 160 步时发生训练崩溃（表中取崩溃前最佳点），标准 GRPO 相比 SFT 仅提升 `0.8`～`3.8` 个百分点。
2. **引入 DIS 解决崩溃瓶颈**：`GRPO + DIS` 将四项指标大幅推高至 `93.5 / 70.8 / 84.0 / 70.0`，证明双边重要性门控有效拦截了异步陈旧梯度的破坏。
3. **单轨迹 + 稳健 Critic 带来进一步跃升**：完整版 SAO 相比 SFT 分别取得 `+16.9`、`+21.5`、`+13.1` 与 `+20.7` 个百分点的绝对增益，且全面超越 `GRPO + DIS`（提升 `3.8`～`4.3` 个百分点）与未做 Critic 稳定化改造的 `SAO (DIS only)`。

表中同时列出了 Claude-Sonnet-4.5、GPT-5 High 与 GLM-4.7 作为外部能力参照。由于各模型的参数规模、预训练语料与推理预算并不相同，严谨的算法结论应聚焦于同一 `Qwen3-30B-A3B` 基座下的受控对比。

![SAO SWE-Bench Verified results](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table02-swe-bench-verified.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 2, CC BY 4.0.*

在最多 300 轮交互、128K 上下文的 SWE-Bench Verified 基准（Table 2）中：

| 方法 | Accuracy |
| --- | ---: |
| Qwen3-30B-A3B | 23.0% |
| + GRPO（w/ DIS） | 27.0% |
| **+ SAO** | **29.8%** |

SAO 相比基础模型提升 `6.8` 个百分点，在同样启用 DIS 的前提下比 `GRPO (w/ DIS)` 高出 `2.8` 个百分点，验证了该方法在真实软件工程仓库修复任务上的迁移有效性。

### 6.3 千步训练曲线（Fig. 3）与关键组件消融（Table 3 & Table 4）

![SAO training performance](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig03-training-performance.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 3, CC BY 4.0.*

Fig. 3 展示了训练满 1000 步过程中的准确率演化。Vanilla GRPO 在约 160 步后性能断崖式下跌；加入 DIS 后，`GRPO + DIS` 与 SAO 均能平稳训练至 1000 步。值得注意的是，前 400 步内两者曲线十分接近，400 步之后 SAO 持续拉开差距。这说明：**避免早期崩溃主要归功于 DIS 门控，而训练中后期的上限突破则源于单轨迹采样多样性与高精度的 Critic 优势估计。**

![SAO value training ablation](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table03-value-training-ablation.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 3, CC BY 4.0.*

Table 3 专门消融了价值网络的训练策略：

| 配置 | Value 参数策略 | 每次 Policy 对应 Critic 更新 ($K$) | AIME2025 | BeyondAIME |
| --- | --- | ---: | ---: | ---: |
| **SAO** | Frozen Attention | 2 | **97.3** | **74.8** |
| Single-step-update | Frozen Attention | 1 | 95.00 | 69.75 |
| Full-Parameter Value Training | Full parameter | 2 | 90.62 | 74.50 |

将 Critic 更新频率从 $K=2$ 降至 $K=1$，BeyondAIME 显著下滑 `5.05` 个百分点，证实了高难度分布下快速追踪策略价值的必要性；而放开全参数训练（不冻结 Attention）则导致 AIME2025 下跌 `6.68` 个百分点。

![SAO main ablation](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table04-main-ablation.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 4, CC BY 4.0.*

Table 4 进一步汇总了完整消融与无 Critic 基线：

| 方法 | AIME2025 | BeyondAIME |
| --- | ---: | ---: |
| **SAO** | **97.3** | **74.8** |
| SAO w/o Faster Value | 95.0 | 69.8 |
| SAO w/o Frozen Attention | 90.6 | 74.5 |
| Vanilla VAPO（w/o DIS） | 91.3 | 69.0 |
| Running Mean Baseline | 79.8 | 55.3 |

其中，未配备 DIS 的 Vanilla VAPO 在异步单轨迹环境下约 90 步即发生崩溃；而使用每个 prompt 最近 8 次历史奖励均值作为无 Critic 基线的 `Running Mean Baseline`，最终得分仅停留在 `79.8 / 55.3`，甚至略低于 SFT 起点，证明简单的历史标量均值无法替代状态相关的价值网络。

### 6.4 训练动态机制验证（Fig. 4）：EV、梯度范数与门控比例

![SAO training dynamics](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig04-training-dynamics.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 4, CC BY 4.0.*

Fig. 4 从三个内部训练指标解释了上述消融结果背后的优化动力学：

1. **Explained Variance（Fig. 4a）**：用于衡量价值预测 $V(s)$ 对真实回报 $R$ 方差的解释能力：
   $$
   \operatorname{EV}
   =1-\frac{\operatorname{Var}(R-V(s))}{\operatorname{Var}(R)}.
   $$
   大约在 400 步之后，双倍更新（$K=2$）的 EV 稳定高于单次更新（$K=1$），这一时间拐点与 Fig. 3 中准确率开始拉开差距的时刻高度吻合。
2. **Critic 梯度范数（Fig. 4b）**：全参数 Critic 的梯度范数随训练推进大幅攀升且剧烈抖动，而 Frozen-Attention Critic 的梯度始终维持在低位平滑区间，直接佐证了结构正则化的稳定作用。
3. **Clipped Tokens Ratio（Fig. 4c）**：缺乏 DIS 的 VAPO 裁剪比例几乎为 0，无法过滤异步离策样本并迅速崩溃；SAO 的 DIS 门控比例随策略演化动态调节，最高处仅屏蔽约 `0.006`（0.6%）的 Token，说明异步训练中真正致命的往往是占比极小但比率极端的尾部离策 Token。

## 7. 在线学习适应模拟与动作粒度消融

### 7.1 非平稳偏好下的在线适应实验（Fig. 5）

真实在线交互场景中，系统针对用户的单次请求仅生成一条响应并获取一次即时反馈，天然契合 single-rollout 模式。为检验算法在非平稳环境下的适应速度，论文构建了一个可控的风格切换模拟任务：模型需在 Academic、Cute、Chuunibyou 与 Classical 四种文风中作答，环境偏好看似分阶段突变。由 GLM-4.7 充当裁判，联合评估内容质量与风格匹配度：

$$
r=r_{\text{quality}}\times r_{\text{style}},
\qquad
r_{\text{quality}},r_{\text{style}}\in\{0,1\}.
$$

对照基线采用最近 128 个全局奖励的滑动窗口均值（不同于 Table 4 中按单 prompt 统计最近 8 次奖励的设置）：

$$
b_t=\mathbb E[r_{t-127:t}],
\qquad
\hat A_t=r_t-b_t.
$$

![SAO online learning simulation](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig05-online-learning.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 5, CC BY 4.0.*

如 Fig. 5 所示，每当环境偏好发生切换（灰色阴影区），新目标风格的准确率迅速攀升，旧风格快速退场（左图）；在右图的训练回报曲线中，SAO 在每次分布突变后的恢复速度与收敛高度均明显优于滑动窗口均值，因为后者的历史窗口内仍残留着旧偏好阶段的奖励惯性，而高频更新的 Critic 能依托当前上下文更快校准预期回报。

### 7.2 Token-level 与 Step-level 粒度消融（Fig. 6 & Table 5）

![SAO token-level versus step-level training](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig06-token-vs-step-training.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 6, CC BY 4.0.*

Fig. 6 对比了主线 Token-level GAE 与 5.3 节介绍的两种 Step-level 变体（Average 与 Last-Token）。前 150 步内三条曲线表现接近，此后 Token-level 的平均训练奖励持续走高，而两个按轮次聚合的变体则过早进入平台期。

![SAO action granularity ablation](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table05-action-granularity.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 5, CC BY 4.0.*

Table 5 给出了训练至第 400 步时的基准对比（注意此处为 400 步阶段性快照，因此绝对数值低于 Table 1 的 1000 步最终结果）：

| Action 粒度（400 steps） | AIME2025 | BeyondAIME |
| --- | ---: | ---: |
| Step-level（Average） | 85.8 | 60.5 |
| Step-level（Last-Token） | 87.3 | 62.8 |
| **Token-level** | **89.8** | **66.8** |

`Last-Token` 优于 `Average`，表明单轮末尾的隐状态比整轮平均池化更能概括该步的语义结果；而 `Token-level` 在两项测试中进一步领先 `2.5` 与 `4.0` 个百分点，印证了在单轮包含数千推理 Token 的数学智能体中，保持 Token 级细粒度信用分配的重要性。

## 8. 算法谱系对比与工程实现参考

### 8.1 与 PPO、GRPO、VAPO、GSPO、IMPALA 及异步 RL 框架的横向对比

将 SAO 置于强化学习算法与系统谱系中，可以清晰界定其设计权衡：

| 方法 / 框架 | Baseline 来源 | 每 Prompt 采样数 | 是否需要 Critic | 异步与长时域适配特征 |
| --- | --- | ---: | --- | --- |
| PPO | 学习型 Value Model | 1+ | 是 | 依赖 $\pi_{\theta_{\text{old}}}$ 软裁剪，未专门处理环境观测跳跃 |
| GRPO / RLOO | Prompt 内组统计量 | $G > 1$ | 否 | 存在组内同步屏障，受最慢轨迹拖累 |
| Running Mean | 历史奖励滑动窗口 | 1 | 否 | 无状态区分能力，非平稳分布下存在滞后惯性 |
| VAPO | 学习型 Value Model | 1+ | 是 | 提出 Length-Adaptive GAE，但缺乏异步 DIS 硬门控 |
| GSPO | 组内相对统计 | $G > 1$ | 否 | 采用序列级重要性比率，仍保留组采样结构 |
| IMPALA | V-trace 修正价值 | 1+ | 是 | 截断重要性权重修正回报，未针对超长自回归与工具观测定制 |
| **SAO** | **Token-level Value Model** | **1** | **是** | **Rollout Log-Prob DIS 硬门控 + $K=2$ 冻结注意力 Critic + Skip-Obs GAE** |

在相关工作脉络中，SAO 的双边门控思想与 *Every Step Evolves* 中的 IcePop 过滤机制一脉相承。DIS 的核心贡献在于彻底去除对历史快照 $\pi_{\theta_{\text{old}}}$ 的依赖、直接利用采样时刻持久化的行为日志，并与单轨迹异步采样及 Critic 稳定化策略形成闭环。在系统层面，SAO 关注的是算法侧如何消化异步滞后与单轨迹方差，与 [THUDM/slime](https://github.com/THUDM/slime)、[THUDM/AgentRL](https://github.com/THUDM/AgentRL)、[AReaL](https://github.com/inclusionAI/AReaL) 及 [Alibaba ROLL](https://github.com/alibaba/ROLL) 等负责分布式 rollout-learner 调度与容器编排的底层框架呈互补关系。

### 8.2 SAO 异步训练与 DIS 门控参考伪代码

以下参考伪代码将论文中的异步单轨迹入队、双倍 Critic 更新、Skip-Observation GAE 与 DIS 硬门控串联为完整的数据流：

```python
ready_queue = AsyncTrajectoryQueue()

async def rollout_worker(prompt, rollout_model, environment):
    trace = []
    state = environment.reset(prompt)

    while not state.terminal:
        action, token_ids, logprobs = rollout_model.generate(state.context)
        trace.append({
            "role": "action",
            "token_ids": token_ids,
            "rollout_logprobs": logprobs,
        })

        observation, reward, done = environment.step(action)
        trace.append({
            "role": "observation",
            "text": observation,
        })
        state = state.advance(observation, reward, done)

    # 单条轨迹完成后立即入队，无需等待同 prompt 其他样本
    await ready_queue.put(finalize(trace, reward=state.reward))


def learner_step(batch):
    token_batch = tokenize_with_action_and_observation_spans(batch)

    # 1. Faster Value Update (K = 2) + Frozen-Attention Critic
    for _ in range(2):
        values = critic(token_batch)
        value_targets = build_skip_observation_returns(token_batch, values)
        critic_loss = masked_value_loss(values, value_targets, token_batch.action_mask)
        update_frozen_attention_critic(critic_loss)

    # 2. Skip-Observation Token-level GAE
    with no_grad():
        advantages = skip_observation_gae(token_batch, critic)

    # 3. Direct Double-Sided Importance Sampling (DIS)
    current_logprobs = policy.logprobs(token_batch.input_ids)
    log_ratio = current_logprobs - token_batch.rollout_logprobs
    trusted = (log_ratio > log(1.0 - eps_low)) & (log_ratio < log(1.0 + eps_high))

    effective_mask = token_batch.action_mask & trusted
    ratio = where(effective_mask, exp(log_ratio), 0.0)

    actor_loss = -(ratio.detach() * advantages * current_logprobs)[effective_mask].mean()
    update_policy(actor_loss)
    publish_weights_if_needed()
```

### 8.3 轨迹数据契约、Token 对齐不变量与队列新鲜度门禁

要把上述流程落地为可靠的分布式管线，核心在于保证 rollout 侧与 learner 侧的数据契约严格对齐。一条标准的轨迹记录至少应包含以下字段与不变量：

```json
{
  "trajectory_id": "traj_01J...",
  "prompt_id": "prompt_4821",
  "rollout_model_version": 438,
  "token_ids": [151644, 198, 785, ...],
  "rollout_logprobs": [0.0, 0.0, -0.12, -1.34, ...],
  "action_mask": [0, 0, 1, 1, ...],
  "next_action_index": [-1, -1, 3, 812, ...],
  "reward": 1.0,
  "terminal_reason": "tests_passed"
}
```

1. **Token 与行为概率严格对齐**：`token_ids`、`rollout_logprobs`、`action_mask` 与预计算的 `next_action_index` 长度必须完全一致；系统提示词、用户输入及环境 observation 位置的 `action_mask` 恒为 `0`。
2. **队列新鲜度与有效训练量监控**：DIS 虽然能够屏蔽极端离策 Token，但不意味着队列可以无限积压旧版本数据。可定义单条轨迹的有效 Token 留存率：
   $$
   q(\tau)
   =\frac{\sum_t m_t^{\text{action}}m_t^{\text{DIS}}}
   {\sum_t m_t^{\text{action}}},
   \qquad
   N_{\text{effective}}
   =\sum_{\tau\in\mathcal B}\sum_t
   m^{\text{action}}_{\tau,t}m^{\text{DIS}}_{\tau,t}.
   $$
   当某条陈旧轨迹的 $q(\tau)$ 过低时，说明其大部分动作已偏离当前策略的信任区间，继续送入 learner 只会空耗前向计算，应由队列新鲜度门禁直接丢弃。
3. **防范完成顺序偏差（Completion-Order Bias）**：异步流式队列中，耗时短的简单样本会比长耗时困难样本更早完成入队。若完全按先到先得消费，训练早期的梯度会被简单任务主导，因此工程上宜配合时间窗内的任务/长度配额采样。

## 9. 证据边界、局限性与结论

### 9.1 论文证据盘点与方法局限性

综合全文实验与公式推导，我们需要客观区分论文已充分证实的结论与尚未覆盖的边界：

- **已坚实验证的结论**：在相同的 `Qwen3-30B-A3B` 基座与每步 128 条轨迹预算下，SAO 在四项数学推理基准和 SWE-Bench Verified 上全面超越异步 `GRPO + DIS`；DIS 双边硬门控成功阻止了异步训练在 90～160 步附近的早期崩溃；$K=2$ 的高频 Critic 更新与冻结 Attention 策略有效提升了价值网络的 Explained Variance 并降低了梯度范数；Token 级优势估计显著优于轮次级（Step-level）聚合。
- **系统效率未被量化**：论文以消除同步空泡作为核心动机，但全文横轴均为优化步数（training steps），未报告端到端 wall-clock 训练耗时、tokens/s 吞吐率、GPU 利用率或实际 policy lag 分布。同时，SAO 重新引入了与策略同规模的 Critic 并每步更新两次，learner 侧显存与算力开销显著高于无 Critic 的 GRPO，端到端时间收益取决于环境交互耗时占比。
- **架构耦合与复现缺口**：Frozen-Attention Critic 依赖 MoE 架构提供的专家投影可训练子空间，向稠密模型迁移的效果尚待验证；论文强调的 Critic 预训练冷启动语料未公开细节；在线学习实验仅在 4 类离散文风与单一 LLM 裁判的合成环境中验证，距离真实用户在线反馈治理仍有较大跨度；关于 GLM-5.2（750B-A40B）的部署仅作为摘要与引言中的生产应用声明，未提供对应的训练曲线与消融数据。

### 9.2 总结与落地建议

SAO 为长时域 Agentic RL 提供了一套自洽的异步单轨迹训练范式：顺应交互反馈逐条到达的自然节律，将每 prompt 采样数降为 1 以彻底解除组内等待；依靠生成阶段持久化的 rollout log-prob 执行双边硬门控，以极小的 Token 屏蔽比例遏制离策漂移；再通过双倍速更新、冻结注意力的 Critic 与跳过环境观测的 Skip-Observation GAE，解决单轨迹下的高方差与跨工具信用分配难题。

在实际工程落地中，建议采取渐进式验证路径：首先打通推理与训练引擎的 Token 级 `rollout_logprobs` 透传与 `action_mask` / `next_action_index` 数据契约，通过同版本重放测试确认数值对齐；随后在受控异步滞后下验证 DIS 门控与 Critic 的 Explained Variance 收敛性；最后结合具体环境的长尾程度权衡 Critic 算力开销与移除组屏障带来的系统吞吐收益。

## 参考资料

### 主论文与模型状态

1. Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong. [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508). arXiv:2607.07508v1, 2026.
2. [论文 PDF](https://arxiv.org/pdf/2607.07508) 与 [TeX Source](https://arxiv.org/e-print/2607.07508).
3. [arXiv HTML version](https://arxiv.org/html/2607.07508) 与 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
4. Z.AI. [GLM-5.2: Built for Long-Horizon Tasks](https://z.ai/blog/glm-5.2).
5. Z.AI. [GLM-5.2 model card](https://huggingface.co/zai-org/GLM-5.2).

### 强化学习算法

6. Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
7. Schulman et al. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015.
8. Shao et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.
9. Ahmadian et al. [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740), 2024.
10. Yue et al. [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118), 2025.
11. Zheng et al. [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071), 2025.
12. Xu and Ding. [Single-Stream Policy Optimization](https://arxiv.org/abs/2509.13232), 2025.
13. Ling Team et al. [Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model](https://arxiv.org/abs/2510.18855), 2025.

### 异步与 Agentic RL 系统

14. Mnih et al. [Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html), ICML 2016.
15. Espeholt et al. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://proceedings.mlr.press/v80/espeholt18a.html), ICML 2018.
16. Noukhovitch et al. [Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models](https://arxiv.org/abs/2410.18252), 2024.
17. Fu et al. [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298), 2025.
18. Lu et al. [ROLL Flash: Accelerating RLVR and Agentic Training with Asynchrony](https://arxiv.org/abs/2510.11345), 2025.
19. Xu et al. [MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents](https://arxiv.org/abs/2509.18119), 2025.
20. [THUDM/slime](https://github.com/THUDM/slime), [THUDM/AgentRL](https://github.com/THUDM/AgentRL), [AReaL](https://github.com/inclusionAI/AReaL), [Alibaba ROLL](https://github.com/alibaba/ROLL).

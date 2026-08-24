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

大语言模型强化学习正在从“对一个问题生成一段答案”转向“让模型在环境里工作很久”。数学推理 Agent 会交替写自然语言、调用 Python、读取执行结果再继续推理；编码 Agent 会浏览仓库、修改文件、运行测试、分析失败，完整轨迹可能持续数十分钟甚至数小时。此时，强化学习系统面对的不只是算法目标，还要面对一个很具体的系统事实：不同轨迹完成时间差异巨大。

同步训练通常先收集一整批 rollout，再统一计算奖励和更新参数。只要这一批中有一条轨迹特别慢，已经完成的轨迹就只能等待。Group Relative Policy Optimization（GRPO）还要求同一个 prompt 的多条 response 组成完整 group，才能计算组内相对 advantage。它在同步推理任务里是一种实用的无 Critic 方法，却在异步 Agentic RL 中形成了额外同步屏障。

**Single-Rollout Asynchronous Optimization（SAO）** 的判断是：如果长时域任务天然只能或更适合逐条获得反馈，就不应强行维持 prompt 内 group。SAO 让每个 prompt 只生成一条 rollout，轨迹一完成就可以进入训练队列；为补上 GRPO 组内 baseline 消失后的方差控制能力，它重新引入 value model，并用更快的 Critic 更新、冻结 Attention 的 Critic 训练和 Skip-Observation GAE 让单轨迹 Actor-Critic 可用。针对异步数据的 policy lag，SAO 又用 rollout 时保存的 Token log-probability 直接计算 importance ratio，并把偏离信任区间的 Token 从梯度中完全屏蔽。

这篇论文最值得精读的地方不是“group size 从 8 改成 1”，而是它把 **异步调度、行为策略证据、优势估计、价值模型稳定性和 Agent 轨迹语义** 放进同一个训练设计。它也留下不少重要空白：论文没有给出吞吐提升、GPU 利用率、训练硬件、policy lag 分布或完整代码；“成功部署到 GLM-5.2”是生产使用声明，不是公开的 750B 模型消融实验。

本文以 arXiv `2607.07508v1` 为准，做论文、TeX source 和图表层面的精读，并进一步给出一个不冒充官方实现的工程数据契约与训练伪代码。本文不声称复现 SAO、Qwen3-30B-A3B 实验或 GLM-5.2 训练管线。

## 1. 一句话贡献

SAO 的一句话贡献可以概括为：

> 用“一条 prompt 对应一条立即可训练的 rollout”移除 GRPO 的组内同步屏障，再通过 rollout log-prob 驱动的双边 Token 门控和稳定的 value model，把异步 Agentic RL 的吞吐动机转化为可持续训练的优化方法。

这个贡献由五个相互依赖的部件组成。

1. **Single-rollout sampling**：每个 prompt 只采一条轨迹，不等待同 prompt 的其他轨迹。
2. **Direct Double-Sided Importance Sampling（DIS）**：直接比较当前策略与实际 rollout 策略的 Token 概率。
3. **Faster Value Update**：每次 policy 更新对应两次 value 更新，让 Critic 更快追踪变化中的策略。
4. **Frozen-Attention Value Model**：RL 阶段冻结 value model 的 Attention，只更新 MoE projection，降低 Critic 梯度波动。
5. **Skip-Observation Token-level GAE**：优势递推跨过环境 observation，只在模型生成 action Token 之间传播。

任何一个部件都不能单独代表 SAO。只做 single rollout 会遭遇高方差；只做 DIS 仍可能保留 group barrier；只训练 Critic 而不保存 rollout log-prob，又无法对异步数据进行可信的 off-policy 诊断。

## 2. 论文信息与发布边界

| 项目 | 内容 |
| --- | --- |
| 题名 | Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning |
| 作者 | Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong |
| 机构 | Tsinghua University |
| arXiv | `2607.07508v1` |
| 提交日期 | 2026-07-08 |
| PDF | 14 pages |
| DOI | `10.48550/arXiv.2607.07508` |
| 许可 | CC BY 4.0 |
| 主实验模型 | Qwen3-30B-A3B-Thinking-2507 |
| 生产部署声明 | 用于开放模型 GLM-5.2（750B-A40B）的 Agentic RL pipeline |
| 官方 SAO 代码 | 截至 2026-07-15 未发现公开链接 |
| 本文状态 | 论文精读与系统设计复盘，不做训练复现 |

论文源文件使用 `neurips_2026` preprint 样式，并保留了 `Under review, Feb 2026` 的模板提示，但 arXiv 页面没有会议录用信息。模板、投稿状态和正式接收是三件不同的事，因此本文只称它为 **arXiv v1 预印本**，不写成“NeurIPS 2026 论文”。

论文作者栏只列清华大学，并注明前两位作者在 Z.AI 实习期间完成相关工作。摘要称 SAO 已用于训练开放的 GLM-5.2；GLM-5.2 模型权重公开并不意味着 SAO 的训练代码、Critic 初始化数据、异步调度器或生产参数同步实现已经公开。

## 3. 论文到底解决什么问题

SAO 不是通用强化学习算法的完整替代品。它针对的是一组相当具体的条件：

- 模型以多轮 Agent 方式与工具或环境交互；
- 不同 rollout 的完成时间高度不均衡；
- rollout 生成和参数训练已经解耦，可以并行运行；
- learner 更新期间，rollout engine 仍在用稍旧版本生成数据；
- 环境通常只为一次真实交互返回一条反馈，不适合为同一 prompt 人为复制很多次；
- 系统能保存每个生成 Token 在 rollout 时的 log-probability；
- 团队有能力训练并部署一个额外 value model。

如果任务只是短答案、生成长度接近、每个 prompt 可以廉价采样很多次，那么同步 GRPO 的 barrier 未必构成主要问题。如果系统无法稳定保存 rollout log-prob，SAO 的 DIS 也缺少行为策略证据。如果显存预算无法容纳 Critic，重新引入 Actor-Critic 的成本可能抵消 single rollout 的系统收益。

因此，SAO 的问题陈述应写成“**如何让单轨迹异步 Agentic RL 稳定且有效**”，而不是“如何让所有 RLHF 更快”。

## 4. Fig. 1：先看论文最想展示的结果

![SAO benchmark results](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig01-benchmark-results.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 1, CC BY 4.0.*

Fig. 1 把 AIME2025、BeyondAIME、HMMT Nov 2025、IMOAnswerBench 和 SWE-Bench Verified 放在同一张柱状图中。前四项是允许 Python 工具的推理任务，编码任务使用 OpenHands scaffold。图中的三组柱分别是基础或 SFT baseline、GRPO 和 SAO。

SAO 在五项任务上都高于图中的对应 baseline。最显眼的是 AIME2025，从 SFT 的 `80.4` 提升到 `97.3`；BeyondAIME 从 `53.3` 提升到 `74.8`；SWE-Bench Verified 从基础模型 `23.0` 提升到 `29.8`。

但 Fig. 1 只能证明论文报告的任务效果，不能证明异步训练的系统效率。图里没有 wall-clock time、rollout throughput、GPU utilization、queue age、平均 policy lag、训练成本或同等时间预算下的收敛曲线。异步架构的效率动机很合理，却没有在本论文中被系统 benchmark 量化。这会成为全文批判性阅读的一条主线。

## 5. 为什么 Agentic Rollout 会形成 Straggler

假设一个训练 batch 有 128 条轨迹。普通数学题可能 30 秒完成，复杂题可能调用 Python 十几次并生成数万 Token；SWE-Bench 中简单修复可能很快通过测试，复杂 issue 可能在 300 轮上限附近才结束。同步调度的完成时间近似由最慢轨迹决定：

$$
T_{\text{batch}}^{\text{sync}}
\approx \max_{i\in\{1,\ldots,B\}} T_i.
$$

平均轨迹耗时并不能反映同步 barrier 的成本，真正决定等待的是尾部。轨迹时长分布越偏、环境 I/O 越不稳定、turn 上限越高，尾部越严重。

异步 actor-learner 试图把这个批边界拆掉。完成的轨迹先写入队列，learner 按可用数据持续更新；长轨迹继续执行，不阻塞已经完成的数据：

```text
rollout workers              learner
---------------              -------
trajectory A --done------->  replay / ready queue
trajectory B --------done->  sample -> optimize
trajectory C --done------->  publish new weights
trajectory D ------------->  continue generation
```

这样做提高了资源重叠，却引入新问题：trajectory D 可能由旧策略启动，在它完成时 learner 已更新很多次。异步不是免费并行，它用 **policy lag 和 off-policy drift** 换取更少等待。

## 6. PPO：SAO 重新引入 Critic 的起点

给定 query $q$ 和模型生成序列 $y=[y_1,\ldots,y_{|y|}]$，策略写作 $\pi_\theta(y\mid q)$。PPO 使用当前策略与收集数据时 old policy 的概率比：

$$
r_t(\theta)=
\frac{\pi_\theta(y_t\mid q,y_{<t})}
{\pi_{\theta_{\text{old}}}(y_t\mid q,y_{<t})}.
$$

论文给出的统一 clipped surrogate objective 是：

$$
\mathbb{E}\left[
\frac{1}{|y|}\sum_{t=1}^{|y|}
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

PPO 的 advantage 通常依赖 value model $V_\phi$。Critic 用均方误差拟合 return：

$$
\mathcal L_{\phi}^{\mathrm{VF}}
=\mathbb E\left[(V_\phi(q,y_{<t})-R)^2\right].
$$

Generalized Advantage Estimation（GAE）通过折扣累积 temporal-difference residual 平衡偏差与方差：

$$
\hat A_t^{\mathrm{GAE}}
=\sum_{l=0}^{|y|-t-1}(\gamma\lambda)^l\delta_{t+l},
\qquad
\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t).
$$

Critic 的好处是 baseline 可以依赖当前状态，不要求同一个 prompt 同时出现多条 response。代价是额外模型参数、显存、前反向计算和训练不稳定性。SAO 为了支持 single rollout，主动接受这个代价。

## 7. GRPO：为什么 Group Baseline 既有价值又形成屏障

GRPO 为每个 prompt $q$ 采样 $G$ 条输出 $\{y_1,\ldots,y_G\}$，得到奖励 $\{R_1,\ldots,R_G\}$，用组内均值和标准差构造 advantage：

$$
\hat A_{i,t}
=\frac{R_i-\mu_R}{\sigma_R},
\qquad
\mu_R=\frac{1}{G}\sum_{j=1}^{G}R_j,
$$

$$
\sigma_R=
\sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j-\mu_R)^2+\epsilon_{\text{stab}}}.
$$

如果奖励只在 sequence 末尾给出，同一条 response 的所有 Token 往往共享这个组内 advantage。GRPO 不需要学习 Critic，节省了 value model 的成本，也避免 value prediction 自身的误差。

问题在于 $\mu_R$ 和 $\sigma_R$ 必须等组内奖励齐全后才能计算。假设一组 8 条 response 中 7 条早已完成，最后一条因为工具调用或测试执行很慢，前 7 条也不能立即成为完整 GRPO 样本。这不是普通 batch packing 能完全解决的问题，而是 advantage 定义本身引入的 group dependency。

SAO 不是说 GRPO 的统计方法错误，而是说它与“反馈按单条轨迹到达”的异步系统语义不匹配。

## 8. 三种 Policy：Current、Old 与 Rollout

异步 RL 讨论中最容易混淆的是三个策略版本。

| 名称 | 记号 | 作用 |
| --- | --- | --- |
| Current policy | $\pi_\theta$ | learner 当前正在更新的模型 |
| Old policy | $\pi_{\theta_{\text{old}}}$ | PPO 优化 epoch 中用于构造相对更新约束的快照 |
| Rollout policy | $\pi_{\text{rollout}}$ | 真正生成某个 Token 时使用的行为策略 |

同步 PPO 中，old policy 和 rollout policy 常常近似同一快照；异步系统里，它们可能分开。rollout worker 的权重同步有延迟，长轨迹甚至可能跨越 learner 的多个更新周期。理论上若要精确追踪每个版本，需要保存大量 checkpoint 或确保每条轨迹从头到尾绑定一个不可变行为模型。

SAO 的工程简化是：不再依赖一个“最新 old policy”去近似历史行为策略，而是直接保存生成时的 rollout log-prob，并计算 current/rollout ratio。这避免重新加载历史 checkpoint，但要求 rollout 记录的概率可靠、与实际采样 Token 一一对应。

## 9. Policy Lag 不只是版本号之差

可以把 lag 粗略写成 learner 版本与 rollout 版本的差：

$$
\operatorname{lag}(\tau)=v_{\text{learner-at-consume}}-v_{\text{rollout}}.
$$

但版本差只是一种代理指标。一次更新的幅度、模型参数规模、不同 Token 的概率敏感度和数值后端差异都会影响真实 off-policy 程度。更直接的诊断是 Token ratio 或 log-ratio：

$$
\Delta\ell_t
=\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{rollout}}(a_t\mid s_t).
$$

同样是落后 5 个版本，有的 Token 概率几乎没变，有的 Token 已偏离几个数量级。SAO 的 DIS 正是从 Token 概率而不是单纯版本号判断是否信任样本。

生产系统仍应同时记录两类指标：版本 lag 用于调度和排队诊断，ratio 分布用于优化稳定性诊断。论文只展示 clipped-token ratio，没有披露 queue age、版本 lag 或 ratio 分位数。

## 10. Fig. 2：SAO 的单轨迹异步设计

![SAO single-rollout overview](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig02-single-rollout-overview.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 2, CC BY 4.0.*

图的上半部分是 GRPO。不同编号代表轨迹完成顺序；训练不能按完成顺序直接消费，而要先凑齐属于同一 group 的样本。下半部分是 SAO，轨迹按 `9, 8, ..., 3, 2, 1` 的完成顺序进入可训练状态，不再等待 prompt 内兄弟样本。

图右侧把 trust region 画成 rollout policy 与 current policy 的比值区间。普通示意强调一侧 PPO 条件；SAO 使用上下两侧边界，把越界 Token 排除。需要注意，图是方法示意，不是完整系统架构：它没有画 rollout worker 数量、ready queue、参数广播、Critic placement、reward service、工具 sandbox 或 checkpoint 管理。

## 11. Single Rollout 不等于 Batch Size 1

论文实验使用：

- policy batch size：`128`；
- SAO group size：`1`；
- GRPO baseline：每批 `16` 个 prompt，每个 prompt `8` 条 rollout，总样本同样为 `128`。

所以 single rollout 的准确含义是 **one rollout per prompt**，不是一次 optimizer step 只看一个样本。learner 仍然可以从不同 prompt 的已完成轨迹聚合一个大 batch，以获得并行效率和更稳定的梯度估计。

这个区别直接影响工程实现。如果把 group size 1 误写成 microbatch 1，就会错误地把梯度累积、global batch、prompt sampling 和异步到达顺序混为一谈。一个合理的数据路径是：

```text
many prompts
  -> one rollout per prompt
  -> completion-ordered ready queue
  -> freshness / validity filter
  -> global training batch of 128 trajectories
  -> two critic updates + one policy update
```

论文没有详细说明 ready queue 如何组成 batch、是否按长度 packing、是否设置最大 staleness 或是否对旧样本降权。这些属于复现时必须补齐的系统决策。

## 12. Direct Importance Sampling：直接相信 Rollout 日志

SAO 使用以下 Token ratio：

$$
r_t(\theta)
=\exp\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{rollout}}(a_t\mid s_t)
\right).
$$

这里的 $\pi_{\text{rollout}}$ 不是事后用一个近似 old model 重算的概率，而是 rollout engine 在真正采样 $a_t$ 时记录的概率。它回答的是最关键的统计问题：**这条数据由什么行为分布生成？**

这种设计省去历史策略集合，也避免用“当前最近的 old policy”错误代表很久以前的轨迹。它对日志系统提出严格要求：

1. 保存的是实际采样前、应用 temperature/top-p 等变换后的行为概率，还是基础 logits 的概率，必须定义清楚。
2. Tokenizer、chat template、工具 observation 拼接和位置编码必须与 learner 重算完全一致。
3. rollout engine 和 training engine 的浮点精度、并行切分或 kernel 差异可能导致 log-prob mismatch。
4. 环境 Token 不由策略采样，不能伪造 rollout log-prob。
5. 若权重在一条轨迹中途被热更新，必须按 Token 保留真实概率，不能只记录单个 trajectory version。

论文说明了“使用 rollout logs”，却没有给出这些协议细节。复现失败很可能首先表现为 ratio 异常，而不是 loss 代码语法错误。

## 13. DIS 目标函数：不是普通 Clip

论文把 SAO policy objective 写成：

$$
L(\theta)
=\hat{\mathbb E}_t\left[
f(r_t(\theta);\epsilon_l,\epsilon_h)
\hat A_t
\log\pi_\theta(a_t\mid s_t)
\right].
$$

校准函数为：

$$
f(x;\epsilon_l,\epsilon_h)=
\begin{cases}
x, & 1-\epsilon_l < x < 1+\epsilon_h,\\
0, & \text{otherwise}.
\end{cases}
$$

这和标准 PPO 的 `clip(x, low, high)` 有本质差别。标准 clip 会把越界 ratio 截到边界值，并根据 advantage 符号选择保守 surrogate；SAO 的函数在越界时直接返回 0，相应 Token 不再贡献 policy gradient。

更准确的工程描述是：

```python
ratio = exp(current_logprob - rollout_logprob)
trusted = (ratio > 1 - eps_low) & (ratio < 1 + eps_high)
weight = where(trusted, ratio, 0.0)
policy_loss = -(weight.detach() * advantage * current_logprob).mean()
```

上面是便于理解的自写伪代码，不是官方实现。实际实现还必须处理 action mask、sequence normalization、梯度是否穿过 ratio、混合精度和分布式 reduction。

## 14. 为什么叫 Double-Sided

普通 PPO 的 surrogate 对不同 advantage 符号关注不同越界方向。SAO 则无论 advantage 正负，只要 ratio 低于下界或高于上界，都屏蔽该 Token。论文因此称它为 double-sided calibration。

推理实验的阈值是：

$$
\epsilon_{\text{low}}=0.3,
\qquad
\epsilon_{\text{high}}=5.0,
$$

对应可信 ratio 区间：

$$
0.7 < r_t < 6.0.
$$

编码实验使用：

$$
\epsilon_{\text{low}}=0.8,
\qquad
\epsilon_{\text{high}}=3.0,
$$

对应：

$$
0.2 < r_t < 4.0.
$$

这些区间并不比常见 PPO `0.8-1.2` 更窄。“strict”指的是两侧越界都完全拒绝，而不是 epsilon 数值很小。文章若简单写“更严格、更窄的 clip”会误导读者。

上下界高度不对称也值得注意。概率显著上升和显著下降对训练的风险不同，论文允许较宽的上侧变化；具体阈值又随推理/编码任务变化，说明它仍是经验性超参数，不是无调参理论结论。

## 15. DIS 解决什么，又引入什么偏差

DIS 的直接收益有三点。

- 不需要维护大量历史 old-policy checkpoint。
- 可以按 Token 识别极端 off-policy 数据，而不是整条轨迹一刀切。
- 越界 Token 不会用巨大或极小 ratio 产生破坏性梯度。

代价同样明确。

- 被屏蔽 Token 不再提供学习信号，数据利用率下降。
- mask 与当前策略相关，可能引入选择偏差，只学习“仍与当前模型接近”的部分。
- 如果早期 action 越界而后续 action 未越界，保留后续 Token 是否仍有一致的 trajectory-level 解释，需要谨慎理解。
- rollout log-prob 若因引擎差异不准确，门控会把实现误差当作 policy drift。
- 论文没有给出有效样本量、不同位置的 mask 分布或 lag 与 mask 的关系。

因此，线上监控不能只看平均 clip ratio。至少应按 policy version lag、trajectory length、task domain、turn index 和 reward 分桶观察 masked-token ratio。

### 15.1 DIS 的数值实现不能直接除概率

真实系统通常不会先把两个 Token 概率从 log space 还原，再执行除法。低概率 Token 在 `bf16/fp16` 下可能下溢，两个推理引擎的归一化细节也可能产生微小差异。更稳妥的计算是先得到：

$$
\Delta_t
=\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{rollout}}(a_t\mid s_t),
$$

再在足够精度下计算 $r_t=\exp(\Delta_t)$。门控本身也可以直接在 log-ratio 空间比较：

$$
\log(1-\epsilon_l)<\Delta_t<\log(1+\epsilon_h),
$$

这样可以避免先指数化一个已经极端的数。无论采用哪种写法，`rollout_logprobs` 都必须对应“采样前、经过相同 temperature/top-p 语义处理后”的行为分布。若 rollout engine 保存的是原始 logits，而 learner 重算的是采样分布 log-prob，ratio 在同一模型版本上也不会接近 1。

上线前应建立 **same-version replay test**：固定模型权重、输入 Token、attention mask、position id 和采样配置，分别由 rollout engine 与 learner engine 计算 action Token log-prob。差异要按最大绝对误差、均值、分位数和序列位置观察。量化、tensor parallel、不同 fused kernel 或 vocabulary padding 都可能造成偏差；只有先测出同版本噪声底线，才能把线上 ratio 偏移解释为 policy lag。

还要区分“梯度为零”和“样本从计算图删除”。一种实现是保留所有 Token 的 forward，最后将越界位置乘零 mask；另一种实现会 compact 有效 Token 后再归约。前者逻辑简单但不节省前向计算，后者可能提升效率，却必须保持 advantage、position 与序列归一化一致。论文只定义优化目标，没有承诺某一种 kernel 实现。

## 16. 为什么 Single Rollout 需要 Value Model

当每个 prompt 只有一个 reward 时，无法用同 prompt 其他样本的均值回答“这次结果比正常水平好多少”。直接使用 raw reward 类似 REINFORCE，梯度方差往往很高。

Critic 提供状态相关 baseline：

$$
\hat A_t \approx R_t - V_\phi(s_t).
$$

如果一个难题得到中等 reward，Critic 可以判断它相对该状态的预期结果是否已经不错；全局 running mean 则把难题和简单题混在一起。对于在线分布变化，状态相关 baseline 也有机会比历史窗口更快适应。

但不准确的 Critic 会产生错误 advantage。single rollout 没有组内统计提供第二重保护，因此 SAO 把 value model 质量视为主要瓶颈，而不是附属模块。

## 17. Faster Value Update：让 Critic 追上 Actor

策略每次更新后，状态分布和 return 分布都会改变。如果 Critic 仍拟合旧策略，它产生的 advantage 会滞后于 Actor。SAO 将更新频率解耦：

$$
\text{for each policy update:}
\qquad
\underbrace{K\text{ critic updates}}_{K=2}
+\underbrace{1\text{ actor update}}_{1}.
$$

论文把这一设计称为 Faster Value Update，也可以从 two-timescale update 的角度理解：Critic 使用更快时间尺度逼近当前策略的 value，Actor 在相对稳定的 baseline 上更新。

这不等于机械地把 Critic learning rate 翻倍。实验中 policy learning rate 为 $1\times10^{-6}$，value model learning rate 为 $5\times10^{-6}$，同时 Critic 每批更新两次。更新次数、学习率、数据复用和参数子集共同决定实际时间尺度。

更快 Critic 也有风险：同一批轨迹重复优化可能过拟合，尤其当 reward 噪声大、batch 相关性高或 value pretraining 很弱。论文只消融 $K=1$ 与 $K=2$，没有给出更大 $K$ 的曲线。

## 18. Frozen-Attention Value Model

作者观察到，value model 全参数 RL 训练时梯度范数明显偏大；分解后认为不稳定主要来自 Full Attention 层，而 MoE 层相对稳定。SAO 因此在 RL 阶段冻结 Critic 的 Attention 参数，只优化 MoE projection。

直觉是：预训练 Attention 已经具备读取上下文和定位相关 Token 的能力，Critic 更需要学习的是把已有表示映射到 value。冻结 Attention 相当于正则化，避免稀疏、高方差 reward 破坏底层语义路由。

这个结论有很强的架构条件。主实验使用 Qwen3-30B-A3B MoE；“只训练 MoE projection”在 dense Transformer 上没有直接对应项。即使同为 MoE，不同专家路由、共享专家、value head 和并行策略也会改变稳定性。论文自己在 limitations 中承认结果不一定迁移到更小模型和非 Agentic RLHF。

## 19. Value Model Cold Start

论文强调需要扩大 value pretraining corpus，为 Critic 提供稳健初始化。原因很直接：RL 开始时 Actor 会立刻依赖 value estimate；如果 Critic 从近似随机的状态开始，早期错误 advantage 可能把策略推离可恢复区域。

理想的 value pretraining 数据至少需要覆盖：

- 与 RL prompt 相近的任务分布；
- 成功和失败轨迹；
- 不同长度、turn 数和工具调用模式；
- 每个 action Token 或状态位置对应的 return target；
- reward/verifier 版本；
- observation mask 和 episode termination 语义。

论文没有公开 corpus 大小、构建方法、训练步数、硬件或单独消融数字。TeX 中被注释的草稿内容不能当作正式论文证据。因此“Scaling Value Pretraining”应被写成重要经验主张，而不是可直接复现的 recipe。

## 20. Agentic Trajectory：Action 与 Observation 不是同一种 Token

论文将多轮轨迹写成：

$$
\mathcal T=[a_0,o_0,a_1,o_1,\ldots],
$$

其中 $a_i$ 是模型生成的 action，$o_i$ 是环境反馈。一个 action 可以是自然语言推理、工具调用 JSON 或代码 patch；observation 可以是 Python 输出、测试日志、文件内容或工具错误。

从 Transformer 输入看，它们最后都可能变成 Token 序列；从策略梯度看，它们完全不同。模型选择了 action Token，所以这些 Token 有行为概率；环境 observation 不是模型动作，不应该被当作 policy sample，也不应该要求模型为其 log-prob 负责。

训练数据必须至少有两种 mask：

```text
loss_mask:       1 for model action tokens, 0 for environment observations
value_mask:      define which states receive value supervision
```

如果仅靠特殊字符串猜测 observation 边界，工具输出中出现转义标记就可能破坏训练。可靠实现应在轨迹事件层保留结构，再由统一 tokenizer 构造 Token span。

## 21. Skip-Observation Token-level GAE

标准 Token-level GAE 会沿相邻 Token 递推。Agent 轨迹中，当前 action 的最后一个 Token 后面不是模型选择的下一个 action，而是一段外部 observation。若直接跨相邻位置计算 TD residual，value model 就被要求解释环境生成文本的价值，噪声会进入 advantage。

SAO 令 $a_{i,N}$ 表示第 $i$ 个 action 的最后一个 Token，$a_{i+1,0}$ 表示下一个 action 的第一个 Token，直接跨过 $o_i$：

$$
\hat A(a_{i,N})
=\delta_i+\gamma\lambda\hat A(a_{i+1,0}),
$$

其中：

$$
\delta_i
=r_i+\gamma V(a_{i+1,0})-V(a_{i,N}).
$$

这不是删除 observation。下一次 action 的模型输入仍然包含 observation，因此 $V(a_{i+1,0})$ 可以基于环境反馈判断后续成功概率。被跳过的是“把 observation Token 当作模型动作逐 Token 计算 value transition”的做法。

对 action 内部 Token，仍可沿正常自回归位置传播 advantage；在 action-to-observation-to-next-action 的边界处，使用显式 `next_action_index` 跳转。工程实现不应只检查是否遇到 tool role，而应在预处理阶段构造下一可学习 Token 的索引。

一个简化的边界表可以是：

| Token 区间 | 生成者 | Policy loss | Value target | 下一递推位置 |
| --- | --- | --- | --- | --- |
| Assistant reasoning | Policy | 是 | 是 | 下一个 action Token |
| Tool call | Policy | 是 | 是 | 下一个 action Token 或下一轮首 Token |
| Tool observation | Environment | 否 | 通常屏蔽 | 跳过 |
| Final answer | Policy | 是 | 是 | terminal |

论文没有展开 terminal、工具失败、中途截断和超长轨迹 truncation 的具体处理。复现时必须定义：超出 128K 是 failure、bootstrap state 还是丢弃样本；环境异常退出是否给 reward；最后一个 action 的 bootstrap value 是否为 0。

### 21.1 Terminal、Timeout 与 Truncation 必须分开

在普通 episodic RL 中，真正终止（terminal）和时间上限截断（truncated）已经不能混写；Agentic RL 还多出 sandbox crash、工具限流、用户取消、上下文溢出和基础设施抢占。若所有情况都把最后 bootstrap value 设为 0，系统就会把“任务尚未完成但被基础设施中断”误学成失败终点。

一个可操作的终止协议至少区分：

| 终止类型 | 是否是任务语义终点 | 建议 bootstrap | 是否进入 Actor |
| --- | --- | --- | --- |
| `success` / `verified_failure` | 是 | 0 | 是 |
| `max_turns` / `max_tokens` | 否或不确定 | 用末状态 value，另记惩罚 | 需策略决定 |
| `tool_error` | 取决于任务定义 | 末状态 value 或显式环境 reward | 通常可用 |
| `infra_preempted` / `worker_lost` | 否 | 不应伪造 0 | 通常隔离 |
| `policy_cancelled` | 否 | 不应伪造 0 | 通常隔离 |

`Skip-Observation GAE` 也需要为每个 action Token 预先构造 `next_learnable_token_index`。普通位置指向下一个 action Token；action 尾部指向下一轮 action 首 Token；真正 terminal 指向哨兵并令 bootstrap 为 0；truncation 则指向末状态 value。这个索引应与 `action_mask` 一起持久化或由同一个确定性预处理器生成，不能让 Actor loss 和 Critic target 各自猜一次边界。

工具调用失败尤其容易误标。如果“选择了不存在的工具”是 policy 行为造成的任务失败，它应保留 action gradient；如果工具后端因为网络故障返回 500，则 observation 反映的是环境噪声，不应默认把全部负 reward 归因给前一个 Token。成熟系统会同时记录 `agent_error` 与 `environment_error`，在 reward pipeline 中分别处理，并把基础设施错误率纳入训练数据门禁。

## 22. Length-Adaptive GAE

实验采用 VAPO 的 length-adaptive GAE，policy 的 $\lambda$ 写成：

$$
\lambda_{\text{policy}}
=1-\frac{1}{\alpha l},
\qquad \alpha=1.5,
$$

其中 $l$ 与有效序列长度有关。长轨迹使用更接近 1 的 $\lambda$，让 reward 能传播得更远；短轨迹的 $\lambda$ 较小，降低高方差远距离估计的影响。value model 训练使用 $\lambda_{\text{critic}}=1$。

Agentic RL 的“长度”不是一个无歧义字段。可以按全部 Token、action Token、turn 数或剩余 horizon 计算。主文给出公式和超参数，但没有完整解释 padding、observation 和截断如何进入 $l$。附录的 step-level variant 明确将 `step number` 代替原始 Token 长度，说明主线方法更接近 Token 粒度。

如果实现时把 observation Token 也算入 $l$，工具日志很长的轨迹会获得更接近 1 的衰减系数；如果只算 action Token，则更接近模型真正做出的决策长度。两者可能产生明显差异，必须通过单元测试锁定。

## 23. Token-level 与 Step-level Value 的区别

附录定义一个 step 为一轮对话，并尝试两种 step value：

**Step Average**：

$$
V(S_i)=\frac{1}{n}\sum_{j=1}^{n}v_{i,j}.
$$

**Last-Token Prediction**：

$$
V(S_i)=v_{i,n}.
$$

随后按 step 计算：

$$
\delta_i=R_i+\gamma V(S_{i+1})-V(S_i),
$$

得到一个 $\hat A_i$，再把同一 advantage 分给 step 中所有 Token。这种做法平滑了局部 value noise，也降低存储和计算复杂度，但会让一个长 action 内的早期探索、关键工具选择和最终答案共享同一学习信号。

论文结果显示 Token-level 更好。这个结果并不证明 step-level 在所有 Agent 中都差。实验的 step 定义是 conversation turn；对动作天然离散的 GUI Agent、游戏 Agent 或 API workflow，step-level 可能更符合环境语义。论文只在数学 Agent 设置中比较了两个聚合方法。

## 24. 一份不冒充官方代码的 SAO 训练伪代码

下面的伪代码只用于把论文模块串起来。它省略分布式并行、梯度累积、KL、长度归一化和混合精度细节。

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

    await ready_queue.put(finalize(trace, reward=state.reward))


def learner_step(batch):
    token_batch = tokenize_with_action_and_observation_spans(batch)

    # Critic tracks the moving policy faster than the actor.
    for _ in range(2):
        values = critic(token_batch)
        value_targets = build_skip_observation_returns(token_batch, values)
        critic_loss = masked_value_loss(values, value_targets)
        update_frozen_attention_critic(critic_loss)

    with no_grad():
        advantages = skip_observation_gae(token_batch, critic)

    current_logprobs = policy.logprobs(token_batch.action_token_ids)
    ratio = exp(current_logprobs - token_batch.rollout_logprobs)
    trusted = (ratio > 1 - eps_low) & (ratio < 1 + eps_high)

    effective_mask = token_batch.action_mask & trusted
    actor_loss = masked_policy_loss(
        current_logprobs,
        ratio,
        advantages,
        effective_mask,
    )
    update_policy(actor_loss)
    publish_weights_if_needed()
```

这个伪代码暴露了几个论文图没有展示的顺序问题。Critic 更新用的是同一批数据还是独立 Critic replay？advantage 是在两次 Critic 更新后重算，还是在更新前缓存？参数发布频率是否等于 optimizer step？论文没有给出答案。实现者需要通过实验选择，但不能把自己的选择回写成论文事实。

## 25. 训练数据契约

异步 RL 的训练正确性首先是日志契约问题。一个最小的 `TrajectoryRecordV1` 可以写成：

```json
{
  "trajectory_id": "traj_01J...",
  "prompt_id": "prompt_4821",
  "task": "swe_bench_verified",
  "rollout_worker_id": "rollout-17",
  "rollout_model_version": 438,
  "started_at": "2026-07-15T01:05:04Z",
  "completed_at": "2026-07-15T01:18:52Z",
  "token_ids": [151644, 198, 785,  ...],
  "rollout_logprobs": [-0.12, -1.34, -0.03, ...],
  "action_mask": [0, 0, 1, 1, 1, 0, ...],
  "observation_spans": [[341, 812], [1305, 2217]],
  "turn_boundaries": [0, 341, 812, 1305, 2217],
  "rewards": {
    "task_success": 1.0,
    "format_valid": 1.0
  },
  "terminal_reason": "tests_passed",
  "tokenizer_version": "qwen3-2507",
  "chat_template_version": "agent-v4",
  "environment_version": "openhands-sandbox-v12"
}
```

关键约束如下。

1. `rollout_logprobs` 必须与 `token_ids` 同长度或有可验证的 action-only 映射。
2. `action_mask` 不能把系统提示、用户输入或 observation 标成策略动作。
3. `rollout_model_version` 与参数发布时间必须可追溯，便于计算 queue lag。
4. reward 要保留组成项和 verifier 版本，不能只存最终标量。
5. 轨迹截断、环境异常、用户取消和 sandbox timeout 必须有不同 `terminal_reason`。
6. Tokenizer 与模板版本必须进入样本，否则重算 log-prob 时可能错位。

还应保存 learner 消费信息：`consumed_at`、`learner_model_version`、ratio 分位数、masked-token count、Critic target 版本和是否进入 actor update。这些字段支持事故追踪和离线重放。

### 25.1 一条轨迹从生成到训练的状态机

仅有最终 JSON 不足以支撑异步系统恢复。轨迹在数十分钟内会跨越调度器、rollout worker、sandbox、reward service、ready queue 和 learner，任何一步都可能重试。建议把样本生命周期建模为显式状态机：

```text
scheduled
  -> generating
  -> awaiting_reward
  -> validated
  -> queued
  -> leased_by_learner
  -> critic_updated
  -> actor_updated
  -> committed
```

失败分支进入 `quarantined`、`expired` 或 `cancelled`，而不是直接删除。状态转换要携带 `attempt_id`、幂等键和时间戳。rollout worker 重启后再次上传同一 `trajectory_id` 时，队列应识别重复；learner 获得 lease 后崩溃，lease 到期可以重新消费，但不能在不知道 optimizer 是否提交的情况下重复更新。

这要求 optimizer step 具有可审计的提交边界。可以把一个训练批次标识为 `training_batch_id`，在 checkpoint metadata 中保存其最大 queue offset。只有 Actor、Critic、optimizer、scheduler 和 batch offset 都持久化成功，样本才从“已租用”转为“已提交”。否则恢复时可能重复消费一批高 reward 轨迹，造成难以发现的数据权重偏移。

### 25.2 Token 对齐是比 Schema 更严格的不变量

训练前应执行逐样本一致性检查：

1. 重新 tokenize 结构化消息，结果与保存的 `token_ids` 完全一致；
2. action span 覆盖的 Token 数与 `rollout_logprobs` 可学习位置一一对应；
3. observation span 不重叠、不越界，并覆盖所有环境事件；
4. turn boundary 单调递增，terminal 后没有新的 action；
5. policy version 能解析到不可变的权重 digest，而不是只靠可复用的模型名称；
6. reward 生成时间晚于轨迹完成时间，且 verifier 输入 digest 与轨迹 digest 一致。

若 chat template 在 rollout 后升级，即使文本完全相同，BOS、role token、tool schema 或换行方式也可能改变 Token 序列。此时最安全的策略是使用原始 Token 训练并保留当时的 attention/position 语义；不能用新模板重新编码后把旧 log-prob 按长度强行对齐。DIS 对 behavior probability 的要求使这类“看似只是格式差异”的问题直接上升为算法正确性问题。

### 25.3 Reward 也要版本化

Agentic reward 常由单元测试、答案 verifier、格式检查、工具安全规则和 LLM judge 共同组成。每个组件都可能升级。记录最终 `reward=1` 而不记录组成项，会让历史样本在规则变化后无法解释。

建议保存 `reward_spec_version`、各项原始输出、聚合公式、judge model/version、测试镜像 digest 和执行日志摘要。若之后修复 verifier bug，可以离线重算尚未训练的样本，并标记已训练样本受到的范围；如果奖励只剩一个标量，就只能回滚整个时间窗口。论文的 benchmark reward 相对清晰，但真实在线 Agent pipeline 的奖励治理通常比优化器本身更复杂。

## 26. 实验设置精读

数学推理实验从 `Qwen3-30B-A3B-Thinking-2507` 出发，先在 GPT-OSS-120B 生成的 Tool-Integrated Reasoning（TIR）数据上 SFT 3 个 epoch，再用这个模型初始化 policy 和 value model。TIR 要求模型在自然语言推理中穿插 Python 工具调用。

| 配置 | 数值 |
| --- | --- |
| SAO policy batch size | 128 trajectories |
| SAO group size | 1 rollout per prompt |
| GRPO 对照 | 16 prompts × 8 rollouts = 128 |
| 最大长度 | 128K Token |
| Policy learning rate | $1\times10^{-6}$ |
| Value learning rate | $5\times10^{-6}$ |
| Value warmup | 10 steps |
| Critic updates per policy step | $K=2$ |
| Reasoning DIS | $\epsilon_l=0.3,\epsilon_h=5.0$ |
| Coding DIS | $\epsilon_l=0.8,\epsilon_h=3.0$ |
| Reasoning max turns | 50 |
| SWE-Bench max turns | 300 |
| Sampling | top-$p=1.0$, temperature $1.0$ |

SWE-Bench Verified 直接从 Qwen3-30B-A3B-Thinking-2507 开始 RL，并使用 OpenHands scaffold。数学评估中，AIME2025、HMMT、IMOAnswerBench 平均 16 次运行，BeyondAIME 平均 4 次运行。

这些重复次数主要降低评估采样方差，不等于训练运行了 16 个独立 seed。论文没有披露独立 RL 训练 seed、训练 GPU 数、节点拓扑、wall-clock 时间或 rollout/learner 配比。因此不能根据 evaluation mean 推断训练结果的 seed 稳定性。

## 27. Table 1：数学推理主结果

![SAO math reasoning results](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table01-math-reasoning-results.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 1, CC BY 4.0.*

为了便于检索，核心行转录如下。

| 模型/训练策略 | AIME2025 | BeyondAIME | HMMT Nov 2025 | IMOAnswerBench |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B SFT（w/ Python） | 80.4 | 53.3 | 75.2 | 53.3 |
| GRPO（w/ Python） | 84.2 | 54.8 | 76.0 | 55.8 |
| SAO | **97.3** | **74.8** | **88.3** | **74.0** |
| SAO（DIS only） | 94.2 | 71.5 | 86.7 | 71.3 |
| GRPO + DIS | 93.5 | 70.8 | 84.0 | 70.0 |

SAO 相对 SFT 的绝对提升分别为 `+16.9`、`+21.5`、`+13.1` 和 `+20.7` 个百分点。相对标准 GRPO 的提升为 `+13.1`、`+20.0`、`+12.3` 和 `+18.2` 个百分点。

表中还列出 Claude-Sonnet-4.5、GPT-5 High 和 GLM-4.7。这些是能力参照，不是严格受控的训练算法 baseline：模型参数、数据、推理预算、工具集和系统提示都不同。不能从 SAO 的 AIME `97.3` 高于 GPT-5 High 的 `94.6` 推导“SAO 模型整体强于 GPT-5”。

另一个需要解释的现象是 Qwen3 基础模型 `w/ python` 行分数很低，而 `w/o python` 行较高；SFT 后情况发生变化。论文没有充分解释不同模式的模板与工具可用性差异。主结论应建立在同一 TIR/SFT/工具设定下的 SFT、GRPO、GRPO+DIS 和 SAO 对比上，而不是混合所有行。

## 28. Table 2：SWE-Bench Verified

![SAO SWE-Bench Verified results](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table02-swe-bench-verified.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 2, CC BY 4.0.*

| 方法 | Accuracy |
| --- | ---: |
| Qwen3-30B-A3B | 23.0% |
| + GRPO（w/ DIS） | 27.0% |
| + SAO | **29.8%** |

SAO 相对基础模型提升 `6.8` 个百分点，相对 GRPO+DIS 提升 `2.8` 个百分点。后一个差距比数学 benchmark 小，但仍说明在相同 DIS 稳定机制下，single rollout + value design 提供了额外收益。

SWE-Bench 结果强依赖 Agent scaffold、工具权限、上下文管理、测试超时和容器。论文使用 OpenHands，最多 300 turns，128K context。这个 `29.8%` 不能直接迁移到 Claude Code、SWE-agent 或不同 OpenHands 版本，也不能说明生产代码 Agent 的所有任务成功率。

## 29. Fig. 3：一千步训练曲线

![SAO training performance](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig03-training-performance.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 3, CC BY 4.0.*

Fig. 3 展示 AIME2025、BeyondAIME 和 HMMT Nov 2025 随训练 step 的准确率。Vanilla GRPO 在早期出现崩溃，论文称约 160 step；表中 GRPO 分数是崩溃前最后有效值。GRPO+DIS 能继续训练，说明直接双边门控本身就解决了重要稳定性问题。

SAO 与 GRPO+DIS 在约 400 step 前相近，之后逐渐拉开。这一形状支持两层结论：

1. **稳定到不崩溃主要依赖 DIS**；
2. **长期效果差异更多来自 single rollout 和 value model 设计**。

如果只比较最终分数，容易把所有增益都归给 single rollout。曲线说明 DIS 是必要基座，SAO 的额外设计在训练中后期体现。

图也有局限。横轴是 optimizer step，不是 wall-clock time；GRPO 每步的数据组织与 SAO 不同，即使 global sample count 都是 128，rollout 等待、数据新鲜度和训练耗时仍可能不同。论文标题强调异步优化，却没有给同等时间预算曲线。

## 30. Table 3：Value 训练策略与更新频率

![SAO value training ablation](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table03-value-training-ablation.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 3, CC BY 4.0.*

| 配置 | Value 参数策略 | 每次 Policy 对应 Critic 更新 | AIME2025 | BeyondAIME |
| --- | --- | ---: | ---: | ---: |
| SAO | Frozen Attention | 2 | **97.3** | **74.8** |
| Single-step-update | Frozen Attention | 1 | 95.00 | 69.75 |
| Full-Parameter Value Training | Full parameter | 2 | 90.62 | 74.50 |

把 Critic 更新从 2 次减到 1 次，BeyondAIME 下降约 `5.05` 个百分点，说明难题分布下 Critic 追踪速度尤其重要。全参数训练在 AIME 上下降 `6.68`，BeyondAIME 几乎不变，说明冻结 Attention 的收益并非所有 benchmark 一致。

表中只比较三个离散点，没有给 Critic 训练计算开销。两次更新提高效果，也会增加 learner FLOPs；若系统瓶颈已经从 rollout 转移到训练，最优 $K$ 可能不同。

## 31. Table 4：完整消融与 Baseline

![SAO main ablation](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table04-main-ablation.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 4, CC BY 4.0.*

| 方法 | AIME2025 | BeyondAIME |
| --- | ---: | ---: |
| SAO | **97.3** | **74.8** |
| SAO w/o Faster Value | 95.0 | 69.8 |
| SAO w/o Frozen Attention | 90.6 | 74.5 |
| Vanilla VAPO（w/o DIS） | 91.3 | 69.0 |
| Running Mean Baseline | 79.8 | 55.3 |

Vanilla VAPO 在约 90 step 崩溃，说明把一个同步 reasoning RL 方法直接放到异步 single-rollout 环境中并不可靠。Running mean 用每个 prompt 最近 8 个 reward 的均值作 baseline，避免训练 Critic，但结果明显落后。

这里的 running mean 与后面在线实验不是同一个窗口：本消融讨论的是每个 prompt 最近 8 个 reward；在线模拟使用最近 128 个 reward 的滑动窗口。正文必须分开描述，否则会制造“论文窗口到底是 8 还是 128”的表面矛盾。

## 32. Fig. 4：训练动态为什么支持这些设计

![SAO training dynamics](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig04-training-dynamics.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 4, CC BY 4.0.*

Fig. 4(a) 使用 Explained Variance 衡量 value prediction 与 return 的一致程度：

$$
\operatorname{EV}
=1-\frac{\operatorname{Var}(R-V(s))}{\operatorname{Var}(R)}.
$$

EV 越高，说明 Critic 越能解释 return 方差。两次 Critic 更新的 SAO 在约 400 step 后高于单次更新，和主结果开始分化的时间大致一致。这提供了“Critic 追踪改善推动中后期性能”的机制证据。

Fig. 4(b) 比较全参数 Critic 与 Frozen-Attention Critic 的 gradient norm。全参数版本上升到更高且更波动的区间，冻结 Attention 后曲线更低、更平滑。梯度范数较小不自动等于模型更好，但结合 Table 3 的准确率，它支持正则化解释。

Fig. 4(c) 比较 SAO DIS 与没有 DIS 的 VAPO。VAPO 的 clip ratio 近乎为零，这不是好消息，而是说明它没有识别并屏蔽异步 drift；随后训练崩溃。SAO 的 masked ratio 会动态升高，在高风险阶段拒绝更多 Token。

图中的绝对 clip ratio 最高约 `0.006`，比例很小却足以影响稳定性。这提示极端 Token 可能具有不成比例的梯度影响。生产监控应同时报告数量比例和被屏蔽 Token 原始 ratio/advantage 的尾部统计。

## 33. 在线学习模拟：为什么 Single Rollout 更自然

真实在线交互通常不能为同一用户状态同时生成 8 条 response，再让用户分别体验和评分。一次请求产生一次实际轨迹和一次反馈，天然更接近 single rollout。

论文构造了一个可控的写作风格模拟。模型从 Academic、Cute、Chuunibyou、Classical 中选择风格，环境偏好分阶段切换。GLM-4.7 作为 LLM judge，分别判断回答质量与风格一致性，最终奖励是两个二值信号的乘积：

$$
r=r_{\text{quality}}\times r_{\text{style}},
\qquad
r_{\text{quality}},r_{\text{style}}\in\{0,1\}.
$$

乘法意味着任一条件失败，最终 reward 都为 0。它制造了清晰的非平稳分布，便于观察模型能否在偏好切换后重新对齐。

对照方法维护最近 128 个 reward 的滑动窗口：

$$
b_t=\mathbb E[r_{t-127:t}],
\qquad
\hat A_t=r_t-b_t.
$$

窗口 baseline 不依赖 prompt group，但环境突变后仍混有旧偏好 reward，存在惯性。Critic 理论上可根据当前状态和新数据更快变化。

## 34. Fig. 5：非平稳偏好下的适应

![SAO online learning simulation](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig05-online-learning.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 5, CC BY 4.0.*

左图展示各写作风格在 held-out set 上的准确率。灰色区域是偏好切换，原本占优的风格快速下降，新目标风格上升。它证明 SAO 至少能在受控、离散且奖励可验证的风格任务中跟随环境变化。

右图比较 SAO 与 running mean 的训练 reward。每次分布切换后两者都会下降，SAO 恢复更快且稳定值更高；running mean 因历史窗口仍包含旧奖励而滞后。

不能把这个实验写成“SAO 已验证真实在线学习”。它没有真实用户、隐私约束、反馈延迟、恶意反馈、偏好冲突、灾难性遗忘或安全目标；reward shift 是人为设计的三阶段变化，judge 也是单一 GLM-4.7。论文 limitations 明确要求真实用户在线适应增加 safeguards、monitoring 和 privacy review。

## 35. Fig. 6 与 Table 5：Token 还是 Agent Step

![SAO token-level versus step-level training](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-fig06-token-vs-step-training.webp)

*Source: Hou et al., arXiv:2607.07508v1, Fig. 6, CC BY 4.0.*

Fig. 6 比较主线 Token-level SAO、Step-level Average 和 Step-level Last-Token。三条曲线早期接近，约 150 step 后 Token-level reward 持续上升，两个 step variant 停留在较低区间。

![SAO action granularity ablation](/images/blog/sao-single-rollout-asynchronous-agentic-reinforcement-learning/sao-table05-action-granularity.webp)

*Source: Hou et al., arXiv:2607.07508v1, Table 5, CC BY 4.0.*

| Action 粒度 | AIME2025 | BeyondAIME |
| --- | ---: | ---: |
| Step-level（Average） | 85.8 | 60.5 |
| Step-level（Last-Token） | 87.3 | 62.8 |
| Token-level | **89.8** | **66.8** |

Table 5 在相同 400 training steps 下比较。Last-Token 比 Average 好，说明一轮末尾表示可能比整轮平均更接近完整语义；Token-level 继续领先，说明在长推理链中，细粒度 credit assignment 仍有价值。

这里的 Token-level 最终数值低于 Table 1 的 `97.3/74.8`，因为 Table 5 只训练 400 step，而主结果运行到约 1000 step。不能横向把两个表当作同一步数结果。

## 36. “稳定训练一千步”的正确读法

摘要强调 SAO 能稳定训练一千步。这个表述至少包含三层含义。

第一，训练没有像标准 GRPO 或 VAPO baseline 那样在约 160/90 step 发生明显 performance collapse。第二，关键 benchmark 在中后期仍总体上升。第三，Critic 的 EV、gradient norm 和 masked-token ratio 没有显示失控。

它不意味着：

- loss 在每一步单调下降；
- 所有 seed 都能稳定 1000 step；
- 1000 step 是理论稳定上限；
- 更大模型或更长训练自动稳定；
- 训练过程没有短期回撤；
- 同步 GRPO 在其他实现里一定会崩溃。

论文没有报告多个训练 seed 的 collapse rate，也没有展示超过 1000 step 的行为。更严谨的结论是：“在作者给出的 Qwen3-30B-A3B Agentic reasoning/coding 设置中，SAO 的代表性训练运行持续约 1000 step，并优于文中异步 GRPO/VAPO 对照。”

## 37. GLM-5.2 部署声明的边界

论文摘要称 SAO 已成功用于开放 GLM-5.2（750B 总参数、40B 激活参数）的 Agentic RL pipeline。这是重要的生产采用信号，因为它说明方法并非只在 30B 实验模型上构想。

但公开证据仍有边界：

| 维度 | 公开实验 | GLM-5.2 声明 |
| --- | --- | --- |
| 模型 | Qwen3-30B-A3B | GLM-5.2 750B-A40B |
| Benchmark | 数学、SWE-Bench、模拟写作 | 未给 SAO 专项对照 |
| 超参数 | 部分公开 | 未公开 |
| 训练硬件 | 未披露 | 未披露 |
| Critic 结构 | Frozen Attention + MoE | 未具体披露 |
| 代码 | 未公开 | 未公开 |
| 吞吐/成本 | 未报告 | 未报告 |

因此，正文可以写“论文报告已部署于 GLM-5.2 训练”，不能写“论文在 750B 上验证了 Table 1 的提升”或“开源 GLM-5.2 可直接复现 SAO”。模型权重只包含训练后的 policy，不包含生成它的异步队列、value model、rollout log、reward pipeline 和 optimizer 状态。

## 38. 论文没有公开的系统实现

截至本文写作时，arXiv 页面、TeX source 和公开代码搜索没有提供 SAO 官方仓库。论文也没有指明具体基于 slime、AgentRL、AReaL、ROLL 或其他框架实现。即使同一机构已有异步 RL 基础设施，也不能据此把某个目录或配置当作 SAO 源码。

公开缺口主要包括：

- rollout/learner 的 GPU allocation；
- rollout engine 与 training engine；
- 参数同步频率和传输协议；
- ready queue 的采样与 staleness policy；
- trajectory packing 与 128K 上下文内存策略；
- value model 的初始化、head 和预训练数据；
- Token log-prob 的精确语义；
- reward/verifier 的实现；
- checkpoint、resume 和异常轨迹处理；
- 训练吞吐、集群规模和成本。

本文因此使用“论文-系统实现边界”，而不是“论文-代码对照”。后面的工程清单描述实现 SAO 所需的接口，不宣称它们就是作者内部系统。

## 39. 与 THUDM/slime 的关系

[slime](https://github.com/THUDM/slime) 是 THUDM 开源的 LLM post-training framework，连接 Megatron-LM 训练与 SGLang rollout，并提供 data buffer、custom generation、reward 和 fully asynchronous examples。它是理解 SAO 可能需要哪些系统能力的重要参考。

从能力映射看：

| SAO 需求 | slime 可参考能力 | 是否等于 SAO 官方实现 |
| --- | --- | --- |
| 训练/生成解耦 | Megatron training + SGLang rollout | 否 |
| 异步样本到达 | data buffer / fully async example | 否 |
| Agent 多轮环境 | custom generation function | 否 |
| 参数同步 | framework weight update path | 否 |
| Reward | custom reward interface | 否 |
| DIS/Skip-Observation GAE | 需专门核验或实现 | 否 |

slime 官方材料说明它支撑多个 GLM 系列的 RL scaling，但 SAO 论文没有给出代码 permalink 或配置。文章只能把 slime 写作“可承载类似数据流的相关基础设施”，不能写成“SAO 基于 slime 的某个现成算法开关”。

## 40. 与 AgentRL、AReaL 和 ROLL 的关系

[AgentRL](https://github.com/THUDM/AgentRL) 关注多轮、多任务 Agentic RL，作者列表和 SAO 有交集。它能帮助理解 environment、trajectory 和 task orchestration，但不是 SAO 代码发布。

[AReaL](https://github.com/inclusionAI/AReaL) 是论文相关工作中的大型异步 RL 系统，强调 rollout 与训练完全解耦、staleness-aware training 和多类 Agent 工作流。它主要回答“如何高效组织异步集群”；SAO 更聚焦“单轨迹情况下如何做稳定的优化和 value estimation”。

[ROLL](https://github.com/alibaba/ROLL) 及 ROLL Flash 关注 RLVR/Agentic training 的并行与异步效率，包括长尾 rollout、角色分离和训练推理后端。它与 SAO 的关系同样是互补：系统并行可以提高吞吐，SAO 的 DIS/Critic/GAE 处理随之而来的优化问题。

这三类工作构成一个有用的分层：

```text
agent environment / scaffold
        |
        v
rollout and reward orchestration
        |
        v
asynchronous queue + parameter synchronization
        |
        v
SAO-style off-policy gate + critic + GAE
        |
        v
distributed optimizer and checkpoint
```

只实现最下面的 loss 不会自动得到高效 Agentic RL 系统；只实现上面的异步队列，也不会自动获得稳定策略更新。

## 41. 一个可实施的异步训练控制面

如果要在成熟训练平台中实现 SAO，建议把控制面拆成六个可独立观测的组件。

### 41.1 Prompt Scheduler

负责从训练池选 prompt，确保一条 prompt 在当前阶段只创建一个 rollout。它还要处理 domain mixture、难度分桶、重复采样间隔和数据去重。

### 41.2 Rollout Worker

加载某个明确的 policy version，与环境交互，逐 Token 保存行为 log-prob。worker 不应在未记录版本变化的情况下热切权重；更安全的方式是一条 active trajectory 固定版本，下一条轨迹再切换。

### 41.3 Ready Queue

保存完成轨迹并记录 `completed_at`。队列应支持最大 age、版本 lag 门禁、任务配额和异常样本隔离。不能简单 FIFO 到底，因为极旧轨迹可能大量触发 DIS mask。

### 41.4 Critic Learner

根据结构化 action/observation mask 计算 value target。它与 Actor 使用相同 backbone 还是独立部署、是否共享 embedding、如何冻结 Attention，都需要显式配置和 checkpoint。

### 41.5 Policy Learner

重算 current log-prob，构造 ratio，应用双边门控，再进行 distributed optimizer step。policy learner 必须验证 rollout 与 current Token 序列一致，发现错位应拒绝整条样本而不是静默补齐。

### 41.6 Weight Publisher

把新参数发布给 rollout workers，并维护单调递增的 model version。publisher 应区分“optimizer step 已完成”“checkpoint 已持久化”“rollout engine 已加载”三个状态。

## 42. 参数同步与轨迹一致性

异步系统常见两种同步策略。

**Trajectory-bound snapshot**：一条轨迹从开始到结束使用同一 policy version。统计语义最清楚，但超长轨迹会长期占用旧权重实例，更新传播慢。

**Mid-trajectory refresh**：在 Agent turn 或 Token block 边界加载新权重。数据更鲜，但一条轨迹由多个 behavior policy 混合生成，必须逐 Token 或逐 span 保存 version/log-prob，KV cache 一致性也更复杂。

SAO 的 DIS 依赖 Token rollout log-prob，理论上可处理混合 behavior policy；论文却没有明确生产实现是否 mid-trajectory refresh。复现第一版应优先 trajectory-bound snapshot，减少变量。只有在证明长轨迹造成不可接受 staleness 后，再引入 turn-boundary refresh。

无论哪种策略，参数发布都要防止部分加载。一个 worker 不能让一半层来自版本 438、另一半来自 439。需要原子切换、checksum 和加载完成 ack。

## 43. Queue Policy：异步不等于无限接受旧数据

DIS 可以屏蔽极端 Token，但不应成为无限 replay 的借口。ready queue 至少应有三层门禁：

1. **结构门禁**：Token、log-prob、mask、reward 和终止状态完整。
2. **新鲜度门禁**：trajectory age 或 model version lag 不超过上限。
3. **统计门禁**：预检 ratio 的有效 Token 比例足够高。

可以定义轨迹有效率：

$$
q(\tau)
=\frac{\sum_t m_t^{\text{action}}m_t^{\text{DIS}}}
{\sum_t m_t^{\text{action}}}.
$$

若 $q(\tau)$ 太低，整条轨迹虽然形式完整，实际 policy gradient 信息已经很少。系统可以丢弃、降权或仅用于 Critic；具体策略需实验决定。

还要防止 completion-order bias。短、简单轨迹更早进入队列，如果完全按完成顺序训练，数据分布会偏向短任务。可按 domain、长度和难度设置 reservoir 或配额，同时保持足够新鲜。

### 43.1 完成顺序偏差不是普通 Shuffle 能自动消除

同步 batch 在所有样本完成后再 shuffle；异步 learner 看到的是“谁先完成谁先进入候选池”。假设简单数学题平均 20 秒，困难题平均 8 分钟，即使 prompt scheduler 以 1:1 发出，训练前半小时的 ready queue 也会严重偏向简单题。随着策略学习，轨迹时长本身还会变化：模型更会使用工具后，可能因为尝试更多步骤而变慢，completion order 因而与策略能力相关。

一种保守做法是在很短的时间窗内做分层缓冲：按 task、长度预测桶和难度桶保留配额，learner 从各桶采样，同时给每条轨迹设置最大 age。窗口过小无法校正偏差，窗口过大又增加 policy lag，所以这是一个需要用 wall-clock 与 ratio 数据共同调节的系统参数。

也可以记录调度分布 $p_{\text{scheduled}}(x)$ 与实际消费分布 $p_{\text{consumed}}(x)$，对 task-level 失衡做采样权重修正。但不要轻易按完整轨迹完成概率做高方差 importance weighting；SAO 的 DIS 已在 Token 层处理 policy drift，再叠加无界数据分布权重可能重新引入不稳定。第一版更适合使用有上限的配额和监控，而不是复杂的二重校正。

### 43.2 Mask Ratio 要转成有效训练量

若 global batch 固定为 128 条轨迹，但其中一半 action Token 被 DIS 屏蔽，那么“每步 128 条”已不能代表实际梯度规模。可以同时统计：

$$
N_{\text{effective}}
=\sum_{\tau\in\mathcal B}\sum_t
m^{\text{action}}_{\tau,t}m^{\text{DIS}}_{\tau,t},
$$

以及按轨迹归一化后的有效率。loss denominator 应明确使用有效 Token 数、原 action Token 数还是轨迹数；三者会改变 mask ratio 波动时的梯度尺度。若 denominator 仍是全部 action Token，大量屏蔽会自动减小更新；若只除有效 Token，剩余少数 Token 的平均梯度可能反而被放大。

论文没有披露这个归约细节。复现时应固定一种定义，并在有效 Token 极少时跳过 optimizer step，而不是让除数接近零。还应报告“每 GPU-hour 有效 action Token”而非只有原始 rollout Token，因为超长 observation 和被屏蔽 Token 都不直接贡献 Actor 梯度。

## 44. 必须监控的训练指标

仅看 reward 和 benchmark 不足以排查异步训练。建议指标分成五组。

### 数据与队列

- ready queue depth、等待时间 P50/P95/P99；
- trajectory duration 与 turn/token 长度分布；
- completion-order 对 task/difficulty 的偏差；
- malformed、timeout、cancelled trajectory 比例；
- rollout-to-learner version lag 和 wall-clock age。

### Importance Sampling

- $\log r_t$ 的 P1/P50/P99；
- 上界、下界分别 mask 的 Token 比例；
- 每条轨迹有效率 $q(\tau)$；
- mask ratio 按 turn、Token position、reward、task 分桶；
- rollout/training engine 同版本 log-prob 差异基线。

### Critic

- value loss、Explained Variance、gradient norm；
- predicted value 按 reward label 的分布；
- return calibration、value saturation；
- Critic/Actor 更新耗时和样本吞吐；
- Frozen 与可训练参数数量、专家负载。

### Policy

- policy loss、entropy、KL 或行为漂移；
- action Token 数和有效梯度 Token 数；
- reward、pass rate、长度和工具调用频率；
- collapse detector 与最近稳定 checkpoint；
- 各 benchmark 的独立回归曲线。

### 系统效率

- rollout GPU utilization、learner GPU utilization；
- tokens/s、trajectories/hour；
- 参数同步耗时与失败率；
- sandbox/tool latency；
- 同等 wall-clock 和同等 GPU-hour 下的有效提升。

论文主要覆盖 policy 效果与少数 Critic 动态，没有覆盖完整生产观测面。

## 45. 稳定性门禁与自动回滚

异步训练崩溃可能比同步训练更快扩散，因为坏策略一旦发布，就会同时污染后续 rollout。生产系统应把模型发布与 optimizer step 解耦。

一个保守流程是：

```text
optimizer step
  -> numerical checks
  -> short held-out verifier checks
  -> ratio/mask/entropy guardrails
  -> checkpoint persisted
  -> canary rollout workers
  -> broader rollout publication
```

建议设置的 kill conditions 包括：

- loss/gradient 出现 NaN 或 Inf；
- masked-token ratio 突然超过历史分位数；
- EV 连续下降并低于 0；
- reward 上升但格式、工具安全或 held-out pass rate 下降；
- action length 爆炸；
- rollout engine 与 learner 同版本 log-prob 对不齐；
- 新版本在 canary 环境造成异常工具行为。

回滚不能只恢复 Actor。Critic、optimizer、scheduler、reward model/version 和 queue watermark 都要与 checkpoint 对齐，否则会用旧 Actor 配新 Critic 或消费由坏版本生成的数据。

### 45.1 一次异步训练事故应如何定位

假设 reward 在 30 分钟内快速下降，同时 masked-token ratio 从 8% 升到 45%。排查顺序应从数据与版本事实开始，而不是立即调学习率。

1. **冻结发布**：停止向新 rollout 发布模型，但允许当前轨迹完成并进入隔离队列。
2. **锁定时间窗**：记录第一个异常 optimizer step、checkpoint、policy version 和对应 queue offset。
3. **检查同版本 log-prob**：若同版本 ratio 已偏离 1，优先排查 tokenizer、采样温度、推理/训练 kernel 或权重加载。
4. **检查 lag 分布**：若同版本正常但 queue age 激增，排查 rollout 变慢、learner 加速、参数发布频率或网络拥塞。
5. **检查 Critic**：观察 EV、value loss、gradient norm 和 value target 分布，区分 baseline 崩溃与 Actor drift。
6. **检查 reward**：确认 verifier、测试镜像、judge 模型和聚合规则是否在异常前变更。
7. **离线重放**：用最后稳定 checkpoint 重算异常批次的 current log-prob、DIS mask 和 advantage，复现差异。

若问题来自 rollout engine 与 learner 的 log-prob 语义不一致，只回滚模型不会解决；若问题来自某个坏 policy version，则必须丢弃或重新标记由它生成且尚未消费的轨迹；若问题来自 reward service，则已经进入 optimizer 的样本影响需要按 checkpoint 区间评估。

### 45.2 Canary 不应只看平均 Reward

新版本先发布给少量 worker 时，应对固定探针 prompt 和真实分布 prompt 同时运行。固定探针用于比较 action format、工具调用和 log-prob；真实样本用于发现分布外行为。门禁至少包含：

- 同版本 rollout/learner log-prob 差异；
- 新旧版本的 action length、tool-call count 与 failure taxonomy；
- verifier pass rate 和 held-out benchmark；
- DIS 预估 mask ratio；
- 安全规则、敏感工具和 sandbox 越权检查；
- P95/P99 trajectory duration，防止质量看似上升但长尾时延失控。

Agentic RL 的策略升级会同时改变答案、工具行为和系统负载。只看 reward 可能把“无限尝试直到成功”的策略当成进步，却让每条轨迹成本成倍增长。部署门禁应把质量、延迟、工具安全和有效训练量放在同一张版本卡片中。

## 46. 复现路线：先证明数据正确，再追求结果

不应一开始就用 30B MoE、128K context 和 300-turn coding Agent。更稳妥的复现分四阶段。

### 阶段 0：离线算子测试

- 构造短 action/observation 序列；
- 验证 Skip-Observation next index；
- 人工设置 current/rollout log-prob，验证上下界门控；
- 确认 observation 永远不进入 policy loss；
- 验证 padding、terminal 和 truncation。

### 阶段 1：同步 Single-Rollout

先关闭异步，使用单 rollout + Critic。若同步都不稳定，不要把问题归因于 queue lag。比较 raw reward、running mean、step value 和 token value。

### 阶段 2：受控异步

固定最大 lag 为 1-2 个版本，记录 ratio 分布。验证 DIS 打开后能处理轻度 staleness，再逐步增加 rollout/learner 解耦程度。

### 阶段 3：长时域 Agent

加入工具调用、环境 observation、长度长尾和真实 verifier。最后才扩大模型、上下文和集群规模。

每个阶段都要保留同步 PPO/GRPO 或同等强度 baseline。否则即使 reward 上升，也无法判断收益来自 single rollout、Critic、数据变更还是更多计算。

## 47. 最小单元测试清单

实现 SAO 时，以下测试比先跑 benchmark 更重要。

| 测试 | 预期结果 |
| --- | --- |
| ratio 正好在下界 | 按论文严格不等式，应被 mask |
| ratio 正好在上界 | 应被 mask |
| ratio 在区间内 | 保留原 ratio，不截断到 1 |
| observation Token | policy loss 始终为 0 |
| action 后接长 observation | GAE 跳到下一 action 首 Token |
| terminal action | 不访问不存在的 next action |
| 同版本双引擎重算 | log-prob 差异在允许误差内 |
| Tokenizer 版本不匹配 | 整条样本拒绝，而非静默训练 |
| reward 缺失 | 进入 quarantine，不写默认 0 |
| Critic Frozen Attention | Attention 参数梯度为空且权重不变 |
| K=2 | 每个 policy step 准确执行两次 value optimizer step |
| resume | Actor/Critic/queue watermark 一致恢复 |

严格不等式边界是容易漏掉的细节。论文函数写的是 $1-\epsilon_l<x<1+\epsilon_h$；若实现使用 `<=`，边界 Token 行为会不同，虽然实际浮点命中概率可能很低，仍应明确。

## 48. 与 PPO、GRPO、RLOO、SPO 的统一比较

| 方法 | Baseline 来源 | 每 prompt rollout 数 | 是否需要 Critic | 异步适配关键点 |
| --- | --- | ---: | --- | --- |
| PPO | Learned value | 1+ | 是 | 需要 off-policy correction |
| GRPO | Prompt 内 group statistics | 多条 | 否 | group barrier 明显 |
| RLOO | Leave-one-out group baseline | 多条 | 否 | 仍依赖多样本 |
| Running mean | 历史 reward | 1 | 否 | 分布变化时有惯性 |
| SPO | 历史/难度等先验 | 1 | 视实现 | 依赖先验质量 |
| SAO | Token-level value model | 1 | 是 | DIS + Critic + Skip-Observation GAE |

SAO 并没有证明 learned value 永远优于无 Critic 方法。它证明的是，在论文的长时域、异步、单轨迹条件下，训练一个足够好的状态相关 Critic 比简单 running mean 更有效。

## 49. 与 VAPO、GSPO 和 IcePop 的关系

VAPO 面向高级 reasoning RL 的稳定训练，提供 length-adaptive GAE 等设计。SAO 继承其长度自适应思想，但论文中的 vanilla VAPO 没有 DIS，在异步 single-rollout 设置中约 90 step 崩溃。

GSPO 用 sequence-level importance ratio 改善 group policy optimization 的稳定性，重点仍是 group/sequence 目标。SAO 使用 Token-level rollout/current ratio，并处理 single rollout 和 Agent observation 边界。

论文称 DIS 与 Every Step Evolves 中的 IcePop 机制相似。共同点是识别并过滤严重 off-policy Token；SAO 进一步移除对 $\pi_{\theta_{\text{old}}}$ 的依赖，直接使用 rollout log-prob。文章不应把 DIS 写成完全从零出现的 clipping 思想，而应把创新放在其与 single-rollout async/value design 的组合。

## 50. 与 A3C、IMPALA 和异步 RLHF 的关系

A3C 早已展示并行 actor 异步更新的价值；IMPALA 进一步用 actor-learner 架构和 V-trace 修正 off-policy trajectory。SAO 面对的是 LLM 特有的超长自回归 Token、巨大模型、工具 observation 和 RL post-training 基础设施。

IMPALA 的 V-trace 使用截断 importance weight 构造修正 return；SAO 的 DIS 直接屏蔽超出区间的 Token，再配合 GAE 和 Critic。两者都承认异步数据需要 off-policy 控制，但目标和实现粒度不同。

Asynchronous RLHF、AReaL 和 ROLL Flash 更系统地研究训练/rollout 解耦和吞吐。SAO 的独特问题是 GRPO group 与单次真实反馈不匹配，因此把算法结构改回 single-rollout Actor-Critic。它是算法与系统共同设计，而不是首个异步 RL 系统。

## 51. 与本站长期时域 Agentic RL 文章的关系

本站已有的《长期收益强化学习算法研究进展》关注 sparse/delayed reward、时间信用分配、hierarchical RL、world model 和 Agent 长期行为。SAO 可以放在其中的“训练数据如何流动、优势如何跨环境 observation 传播”层。

两者解决的问题不同：

- 长期信用分配问“最终 reward 应归因给哪些早期决策”；
- SAO 问“长轨迹异步到达且每个 prompt 只有一个 rollout 时，如何稳定更新模型”；
- Skip-Observation GAE 提供局部的 action-to-action credit bridge，但没有解决跨数百 turn 的所有因果归因问题；
- DIS 控制数据新鲜度引起的 off-policy 风险，不判断某个早期工具调用是否真正导致最终成功。

因此，SAO 是长时域 Agentic RL 的训练机制之一，不是完整 credit assignment 答案。

## 52. 论文证据最强与最弱的部分

**证据较强的部分**：

- 在相同 Qwen3 backbone 和 batch sample count 下，SAO 明显高于 GRPO+DIS；
- DIS 能阻止文中标准 GRPO/VAPO 的早期 collapse；
- Critic 两次更新提高 EV 和结果；
- Frozen Attention 降低 gradient norm，并在部分 benchmark 提升准确率；
- Token-level action 粒度优于两个 step-level variant；
- 模拟在线环境中 Critic 比 running mean 更快适应。

**证据较弱或未覆盖的部分**：

- 没有 wall-clock speedup 或 GPU utilization；
- 没有多 seed collapse rate；
- 没有 policy lag/staleness 分布；
- 没有 value pretraining 数据和独立消融；
- 没有 dense model、小模型或短任务验证；
- 没有 750B GLM-5.2 的 SAO 对照；
- 没有真实在线用户实验；
- 没有代码和完整超参数。

这不否定论文贡献，但决定了我们能把哪些结论写成事实，哪些只能写成待验证工程假设。

## 53. 局限性与批判

### 53.1 效率是动机，不是本论文量化结果

论文反复说明异步可减少等待，却没有展示同步与异步的 tokens/s、GPU-hours 或 time-to-quality。SAO 还增加 Critic 和两次 value update，learner 侧成本更高。最终端到端是否更快取决于 rollout 占比和资源配比。

### 53.2 Frozen Attention 可能依赖 MoE 架构

Qwen3-30B-A3B 的 MoE projection 提供了可训练子空间。dense model 没有同样的“冻结 Attention、只训练专家”结构，迁移时需要 LoRA、FFN-only 或其他参数子集实验。

### 53.3 DIS 可能丢弃关键困难 Token

最 off-policy 的 Token 可能恰好来自高回报的新行为或困难状态。完全置零保护稳定性，也可能减慢对新策略区域的学习。论文没有比较 soft weighting、trajectory rejection 或 V-trace 类修正。

### 53.4 Completion-order Bias 未被讨论

轨迹完成即训练会偏向短任务。即使 prompt sampling 原本均匀，learner 实际看到的时间顺序也不均匀。对持续在线学习，这可能形成“简单任务更新更频繁”的隐性 curriculum。

### 53.5 Critic 预训练不可复现

论文承认 value cold start 是主要瓶颈，却未披露核心数据。这使单轨迹方法最重要的初始化条件无法独立验证。

### 53.6 在线实验过于受控

风格切换只有少数离散类别，reward 由单一 LLM judge 提供。真实用户偏好可能连续、冲突、迟到且包含安全敏感信息。

### 53.7 生产声明缺乏公开诊断

GLM-5.2 部署说明方法具备实际价值，但没有提供大模型上的稳定性曲线、吞吐、成本和事故率，外部读者无法审计。

## 54. 安全、隐私与在线学习治理

如果 SAO 用于真实在线反馈，训练系统会不断吸收用户交互。至少需要：

- PII 检测、脱敏和 retention policy；
- 用户是否同意训练的明确边界；
- 防 prompt injection 和 reward manipulation；
- 高风险工具行为的独立安全 verifier；
- 单用户或小群体反馈的权重上限；
- 防止短期偏好覆盖长期安全 policy；
- 可删除数据与对应 checkpoint 影响评估；
- 人工审核、canary 和紧急停止机制。

单轨迹在线学习的快速适应既是优点也是风险。若 reward 被攻击或错误 judge 持续给分，Critic 可能比 running mean 更快追随错误目标。训练稳定不等于目标安全。

## 55. 推荐阅读路径

第一次读论文，可以按以下顺序：

1. Abstract 与 Fig. 2，先理解为什么 group sampling 不适合异步。
2. Section 3.1，逐式读 DIS ratio 和校准函数。
3. Section 3.2，读 Faster Value、Frozen Attention 和 Skip-Observation GAE。
4. Table 1/2 与 Fig. 3，区分 DIS 的稳定性贡献和完整 SAO 的效果贡献。
5. Table 3/4 与 Fig. 4，检查作者给出的机制证据。
6. Section 4.5 与 Fig. 5，理解 online simulation 的适用范围。
7. Appendix A 的 Fig. 6/Table 5，理解 Token-level 与 step-level 边界。
8. Appendix B，最后核对作者自己承认的外推限制。

补充阅读建议依次为 PPO、DeepSeekMath/GRPO、VAPO、GSPO、Every Step Evolves/IcePop、IMPALA、Asynchronous RLHF、AReaL、ROLL Flash 和 MobileRL。

## 56. 结论

SAO 的长期价值不在于一个孤立的新 loss，而在于它给出了一套适合 Agentic RL 的一致设计：数据按单条轨迹自然到达；rollout 时保存行为概率；current learner 用双边 Token 门控控制异步 drift；状态相关 Critic 替代 prompt group；Critic 更快更新并限制可训练参数；GAE 显式跳过环境 observation。

论文实验支持三个可信判断。第一，GRPO 的 group barrier 与长时域异步 rollout 确实存在结构冲突。第二，极少量但极端的 off-policy Token 足以破坏训练，直接双边屏蔽能显著改善稳定性。第三，single rollout 要有效，Critic 不是可有可无的附件，其初始化、更新频率和参数策略决定最终效果。

论文也没有完成全部论证。它没有量化异步吞吐，没有开放训练代码，没有披露 value pretraining recipe，也没有在公开实验中验证 750B 部署细节。工程团队不能只抄三条公式就期待得到 GLM-5.2 级训练系统。

对实际项目，最合理的采用方式不是立即替换现有 PPO/GRPO，而是先建立 Token log-prob 与 action/observation 数据契约，在小模型上验证 Skip-Observation GAE 和 DIS 边界，再逐步打开异步队列。只有当 wall-clock time-to-quality、稳定性和任务效果同时优于基线，SAO 才真正完成从论文方法到生产训练方案的转化。

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

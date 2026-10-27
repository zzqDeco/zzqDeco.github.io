---
title: "Claude Code 精读：工具证据、上下文工程与减少模型幻觉的设计"
description: "基于独立架构分析与官方文档，精读 Claude Code 如何通过工具取证、编辑前置检查、上下文管理、任务验证和执行约束降低无依据推断与虚假完成风险"
pubDate: "2026-10-27T11:28:02+08:00"
updatedDate: "2026-10-27T11:28:02+08:00"
tags:
  - "Deep Reading"
  - "Claude Code"
  - "AI Agents"
  - "Context Engineering"
  - "LLM Reliability"
  - "Code Reading"
draft: false
---

编程助手最隐蔽的故障往往超越了立即报错的语法缺陷，表现为缺乏外部事实支撑的工程声明：声称某个函数存在，实际上仅凭命名惯例猜测；声称测试已经通过，实际运行的是修改前的工作树；声称已读过目标文件，上下文里只剩截断或压缩后的摘要片段；或者声称子 Agent 已完成独立验证，而子 Agent 仅仅复述了主 Agent 的总结。这类输出在语言形式上符合工程沟通规范，却缺失了将自然语言声明锚定到外部系统事实的关键证据链。

本文以独立架构分析仓库 [how-claude-code-works](https://github.com/Windy3f3f3f3f/how-claude-code-works/tree/f4d6505ed9162a0ee6be089190f74c419ecacb19)（固定提交 `f4d6505ed9162a0ee6be089190f74c419ecacb19`，2026-08-17）与 Anthropic 官方文档（2026-10-27 核验，对应公开 release [`v2.1.311`](https://github.com/anthropics/claude-code/releases/tag/v2.1.311)）为分析对象，系统拆解 Claude Code 如何通过工具取证、编辑前置检查、上下文压缩恢复、任务验证与扩展机制抑制无依据推断与虚假完成。需要明确证据边界：社区仓库提供了早期快照（约 `v2.1.6x`～`v2.1.88`）的架构拆解与后续版本（`2.1.201/202`）的黑盒观察，官方文档界定了当前公开接口规范，而文中的形式化模型、故障反例与验证合同则属于面向 Agent 架构设计的工程推演。

---

## 1. 错误分类、证据边界与五层可靠性分工

### 1.1 区分模型生成与运行系统控制

用户在终端看到的每一次响应，均由语言模型推理与宿主运行系统共同塑造。语言模型根据当前可见上下文预测下一段文本或工具调用请求；宿主运行系统则负责维护工具注册表、校验参数 Schema、核对文件状态与权限策略、执行外部命令并将结构化观测结果回传给模型。评价 Agent 的可靠性时，必须遵循三条核心判据：

1. **观察区别于理解**：读取文件只能证明特定字节进入了工具返回结果，无法自动保证模型正确解析了跨模块调用关系；检索到测试函数定义，也不代表测试运行器实际收集并执行了该用例。
2. **执行区别于生效**：命令进程退出、文件落盘或接口返回，仅代表某一阶段的动作完成，动作对系统产生的实际语义后果仍需独立检验。
3. **局部生效区别于完整交付**：单元测试通过可以支撑局部逻辑结论，但无法自动外推至生产部署、权限边界、跨服务集成或业务验收。

各资料来源在本文中的证据定位与适用范围如下表所示：

| 证据对象 | 本文固定方式 | 可以支持的判断 | 证据适用边界 |
| --- | --- | --- | --- |
| 独立分析仓库 | 固定 Git commit `f4d6505e` | 作者在该提交中的历史机制分析与图示 | 二手历史快照分析，不等同于当前产品源码逐行一致 |
| 后续黑盒材料 | 记录观察版本（`2.1.201/202`） | 特定版本与实验开关下出现的行为线索 | 受特性开关控制，不等同于所有环境默认启用 |
| 官方滚动文档 | URL 与查阅日期（`2026-10-27`） | 当前公开接口契约与使用说明 | 反映查阅时点的公开协议，不直接回填历史内部实现 |
| 官方 Release | [`v2.1.311`](https://github.com/anthropics/claude-code/releases/tag/v2.1.311)（2026-10-26） | 当前版本公开发布状态 | 仅作时间线参照，本文属于静态架构精读而非实测基准报告 |

### 1.2 编程任务中的事实性错误与状态声明失真

将一切不理想输出笼统归结为“模型幻觉”会掩盖具体的工程防线位置。在软件工程工作流中，需要重点治理两类直接违背外部事实的故障：一是**事实虚构**（捏造不存在的文件、类、工具或跨版本 API），二是**行动与状态失真**（未执行检查却宣告测试通过、异步进程尚未结束即汇报完成、用过期的工作树测试结果背书新修改的代码）。

| 错误类型 | 典型表现 | 首要核验依据 | 常见混淆情形 |
| --- | --- | --- | --- |
| 对象虚构 | 文件、类、工具或配置项不存在 | 仓库文件树、工具目录、依赖导出表 | 构建期动态生成的符号未被静态搜索覆盖 |
| 版本错配 | 调用了另一版本的接口或参数签名 | 锁文件（Lockfile）、本地类型声明、对应版本文档 | 同名接口跨主版本发生语义变更 |
| 结果误报 | 命令报错或零测试运行却汇报成功 | 原始标准输出/错误流、进程退出码 | 前置步骤编译成功但后续打包或断言失败 |
| 状态过期 | 用旧工作树的检查报告支撑新代码修改 | 内容指纹、文件修改时间戳、依赖快照 | Git `HEAD` 未变但未提交工作树已发生改动 |
| 无依据完成 | 跳过关键验收条件直接宣布任务交付 | 验收清单与实际工具执行记录 | 代码编写完成但环境验证尚未执行 |
| 普通逻辑缺陷 | 边界条件或算法分支处理错误 | 单元测试与运行时行为断言 | 代码逻辑有误但并未虚构外部事实 |
| 需求偏差 | 正确实现了偏离用户真实意图的功能 | 用户真实需求与显式审批记录 | 任务目标本身存在多义性 |
| 越权操作 | 执行了技术上正确但未经授权的破坏性命令 | 权限策略与用户确认记录 | 结果符合技术预期但违反安全边界 |

任何工程事实都具备明确的**时间有效性与覆盖范围**。“测试通过”必须绑定测试目录、筛选参数、目标工作树快照、依赖环境与实际执行的用例数量；“接口不存在”则受限于当前检索路径与锁定依赖范围。系统架构的核心任务，就是防止局部、静态或过期的观察被无条件放大为全局结论。

### 1.3 可靠性机制的五层协同架构

[![本文整理：证据、上下文、操作前置检查、执行验证与权限边界](/images/blog/claude-code-hallucination-reduction-design/claude-code-analysis01-reliability-layers.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-analysis01-reliability-layers.webp)

*来源：本文整理，2026-10-27。此图是分析框架，不是 Anthropic 官方实现图。*

如上图所示，完整的智能体可靠性体系沿着数据流与控制流划分为五个正交层次：

1. **取证层（Evidence Acquisition）**：通过文件读取、代码搜索、依赖查询与环境探测消除未知项，使模型决策建立在当前仓库事实之上。
2. **上下文层（Context Maintenance）**：在长会话组装与多级压缩中保留证据的来源标签、时间戳、失败条件与可回查路径，防止摘要抹平关键否定证据。
3. **操作前置检查层（Pre-execution Gate）**：在工具执行入口校验名称注册、参数 Schema、文件已读状态（Read-Before-Edit）、替换锚点唯一性与修改时间新鲜度，将一部分模型错误转化为确定性的程序拒绝。
4. **效果验证层（Outcome Verification）**：借助 LSP 诊断、类型检查、测试运行器、Hooks 与完成证据合同检验修改的实际后果；当验证不充分时驱动循环回退取证，而非直接放行完成声明。
5. **权限与安全边界层（Permission & Sandbox Boundary）**：通过权限规则、沙箱隔离与人工确认限制单次错误动作的破坏半径，防止外部不可信文本升级为执行指令。

不同控制形式在判定主体与保障边界上存在本质差异：

| 控制形式 | 判定主体 | 核心工程职责 | 能力边界 |
| --- | --- | --- | --- |
| 提示词与项目规则 | 语言模型 | 引导探索习惯、沟通规范与不确定性表达 | 属于软性行为引导，无法保证 100% 遵循 |
| 工具 Schema 校验 | 宿主程序 | 拦截未知工具名与非法参数类型/结构 | 无法判断合法字符串背后的业务语义是否正确 |
| 状态前置检查 | 宿主程序 | 阻止未读覆盖、歧义锚点替换与过期状态写入 | 仅覆盖受控工具入口，不提供跨进程全局事务隔离 |
| 独立模型审查 | 子 Agent / 裁判模型 | 从反例视角复核实现缺陷与遗漏边界 | 与主模型共享基础偏差，无法替代真实运行验证 |
| 测试与类型检查 | 编译器 / 测试运行器 | 验证已编码为静态类型或测试用例的程序性质 | 受限于测试覆盖率，无法捕获未写入规格的需求偏差 |
| 人工审批与验收 | 授权开发者 | 裁定高风险动作授权、业务取舍与最终交付接受度 | 受限于人工审阅精力与信息呈现完整度 |

---

## 2. Agent Loop、主动取证与工具执行入口

### 2.1 双层生成器循环与状态投影模型

[![仓库原图：QueryEngine 与 query 的双层生成器架构](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo01-agent-loop.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo01-agent-loop.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/02-agent-loop.md, 双层生成器架构, MIT. 非官方历史实现分析图。*

根据社区仓库 [Agent Loop 专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/02-agent-loop.md) 的历史快照分析，Claude Code 采用 `QueryEngine`（管理会话生命周期与用户输入）与 `query()`（组织单轮模型调用、工具执行、结果回填与续跑判定）的双层异步生成器架构。为精确分析证据流转，可将第 $t$ 轮主循环抽象为如下状态转移过程（本文形式化表述）：

$$
C_t=P(S_t),\qquad a_t\sim\pi_\theta(\cdot\mid C_t),
$$

$$
o_t=\operatorname{Execute}(a_t,S_t),\qquad
S_{t+1}=U(S_t,a_t,o_t).
$$

其中 $S_t$ 为系统完整运行状态（涵盖磁盘工作树、后台进程、文件读取缓存与任务列表），$P(S_t)$ 为投影函数，决定哪些信息真正进入第 $t$ 次模型调用的可见上下文 $C_t$；模型采样出拟议动作 $a_t$ 后，由执行器在状态 $S_t$ 下产生观测 $o_t$，并通过状态更新函数 $U$ 同步刷新上下文与内部元数据。这一形式化揭示了两个核心脆弱点：一是投影 $P$ 在截断或压缩时可能丢失关键约束，二是模型必须根据观测 $o_t$ 真实更新假设，并根据异步命令的最终状态（而非早期打印的阶段性日志）判定循环终止条件。

### 2.2 先搜索再判断：用环境观察替代名称联想

预训练知识能够为模型提供常见框架的目录惯例，但无法充当当前代码库的事实源。可靠的探索工作流通过 `Glob`（按路径模式收窄候选文件）、`Grep`（按正则表达式定位符号与调用点）与 `Read`（读取实际代码上下文）分三阶段收集最小充分证据：首先确认项目配置、锁定依赖版本与测试命令入口，接着沿真实数据流追踪核心调用链，最后核对受影响的所有调用方与边界条件。

在处理检索结果时，正向命中与负向未命中均需结合范围解释。文本命中可能落在测试夹具、注释或已废弃分支中，必须核对导出表与构建入口；单次搜索未命中则可能源于大小写、忽略规则、别名导出或构建期代码生成，需扩大检索维度确认后方可判断符号确实不存在。

### 2.3 工具执行生命周期与 Schema 校验

[![仓库原图：从工具名称查找到工具结果返回的生命周期](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo04-tool-lifecycle.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo04-tool-lifecycle.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/04-tool-system.md, 工具执行生命周期, MIT. 非官方历史实现分析图。*

如 [工具系统专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/04-tool-system.md) 所示，模型生成的工具调用需依次跨越名称查找、Schema 校验、权限与前置 Hook 审查、实际执行及结果格式化。注册表查找构成了第一道硬性防线：任何未注册的工具名称（且无合法别名映射）会被立即拦截并返回明确错误，直接阻断“虚构工具名”幻觉。随后，Zod Schema 校验负责核对字段存在性、数据类型与枚举范围（注：图中“Zod Schema 解析+强制”仅指显式声明了 coercion 或 transform 的字段转换，标准 `safeParse` 不会无条件将任意类型强制转换，详见 [Zod 官方 coercion 说明](https://zod.dev/api#coercion)）。

### 2.4 结构化错误恢复与 Tool Search 按需加载

当工具调用失败时，宿主返回的结构化错误信息直接决定了模型下一步是理性纠错还是盲目重试。系统需严格区分三类恢复路径：**修正调用参数**（应对字段缺失、路径拼写错误或锚点不唯一）、**刷新环境状态**（应对文件被外部修改或版本过期）以及**等待或升级请求授权**（应对网络限流、服务不可用或权限拒绝）。尤其需要防止工具将“查询执行失败”静默转换为“返回空数组”，以免模型将执行故障误判为“目标不存在”的负向事实证据。

当会话挂载海量 MCP 或外部工具时，全量工具定义会严重挤占上下文注意力。根据 [官方 Tool Search 文档](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) 与仓库修正记录，客户端仍在请求的工具数组中传入完整定义，但为延迟工具标记 `defer_loading`；初始模型上下文仅暴露搜索工具与核心常驻工具，待模型通过搜索命中后再按需展开目标工具的完整 Schema。这一机制降低了初始上下文噪声，但要求模型在遇到工具名称变更或调用报错时主动回到发现步骤刷新定义。

---

## 3. 代码编辑约束：Read-Before-Edit、锚点替换与状态新鲜度

### 3.1 Read-Before-Edit 与局部视图边界

[代码编辑策略专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/05-code-editing-strategy.md) 显示，历史实现的文件修改工具在编辑已有文件前会强制检查 `readFileState`：若当前会话尚未记录该文件的有效读取状态，工具直接拒绝写入并要求先执行读取。与单纯依赖系统提示词叮嘱“修改前请先阅读”相比，入口级前置检查将操作依赖编码进了执行器。我们可以将受控编辑工具的准入条件抽象为四元合取谓词（本文工程形式化）：

$$
\operatorname{AdmitEdit}(p)=
\operatorname{ReadableObservation}(p)
\land\operatorname{FreshEnough}(p)
\land\operatorname{ValidAnchor}(p)
\land\operatorname{Authorized}(p).
$$

此外，文件在系统中可能同时存在磁盘原始字节、Read 工具返回片段、自动注入提示词的裁剪文本以及压缩摘要四种不同表示。社区分析指出，内部状态通过 `isPartialView` 标记经过裁剪或脱敏的自动注入视图，防止模型将剥离了注释或元数据的不完整视图当作磁盘原文直接覆盖。

### 3.2 唯一锚点替换与定位校验

专用编辑工具采用“路径 + 旧文本锚点（`old_string`）+ 新文本（`new_string`）+ 是否全局替换（`replace_all`）”的契约形式（以下为接口结构示意）：

```json
{
  "file_path": "/workspace/src/date.ts",
  "old_string": "return format(value, defaultLocale);",
  "new_string": "return format(value, requestedLocale);",
  "replace_all": false
}
```

将 `old_string` 作为定位依据，本质上是要求模型提供一条可机械核对的当前文件观察证明。不同定位检查结果对应的系统行为与认知边界如下：

| 匹配场景 | 工具定位检查结果 | 合理的恢复动作 | 证据支持边界 |
| --- | --- | --- | --- |
| `old_string` 未命中 | 拒绝替换 | 重新读取目标区域，核对当前文件内容 | 仅说明锚点不匹配，不能直接推定文件不存在 |
| `old_string` 命中多次且 `replace_all=false` | 拒绝单次替换 | 扩大上下文行以构造唯一锚点，或确认批量意图 | 防止执行器在同名代码块中盲目猜测修改位置 |
| 唯一锚点命中 | 允许执行替换 | 检查生成的 diff、调用链兼容性与类型诊断 | 证明替换位置无歧义，但尚不证明新代码业务逻辑正确 |
| 显式指定 `replace_all=true` | 允许批量替换 | 逐一核查各处命中的语法作用域（如区分生产代码与注释/测试） | 文本字面相同不代表程序符号语义完全一致 |
| 新旧文本完全相同 | 拒绝无实质变更的请求 | 检查是否重复提交或遗漏实际修改内容 | 无差异不代表原缺陷已自动修复 |

### 3.3 状态新鲜度（Freshness）、mtime 与多写入入口对比

[![本文整理：读取、外部修改、拒绝过时编辑与重新验证的时序](/images/blog/claude-code-hallucination-reduction-design/claude-code-analysis02-read-edit-freshness.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-analysis02-read-edit-freshness.webp)

*来源：本文整理。内容指纹与验证合同用于说明更强的工程设计，不代表 Claude Code 全部采用此实现。*

在人机协作或多智能体并行工作区中，用户在 IDE 中的保存、代码格式化器触发或构建脚本生成，都可能在 Agent 读取文件之后改变磁盘内容。如上图所示，文件工具在写入前通过比对当前文件修改时间（`mtime`）与读取记录时间戳（并在特定平台回退核对完整内容）拦截过时写入，并在自身写入成功后同步刷新读取缓存。一旦检测到外部修改，Agent 必须重新读取最新内容并与用户改动共存合并，严禁通过全量覆盖强行抹除他人的未提交工作。

[![仓库原图：编辑、写入与 Notebook 工具的整合](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo05-edit-tool-integration.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo05-edit-tool-integration.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/05-code-editing-strategy.md, 与工具系统的整合, MIT. 非官方历史实现分析图。*

同时需要清醒认识不同写入入口的约束差异：`Edit`、`Write` 与 `NotebookEdit` 具备专门的状态检查与结构保持逻辑（例如新建不存在的文件无需前置读取，而覆盖已有文件或编辑 Notebook 单元格则受相应规则约束）；但通用 `Bash` 工具与外部 MCP 服务能够直接调用系统命令或远端接口修改状态，并不自动继承 `Edit` 工具内部的 `readFileState` 检查。此外，文件修改落盘后触发的 LSP 通知与单元测试，必须构成从“文本写入成功”到“静态类型合法”再到“目标运行行为成立”的递进证据链，严禁将退出码为 0 但实际发现 0 个测试用例的空跑结果当作有效验证。

---

## 4. 上下文工程、五级压缩流水线与证据恢复合同

### 4.1 系统提示词引导与上下文全景组装

[![仓库原图：系统提示词、用户上下文和工具相关内容的组装](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo02-context-construction.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo02-context-construction.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/03-context-engineering.md, 上下文构建全景, MIT. 非官方历史实现分析图。*

[系统提示词专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/14-system-prompt-design.md) 与 [上下文工程专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/03-context-engineering.md) 展示了模型输入是如何由系统行为准则、项目指令（`CLAUDE.md` / `AGENTS.md`）、动态环境事实（工作目录、平台、Git 状态）、工具定义与会话历史分层组装而成的。有效的提示词规则不会停留在抽象的“禁止幻觉”，而是转化为可检验的外部动作要求（如修改前定位源码、不确定时显式标注待验证边界、区分已观察事实与合理推断）。在故障排查中，必须严格区分**文件在磁盘可访问**、**历史轮次曾读取**与**当前上下文窗口仍可见**三种状态，并始终保留文本的来源身份，防止网页或日志中的外部数据伪装成高优先级系统指令。

### 4.2 五级压缩流水线与信息遗失风险

[![仓库原图：工具结果预算、历史裁剪、微压缩、上下文折叠与自动压缩](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo03-compaction-pipeline.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo03-compaction-pipeline.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/03-context-engineering.md, 五级压缩流水线, MIT. 非官方历史实现分析图。*

随着长会话不断累积工具输出，Claude Code 通过渐进式压缩流水线控制窗口体积（注：社区仓库 README 概述曾写作“四级”，而专题正文与架构图详细拆解为五级机制）：

1. **工具结果预算（Tool Result Budget）**：将超大工具输出持久化至磁盘，上下文中仅保留预览片段与文件路径引用，支持按需回查。
2. **历史裁剪（History Snip）**：移除早期冗余或可舍弃的历史交互片段。
3. **微压缩（Microcompact）**：对陈旧工具结果做轻量级清理或占位替换。
4. **上下文折叠（Context Collapse）**：在保留底层原始消息记录的前提下，以投影方式折叠当前模型可见视图。
5. **自动摘要压缩（Autocompact）**：当上下文逼近阈值时，调用摘要模型将长历史浓缩为结构化交接摘要，并重新挂载最近访问的核心文件、活跃计划与技能状态。

[![本文整理：失败证据在摘要中丢失，以及回到原始证据的恢复路径](/images/blog/claude-code-hallucination-reduction-design/claude-code-analysis03-evidence-loss-and-recovery.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-analysis03-evidence-loss-and-recovery.webp)

*来源：本文构造的失败模式与恢复路径，不是记录到的 Claude Code 事故。*

压缩在释放窗口容量的同时，极易引发**二次幻觉（Compaction-Induced Hallucination）**。如上图所示，自然语言摘要天生倾向于概括正面进展（如“已修改认证模块代码”），却容易遗漏具体的否定条件与未完成项（如“集成测试因缺少测试库凭据尚未重跑”）；经过多轮转述后，早期的“候选假设”会被逐步固化为“已确认事实”。

### 4.3 证据恢复与新鲜度合同

为防止压缩与交接洗白失败状态，自建或审计 Agent 系统时应为关键工程证据赋予明确的六元组身份（本文抽象模型）：

$$
e=(s,v,\ell,t,r,\Omega),
$$

其中 $s$ 表示证据来源（如测试运行器或源码文件），$v$ 表示内容指纹或版本标识，$\ell$ 表示具体定位路径，$t$ 表示观察时间戳，$r$ 表示显式结果状态，$\Omega$ 表示实际覆盖范围。一条抗压缩的验证记录示例如下（本文构造的 Schema 示意）：

```json
{
  "kind": "verification",
  "source": "project-test-runner",
  "run_id": "example-run-01",
  "subject": {
    "repository_revision": "COMMIT_PLACEHOLDER",
    "worktree_fingerprint": "CONTENT_FINGERPRINT_PLACEHOLDER",
    "dependency_fingerprint": "LOCKFILE_FINGERPRINT_PLACEHOLDER"
  },
  "command": "project-defined test command",
  "status": "failed",
  "coverage": ["date conversion boundary cases"],
  "artifact": "artifacts/example-run-01.log",
  "observed_at": "2026-10-27T11:00:00+08:00"
}
```

恢复上下文时，Agent 必须先通过指针 $\ell$ 核对原始结果 $r$，再校验当前工作树指纹是否仍与 $v$ 一致，从而阻断“用旧日志背书新代码”或“把已处理动作等同于已验证通过”的证据断裂。

---

## 5. 长期记忆、按需技能（Skills）与规划模式（Plan Mode）

### 5.1 项目指令与长期记忆的固化边界

[![仓库原图：哪些信息应保存，以及四种记忆类型的决策流程](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo08-memory-decision.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo08-memory-decision.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/08-memory-system.md, 记忆决策流程, MIT. 非官方历史设计分析。*

结合 [记忆系统专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/08-memory-system.md) 与 [官方 Memory 文档](https://code.claude.com/docs/en/memory)（含 `CLAUDE.md` 与 `AGENTS.md` 加载机制），长期记忆适合沉淀稳定的协作偏好、架构所有权边界与无法从代码中直接推导的业务背景，并应记录具体原因与绝对日期；而当前文件内容、分支合并状态与构建结果等易变事实，必须坚持从实时工作区重新读取。若将某次调试中未经证实的根因猜测写入长期记忆，后续会话在检索命中时会将其误作既定前提，导致错误假设跨会话自我强化。因此，记忆命中只能作为探索线索，不能替代实时取证。

### 5.2 Skills：程序性知识的按需注入

[![仓库原图：技能发现、执行模式选择与加载流程](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo09-skill-execution.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo09-skill-execution.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/09-skills-system.md, 执行流程概览, MIT. 非官方历史机制分析。*

如 [技能系统专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/09-skills-system.md) 所示，Skills 将特定领域的操作流程、脚本入口与验收标准封装为按需加载的模块化文档（常驻上下文仅保留简短描述与触发条件，匹配后再注入正文或派生独立执行上下文）。这减少了模型凭空猜测团队专属流程的概率，但在审计时仍需区分“技能被成功检索”、“技能正文被读取”与“技能要求的验证步骤被实际执行”三个不同阶段，同时防范过期技能文档与第三方不可信技能包带来的命令注入风险。

### 5.3 Plan Mode：在编码前暴露需求与架构假设

[![仓库原图：进入规划、探索、用户审批和返回执行的时序](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo10-plan-sequence.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo10-plan-sequence.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/10-plan-mode.md, 完整 Plan 模式流程, MIT. 非官方历史实现分析图。*

[Plan 模式专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/10-plan-mode.md) 展示了“只读探索 → 方案拟定与歧义提问 → 用户审批 → 退出规划进入执行”的完整时序。其核心价值在于将模型对多义需求的隐式猜测转化为用户可见的显式决策，并遵循两条工程纪律：凡是可通过仓库搜索确认的客观事实（如文件路径、测试框架、现有接口）均由 Agent 自行探索解决，仅将产品偏好、风险取舍与外部约束交由用户裁定；同时，用户批准计划仅代表认可当前实施方向，既不自动豁免后续高危操作（如强推远端或删除生产数据）的权限边界，也不等于提前证明了代码实现的正确性。

---

## 6. 多 Agent 上下文隔离、共享状态与任务编排

### 6.1 父子上下文隔离与相关错误偏差

[![仓库原图：父子 ToolUseContext 中克隆、替换与共享的状态](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo07-subagent-context.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo07-subagent-context.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/07-multi-agent.md, 上下文隔离深度解析, MIT. 非官方历史状态结构图。*

[多 Agent 专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/07-multi-agent.md) 与 [官方 Sub-agents 文档](https://code.claude.com/docs/en/sub-agents) 剖析了父子 Agent 之间的状态边界：在 `ToolUseContext` 中，`readFileState` 采用克隆（避免子任务污染父任务的读取记录），取消控制器（`AbortController`）采用包装或新建，而应用状态与任务追踪信息则存在受控共享通道。

在运用多 Agent 提升可靠性时，必须破除“多个 Agent 达成一致即代表统计独立验证”的迷思：

1. **同源模型偏差**：即便上下文窗口物理隔离，共享同一基座模型与相同错误摘要的多个子 Agent 极易在模糊接口上犯下完全相同的错误；真正有效的独立审查必须让验证者直接接触原始代码、运行异构行为测试并专门寻找反例。
2. **三层隔离边界**：上下文隔离（模型可见文本）、文件系统隔离（如 Git `worktree`）与外部运行资源隔离（本地端口、测试数据库、远端环境）分属不同维度。子任务在各自 `worktree` 中单独通过测试，合并回主分支后仍可能因接口假设冲突或共享端口/数据库污染而失效，主 Agent 必须承担集成后的最终验证责任。
3. **跨会话授权不传递**：子 Agent 回传的报告仅提供观察线索，不能自动升级为原始用户对高危操作的授权指令。

### 6.2 Task 状态管理、Verification Nudge 与动态工作流

[任务系统专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/15-task-system.md) 分析了任务拆解、依赖追踪与历史快照中的验证提醒（Verification Nudge）。需要明确：**将任务状态更新为 `completed` 仅代表工作流管理标记的写入，绝不等同于验证已经执行或通过**。在社区历史快照中，Verification Nudge 仅在主线程、全部任务完成、任务数 $\ge 3$ 且标题未匹配 `verif` 字符串、同时开启 `VERIFICATION_AGENT` 与 `tengu_hive_evidence` 两个实验开关时触发；这种基于标题字符串的启发式提醒虽能减少遗漏，但无法替代对测试产物与覆盖范围的实质核验。

对于后续版本观察到的 [自主目标循环（Goal Loop）](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/17-autonomy-goal-loop.md)、[动态工作流（Dynamic Workflows）](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/19-dynamic-workflows.md) 与 [Agent Teams](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/20-agent-teams.md)，将固定依赖调度交给确定性脚本有助于降低主模型的记忆负担，但脚本层面的结构化返回（如 `{"verified": true}`）仅解决了数据传递格式问题；若缺乏工作树指纹与测试报告定位，延长续跑轮数反而可能让模型沿着错误方向持续累积副作用。

---

## 7. Hooks 执行引擎、完成证据合同与权限安全边界

### 7.1 Hooks 生命周期与确定性门禁设计

[![仓库原图：Hook 执行引擎与输入输出路径](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo06-hook-engine.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-repo06-hook-engine.webp)

*Source: Windy3f3f3f3f/how-claude-code-works, commit f4d6505ed9162a0ee6be089190f74c419ecacb19, docs/06-hooks-extensibility.md, Hook 执行引擎, MIT. 图示为社区历史分析，不是当前完整事件规范。*

[Hooks 专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/06-hooks-extensibility.md) 与 [官方 Hooks 文档](https://code.claude.com/docs/en/hooks) 揭示了如何将检查逻辑从“指望模型自觉遵守的提示词”升级为“生命周期事件触发的程序执行点”。根据当前官方文档规范（要求 `v2.1.295+`），`command` 与 `http` 类型的 Hook 支持配置 `onFailure: "block"` 以在检查器异常时阻断动作，但该选项不适用于 `Stop`、`SubagentStop`、`TaskCompleted`、`TeammateIdle` 或后台异步命令 Hook；同时，`Stop` 事件不拦截用户主动中断，API 错误走无决策控制的 `StopFailure`，且连续续跑受 8 次上限约束（其间若发生工具调用则重置计数）。以下为前置拦截 Hook 的配置结构示意（本文构造示例）：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "node",
            "args": ["${CLAUDE_PROJECT_DIR}/.claude/hooks/check-command.js"],
            "timeout": 15,
            "onFailure": "block"
          }
        ]
      }
    ]
  }
}
```

在工程实践中，前置 Hook（`PreToolUse`）位于副作用发生前，适合执行确定性的安全与前提拦截；后置与结束 Hook（`PostToolUse` / `Stop`）则适合自动触发格式化、静态诊断或提醒补齐验证，但绝不能将客户端会话层的 `Stop` 检查当作生产环境的最终发布锁——真正的交付门禁必须部署在独立的 CI/CD 与发布系统中。

### 7.2 防止虚假完成的证据合同

[![本文整理：从工作树快照到最终完成声明的证据链](/images/blog/claude-code-hallucination-reduction-design/claude-code-analysis04-completion-evidence.webp)](/images/blog/claude-code-hallucination-reduction-design/claude-code-analysis04-completion-evidence.webp)

*本文整理，基于本报告对执行、验证与声明边界的分析。不是 Claude Code 官方架构，也不是已部署门槛。*

为根治“未运行测试、测试数为零或复用旧报告却宣告任务完成”的顽疾，验证检查器不能仅在日志字符串中正则匹配 `"passed"`，而应执行严格的快照对齐与用例计数核验（本文设计的验收逻辑伪代码）：

```text
verify_report(report, current_snapshot, required_checks):
    if report is missing or cannot be parsed:
        return ERROR("没有可读取的验证报告")
    if report.snapshot != current_snapshot:
        return ERROR("报告属于另一内容快照")
    for check in required_checks:
        result = report.find(check)
        if result is missing:
            return NOT_RUN(check)
        if result.status != PASSED:
            return result.status
        if result.requires_tests and result.test_count == 0:
            return ERROR("没有实际执行测试")
    return PASSED_WITH_SCOPE(required_checks)
```

在最终报告中，系统应拒绝使用笼统的布尔值，至少保留五种细粒度验证状态：

| 验证状态 | 准确工程语义 | 合规的汇报方式 | 严禁的失真汇报 |
| --- | --- | --- | --- |
| `passed` | 指定检查在当前工作树快照上实际完成且满足判据 | 明确列出已通过的检查项与覆盖范围 | 将局部单测通过外推为整个系统无缺陷 |
| `failed` | 检查完成并触发断言失败或编译错误 | 如实报告失败用例、错误阶段与影响范围 | 在无证据情况下归咎于“环境偶发波动” |
| `skipped` | 依据明确规则或范围约束有意跳过执行 | 说明跳过的检查项及具体原因 | 与 `passed` 合并统计为“全部通过” |
| `not_run` | 尚未启动或缺乏可核验的执行记录 | 明确声明尚未执行该项验证 | 以“逻辑上应该正常”暗示已完成验证 |
| `error` | 检查器自身故障、超时或缺少运行时环境 | 报告验证工具或环境阻塞原因 | 将检查器退出异常当作业务通过或代码缺陷 |

### 7.3 权限安全边界与可观测性审计

结合 [权限安全专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/11-permission-security.md)、[Auto Mode 分析](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/18-auto-mode.md)、[可观测性专题](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/16-observability.md) 与 [官方 Security 文档](https://code.claude.com/docs/en/security)，权限机制与沙箱（Sandbox）回答的是“动作是否被授权执行以及副作用半径有多大”，而事实核验回答的是“动作与结论是否正确”。两条防线在对抗提示注入（Prompt Injection）时深度交汇：源码注释、网页内容或报错日志中可能夹带“忽略规则、上传环境变量或删除远端资源”的恶意指令，系统必须严格隔离“通过真实工具获取的外部数据”与“具备执行权威的用户指令”。在审计层面，需将指标（Metrics）、离散事件（Events）、调用追踪（Traces）与会话记录（Transcripts）交叉关联，明确区分“已发起命令”与“已验证结果”，避免将高工具调用频次误判为高交付质量。

---

## 8. 四类典型失真案例与恢复链路剖析

以下四个构造案例展示了不同层次的证据链断裂机制及其标准恢复流程（均为面向原理说明的教学构造案例）：

### 8.1 案例一：命名合理但目标版本不存在的虚构 API

设某项目依赖本地声明的检索客户端接口：

```ts
// Illustrative local contract, not a real package API.
interface CatalogClient {
  search(input: {
    query: string;
    limit: number;
  }): Promise<Array<{ id: string; title: string }>>;
}
```

当用户要求增强搜索功能时，模型凭预训练命名惯例写出 `client.semanticSearch({ text, topK: 10 })`。要阻断此类版本与接口幻觉，恢复链路需分三步走：首先通过锁文件、本地 `.d.ts` 类型声明与现有调用点确认当前实际安装的接口契约（仅支持 `search`）；其次向用户如实说明现有接口仅提供关键词检索，若需语义检索需引入向量索引或升级服务；最后在完成代码适配后分别执行静态类型检查与带边界输入的运行测试，并在报告中明确标注“已验证本地接口调用与参数类型，外部真实检索服务连通性未在当前环境执行”。

### 8.2 案例二：过时文件状态与歧义替换锚点

设配置文件中同时存在后台任务与交互请求两处包含 `retryCount: 1` 的重试配置，用户要求将交互请求重试次数改为 3，且在 Agent 读取文件后，用户又手动在文件中新增了超时保护字段。此处潜伏着四重风险：未读取即盲目覆盖、仅用 `retryCount: 1` 作为锚点导致多匹配歧义、基于旧读取记录写入覆盖掉用户刚新增的超时字段，或者锚点唯一却误改了后台任务配置。标准恢复流程是：收到多匹配或 `mtime` 过期拒绝后，立即重新读取最新文件全文，扩充 `old_string` 上下文以精确锁定交互请求代码块，保留用户新增的超时保护配置，并在修改后结合超时预算验证尾延迟行为。

### 8.3 案例三：长会话压缩引发的失败条件遗忘

设集成测试因认证失败未能进入核心业务断言，模型修改认证配置后恰逢长会话触发自动压缩（Autocompact）。若生成的摘要仅记录“已修复认证配置，准备整理交付报告”，而遗漏了“集成测试尚未重跑验证”，恢复后的模型或下游子 Agent 极易将“修改动作已完成”当成“问题已验证解决”。阻断此类级联失真的关键在于：压缩摘要必须单列“未完成验证项”与原始失败日志指针；模型在压缩恢复后必须先核对验证状态表，发现缺少当前工作树快照下的成功测试记录时，自动将结论降级为 `not_run` 并触发重测，严禁把风格审查子 Agent 的通过结论包装成集成业务验收。

### 8.4 案例四：未运行或测错对象却宣布测试通过

在交付收尾阶段，常见的六种“假通过”形态及其合规处置方式如下表所示：

| 构造故障情形 | 实际可确认的客观事实 | 合规报告状态 | 标准恢复动作 |
| --- | --- | --- | --- |
| 仅在回复中写出测试命令代码块，未发起工具调用 | 仅生成了文本计划 | `not_run` | 在授权范围内实际调用命令工具执行 |
| 测试进程启动后观察窗口超时，后台仍在运行 | 执行尚未结束，无最终结论 | `error` / 待定 | 查询后台进程状态，严禁盲目并发重跑造成资源竞争 |
| 测试运行器退出码为 0，但收集到的测试数量为 0 | 筛选器或路径配置未命中任何用例 | 拒绝认定为通过 | 核对测试文件路径、匹配规则与运行器配置 |
| 测试报告生成时间早于最后一次代码或配置修改 | 历史快照曾通过验证 | 当前快照 `not_run` | 针对最新工作树重新执行相关检查 |
| 浏览器或测试脚本连入了另一目录启动的旧服务端口 | 旧服务进程响应正常 | 对象错配（无效） | 校验服务进程工作目录、版本标识与端口映射关系 |
| 为求绿色通过而删减失败断言或放宽测试容忍度 | 验证契约被人为削弱 | 需重点审计 | 同时审阅生产代码与测试代码的 diff，恢复有效断言 |

---

## 9. 可靠性评测协议、迁移优先级与工程采用清单

### 9.1 受控评测协议与核心指标定义

静态架构分析只能解释机制的工程合理性，要定量测定某项机制到底降低了多少幻觉，必须在固定仓库快照、相同模型版本与统一预算下开展控制变量消融实验。若记 Agent 输出中需要外部事实支撑的声明集合为 $C$，其中缺乏合格证据支持或与实际证据冲突的声明子集为 $U$，则无依据事实声明率定义为：

$$
R_{\mathrm{unsupported}} = \frac{|U|}{|C|}.
$$

为防止系统通过“拒绝回答一切问题”或“拒绝执行任何操作”刷低错误率，评测必须同步监控以下多维指标：

| 评估指标 | 严谨定义 | 必须同步记录的分母与边界条件 |
| --- | --- | --- |
| 无依据事实声明率（$R_{\mathrm{unsupported}}$） | 缺少合格工具证据支撑或与证据冲突的事实声明占比 | 明确事实声明提取规则，同步报告有效声明总覆盖量 $|C|$ |
| 虚假完成率 | 宣称已完成交付但未通过独立验收脚本的任务占比 | 区分以“全部任务”还是“发出完成声明的任务”为分母 |
| 真实任务成功率 | 最终工作树通过隐藏测试集与预设验收标准的任务占比 | 验收标准必须在实验前冻结，严禁事后按输出修改判据 |
| 误拦截率（False Block Rate） | 合规且安全的正常工具调用被前置检查错误拒绝的占比 | 衡量门禁过严对可用性的损耗 |
| 故障恢复率 | 注入指定错误（如过时文件、报错命令）后成功纠正的任务占比 | 记录恢复消耗的额外轮数与是否产生多余副作用 |
| 证据新鲜度违规率 | 引用过期工作树或错配环境的检查报告支撑当前结论的占比 | 明确内容指纹与环境依赖的哈希计算范围 |
| 端到端成本与延迟 | 单任务消耗的 Token 总量、工具调用次数与墙钟时间（Wall-clock time） | 拆解模型推理耗时、工具执行耗时与缓存命中率 |

### 9.2 自建 Agent 系统的四级迁移优先级

团队在将 Claude Code 的设计经验迁移至自建智能体平台时，无需照搬内部函数名或实验特性开关，而应按投入产出比遵循四个递进阶段：

1. **第一优先级：打通可定位的取证与结构化错误回传**。确保所有读写与检索工具返回清晰的路径、版本、截断标志与错误分类（区分未知工具、参数不合法、状态过期与权限拒绝），杜绝将异常吞没为空结果。
2. **第二优先级：在高频出错入口部署确定性前置检查**。为文件编辑工具落地 Read-Before-Edit、锚点唯一性校验与 `mtime`/内容指纹新鲜度比对；为测试报告验收增加“非零用例计数”与“晚于最后一次修改”的硬性校验。
3. **第三优先级：构建抗压缩的上下文状态与记忆治理**。在长会话摘要中强制单列“未验证项”与“失败日志文件指针”，将大体积工具输出转存为可回查的磁盘引用，并为长期记忆标注来源与适用前提。
4. **第四优先级：引入多 Agent 独立反例审查与自动化生命周期门禁**。在单 Agent 证据链扎实的基础上，利用 Hooks 在关键生命周期节点触发确定性检查，通过独立 `worktree` 与反例导向的验证子 Agent 开展交叉核验，并在最终发布入口由独立 CI 校验工作树快照与测试产物的绑定关系。

---

## 参考文献与开源许可说明

### 核心参考资料

- **独立架构分析基线**：Windy3f3f3f3f, [how-claude-code-works](https://github.com/Windy3f3f3f3f/how-claude-code-works/tree/f4d6505ed9162a0ee6be089190f74c419ecacb19)（固定提交 `f4d6505ed9162a0ee6be089190f74c419ecacb19`，重点涵盖 `02-agent-loop`、`03-context-engineering`、`04-tool-system`、`05-code-editing-strategy`、`06-hooks-extensibility`、`07-multi-agent`、`08-memory-system`、`09-skills-system`、`10-plan-mode`、`11-permission-security`、`14-system-prompt-design`、`15-task-system`、`16-observability`、`17-autonomy-goal-loop`、`18-auto-mode`、`19-dynamic-workflows` 与 `20-agent-teams`）。
- **Anthropic 官方文档与工程指南**（2026-10-27 查阅）：[How Claude Code Works](https://code.claude.com/docs/en/how-claude-code-works)、[Best Practices](https://code.claude.com/docs/en/best-practices)、[Memory](https://code.claude.com/docs/en/memory)、[Sub-agents](https://code.claude.com/docs/en/sub-agents)、[Hooks Reference](https://code.claude.com/docs/en/hooks)、[Security](https://code.claude.com/docs/en/security)、[Tool Search Tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)、[Reduce Hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)、[Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) 及 [Claude Code `v2.1.311` Release](https://github.com/anthropics/claude-code/releases/tag/v2.1.311)。
- **参数校验参考**：[Zod Official Documentation — Coercion](https://zod.dev/api#coercion)。

### 图像来源与 MIT 许可声明

本文引用的 10 张仓库架构图渲染自 `Windy3f3f3f3f/how-claude-code-works`（commit `f4d6505ed9162a0ee6be089190f74c419ecacb19`）中的 Mermaid 源码，另有 4 张分析图由本文整理绘制。目标分析仓库采用 [MIT License](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/LICENSE) 授权（`Copyright (c) 2025 Windy3f3f3f3f`），该授权仅覆盖仓库作者有权许可的分析文档与图示，不扩展至 Anthropic 官方产品源码、商标或内部资产。

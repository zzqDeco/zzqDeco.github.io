# Claude Code reliability reading: diagram sources

Accessed: 2026-10-27 (Asia/Shanghai).

Repository: https://github.com/Windy3f3f3f3f/how-claude-code-works

Pinned commit: f4d6505ed9162a0ee6be089190f74c419ecacb19.

The first ten diagrams are unchanged Mermaid blocks from the independent analysis repository, not official Anthropic diagrams. Rendering changes layout only. Historical implementation claims in the diagrams are qualified in the article. The last four diagrams are editorial synthesis, not repository originals. PNG masters are rendered on white with PingFang SC; the site pipeline generates responsive WebP. No contributor photographs or QR codes are reused.

## claude-code-repo01-agent-loop

Source: [02-agent-loop.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/02-agent-loop.md), 2.2 双层生成器架构. License: MIT.

```mermaid
graph TB
    subgraph QueryEngine ["QueryEngine (src/QueryEngine.ts)"]
        direction TB
        SM[submitMessage] --> PI[processUserInput]
        PI --> QC[query 调用]

        subgraph QueryFn ["query() (src/query.ts)"]
            direction TB
            Norm[消息规范化+压缩] --> APICall[API 流式调用]
            APICall --> ToolExec[工具执行]
            ToolExec --> Continue{继续/终止?}
            Continue -->|继续| Norm
        end

        QC --> QueryFn
    end
```

## claude-code-repo02-context-construction

Source: [03-context-engineering.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/03-context-engineering.md), 3.1 上下文构建全景. License: MIT.

```mermaid
graph TD
    subgraph 系统提示词组装
        A1[归属头 Attribution Header] --> SP[完整系统提示词]
        A2[CLI 系统提示词前缀] --> SP
        A3[工具描述与 prompt] --> SP
        A4[工具搜索指令] --> SP
        A5[顾问指令] --> SP
    end

    subgraph 系统上下文 ["系统上下文 (getSystemContext)"]
        B1[Git 状态<br/>分支/暂存/最近提交] --> SC[systemContext]
    end

    subgraph 用户上下文 ["用户上下文 (getUserContext)"]
        C1[CLAUDE.md 文件发现] --> UC[userContext]
        C2[当前日期 ISO格式] --> UC
    end

    SP --> Final[最终 API 请求]
    SC --> Final
    UC --> Final
    D[对话历史 messages] --> Final
```

## claude-code-repo03-compaction-pipeline

Source: [03-context-engineering.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/03-context-engineering.md), 3.4 五级压缩流水线. License: MIT.

```mermaid
flowchart TD
    Input[消息列表] --> L1[1. Tool Result 预算裁剪<br/>applyToolResultBudget<br/>大结果持久化到磁盘]
    L1 --> L2[2. History Snip 剪裁<br/>snipCompactIfNeeded<br/>Feature-gated 释放Token]
    L2 --> L3[3. Microcompact 微压缩<br/>两条路径：基于时间 / 缓存编辑<br/>清理旧工具结果]
    L3 --> L4[4. Context Collapse 上下文折叠<br/>投影式只读视图<br/>不修改原始消息]
    L4 --> L5[5. Autocompact 自动全量压缩<br/>fork子Agent生成摘要<br/>最后手段]

    style L1 fill:#e1f5fe
    style L2 fill:#e8f5e9
    style L3 fill:#fff3e0
    style L4 fill:#fce4ec
    style L5 fill:#f3e5f5
```

## claude-code-repo04-tool-lifecycle

Source: [04-tool-system.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/04-tool-system.md), 4.4 工具执行生命周期. License: MIT.

```mermaid
flowchart TD
    Input[模型输出 tool_use block] --> Find[1. 工具查找<br/>按name/alias查找<br/>检查废弃别名]
    Find --> Validate[2. 输入验证<br/>Zod Schema解析+强制<br/>validateInput检查]
    Validate --> Parallel[3. 并行启动]

    subgraph 并行
        Hook[Pre-Tool Hook<br/>hooks配置]
        Classifier[Bash分类器<br/>投机执行]
    end

    Parallel --> Hook
    Parallel --> Classifier
    Hook --> Perm[4. 权限检查<br/>规则匹配<br/>分类器自动审批<br/>Hook覆盖<br/>交互式确认]
    Classifier --> Perm

    Perm --> Exec[5. 工具执行<br/>tool.call<br/>流式进度事件<br/>超时/沙箱]
    Exec --> Result[6. 结果处理<br/>mapToolResult<br/>大结果持久化到磁盘]
    Result --> PostHook[7. Post-Tool Hook<br/>PostToolUse 成功<br/>PostToolUseFailure 失败]
    PostHook --> Emit[8. 消息发射<br/>tool_result block]
```

## claude-code-repo05-edit-tool-integration

Source: [05-code-editing-strategy.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/05-code-editing-strategy.md), 10.10 与工具系统的整合. License: MIT.

```mermaid
graph TB
    Model[模型决策] --> Choice{选择工具}
    Choice -->|修改已有文件| Edit[FileEditTool<br/>search-and-replace<br/>isReadOnly=false<br/>isDestructive=false]
    Choice -->|创建新文件| Write[FileWriteTool<br/>全文件写入<br/>isReadOnly=false<br/>isDestructive=false]
    Choice -->|Notebook| NB[NotebookEditTool<br/>cell级编辑]

    Edit --> Perm[权限检查]
    Write --> Perm
    NB --> Perm

    Perm -->|acceptEdits模式| Auto[自动批准]
    Perm -->|default模式| Ask[用户确认]
```

## claude-code-repo06-hook-engine

Source: [06-hooks-extensibility.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/06-hooks-extensibility.md), 7.4 Hook 执行引擎. License: MIT.

```mermaid
flowchart TD
    Trigger[Hook 事件触发] --> Fast{"快速存在性检查<br/>hasHookForEvent()"}
    Fast -->|"无配置"| Skip[直接返回]
    Fast -->|"有配置"| Trust["1. 信任检查<br/>shouldSkipHookDueToTrust()"]
    Trust --> Match["2. Matcher + if 匹配<br/>getMatchingHooks()"]
    Match --> Dedup["3. 去重<br/>hookDedupKey()"]
    Dedup --> Input["4. 输入构建 + 并行执行"]
    Input --> Parse["5. 输出解析 + 退出码语义"]
    Parse --> Aggregate["6. 结果聚合 + 事件发射"]
```

## claude-code-repo07-subagent-context

Source: [07-multi-agent.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/07-multi-agent.md), 上下文隔离深度解析. License: MIT.

```mermaid
flowchart TB
    subgraph Parent ["父级 ToolUseContext"]
        direction TB
        P_RFS["readFileState"]
        P_AC["abortController"]
        P_GAS["getAppState"]
        P_SAS["setAppState"]
        P_SAST["setAppStateForTasks"]
        P_QT["queryTracking<br/>{chainId: X, depth: N}"]
        P_CRS["contentReplacementState"]
    end

    subgraph Child ["子级 ToolUseContext"]
        direction TB
        C_RFS["readFileState<br/>(克隆副本)"]
        C_AC["abortController<br/>(新建子控制器)"]
        C_GAS["getAppState<br/>(包装: shouldAvoid<br/>PermissionPrompts=true)"]
        C_SAS["setAppState<br/>(no-op)"]
        C_SAST["setAppStateForTasks<br/>(共享!)"]
        C_QT["queryTracking<br/>{chainId: Y, depth: N+1}"]
        C_CRS["contentReplacementState<br/>(克隆副本)"]
    end

    P_RFS -->|"cloneFileStateCache()"| C_RFS
    P_AC -->|"createChildAbortController()"| C_AC
    P_GAS -->|"包装"| C_GAS
    P_SAS -->|"替换为 no-op"| C_SAS
    P_SAST -->|"直接共享"| C_SAST
    P_QT -->|"新 UUID + depth+1"| C_QT
    P_CRS -->|"cloneContentReplacementState()"| C_CRS
```

## claude-code-repo08-memory-decision

Source: [08-memory-system.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/08-memory-system.md), 记忆决策流程. License: MIT.

```mermaid
flowchart TD
    Input[获取到一条信息] --> Q1{能否从代码/Git/文档<br/>直接获取？}
    Q1 -->|能| Skip[不保存]
    Q1 -->|不能| Q2{已经在 CLAUDE.md 中？}
    Q2 -->|是| Skip
    Q2 -->|否| Q3{属于哪种类型？}
    Q3 -->|用户身份/偏好| User[保存为 user]
    Q3 -->|行为纠正/肯定| FB[保存为 feedback<br/>必须含 Why + How to apply]
    Q3 -->|项目动态/决策| Proj[保存为 project<br/>相对日期→绝对日期]
    Q3 -->|外部系统位置| Ref[保存为 reference]
    Q3 -->|都不是| Skip
```

## claude-code-repo09-skill-execution

Source: [09-skills-system.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/09-skills-system.md), 执行流程概览. License: MIT.

```mermaid
flowchart TD
    Input["接收技能调用<br/>skill name + args"] --> Lookup["查找命令<br/>findCommand()"]
    Lookup --> Found{找到？}
    Found -->|否| Error["返回错误"]
    Found -->|是| ContextCheck{"context == 'fork'?"}
    ContextCheck -->|fork| Fork["executeForkedSkill()<br/>创建隔离子 Agent"]
    ContextCheck -->|inline| Process["processPromptSlashCommand()<br/>加载并替换提示词"]
    Process --> Inject["注入技能内容到对话<br/>+ contextModifier"]
    Fork --> Return["返回结果文本"]
    Inject --> Return2["继续主对话"]
```

## claude-code-repo10-plan-sequence

Source: [10-plan-mode.md](https://github.com/Windy3f3f3f3f/how-claude-code-works/blob/f4d6505ed9162a0ee6be089190f74c419ecacb19/docs/10-plan-mode.md), 9.2 全景：一次完整的 Plan 模式流程. License: MIT.

```mermaid
sequenceDiagram
    participant U as 用户
    participant M as 模型
    participant S as 状态机
    participant F as 计划文件

    U->>M: "重构认证系统"
    M->>S: 调用 EnterPlanMode
    S->>S: handlePlanModeTransition(default → plan)
    S->>S: prepareContextForPlanMode()
    S-->>M: 确认：已进入 plan 模式

    Note over M: 系统注入 plan_mode 附件<br/>（5 阶段工作流指令）

    M->>M: Phase 1: 启动 Explore Agent 探索代码
    M->>M: Phase 2: 启动 Plan Agent 设计方案
    M->>M: Phase 3: 审阅并向用户提问
    M->>F: Phase 4: 写入计划到 ~/.claude/plans/bold-eagle.md
    M->>S: Phase 5: 调用 ExitPlanMode

    S->>U: 弹出审批对话框（展示计划内容）
    U->>S: 批准（可编辑计划后批准）

    S->>S: 恢复 prePlanMode（如 default）
    S->>S: setHasExitedPlanMode(true)
    S-->>M: 返回审批后的完整计划文本

    Note over M: 系统注入 plan_mode_exit 附件<br/>"你已退出 plan 模式，可以开始实施"

    M->>M: 按计划开始编辑文件
```

## claude-code-analysis01-reliability-layers

Source: original editorial synthesis for this report, 2026-10-27. This is not an Anthropic implementation diagram.

```mermaid
flowchart TB
    U["用户目标与验收条件"] --> E["证据获取：文件、文档、工具结果"]
    E --> C["上下文维护：来源、版本、压缩与重读"]
    C --> M["模型提出解释或操作"]
    M --> P["操作前置检查：接口、状态、定位"]
    P --> X["执行与结果记录"]
    X --> V["验证：测试、实际场景、人工验收"]
    V --> R["报告：已证实、失败、未验证"]
    V -->|"证据不足，继续取证"| E
    A["授权与隔离边界\n限制可执行行为，不证明结论正确"] -.-> P
    A -.-> X
```

## claude-code-analysis02-read-edit-freshness

Source: original editorial synthesis. The content-fingerprint contract illustrates a stronger engineering design; it is not a claim about Claude Code's exact filesystem implementation.

```mermaid
sequenceDiagram
    participant A as Agent
    participant F as 文件系统
    participant B as 其他写入者
    participant V as 验证器
    A->>F: 读取文件，记录观察状态
    F-->>A: 内容与读取证据
    B->>F: 修改同一文件
    A->>F: 请求替换旧锚点
    F-->>A: 新鲜度或锚点校验失败
    A->>F: 重新读取并重新决定修改
    F-->>A: 当前内容
    A->>F: 应用针对当前内容的修改
    A->>V: 验证实际修改后的工作树
    V-->>A: 状态、覆盖范围、内容指纹
    Note over A,V: 定位有效不等于逻辑正确；进程内顺序不等于跨进程事务
```

## claude-code-analysis03-evidence-loss-and-recovery

Source: original editorial synthesis. The examples illustrate failure modes, not recorded product incidents.

```mermaid
flowchart TB
    R["原始工具结果：测试失败，日志完整"] --> S["压缩摘要：保留进展，可能漏掉否定条件"]
    S --> H["错误推断：已修改，所以应该通过"]
    R --> P["持久化证据：路径、内容指纹、状态"]
    P --> Q["发现摘要证据不足"]
    Q --> D["重读原始日志与当前文件"]
    D --> T{"证据仍对应当前工作树？"}
    T -->|"否"| N["重新运行相关检查"]
    T -->|"是"| B["保留原状态与适用范围"]
    N --> B
    B --> O["报告已验证与未验证项"]
    H -.->|"需要外部证据纠正"| Q
```

## claude-code-analysis04-completion-evidence

Source: original editorial synthesis of a completion contract, not a built-in Claude Code state machine.

```mermaid
flowchart TB
    C["准备声明完成"] --> D["确定交付对象与验收范围"]
    D --> F["绑定当前工作树、依赖与配置指纹"]
    F --> T["运行与目标对应的检查"]
    T --> S{"验证状态"}
    S -->|"passed"| P["核对覆盖范围和结果新鲜度"]
    S -->|"failed"| R["报告失败并修复"]
    S -->|"skipped / not_run / error"| U["报告未验证，不升级为成功"]
    P --> G{"还有必须验证的交付边界？"}
    G -->|"有"| T
    G -->|"无"| O["有范围、有证据的完成说明"]
    R --> F
    U --> O
    Note["提交、推送、CI、部署、运行与业务验收分别举证"] -.-> D
```

## Repository license

The MIT notice below applies to the cited repository material. It does not relicense Anthropic software, proprietary prompt text, trademarks, or official documentation. The report paraphrases implementation descriptions rather than redistributing full internal prompt collections.

```text
MIT License

Copyright (c) 2025 Windy3f3f3f3f

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

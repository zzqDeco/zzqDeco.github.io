---
title: "Git 分支工作流：从 Git Flow 到 Trunk-Based"
description: "横向比较 GitHub Flow、GitLab Flow、OneFlow、Trunk-Based Development 与 Git Flow 的分支模型、发布方式、成本和适用场景。"
pubDate: "2026-05-16T14:27:17+08:00"
updatedDate: "2026-05-16T14:27:17+08:00"
tags:
  - "Git"
  - "Engineering"
  - "Workflow"
draft: false
---

分支工作流看起来是在讨论 Git，其实更多是在讨论团队如何处理四个东西：集成频率、发布控制、回滚成本、协作成本。

如果一个团队每天都能把小变更合入主干，并且 CI/CD、灰度、feature flag 都很成熟，那么长期分支会更像负担；如果一个团队要维护多个版本线、多个客户交付包、严格的测试窗口，那么“所有东西都尽快进 main”又可能过于理想化。

所以没有一个 flow 是永远正确的。真正要问的是：这个 flow 把复杂度放在了哪里，团队有没有能力在那里承受它。

![Git branching workflow flowchart](/images/blog/git-branching-workflows/git-workflows-flowchart.webp)

图里的颜色保持一致：蓝色表示 `main`，绿色表示 `prod` 或 Git Flow 中的 `develop`，紫色表示 feature/hotfix 类临时分支，橙色表示 release 类发布分支。

## 先给结论

大多数现代 Web 服务、SaaS、内部平台，默认应该从 **Trunk-Based Development** 或 **GitHub Flow** 开始。它们把长期分支压到最少，强迫团队投资自动化测试、持续集成、小步提交和快速回滚。

如果发布不是一次性从 main 到 production，而是要经过 test、staging、pre-prod、prod 这些环境，**GitLab Flow** 会更贴近真实部署链路。它不是单纯多几个分支，而是把“代码已经合并”和“代码已经发布到某个环境”拆开表达。

如果你喜欢 Git Flow 的 release/hotfix 语义，但不想长期维护 `main` 和 `develop` 两条永久主线，**OneFlow** 是更轻的替代。

只有当团队确实需要多个稳定版本并行维护、固定版本列车、复杂 hotfix 策略时，才应该认真考虑 **Git Flow**。它不是错，只是成本很高，而且这个成本会在 CI、合并冲突、回归测试和版本管理里不断显现。

## 横向比较

| 工作流 | 长期分支 | 发布来源 | CI/CD 依赖 | 适合团队 | 主要问题 |
| --- | --- | --- | --- | --- | --- |
| GitHub Flow | `main` | 通常从 `main` 或 PR 预览环境发布 | 高 | Web 服务、持续部署、小团队到中型团队 | 对多版本维护和复杂发布窗口支持弱 |
| GitLab Flow | `main` + 环境分支或 release 分支 | 从环境分支、release 分支或 tag 发布 | 中到高 | 有 staging/pre-prod/prod 推进链路的团队 | 环境分支容易漂移，需要严格合并方向 |
| OneFlow | `main` | 从 `main` tag 或临时 release 分支发布 | 中 | 想保留 release/hotfix 但降低 Git Flow 复杂度的团队 | 知名度和工具支持弱于 Git Flow |
| Trunk-Based | `main`/`trunk` | 从 trunk 或 release cut 发布 | 很高 | CI 强、测试强、发布频繁的团队 | 对自动化、拆分能力、feature flag 要求高 |
| Git Flow | `main` + `develop` | 从 release 分支稳定后合入 `main` 并打 tag | 中 | 版本型产品、客户端、SDK、多版本维护 | 长期分支多，集成延迟，合并成本高 |

## GitHub Flow

GitHub Flow 的核心非常简单：`main` 永远是可部署状态；任何工作都从 `main` 拉出短生命周期分支；通过 Pull Request 做讨论、review、CI；确认无问题后合入 `main`，再部署或已经通过自动化部署。

它的分支结构通常是这样：

![GitHub Flow branch structure](/images/blog/git-branching-workflows/github-flow.webp)

这个模型最大的优点是认知负担低。开发者只需要理解一个长期分支和短期 feature branch。团队的协作焦点不在“我应该合到 develop 还是 release”，而在“这个 PR 是否足够小、是否被测试覆盖、是否可以安全进入 main”。

但 GitHub Flow 的简单不是免费的。它隐含了几个前提：

1. `main` 不能长期坏掉。
2. PR 必须小，否则 review 和回滚都会变慢。
3. CI 要足够可信，否则合入 main 只是把风险推后。
4. 未完成能力要能隐藏，比如通过 feature flag、配置开关、兼容性代码或后端双写。

GitHub Flow 很适合持续部署的 Web 服务，因为 Web 服务通常只有一个线上版本，回滚也可以通过重新部署旧版本或关闭开关完成。它不太适合同时维护多个客户版本的产品。如果一个 bug fix 要同时进 `v1.9`、`v2.0`、`v2.1`，单一 `main` 的模型就需要额外的 release branch 或 backport 策略补上。

一句话：GitHub Flow 把流程复杂度降到最低，但把工程纪律要求抬高了。

## GitLab Flow

GitLab Flow 试图解决一个很现实的问题：代码合并和代码发布经常不是同一件事。

很多公司不是 merge 到 `main` 就直接上线，而是会经过 `test`、`staging`、`pre-production`、`production`。GitLab Flow 会把这些环境或者版本线显式体现在分支策略中。常见做法有两类：

| 类型 | 分支形态 | 典型用途 |
| --- | --- | --- |
| 环境分支 | `main` -> `pre-production` -> `production` | 代码逐级推进到不同部署环境 |
| 版本分支 | `main` -> `release/1.2` | 对外发布固定版本，需要 bug fix 和补丁 |

环境分支或者 release 分支的典型关系可以抽象成这样：

![GitLab Flow branch structure](/images/blog/git-branching-workflows/gitlab-flow.webp)

它的价值在于可解释性很强。一个提交在 `main`，代表它已经完成开发集成；一个提交在 `pre-production`，代表它已经进入预生产验证；一个提交在 `production`，代表它已经上线。分支不只是代码容器，也变成了发布状态的表达。

风险也在这里。环境分支一旦被当成“不同团队各玩各的长期分支”，就会快速漂移。最糟糕的情况是：`production` 有线上 hotfix，`staging` 有测试中的 feature，`main` 又有新的开发提交，三条线互相 cherry-pick，最后没有人能直观看出哪个提交应该去哪里。

所以 GitLab Flow 要落地，需要明确规则：

1. 合并方向要固定，通常从上游开发分支逐级推进到下游环境分支。
2. 线上 hotfix 必须及时回灌到 `main`，不能只活在 `production`。
3. 环境分支应该表达部署状态，而不是成为长期开发分支。
4. 每个环境分支都要有自动化部署、回滚和审计记录，否则分支只是名字好看。

GitLab Flow 适合“发布需要排队、验证、审批”的团队。它比 GitHub Flow 更现实，也比 Git Flow 少一些永久分支负担。

## OneFlow

OneFlow 可以理解成对 Git Flow 的一次瘦身：保留 release branch、hotfix、tag 这些版本语义，但去掉永久存在的 `develop` 分支。所有已经完成的工作最终都回到唯一长期主线 `main`。

它的基本思想是：长期主线只留一条，其他分支都是临时的。

![OneFlow branch structure](/images/blog/git-branching-workflows/oneflow.webp)

OneFlow 解决的是 Git Flow 的一个典型痛点：`main` 和 `develop` 长期分离后，团队要不断回答“这次修复到底该先进哪里”。在 OneFlow 里，`main` 是事实主线，release branch 是为了稳定某个即将发布的版本而临时切出来的隔离区。

它适合这样的团队：

1. 产品仍然有明确版本号和发布窗口。
2. 发布前需要稳定分支做最后 bug fix。
3. 不想维护 `develop` 这条永久集成分支。
4. 接受用 tag 和临时 release branch 表达版本历史。

OneFlow 的缺点是生态共识没有 Git Flow 和 GitHub Flow 强。新人可能没听过，工具也不会天然假设你的流程叫 OneFlow。它更像是一套原则：减少永久分支，只在真正需要发布隔离时创建分支。

如果团队已经被 Git Flow 的分支数量拖慢，但又不能一步切到 Trunk-Based，OneFlow 是一个合理的中间态。

## Trunk-Based Development

Trunk-Based Development 的立场最激进也最现代：开发者应该频繁集成到单一主干，长期分支应该被避免。这里的 trunk 通常就是 `main`。

它不是“不建分支”。很多团队仍然会使用短生命周期 feature branch，但分支生命周期要足够短，通常以小时或一两天计算，而不是以周计算。更极端的团队会直接向 trunk 提交，但这要求测试和 review 机制非常成熟。

![Trunk-Based Development branch structure](/images/blog/git-branching-workflows/trunk-based.webp)

Trunk-Based 的优势来自高频集成。冲突会更早暴露，集成测试会更早运行，功能拆分会被迫变小。它的目标不是让 Git 历史更漂亮，而是让软件一直处在接近可发布的状态。

但这套模型对工程能力要求很硬：

1. CI 必须快，不然高频合入会被队列拖死。
2. 测试必须可信，不然 trunk 只是风险集中地。
3. 代码要能做增量交付，不能每个功能都憋成大爆炸。
4. 未完成能力要通过 feature flag、branch by abstraction、兼容数据结构等方式隐藏。
5. 回滚策略要清楚，因为发布频率高意味着线上变化也更频繁。

Trunk-Based 不适合用来掩盖工程能力不足。它会把问题暴露得更快：测试慢、模块边界差、PR 太大、review 队列拥堵、发布不可回滚，都会在 trunk 上变成日常事故。

但如果团队愿意补这些能力，它也是长期收益最高的模型之一。很多所谓“敏捷发布”的关键并不是分支策略名字，而是 trunk 能不能一直保持健康。

## Git Flow

Git Flow 是最经典、也最容易被滥用的模型。它定义了多种分支角色：

| 分支 | 角色 |
| --- | --- |
| `main` | 只保存正式发布历史，通常每次发布打 tag |
| `develop` | 日常集成分支，下一版功能先合到这里 |
| `feature/*` | 功能开发分支，从 `develop` 切出并合回 `develop` |
| `release/*` | 发布候选分支，用于稳定、修 bug、准备版本 |
| `hotfix/*` | 线上紧急修复，从 `main` 切出，修完合回 `main` 和 `develop` |

它的模型大概是这样：

![Git Flow branch structure](/images/blog/git-branching-workflows/git-flow.webp)

Git Flow 的优势是语义非常明确。发布前有 release 分支稳定版本，线上紧急问题有 hotfix 分支，正式发布历史在 `main` 上可追踪。对于传统客户端、嵌入式、SDK、on-premise 产品、多版本维护产品，这些语义仍然有价值。

问题是，它把集成延迟制度化了。feature 先合到 `develop`，release 从 `develop` 切出，修复再回灌到 `develop` 和 `main`。分支越多，团队越容易产生这些成本：

1. 合并路径变长，冲突发现更晚。
2. `main` 和 `develop` 的差异需要持续维护。
3. hotfix 要双向合并，漏一次就会出现线上修了、下一版又回归。
4. release 分支存在时间越长，回灌成本越高。
5. CI 需要覆盖更多分支组合，反馈速度变慢。

Git Flow 不是过时到不能用，而是不应该作为默认选项。它适合版本管理复杂的产品，不适合一个每天都能自动部署的 Web 服务。

## 选择建议

如果你不知道选什么，先问三个问题。

第一个问题：你们是不是可以从 `main` 自动发布，并且可以快速回滚？

如果答案是 yes，优先选择 GitHub Flow 或 Trunk-Based。两者的区别在于合入频率和工程成熟度。PR 仍然是主要协作单元时，GitHub Flow 更自然；团队已经能做到小步高频集成、强 CI、feature flag 时，Trunk-Based 更彻底。

第二个问题：发布是否必须经过多个环境、审批或人工验证？

如果答案是 yes，考虑 GitLab Flow。它能把“合入主线”和“推进到生产”拆开表达。但要注意，环境分支只能表达部署状态，不能变成另一个 develop。

第三个问题：你们是否要维护多个已经发布的版本？

如果答案是 yes，考虑 OneFlow 或 Git Flow。版本维护越复杂，Git Flow 的分支语义越有价值；如果只是需要 release branch 做短期稳定，OneFlow 通常更轻。

可以把选择压缩成一句话：

> 分支越多，发布控制越显式，但集成成本越高；分支越少，协作路径越短，但自动化和工程纪律必须更强。

## 我的偏好

我会把 Trunk-Based 当成长期目标，把 GitHub Flow 当成大多数团队的起点。

原因不是它们看起来更“先进”，而是它们逼迫团队面对真正的问题：测试是否可信、发布是否可回滚、功能是否能拆小、review 是否及时、线上是否能观测。如果这些问题没有解决，换成 Git Flow 只是把风险藏进更多分支里。

但对于发布链路复杂的团队，我不会强推 Trunk-Based。工程实践不是口号，发布约束是真实存在的。需要环境推进就用 GitLab Flow，需要多版本维护就用 OneFlow 或 Git Flow。关键是承认每个分支都不是免费的：它要被测试、被合并、被回灌、被解释，也要在事故发生时被人理解。

## References

- [GitHub Docs: GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)
- [GitLab Docs: GitLab flow](https://docs.gitlab.com/topics/version_control/gitlab_flow/)
- [End of Line Blog: OneFlow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow)
- [Trunk-Based Development](https://trunkbaseddevelopment.com/)
- [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
- [Atlassian Gitflow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

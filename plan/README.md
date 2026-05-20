# Blog Plan Index

`plan/` 保存活跃实施计划和近期已交付变更记录。当前系统行为应沉淀到 [doc/](../doc/README.md)，计划文档只保留足够的背景、决策和验证结果。

## Status Definitions

| Status | Meaning |
| --- | --- |
| Active | 正在设计或实现，仍作为决策来源 |
| Delivered | 已实现并验证，短期内仍有回溯价值 |
| Absorbed | 稳定行为已经沉淀到 `doc/`，后续可以清理 |

## Plans

| Document | Purpose | Status |
| --- | --- | --- |
| [blog-editor-image-rendering.plan.md](blog-editor-image-rendering.plan.md) | 修复本地 Blog Editor 对 `/images/*` 的渲染、插入和样式支持 | Delivered |
| [project-documentation-structure.plan.md](project-documentation-structure.plan.md) | 补齐 README、`doc/` 和 `plan/` 目录约定 | Delivered |

## Maintenance Rules

- 新增功能、修复或结构性调整时，优先补一份窄范围计划。
- 计划完成后把状态改为 `Delivered`，并把稳定行为沉淀到 `doc/`。
- 新增、归档或删除计划时同步更新本索引。
- 不在计划里重复长篇当前状态说明，链接到 `doc/`。

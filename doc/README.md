# Blog Documentation

`doc/` 保存项目当前状态文档。它描述系统现在如何工作，不承担历史计划归档职责。

## Start Here

| Document | Purpose |
| --- | --- |
| [Architecture](architecture.md) | Astro 站点、内容集合、静态资源、部署链路和本地工具边界 |
| [Blog Editor](blog-editor.md) | 本地编辑器的服务范围、图片渲染规则、上传行为和验证方法 |

## Documentation Rules

- 当前行为和工程约定放在 `doc/`。
- 活跃计划、已交付计划和变更记录放在 `plan/`。
- 新增或移动文档时同步更新本索引。
- 避免在多个文档重复大段说明，优先链接到更窄的文档。
- 文档只改格式或说明时至少运行 `git diff --check`；涉及站点或编辑器行为时运行 `npm run build`。

## Related Files

- [README.md](../README.md): 项目入口、命令和写作流程。
- [tools/blog-editor/README.md](../tools/blog-editor/README.md): 本地编辑器完整使用说明。
- [plan/README.md](../plan/README.md): 实施计划索引。

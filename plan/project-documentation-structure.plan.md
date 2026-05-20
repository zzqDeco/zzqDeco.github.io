# Project Documentation Structure

Status: Delivered

## Goal

补齐当前博客仓库的项目说明和工程记录结构，使它和同级项目保持一致：根目录提供快速入口，`doc/` 描述当前系统，`plan/` 保存实施计划和近期已交付记录。

## Changes

- 新增 `README.md`，记录项目用途、目录结构、常用命令、写作流程和部署方式。
- 新增 `doc/README.md`，作为当前状态文档索引。
- 新增 `doc/architecture.md`，说明 Astro 内容模型、页面结构、静态资源和部署链路。
- 新增 `doc/blog-editor.md`，说明本地编辑器的图片渲染、上传和验证规则。
- 新增 `plan/README.md`，定义计划状态和索引维护规则。
- 新增本计划文档，记录这次文档结构补齐。

## Verification

- `git diff --check`
- `npm run build`

## Maintenance

后续变更遵循同级项目习惯：

- 稳定事实写入 `doc/`。
- 实施过程、权衡和验证记录写入 `plan/`。
- 每次新增或移动文档时更新对应索引。

# Blog Architecture

## Overview

这个仓库是一个 Astro 静态博客。线上站点只由 Astro 构建产物提供服务，本地 Blog Editor 是仓库内的写作工具，不部署到 GitHub Pages。

## Content Model

博客文章位于 `src/content/blog/*.md`，由 `src/content.config.ts` 定义 collection schema：

| Field | Type | Notes |
| --- | --- | --- |
| `title` | string | 页面标题和列表标题 |
| `description` | string | SEO 描述和文章摘要 |
| `pubDate` | date | 列表排序字段 |
| `updatedDate` | optional date | 可选更新时间 |
| `tags` | string array | 文章标签 |
| `draft` | boolean | 生产构建中隐藏 `true` |

生产环境的首页和文章归档会过滤 `draft: true`；本地开发环境会展示草稿，方便预览。

## Runtime Pages

| Path | Role |
| --- | --- |
| `src/pages/index.astro` | 首页和 recent posts |
| `src/pages/blog/index.astro` | 文章归档第一页 |
| `src/pages/blog/page/[page].astro` | 文章归档后续分页 |
| `src/pages/blog/tag/[tag]/[...page].astro` | tag 筛选归档和分页 |
| `src/pages/blog/[...slug].astro` | 单篇文章页面 |
| `src/pages/rss.xml.js` | RSS 输出 |

共享布局和全局样式分别位于 `src/layouts/BaseLayout.astro` 和 `src/styles/global.css`。

文章归档默认每页展示 10 篇。`/blog/` 是第一页，后续页使用 `/blog/page/2/`、`/blog/page/3/` 形式。tag 页面使用静态 URL，例如 `/blog/tag/paper-reading/` 和 `/blog/tag/paper-reading/2/`。

tag slug 由 `src/lib/blog.ts` 统一生成：trim、lowercase、NFKC normalization，非 Unicode letter/number 字符折叠为 `-`。`page` 和 `tag` 是 `/blog/` 下的保留路径段，不应用作文章 id 的首段。

## Static Assets

Astro 按原样发布 `public/`。文章图片应放在：

```text
public/images/blog/<slug>/
```

Markdown 中使用线上和本地 Astro 都可识别的 root-relative 路径：

```md
![alt](/images/blog/<slug>/<file>.png)
```

不要在文章中引用本机绝对路径，也不要引用本地编辑器的临时目录。

## Local Editor Boundary

`tools/blog-editor/` 是本地辅助工具。它负责：

- 读取、创建、保存和软删除 `src/content/blog/*.md`。
- 上传图片到 `public/images/blog/<slug>/`。
- 在编辑器服务内暴露 `/images/*`，使文章里的图片路径能直接渲染。

它不负责：

- 线上认证。
- 数据库或 CMS。
- 改变 Astro content schema。
- 参与 GitHub Pages 构建。

## Deployment

`.github/workflows/deploy.yml` 在 `main` 分支 push 后运行 Astro 构建并部署到 GitHub Pages。部署前本地至少运行：

```bash
npm run build
```

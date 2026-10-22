# zzqDeco Blog

个人技术博客，基于 Astro 生成静态站点，部署到 GitHub Pages。Markdown 是文章的唯一来源，本地 Blog Editor 只作为写作辅助工具，不参与线上构建。

## Project Layout

| Path | Purpose |
| --- | --- |
| `src/content/blog/` | 博客 Markdown 源文件 |
| `src/pages/` | Astro 页面路由 |
| `src/components/` | 页面组件 |
| `src/layouts/` | 站点布局 |
| `src/styles/` | 全局样式 |
| `public/images/blog/` | 文章图片资源 |
| `tools/blog-editor/` | 本地 Markdown 编辑器 |
| `doc/` | 当前项目文档 |
| `plan/` | 实施计划和已交付变更记录 |

## Commands

```bash
npm install
npm run dev
npm run build
npm run preview
npm --prefix tools/blog-editor install  # 首次使用编辑器前
npm run editor
```

默认 Node.js 版本要求见 `package.json` 和 `.nvmrc`。

## Writing

文章放在 `src/content/blog/*.md`，frontmatter 遵循 `src/content.config.ts`：

```yaml
title: "Post title"
description: "Short summary"
pubDate: 2026-05-16
updatedDate: 2026-05-16
tags:
  - Git
draft: false
```

原图母版存放在 `assets/img-src/blog/<slug>/`，经 `tools/image-pipeline/convert_images.py` 生成多档 WebP 到 `public/images/blog/<slug>/` 并更新 `src/lib/image-manifest.json`。文章中使用 root-relative 路径：

```md
![alt](/images/blog/<slug>/<file>.webp)
```

本地写作可以运行 `npm run editor`，打开 `http://127.0.0.1:4322`。编辑器会读取和保存 `src/content/blog/`，上传光栅图片时会自动保存原图至 `assets/img-src/blog/<slug>/` 并调用图片管线生成多档 WebP 与更新 manifest。

## Deployment

推送到 `main` 后，`.github/workflows/deploy.yml` 会构建 Astro 站点并部署到 GitHub Pages。

## Documentation

- [Project documentation](doc/README.md)
- [Implementation plan index](plan/README.md)

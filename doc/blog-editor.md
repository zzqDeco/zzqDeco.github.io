# Blog Editor

## Purpose

本地 Blog Editor 是一个面向 `src/content/blog/*.md` 的 Markdown 写作工具。它让 frontmatter、正文编辑和文章图片上传集中在同一个本地页面中，但不会改变线上站点的构建方式。

启动命令（首次使用前需先安装编辑器自己的依赖，见下）：

```bash
npm --prefix tools/blog-editor install  # 首次使用
npm run editor
```

编辑器是 `tools/blog-editor/` 下的独立 npm package（独立 lockfile），根目录安装不会带入它的依赖。服务器基于自身文件位置推导仓库根，从任意 cwd 启动均可。

默认地址：

```text
http://127.0.0.1:4322
```

## Static Image Rendering

文章中的图片使用 Astro 站点路径：

```md
![alt](/images/blog/<slug>/<file>.png)
```

编辑器服务需要和 Astro 一样能读取这些路径。`tools/blog-editor/server.mjs` 对 `/images/*` 做了受限静态文件服务：

- 只映射到仓库的 `public/images/*`。
- 不开放整个仓库根目录。
- 保留 API、vendor、编辑器自身 `app.js` 和 `style.css` 的路由优先级。
- 对图片响应使用 `Cache-Control: no-store`，上传或替换后可以立即刷新。
- 对 path traversal 请求返回 `403` 或 `404`。

## Image Insert And Upload

`Image URL` 支持三类路径：

- `https://...`
- `http://...`
- `/images/...`

上传图片会写入：

```text
public/images/blog/<slug>/
```

插入正文时仍保存为标准 Markdown：

```md
![alt](/images/blog/<slug>/<filename>)
```

## Editor Styling Contract

编辑器的图片样式覆盖 Milkdown 和 ProseMirror 常见 wrapper。目标是让大图在编辑栏内稳定缩放：

- 宽图不超过编辑器宽度。
- 高图受 `max-height` 约束，并用 `object-fit: contain` 保持比例。
- 图片块有统一背景、边框、圆角和选中态。

这些样式只影响本地编辑器，不影响 Astro 站点的文章样式。

## Verification

请求级检查：

```bash
curl -I http://127.0.0.1:4322/images/blog/git-branching-workflows/github-flow.png
curl -I http://127.0.0.1:4322/images/blog/git-branching-workflows/missing.png
curl --path-as-is -I 'http://127.0.0.1:4322/images/%2e%2e/style.css'
```

站点回归检查：

```bash
npm run build
```

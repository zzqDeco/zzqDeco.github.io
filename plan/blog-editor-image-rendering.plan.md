# Blog Editor Image Rendering

Status: Delivered

## Problem

Astro 站点可以从 `public/images/blog/...` 渲染文章图片，但本地 Blog Editor 服务此前没有暴露仓库 `public/images/*`。因此文章正文里的 `/images/blog/...` 在 `http://127.0.0.1:4322/images/...` 下返回 `404`。

同一个问题也影响编辑器上传图片：上传接口会把文件写入 `public/images/blog/<slug>/`，并返回 `/images/blog/...`，但前端随后无法从同一服务读回图片。

## Scope

- 只开放 `/images/*` 到 `public/images/*`。
- 不开放整个仓库根目录。
- 不改变 Astro 站点构建。
- 不新增 Mermaid、SVG 流程图或前端依赖。

## Implementation

- 在 `tools/blog-editor/server.mjs` 中增加 `sitePublicDir` 和 `/images/*` 静态路由。
- 保持 `/api/*`、`/vendor/blog-editor/*`、`/style.css`、`/app.js` 路由优先级。
- 对 `/images/*` 复用 content-type 映射和 path traversal 防护。
- 对本地图片响应使用 `Cache-Control: no-store`。
- 将 Image URL 输入改为允许 root-relative `/images/...`。
- 在前端 URL 校验中允许 `http://`、`https://` 和 `/images/`。
- 为 Milkdown/ProseMirror 图片块补齐稳定样式，避免大图撑爆编辑区。

## Verification

- `node --check tools/blog-editor/server.mjs`
- `node --check tools/blog-editor/public/app.js`
- `curl -I http://127.0.0.1:4322/images/blog/git-branching-workflows/github-flow.png`
- `curl -I http://127.0.0.1:4322/images/blog/git-branching-workflows/missing.png`
- `curl --path-as-is -I 'http://127.0.0.1:4322/images/%2e%2e/style.css'`
- 浏览器检查 `git-branching-workflows.md` 中 6 张 PNG 全部可见且无 broken image。
- 上传图片到临时 slug 后确认返回路径可立即读取。
- `npm run build`

## Follow-up

如果后续编辑器需要预览 `public/` 下其他类型资源，应新增明确的路由白名单，而不是把整个 `public/` 或仓库根目录作为静态目录暴露。

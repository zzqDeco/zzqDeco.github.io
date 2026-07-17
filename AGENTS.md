# AGENTS.md

Astro 6 静态博客，构建后部署到 GitHub Pages。本文件是入口索引，架构细节见 `doc/architecture.md`。

## 常用命令

与 `package.json` scripts 一致；Node 版本要求见 `engines`（>=22.12.0 <23）。

```bash
npm install            # 安装站点依赖
npm run dev            # 本地开发服务器
npm run build          # 构建到 dist/
npm run preview        # 预览构建产物
npm run editor         # 启动本地博客编辑器
```

编辑器是独立 npm 包（自带 lockfile），首次使用先装依赖，根目录 `npm install` 不会安装它：

```bash
npm --prefix tools/blog-editor install
```

编辑器用法详见 `tools/blog-editor/README.md`。

## 内容模型（`src/content.config.ts`）

- 文章位于 `src/content/blog/**/*.md`；frontmatter 必填 `title` / `description` / `pubDate`，可选 `updatedDate` / `tags` / `draft` / `image`。
- `draft: true` 仅在生产构建（PROD）时过滤，dev 下可见（`src/lib/blog.ts`）。
- 两个 tag 归一化后得到相同 slug 会 build 报错（`getBlogTags`）。
- 文章 id 首段不能是 `page` / `tag`（保留路由，`assertNoReservedBlogPostIds`）。

## 目录约定

- `src/pages/`：路由页面；`src/layouts/BaseLayout.astro`：唯一布局，props 含 `ogType` / `image` / `publishedTime` / `modifiedTime` / `noindex` / `showProfile`。
- `src/components/`：组件；`PostEntry.astro` 用 `actionLabel` 区分场景。
- `src/lib/`：共享逻辑；`image-manifest.json` 由图片管线脚本生成，勿手改。
- `public/images/blog/<dir>/`：优化后的 WebP 产物（会部署）；原图母版放 `assets/img-src/blog/<dir>/`，不要放回 `public/`。
- `tools/blog-editor/`：本地编辑器，独立 lockfile，不参与构建与部署。
- `tools/image-pipeline/`：图片优化管线。

## 图片管线

新文章图片流程见 `tools/image-pipeline/README.md`：原图入 `assets/img-src/blog/<dir>/`，运行 `convert_images.py` 生成多档 WebP 并更新 `src/lib/image-manifest.json`。rehype 插件 `src/lib/rehype-image-attrs.mjs` 在构建期自动注入宽高 / srcset / 加载属性，Markdown 里无需手写。

## 构建期行为约定

- KaTeX CSS 按需加载：`src/lib/math.ts` 用 Markdown AST 检测公式，仅含公式的文章页加载 KaTeX 样式（`src/pages/blog/[...slug].astro`）。
- prefetch：仅导航（`Header.astro`）与 `PostEntry.astro` 的链接带 `data-astro-prefetch`，新增链接保持这一范围。

## 部署

push / merge 到 `main` 触发 `.github/workflows/deploy.yml`，自动构建并部署到 GitHub Pages（Node 22），无需手动发布。

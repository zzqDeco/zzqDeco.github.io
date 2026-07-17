# 全量推广报告：博客图片优化（issue #6 全量阶段）

日期：2026-07-17 ｜ 分支：perf/issue-6-image-rollout ｜ 范围：全站剩余 20 个图片目录
（281 张 PNG）+ Diorama 补 srcset 档位 + 头像本地化。pilot 约定与阈值来源见
`PILOT-REPORT-diorama.md`（路线 1：换 .webp 扩展名，接受旧 .png 直链 404）。

## 1. 本轮改动总览

| 项 | 数量 |
| --- | ---: |
| 转换的原图 PNG（git mv 到 `assets/img-src/blog/`） | 281 张 / 20 个目录 |
| Diorama 补档（768/1280w，1920 档重出且字节不变） | 24 张 |
| WebP 产物（`public/images/blog/`，含各 srcset 档） | 719 个 |
| manifest 条目（`src/lib/image-manifest.json`） | 305 条 |
| 更新 markdown 引用的文章 | 20 篇 / 280 处 `.png`→`.webp` |
| 未动的非 PNG 资产（graphiti 的 6 svg + 2 gif） | 8 个（本轮不处理，见 §6） |
| 头像 | `public/images/avatar.webp`（144×144，5,460 B；母版 `assets/img-src/avatar/avatar.jpg` 460×460） |

管线与插件变更：

- `convert_images.py`：`--max-width` 单档 → `--widths 768,1280,1920` 多档（默认）；
  最大档保持 `<stem>.webp`（markdown 引用与 pilot 既有 URL 不变），小档命名
  `<stem>-<w>w.webp`；原图不足档宽按实际宽度出最大档，源宽 <768px 只出单档；
  新增**无损回退**：某档 lossy 产物大于原图时自动改试无损 WebP 取更小者
  （manifest 该档记 `"lossless": true`），保证产物不大于原图。
- `rehype-image-attrs.mjs`：manifest 含 ≥2 档 srcset 的图注入
  `srcset` + `sizes="(max-width: 760px) 100vw, 760px"`（正文容器 --w-reading 760px）；
  作者显式写出的同名属性不覆盖。
- `ProfileSidebar.astro` / `githubProfile.ts`：渲染改用本地 `/images/avatar.webp`
  （常量注入，API 返回的外链 avatarUrl 保留在数据中不再用于渲染）；
  fallback 数据同步改为本地路径。提交的静态资产，build 不写 `public/`。

## 2. 前后字节对比（public/images）

| 范围 | 前 | 后 | 相对 |
| --- | ---: | ---: | ---: |
| `public/images/` 全量 | 142,515,928 B（135.92 MiB） | 42,607,003 B（40.63 MiB） | **−70.1%** |
| 其中 blog 目录合计 | 136.42 MiB | 41.94 MiB | 30.7% |

"后"含全部 srcset 档位（部署体积）。**页面实际下载量远低于此**：浏览器按 sizes
只选一档，桌面最坏情况 = 最大档合计 20.6 MiB/全站（各篇 ≤2.2 MiB，见 §3）。

### 每目录对比（du 实测，部署字节含所有档位）

| 目录 | 前 KiB | 后 KiB | 比值 | 说明 |
| --- | ---: | ---: | ---: | --- |
| agentrl-… | 2,496 | 2,136 | 85.6% | |
| assetgen-… | 21,972 | 3,732 | 17.0% | |
| blended-nerf-… | 17,304 | 3,228 | 18.7% | |
| clippy-… | 1,780 | 1,392 | 78.2% | |
| componerf-… | 6,980 | 2,636 | 37.8% | |
| diorama-… | 2,156 | 4,284 | 198.7% | pilot 单档 1920 → 本轮 3 档，桌面页重不变（2.07 MiB） |
| dspark-… | 824 | 648 | 78.6% | |
| einops-… | 956 | 1,464 | 153.1% | 源为极小文字截图；页重 932→695 KiB（无损回退），部署多一档 768w |
| generative-recommenders-hstu | 2,668 | 2,724 | 102.1% | 源已小；页重 2,622→1,304 KiB，部署含多档 |
| git-branching-workflows | 912 | 488 | 53.5% | |
| graphiti-… | 1,788 | 1,828 | 102.2% | gif/svg 未动占大头；5 张 PNG 页重 817→423 KiB |
| hello | 0 | 0 | — | 无图 |
| layout2scene-… | 7,872 | 1,296 | 16.5% | |
| openfga-… | 492 | 412 | 83.7% | |
| openspg-… | 3,988 | 2,596 | 65.1% | |
| qwen3-vl-seg-… | 18,532 | 4,536 | 24.5% | |
| sao-… | 824 | 936 | 113.6% | 同 einops：无损回退后页重 804→353 KiB，部署含多档 |
| sf3d-… | 2,784 | 596 | 21.4% | |
| sim-… | 1,404 | 852 | 60.7% | |
| trellis2-… | 6,352 | 2,624 | 41.3% | |
| triposg-… | 36,652 | 3,724 | 10.2% | 全站最大目录，−89.8% |
| youtube-dnn-… | 960 | 816 | 85.0% | |

## 3. 阈值检查（pilot §9：单产物 ≤500 KiB、单篇 ≤3 MiB）

- **单产物 >500 KiB：0 个**。最大 triposg-fig12-textured-results.webp = 391 KiB（1920×2498）。
- **单篇页重（最大档合计 = 桌面最坏下载量）：全部 ≤ 2,255 KiB < 3 MiB**，
  最大 qwen3-vl-seg 2,255 KiB、 diorama 2,066 KiB、triposg 1,949 KiB。
- 部署体积（所有档位合计）超 3 MiB 的有 5 篇：qwen3-vl-seg 4,436 KiB、diorama 4,154、
  triposg 3,650、assetgen 3,601、blended-nerf 3,143 KiB。这是 srcset 三档并存的固有结果，
  **不影响任何客户端的实际下载量**（每图只取一档）；3 MiB 阈值按 pilot 语义指页重，均达标。
- **无损回退触发 19 张**（einops 11 + sao 8，均为文字截图/表格，lossy q80 反而大于原图）。
  例：einops-paper-pattern-language 源 127 KiB → lossy q80 190 KiB → 无损 98 KiB 采用。
  回退后 einops 篇页重 932→695 KiB（74.6%）、sao 篇 804→353 KiB（43.9%），
  全 manifest 无"产物大于原图"条目。

### Top 10 最大产物（磁盘实测）

| KiB | 文件 |
| ---: | --- |
| 391 | triposg-fig12-textured-results.webp |
| 353 | qwen3-vl-seg-fig08-single-instance-results.webp |
| 331 | qwen3-vl-seg-fig09-multiple-instance-results.webp |
| 322 | qwen3-vl-seg-fig10-phrasal-descriptive-results.webp |
| 318 | triposg-fig11-texture-free-results.webp |
| 275 | triposg-fig07-demo-comparison.webp |
| 267 | qwen3-vl-seg-fig03-sa1b-ors-examples.webp |
| 250 | assetgen-fig01-teaser.webp |
| 246 | diorama-fig14-more-ssdb.webp |
| 222 | graphiti-zep-paper-title-abstract.webp |

## 4. srcset 产物审计（dist）

- 正文 `/images/` 图共 312 张（21 篇文章）：
  - **254 张带 srcset + sizes**（多档）；
  - **50 张单档无 srcset**：源宽 510–766px（<768px 最小档），按规则只出原宽单档，符合预期；
  - **8 张无 width/height**：graphiti 的 gif/svg（本轮未处理，无 manifest 尺寸，
    插件按设计只注入 loading/decoding）。
- 每篇文章第一张内容图 `loading="eager"` + `fetchpriority="high"`（21/21），
  其余 `loading="lazy"`，全部 `decoding="async"`。
- 头像 `<img>` 为组件模板（rehype 不触碰），本地 `/images/avatar.webp`，宽高 72 写死。

## 5. 头像本地化（issue #6 拆分并入项）

- `https://avatars.githubusercontent.com/u/41999232?v=4`（460×460 JPEG，27 KiB）
  → 144×144 WebP q85 = **5,460 B**，提交为 `public/images/avatar.webp`（72 CSS px × DPR 2）；
  母版存 `assets/img-src/avatar/avatar.jpg`。
- 首屏网络面板实测：不再出现 `avatars.githubusercontent.com` 第三方请求，
  仅 `GET /images/avatar.webp [200]`。

## 6. 本轮未处理项（明示）

- graphiti 目录 6 svg + 2 gif：svg 为矢量无需光栅优化，gif 为动画（WebP 动画管线
  本轮不引入）；保留原格式原 URL，报告中备案。两张 gif 合计约 0.9 MiB，lazy 加载。
- `public/images/og-default.png`：og:image 站点级资产，与 #10 协调，不动。
- `generative-recommenders-hstu/hstu-fig08-sequential-vs-gr.png`：仓库内无任何引用的
  孤儿文件。仍按"全部转换"处理（源已 git mv、产物 webp 已部署、manifest 已收录），
  保持与转换前一致的公开 URL 语义；是否删除留给后续清理决定。

## 7. 验证记录

- `npm run build`：通过，126 页。
- dist 404 扫描：提取全部 HTML/CSS/XML 中 `/images/` 引用 **726 个，0 缺失**；
  manifest 记录的 305 条产物（含各档）在 dist 全部存在。
- `npm run preview` + chrome-devtools 抽查 3 篇图多的文章（亮/暗两主题截图）：
  - triposg：16/16 图加载成功、0 broken，首屏仅 teaser（eager）+ 本地头像；
  - blended-nerf：16/16，fig07 纹理对比图两主题下清晰；
  - qwen3-vl-seg：16/16，fig08 九列分割对比图行/列标签均可读；
  - 三篇均无 404、无布局异常；懒加载生效（滚动前折叠线下图 0 请求）。
- 引用更新复核：20 篇 md 共 280 处替换；全仓 grep 无残留 `/images/…​.png` 引用；
  代码块/行内代码中的 `.png` 文字（depth.png、chair1.png 等）未误改
  （替换按"真实存在的文件名 + /images/blog/ 前缀"精确匹配）。
- Diorama 重跑回归：1920 档字节与 pilot 完全一致（2,166,118 B），引用零改动。

## 8. 遗留与后续

- 旧 `.png` 直链 404（路线 1 既定决策，pilot §2 已评估；文章发布时间近、无外部引用记录）。
- einops/sao 类文字截图若未来想进一步省体积，可评估调色板 PNG 或 AVIF；当前无损回退
  已保证不回退于原图。
- gif 动画优化（animated WebP/AVIF）与 svg 未纳入本轮范围。

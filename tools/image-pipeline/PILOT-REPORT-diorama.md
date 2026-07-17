# Pilot 报告：Diorama 文章图片优化（issue #6）

日期：2026-07-17 ｜ 分支：perf/issue-6-image-pilot ｜ 范围：仅 Diorama 一篇（24 张图）
方法：`tools/image-pipeline/convert_images.py`，限宽 1920px（不放大）、WebP q80 method=6、
对照 PNG 同尺寸无损（optimize + compress_level=9）。逐文件数据见 `pilot-diorama-conversion.log`。

## 1. 三组字节对比（24 张合计）

| 组别 | 字节数 | MiB | 相对原图 |
| --- | ---: | ---: | ---: |
| 原始 PNG（assets/img-src/） | 25,884,507 | 24.69 | 100% |
| **WebP @1920/q80（部署产物）** | **2,166,118** | **2.07** | **8.4%（−91.6%，11.9 倍压缩）** |
| 对照 PNG @1920 无损 | 19,117,350 | 18.23 | 73.9%（−26.1%） |

对照 PNG 体积是 WebP 的 **8.8 倍**；差距主要来自 PNG 无损编码对摄影/渲染内容的固有劣势。
最大的单张产物：fig14-more-ssdb.webp = 246 KiB（原 4.74 MiB）。

## 2. URL 兼容策略对比（数据与推荐；最终决策留给 review）

| | 路线 1：换 .webp 扩展名（本 pilot 已实施） | 路线 2：原地压 PNG 保 URL |
| --- | --- | --- |
| 整篇字节 | 2.07 MiB | 18.23 MiB |
| 首图（teaser） | 167 KiB | 1.70 MiB |
| 旧 `.png` 直链 | **404**（GitHub Pages 无重定向能力） | 不变 |
| 站内引用 | 已全部改 .webp 并 grep 复核 | 不用改 |
| og:image | 不受影响（用 `/images/og-default.png`，frontmatter 无 image 字段） | 不受影响 |
| 进一步压缩空间 | 可降档/降质、可加 srcset | 无损已到极限；再压需有损量化（新工具链） |

旧链 404 的实际影响面评估：文章发布于 2026-07-15（pilot 时仅 2 天），仓库无外部引用记录；
图片直链被第三方引用的概率低；搜索引擎图片索引会随重新抓取更新为新 URL。

**推荐路线 1**。两条路线相差 16.16 MiB/篇（8.8 倍），路线 2 保住的旧链价值对本站接近零。
若 review 仍不接受旧链 404，折中方案是路线 2 产物也放 public/（旧 URL 指向压 PNG、
markdown 指 WebP），代价是部署体积 20.3 MiB 且后续每篇都要双份产物——不推荐。

## 3. 产物 `<img>` 属性（dist 实测）

- 24/24 内容图：`width`/`height`（读自 manifest 的产物实际尺寸）+ `decoding="async"`；
- 文档第一张内容图（fig01-teaser）：`loading="eager"` + `fetchpriority="high"`；
- 其余 23 张：`loading="lazy"`；
- ProfileSidebar 头像（组件内写死 `loading="lazy"`）未被插件触碰——rehype 只处理 markdown 产物；
- 插件不覆盖作者已显式写出的同名属性。

## 4. 首屏请求与体积（`npm run preview` + chrome-devtools 实测）

- 初始加载图片请求：**1 张内容图**（fig01-teaser.webp，171,026 B / 167 KiB，200），
  外加 GitHub 头像（第三方，不在本 issue 范围）；
- 折叠线下 23 张 lazy 图首屏 **0 请求**；滚动后 24 张全部加载成功，无 404；
- 改造前首图 teaser.png = 3,573,904 B（3.41 MiB）→ 首屏图片体积 **−95.2%**。

## 5. CLS

- 全页 layout-shift 合计 0.0093（远低于 0.1 的"good"阈值）；
- 位移来源是 Shiki 代码块（§2 的 JSON 示例，字体/高亮相关），**与图片无关**；
- 图片贡献 CLS = 0：`width/height` 属性 + global.css 既有 `img { max-width: 100%; height: auto }`
  使浏览器在加载前预留纵横比空间，改造前图片加载时的跳动消除。

## 6. 图表文字可读性抽查（按 VISUAL-CHECKLIST.md，chrome-devtools 截图核对）

| 图 | 结论 |
| --- | --- |
| fig02-pipeline（文字密度最高） | 模块框内最小文字（"Text query: Ergonomic office chair…"等）清晰可读，箭头/框线不断裂 |
| table01-zero-shot-9dof（1690 宽未缩放） | 表格数字、±/↑↓ 符号、加粗最优值均可逐一辨认，q80 对文字截图无可见损伤 |
| fig07-scannet-comparison | 四列（GT/ROCA/DiffCAD/Ours）列标题清晰，列间纹理差异可分辨，无压缩噪声掩盖细节 |
| fig01-teaser（首图） | 三面板标签与斜体 prompt 文字可读，渐变区域无 banding |

结论：**1920px + WebP q80 对本篇全部 24 张图通过视觉抽查**，无需调参重跑。

## 7. 移动端候选尺寸建议（实测阶梯，WebP q80 全篇合计）

| 限宽 | 全篇合计 | 适用 |
| ---: | ---: | --- |
| 1920 | 2.07 MiB | 桌面正文列（760 CSS px，实测带 TOC 约 664）DPR 2–2.9x |
| 1600 | 1.78 MiB | 桌面 DPR 2x 恰好覆盖 |
| 1280 | 1.40 MiB | 移动端 DPR 3x（390px 视口 ≈ 1074 物理 px） |
| 768 | 0.78 MiB | 移动端 DPR 2x（≈ 716 物理 px） |

pilot 按 issue 方案 A 先上单档 1920。全量推广时建议生成 `srcset` 768/1280/1920 +
`sizes="(max-width: 760px) 100vw, 760px"`，移动首图可从 167 KiB 再降到约 60–80 KiB；
需要插件同步支持 srcset 注入，届时再评估。

## 8. 目录与入库决策

- `assets/img-src/`（原图母版 24.69 MiB）：**入库**。管线可复现的输入；字节量与从
  public/ 移出的相同，仓库历史净增 ≈ 0；未来调高参数重跑必须有母版。
- `assets/img-opt-png/`（对照 PNG 18.23 MiB）：**不入库**（已加 .gitignore）。一次性对比证据，
  字节数据已留在报告与 conversion log；一条命令可重新生成；避免 18 MiB 永久进入 git 历史。
  文件仍保留在本地工作区供 review 查看。若 review 选定路线 2，重新生成并部署即可。
- `public/images/blog/diorama-…/`：仅 24 张 WebP；原图已不在 public/。

## 9. 全站推广阈值建议（由本 pilot 数据驱动）

- 默认参数：限宽 1920 + WebP q80（本篇 24/24 通过视觉抽查）；
- 验收阈值建议：单产物 ≤ 500 KiB、单篇合计 ≤ 3 MiB（本篇最大单张 246 KiB、整篇 2.07 MiB）；
  超限再个案降档，不预设全站总量目标；
- 外推参考（非承诺）：全站 158.44 MiB 按 −91.6% 外推约 13–20 MiB 量级，以逐篇实测为准。

## 10. 验证记录

- `npm run build`：通过，126 页；产物 HTML 属性见第 3 节；
- `npm run preview` + chrome-devtools：第 4–6 节数据；
- `public/` 下 Diorama 原图：0 张（`find … -name '*.png'` 为空）；
- 全仓 grep 旧 PNG 引用：仅 conversion log 中的历史文件名记录。

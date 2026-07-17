# image-pipeline — 博客图片优化管线

issue #6 的可复现图片管线。pilot 范围：Diorama 一篇文章；全站推广复用同一套脚本与流程。

## 目录约定

| 位置 | 作用 | 是否部署 | 是否入库 |
| --- | --- | --- | --- |
| `assets/img-src/blog/<slug>/` | 原始 PNG 源母版 | 否（不在 `public/`） | 是（可复现的输入） |
| `public/images/blog/<slug>/` | 优化后 WebP 产物 | 是 | 是 |
| `assets/img-opt-png/blog/<slug>/` | 同尺寸无损对照 PNG | 否 | 否（gitignored；一次性对比证据，可用脚本重新生成） |
| `src/lib/image-manifest.json` | 产物 URL → 宽高/字节清单 | 否（构建期读取） | 是 |

原图**不得**放回 `public/`（会原样部署且绕过优化）。

## 对一篇新文章跑管线

```bash
# 1. 原图搬出部署目录（保留 git 历史）
git mv public/images/blog/<slug> assets/img-src/blog/<slug>
mkdir -p public/images/blog/<slug>

# 2. 转换（参数见脚本头部注释；档位 1920/1600/1280/768 可调）
python3 tools/image-pipeline/convert_images.py \
  --src assets/img-src/blog/<slug> \
  --out-webp public/images/blog/<slug> \
  --out-png assets/img-opt-png/blog/<slug> \
  --url-prefix /images/blog/<slug> \
  --manifest src/lib/image-manifest.json \
  --max-width 1920 --quality 80

# 3. markdown 引用 .png → .webp（注意别把代码块里的文字误改）

# 4. 按 VISUAL-CHECKLIST.md 抽查，然后 npm run build 验证
```

`src/lib/rehype-image-attrs.mjs` 在构建期读 manifest，自动为 markdown 产物 `<img>`
注入 `width/height`、`decoding="async"`，并按位置注入 `loading`
（文档第一张内容图 `eager` + `fetchpriority="high"`，其余 `lazy`），无需手动标注。

## 文件

- `convert_images.py` — 转换脚本（用法见文件头 docstring 或 `--help`）
- `VISUAL-CHECKLIST.md` — 转换后必做的视觉抽查清单
- `PILOT-REPORT-diorama.md` — Diorama pilot 数据与结论
- `pilot-diorama-conversion.log` — pilot 转换的原始输出留档

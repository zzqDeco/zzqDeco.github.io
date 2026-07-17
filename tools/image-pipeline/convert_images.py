#!/usr/bin/env python3
"""convert_images.py — 博客图片优化管线（issue #6）

用途
----
把一篇文章的原始 PNG（源母版，存放在仓库非部署目录 assets/img-src/）批量转换为：

1. 部署用 WebP：限宽 + 质量压缩，写入 public/ 下的部署目录；
2. 同尺寸对照 PNG：与 WebP 相同尺寸的无损优化 PNG，写入非部署目录，
   用于"原地压 PNG 保 URL"与"换 .webp 扩展名"两条 URL 策略的字节数对比；
3. 尺寸清单 manifest.json：记录每张产物图的 URL → {width, height, bytes}，
   供 src/lib/rehype-image-attrs.mjs 在构建期为 <img> 注入 width/height。

依赖：系统 python3 + Pillow 9.4+（需 libwebp，macOS 自带 python3 已满足）。
不引入任何第三方命令行工具，可复现。

用法
----
  python3 tools/image-pipeline/convert_images.py \
      --src        assets/img-src/blog/<post-slug> \
      --out-webp   public/images/blog/<post-slug> \
      --out-png    assets/img-opt-png/blog/<post-slug> \
      --url-prefix /images/blog/<post-slug> \
      --manifest   src/lib/image-manifest.json \
      --max-width  1920 \
      --quality    80

参数
----
  --src         源母版目录（只读，不改动其中的文件）
  --out-webp    WebP 产物目录（通常指到 public/ 下；传 "-" 可跳过 WebP）
  --out-png     对照 PNG 产物目录（传 "-" 可跳过）
  --url-prefix  产物在站点中的 URL 前缀，用于生成 manifest 的键
  --manifest    尺寸清单 JSON 路径（存在则合并更新，按 URL 去重）
  --max-width   产物最大宽度（像素）。更宽的图等比缩小，更窄的图保持原尺寸，
                绝不放大。常用档位：1920 / 1600 / 1280 / 768
  --quality     WebP 质量（1-100，默认 80；论文截图/渲染图 80 肉眼无损）

输出
----
  - 逐文件与汇总的三组字节数（原 PNG / WebP / 对照 PNG）打到 stdout，
    可 `| tee` 留存为 pilot 数据记录；
  - manifest.json 供 rehype 插件读取。

注意
----
  - 脚本不修改/删除 --src 中的原图；原图从 public/ 搬出用 git mv 手动完成。
  - PNG 对照组是无损压缩（ Pillow optimize + compress_level=9 ），
    体积收益主要来自限宽降采样，这是"保 URL"路线的真实上限（不做调色板量化）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# 禁用 Pillow 的图片像素上限告警；本仓库图均为本地可信源母版
Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="博客图片优化管线：限宽 + WebP/对照PNG 双产物 + 尺寸清单",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True, type=Path, help="源母版目录（只读）")
    p.add_argument("--out-webp", required=True, type=Path, help="WebP 产物目录，'-' 跳过")
    p.add_argument("--out-png", required=True, type=Path, help="对照 PNG 产物目录，'-' 跳过")
    p.add_argument("--url-prefix", required=True,
                   help="产物 URL 前缀，如 /images/blog/<post-slug>")
    p.add_argument("--manifest", type=Path, default=None,
                   help="尺寸清单 JSON（存在则合并更新）")
    p.add_argument("--max-width", type=int, default=1920,
                   help="产物最大宽度，常用档位 1920/1600/1280/768")
    p.add_argument("--quality", type=int, default=80, help="WebP 质量 1-100")
    return p.parse_args()


def fit_width(img: Image.Image, max_width: int) -> tuple[Image.Image, bool]:
    """等比限宽，不放大。返回 (图片, 是否缩放过)。"""
    w, h = img.size
    if w <= max_width:
        return img, False
    new_h = round(h * max_width / w)
    return img.resize((max_width, new_h), Image.LANCZOS), True


def main() -> int:
    args = parse_args()
    if not 1 <= args.quality <= 100:
        print("error: --quality 必须在 1-100", file=sys.stderr)
        return 2

    srcs = sorted(args.src.glob("*.png"))
    if not srcs:
        print(f"error: {args.src} 下没有 .png 文件", file=sys.stderr)
        return 1

    do_webp = str(args.out_webp) != "-"
    do_png = str(args.out_png) != "-"
    if do_webp:
        args.out_webp.mkdir(parents=True, exist_ok=True)
    if do_png:
        args.out_png.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    total_src = total_webp = total_png = 0
    header = f"{'file':<46} {'src dims':>11} {'out dims':>11} " \
             f"{'src KiB':>9} {'webp KiB':>9} {'png KiB':>9}"
    print(header)
    print("-" * len(header))

    for src_path in srcs:
        src_bytes = src_path.stat().st_size
        with Image.open(src_path) as im:
            im.load()
            src_w, src_h = im.size
            img, _ = fit_width(im, args.max_width)
            out_w, out_h = img.size

            webp_bytes = png_bytes = 0
            if do_webp:
                webp_path = args.out_webp / f"{src_path.stem}.webp"
                img.save(webp_path, "WEBP", quality=args.quality, method=6)
                webp_bytes = webp_path.stat().st_size
            if do_png:
                png_path = args.out_png / src_path.name
                img.save(png_path, "PNG", optimize=True, compress_level=9)
                png_bytes = png_path.stat().st_size

        if args.manifest and do_webp:
            url = f"{args.url_prefix.rstrip('/')}/{src_path.stem}.webp"
            manifest[url] = {
                "width": out_w,
                "height": out_h,
                "bytes": webp_bytes,
                "sourceBytes": src_bytes,
            }

        total_src += src_bytes
        total_webp += webp_bytes
        total_png += png_bytes
        src_dims = f"{src_w}x{src_h}"
        print(f"{src_path.name:<46} {src_dims:>11} {f'{out_w}x{out_h}':>11} "
              f"{src_bytes/1024:>9.0f} {webp_bytes/1024 if do_webp else 0:>9.0f} "
              f"{png_bytes/1024 if do_png else 0:>9.0f}")

    print("-" * len(header))
    print(f"{'TOTAL (' + str(len(srcs)) + ' files)':<46} {'':>11} {'':>11} "
          f"{total_src/1024:>9.0f} {total_webp/1024:>9.0f} {total_png/1024:>9.0f}")
    mib = 1024 * 1024
    print(f"\n源 PNG 合计 : {total_src} bytes ({total_src/mib:.2f} MiB)")
    if do_webp:
        print(f"WebP 合计   : {total_webp} bytes ({total_webp/mib:.2f} MiB) "
              f"({total_webp/total_src:.1%} of src)")
    if do_png:
        print(f"对照PNG 合计: {total_png} bytes ({total_png/mib:.2f} MiB) "
              f"({total_png/total_src:.1%} of src)")

    if args.manifest and do_webp:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(dict(sorted(manifest.items())), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"manifest    : {args.manifest} ({len(manifest)} entries)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

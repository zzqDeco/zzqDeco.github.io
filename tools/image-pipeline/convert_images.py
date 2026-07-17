#!/usr/bin/env python3
"""convert_images.py — 博客图片优化管线（issue #6）

用途
----
把一篇文章的原始 PNG（源母版，存放在仓库非部署目录 assets/img-src/）批量转换为：

1. 部署用多档 WebP（响应式 srcset 产物）：按宽度档（默认 768/1280/1920）等比限宽
   （不放大）+ 质量压缩，写入 public/ 下的部署目录。最大档保持 `<stem>.webp`
   文件名（markdown 引用与既有 URL 不变），较小档命名为 `<stem>-<w>w.webp`。
   原图不足最大档宽度的，按实际宽度出最大档，并跳过不小于它的档位。
2. 同尺寸对照 PNG（可选）：与最大档 WebP 相同尺寸的无损优化 PNG，写入非部署目录，
   用于"原地压 PNG 保 URL"与"换 .webp 扩展名"两条 URL 策略的字节数对比；
   全量推广已选定 WebP 路线，日常运行传 `--out-png -` 跳过。
3. 尺寸清单 manifest.json：记录每张产物图的 URL → {width, height, bytes,
   sourceBytes, srcset:[{url,width,height,bytes}...]}，供
   src/lib/rehype-image-attrs.mjs 在构建期为 <img> 注入 width/height/srcset/sizes。

依赖：系统 python3 + Pillow 9.4+（需 libwebp，macOS 自带 python3 已满足）。
不引入任何第三方命令行工具，可复现。

用法
----
  python3 tools/image-pipeline/convert_images.py \
      --src        assets/img-src/blog/<post-slug> \
      --out-webp   public/images/blog/<post-slug> \
      --out-png    - \
      --url-prefix /images/blog/<post-slug> \
      --manifest   src/lib/image-manifest.json \
      --widths     768,1280,1920 \
      --quality    80

参数
----
  --src         源母版目录（只读，不改动其中的文件）
  --out-webp    WebP 产物目录（通常指到 public/ 下；传 "-" 可跳过 WebP）
  --out-png     对照 PNG 产物目录（传 "-" 可跳过；仅出最大档尺寸）
  --url-prefix  产物在站点中的 URL 前缀，用于生成 manifest 的键
  --manifest    尺寸清单 JSON 路径（存在则合并更新，按 URL 去重）
  --widths      逗号分隔的宽度档（升序），默认 768,1280,1920。
                更宽的图逐档等比缩小；原图宽度即最大档（绝不放大），
                并跳过 ≥ 原图宽度的档位。
  --max-width   单档模式（pilot 兼容参数）：显式传入时等价于 --widths <N>
  --quality     WebP 质量（1-100，默认 80；论文截图/渲染图 80 肉眼无损）

输出
----
  - 逐文件与汇总的字节数（原 PNG / 各档 WebP / 对照 PNG）打到 stdout，
    可 `| tee` 留存为数据记录；
  - manifest.json 供 rehype 插件读取。

注意
----
  - 脚本不修改/删除 --src 中的原图；原图从 public/ 搬出用 git mv 手动完成。
  - PNG 对照组是无损压缩（ Pillow optimize + compress_level=9 ），
    体积收益主要来自限宽降采样，这是"保 URL"路线的真实上限（不做调色板量化）。
  - 文字截图/线稿类原图本身极小，lossy WebP 可能反而更大：脚本对
    "lossy 产物 > 原图" 的单档自动改试无损 WebP 并取两者更小者
    （理论上 lossless 也可能仍大于原图，此时保留更小者而非保证不超原图）；
    manifest 中该档标记 "lossless": true。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image

# 禁用 Pillow 的图片像素上限告警；本仓库图均为本地可信源母版
Image.MAX_IMAGE_PIXELS = None

DEFAULT_WIDTHS = "768,1280,1920"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="博客图片优化管线：多档限宽 WebP（srcset）+ 可选对照PNG + 尺寸清单",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", required=True, type=Path, help="源母版目录（只读）")
    p.add_argument("--out-webp", required=True, type=Path, help="WebP 产物目录，'-' 跳过")
    p.add_argument("--out-png", required=True, type=Path, help="对照 PNG 产物目录，'-' 跳过")
    p.add_argument("--url-prefix", required=True,
                   help="产物 URL 前缀，如 /images/blog/<post-slug>")
    p.add_argument("--manifest", type=Path, default=None,
                   help="尺寸清单 JSON（存在则合并更新）")
    p.add_argument("--widths", default=None,
                   help="逗号分隔宽度档（升序），默认 " + DEFAULT_WIDTHS)
    p.add_argument("--max-width", type=int, default=None,
                   help="单档模式（pilot 兼容）：显式传入时等价于 --widths <N>")
    p.add_argument("--quality", type=int, default=80, help="WebP 质量 1-100")
    return p.parse_args()


def parse_widths(args: argparse.Namespace) -> list[int]:
    raw = args.widths if args.widths else (
        str(args.max_width) if args.max_width else DEFAULT_WIDTHS)
    widths = sorted({int(w) for w in raw.split(",") if w.strip()})
    if not widths or any(w <= 0 for w in widths):
        raise ValueError(f"--widths 解析失败: {raw!r}")
    return widths


def fit_width(img: Image.Image, max_width: int) -> tuple[Image.Image, bool]:
    """等比限宽，不放大。返回 (图片, 是否缩放过)。"""
    w, h = img.size
    if w <= max_width:
        return img, False
    new_h = round(h * max_width / w)
    return img.resize((max_width, new_h), Image.LANCZOS), True


def tier_widths(src_width: int, widths: list[int]) -> list[int]:
    """实际出图档位（升序）：最大档 = min(原图宽, 配置最大档)，跳过不小于它的档。"""
    top = min(src_width, widths[-1])
    return [w for w in widths if w < top] + [top]


def tier_name(stem: str, width: int, top: int) -> str:
    """最大档保持 <stem>.webp（markdown 引用与既有 URL 不变），小档加 -<w>w 后缀。"""
    return f"{stem}.webp" if width == top else f"{stem}-{width}w.webp"


def main() -> int:
    args = parse_args()
    if not 1 <= args.quality <= 100:
        print("error: --quality 必须在 1-100", file=sys.stderr)
        return 2
    try:
        widths = parse_widths(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
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

    total_src = total_webp = total_webp_top = total_png = 0
    header = f"{'file':<46} {'src dims':>11} {'top dims':>11} " \
             f"{'src KiB':>9} {'top KiB':>9} {'all KiB':>9} {'png KiB':>9}"
    print(header)
    print("-" * len(header))

    for src_path in srcs:
        src_bytes = src_path.stat().st_size
        with Image.open(src_path) as im:
            im.load()
            src_w, src_h = im.size
            tiers = tier_widths(src_w, widths)
            top = tiers[-1]

            variants: list[dict] = []
            top_w = top_h = top_bytes = 0
            all_bytes = 0
            if do_webp:
                for w in tiers:
                    img, _ = fit_width(im, w)
                    out_w, out_h = img.size
                    webp_path = args.out_webp / tier_name(src_path.stem, w, top)
                    img.save(webp_path, "WEBP", quality=args.quality, method=6)
                    webp_bytes = webp_path.stat().st_size
                    lossless = False
                    # 文字截图/线稿类内容 lossy 产物可能比原 PNG 还大；
                    # 此时改试无损 WebP，取更小者，保证产物不大于原图
                    if webp_bytes > src_bytes:
                        buf = io.BytesIO()
                        img.save(buf, "WEBP", lossless=True, quality=100, method=6)
                        if buf.tell() < webp_bytes:
                            webp_path.write_bytes(buf.getvalue())
                            webp_bytes = buf.tell()
                            lossless = True
                    variants.append({
                        "url": f"{args.url_prefix.rstrip('/')}/{webp_path.name}",
                        "width": out_w,
                        "height": out_h,
                        "bytes": webp_bytes,
                        **({"lossless": True} if lossless else {}),
                    })
                    all_bytes += webp_bytes
                    if w == top:
                        top_w, top_h, top_bytes = out_w, out_h, webp_bytes
            else:
                top_img, _ = fit_width(im, top)
                top_w, top_h = top_img.size

            png_bytes = 0
            if do_png:
                img, _ = fit_width(im, top)
                png_path = args.out_png / src_path.name
                img.save(png_path, "PNG", optimize=True, compress_level=9)
                png_bytes = png_path.stat().st_size

        if args.manifest and do_webp:
            url = f"{args.url_prefix.rstrip('/')}/{src_path.stem}.webp"
            entry = {
                "width": top_w,
                "height": top_h,
                "bytes": top_bytes,
                "sourceBytes": src_bytes,
            }
            if len(variants) >= 2:
                entry["srcset"] = variants
            manifest[url] = entry

        total_src += src_bytes
        total_webp += all_bytes
        total_webp_top += top_bytes
        total_png += png_bytes
        src_dims = f"{src_w}x{src_h}"
        n_lossless = sum(1 for v in variants if v.get("lossless"))
        tiers_note = f"{len(tiers)}档" if len(tiers) >= 2 else "单档"
        if n_lossless:
            tiers_note += f"({n_lossless}无损)"
        print(f"{src_path.name:<46} {src_dims:>11} {f'{top_w}x{top_h}':>11} "
              f"{src_bytes/1024:>9.0f} {top_bytes/1024 if do_webp else 0:>9.0f} "
              f"{all_bytes/1024 if do_webp else 0:>9.0f} "
              f"{png_bytes/1024 if do_png else 0:>9.0f} {tiers_note}")

    print("-" * len(header))
    print(f"{'TOTAL (' + str(len(srcs)) + ' files)':<46} {'':>11} {'':>11} "
          f"{total_src/1024:>9.0f} {total_webp_top/1024:>9.0f} "
          f"{total_webp/1024:>9.0f} {total_png/1024:>9.0f}")
    mib = 1024 * 1024
    print(f"\n源 PNG 合计      : {total_src} bytes ({total_src/mib:.2f} MiB)")
    if do_webp:
        print(f"WebP 最大档合计  : {total_webp_top} bytes ({total_webp_top/mib:.2f} MiB) "
              f"({total_webp_top/total_src:.1%} of src，单页桌面端最坏情况下载量)")
        print(f"WebP 全档位合计  : {total_webp} bytes ({total_webp/mib:.2f} MiB) "
              f"({total_webp/total_src:.1%} of src，部署体积)")
    if do_png:
        print(f"对照PNG 合计     : {total_png} bytes ({total_png/mib:.2f} MiB) "
              f"({total_png/total_src:.1%} of src)")

    if args.manifest and do_webp:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(dict(sorted(manifest.items())), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"manifest         : {args.manifest} ({len(manifest)} entries)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

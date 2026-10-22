/**
 * rehype-image-attrs.mjs — 为 markdown 渲染产物中的 <img> 注入加载与尺寸属性，并校验本地图片存在性（issue #6）
 *
 * 只作用于经 remark/rehype 处理的 markdown 内容（即 .article-body 内的文章图片），
 * 不影响 .astro 组件中手写的 <img>（如 ProfileSidebar 头像——组件模板不经过 rehype）。
 *
 * 行为：
 *  - 对以 `/images/` 开头的本地图片路径，构建期强校验 `public/images/...` 物理文件是否存在
 *    （含 manifest 中的 srcset 派生档位），缺失直接抛错阻断构建，防止死链静默上线；
 *  - 从 src/lib/image-manifest.json（由 tools/image-pipeline/convert_images.py 生成）
 *    按图片 URL 查出产物尺寸，注入 width/height（配合 global.css 的
 *    `img { max-width: 100%; height: auto }`，浏览器据此预留纵横比空间，消除 CLS）；
 *  - manifest 中带 srcset 多档产物（768/1280/1920 等宽度档）的图片，注入
 *    srcset 与 sizes="(max-width: 760px) 100vw, 760px"（正文容器 --w-reading 760px，
 *    窄屏图片铺满视口宽，宽屏不超过正文列宽）；
 *  - 全部图片注入 decoding="async"；
 *  - 加载属性按位置区分：文档中第一张内容图 loading="eager" + fetchpriority="high"
 *    （视为 LCP 候选），其余 loading="lazy"；
 *  - 作者已在 markdown/HTML 中显式写出的同名属性不被覆盖。
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const manifestPath = new URL('./image-manifest.json', import.meta.url);
const defaultPublicDir = fileURLToPath(new URL('../../public', import.meta.url));

function loadManifest() {
  try {
    return JSON.parse(readFileSync(manifestPath, 'utf8'));
  } catch {
    return {};
  }
}

function assertLocalImageExists(url, publicDir) {
  if (typeof url !== 'string' || !url.startsWith('/images/')) return;
  const cleanPath = url.split(/[?#]/)[0];
  const filePath = path.join(publicDir, cleanPath.slice(1));
  if (!existsSync(filePath)) {
    throw new Error(`Local blog image "${url}" does not exist in public/ (${filePath}).`);
  }
}

// hast 树很小，手写递归遍历即可，避免新增 unist-util-visit 依赖
function walk(node, visit) {
  if (!node || typeof node !== 'object') return;
  visit(node);
  if (Array.isArray(node.children)) {
    for (const child of node.children) walk(child, visit);
  }
}

export default function rehypeImageAttrs(options = {}) {
  const manifest = options.manifest ?? loadManifest();
  const publicDir = options.publicDir === undefined ? defaultPublicDir : options.publicDir;

  return (tree) => {
    let seenFirstImage = false;

    walk(tree, (node) => {
      if (node.type !== 'element' || node.tagName !== 'img') return;

      const props = (node.properties ??= {});
      const src = typeof props.src === 'string' ? props.src : undefined;

      if (src && publicDir) {
        assertLocalImageExists(src, publicDir);
      }

      const meta = src ? manifest[src] : undefined;

      if (meta) {
        props.width ??= meta.width;
        props.height ??= meta.height;
        if (Array.isArray(meta.srcset) && meta.srcset.length >= 2) {
          if (publicDir) {
            for (const variant of meta.srcset) {
              assertLocalImageExists(variant.url, publicDir);
            }
          }
          props.srcset ??= meta.srcset.map((v) => `${v.url} ${v.width}w`).join(', ');
          props.sizes ??= '(max-width: 760px) 100vw, 760px';
        }
      }

      props.decoding ??= 'async';

      if (props.loading === undefined) {
        if (!seenFirstImage) {
          props.loading = 'eager';
          props.fetchpriority ??= 'high';
        } else {
          props.loading = 'lazy';
        }
      }
      seenFirstImage = true;
    });
  };
}

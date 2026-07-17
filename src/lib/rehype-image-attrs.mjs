/**
 * rehype-image-attrs.mjs — 为 markdown 渲染产物中的 <img> 注入加载与尺寸属性（issue #6）
 *
 * 只作用于经 remark/rehype 处理的 markdown 内容（即 .article-body 内的文章图片），
 * 不影响 .astro 组件中手写的 <img>（如 ProfileSidebar 头像——组件模板不经过 rehype）。
 *
 * 行为：
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
 *
 * 未收录进 manifest 的图片（尚未走管线的旧文章）没有可靠尺寸来源，
 * 只注入 loading/decoding，不注入 width/height——与该插件接入前的现状一致。
 */

import { readFileSync } from 'node:fs';

const manifestPath = new URL('./image-manifest.json', import.meta.url);

function loadManifest() {
  try {
    return JSON.parse(readFileSync(manifestPath, 'utf8'));
  } catch {
    return {};
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

export default function rehypeImageAttrs() {
  const manifest = loadManifest();

  return (tree) => {
    let seenFirstImage = false;

    walk(tree, (node) => {
      if (node.type !== 'element' || node.tagName !== 'img') return;

      const props = (node.properties ??= {});
      const src = typeof props.src === 'string' ? props.src : undefined;
      const meta = src ? manifest[src] : undefined;

      if (meta) {
        props.width ??= meta.width;
        props.height ??= meta.height;
        if (Array.isArray(meta.srcset) && meta.srcset.length >= 2) {
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

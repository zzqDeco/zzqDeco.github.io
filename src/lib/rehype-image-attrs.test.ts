import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { test } from 'node:test';

import rehypeImageAttrs from './rehype-image-attrs.mjs';

function makeTree(imgs: Array<Record<string, unknown>>) {
  return {
    type: 'root',
    children: imgs.map((properties) => ({
      type: 'element',
      tagName: 'img',
      properties: { ...properties },
    })),
  };
}

test('rehypeImageAttrs：首图 eager + high，后续图 lazy，并注入 manifest 宽高与 srcset', () => {
  const tempPublic = mkdtempSync(path.join(tmpdir(), 'blog-img-test-'));
  try {
    const postImgDir = path.join(tempPublic, 'images/blog/demo');
    mkdirSync(postImgDir, { recursive: true });
    writeFileSync(path.join(postImgDir, 'hero.webp'), 'x');
    writeFileSync(path.join(postImgDir, 'hero-768w.webp'), 'x');
    writeFileSync(path.join(postImgDir, 'second.svg'), 'x');

    const manifest = {
      '/images/blog/demo/hero.webp': {
        width: 1280,
        height: 720,
        srcset: [
          { url: '/images/blog/demo/hero-768w.webp', width: 768, height: 432 },
          { url: '/images/blog/demo/hero.webp', width: 1280, height: 720 },
        ],
      },
    };

    const tree = makeTree([
      { src: '/images/blog/demo/hero.webp', alt: 'hero' },
      { src: '/images/blog/demo/second.svg', alt: 'second' },
    ]);

    const transform = rehypeImageAttrs({ manifest, publicDir: tempPublic });
    transform(tree);

    const [first, second] = tree.children;
    assert.equal(first.properties.width, 1280);
    assert.equal(first.properties.height, 720);
    assert.equal(
      first.properties.srcset,
      '/images/blog/demo/hero-768w.webp 768w, /images/blog/demo/hero.webp 1280w',
    );
    assert.equal(first.properties.sizes, '(max-width: 760px) 100vw, 760px');
    assert.equal(first.properties.loading, 'eager');
    assert.equal(first.properties.fetchpriority, 'high');
    assert.equal(first.properties.decoding, 'async');

    assert.equal(second.properties.width, undefined);
    assert.equal(second.properties.loading, 'lazy');
    assert.equal(second.properties.fetchpriority, undefined);
    assert.equal(second.properties.decoding, 'async');
  } finally {
    rmSync(tempPublic, { recursive: true, force: true });
  }
});

test('rehypeImageAttrs：保留作者显式指定的属性，不覆盖', () => {
  const tree = makeTree([
    { src: 'https://example.com/a.webp', width: 400, height: 300, loading: 'lazy' },
  ]);

  const transform = rehypeImageAttrs({ manifest: {}, publicDir: null });
  transform(tree);

  const [img] = tree.children;
  assert.equal(img.properties.width, 400);
  assert.equal(img.properties.height, 300);
  assert.equal(img.properties.loading, 'lazy');
  assert.equal(img.properties.fetchpriority, undefined);
});

test('rehypeImageAttrs：本地 /images/ 文件缺失时抛错阻断构建', () => {
  const tempPublic = mkdtempSync(path.join(tmpdir(), 'blog-img-missing-'));
  try {
    const tree = makeTree([{ src: '/images/blog/demo/not-found.webp' }]);
    const transform = rehypeImageAttrs({ manifest: {}, publicDir: tempPublic });
    assert.throws(() => transform(tree), /does not exist in public\//);
  } finally {
    rmSync(tempPublic, { recursive: true, force: true });
  }
});

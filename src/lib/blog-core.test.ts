import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  assertNoReservedBlogPostIds,
  createBlogPage,
  filterBlogPostsByTagSlug,
  getBlogPageHref,
  getBlogTagHref,
  getBlogTags,
  groupBlogPostsByArchive,
  slugifyBlogTag,
  sortBlogPostsByPubDateDesc,
  type BlogPostLike,
} from './blog-core.ts';
import { formatBlogDateTime } from './date.ts';

function makePost(
  id: string,
  pubDate: string,
  tags: string[] = [],
  draft = false,
): BlogPostLike {
  return {
    id,
    data: {
      title: `Post ${id}`,
      description: `Description for ${id}`,
      pubDate: new Date(pubDate),
      tags,
      draft,
    },
  };
}

test('slugifyBlogTag：规范化大小写、空白、NFKC 与多语言字符', () => {
  assert.equal(slugifyBlogTag('  3D Generation '), '3d-generation');
  assert.equal(slugifyBlogTag('Paper_Reading / AI'), 'paper-reading-ai');
  assert.equal(slugifyBlogTag('ＡＩ-Agent'), 'ai-agent');
  assert.equal(slugifyBlogTag('强化学习 RL'), '强化学习-rl');
  assert.throws(() => slugifyBlogTag('   ---   '), /cannot be converted to a URL slug/);
});

test('getBlogTags：统计去重、按中文区域排序、检测同 slug 异名冲突', () => {
  const posts = [
    makePost('p1', '2026-07-01T10:00:00+08:00', ['AI', '3D Generation', 'AI']),
    makePost('p2', '2026-07-02T10:00:00+08:00', ['AI', 'Reinforcement Learning']),
  ];

  const tags = getBlogTags(posts);
  assert.equal(tags.length, 3);
  const aiTag = tags.find((t) => t.slug === 'ai');
  assert.deepEqual(aiTag, {
    name: 'AI',
    slug: 'ai',
    count: 2,
    href: '/blog/tag/ai/',
  });

  const conflictingPosts = [
    makePost('c1', '2026-07-01T10:00:00+08:00', ['3D-Generation']),
    makePost('c2', '2026-07-02T10:00:00+08:00', ['3d generation']),
  ];
  assert.throws(
    () => getBlogTags(conflictingPosts),
    /resolve to the same slug "3d-generation"/,
  );
});

test('assertNoReservedBlogPostIds：拦截 page 与 tag 保留路由首段', () => {
  assert.doesNotThrow(() =>
    assertNoReservedBlogPostIds([
      makePost('hello', '2026-07-01T00:00:00+08:00'),
      makePost('page-reading', '2026-07-01T00:00:00+08:00'),
    ]),
  );

  assert.throws(
    () => assertNoReservedBlogPostIds([makePost('page', '2026-07-01T00:00:00+08:00')]),
    /conflicts with reserved blog route "page"/,
  );
  assert.throws(
    () => assertNoReservedBlogPostIds([makePost('tag/sub', '2026-07-01T00:00:00+08:00')]),
    /conflicts with reserved blog route "tag"/,
  );
});

test('sortBlogPostsByPubDateDesc & groupBlogPostsByArchive：按上海时区归档分组', () => {
  // 2026-06-30T20:30:00Z 对应北京时间 2026-07-01 04:30
  const postJuly = makePost('july-post', '2026-06-30T20:30:00Z', ['AI']);
  const postJune = makePost('june-post', '2026-06-30T10:00:00Z', ['AI']);
  const post2025 = makePost('old-post', '2025-12-15T08:00:00+08:00', ['Notes']);

  const sorted = sortBlogPostsByPubDateDesc([postJune, post2025, postJuly]);
  assert.deepEqual(
    sorted.map((p) => p.id),
    ['july-post', 'june-post', 'old-post'],
  );

  assert.equal(formatBlogDateTime(postJuly.data.pubDate), '2026/07/01 04:30');

  const groups = groupBlogPostsByArchive(sorted);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].year, '2026');
  assert.deepEqual(
    groups[0].months.map((m) => m.label),
    ['2026.07', '2026.06'],
  );
  assert.equal(groups[1].year, '2025');
  assert.equal(groups[1].months[0].label, '2025.12');
});

test('filterBlogPostsByTagSlug & createBlogPage：筛选与分页链接构造', () => {
  const posts = Array.from({ length: 12 }, (_, i) =>
    makePost(`post-${i + 1}`, `2026-07-${String(12 - i).padStart(2, '0')}T10:00:00+08:00`, [
      i % 2 === 0 ? 'Paper Reading' : 'Systems',
    ]),
  );

  const paperPosts = filterBlogPostsByTagSlug(posts, 'paper-reading');
  assert.equal(paperPosts.length, 6);
  assert.equal(getBlogTagHref('Paper Reading'), '/blog/tag/paper-reading/');

  const page1 = createBlogPage(posts, 1, getBlogPageHref, 10);
  assert.equal(page1.data.length, 10);
  assert.equal(page1.lastPage, 2);
  assert.equal(page1.url.current, '/blog/');
  assert.equal(page1.url.prev, undefined);
  assert.equal(page1.url.next, '/blog/page/2/');

  const page2 = createBlogPage(posts, 2, getBlogPageHref, 10);
  assert.equal(page2.data.length, 2);
  assert.equal(page2.url.current, '/blog/page/2/');
  assert.equal(page2.url.prev, '/blog/');
  assert.equal(page2.url.next, undefined);

  assert.throws(() => createBlogPage(posts, 3, getBlogPageHref, 10), /outside the valid range/);
});

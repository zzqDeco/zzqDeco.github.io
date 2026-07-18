import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  createPaginationWindow,
  PAGINATION_FULL_WINDOW_LIMIT,
  type PaginationItem,
} from './pagination.ts';

const hrefOf = (page: number) => (page <= 1 ? '/blog/' : `/blog/page/${page}/`);

/** 渲染成紧凑形态便于断言：页码为数字，省略号为 …，当前页带 *。 */
function shape(items: PaginationItem[]): string {
  return items
    .map((item) => (item.type === 'gap' ? '…' : item.current ? `${item.number}*` : `${item.number}`))
    .join(' ');
}

function windowShape(currentPage: number, lastPage: number, siblingCount?: number): string {
  return shape(createPaginationWindow(currentPage, lastPage, hrefOf, siblingCount));
}

test('总页数 1-7 全量渲染，无省略号', () => {
  assert.equal(PAGINATION_FULL_WINDOW_LIMIT, 7);
  assert.equal(windowShape(1, 1), '1*');
  assert.equal(windowShape(1, 2), '1* 2');
  assert.equal(windowShape(2, 3), '1 2* 3');
  assert.equal(windowShape(3, 5), '1 2 3* 4 5');
  assert.equal(windowShape(1, 7), '1* 2 3 4 5 6 7');
  assert.equal(windowShape(4, 7), '1 2 3 4* 5 6 7');
  assert.equal(windowShape(7, 7), '1 2 3 4 5 6 7*');
});

test('8 页：恰好进入窗口化的临界页数', () => {
  assert.equal(windowShape(1, 8), '1* 2 3 … 8');
  assert.equal(windowShape(2, 8), '1 2* 3 4 … 8');
  assert.equal(windowShape(3, 8), '1 2 3* 4 5 … 8');
  // 只藏住一页时不放省略号，直接补出该页
  assert.equal(windowShape(4, 8), '1 2 3 4* 5 6 7 8');
  assert.equal(windowShape(5, 8), '1 2 3 4 5* 6 7 8');
  assert.equal(windowShape(6, 8), '1 … 4 5 6* 7 8');
  assert.equal(windowShape(7, 8), '1 … 5 6 7* 8');
  assert.equal(windowShape(8, 8), '1 … 6 7 8*');
});

test('第一页附近（12 页）：仅右侧省略', () => {
  assert.equal(windowShape(1, 12), '1* 2 3 … 12');
  assert.equal(windowShape(2, 12), '1 2* 3 4 … 12');
  assert.equal(windowShape(3, 12), '1 2 3* 4 5 … 12');
  assert.equal(windowShape(4, 12), '1 2 3 4* 5 6 … 12');
});

test('中间页（12 页）：双侧省略', () => {
  assert.equal(windowShape(5, 12), '1 2 3 4 5* 6 7 … 12');
  assert.equal(windowShape(6, 12), '1 … 4 5 6* 7 8 … 12');
  assert.equal(windowShape(7, 12), '1 … 5 6 7* 8 9 … 12');
  // 与末页只差一页时补出该页而非省略
  assert.equal(windowShape(8, 12), '1 … 6 7 8* 9 10 11 12');
});

test('末页附近（12 页）：仅左侧省略', () => {
  assert.equal(windowShape(9, 12), '1 … 7 8 9* 10 11 12');
  assert.equal(windowShape(10, 12), '1 … 8 9 10* 11 12');
  assert.equal(windowShape(11, 12), '1 … 9 10 11* 12');
  assert.equal(windowShape(12, 12), '1 … 10 11 12*');
});

test('视图项字段：href 正确、current 唯一、gap 带稳定 key', () => {
  const items = createPaginationWindow(6, 12, hrefOf);
  const current = items.filter((item) => item.type === 'page' && item.current);
  assert.equal(current.length, 1);
  assert.deepEqual(current[0], { type: 'page', number: 6, href: '/blog/page/6/', current: true });

  const page4 = items.find((item) => item.type === 'page' && item.number === 4);
  assert.deepEqual(page4, { type: 'page', number: 4, href: '/blog/page/4/', current: false });

  const gaps = items.filter((item) => item.type === 'gap');
  assert.deepEqual(
    gaps.map((gap) => gap.key),
    ['gap-after-1', 'gap-after-8'],
  );

  // 第 1 页的 href 走 getHref(1) 的原样输出
  const first = items[0];
  assert.deepEqual(first, { type: 'page', number: 1, href: '/blog/', current: false });
});

test('不变量穷举：1-30 页 × 所有当前页，无重复首末、无连续省略号', () => {
  for (let lastPage = 1; lastPage <= 30; lastPage += 1) {
    for (let currentPage = 1; currentPage <= lastPage; currentPage += 1) {
      const items = createPaginationWindow(currentPage, lastPage, hrefOf);
      const context = `lastPage=${lastPage}, currentPage=${currentPage}`;

      // 首项是第 1 页、末项是最后一页
      assert.deepEqual(items[0], { type: 'page', number: 1, href: hrefOf(1), current: currentPage === 1 }, context);
      const last = items[items.length - 1];
      assert.equal(last.type, 'page', context);
      assert.equal(last.type === 'page' && last.number, lastPage, context);

      const pageNumbers: number[] = [];
      const gapKeys = new Set<string>();
      let previousWasGap = false;
      for (const item of items) {
        if (item.type === 'gap') {
          assert.equal(previousWasGap, false, `连续省略号：${context}`);
          assert.equal(gapKeys.has(item.key), false, `重复 gap key：${context}`);
          gapKeys.add(item.key);
          previousWasGap = true;
        } else {
          pageNumbers.push(item.number);
          previousWasGap = false;
        }
      }

      // 页码严格递增且无重复（首末页不重复由此保证）
      for (let index = 1; index < pageNumbers.length; index += 1) {
        assert.ok(pageNumbers[index] > pageNumbers[index - 1], `页码未严格递增：${context}`);
      }

      // current 恰好标记一次
      assert.equal(
        items.filter((item) => item.type === 'page' && item.current).length,
        1,
        context,
      );
    }
  }
});

test('siblingCount 可调', () => {
  assert.equal(windowShape(6, 12, 1), '1 … 5 6* 7 … 12');
  assert.equal(windowShape(1, 12, 3), '1* 2 3 4 … 12');
  // siblingCount = 0 时仍保留首末页与当前页
  assert.equal(windowShape(6, 12, 0), '1 … 6* … 12');
});

test('非法输入抛错', () => {
  assert.throws(() => createPaginationWindow(0, 5, hrefOf), /outside the valid range/);
  assert.throws(() => createPaginationWindow(6, 5, hrefOf), /outside the valid range/);
  assert.throws(() => createPaginationWindow(1, 0, hrefOf), /positive integer/);
  assert.throws(() => createPaginationWindow(1.5, 5, hrefOf), /outside the valid range/);
  assert.throws(() => createPaginationWindow(1, 5, hrefOf, -1), /non-negative integer/);
});

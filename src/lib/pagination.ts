/**
 * 分页页码窗口化（issue #17）。
 * 纯函数模块，不依赖 astro:content，可用 `npm test`（node:test）直接验证。
 */

export type PaginationItem =
  | { type: 'page'; number: number; href: string; current: boolean }
  | { type: 'gap'; key: string };

export type PaginationHrefFactory = (page: number) => string;

/**
 * 总页数不超过该值时不做窗口化，全量渲染页码。
 * 取 7 的理由：默认 siblingCount = 2 时，双侧省略的窗口恰好渲染 7 个页码项
 * （首页 + 当前页 ±2 + 末页），因此总页数 ≤ 7 时全量渲染并不比窗口化多占空间。
 */
export const PAGINATION_FULL_WINDOW_LIMIT = 7;

/**
 * 生成窗口化的分页视图项：当前页 ±siblingCount + 首页/末页，
 * 两侧被跳过的页数超过 1 时用 gap 表示；只跳过 1 页时直接补出该页
 * （一个省略号和一个页码等宽，省略没有意义）。
 * 保证：首页/末页不重复、gap 不连续、页码严格递增。
 */
export function createPaginationWindow(
  currentPage: number,
  lastPage: number,
  getHref: PaginationHrefFactory,
  siblingCount = 2,
): PaginationItem[] {
  if (!Number.isInteger(lastPage) || lastPage < 1) {
    throw new Error(`lastPage must be a positive integer, got ${lastPage}.`);
  }
  if (!Number.isInteger(currentPage) || currentPage < 1 || currentPage > lastPage) {
    throw new Error(`currentPage ${currentPage} is outside the valid range 1-${lastPage}.`);
  }
  if (!Number.isInteger(siblingCount) || siblingCount < 0) {
    throw new Error(`siblingCount must be a non-negative integer, got ${siblingCount}.`);
  }

  const toPageItem = (number: number): PaginationItem => ({
    type: 'page',
    number,
    href: getHref(number),
    current: number === currentPage,
  });

  if (lastPage <= PAGINATION_FULL_WINDOW_LIMIT) {
    return Array.from({ length: lastPage }, (_, index) => toPageItem(index + 1));
  }

  // 候选页 = 首页 + 末页 + 当前页 ±siblingCount，Set 保证首末页与窗口重叠时不重复。
  const candidates = new Set<number>([1, lastPage]);
  for (let page = currentPage - siblingCount; page <= currentPage + siblingCount; page += 1) {
    if (page >= 1 && page <= lastPage) {
      candidates.add(page);
    }
  }

  const items: PaginationItem[] = [];
  let previous = 0;
  for (const number of [...candidates].sort((a, b) => a - b)) {
    const hidden = number - previous - 1;
    if (previous > 0 && hidden === 1) {
      items.push(toPageItem(previous + 1));
    } else if (hidden > 1) {
      items.push({ type: 'gap', key: `gap-after-${previous}` });
    }
    items.push(toPageItem(number));
    previous = number;
  }

  return items;
}

import { getBlogDateParts } from './date.ts';
import {
  createPaginationWindow,
  PAGINATION_FULL_WINDOW_LIMIT,
  type PaginationItem,
} from './pagination.ts';
import { isReservedBlogSegment } from './site.ts';

export { createPaginationWindow, PAGINATION_FULL_WINDOW_LIMIT } from './pagination.ts';
export type { PaginationItem } from './pagination.ts';

export const BLOG_PAGE_SIZE = 10;

export interface BlogPostDataLike {
  title: string;
  description: string;
  pubDate: Date;
  updatedDate?: Date;
  tags: string[];
  draft: boolean;
  image?: string;
}

export interface BlogPostLike {
  id: string;
  body?: string;
  data: BlogPostDataLike;
}

export interface BlogTag {
  name: string;
  slug: string;
  count: number;
  href: string;
}

export interface BlogMonthGroup<T extends BlogPostLike = BlogPostLike> {
  month: string;
  label: string;
  posts: T[];
}

export interface BlogYearGroup<T extends BlogPostLike = BlogPostLike> {
  year: string;
  months: BlogMonthGroup<T>[];
}

export interface BlogPage<T extends BlogPostLike = BlogPostLike> {
  data: T[];
  start: number;
  end: number;
  total: number;
  currentPage: number;
  size: number;
  lastPage: number;
  /** 窗口化分页视图项（issue #17），由 createPaginationWindow 生成。 */
  pagination: PaginationItem[];
  url: {
    current: string;
    prev?: string;
    next?: string;
    first?: string;
    last?: string;
  };
}

type PageHrefFactory = (page: number) => string;

export function assertNoReservedBlogPostIds<T extends Pick<BlogPostLike, 'id'>>(posts: T[]) {
  for (const post of posts) {
    const firstSegment = post.id.split('/')[0];
    if (isReservedBlogSegment(firstSegment)) {
      throw new Error(
        `Blog post id "${post.id}" conflicts with reserved blog route "${firstSegment}".`
      );
    }
  }
}

export function sortBlogPostsByPubDateDesc<T extends BlogPostLike>(posts: T[]): T[] {
  return [...posts].sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

export function slugifyBlogTag(tag: string) {
  const slug = tag
    .trim()
    .toLowerCase()
    .normalize('NFKC')
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '');

  if (!slug) {
    throw new Error(`Blog tag "${tag}" cannot be converted to a URL slug.`);
  }

  return slug;
}

export function getBlogTagHref(tag: string) {
  return getBlogTagPageHref(slugifyBlogTag(tag), 1);
}

export function getBlogTagPageHref(tagSlug: string, page: number) {
  return page <= 1 ? `/blog/tag/${tagSlug}/` : `/blog/tag/${tagSlug}/${page}/`;
}

export function getBlogPageHref(page: number) {
  return page <= 1 ? '/blog/' : `/blog/page/${page}/`;
}

export function getBlogTags<T extends BlogPostLike>(posts: T[]): BlogTag[] {
  const tagsBySlug = new Map<string, BlogTag>();

  for (const post of posts) {
    for (const tag of new Set(post.data.tags)) {
      const slug = slugifyBlogTag(tag);
      const existing = tagsBySlug.get(slug);

      if (existing && existing.name !== tag) {
        throw new Error(
          `Blog tags "${existing.name}" and "${tag}" resolve to the same slug "${slug}".`
        );
      }

      tagsBySlug.set(slug, {
        name: tag,
        slug,
        count: (existing?.count ?? 0) + 1,
        href: getBlogTagPageHref(slug, 1),
      });
    }
  }

  return Array.from(tagsBySlug.values()).sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'));
}

export function filterBlogPostsByTagSlug<T extends BlogPostLike>(posts: T[], tagSlug: string): T[] {
  return posts.filter((post) => post.data.tags.some((tag) => slugifyBlogTag(tag) === tagSlug));
}

export function createBlogPage<T extends BlogPostLike>(
  posts: T[],
  currentPage: number,
  getPageHref: PageHrefFactory,
  pageSize = BLOG_PAGE_SIZE
): BlogPage<T> {
  const lastPage = Math.max(1, Math.ceil(posts.length / pageSize));

  if (currentPage < 1 || currentPage > lastPage) {
    throw new Error(`Blog page ${currentPage} is outside the valid range 1-${lastPage}.`);
  }

  const start = (currentPage - 1) * pageSize;
  const data = posts.slice(start, start + pageSize);
  const end = data.length > 0 ? start + data.length - 1 : start;

  return {
    data,
    start,
    end,
    total: posts.length,
    currentPage,
    size: pageSize,
    lastPage,
    pagination: createPaginationWindow(currentPage, lastPage, getPageHref),
    url: {
      current: getPageHref(currentPage),
      prev: currentPage > 1 ? getPageHref(currentPage - 1) : undefined,
      next: currentPage < lastPage ? getPageHref(currentPage + 1) : undefined,
      first: currentPage > 1 ? getPageHref(1) : undefined,
      last: currentPage < lastPage ? getPageHref(lastPage) : undefined,
    },
  };
}

export function groupBlogPostsByArchive<T extends BlogPostLike>(posts: T[]): BlogYearGroup<T>[] {
  return posts.reduce((years, post) => {
    const { year, month } = getBlogDateParts(post.data.pubDate);
    const monthLabel = `${year}.${month}`;

    let yearGroup = years.find((group) => group.year === year);
    if (!yearGroup) {
      yearGroup = { year, months: [] };
      years.push(yearGroup);
    }

    let monthGroup = yearGroup.months.find((group) => group.month === month);
    if (!monthGroup) {
      monthGroup = { month, label: monthLabel, posts: [] };
      yearGroup.months.push(monthGroup);
    }

    monthGroup.posts.push(post);
    return years;
  }, [] as BlogYearGroup<T>[]);
}

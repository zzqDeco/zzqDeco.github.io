import { getCollection, type CollectionEntry } from 'astro:content';
import { getBlogDateParts } from './date';

export const BLOG_PAGE_SIZE = 10;

export type BlogPost = CollectionEntry<'blog'>;

export interface BlogTag {
  name: string;
  slug: string;
  count: number;
  href: string;
}

export interface BlogMonthGroup {
  month: string;
  label: string;
  posts: BlogPost[];
}

export interface BlogYearGroup {
  year: string;
  months: BlogMonthGroup[];
}

export interface BlogPage {
  data: BlogPost[];
  start: number;
  end: number;
  total: number;
  currentPage: number;
  size: number;
  lastPage: number;
  pages: Array<{
    number: number;
    href: string;
  }>;
  url: {
    current: string;
    prev?: string;
    next?: string;
    first?: string;
    last?: string;
  };
}

type PageHrefFactory = (page: number) => string;

export async function getVisibleBlogPosts() {
  const posts = await getCollection('blog', ({ data }) => (
    import.meta.env.PROD ? data.draft !== true : true
  ));

  assertNoReservedBlogPostIds(posts);

  return posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
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

export function getBlogTags(posts: BlogPost[]) {
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

export function filterBlogPostsByTagSlug(posts: BlogPost[], tagSlug: string) {
  return posts.filter((post) => post.data.tags.some((tag) => slugifyBlogTag(tag) === tagSlug));
}

export function createBlogPage(
  posts: BlogPost[],
  currentPage: number,
  getPageHref: PageHrefFactory,
  pageSize = BLOG_PAGE_SIZE
): BlogPage {
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
    pages: Array.from({ length: lastPage }, (_, index) => {
      const number = index + 1;
      return { number, href: getPageHref(number) };
    }),
    url: {
      current: getPageHref(currentPage),
      prev: currentPage > 1 ? getPageHref(currentPage - 1) : undefined,
      next: currentPage < lastPage ? getPageHref(currentPage + 1) : undefined,
      first: currentPage > 1 ? getPageHref(1) : undefined,
      last: currentPage < lastPage ? getPageHref(lastPage) : undefined,
    },
  };
}

export function groupBlogPostsByArchive(posts: BlogPost[]) {
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
  }, [] as BlogYearGroup[]);
}

function assertNoReservedBlogPostIds(posts: BlogPost[]) {
  for (const post of posts) {
    const firstSegment = post.id.split('/')[0];
    if (firstSegment === 'page' || firstSegment === 'tag') {
      throw new Error(`Blog post id "${post.id}" conflicts with reserved blog route "${firstSegment}".`);
    }
  }
}

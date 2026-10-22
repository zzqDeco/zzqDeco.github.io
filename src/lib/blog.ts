import { getCollection, type CollectionEntry } from 'astro:content';
import {
  assertNoReservedBlogPostIds,
  sortBlogPostsByPubDateDesc,
  type BlogMonthGroup as CoreBlogMonthGroup,
  type BlogPage as CoreBlogPage,
  type BlogYearGroup as CoreBlogYearGroup,
} from './blog-core.ts';

export {
  assertNoReservedBlogPostIds,
  BLOG_PAGE_SIZE,
  createBlogPage,
  createPaginationWindow,
  filterBlogPostsByTagSlug,
  getBlogPageHref,
  getBlogTagHref,
  getBlogTagPageHref,
  getBlogTags,
  groupBlogPostsByArchive,
  PAGINATION_FULL_WINDOW_LIMIT,
  slugifyBlogTag,
  sortBlogPostsByPubDateDesc,
} from './blog-core.ts';
export type { BlogTag, PaginationItem } from './blog-core.ts';

export type BlogPost = CollectionEntry<'blog'>;
export type BlogMonthGroup = CoreBlogMonthGroup<BlogPost>;
export type BlogYearGroup = CoreBlogYearGroup<BlogPost>;
export type BlogPage = CoreBlogPage<BlogPost>;

export async function getVisibleBlogPosts(options?: { excludeDrafts?: boolean }) {
  const excludeDrafts = options?.excludeDrafts ?? import.meta.env.PROD;
  const posts = await getCollection('blog', ({ data }) => (
    excludeDrafts ? data.draft !== true : true
  ));

  assertNoReservedBlogPostIds(posts);

  return sortBlogPostsByPubDateDesc(posts);
}

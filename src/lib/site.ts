export const SITE_NAME = 'zzqDeco Blog';
export const SITE_DESCRIPTION = 'zzqDeco 的个人博客';
export const DEFAULT_SHARE_IMAGE = '/images/og-default.png';

export const RESERVED_BLOG_SEGMENTS = ['page', 'tag'] as const;

export function isReservedBlogSegment(segment: string): boolean {
  return (RESERVED_BLOG_SEGMENTS as readonly string[]).includes(segment);
}

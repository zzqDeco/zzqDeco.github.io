import rss from '@astrojs/rss';
import { getVisibleBlogPosts } from '../lib/blog';
import { SITE_DESCRIPTION, SITE_NAME } from '../lib/site';

export async function GET(context) {
  const posts = await getVisibleBlogPosts({ excludeDrafts: true });

  return rss({
    title: SITE_NAME,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.pubDate,
      link: `/blog/${post.id}/`,
    })),
    customData: '<language>zh-CN</language>',
  });
}

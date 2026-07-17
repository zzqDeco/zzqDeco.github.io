import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import rehypeKatex from 'rehype-katex';
import rehypeImageAttrs from './src/lib/rehype-image-attrs.mjs';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

export default defineConfig({
  site: 'https://zzqDeco.github.io',
  // 仅加载预取脚本，默认只处理带 data-astro-prefetch 的链接（hover 策略）；
  // 不开 prefetchAll / defaultStrategy: 'viewport'，避免全站预取浪费（文章页 gzip ~60KB）。
  prefetch: true,
  integrations: [sitemap()],
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark-dimmed',
      },
      defaultColor: false,
      wrap: true,
    },
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeKatex, rehypeImageAttrs],
  },
});

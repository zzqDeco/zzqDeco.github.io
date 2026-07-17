import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import rehypeKatex from 'rehype-katex';
import rehypeImageAttrs from './src/lib/rehype-image-attrs.mjs';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

export default defineConfig({
  site: 'https://zzqDeco.github.io',
  integrations: [sitemap()],
  markdown: {
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark-dimmed',
      },
      wrap: true,
    },
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeKatex, rehypeImageAttrs],
  },
});

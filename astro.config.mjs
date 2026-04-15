import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://zzqDeco.github.io',
  integrations: [sitemap()],
});

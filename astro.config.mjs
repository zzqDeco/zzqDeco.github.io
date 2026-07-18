import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import rehypeKatex from 'rehype-katex';
import rehypeImageAttrs from './src/lib/rehype-image-attrs.mjs';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

// 自定义 shiki 双主题（设计迭代：暖调石墨 × Gruvbox 系暖调多彩，方案 C）。
// 配色以 token 角色定义一次，亮/暗各取一套色值；bg 与 --bg-inset 同值。
// fontStyle 只支持 bold/italic（shiki 规范），对应原型中的 600 粗与斜体；
// 无样式角色必须显式写 fontStyle: ''（解析为 FontStyle.None），
// 否则更具体的 scope 命中无样式规则时会继承前缀规则（keyword/constant 等）的 bold。
const syntaxRoles = {
  keyword: { light: '#9d0006', dark: '#ea6962', fontStyle: 'bold' },
  type: { light: '#b57614', dark: '#d8a657', fontStyle: 'bold' },
  function: { light: '#076678', dark: '#83a598', fontStyle: 'bold' },
  string: { light: '#79740e', dark: '#a9b665', fontStyle: '' },
  number: { light: '#8f3f71', dark: '#d3869b', fontStyle: '' },
  property: { light: '#427b58', dark: '#89b482', fontStyle: '' },
  comment: { light: '#928374', dark: '#7c6f64', fontStyle: 'italic' },
  operator: { light: '#44403c', dark: '#c9c3bd', fontStyle: '' },
};

// TextMate scope → 角色。注意前缀匹配规则：同栈命中多条时更长的 scope 选择器优先，
// 因此 keyword.operator / punctuation.definition.string / support.type.property-name /
// constant.numeric / constant.language 等更具体的条目会盖过其前缀组（keyword /
// punctuation / support.type / constant）。
const syntaxScopes = {
  comment: ['comment', 'punctuation.definition.comment'],
  keyword: ['keyword', 'storage'],
  operator: ['keyword.operator', 'punctuation'],
  type: [
    'entity.name.type',
    'support.type',
    'entity.name.class',
    'support.class',
    'entity.other.inherited-class',
    'constant',
    'markup.heading',
    'entity.name.section',
  ],
  function: [
    'entity.name.function',
    'support.function',
    'variable.function',
    // python 方法/函数调用名（该 grammar 不给 entity.name.function）
    'meta.function-call.generic',
    'markup.underline.link',
    'string.other.link',
  ],
  string: [
    // 不用裸 'string' 前缀：bash 未加引号的参数是 string.unquoted.*，按原型保持基色
    'string.quoted',
    'string.template',
    'string.interpolated',
    'string.regexp',
    'constant.other.symbol',
    'punctuation.definition.string',
    'markup.inline.raw.string',
  ],
  number: [
    'constant.numeric',
    'constant.language',
    'constant.character',
    'keyword.other.unit',
    'support.constant',
  ],
  property: [
    'variable.other.property',
    'variable.other.object.property',
    'support.type.property-name',
    'entity.name.tag',
    'entity.other.attribute-name',
    'variable.parameter',
    'meta.object-literal.key',
  ],
};

// 需要强制回退为基础前景的 scope（比角色规则更长，命中后胜出）：
// bash 命令名（entity.name.function.call/command.shell 会误入 function 角色）、
// bash 选项与未加引号参数，均按原型保持基色。
const syntaxPlainScopes = [
  'entity.name.function.call',
  'entity.name.command',
  'constant.other.option',
  'string.unquoted',
];

function buildWarmGraphiteTheme(mode) {
  const isLight = mode === 'light';
  const fg = isLight ? '#1c1917' : '#e8e4e0';
  const bg = isLight ? '#f0ede8' : '#181614';
  return {
    name: `warm-graphite-${mode}`,
    type: mode,
    fg,
    bg,
    colors: {
      'editor.background': bg,
      'editor.foreground': fg,
    },
    settings: [
      { scope: syntaxPlainScopes, settings: { foreground: fg, fontStyle: '' } },
      ...Object.entries(syntaxScopes).map(([role, scopes]) => ({
        scope: scopes,
        settings: {
          foreground: syntaxRoles[role][mode],
          fontStyle: syntaxRoles[role].fontStyle,
        },
      })),
      { scope: 'markup.bold', settings: { fontStyle: 'bold' } },
      { scope: 'markup.italic', settings: { fontStyle: 'italic' } },
    ],
  };
}

export default defineConfig({
  site: 'https://zzqDeco.github.io',
  // 仅加载预取脚本，默认只处理带 data-astro-prefetch 的链接（hover 策略）；
  // 不开 prefetchAll / defaultStrategy: 'viewport'，避免全站预取浪费（文章页 gzip ~60KB）。
  prefetch: true,
  integrations: [sitemap()],
  markdown: {
    shikiConfig: {
      themes: {
        light: buildWarmGraphiteTheme('light'),
        dark: buildWarmGraphiteTheme('dark'),
      },
      defaultColor: false,
      wrap: true,
    },
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeKatex, rehypeImageAttrs],
  },
});

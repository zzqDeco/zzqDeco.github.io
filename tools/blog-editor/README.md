# Local Blog Editor

A local-only Typora-like Markdown editor for `src/content/blog/*.md`.

It is intentionally not part of the Astro site build and is not deployed to GitHub Pages. It exists to make writing posts safer and faster than editing frontmatter by hand.

## Start

```bash
npm run editor
```

Open:

```text
http://127.0.0.1:4322
```

The server binds to `127.0.0.1` only. Set `BLOG_EDITOR_PORT` if port `4322` is already in use.

```bash
BLOG_EDITOR_PORT=4323 npm run editor
```

## Writing experience

The editor uses Milkdown Crepe as a Markdown-first WYSIWYG editor. It does not show a separate preview pane: the writing surface itself renders headings, lists, blockquotes, tables, code blocks, images, and math while still saving standard Markdown.

Markdown input rules are supported in the editor:

- `# Heading`, `## Heading`, `### Heading`
- `- item`, `1. item`, `- [ ] task`
- `> quote`
- fenced code blocks
- GFM tables
- `**bold**`, `*italic*`, inline code, links, and images
- `$inline math$` and `$$block math$$`

Available quick actions:

- `Zen`: enter page-level Zen mode
- `Save`: save the current post
- `H1`, `H2`, `H3`
- `Bold`, `Italic`
- `Link`: insert a Markdown link
- `Image URL`: insert an image from a URL
- `Upload Image`: upload a local image and insert it into the post
- `Bullet`, `Ordered`, `Task`, `Quote`, `Code`, `Table`, `Math`
- `Undo`, `Redo`

Keyboard shortcuts use `Mod` for `Cmd` on macOS and `Ctrl` on Windows/Linux:

| Action | Shortcut |
| --- | --- |
| Save | `Mod+S` |
| Enter Zen / Focus mode | `Mod+Shift+Enter` |
| Exit Zen mode | `Esc` |
| H1 / H2 / H3 | `Alt+1`, `Alt+2`, `Alt+3` |
| Bold / Italic | `Mod+B`, `Mod+I` |
| Link | `Mod+K` |
| Image URL | `Mod+Alt+I` |
| Upload Image | `Mod+Alt+U` |
| Bullet list | `Mod+Shift+8` |
| Ordered list | `Mod+Shift+7` |
| Task list | `Mod+Alt+X` |
| Quote | `Mod+Alt+Q` |
| Code block | `Mod+Alt+C` |
| Table | `Mod+Alt+L` |
| Math block | `Mod+Alt+M` |
| Undo | `Mod+Z` |
| Redo | `Mod+Shift+Z`, or `Ctrl+Y` on Windows/Linux |

Shortcut scope:

- `Save`, `Enter Zen`, and `Exit Zen` are global.
- Formatting shortcuts only run inside the Markdown editor surface.
- When a frontmatter field is focused, formatting shortcuts are ignored so typing metadata is safe.
- When the Link or Image URL dialog is open, `Esc` closes the dialog and other formatting shortcuts are ignored.

## Supported frontmatter

The editor follows the current Astro content collection schema:

```yaml
title: "Post title"
description: "Short summary"
pubDate: 2026-04-15
updatedDate: 2026-04-15
tags:
  - Astro
draft: false
```

Fields:

- `title`: required string
- `description`: required string
- `pubDate`: required date, `YYYY-MM-DD`
- `updatedDate`: optional date, `YYYY-MM-DD`
- `tags`: comma-separated input in the UI, saved as a YAML array
- `draft`: boolean; production pages hide `draft: true`

## Images

Image URL inserts a normal Markdown image reference.

Upload Image saves files under:

```text
public/images/blog/<slug>/
```

The inserted Markdown path uses the public Astro URL:

```md
![alt](/images/blog/<slug>/<filename>)
```

Rules:

- Existing posts use the current post slug.
- New posts must have a valid slug before uploading images.
- Supported types: `png`, `jpg/jpeg`, `webp`, `gif`, `svg`.
- Maximum file size: 5MB.
- Uploaded image files are repo files. Commit them together with the article.

## Math

The editor supports inline and block math syntax:

```md
Inline math: $E = mc^2$

$$
\int_0^1 x^2 dx = \frac{1}{3}
$$
```

The Astro site is configured with `remark-math` and `rehype-katex`, so the same Markdown renders as KaTeX in published posts.

## Behavior

- New posts are saved to `src/content/blog/<slug>.md`.
- Slugs must use lowercase letters, numbers, and hyphens.
- Existing post slugs are locked in the UI to avoid accidental path changes.
- Save rewrites frontmatter in a stable field order.
- Delete is a soft delete: files move to `.trash/blog/<slug>-<timestamp>.md`.
- Zen mode hides the file explorer and frontmatter panel, but keeps save/status controls available.

## API

The editor uses these local endpoints:

- `GET /api/posts`
- `GET /api/posts/:slug`
- `POST /api/posts`
- `PUT /api/posts/:slug`
- `DELETE /api/posts/:slug`
- `POST /api/assets?slug=<slug>&filename=<filename>`

All API errors return JSON with either `{ "error": "..." }` or `{ "errors": ["..."] }`.

## Scope

This tool does not add a database, authentication, CMS, MDX support, or new frontmatter fields. If the blog schema changes, update both `src/content.config.ts` and this editor together.

# Local Blog Editor

A local-only Markdown editor for `src/content/blog/*.md`.

It is intentionally not part of the Astro site build and is not deployed to GitHub Pages. It exists to make writing posts safer than editing frontmatter by hand.

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

## Behavior

- New posts are saved to `src/content/blog/<slug>.md`.
- Slugs must use lowercase letters, numbers, and hyphens.
- Existing post slugs are locked in the UI to avoid accidental path changes.
- Save rewrites frontmatter in a stable field order.
- Delete is a soft delete: files move to `.trash/blog/<slug>-<timestamp>.md`.
- Markdown preview is rendered locally by the editor server.

## API

The editor uses these local endpoints:

- `GET /api/posts`
- `GET /api/posts/:slug`
- `POST /api/posts`
- `PUT /api/posts/:slug`
- `DELETE /api/posts/:slug`
- `POST /api/preview`

All API errors return JSON with either `{ "error": "..." }` or `{ "errors": ["..."] }`.

## Scope

This tool does not add a database, authentication, CMS, WYSIWYG editor, MDX support, or new frontmatter fields. If the blog schema changes, update both `src/content.config.ts` and this editor together.

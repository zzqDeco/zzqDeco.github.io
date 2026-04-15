import { createServer } from 'node:http';
import { mkdir, readFile, readdir, rename, stat, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import matter from 'gray-matter';
import { build } from 'esbuild';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = process.cwd();
const blogDir = path.join(repoRoot, 'src/content/blog');
const trashDir = path.join(repoRoot, '.trash/blog');
const assetDir = path.join(repoRoot, 'public/images/blog');
const publicDir = path.join(__dirname, 'public');
const bundleDir = path.join(repoRoot, 'node_modules/.cache/blog-editor');
const editorBundlePath = path.join(bundleDir, 'milkdown-editor.js');
const editorEntryPath = path.join(__dirname, 'client/milkdown-entry.js');
const host = '127.0.0.1';
const port = Number(process.env.BLOG_EDITOR_PORT ?? 4322);
const slugPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const maxAssetBytes = 5 * 1024 * 1024;
const assetTypes = new Map([
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.webp', 'image/webp'],
  ['.gif', 'image/gif'],
  ['.svg', 'image/svg+xml'],
]);

const json = (response, statusCode, payload) => {
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  response.end(JSON.stringify(payload));
};

const text = (response, statusCode, payload) => {
  response.writeHead(statusCode, {
    'Content-Type': 'text/plain; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  response.end(payload);
};

const safeSlug = (slug) => {
  if (!slugPattern.test(slug)) {
    throw new Error('Slug must use lowercase letters, numbers, and hyphens only.');
  }
  return slug;
};

const postPath = (slug) => path.join(blogDir, `${safeSlug(slug)}.md`);

const toDateInput = (value) => {
  if (!value) return '';
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  return String(value).slice(0, 10);
};

const normalizeData = (data = {}) => ({
  title: String(data.title ?? ''),
  description: String(data.description ?? ''),
  pubDate: toDateInput(data.pubDate),
  updatedDate: toDateInput(data.updatedDate),
  tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
  draft: Boolean(data.draft),
});

const readPost = async (slug) => {
  const file = await readFile(postPath(slug), 'utf8');
  const parsed = matter(file);
  return {
    slug,
    data: normalizeData(parsed.data),
    content: parsed.content.replace(/^\n+/, ''),
  };
};

const listPosts = async () => {
  const files = await readdir(blogDir);
  const posts = await Promise.all(
    files
      .filter((file) => file.endsWith('.md'))
      .sort()
      .map(async (file) => {
        const slug = file.slice(0, -3);
        const post = await readPost(slug);
        return {
          slug,
          title: post.data.title,
          description: post.data.description,
          pubDate: post.data.pubDate,
          draft: post.data.draft,
          tags: post.data.tags,
        };
      }),
  );
  return posts.sort((a, b) => String(b.pubDate).localeCompare(String(a.pubDate)));
};

const quoteYaml = (value) => JSON.stringify(String(value ?? ''));

const serializePost = ({ data, content }) => {
  const clean = normalizeData(data);
  const tags = clean.tags.map((tag) => `  - ${quoteYaml(tag)}`).join('\n');
  const lines = [
    '---',
    `title: ${quoteYaml(clean.title)}`,
    `description: ${quoteYaml(clean.description)}`,
    `pubDate: ${clean.pubDate}`,
  ];

  if (clean.updatedDate) {
    lines.push(`updatedDate: ${clean.updatedDate}`);
  }

  lines.push('tags:');
  lines.push(tags || '  []');
  lines.push(`draft: ${clean.draft ? 'true' : 'false'}`);
  lines.push('---', '');
  lines.push(String(content ?? '').replace(/\s+$/, ''));
  lines.push('');

  return lines.join('\n');
};

const validatePayload = ({ data, content }) => {
  const clean = normalizeData(data);
  const errors = [];

  if (!clean.title.trim()) errors.push('title is required');
  if (!clean.description.trim()) errors.push('description is required');
  if (!/^\d{4}-\d{2}-\d{2}$/.test(clean.pubDate)) errors.push('pubDate must be YYYY-MM-DD');
  if (clean.updatedDate && !/^\d{4}-\d{2}-\d{2}$/.test(clean.updatedDate)) {
    errors.push('updatedDate must be YYYY-MM-DD');
  }
  if (typeof content !== 'string') errors.push('content must be a string');

  return errors;
};

const readJsonBody = async (request) => {
  let raw = '';
  for await (const chunk of request) raw += chunk;
  if (!raw.trim()) return {};
  return JSON.parse(raw);
};

const readBinaryBody = async (request) => {
  const chunks = [];
  let size = 0;

  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > maxAssetBytes) {
      throw new Error('Image must be 5MB or smaller.');
    }
    chunks.push(buffer);
  }

  return Buffer.concat(chunks);
};

const sendPost = async (response, slug) => {
  try {
    json(response, 200, await readPost(slug));
  } catch (error) {
    json(response, 404, { error: error.message });
  }
};

const createPost = async (request, response) => {
  try {
    const payload = await readJsonBody(request);
    const slug = safeSlug(String(payload.slug ?? ''));
    const target = postPath(slug);

    try {
      await stat(target);
      json(response, 409, { error: 'Post already exists.' });
      return;
    } catch {
      // File does not exist. Continue.
    }

    const errors = validatePayload(payload);
    if (errors.length > 0) {
      json(response, 400, { errors });
      return;
    }

    await writeFile(target, serializePost(payload), 'utf8');
    json(response, 201, await readPost(slug));
  } catch (error) {
    json(response, 400, { error: error.message });
  }
};

const updatePost = async (request, response, slug) => {
  try {
    const payload = await readJsonBody(request);
    const errors = validatePayload(payload);
    if (errors.length > 0) {
      json(response, 400, { errors });
      return;
    }

    await writeFile(postPath(slug), serializePost(payload), 'utf8');
    json(response, 200, await readPost(slug));
  } catch (error) {
    json(response, 400, { error: error.message });
  }
};

const deletePost = async (response, slug) => {
  try {
    await mkdir(trashDir, { recursive: true });
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const trashPath = path.join(trashDir, `${safeSlug(slug)}-${stamp}.md`);
    await rename(postPath(slug), trashPath);
    json(response, 200, { deleted: true, trashPath: path.relative(repoRoot, trashPath) });
  } catch (error) {
    json(response, 400, { error: error.message });
  }
};

const safeAssetName = (filename) => {
  const base = path.basename(String(filename || 'image'));
  const rawExt = path.extname(base).toLowerCase();
  const ext = rawExt === '.jpeg' ? '.jpg' : rawExt;

  if (!assetTypes.has(rawExt) && !assetTypes.has(ext)) {
    throw new Error('Image must be png, jpg, webp, gif, or svg.');
  }

  const stem = path
    .basename(base, rawExt)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60) || 'image';

  return {
    filename: `${Date.now()}-${stem}${ext}`,
    contentType: assetTypes.get(ext) ?? assetTypes.get(rawExt),
  };
};

const uploadAsset = async (request, response, url) => {
  try {
    const slug = safeSlug(String(url.searchParams.get('slug') ?? ''));
    const { filename, contentType } = safeAssetName(url.searchParams.get('filename'));
    const body = await readBinaryBody(request);

    if (body.length === 0) {
      json(response, 400, { error: 'Image body is empty.' });
      return;
    }

    const targetDir = path.join(assetDir, slug);
    const target = path.join(targetDir, filename);
    await mkdir(targetDir, { recursive: true });
    await writeFile(target, body);

    json(response, 201, {
      url: `/images/blog/${slug}/${filename}`,
      filename,
      contentType,
      size: body.length,
    });
  } catch (error) {
    json(response, 400, { error: error.message });
  }
};

const ensureEditorBundle = async () => {
  await mkdir(bundleDir, { recursive: true });
  await build({
    entryPoints: [editorEntryPath],
    bundle: true,
    outfile: editorBundlePath,
    format: 'iife',
    platform: 'browser',
    target: 'es2020',
    define: {
      __VUE_OPTIONS_API__: 'false',
      __VUE_PROD_DEVTOOLS__: 'false',
      __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: 'false',
    },
    assetNames: 'assets/[name]-[hash]',
    loader: {
      '.woff': 'file',
      '.woff2': 'file',
      '.ttf': 'file',
      '.otf': 'file',
    },
    logLevel: 'silent',
  });
};

const serveStatic = async (request, response, pathname) => {
  if (pathname === '/favicon.ico') {
    response.writeHead(204, { 'Cache-Control': 'no-store' });
    response.end();
    return;
  }

  const isEditorVendor = pathname.startsWith('/vendor/blog-editor/');
  const relative = isEditorVendor
    ? pathname.replace('/vendor/blog-editor/', '')
    : pathname === '/'
      ? 'index.html'
      : pathname.slice(1);
  const root = isEditorVendor ? bundleDir : publicDir;
  const target = path.normalize(path.join(root, relative));
  const publicRoot = `${root}${path.sep}`;

  if (target !== root && !target.startsWith(publicRoot)) {
    text(response, 403, 'Forbidden');
    return;
  }

  try {
    const fileStat = await stat(target);
    if (!fileStat.isFile()) {
      text(response, 404, 'Not found');
      return;
    }

    const ext = path.extname(target);
    const contentTypes = {
      '.html': 'text/html; charset=utf-8',
      '.css': 'text/css; charset=utf-8',
      '.js': 'text/javascript; charset=utf-8',
      '.svg': 'image/svg+xml',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.webp': 'image/webp',
      '.gif': 'image/gif',
      '.woff': 'font/woff',
      '.woff2': 'font/woff2',
      '.ttf': 'font/ttf',
      '.otf': 'font/otf',
    };
    response.writeHead(200, {
      'Content-Type': contentTypes[ext] ?? 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    const stream = createReadStream(target);
    stream.on('error', () => {
      if (!response.headersSent) {
        text(response, 404, 'Not found');
        return;
      }
      response.end();
    });
    stream.pipe(response);
  } catch {
    text(response, 404, 'Not found');
  }
};

const server = createServer(async (request, response) => {
  const url = new URL(request.url ?? '/', `http://${host}:${port}`);
  const pathname = decodeURIComponent(url.pathname);
  const segments = pathname.split('/').filter(Boolean);

  try {
    if (pathname === '/api/posts' && request.method === 'GET') {
      json(response, 200, { posts: await listPosts() });
      return;
    }

    if (pathname === '/api/posts' && request.method === 'POST') {
      await createPost(request, response);
      return;
    }

    if (segments[0] === 'api' && segments[1] === 'posts' && segments[2]) {
      const slug = safeSlug(segments[2]);
      if (request.method === 'GET') {
        await sendPost(response, slug);
        return;
      }
      if (request.method === 'PUT') {
        await updatePost(request, response, slug);
        return;
      }
      if (request.method === 'DELETE') {
        await deletePost(response, slug);
        return;
      }
    }

    if (pathname === '/api/assets' && request.method === 'POST') {
      await uploadAsset(request, response, url);
      return;
    }

    if (segments[0] === 'api') {
      json(response, 404, { error: 'Not found.' });
      return;
    }

    await serveStatic(request, response, pathname);
  } catch (error) {
    json(response, 500, { error: error.message });
  }
});

await ensureEditorBundle();

server.listen(port, host, () => {
  console.log(`Blog editor running at http://${host}:${port}`);
});

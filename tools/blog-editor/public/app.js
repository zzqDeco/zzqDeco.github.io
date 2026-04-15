const state = {
  posts: [],
  currentSlug: '',
  isNew: false,
  previewTimer: null,
};

const $ = (selector) => document.querySelector(selector);

const elements = {
  status: $('#status'),
  postList: $('#post-list'),
  newPost: $('#new-post'),
  savePost: $('#save-post'),
  deletePost: $('#delete-post'),
  refreshPreview: $('#refresh-preview'),
  form: $('#post-form'),
  slug: $('#slug'),
  title: $('#title'),
  description: $('#description'),
  pubDate: $('#pubDate'),
  updatedDate: $('#updatedDate'),
  tags: $('#tags'),
  draft: $('#draft'),
  content: $('#content'),
  preview: $('#preview'),
  currentTitle: $('#current-title'),
  currentPath: $('#current-path'),
  slugLabel: $('#slug-label'),
};

const setStatus = (message, tone = 'info') => {
  elements.status.textContent = message;
  elements.status.dataset.tone = tone;
};

const escapeHtml = (value) =>
  String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');

const request = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers ?? {}) },
    ...options,
  });

  const raw = await response.text();
  let payload = {};
  if (raw.trim()) {
    try {
      payload = JSON.parse(raw);
    } catch {
      payload = { error: raw };
    }
  }

  if (!response.ok) {
    throw new Error(payload.error || payload.errors?.join(', ') || 'Request failed');
  }

  return payload;
};

const today = () => new Date().toISOString().slice(0, 10);

const formPayload = () => ({
  slug: elements.slug.value.trim(),
  data: {
    title: elements.title.value.trim(),
    description: elements.description.value.trim(),
    pubDate: elements.pubDate.value,
    updatedDate: elements.updatedDate.value,
    tags: elements.tags.value
      .split(',')
      .map((tag) => tag.trim())
      .filter(Boolean),
    draft: elements.draft.checked,
  },
  content: elements.content.value,
});

const validateClient = (payload) => {
  const errors = [];
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(payload.slug)) {
    errors.push('slug must use lowercase letters, numbers, and hyphens only');
  }
  if (!payload.data.title) errors.push('title is required');
  if (!payload.data.description) errors.push('description is required');
  if (!payload.data.pubDate) errors.push('pubDate is required');
  return errors;
};

const updateChrome = () => {
  const slug = state.currentSlug || elements.slug.value.trim();
  const title = elements.title.value.trim() || (state.isNew ? 'New draft' : 'Select a post');
  elements.currentTitle.textContent = title;
  elements.currentPath.textContent = slug ? `src/content/blog/${slug}.md` : 'src/content/blog';
  elements.slugLabel.textContent = slug ? `${slug}.md` : 'unsaved.md';
  elements.deletePost.disabled = state.isNew || !state.currentSlug;
};

const fillForm = ({ slug, data, content }) => {
  state.currentSlug = slug;
  elements.slug.value = slug;
  elements.slug.disabled = !state.isNew;
  elements.title.value = data.title ?? '';
  elements.description.value = data.description ?? '';
  elements.pubDate.value = data.pubDate ?? today();
  elements.updatedDate.value = data.updatedDate ?? '';
  elements.tags.value = (data.tags ?? []).join(', ');
  elements.draft.checked = Boolean(data.draft);
  elements.content.value = content ?? '';
  updateChrome();
};

const renderList = () => {
  elements.postList.innerHTML = '';

  if (state.posts.length === 0) {
    elements.postList.innerHTML = '<p class="empty-state">No markdown files yet.</p>';
    return;
  }

  for (const post of state.posts) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `file-row${post.slug === state.currentSlug ? ' is-active' : ''}`;
    button.innerHTML = `
      <span class="file-main">
        <span class="file-ext">md</span>
        <span class="file-name">${escapeHtml(post.slug)}.md</span>
      </span>
      <span class="file-title">${escapeHtml(post.title || 'Untitled')}</span>
      <span class="file-meta">
        <time>${escapeHtml(post.pubDate || 'no date')}</time>
        ${post.draft ? '<b>draft</b>' : ''}
      </span>
    `;
    button.addEventListener('click', () => loadPost(post.slug));
    elements.postList.append(button);
  }
};

const refreshPosts = async () => {
  const payload = await request('/api/posts');
  state.posts = payload.posts ?? [];
  renderList();
};

const refreshPreview = async () => {
  const source = elements.content.value;
  if (!source.trim()) {
    elements.preview.innerHTML = '<p class="empty-preview">Nothing to preview yet.</p>';
    return;
  }

  const payload = await request('/api/preview', {
    method: 'POST',
    body: JSON.stringify({ content: source }),
  });
  elements.preview.innerHTML = payload.html || '<p class="empty-preview">Nothing to preview yet.</p>';
};

const loadPost = async (slug) => {
  state.isNew = false;
  setStatus(`loading ${slug}.md...`, 'loading');

  try {
    const post = await request(`/api/posts/${slug}`);
    fillForm(post);
    renderList();
    await refreshPreview();
    setStatus(`loaded ${slug}.md`, 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  }
};

const newPost = () => {
  state.isNew = true;
  state.currentSlug = '';
  fillForm({
    slug: '',
    data: {
      title: '',
      description: '',
      pubDate: today(),
      updatedDate: '',
      tags: [],
      draft: true,
    },
    content: 'Start writing here.\n',
  });
  renderList();
  elements.preview.innerHTML = '<p class="empty-preview">Write markdown to see the preview.</p>';
  setStatus('new draft', 'info');
  elements.slug.focus();
};

const savePost = async () => {
  const payload = formPayload();
  const errors = validateClient(payload);
  if (errors.length > 0) {
    setStatus(errors.join(', '), 'error');
    return;
  }

  const url = state.isNew ? '/api/posts' : `/api/posts/${state.currentSlug}`;
  const method = state.isNew ? 'POST' : 'PUT';
  setStatus('saving...', 'loading');

  try {
    const saved = await request(url, {
      method,
      body: JSON.stringify(payload),
    });
    state.isNew = false;
    fillForm(saved);
    await refreshPosts();
    await refreshPreview();
    setStatus(`saved ${new Date().toLocaleTimeString()}`, 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  }
};

const deletePost = async () => {
  if (state.isNew || !state.currentSlug) {
    setStatus('nothing to delete', 'info');
    return;
  }

  const confirmed = confirm(`Move ${state.currentSlug}.md to .trash/blog/?`);
  if (!confirmed) return;

  setStatus(`moving ${state.currentSlug}.md...`, 'loading');
  try {
    const payload = await request(`/api/posts/${state.currentSlug}`, { method: 'DELETE' });
    state.currentSlug = '';
    await refreshPosts();

    if (state.posts[0]) {
      await loadPost(state.posts[0].slug);
    } else {
      newPost();
    }

    setStatus(`moved to ${payload.trashPath}`, 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  }
};

const boot = async () => {
  setStatus('loading posts...', 'loading');
  try {
    await refreshPosts();
    if (state.posts[0]) {
      await loadPost(state.posts[0].slug);
    } else {
      newPost();
    }
  } catch (error) {
    elements.preview.innerHTML = '<p class="empty-preview">Select a post or create a new draft.</p>';
    setStatus(error.message, 'error');
  }
};

elements.newPost.addEventListener('click', newPost);
elements.deletePost.addEventListener('click', deletePost);
elements.refreshPreview.addEventListener('click', () => {
  setStatus('rendering preview...', 'loading');
  refreshPreview()
    .then(() => setStatus('preview refreshed', 'success'))
    .catch((error) => setStatus(error.message, 'error'));
});
elements.form.addEventListener('submit', (event) => {
  event.preventDefault();
  savePost();
});
elements.title.addEventListener('input', updateChrome);
elements.slug.addEventListener('input', updateChrome);
elements.content.addEventListener('input', () => {
  clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(() => {
    refreshPreview().catch((error) => setStatus(error.message, 'error'));
  }, 350);
});

await boot();

const state = {
  posts: [],
  currentSlug: '',
  isNew: false,
  editor: null,
  dialogMode: '',
  isZen: false,
  isDirty: false,
  lastSavedMarkdown: '',
};

const $ = (selector) => document.querySelector(selector);

const elements = {
  shell: $('#editor-shell'),
  status: $('#status'),
  postList: $('#post-list'),
  newPost: $('#new-post'),
  savePost: $('#save-post'),
  deletePost: $('#delete-post'),
  form: $('#post-form'),
  slug: $('#slug'),
  title: $('#title'),
  description: $('#description'),
  pubDate: $('#pubDate'),
  updatedDate: $('#updatedDate'),
  tags: $('#tags'),
  draft: $('#draft'),
  content: $('#content'),
  currentTitle: $('#current-title'),
  currentPath: $('#current-path'),
  slugLabel: $('#slug-label'),
  assetInput: $('#asset-input'),
  dialog: $('#insert-dialog'),
  dialogForm: $('#insert-form'),
  dialogKicker: $('#dialog-kicker'),
  dialogTitle: $('#dialog-title'),
  dialogTextLabel: $('#dialog-text-label'),
  dialogUrlLabel: $('#dialog-url-label'),
  dialogText: $('#dialog-text'),
  dialogUrl: $('#dialog-url'),
  dialogCancel: $('#dialog-cancel'),
};

const isMac = /Mac|iPhone|iPad|iPod/.test(navigator.platform);
const platformModLabel = isMac ? '⌘' : 'Ctrl';
const platformAltLabel = isMac ? '⌥' : 'Alt';

const shortcuts = [
  { command: 'save', label: 'Save', keys: ['Mod+S'], scope: 'global' },
  { command: 'zen', label: 'Focus', keys: ['Mod+Shift+Enter'], scope: 'global' },
  { command: 'exit-zen', label: 'Exit Zen', keys: ['Esc'], scope: 'global' },
  { command: 'h1', label: 'H1', keys: ['Alt+1'], scope: 'editor' },
  { command: 'h2', label: 'H2', keys: ['Alt+2'], scope: 'editor' },
  { command: 'h3', label: 'H3', keys: ['Alt+3'], scope: 'editor' },
  { command: 'bold', label: 'Bold', keys: ['Mod+B'], scope: 'editor' },
  { command: 'italic', label: 'Italic', keys: ['Mod+I'], scope: 'editor' },
  { command: 'link', label: 'Link', keys: ['Mod+K'], scope: 'editor' },
  { command: 'image-url', label: 'Image URL', keys: ['Mod+Alt+I'], scope: 'editor' },
  { command: 'upload-image', label: 'Upload Image', keys: ['Mod+Alt+U'], scope: 'editor' },
  { command: 'bullet', label: 'Bullet List', keys: ['Mod+Shift+8'], scope: 'editor' },
  { command: 'ordered', label: 'Ordered List', keys: ['Mod+Shift+7'], scope: 'editor' },
  { command: 'task', label: 'Task List', keys: ['Mod+Alt+X'], scope: 'editor' },
  { command: 'quote', label: 'Quote', keys: ['Mod+Alt+Q'], scope: 'editor' },
  { command: 'code', label: 'Code Block', keys: ['Mod+Alt+C'], scope: 'editor' },
  { command: 'table', label: 'Table', keys: ['Mod+Alt+L'], scope: 'editor' },
  { command: 'math', label: 'Math Block', keys: ['Mod+Alt+M'], scope: 'editor' },
  { command: 'undo', label: 'Undo', keys: ['Mod+Z'], scope: 'editor' },
  { command: 'redo', label: 'Redo', keys: ['Mod+Shift+Z', 'Ctrl+Y'], scope: 'editor' },
];

const shortcutByCommand = new Map(shortcuts.map((shortcut) => [shortcut.command, shortcut]));

const displayShortcut = (shortcut) =>
  shortcut
    .split('+')
    .map((part) => {
      if (part === 'Mod') return platformModLabel;
      if (part === 'Alt') return platformAltLabel;
      if (part === 'Shift') return isMac ? '⇧' : 'Shift';
      if (part === 'Ctrl') return 'Ctrl';
      return part;
    })
    .join(isMac && !shortcut.includes('Ctrl') ? '' : '+');

const ariaShortcut = (shortcut) => {
  const parts = shortcut.split('+');
  const variants = parts.includes('Mod')
    ? [
        parts.map((part) => (part === 'Mod' ? 'Meta' : part)).join('+'),
        parts.map((part) => (part === 'Mod' ? 'Control' : part)).join('+'),
      ]
    : [parts.join('+')];
  return variants.join(' ');
};

const shortcutMatches = (event, shortcut) => {
  const parts = shortcut.split('+');
  const key = parts.at(-1);
  const wantsMod = parts.includes('Mod');
  const wantsCtrl = parts.includes('Ctrl') || (wantsMod && !isMac);
  const wantsMeta = parts.includes('Meta') || (wantsMod && isMac);
  const wantsAlt = parts.includes('Alt');
  const wantsShift = parts.includes('Shift');
  const keyValue = event.key.length === 1 ? event.key.toUpperCase() : event.key;
  const keyMatches =
    keyValue === key ||
    (key === 'Esc' && event.key === 'Escape') ||
    (key.length === 1 && event.code === `Key${key}`) ||
    (/^\d$/.test(key) && event.code === `Digit${key}`);

  return (
    keyMatches &&
    event.altKey === wantsAlt &&
    event.shiftKey === wantsShift &&
    event.ctrlKey === wantsCtrl &&
    event.metaKey === wantsMeta
  );
};

const isDialogOpen = () => elements.dialog.open;

const isMetadataTarget = (target) =>
  Boolean(target.closest?.('#post-form') || target.closest?.('.insert-dialog'));

const isEditorTarget = (target) =>
  Boolean(target.closest?.('.milkdown-host') || target.closest?.('.typora-surface'));

const findShortcut = (event) =>
  shortcuts.find((shortcut) => shortcut.keys.some((key) => shortcutMatches(event, key)));

const enhanceToolbarShortcuts = () => {
  for (const button of document.querySelectorAll('[data-command]')) {
    const shortcut = shortcutByCommand.get(button.dataset.command);
    if (!shortcut) continue;
    const display = shortcut.keys.map(displayShortcut).join(' / ');
    button.title = `${shortcut.label} (${display})`;
    button.dataset.shortcut = display;
    button.setAttribute('aria-keyshortcuts', shortcut.keys.map(ariaShortcut).join(' '));
  }
};

const setStatus = (message, tone = 'info') => {
  elements.status.textContent = message;
  elements.status.dataset.tone = tone;
};

const setDirty = (value) => {
  state.isDirty = value;
  elements.shell.classList.toggle('is-dirty', value);
  if (value) setStatus('unsaved changes', 'loading');
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
const validateSlug = (slug) => /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug);
const currentMarkdown = () => state.editor?.getMarkdown() ?? '';

const confirmDiscard = () => {
  if (!state.isDirty) return true;
  return confirm('Discard unsaved changes?');
};

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
  content: currentMarkdown(),
});

const validateClient = (payload) => {
  const errors = [];
  if (!validateSlug(payload.slug)) {
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

const setEditorMarkdown = (content) => {
  state.editor?.setMarkdown(content ?? '');
  state.lastSavedMarkdown = content ?? '';
  setDirty(false);
};

const focusEditor = () => {
  state.editor?.focus();
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
  setEditorMarkdown(content ?? '');
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

const loadPost = async (slug) => {
  if (!confirmDiscard()) return;
  state.isNew = false;
  setStatus(`loading ${slug}.md...`, 'loading');

  try {
    const post = await request(`/api/posts/${slug}`);
    fillForm(post);
    renderList();
    setStatus(`loaded ${slug}.md`, 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  }
};

const newPost = () => {
  if (!confirmDiscard()) return;
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
    state.lastSavedMarkdown = saved.content ?? payload.content;
    setDirty(false);
    await refreshPosts();
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
  if (!confirmDiscard()) return;

  const confirmed = confirm(`Move ${state.currentSlug}.md to .trash/blog/?`);
  if (!confirmed) return;

  setStatus(`moving ${state.currentSlug}.md...`, 'loading');
  try {
    const payload = await request(`/api/posts/${state.currentSlug}`, { method: 'DELETE' });
    state.currentSlug = '';
    await refreshPosts();

    if (state.posts[0]) {
      state.isDirty = false;
      await loadPost(state.posts[0].slug);
    } else {
      state.isDirty = false;
      newPost();
    }

    setStatus(`moved to ${payload.trashPath}`, 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  }
};

const uploadAsset = async (file) => {
  const slug = state.currentSlug || elements.slug.value.trim();
  if (!validateSlug(slug)) {
    throw new Error('set a valid slug before uploading image');
  }

  const response = await fetch(`/api/assets?slug=${encodeURIComponent(slug)}&filename=${encodeURIComponent(file.name)}`, {
    method: 'POST',
    headers: { 'Content-Type': file.type || 'application/octet-stream' },
    body: file,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || 'image upload failed');
  }

  return payload;
};

const insertMarkdown = (markdown, inline = false) => {
  state.editor?.insertMarkdown(markdown, inline);
  focusEditor();
};

const insertLink = ({ text, url }) => {
  insertMarkdown(`[${text || url}](${url})`, true);
  setStatus('link inserted', 'success');
};

const insertImage = ({ alt, url }) => {
  insertMarkdown(`![${alt || 'image'}](${url})\n`);
  setStatus('image inserted', 'success');
};

const openInsertDialog = (mode) => {
  state.dialogMode = mode;
  const isImage = mode === 'image-url';
  elements.dialogKicker.textContent = isImage ? 'insert image' : 'insert link';
  elements.dialogTitle.textContent = isImage ? 'Image URL' : 'Link';
  elements.dialogTextLabel.textContent = isImage ? 'Alt text' : 'Link text';
  elements.dialogUrlLabel.textContent = isImage ? 'Image URL' : 'URL';
  elements.dialogText.value = '';
  elements.dialogUrl.value = '';
  elements.dialog.showModal();
  elements.dialogText.focus();
};

const closeInsertDialog = () => {
  elements.dialog.close();
  state.dialogMode = '';
  focusEditor();
};

const submitInsertDialog = () => {
  const text = elements.dialogText.value.trim();
  const url = elements.dialogUrl.value.trim();
  if (!url) {
    setStatus('URL is required', 'error');
    return;
  }

  if (state.dialogMode === 'image-url') {
    insertImage({ alt: text, url });
  } else {
    insertLink({ text, url });
  }
  closeInsertDialog();
};

const uploadSelectedAsset = async (file) => {
  if (!file) return;
  setStatus(`uploading ${file.name}...`, 'loading');
  try {
    const payload = await uploadAsset(file);
    const alt = file.name.replace(/\.[^.]+$/, '') || 'image';
    insertImage({ alt, url: payload.url });
    setStatus(`uploaded ${payload.url}`, 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    elements.assetInput.value = '';
  }
};

const toggleZen = (force) => {
  state.isZen = typeof force === 'boolean' ? force : !state.isZen;
  elements.shell.classList.toggle('is-zen', state.isZen);
  setStatus(state.isZen ? 'zen mode' : 'zen mode exited', 'info');
  focusEditor();
};

const handleToolbarCommand = (command) => {
  const snippets = {
    h1: '# Heading\n',
    h2: '## Heading\n',
    h3: '### Heading\n',
    bold: '**bold**',
    italic: '*italic*',
    bullet: '- list item\n',
    ordered: '1. list item\n',
    task: '- [ ] task item\n',
    quote: '> quote\n',
    code: '```js\nconsole.log("hello");\n```\n',
    table: '| Column A | Column B |\n| --- | --- |\n| Cell A | Cell B |\n',
    math: '$$\nx^2 + y^2 = z^2\n$$\n',
  };

  if (command === 'zen') toggleZen(true);
  if (command === 'exit-zen') toggleZen(false);
  if (command === 'save') savePost();
  if (command === 'link') openInsertDialog('link');
  if (command === 'image-url') openInsertDialog('image-url');
  if (command === 'upload-image') elements.assetInput.click();
  if (command === 'undo') state.editor?.undo();
  if (command === 'redo') state.editor?.redo();
  if (snippets[command]) insertMarkdown(snippets[command], command === 'bold' || command === 'italic');
};

const runCommand = (command, source = 'toolbar') => {
  const shortcut = shortcutByCommand.get(command);
  if (source === 'shortcut' && shortcut) setStatus(`shortcut: ${shortcut.label}`, 'info');
  handleToolbarCommand(command);
};

const handleKeyboardShortcut = (event) => {
  if (event.isComposing) return;

  if (isDialogOpen()) {
    if (event.key === 'Escape') {
      event.preventDefault();
      closeInsertDialog();
    }
    return;
  }

  const shortcut = findShortcut(event);
  if (!shortcut) return;
  if (shortcut.command === 'exit-zen' && !state.isZen) return;
  if (shortcut.command === 'zen' && state.isZen) {
    event.preventDefault();
    focusEditor();
    setStatus('zen mode', 'info');
    return;
  }
  if (shortcut.scope === 'editor' && (!isEditorTarget(event.target) || isMetadataTarget(event.target))) return;

  event.preventDefault();
  runCommand(shortcut.command, 'shortcut');
};

const initEditor = async () => {
  const factory = window.BlogMilkdown?.createEditor;
  if (!factory) {
    setStatus('Milkdown failed to load', 'error');
    return false;
  }

  state.editor = await factory({
    root: elements.content,
    defaultValue: '',
    onChange: (markdown) => {
      if (markdown !== state.lastSavedMarkdown) setDirty(true);
    },
    onUpload: async (file) => {
      const payload = await uploadAsset(file);
      setStatus(`uploaded ${payload.url}`, 'success');
      return payload.url;
    },
  });
  return true;
};

const boot = async () => {
  const ready = await initEditor();
  if (!ready) return;

  setStatus('loading posts...', 'loading');
  try {
    await refreshPosts();
    if (state.posts[0]) {
      state.isDirty = false;
      await loadPost(state.posts[0].slug);
    } else {
      state.isDirty = false;
      newPost();
    }
  } catch (error) {
    setStatus(error.message, 'error');
  }
};

elements.newPost.addEventListener('click', newPost);
elements.deletePost.addEventListener('click', deletePost);
elements.form.addEventListener('submit', (event) => {
  event.preventDefault();
  savePost();
});
elements.title.addEventListener('input', () => {
  updateChrome();
  setDirty(true);
});
elements.slug.addEventListener('input', updateChrome);
elements.description.addEventListener('input', () => setDirty(true));
elements.pubDate.addEventListener('input', () => setDirty(true));
elements.updatedDate.addEventListener('input', () => setDirty(true));
elements.tags.addEventListener('input', () => setDirty(true));
elements.draft.addEventListener('change', () => setDirty(true));
elements.assetInput.addEventListener('change', () => uploadSelectedAsset(elements.assetInput.files?.[0]));
elements.dialogCancel.addEventListener('click', closeInsertDialog);
elements.dialogForm.addEventListener('submit', (event) => {
  event.preventDefault();
  submitInsertDialog();
});
document.addEventListener('click', (event) => {
  const button = event.target.closest('[data-command]');
  if (!button) return;
  runCommand(button.dataset.command);
});
document.addEventListener('keydown', handleKeyboardShortcut);
window.addEventListener('beforeunload', (event) => {
  if (!state.isDirty) return;
  event.preventDefault();
  event.returnValue = '';
});

enhanceToolbarShortcuts();
await boot();

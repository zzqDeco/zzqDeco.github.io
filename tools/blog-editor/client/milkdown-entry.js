import { Crepe } from '@milkdown/crepe';
import { editorViewCtx } from '@milkdown/kit/core';
import { redo, undo } from '@milkdown/kit/prose/history';
import { insert, replaceAll } from '@milkdown/kit/utils';
import '@milkdown/crepe/theme/common/style.css';
import '@milkdown/crepe/theme/frame-dark.css';
import 'katex/dist/katex.min.css';

const createEditor = async ({ root, defaultValue = '', onChange, onUpload }) => {
  let suppressChange = false;

  const crepe = new Crepe({
    root,
    defaultValue,
    features: {
      [Crepe.Feature.TopBar]: false,
    },
    featureConfigs: {
      [Crepe.Feature.ImageBlock]: {
        onUpload,
        inlineOnUpload: onUpload,
        blockOnUpload: onUpload,
      },
    },
  });

  crepe.on((listener) => {
    listener.markdownUpdated((_ctx, markdown, previousMarkdown) => {
      if (!suppressChange) onChange?.(markdown, previousMarkdown);
    });
  });

  await crepe.create();

  const focus = () => {
    crepe.editor.action((ctx) => {
      ctx.get(editorViewCtx).focus();
    });
  };

  const setMarkdown = (markdown) => {
    suppressChange = true;
    crepe.editor.action(replaceAll(markdown ?? '', true));
    queueMicrotask(() => {
      suppressChange = false;
    });
  };

  const insertMarkdown = (markdown, inline = false) => {
    crepe.editor.action(insert(markdown, inline));
    focus();
  };

  const runHistory = (direction) => {
    crepe.editor.action((ctx) => {
      const view = ctx.get(editorViewCtx);
      const command = direction === 'redo' ? redo : undo;
      command(view.state, view.dispatch);
      view.focus();
    });
  };

  return {
    crepe,
    focus,
    getMarkdown: () => crepe.getMarkdown(),
    setMarkdown,
    insertMarkdown,
    undo: () => runHistory('undo'),
    redo: () => runHistory('redo'),
    destroy: () => crepe.destroy(),
  };
};

window.BlogMilkdown = { createEditor };

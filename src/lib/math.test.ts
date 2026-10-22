import assert from 'node:assert/strict';
import { test } from 'node:test';

import { markdownHasMath } from './math.ts';

test('markdownHasMath：检测行内公式与块级公式', () => {
  assert.equal(markdownHasMath('行内公式 $E = mc^2$ 示例'), true);
  assert.equal(markdownHasMath('$$\n\\int_0^1 x^2 dx\n$$'), true);
});

test('markdownHasMath：纯文本、代码块或转义美元符不误判为公式', () => {
  assert.equal(markdownHasMath('这是一篇没有任何数学公式的普通文章。'), false);
  assert.equal(markdownHasMath('行内代码 `$HOME` 与 `$PATH` 不应触发 KaTeX'), false);
  assert.equal(
    markdownHasMath('```bash\necho $USER\nexport PRICE=$100\n```'),
    false,
  );
  assert.equal(markdownHasMath('价格是 \\$100 到 \\$200 之间'), false);
});

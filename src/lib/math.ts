import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

interface MdastNode {
  type: string;
  children?: MdastNode[];
}

// 与 astro.config.mjs 的 markdown 管线保持一致：remark-parse + remark-gfm + remark-math。
// 代码块/行内代码在 AST 中是 code/inlineCode 节点，不会解析出 math，
// 因此比“正文是否含 $”的字符串判断可靠（不会误判转义 \$ 或代码里的 $）。
const parser = unified().use(remarkParse).use(remarkGfm).use(remarkMath);

/** 用 Markdown AST 检测正文是否含数学公式（$...$ 或 $$...$$） */
export function markdownHasMath(markdown: string): boolean {
  const tree = parser.parse(markdown) as MdastNode;
  const stack: MdastNode[] = [tree];

  while (stack.length > 0) {
    const node = stack.pop()!;
    if (node.type === 'math' || node.type === 'inlineMath') {
      return true;
    }
    if (node.children) {
      stack.push(...node.children);
    }
  }

  return false;
}

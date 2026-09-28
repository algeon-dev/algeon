// Highlights the lines of each SQL block that open a GPU table function call,
// so a reader can see where cuGraph, cuVS, or cuML work enters the statement.
// A block whose metastring already sets a `{…}` line range keeps that range.
//
// Generated function pages (scripts/gpu-docs-common.mjs) open with a
// `## Quickstart` template call to the page's own function; highlighting it
// marks nothing new, so that section stays plain.

type MdastNode = {
  type: string;
  depth?: number;
  lang?: string | null;
  meta?: string | null;
  value?: string;
  children?: MdastNode[];
};

const GPU_CALL = /\b(?:cugraph|cuvs|cuml|cuopt)_\w+\s*\(/;
const UNHIGHLIGHTED_SECTION = 'Quickstart';

export default function highlightGpuCalls() {
  return (tree: MdastNode): void => {
    let section: string | undefined;
    for (const child of tree.children ?? []) {
      if (child.type === 'heading' && child.depth === 2) {
        section = textOf(child);
      }
      if (section !== UNHIGHLIGHTED_SECTION) {
        visit(child);
      }
    }
  };
}

function visit(node: MdastNode): void {
  if (node.type === 'code' && node.lang === 'sql' && !node.meta?.includes('{')) {
    const lines = (node.value ?? '')
      .split('\n')
      .flatMap((line, index) => (GPU_CALL.test(line) ? [index + 1] : []));
    if (lines.length > 0) {
      node.meta = [`{${lines.join(',')}}`, node.meta].filter(Boolean).join(' ');
    }
  }
  node.children?.forEach(visit);
}

function textOf(node: MdastNode): string {
  return node.value ?? (node.children ?? []).map(textOf).join('');
}

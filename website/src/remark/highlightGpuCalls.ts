// Highlights the lines of each SQL block that open a GPU UDTF call,
// so a reader can see where cuGraph, cuVS, or cuML work enters the statement.
// A block whose metastring already sets a `{…}` line range keeps that range.
//
// Generated UDTF pages (scripts/gpu-docs-common.mjs) open with a
// `## Quickstart` template call to the page's own UDTF; highlighting it
// marks nothing new, so that section stays plain.

import {gpuCallMetastring} from '../utils/gpuCallHighlights';

type MdastNode = {
  type: string;
  depth?: number;
  lang?: string | null;
  meta?: string | null;
  value?: string;
  children?: MdastNode[];
};

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
  if (node.type === 'code' && node.lang === 'sql') {
    node.meta = gpuCallMetastring(node.value ?? '', node.meta ?? undefined);
  }
  node.children?.forEach(visit);
}

function textOf(node: MdastNode): string {
  return node.value ?? (node.children ?? []).map(textOf).join('');
}

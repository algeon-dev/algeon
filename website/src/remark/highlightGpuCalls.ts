// Highlights the lines of each SQL block in a demo-dataset example partial
// (`docs/*/_examples/`) that open a GPU UDTF call, so a reader can see where
// cuGraph, cuVS, or cuML work enters the scenario's statement. Reference,
// design, and integration pages show call syntax, and generated UDTF pages
// show a template call, so their blocks stay plain.
// A block whose metastring already sets a `{…}` line range keeps that range,
// and a block where every non-blank line opens a call stays plain.

import {gpuCallMetastring} from '../utils/gpuCallHighlights';

type MdastNode = {
  type: string;
  lang?: string | null;
  meta?: string | null;
  value?: string;
  children?: MdastNode[];
};

const EXAMPLE_PARTIAL = /[\\/]_examples[\\/]/;

export default function highlightGpuCalls() {
  return (tree: MdastNode, file: {path?: string}): void => {
    if (file.path !== undefined && EXAMPLE_PARTIAL.test(file.path)) {
      visit(tree);
    }
  };
}

function visit(node: MdastNode): void {
  if (node.type === 'code' && node.lang === 'sql') {
    node.meta = gpuCallMetastring(node.value ?? '', node.meta ?? undefined);
  }
  node.children?.forEach(visit);
}

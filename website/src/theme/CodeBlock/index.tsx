import React, {type ReactNode} from 'react';
import CodeBlock from '@theme-original/CodeBlock';
import type {Props} from '@theme/CodeBlock';
import {gpuCallMetastring} from '@site/src/utils/gpuCallHighlights';
import {foldRanges, type Fold} from '@site/src/utils/sqlFolds';
import {FoldContext} from './foldContext';
import {PreviewLinesContext} from './previewContext';

type CodeBlockProps = Props & {
  folds?: Fold[];
  /** Preview ends before the first line containing this marker. */
  previewBefore?: string;
};

export default function CodeBlockWrapper({folds, previewBefore, ...props}: CodeBlockProps): ReactNode {
  // Fenced blocks use className and the remark pass, which highlights only
  // example partials. Explicit language props cover captured demo SQL strings.
  const metastring = props.language === 'sql' && typeof props.children === 'string'
    ? gpuCallMetastring(props.children, props.metastring)
    : props.metastring;
  let block = <CodeBlock {...props} metastring={metastring} />;
  if (previewBefore !== undefined && typeof props.children === 'string') {
    const previewLines = props.children.split('\n').findIndex(line => line.includes(previewBefore));
    if (previewLines < 0) {
      throw new Error(`code preview marker not found: ${previewBefore}`);
    }
    block = <PreviewLinesContext.Provider value={previewLines}>{block}</PreviewLinesContext.Provider>;
  }
  if (!folds || typeof props.children !== 'string') {
    return block;
  }
  return <FoldContext.Provider value={foldRanges(props.children, folds)}>{block}</FoldContext.Provider>;
}

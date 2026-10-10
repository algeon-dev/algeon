import React, {type ReactNode} from 'react';
import CodeBlock from '@theme-original/CodeBlock';
import type {Props} from '@theme/CodeBlock';
import {gpuCallMetastring} from '@site/src/utils/gpuCallHighlights';
import {foldRanges, type Fold} from '@site/src/utils/sqlFolds';
import {FoldContext} from './foldContext';

export default function CodeBlockWrapper({folds, ...props}: Props & {folds?: Fold[]}): ReactNode {
  // Fenced blocks use className and the remark pass, which also preserves
  // Quickstart exclusions. Explicit language props cover captured SQL strings.
  const metastring = props.language === 'sql' && typeof props.children === 'string'
    ? gpuCallMetastring(props.children, props.metastring)
    : props.metastring;
  const block = <CodeBlock {...props} metastring={metastring} />;
  if (!folds || typeof props.children !== 'string') {
    return block;
  }
  return <FoldContext.Provider value={foldRanges(props.children, folds)}>{block}</FoldContext.Provider>;
}

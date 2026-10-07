/**
 * Renders a code block whose `folds` prop (see the CodeBlock wrapper) marks
 * line ranges to collapse into a row that names what is hidden and expands in
 * place. Blocks without folds use the original Content. The copy button still
 * copies the whole code because the metadata is untouched.
 */
import React, {useContext, useState, type ReactNode} from 'react';
import clsx from 'clsx';
import {useCodeBlockContext} from '@docusaurus/theme-common/internal';
import {usePrismTheme} from '@docusaurus/theme-common';
import {Highlight} from 'prism-react-renderer';
import Content from '@theme-original/CodeBlock/Content';
import type ContentType from '@theme/CodeBlock/Content';
import Line from '@theme/CodeBlock/Line';
import type {WrapperProps} from '@docusaurus/types';
import type {FoldRange} from '@site/src/utils/sqlFolds';
import {FoldContext} from '../foldContext';
import contentStyles from '@docusaurus/theme-classic/lib/theme/CodeBlock/Content/styles.module.css';

type Props = WrapperProps<typeof ContentType>;

function FoldRow({fold, open, onToggle}: {fold: FoldRange; open: boolean; onToggle: () => void}) {
  const count = fold.end - fold.start + 1;
  return (
    <button type="button" className="code-fold" aria-expanded={open} onClick={onToggle}>
      <span className="code-fold__count">{count} lines</span>
      <span className="code-fold__label">{fold.label}</span>
    </button>
  );
}

function FoldedContent({className: classNameProp, folds}: Props & {folds: FoldRange[]}): ReactNode {
  const {metadata, wordWrap} = useCodeBlockContext();
  const prismTheme = usePrismTheme();
  const {code, language, lineClassNames} = metadata;
  const [open, setOpen] = useState<boolean[]>(() => folds.map(() => false));
  const foldAt = (lineNumber: number) => folds.findIndex((fold) => fold.start === lineNumber);
  const foldOf = (lineNumber: number) =>
    folds.findIndex((fold) => fold.start <= lineNumber && lineNumber <= fold.end);

  return (
    <Highlight theme={prismTheme} code={code} language={language}>
      {({className, style, tokens: lines, getLineProps, getTokenProps}) => (
        <pre
          ref={wordWrap.codeBlockRef}
          tabIndex={0}
          className={clsx(classNameProp, className, contentStyles.codeBlock, 'thin-scrollbar')}
          style={style}>
          <code className={contentStyles.codeBlockLines}>
            {lines.map((line, i) => {
              const lineNumber = i + 1;
              const started = foldAt(lineNumber);
              const inside = foldOf(lineNumber);
              const row =
                started >= 0 ? (
                  <FoldRow
                    key={`fold-${started}`}
                    fold={folds[started]}
                    open={open[started]}
                    onToggle={() => setOpen(open.map((value, index) => (index === started ? !value : value)))}
                  />
                ) : null;
              if (inside >= 0 && !open[inside]) {
                return row;
              }
              return (
                <React.Fragment key={i}>
                  {row}
                  <Line
                    line={line}
                    getLineProps={getLineProps}
                    getTokenProps={getTokenProps}
                    classNames={lineClassNames[i]}
                    showLineNumbers={false}
                  />
                </React.Fragment>
              );
            })}
          </code>
        </pre>
      )}
    </Highlight>
  );
}

export default function ContentWrapper(props: Props): ReactNode {
  const folds = useContext(FoldContext);
  if (folds.length === 0) {
    return <Content {...props} />;
  }
  return <FoldedContent {...props} folds={folds} />;
}

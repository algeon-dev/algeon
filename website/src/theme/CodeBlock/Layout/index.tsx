/**
 * Collapses long code blocks to their first lines behind a fade, with a
 * button that reveals the rest. The copy button still copies the full code.
 * The no-collapse class keeps report queries visible in the reading flow.
 */
import React, {useContext, useId, useRef, useState, type CSSProperties, type ReactNode} from 'react';
import clsx from 'clsx';
import {usePrismTheme} from '@docusaurus/theme-common';
import {useCodeBlockContext} from '@docusaurus/theme-common/internal';
import Layout from '@theme-original/CodeBlock/Layout';
import type LayoutType from '@theme/CodeBlock/Layout';
import type {WrapperProps} from '@docusaurus/types';
import {PreviewLinesContext} from '../previewContext';
import styles from './styles.module.css';

// Blocks up to this length stay open; collapsing a block only a few lines
// past the visible height would hide almost nothing.
const COLLAPSE_ABOVE_LINES = 30;

type Props = WrapperProps<typeof LayoutType>;

export default function LayoutWrapper(props: Props): ReactNode {
  const {metadata} = useCodeBlockContext();
  const previewLines = useContext(PreviewLinesContext);
  const prismTheme = usePrismTheme();
  const [expanded, setExpanded] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const clipId = useId();
  const lineCount = metadata.code.split('\n').length;

  const collapsible = previewLines === undefined
    ? lineCount > COLLAPSE_ABOVE_LINES && !metadata.className.split(/\s+/).includes('no-collapse')
    : previewLines < lineCount;

  if (!collapsible) {
    return <Layout {...props} />;
  }

  const toggle = () => {
    setExpanded(!expanded);
    // Collapsing a block read past its top would leave the reader below it.
    const wrapper = wrapperRef.current;
    if (expanded && wrapper && wrapper.getBoundingClientRect().top < 0) {
      wrapper.scrollIntoView({block: 'start'});
    }
  };

  return (
    <div
      ref={wrapperRef}
      className={clsx(styles.collapsible, previewLines !== undefined && styles.preview, !expanded && styles.collapsed)}
      style={previewLines === undefined ? undefined : {
        '--code-preview-lines': previewLines,
        '--code-preview-background': prismTheme.plain.backgroundColor,
      } as CSSProperties}>
      <div id={clipId} className={styles.clip}>
        <Layout {...props} />
      </div>
      <button
        type="button"
        className={styles.toggle}
        aria-expanded={expanded}
        aria-controls={clipId}
        onClick={toggle}>
        {expanded ? 'Show less' : previewLines === undefined ? `Show all ${lineCount} lines` : 'Show more'}
      </button>
    </div>
  );
}

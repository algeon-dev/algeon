/**
 * Collapses long code blocks to their first lines behind a fade, with a
 * button that reveals the rest. The copy button still copies the full code.
 */
import React, {useId, useRef, useState, type ReactNode} from 'react';
import clsx from 'clsx';
import {useCodeBlockContext} from '@docusaurus/theme-common/internal';
import Layout from '@theme-original/CodeBlock/Layout';
import type LayoutType from '@theme/CodeBlock/Layout';
import type {WrapperProps} from '@docusaurus/types';
import styles from './styles.module.css';

// Blocks up to this length stay open; collapsing a block only a few lines
// past the visible height would hide almost nothing.
const COLLAPSE_ABOVE_LINES = 30;

type Props = WrapperProps<typeof LayoutType>;

export default function LayoutWrapper(props: Props): ReactNode {
  const {metadata} = useCodeBlockContext();
  const [expanded, setExpanded] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const clipId = useId();
  const lineCount = metadata.code.split('\n').length;

  if (lineCount <= COLLAPSE_ABOVE_LINES) {
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
      className={clsx(styles.collapsible, !expanded && styles.collapsed)}>
      <div id={clipId} className={styles.clip}>
        <Layout {...props} />
      </div>
      <button
        type="button"
        className={styles.toggle}
        aria-expanded={expanded}
        aria-controls={clipId}
        onClick={toggle}>
        {expanded ? 'Show less' : `Show all ${lineCount} lines`}
      </button>
    </div>
  );
}

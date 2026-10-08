import React, {type ReactNode, useEffect, useId, useState} from 'react';
import clsx from 'clsx';

import {useFocusMode} from './FocusModeProvider';
import styles from './styles.module.css';

function isApplePlatform(): boolean {
  const nav = navigator as Navigator & {userAgentData?: {platform: string}};
  return /mac|iphone|ipad|ipod/i.test(
    nav.userAgentData?.platform || nav.platform || nav.userAgent,
  );
}

// Lucide "focus": a dot inside four corner brackets, so it does not read as
// the browser's fullscreen arrows.
function IconFocus() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="3" />
      <path d="M3 7V5a2 2 0 0 1 2-2h2" />
      <path d="M17 3h2a2 2 0 0 1 2 2v2" />
      <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
      <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
    </svg>
  );
}

export default function FocusModeToggle(): ReactNode {
  const {active, setActive} = useFocusMode();
  const detailId = useId();
  // Server output and hydration use the non-Apple label.
  const [shortcut, setShortcut] = useState('Ctrl+Shift+F');

  useEffect(() => {
    if (isApplePlatform()) {
      setShortcut('⌘⇧F');
    }
  }, []);

  return (
    <span className={styles.toggleContainer}>
      <button
        type="button"
        className={clsx(styles.floatingControl, styles.toggle)}
        aria-label="Focus mode"
        aria-describedby={detailId}
        aria-pressed={active}
        aria-keyshortcuts="Control+Shift+F Meta+Shift+F"
        onClick={() => setActive(!active)}
      >
        <span id={detailId} className={styles.controlDetail}>
          Focus mode
          <kbd className={styles.controlKey}>{shortcut}</kbd>
        </span>
        <span className={styles.controlIcon}>
          <IconFocus />
        </span>
      </button>
    </span>
  );
}

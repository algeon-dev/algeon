import React, {type ReactNode} from 'react';
import clsx from 'clsx';

import {useFocusMode} from './FocusModeProvider';
import styles from './styles.module.css';

function IconClose() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}

/**
 * The exit button and the live region for focus mode switches. Both are always
 * rendered: CSS shows the button from the first paint after a reload, and the
 * live region must exist before its text changes.
 */
export default function FocusModeExit(): ReactNode {
  const {setActive, announcement} = useFocusMode();
  return (
    <>
      <button
        type="button"
        className={clsx(styles.floatingControl, styles.exit)}
        aria-label="Exit focus mode"
        aria-keyshortcuts="Escape"
        onClick={() => setActive(false)}
      >
        <span className={styles.exitDetail}>
          Focus mode
          <kbd className={styles.exitKey}>Esc</kbd>
        </span>
        <span className={styles.exitIcon}>
          <IconClose />
        </span>
      </button>
      <div className={styles.visuallyHidden} aria-live="polite">
        {announcement}
      </div>
    </>
  );
}

import React, {
  type ReactNode,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react';
import clsx from 'clsx';
import TOCItems from '@theme/TOCItems';
import type {Props as TOCItemsProps} from '@theme/TOCItems';
import focusModeStyles from '@site/src/components/focusMode/styles.module.css';

import styles from './styles.module.css';

const CLOSE_DELAY_MS = 220;

export interface Props {
  readonly toc: TOCItemsProps['toc'];
  readonly minHeadingLevel?: number;
  readonly maxHeadingLevel?: number;
  /** Changes with the page, so a panel left open closes on navigation. */
  readonly pageKey: string;
  /**
   * The drawer shows in focus mode on desktop widths by itself; an owning
   * layout adds a class with the other widths at which it shows.
   */
  readonly className?: string;
}

function IconOutline({className}: {className?: string}) {
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
      className={className}
      aria-hidden="true"
    >
      <line x1="21" y1="6" x2="9" y2="6" />
      <line x1="21" y1="12" x2="3" y2="12" />
      <line x1="21" y1="18" x2="7" y2="18" />
    </svg>
  );
}

/**
 * The right-edge outline: a round button matching the focus mode exit
 * button; hovering or clicking it grows the "On this page" panel out of the
 * button over the content. Docs use it below the pinned-column width and in
 * focus mode; blog posts use it in focus mode, where their outline column is
 * hidden.
 */
export default function OutlineDrawer({
  toc,
  minHeadingLevel,
  maxHeadingLevel,
  pageKey,
  className,
}: Props): ReactNode {
  const panelId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const focusOnOpenRef = useRef(false);
  const drawerRef = useRef<HTMLDivElement>(null);
  const handleRef = useRef<HTMLButtonElement>(null);

  const cancelClose = () => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };

  const closeAfter = (delay: number) => {
    cancelClose();
    closeTimerRef.current = setTimeout(() => setIsOpen(false), delay);
  };

  useEffect(() => {
    setIsOpen(false);
    return cancelClose;
  }, [pageKey]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    if (focusOnOpenRef.current) {
      focusOnOpenRef.current = false;
      const panel = drawerRef.current;
      const link =
        panel?.querySelector<HTMLElement>('.table-of-contents__link--active') ??
        panel?.querySelector<HTMLElement>('a');
      link?.focus();
    }
    // On document so it runs before the focus mode Esc listener on window,
    // which skips events that are already handled.
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) {
        return;
      }
      event.preventDefault();
      cancelClose();
      setIsOpen(false);
      if (drawerRef.current?.contains(document.activeElement)) {
        handleRef.current?.focus();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [isOpen]);

  const open = (fromKeyboard: boolean) => {
    cancelClose();
    focusOnOpenRef.current = fromKeyboard;
    setIsOpen(true);
  };

  return (
    <div
      ref={drawerRef}
      className={clsx(styles.outlineDrawer, isOpen && styles.outlineDrawerOpen, className)}
      onMouseEnter={() => open(false)}
      onMouseLeave={() => closeAfter(CLOSE_DELAY_MS)}
      // Close when keyboard focus moves to something outside. Focus that
      // drops to nothing, as after a fragment link click, leaves the
      // panel to the pointer.
      onBlur={(event) => {
        const next = event.relatedTarget;
        if (next && !event.currentTarget.contains(next)) {
          closeAfter(0);
        }
      }}
    >
      <button
        ref={handleRef}
        type="button"
        className={clsx(focusModeStyles.floatingControl, styles.outlineHandle)}
        // A keyboard click has no pointer position (detail 0).
        onClick={(event) => open(event.detail === 0)}
        aria-label="On this page"
        aria-expanded={isOpen}
        aria-controls={panelId}
      >
        <IconOutline className={styles.iconOutline} />
      </button>

      <nav id={panelId} className={styles.outlinePanel} aria-label="On this page">
        {/* A link click only scrolls; the panel stays until the pointer
            or focus leaves, so the reader can jump between sections. */}
        <div className={clsx(styles.outlinePanelBody, 'thin-scrollbar')}>
          <TOCItems
            toc={toc}
            minHeadingLevel={minHeadingLevel}
            maxHeadingLevel={maxHeadingLevel}
            // The highlight hook marks only the first matching link among
            // elements with all of these classes; a class of its own keeps
            // the page's hidden outline copy from taking the highlight.
            linkClassName="table-of-contents__link outline-drawer-link"
            linkActiveClassName="table-of-contents__link--active"
          />
        </div>
      </nav>
    </div>
  );
}

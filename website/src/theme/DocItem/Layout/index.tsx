import React, {
  type ReactNode,
  useState,
  useEffect,
  useId,
  useRef,
} from 'react';
import clsx from 'clsx';
import {useWindowSize} from '@docusaurus/theme-common';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import DocItemPaginator from '@theme/DocItem/Paginator';
import DocVersionBanner from '@theme/DocVersionBanner';
import DocVersionBadge from '@theme/DocVersionBadge';
import DocItemFooter from '@theme/DocItem/Footer';
import DocItemTOCMobile from '@theme/DocItem/TOC/Mobile';
import DocItemTOCDesktop from '@theme/DocItem/TOC/Desktop';
import DocItemContent from '@theme/DocItem/Content';
import DocBreadcrumbs from '@theme/DocBreadcrumbs';
import ContentVisibility from '@theme/ContentVisibility';
import TOCItems from '@theme/TOCItems';
import type {Props} from '@theme/DocItem/Layout';
import FocusModeToggle from '@site/src/components/focusMode/FocusModeToggle';
import focusModeStyles from '@site/src/components/focusMode/styles.module.css';

import styles from './styles.module.css';

const CLOSE_DELAY_MS = 220;

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

export default function DocItemLayout({children}: Props): ReactNode {
  const {frontMatter, toc, metadata} = useDoc();
  const windowSize = useWindowSize();

  const hidden = frontMatter.hide_table_of_contents;
  const canRender = !hidden && toc.length > 0;
  const isDesktop = windowSize === 'desktop' || windowSize === 'ssr';
  const showOutline = canRender && isDesktop;

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
  }, [metadata.id]);

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
    // Which outline shows (pinned column or right-edge drawer) depends on the
    // viewport width and focus mode, both resolved in CSS so the right one is
    // there from the first paint and inside the focus mode view transition.
    <div
      className={clsx(
        'row',
        styles.docRow,
        metadata.id === 'intro' && styles.pinnedOnDesktop,
      )}
    >
      <div
        className={clsx(
          'col',
          !hidden && (canRender ? styles.docItemCol : styles.docItemColFull),
        )}
      >
        <ContentVisibility metadata={metadata} />
        <DocVersionBanner />
        <div className={styles.docItemContainer} data-focus-mode-content>
          <article>
            <div className={styles.pageTools}>
              <DocBreadcrumbs />
              <FocusModeToggle />
            </div>
            <DocVersionBadge />
            {canRender && <DocItemTOCMobile />}
            <DocItemContent>{children}</DocItemContent>
            <DocItemFooter />
          </article>
          <DocItemPaginator />
        </div>
      </div>

      {showOutline && (
        <div className={clsx('col col--3', styles.tocCol)}>
          <div className={styles.pinnedTocContainer}>
            <DocItemTOCDesktop />
          </div>
        </div>
      )}

      {showOutline && (
        <div
          ref={drawerRef}
          className={clsx(styles.outlineDrawer, isOpen && styles.outlineDrawerOpen)}
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
                minHeadingLevel={frontMatter.toc_min_heading_level}
                maxHeadingLevel={frontMatter.toc_max_heading_level}
                // The highlight hook marks only the first matching link among
                // elements with all of these classes; a class of its own keeps
                // the hidden pinned copy from taking the highlight.
                linkClassName="table-of-contents__link outline-drawer-link"
                linkActiveClassName="table-of-contents__link--active"
              />
            </div>
          </nav>
        </div>
      )}
    </div>
  );
}

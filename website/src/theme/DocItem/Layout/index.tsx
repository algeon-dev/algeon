import React, {
  type ReactNode,
  useState,
  useEffect,
  useRef,
  useCallback,
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

import styles from './styles.module.css';

type TocMode = 'pinned' | 'floating';

const STORAGE_KEY = 'algeon-doc-toc-mode';
const WIDE_SCREEN_BREAKPOINT = 1360;

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

function IconArrowRight({className}: {className?: string}) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}

function IconPin({className}: {className?: string}) {
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
      className={className}
      aria-hidden="true"
    >
      <line x1="12" y1="17" x2="12" y2="22" />
      <path d="M5 17h14v-2l-2-3V5a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7l-2 3v2z" />
    </svg>
  );
}


export default function DocItemLayout({children}: Props): ReactNode {
  const {frontMatter, toc, metadata} = useDoc();
  const windowSize = useWindowSize();

  const hidden = frontMatter.hide_table_of_contents;
  const canRender = !hidden && toc.length > 0;

  const [tocMode, setTocMode] = useState<TocMode>('pinned');
  const [isHoverOpen, setIsHoverOpen] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const leaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setIsClient(true);
    setIsHoverOpen(false);

    if (metadata.id === 'intro') {
      setTocMode('pinned');
      return;
    }

    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'pinned' || stored === 'floating') {
      setTocMode(stored);
      return;
    }
    const defaultMode =
      window.innerWidth >= WIDE_SCREEN_BREAKPOINT ? 'pinned' : 'floating';
    setTocMode(defaultMode);

    const onResize = () => {
      const explicit = localStorage.getItem(STORAGE_KEY);
      if (!explicit) {
        setTocMode(
          window.innerWidth >= WIDE_SCREEN_BREAKPOINT ? 'pinned' : 'floating',
        );
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [metadata.id]);

  const updateTocMode = useCallback((mode: TocMode) => {
    setTocMode(mode);
    setIsHoverOpen(false);
    if (metadata.id === 'intro') {
      return;
    }
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // ignore storage failure in private tabs
    }
  }, [metadata.id]);

  const handleMouseEnter = () => {
    if (leaveTimerRef.current) {
      clearTimeout(leaveTimerRef.current);
      leaveTimerRef.current = null;
    }
    setIsHoverOpen(true);
  };

  const handleMouseLeave = () => {
    if (leaveTimerRef.current) {
      clearTimeout(leaveTimerRef.current);
    }
    leaveTimerRef.current = setTimeout(() => {
      setIsHoverOpen(false);
    }, 220);
  };

  const handleTocLinkClick = () => {
    setTimeout(() => setIsHoverOpen(false), 260);
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isHoverOpen) {
        setIsHoverOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isHoverOpen]);

  const isDesktop = windowSize === 'desktop' || windowSize === 'ssr';
  const showPinnedTOC = canRender && isDesktop && tocMode === 'pinned';
  const showFloatingTOC = canRender && isDesktop && tocMode === 'floating' && isClient;

  return (
    <div className="row">
      <div
        className={clsx(
          'col',
          !hidden && (showPinnedTOC ? styles.docItemCol : styles.docItemColFull),
        )}
      >
        <ContentVisibility metadata={metadata} />
        <DocVersionBanner />
        <div className={styles.docItemContainer}>
          <article>
            <DocBreadcrumbs />
            <DocVersionBadge />
            {canRender && <DocItemTOCMobile />}
            <DocItemContent>{children}</DocItemContent>
            <DocItemFooter />
          </article>
          <DocItemPaginator />
        </div>
      </div>

      {showPinnedTOC && (
        <div className="col col--3">
          <div className={styles.pinnedTocContainer}>
            <div className={styles.pinnedTocLineHeader}>
              <button
                type="button"
                className={styles.tocCollapseLineButton}
                onClick={() => updateTocMode('floating')}
                title="Collapse outline"
                aria-label="Collapse outline"
              >
                <IconArrowRight />
              </button>
            </div>
            <div className={styles.pinnedTocTitle}>On this page</div>
            <DocItemTOCDesktop />
          </div>
        </div>
      )}

      {showFloatingTOC && (
        <aside
          className={clsx(
            styles.floatingTocWrapper,
            isHoverOpen && styles.floatingTocWrapperOpen,
          )}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
          aria-label="Outline navigation"
        >
          <button
            type="button"
            className={styles.floatingTocPill}
            onClick={() => setIsHoverOpen((v) => !v)}
            title="Outline"
            aria-label="Table of contents outline"
            aria-expanded={isHoverOpen}
          >
            <IconOutline className={styles.iconOutline} />
          </button>

          <div className={styles.floatingTocPanel}>
            <div className={styles.floatingTocPanelHeader}>
              <span className={styles.floatingTocTitle}>Outlines</span>
              <div className={styles.floatingTocHeaderActions}>
                <button
                  type="button"
                  className={styles.tocActionBtn}
                  onClick={() => updateTocMode('pinned')}
                  title="Pin outline to side"
                  aria-label="Pin outline to side"
                >
                  <IconPin />
                </button>
              </div>
            </div>
            <div
              className={clsx(styles.floatingTocPanelBody, 'thin-scrollbar')}
              onClick={handleTocLinkClick}
            >
              <TOCItems
                toc={toc}
                minHeadingLevel={frontMatter.toc_min_heading_level}
                maxHeadingLevel={frontMatter.toc_max_heading_level}
                linkClassName="table-of-contents__link toc-highlight"
                linkActiveClassName="table-of-contents__link--active"
              />
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}

import React, {type ReactNode} from 'react';
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
import type {Props} from '@theme/DocItem/Layout';
import FocusModeToggle from '@site/src/components/focusMode/FocusModeToggle';
import OutlineDrawer from '@site/src/components/outline/OutlineDrawer';

import styles from './styles.module.css';

export default function DocItemLayout({children}: Props): ReactNode {
  const {frontMatter, toc, metadata} = useDoc();
  const windowSize = useWindowSize();

  const hidden = frontMatter.hide_table_of_contents;
  const canRender = !hidden && toc.length > 0;
  const isDesktop = windowSize === 'desktop' || windowSize === 'ssr';
  const showOutline = canRender && isDesktop;

  // The homepage opens on its hero: no Home / Overview trail, no collapsed
  // outline above it on narrow screens, and the tools row (then only the
  // focus mode toggle) is pinned at the top right so it takes no height. It
  // also keeps the pinned outline column at widths where other pages use
  // the drawer.
  const isHome = metadata.id === 'intro';

  return (
    // Which outline shows (pinned column or right-edge drawer) depends on the
    // viewport width and focus mode, both resolved in CSS so the right one is
    // there from the first paint and inside the focus mode view transition.
    <div
      className={clsx(
        'row',
        styles.docRow,
        isHome && styles.home,
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
              {!isHome && <DocBreadcrumbs />}
              <FocusModeToggle />
            </div>
            <DocVersionBadge />
            {canRender && !isHome && <DocItemTOCMobile />}
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
        <OutlineDrawer
          toc={toc}
          minHeadingLevel={frontMatter.toc_min_heading_level}
          maxHeadingLevel={frontMatter.toc_max_heading_level}
          pageKey={metadata.id}
          className={styles.docDrawer}
        />
      )}
    </div>
  );
}

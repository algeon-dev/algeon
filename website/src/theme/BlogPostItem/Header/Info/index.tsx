import React, {type ReactNode} from 'react';
import {useBlogPost} from '@docusaurus/plugin-content-blog/client';
import BlogPostItemHeaderInfo from '@theme-original/BlogPostItem/Header/Info';
import type {Props} from '@theme/BlogPostItem/Header/Info';
import FocusModeToggle from '@site/src/components/focusMode/FocusModeToggle';

import styles from './styles.module.css';

export default function BlogPostInfo(props: Props): ReactNode {
  const {isBlogPostPage} = useBlogPost();

  if (!isBlogPostPage) {
    return <BlogPostItemHeaderInfo {...props} />;
  }

  return (
    <div className={styles.infoRow}>
      <BlogPostItemHeaderInfo {...props} />
      <div className={styles.pageTools}>
        <FocusModeToggle />
      </div>
    </div>
  );
}

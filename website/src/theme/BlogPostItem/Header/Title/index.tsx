import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useBlogPost} from '@docusaurus/plugin-content-blog/client';
import type {Props} from '@theme/BlogPostItem/Header/Title';

import styles from './styles.module.css';

const SEPARATOR = ': ';

/**
 * A post title written as "Name: Subtitle" renders on two lines, the name as
 * the heading and the subtitle beneath it in a lighter weight. The
 * frontmatter keeps the full string, so the document title, metadata and
 * feeds are unchanged.
 */
export default function BlogPostItemHeaderTitle({className}: Props): ReactNode {
  const {metadata, isBlogPostPage} = useBlogPost();
  const {permalink, title} = metadata;
  const TitleHeading = isBlogPostPage ? 'h1' : 'h2';

  const separatorAt = title.indexOf(SEPARATOR);
  const text =
    separatorAt > 0 ? (
      <>
        {title.slice(0, separatorAt)}
        <span className={styles.subtitle}>{title.slice(separatorAt + SEPARATOR.length)}</span>
      </>
    ) : (
      title
    );

  return (
    <TitleHeading className={clsx(styles.title, className)}>
      {isBlogPostPage ? text : <Link to={permalink}>{text}</Link>}
    </TitleHeading>
  );
}

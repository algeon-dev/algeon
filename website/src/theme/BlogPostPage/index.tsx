import React, {type ReactNode} from 'react';
import {useWindowSize} from '@docusaurus/theme-common';
import BlogPostPage from '@theme-original/BlogPostPage';
import type {Props} from '@theme/BlogPostPage';
import OutlineDrawer from '@site/src/components/outline/OutlineDrawer';

/**
 * Focus mode hides the blog's outline column with the rest of the chrome
 * (custom.css), so the post gets the same right-edge outline drawer as a
 * doc. In normal mode the drawer stays hidden and the column serves.
 */
export default function BlogPostPageWrapper(props: Props): ReactNode {
  const {toc, frontMatter, metadata} = props.content;
  const windowSize = useWindowSize();
  const isDesktop = windowSize === 'desktop' || windowSize === 'ssr';
  const showOutline =
    !frontMatter.hide_table_of_contents && toc.length > 0 && isDesktop;

  return (
    <>
      <BlogPostPage {...props} />
      {showOutline && (
        <OutlineDrawer
          toc={toc}
          minHeadingLevel={frontMatter.toc_min_heading_level}
          maxHeadingLevel={frontMatter.toc_max_heading_level}
          pageKey={metadata.permalink}
        />
      )}
    </>
  );
}

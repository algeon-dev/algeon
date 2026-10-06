import React, {type ReactNode} from 'react';
import DocRootLayout from '@theme-original/DocRoot/Layout';
import type {Props} from '@theme/DocRoot/Layout';

import styles from './styles.module.css';

export default function DocRootLayoutWrapper(props: Props): ReactNode {
  return (
    <div className={styles.layout}>
      <DocRootLayout {...props} />
    </div>
  );
}

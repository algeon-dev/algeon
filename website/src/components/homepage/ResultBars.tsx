import React, {type ReactNode} from 'react';
import styles from './ResultBars.module.css';

export type ResultBar = {
  label: ReactNode;
  value: number;
  display: string;
  detail?: ReactNode;
};

// Captured result rows for a homepage example. Bars scale from zero to the
// largest value, so equal-looking bars mean close values, never a clipped axis.
export default function ResultBars({
  summary,
  context,
  rows,
}: {
  summary: ReactNode;
  context?: ReactNode;
  rows: ResultBar[];
}): React.ReactElement {
  const max = Math.max(...rows.map(row => row.value));
  return (
    <figure className={styles.result}>
      <figcaption className={styles.summary}>
        <span className={styles.badge}>Result</span>
        <span>{summary}</span>
      </figcaption>
      {context && <p className={styles.context}>{context}</p>}
      <ol className={styles.rows}>
        {rows.map((row, index) => (
          <li key={index} className={styles.row}>
            <span className={styles.label}>{row.label}</span>
            <span className={styles.track} aria-hidden="true">
              <span className={styles.bar} style={{width: `${(row.value / max) * 100}%`}} />
            </span>
            <span className={styles.value}>{row.display}</span>
            <span className={styles.detail}>{row.detail}</span>
          </li>
        ))}
      </ol>
    </figure>
  );
}

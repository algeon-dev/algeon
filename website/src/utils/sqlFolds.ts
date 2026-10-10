/**
 * Fold ranges for long captured statements. A fold is expressed by text
 * markers so that a recapture with different line numbers keeps working:
 * `to` folds everything before the first line containing it, `from` folds
 * everything from the first line containing it. The CodeBlock wrapper turns
 * the `folds` prop into ranges and the Content wrapper renders them as
 * collapsible rows.
 */

export type Fold = {from?: string; to?: string; label: string};

export type FoldRange = {start: number; end: number; label: string};

export function foldRanges(code: string, folds: Fold[]): FoldRange[] {
  const lines = code.split('\n');
  const find = (marker: string): number => {
    const index = lines.findIndex((line) => line.includes(marker));
    if (index < 0) {
      throw new Error(`fold marker not found: ${marker}`);
    }
    return index + 1;
  };
  return folds.map(({from, to, label}) => {
    const start = from === undefined ? 1 : find(from);
    const end = to === undefined ? lines.length : find(to) - 1;
    if (end < start) {
      throw new Error(`empty fold: ${label}`);
    }
    return {start, end, label};
  });
}

import clsx from 'clsx';
import React from 'react';
import {formatNumber} from './benchmarkChartUtils';
import styles from './benchmarks.module.css';

type Distribution = {readonly min: number; readonly median: number; readonly max: number};

type ParquetLayoutFile = {
  readonly bytes: number;
  readonly rows: number;
  readonly createdBy: string;
  readonly rowGroups: number;
  readonly rowGroupRows: Distribution;
  readonly rowGroupUncompressedBytes: Distribution;
  readonly dataPages: number;
  readonly rowsPerPage: {readonly median: number; readonly max: number};
  readonly pageSource: string;
  readonly codecs: readonly string[];
  readonly dictionaryChunks: number;
  readonly columnChunks: number;
  readonly nonOverlappingColumns: readonly string[];
};

// The measured `layout.json` of fixture/tpc/describe_layout.py.
export type ParquetLayout = {
  readonly generation: {
    readonly layout: string;
    readonly compression: string;
    readonly spark_version?: string;
    readonly parquet_java_version?: string;
    readonly parquet_block_size?: number;
    readonly parquet_page_size?: number;
    readonly parquet_page_row_count_limit?: number;
    readonly row_group_bytes?: number;
    readonly tables: readonly {readonly name: string; readonly sort_keys: readonly string[]}[];
  };
  readonly partitioning: {readonly kind: string; readonly fileCount: number; readonly hiveKeys: readonly string[]};
  readonly tables: Readonly<Record<string, ParquetLayoutFile>>;
};

export type ParquetLayoutColumn = {readonly label: string; readonly layout: ParquetLayout};

const bytesPerMiB = 1024 ** 2;
const bytesPerGiB = 1024 ** 3;

function mib(bytes: number): string {
  return `${formatNumber(Math.round(bytes / bytesPerMiB))} MiB`;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function describeColumn({layout}: ParquetLayoutColumn): string[] {
  const {generation, partitioning} = layout;
  const files = Object.values(layout.tables);
  const [largestName, largest] = Object.entries(layout.tables).sort((a, b) => b[1].rows - a[1].rows)[0];
  const sorted = generation.layout.endsWith('sorted_row_groups');
  const sortKeys = generation.tables.find((table) => table.name === largestName)?.sort_keys ?? [];
  const duckdbVersion = /^DuckDB version (v[\d.]+)/.exec(largest.createdBy)?.[1];
  const writer = generation.spark_version
    ? `Spark ${generation.spark_version}, parquet-java ${generation.parquet_java_version}`
    : duckdbVersion
      ? `DuckDB ${duckdbVersion}`
      : largest.createdBy;
  const rowGroupTarget = generation.parquet_block_size
    ? `${mib(generation.parquet_block_size)} (parquet.block.size)`
    : `${mib(generation.row_group_bytes ?? 0)} (row_group_bytes)`;
  const dictionary = `${formatNumber(sum(files.map((f) => f.dictionaryChunks)))} of ${formatNumber(
    sum(files.map((f) => f.columnChunks)),
  )} column chunks`;
  return [
    writer,
    sorted ? `Sorted by ${sortKeys.join(', ')} (${largestName})` : 'dbgen order, no sort',
    rowGroupTarget,
    `${formatNumber(largest.rowGroups)} row groups, median ${formatNumber(largest.rowGroupRows.median)} rows ` +
      `(${formatNumber(largest.rowGroupRows.min)} to ${formatNumber(largest.rowGroupRows.max)}), ` +
      `median ${mib(largest.rowGroupUncompressedBytes.median)} uncompressed`,
    formatNumber(sum(files.map((f) => f.rowGroups))),
    generation.parquet_page_size ? mib(generation.parquet_page_size) : 'Writer default, not recorded',
    generation.parquet_page_row_count_limit
      ? `${formatNumber(generation.parquet_page_row_count_limit)} rows`
      : 'Writer default, not recorded',
    `${formatNumber(largest.dataPages)} data pages, median ${formatNumber(largest.rowsPerPage.median)} rows, ` +
      `maximum ${formatNumber(largest.rowsPerPage.max)} rows`,
    `${generation.compression} in every column chunk`,
    `${dictionary} carry a dictionary page`,
    files.every((f) => f.pageSource === 'offset_index') ? 'Yes' : 'No, page headers read',
    largest.nonOverlappingColumns.length > 0 ? largest.nonOverlappingColumns.join(', ') : 'None',
    `${partitioning.fileCount} files, one per table; ${
      partitioning.hiveKeys.length === 0 ? 'no Hive partition keys' : `Hive keys ${partitioning.hiveKeys.join(', ')}`
    }`,
    `${formatNumber(sum(files.map((f) => f.rows)))} rows, ${(sum(files.map((f) => f.bytes)) / bytesPerGiB).toFixed(1)} GiB`,
  ];
}

const rowLabels = [
  'Writer',
  'Row order',
  'Row group target',
  'Largest table row groups',
  'Row groups, all tables',
  'Data page size target',
  'Data page row limit',
  'Largest table data pages',
  'Compression',
  'Dictionary encoding',
  'Offset index',
  'Largest table, row-group ranges disjoint on',
  'Files',
  'Total',
];

export default function ParquetLayoutTable({columns}: {columns: readonly ParquetLayoutColumn[]}): React.ReactElement {
  const cells = columns.map(describeColumn);
  return (
    <div className={clsx(styles.tablePanel, 'thin-scrollbar')}>
      <table className={clsx(styles.resultsTable, styles.wrapHeaders, styles.wrapCells)}>
        <thead>
          <tr>
            <th>Property</th>
            {columns.map((column) => (
              <th key={column.label}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rowLabels.map((label, index) => (
            <tr key={label}>
              <td className={styles.queryCell}>{label}</td>
              {cells.map((column, columnIndex) => (
                <td key={columns[columnIndex].label}>{column[index]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

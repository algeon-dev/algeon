import clsx from 'clsx';
import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import {tpchSf100Benchmark} from '@site/src/data/tpchSf100Benchmark';
import {
  bytesPerGiB,
  formatGiB,
  formatGiBShort,
  formatNumber,
  tooltipContentStyle,
  tooltipCursorStyle,
  tooltipItemStyle,
  tooltipLabelStyle,
} from './benchmarkChartUtils';
import styles from './TpchSf100Benchmark.module.css';

// Palette values match the Tremor chart colors.
const colorCpu = '#0071C5'; // cpu-blue
const colorGpu = '#76B900'; // gpu-green
const colorSpeedup = '#10b981'; // emerald
const colorMemory = '#6366f1'; // indigo

function formatMs(value: number): string {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}s`;
  }
  return `${value.toFixed(0)}ms`;
}

function formatSeconds(value: number): string {
  return `${(value / 1000).toFixed(1)}s`;
}

function formatSpeedup(value: number): string {
  return `${value.toFixed(1)}x`;
}

export default function TpchSf100Benchmark(): React.ReactElement {
  const {dataset, config, rows, summary} = tpchSf100Benchmark;
  const runtimeData = rows.map((row) => ({
    query: row.queryId,
    CPU: row.cpuMs,
    GPU: row.gpuMs,
  }));
  const speedupData = rows.map((row) => ({
    query: row.queryId,
    'CPU/GPU': row.speedup,
  }));
  const memoryData = rows.map((row) => ({
    query: row.queryId,
    'Task memory peak': row.peakMemoryOperatorClassBytes / bytesPerGiB,
  }));

  return (
    <div className={styles.benchmark}>
      <section className={styles.header}>
        <div>
          <ul className={styles.benchmarkFacts}>
            <li>
              <strong>Result:</strong> CPU {formatSeconds(summary.cpuTotalMs)},
              GPU {formatSeconds(summary.gpuTotalMs)}, CPU / GPU{' '}
              {formatSpeedup(summary.speedup)}, largest task memory peak{' '}
              {formatGiB(summary.maxPeakMemoryOperatorClassBytes)}.
            </li>
          </ul>
        </div>
      </section>

      <section className={styles.chartPanel} aria-label="Benchmark charts">
        <div className={`${styles.chartCard} ${styles.tabChartCard}`}>
          <Tabs className={styles.chartTabs} groupId="tpch-sf100-chart-tabs" queryString={false}>
            <TabItem value="runtime" label="Per-query runtime" default>
              <p className={styles.chartCaption}>Wall time, measured iteration mean.</p>
              <div className={styles.chart}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={runtimeData} margin={{top: 8, right: 16, left: 0, bottom: 0}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--ifm-color-emphasis-200)" />
                    <XAxis dataKey="query" tick={{fontSize: 12}} />
                    <YAxis tick={{fontSize: 12}} width={58} tickFormatter={formatMs} />
                    <Tooltip
                      contentStyle={tooltipContentStyle}
                      labelStyle={tooltipLabelStyle}
                      itemStyle={tooltipItemStyle}
                      cursor={tooltipCursorStyle}
                      formatter={(value) => formatMs(Number(value))}
                    />
                    <Legend />
                    <Bar dataKey="CPU" fill={colorCpu} />
                    <Bar dataKey="GPU" fill={colorGpu} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabItem>
            <TabItem value="speedup" label="Speedup">
              <p className={styles.chartCaption}>CPU wall time divided by GPU wall time.</p>
              <div className={styles.chart}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={speedupData} margin={{top: 8, right: 16, left: 0, bottom: 0}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--ifm-color-emphasis-200)" />
                    <XAxis dataKey="query" tick={{fontSize: 12}} />
                    <YAxis tick={{fontSize: 12}} width={44} tickFormatter={formatSpeedup} />
                    <Tooltip
                      contentStyle={tooltipContentStyle}
                      labelStyle={tooltipLabelStyle}
                      itemStyle={tooltipItemStyle}
                      cursor={tooltipCursorStyle}
                      formatter={(value) => formatSpeedup(Number(value))}
                    />
                    <Bar dataKey="CPU/GPU" fill={colorSpeedup} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabItem>
            <TabItem value="memory" label="Memory pressure">
              <p className={styles.chartCaption}>
                Largest cuDF allocator peak that one compiled task reported, per query.
              </p>
              <div className={styles.chart}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={memoryData} margin={{top: 8, right: 16, left: 0, bottom: 0}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--ifm-color-emphasis-200)" />
                    <XAxis dataKey="query" tick={{fontSize: 12}} />
                    <YAxis tick={{fontSize: 12}} width={58} tickFormatter={formatGiBShort} />
                    <Tooltip
                      contentStyle={tooltipContentStyle}
                      labelStyle={tooltipLabelStyle}
                      itemStyle={tooltipItemStyle}
                      cursor={tooltipCursorStyle}
                      formatter={(value) => formatGiBShort(Number(value))}
                    />
                    <Bar dataKey="Task memory peak" fill={colorMemory} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabItem>
            <TabItem value="results" label="Per-query results">
              <p className={styles.chartCaption}>{summary.queryCount} TPC-H queries.</p>
              <div className={clsx(styles.tablePanel, 'thin-scrollbar')}>
                <table className={styles.resultsTable}>
                  <thead>
                    <tr>
                      <th>Query</th>
                      <th>CPU</th>
                      <th>GPU</th>
                      <th>Speedup</th>
                      <th>Peak class</th>
                      <th>Peak bytes</th>
                      <th>Row groups</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row) => (
                      <tr key={row.queryId}>
                        <td className={styles.queryCell}>{row.queryId}</td>
                        <td>{formatMs(row.cpuMs)}</td>
                        <td>{formatMs(row.gpuMs)}</td>
                        <td>{formatSpeedup(row.speedup)}</td>
                        <td>{row.peakMemoryOperatorClass}</td>
                        <td>{formatGiB(row.peakMemoryOperatorClassBytes)}</td>
                        <td>{formatNumber(row.rowGroupsSelected)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </TabItem>
          </Tabs>
        </div>
      </section>
    </div>
  );
}

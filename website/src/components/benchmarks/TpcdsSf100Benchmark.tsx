import clsx from 'clsx';
import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import {tpcdsSf100Benchmark, type TpcdsSf100BenchmarkRow} from '@site/src/data/tpcdsSf100Benchmark';
import styles from './TpchSf100Benchmark.module.css';

const bytesPerGiB = 1024 ** 3;
const colorCpu = '#0071C5';
const colorGpu = '#76B900';
const colorSpeedup = '#10b981';
const colorSpeedupSlow = '#C92A2A';
const colorMemory = '#6366f1';
const colorFailed = '#6C757D';

// Q72 CPU takes 71.5s; the remaining queries finish below 35s.
// The axis break covers a range with no measured values.
const RUNTIME_Y_BREAK_START_MS = 35_000;
const RUNTIME_Y_BREAK_END_MS = 60_000;
const RUNTIME_Y_BREAK_WIDTH_MS = 2_500;
const RUNTIME_Y_TAIL_SCALE = 0.4;

function brokenRuntimeMs(valueMs: number): number {
  if (valueMs <= RUNTIME_Y_BREAK_START_MS) {
    return valueMs;
  }
  if (valueMs >= RUNTIME_Y_BREAK_END_MS) {
    return (
      RUNTIME_Y_BREAK_START_MS +
      RUNTIME_Y_BREAK_WIDTH_MS +
      (valueMs - RUNTIME_Y_BREAK_END_MS) * RUNTIME_Y_TAIL_SCALE
    );
  }
  // Values inside the skipped band (unexpected) are compressed into the gap.
  return (
    RUNTIME_Y_BREAK_START_MS +
    ((valueMs - RUNTIME_Y_BREAK_START_MS) / (RUNTIME_Y_BREAK_END_MS - RUNTIME_Y_BREAK_START_MS)) *
      RUNTIME_Y_BREAK_WIDTH_MS
  );
}

function runtimeYTicks(maxRealMs: number): {tickVals: number[]; tickText: string[]} {
  const lowerSeconds = [0, 10, 20, 30];
  const tickVals = lowerSeconds.map((seconds) => brokenRuntimeMs(seconds * 1000));
  const tickText = lowerSeconds.map((seconds) => `${seconds}s`);

  if (maxRealMs > RUNTIME_Y_BREAK_START_MS) {
    const upperSeconds = Math.ceil(maxRealMs / 10_000) * 10;
    tickVals.push(brokenRuntimeMs(RUNTIME_Y_BREAK_END_MS), brokenRuntimeMs(upperSeconds * 1000));
    tickText.push(`${RUNTIME_Y_BREAK_END_MS / 1000}s`, `${upperSeconds}s`);
  }

  return {tickVals, tickText};
}

function formatMs(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return '-';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)}s`;
  }
  return `${value.toFixed(0)}ms`;
}

function formatSeconds(value: number): string {
  return `${(value / 1000).toFixed(1)}s`;
}

function formatSpeedup(value: number | null | undefined, decimals = 1): string {
  if (value == null || Number.isNaN(value)) {
    return '-';
  }
  return `${value.toFixed(decimals)}x`;
}

function formatGiB(value: number): string {
  return `${(value / bytesPerGiB).toFixed(1)} GiB`;
}

function formatGiBShort(value: number): string {
  return `${value.toFixed(1)} GiB`;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

const tooltipContentStyle: React.CSSProperties = {
  background: 'var(--ifm-background-surface-color)',
  border: '1px solid var(--ifm-color-emphasis-300)',
  borderRadius: '6px',
  boxShadow: 'var(--ifm-global-shadow-md)',
  color: 'var(--ifm-font-color-base)',
  fontSize: '0.82rem',
  lineHeight: 1.4,
  padding: '0.5rem 0.75rem',
};

const tooltipLabelStyle: React.CSSProperties = {
  color: 'var(--ifm-heading-color)',
  fontWeight: 600,
  marginBottom: '0.25rem',
};

const tooltipItemStyle: React.CSSProperties = {
  color: 'var(--ifm-font-color-base)',
  padding: 0,
};

const tooltipCursorStyle = {
  fill: 'var(--ifm-color-emphasis-300)',
  fillOpacity: 0.35,
} as const;

export default function TpcdsSf100Benchmark(): React.ReactElement {
  const {dataset, config, rows, summary} = tpcdsSf100Benchmark;
  const typedRows = rows as readonly TpcdsSf100BenchmarkRow[];
  const typedSummary = summary as {
    queryCount: number;
    cpuTotalMs: number;
    gpuSuccessTotalMs: number;
    gpuSuccessCount: number;
    speedup: number | null;
    maxPeakMemoryOperatorClassBytes: number;
    rowsMatchCount: number;
    gpuFasterCount: number;
    gpuSlowerCount: number;
  };
  const typedConfig = config as {
    warmups: number;
    iterations: number;
    nativeProfileName: string;
    kvikioCompatMode: string;
  };
  const typedDataset = dataset as {scaleFactor: string};
  const runtimeData = typedRows.map((row) => {
    const cpuMs = row.cpuMs;
    const gpuMs = row.gpuMs ?? row.gpuWallMs ?? 0;
    return {
      query: row.queryId,
      // Transformed values drive bar height on the broken Y-axis.
      CPU: brokenRuntimeMs(cpuMs),
      GPU: brokenRuntimeMs(gpuMs),
      // Raw values feed the tooltip, which reports the measured wall time.
      cpuRawMs: cpuMs,
      gpuRawMs: gpuMs,
      gpuOutcome: row.gpuOutcome,
    };
  });
  const runtimeMaxRealMs = Math.max(
    0,
    ...runtimeData.flatMap((row) => [row.cpuRawMs, row.gpuRawMs]),
  );
  const runtimeUsesBrokenAxis = runtimeMaxRealMs > RUNTIME_Y_BREAK_START_MS;
  const {tickVals: runtimeTickVals, tickText: runtimeTickText} = runtimeYTicks(runtimeMaxRealMs);
  const runtimeYDomainMax = runtimeUsesBrokenAxis
    ? Math.max(brokenRuntimeMs(runtimeMaxRealMs), ...runtimeTickVals) * 1.06
    : Math.max(runtimeMaxRealMs * 1.08, 1);
  const speedupData = typedRows
    .filter((row) => row.speedup != null)
    .slice()
    .sort((a, b) => (a.speedup ?? 0) - (b.speedup ?? 0))
    .map((row) => ({
      query: row.queryId,
      'CPU/GPU': row.speedup ?? 0,
    }));
  const memoryData = typedRows.map((row) => ({
    query: row.queryId,
    'Task memory peak': row.peakMemoryOperatorClassBytes / bytesPerGiB,
  }));

  return (
    <div className={styles.benchmark}>
      <section className={styles.header}>
        <div>
          <ul className={styles.benchmarkFacts}>
            <li>
              <strong>Result:</strong> CPU {formatSeconds(typedSummary.cpuTotalMs)}, GPU{' '}
              {formatSeconds(typedSummary.gpuSuccessTotalMs)} across{' '}
              {typedSummary.gpuSuccessCount}/{typedSummary.queryCount} successful native
              queries, CPU / GPU {formatSpeedup(typedSummary.speedup, 2)}, largest task memory peak{' '}
              {formatGiB(typedSummary.maxPeakMemoryOperatorClassBytes)}. Row counts matched on{' '}
              {typedSummary.rowsMatchCount}/{typedSummary.queryCount} queries.
            </li>
          </ul>
        </div>
      </section>

      <section className={styles.chartPanel} aria-label="TPC-DS benchmark charts">
        <div className={`${styles.chartCard} ${styles.tabChartCard}`}>
          <Tabs className={styles.chartTabs} groupId="tpcds-sf100-chart-tabs" queryString={false}>
            <TabItem value="runtime" label="Per-query runtime" default>
              <p className={styles.chartCaption}>
                Wall time for the measured iteration.
                {runtimeUsesBrokenAxis
                  ? ` The Y-axis skips ${RUNTIME_Y_BREAK_START_MS / 1000}–${RUNTIME_Y_BREAK_END_MS / 1000}s.`
                  : ''}
              </p>
              <p className={styles.chartScrollHint}>
                Scroll horizontally to inspect all {typedRows.length} queries.
              </p>
              <div className={clsx(styles.chartScroll, 'thin-scrollbar')} role="region" aria-label="Per-query runtime chart, scroll horizontally">
                <div
                  className={`${styles.chart} ${styles.chartScrollTrack}`}
                  style={{width: Math.max(720, typedRows.length * 28)}}
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={runtimeData}
                      margin={{top: 18, right: 16, left: 0, bottom: 24}}
                      barCategoryGap="18%"
                      barGap={2}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--ifm-color-emphasis-200)" />
                      <XAxis
                        dataKey="query"
                        tick={{fontSize: 11}}
                        interval={0}
                        angle={-60}
                        textAnchor="end"
                        height={78}
                        minTickGap={0}
                      />
                      <YAxis
                        tick={{fontSize: 12}}
                        width={58}
                        domain={[0, runtimeYDomainMax]}
                        ticks={runtimeUsesBrokenAxis ? runtimeTickVals : undefined}
                        tickFormatter={(value) => {
                          if (!runtimeUsesBrokenAxis) {
                            return formatMs(Number(value));
                          }
                          const index = runtimeTickVals.indexOf(Number(value));
                          return index >= 0 ? runtimeTickText[index] : formatMs(Number(value));
                        }}
                        allowDataOverflow
                      />
                      {runtimeUsesBrokenAxis ? (
                        <ReferenceArea
                          y1={RUNTIME_Y_BREAK_START_MS}
                          y2={RUNTIME_Y_BREAK_START_MS + RUNTIME_Y_BREAK_WIDTH_MS}
                          fill="var(--ifm-color-emphasis-200)"
                          fillOpacity={0.55}
                          strokeOpacity={0}
                          ifOverflow="extendDomain"
                          label={{
                            value: `${RUNTIME_Y_BREAK_START_MS / 1000}–${RUNTIME_Y_BREAK_END_MS / 1000}s skipped`,
                            position: 'insideTopRight',
                            fill: 'var(--ifm-font-color-secondary)',
                            fontSize: 11,
                          }}
                        />
                      ) : null}
                      <Tooltip
                        contentStyle={tooltipContentStyle}
                        labelStyle={tooltipLabelStyle}
                        itemStyle={tooltipItemStyle}
                        cursor={tooltipCursorStyle}
                        formatter={(value, name, item) => {
                          const payload = item?.payload as
                            | {cpuRawMs?: number; gpuRawMs?: number}
                            | undefined;
                          if (name === 'CPU' && payload?.cpuRawMs != null) {
                            return [formatMs(payload.cpuRawMs), 'CPU'];
                          }
                          if (name === 'GPU' && payload?.gpuRawMs != null) {
                            return [formatMs(payload.gpuRawMs), 'GPU'];
                          }
                          return [formatMs(Number(value)), String(name)];
                        }}
                      />
                      <Legend />
                      <Bar dataKey="CPU" fill={colorCpu} />
                      <Bar dataKey="GPU" fill={colorGpu} name="GPU">
                        {runtimeData.map((entry) => (
                          <Cell
                            key={entry.query}
                            fill={entry.gpuOutcome === 'succeeded' ? colorGpu : colorFailed}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </TabItem>
            <TabItem value="speedup" label="Speedup">
              <p className={styles.chartCaption}>
                CPU wall / successful GPU wall, sorted ascending. Green is GPU
                faster; red is GPU slower.
              </p>
              <div className={styles.chart} style={{height: 420}}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={speedupData} margin={{top: 8, right: 16, left: 0, bottom: 0}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--ifm-color-emphasis-200)" />
                    <XAxis dataKey="query" hide tick={{fontSize: 10}} />
                    <YAxis tick={{fontSize: 12}} width={44} tickFormatter={formatSpeedup} />
                    <Tooltip
                      contentStyle={tooltipContentStyle}
                      labelStyle={tooltipLabelStyle}
                      itemStyle={tooltipItemStyle}
                      cursor={tooltipCursorStyle}
                      formatter={(value) => formatSpeedup(Number(value))}
                    />
                    <Bar dataKey="CPU/GPU">
                      {speedupData.map((entry) => (
                        <Cell
                          key={entry.query}
                          fill={(entry['CPU/GPU'] as number) >= 1 ? colorSpeedup : colorSpeedupSlow}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabItem>
            <TabItem value="memory" label="Memory pressure">
              <p className={styles.chartCaption}>
                Largest cuDF allocator peak that one compiled task reported, per query.
              </p>
              <div className={styles.chart} style={{height: 420}}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={memoryData} margin={{top: 8, right: 16, left: 0, bottom: 0}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--ifm-color-emphasis-200)" />
                    <XAxis dataKey="query" hide />
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
              <p className={styles.chartCaption}>
                {typedSummary.queryCount} TPC-DS queries. GPU faster on{' '}
                {typedSummary.gpuFasterCount}; slower on {typedSummary.gpuSlowerCount}.
              </p>
              <div className={clsx(styles.tablePanel, 'thin-scrollbar')}>
                <table className={styles.resultsTable}>
                  <thead>
                    <tr>
                      <th>Query</th>
                      <th>CPU</th>
                      <th>GPU</th>
                      <th>Speedup</th>
                      <th>GPU outcome</th>
                      <th>Rows</th>
                      <th>Runtime class</th>
                      <th>Task memory peak</th>
                    </tr>
                  </thead>
                  <tbody>
                    {typedRows.map((row) => (
                      <tr key={row.queryId}>
                        <td className={styles.queryCell}>{row.queryId}</td>
                        <td>{formatMs(row.cpuMs)}</td>
                        <td>{formatMs(row.gpuMs ?? row.gpuWallMs)}</td>
                        <td>{formatSpeedup(row.speedup)}</td>
                        <td>{row.gpuOutcome}</td>
                        <td>{row.rowsMatch ? 'match' : 'mismatch'}</td>
                        <td>{row.runtimeOperatorClass || '-'}</td>
                        <td>{formatGiB(row.peakMemoryOperatorClassBytes)}</td>
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

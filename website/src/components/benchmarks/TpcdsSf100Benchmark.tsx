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
import {
  bytesPerGiB,
  formatGiB,
  formatGiBShort,
  tooltipContentStyle,
  tooltipCursorStyle,
  tooltipItemStyle,
  tooltipLabelStyle,
} from './benchmarkChartUtils';
import styles from './TpchSf100Benchmark.module.css';

const colorCpu = '#0071C5';
const colorGpu = '#76B900';
const colorSpeedup = '#10b981';
const colorSpeedupSlow = '#C92A2A';
const colorMemory = '#6366f1';
const colorFailed = '#6C757D';

type RuntimeAxisBreak = {startMs: number; endMs: number; widthMs: number};

const RUNTIME_Y_TAIL_SCALE = 0.4;

// One outlier can run several times longer than every other bar. The
// axis skips the empty band between the second-slowest bar and the outlier so the other
// bars keep a readable height; without such an outlier the axis stays linear.
function runtimeAxisBreak(valuesMs: number[]): RuntimeAxisBreak | null {
  const sorted = [...valuesMs].sort((a, b) => b - a);
  if (sorted.length < 2 || sorted[0] < 2 * sorted[1]) {
    return null;
  }
  const startMs = Math.ceil(sorted[1] / 5_000) * 5_000;
  const endMs = Math.floor(sorted[0] / 10_000) * 10_000;
  if (endMs - startMs < 10_000) {
    return null;
  }
  return {startMs, endMs, widthMs: Math.max(1_000, startMs * 0.08)};
}

function brokenRuntimeMs(valueMs: number, axisBreak: RuntimeAxisBreak | null): number {
  if (axisBreak == null || valueMs <= axisBreak.startMs) {
    return valueMs;
  }
  if (valueMs >= axisBreak.endMs) {
    return axisBreak.startMs + axisBreak.widthMs + (valueMs - axisBreak.endMs) * RUNTIME_Y_TAIL_SCALE;
  }
  return (
    axisBreak.startMs +
    ((valueMs - axisBreak.startMs) / (axisBreak.endMs - axisBreak.startMs)) * axisBreak.widthMs
  );
}

function runtimeYTicks(
  maxRealMs: number,
  axisBreak: RuntimeAxisBreak,
): {tickVals: number[]; tickText: string[]} {
  const stepSeconds = axisBreak.startMs <= 20_000 ? 5 : 10;
  const lowerSeconds: number[] = [];
  for (let seconds = 0; seconds * 1000 <= axisBreak.startMs; seconds += stepSeconds) {
    lowerSeconds.push(seconds);
  }
  const upperSeconds = Math.ceil(maxRealMs / 10_000) * 10;
  const tickSeconds = [...lowerSeconds, axisBreak.endMs / 1000, upperSeconds];
  return {
    tickVals: tickSeconds.map((seconds) => brokenRuntimeMs(seconds * 1000, axisBreak)),
    tickText: tickSeconds.map((seconds) => `${seconds}s`),
  };
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
  const runtimeAxis = runtimeAxisBreak(
    typedRows.flatMap((row) => [row.cpuMs, row.gpuMs ?? row.gpuWallMs ?? 0]),
  );
  const runtimeData = typedRows.map((row) => {
    const cpuMs = row.cpuMs;
    const gpuMs = row.gpuMs ?? row.gpuWallMs ?? 0;
    return {
      query: row.queryId,
      // Transformed values drive bar height on the broken Y-axis.
      CPU: brokenRuntimeMs(cpuMs, runtimeAxis),
      GPU: brokenRuntimeMs(gpuMs, runtimeAxis),
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
  const {tickVals: runtimeTickVals, tickText: runtimeTickText} = runtimeAxis
    ? runtimeYTicks(runtimeMaxRealMs, runtimeAxis)
    : {tickVals: [], tickText: []};
  const runtimeYDomainMax = runtimeAxis
    ? Math.max(brokenRuntimeMs(runtimeMaxRealMs, runtimeAxis), ...runtimeTickVals) * 1.06
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
              {formatGiB(typedSummary.maxPeakMemoryOperatorClassBytes)}.
            </li>
          </ul>
        </div>
      </section>

      <section className={styles.chartPanel} aria-label="TPC-DS benchmark charts">
        <div className={`${styles.chartCard} ${styles.tabChartCard}`}>
          <Tabs className={styles.chartTabs} groupId="tpcds-sf100-chart-tabs" queryString={false}>
            <TabItem value="runtime" label="Per-query runtime" default>
              <p className={styles.chartCaption}>
                Wall time of the single measured iteration after one warm-up. Parquet in the OS page cache; no GPU resident cache.
                {runtimeAxis
                  ? ` The Y-axis skips ${runtimeAxis.startMs / 1000}–${runtimeAxis.endMs / 1000}s.`
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
                        ticks={runtimeAxis ? runtimeTickVals : undefined}
                        tickFormatter={(value) => {
                          if (!runtimeAxis) {
                            return formatMs(Number(value));
                          }
                          const index = runtimeTickVals.indexOf(Number(value));
                          return index >= 0 ? runtimeTickText[index] : formatMs(Number(value));
                        }}
                        allowDataOverflow
                      />
                      {runtimeAxis ? (
                        <ReferenceArea
                          y1={runtimeAxis.startMs}
                          y2={runtimeAxis.startMs + runtimeAxis.widthMs}
                          fill="var(--ifm-color-emphasis-200)"
                          fillOpacity={0.55}
                          strokeOpacity={0}
                          ifOverflow="extendDomain"
                          label={{
                            value: `${runtimeAxis.startMs / 1000}–${runtimeAxis.endMs / 1000}s skipped`,
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
                      <th>Result values</th>
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
                        <td title={row.parityDetail ?? undefined}>
                          {row.parityStatus} · {row.comparisonMode}
                        </td>
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

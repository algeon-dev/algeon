import clsx from 'clsx';
import React from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import {serverLoadBenchmark, type Spread} from '@site/src/data/serverLoadBenchmark';
import {
  formatGiB,
  formatNumber,
  tooltipContentStyle,
  tooltipItemStyle,
  tooltipLabelStyle,
} from './benchmarkChartUtils';
import styles from './TpchSf100Benchmark.module.css';

type Series = (typeof serverLoadBenchmark.series)[number];
type Metric = 'completedQps' | 'p95Seconds';

const colorCpu = '#0071C5';
// Darker to lighter green as max_active_attempts rises.
const gpuColors: Record<number, string> = {1: '#3F6212', 2: '#76B900', 3: '#0D9488'};

function seriesLabel(series: Series): string {
  if (series.executionMode === 'functions_only') {
    return 'CPU path (functions_only)';
  }
  return `GPU, A=${series.maxActiveAttempts}${series.admissionDefaults ? ' (default)' : ''}`;
}

function seriesColor(series: Series): string {
  return series.maxActiveAttempts === null ? colorCpu : gpuColors[series.maxActiveAttempts] ?? '#76B900';
}

function formatSpread(spread: Spread, digits: number, unit: string): string {
  const [median, min, max] = [spread.median, spread.min, spread.max].map((value) => value.toFixed(digits));
  if (min === median && max === median) {
    return `${median}${unit}`;
  }
  return `${median}${unit} (${min}-${max})`;
}

function MetricChart({metric, unit, digits, step}: {metric: Metric; unit: string; digits: number; step: number}) {
  const {series, config} = serverLoadBenchmark;
  const labels = series.map(seriesLabel);
  const data = config.streams.map((streams) => {
    const row: Record<string, number | string> = {streams: String(streams)};
    series.forEach((entry, index) => {
      const point = entry.points.find((p) => p.streams === streams);
      if (point) {
        row[labels[index]] = point[metric].median;
      }
    });
    return row;
  });
  const spreadFor = (label: string, streams: string): Spread | undefined => {
    const entry = series[labels.indexOf(label)];
    return entry?.points.find((p) => String(p.streams) === streams)?.[metric];
  };
  const top = Math.ceil(Math.max(...series.flatMap((entry) => entry.points.map((p) => p[metric].median))) / step) * step;
  const ticks = Array.from({length: Math.round(top / step) + 1}, (_, index) => index * step);

  return (
    <div className={styles.chart}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{top: 8, right: 16, left: 0, bottom: 16}}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--ifm-color-emphasis-200)" />
          <XAxis
            dataKey="streams"
            tick={{fontSize: 12}}
            label={{value: 'Concurrent streams', position: 'insideBottom', offset: -8, fontSize: 12}}
          />
          <YAxis
            tick={{fontSize: 12}}
            width={52}
            domain={[0, top]}
            ticks={ticks}
            tickFormatter={(value: number) => `${value}${unit}`}
          />
          <Tooltip
            contentStyle={tooltipContentStyle}
            labelStyle={tooltipLabelStyle}
            itemStyle={tooltipItemStyle}
            labelFormatter={(streams) => `${streams} streams, median (range of ${config.repeats} runs)`}
            formatter={(value, name, item) => {
              const spread = spreadFor(String(name), String(item.payload.streams));
              return spread ? formatSpread(spread, digits, unit) : String(value);
            }}
          />
          <Legend verticalAlign="top" />
          {series.map((entry, index) => (
            <Line
              key={entry.key}
              dataKey={labels[index]}
              stroke={seriesColor(entry)}
              strokeWidth={entry.admissionDefaults ? 3 : 2}
              strokeDasharray={entry.executionMode === 'native_required' && !entry.admissionDefaults ? '6 4' : undefined}
              dot={{r: 3}}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function ServerLoadBenchmark(): React.ReactElement {
  const {series, config} = serverLoadBenchmark;
  return (
    <div className={styles.benchmark}>
      <section className={styles.chartPanel} aria-label="Server load charts">
        <div className={`${styles.chartCard} ${styles.tabChartCard}`}>
          <Tabs className={styles.chartTabs} groupId="server-load-chart-tabs" queryString={false}>
            <TabItem value="throughput" label="Throughput" default>
              <p className={styles.chartCaption}>
                Completed queries per second over the measured pass, median of {config.repeats} runs.
              </p>
              <MetricChart metric="completedQps" unit="" digits={2} step={0.5} />
            </TabItem>
            <TabItem value="p95" label="p95 latency">
              <p className={styles.chartCaption}>
                95th percentile query latency in seconds, from request to the last result batch.
              </p>
              <MetricChart metric="p95Seconds" unit="s" digits={1} step={20} />
            </TabItem>
            <TabItem value="table" label="All values">
              <p className={styles.chartCaption}>
                Median of {config.repeats} runs with the range in parentheses.
              </p>
              <div className={clsx(styles.tablePanel, 'thin-scrollbar')}>
                <table className={styles.resultsTable}>
                  <thead>
                    <tr>
                      <th>Configuration</th>
                      <th>Query cap</th>
                      <th>Streams</th>
                      <th>Queries/hour</th>
                      <th>QPS</th>
                      <th>p50</th>
                      <th>p95</th>
                    </tr>
                  </thead>
                  <tbody>
                    {series.flatMap((entry) =>
                      entry.points.map((point) => (
                        <tr key={`${entry.key}-${point.streams}`}>
                          <td className={styles.queryCell}>{seriesLabel(entry)}</td>
                          <td>{entry.queryLimitBytes === null ? 'n/a' : formatGiB(entry.queryLimitBytes)}</td>
                          <td>{point.streams}</td>
                          <td>{formatNumber(Math.round(point.completedQps.median * 3600))}</td>
                          <td>{formatSpread(point.completedQps, 2, '')}</td>
                          <td>{formatSpread(point.p50Seconds, 1, 's')}</td>
                          <td>{formatSpread(point.p95Seconds, 1, 's')}</td>
                        </tr>
                      )),
                    )}
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

function formatRejection(spread: Spread | null): string {
  if (spread === null) {
    return 'none';
  }
  return spread.median < 1 ? `${Math.round(spread.median * 1000)}ms` : `${spread.median.toFixed(2)}s`;
}

export function ServerLoadOverloadTable(): React.ReactElement {
  const {levels} = serverLoadBenchmark.overload;
  return (
    <div className={clsx(styles.tablePanel, 'thin-scrollbar')}>
      <table className={clsx(styles.resultsTable, styles.wrapHeaders)}>
        <thead>
          <tr>
            <th>Offered rate</th>
            <th>Offered QPS</th>
            <th>Completed QPS</th>
            <th>Rejected</th>
            <th>Rejected when queue full, p50</th>
            <th>Rejected after queue wait, p50</th>
            <th>Completed p95</th>
          </tr>
        </thead>
        <tbody>
          {levels.map((level) => (
            <tr key={level.level}>
              <td className={styles.queryCell}>{level.factor.toFixed(1)} × C</td>
              <td>{level.offeredQps.median.toFixed(2)}</td>
              <td>{formatSpread(level.completedQps, 2, '')}</td>
              <td>{(level.rejectedFraction.median * 100).toFixed(1)}%</td>
              <td>{formatRejection(level.rejectionP50Seconds.admissionQueueFull)}</td>
              <td>{formatRejection(level.rejectionP50Seconds.queueWaitElapsed)}</td>
              <td>{formatSpread(level.completedP95Seconds, 1, 's')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

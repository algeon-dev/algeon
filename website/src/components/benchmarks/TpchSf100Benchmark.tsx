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
import styles from './TpchSf100Benchmark.module.css';

const bytesPerGiB = 1024 ** 3;

// Palette mirrors the previous Tremor chart colors.
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

function formatGiB(value: number): string {
  return `${(value / bytesPerGiB).toFixed(1)} GiB`;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US').format(value);
}

function formatGiBShort(value: number): string {
  return `${value.toFixed(1)} GiB`;
}

// recharts renders its default tooltip with hard-coded white styling, which
// clashes with the dark theme. Style it through Infima CSS variables so it
// tracks the active theme automatically.
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
    'Reservation peak': row.reservationLivePeakBytes / bytesPerGiB,
  }));

  return (
    <div className={styles.benchmark}>
      <section className={styles.header}>
        <div>
          <p className={styles.eyebrow}>AWS EC2 g7e.4xlarge local NVMe</p>
          <ul className={styles.benchmarkFacts}>
            <li>
              <strong>Environment:</strong> AWS EC2 g7e.4xlarge, 16 vCPUs,
              128 GiB system memory, local NVMe fixture storage, RTX PRO 6000
              GPU (96 GiB).
            </li>
            <li>
              <strong>Run:</strong> release build, {dataset.scaleFactor.toUpperCase()},
              {config.warmups} warmup / {config.iterations} measured iteration,
              source cache <code>{config.nativeSourceCachePolicy}</code>.
            </li>
            <li>
              <strong>Result:</strong> CPU {formatSeconds(summary.cpuTotalMs)},
              GPU {formatSeconds(summary.gpuTotalMs)}, CPU / GPU{' '}
              {formatSpeedup(summary.speedup)}, peak reservation{' '}
              {formatGiB(summary.maxReservationLivePeakBytes)}.
            </li>
            <li>
              <strong>KvikIO:</strong> local cuDF/libcudf file reads with cache
              disabled; this run forced KvikIO compatibility I/O with{' '}
              <code>KVIKIO_COMPAT_MODE={config.kvikioCompatMode}</code>. On this
              g7e.4xlarge host, <code>OFF</code> enters the cuFile path, but
              cuFile compatibility mode can still fall back; it does not prove
              native GDS.
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
                Query-level reservation ledger high-water marks used by bounded admission.
              </p>
              <ul className={styles.chartNotes}>
                <li>
                  <strong>Reservation peak</strong> is the whole-query live reserved
                  byte peak, not an operator-class diagnostic.
                </li>
                <li>
                  <code>nvidia-smi</code> can be higher or lower because it samples
                  process/GPU memory externally and includes CUDA/libcudf allocator
                  state outside the query reservation ledger.
                </li>
              </ul>
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
                    <Bar dataKey="Reservation peak" fill={colorMemory} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabItem>
            <TabItem value="results" label="Per-query results">
              <p className={styles.chartCaption}>{summary.queryCount} TPC-H queries.</p>
              <div className={styles.tablePanel}>
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

import React from 'react';
import {
  BarChart,
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from '@tremor/react';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import {tpchSf100Benchmark} from '@site/src/data/tpchSf100Benchmark';
import styles from './TpchSf100Benchmark.module.css';

const bytesPerGiB = 1024 ** 3;

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
        <Card className={`${styles.chartCard} ${styles.tabChartCard}`}>
          <Tabs className={styles.chartTabs} groupId="tpch-sf100-chart-tabs" queryString={false}>
            <TabItem value="runtime" label="Per-query runtime" default>
              <p className={styles.chartCaption}>Wall time, measured iteration mean.</p>
              <BarChart
                className={styles.chart}
                data={runtimeData}
                index="query"
                categories={['CPU', 'GPU']}
                colors={['cpu-blue', 'gpu-green']}
                valueFormatter={formatMs}
                yAxisWidth={58}
                showLegend
                showGridLines
              />
            </TabItem>
            <TabItem value="speedup" label="Speedup">
              <p className={styles.chartCaption}>CPU wall time divided by GPU wall time.</p>
              <BarChart
                className={styles.chart}
                data={speedupData}
                index="query"
                categories={['CPU/GPU']}
                colors={['emerald']}
                valueFormatter={formatSpeedup}
                yAxisWidth={44}
                showLegend={false}
                showGridLines
              />
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
              <BarChart
                className={styles.chart}
                data={memoryData}
                index="query"
                categories={['Reservation peak']}
                colors={['indigo']}
                valueFormatter={(value) => `${value.toFixed(1)} GiB`}
                yAxisWidth={58}
                showLegend={false}
                showGridLines
              />
            </TabItem>
            <TabItem value="results" label="Per-query results">
              <p className={styles.chartCaption}>{summary.queryCount} TPC-H queries.</p>
              <div className={styles.tablePanel}>
                <Table className={styles.resultsTable}>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Query</TableHeaderCell>
                      <TableHeaderCell>CPU</TableHeaderCell>
                      <TableHeaderCell>GPU</TableHeaderCell>
                      <TableHeaderCell>Speedup</TableHeaderCell>
                      <TableHeaderCell>Peak class</TableHeaderCell>
                      <TableHeaderCell>Peak bytes</TableHeaderCell>
                      <TableHeaderCell>Row groups</TableHeaderCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {rows.map((row) => (
                      <TableRow key={row.queryId}>
                        <TableCell className={styles.queryCell}>{row.queryId}</TableCell>
                        <TableCell>{formatMs(row.cpuMs)}</TableCell>
                        <TableCell>{formatMs(row.gpuMs)}</TableCell>
                        <TableCell>{formatSpeedup(row.speedup)}</TableCell>
                        <TableCell>{row.peakMemoryOperatorClass}</TableCell>
                        <TableCell>{formatGiB(row.peakMemoryOperatorClassBytes)}</TableCell>
                        <TableCell>{formatNumber(row.rowGroupsSelected)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </TabItem>
          </Tabs>
        </Card>
      </section>
    </div>
  );
}

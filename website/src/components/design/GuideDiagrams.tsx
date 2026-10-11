import React, {useState} from 'react';
import {BackedLabel, Box, COLOR, Figure, TONE} from './Diagram';
import type {DiagramNode, Tone} from './Diagram';
import {SANS, sketchLine, sketchRoundRect} from './sketch';
import styles from './styles.module.css';

type Point = [number, number];

function Card({x, y, w, h = 72, name, note, tone = 'plain', kind, dashed = false}: DiagramNode & {
  x: number;
  y: number;
  w: number;
  h?: number;
  dashed?: boolean;
}) {
  return <Box node={{name, note, tone, kind}} x={x} y={y} w={w} h={h} borderDashed={dashed} />;
}

function Arrow({id, points, dashed = false, arrowhead = true}: {
  id: string; points: Point[]; dashed?: boolean; arrowhead?: boolean;
}) {
  const end = points[points.length - 1];
  const previous = points[points.length - 2];
  const angle = Math.atan2(end[1] - previous[1], end[0] - previous[0]) * 180 / Math.PI;
  const path = [`M ${points[0][0]} ${points[0][1]}`];
  for (let i = 1; i < points.length - 1; i++) {
    const [x, y] = points[i];
    const [px, py] = points[i - 1];
    const [nx, ny] = points[i + 1];
    const incoming = Math.hypot(x - px, y - py);
    const outgoing = Math.hypot(nx - x, ny - y);
    const radius = Math.min(12, incoming / 2, outgoing / 2);
    path.push(
      `L ${x + (px - x) * radius / incoming} ${y + (py - y) * radius / incoming}`,
      `Q ${x} ${y} ${x + (nx - x) * radius / outgoing} ${y + (ny - y) * radius / outgoing}`,
    );
  }
  path.push(`L ${end[0]} ${end[1]}`);
  return <g data-connector={id} fill="none" stroke={COLOR.fg} strokeWidth={1.4}
    strokeLinecap="round" strokeLinejoin="round">
    <path d={path.join(' ')} strokeDasharray={dashed ? '5 5' : undefined} />
    {arrowhead && <path d="M -7 -4 L 0 0 L -7 4" transform={`translate(${end[0]} ${end[1]}) rotate(${angle})`} />}
  </g>;
}

function Label({x, y, children, textAnchor = 'start'}: {
  x: number; y: number; children: React.ReactNode; textAnchor?: 'start' | 'middle' | 'end';
}) {
  return <text x={x} y={y} textAnchor={textAnchor}
    fill={COLOR.muted} fontFamily={SANS} fontSize={13}>{children}</text>;
}

function Region({x, y, w, h, label, tone = 'plain', labelOffset = 14}: {
  x: number; y: number; w: number; h: number; label?: string; tone?: Tone; labelOffset?: number;
}) {
  return <g>
    <rect x={x} y={y} width={w} height={h} rx={16} fill={TONE[tone]} opacity={0.05} />
    {sketchRoundRect(`guide-region:${label ?? `${x}:${y}`}`, x, y, w, h, 16,
      {stroke: TONE[tone], strokeWidth: 1, dashed: true})}
    {label ? <Label x={x + labelOffset} y={y + 22}>{label}</Label> : null}
  </g>;
}

function GuideFigure({height, width = 720, minWidth = 600, label, caption, children}: {
  height: number; width?: number; minWidth?: number;
  label: string; caption: React.ReactNode; children: React.ReactNode;
}) {
  return <Figure svgW={width} svgH={height} minW={minWidth} ariaLabel={label} caption={caption} center>
    {children}
  </Figure>;
}

export function QueryArchitecture() {
  return <GuideFigure height={720}
    label="Query flow and module boundaries. DataFusion plans SQL on the CPU; algeon-datafusion lowers the physical plan and invokes query-engine. Query-planning owns IR, capability proof and the compiled schedule. Query-runtime owns the attempt, device grant and cache. Query-engine submits GPU operations through RAPIDS, reading Parquet from local NVMe or S3 and returning host Arrow batches through the adapter and DataFusion. Dashed arrows show direct imports: the adapter imports DataFusion and all three query crates; the engine imports planning and runtime; runtime imports planning."
    caption="Follow the solid arrows from SQL to the result. Dashed arrows point to imported crates. The native query crates orchestrate GPU work from the CPU; RAPIDS operations run on the device.">
    <Arrow id="architecture:flow-legend" points={[[8, 14], [58, 14]]} />
    <Label x={70} y={18}>Execution / data flow</Label>
    <Arrow id="architecture:import-legend" points={[[326, 14], [376, 14]]} dashed />
    <Label x={388} y={18}>Imports from</Label>
    <Region x={24} y={194} w={688} h={266} label="DataFusion-free query crates" tone="engine" labelOffset={284} />
    <g opacity={0.6}>
      <Arrow id="architecture:adapter-datafusion" points={[[386, 114], [300, 114]]} dashed />
      <Arrow id="architecture:adapter-planning" points={[[464, 146], [464, 184], [175, 184], [175, 232]]} dashed />
      <Arrow id="architecture:adapter-runtime" points={[[424, 146], [424, 164], [8, 164], [8, 408], [50, 408]]} dashed />
      <Arrow id="architecture:adapter-engine" points={[[664, 146], [664, 232]]} dashed />
      <Arrow id="architecture:engine-planning" points={[[452, 280], [300, 280]]} dashed />
      <Arrow id="architecture:engine-runtime" points={[[518, 328], [518, 408], [300, 408]]} dashed />
      <Arrow id="architecture:runtime-planning" points={[[175, 362], [175, 328]]} dashed />
    </g>
    <Arrow id="architecture:physical-plan" points={[[300, 82], [386, 82]]} />
    <Label x={343} y={74} textAnchor="middle">plan</Label>
    <Arrow id="architecture:execute" points={[[577, 146], [577, 232]]} />
    <Label x={585} y={184}>execute</Label>
    <Arrow id="architecture:submit" points={[[577, 328], [577, 546]]} />
    <Label x={585} y={511}>GPU submissions</Label>
    <Arrow id="architecture:read" points={[[300, 546], [343, 546], [343, 578], [386, 578]]} />
    <Label x={322} y={538} textAnchor="middle">read</Label>
    <Arrow id="architecture:result" points={[[386, 630], [343, 630], [343, 662], [300, 662]]} />
    <Label x={343} y={682} textAnchor="middle">device → host</Label>
    <Card x={40} y={56} w={254} h={84} name="DataFusion"
      note={['SQL / DataFrame', 'Physical plan on CPU']} tone="plan" />
    <Card x={392} y={56} w={304} h={84} name="algeon-datafusion"
      note={['Lower the physical plan', 'Wrap execution and Arrow results']} tone="engine" />
    <Card x={56} y={238} w={238} h={84} name="query-planning"
      note={['IR and capability proof', 'Compiled GPU execution schedule']} tone="plan" />
    <Card x={458} y={238} w={238} h={84} name="query-engine"
      note={['Execute the schedule', 'Use planning + runtime contracts']} tone="bridge" />
    <Card x={56} y={368} w={238} h={80} name="query-runtime"
      note={['One device + grant', 'Attempt lifetime and cache']} tone="engine" />
    <Card x={40} y={504} w={254} h={84} name="Parquet storage"
      note={['Local NVMe or S3', 'Selected column pages']} tone="lake" />
    <Card x={392} y={552} w={304} h={104} name="GPU operations" tone="gpu"
      note={['cuDF · cuGraph · cuVS · cuML', 'Decode → filter / join / aggregate']} />
    <Card x={40} y={620} w={254} h={84} name="Arrow result" tone="bridge"
      note="Via adapter → DataFusion → caller" />
  </GuideFigure>;
}

export function PlanTranslation() {
  const [selected, select] = useState<'source' | 'filter' | 'aggregate' | 'partitions' | 'result'>('aggregate');
  const details = {
    source: {
      title: 'A Parquet source becomes a source_read task',
      body: <><code>DataSourceExec</code> supplies file groups, the scan schema and projected columns.
        Lowering constructs a <code>LocalParquetSource</code> with the source-read contract.
        The <code>source_read</code> task produces GPU columns. Its source-chunk barrier
        separates planned input chunks; the CPU partition count does not set the number of chunks.</>,
      observe: <><code>read_columns</code>, <code>row_groups_selected</code> and{' '}
        <code>source_chunks_committed</code> describe different units: columns, row groups and native chunks.</>,
    },
    filter: {
      title: 'The row predicate remains part of the native plan',
      body: <>The <code>status = 'paid'</code> expression lowers to a native <code>Filter</code>,
        represented by a <code>unary</code> task. Parquet statistics may exclude row groups before
        decoding. Remaining rows still need the predicate unless the source reader already applies
        it under a compatible contract. The diagram shows the separate-filter form.</>,
      observe: <><code>plan_task_mapping_rows</code> connects the native Filter node to its task.
        Source pruning counts and post-decode filtering counts describe different reductions.</>,
    },
    aggregate: {
      title: 'Partial + final aggregates lower to one grouped SUM',
      body: <>For a compatible partial/final pair, lowering recovers <code>SUM(amount)</code> and
        the <code>customer_id</code> grouping key as one native <code>Aggregate</code>.
        Its compiled task has kind <code>aggregate</code>. Chunk execution may produce partial
        states and merge them before downstream output. Those runtime partials do not add a
        separate compiled task for each chunk.</>,
      observe: <><code>partial_aggregate_count</code>, <code>partial_aggregate_input_rows</code> and{' '}
        <code>final_merge_attempts</code> explain chunk execution. Use <code>task_runtime_rows</code>{' '}
        for the compiled task's observations.</>,
    },
    partitions: {
      title: 'Hash partitioning is a DataFusion input contract',
      body: <><code>RepartitionExec</code> routes equal keys to the same final aggregate partition.
        After the complete aggregate subtree is replaced, native grouping and final merging preserve
        that SQL result. There is no corresponding GPU hash-repartition task in this example.
        The replacement exposes one bounded output partition. A retained CPU parent must accept
        its distribution and ordering.</>,
      observe: <>Inspect <code>post_native_plan_root</code> and <code>host_datafusion_boundary_count</code>{' '}
        to check the completed plan. A selected candidate alone does not establish whole-query GPU coverage.</>,
    },
    result: {
      title: 'The result crosses one host Arrow boundary',
      body: <>A native <code>Projection</code> preserves the requested field names and order, including{' '}
        <code>total</code>. The adapter uses a <code>DeferredHostArrow</code> sink: the compiled
        schedule produces a GPU-resident result, and <code>AlgeonNativeExec</code> materializes it
        incrementally as host Arrow batches. This wrapper step is outside the compiled GPU tasks.
        The admission wrapper associates execution with the query attempt.</>,
      observe: <>Compare <code>output_rows</code> with <code>device_to_host_materialized_rows</code>.
        Output bytes describe result batches; they do not measure peak device allocation.</>,
    },
  };
  const detail = details[selected];
  function node(id: typeof selected, x: number, y: number, name: string, note: string | string[],
    h = 66, tone: Tone = x < 360 ? 'plan' : 'gpu') {
    return <Box x={x} y={y} w={292} h={h}
      node={{name, note, tone}}
      selected={selected === id} onSelect={() => select(id)} />;
  }
  return <Figure svgW={720} svgH={594} minW={650} center interactive
    ariaLabel="Interactive grouped-sum lowering example. Data flows upward from Parquet. Select a DataFusion operator or native task to inspect its mapping and related observation fields."
    caption="Select a node to inspect its mapping. Data flows upward; the dashed branches merge into one arrow to the native aggregate. The top-right result is materialized by the wrapper after the GPU schedule. Batching and scan-repartition wrappers are omitted."
    panel={<div className={styles.panel} aria-live="polite">
      <div className={styles.panelHead}><span className={styles.panelName}>{detail.title}</span></div>
      <div className={styles.panelBody}>
        <p>{detail.body}</p>
        <p><strong>Observe this boundary.</strong> {detail.observe}</p>
      </div>
    </div>}>
    <Label x={10} y={18}>DataFusion physical plan</Label>
    <Label x={418} y={18}>Native GPU execution schedule</Label>
    {[134, 230, 326, 422, 518].map((y) =>
      <Arrow key={y} id={`translate:cpu-${y}`} points={[[156, y - 6], [156, y - 24]]} />)}
    <Arrow id="translate:gpu-result" points={[[564, 128], [564, 110]]} />
    <Arrow id="translate:gpu-aggregate" points={[[564, 224], [564, 206]]} />
    <Arrow id="translate:gpu-filter" points={[[564, 320], [564, 302]]} />
    <Arrow id="translate:gpu-source" points={[[564, 416], [564, 398]]} />
    <g opacity={selected === 'aggregate' ? 0.9 : 0.35}>
      <Arrow id="translate:final-aggregate" points={[[308, 167], [356, 167], [356, 263], [380, 263]]} dashed arrowhead={false} />
      <Arrow id="translate:partial-aggregate" points={[[308, 359], [356, 359], [356, 263], [380, 263]]} dashed arrowhead={false} />
      <Arrow id="translate:merged-aggregate" points={[[380, 263], [412, 263]]} dashed />
    </g>
    <BackedLabel x={637} y={123} text="device → host" />
    <Label x={582} y={219}>aggregate barrier</Label>
    {node('result', 10, 38, 'ProjectionExec', 'customer_id, sum(...) AS total')}
    {node('aggregate', 10, 134, 'AggregateExec', 'FinalPartitioned · customer_id')}
    {node('partitions', 10, 230, 'RepartitionExec', 'Hash(customer_id) · P partitions')}
    {node('aggregate', 10, 326, 'AggregateExec', 'Partial · SUM(amount)')}
    {node('filter', 10, 422, 'FilterExec', "status = 'paid'")}
    {node('source', 10, 518, 'DataSourceExec', 'Parquet · projected scan columns')}
    {node('result', 418, 38, 'Host Arrow result', 'Wrapper materialization', 66, 'bridge')}
    {node('result', 418, 134, 'unary (projection)', 'GPU result · deferred_host_arrow')}
    {node('aggregate', 418, 230, 'aggregate', 'Grouped SUM · chunk partials + merge')}
    {node('filter', 418, 326, 'unary (filter)', "Filter(status = 'paid')")}
    {node('source', 418, 422, 'source_read', 'LocalParquetSource → GPU columns')}
  </Figure>;
}

export function PlanSelection() {
  return <GuideFigure height={300}
    label="Under NativePreferred, a supported whole-query candidate selected by cost becomes native. A cost-declined candidate keeps the CPU plan. If whole-query lowering is unsupported, the adapter can try supported child fragments and preserve valid parent contracts, producing a CPU or mixed plan."
    caption="Selection happens before execution. A running native query reports failures to its caller without replaying the query on CPU.">
    <Arrow id="selection:selected" points={[[232, 42], [418, 42]]} />
    <BackedLabel x={325} y={28} text="selected" />
    <Arrow id="selection:cost" points={[[232, 76], [320, 76], [320, 154], [418, 154]]} />
    <BackedLabel x={328} y={129} text="declined by cost" />
    <Arrow id="selection:unsupported" points={[[112, 92], [112, 218]]} />
    <BackedLabel x={112} y={158} text="unsupported" />
    <Arrow id="selection:fragments" points={[[232, 254], [418, 254]]} />
    <BackedLabel x={325} y={240} text="parent contracts hold" />
    <Card x={8} y={20} w={224} name="Whole query" note="Attempt native lowering" tone="engine" />
    <Card x={418} y={8} w={292} name="Native plan" note="GPU query operators" tone="gpu" />
    <Card x={418} y={118} w={292} name="CPU plan" note="DataFusion execution" tone="plan" />
    <Card x={8} y={218} w={224} name="Try child fragments" note="Keep valid replacements" tone="engine" />
    <Card x={418} y={218} w={292} name="CPU or mixed plan" note="Host Arrow at CPU boundaries" tone="bridge" />
  </GuideFigure>;
}

export function ParquetRead() {
  const columns = ['customer', 'amount', 'status', 'notes'];
  return <GuideFigure height={334}
    label="Illustrative Parquet scan after file selection. Each grid cell is a column chunk in a row group. Projection excludes notes; footer-based row-group pruning excludes all of RG 1. Retained compressed pages pass through KvikIO into GPU memory, then cuDF decodes them into columns."
    caption="In this single-file example, projection omits the notes column and metadata pruning skips all of RG 1. The retained row groups still require predicate evaluation.">
    <Region x={8} y={18} w={350} h={298} label="Selected Parquet file" tone="lake" />
    {columns.map((column, c) => <Label key={column} x={30 + c * 80} y={77}>{column}</Label>)}
    {[0, 1, 2].map((row) => <g key={row} opacity={row === 0 ? 0.35 : 1}>
      {columns.map((column, c) => <g key={column} opacity={c === 3 ? 0.35 : 1}>
        <rect x={27 + c * 80} y={92 + row * 46} width={70} height={32} rx={5}
          fill={TONE.lake} fillOpacity={0.12} stroke={COLOR.border} />
        <Label x={35 + c * 80} y={113 + row * 46}>RG {row + 1}</Label>
      </g>)}
    </g>)}
    <Label x={27} y={234}>Omitted: RG 1 and notes</Label>
    <Card x={24} y={248} w={318} h={52} name="Footer metadata" note="Schema, offsets, statistics" tone="lake" />
    <Arrow id="parquet:metadata" points={[[348, 274], [390, 274], [390, 54], [424, 54]]} />
    <Arrow id="parquet:select" points={[[570, 96], [570, 125]]} />
    <Arrow id="parquet:decode" points={[[570, 209], [570, 238]]} />
    <Label x={590} y={227}>cuDF decode</Label>
    <Card x={430} y={18} w={280} name="Project and prune" note="Columns · row groups" tone="plan" />
    <Card x={430} y={131} w={280} name="Read selected pages" note="KvikIO → GPU byte buffers" tone="bridge" />
    <Card x={430} y={244} w={280} name="Decoded columns" note="VRAM buffers for GPU operators" tone="gpu" />
  </GuideFigure>;
}

export function StorageRoutes() {
  return <GuideFigure width={960} minWidth={880} height={592}
    label="Payload routes for a PCIe-attached GPU. Default local compatibility path: NVMe file, POSIX read to pinned host bounce buffer, PCIe copy to VRAM. Conditional GDS path: storage DMA through cuFile to GPU memory. S3 path: HTTP range request through libcurl host buffers, copy to pinned host staging, PCIe copy to VRAM. cuObject path, drawn dashed and not yet supported by Algeon: an object server with cuObject writes the payload by RDMA over InfiniBand or RoCE into a registered VRAM buffer. Each route occupies one horizontal row."
    caption="The diagram shows a PCIe-attached GPU. Payload ends in VRAM; metadata and I/O submission remain on the CPU. The GDS row requires a supported and configured direct path. The dashed cuObject row shows a route Algeon does not support yet.">
    <Label x={8} y={18}>Local file · cuDF default compatibility path</Label>
    <Arrow id="io:posix" points={[[182, 82], [474, 82]]} />
    <BackedLabel x={328} y={68} text="pread" />
    <Arrow id="io:local-copy" points={[[710, 82], [798, 82]]} />
    <BackedLabel x={754} y={68} text="PCIe copy" />
    <Card x={8} y={46} w={168} name="NVMe file" note="Compressed pages" tone="lake" />
    <Card x={480} y={46} w={224} name="Pinned host buffer" note="Reusable bounce buffer" tone="bridge" />
    <Card x={804} y={46} w={148} name="GPU VRAM" note="For decoding" tone="gpu" />
    {sketchLine('io:divider1', 8, 146, 952, 146, {stroke: COLOR.border, dashed: true})}
    <Label x={8} y={177}>Local file · conditional GPUDirect Storage path</Label>
    <Arrow id="io:gds" points={[[182, 237], [798, 237]]} />
    <BackedLabel x={490} y={221} text="Storage DMA via GDS / cuFile over PCIe" />
    <Card x={8} y={201} w={168} name="NVMe file" note="Supported storage" tone="lake" />
    <Card x={804} y={201} w={148} name="GPU VRAM" note="For decoding" tone="gpu" />
    {sketchLine('io:divider2', 8, 298, 952, 298, {stroke: COLOR.border, dashed: true})}
    <Label x={8} y={329}>S3 · HTTP range reads through KvikIO</Label>
    <Arrow id="io:http" points={[[182, 393], [250, 393]]} />
    <BackedLabel x={216} y={379} text="HTTP" />
    <Arrow id="io:staging" points={[[436, 393], [508, 393]]} />
    <BackedLabel x={472} y={379} text="host copy" />
    <Arrow id="io:remote-copy" points={[[710, 393], [798, 393]]} />
    <BackedLabel x={754} y={379} text="PCIe copy" />
    <Card x={8} y={357} w={168} name="S3 object" note="Selected ranges" tone="lake" />
    <Card x={256} y={357} w={174} name="libcurl buffers" note="Host memory" tone="plan" />
    <Card x={514} y={357} w={190} name="Pinned staging" note="Batched curl chunks" tone="bridge" />
    <Card x={804} y={357} w={148} name="GPU VRAM" note="For decoding" tone="gpu" />
    {sketchLine('io:divider3', 8, 450, 952, 450, {stroke: COLOR.border, dashed: true})}
    <Label x={8} y={481}>S3 · cuObject RDMA path · not yet supported</Label>
    <Arrow id="io:cuobject" points={[[182, 545], [798, 545]]} dashed />
    <BackedLabel x={490} y={529} text="RDMA write via cuObject over InfiniBand / RoCE" />
    <Card x={8} y={509} w={168} name="S3 object" note="cuObject server" tone="lake" dashed />
    <Card x={804} y={509} w={148} name="GPU VRAM" note="For decoding" tone="gpu" dashed />
  </GuideFigure>;
}

export function ColumnLayout() {
  const rowLabels = ['Row 0', 'Row 1', 'Row 2', 'Row 3'];
  const fixedColWidth = 120;
  const startX = 210;
  const lanes = ['Lane 0', 'Lane 1', 'Lane 2', 'Lane 3'];
  const amountValues = ['12', '5', '8', '9'];
  const customerValues = ['7', '9', '7', '2'];
  const stringOffsets = ['0', '4', '8', '12', '20'];
  const stringChars = [
    {text: 'paid', bytes: '4 B', slice: '[0..4)', width: 96, x: startX},
    {text: 'open', bytes: '4 B', slice: '[4..8)', width: 96, x: startX + 96},
    {text: 'paid', bytes: '4 B', slice: '[8..12)', width: 96, x: startX + 192},
    {text: 'refunded', bytes: '8 B', slice: '[12..20)', width: 192, x: startX + 288},
  ];
  const validityBits = ['1', '1', '1', '1'];

  return (
    <GuideFigure
      height={472}
      label="Decoded GPU column buffers and coalesced memory access for four rows. A GPU Warp (32 threads) uses lanes 0 to 3 to scan adjacent amount values in one coalesced memory transaction, skipping customer_id. Variable-width status strings use offsets 0, 4, 8, 12, 20 to slice into the character buffer, and a validity bitmap tracks null values."
      caption="A GPU warp coalesces adjacent lane reads into fewer memory transactions, while unprojected columns consume no memory bandwidth. Strings use offset pairs to slice contiguous character bytes; validity bitmaps track nulls.">
      {/* Logical row header pills */}
      {rowLabels.map((label, i) => {
        const cx = startX + i * fixedColWidth + fixedColWidth / 2;
        return (
          <g key={label}>
            <rect
              x={cx - 32}
              y={10}
              width={64}
              height={22}
              rx={6}
              fill={COLOR.surface}
              stroke={COLOR.border}
              strokeWidth={1}
            />
            <text
              x={cx}
              y={25}
              textAnchor="middle"
              fontFamily={SANS}
              fontSize={11.5}
              fill={COLOR.muted}>
              {label}
            </text>
          </g>
        );
      })}

      {/* GPU Warp section: Left explanation & right spacious container */}
      <Label x={8} y={64}>GPU Warp</Label>
      <text x={8} y={80} fill={COLOR.muted} fontFamily={SANS} fontSize={11}>
        32 lanes (SIMT)
      </text>
      <text x={8} y={95} fill={COLOR.subtle} fontFamily={SANS} fontSize={10.5}>
        lanes 0..3 shown
      </text>

      <rect
        x={startX - 6}
        y={44}
        width={492}
        height={56}
        rx={10}
        fill={TONE.gpu}
        fillOpacity={0.04}
        stroke={TONE.gpu}
        strokeWidth={1}
        strokeDasharray="4 4"
      />
      <text
        x={startX + 10}
        y={58}
        fontFamily={SANS}
        fontSize={10}
        letterSpacing="0.04em"
        fill={COLOR.muted}>
        WARP 0 · ACTIVE EXECUTION LANES
      </text>

      {lanes.map((lane, i) => {
        const cx = startX + i * fixedColWidth + fixedColWidth / 2;
        return (
          <g key={lane}>
            <rect
              x={cx - 48}
              y={64}
              width={96}
              height={27}
              rx={6}
              fill={TONE.gpu}
              fillOpacity={0.14}
              stroke={COLOR.border}
            />
            <text
              x={cx}
              y={82}
              textAnchor="middle"
              fontFamily={SANS}
              fontSize={12}
              fill={COLOR.fg}>
              {lane}
            </text>
            <Arrow id={`coalesce:l${i}`} points={[[cx, 99], [cx, 118]]} />
          </g>
        );
      })}

      {/* amount column (scanned) */}
      <Label x={8} y={138}>amount</Label>
      <text x={8} y={154} fill={COLOR.muted} fontFamily={SANS} fontSize={11}>
        INT32 (scan target)
      </text>
      {amountValues.map((val, i) => {
        const x = startX + i * fixedColWidth;
        return (
          <g key={i}>
            <rect
              x={x}
              y={120}
              width={fixedColWidth}
              height={44}
              fill={TONE.gpu}
              fillOpacity={0.12}
              stroke={COLOR.border}
            />
            <text
              x={x + fixedColWidth / 2}
              y={147}
              textAnchor="middle"
              fontSize={16}
              fontFamily={SANS}
              fill={COLOR.fg}>
              {val}
            </text>
          </g>
        );
      })}

      {/* Coalesced transaction callout badge */}
      <path
        d={`M ${startX} 169 L ${startX} 175 L ${startX + 480} 175 L ${startX + 480} 169`}
        fill="none"
        stroke={TONE.gpu}
        strokeWidth={1.2}
      />
      <rect
        x={startX + 80}
        y={180}
        width={320}
        height={22}
        rx={11}
        fill={TONE.gpu}
        fillOpacity={0.12}
        stroke={TONE.gpu}
        strokeWidth={1}
      />
      <text
        x={startX + 240}
        y={195}
        textAnchor="middle"
        fontFamily={SANS}
        fontSize={11}
        fill={COLOR.fg}>
        1 coalesced memory transaction (16 contiguous bytes)
      </text>

      {/* customer_id column (unprojected / skipped) */}
      <Label x={8} y={230}>customer_id</Label>
      <text x={8} y={246} fill={COLOR.subtle} fontFamily={SANS} fontSize={11}>
        INT32 (unread)
      </text>
      {customerValues.map((val, i) => {
        const x = startX + i * fixedColWidth;
        return (
          <g key={i} opacity={0.4}>
            <rect
              x={x}
              y={212}
              width={fixedColWidth}
              height={40}
              fill={TONE.plain}
              fillOpacity={0.06}
              stroke={COLOR.border}
              strokeDasharray="4 4"
            />
            <text
              x={x + fixedColWidth / 2}
              y={237}
              textAnchor="middle"
              fontSize={15}
              fontFamily={SANS}
              fill={COLOR.fg}>
              {val}
            </text>
          </g>
        );
      })}
      <text
        x={startX + 240}
        y={267}
        textAnchor="middle"
        fontFamily={SANS}
        fontSize={11}
        fill={COLOR.subtle}>
        Skipped in projection: 0 bytes transferred from device memory
      </text>

      {/* Divider */}
      {sketchLine('layout:string-boundary', 8, 282, 696, 282, {stroke: COLOR.border, dashed: true})}
      <text x={8} y={298} fill={COLOR.muted} fontFamily={SANS} fontSize={11.5}>
        Variable-width column: offsets index into character buffer
      </text>

      {/* status offsets */}
      <Label x={8} y={328}>status offsets</Label>
      <text x={8} y={344} fill={COLOR.muted} fontFamily={SANS} fontSize={11}>
        INT32 (5 elements)
      </text>
      {stringOffsets.map((val, i) => {
        const x = startX + i * 96;
        return (
          <g key={i}>
            <rect
              x={x}
              y={312}
              width={96}
              height={34}
              fill={TONE.bridge}
              fillOpacity={0.12}
              stroke={COLOR.border}
            />
            <text
              x={x + 48}
              y={334}
              textAnchor="middle"
              fontSize={14}
              fontFamily={SANS}
              fill={COLOR.fg}>
              {val}
            </text>
          </g>
        );
      })}

      {/* status characters */}
      <Label x={8} y={380}>status bytes</Label>
      <text x={8} y={396} fill={COLOR.muted} fontFamily={SANS} fontSize={11}>
        UTF-8 bytes
      </text>
      {stringChars.map((item, i) => (
        <g key={i}>
          <rect
            x={item.x}
            y={358}
            width={item.width}
            height={48}
            fill={TONE.bridge}
            fillOpacity={0.12}
            stroke={COLOR.border}
          />
          <text
            x={item.x + item.width / 2}
            y={379}
            textAnchor="middle"
            fontSize={15}
            fontFamily={SANS}
            fill={COLOR.fg}>
            {item.text}
          </text>
          <text
            x={item.x + item.width / 2}
            y={396}
            textAnchor="middle"
            fontSize={10.5}
            fontFamily={SANS}
            fill={COLOR.muted}>
            {item.slice} ({item.bytes})
          </text>
        </g>
      ))}

      {/* validity bitmap */}
      <Label x={8} y={432}>validity bitmap</Label>
      <text x={8} y={448} fill={COLOR.muted} fontFamily={SANS} fontSize={11}>
        1 bit / row
      </text>
      {validityBits.map((bit, i) => {
        const x = startX + i * fixedColWidth;
        return (
          <g key={i}>
            <rect
              x={x}
              y={418}
              width={fixedColWidth}
              height={32}
              fill={TONE.gpu}
              fillOpacity={0.06}
              stroke={COLOR.border}
            />
            <text
              x={x + fixedColWidth / 2}
              y={439}
              textAnchor="middle"
              fontSize={13}
              fontFamily={SANS}
              fill={COLOR.fg}>
              bit {bit} (valid)
            </text>
          </g>
        );
      })}
    </GuideFigure>
  );
}

export function StreamSchedule() {
  const cardW = 330;
  const leftX = 45;
  const leftCx = leftX + cardW / 2; // 210
  const rightX = 425;
  const rightCx = rightX + cardW / 2; // 590
  const midX = 400;

  return (
    <GuideFigure
      width={800}
      minWidth={680}
      height={640}
      label="Asynchronous chunked aggregation across CUDA streams in one query attempt. Two blocking-lane jobs on the device's algeon-gpu-{d} runtime each enqueue one chunk into an independent CUDA stream, record a readiness event and return their permit. Stream A and Stream B execute decode, filter, and partial aggregation concurrently in shared VRAM. A GPU-side completion fence (cudaStreamWaitEvent) synchronizes both streams into the root stream for the final grouped merge without CPU blocking."
      caption="Each blocking-lane job on the device runtime enqueues its chunk, records an event and returns its permit. Streams A and B execute concurrently on the GPU, sharing VRAM and memory bandwidth under one query cap. The root stream waits on GPU events before merging partial results.">
      {/* 1. Background Regions with native container labels */}
      {/* Top Tier: Host CPU Worker Pool */}
      <Region
        x={16}
        y={6}
        w={768}
        h={110}
        label="Host CPU · algeon-gpu-{d} runtime, blocking lane"
        labelOffset={16}
        tone="plain"
      />

      {/* Main Tier: GPU VRAM Attempt Container */}
      <Region
        x={16}
        y={132}
        w={768}
        h={496}
        label="GPU VRAM · single admitted query attempt (shared memory cap & bandwidth)"
        labelOffset={16}
        tone="engine"
      />

      {/* Lane 1: Stream A */}
      <Region
        x={28}
        y={168}
        w={360}
        h={240}
        label="Lane 1 · Stream A (ordered FIFO queue)"
        labelOffset={16}
        tone="gpu"
      />

      {/* Lane 2: Stream B */}
      <Region
        x={412}
        y={168}
        w={360}
        h={240}
        label="Lane 2 · Stream B (ordered FIFO queue)"
        labelOffset={16}
        tone="gpu"
      />

      {/* Bottom Tier: Root Stream Region */}
      <Region
        x={28}
        y={486}
        w={744}
        h={132}
        label="Root Stream · query coordinator"
        labelOffset={16}
        tone="engine"
      />

      {/* 2. Arrows */}
      {/* Transition: Non-blocking Enqueue from Host to GPU Lanes */}
      <Arrow
        id="streams:enqueue-a"
        points={[[leftCx, 108], [leftCx, 204]]}
      />
      <Arrow
        id="streams:enqueue-b"
        points={[[rightCx, 108], [rightCx, 204]]}
      />

      {/* Lane 1 vertical flow */}
      <Arrow
        id="streams:flow-a"
        points={[[leftCx, 274], [leftCx, 298]]}
      />

      {/* Lane 2 vertical flow */}
      <Arrow
        id="streams:flow-b"
        points={[[rightCx, 274], [rightCx, 298]]}
      />

      {/* Merged Streams into Root Stream (Fork-Join merge) */}
      <Arrow
        id="streams:merge-a"
        points={[[leftCx, 368], [leftCx, 432], [midX, 432], [midX, 514]]}
      />
      <Arrow
        id="streams:merge-b"
        points={[[rightCx, 368], [rightCx, 432], [midX, 432], [midX, 514]]}
        arrowhead={false}
      />

      {/* 3. Event Milestone Markers on Stream Arrows */}
      <BackedLabel x={leftCx} y={400} text="◆ CUDA Event A recorded" />
      <BackedLabel x={rightCx} y={400} text="◆ CUDA Event B recorded" />

      {/* 4. Annotation beside the merged stream arrow */}
      <text
        x={midX + 14}
        y={452}
        fontFamily={SANS}
        fontSize={11}
        fill={COLOR.muted}>
        <tspan x={midX + 14} dy="0" fontWeight={600} fill={COLOR.fg}>
          GPU completion fence
        </tspan>
        <tspan x={midX + 14} dy="14">
          cudaStreamWaitEvent · unblocks Root Stream
        </tspan>
      </text>

      {/* 5. Operator Cards */}
      <Card
        x={leftX}
        y={40}
        w={cardW}
        h={68}
        name="Lane 1 · blocking-lane job"
        note={['Enqueues Chunk 1 to Stream A', 'Records Event A, returns its permit']}
        tone="plain"
      />
      <Card
        x={rightX}
        y={40}
        w={cardW}
        h={68}
        name="Lane 2 · blocking-lane job"
        note={['Enqueues Chunk 2 to Stream B', 'Records Event B, returns its permit']}
        tone="plain"
      />

      <Card
        x={leftX}
        y={204}
        w={cardW}
        h={70}
        name="1A. Decode + filter"
        note={['Scan Chunk 1 Parquet buffers', 'Evaluate predicates, gather matching rows']}
        tone="gpu"
      />
      <Card
        x={leftX}
        y={298}
        w={cardW}
        h={70}
        name="2A. Partial sums"
        note={['Group state hash table for Chunk 1', 'Retains intermediate state in VRAM']}
        tone="gpu"
      />

      <Card
        x={rightX}
        y={204}
        w={cardW}
        h={70}
        name="1B. Decode + filter"
        note={['Scan Chunk 2 Parquet buffers', 'Runs concurrently with Stream A']}
        tone="gpu"
      />
      <Card
        x={rightX}
        y={298}
        w={cardW}
        h={70}
        name="2B. Partial sums"
        note={['Group state hash table for Chunk 2', 'Competes for VRAM bus bandwidth']}
        tone="gpu"
      />

      <Card
        x={45}
        y={514}
        w={710}
        h={92}
        name="3. Merge partial results"
        note={[
          'Combines Chunk 1 and Chunk 2 partial group states into unified hash table',
          'Evaluates global aggregate functions (SUM, COUNT, AVG) on final keys',
          'Materializes resident result relation under the same query VRAM cap',
        ]}
        tone="engine"
      />
    </GuideFigure>
  );
}

export function GpuUdtfFlow() {
  const cardW = 348;
  const leftX = 50;
  const leftCx = leftX + cardW / 2; // 224
  const leftOuterX = leftX + 50; // 100 (outermost left departure/entry)
  const rightX = 482;
  const rightCx = rightX + cardW / 2; // 656
  const rightOuterX = rightX + cardW - 50; // 780 (outermost right departure/entry)

  return (
    <GuideFigure
      width={880}
      minWidth={760}
      height={810}
      label="GPU UDTF execution flow for PageRank within a SQL query. The relational pipeline on the left filters edge columns in cuDF. A zero-copy read lease passes edge buffers to cuGraph for graph construction and PageRank. The result is copied in device memory into independently owned cuDF columns for downstream filtering, joining, and aggregation, while native graph memory is settled and freed."
      caption="Relational SQL operators (left) and the graph algorithm (right) share VRAM under one admission grant. Edge inputs are borrowed via a zero-copy read lease; algorithm results are copied into owned cuDF columns so native graph memory can be freed for downstream relational operators.">
      {/* Outer VRAM Attempt Container */}
      <Region
        x={12}
        y={12}
        w={856}
        h={786}
        label="GPU VRAM · single admitted query attempt (shared memory cap)"
        tone="engine"
        labelOffset={20}
      />

      {/* Swimlane 1: Relational Pipeline (cuDF) */}
      <Region
        x={28}
        y={48}
        w={392}
        h={734}
        label="Relational Pipeline (cuDF)"
        tone="gpu"
        labelOffset={16}
      />

      {/* Swimlane 2: Algorithm Runtime (cuGraph UDTF) */}
      <Region
        x={460}
        y={48}
        w={392}
        h={734}
        label="Algorithm Runtime (cuGraph UDTF)"
        tone="graph"
        labelOffset={16}
      />

      {/* Arrow: Step 1 -> Step 2 (Zero-Copy Lease crossing from outermost points) */}
      <Arrow
        id="udtf:lease-arrow"
        points={[[leftOuterX, 168], [leftOuterX, 206], [rightOuterX, 206], [rightOuterX, 244]]}
      />

      {/* Arrow: Step 2 -> Step 3 (Inside Right Lane - center axis) */}
      <Arrow
        id="udtf:graph-to-pagerank"
        points={[[rightCx, 332], [rightCx, 368]]}
      />

      {/* Arrow: Step 3 -> Step 4 (Device Memory Copy crossing from outermost points) */}
      <Arrow
        id="udtf:copy-arrow"
        points={[[rightOuterX, 456], [rightOuterX, 484], [leftOuterX, 484], [leftOuterX, 532]]}
      />

      {/* Arrow: Step 4 -> Step 5 (Inside Left Lane - center axis) */}
      <Arrow
        id="udtf:result-to-downstream"
        points={[[leftCx, 620], [leftCx, 650]]}
      />

      {/* Dashed Arrow: Step 3 -> Resource Settle & Free (center axis of right lane) */}
      <Arrow
        id="udtf:settle-arrow"
        points={[[rightCx, 456], [rightCx, 530]]}
        dashed
      />

      {/* Badge: Zero-Copy Read Lease */}
      <rect
        x={220}
        y={192}
        width={440}
        height={28}
        rx={14}
        fill={COLOR.surface}
        stroke={TONE.bridge}
        strokeWidth={1.4}
      />
      <text
        x={440}
        y={210}
        textAnchor="middle"
        fontFamily={SANS}
        fontSize={12}
        fill={COLOR.fg}>
        Zero-Copy Read Lease (PreparedTableExport · borrows edge buffers)
      </text>

      {/* Badge: Device Memory Copy (positioned clear of the right-lane dashed arrow) */}
      <rect
        x={165}
        y={470}
        width={420}
        height={28}
        rx={14}
        fill={COLOR.surface}
        stroke={TONE.gpu}
        strokeWidth={1.4}
      />
      <text
        x={375}
        y={488}
        textAnchor="middle"
        fontFamily={SANS}
        fontSize={12}
        fill={COLOR.fg}>
        Device Memory Copy · to_dataframe(allocates owned cuDF columns)
      </text>

      {/* Card 1: Input Subquery */}
      <Card
        x={leftX}
        y={80}
        w={cardW}
        h={88}
        name="1. Edge Subquery (citations)"
        note={['WHERE citation_year >= 2020', 'cuDF owns src, dst columns']}
        tone="gpu"
      />

      {/* Card 2: Graph Construction */}
      <Card
        x={rightX}
        y={244}
        w={cardW}
        h={88}
        name="2. Graph Construction"
        note={['Validate column types and active ranges', 'Build CSR representation in VRAM']}
        tone="graph"
      />

      {/* Card 3: PageRank Kernel */}
      <Card
        x={rightX}
        y={368}
        w={cardW}
        h={88}
        name="3. PageRank Kernel"
        note={['Iterative power iteration on GPU', 'Owns native vertex and score arrays']}
        tone="graph"
      />

      {/* Card 4: UDTF Result Relation */}
      <Card
        x={leftX}
        y={532}
        w={cardW}
        h={88}
        name="4. UDTF Result (ranked)"
        note={['Independently owned vertex, value columns', 'SQL-facing schema normalization']}
        tone="gpu"
      />

      {/* Card 5: Downstream SQL Pipeline */}
      <Card
        x={leftX}
        y={650}
        w={cardW}
        h={110}
        name="5. Downstream SQL Pipeline"
        note={[
          'Filter: value > 0.0001',
          'Hash Join: papers ON paper_id = vertex',
          'Aggregate: GROUP BY field',
        ]}
        tone="gpu"
      />

      {/* Card 6: Resource Settle & Free */}
      <Card
        x={rightX}
        y={532}
        w={cardW}
        h={124}
        name="Resource Settle & Free"
        note={[
          'PageRank native arrays freed after copy',
          'Read lease on edge buffers released',
          'VRAM returned to query pool for hash join',
        ]}
        tone="plain"
      />
      <BackedLabel x={rightCx} y={508} text="after copy completes" />
    </GuideFigure>
  );
}

export function AttemptLifetime() {
  return <GuideFigure height={460} minWidth={650}
    label="Attempt lifecycle across three thread contexts. On the host runtime, the capability proof precedes execution and the attempt waits in the admission queue on the caller's task. On the device's algeon-gpu-{d} runtime, an envelope lifecycle job takes the lane, root stream and allocation domain, one QueryExecution job runs the whole GPU schedule on one worker, and each poll submits one job that materializes a chunk of at most 4 MiB, which the host runtime receives as an Arrow batch. The next poll loops back to the materialize job. When the stream reaches EOF, is dropped or is cancelled, the cleanup thread synchronizes the attempt's streams and returns capacity."
    caption="The first poll waits for admission on the caller's task, then runs the whole GPU schedule as one pool job. Each later poll submits one job that moves at most 4 MiB to the host. Release follows the terminal outcome and the cleanup thread's synchronize.">
    <Region x={8} y={8} w={704} h={116} label="host runtime" tone="plan" labelOffset={16} />
    <Region x={8} y={172} w={704} h={116} label="algeon-gpu-{d} runtime" tone="gpu" labelOffset={16} />
    <Region x={8} y={336} w={704} h={116} label="cleanup thread" tone="engine" labelOffset={16} />
    <Arrow id="attempt:proof-queue" points={[[188, 78], [236, 78]]} />
    <Arrow id="attempt:queue-envelope" points={[[318, 112], [318, 148], [208, 148], [208, 208]]} />
    <Arrow id="attempt:envelope-execute" points={[[229, 242], [273, 242]]} />
    <Arrow id="attempt:execute-materialize" points={[[433, 242], [477, 242]]} />
    <Arrow id="attempt:materialize-batch" points={[[545, 208], [545, 112]]} />
    <Arrow id="attempt:next-poll" points={[[620, 112], [620, 208]]} dashed />
    <BackedLabel x={620} y={152} text="next poll" />
    <Arrow id="attempt:release" points={[[674, 112], [674, 372]]} />
    <BackedLabel x={600} y={316} text="EOF · drop · cancel" />
    <Card x={24} y={44} w={164} h={68} name="Capability proof" note="Before execution" tone="plan" />
    <Card x={236} y={44} w={164} h={68} name="Admission queue"
      note={['Wait for capacity', "on the caller's task"]} tone="plan" />
    <Card x={520} y={44} w={176} h={68} name="Arrow batch" note="≤ 4 MiB per poll" tone="bridge" />
    <Card x={24} y={208} w={205} h={68} name="Envelope"
      note={['Lifecycle job', 'Lane, root stream, domain']} tone="gpu" />
    <Card x={273} y={208} w={160} h={68} name="Execute"
      note={['QueryExecution job', 'One worker']} tone="gpu" />
    <Card x={477} y={208} w={174} h={68} name="Materialize chunk" note="One job per poll" tone="gpu" />
    <Card x={406} y={372} w={290} h={68} name="Synchronize, return capacity"
      note="After the attempt's terminal outcome" tone="engine" />
  </GuideFigure>;
}

export function AllocationPath() {
  return <GuideFigure height={228}
    label="GPU operations allocate through the query's DeviceMemoryResource. The query resource checks the immutable cap. Accepted allocations use the device RMM resource and GPU storage; an allocation beyond the cap returns a typed failure."
    caption="The grant authorizes a byte limit. Physical buffers are obtained as operations allocate; unused grant capacity is still committed in the device ledger.">
    <Arrow id="allocation:request" points={[[206, 48], [258, 48]]} />
    <Arrow id="allocation:accept" points={[[462, 48], [514, 48]]} />
    <Arrow id="allocation:storage" points={[[612, 84], [612, 150]]} />
    <Arrow id="allocation:reject" points={[[360, 84], [360, 150]]} dashed />
    <BackedLabel x={360} y={125} text="over cap" />
    <Card x={8} y={12} w={198} name="GPU operation" note="cuDF / graph / vector / ML" tone="gpu" />
    <Card x={258} y={12} w={204} name="Query resource" note="Enforce the grant" tone="engine" />
    <Card x={514} y={12} w={196} name="RMM resource" note="Reuse or allocate" tone="bridge" />
    <Card x={258} y={150} w={204} name="Typed failure" note="Allocation rejected" tone="reject" />
    <Card x={514} y={150} w={196} name="GPU storage" note="Columns and scratch" tone="gpu" />
  </GuideFigure>;
}

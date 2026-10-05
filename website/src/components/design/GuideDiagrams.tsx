import React, {useState} from 'react';
import {BackedLabel, Box, COLOR, Figure, TONE} from './Diagram';
import type {DiagramNode, Tone} from './Diagram';
import {SANS, sketchLine, sketchRoundRect} from './sketch';
import styles from './styles.module.css';

type Point = [number, number];

function Card({x, y, w, h = 72, name, note, tone = 'plain', kind}: DiagramNode & {
  x: number;
  y: number;
  w: number;
  h?: number;
}) {
  return <Box node={{name, note, tone, kind}} x={x} y={y} w={w} h={h} />;
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
  x: number; y: number; w: number; h: number; label: string; tone?: Tone; labelOffset?: number;
}) {
  return <g>
    <rect x={x} y={y} width={w} height={h} rx={16} fill={TONE[tone]} opacity={0.05} />
    {sketchRoundRect(`guide-region:${label}`, x, y, w, h, 16,
      {stroke: TONE[tone], strokeWidth: 1, dashed: true})}
    <Label x={x + labelOffset} y={y + 22}>{label}</Label>
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
  return <GuideFigure width={960} minWidth={880} height={440}
    label="Payload routes for a PCIe-attached GPU. Default local compatibility path: NVMe file, POSIX read to pinned host bounce buffer, PCIe copy to VRAM. Conditional GDS path: storage DMA through cuFile to GPU memory. S3 path: HTTP range request through libcurl host buffers, copy to pinned host staging, PCIe copy to VRAM. Each route occupies one horizontal row."
    caption="The diagram shows a PCIe-attached GPU. Payload ends in VRAM; metadata and I/O submission remain on the CPU. The GDS row requires a supported and configured direct path.">
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
  </GuideFigure>;
}

export function ColumnLayout() {
  const rows = [
    {name: 'customer_id', values: ['7', '9', '7', '2'], y: 28, tone: 'gpu' as Tone},
    {name: 'amount', values: ['12', '5', '8', '9'], y: 91, tone: 'gpu' as Tone},
    {name: 'status offsets', values: ['0', '4', '8', '12', '20'], y: 185, tone: 'bridge' as Tone},
    {name: 'status characters', values: ['paid', 'open', 'paid', 'refunded'],
      widths: [94, 94, 94, 188], y: 248, tone: 'bridge' as Tone},
  ];
  return <GuideFigure height={318}
    label="Decoded column buffers for four rows. customer_id contains 7, 9, 7, 2; amount contains 12, 5, 8, 9. String status uses offsets 0, 4, 8, 12, 20 into the character bytes paidopenpaidrefunded. The longer last string occupies twice as many bytes. Each column has separate contiguous buffers."
    caption="Each offset pair selects one string's bytes; refunded needs more space than paid. Row position connects values across columns. Optional validity bitmaps track nulls separately.">
    {rows.map((row) => <g key={row.name}>
      <Label x={8} y={row.y + 27}>{row.name}</Label>
      {row.values.map((value, i) => {
        const widths = row.widths ?? row.values.map(() => row.values.length === 5 ? 94 : 118);
        const width = widths[i];
        const x = 214 + widths.slice(0, i).reduce((sum, size) => sum + size, 0);
        return <g key={i}>
          <rect x={x} y={row.y} width={width} height={44}
            fill={TONE[row.tone]} fillOpacity={0.09} stroke={COLOR.border} />
          <text x={x + width / 2} y={row.y + 29} textAnchor="middle"
            fontSize={16} fontFamily={SANS} fill={COLOR.fg}>{value}</text>
        </g>;
      })}
    </g>)}
    {sketchLine('layout:string-boundary', 8, 158, 702, 158, {stroke: COLOR.border, dashed: true})}
  </GuideFigure>;
}

export function StreamSchedule() {
  return <GuideFigure height={384}
    label="Illustrative chunked aggregation on one GPU. Two parallel columns show Stream A decoding, filtering and partially aggregating chunk 1, and Stream B doing the same for chunk 2. Symmetric completion arrows connect both partial results to the root stream's merge."
    caption="Both streams operate on the same GPU's VRAM. The merge waits for both partial results; independent chunks may overlap when resources permit.">
    <Label x={24} y={18}>Stream A</Label>
    <Label x={424} y={18}>Stream B</Label>
    <Label x={24} y={342}>Root stream</Label>
    <Arrow id="streams:a" points={[[160, 116], [160, 134]]} />
    <Arrow id="streams:b" points={[[560, 116], [560, 134]]} />
    <Arrow id="streams:join-a" points={[[160, 218], [160, 262], [284, 262], [284, 294]]} />
    <Arrow id="streams:join-b" points={[[560, 218], [560, 262], [436, 262], [436, 294]]} />
    <BackedLabel x={360} y={246} text="wait for both producers" />
    <Card x={24} y={38} w={272} name="Decode + filter" note="Chunk 1" tone="gpu" />
    <Card x={24} y={140} w={272} name="Partial sums" note="Group state for chunk 1" tone="gpu" />
    <Card x={424} y={38} w={272} name="Decode + filter" note="Chunk 2" tone="gpu" />
    <Card x={424} y={140} w={272} name="Partial sums" note="Group state for chunk 2" tone="gpu" />
    <Card x={208} y={300} w={304} name="Merge partial results" note="Preserve the grouped-sum result" tone="engine" />
  </GuideFigure>;
}

export function GpuUdtfFlow() {
  return <GuideFigure width={880} minWidth={760} height={390}
    label="PageRank within a native GPU query. Filtered cuDF edge columns are exposed to cuGraph through a prepared export and read lease. Graph construction and PageRank produce native arrays that are copied on the device into owned cuDF result columns. A resident relation then feeds native filter, join and aggregate operations."
    caption="Illustrative uncached path on one GPU. Producer storage remains owned through consumer reads. Graph preparation and the PageRank result conversion can allocate and copy within VRAM; host output is materialized at a later CPU or client boundary.">
    <Region x={8} y={8} w={864} h={372} label="GPU VRAM · one admitted attempt" tone="engine" />
    <Arrow id="udtf:input" points={[[272, 107], [542, 107]]} />
    <Label x={407} y={95} textAnchor="middle">prepared export + read lease</Label>
    <Arrow id="udtf:result" points={[[698, 152], [698, 196], [145, 196], [145, 240]]} />
    <Label x={421} y={184} textAnchor="middle">Native result → device copy into cuDF</Label>
    <Arrow id="udtf:consumer" points={[[272, 294], [542, 294]]} />
    <Label x={407} y={282} textAnchor="middle">resident relation</Label>
    <Card x={24} y={68} w={242} h={78} name="Filtered cuDF edges" note="Owns src / dst columns" tone="gpu" />
    <Card x={548} y={68} w={300} h={78} name="cuGraph PageRank" note="Normalize edges · build graph" tone="graph" />
    <Card x={24} y={246} w={242} h={96} name="cuDF result" note={['Owns vertex / value columns', 'Independent result allocations']} tone="gpu" />
    <Card x={548} y={246} w={300} h={96} name="Filter · join · aggregate" note="Native GPU consumers" tone="gpu" />
  </GuideFigure>;
}

export function AttemptLifetime() {
  return <GuideFigure height={276}
    label="Attempt lifecycle: planning proves required capabilities, execution joins admission, the runtime commits one device and grant, all fragments run, terminal work and retained owners are settled, then resources return to the ledger."
    caption="Planning alone acquires no grant. Release follows terminal execution and successful cleanup, including retained outputs and pending GPU work.">
    <Arrow id="attempt:queue" points={[[214, 48], [258, 48]]} />
    <Arrow id="attempt:grant" points={[[462, 48], [506, 48]]} />
    <Arrow id="attempt:run" points={[[608, 84], [608, 192]]} />
    <BackedLabel x={492} y={140} text="one attempt across fragments" />
    <Arrow id="attempt:settle" points={[[506, 228], [462, 228]]} />
    <Arrow id="attempt:release" points={[[258, 228], [214, 228]]} />
    <Card x={10} y={12} w={204} name="Capability proof" note="Before execution" tone="plan" />
    <Card x={258} y={12} w={204} name="Admission queue" note="Wait for capacity" tone="engine" />
    <Card x={506} y={12} w={204} name="Device + grant" note="Fixed for the attempt" tone="engine" />
    <Card x={506} y={192} w={204} name="Execute" note="Shared allocation cap" tone="gpu" />
    <Card x={258} y={192} w={204} name="Settle resources" note="Wait; release owners" tone="bridge" />
    <Card x={10} y={192} w={204} name="Return capacity" note="After safe cleanup" tone="engine" />
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

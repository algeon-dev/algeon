import React, {useState} from 'react';
import {BackedLabel, Box, COLOR, Figure, LABEL, TONE, textWidth} from './Diagram';
import type {DiagramNode, Tone} from './Diagram';
import {SANS, sketchLine, sketchRoundRect} from './sketch';
import styles from './styles.module.css';

export type Point = [number, number];

export function Card({x, y, w, h = 72, name, note, tone = 'plain', kind, dashed = false}: DiagramNode & {
  x: number;
  y: number;
  w: number;
  h?: number;
  dashed?: boolean;
}) {
  return <Box node={{name, note, tone, kind}} x={x} y={y} w={w} h={h} borderDashed={dashed} />;
}

export function Arrow({id, points, dashed = false, arrowhead = true}: {
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

export function Label({x, y, children, textAnchor = 'start'}: {
  x: number; y: number; children: React.ReactNode; textAnchor?: 'start' | 'middle' | 'end';
}) {
  return <text x={x} y={y} textAnchor={textAnchor}
    fill={COLOR.muted} fontFamily={SANS} fontSize={13}>{children}</text>;
}

export function EdgeLabel({x, y, text, anchor = 'middle'}: {
  x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end';
}) {
  const width = textWidth(text, LABEL);
  const left = anchor === 'start' ? x : anchor === 'end' ? x - width : x - width / 2;
  return <g>
    <rect x={left - 6} y={y - 13} width={width + 12} height={20} rx={3} fill={COLOR.surface} />
    <text x={x} y={y} textAnchor={anchor} fontFamily={SANS} fontSize={12} fill={COLOR.muted}>{text}</text>
  </g>;
}

export function Region({x, y, w, h, label, tone = 'plain', labelOffset = 14}: {
  x: number; y: number; w: number; h: number; label?: string; tone?: Tone; labelOffset?: number;
}) {
  return <g>
    <rect x={x} y={y} width={w} height={h} rx={16} fill={TONE[tone]} opacity={0.05} />
    {sketchRoundRect(`guide-region:${label ?? `${x}:${y}`}`, x, y, w, h, 16,
      {stroke: TONE[tone], strokeWidth: 1, dashed: true})}
    {label ? <Label x={x + labelOffset} y={y + 22}>{label}</Label> : null}
  </g>;
}

export function GuideFigure({height, width = 800, minWidth = 640, maxWidth = 680, label, title, caption, children}: {
  height: number; width?: number; minWidth?: number; maxWidth?: number;
  title?: string; label: string; caption: React.ReactNode; children: React.ReactNode;
}) {
  return <Figure svgW={width} svgH={height} minW={minWidth} maxW={maxWidth} ariaLabel={label} title={title} caption={caption} center>
    {children}
  </Figure>;
}

export function QueryArchitecture() {
  return <GuideFigure title="Query flow and module boundaries" height={808}
    label="Query flow and module boundaries. DataFusion plans SQL on the CPU. algeon-datafusion lowers the physical plan and invokes query-engine. Query-planning owns IR, capability proof and the compiled schedule. Query-runtime owns the attempt, device grant and cache. Query-engine submits GPU operations through RAPIDS, reading Parquet from local NVMe or S3 and returning host Arrow batches through the adapter and DataFusion. Dashed arrows show direct imports: the adapter imports DataFusion and all three query crates; the engine imports planning and runtime; runtime imports planning."
    caption="Follow the solid arrows from SQL to the result. Dashed arrows point to imported crates. The native query crates orchestrate GPU work from the CPU; RAPIDS operations run on the device.">

    <Region x={16} y={192} w={768} h={280} label="DataFusion-free query crates · CPU orchestration" labelOffset={192} tone="engine" />
    <Arrow id="architecture:physical-plan" points={[[296, 68], [448, 68]]} />
    <EdgeLabel x={372} y={52} text="physical plan" />
    <Arrow id="architecture:adapter-datafusion" points={[[448, 104], [296, 104]]} dashed />
    <Arrow id="architecture:execute" points={[[612, 128], [612, 240]]} />
    <EdgeLabel x={628} y={176} text="execute" anchor="start" />
    <Arrow id="architecture:adapter-engine" points={[[744, 128], [744, 240]]} dashed />
    <Arrow id="architecture:adapter-planning" points={[[520, 128], [520, 176], [160, 176], [160, 240]]} dashed />
    <Arrow id="architecture:adapter-runtime" points={[[472, 128], [472, 152], [4, 152], [4, 412], [40, 412]]} dashed />
    <Arrow id="architecture:engine-planning" points={[[448, 280], [296, 280]]} dashed />
    <Arrow id="architecture:engine-runtime" points={[[488, 328], [488, 412], [296, 412]]} dashed />
    <Arrow id="architecture:runtime-planning" points={[[160, 372], [160, 328]]} dashed />
    <Arrow id="architecture:submit" points={[[612, 328], [612, 552]]} />
    <EdgeLabel x={628} y={512} text="GPU submissions" anchor="start" />
    <Arrow id="architecture:read" points={[[296, 544], [372, 544], [372, 580], [448, 580]]} />
    <EdgeLabel x={372} y={528} text="read pages" />
    <Arrow id="architecture:result" points={[[448, 628], [372, 628], [372, 668], [296, 668]]} />
    <EdgeLabel x={372} y={684} text="device → host" />
    <Card x={40} y={40} w={256} h={88} name="DataFusion"
      note={['SQL / DataFrame', 'Physical plan on CPU']} tone="plan" />
    <Card x={448} y={40} w={328} h={88} name="algeon-datafusion"
      note={['Lower the physical plan', 'Wrap execution and Arrow results']} tone="engine" />
    <Card x={40} y={240} w={256} h={88} name="query-planning"
      note={['IR and capability proof', 'Compiled GPU execution schedule']} tone="plan" />
    <Card x={448} y={240} w={328} h={88} name="query-engine"
      note={['Execute the schedule', 'Use planning + runtime contracts']} tone="bridge" />
    <Card x={40} y={372} w={256} h={80} name="query-runtime"
      note={['One device + grant', 'Attempt lifetime and cache']} tone="engine" />
    <Card x={40} y={504} w={256} h={80} name="Parquet storage"
      note={['Local NVMe or S3', 'Selected column pages']} tone="lake" />
    <Card x={448} y={552} w={328} h={104} name="GPU operations" tone="gpu"
      note={['cuDF · cuGraph · cuVS · cuML', 'Decode → filter / join / aggregate']} />
    <Card x={40} y={628} w={256} h={80} name="Arrow result" tone="bridge"
      note="Via adapter → DataFusion → caller" />
    <Arrow id="architecture:flow-legend" points={[[40, 784], [88, 784]]} />
    <Label x={100} y={788}>Execution / data flow</Label>
    <Arrow id="architecture:import-legend" points={[[448, 784], [496, 784]]} dashed />
    <Label x={508} y={788}>Imports from</Label>
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
        separates planned input chunks. The CPU partition count does not set the number of chunks.</>,
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
        Output bytes describe result batches. They do not measure peak device allocation.</>,
    },
  };
  const detail = details[selected];
  function node(id: typeof selected, x: number, y: number, name: string, note: string | string[],
    tone: Tone = x < 400 ? 'plan' : 'gpu') {
    return <Box x={x} y={y} w={296} h={76} node={{name, note, tone}}
      selected={selected === id} onSelect={() => select(id)} />;
  }
  return <Figure title="Grouped aggregate lowering" svgW={800} svgH={744} minW={640} maxW={680} center interactive
    ariaLabel="Interactive grouped-sum lowering example. Data flows upward from Parquet. Select a DataFusion operator or native task to inspect its mapping and related observation fields."
    caption="Select a node to inspect its mapping. Data flows upward. The two dashed arrows map the partial and final aggregates to one native aggregate. Host Arrow materialization belongs to the wrapper. Batching and scan-repartition wrappers are omitted."
    panel={<div className={styles.panel} aria-live="polite">
      <div className={styles.panelHead}><span className={styles.panelName}>{detail.title}</span></div>
      <div className={styles.panelBody}>
        <p>{detail.body}</p>
        <p><strong>Observe this boundary.</strong> {detail.observe}</p>
      </div>
    </div>}>

    <Region x={8} y={8} w={328} h={696} label="DataFusion physical plan" tone="plan" />
    <Region x={464} y={8} w={328} h={696} label="Native GPU schedule + result wrapper" tone="gpu" />
    {[164, 268, 372, 476, 580].map((y) =>
      <Arrow key={y} id={`translate:cpu-${y}`} points={[[172, y], [172, y - 28]]} />)}
    {[164, 372, 476].map((y) =>
      <Arrow key={y} id={`translate:gpu-${y}`} points={[[628, y], [628, y - 28]]} />)}
    <Arrow id="translate:gpu-aggregate" points={[[628, 268], [628, 240]]} />
    <g opacity={selected === 'aggregate' ? 1 : 0.45}>
      <Arrow id="translate:final-aggregate" points={[[320, 202], [400, 202], [400, 292], [480, 292]]} dashed />
      <Arrow id="translate:partial-aggregate" points={[[320, 410], [400, 410], [400, 320], [480, 320]]} dashed />
    </g>
    <EdgeLabel x={648} y={152} text="device → host" anchor="start" />
    <EdgeLabel x={648} y={256} text="merge barrier" anchor="start" />
    {node('result', 24, 60, 'ProjectionExec', 'customer_id, sum(...) AS total')}
    {node('aggregate', 24, 164, 'AggregateExec', 'FinalPartitioned · customer_id')}
    {node('partitions', 24, 268, 'RepartitionExec', 'Hash(customer_id) · P partitions')}
    {node('aggregate', 24, 372, 'AggregateExec', 'Partial · SUM(amount)')}
    {node('filter', 24, 476, 'FilterExec', "status = 'paid'")}
    {node('source', 24, 580, 'DataSourceExec', 'Parquet · projected scan columns')}
    {node('result', 480, 60, 'Host Arrow result', 'Wrapper materialization', 'bridge')}
    {node('result', 480, 164, 'unary (projection)', 'GPU result · deferred_host_arrow')}
    {node('aggregate', 480, 268, 'aggregate', ['Grouped SUM', 'Chunk partials + merge'])}
    {node('filter', 480, 372, 'unary (filter)', "Filter(status = 'paid')")}
    {node('source', 480, 476, 'source_read', 'LocalParquetSource → GPU columns')}
    <Label x={480} y={604}>Native grouping absorbs hash repartition.</Label>
    <Label x={480} y={632}>One bounded output partition.</Label>
    <Arrow id="translate:mapping-legend" points={[[24, 728], [72, 728]]} dashed />
    <Label x={84} y={732}>Partial + final → one compiled aggregate task</Label>
  </Figure>;
}

export function PlanSelection() {
  return <GuideFigure title="Native plan selection" height={384}
    label="Under NativePreferred, a supported whole-query candidate selected by cost becomes native. A cost-declined candidate keeps the CPU plan. If whole-query lowering is unsupported, the adapter can try supported child fragments and preserve valid parent contracts, producing a CPU or mixed plan."
    caption="Selection happens before execution. A running native query reports failures to its caller without replaying the query on CPU.">

    <Arrow id="selection:selected" points={[[272, 72], [496, 72]]} />
    <EdgeLabel x={384} y={56} text="selected" />
    <Arrow id="selection:cost" points={[[272, 104], [384, 104], [384, 208], [496, 208]]} />
    <EdgeLabel x={400} y={156} text="cost declined" anchor="start" />
    <Arrow id="selection:unsupported" points={[[148, 128], [148, 288]]} />
    <EdgeLabel x={164} y={208} text="unsupported" anchor="start" />
    <Arrow id="selection:fragments" points={[[272, 324], [496, 324]]} />
    <EdgeLabel x={384} y={308} text="valid parents" />
    <Card x={24} y={48} w={248} h={80} name="Whole query" note="Attempt native lowering" tone="engine" />
    <Card x={496} y={32} w={280} h={80} name="Native plan" note="GPU query operators" tone="gpu" />
    <Card x={496} y={168} w={280} h={80} name="CPU plan" note="DataFusion execution" tone="plan" />
    <Card x={24} y={288} w={248} h={72} name="Try child fragments" note="Keep valid replacements" tone="engine" />
    <Card x={496} y={288} w={280} h={72} name="CPU or mixed plan" note="Host Arrow at CPU boundaries" tone="bridge" />
  </GuideFigure>;
}

export function ParquetRead() {
  return <GuideFigure title="Parquet projection and pruning" height={408}
    label="Illustrative Parquet scan after file selection. Each grid cell is a column chunk in a row group. Projection excludes notes; footer-based row-group pruning excludes all of RG 1. Retained compressed pages pass through KvikIO into GPU memory, then cuDF decodes them into columns."
    caption="In this single-file example, projection omits the notes column and metadata pruning skips all of RG 1. The retained row groups still require predicate evaluation.">

    <Region x={16} y={16} w={360} h={376} label="Selected Parquet file" tone="lake" />
    <Arrow id="parquet:metadata" points={[[352, 340], [400, 340], [400, 56], [480, 56]]} />
    <EdgeLabel x={440} y={40} text="metadata" />
    <Arrow id="parquet:select" points={[[628, 96], [628, 164]]} />
    <Arrow id="parquet:decode" points={[[628, 244], [628, 312]]} />
    <EdgeLabel x={644} y={282} text="cuDF decode" anchor="start" />
    {['customer', 'amount', 'status', 'notes'].map((column, c) =>
      <Label key={column} x={76 + c * 80} y={88} textAnchor="middle">{column}</Label>)}
    {[0, 1, 2].map((row) => <g key={row}>
      {[0, 1, 2, 3].map((c) => {
        const omitted = row === 0 || c === 3;
        return <g key={c}>
          <rect x={40 + c * 80} y={108 + row * 48} width={72} height={36} rx={4}
            fill={omitted ? COLOR.surface : TONE.lake} fillOpacity={omitted ? 1 : 0.14}
            stroke={COLOR.border} strokeDasharray={omitted ? '4 4' : undefined} />
          <Label x={76 + c * 80} y={131 + row * 48} textAnchor="middle">{omitted ? 'skip' : `RG ${row + 1}`}</Label>
        </g>;
      })}
    </g>)}
    <Label x={40} y={276}>Projection skips notes; statistics skip RG 1.</Label>
    <Card x={40} y={304} w={312} h={72} name="Footer metadata" note="Schema, offsets, statistics" tone="lake" />
    <Card x={480} y={16} w={296} h={80} name="Project and prune" note="Columns · row groups" tone="plan" />
    <Card x={480} y={164} w={296} h={80} name="Read selected pages" note="KvikIO → GPU byte buffers" tone="bridge" />
    <Card x={480} y={312} w={296} h={80} name="Decoded columns" note="VRAM buffers for GPU operators" tone="gpu" />
  </GuideFigure>;
}

export function StorageRoutes() {
  return <GuideFigure title="Storage payload routes" width={960} minWidth={768} maxWidth={816} height={640}
    label="Payload routes for a PCIe-attached GPU. Default local compatibility path: NVMe file, POSIX read to pinned host bounce buffer, PCIe copy to VRAM. Conditional GDS path: storage DMA through cuFile to GPU memory. S3 path: HTTP range request through libcurl host buffers, copy to pinned host staging, PCIe copy to VRAM. cuObject path, drawn dashed and not yet supported by Algeon: an object server with cuObject writes the payload by RDMA over InfiniBand or RoCE into a registered VRAM buffer. Each route occupies one horizontal row."
    caption="The diagram shows a PCIe-attached GPU. Payload ends in VRAM; metadata and I/O submission remain on the CPU. The GDS row requires a supported and configured direct path. The dashed cuObject row shows a route Algeon does not support yet.">

    {[
      'Local file · default compatibility path',
      'Local file · conditional GPUDirect Storage path',
      'S3 · HTTP range reads',
      'S3 · cuObject RDMA · not yet supported',
    ].map((label, i) => <g key={label}>
      <Label x={24} y={28 + i * 160}>{label}</Label>
      {i > 0 && sketchLine(`io:divider-${i}`, 24, i * 160 - 8, 936, i * 160 - 8,
        {stroke: COLOR.border, dashed: true})}
    </g>)}
    <Arrow id="io:posix" points={[[184, 96], [512, 96]]} />
    <EdgeLabel x={348} y={80} text="pread" />
    <Arrow id="io:local-copy" points={[[712, 96], [800, 96]]} />
    <EdgeLabel x={756} y={80} text="PCIe copy" />
    <Arrow id="io:gds" points={[[184, 256], [800, 256]]} />
    <EdgeLabel x={492} y={240} text="Storage DMA" />
    <Label x={492} y={288} textAnchor="middle">GDS / cuFile over PCIe · supported storage and configuration</Label>
    <Arrow id="io:http" points={[[184, 416], [280, 416]]} />
    <EdgeLabel x={232} y={400} text="HTTP" />
    <Arrow id="io:staging" points={[[440, 416], [512, 416]]} />
    <EdgeLabel x={476} y={400} text="host copy" />
    <Arrow id="io:remote-copy" points={[[712, 416], [800, 416]]} />
    <EdgeLabel x={756} y={400} text="PCIe copy" />
    <Arrow id="io:cuobject" points={[[184, 576], [800, 576]]} dashed />
    <EdgeLabel x={492} y={560} text="RDMA write" />
    <Label x={492} y={608} textAnchor="middle">cuObject over InfiniBand / RoCE</Label>
    <Card x={24} y={56} w={160} h={80} name="NVMe file" note="Compressed pages" tone="lake" />
    <Card x={512} y={56} w={200} h={80} name="Pinned host buffer" note="Reusable bounce buffer" tone="bridge" />
    <Card x={800} y={56} w={136} h={80} name="GPU VRAM" note="For decoding" tone="gpu" />
    <Card x={24} y={216} w={160} h={80} name="NVMe file" note="Supported storage" tone="lake" />
    <Card x={800} y={216} w={136} h={80} name="GPU VRAM" note="For decoding" tone="gpu" />
    <Card x={24} y={376} w={160} h={80} name="S3 object" note="Selected ranges" tone="lake" />
    <Card x={280} y={376} w={160} h={80} name="libcurl buffers" note="Host memory" tone="plan" />
    <Card x={512} y={376} w={200} h={80} name="Pinned staging" note="Batched curl chunks" tone="bridge" />
    <Card x={800} y={376} w={136} h={80} name="GPU VRAM" note="For decoding" tone="gpu" />
    <Card x={24} y={536} w={160} h={80} name="S3 object" note="cuObject server" tone="lake" dashed />
    <Card x={800} y={536} w={136} h={80} name="GPU VRAM" note="For decoding" tone="gpu" dashed />
  </GuideFigure>;
}

export function ColumnLayout() {
  const startX = 248;
  const cellW = 120;
  function cells(values: string[], y: number, tone: Tone, width = cellW, dashed = false) {
    return values.map((value, i) => <g key={i}>
      <rect x={startX + i * width} y={y} width={width} height={40} rx={3}
        fill={TONE[tone]} fillOpacity={dashed ? 0.03 : 0.12} stroke={COLOR.border}
        strokeDasharray={dashed ? '4 4' : undefined} />
      <text x={startX + (i + 0.5) * width} y={y + 26} textAnchor="middle"
        fontFamily={SANS} fontSize={14} fill={COLOR.fg}>{value}</text>
    </g>);
  }
  return <GuideFigure title="GPU column buffers and lane reads" height={568}
    label="Decoded GPU column buffers for four rows. Lanes 0 to 3 of a 32-thread warp read adjacent amount values; customer_id is unread. Status strings use offsets 0, 4, 8, 12, 20 into a contiguous character buffer. A validity bitmap tracks null values."
    caption="A GPU warp coalesces adjacent lane reads into fewer memory transactions, while unprojected columns consume no memory bandwidth. Strings use offset pairs to slice contiguous character bytes. Validity bitmaps track nulls.">
    <Region x={232} y={44} w={512} h={76} label="Warp 0 · lanes 0..3 of 32 shown" tone="gpu" />
    {[0, 1, 2, 3].map((i) => <g key={i}>
      <Label x={startX + (i + 0.5) * cellW} y={28} textAnchor="middle">Row {i}</Label>
      <Arrow id={`coalesce:lane-${i}`} points={[[startX + (i + 0.5) * cellW, 116], [startX + (i + 0.5) * cellW, 152]]} />
    </g>)}
    <Label x={24} y={92}>GPU warp</Label>
    <Label x={24} y={112}>32 lanes · SIMT</Label>
    {cells(['Lane 0', 'Lane 1', 'Lane 2', 'Lane 3'], 76, 'gpu')}
    <Label x={24} y={172}>amount · INT32</Label>
    <Label x={24} y={192}>Scan target</Label>
    {cells(['12', '5', '8', '9'], 152, 'gpu')}
    <path d="M 248 200 V 208 H 728 V 200" fill="none" stroke={TONE.gpu} strokeWidth={1.4} />
    <Label x={488} y={232} textAnchor="middle">16 contiguous bytes · coalesced lane reads</Label>
    <Label x={24} y={276}>customer_id · INT32</Label>
    <Label x={24} y={296}>Unread column</Label>
    {cells(['7', '9', '7', '2'], 256, 'plain', cellW, true)}
    <Label x={488} y={320} textAnchor="middle">Skipped in projection · 0 bytes read from this buffer</Label>
    {sketchLine('layout:strings', 24, 344, 776, 344, {stroke: COLOR.border, dashed: true})}
    <Label x={24} y={388}>status offsets · INT32</Label>
    <Label x={24} y={408}>5 offsets for 4 strings</Label>
    {cells(['0', '4', '8', '12', '20'], 368, 'bridge', 96)}
    <Label x={24} y={452}>status bytes · UTF-8</Label>
    <Label x={24} y={472}>Contiguous characters</Label>
    {[
      {text: 'paid', slice: '[0..4) · 4 B', x: 248, w: 96},
      {text: 'open', slice: '[4..8) · 4 B', x: 344, w: 96},
      {text: 'paid', slice: '[8..12) · 4 B', x: 440, w: 96},
      {text: 'refunded', slice: '[12..20) · 8 B', x: 536, w: 192},
    ].map(({text, slice, x, w}) => <g key={x}>
      <rect x={x} y={432} width={w} height={56} rx={3} fill={TONE.bridge}
        fillOpacity={0.12} stroke={COLOR.border} />
      <text x={x + w / 2} y={454} textAnchor="middle" fontFamily={SANS} fontSize={14} fill={COLOR.fg}>{text}</text>
      <text x={x + w / 2} y={476} textAnchor="middle" fontFamily={SANS} fontSize={11} fill={COLOR.muted}>{slice}</text>
    </g>)}
    <Label x={24} y={532}>Validity bitmap</Label>
    <Label x={24} y={552}>1 bit per row</Label>
    {cells(['1 · valid', '1 · valid', '1 · valid', '1 · valid'], 512, 'gpu')}
  </GuideFigure>;
}

export function StreamSchedule({parallel = false}: {parallel?: boolean}) {
  if (!parallel) {
    return <GuideFigure title="Default inline chunk execution" height={400}
      label="Default chunk execution with source_work_per_attempt set to one. The coordinator submits chunk work on the root CUDA stream and synchronizes that stream before accepting each chunk's output. Partial states are merged on the root stream under the query's grant."
      caption="The default chunk dispatcher runs inline on the coordinator. It accepts each chunk's output after synchronizing its execution stream. This drawing follows chunk work; source prefetch and RAPIDS internal streams are omitted.">
      <Region x={16} y={16} w={768} h={132}
        label="CPU · coordinator on the GPU execution runtime" />
      <Region x={16} y={224} w={768} h={156}
        label="GPU · root CUDA stream · one query grant" labelOffset={320} tone="gpu" />
      <Arrow id="streams:inline-submit" points={[[280, 132], [280, 184], [144, 184], [144, 272]]} />
      <EdgeLabel x={212} y={168} text="submit" />
      <Arrow id="streams:inline-accept" points={[[208, 272], [208, 208], [520, 208], [520, 132]]} dashed />
      <EdgeLabel x={396} y={192} text="synchronize, then accept output" />
      <Arrow id="streams:inline-next" points={[[256, 316], [292, 316]]} />
      <Arrow id="streams:inline-merge" points={[[508, 316], [544, 316]]} />
      <Card x={40} y={60} w={720} h={72} name="Submit and accept each chunk"
        note={['Enqueue work on the root CUDA stream',
          'Wait in cudaStreamSynchronize before accepting the output']} tone="engine" />
      <Card x={40} y={272} w={216} h={88} name="Chunk 1"
        note={['Decode input columns', 'Compute partial state']} tone="gpu" />
      <Card x={292} y={272} w={216} h={88} name="Chunk 2"
        note={['Decode input columns', 'Compute partial state']} tone="gpu" />
      <Card x={544} y={272} w={216} h={88} name="Merge partials"
        note={['Combine completed states', 'Produce the grouped result']} tone="engine" />
    </GuideFigure>;
  }

  return <GuideFigure width={960} minWidth={800} maxWidth={960} height={656}
    label="Two execution lanes in one query, arranged by execution order from left to right. The CPU coordinator dispatches lane jobs and waits for readiness. GPU lane B finishes chunk 2 before lane A finishes chunk 1. The coordinator observes Event 1, commits chunk 1, then observes the already completed Event 2 and commits chunk 2. It submits the final merge on the root CUDA stream."
    caption="Read left to right across the CPU and GPU rows. In this example, chunk 2 completes first; its partial state waits until chunk 1 can be committed. The coordinator calls cudaEventSynchronize before each commit. Positions show an illustrative order without a timing scale. Lane jobs record their events after GPU submissions and return their execution-job permits while GPU work may continue.">
    <Label x={24} y={24}>One admitted query · shared memory grant</Label>
    <Label x={664} y={48}>Execution order</Label>
    <Arrow id="streams:order" points={[[200, 64], [944, 64]]} />

    {[
      {y: 96, owner: 'CPU', name: 'Coordinator', tone: 'plain' as Tone},
      {y: 216, owner: 'GPU', name: 'Lane A stream', tone: 'gpu' as Tone},
      {y: 336, owner: 'GPU', name: 'Lane B stream', tone: 'gpu' as Tone},
      {y: 456, owner: 'GPU', name: 'Root stream', tone: 'gpu' as Tone},
    ].map(({y, owner, name, tone}) => <g key={name}>
      <rect x={16} y={y} width={928} height={88} rx={8} fill={TONE[tone]} opacity={0.05} />
      <text x={24} y={y + 32} fill={COLOR.fg} fontFamily={SANS} fontSize={14} fontWeight={600}>
        {owner}
      </text>
      <Label x={24} y={y + 54}>{name}</Label>
      <line x1={16} y1={y + 104} x2={944} y2={y + 104} stroke={COLOR.border} strokeWidth={1} />
    </g>)}

    <Arrow id="streams:dispatch-a" points={[[292, 180], [292, 260], [360, 260]]} />
    <Arrow id="streams:dispatch-b" points={[[240, 180], [240, 380], [360, 380]]} />
    <Arrow id="streams:wait" points={[[336, 140], [360, 140]]} />
    <Arrow id="streams:commit" points={[[664, 140], [688, 140]]} />
    <Arrow id="streams:ready-a" points={[[600, 260], [640, 260]]} />
    <Arrow id="streams:ready-b" points={[[512, 380], [544, 380]]} />
    <Arrow id="streams:observe-a" points={[[648, 252], [648, 180]]} dashed />
    <Arrow id="streams:observe-b" points={[[552, 372], [552, 312], [752, 312], [752, 180]]} dashed />
    <Arrow id="streams:merge-submit" points={[[816, 140], [856, 140], [856, 460]]} />

    <Card x={200} y={100} w={136} h={80} name="Dispatch"
      note="Two lane jobs" />
    <Card x={360} y={100} w={304} h={80} name="Wait for chunk 1"
      note="cudaEventSynchronize(E1)" tone="engine" />
    <Card x={688} y={100} w={128} h={80} name="Commit"
      note={['1, then 2', 'E2 already ready']} />
    <Card x={360} y={220} w={240} h={80} name="Chunk 1"
      note={['Decode · filter', 'Partial aggregate']} tone="gpu" />
    <Card x={360} y={340} w={152} h={80} name="Chunk 2"
      note={['Decode · filter', 'Partial aggregate']} tone="gpu" />
    <Card x={832} y={460} w={112} h={80} name="Merge"
      note="Partials" tone="gpu" />

    {[{x: 648, y: 260, event: 'E1 ready'}, {x: 552, y: 380, event: 'E2 ready'}].map(({x, y, event}) =>
      <g key={event}>
        <path d={`M ${x} ${y - 8} L ${x + 8} ${y} L ${x} ${y + 8} L ${x - 8} ${y} Z`}
          fill={COLOR.surface} stroke={TONE.gpu} strokeWidth={2} />
        <Label x={x} y={y + 32} textAnchor="middle">{event}</Label>
      </g>)}

    <Arrow id="streams:legend-order" points={[[24, 596], [72, 596]]} />
    <Label x={84} y={600}>Dispatch / next step</Label>
    <Arrow id="streams:legend-ready" points={[[344, 596], [392, 596]]} dashed />
    <Label x={404} y={600}>Readiness observed</Label>
    <path d="M 704 588 L 712 596 L 704 604 L 696 596 Z"
      fill={COLOR.surface} stroke={TONE.gpu} strokeWidth={2} />
    <Label x={724} y={600}>GPU event complete</Label>
    <Label x={24} y={640}>Auxiliary permits and stream leases stay with the compiled run.</Label>
  </GuideFigure>;
}

export function ResultEgress() {
  return <GuideFigure title="Result egress and host synchronization" height={568}
    label="Result egress for one native fragment. On the host runtime, the first poll of the DataFusion result stream runs admission and the native schedule. Each chunk then takes one GPU pool job on algeon-gpu-{d}. The job sizes the chunk to at most 4 MiB of Arrow buffers and reserves that estimate in DataFusion's MemoryPool, enqueues cudaMemcpyAsync on the execution stream after the upstream kernels, waits in cudaStreamSynchronize, wraps the host buffers as Arrow arrays and shrinks the reservation to the batch size. The stream yields the RecordBatch, the next poll moves the next chunk, and the last chunk ends the stream."
    caption="Each chunk is one GPU pool job. The job reserves its size estimate before the copy, enqueues the copy on the execution stream and holds its worker in cudaStreamSynchronize until the copy completes. The result stream keeps one chunk reservation at a time.">

    <Region x={16} y={16} w={768} h={144} label="Host runtime · DataFusion result stream" />
    <Region x={16} y={216} w={768} h={144} label="GPU pool · one algeon-gpu-{d} job per chunk" labelOffset={256} />
    <Region x={16} y={416} w={768} h={136} label="GPU VRAM · execution stream" tone="gpu" />
    <Arrow id="egress:next-poll" points={[[488, 104], [312, 104]]} dashed />
    <EdgeLabel x={400} y={88} text="next poll" />
    <Arrow id="egress:submit" points={[[136, 144], [136, 264]]} />
    <EdgeLabel x={152} y={192} text="pool job" anchor="start" />
    <Arrow id="egress:reserved" points={[[248, 304], [288, 304]]} />
    <Arrow id="egress:copied" points={[[512, 304], [552, 304]]} />
    <Arrow id="egress:yield" points={[[664, 264], [664, 144]]} />
    <EdgeLabel x={680} y={192} text="yield batch" anchor="start" />
    <Arrow id="egress:enqueue" points={[[368, 344], [368, 464]]} />
    <EdgeLabel x={352} y={396} text="enqueue" anchor="end" />
    <Arrow id="egress:complete" points={[[432, 464], [432, 344]]} />
    <EdgeLabel x={448} y={396} text="copy complete" anchor="start" />
    <Arrow id="egress:stream-order" points={[[248, 504], [288, 504]]} />
    <Card x={24} y={64} w={288} h={80} name="DataFusion poll"
      note={['First poll: admission, native schedule', 'Each chunk: one pool job']} />
    <Card x={488} y={64} w={288} h={80} name="RecordBatch to the caller" tone="bridge"
      note={['Holds one chunk reservation', 'Last chunk ends the stream']} />
    <Card x={24} y={264} w={224} h={80} name="1. Size and reserve"
      note={['Estimate ≤ 4 MiB of Arrow', 'try_grow in MemoryPool']} />
    <Card x={288} y={264} w={224} h={80} name="2. Copy and wait"
      note={['Enqueue cudaMemcpyAsync', 'cudaStreamSynchronize']} />
    <Card x={552} y={264} w={224} h={80} name="3. Build the batch"
      note={['Wrap host buffers as Arrow', 'Shrink reservation to size']} />
    <Card x={24} y={464} w={224} h={72} name="Result columns" tone="gpu"
      note="Written by upstream kernels" />
    <Card x={288} y={464} w={224} h={72} name="Device-to-host copy" tone="gpu"
      note={['Starts after upstream kernels', 'Crosses PCIe into host buffers']} />
  </GuideFigure>;
}

export function GpuUdtfFlow() {
  return <GuideFigure title="PageRank inside a SQL query" height={824}
    label="GPU UDTF execution flow for PageRank within a SQL query. The relational pipeline on the left filters edge columns in cuDF. A zero-copy read lease passes edge buffers to cuGraph for graph construction and PageRank. The result is copied in device memory into independently owned cuDF columns for downstream filtering, joining, and aggregation, while native graph memory is settled and freed."
      caption="Relational SQL operators (left) and the graph algorithm (right) share VRAM under one admission grant. Edge inputs are borrowed via a zero-copy read lease; algorithm results are copied into owned cuDF columns so native graph memory can be freed for downstream relational operators.">

    <Region x={8} y={8} w={784} h={800}
      label="GPU VRAM · one admitted attempt · shared memory cap" tone="engine" />
    <Region x={24} y={52} w={336} h={740} label="Relational pipeline · cuDF" tone="gpu" />
    <Region x={440} y={52} w={336} h={740} label="Algorithm runtime · cuGraph UDTF" tone="graph" />
    <Arrow id="udtf:lease" points={[[192, 184], [192, 224], [608, 224], [608, 264]]} />
    <EdgeLabel x={400} y={208} text="zero-copy read lease" />
    <Arrow id="udtf:pagerank" points={[[608, 352], [608, 392]]} />
    <Arrow id="udtf:copy" points={[[608, 480], [608, 520], [192, 520], [192, 560]]} />
    <EdgeLabel x={400} y={504} text="device memory copy" />
    <Arrow id="udtf:downstream" points={[[192, 648], [192, 688]]} />
    <Arrow id="udtf:settle" points={[[344, 604], [608, 604], [608, 688]]} dashed />
    <EdgeLabel x={624} y={660} text="copy complete" anchor="start" />
    <Card x={40} y={96} w={304} h={88} name="1. Edge subquery (citations)"
      note={['WHERE citation_year >= 2020', 'cuDF owns src, dst columns']} tone="gpu" />
    <Label x={40} y={268}>PreparedTableExport</Label>
    <Label x={40} y={292}>Borrows the edge buffers.</Label>
    <Card x={456} y={264} w={304} h={88} name="2. Graph construction"
      note={['Validate column types and active ranges', 'Build CSR representation in VRAM']} tone="graph" />
    <Card x={456} y={392} w={304} h={88} name="3. PageRank kernel"
      note={['Iterative power iteration on GPU', 'Owns native vertex and score arrays']} tone="graph" />
    <Card x={40} y={560} w={304} h={88} name="4. UDTF result (ranked)"
      note={['to_dataframe → owned cuDF columns', 'vertex, value · SQL schema normalization']} tone="gpu" />
    <Card x={40} y={688} w={304} h={88} name="5. Downstream SQL"
      note={['Filter: value > 0.0001', 'Join: papers ON paper_id = vertex', 'Aggregate: GROUP BY field']} tone="gpu" />
    <Card x={456} y={688} w={304} h={88} name="Settle and free"
      note={['Free PageRank arrays; release read lease', 'Return VRAM to query pool', 'Capacity available for the hash join']} />
  </GuideFigure>;
}

export function AttemptLifetime() {
  return <GuideFigure title="Query attempt lifetime" width={824} height={560}
    label="Attempt lifecycle across three thread contexts. On the host runtime, the capability proof precedes execution and the attempt waits in the admission queue on the caller's task. On the device's algeon-gpu-{d} runtime, an envelope lifecycle job takes the lane, root stream and allocation domain, one QueryExecution job runs the whole GPU schedule on one worker, and each poll submits one job that materializes a chunk of at most 4 MiB, which the host runtime receives as an Arrow batch. The next poll loops back to the materialize job. When the stream reaches EOF, is dropped or is cancelled, the cleanup thread synchronizes the attempt's streams and returns capacity."
    caption="The first poll waits for admission on the caller's task, then runs the whole GPU schedule as one pool job. Each later poll submits one job that moves at most 4 MiB to the host. Release follows the terminal outcome and the cleanup thread's synchronize.">

    <Region x={8} y={16} w={784} h={140} label="Host runtime · caller task" tone="plan" />
    <Region x={8} y={224} w={784} h={140} label="algeon-gpu-{d} runtime" labelOffset={280} tone="gpu" />
    <Region x={8} y={432} w={784} h={112} label="Cleanup thread" tone="engine" />
    <Arrow id="attempt:proof-queue" points={[[216, 108], [272, 108]]} />
    <Arrow id="attempt:queue-envelope" points={[[380, 148], [380, 192], [132, 192], [132, 272]]} />
    <EdgeLabel x={256} y={176} text="admitted" />
    <Arrow id="attempt:envelope-execute" points={[[240, 312], [288, 312]]} />
    <Arrow id="attempt:execute-materialize" points={[[484, 312], [532, 312]]} />
    <Arrow id="attempt:materialize-batch" points={[[624, 272], [624, 148]]} />
    <Arrow id="attempt:next-poll" points={[[712, 148], [712, 272]]} dashed />
    <EdgeLabel x={696} y={192} text="next poll" anchor="end" />
    <Arrow id="attempt:release" points={[[760, 108], [808, 108], [808, 488], [760, 488]]} />
    <EdgeLabel x={792} y={404} text="EOF · drop · cancel" anchor="end" />
    <Card x={24} y={68} w={192} h={80} name="Capability proof" note="Before execution" tone="plan" />
    <Card x={272} y={68} w={216} h={80} name="Admission queue"
      note={['Wait for capacity', "On the caller's task"]} tone="plan" />
    <Card x={584} y={68} w={176} h={80} name="Arrow batch" note="≤ 4 MiB per poll" tone="bridge" />
    <Card x={24} y={272} w={216} h={80} name="Envelope"
      note={['Lifecycle job', 'Lane, root stream, domain']} tone="gpu" />
    <Card x={288} y={272} w={196} h={80} name="Execute"
      note={['QueryExecution job', 'One worker']} tone="gpu" />
    <Card x={532} y={272} w={228} h={80} name="Materialize chunk" note="One job per poll" tone="gpu" />
    <Card x={424} y={464} w={336} h={64} name="Synchronize, return capacity"
      note="After the attempt's terminal outcome" tone="engine" />
  </GuideFigure>;
}

export function AllocationPath() {
  return <GuideFigure title="Query allocation path" height={312}
    label="GPU operations allocate through the query's DeviceMemoryResource. The query resource checks the immutable cap. Accepted allocations use the device RMM resource and GPU storage; an allocation beyond the cap returns a typed failure."
    caption="The grant authorizes a byte limit. Physical buffers are obtained as operations allocate. Unused grant capacity is still committed in the device ledger.">

    <Arrow id="allocation:request" points={[[232, 72], [296, 72]]} />
    <Arrow id="allocation:accept" points={[[504, 72], [568, 72]]} />
    <EdgeLabel x={536} y={56} text="fits" />
    <Arrow id="allocation:storage" points={[[672, 112], [672, 208]]} />
    <Arrow id="allocation:reject" points={[[400, 112], [400, 208]]} dashed />
    <EdgeLabel x={416} y={164} text="over cap" anchor="start" />
    <Card x={24} y={32} w={208} h={80} name="GPU operation" note="cuDF / graph / vector / ML" tone="gpu" />
    <Card x={296} y={32} w={208} h={80} name="Query resource" note="Enforce the immutable cap" tone="engine" />
    <Card x={568} y={32} w={208} h={80} name="RMM resource" note="Reuse or allocate" tone="bridge" />
    <Card x={296} y={208} w={208} h={80} name="Typed failure" note="Allocation rejected" tone="reject" />
    <Card x={568} y={208} w={208} h={80} name="GPU storage" note="Columns and scratch" tone="gpu" />
  </GuideFigure>;
}

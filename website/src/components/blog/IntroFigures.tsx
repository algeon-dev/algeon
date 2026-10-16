import React from 'react';
import {
  BackedLabel,
  Box,
  COLOR,
  COMPACT_FONTS,
  EDGE,
  Figure,
  LABEL,
  TONE,
  measureNode,
  textWidth,
} from '@site/src/components/design/Diagram';
import type {BoxFonts, DiagramNode, FontSpec} from '@site/src/components/design/Diagram';
import {SANS, sketchHead, sketchLine, sketchPath, sketchRoundRect} from '@site/src/components/design/sketch';

/**
 * The introduction post's bespoke figures: the export-and-notebook detour
 * against one SQL query, and the agent's investigation loop through the
 * application's ontology, its backend service, Algeon, and the lakehouse.
 * Built from the Design diagrams' box, tone, figure, and sketch primitives so
 * the blog reads as the same hand-drawn visual system.
 */

const HEAD = {size: 11, track: 0.08};
const NOTE = {size: 12.5, lh: 17};
const INK = COLOR.fg;
const EDGE_W = 1.4;
const GROUP_RADIUS = 18;

interface Placed {
  x: number;
  y: number;
  w: number;
  h: number;
}

export function QueryInterfaceFigure({caption}: {caption?: React.ReactNode}) {
  const panelW = 344;
  const label = (x: number, y: number, value: string, size = 13, strong = false, muted = false) => (
    <text x={x} y={y} fontFamily={SANS} fontSize={size} fontWeight={strong ? 600 : 400}
      fill={muted ? COLOR.muted : COLOR.fg}>{value}</text>
  );
  const eyebrow = (x: number, y: number, value: string) => (
    <text x={x} y={y} fontFamily={SANS} fontSize={10.5} letterSpacing="0.09em"
      fill={COLOR.muted}>{value}</text>
  );
  const arrow = (x: number, y: number, direction: 'up' | 'down' | 'right') => (
    <path d="M -4 -5 L 0 0 L 4 -5" fill="none" stroke={COLOR.muted} strokeWidth={1.3}
      strokeLinecap="round" strokeLinejoin="round"
      transform={`translate(${x} ${y}) rotate(${direction === 'up' ? 180 : direction === 'right' ? -90 : 0})`} />
  );
  const vertical = (x: number, from: number, to: number, returning = false) => (
    <g>
      <path d={`M ${x} ${from} V ${to}`} fill="none" stroke={COLOR.muted}
        strokeWidth={1.2} strokeDasharray={returning ? '4 4' : undefined} />
      {arrow(x, to, to > from ? 'down' : 'up')}
    </g>
  );
  const box = (x: number, y: number, w: number, h: number) => (
    <rect x={x} y={y} width={w} height={h} rx={8}
      fill={COLOR.surface} stroke={COLOR.border} strokeWidth={1.2} />
  );
  const source = (x: number, y: number, w: number, compact = false) => (
    <g>
      <rect x={x} y={y} width={w} height={72} rx={8}
        fill="var(--ag-dg-surface-2)" stroke={COLOR.border} />
      <g transform={`translate(${x + 16} ${y + 22})`} fill="none" stroke={COLOR.muted} strokeWidth={1.2}>
        <rect width={24} height={28} rx={2} />
        <path d="M 0 8 H 24 M 0 18 H 24 M 8 8 V 28 M 16 8 V 28" />
      </g>
      {label(x + 52, y + 30, 'Existing Parquet tables', 16, true)}
      {label(x + 52, y + 51, compact ? 'Prepared by batch jobs' : 'Prepared by scheduled lakehouse jobs', 12.5, false, true)}
    </g>
  );
  const service = (x: number, title: string[], note: string) => (
    <g>
      {box(x, 240, 104, 88)}
      {title.map((line, i) => <React.Fragment key={line}>
        {label(x + 12, 266 + i * 18, line, 14, true)}
      </React.Fragment>)}
      {label(x + 12, 310, note, 11.5, false, true)}
    </g>
  );
  const panel = (x: number, y: number, inQuery: boolean) => (
    <g transform={`translate(${x} ${y})`}>
      {eyebrow(0, 12, inQuery ? 'COMPOSED IN THE QUERY' : 'COMPOSED IN THE SERVICE')}
      {label(0, 40, inQuery ? 'Change the SQL statement' : 'Extend the backend workflow', 18, true)}
      {label(0, 59, inQuery ? 'Within the supported tables and functions' : 'When the existing endpoints cannot express it', 12, false, true)}
      {inQuery ? <>
        <rect x={0} y={224} width={panelW} height={204} rx={8}
          fill={TONE.engine} fillOpacity={0.05} stroke={TONE.engine} strokeWidth={1.3} />
        {vertical(72, 156, 224)}
        {vertical(252, 224, 156, true)}
        {eyebrow(84, 185, 'FLIGHT SQL')}
        {eyebrow(84, 201, 'OR EMBEDDED')}
        {eyebrow(264, 185, 'ROWS /')}
        {eyebrow(264, 201, 'ERRORS')}
        {label(16, 252, 'Algeon', 18, true)}
        {label(104, 252, 'one query plan', 12.5, false, true)}
        {[104, 216].map(x0 => <g key={x0}>
          <path d={`M ${x0} 308 H ${x0 + 24}`} stroke={COLOR.muted} strokeWidth={1.2} />
          {arrow(x0 + 24, 308, 'right')}
        </g>)}
        {[
          {x: 16, lines: ['Filter', '+ join']},
          {x: 128, lines: ['Graph +', 'vector']},
          {x: 240, lines: ['Score', '+ rank']},
        ].map(stage => <g key={stage.x}>
          {box(stage.x, 276, 88, 64)}
          {stage.lines.map((line, i) => <React.Fragment key={line}>
            {label(stage.x + 12, 302 + i * 19, line, 14, true)}
          </React.Fragment>)}
        </g>)}
        <path d="M 16 360 H 328" stroke={COLOR.border} />
        {label(16, 383, 'Native stages keep intermediates on GPU', 12.5, false, true)}
        {label(16, 405, 'One GPU · memory cap per query attempt', 12.5, false, true)}
        {vertical(172, 468, 428)}
        {eyebrow(184, 452, 'SCAN')}
      </> : <>
        {[52, 172, 292].map((cx, i) => <g key={cx}>
          {vertical(cx - 8, 156, 240)}
          {vertical(cx + 8, 240, 156, true)}
          {eyebrow(cx + 20, 204, i === 0 ? 'SQL' : 'RPC')}
        </g>)}
        {service(0, ['SQL', 'engine'], 'eligible rows')}
        {service(120, ['Graph +', 'vector'], 'candidates')}
        {service(240, ['Model', 'service'], 'scores')}
        {vertical(52, 468, 328)}
        {eyebrow(64, 448, 'SCAN')}
        {eyebrow(120, 372, 'BETWEEN CALLS')}
        {label(120, 397, 'Intermediate results return', 12.5, false, true)}
        {label(120, 416, 'to the application service.', 12.5, false, true)}
      </>}
      {box(0, 80, panelW, 76)}
      {label(16, 106, inQuery ? 'Agent + application service' : 'Application service', 16, true)}
      {label(16, 127, inQuery ? 'Domain terms, query formulation' : 'Query templates, call sequencing', 12.5, false, true)}
      {label(16, 145, inQuery ? 'and evaluation of returned rows' : 'and exchange of intermediate data', 12.5, false, true)}
    </g>
  );
  const header = (compact = false) => (
    <g>
      {eyebrow(16, 16, 'CAMPAIGN PLANNING')}
      {(compact ? ['Which 5,000 shoppers should', 'receive a Pokémon offer?']
        : ['Which 5,000 shoppers should receive a Pokémon offer?']).map((line, i) => (
        <React.Fragment key={line}>{label(16, 44 + i * 26, line, 21, true)}</React.Fragment>
      ))}
      {!compact && label(16, 69, 'Eligibility filters + graph affinity + taste similarity + model scoring', 13, false, true)}
    </g>
  );

  return (
    <Figure caption={caption} svgW={744} svgH={644} minW={720} center
      title="Composing an analytical workflow in a service or in SQL"
      ariaLabel="A campaign needs 5,000 shoppers selected using eligibility filters, graph affinity, vector similarity and model scoring. In the service composition, the application coordinates SQL, graph or vector services and a model service, exchanging intermediate results between calls. In the SQL composition, the agent and application express the workflow in a statement; Algeon plans the operators and bounds device memory. Supported native stages retain intermediates on the GPU. Both approaches read existing Parquet data prepared by batch jobs."
      narrow={{svgW: 376, svgH: 1224, minW: 260, children: <>
        {header(true)}
        {panel(16, 100, false)}
        {source(16, 568, panelW, true)}
        {panel(16, 672, true)}
        {source(16, 1140, panelW, true)}
      </>}}>
      {header()}
      <path d="M 16 84 H 728" stroke={COLOR.border} />
      {panel(16, 96, false)}
      {panel(384, 96, true)}
      {source(16, 564, 712)}
    </Figure>
  );
}

const LEFT_NODES: DiagramNode[] = [
  {kind: 'warehouse', name: 'Select the rows and columns', note: 'one tenant · one quarter', tone: 'plan'},
  {kind: 'export job', name: 'Write data to a bucket', tone: 'plain'},
  {kind: 'notebook', name: 'Load, clean, reshape', note: 'ids · nulls · duplicates · casts', tone: 'plain'},
  {kind: 'gpu library', name: 'cu* call', tone: 'gpu'},
  {kind: 'export again', name: 'Write the result', tone: 'plain'},
  {kind: 'another system', name: 'Import and join back', tone: 'plan'},
];
const LEFT_STEPS = ['export', 'download', 'in process', 'write', 'import'];
const LEFT_HEAD = 'TODAY: THE DETOUR';
const LEFT_NOTE = ['each step relies on ad-hoc ETL scripts', 'written for a small fraction of the dataset'];

const RIGHT_NODES: DiagramNode[] = [
  {kind: 'sql', name: 'Select and clean the input', note: 'WHERE · JOIN · CAST · DISTINCT', tone: 'plan'},
  {
    kind: 'table function', name: 'cugraph_*(edges => (SELECT …))',
    note: 'documented input and output schema', tone: 'gpu',
  },
  {kind: 'sql', name: 'Join and aggregate the output', note: 'rows back to the caller', tone: 'plan'},
];
const RIGHT_HEAD = 'WITH ALGEON: ONE PLAN';
const RIGHT_GROUP = 'one SQL query · one process';
const RIGHT_NOTE = ['the planner owns scan, pushdown,', 'join order and GPU placement'];

function Spine({id, a, b, cx, label}: {id: string; a: Placed; b: Placed; cx: number; label?: string}) {
  const top = a.y + a.h;
  const edge = {stroke: INK, strokeWidth: EDGE_W};
  return (
    <g>
      {sketchLine(`glue:spine:${id}`, cx, top + 2, cx, b.y - 2, edge)}
      {sketchHead(`glue:spine:${id}:head`, cx, b.y, 'down', edge)}
      {label ? (
        <text x={cx + 12} y={(top + b.y) / 2 + LABEL.size * 0.36} fontSize={LABEL.size}
          fontFamily={SANS} letterSpacing={`${LABEL.track ?? 0}em`} fill={COLOR.muted}>
          {label}
        </text>
      ) : null}
    </g>
  );
}

export function GlueVsPlanFigure({caption}: {caption?: React.ReactNode}) {
  const fonts: BoxFonts = COMPACT_FONTS;
  const ls = LEFT_NODES.map((n) => measureNode(n, fonts));
  const rs = RIGHT_NODES.map((n) => measureNode(n, fonts));
  const leftW = Math.max(...ls.map((s) => s.w));
  const rightW = Math.max(...rs.map((s) => s.w));
  const HEADER_H = 14 + 12 + 2 * NOTE.lh + 18;
  const GAP = 42;
  const COL_GAP = 88;
  const GROUP_PAD = 18;
  const GROUP_HEAD = 36;
  const GROUP_FOOT = 18;

  const leftX = EDGE + 8;
  const top = EDGE + HEADER_H;
  const lp: Placed[] = [];
  let y = top;
  LEFT_NODES.forEach((_, i) => {
    if (i > 0) y += GAP;
    lp.push({x: leftX, y, w: leftW, h: ls[i].h});
    y += ls[i].h;
  });
  const leftBottom = y;

  const groupX = leftX + leftW + COL_GAP;
  const groupW = rightW + 2 * GROUP_PAD;
  const innerH = rs.reduce((acc, s) => acc + s.h, 0) + GAP * (RIGHT_NODES.length - 1);
  const groupH = GROUP_HEAD + innerH + GROUP_FOOT;
  // Both columns start at the same height so the eye compares step one with step one.
  const groupY = top;
  const rp: Placed[] = [];
  let ry = groupY + GROUP_HEAD;
  RIGHT_NODES.forEach((_, i) => {
    if (i > 0) ry += GAP;
    rp.push({x: groupX + GROUP_PAD, y: ry, w: rightW, h: rs[i].h});
    ry += rs[i].h;
  });

  const svgW = groupX + groupW + 8 + EDGE;
  const svgH = leftBottom + 8 + EDGE;
  const lcx = leftX + leftW / 2;
  const rcx = groupX + GROUP_PAD + rightW / 2;

  // At 90% the figure and its caption fit below the navbar on an 800 px tall screen.
  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={620} maxW={Math.round(svgW * 0.9)} center
      ariaLabel="Left: export, notebook, GPU call, and import as separate steps. Right: one SQL query with the algorithm as a UDTF.">
      <text x={leftX} y={EDGE + 14} fontSize={HEAD.size} letterSpacing={`${HEAD.track}em`}
        fill={COLOR.muted}>
        {LEFT_HEAD}
      </text>
      <text x={groupX} y={EDGE + 14} fontSize={HEAD.size} letterSpacing={`${HEAD.track}em`}
        fill={COLOR.muted}>
        {RIGHT_HEAD}
      </text>
      <rect x={groupX} y={groupY} width={groupW} height={groupH} rx={GROUP_RADIUS}
        fill={TONE.gpu} opacity={0.06} />
      <g opacity={0.7}>
        {sketchRoundRect('glue:group', groupX, groupY, groupW, groupH, GROUP_RADIUS,
          {stroke: TONE.gpu, strokeWidth: 1.2, dashed: true})}
      </g>
      <text x={groupX + 14} y={groupY + 20} fontSize={LABEL.size}
        letterSpacing={`${LABEL.track ?? 0}em`} fill={COLOR.muted}>
        {RIGHT_GROUP}
      </text>
      <g opacity={0.6}>
        {sketchLine('glue:divider', leftX + leftW + COL_GAP / 2, EDGE + 4, leftX + leftW + COL_GAP / 2, svgH - EDGE - 4,
          {stroke: COLOR.border, strokeWidth: 1, dashed: true, roughness: 0.5})}
      </g>
      {lp.slice(1).map((b, i) => (
        <Spine key={`l-${i}`} id={`l${i}`} a={lp[i]} b={b} cx={lcx} label={LEFT_STEPS[i]} />
      ))}
      {rp.slice(1).map((b, i) => (
        <Spine key={`r-${i}`} id={`r${i}`} a={rp[i]} b={b} cx={rcx} />
      ))}
      {LEFT_NOTE.map((line, i) => (
        <text key={`ln-${i}`} x={leftX} y={EDGE + 14 + 12 + NOTE.lh * 0.8 + i * NOTE.lh} fontSize={NOTE.size}
          fill={COLOR.muted}>
          {line}
        </text>
      ))}
      {RIGHT_NOTE.map((line, i) => (
        <text key={`rn-${i}`} x={groupX} y={EDGE + 14 + 12 + NOTE.lh * 0.8 + i * NOTE.lh} fontSize={NOTE.size}
          fill={COLOR.muted}>
          {line}
        </text>
      ))}
      {LEFT_NODES.map((n, i) => (
        <Box key={n.name} node={n} x={lp[i].x} y={lp[i].y} w={lp[i].w} h={lp[i].h} fonts={fonts} />
      ))}
      {RIGHT_NODES.map((n, i) => (
        <Box key={n.name} node={n} x={rp[i].x} y={rp[i].y} w={rp[i].w} h={rp[i].h} fonts={fonts} />
      ))}
    </Figure>
  );
}

/* ====================== AgentLoopFigure ====================== */

const LOOP_AGENT: DiagramNode = {
  kind: 'agentic loop', name: 'Agent and human',
  note: ['hypothesis → question', 'reads rows, asks again'], tone: 'plain',
};
const LOOP_SEMANTIC: DiagramNode = {
  kind: 'ontology', name: 'Semantic layer',
  note: ['terms → tables · keys · joins', 'application-owned'], tone: 'engine',
};
const LOOP_BACKEND: DiagramNode = {
  kind: 'your service', name: 'Backend service',
  note: ['embeds Algeon · Flight SQL', 'sessions · policy · retries'], tone: 'plan',
};
const LOOP_ALGEON: DiagramNode = {
  kind: 'GPU query engine', name: 'Algeon',
  note: ['plan · admit · run on the GPU', 'graph · vector · ML UDTFs'], tone: 'gpu',
};
const LOOP_LAKE: DiagramNode = {
  kind: 'world state · physical store', name: 'Lakehouse',
  note: ['Parquet · disk or object storage', 'tables · views · workspace'], tone: 'lake',
};
const LOOP_OLAP: DiagramNode = {
  kind: 'already on the lake', name: 'OLAP engine',
  note: ['Trino · StarRocks · ClickHouse', 'dashboards · BI'], tone: 'plain',
};
const LOOP_SEMANTIC_ROUTED: DiagramNode = {
  ...LOOP_SEMANTIC,
  note: ['terms → tables · keys · joins', 'functions → engine', 'application-owned'],
};
const LOOP_BAND = 'APPLICATION · AGENT, ONTOLOGY, SERVICE';
const LOOP_RETURN = 'rows and typed diagnostics feed the next hypothesis';

/**
 * The investigation loop as a ring: the agent's question passes through the
 * application's semantic layer and backend service into Algeon, which reads
 * the lakehouse and returns rows plus structured diagnostics to the agent.
 * With `olap`, an existing OLAP engine stands beside Algeon under the same
 * service and reads the same lake; the semantic layer then also records which
 * functions each engine offers, since the agent picks the engine before it
 * writes the query. Each engine edge then carries the query down and the rows
 * back, which is one Flight SQL call, so the return rail is not drawn.
 */
export function AgentLoopFigure({caption, olap = false}: {caption?: React.ReactNode; olap?: boolean}) {
  const fonts: BoxFonts = COMPACT_FONTS;
  const semanticNode = olap ? LOOP_SEMANTIC_ROUTED : LOOP_SEMANTIC;
  const [agent, semantic, backend, algeon, lake, olapEngine] =
    [LOOP_AGENT, semanticNode, LOOP_BACKEND, LOOP_ALGEON, LOOP_LAKE, LOOP_OLAP].map((n) => measureNode(n, fonts));
  const BAND_PAD = 18;
  const BAND_HEAD = 34;
  const BAND_FOOT = 16;
  const V_GAP = olap ? 96 : 64;
  const RAIL_DROP = 34;
  const R = 10;
  const label01 = ['question in', 'domain terms'];
  const label12 = ['SQL over', 'registered relations'];
  // With `olap` each engine edge is one call: the query goes down, rows come
  // back on the same edge, so no return rail circles the figure.
  const labelAlgeon = olap ? ['graph · vector · ML queries', 'rows and typed', 'diagnostics back'] : ['one query'];
  const labelOlap = ['everything else', 'rows back'];
  const gapFor = (lines: string[]) =>
    Math.max(72, Math.ceil(Math.max(...lines.map((l) => textWidth(l, LABEL)))) + 48);

  // Row 1: the application-owned chain, left to right, inside one band.
  const bandY = EDGE + 4;
  const row1Top = bandY + BAND_HEAD;
  const row1H = Math.max(agent.h, semantic.h, backend.h);
  const x0 = EDGE + 8 + BAND_PAD;
  // With the OLAP engine under the agent, row 2 needs room for its arrow and
  // label into the lakehouse, whichever of the two columns is wider.
  const row2Gap = olap ? (olapEngine.w - agent.w) / 2 + (lake.w - semantic.w) / 2 + 112 : 0;
  const x1 = x0 + agent.w + Math.max(gapFor(label01), row2Gap);
  const x2 = x1 + semantic.w + gapFor(label12);
  const place = (x: number, s: {w: number; h: number}) =>
    ({x, y: row1Top + (row1H - s.h) / 2, w: s.w, h: s.h, cx: x + s.w / 2, cy: row1Top + row1H / 2});
  const A = place(x0, agent);
  const S = place(x1, semantic);
  const B = place(x2, backend);
  const bandX = x0 - BAND_PAD;
  const bandW = B.x + B.w + BAND_PAD - bandX;
  const bandH = BAND_HEAD + row1H + BAND_FOOT;

  // Row 2: Algeon under the service, the lakehouse under the semantic layer,
  // and with `olap` the OLAP engine under the agent so both engines flank the lake.
  const row2Top = bandY + bandH + V_GAP;
  const row2H = Math.max(algeon.h, lake.h, olap ? olapEngine.h : 0);
  const G = {x: B.cx - algeon.w / 2, y: row2Top + (row2H - algeon.h) / 2, w: algeon.w, h: algeon.h};
  const L = {x: S.cx - lake.w / 2, y: row2Top + (row2H - lake.h) / 2, w: lake.w, h: lake.h};
  const O = {x: A.cx - olapEngine.w / 2, y: row2Top + (row2H - olapEngine.h) / 2, w: olapEngine.w, h: olapEngine.h};
  const Gcx = G.x + G.w / 2;
  const row2Cy = row2Top + row2H / 2;

  const railY = row2Top + row2H + RAIL_DROP;
  const svgH = olap ? row2Top + row2H + 8 + EDGE : railY + LABEL.lh + 8 + EDGE;
  // The Algeon edge labels sit to the right of the band when they are long.
  const algeonLabelW = Math.ceil(Math.max(...labelAlgeon.map((l) => textWidth(l, LABEL))));
  const svgW = Math.max(bandX + bandW, G.x + G.w, B.cx + 12 + algeonLabelW) + 8 + EDGE;
  const edge = {stroke: INK, strokeWidth: EDGE_W};
  const rail = [
    `M ${Gcx} ${G.y + G.h}`,
    `L ${Gcx} ${railY - R}`,
    `Q ${Gcx} ${railY} ${Gcx - R} ${railY}`,
    `L ${A.cx + R} ${railY}`,
    `Q ${A.cx} ${railY} ${A.cx} ${railY - R}`,
    `L ${A.cx} ${A.y + A.h + 1}`,
  ].join(' ');
  // The service's second edge drops into the gap between the rows, runs left
  // above the lakehouse, and enters the OLAP engine from above.
  const elbowY = bandY + bandH + V_GAP / 2 + 6;
  const elbowX = B.cx - 40;
  const olapEdge = [
    `M ${elbowX} ${B.y + B.h + 2}`,
    `L ${elbowX} ${elbowY - R}`,
    `Q ${elbowX} ${elbowY} ${elbowX - R} ${elbowY}`,
    `L ${A.cx + R} ${elbowY}`,
    `Q ${A.cx} ${elbowY} ${A.cx} ${elbowY + R}`,
    `L ${A.cx} ${O.y - 2}`,
  ].join(' ');

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={640} center
      ariaLabel={olap
        ? "The agent's question passes through the application's semantic layer, which records each engine's tables and functions, and the backend service submits the query: graph, vector, and ML queries go to Algeon and everything else to the existing OLAP engine. Both engines read the lakehouse, and rows and diagnostics return to the agent for the next hypothesis."
        : "The agent's question passes through the application's semantic layer and backend service into Algeon, which reads the lakehouse and returns rows and diagnostics to the agent for the next hypothesis."}>
      {/* the application band */}
      <rect x={bandX} y={bandY} width={bandW} height={bandH} rx={GROUP_RADIUS} fill={TONE.plan} opacity={0.05} />
      <g opacity={0.7}>
        {sketchRoundRect('loop:band', bandX, bandY, bandW, bandH, GROUP_RADIUS,
          {stroke: TONE.plan, strokeWidth: 1.2, dashed: true})}
      </g>
      <text x={bandX + 16} y={bandY + 20} fontSize={HEAD.size} letterSpacing={`${HEAD.track}em`}
        fill={COLOR.muted}>
        {LOOP_BAND}
      </text>
      {/* row 1 arrows */}
      {sketchLine('loop:a-s', A.x + A.w + 2, A.cy, S.x - 2, A.cy, edge)}
      {sketchHead('loop:a-s:head', S.x, A.cy, 'right', edge)}
      {label01.map((line, j) => (
        <BackedLabel key={`l01-${j}`} x={(A.x + A.w + S.x) / 2}
          y={A.cy - 14 - (label01.length - 1 - j) * LABEL.lh} text={line} />
      ))}
      {sketchLine('loop:s-b', S.x + S.w + 2, S.cy, B.x - 2, S.cy, edge)}
      {sketchHead('loop:s-b:head', B.x, S.cy, 'right', edge)}
      {label12.map((line, j) => (
        <BackedLabel key={`l12-${j}`} x={(S.x + S.w + B.x) / 2}
          y={S.cy - 14 - (label12.length - 1 - j) * LABEL.lh} text={line} />
      ))}
      {/* the query goes down into Algeon */}
      {sketchLine('loop:b-g', B.cx, B.y + B.h + 2, B.cx, G.y - 2, edge)}
      {sketchHead('loop:b-g:head', B.cx, G.y, 'down', edge)}
      {olap && sketchHead('loop:b-g:back', B.cx, B.y + B.h, 'up', edge)}
      {labelAlgeon.map((line, j) => (
        <text key={`la-${j}`} x={B.cx + 12} y={(B.y + B.h + G.y) / 2 + LABEL.size * 0.36 + (j - (labelAlgeon.length - 1) / 2) * LABEL.lh}
          fontSize={LABEL.size} fontFamily={SANS} fill={COLOR.muted}>
          {line}
        </text>
      ))}
      {/* everything else goes to the OLAP engine */}
      {olap && sketchPath('loop:b-o', olapEdge, edge)}
      {olap && sketchHead('loop:b-o:head', A.cx, O.y, 'down', edge)}
      {olap && sketchHead('loop:b-o:back', elbowX, B.y + B.h, 'up', edge)}
      {olap && labelOlap.map((line, j) => (
        <BackedLabel key={`lo-${j}`} x={(A.cx + elbowX) / 2} y={elbowY - 14 - (labelOlap.length - 1 - j) * LABEL.lh} text={line} />
      ))}
      {/* Algeon reads the lakehouse */}
      {sketchLine('loop:g-l', G.x - 2, row2Cy, L.x + L.w + 2, row2Cy, edge)}
      {sketchHead('loop:g-l:head', L.x + L.w, row2Cy, 'left', edge)}
      <BackedLabel x={(L.x + L.w + G.x) / 2} y={row2Cy - 14} text="KvikIO" />
      {/* the OLAP engine reads the same tables */}
      {olap && sketchLine('loop:o-l', O.x + O.w + 2, row2Cy, L.x - 2, row2Cy, edge)}
      {olap && sketchHead('loop:o-l:head', L.x, row2Cy, 'right', edge)}
      {olap && <BackedLabel x={(O.x + O.w + L.x) / 2} y={row2Cy - 14} text="same tables" />}
      {/* rows and diagnostics return to the agent */}
      {!olap && sketchPath('loop:rail', rail, {...edge, dashed: true})}
      {!olap && sketchHead('loop:rail:head', A.cx, A.y + A.h, 'up', edge)}
      {!olap && (
        <text x={(A.cx + Gcx) / 2} y={railY + LABEL.lh} fontSize={LABEL.size} fontFamily={SANS}
          textAnchor="middle" fill={COLOR.muted}>
          {LOOP_RETURN}
        </text>
      )}
      {/* boxes last so edge ends tuck under their borders */}
      <Box node={LOOP_AGENT} x={A.x} y={A.y} w={A.w} h={A.h} fonts={fonts} />
      <Box node={semanticNode} x={S.x} y={S.y} w={S.w} h={S.h} fonts={fonts} />
      <Box node={LOOP_BACKEND} x={B.x} y={B.y} w={B.w} h={B.h} fonts={fonts} />
      <Box node={LOOP_ALGEON} x={G.x} y={G.y} w={G.w} h={G.h} fonts={fonts} />
      <Box node={LOOP_LAKE} x={L.x} y={L.y} w={L.w} h={L.h} fonts={fonts} />
      {olap && <Box node={LOOP_OLAP} x={O.x} y={O.y} w={O.w} h={O.h} fonts={fonts} />}
    </Figure>
  );
}

/* ====================== BarbieTitleShareFigure ====================== */

export interface TitleShareRecord {
  brand_name: string;
  barbie_pct_may_june: number;
  barbie_pct_july_august: number;
}

const SHARE_LABEL: FontSpec = {size: 13, lh: 18, face: 'sans'};
const SHARE_ROW_H = 30;
const SHARE_DOT_R = 5;
const SHARE_PLOT_W = 360;
const SHARE_HEAD = 'SHARE OF LIKES ON LISTINGS NAMED "BARBIE"';

const pct = (value: number) => `${value >= 1 ? value.toFixed(1) : value.toFixed(2)}%`;

/**
 * Dumbbell of the listing-name check: for each brand, the share of its likes
 * that landed on listings whose name contains "Barbie", before and after the
 * film's release. Rows sort by the July-August share so the brands whose
 * inventory moved sit at the top and the unexplained ones at the bottom.
 */
export function BarbieTitleShareFigure({records, caption}: {records: TitleShareRecord[]; caption?: React.ReactNode}) {
  const rows = [...records].sort((a, b) => b.barbie_pct_july_august - a.barbie_pct_july_august);
  const maxPct = Math.max(...rows.map((r) => r.barbie_pct_july_august));
  const axisMax = Math.ceil(maxPct / 10) * 10;
  const ticks = Array.from({length: axisMax / 10 + 1}, (_, i) => i * 10);

  const labelW = Math.ceil(Math.max(...rows.map((r) => textWidth(r.brand_name, SHARE_LABEL))));
  // Room for a before-value label to the left of a dot that sits at zero.
  const plotX = EDGE + 8 + labelW + 16 + 44;
  const headH = 14 + 12 + LABEL.lh;
  const top = EDGE + headH;
  const plotBottom = top + rows.length * SHARE_ROW_H;
  const axisY = plotBottom + 8;
  const svgH = axisY + LABEL.lh + 8 + EDGE;
  const svgW = plotX + SHARE_PLOT_W + 56 + EDGE;
  const x = (value: number) => plotX + (value / axisMax) * SHARE_PLOT_W;
  const edge = {stroke: INK, strokeWidth: EDGE_W};

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={440} center
      ariaLabel={`For each brand, the share of likes on listings whose name contains Barbie in May-June and in July-August 2023. ${
        rows.map((r) => `${r.brand_name}: ${pct(r.barbie_pct_may_june)} to ${pct(r.barbie_pct_july_august)}`).join('; ')}.`}>
      <text x={EDGE + 8} y={EDGE + 14} fontSize={HEAD.size} letterSpacing={`${HEAD.track}em`} fill={COLOR.muted}>
        {SHARE_HEAD}
      </text>
      {/* legend: the two marks differ by fill as well as by label */}
      <g transform={`translate(${EDGE + 8}, ${EDGE + 14 + 12 + LABEL.lh * 0.5})`}>
        <circle cx={SHARE_DOT_R} cy={0} r={SHARE_DOT_R} fill={COLOR.surface} stroke={INK} strokeWidth={EDGE_W} />
        <text x={SHARE_DOT_R * 2 + 6} y={LABEL.size * 0.36} fontSize={LABEL.size} fontFamily={SANS} fill={COLOR.muted}>
          May-June
        </text>
        <circle cx={SHARE_DOT_R + 96} cy={0} r={SHARE_DOT_R} fill={INK} />
        <text x={SHARE_DOT_R * 2 + 102} y={LABEL.size * 0.36} fontSize={LABEL.size} fontFamily={SANS} fill={COLOR.muted}>
          July-August
        </text>
      </g>
      {/* grid and axis stay recessive */}
      {ticks.map((t) => (
        <g key={`tick-${t}`}>
          {t > 0 && (
            <line x1={x(t)} y1={top} x2={x(t)} y2={plotBottom} stroke={COLOR.border} strokeDasharray="3 4" opacity={0.7} />
          )}
          <text x={x(t)} y={axisY + LABEL.lh} fontSize={LABEL.size} fontFamily={SANS} textAnchor="middle" fill={COLOR.subtle}>
            {t}%
          </text>
        </g>
      ))}
      {sketchLine('share:axis', plotX, axisY, plotX + SHARE_PLOT_W, axisY, {stroke: COLOR.border, strokeWidth: 1, roughness: 0.5})}
      {rows.map((r, i) => {
        const cy = top + i * SHARE_ROW_H + SHARE_ROW_H / 2;
        const xb = x(r.barbie_pct_may_june);
        const xa = x(r.barbie_pct_july_august);
        return (
          <g key={r.brand_name}>
            <title>{`${r.brand_name}: ${pct(r.barbie_pct_may_june)} in May-June, ${pct(r.barbie_pct_july_august)} in July-August`}</title>
            <text x={plotX - 44 - 16} y={cy + SHARE_LABEL.size * 0.36} fontSize={SHARE_LABEL.size} fontFamily={SANS}
              textAnchor="end" fill={COLOR.fg}>
              {r.brand_name}
            </text>
            {xa - xb > SHARE_DOT_R * 2 && sketchLine(`share:${r.brand_name}`, xb + SHARE_DOT_R, cy, xa - SHARE_DOT_R, cy,
              {stroke: COLOR.muted, strokeWidth: 1.6, roughness: 0.4})}
            <circle cx={xb} cy={cy} r={SHARE_DOT_R} fill={COLOR.surface} stroke={INK} strokeWidth={EDGE_W} />
            <circle cx={xa} cy={cy} r={SHARE_DOT_R + 1} fill={COLOR.surface} />
            <circle cx={xa} cy={cy} r={SHARE_DOT_R} fill={INK} />
            <text x={xb - SHARE_DOT_R - 5} y={cy + LABEL.size * 0.36} fontSize={LABEL.size} fontFamily={SANS}
              textAnchor="end" fill={COLOR.subtle}>
              {pct(r.barbie_pct_may_june)}
            </text>
            <text x={xa + SHARE_DOT_R + 6} y={cy + LABEL.size * 0.36} fontSize={LABEL.size} fontFamily={SANS} fill={COLOR.fg}>
              {pct(r.barbie_pct_july_august)}
            </text>
          </g>
        );
      })}
    </Figure>
  );
}

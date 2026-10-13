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
import type {BoxFonts, DiagramNode} from '@site/src/components/design/Diagram';
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
const LOOP_BAND = 'APPLICATION · AGENT, ONTOLOGY, SERVICE';
const LOOP_RETURN = 'rows and typed diagnostics feed the next hypothesis';

/**
 * The investigation loop as a ring: the agent's question passes through the
 * application's semantic layer and backend service into Algeon, which reads
 * the lakehouse and returns rows plus structured diagnostics to the agent.
 */
export function AgentLoopFigure({caption}: {caption?: React.ReactNode}) {
  const fonts: BoxFonts = COMPACT_FONTS;
  const [agent, semantic, backend, algeon, lake] =
    [LOOP_AGENT, LOOP_SEMANTIC, LOOP_BACKEND, LOOP_ALGEON, LOOP_LAKE].map((n) => measureNode(n, fonts));
  const BAND_PAD = 18;
  const BAND_HEAD = 34;
  const BAND_FOOT = 16;
  const V_GAP = 64;
  const RAIL_DROP = 34;
  const R = 10;
  const label01 = ['question in', 'domain terms'];
  const label12 = ['SQL over', 'registered relations'];
  const gapFor = (lines: string[]) =>
    Math.max(72, Math.ceil(Math.max(...lines.map((l) => textWidth(l, LABEL)))) + 48);

  // Row 1: the application-owned chain, left to right, inside one band.
  const bandY = EDGE + 4;
  const row1Top = bandY + BAND_HEAD;
  const row1H = Math.max(agent.h, semantic.h, backend.h);
  const x0 = EDGE + 8 + BAND_PAD;
  const x1 = x0 + agent.w + gapFor(label01);
  const x2 = x1 + semantic.w + gapFor(label12);
  const place = (x: number, s: {w: number; h: number}) =>
    ({x, y: row1Top + (row1H - s.h) / 2, w: s.w, h: s.h, cx: x + s.w / 2, cy: row1Top + row1H / 2});
  const A = place(x0, agent);
  const S = place(x1, semantic);
  const B = place(x2, backend);
  const bandX = x0 - BAND_PAD;
  const bandW = B.x + B.w + BAND_PAD - bandX;
  const bandH = BAND_HEAD + row1H + BAND_FOOT;

  // Row 2: Algeon under the service, the lakehouse under the semantic layer.
  const row2Top = bandY + bandH + V_GAP;
  const row2H = Math.max(algeon.h, lake.h);
  const G = {x: B.cx - algeon.w / 2, y: row2Top + (row2H - algeon.h) / 2, w: algeon.w, h: algeon.h};
  const L = {x: S.cx - lake.w / 2, y: row2Top + (row2H - lake.h) / 2, w: lake.w, h: lake.h};
  const Gcx = G.x + G.w / 2;
  const row2Cy = row2Top + row2H / 2;

  const railY = row2Top + row2H + RAIL_DROP;
  const svgH = railY + LABEL.lh + 8 + EDGE;
  const svgW = Math.max(bandX + bandW, G.x + G.w) + 8 + EDGE;
  const edge = {stroke: INK, strokeWidth: EDGE_W};
  const rail = [
    `M ${Gcx} ${G.y + G.h}`,
    `L ${Gcx} ${railY - R}`,
    `Q ${Gcx} ${railY} ${Gcx - R} ${railY}`,
    `L ${A.cx + R} ${railY}`,
    `Q ${A.cx} ${railY} ${A.cx} ${railY - R}`,
    `L ${A.cx} ${A.y + A.h + 1}`,
  ].join(' ');

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={640} center
      ariaLabel="The agent's question passes through the application's semantic layer and backend service into Algeon, which reads the lakehouse and returns rows and diagnostics to the agent for the next hypothesis.">
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
      <text x={B.cx + 12} y={(B.y + B.h + G.y) / 2 + LABEL.size * 0.36} fontSize={LABEL.size}
        fontFamily={SANS} fill={COLOR.muted}>
        one query
      </text>
      {/* Algeon reads the lakehouse */}
      {sketchLine('loop:g-l', G.x - 2, row2Cy, L.x + L.w + 2, row2Cy, edge)}
      {sketchHead('loop:g-l:head', L.x + L.w, row2Cy, 'left', edge)}
      <BackedLabel x={(L.x + L.w + G.x) / 2} y={row2Cy - 14} text="GPU file scans" />
      {/* rows and diagnostics return to the agent */}
      {sketchPath('loop:rail', rail, {...edge, dashed: true})}
      {sketchHead('loop:rail:head', A.cx, A.y + A.h, 'up', edge)}
      <text x={(A.cx + Gcx) / 2} y={railY + LABEL.lh} fontSize={LABEL.size} fontFamily={SANS}
        textAnchor="middle" fill={COLOR.muted}>
        {LOOP_RETURN}
      </text>
      {/* boxes last so edge ends tuck under their borders */}
      <Box node={LOOP_AGENT} x={A.x} y={A.y} w={A.w} h={A.h} fonts={fonts} />
      <Box node={LOOP_SEMANTIC} x={S.x} y={S.y} w={S.w} h={S.h} fonts={fonts} />
      <Box node={LOOP_BACKEND} x={B.x} y={B.y} w={B.w} h={B.h} fonts={fonts} />
      <Box node={LOOP_ALGEON} x={G.x} y={G.y} w={G.w} h={G.h} fonts={fonts} />
      <Box node={LOOP_LAKE} x={L.x} y={L.y} w={L.w} h={L.h} fonts={fonts} />
    </Figure>
  );
}

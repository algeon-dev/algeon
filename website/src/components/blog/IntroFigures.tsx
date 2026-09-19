import React from 'react';
import {
  Arrowhead,
  Box,
  COLOR,
  COMPACT_FONTS,
  EDGE,
  FLOW_FONTS,
  Figure,
  LABEL,
  TONE,
  measureNode,
  smallHeadDown,
  smallHeadUp,
  textWidth,
} from '@site/src/components/design/Diagram';
import type {BoxFonts, DiagramNode} from '@site/src/components/design/Diagram';

/**
 * Static figures for the introduction post. They are assembled from the
 * Design diagrams' box, tone, and figure primitives so the blog reads as the
 * same visual system, and each one carries its meaning in labels and a
 * caption rather than in color alone.
 */

const HEAD = {size: 10.5, track: 0.07};
const FOOT = {size: 11, lh: 15};

interface Placed {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Label with a surface-colored backing so it stays legible over lines. */
function BackedLabel({x, y, text, fill}: {x: number; y: number; text: string; fill: string}) {
  const w = textWidth(text, LABEL);
  return (
    <g>
      <rect x={x - w / 2 - 5} y={y - 9.5} width={w + 10} height={13} rx={3}
        fill={COLOR.surface} opacity={0.92} />
      <text x={x} y={y} fontSize={LABEL.size} letterSpacing={`${LABEL.track}em`}
        textAnchor="middle" fill={fill}>
        {text}
      </text>
    </g>
  );
}

/* ====================== WorkingSetFigure ====================== */

const WS_NODES: DiagramNode[] = [
  {kind: 'terabytes', name: 'Lakehouse', note: 'Parquet · tables · views', tone: 'lake'},
  {kind: 'gigabytes', name: 'Working set', note: 'fits one GPU', tone: 'engine'},
  {
    kind: 'gpu algorithm', name: 'Algorithm output',
    note: ['paths · communities', 'neighbors · scores'], tone: 'gpu',
  },
  {kind: 'rows', name: 'Answer', note: 'back to the caller', tone: 'plain'},
];

const WS_EDGE_LABELS: string[][] = [
  ['filter · join', 'aggregate'],
  ['cugraph_* · cuvs_*', 'cuml_*'],
  ['join · filter', 'aggregate'],
];

const WS_BAND_LABEL = 'ONE SQL STATEMENT · ONE PHYSICAL PLAN';
const WS_RETURN_LABEL = 'next stage input';

/**
 * The relational reduction on the left, the GPU algorithm after it, and the
 * output returning as rows: a left-to-right flow with a tinted band over the
 * part that is one statement, plus a dashed loop for output feeding the next
 * stage.
 */
export function WorkingSetFigure({caption}: {caption?: React.ReactNode}) {
  const fonts: BoxFonts = FLOW_FONTS;
  const sizes = WS_NODES.map((n) => measureNode(n, fonts));
  const rowH = Math.max(...sizes.map((s) => s.h));
  const BAND_HEAD = 32;
  const RAIL_DROP = 28;
  const boxTop = EDGE + BAND_HEAD;

  // The gap before each target box is sized to its widest label line.
  const gaps = WS_EDGE_LABELS.map((lines) =>
    Math.max(76, Math.ceil(Math.max(...lines.map((line) => textWidth(line, LABEL)))) + 34),
  );
  const placed: Placed[] = [];
  let x = EDGE + 8;
  WS_NODES.forEach((_, i) => {
    if (i > 0) x += gaps[i - 1];
    const s = sizes[i];
    placed.push({x, y: boxTop + (rowH - s.h) / 2, w: s.w, h: s.h});
    x += s.w;
  });
  const railY = boxTop + rowH + RAIL_DROP;
  const svgH = railY + FOOT.lh + 10 + EDGE;
  const svgW = x + 8 + EDGE;

  const bandX = placed[1].x - 16;
  const bandW = placed[3].x + placed[3].w + 16 - bandX;
  const cy = boxTop + rowH / 2;
  const wsCx = placed[1].x + placed[1].w / 2;
  const algoCx = placed[2].x + placed[2].w / 2;
  const wsBottom = placed[1].y + placed[1].h;
  const algoBottom = placed[2].y + placed[2].h;
  const R = 10;
  const rail = [
    `M ${algoCx} ${algoBottom}`,
    `L ${algoCx} ${railY - R}`,
    `Q ${algoCx} ${railY} ${algoCx - R} ${railY}`,
    `L ${wsCx + R} ${railY}`,
    `Q ${wsCx} ${railY} ${wsCx} ${railY - R}`,
    `L ${wsCx} ${wsBottom + 1}`,
  ].join(' ');

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={720} center>
      <rect x={bandX} y={EDGE} width={bandW} height={svgH - 2 * EDGE} rx={10}
        fill={TONE.gpu} opacity={0.05} />
      <rect x={bandX} y={EDGE} width={bandW} height={svgH - 2 * EDGE} rx={10}
        fill="none" stroke={TONE.gpu} strokeWidth={1} strokeDasharray="5 4" opacity={0.5} />
      <text x={bandX + 12} y={EDGE + 18} fontSize={HEAD.size} letterSpacing={`${HEAD.track}em`}
        fill={TONE.gpu}>
        {WS_BAND_LABEL}
      </text>
      {placed.slice(1).map((b, i) => {
        const a = placed[i];
        const sx = a.x + a.w;
        const dx = b.x;
        const mid = (sx + dx) / 2;
        const lines = WS_EDGE_LABELS[i];
        return (
          <g key={`edge-${i}`}>
            <line x1={sx} y1={cy} x2={dx - 1} y2={cy} stroke={COLOR.border} strokeWidth={1} />
            <Arrowhead x={dx} y={cy} stroke={COLOR.border} />
            {lines.map((line, j) => (
              <BackedLabel key={`l-${j}`} x={mid} y={cy - 10 - (lines.length - 1 - j) * 14}
                text={line} fill={COLOR.subtle} />
            ))}
          </g>
        );
      })}
      <path d={rail} fill="none" stroke={COLOR.border} strokeWidth={1} strokeDasharray="5 4"
        strokeLinejoin="round" />
      <path d={smallHeadUp(wsCx, wsBottom)} fill="none" stroke={COLOR.border} strokeWidth={1.25}
        strokeLinecap="round" strokeLinejoin="round" />
      <text x={(wsCx + algoCx) / 2} y={railY + FOOT.lh} fontSize={LABEL.size}
        letterSpacing={`${LABEL.track}em`} textAnchor="middle" fill={COLOR.subtle}>
        {WS_RETURN_LABEL}
      </text>
      {WS_NODES.map((n, i) => (
        <Box key={n.name} node={n} x={placed[i].x} y={placed[i].y} w={placed[i].w} h={placed[i].h}
          fonts={fonts} />
      ))}
    </Figure>
  );
}

/* ====================== GlueVsPlanFigure ====================== */

const LEFT_NODES: DiagramNode[] = [
  {kind: 'warehouse', name: 'Select the rows', note: 'one tenant · one quarter', tone: 'plan'},
  {kind: 'export job', name: 'Write Parquet to a bucket', tone: 'plain'},
  {kind: 'notebook', name: 'Load, clean, reshape', note: 'ids · nulls · duplicates · casts', tone: 'plain'},
  {kind: 'gpu library', name: 'cuGraph or cuML call', tone: 'gpu'},
  {kind: 'export again', name: 'Write the result', tone: 'plain'},
  {kind: 'another system', name: 'Import and join back', tone: 'plan'},
];
const LEFT_STEPS = ['export', 'download', 'in process', 'write', 'import'];
const LEFT_HEAD = 'TODAY: THE DETOUR';
const LEFT_FOOT = ['every step is application code', 'written for one dataset'];

const RIGHT_NODES: DiagramNode[] = [
  {kind: 'sql', name: 'Select and clean the input', note: 'WHERE · JOIN · CAST · DISTINCT', tone: 'plan'},
  {
    kind: 'table function', name: 'cugraph_*(edges => (SELECT …))',
    note: 'documented input and output schema', tone: 'gpu',
  },
  {kind: 'sql', name: 'Join and aggregate the output', note: 'rows back to the caller', tone: 'plan'},
];
const RIGHT_HEAD = 'WITH ALGEON: ONE PLAN';
const RIGHT_GROUP = 'one SQL statement · one process';
const RIGHT_FOOT = ['the planner owns scan, pushdown,', 'join order and GPU placement'];

function Spine({a, b, cx, label}: {a: Placed; b: Placed; cx: number; label?: string}) {
  const top = a.y + a.h;
  return (
    <g>
      <line x1={cx} y1={top} x2={cx} y2={b.y - 1} stroke={COLOR.border} strokeWidth={1} />
      <path d={smallHeadDown(cx, b.y)} fill="none" stroke={COLOR.border} strokeWidth={1.25}
        strokeLinecap="round" strokeLinejoin="round" />
      {label ? (
        <text x={cx + 10} y={(top + b.y) / 2 + LABEL.size * 0.36} fontSize={LABEL.size}
          letterSpacing={`${LABEL.track}em`} fill={COLOR.subtle}>
          {label}
        </text>
      ) : null}
    </g>
  );
}

/**
 * Two columns: the export-and-notebook detour on the left, one SQL statement
 * with the algorithm as a table function on the right.
 */
export function GlueVsPlanFigure({caption}: {caption?: React.ReactNode}) {
  const fonts: BoxFonts = COMPACT_FONTS;
  const ls = LEFT_NODES.map((n) => measureNode(n, fonts));
  const rs = RIGHT_NODES.map((n) => measureNode(n, fonts));
  const leftW = Math.max(...ls.map((s) => s.w));
  const rightW = Math.max(...rs.map((s) => s.w));
  const HEADER_H = 30;
  const GAP = 40;
  const COL_GAP = 88;
  const GROUP_PAD = 16;
  const GROUP_HEAD = 32;
  const GROUP_FOOT = 16;
  const FOOTER_H = 2 * FOOT.lh + 18;

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
  // The shorter column is centered against the taller one.
  const groupY = top + (leftBottom - top - groupH) / 2;
  const rp: Placed[] = [];
  let ry = groupY + GROUP_HEAD;
  RIGHT_NODES.forEach((_, i) => {
    if (i > 0) ry += GAP;
    rp.push({x: groupX + GROUP_PAD, y: ry, w: rightW, h: rs[i].h});
    ry += rs[i].h;
  });

  const svgW = groupX + groupW + 8 + EDGE;
  const svgH = leftBottom + FOOTER_H + EDGE;
  const lcx = leftX + leftW / 2;
  const rcx = groupX + GROUP_PAD + rightW / 2;

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={620} center>
      <text x={leftX} y={EDGE + 13} fontSize={HEAD.size} letterSpacing={`${HEAD.track}em`}
        fill={COLOR.subtle}>
        {LEFT_HEAD}
      </text>
      <text x={groupX} y={EDGE + 13} fontSize={HEAD.size} letterSpacing={`${HEAD.track}em`}
        fill={TONE.gpu}>
        {RIGHT_HEAD}
      </text>
      <rect x={groupX} y={groupY} width={groupW} height={groupH} rx={9}
        fill={TONE.gpu} opacity={0.05} />
      <rect x={groupX} y={groupY} width={groupW} height={groupH} rx={9}
        fill="none" stroke={TONE.gpu} strokeWidth={1} strokeDasharray="5 4" opacity={0.55} />
      <text x={groupX + 12} y={groupY + 18} fontSize={LABEL.size}
        letterSpacing={`${LABEL.track}em`} fill={TONE.gpu}>
        {RIGHT_GROUP}
      </text>
      {lp.slice(1).map((b, i) => (
        <Spine key={`l-${i}`} a={lp[i]} b={b} cx={lcx} label={LEFT_STEPS[i]} />
      ))}
      {rp.slice(1).map((b, i) => (
        <Spine key={`r-${i}`} a={rp[i]} b={b} cx={rcx} />
      ))}
      {LEFT_FOOT.map((line, i) => (
        <text key={`lf-${i}`} x={leftX} y={leftBottom + 20 + i * FOOT.lh} fontSize={FOOT.size}
          fill={COLOR.muted}>
          {line}
        </text>
      ))}
      {RIGHT_FOOT.map((line, i) => (
        <text key={`rf-${i}`} x={groupX} y={leftBottom + 20 + i * FOOT.lh} fontSize={FOOT.size}
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

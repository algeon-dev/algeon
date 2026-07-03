import React, {useState} from 'react';
import styles from './styles.module.css';

/**
 * Interactive system diagrams for the Design page.
 *
 * SVG with a computed layout: every label is monospace, so text widths are
 * deterministic (chars × advance width) and boxes are sized to their content —
 * nothing can wrap or overlap. Figures scale down proportionally on narrow
 * viewports (viewBox), with a legibility floor after which the container
 * scrolls horizontally.
 *
 * Two shapes:
 * - SystemDiagram — a lane-based DAG (host / GPU swim lanes, fork & merge
 *   edges, optional region underlays for shared invariants).
 * - StackDiagram — a top-down layering with optional boundary dividers.
 *
 * Nodes carrying a `detail` are clickable (and keyboard-focusable); the
 * selected node's detail renders in an HTML panel under the SVG, so the
 * diagram stays sparse while the content stays rich.
 *
 * Colors come from the theme-aware --nx-* tokens in src/css/custom.css, with
 * literal fallbacks so an extracted SVG still renders standalone. Tones are a
 * small fixed palette; meaning is always carried by an adjacent label, never
 * by color alone.
 */

export type Tone = 'plan' | 'gpu' | 'engine' | 'bridge' | 'graph' | 'foreign' | 'reject' | 'plain';

export interface DiagramNode {
  /** Quiet uppercase type prefix above the name. */
  kind?: string;
  /** Main label — one line, never wrapped. */
  name: string;
  /** Secondary line(s) — pass an array for explicit line breaks. */
  note?: string | string[];
  tone?: Tone;
  /** Small mono source pointer shown in the detail panel header. */
  sub?: string;
  /** Rich content for the detail panel; makes the node clickable. */
  detail?: React.ReactNode;
  /** StackDiagram: labelled dashed divider below this tier. */
  boundary?: string;
}

const MONO = 'var(--nx-dg-mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace)';

const COLOR = {
  border: 'var(--nx-dg-border-strong, #c3c9d4)',
  surface: 'var(--nx-dg-surface, #ffffff)',
  fg: 'var(--nx-dg-fg, #1c1e21)',
  muted: 'var(--nx-dg-fg-muted, #5b6470)',
  subtle: 'var(--nx-dg-fg-subtle, #8a93a2)',
};

// A DataFusion-orange → NVIDIA-green ramp: the closer a component sits to the
// GPU, the greener its tone (see the --nx-design-* tokens in custom.css).
const TONE: Record<Tone, string> = {
  plan: 'var(--nx-design-df, #f1543f)',
  engine: 'var(--nx-design-nexus, #fbb040)',
  bridge: 'var(--nx-design-bridge, #b5c400)',
  gpu: 'var(--nx-design-gpu, #76b900)',
  graph: 'var(--nx-design-graph, #2fb56b)',
  foreign: 'var(--nx-tone-foreign, #e8716d)',
  reject: 'var(--nx-design-df, #f1543f)',
  plain: 'var(--nx-dg-border-strong, #c3c9d4)',
};

// Monospace advance width as a fraction of font size. Common stacks sit at
// 0.55–0.61; 0.62 over-estimates slightly so boxes never run tight.
const CH = 0.62;

const KIND = {size: 10, lh: 15, track: 0.07};
const NAME = {size: 13.5, lh: 19};
const NOTE = {size: 11.5, lh: 16};
const LABEL = {size: 10.5, track: 0.03}; // edge / lane / region labels
const PAD_X = 15;
const PAD_Y = 12;
const EDGE = 1; // stroke bleed margin around the drawing

function textWidth(text: string, font: {size: number; track?: number}): number {
  const track = font.track ?? 0;
  return text.length * font.size * CH + Math.max(0, text.length - 1) * track * font.size;
}

function noteLines(node: DiagramNode): string[] {
  if (!node.note) return [];
  return Array.isArray(node.note) ? node.note : [node.note];
}

function measureNode(node: DiagramNode): {w: number; h: number} {
  const notes = noteLines(node);
  const w = Math.max(
    node.kind ? textWidth(node.kind.toUpperCase(), KIND) : 0,
    textWidth(node.name, NAME),
    ...notes.map((line) => textWidth(line, NOTE)),
  );
  const h = (node.kind ? KIND.lh : 0) + NAME.lh + notes.length * NOTE.lh;
  return {w: Math.ceil(w) + 2 * PAD_X, h: h + 2 * PAD_Y};
}

/** One node box: hairline rect, tone accent bar, mono text block, optional interactivity. */
function Box({node, x, y, w, h, selected, onSelect}: {
  node: DiagramNode;
  x: number;
  y: number;
  w: number;
  h: number;
  selected?: boolean;
  onSelect?: () => void;
}) {
  const notes = noteLines(node);
  const tone = node.tone ?? 'plain';
  const toneColor = TONE[tone];
  const clickable = Boolean(onSelect);
  const contentH = (node.kind ? KIND.lh : 0) + NAME.lh + notes.length * NOTE.lh;
  let lineY = y + (h - contentH) / 2;
  const tx = x + PAD_X;
  const lines: React.ReactNode[] = [];
  if (node.kind) {
    lines.push(
      <text key="kind" x={tx} y={lineY + KIND.lh * 0.74} fontSize={KIND.size}
        letterSpacing={`${KIND.track}em`} fill={tone === 'plain' ? COLOR.subtle : toneColor}>
        {node.kind.toUpperCase()}
      </text>,
    );
    lineY += KIND.lh;
  }
  lines.push(
    <text key="name" x={tx} y={lineY + NAME.lh * 0.72} fontSize={NAME.size} fontWeight={600} fill={COLOR.fg}>
      {node.name}
    </text>,
  );
  lineY += NAME.lh;
  notes.forEach((line, i) => {
    lines.push(
      <text key={`note-${i}`} x={tx} y={lineY + NOTE.lh * 0.72} fontSize={NOTE.size} fill={COLOR.muted}>
        {line}
      </text>,
    );
    lineY += NOTE.lh;
  });
  return (
    <g
      className={clickable ? styles.nodeClickable : undefined}
      style={clickable ? ({'--node-tone': toneColor} as React.CSSProperties) : undefined}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      aria-label={clickable ? `${node.name} — show details` : undefined}
      aria-pressed={clickable ? selected : undefined}
      onClick={onSelect}
      onKeyDown={
        clickable
          ? (e: React.KeyboardEvent) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelect?.();
              }
            }
          : undefined
      }>
      <rect className={styles.boxRect} x={x} y={y} width={w} height={h} rx={6}
        fill={COLOR.surface} stroke={selected ? toneColor : COLOR.border} strokeWidth={selected ? 1.4 : 1} />
      {selected ? (
        <rect x={x} y={y} width={w} height={h} rx={6} fill={toneColor} opacity={0.07} pointerEvents="none" />
      ) : null}
      {tone !== 'plain' ? (
        <line x1={x + 1.25} y1={y + 6} x2={x + 1.25} y2={y + h - 6}
          stroke={toneColor} strokeWidth={2.5} strokeLinecap="round" />
      ) : null}
      {lines}
    </g>
  );
}

function Arrowhead({x, y, stroke}: {x: number; y: number; stroke: string}) {
  return (
    <path d={`M ${x - 5.5} ${y - 4} L ${x - 0.5} ${y} L ${x - 5.5} ${y + 4}`}
      fill="none" stroke={stroke} strokeWidth={1.25} strokeLinecap="round" strokeLinejoin="round" />
  );
}

function Figure({hint, caption, svgW, svgH, minW, interactive, center, children, panel}: {
  hint?: string;
  caption?: React.ReactNode;
  svgW: number;
  svgH: number;
  minW: number;
  interactive?: boolean;
  center?: boolean;
  children: React.ReactNode;
  panel?: React.ReactNode;
}) {
  return (
    <div className={styles.figure}>
      <figure style={{margin: 0}}>
        {hint ? <span className={styles.hint}>{hint}</span> : null}
        <div className={styles.scroll}>
          <svg viewBox={`0 0 ${svgW} ${svgH}`} width="100%" role={interactive ? 'group' : undefined}
            aria-hidden={interactive ? undefined : true}
            style={{maxWidth: svgW, minWidth: Math.min(svgW, minW), display: 'block',
              margin: center ? '0 auto' : undefined, fontFamily: MONO}}>
            {children}
          </svg>
        </div>
        {caption ? <figcaption className={styles.caption}>{caption}</figcaption> : null}
      </figure>
      {panel}
    </div>
  );
}

/** Orthogonal elbow with rounded bends: right → (down|up) → right. */
function elbowPath(sx: number, sy: number, midX: number, dy: number, dx: number): string {
  const R = 10;
  const vDir = dy > sy ? 1 : -1;
  const r1 = Math.min(R, Math.abs(midX - sx), Math.abs(dy - sy) / 2);
  const r2 = Math.min(R, Math.abs(dx - midX), Math.abs(dy - sy) / 2);
  return [
    `M ${sx} ${sy}`,
    `L ${midX - r1} ${sy}`,
    `Q ${midX} ${sy} ${midX} ${sy + r1 * vDir}`,
    `L ${midX} ${dy - r2 * vDir}`,
    `Q ${midX} ${dy} ${midX + r2} ${dy}`,
    `L ${dx - 1} ${dy}`,
  ].join(' ');
}

function DetailPanel({node}: {node: DiagramNode}) {
  const tone = node.tone ?? 'plain';
  return (
    <div className={styles.panel} aria-live="polite">
      <div className={styles.panelHead}>
        <span className={styles.panelDot} style={{background: TONE[tone]}} />
        <span className={styles.panelName}>{node.name}</span>
        {node.sub ? <span className={styles.panelSub}>{node.sub}</span> : null}
      </div>
      <div className={styles.panelBody}>{node.detail}</div>
    </div>
  );
}

/* ======================= SystemDiagram ======================= */

export interface SystemLane {
  id: string;
  /** Short mono gutter label, e.g. "host" / "gpu". */
  label: string;
  /** Tints the lane band. */
  tone?: Tone;
}

export interface SystemNode extends DiagramNode {
  id: string;
  lane: string;
  col: number;
}

export interface SystemEdge {
  from: string;
  to: string;
  label?: string;
  dashed?: boolean;
}

export interface SystemRegion {
  /** Mono label drawn inside the region's top-left corner. */
  label: string;
  /** Node ids the region encloses. */
  nodes: string[];
  tone?: Tone;
}

export interface SystemDiagramProps {
  lanes: SystemLane[];
  nodes: SystemNode[];
  edges: SystemEdge[];
  regions?: SystemRegion[];
  caption?: React.ReactNode;
}

const LANE_PAD = 16;
const REGION_HEAD = 28; // headroom above nodes for the region label
const REGION_FOOT = 10;
const MIN_GAP = 56;

/**
 * A lane-based DAG: nodes on host/GPU swim lanes, orthogonal fork & merge
 * edges, optional region underlays. Nodes with `detail` are clickable and
 * drive the panel below the SVG.
 */
export function SystemDiagram({lanes, nodes, edges, regions = [], caption}: SystemDiagramProps) {
  const [selectedId, setSelectedId] = useState<string | null>(
    nodes.find((n) => n.detail)?.id ?? null,
  );

  const sizes = new Map(nodes.map((n) => [n.id, measureNode(n)]));
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const nCols = Math.max(...nodes.map((n) => n.col)) + 1;

  // Column widths from the widest node in each column.
  const colW: number[] = Array.from({length: nCols}, (_, c) =>
    Math.max(...nodes.filter((n) => n.col === c).map((n) => sizes.get(n.id)!.w)),
  );

  // Gap before each column, sized so edge labels landing there always fit.
  const outCount = new Map<string, number>();
  const inCount = new Map<string, number>();
  edges.forEach((e) => {
    outCount.set(e.from, (outCount.get(e.from) ?? 0) + 1);
    inCount.set(e.to, (inCount.get(e.to) ?? 0) + 1);
  });
  const gap: number[] = Array.from({length: nCols}, () => MIN_GAP);
  edges.forEach((e) => {
    const c = byId.get(e.to)!.col;
    if (!e.label || c === 0) return;
    const straight = byId.get(e.from)!.lane === byId.get(e.to)!.lane;
    const need = Math.ceil(textWidth(e.label, LABEL)) + (straight ? 28 : 52);
    gap[c] = Math.max(gap[c], need);
  });

  // Lane geometry. A lane hosting region nodes gets label headroom.
  const regionLanes = new Set(
    regions.flatMap((r) => r.nodes.map((id) => byId.get(id)!.lane)),
  );
  const laneTop = new Map<string, number>();
  const laneH = new Map<string, number>();
  let y = EDGE;
  lanes.forEach((lane) => {
    const padTop = LANE_PAD + (regionLanes.has(lane.id) ? REGION_HEAD : 0);
    const padBottom = LANE_PAD + (regionLanes.has(lane.id) ? REGION_FOOT : 0);
    const maxH = Math.max(...nodes.filter((n) => n.lane === lane.id).map((n) => sizes.get(n.id)!.h));
    laneTop.set(lane.id, y);
    laneH.set(lane.id, padTop + maxH + padBottom);
    y += padTop + maxH + padBottom;
  });
  const svgH = y + EDGE;

  const gutter = Math.max(...lanes.map((l) => textWidth(l.label.toUpperCase(), KIND))) + 24;
  const colX: number[] = [];
  let x = gutter;
  for (let c = 0; c < nCols; c += 1) {
    if (c > 0) x += gap[c];
    colX.push(x);
    x += colW[c];
  }
  const svgW = x + EDGE + 4;

  // Node rects (centered in their column and lane).
  const rect = new Map<string, {x: number; y: number; w: number; h: number; cy: number}>();
  nodes.forEach((n) => {
    const s = sizes.get(n.id)!;
    const lane = lanes.find((l) => l.id === n.lane)!;
    const padTop = LANE_PAD + (regionLanes.has(lane.id) ? REGION_HEAD : 0);
    const padBottom = LANE_PAD + (regionLanes.has(lane.id) ? REGION_FOOT : 0);
    const innerH = laneH.get(lane.id)! - padTop - padBottom;
    const nx = colX[n.col] + (colW[n.col] - s.w) / 2;
    const ny = laneTop.get(lane.id)! + padTop + (innerH - s.h) / 2;
    rect.set(n.id, {x: nx, y: ny, w: s.w, h: s.h, cy: ny + s.h / 2});
  });

  const selected = selectedId ? byId.get(selectedId) : undefined;

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={660} interactive
      hint="click a stage for details"
      panel={selected?.detail ? <DetailPanel node={selected} /> : null}>
      {/* lane bands */}
      {lanes.map((lane) => {
        const top = laneTop.get(lane.id)!;
        const h = laneH.get(lane.id)!;
        return (
          <g key={lane.id}>
            {lane.tone ? (
              <rect x={0} y={top} width={svgW} height={h} fill={TONE[lane.tone]} opacity={0.04} />
            ) : null}
            <text x={10} y={top + h / 2 + 3.5} fontSize={KIND.size} letterSpacing={`${KIND.track}em`}
              fill={lane.tone ? TONE[lane.tone] : COLOR.subtle} opacity={0.9}>
              {lane.label.toUpperCase()}
            </text>
          </g>
        );
      })}
      {/* lane separators */}
      {lanes.slice(1).map((lane) => (
        <line key={`sep-${lane.id}`} x1={0} y1={laneTop.get(lane.id)!} x2={svgW} y2={laneTop.get(lane.id)!}
          stroke={COLOR.border} strokeWidth={1} strokeDasharray="2 5" opacity={0.7} />
      ))}
      {/* regions */}
      {regions.map((region, i) => {
        const rs = region.nodes.map((id) => rect.get(id)!);
        const rx = Math.min(...rs.map((r) => r.x)) - 10;
        const ry = Math.min(...rs.map((r) => r.y)) - REGION_HEAD;
        const rw = Math.max(...rs.map((r) => r.x + r.w)) + 10 - rx;
        const rh = Math.max(...rs.map((r) => r.y + r.h)) + REGION_FOOT - ry;
        const toneColor = TONE[region.tone ?? 'plain'];
        return (
          <g key={`region-${i}`}>
            <rect x={rx} y={ry} width={rw} height={rh} rx={8} fill={toneColor} opacity={0.05} />
            <rect x={rx} y={ry} width={rw} height={rh} rx={8} fill="none"
              stroke={toneColor} strokeWidth={1} strokeDasharray="5 4" opacity={0.55} />
            <text x={rx + 10} y={ry + 16} fontSize={LABEL.size} letterSpacing={`${LABEL.track}em`}
              fill={toneColor}>
              {region.label}
            </text>
          </g>
        );
      })}
      {/* edges */}
      {edges.map((e, i) => {
        const a = rect.get(e.from)!;
        const b = rect.get(e.to)!;
        const sx = a.x + a.w;
        const sy = a.cy;
        const dx = b.x;
        const dy = b.cy;
        const connected = selectedId === e.from || selectedId === e.to;
        const stroke = connected && selected ? TONE[selected.tone ?? 'plain'] : COLOR.border;
        const dash = e.dashed ? '5 4' : undefined;
        const labelText = (lx: number, ly: number) => {
          if (!e.label) return null;
          const lw = textWidth(e.label, LABEL);
          return (
            <g>
              <rect x={lx - lw / 2 - 5} y={ly - 9.5} width={lw + 10} height={13} rx={3}
                fill={COLOR.surface} opacity={0.92} />
              <text x={lx} y={ly} fontSize={LABEL.size} letterSpacing={`${LABEL.track}em`}
                textAnchor="middle" fill={connected && selected ? stroke : COLOR.subtle}>
                {e.label}
              </text>
            </g>
          );
        };
        if (byId.get(e.from)!.lane === byId.get(e.to)!.lane) {
          return (
            <g key={`edge-${i}`}>
              <line x1={sx} y1={sy} x2={dx - 1} y2={dy} stroke={stroke} strokeWidth={connected ? 1.25 : 1}
                strokeDasharray={dash} />
              <Arrowhead x={dx} y={dy} stroke={stroke} />
              {labelText((sx + dx) / 2, sy - 7)}
            </g>
          );
        }
        const midX = (outCount.get(e.from) ?? 0) > 1 ? sx + 22
          : (inCount.get(e.to) ?? 0) > 1 ? dx - 26
          : (sx + dx) / 2;
        const seg1 = Math.abs(midX - sx);
        const seg2 = Math.abs(dx - midX);
        return (
          <g key={`edge-${i}`}>
            <path d={elbowPath(sx, sy, midX, dy, dx)}
              fill="none" stroke={stroke} strokeWidth={connected ? 1.25 : 1}
              strokeDasharray={dash} strokeLinejoin="round" />
            <Arrowhead x={dx} y={dy} stroke={stroke} />
            {seg2 >= seg1 ? labelText((midX + dx) / 2, dy - 7) : labelText((sx + midX) / 2, sy - 7)}
          </g>
        );
      })}
      {/* nodes */}
      {nodes.map((n) => {
        const r = rect.get(n.id)!;
        return (
          <Box key={n.id} node={n} x={r.x} y={r.y} w={r.w} h={r.h}
            selected={selectedId === n.id}
            onSelect={n.detail ? () => setSelectedId(n.id) : undefined} />
        );
      })}
    </Figure>
  );
}

/* ======================= StackDiagram ======================= */

export interface StackDiagramProps {
  nodes: DiagramNode[];
  caption?: React.ReactNode;
}

/** A top-down layered stack with optional labelled boundary dividers. */
export function StackDiagram({nodes, caption}: StackDiagramProps) {
  const interactive = nodes.some((n) => n.detail);
  const [selectedIdx, setSelectedIdx] = useState<number>(nodes.findIndex((n) => n.detail));
  const sizes = nodes.map(measureNode);
  const boxW = Math.max(380, ...sizes.map((s) => s.w));
  const svgW = boxW + 2 * EDGE;
  const GAP = 12;
  const BOUNDARY_H = 24;
  const parts: React.ReactNode[] = [];
  let y = EDGE;
  nodes.forEach((node, i) => {
    if (i > 0) y += GAP;
    parts.push(
      <Box key={`b${i}`} node={node} x={EDGE} y={y} w={boxW} h={sizes[i].h}
        selected={interactive && selectedIdx === i}
        onSelect={node.detail ? () => setSelectedIdx(i) : undefined} />,
    );
    y += sizes[i].h;
    if (node.boundary) {
      const midY = y + GAP / 2 + BOUNDARY_H / 2;
      const labelW = textWidth(node.boundary, LABEL);
      const gapHalf = labelW / 2 + 12;
      parts.push(
        <g key={`d${i}`}>
          <line x1={EDGE} y1={midY} x2={svgW / 2 - gapHalf} y2={midY}
            stroke={TONE.foreign} strokeWidth={1} strokeDasharray="5 4" />
          <line x1={svgW / 2 + gapHalf} y1={midY} x2={svgW - EDGE} y2={midY}
            stroke={TONE.foreign} strokeWidth={1} strokeDasharray="5 4" />
          <text x={svgW / 2} y={midY + LABEL.size * 0.36} fontSize={LABEL.size}
            letterSpacing={`${LABEL.track}em`} textAnchor="middle" fill={TONE.foreign}>
            {node.boundary}
          </text>
        </g>,
      );
      y += GAP + BOUNDARY_H;
    }
  });
  const selected = interactive && selectedIdx >= 0 ? nodes[selectedIdx] : undefined;
  return (
    <Figure caption={caption} svgW={svgW} svgH={y + EDGE} minW={420} interactive={interactive} center
      hint={interactive ? 'click a layer for details' : undefined}
      panel={selected?.detail ? <DetailPanel node={selected} /> : null}>
      {parts}
    </Figure>
  );
}

export default SystemDiagram;

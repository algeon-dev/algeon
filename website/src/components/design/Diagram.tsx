import React, {useState} from 'react';
import styles from './styles.module.css';

/**
 * Interactive system diagrams for the Design page.
 *
 * SVG with a computed layout: every label is monospace, so text widths are
 * deterministic (chars × advance width) and boxes are sized to their content;
 * nothing can wrap or overlap. Figures scale down proportionally on narrow
 * viewports (viewBox), with a legibility floor after which the container
 * scrolls horizontally.
 *
 * Two shapes:
 * - SystemDiagram: a lane-based DAG (host / GPU swim lanes, fork & merge
 *   edges, optional region underlays for shared invariants).
 * - StackDiagram: a top-down layering with optional boundary dividers.
 *
 * Nodes carrying a `detail` are clickable (and keyboard-focusable); the
 * selected node's detail renders in an HTML panel under the SVG, so the
 * diagram stays sparse while the content stays rich.
 *
 * Colors come from the theme-aware --ag-* tokens in src/css/custom.css, with
 * literal fallbacks so an extracted SVG still renders standalone. Tones are a
 * small fixed palette; meaning is always carried by an adjacent label, never
 * by color alone.
 */

export type Tone = 'plan' | 'gpu' | 'engine' | 'bridge' | 'graph' | 'lake' | 'foreign' | 'reject' | 'plain';

export interface DiagramNode {
  /** Quiet uppercase type prefix above the name. */
  kind?: string;
  /** Main label (one line, never wrapped). */
  name: string;
  /** Secondary line(s); pass an array for explicit line breaks. */
  note?: string | string[];
  tone?: Tone;
  /** Small mono source pointer shown in the detail panel header. */
  sub?: string;
  /** Rich content for the detail panel; makes the node clickable. */
  detail?: React.ReactNode;
  /** StackDiagram: labelled dashed divider below this tier. */
  boundary?: string;
}

const MONO = 'var(--ag-dg-mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace)';

export const COLOR = {
  border: 'var(--ag-dg-border-strong, #c3c9d4)',
  surface: 'var(--ag-dg-surface, #ffffff)',
  fg: 'var(--ag-dg-fg, #1c1e21)',
  muted: 'var(--ag-dg-fg-muted, #5b6470)',
  subtle: 'var(--ag-dg-fg-subtle, #8a93a2)',
};

// A DataFusion-orange → NVIDIA-green ramp: the closer a component sits to the
// GPU, the greener its tone (see the --ag-design-* tokens in custom.css).
export const TONE: Record<Tone, string> = {
  plan: 'var(--ag-design-df, #f1543f)',
  engine: 'var(--ag-design-algeon, #fbb040)',
  bridge: 'var(--ag-design-bridge, #b5c400)',
  gpu: 'var(--ag-design-gpu, #76b900)',
  graph: 'var(--ag-design-graph, #2fb56b)',
  lake: 'var(--ag-design-lake, #5fb3e4)',
  foreign: 'var(--ag-tone-foreign, #e8716d)',
  reject: 'var(--ag-design-df, #f1543f)',
  plain: 'var(--ag-dg-border-strong, #c3c9d4)',
};

// Monospace advance width as a fraction of font size. Common stacks sit at
// 0.55–0.61; 0.62 over-estimates slightly so boxes never run tight.
const CH = 0.62;

const KIND = {size: 10, lh: 15, track: 0.07};
const NAME = {size: 13.5, lh: 19};
const NOTE = {size: 11.5, lh: 16};
export const LABEL = {size: 10.5, track: 0.03}; // edge / lane / region labels
const PAD_X = 15;
const PAD_Y = 12;
export const EDGE = 1; // stroke bleed margin around the drawing

export function textWidth(text: string, font: {size: number; track?: number}): number {
  const track = font.track ?? 0;
  return text.length * font.size * CH + Math.max(0, text.length - 1) * track * font.size;
}

function noteLines(node: DiagramNode): string[] {
  if (!node.note) return [];
  return Array.isArray(node.note) ? node.note : [node.note];
}

/** Font/padding bundle so the same Box renders at compact (Design DAG) or
 * glanceable (Overview/Flow) scale. */
export interface BoxFonts {
  kind: {size: number; lh: number; track: number};
  name: {size: number; lh: number};
  note: {size: number; lh: number};
  padX: number;
  padY: number;
}

export const COMPACT_FONTS: BoxFonts = {kind: KIND, name: NAME, note: NOTE, padX: PAD_X, padY: PAD_Y};

export function measureNode(node: DiagramNode, fonts: BoxFonts = COMPACT_FONTS): {w: number; h: number} {
  const notes = noteLines(node);
  const w = Math.max(
    node.kind ? textWidth(node.kind.toUpperCase(), fonts.kind) : 0,
    textWidth(node.name, fonts.name),
    ...notes.map((line) => textWidth(line, fonts.note)),
  );
  const h = (node.kind ? fonts.kind.lh : 0) + fonts.name.lh + notes.length * fonts.note.lh;
  return {w: Math.ceil(w) + 2 * fonts.padX, h: h + 2 * fonts.padY};
}

/** One node box: hairline rect, tone accent bar, mono text block, optional interactivity. */
export function Box({node, x, y, w, h, selected, onSelect, fonts = COMPACT_FONTS, kindColor,
  borderColor = COLOR.border, borderWidth = 1}: {
  node: DiagramNode;
  x: number;
  y: number;
  w: number;
  h: number;
  selected?: boolean;
  onSelect?: () => void;
  fonts?: BoxFonts;
  kindColor?: string;
  borderColor?: string;
  borderWidth?: number;
}) {
  const notes = noteLines(node);
  const tone = node.tone ?? 'plain';
  const toneColor = TONE[tone];
  const clickable = Boolean(onSelect);
  const contentH = (node.kind ? fonts.kind.lh : 0) + fonts.name.lh + notes.length * fonts.note.lh;
  let lineY = y + (h - contentH) / 2;
  const tx = x + fonts.padX;
  const lines: React.ReactNode[] = [];
  if (node.kind) {
    lines.push(
      <text key="kind" x={tx} y={lineY + fonts.kind.lh * 0.74} fontSize={fonts.kind.size}
        letterSpacing={`${fonts.kind.track}em`}
        fill={kindColor ?? (tone === 'plain' ? COLOR.subtle : toneColor)}>
        {node.kind.toUpperCase()}
      </text>,
    );
    lineY += fonts.kind.lh;
  }
  lines.push(
    <text key="name" x={tx} y={lineY + fonts.name.lh * 0.72} fontSize={fonts.name.size} fontWeight={600} fill={COLOR.fg}>
      {node.name}
    </text>,
  );
  lineY += fonts.name.lh;
  notes.forEach((line, i) => {
    lines.push(
      <text key={`note-${i}`} x={tx} y={lineY + fonts.note.lh * 0.72} fontSize={fonts.note.size} fill={COLOR.muted}>
        {line}
      </text>,
    );
    lineY += fonts.note.lh;
  });
  return (
    <g
      className={clickable ? styles.nodeClickable : undefined}
      style={clickable ? ({'--node-tone': toneColor} as React.CSSProperties) : undefined}
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      aria-label={clickable ? `${node.name}: show details` : undefined}
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
        fill={COLOR.surface} stroke={selected ? toneColor : borderColor}
        strokeWidth={selected ? 1.4 : borderWidth} />
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

export function Arrowhead({x, y, stroke}: {x: number; y: number; stroke: string}) {
  return (
    <path d={`M ${x - 5.5} ${y - 4} L ${x - 0.5} ${y} L ${x - 5.5} ${y + 4}`}
      fill="none" stroke={stroke} strokeWidth={1.25} strokeLinecap="round" strokeLinejoin="round" />
  );
}

export function Figure({caption, svgW, svgH, minW, interactive, ariaLabel, center, children, panel}: {
  caption?: React.ReactNode;
  svgW: number;
  svgH: number;
  minW: number;
  interactive?: boolean;
  ariaLabel?: string;
  center?: boolean;
  children: React.ReactNode;
  panel?: React.ReactNode;
}) {
  return (
    <div className={styles.figure}>
      <figure style={{margin: 0}}>
        <div className={styles.scroll}>
          <svg viewBox={`0 0 ${svgW} ${svgH}`} width="100%" role={interactive ? 'group' : undefined}
            aria-hidden={interactive ? undefined : true}
            aria-label={interactive ? ariaLabel : undefined}
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
  ariaLabel?: string;
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
export function SystemDiagram({ariaLabel, lanes, nodes, edges, regions = [], caption}: SystemDiagramProps) {
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
    // Elbow labels sit at the midpoint of one leg of the gap, so the gap must
    // be roughly twice the label width for the text to clear the target box.
    const lw = Math.ceil(textWidth(e.label, LABEL));
    const need = straight ? lw + 28 : 2 * lw + 48;
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
      ariaLabel={ariaLabel ?? 'Interactive system diagram'}
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
  ariaLabel?: string;
  nodes: DiagramNode[];
  caption?: React.ReactNode;
}

/** A top-down layered stack with optional labelled boundary dividers. */
export function StackDiagram({ariaLabel, nodes, caption}: StackDiagramProps) {
  const interactive = nodes.some((n) => n.detail);
  const [selectedIdx, setSelectedIdx] = useState<number>(nodes.findIndex((n) => n.detail));
  const sizes = nodes.map((n) => measureNode(n));
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
      ariaLabel={interactive ? (ariaLabel ?? 'Interactive stack diagram') : undefined}
      panel={selected?.detail ? <DetailPanel node={selected} /> : null}>
      {parts}
    </Figure>
  );
}

const ARCH_KIND = {size: 11.5, lh: 18, track: 0.07};
const ARCH_NAME = {size: 17, lh: 25};
const ARCH_NOTE = {size: 13, lh: 19};
const ARCH_FONTS: BoxFonts = {
  kind: ARCH_KIND,
  name: ARCH_NAME,
  note: ARCH_NOTE,
  padX: 20,
  padY: 15,
};
const ARCH_CARD_BORDER = 'var(--ag-arch-card-border, #d7dce1)';
const ARCH_EDGE = 'var(--ag-arch-edge, #748292)';
const ARCH_CARD_BORDER_WIDTH = 1.2;
const ARCH_EDGE_WIDTH = 1.1;

export function smallHeadDown(x: number, y: number): string {
  return `M ${x - 4} ${y - 5} L ${x} ${y - 0.5} L ${x + 4} ${y - 5}`;
}

export function smallHeadLeft(x: number, y: number): string {
  return `M ${x + 5} ${y - 4} L ${x + 0.5} ${y} L ${x + 5} ${y + 4}`;
}

export function smallHeadRight(x: number, y: number): string {
  return `M ${x - 5} ${y - 4} L ${x - 0.5} ${y} L ${x - 5} ${y + 4}`;
}

export function smallHeadUp(x: number, y: number): string {
  return `M ${x - 4} ${y + 5} L ${x} ${y + 0.5} L ${x + 4} ${y + 5}`;
}

export function ArchitectureDiagram({caption}: {caption?: React.ReactNode}) {
  const colX = 140;
  const outCx = 830;
  const sourceCx = 640;

  const top: DiagramNode = {
    kind: 'query interfaces',
    name: 'SQL · DataFrame · Flight SQL',
    note: 'embedded library · remote ADBC clients',
    tone: 'plain',
  };
  const rule: DiagramNode = {
    kind: 'DataFusion optimizer rule',
    name: 'algeon_datafusion_native',
    note: 'selects a whole relational candidate',
    tone: 'engine',
  };
  const gpu: DiagramNode = {
    kind: 'native · same process',
    name: 'GPU execution',
    note: [
      'cuDF · SQL algorithm functions',
      'admission queue · GPU memory ledger',
    ],
    tone: 'gpu',
  };
  const out: DiagramNode = {
    kind: 'same caller contract',
    name: 'Arrow RecordBatch stream',
    note: 'returns to the original query interface',
    tone: 'plain',
  };
  const disk: DiagramNode = {
    kind: 'mutable workspace',
    name: 'Local disk',
    note: 'Parquet files · scratch tables',
    tone: 'bridge',
  };
  const source: DiagramNode = {
    kind: 'S3-compatible',
    name: 'Remote Iceberg',
    note: 'REST / Glue catalog metadata',
    tone: 'lake',
  };

  const topSize = measureNode(top, ARCH_FONTS);
  const ruleSize = measureNode(rule, ARCH_FONTS);
  const gpuSize = measureNode(gpu, ARCH_FONTS);
  const outSize = measureNode(out, ARCH_FONTS);
  const diskSize = measureNode(disk, ARCH_FONTS);
  const sourceSize = measureNode(source, ARCH_FONTS);

  const topY = EDGE;
  const topBottom = topY + topSize.h;
  const topCy = topY + topSize.h / 2;
  const ruleY = topBottom + 64;
  const ruleBottom = ruleY + ruleSize.h;
  const ruleCy = ruleY + ruleSize.h / 2;
  const gpuY = ruleBottom + 64;
  const gpuBottom = gpuY + gpuSize.h;
  const gpuCy = gpuY + gpuSize.h / 2;
  // The result card shares the optimizer row so the CPU fallback edge is a
  // straight read across the page.
  const outY = ruleCy - outSize.h / 2;
  const outBottom = outY + outSize.h;
  // Storage cards share one baseline even though their heights differ, and
  // one container encloses both: the single KvikIO arrow below rises from
  // the container's top edge, not from either card.
  const inputY = gpuBottom + 96;
  const inputBottom = inputY + Math.max(diskSize.h, sourceSize.h);
  const diskY = inputBottom - diskSize.h;
  const sourceY = inputBottom - sourceSize.h;
  const storeLeft = Math.min(colX, sourceCx - sourceSize.w / 2) - 26;
  const storeRight = Math.max(colX + diskSize.w, sourceCx + sourceSize.w / 2) + 26;
  const storeY = inputY - 24;
  const storeBottom = inputBottom + 24;
  const H = Math.ceil(storeBottom) + EDGE;

  // Left-aligned column: the spine runs at the narrowest card's center, so
  // every vertical arrow still lands inside its box.
  const spineX = colX + Math.min(topSize.w, ruleSize.w, gpuSize.w) / 2;
  const topRight = colX + topSize.w;
  const ruleRight = colX + ruleSize.w;
  const gpuRight = colX + gpuSize.w;
  const outLeft = outCx - outSize.w / 2;
  const outRight = outCx + outSize.w / 2;
  const R = 12;
  const W = Math.ceil(outRight + 8);

  const stroke = ARCH_EDGE;
  const head = (d: string) => <path d={d} fill={stroke} stroke="none" />;

  return (
    <Figure caption={caption} svgW={W} svgH={H} minW={680} center>
      {/* The main query path runs straight down the left column. */}
      <line x1={spineX} y1={topBottom} x2={spineX} y2={ruleY - 1}
        stroke={stroke} strokeWidth={ARCH_EDGE_WIDTH} />
      {head(smallHeadDown(spineX, ruleY))}
      <line x1={spineX} y1={ruleBottom} x2={spineX} y2={gpuY - 1}
        stroke={stroke} strokeWidth={ARCH_EDGE_WIDTH} />
      {head(smallHeadDown(spineX, gpuY))}

      {/* A non-selected candidate falls back to DataFusion CPU execution
          (dashed, as the non-selected path) and still lands in the same
          Arrow stream. */}
      <line x1={ruleRight} y1={ruleCy} x2={outLeft - 1} y2={ruleCy}
        stroke={stroke} strokeWidth={ARCH_EDGE_WIDTH} strokeDasharray="5 4" />
      {head(smallHeadRight(outLeft, ruleCy))}
      <text x={(ruleRight + outLeft) / 2} y={ruleCy + 20} textAnchor="middle"
        fontSize={ARCH_KIND.size} fontWeight={600} fill={COLOR.muted}>
        DataFusion CPU fallback
      </text>

      {/* GPU results join the same caller contract from below. */}
      <path
        d={[
          `M ${gpuRight} ${gpuCy}`,
          `L ${outCx - R} ${gpuCy}`,
          `Q ${outCx} ${gpuCy} ${outCx} ${gpuCy - R}`,
          `L ${outCx} ${outBottom + 1}`,
        ].join(' ')}
        fill="none" stroke={stroke} strokeWidth={ARCH_EDGE_WIDTH} strokeLinejoin="round" />
      {head(smallHeadUp(outCx, outBottom))}

      {/* Both storage cards share one KvikIO read path: a single trunk
          climbs from the storage container's top edge into the GPU. */}
      <line x1={spineX} y1={storeY} x2={spineX} y2={gpuBottom + 1}
        stroke={stroke} strokeWidth={ARCH_EDGE_WIDTH} />
      {head(smallHeadUp(spineX, gpuBottom))}
      <text x={spineX - 10} y={(storeY + gpuBottom) / 2 + ARCH_KIND.size * 0.35}
        fontSize={ARCH_KIND.size} fontWeight={600} textAnchor="end" fill={COLOR.muted}>
        KvikIO
      </text>

      {/* Results return outside the query path, so the loop is visible
          without crossing any card. */}
      <path
        d={[
          `M ${outCx} ${outY}`,
          `L ${outCx} ${topCy + R}`,
          `Q ${outCx} ${topCy} ${outCx - R} ${topCy}`,
          `L ${topRight + 1} ${topCy}`,
        ].join(' ')}
        fill="none" stroke={stroke} strokeWidth={ARCH_EDGE_WIDTH} strokeLinejoin="round" />
      {head(smallHeadLeft(topRight, topCy))}

      {/* The storage container groups both inputs under the shared read
          path; cards draw on top of it. */}
      <rect x={storeLeft} y={storeY} width={storeRight - storeLeft}
        height={storeBottom - storeY} rx={10}
        fill={COLOR.fg} fillOpacity={0.04}
        stroke={ARCH_CARD_BORDER} strokeWidth={1} />
      {/* Boxes are last so edge endpoints tuck under their borders. */}
      <Box node={top} x={colX} y={topY} w={topSize.w} h={topSize.h}
        fonts={ARCH_FONTS} kindColor={COLOR.muted}
        borderColor={ARCH_CARD_BORDER} borderWidth={ARCH_CARD_BORDER_WIDTH} />
      <rect x={colX - 7} y={ruleY - 7}
        width={ruleSize.w + 14} height={ruleSize.h + 14} rx={12}
        fill={TONE.engine} opacity={0.1} />
      <Box node={rule} x={colX} y={ruleY} w={ruleSize.w} h={ruleSize.h}
        fonts={ARCH_FONTS} kindColor={COLOR.muted}
        borderColor={ARCH_CARD_BORDER} borderWidth={ARCH_CARD_BORDER_WIDTH} />
      <Box node={gpu} x={colX} y={gpuY} w={gpuSize.w} h={gpuSize.h}
        fonts={ARCH_FONTS} kindColor={COLOR.muted}
        borderColor={ARCH_CARD_BORDER} borderWidth={ARCH_CARD_BORDER_WIDTH} />
      <Box node={out} x={outLeft} y={outY} w={outSize.w} h={outSize.h}
        fonts={ARCH_FONTS} kindColor={COLOR.muted}
        borderColor={ARCH_CARD_BORDER} borderWidth={ARCH_CARD_BORDER_WIDTH} />
      <Box node={disk} x={colX} y={diskY} w={diskSize.w} h={diskSize.h}
        fonts={ARCH_FONTS} kindColor={COLOR.muted}
        borderColor={ARCH_CARD_BORDER} borderWidth={ARCH_CARD_BORDER_WIDTH} />
      <Box node={source} x={sourceCx - sourceSize.w / 2} y={sourceY} w={sourceSize.w} h={sourceSize.h}
        fonts={ARCH_FONTS} kindColor={COLOR.muted}
        borderColor={ARCH_CARD_BORDER} borderWidth={ARCH_CARD_BORDER_WIDTH} />
    </Figure>
  );
}

/* ======================= FlowDiagram ======================= */

// Flow boxes sit between the compact Design-DAG type and the homepage's
// ArchitectureDiagram type: a notch larger than StackDiagram, without its density.
const FLOW_KIND = {size: 10.5, lh: 16, track: 0.07};
const FLOW_NAME = {size: 14.5, lh: 21};
const FLOW_NOTE = {size: 12, lh: 17};
export const FLOW_FONTS: BoxFonts = {kind: FLOW_KIND, name: FLOW_NAME, note: FLOW_NOTE, padX: 16, padY: 12};
const FL_LABEL = {size: 11, lh: 16, track: 0.03};
const FL_GROUP_PAD_X = 16;
const FL_GROUP_HEAD = 30; // headroom inside a group for its corner label
const FL_GROUP_FOOT = 14;

export interface FlowNode extends DiagramNode {
  id: string;
  /** Label line(s) drawn beside the arrow leading into this node. */
  edgeLabel?: string | string[];
}

export interface FlowGroup {
  /** Corner label, rendered as given (no case transform). */
  label: string;
  /** Contiguous node ids the group encloses. */
  nodes: string[];
}

export interface FlowReturn {
  from: string;
  to: string;
  /** Label line(s) beside the return rail. */
  label?: string | string[];
}

export interface FlowDiagramProps {
  ariaLabel?: string;
  nodes: FlowNode[];
  groups?: FlowGroup[];
  returns?: FlowReturn[];
  caption?: React.ReactNode;
}

/**
 * A top-down flow at ArchitectureDiagram scale: one centered column of large-type
 * boxes, arrow labels beside the spine, dashed ownership containers, and an
 * optional dashed return rail on the right for the backflow. Nodes with
 * `detail` are clickable and drive the panel below the SVG, blending the
 * homepage figure's glanceability and the Design diagrams' interactivity.
 */
export function FlowDiagram({ariaLabel, nodes, groups = [], returns = [], caption}: FlowDiagramProps) {
  const [selectedId, setSelectedId] = useState<string | null>(
    nodes.find((n) => n.detail)?.id ?? null,
  );
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const groupOf = new Map<string, FlowGroup>();
  groups.forEach((g) => g.nodes.forEach((id) => groupOf.set(id, g)));
  const edgeLines = (n: FlowNode): string[] =>
    !n.edgeLabel ? [] : Array.isArray(n.edgeLabel) ? n.edgeLabel : [n.edgeLabel];
  const returnLines = (r: FlowReturn): string[] =>
    !r.label ? [] : Array.isArray(r.label) ? r.label : [r.label];

  const sizes = new Map(nodes.map((n) => [n.id, measureNode(n, FLOW_FONTS)]));
  const boxW = Math.max(380, ...nodes.map((n) => sizes.get(n.id)!.w));
  const groupX = EDGE + 2;
  const boxX = groupX + FL_GROUP_PAD_X;
  const boxRight = boxX + boxW;
  const groupW = boxW + 2 * FL_GROUP_PAD_X;
  const cx = boxX + boxW / 2;

  // Vertical layout: arrow gaps sized so their labels always fit; entering or
  // leaving a dashed group adds that group's label headroom / footroom.
  const rect = new Map<string, {y: number; h: number; cy: number}>();
  const groupRect = new Map<FlowGroup, {top: number; bottom: number}>();
  let y = EDGE + 2;
  nodes.forEach((n, i) => {
    const g = groupOf.get(n.id);
    const prevG = i > 0 ? groupOf.get(nodes[i - 1].id) : undefined;
    if (i === 0) {
      if (g) y += FL_GROUP_HEAD;
    } else {
      const lines = edgeLines(n).length;
      let gap = lines > 0 ? Math.max(54, lines * FL_LABEL.lh + 30) : 46;
      if (prevG && prevG !== g) gap += FL_GROUP_FOOT;
      if (g && g !== prevG) gap += FL_GROUP_HEAD;
      y += gap;
    }
    if (g && g !== prevG) groupRect.set(g, {top: y - FL_GROUP_HEAD, bottom: 0});
    const h = sizes.get(n.id)!.h;
    rect.set(n.id, {y, h, cy: y + h / 2});
    y += h;
    if (g) groupRect.get(g)!.bottom = y + FL_GROUP_FOOT;
  });
  const svgH = Math.max(y, ...[...groupRect.values()].map((r) => r.bottom)) + EDGE + 2;

  // Width: spine labels sit to the right of the arrows; the return rail sits
  // clear of both the boxes and the widest spine label.
  const maxEdgeLabelW = Math.max(
    0, ...nodes.flatMap((n) => edgeLines(n).map((line) => textWidth(line, FL_LABEL))),
  );
  const maxReturnLabelW = Math.max(
    0, ...returns.flatMap((r) => returnLines(r).map((line) => textWidth(line, FL_LABEL))),
  );
  const railX = Math.max(boxRight + FL_GROUP_PAD_X, cx + 12 + maxEdgeLabelW) + 34;
  const svgW = Math.max(
    groupX + groupW + EDGE + 4,
    cx + 12 + maxEdgeLabelW + EDGE + 4,
    returns.length ? railX + 12 + maxReturnLabelW + EDGE + 4 : 0,
  );

  const selected = selectedId ? byId.get(selectedId) : undefined;

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={560} interactive
      ariaLabel={ariaLabel ?? 'Interactive flow diagram'}
      panel={selected?.detail ? <DetailPanel node={selected} /> : null}>
      {/* dashed ownership containers */}
      {groups.map((g, i) => {
        const r = groupRect.get(g);
        if (!r) return null;
        return (
          <g key={`group-${i}`}>
            <rect x={groupX} y={r.top} width={groupW} height={r.bottom - r.top} rx={9}
              fill="none" stroke={COLOR.border} strokeWidth={1} strokeDasharray="5 4" />
            <text x={groupX + 14} y={r.top + 18} fontSize={FLOW_KIND.size}
              letterSpacing={`${FLOW_KIND.track}em`} fill={COLOR.subtle}>
              {g.label}
            </text>
          </g>
        );
      })}
      {/* spine arrows + labels */}
      {nodes.map((n, i) => {
        if (i === 0) return null;
        const prev = rect.get(nodes[i - 1].id)!;
        const cur = rect.get(n.id)!;
        const top = prev.y + prev.h;
        const bottom = cur.y;
        const lines = edgeLines(n);
        // Center labels in the visible clear gap: when the arrow crosses a
        // group border, exclude the group's foot/head bands from the segment.
        const g = groupOf.get(n.id);
        const prevG = groupOf.get(nodes[i - 1].id);
        const segTop = top + (prevG && prevG !== g ? FL_GROUP_FOOT : 0);
        const segBottom = bottom - (g && g !== prevG ? FL_GROUP_HEAD : 0);
        const mid = (segTop + segBottom) / 2;
        return (
          <g key={`spine-${i}`}>
            <line x1={cx} y1={top} x2={cx} y2={bottom - 1} stroke={COLOR.border} strokeWidth={1} />
            <path d={smallHeadDown(cx, bottom)} fill="none" stroke={COLOR.border}
              strokeWidth={1.25} strokeLinecap="round" strokeLinejoin="round" />
            {lines.map((line, j) => (
              <text key={`l-${j}`} x={cx + 12}
                y={mid - ((lines.length - 1) * FL_LABEL.lh) / 2 + j * FL_LABEL.lh + FL_LABEL.size * 0.36}
                fontSize={FL_LABEL.size} letterSpacing={`${FL_LABEL.track}em`} fill={COLOR.subtle}>
                {line}
              </text>
            ))}
          </g>
        );
      })}
      {/* return rails */}
      {returns.map((r, i) => {
        const from = rect.get(r.from)!;
        const to = rect.get(r.to)!;
        const R = 10;
        const dir = to.cy < from.cy ? -1 : 1;
        const d = [
          `M ${boxRight} ${from.cy}`,
          `L ${railX - R} ${from.cy}`,
          `Q ${railX} ${from.cy} ${railX} ${from.cy + R * dir}`,
          `L ${railX} ${to.cy - R * dir}`,
          `Q ${railX} ${to.cy} ${railX - R} ${to.cy}`,
          `L ${boxRight + 1} ${to.cy}`,
        ].join(' ');
        const lines = returnLines(r);
        const mid = (from.cy + to.cy) / 2;
        return (
          <g key={`return-${i}`}>
            <path d={d} fill="none" stroke={COLOR.border} strokeWidth={1}
              strokeDasharray="5 4" strokeLinejoin="round" />
            <path d={smallHeadLeft(boxRight, to.cy)} fill="none" stroke={COLOR.border}
              strokeWidth={1.25} strokeLinecap="round" strokeLinejoin="round" />
            {lines.map((line, j) => (
              <text key={`l-${j}`} x={railX + 12}
                y={mid - ((lines.length - 1) * FL_LABEL.lh) / 2 + j * FL_LABEL.lh + FL_LABEL.size * 0.36}
                fontSize={FL_LABEL.size} letterSpacing={`${FL_LABEL.track}em`} fill={COLOR.subtle}>
                {line}
              </text>
            ))}
          </g>
        );
      })}
      {/* nodes */}
      {nodes.map((n) => {
        const r = rect.get(n.id)!;
        return (
          <Box key={n.id} node={n} x={boxX} y={r.y} w={boxW} h={r.h} fonts={FLOW_FONTS}
            selected={selectedId === n.id}
            onSelect={n.detail ? () => setSelectedId(n.id) : undefined} />
        );
      })}
    </Figure>
  );
}

export default SystemDiagram;

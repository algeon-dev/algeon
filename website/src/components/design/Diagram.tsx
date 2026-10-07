import clsx from 'clsx';
import React, {useState} from 'react';
import styles from './styles.module.css';
import {
  HAND, SANS, handTextWidth, sansTextWidth,
  sketchHead, sketchLine, sketchPath, sketchRoundRect, sketchRoundRectLayers,
} from './sketch';

/**
 * Hand-drawn system diagrams for the docs (Excalidraw-style).
 *
 * SVG with a computed layout. Titles are set in Excalifont, whose glyph
 * advance widths are tabulated in excalifontMetrics.ts; notes and labels use
 * the page's own sans face for legibility at small sizes, with a conservative
 * width estimate. Either way text widths are deterministic on the server and
 * in the browser, so boxes are sized to their content and nothing wraps or
 * overlaps. Shapes are drawn by rough.js through sketch.ts with seeds derived
 * from node ids, so hydration reproduces the same strokes. Figures scale down
 * proportionally on narrow viewports (viewBox), with a legibility floor after
 * which the container scrolls horizontally.
 *
 * Shapes:
 * - SystemDiagram: a lane-based DAG (host / GPU swim lanes, fork & merge
 *   edges, optional region underlays for shared invariants).
 * - StackDiagram: a top-down layering with optional boundary dividers.
 * - ArchitectureDiagram: the overview page's fixed component map.
 * - FlowDiagram: a left-to-right pipeline with optional bands and return
 *   rails, for the "one statement" story on the overview and in the blog.
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

export type Face = 'hand' | 'sans';

export interface FontSpec {
  size: number;
  lh: number;
  face: Face;
  track?: number;
}

// Three scales. Titles are hand-lettered; everything a reader has to parse at
// a glance (type prefixes, notes, edge labels) is the page's sans face.
const KIND: FontSpec = {size: 10.5, lh: 16, face: 'sans', track: 0.08};
const NAME: FontSpec = {size: 17, lh: 23, face: 'hand'};
const NOTE: FontSpec = {size: 12.5, lh: 18, face: 'sans'};
export const LABEL: FontSpec = {size: 12, lh: 16, face: 'sans', track: 0.01}; // edge / lane / region labels
const PAD_X = 16;
const PAD_Y = 12;
export const EDGE = 2; // stroke bleed margin around the drawing
const INK = COLOR.fg;
const EDGE_WIDTH = 1.4;
const BOX_RADIUS = 14;
const GROUP_RADIUS = 18;

export function textWidth(text: string, font: FontSpec): number {
  return font.face === 'hand'
    ? handTextWidth(text, font.size, font.track ?? 0)
    : sansTextWidth(text, font.size, font.track ?? 0);
}

const fontFamily = (font: FontSpec) => (font.face === 'hand' ? HAND : SANS);

function noteLines(node: DiagramNode): string[] {
  if (!node.note) return [];
  return Array.isArray(node.note) ? node.note : [node.note];
}

/** Font/padding bundle so the same Box renders at compact (Design DAG) or
 * glanceable (Overview/Flow) scale. */
export interface BoxFonts {
  kind: FontSpec;
  name: FontSpec;
  note: FontSpec;
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

/** One node box: sketched rounded outline, hachure tint in the node's tone
 * over an opaque surface, hand-lettered title with sans notes, optional
 * interactivity. */
export function Box({node, x, y, w, h, selected, onSelect, fonts = COMPACT_FONTS,
  borderColor = INK, borderWidth = 1.5}: {
  node: DiagramNode;
  x: number;
  y: number;
  w: number;
  h: number;
  selected?: boolean;
  onSelect?: () => void;
  fonts?: BoxFonts;
  borderColor?: string;
  borderWidth?: number;
}) {
  const notes = noteLines(node);
  const tone = node.tone ?? 'plain';
  const toneColor = TONE[tone];
  const tinted = tone !== 'plain';
  const clickable = Boolean(onSelect);
  const contentH = (node.kind ? fonts.kind.lh : 0) + fonts.name.lh + notes.length * fonts.note.lh;
  let lineY = y + (h - contentH) / 2;
  const tx = x + fonts.padX;
  const lines: React.ReactNode[] = [];
  if (node.kind) {
    lines.push(
      <text key="kind" x={tx} y={lineY + fonts.kind.lh * 0.76} fontSize={fonts.kind.size}
        fontFamily={fontFamily(fonts.kind)} letterSpacing={`${fonts.kind.track ?? 0}em`}
        fill={COLOR.muted}>
        {node.kind.toUpperCase()}
      </text>,
    );
    lineY += fonts.kind.lh;
  }
  lines.push(
    <text key="name" x={tx} y={lineY + fonts.name.lh * 0.74} fontSize={fonts.name.size}
      fontFamily={fontFamily(fonts.name)} fill={COLOR.fg}>
      {node.name}
    </text>,
  );
  lineY += fonts.name.lh;
  notes.forEach((line, i) => {
    lines.push(
      <text key={`note-${i}`} x={tx} y={lineY + fonts.note.lh * 0.74} fontSize={fonts.note.size}
        fontFamily={fontFamily(fonts.note)} fill={COLOR.muted}>
        {line}
      </text>,
    );
    lineY += fonts.note.lh;
  });
  const {fill, outline} = sketchRoundRectLayers(`box:${node.kind ?? ''}:${node.name}`, x, y, w, h, BOX_RADIUS, {
    stroke: selected ? toneColor : borderColor,
    strokeWidth: selected ? borderWidth + 0.8 : borderWidth,
    fill: tinted ? toneColor : undefined,
    className: styles.boxRect,
  });
  const r = Math.min(BOX_RADIUS - 2, w / 2, h / 2);
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
      {/* Opaque surface first, so lane bands and edges never show through the
          text and a click anywhere inside the box reaches the group; the
          hachure sits on top at low opacity, the way a lightly shaded
          Excalidraw box reads. */}
      <rect x={x + 2} y={y + 2} width={w - 4} height={h - 4} rx={r} fill={COLOR.surface} />
      {tinted ? (
        <>
          <rect className={styles.boxTint} x={x + 2} y={y + 2} width={w - 4} height={h - 4} rx={r}
            fill={toneColor} pointerEvents="none" />
          <g className={styles.boxHatch} pointerEvents="none">{fill}</g>
        </>
      ) : null}
      {outline}
      {lines}
    </g>
  );
}

/** Edge label with a surface-colored backing so it stays legible over lines. */
export function BackedLabel({x, y, text, fill = COLOR.muted}: {x: number; y: number; text: string; fill?: string}) {
  const w = textWidth(text, LABEL);
  return (
    <g>
      <rect x={x - w / 2 - 5} y={y - 11} width={w + 10} height={15} rx={4}
        fill={COLOR.surface} opacity={0.92} />
      <text x={x} y={y} fontSize={LABEL.size} fontFamily={SANS} letterSpacing={`${LABEL.track ?? 0}em`}
        textAnchor="middle" fill={fill}>
        {text}
      </text>
    </g>
  );
}

export function Figure({caption, svgW, svgH, minW, maxW, interactive, ariaLabel, center, children, panel}: {
  caption?: React.ReactNode;
  svgW: number;
  svgH: number;
  minW: number;
  /** Displayed width cap in CSS px; defaults to the drawing's own width. */
  maxW?: number;
  interactive?: boolean;
  ariaLabel?: string;
  center?: boolean;
  children: React.ReactNode;
  panel?: React.ReactNode;
}) {
  return (
    <div className={styles.figure}>
      <figure style={{margin: 0}}>
        <div className={clsx(styles.scroll, 'thin-scrollbar')}>
          <svg viewBox={`0 0 ${svgW} ${svgH}`} width="100%" role={interactive ? 'group' : 'img'}
            aria-label={ariaLabel}
            style={{maxWidth: maxW ?? svgW, minWidth: Math.min(maxW ?? svgW, minW), display: 'block',
              margin: center ? '0 auto' : undefined, fontFamily: SANS}}>
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
  /** Short gutter label, e.g. "host" / "gpu". */
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
  /** Label drawn inside the region's top-left corner. */
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
const REGION_HEAD = 30; // headroom above nodes for the region label
const REGION_FOOT = 12;
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
              <rect x={0} y={top} width={svgW} height={h} fill={TONE[lane.tone]} opacity={0.05} />
            ) : null}
            <text x={10} y={top + h / 2 + 4} fontSize={KIND.size} letterSpacing={`${KIND.track}em`}
              fill={COLOR.muted}>
              {lane.label.toUpperCase()}
            </text>
          </g>
        );
      })}
      {/* lane separators */}
      {lanes.slice(1).map((lane) => (
        <g key={`sep-${lane.id}`} opacity={0.7}>
          {sketchLine(`sep:${lane.id}`, 0, laneTop.get(lane.id)!, svgW, laneTop.get(lane.id)!,
            {stroke: COLOR.border, strokeWidth: 1, dashed: true, roughness: 0.5})}
        </g>
      ))}
      {/* regions */}
      {regions.map((region, i) => {
        const rs = region.nodes.map((id) => rect.get(id)!);
        const rx = Math.min(...rs.map((r) => r.x)) - 12;
        const ry = Math.min(...rs.map((r) => r.y)) - REGION_HEAD;
        const rw = Math.max(...rs.map((r) => r.x + r.w)) + 12 - rx;
        const rh = Math.max(...rs.map((r) => r.y + r.h)) + REGION_FOOT - ry;
        const toneColor = TONE[region.tone ?? 'plain'];
        return (
          <g key={`region-${i}`}>
            <rect x={rx} y={ry} width={rw} height={rh} rx={GROUP_RADIUS} fill={toneColor} opacity={0.06} />
            <g opacity={0.7}>
              {sketchRoundRect(`region:${region.label}`, rx, ry, rw, rh, GROUP_RADIUS,
                {stroke: toneColor, strokeWidth: 1.2, dashed: true})}
            </g>
            <text x={rx + 14} y={ry + 18} fontSize={LABEL.size} letterSpacing={`${LABEL.track}em`}
              fill={COLOR.muted}>
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
        const stroke = connected && selected ? TONE[selected.tone ?? 'plain'] : INK;
        const style = {stroke, strokeWidth: connected ? EDGE_WIDTH + 0.5 : EDGE_WIDTH, dashed: e.dashed};
        const seed = `edge:${e.from}->${e.to}`;
        const label = (lx: number, ly: number) =>
          e.label ? <BackedLabel x={lx} y={ly} text={e.label} fill={connected && selected ? stroke : COLOR.muted} /> : null;
        if (byId.get(e.from)!.lane === byId.get(e.to)!.lane) {
          return (
            <g key={`edge-${i}`}>
              {sketchLine(seed, sx, sy, dx - 1, dy, style)}
              {sketchHead(`${seed}:head`, dx, dy, 'right', style)}
              {label((sx + dx) / 2, sy - 9)}
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
            {sketchPath(seed, elbowPath(sx, sy, midX, dy, dx), style)}
            {sketchHead(`${seed}:head`, dx, dy, 'right', style)}
            {seg2 >= seg1 ? label((midX + dx) / 2, dy - 9) : label((sx + midX) / 2, sy - 9)}
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
  const boxW = Math.max(400, ...sizes.map((s) => s.w));
  const svgW = boxW + 2 * EDGE;
  const GAP = 14;
  const BOUNDARY_H = 26;
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
      const style = {stroke: TONE.foreign, strokeWidth: 1.2, dashed: true};
      parts.push(
        <g key={`d${i}`}>
          {sketchLine(`boundary:${i}:l`, EDGE, midY, svgW / 2 - gapHalf, midY, style)}
          {sketchLine(`boundary:${i}:r`, svgW / 2 + gapHalf, midY, svgW - EDGE, midY, style)}
          <text x={svgW / 2} y={midY + LABEL.size * 0.36} fontSize={LABEL.size}
            letterSpacing={`${LABEL.track}em`} textAnchor="middle" fill={COLOR.muted}>
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
      ariaLabel={ariaLabel ?? 'Stack diagram'}
      panel={selected?.detail ? <DetailPanel node={selected} /> : null}>
      {parts}
    </Figure>
  );
}

/* ======================= ArchitectureDiagram ======================= */

const ARCH_FONTS: BoxFonts = {
  kind: {size: 11, lh: 17, face: 'sans', track: 0.08},
  name: {size: 20, lh: 27, face: 'hand'},
  note: {size: 13, lh: 19, face: 'sans'},
  padX: 20,
  padY: 15,
};

export function ArchitectureDiagram({caption}: {caption?: React.ReactNode}) {
  const colX = 140;
  const outCx = 840;
  const sourceCx = 480;

  const top: DiagramNode = {
    kind: 'your query interface',
    name: 'SQL · DataFrame · Flight SQL',
    note: 'embedded in a Rust service · remote ADBC clients',
    tone: 'plain',
  };
  const rule: DiagramNode = {
    kind: 'Algeon optimizer rule in DataFusion',
    name: 'Algeon DataFusion adapter',
    note: 'picks whole relational candidates for the GPU',
    tone: 'engine',
  };
  const gpu: DiagramNode = {
    kind: 'same process · one device · one memory grant',
    name: 'GPU execution',
    note: [
      'cuDF for the relational plan',
      'cugraph_* · cuvs_* · cuml_* UDTFs',
    ],
    tone: 'gpu',
  };
  const out: DiagramNode = {
    kind: 'same caller contract',
    name: 'Arrow batches',
    note: 'rows back to the interface you started from',
    tone: 'plain',
  };
  const disk: DiagramNode = {
    kind: 'local files',
    name: 'Local disk',
    note: 'Parquet · read-only',
    tone: 'bridge',
  };
  const source: DiagramNode = {
    kind: 'S3-compatible',
    name: 'Object storage',
    note: 'Parquet · read-only',
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
  const storeY = inputY - 26;
  const storeBottom = inputBottom + 26;
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
  const W = Math.ceil(Math.max(outRight, storeRight) + 8);

  const edge = {stroke: INK, strokeWidth: EDGE_WIDTH};
  const dashedEdge = {...edge, dashed: true};

  return (
    <Figure caption={caption} svgW={W} svgH={H} minW={560} maxW={Math.round(W * 0.82)} center
      ariaLabel="Where Algeon sits inside a DataFusion query: the optimizer rule, GPU execution, the CPU fallback, and the Parquet sources">
      {/* The main query path runs straight down the left column. */}
      {sketchLine('arch:top-rule', spineX, topBottom, spineX, ruleY - 1, edge)}
      {sketchHead('arch:top-rule:head', spineX, ruleY, 'down', edge)}
      {sketchLine('arch:rule-gpu', spineX, ruleBottom, spineX, gpuY - 1, edge)}
      {sketchHead('arch:rule-gpu:head', spineX, gpuY, 'down', edge)}

      {/* A non-selected candidate stays on DataFusion CPU execution (dashed,
          as the non-selected path) and still lands in the same Arrow stream. */}
      {sketchLine('arch:fallback', ruleRight, ruleCy, outLeft - 1, ruleCy, dashedEdge)}
      {sketchHead('arch:fallback:head', outLeft, ruleCy, 'right', edge)}
      <BackedLabel x={(ruleRight + outLeft) / 2} y={ruleCy - 12} text="unselected candidates" />
      <BackedLabel x={(ruleRight + outLeft) / 2} y={ruleCy + 20} text="stay on DataFusion CPU" />

      {/* GPU results join the same caller contract from below. */}
      {sketchPath('arch:gpu-out', [
        `M ${gpuRight} ${gpuCy}`,
        `L ${outCx - R} ${gpuCy}`,
        `Q ${outCx} ${gpuCy} ${outCx} ${gpuCy - R}`,
        `L ${outCx} ${outBottom + 1}`,
      ].join(' '), edge)}
      {sketchHead('arch:gpu-out:head', outCx, outBottom, 'up', edge)}
      <BackedLabel x={(gpuRight + outCx) / 2} y={gpuCy - 10} text="GPU results" />

      {/* Both storage cards share one KvikIO read path: a single trunk
          climbs from the storage container's top edge into the GPU. */}
      {sketchLine('arch:store-gpu', spineX, storeY, spineX, gpuBottom + 1, edge)}
      {sketchHead('arch:store-gpu:head', spineX, gpuBottom, 'up', edge)}
      <text x={spineX + 12} y={(storeY + gpuBottom) / 2 + LABEL.size * 0.36}
        fontSize={LABEL.size} fill={COLOR.muted}>
        KvikIO
      </text>

      {/* Results return outside the query path, so the loop is visible
          without crossing any card. */}
      {sketchPath('arch:out-top', [
        `M ${outCx} ${outY}`,
        `L ${outCx} ${topCy + R}`,
        `Q ${outCx} ${topCy} ${outCx - R} ${topCy}`,
        `L ${topRight + 1} ${topCy}`,
      ].join(' '), edge)}
      {sketchHead('arch:out-top:head', topRight, topCy, 'left', edge)}

      {/* The storage container groups both inputs under the shared read
          path; cards draw on top of it. */}
      <rect x={storeLeft} y={storeY} width={storeRight - storeLeft}
        height={storeBottom - storeY} rx={GROUP_RADIUS} fill={COLOR.fg} fillOpacity={0.04} />
      {sketchRoundRect('arch:store', storeLeft, storeY, storeRight - storeLeft, storeBottom - storeY, GROUP_RADIUS,
        {stroke: COLOR.muted, strokeWidth: 1.2, dashed: true})}
      {/* Boxes are last so edge endpoints tuck under their borders. */}
      <Box node={top} x={colX} y={topY} w={topSize.w} h={topSize.h} fonts={ARCH_FONTS} />
      <Box node={rule} x={colX} y={ruleY} w={ruleSize.w} h={ruleSize.h} fonts={ARCH_FONTS}
        borderColor={TONE.engine} borderWidth={2.2} />
      <Box node={gpu} x={colX} y={gpuY} w={gpuSize.w} h={gpuSize.h} fonts={ARCH_FONTS} />
      <Box node={out} x={outLeft} y={outY} w={outSize.w} h={outSize.h} fonts={ARCH_FONTS} />
      <Box node={disk} x={colX} y={diskY} w={diskSize.w} h={diskSize.h} fonts={ARCH_FONTS} />
      <Box node={source} x={sourceCx - sourceSize.w / 2} y={sourceY} w={sourceSize.w} h={sourceSize.h}
        fonts={ARCH_FONTS} />
    </Figure>
  );
}

/* ======================= FlowDiagram ======================= */

export const FLOW_FONTS: BoxFonts = {
  kind: {size: 11, lh: 17, face: 'sans', track: 0.08},
  name: {size: 19, lh: 26, face: 'hand'},
  note: {size: 13, lh: 19, face: 'sans'},
  padX: 18,
  padY: 14,
};
const FL_MIN_GAP = 72;
const FL_BAND_PAD = 18;
const FL_BAND_HEAD = 34; // headroom inside a band for its corner label
const FL_BAND_FOOT = 14;
const FL_RAIL_DROP = 34;

export interface FlowNode extends DiagramNode {
  id: string;
  /** Label line(s) drawn above the arrow leading into this node. */
  edgeLabel?: string | string[];
}

export interface FlowGroup {
  /** Corner label, rendered as given (no case transform). */
  label: string;
  /** Contiguous node ids the band encloses. */
  nodes: string[];
  tone?: Tone;
}

export interface FlowReturn {
  from: string;
  to: string;
  /** Label under the return rail. */
  label?: string;
}

export interface FlowDiagramProps {
  ariaLabel?: string;
  nodes: FlowNode[];
  groups?: FlowGroup[];
  returns?: FlowReturn[];
  caption?: React.ReactNode;
  /** Minimum displayed width before the figure scrolls horizontally. */
  minWidth?: number;
}

/**
 * A left-to-right pipeline: boxes sized to their content, arrow labels above
 * the arrows, optional dashed bands over a run of stages, and optional dashed
 * return rails below for output that feeds an earlier stage. Nodes with
 * `detail` are clickable and drive the panel below the SVG; without any, the
 * figure is a static picture.
 */
export function FlowDiagram({ariaLabel, nodes, groups = [], returns = [], caption, minWidth = 680}: FlowDiagramProps) {
  const interactive = nodes.some((n) => n.detail);
  const [selectedId, setSelectedId] = useState<string | null>(
    nodes.find((n) => n.detail)?.id ?? null,
  );
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const groupOf = new Map<string, FlowGroup>();
  groups.forEach((g) => g.nodes.forEach((id) => groupOf.set(id, g)));
  const edgeLines = (n: FlowNode): string[] =>
    !n.edgeLabel ? [] : Array.isArray(n.edgeLabel) ? n.edgeLabel : [n.edgeLabel];

  const sizes = new Map(nodes.map((n) => [n.id, measureNode(n, FLOW_FONTS)]));
  const rowH = Math.max(...[...sizes.values()].map((s) => s.h));
  const hasBands = groups.length > 0;
  const top = EDGE + 6 + (hasBands ? FL_BAND_HEAD : 0);

  // Horizontal layout: each gap is sized to the widest label line above its arrow.
  const placed = new Map<string, {x: number; y: number; w: number; h: number; cx: number; cy: number}>();
  let x = EDGE + 8 + (hasBands ? FL_BAND_PAD : 0);
  nodes.forEach((n, i) => {
    if (i > 0) {
      const lw = Math.max(0, ...edgeLines(n).map((line) => textWidth(line, LABEL)));
      x += Math.max(FL_MIN_GAP, Math.ceil(lw) + 48);
    }
    const s = sizes.get(n.id)!;
    placed.set(n.id, {x, y: top + (rowH - s.h) / 2, w: s.w, h: s.h, cx: x + s.w / 2, cy: top + rowH / 2});
    x += s.w;
  });
  const rowBottom = top + rowH;
  const railY = rowBottom + FL_RAIL_DROP;
  const contentBottom = returns.length ? railY + LABEL.lh + 6 : rowBottom;
  const svgH = contentBottom + (hasBands ? FL_BAND_FOOT : 0) + EDGE + 6;
  const svgW = x + 8 + (hasBands ? FL_BAND_PAD : 0) + EDGE;

  const selected = selectedId ? byId.get(selectedId) : undefined;
  const edge = {stroke: INK, strokeWidth: EDGE_WIDTH};

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={minWidth} interactive={interactive} center
      ariaLabel={ariaLabel ?? 'Flow diagram'}
      panel={selected?.detail ? <DetailPanel node={selected} /> : null}>
      {/* dashed bands over a run of stages */}
      {groups.map((g, i) => {
        const members = g.nodes.map((id) => placed.get(id)!);
        const bx = Math.min(...members.map((m) => m.x)) - FL_BAND_PAD;
        const bw = Math.max(...members.map((m) => m.x + m.w)) + FL_BAND_PAD - bx;
        const by = EDGE + 4;
        const bh = svgH - EDGE - 4 - by;
        const toneColor = TONE[g.tone ?? 'gpu'];
        return (
          <g key={`band-${i}`}>
            <rect x={bx} y={by} width={bw} height={bh} rx={GROUP_RADIUS} fill={toneColor} opacity={0.06} />
            <g opacity={0.7}>
              {sketchRoundRect(`band:${g.label}`, bx, by, bw, bh, GROUP_RADIUS,
                {stroke: toneColor, strokeWidth: 1.2, dashed: true})}
            </g>
            <text x={bx + 16} y={by + 20} fontSize={KIND.size} letterSpacing={`${KIND.track}em`}
              fill={COLOR.muted}>
              {g.label.toUpperCase()}
            </text>
          </g>
        );
      })}
      {/* arrows + labels */}
      {nodes.map((n, i) => {
        if (i === 0) return null;
        const a = placed.get(nodes[i - 1].id)!;
        const b = placed.get(n.id)!;
        const sx = a.x + a.w;
        const dx = b.x;
        const cy = a.cy;
        // Center labels on the visible arrow shaft, and when the arrow enters a
        // band the previous node is outside of, on the clear space before the
        // band's edge rather than on the gap as a whole.
        const entersBand = groupOf.has(n.id) && groupOf.get(n.id) !== groupOf.get(nodes[i - 1].id);
        const right = entersBand ? dx - FL_BAND_PAD : dx - 10;
        const mid = (sx + 2 + right) / 2;
        const lines = edgeLines(n);
        return (
          <g key={`edge-${n.id}`}>
            {sketchLine(`flow:${n.id}`, sx + 2, cy, dx - 2, cy, edge)}
            {sketchHead(`flow:${n.id}:head`, dx, cy, 'right', edge)}
            {lines.map((line, j) => (
              <BackedLabel key={`l-${j}`} x={mid} y={cy - 14 - (lines.length - 1 - j) * LABEL.lh} text={line} />
            ))}
          </g>
        );
      })}
      {/* return rails */}
      {returns.map((r, i) => {
        const from = placed.get(r.from)!;
        const to = placed.get(r.to)!;
        const R = 10;
        const dir = to.cx < from.cx ? -1 : 1;
        const d = [
          `M ${from.cx} ${from.y + from.h}`,
          `L ${from.cx} ${railY - R}`,
          `Q ${from.cx} ${railY} ${from.cx + R * dir} ${railY}`,
          `L ${to.cx - R * dir} ${railY}`,
          `Q ${to.cx} ${railY} ${to.cx} ${railY - R}`,
          `L ${to.cx} ${to.y + to.h + 1}`,
        ].join(' ');
        return (
          <g key={`return-${i}`}>
            {sketchPath(`rail:${r.from}->${r.to}`, d, {...edge, dashed: true})}
            {sketchHead(`rail:${r.from}->${r.to}:head`, to.cx, to.y + to.h, 'up', edge)}
            {r.label ? (
              <text x={(from.cx + to.cx) / 2} y={railY + LABEL.lh} fontSize={LABEL.size}
                letterSpacing={`${LABEL.track}em`} textAnchor="middle" fill={COLOR.muted}>
                {r.label}
              </text>
            ) : null}
          </g>
        );
      })}
      {/* nodes */}
      {nodes.map((n) => {
        const p = placed.get(n.id)!;
        return (
          <Box key={n.id} node={n} x={p.x} y={p.y} w={p.w} h={p.h} fonts={FLOW_FONTS}
            selected={selectedId === n.id}
            onSelect={n.detail ? () => setSelectedId(n.id) : undefined} />
        );
      })}
    </Figure>
  );
}

export default SystemDiagram;

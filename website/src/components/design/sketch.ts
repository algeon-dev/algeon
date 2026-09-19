import React from 'react';
import rough from 'roughjs';
import type {Drawable, Options} from 'roughjs/bin/core';
import {EXCALIFONT_FALLBACK_WIDTH, EXCALIFONT_WIDTHS} from './excalifontMetrics';

/**
 * Hand-drawn primitives for the diagrams: rough.js geometry rendered as plain
 * SVG paths, so it works during static rendering and hydrates without a DOM.
 *
 * Every shape takes a `seed` string; rough.js derives its jitter from the
 * numeric seed, so the same input draws the same strokes on the server and in
 * the browser and a re-render never redraws a box differently.
 */

/** Hand-lettered face for titles. */
export const HAND = "'Excalifont', 'Segoe Print', 'Comic Sans MS', cursive";
/** The page's own text face for notes and labels, where legibility at small
 * sizes matters more than character. */
export const SANS = 'var(--ifm-font-family-base, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif)';

const generator = rough.generator();

/** FNV-1a over the seed string, kept in rough.js's accepted seed range. */
export function seedFrom(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return (h % 2147483646) + 1;
}

/** Advance width of `text` in Excalifont at `size`, from measured glyph widths. */
export function handTextWidth(text: string, size: number, track = 0): number {
  let w = 0;
  for (const ch of text) w += EXCALIFONT_WIDTHS[ch] ?? EXCALIFONT_FALLBACK_WIDTH;
  return w * size + Math.max(0, text.length - 1) * track * size;
}

/** Advance width estimate for the system sans stack. The stack differs per
 * platform, so this over-estimates slightly (wide capitals, roomy lowercase)
 * and boxes never run tight. */
export function sansTextWidth(text: string, size: number, track = 0): number {
  let w = 0;
  for (const ch of text) {
    if (ch === ' ') w += 0.3;
    else if (ch === '·' || ch === '.' || ch === ',' || ch === ':' || ch === 'i' || ch === 'l' || ch === 'j' || ch === "'") w += 0.3;
    else if (ch >= 'A' && ch <= 'Z') w += 0.74;
    else if (ch >= '0' && ch <= '9') w += 0.58;
    else if (ch === 'm' || ch === 'w') w += 0.85;
    else if (ch === '_' || ch === '*' || ch === '(' || ch === ')' || ch === '/') w += 0.45;
    else w += 0.6;
  }
  return w * size + Math.max(0, text.length - 1) * track * size;
}

export interface SketchStyle {
  stroke: string;
  strokeWidth?: number;
  roughness?: number;
  /** Tone color for a hachure fill; omit for an unfilled outline. */
  fill?: string;
  dashed?: boolean;
  className?: string;
}

function options(seed: string, s: SketchStyle): Options {
  return {
    seed: seedFrom(seed),
    roughness: s.roughness ?? 1,
    bowing: 1,
    stroke: s.stroke,
    strokeWidth: s.strokeWidth ?? 1.5,
    fill: s.fill,
    fillStyle: 'hachure',
    hachureAngle: -41,
    hachureGap: 7,
    fillWeight: 0.55,
    preserveVertices: true,
  };
}

/** rough.js emits the hachure fill first and the outline last; the outline
 * carries the dash pattern and the caller's className (hover/selection). */
function layers(seed: string, drawable: Drawable, s: SketchStyle): {fill: React.ReactNode; outline: React.ReactNode} {
  const list = generator.toPaths(drawable);
  const node = (p: (typeof list)[number], i: number, outline: boolean) =>
    React.createElement('path', {
      key: `${seed}-${i}`,
      d: p.d,
      fill: 'none',
      stroke: p.stroke,
      strokeWidth: p.strokeWidth,
      strokeLinecap: 'round',
      strokeLinejoin: 'round',
      strokeDasharray: outline && s.dashed ? '6 5' : undefined,
      className: outline ? s.className : undefined,
    });
  const last = list.length - 1;
  return {
    fill: list.slice(0, last).map((p, i) => node(p, i, false)),
    outline: node(list[last], last, true),
  };
}

function flat(seed: string, drawable: Drawable, s: SketchStyle): React.ReactNode {
  const l = layers(seed, drawable, s);
  return [l.fill, l.outline];
}

/** Closed rounded-rectangle path; `r` is clamped to half the shorter side. */
export function roundRectPath(x: number, y: number, w: number, h: number, r: number): string {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  return [
    `M ${x + rr} ${y}`,
    `L ${x + w - rr} ${y}`,
    `Q ${x + w} ${y} ${x + w} ${y + rr}`,
    `L ${x + w} ${y + h - rr}`,
    `Q ${x + w} ${y + h} ${x + w - rr} ${y + h}`,
    `L ${x + rr} ${y + h}`,
    `Q ${x} ${y + h} ${x} ${y + h - rr}`,
    `L ${x} ${y + rr}`,
    `Q ${x} ${y} ${x + rr} ${y}`,
    'Z',
  ].join(' ');
}

/** Sketchy rounded rectangle split into fill and outline so a caller can slip
 * a legibility underlay between the hachure and the text. */
export function sketchRoundRectLayers(seed: string, x: number, y: number, w: number, h: number, r: number, s: SketchStyle) {
  return layers(seed, generator.path(roundRectPath(x, y, w, h, r), options(seed, s)), s);
}

export function sketchRoundRect(seed: string, x: number, y: number, w: number, h: number, r: number, s: SketchStyle): React.ReactNode {
  return flat(seed, generator.path(roundRectPath(x, y, w, h, r), options(seed, s)), s);
}

export function sketchLine(seed: string, x1: number, y1: number, x2: number, y2: number, s: SketchStyle): React.ReactNode {
  return flat(seed, generator.line(x1, y1, x2, y2, options(seed, s)), s);
}

/** Sketchy rendering of an arbitrary SVG path string (elbows, return rails). */
export function sketchPath(seed: string, d: string, s: SketchStyle): React.ReactNode {
  return flat(seed, generator.path(d, options(seed, s)), s);
}

/** Open arrowhead pointing in `dir`, tip at (x, y). */
export function sketchHead(seed: string, x: number, y: number, dir: 'up' | 'down' | 'left' | 'right', s: SketchStyle): React.ReactNode {
  const L = 9;
  const pts: [number, number][] =
    dir === 'down' ? [[x - 6, y - L], [x, y], [x + 6, y - L]]
    : dir === 'up' ? [[x - 6, y + L], [x, y], [x + 6, y + L]]
    : dir === 'left' ? [[x + L, y - 6], [x, y], [x + L, y + 6]]
    : [[x - L, y - 6], [x, y], [x - L, y + 6]];
  return flat(seed, generator.linearPath(pts, {...options(seed, s), roughness: 0.8}), s);
}

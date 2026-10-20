import React from 'react';
import {
  BackedLabel,
  Box,
  COLOR,
  COMPACT_FONTS,
  EDGE,
  Figure,
  LABEL,
  measureNode,
  textWidth,
} from '@site/src/components/design/Diagram';
import type {BoxFonts, DiagramNode, FontSpec} from '@site/src/components/design/Diagram';
import {SANS, sketchHead, sketchLine, sketchPath} from '@site/src/components/design/sketch';

/**
 * The decision-model post's figures: the loop that puts a hosted model's
 * typed answers back into a GPU query as a registered table, and the rank
 * shift of the bridge brands before and after that table filters them.
 * Built from the Design diagrams' box, figure, and sketch primitives so they
 * match the introduction post's figures.
 */

const HEAD = {size: 11, track: 0.08};
const INK = COLOR.fg;
const EDGE_W = 1.4;

/* ====================== DecisionLoopFigure ====================== */

const DECISION_PROBE: DiagramNode = {
  kind: 'GPU query', name: 'Probe',
  note: ['graph UDTFs over all rows', 'bounded candidate set + context'], tone: 'gpu',
};
const DECISION_MODEL: DiagramNode = {
  kind: 'hosted model', name: 'Decision model',
  note: ['fixed question set per candidate', 'choice · probability · score'], tone: 'foreign',
};
const DECISION_POLICY: DiagramNode = {
  kind: 'application rule', name: 'Policy',
  note: ['accept ≥ 0.70 with the kind', 'reject < 0.30 · else review'], tone: 'plan',
};
const DECISION_TABLE: DiagramNode = {
  kind: 'Parquet · CREATE EXTERNAL TABLE', name: 'Decision table',
  note: ['one row per candidate', 'band · kind · probabilities'], tone: 'lake',
};
const DECISION_RECOMPUTE: DiagramNode = {
  kind: 'GPU query', name: 'Recompute',
  note: ['same query joined to the table', 'all rows, no host boundary'], tone: 'gpu',
};
const DECISION_REGISTRY = 'registry: one call per distinct state';
const DECISION_RETURN = 'the recompute is the next probe';

/**
 * The decision model in the query loop. Probe, model and policy run left to
 * right; the decision table sits under the policy and the recompute turns
 * back toward the probe, so five boxes fit a 640 px figure in two rows.
 */
export function DecisionLoopFigure({caption}: {caption?: React.ReactNode}) {
  const fonts: BoxFonts = COMPACT_FONTS;
  const [probe, model, policy, table, recompute] =
    [DECISION_PROBE, DECISION_MODEL, DECISION_POLICY, DECISION_TABLE, DECISION_RECOMPUTE]
      .map((n) => measureNode(n, fonts));
  const V_GAP = 64;
  const RAIL_DROP = 34;
  const R = 10;
  const label12 = 'rows';
  const label23 = 'typed answers';
  const label34 = 'bands';
  const label45 = 'registered relation';
  const gapFor = (line: string) => Math.max(72, Math.ceil(textWidth(line, LABEL)) + 48);
  const place = (x: number, rowTop: number, rowH: number, s: {w: number; h: number}) =>
    ({x, y: rowTop + (rowH - s.h) / 2, w: s.w, h: s.h, cx: x + s.w / 2, cy: rowTop + rowH / 2});

  // The registry note sits above the model box, so row 1 starts one label line down.
  const row1Top = EDGE + LABEL.lh + 12;
  const row1H = Math.max(probe.h, model.h, policy.h);
  const P = place(EDGE + 8, row1Top, row1H, probe);
  const M = place(P.x + P.w + gapFor(label12), row1Top, row1H, model);
  const Q = place(M.x + M.w + gapFor(label23), row1Top, row1H, policy);

  const row2Top = row1Top + row1H + V_GAP;
  const row2H = Math.max(table.h, recompute.h);
  const T = place(Q.cx - table.w / 2, row2Top, row2H, table);
  const C = place(T.x - gapFor(label45) - recompute.w, row2Top, row2H, recompute);

  const railY = row2Top + row2H + RAIL_DROP;
  const svgH = railY + LABEL.lh + 8 + EDGE;
  const svgW = Math.max(Q.x + Q.w, T.x + T.w, Q.cx + 12 + Math.ceil(textWidth(label34, LABEL))) + 8 + EDGE;
  const edge = {stroke: INK, strokeWidth: EDGE_W};
  const rail = [
    `M ${C.cx} ${C.y + C.h}`,
    `L ${C.cx} ${railY - R}`,
    `Q ${C.cx} ${railY} ${C.cx - R} ${railY}`,
    `L ${P.cx + R} ${railY}`,
    `Q ${P.cx} ${railY} ${P.cx} ${railY - R}`,
    `L ${P.cx} ${P.y + P.h + 1}`,
  ].join(' ');

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={640} center
      ariaLabel="A GPU probe query runs graph UDTFs over all rows and returns a bounded candidate set with context. A hosted decision model answers a fixed question set per candidate, with a registry making one call per distinct state. An application policy turns the answers into accept, review, or reject bands, which are written to Parquet and registered as an external table. The same GPU query joined to that table recomputes over all rows, and the recompute is the next probe.">
      <text x={M.cx} y={row1Top - 10} fontSize={LABEL.size} fontFamily={SANS}
        letterSpacing={`${LABEL.track ?? 0}em`} textAnchor="middle" fill={COLOR.muted}>
        {DECISION_REGISTRY}
      </text>
      {sketchLine('decision:p-m', P.x + P.w + 2, P.cy, M.x - 2, P.cy, edge)}
      {sketchHead('decision:p-m:head', M.x, P.cy, 'right', edge)}
      <BackedLabel x={(P.x + P.w + M.x) / 2} y={P.cy - 14} text={label12} />
      {sketchLine('decision:m-q', M.x + M.w + 2, M.cy, Q.x - 2, M.cy, edge)}
      {sketchHead('decision:m-q:head', Q.x, M.cy, 'right', edge)}
      <BackedLabel x={(M.x + M.w + Q.x) / 2} y={M.cy - 14} text={label23} />
      {sketchLine('decision:q-t', Q.cx, Q.y + Q.h + 2, Q.cx, T.y - 2, edge)}
      {sketchHead('decision:q-t:head', Q.cx, T.y, 'down', edge)}
      <text x={Q.cx + 12} y={(Q.y + Q.h + T.y) / 2 + LABEL.size * 0.36} fontSize={LABEL.size}
        fontFamily={SANS} fill={COLOR.muted}>
        {label34}
      </text>
      {sketchLine('decision:t-c', T.x - 2, T.cy, C.x + C.w + 2, T.cy, edge)}
      {sketchHead('decision:t-c:head', C.x + C.w, T.cy, 'left', edge)}
      <BackedLabel x={(C.x + C.w + T.x) / 2} y={T.cy - 14} text={label45} />
      {sketchPath('decision:rail', rail, {...edge, dashed: true})}
      {sketchHead('decision:rail:head', P.cx, P.y + P.h, 'up', edge)}
      <text x={(P.cx + C.cx) / 2} y={railY + LABEL.lh} fontSize={LABEL.size} fontFamily={SANS}
        textAnchor="middle" fill={COLOR.muted}>
        {DECISION_RETURN}
      </text>
      {/* boxes last so edge ends tuck under their borders */}
      <Box node={DECISION_PROBE} x={P.x} y={P.y} w={P.w} h={P.h} fonts={fonts} />
      <Box node={DECISION_MODEL} x={M.x} y={M.y} w={M.w} h={M.h} fonts={fonts} />
      <Box node={DECISION_POLICY} x={Q.x} y={Q.y} w={Q.w} h={Q.h} fonts={fonts} />
      <Box node={DECISION_TABLE} x={T.x} y={T.y} w={T.w} h={T.h} fonts={fonts} />
      <Box node={DECISION_RECOMPUTE} x={C.x} y={C.y} w={C.w} h={C.h} fonts={fonts} />
    </Figure>
  );
}

/* ====================== RankShiftFigure ====================== */

export interface RankShiftRow {
  name: string;
  original_rank: number | null;
  postfilter_rank: number | null;
  purified_rank: number;
}

const RANK_LABEL: FontSpec = {size: 13, lh: 18, face: 'sans'};
const RANK_HEAD: FontSpec = {size: HEAD.size, lh: 16, face: 'sans', track: HEAD.track};
const RANK_ROW_H = 30;
const RANK_SLOPE_W = 130;
const RANK_GAP = 6;
const RANK_ABSENT = 'not in top 120';
const RANK_HEADS = ['ORIGINAL', 'POST-FILTER', 'PURIFIED'] as const;

/** Vertical slot of each row in a column ordered by `ranks`. Rows without a
 * rank take the slots after the ranked ones, in input order. */
function rankSlots(ranks: (number | null)[]): number[] {
  const order = ranks
    .map((rank, i) => ({rank, i}))
    .sort((a, b) => (a.rank ?? Infinity) - (b.rank ?? Infinity) || a.i - b.i);
  const slots = new Array<number>(ranks.length);
  order.forEach(({i}, slot) => {
    slots[i] = slot;
  });
  return slots;
}

/**
 * Slope chart of the bridge ranking across the original, post-filter, and
 * purified stages. Rows arrive sorted by purified rank. Each of the two
 * earlier columns reorders the same brands by its own rank, and a line joins
 * a brand's positions so the distance it moved reads as the line's slope.
 */
export function RankShiftFigure({rows, caption}: {rows: RankShiftRow[]; caption?: React.ReactNode}) {
  const original = rows.map((r) => r.original_rank);
  const postfilter = rows.map((r) => r.postfilter_rank);
  const originalSlots = rankSlots(original);
  const postfilterSlots = rankSlots(postfilter);

  const allRanks = rows.flatMap((r) => [r.original_rank, r.postfilter_rank, r.purified_rank]);
  const numW = Math.ceil(Math.max(...allRanks.map((rank) => (rank === null ? 0 : textWidth(String(rank), RANK_LABEL)))));
  const absentW = Math.ceil(textWidth(RANK_ABSENT, RANK_LABEL));
  const headW = RANK_HEADS.map((head) => Math.ceil(textWidth(head, RANK_HEAD)));
  const nameW = Math.ceil(Math.max(...rows.map((r) => textWidth(r.name, RANK_LABEL))));

  // Column 1 is right-aligned so lines leave from its numbers, column 2 is
  // centered so lines meet its numbers from both sides, and column 3 starts
  // with the rank so lines end where its rows begin.
  const col1Right = EDGE + 8 + Math.max(numW, absentW, headW[0]);
  const col2X = col1Right + RANK_SLOPE_W;
  const col2W = Math.max(numW, absentW, headW[1]);
  const col2Cx = col2X + col2W / 2;
  const col3X = col2X + col2W + RANK_SLOPE_W;
  const nameX = col3X + numW + 10;
  const svgW = Math.max(nameX + nameW, col3X + headW[2]) + 8 + EDGE;

  const top = EDGE + 14 + 10;
  const svgH = top + rows.length * RANK_ROW_H + 8 + EDGE;
  const slotY = (slot: number) => top + slot * RANK_ROW_H + RANK_ROW_H / 2;
  const baseline = (y: number) => y + RANK_LABEL.size * 0.36;
  const halfNum = (rank: number) => textWidth(String(rank), RANK_LABEL) / 2;
  const line = {stroke: COLOR.muted, strokeWidth: 1.2, roughness: 0.4};
  const headX = [col1Right, col2Cx, col3X];
  const headAnchor = ['end', 'middle', 'start'] as const;

  const absent = (key: string, x: number, y: number, anchor: 'end' | 'middle') => {
    const left = anchor === 'end' ? x - absentW : x - absentW / 2;
    // The surface backing hides any line that slopes through the bottom slots.
    return (
      <g key={key}>
        <rect x={left - 4} y={y - 9} width={absentW + 8} height={18} rx={4} fill={COLOR.surface} opacity={0.92} />
        <text x={x} y={baseline(y)} fontSize={RANK_LABEL.size} fontFamily={SANS} fontStyle="italic"
          textAnchor={anchor} fill={COLOR.muted}>
          {RANK_ABSENT}
        </text>
      </g>
    );
  };
  const rankText = (key: string, x: number, y: number, rank: number, anchor: 'end' | 'middle') => (
    <text key={key} x={x} y={baseline(y)} fontSize={RANK_LABEL.size} fontFamily={SANS}
      textAnchor={anchor} fill={COLOR.fg}>
      {rank}
    </text>
  );
  const describe = (rank: number | null) => (rank === null ? RANK_ABSENT : `rank ${rank}`);

  return (
    <Figure caption={caption} svgW={svgW} svgH={svgH} minW={640} center
      ariaLabel={`Slope chart of brand ranks in three columns: the original ranking, the post-filter ranking, and the purified ranking. ${
        rows.map((r) => `${r.name}: original ${describe(r.original_rank)}, post-filter ${describe(r.postfilter_rank)}, purified rank ${r.purified_rank}`).join('; ')}.`}>
      {RANK_HEADS.map((head, i) => (
        <text key={head} x={headX[i]} y={EDGE + 14} fontSize={HEAD.size} fontFamily={SANS}
          letterSpacing={`${HEAD.track}em`} textAnchor={headAnchor[i]} fill={COLOR.muted}>
          {head}
        </text>
      ))}
      {rows.map((r, i) => {
        const y1 = slotY(originalSlots[i]);
        const y2 = slotY(postfilterSlots[i]);
        const y3 = slotY(i);
        return (
          <g key={`lines-${r.name}`}>
            {r.original_rank !== null && r.postfilter_rank !== null &&
              sketchLine(`rank:${r.name}:a`, col1Right + RANK_GAP, y1,
                col2Cx - halfNum(r.postfilter_rank) - RANK_GAP, y2, line)}
            {r.postfilter_rank !== null &&
              sketchLine(`rank:${r.name}:b`, col2Cx + halfNum(r.postfilter_rank) + RANK_GAP, y2,
                col3X - RANK_GAP, y3, line)}
          </g>
        );
      })}
      {rows.map((r, i) => {
        const y1 = slotY(originalSlots[i]);
        const y2 = slotY(postfilterSlots[i]);
        const y3 = slotY(i);
        return (
          <g key={`text-${r.name}`}>
            {r.original_rank === null
              ? absent('original', col1Right, y1, 'end')
              : rankText('original', col1Right, y1, r.original_rank, 'end')}
            {r.postfilter_rank === null
              ? absent('postfilter', col2Cx, y2, 'middle')
              : rankText('postfilter', col2Cx, y2, r.postfilter_rank, 'middle')}
            {rankText('purified', col3X + numW, y3, r.purified_rank, 'end')}
            <text x={nameX} y={baseline(y3)} fontSize={RANK_LABEL.size} fontFamily={SANS} fill={COLOR.fg}>
              {r.name}
            </text>
          </g>
        );
      })}
    </Figure>
  );
}

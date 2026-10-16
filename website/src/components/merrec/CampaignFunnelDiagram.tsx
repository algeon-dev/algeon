import React from 'react';
import {Arrow, Card, EdgeLabel, GuideFigure, Label} from '@site/src/components/design/GuideDiagrams';
import capture from '@site/static/data/merrec-scale-audience-funnel.json';

const stage = Object.fromEntries(capture.records.map((row) => [row.stage, row]));
const n = (value: number) => value.toLocaleString('en-US');

// Columns: SQL input, GPU table function, per-shopper result, combine.
const C1 = {x: 24, w: 224};
const C2 = {x: 288, w: 224};
const C3 = {x: 552, w: 184};
const C4 = {x: 776, w: 184};
const JOIN_X = 756;
const C4_MID = C4.x + C4.w / 2;
// Rows: graph lane, taste lane, model lane, model file and held-out outcome.
const ROW = [44, 168, 292, 416];
const H = 100;
const mid = (row: number) => ROW[row] + H / 2;

/**
 * The MerRec campaign query as three candidate lanes (cuGraph, cuVS, cuML)
 * that meet in SQL joins and window ranks. Counts come from the capture.
 */
export default function CampaignFunnelDiagram(): React.JSX.Element {
  const pool = stage[5];
  const best = stage[8];
  return (
    <GuideFigure
      width={984}
      height={572}
      minWidth={720}
      maxWidth={940}
      title="Pokemon campaign query"
      label={`Pokemon campaign query. Three lanes run inside one SQL query. Graph lane: user-brand likes feed cuGraph personalized PageRank from the Pokemon brand vertex, keeping the top ${n(stage[3].shoppers)} eligible shoppers. Taste lane: 1,302 seed buyers query cuVS brute-force kNN over eligible taste vectors, giving ${n(stage[4].shoppers)} lookalikes. Model lane: Jun-Aug activity features and propensity_xgb.json feed cuML forest predict, scoring every shopper. The graph and taste audiences form a pool of ${n(pool.shoppers)}; window functions keep the 5,000 highest scores per audience; a GROUP BY counts Sep-Oct Pokemon buyers in each list. The pool's top 5,000 holds ${best.pokemon_buyers_sep_oct} buyers.`}
      caption="The top three rows are the candidate signals. Every solid box is a step of the same SQL query on one GPU; the dashed box is the model file that cuml_forest_predict loads."
    >
      <Arrow id="funnel:likes-ppr" points={[[C1.x + C1.w, mid(0)], [C2.x, mid(0)]]} />
      <Arrow id="funnel:ppr-graph" points={[[C2.x + C2.w, mid(0)], [C3.x, mid(0)]]} />
      <Arrow id="funnel:seed-knn" points={[[C1.x + C1.w, mid(1)], [C2.x, mid(1)]]} />
      <Arrow id="funnel:knn-taste" points={[[C2.x + C2.w, mid(1)], [C3.x, mid(1)]]} />
      <Arrow id="funnel:features-forest" points={[[C1.x + C1.w, mid(2)], [C2.x, mid(2)]]} />
      <Arrow id="funnel:forest-score" points={[[C2.x + C2.w, mid(2)], [C3.x, mid(2)]]} />
      <Arrow id="funnel:model-forest" points={[[C2.x + C2.w / 2, ROW[3]], [C2.x + C2.w / 2, ROW[2] + H]]} dashed />

      <Arrow id="funnel:graph-pool"
        points={[[C3.x + C3.w, mid(0)], [JOIN_X, mid(0)], [JOIN_X, 136], [C4.x, 136]]} />
      <Arrow id="funnel:taste-pool"
        points={[[C3.x + C3.w, mid(1)], [JOIN_X, mid(1)], [JOIN_X, 176], [C4.x, 176]]} />
      <Arrow id="funnel:pool-rank" points={[[C4_MID, 206], [C4_MID, ROW[2]]]} />
      <EdgeLabel x={C4_MID + 12} y={253} text="in_pool flag" anchor="start" />
      <Arrow id="funnel:score-rank" points={[[C3.x + C3.w, mid(2)], [C4.x, mid(2)]]} />
      <Arrow id="funnel:rank-evaluate" points={[[C4_MID, ROW[2] + H], [C4_MID, ROW[3]]]} />
      <Arrow id="funnel:outcome-evaluate" points={[[C3.x + C3.w, mid(3)], [C4.x, mid(3)]]} />

      <Card x={C1.x} y={ROW[0]} w={C1.w} h={H} kind="SQL · likes" name="User-brand likes"
        note={['item_like before September', 'shopper → brand edges']} />
      <Card x={C1.x} y={ROW[1]} w={C1.w} h={H} kind="SQL · mr_taste64" name="Seeds and candidates"
        note={['1,302 Jul-Aug Pokemon buyers', `${n(stage[1].shoppers)} eligible shoppers`]} />
      <Card x={C1.x} y={ROW[2]} w={C1.w} h={H} kind="SQL · features" name="Shopper activity"
        note={['Jun-Aug views, likes,', 'cart intents and purchases']} />

      <Card x={C2.x} y={ROW[0]} w={C2.w} h={H} kind="cuGraph" name="Personalized PageRank"
        note={['Restarts at the Pokemon', 'brand vertex']} tone="graph" />
      <Card x={C2.x} y={ROW[1]} w={C2.w} h={H} kind="cuVS" name="Brute-force kNN"
        note={['20 nearest neighbors', 'of each seed buyer']} tone="gpu" />
      <Card x={C2.x} y={ROW[2]} w={C2.w} h={H} kind="cuML" name="Forest predict"
        note={['XGBoost purchase', 'propensity per shopper']} tone="gpu" />
      <Card x={C2.x} y={ROW[3]} w={C2.w} h={H} kind="Model file" name="propensity_xgb.json"
        note={['Trained in the buyer', 'shortlist example']} tone="lake" dashed />

      <Card x={C3.x} y={ROW[0]} w={C3.w} h={H} kind="Graph affinity" name={`Top ${n(stage[3].shoppers)}`}
        note={['by PageRank score']} tone="graph" />
      <Card x={C3.x} y={ROW[1]} w={C3.w} h={H} kind="Taste lookalikes" name={`${n(stage[4].shoppers)} shoppers`}
        note={['distinct neighbors']} tone="gpu" />
      <Card x={C3.x} y={ROW[2]} w={C3.w} h={H} kind="Model score" name="Every shopper"
        note={['0 without Jun-Aug', 'activity']} tone="gpu" />
      <Card x={C3.x} y={ROW[3]} w={C3.w} h={H} kind="Held-out outcome" name="Sep-Oct buyers"
        note={[`${n(stage[1].pokemon_buyers_sep_oct)} bought Pokemon`, 'after the cutoff']} />

      <Card x={C4.x} y={106} w={C4.w} h={H} kind="SQL join" name="Candidate pool"
        note={['graph ∪ taste', `${n(pool.shoppers)} shoppers`]} />
      <Card x={C4.x} y={ROW[2]} w={C4.w} h={H} kind="SQL window" name="Top 5,000"
        note={['by score, ranked inside', 'each audience']} />
      <Card x={C4.x} y={ROW[3]} w={C4.w} h={H} kind="SQL GROUP BY" name="Score 8 lists"
        note={['Pool top 5,000:', `${best.pokemon_buyers_sep_oct} buyers, ${(best.purchase_rate * 100).toFixed(2)}%`]}
        tone="engine" />

      <Arrow id="funnel:legend-flow" points={[[C1.x, 548], [C1.x + 48, 548]]} />
      <Label x={C1.x + 60} y={552}>Data flow inside the query</Label>
      <Arrow id="funnel:legend-file" points={[[C2.x + 40, 548], [C2.x + 88, 548]]} dashed />
      <Label x={C2.x + 100} y={552}>Model file loaded at execution</Label>
    </GuideFigure>
  );
}

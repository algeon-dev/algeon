---
title: "Decision Models in the Query Loop: Model Answers Go Back into the SQL, and the Controls Check What Changed"
sidebar_label: "Decision Models in the Query Loop"
description: "Two passes of model answers about catalog entries rebuild a 5,000-brand ranking on the GPU, a merge pass corrects 18 brand fan counts, and a hand-written list and a second window check what the answers changed."
slug: decision-model-in-the-loop
date: 2026-10-19
tags: [jev, cugraph]
authors: morris
---

import CodeBlock from '@theme/CodeBlock';
import Link from '@docusaurus/Link';
import Table from '@site/src/theme/MDXComponents/Table';
import {DecisionLoopFigure, RankShiftFigure} from '@site/src/components/blog/DecisionFigures';
import bridges from '@site/static/data/merrec-decision-bridges.json';
import twins from '@site/static/data/merrec-decision-twins.json';

export const n = value => value.toLocaleString('en-US');
export const usd = value => `$${value.toFixed(4)}`;
export const all = bridges.windows.all;
export const sepOct = bridges.windows.sep_oct;
export const stage = (window, name) => window.stages.find(item => item.name === name);
export const purifiedTop = all.ranking.filter(row => row.purified_rank !== null && row.purified_rank <= 15)
  .sort((a, b) => a.purified_rank - b.purified_rank);
export const byName = name => all.ranking.find(row => row.name === name);
export const claires = all.review_queue.find(row => row.name === "Claire's");
export const dior = twins.clusters.find(row => row.canonical_name === 'Christian Dior');
export const estimate = twins.full_pairwise_cost_estimate;
export const unjudged = all.ranking.filter(row => row.purified_rank !== null && row.purified_rank <= 120 && row.band === null).length;

In the [MerRec case study](/guides/datasets/merrec#bridges), we rank 5,000 brands
by betweenness centrality in a graph of shared fans to find the brands that
connect different shopping communities. But when you run this on real marketplace
data, you run into an immediate catalog problem: twelve of the top twenty entries
aren't independent brands at all. They are retailers (Walmart, JCPenney, Macy's),
store brands (Mainstays), or catch-all bucket labels that sellers select from
the brand dropdown (As Seen on TV, Customized & Personalized, Eyeshadow).

Because shoppers across every category buy from Walmart, Walmart creates artificial
shortcuts between worlds that no actual manufacturer provides. If you want an
honest picture of brand affinity, you have to get those catalog artifacts out of the
graph. This post walks through how we pull them out, why naive filtering fails, and
how the ranking shifts once the graph is rebuilt.
{/* truncate */}

A decision model does part of the work. Distinguishing a catch-all label from a
real brand requires subjective judgment about messy catalog text, and a hosted
decision model ([Jev 1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13))
can answer that for a fraction of a cent. But the heavy lifting still belongs to the
query engine: we have to rebuild the entire graph without the rejected nodes, and
we need rigorous controls to verify what the model actually changed. On Algeon,
recomputing the graph across all 182.9 million like events takes just nine seconds,
making it practical to run iterative passes, evaluate alternative time windows, and
benchmark against baseline controls.

## Deleting from the output is not enough

The most tempting shortcut is post-filtering: just run the query and add a
`WHERE brand NOT IN (...)` on the final results. But graph metrics do not work
that way.

Betweenness centrality measures how often a node sits on the shortest paths
between pairs of other nodes. When Walmart is in the graph, millions of shortest
paths route straight through it. That inflates Walmart's score, but it also
depresses or distorts the centrality of every other brand around it. Simply
deleting Walmart from the final output table leaves everyone else's distorted
scores untouched. To get true shortest paths, you must remove the node from the
graph itself and let the network reroute.

Because of that, the model's decisions have to feed back into the query. The original
query uses a CTE to construct the fan graph; the rebuilt query joins that CTE against
a decisions table, pruning any node that the policy rejected or flagged for review.

<CodeBlock language="sql" className="no-collapse" folds={[
  {from: '), top_brands AS (', label: 'the rest of the bridges query, unchanged'},
]}>{bridges.sql.recompute}</CodeBlock>

## The loop

We structure the workflow as a five-step loop, where two of the steps are standard SQL queries.

<DecisionLoopFigure
  caption={<>A probe query bounds the set of entries worth asking about and
  collects their context. A hosted model answers a fixed question set per
  entry. The application's policy turns the answers into bands, the bands
  become a Parquet table registered with the server, and the recompute is
  the original query joined to that table.</>}
/>

Calling an LLM on every one of the 5,000 brands would be wasteful and slow.
Instead, we use a SQL probe, which is just the original bridges query, to identify
the top 120 nodes dominating centrality. A second SQL query then gathers catalog
context for each candidate: active listings, distinct categories, top category paths,
mean price, and four representative titles. The model only inspects this catalog text,
never the graph topology or centrality scores.

For each candidate, the model answers three structured questions:
1. Classification into one of four kinds (standalone brand, multi-brand retailer, superstore house brand, generic catch-all) with a confidence probability.
2. Direct probability for a binary check: is this a standalone brand?
3. A score from 0 to 3 indicating whether the entry conflates unrelated communities.

Our application policy cross-checks these answers deterministically: a probability of
{bridges.policy.accept_min.toFixed(2)} or higher paired with `standalone_brand` is accepted,
anything below {bridges.policy.review_min.toFixed(2)} is rejected, and ambiguous cases
(such as high probability alongside an unexpected kind) route to a human review queue.
Items in the review queue remain excluded from the graph until manually confirmed.

To keep queries deterministic and inexpensive, every decision is cached in a registry
keyed by the hash of the catalog context. Because LLM responses can fluctuate across runs,
caching ensures that subsequent recomputes are stable, reproducible, and incur zero model latency.

## Two passes rebuild the ranking

When we ran the first pass and dropped the rejected entries, the graph recomputed, and
{all.passes[1].new_entries} new brands surfaced into the top 120 that were previously
masked by the superstores. A second pass judged those newcomers and recomputed once more,
leaving only {unjudged} unjudged entries at the tail of the top 120 (which a third pass
could clean up if needed).

<Table>
<thead><tr><th>Pass</th><th>Entries</th><th>New</th><th>Accept</th><th>Review</th><th>Reject</th><th>Model calls</th><th>Cost</th><th>Wall</th></tr></thead>
<tbody>{all.passes.map((pass, index) => <tr key={pass.pass}><td>{pass.pass}</td><td>{all.decisions[index].entities}</td><td>{pass.new_entries}</td><td>{pass.new_band_counts.accept}</td><td>{pass.new_band_counts.review}</td><td>{pass.new_band_counts.reject}</td><td>{pass.api_calls}</td><td>{usd(all.decisions[index].cost_usd)}</td><td>{all.decisions[index].wall_s.toFixed(1)} s</td></tr>)}</tbody>
</Table>

<RankShiftFigure
  rows={purifiedTop.map(row => ({name: row.name, original_rank: row.original_rank, postfilter_rank: row.postfilter_rank, purified_rank: row.purified_rank}))}
  caption={<>The purified top 15 and where each brand stood in the published
  ranking and in the published ranking with the rejected entries deleted
  after the fact. Post-filtering keeps 11 of the 15 but in another order,
  and it cannot place Bella, which was not in the original top 120.</>}
/>

The structural impact on the ranking is striking. Samsung jumps from sixth to first place,
multiplying its betweenness centrality by {(byName('Samsung').purified_betweenness / byName('Samsung').original_betweenness).toFixed(1)}x.
Even more telling is what happens to personal care and tool makers: Conair leaps from
rank {byName('Conair').original_rank} to {byName('Conair').purified_rank}, and Black & Decker
climbs from {byName('Black & Decker').original_rank} to {byName('Black & Decker').purified_rank}.
Once artificial retail hubs are cleared out, practical household and hardware brands emerge
as the true bridges connecting disparate shopping circles.

The review queue catches borderline cases where automatic decisions would be risky.
For instance, Claire's received a standalone brand probability of {claires.is_standalone_brand.toFixed(2)},
yet its classification picked `multi_brand_retailer` at {claires.entity_kind_p.toFixed(2)}.
Because the signals conflicted, our policy safely held it out of the graph and flagged it
for human review. Across both passes, {all.review_queue.length} entries entered this queue.

## Validating with a hand list and a second window

Whenever you insert an LLM into an analytical pipeline, healthy engineering skepticism
demands two sanity checks:
1. Could we have achieved the same result with a simple hand-curated denylist?
2. Does the model generalize across time, or is it brittle to changing data?

To test the first question, we compared the model against a curated list of forty-nine
retailers and catch-all labels. Recomputing with either approach yields almost the same
macro outcome: 13 of the top 15 purified brands are identical, and hardware brands rise in
both. But the differences highlight where automation shines: the model caught
{all.denylist_vs_model.only_model.length} subtle catch-alls that the human list missed
(such as Bioworld, Custom Accessories, and Pink Rose), while explicitly flagging uncertain
brands in the review queue rather than forcing a silent guess.

The second check tested temporal generalization using a September–October window.
Because shopping behavior evolves, the probe surfaced a different top 120. Here, our
cached registry paid off: {sepOct.totals.registry_hits} of {sepOct.totals.entities} lookups
were instant cache hits, and the remaining {sepOct.totals.api_calls} calls cost just
{usd(sepOct.totals.cost_usd)}. Meanwhile, the static human list began to decay immediately—it
contained {sepOct.denylist_vs_model.denylist_unmatched.length} names absent from the new
window and failed to flag {sepOct.denylist_vs_model.only_model.length} seasonal retailers
like Amazon, Sephora, and QVC. A static list quickly goes stale; the query loop adapts
dynamically to whatever the probe returns.

## Solving entity fragmentation with the same loop

The exact same loop pattern tackles another classic data engineering headache: duplicate
and fragmented entities. In marketplace catalogs, brands frequently exist under multiple
variations, such as TXT vs. Tomorrow X Together, Bape vs. A Bathing Ape, Dior vs. Christian Dior.
Every analytical metric computed on those brands is splintered across fragments.

Pure string matching cannot resolve acronyms or rebrands, while prompting a model on all
{n(estimate.pairs)} possible pairs across 20,000 brands is economically out of the question,
costing roughly ${Math.round(estimate.estimated_cost_usd).toLocaleString('en-US')}.

Once again, a SQL probe makes the problem tractable. We run a pairing query that computes
fan Jaccard similarity across the top 2,000 brands, narrowing the candidate pool down to the
120 highest-overlap pairs. The model then evaluates only those 120 candidates using their
catalog metadata, completing in {twins.decisions[0].wall_s.toFixed(1)} seconds for just
{usd(twins.decisions[0].cost_usd)}. The policy accepts {twins.band_counts.accept} pairs,
flags {twins.band_counts.review} for review, and rejects {twins.band_counts.reject}.
Crucially, the model avoids merging related entities that share heavy fan overlap but are
distinct commercial entities, such as Hello Kitty and Sanrio, or Konami and Yu-Gi-Oh!.

We treat the accepted pairs as edges in a graph. Running `cugraph_weakly_connected_components`
in SQL assigns each cluster a canonical identifier, and an aggregation over like events calculates
the true deduplicated audience size.

<Table>
<thead><tr><th>Brand</th><th>Most fans on one id</th><th>Both ids summed</th><th>Distinct fans</th><th>Gain</th></tr></thead>
<tbody>{twins.clusters.slice(0, 6).map(row => <tr key={row.canonical_id}><td>{row.canonical_name}</td><td>{n(row.primary_fans)}</td><td>{n(row.naive_sum_fans)}</td><td>{n(row.exact_union_fans)}</td><td>+{row.gain_pct.toFixed(1)}%</td></tr>)}</tbody>
</Table>

The correction is substantial. If an analytics report only tracked the primary ID for Dior,
it missed nearly a third of the brand's total followers. On the other hand, if a naive script
summed both IDs without deduplication, it double-counted a quarter of the audience.
The full table of {twins.clusters.length} unified clusters is available on the
[case study page](/guides/datasets/merrec#brand-twins).

As a baseline comparison, standard trigram similarity across all 20,000 brands only recovers
{twins.trigram_control.accepted_found} of the {twins.trigram_control.accepted_aliases} merged
pairs, completely missing non-textual aliases like TXT, Bape, Dior, and LOFT. Furthermore, the
trigram heuristic produces false positives by eagerly merging sub-brands like Air Jordan with
Jordan, or Crystal with Crystal Rock.

## Practical lessons for decision loops

The design pattern here reflects a clear separation of concerns between models and engines:
let the model handle fuzzy semantic judgment on small, bounded batches, and let the GPU query
engine handle large-scale filtering, graph algorithms, and recomputation.

- **Use SQL probes to bound LLM costs.** Evaluating 120 targeted pairs cost less than a cent;
  brute-forcing 20,000 names would have cost ten thousand dollars. The query engine acts as
  an intelligent funnel.
- **Fast recomputes enable honest validation.** Because rebuilding the 183M-event graph takes
  only nine seconds on the GPU without host data transfers, running multi-pass refinements,
  historical comparisons, and sanity-check controls is cheap enough to be standard practice.
- **Keep decisions typed and deterministic.** Models should return structured classifications
  and confidence scores, not freeform text. Caching decisions by context hash eliminates
  nondeterministic query drift, while a review band keeps human oversight where uncertainty is high.

When the database engine is fast enough to make graph recomputation trivial, you do not need
to compromise with inaccurate post-query hacks. You can feed verified model judgments directly
back into the query loop and compute exact answers.

## Reproduce

Both workflows are documented in depth on the [MerRec case study](/guides/datasets/merrec#bridges)
page with their full SQL, data tables, and provenance. The dataset files
([bridges](/data/merrec-decision-bridges.json) and [twins](/data/merrec-decision-twins.json))
contain all question sets, decision policies, executed queries, and pipeline metrics under CC BY-NC terms.
You can find the runnable runner in the repository under
[`fixture/demo/decision_models`](https://github.com/algeon-dev/algeon/tree/main/fixture/demo/decision_models).
Running it live requires your own model endpoint; you can also use replay mode to rerun the exact
SQL queries against pre-recorded decisions without making external API calls.

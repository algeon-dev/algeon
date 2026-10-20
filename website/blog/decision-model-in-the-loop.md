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

The [MerRec case study](/guides/datasets/merrec#bridges) ranks 5,000 brands
by betweenness centrality in a graph of shared fans, to find the brands that
connect shopping worlds. Twelve of the first twenty entries in that
ranking are retailers (Walmart, JCPenney, Macy's), house brands (Mainstays)
and catch-all labels that sellers pick from the brand dropdown (As Seen on
TV, Customized & Personalized, Eyeshadow). Shoppers from every world like
each of them, which makes each one a short cut between worlds that no
manufacturer provides. This post follows what it takes to get those entries
out of the ranking, and what changes when they are gone.
{/* truncate */}

A decision model does part of the work. Telling a catch-all label from a
brand is a judgment about catalog text, and a hosted decision model
([Jev 1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13)) answers it
for a fraction of a cent. The rest falls on the query engine: the graph has
to be rebuilt without the rejected entries, and the rebuilt graph has to be
checked against something other than the model. On Algeon a recompute over
all 182.9 million like events takes nine seconds, which leaves room for a
control run, a second window and a second pass.

## Deleting from the output is not enough

Betweenness counts how often a node lies on the shortest paths between
other nodes. Walmart's score comes from paths that run through Walmart, and
so does part of every other node's score, because those same paths decide
which brands are on the way between two worlds. Removing Walmart from the
result table leaves every other number as it was. Removing Walmart from the
graph moves the paths, and with them the ranking.

Because of that, the model's answers go into the query. The published query has one CTE that
builds the fan graph. The rebuilt query joins that CTE to a table of
decisions and drops the entries the policy rejected or held for review.

<CodeBlock language="sql" className="no-collapse" folds={[
  {from: '), top_brands AS (', label: 'the rest of the bridges query, unchanged'},
]}>{bridges.sql.recompute}</CodeBlock>

## The loop

Two of the five steps in the figure are ordinary SQL queries.

<DecisionLoopFigure
  caption={<>A probe query bounds the set of entries worth asking about and
  collects their context. A hosted model answers a fixed question set per
  entry. The application's policy turns the answers into bands, the bands
  become a Parquet table registered with the server, and the recompute is
  the original query joined to that table.</>}
/>

The probe is the published bridges query itself, and its output is the top
120 entries. A context query joins each of them to its catalog footprint:
listings, distinct categories, the three largest category paths, mean price
and four sample titles. The model sees that text and no graph numbers.

For each entry the model answers three questions, each with a fixed shape.
The first picks one of four kinds (standalone brand, multi-brand retailer,
superstore house brand, generic catch-all) and returns a probability for
the pick. The second returns a probability for a yes-or-no question, whether
the entry is a standalone brand. The third returns a score from 0 to 3 for
how much the entry conflates unrelated communities. The application's policy
reads the second answer and checks it against the first:
{bridges.policy.accept_min.toFixed(2)} or more with the kind `standalone_brand`
is accepted, below {bridges.policy.review_min.toFixed(2)} is rejected, and
everything else goes to a review queue, including a high probability paired
with another kind. Review entries stay out of the graph until a person looks
at them.

Every answer is cached in a registry keyed by a hash of the entry's context
text, and a later run over the same entry makes no call. The model's answers
to the same entry vary slightly between calls; without the registry a rerun
would be a new sample.

## Two passes rebuild the ranking

The first pass judged the published top 120 and the recompute dropped the
rejected and review entries. The rebuilt top 120 contained {all.passes[1].new_entries}
entries the first pass had never seen, so a second pass judged those and
recomputed once more. That recompute brought {unjudged} further entries into
the top 120 that no pass has judged; a third pass would take them.

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

Samsung moves from sixth to first with
{(byName('Samsung').purified_betweenness / byName('Samsung').original_betweenness).toFixed(1)}
times its betweenness. Conair rises from {byName('Conair').original_rank} to
{byName('Conair').purified_rank} and Black & Decker from
{byName('Black & Decker').original_rank} to {byName('Black & Decker').purified_rank}.
Two hardware brands moved near the top of a ranking of the brands that
connect shopping worlds, once the superstore labels left the graph.

An entry lands in the review queue when the model's two answers disagree or
the probability sits in the band. Claire's has a standalone probability of
{claires.is_standalone_brand.toFixed(2)} and a kind of `multi_brand_retailer` at
{claires.entity_kind_p.toFixed(2)}; the policy excludes it from the graph and lists
it for a person. The {all.review_queue.length} entries
in the queue from both passes, with their context, are in the data file.

## A list and a second window

One control is a hand-written list of forty-nine retailer and catch-all
names for this window, run through the same recompute. The list
and the model exclude {all.denylist_vs_model.overlap} of the same entries. The
model also excludes {all.denylist_vs_model.only_model.length} the list missed
(Bioworld, Custom Accessories, Pink Rose and nine more), and the list
excludes {all.denylist_vs_model.only_denylist.length} the model accepted (Charming
Charlie, Cherokee, Loot Crate, Rue21). The two purified rankings share 13 of
their top 15. The structural effect, the hardware brands rising, is the same
either way. The model produces the list more cheaply than a person and
marks in the review queue where it is unsure.

The other check is a second window. On September to October the probe
returns a different top 120, {sepOct.totals.registry_hits} of the
{sepOct.totals.entities} lookups across both passes hit the registry, and the
remaining {sepOct.totals.api_calls} calls cost {usd(sepOct.totals.cost_usd)}. The
hand list, written for the first window, has {sepOct.denylist_vs_model.denylist_unmatched.length}
names that do not appear in this window's top 120 and misses
{sepOct.denylist_vs_model.only_model.length} entries the model rejects, among them
Amazon, Sephora and QVC. The list fits one ranking, while the model answers
for whatever the probe returns next.

## The same loop merges split brands

The [brand twins query](/guides/datasets/merrec#brand-twins) pairs entries
whose fans overlap. Many of the highest pairs are one brand entered twice,
TXT and Tomorrow X Together, Bape and A Bathing Ape, Dior and Christian
Dior, and every fan count for such a brand is computed on a fragment. A
string comparison misses the acronyms and the rebrands. Asking a model about
all {n(estimate.pairs)} pairs of 20,000 names is not an option either; at this
run's cost per decision it would come to about
${Math.round(estimate.estimated_cost_usd).toLocaleString('en-US')}.

The probe keeps the question affordable. The pairing query, with
the rejected entries of the bridges run removed from the fan graph, returns
the 120 pairs with the highest fan Jaccard among the 2,000 most liked
brands. The model judged those 120 with each pair's score and both entries'
catalog context, for {usd(twins.decisions[0].cost_usd)} in
{twins.decisions[0].wall_s.toFixed(1)} s. The policy merges {twins.band_counts.accept},
holds {twins.band_counts.review} for review and leaves {twins.band_counts.reject}
apart. The pairs left apart include the ones a fan-overlap rule alone would
merge: Hello Kitty and Sanrio, Konami and Yu-Gi-Oh!, Panini and Topps.

The accepted pairs are edges. `cugraph_weakly_connected_components` assigns
each entry the smallest id of its component, and a scan of every like event
counts, per merged brand, the exact number of distinct users who liked any
member.

<Table>
<thead><tr><th>Brand</th><th>Most fans on one id</th><th>Both ids summed</th><th>Distinct fans</th><th>Gain</th></tr></thead>
<tbody>{twins.clusters.slice(0, 6).map(row => <tr key={row.canonical_id}><td>{row.canonical_name}</td><td>{n(row.primary_fans)}</td><td>{n(row.naive_sum_fans)}</td><td>{n(row.exact_union_fans)}</td><td>+{row.gain_pct.toFixed(1)}%</td></tr>)}</tbody>
</Table>

Whichever id a report used for Dior, it was missing about a third of the
brand's fans, and a report that added the two ids would have counted a
quarter of its total twice. These are exact counts over every like event,
and the whole table of {twins.clusters.length} merged brands is on the
[case study page](/guides/datasets/merrec#brand-twins).

The control here is trigram similarity over all 20,000 names. It finds
{twins.trigram_control.accepted_found} of the {twins.trigram_control.accepted_aliases}
pairs the policy merged and misses TXT, Bape, Dior, LOFT, HERMES and
Thierry Mugler. The pairs it proposes that the policy did not merge are
sub-brands and near names, Air Jordan and Jordan, Crystal and Crystal Rock.

## What the model is for

The model answers a bounded set of questions about catalog text, with a
probability the application can act on, and the GPU does the rest: the probe
that bounds the set, the recompute that turns the answers into a new graph,
and the controls that show how much of the change depends on the model's
particular answers.

- The probe limits the cost. Judging the 120 pairs it returned cost under a
  cent; judging every pair of 20,000 names would cost about ten thousand
  dollars.
- The recompute is cheap enough to run several times. A control list, a
  second window and a second pass were each one more query, nine seconds on
  the full window and four on the shorter one, and every recompute ran
  natively on the GPU with zero host boundaries.
- The answers are typed. A choice with a probability, a probability for a
  question, a score. The policy is a few lines the application owns, and a
  person handles the review band.

The question set, the policy and the recompute belong to the application,
and the list control shows the result with no model at all. The loop needs
the recompute and the controls, at a cost below the question itself, and it
works with any model that returns typed answers.

## Reproduce

Both follow-ups are on the [MerRec case study](/guides/datasets/merrec#bridges)
page with their SQL, tables and provenance. The data files
([bridges](/data/merrec-decision-bridges.json), [twins](/data/merrec-decision-twins.json))
hold the question sets, the policy, the executed SQL, the stage metrics,
every judgment and every ranking, under the dataset's CC BY-NC terms. The
runner is in the repository under
[`fixture/demo/decision_models`](https://github.com/algeon-dev/algeon/tree/main/fixture/demo/decision_models).
A live run needs a model endpoint of your own; its replay mode reruns the
queries from that run's capture with no model calls.

# MerRec derived files

The `merrec-*.json` files in this directory are query results computed from
the MerRec dataset:

- Source: [mercari-us/merrec](https://huggingface.co/datasets/mercari-us/merrec),
  revision `dd2b3c6f6f19d4814bb839ee6e8ab41926a14343`.
- Copyright Mercari, Inc.
- License: [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

Each file is Adapted Material under that license and is distributed under the
same terms. They contain brand names, item titles, category labels, counts,
model scores and coordinates computed from the dataset, together with the SQL,
query IDs, timings and server configuration that produced them. They are not
covered by the Apache License 2.0 that applies to the rest of this repository.
Each file carries its terms in a top-level `license` field.

The dataset is described in: Lichi Li, Zainul Abi Din, Zhen Tan, Sam London,
Tianlong Chen and Ajay Daptardar. *MerRec: A Large-scale Multipurpose Mercari
Dataset for Consumer-to-Consumer Recommendation Systems*. KDD 2025.

Files covered:

- `merrec-agent-session.json`
- `merrec-barbie-control-seeds.json`
- `merrec-barbie-listing-check.json`
- `merrec-brand-map.json`
- `merrec-decision-bridges.json`
- `merrec-decision-twins.json`
- `merrec-pokemon-interest-check.json`
- `merrec-scale-ann.json`
- `merrec-scale-audience-funnel.json`
- `merrec-scale-barbie-tie-ins.json`
- `merrec-scale-brand-overlap.json`
- `merrec-scale-brand-twins.json`
- `merrec-scale-gmv-forecast.json`
- `merrec-scale-pokemon-lookalikes.json`
- `merrec-scale-propensity-xgboost.json`
- `merrec-scale-propensity.json`
- `merrec-scale-session-audit.json`
- `merrec-scale-shopper-personas.json`
- `merrec-scale-shopping-worlds.json`
- `merrec-scale-world-bridges.json`

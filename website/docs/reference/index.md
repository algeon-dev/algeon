---
sidebar_position: 1
slug: /reference
description: Reference material for GPU execution support boundaries, benchmark evidence, and contracts.
---

# Reference

## Available reference pages

- [Native SQL Support](./native-sql-support.mdx) - which relational
  operators, expressions, aggregates, windows, joins, and sources run on the
  GPU.
- [Native Evaluation Semantics](./native-evaluation-semantics.mdx) - which
  native expressions can fail a query, on which rows, and how `CASE`, `AND`
  and `OR` limit that.
- [Data Type Support](./data-type-support.md) - which Arrow types the GPU
  path holds or only carries, and what the UDTF families accept.
- [GPU UDTF Catalog API](./gpu-udtf-catalog.mdx) - the common
  cuGraph, cuVS and cuML catalog, descriptor, and static-validation schema reference.
- [GPU Coverage Report](./gpu-coverage-report.mdx) - the columns, JSON
  fields, blocking kinds, remedies and runtime caveats of
  `algeon_explain_coverage` and `EXPLAIN GPU`.
- [Benchmarks](./benchmarks/index.mdx) - MerRec UDTF execution,
  TPC-H and TPC-DS SF100 runs, and concurrent server streams.
- [Glossary](./glossary.md) - one-line definitions of Algeon-specific terms,
  linking to the owning contract pages.
- [Roadmap](./roadmap.md) - what is being worked on next, what waits for an
  upstream release or an adopter, and what waits for hardware.

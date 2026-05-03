---
sidebar_position: 3
description: Arrow, cuDF, and cuGraph data type support boundaries for native DataFusion Nexus execution.
---

# Data Type Support

DataFusion Nexus checks Arrow data types before a whole relational candidate
enters native execution. A candidate with an unsupported type keeps the
executable DataFusion baseline. Unified Iceberg scans retain their executable
CPU delegate when native type admission does not select acceleration. A
`NoDataFusionCpu` final-plan requirement instead rejects a completed plan that
retains DataFusion CPU execution.

This page covers column representation. Operator and function support is
documented separately in [Native SQL Support](./native-sql-support.mdx), and
the exact physical plan should be checked with
[`nexus_explain_coverage`](../integration/gpu-coverage-validation.mdx).

## Native relational columns

The native engine has a cuDF representation for these Arrow data types:

- `Boolean`
- Signed integers: `Int8`, `Int16`, `Int32`, `Int64`
- Unsigned integers: `UInt8`, `UInt16`, `UInt32`, `UInt64`
- Floating point: `Float32`, `Float64`
- `Decimal128` with valid Arrow decimal precision and scale
- Strings: `Utf8`, `LargeUtf8`, `Utf8View`
- `Date32`, `Date64`
- `Time32` and `Time64` at their Arrow-supported resolutions
- `Timestamp` at second, millisecond, microsecond, or nanosecond resolution
- `Duration` at second, millisecond, microsecond, or nanosecond resolution
- `List` / `LargeList` of UTF-8 strings as carry/projection columns

Timezone-aware timestamps and UTF-8 list columns are carry types. They can pass
through supported source/projection paths, but scalar operators, joins, sorts,
aggregates, and windows may reject them when they need timezone-free scalar
keys or list semantics. Temporal scalar functions also have narrower contracts
than relational carry/projection support; for example, some date operators
accept `Date32` or timezone-free timestamps but not `Date64`.

## Unsupported relational types

These types do not have native execution support today:

- Nested / complex: `Struct`, `Map`, `Union`, `Dictionary`,
  `FixedSizeList`, `ListView`, `LargeListView`, `RunEndEncoded`
- Numeric / decimal: `Float16`, `Decimal32`, `Decimal64`, `Decimal256`
- Binary: `Binary`, `LargeBinary`, `FixedSizeBinary`, `BinaryView`
- Temporal: `Interval`
- Lists whose child type is not UTF-8
- `Null` (the untyped null column type)

## cuGraph edge columns

:::warning[cuGraph edge columns are stricter]
cuGraph table functions have a narrower edge-table contract:

- Source and destination vertex columns are normally `Int32` or `Int64`, and
  both columns must use the same dtype.
- `cugraph_bfs` alone also accepts string (`Utf8`, `LargeUtf8`, `Utf8View`)
  source and destination columns through logical vertex-id mapping; every
  other algorithm requires integer vertex columns.
- `cugraph_minimum_spanning_tree` and
  `cugraph_spectral_modularity_maximization` use cuGraph C API edge-type
  dispatch and accept `Int32` vertex columns only.
- Edge weight columns, when supplied, must be `Float32` or `Float64`.
- `edge_id_col`, when supplied, must use the same vertex-id domain as the edge
  endpoints. BFS logical string vertex mapping does not support `edge_id_col`
  or the edge-id predicate options (`include_edge_ids_*`,
  `include_edge_id_col`).
:::

The unified SQL metadata functions expose the same contract at runtime. Use
`gpu_describe_function('<function>')` and
`gpu_validate_call('<function>', '<call_json>')` before running a new graph
call. The validation envelope names the edge table or view under
`relations.edges.table` and supplies graph bindings under `options`.

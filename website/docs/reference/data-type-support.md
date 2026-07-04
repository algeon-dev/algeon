---
sidebar_position: 2
description: Arrow, cuDF, and cuGraph data type support boundaries for native DataFusion Nexus execution.
---

# Data Type Support

:::note[Fallback depends on source readability]
DataFusion Nexus checks Arrow data types before a plan fragment enters native
execution. Unsupported CPU-readable fragments stay on the standard DataFusion
CPU path under the default GPU-preferred policy; GPU-only source leaves and
GPU-only policies fail planning with a structured reason instead.
:::

## Native relational columns

The native engine has a cuDF representation for these Arrow data types:

- `Boolean`
- Signed integers: `Int8`, `Int16`, `Int32`, `Int64`
- Unsigned integers: `UInt8`, `UInt16`, `UInt32`, `UInt64`
- Floating point: `Float32`, `Float64`
- `Decimal128` with valid Arrow decimal precision and scale
- Strings: `Utf8`, `LargeUtf8`, `Utf8View`
- `Date32`
- `Timestamp` at second, millisecond, microsecond, or nanosecond resolution
- `Duration` at second, millisecond, microsecond, or nanosecond resolution
- `List` / `LargeList` of UTF-8 strings as carry/projection columns

Timezone-aware timestamps and UTF-8 list columns are carry types. They can pass
through supported source/projection paths, but scalar operators, joins, sorts,
aggregates, and windows may reject them when they need timezone-free scalar
keys or list semantics.

## Unsupported relational types

These types do not have native execution support today:

- Nested / complex: `Struct`, `Map`, `Union`, `Dictionary`,
  `FixedSizeList`, `ListView`, `LargeListView`, `RunEndEncoded`
- Numeric / decimal: `Float16`, `Decimal32`, `Decimal64`, `Decimal256`
- Binary: `Binary`, `LargeBinary`, `FixedSizeBinary`, `BinaryView`
- Temporal: `Date64`, `Time32`, `Time64`, `Interval`
- Lists whose child type is not UTF-8

## cuGraph edge columns

:::warning[cuGraph edge columns are stricter]
cuGraph table functions have a narrower edge-table contract:

- Source and destination vertex columns are normally `Int32` or `Int64`, and
  both columns must use the same dtype.
- `cugraph_minimum_spanning_tree`,
  `cugraph_spectral_modularity_maximization`, and
  `cugraph_balanced_cut_clustering` use legacy cuGraph edge-type dispatch and
  accept `Int32` vertex columns only.
- Edge weight columns, when supplied, must be `Float32` or `Float64`.
- `edge_id_col`, when supplied, must use the same vertex-id domain as the edge
  endpoints. BFS logical string vertex mapping does not support `edge_id_col`.
:::

The SQL metadata functions expose the same contract at runtime. Use
`cugraph_describe_algorithm('<function>', 'verbose')` and
`cugraph_validate_call('<function>', '<table>', '<call_json>')` before running a
new graph call.

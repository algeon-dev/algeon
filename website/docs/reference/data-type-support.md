---
sidebar_position: 3
description: Arrow, cuDF, cuGraph, and cuVS data type support boundaries for native Algeon execution.
---

# Data Type Support

Algeon checks Arrow data types before a whole relational candidate
enters native execution. A candidate with an unsupported type keeps the
executable DataFusion baseline. Unified Iceberg scans retain their executable
CPU delegate when native type admission does not select acceleration. A
`NoDataFusionCpu` final-plan requirement instead rejects a completed plan that
retains DataFusion CPU execution.

This page covers column representation: which Arrow types the GPU execution
path (Algeon's "native" path) can hold, and what it can do with each. Operator
and function support is documented separately in
[Native SQL Support](./native-sql-support.mdx), and the exact physical plan
should be checked with
[`algeon_explain_coverage`](../integration/gpu-coverage-validation.mdx).

Two levels of support appear below:

- **Full support**: the column can be used anywhere the operator allows: in
  expressions, predicates, join keys, grouping keys, sort keys, and aggregate
  or window inputs.
- **Carry support** (also written "carry/projection"): the column can be read
  from a source, selected, aliased, kept through row selection, `UNION ALL`,
  and the non-key side of a join or aggregation, and returned unchanged. It
  does not mean the column can be used in an expression, a join key, a
  grouping key, a sort key, or an aggregate input. A candidate that uses a
  carry-only column in one of those positions keeps the DataFusion baseline.

The carry matrices further down list exactly which types are carry-only and
what the output schema preserves for each.

## Native relational columns

The native engine has a cuDF representation for these Arrow data types:

- `Boolean`
- Signed integers: `Int8`, `Int16`, `Int32`, `Int64`
- Unsigned integers: `UInt8`, `UInt16`, `UInt32`, `UInt64`
- Floating point: `Float32`, `Float64`
- `Decimal128` with valid Arrow decimal precision and scale
- `Decimal32` and `Decimal64` with valid Arrow decimal precision and scale
  (carry support only)
- Strings: `Utf8`, `LargeUtf8`, `Utf8View`
- `Date32`, `Date64`
- `Time32` and `Time64` at their Arrow-supported resolutions
- `Timestamp` at second, millisecond, microsecond, or nanosecond resolution
- `Duration` at second, millisecond, microsecond, or nanosecond resolution

### Narrow-decimal carry matrix

`Decimal32(P, S)` and `Decimal64(P, S)` are carry/projection columns when the
precision and scale pair passes Arrow's validator:

| Arrow type | Raw fixed-point width | Schema-aware output |
| --- | --- | --- |
| `Decimal32(P, S)` | 4 bytes | `Decimal32(P, S)` |
| `Decimal64(P, S)` | 8 bytes | `Decimal64(P, S)` |

Carry paths preserve raw values, nulls, logical width, precision, and scale
through scans, projections, aliases, row selection, `UNION ALL`, and non-key
payloads (columns that ride along beside a join or grouping key without being
one). "Schema-aware output" means the Arrow schema Algeon was given for the
query; when that schema is present the narrow type is returned as declared.
Without an external Arrow schema, untyped cuDF Decimal32/64 output retains the
canonical `Decimal128(38, S)` mapping. Narrow decimals are rejected
by local capability analysis in expressions, predicates, literals, comparisons,
casts, join or grouping keys, sort keys, partitions, aggregate inputs, and
window inputs.

### Basic-list carry matrix

`List<T>` and `LargeList<T>` are supported as carry/projection columns for this
exact set of child types:

| Container | Supported child types |
| --- | --- |
| `List<T>` | `Boolean`, `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Float32`, `Float64`, `Utf8`, `LargeUtf8`, `Utf8View` |
| `LargeList<T>` | `Boolean`, `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Float32`, `Float64`, `Utf8`, `LargeUtf8`, `Utf8View` |

This is representation and carry support, not list-expression support. These
columns can pass through supported source and projection paths, including as
non-key payloads, but cannot be scalar or filter values, join keys, grouping or
sort values, partitions, aggregate inputs, or window inputs. Child types are
not implicitly cast into this matrix.

The Arrow output schema preserves `List` versus `LargeList`, including the
declared child field and nullability. Native list offsets use a bounded Int32
representation: active offsets and child lengths must fit the nonnegative
Int32 range. An out-of-range `LargeList` is rejected rather than truncated.
NaN, +infinity, -infinity, signed zero, root and child nulls, and empty valid
lists are payload values preserved by carry paths, not scalar finite-value
rules.

### Numeric fixed-size-list carry matrix

`FixedSizeList<T, D>` is supported as a carry/projection column for this exact
numeric child matrix:

| Container | Supported child types |
| --- | --- |
| `FixedSizeList<T, D>` | `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `Float32`, `Float64` |

The dimension must be positive. Native import uses cuDF `LIST` offsets, so
`rows * D` and every synthesized or observed child offset must fit the
nonnegative Int32 range. Schema-aware Arrow output remains `FixedSizeList`,
preserving `D`, the declared child field, and root and child nullability; every
non-null row must have exactly `D` child positions. These columns are
carry/projection payloads, not scalar or key operands.

Timezone-aware timestamps are also carry types. Temporal scalar functions have
narrower contracts than relational carry/projection support; for example, some
date operators accept `Date32` or timezone-free timestamps but not `Date64`.

## Unsupported relational types

These types do not have native execution support today:

- Nested / complex: `Struct`, `Map`, `Union`, `Dictionary`, `ListView`,
  `LargeListView`, `RunEndEncoded`
- Numeric / decimal: `Float16`, `Decimal256`
- Binary: `Binary`, `LargeBinary`, `FixedSizeBinary`, `BinaryView`
- Temporal: `Interval`
- `FixedSizeList` with `D <= 0` or a child outside the numeric fixed-size-list
  matrix, including Boolean, string, decimal, temporal, binary, nested, and
  other complex children
- `List` / `LargeList` with any child type outside the basic-list carry matrix,
  including decimal, temporal, binary, nested-list, and other complex children
- `Null` (the untyped null column type)

## cuGraph edge columns

:::warning[cuGraph edge columns are stricter]
cuGraph table functions have a narrower edge-table contract:

- Source and destination vertex columns must use one of the numeric types
  declared for the function in the generated [cuGraph vertex ID support
  matrix](../cugraph/graph-options.mdx#vertex-id-support-by-function), or must
  both belong to the logical string family (`Utf8`, `LargeUtf8`, `Utf8View`)
  when that function declares logical string support.
- Logical string endpoints share one codebook even when their physical Arrow
  string types differ. Vertex-identity output columns are canonicalized to
  `Utf8`; numeric scores, distances, counts, coordinates, and opaque labels
  keep their function-defined numeric types.
- Some legacy cuGraph dispatch paths remain `Int32`-only. The generated
  matrix is the source for that per-function distinction; this overview does
  not maintain a second algorithm inventory.
- Edge weight columns, when supplied, must be `Float32` or `Float64`.
- `edge_id_col`, when supplied for a numeric graph, must use the same numeric
  vertex-id domain as the edge endpoints. Logical string vertex mapping does
  not support `edge_id_col` or edge-ID predicate options such as
  `include_edge_ids_*` and `include_edge_id_col`.
:::

The unified SQL metadata functions expose the same contract at runtime. Use
`gpu_describe_function('<function>')` and
`gpu_validate_call('<function>', '<call_json>')` before running a new graph
call. The validation envelope names the edge table or view under
`relations.edges.table` and supplies graph bindings under `options`.

## cuVS vector elements

:::warning[cuVS vector types depend on the function and metric]
cuVS functions accept dense vectors in wide feature columns or one list column. Every wide feature column and every vector binding in a call must have the same element type. The generated [cuVS element and metric matrix](../cuvs/vector-inputs.mdx#vector-element-and-metric-support-by-function) lists the verified combinations. In particular, integer IVF-PQ does not support cosine. Pairwise `distance` matches its `Float32` or `Float64` input; ANN distance stays `Float32`, and KMeans `cluster_id` stays `Int32`.

Floating elements must be finite. Integer elements have no finite-value scan. No cuVS call implicitly casts vector elements; use `CAST` in the relation subquery. `Float16` is unsupported because cuDF has no Float16 representation.
:::

Use `gpu_describe_function('<function>')` for the catalog contract and `gpu_validate_call('<function>', '<call_json>')` for static input validation.

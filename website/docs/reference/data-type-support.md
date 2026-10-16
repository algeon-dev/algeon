---
sidebar_position: 4
description: Which Arrow types the GPU relational path holds, which it only carries, and what the UDTF families accept.
---

# Data Type Support

Algeon checks Arrow types before a relational candidate enters native
execution. A candidate with an unsupported type keeps the DataFusion baseline,
and `native_required` mode rejects it instead. Operator and
function support is on [Native SQL Support](./native-sql-support.mdx). The
plan for one query is checked with
[`algeon_explain_coverage`](../integration/gpu-coverage-validation.mdx).

The table uses two levels of support.

| Support | Where the column may appear |
| --- | --- |
| Full | Anywhere the operator allows: expressions, predicates, join keys, grouping keys, sort keys, aggregate and window inputs. |
| Carry | Read from a source, selected, aliased, kept through row selection, `UNION ALL` and the non-key side of a join or aggregation, and returned unchanged. |

A candidate that uses a Carry column as an expression value, a key, a sort or
partition value, a literal, or an aggregate or window input keeps the
DataFusion baseline. A global `COUNT(column)` without `DISTINCT`, `FILTER` or
`GROUP BY` over a `List`, `LargeList` or `FixedSizeList` column is native: it
reads only the top-level validity of the column.

## Relational columns

| Arrow type | Support | Notes |
| --- | --- | --- |
| `Boolean` | Full | |
| `Int8`, `Int16`, `Int32`, `Int64`, `UInt8`, `UInt16`, `UInt32`, `UInt64` | Full | |
| `Float32`, `Float64` | Full | |
| `Decimal128(P, S)` | Full | Valid Arrow precision and scale. |
| `Utf8`, `LargeUtf8`, `Utf8View` | Full | |
| `Date32`, `Date64` | Full | Some temporal functions accept `Date32` but not `Date64`. See the function table. |
| `Time32`, `Time64` | Full | At their Arrow resolutions. |
| `Timestamp` without a time zone | Full | Second to nanosecond resolution. |
| `Duration` | Full | Second to nanosecond resolution. |
| `Timestamp` with a time zone | Carry | |
| `Decimal32(P, S)`, `Decimal64(P, S)` | Carry | Returned as declared when the query has an Arrow schema. Untyped cuDF output maps to `Decimal128(38, S)`. |
| `List<T>`, `LargeList<T>` | Carry | `T` is `Boolean`, an integer type, `Float32`, `Float64` or a string type. Output keeps `List` versus `LargeList`, the child field and nullability. |
| `FixedSizeList<T, D>` | Carry | `T` is an integer type, `Float32` or `Float64`. `D` is positive. Output keeps `D`, the child field and nullability. |
| `Float16`, `Decimal256` | None | |
| `Binary`, `LargeBinary`, `FixedSizeBinary`, `BinaryView` | None | |
| `Interval` | None | |
| `Struct`, `Map`, `Union`, `Dictionary`, `ListView`, `LargeListView`, `RunEndEncoded` | None | |
| `List`, `LargeList`, `FixedSizeList` with any other child | None | No implicit cast of the child type. |
| `Null` | None | The untyped null column type. |

Native lists use Int32 offsets. Every offset, child length and `rows * D`
must fit the nonnegative Int32 range. A `LargeList` outside that range is
rejected. Each non-null `FixedSizeList` row holds exactly `D` child
positions. NaN, infinities, signed zero, root and child nulls, and empty valid
lists are payload values that carry paths preserve.

## UDTF inputs

Each UDTF family checks its own input types, separately from the relational
path. `gpu_describe_function('<udtf>')` returns the contract and
`gpu_validate_call('<udtf>', '<call_json>')` validates a call before it runs.

### cuGraph edge columns

- Source and destination columns use a numeric type listed for the UDTF in
  the [vertex ID matrix](../cugraph/graph-options.mdx#vertex-id-support-by-udtf),
  or both are strings (`Utf8`, `LargeUtf8`, `Utf8View`) when the UDTF lists
  logical string support. String endpoints share one codebook even when their
  physical types differ, and vertex identity outputs are returned as `Utf8`.
- Edge weights, when supplied, are `Float32` or `Float64`.
- `edge_id_col` uses the same numeric domain as the endpoints. String vertex
  mapping does not support `edge_id_col` or the `include_edge_id*` options.

### cuVS vector elements

- Vectors arrive as wide feature columns or one list column, and every
  feature column and vector binding in a call has the same element type. The
  [element and metric matrix](../cuvs/vector-inputs.mdx#vector-element-and-metric-support-by-udtf)
  lists the supported combinations. Integer IVF-PQ does not support cosine.
- Floating elements must be finite. Elements are not cast implicitly; use
  `CAST` in the subquery. `Float16` has no cuDF representation.
- Pairwise `distance` matches its `Float32` or `Float64` input, ANN distance is
  `Float32` and KMeans `cluster_id` is `Int32`.

### cuML feature relations

- Features are `Float32` or `Float64`, as wide columns or one list column
  under any name: a `FixedSizeList`, or a uniform `List` or `LargeList`. Every
  UDTF except `cuml_forest_predict` narrows `Float64` to `Float32` on the
  device, and the values cuML reads must be finite. `cuml_forest_predict`
  takes `Float32` only.
- Every relation has an `id` of type `Int32`, `Int64`, `Utf8`, `LargeUtf8` or
  `Utf8View`. Repeated IDs are allowed. `row_ordinal UInt64` identifies each
  output row, beside an `id` of the input type.
- Classifiers take `Int32` labels and the regressor finite `Float32` targets
  in `training`. `predict` rejects a `label` column, and unsupervised `input`
  treats every non-`id` column as a feature, including one named `label`.
- Nullability is not checked in the schema. A null value fails the query with
  an invalid-input error that names the column.

[cuML SQL inputs](../cuml/ml-inputs.mdx) describes relation bindings and the
[cuML UDTF index](../cuml/udtfs/index.mdx) each UDTF's output schema.

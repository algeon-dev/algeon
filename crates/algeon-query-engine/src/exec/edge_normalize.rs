use std::time::Instant;

use cudf::{DataType, Expr, Series, TypeId, ops::binary::BinaryOperator};
use query_planning::plan::{EdgeDedupStrategy, EdgeNormalizeNode, EdgeNormalizeSpec, EdgeWeightCombiner};

use super::{
    CudfSortOptions, DataFrame, DropDuplicatesOptions, DuplicateKeepOption, Exec, ExecutionContext, ExecutionMetrics,
    GpuDataFrame, NodeId, Result, col, cudf_query_error, lit,
};

/// Applies `node.spec` to the edge rows that pass the null filter.
///
/// The output holds the edge set and, for each `(source, destination)` pair,
/// the weight and edge id of the row that a stable sort by source,
/// destination and, for [`EdgeWeightCombiner::MinWeight`], weight (nulls
/// last) places first or last. The output row order is unspecified.
///
/// Deduplication is skipped where it cannot remove a row: when the spec
/// declares distinct pairs and either no reverse edges are added or the source
/// and destination value ranges are disjoint. Otherwise the rows are sorted.
/// When both endpoint columns are Int32 or Int64 and their values span at most
/// `2^32` together, each edge is packed into one `UInt64` key, and a key-only
/// sort takes libcudf's radix path. Other inputs, and `MinWeight` inputs,
/// sort through libcudf's row comparator.
///
/// Peak device memory stays at or below the row-comparator path's. For `n`
/// rows after reverse expansion, that path holds the 16-byte endpoint
/// columns, the 4-byte kept-row map and their 16-byte copies during its final
/// gather (36 bytes per row before payload columns). The packed path drops
/// the endpoint columns before sorting. With payload columns, its radix sort
/// holds the 8-byte key and libcudf's 28-byte temporaries for an 8-byte key
/// (index map, key copy, identity map and CUB's alternate key and index
/// buffers). Without payload columns, the key is sorted by value: the key, its
/// sorted copy and CUB's alternate key buffer take 24 bytes per row, and
/// unpacking peaks at 32 (the kept keys, both endpoint columns and one 8-byte
/// intermediate).
pub(super) fn execute_edge_normalize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &EdgeNormalizeNode,
    mut input: GpuDataFrame,
    metrics: &ExecutionMetrics,
) -> Result<GpuDataFrame> {
    let retained_normalization_source = input.take_resident_owner();
    let graph_residency = input.graph_residency().cloned();
    // Edge normalization only restructures topology (reverse/dedup); a null
    // weight or edge_id must not drop an edge here. Weight-null conflicts are
    // resolved by the weight combiner during deduplication; the null filter keeps those rows.
    let null_filter_scope = super::edge_null_filter::EdgeNullFilterScope { weight: false, edge_id: false };
    let frame = super::edge_null_filter::filter_null_edge_rows(
        context,
        node_id,
        "graph_edge_null_filter",
        input.frame().clone(),
        &node.spec.edge_columns,
        null_filter_scope,
        metrics,
    )?;
    let timer = Instant::now();
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let spec = &node.spec;
    input = GpuDataFrame::try_new_with_source_residency(project_edge_columns(frame, spec, node_id)?, &input)?;

    // Each stage replaces `input`, so the previous stage's columns are freed
    // when no other owner holds them.
    if spec.dedup == EdgeDedupStrategy::None || spec.distinct_edge_pairs && !spec.add_reverse {
        if spec.add_reverse {
            let expanded = with_reverse_edges(input.frame(), spec, node_id, overrides)?;
            advance(&mut input, expanded)?;
        }
    } else {
        let ranges = endpoint_ranges(input.frame(), spec, node_id, overrides)?;
        if spec.distinct_edge_pairs && ranges.as_ref().is_some_and(EndpointRanges::are_disjoint) {
            let expanded = with_reverse_edges(input.frame(), spec, node_id, overrides)?;
            advance(&mut input, expanded)?;
        } else if let Some(packing) = ranges.and_then(EndpointRanges::packing) {
            let source = spec.edge_columns.source.as_str();
            let keyed = packed_edge_rows(input.frame(), spec, &packing, node_id, overrides)?;
            advance(&mut input, keyed)?;
            let deduplicated = drop_duplicate_edges(input.frame(), spec, &[source], node_id, overrides)?;
            advance(&mut input, deduplicated)?;
            let unpacked = unpacked_edge_rows(input.frame(), spec, &packing, node_id, overrides)?;
            advance(&mut input, unpacked)?;
        } else {
            if spec.add_reverse {
                let expanded = with_reverse_edges(input.frame(), spec, node_id, overrides)?;
                advance(&mut input, expanded)?;
            }
            let endpoints = [spec.edge_columns.source.as_str(), spec.edge_columns.destination.as_str()];
            let deduplicated = drop_duplicate_edges(input.frame(), spec, &endpoints, node_id, overrides)?;
            advance(&mut input, deduplicated)?;
        }
    }

    metrics.add_operator_time(timer.elapsed());
    let output = input.with_retained_normalization_source(retained_normalization_source);
    Ok(match graph_residency {
        Some(graph_residency) => output.with_graph_residency(graph_residency),
        None => output,
    })
}

fn advance(input: &mut GpuDataFrame, next: DataFrame) -> Result<()> {
    *input = GpuDataFrame::try_new_with_source_residency(next, input)?;
    Ok(())
}

/// Inclusive value ranges of the source and destination columns.
struct EndpointRanges {
    vertex_type: DataType,
    source: (i64, i64),
    destination: (i64, i64),
}

impl EndpointRanges {
    /// With disjoint ranges no reversed edge equals an input edge and no edge
    /// is a self loop.
    fn are_disjoint(&self) -> bool {
        self.source.1 < self.destination.0 || self.destination.1 < self.source.0
    }

    fn packing(self) -> Option<EdgeKeyPacking> {
        let low = self.source.0.min(self.destination.0);
        let high = self.source.1.max(self.destination.1);
        let bits = u64::BITS - high.abs_diff(low).leading_zeros();
        (bits <= EdgeKeyPacking::MAX_ENDPOINT_BITS).then_some(EdgeKeyPacking {
            vertex_type: self.vertex_type,
            low,
            bits,
        })
    }
}

/// Reads the endpoint ranges of Int32 or Int64 vertex columns. Other vertex
/// types and an empty input have none.
fn endpoint_ranges(
    frame: &DataFrame,
    spec: &EdgeNormalizeSpec,
    node_id: NodeId,
    overrides: Exec<'_>,
) -> Result<Option<EndpointRanges>> {
    let range_error = |error| {
        cudf_query_error(
            "query_engine.edge_normalize.endpoint_range",
            "failed to read endpoint value ranges for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
    };
    let source = frame.series(&spec.edge_columns.source).map_err(range_error)?;
    let destination = frame.series(&spec.edge_columns.destination).map_err(range_error)?;
    let vertex_type = source.dtype().map_err(range_error)?;
    if destination.dtype().map_err(range_error)? != vertex_type {
        return Ok(None);
    }
    let range = |series: &Series| -> cudf::Result<Option<(i64, i64)>> {
        let ops = series.on(overrides);
        let (min, max) = match vertex_type.id() {
            TypeId::Int32 => {
                let (min, max) = ops.minmax_values::<i32>()?;
                (min.map(i64::from), max.map(i64::from))
            },
            TypeId::Int64 => ops.minmax_values::<i64>()?,
            _ => return Ok(None),
        };
        Ok(min.zip(max))
    };
    let (Some(source), Some(destination)) =
        (range(&source).map_err(range_error)?, range(&destination).map_err(range_error)?)
    else {
        return Ok(None);
    };
    Ok(Some(EndpointRanges { vertex_type, source, destination }))
}

/// Packs an edge into one `UInt64` key: the source offset from `low` in the
/// high `bits` and the destination offset in the low `bits`. Both offsets are
/// below `2^bits`, so key order is `(source, destination)` order and equal
/// keys are equal pairs.
struct EdgeKeyPacking {
    vertex_type: DataType,
    low: i64,
    bits: u32,
}

impl EdgeKeyPacking {
    const MAX_ENDPOINT_BITS: u32 = 32;

    fn key(&self, high: &str, low: &str) -> Expr {
        self.offset(high)
            .binary_op_with_type(lit(u64::from(self.bits)), BinaryOperator::ShiftLeft, TypeId::UInt64)
            .binary_op_with_type(self.offset(low), BinaryOperator::BitwiseOr, TypeId::UInt64)
    }

    /// libcudf evaluates the subtraction in the common type of its operands
    /// and the `UInt64` output, modulo `2^64`. The offset is below `2^32`, so
    /// the result is exact for Int32 and Int64 vertex ids.
    fn offset(&self, column: &str) -> Expr {
        col(column).binary_op_with_type(lit(self.low), BinaryOperator::Sub, TypeId::UInt64)
    }

    fn source(&self, key: &str) -> Expr {
        self.vertex(col(key).binary_op_with_type(
            lit(u64::from(self.bits)),
            BinaryOperator::ShiftRightUnsigned,
            TypeId::Int64,
        ))
    }

    fn destination(&self, key: &str) -> Expr {
        let mask = (1_u64 << self.bits) - 1;
        self.vertex(col(key).binary_op_with_type(lit(mask), BinaryOperator::BitwiseAnd, TypeId::Int64))
    }

    fn vertex(&self, offset: Expr) -> Expr {
        offset.binary_op_with_type(lit(self.low), BinaryOperator::Add, self.vertex_type)
    }
}

/// The edge rows, reversed rows appended when the spec adds them, with the
/// packed key in place of both endpoint columns. The key keeps the source
/// column's name; payload columns follow it unchanged. Row `i` corresponds to
/// row `i` of [`with_reverse_edges`].
fn packed_edge_rows(
    frame: &DataFrame,
    spec: &EdgeNormalizeSpec,
    packing: &EdgeKeyPacking,
    node_id: NodeId,
    overrides: Exec<'_>,
) -> Result<DataFrame> {
    let source = spec.edge_columns.source.as_str();
    let destination = spec.edge_columns.destination.as_str();
    let pack_error = |error| {
        cudf_query_error(
            "query_engine.edge_normalize.pack",
            "failed to pack edge keys for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
    };
    // Every column is produced through `overrides`: a frame holds columns of
    // one provenance, and the input frame can have another.
    let forward = packing.key(source, destination).eval_in(frame, overrides).map_err(pack_error)?;
    let mut columns = Vec::new();
    if spec.add_reverse {
        let reverse = packing.key(destination, source).eval_in(frame, overrides).map_err(pack_error)?;
        columns.push((source.to_owned(), Series::concat(&[&forward, &reverse], overrides).map_err(pack_error)?));
        for payload in payload_columns(spec) {
            let values = frame.series(payload).map_err(pack_error)?;
            columns.push((payload.to_owned(), Series::concat(&[&values, &values], overrides).map_err(pack_error)?));
        }
    } else {
        columns.push((source.to_owned(), forward));
        for payload in payload_columns(spec) {
            columns.push((payload.to_owned(), col(payload).eval_in(frame, overrides).map_err(pack_error)?));
        }
    }
    DataFrame::from_series(columns).map_err(pack_error)
}

/// The inverse of [`packed_edge_rows`]: endpoint columns of the original
/// vertex type, then the payload columns.
fn unpacked_edge_rows(
    frame: &DataFrame,
    spec: &EdgeNormalizeSpec,
    packing: &EdgeKeyPacking,
    node_id: NodeId,
    overrides: Exec<'_>,
) -> Result<DataFrame> {
    let key = spec.edge_columns.source.as_str();
    let unpack_error = |error| {
        cudf_query_error(
            "query_engine.edge_normalize.unpack",
            "failed to unpack edge keys for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
    };
    let source = packing.source(key).eval_in(frame, overrides).map_err(unpack_error)?;
    let destination = packing.destination(key).eval_in(frame, overrides).map_err(unpack_error)?;
    let mut columns =
        vec![(spec.edge_columns.source.clone(), source), (spec.edge_columns.destination.clone(), destination)];
    for payload in payload_columns(spec) {
        columns.push((payload.to_owned(), frame.series(payload).map_err(unpack_error)?));
    }
    DataFrame::from_series(columns).map_err(unpack_error)
}

/// Keeps one row per distinct `keys` value: the first or last in stable
/// `keys` order, with the weight as the last sort key for
/// [`EdgeWeightCombiner::MinWeight`].
///
/// A frame that holds only one key column has identical duplicate rows, so
/// it is sorted by value and collapsed with no row-index map or gather.
/// Other frames sort an index map and gather the kept rows.
fn drop_duplicate_edges(
    frame: &DataFrame,
    spec: &EdgeNormalizeSpec,
    keys: &[&str],
    node_id: NodeId,
    overrides: Exec<'_>,
) -> Result<DataFrame> {
    let sort_error = |error| {
        cudf_query_error(
            "query_engine.edge_normalize.sort",
            "failed to sort edges for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
    };
    let dedup_error = |error| {
        cudf_query_error(
            "query_engine.edge_normalize.dedup",
            "failed to collapse duplicate edges for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
    };
    let options = DropDuplicatesOptions::builder()
        .subset(keys)
        .keep(match spec.dedup {
            EdgeDedupStrategy::None => DuplicateKeepOption::KeepAny,
            EdgeDedupStrategy::KeepFirst => DuplicateKeepOption::KeepFirst,
            EdgeDedupStrategy::KeepLast => DuplicateKeepOption::KeepLast,
        })
        .build();
    let stable = CudfSortOptions::new().with_stable(true);
    if keys.len() == 1 && frame.column_names() == keys {
        let sorted = frame.on(overrides).sort_values(keys, stable).map_err(sort_error)?;
        return sorted.on(overrides).drop_consecutive_duplicates(options).map_err(dedup_error);
    }
    let mut sort_by = keys.to_vec();
    if spec.weight_combiner == EdgeWeightCombiner::MinWeight
        && let Some(weight) = spec.edge_columns.weight.as_deref()
    {
        sort_by.push(weight);
    }
    let prepared = frame.on(overrides).prepare_sort_values(sort_by, stable).map_err(sort_error)?;
    prepared.materialize_drop_consecutive_duplicates(options, overrides).map_err(dedup_error)
}

fn payload_columns(spec: &EdgeNormalizeSpec) -> impl Iterator<Item = &str> {
    spec.edge_columns.weight.as_deref().into_iter().chain(spec.edge_columns.edge_id.as_deref())
}

fn project_edge_columns(frame: DataFrame, spec: &EdgeNormalizeSpec, node_id: NodeId) -> Result<DataFrame> {
    let columns = [spec.edge_columns.source.as_str(), spec.edge_columns.destination.as_str()]
        .into_iter()
        .chain(payload_columns(spec))
        .collect::<Vec<_>>();
    frame.select(columns).map_err(|error| {
        cudf_query_error(
            "query_engine.edge_normalize.project",
            "failed to project edge-list columns for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("src_col", &spec.edge_columns.source)
        .with_fact("dst_col", &spec.edge_columns.destination)
        .with_fact("weight_col", spec.edge_columns.weight.as_deref().unwrap_or("<none>"))
        .with_fact("edge_id_col", spec.edge_columns.edge_id.as_deref().unwrap_or("<none>"))
    })
}

fn with_reverse_edges(
    frame: &DataFrame,
    spec: &EdgeNormalizeSpec,
    node_id: NodeId,
    overrides: Exec<'_>,
) -> Result<DataFrame> {
    let reverse = reverse_edges(frame, spec, node_id)?;
    frame.on(overrides).concat_many(&[&reverse]).map_err(|error| {
        cudf_query_error(
            "query_engine.edge_normalize.reverse_concat",
            "failed to append reverse edges for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
    })
}

fn reverse_edges(frame: &DataFrame, spec: &EdgeNormalizeSpec, node_id: NodeId) -> Result<DataFrame> {
    let source = frame.series(&spec.edge_columns.source).map_err(|error| {
        cudf_query_error(
            "query_engine.edge_normalize.reverse_source",
            "failed to access source column for native reverse-edge expansion",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("column", &spec.edge_columns.source)
    })?;
    let destination = frame.series(&spec.edge_columns.destination).map_err(|error| {
        cudf_query_error(
            "query_engine.edge_normalize.reverse_destination",
            "failed to access destination column for native reverse-edge expansion",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("column", &spec.edge_columns.destination)
    })?;
    let mut pairs =
        vec![(spec.edge_columns.source.clone(), destination), (spec.edge_columns.destination.clone(), source)];
    if let Some(weight) = &spec.edge_columns.weight {
        let series = frame.series(weight).map_err(|error| {
            cudf_query_error(
                "query_engine.edge_normalize.reverse_weight",
                "failed to access weight column for native reverse-edge expansion",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("column", weight)
        })?;
        pairs.push((weight.clone(), series));
    }
    if let Some(edge_id) = &spec.edge_columns.edge_id {
        let series = frame.series(edge_id).map_err(|error| {
            cudf_query_error(
                "query_engine.edge_normalize.reverse_edge_id",
                "failed to access edge-id column for native reverse-edge expansion",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("column", edge_id)
        })?;
        pairs.push((edge_id.clone(), series));
    }
    DataFrame::from_series(pairs).map_err(|error| {
        cudf_query_error(
            "query_engine.edge_normalize.reverse_frame",
            "failed to assemble reverse edges for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
    })
}

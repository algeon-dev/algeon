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
/// and destination value ranges are disjoint. With disjoint ranges and no such
/// declaration, the input rows are deduplicated and the reversed copies of the
/// kept rows are appended. No reversed edge can then equal an input edge. The
/// reversed rows of a pair `(v, u)` are the copies of the input rows of
/// `(u, v)` in the same order, and the kept reversed row is the copy of the
/// kept input row. Other undirected inputs are deduplicated after the reversed
/// rows are appended.
///
/// When both endpoint columns are Int32 or Int64, the rows to deduplicate are
/// packed into one `UInt64` key per edge if the key fits in 64 bits
/// ([`EdgeKeyPacking`]), and a key-only sort takes libcudf's radix path. Input
/// rows pack each column over its own value range. Rows that include reversed
/// rows pack both columns over the union of the two ranges, because a reversed
/// row holds a destination value in the source column. Other inputs sort the
/// endpoint columns through libcudf's row comparator. `MinWeight` inputs sort
/// the key or the endpoint columns together with the weight through it.
///
/// Peak device memory stays at or below that of the row-comparator sort of
/// the expanded rows. For `n` rows after reverse expansion, that sort holds the
/// 16-byte endpoint columns, the 4-byte kept-row map and their 16-byte copies
/// during its final gather (36 bytes per row before payload columns). The
/// packed path drops the endpoint columns before sorting. With payload
/// columns, its radix sort holds the 8-byte key and libcudf's 28-byte
/// temporaries for an 8-byte key (index map, key copy, identity map and CUB's
/// alternate key and index buffers). Without payload columns, the key is
/// sorted by value: the key, its sorted copy and CUB's alternate key buffer
/// take 24 bytes per row, and unpacking peaks at 32 (the kept keys, both
/// endpoint columns and one 8-byte intermediate). Packing the input rows alone
/// holds their endpoint columns and two 8-byte intermediates (32 bytes per
/// row). Packing them with their reversed rows peaks at 24 bytes per expanded
/// row while it concatenates the two keys. Deduplicating only the `n / 2`
/// input rows of an undirected graph halves each count above per row of `n`.
/// The append that follows holds the at most `n / 2` kept rows and the
/// concatenated output, 48 bytes per kept row or at most 24 per row of `n`.
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
    match EdgeNormalization::choose(spec, || endpoint_ranges(input.frame(), spec, node_id, overrides))? {
        EdgeNormalization::AppendReverse => append_reverse_edges(&mut input, spec, node_id, overrides)?,
        EdgeNormalization::DeduplicateInput(packing) => {
            deduplicate_edges(&mut input, spec, packing.as_ref(), false, node_id, overrides)?;
            append_reverse_edges(&mut input, spec, node_id, overrides)?;
        },
        EdgeNormalization::DeduplicateExpanded(packing) => {
            deduplicate_edges(&mut input, spec, packing.as_ref(), true, node_id, overrides)?;
        },
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

/// The steps that normalize the edge rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeNormalization {
    /// No row repeats a pair. The reversed rows are appended when the spec
    /// adds them.
    AppendReverse,
    /// The input rows are deduplicated, then the reversed copies of the kept
    /// rows are appended when the spec adds them.
    DeduplicateInput(Option<EdgeKeyPacking>),
    /// The reversed rows are appended, then all rows are deduplicated.
    DeduplicateExpanded(Option<EdgeKeyPacking>),
}

impl EdgeNormalization {
    /// Chooses the steps for `spec`. `read_ranges` runs only when the choice
    /// depends on the endpoint ranges.
    fn choose(spec: &EdgeNormalizeSpec, read_ranges: impl FnOnce() -> Result<Option<EndpointRanges>>) -> Result<Self> {
        if spec.dedup == EdgeDedupStrategy::None || spec.distinct_edge_pairs && !spec.add_reverse {
            return Ok(Self::AppendReverse);
        }
        Ok(match read_ranges()? {
            None if spec.add_reverse => Self::DeduplicateExpanded(None),
            None => Self::DeduplicateInput(None),
            Some(ranges) if !spec.add_reverse => Self::DeduplicateInput(ranges.packing()),
            Some(ranges) if !ranges.are_disjoint() => Self::DeduplicateExpanded(ranges.union_packing()),
            Some(_) if spec.distinct_edge_pairs => Self::AppendReverse,
            Some(ranges) => Self::DeduplicateInput(ranges.packing()),
        })
    }
}

fn append_reverse_edges(
    input: &mut GpuDataFrame,
    spec: &EdgeNormalizeSpec,
    node_id: NodeId,
    overrides: Exec<'_>,
) -> Result<()> {
    if !spec.add_reverse {
        return Ok(());
    }
    let expanded = with_reverse_edges(input.frame(), spec, node_id, overrides)?;
    advance(input, expanded)
}

/// Keeps one row per pair of `input`, after appending the reversed rows when
/// `reverse` is set. A `packing` sorts the packed keys instead of the
/// endpoint columns.
fn deduplicate_edges(
    input: &mut GpuDataFrame,
    spec: &EdgeNormalizeSpec,
    packing: Option<&EdgeKeyPacking>,
    reverse: bool,
    node_id: NodeId,
    overrides: Exec<'_>,
) -> Result<()> {
    let Some(packing) = packing else {
        if reverse {
            let expanded = with_reverse_edges(input.frame(), spec, node_id, overrides)?;
            advance(input, expanded)?;
        }
        let endpoints = [spec.edge_columns.source.as_str(), spec.edge_columns.destination.as_str()];
        let deduplicated = drop_duplicate_edges(input.frame(), spec, &endpoints, node_id, overrides)?;
        return advance(input, deduplicated);
    };
    let keyed = packed_edge_rows(input.frame(), spec, packing, reverse, node_id, overrides)?;
    advance(input, keyed)?;
    let deduplicated =
        drop_duplicate_edges(input.frame(), spec, &[spec.edge_columns.source.as_str()], node_id, overrides)?;
    advance(input, deduplicated)?;
    let unpacked = unpacked_edge_rows(input.frame(), spec, packing, node_id, overrides)?;
    advance(input, unpacked)
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

    /// The key of rows whose source and destination columns hold values of
    /// their own ranges.
    fn packing(&self) -> Option<EdgeKeyPacking> {
        EdgeKeyPacking::new(self.vertex_type, self.source, self.destination)
    }

    /// The key of rows that include reversed rows, whose source column also
    /// holds destination values.
    fn union_packing(&self) -> Option<EdgeKeyPacking> {
        let union = (self.source.0.min(self.destination.0), self.source.1.max(self.destination.1));
        EdgeKeyPacking::new(self.vertex_type, union, union)
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

/// Packs an edge into one `UInt64` key,
/// `(source - source_low) << destination_bits | (destination - destination_low)`.
/// Each low is the minimum of the values its field holds, and the two field
/// widths sum to at most 64 bits. Both offsets fit their fields, so key order
/// is `(source, destination)` order and equal keys are equal pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EdgeKeyPacking {
    vertex_type: DataType,
    source_low: i64,
    destination_low: i64,
    destination_bits: u32,
}

impl EdgeKeyPacking {
    /// A field of `w` bits holds a range whose maximum minus minimum is below
    /// `2^w`, and a single value takes 0 bits. The destination field stays
    /// below 64 bits: a 64-bit field would shift by 64, which C++ leaves
    /// undefined and which overflows the Rust mask. A 64-bit source field
    /// leaves a 0-bit destination field and a shift of 0.
    fn new(vertex_type: DataType, source: (i64, i64), destination: (i64, i64)) -> Option<Self> {
        let width = |(low, high): (i64, i64)| u64::BITS - high.abs_diff(low).leading_zeros();
        let (source_bits, destination_bits) = (width(source), width(destination));
        (source_bits + destination_bits <= u64::BITS && destination_bits < u64::BITS).then_some(Self {
            vertex_type,
            source_low: source.0,
            destination_low: destination.0,
            destination_bits,
        })
    }

    /// libcudf evaluates each step in the common type of its operands and its
    /// `UInt64` output (`get_common_type` in the compiled binary operations).
    /// That type is `UInt64` for Int32 or Int64 columns and `UInt64` literals,
    /// and each step is exact modulo `2^64`; the key itself is below `2^64`.
    /// Adding the destination before subtracting `destination_low` gives each
    /// step one computed operand, and a step holds two 8-byte columns.
    fn key(&self, source: &str, destination: &str) -> Expr {
        col(source)
            .binary_op_with_type(lit(self.source_low.cast_unsigned()), BinaryOperator::Sub, TypeId::UInt64)
            .binary_op_with_type(lit(u64::from(self.destination_bits)), BinaryOperator::ShiftLeft, TypeId::UInt64)
            .binary_op_with_type(col(destination), BinaryOperator::Add, TypeId::UInt64)
            .binary_op_with_type(lit(self.destination_low.cast_unsigned()), BinaryOperator::Sub, TypeId::UInt64)
    }

    fn source(&self, key: &str) -> Expr {
        let offset = col(key).binary_op_with_type(
            lit(u64::from(self.destination_bits)),
            BinaryOperator::ShiftRightUnsigned,
            TypeId::UInt64,
        );
        self.vertex(offset, self.source_low)
    }

    fn destination(&self, key: &str) -> Expr {
        let mask = (1_u64 << self.destination_bits) - 1;
        let offset = col(key).binary_op_with_type(lit(mask), BinaryOperator::BitwiseAnd, TypeId::UInt64);
        self.vertex(offset, self.destination_low)
    }

    /// The `UInt64` offset of a value from `low` plus `low`, modulo `2^64`, is
    /// the value modulo `2^64`. libcudf writes it to the Int32 or Int64 output
    /// with a C++20 conversion, which is modulo the output width and restores
    /// the value.
    fn vertex(&self, offset: Expr, low: i64) -> Expr {
        offset.binary_op_with_type(lit(low.cast_unsigned()), BinaryOperator::Add, self.vertex_type)
    }
}

/// The edge rows, reversed rows appended when `reverse` is set, with the
/// packed key in place of both endpoint columns. The key keeps the source
/// column's name; payload columns follow it unchanged. With `reverse`, row
/// `i` corresponds to row `i` of [`with_reverse_edges`].
fn packed_edge_rows(
    frame: &DataFrame,
    spec: &EdgeNormalizeSpec,
    packing: &EdgeKeyPacking,
    reverse: bool,
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
    if reverse {
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

#[cfg(test)]
mod tests;

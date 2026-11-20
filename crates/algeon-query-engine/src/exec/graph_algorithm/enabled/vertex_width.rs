//! Vertex id width of the constructed cuGraph graph.
//!
//! libcugraph keeps the vertex type of its input edge arrays through
//! renumbering, stores one vertex id per adjacency entry and copies the
//! endpoint arrays during construction. When every id of an Int64 call fits
//! Int32, the call runs on Int32 copies of its edge and side-input id columns
//! as the same call declared with Int32 ids does, and its result columns are
//! widened back to the declared types.

use std::collections::BTreeMap;

use arrow::datatypes::DataType as ArrowDataType;
use cudf::{DataType as CudfDataType, Exec, TypeId};
use query_planning::plan::{GraphAlgorithmSpec, graph_unreachable_vertex_sentinel};

use super::{
    DataFrame, ExecutionContext, GpuDataFrame, GraphAlgorithmExecutionInputs, GraphAlgorithmExecutionResult, Result,
    cudf_query_error, normalize_graph_result_schema,
};
use crate::exec::{col, lit, when};

const OPERATION: &str = "query_engine.graph_algorithm.vertex_width";

/// Inclusive range of the valid values of one id column or literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct IdRange {
    min: i64,
    max: i64,
}

impl IdRange {
    pub(super) const fn new(min: i64, max: i64) -> Self {
        Self { min, max }
    }

    const fn union(self, other: Self) -> Self {
        Self {
            min: if self.min < other.min { self.min } else { other.min },
            max: if self.max > other.max { self.max } else { other.max },
        }
    }
}

/// Whether a graph with `edge_rows` edge rows, endpoint ids in `endpoints`
/// and every other id it converts to the vertex type (edge ids, side-input ids
/// and literal vertex ids) in `other_ids` can be built with Int32 vertex ids.
///
/// - libcugraph's graph construction accepts Int32 vertex arrays only below
///   `i32::MAX` rows and then uses Int32 edge offsets and edge ids.
/// - Every id stays below `i32::MAX`. A graph built without renumbering has
///   `max + 1` vertices, and libcugraph marks unreached vertices with the
///   vertex type's maximum.
/// - The renumbered vertex count fits Int32. It is at most the number of
///   distinct endpoint values, bounded by the endpoint span and by two
///   endpoints per row.
///
/// A graph without a valid endpoint id keeps its declared type.
pub(super) fn graph_fits_int32_vertex_ids(edge_rows: usize, endpoints: Option<IdRange>, other_ids: &[IdRange]) -> bool {
    let Some(endpoints) = endpoints else {
        return false;
    };
    let id_limit = i64::from(i32::MAX);
    let ids_fit = std::iter::once(&endpoints)
        .chain(other_ids)
        .all(|range| range.min >= i64::from(i32::MIN) && range.max < id_limit);
    let endpoint_span = i128::from(endpoints.max) - i128::from(endpoints.min) + 1;
    let vertex_bound = endpoint_span.min(2 * edge_rows as i128);
    ids_fit && (edge_rows as i128) < i128::from(id_limit) && vertex_bound <= i128::from(id_limit)
}

/// Moves a graph call to Int32 vertex ids when its edge id columns are Int64
/// and [`graph_fits_int32_vertex_ids`] holds for its edges, its side-input id
/// columns and its literal vertex ids.
///
/// Returns the spec the executor runs with, whose vertex result columns are
/// Int32, together with the narrowed inputs. The edge frame's graph residency
/// descriptor names the Int32 vertex type, so a resident graph key describes
/// the graph it caches. A call whose ids do not all fit, or whose id columns
/// are absent or not Int64, keeps its inputs and returns no spec; the executor
/// then reports any input error as it does for the declared types.
pub(super) fn narrow_graph_vertex_ids(
    context: &ExecutionContext,
    spec: &GraphAlgorithmSpec,
    inputs: GraphAlgorithmExecutionInputs,
) -> Result<(Option<GraphAlgorithmSpec>, GraphAlgorithmExecutionInputs)> {
    let edge_id_columns = [Some(spec.edge_columns.source.as_str()), Some(spec.edge_columns.destination.as_str())]
        .into_iter()
        .chain([spec.edge_columns.edge_id.as_deref()])
        .flatten()
        .collect::<Vec<_>>();
    let mut side_input_columns = BTreeMap::<usize, Vec<&str>>::new();
    for (input, column) in spec.params.vertex_dtype_side_input_columns() {
        let columns = side_input_columns.entry(input).or_default();
        if !columns.contains(&column) {
            columns.push(column);
        }
    }
    let edges = inputs.edge_list.frame();
    if !all_int64(edges, &edge_id_columns)? {
        return Ok((None, inputs));
    }
    for (input, columns) in &side_input_columns {
        let Some(frame) = inputs.side_inputs.get(input) else {
            return Ok((None, inputs));
        };
        if !all_int64(frame.frame(), columns)? {
            return Ok((None, inputs));
        }
    }

    let runtime = context.runtime()?;
    let overrides = runtime.exec();
    let source = id_range(edges, &spec.edge_columns.source, overrides)?;
    let destination = id_range(edges, &spec.edge_columns.destination, overrides)?;
    let endpoints = match (source, destination) {
        (Some(source), Some(destination)) => Some(source.union(destination)),
        (range, None) | (None, range) => range,
    };
    let mut other_ids =
        spec.params.vertex_id_literals().into_iter().map(|vertex| IdRange::new(vertex, vertex)).collect::<Vec<_>>();
    if let Some(edge_id) = spec.edge_columns.edge_id.as_deref() {
        other_ids.extend(id_range(edges, edge_id, overrides)?);
    }
    for (input, columns) in &side_input_columns {
        let frame = inputs.side_inputs[input].frame();
        for column in columns {
            other_ids.extend(id_range(frame, column, overrides)?);
        }
    }
    if !graph_fits_int32_vertex_ids(edges.num_rows(), endpoints, &other_ids) {
        return Ok((None, inputs));
    }

    let GraphAlgorithmExecutionInputs { edge_list, mut side_inputs } = inputs;
    let descriptor = edge_list.graph_residency().cloned();
    let mut edge_list = narrow_id_columns(edge_list, &edge_id_columns, overrides)?;
    if let Some(mut descriptor) = descriptor {
        descriptor.vertex_data_type = ArrowDataType::Int32;
        edge_list = edge_list.with_graph_residency(descriptor);
    }
    for (input, columns) in &side_input_columns {
        let frame = side_inputs.remove(input).expect("side-input presence was checked above");
        side_inputs.insert(*input, narrow_id_columns(frame, columns, overrides)?);
    }
    Ok((
        Some(spec.with_output_vertex_type(&ArrowDataType::Int32)),
        GraphAlgorithmExecutionInputs { edge_list, side_inputs },
    ))
}

/// Returns a narrowed call's result in `declared`'s output schema: vertex
/// columns widen to the declared type, and an unreachable-vertex marker in a
/// column the planning definition names becomes the declared type's marker.
pub(super) fn restore_declared_vertex_type(
    result: GraphAlgorithmExecutionResult,
    declared: &GraphAlgorithmSpec,
) -> Result<GraphAlgorithmExecutionResult> {
    let narrowed_sentinel =
        graph_unreachable_vertex_sentinel(&ArrowDataType::Int32).expect("Int32 is a graph vertex type");
    let declared_sentinel =
        graph_unreachable_vertex_sentinel(&ArrowDataType::Int64).expect("Int64 is a graph vertex type");
    let mut frame = result.frame;
    for &column in declared.algorithm.definition().unreachable_sentinel_columns(&declared.params) {
        let restored = when(col(column).eq(lit(narrowed_sentinel)))
            .then(lit(declared_sentinel))
            .otherwise(col(column))
            .eval_in(&frame, Exec::RETAINED)
            .and_then(|series| frame.on(Exec::RETAINED).with_column(column, series))
            .map_err(|error| {
                cudf_query_error(OPERATION, "failed to restore the declared unreachable-vertex marker", error)
                    .with_fact("column", column)
            })?;
        frame = restored;
    }
    Ok(GraphAlgorithmExecutionResult {
        frame: normalize_graph_result_schema(frame, &declared.output_schema)?,
        ..result
    })
}

fn all_int64(frame: &DataFrame, columns: &[&str]) -> Result<bool> {
    let names = frame.column_names();
    for column in columns {
        if !names.contains(column) {
            return Ok(false);
        }
        let dtype = frame.series(column).and_then(|series| series.dtype()).map_err(|error| {
            cudf_query_error(OPERATION, "failed to inspect a graph id column type", error).with_fact("column", *column)
        })?;
        if dtype.id() != TypeId::Int64 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn id_range(frame: &DataFrame, column: &str, overrides: Exec<'_>) -> Result<Option<IdRange>> {
    let (min, max) =
        frame.series(column).and_then(|series| series.on(overrides).minmax_values::<i64>()).map_err(|error| {
            cudf_query_error(OPERATION, "failed to read the value range of a graph id column", error)
                .with_fact("column", column)
        })?;
    Ok(min.zip(max).map(|(min, max)| IdRange::new(min, max)))
}

/// Casts `columns` of `input` to Int32 one at a time. Each step replaces the
/// previous frame, so an Int64 column that no other owner holds is released
/// as soon as its Int32 column exists.
pub(in crate::exec) fn narrow_id_columns(
    mut input: GpuDataFrame,
    columns: &[&str],
    overrides: Exec<'_>,
) -> Result<GpuDataFrame> {
    for column in columns {
        let narrowed =
            input.frame().on(overrides).cast_column(column, CudfDataType::new(TypeId::Int32)).map_err(|error| {
                cudf_query_error(OPERATION, "failed to cast a graph id column to Int32", error)
                    .with_fact("column", *column)
            })?;
        input = GpuDataFrame::try_new_with_source_residency(narrowed, &input)?;
    }
    Ok(input)
}

#[cfg(test)]
mod tests {
    use super::{IdRange, graph_fits_int32_vertex_ids};

    const LIMIT: i64 = i32::MAX as i64;

    #[test]
    fn int32_vertex_ids_require_every_id_below_the_int32_maximum() {
        let endpoints = Some(IdRange::new(0, 1_060_000_000));
        assert!(graph_fits_int32_vertex_ids(1_740_318_538, endpoints, &[]));
        assert!(graph_fits_int32_vertex_ids(4, Some(IdRange::new(i64::from(i32::MIN), LIMIT - 1)), &[]));
        assert!(!graph_fits_int32_vertex_ids(4, Some(IdRange::new(0, LIMIT)), &[]));
        assert!(!graph_fits_int32_vertex_ids(4, Some(IdRange::new(i64::from(i32::MIN) - 1, 0)), &[]));
        assert!(!graph_fits_int32_vertex_ids(4, None, &[]));
    }

    #[test]
    fn int32_vertex_ids_require_side_input_and_literal_ids_to_fit() {
        let endpoints = Some(IdRange::new(0, 10));
        assert!(graph_fits_int32_vertex_ids(4, endpoints, &[IdRange::new(-5, LIMIT - 1)]));
        assert!(!graph_fits_int32_vertex_ids(4, endpoints, &[IdRange::new(3, 3), IdRange::new(0, LIMIT)]));
        assert!(!graph_fits_int32_vertex_ids(4, endpoints, &[IdRange::new(5_000_000_000, 5_000_000_000)]));
        assert!(!graph_fits_int32_vertex_ids(4, endpoints, &[IdRange::new(i64::from(i32::MIN) - 1, 0)]));
    }

    #[test]
    fn int32_vertex_ids_require_edge_rows_and_vertex_count_to_fit() {
        let endpoints = Some(IdRange::new(0, 10));
        assert!(graph_fits_int32_vertex_ids(LIMIT as usize - 1, endpoints, &[]));
        assert!(!graph_fits_int32_vertex_ids(LIMIT as usize, endpoints, &[]));
        // Ids spanning the whole Int32 range can name more than `i32::MAX`
        // distinct vertices once there are enough rows to hold them.
        let full_span = Some(IdRange::new(i64::from(i32::MIN), LIMIT - 1));
        assert!(graph_fits_int32_vertex_ids(LIMIT as usize / 2, full_span, &[]));
        assert!(!graph_fits_int32_vertex_ids(LIMIT as usize / 2 + 1, full_span, &[]));
    }
}

use std::time::Instant;

use query_planning::plan::GraphEdgeColumnBindings;
use query_runtime::metrics::OperatorMetrics;

use super::{DataFrame, ExecutionContext, ExecutionMetrics, NodeId, Result, col, cudf_query_error};

/// Which optional edge columns may gate null-row dropping for an operation.
///
/// `source`/`destination` always gate: a null endpoint is not a valid graph
/// edge. Optional columns only gate when the consuming operation reads them;
/// dropping an edge because an unread column is null corrupts the graph
/// topology (e.g. BFS reachability). See the BFS regression in
/// `graph_algorithm_execution.rs`.
#[derive(Clone, Copy)]
pub(super) struct EdgeNullFilterScope {
    /// Whether a null `weight` should drop its edge. True only for algorithms
    /// that read edge weights (SSSP, minimum spanning tree).
    pub(super) weight: bool,
    /// Whether a null `edge_id` should drop its edge. True only when edge ids
    /// are consumed (edge-id graph construction / edge-id predicates).
    pub(super) edge_id: bool,
}

pub(super) fn filter_null_edge_rows(
    context: &ExecutionContext,
    node_id: NodeId,
    operator_kind: &'static str,
    frame: DataFrame,
    edge_columns: &GraphEdgeColumnBindings,
    scope: EdgeNullFilterScope,
    metrics: &ExecutionMetrics,
) -> Result<DataFrame> {
    let columns = edge_filter_columns(edge_columns, scope);
    let mut null_counts = Vec::new();
    for (role, column) in &columns {
        let series = frame.series(column).map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.edge_null_filter",
                "failed to inspect cuGraph edge-list column nulls",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("edge_column_role", *role)
            .with_fact("column", *column)
        })?;
        let null_count = series.null_count();
        if null_count > 0 {
            null_counts.push((*role, *column, null_count));
        }
    }

    if null_counts.is_empty() {
        return Ok(frame);
    }

    let input_rows = frame.num_rows() as u64;
    let predicates = columns.iter().map(|(_, column)| col(*column).is_not_null()).collect::<Vec<_>>();
    let timer = Instant::now();
    let exec_runtime = context.runtime()?;
    let overrides = exec_runtime.exec();
    let output = frame.on(overrides).filter_all(predicates).map_err(|error| {
        cudf_query_error(
            "query_engine.graph_algorithm.edge_null_filter",
            "failed to drop null cuGraph edge-list rows",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("null_edge_columns", null_counts_metadata(&null_counts))
    })?;
    metrics.add_operator_time(timer.elapsed());

    let output_rows = output.num_rows() as u64;
    metrics.record_operator_metrics(OperatorMetrics {
        node_id: node_id.to_string(),
        operator_kind: operator_kind.to_string(),
        input_rows,
        output_rows,
        ..Default::default()
    });
    metrics.record_null_edge_rows_dropped(node_id, input_rows.saturating_sub(output_rows));
    Ok(output)
}

fn edge_filter_columns(
    edge_columns: &GraphEdgeColumnBindings,
    scope: EdgeNullFilterScope,
) -> Vec<(&'static str, &str)> {
    let mut columns =
        vec![("source", edge_columns.source.as_str()), ("destination", edge_columns.destination.as_str())];
    if scope.weight
        && let Some(weight) = edge_columns.weight.as_deref()
    {
        columns.push(("weight", weight));
    }
    if scope.edge_id
        && let Some(edge_id) = edge_columns.edge_id.as_deref()
    {
        columns.push(("edge_id", edge_id));
    }
    columns
}

fn null_counts_metadata(null_counts: &[(&'static str, &str, usize)]) -> String {
    null_counts.iter().map(|(role, column, count)| format!("{role}:{column}:{count}")).collect::<Vec<_>>().join(",")
}

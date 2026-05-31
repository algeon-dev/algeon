use super::*;
use crate::{metrics::OperatorMetrics, plan::GraphEdgeColumnBindings};

pub(super) fn filter_null_edge_rows(
    context: &ExecutionContext,
    node_id: NodeId,
    operator_kind: &'static str,
    frame: DataFrame,
    edge_columns: &GraphEdgeColumnBindings,
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let columns = edge_filter_columns(edge_columns);
    let mut null_counts = Vec::new();
    for (role, column) in &columns {
        let series = frame.series(column).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.graph_algorithm.edge_null_filter",
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
    let output = frame.with_execution(&execution_overrides(context)).filter_all(predicates).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.graph_algorithm.edge_null_filter",
            "failed to drop null cuGraph edge-list rows",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("null_edge_columns", null_counts_metadata(&null_counts))
    })?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    metrics.record_operator_metrics(OperatorMetrics {
        node_id: node_id.to_string(),
        operator_kind: operator_kind.to_string(),
        input_rows,
        output_rows,
        ..Default::default()
    });
    Ok(output)
}

fn edge_filter_columns(edge_columns: &GraphEdgeColumnBindings) -> Vec<(&'static str, &str)> {
    let mut columns =
        vec![("source", edge_columns.source.as_str()), ("destination", edge_columns.destination.as_str())];
    if let Some(weight) = edge_columns.weight.as_deref() {
        columns.push(("weight", weight));
    }
    if let Some(edge_id) = edge_columns.edge_id.as_deref() {
        columns.push(("edge_id", edge_id));
    }
    columns
}

fn null_counts_metadata(null_counts: &[(&'static str, &str, usize)]) -> String {
    null_counts.iter().map(|(role, column, count)| format!("{role}:{column}:{count}")).collect::<Vec<_>>().join(",")
}

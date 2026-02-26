use super::*;
use crate::plan::{EdgeDedupStrategy, EdgeNormalizeNode, EdgeNormalizeSpec, EdgeWeightCombiner};

pub(super) fn execute_edge_normalize(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &EdgeNormalizeNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let timer = Instant::now();
    let overrides = execution_overrides(context);
    let mut frame = project_edge_columns(input.into_frame(), &node.spec, node_id)?;

    if node.spec.add_reverse {
        let reverse = reverse_edges(&frame, &node.spec, node_id)?;
        frame = frame.with_execution(&overrides).concat_many(&[&reverse]).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.edge_normalize.reverse_concat",
                "failed to append reverse edges for native edge normalization",
                error,
            )
            .with_fact("node_id", node_id)
        })?;
    }

    if node.spec.dedup != EdgeDedupStrategy::None {
        let mut sort_by = vec![node.spec.edge_columns.source.as_str(), node.spec.edge_columns.destination.as_str()];
        if node.spec.weight_combiner == EdgeWeightCombiner::MinWeight
            && let Some(weight) = node.spec.edge_columns.weight.as_deref()
        {
            sort_by.push(weight);
        }
        frame = frame
            .with_execution(&overrides)
            .sort_values(sort_by, CudfSortOptions::new().with_stable(true))
            .map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.edge_normalize.sort",
                    "failed to sort edges for native edge normalization",
                    error,
                )
                .with_fact("node_id", node_id)
            })?;
        frame = frame
            .with_execution(&overrides)
            .drop_duplicates(
                DropDuplicatesOptions::builder()
                    .subset(&[node.spec.edge_columns.source.as_str(), node.spec.edge_columns.destination.as_str()][..])
                    .keep(match node.spec.dedup {
                        EdgeDedupStrategy::None => DuplicateKeepOption::KeepAny,
                        EdgeDedupStrategy::KeepFirst => DuplicateKeepOption::KeepFirst,
                        EdgeDedupStrategy::KeepLast => DuplicateKeepOption::KeepLast,
                    })
                    .build(),
            )
            .map_err(|error| {
                QueryError::from_cudf(
                    "nexus_query_engine.edge_normalize.dedup",
                    "failed to collapse duplicate edges for native edge normalization",
                    error,
                )
                .with_fact("node_id", node_id)
            })?;
    }

    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(frame))
}

fn project_edge_columns(frame: DataFrame, spec: &EdgeNormalizeSpec, node_id: NodeId) -> Result<DataFrame> {
    let mut columns = vec![spec.edge_columns.source.as_str(), spec.edge_columns.destination.as_str()];
    if let Some(weight) = spec.edge_columns.weight.as_deref() {
        columns.push(weight);
    }
    if let Some(edge_id) = spec.edge_columns.edge_id.as_deref() {
        columns.push(edge_id);
    }
    frame.select(columns).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.edge_normalize.project",
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

fn reverse_edges(frame: &DataFrame, spec: &EdgeNormalizeSpec, node_id: NodeId) -> Result<DataFrame> {
    let source = frame.series(&spec.edge_columns.source).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.edge_normalize.reverse_source",
            "failed to access source column for native reverse-edge expansion",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("column", &spec.edge_columns.source)
    })?;
    let destination = frame.series(&spec.edge_columns.destination).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.edge_normalize.reverse_destination",
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
            QueryError::from_cudf(
                "nexus_query_engine.edge_normalize.reverse_weight",
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
            QueryError::from_cudf(
                "nexus_query_engine.edge_normalize.reverse_edge_id",
                "failed to access edge-id column for native reverse-edge expansion",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("column", edge_id)
        })?;
        pairs.push((edge_id.clone(), series));
    }
    DataFrame::from_series(pairs).map_err(|error| {
        QueryError::from_cudf(
            "nexus_query_engine.edge_normalize.reverse_frame",
            "failed to assemble reverse edges for native edge normalization",
            error,
        )
        .with_fact("node_id", node_id)
    })
}

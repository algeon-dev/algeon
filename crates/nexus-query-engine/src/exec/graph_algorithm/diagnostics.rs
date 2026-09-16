use arrow::datatypes::SchemaRef;

use super::{
    DataFrame, NodeId, QueryError, QueryErrorCode, QueryMetrics, Result, cudf_dtype_from_arrow_type, cudf_query_error,
};

#[cfg(feature = "cugraph")]
pub(super) fn graph_algorithm_metadata_with_null_drops(metadata: String, null_edge_rows_dropped: u64) -> String {
    let null_metadata = format!("null_edge_rows_dropped={null_edge_rows_dropped}");
    if metadata.is_empty() { null_metadata } else { format!("{metadata},{null_metadata}") }
}

#[cfg(feature = "cugraph")]
pub(in crate::exec) fn graph_input_null_edge_rows_dropped(metrics: &QueryMetrics, input: NodeId) -> u64 {
    let input_node_id = input.to_string();
    metrics
        .operators
        .iter()
        .filter(|row| row.node_id == input_node_id && row.operator_kind == "graph_edge_null_filter")
        .map(|row| row.input_rows.saturating_sub(row.output_rows))
        .sum()
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_algorithm_parameters(params: &crate::plan::GraphAlgorithmParams) -> String {
    match params {
        crate::plan::GraphAlgorithmParams::PageRank(params) => {
            format!("alpha={},epsilon={},max_iterations={}", params.alpha, params.epsilon, params.max_iterations)
        },
        crate::plan::GraphAlgorithmParams::PersonalizedPageRank(params) => format!(
            "alpha={},epsilon={},max_iterations={},personalization_node={},personalization_vertex_col={},personalization_value_col={}",
            params.pagerank.alpha,
            params.pagerank.epsilon,
            params.pagerank.max_iterations,
            params.personalization.input,
            params.personalization.vertex,
            params.personalization.value,
        ),
        crate::plan::GraphAlgorithmParams::Bfs(params) => format!(
            "source_vertex={},depth_limit={}",
            params.source_vertex,
            params.depth_limit.map_or_else(|| "none".to_string(), |value| value.to_string())
        ),
        crate::plan::GraphAlgorithmParams::Louvain(params) => {
            format!("max_level={},threshold={},resolution={}", params.max_level, params.threshold, params.resolution)
        },
        crate::plan::GraphAlgorithmParams::EigenvectorCentrality(params) => {
            format!("epsilon={},max_iterations={}", params.epsilon, params.max_iterations)
        },
        crate::plan::GraphAlgorithmParams::KatzCentrality(params) => format!(
            "alpha={},beta={},epsilon={},max_iterations={}",
            params.alpha, params.beta, params.epsilon, params.max_iterations,
        ),
        crate::plan::GraphAlgorithmParams::Hits(params) => format!(
            "epsilon={},max_iterations={},normalize={}",
            params.epsilon, params.max_iterations, params.normalize,
        ),
        crate::plan::GraphAlgorithmParams::BetweennessCentrality(params) => format!(
            "seeds={},k={},normalized={},include_endpoints={},exact_vertex_threshold={}",
            graph_betweenness_seeds_label(params.seeds.as_ref()),
            params.k.map_or_else(|| "none".to_string(), |k| k.to_string()),
            params.normalized,
            params.include_endpoints,
            params.exact_vertex_threshold,
        ),
        crate::plan::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => format!(
            "seeds={},k={},normalized={},exact_vertex_threshold={}",
            graph_betweenness_seeds_label(params.seeds.as_ref()),
            params.k.map_or_else(|| "none".to_string(), |k| k.to_string()),
            params.normalized,
            params.exact_vertex_threshold,
        ),
        crate::plan::GraphAlgorithmParams::Sssp(params) => format!(
            "source_vertex={},cutoff={}",
            graph_sssp_source_label(&params.source_vertex),
            params.cutoff.map_or_else(|| "none".to_string(), |v| v.to_string()),
        ),
        crate::plan::GraphAlgorithmParams::DegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::InDegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::OutDegreesAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::WeaklyConnectedComponents(_) => String::new(),
        crate::plan::GraphAlgorithmParams::StronglyConnectedComponents(_) => String::new(),
        crate::plan::GraphAlgorithmParams::TriangleCountAll(_) => String::new(),
        crate::plan::GraphAlgorithmParams::CoreNumber(params) => {
            format!("degree_type={}", params.degree_type.as_str())
        },
        crate::plan::GraphAlgorithmParams::KCore(params) => {
            format!("k={},degree_type={}", params.k, params.degree_type.as_str())
        },
        crate::plan::GraphAlgorithmParams::MinimumSpanningTree(_) => String::new(),
        crate::plan::GraphAlgorithmParams::Leiden(params) => format!(
            "max_level={},resolution={},theta={},seed={}",
            params.max_level, params.resolution, params.theta, params.seed,
        ),
        crate::plan::GraphAlgorithmParams::Ecg(params) => format!(
            "min_weight={},ensemble_size={},max_level={},threshold={},resolution={},seed={}",
            params.min_weight, params.ensemble_size, params.max_level, params.threshold, params.resolution, params.seed,
        ),
        crate::plan::GraphAlgorithmParams::SpectralModularityMaximization(params) => format!(
            "n_clusters={},n_eigenvectors={},evs_tolerance={},evs_max_iterations={},\
             k_means_tolerance={},k_means_max_iterations={},seed={}",
            params.n_clusters,
            params.n_eigenvectors,
            params.evs_tolerance,
            params.evs_max_iterations,
            params.k_means_tolerance,
            params.k_means_max_iterations,
            params.seed,
        ),
        crate::plan::GraphAlgorithmParams::ForceAtlas2(params) => format!(
            "max_iter={},outbound_attraction_distribution={},lin_log_mode={},prevent_overlapping={},\
             overlap_scaling_ratio={},edge_weight_influence={},jitter_tolerance={},barnes_hut_optimize={},\
             barnes_hut_theta={},scaling_ratio={},strong_gravity_mode={},gravity={},verbose={},seed={}",
            params.max_iter,
            params.outbound_attraction_distribution,
            params.lin_log_mode,
            params.prevent_overlapping,
            params.overlap_scaling_ratio,
            params.edge_weight_influence,
            params.jitter_tolerance,
            params.barnes_hut_optimize,
            params.barnes_hut_theta,
            params.scaling_ratio,
            params.strong_gravity_mode,
            params.gravity,
            params.verbose,
            params.seed,
        ),
        crate::plan::GraphAlgorithmParams::Similarity(params) => format!(
            "vertex_pairs_node={},first_column={},second_column={}",
            params.vertex_pairs.input, params.vertex_pairs.first_column, params.vertex_pairs.second_column,
        ),
    }
}

#[cfg(feature = "cugraph")]
fn graph_betweenness_seeds_label(seeds: Option<&crate::plan::GraphBetweennessSeeds>) -> String {
    seeds.map_or_else(
        || "all".to_string(),
        |seeds| match seeds {
            crate::plan::GraphBetweennessSeeds::I64(values) => values.len().to_string(),
            crate::plan::GraphBetweennessSeeds::Input { input, row_count_upper_bound } => {
                format!("input:{}:{}:rows<={row_count_upper_bound}", input.input, input.column)
            },
        },
    )
}

#[cfg(feature = "cugraph")]
fn graph_sssp_source_label(source: &crate::plan::GraphSsspSource) -> String {
    match source {
        crate::plan::GraphSsspSource::I64(value) => value.to_string(),
        crate::plan::GraphSsspSource::Input(input) => format!("input:{}:{}", input.input, input.column),
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn boundary_delta(
    before: &crate::metrics::BoundaryMetrics,
    after: &crate::metrics::BoundaryMetrics,
) -> crate::metrics::BoundaryMetrics {
    crate::metrics::BoundaryMetrics {
        gpu_resident_input_handoffs: after
            .gpu_resident_input_handoffs
            .saturating_sub(before.gpu_resident_input_handoffs),
        host_to_device_uploads: after.host_to_device_uploads.saturating_sub(before.host_to_device_uploads),
        host_to_device_uploaded_rows: after
            .host_to_device_uploaded_rows
            .saturating_sub(before.host_to_device_uploaded_rows),
        host_to_device_uploaded_bytes: after
            .host_to_device_uploaded_bytes
            .saturating_sub(before.host_to_device_uploaded_bytes),
        device_to_host_materializations: after
            .device_to_host_materializations
            .saturating_sub(before.device_to_host_materializations),
        device_to_host_materialized_rows: after
            .device_to_host_materialized_rows
            .saturating_sub(before.device_to_host_materialized_rows),
        device_to_host_materialized_bytes: after
            .device_to_host_materialized_bytes
            .saturating_sub(before.device_to_host_materialized_bytes),
        device_to_device_copies: after.device_to_device_copies.saturating_sub(before.device_to_device_copies),
        device_to_device_copied_bytes: after
            .device_to_device_copied_bytes
            .saturating_sub(before.device_to_device_copied_bytes),
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn normalize_graph_result_schema(mut frame: DataFrame, schema: &SchemaRef) -> Result<DataFrame> {
    for field in schema.fields() {
        let field_name = field.name();
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "nexus_query_engine.graph_algorithm.schema_normalize",
                "cuGraph result schema contains an unsupported output type",
            )
            .with_fact("field_name", field_name)
            .with_fact("expected_type", format!("{:?}", field.data_type()))
        })?;
        let actual_type = frame.series(field_name).and_then(|series| series.dtype()).map_err(|error| {
            cudf_query_error(
                "nexus_query_engine.graph_algorithm.schema_normalize",
                "failed to inspect cuGraph result column",
                error,
            )
            .with_fact("field_name", field_name)
        })?;
        if actual_type != target_type {
            frame = frame.cast_column(field_name, target_type).map_err(|error| {
                cudf_query_error(
                    "nexus_query_engine.graph_algorithm.schema_normalize",
                    "failed to cast cuGraph result column to the stable SQL schema",
                    error,
                )
                .with_fact("field_name", field_name)
                .with_fact("actual_cudf_type", format!("{actual_type:?}"))
                .with_fact("expected_arrow_type", format!("{:?}", field.data_type()))
            })?;
        }
    }
    let columns = schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    frame.select(columns).map_err(|error| {
        cudf_query_error(
            "nexus_query_engine.graph_algorithm.schema_normalize",
            "failed to order cuGraph result columns according to the stable SQL schema",
            error,
        )
    })
}

#[cfg(feature = "cugraph")]
pub(super) fn map_cugraph_error(
    operation: &'static str,
    message: &'static str,
) -> impl FnOnce(cugraph::Error) -> QueryError {
    move |error| {
        let query_error = QueryError::new(code_for_cugraph_error(&error), operation, message)
            .with_fact("cugraph_kind", error.kind())
            .with_fact("cugraph_status", error.status())
            .with_fact("cugraph_operation", error.operation());
        let query_error = ["algorithm", "iterations", "max_iterations", "epsilon"].into_iter().fold(
            query_error,
            |query_error, key| match error.context_value(key) {
                Some(value) => query_error.with_fact(key, value),
                None => query_error,
            },
        );
        let query_error = if error.kind() == cugraph::ErrorKind::OutOfMemory {
            let source = match error.allocation_source() {
                cugraph::AllocationSource::Unknown => crate::error::CugraphAllocationSource::Unknown,
                cugraph::AllocationSource::RmmOutOfMemory => crate::error::CugraphAllocationSource::RmmOutOfMemory,
                cugraph::AllocationSource::RmmBadAlloc => crate::error::CugraphAllocationSource::RmmBadAlloc,
                cugraph::AllocationSource::StdBadAlloc => crate::error::CugraphAllocationSource::StdBadAlloc,
            };
            query_error
                .with_cugraph_allocation_source(source)
                .with_fact("cugraph_allocation_failure_source", error.allocation_failure_source().as_str())
        } else {
            query_error
        };
        query_error.with_source(error)
    }
}

/// Classifies a cuGraph allocation failure by the owner it was attributed to.
///
/// This is the same attribution table the cuDF boundary uses: only a proven
/// physical provenance may become `DeviceOutOfMemory`, an owner reservation
/// limit is the query cap rejecting an allocation, and an unattributed
/// failure never impersonates either.
#[cfg(feature = "cugraph")]
fn code_for_cugraph_error(error: &cugraph::Error) -> QueryErrorCode {
    match error.kind() {
        cugraph::ErrorKind::OutOfMemory => match error.allocation_failure_source() {
            cugraph::AllocationFailureSource::ReservationLimit => QueryErrorCode::QueryMemoryLimitExceeded,
            cugraph::AllocationFailureSource::Physical => QueryErrorCode::DeviceOutOfMemory,
            cugraph::AllocationFailureSource::Unknown => QueryErrorCode::DeviceAllocationFailed,
        },
        cugraph::ErrorKind::InvalidInput | cugraph::ErrorKind::InvalidHandle => QueryErrorCode::CudfInvalidInput,
        cugraph::ErrorKind::UnsupportedType | cugraph::ErrorKind::NotImplemented => QueryErrorCode::CudfUnsupported,
        cugraph::ErrorKind::InvalidState => QueryErrorCode::InternalInvariantViolation,
        cugraph::ErrorKind::NonConvergence => QueryErrorCode::AlgorithmDidNotConverge,
        cugraph::ErrorKind::IoError | cugraph::ErrorKind::Unexpected => match error.status() {
            cugraph::ErrorStatus::Temporary => QueryErrorCode::CudfTemporaryFailure,
            cugraph::ErrorStatus::Persistent => QueryErrorCode::DeviceFault,
            cugraph::ErrorStatus::Permanent => QueryErrorCode::CudfDependencyFailure,
        },
        _ => QueryErrorCode::CudfDependencyFailure,
    }
}

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use super::*;

    #[test]
    fn cugraph_oom_mapping_preserves_fail_closed_allocation_source_fact() {
        let source = cugraph::Error::new(cugraph::ErrorKind::OutOfMemory, "cugraph.test.oom").set_temporary();

        let error = map_cugraph_error("nexus_query_engine.graph_algorithm.test", "cuGraph test OOM")(source);

        // An unattributed cuGraph allocation failure never impersonates a
        // physical device OOM or a query-cap rejection.
        assert_eq!(error.code(), QueryErrorCode::DeviceAllocationFailed);
        assert_eq!(error.fact_value("cugraph_allocation_failure_source"), Some("unknown"));
        assert_eq!(error.fact_value("cugraph_allocation_source"), Some("unknown"));
        assert_eq!(error.cugraph_allocation_source(), Some(crate::error::CugraphAllocationSource::Unknown));

        let dependency_error =
            map_cugraph_error("nexus_query_engine.graph_algorithm.test", "cuGraph dependency failure")(
                cugraph::Error::new(cugraph::ErrorKind::InvalidInput, "cugraph.test.invalid_input"),
            );

        assert_eq!(dependency_error.fact_value("cugraph_allocation_source"), None);
        assert_eq!(dependency_error.cugraph_allocation_source(), None);
    }

    #[test]
    fn cugraph_allocation_failures_map_to_the_same_taxonomy_as_cudf() {
        // The cuGraph boundary must classify allocation failures by the same
        // attributed source as the cuDF boundary: an owner reservation limit
        // is the query cap rejecting an allocation, a proven physical failure
        // is device OOM, and an unattributed one is neither.
        for (failure_source, expected) in [
            (cugraph::AllocationFailureSource::ReservationLimit, QueryErrorCode::QueryMemoryLimitExceeded),
            (cugraph::AllocationFailureSource::Physical, QueryErrorCode::DeviceOutOfMemory),
            (cugraph::AllocationFailureSource::Unknown, QueryErrorCode::DeviceAllocationFailed),
        ] {
            let source = cugraph::Error::new(cugraph::ErrorKind::OutOfMemory, "cugraph.test.attributed_oom")
                .with_allocation_failure_source(failure_source);

            let error = map_cugraph_error("nexus_query_engine.graph_algorithm.test", "cuGraph attributed OOM")(source);

            assert_eq!(error.code(), expected, "failure source {failure_source} must map to {expected:?}");
            assert_eq!(error.fact_value("cugraph_allocation_failure_source"), Some(failure_source.as_str()));
        }

        // Routing and lifetime contradictions are programming defects, not
        // user input errors.
        let invalid_state = map_cugraph_error("nexus_query_engine.graph_algorithm.test", "cuGraph invalid state")(
            cugraph::Error::new(cugraph::ErrorKind::InvalidState, "cugraph.test.invalid_state"),
        );
        assert_eq!(invalid_state.code(), QueryErrorCode::InternalInvariantViolation);
    }

    #[test]
    fn cugraph_nonconvergence_mapping_preserves_actionable_facts() {
        let source = cugraph::Error::new(cugraph::ErrorKind::NonConvergence, "eigenvector_centrality")
            .with_context("algorithm", "eigenvector_centrality")
            .with_context("iterations", "200")
            .with_context("max_iterations", "200")
            .with_context("epsilon", "0.00000001");

        let error = map_cugraph_error(
            "nexus_query_engine.graph_algorithm.eigenvector_centrality",
            "cuGraph eigenvector centrality failed",
        )(source);

        assert_eq!(error.code(), QueryErrorCode::AlgorithmDidNotConverge);
        assert_eq!(error.fact_value("algorithm"), Some("eigenvector_centrality"));
        assert_eq!(error.fact_value("iterations"), Some("200"));
        assert_eq!(error.fact_value("max_iterations"), Some("200"));
        assert_eq!(error.fact_value("epsilon"), Some("0.00000001"));
        assert_eq!(error.fact_value("cugraph_status"), Some("permanent"));
    }
}

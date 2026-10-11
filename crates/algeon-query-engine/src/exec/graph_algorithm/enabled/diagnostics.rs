use arrow::datatypes::SchemaRef;
use cudf::Exec;
#[cfg(feature = "cugraph")]
use cugraph::AllocationFailureSource;
#[cfg(feature = "cugraph")]
use cugraph::AllocationSource as CugraphAllocationSource;
#[cfg(feature = "cugraph")]
use cugraph::ErrorKind;
#[cfg(feature = "cugraph")]
use cugraph::ErrorStatus;
#[cfg(feature = "cugraph")]
use query_planning::error::AllocationSource as QueryAllocationSource;
#[cfg(feature = "cugraph")]
use query_planning::error::redact_uri_tokens;
#[cfg(feature = "cugraph")]
use query_planning::plan::GraphAlgorithmParams;
#[cfg(feature = "cugraph")]
use query_planning::plan::GraphBetweennessSeeds;
#[cfg(feature = "cugraph")]
use query_planning::plan::GraphSsspSource;
#[cfg(feature = "cugraph")]
use query_runtime::metrics::BoundaryMetrics;

use super::{
    DataFrame, QueryError, QueryErrorCode, Result, cudf_dtype_from_arrow_type, cudf_query_error,
    cudf_unsupported_type_error,
};

#[cfg(feature = "cugraph")]
pub(super) fn graph_algorithm_metadata_with_null_drops(metadata: String, null_edge_rows_dropped: u64) -> String {
    let null_metadata = format!("null_edge_rows_dropped={null_edge_rows_dropped}");
    if metadata.is_empty() { null_metadata } else { format!("{metadata},{null_metadata}") }
}

#[cfg(feature = "cugraph")]
pub(super) fn graph_algorithm_parameters(params: &GraphAlgorithmParams) -> String {
    match params {
        GraphAlgorithmParams::PageRank(params) => {
            format!("alpha={},epsilon={},max_iterations={}", params.alpha, params.epsilon, params.max_iterations)
        },
        GraphAlgorithmParams::PersonalizedPageRank(params) => format!(
            "alpha={},epsilon={},max_iterations={},personalization_input={},personalization_vertex_col={},personalization_value_col={}",
            params.pagerank.alpha,
            params.pagerank.epsilon,
            params.pagerank.max_iterations,
            params.personalization.input,
            params.personalization.vertex,
            params.personalization.value,
        ),
        GraphAlgorithmParams::Bfs(params) => format!(
            "source_vertex={},depth_limit={}",
            params.source_vertex,
            params.depth_limit.map_or_else(|| "none".to_string(), |value| value.to_string())
        ),
        GraphAlgorithmParams::Louvain(params) => {
            format!("max_level={},threshold={},resolution={}", params.max_level, params.threshold, params.resolution)
        },
        GraphAlgorithmParams::EigenvectorCentrality(params) => {
            format!("epsilon={},max_iterations={}", params.epsilon, params.max_iterations)
        },
        GraphAlgorithmParams::KatzCentrality(params) => format!(
            "alpha={},beta={},epsilon={},max_iterations={}",
            params.alpha, params.beta, params.epsilon, params.max_iterations,
        ),
        GraphAlgorithmParams::Hits(params) => format!(
            "epsilon={},max_iterations={},normalize={}",
            params.epsilon, params.max_iterations, params.normalize,
        ),
        GraphAlgorithmParams::BetweennessCentrality(params) => format!(
            "seeds={},k={},normalized={},include_endpoints={},exact_vertex_threshold={}",
            graph_betweenness_seeds_label(params.seeds.as_ref()),
            params.k.map_or_else(|| "none".to_string(), |k| k.to_string()),
            params.normalized,
            params.include_endpoints,
            params.exact_vertex_threshold,
        ),
        GraphAlgorithmParams::EdgeBetweennessCentrality(params) => format!(
            "seeds={},k={},normalized={},exact_vertex_threshold={}",
            graph_betweenness_seeds_label(params.seeds.as_ref()),
            params.k.map_or_else(|| "none".to_string(), |k| k.to_string()),
            params.normalized,
            params.exact_vertex_threshold,
        ),
        GraphAlgorithmParams::Sssp(params) => format!(
            "source_vertex={},cutoff={}",
            graph_sssp_source_label(&params.source_vertex),
            params.cutoff.map_or_else(|| "none".to_string(), |v| v.to_string()),
        ),
        GraphAlgorithmParams::DegreesAll(_) => String::new(),
        GraphAlgorithmParams::InDegreesAll(_) => String::new(),
        GraphAlgorithmParams::OutDegreesAll(_) => String::new(),
        GraphAlgorithmParams::WeaklyConnectedComponents(_) => String::new(),
        GraphAlgorithmParams::StronglyConnectedComponents(_) => String::new(),
        GraphAlgorithmParams::TriangleCountAll(_) => String::new(),
        GraphAlgorithmParams::CoreNumber(params) => {
            format!("degree_type={}", params.degree_type.as_str())
        },
        GraphAlgorithmParams::KCore(params) => {
            format!("k={},degree_type={}", params.k, params.degree_type.as_str())
        },
        GraphAlgorithmParams::MinimumSpanningTree(_) => String::new(),
        GraphAlgorithmParams::Leiden(params) => {
            format!("max_level={},resolution={},seed={}", params.max_level, params.resolution, params.seed)
        },
        GraphAlgorithmParams::Ecg(params) => format!(
            "min_weight={},ensemble_size={},max_level={},threshold={},resolution={},seed={}",
            params.min_weight, params.ensemble_size, params.max_level, params.threshold, params.resolution, params.seed,
        ),
        GraphAlgorithmParams::SpectralModularityMaximization(params) => format!(
            "n_clusters={},n_eigenvectors={},evs_tolerance={},seed={}",
            params.n_clusters, params.n_eigenvectors, params.evs_tolerance, params.seed,
        ),
        GraphAlgorithmParams::ForceAtlas2(params) => format!(
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
        GraphAlgorithmParams::Similarity(params) => format!(
            "vertex_pairs_input={},first_column={},second_column={}",
            params.vertex_pairs.input, params.vertex_pairs.first_column, params.vertex_pairs.second_column,
        ),
    }
}

#[cfg(feature = "cugraph")]
fn graph_betweenness_seeds_label(seeds: Option<&GraphBetweennessSeeds>) -> String {
    seeds.map_or_else(
        || "all".to_string(),
        |seeds| match seeds {
            GraphBetweennessSeeds::I64(values) => values.len().to_string(),
            GraphBetweennessSeeds::Input { input, row_count_upper_bound } => {
                format!("input:{}:{}:rows<={row_count_upper_bound}", input.input, input.column)
            },
        },
    )
}

#[cfg(feature = "cugraph")]
fn graph_sssp_source_label(source: &GraphSsspSource) -> String {
    match source {
        GraphSsspSource::I64(value) => value.to_string(),
        GraphSsspSource::Input(input) => format!("input:{}:{}", input.input, input.column),
    }
}

#[cfg(feature = "cugraph")]
pub(super) fn boundary_delta(before: &BoundaryMetrics, after: &BoundaryMetrics) -> BoundaryMetrics {
    BoundaryMetrics {
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
        let target_type = cudf_dtype_from_arrow_type(field.data_type()).map_err(|error| {
            cudf_unsupported_type_error(
                "query_engine.graph_algorithm.schema_normalize",
                "cuGraph result schema contains an unsupported output type",
                error,
            )
            .with_fact("field_name", field_name)
            .with_fact("expected_type", format!("{:?}", field.data_type()))
        })?;
        let actual_type = frame.series(field_name).and_then(|series| series.dtype()).map_err(|error| {
            cudf_query_error(
                "query_engine.graph_algorithm.schema_normalize",
                "failed to inspect cuGraph result column",
                error,
            )
            .with_fact("field_name", field_name)
        })?;
        if actual_type != target_type {
            frame = frame.on(Exec::RETAINED).cast_column(field_name, target_type).map_err(|error| {
                cudf_query_error(
                    "query_engine.graph_algorithm.schema_normalize",
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
            "query_engine.graph_algorithm.schema_normalize",
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
        let cugraph_message =
            Some(error.message().trim()).filter(|cugraph_message| !cugraph_message.is_empty()).map(redact_uri_tokens);
        // An invalid-input or unsupported cuGraph error states what is wrong with the caller's
        // call, and libcugraph reports precondition failures without source locations, so its
        // text joins the message that reaches the client as `native_message`.
        let message = match &cugraph_message {
            Some(cugraph_message)
                if matches!(
                    error.kind(),
                    ErrorKind::InvalidInput | ErrorKind::UnsupportedType | ErrorKind::NotImplemented
                ) =>
            {
                format!("{message}: {cugraph_message}")
            },
            _ => message.to_string(),
        };
        let query_error = QueryError::new(code_for_cugraph_error(&error), operation, message)
            .with_fact("cugraph_kind", error.kind())
            .with_fact("cugraph_status", error.status())
            .with_fact("cugraph_operation", error.operation());
        let query_error = match cugraph_message {
            Some(cugraph_message) => query_error.with_fact("cugraph_message", cugraph_message),
            None => query_error,
        };
        let query_error = ["algorithm", "iterations", "max_iterations", "epsilon", "remedy"].into_iter().fold(
            query_error,
            |query_error, key| match error.context_value(key) {
                Some(value) => query_error.with_fact(key, value),
                None => query_error,
            },
        );
        let query_error = if error.kind() == ErrorKind::OutOfMemory {
            let source = match error.allocation_source() {
                CugraphAllocationSource::Unknown => QueryAllocationSource::Unknown,
                CugraphAllocationSource::RmmOutOfMemory => QueryAllocationSource::RmmOutOfMemory,
                CugraphAllocationSource::RmmBadAlloc => QueryAllocationSource::RmmBadAlloc,
                CugraphAllocationSource::StdBadAlloc => QueryAllocationSource::StdBadAlloc,
            };
            query_error
                .with_allocation_source(source)
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
/// failure maps to neither of them.
#[cfg(feature = "cugraph")]
fn code_for_cugraph_error(error: &cugraph::Error) -> QueryErrorCode {
    match error.kind() {
        ErrorKind::OutOfMemory => match error.allocation_failure_source() {
            AllocationFailureSource::ReservationLimit => QueryErrorCode::QueryMemoryLimitExceeded,
            AllocationFailureSource::Physical => QueryErrorCode::DeviceOutOfMemory,
            AllocationFailureSource::Unknown => QueryErrorCode::DeviceAllocationFailed,
        },
        ErrorKind::InvalidInput | ErrorKind::InvalidHandle => QueryErrorCode::CudfInvalidInput,
        ErrorKind::UnsupportedType | ErrorKind::NotImplemented => QueryErrorCode::CudfUnsupported,
        ErrorKind::InvalidState => QueryErrorCode::InternalInvariantViolation,
        ErrorKind::NonConvergence => QueryErrorCode::AlgorithmDidNotConverge,
        ErrorKind::IoError | ErrorKind::Unexpected => match error.status() {
            ErrorStatus::Temporary => QueryErrorCode::CudfTemporaryFailure,
            ErrorStatus::Persistent => QueryErrorCode::DeviceFault,
            ErrorStatus::Permanent => QueryErrorCode::CudfDependencyFailure,
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

        let error = map_cugraph_error("query_engine.graph_algorithm.test", "cuGraph test OOM")(source);

        // An unattributed cuGraph allocation failure maps to neither a physical
        // device OOM nor a query-cap rejection.
        assert_eq!(error.code(), QueryErrorCode::DeviceAllocationFailed);
        assert_eq!(error.fact_value("cugraph_allocation_failure_source"), Some("unknown"));
        assert_eq!(error.context().allocation_source, Some(QueryAllocationSource::Unknown));
        assert!(error.wire_facts().any(|(key, value)| key == "cugraph_allocation_source" && value == "unknown"));

        let dependency_error = map_cugraph_error("query_engine.graph_algorithm.test", "cuGraph dependency failure")(
            cugraph::Error::new(cugraph::ErrorKind::InvalidInput, "cugraph.test.invalid_input"),
        );

        assert_eq!(dependency_error.context().allocation_source, None);
    }

    #[cfg(feature = "test-hooks")]
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

            let error = map_cugraph_error("query_engine.graph_algorithm.test", "cuGraph attributed OOM")(source);

            assert_eq!(error.code(), expected, "failure source {failure_source} must map to {expected:?}");
            assert_eq!(error.fact_value("cugraph_allocation_failure_source"), Some(failure_source.as_str()));
        }

        // Routing and lifetime contradictions are programming defects, not
        // user input errors.
        let invalid_state = map_cugraph_error("query_engine.graph_algorithm.test", "cuGraph invalid state")(
            cugraph::Error::new(cugraph::ErrorKind::InvalidState, "cugraph.test.invalid_state"),
        );
        assert_eq!(invalid_state.code(), QueryErrorCode::InternalInvariantViolation);
    }

    #[test]
    fn cugraph_nonconvergence_mapping_preserves_actionable_facts() {
        for (algorithm, remedy) in [
            ("eigenvector_centrality", "raise max_iterations or epsilon"),
            ("katz_centrality", "lower alpha (currently 0.5) below 1/lambda_max or raise max_iterations"),
            ("hits", "raise max_iterations or epsilon"),
        ] {
            let source = cugraph::Error::new(cugraph::ErrorKind::NonConvergence, algorithm)
                .with_context("algorithm", algorithm)
                .with_context("iterations", "200")
                .with_context("max_iterations", "200")
                .with_context("epsilon", "0.00000001")
                .with_context("remedy", remedy);

            let error =
                map_cugraph_error("query_engine.graph_algorithm.centrality", "cuGraph centrality failed")(source);

            assert_eq!(error.code(), QueryErrorCode::AlgorithmDidNotConverge, "{algorithm}");
            assert_eq!(error.fact_value("algorithm"), Some(algorithm));
            assert_eq!(error.fact_value("iterations"), Some("200"));
            assert_eq!(error.fact_value("max_iterations"), Some("200"));
            assert_eq!(error.fact_value("epsilon"), Some("0.00000001"));
            assert_eq!(error.fact_value("remedy"), Some(remedy));
            assert_eq!(error.fact_value("cugraph_status"), Some("permanent"));
        }
    }

    #[test]
    fn cugraph_invalid_input_text_joins_the_message_and_internal_text_stays_a_fact() {
        let invalid = map_cugraph_error("query_engine.graph_algorithm.sssp", "cuGraph SSSP failed")(
            cugraph::Error::new(cugraph::ErrorKind::InvalidInput, "sssp")
                .with_message("SSSP source vertex is not in the graph"),
        );
        assert_eq!(invalid.code(), QueryErrorCode::CudfInvalidInput);
        assert_eq!(invalid.message(), "cuGraph SSSP failed: SSSP source vertex is not in the graph");

        let unexpected = map_cugraph_error("query_engine.graph_algorithm.sssp", "cuGraph SSSP failed")(
            cugraph::Error::new(cugraph::ErrorKind::Unexpected, "sssp").with_message("internal scheduler detail"),
        );
        assert_eq!(unexpected.code(), QueryErrorCode::CudfDependencyFailure);
        assert_eq!(unexpected.message(), "cuGraph SSSP failed");
        assert_eq!(unexpected.fact_value("cugraph_message"), Some("internal scheduler detail"));
    }
}

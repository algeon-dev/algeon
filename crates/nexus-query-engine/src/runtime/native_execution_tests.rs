use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow::{
    array::{
        Array, BooleanArray, Date32Array, Decimal128Array, DurationNanosecondArray, Float32Array, Float64Array,
        Int8Array, Int32Array, Int64Array, LargeListArray, LargeStringArray, ListArray, StringArray,
        TimestampNanosecondArray, UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
#[cfg(feature = "cugraph")]
use nexus_query_engine::exec::execute_query_plan_with_gpu_inputs_observed;
#[cfg(feature = "cugraph")]
use nexus_query_engine::observability::{
    PlanExecutionBoundaryKind, RuntimeObservationConfig, RuntimeObservationPayload, RuntimeTerminalOutcome,
};
use nexus_query_engine::{
    error::QueryErrorCode,
    exec::{execute_query_plan, execute_query_plan_with_gpu_inputs, output::QueryOutput},
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortExpr, SortSpec,
    },
    metrics::{MemoryReservationEventAction, QueryMetrics},
    mode::StatisticalAggregateExecutionMode,
    plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, ProjectionOutput, QueryPlanBuilder},
    runtime::{
        BoundedJoinReservationContract, ExecutionBoundaryPolicy, ExecutionContext, MemoryControlPolicy,
        ReservationKind,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{
        GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource, PartitionColumn, SourceNode,
        ZERO_COLUMN_COUNT_COLUMN,
    },
};
#[cfg(feature = "cugraph")]
use nexus_query_engine::{
    exec::force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test,
    exec::force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test,
    graph_memory::GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES, graph_memory::GraphMemoryCalibrationOverride,
    graph_memory::GraphReservationOutcome, plan::GraphAlgorithm, plan::GraphAlgorithmParams, plan::GraphAlgorithmSpec,
    plan::GraphBetweennessCentralityParams, plan::GraphBetweennessSeeds, plan::GraphBfsEndpointPolicy,
    plan::GraphBfsParams, plan::GraphBfsPredicateInputs, plan::GraphConstructionOptions, plan::GraphConstructionPolicy,
    plan::GraphEdgeBetweennessCentralityParams, plan::GraphEdgeColumnBindings, plan::GraphPageRankParams,
    plan::GraphPredicateInput, plan::GraphSimilarityParams, plan::GraphVertexPairInput,
    plan::stable_graph_result_schema,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

#[test]
fn execute_query_plan_stamps_correlation_on_early_errors() {
    use nexus_query_engine::{
        metrics::{QueryId, TraceId},
        runtime::ExecutionCancellation,
    };

    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch])));
    let plan = builder.finish_with_host_sink(source);

    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();
    let context = ExecutionContext::builder()
        .with_cancellation(cancellation)
        .with_query_id(QueryId::new("query-exec"))
        .with_trace_id(TraceId::new("trace-exec"))
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("cancelled context should fail before execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.query_id().map(QueryId::as_str), Some("query-exec"));
    assert_eq!(error.trace_id().map(TraceId::as_str), Some("trace-exec"));
    assert_eq!(error.fact_value("query_id"), Some("query-exec"));
    assert_eq!(error.fact_value("trace_id"), Some("trace-exec"));
}

#[path = "native_execution_tests/aggregate_test_support.rs"]
mod aggregate_test_support;
#[path = "native_execution_tests/join_test_support.rs"]
mod join_test_support;
#[path = "native_execution_tests/metrics_assertions.rs"]
mod metrics_assertions;
#[path = "native_execution_tests/parquet_fixtures.rs"]
mod parquet_fixtures;
#[path = "native_execution_tests/test_support.rs"]
mod test_support;

#[path = "native_execution_tests/aggregate_execution.rs"]
mod aggregate_execution;
#[path = "native_execution_tests/bounded_budgeting.rs"]
mod bounded_budgeting;
#[path = "native_execution_tests/bounded_joins.rs"]
mod bounded_joins;
#[path = "native_execution_tests/core_execution.rs"]
mod core_execution;
#[cfg(feature = "cugraph")]
#[path = "native_execution_tests/graph_algorithm_execution.rs"]
mod graph_algorithm_execution;
#[path = "native_execution_tests/native_join_execution.rs"]
mod native_join_execution;
#[path = "native_execution_tests/semijoin_reduction.rs"]
mod semijoin_reduction;
#[path = "native_execution_tests/unbounded_pruning_regressions.rs"]
mod unbounded_pruning_regressions;
#[cfg(feature = "cuvs")]
#[path = "native_execution_tests/vector_algorithm_execution.rs"]
mod vector_algorithm_execution;
// This uncached execution suite runs with durable source caching disabled.

use join_test_support::*;
use parquet_fixtures::*;
use test_support::*;

fn assert_bounded_join_rejection_contract(error: &nexus_query_engine::error::QueryError) {
    let contract = BoundedJoinReservationContract;
    let expected_facts = contract.unsupported_shape_facts(ExecutionBoundaryPolicy::default());

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "nexus_query_engine.bounded.plan");
    assert_eq!(error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));

    for (key, expected) in &expected_facts {
        if matches!(
            *key,
            "bounded_join_reservation_kinds"
                | "bounded_join_reservation_statuses"
                | "blocked_reservations"
                | "missing_reservations"
                | "modeled_reservations"
                | "bounded_join_lower_level_blockers"
                | "bounded_join_reservation_requirements"
        ) {
            continue;
        }
        assert_eq!(
            error.fact_value(key),
            Some(expected.as_str()),
            "bounded join unsupported-shape fact mismatch for {key}"
        );
    }

    assert_eq!(error.fact_value("bounded_join_reservation_model"), Some(contract.model_status()));
    assert_eq!(error.fact_value("bounded_join_execution_families"), None);
    assert_eq!(error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(
        error.fact_value("boundary_policy"),
        Some(
            "source_read=direct_device_local_parquet,build_materialization=query_scoped_gpu_hash_join,probe_execution=reusable_gpu_hash_join,result_materialization=explicit_sink"
        )
    );
    assert_eq!(error.fact_value("source_read_boundary"), Some("direct_device_local_parquet"));
    assert_eq!(error.fact_value("build_materialization_boundary"), Some("query_scoped_gpu_hash_join"));
    assert_eq!(error.fact_value("probe_execution_boundary"), Some("reusable_gpu_hash_join"));
    assert_eq!(error.fact_value("result_materialization_boundary"), Some("explicit_sink"));
    assert_bounded_join_rejection_reservation_facts_are_consistent(error);
    assert_eq!(error.fact_value("deferred_reservations"), Some(contract.deferred_reservation_kinds_csv().as_str()));
    assert_eq!(error.fact_value("cudf_rs_hash_table_observability"), Some("hash_table_device_allocated_size_bytes"));
    assert_eq!(error.fact_value("cudf_rs_gather_map_observability"), Some("join_indices_device_allocated_size_bytes"));
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_gather_map_reservation_contract"),
        Some("hash_join_gather_map_reservation_size_bytes")
    );
    assert_eq!(error.fact_value("cudf_rs_hash_join_output_size_observability"), Some("hash_join_exact_output_size"));
    assert_eq!(error.fact_value("cudf_rs_hash_join_probe_output_size_hint"), Some("hash_join_probe_with_output_size"));
    assert_eq!(
        error.fact_value("cudf_rs_conditional_join_output_size_observability"),
        Some("conditional_join_exact_output_size")
    );
    assert_eq!(
        error.fact_value("cudf_rs_conditional_join_output_size_hint"),
        Some("conditional_join_indices_with_output_size")
    );
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_pre_build_reservation_contract"),
        Some("hash_join_pre_build_reservation_size_bytes")
    );
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_pre_build_reservation_status"),
        Some("available_exact_flat_hash_join_builds")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_hash_table_observability"),
        Some("filtered_join_device_allocated_size_bytes")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_output_size_observability"),
        Some("filtered_join_probe_state_output_size")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_probe_state_contract"),
        Some("filtered_join_retained_probe_state")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_probe_state_observability"),
        Some("filtered_join_probe_device_allocated_size_bytes")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_probe_materialization_contract"),
        Some("filtered_join_probe_state_materialize_indices")
    );
    assert_eq!(
        error.fact_value("cudf_rs_dataframe_gather_output_reservation_contract"),
        Some("dataframe_gather_output_reservation_size_bytes")
    );
    assert_eq!(error.fact_value("missing_lower_level_contract"), None);
    assert_eq!(error.fact_value("missing_reservations"), Some(contract.blocking_reservation_kinds_csv().as_str()));
    assert!(error.facts().iter().all(|(key, value)| !key.contains("estimate") && !value.contains("estimate")));
    assert_eq!(error.fact_value("path"), None);
    assert_eq!(error.fact_value("file_index"), None);
    assert_eq!(error.fact_value("row_group_indexes"), None);
    assert_eq!(error.fact_value("row_group_count"), None);
    assert_eq!(error.fact_value("chunk_index"), None);
    assert!(error.source_error().is_none());
}

fn assert_bounded_join_rejection_reservation_facts_are_consistent(error: &nexus_query_engine::error::QueryError) {
    let modeled = error
        .fact_value("modeled_reservations")
        .map(csv_values)
        .expect("bounded join rejection should include modeled reservations");
    let statuses = error
        .fact_value("bounded_join_reservation_statuses")
        .map(csv_pairs)
        .expect("bounded join rejection should include reservation statuses");
    let requirements = error
        .fact_value("bounded_join_reservation_requirements")
        .map(csv_triples)
        .expect("bounded join rejection should include reservation requirements");
    let kinds = error
        .fact_value("bounded_join_reservation_kinds")
        .map(csv_values)
        .expect("bounded join rejection should include reservation kinds");

    assert_eq!(statuses.iter().map(|(kind, _)| *kind).collect::<Vec<_>>(), kinds);
    assert_eq!(requirements.iter().map(|(kind, _, _)| *kind).collect::<Vec<_>>(), kinds);
    assert_eq!(
        modeled,
        statuses.iter().filter_map(|(kind, status)| (*status == "modeled").then_some(*kind)).collect::<Vec<_>>()
    );
    assert_eq!(error.fact_value("blocked_reservations"), Some(""));
    assert_eq!(error.fact_value("missing_reservations"), Some(""));
    assert_eq!(error.fact_value("bounded_join_lower_level_blockers"), Some(""));
}

fn csv_values(value: &str) -> Vec<&str> {
    value.split(',').filter(|value| !value.is_empty()).collect()
}

fn csv_pairs(value: &str) -> Vec<(&str, &str)> {
    value.split(',').filter_map(|row| row.split_once(':')).collect()
}

fn csv_triples(value: &str) -> Vec<(&str, &str, &str)> {
    value
        .split(',')
        .filter_map(|row| {
            let (left, detail) = row.rsplit_once(':')?;
            let (kind, contract) = left.split_once(':')?;
            Some((kind, contract, detail))
        })
        .collect()
}

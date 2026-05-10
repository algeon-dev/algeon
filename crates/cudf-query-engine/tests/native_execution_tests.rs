use std::{
    collections::HashMap,
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow::{
    array::{
        Array, BooleanArray, Date32Array, Decimal128Array, DurationNanosecondArray, Float32Array, Float64Array,
        Int8Array, Int32Array, Int64Array, StringArray, TimestampNanosecondArray,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
use cudf_query_engine::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, BoundedJoinReservationContract,
    BuildMaterializationBoundary, ExecutionBoundaryPolicy, ExecutionContext, Expr, GpuDataFrame, GpuHashJoinBuildCache,
    GpuLocalParquetSourceCache, GpuResidentInputMap, GpuResidentSource, GpuResultHandle, JoinKey, JoinNullEquality,
    JoinOutputColumn, JoinSide, JoinSpec, JoinType, LimitSpec, LiteralValue, LocalParquetFile, LocalParquetSource,
    MemoryControlPolicy, MemorySpaceId, MissingColumn, PartitionColumn, ProjectionOutput, QueryErrorCode, QueryMetrics,
    QueryOutput, QueryPlanBuilder, ReservationKind, SortExpr, SortSpec, SourceChunkPlanningPolicy, SourceNode,
    SourceSchedulerPolicy, ZERO_COLUMN_COUNT_COLUMN, execute_query_plan, execute_query_plan_with_gpu_inputs,
    materialize_gpu_result_to_host, plan_local_parquet_source_chunks,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

#[path = "native_execution_tests/aggregate_test_support.rs"]
mod aggregate_test_support;
#[path = "native_execution_tests/bounded_budgeting.rs"]
mod bounded_budgeting;
#[path = "native_execution_tests/bounded_joins.rs"]
mod bounded_joins;
#[path = "native_execution_tests/join_test_support.rs"]
mod join_test_support;
#[path = "native_execution_tests/metrics_assertions.rs"]
mod metrics_assertions;
#[path = "native_execution_tests/native_join_execution.rs"]
mod native_join_execution;
#[path = "native_execution_tests/parquet_fixtures.rs"]
mod parquet_fixtures;
#[path = "native_execution_tests/projection_test_support.rs"]
mod projection_test_support;
#[path = "native_execution_tests/projection_validation.rs"]
mod projection_validation;
#[path = "native_execution_tests/test_support.rs"]
mod test_support;

#[path = "native_execution_tests/aggregate_execution.rs"]
mod aggregate_execution;
#[path = "native_execution_tests/core_execution.rs"]
mod core_execution;
#[path = "native_execution_tests/partition_execution.rs"]
mod partition_execution;

use aggregate_test_support::*;
use join_test_support::*;
use metrics_assertions::*;
use parquet_fixtures::*;
use projection_test_support::*;
use test_support::*;

fn assert_bounded_join_rejection_contract(error: &cudf_query_engine::QueryError) {
    let contract = BoundedJoinReservationContract;
    let expected_facts = contract.unsupported_shape_facts(ExecutionBoundaryPolicy::default());

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "cudf_query_engine.bounded.plan");
    assert_eq!(error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));

    for (key, expected) in &expected_facts {
        assert_eq!(
            error.fact_value(key),
            Some(expected.as_str()),
            "bounded join unsupported-shape fact mismatch for {key}"
        );
    }

    assert_eq!(error.fact_value("bounded_join_reservation_model"), Some(contract.model_status()));
    assert_eq!(error.fact_value("bounded_join_execution_families"), Some(contract.execution_families_csv().as_str()));
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
    assert_eq!(error.fact_value("bounded_join_reservation_kinds"), Some(contract.reservation_kinds_csv().as_str()));
    assert_eq!(
        error.fact_value("bounded_join_reservation_statuses"),
        Some(contract.reservation_statuses_csv().as_str())
    );
    assert_eq!(error.fact_value("blocked_reservations"), Some(contract.blocking_reservation_kinds_csv().as_str()));
    assert_eq!(error.fact_value("modeled_reservations"), Some(contract.modeled_reservation_kinds_csv().as_str()));
    assert_eq!(error.fact_value("deferred_reservations"), Some(contract.deferred_reservation_kinds_csv().as_str()));
    assert_eq!(
        error.fact_value("bounded_join_lower_level_blockers"),
        Some(contract.lower_level_blockers_csv().as_str())
    );
    assert_eq!(
        error.fact_value("bounded_join_reservation_requirements"),
        Some(contract.requirement_details_csv().as_str())
    );
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
    assert_eq!(error.fact_value("row_groups"), None);
    assert_eq!(error.fact_value("chunk_index"), None);
    assert!(error.source_error().is_none());
}

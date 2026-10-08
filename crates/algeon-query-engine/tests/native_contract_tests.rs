#[path = "backend_metric_descriptor_tests.rs"]
mod backend_metric_descriptor_tests;

use std::{
    collections::BTreeSet,
    fs::{self, File},
    mem::size_of,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Int32Array, StructArray},
    datatypes::{DataType, Field, Fields, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::{
    DataType as CudfDataType, TypeId,
    interop::arrow_host::{cudf_dtype_from_arrow_type, is_supported_arrow_type},
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use query_engine::chunk::{
    SourceMetadataCache, plan_local_parquet_source_chunks, plan_local_parquet_source_chunks_with_metadata_cache,
};
use query_planning::{
    capability::{
        AggregateExecutionStage, AggregateExecutionStrategy, CapabilityDecision, EdgeContractKind,
        KeyedHashJoinStrategyOptions, NodeOperator, SourceFormatCapability, SubgraphInvariantKind,
        aggregate_strategy_capability_decision,
    },
    chunk::{SourceChunkPlanningPolicy, SourceTaskMemoryModel},
    correlation::{QueryId, TraceId},
    datatypes::{is_supported_arrow_cast, is_supported_arrow_implicit_cast, is_supported_list_carry_type},
    error::{QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus},
    execution_contracts::MemoryControlPolicy,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, DatePart, Expr, LimitSpec,
        LiteralValue, SortExpr, SortSpec,
    },
    plan::{
        JoinKey, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanNode, ProjectionOutput,
        QueryPlanBuilder, ResultSink, WindowFunction, WindowSpec,
    },
    source::{
        CredentialMode, GpuResidentSource, LakehouseCacheFillMode, LakehouseCacheReadMode, LakehouseCacheStatus,
        LiteralBatchSource, LocalParquetFile, LocalParquetSource, MissingColumn, ParquetReaderDecimalType,
        ParquetReaderDecimalTypeId, PartitionColumn, PartitionColumnDisposition, PredicateDisposition,
        RejectedSourceFeature, ResolvedParquetSource, ScanLimitDisposition, SchemaMappingDisposition, SourceColumnRef,
        SourceNode, ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition, validate_local_parquet_source,
    },
};
use query_runtime::{
    metrics::{BoundaryMetrics, JoinOperatorMetrics, OperatorMetrics, QueryMetrics},
    runtime::{ExecutionCancellation, SourceSchedulerPolicy},
};

fn projection_outputs(expressions: Vec<Expr>, schema: Arc<Schema>) -> Vec<ProjectionOutput> {
    assert_eq!(expressions.len(), schema.fields().len(), "projection outputs must match schema");
    expressions
        .into_iter()
        .zip(schema.fields().iter())
        .map(|(expr, field)| ProjectionOutput::new(expr, field.as_ref().clone()))
        .collect()
}

#[test]
fn execution_cancellation_reports_stable_error_contract() {
    let cancellation = ExecutionCancellation::new();
    assert!(!cancellation.is_cancelled());
    cancellation.cancel();

    let error = cancellation.check("query_engine.test.cancel").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.kind(), QueryErrorKind::Cancelled);
    assert_eq!(error.status(), QueryErrorStatus::Permanent);
    assert_eq!(error.operation(), "query_engine.test.cancel");
    assert_eq!(error.fact_value("cancellation_requested"), Some("true"));
}

#[test]
fn execution_cancellation_clones_share_state_across_threads() {
    // The Flight SQL `do_action_cancel_query` handler flips the cancellation
    // flag from a tokio worker thread while synchronous native execution
    // observes the flag from the executor thread. Both sides hold cloned
    // `ExecutionCancellation` handles; cancellation only short-circuits
    // native execution if the underlying `Arc<AtomicBool>` is shared. This
    // test pins that contract so a future refactor cannot quietly drop the
    // Arc and reintroduce the cooperative-cancellation regression Bug 9
    // describes.
    let cancellation = ExecutionCancellation::new();
    let observer = cancellation.clone();
    assert!(!observer.is_cancelled());

    let writer = cancellation.clone();
    let handle = std::thread::spawn(move || {
        writer.cancel();
    });
    handle.join().expect("cancellation writer thread completed");

    assert!(cancellation.is_cancelled(), "original handle must observe a clone's cancel()");
    assert!(observer.is_cancelled(), "independent clone must observe a sibling clone's cancel()");

    let error = observer.check("query_engine.test.cross_thread_cancel").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.operation(), "query_engine.test.cross_thread_cancel");
}

#[test]
fn execution_cancellation_default_is_independent_of_other_instances() {
    // `ExecutionContext::new` / `AlgeonNativeExecutionConfig::new` construct a
    // fresh `ExecutionCancellation` by default. Two independently-constructed
    // contexts must not share a flag; otherwise cancelling one query would
    // poison every other in-flight query in the server.
    let one = ExecutionCancellation::default();
    let two = ExecutionCancellation::default();
    one.cancel();

    assert!(one.is_cancelled());
    assert!(!two.is_cancelled(), "default-constructed instances must not share state");
    two.check("query_engine.test.independent_default").expect("uncancelled instance check passes");
}

#[path = "native_contract_tests/attempt_lifecycle.rs"]
mod attempt_lifecycle;
#[path = "native_contract_tests/bounded_join_admission.rs"]
pub(crate) mod bounded_join_admission;
#[path = "native_contract_tests/capability_first_pipeline_admission.rs"]
mod capability_first_pipeline_admission;
#[path = "native_contract_tests/dependency_direction.rs"]
mod dependency_direction;
#[path = "native_contract_tests/expr_contracts.rs"]
mod expr_contracts;
#[path = "native_contract_tests/graph_algorithm.rs"]
mod graph_algorithm;
#[path = "native_contract_tests/ml_algorithm.rs"]
mod ml_algorithm;
#[path = "native_contract_tests/optimizer_contracts.rs"]
mod optimizer_contracts;
#[path = "native_contract_tests/plan_contracts.rs"]
mod plan_contracts;
#[path = "native_contract_tests/plan_validation.rs"]
mod plan_validation;
#[path = "native_contract_tests/preference_contracts.rs"]
mod preference_contracts;
#[path = "native_contract_tests/public_facade.rs"]
mod public_facade;
#[path = "native_contract_tests/runtime_observability.rs"]
mod runtime_observability;
#[path = "native_contract_tests/shadow.rs"]
mod shadow;
#[path = "native_contract_tests/source_contracts.rs"]
mod source_contracts;
#[path = "native_contract_tests/type_support.rs"]
mod type_support;
#[path = "native_contract_tests/vector_algorithm.rs"]
mod vector_algorithm;

#[test]
fn query_error_size_is_pinned() {
    assert_eq!(size_of::<QueryError>(), 8);
    assert!(size_of::<Result<(), QueryError>>() <= 16);
}

#[test]
fn query_error_codes_and_facts_are_stable_without_datafusion() {
    let error = QueryError::new(
        QueryErrorCode::UnsupportedExpression,
        "query_engine.plan.validate",
        "expression is not supported by algeon native query engine",
    )
    .with_fact("expression", "Utf8 LIKE")
    .with_fact("reason", "unsupported_expression");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.code().as_str(), "unsupported_expression");
    assert_eq!(error.kind(), QueryErrorKind::Unsupported);
    assert_eq!(error.status(), QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("reason"), Some("unsupported_expression"));
    assert!(error.to_string().contains("code=unsupported_expression"));
    assert!(error.to_string().contains("expression=\"Utf8 LIKE\""));
}

#[test]
fn query_error_carries_typed_correlation_and_fact_labels() {
    let error =
        QueryError::new(QueryErrorCode::MemoryReservationFailed, "query_engine.test.correlation", "reservation denied")
            .with_query_id(QueryId::new("query-42"))
            .with_trace_id(TraceId::new("trace-7"));

    assert_eq!(error.query_id().map(QueryId::as_str), Some("query-42"));
    assert_eq!(error.trace_id().map(TraceId::as_str), Some("trace-7"));
    assert_eq!(error.fact_value("query_id"), Some("query-42"));
    assert_eq!(error.fact_value("trace_id"), Some("trace-7"));
    assert!(error.to_string().contains("query_id=query-42"));
}

#[test]
fn query_error_correlation_stamping_replaces_existing_correlation_facts() {
    let error = QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.test.correlation_replace",
        "correlation changed at adapter boundary",
    )
    .with_fact("query_id", "service-local-query")
    .with_fact("trace_id", "service-local-trace")
    .with_query_id(QueryId::new("request-query"))
    .with_trace_id(TraceId::new("request-trace"));

    assert_eq!(error.query_id().map(QueryId::as_str), Some("request-query"));
    assert_eq!(error.trace_id().map(TraceId::as_str), Some("request-trace"));
    assert_eq!(error.fact_value("query_id"), Some("request-query"));
    assert_eq!(error.fact_value("trace_id"), Some("request-trace"));
}

#[test]
fn source_storage_error_codes_have_stable_policy() {
    let cases = [
        (QueryErrorCode::SourceAccessDenied, "source_access_denied", QueryErrorStatus::Permanent),
        (QueryErrorCode::SourceUnauthenticated, "source_unauthenticated", QueryErrorStatus::Permanent),
        (QueryErrorCode::SourceObjectNotFound, "source_object_not_found", QueryErrorStatus::Permanent),
        (QueryErrorCode::SourceTemporaryFailure, "source_temporary_failure", QueryErrorStatus::Temporary),
    ];

    for (code, label, status) in cases {
        assert_eq!(code.as_str(), label);
        assert_eq!(code.kind(), QueryErrorKind::DependencyFailure);
        assert_eq!(code.status(), status);
    }
}

#[test]
fn memory_control_policy_converts_to_source_chunk_planning_policy() {
    let bounded = MemoryControlPolicy {
        device_budget_bytes: 4 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 512 * 1024 * 1024,
        max_row_groups_per_chunk: 2,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    let chunk_policy = SourceChunkPlanningPolicy::from(bounded);
    assert_eq!(chunk_policy.max_source_chunk_bytes, Some(512 * 1024 * 1024));
    assert_eq!(chunk_policy.max_row_groups_per_chunk, 2);
}

#[test]
fn source_scheduler_policy_defaults_to_prefetch_and_rejects_zero_in_flight() {
    let policy = SourceSchedulerPolicy::default();
    assert_eq!(policy.max_source_reads_in_flight(), 2);
    assert!(policy.enables_prefetch());

    let serialized = SourceSchedulerPolicy::serialized();
    assert_eq!(serialized.max_source_reads_in_flight(), 1);
    assert!(!serialized.enables_prefetch());

    let error = SourceSchedulerPolicy::new(0).validate().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration);
    assert_eq!(error.operation(), "query_engine.source_scheduler_policy.validate");
    assert_eq!(error.fact_value("field"), Some("max_source_reads_in_flight"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn bounded_memory_control_policy_rejects_zero_fields_before_runtime_initialization() {
    for (field, policy) in [
        (
            "device_budget_bytes",
            MemoryControlPolicy {
                device_budget_bytes: 0,
                max_source_chunk_bytes: 512 * 1024 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            },
        ),
        (
            "max_source_chunk_bytes",
            MemoryControlPolicy {
                device_budget_bytes: 4 * 1024 * 1024,
                max_source_chunk_bytes: 0,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            },
        ),
        (
            "max_row_groups_per_chunk",
            MemoryControlPolicy {
                device_budget_bytes: 4 * 1024 * 1024,
                max_source_chunk_bytes: 512 * 1024,
                max_row_groups_per_chunk: 0,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            },
        ),
        (
            "max_retry_attempts",
            MemoryControlPolicy {
                device_budget_bytes: 4 * 1024 * 1024,
                max_source_chunk_bytes: 512 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 0,
                min_retry_chunk_row_groups: 1,
            },
        ),
        (
            "min_retry_chunk_row_groups",
            MemoryControlPolicy {
                device_budget_bytes: 4 * 1024 * 1024,
                max_source_chunk_bytes: 512 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 0,
            },
        ),
    ] {
        let error = policy.validate().unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration, "{field}");
        assert_eq!(error.operation(), "query_engine.memory_policy.validate", "{field}");
        assert_eq!(error.fact_value("policy"), Some("bounded"), "{field}");
        assert_eq!(error.fact_value("field"), Some(field), "{field}");
        assert_eq!(error.fact_value("reason"), Some("zero"), "{field}");
    }
}

#[test]
fn source_chunk_planning_policy_rejects_zero_row_group_limit() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/part-000.parquet").with_file_size_bytes(300).with_row_groups(vec![0])],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));

    let error = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(Some(1024), 0))
        .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.source_chunk.policy");
    assert_eq!(error.fact_value("field"), Some("max_row_groups_per_chunk"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn query_metrics_make_host_device_boundaries_explicit() {
    let mut metrics = QueryMetrics::default();
    metrics.record_parquet_source(Some(3_235_196_281), 12, 34);
    metrics.boundary.record_gpu_resident_input_handoff();
    metrics.boundary.record_device_to_host_materialization(100, 8192);

    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.parquet_bytes_known, 3_235_196_281);
    assert_eq!(metrics.row_groups_selected, 12);
    assert_eq!(metrics.read_columns, 34);
    assert_eq!(metrics.boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 100);

    let mut boundary = BoundaryMetrics::default();
    boundary.record_host_to_device_upload(10, 1024);
    assert_eq!(boundary.host_to_device_uploads, 1);
    assert_eq!(boundary.host_to_device_uploaded_bytes, 1024);
}

#[test]
fn query_metrics_have_join_operator_contract() {
    let mut metrics = QueryMetrics::default();
    metrics.record_operator_metrics(OperatorMetrics::join(
        "n2",
        JoinOperatorMetrics {
            query_id: None,
            join_type: "inner".into(),
            execution_strategy: "hash_join".into(),
            key_count: 2,
            residual_predicate: true,
            null_equality: "nulls_do_not_match".into(),
            build_side: "right".into(),
            probe_side: "left".into(),
            build_rows: 10,
            build_bytes: 1024,
            probe_rows: 20,
            probe_bytes: 2048,
            hash_table_bytes: 4096,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes: 256,
            output_reservation_bytes: Some(0),
            output_rows: 5,
            output_bytes: 512,
        },
    ));

    assert_eq!(metrics.operators.len(), 1);
    let join = metrics.operators[0].join.as_ref().expect("join metrics should be attached");
    assert_eq!(join.join_type, "inner");
    assert_eq!(join.execution_strategy, "hash_join");
    assert_eq!(join.key_count, 2);
    assert_eq!(join.hash_table_bytes, 4096);
    assert_eq!(join.filtered_probe_state_bytes, 0);
    assert_eq!(join.mixed_size_data_bytes, 0);
    assert_eq!(join.gather_map_bytes, 256);
    assert_eq!(join.build_side, "right");
    assert_eq!(join.probe_side, "left");

    metrics.record_hash_join_build();
    metrics.record_hash_join_hash_table_bytes(4096);
    metrics.record_filtered_join_build();
    metrics.record_filtered_join_hash_table_bytes(2048);
    metrics.record_filtered_join_probe_state_bytes(1024);
    metrics.record_mixed_join_size_data_bytes(512);
    metrics.record_one_shot_join();
    metrics.record_one_shot_join_index_bytes(128);
    metrics.record_one_shot_join_materialization_bytes(256);
    metrics.record_one_shot_join_output_bytes(384);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.filtered_join_build_count, 1);
    assert_eq!(metrics.one_shot_join_count, 1);
    assert_eq!(metrics.hash_join_hash_table_bytes, 4096);
    assert_eq!(metrics.filtered_join_hash_table_bytes, 2048);
    assert_eq!(metrics.filtered_join_probe_state_bytes, 1024);
    assert_eq!(metrics.mixed_join_size_data_bytes, 512);
    assert_eq!(metrics.one_shot_join_index_bytes, 128);
    assert_eq!(metrics.one_shot_join_materialization_bytes, 256);
    assert_eq!(metrics.one_shot_join_output_bytes, 384);
}

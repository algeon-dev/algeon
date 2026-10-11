use cudf::testing::run_named_cases;
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
        KeyedHashJoinStrategyOptions, SubgraphInvariantKind, aggregate_strategy_capability_decision,
    },
    chunk::{SourceChunkPlanningPolicy, SourceTaskMemoryModel},
    correlation::{QueryId, TraceId},
    datatypes::{is_supported_arrow_cast, is_supported_arrow_implicit_cast, is_supported_list_carry_type},
    error::{QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus},
    execution_contracts::MemoryControlPolicy,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, Expr, LimitSpec, LiteralValue,
        SortExpr, SortSpec,
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
    metrics::QueryMetrics,
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

mod attempt_lifecycle;
pub(crate) mod bounded_join_admission;
mod capability_first_pipeline_admission;
mod dependency_direction;
mod expr_contracts;
mod graph_algorithm;
mod ml_algorithm;
mod optimizer_contracts;
mod plan_contracts;
mod plan_validation;
mod preference_contracts;
mod public_api_boundary;
mod public_facade;
mod runtime_observability;
mod shadow;
mod source_contracts;
mod type_support;
mod vector_algorithm;
fn query_error_size_is_pinned() {
    assert_eq!(size_of::<QueryError>(), 8);
    assert!(size_of::<Result<(), QueryError>>() <= 16);
}

#[test]
fn t_query_error_main_cases() {
    run_named_cases(&[
        ("query_error_size_is_pinned", query_error_size_is_pinned),
        (
            "query_error_codes_and_facts_are_stable_without_datafusion",
            query_error_codes_and_facts_are_stable_without_datafusion,
        ),
        (
            "query_error_carries_typed_correlation_and_fact_labels",
            query_error_carries_typed_correlation_and_fact_labels,
        ),
        (
            "query_error_correlation_stamping_replaces_existing_correlation_facts",
            query_error_correlation_stamping_replaces_existing_correlation_facts,
        ),
        ("source_storage_error_codes_have_stable_policy", source_storage_error_codes_have_stable_policy),
    ]);
}

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

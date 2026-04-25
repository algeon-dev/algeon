use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{ArrayRef, Int32Array, StructArray},
    datatypes::{DataType, Field, Fields, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::{DataType as CudfDataType, TypeId};
use nexus_query_engine::{
    capability::{
        AggregateExecutionStage, AggregateExecutionStrategy, CapabilityDecision, ContractQuality, ContractStatus,
        EdgeConsumer, EdgeContractKind, InputRole, KeyedHashJoinStrategyOptions, NodeOperator, OrderingRequirement,
        RequiredProperty, ReservationRequirementScope, ResidualRequirement, ResultSinkKind, ScheduleConstraint,
        SharedProducerIdentity, SourceFormatCapability, SubgraphInvariantKind, aggregate_strategy_capability_decision,
    },
    chunk::{
        SourceChunkPlanningPolicy, SourceMetadataCache, SourceTaskMemoryModel, plan_local_parquet_source_chunks,
        plan_local_parquet_source_chunks_with_metadata_cache,
    },
    contract_report::{StrictBoundedContractOwner, StrictBoundedContractReport},
    datatypes::{
        cudf_dtype_from_arrow_type, is_supported_arrow_cast, is_supported_arrow_implicit_cast, is_supported_arrow_type,
        is_supported_list_carry_type,
    },
    error::{QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus},
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, DatePart, Expr, LimitSpec,
        LiteralValue, SortExpr, SortSpec,
    },
    metrics::{BoundaryMetrics, JoinOperatorMetrics, OperatorMetrics, QueryId, QueryMetrics, TraceId},
    mode::StrictBoundedExecutionMode,
    optimizer::NativeOptimizerOptions,
    plan::{
        DataPlacement, JoinKey, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType, NodeId, PlanNode,
        ProjectionOutput, QueryPlanBuilder, ResultSink, SchemaBehavior, WindowFunction, WindowSpec,
    },
    runtime::{
        BoundedJoinReservationContract, BoundedJoinReservationStatus, BuildMaterializationBoundary,
        ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext, MemoryControlPolicy,
        MemoryReservationRequest, MemorySpaceId, ProbeExecutionBoundary, ReservationKind,
        ResultMaterializationBoundary, RetryFacts, SourceReadBoundary, SourceSchedulerPolicy,
        cache::local_parquet::GpuLocalParquetSourceCacheKey,
    },
    source::{
        ByteRangeDisposition, CredentialMode, GpuResidentSource, LakehouseCacheFillMode, LakehouseCacheReadMode,
        LakehouseCacheStatus, LiteralBatchSource, LocalParquetFile, LocalParquetSource, MissingColumn,
        ParquetEqualityDeleteFile, ParquetFieldIdReadBinding, ParquetPositionDeleteFile, ParquetReaderDecimalType,
        ParquetReaderDecimalTypeId, PartitionColumn, PartitionColumnDisposition, PredicateDisposition,
        RejectedSourceFeature, RemoteParquetReadMode, ResolvedParquetSource, ScanLimitDisposition,
        SchemaMappingDisposition, SourceColumnRef, SourceNode, ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition,
        validate_local_parquet_source,
    },
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

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

    let error = cancellation.check("nexus_query_engine.test.cancel").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.kind(), QueryErrorKind::Cancelled);
    assert_eq!(error.status(), QueryErrorStatus::Permanent);
    assert_eq!(error.operation(), "nexus_query_engine.test.cancel");
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

    let error = observer.check("nexus_query_engine.test.cross_thread_cancel").unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.operation(), "nexus_query_engine.test.cross_thread_cancel");
}

#[test]
fn execution_cancellation_default_is_independent_of_other_instances() {
    // `ExecutionContext::new` / `NativeCudfExecutionConfig::new` construct a
    // fresh `ExecutionCancellation` by default. Two independently-constructed
    // contexts must not share a flag — otherwise cancelling one query would
    // poison every other in-flight query in the server.
    let one = ExecutionCancellation::default();
    let two = ExecutionCancellation::default();
    one.cancel();

    assert!(one.is_cancelled());
    assert!(!two.is_cancelled(), "default-constructed instances must not share state");
    two.check("nexus_query_engine.test.independent_default").expect("uncancelled instance check passes");
}

#[path = "native_contract_tests/bounded_join_admission.rs"]
pub(crate) mod bounded_join_admission;
#[path = "native_contract_tests/bounded_join_sizing.rs"]
mod bounded_join_sizing;
#[path = "native_contract_tests/capability_module_boundaries.rs"]
mod capability_module_boundaries;
#[path = "native_contract_tests/dependency_direction.rs"]
mod dependency_direction;
#[path = "native_contract_tests/expr_contracts.rs"]
mod expr_contracts;
#[path = "native_contract_tests/graph_algorithm.rs"]
mod graph_algorithm;
#[path = "native_contract_tests/multi_device_registry.rs"]
mod multi_device_registry;
#[path = "native_contract_tests/optimizer_contracts.rs"]
mod optimizer_contracts;
#[path = "native_contract_tests/partial_aggregate_sizing.rs"]
mod partial_aggregate_sizing;
#[path = "native_contract_tests/pipeline_compiler.rs"]
mod pipeline_compiler;
#[path = "native_contract_tests/plan_contracts.rs"]
mod plan_contracts;
#[path = "native_contract_tests/preference_contracts.rs"]
mod preference_contracts;
#[path = "native_contract_tests/public_facade.rs"]
mod public_facade;
#[path = "../../../tests/support/public_surface.rs"]
mod public_surface;
#[path = "native_contract_tests/reservation_tracker_contracts.rs"]
mod reservation_tracker_contracts;
#[path = "native_contract_tests/source_contracts.rs"]
mod source_contracts;
#[path = "native_contract_tests/type_support.rs"]
mod type_support;
#[path = "native_contract_tests/yielded_frame.rs"]
mod yielded_frame;

#[test]
fn query_error_codes_and_facts_are_stable_without_datafusion() {
    let error = QueryError::new(
        QueryErrorCode::UnsupportedExpression,
        "nexus_query_engine.plan.validate",
        "expression is not supported by native cudf query engine",
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
    let error = QueryError::new(
        QueryErrorCode::MemoryReservationFailed,
        "nexus_query_engine.test.correlation",
        "reservation denied",
    )
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
        "nexus_query_engine.test.correlation_replace",
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
fn execute_query_plan_stamps_correlation_on_early_errors() {
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

    let error = match nexus_query_engine::exec::execute_query_plan(&context, &plan) {
        Ok(_) => panic!("cancelled context should fail before execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    assert_eq!(error.query_id().map(QueryId::as_str), Some("query-exec"));
    assert_eq!(error.trace_id().map(TraceId::as_str), Some("trace-exec"));
    assert_eq!(error.fact_value("query_id"), Some("query-exec"));
    assert_eq!(error.fact_value("trace_id"), Some("trace-exec"));
}

#[test]
fn memory_reservation_failure_is_temporary_resource_exhaustion() {
    let request = MemoryReservationRequest::new(ReservationKind::SourceRead, 1024, 512)
        .with_memory_boundary("source_chunk_read")
        .with_node_id("n0")
        .with_stage("source_read")
        .with_chunk_index(3)
        .with_file_index(1)
        .with_path("/tmp/input.parquet")
        .with_row_groups(vec![4, 5])
        .with_attempt(2, 4);

    let error = request.denied_error(256);

    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.code().as_str(), "memory_reservation_failed");
    assert_eq!(error.kind(), QueryErrorKind::ResourceExhausted);
    assert_eq!(error.status(), QueryErrorStatus::Temporary);
    assert_eq!(error.fact_value("memory_boundary"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("requested_bytes"), Some("1024"));
    assert_eq!(error.fact_value("granted_bytes"), Some("256"));
    assert_eq!(error.fact_value("budget_bytes"), Some("512"));
    assert_eq!(error.fact_value("node_id"), Some("n0"));
    assert_eq!(error.fact_value("chunk_index"), Some("3"));
    assert_eq!(error.fact_value("file_index"), Some("1"));
    assert_eq!(error.fact_value("path"), Some("/tmp/input.parquet"));
    assert_eq!(error.fact_value("row_group_indexes"), Some("4,5"));
    assert_eq!(error.fact_value("row_group_count"), Some("2"));
    assert_eq!(error.fact_value("attempt"), Some("2"));
    assert_eq!(error.fact_value("max_attempts"), Some("4"));
    assert_eq!(error.fact_value("memory_space"), None);
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
fn memory_space_id_display_and_request_defaults_are_stable() {
    let device_zero = MemorySpaceId::Device { device_ordinal: 0 };
    let device_one = MemorySpaceId::Device { device_ordinal: 1 };

    assert_eq!(device_zero.to_string(), "GPU:0");
    assert_eq!(device_one.to_string(), "GPU:1");

    let default_request = MemoryReservationRequest::new(ReservationKind::SourceRead, 1024, 4096);
    assert_eq!(default_request.memory_space, device_zero);

    let explicit_request = default_request.with_memory_space(device_one);
    assert_eq!(explicit_request.memory_space, device_one);
}

#[test]
fn memory_control_policy_names_bounded_and_unbounded_modes() {
    let unbounded = MemoryControlPolicy::Unbounded;
    let bounded = MemoryControlPolicy::Bounded {
        device_budget_bytes: 4 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 512 * 1024 * 1024,
        max_row_groups_per_chunk: 2,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    };

    assert_eq!(unbounded.as_str(), "unbounded");
    assert!(!unbounded.is_bounded());
    assert_eq!(bounded.as_str(), "bounded");
    assert!(bounded.is_bounded());
    assert_eq!(ReservationKind::SourceRead.as_str(), "source_read");
    assert_eq!(ReservationKind::SourceCacheRetained.as_str(), "source_cache_retained");
    assert_eq!(ReservationKind::ChunkOperator.as_str(), "chunk_operator");
    assert_eq!(ReservationKind::OperatorOutput.as_str(), "operator_output");
    assert_eq!(ReservationKind::FinalMerge.as_str(), "final_merge");
    assert_eq!(ReservationKind::BuildSide.as_str(), "build_side");
    assert_eq!(ReservationKind::ProbeSide.as_str(), "probe_side");
    assert_eq!(ReservationKind::HashTable.as_str(), "hash_table");
    assert_eq!(ReservationKind::FilteredProbeState.as_str(), "filtered_probe_state");
    assert_eq!(ReservationKind::MixedSizeData.as_str(), "mixed_size_data");
    assert_eq!(ReservationKind::GatherMap.as_str(), "gather_map");
    assert_eq!(ReservationKind::JoinOutput.as_str(), "join_output");
    assert_eq!(ReservationKind::PartialState.as_str(), "partial_state");
    assert_eq!(ReservationKind::FinalSort.as_str(), "final_sort");
    assert_eq!(ReservationKind::Materialization.as_str(), "materialization");

    let chunk_policy = SourceChunkPlanningPolicy::from(bounded);
    assert_eq!(chunk_policy.max_source_chunk_bytes, Some(512 * 1024 * 1024));
    assert_eq!(chunk_policy.max_row_groups_per_chunk, 2);
}

#[test]
fn execution_boundary_policy_defaults_to_native_gpu_direct_contract() {
    let policy = ExecutionBoundaryPolicy::default();

    assert_eq!(policy.source_read(), SourceReadBoundary::DirectDeviceLocalParquet);
    assert_eq!(policy.source_read().as_str(), "direct_device_local_parquet");
    assert_eq!(policy.build_materialization(), BuildMaterializationBoundary::QueryScopedGpuHashJoin);
    assert_eq!(policy.build_materialization().as_str(), "query_scoped_gpu_hash_join");
    assert_eq!(policy.probe_execution(), ProbeExecutionBoundary::ReusableGpuHashJoin);
    assert_eq!(policy.probe_execution().as_str(), "reusable_gpu_hash_join");
    assert_eq!(policy.result_materialization(), ResultMaterializationBoundary::ExplicitSink);
    assert_eq!(policy.result_materialization().as_str(), "explicit_sink");
    assert_eq!(
        policy.to_string(),
        "source_read=direct_device_local_parquet,build_materialization=query_scoped_gpu_hash_join,probe_execution=reusable_gpu_hash_join,result_materialization=explicit_sink"
    );

    let explicit = ExecutionBoundaryPolicy::new(
        SourceReadBoundary::DirectDeviceLocalParquet,
        BuildMaterializationBoundary::QueryScopedGpuHashJoin,
        ProbeExecutionBoundary::ReusableGpuHashJoin,
        ResultMaterializationBoundary::ExplicitSink,
    );
    assert_eq!(explicit, policy);
    explicit.validate(false).unwrap();
}

#[test]
fn source_scheduler_policy_defaults_to_prefetch_and_rejects_zero_in_flight() {
    let policy = SourceSchedulerPolicy::default();
    assert_eq!(policy.max_source_reads_in_flight(), 2);
    assert!(policy.enables_prefetch());

    let serialized = SourceSchedulerPolicy::serialized();
    assert_eq!(serialized.max_source_reads_in_flight(), 1);
    assert!(!serialized.enables_prefetch());

    let error =
        ExecutionContext::builder().with_source_scheduler_policy(SourceSchedulerPolicy::new(0)).build().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "nexus_query_engine.source_scheduler_policy.validate");
    assert_eq!(error.fact_value("field"), Some("max_source_reads_in_flight"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn service_stream_pool_rejects_zero_capacity_before_runtime_initialization() {
    let error = ExecutionContext::builder().with_service_stream_pool_size(0).build().unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "nexus_query_engine.service_stream_pool.validate");
    assert_eq!(error.fact_value("field"), Some("service_stream_pool_size"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn bounded_memory_control_policy_rejects_zero_fields_before_runtime_initialization() {
    for (field, policy) in [
        (
            "device_budget_bytes",
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 0,
                max_source_chunk_bytes: 512 * 1024 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            },
        ),
        (
            "max_source_chunk_bytes",
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 4 * 1024 * 1024,
                max_source_chunk_bytes: 0,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            },
        ),
        (
            "max_row_groups_per_chunk",
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 4 * 1024 * 1024,
                max_source_chunk_bytes: 512 * 1024,
                max_row_groups_per_chunk: 0,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            },
        ),
        (
            "max_retry_attempts",
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 4 * 1024 * 1024,
                max_source_chunk_bytes: 512 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 0,
                min_retry_chunk_row_groups: 1,
            },
        ),
        (
            "min_retry_chunk_row_groups",
            MemoryControlPolicy::Bounded {
                device_budget_bytes: 4 * 1024 * 1024,
                max_source_chunk_bytes: 512 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 0,
            },
        ),
    ] {
        let error = ExecutionContext::builder().with_memory_control_policy(policy).build().unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::InvalidPlan, "{field}");
        assert_eq!(error.operation(), "nexus_query_engine.memory_policy.validate", "{field}");
        assert_eq!(error.fact_value("policy"), Some("bounded"), "{field}");
        assert_eq!(error.fact_value("field"), Some(field), "{field}");
        assert_eq!(error.fact_value("reason"), Some("zero"), "{field}");
    }
}

#[test]
fn durable_hash_join_boundary_requires_explicit_cache() {
    let policy =
        ExecutionBoundaryPolicy::default().with_build_materialization(BuildMaterializationBoundary::DurableGpuHashJoin);

    let error = ExecutionContext::builder().with_boundary_policy(policy).build().unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "nexus_query_engine.boundary_policy.validate");
    assert_eq!(error.fact_value("build_materialization_boundary"), Some("durable_gpu_hash_join"));
    assert_eq!(error.fact_value("reason"), Some("missing_hash_join_build_cache"));
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
    assert_eq!(error.operation(), "nexus_query_engine.source_chunk.policy");
    assert_eq!(error.fact_value("field"), Some("max_row_groups_per_chunk"));
    assert_eq!(error.fact_value("reason"), Some("zero"));
}

#[test]
fn retry_facts_attach_stable_resume_and_merge_context() {
    let request = MemoryReservationRequest::new(ReservationKind::FinalMerge, 2048, 1024);
    let error = RetryFacts::new()
        .with_retry_action("reduce_merge_fan_in")
        .with_resume_boundary("final_merge")
        .with_committed_chunks(8)
        .with_committed_partials(8)
        .with_merge_level(1)
        .with_merge_fan_in(2)
        .attach_to_error(request.denied_error(1024));

    assert_eq!(error.fact_value("retry_action"), Some("reduce_merge_fan_in"));
    assert_eq!(error.fact_value("resume_boundary"), Some("final_merge"));
    assert_eq!(error.fact_value("committed_chunks"), Some("8"));
    assert_eq!(error.fact_value("committed_partials"), Some("8"));
    assert_eq!(error.fact_value("merge_level"), Some("1"));
    assert_eq!(error.fact_value("merge_fan_in"), Some("2"));
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
            output_reservation_bytes: 0,
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

    metrics.record_hash_join_cache_miss();
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
    metrics.record_hash_join_cache_hit();
    metrics.record_hash_join_durable_cache_miss();
    metrics.record_hash_join_durable_cache_insert();
    metrics.record_hash_join_durable_cache_hit();
    assert_eq!(metrics.hash_join_cache_misses, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.filtered_join_build_count, 1);
    assert_eq!(metrics.one_shot_join_count, 1);
    assert_eq!(metrics.hash_join_cache_hits, 1);
    assert_eq!(metrics.hash_join_durable_cache_misses, 1);
    assert_eq!(metrics.hash_join_durable_cache_inserts, 1);
    assert_eq!(metrics.hash_join_durable_cache_hits, 1);
    assert_eq!(metrics.hash_join_hash_table_bytes, 4096);
    assert_eq!(metrics.filtered_join_hash_table_bytes, 2048);
    assert_eq!(metrics.filtered_join_probe_state_bytes, 1024);
    assert_eq!(metrics.mixed_join_size_data_bytes, 512);
    assert_eq!(metrics.one_shot_join_index_bytes, 128);
    assert_eq!(metrics.one_shot_join_materialization_bytes, 256);
    assert_eq!(metrics.one_shot_join_output_bytes, 384);
}

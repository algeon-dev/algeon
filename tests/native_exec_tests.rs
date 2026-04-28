use std::{
    fs,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use arrow::{
    array::{Array, Int32Array, StringArray, StringViewArray},
    compute::SortOptions as ArrowSortOptions,
};
use datafusion::{
    common::{DataFusionError, ScalarValue, config::ConfigOptions},
    datasource::{memory::MemorySourceConfig, source::DataSourceExec},
    execution::{context::TaskContext, object_store::ObjectStoreUrl},
    logical_expr::Operator,
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        expressions::{LikeExpr, col, lit},
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, collect,
        coop::CooperativeExec,
        displayable,
        execution_plan::{Boundedness, EmissionType},
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
};
use datafusion_nexus::{
    backend::{GpuMemoryOwnership, NexusGpuBackend, NexusMemoryPolicy},
    capability::CapabilityReason,
    error::{DataFusionErrorExt, ErrorCode},
    native::lower_physical_plan_to_native,
    native_exec::{NexusNativeComposedExec, NexusNativeExec, NexusNativeExecutionConfig, NexusNativeGpuProducerExec},
    native_report::{NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport},
    planner::{
        FinalPlanDisposition, NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME, NexusFinalPlanRequirementRule,
        NexusNativeOptimizerConfig, NexusNativePhysicalOptimizerRule,
        STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON, classify_final_plan,
    },
    report::CandidateOutcome,
    session::nexus_native_physical_optimizer_rules,
    session_config::DataFusionNexusSessionConfig,
};
use nexus_query_engine::{
    capability::ContractQuality,
    chunk::SourceChunkPlanningOptions,
    mode::{NativePipelineExecutionMode, StrictBoundedExecutionMode},
    pipeline::{PipelineTaskKind, compile_pipeline},
    plan::{NativePlanCostOptions, NativePlanCostReason},
    runtime::{
        BoundedJoinReservationContract, ExecutionCancellation, LakehouseCacheFillConfig, MemoryControlPolicy,
        ReservationKind, SourceChunkPlanningMode,
    },
};
use nexus_workloads::tpch::{TPCH_QUERIES, write_tiny_tpch_parquet_dataset};
use proptest::prelude::*;

#[path = "native_exec_tests/bounded_join.rs"]
mod bounded_join;
#[path = "native_fallback_pressure_tests/mixed_fallback.rs"]
mod mixed_fallback;
#[path = "native_exec_tests/support.rs"]
mod support;
#[path = "support/tpch_session.rs"]
mod tpch_session;

use support::*;

fn test_gpu_backend() -> &'static NexusGpuBackend {
    static BACKEND: OnceLock<NexusGpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        NexusGpuBackend::builder()
            .memory_policy(NexusMemoryPolicy::Bounded {
                device_ordinals: vec![0],
                device_budget_bytes: 256 * 1024 * 1024,
                max_source_chunk_bytes: 64 * 1024 * 1024,
                max_row_groups_per_chunk: 2,
                max_retry_attempts: 3,
                min_retry_chunk_row_groups: 1,
            })
            .device_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)])
            .build()
            .expect("native execution tests require a backend-owned GPU")
    })
}

fn admitted_execution_config(config: NexusNativeExecutionConfig) -> NexusNativeExecutionConfig {
    let backend = test_gpu_backend();
    let handle = backend.admit_query().expect("test query must be admitted by the backend");
    config.with_query_service_handle(Arc::new(handle))
}

fn bounded_config() -> NexusNativeExecutionConfig {
    admitted_execution_config(support::bounded_config())
}

fn native_optimizer_config() -> NexusNativeOptimizerConfig {
    support::native_optimizer_config().with_native_execution_config(bounded_config())
}

#[test]
fn native_exec_constructs_host_arrow_leaf_with_conservative_properties() {
    let native_plan = host_sink_plan();
    let schema = Arc::clone(native_plan.output_schema());
    let exec = NexusNativeExec::try_new(native_plan, bounded_config()).unwrap();

    assert_eq!(exec.name(), "NexusNativeExec");
    assert!(exec.children().is_empty());
    assert_eq!(exec.schema(), schema);
    assert_eq!(exec.properties().output_partitioning().partition_count(), 1);
    assert_eq!(exec.properties().boundedness, Boundedness::Bounded);
    assert!(exec.properties().output_ordering().is_none());

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("NexusNativeExec"));
    assert!(display.contains("sink=host_arrow"));
    assert!(display.contains("memory_control_policy=bounded"));
    assert!(display.contains("boundary_policy=source_read=direct_device_local_parquet"));
    assert!(display.contains("edge_normalize_inserted=false"));

    let tree = displayable(&exec).indent(false).to_string();
    assert!(tree.contains("native_nodes=1"));
    assert!(tree.contains("boundary_policy=source_read=direct_device_local_parquet"));
    assert!(tree.contains("output_boundary=device_to_host"));
    assert!(tree.contains("edge_normalize_inserted=false"));

    let verbose = displayable(&exec).indent(true).to_string();
    assert!(verbose.contains("native_plan=QueryPlan"), "{verbose}");
    assert!(verbose.contains("LocalParquetSource"), "{verbose}");
}

#[test]
fn native_exec_display_records_selected_adaptive_pipeline_mode() {
    let native_plan = host_sink_plan();
    let config = bounded_config().with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive);
    let exec = NexusNativeExec::try_new(native_plan, config).unwrap();

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("native_pipeline_execution_mode=adaptive"));
}

#[test]
fn native_exec_rejects_gpu_resident_sink_for_datafusion_output() {
    let error = NexusNativeExec::try_new(gpu_resident_plan(), bounded_config()).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("result_sink"), Some("gpu_resident"));
}

#[test]
fn native_exec_accepts_unbounded_memory_control_policy() {
    // Unbounded mode remains part of the public API for explicit diagnostic and
    // regression coverage even though production defaults are bounded.
    let exec = NexusNativeExec::try_new(host_sink_plan(), NexusNativeExecutionConfig::unbounded()).unwrap();

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("NexusNativeExec"));
    assert!(display.contains("memory_control_policy=unbounded"));
}

#[test]
fn bounded_auto_sized_uses_bounded_defaults_and_prefetch_scheduler() {
    let config = NexusNativeExecutionConfig::bounded_auto_sized();

    let MemoryControlPolicy::Bounded {
        device_budget_bytes,
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts,
        min_retry_chunk_row_groups,
    } = config.memory_control_policy()
    else {
        panic!("bounded_auto_sized must create a bounded memory-control policy");
    };
    assert!(device_budget_bytes > 0);
    assert_eq!(max_source_chunk_bytes, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(max_row_groups_per_chunk, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(max_retry_attempts, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS);
    assert_eq!(min_retry_chunk_row_groups, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS);
    assert_eq!(config.source_scheduler_policy().max_source_reads_in_flight(), 2);
    assert_eq!(config.source_chunk_planning_mode(), SourceChunkPlanningMode::Conservative);
    assert_eq!(config.lakehouse_cache_fill_config().acquire_timeout, LakehouseCacheFillConfig::DEFAULT_ACQUIRE_TIMEOUT);
    assert_eq!(
        config
            .clone()
            .with_lakehouse_cache_fill_config(LakehouseCacheFillConfig {
                acquire_timeout: Duration::from_secs(11),
                ..Default::default()
            })
            .lakehouse_cache_fill_config()
            .acquire_timeout,
        Duration::from_secs(11)
    );
    assert_eq!(
        config
            .with_source_chunk_planning_mode(SourceChunkPlanningMode::LocalParquetThroughput)
            .source_chunk_planning_mode(),
        SourceChunkPlanningMode::LocalParquetThroughput
    );
    let source_chunk_planning_options =
        SourceChunkPlanningOptions::default().with_local_parquet_throughput_max_row_groups_per_chunk(2048);
    assert_eq!(
        NexusNativeExecutionConfig::bounded_auto_sized()
            .with_source_chunk_planning_options(source_chunk_planning_options)
            .source_chunk_planning_options(),
        source_chunk_planning_options
    );
    assert!(!NexusNativeExecutionConfig::bounded_auto_sized().host_output_memory_pool_accounting());
    assert!(
        NexusNativeExecutionConfig::bounded_auto_sized()
            .with_host_output_memory_pool_accounting(true)
            .host_output_memory_pool_accounting()
    );
}

#[test]
fn optimizer_config_default_uses_bounded_native_execution() {
    let config = NexusNativeOptimizerConfig::default();

    assert!(config.native_execution_config().memory_control_policy().is_bounded());
    assert_eq!(config.native_execution_config().source_scheduler_policy().max_source_reads_in_flight(), 2);
}

#[test]
fn native_composed_exec_hands_gpu_result_between_fragments_without_host_materialization() {
    let dir = unique_temp_dir("datafusion-nexus-native-composed-gpu-resident");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let schema = parquet_file_schema();
    let config = admitted_execution_config(NexusNativeExecutionConfig::unbounded());

    let producer_plan = local_parquet_gpu_resident_plan(&path, Arc::clone(&schema));
    let producer = Arc::new(NexusNativeGpuProducerExec::try_new("fragment_input", producer_plan).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let consumer_plan = gpu_resident_host_sink_plan("fragment_input", Arc::clone(&schema));
    let composed = Arc::new(NexusNativeComposedExec::try_new(consumer_plan, vec![producer_child], config).unwrap());

    assert_eq!(composed.name(), "NexusNativeComposedExec");
    assert_eq!(composed.children().len(), 1);
    let display = displayable(composed.as_ref()).one_line().to_string();
    assert!(display.contains("input_boundary=gpu_resident"));
    assert!(display.contains("output_boundary=device_to_host"));

    let batches = futures::executor::block_on(collect(
        Arc::clone(&composed) as Arc<dyn ExecutionPlan>,
        Arc::new(TaskContext::default()),
    ))
    .unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 4);
    assert_eq!(int32_array(&batches[0], 0).values().as_ref(), &[1, 2, 1]);

    let producer_metrics = producer.metrics().unwrap();
    assert_eq!(producer_metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 0);
    let composed_metrics = composed.metrics().unwrap();
    assert_eq!(composed_metrics.sum_by_name("gpu_resident_input_handoffs").unwrap().as_usize(), 1);
    assert_eq!(composed_metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);
    assert_eq!(composed_metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_composed_host_output_memory_pool_accounting_fails_fast() {
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;

    let dir = unique_temp_dir("datafusion-nexus-native-composed-host-output-memory-pool");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let schema = parquet_file_schema();
    let config = admitted_execution_config(
        NexusNativeExecutionConfig::unbounded().with_host_output_memory_pool_accounting(true),
    );

    let producer_plan = local_parquet_gpu_resident_plan(&path, Arc::clone(&schema));
    let producer = Arc::new(NexusNativeGpuProducerExec::try_new("fragment_input", producer_plan).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let consumer_plan = gpu_resident_host_sink_plan("fragment_input", Arc::clone(&schema));
    let composed = Arc::new(NexusNativeComposedExec::try_new(consumer_plan, vec![producer_child], config).unwrap());
    // The four source columns contain only 72 logical cuDF payload bytes, but
    // their four Arrow destination buffers retain 4 * 64 = 256 bytes. A
    // 128-byte pool therefore proves admission uses the destination estimate:
    // source-payload accounting would incorrectly allow the D2H copy.
    let runtime_env = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(128, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime_env)));

    let error = futures::executor::block_on(collect(Arc::clone(&composed) as Arc<dyn ExecutionPlan>, task_context))
        .expect_err("composed native host output should be charged to DataFusion's bounded memory pool");

    assert!(
        matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
        "expected ResourcesExhausted from composed host-output accounting, got {error:?}"
    );
    assert!(
        error.to_string().contains("NexusNativeComposedExec[host-output]"),
        "error should identify the composed native host-output memory consumer: {error}"
    );
    assert_eq!(
        composed.metrics().unwrap().sum_by_name("device_to_host_materializations").unwrap().as_usize(),
        0,
        "host memory reservation denial must happen before D2H"
    );
    assert_eq!(runtime_env.memory_pool.reserved(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_host_output_memory_pool_accounting_fails_fast_in_hybrid_plan() {
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;

    let dir = unique_temp_dir("datafusion-nexus-native-host-output-memory-pool");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);

    let config = admitted_execution_config(
        NexusNativeExecutionConfig::unbounded().with_host_output_memory_pool_accounting(true),
    );
    let native_plan = local_parquet_host_sink_plan(&path, parquet_file_schema());
    let native_child = Arc::new(NexusNativeExec::try_new(native_plan, config).unwrap()) as Arc<dyn ExecutionPlan>;
    let native_schema = native_child.schema();
    let hybrid_plan = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("key", &native_schema).unwrap(), alias: "key".to_string() }],
            native_child,
        )
        .unwrap(),
    ) as Arc<dyn ExecutionPlan>;
    // Source payload is 72 bytes; final Arrow buffer capacity is 256 bytes.
    // The limit intentionally lies between them to lock pre-D2H destination
    // accounting rather than source-payload accounting.
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(128, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));

    let error = futures::executor::block_on(collect(Arc::clone(&hybrid_plan), task_context))
        .expect_err("native host output should be charged to the bounded DataFusion memory pool");

    assert!(
        matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
        "expected ResourcesExhausted from DataFusion MemoryPool, got {error:?}"
    );
    assert!(
        error.to_string().contains("NexusNativeExec[host-output]"),
        "error should identify the native host-output memory consumer: {error}"
    );
    assert_eq!(runtime.memory_pool.reserved(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_host_output_memory_pool_accounting_releases_on_stream_eof() {
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use futures::StreamExt;

    let dir = unique_temp_dir("datafusion-nexus-native-host-output-memory-pool-eof");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);

    let config = admitted_execution_config(
        NexusNativeExecutionConfig::unbounded().with_host_output_memory_pool_accounting(true),
    );
    let native_plan = local_parquet_host_sink_plan(&path, parquet_file_schema());
    let native_exec = NexusNativeExec::try_new(native_plan, config).unwrap();
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let mut stream = native_exec.execute(0, task_context).unwrap();

    let batch = futures::executor::block_on(stream.next()).expect("native stream should emit one batch").unwrap();
    assert_eq!(batch.num_rows(), 3);
    assert!(runtime.memory_pool.reserved() > 0);

    let eof = futures::executor::block_on(stream.next());
    assert!(eof.is_none());
    assert_eq!(runtime.memory_pool.reserved(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_composes_feature_neutral_gpu_resident_producer_child() {
    let producer = Arc::new(NexusNativeGpuProducerExec::try_new("fragment_input", gpu_resident_plan()).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let child_schema = producer_child.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("key", &child_schema).unwrap(), alias: "key".to_string() }],
            producer_child,
        )
        .unwrap(),
    );
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let composed = optimized
        .downcast_ref::<NexusNativeComposedExec>()
        .expect("GPU-resident producer children should lower through composed native execution");
    assert_eq!(composed.children().len(), 1);
    assert!(composed.children()[0].downcast_ref::<NexusNativeGpuProducerExec>().is_some());
    assert!(composed.native_plan().to_string().contains("GpuResidentSource input_name=fragment_input"));
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
}

#[test]
fn native_optimizer_composed_exec_preserves_root_sort_ordering_properties() {
    let producer = Arc::new(NexusNativeGpuProducerExec::try_new("fragment_input", gpu_resident_plan()).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let child_schema = producer_child.schema();
    let ordering = LexOrdering::new([PhysicalSortExpr::new(
        col("key", &child_schema).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .expect("root sort ordering should be non-empty");
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(ordering, producer_child));
    let expected_ordering = plan.properties().output_ordering().cloned();
    assert!(expected_ordering.is_some());
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NexusNativeComposedExec>().is_some());
    assert_eq!(optimized.properties().output_ordering().cloned(), expected_ordering);
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
}

#[test]
fn native_optimizer_rule_rewrites_supported_physical_plan() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-optimizer");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NexusNativeExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert!(report.query_id().is_some_and(|query_id| query_id.as_str().starts_with("q-")));
    match &report.outcomes()[0] {
        CandidateOutcome::Selected { candidate } => {
            assert!(candidate.plan_node_id.is_some(), "rewritten candidate should carry native plan-node evidence");
        },
        other => panic!("expected rewritten candidate, got {other:?}"),
    }
    assert_eq!(rule.last_source_diagnostics().len(), 1);
    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain diagnostics");
    assert!(diagnostics.strict_bounded_contract.accepted_candidates.is_empty());
    assert!(diagnostics.strict_bounded_contract.rejected_candidates.is_empty());
    assert_eq!(diagnostics.strict_bounded_contract.non_join_bounded_plan_count, 1);
    assert_eq!(
        diagnostics.strict_bounded_contract.non_join_bounded_plan_reason.as_deref(),
        Some(STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON)
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_is_mixed_and_idempotent() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-property-checked-ilike");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let plan = ilike_filter_plan(&path);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let report = NativeDataFusionPlanningReport::from_retained_rule("ilike_boundary", Arc::clone(&optimized), &rule);

    assert_eq!(report.rows.len(), 1);
    let row = &report.rows[0];
    assert_eq!(row.outcome, Some(NativeDataFusionPlanningOutcome::Selected));
    assert_eq!(row.final_disposition, FinalPlanDisposition::Mixed);
    assert_eq!(row.native_plan_fragment_count, 1);
    assert_eq!(row.host_datafusion_boundary_count, 1);
    assert_eq!(row.source_diagnostics.len(), 1);
    assert!(row.source_diagnostics[0].direct_device_read);

    let filter = optimized.downcast_ref::<FilterExec>().expect("P2a boundary must retain the CPU root filter");
    let native = filter
        .input()
        .downcast_ref::<NexusNativeExec>()
        .expect("P2a boundary must replace only the local-Parquet leaf");
    assert_p2_mixed_properties(filter, native);

    let report_before_repeat = rule.last_report().expect("selected P2a boundary must retain a report");
    let repeated = rule.optimize(Arc::clone(&optimized), &ConfigOptions::default()).unwrap();
    assert!(Arc::ptr_eq(&optimized, &repeated), "P2a mixed boundary must be an optimizer fixed point");
    assert_eq!(rule.last_report(), Some(report_before_repeat), "idempotence must preserve the selected-leaf report");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_allows_transparent_cooperative_wrapper() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-property-checked-ilike-cooperative");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(ilike_filter_plan(&path)));
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let cooperative = optimized
        .downcast_ref::<CooperativeExec>()
        .expect("the certified transparent wrapper should remain in the final plan");
    let filter = cooperative
        .input()
        .downcast_ref::<FilterExec>()
        .expect("cooperative wrapper must still contain the root filter");
    let native = filter.input().downcast_ref::<NexusNativeExec>().expect("filter child must be the one native leaf");
    assert_p2_mixed_properties(filter, native);
    assert_eq!(optimized.properties().emission_type, EmissionType::Final);
    assert_eq!(classify_final_plan(optimized.as_ref()), FinalPlanDisposition::Mixed);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_rejects_nonlocal_parquet() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-property-checked-ilike-nonlocal");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let plan = ilike_filter_over_scan(parquet_scan_with_object_store_url(
        &path,
        ObjectStoreUrl::parse("s3://p2-boundary-test").unwrap(),
    ));
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(
        Arc::ptr_eq(&optimized, &plan),
        "the P2a certificate must not replace a remote Parquet source through its local-only boundary"
    );
    let report = rule.last_report().expect("native optimizer should record the whole-plan rejection");
    assert_eq!(report.selected_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_is_rejected_by_no_datafusion_cpu() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-property-checked-ilike-no-datafusion-cpu");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let config = NexusNativeOptimizerConfig::default()
        .with_no_datafusion_cpu_requirement()
        .with_native_execution_config(bounded_config());
    let native_rule = NexusNativePhysicalOptimizerRule::new(config.clone());
    let requirement_rule = NexusFinalPlanRequirementRule::new(config);

    let planned = native_rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let report = NativeDataFusionPlanningReport::from_retained_rule("ilike_no_cpu", Arc::clone(&planned), &native_rule);
    assert_eq!(report.rows[0].final_disposition, FinalPlanDisposition::Mixed);
    assert_eq!(report.rows[0].native_plan_fragment_count, 1);
    assert_eq!(report.rows[0].host_datafusion_boundary_count, 1);

    let error = requirement_rule
        .optimize(planned, &ConfigOptions::default())
        .expect_err("NoDataFusionCpu must reject the exact selected mixed plan");
    let adapter_error = adapter_error(&error);
    assert_eq!(adapter_error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(adapter_error.fact_value("requirement"), Some("no_datafusion_cpu"));
    assert_eq!(adapter_error.fact_value("observed_final_plan_disposition"), Some("mixed"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_matches_cpu_results() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-property-checked-ilike-parity");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);

    let cpu = futures::executor::block_on(collect(ilike_filter_plan(&path), Arc::new(TaskContext::default()))).unwrap();
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());
    let native_plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let native_leaf = p2_native_leaf(native_plan.as_ref());
    let native =
        futures::executor::block_on(collect(Arc::clone(&native_plan), Arc::new(TaskContext::default()))).unwrap();

    assert_eq!(ilike_rows(&native), ilike_rows(&cpu));
    let metrics = native_leaf.metrics().expect("the native leaf owns runtime metrics for the one D2H boundary");
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("host_to_device_uploads").unwrap().as_usize(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_reserves_host_output_before_d2h() {
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;

    let dir = unique_temp_dir("datafusion-nexus-native-exec-property-checked-ilike-host-budget");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let backend = test_gpu_backend();
    let handle = backend.admit_query().expect("test query must be admitted by the backend");
    let config = NexusNativeOptimizerConfig::default().with_native_execution_config(
        support::bounded_config()
            .with_host_output_memory_pool_accounting(true)
            .with_query_service_handle(Arc::new(handle)),
    );
    let rule = NexusNativePhysicalOptimizerRule::new(config);
    let plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let runtime_env = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime_env)));

    let error = futures::executor::block_on(collect(Arc::clone(&plan), task_context))
        .expect_err("the optimizer-built P2a boundary must reserve its host Arrow output before D2H");

    assert!(matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)), "unexpected error: {error:?}");
    assert!(error.to_string().contains("NexusNativeExec[host-output]"), "unexpected error: {error}");
    assert_eq!(
        p2_native_leaf(plan.as_ref())
            .metrics()
            .unwrap()
            .sum_by_name("device_to_host_materializations")
            .unwrap()
            .as_usize(),
        0,
        "host-memory denial must occur before the P2a native leaf copies data from device to host"
    );
    assert_eq!(runtime_env.memory_pool.reserved(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_honors_pre_execution_cancellation() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-property-checked-ilike-cancelled");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let cancellation = ExecutionCancellation::new();
    cancellation.cancel();
    let config = NexusNativeOptimizerConfig::default()
        .with_native_execution_config(bounded_config().with_cancellation(cancellation));
    let rule = NexusNativePhysicalOptimizerRule::new(config);
    let plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();

    let error = futures::executor::block_on(collect(plan, Arc::new(TaskContext::default())))
        .expect_err("the P2a native leaf must observe cancellation before it starts native execution");
    assert_eq!(adapter_error(&error).code(), ErrorCode::Cancelled);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_uses_one_backend_admission_and_observation_owner() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-property-checked-ilike-admission");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let backend = test_gpu_backend();
    let rule = NexusNativePhysicalOptimizerRule::new(backend.optimizer_config().clone());
    let plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let native_leaf = p2_native_leaf(plan.as_ref());

    let batches = futures::executor::block_on(collect(Arc::clone(&plan), Arc::new(TaskContext::default()))).unwrap();

    assert_eq!(ilike_rows(&batches), vec![(1, Some("Alpha".to_string())), (3, Some("ALMANAC".to_string()))]);
    let metrics =
        native_leaf.metrics().expect("the selected native leaf must expose the shared observation projection");
    assert_eq!(metrics.sum_by_name("admission_sequence_observed").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("admission_wait_time_observed").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("query_grant_bytes_observed").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    let deadline = Instant::now() + Duration::from_secs(5);
    while backend.admission_snapshot().unwrap().active_queries != 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let snapshot = backend.admission_snapshot().unwrap();
    assert_eq!(snapshot.active_queries, 0, "the one P2a admission ticket must release after its stream completes");
    assert_eq!(snapshot.queued_queries, 0);

    fs::remove_dir_all(dir).unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 16,
        .. ProptestConfig::default()
    })]

    /// Generates only the P2a-certified family. This proves the invariant
    /// across nullable UTF-8 view inputs and deterministic ASCII prefixes
    /// without broadening the root-only local-Parquet certificate.
    #[test]
    fn native_optimizer_property_based_ilike_boundary_preserves_contract(
        mut labels in prop::collection::vec(prop::option::of("[A-Za-z]{0,12}"), 1..16),
        prefix in "[A-Za-z]{0,5}",
    ) {
        // Retain a null in every generated P2a fixture while still varying
        // nullable input values in the generated rows.
        labels.push(None);
        let dir = unique_temp_dir("datafusion-nexus-native-exec-property-based-ilike");
        let path = dir.join("input.parquet");
        write_ilike_parquet_file_with_labels(&path, &labels);
        let pattern = format!("{prefix}%");

        let cpu = futures::executor::block_on(collect(
            ilike_filter_plan_with_pattern(&path, pattern.clone()),
            Arc::new(TaskContext::default()),
        ))
        .unwrap();
        let config = native_optimizer_config().with_no_datafusion_cpu_requirement();
        let native_rule = NexusNativePhysicalOptimizerRule::new(config.clone());
        let planned = native_rule
            .optimize(ilike_filter_plan_with_pattern(&path, pattern), &ConfigOptions::default())
            .unwrap();
        let report = NativeDataFusionPlanningReport::from_retained_rule(
            "generated_ilike_boundary",
            Arc::clone(&planned),
            &native_rule,
        );

        prop_assert_eq!(report.rows.len(), 1);
        let row = &report.rows[0];
        prop_assert_eq!(row.outcome, Some(NativeDataFusionPlanningOutcome::Selected));
        prop_assert_eq!(row.final_disposition, FinalPlanDisposition::Mixed);
        prop_assert_eq!(row.native_plan_fragment_count, 1);
        prop_assert_eq!(row.host_datafusion_boundary_count, 1);
        let filter = planned.downcast_ref::<FilterExec>().expect("P2a must retain the CPU root filter");
        let native = filter.input().downcast_ref::<NexusNativeExec>().expect("P2a must replace only the source leaf");
        assert_p2_mixed_properties(filter, native);

        let mixed = futures::executor::block_on(collect(Arc::clone(&planned), Arc::new(TaskContext::default()))).unwrap();
        prop_assert_eq!(ilike_rows(&mixed), ilike_rows(&cpu));

        let retained_report = native_rule.last_report().expect("P2a selected report should be retained");
        let repeated = native_rule.optimize(Arc::clone(&planned), &ConfigOptions::default()).unwrap();
        prop_assert!(Arc::ptr_eq(&planned, &repeated));
        prop_assert_eq!(native_rule.last_report(), Some(retained_report));

        let requirement_rule = NexusFinalPlanRequirementRule::new(config);
        let error = requirement_rule
            .optimize(planned, &ConfigOptions::default())
            .expect_err("NoDataFusionCpu must reject the exact generated mixed plan");
        let error = adapter_error(&error);
        prop_assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        prop_assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("mixed"));

        fs::remove_dir_all(dir).unwrap();
    }
}

fn ilike_filter_plan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    ilike_filter_plan_with_pattern(path, "al%".to_string())
}

fn ilike_filter_plan_with_pattern(path: &std::path::Path, pattern: String) -> Arc<dyn ExecutionPlan> {
    ilike_filter_over_scan_with_pattern(parquet_scan(path), pattern)
}

fn ilike_filter_over_scan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    ilike_filter_over_scan_with_pattern(scan, "al%".to_string())
}

fn ilike_filter_over_scan_with_pattern(scan: Arc<dyn ExecutionPlan>, pattern: String) -> Arc<dyn ExecutionPlan> {
    let schema = scan.schema();
    let predicate =
        Arc::new(LikeExpr::new(false, true, col("label", &schema).unwrap(), lit(ScalarValue::Utf8View(Some(pattern)))));
    Arc::new(FilterExec::try_new(predicate, scan).unwrap())
}

fn p2_native_leaf(plan: &dyn ExecutionPlan) -> &NexusNativeExec {
    let filter = plan.downcast_ref::<FilterExec>().expect("P2a plan must retain the CPU root filter");
    filter.input().downcast_ref::<NexusNativeExec>().expect("P2a filter input must be the native leaf")
}

fn assert_p2_mixed_properties(filter: &FilterExec, native: &NexusNativeExec) {
    assert_eq!(filter.schema().as_ref(), native.schema().as_ref());
    assert!(matches!(
        filter.properties().output_partitioning(),
        datafusion::physical_expr::Partitioning::UnknownPartitioning(1)
    ));
    assert_eq!(filter.properties().emission_type, EmissionType::Final);
    assert_eq!(filter.properties().boundedness, Boundedness::Bounded);
    assert!(filter.properties().output_ordering().is_none());
    assert!(filter.properties().equivalence_properties().eq_group().is_empty());
    assert!(filter.properties().equivalence_properties().oeq_class().is_empty());
    assert!(filter.properties().equivalence_properties().constraints().is_empty());

    assert!(matches!(
        native.properties().output_partitioning(),
        datafusion::physical_expr::Partitioning::UnknownPartitioning(1)
    ));
    assert_eq!(native.properties().emission_type, EmissionType::Final);
    assert_eq!(native.properties().boundedness, Boundedness::Bounded);
    assert!(native.properties().output_ordering().is_none());
    assert!(native.properties().equivalence_properties().eq_group().is_empty());
    assert!(native.properties().equivalence_properties().oeq_class().is_empty());
    assert!(native.properties().equivalence_properties().constraints().is_empty());
    assert_eq!(native.native_plan().nodes().len(), 1);
}

fn ilike_rows(batches: &[arrow::record_batch::RecordBatch]) -> Vec<(i32, Option<String>)> {
    let mut rows = batches
        .iter()
        .flat_map(|batch| {
            let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let labels = batch.column(1);
            let label_at = |row| {
                if let Some(labels) = labels.as_any().downcast_ref::<StringArray>() {
                    return (!labels.is_null(row)).then(|| labels.value(row).to_string());
                }
                let labels = labels.as_any().downcast_ref::<StringViewArray>().unwrap();
                (!labels.is_null(row)).then(|| labels.value(row).to_string())
            };
            (0..batch.num_rows()).map(move |row| (ids.value(row), label_at(row)))
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

#[test]
fn native_optimizer_rule_propagates_declared_scan_output_ordering_to_wrapper_properties() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-ordered-scan-properties");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let plan = parquet_scan_with_declared_output_ordering(&path);
    let expected_ordering = plan.properties().output_ordering().cloned();
    assert!(expected_ordering.is_some());
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NexusNativeExec>().is_some());
    assert_eq!(optimized.properties().output_ordering().cloned(), expected_ordering);
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_reestablishes_declared_scan_ordering_through_aggregate() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-ordered-aggregate-composition");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let plan = grouped_sum_aggregate_plan(scan);
    let expected_ordering = plan.properties().output_ordering().cloned();
    assert!(expected_ordering.is_some());
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let native = optimized
        .downcast_ref::<NexusNativeExec>()
        .expect("ordered aggregate should lower with an explicit final native sort");
    assert!(matches!(native.native_plan().root_node().kind, nexus_query_engine::plan::PlanNode::Sort(_)));
    assert_eq!(optimized.properties().output_ordering().cloned(), expected_ordering);
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1, "outcomes={:?}", report.outcomes());
    let CandidateOutcome::Selected { candidate } = &report.outcomes()[0] else {
        panic!("ordered aggregate should be reported as rewritten");
    };
    assert!(
        candidate.shape.starts_with("SortExec(AggregateExec("),
        "post-optimizer appended declared-ordering sort should be visible in candidate shape: {}",
        candidate.shape
    );
    assert_eq!(
        candidate.plan_node_id.as_ref().map(ToString::to_string),
        Some(native.native_plan().root_id().to_string())
    );
    assert!(!report.outcomes().iter().any(|outcome| {
        matches!(
            outcome,
            CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeOrderedOutputUnsupported, .. }
        )
    }));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_execution_reestablished_declared_scan_ordering_is_sorted() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-ordered-aggregate-output");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let plan = grouped_sum_aggregate_plan(scan);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let native = optimized
        .downcast_ref::<NexusNativeExec>()
        .expect("ordered aggregate should lower with an explicit final native sort");
    assert!(matches!(native.native_plan().root_node().kind, nexus_query_engine::plan::PlanNode::Sort(_)));
    let batches =
        futures::executor::block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(int32_array(&batches[0], 0).values().as_ref(), &[1, 2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_execution_sort_preserving_merge_matches_datafusion_sorted_input() {
    let scan = sorted_two_partition_memory_source();
    let ordering = LexOrdering::new([PhysicalSortExpr::new(
        col("key", &scan.schema()).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .expect("SPM test ordering should be non-empty");
    let sorted: Arc<dyn ExecutionPlan> =
        Arc::new(SortExec::new(ordering.clone(), scan).with_preserve_partitioning(true));
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortPreservingMergeExec::new(ordering, sorted));
    let expected = futures::executor::block_on(collect(Arc::clone(&plan), Arc::new(TaskContext::default()))).unwrap();
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let native = optimized.downcast_ref::<NexusNativeExec>().expect("SPM over sorted input should lower to native");
    assert!(matches!(native.native_plan().root_node().kind, nexus_query_engine::plan::PlanNode::Sort(_)));
    let actual =
        futures::executor::block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(actual.len(), expected.len());
    assert_eq!(int32_array(&actual[0], 0).values().as_ref(), int32_array(&expected[0], 0).values().as_ref());
    assert_eq!(int32_array(&actual[0], 0).values().as_ref(), &[1, 2, 3, 4]);
}

fn sorted_two_partition_memory_source() -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "key",
        arrow::datatypes::DataType::Int32,
        false,
    )]));
    let first = arrow::record_batch::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow::array::Int32Array::from(vec![1, 3]))],
    )
    .unwrap();
    let second = arrow::record_batch::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow::array::Int32Array::from(vec![2, 4]))],
    )
    .unwrap();
    let ordering =
        LexOrdering::new([PhysicalSortExpr::new(col("key", &schema).unwrap(), ArrowSortOptions::new(false, false))])
            .expect("memory source ordering should be non-empty");
    let source = MemorySourceConfig::try_new(&[vec![first], vec![second]], Arc::clone(&schema), None)
        .unwrap()
        .try_with_sort_information(vec![ordering])
        .unwrap();
    DataSourceExec::from_data_source(source)
}

#[test]
fn native_optimizer_rule_uses_datafusion_config_options_at_optimize_time() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-optimizer-session-config");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    // This is a pure planning-policy override test. It intentionally starts
    // without backend-frozen admission policy so the session option remains
    // caller-configurable.
    let config = support::native_optimizer_config();
    let rule = NexusNativePhysicalOptimizerRule::new(config.clone());
    let mut options = ConfigOptions::default();
    options.extensions.insert(DataFusionNexusSessionConfig::from_optimizer_config(&config));
    options.set("datafusion_nexus.native.memory_policy", "unbounded").unwrap();

    let optimized = rule.optimize(plan, &options).unwrap();

    let native =
        optimized.downcast_ref::<NexusNativeExec>().expect("supported plan should optimize to NexusNativeExec");
    let display = displayable(native).one_line().to_string();
    assert!(display.contains("memory_control_policy=unbounded"), "{display}");
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_preserves_rejected_plan_as_datafusion_baseline() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-optimizer-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<LocalLimitExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 0);
    assert_eq!(report.not_selected_count(), 1);
    assert!(report.query_id().is_some_and(|query_id| query_id.as_str().starts_with("q-")));
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn session_without_native_rule_preserves_datafusion_plan() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-cpu-only");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let optimized = plan;

    assert!(optimized.downcast_ref::<LocalLimitExec>().is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_does_not_select_small_selective_topk_join_by_cost_when_enabled() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-cost-topk");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let config = native_optimizer_config().with_native_plan_cost_options(NativePlanCostOptions::default());
    let rule = NexusNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(
        optimized.downcast_ref::<NexusNativeExec>().is_none(),
        "cost-declined plan should not be replaced by a native execution boundary"
    );
    assert!(Arc::ptr_eq(&optimized, &plan), "a cost-rejected candidate must preserve the exact baseline plan");
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 0);
    assert_eq!(report.not_selected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::NotSelectedByCost { cost_reason: NativePlanCostReason::SmallSelectiveTopK, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_does_not_select_small_selective_full_sort_join_by_cost_when_enabled() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-cost-full-sort");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_sql_plan(&left, &right);
    let config = native_optimizer_config().with_native_plan_cost_options(NativePlanCostOptions::default());
    let rule = NexusNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(
        optimized.downcast_ref::<NexusNativeExec>().is_none(),
        "cost-declined plan should not be replaced by a native execution boundary"
    );
    assert!(Arc::ptr_eq(&optimized, &plan), "a cost-rejected candidate must preserve the exact baseline plan");
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 0);
    assert_eq!(report.not_selected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::NotSelectedByCost { cost_reason: NativePlanCostReason::SmallSelectiveFullSort, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_errors_for_cost_declined_plan_with_no_datafusion_cpu_requirement() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-cost-no-datafusion-cpu");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let config = NexusNativeOptimizerConfig::default()
        .with_no_datafusion_cpu_requirement()
        .with_native_execution_config(bounded_config())
        .with_native_plan_cost_options(NativePlanCostOptions::default());
    let rules = nexus_native_physical_optimizer_rules(config);
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native")
        .expect("native optimizer rules should contain the final native rule");
    let final_requirement = rules
        .iter()
        .find(|rule| rule.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("native optimizer rules should contain the final-plan requirement rule");
    let planned = final_native.optimize(plan, &ConfigOptions::default()).unwrap();
    let error = final_requirement
        .optimize(planned, &ConfigOptions::default())
        .expect_err("the final-plan requirement must reject a cost-declined completed plan");
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(adapter_error.fact_value("requirement"), Some("no_datafusion_cpu"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_obeys_no_datafusion_cpu_requirement_for_rejected_plan() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-no-datafusion-cpu-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let config = NexusNativeOptimizerConfig::default()
        .with_no_datafusion_cpu_requirement()
        .with_native_execution_config(bounded_config());
    let rules = nexus_native_physical_optimizer_rules(config);
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native")
        .expect("native optimizer rules should contain the final native rule");
    let final_requirement = rules
        .iter()
        .find(|rule| rule.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("native optimizer rules should contain the final-plan requirement rule");
    let planned = final_native.optimize(plan, &ConfigOptions::default()).unwrap();
    let error = final_requirement
        .optimize(planned, &ConfigOptions::default())
        .expect_err("the final-plan requirement must reject a rejected native candidate's CPU plan");
    let adapter_error = error.find_nexus_error().expect("final-plan rejection must preserve the typed Nexus error");

    assert_eq!(adapter_error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(adapter_error.fact_value("requirement"), Some("no_datafusion_cpu"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn no_datafusion_cpu_count_only_probe_miss_reaches_final_native_optimizer() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-no-datafusion-cpu-probe-miss");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let config = NexusNativeOptimizerConfig::default()
        .with_no_datafusion_cpu_requirement()
        .with_native_execution_config(bounded_config());
    let rules = nexus_native_physical_optimizer_rules(config);
    let count_only_probe = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native_count_only")
        .expect("native optimizer rules should contain the count-only probe");
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native")
        .expect("native optimizer rules should contain the final native rule");

    let after_probe = count_only_probe
        .optimize(plan, &ConfigOptions::default())
        .expect("a non-count plan must not be rejected by the NoDataFusionCpu count-only probe");
    assert!(
        after_probe.downcast_ref::<NexusNativeExec>().is_none(),
        "the count-only probe should decline an analytical SUM plan"
    );
    let probe_report = (count_only_probe.as_ref() as &dyn std::any::Any)
        .downcast_ref::<NexusNativePhysicalOptimizerRule>()
        .expect("count-only optimizer rule should be native")
        .last_report()
        .expect("count-only probe should record its own rejection evidence");
    assert_eq!(probe_report.selected_count(), 0);
    assert_eq!(probe_report.not_selected_count(), 1);

    let optimized = final_native
        .optimize(after_probe, &ConfigOptions::default())
        .expect("the final native optimizer must run after a count-only probe miss");
    assert!(
        optimized.downcast_ref::<NexusNativeExec>().is_some(),
        "the final native optimizer should rewrite the supported analytical plan"
    );
    let final_report = (final_native.as_ref() as &dyn std::any::Any)
        .downcast_ref::<NexusNativePhysicalOptimizerRule>()
        .expect("final optimizer rule should be native")
        .last_report()
        .expect("final optimizer should record separate native rewrite evidence");
    assert_eq!(final_report.selected_count(), 1);
    assert_eq!(final_report.not_selected_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn no_datafusion_cpu_final_rejection_replaces_count_only_probe_diagnostics_before_error() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-no-datafusion-cpu-final-rejection-evidence");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let rules = nexus_native_physical_optimizer_rules(
        NexusNativeOptimizerConfig::default()
            .with_no_datafusion_cpu_requirement()
            .with_native_execution_config(bounded_config()),
    );
    let count_only_probe = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native_count_only")
        .expect("native optimizer rules should contain the count-only probe");
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native")
        .expect("native optimizer rules should contain the final native rule");

    let after_probe = count_only_probe.optimize(plan, &ConfigOptions::default()).expect(
        "a normal count-only probe miss must remain non-terminal before NoDataFusionCpu final-plan enforcement",
    );
    let probe_report = (count_only_probe.as_ref() as &dyn std::any::Any)
        .downcast_ref::<NexusNativePhysicalOptimizerRule>()
        .expect("count-only optimizer rule should be native")
        .last_report()
        .expect("count-only probe should retain its own evidence");
    assert!(matches!(
        probe_report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedOperator, .. }
    ));

    let planned = final_native
        .optimize(after_probe, &ConfigOptions::default())
        .expect("the final native optimizer must retain its rejected-candidate diagnostics");
    let final_report = (final_native.as_ref() as &dyn std::any::Any)
        .downcast_ref::<NexusNativePhysicalOptimizerRule>()
        .expect("final optimizer rule should be native")
        .last_report()
        .expect("final rejection should replace the probe's retained evidence before returning an error");
    assert!(matches!(
        final_report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));
    let final_requirement = rules
        .iter()
        .find(|rule| rule.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("native optimizer rules should contain the final-plan requirement rule");
    let error = final_requirement
        .optimize(planned, &ConfigOptions::default())
        .expect_err("the final-plan requirement must reject the completed DataFusion plan");
    assert_eq!(adapter_error(&error).code(), ErrorCode::FinalPlanRequirementUnsatisfied);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn no_datafusion_cpu_count_only_probe_retains_its_native_rewrite() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-no-datafusion-cpu-count-probe");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = count_only_sort_limit_plan(parquet_scan(&path));
    let config = NexusNativeOptimizerConfig::default()
        .with_no_datafusion_cpu_requirement()
        .with_native_execution_config(bounded_config());
    let rules = nexus_native_physical_optimizer_rules(config);
    let count_only_probe = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native_count_only")
        .expect("native optimizer rules should contain the count-only probe");
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native")
        .expect("native optimizer rules should contain the final native rule");

    let after_probe = count_only_probe
        .optimize(plan, &ConfigOptions::default())
        .expect("NoDataFusionCpu count-only probe should retain supported native count rewrites");
    assert!(after_probe.downcast_ref::<NexusNativeExec>().is_some());
    let probe_report = (count_only_probe.as_ref() as &dyn std::any::Any)
        .downcast_ref::<NexusNativePhysicalOptimizerRule>()
        .expect("count-only optimizer rule should be native")
        .last_report()
        .expect("count-only probe should record its rewrite evidence");
    assert_eq!(probe_report.selected_count(), 1);
    assert_eq!(probe_report.not_selected_count(), 0);
    let after_final = final_native
        .optimize(after_probe, &ConfigOptions::default())
        .expect("the final native optimizer should preserve a count-only native rewrite");
    assert!(after_final.downcast_ref::<NexusNativeExec>().is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn session_without_native_rule_has_no_native_optimizer_stages() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-cpu-only-count-probe");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let state = datafusion::execution::SessionStateBuilder::new_with_default_features().build();
    let rule_names = state.physical_optimizers().iter().map(|rule| rule.name()).collect::<Vec<_>>();
    assert!(!rule_names.contains(&"datafusion_nexus_native_count_only"));
    assert!(!rule_names.contains(&"datafusion_nexus_native"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_retains_unsupported_relational_parent_as_datafusion() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-partial-host-arrow-boundary");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let child = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let child_schema = child.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("key", &child_schema).unwrap(), alias: "key".to_string() }],
            child,
        )
        .unwrap(),
    );
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(Arc::ptr_eq(&optimized, &plan), "a relational lowering miss must retain the baseline DataFusion plan");
    assert_eq!(classify_final_plan(optimized.as_ref()), FinalPlanDisposition::DataFusion);
    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        !optimized_tree.contains("NexusNativeExec"),
        "ordinary relational DataFusion baseline must not retain a native fragment"
    );
    assert!(
        !optimized_tree.contains("NexusNativeGpuProducerExec"),
        "ordinary relational DataFusion baseline must not producerize a native child"
    );
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 0);
    assert_eq!(report.not_selected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedOperator, .. }
    ));
    assert!(rule.last_source_diagnostics().is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn final_plan_requirement_rejects_unsupported_relational_parent_after_datafusion_disposition() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-composition-no-datafusion-cpu");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let child = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let child_schema = child.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("key", &child_schema).unwrap(), alias: "key".to_string() }],
            child,
        )
        .unwrap(),
    );
    let config = NexusNativeOptimizerConfig::default()
        .with_no_datafusion_cpu_requirement()
        .with_native_execution_config(bounded_config());
    let rules = nexus_native_physical_optimizer_rules(config);
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "datafusion_nexus_native")
        .expect("native optimizer rules should contain the final native rule");
    let final_requirement = rules
        .iter()
        .find(|rule| rule.name() == NEXUS_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("native optimizer rules should contain the final-plan requirement rule");

    let planned = final_native.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(Arc::ptr_eq(&planned, &plan), "a relational lowering miss must remain the original DataFusion plan");
    assert_eq!(classify_final_plan(planned.as_ref()), FinalPlanDisposition::DataFusion);
    let planned_tree = displayable(planned.as_ref()).indent(false).to_string();
    assert!(!planned_tree.contains("NexusNativeGpuProducerExec"));
    let report = (final_native.as_ref() as &dyn std::any::Any)
        .downcast_ref::<NexusNativePhysicalOptimizerRule>()
        .expect("final optimizer rule should be native")
        .last_report()
        .expect("final native rule should record its rejection");
    assert_eq!(report.selected_count(), 0);
    assert_eq!(report.not_selected_count(), 1);

    let error = final_requirement
        .optimize(planned, &ConfigOptions::default())
        .expect_err("the final-plan requirement must reject the completed DataFusion plan");
    let adapter_error = adapter_error(&error);
    assert_eq!(adapter_error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(adapter_error.fact_value("observed_final_plan_disposition"), Some("datafusion"));
    assert_eq!(adapter_error.fact_value("terminal_plan_disposition"), Some("rejected"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_retains_distinct_aggregate_over_anti_join_as_datafusion() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-distinct-anti-composition");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = single_partition_sql_plan_with_parquet_tables(
        &[("left_input", &left), ("right_input", &right)],
        "SELECT COUNT(DISTINCT l.key) AS distinct_keys \
         FROM left_input l \
         WHERE l.key NOT IN (SELECT r.key FROM right_input r WHERE r.r_payload IS NOT NULL)",
    );
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        Arc::ptr_eq(&optimized, &plan),
        "an unsupported relational parent must preserve its original DataFusion plan"
    );
    assert_eq!(classify_final_plan(optimized.as_ref()), FinalPlanDisposition::DataFusion);
    assert!(
        !optimized_tree.contains("NexusNativeExec"),
        "relational DataFusion baseline must not retain a native fragment"
    );
    assert!(
        !optimized_tree.contains("NexusNativeGpuProducerExec"),
        "relational DataFusion baseline must not create a GPU producer"
    );
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 0);
    assert_eq!(report.not_selected_count(), 1);

    let batches =
        futures::executor::block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(int64_array(&batches[0], 0).value(0), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_precision_sensitive_stddev_sort_subtree_after_order_contract() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-stddev-sort-cpu-boundary");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = stddev_sort_sql_plan(&path);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NexusNativeExec>().is_some());
    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        optimized_tree.contains("NexusNativeExec"),
        "stddev subtree should be rewritten once native order tracking is available:\n{optimized_tree}"
    );
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_is_inserted_before_final_validation() {
    let rules = nexus_native_physical_optimizer_rules(native_optimizer_config());
    let names = rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();

    let count_only_position = names.iter().position(|name| *name == "datafusion_nexus_native_count_only").unwrap();
    let aggregate_statistics_position = names.iter().position(|name| *name == "aggregate_statistics").unwrap();
    let native_position = names.iter().position(|name| *name == "datafusion_nexus_native").unwrap();
    let sanity_position = names.iter().position(|name| *name == "SanityCheckPlan").unwrap();
    assert!(count_only_position < aggregate_statistics_position);
    assert!(aggregate_statistics_position < native_position);
    assert!(native_position < sanity_position);

    let default_rules = nexus_native_physical_optimizer_rules(NexusNativeOptimizerConfig::default());
    let default_names = default_rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();
    assert!(default_names.contains(&"datafusion_nexus_native"));
    assert!(default_names.contains(&"datafusion_nexus_native_count_only"));
}

#[test]
fn native_optimizer_rule_executes_try_cast_string_to_integer_with_null_on_failure() {
    use std::fs::File;

    use arrow::{
        array::{Array, Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::{
        common::Result as DfResult,
        execution::{SessionStateBuilder, context::TaskContext},
        parquet::arrow::ArrowWriter,
        prelude::{ParquetReadOptions, SessionConfig, SessionContext},
    };
    use datafusion_nexus::session::NexusSessionStateBuilderExt;

    let dir = unique_temp_dir("datafusion-nexus-native-exec-try-cast");
    let path = dir.join("input.parquet");
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec![Some("42"), Some("abc"), Some("-7"), None, Some("999999999999")])),
            ],
        )
        .unwrap();
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false))
        .with_nexus_native(native_optimizer_config())
        .build();
    let ctx = SessionContext::new_with_state(state);
    futures::executor::block_on(ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()))
        .unwrap();

    let df =
        futures::executor::block_on(ctx.sql("SELECT id, TRY_CAST(name AS INT) AS maybe_int FROM bench ORDER BY id"))
            .unwrap();
    let plan = futures::executor::block_on(df.create_physical_plan()).unwrap();
    let optimized_tree = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        optimized_tree.contains("NexusNativeExec"),
        "TRY_CAST(string -> int) should lower into a native execution node:\n{optimized_tree}"
    );

    let batches: DfResult<Vec<RecordBatch>> =
        futures::executor::block_on(collect(plan, Arc::new(TaskContext::default())));
    let batches = batches.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 5);

    let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
    let mut ids = Vec::with_capacity(row_count);
    let mut values = Vec::with_capacity(row_count);
    for batch in &batches {
        let id_array = int32_array(batch, 0);
        let value_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("TRY_CAST projection should produce an Int32 column");
        for row in 0..batch.num_rows() {
            ids.push(id_array.value(row));
            values.push(if value_array.is_null(row) { None } else { Some(value_array.value(row)) });
        }
    }
    assert_eq!(ids, vec![1, 2, 3, 4, 5]);
    // Row 1 parses ("42"), row 2 fails ("abc" → null), row 3 parses ("-7"),
    // row 4 is null input (stays null), row 5 overflows Int32 ("999999999999" → null).
    assert_eq!(values, vec![Some(42), None, Some(-7), None, None]);

    fs::remove_dir_all(dir).unwrap();
}

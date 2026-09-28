//! Adapter exec-node tests that require access to hidden APIs.
//!
//! These tests construct `AlgeonNativeExec`, `AlgeonNativeComposedExec`, and
//! `AlgeonNativeGpuProducerExec` directly through their crate-private
//! constructors. Tests that execute plans hold a per-test backend fixture for
//! admission; the fixture closes on drop so the process-exclusive backend
//! lease is never held across tests.

#[path = "../../tests/native_exec_tests/support_core.rs"]
mod support_core;

mod bounded_join;
mod exec_plans;

use std::{
    collections::{BTreeSet, HashMap},
    fs,
    sync::{Arc, mpsc},
    time::Duration,
};

use algeon_datafusion::{
    attempt::AttemptOptions,
    backend::{AlgeonGpuDeviceProfile, AlgeonGpuQueuePolicy, GpuDeviceMemoryLimit, InstalledTestGpuBackend},
    error::ErrorCode,
    native_exec::{
        AlgeonNativeComposedExec, AlgeonNativeExec, AlgeonNativeExecutionConfig, AlgeonNativeGpuProducerExec,
        SourceChunking,
    },
    native_report::NativeDataFusionExecutionMetrics,
    planner::{
        AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule, native::wrap_direct_native_plan_with_admission,
    },
    report::CandidateOutcome,
};
use arrow::{
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
};
use datafusion::{
    common::{DataFusionError, config::ConfigOptions},
    execution::context::TaskContext,
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, collect, displayable,
        execution_plan::Boundedness,
        metrics::ExecutionPlanMetricsSet,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::SessionContext,
};
use exec_plans::*;
use futures::StreamExt;
use query_engine::{
    chunk::SourceChunkPlanningOptions,
    pipeline::{PipelineTaskKind, compile_pipeline},
    plan::QueryPlanBuilder,
    runtime::{LakehouseCacheFillConfig, SourceChunkPlanningMode},
    source::{GpuResidentSource, SourceNode},
};
use support_core::*;

use super::AlgeonNativeMetrics;

fn backend_fixture() -> InstalledTestGpuBackend {
    InstalledTestGpuBackend::with_config(
        template_execution_config(),
        AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 }),
        AlgeonGpuQueuePolicy::default(),
    )
}

fn bound_execution_config(fixture: &InstalledTestGpuBackend) -> AlgeonNativeExecutionConfig {
    fixture.backend().optimizer_config().native_execution_config().clone()
}

fn admitted_plan(fixture: &InstalledTestGpuBackend, plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let config = bound_execution_config(fixture);
    wrap_direct_native_plan_with_admission(plan, &config)
        .expect("a directly built native plan must mint its execution admission ticket")
}

fn template_execution_config() -> AlgeonNativeExecutionConfig {
    AlgeonNativeExecutionConfig::new(SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"))
}

fn template_optimizer_config() -> AlgeonNativeOptimizerConfig {
    AlgeonNativeOptimizerConfig::default().with_native_execution_config(template_execution_config())
}
#[test]
fn native_exec_constructs_host_arrow_leaf_with_conservative_properties() {
    let native_plan = host_sink_plan();
    let schema = Arc::clone(native_plan.output_schema());
    let exec = AlgeonNativeExec::try_new(native_plan, template_execution_config()).unwrap();

    assert_eq!(exec.name(), "AlgeonNativeExec");
    assert!(exec.children().is_empty());
    assert_eq!(exec.schema(), schema);
    assert_eq!(exec.properties().output_partitioning().partition_count(), 1);
    assert_eq!(exec.properties().boundedness, Boundedness::Bounded);
    assert!(exec.properties().output_ordering().is_none());

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("AlgeonNativeExec"));
    assert!(display.contains("sink=host_arrow"));
    assert!(display.contains("native_pipeline_execution_mode=bounded"));
    assert!(display.contains("edge_normalize_inserted=false"));

    let tree = displayable(&exec).indent(false).to_string();
    assert!(tree.contains("native_nodes=1"));
    assert!(tree.contains("output_boundary=device_to_host"));
    assert!(tree.contains("edge_normalize_inserted=false"));

    let verbose = displayable(&exec).indent(true).to_string();
    assert!(verbose.contains("native_plan=QueryPlan"), "{verbose}");
    assert!(verbose.contains("LocalParquetSource"), "{verbose}");
}

#[test]
fn native_exec_rejects_gpu_resident_sink_for_datafusion_output() {
    let error = AlgeonNativeExec::try_new(gpu_resident_plan(), template_execution_config()).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("result_sink"), Some("gpu_resident"));
}

#[test]
fn bounded_auto_sized_uses_bounded_defaults_and_prefetch_scheduler() {
    let config = AlgeonNativeExecutionConfig::default();

    assert_eq!(
        config.source_chunking().max_source_chunk_bytes.get(),
        SourceChunking::DEFAULT_MAX_SOURCE_CHUNK_BYTES.get()
    );
    assert_eq!(
        config.source_chunking().max_row_groups_per_chunk.get(),
        SourceChunking::DEFAULT_MAX_ROW_GROUPS_PER_CHUNK.get()
    );
    assert_eq!(config.source_chunking().max_retry_attempts.get(), SourceChunking::DEFAULT_MAX_RETRY_ATTEMPTS.get());
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
        AlgeonNativeExecutionConfig::default()
            .with_source_chunk_planning_options(source_chunk_planning_options)
            .source_chunk_planning_options(),
        source_chunk_planning_options
    );
    assert!(!AlgeonNativeExecutionConfig::default().host_output_memory_pool_accounting());
    assert!(
        AlgeonNativeExecutionConfig::default()
            .with_host_output_memory_pool_accounting(true)
            .host_output_memory_pool_accounting()
    );
}

#[test]
fn optimizer_config_default_uses_bounded_native_execution() {
    let config = AlgeonNativeOptimizerConfig::default();

    assert!(config.native_execution_config().source_chunking().max_source_chunk_bytes.get() > 0);
    assert_eq!(config.native_execution_config().source_scheduler_policy().max_source_reads_in_flight(), 2);
}

#[tokio::test]
async fn native_composed_exec_hands_gpu_result_between_fragments_without_host_materialization() {
    let dir = unique_temp_dir("algeon-datafusion-native-composed-gpu-resident");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let schema = parquet_file_schema();
    let fixture = backend_fixture();

    let producer_plan = local_parquet_gpu_resident_plan(&path, Arc::clone(&schema));
    let consumer_plan = gpu_resident_host_sink_plan("fragment_input", Arc::clone(&schema));
    let config = bound_execution_config(&fixture);
    let producer = Arc::new(AlgeonNativeGpuProducerExec::try_new("fragment_input", producer_plan).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let composed = Arc::new(AlgeonNativeComposedExec::try_new(consumer_plan, vec![producer_child], config).unwrap());

    assert_eq!(composed.name(), "AlgeonNativeComposedExec");
    assert_eq!(composed.children().len(), 1);
    let display = displayable(composed.as_ref()).one_line().to_string();
    assert!(display.contains("input_boundary=gpu_resident"));
    assert!(display.contains("output_boundary=device_to_host"));

    let admitted = admitted_plan(&fixture, Arc::clone(&composed) as Arc<dyn ExecutionPlan>);
    let batches = collect(admitted, Arc::new(TaskContext::default())).await.unwrap();

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

#[tokio::test]
async fn native_and_composed_host_outputs_bind_duplicate_external_names_by_ordinal() {
    let dir = unique_temp_dir("algeon-datafusion-native-host-schema-binding");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let internal_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let external_schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("first", DataType::Int32, true)
                .with_metadata(HashMap::from([("source".to_owned(), "left".to_owned())])),
            Field::new("first", DataType::Int32, true)
                .with_metadata(HashMap::from([("source".to_owned(), "right".to_owned())])),
            Field::new("shared_targets", DataType::Int64, true)
                .with_metadata(HashMap::from([("aggregate".to_owned(), "count".to_owned())])),
        ],
        HashMap::from([("query".to_owned(), "self_join_grouped".to_owned())]),
    ));
    let source = local_parquet_source_with_exact_metadata(
        &path,
        Arc::clone(&internal_schema),
        vec!["key".to_owned(), "payload".to_owned(), "m00".to_owned()],
    );

    let normal_plan = {
        let mut builder = QueryPlanBuilder::new();
        let root = builder.add_source(SourceNode::LocalParquet(source.clone()));
        builder.finish_with_host_sink_schema(root, Arc::clone(&external_schema)).unwrap()
    };
    let fixture = backend_fixture();
    let normal_config = bound_execution_config(&fixture);
    let normal = Arc::new(AlgeonNativeExec::try_new(normal_plan, normal_config).unwrap());
    let normal_admitted = admitted_plan(&fixture, Arc::clone(&normal) as Arc<dyn ExecutionPlan>);
    let normal_batches = collect(normal_admitted, Arc::new(TaskContext::default())).await.unwrap();
    drop(normal);

    let producer_plan = {
        let mut builder = QueryPlanBuilder::new();
        let root = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_gpu_resident(root).unwrap()
    };
    let consumer_plan = {
        let mut builder = QueryPlanBuilder::new();
        let root = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("fragment_input", Arc::clone(&internal_schema)).with_row_count_upper_bound(3),
        ));
        builder.finish_with_host_sink_schema(root, Arc::clone(&external_schema)).unwrap()
    };
    let composed_config = bound_execution_config(&fixture);
    let producer = Arc::new(AlgeonNativeGpuProducerExec::try_new("fragment_input", producer_plan).unwrap());
    let composed = Arc::new(
        AlgeonNativeComposedExec::try_new(consumer_plan, vec![producer as Arc<dyn ExecutionPlan>], composed_config)
            .unwrap(),
    );
    let composed_admitted = admitted_plan(&fixture, Arc::clone(&composed) as Arc<dyn ExecutionPlan>);
    let composed_batches = collect(composed_admitted, Arc::new(TaskContext::default())).await.unwrap();

    for batch in [&normal_batches[0], &composed_batches[0]] {
        assert_eq!(batch.schema(), external_schema);
        assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1]);
        assert_eq!(int32_array(batch, 1).values().as_ref(), &[5, 10, 20]);
        assert_eq!(int64_array(batch, 2).values().as_ref(), &[100, 200, 300]);
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_composed_host_output_memory_pool_accounting_fails_fast() {
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;

    let dir = unique_temp_dir("algeon-datafusion-native-composed-host-output-memory-pool");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let schema = parquet_file_schema();
    let fixture = backend_fixture();
    let config = bound_execution_config(&fixture).with_host_output_memory_pool_accounting(true);

    let producer_plan = local_parquet_gpu_resident_plan(&path, Arc::clone(&schema));
    let consumer_plan = gpu_resident_host_sink_plan("fragment_input", Arc::clone(&schema));
    let producer = Arc::new(AlgeonNativeGpuProducerExec::try_new("fragment_input", producer_plan).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let composed = Arc::new(AlgeonNativeComposedExec::try_new(consumer_plan, vec![producer_child], config).unwrap());
    // The four source columns contain only 72 logical cuDF payload bytes, but
    // their four Arrow destination buffers retain 4 * 64 = 256 bytes. A
    // 128-byte pool therefore proves admission uses the destination estimate:
    // source-payload accounting would incorrectly allow the D2H copy.
    let runtime_env = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(128, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime_env)));

    let admitted = admitted_plan(&fixture, Arc::clone(&composed) as Arc<dyn ExecutionPlan>);
    let error = collect(admitted, task_context)
        .await
        .expect_err("composed native host output should be charged to DataFusion's bounded memory pool");

    assert!(
        matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
        "expected ResourcesExhausted from composed host-output accounting, got {error:?}"
    );
    assert!(
        error.to_string().contains("AlgeonNativeComposedExec[host-output]"),
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

#[tokio::test]
async fn native_host_output_memory_pool_accounting_fails_fast_in_hybrid_plan() {
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;

    let dir = unique_temp_dir("algeon-datafusion-native-host-output-memory-pool");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);

    let native_plan = local_parquet_host_sink_plan(&path, parquet_file_schema());
    let fixture = backend_fixture();
    let config = bound_execution_config(&fixture).with_host_output_memory_pool_accounting(true);
    let native_child = Arc::new(AlgeonNativeExec::try_new(native_plan, config).unwrap()) as Arc<dyn ExecutionPlan>;
    let native_child = admitted_plan(&fixture, native_child);
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

    let error = collect(Arc::clone(&hybrid_plan), task_context)
        .await
        .expect_err("native host output should be charged to the bounded DataFusion memory pool");

    assert!(
        matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
        "expected ResourcesExhausted from DataFusion MemoryPool, got {error:?}"
    );
    assert!(
        error.to_string().contains("AlgeonNativeExec[host-output]"),
        "error should identify the native host-output memory consumer: {error}"
    );
    assert_eq!(runtime.memory_pool.reserved(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_host_output_memory_pool_accounting_releases_on_stream_eof() {
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use futures::StreamExt;

    let dir = unique_temp_dir("algeon-datafusion-native-host-output-memory-pool-eof");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);

    let native_plan = local_parquet_host_sink_plan(&path, parquet_file_schema());
    let fixture = backend_fixture();
    let config = bound_execution_config(&fixture).with_host_output_memory_pool_accounting(true);
    let native_exec = AlgeonNativeExec::try_new(native_plan, config).unwrap();
    let admitted = admitted_plan(&fixture, Arc::new(native_exec));
    let runtime = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1024 * 1024, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime)));
    let mut stream = admitted.execute(0, task_context).unwrap();

    let batch = stream.next().await.expect("native stream should emit one batch").unwrap();
    assert_eq!(batch.num_rows(), 3);
    assert!(runtime.memory_pool.reserved() > 0);

    let eof = stream.next().await;
    assert!(eof.is_none());
    assert_eq!(runtime.memory_pool.reserved(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_composes_feature_neutral_gpu_resident_producer_child() {
    let producer = Arc::new(AlgeonNativeGpuProducerExec::try_new("fragment_input", gpu_resident_plan()).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let child_schema = producer_child.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("key", &child_schema).unwrap(), alias: "key".to_string() }],
            producer_child,
        )
        .unwrap(),
    );
    let rule = AlgeonNativePhysicalOptimizerRule::new(template_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    let composed = optimized
        .downcast_ref::<AlgeonNativeComposedExec>()
        .expect("GPU-resident producer children should lower through composed native execution");
    assert_eq!(composed.children().len(), 1);
    assert!(composed.children()[0].downcast_ref::<AlgeonNativeGpuProducerExec>().is_some());
    assert!(composed.native_plan().to_string().contains("GpuResidentSource input_name=fragment_input"));
    let report = diagnostics.expect("native optimizer should record planning diagnostics").report;
    assert_eq!(report.selected_count(), 1);
}

#[test]
fn native_optimizer_composed_exec_preserves_root_sort_ordering_properties() {
    let producer = Arc::new(AlgeonNativeGpuProducerExec::try_new("fragment_input", gpu_resident_plan()).unwrap());
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
    let rule = AlgeonNativePhysicalOptimizerRule::new(template_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeComposedExec>().is_some());
    assert_eq!(optimized.properties().output_ordering().cloned(), expected_ordering);
    let report = diagnostics.expect("native optimizer should record planning diagnostics").report;
    assert_eq!(report.selected_count(), 1);
}

#[tokio::test]
async fn deferred_host_wrapper_metrics_project_the_plans_final_metrics() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-deferred-host-metrics");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let fixture = backend_fixture();
    let config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(bound_execution_config(&fixture));
    let rule = AlgeonNativePhysicalOptimizerRule::new(config);
    let plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let attempt = fixture.backend().begin_attempt(AttemptOptions::default()).expect("attempt begins");
    let (sealed_sender, sealed) = mpsc::channel();
    attempt.register_sealed_execution_observer(move |_report, snapshot, native_execution| {
        sealed_sender.send((snapshot, native_execution)).expect("the test receives the seal");
    });

    let mut stream = attempt.execute_physical_plan(plan, &SessionContext::new()).unwrap();
    while let Some(batch) = stream.next().await {
        batch.unwrap();
    }
    attempt.completion().await;
    let (snapshot, native_execution) = sealed.recv_timeout(Duration::from_secs(5)).expect("the attempt seals");
    let actual = native_execution.expect("the native wrapper's metrics seal with the attempt").metrics;

    let plans = snapshot.events().iter().filter_map(|event| event.plan_execution_id.clone()).collect::<BTreeSet<_>>();
    assert_eq!(plans.len(), 1, "the query runs one deferred-host plan: {plans:?}");
    let plan_metrics = snapshot.query_metrics(plans.first().unwrap()).expect("the seal carries the plan's metrics");
    // The wrapper labels its plan-task rows with its own DataFusion shape.
    let shape = actual.plan_task_mapping_rows.first().and_then(|row| row.datafusion_shape.clone());
    let projected = ExecutionPlanMetricsSet::new();
    AlgeonNativeMetrics::new_with_capture_and_shape(&projected, 0, None, shape.as_deref())
        .record_query_metrics(plan_metrics);
    let expected = NativeDataFusionExecutionMetrics {
        admission_sequence: actual.admission_sequence,
        admission_wait_time_ns: actual.admission_wait_time_ns,
        device_ordinal: actual.device_ordinal,
        query_grant_bytes: actual.query_grant_bytes,
        ..NativeDataFusionExecutionMetrics::from_metrics(&projected.clone_inner())
    };
    assert_eq!(actual.device_to_host_materializations, 1, "the cursor copies the one result batch");
    assert_eq!(actual, expected, "DataFusion metrics are the projection of the plan's final metrics");

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_optimizer_property_checked_ilike_boundary_honors_pre_execution_cancellation() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike-cancelled");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let fixture = backend_fixture();
    let config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(bound_execution_config(&fixture));
    let rule = AlgeonNativePhysicalOptimizerRule::new(config);
    let plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let attempt = fixture.backend().begin_attempt(AttemptOptions::default()).expect("attempt begins");
    attempt.cancellation().cancel();

    let error = match attempt.execute_physical_plan(plan, &SessionContext::new()) {
        Ok(_) => panic!("the mixed plan must observe cancellation before it starts native execution"),
        Err(error) => error,
    };
    assert_eq!(adapter_error(&error).code(), ErrorCode::Cancelled);

    fs::remove_dir_all(dir).unwrap();
}

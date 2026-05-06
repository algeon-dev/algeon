use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf_query_engine::{
    ColumnRef, GpuResidentSource, JoinKey, JoinOutputColumn, JoinSpec, JoinType, LocalParquetFile, LocalParquetSource,
    QueryPlan, QueryPlanBuilder, SourceNode,
};
use datafusion::{
    common::{DataFusionError, NullEquality, ScalarValue, config::ConfigOptions},
    execution::context::TaskContext,
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    parquet::arrow::ArrowWriter,
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{binary, col, lit},
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        collect, displayable,
        execution_plan::Boundedness,
        filter::FilterExec,
        joins::{HashJoinExec, PartitionMode},
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, ErrorCode, FallbackReason,
    GpuFallbackPolicy, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec,
    cudf_physical_optimizer_rules,
};

#[test]
fn native_exec_constructs_host_arrow_leaf_with_conservative_properties() {
    let native_plan = host_sink_plan();
    let schema = Arc::clone(native_plan.output_schema());
    let exec = NativeCudfExec::try_new(native_plan, bounded_config()).unwrap();

    assert_eq!(exec.name(), "NativeCudfExec");
    assert!(exec.children().is_empty());
    assert_eq!(exec.schema(), schema);
    assert_eq!(exec.properties().output_partitioning().partition_count(), 1);
    assert_eq!(exec.properties().boundedness, Boundedness::Bounded);
    assert!(exec.properties().output_ordering().is_none());

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("NativeCudfExec"));
    assert!(display.contains("sink=host_arrow"));
    assert!(display.contains("memory_control_policy=bounded"));
    assert!(display.contains("boundary_policy=source_read=direct_device_local_parquet"));

    let tree = displayable(&exec).indent(false).to_string();
    assert!(tree.contains("native_nodes=1"));
    assert!(tree.contains("boundary_policy=source_read=direct_device_local_parquet"));
    assert!(tree.contains("output_boundary=device_to_host"));
}

#[test]
fn native_exec_rejects_gpu_resident_sink_for_datafusion_output() {
    let error = NativeCudfExec::try_new(gpu_resident_plan(), bounded_config()).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("result_sink"), Some("gpu_resident"));
}

#[test]
fn native_exec_accepts_unbounded_memory_control_policy() {
    let exec = NativeCudfExec::try_new(host_sink_plan(), NativeCudfExecutionConfig::unbounded()).unwrap();

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("NativeCudfExec"));
    assert!(display.contains("memory_control_policy=unbounded"));
}

#[test]
fn native_composed_exec_hands_gpu_result_between_fragments_without_host_materialization() {
    let dir = unique_temp_dir("datafusion-cudf-native-composed-gpu-resident");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let schema = parquet_file_schema();
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let config = NativeCudfExecutionConfig::unbounded().with_shared_runtime(Arc::clone(&runtime));

    let producer_plan = local_parquet_gpu_resident_plan(&path, Arc::clone(&schema));
    let producer = Arc::new(NativeCudfGpuProducerExec::try_new("fragment_input", producer_plan).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let consumer_plan = gpu_resident_host_sink_plan("fragment_input", Arc::clone(&schema));
    let composed = Arc::new(NativeCudfComposedExec::try_new(consumer_plan, vec![producer_child], config).unwrap());

    assert_eq!(composed.name(), "NativeCudfComposedExec");
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
fn native_optimizer_rule_rewrites_supported_physical_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-optimizer");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_preserves_rejected_plan_for_downstream_fallback() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-optimizer-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::BitwiseAnd);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<LocalLimitExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 0);
    assert_eq!(report.rejected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_supported_child_under_host_arrow_cpu_parent() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-partial-host-arrow-boundary");
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
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let projection = optimized
        .as_any()
        .downcast_ref::<ProjectionExec>()
        .expect("unsupported CPU parent should remain above the host-arrow native boundary");
    assert!(projection.input().as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedOperator, .. }
    ));
    assert!(matches!(report.outcomes()[1], CandidateOutcome::Rewritten { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rejects_bounded_join_plan_before_replacement() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-bounded-join-rejected");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_left_semi_join_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_none());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 0);
    assert_eq!(report.rejected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeBoundedJoinUnsupported, .. }
    ));
    assert_eq!(rule.last_source_diagnostics().len(), 2);
    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain diagnostics");
    assert_eq!(diagnostics.bounded_join_reservation_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(
        diagnostics.bounded_join_reservation_facts.get("modeled_reservations").map(String::as_str),
        Some("build_side,hash_table,probe_side,gather_map,join_output,materialization")
    );
    assert_eq!(
        diagnostics.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert_eq!(
        diagnostics.bounded_join_reservation_facts.get("bounded_join_unsupported_shape").map(String::as_str),
        Some("requires_root_reusable_hash_or_keyless_conditional_join")
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_join_aggregate_pipeline() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-bounded-join-aggregate");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));

    let batches = futures::executor::block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(batches[0].num_columns(), 2);
    assert_eq!(int32_array(&batches[0], 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(&batches[0], 1).values().as_ref(), &[303, 202]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_root_hash_join() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-bounded-root-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_hash_join_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));

    let batches = futures::executor::block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 3);
    let mut rows = (0..batches[0].num_rows())
        .map(|row| {
            (
                int32_array(&batches[0], 0).value(row),
                int64_array(&batches[0], 1).value(row),
                int64_array(&batches[0], 2).value(row),
            )
        })
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 100, 1), (1, 300, 1), (2, 200, 2)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_exec_preserves_bounded_join_rejection_facts_from_native_runtime() {
    let native_plan = bounded_left_semi_join_host_sink_plan();
    let exec = Arc::new(NativeCudfExec::try_new(native_plan, bounded_config()).unwrap()) as Arc<dyn ExecutionPlan>;

    let error = futures::executor::block_on(collect(exec, Arc::new(TaskContext::default()))).unwrap_err();
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(adapter_error.fact_value("native_code"), Some("unsupported_plan_shape"));
    assert_eq!(adapter_error.fact_value("native_operation"), Some("cudf_query_engine.bounded.plan"));
    assert_eq!(adapter_error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    assert_eq!(adapter_error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(
        adapter_error.fact_value("boundary_policy"),
        Some(
            "source_read=direct_device_local_parquet,build_materialization=query_scoped_gpu_hash_join,probe_execution=reusable_gpu_hash_join,result_materialization=explicit_sink"
        )
    );
    assert_eq!(adapter_error.fact_value("source_read_boundary"), Some("direct_device_local_parquet"));
    assert_eq!(adapter_error.fact_value("build_materialization_boundary"), Some("query_scoped_gpu_hash_join"));
    assert_eq!(adapter_error.fact_value("probe_execution_boundary"), Some("reusable_gpu_hash_join"));
    assert_eq!(adapter_error.fact_value("result_materialization_boundary"), Some("explicit_sink"));
    assert_eq!(adapter_error.fact_value("blocked_reservations"), Some(""));
    assert_eq!(
        adapter_error.fact_value("modeled_reservations"),
        Some("build_side,hash_table,probe_side,gather_map,join_output,materialization")
    );
    assert_eq!(adapter_error.fact_value("bounded_join_lower_level_blockers"), Some(""));
    assert_eq!(
        adapter_error.fact_value("cudf_rs_dataframe_gather_output_reservation_contract"),
        Some("dataframe_gather_output_reservation_size_bytes")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_conditional_join_output_size_observability"),
        Some("conditional_join_exact_output_size")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_conditional_join_output_size_hint"),
        Some("conditional_join_indices_with_output_size")
    );
    assert_eq!(adapter_error.fact_value("path"), None);
    assert_eq!(adapter_error.fact_value("file_index"), None);
    assert_eq!(adapter_error.fact_value("row_groups"), None);
    assert_eq!(adapter_error.fact_value("chunk_index"), None);
}

#[test]
fn bounded_join_rejection_split_between_planner_and_runtime_paths_is_pinned() {
    // The planner path (`native_execution_boundary_rejection_facts`) and the
    // runtime path (`reject_bounded_join_plan`) intentionally emit different
    // values for `bounded_join_execution_status` for the same logical
    // condition (an unsupported bounded-join shape). This test pins the
    // documented split so future drift in either side is immediately visible.

    // --- Planner path ---
    let dir = unique_temp_dir("datafusion-cudf-bounded-join-rejection-split");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_left_semi_join_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());
    let _ = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain diagnostics");
    let planner_facts = &diagnostics.bounded_join_reservation_facts;

    assert_eq!(planner_facts.get("bounded_join_enabled").map(String::as_str), Some("true"));
    assert_eq!(planner_facts.get("bounded_join_execution_status").map(String::as_str), Some("unsupported_plan_shape"));
    assert_eq!(
        planner_facts.get("bounded_join_unsupported_shape").map(String::as_str),
        Some("requires_root_reusable_hash_or_keyless_conditional_join")
    );
    assert_eq!(
        planner_facts.get("modeled_reservations").map(String::as_str),
        Some("build_side,hash_table,probe_side,gather_map,join_output,materialization")
    );
    assert_eq!(planner_facts.get("blocked_reservations").map(String::as_str), Some(""));

    fs::remove_dir_all(dir).unwrap();

    // --- Runtime path ---
    let native_plan = bounded_left_semi_join_host_sink_plan();
    let exec = Arc::new(NativeCudfExec::try_new(native_plan, bounded_config()).unwrap()) as Arc<dyn ExecutionPlan>;
    let error = futures::executor::block_on(collect(exec, Arc::new(TaskContext::default()))).unwrap_err();
    let runtime_error = adapter_error(&error);

    assert_eq!(runtime_error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(runtime_error.fact_value("bounded_join_execution_status"), Some("enabled"));
    assert_eq!(
        runtime_error.fact_value("bounded_join_execution_families"),
        Some("root_reusable_hash_join,keyless_conditional_join")
    );
    assert_eq!(runtime_error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    assert_eq!(runtime_error.fact_value("bounded_join_unsupported_shape"), None);
    assert_eq!(
        runtime_error.fact_value("modeled_reservations"),
        Some("build_side,hash_table,probe_side,gather_map,join_output,materialization")
    );
    assert_eq!(runtime_error.fact_value("blocked_reservations"), Some(""));

    // --- The split itself ---
    // Both paths advertise that the contract is enabled and the same reservation
    // kinds are modeled, but the planner downgrades `execution_status` to
    // `unsupported_plan_shape` and adds `bounded_join_unsupported_shape`, while
    // the runtime keeps the executable bounded root-join status and surfaces a
    // `reason` instead.
    assert_ne!(
        planner_facts.get("bounded_join_execution_status").map(String::as_str),
        runtime_error.fact_value("bounded_join_execution_status")
    );
    assert_eq!(
        planner_facts.get("modeled_reservations").map(String::as_str),
        runtime_error.fact_value("modeled_reservations")
    );
    assert_eq!(
        planner_facts.get("blocked_reservations").map(String::as_str),
        runtime_error.fact_value("blocked_reservations")
    );
}

#[test]
fn native_optimizer_rule_is_inserted_before_gpu_island_rule_when_enabled() {
    let rules = cudf_physical_optimizer_rules(native_optimizer_config());
    let names = rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();

    let count_only_position = names.iter().position(|name| *name == "datafusion_cudf_native_count_only").unwrap();
    let aggregate_statistics_position = names.iter().position(|name| *name == "aggregate_statistics").unwrap();
    let native_position = names.iter().position(|name| *name == "datafusion_cudf_native").unwrap();
    let gpu_position = names.iter().position(|name| *name == "datafusion_cudf_gpu_island").unwrap();
    let sanity_position = names.iter().position(|name| *name == "SanityCheckPlan").unwrap();
    assert!(count_only_position < aggregate_statistics_position);
    assert!(aggregate_statistics_position < native_position);
    assert!(native_position < gpu_position);
    assert!(gpu_position < sanity_position);

    let default_rules = cudf_physical_optimizer_rules(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred));
    let default_names = default_rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();
    assert!(!default_names.contains(&"datafusion_cudf_native"));
    assert!(!default_names.contains(&"datafusion_cudf_native_count_only"));
}

fn bounded_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1)
}

fn native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(bounded_config())
}

fn host_sink_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_with_host_sink(source)
}

fn gpu_resident_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_gpu_resident(source)
}

fn bounded_left_semi_join_host_sink_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    let right = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    let key = JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"));
    let spec = JoinSpec::new(JoinType::LeftSemi, vec![key])
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key")]);
    let join = builder.add_join(left, right, spec, schema());
    builder.finish_with_host_sink(join)
}

fn local_parquet_gpu_resident_plan(path: &std::path::Path, schema: Arc<Schema>) -> QueryPlan {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        schema,
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    builder.finish_gpu_resident(source)
}

fn gpu_resident_host_sink_plan(input_name: &str, schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    builder.finish_with_host_sink(source)
}

fn local_parquet_source() -> LocalParquetSource {
    LocalParquetSource::new("/tmp/datafusion-cudf-native-exec-test.parquet", schema(), vec!["key".to_string()])
}

fn int32_array(batch: &RecordBatch, index: usize) -> &Int32Array {
    batch.column(index).as_any().downcast_ref::<Int32Array>().unwrap()
}

fn int64_array(batch: &RecordBatch, index: usize) -> &Int64Array {
    batch.column(index).as_any().downcast_ref::<Int64Array>().unwrap()
}

fn adapter_error(error: &DataFusionError) -> &datafusion_cudf::Error {
    let DataFusionError::External(source) = error else {
        panic!("native execution errors should cross DataFusion as external adapter errors: {error}");
    };
    source.downcast_ref::<datafusion_cudf::Error>().expect("DataFusion external error should preserve adapter Error")
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]))
}

fn parquet_file_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]))
}

fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1),
    );
    let df =
        futures::executor::block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

fn join_aggregate_sort_limit_sql_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_repartition_file_scans(false)
            .with_repartition_joins(false)
            .with_repartition_sorts(false)
            .with_target_partitions(1),
    );
    futures::executor::block_on(ctx.register_parquet(
        "left_input",
        left.to_str().unwrap(),
        ParquetReadOptions::default(),
    ))
    .unwrap();
    futures::executor::block_on(ctx.register_parquet(
        "right_input",
        right.to_str().unwrap(),
        ParquetReadOptions::default(),
    ))
    .unwrap();
    let sql = "
SELECT l.key, SUM(l.m00 + r.m01) AS sum_x00
FROM left_input l
JOIN right_input r ON l.key = r.key
WHERE l.payload >= 10 AND r.payload >= 10
GROUP BY l.key
ORDER BY sum_x00 DESC, l.key ASC
LIMIT 2
";
    let df = futures::executor::block_on(ctx.sql(sql)).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

fn root_hash_join_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let left = parquet_scan(left);
    let right = parquet_scan(right);
    let left_schema = left.schema();
    let right_schema = right.schema();
    Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            None,
            &datafusion::logical_expr::JoinType::Inner,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("r_m01").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    )
}

fn root_left_semi_join_plan(left: &std::path::Path, right: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let left = parquet_scan(left);
    let right = parquet_scan(right);
    let left_schema = left.schema();
    let right_schema = right.schema();
    Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("key", &left_schema).unwrap(), col("key", &right_schema).unwrap())],
            None,
            &datafusion::logical_expr::JoinType::LeftSemi,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    )
}

fn aggregate_sort_limit_plan(scan: Arc<dyn ExecutionPlan>, projection_operator: Operator) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let filter_schema = filter.schema();
    let x00 = binary(
        col("m00", &filter_schema).unwrap(),
        projection_operator,
        col("m01", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));
    Arc::new(LocalLimitExec::new(sort, 2))
}

fn write_parquet_file(path: &std::path::Path) {
    let schema = parquet_file_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_join_left_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("l_payload", DataType::Int32, false),
        Field::new("l_m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 20, 30])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

fn write_join_right_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("r_payload", DataType::Int32, false),
        Field::new("r_m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 10, 10])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

fn write_batch(path: &std::path::Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

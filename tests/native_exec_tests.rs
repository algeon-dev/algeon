use std::{fs, sync::Arc};

use datafusion::{
    common::{DataFusionError, config::ConfigOptions},
    execution::context::TaskContext,
    logical_expr::Operator,
    physical_expr::expressions::col,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, collect, displayable,
        execution_plan::Boundedness,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
    },
};
use datafusion_nexus::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, ErrorCode, FallbackReason,
    GpuFallbackPolicy, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec,
    NativePipelineExecutionMode, NativePlanPreferenceOptions, STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
    cudf_native_physical_optimizer_rules, lower_physical_plan_to_native,
};
use datafusion_nexus_bench::{TPCH_QUERIES, write_tiny_tpch_parquet_dataset};
use nexus_query_engine::{
    BoundedJoinReservationContract, MemoryControlPolicy, PipelineTaskKind, ReservationKind, SourceChunkPlanningMode,
    StrictBoundedExecutionMode, capability::ContractQuality, compile_pipeline,
};

#[path = "native_exec_tests/bounded_join.rs"]
mod bounded_join;
#[path = "native_exec_tests/support.rs"]
mod support;

use support::*;

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
    assert!(display.contains("edge_normalize_inserted=false"));

    let tree = displayable(&exec).indent(false).to_string();
    assert!(tree.contains("native_nodes=1"));
    assert!(tree.contains("boundary_policy=source_read=direct_device_local_parquet"));
    assert!(tree.contains("output_boundary=device_to_host"));
    assert!(tree.contains("edge_normalize_inserted=false"));
}

#[test]
fn native_exec_display_records_selected_adaptive_pipeline_mode() {
    let native_plan = host_sink_plan();
    let config = bounded_config().with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive);
    let exec = NativeCudfExec::try_new(native_plan, config).unwrap();

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("native_pipeline_execution_mode=adaptive"));
}

#[test]
fn native_exec_rejects_gpu_resident_sink_for_datafusion_output() {
    let error = NativeCudfExec::try_new(gpu_resident_plan(), bounded_config()).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("result_sink"), Some("gpu_resident"));
}

#[test]
fn native_exec_accepts_unbounded_memory_control_policy() {
    // Unbounded mode remains part of the public API for explicit diagnostic and
    // regression coverage even though production defaults are bounded.
    let exec = NativeCudfExec::try_new(host_sink_plan(), NativeCudfExecutionConfig::unbounded()).unwrap();

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("NativeCudfExec"));
    assert!(display.contains("memory_control_policy=unbounded"));
}

#[test]
fn bounded_auto_sized_uses_bounded_defaults_and_prefetch_scheduler() {
    let config = NativeCudfExecutionConfig::bounded_auto_sized();

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
    assert_eq!(max_source_chunk_bytes, NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(max_row_groups_per_chunk, NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(max_retry_attempts, NativeCudfExecutionConfig::DEFAULT_BOUNDED_MAX_RETRY_ATTEMPTS);
    assert_eq!(min_retry_chunk_row_groups, NativeCudfExecutionConfig::DEFAULT_BOUNDED_MIN_RETRY_CHUNK_ROW_GROUPS);
    assert_eq!(config.source_scheduler_policy().max_source_reads_in_flight(), 2);
    assert_eq!(config.source_chunk_planning_mode(), SourceChunkPlanningMode::Conservative);
    assert_eq!(
        config
            .with_source_chunk_planning_mode(SourceChunkPlanningMode::LocalParquetThroughput)
            .source_chunk_planning_mode(),
        SourceChunkPlanningMode::LocalParquetThroughput
    );
}

#[test]
fn optimizer_config_default_uses_bounded_native_execution() {
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred);

    assert!(config.native_execution_config().memory_control_policy().is_bounded());
    assert_eq!(config.native_execution_config().source_scheduler_policy().max_source_reads_in_flight(), 2);
}

#[test]
fn native_composed_exec_hands_gpu_result_between_fragments_without_host_materialization() {
    let dir = unique_temp_dir("datafusion-nexus-native-composed-gpu-resident");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let schema = parquet_file_schema();
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    // This test isolates GPU-resident fragment handoff semantics from bounded
    // source scheduling and reservation behavior.
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
fn native_optimizer_composes_feature_neutral_gpu_resident_producer_child() {
    let producer = Arc::new(NativeCudfGpuProducerExec::try_new("fragment_input", gpu_resident_plan()).unwrap());
    let producer_child = Arc::clone(&producer) as Arc<dyn ExecutionPlan>;
    let child_schema = producer_child.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("key", &child_schema).unwrap(), alias: "key".to_string() }],
            producer_child,
        )
        .unwrap(),
    );
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let composed = optimized
        .downcast_ref::<NativeCudfComposedExec>()
        .expect("GPU-resident producer children should lower through composed native execution");
    assert_eq!(composed.children().len(), 1);
    assert!(composed.children()[0].downcast_ref::<NativeCudfGpuProducerExec>().is_some());
    assert!(composed.native_plan().to_string().contains("GpuResidentSource input_name=fragment_input"));
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
}

#[test]
fn native_optimizer_rule_rewrites_supported_physical_plan() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-optimizer");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
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
fn native_optimizer_rule_preserves_rejected_plan_for_downstream_fallback() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-optimizer-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<LocalLimitExec>().is_some());
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
fn native_optimizer_rule_obeys_cpu_only_policy_for_supported_plan() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-cpu-only");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::CpuOnly).with_native_execution_config(bounded_config());
    let rule = CudfNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<LocalLimitExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 0);
    assert_eq!(report.rejected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::ExplicitCpuPolicy, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_prefers_cpu_for_small_selective_topk_join_when_enabled() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-preference-topk");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let config = native_optimizer_config().with_native_plan_preference_options(NativePlanPreferenceOptions::default());
    let rule = CudfNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(
        optimized.downcast_ref::<NativeCudfExec>().is_none(),
        "CPU-preferred plan should not be replaced by a native execution boundary"
    );
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 0);
    assert_eq!(report.rejected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSmallSelectiveTopkCpuPreferred, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_prefers_cpu_for_small_selective_full_sort_join_when_enabled() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-preference-full-sort");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_sql_plan(&left, &right);
    let config = native_optimizer_config().with_native_plan_preference_options(NativePlanPreferenceOptions::default());
    let rule = CudfNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(
        optimized.downcast_ref::<NativeCudfExec>().is_none(),
        "CPU-preferred plan should not be replaced by a native execution boundary"
    );
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 0);
    assert_eq!(report.rejected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSmallSelectiveFullSortCpuPreferred, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_errors_for_cpu_preferred_plan_under_gpu_only_policy() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-preference-gpu-only");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuOnly)
        .with_native_execution_config(bounded_config())
        .with_native_plan_preference_options(NativePlanPreferenceOptions::default());
    let rule = CudfNativePhysicalOptimizerRule::new(config);

    let error = rule
        .optimize(plan, &ConfigOptions::default())
        .expect_err("GpuOnly native optimizer must fail when preference rejects a supported candidate");
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::NativeCandidateRejected);
    assert_eq!(adapter_error.fact_value("fallback_reason"), Some("native_small_selective_topk_cpu_preferred"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_obeys_gpu_only_policy_for_rejected_plan() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-gpu-only-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuOnly).with_native_execution_config(bounded_config());
    let rule = CudfNativePhysicalOptimizerRule::new(config);

    let error = rule
        .optimize(plan, &ConfigOptions::default())
        .expect_err("GpuOnly native optimizer must fail when a native candidate is rejected");
    let DataFusionError::External(source) = &error else {
        panic!("GpuOnly native rejection should preserve the typed adapter error");
    };
    let adapter_error = source.downcast_ref::<datafusion_nexus::Error>().expect("expected datafusion-nexus Error");

    assert_eq!(adapter_error.code(), ErrorCode::NativeCandidateRejected);
    assert_eq!(adapter_error.fact_value("fallback_reason"), Some("unsupported_expression"));
    assert!(error.to_string().contains("unsupported_expression"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_composes_supported_host_arrow_parent_over_native_child() {
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
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let composed = optimized
        .downcast_ref::<NativeCudfComposedExec>()
        .expect("supported host-arrow parent should compose over the native child fragment");
    assert_eq!(composed.children().len(), 1);
    assert!(composed.children()[0].downcast_ref::<NativeCudfGpuProducerExec>().is_some());
    assert!(composed.native_plan().to_string().contains("GpuResidentSource input_name=__native_fragment_0"));
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 2);
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
fn native_optimizer_rule_gpu_only_allows_composed_native_root_with_superseded_rejection() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-composition-gpu-only");
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
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuOnly).with_native_execution_config(bounded_config());
    let rule = CudfNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NativeCudfComposedExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 2);
    assert_eq!(report.rejected_count(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_composes_distinct_aggregate_over_anti_join_fragment() {
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
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        !optimized_tree.contains("NativeCudfGpuProducerExec"),
        "anti-join fragments must not be producerized until the bounded join GPU-resident sink is admitted:\n{optimized_tree}",
    );
    assert!(
        optimized_tree.contains("NativeCudfExec"),
        "the anti-join fragment should still use the safe host-arrow native boundary:\n{optimized_tree}",
    );
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert!(
        report.rewritten_count() >= 1,
        "anti-join child should be reported as a native rewrite even when producerization is rejected: {report:?}"
    );

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
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NativeCudfExec>().is_some());
    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        optimized_tree.contains("NativeCudfExec"),
        "stddev subtree should be rewritten once native order tracking is available:\n{optimized_tree}"
    );
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_is_inserted_before_final_validation() {
    let rules = cudf_native_physical_optimizer_rules(native_optimizer_config());
    let names = rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();

    let count_only_position = names.iter().position(|name| *name == "datafusion_nexus_native_count_only").unwrap();
    let aggregate_statistics_position = names.iter().position(|name| *name == "aggregate_statistics").unwrap();
    let native_position = names.iter().position(|name| *name == "datafusion_nexus_native").unwrap();
    let sanity_position = names.iter().position(|name| *name == "SanityCheckPlan").unwrap();
    assert!(count_only_position < aggregate_statistics_position);
    assert!(aggregate_statistics_position < native_position);
    assert!(native_position < sanity_position);

    let default_rules = cudf_native_physical_optimizer_rules(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred));
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
    use datafusion_nexus::CudfSessionStateBuilderExt;

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
        .with_cudf_native(native_optimizer_config())
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
        optimized_tree.contains("NativeCudfExec"),
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

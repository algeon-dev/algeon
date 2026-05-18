use std::{fs, sync::Arc};

use cudf_query_engine::{
    BoundedJoinReservationContract, PipelineTaskKind, ReservationKind, StrictBoundedExecutionMode,
    capability::ContractQuality, compile_pipeline,
};
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
use datafusion_cudf::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, ErrorCode, FallbackReason,
    GpuFallbackPolicy, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig, NativeCudfGpuProducerExec,
    NativePipelineExecutionMode, STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON, TPCH_QUERIES,
    cudf_native_physical_optimizer_rules, lower_physical_plan_to_native, write_tiny_tpch_parquet_dataset,
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

    let tree = displayable(&exec).indent(false).to_string();
    assert!(tree.contains("native_nodes=1"));
    assert!(tree.contains("boundary_policy=source_read=direct_device_local_parquet"));
    assert!(tree.contains("output_boundary=device_to_host"));
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
        .as_any()
        .downcast_ref::<NativeCudfComposedExec>()
        .expect("GPU-resident producer children should lower through composed native execution");
    assert_eq!(composed.children().len(), 1);
    assert!(composed.children()[0].as_any().downcast_ref::<NativeCudfGpuProducerExec>().is_some());
    assert!(composed.native_plan().to_string().contains("GpuResidentSource input_name=fragment_input"));
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
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
    let dir = unique_temp_dir("datafusion-cudf-native-exec-optimizer-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_distinct_projection_aggregate_sort_limit_plan(parquet_scan(&path));
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
fn native_optimizer_rule_obeys_cpu_only_policy_for_supported_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-cpu-only");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::CpuOnly).with_native_execution_config(bounded_config());
    let rule = CudfNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<LocalLimitExec>().is_some());
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
fn native_optimizer_rule_obeys_gpu_only_policy_for_rejected_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-gpu-only-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_distinct_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuOnly).with_native_execution_config(bounded_config());
    let rule = CudfNativePhysicalOptimizerRule::new(config);

    let error = rule
        .optimize(plan, &ConfigOptions::default())
        .expect_err("GpuOnly native optimizer must fail when a native candidate is rejected");
    let DataFusionError::External(source) = &error else {
        panic!("GpuOnly native rejection should preserve the typed adapter error");
    };
    let adapter_error = source.downcast_ref::<datafusion_cudf::Error>().expect("expected datafusion-cudf Error");

    assert_eq!(adapter_error.code(), ErrorCode::NativeCandidateRejected);
    assert_eq!(adapter_error.fact_value("fallback_reason"), Some("unsupported_expression"));
    assert!(error.to_string().contains("unsupported_expression"));

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
fn native_optimizer_rule_rewrites_precision_sensitive_stddev_sort_subtree_after_order_contract() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-stddev-sort-cpu-boundary");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = stddev_sort_sql_plan(&path);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
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

    let count_only_position = names.iter().position(|name| *name == "datafusion_cudf_native_count_only").unwrap();
    let aggregate_statistics_position = names.iter().position(|name| *name == "aggregate_statistics").unwrap();
    let native_position = names.iter().position(|name| *name == "datafusion_cudf_native").unwrap();
    let sanity_position = names.iter().position(|name| *name == "SanityCheckPlan").unwrap();
    assert!(count_only_position < aggregate_statistics_position);
    assert!(aggregate_statistics_position < native_position);
    assert!(native_position < sanity_position);

    let default_rules = cudf_native_physical_optimizer_rules(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred));
    let default_names = default_rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();
    assert!(default_names.contains(&"datafusion_cudf_native"));
    assert!(default_names.contains(&"datafusion_cudf_native_count_only"));
}

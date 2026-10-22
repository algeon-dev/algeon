use std::{
    fs::{self, File},
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use algeon_datafusion::{
    backend::{
        AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuDeviceMemoryLimit, GpuMemoryOwnership,
    },
    capability::CapabilityReason,
    error::{ClientDetail, DataFusionErrorExt, ErrorCode},
    native_exec::{AlgeonNativeExec, AlgeonNativeExecutionConfig, SourceChunking, below_native_admission},
    native_report::{CandidateOutcomeKind, NativeDataFusionPlanningReport, NativeDataFusionPlanningWorkloadItem},
    planner::{
        AlgeonGpuExecutionMode, AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule, FinalPlanDisposition,
        classify_final_plan,
    },
    report::CandidateOutcome,
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{
        Array, ArrayRef, Date32Array, Decimal128Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
        ListArray, StringArray, StringViewArray,
    },
    buffer::{OffsetBuffer, ScalarBuffer},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::testing;
use datafusion::{
    arrow::util::pretty::pretty_format_batches,
    common::{DataFusionError, config::ConfigOptions},
    datasource::{memory::MemorySourceConfig, source::DataSourceExec},
    execution::{SessionStateBuilder, context::TaskContext},
    logical_expr::Operator,
    physical_expr::{LexOrdering, Partitioning, PhysicalSortExpr, expressions::col},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, collect,
        coop::CooperativeExec,
        displayable,
        execution_plan::{Boundedness, EmissionType},
        filter::FilterExec,
        limit::LocalLimitExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::executor::block_on;
use parquet::{
    arrow::ArrowWriter,
    file::{metadata::SortingColumn, properties::WriterProperties},
};
use proptest::prelude::*;
use query_planning::mode::StatisticalAggregateExecutionMode;
use tokio::runtime::Builder;

mod bounded_join;
mod mixed_fallback;

mod support;
#[path = "../support/support_core.rs"]
mod support_core;
#[path = "../support/tpch_session.rs"]
mod tpch_session;

use support::*;

fn test_gpu_backend() -> &'static AlgeonGpuBackend {
    static BACKEND: OnceLock<AlgeonGpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        let profile = AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 });
        let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
        AlgeonGpuBackend::builder()
            .with_native_execution_config(AlgeonNativeExecutionConfig::new(
                SourceChunking::try_new(64 * 1024 * 1024, 2, 3, 1).expect("valid source chunking"),
            ))
            .devices([device])
            .build()
            .expect("native execution tests require a backend-owned GPU")
    })
}

fn bounded_config() -> AlgeonNativeExecutionConfig {
    test_gpu_backend().optimizer_config().native_execution_config().clone()
}

fn native_optimizer_config() -> AlgeonNativeOptimizerConfig {
    support::native_optimizer_config().with_native_execution_config(bounded_config())
}

fn installed_physical_optimizer_rules(
    config: AlgeonNativeOptimizerConfig,
) -> Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>> {
    SessionStateBuilder::new().with_algeon_native(config).build().physical_optimizers().to_vec()
}

fn installed_native_stage<'a>(
    rules: &'a [Arc<dyn PhysicalOptimizerRule + Send + Sync>],
    name: &str,
) -> &'a AlgeonNativePhysicalOptimizerRule {
    let rule = rules
        .iter()
        .find(|rule| rule.name() == name)
        .unwrap_or_else(|| panic!("native optimizer rules should contain {name}"));
    (rule.as_ref() as &dyn std::any::Any)
        .downcast_ref::<AlgeonNativePhysicalOptimizerRule>()
        .unwrap_or_else(|| panic!("{name} should be an AlgeonNativePhysicalOptimizerRule"))
}

fn native_optimizer_rule_rewrites_supported_physical_plan() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-optimizer");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    // The admission root is a visible node over the native fragment.
    assert_eq!(optimized.name(), "NativeAdmissionExec");
    assert_eq!(optimized.children().len(), 1);
    assert!(displayable(optimized.as_ref()).indent(false).to_string().starts_with("NativeAdmissionExec\n"));
    assert!(below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some());
    let diagnostics = diagnostics.expect("native optimizer should retain diagnostics");
    let report = &diagnostics.report;
    assert_eq!(report.selected_count(), 1);
    assert!(report.query_id().is_some_and(|query_id| query_id.as_str().starts_with("q-")));
    match &report.outcomes()[0] {
        CandidateOutcome::Selected { candidate } => {
            assert!(candidate.plan_node_id.is_some(), "rewritten candidate should carry native plan-node evidence");
        },
        other => panic!("expected rewritten candidate, got {other:?}"),
    }
    assert_eq!(diagnostics.source_diagnostics.len(), 1);
    // A plan without a join is decided at planning like any other candidate.
    assert_eq!(diagnostics.capability_contract.accepted_candidates.len(), 1);
    assert!(diagnostics.capability_contract.rejected_candidates.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_ilike_filter_over_native_scan_fragment_is_mixed_and_idempotent() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ilike-fragment");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let plan = ilike_filter_plan(&path);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();
    let diagnostics = diagnostics.expect("native optimizer should retain diagnostics");
    let report = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
        "ilike_fragment",
        Arc::clone(&optimized),
        diagnostics,
    )]);

    // The root filter's ILIKE has no native expression; the scan below it is
    // the one native fragment.
    let [root, fragment] = report.rows.as_slice() else {
        panic!("expected the root rejection and the fragment: {:?}", report.rows);
    };
    assert_eq!(root.outcome, Some(CandidateOutcomeKind::NotSupported));
    assert_eq!(root.reason_code.as_deref(), Some("unsupported_expression"));
    assert_eq!(fragment.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(fragment.native_candidate_shape.as_deref(), Some("LocalParquetScan"));
    assert_eq!(fragment.final_disposition, Some(FinalPlanDisposition::Mixed));
    assert_eq!(fragment.native_plan_fragment_count, Some(1));
    assert_eq!(fragment.host_datafusion_boundary_count, Some(1));
    assert!(fragment.source_diagnostics.iter().any(|source| source.direct_device_read));

    let filter = below_native_admission(&optimized)
        .downcast_ref::<FilterExec>()
        .expect("the CPU root filter stays on DataFusion");
    let native = filter.input().downcast_ref::<AlgeonNativeExec>().expect("the scan is replaced by a native fragment");
    assert_mixed_boundary_properties(filter, native);

    let (repeated, repeat_diagnostics) =
        rule.optimize_with_diagnostics(Arc::clone(&optimized), &ConfigOptions::default()).unwrap();
    assert!(Arc::ptr_eq(&optimized, &repeated), "a plan holding a fragment is an optimizer fixed point");
    assert!(
        repeat_diagnostics.is_none(),
        "the rule records nothing for a plan it already selected, so idempotence keeps the fragment report"
    );

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_ilike_fragment_keeps_a_transparent_cooperative_wrapper() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ilike-fragment-cooperative");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(ilike_filter_plan(&path)));
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let cooperative = below_native_admission(&optimized)
        .downcast_ref::<CooperativeExec>()
        .expect("the transparent wrapper should remain in the final plan");
    let filter = cooperative
        .input()
        .downcast_ref::<FilterExec>()
        .expect("cooperative wrapper must still contain the root filter");
    let native = filter.input().downcast_ref::<AlgeonNativeExec>().expect("filter child must be the one native leaf");
    assert_mixed_boundary_properties(filter, native);
    assert_eq!(optimized.properties().emission_type, EmissionType::Final);
    assert_eq!(classify_final_plan(optimized.as_ref()), FinalPlanDisposition::Mixed);

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_ilike_fragment_reserves_host_output_before_d2h() {
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;

    let dir = unique_temp_dir("algeon-datafusion-native-exec-ilike-fragment-host-budget");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let backend = test_gpu_backend();
    let config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(
        backend.optimizer_config().native_execution_config().clone().with_host_output_memory_pool_accounting(true),
    );
    let rule = AlgeonNativePhysicalOptimizerRule::new(config);
    let plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let runtime_env = Arc::new(RuntimeEnvBuilder::new().with_memory_limit(1, 1.0).build().unwrap());
    let task_context = Arc::new(TaskContext::default().with_runtime(Arc::clone(&runtime_env)));

    let error = block_on(collect(Arc::clone(&plan), task_context))
        .expect_err("the optimizer-built fragment must reserve its host Arrow output before D2H");

    assert!(matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)), "unexpected error: {error:?}");
    assert!(error.to_string().contains("AlgeonNativeExec[host-output]"), "unexpected error: {error}");
    assert_eq!(
        mixed_boundary_native_leaf(below_native_admission(&plan).as_ref())
            .metrics()
            .unwrap()
            .sum_by_name("device_to_host_materializations")
            .unwrap()
            .as_usize(),
        0,
        "host-memory denial must occur before the mixed native leaf copies data from device to host"
    );
    assert_eq!(runtime_env.memory_pool.reserved(), 0);

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_ilike_fragment_uses_one_backend_admission_and_observation_owner() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ilike-fragment-admission");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let backend = test_gpu_backend();
    let rule = AlgeonNativePhysicalOptimizerRule::new(backend.optimizer_config().clone());
    let plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let native_leaf = mixed_boundary_native_leaf(below_native_admission(&plan).as_ref());

    let batches = block_on(collect(Arc::clone(&plan), Arc::new(TaskContext::default()))).unwrap();

    assert_eq!(ilike_rows(&batches), vec![(1, Some("Alpha".to_string())), (3, Some("ALMANAC".to_string()))]);
    let metrics =
        native_leaf.metrics().expect("the selected native leaf must expose the shared observation projection");
    assert_eq!(metrics.sum_by_name("admission_sequence_observed").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("admission_wait_time_observed").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("query_grant_bytes_observed").unwrap().as_usize(), 1);
    assert_eq!(metrics.sum_by_name("device_to_host_materializations").unwrap().as_usize(), 1);
    let deadline = Instant::now() + Duration::from_secs(5);
    while backend.diagnostics().unwrap().admission.active_queries != 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let snapshot = backend.diagnostics().unwrap().admission;
    assert_eq!(snapshot.active_queries, 0, "the fragment's admission ticket must release after its stream completes");
    assert_eq!(snapshot.queued_queries, 0);

    fs::remove_dir_all(dir).unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 16,
        .. ProptestConfig::default()
    })]

    /// An ILIKE filter over a native scan fragment keeps its contract across
    /// nullable UTF-8 view inputs and deterministic ASCII prefixes.
    #[test]
    fn native_optimizer_property_based_ilike_fragment_preserves_contract(
        mut labels in prop::collection::vec(prop::option::of("[A-Za-z]{0,12}"), 1..16),
        prefix in "[A-Za-z]{0,5}",
    ) {
        // Retain a null in every generated fixture while still varying
        // nullable input values in the generated rows.
        labels.push(None);
        let dir = unique_temp_dir("algeon-datafusion-native-exec-property-based-ilike");
        let path = dir.join("input.parquet");
        write_ilike_parquet_file_with_labels(&path, &labels);
        let pattern = format!("{prefix}%");

        let cpu = block_on(collect(
            ilike_filter_plan_with_pattern(&path, pattern.clone()),
            Arc::new(TaskContext::default()),
        ))
        .unwrap();
        let config = native_optimizer_config();
        let native_rule = AlgeonNativePhysicalOptimizerRule::new(config.clone());
        let (planned, diagnostics) = native_rule
            .optimize_with_diagnostics(ilike_filter_plan_with_pattern(&path, pattern.clone()), &ConfigOptions::default())
            .unwrap();
        let diagnostics = diagnostics.expect("the fragment report should be retained");
        let report = NativeDataFusionPlanningReport::collect([NativeDataFusionPlanningWorkloadItem::new(
            "generated_ilike_fragment",
            Arc::clone(&planned),
            diagnostics,
        )]);

        prop_assert_eq!(report.rows.len(), 2);
        prop_assert_eq!(report.rows[0].outcome, Some(CandidateOutcomeKind::NotSupported));
        let row = &report.rows[1];
        prop_assert_eq!(row.outcome, Some(CandidateOutcomeKind::Selected));
        prop_assert_eq!(row.final_disposition, Some(FinalPlanDisposition::Mixed));
        prop_assert_eq!(row.native_plan_fragment_count, Some(1));
        prop_assert_eq!(row.host_datafusion_boundary_count, Some(1));
        let filter = below_native_admission(&planned).downcast_ref::<FilterExec>().expect("the CPU root filter stays on DataFusion");
        let native = filter.input().downcast_ref::<AlgeonNativeExec>().expect("the scan is replaced by a native fragment");
        assert_mixed_boundary_properties(filter, native);

        let runtime = Builder::new_current_thread().enable_all().build().unwrap();
        let mixed = runtime.block_on(collect(Arc::clone(&planned), Arc::new(TaskContext::default()))).unwrap();
        prop_assert_eq!(ilike_rows(&mixed), ilike_rows(&cpu));

        let (repeated, repeat_diagnostics) =
            native_rule.optimize_with_diagnostics(Arc::clone(&planned), &ConfigOptions::default()).unwrap();
        prop_assert!(Arc::ptr_eq(&planned, &repeated));
        prop_assert!(
            repeat_diagnostics.is_none(),
            "the rule records nothing for a plan it already selected, so idempotence keeps the selected report"
        );

        let reject_rule = AlgeonNativePhysicalOptimizerRule::new(
            config.with_execution_mode(AlgeonGpuExecutionMode::NativeRequired),
        );
        let error = reject_rule
            .optimize(ilike_filter_plan_with_pattern(&path, pattern), &ConfigOptions::default())
            .expect_err("NativeRequired must reject a plan whose root filter stays on DataFusion");
        let error = adapter_error(&error);
        prop_assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        prop_assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("datafusion"));

        fs::remove_dir_all(dir).unwrap();
    }
}

fn mixed_boundary_native_leaf(plan: &dyn ExecutionPlan) -> &AlgeonNativeExec {
    let filter = plan.downcast_ref::<FilterExec>().expect("the CPU root filter stays on DataFusion");
    filter.input().downcast_ref::<AlgeonNativeExec>().expect("the filter input is the native fragment")
}

fn assert_mixed_boundary_properties(filter: &FilterExec, native: &AlgeonNativeExec) {
    assert_eq!(filter.schema().as_ref(), native.schema().as_ref());
    assert!(matches!(filter.properties().output_partitioning(), Partitioning::UnknownPartitioning(1)));
    assert_eq!(filter.properties().emission_type, EmissionType::Final);
    assert_eq!(filter.properties().boundedness, Boundedness::Bounded);
    assert!(filter.properties().output_ordering().is_none());
    assert!(filter.properties().equivalence_properties().eq_group().is_empty());
    assert!(filter.properties().equivalence_properties().oeq_class().is_empty());
    assert!(filter.properties().equivalence_properties().constraints().is_empty());

    assert!(matches!(native.properties().output_partitioning(), Partitioning::UnknownPartitioning(1)));
    assert_eq!(native.properties().emission_type, EmissionType::Final);
    assert_eq!(native.properties().boundedness, Boundedness::Bounded);
    assert!(native.properties().output_ordering().is_none());
    assert!(native.properties().equivalence_properties().eq_group().is_empty());
    assert!(native.properties().equivalence_properties().oeq_class().is_empty());
    assert!(native.properties().equivalence_properties().constraints().is_empty());
}

fn ilike_rows(batches: &[RecordBatch]) -> Vec<(i32, Option<String>)> {
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

fn native_optimizer_rule_propagates_declared_scan_output_ordering_to_wrapper_properties() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ordered-scan-properties");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let plan = parquet_scan_with_declared_output_ordering(&path);
    let expected_ordering = plan.properties().output_ordering().cloned();
    assert!(expected_ordering.is_some());
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    assert!(below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some());
    assert_eq!(optimized.properties().output_ordering().cloned(), expected_ordering);
    let diagnostics = diagnostics.expect("native optimizer should record a planning report");
    let report = &diagnostics.report;
    assert_eq!(report.selected_count(), 1);

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_reestablishes_declared_scan_ordering_through_aggregate() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ordered-aggregate-composition");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let plan = grouped_sum_aggregate_plan(scan);
    let expected_ordering = plan.properties().output_ordering().cloned();
    assert!(expected_ordering.is_some());
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    assert!(below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some());
    assert_eq!(optimized.properties().output_ordering().cloned(), expected_ordering);
    let diagnostics = diagnostics.expect("native optimizer should record a planning report");
    let report = &diagnostics.report;
    assert_eq!(report.selected_count(), 1, "outcomes={:?}", report.outcomes());
    let CandidateOutcome::Selected { candidate } = &report.outcomes()[0] else {
        panic!("ordered aggregate should be reported as rewritten");
    };
    assert!(
        candidate.shape.to_string().starts_with("SortExec(AggregateExec("),
        "post-optimizer appended declared-ordering sort should be visible in candidate shape: {}",
        candidate.shape
    );
    assert!(!report.outcomes().iter().any(|outcome| {
        matches!(
            outcome,
            CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeOrderedOutputUnsupported, .. }
        )
    }));

    fs::remove_dir_all(dir).unwrap();
}

fn native_execution_reestablished_declared_scan_ordering_is_sorted() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ordered-aggregate-output");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let plan = grouped_sum_aggregate_plan(scan);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some());
    let batches = block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(int32_array(&batches[0], 0).values().as_ref(), &[1, 2]);

    fs::remove_dir_all(dir).unwrap();
}

fn native_execution_sort_preserving_merge_matches_datafusion_sorted_input() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-sort-preserving-merge");
    write_sorted_parquet_file(&dir.join("first.parquet"));
    write_sorted_parquet_file(&dir.join("second.parquet"));
    let scan = parquet_scan_with_target_partitions(&dir, 2);
    assert_eq!(scan.properties().output_partitioning().partition_count(), 2);
    let ordering = LexOrdering::new([PhysicalSortExpr::new(
        col("key", &scan.schema()).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .expect("SPM test ordering should be non-empty");
    let sorted: Arc<dyn ExecutionPlan> =
        Arc::new(SortExec::new(ordering.clone(), scan).with_preserve_partitioning(true));
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortPreservingMergeExec::new(ordering, sorted));
    let expected = block_on(collect(Arc::clone(&plan), Arc::new(TaskContext::default()))).unwrap();
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some());
    let actual = block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(actual.len(), expected.len());
    assert_eq!(int32_array(&actual[0], 0).values().as_ref(), int32_array(&expected[0], 0).values().as_ref());
    assert_eq!(int32_array(&actual[0], 0).values().as_ref(), &[1, 1, 1, 1, 2, 2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_uses_configured_optimizer_config_at_optimize_time() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-optimizer-config");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let config = support::native_optimizer_config();
    let rule = AlgeonNativePhysicalOptimizerRule::new(config);
    let options = ConfigOptions::default();

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &options).unwrap();

    let native =
        optimized.downcast_ref::<AlgeonNativeExec>().expect("supported plan should optimize to AlgeonNativeExec");
    let display = displayable(native).one_line().to_string();
    assert!(display.contains("native_pipeline_execution_mode=bounded"), "{display}");
    let diagnostics = diagnostics.expect("native optimizer should record a planning report");
    let report = &diagnostics.report;
    assert_eq!(report.selected_count(), 1);

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_rule_runs_the_filtered_scan_below_an_unsupported_projection_as_a_fragment() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-optimizer-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    assert!(below_native_admission(&optimized).downcast_ref::<LocalLimitExec>().is_some());
    assert_eq!(classify_final_plan(optimized.as_ref()), FinalPlanDisposition::Mixed);
    let diagnostics = diagnostics.expect("native optimizer should record a planning report");
    let report = &diagnostics.report;
    assert!(report.query_id().is_some_and(|query_id| query_id.as_str().starts_with("q-")));
    let [
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. },
        CandidateOutcome::Selected { candidate },
    ] = report.outcomes()
    else {
        panic!("expected the root rejection and one fragment: {:?}", report.outcomes());
    };
    assert_eq!(candidate.shape.to_string(), "FilterExec(LocalParquetScan)");

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_rule_obeys_no_datafusion_cpu_requirement_for_rejected_plan() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-no-datafusion-cpu-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let config = AlgeonNativeOptimizerConfig::default()
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");
    let error = final_native
        .optimize(plan, &ConfigOptions::default())
        .expect_err("the final native rule must reject a rejected native candidate's CPU plan");
    let adapter_error = error.find_algeon_error().expect("final-plan rejection must preserve the typed Algeon error");

    assert_eq!(adapter_error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(adapter_error.fact_value("requirement"), Some("no_datafusion_cpu"));

    fs::remove_dir_all(dir).unwrap();
}

fn final_plan_requirement_reports_first_unsupported_projected_column() {
    let dir = unique_temp_dir("algeon-datafusion-native-unsupported-list-detail");
    let path = dir.join("input.parquet");
    let child = Arc::new(Field::new("element", DataType::Decimal128(10, 2), true));
    let first = ListArray::new(
        Arc::clone(&child),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2])),
        Arc::new(Decimal128Array::from(vec![100_i128, 200]).with_precision_and_scale(10, 2).unwrap()),
        None,
    );
    let second = ListArray::new(
        Arc::clone(&child),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2])),
        Arc::new(Decimal128Array::from(vec![300_i128, 400]).with_precision_and_scale(10, 2).unwrap()),
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("first_unsupported", DataType::List(Arc::clone(&child)), true),
        Field::new("second_unsupported", DataType::List(child), true),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(first), Arc::new(second)]).unwrap();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let config = AlgeonNativeOptimizerConfig::default()
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let native_rule = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");
    let error = native_rule
        .optimize(parquet_scan(&path), &ConfigOptions::default())
        .expect_err("unsupported List<Decimal128> must fail the strict GPU requirement");
    let error = adapter_error(&error);

    assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(error.message(), "GPU execution does not support a projected column with an unsupported Arrow type");
    assert_eq!(error.fact_value("reason"), Some("unsupported_type"));
    assert_eq!(
        error.client_detail(),
        Some(&ClientDetail::UnsupportedColumn {
            name: "first_unsupported".to_string(),
            data_type: "List(Decimal128(10, 2))".to_string(),
            role: "projected",
        })
    );

    fs::remove_dir_all(dir).unwrap();
}

fn no_datafusion_cpu_count_only_probe_miss_reaches_final_native_optimizer() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-no-datafusion-cpu-probe-miss");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let config = AlgeonNativeOptimizerConfig::default()
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let count_only_probe = installed_native_stage(&rules, "algeon_datafusion_native_count_only");
    let final_native = installed_native_stage(&rules, "algeon_datafusion_native");

    let (after_probe, probe_diagnostics) = count_only_probe
        .optimize_with_diagnostics(plan, &ConfigOptions::default())
        .expect("a non-count plan must not be rejected by the NoDataFusionCpu count-only probe");
    assert!(
        below_native_admission(&after_probe).downcast_ref::<AlgeonNativeExec>().is_none(),
        "the count-only probe should decline an analytical SUM plan"
    );
    let probe_report = probe_diagnostics.expect("count-only probe should record its own rejection evidence").report;
    assert_eq!(probe_report.selected_count(), 0);
    assert_eq!(probe_report.not_selected_count(), 1);

    let (optimized, final_diagnostics) = final_native
        .optimize_with_diagnostics(after_probe, &ConfigOptions::default())
        .expect("the final native optimizer must run after a count-only probe miss");
    assert!(
        below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some(),
        "the final native optimizer should rewrite the supported analytical plan"
    );
    let final_report =
        final_diagnostics.expect("final optimizer should record separate native rewrite evidence").report;
    assert_eq!(final_report.selected_count(), 1);
    assert_eq!(final_report.not_selected_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

fn no_datafusion_cpu_final_rejection_names_final_stage_reason_not_count_only_probe_reason() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-no-datafusion-cpu-final-rejection-evidence");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let rules = installed_physical_optimizer_rules(
        AlgeonNativeOptimizerConfig::default()
            .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
            .with_native_execution_config(bounded_config()),
    );
    let count_only_probe = installed_native_stage(&rules, "algeon_datafusion_native_count_only");
    let final_native = installed_native_stage(&rules, "algeon_datafusion_native");

    let (after_probe, probe_diagnostics) =
        count_only_probe.optimize_with_diagnostics(plan, &ConfigOptions::default()).expect(
            "a normal count-only probe miss must remain non-terminal before NoDataFusionCpu final-plan enforcement",
        );
    let probe_report = probe_diagnostics.expect("count-only probe should retain its own evidence").report;
    assert!(matches!(
        probe_report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedOperator, .. }
    ));

    let error = final_native
        .optimize_with_diagnostics(after_probe, &ConfigOptions::default())
        .expect_err("the final native optimizer must reject the completed DataFusion plan");
    let error = adapter_error(&error);
    assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(
        error.fact_value("capability_reason"),
        Some(CapabilityReason::UnsupportedExpression.code()),
        "the rejection must carry the final stage's reason, not the count-only probe's"
    );

    fs::remove_dir_all(dir).unwrap();
}

fn no_datafusion_cpu_count_only_probe_retains_its_native_rewrite() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-no-datafusion-cpu-count-probe");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = count_only_sort_limit_plan(parquet_scan(&path));
    let config = AlgeonNativeOptimizerConfig::default()
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let count_only_probe = installed_native_stage(&rules, "algeon_datafusion_native_count_only");
    let final_native = installed_native_stage(&rules, "algeon_datafusion_native");

    let (after_probe, probe_diagnostics) = count_only_probe
        .optimize_with_diagnostics(plan, &ConfigOptions::default())
        .expect("NoDataFusionCpu count-only probe should retain supported native count rewrites");
    assert!(below_native_admission(&after_probe).downcast_ref::<AlgeonNativeExec>().is_some());
    let probe_report = probe_diagnostics.expect("count-only probe should record its rewrite evidence").report;
    assert_eq!(probe_report.selected_count(), 1);
    assert_eq!(probe_report.not_selected_count(), 0);
    let after_final = final_native
        .optimize(after_probe, &ConfigOptions::default())
        .expect("the final native optimizer should preserve a count-only native rewrite");
    assert!(below_native_admission(&after_final).downcast_ref::<AlgeonNativeExec>().is_some());

    fs::remove_dir_all(dir).unwrap();
}

fn no_datafusion_cpu_count_only_probe_rewrites_materialized_memory_table() {
    let schema = Arc::new(Schema::new(vec![Field::new("src", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from_iter_values(0..65))])
        .expect("materialized workspace batch");
    let source =
        MemorySourceConfig::try_new(&[vec![batch]], Arc::clone(&schema), None).expect("materialized workspace source");
    let scan = DataSourceExec::from_data_source(source) as Arc<dyn ExecutionPlan>;
    let plan = count_only_plan(scan);
    let rules = installed_physical_optimizer_rules(
        AlgeonNativeOptimizerConfig::default()
            .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
            .with_native_execution_config(bounded_config()),
    );
    let count_only_probe = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native_count_only")
        .expect("native optimizer rules should contain the count-only probe");

    let optimized = count_only_probe
        .optimize(plan, &ConfigOptions::default())
        .expect("materialized workspace COUNT(*) must remain GPU-only");

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
}

fn native_optimizer_rule_runs_the_supported_child_of_an_unsupported_parent_as_one_fragment() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-partial-host-arrow-boundary");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = partial_sort_parent_plan(aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus));
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) =
        rule.optimize_with_diagnostics(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert_eq!(classify_final_plan(optimized.as_ref()), FinalPlanDisposition::Mixed);
    let partial_sort = below_native_admission(&optimized);
    assert_eq!(partial_sort.name(), "PartialSortExec");
    assert!(
        partial_sort.children()[0].downcast_ref::<AlgeonNativeExec>().is_some(),
        "the parent reads one native fragment"
    );
    let diagnostics = diagnostics.expect("native optimizer should record a planning report");
    let report = &diagnostics.report;
    assert!(
        matches!(
            report.outcomes(),
            [
                CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedOperator, .. },
                CandidateOutcome::Selected { .. },
            ]
        ),
        "{:?}",
        report.outcomes()
    );
    assert_eq!(diagnostics.source_diagnostics.len(), 1, "the fragment reports its source");

    fs::remove_dir_all(dir).unwrap();
}

fn final_plan_requirement_rejects_unsupported_relational_parent_after_datafusion_disposition() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-composition-no-datafusion-cpu");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = partial_sort_parent_plan(aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus));
    let config = AlgeonNativeOptimizerConfig::default()
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let final_native = installed_native_stage(&rules, "algeon_datafusion_native");

    let error = final_native
        .optimize_with_diagnostics(plan, &ConfigOptions::default())
        .expect_err("the final native rule must reject the completed DataFusion plan");
    let adapter_error = adapter_error(&error);
    assert_eq!(adapter_error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(adapter_error.fact_value("observed_final_plan_disposition"), Some("datafusion"));
    assert_eq!(adapter_error.fact_value("terminal_plan_disposition"), Some("rejected"));
    assert_eq!(
        adapter_error.fact_value("capability_reason"),
        Some(CapabilityReason::UnsupportedOperator.code()),
        "the rejection must name the relational lowering miss that left the plan on DataFusion"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_exec_cases_1() {
    testing::run_named_cases(&[
        (
            "native_optimizer_rule_executes_distinct_aggregate_over_anti_join_natively",
            native_optimizer_rule_executes_distinct_aggregate_over_anti_join_natively,
        ),
        (
            "native_optimizer_admits_topk_grouped_aggregate_left_join_under_bounded_contract",
            native_optimizer_admits_topk_grouped_aggregate_left_join_under_bounded_contract,
        ),
        (
            "native_optimizer_rule_rewrites_precision_sensitive_stddev_sort_subtree_after_order_contract",
            native_optimizer_rule_rewrites_precision_sensitive_stddev_sort_subtree_after_order_contract,
        ),
        (
            "native_optimizer_rule_executes_try_cast_string_to_integer_with_null_on_failure",
            native_optimizer_rule_executes_try_cast_string_to_integer_with_null_on_failure,
        ),
        (
            "native_optimizer_rule_rewrites_supported_physical_plan",
            native_optimizer_rule_rewrites_supported_physical_plan,
        ),
        (
            "native_float32_casts_match_arrow_cpu_casts_bit_for_bit",
            native_float32_casts_match_arrow_cpu_casts_bit_for_bit,
        ),
    ]);
}

#[test]
fn native_exec_cases_2() {
    testing::run_named_cases(&[
        (
            "rank_over_computed_window_keys_runs_natively_and_matches_datafusion",
            rank_over_computed_window_keys_runs_natively_and_matches_datafusion,
        ),
        (
            "row_number_beside_a_partition_aggregate_numbers_each_partition_from_one",
            row_number_beside_a_partition_aggregate_numbers_each_partition_from_one,
        ),
        (
            "rank_and_row_number_accept_an_order_key_that_is_also_the_partition_key",
            rank_and_row_number_accept_an_order_key_that_is_also_the_partition_key,
        ),
        (
            "partition_aggregate_window_output_keeps_the_order_the_query_relies_on",
            partition_aggregate_window_output_keeps_the_order_the_query_relies_on,
        ),
        (
            "windows_sharing_a_sort_keep_the_order_the_query_relies_on",
            windows_sharing_a_sort_keep_the_order_the_query_relies_on,
        ),
        (
            "native_optimizer_ilike_filter_over_native_scan_fragment_is_mixed_and_idempotent",
            native_optimizer_ilike_filter_over_native_scan_fragment_is_mixed_and_idempotent,
        ),
    ]);
}

#[test]
fn native_exec_cases_3() {
    testing::run_named_cases(&[
        (
            "dense_rank_and_percent_rank_windows_stay_in_datafusion",
            dense_rank_and_percent_rank_windows_stay_in_datafusion,
        ),
        (
            "native_preferred_keeps_an_admission_refusal_on_datafusion",
            native_preferred_keeps_an_admission_refusal_on_datafusion,
        ),
        (
            "native_required_rejection_names_the_admission_refusal",
            native_required_rejection_names_the_admission_refusal,
        ),
        (
            "native_optimizer_ilike_fragment_keeps_a_transparent_cooperative_wrapper",
            native_optimizer_ilike_fragment_keeps_a_transparent_cooperative_wrapper,
        ),
        (
            "native_optimizer_ilike_fragment_reserves_host_output_before_d2h",
            native_optimizer_ilike_fragment_reserves_host_output_before_d2h,
        ),
        (
            "native_optimizer_ilike_fragment_uses_one_backend_admission_and_observation_owner",
            native_optimizer_ilike_fragment_uses_one_backend_admission_and_observation_owner,
        ),
    ]);
}

#[test]
fn native_exec_cases_4() {
    testing::run_named_cases(&[
        (
            "native_optimizer_rule_propagates_declared_scan_output_ordering_to_wrapper_properties",
            native_optimizer_rule_propagates_declared_scan_output_ordering_to_wrapper_properties,
        ),
        (
            "native_optimizer_reestablishes_declared_scan_ordering_through_aggregate",
            native_optimizer_reestablishes_declared_scan_ordering_through_aggregate,
        ),
        (
            "native_execution_reestablished_declared_scan_ordering_is_sorted",
            native_execution_reestablished_declared_scan_ordering_is_sorted,
        ),
        (
            "native_execution_sort_preserving_merge_matches_datafusion_sorted_input",
            native_execution_sort_preserving_merge_matches_datafusion_sorted_input,
        ),
        (
            "native_optimizer_rule_runs_the_filtered_scan_below_an_unsupported_projection_as_a_fragment",
            native_optimizer_rule_runs_the_filtered_scan_below_an_unsupported_projection_as_a_fragment,
        ),
    ]);
}

#[test]
fn native_exec_cases_5() {
    testing::run_named_cases(&[
        (
            "native_optimizer_rule_obeys_no_datafusion_cpu_requirement_for_rejected_plan",
            native_optimizer_rule_obeys_no_datafusion_cpu_requirement_for_rejected_plan,
        ),
        (
            "final_plan_requirement_reports_first_unsupported_projected_column",
            final_plan_requirement_reports_first_unsupported_projected_column,
        ),
        (
            "no_datafusion_cpu_count_only_probe_miss_reaches_final_native_optimizer",
            no_datafusion_cpu_count_only_probe_miss_reaches_final_native_optimizer,
        ),
    ]);
}

#[test]
fn native_exec_cases_6() {
    testing::run_named_cases(&[
        (
            "no_datafusion_cpu_final_rejection_names_final_stage_reason_not_count_only_probe_reason",
            no_datafusion_cpu_final_rejection_names_final_stage_reason_not_count_only_probe_reason,
        ),
        (
            "no_datafusion_cpu_count_only_probe_retains_its_native_rewrite",
            no_datafusion_cpu_count_only_probe_retains_its_native_rewrite,
        ),
        (
            "no_datafusion_cpu_count_only_probe_rewrites_materialized_memory_table",
            no_datafusion_cpu_count_only_probe_rewrites_materialized_memory_table,
        ),
        (
            "native_optimizer_rule_runs_the_supported_child_of_an_unsupported_parent_as_one_fragment",
            native_optimizer_rule_runs_the_supported_child_of_an_unsupported_parent_as_one_fragment,
        ),
        (
            "final_plan_requirement_rejects_unsupported_relational_parent_after_datafusion_disposition",
            final_plan_requirement_rejects_unsupported_relational_parent_after_datafusion_disposition,
        ),
    ]);
}

fn native_optimizer_rule_executes_distinct_aggregate_over_anti_join_natively() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-distinct-anti-composition");
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
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) =
        rule.optimize_with_diagnostics(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    // S3 admits scalar distinct aggregation over a keyed anti join through the
    // composed pipeline. The anti join executes with null-aware semantics, and
    // the scalar distinct count is exact.
    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some(),
        "distinct aggregate over anti join should execute natively:\n{optimized_tree}"
    );
    let diagnostics = diagnostics.expect("native optimizer should record a planning report");
    let report = &diagnostics.report;
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);

    // Row parity with the DataFusion baseline: no key survives the NOT IN.
    let batches = block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(int64_array(&batches[0], 0).value(0), 0);

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_admits_topk_grouped_aggregate_left_join_under_bounded_contract() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-topk-aggregate-left-join");
    let edges = dir.join("edges.parquet");
    let nodes = dir.join("nodes.parquet");
    write_join_left_parquet_file(&edges);
    write_join_right_parquet_file(&nodes);
    let plan = single_partition_sql_plan_with_parquet_tables(
        &[("edges", &edges), ("nodes", &nodes)],
        "WITH top_degrees AS ( \
             SELECT key AS vertex_id, COUNT(*) AS out_degree \
             FROM edges \
             GROUP BY key \
             ORDER BY out_degree DESC \
             LIMIT 30 \
         ) \
         SELECT d.vertex_id, n.r_payload, d.out_degree \
         FROM top_degrees d \
         LEFT JOIN nodes n ON d.vertex_id = n.key \
         ORDER BY d.out_degree DESC \
         LIMIT 30",
    );
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    let diagnostics = diagnostics.expect("native optimizer should record a planning report");
    let report = &diagnostics.report;
    assert!(
        below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some(),
        "outcomes={:?}; plan={}",
        report.outcomes(),
        displayable(optimized.as_ref()).indent(false)
    );
    assert_eq!(report.selected_count(), 1, "outcomes={:?}", report.outcomes());
    assert_eq!(report.not_selected_count(), 0, "outcomes={:?}", report.outcomes());

    fs::remove_dir_all(dir).unwrap();
}

fn native_optimizer_rule_rewrites_precision_sensitive_stddev_sort_subtree_after_order_contract() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-stddev-sort-cpu-boundary");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = stddev_sort_sql_plan(&path);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    assert!(below_native_admission(&optimized).downcast_ref::<AlgeonNativeExec>().is_some());
    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        optimized_tree.contains("AlgeonNativeExec"),
        "stddev subtree should be rewritten once native order tracking is available:\n{optimized_tree}"
    );
    let diagnostics = diagnostics.expect("native optimizer should record a planning report");
    let report = &diagnostics.report;
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_is_inserted_before_final_validation() {
    let rules = installed_physical_optimizer_rules(AlgeonNativeOptimizerConfig::default());
    let names = rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();

    let count_only_position = names.iter().position(|name| *name == "algeon_datafusion_native_count_only").unwrap();
    let aggregate_statistics_position = names.iter().position(|name| *name == "aggregate_statistics").unwrap();
    let native_position = names.iter().position(|name| *name == "algeon_datafusion_native").unwrap();
    let sanity_position = names.iter().position(|name| *name == "SanityCheckPlan").unwrap();
    assert!(count_only_position < aggregate_statistics_position);
    assert!(aggregate_statistics_position < native_position);
    assert!(native_position < sanity_position);

    let default_rules = installed_physical_optimizer_rules(AlgeonNativeOptimizerConfig::default());
    let default_names = default_rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();
    assert!(default_names.contains(&"algeon_datafusion_native"));
    assert!(default_names.contains(&"algeon_datafusion_native_count_only"));
}

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
    let dir = unique_temp_dir("algeon-datafusion-native-exec-try-cast");
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

    let state = test_gpu_backend()
        .install_on(
            SessionStateBuilder::new()
                .with_default_features()
                .with_config(SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false)),
        )
        .unwrap()
        .build();
    let ctx = SessionContext::new_with_state(state);
    block_on(ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();

    let df = block_on(ctx.sql("SELECT id, TRY_CAST(name AS INT) AS maybe_int FROM bench ORDER BY id")).unwrap();
    let plan = block_on(df.create_physical_plan()).unwrap();
    let optimized_tree = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        optimized_tree.contains("AlgeonNativeExec"),
        "TRY_CAST(string -> int) should lower into a native execution node:\n{optimized_tree}"
    );

    let batches: DfResult<Vec<RecordBatch>> = block_on(collect(plan, Arc::new(TaskContext::default())));
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

/// Native `CAST(Decimal128 AS REAL)` and `CAST(DOUBLE AS REAL)` must equal
/// Arrow's CPU casts bit for bit. Arrow rounds a decimal to `Float64` and then
/// to `Float32`; the first two decimals sit within half a `Float64` ulp of a
/// `Float32` midpoint, so a single correctly rounded cast to `Float32` picks
/// the other neighbor.
fn native_float32_casts_match_arrow_cpu_casts_bit_for_bit() {
    let decimals: Vec<i128> = [16_777_217_000_000_001, 16_777_218_999_999_999, -16_777_217_000_000_001]
        .into_iter()
        .chain((0..512_i128).map(|i| (i * 7_919 + 1) * 1_000_003_i128.pow(2) * if i % 2 == 0 { 1 } else { -1 }))
        .collect();
    let doubles: Vec<f64> = [16_777_217.0, 0.1, 1e39, -1e39, 1e-40, f64::MAX, f64::MIN_POSITIVE, -0.0, f64::NAN]
        .into_iter()
        .chain((0..decimals.len() - 9).map(|i| (i as f64 + 0.5).powf(3.3) / 7.0 * if i % 2 == 0 { 1.0 } else { -1.0 }))
        .collect();
    let dir = unique_temp_dir("algeon-datafusion-native-exec-float32-casts");
    let path = dir.join("input.parquet");
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("d", DataType::Decimal128(38, 9), false),
            Field::new("f", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(0..decimals.len() as i32)),
                Arc::new(Decimal128Array::from(decimals).with_precision_and_scale(38, 9).unwrap()),
                Arc::new(Float64Array::from(doubles)),
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let config = || SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
    let native = SessionContext::new_with_state(
        test_gpu_backend()
            .install_on(SessionStateBuilder::new().with_default_features().with_config(config()))
            .unwrap()
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    let query = "SELECT id, CAST(d AS REAL) AS d32, CAST(f AS REAL) AS f32 FROM input ORDER BY id";
    let bits = |ctx: &SessionContext, expect_native: bool| {
        block_on(ctx.register_parquet("input", path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
        let plan = block_on(block_on(ctx.sql(query)).unwrap().create_physical_plan()).unwrap();
        let tree = displayable(plan.as_ref()).indent(false).to_string();
        assert_eq!(tree.contains("AlgeonNativeExec") && !tree.contains("ProjectionExec"), expect_native, "{tree}");
        let mut rows = Vec::new();
        for batch in block_on(collect(plan, Arc::new(TaskContext::default()))).unwrap() {
            let column = |index: usize| batch.column(index).as_any().downcast_ref::<Float32Array>().unwrap().clone();
            let (d32, f32) = (column(1), column(2));
            rows.extend((0..batch.num_rows()).map(|row| (d32.value(row).to_bits(), f32.value(row).to_bits())));
        }
        rows
    };
    let native_bits = bits(&native, true);
    let cpu_bits = bits(&cpu, false);
    assert_eq!(native_bits.len(), cpu_bits.len());
    for (row, (native, cpu)) in native_bits.iter().zip(&cpu_bits).enumerate() {
        assert_eq!(
            native,
            cpu,
            "row {row}: native (d32, f32) = ({}, {}), Arrow = ({}, {})",
            f32::from_bits(native.0),
            f32::from_bits(native.1),
            f32::from_bits(cpu.0),
            f32::from_bits(cpu.1)
        );
    }
    assert_eq!(f32::from_bits(cpu_bits[0].0), 16_777_216.0, "Arrow double-rounds the first decimal");

    fs::remove_dir_all(dir).unwrap();
}

/// A window whose partition and order keys are computed expressions runs
/// natively when the sort below it computed the same expressions: the window
/// reads the sort's key columns, so the sort's ordering covers the window.
fn rank_over_computed_window_keys_runs_natively_and_matches_datafusion() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-computed-window-keys");
    let path = dir.join("input.parquet");
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("key", DataType::Int32, false),
            Field::new("payload", DataType::Int32, false),
        ]));
        // The CASE key is NULL for keys 1 and 2; payload ties inside a
        // partition share a rank.
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
                Arc::new(Int32Array::from(vec![1, 3, 1, 3, 2, 4, 2, 1, 3, 4])),
                Arc::new(Int32Array::from(vec![30, 10, 10, 20, 7, 5, 7, 20, 10, 9])),
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let config = || SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
    let required = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(config())
            .with_algeon_native(native_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired))
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    for ctx in [&required, &cpu] {
        block_on(ctx.register_parquet("input", path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    }
    let query = "SELECT id, RANK() OVER (PARTITION BY key + 1, CASE WHEN key > 2 THEN key END \
                 ORDER BY payload * 2) AS r FROM input ORDER BY id";

    let plan = block_on(block_on(required.sql(query)).unwrap().create_physical_plan())
        .expect("NativeRequired plans the window natively");
    let rows = |batches: Vec<RecordBatch>| pretty_format_batches(&batches).unwrap().to_string();
    let actual = rows(block_on(collect(plan, Arc::new(TaskContext::default()))).unwrap());
    let expected = rows(block_on(block_on(cpu.sql(query)).unwrap().collect()).unwrap());
    assert_eq!(actual, expected);

    fs::remove_dir_all(dir).unwrap();
}

/// Writes rows whose partition keys interleave, with enough rows and
/// partitions that a hash join's output order differs from its input, and
/// returns a NativeRequired context and a DataFusion CPU context over them.
fn partition_window_contexts(prefix: &str) -> (std::path::PathBuf, SessionContext, SessionContext) {
    const ROWS: i32 = 20_000;
    let dir = unique_temp_dir(prefix);
    let path = dir.join("input.parquet");
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("p", DataType::Int32, false),
            Field::new("o", DataType::Int32, false),
            Field::new("x", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values((0..ROWS).map(|row| (row * 7919) % 211))),
                Arc::new(Int32Array::from_iter_values((0..ROWS).map(|row| (row * 31) % 97))),
                Arc::new(Int64Array::from_iter_values((0..ROWS).map(|row| i64::from(row % 1000)))),
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let config = || SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
    let required = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(config())
            .with_algeon_native(native_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired))
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    for ctx in [&required, &cpu] {
        block_on(ctx.register_parquet("input", path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    }
    (dir, required, cpu)
}

fn native_and_datafusion_rows(required: &SessionContext, cpu: &SessionContext, query: &str) -> (String, String) {
    let plan = block_on(block_on(required.sql(query)).unwrap().create_physical_plan())
        .expect("NativeRequired plans the window natively");
    let rows = |batches: Vec<RecordBatch>| pretty_format_batches(&batches).unwrap().to_string();
    let actual = rows(block_on(collect(plan, Arc::new(TaskContext::default()))).unwrap());
    let expected = rows(block_on(block_on(cpu.sql(query)).unwrap().collect()).unwrap());
    (actual, expected)
}

/// A partition aggregate and `ROW_NUMBER()` in one window node: the row
/// number runs after the aggregate, over partitions that must still be
/// contiguous, so each partition is numbered 1 to n.
fn row_number_beside_a_partition_aggregate_numbers_each_partition_from_one() {
    let (dir, required, cpu) = partition_window_contexts("algeon-datafusion-native-exec-window-row-number");
    let query = "SELECT p, s, rn FROM (SELECT p, SUM(x) OVER (PARTITION BY p) AS s, \
                 ROW_NUMBER() OVER (PARTITION BY p) AS rn FROM input) ORDER BY p, rn";
    let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
    assert_eq!(actual, expected);
    fs::remove_dir_all(dir).unwrap();
}

/// An order key that repeats a partition key orders nothing inside a
/// partition: every row of a partition is a peer, so `RANK()` is 1 throughout
/// and `ROW_NUMBER()` still counts the partition's rows.
fn rank_and_row_number_accept_an_order_key_that_is_also_the_partition_key() {
    let (dir, required, cpu) = partition_window_contexts("algeon-datafusion-native-exec-window-repeated-key");
    for query in [
        "SELECT p, RANK() OVER (PARTITION BY p ORDER BY p) AS r FROM input ORDER BY p, r",
        "SELECT p, COUNT(*) AS n, MAX(rn) AS max_rn FROM \
         (SELECT p, ROW_NUMBER() OVER (PARTITION BY p ORDER BY p) AS rn FROM input) GROUP BY p ORDER BY p",
    ] {
        let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
        assert_eq!(actual, expected, "{query}");
    }
    fs::remove_dir_all(dir).unwrap();
}

/// DataFusion drops the final sort because the window keeps its input's
/// order, so the window's own output order is the query's result order.
fn partition_aggregate_window_output_keeps_the_order_the_query_relies_on() {
    let (dir, required, cpu) = partition_window_contexts("algeon-datafusion-native-exec-window-order");
    let query = "SELECT p, SUM(x) OVER (PARTITION BY p) AS s FROM input ORDER BY p";
    let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
    assert_eq!(actual, expected);
    fs::remove_dir_all(dir).unwrap();
}

/// Two windows on one sort: DataFusion stacks the partition aggregate window
/// above the rank window and drops the final sort, so the result order is the
/// order the aggregate window keeps from the rank window's sorted output.
fn windows_sharing_a_sort_keep_the_order_the_query_relies_on() {
    let (dir, required, cpu) = partition_window_contexts("algeon-datafusion-native-exec-window-shared-sort");
    let query = "SELECT p, o, SUM(x) OVER (PARTITION BY p) AS s, RANK() OVER (PARTITION BY p ORDER BY o) AS r \
                 FROM input ORDER BY p, o, r";
    let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
    assert_eq!(actual, expected);
    fs::remove_dir_all(dir).unwrap();
}

/// An empty OVER clause aggregates the whole input. The aggregate may appear
/// alone, beside a ranking window with its own ordering, or as the total a rank
/// is compared with.
#[test]
fn whole_input_aggregate_windows_match_datafusion() {
    let (dir, required, cpu) = partition_window_contexts("algeon-datafusion-native-exec-window-whole-input");
    for query in [
        "SELECT p, o, x, COUNT(*) OVER () AS n, SUM(x) OVER () AS s FROM input ORDER BY p, o, x",
        "SELECT p, rk, n FROM (SELECT p, ROW_NUMBER() OVER (ORDER BY p, o, x) AS rk, COUNT(*) OVER () AS n \
         FROM input) ORDER BY rk",
        "SELECT p, o, x FROM (SELECT p, o, x, ROW_NUMBER() OVER (ORDER BY x DESC, p, o) AS rk, \
         COUNT(*) OVER () AS n FROM input) WHERE rk <= n / 100 ORDER BY rk",
    ] {
        let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
        assert_eq!(actual, expected, "{query}");
    }
    fs::remove_dir_all(dir).unwrap();
}

/// At four partitions DataFusion merges sorted partitions by an expression
/// key, and plans a `UNION ALL` of joins that share the grouping key's hash
/// partitioning as an `InterleaveExec`. Each query lowers whole under
/// NativeRequired and returns DataFusion's rows.
#[test]
fn multi_partition_expression_merge_and_interleaved_union_match_datafusion() {
    const FILES: i32 = 3;
    const ROWS_PER_FILE: i32 = 2_000;
    let dir = unique_temp_dir("algeon-datafusion-native-exec-merge-interleave");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    for file in 0..FILES {
        let ids = (file * ROWS_PER_FILE)..((file + 1) * ROWS_PER_FILE);
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(ids.clone())),
                Arc::new(Int32Array::from_iter_values(ids.clone().map(|id| id % 997))),
                Arc::new(Float64Array::from_iter_values(ids.clone().map(|id| f64::from((id * 7919) % 1009)))),
                Arc::new(Float64Array::from_iter_values(ids.map(|id| f64::from(id % 13 + 1)))),
            ],
        )
        .unwrap();
        let file = File::create(dir.join(format!("part{file}.parquet"))).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let config = || {
        let mut config = SessionConfig::new().with_target_partitions(4);
        config.options_mut().optimizer.hash_join_single_partition_threshold = 0;
        config.options_mut().optimizer.hash_join_single_partition_threshold_rows = 0;
        config
    };
    let required = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(config())
            .with_algeon_native(native_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired))
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    let cases = [
        ("SortPreservingMergeExec", "SELECT id, value, weight FROM input ORDER BY value / weight DESC, id LIMIT 20"),
        (
            "InterleaveExec",
            "SELECT key, COUNT(*) AS n, SUM(value) AS s FROM (\
             SELECT a.key, a.value FROM input a JOIN input b ON a.key = b.key WHERE a.weight >= 7 \
             UNION ALL SELECT a.key, a.value FROM input a JOIN input b ON a.key = b.key WHERE a.weight < 7) u \
             GROUP BY key ORDER BY key",
        ),
    ];
    let runtime = Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
    runtime.block_on(async {
        for ctx in [&required, &cpu] {
            ctx.register_parquet("input", dir.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
        }
        for (operator, query) in cases {
            let cpu_plan = cpu.sql(query).await.unwrap().create_physical_plan().await.unwrap();
            let cpu_tree = displayable(cpu_plan.as_ref()).indent(false).to_string();
            assert!(cpu_tree.contains(operator), "{query}\n{cpu_tree}");
            let native_plan = required
                .sql(query)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap_or_else(|error| panic!("NativeRequired plans {query}: {error}"));
            assert!(
                below_native_admission(&native_plan).downcast_ref::<AlgeonNativeExec>().is_some(),
                "{query}\n{}",
                displayable(native_plan.as_ref()).indent(false)
            );
            let rows = |batches: Vec<RecordBatch>| pretty_format_batches(&batches).unwrap().to_string();
            let actual = rows(collect(native_plan, Arc::new(TaskContext::default())).await.unwrap());
            let expected = rows(collect(cpu_plan, Arc::new(TaskContext::default())).await.unwrap());
            assert_eq!(actual, expected, "{query}");
        }
    });
    fs::remove_dir_all(dir).unwrap();
}

/// DataFusion inserts a multi-partition partition-preserving SortExec under
/// window aggregates when target_partitions > 1. The native planner must
/// normalize this input into a native sort and lower the entire window plan
/// to native GPU execution with zero host boundaries.
#[test]
fn window_agg_over_partition_preserving_sort_in_multi_partition_session() {
    const ROWS: i32 = 20_000;
    let dir = unique_temp_dir("algeon-datafusion-native-exec-window-partition-preserving");
    let path = dir.join("input.parquet");
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("p", DataType::Int32, false),
            Field::new("x", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values((0..ROWS).map(|row| (row * 7919) % 211))),
                Arc::new(Int64Array::from_iter_values((0..ROWS).map(|row| i64::from(row % 1000)))),
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let config = || SessionConfig::new().with_target_partitions(4).with_repartition_file_scans(true);
    let required = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(config())
            .with_algeon_native(native_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired))
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    let runtime = Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap();
    let (actual, expected) = runtime.block_on(async {
        for ctx in [&required, &cpu] {
            ctx.register_parquet("input", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
        }
        let query = "SELECT p, SUM(x) OVER (PARTITION BY p) AS s FROM input ORDER BY p, s";
        let req_plan = required
            .sql(query)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .expect("NativeRequired plans the window natively");
        let tree = displayable(req_plan.as_ref()).indent(false).to_string();
        assert!(
            below_native_admission(&req_plan).downcast_ref::<AlgeonNativeExec>().is_some(),
            "window plan must lower to AlgeonNativeExec:\n{tree}"
        );
        let rows = |batches: Vec<RecordBatch>| pretty_format_batches(&batches).unwrap().to_string();
        let actual = rows(collect(req_plan, Arc::new(TaskContext::default())).await.unwrap());
        let expected = rows(cpu.sql(query).await.unwrap().collect().await.unwrap());
        (actual, expected)
    });
    assert_eq!(actual, expected);
    fs::remove_dir_all(dir).unwrap();
}

/// DataFusion's `rank()`, `dense_rank()`, and `percent_rank()` are one `Rank`
/// type that differs only in a private rank kind. Only `rank()` has a native
/// kernel, so the other two stay in DataFusion under NativePreferred and
/// NativeRequired rejects the completed plan.
fn dense_rank_and_percent_rank_windows_stay_in_datafusion() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-rank-family");
    let path = dir.join("input.parquet");
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("key", DataType::Int32, false),
            Field::new("payload", DataType::Int32, false),
        ]));
        // The tie inside partition 1 is where rank() and dense_rank() differ.
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
                Arc::new(Int32Array::from(vec![1, 1, 1, 1, 2, 2])),
                Arc::new(Int32Array::from(vec![10, 10, 20, 30, 5, 7])),
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let config = || SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
    let preferred = SessionContext::new_with_state(
        test_gpu_backend()
            .install_on(SessionStateBuilder::new().with_default_features().with_config(config()))
            .unwrap()
            .build(),
    );
    let required = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(config())
            .with_algeon_native(native_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired))
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    for ctx in [&preferred, &required, &cpu] {
        block_on(ctx.register_parquet("input", path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    }

    for function in ["dense_rank", "percent_rank"] {
        let query =
            format!("SELECT id, {function}() OVER (PARTITION BY key ORDER BY payload) AS r FROM input ORDER BY id");
        let plan = block_on(block_on(preferred.sql(&query)).unwrap().create_physical_plan()).unwrap();
        let tree = displayable(plan.as_ref()).indent(false).to_string();
        assert!(tree.contains("WindowAggExec"), "{function}() must stay a DataFusion window:\n{tree}");
        let rows = |batches: Vec<RecordBatch>| pretty_format_batches(&batches).unwrap().to_string();
        let actual = rows(block_on(collect(plan, Arc::new(TaskContext::default()))).unwrap());
        let expected = rows(block_on(block_on(cpu.sql(&query)).unwrap().collect()).unwrap());
        assert_eq!(actual, expected, "{function}() under NativePreferred must match DataFusion");

        let error = block_on(block_on(required.sql(&query)).unwrap().create_physical_plan())
            .expect_err("NativeRequired must reject a window without native semantics");
        let adapter_error = error.find_algeon_error().expect("the rejection must stay a typed Algeon error");
        assert_eq!(adapter_error.code(), ErrorCode::FinalPlanRequirementUnsatisfied, "{function}()");
    }

    fs::remove_dir_all(dir).unwrap();
}

/// An Int16-keyed STDDEV under host-exact statistical execution: a legal
/// aggregate that admission refuses (the exact host kernel has no Int16
/// group key), returning its physical plan and the final native rule
/// configured with `policy`.
fn admission_refused_stddev_plan(
    dir: &std::path::Path,
    policy: AlgeonGpuExecutionMode,
) -> (Arc<dyn ExecutionPlan>, Arc<dyn PhysicalOptimizerRule + Send + Sync>) {
    let path = dir.join("input.parquet");
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int16, false), Field::new("x", DataType::Float64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int16Array::from(vec![1, 2, 1, 2])), Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 5.0]))],
    )
    .unwrap();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    block_on(ctx.register_parquet("input", path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    let plan = block_on(
        block_on(ctx.sql("SELECT k, STDDEV(x) AS s FROM input GROUP BY k ORDER BY k")).unwrap().create_physical_plan(),
    )
    .unwrap();
    let config = AlgeonNativeOptimizerConfig::default().with_execution_mode(policy).with_native_execution_config(
        bounded_config()
            .with_statistical_aggregate_execution_mode(StatisticalAggregateExecutionMode::DataFusionExactHostWelford),
    );
    let final_native = installed_physical_optimizer_rules(config)
        .into_iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");
    (plan, final_native)
}

fn native_preferred_keeps_an_admission_refusal_on_datafusion() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-admission-fallback");
    let (plan, final_native) = admission_refused_stddev_plan(&dir, AlgeonGpuExecutionMode::NativePreferred);

    let optimized = final_native
        .optimize(plan, &ConfigOptions::default())
        .expect("a legal plan admission refuses stays on DataFusion");

    // The refused aggregate stays on DataFusion over a native scan fragment.
    // The fragment is admitted under the host-exact statistical mode, which
    // the shared test backend does not execute, so the plan is not run.
    assert_eq!(classify_final_plan(optimized.as_ref()), FinalPlanDisposition::Mixed);
    let mut node = below_native_admission(&optimized);
    while node.name() != "AggregateExec" {
        node = node.children()[0];
    }
    assert!(
        node.children()[0].downcast_ref::<AlgeonNativeExec>().is_some(),
        "{}",
        displayable(optimized.as_ref()).indent(false)
    );
    fs::remove_dir_all(dir).unwrap();
}

/// The NativeRequired error names the refusal from the production admission
/// decision, in addition to the coarse capability reason.
fn native_required_rejection_names_the_admission_refusal() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-admission-refusal");
    let (plan, final_native) = admission_refused_stddev_plan(&dir, AlgeonGpuExecutionMode::NativeRequired);

    let error = final_native
        .optimize(plan, &ConfigOptions::default())
        .expect_err("NativeRequired cannot keep the refused aggregate on DataFusion");
    let error = error.find_algeon_error().expect("the final-plan rejection keeps its typed error");

    assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    let facts = error.facts().iter().collect::<Vec<_>>();
    assert!(error.fact_value("capability_rejection_reason").is_some(), "{facts:?}");
    assert!(error.fact_value("capability_rejection_boundary").is_some(), "{facts:?}");
    fs::remove_dir_all(dir).unwrap();
}

/// A fact table `t` and a smaller dimension `u`, written so that DataFusion
/// builds the join's hash table on `u` and pushes a sort on `t`'s columns to
/// the probe input. `t.id` makes every window ordering total.
fn window_join_contexts(prefix: &str) -> (std::path::PathBuf, SessionContext, SessionContext) {
    const FACT_ROWS: i64 = 20_000;
    const DIMENSION_ROWS: i64 = 997;
    let dir = unique_temp_dir(prefix);
    let write = |name: &str, columns: Vec<(&str, Vec<i64>)>| {
        let schema = Arc::new(Schema::new(
            columns.iter().map(|(column, _)| Field::new(*column, DataType::Int64, false)).collect::<Vec<_>>(),
        ));
        let arrays = columns.into_iter().map(|(_, values)| Arc::new(Int64Array::from(values)) as ArrayRef).collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
        let mut writer = ArrowWriter::try_new(File::create(dir.join(name)).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    };
    write(
        "t.parquet",
        vec![
            ("id", (0..FACT_ROWS).collect()),
            ("k", (0..FACT_ROWS).map(|row| (row * 31) % DIMENSION_ROWS).collect()),
            ("a", (0..FACT_ROWS).map(|row| (row * 7919) % 53).collect()),
            ("b", (0..FACT_ROWS).map(|row| (row * 131) % 211).collect()),
            ("v", (0..FACT_ROWS).map(|row| row % 1000).collect()),
        ],
    );
    write(
        "u.parquet",
        vec![("k", (0..DIMENSION_ROWS).collect()), ("w", (0..DIMENSION_ROWS).map(|row| row * 3).collect())],
    );
    let config = || SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
    let required = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(config())
            .with_algeon_native(native_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired))
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    for ctx in [&required, &cpu] {
        for table in ["t", "u"] {
            let path = dir.join(format!("{table}.parquet"));
            block_on(ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
        }
    }
    (dir, required, cpu)
}

/// DataFusion pushes the window's sort below the hash join and relies on the
/// join to keep the probe order. The native plan sorts the join output itself,
/// so a window over a join plans natively under `NativeRequired` and returns
/// the rows DataFusion returns.
#[test]
fn windows_over_a_hash_join_match_datafusion() {
    let (dir, required, cpu) = window_join_contexts("algeon-datafusion-native-exec-window-over-join");
    for query in [
        "SELECT t.id, LEAD(t.v) OVER (PARTITION BY t.a ORDER BY t.b, t.id) AS next \
         FROM t JOIN u ON t.k = u.k ORDER BY t.id",
        "SELECT t.id, LAG(u.w, 2, -1) OVER (PARTITION BY t.a ORDER BY t.b, t.id) AS previous \
         FROM t JOIN u ON t.k = u.k ORDER BY t.id",
        "SELECT t.id, ROW_NUMBER() OVER (PARTITION BY t.a ORDER BY t.b, t.id) AS rn \
         FROM t JOIN u ON t.k = u.k ORDER BY t.id",
        "SELECT t.id, RANK() OVER (PARTITION BY t.a ORDER BY t.b) AS r FROM t JOIN u ON t.k = u.k ORDER BY t.id",
        "SELECT t.id, SUM(u.w) OVER (PARTITION BY t.a) AS partition_total FROM t JOIN u ON t.k = u.k ORDER BY t.id",
        // No final ORDER BY: the window's own output order is the result order.
        "SELECT t.a, t.b, t.id, LEAD(t.v) OVER (PARTITION BY t.a ORDER BY t.b, t.id) AS next \
         FROM t JOIN u ON t.k = u.k",
    ] {
        let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
        assert_eq!(actual, expected, "{query}");
    }
    fs::remove_dir_all(dir).unwrap();
}

/// Branches of a UNION ALL that read one table, or one aggregate CTE, lower to
/// one native node that the union reads through two inputs. Each input yields
/// every row, so the result holds each row twice, as in DataFusion.
#[test]
fn unions_reading_one_producer_twice_match_datafusion() {
    // DataFusion runs the partitions of a union on spawned tasks.
    let runtime = Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let _runtime = runtime.enter();
    let (dir, required, cpu) = window_join_contexts("algeon-datafusion-native-exec-self-union");
    for query in [
        "SELECT id FROM t UNION ALL SELECT id FROM t ORDER BY id",
        // DataFusion prunes the unused window, which leaves the same scan twice.
        "SELECT id FROM t UNION ALL SELECT id FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY a, id) AS rk FROM t) x \
         ORDER BY id",
        "WITH c AS (SELECT a, SUM(v) AS s FROM t GROUP BY a) SELECT a, s FROM c UNION ALL SELECT a, s FROM c \
         ORDER BY a, s",
        "SELECT COUNT(*), SUM(id) FROM (SELECT id FROM t UNION ALL SELECT id FROM t UNION ALL SELECT id FROM t)",
    ] {
        let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
        assert_eq!(actual, expected, "{query}");
    }
    fs::remove_dir_all(dir).unwrap();
}

/// Writes a node table and an edge table whose Parquet footers declare their
/// sort order, as the DBLP citation fixture does, and registers a view of the
/// edges between filtered nodes. Returns a NativeRequired context and a
/// DataFusion CPU context over them.
fn sorted_graph_contexts(prefix: &str) -> (std::path::PathBuf, SessionContext, SessionContext) {
    const NODES: i64 = 3_000;
    const EDGES: i64 = 3_000;
    let dir = unique_temp_dir(prefix);
    let write = |name: &str, columns: Vec<(&str, Vec<i64>)>| {
        let schema = Arc::new(Schema::new(
            columns.iter().map(|(column, _)| Field::new(*column, DataType::Int64, false)).collect::<Vec<_>>(),
        ));
        let sorting_columns = (0..columns.len().min(2))
            .map(|column_idx| SortingColumn { column_idx: column_idx as i32, descending: false, nulls_first: false })
            .collect::<Vec<_>>();
        let arrays = columns.into_iter().map(|(_, values)| Arc::new(Int64Array::from(values)) as ArrayRef).collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
        let properties = WriterProperties::builder().set_sorting_columns(Some(sorting_columns)).build();
        let mut writer = ArrowWriter::try_new(File::create(dir.join(name)).unwrap(), schema, Some(properties)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    };
    write(
        "nodes.parquet",
        vec![("node_id", (0..NODES).collect()), ("year", (0..NODES).map(|node| 1900 + (node * 7) % 130).collect())],
    );
    let mut edges = (0..EDGES).map(|edge| ((edge * 37) % NODES, (edge * 101 + 7) % NODES)).collect::<Vec<_>>();
    edges.sort_unstable();
    write(
        "edges.parquet",
        vec![
            ("src", edges.iter().map(|(src, _)| *src).collect()),
            ("dst", edges.iter().map(|(_, dst)| *dst).collect()),
        ],
    );
    let config = || SessionConfig::new().with_target_partitions(1);
    let required = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(config())
            .with_algeon_native(native_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired))
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    for ctx in [&required, &cpu] {
        for table in ["nodes", "edges"] {
            let path = dir.join(format!("{table}.parquet"));
            block_on(ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
        }
        for view in [
            "CREATE VIEW era_nodes AS SELECT node_id, year FROM nodes WHERE year BETWEEN 1900 AND 2020",
            "CREATE VIEW era_edges AS SELECT e.src, e.dst, CAST(ABS(a.year - b.year) AS DOUBLE) AS year_gap \
             FROM edges e JOIN era_nodes a ON a.node_id = e.src JOIN era_nodes b ON b.node_id = e.dst",
        ] {
            block_on(ctx.sql(view)).unwrap();
        }
    }
    (dir, required, cpu)
}

/// A source with a declared sort order makes DataFusion stream the final
/// aggregate of a distinct `UNION` over sorted groups. It merges the sorted
/// runs of the partial aggregate below the union with a
/// `SortPreservingMergeExec`. The aggregate lowers natively, and the rows come
/// back in the order DataFusion declares for them.
#[test]
fn distinct_union_over_joins_of_sorted_sources_matches_datafusion() {
    // DataFusion runs the partitions of a union on spawned tasks.
    let runtime = Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let _runtime = runtime.enter();
    let (dir, required, cpu) = sorted_graph_contexts("algeon-datafusion-native-exec-sorted-distinct-union");
    let union = "SELECT src AS v FROM era_edges UNION SELECT dst AS v FROM era_edges";
    let datafusion_plan = block_on(block_on(cpu.sql(union)).unwrap().create_physical_plan()).unwrap();
    let rendered = displayable(datafusion_plan.as_ref()).indent(false).to_string();
    assert!(
        rendered.contains("AggregateExec: mode=Final") && rendered.contains("SortPreservingMergeExec"),
        "DataFusion should merge sorted partial aggregates below the final aggregate:\n{rendered}"
    );
    for query in [
        union,
        "SELECT COUNT(*), SUM(v) FROM (SELECT src AS v FROM era_edges UNION SELECT dst AS v FROM era_edges)",
        "SELECT src AS v FROM era_edges UNION SELECT dst AS v FROM era_edges ORDER BY v DESC",
    ] {
        let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
        assert_eq!(actual, expected, "{query}");
    }
    fs::remove_dir_all(dir).unwrap();
}

/// The native window kernel reads columns, so the `1` of `COUNT(*)` and the
/// `x + 1` of a window aggregate or LAG are computed below the window.
#[test]
fn window_functions_over_computed_arguments_match_datafusion() {
    let (dir, required, cpu) = partition_window_contexts("algeon-datafusion-native-exec-window-computed-argument");
    for query in [
        "SELECT p, COUNT(*) OVER (PARTITION BY p) AS n FROM input ORDER BY p, n",
        "SELECT p, x, SUM(x + 1) OVER (PARTITION BY p) AS s FROM input ORDER BY p, x, s",
        "SELECT p, o, x, LAG(x + 1) OVER (PARTITION BY p ORDER BY o, x) AS previous FROM input \
         ORDER BY p, o, x, previous",
    ] {
        let (actual, expected) = native_and_datafusion_rows(&required, &cpu, query);
        assert_eq!(actual, expected, "{query}");
    }
    fs::remove_dir_all(dir).unwrap();
}

/// Writes `input(p, x, f, d)` in eight row groups, a NativeRequired context
/// and a DataFusion context. Both keep DataFusion's defaults apart from single-partition
/// planning and `collect_statistics`, so DataFusion plans each uncorrelated
/// scalar subquery as a `ScalarSubqueryExec` and the Algeon session plans the
/// join its builder selects.
fn scalar_subquery_contexts(
    prefix: &str,
    collect_statistics: bool,
) -> (std::path::PathBuf, SessionContext, SessionContext) {
    const ROWS: i32 = 2_000;
    let dir = unique_temp_dir(prefix);
    let path = dir.join("input.parquet");
    {
        let schema = Arc::new(Schema::new(vec![
            Field::new("p", DataType::Int32, false),
            Field::new("x", DataType::Int64, false),
            Field::new("f", DataType::Float64, false),
            Field::new("d", DataType::Date32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values((0..ROWS).map(|row| (row * 7919) % 211))),
                Arc::new(Int64Array::from_iter_values((0..ROWS).map(|row| i64::from(row % 1000)))),
                Arc::new(Float64Array::from_iter_values((0..ROWS).map(|row| f64::from(row % 97) / 4.0))),
                Arc::new(Date32Array::from_iter_values((0..ROWS).map(|row| 19_000 + row % 365))),
            ],
        )
        .unwrap();
        // Eight row groups make a scan that two consumers share arrive in
        // several source chunks.
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), schema, None).unwrap();
        for offset in (0..batch.num_rows()).step_by(250) {
            writer.write(&batch.slice(offset, 250)).unwrap();
            writer.flush().unwrap();
        }
        writer.close().unwrap();
    }
    let config = || {
        let mut config = SessionConfig::new().with_target_partitions(1).with_repartition_file_scans(false);
        config.options_mut().execution.collect_statistics = collect_statistics;
        config
    };
    let required = SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(config())
            .with_algeon_native(native_optimizer_config().with_execution_mode(AlgeonGpuExecutionMode::NativeRequired))
            .build(),
    );
    let cpu = SessionContext::new_with_config(config());
    for ctx in [&required, &cpu] {
        block_on(ctx.register_parquet("input", path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    }
    (dir, required, cpu)
}

/// An Algeon session plans an uncorrelated scalar subquery as a join with its
/// one-row aggregate, so a scalar aggregate in a projection, an arithmetic
/// expression, a HAVING or a filter plans under `NativeRequired` and returns
/// DataFusion's values and types. The subquery can aggregate the scan the
/// outer query reads with a computed argument (`AVG(x)` reads
/// `CAST(x AS DOUBLE)`, `COUNT(*)` reads a literal). With DataFusion's default
/// statistics collection, an unfiltered `MIN`, `MAX` or `COUNT` arrives as
/// literals over a placeholder row.
#[test]
fn uncorrelated_scalar_subqueries_plan_natively_as_joins_and_match_datafusion() {
    for collect_statistics in [false, true] {
        let (dir, required, cpu) =
            scalar_subquery_contexts("algeon-datafusion-native-exec-scalar-subquery", collect_statistics);
        for query in [
            "WITH cte AS (SELECT p, x FROM input WHERE p < 50) \
             SELECT x, x / (SELECT COUNT(*) FROM cte) AS r FROM cte ORDER BY x, r",
            "SELECT p, x, (SELECT MAX(x) FROM input) AS mx, (SELECT COUNT(*) FROM input WHERE p = 3) AS c \
             FROM input ORDER BY p, x",
            "SELECT p, SUM(x) AS s FROM input GROUP BY p HAVING SUM(x) > (SELECT SUM(x) / 300 FROM input) ORDER BY p",
            "SELECT p, x FROM input WHERE x * 2 > (SELECT MAX(x) FROM input) ORDER BY p, x",
            "SELECT p, x FROM input WHERE x < (SELECT MIN(x) FROM input) + 5 ORDER BY p, x",
            "SELECT p, x FROM input WHERE x > (SELECT AVG(x) FROM input) ORDER BY p, x",
            "SELECT p, x FROM input WHERE x > (SELECT SUM(x) / COUNT(*) FROM input) ORDER BY p, x",
            "SELECT p, x, x - (SELECT AVG(x) FROM input) AS centered FROM input ORDER BY p, x",
            "SELECT p, x, CAST(x AS DOUBLE) / (SELECT COUNT(*) FROM input) AS share FROM input ORDER BY p, x",
            "SELECT p, x - (SELECT MIN(x) FROM input) AS d FROM input ORDER BY p, d",
            "SELECT p, d, (SELECT MAX(p) FROM input) AS mp, (SELECT MAX(f) FROM input) AS mf, \
             (SELECT MIN(d) FROM input) AS md FROM input ORDER BY p, d",
        ] {
            let plan = block_on(block_on(required.sql(query)).unwrap().create_physical_plan())
                .unwrap_or_else(|error| panic!("{query} collect_statistics={collect_statistics}: {error}"));
            let actual = block_on(collect(plan, Arc::new(TaskContext::default()))).unwrap();
            let expected = block_on(block_on(cpu.sql(query)).unwrap().collect()).unwrap();
            let types = |batches: &[RecordBatch]| {
                batches[0].schema().fields().iter().map(|field| field.data_type().clone()).collect::<Vec<_>>()
            };
            assert_eq!(types(&actual), types(&expected), "{query} collect_statistics={collect_statistics}");
            let rows = |batches: &[RecordBatch]| pretty_format_batches(batches).unwrap().to_string();
            let (actual, expected) = (rows(&actual), rows(&expected));
            assert!(expected.lines().count() > 4, "{query} returns rows");
            assert_eq!(actual, expected, "{query} collect_statistics={collect_statistics}");
        }
        fs::remove_dir_all(dir).unwrap();
    }
}

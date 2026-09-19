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
    error::{DataFusionErrorExt, ErrorCode},
    native_exec::{AlgeonNativeExec, AlgeonNativeExecutionConfig},
    native_report::{CandidateOutcomeKind, NativeDataFusionPlanningReport},
    planner::{
        ALGEON_FINAL_PLAN_REQUIREMENT_RULE_NAME, AlgeonFinalPlanRequirementRule, AlgeonNativeOptimizerConfig,
        AlgeonNativePhysicalOptimizerRule, CAPABILITY_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
        DataFusionCpuExecutionPolicy, FinalPlanDisposition, classify_final_plan,
    },
    report::CandidateOutcome,
    session::AlgeonSessionStateBuilderExt,
};
use arrow::{
    array::{Array, Decimal128Array, Int32Array, ListArray, StringArray, StringViewArray},
    buffer::{OffsetBuffer, ScalarBuffer},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    common::{DataFusionError, config::ConfigOptions},
    datasource::{memory::MemorySourceConfig, source::DataSourceExec},
    execution::{SessionStateBuilder, context::TaskContext, object_store::ObjectStoreUrl},
    logical_expr::Operator,
    physical_expr::{LexOrdering, PhysicalSortExpr, expressions::col},
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
use parquet::arrow::ArrowWriter;
use proptest::prelude::*;
use query_engine::{
    mode::CapabilityEnvelopeStatus,
    plan::{NativePlanCostOptions, NativePlanCostReason},
};
use workloads::tpch::{TPCH_QUERIES, write_tiny_tpch_parquet_dataset};

#[path = "native_exec_tests/bounded_join.rs"]
mod bounded_join;
#[path = "native_fallback_pressure_tests/mixed_fallback.rs"]
mod mixed_fallback;
#[path = "native_exec_tests/support.rs"]
mod support;
#[path = "native_exec_tests/support_core.rs"]
mod support_core;
#[path = "support/tpch_session.rs"]
mod tpch_session;

use support::*;

fn test_gpu_backend() -> &'static AlgeonGpuBackend {
    static BACKEND: OnceLock<AlgeonGpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        let profile = AlgeonGpuDeviceProfile::new(0)
            .with_memory_limit(GpuDeviceMemoryLimit::Explicit { limit_bytes: 256 * 1024 * 1024 });
        let device = AlgeonGpuDeviceConfig::new(profile, GpuMemoryOwnership::WholeDeviceExclusive);
        AlgeonGpuBackend::builder()
            .with_native_execution_config(AlgeonNativeExecutionConfig::bounded(
                256 * 1024 * 1024,
                64 * 1024 * 1024,
                2,
                3,
                1,
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

#[test]
fn native_optimizer_rule_rewrites_supported_physical_plan() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-optimizer");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
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
    assert!(diagnostics.capability_contract.accepted_candidates.is_empty());
    assert!(diagnostics.capability_contract.rejected_candidates.is_empty());
    assert_eq!(diagnostics.capability_contract.non_join_bounded_plan_count, 1);
    assert_eq!(
        diagnostics.capability_contract.non_join_bounded_plan_reason.as_deref(),
        Some(CAPABILITY_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON)
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_is_mixed_and_idempotent() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let plan = ilike_filter_plan(&path);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let report = NativeDataFusionPlanningReport::from_retained_rule("ilike_boundary", Arc::clone(&optimized), &rule);

    assert_eq!(report.rows.len(), 1);
    let row = &report.rows[0];
    assert_eq!(row.outcome, Some(CandidateOutcomeKind::Selected));
    assert_eq!(row.final_disposition, Some(FinalPlanDisposition::Mixed));
    assert_eq!(row.native_plan_fragment_count, Some(1));
    assert_eq!(row.host_datafusion_boundary_count, Some(1));
    assert_eq!(row.source_diagnostics.len(), 1);
    assert!(row.source_diagnostics[0].direct_device_read);

    let filter = optimized.downcast_ref::<FilterExec>().expect("mixed boundary must retain the CPU root filter");
    let native = filter
        .input()
        .downcast_ref::<AlgeonNativeExec>()
        .expect("mixed boundary must replace only the local-Parquet leaf");
    assert_mixed_boundary_properties(filter, native);

    let report_before_repeat = rule.last_report().expect("selected mixed boundary must retain a report");
    let repeated = rule.optimize(Arc::clone(&optimized), &ConfigOptions::default()).unwrap();
    assert!(Arc::ptr_eq(&optimized, &repeated), "mixed boundary must be an optimizer fixed point");
    assert_eq!(rule.last_report(), Some(report_before_repeat), "idempotence must preserve the selected-leaf report");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_allows_transparent_cooperative_wrapper() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike-cooperative");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(CooperativeExec::new(ilike_filter_plan(&path)));
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let cooperative = optimized
        .downcast_ref::<CooperativeExec>()
        .expect("the certified transparent wrapper should remain in the final plan");
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

#[test]
fn native_optimizer_property_checked_ilike_boundary_rejects_nonlocal_parquet() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike-nonlocal");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let plan = ilike_filter_over_scan(parquet_scan_with_object_store_url(
        &path,
        ObjectStoreUrl::parse("s3://mixed-boundary-test").unwrap(),
    ));
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(
        Arc::ptr_eq(&optimized, &plan),
        "the mixed certificate must not replace a remote Parquet source through its local-only boundary"
    );
    let report = rule.last_report().expect("native optimizer should record the whole-plan rejection");
    assert_eq!(report.selected_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_rejects_partition_grouped_scan() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike-partition-grouped");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let scan = support::parquet_scan_with_partition_field(&path, Field::new("part", DataType::Int32, false));
    let partition_grouped = scan
        .downcast_ref::<datafusion::datasource::source::DataSourceExec>()
        .expect("partition scan should be DataSourceExec");
    let (file_scan, _parquet_source) = partition_grouped
        .downcast_to_file_source::<datafusion::datasource::physical_plan::ParquetSource>()
        .expect("partition scan should use ParquetSource");
    assert!(!file_scan.table_partition_cols().is_empty());
    assert!(matches!(file_scan.output_partitioning, Some(datafusion::physical_expr::Partitioning::Hash(_, _))));
    let plan = ilike_filter_over_scan(scan);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(
        Arc::ptr_eq(&optimized, &plan),
        "the mixed certificate must not replace a partition-grouped scan whose partition state is observable DataFusion 55 state"
    );
    let report = rule.last_report().expect("native optimizer should record the whole-plan rejection");
    assert_eq!(report.selected_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_property_checked_ilike_boundary_is_rejected_by_no_datafusion_cpu() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike-no-datafusion-cpu");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let config = AlgeonNativeOptimizerConfig::default()
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
        .with_native_execution_config(bounded_config());
    let native_rule = AlgeonNativePhysicalOptimizerRule::new(config.clone());
    let requirement_rule = AlgeonFinalPlanRequirementRule::new(config);

    let planned = native_rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let report = NativeDataFusionPlanningReport::from_retained_rule("ilike_no_cpu", Arc::clone(&planned), &native_rule);
    assert_eq!(report.rows[0].final_disposition, Some(FinalPlanDisposition::Mixed));
    assert_eq!(report.rows[0].native_plan_fragment_count, Some(1));
    assert_eq!(report.rows[0].host_datafusion_boundary_count, Some(1));

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
    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike-parity");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);

    let cpu = futures::executor::block_on(collect(ilike_filter_plan(&path), Arc::new(TaskContext::default()))).unwrap();
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());
    let native_plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let native_leaf = mixed_boundary_native_leaf(native_plan.as_ref());
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

    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike-host-budget");
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

    let error = futures::executor::block_on(collect(Arc::clone(&plan), task_context))
        .expect_err("the optimizer-built mixed boundary must reserve its host Arrow output before D2H");

    assert!(matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)), "unexpected error: {error:?}");
    assert!(error.to_string().contains("AlgeonNativeExec[host-output]"), "unexpected error: {error}");
    assert_eq!(
        mixed_boundary_native_leaf(plan.as_ref())
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

#[test]
fn native_optimizer_property_checked_ilike_boundary_uses_one_backend_admission_and_observation_owner() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-property-checked-ilike-admission");
    let path = dir.join("input.parquet");
    write_ilike_parquet_file(&path);
    let backend = test_gpu_backend();
    let rule = AlgeonNativePhysicalOptimizerRule::new(backend.optimizer_config().clone());
    let plan = rule.optimize(ilike_filter_plan(&path), &ConfigOptions::default()).unwrap();
    let native_leaf = mixed_boundary_native_leaf(plan.as_ref());

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
    assert_eq!(
        snapshot.active_queries, 0,
        "the mixed-boundary admission ticket must release after its stream completes"
    );
    assert_eq!(snapshot.queued_queries, 0);

    fs::remove_dir_all(dir).unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 16,
        .. ProptestConfig::default()
    })]

    /// Generates only the certified mixed-boundary family. This proves the invariant
    /// across nullable UTF-8 view inputs and deterministic ASCII prefixes
    /// without broadening the root-only local-Parquet certificate.
    #[test]
    fn native_optimizer_property_based_ilike_boundary_preserves_contract(
        mut labels in prop::collection::vec(prop::option::of("[A-Za-z]{0,12}"), 1..16),
        prefix in "[A-Za-z]{0,5}",
    ) {
        // Retain a null in every generated mixed-boundary fixture while still varying
        // nullable input values in the generated rows.
        labels.push(None);
        let dir = unique_temp_dir("algeon-datafusion-native-exec-property-based-ilike");
        let path = dir.join("input.parquet");
        write_ilike_parquet_file_with_labels(&path, &labels);
        let pattern = format!("{prefix}%");

        let cpu = futures::executor::block_on(collect(
            ilike_filter_plan_with_pattern(&path, pattern.clone()),
            Arc::new(TaskContext::default()),
        ))
        .unwrap();
        let config = native_optimizer_config().with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject);
        let native_rule = AlgeonNativePhysicalOptimizerRule::new(config.clone());
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
        prop_assert_eq!(row.outcome, Some(CandidateOutcomeKind::Selected));
        prop_assert_eq!(row.final_disposition, Some(FinalPlanDisposition::Mixed));
        prop_assert_eq!(row.native_plan_fragment_count, Some(1));
        prop_assert_eq!(row.host_datafusion_boundary_count, Some(1));
        let filter = planned.downcast_ref::<FilterExec>().expect("mixed boundary must retain the CPU root filter");
        let native = filter.input().downcast_ref::<AlgeonNativeExec>().expect("mixed boundary must replace only the source leaf");
        assert_mixed_boundary_properties(filter, native);

        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let mixed = runtime.block_on(collect(Arc::clone(&planned), Arc::new(TaskContext::default()))).unwrap();
        prop_assert_eq!(ilike_rows(&mixed), ilike_rows(&cpu));

        let retained_report = native_rule.last_report().expect("mixed-boundary selected report should be retained");
        let repeated = native_rule.optimize(Arc::clone(&planned), &ConfigOptions::default()).unwrap();
        prop_assert!(Arc::ptr_eq(&planned, &repeated));
        prop_assert_eq!(native_rule.last_report(), Some(retained_report));

        let requirement_rule = AlgeonFinalPlanRequirementRule::new(config);
        let error = requirement_rule
            .optimize(planned, &ConfigOptions::default())
            .expect_err("NoDataFusionCpu must reject the exact generated mixed plan");
        let error = adapter_error(&error);
        prop_assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
        prop_assert_eq!(error.fact_value("observed_final_plan_disposition"), Some("mixed"));

        fs::remove_dir_all(dir).unwrap();
    }
}

fn mixed_boundary_native_leaf(plan: &dyn ExecutionPlan) -> &AlgeonNativeExec {
    let filter = plan.downcast_ref::<FilterExec>().expect("mixed boundary must retain the CPU root filter");
    filter.input().downcast_ref::<AlgeonNativeExec>().expect("mixed-boundary filter input must be the native leaf")
}

fn assert_mixed_boundary_properties(filter: &FilterExec, native: &AlgeonNativeExec) {
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
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ordered-scan-properties");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let plan = parquet_scan_with_declared_output_ordering(&path);
    let expected_ordering = plan.properties().output_ordering().cloned();
    assert!(expected_ordering.is_some());
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
    assert_eq!(optimized.properties().output_ordering().cloned(), expected_ordering);
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_reestablishes_declared_scan_ordering_through_aggregate() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ordered-aggregate-composition");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let plan = grouped_sum_aggregate_plan(scan);
    let expected_ordering = plan.properties().output_ordering().cloned();
    assert!(expected_ordering.is_some());
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
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
    let dir = unique_temp_dir("algeon-datafusion-native-exec-ordered-aggregate-output");
    let path = dir.join("input.parquet");
    write_sorted_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let plan = grouped_sum_aggregate_plan(scan);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
    let batches =
        futures::executor::block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(int32_array(&batches[0], 0).values().as_ref(), &[1, 2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
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
    let expected = futures::executor::block_on(collect(Arc::clone(&plan), Arc::new(TaskContext::default()))).unwrap();
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
    let actual =
        futures::executor::block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
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
    let rule = AlgeonNativePhysicalOptimizerRule::new(config.clone());
    let options = ConfigOptions::default();

    let optimized = rule.optimize(plan, &options).unwrap();

    let native =
        optimized.downcast_ref::<AlgeonNativeExec>().expect("supported plan should optimize to AlgeonNativeExec");
    let display = displayable(native).one_line().to_string();
    assert!(display.contains("memory_control_policy=bounded"), "{display}");
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_preserves_rejected_plan_as_datafusion_baseline() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-optimizer-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

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
    let dir = unique_temp_dir("algeon-datafusion-native-exec-cpu-only");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let optimized = plan;

    assert!(optimized.downcast_ref::<LocalLimitExec>().is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_does_not_select_small_selective_topk_join_by_cost_when_enabled() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-cost-topk");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let config = native_optimizer_config().with_native_plan_cost_options(NativePlanCostOptions::default());
    let rule = AlgeonNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(
        optimized.downcast_ref::<AlgeonNativeExec>().is_none(),
        "cost-declined plan should not be replaced by a native execution boundary"
    );
    assert!(Arc::ptr_eq(&optimized, &plan), "a cost-rejected candidate must preserve the exact baseline plan");
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.not_selected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::NotSelectedByCost { cost_reason: NativePlanCostReason::SmallSelectiveTopK, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_does_not_select_small_selective_full_sort_join_by_cost_when_enabled() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-cost-full-sort");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_sql_plan(&left, &right);
    let config = native_optimizer_config().with_native_plan_cost_options(NativePlanCostOptions::default());
    let rule = AlgeonNativePhysicalOptimizerRule::new(config);

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(
        optimized.downcast_ref::<AlgeonNativeExec>().is_none(),
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
fn native_optimizer_rule_errors_for_cost_declined_plan_with_reject_cpu_policy() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-cost-no-datafusion-cpu");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let config = AlgeonNativeOptimizerConfig::default()
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
        .with_native_execution_config(bounded_config())
        .with_native_plan_cost_options(NativePlanCostOptions::default());
    let rules = installed_physical_optimizer_rules(config);
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");
    let final_requirement = rules
        .iter()
        .find(|rule| rule.name() == ALGEON_FINAL_PLAN_REQUIREMENT_RULE_NAME)
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
    let dir = unique_temp_dir("algeon-datafusion-native-exec-no-datafusion-cpu-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let config = AlgeonNativeOptimizerConfig::default()
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");
    let final_requirement = rules
        .iter()
        .find(|rule| rule.name() == ALGEON_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("native optimizer rules should contain the final-plan requirement rule");
    let planned = final_native.optimize(plan, &ConfigOptions::default()).unwrap();
    let error = final_requirement
        .optimize(planned, &ConfigOptions::default())
        .expect_err("the final-plan requirement must reject a rejected native candidate's CPU plan");
    let adapter_error = error.find_algeon_error().expect("final-plan rejection must preserve the typed Algeon error");

    assert_eq!(adapter_error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(adapter_error.fact_value("requirement"), Some("no_datafusion_cpu"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
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
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let native_rule = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");
    let requirement_rule = rules
        .iter()
        .find(|rule| rule.name() == ALGEON_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("native optimizer rules should contain the final-plan requirement rule");
    let planned = native_rule.optimize(parquet_scan(&path), &ConfigOptions::default()).unwrap();
    let error = requirement_rule
        .optimize(planned, &ConfigOptions::default())
        .expect_err("unsupported List<Decimal128> must fail the strict GPU requirement");
    let error = adapter_error(&error);

    assert_eq!(error.code(), ErrorCode::FinalPlanRequirementUnsatisfied);
    assert_eq!(
        error.message(),
        "GPU execution does not support projected column `first_unsupported` with Arrow type `List(Decimal128(10, 2))`"
    );
    assert_eq!(error.fact_value("reason"), Some("unsupported_type"));
    assert_eq!(error.fact_value("unsupported_column"), Some("first_unsupported"));
    assert_eq!(error.fact_value("unsupported_column_type"), Some("List(Decimal128(10, 2))"));
    assert_eq!(error.fact_value("unsupported_column_role"), Some("projected"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn no_datafusion_cpu_count_only_probe_miss_reaches_final_native_optimizer() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-no-datafusion-cpu-probe-miss");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let config = AlgeonNativeOptimizerConfig::default()
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let count_only_probe = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native_count_only")
        .expect("native optimizer rules should contain the count-only probe");
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");

    let after_probe = count_only_probe
        .optimize(plan, &ConfigOptions::default())
        .expect("a non-count plan must not be rejected by the NoDataFusionCpu count-only probe");
    assert!(
        after_probe.downcast_ref::<AlgeonNativeExec>().is_none(),
        "the count-only probe should decline an analytical SUM plan"
    );
    let probe_report = (count_only_probe.as_ref() as &dyn std::any::Any)
        .downcast_ref::<AlgeonNativePhysicalOptimizerRule>()
        .expect("count-only optimizer rule should be native")
        .last_report()
        .expect("count-only probe should record its own rejection evidence");
    assert_eq!(probe_report.selected_count(), 0);
    assert_eq!(probe_report.not_selected_count(), 1);

    let optimized = final_native
        .optimize(after_probe, &ConfigOptions::default())
        .expect("the final native optimizer must run after a count-only probe miss");
    assert!(
        optimized.downcast_ref::<AlgeonNativeExec>().is_some(),
        "the final native optimizer should rewrite the supported analytical plan"
    );
    let final_report = (final_native.as_ref() as &dyn std::any::Any)
        .downcast_ref::<AlgeonNativePhysicalOptimizerRule>()
        .expect("final optimizer rule should be native")
        .last_report()
        .expect("final optimizer should record separate native rewrite evidence");
    assert_eq!(final_report.selected_count(), 1);
    assert_eq!(final_report.not_selected_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn no_datafusion_cpu_final_rejection_replaces_count_only_probe_diagnostics_before_error() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-no-datafusion-cpu-final-rejection-evidence");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = unsupported_try_cast_projection_aggregate_sort_limit_plan(parquet_scan(&path));
    let rules = installed_physical_optimizer_rules(
        AlgeonNativeOptimizerConfig::default()
            .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
            .with_native_execution_config(bounded_config()),
    );
    let count_only_probe = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native_count_only")
        .expect("native optimizer rules should contain the count-only probe");
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");

    let after_probe = count_only_probe.optimize(plan, &ConfigOptions::default()).expect(
        "a normal count-only probe miss must remain non-terminal before NoDataFusionCpu final-plan enforcement",
    );
    let probe_report = (count_only_probe.as_ref() as &dyn std::any::Any)
        .downcast_ref::<AlgeonNativePhysicalOptimizerRule>()
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
        .downcast_ref::<AlgeonNativePhysicalOptimizerRule>()
        .expect("final optimizer rule should be native")
        .last_report()
        .expect("final rejection should replace the probe's retained evidence before returning an error");
    assert!(matches!(
        final_report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));
    let final_requirement = rules
        .iter()
        .find(|rule| rule.name() == ALGEON_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("native optimizer rules should contain the final-plan requirement rule");
    let error = final_requirement
        .optimize(planned, &ConfigOptions::default())
        .expect_err("the final-plan requirement must reject the completed DataFusion plan");
    assert_eq!(adapter_error(&error).code(), ErrorCode::FinalPlanRequirementUnsatisfied);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn no_datafusion_cpu_count_only_probe_retains_its_native_rewrite() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-no-datafusion-cpu-count-probe");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = count_only_sort_limit_plan(parquet_scan(&path));
    let config = AlgeonNativeOptimizerConfig::default()
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let count_only_probe = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native_count_only")
        .expect("native optimizer rules should contain the count-only probe");
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");

    let after_probe = count_only_probe
        .optimize(plan, &ConfigOptions::default())
        .expect("NoDataFusionCpu count-only probe should retain supported native count rewrites");
    assert!(after_probe.downcast_ref::<AlgeonNativeExec>().is_some());
    let probe_report = (count_only_probe.as_ref() as &dyn std::any::Any)
        .downcast_ref::<AlgeonNativePhysicalOptimizerRule>()
        .expect("count-only optimizer rule should be native")
        .last_report()
        .expect("count-only probe should record its rewrite evidence");
    assert_eq!(probe_report.selected_count(), 1);
    assert_eq!(probe_report.not_selected_count(), 0);
    let after_final = final_native
        .optimize(after_probe, &ConfigOptions::default())
        .expect("the final native optimizer should preserve a count-only native rewrite");
    assert!(after_final.downcast_ref::<AlgeonNativeExec>().is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
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
            .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
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

#[test]
fn session_without_native_rule_has_no_native_optimizer_stages() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-cpu-only-count-probe");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let state = datafusion::execution::SessionStateBuilder::new_with_default_features().build();
    let rule_names = state.physical_optimizers().iter().map(|rule| rule.name()).collect::<Vec<_>>();
    assert!(!rule_names.contains(&"algeon_datafusion_native_count_only"));
    assert!(!rule_names.contains(&"algeon_datafusion_native"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_retains_unsupported_relational_parent_as_datafusion() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-partial-host-arrow-boundary");
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
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(Arc::ptr_eq(&optimized, &plan), "a relational lowering miss must retain the baseline DataFusion plan");
    assert_eq!(classify_final_plan(optimized.as_ref()), FinalPlanDisposition::DataFusion);
    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        !optimized_tree.contains("AlgeonNativeExec"),
        "ordinary relational DataFusion baseline must not retain a native fragment"
    );
    assert!(
        !optimized_tree.contains("AlgeonNativeGpuProducerExec"),
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
    let dir = unique_temp_dir("algeon-datafusion-native-exec-composition-no-datafusion-cpu");
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
    let config = AlgeonNativeOptimizerConfig::default()
        .with_datafusion_cpu_execution_policy(DataFusionCpuExecutionPolicy::Reject)
        .with_native_execution_config(bounded_config());
    let rules = installed_physical_optimizer_rules(config);
    let final_native = rules
        .iter()
        .find(|rule| rule.name() == "algeon_datafusion_native")
        .expect("native optimizer rules should contain the final native rule");
    let final_requirement = rules
        .iter()
        .find(|rule| rule.name() == ALGEON_FINAL_PLAN_REQUIREMENT_RULE_NAME)
        .expect("native optimizer rules should contain the final-plan requirement rule");

    let planned = final_native.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(Arc::ptr_eq(&planned, &plan), "a relational lowering miss must remain the original DataFusion plan");
    assert_eq!(classify_final_plan(planned.as_ref()), FinalPlanDisposition::DataFusion);
    let planned_tree = displayable(planned.as_ref()).indent(false).to_string();
    assert!(!planned_tree.contains("AlgeonNativeGpuProducerExec"));
    let report = (final_native.as_ref() as &dyn std::any::Any)
        .downcast_ref::<AlgeonNativePhysicalOptimizerRule>()
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

    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    // S3 admits scalar distinct aggregation over a keyed anti join through the
    // composed pipeline: the anti join executes with null-aware semantics and
    // the scalar distinct count is exact.
    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        optimized.downcast_ref::<AlgeonNativeExec>().is_some(),
        "distinct aggregate over anti join should execute natively:\n{optimized_tree}"
    );
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);

    // Row parity with the DataFusion baseline: no key survives the NOT IN.
    let batches =
        futures::executor::block_on(collect(Arc::clone(&optimized), Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(int64_array(&batches[0], 0).value(0), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
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

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert!(
        optimized.downcast_ref::<AlgeonNativeExec>().is_some(),
        "outcomes={:?}; plan={}",
        report.outcomes(),
        displayable(optimized.as_ref()).indent(false)
    );
    assert_eq!(report.selected_count(), 1, "outcomes={:?}", report.outcomes());
    assert_eq!(report.not_selected_count(), 0, "outcomes={:?}", report.outcomes());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_precision_sensitive_stddev_sort_subtree_after_order_contract() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-stddev-sort-cpu-boundary");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = stddev_sort_sql_plan(&path);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
    let optimized_tree = displayable(optimized.as_ref()).indent(false).to_string();
    assert!(
        optimized_tree.contains("AlgeonNativeExec"),
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
    let rules = installed_physical_optimizer_rules(native_optimizer_config());
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
    futures::executor::block_on(ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()))
        .unwrap();

    let df =
        futures::executor::block_on(ctx.sql("SELECT id, TRY_CAST(name AS INT) AS maybe_int FROM bench ORDER BY id"))
            .unwrap();
    let plan = futures::executor::block_on(df.create_physical_plan()).unwrap();
    let optimized_tree = displayable(plan.as_ref()).indent(false).to_string();
    assert!(
        optimized_tree.contains("AlgeonNativeExec"),
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

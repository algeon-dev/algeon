use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array};
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use futures::executor::block_on;
use query_engine::{
    optimizer::{
        JoinBuildSideSelectionBasis, JoinBuildSideSelectionCandidateDiagnostics, JoinBuildSideSelectionOptions,
        JoinBuildSideSelectionRejectedReason, NativeOptimizerOptions,
    },
    plan::JoinSide,
};

use super::*;
use crate::tpch_session::tpch_report_session_config;

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_left_semi_join_plan() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-bounded-left-semi-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_left_semi_join_plan(&left, &right);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 2);

    let batches = block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 3);
    let mut rows = (0..batches[0].num_rows())
        .map(|row| {
            (
                int32_array(&batches[0], 0).value(row),
                int32_array(&batches[0], 1).value(row),
                int64_array(&batches[0], 2).value(row),
            )
        })
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 5, 100), (1, 30, 300), (2, 20, 200)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_left_anti_join_plan() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-bounded-left-anti-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_left_anti_join_plan(&left, &right);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 2);

    let batches = block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 0);
    assert_eq!(batches[0].num_columns(), 3);

    fs::remove_dir_all(dir).unwrap();
}

fn null_aware_anti_join_rows(batches: &[RecordBatch]) -> Vec<(Option<i32>, i32)> {
    let mut rows = batches
        .iter()
        .flat_map(|batch| {
            let keys = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let payloads = batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..batch.num_rows()).map(|row| (keys.is_valid(row).then(|| keys.value(row)), payloads.value(row)))
        })
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows
}

fn assert_null_aware_anti_join_native_matches_datafusion(
    plan: Arc<dyn ExecutionPlan>,
    expected_rows: Vec<(Option<i32>, i32)>,
) {
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());
    let optimized = rule.optimize(Arc::clone(&plan), &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1, "report={report:?}; diagnostics={:?}", rule.last_planning_diagnostics());

    let native_rows =
        null_aware_anti_join_rows(&block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap());
    let datafusion_rows =
        null_aware_anti_join_rows(&block_on(collect(plan, Arc::new(TaskContext::default()))).unwrap());
    assert_eq!(native_rows, expected_rows);
    assert_eq!(datafusion_rows, expected_rows);
}

#[test]
fn native_optimizer_rule_executes_null_aware_left_anti_join_with_null_right_key_as_empty() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-bounded-null-aware-anti-join-null-right-key");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_null_aware_anti_join_left_parquet_file(&left);
    write_null_aware_anti_join_right_parquet_file(&right, vec![Some(1), None]);
    let plan = root_null_aware_left_anti_join_plan(&left, parquet_scan(&right));

    assert_null_aware_anti_join_native_matches_datafusion(plan, vec![]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_executes_null_aware_left_anti_join_drops_null_left_key_when_right_is_nonempty() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-bounded-null-aware-anti-join-drops-null-left-key");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_null_aware_anti_join_left_parquet_file(&left);
    write_null_aware_anti_join_right_parquet_file(&right, vec![Some(1), Some(2)]);
    let plan = root_null_aware_left_anti_join_plan(&left, parquet_scan(&right));

    assert_null_aware_anti_join_native_matches_datafusion(plan, vec![(Some(4), 40)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_executes_null_aware_left_anti_join_keeps_null_left_key_when_right_is_empty_at_runtime() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-bounded-null-aware-anti-join-empty-right");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_null_aware_anti_join_left_parquet_file(&left);
    write_null_aware_anti_join_right_parquet_file(&right, vec![Some(7)]);
    let plan = root_null_aware_left_anti_join_plan(&left, filter_scan_to_empty_at_runtime(parquet_scan(&right)));

    assert_null_aware_anti_join_native_matches_datafusion(
        plan,
        vec![(None, 30), (Some(1), 10), (Some(2), 20), (Some(4), 40)],
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_bounded_q11_aggregate_over_join_subplan_inputs() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-bounded-q11-admission");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q11 = TPCH_QUERIES.iter().find(|query| query.id == "q11").expect("TPC-H Q11 query must exist");
    let plan = tpch_sql_plan(&dir, q11.sql);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert!(
        optimized.downcast_ref::<AlgeonNativeExec>().is_some(),
        "Q11 should optimize to AlgeonNativeExec; report={report:?}; diagnostics={:?}",
        rule.last_planning_diagnostics()
    );
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));

    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain diagnostics");
    assert_eq!(diagnostics.capability_contract.accepted_candidates.len(), 1);
    assert!(diagnostics.capability_contract.rejected_candidates.is_empty());
    let accepted = &diagnostics.capability_contract.accepted_candidates[0];
    assert_eq!(accepted.compiled_graph.sink_kind, "host_arrow_materialization");
    assert!(accepted.compiled_graph.resource_count > 0);
    let admission_report = &accepted.report;
    assert_eq!(admission_report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
    assert!(admission_report.is_ready());
    assert!(admission_report.decision_rows().is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_root_hash_join() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-bounded-root-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_hash_join_plan(&left, &right);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<AlgeonNativeExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));

    let batches = block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
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

/// `dim ⋈ mid ⋈ fact` where DataFusion keeps the join-fed side as its own
/// build side while native lowering defaults to building the parquet leaf.
/// Only DataFusion's advisory output estimate can tell the optimizer that the
/// join-fed side is the smaller build; the leaf side keeps its conservative
/// footer estimate.
#[test]
fn native_optimizer_rule_builds_join_fed_side_from_advisory_statistics_and_matches_cpu_results() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-advisory-build-side");
    write_advisory_build_side_fixture(&dir, false);

    let (candidate, native_rows, cpu_rows) = run_advisory_build_side_query(&dir, true);

    assert_eq!(candidate.before_build_side, JoinSide::Right);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    assert!(candidate.applied);
    assert_eq!(candidate.selection_basis, Some(JoinBuildSideSelectionBasis::AdvisoryRows));
    assert_eq!(candidate.left_estimated_rows, None);
    assert_eq!(candidate.right_estimated_rows, Some(2_000));
    let advised_rows = candidate.left_advisory_rows.expect("DataFusion should estimate the join-fed side");
    assert!(advised_rows < 2_000, "advisory rows {advised_rows} should undercut the leaf estimate");
    assert_eq!(native_rows, cpu_rows);
    assert_eq!(native_rows.len(), 100);

    // Without frontend statistics there is no advisory signal: the pass keeps
    // the lowering default, the plan stays selected, and results still match.
    let (candidate, native_rows, cpu_rows) = run_advisory_build_side_query(&dir, false);

    assert_eq!(candidate.before_build_side, JoinSide::Right);
    assert_eq!(candidate.after_build_side, None);
    assert!(!candidate.applied);
    assert_eq!(candidate.selection_basis, None);
    assert_eq!(candidate.left_advisory_rows, None);
    assert_eq!(candidate.rejected_reason, Some(JoinBuildSideSelectionRejectedReason::NoBetterBuildSide));
    assert_eq!(native_rows, cpu_rows);
    assert_eq!(native_rows.len(), 100);

    fs::remove_dir_all(dir).unwrap();
}

/// Same shape with skewed keys: DataFusion's estimate for `dim ⋈ mid` is far
/// below its true size, so the advisory-selected build side is actually the
/// larger input. The orientation is only a strategy choice; results must
/// still match.
#[test]
fn native_optimizer_rule_keeps_results_correct_when_advisory_build_side_is_misestimated() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-advisory-build-side-misestimate");
    write_advisory_build_side_fixture(&dir, true);

    let (candidate, native_rows, cpu_rows) = run_advisory_build_side_query(&dir, true);

    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    assert_eq!(candidate.selection_basis, Some(JoinBuildSideSelectionBasis::AdvisoryRows));
    let advised_rows = candidate.left_advisory_rows.expect("DataFusion should estimate the join-fed side");
    // dim keeps 50 copies of key 1 and mid keys are all 1, so the true join
    // output is 5,000 rows while the range-based NDV estimate stays near 200.
    assert!(advised_rows < 2_000, "advisory rows {advised_rows} should undercut the leaf estimate");
    assert_eq!(native_rows, cpu_rows);
    assert_eq!(native_rows.len(), 100);
    let total: i64 = native_rows.iter().map(|(_, total)| *total).sum();
    let fact_total: i64 = (0..2_000_i64).sum();
    assert_eq!(total, fact_total * 50);

    fs::remove_dir_all(dir).unwrap();
}

const ADVISORY_BUILD_SIDE_SQL: &str = "
SELECT m.m_id, SUM(f.f_val) AS total
FROM dim d
JOIN mid m ON d.d_key = m.m_dkey
JOIN fact f ON m.m_id = f.f_mid
GROUP BY m.m_id
ORDER BY m.m_id
LIMIT 1000
";

fn write_advisory_build_side_fixture(dir: &std::path::Path, skewed_dim_keys: bool) {
    let dim_keys = if skewed_dim_keys {
        std::iter::repeat_n(1, 50).chain(2..=51).collect::<Vec<i32>>()
    } else {
        (1..=100).collect::<Vec<i32>>()
    };
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("d_key", DataType::Int32, false),
        Field::new("d_pad", DataType::Int64, false),
    ]));
    let dim = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![
            Arc::new(Int32Array::from(dim_keys.clone())),
            Arc::new(Int64Array::from(dim_keys.iter().map(|key| i64::from(*key) * 10).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    write_batch(&dir.join("dim.parquet"), dim_schema, dim);

    let mid_schema = Arc::new(Schema::new(vec![
        Field::new("m_id", DataType::Int32, false),
        Field::new("m_dkey", DataType::Int32, false),
    ]));
    let mid = RecordBatch::try_new(
        Arc::clone(&mid_schema),
        vec![Arc::new(Int32Array::from((1..=100).collect::<Vec<i32>>())), Arc::new(Int32Array::from(vec![1; 100]))],
    )
    .unwrap();
    write_batch(&dir.join("mid.parquet"), mid_schema, mid);

    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("f_mid", DataType::Int32, false),
        Field::new("f_val", DataType::Int64, false),
    ]));
    let fact = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![
            Arc::new(Int32Array::from((0..2_000).map(|row| row % 100 + 1).collect::<Vec<i32>>())),
            Arc::new(Int64Array::from((0..2_000_i64).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    write_batch(&dir.join("fact.parquet"), fact_schema, fact);
}

/// Outer-join build-side candidate plus the native and CPU `(m_id, total)` rows.
type AdvisoryBuildSideRun = (JoinBuildSideSelectionCandidateDiagnostics, Vec<(i32, i64)>, Vec<(i32, i64)>);

fn run_advisory_build_side_query(dir: &std::path::Path, collect_statistics: bool) -> AdvisoryBuildSideRun {
    let mut config = tpch_report_session_config();
    config.options_mut().execution.collect_statistics = collect_statistics;
    let ctx = SessionContext::new_with_config(config);
    for table in ["dim", "mid", "fact"] {
        let path = dir.join(format!("{table}.parquet"));
        block_on(ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    }
    let plan = |sql: &str| {
        let df = block_on(ctx.sql(sql)).unwrap();
        block_on(df.create_physical_plan()).unwrap()
    };
    let cpu = block_on(collect(plan(ADVISORY_BUILD_SIDE_SQL), Arc::new(TaskContext::default()))).unwrap();

    // The fixture is tiny, so lower the small-build thresholds below the
    // leaf's footer estimate; otherwise the pass stops at "current side is
    // small" before any advisory evidence is consulted.
    let config = native_optimizer_config();
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions {
            small_build_side_bytes: 1024,
            small_build_side_rows: 100,
            ..config.native_optimizer_options().join_build_side_selection.clone()
        },
        ..config.native_optimizer_options().clone()
    };
    let rule = AlgeonNativePhysicalOptimizerRule::new(config.with_native_optimizer_options(optimizer_options));
    let optimized = rule.optimize(plan(ADVISORY_BUILD_SIDE_SQL), &ConfigOptions::default()).unwrap();
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1, "outcomes={:?}", report.outcomes());
    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain planning diagnostics");
    let candidate = diagnostics
        .native_optimizer
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.left_estimated_rows.is_none() && candidate.right_estimated_rows.is_some())
        .cloned()
        .expect("the join-fed outer join should be a build-side candidate");
    let native = block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();

    (candidate, grouped_total_rows(&native), grouped_total_rows(&cpu))
}

fn grouped_total_rows(batches: &[RecordBatch]) -> Vec<(i32, i64)> {
    batches
        .iter()
        .flat_map(|batch| {
            (0..batch.num_rows()).map(|row| (int32_array(batch, 0).value(row), int64_array(batch, 1).value(row)))
        })
        .collect()
}

#[test]
fn bounded_planner_reports_compiled_graph_for_supported_join_path() {
    let dir = unique_temp_dir("algeon-datafusion-bounded-join-planner-rows");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_nested_loop_join_plan(&left, &right);
    let rule = AlgeonNativePhysicalOptimizerRule::new(native_optimizer_config());
    let _ = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain diagnostics");

    assert_eq!(diagnostics.capability_contract.accepted_candidates.len(), 1);
    assert!(diagnostics.capability_contract.rejected_candidates.is_empty());
    let accepted = &diagnostics.capability_contract.accepted_candidates[0];
    assert!(accepted.report.decision_rows().is_empty());
    assert!(accepted.compiled_graph.resource_count > 0);

    fs::remove_dir_all(dir).unwrap();
}

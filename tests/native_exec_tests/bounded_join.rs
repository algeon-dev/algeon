use std::sync::Arc;

use super::*;

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_left_semi_join_plan() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-bounded-left-semi-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_left_semi_join_plan(&left, &right);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NexusNativeExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 2);

    let batches = futures::executor::block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
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
    let dir = unique_temp_dir("datafusion-nexus-native-exec-bounded-left-anti-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_left_anti_join_plan(&left, &right);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NexusNativeExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 2);

    let batches = futures::executor::block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 0);
    assert_eq!(batches[0].num_columns(), 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_bounded_q11_aggregate_over_join_subplan_inputs() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-bounded-q11-admission");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q11 = TPCH_QUERIES.iter().find(|query| query.id == "q11").expect("TPC-H Q11 query must exist");
    let plan = tpch_sql_plan(&dir, q11.sql);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert!(
        optimized.downcast_ref::<NexusNativeExec>().is_some(),
        "Q11 should optimize to NexusNativeExec; report={report:?}; diagnostics={:?}",
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
    assert!(accepted.compiled_graph.gate_count > 0);
    assert!(accepted.compiled_graph.sink_gate_count > 0);
    let admission_report = &accepted.report;
    assert_eq!(admission_report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
    assert!(admission_report.is_ready());
    assert_admission_report_has_host_sink_boundary(admission_report);
    assert_admission_report_has_join_path_gates(admission_report);
    assert!(
        admission_report
            .stable_kv_rows()
            .iter()
            .all(|row| !row.contains("bounded_join_execution_family") && !row.contains("BoundedJoinExecutionPath")),
        "bounded planner diagnostics must not require family/path vocabulary"
    );

    fs::remove_dir_all(dir).unwrap();
}

fn assert_admission_report_has_host_sink_boundary(
    report: &nexus_query_engine::contract_report::CapabilityContractReport,
) {
    assert!(
        report.gate_rows().iter().any(|row| {
            row.owner.to_string() == "sink_boundary"
                && row.reservation_kind == ReservationKind::Materialization
                && row.memory_boundary == "host_materialization"
                && row.required_contract == "query_engine_explicit_sink_materialization_reservation"
                && row.contract_quality == ContractQuality::QueryEngineOwnedBoundary
        }),
        "accepted bounded diagnostics must identify the host materialization sink boundary"
    );
}

fn assert_admission_report_has_join_path_gates(report: &nexus_query_engine::contract_report::CapabilityContractReport) {
    for kind in [
        ReservationKind::SourceRead,
        ReservationKind::BuildSide,
        ReservationKind::ProbeSide,
        ReservationKind::HashTable,
        ReservationKind::GatherMap,
        ReservationKind::JoinOutput,
        ReservationKind::Materialization,
    ] {
        assert!(
            report.gate_rows().iter().any(|row| row.reservation_kind == kind),
            "accepted bounded diagnostics should include reservation gate {kind:?}"
        );
    }

    let stable_rows = report.stable_kv_rows().join("|");
    for retired_term in [
        "BoundedJoinExecutionFamily",
        "BoundedJoinExecutionPath",
        "bounded_join_execution_family",
        "bounded_join_execution_families",
        "bounded_join_required_shape_contract",
        "bounded_join_contract_modes",
        "bounded_join_unresolved_preflight_contracts",
    ] {
        assert!(
            !stable_rows.contains(retired_term),
            "bounded diagnostics must not contain retired family/path term {retired_term}"
        );
    }
}

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_root_hash_join() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-bounded-root-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_hash_join_plan(&left, &right);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.downcast_ref::<NexusNativeExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));

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
fn bounded_planner_rows_cover_supported_join_path() {
    let dir = unique_temp_dir("datafusion-nexus-bounded-join-planner-rows");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_nested_loop_join_plan(&left, &right);
    let rule = NexusNativePhysicalOptimizerRule::new(native_optimizer_config());
    let _ = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain diagnostics");

    assert_eq!(diagnostics.capability_contract.accepted_candidates.len(), 1);
    assert!(diagnostics.capability_contract.rejected_candidates.is_empty());
    let admission_rows = diagnostics.capability_contract.accepted_candidates[0].report.gate_rows();
    assert!(admission_rows.iter().any(|row| row.reservation_kind == ReservationKind::Materialization));
    assert!(
        admission_rows
            .iter()
            .map(|row| row.to_stable_kv_row())
            .all(|row| !row.contains("bounded_join_execution_family") && !row.contains("BoundedJoinExecutionPath"))
    );

    fs::remove_dir_all(dir).unwrap();
}

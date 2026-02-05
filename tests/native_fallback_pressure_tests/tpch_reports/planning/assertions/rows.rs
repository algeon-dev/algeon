use super::*;

pub(super) fn assert_planning_rows(report: &NativeDataFusionPlanningReport) {
    for row in &report.rows {
        if EXPECTED_BOUNDED_TPCH_ACCEPTED.contains(&row.query_id.as_str()) {
            assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
            assert_eq!(row.reason_code.as_deref(), None);
            assert_eq!(row.unsupported_category.as_deref(), None);
            if EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED.contains(&row.query_id.as_str()) {
                assert!(
                    !row.strict_bounded_contract_rows.is_empty(),
                    "{} accepted bounded join row should expose strict contract gates",
                    row.query_id
                );
                assert!(
                    !row.strict_bounded_compiled_graphs.is_empty(),
                    "{} accepted bounded join row should expose compiled graph diagnostics",
                    row.query_id
                );
                assert!(
                    row.strict_bounded_contract_rows.iter().all(|gate| {
                        gate.execution_mode == StrictBoundedExecutionMode::StrictBounded
                            && gate.contract_status == ContractStatus::Modeled
                    }),
                    "{} accepted strict bounded gates should be modeled strict-bounded rows",
                    row.query_id
                );
                assert!(
                    row.strict_bounded_contract_rows
                        .iter()
                        .any(|gate| gate.reservation_kind == ReservationKind::Materialization),
                    "{} accepted strict bounded gates should include host materialization",
                    row.query_id
                );
                assert!(
                    row.strict_bounded_contract_rows
                        .iter()
                        .map(|gate| gate.to_stable_kv_row())
                        .all(|rendered| strict_contract_row_is_family_path_free(&rendered)),
                    "{} strict contract rows should not preserve family/path vocabulary",
                    row.query_id
                );
            } else {
                assert!(
                    row.strict_bounded_contract_rows.is_empty(),
                    "{} non-join accepted row is outside the current strict join gate report surface",
                    row.query_id
                );
                assert!(row.strict_bounded_compiled_graphs.is_empty());
            }
            assert_eq!(row.post_native_plan_root, "NativeCudfExec");
            assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            assert!(!row.rejected_before_execution, "{} should be replaced by native execution", row.query_id);
        } else {
            let (expected_reason, expected_category) =
                expected_tpch_datafusion_planning_rejection(row.query_id.as_str());
            assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::RejectedBeforeExecution);
            assert_eq!(row.reason_code.as_deref(), Some(expected_reason), "unexpected reason for {}", row.query_id);
            assert_eq!(
                row.unsupported_category.as_deref(),
                Some(expected_category),
                "unexpected unsupported category for {}",
                row.query_id
            );
            assert!(
                row.strict_bounded_contract_rows
                    .iter()
                    .map(|gate| gate.to_stable_kv_row())
                    .all(|rendered| strict_contract_row_is_family_path_free(&rendered)),
                "{} rejected strict contract rows should not preserve family/path vocabulary",
                row.query_id
            );
            assert_ne!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
            assert!(row.rejected_before_execution, "{} should reject before native execution", row.query_id);
        }
        assert_eq!(row.physical_root, expected_tpch_physical_root(row.query_id.as_str()));
        assert_eq!(
            row.native_candidate_shape.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape(row.query_id.as_str()),
            "unexpected native candidate shape for {}",
            row.query_id
        );
        assert_eq!(
            row.native_candidate_shape_family.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape_family(row.query_id.as_str()),
            "unexpected native candidate shape family for {}",
            row.query_id
        );
        assert!(
            row.physical_shape.contains(expected_tpch_physical_shape_marker(row.query_id.as_str())),
            "unexpected physical shape for {}: {}",
            row.query_id,
            row.physical_shape
        );
        if EXPECTED_BOUNDED_TPCH_JOIN_ACCEPTED.contains(&row.query_id.as_str())
            && row.outcome == NativeDataFusionPlanningOutcome::AcceptedReplaced
        {
            assert!(
                row.join_diagnostics.is_empty(),
                "{} accepted replacement should not expose join diagnostics",
                row.query_id
            );
        } else {
            assert_tpch_join_diagnostics(row.query_id.as_str(), &row.join_diagnostics);
        }
    }
}

pub(super) fn strict_contract_row_is_family_path_free(rendered: &str) -> bool {
    !rendered.contains("bounded_join_execution_family")
        && !rendered.contains("bounded_join_required_shape_contract")
        && !rendered.contains("bounded_join_contract_modes")
        && !rendered.contains("bounded_join_unresolved_preflight_contracts")
        && !rendered.contains("BoundedJoinExecutionFamily")
        && !rendered.contains("BoundedJoinExecutionPath")
}

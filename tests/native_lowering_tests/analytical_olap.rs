use super::*;

#[tokio::test]
async fn native_pipeline_analytical_olap_workload_lowers_into_native_query_plans() {
    let dir = unique_temp_dir("datafusion-nexus-analytical-olap-lowering");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let items = analytical_olap_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();

    assert_eq!(items.len(), ANALYTICAL_OLAP_QUERY_COUNT);
    assert_eq!(
        items.iter().map(|item| item.query_id.as_str()).collect::<Vec<_>>(),
        ANALYTICAL_OLAP_QUERIES.iter().map(|query| query.id).collect::<Vec<_>>()
    );
    assert!(
        ANALYTICAL_OLAP_QUERIES.iter().any(
            |query| query.capability_first_admission == AnalyticalOlapCapabilityFirstAdmission::ExpectedSupersetWin
        ),
        "analytical supplement must carry at least one expected capability-first superset win"
    );

    for query in ANALYTICAL_OLAP_QUERIES {
        assert_eq!(query.expected_lowering, AnalyticalOlapLoweringExpectation::NativeQueryPlan);
        let item = items.iter().find(|item| item.query_id == query.id).expect("analytical OLAP item should exist");
        let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &item.input else {
            panic!("{} should produce a physical plan", query.id);
        };
        let result = lower_physical_plan_to_native_for_test(Arc::clone(plan));
        assert_eq!(result.candidate_count(), 1, "{} outcomes={:?}", query.id, result.report.outcomes());
        assert_eq!(result.selected_count(), 1, "{} outcomes={:?}", query.id, result.report.outcomes());
        assert_eq!(result.not_selected_count(), 0, "{} outcomes={:?}", query.id, result.report.outcomes());
        assert!(result.plans[0].contains_join(), "{} should lower to a join-bearing native plan", query.id);
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_pipeline_analytical_olap_workload_has_required_shape_coverage() {
    let classes = ANALYTICAL_OLAP_QUERIES.iter().map(|query| query.workload_class).collect::<BTreeSet<_>>();

    assert_eq!(
        classes,
        BTreeSet::from([
            AnalyticalOlapWorkloadClass::AggregateSortTopK,
            AnalyticalOlapWorkloadClass::DecorrelatableSubquery,
            AnalyticalOlapWorkloadClass::MultiJoinResidentPressure,
            AnalyticalOlapWorkloadClass::SelfJoin,
            AnalyticalOlapWorkloadClass::StarSchemaJoin,
        ])
    );
}

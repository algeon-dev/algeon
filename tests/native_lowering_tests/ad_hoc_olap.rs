use super::*;

#[tokio::test]
async fn native_pipeline_ad_hoc_olap_workload_lowers_into_native_query_plans() {
    let dir = unique_temp_dir("datafusion-cudf-ad-hoc-olap-lowering");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let items = ad_hoc_olap_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();

    assert_eq!(items.len(), AD_HOC_OLAP_QUERY_COUNT);
    assert_eq!(
        items.iter().map(|item| item.query_id.as_str()).collect::<Vec<_>>(),
        ad_hoc_olap_query_ids().collect::<Vec<_>>()
    );
    assert!(
        AD_HOC_OLAP_QUERIES
            .iter()
            .any(|query| query.capability_first_admission == AdHocOlapCapabilityFirstAdmission::ExpectedSupersetWin),
        "ad-hoc supplement must carry at least one expected capability-first superset win"
    );

    for query in AD_HOC_OLAP_QUERIES {
        assert_eq!(query.expected_lowering, AdHocOlapLoweringExpectation::NativeQueryPlan);
        let item = items.iter().find(|item| item.query_id == query.id).expect("ad-hoc OLAP item should exist");
        let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &item.input else {
            panic!("{} should produce a physical plan", query.id);
        };
        let result = lower_physical_plan_to_native(Arc::clone(plan));
        assert_eq!(result.candidate_count(), 1, "{} outcomes={:?}", query.id, result.report.outcomes());
        assert_eq!(result.rewritten_count(), 1, "{} outcomes={:?}", query.id, result.report.outcomes());
        assert_eq!(result.rejected_count(), 0, "{} outcomes={:?}", query.id, result.report.outcomes());
        assert!(result.plans[0].contains_join(), "{} should lower to a join-bearing native plan", query.id);
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_pipeline_ad_hoc_olap_workload_has_required_shape_coverage() {
    let classes = AD_HOC_OLAP_QUERIES.iter().map(|query| query.workload_class.as_str()).collect::<BTreeSet<_>>();

    assert_eq!(
        classes,
        BTreeSet::from([
            "aggregate_sort_topk",
            "decorrelatable_subquery",
            "multi_join_resident_pressure",
            "self_join",
            "star_schema_join",
        ])
    );
}

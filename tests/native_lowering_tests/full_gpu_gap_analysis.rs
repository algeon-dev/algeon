use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum GpuCoverageSqlFeature {
    AggregateFilter,
    AggregateFunction,
    Window,
    ScalarExpression,
    RootSource,
    ExistingCoverage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectedGpuCoverageOutcome {
    Selected,
    NotSupported { reason: CapabilityReason, unsupported_category: &'static str },
}

#[derive(Debug, Clone, Copy)]
struct GpuCoverageSqlCase {
    id: &'static str,
    feature: GpuCoverageSqlFeature,
    sql: &'static str,
    expected: ExpectedGpuCoverageOutcome,
}

const AGGREGATE_FILTER_CASES: &[GpuCoverageSqlCase] = &[
    GpuCoverageSqlCase {
        id: "aggregate_filter_sum",
        feature: GpuCoverageSqlFeature::AggregateFilter,
        sql: "SELECT key, SUM(payload) FILTER (WHERE flag) AS s FROM bench GROUP BY key",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    // TRY_CAST(string AS int) is admitted natively (Utf8/LargeUtf8/Utf8View →
    // any signed/unsigned integer); the resulting nullable predicate is a
    // valid aggregate FILTER expression.
    GpuCoverageSqlCase {
        id: "aggregate_filter_try_cast",
        feature: GpuCoverageSqlFeature::AggregateFilter,
        sql: "SELECT key, SUM(payload) FILTER (WHERE TRY_CAST(name AS INT) > 0) AS s FROM bench GROUP BY key",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
];

const AGGREGATE_FUNCTION_CASES: &[GpuCoverageSqlCase] = &[GpuCoverageSqlCase {
    id: "aggregate_approx_distinct",
    feature: GpuCoverageSqlFeature::AggregateFunction,
    sql: "SELECT APPROX_DISTINCT(name) AS distinct_names FROM bench",
    expected: ExpectedGpuCoverageOutcome::NotSupported {
        reason: CapabilityReason::UnsupportedOperator,
        unsupported_category: "aggregate_approx_distinct",
    },
}];

const WINDOW_CASES: &[GpuCoverageSqlCase] = &[
    GpuCoverageSqlCase {
        id: "window_lag",
        feature: GpuCoverageSqlFeature::Window,
        sql: "SELECT id, LAG(payload) OVER (PARTITION BY key ORDER BY id) AS prev_payload FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "window_lag_offset_default",
        feature: GpuCoverageSqlFeature::Window,
        sql: "SELECT id, LAG(payload, 2, 0) OVER (PARTITION BY key ORDER BY id) AS prev_payload \
              FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "window_lead",
        feature: GpuCoverageSqlFeature::Window,
        sql: "SELECT id, LEAD(payload) OVER (PARTITION BY key ORDER BY id) AS next_payload FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "window_row_number",
        feature: GpuCoverageSqlFeature::Window,
        sql: "SELECT id, ROW_NUMBER() OVER (PARTITION BY key ORDER BY payload) AS rn FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "window_bounded_sum",
        feature: GpuCoverageSqlFeature::Window,
        sql: "SELECT id, SUM(payload) OVER (PARTITION BY key ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM bench",
        expected: ExpectedGpuCoverageOutcome::NotSupported {
            reason: CapabilityReason::UnsupportedOperator,
            unsupported_category: "window_frame_rows_1_preceding_to_current_row",
        },
    },
];

const SCALAR_EXPRESSION_CASES: &[GpuCoverageSqlCase] = &[
    // TRY_CAST(string AS integer) is admitted natively. DataFusion projection
    // pushdown collapses the cast into the DataSourceExec projection; the
    // adapter still recognises it as a supported try-cast.
    GpuCoverageSqlCase {
        id: "try_cast_string_to_int",
        feature: GpuCoverageSqlFeature::ScalarExpression,
        sql: "SELECT id, TRY_CAST(name AS INT) AS maybe_int FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    // TRY_CAST to a non-integer target (Float64) remains unsupported until
    // libcudf grows a string→float null-on-failure primitive; the structured
    // reject category still pins the unsupported type pair.
    GpuCoverageSqlCase {
        id: "try_cast_string_to_float",
        feature: GpuCoverageSqlFeature::ScalarExpression,
        sql: "SELECT id, TRY_CAST(name AS DOUBLE) AS maybe_float FROM bench",
        expected: ExpectedGpuCoverageOutcome::NotSupported {
            reason: CapabilityReason::UnsupportedExpression,
            unsupported_category: "expression_try_cast_utf8view_to_float64",
        },
    },
    // TRY_CAST whose source is not a string column (here a numeric column)
    // also stays unsupported in v1 — the lower-level primitive is
    // string→integer only.
    GpuCoverageSqlCase {
        id: "try_cast_int_to_int",
        feature: GpuCoverageSqlFeature::ScalarExpression,
        sql: "SELECT id, TRY_CAST(payload AS SMALLINT) AS narrowed FROM bench",
        expected: ExpectedGpuCoverageOutcome::NotSupported {
            reason: CapabilityReason::UnsupportedExpression,
            unsupported_category: "expression_try_cast_int64_to_int16",
        },
    },
    GpuCoverageSqlCase {
        id: "integer_to_timestamp_cast",
        feature: GpuCoverageSqlFeature::ScalarExpression,
        sql: "SELECT id, CAST(payload AS TIMESTAMP) AS ts FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "sqrt_scalar_function",
        feature: GpuCoverageSqlFeature::ScalarExpression,
        sql: "SELECT id, sqrt(score) AS root_score FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "string_concat_operator",
        feature: GpuCoverageSqlFeature::ScalarExpression,
        sql: "SELECT id, name || '-x' AS label FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "is_distinct_from",
        feature: GpuCoverageSqlFeature::ScalarExpression,
        sql: "SELECT id FROM bench WHERE name IS DISTINCT FROM 'alpha'",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
];

const ROOT_SOURCE_CASES: &[GpuCoverageSqlCase] = &[
    GpuCoverageSqlCase {
        id: "offset_without_limit",
        feature: GpuCoverageSqlFeature::RootSource,
        sql: "SELECT id, payload FROM bench OFFSET 2",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "explain_query",
        feature: GpuCoverageSqlFeature::RootSource,
        sql: "EXPLAIN SELECT id FROM bench WHERE key = 10",
        expected: ExpectedGpuCoverageOutcome::NotSupported {
            reason: CapabilityReason::UnsupportedOperator,
            unsupported_category: "operator_explain",
        },
    },
    GpuCoverageSqlCase {
        id: "literal_only_projection",
        feature: GpuCoverageSqlFeature::RootSource,
        sql: "SELECT 1 AS one",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "values_table",
        feature: GpuCoverageSqlFeature::RootSource,
        sql: "SELECT a, b FROM (VALUES (1, 10), (2, 20)) AS t(a, b)",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
];

const EXISTING_COVERAGE_CASES: &[GpuCoverageSqlCase] = &[
    GpuCoverageSqlCase {
        id: "filter_projection",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT id, payload + 1 AS payload_plus FROM bench WHERE key = 10",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "grouped_aggregate_order_limit",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT key, SUM(payload) AS s FROM bench GROUP BY key ORDER BY s DESC, key ASC LIMIT 2",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "having_after_aggregate",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT key, SUM(payload) AS s FROM bench GROUP BY key HAVING SUM(payload) > 10",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "self_hash_join_aggregate",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT b.key, COUNT(*) AS rows FROM bench b JOIN bench s ON b.key = s.key GROUP BY b.key",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "left_join",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT b.id, s.payload AS matched_payload FROM bench b LEFT JOIN bench s ON b.key = s.key",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "non_equi_inner_join",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT b.id, s.id AS sid FROM bench b JOIN bench s ON b.payload < s.payload",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "union_all",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT key FROM bench WHERE flag UNION ALL SELECT key FROM bench WHERE payload > 10",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "count_distinct",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT COUNT(DISTINCT name) AS distinct_names FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "avg_distinct",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT AVG(DISTINCT payload) AS avg_payload FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "rank_window",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT id, RANK() OVER (PARTITION BY key ORDER BY payload) AS r FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "starts_with_like_rewrite",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT id FROM bench WHERE starts_with(name, 'a')",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "regexp_replace",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT id, regexp_replace(name, 'a', 'x') AS rewritten_name FROM bench",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "order_limit_offset",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT id, payload FROM bench ORDER BY payload ASC NULLS LAST LIMIT 2 OFFSET 1",
        expected: ExpectedGpuCoverageOutcome::NotSupported {
            reason: CapabilityReason::NativeOrderedOutputUnsupported,
            unsupported_category: "operator_ordering",
        },
    },
    GpuCoverageSqlCase {
        id: "intersect_rewrite",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT key FROM bench WHERE flag INTERSECT SELECT key FROM bench WHERE payload > 10",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "except_rewrite",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT key FROM bench EXCEPT SELECT key FROM bench WHERE flag",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "exists_rewrite",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT id FROM bench b WHERE EXISTS (SELECT 1 FROM bench s WHERE s.key = b.key AND s.flag)",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
    GpuCoverageSqlCase {
        id: "rollup",
        feature: GpuCoverageSqlFeature::ExistingCoverage,
        sql: "SELECT key, SUM(payload) AS s FROM bench GROUP BY ROLLUP(key)",
        expected: ExpectedGpuCoverageOutcome::Selected,
    },
];

#[tokio::test]
async fn full_gpu_gap_analysis_sql_planning_coverage_is_stable() {
    let dir = unique_temp_dir("datafusion-nexus-full-gpu-gap-analysis");
    let path = dir.join("bench.parquet");
    write_full_gpu_gap_analysis_parquet_file(&path);

    for case in gpu_coverage_sql_cases() {
        let plan = sql_plan(&path, direct_friendly_sql_session_config(), case.sql).await;
        let lowering = lower_physical_plan_to_native_for_test(Arc::clone(&plan));
        assert_planning_report(&case, &lowering);
        assert_native_ir_details(&case, &lowering);

        let report = NativeFallbackPressureReport::collect([NativeFallbackPressureWorkloadItem::new(case.id, plan)]);
        assert_eq!(report.rows.len(), 1, "{}", case.id);
        let row = &report.rows[0];
        match case.expected {
            ExpectedGpuCoverageOutcome::Selected => {
                assert_eq!(row.outcome, Some(NativeFallbackPressureOutcome::Selected), "{}", case.id);
                assert_eq!(row.reason_code, None, "{}", case.id);
                assert_eq!(row.unsupported_category, None, "{}", case.id);
                assert!(!row.rejected_before_execution, "{}", case.id);
            },
            ExpectedGpuCoverageOutcome::NotSupported { reason, unsupported_category } => {
                assert_eq!(row.outcome, Some(NativeFallbackPressureOutcome::NotSupported), "{}", case.id);
                assert_eq!(row.reason_code.as_deref(), Some(reason.code()), "{}", case.id);
                assert_eq!(row.unsupported_category.as_deref(), Some(unsupported_category), "{}", case.id);
                assert!(!row.rejected_before_execution, "{}", case.id);
            },
        }
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn full_gpu_gap_analysis_sql_fixtures_are_grouped_by_feature() {
    let groups = gpu_coverage_sql_cases().iter().map(|case| case.feature).collect::<BTreeSet<_>>();
    assert_eq!(
        groups,
        BTreeSet::from([
            GpuCoverageSqlFeature::AggregateFilter,
            GpuCoverageSqlFeature::AggregateFunction,
            GpuCoverageSqlFeature::Window,
            GpuCoverageSqlFeature::ScalarExpression,
            GpuCoverageSqlFeature::RootSource,
            GpuCoverageSqlFeature::ExistingCoverage,
        ])
    );
}

fn gpu_coverage_sql_cases() -> Vec<GpuCoverageSqlCase> {
    [
        AGGREGATE_FILTER_CASES,
        AGGREGATE_FUNCTION_CASES,
        WINDOW_CASES,
        SCALAR_EXPRESSION_CASES,
        ROOT_SOURCE_CASES,
        EXISTING_COVERAGE_CASES,
    ]
    .into_iter()
    .flatten()
    .copied()
    .collect()
}

fn assert_planning_report(case: &GpuCoverageSqlCase, lowering: &NativeLoweringResult) {
    assert_eq!(lowering.candidate_count(), 1, "{} outcomes={:?}", case.id, lowering.report.outcomes());
    match case.expected {
        ExpectedGpuCoverageOutcome::Selected => {
            assert_eq!(lowering.selected_count(), 1, "{} outcomes={:?}", case.id, lowering.report.outcomes());
            assert_eq!(lowering.not_selected_count(), 0, "{} outcomes={:?}", case.id, lowering.report.outcomes());
            assert_eq!(lowering.plans.len(), 1, "{}", case.id);
            assert!(matches!(lowering.report.outcomes()[0], CandidateOutcome::Selected { .. }), "{}", case.id);
        },
        ExpectedGpuCoverageOutcome::NotSupported { reason, .. } => {
            assert!(lowering.plans.is_empty(), "{}", case.id);
            assert_eq!(lowering.selected_count(), 0, "{} outcomes={:?}", case.id, lowering.report.outcomes());
            assert_eq!(lowering.not_selected_count(), 1, "{} outcomes={:?}", case.id, lowering.report.outcomes());
            assert!(
                matches!(lowering.report.outcomes()[0], CandidateOutcome::NotSupported { capability_reason: actual, .. } if actual == reason),
                "{} outcomes={:?}",
                case.id,
                lowering.report.outcomes()
            );
        },
    }
}

fn assert_native_ir_details(case: &GpuCoverageSqlCase, lowering: &NativeLoweringResult) {
    match case.id {
        "window_row_number" => {
            let plan = lowering.plans.first().expect("row_number case should lower to a native plan");
            assert!(
                plan.nodes().iter().any(|node| {
                    matches!(
                        &node.kind,
                        nexus_query_engine::plan::PlanNode::Window(window)
                            if window
                                .spec
                                .functions
                                .iter()
                                .any(|function| matches!(function, nexus_query_engine::plan::WindowFunction::RowNumber { .. }))
                    )
                }),
                "window_row_number should lower to native RowNumber IR"
            );
        },
        "window_lag" => {
            let plan = lowering.plans.first().expect("lag case should lower to a native plan");
            let observed = collect_lag_lead_functions(plan);
            assert!(
                observed.iter().any(|function| matches!(
                    function,
                    nexus_query_engine::plan::WindowFunction::LagLead {
                        offset: 1,
                        default: None,
                        direction: nexus_query_engine::plan::ShiftDirection::Lag,
                        ..
                    }
                )),
                "window_lag should lower to LagLead with offset=1, default=NULL, direction=Lag; got {observed:#?}"
            );
        },
        "window_lag_offset_default" => {
            let plan = lowering.plans.first().expect("LAG(payload, 2, 0) case should lower to a native plan");
            let observed = collect_lag_lead_functions(plan);
            assert!(
                observed.iter().any(|function| matches!(
                    function,
                    nexus_query_engine::plan::WindowFunction::LagLead {
                        offset: 2,
                        default: Some(nexus_query_engine::expr::LiteralValue::Int64(0)),
                        direction: nexus_query_engine::plan::ShiftDirection::Lag,
                        ..
                    }
                )),
                "window_lag_offset_default should lower to LagLead with offset=2, default=Int64(0), direction=Lag; got {observed:#?}"
            );
        },
        "window_lead" => {
            let plan = lowering.plans.first().expect("lead case should lower to a native plan");
            let observed = collect_lag_lead_functions(plan);
            assert!(
                observed.iter().any(|function| matches!(
                    function,
                    nexus_query_engine::plan::WindowFunction::LagLead {
                        offset: 1,
                        default: None,
                        direction: nexus_query_engine::plan::ShiftDirection::Lead,
                        ..
                    }
                )),
                "window_lead should lower to LagLead with offset=1, default=NULL, direction=Lead; got {observed:#?}"
            );
        },
        "offset_without_limit" => {
            let plan = lowering.plans.first().expect("offset case should lower to a native plan");
            assert!(
                matches!(
                    &plan.root_node().kind,
                    nexus_query_engine::plan::PlanNode::Limit(limit) if limit.spec.skip == 2 && limit.spec.fetch.is_none()
                ),
                "offset_without_limit should lower to an offset-only native Limit"
            );
        },
        "literal_only_projection" | "values_table" => {
            let plan = lowering.plans.first().expect("literal source case should lower to a native plan");
            assert!(
                plan.nodes().iter().any(|node| {
                    matches!(
                        &node.kind,
                        nexus_query_engine::plan::PlanNode::Source(source)
                            if matches!(source.as_ref(), nexus_query_engine::source::SourceNode::LiteralBatch(_))
                    )
                }),
                "{} should lower to a native literal batch source",
                case.id
            );
        },
        _ => {},
    }
}

fn collect_lag_lead_functions(
    plan: &nexus_query_engine::plan::QueryPlan,
) -> Vec<nexus_query_engine::plan::WindowFunction> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            nexus_query_engine::plan::PlanNode::Window(window) => Some(window.spec.functions.clone()),
            _ => None,
        })
        .flatten()
        .filter(|function| matches!(function, nexus_query_engine::plan::WindowFunction::LagLead { .. }))
        .collect()
}

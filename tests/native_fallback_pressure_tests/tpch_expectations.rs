pub(super) fn expected_tpch_physical_root(query_id: &str) -> &'static str {
    match query_id {
        "q02" | "q03" | "q10" | "q18" | "q21" => "SortExec(TopK)",
        "q06" | "q14" | "q17" | "q19" => "ProjectionExec",
        "q01" | "q04" | "q05" | "q07" | "q08" | "q09" | "q11" | "q12" | "q13" | "q15" | "q16" | "q20" | "q22" => {
            "SortExec"
        },
        other => panic!("missing TPC-H physical root expectation for {other}"),
    }
}

pub(super) fn expected_tpch_native_candidate_shape(query_id: &str) -> &'static str {
    match query_id {
        "q01" => "SortExec(ProjectionExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))",
        "q02" => {
            "SortExec(fetch)(ProjectionExec(HashJoinExec(HashJoinExec(HashJoinExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),LocalParquetScan),LocalParquetScan),FilterExec(LocalParquetScan)),ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))))))))"
        },
        "q03" => {
            "SortExec(fetch)(ProjectionExec(AggregateExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),FilterExec(LocalParquetScan)),FilterExec(LocalParquetScan)))))"
        },
        "q04" => {
            "SortExec(ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),FilterExec(LocalParquetScan)))))"
        },
        "q05" => {
            "SortExec(ProjectionExec(AggregateExec(HashJoinExec(HashJoinExec(HashJoinExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),LocalParquetScan),LocalParquetScan),LocalParquetScan),FilterExec(LocalParquetScan)))))"
        },
        "q06" => "ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))",
        "q07" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(HashJoinExec(ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)))))),FilterExec(LocalParquetScan)),FilterExec(LocalParquetScan))))))"
        },
        "q08" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),HashJoinExec(ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),LocalParquetScan))))),LocalParquetScan)),LocalParquetScan)))))"
        },
        "q09" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(ProjectionExec(HashJoinExec(LocalParquetScan,ProjectionExec(HashJoinExec(LocalParquetScan,HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),LocalParquetScan))))),LocalParquetScan)))))"
        },
        "q10" => {
            "SortExec(fetch)(ProjectionExec(AggregateExec(HashJoinExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),FilterExec(LocalParquetScan)),LocalParquetScan))))"
        },
        "q12" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)))))"
        },
        "q14" => {
            "ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan))))"
        },
        "q13" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(AggregateExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)))))))"
        },
        "q16" => {
            "SortExec(ProjectionExec(AggregateExec(AggregateExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),FilterExec(LocalParquetScan))))))"
        },
        "q17" => {
            "ProjectionExec(AggregateExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),ProjectionExec(AggregateExec(LocalParquetScan)))))"
        },
        "q18" => {
            "SortExec(fetch)(ProjectionExec(AggregateExec(HashJoinExec(FilterExec(AggregateExec(LocalParquetScan)),HashJoinExec(HashJoinExec(LocalParquetScan,LocalParquetScan),LocalParquetScan)))))"
        },
        "q19" => {
            "ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),FilterExec(LocalParquetScan))))"
        },
        "q11" => {
            "SortExec(ProjectionExec(NestedLoopJoinExec(ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),ProjectionExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))),ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),ProjectionExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))))))"
        },
        "q15" => {
            "SortExec(HashJoinExec(ProjectionExec(HashJoinExec(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))),LocalParquetScan)),AggregateExec(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))))"
        },
        "q20" => {
            "SortExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),HashJoinExec(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))),HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan))))"
        },
        "q21" => {
            "SortExec(fetch)(ProjectionExec(AggregateExec(HashJoinExec(HashJoinExec(HashJoinExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),FilterExec(LocalParquetScan)),FilterExec(LocalParquetScan)),LocalParquetScan),FilterExec(LocalParquetScan)))))"
        },
        "q22" => {
            "SortExec(ProjectionExec(AggregateExec(ProjectionExec(NestedLoopJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),AggregateExec(FilterExec(LocalParquetScan)))))))"
        },
        other => panic!("missing TPC-H native candidate shape expectation for {other}"),
    }
}

pub(super) fn expected_tpch_native_candidate_shape_family(query_id: &str) -> &'static str {
    match query_id {
        "q01" => "sort_local_parquet_aggregate",
        "q02" => "sort_fetch_hash_join_input",
        "q03" | "q10" | "q18" | "q21" => "sort_fetch_hash_join_aggregate",
        "q04" | "q05" | "q07" | "q08" | "q09" | "q12" | "q13" | "q16" => "sort_hash_join_aggregate",
        "q14" | "q17" | "q19" => "projected_global_hash_join_aggregate",
        "q06" => "projected_global_local_parquet_aggregate",
        "q11" => "sort_mixed_hash_nested_loop_join_input",
        "q15" | "q20" => "sort_hash_join_input",
        "q22" => "sort_mixed_hash_nested_loop_join_aggregate",
        other => panic!("missing TPC-H native candidate shape family expectation for {other}"),
    }
}

pub(super) fn expected_tpch_datafusion_planning_rejection(query_id: &str) -> (&'static str, &'static str) {
    match query_id {
        "q02" | "q15" | "q18" | "q20" => ("native_bounded_join_unsupported", "operator_hashjoin"),
        _ => ("unsupported_operator", expected_tpch_operator_unsupported_category(query_id)),
    }
}

pub(super) fn expected_tpch_bounded_join_lower_level_blockers(query_id: &str) -> &'static str {
    match query_id {
        "q15" => "source_contract,aggregate_contract,final_sort_contract",
        "q17" => "source_contract,aggregate_contract",
        _ => "",
    }
}

pub(super) fn expected_tpch_bounded_join_blocked_reservations(query_id: &str) -> &'static str {
    match query_id {
        "q15" => "source_read,operator_output,final_sort",
        "q17" => "source_read,operator_output",
        _ => "",
    }
}

pub(super) fn expected_tpch_bounded_join_blocked_family(query_id: &str) -> Option<&'static str> {
    match query_id {
        "q15" => Some("keyed_hash_join_resident_subplan_pipeline"),
        "q17" => Some("mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline"),
        _ => None,
    }
}

fn expected_tpch_operator_unsupported_category(query_id: &str) -> &'static str {
    match query_id {
        "q02" | "q10" | "q18" => "operator_sort_topk",
        "q11" | "q22" => "operator_nestedloopjoin",
        "q04" | "q05" | "q08" | "q09" | "q12" | "q15" | "q16" | "q17" | "q20" => "operator_hashjoin",
        "q01" => "operator_shape",
        other => panic!("missing TPC-H operator unsupported category expectation for {other}"),
    }
}

pub(super) fn expected_tpch_physical_shape_marker(query_id: &str) -> &'static str {
    match query_id {
        "q01" => "AggregateExec(ProjectionExec(FilterExec(DataSourceExec)))",
        "q02" | "q03" | "q04" | "q05" | "q07" | "q08" | "q09" | "q10" | "q12" | "q13" | "q14" | "q15" | "q16"
        | "q17" | "q18" | "q19" | "q20" | "q21" => "HashJoinExec",
        "q06" => "ProjectionExec(AggregateExec(FilterExec(DataSourceExec)))",
        "q11" | "q22" => "NestedLoopJoinExec",
        other => panic!("missing TPC-H physical shape marker expectation for {other}"),
    }
}

pub(super) fn assert_tpch_join_diagnostics(
    query_id: &str,
    diagnostics: &[datafusion_cudf::NativeJoinDiagnosticSummary],
) {
    match query_id {
        "q02" | "q03" | "q04" | "q05" | "q07" | "q08" | "q09" | "q10" | "q12" | "q13" | "q14" | "q15" | "q16"
        | "q17" | "q18" | "q19" | "q20" | "q21" => {
            assert!(!diagnostics.is_empty(), "{query_id} should expose join diagnostics");
            let first = &diagnostics[0];
            assert_eq!(first.operator, "HashJoinExec");
            assert!(first.key_count > 0, "{query_id} should expose hash join key count");
            assert_ne!(first.key_shape, "", "{query_id} should expose hash join key shape");
            assert_eq!(first.build_side, "left");
            assert_eq!(first.probe_side, "right");
        },
        "q11" | "q22" => {
            assert!(!diagnostics.is_empty(), "{query_id} should expose join diagnostics");
            assert_eq!(diagnostics[0].operator, "NestedLoopJoinExec");
            assert_eq!(diagnostics[0].key_count, 0);
            assert_eq!(diagnostics[0].key_shape, "none");
        },
        "q01" | "q06" => assert!(diagnostics.is_empty(), "{query_id} should not expose join diagnostics"),
        other => panic!("missing TPC-H join diagnostics expectation for {other}"),
    }
}

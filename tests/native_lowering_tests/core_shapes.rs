use super::*;

#[test]
fn benchmark_shape_lowers_to_one_native_query_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.plans[0].node_count(), 6);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Rewritten { .. }));

    let plan = &result.plans[0];
    let PlanNode::Limit(limit) = &plan.root_node().kind else {
        panic!("native root should be a limit node");
    };
    assert_eq!(limit.spec.fetch, 2);
    assert!(matches!(plan.sink(), ResultSink::HostArrow { input, .. } if *input == plan.root_id()));
    let sort_node = plan.node(limit.input).expect("limit input should exist");
    let PlanNode::Sort(sort) = &sort_node.kind else {
        panic!("limit input should be a sort node");
    };
    assert_eq!(sort.spec.expressions.len(), 2);
    assert_eq!(sort_node.facts.output_ordering.len(), 2);
    let aggregate_node = plan.node(sort.input).expect("sort input should exist");
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        panic!("sort input should be an aggregate node");
    };
    assert_eq!(aggregate.spec.aggr_exprs.len(), 1);
    let projection_node = plan.node(aggregate.input).expect("aggregate input should exist");
    let PlanNode::Projection(projection) = &projection_node.kind else {
        panic!("aggregate input should be a projection node");
    };
    assert_eq!(projection.outputs.len(), 2);
    let filter_node = plan.node(projection.input).expect("projection input should exist");
    let PlanNode::Filter(filter) = &filter_node.kind else {
        panic!("projection input should be a filter node");
    };
    assert!(matches!(
        &filter.predicate,
        Expr::Binary { right, .. } if matches!(right.as_ref(), Expr::Literal(LiteralValue::Int32(10)))
    ));
    let source_node = plan.node(filter.input).expect("filter input should exist");
    let PlanNode::Source(source_node) = &source_node.kind else {
        panic!("filter input should be a local parquet source");
    };
    let SourceNode::LocalParquet(source) = source_node.as_ref() else {
        panic!("filter input should be a local parquet source");
    };
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].file_size_bytes, Some(fs::metadata(&path).unwrap().len()));
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    assert_eq!(source.read_column_file_indices, vec![0, 2, 3, 1]);
    assert_eq!(source.projected_output_columns, vec!["key".to_string(), "m00".to_string(), "m01".to_string()]);
    assert_eq!(
        source.projected_output_column_source_refs,
        vec![SourceColumnRef::File(0), SourceColumnRef::File(2), SourceColumnRef::File(3)]
    );
    assert_eq!(source.pure_filter_columns, vec!["payload".to_string()]);
    assert_eq!(source.pure_filter_column_source_refs, vec![SourceColumnRef::File(1)]);
    assert_eq!(source.diagnostics.read_column_count, 4);
    assert_eq!(source.diagnostics.projected_output_column_count, 3);
    assert_eq!(source.diagnostics.pure_filter_column_count, 1);
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "full_files");
    assert_eq!(source.diagnostics.partition_column_disposition.as_str(), "absent");
    assert_eq!(source.diagnostics.zero_column_count_disposition.as_str(), "not_required");
    assert_eq!(source.diagnostics.schema_mapping_disposition.as_str(), "direct_by_name");
    assert!(result.plans[0].to_string().contains("LocalParquetSource"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn multi_full_file_parquet_scan_lowers_to_one_native_source() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-multi-file");
    let first = dir.join("part-000.parquet");
    let second = dir.join("part-001.parquet");
    write_numeric_parquet_file(&first);
    write_numeric_parquet_file(&second);
    let plan = aggregate_sort_limit_plan(parquet_scan(&dir), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let plan = &result.plans[0];
    let source = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source_node) => match source_node.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                SourceNode::GpuResident(_) => None,
            },
            _ => None,
        })
        .expect("native plan should contain a local parquet source");
    assert_eq!(source.file_count(), 2);
    assert_eq!(
        source.diagnostics.known_bytes,
        Some(fs::metadata(&first).unwrap().len() + fs::metadata(&second).unwrap().len())
    );
    assert!(plan.to_string().contains("LocalParquetSource files=2"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn explicit_parquet_row_group_access_plan_lowers_to_native_row_groups() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-row-groups");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let access_plan = ParquetAccessPlan::new(vec![
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
    ]);
    let plan = aggregate_sort_limit_plan(parquet_scan_with_access_plan(&path, access_plan), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1, 3][..]));
    assert_eq!(source.selected_row_group_count(), 2);
    assert_eq!(source.diagnostics.file_count, 1);
    assert_eq!(source.diagnostics.ranged_file_count, 0);
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert_eq!(source.diagnostics.predicate_disposition.as_str(), "absent");
    assert_eq!(source.diagnostics.read_column_count, 4);
    assert_eq!(source.diagnostics.projected_output_column_count, 3);
    assert_eq!(source.diagnostics.pure_filter_column_count, 1);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert_eq!(result.source_diagnostics[0].ranged_file_count, 0);
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    assert!(result.plans[0].to_string().contains("selected_row_groups=2"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parquet_row_level_access_plan_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-row-selection");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let selection = RowSelection::from(vec![RowSelector::select(1), RowSelector::skip(1)]);
    let access_plan = ParquetAccessPlan::new(vec![RowGroupAccess::Selection(selection)]);
    let plan = aggregate_sort_limit_plan(parquet_scan_with_access_plan(&path, access_plan), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceRowSelection, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(1));
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "row_selection"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn modulo_projection_expression_lowers_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-unsupported-expression");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Modulo);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.plans.len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn modulo_with_non_integral_operands_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-modulo-float");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Modulo);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tpch_like_source_types_do_not_reject_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-tpch-types");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let scan = parquet_scan(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn order_insensitive_pushed_down_scan_limit_lowers_to_native_source() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-scan-property");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path).with_fetch(Some(1)).unwrap();
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(1));
    assert_eq!(source.diagnostics.scan_limit_disposition.as_str(), "represented");
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].scan_limit_disposition.as_str(), "represented");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn topk_sort_fetch_lowers_to_native_sort_plus_limit() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-topk");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_topk_plan(parquet_scan(&path), Operator::Plus, 2);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let PlanNode::Limit(limit) = &result.plans[0].root_node().kind else {
        panic!("native root should be a limit node");
    };
    assert_eq!(limit.spec.fetch, 2);
    let sort_node = result.plans[0].node(limit.input).expect("limit input should exist");
    assert!(matches!(sort_node.kind, PlanNode::Sort(_)));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn phase5_sort_without_fetch_over_aggregate_lowers_without_native_limit() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sort-no-fetch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
    assert!(matches!(&result.plans[0].root_node().kind, PlanNode::Sort(_)));
    assert!(!result.plans[0].nodes().iter().any(|node| matches!(&node.kind, PlanNode::Limit(_))));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_preserving_merge_is_rejected_with_stable_native_reason() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sort-preserving-merge");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sort = aggregate_topk_plan(parquet_scan(&path), Operator::Plus, 2);
    let sort = sort.as_any().downcast_ref::<SortExec>().expect("helper returns SortExec");
    let merge: Arc<dyn ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(sort.expr().clone(), Arc::clone(sort.input())));

    let result = lower_physical_plan_to_native(merge);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSortPreservingMergeUnsupported, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_preserving_merge_with_child_fetch_smaller_than_root_fetch_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sort-preserving-merge-fetch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sort = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);
    let sort = sort.as_any().downcast_ref::<SortExec>().expect("helper returns SortExec");
    let child_sort: Arc<dyn ExecutionPlan> =
        Arc::new(SortExec::new(sort.expr().clone(), Arc::clone(sort.input())).with_fetch(Some(1)));
    let merge: Arc<dyn ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(sort.expr().clone(), child_sort).with_fetch(Some(2)));

    let result = lower_physical_plan_to_native(merge);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSortPreservingMergeUnsupported, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grouped_aggregate_late_original_index_columns_bind_after_source_compaction() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-aggregate-compacted-bindings");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = grouped_late_column_multi_sum_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let plan = &result.plans[0];
    let synthetic_projection = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Projection(projection)
                if projection
                    .expressions()
                    .any(|expr| matches!(expr, Expr::Alias { alias, .. } if alias == "__cudf_aggregate_arg_0")) =>
            {
                Some(projection)
            },
            _ => None,
        })
        .expect("aggregate input should materialize synthetic aggregate arguments");
    assert!(
        synthetic_projection
            .expressions()
            .any(|expr| matches!(expr, Expr::Column(column) if column.name() == "payload")),
        "group expression should bind original payload, not compacted ordinal m00: {:?}",
        synthetic_projection.expressions().collect::<Vec<_>>()
    );
    assert!(
        synthetic_projection.expressions().any(|expr| {
            matches!(
                expr,
                Expr::Alias { expr, alias }
                    if alias == "__cudf_aggregate_arg_0"
                        && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "m00")
            )
        }),
        "first aggregate arg should bind original m00, not compacted ordinal m01: {:?}",
        synthetic_projection.expressions().collect::<Vec<_>>()
    );
    assert!(
        synthetic_projection.expressions().any(|expr| {
            matches!(
                expr,
                Expr::Alias { expr, alias }
                    if alias == "__cudf_aggregate_arg_1"
                        && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "m01")
            )
        }),
        "second aggregate arg should bind original m01: {:?}",
        synthetic_projection.expressions().collect::<Vec<_>>()
    );
    let aggregate_exprs = native_aggregate_exprs(plan);
    assert_eq!(aggregate_exprs.len(), 2);
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(column) if column.name() == "__cudf_aggregate_arg_0"));
    assert!(matches!(&aggregate_exprs[1].expr, Expr::Column(column) if column.name() == "__cudf_aggregate_arg_1"));

    fs::remove_dir_all(dir).unwrap();
}

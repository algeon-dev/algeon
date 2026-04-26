use arrow::{array::Int32Array, record_batch::RecordBatch};
use datafusion::datasource::memory::MemorySourceConfig;

use super::*;

#[test]
fn benchmark_shape_lowers_to_one_native_query_plan() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.plans[0].node_count(), 6);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Selected { .. }));

    let plan = &result.plans[0];
    let PlanNode::Limit(limit) = &plan.root_node().kind else {
        panic!("native root should be a limit node");
    };
    assert_eq!(limit.spec.fetch, Some(2));
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
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-multi-file");
    let first = dir.join("part-000.parquet");
    let second = dir.join("part-001.parquet");
    write_numeric_parquet_file(&first);
    write_numeric_parquet_file(&second);
    let plan = aggregate_sort_limit_plan(parquet_scan(&dir), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let plan = &result.plans[0];
    let source = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source_node) => match source_node.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                SourceNode::GpuResident(_) => None,
                SourceNode::LiteralBatch(_) => None,
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
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-row-groups");
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

    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1, 3][..]));
    assert_eq!(source.selected_row_group_count(), 2);
    assert_eq!(source.diagnostics.file_count, 1);
    assert_eq!(source.diagnostics.ranged_file_count, 0);
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert_eq!(source.diagnostics.predicate_disposition.as_str(), "native_post_scan_only");
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
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-row-selection");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let selection = RowSelection::from(vec![RowSelector::select(1), RowSelector::skip(1)]);
    let access_plan = ParquetAccessPlan::new(vec![RowGroupAccess::Selection(selection)]);
    let plan = aggregate_sort_limit_plan(parquet_scan_with_access_plan(&path, access_plan), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourceRowSelection, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(1));
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "row_selection"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn integer_modulo_projection_expression_is_rejected_during_native_planning() {
    // m00/m01 are Int64; integer modulo stays on CPU until checked libcudf
    // semantics exist.
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-integer-modulo-shape");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Modulo);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty(), "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn modulo_with_non_integral_operands_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-modulo-float");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Modulo);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tpch_like_source_types_do_not_reject_native_planning() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-tpch-types");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let scan = parquet_scan(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn order_insensitive_pushed_down_scan_limit_lowers_to_native_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-scan-property");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path).with_fetch(Some(1)).unwrap();
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(1));
    assert_eq!(source.diagnostics.scan_limit_disposition.as_str(), "represented");
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].scan_limit_disposition.as_str(), "represented");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn order_sensitive_single_file_group_scan_limit_lowers_to_native_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-order-sensitive-scan-limit");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_order_sensitive_limit(&path, 1);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(1));
    assert_eq!(source.diagnostics.scan_limit_disposition.as_str(), "represented");
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].scan_limit_disposition.as_str(), "represented");
    assert!(result.source_diagnostics[0].rejected_features.iter().all(|feature| feature.as_str() != "preserve_order"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parquet_sorting_columns_scan_limit_lowers_to_native_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sorting-columns-scan-limit");
    let path = dir.join("input.parquet");
    write_sorted_numeric_parquet_file_with_sorting_columns(&path);
    let scan = parquet_scan(&path).with_fetch(Some(1)).unwrap();
    assert!(scan.properties().output_ordering().is_some());

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(1));
    assert_eq!(source.diagnostics.scan_limit_disposition.as_str(), "represented");
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].scan_limit_disposition.as_str(), "represented");
    assert!(result.source_diagnostics[0].rejected_features.iter().all(|feature| feature.as_str() != "file_ordering"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn order_sensitive_multi_file_group_scan_limit_remains_rejected() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-order-sensitive-multi-group");
    write_numeric_parquet_file(&dir.join("part-000.parquet"));
    write_numeric_parquet_file(&dir.join("part-001.parquet"));
    let scan = parquet_scan_with_order_sensitive_limit_one_file_per_group(&dir, 1);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourcePreserveOrder, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].scan_limit_disposition.as_str(), "rejected");
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "preserve_order"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn preserve_order_multi_file_group_without_scan_limit_remains_rejected() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-preserve-order-multi-group");
    write_numeric_parquet_file(&dir.join("part-000.parquet"));
    write_numeric_parquet_file(&dir.join("part-001.parquet"));
    let scan = parquet_scan_with_preserve_order_one_file_per_group(&dir);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourcePreserveOrder, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].scan_limit_disposition.as_str(), "absent");
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "preserve_order"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn declared_output_ordering_single_file_group_scan_lowers_to_native_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-declared-output-ordering");
    let path = dir.join("input.parquet");
    write_sorted_numeric_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    assert!(scan.properties().output_ordering().is_some());
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].rejected_features.iter().all(|feature| feature.as_str() != "output_ordering"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parquet_sorting_columns_inferred_output_ordering_lowers_to_native_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-parquet-sorting-columns");
    let path = dir.join("input.parquet");
    write_sorted_numeric_parquet_file_with_sorting_columns(&path);
    let scan = parquet_scan(&path);
    assert!(scan.properties().output_ordering().is_some());

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].rejected_features.iter().all(|feature| feature.as_str() != "output_ordering"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn declared_output_ordering_single_file_group_multi_file_scan_lowers_to_native_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-declared-output-ordering-multi-file");
    write_sorted_numeric_parquet_file_with_keys(&dir.join("part-000.parquet"), &[1, 1, 2]);
    write_sorted_numeric_parquet_file_with_keys(&dir.join("part-001.parquet"), &[3, 3, 4]);
    let scan = parquet_scan_with_declared_output_ordering(&dir);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 2);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].rejected_features.iter().all(|feature| feature.as_str() != "output_ordering"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn declared_output_ordering_multi_file_group_scan_with_overlapping_keys_is_rejected() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-declared-output-ordering-multi-group");
    write_sorted_numeric_parquet_file_with_keys(&dir.join("part-000.parquet"), &[1, 3, 5]);
    write_sorted_numeric_parquet_file_with_keys(&dir.join("part-001.parquet"), &[2, 3, 4]);
    let scan = parquet_scan_with_declared_output_ordering_one_file_per_group(&dir);
    assert!(scan.properties().output_ordering().is_some());
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourceOutputOrdering, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "output_ordering"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn declared_output_ordering_claim_through_aggregate_is_reestablished_with_native_sort() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-ordered-aggregate-claim");
    let path = dir.join("input.parquet");
    write_sorted_numeric_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let plan = grouped_sum_aggregate_plan(scan);
    assert!(plan.properties().output_ordering().is_some());

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("ordered aggregate should lower with an explicit native sort");
    };
    assert!(
        candidate.shape.starts_with("SortExec(AggregateExec("),
        "appended declared-ordering sort should be visible in candidate shape: {}",
        candidate.shape
    );
    assert!(matches!(result.plans[0].root_node().kind, PlanNode::Sort(_)));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn declared_output_ordering_limit_above_aggregate_is_rejected_instead_of_sorted_after_limit() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-ordered-limit-aggregate");
    let path = dir.join("input.parquet");
    write_sorted_numeric_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let aggregate = grouped_sum_aggregate_plan(scan);
    assert!(aggregate.properties().output_ordering().is_some());
    let plan: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(aggregate, 0, Some(5)));
    assert!(plan.properties().output_ordering().is_some());

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, CapabilityReason::NativeOrderedOutputUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn ordered_window_over_order_breaking_aggregate_is_rejected_before_root_sort_repair() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-window-over-aggregate-order");
    let path = dir.join("input.parquet");
    write_sorted_numeric_parquet_file(&path);
    let scan = parquet_scan_with_declared_output_ordering(&path);
    let aggregate = grouped_sum_aggregate_plan(scan);
    assert!(aggregate.properties().output_ordering().is_some());
    let aggregate_schema = aggregate.schema();
    let window_expr: Arc<dyn WindowExpr> = Arc::new(StandardWindowExpr::new(
        create_udwf_window_expr(
            &row_number_udwf(),
            &[],
            aggregate_schema.as_ref(),
            "row_number() ORDER BY [key]".to_string(),
            false,
        )
        .unwrap(),
        &[],
        &[PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false))],
        Arc::new(WindowFrame::new(None)),
    ));
    let plan: Arc<dyn ExecutionPlan> =
        Arc::new(BoundedWindowAggExec::try_new(vec![window_expr], aggregate, InputOrderMode::Sorted, true).unwrap());

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, CapabilityReason::NativeOrderedOutputUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn declared_output_ordering_multi_partition_memory_source_is_rejected() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let first_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap();
    let second_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3, 4]))]).unwrap();
    let ordering =
        LexOrdering::new([PhysicalSortExpr::new(col("key", &schema).unwrap(), ArrowSortOptions::new(false, false))])
            .unwrap();
    let source = MemorySourceConfig::try_new(&[vec![first_batch], vec![second_batch]], Arc::clone(&schema), None)
        .unwrap()
        .try_with_sort_information(vec![ordering])
        .unwrap();
    let plan = DataSourceExec::from_data_source(source) as Arc<dyn ExecutionPlan>;
    assert!(plan.properties().output_ordering().is_some());

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeOrderedOutputUnsupported, .. }
    ));
}

#[test]
fn sort_preserving_merge_over_partition_preserving_memory_sort_lowers() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let first_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 3]))]).unwrap();
    let second_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![2, 4]))]).unwrap();
    let ordering =
        LexOrdering::new([PhysicalSortExpr::new(col("key", &schema).unwrap(), ArrowSortOptions::new(false, false))])
            .unwrap();
    let source = MemorySourceConfig::try_new(&[vec![first_batch], vec![second_batch]], Arc::clone(&schema), None)
        .unwrap()
        .try_with_sort_information(vec![ordering.clone()])
        .unwrap();
    let scan = DataSourceExec::from_data_source(source) as Arc<dyn ExecutionPlan>;
    let sorted: Arc<dyn ExecutionPlan> =
        Arc::new(SortExec::new(ordering.clone(), scan).with_preserve_partitioning(true));
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortPreservingMergeExec::new(ordering, sorted));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("partition-preserving SPM should lower to native");
    };
    assert_eq!(candidate.shape, "SortPreservingMergeExec(SortExec(MemorySource))");
    assert!(matches!(result.plans[0].root_node().kind, PlanNode::Sort(_)));
}

#[test]
fn sort_preserving_merge_over_ordered_memory_source_lowers_without_child_sort() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let first_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 3]))]).unwrap();
    let second_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![2, 4]))]).unwrap();
    let ordering =
        LexOrdering::new([PhysicalSortExpr::new(col("key", &schema).unwrap(), ArrowSortOptions::new(false, false))])
            .unwrap();
    let source = MemorySourceConfig::try_new(&[vec![first_batch], vec![second_batch]], Arc::clone(&schema), None)
        .unwrap()
        .try_with_sort_information(vec![ordering.clone()])
        .unwrap();
    let scan = DataSourceExec::from_data_source(source) as Arc<dyn ExecutionPlan>;
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortPreservingMergeExec::new(ordering, scan));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("SPM over an ordered input should lower to native");
    };
    assert_eq!(candidate.shape, "SortPreservingMergeExec(MemorySource)");
    assert!(matches!(result.plans[0].root_node().kind, PlanNode::Sort(_)));
}

#[test]
fn partition_preserving_sort_with_single_input_partition_lowers() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![2, 1]))]).unwrap();
    let ordering =
        LexOrdering::new([PhysicalSortExpr::new(col("key", &schema).unwrap(), ArrowSortOptions::new(false, false))])
            .unwrap();
    let source = MemorySourceConfig::try_new(&[vec![batch]], Arc::clone(&schema), None).unwrap();
    let scan = DataSourceExec::from_data_source(source) as Arc<dyn ExecutionPlan>;
    assert_eq!(scan.properties().output_partitioning().partition_count(), 1);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(ordering, scan).with_preserve_partitioning(true));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("single-partition partition-preserving sort should lower to native");
    };
    assert_eq!(candidate.shape, "SortExec(MemorySource)");
    assert!(matches!(result.plans[0].root_node().kind, PlanNode::Sort(_)));
}

#[test]
fn partition_preserving_sort_with_multiple_input_partitions_is_rejected() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let first_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 3]))]).unwrap();
    let second_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![2, 4]))]).unwrap();
    let ordering =
        LexOrdering::new([PhysicalSortExpr::new(col("key", &schema).unwrap(), ArrowSortOptions::new(false, false))])
            .unwrap();
    let source =
        MemorySourceConfig::try_new(&[vec![first_batch], vec![second_batch]], Arc::clone(&schema), None).unwrap();
    let scan = DataSourceExec::from_data_source(source) as Arc<dyn ExecutionPlan>;
    assert_eq!(scan.properties().output_partitioning().partition_count(), 2);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(ordering, scan).with_preserve_partitioning(true));

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported {
            capability_reason: CapabilityReason::NativePartitionPreservingSortUnsupported,
            ..
        }
    ));
}

#[test]
fn topk_sort_fetch_lowers_to_native_sort_plus_limit() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-topk");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_topk_plan(parquet_scan(&path), Operator::Plus, 2);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let PlanNode::Limit(limit) = &result.plans[0].root_node().kind else {
        panic!("native root should be a limit node");
    };
    assert_eq!(limit.spec.fetch, Some(2));
    let sort_node = result.plans[0].node(limit.input).expect("limit input should exist");
    assert!(matches!(sort_node.kind, PlanNode::Sort(_)));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn input_tree_sort_fetch_lowers_to_native_sort_plus_limit() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-input-topk");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = count_over_topk_sort_fetch_input_plan(parquet_scan(&path), 2);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("input-tree top-k aggregate should lower to native");
    };
    assert_eq!(candidate.shape, "AggregateExec(SortExec(fetch)(LocalParquetScan))");
    let PlanNode::Aggregate(_aggregate) = &result.plans[0].root_node().kind else {
        panic!("native root should be an aggregate node");
    };
    let has_sort_fetch_limit = result.plans[0].nodes().iter().any(|node| {
        let PlanNode::Limit(limit) = &node.kind else {
            return false;
        };
        if limit.spec.fetch != Some(2) {
            return false;
        }
        let Some(input) = result.plans[0].node(limit.input) else {
            return false;
        };
        match &input.kind {
            PlanNode::Sort(_) => true,
            PlanNode::Projection(projection) => {
                result.plans[0].node(projection.input).is_some_and(|input| matches!(input.kind, PlanNode::Sort(_)))
            },
            _ => false,
        }
    });
    assert!(has_sort_fetch_limit, "native plan should keep SortExec.fetch as a limit above sort:\n{}", result.plans[0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_without_fetch_over_aggregate_lowers_without_native_limit() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sort-no-fetch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert!(matches!(&result.plans[0].root_node().kind, PlanNode::Sort(_)));
    assert!(!result.plans[0].nodes().iter().any(|node| matches!(&node.kind, PlanNode::Limit(_))));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn aggregate_root_global_limit_does_not_push_scan_limit_to_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-aggregate-root-limit-no-source-pushdown");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sorted = aggregate_sort_plan_without_filter(parquet_scan(&path), Operator::Plus);
    let sort = sorted.downcast_ref::<SortExec>().expect("helper returns SortExec");
    let aggregate = Arc::clone(sort.input());
    let plan: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(aggregate, 0, Some(2)));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, None, "LIMIT after aggregate must not be applied before aggregate input");
    assert!(matches!(result.plans[0].root_node().kind, PlanNode::Limit(_)));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn nested_limit_offsets_push_enough_rows_to_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-nested-limit-source-pushdown");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let inner: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(scan, 50, Some(100)));
    let outer: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(inner, 100, Some(10)));

    let result = lower_physical_plan_to_native(outer);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.scan_limit,
        Some(150),
        "source must keep the inner offset plus the rows needed by the outer limit"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn zero_fetch_global_limit_pushes_zero_scan_limit_to_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-zero-limit-source-pushdown");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(parquet_scan(&path), 100, Some(0)));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, Some(0), "LIMIT 0 can be answered without reading source rows");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_preserving_merge_without_fetch_lowers_to_native_sort() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sort-preserving-merge");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sort_plan = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);
    let sort = sort_plan.downcast_ref::<SortExec>().expect("helper returns SortExec");
    let merge: Arc<dyn ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(sort.expr().clone(), Arc::clone(&sort_plan)));

    let result = lower_physical_plan_to_native(merge);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("non-fetch SPM should lower to native");
    };
    assert!(
        candidate.shape.starts_with("SortPreservingMergeExec(SortExec(AggregateExec("),
        "unexpected candidate shape: {}",
        candidate.shape
    );
    assert!(matches!(result.plans[0].root_node().kind, PlanNode::Sort(_)));
    assert!(!result.plans[0].nodes().iter().any(|node| matches!(node.kind, PlanNode::Limit(_))));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_preserving_merge_fetch_with_matching_child_sort_lowers_to_native_sort_plus_limit() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sort-preserving-merge-fetch-matching-child-sort");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sort_plan = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);
    let sort = sort_plan.downcast_ref::<SortExec>().expect("helper returns SortExec");
    let merge: Arc<dyn ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(sort.expr().clone(), Arc::clone(&sort_plan)).with_fetch(Some(2)));

    let result = lower_physical_plan_to_native(merge);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let PlanNode::Limit(limit) = &result.plans[0].root_node().kind else {
        panic!("SPM fetch should lower to a native limit above sort:\n{}", result.plans[0]);
    };
    assert_eq!(limit.spec.fetch, Some(2));
    let sort_node = result.plans[0].node(limit.input).expect("limit input should exist");
    assert!(matches!(sort_node.kind, PlanNode::Sort(_)));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_preserving_merge_fetch_without_child_ordering_is_rejected() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sort-preserving-merge-fetch-no-child-ordering");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sort = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);
    let sort = sort.downcast_ref::<SortExec>().expect("helper returns SortExec");
    let merge: Arc<dyn ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(sort.expr().clone(), Arc::clone(sort.input())).with_fetch(Some(2)));

    let result = lower_physical_plan_to_native(merge);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported {
            capability_reason: CapabilityReason::NativeSortPreservingMergeUnsupported,
            ..
        }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_preserving_merge_fetch_with_mismatched_child_sort_is_rejected() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sort-preserving-merge-fetch-mismatched-child-sort");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sort = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);
    let sort = sort.downcast_ref::<SortExec>().expect("helper returns SortExec");
    let first_sort_expr = &sort.expr()[0];
    let child_ordering = LexOrdering::new([PhysicalSortExpr::new(
        Arc::clone(&first_sort_expr.expr),
        ArrowSortOptions::new(!first_sort_expr.options.descending, first_sort_expr.options.nulls_first),
    )])
    .expect("mismatched child ordering should be non-empty");
    let child_sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(child_ordering, Arc::clone(sort.input())));
    let merge: Arc<dyn ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(sort.expr().clone(), child_sort).with_fetch(Some(2)));

    let result = lower_physical_plan_to_native(merge);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported {
            capability_reason: CapabilityReason::NativeSortPreservingMergeUnsupported,
            ..
        }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_preserving_merge_with_same_name_different_index_ordering_is_rejected() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sort-preserving-merge-same-name-different-index");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sort_plan = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);
    let sort = sort_plan.downcast_ref::<SortExec>().expect("helper returns SortExec");
    let merge_ordering = LexOrdering::new([PhysicalSortExpr::new(
        Arc::new(datafusion::physical_expr::expressions::Column::new("sum_x00", 0)),
        sort.expr()[0].options,
    )])
    .expect("SPM ordering should be non-empty");
    let merge: Arc<dyn ExecutionPlan> = Arc::new(SortPreservingMergeExec::new(merge_ordering, sort_plan));

    let result = lower_physical_plan_to_native(merge);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported {
            capability_reason: CapabilityReason::NativeSortPreservingMergeUnsupported,
            ..
        }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sort_preserving_merge_with_child_fetch_smaller_than_root_fetch_is_rejected() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sort-preserving-merge-fetch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let sort = aggregate_sort_plan(parquet_scan(&path), Operator::Plus);
    let sort = sort.downcast_ref::<SortExec>().expect("helper returns SortExec");
    let child_sort: Arc<dyn ExecutionPlan> =
        Arc::new(SortExec::new(sort.expr().clone(), Arc::clone(sort.input())).with_fetch(Some(1)));
    let merge: Arc<dyn ExecutionPlan> =
        Arc::new(SortPreservingMergeExec::new(sort.expr().clone(), child_sort).with_fetch(Some(2)));

    let result = lower_physical_plan_to_native(merge);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported {
            capability_reason: CapabilityReason::NativeSortPreservingMergeUnsupported,
            ..
        }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grouped_aggregate_late_original_index_columns_bind_after_source_compaction() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-aggregate-compacted-bindings");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = grouped_late_column_multi_sum_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let plan = &result.plans[0];
    let synthetic_projection = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Projection(projection)
                if projection.expressions().any(
                    |expr| matches!(expr, Expr::Alias { alias, .. } if alias == "__nexus_native_aggregate_arg_0"),
                ) =>
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
                    if alias == "__nexus_native_aggregate_arg_0"
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
                    if alias == "__nexus_native_aggregate_arg_1"
                        && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "m01")
            )
        }),
        "second aggregate arg should bind original m01: {:?}",
        synthetic_projection.expressions().collect::<Vec<_>>()
    );
    let aggregate_exprs = native_aggregate_exprs(plan);
    assert_eq!(aggregate_exprs.len(), 2);
    assert!(
        matches!(&aggregate_exprs[0].expr, Expr::Column(column) if column.name() == "__nexus_native_aggregate_arg_0")
    );
    assert!(
        matches!(&aggregate_exprs[1].expr, Expr::Column(column) if column.name() == "__nexus_native_aggregate_arg_1")
    );

    fs::remove_dir_all(dir).unwrap();
}

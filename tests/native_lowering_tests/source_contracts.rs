use super::*;

#[test]
fn pushed_down_parquet_predicate_with_filter_is_reader_pushdown() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_some());
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn reader_pushdown_covered_filter_keeps_frontend_projection_order_after_reordered_source_read() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-source-order");
    let path = dir.join("input.parquet");
    write_snap_edge_parquet_file(&path);
    let scan = parquet_edge_scan_with_projection_and_pushed_src_predicate(&path);
    let scan_schema = scan.schema();
    let predicate =
        binary(col("src", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int64(Some(10))), &scan_schema)
            .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["dst".to_string(), "src".to_string(), "weight".to_string()]);
    assert_eq!(source.projected_output_columns, vec!["src".to_string(), "dst".to_string(), "weight".to_string()]);
    assert!(source.reader_predicate.is_some());
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    let PlanNode::Projection(projection) = &result.plans[0].root_node().kind else {
        panic!("reader-pushdown-covered source must project back to frontend scan order");
    };
    let projected_columns = projection
        .expressions()
        .map(|expr| match expr {
            Expr::Column(column) => column.name().to_string(),
            other => panic!("expected column projection, got {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(projected_columns, vec!["src".to_string(), "dst".to_string(), "weight".to_string()]);
    let projected_fields = projection.output_fields().map(|field| field.name().as_str()).collect::<Vec<_>>();
    assert_eq!(projected_fields, vec!["src", "dst", "weight"]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_parquet_predicate_subset_of_filter_is_reader_pushdown_with_native_verification() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate-subset");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate(&path);
    let scan_schema = scan.schema();
    let pushed =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let retained =
        binary(col("key", &scan_schema).unwrap(), Operator::NotEq, lit(ScalarValue::Int32(Some(2))), &scan_schema)
            .unwrap();
    let filter_predicate = binary(pushed, Operator::And, retained, &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, filter_predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_some());
    assert_eq!(native_filter_exprs(&result.plans[0]).len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_parquet_predicate_uses_filter_input_schema_for_validation() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate-filter-schema");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let scan = parquet_scan_with_projection_and_pushed_label_predicate(&path);
    let scan_schema = scan.schema();
    let predicate = binary(
        col("label", &scan_schema).unwrap(),
        Operator::NotEq,
        lit(ScalarValue::Utf8View(Some("z".to_string()))),
        &scan_schema,
    )
    .unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_some());
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_substring_prefix_predicate_is_rewritten_to_reader_pushdown_ranges() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-prefix-predicate");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let scan = parquet_scan_with_pushed_down_label_prefix_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    let reader_predicate = source.reader_predicate.as_ref().expect("prefix predicate should be pushed to the reader");
    let reader_predicate_text = reader_predicate.to_string();
    assert!(!reader_predicate_text.contains("SUBSTRING"), "{reader_predicate_text}");
    assert!(reader_predicate_text.contains("label >= \"a\""), "{reader_predicate_text}");
    assert!(reader_predicate_text.contains("label < \"b\""), "{reader_predicate_text}");
    assert!(reader_predicate_text.contains("label >= \"b\""), "{reader_predicate_text}");
    assert!(reader_predicate_text.contains("label < \"c\""), "{reader_predicate_text}");
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_parquet_predicate_without_filter_is_reader_pushdown() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate-without-filter");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_some());
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_parquet_predicate_with_source_limit_lowers_limit_after_reader_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate-source-limit");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_source_limit(parquet_scan_with_pushed_down_predicate(&path), 3);

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, None, "pushed predicates must be applied before the scan limit");
    assert!(source.reader_predicate.is_some());
    let limits = result.plans[0]
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Limit(limit) => Some(limit),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(limits.len(), 1, "native plan should apply the scan limit after the predicate:\n{}", result.plans[0]);
    assert_eq!(limits[0].spec.fetch, Some(3));
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_post_scan_predicate_with_source_limit_lowers_filter_then_limit() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-post-scan-predicate-source-limit");
    let path = dir.join("input.parquet");
    write_household_demographics_like_parquet_file(&path);
    let scan = parquet_scan_with_source_limit(parquet_scan_with_pushed_down_case_decimal_divide_predicate(&path), 3);

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "native_post_scan_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.scan_limit, None, "pushed predicates must be applied before the scan limit");
    assert!(source.reader_predicate.is_none());
    let post_scan_filters = native_filter_exprs(&result.plans[0]);
    assert_eq!(post_scan_filters.len(), 1);
    let limit_node = result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Limit(limit) if limit.spec.fetch == Some(3) => Some(limit),
            _ => None,
        })
        .expect("native plan should apply scan limit after post-scan filter");
    let limit_input = result.plans[0].node(limit_node.input).expect("limit input should exist");
    let limit_is_after_filter = match &limit_input.kind {
        PlanNode::Filter(_) => true,
        PlanNode::Projection(projection) => {
            result.plans[0].node(projection.input).is_some_and(|node| matches!(node.kind, PlanNode::Filter(_)))
        },
        _ => false,
    };
    assert!(limit_is_after_filter, "limit should be above the predicate filter:\n{}", result.plans[0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_local_parquet_scan_lowers_to_empty_literal_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-empty-local-scan");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_without_files(parquet_scan_with_projection_indices(&path, vec![0, 2]));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(result.source_diagnostics.is_empty());
    assert!(native_local_parquet_sources(&result.plans[0]).is_empty());
    let literal_sources = native_literal_batch_sources(&result.plans[0]);
    assert_eq!(literal_sources.len(), 1);
    assert_eq!(literal_sources[0].row_count(), 0);
    let field_names = literal_sources[0].schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    assert_eq!(field_names, vec!["key", "m00"]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_s3_parquet_scan_lowers_to_empty_literal_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-empty-s3-scan");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_without_files(parquet_scan_with_object_store_url_and_location(
        &path,
        ObjectStoreUrl::parse("s3://bucket").unwrap(),
        "table/input.parquet",
    ));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(result.source_diagnostics.is_empty());
    assert!(native_local_parquet_sources(&result.plans[0]).is_empty());
    let literal_sources = native_literal_batch_sources(&result.plans[0]);
    assert_eq!(literal_sources.len(), 1);
    assert_eq!(literal_sources[0].row_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_local_parquet_scan_with_filter_projection_lowers_to_projected_empty_literal_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-empty-local-filter-projection");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_without_files(parquet_scan(&path));
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        datafusion::physical_plan::filter::FilterExecBuilder::new(predicate, scan)
            .apply_projection(Some(vec![0, 2]))
            .unwrap()
            .build()
            .unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(result.source_diagnostics.is_empty());
    assert!(native_filter_exprs(&result.plans[0]).is_empty());
    assert!(native_local_parquet_sources(&result.plans[0]).is_empty());
    let literal_sources = native_literal_batch_sources(&result.plans[0]);
    assert_eq!(literal_sources.len(), 1);
    assert_eq!(literal_sources[0].row_count(), 0);
    let field_names = literal_sources[0].schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    assert_eq!(field_names, vec!["key", "m00"]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_local_parquet_scan_with_source_limit_lowers_to_empty_literal_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-empty-local-source-limit");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_without_files(parquet_scan_with_source_limit(parquet_scan(&path), 3));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(result.source_diagnostics.is_empty());
    assert!(native_local_parquet_sources(&result.plans[0]).is_empty());
    assert!(result.plans[0].nodes().iter().all(|node| !matches!(node.kind, PlanNode::Limit(_))));
    let literal_sources = native_literal_batch_sources(&result.plans[0]);
    assert_eq!(literal_sources.len(), 1);
    assert_eq!(literal_sources[0].row_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_local_parquet_scan_with_pushed_predicate_and_source_limit_lowers_to_empty_literal_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-empty-local-predicate-source-limit");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan =
        parquet_scan_without_files(parquet_scan_with_source_limit(parquet_scan_with_pushed_down_predicate(&path), 3));

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(result.source_diagnostics.is_empty());
    assert!(native_local_parquet_sources(&result.plans[0]).is_empty());
    assert!(native_filter_exprs(&result.plans[0]).is_empty());
    assert!(result.plans[0].nodes().iter().all(|node| !matches!(node.kind, PlanNode::Limit(_))));
    let literal_sources = native_literal_batch_sources(&result.plans[0]);
    assert_eq!(literal_sources.len(), 1);
    assert_eq!(literal_sources[0].row_count(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_local_parquet_aggregate_lowers_through_literal_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-empty-local-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = count_only_sort_limit_plan(parquet_scan_without_files(parquet_scan(&path)));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(result.source_diagnostics.is_empty());
    assert!(native_local_parquet_sources(&result.plans[0]).is_empty());
    let literal_sources = native_literal_batch_sources(&result.plans[0]);
    assert_eq!(literal_sources.len(), 1);
    assert_eq!(literal_sources[0].row_count(), 0);
    assert_eq!(native_aggregate_functions(&result.plans[0]), vec![AggregateFunction::Count]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_scan_only_decimal_predicate_uses_reader_pushdown() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-decimal-post-scan");
    let path = dir.join("input.parquet");
    write_decimal_filter_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_decimal_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_some());
    assert_eq!(source.pure_filter_columns, vec!["payload"]);
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_decimal_equality_predicate_uses_reader_pushdown() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-decimal-equality-reader");
    let path = dir.join("input.parquet");
    write_decimal_filter_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_decimal_equality_predicate(&path);
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::Eq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_some());
    assert_eq!(source.pure_filter_columns, vec!["payload"]);
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_decimal_in_list_predicate_uses_reader_pushdown() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-decimal-in-list-reader");
    let path = dir.join("input.parquet");
    write_decimal_filter_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_decimal_in_list_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "reader_pushdown");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_some());
    assert_eq!(source.pure_filter_columns, vec!["payload"]);
    assert!(native_filter_exprs(&result.plans[0]).is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_case_decimal_divide_predicate_uses_native_post_scan_filter() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-case-decimal-divide");
    let path = dir.join("input.parquet");
    write_household_demographics_like_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_case_decimal_divide_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "native_post_scan_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_none());
    assert_eq!(native_filter_exprs(&result.plans[0]).len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_integer_divide_predicate_is_rejected_during_native_planning() {
    // Integer divide in scan-only predicates falls back to CPU; decimal divide
    // still lowers above.
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-integer-divide");
    let path = dir.join("input.parquet");
    write_household_demographics_like_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_integer_divide_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

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
fn pushed_down_parquet_predicate_with_different_filter_uses_native_post_scan_filter() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate-mismatch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate_value(&path, 20);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Selected { .. }));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "native_post_scan_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_none());
    assert_eq!(native_filter_exprs(&result.plans[0]).len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn full_file_byte_range_coalesces_to_native_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-ranged-file");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_byte_range(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].file_size_bytes, Some(fs::metadata(&path).unwrap().len()));
    let expected_stable_id_prefix = format!("local:{}:{}:mtime=", path.display(), fs::metadata(&path).unwrap().len());
    assert!(
        source.files[0].stable_id.starts_with(&expected_stable_id_prefix),
        "stable id should include local path, size, and mtime: {}",
        source.files[0].stable_id
    );
    assert!(source.diagnostics.direct_device_read);
    assert_eq!(source.diagnostics.ranged_file_count, 1);
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "coalesced_full_files");
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].ranged_file_count, 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "coalesced_full_files");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn partial_byte_range_maps_to_row_group_selection() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partial-ranged-file");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let row_group_offsets = row_group_range_pruning_offsets(&path);
    assert!(row_group_offsets.len() >= 2, "fixture must have at least two row groups");
    let scan = parquet_scan_with_explicit_byte_ranges(&path, vec![(row_group_offsets[0], row_group_offsets[1])]);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[0][..]));
    assert_eq!(source.diagnostics.ranged_file_count, 1);
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(source.diagnostics.selected_row_groups, Some(1));
    assert!(source.diagnostics.direct_device_read);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].ranged_file_count, 1);
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(1));
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn partial_byte_range_owns_row_groups_by_start_offset_boundaries() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-row-group-byte-range");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let row_group_offsets = row_group_range_pruning_offsets(&path);
    assert!(row_group_offsets.len() >= 4, "fixture must have at least four row groups");
    let scan = parquet_scan_with_explicit_byte_ranges(&path, vec![(row_group_offsets[1], row_group_offsets[3])]);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1, 2][..]));
    assert_eq!(source.diagnostics.ranged_file_count, 1);
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert!(source.diagnostics.direct_device_read);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn partial_byte_range_with_access_plan_lowers_to_selected_row_groups() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-ranged-access-plan");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let row_group_offsets = row_group_range_pruning_offsets(&path);
    assert!(row_group_offsets.len() >= 4, "fixture must have at least four row groups");
    let access_plan = ParquetAccessPlan::new(vec![
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
    ]);
    let scan = parquet_scan_with_explicit_byte_ranges_and_access_plan(
        &path,
        vec![(row_group_offsets[1], row_group_offsets[3])],
        access_plan,
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1][..]));
    assert_eq!(source.diagnostics.ranged_file_count, 1);
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(source.diagnostics.selected_row_groups, Some(1));
    assert!(source.diagnostics.direct_device_read);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(1));
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn partial_byte_range_ignores_row_level_selection_outside_range() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-ranged-selection-outside");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let row_group_offsets = row_group_range_pruning_offsets(&path);
    assert!(row_group_offsets.len() >= 4, "fixture must have at least four row groups");
    let row_selection = RowSelection::from(vec![RowSelector::select(1), RowSelector::skip(1)]);
    let access_plan = ParquetAccessPlan::new(vec![
        RowGroupAccess::Selection(row_selection),
        RowGroupAccess::Scan,
        RowGroupAccess::Skip,
        RowGroupAccess::Skip,
    ]);
    let scan = parquet_scan_with_explicit_byte_ranges_and_access_plan(
        &path,
        vec![(row_group_offsets[1], row_group_offsets[2])],
        access_plan,
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1][..]));
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(source.diagnostics.selected_row_groups, Some(1));
    assert!(source.diagnostics.rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn partial_byte_range_rejects_access_plan_length_mismatch() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-ranged-access-plan-len");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let row_group_offsets = row_group_range_pruning_offsets(&path);
    assert!(row_group_offsets.len() >= 4, "fixture must have at least four row groups");
    let access_plan = ParquetAccessPlan::new(vec![RowGroupAccess::Scan]);
    let scan = parquet_scan_with_explicit_byte_ranges_and_access_plan(
        &path,
        vec![(row_group_offsets[2], row_group_offsets[3])],
        access_plan,
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourceByteRange, .. }
    ));
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "rejected");
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "byte_range"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn mixed_partial_byte_range_and_full_file_reports_selected_row_group_total() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-mixed-ranged-files");
    let first = dir.join("part-000.parquet");
    let second = dir.join("part-001.parquet");
    write_row_group_pruning_parquet_file(&first);
    write_row_group_pruning_parquet_file(&second);
    let row_group_offsets = row_group_range_pruning_offsets(&first);
    assert!(row_group_offsets.len() >= 2, "fixture must have at least two row groups");
    let scan = parquet_scan_with_named_file_byte_ranges(
        &dir,
        "part-000.parquet",
        vec![(row_group_offsets[0], row_group_offsets[1])],
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 2);
    assert_eq!(source.diagnostics.ranged_file_count, 1);
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(source.diagnostics.selected_row_groups, Some(5));
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(5));
    assert!(source.diagnostics.rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn partial_byte_range_with_access_plan_and_extra_extension_is_rejected() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-ranged-mixed-extension");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let access_plan = ParquetAccessPlan::new(vec![
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
    ]);
    let scan = parquet_scan_with_partial_byte_range_access_plan_and_extra_extension(&path, access_plan);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourceByteRange, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "rejected");
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "byte_range"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn split_full_file_byte_ranges_coalesce_without_duplicate_native_files() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-split-ranged-file");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_split_full_byte_ranges(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].file_size_bytes, Some(fs::metadata(&path).unwrap().len()));
    assert_eq!(source.diagnostics.file_count, 1);
    assert_eq!(source.diagnostics.ranged_file_count, 2);
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "coalesced_full_files");
    assert!(source.diagnostics.direct_device_read);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert_eq!(result.source_diagnostics[0].ranged_file_count, 2);
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn overlapping_byte_ranges_are_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-overlap-ranged-file");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_overlapping_byte_ranges(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourceByteRange, .. }
    ));
    assert_eq!(result.source_diagnostics[0].ranged_file_count, 2);
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "rejected");
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "byte_range"));

    fs::remove_dir_all(dir).unwrap();
}

fn assert_s3_remote_native_source(source: &nexus_query_engine::source::LocalParquetSource, expected_uri: &str) {
    assert_eq!(source.file_count(), 1);
    let file = &source.files[0];
    assert_eq!(file.location.read_capability(), "object_store:s3");
    assert_eq!(file.resolved_source.original_location, expected_uri);
    assert_eq!(file.resolved_source.reader_location(), expected_uri);
    assert!(file.stable_id.starts_with("object_store:s3:fnv64:"));
    assert!(!file.stable_id.contains("bucket"));
    assert!(!file.stable_id.contains("table/input.parquet"));
    assert_eq!(source.diagnostics.source_read_strategy, file.resolved_source.read_strategy().as_str());
    assert_eq!(source.diagnostics.credential_mode, file.resolved_source.credential_mode().as_str());
    assert!(!source.diagnostics.direct_device_read);
    assert!(source.diagnostics.rejected_features.iter().all(|feature| feature.as_str() != "non_local_object_store"));
}

#[test]
fn s3_parquet_object_store_scan_lowers_to_remote_native_source() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-s3-store");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_object_store_url_and_location(
        &path,
        ObjectStoreUrl::parse("s3://bucket").unwrap(),
        "table/input.parquet",
    );
    let expected_uri = "s3://bucket/table/input.parquet";

    let result = lower_physical_plan_to_native(scan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    assert!(!result.source_diagnostics[0].direct_device_read);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_s3_remote_native_source(source, expected_uri);
    assert_eq!(result.source_diagnostics[0].source_read_strategy, source.diagnostics.source_read_strategy);
    assert_eq!(result.source_diagnostics[0].credential_mode, source.diagnostics.credential_mode);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn s3_multi_file_parquet_object_store_scan_rejects_unverified_footer_schema() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-s3-store-multi-file");
    let first_path = dir.join("part-0.parquet");
    let second_path = dir.join("part-1.parquet");
    write_numeric_parquet_file(&first_path);
    write_numeric_parquet_file(&second_path);
    let scan = parquet_scan_with_object_store_url(&dir, ObjectStoreUrl::parse("s3://bucket").unwrap());

    let result = lower_physical_plan_to_native(scan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourceSchemaMapping, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn s3_parquet_object_store_aggregate_root_preserves_remote_source_diagnostics() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-s3-store-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_object_store_url_and_location(
        &path,
        ObjectStoreUrl::parse("s3://bucket").unwrap(),
        "table/input.parquet",
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);
    let expected_uri = "s3://bucket/table/input.parquet";

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_s3_remote_native_source(source, expected_uri);
    assert_eq!(result.source_diagnostics[0].source_read_strategy, source.diagnostics.source_read_strategy);
    assert_eq!(result.source_diagnostics[0].credential_mode, source.diagnostics.credential_mode);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn non_s3_parquet_object_store_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-remote-store");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_object_store_url(&path, ObjectStoreUrl::parse("test:///").unwrap());
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourceNonLocalObjectStore, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert!(
        result.source_diagnostics[0]
            .rejected_features
            .iter()
            .any(|feature| feature.as_str() == "non_local_object_store")
    );
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "absent");

    fs::remove_dir_all(dir).unwrap();
}

fn row_group_range_pruning_offsets(path: &std::path::Path) -> Vec<i64> {
    let reader = SerializedFileReader::try_from(path).unwrap();
    reader
        .metadata()
        .row_groups()
        .iter()
        .map(|row_group| {
            let column = row_group.column(0);
            column.dictionary_page_offset().unwrap_or_else(|| column.data_page_offset())
        })
        .collect()
}

#[test]
fn custom_parquet_file_reader_factory_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-custom-reader-factory");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_custom_reader_factory(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported {
            capability_reason: CapabilityReason::NativeSourceParquetFileReaderFactory,
            ..
        }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(!result.source_diagnostics[0].direct_device_read);
    assert!(
        result.source_diagnostics[0]
            .rejected_features
            .iter()
            .any(|feature| feature.as_str() == "parquet_file_reader_factory")
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parquet_partition_columns_are_represented_in_native_source_contract() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partition-columns");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_partition_value(&path, ScalarValue::Int32(Some(7)));
    let plan = partition_column_group_by_plan(scan);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "represented");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.partition_columns.len(), 1);
    assert_eq!(source.partition_columns[0].name, "part");
    assert_eq!(source.files[0].partition_values, vec![LiteralValue::Int32(7)]);
    assert!(!source.read_columns.contains(&"part".to_string()));
    assert!(source.projected_output_column_source_refs.contains(&SourceColumnRef::Partition(0)));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parquet_partition_typed_null_values_are_represented_in_native_source_contract() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partition-typed-null");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_partition_value(&path, ScalarValue::Int32(None));
    let plan = partition_column_group_by_plan(scan);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "represented");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.partition_columns.len(), 1);
    assert_eq!(source.partition_columns[0].data_type, DataType::Int32);
    assert_eq!(source.files[0].partition_values, vec![LiteralValue::Null(DataType::Int32)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parquet_partition_columns_without_literal_values_are_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partition-missing-values");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_partition_columns(&path);
    let plan = partition_column_group_by_plan(scan);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.not_selected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::NotSupported { capability_reason: CapabilityReason::NativeSourcePartitionValues, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "rejected");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unused_parquet_binary_partition_columns_do_not_block_native_lowering() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partition-binary");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_typed_partition_value(
        &path,
        Field::new("part", DataType::Binary, false),
        ScalarValue::Binary(Some(vec![1, 2, 3])),
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "absent");
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.partition_columns.is_empty());
    assert!(source.files[0].partition_values.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unused_parquet_utf8_partition_columns_are_not_represented_in_native_source_contract() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partition-utf8");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_typed_partition_value(
        &path,
        Field::new("part", DataType::Utf8, false),
        ScalarValue::Utf8(Some("2026".to_string())),
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "absent");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.partition_columns.is_empty());
    assert!(source.files[0].partition_values.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unused_parquet_utf8_view_partition_values_do_not_enter_native_source_contract() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partition-utf8-view");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_typed_partition_value(
        &path,
        Field::new("part", DataType::Utf8View, false),
        ScalarValue::Utf8View(Some("2026".to_string())),
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "absent");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.partition_columns.is_empty());
    assert!(source.files[0].partition_values.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unused_parquet_utf8_view_partition_typed_null_values_do_not_enter_native_source_contract() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partition-utf8-view-null");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_typed_partition_value(
        &path,
        Field::new("part", DataType::Utf8View, true),
        ScalarValue::Utf8View(None),
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "absent");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.partition_columns.is_empty());
    assert!(source.files[0].partition_values.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unused_unsupported_binary_schema_fields_do_not_block_native_lowering() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-source-binary-schema");
    let path = dir.join("input.parquet");
    write_binary_schema_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(!source.read_columns.contains(&"blob".to_string()));
    assert!(source.schema.field_with_name("blob").is_err());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn list_utf8_projection_lowers_as_native_carry_column() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-list-carry");
    let path = dir.join("input.parquet");
    write_list_utf8_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &scan_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: col("tags", &scan_schema).unwrap(), alias: "tags".to_string() },
            ],
            scan,
        )
        .unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "tags".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn list_utf8_sort_key_is_rejected_locally_during_native_planning() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-list-sort-reject");
    let path = dir.join("input.parquet");
    write_list_utf8_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let sort_exprs =
        LexOrdering::new([PhysicalSortExpr::new(col("tags", &scan_schema).unwrap(), ArrowSortOptions::default())])
            .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, scan));

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unprojected_struct_and_map_schema_fields_do_not_block_native_lowering() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-unprojected-struct-map");
    let path = dir.join("input.parquet");
    write_struct_map_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &scan_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: col("m00", &scan_schema).unwrap(), alias: "m00".to_string() },
            ],
            scan,
        )
        .unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string()]);
    assert!(source.schema.field_with_name("details").is_err());
    assert!(source.schema.field_with_name("attrs").is_err());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn projected_struct_schema_field_is_rejected_with_unsupported_type() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-projected-struct-reject");
    let path = dir.join("input.parquet");
    write_struct_map_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("details", &scan_schema).unwrap(), alias: "details".to_string() }],
            scan,
        )
        .unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn projected_map_schema_field_is_rejected_with_unsupported_type() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-projected-map-reject");
    let path = dir.join("input.parquet");
    write_struct_map_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [ProjectionExpr { expr: col("attrs", &scan_schema).unwrap(), alias: "attrs".to_string() }],
            scan,
        )
        .unwrap(),
    );

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parquet_missing_nullable_columns_are_represented_as_native_typed_nulls() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-missing-nullable-columns");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_missing_nullable_column(&path);
    let scan_schema = scan.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(
        col("missing_payload", &scan_schema).unwrap(),
        "missing_payload".to_string(),
    )]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &scan_schema).unwrap()])
        .schema(Arc::clone(&scan_schema))
        .alias("sum_m00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(AggregateMode::Single, group_by, vec![Arc::new(sum)], vec![None], scan, scan_schema)
            .unwrap(),
    );
    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("sum_m00", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(true, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(1)));
    let plan = Arc::new(LocalLimitExec::new(sort, 1));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].schema_mapping_disposition.as_str(), "missing_nullable_columns");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.missing_columns.len(), 1);
    assert_eq!(source.missing_columns[0].name, "missing_payload");
    assert_eq!(
        source.projected_output_column_source_refs,
        vec![SourceColumnRef::File(2), SourceColumnRef::MissingNull(0)]
    );

    fs::remove_dir_all(dir).unwrap();
}

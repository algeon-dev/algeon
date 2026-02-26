use super::*;

#[test]
fn pushed_down_parquet_predicate_with_filter_is_reader_pushdown() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
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

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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
fn pushed_down_parquet_predicate_without_filter_is_reader_pushdown() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate-without-filter");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
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
fn pushed_down_scan_only_decimal_predicate_uses_reader_pushdown() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-decimal-post-scan");
    let path = dir.join("input.parquet");
    write_decimal_filter_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_decimal_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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
fn pushed_down_parquet_predicate_with_different_filter_uses_native_post_scan_filter() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-pushed-predicate-mismatch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate_value(&path, 20);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert!(matches!(result.report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
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

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].file_size_bytes, Some(fs::metadata(&path).unwrap().len()));
    assert_eq!(source.files[0].stable_id, format!("local:{}:{}", path.display(), fs::metadata(&path).unwrap().len()));
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
fn partial_byte_range_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-partial-ranged-file");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_partial_byte_range(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceByteRange, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].ranged_file_count, 1);
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "rejected");
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "byte_range"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn partial_byte_range_with_access_plan_lowers_to_selected_row_groups() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-ranged-access-plan");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let access_plan = ParquetAccessPlan::new(vec![
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
        RowGroupAccess::Skip,
        RowGroupAccess::Scan,
    ]);
    let scan = parquet_scan_with_partial_byte_range_and_access_plan(&path, access_plan);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.file_count(), 1);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1, 3][..]));
    assert_eq!(source.diagnostics.ranged_file_count, 1);
    assert_eq!(source.diagnostics.byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert!(source.diagnostics.direct_device_read);
    assert!(source.diagnostics.rejected_features.is_empty());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "row_group_selection");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

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

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
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
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceByteRange, .. }
    ));
    assert_eq!(result.source_diagnostics[0].ranged_file_count, 2);
    assert_eq!(result.source_diagnostics[0].byte_range_disposition.as_str(), "rejected");
    assert!(result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "byte_range"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn remote_parquet_object_store_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-remote-store");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_object_store_url(&path, ObjectStoreUrl::parse("test:///").unwrap());
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceNonLocalObjectStore, .. }
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

#[test]
fn custom_parquet_file_reader_factory_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-custom-reader-factory");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_custom_reader_factory(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceParquetFileReaderFactory, .. }
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourcePartitionValues, .. }
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(!source.read_columns.contains(&"blob".to_string()));
    assert!(source.schema.field_with_name("blob").is_err());

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

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
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

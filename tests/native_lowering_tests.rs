use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Float64Array, Int32Array, Int64Array, StringArray},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf_query_engine::{
    AggregateFunction, Expr, LiteralValue, PlanNode, ResultSink, SourceColumnRef, SourceNode, ZERO_COLUMN_COUNT_COLUMN,
};
use datafusion::{
    common::ScalarValue,
    datasource::{
        physical_plan::{
            FileGroup, FileScanConfigBuilder, FileSource, ParquetSource,
            parquet::{ParquetAccessPlan, RowGroupAccess},
        },
        source::DataSourceExec,
        table_schema::TableSchema,
    },
    execution::object_store::ObjectStoreUrl,
    functions_aggregate::{count::count_udaf, first_last::first_value_udaf, min_max::max_udaf, sum::sum_udaf},
    logical_expr::Operator,
    parquet::{
        arrow::{
            ArrowWriter,
            arrow_reader::{RowSelection, RowSelector},
        },
        file::{
            properties::WriterProperties,
            reader::{FileReader, SerializedFileReader},
        },
    },
    physical_expr::{
        LexOrdering, PhysicalExpr, PhysicalSortExpr,
        aggregate::{AggregateExprBuilder, AggregateFunctionExpr},
        expressions::{CastExpr, NotExpr, TryCastExpr, binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{CandidateOutcome, FallbackReason, NativeLoweringResult, lower_physical_plan_to_native};

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
    assert_eq!(projection.expressions.len(), 2);
    let filter_node = plan.node(projection.input).expect("projection input should exist");
    let PlanNode::Filter(filter) = &filter_node.kind else {
        panic!("projection input should be a filter node");
    };
    assert!(matches!(
        &filter.predicate,
        Expr::Binary { right, .. } if matches!(right.as_ref(), Expr::Literal(LiteralValue::Int32(10)))
    ));
    let source_node = plan.node(filter.input).expect("filter input should exist");
    let PlanNode::Source(SourceNode::LocalParquet(source)) = &source_node.kind else {
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
            PlanNode::Source(SourceNode::LocalParquet(source)) => Some(source),
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
fn unsupported_projection_expression_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-unsupported-expression");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Modulo);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rewritten_count(), 0);
    assert_eq!(result.rejected_count(), 1);
    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(LocalParquetScan)))))",
                2,
                5,
            ),
            reason: FallbackReason::UnsupportedExpression,
        },
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unsupported_source_type_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-unsupported-type");
    let path = dir.join("input.parquet");
    write_parquet_file_with_string_column(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_scan_limit_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-scan-property");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path).with_fetch(Some(1)).unwrap();
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourceScanLimit, .. }
    ));

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
fn pushed_down_parquet_predicate_with_filter_is_pruning_only_evidence() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-pushed-predicate");
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
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_parquet_predicate_without_filter_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-pushed-predicate-without-filter");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate(&path);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourcePushedDownParquetPredicate, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "rejected");
    assert!(
        result.source_diagnostics[0]
            .rejected_features
            .iter()
            .any(|feature| feature.as_str() == "pushed_down_parquet_predicate")
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_down_parquet_predicate_with_different_filter_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-pushed-predicate-mismatch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate_value(&path, 20);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourcePushedDownParquetPredicate, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(!result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "rejected");
    assert!(
        result.source_diagnostics[0]
            .rejected_features
            .iter()
            .any(|feature| feature.as_str() == "pushed_down_parquet_predicate")
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn full_file_byte_range_coalesces_to_native_source() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-ranged-file");
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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-partial-ranged-file");
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
fn split_full_file_byte_ranges_coalesce_without_duplicate_native_files() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-split-ranged-file");
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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-overlap-ranged-file");
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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-remote-store");
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
fn parquet_partition_columns_are_represented_in_native_source_contract() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-partition-columns");
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
fn parquet_partition_columns_without_literal_values_are_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-partition-missing-values");
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
fn parquet_utf8_partition_columns_are_rejected_until_string_partition_contract_is_modeled() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-partition-utf8");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_typed_partition_value(
        &path,
        Field::new("part", DataType::Utf8, false),
        ScalarValue::Utf8(Some("2026".to_string())),
    );
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].partition_column_disposition.as_str(), "rejected");

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_physical_plan_has_stable_native_lowering_outcome() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-friendly");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = result.report.outcomes()[0] else {
        panic!("direct-friendly SQL plan should lower to native");
    };
    assert_eq!(candidate.shape, "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))");
    assert_eq!(candidate.operator_count, 5);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_native_aggregate_parity_functions_lower() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-aggregate-parity");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_NATIVE_AGGREGATE_PARITY_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let functions = native_aggregate_functions(&result.plans[0]);
    assert_eq!(functions, vec![AggregateFunction::Min, AggregateFunction::Max, AggregateFunction::Count]);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_count_star_lowers_with_synthetic_argument_column() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-count-star");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_COUNT_STAR_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 2);
    assert_eq!(native_aggregate_functions(&result.plans[0]), vec![AggregateFunction::Sum, AggregateFunction::Count]);
    let count_expr = aggregate_exprs
        .iter()
        .find(|expr| expr.function == AggregateFunction::Count)
        .expect("native aggregate should include COUNT(*)");
    assert!(matches!(
        &count_expr.expr,
        Expr::Column(name) if name.starts_with("__cudf_aggregate_arg_")
    ));
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_zero_column_count_lowers_to_represented_source() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-zero-column-count");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = count_only_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].read_column_count, 0);
    assert_eq!(result.source_diagnostics[0].projected_output_column_count, 0);
    assert_eq!(result.source_diagnostics[0].pure_filter_column_count, 0);
    assert_eq!(result.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.is_zero_column_count_source());
    assert!(source.read_columns.is_empty());
    assert!(source.read_column_file_indices.is_empty());
    assert_eq!(source.diagnostics.zero_column_count_disposition.as_str(), "represented");
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(name) if name == ZERO_COLUMN_COUNT_COLUMN));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_count_only_lowers_to_native_zero_column_count_without_manual_plan_shape() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-count-only");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, count_only_sql_session_config(), SQL_COUNT_ONLY_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rewritten {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "ProjectionExec(AggregateExec(LocalParquetScan))",
                1,
                3,
            ),
        }
    );
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].read_column_count, 0);
    assert_eq!(result.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");

    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.is_zero_column_count_source());
    assert!(source.read_columns.is_empty());
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(name) if name == ZERO_COLUMN_COUNT_COLUMN));
    assert!(matches!(result.plans[0].root_node().kind, PlanNode::Projection(_)));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_avg_aggregate_lowers_to_native_avg() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_AVG_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Avg);
    assert_eq!(aggregate_exprs[0].alias, "avg(bench.m00)");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "payload".to_string()]);
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_global_avg_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-global-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, FallbackReason::UnsupportedOperator);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_filtered_avg_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-filtered-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_FILTERED_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, FallbackReason::UnsupportedExpression);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_distinct_avg_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distinct-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_DISTINCT_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, FallbackReason::UnsupportedOperator);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_pushed_predicate_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(total_row_groups, 4);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].selected_row_groups.unwrap() < total_row_groups);

    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[2, 3][..]));
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert_eq!(source.selected_row_group_count(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_filter_without_scan_predicate_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-filter-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;
    let plan = sql_plan_without_scan_predicate(plan);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(total_row_groups, 4);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "absent");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].selected_row_groups.unwrap() < total_row_groups);

    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[2, 3][..]));
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert_eq!(source.selected_row_group_count(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_physical_plan_lowers_to_native_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-default");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = result.report.outcomes()[0] else {
        panic!("default SQL plan should lower to native");
    };
    assert_eq!(
        candidate.shape,
        "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(FinalAggregateExec(RepartitionExec(PartialAggregateExec(FilterExec(RepartitionExec(LocalParquetScan)))))))"
    );
    assert_eq!(candidate.operator_count, 9);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_avg_aggregate_lowers_to_native_avg() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-avg");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_AVG_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Avg);
    assert_eq!(aggregate_exprs[0].alias, "avg(bench.m00)");
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(name) if name == "__cudf_aggregate_arg_0"));
    assert!(native_projection_exprs(&result.plans[0]).iter().any(|expr| matches!(
        expr,
        Expr::Alias { expr, alias }
            if alias == "__cudf_aggregate_arg_0"
                && matches!(
                    expr.as_ref(),
                    Expr::Cast { expr, target_type }
                        if target_type == &DataType::Float64
                            && matches!(expr.as_ref(), Expr::Column(name) if name == "m00")
                )
    )));
    assert_eq!(result.plans[0].output_schema().field_with_name("avg_m00").unwrap().data_type(), &DataType::Float64);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "payload".to_string()]);
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_aggregate_filter_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-filtered-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_FILTERED_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_distinct_aggregate_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-distinct-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_DISTINCT_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::UnsupportedOperator);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_final_partial_aggregate_function_mismatch_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-function-mismatch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_final_aggregate(plan, |final_aggregate| {
        let original = &final_aggregate.aggr_expr()[0];
        aggregate_expr_on_column_like(original, max_udaf(), final_aggregate.input().schema(), 1, false, vec![])
    });

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_filter_expr_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-filter-mutation");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_partial_aggregate(plan, |partial_aggregate| {
        let input_schema = partial_aggregate.input().schema();
        let predicate = binary(
            col("key", &input_schema).unwrap(),
            Operator::GtEq,
            lit(ScalarValue::Int32(Some(10))),
            &input_schema,
        )
        .unwrap();
        Arc::new(
            AggregateExec::try_new(
                *partial_aggregate.mode(),
                partial_aggregate.group_expr().clone(),
                partial_aggregate.aggr_expr().to_vec(),
                vec![Some(predicate)],
                Arc::clone(partial_aggregate.input()),
                partial_aggregate.input_schema(),
            )
            .unwrap(),
        )
    });

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_distinct_expr_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-distinct-mutation");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_partial_aggregate(plan, |partial_aggregate| {
        let original = &partial_aggregate.aggr_expr()[0];
        let distinct = aggregate_expr_like(original, sum_udaf(), partial_aggregate.input().schema(), true, vec![]);
        Arc::new(partial_aggregate.with_new_aggr_exprs(vec![distinct]))
    });

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_order_by_expr_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-order-by-mutation");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_partial_and_final_aggregate(
        plan,
        |partial_aggregate| {
            let input_schema = partial_aggregate.input().schema();
            let order_by =
                vec![PhysicalSortExpr::new(col("key", &input_schema).unwrap(), ArrowSortOptions::new(false, false))];
            let original = &partial_aggregate.aggr_expr()[0];
            let ordered = aggregate_expr_like(original, first_value_udaf(), input_schema, false, order_by);
            Arc::new(partial_aggregate.with_new_aggr_exprs(vec![ordered]))
        },
        |final_aggregate| {
            let input_schema = final_aggregate.input().schema();
            let order_by =
                vec![PhysicalSortExpr::new(col("key", &input_schema).unwrap(), ArrowSortOptions::new(false, false))];
            let original = &final_aggregate.aggr_expr()[0];
            let ordered = aggregate_expr_on_column_like(original, first_value_udaf(), input_schema, 1, false, order_by);
            ordered
        },
    );

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_without_partial_source_shape_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-unexpected-source");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_distributed_final_input_replaced_by_scan(plan);

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_filter_plan_lowers_when_scan_predicate_is_absent() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-filter-rewrite");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;
    let plan = sql_plan_without_scan_predicate(plan);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = result.report.outcomes()[0] else {
        panic!("direct-friendly SQL filter plan should lower to native");
    };
    assert_eq!(candidate.shape, "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))");
    assert_eq!(candidate.operator_count, 5);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.plans[0].node_count(), 7);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "absent");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_not_filter_lowers_to_native_expr() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-not-filter");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let payload_lt_10 =
        binary(col("payload", &scan_schema).unwrap(), Operator::Lt, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let predicate = Arc::new(NotExpr::new(payload_lt_10));
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let filter = native_filter_expr(&result.plans[0]);
    assert_eq!(filter.to_string(), "NOT (payload < 10)");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn explicit_numeric_cast_lowers_to_native_expr_and_reads_inner_column() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let filter_schema = scan_schema;
    let cast_payload = Arc::new(CastExpr::new(col("payload", &filter_schema).unwrap(), DataType::Int64, None));
    let x00 = binary(cast_payload, Operator::Plus, col("m00", &filter_schema).unwrap(), &filter_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_projected_x00(scan, predicate, x00);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let projection = native_projection_exprs(&result.plans[0])
        .iter()
        .find(|expr| matches!(expr, Expr::Alias { alias, .. } if alias == "x00"))
        .expect("native plan should include x00 projection");
    assert!(matches!(
        projection,
        Expr::Alias { expr, .. }
            if matches!(
                expr.as_ref(),
                Expr::Binary { left, .. }
                    if matches!(
                        left.as_ref(),
                        Expr::Cast { expr, target_type }
                            if matches!(expr.as_ref(), Expr::Column(name) if name == "payload")
                                && *target_type == DataType::Int64
                    )
            )
    ));
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn try_cast_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-try-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let try_cast = Arc::new(TryCastExpr::new(col("payload", &scan_schema).unwrap(), DataType::Int64));
    let predicate = binary(try_cast, Operator::GtEq, lit(ScalarValue::Int64(Some(10))), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn non_numeric_cast_targets_are_rejected_during_native_planning() {
    for (name, target_type, literal) in [
        ("string", DataType::Utf8, ScalarValue::Utf8(Some("10".to_string()))),
        ("decimal", DataType::Decimal128(10, 0), ScalarValue::Decimal128(Some(10), 10, 0)),
        (
            "timestamp",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            ScalarValue::TimestampNanosecond(Some(10), None),
        ),
    ] {
        let dir = unique_temp_dir(format!("datafusion-cudf-native-lowering-{name}-cast").as_str());
        let path = dir.join("input.parquet");
        write_numeric_parquet_file(&path);
        let scan = parquet_scan(&path);
        let scan_schema = scan.schema();
        let cast = Arc::new(CastExpr::new(col("payload", &scan_schema).unwrap(), target_type, None));
        let predicate = binary(cast, Operator::Eq, lit(literal), &scan_schema).unwrap();
        let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

        let result = lower_physical_plan_to_native(plan);

        assert!(result.plans.is_empty(), "{name} cast should not lower");
        assert_eq!(result.rejected_count(), 1, "{name} cast outcomes={:?}", result.report.outcomes());
        assert!(
            matches!(
                result.report.outcomes()[0],
                CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
            ),
            "{name} cast outcomes={:?}",
            result.report.outcomes()
        );

        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn lossy_numeric_cast_is_rejected_during_native_planning() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-lossy-cast");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let cast = Arc::new(CastExpr::new(col("m00", &scan_schema).unwrap(), DataType::Int32, None));
    let predicate = binary(cast, Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema).unwrap();
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_physical_plan_without_scan_predicate_lowers_to_native_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-rewrite");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_NO_FILTER_BENCHMARK_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = result.report.outcomes()[0] else {
        panic!("direct-friendly SQL no-filter plan should lower to native");
    };
    assert_eq!(candidate.shape, "SortExec(fetch)(ProjectionExec(AggregateExec(LocalParquetScan)))");
    assert_eq!(candidate.operator_count, 4);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "absent");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "m01".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn every_rejected_native_candidate_has_stable_rejection_code() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-rejection-codes");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let rejected_results = [
        lower_physical_plan_to_native(aggregate_sort_limit_plan(
            parquet_scan_with_partial_byte_range(&path),
            Operator::Plus,
        )),
        lower_physical_plan_to_native(aggregate_sort_limit_plan_without_filter(
            parquet_scan_with_pushed_down_predicate(&path),
            Operator::Plus,
        )),
    ];

    for result in rejected_results {
        assert_eq!(result.rejected_count(), 1);
        for outcome in result.report.outcomes() {
            let CandidateOutcome::Rejected { reason, .. } = outcome else {
                panic!("expected only rejected native candidates");
            };
            assert!(!reason.code().is_empty());
            assert_ne!(reason.code(), "plan_property_mismatch");
        }
    }

    fs::remove_dir_all(dir).unwrap();
}

fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1),
    );
    let df =
        futures::executor::block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

async fn sql_benchmark_plan(path: &std::path::Path, config: SessionConfig) -> Arc<dyn ExecutionPlan> {
    sql_plan(path, config, SQL_BENCHMARK_QUERY).await
}

async fn sql_plan(path: &std::path::Path, config: SessionConfig, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(config);
    ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    let df = ctx.sql(sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

fn sql_plan_without_scan_predicate(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let sort_plan = plan;
    let sort = sort_plan.as_any().downcast_ref::<SortExec>().expect("SQL plan root should be SortExec");
    let post_projection_plan = Arc::clone(sort.input());
    let post_projection = post_projection_plan
        .as_any()
        .downcast_ref::<ProjectionExec>()
        .expect("SQL plan should have a post-aggregate ProjectionExec");
    let aggregate_plan = Arc::clone(post_projection.input());
    let aggregate = aggregate_plan
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("SQL plan should have an AggregateExec below post projection");
    let filter_plan = Arc::clone(aggregate.input());
    let filter =
        filter_plan.as_any().downcast_ref::<FilterExec>().expect("SQL plan should have a FilterExec below aggregate");
    let source_plan = Arc::clone(filter.input());
    let source = source_plan
        .as_any()
        .downcast_ref::<DataSourceExec>()
        .expect("SQL plan filter input should be a DataSourceExec");

    let source_without_predicate = parquet_source_exec_without_predicate(source);
    let filter_without_predicate = filter_plan.with_new_children(vec![source_without_predicate]).unwrap();
    let aggregate_without_predicate = aggregate_plan.with_new_children(vec![filter_without_predicate]).unwrap();
    let projection_without_predicate =
        post_projection_plan.with_new_children(vec![aggregate_without_predicate]).unwrap();
    sort_plan.with_new_children(vec![projection_without_predicate]).unwrap()
}

fn parquet_source_exec_without_predicate(source: &DataSourceExec) -> Arc<dyn ExecutionPlan> {
    let (file_scan, _parquet_source) =
        source.downcast_to_file_source::<ParquetSource>().expect("SQL scan should use ParquetSource");
    let projection_indices = file_scan.file_source().projection().map(|projection| projection.column_indices());
    let parquet_source = ParquetSource::new(file_scan.file_source().table_schema().clone());
    let mut builder =
        FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source) as Arc<dyn FileSource>);
    if let Some(projection_indices) = projection_indices {
        builder = builder.with_projection_indices(Some(projection_indices)).unwrap();
    }
    DataSourceExec::from_data_source(builder.build())
}

struct DefaultDistributedSqlPlanParts {
    root: Arc<dyn ExecutionPlan>,
    sort: Arc<dyn ExecutionPlan>,
    projection: Arc<dyn ExecutionPlan>,
    final_aggregate: Arc<dyn ExecutionPlan>,
    between_aggregate_repartition: Arc<dyn ExecutionPlan>,
    partial_aggregate: Arc<dyn ExecutionPlan>,
    scan_input: Arc<dyn ExecutionPlan>,
}

fn default_distributed_sql_plan_parts(plan: Arc<dyn ExecutionPlan>) -> DefaultDistributedSqlPlanParts {
    let root = plan;
    let root_spm = root.as_any().downcast_ref::<SortPreservingMergeExec>().expect("default SQL root should be SPM");
    let sort = Arc::clone(root_spm.input());
    let sort_exec = sort.as_any().downcast_ref::<SortExec>().expect("default SQL SPM input should be SortExec");
    let projection = Arc::clone(sort_exec.input());
    let projection_exec =
        projection.as_any().downcast_ref::<ProjectionExec>().expect("default SQL sort input should be ProjectionExec");
    let final_aggregate = Arc::clone(projection_exec.input());
    let final_aggregate_exec = final_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL projection input should be final AggregateExec");
    assert!(matches!(final_aggregate_exec.mode(), AggregateMode::FinalPartitioned));
    let between_aggregate_repartition = Arc::clone(final_aggregate_exec.input());
    let repartition = between_aggregate_repartition
        .as_any()
        .downcast_ref::<RepartitionExec>()
        .expect("default SQL final aggregate input should be RepartitionExec");
    let partial_aggregate = Arc::clone(repartition.input());
    let partial_aggregate_exec = partial_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL repartition input should be partial AggregateExec");
    assert!(matches!(partial_aggregate_exec.mode(), AggregateMode::Partial));
    let filter = partial_aggregate_exec
        .input()
        .as_any()
        .downcast_ref::<FilterExec>()
        .expect("default SQL partial aggregate input should be FilterExec");
    let source_repartition = filter
        .input()
        .as_any()
        .downcast_ref::<RepartitionExec>()
        .expect("default SQL filter input should be RepartitionExec");
    let scan_input = Arc::clone(source_repartition.input());

    DefaultDistributedSqlPlanParts {
        root,
        sort,
        projection,
        final_aggregate,
        between_aggregate_repartition,
        partial_aggregate,
        scan_input,
    }
}

fn rebuild_default_distributed_sql_plan(
    parts: &DefaultDistributedSqlPlanParts,
    final_aggregate: Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let projection = Arc::clone(&parts.projection).with_new_children(vec![final_aggregate]).unwrap();
    let sort = Arc::clone(&parts.sort).with_new_children(vec![projection]).unwrap();
    Arc::clone(&parts.root).with_new_children(vec![sort]).unwrap()
}

fn with_rewritten_distributed_final_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl FnOnce(&AggregateExec) -> Arc<AggregateFunctionExpr>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let final_aggregate = parts
        .final_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL final aggregate should be AggregateExec");
    let rewritten_final: Arc<dyn ExecutionPlan> =
        Arc::new(final_aggregate.with_new_aggr_exprs(vec![rewrite(final_aggregate)]));
    rebuild_default_distributed_sql_plan(&parts, rewritten_final)
}

fn with_rewritten_distributed_partial_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    rewrite: impl FnOnce(&AggregateExec) -> Arc<dyn ExecutionPlan>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let partial_aggregate = parts
        .partial_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL partial aggregate should be AggregateExec");
    let rewritten_partial = rewrite(partial_aggregate);
    let repartition =
        Arc::clone(&parts.between_aggregate_repartition).with_new_children(vec![rewritten_partial]).unwrap();
    let final_aggregate = Arc::clone(&parts.final_aggregate).with_new_children(vec![repartition]).unwrap();
    rebuild_default_distributed_sql_plan(&parts, final_aggregate)
}

fn with_rewritten_distributed_partial_and_final_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    rewrite_partial: impl FnOnce(&AggregateExec) -> Arc<dyn ExecutionPlan>,
    rewrite_final: impl FnOnce(&AggregateExec) -> Arc<AggregateFunctionExpr>,
) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let partial_aggregate = parts
        .partial_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL partial aggregate should be AggregateExec");
    let final_aggregate = parts
        .final_aggregate
        .as_any()
        .downcast_ref::<AggregateExec>()
        .expect("default SQL final aggregate should be AggregateExec");
    let rewritten_partial = rewrite_partial(partial_aggregate);
    let repartition =
        Arc::clone(&parts.between_aggregate_repartition).with_new_children(vec![rewritten_partial]).unwrap();
    let rewritten_final: Arc<dyn ExecutionPlan> =
        Arc::new(final_aggregate.with_new_aggr_exprs(vec![rewrite_final(final_aggregate)]));
    let rewritten_final = rewritten_final.with_new_children(vec![repartition]).unwrap();
    rebuild_default_distributed_sql_plan(&parts, rewritten_final)
}

fn with_distributed_final_input_replaced_by_scan(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let parts = default_distributed_sql_plan_parts(plan);
    let final_aggregate =
        Arc::clone(&parts.final_aggregate).with_new_children(vec![Arc::clone(&parts.scan_input)]).unwrap();
    rebuild_default_distributed_sql_plan(&parts, final_aggregate)
}

fn aggregate_expr_like(
    existing: &AggregateFunctionExpr,
    udaf: Arc<datafusion::logical_expr::AggregateUDF>,
    schema: Arc<Schema>,
    distinct: bool,
    order_by: Vec<PhysicalSortExpr>,
) -> Arc<AggregateFunctionExpr> {
    Arc::new(
        AggregateExprBuilder::new(udaf, existing.expressions())
            .schema(schema)
            .order_by(order_by)
            .with_distinct(distinct)
            .alias(existing.name())
            .build()
            .unwrap(),
    )
}

fn aggregate_expr_on_column_like(
    existing: &AggregateFunctionExpr,
    udaf: Arc<datafusion::logical_expr::AggregateUDF>,
    schema: Arc<Schema>,
    column_index: usize,
    distinct: bool,
    order_by: Vec<PhysicalSortExpr>,
) -> Arc<AggregateFunctionExpr> {
    let column_name = schema.field(column_index).name();
    Arc::new(
        AggregateExprBuilder::new(udaf, vec![col(column_name, &schema).unwrap()])
            .schema(schema)
            .order_by(order_by)
            .with_distinct(distinct)
            .alias(existing.name())
            .build()
            .unwrap(),
    )
}

fn assert_distributed_aggregate_rejection(result: &NativeLoweringResult, expected_reason: FallbackReason) {
    assert_native_rejection(result, expected_reason);
}

fn assert_native_rejection(result: &NativeLoweringResult, expected_reason: FallbackReason) {
    assert!(result.plans.is_empty());
    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 0);
    assert_eq!(result.rejected_count(), 1);
    let CandidateOutcome::Rejected { reason, .. } = result.report.outcomes()[0].clone() else {
        panic!("expected rejected native candidate");
    };
    assert_eq!(reason, expected_reason);
}

fn direct_friendly_sql_session_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

fn count_only_sql_session_config() -> SessionConfig {
    let mut config = direct_friendly_sql_session_config().with_collect_statistics(false);
    config.options_mut().execution.collect_statistics = false;
    config
}

const SQL_BENCHMARK_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_NO_FILTER_BENCHMARK_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00
FROM bench
GROUP BY key
ORDER BY sum_x00 DESC, key ASC
LIMIT 2
";

const SQL_NATIVE_AGGREGATE_PARITY_QUERY: &str = "
SELECT key, MIN(m00) AS min_m00, MAX(m01) AS max_m01, COUNT(payload) AS count_payload
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

const SQL_COUNT_STAR_QUERY: &str = "
SELECT key, SUM(m00 + m01) AS sum_x00, COUNT(*) AS count_rows
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY key ASC
LIMIT 2
";

const SQL_COUNT_ONLY_QUERY: &str = "
SELECT COUNT(*) AS count_rows
FROM bench
";

const SQL_AVG_AGGREGATE_QUERY: &str = "
SELECT key, AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_GLOBAL_AVG_QUERY: &str = "
SELECT AVG(m00) AS avg_m00
FROM bench
WHERE payload >= 10
ORDER BY avg_m00 DESC
LIMIT 1
";

const SQL_FILTERED_AVG_QUERY: &str = "
SELECT key, AVG(m00) FILTER (WHERE payload >= 10) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_DISTINCT_AVG_QUERY: &str = "
SELECT key, AVG(DISTINCT m00) AS avg_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY avg_m00 DESC, key ASC
LIMIT 2
";

const SQL_FILTERED_AGGREGATE_QUERY: &str = "
SELECT key, SUM(m00) FILTER (WHERE payload >= 10) AS sum_m00
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY sum_m00 DESC, key ASC
LIMIT 2
";

const SQL_DISTINCT_AGGREGATE_QUERY: &str = "
SELECT key, COUNT(DISTINCT payload) AS distinct_payload
FROM bench
WHERE payload >= 10
GROUP BY key
ORDER BY distinct_payload DESC, key ASC
LIMIT 2
";

fn parquet_scan_with_pushed_down_predicate(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_pushed_down_predicate_value(path, 10)
}

fn parquet_scan_with_pushed_down_predicate_value(path: &std::path::Path, value: i32) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let predicate = binary(
        col("payload", &scan_schema).unwrap(),
        Operator::GtEq,
        lit(ScalarValue::Int32(Some(value))),
        &scan_schema,
    )
    .unwrap();
    let parquet_source = parquet_source.with_predicate(predicate);
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_source(Arc::new(parquet_source)).build();
    DataSourceExec::from_data_source(file_scan)
}

fn parquet_scan_with_access_plan(path: &std::path::Path, access_plan: ParquetAccessPlan) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .map(|group| {
            FileGroup::new(
                group.files().iter().cloned().map(|file| file.with_extensions(Arc::new(access_plan.clone()))).collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

fn parquet_scan_with_byte_range(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| vec![(0, size)])
}

fn parquet_scan_with_partial_byte_range(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| vec![(0, size / 2)])
}

fn parquet_scan_with_split_full_byte_ranges(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| {
        let midpoint = size / 2;
        vec![(0, midpoint), (midpoint, size)]
    })
}

fn parquet_scan_with_overlapping_byte_ranges(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_byte_ranges(path, |size| {
        let midpoint = size / 2;
        vec![(0, midpoint + 1), (midpoint, size)]
    })
}

fn parquet_scan_with_byte_ranges(
    path: &std::path::Path,
    ranges: impl Fn(i64) -> Vec<(i64, i64)>,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .map(|group| {
            FileGroup::new(
                group
                    .files()
                    .iter()
                    .flat_map(|file| {
                        let size = file.object_meta.size as i64;
                        ranges(size)
                            .into_iter()
                            .map(|(start, end)| file.clone().with_range(start, end))
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

fn parquet_scan_with_object_store_url(
    path: &std::path::Path,
    object_store_url: ObjectStoreUrl,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_scan = FileScanConfigBuilder::new(object_store_url, Arc::clone(file_scan.file_source()))
        .with_file_groups(file_scan.file_groups.clone())
        .build();
    DataSourceExec::from_data_source(file_scan)
}

fn parquet_scan_with_partition_columns(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_partition_field(path, Field::new("part", DataType::Int32, false))
}

fn parquet_scan_with_partition_value(path: &std::path::Path, value: ScalarValue) -> Arc<dyn ExecutionPlan> {
    parquet_scan_with_typed_partition_value(path, Field::new("part", DataType::Int32, false), value)
}

fn parquet_scan_with_typed_partition_value(
    path: &std::path::Path,
    partition_field: Field,
    value: ScalarValue,
) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan_with_partition_field(path, partition_field);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_groups = file_scan
        .file_groups
        .iter()
        .map(|group| {
            FileGroup::new(
                group.files().iter().cloned().map(|file| file.with_partition_values(vec![value.clone()])).collect(),
            )
        })
        .collect();
    let file_scan = FileScanConfigBuilder::from(file_scan.clone()).with_file_groups(file_groups).build();
    DataSourceExec::from_data_source(file_scan)
}

fn parquet_scan_with_partition_field(path: &std::path::Path, partition_field: Field) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let projected_indices = (0..=file_scan.file_schema().fields().len()).collect::<Vec<_>>();
    let table_schema = TableSchema::new(Arc::clone(file_scan.file_schema()), vec![Arc::new(partition_field)]);
    let parquet_source = ParquetSource::new(table_schema);
    let file_scan = FileScanConfigBuilder::new(file_scan.object_store_url.clone(), Arc::new(parquet_source))
        .with_file_groups(file_scan.file_groups.clone())
        .with_projection_indices(Some(projected_indices))
        .unwrap()
        .build();
    DataSourceExec::from_data_source(file_scan)
}

fn partition_column_group_by_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let predicate =
        binary(col("part", &scan_schema).unwrap(), Operator::Eq, lit(ScalarValue::Int32(Some(7))), &scan_schema)
            .unwrap();
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());
    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("part", &filter_schema).unwrap(), alias: "part".to_string() },
                ProjectionExpr { expr: col("m00", &filter_schema).unwrap(), alias: "m00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );
    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("part", &projection_schema).unwrap(), "part".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_m00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );
    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("part", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(1)));
    Arc::new(LocalLimitExec::new(sort, 1))
}

fn native_local_parquet_source(plan: &cudf_query_engine::QueryPlan) -> &cudf_query_engine::LocalParquetSource {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(SourceNode::LocalParquet(source)) => Some(source),
            _ => None,
        })
        .expect("native plan should contain a local parquet source")
}

fn native_aggregate_functions(plan: &cudf_query_engine::QueryPlan) -> Vec<AggregateFunction> {
    native_aggregate_exprs(plan).iter().map(|expr| expr.function).collect()
}

fn native_aggregate_exprs(plan: &cudf_query_engine::QueryPlan) -> &[cudf_query_engine::AggregateExpr] {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate.spec.aggr_exprs.as_slice()),
            _ => None,
        })
        .expect("native plan should contain an aggregate node")
}

fn native_projection_exprs(plan: &cudf_query_engine::QueryPlan) -> &[Expr] {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Projection(projection) => Some(projection.expressions.as_slice()),
            _ => None,
        })
        .expect("native plan should contain a projection node")
}

fn native_filter_expr(plan: &cudf_query_engine::QueryPlan) -> &Expr {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Filter(filter) => Some(&filter.predicate),
            _ => None,
        })
        .expect("native plan should contain a filter node")
}

fn aggregate_sort_limit_plan(scan: Arc<dyn ExecutionPlan>, projection_operator: Operator) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan(scan, projection_operator);
    Arc::new(LocalLimitExec::new(sort, 2))
}

fn count_only_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(ScalarValue::Int64(Some(1)))])
        .schema(Arc::clone(&scan_schema))
        .alias("count_rows")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![Arc::new(count)],
            vec![None],
            scan,
            scan_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("count_rows", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(true, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(1)));
    Arc::new(LocalLimitExec::new(sort, 1))
}

fn aggregate_sort_limit_plan_with_filter(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan_with_filter(scan, predicate, projection_operator);
    Arc::new(LocalLimitExec::new(sort, 2))
}

fn aggregate_sort_limit_plan_without_filter(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan_without_filter(scan, projection_operator);
    Arc::new(LocalLimitExec::new(sort, 2))
}

fn aggregate_topk_plan(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
    fetch: usize,
) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan(scan, projection_operator);
    let sort = sort.as_any().downcast_ref::<SortExec>().expect("aggregate_sort_plan returns SortExec");
    Arc::new(sort.with_fetch(Some(fetch)))
}

fn aggregate_sort_plan(scan: Arc<dyn ExecutionPlan>, projection_operator: Operator) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    aggregate_sort_plan_with_filter(scan, predicate, projection_operator)
}

fn aggregate_sort_plan_with_filter(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let filter_schema = filter.schema();
    let x00 = binary(
        col("m00", &filter_schema).unwrap(),
        projection_operator,
        col("m01", &filter_schema).unwrap(),
        &filter_schema,
    )
    .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    Arc::new(SortExec::new(sort_exprs, aggregate))
}

fn aggregate_sort_limit_plan_with_projected_x00(
    scan: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
    x00: Arc<dyn PhysicalExpr>,
) -> Arc<dyn ExecutionPlan> {
    let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(predicate, scan).unwrap());

    let filter_schema = filter.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &filter_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            filter,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));
    Arc::new(LocalLimitExec::new(sort, 2))
}

fn aggregate_sort_plan_without_filter(
    scan: Arc<dyn ExecutionPlan>,
    projection_operator: Operator,
) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let x00 =
        binary(col("m00", &scan_schema).unwrap(), projection_operator, col("m01", &scan_schema).unwrap(), &scan_schema)
            .unwrap();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &scan_schema).unwrap(), alias: "key".to_string() },
                ProjectionExpr { expr: x00, alias: "x00".to_string() },
            ],
            scan,
        )
        .unwrap(),
    );

    let projection_schema = projection.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &projection_schema).unwrap(), "key".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("x00", &projection_schema).unwrap()])
        .schema(Arc::clone(&projection_schema))
        .alias("sum_x00")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(sum)],
            vec![None],
            projection,
            projection_schema,
        )
        .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([
        PhysicalSortExpr::new(col("sum_x00", &aggregate_schema).unwrap(), ArrowSortOptions::new(true, false)),
        PhysicalSortExpr::new(col("key", &aggregate_schema).unwrap(), ArrowSortOptions::new(false, false)),
    ])
    .unwrap();
    Arc::new(SortExec::new(sort_exprs, aggregate))
}

fn write_numeric_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

fn write_float_measure_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

fn write_row_group_pruning_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 2, 1, 2])),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 10, 11, 20, 21])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_parquet_file_with_string_column(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

fn write_batch(path: &std::path::Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

fn parquet_row_group_count(path: &std::path::Path) -> usize {
    SerializedFileReader::try_from(path).unwrap().metadata().num_row_groups()
}

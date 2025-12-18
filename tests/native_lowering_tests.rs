use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int32Array, Int64Array, StringArray},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf_query_engine::{Expr, LiteralValue, PlanNode, ResultSink, SourceNode};
use datafusion::{
    common::ScalarValue,
    datasource::{
        physical_plan::{
            FileGroup, FileScanConfigBuilder, ParquetSource,
            parquet::{ParquetAccessPlan, RowGroupAccess},
        },
        source::DataSourceExec,
        table_schema::TableSchema,
    },
    execution::object_store::ObjectStoreUrl,
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    parquet::arrow::{
        ArrowWriter,
        arrow_reader::{RowSelection, RowSelector},
    },
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{binary, col, lit},
    },
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{CandidateOutcome, FallbackReason, lower_physical_plan_to_native};

#[test]
fn benchmark_shape_lowers_to_one_native_query_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
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

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
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
fn pushed_down_parquet_predicate_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-pushed-predicate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_pushed_down_predicate(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

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
fn ranged_parquet_file_is_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-ranged-file");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_byte_range(&path);
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

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn parquet_partition_columns_are_rejected_with_source_reason() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-partition-columns");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_partition_columns(&path);
    let plan = aggregate_sort_limit_plan(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert!(result.plans.is_empty());
    assert_eq!(result.rejected_count(), 1);
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::NativeSourcePartitionColumns, .. }
    ));
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(
        result.source_diagnostics[0].rejected_features.iter().any(|feature| feature.as_str() == "partition_columns")
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn every_rejected_native_candidate_has_stable_rejection_code() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-rejection-codes");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let rejected_results = [
        lower_physical_plan_to_native(aggregate_sort_limit_plan(parquet_scan_with_byte_range(&path), Operator::Plus)),
        lower_physical_plan_to_native(aggregate_sort_limit_plan(
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

fn parquet_scan_with_pushed_down_predicate(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let scan_schema = scan.schema();
    let predicate =
        binary(col("payload", &scan_schema).unwrap(), Operator::GtEq, lit(ScalarValue::Int32(Some(10))), &scan_schema)
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
                    .cloned()
                    .map(|file| {
                        let size = file.object_meta.size as i64;
                        file.with_range(0, size)
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
    let scan = parquet_scan(path);
    let data_source = scan.as_any().downcast_ref::<DataSourceExec>().expect("parquet scan should be DataSourceExec");
    let (file_scan, _parquet_source) =
        data_source.downcast_to_file_source::<ParquetSource>().expect("parquet scan should use ParquetSource");
    let file_column_indices = (0..file_scan.file_schema().fields().len()).collect::<Vec<_>>();
    let table_schema = TableSchema::new(
        Arc::clone(file_scan.file_schema()),
        vec![Arc::new(Field::new("part", DataType::Int32, false))],
    );
    let parquet_source = ParquetSource::new(table_schema);
    let file_scan = FileScanConfigBuilder::new(file_scan.object_store_url.clone(), Arc::new(parquet_source))
        .with_file_groups(file_scan.file_groups.clone())
        .with_projection_indices(Some(file_column_indices))
        .unwrap()
        .build();
    DataSourceExec::from_data_source(file_scan)
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

fn aggregate_sort_limit_plan(scan: Arc<dyn ExecutionPlan>, projection_operator: Operator) -> Arc<dyn ExecutionPlan> {
    let sort = aggregate_sort_plan(scan, projection_operator);
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

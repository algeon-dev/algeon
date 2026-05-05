use std::{
    fs::{self, File},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{Int32Array, Int64Array},
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf_query_engine::{LocalParquetSource, MemoryControlPolicy, QueryPlan, QueryPlanBuilder, SourceNode};
use datafusion::{
    common::{ScalarValue, config::ConfigOptions},
    functions_aggregate::sum::sum_udaf,
    logical_expr::Operator,
    parquet::arrow::ArrowWriter,
    physical_expr::{
        LexOrdering, PhysicalSortExpr,
        aggregate::AggregateExprBuilder,
        expressions::{binary, col, lit},
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        displayable,
        execution_plan::Boundedness,
        filter::FilterExec,
        limit::LocalLimitExec,
        projection::{ProjectionExec, ProjectionExpr},
        sorts::sort::SortExec,
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_cudf::{
    CandidateOutcome, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, ErrorCode, FallbackReason,
    GpuFallbackPolicy, NativeCudfExec, NativeCudfExecutionConfig, cudf_physical_optimizer_rules,
};

#[test]
fn native_exec_constructs_host_arrow_leaf_with_conservative_properties() {
    let native_plan = host_sink_plan();
    let schema = Arc::clone(native_plan.output_schema());
    let exec = NativeCudfExec::try_new(native_plan, bounded_config()).unwrap();

    assert_eq!(exec.name(), "NativeCudfExec");
    assert!(exec.children().is_empty());
    assert_eq!(exec.schema(), schema);
    assert_eq!(exec.properties().output_partitioning().partition_count(), 1);
    assert_eq!(exec.properties().boundedness, Boundedness::Bounded);
    assert!(exec.properties().output_ordering().is_none());

    let display = displayable(&exec).one_line().to_string();
    assert!(display.contains("NativeCudfExec"));
    assert!(display.contains("sink=host_arrow"));
    assert!(display.contains("memory_control_policy=bounded"));

    let tree = displayable(&exec).indent(false).to_string();
    assert!(tree.contains("native_nodes=1"));
    assert!(tree.contains("output_boundary=device_to_host"));
}

#[test]
fn native_exec_rejects_gpu_resident_sink_for_datafusion_output() {
    let error = NativeCudfExec::try_new(gpu_resident_plan(), bounded_config()).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("result_sink"), Some("gpu_resident"));
}

#[test]
fn native_exec_rejects_unbounded_memory_control_policy() {
    let config = bounded_config().with_memory_control_policy(MemoryControlPolicy::Unbounded);
    let error = NativeCudfExec::try_new(host_sink_plan(), config).unwrap_err();

    assert_eq!(error.code(), ErrorCode::InvalidExecutionPlan);
    assert_eq!(error.fact_value("memory_control_policy"), Some("unbounded"));
}

#[test]
fn native_optimizer_rule_rewrites_supported_physical_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-optimizer");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::Plus);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_preserves_rejected_plan_for_downstream_fallback() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-optimizer-rejected");
    let path = dir.join("input.parquet");
    write_parquet_file(&path);
    let plan = aggregate_sort_limit_plan(parquet_scan(&path), Operator::BitwiseAnd);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<LocalLimitExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 0);
    assert_eq!(report.rejected_count(), 1);
    assert!(matches!(
        report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedExpression, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_is_inserted_before_gpu_island_rule_when_enabled() {
    let rules = cudf_physical_optimizer_rules(native_optimizer_config());
    let names = rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();

    let count_only_position = names.iter().position(|name| *name == "datafusion_cudf_native_count_only").unwrap();
    let aggregate_statistics_position = names.iter().position(|name| *name == "aggregate_statistics").unwrap();
    let native_position = names.iter().position(|name| *name == "datafusion_cudf_native").unwrap();
    let gpu_position = names.iter().position(|name| *name == "datafusion_cudf_gpu_island").unwrap();
    let sanity_position = names.iter().position(|name| *name == "SanityCheckPlan").unwrap();
    assert!(count_only_position < aggregate_statistics_position);
    assert!(aggregate_statistics_position < native_position);
    assert!(native_position < gpu_position);
    assert!(gpu_position < sanity_position);

    let default_rules = cudf_physical_optimizer_rules(CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred));
    let default_names = default_rules.iter().map(|rule| rule.name()).collect::<Vec<_>>();
    assert!(!default_names.contains(&"datafusion_cudf_native"));
    assert!(!default_names.contains(&"datafusion_cudf_native_count_only"));
}

fn bounded_config() -> NativeCudfExecutionConfig {
    NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 2, 3, 1)
}

fn native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(bounded_config())
}

fn host_sink_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_with_host_sink(source)
}

fn gpu_resident_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_parquet_source()));
    builder.finish_gpu_resident(source)
}

fn local_parquet_source() -> LocalParquetSource {
    LocalParquetSource::new("/tmp/datafusion-cudf-native-exec-test.parquet", schema(), vec!["key".to_string()])
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]))
}

fn parquet_scan(path: &std::path::Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().with_repartition_file_scans(false).with_target_partitions(1),
    );
    let df =
        futures::executor::block_on(ctx.read_parquet(path.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    futures::executor::block_on(df.create_physical_plan()).unwrap()
}

fn aggregate_sort_limit_plan(scan: Arc<dyn ExecutionPlan>, projection_operator: Operator) -> Arc<dyn ExecutionPlan> {
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
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate));
    Arc::new(LocalLimitExec::new(sort, 2))
}

fn write_parquet_file(path: &std::path::Path) {
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

use std::{
    collections::BTreeSet,
    fs,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    compute::SortOptions as ArrowSortOptions,
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
};
use datafusion::{
    common::{JoinSide, JoinType, NullEquality, ScalarValue},
    config::ConfigOptions,
    datasource::{
        physical_plan::{
            FileGroup, FileScanConfigBuilder, FileSource, ParquetSource,
            parquet::{ParquetAccessPlan, RowGroupAccess},
        },
        source::DataSourceExec,
        table_schema::TableSchema,
    },
    execution::object_store::ObjectStoreUrl,
    functions::{datetime, string, unicode},
    functions_aggregate::{count::count_udaf, first_last::first_value_udaf, min_max::max_udaf, sum::sum_udaf},
    functions_window::row_number::row_number_udwf,
    logical_expr::{Operator, WindowFrame},
    parquet::{
        arrow::arrow_reader::{RowSelection, RowSelector},
        file::reader::{FileReader, SerializedFileReader},
    },
    physical_expr::{
        LexOrdering, PhysicalExpr, PhysicalSortExpr, ScalarFunctionExpr,
        aggregate::{AggregateExprBuilder, AggregateFunctionExpr},
        expressions::{
            CaseExpr, CastExpr, DynamicFilterPhysicalExpr, IsNullExpr, LikeExpr, NotExpr, TryCastExpr, binary, col,
            in_list, is_null, lit,
        },
        window::{StandardWindowExpr, WindowExpr},
    },
    physical_plan::{
        ExecutionPlan, InputOrderMode,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        filter::FilterExec,
        joins::{
            HashJoinExec, NestedLoopJoinExec, PartitionMode,
            utils::{ColumnIndex, JoinFilter},
        },
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::{ProjectionExec, ProjectionExpr},
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
        windows::{BoundedWindowAggExec, create_udwf_window_expr},
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    native::{NativeLoweringResult, lower_physical_plan_to_native},
    native_report::{
        NativeFallbackPressureOutcome, NativeFallbackPressureReport, NativeFallbackPressureWorkloadInput,
        NativeFallbackPressureWorkloadItem,
    },
    policy::FallbackReason,
    report::CandidateOutcome,
};
use datafusion_nexus_workloads::tpch::write_tiny_tpch_parquet_dataset;
use nexus_query_engine::{
    expr::{AggregateFunction, Expr, LiteralValue},
    plan::{
        JoinNullEquality as NativeJoinNullEquality, JoinOutputColumn as NativeJoinOutputColumn,
        JoinSide as NativeJoinSide, JoinType as NativeJoinType, NodeId, PlanNode, QueryPlan, ResultSink,
    },
    source::{SourceColumnRef, SourceNode, ZERO_COLUMN_COUNT_COLUMN},
};

#[path = "support/analytical_olap_workload.rs"]
mod analytical_olap_workload;
#[path = "support/tpch_fallback_workload.rs"]
mod tpch_fallback_workload;
#[path = "support/tpch_session.rs"]
mod tpch_session;

use analytical_olap_workload::*;
use tpch_fallback_workload::tpch_fallback_pressure_workload_items;
use tpch_session::tpch_report_session_config;

#[path = "native_lowering_tests/analytical_olap.rs"]
mod analytical_olap;
#[path = "native_lowering_tests/fixtures.rs"]
mod fixtures;
#[path = "native_lowering_tests/full_gpu_gap_analysis.rs"]
mod full_gpu_gap_analysis;
#[path = "native_lowering_tests/join_plan_builders.rs"]
mod join_plan_builders;
#[path = "native_lowering_tests/join_shapes.rs"]
mod join_shapes;
#[path = "native_lowering_tests/semijoin_reduction.rs"]
mod semijoin_reduction;
#[path = "native_lowering_tests/source_contracts.rs"]
mod source_contracts;
#[path = "native_lowering_tests/source_fixtures.rs"]
mod source_fixtures;
#[path = "native_lowering_tests/sql.rs"]
mod sql;
#[path = "native_lowering_tests/tpch_shapes.rs"]
mod tpch_shapes;

#[path = "native_lowering_tests/aggregate_plan_builders.rs"]
mod aggregate_plan_builders;
#[path = "native_lowering_tests/sql_aggregate_shapes.rs"]
mod sql_aggregate_shapes;

#[path = "native_lowering_tests/core_shapes.rs"]
mod core_shapes;
#[path = "native_lowering_tests/expression_validation.rs"]
mod expression_validation;
#[path = "native_lowering_tests/source_column_pruning.rs"]
mod source_column_pruning;

use aggregate_plan_builders::*;
use fixtures::*;
use join_plan_builders::*;
use source_fixtures::*;
use sql::*;

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

fn aggregate_expr_on_column_with_alias(
    udaf: Arc<datafusion::logical_expr::AggregateUDF>,
    schema: Arc<Schema>,
    column_index: usize,
    alias: &str,
    distinct: bool,
    order_by: Vec<PhysicalSortExpr>,
) -> Arc<AggregateFunctionExpr> {
    let column_name = schema.field(column_index).name();
    Arc::new(
        AggregateExprBuilder::new(udaf, vec![col(column_name, &schema).unwrap()])
            .schema(schema)
            .order_by(order_by)
            .with_distinct(distinct)
            .alias(alias)
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

fn native_local_parquet_source(
    plan: &nexus_query_engine::plan::QueryPlan,
) -> &nexus_query_engine::source::LocalParquetSource {
    native_local_parquet_sources(plan).into_iter().next().expect("native plan should contain a local parquet source")
}

fn native_local_parquet_sources(
    plan: &nexus_query_engine::plan::QueryPlan,
) -> Vec<&nexus_query_engine::source::LocalParquetSource> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source_node) => match source_node.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                SourceNode::GpuResident(_) => None,
                SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .collect()
}

fn native_literal_batch_sources(
    plan: &nexus_query_engine::plan::QueryPlan,
) -> Vec<&nexus_query_engine::source::LiteralBatchSource> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source_node) => match source_node.as_ref() {
                SourceNode::LocalParquet(_) | SourceNode::GpuResident(_) => None,
                SourceNode::LiteralBatch(source) => Some(source),
            },
            _ => None,
        })
        .collect()
}

fn local_parquet_source_table(source: &nexus_query_engine::source::LocalParquetSource) -> Option<&str> {
    source.files.first()?.path.file_stem()?.to_str()
}

fn native_aggregate_functions(plan: &nexus_query_engine::plan::QueryPlan) -> Vec<AggregateFunction> {
    native_aggregate_exprs(plan).iter().map(|expr| expr.function).collect()
}

fn native_aggregate_exprs(plan: &nexus_query_engine::plan::QueryPlan) -> &[nexus_query_engine::expr::AggregateExpr] {
    native_aggregate_nodes(plan)
        .into_iter()
        .next()
        .map(|aggregate| aggregate.spec.aggr_exprs.as_slice())
        .expect("native plan should contain an aggregate node")
}

fn native_aggregate_nodes(plan: &nexus_query_engine::plan::QueryPlan) -> Vec<&nexus_query_engine::plan::AggregateNode> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect()
}

fn native_aggregate_group_exprs(plan: &nexus_query_engine::plan::QueryPlan) -> &[Expr] {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate.spec.group_exprs.as_slice()),
            _ => None,
        })
        .expect("native plan should contain an aggregate node")
}

fn native_projection_exprs(plan: &nexus_query_engine::plan::QueryPlan) -> Vec<&Expr> {
    plan.nodes()
        .iter()
        .flat_map(|node| match &node.kind {
            PlanNode::Projection(projection) => projection.expressions().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn native_filter_expr(plan: &nexus_query_engine::plan::QueryPlan) -> &Expr {
    native_filter_exprs(plan).into_iter().next().expect("native plan should contain a filter node")
}

fn native_filter_exprs(plan: &nexus_query_engine::plan::QueryPlan) -> Vec<&Expr> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Filter(filter) => Some(&filter.predicate),
            _ => None,
        })
        .collect()
}

fn native_joins(plan: &nexus_query_engine::plan::QueryPlan) -> Vec<&nexus_query_engine::plan::JoinNode> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .collect()
}

fn rewritten_native_join(result: &NativeLoweringResult) -> &nexus_query_engine::plan::JoinNode {
    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node")
}

fn assert_right_hash_join_contract(
    join: &nexus_query_engine::plan::JoinNode,
    expected_join_type: NativeJoinType,
    expected_build_side: NativeJoinSide,
    expected_projection: &[(NativeJoinSide, &str, &str)],
) {
    assert_eq!(join.spec.join_type, expected_join_type);
    assert_eq!(join.spec.null_equality, NativeJoinNullEquality::NullsDoNotMatch);
    assert_eq!(join.spec.build_side, expected_build_side);
    assert_eq!(join.spec.keys.len(), 1);
    assert_eq!(join.spec.keys[0].left.to_string(), "input0.key");
    assert_eq!(join.spec.keys[0].right.to_string(), "input1.key");
    assert!(join.spec.residual_predicate.is_none());
    assert_join_output_projection(&join.spec.output_projection, expected_projection);
}

fn assert_join_output_projection(
    actual_projection: &[NativeJoinOutputColumn],
    expected_projection: &[(NativeJoinSide, &str, &str)],
) {
    let actual = actual_projection
        .iter()
        .map(|output| (output.side, output.column.to_string(), output.output_name.as_str()))
        .collect::<Vec<_>>();
    let expected = expected_projection
        .iter()
        .map(|(side, column, output_name)| (*side, (*column).to_string(), *output_name))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

fn projection_source_for_alias<'a>(expressions: &'a [String], alias: &str) -> &'a str {
    expressions
        .iter()
        .find_map(|expr| expr.strip_suffix(format!(" AS {alias}").as_str()))
        .unwrap_or_else(|| panic!("projection should contain alias {alias}: {expressions:?}"))
}

fn assert_join_input_materializes_decimal_cast(
    plan: &nexus_query_engine::plan::QueryPlan,
    input: nexus_query_engine::plan::NodeId,
    alias: &str,
    source_column: &str,
) {
    let input_node = plan.node(input).expect("join input should exist");
    let PlanNode::Projection(projection) = &input_node.kind else {
        panic!("join input should be a residual materialization projection");
    };
    assert!(input_node.output_schema().field_with_name(alias).is_ok());
    assert!(
        projection.expressions().any(|expr| {
            matches!(
                expr,
                Expr::Alias { expr, alias: actual_alias }
                    if actual_alias == alias
                        && matches!(
                            expr.as_ref(),
                            Expr::Cast { expr, target_type }
                                if matches!(target_type, DataType::Decimal128(10, 0))
                                    && matches!(
                                        expr.as_ref(),
                                        Expr::Column(column) if column.name() == source_column
                                    )
                        )
            )
        }),
        "join input projection should materialize {source_column} decimal cast as {alias}: {:?}",
        projection.expressions().collect::<Vec<_>>()
    );
}

fn assert_join_input_materializes_substring(
    plan: &nexus_query_engine::plan::QueryPlan,
    input: nexus_query_engine::plan::NodeId,
    alias: &str,
    source_column: &str,
) {
    let input_node = plan.node(input).expect("join input should exist");
    let PlanNode::Projection(projection) = &input_node.kind else {
        panic!("join input should be a residual materialization projection");
    };
    assert!(input_node.output_schema().field_with_name(alias).is_ok());
    assert!(
        projection.expressions().any(|expr| {
            matches!(
                expr,
                Expr::Alias { expr, alias: actual_alias }
                    if actual_alias == alias
                        && matches!(
                            expr.as_ref(),
                            Expr::Substring { expr, start: 1, length: Some(5) }
                                if matches!(expr.as_ref(), Expr::Column(column) if column.name() == source_column)
                        )
            )
        }),
        "join input projection should materialize substr({source_column}, 1, 5) as {alias}: {:?}",
        projection.expressions().collect::<Vec<_>>()
    );
}

fn assert_join_input_materializes_case_decimal_divide(
    plan: &nexus_query_engine::plan::QueryPlan,
    input: nexus_query_engine::plan::NodeId,
    alias: &str,
    source_column: &str,
) {
    let input_node = plan.node(input).expect("join input should exist");
    let PlanNode::Projection(projection) = &input_node.kind else {
        panic!("join input should be a residual materialization projection");
    };
    assert!(input_node.output_schema().field_with_name(alias).is_ok());
    assert!(
        projection.expressions().any(|expr| {
            matches!(
                expr,
                Expr::Alias { expr, alias: actual_alias }
                    if actual_alias == alias
                        && matches!(
                            expr.as_ref(),
                            Expr::Case { branches, .. }
                                if branches.len() == 1
                                    && matches!(
                                        &branches[0].then,
                                        Expr::Binary {
                                            op: nexus_query_engine::expr::BinaryOp::Divide,
                                            output_type: Some(DataType::Decimal128(_, _)),
                                            left,
                                            right,
                                        } if matches!(left.as_ref(), Expr::Column(column) if column.name() == source_column)
                                            && matches!(right.as_ref(), Expr::Column(column) if column.name() == source_column)
                                    )
                        )
            )
        }),
        "join input projection should materialize CASE decimal divide over {source_column} as {alias}: {:?}",
        projection.expressions().collect::<Vec<_>>()
    );
}

fn assert_join_input_materializes_upper_utf8view_cast(
    plan: &nexus_query_engine::plan::QueryPlan,
    input: nexus_query_engine::plan::NodeId,
    alias: &str,
    source_column: &str,
) {
    let input_node = plan.node(input).expect("join input should exist");
    let PlanNode::Projection(projection) = &input_node.kind else {
        panic!("join input should be a residual materialization projection");
    };
    assert!(input_node.output_schema().field_with_name(alias).is_ok());
    assert!(
        projection.expressions().any(|expr| {
            matches!(
                expr,
                Expr::Alias { expr, alias: actual_alias }
                    if actual_alias == alias
                        && matches!(
                            expr.as_ref(),
                            Expr::Upper(expr)
                                if matches!(
                                    expr.as_ref(),
                                    Expr::Column(column) if column.name() == source_column
                                )
                        )
            )
        }),
        "join input projection should materialize upper({source_column}) as {alias}: {:?}",
        projection.expressions().collect::<Vec<_>>()
    );
}

fn duplicate_named_root_aggregate_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(col("key", &scan_schema).unwrap(), "dup".to_string())]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &scan_schema).unwrap()])
        .schema(Arc::clone(&scan_schema))
        .alias("dup")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(AggregateMode::Single, group_by, vec![Arc::new(sum)], vec![None], scan, scan_schema)
            .unwrap(),
    );

    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        Arc::new(datafusion::physical_expr::expressions::Column::new("dup", 0)),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(3)));
    Arc::new(LocalLimitExec::new(sort, 3))
}

fn duplicate_group_alias_root_aggregate_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let aggregate = duplicate_group_alias_aggregate(scan);
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        Arc::new(datafusion::physical_expr::expressions::Column::new("dup", 0)),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(3)));
    Arc::new(LocalLimitExec::new(sort, 3))
}

fn aggregate_over_duplicate_group_alias_input_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let inner = duplicate_group_alias_aggregate(scan);
    let inner_schema = inner.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(
        Arc::new(datafusion::physical_expr::expressions::Column::new("dup", 0)),
        "outer_dup".to_string(),
    )]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("sum_m00", &inner_schema).unwrap()])
        .schema(Arc::clone(&inner_schema))
        .alias("outer_sum")
        .build()
        .unwrap();
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(AggregateMode::Single, group_by, vec![Arc::new(sum)], vec![None], inner, inner_schema)
            .unwrap(),
    );

    let aggregate_schema = aggregate.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("outer_dup", &aggregate_schema).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(3)));
    Arc::new(LocalLimitExec::new(sort, 3))
}

fn duplicate_group_alias_aggregate(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let input_schema = input.schema();
    let group_by = PhysicalGroupBy::new_single(vec![
        (col("key", &input_schema).unwrap(), "dup".to_string()),
        (col("payload", &input_schema).unwrap(), "dup".to_string()),
    ]);
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("m00", &input_schema).unwrap()])
        .schema(Arc::clone(&input_schema))
        .alias("sum_m00")
        .build()
        .unwrap();
    Arc::new(
        AggregateExec::try_new(AggregateMode::Single, group_by, vec![Arc::new(sum)], vec![None], input, input_schema)
            .unwrap(),
    )
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

const RUNTIME_POLICY_BACKSTOP_CHILD: &str = "DATAFUSION_NEXUS_RUNTIME_POLICY_BACKSTOP_CHILD";

#[test]
fn cpu_lane_runtime_policy_backstop_child() {
    if std::env::var_os(RUNTIME_POLICY_BACKSTOP_CHILD).is_none() {
        return;
    }

    let error = cudf::Runtime::new().expect_err("CPU-lane policy must reject accidental runtime creation");
    assert_eq!(error.code(), cudf::ErrorCode::RuntimeCreationForbidden);
    assert_eq!(error.kind(), cudf::ErrorKind::ExecutionContext);
    assert_eq!(error.status(), cudf::ErrorStatus::Permanent);
    assert_eq!(error.operation(), "runtime::RuntimeBuilder::build");
    assert_eq!(error.detail_value("policy"), Some("forbid"));
    assert_eq!(error.detail_value("reason"), Some("runtime_creation_forbidden_by_policy"));
    assert!(error.detail_value("remedy").is_some());
}

#[test]
fn cpu_lane_runtime_policy_backstop_returns_typed_error_before_cuda() {
    let output = Command::new(std::env::current_exe().expect("current native_lowering_tests executable must exist"))
        .arg("cpu_lane_runtime_policy_backstop_child")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(RUNTIME_POLICY_BACKSTOP_CHILD, "1")
        .env("CUDF_RUNTIME_POLICY", "forbid")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .expect("runtime-policy backstop child must start");

    assert!(
        output.status.success(),
        "runtime-policy backstop child failed with status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

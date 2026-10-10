use std::{
    collections::BTreeSet,
    fs,
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
    logical_expr::{AggregateUDF, Operator, ScalarUDF, WindowFrame},
    parquet::{
        arrow::arrow_reader::{RowSelection, RowSelector},
        file::reader::{FileReader, SerializedFileReader},
    },
    physical_expr::{
        LexOrdering, PhysicalExpr, PhysicalSortExpr, ScalarFunctionExpr,
        aggregate::{AggregateExprBuilder, AggregateFunctionExpr},
        expressions::{
            CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, IsNullExpr, LikeExpr, NotExpr, TryCastExpr, binary,
            col, in_list, is_null, lit,
        },
        window::{StandardWindowExpr, WindowExpr},
    },
    physical_plan::{
        ChildrenPropertiesMode, ExecutionPlan, InputOrderMode, ReplaceChildrenOptions,
        aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
        filter::{FilterExec, FilterExecBuilder},
        joins::{
            HashJoinExec, NestedLoopJoinExec, PartitionMode,
            utils::{ColumnIndex, JoinFilter},
        },
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::{ProjectionExec, ProjectionExpr},
        repartition::RepartitionExec,
        sorts::{sort::SortExec, sort_preserving_merge::SortPreservingMergeExec},
        union::UnionExec,
        windows::{BoundedWindowAggExec, create_udwf_window_expr},
    },
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use query_planning::{
    expr::{AggregateExpr, AggregateFunction, BinaryOp, Expr, LimitSpec, LiteralValue},
    plan::{
        AggregateNode, JoinNode, JoinNullEquality as NativeJoinNullEquality,
        JoinOutputColumn as NativeJoinOutputColumn, JoinSide as NativeJoinSide, JoinType as NativeJoinType, NodeId,
        PlanNode, QueryPlan, ResultSink,
    },
    source::{LiteralBatchSource, LocalParquetSource, SourceColumnRef, SourceNode, ZERO_COLUMN_COUNT_COLUMN},
};
use workloads::tpch::write_tiny_tpch_parquet_dataset;

use crate::{
    capability::CapabilityReason,
    gpu_coverage::QueryGpuRemedy,
    native::{FooterScope, NativeLoweringResult, lower_physical_plan_to_native},
    native_exec::{AlgeonNativeExec, below_native_admission},
    native_report::{
        CandidateOutcomeKind, NativeFallbackPressureReport, NativeFallbackPressureWorkloadInput,
        NativeFallbackPressureWorkloadItem,
    },
    planner::{AlgeonNativeOptimizerConfig, AlgeonNativePhysicalOptimizerRule},
    report::{CandidateOutcome, PlanningReport},
};

mod advisory;

#[path = "../../tests/support/analytical_olap_workload.rs"]
mod analytical_olap_workload;
#[path = "../../tests/support/tpch_fallback_workload.rs"]
mod tpch_fallback_workload;
#[path = "../../tests/support/tpch_session.rs"]
mod tpch_session;

use self::{
    analytical_olap_workload::*, tpch_fallback_workload::tpch_fallback_pressure_workload_items,
    tpch_session::tpch_report_session_config,
};

mod analytical_olap;
mod demand_plans;
mod fixtures;
mod full_gpu_gap_analysis;
mod join_plan_builders;
mod join_shapes;
mod plan_shape_remedies;
mod semijoin_reduction;
mod source_contracts;
mod source_fixtures;
mod sql;
mod tpch_shapes;
mod union_inputs;
mod window_input_ordering;

mod aggregate_plan_builders;
mod sql_aggregate_shapes;

mod core_shapes;
mod expression_validation;
mod source_column_pruning;

use aggregate_plan_builders::*;
use fixtures::*;
use join_plan_builders::*;
use source_fixtures::*;
use sql::*;

fn aggregate_expr_like(
    existing: &AggregateFunctionExpr,
    udaf: Arc<AggregateUDF>,
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
    udaf: Arc<AggregateUDF>,
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
    udaf: Arc<AggregateUDF>,
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

fn assert_distributed_aggregate_rejection(result: &NativeLoweringResult, expected_reason: CapabilityReason) {
    assert_native_rejection(result, expected_reason);
}

fn assert_native_rejection(result: &NativeLoweringResult, expected_reason: CapabilityReason) {
    assert!(result.plan.is_none());
    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 0);
    assert_eq!(result.not_selected_count(), 1);
    let CandidateOutcome::NotSupported { capability_reason, .. } = result.report.outcomes()[0].clone() else {
        panic!("expected rejected native candidate");
    };
    assert_eq!(capability_reason, expected_reason);
}

/// The remedy that the coverage report attaches to the one rejected candidate
/// of `result`.
fn rejected_candidate_remedy(result: &NativeLoweringResult) -> QueryGpuRemedy {
    let [CandidateOutcome::NotSupported { capability_reason, detail, .. }] = result.report.outcomes() else {
        panic!("expected one rejected native candidate: {:?}", result.report.outcomes());
    };
    capability_reason.remedy(detail)
}

/// Runs the native optimizer rule, which lowers the plan, optimizes the
/// native plan and then enforces its orderings, and returns the rule's report
/// with the native plan it kept.
fn optimize_with_native_rule(plan: Arc<dyn ExecutionPlan>) -> (PlanningReport, Option<QueryPlan>) {
    let rule = AlgeonNativePhysicalOptimizerRule::new(AlgeonNativeOptimizerConfig::default());
    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();
    let report = diagnostics.expect("the native rule reports its candidates").report;
    let native_plan = below_native_admission(&optimized)
        .downcast_ref::<AlgeonNativeExec>()
        .map(|native| native.native_plan().clone());
    (report, native_plan)
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

fn native_local_parquet_source(plan: &QueryPlan) -> &LocalParquetSource {
    native_local_parquet_sources(plan).into_iter().next().expect("native plan should contain a local parquet source")
}

fn native_local_parquet_sources(plan: &QueryPlan) -> Vec<&LocalParquetSource> {
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

fn native_literal_batch_sources(plan: &QueryPlan) -> Vec<&LiteralBatchSource> {
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

fn local_parquet_source_table(source: &LocalParquetSource) -> Option<&str> {
    source.files.first()?.path.file_stem()?.to_str()
}

fn native_aggregate_functions(plan: &QueryPlan) -> Vec<AggregateFunction> {
    native_aggregate_exprs(plan).iter().map(|expr| expr.function).collect()
}

fn native_aggregate_exprs(plan: &QueryPlan) -> &[AggregateExpr] {
    native_aggregate_nodes(plan)
        .into_iter()
        .next()
        .map(|aggregate| aggregate.spec.aggr_exprs.as_slice())
        .expect("native plan should contain an aggregate node")
}

fn native_aggregate_nodes(plan: &QueryPlan) -> Vec<&AggregateNode> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect()
}

fn native_aggregate_group_exprs(plan: &QueryPlan) -> &[Expr] {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate.spec.group_exprs.as_slice()),
            _ => None,
        })
        .expect("native plan should contain an aggregate node")
}

fn native_projection_exprs(plan: &QueryPlan) -> Vec<&Expr> {
    plan.nodes()
        .iter()
        .flat_map(|node| match &node.kind {
            PlanNode::Projection(projection) => projection.expressions().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn native_filter_expr(plan: &QueryPlan) -> &Expr {
    native_filter_exprs(plan).into_iter().next().expect("native plan should contain a filter node")
}

fn native_filter_exprs(plan: &QueryPlan) -> Vec<&Expr> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Filter(filter) => Some(&filter.predicate),
            _ => None,
        })
        .collect()
}

fn native_joins(plan: &QueryPlan) -> Vec<&JoinNode> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .collect()
}

fn rewritten_native_join(result: &NativeLoweringResult) -> &JoinNode {
    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    result
        .selected_plan()
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node")
}

fn assert_right_hash_join_contract(
    join: &JoinNode,
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

fn assert_join_input_materializes_decimal_cast(plan: &QueryPlan, input: NodeId, alias: &str, source_column: &str) {
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

fn assert_join_input_materializes_substring(plan: &QueryPlan, input: NodeId, alias: &str, source_column: &str) {
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

fn assert_join_input_materializes_upper_utf8view_cast(
    plan: &QueryPlan,
    input: NodeId,
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

    let sort_exprs =
        LexOrdering::new([PhysicalSortExpr::new(Arc::new(Column::new("dup", 0)), ArrowSortOptions::new(false, false))])
            .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(3)));
    Arc::new(LocalLimitExec::new(sort, 3))
}

fn duplicate_group_alias_root_aggregate_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let aggregate = duplicate_group_alias_aggregate(scan);
    let sort_exprs =
        LexOrdering::new([PhysicalSortExpr::new(Arc::new(Column::new("dup", 0)), ArrowSortOptions::new(false, false))])
            .unwrap();
    let sort: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, aggregate).with_fetch(Some(3)));
    Arc::new(LocalLimitExec::new(sort, 3))
}

fn aggregate_over_duplicate_group_alias_input_sort_limit_plan(scan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let inner = duplicate_group_alias_aggregate(scan);
    let inner_schema = inner.schema();
    let group_by = PhysicalGroupBy::new_single(vec![(Arc::new(Column::new("dup", 0)), "outer_dup".to_string())]);
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

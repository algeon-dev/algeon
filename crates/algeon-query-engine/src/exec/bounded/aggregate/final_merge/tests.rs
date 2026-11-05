use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use query_planning::{
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, LimitSpec, LiteralValue},
    plan::{PlanNode, ProjectionOutput, QueryPlan, QueryPlanBuilder},
    source::{LocalParquetFile, LocalParquetSource, SourceNode},
};

use super::*;

/// `SELECT tenant, <sort_output> FROM asset GROUP BY tenant ORDER BY <sort_output> DESC LIMIT 3`
/// over aggregate outputs `tenant`, `hi` (a sum) and `n` (a count).
fn grouped_topk_plan(sort_output: ProjectionOutput) -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("tenant", DataType::Utf8, false),
        Field::new("hi", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("tenant", DataType::Utf8, false),
        Field::new("hi_sum", DataType::Int64, true),
        Field::new("n", DataType::Int64, false),
    ]));
    let sort_column = sort_output.field.name().clone();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("asset.parquet")],
        source_schema,
        vec!["tenant".to_string(), "hi".to_string()],
    )));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("tenant")],
                vec![
                    AggregateExpr::new(AggregateFunction::Sum, Expr::column("hi"), "hi_sum"),
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("tenant"), "n"),
                ],
            ),
            aggregate_schema,
        )
        .unwrap();
    let projection = builder
        .add_projection(
            aggregate,
            vec![
                ProjectionOutput::new(Expr::column("tenant"), Field::new("tenant", DataType::Utf8, false)),
                sort_output,
            ],
        )
        .unwrap();
    let sort = builder
        .add_sort(projection, SortSpec::new(vec![SortExpr::new(Expr::column(sort_column), true, false)]))
        .unwrap();
    let limit = builder.add_limit(sort, LimitSpec::new(0, 3)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

/// The sort columns of the per-partition top-k the merge chooses for `plan`, or `None` without one.
fn partition_topk_sort_columns(plan: &QueryPlan) -> Option<Vec<String>> {
    let mut source = None;
    let mut aggregate = None;
    let mut post_aggregate_stages = Vec::new();
    let mut sort = None;
    let mut limit = None;
    for node in plan.nodes() {
        match &node.kind {
            PlanNode::Source(source_node) => match source_node.as_ref() {
                SourceNode::LocalParquet(parquet) => source = Some((node.id, parquet)),
                _ => panic!("fixture reads local Parquet"),
            },
            PlanNode::Aggregate(aggregate_node) => {
                aggregate = Some((node.id, aggregate_node, plan.node(aggregate_node.input).unwrap().output_schema()));
            },
            PlanNode::Projection(projection) => {
                post_aggregate_stages.push(BoundedUnaryStage::Projection { node: projection });
            },
            PlanNode::Sort(sort_node) => sort = Some(sort_node),
            PlanNode::Limit(limit_node) => limit = Some((node.id, limit_node)),
            _ => {},
        }
    }
    let (source_id, source) = source.unwrap();
    let (aggregate_id, aggregate_node, aggregate_input_schema) = aggregate.unwrap();
    let bounded = BoundedPlan {
        source_id,
        source,
        pre_filter_projections: Vec::new(),
        filter: None,
        post_filter_projections: Vec::new(),
        aggregate_id,
        aggregate_input_schema,
        aggregate: (aggregate_node, plan.node(aggregate_id).unwrap().output_schema()),
        post_aggregate_stages,
        sort,
        limit,
    };
    partitioned_final_merge_topk_candidate(&bounded).unwrap().map(|topk| {
        topk.sort
            .spec
            .expressions
            .iter()
            .map(|sort_expr| lower_column_expr(&sort_expr.expr, "final_merge_test").unwrap())
            .collect()
    })
}

#[test]
fn partition_topk_needs_a_sort_key_that_renames_an_aggregate_output() {
    let renamed = grouped_topk_plan(ProjectionOutput::new(
        Expr::column("hi_sum").alias("hi_count"),
        Field::new("hi_count", DataType::Int64, true),
    ));
    assert_eq!(partition_topk_sort_columns(&renamed), Some(vec!["hi_sum".to_string()]));

    // The projection computes `100.0 * hi_sum / n AS pct` after the merge.
    // The merged partitions keep every group, and the sort over the
    // projection selects the rows.
    let ratio = Expr::binary(
        Expr::binary(
            Expr::literal(LiteralValue::Float64(100.0)),
            BinaryOp::Multiply,
            Expr::cast(Expr::column("hi_sum"), DataType::Float64),
        ),
        BinaryOp::Divide,
        Expr::cast(Expr::column("n"), DataType::Float64),
    );
    let computed =
        grouped_topk_plan(ProjectionOutput::new(ratio.alias("pct"), Field::new("pct", DataType::Float64, true)));
    assert_eq!(partition_topk_sort_columns(&computed), None);
}

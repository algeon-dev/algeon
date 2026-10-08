use std::sync::Arc;

use arrow::datatypes::DataType;

use super::*;
use crate::{
    expr::{CaseWhen, GroupingSetsSpec, LiteralValue},
    plan::QueryPlanBuilder,
    prepared::{aggregate_spec_requires_whole_input, avg_final_expr},
    source::{GpuResidentSource, SourceNode},
};

#[test]
fn grouped_distinct_count_with_mergeable_aggregates_splits_into_two_streaming_aggregates() {
    let plan = aggregate_plan(
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![
                AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx"),
                AggregateExpr::new(AggregateFunction::Sum, Expr::column("y"), "sy"),
                AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "cy"),
                AggregateExpr::new(AggregateFunction::Min, Expr::column("y"), "lo"),
                AggregateExpr::new(AggregateFunction::Max, Expr::column("y"), "hi"),
                AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "cf")
                    .with_filter(Some(Expr::column("f"))),
            ],
        ),
        vec![
            Field::new("k", DataType::Int32, true),
            count_field("dx"),
            value_field("sy"),
            count_field("cy"),
            value_field("lo"),
            value_field("hi"),
            count_field("cf"),
        ],
    );

    let result = split(&plan);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.applied_count, 1);
    assert_eq!(result.diagnostics.candidates[0].distinct_column.as_deref(), Some("x"));
    assert_eq!(result.plan.output_schema(), plan.output_schema());
    let (partial, merge) = partial_and_merge(&result.plan, result.plan.root_id());
    assert!(!aggregate_spec_requires_whole_input(&partial.spec));
    assert!(!aggregate_spec_requires_whole_input(&merge.spec));
    assert_eq!(partial.spec.group_exprs, vec![Expr::column("k"), Expr::column("x").alias("x")]);
    assert_eq!(
        partial.spec.aggr_exprs.iter().map(|expr| (expr.function, expr.filter.clone())).collect::<Vec<_>>(),
        vec![
            (AggregateFunction::Sum, None),
            (AggregateFunction::Count, None),
            (AggregateFunction::Min, None),
            (AggregateFunction::Max, None),
            (AggregateFunction::Count, Some(Expr::column("f"))),
        ]
    );
    assert_eq!(merge.spec.group_exprs, vec![Expr::column("k")]);
    assert_eq!(
        merge.spec.aggr_exprs.iter().map(|expr| (expr.function, expr.alias.as_str())).collect::<Vec<_>>(),
        vec![
            (AggregateFunction::Count, "dx"),
            (AggregateFunction::Sum, "sy"),
            (AggregateFunction::Sum, "cy"),
            (AggregateFunction::Min, "lo"),
            (AggregateFunction::Max, "hi"),
            (AggregateFunction::Sum, "cf"),
        ]
    );
    assert_eq!(merge.spec.aggr_exprs[0].expr, Expr::column("x"));
}

#[test]
fn global_distinct_count_split_fills_counts_of_empty_input_with_zero() {
    let plan = aggregate_plan(
        AggregateSpec::new(
            Vec::new(),
            vec![
                AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx"),
                AggregateExpr::new(AggregateFunction::Count, Expr::column("y"), "cy"),
                AggregateExpr::new(AggregateFunction::Sum, Expr::column("y"), "sy"),
            ],
        ),
        vec![count_field("dx"), count_field("cy"), value_field("sy")],
    );

    let result = split(&plan);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.plan.output_schema(), plan.output_schema());
    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("a global split restores counts in a projection: {}", result.plan);
    };
    let (partial, merge) = partial_and_merge(&result.plan, projection.input);
    assert_eq!(partial.spec.group_exprs, vec![Expr::column("x").alias("x")]);
    assert!(merge.spec.group_exprs.is_empty());
    let zero_filled = Expr::case(
        vec![CaseWhen::new(Expr::is_null(Expr::column("cy")), Expr::literal(LiteralValue::Int64(0)))],
        Expr::column("cy"),
    )
    .alias("cy");
    assert_eq!(
        projection.expressions().cloned().collect::<Vec<_>>(),
        vec![Expr::column("dx"), zero_filled, Expr::column("sy")]
    );
}

#[test]
fn grouped_distinct_count_with_avg_carries_sum_and_count_state_to_a_finalizing_projection() {
    let plan = aggregate_plan(
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![
                AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("y"), "ay")
                    .with_filter(Some(Expr::column("f"))),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("d"), "ad"),
            ],
        ),
        vec![
            Field::new("k", DataType::Int32, true),
            count_field("dx"),
            Field::new("ay", DataType::Float64, true),
            Field::new("ad", DataType::Decimal128(13, 6), true),
        ],
    );

    let result = split(&plan);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.plan.output_schema(), plan.output_schema());
    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("a split with AVG finalizes it in a projection: {}", result.plan);
    };
    let (partial, merge) = partial_and_merge(&result.plan, projection.input);
    assert!(!aggregate_spec_requires_whole_input(&partial.spec));
    assert_eq!(
        partial.spec.aggr_exprs.iter().map(|expr| (expr.function, expr.filter.clone())).collect::<Vec<_>>(),
        vec![
            (AggregateFunction::Sum, Some(Expr::column("f"))),
            (AggregateFunction::Count, Some(Expr::column("f"))),
            (AggregateFunction::Sum, None),
            (AggregateFunction::Count, None),
        ]
    );
    let partial_schema = result.plan.node(merge.input).unwrap().output_schema();
    assert_eq!(partial_schema.field(2).data_type(), &DataType::Float64);
    assert_eq!(partial_schema.field(4).data_type(), &DataType::Decimal128(38, 2));
    let states = partial.spec.aggr_exprs.iter().map(|expr| expr.alias.as_str()).collect::<Vec<_>>();
    assert_eq!(
        merge.spec.aggr_exprs.iter().map(|expr| (expr.function, expr.alias.as_str())).collect::<Vec<_>>(),
        vec![
            (AggregateFunction::Count, "dx"),
            (AggregateFunction::Sum, states[0]),
            (AggregateFunction::Sum, states[1]),
            (AggregateFunction::Sum, states[2]),
            (AggregateFunction::Sum, states[3]),
        ]
    );
    let avg = |sum: &str, count: &str, output: DataType, alias: &str| {
        avg_final_expr(Expr::column(sum), Expr::column(count), &output).unwrap().alias(alias)
    };
    assert_eq!(
        projection.expressions().cloned().collect::<Vec<_>>(),
        vec![
            Expr::column("k"),
            Expr::column("dx"),
            avg(states[0], states[1], DataType::Float64, "ay"),
            avg(states[2], states[3], DataType::Decimal128(13, 6), "ad"),
        ]
    );
}

#[test]
fn global_distinct_count_with_avg_merges_state_whose_count_is_null_over_empty_input() {
    let plan = aggregate_plan(
        AggregateSpec::new(
            Vec::new(),
            vec![
                AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx"),
                AggregateExpr::new(AggregateFunction::Avg, Expr::column("d"), "ad"),
            ],
        ),
        vec![count_field("dx"), Field::new("ad", DataType::Decimal128(13, 6), true)],
    );

    let result = split(&plan);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.plan.output_schema(), plan.output_schema());
    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("a split with AVG finalizes it in a projection: {}", result.plan);
    };
    let merge_schema = result.plan.node(projection.input).unwrap().output_schema();
    let count_state = merge_schema.field(2);
    assert!(count_state.is_nullable(), "a global merge over no partial rows has a null count: {count_state:?}");
    assert_eq!(
        projection.expressions().nth(1).cloned(),
        Some(
            avg_final_expr(
                Expr::column(merge_schema.field(1).name().as_str()),
                Expr::column(count_state.name().as_str()),
                &DataType::Decimal128(13, 6),
            )
            .unwrap()
            .alias("ad")
        )
    );
}

#[test]
fn aggregates_without_one_shared_unfiltered_distinct_argument_or_mergeable_state_stay_whole() {
    use SingleDistinctAggregateSplitRejectedReason as Reason;

    let distinct_x = || AggregateExpr::new(AggregateFunction::NUnique, Expr::column("x"), "dx");
    let cases = [
        (
            "two distinct arguments",
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![distinct_x(), AggregateExpr::new(AggregateFunction::NUnique, Expr::column("y"), "dy")],
            ),
            None,
            Reason::MultipleDistinctArguments,
        ),
        (
            "filtered distinct",
            AggregateSpec::new(vec![Expr::column("k")], vec![distinct_x().with_filter(Some(Expr::column("f")))]),
            None,
            Reason::FilteredDistinctAggregate,
        ),
        (
            "median",
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![distinct_x(), AggregateExpr::new(AggregateFunction::Median, Expr::column("y"), "my")],
            ),
            Some("x"),
            Reason::NonDecomposableAggregate,
        ),
        (
            "statistical",
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![distinct_x(), AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("y"), "my")],
            ),
            Some("x"),
            Reason::NonDecomposableAggregate,
        ),
        (
            "avg over a decimal DataFusion sums as Decimal256",
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![distinct_x(), AggregateExpr::new(AggregateFunction::Avg, Expr::column("wide"), "aw")],
            ),
            Some("x"),
            Reason::NonDecomposableAggregate,
        ),
        (
            "grouping sets",
            AggregateSpec::new(vec![Expr::column("k")], vec![distinct_x()])
                .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
            Some("x"),
            Reason::GroupingSets,
        ),
    ];

    for (label, spec, distinct_column, reason) in cases {
        let mut fields = vec![Field::new("k", DataType::Int32, true)];
        if spec.grouping_sets.is_some() {
            fields.push(Field::new("__grouping_id", DataType::UInt8, false));
        }
        fields.extend(spec.aggr_exprs.iter().map(|expr| match expr.function {
            AggregateFunction::Avg => Field::new(expr.alias.as_str(), DataType::Decimal128(34, 6), true),
            _ => count_field(expr.alias.as_str()),
        }));
        let plan = aggregate_plan(spec, fields);

        let result = split(&plan);

        assert_eq!(result.change, NativeOptimizerChange::Unchanged, "{label}");
        assert_eq!(result.plan, plan, "{label}");
        assert_eq!(result.diagnostics.considered_count, 1, "{label}");
        let candidate = &result.diagnostics.candidates[0];
        assert_eq!(candidate.distinct_column.as_deref(), distinct_column, "{label}");
        assert_eq!(candidate.rejected_reason, Some(reason), "{label}");
    }
}

fn split(plan: &QueryPlan) -> SingleDistinctAggregateSplitResult {
    optimize_single_distinct_aggregate_split(plan.clone(), &SingleDistinctAggregateSplitOptions::default()).unwrap()
}

fn aggregate_plan(spec: AggregateSpec, output_fields: Vec<Field>) -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("x", DataType::Int64, true),
        Field::new("y", DataType::Int64, true),
        Field::new("f", DataType::Boolean, true),
        Field::new("d", DataType::Decimal128(9, 2), true),
        Field::new("wide", DataType::Decimal128(30, 2), true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", input_schema)));
    let aggregate = builder.add_aggregate(source, spec, Arc::new(Schema::new(output_fields))).unwrap();
    builder.finish_gpu_resident(aggregate).unwrap()
}

/// The merge aggregate at `merge_id` and the partial aggregate it reads.
fn partial_and_merge(plan: &QueryPlan, merge_id: NodeId) -> (&AggregateNode, &AggregateNode) {
    let aggregate = |id: NodeId| match &plan.node(id).unwrap().kind {
        PlanNode::Aggregate(aggregate) => aggregate,
        _ => panic!("node {id} should be an aggregate: {plan}"),
    };
    let merge = aggregate(merge_id);
    (aggregate(merge.input), merge)
}

fn count_field(name: &str) -> Field {
    Field::new(name, DataType::Int64, false)
}

fn value_field(name: &str) -> Field {
    Field::new(name, DataType::Int64, true)
}

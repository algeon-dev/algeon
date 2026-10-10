use std::sync::Arc;

use query_planning::plan::{NodeId, QueryPlan};

use super::*;

/// Mutates the non-root side of a keyless conditional root join. The admitted
/// shape is deliberately narrow: a projection-wrapped left-anti subplan on one
/// side and a scalar aggregate on the other.
#[derive(Debug, Clone, Copy)]
pub(crate) enum KeylessInputSubplanVariant {
    Admitted,
    /// Replaces the scalar side with a grouped aggregate, which changes the
    /// cardinality contract the keyless residual join relies on.
    GroupedAggregate,
    /// Adds an ordering barrier below the root keyless join; this input shape is
    /// outside the supported keyless join contract.
    SortInput,
    /// Adds a limit barrier below the root keyless join, breaking the bounded
    /// input-subplan shape.
    LimitInput,
    /// Omits the projection wrapper around the left-anti join. The wrapper is
    /// the required boundary around an anti-join input subplan.
    BareLeftAntiInput,
    /// Swaps the left-anti input subplan for an unsupported ordinary join input.
    InnerJoinInput,
    /// Adds a join below the left-anti input, which would make the input subplan
    /// recursively multi-join instead of a bounded anti-join leaf.
    NestedJoinInput,
}

#[path = "../../../support/remaining_topologies.rs"]
mod remaining_topologies;
mod resident_subplan;
pub(crate) use remaining_topologies::*;
pub(crate) use resident_subplan::{
    KeyedHashJoinResidentSubplanPipelineVariant, keyed_hash_join_resident_subplan_pipeline_plan,
};

pub(crate) fn keyless_conditional_join_input_subplan_plan(variant: KeylessInputSubplanVariant) -> QueryPlan {
    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let orders_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let left_subplan_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let grouped_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("threshold", DataType::Int64, false),
    ]));
    let scalar_aggregate_schema = Arc::new(Schema::new(vec![Field::new("threshold", DataType::Int32, true)]));
    let threshold_type =
        if matches!(variant, KeylessInputSubplanVariant::GroupedAggregate) { DataType::Int64 } else { DataType::Int32 };
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("threshold", threshold_type, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let lineitem = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/keyless-input-lineitem.parquet",
        Arc::clone(&lineitem_schema),
        vec!["order_id".into(), "qty".into()],
    )));
    let anti_orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/keyless-input-anti-orders.parquet",
        Arc::clone(&orders_schema),
        vec!["order_id".into()],
    )));
    let threshold_source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/keyless-input-threshold.parquet",
        Arc::clone(&orders_schema),
        vec!["order_id".into()],
    )));

    let nested_join_left = if matches!(variant, KeylessInputSubplanVariant::NestedJoinInput) {
        let extra_orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/keyless-input-extra-orders.parquet",
            Arc::clone(&orders_schema),
            vec!["order_id".into()],
        )));
        builder
            .add_join(
                lineitem,
                extra_orders,
                JoinSpec::new(
                    JoinType::Inner,
                    vec![JoinKey::new(
                        ColumnRef::input(0, "order_id", "left_order_id"),
                        ColumnRef::input(1, "order_id", "right_order_id"),
                    )],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                ])
                .with_build_side(JoinSide::Right),
                Arc::clone(&left_subplan_schema),
            )
            .unwrap()
    } else {
        lineitem
    };

    let anti_join_type = if matches!(variant, KeylessInputSubplanVariant::InnerJoinInput) {
        JoinType::Inner
    } else {
        JoinType::LEFT_ANTI
    };
    let anti_join = builder
        .add_join(
            nested_join_left,
            anti_orders,
            JoinSpec::new(
                anti_join_type,
                vec![JoinKey::new(
                    ColumnRef::input(0, "order_id", "left_order_id"),
                    ColumnRef::input(1, "order_id", "right_order_id"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&left_subplan_schema),
        )
        .unwrap();
    let left_input = if matches!(variant, KeylessInputSubplanVariant::BareLeftAntiInput) {
        anti_join
    } else {
        builder
            .add_projection(
                anti_join,
                projection_outputs(
                    vec![Expr::column("order_id"), Expr::column("qty")],
                    Arc::clone(&left_subplan_schema),
                ),
            )
            .unwrap()
    };

    let right_input = match variant {
        KeylessInputSubplanVariant::GroupedAggregate => builder
            .add_aggregate(
                threshold_source,
                AggregateSpec::new(
                    vec![Expr::column("order_id")],
                    vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_id"), "threshold")],
                ),
                grouped_aggregate_schema,
            )
            .unwrap(),
        KeylessInputSubplanVariant::SortInput => {
            let scalar_aggregate = threshold_scalar_aggregate(&mut builder, threshold_source, &scalar_aggregate_schema);
            builder
                .add_sort(scalar_aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("threshold"), false, false)]))
                .unwrap()
        },
        KeylessInputSubplanVariant::LimitInput => {
            let scalar_aggregate = threshold_scalar_aggregate(&mut builder, threshold_source, &scalar_aggregate_schema);
            builder.add_limit(scalar_aggregate, LimitSpec::fetch(1)).unwrap()
        },
        KeylessInputSubplanVariant::Admitted
        | KeylessInputSubplanVariant::BareLeftAntiInput
        | KeylessInputSubplanVariant::InnerJoinInput
        | KeylessInputSubplanVariant::NestedJoinInput => {
            threshold_scalar_aggregate(&mut builder, threshold_source, &scalar_aggregate_schema)
        },
    };

    let root = builder
        .add_join(
            left_input,
            right_input,
            JoinSpec::new(JoinType::Inner, Vec::new())
                .with_residual_predicate(Expr::binary(
                    Expr::input_column(0, "qty", "qty"),
                    BinaryOp::Gt,
                    Expr::input_column(1, "threshold", "threshold"),
                ))
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "threshold", "threshold"), "threshold"),
                ])
                .with_build_side(JoinSide::Left),
            output_schema,
        )
        .unwrap();
    builder.finish_with_host_sink(root).unwrap()
}

fn threshold_scalar_aggregate(
    builder: &mut QueryPlanBuilder,
    threshold_source: NodeId,
    scalar_aggregate_schema: &Arc<Schema>,
) -> NodeId {
    builder
        .add_aggregate(
            threshold_source,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Max, Expr::column("order_id"), "threshold")],
            ),
            Arc::clone(scalar_aggregate_schema),
        )
        .unwrap()
}

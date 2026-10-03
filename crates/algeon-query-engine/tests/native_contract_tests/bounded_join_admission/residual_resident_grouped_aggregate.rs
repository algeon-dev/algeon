use std::sync::Arc;

use super::*;

#[test]
fn bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_variants_are_admitted() {
    // S3 admits multi-join aggregate pipelines through the compiled task
    // contracts, so these variants are executable.
    for variant in [
        ResidualResidentGroupedAggregateInputVariant::RootJoinWithoutResidual,
        ResidualResidentGroupedAggregateInputVariant::BareGroupedAggregateInput,
        ResidualResidentGroupedAggregateInputVariant::ScalarAggregateInput,
        ResidualResidentGroupedAggregateInputVariant::GroupedAggregateGpuResidentInput,
        ResidualResidentGroupedAggregateInputVariant::BareGlobalAggregateInput,
        ResidualResidentGroupedAggregateInputVariant::FinalSort,
        ResidualResidentGroupedAggregateInputVariant::RootOutputIncludesGroupedAggregate,
    ] {
        let plan = residual_resident_grouped_aggregate_input_plan(variant);

        assert!(
            unsupported_facts(
                &plan,
                StatisticalAggregateExecutionMode::default(),
                rapids_analysis_context(PlanningPolicy::default())
            )
            .is_none(),
            "{variant:?}"
        );
    }
}

/// Mutates a residual join shape used to prove the mixed keyed
/// residual join + resident grouped-aggregate input contract.
#[derive(Debug, Clone, Copy)]
#[expect(dead_code, reason = "shared topology fixtures are compiled by test binaries with different coverage")]
pub(crate) enum ResidualResidentGroupedAggregateInputVariant {
    /// Baseline positive fixture whose surrounding shape is kept out of broader
    /// bounded families, so only this narrow residual-resident evidence admits it.
    BlockedFamily,
    RootJoinWithoutResidual,
    BareGroupedAggregateInput,
    ScalarAggregateInput,
    /// Starts the grouped input from a resident GPU frame; this must not be
    /// confused with the local source-to-aggregate reduction evidence.
    GroupedAggregateGpuResidentInput,
    BareGlobalAggregateInput,
    FinalSort,
    /// Positive fixture: admission is about the resident grouped input and
    /// residual memory contract, not the final aggregate being exactly SUM.
    GlobalAggregateNotSum,
    /// Positive fixture: the grouped input only has to be a reusable grouped
    /// aggregate, not specifically the AVG expression used by the baseline.
    GroupedAggregateNotAvg,
    /// Positive fixtures that keep the residual connected to the grouped input
    /// while varying the scaled-average idiom the optimizer usually emits.
    ResidualWithoutScaledAverage,
    ResidualWrongMultiplier,
    PreprojectedResidualWrongMultiplier,
    RootOutputIncludesGroupedAggregate,
    /// Rejection locality case: the shape is otherwise close, but the residual
    /// contains an expression family the join capability cannot lower.
    UnsupportedResidualExpression,
}

pub(crate) fn residual_resident_grouped_aggregate_input_plan(
    variant: ResidualResidentGroupedAggregateInputVariant,
) -> QueryPlan {
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int32, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_container", DataType::Utf8, false),
    ]));
    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int32, false),
        Field::new("l_quantity", DataType::Float64, false),
        Field::new("l_extendedprice", DataType::Float64, false),
    ]));
    let joined_lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int32, false),
        Field::new("l_quantity", DataType::Float64, false),
        Field::new("l_extendedprice", DataType::Float64, false),
    ]));
    let grouped_quantity_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int32, false),
        Field::new("avg_quantity", DataType::Float64, true),
    ]));
    let grouped_scaled_quantity_schema = Arc::new(Schema::new(vec![
        Field::new("l_partkey", DataType::Int32, false),
        Field::new("scaled_avg_quantity", DataType::Float64, true),
    ]));
    let root_output_schema = Arc::new(Schema::new(vec![Field::new("l_extendedprice", DataType::Float64, false)]));
    let root_output_with_grouped_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("l_extendedprice", DataType::Float64, false),
        Field::new("avg_quantity", DataType::Float64, true),
    ]));
    let global_aggregate_schema = Arc::new(Schema::new(vec![Field::new("sum_extendedprice", DataType::Float64, true)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("avg_yearly", DataType::Float64, true)]));

    let mut builder = QueryPlanBuilder::new();
    let lineitem = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/residual-resident-lineitem.parquet",
        Arc::clone(&lineitem_schema),
        vec!["l_partkey".into(), "l_quantity".into(), "l_extendedprice".into()],
    )));
    let filtered_lineitem = builder
        .add_filter(
            lineitem,
            Expr::binary(Expr::column("l_quantity"), BinaryOp::Gt, Expr::literal(LiteralValue::Float64(0.0))),
        )
        .unwrap();
    let part = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/residual-resident-part.parquet",
        part_schema,
        vec!["p_partkey".into(), "p_brand".into(), "p_container".into()],
    )));
    let part_lineitem = builder
        .add_join(
            filtered_lineitem,
            part,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "l_partkey", "l_partkey"),
                    ColumnRef::input(1, "p_partkey", "p_partkey"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "l_partkey", "l_partkey"), "l_partkey"),
                JoinOutputColumn::left(ColumnRef::input(0, "l_quantity", "l_quantity"), "l_quantity"),
                JoinOutputColumn::left(ColumnRef::input(0, "l_extendedprice", "l_extendedprice"), "l_extendedprice"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&joined_lineitem_schema),
        )
        .unwrap();
    let joined_input = builder
        .add_projection(
            part_lineitem,
            projection_outputs(
                vec![Expr::column("l_partkey"), Expr::column("l_quantity"), Expr::column("l_extendedprice")],
                Arc::clone(&joined_lineitem_schema),
            ),
        )
        .unwrap();
    let joined_input = builder
        .add_projection(
            joined_input,
            projection_outputs(
                vec![
                    Expr::column("l_partkey"),
                    Expr::column("l_quantity"),
                    Expr::column("l_extendedprice"),
                    Expr::cast(Expr::column("l_quantity"), DataType::Float64)
                        .alias("__algeon_native_join_residual_left_0"),
                ],
                Arc::new(Schema::new(vec![
                    Field::new("l_partkey", DataType::Int32, false),
                    Field::new("l_quantity", DataType::Float64, false),
                    Field::new("l_extendedprice", DataType::Float64, false),
                    Field::new("__algeon_native_join_residual_left_0", DataType::Float64, false),
                ])),
            ),
        )
        .unwrap();

    let grouped_quantity_source = match variant {
        ResidualResidentGroupedAggregateInputVariant::GroupedAggregateGpuResidentInput => {
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
                "residual_resident_lineitem_grouped",
                Arc::clone(&lineitem_schema),
            )))
        },
        ResidualResidentGroupedAggregateInputVariant::BlockedFamily
        | ResidualResidentGroupedAggregateInputVariant::RootJoinWithoutResidual
        | ResidualResidentGroupedAggregateInputVariant::BareGroupedAggregateInput
        | ResidualResidentGroupedAggregateInputVariant::ScalarAggregateInput
        | ResidualResidentGroupedAggregateInputVariant::BareGlobalAggregateInput
        | ResidualResidentGroupedAggregateInputVariant::FinalSort
        | ResidualResidentGroupedAggregateInputVariant::GlobalAggregateNotSum
        | ResidualResidentGroupedAggregateInputVariant::GroupedAggregateNotAvg
        | ResidualResidentGroupedAggregateInputVariant::ResidualWithoutScaledAverage
        | ResidualResidentGroupedAggregateInputVariant::ResidualWrongMultiplier
        | ResidualResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier
        | ResidualResidentGroupedAggregateInputVariant::RootOutputIncludesGroupedAggregate
        | ResidualResidentGroupedAggregateInputVariant::UnsupportedResidualExpression => {
            builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
                "/tmp/residual-resident-lineitem-aggregate.parquet",
                Arc::clone(&lineitem_schema),
                vec!["l_partkey".into(), "l_quantity".into()],
            )))
        },
    };
    let quantity_aggregate = AggregateExpr::new(
        if matches!(variant, ResidualResidentGroupedAggregateInputVariant::GroupedAggregateNotAvg) {
            AggregateFunction::Sum
        } else {
            AggregateFunction::Avg
        },
        Expr::column("l_quantity"),
        "avg_quantity",
    );
    // A scalar aggregate has no group key to join on, so it carries the key
    // column as an aggregate output to keep the root join keyed.
    let aggregate_spec = if matches!(variant, ResidualResidentGroupedAggregateInputVariant::ScalarAggregateInput) {
        AggregateSpec::new(
            Vec::new(),
            vec![
                AggregateExpr::new(AggregateFunction::Min, Expr::column("l_partkey"), "l_partkey"),
                quantity_aggregate,
            ],
        )
    } else {
        AggregateSpec::new(vec![Expr::column("l_partkey")], vec![quantity_aggregate])
    };
    let grouped_quantity =
        builder.add_aggregate(grouped_quantity_source, aggregate_spec, Arc::clone(&grouped_quantity_schema)).unwrap();
    let grouped_input = if matches!(variant, ResidualResidentGroupedAggregateInputVariant::BareGroupedAggregateInput) {
        grouped_quantity
    } else if matches!(variant, ResidualResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier) {
        builder
            .add_projection(
                grouped_quantity,
                projection_outputs(
                    vec![
                        Expr::column("l_partkey"),
                        Expr::binary(
                            Expr::column("avg_quantity"),
                            BinaryOp::Multiply,
                            Expr::literal(LiteralValue::Float64(2.0)),
                        )
                        .alias("scaled_avg_quantity"),
                    ],
                    Arc::clone(&grouped_scaled_quantity_schema),
                ),
            )
            .unwrap()
    } else {
        builder
            .add_projection(
                grouped_quantity,
                projection_outputs(
                    vec![Expr::column("l_partkey"), Expr::column("avg_quantity")],
                    Arc::clone(&grouped_quantity_schema),
                ),
            )
            .unwrap()
    };

    let mut root_spec = JoinSpec::new(
        JoinType::Inner,
        vec![JoinKey::new(
            ColumnRef::input(0, "l_partkey", "l_partkey"),
            ColumnRef::input(1, "l_partkey", "l_partkey"),
        )],
    )
    .with_output_projection(vec![JoinOutputColumn::left(
        ColumnRef::input(0, "l_extendedprice", "l_extendedprice"),
        "l_extendedprice",
    )])
    .with_build_side(JoinSide::Right);
    let root_schema =
        if matches!(variant, ResidualResidentGroupedAggregateInputVariant::RootOutputIncludesGroupedAggregate) {
            root_spec = root_spec.with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "l_extendedprice", "l_extendedprice"), "l_extendedprice"),
                JoinOutputColumn::right(ColumnRef::input(1, "avg_quantity", "avg_quantity"), "avg_quantity"),
            ]);
            Arc::clone(&root_output_with_grouped_aggregate_schema)
        } else {
            Arc::clone(&root_output_schema)
        };
    if !matches!(variant, ResidualResidentGroupedAggregateInputVariant::RootJoinWithoutResidual) {
        root_spec = if matches!(variant, ResidualResidentGroupedAggregateInputVariant::UnsupportedResidualExpression) {
            root_spec.with_residual_predicate(Expr::binary(
                Expr::like(Expr::input_column(0, "l_quantity", "l_quantity"), "1%"),
                BinaryOp::And,
                Expr::binary(
                    Expr::binary(
                        Expr::input_column(0, "l_quantity", "l_quantity"),
                        BinaryOp::Lt,
                        Expr::input_column(1, "avg_quantity", "avg_quantity"),
                    ),
                    BinaryOp::Eq,
                    Expr::literal(LiteralValue::Boolean(true)),
                ),
            ))
        } else {
            let residual_right =
                if matches!(variant, ResidualResidentGroupedAggregateInputVariant::ResidualWithoutScaledAverage) {
                    Expr::input_column(1, "avg_quantity", "avg_quantity")
                } else if matches!(
                    variant,
                    ResidualResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier
                ) {
                    Expr::input_column(1, "scaled_avg_quantity", "scaled_avg_quantity")
                } else {
                    let multiplier =
                        if matches!(variant, ResidualResidentGroupedAggregateInputVariant::ResidualWrongMultiplier) {
                            2.0
                        } else {
                            0.2
                        };
                    Expr::binary(
                        Expr::input_column(1, "avg_quantity", "avg_quantity"),
                        BinaryOp::Multiply,
                        Expr::literal(LiteralValue::Float64(multiplier)),
                    )
                };
            root_spec.with_residual_predicate(Expr::binary(
                Expr::input_column(0, "l_quantity", "l_quantity"),
                BinaryOp::Lt,
                residual_right,
            ))
        };
    }
    let root = builder.add_join(joined_input, grouped_input, root_spec, root_schema).unwrap();
    let (aggregate_input, aggregate_arg) =
        if matches!(variant, ResidualResidentGroupedAggregateInputVariant::BareGlobalAggregateInput) {
            (root, "l_extendedprice")
        } else {
            let aggregate_arg_schema =
                Arc::new(Schema::new(vec![Field::new("__algeon_native_aggregate_arg_0", DataType::Float64, false)]));
            let projection = builder
                .add_projection(
                    root,
                    projection_outputs(
                        vec![Expr::column("l_extendedprice").alias("__algeon_native_aggregate_arg_0")],
                        aggregate_arg_schema,
                    ),
                )
                .unwrap();
            (projection, "__algeon_native_aggregate_arg_0")
        };
    let global = builder
        .add_aggregate(
            aggregate_input,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    if matches!(variant, ResidualResidentGroupedAggregateInputVariant::GlobalAggregateNotSum) {
                        AggregateFunction::Count
                    } else {
                        AggregateFunction::Sum
                    },
                    Expr::column(aggregate_arg),
                    "sum_extendedprice",
                )],
            ),
            Arc::clone(&global_aggregate_schema),
        )
        .unwrap();
    let projected = builder
        .add_projection(
            global,
            projection_outputs(
                vec![
                    Expr::binary(
                        Expr::column("sum_extendedprice"),
                        BinaryOp::Divide,
                        Expr::literal(LiteralValue::Float64(7.0)),
                    )
                    .alias("avg_yearly"),
                ],
                Arc::clone(&output_schema),
            ),
        )
        .unwrap();
    let output = if matches!(variant, ResidualResidentGroupedAggregateInputVariant::FinalSort) {
        builder
            .add_sort(projected, SortSpec::new(vec![SortExpr::new(Expr::column("avg_yearly"), false, false)]))
            .unwrap()
    } else {
        projected
    };
    builder.finish_with_host_sink(output).unwrap()
}

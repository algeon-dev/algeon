use super::*;

#[test]
fn bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_is_executable() {
    for variant in [
        Q17ResidentGroupedAggregateInputVariant::BlockedFamily,
        Q17ResidentGroupedAggregateInputVariant::GlobalAggregateNotSum,
        Q17ResidentGroupedAggregateInputVariant::GroupedAggregateNotAvg,
        Q17ResidentGroupedAggregateInputVariant::ResidualWithoutScaledAverage,
        Q17ResidentGroupedAggregateInputVariant::ResidualWrongMultiplier,
        Q17ResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier,
    ] {
        let plan = q17_shaped_resident_grouped_aggregate_input_plan(variant);

        assert!(!plan.has_bounded_executable_root_join(), "{variant:?}");
        assert!(!plan.has_bounded_executable_join_pipeline(), "{variant:?}");
        assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline(), "{variant:?}");
        assert!(
            !plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(!plan.has_bounded_executable_multi_join_pipeline(), "{variant:?}");
        assert!(
            plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(),
            "{variant:?}"
        );
        assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none(), "{variant:?}");
        let facts = plan
            .bounded_join_modeled_facts(ExecutionBoundaryPolicy::default())
            .expect("executable q17-style family should emit modeled facts");
        assert_eq!(
            facts.get("bounded_join_execution_family").map(String::as_str),
            Some("mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline"),
            "{variant:?}"
        );
        assert_eq!(
            facts.get("bounded_join_contract_modes").map(String::as_str),
            Some("source_contract:hard_budgeted_report,aggregate_contract:hard_budgeted_report"),
            "{variant:?}"
        );
        assert_eq!(
            facts.get("bounded_join_unresolved_preflight_contracts").map(String::as_str),
            Some(
                "source_contract:libcudf_parquet_decoded_output_preflight_unavailable,aggregate_contract:libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable"
            ),
            "{variant:?}"
        );
        assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""), "{variant:?}");
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""), "{variant:?}");
        assert_eq!(
            facts.get("modeled_reservations").map(String::as_str),
            Some(
                "source_read,operator_output,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization"
            ),
            "{variant:?}"
        );
        assert!(
            !facts.get("bounded_join_contract_modes").is_some_and(|value| value.contains("final_sort_contract")),
            "{variant:?}"
        );
    }
}

#[test]
fn bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_recognizer_stays_narrow() {
    for variant in [
        Q17ResidentGroupedAggregateInputVariant::RootJoinWithoutResidual,
        Q17ResidentGroupedAggregateInputVariant::BareGroupedAggregateInput,
        Q17ResidentGroupedAggregateInputVariant::ScalarAggregateInput,
        Q17ResidentGroupedAggregateInputVariant::GroupedAggregateGpuResidentInput,
        Q17ResidentGroupedAggregateInputVariant::BareGlobalAggregateInput,
        Q17ResidentGroupedAggregateInputVariant::FinalSort,
        Q17ResidentGroupedAggregateInputVariant::RootOutputIncludesGroupedAggregate,
    ] {
        let plan = q17_shaped_resident_grouped_aggregate_input_plan(variant);

        assert!(!plan.has_bounded_executable_root_join(), "{variant:?}");
        assert!(!plan.has_bounded_executable_join_pipeline(), "{variant:?}");
        assert!(!plan.has_bounded_executable_keyless_conditional_join_input_subplan_pipeline(), "{variant:?}");
        assert!(
            !plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(!plan.has_bounded_executable_multi_join_pipeline(), "{variant:?}");
        assert!(
            !plan.has_bounded_executable_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(),
            "{variant:?}"
        );
        let facts = plan
            .bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default())
            .expect("unsupported resident grouped aggregate input variant should produce facts");
        assert_ne!(
            facts.get("bounded_join_blocked_join_family").map(String::as_str),
            Some("mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline"),
            "{variant:?} must not be classified as the blocked Q17-shaped family"
        );
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

#[derive(Debug, Clone, Copy)]
enum Q17ResidentGroupedAggregateInputVariant {
    BlockedFamily,
    RootJoinWithoutResidual,
    BareGroupedAggregateInput,
    ScalarAggregateInput,
    GroupedAggregateGpuResidentInput,
    BareGlobalAggregateInput,
    FinalSort,
    GlobalAggregateNotSum,
    GroupedAggregateNotAvg,
    ResidualWithoutScaledAverage,
    ResidualWrongMultiplier,
    PreprojectedResidualWrongMultiplier,
    RootOutputIncludesGroupedAggregate,
}

fn q17_shaped_resident_grouped_aggregate_input_plan(
    variant: Q17ResidentGroupedAggregateInputVariant,
) -> cudf_query_engine::QueryPlan {
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
        "/tmp/q17-lineitem.parquet",
        Arc::clone(&lineitem_schema),
        vec!["l_partkey".into(), "l_quantity".into(), "l_extendedprice".into()],
    )));
    let filtered_lineitem = builder.add_filter(
        lineitem,
        Expr::binary(Expr::column("l_quantity"), BinaryOp::Gt, Expr::literal(LiteralValue::Float64(0.0))),
        Arc::clone(&lineitem_schema),
    );
    let part = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q17-part.parquet",
        part_schema,
        vec!["p_partkey".into(), "p_brand".into(), "p_container".into()],
    )));
    let part_lineitem = builder.add_join(
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
    );
    let joined_input = builder.add_projection(
        part_lineitem,
        projection_outputs(
            vec![Expr::column("l_partkey"), Expr::column("l_quantity"), Expr::column("l_extendedprice")],
            Arc::clone(&joined_lineitem_schema),
        ),
    );
    let joined_input = builder.add_projection(
        joined_input,
        projection_outputs(
            vec![
                Expr::column("l_partkey"),
                Expr::column("l_quantity"),
                Expr::column("l_extendedprice"),
                Expr::cast(Expr::column("l_quantity"), DataType::Float64).alias("__cudf_join_residual_left_0"),
            ],
            Arc::new(Schema::new(vec![
                Field::new("l_partkey", DataType::Int32, false),
                Field::new("l_quantity", DataType::Float64, false),
                Field::new("l_extendedprice", DataType::Float64, false),
                Field::new("__cudf_join_residual_left_0", DataType::Float64, false),
            ])),
        ),
    );

    let grouped_quantity_source = match variant {
        Q17ResidentGroupedAggregateInputVariant::GroupedAggregateGpuResidentInput => builder.add_source(
            SourceNode::GpuResident(GpuResidentSource::new("q17_lineitem_grouped", Arc::clone(&lineitem_schema))),
        ),
        Q17ResidentGroupedAggregateInputVariant::BlockedFamily
        | Q17ResidentGroupedAggregateInputVariant::RootJoinWithoutResidual
        | Q17ResidentGroupedAggregateInputVariant::BareGroupedAggregateInput
        | Q17ResidentGroupedAggregateInputVariant::ScalarAggregateInput
        | Q17ResidentGroupedAggregateInputVariant::BareGlobalAggregateInput
        | Q17ResidentGroupedAggregateInputVariant::FinalSort
        | Q17ResidentGroupedAggregateInputVariant::GlobalAggregateNotSum
        | Q17ResidentGroupedAggregateInputVariant::GroupedAggregateNotAvg
        | Q17ResidentGroupedAggregateInputVariant::ResidualWithoutScaledAverage
        | Q17ResidentGroupedAggregateInputVariant::ResidualWrongMultiplier
        | Q17ResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier
        | Q17ResidentGroupedAggregateInputVariant::RootOutputIncludesGroupedAggregate => {
            builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
                "/tmp/q17-lineitem-aggregate.parquet",
                Arc::clone(&lineitem_schema),
                vec!["l_partkey".into(), "l_quantity".into()],
            )))
        },
    };
    let aggregate_group_exprs = if matches!(variant, Q17ResidentGroupedAggregateInputVariant::ScalarAggregateInput) {
        Vec::new()
    } else {
        vec![Expr::column("l_partkey")]
    };
    let grouped_quantity = builder.add_aggregate(
        grouped_quantity_source,
        AggregateSpec::new(
            aggregate_group_exprs,
            vec![AggregateExpr::new(
                if matches!(variant, Q17ResidentGroupedAggregateInputVariant::GroupedAggregateNotAvg) {
                    AggregateFunction::Sum
                } else {
                    AggregateFunction::Avg
                },
                Expr::column("l_quantity"),
                "avg_quantity",
            )],
        ),
        Arc::clone(&grouped_quantity_schema),
    );
    let grouped_input = if matches!(variant, Q17ResidentGroupedAggregateInputVariant::BareGroupedAggregateInput) {
        grouped_quantity
    } else if matches!(variant, Q17ResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier) {
        builder.add_projection(
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
    } else {
        builder.add_projection(
            grouped_quantity,
            projection_outputs(
                vec![Expr::column("l_partkey"), Expr::column("avg_quantity")],
                Arc::clone(&grouped_quantity_schema),
            ),
        )
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
    let root_schema = if matches!(variant, Q17ResidentGroupedAggregateInputVariant::RootOutputIncludesGroupedAggregate)
    {
        root_spec = root_spec.with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_extendedprice", "l_extendedprice"), "l_extendedprice"),
            JoinOutputColumn::right(ColumnRef::input(1, "avg_quantity", "avg_quantity"), "avg_quantity"),
        ]);
        Arc::clone(&root_output_with_grouped_aggregate_schema)
    } else {
        Arc::clone(&root_output_schema)
    };
    if !matches!(variant, Q17ResidentGroupedAggregateInputVariant::RootJoinWithoutResidual) {
        let residual_right = if matches!(variant, Q17ResidentGroupedAggregateInputVariant::ResidualWithoutScaledAverage)
        {
            Expr::input_column(1, "avg_quantity", "avg_quantity")
        } else if matches!(variant, Q17ResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier) {
            Expr::input_column(1, "scaled_avg_quantity", "scaled_avg_quantity")
        } else {
            let multiplier = if matches!(variant, Q17ResidentGroupedAggregateInputVariant::ResidualWrongMultiplier) {
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
        root_spec = root_spec.with_residual_predicate(Expr::binary(
            Expr::input_column(0, "l_quantity", "l_quantity"),
            BinaryOp::Lt,
            residual_right,
        ));
    }
    let root = builder.add_join(joined_input, grouped_input, root_spec, root_schema);
    let aggregate_input = if matches!(variant, Q17ResidentGroupedAggregateInputVariant::BareGlobalAggregateInput) {
        root
    } else {
        builder.add_projection(
            root,
            projection_outputs(
                vec![Expr::column("l_extendedprice").alias("__cudf_aggregate_arg_0")],
                root_output_schema,
            ),
        )
    };
    let global = builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(
                if matches!(variant, Q17ResidentGroupedAggregateInputVariant::GlobalAggregateNotSum) {
                    AggregateFunction::Count
                } else {
                    AggregateFunction::Sum
                },
                Expr::column("__cudf_aggregate_arg_0"),
                "sum_extendedprice",
            )],
        ),
        Arc::clone(&global_aggregate_schema),
    );
    let projected = builder.add_projection(
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
    );
    let output = if matches!(variant, Q17ResidentGroupedAggregateInputVariant::FinalSort) {
        builder.add_sort(
            projected,
            SortSpec::new(vec![SortExpr::new(Expr::column("avg_yearly"), false, false)]),
            output_schema,
        )
    } else {
        projected
    };
    builder.finish_with_host_sink(output)
}

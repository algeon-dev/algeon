use super::*;

#[test]
fn bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline_is_executable() {
    for variant in [
        ResidualResidentGroupedAggregateInputVariant::BlockedFamily,
        ResidualResidentGroupedAggregateInputVariant::GlobalAggregateNotSum,
        ResidualResidentGroupedAggregateInputVariant::GroupedAggregateNotAvg,
        ResidualResidentGroupedAggregateInputVariant::ResidualWithoutScaledAverage,
        ResidualResidentGroupedAggregateInputVariant::ResidualWrongMultiplier,
        ResidualResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier,
    ] {
        let plan = residual_resident_grouped_aggregate_input_plan(variant);

        assert!(!plan.capability_analysis().admits_root_join(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline(), "{variant:?}");
        assert!(
            !plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph(), "{variant:?}");
        assert!(
            plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(),
            "{variant:?}"
        );
        assert!(unsupported_facts(&plan, ExecutionBoundaryPolicy::default()).is_none(), "{variant:?}");
        let facts = modeled_facts(&plan, ExecutionBoundaryPolicy::default())
            .expect("executable residual-resident grouped aggregate family should emit modeled facts");
        assert_eq!(facts.get("bounded_join_execution_family"), None, "{variant:?}");
        assert_eq!(facts.get("blocked_reservations").map(String::as_str), Some(""), "{variant:?}");
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""), "{variant:?}");
        let modeled = facts.get("modeled_reservations").expect("modeled reservations should be reported");
        for expected in [
            "source_read",
            "operator_output",
            "build_side",
            "hash_table",
            "probe_side",
            "mixed_size_data",
            "gather_map",
            "join_output",
            "materialization",
        ] {
            assert!(modeled.split(',').any(|kind| kind == expected), "{variant:?}: missing modeled {expected}");
        }
        assert!(
            !facts
                .get("bounded_join_reservation_requirements")
                .is_some_and(|value| value.contains("final_sort_contract")),
            "{variant:?}"
        );
        assert_modeled_reservation_facts_are_consistent(
            &facts,
            "mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline",
        );
    }
}

#[test]
fn bounded_mixed_keyed_residual_join_resident_grouped_aggregate_input_capability_proof_stays_narrow() {
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

        assert!(!plan.capability_analysis().admits_root_join(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline(), "{variant:?}");
        assert!(!plan.capability_analysis().admits_keyless_conditional_join_input_subplan_pipeline(), "{variant:?}");
        assert!(
            !plan.capability_analysis().admits_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
            "{variant:?}"
        );
        assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph(), "{variant:?}");
        assert!(
            !plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(),
            "{variant:?}"
        );
        let facts = unsupported_facts(&plan, ExecutionBoundaryPolicy::default())
            .expect("unsupported resident grouped aggregate input variant should produce facts");
        assert_rejection_facts_are_family_free(&facts, "unsupported_residual_resident_grouped_aggregate_input");
        assert!(facts.contains_key("bounded_join_rejection_reason"), "{variant:?}");
        assert_eq!(facts.get("bounded_join_lower_level_blockers").map(String::as_str), Some(""));
    }
}

#[test]
fn bounded_capability_analysis_extracts_mixed_residual_resident_aggregate_edge_contracts() {
    let plan =
        residual_resident_grouped_aggregate_input_plan(ResidualResidentGroupedAggregateInputVariant::BlockedFamily);
    let analysis = plan.capability_analysis();
    let residual_join = analysis
        .nodes
        .iter()
        .find(|node| {
            matches!(
                node.operator,
                NodeOperator::Join {
                    key_count: 1,
                    residual: ResidualRequirement::Required,
                    build_side: JoinSide::Right,
                    ..
                }
            )
        })
        .expect("mixed residual family should contain the keyed residual root join");
    assert!(residual_join.local_requirements.iter().any(|requirement| {
        requirement.kind == ReservationKind::MixedSizeData
            && requirement.scope == ReservationRequirementScope::Node
            && requirement.quality == ContractQuality::ObservedRetainedState
    }));
    assert!(analysis.edges.iter().any(|edge| {
        edge.contract_kinds
            .contains(&EdgeContractKind::ResidentSourceInput { format: SourceFormatCapability::LocalParquet })
            && edge.contract_kinds.contains(&EdgeContractKind::ResidentAggregateInput)
            && edge.input_role == InputRole::AggregateInput
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::SourceRead
                    && requirement.scope == ReservationRequirementScope::Edge
                    && requirement.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight
            })
    }));
    assert!(analysis.edges.iter().any(|edge| {
        edge.consumer == EdgeConsumer::Node(residual_join.node_id)
            && matches!(edge.input_role, InputRole::JoinBuildInput { side: JoinSide::Right })
            && edge.contract_kinds.contains(&EdgeContractKind::JoinBuildInput { side: JoinSide::Right })
            && edge.contract_kinds.contains(&EdgeContractKind::ResidentFrameFlow)
            && edge.reservation_requirements.iter().any(|requirement| requirement.kind == ReservationKind::BuildSide)
    }));
    assert!(analysis.edges.iter().any(|edge| {
        edge.consumer == EdgeConsumer::Node(residual_join.node_id)
            && matches!(edge.input_role, InputRole::JoinProbeInput { side: JoinSide::Left })
            && edge.contract_kinds.contains(&EdgeContractKind::JoinProbeInput { side: JoinSide::Left })
            && edge.reservation_requirements.iter().any(|requirement| {
                requirement.kind == ReservationKind::ProbeSide
                    && requirement.scope == ReservationRequirementScope::Edge
                    && requirement.quality == ContractQuality::QueryEngineOwnedOrdering
            })
    }));
    assert!(analysis.edges.iter().any(|edge| {
        edge.producer == residual_join.node_id
            && edge.contract_kinds.contains(&EdgeContractKind::JoinMaterialization)
            && edge.reservation_requirements.iter().any(|requirement| requirement.kind == ReservationKind::GatherMap)
            && edge.reservation_requirements.iter().any(|requirement| requirement.kind == ReservationKind::JoinOutput)
    }));
    assert!(
        analysis.requirements.iter().all(|requirement| requirement.kind != ReservationKind::FinalSort),
        "mixed residual resident aggregate family has no final-sort edge contract"
    );
}

#[test]
fn bounded_mixed_residual_resident_grouped_aggregate_admission_uses_capability_proof() {
    let plan =
        residual_resident_grouped_aggregate_input_plan(ResidualResidentGroupedAggregateInputVariant::BlockedFamily);
    let analysis = plan.capability_analysis();

    assert_eq!(
        plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline(),
        analysis.admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline()
    );
    assert_eq!(
        analysis.mixed_keyed_residual_join_resident_grouped_aggregate_input_admission,
        CapabilityDecision::Satisfied
    );
}

#[test]
fn bounded_mixed_residual_resident_grouped_aggregate_preserves_node_failure_locality() {
    let plan = residual_resident_grouped_aggregate_input_plan(
        ResidualResidentGroupedAggregateInputVariant::UnsupportedResidualExpression,
    );
    let analysis = plan.capability_analysis();

    assert!(!plan.capability_analysis().admits_mixed_keyed_residual_join_resident_grouped_aggregate_input_pipeline());
    let CapabilityDecision::Rejected { reason, facts } =
        &analysis.mixed_keyed_residual_join_resident_grouped_aggregate_input_admission
    else {
        panic!("unsupported residual expression should reject mixed residual admission");
    };
    assert_eq!(*reason, "unsupported_join_residual_expression_capability");
    assert_eq!(facts.get("expression_family").map(String::as_str), Some("like"));
    assert!(
        analysis.invariants.iter().any(|invariant| {
            matches!(invariant.kind, SubgraphInvariantKind::ResidualUsesBothInputs { .. })
                && invariant.decision == CapabilityDecision::Satisfied
        }),
        "residual input coverage is satisfied, so admission must report the node-local expression failure first"
    );
}

/// Mutates a residual join shape used to prove the mixed keyed
/// residual join + resident grouped-aggregate input contract.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ResidualResidentGroupedAggregateInputVariant {
    /// Baseline positive fixture whose surrounding shape is kept out of broader
    /// bounded families, so only this narrow residual-resident proof admits it.
    BlockedFamily,
    RootJoinWithoutResidual,
    BareGroupedAggregateInput,
    ScalarAggregateInput,
    /// Starts the grouped input from a resident GPU frame; this must not be
    /// confused with the local source-to-aggregate reduction proof.
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
) -> nexus_query_engine::plan::QueryPlan {
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
    let filtered_lineitem = builder.add_filter(
        lineitem,
        Expr::binary(Expr::column("l_quantity"), BinaryOp::Gt, Expr::literal(LiteralValue::Float64(0.0))),
        Arc::clone(&lineitem_schema),
    );
    let part = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/residual-resident-part.parquet",
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
                Expr::cast(Expr::column("l_quantity"), DataType::Float64).alias("__nexus_native_join_residual_left_0"),
            ],
            Arc::new(Schema::new(vec![
                Field::new("l_partkey", DataType::Int32, false),
                Field::new("l_quantity", DataType::Float64, false),
                Field::new("l_extendedprice", DataType::Float64, false),
                Field::new("__nexus_native_join_residual_left_0", DataType::Float64, false),
            ])),
        ),
    );

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
    let aggregate_group_exprs = if matches!(variant, ResidualResidentGroupedAggregateInputVariant::ScalarAggregateInput)
    {
        Vec::new()
    } else {
        vec![Expr::column("l_partkey")]
    };
    let grouped_quantity = builder.add_aggregate(
        grouped_quantity_source,
        AggregateSpec::new(
            aggregate_group_exprs,
            vec![AggregateExpr::new(
                if matches!(variant, ResidualResidentGroupedAggregateInputVariant::GroupedAggregateNotAvg) {
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
    let grouped_input = if matches!(variant, ResidualResidentGroupedAggregateInputVariant::BareGroupedAggregateInput) {
        grouped_quantity
    } else if matches!(variant, ResidualResidentGroupedAggregateInputVariant::PreprojectedResidualWrongMultiplier) {
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
    let root = builder.add_join(joined_input, grouped_input, root_spec, root_schema);
    let aggregate_input = if matches!(variant, ResidualResidentGroupedAggregateInputVariant::BareGlobalAggregateInput) {
        root
    } else {
        builder.add_projection(
            root,
            projection_outputs(
                vec![Expr::column("l_extendedprice").alias("__nexus_native_aggregate_arg_0")],
                root_output_schema,
            ),
        )
    };
    let global = builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(
                if matches!(variant, ResidualResidentGroupedAggregateInputVariant::GlobalAggregateNotSum) {
                    AggregateFunction::Count
                } else {
                    AggregateFunction::Sum
                },
                Expr::column("__nexus_native_aggregate_arg_0"),
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
    let output = if matches!(variant, ResidualResidentGroupedAggregateInputVariant::FinalSort) {
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

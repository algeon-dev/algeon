#[test]
fn native_optimizer_discovers_repeated_grouped_aggregate_materialization_family() {
    let plan = repeated_grouped_aggregate_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 3);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 3);
    assert_eq!(candidate.aggregate_node_ids.len(), 3);
    assert_eq!(candidate.group_exprs, vec!["county".to_string(), "quarter".to_string(), "year".to_string()]);
    assert_eq!(candidate.exact_source_variant_count, 3);
    assert_eq!(candidate.repeated_selected_source_bytes, Some(3 * 512 * 1024 * 1024));
    assert_eq!(candidate.duplicate_selected_source_bytes, Some(2 * 512 * 1024 * 1024));
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Aggregate(_))).count(), 3);
    assert_eq!(result.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Aggregate(_))).count(), 1);
    assert_eq!(result.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Filter(_))).count(), 3);
    let source_predicates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.reader_predicate.as_ref().map(ToString::to_string),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(source_predicates.len(), 1);
    assert!(source_predicates[0].contains(" OR "));
    assert!(source_predicates[0].contains("quarter = 1"));
    assert!(source_predicates[0].contains("quarter = 2"));
    assert!(source_predicates[0].contains("quarter = 3"));
}

#[test]
fn native_optimizer_materializes_aliased_grouped_aggregates_when_branch_replay_is_unneeded() {
    let plan = repeated_aliased_grouped_aggregate_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert_eq!(diagnostics.candidates[0].reuse_count, 2);
}

#[test]
fn native_optimizer_materializes_grouped_aggregates_with_equivalent_projected_argument_aliases() {
    let plan = repeated_grouped_aggregate_materialization_with_projected_argument_aliases_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 2);
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert!(
        candidate.aggregate_exprs.iter().any(|expr| expr.contains("arg=CAST(sales AS Float64)")),
        "aggregate argument signature should resolve generated projection aliases: {:?}",
        candidate.aggregate_exprs
    );
    assert!(
        candidate.aggregate_exprs.iter().all(|expr| !expr.contains("__common_expr_")),
        "generated projection aliases must not split equivalent aggregate families: {:?}",
        candidate.aggregate_exprs
    );
    assert_eq!(plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Aggregate(_))).count(), 2);
    assert_eq!(result.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Aggregate(_))).count(), 1);
    let widened_predicate = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.reader_predicate.as_ref().map(ToString::to_string),
                _ => None,
            },
            _ => None,
        })
        .next()
        .expect("rewritten plan should retain a widened reader predicate");
    assert!(widened_predicate.contains("sales > 0"), "{widened_predicate}");
    assert!(widened_predicate.contains(" OR "), "{widened_predicate}");
}

#[test]
fn native_optimizer_does_not_materialize_projected_aliases_across_different_local_filters() {
    let plan = repeated_grouped_projected_alias_materialization_with_local_filters_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 0);
    assert_eq!(diagnostics.applied_count, 0);
}

#[test]
fn native_optimizer_combines_repeated_scalar_count_branches_with_filtered_aggregate() {
    let plan = repeated_scalar_count_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 2);
    assert!(candidate.group_exprs.is_empty());
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);

    let aggregates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 2);
    assert!(aggregates[0].spec.aggr_exprs.iter().all(|expr| expr.filter.is_some()));
    assert!(
        aggregates[0]
            .spec
            .aggr_exprs
            .iter()
            .filter_map(|expr| expr.filter.as_ref())
            .any(|filter| { expr_contains_int32_equality(filter, "t_hour", 8) })
    );
    assert!(
        aggregates[0]
            .spec
            .aggr_exprs
            .iter()
            .filter_map(|expr| expr.filter.as_ref())
            .any(|filter| { expr_contains_int32_equality(filter, "t_hour", 9) })
    );
    let source_predicates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.reader_predicate.as_ref().map(ToString::to_string),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(source_predicates.iter().any(|predicate| predicate.contains(" OR ")));
    assert!(plan_has_output_name_prefix(&result.plan, "__nexus_native_shared_scalar_"));
}

#[test]
fn native_optimizer_keeps_shared_subplan_materialization_contract_isolated() {
    let plan = repeated_scalar_count_materialization_with_mismatched_source_contracts_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.shared_subplan_materialization.applied_count, 0);
}

#[test]
fn native_optimizer_keeps_shared_subplan_materialization_decimal_contract_isolated() {
    let plan = repeated_scalar_count_materialization_with_mismatched_decimal_contracts_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.shared_subplan_materialization.applied_count, 0);
}

#[test]
fn native_optimizer_ignores_identity_projection_depth_for_scalar_count_materialization() {
    let plan = repeated_scalar_count_materialization_with_redundant_projection_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert_eq!(diagnostics.candidates[0].reuse_count, 2);
}

#[test]
fn native_optimizer_lifts_reader_predicate_filters_for_scalar_count_materialization() {
    let plan = repeated_scalar_count_materialization_with_reader_predicate_filters_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert_eq!(diagnostics.candidates[0].reuse_count, 2);

    let aggregates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 2);
    assert!(aggregates[0].spec.aggr_exprs.iter().all(|expr| expr.filter.is_some()));
    assert_eq!(
        result.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Filter(_))).count(),
        0,
        "source-adjacent filters represented by reader predicates should move into aggregate filters"
    );
    let source_predicates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.reader_predicate.as_ref().map(ToString::to_string),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(source_predicates.iter().any(|predicate| predicate.contains(" OR ")));
}

#[test]
fn native_optimizer_rejects_repeated_scalar_mixed_distinct_branches_without_streaming_support() {
    let plan = repeated_scalar_mixed_aggregate_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 3);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 3);
    assert!(candidate.group_exprs.is_empty());
    assert!(candidate.aggregate_exprs.iter().any(|expr| expr.contains("fn=avg")));
    assert!(candidate.aggregate_exprs.iter().any(|expr| expr.contains("distinct=true")));
    assert!(!candidate.applied);
    assert_eq!(candidate.filtered_aggregate_count, 9);
    assert_eq!(candidate.filtered_aggregate_memory_shape, "streaming_filtered_state_not_applied");
    assert_eq!(candidate.filtered_aggregate_estimation_input_rows, Some(10_000_000));
    assert!(candidate.filtered_aggregate_estimated_materialization_bytes.unwrap_or(0) > 0);
    assert_eq!(candidate.filtered_aggregate_memory_threshold_policy, "runtime_budget/2");
    assert_eq!(
        candidate.rejected_reason,
        Some(
            nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape
        )
    );
    let local_source_reuse = &result.diagnostics.local_parquet_source_reuse;
    assert_eq!(local_source_reuse.considered_count, 1);
    assert_eq!(local_source_reuse.applied_count, 1);
    let source_reuse_candidate = &local_source_reuse.candidates[0];
    assert!(source_reuse_candidate.applied);
    assert_eq!(source_reuse_candidate.rejected_reason, None);
    assert_eq!(local_parquet_source_count_for_stable_id(&result.plan, "store_sales"), 1);

    let aggregates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 3);
    assert!(!plan_has_output_name_prefix(&result.plan, "__nexus_native_shared_scalar_"));
}

#[test]
fn native_optimizer_combines_q09_like_scalar_branches_with_different_read_columns() {
    let plan = repeated_q09_like_scalar_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 6);
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);

    let sources = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 1);
    assert_eq!(
        sources[0].read_columns,
        vec!["ss_quantity".to_string(), "ss_ext_discount_amt".to_string(), "ss_net_paid".to_string(),]
    );

    let aggregates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 6);
    assert!(aggregates[0].spec.aggr_exprs.iter().all(|expr| expr.filter.is_some()));
    assert_eq!(aggregates[0].spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Avg).count(), 4);
}

#[test]
fn native_optimizer_combines_full_q09_like_scalar_branch_shape() {
    let plan = repeated_q09_like_scalar_materialization_plan_with_order_and_bucket_count(
        &[Q09ScalarBranch::Count, Q09ScalarBranch::AvgDiscount, Q09ScalarBranch::AvgNetPaid],
        5,
    );

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 15);
    assert_eq!(candidate.filtered_aggregate_count, 15);
    assert_eq!(candidate.filtered_aggregate_memory_shape, "runtime_budgeted_segmented_state_available");
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);

    let aggregate_nodes =
        result.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Aggregate(_))).collect::<Vec<_>>();
    assert_eq!(aggregate_nodes.len(), 1);
    let PlanNode::Aggregate(aggregate) = &aggregate_nodes[0].kind else {
        unreachable!("aggregate node filter should only keep aggregate nodes");
    };
    assert_eq!(aggregate.spec.aggr_exprs.len(), 15);
    assert_eq!(aggregate.spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Count).count(), 5);
    assert_eq!(aggregate.spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Avg).count(), 10);
    assert!(aggregate.spec.aggr_exprs.iter().all(|expr| expr.filter.is_some()));
    let aggregate_output_fields = aggregate_nodes[0].output_schema().fields();
    assert_eq!(aggregate_output_fields.len(), aggregate.spec.aggr_exprs.len());
    let avg_output_types = aggregate
        .spec
        .aggr_exprs
        .iter()
        .zip(aggregate_output_fields.iter())
        .filter_map(|(expr, field)| (expr.function == AggregateFunction::Avg).then_some(field.data_type()))
        .collect::<Vec<_>>();
    assert_eq!(avg_output_types.len(), 10);
    assert!(
        avg_output_types.iter().all(|data_type| matches!(data_type, DataType::Decimal128(11, 6))),
        "q09-like AVG outputs should preserve Decimal128 output types:\n{}",
        result.plan
    );
    let mut filters = aggregate
        .spec
        .aggr_exprs
        .iter()
        .filter_map(|expr| expr.filter.as_ref().map(ToString::to_string))
        .collect::<Vec<_>>();
    filters.sort();
    filters.dedup();
    assert_eq!(filters.len(), 5);
}

#[test]
fn native_optimizer_combines_q09_like_scalar_branches_when_projected_count_is_not_representative() {
    let plan = repeated_q09_like_scalar_materialization_plan_with_order(&[
        Q09ScalarBranch::AvgDiscount,
        Q09ScalarBranch::Count,
        Q09ScalarBranch::AvgNetPaid,
    ]);

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    assert_eq!(diagnostics.candidates[0].reuse_count, 6);

    let aggregates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 6);
    assert!(
        aggregates[0]
            .spec
            .aggr_exprs
            .iter()
            .filter(|expr| expr.function == AggregateFunction::Count)
            .all(|expr| matches!(&expr.expr, Expr::Literal(value) if !matches!(value, LiteralValue::Null(_)))),
        "projected literal count-star branches should be canonicalized to COUNT(non-null-literal):\n{}",
        result.plan
    );
}

#[test]
fn native_optimizer_combines_projected_literal_count_star_scalar_branches() {
    let plan = repeated_projected_literal_count_star_scalar_materialization_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 2);
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(candidate.filtered_aggregate_count, 2);
    assert_eq!(candidate.filtered_aggregate_memory_shape, "runtime_budgeted_segmented_state_available");
    assert_eq!(candidate.filtered_aggregate_estimation_input_rows, Some(10_000_000));
    assert!(candidate.filtered_aggregate_estimated_materialization_bytes.unwrap_or(0) > 0);
    assert_eq!(candidate.filtered_aggregate_memory_budget_bytes, None);
    assert_eq!(candidate.filtered_aggregate_memory_threshold_bytes, None);
    assert_eq!(candidate.filtered_aggregate_memory_threshold_policy, "runtime_budget/2");

    let aggregates = result
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 2);
    assert!(aggregates[0].spec.aggr_exprs.iter().all(|expr| expr.filter.is_some()));
    assert!(plan_has_output_name_prefix(&result.plan, "__nexus_native_shared_scalar_"));
}

#[test]
fn native_optimizer_does_not_combine_scalar_branches_with_different_projection_local_filter_semantics() {
    let plan = repeated_scalar_count_with_different_projected_filter_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 0);
    assert_eq!(diagnostics.applied_count, 0);
}

#[test]
fn native_optimizer_does_not_combine_scalar_branches_with_different_local_filters() {
    let plan = repeated_scalar_count_with_different_local_filter_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 0);
    assert_eq!(diagnostics.applied_count, 0);
}

#[test]
fn native_optimizer_rejects_grouped_materialization_when_source_predicate_collides_with_aggregate_alias() {
    let plan = repeated_grouped_aggregate_predicate_alias_collision_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)
    );
}

#[test]
fn native_optimizer_rejects_grouped_materialization_when_predicate_column_has_different_group_lineage() {
    let plan = repeated_grouped_aggregate_predicate_group_name_lineage_collision_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)
    );
}

#[test]
fn native_optimizer_rejects_grouped_materialization_with_duplicate_source_occurrences() {
    let plan = repeated_grouped_aggregate_self_join_duplicate_source_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_rejects_grouped_materialization_with_reused_source_node_occurrences() {
    let plan = repeated_grouped_aggregate_reused_source_node_self_join_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_rejects_unreachable_grouped_materialization_without_claiming_applied_rewrite() {
    let plan = unreachable_grouped_aggregate_materialization_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::NotReachableFromSink)
    );
}

#[test]
fn native_optimizer_rejects_scalar_materialization_with_conflicting_source_refs() {
    let plan = repeated_scalar_avg_with_conflicting_source_refs_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(
        candidate.rejected_reason,
        Some(nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_does_not_combine_scalar_branches_with_limit_below_aggregate() {
    let plan = repeated_scalar_count_with_limit_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.shared_subplan_materialization.considered_count, 0);
}

#[test]
fn native_optimizer_rejects_scalar_materialization_with_duplicate_join_input_column_names() {
    let plan = repeated_scalar_join_count_with_duplicate_input_names_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_does_not_combine_scalar_branches_with_duplicate_rowset_join_inputs() {
    let plan = repeated_scalar_self_join_count_with_duplicate_rowset_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 0);
    assert_eq!(diagnostics.applied_count, 0);
}

#[test]
fn native_optimizer_rejects_scalar_materialization_with_reused_source_node_join_inputs() {
    let plan = repeated_scalar_self_join_count_with_reused_source_node_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 2);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(nexus_query_engine::optimizer::SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported)
    );
}

#[test]
fn native_optimizer_derives_scalar_sum_from_grouped_sum_with_permuted_source_files() {
    let plan = grouped_and_scalar_sum_over_permuted_source_files_plan();

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::default());

    assert_ne!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.shared_subplan_materialization;
    assert_eq!(diagnostics.considered_count, 1);
    assert_eq!(diagnostics.candidate_count, 1);
    assert_eq!(diagnostics.applied_count, 1);
    let candidate = &diagnostics.candidates[0];
    assert_eq!(candidate.reuse_count, 2);
    assert_eq!(candidate.aggregate_node_ids.len(), 2);
    assert!(candidate.applied);

    let scalar_aggregate_node = result
        .plan
        .nodes()
        .iter()
        .find(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty()))
        .expect("optimized plan should retain a scalar aggregate");
    let PlanNode::Aggregate(scalar_aggregate) = &scalar_aggregate_node.kind else {
        unreachable!("matched above");
    };
    let grouped_aggregate_node = result.plan.node(scalar_aggregate.input).expect("scalar aggregate input should exist");
    assert!(
        matches!(&grouped_aggregate_node.kind, PlanNode::Aggregate(aggregate) if !aggregate.spec.group_exprs.is_empty()),
        "scalar aggregate should derive from grouped aggregate output:\n{}",
        result.plan
    );
}

#[test]
fn disabled_shared_subplan_materialization_does_not_derive_scalar_sum_from_grouped_sum() {
    let plan = grouped_and_scalar_sum_over_permuted_source_files_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.shared_subplan_materialization.is_empty());
}

#[test]
fn disabled_shared_subplan_materialization_emits_no_diagnostics() {
    let plan = repeated_grouped_aggregate_materialization_plan();
    let options = NativeOptimizerOptions {
        shared_subplan_materialization: SharedSubplanMaterializationOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.shared_subplan_materialization.is_empty());
}

fn repeated_grouped_aggregate_materialization_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let aggregate_schema = schema(&["county", "quarter", "year", "store_sales"]);
    let mut aggregates = Vec::new();
    for quarter in 1..=3 {
        let source = builder.add_source(estimated_sales_source_with_quarter_predicate(quarter, 512 * 1024 * 1024));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("county"), Expr::column("quarter"), Expr::column("year")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
            ),
            Arc::clone(&aggregate_schema),
        );
        aggregates.push(aggregate);
    }
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_aliased_grouped_aggregate_materialization_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source_schema = schema(&["county", "sales"]);
    let aggregate_schema = schema(&["county_name", "store_sales"]);
    let source = builder.add_source(estimated_local_source_with_stable_id(
        "store_sales_alias",
        "store_sales_alias",
        &["county", "sales"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("county").alias("county_name")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        source,
        SortSpec::new(vec![SortExpr::new(Expr::column("county"), false, false)]),
        source_schema,
    );
    let sorted_aggregate = builder.add_aggregate(
        sorted,
        AggregateSpec::new(
            vec![Expr::column("county").alias("county_name")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let aggregates = vec![aggregate, sorted_aggregate];
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_grouped_aggregate_materialization_with_projected_argument_aliases_plan()
-> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("county", DataType::Int32, false),
        Field::new("quarter", DataType::Int32, false),
        Field::new("year", DataType::Int32, false),
        Field::new("store_sales", DataType::Float64, true),
    ]));
    let mut aggregates = Vec::new();
    for quarter in 1..=2 {
        let source = builder
            .add_source(estimated_sales_source_with_common_sales_and_quarter_predicate(quarter, 512 * 1024 * 1024));
        let projected_sales = format!("__common_expr_{quarter}");
        let projection = builder.add_projection(
            source,
            vec![
                ProjectionOutput::new(Expr::column("county"), Field::new("county", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("quarter"), Field::new("quarter", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("year"), Field::new("year", DataType::Int32, false)),
                ProjectionOutput::new(
                    Expr::Cast { expr: Box::new(Expr::column("sales")), target_type: DataType::Float64 }
                        .alias(projected_sales.clone()),
                    Field::new(projected_sales.as_str(), DataType::Float64, true),
                ),
            ],
        );
        let aggregate = builder.add_aggregate(
            projection,
            AggregateSpec::new(
                vec![Expr::column("county"), Expr::column("quarter"), Expr::column("year")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column(projected_sales), "store_sales")],
            ),
            Arc::clone(&aggregate_schema),
        );
        aggregates.push(aggregate);
    }
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_grouped_projected_alias_materialization_with_local_filters_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source_schema = schema(&["county", "quarter", "year", "sales"]);
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("county", DataType::Int32, false),
        Field::new("quarter", DataType::Int32, false),
        Field::new("year", DataType::Int32, false),
        Field::new("store_sales", DataType::Float64, true),
    ]));
    let mut aggregates = Vec::new();
    for quarter in 1..=2 {
        let source = builder.add_source(estimated_sales_source_without_reader_predicate(512 * 1024 * 1024));
        let filtered = builder.add_filter(
            source,
            Expr::binary(Expr::column("quarter"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(quarter))),
            Arc::clone(&source_schema),
        );
        let projected_sales = format!("__common_expr_{quarter}");
        let projection = builder.add_projection(
            filtered,
            vec![
                ProjectionOutput::new(Expr::column("county"), Field::new("county", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("quarter"), Field::new("quarter", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("year"), Field::new("year", DataType::Int32, false)),
                ProjectionOutput::new(
                    Expr::Cast { expr: Box::new(Expr::column("sales")), target_type: DataType::Float64 }
                        .alias(projected_sales.clone()),
                    Field::new(projected_sales.as_str(), DataType::Float64, true),
                ),
            ],
        );
        let aggregate = builder.add_aggregate(
            projection,
            AggregateSpec::new(
                vec![Expr::column("county"), Expr::column("quarter"), Expr::column("year")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column(projected_sales), "store_sales")],
            ),
            Arc::clone(&aggregate_schema),
        );
        aggregates.push(aggregate);
    }
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn estimated_sales_source_with_common_sales_and_quarter_predicate(quarter: i32, bytes: u64) -> SourceNode {
    let source_schema = schema(&["county", "quarter", "year", "sales"]);
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![10_000_000]);
    let common = Expr::binary(Expr::column("sales"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0)));
    let varying = Expr::binary(Expr::column("quarter"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(quarter)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec!["county".to_string(), "quarter".to_string(), "year".to_string(), "sales".to_string()],
        )
        .with_reader_predicate(Expr::binary(common, BinaryOp::And, varying)),
    )
}

fn estimated_sales_source_without_reader_predicate(bytes: u64) -> SourceNode {
    let source_schema = schema(&["county", "quarter", "year", "sales"]);
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        source_schema,
        vec!["county".to_string(), "quarter".to_string(), "year".to_string(), "sales".to_string()],
    ))
}

fn repeated_grouped_aggregate_predicate_alias_collision_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let aggregate_schema = schema(&["county", "quarter", "year", "store_sales"]);
    let mut aggregates = Vec::new();
    for value in 1..=2 {
        let source = builder.add_source(estimated_sales_source_with_store_sales_predicate(value));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("county"), Expr::column("quarter"), Expr::column("year")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
            ),
            Arc::clone(&aggregate_schema),
        );
        aggregates.push(aggregate);
    }
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_grouped_aggregate_predicate_group_name_lineage_collision_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let join_schema = schema(&["customer_id", "sales"]);
    let aggregate_schema = schema(&["customer_id", "store_sales"]);
    let mut aggregates = Vec::new();
    for customer_id in 1..=2 {
        let fact = builder.add_source(estimated_local_source_with_stable_id(
            "fact_sales",
            "fact_sales",
            &["customer_id", "dim_key", "sales"],
            512 * 1024 * 1024,
            10_000_000,
        ));
        let dim = builder.add_source(estimated_customer_dimension_source_with_customer_predicate(customer_id));
        let joined = builder.add_join(
            fact,
            dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "dim_key", "dim_key"),
                    ColumnRef::input(1, "dim_key", "dim_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "customer_id", "customer_id"), "customer_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
            ]),
            Arc::clone(&join_schema),
        );
        let aggregate = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                vec![Expr::column("customer_id")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
            ),
            Arc::clone(&aggregate_schema),
        );
        aggregates.push(aggregate);
    }
    let union = builder.add_union(aggregates, aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_grouped_aggregate_self_join_duplicate_source_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let join_schema = schema(&["customer_id", "sales"]);
    let aggregate_schema = schema(&["customer_id", "store_sales"]);
    let left = builder.add_source(estimated_customer_fact_source_with_customer_predicate(1));
    let right = builder.add_source(estimated_customer_fact_source_with_customer_predicate(2));
    let joined = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "dim_key", "dim_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer_id", "customer_id"), "customer_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("customer_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        joined,
        SortSpec::new(vec![SortExpr::new(Expr::column("customer_id"), false, false)]),
        join_schema,
    );
    let sorted_aggregate = builder.add_aggregate(
        sorted,
        AggregateSpec::new(
            vec![Expr::column("customer_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let union = builder.add_union(vec![aggregate, sorted_aggregate], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_grouped_aggregate_reused_source_node_self_join_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let join_schema = schema(&["customer_id", "sales"]);
    let aggregate_schema = schema(&["customer_id", "store_sales"]);
    let source = builder.add_source(estimated_customer_fact_source_with_customer_predicate(1));
    let joined = builder.add_join(
        source,
        source,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "dim_key", "dim_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer_id", "customer_id"), "customer_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("customer_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        joined,
        SortSpec::new(vec![SortExpr::new(Expr::column("customer_id"), false, false)]),
        join_schema,
    );
    let sorted_aggregate = builder.add_aggregate(
        sorted,
        AggregateSpec::new(
            vec![Expr::column("customer_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let union = builder.add_union(vec![aggregate, sorted_aggregate], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn unreachable_grouped_aggregate_materialization_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let root = builder.add_source(estimated_local_source("root", "root_key", 1024, 10));
    let aggregate_schema = schema(&["county", "store_sales"]);
    let source = builder.add_source(estimated_local_source_with_stable_id(
        "unreachable_sales",
        "unreachable_sales",
        &["county", "sales"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("county")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sorted = builder.add_sort(
        source,
        SortSpec::new(vec![SortExpr::new(Expr::column("county"), false, false)]),
        schema(&["county", "sales"]),
    );
    let _sorted_aggregate = builder.add_aggregate(
        sorted,
        AggregateSpec::new(
            vec![Expr::column("county")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        aggregate_schema,
    );
    assert_ne!(aggregate, root, "unreachable aggregate should not be the plan root");
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_count_materialization_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for hour in [8, 9] {
        let fact =
            builder.add_source(estimated_local_source("store_sales", "ss_sold_time_sk", 512 * 1024 * 1024, 10_000_000));
        let time = builder.add_source(estimated_time_source_with_hour_predicate(hour));
        let time_key = builder.add_projection(
            time,
            vec![ProjectionOutput::new(Expr::column("t_time_sk"), Field::new("t_time_sk", DataType::Int32, false))],
        );
        let joined = builder.add_join(
            fact,
            time_key,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                    ColumnRef::input(1, "t_time_sk", "t_time_sk"),
                )],
            )
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                "ss_sold_time_sk",
            )])
            .with_build_side(JoinSide::Right),
            Arc::clone(&fact_schema),
        );
        let count = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count")],
            ),
            Arc::clone(&count_schema),
        );
        counts.push(count);
    }
    let union = builder.add_union(counts, count_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_scalar_count_materialization_with_mismatched_source_contracts_plan() -> nexus_query_engine::plan::QueryPlan
{
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for (hour, fact_contract) in
        [8, 9].into_iter().zip([remote_direct_customer_address_contract(), cache_policy_customer_address_contract()])
    {
        let fact = builder.add_source(estimated_source_with_resolved_contract(
            fact_contract,
            "store_sales",
            &["ss_sold_time_sk"],
            512 * 1024 * 1024,
            10_000_000,
        ));
        let time = builder.add_source(estimated_time_source_with_hour_predicate(hour));
        let time_key = builder.add_projection(
            time,
            vec![ProjectionOutput::new(Expr::column("t_time_sk"), Field::new("t_time_sk", DataType::Int32, false))],
        );
        let joined = builder.add_join(
            fact,
            time_key,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                    ColumnRef::input(1, "t_time_sk", "t_time_sk"),
                )],
            )
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                "ss_sold_time_sk",
            )])
            .with_build_side(JoinSide::Right),
            Arc::clone(&fact_schema),
        );
        let count = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count")],
            ),
            Arc::clone(&count_schema),
        );
        counts.push(count);
    }
    let union = builder.add_union(counts, count_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_scalar_count_materialization_with_mismatched_decimal_contracts_plan() -> nexus_query_engine::plan::QueryPlan
{
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for (hour, decimal_contract) in [8, 9].into_iter().zip([false, true]) {
        let mut fact_source = estimated_source_with_resolved_contract(
            remote_direct_customer_address_contract(),
            "store_sales",
            &["ss_sold_time_sk"],
            512 * 1024 * 1024,
            10_000_000,
        );
        if decimal_contract {
            let SourceNode::LocalParquet(source) = &mut fact_source else {
                unreachable!("estimated source helper always builds local Parquet")
            };
            source.files[0].reader_decimal_types =
                vec![ParquetReaderDecimalType::new("ss_sold_time_sk", ParquetReaderDecimalTypeId::Decimal64, -2)];
            source.refresh_diagnostics();
        }
        let fact = builder.add_source(fact_source);
        let time = builder.add_source(estimated_time_source_with_hour_predicate(hour));
        let time_key = builder.add_projection(
            time,
            vec![ProjectionOutput::new(Expr::column("t_time_sk"), Field::new("t_time_sk", DataType::Int32, false))],
        );
        let joined = builder.add_join(
            fact,
            time_key,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                    ColumnRef::input(1, "t_time_sk", "t_time_sk"),
                )],
            )
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                "ss_sold_time_sk",
            )])
            .with_build_side(JoinSide::Right),
            Arc::clone(&fact_schema),
        );
        let count = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count")],
            ),
            Arc::clone(&count_schema),
        );
        counts.push(count);
    }
    let union = builder.add_union(counts, count_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_scalar_count_materialization_with_redundant_projection_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for (hour, projection_depth) in [(8, 1), (9, 3)] {
        let fact =
            builder.add_source(estimated_local_source("store_sales", "ss_sold_time_sk", 512 * 1024 * 1024, 10_000_000));
        let time = builder.add_source(estimated_time_source_with_hour_predicate(hour));
        let time_key = builder.add_projection(
            time,
            vec![ProjectionOutput::new(Expr::column("t_time_sk"), Field::new("t_time_sk", DataType::Int32, false))],
        );
        let mut input = builder.add_join(
            fact,
            time_key,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                    ColumnRef::input(1, "t_time_sk", "t_time_sk"),
                )],
            )
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                "ss_sold_time_sk",
            )])
            .with_build_side(JoinSide::Right),
            Arc::clone(&fact_schema),
        );
        for _ in 0..projection_depth {
            input = builder.add_projection(
                input,
                vec![ProjectionOutput::new(
                    Expr::column("ss_sold_time_sk"),
                    Field::new("ss_sold_time_sk", DataType::Int32, false),
                )],
            );
        }
        let count = builder.add_aggregate(
            input,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count")],
            ),
            Arc::clone(&count_schema),
        );
        counts.push(count);
    }
    let union = builder.add_union(counts, count_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_scalar_count_materialization_with_reader_predicate_filters_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact_schema = schema(&["ss_sold_time_sk"]);
    let time_schema = Arc::new(Schema::new(vec![
        Field::new("t_time_sk", DataType::Int32, false),
        Field::new("t_hour", DataType::Int32, false),
    ]));
    let count_schema = schema(&["count"]);
    let mut counts = Vec::new();
    for hour in [8, 9] {
        let fact =
            builder.add_source(estimated_local_source("store_sales", "ss_sold_time_sk", 512 * 1024 * 1024, 10_000_000));
        let time = builder.add_source(estimated_time_source_with_hour_predicate(hour));
        let predicate = Expr::binary(Expr::column("t_hour"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(hour)));
        let filtered_time = builder.add_filter(time, predicate, Arc::clone(&time_schema));
        let time_key = builder.add_projection(
            filtered_time,
            vec![ProjectionOutput::new(Expr::column("t_time_sk"), Field::new("t_time_sk", DataType::Int32, false))],
        );
        let joined = builder.add_join(
            fact,
            time_key,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                    ColumnRef::input(1, "t_time_sk", "t_time_sk"),
                )],
            )
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "ss_sold_time_sk", "ss_sold_time_sk"),
                "ss_sold_time_sk",
            )])
            .with_build_side(JoinSide::Right),
            Arc::clone(&fact_schema),
        );
        let count = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count")],
            ),
            Arc::clone(&count_schema),
        );
        counts.push(count);
    }
    let union = builder.add_union(counts, count_schema);
    builder.finish_gpu_resident(union)
}

fn repeated_scalar_mixed_aggregate_materialization_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..3 {
        let source = builder.add_source(q28_like_store_sales_source(bucket));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new(format!("B{bucket}_LP"), DataType::Float64, true),
            Field::new(format!("B{bucket}_CNT"), DataType::Int64, false),
            Field::new(format!("B{bucket}_CNTD"), DataType::Int64, false),
        ]));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![
                    AggregateExpr::new(AggregateFunction::Avg, Expr::column("ss_list_price"), format!("B{bucket}_LP")),
                    AggregateExpr::new(
                        AggregateFunction::Count,
                        Expr::column("ss_list_price"),
                        format!("B{bucket}_CNT"),
                    ),
                    AggregateExpr::new(
                        AggregateFunction::Count,
                        Expr::column("ss_list_price"),
                        format!("B{bucket}_CNTD"),
                    )
                    .with_distinct(true),
                ],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    let output_schema = Arc::new(Schema::new(
        branch_schemas
            .iter()
            .flat_map(|schema| schema.fields().iter().map(|field| field.as_ref().clone()))
            .collect::<Vec<_>>(),
    ));
    let root = builder.add_limit(root, LimitSpec::fetch(100), output_schema);
    builder.finish_gpu_resident(root)
}

#[derive(Debug, Clone, Copy)]
enum Q09ScalarBranch {
    Count,
    AvgDiscount,
    AvgNetPaid,
}

fn repeated_q09_like_scalar_materialization_plan() -> nexus_query_engine::plan::QueryPlan {
    repeated_q09_like_scalar_materialization_plan_with_order(&[
        Q09ScalarBranch::Count,
        Q09ScalarBranch::AvgDiscount,
        Q09ScalarBranch::AvgNetPaid,
    ])
}

fn repeated_q09_like_scalar_materialization_plan_with_order(
    branch_order: &[Q09ScalarBranch],
) -> nexus_query_engine::plan::QueryPlan {
    repeated_q09_like_scalar_materialization_plan_with_order_and_bucket_count(branch_order, 2)
}

fn repeated_q09_like_scalar_materialization_plan_with_order_and_bucket_count(
    branch_order: &[Q09ScalarBranch],
    bucket_count: i32,
) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..bucket_count {
        for &branch in branch_order {
            let (aggregate_input, aggregate_expr, field_name, data_type) = match branch {
                Q09ScalarBranch::Count => {
                    let source = builder.add_source(q09_filter_only_count_source(bucket));
                    let count_arg = builder.add_projection(
                        source,
                        vec![
                            ProjectionOutput::new(
                                Expr::literal(LiteralValue::Int64(1)),
                                Field::new("__nexus_native_aggregate_arg_0", DataType::Int64, false),
                            ),
                            ProjectionOutput::new(
                                Expr::column("ss_quantity"),
                                Field::new("ss_quantity", DataType::Int32, false),
                            ),
                        ],
                    );
                    (
                        count_arg,
                        AggregateExpr::new(
                            AggregateFunction::Count,
                            Expr::column("__nexus_native_aggregate_arg_0"),
                            format!("B{bucket}_CNT"),
                        ),
                        format!("B{bucket}_CNT"),
                        DataType::Int64,
                    )
                },
                Q09ScalarBranch::AvgDiscount => {
                    let source = builder.add_source(q09_like_store_sales_source(bucket, branch));
                    (
                        source,
                        AggregateExpr::new(
                            AggregateFunction::Avg,
                            Expr::column("ss_ext_discount_amt"),
                            format!("B{bucket}_AVG_DISCOUNT"),
                        ),
                        format!("B{bucket}_AVG_DISCOUNT"),
                        DataType::Decimal128(11, 6),
                    )
                },
                Q09ScalarBranch::AvgNetPaid => {
                    let source = builder.add_source(q09_like_store_sales_source(bucket, branch));
                    (
                        source,
                        AggregateExpr::new(
                            AggregateFunction::Avg,
                            Expr::column("ss_net_paid"),
                            format!("B{bucket}_AVG_NET"),
                        ),
                        format!("B{bucket}_AVG_NET"),
                        DataType::Decimal128(11, 6),
                    )
                },
            };
            let output_schema = Arc::new(Schema::new(vec![Field::new(field_name, data_type, true)]));
            let aggregate = builder.add_aggregate(
                aggregate_input,
                AggregateSpec::new(Vec::new(), vec![aggregate_expr]),
                Arc::clone(&output_schema),
            );
            branches.push(aggregate);
            branch_schemas.push(output_schema);
        }
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_projected_literal_count_star_scalar_materialization_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let source = builder.add_source(q09_filter_only_count_source(bucket));
        let empty = builder.add_projection(source, Vec::new());
        let count_arg = builder.add_projection(
            empty,
            vec![ProjectionOutput::new(
                Expr::literal(LiteralValue::Int64(1)),
                Field::new("__nexus_native_aggregate_arg_0", DataType::Int64, false),
            )],
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            count_arg,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::column("__nexus_native_aggregate_arg_0"),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_count_with_different_projected_filter_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for projected_column in ["a", "b"] {
        let source = builder.add_source(two_value_projection_filter_source());
        let projection_schema = Arc::new(Schema::new(vec![Field::new("p", DataType::Int32, false)]));
        let projected = builder.add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column(projected_column), Field::new("p", DataType::Int32, false))],
        );
        let filtered = builder.add_filter(
            projected,
            Expr::binary(Expr::column("p"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
            projection_schema,
        );
        let output_schema =
            Arc::new(Schema::new(vec![Field::new(format!("CNT_{projected_column}"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            filtered,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("CNT_{projected_column}"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_count_with_different_local_filter_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let source = builder.add_source(q09_like_store_sales_source(bucket, Q09ScalarBranch::Count));
        let filter_schema = Arc::new(Schema::new(vec![Field::new("ss_quantity", DataType::Int32, false)]));
        let filtered = builder.add_filter(
            source,
            Expr::binary(
                Expr::column("ss_quantity"),
                BinaryOp::GtEq,
                Expr::literal(LiteralValue::Int32(bucket * 20 + 10)),
            ),
            filter_schema,
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            filtered,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_avg_with_conflicting_source_refs_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for file_index in [0, 1] {
        let source = builder.add_source(conflicting_ref_value_source(file_index));
        let output_schema =
            Arc::new(Schema::new(vec![Field::new(format!("AVG_V_{file_index}"), DataType::Float64, true)]));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("v"), format!("AVG_V_{file_index}"))],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_count_with_limit_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let source = builder.add_source(q09_like_store_sales_source(bucket, Q09ScalarBranch::Count));
        let limit_schema = Arc::new(Schema::new(vec![Field::new("ss_quantity", DataType::Int32, false)]));
        let limited = builder.add_limit(source, LimitSpec::fetch(10), limit_schema);
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            limited,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_join_count_with_duplicate_input_names_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let left = builder.add_source(duplicate_named_source("left", bucket));
        let right = builder.add_source(duplicate_named_source("right", 0));
        let join_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let joined = builder.add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "v", "v"), ColumnRef::input(1, "v", "v"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v")]),
            join_schema,
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_self_join_count_with_duplicate_rowset_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let left = builder.add_source(duplicate_named_source("same-rowset", bucket));
        let right = builder.add_source(duplicate_named_source("same-rowset", 0));
        let join_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let joined = builder.add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "v", "v"), ColumnRef::input(1, "v", "v"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v")]),
            join_schema,
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn repeated_scalar_self_join_count_with_reused_source_node_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let mut branches = Vec::new();
    let mut branch_schemas = Vec::new();
    for bucket in 0..2 {
        let source = builder.add_source(duplicate_named_source("same-reused-rowset", bucket));
        let join_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let joined = builder.add_join(
            source,
            source,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "v", "v"), ColumnRef::input(1, "v", "v"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v")]),
            join_schema,
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new(format!("B{bucket}_CNT"), DataType::Int64, true)]));
        let aggregate = builder.add_aggregate(
            joined,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("B{bucket}_CNT"),
                )],
            ),
            Arc::clone(&output_schema),
        );
        branches.push(aggregate);
        branch_schemas.push(output_schema);
    }
    let root = cross_join_scalar_branches(&mut builder, &branches, &branch_schemas);
    builder.finish_gpu_resident(root)
}

fn grouped_and_scalar_sum_over_permuted_source_files_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let grouped_input = builder.add_source(two_file_source(["a", "b"]));
    let scalar_input = builder.add_source(two_file_source(["b", "a"]));
    let grouped = builder.add_aggregate(
        grouped_input,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "sum_v")],
        ),
        schema(&["k", "sum_v"]),
    );
    let scalar = builder.add_aggregate(
        scalar_input,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total_v")]),
        schema(&["total_v"]),
    );
    let root = builder.add_join(
        grouped,
        scalar,
        JoinSpec::new(JoinType::Inner, Vec::new()).with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "sum_v", "sum_v"), "sum_v"),
            JoinOutputColumn::right(ColumnRef::input(1, "total_v", "total_v"), "total_v"),
        ]),
        schema(&["k", "sum_v", "total_v"]),
    );
    builder.finish_gpu_resident(root)
}

fn two_file_source(order: [&str; 2]) -> SourceNode {
    let source_schema = schema(&["k", "v"]);
    let files = order
        .into_iter()
        .map(|name| {
            LocalParquetFile::new(format!("/tmp/{name}.parquet"))
                .with_stable_id(name)
                .with_projected_row_group_bytes(vec![1024], vec![1024])
                .with_row_group_rows(vec![10])
        })
        .collect::<Vec<_>>();
    SourceNode::LocalParquet(LocalParquetSource::from_files(
        files,
        source_schema,
        vec!["k".to_string(), "v".to_string()],
    ))
}

fn estimated_sales_source_with_quarter_predicate(quarter: i32, bytes: u64) -> SourceNode {
    let source_schema = schema(&["county", "quarter", "year", "sales"]);
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![10_000_000]);
    let predicate = Expr::binary(Expr::column("quarter"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(quarter)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec!["county".to_string(), "quarter".to_string(), "year".to_string(), "sales".to_string()],
        )
        .with_reader_predicate(predicate),
    )
}

fn estimated_sales_source_with_store_sales_predicate(value: i32) -> SourceNode {
    let source_schema = schema(&["county", "quarter", "year", "sales", "store_sales"]);
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    let predicate = Expr::binary(Expr::column("store_sales"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(value)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec![
                "county".to_string(),
                "quarter".to_string(),
                "year".to_string(),
                "sales".to_string(),
                "store_sales".to_string(),
            ],
        )
        .with_reader_predicate(predicate),
    )
}

fn estimated_customer_dimension_source_with_customer_predicate(customer_id: i32) -> SourceNode {
    let source_schema = schema(&["dim_key", "customer_id"]);
    let file = LocalParquetFile::new("/tmp/customer_dim.parquet")
        .with_stable_id("customer_dim")
        .with_projected_row_group_bytes(vec![1024 * 1024], vec![1024 * 1024])
        .with_row_group_rows(vec![100_000]);
    let predicate =
        Expr::binary(Expr::column("customer_id"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(customer_id)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec!["dim_key".to_string(), "customer_id".to_string()],
        )
        .with_reader_predicate(predicate),
    )
}

fn estimated_customer_fact_source_with_customer_predicate(customer_id: i32) -> SourceNode {
    let source_schema = schema(&["dim_key", "customer_id", "sales"]);
    let file = LocalParquetFile::new("/tmp/customer_fact.parquet")
        .with_stable_id("customer_fact")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    let predicate =
        Expr::binary(Expr::column("customer_id"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(customer_id)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec!["dim_key".to_string(), "customer_id".to_string(), "sales".to_string()],
        )
        .with_reader_predicate(predicate),
    )
}

fn estimated_time_source_with_hour_predicate(hour: i32) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("t_time_sk", DataType::Int32, false),
        Field::new("t_hour", DataType::Int32, false),
    ]));
    let file = LocalParquetFile::new("/tmp/time_dim.parquet")
        .with_stable_id("time_dim")
        .with_projected_row_group_bytes(vec![1024 * 1024], vec![1024 * 1024])
        .with_row_group_rows(vec![86_400]);
    let predicate = Expr::binary(Expr::column("t_hour"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(hour)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["t_time_sk".to_string(), "t_hour".to_string()])
            .with_column_roles(vec!["t_time_sk".to_string()], vec!["t_hour".to_string()])
            .with_reader_predicate(predicate),
    )
}

fn q28_like_store_sales_source(bucket: i32) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("ss_list_price", DataType::Int32, true),
        Field::new("ss_coupon_amt", DataType::Int32, true),
        Field::new("ss_wholesale_cost", DataType::Int32, true),
    ]));
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            source_schema,
            vec![
                "ss_quantity".to_string(),
                "ss_list_price".to_string(),
                "ss_coupon_amt".to_string(),
                "ss_wholesale_cost".to_string(),
            ],
        )
        .with_reader_predicate(q28_like_bucket_predicate(bucket)),
    )
}

fn q09_like_store_sales_source(bucket: i32, branch: Q09ScalarBranch) -> SourceNode {
    let all_fields = [
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("ss_ext_discount_amt", DataType::Decimal128(7, 2), true),
        Field::new("ss_net_paid", DataType::Decimal128(7, 2), true),
    ];
    let read_columns = match branch {
        Q09ScalarBranch::Count => vec!["ss_quantity".to_string()],
        Q09ScalarBranch::AvgDiscount => vec!["ss_quantity".to_string(), "ss_ext_discount_amt".to_string()],
        Q09ScalarBranch::AvgNetPaid => vec!["ss_quantity".to_string(), "ss_net_paid".to_string()],
    };
    let fields = read_columns
        .iter()
        .map(|column| {
            all_fields.iter().find(|field| field.name() == column).expect("read column should have a field").clone()
        })
        .collect::<Vec<_>>();
    let source_schema = Arc::new(Schema::new(fields));
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, read_columns).with_reader_predicate(between_i32(
            "ss_quantity",
            bucket * 20 + 1,
            bucket * 20 + 20,
        )),
    )
}

fn q09_filter_only_count_source(bucket: i32) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![Field::new("ss_quantity", DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/store_sales.parquet")
        .with_stable_id("store_sales")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["ss_quantity".to_string()])
            .with_column_roles(Vec::new(), vec!["ss_quantity".to_string()])
            .with_reader_predicate(between_i32("ss_quantity", bucket * 20 + 1, bucket * 20 + 20)),
    )
}

fn two_value_projection_filter_source() -> SourceNode {
    let source_schema =
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false), Field::new("b", DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/two-value-projection-filter.parquet")
        .with_stable_id("two-value-projection-filter")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        source_schema,
        vec!["a".to_string(), "b".to_string()],
    ))
}

fn conflicting_ref_value_source(file_index: usize) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
    let file = LocalParquetFile::new("/tmp/conflicting-value-source.parquet")
        .with_stable_id("conflicting-value-source")
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["v".to_string()])
            .with_column_file_indices(vec![file_index], vec![file_index], Vec::new())
            .with_reader_predicate(Expr::binary(
                Expr::column("v"),
                BinaryOp::GtEq,
                Expr::literal(LiteralValue::Float64(0.0)),
            )),
    )
}

fn duplicate_named_source(name: &str, bucket: i32) -> SourceNode {
    let source_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
    let file = LocalParquetFile::new(format!("/tmp/{name}-duplicate-v.parquet"))
        .with_stable_id(name)
        .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
        .with_row_group_rows(vec![10_000_000]);
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["v".to_string()]).with_reader_predicate(
            Expr::binary(Expr::column("v"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(bucket))),
        ),
    )
}

fn q28_like_bucket_predicate(bucket: i32) -> Expr {
    let quantity_min = bucket * 5;
    let quantity_max = quantity_min + 5;
    let list_min = 10 + bucket * 20;
    let coupon_min = 1000 + bucket * 100;
    let cost_min = 30 + bucket * 10;
    Expr::binary(
        between_i32("ss_quantity", quantity_min, quantity_max),
        BinaryOp::And,
        Expr::binary(
            between_i32("ss_list_price", list_min, list_min + 10),
            BinaryOp::Or,
            Expr::binary(
                between_i32("ss_coupon_amt", coupon_min, coupon_min + 1000),
                BinaryOp::Or,
                between_i32("ss_wholesale_cost", cost_min, cost_min + 20),
            ),
        ),
    )
}

fn cross_join_scalar_branches(
    builder: &mut QueryPlanBuilder,
    branches: &[NodeId],
    branch_schemas: &[Arc<Schema>],
) -> NodeId {
    let mut current = branches[0];
    let mut current_fields = branch_schemas[0].fields().iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>();
    for (branch, schema) in branches.iter().copied().zip(branch_schemas).skip(1) {
        let mut output_projection = Vec::new();
        for field in &current_fields {
            output_projection.push(JoinOutputColumn::left(
                ColumnRef::input(0, field.name().clone(), field.name().clone()),
                field.name().clone(),
            ));
        }
        for field in schema.fields() {
            output_projection.push(JoinOutputColumn::right(
                ColumnRef::input(1, field.name().clone(), field.name().clone()),
                field.name().clone(),
            ));
            current_fields.push(field.as_ref().clone());
        }
        let output_schema = Arc::new(Schema::new(current_fields.clone()));
        current = builder.add_join(
            current,
            branch,
            JoinSpec::new(JoinType::Inner, Vec::new()).with_output_projection(output_projection),
            output_schema,
        );
    }
    current
}

fn nested_reader_predicate_with_declared_boolean_output_types() -> Expr {
    let date_lower = Expr::binary_with_output_type(
        Expr::column("inv_date_sk"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Int32(2_452_001)),
        DataType::Boolean,
    );
    let date_upper = Expr::binary_with_output_type(
        Expr::column("inv_date_sk"),
        BinaryOp::Lt,
        Expr::literal(LiteralValue::Int32(2_452_010)),
        DataType::Boolean,
    );
    let date_range = Expr::binary_with_output_type(date_lower, BinaryOp::And, date_upper, DataType::Boolean);
    let item_range = Expr::binary_with_output_type(
        Expr::column("inv_item_sk"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Int32(10)),
        DataType::Boolean,
    );
    Expr::binary_with_output_type(date_range, BinaryOp::Or, item_range, DataType::Boolean)
}

fn assert_no_non_decimal_binary_output_types(expr: &Expr, message: &str) {
    match expr {
        Expr::Binary { left, op, right, output_type } => {
            assert!(
                output_type.is_none()
                    || (*op == BinaryOp::Divide
                        && output_type
                            .as_ref()
                            .is_some_and(|data_type| matches!(data_type, DataType::Decimal128(_, _)))),
                "{message}: {expr}"
            );
            assert_no_non_decimal_binary_output_types(left, message);
            assert_no_non_decimal_binary_output_types(right, message);
        },
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Cast { expr: inner, .. }
        | Expr::TryCast { expr: inner, .. }
        | Expr::Like { expr: inner, .. }
        | Expr::Substring { expr: inner, .. }
        | Expr::Trim { expr: inner, .. }
        | Expr::LTrim { expr: inner, .. }
        | Expr::RTrim { expr: inner, .. }
        | Expr::StringContains { expr: inner, .. }
        | Expr::StartsWith { expr: inner, .. }
        | Expr::EndsWith { expr: inner, .. }
        | Expr::Lower(inner)
        | Expr::Upper(inner)
        | Expr::StringLength { expr: inner, .. }
        | Expr::RegexpReplace { expr: inner, .. }
        | Expr::Abs(inner)
        | Expr::Sqrt { expr: inner, .. }
        | Expr::Round { expr: inner, .. }
        | Expr::Extract { expr: inner, .. }
        | Expr::DateTrunc { expr: inner, .. }
        | Expr::Alias { expr: inner, .. } => assert_no_non_decimal_binary_output_types(inner, message),
        Expr::Concat(exprs) => {
            for expr in exprs {
                assert_no_non_decimal_binary_output_types(expr, message);
            }
        },
        Expr::InList { expr, list, .. } => {
            assert_no_non_decimal_binary_output_types(expr, message);
            for expr in list {
                assert_no_non_decimal_binary_output_types(expr, message);
            }
        },
        Expr::IsDistinctFrom { left, right } | Expr::IsNotDistinctFrom { left, right } => {
            assert_no_non_decimal_binary_output_types(left, message);
            assert_no_non_decimal_binary_output_types(right, message);
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                assert_no_non_decimal_binary_output_types(&branch.when, message);
                assert_no_non_decimal_binary_output_types(&branch.then, message);
            }
            assert_no_non_decimal_binary_output_types(else_expr, message);
        },
        Expr::Column(_) | Expr::Literal(_) => {},
    }
}

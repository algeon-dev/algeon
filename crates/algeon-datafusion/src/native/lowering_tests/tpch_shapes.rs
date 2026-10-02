use query_engine::{
    admission::{admit, bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    dimension_pruning::local_dimension_pruning_resolver,
    dynamic_filter::{DynamicFilterConsumer, DynamicFilterConsumerLowering},
    optimizer::{
        AggregateJoinPushdownRejectedReason, NativeOptimizerChange, NativeOptimizerOptions, SemijoinReductionOptions,
        optimize_query_plan,
    },
    plan::{AggregateNode, JoinNode},
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
};

use super::*;
use crate::native::ParquetReadStrategyConfig;

#[path = "tpch_shapes/q22.rs"]
mod q22;

/// The native optimizer's rule rows for every lowered TPC-H fixture plan. A
/// change to a rule's decisions, its candidate order, or which passes run
/// shows up as a line diff against the golden.
#[tokio::test]
async fn tpch_native_optimizer_rule_rows_match_golden() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-optimizer-golden");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let dataset = dir.display().to_string();

    let mut rows = Vec::new();
    for item in &items {
        let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &item.input else {
            panic!("{} should produce a physical plan", item.query_id);
        };
        let lowered = lower_physical_plan_to_native(Arc::clone(plan), &ParquetReadStrategyConfig::default()).unwrap();
        for (fragment, native_plan) in lowered.plan.iter().enumerate() {
            let optimized = optimize_query_plan(
                native_plan.clone(),
                &NativeOptimizerOptions::default(),
                rapids_analysis_context(PlanningPolicy::default()),
                local_dimension_pruning_resolver(),
            )
            .unwrap();
            for row in optimized.diagnostics.stable_rule_rows() {
                rows.push(format!("{}\t{fragment}\t{}", item.query_id, normalized_rule_row(&row, &dataset)));
            }
        }
    }
    let actual = rows.join("\n") + "\n";
    assert_eq!(actual, include_str!("goldens/tpch_optimizer_rule_rows.txt"));

    fs::remove_dir_all(dir).unwrap();
}

/// Replaces the run-specific parts of a rule row: the temporary dataset
/// directory, file modification times, and hashes of file locations.
fn normalized_rule_row(row: &str, dataset: &str) -> String {
    let mut row = row.replace(dataset, "<dataset>");
    row = replace_after(&row, "mtime=", usize::MAX, |c| c.is_ascii_digit() || matches!(c, '-' | 'T' | ':' | '.' | '+'));
    for prefix in ["fnv64:", "fnv64-", "fnv1a64:"] {
        row = replace_after(&row, prefix, 16, |c| c.is_ascii_hexdigit());
    }
    row
}

/// Replaces up to `limit` characters `keep` accepts after each `prefix` with
/// `<run>`.
fn replace_after(text: &str, prefix: &str, limit: usize, keep: impl Fn(char) -> bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find(prefix) {
        out.push_str(&rest[..index + prefix.len()]);
        rest = &rest[index + prefix.len()..];
        let end = rest
            .char_indices()
            .take(limit)
            .find(|(_, c)| !keep(*c))
            .map_or_else(|| rest.char_indices().nth(limit).map_or(rest.len(), |(end, _)| end), |(end, _)| end);
        out.push_str("<run>");
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

#[tokio::test]
async fn tpch_recent_bounded_topologies_expose_modeled_facts() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-recent-topologies");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();

    let q2 = items.iter().find(|item| item.query_id == "q02").expect("TPC-H workload should include q02");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q2.input else {
        panic!("q02 should produce a physical plan");
    };
    let result = lower_physical_plan_to_native(Arc::clone(plan), &ParquetReadStrategyConfig::default()).unwrap();
    assert_eq!(result.selected_count(), 1, "q02 outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let optimized_q2 = optimize_query_plan(
        native_plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert!(
        optimized_q2.diagnostics.join_filter_reassociation.applied_count >= 1,
        "q02 should use join-filter reassociation to push nation/region filtering below the partsupp join:\n{}",
        optimized_q2.plan
    );
    assert_q02_reassociated_supplier_filter_precedes_partsupp_join(&optimized_q2.plan);
    assert_q02_part_partsupp_dynamic_filter_matches_selected_build_side(&optimized_q2.plan);
    assert!(
        admit(
            &optimized_q2.plan,
            Default::default(),
            PlanningPolicy::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .expect("optimized q02 bounded admission should not fail")
        .admission()
        .is_some(),
        "optimized q02 should remain admitted after post-root filtering joins are exposed:\n{}",
        optimized_q2.plan
    );
    let decision = admit(
        native_plan,
        Default::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("q02 bounded admission should not fail");
    let facts = bounded_join_modeled_facts(native_plan, &decision, rapids_analysis_context(PlanningPolicy::default()))
        .expect("q02 should expose modeled bounded join facts");
    assert_eq!(facts.get("bounded_join_execution_family"), None);

    let q15 = items.iter().find(|item| item.query_id == "q15").expect("TPC-H workload should include q15");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q15.input else {
        panic!("q15 should produce a physical plan");
    };
    let result = lower_physical_plan_to_native(Arc::clone(plan), &ParquetReadStrategyConfig::default()).unwrap();
    assert_eq!(result.selected_count(), 1, "q15 outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let optimized_q15 = optimize_query_plan(
        native_plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert!(
        admit(
            &optimized_q15.plan,
            Default::default(),
            PlanningPolicy::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .expect("optimized q15 bounded admission should not fail")
        .admission()
        .is_some(),
        "optimized q15 should be admitted as a shared aggregate fanout pipeline:\n{}",
        optimized_q15.plan
    );
    let decision = admit(
        native_plan,
        Default::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("q15 bounded admission should not fail");
    let facts = bounded_join_modeled_facts(native_plan, &decision, rapids_analysis_context(PlanningPolicy::default()))
        .expect("q15 should expose modeled bounded join facts");
    assert_eq!(facts.get("bounded_join_execution_family"), None);

    let q18 = items.iter().find(|item| item.query_id == "q18").expect("TPC-H workload should include q18");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q18.input else {
        panic!("q18 should produce a physical plan");
    };
    let result = lower_physical_plan_to_native(Arc::clone(plan), &ParquetReadStrategyConfig::default()).unwrap();
    assert_eq!(result.selected_count(), 1, "q18 outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let decision = admit(
        native_plan,
        Default::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("q18 bounded admission should not fail");
    let facts = bounded_join_modeled_facts(native_plan, &decision, rapids_analysis_context(PlanningPolicy::default()))
        .expect("q18 should expose modeled bounded join facts");
    assert_eq!(facts.get("bounded_join_execution_family"), None);

    let q20 = items.iter().find(|item| item.query_id == "q20").expect("TPC-H workload should include q20");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q20.input else {
        panic!("q20 should produce a physical plan");
    };
    let result = lower_physical_plan_to_native(Arc::clone(plan), &ParquetReadStrategyConfig::default()).unwrap();
    assert_eq!(result.selected_count(), 1, "q20 outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let decision = admit(
        native_plan,
        Default::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("q20 bounded admission should not fail");
    let facts = bounded_join_modeled_facts(native_plan, &decision, rapids_analysis_context(PlanningPolicy::default()))
        .expect("q20 should expose modeled bounded join facts");
    assert_eq!(facts.get("bounded_join_execution_family"), None);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q18_direct_lowering_preserves_pre_semijoin_reduction_late_left_semi_shape() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q18-pre-semijoin-reduction");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q18 = items.into_iter().find(|item| item.query_id == "q18").expect("TPC-H workload should include q18");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q18.input else {
        panic!("q18 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let plan = result.selected_plan();
    let left_semi_joins = native_join_ids(plan, NativeJoinType::LeftSemi);
    assert_eq!(left_semi_joins.len(), 1, "Q18 before semijoin reduction should have one late left-semi join");
    let (_semi_id, semi_join) = left_semi_joins[0];
    assert_eq!(semi_join.spec.build_side, NativeJoinSide::Right);
    assert_eq!(semi_join.spec.null_equality, NativeJoinNullEquality::NullsDoNotMatch);
    assert_eq!(semi_join.spec.keys.len(), 1);
    assert!(
        semi_join.spec.output_projection.iter().all(|column| column.side == NativeJoinSide::Left),
        "left-semi output must expose only preserved-side columns"
    );
    assert_eq!(
        source_tables_below(plan, semi_join.inputs[0]),
        BTreeSet::from(["customer".to_string(), "lineitem".to_string(), "orders".to_string()]),
        "current direct lowering applies Q18's semi filter after the customer/orders/lineitem join tree"
    );
    assert_eq!(
        source_tables_below(plan, semi_join.inputs[1]),
        BTreeSet::from(["lineitem".to_string()]),
        "Q18's semi-filter build side should remain the grouped lineitem keyset branch"
    );
    assert_eq!(
        join_type_count_below(plan, semi_join.inputs[0], NativeJoinType::Inner),
        2,
        "pre-rewrite Q18 should still materialize the two inner joins before the left-semi filter"
    );
    assert_eq!(
        join_type_count_below(plan, semi_join.inputs[1], NativeJoinType::Inner),
        0,
        "the semi-filter keyset branch should not hide an inner join tree"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q18_semijoin_reduction_pushes_filter_to_orders_subtree() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q18-semijoin-reduction");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q18 = items.into_iter().find(|item| item.query_id == "q18").expect("TPC-H workload should include q18");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q18.input else {
        panic!("q18 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    let optimized = optimize_query_plan(
        result.selected_plan().clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(optimized.change, NativeOptimizerChange::Changed);
    assert_eq!(optimized.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(optimized.diagnostics.semijoin_reduction.applied_count, 1);
    assert_eq!(optimized.diagnostics.semijoin_reduction.candidates[0].crossed_join_node_ids.len(), 2);

    let left_semi_joins = native_join_ids(&optimized.plan, NativeJoinType::LeftSemi);
    assert_eq!(left_semi_joins.len(), 1, "Q18 after semijoin reduction should retain one pushed left-semi join");
    let (_semi_id, semi_join) = left_semi_joins[0];
    assert_eq!(
        source_tables_below(&optimized.plan, semi_join.inputs[0]),
        BTreeSet::from(["orders".to_string()]),
        "Q18 semijoin reduction should push the filter to the orders subtree that owns o_orderkey"
    );
    assert_eq!(
        source_tables_below(&optimized.plan, semi_join.inputs[1]),
        BTreeSet::from(["lineitem".to_string()]),
        "Q18 semijoin reduction should keep the grouped lineitem keyset branch on the semijoin right side"
    );
    assert_eq!(
        join_type_count_below(&optimized.plan, semi_join.inputs[0], NativeJoinType::Inner),
        0,
        "Q18 should no longer apply the semi-filter after materializing the customer/orders/lineitem join tree"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q18_late_left_semi_filter_stays_below_the_final_aggregate() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q18-late-semi-before-aggregate");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q18 = items.into_iter().find(|item| item.query_id == "q18").expect("TPC-H workload should include q18");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q18.input else {
        panic!("q18 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    // On SF100 the bounded admission gate refuses the orders-side semijoin
    // rewrite (`root_topk_join_aggregate_requires_source_reduction_evidence`),
    // so the late left-semi survives into aggregate-join pushdown; the tiny
    // dataset admits the rewrite, so disable the pass to reach the same shape.
    let options =
        NativeOptimizerOptions { semijoin_reduction: SemijoinReductionOptions::disabled(), ..Default::default() };
    let optimized = optimize_query_plan(
        result.selected_plan().clone(),
        &options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    let pushdown = &optimized.diagnostics.aggregate_join_pushdown;
    assert!(
        pushdown.candidates.iter().any(|candidate| candidate.rejected_reason
            == Some(AggregateJoinPushdownRejectedReason::LeftSemiAlreadyFiltersFinalGroups)),
        "the o_orderkey semi filter already discards whole groups: {pushdown:?}"
    );

    let left_semi_joins = native_join_ids(&optimized.plan, NativeJoinType::LeftSemi);
    assert_eq!(left_semi_joins.len(), 1);
    let (semi_id, semi_join) = left_semi_joins[0];
    assert_eq!(
        aggregate_count_below(&optimized.plan, semi_join.inputs[0]),
        0,
        "no pre-aggregate may run over the unfiltered customer/orders/lineitem rows below the semi filter"
    );
    assert_eq!(join_type_count_below(&optimized.plan, semi_join.inputs[0], NativeJoinType::Inner), 2);
    let final_aggregates = optimized
        .plan
        .nodes()
        .iter()
        .filter(|node| matches!(&node.kind, PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.len() == 5))
        .collect::<Vec<_>>();
    assert_eq!(final_aggregates.len(), 1, "exactly one five-key grouped aggregate should remain");
    assert!(
        node_reachable_below(&optimized.plan, final_aggregates[0].id, semi_id),
        "the five-key aggregate must consume the semi-filtered join output"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q21_decorrelates_to_native_left_only_mixed_residual_joins() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q21");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q21 = items.into_iter().find(|item| item.query_id == "q21").expect("TPC-H workload should include q21");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q21.input else {
        panic!("q21 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let joins = native_joins(result.selected_plan());
    assert!(
        joins.iter().any(|join| q21_left_only_mixed_residual_join_matches(join, NativeJoinType::LeftSemi)),
        "q21 should lower its EXISTS branch to a right-build native left-semi residual join"
    );
    assert!(
        joins.iter().any(|join| q21_left_only_mixed_residual_join_matches(join, NativeJoinType::LEFT_ANTI)),
        "q21 should lower its NOT EXISTS branch to a right-build native left-anti residual join"
    );

    let optimized = optimize_query_plan(
        result.selected_plan().clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    for build_side in [NativeJoinSide::Right, NativeJoinSide::Left] {
        let mut nodes = optimized.plan.nodes().to_vec();
        for node in &mut nodes {
            if let PlanNode::Join(join) = &mut node.kind
                && matches!(join.spec.join_type, NativeJoinType::LeftSemi | NativeJoinType::LeftAnti { .. })
                && join.spec.residual_predicate.is_some()
            {
                join.spec.build_side = build_side;
            }
        }
        let plan = optimized.plan.with_nodes_for_test(nodes).unwrap();
        let decision = admit(
            &plan,
            Default::default(),
            PlanningPolicy::default(),
            rapids_analysis_context(PlanningPolicy::default()),
        )
        .expect("q21 bounded admission should not fail");
        assert!(
            decision.admission().is_some(),
            "the complete optimized q21 pipeline must admit residual semi/anti joins with {build_side:?} build side: {:?}\n{plan}",
            decision.rejection()
        );
    }

    fs::remove_dir_all(dir).unwrap();
}

fn q21_left_only_mixed_residual_join_matches(join: &JoinNode, expected_join_type: NativeJoinType) -> bool {
    join.spec.join_type == expected_join_type
        && join.spec.build_side == NativeJoinSide::Right
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_some()
        && !join.spec.output_projection.is_empty()
        && join.spec.output_projection.iter().all(|column| column.side == NativeJoinSide::Left)
}

fn assert_q02_reassociated_supplier_filter_precedes_partsupp_join(plan: &QueryPlan) {
    assert!(
        plan.nodes().iter().any(|node| {
            let PlanNode::Join(join) = &node.kind else { return false };
            join.spec.keys.len() == 1
                && join.spec.keys[0].left.name() == "s_nationkey"
                && join.spec.keys[0].right.name() == "n_nationkey"
                && source_tables_below(plan, join.inputs[0]) == BTreeSet::from(["supplier".to_string()])
                && source_tables_below(plan, join.inputs[1])
                    == BTreeSet::from(["nation".to_string(), "region".to_string()])
        }),
        "q02 should build a supplier filter join from nation/region before joining partsupp:\n{plan}"
    );
    assert!(
        plan.nodes().iter().any(|node| {
            let PlanNode::Join(join) = &node.kind else { return false };
            join.spec.keys.len() == 1
                && join.spec.keys[0].left.name() == "s_suppkey"
                && join.spec.keys[0].right.name() == "ps_suppkey"
                && source_tables_below(plan, join.inputs[0])
                    == BTreeSet::from(["nation".to_string(), "region".to_string(), "supplier".to_string()])
                && source_tables_below(plan, join.inputs[1]) == BTreeSet::from(["partsupp".to_string()])
        }),
        "q02 should join partsupp only after supplier has been filtered by nation/region:\n{plan}"
    );
}

fn assert_q02_part_partsupp_dynamic_filter_matches_selected_build_side(plan: &QueryPlan) {
    let part_builds_partsupp_join = plan.nodes().iter().any(|node| {
        let PlanNode::Join(join) = &node.kind else { return false };
        join.spec.build_side == NativeJoinSide::Left
            && join.spec.keys.len() == 1
            && join.spec.keys[0].left.name() == "p_partkey"
            && join.spec.keys[0].right.name() == "ps_partkey"
    });
    let (producer_column, consumer_column, consumer_table, expected_lowering) = if part_builds_partsupp_join {
        ("p_partkey", "ps_partkey", "partsupp", DynamicFilterConsumerLowering::PostDecodeOnly)
    } else {
        ("ps_partkey", "p_partkey", "part", DynamicFilterConsumerLowering::ReaderPredicate)
    };

    assert!(
        plan.dynamic_filter_routes().iter().any(|route| {
            route.producer_columns == [producer_column.to_string()]
                && route.consumer_columns == [consumer_column.to_string()]
                && route.consumer_lowering == expected_lowering
                && matches!(
                    route.consumer,
                    DynamicFilterConsumer::LocalParquetSource {
                        source_node_id
                    } if source_tables_below(plan, source_node_id) == BTreeSet::from([consumer_table.to_string()])
                )
        }),
        "q02 should attach a dynamic filter to the part/partsupp key join for the selected build side:\n{plan}"
    );
}

fn native_join_ids(plan: &QueryPlan, join_type: NativeJoinType) -> Vec<(NodeId, &JoinNode)> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Join(join) if join.spec.join_type == join_type => Some((node.id, join)),
            _ => None,
        })
        .collect()
}

fn aggregate_count_below(plan: &QueryPlan, root: NodeId) -> usize {
    plan.nodes()
        .iter()
        .filter(|node| matches!(node.kind, PlanNode::Aggregate(_)) && node_reachable_below(plan, root, node.id))
        .count()
}

fn node_reachable_below(plan: &QueryPlan, root: NodeId, target: NodeId) -> bool {
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        if node_id == target {
            return true;
        }
        if let Some(node) = plan.node(node_id) {
            stack.extend(node.input_ids().iter().copied());
        }
    }
    false
}

fn join_type_count_below(plan: &QueryPlan, root: NodeId, join_type: NativeJoinType) -> usize {
    let mut count = 0;
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        let node = plan.node(node_id).expect("native subtree node should exist");
        if let PlanNode::Join(join) = &node.kind
            && join.spec.join_type == join_type
        {
            count += 1;
        }
        stack.extend(node.input_ids().iter().copied());
    }
    count
}

#[tokio::test]
async fn tpch_q7_lowers_duplicate_nation_names_through_distinct_native_identity_paths() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q7");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q7 = items.into_iter().find(|item| item.query_id == "q07").expect("TPC-H workload should include q07");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q7.input else {
        panic!("q7 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let decision = admit(
        native_plan,
        Default::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("bounded admission should not fail");
    assert!(
        bounded_join_unsupported_shape_facts(
            native_plan,
            &decision,
            rapids_analysis_context(PlanningPolicy::default())
        )
        .is_none(),
        "q07 should use the bounded multi-join native path:\n{native_plan}"
    );
    let nation_projection = native_plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Projection(projection) => {
                let exprs = projection.expressions().map(ToString::to_string).collect::<Vec<_>>();
                if exprs.iter().any(|expr| expr.ends_with(" AS supp_nation"))
                    && exprs.iter().any(|expr| expr.ends_with(" AS cust_nation"))
                {
                    Some(exprs)
                } else {
                    None
                }
            },
            _ => None,
        })
        .expect("q7 should project supplier and customer nation aliases");
    let supp_source = projection_source_for_alias(&nation_projection, "supp_nation");
    let cust_source = projection_source_for_alias(&nation_projection, "cust_nation");
    assert_ne!(supp_source, cust_source, "q7 nation aliases must read distinct native columns: {nation_projection:?}");
    assert!(
        native_plan.nodes().iter().all(|node| node
            .output_schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<BTreeSet<_>>()
            .len()
            == node.output_schema().fields().len()),
        "native executable node schemas must not contain duplicate output names:\n{native_plan}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q8_final_mkt_share_projection_preserves_decimal_schema() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q8");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q8 = items.into_iter().find(|item| item.query_id == "q08").expect("TPC-H workload should include q08");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q8.input else {
        panic!("q8 should produce a physical plan");
    };
    let datafusion_mkt_share_field = physical_projection_field_with_name(plan.as_ref(), "mkt_share");
    let DataType::Decimal128(_, datafusion_scale) = datafusion_mkt_share_field.data_type() else {
        panic!("DataFusion should declare q8 mkt_share as Decimal128: {datafusion_mkt_share_field:?}");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let PlanNode::Sort(sort) = &native_plan.root_node().kind else {
        panic!("q8 native root should be the final ORDER BY sort");
    };
    let projection_node = native_plan.node(sort.input).expect("q8 final sort input should exist");
    let PlanNode::Projection(projection) = &projection_node.kind else {
        panic!("q8 final sort input should be the final projection");
    };
    let mkt_share_output = projection
        .outputs
        .iter()
        .find(|output| output.field.name() == "mkt_share")
        .expect("q8 final projection should include mkt_share");
    let DataType::Decimal128(_, native_scale) = mkt_share_output.field.data_type() else {
        panic!("native q8 mkt_share projection output should be Decimal128: {mkt_share_output:?}");
    };
    assert_eq!(native_scale, datafusion_scale);
    assert_eq!(&mkt_share_output.field, &datafusion_mkt_share_field);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q11_lowers_joined_grouped_and_scalar_aggregate_inputs_for_bounded_admission() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q11");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q11 = items.into_iter().find(|item| item.query_id == "q11").expect("TPC-H workload should include q11");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q11.input else {
        panic!("q11 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("q11 should lower to native");
    };
    assert_eq!(
        candidate.shape.to_string(),
        "SortExec(ProjectionExec(NestedLoopJoinExec(ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),HashJoinExec(LocalParquetScan,LocalParquetScan)))),ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),HashJoinExec(LocalParquetScan,LocalParquetScan)))))))"
    );

    let plan = result.selected_plan();
    assert_q11_keyless_proof_admitted(plan, "lowered q11");
    let decision =
        admit(plan, Default::default(), PlanningPolicy::default(), rapids_analysis_context(PlanningPolicy::default()))
            .expect("bounded admission should not fail");
    assert!(
        bounded_join_unsupported_shape_facts(plan, &decision, rapids_analysis_context(PlanningPolicy::default()))
            .is_none()
    );
    let output_fields = plan.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    assert_eq!(output_fields, vec!["ps_partkey", "value"]);
    assert!(
        output_fields.iter().all(|field| !field.to_ascii_lowercase().contains("threshold")),
        "Q11 scalar threshold must not leak into final output schema"
    );

    let PlanNode::Sort(sort) = &plan.root_node().kind else {
        panic!("q11 native root should be the final ORDER BY sort");
    };
    assert_eq!(sort.spec.expressions.len(), 1);
    assert_eq!(sort.spec.expressions[0].expr.to_string(), "value");
    assert!(sort.spec.expressions[0].descending);

    let final_projection_node = plan.node(sort.input).expect("q11 sort input should exist");
    let PlanNode::Projection(final_projection) = &final_projection_node.kind else {
        panic!("q11 sort input should be the final output projection");
    };
    let final_projection_text = final_projection.expressions().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(final_projection_text.len(), 2);
    assert!(final_projection_text.iter().any(|expr| expr == "ps_partkey"));
    assert!(
        final_projection_text.iter().any(|expr| expr.ends_with(" AS value") || expr == "value"),
        "q11 final projection should expose value without exposing scalar threshold output: {final_projection_text:?}"
    );

    let join_node = plan.node(final_projection.input).expect("q11 final projection input should exist");
    let PlanNode::Join(root_join) = &join_node.kind else {
        panic!("q11 final projection input should be the root conditional join");
    };
    assert_eq!(root_join.spec.join_type, NativeJoinType::Inner);
    assert!(root_join.spec.keys.is_empty());
    assert!(root_join.spec.residual_predicate.is_some());
    assert!(
        root_join.spec.output_projection.iter().all(|column| column.side == NativeJoinSide::Left),
        "q11 root join output projection should be left-only: {:?}",
        root_join.spec.output_projection
    );
    assert_eq!(root_join.spec.output_projection.len(), 2);
    assert_eq!(root_join.spec.output_projection[0].column.to_string(), "input0.ps_partkey");
    assert_eq!(root_join.spec.output_projection[0].output_name, "ps_partkey");
    assert!(
        root_join.spec.output_projection[1].column.to_string().contains("ps_supplycost"),
        "q11 root join should project the grouped value from the left branch: {:?}",
        root_join.spec.output_projection
    );

    let left_aggregate = projection_wrapped_aggregate(plan, root_join.inputs[0], "q11 left input");
    assert_eq!(
        left_aggregate.spec.group_exprs.iter().map(ToString::to_string).collect::<Vec<_>>(),
        vec!["ps_partkey".to_string()]
    );
    assert_eq!(left_aggregate.spec.aggr_exprs.len(), 1);
    assert_eq!(left_aggregate.spec.aggr_exprs[0].function, AggregateFunction::Sum);
    assert_eq!(
        source_tables_below(plan, left_aggregate.input),
        BTreeSet::from(["nation".to_string(), "partsupp".to_string(), "supplier".to_string()])
    );

    let right_aggregate = projection_wrapped_aggregate(plan, root_join.inputs[1], "q11 right input");
    assert!(right_aggregate.spec.group_exprs.is_empty(), "q11 right branch should be a scalar/global aggregate");
    assert_eq!(right_aggregate.spec.aggr_exprs.len(), 1);
    assert_eq!(right_aggregate.spec.aggr_exprs[0].function, AggregateFunction::Sum);
    assert_eq!(
        source_tables_below(plan, right_aggregate.input),
        BTreeSet::from(["nation".to_string(), "partsupp".to_string(), "supplier".to_string()])
    );

    let optimized = optimize_query_plan(
        plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert_q11_keyless_proof_admitted(&optimized.plan, "optimized q11");
    let aggregate_join_pushdown = &optimized.diagnostics.aggregate_join_pushdown;
    assert_eq!(aggregate_join_pushdown.applied_count, 1, "optimizer diagnostics={aggregate_join_pushdown:?}");
    assert!(
        aggregate_join_pushdown.candidates.iter().any(|candidate| candidate.applied),
        "q11 grouped aggregate-over-join input should be reduced before its join: {aggregate_join_pushdown:?}"
    );
    assert!(
        aggregate_join_pushdown
            .candidates
            .iter()
            .any(|candidate| candidate.rejected_reason == Some(AggregateJoinPushdownRejectedReason::InputNotJoin)),
        "q11 scalar threshold aggregate should be derived from the grouped branch before aggregate-join pushdown: {aggregate_join_pushdown:?}"
    );
    let preaggregates = optimized
        .plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate)
                if aggregate
                    .spec
                    .aggr_exprs
                    .iter()
                    .any(|expr| expr.alias.starts_with("__algeon_native_preagg_sum_")) =>
            {
                Some(aggregate)
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        preaggregates.len(),
        1,
        "optimized q11 should keep one shared SUM pre-aggregate leaf:\n{}",
        optimized.plan
    );
    assert!(
        preaggregates.iter().all(|aggregate| {
            !aggregate.spec.group_exprs.is_empty()
                && aggregate.spec.grouping_sets.is_none()
                && aggregate
                    .spec
                    .aggr_exprs
                    .iter()
                    .all(|expr| expr.function == AggregateFunction::Sum && !expr.distinct && expr.filter.is_none())
        }),
        "q11 pre-aggregate leaves should stay grouped unfiltered SUMs:\n{}",
        optimized.plan
    );
    let scalar_aggregate = optimized
        .plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => Some(aggregate),
            _ => None,
        })
        .expect("optimized q11 should keep a scalar threshold aggregate");
    assert_eq!(scalar_aggregate.spec.aggr_exprs.len(), 1);
    assert_eq!(
        scalar_aggregate.spec.aggr_exprs[0].expr,
        Expr::column("sum(partsupp.ps_supplycost * partsupp.ps_availqty)")
    );

    fs::remove_dir_all(dir).unwrap();
}

fn assert_q11_keyless_proof_admitted(plan: &QueryPlan, context: &str) {
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let decision =
        admit(plan, Default::default(), PlanningPolicy::default(), rapids_analysis_context(PlanningPolicy::default()))
            .expect("bounded admission should not fail");
    assert!(
        decision.admission().is_some(),
        "{context} should be bounded executable: {:?}\n{plan}",
        decision.rejection()
    );
    // The keyless scalar-fanout cardinality proof survives as a node fact, not
    // a plan-family reason: q11's scalar threshold aggregate emits at most one
    // row by its operator alone.
    let scalar_aggregate_id = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => Some(node.id),
            _ => None,
        })
        .expect("q11 should keep a scalar threshold aggregate");
    let scalar_capability = analysis
        .nodes
        .iter()
        .find(|node| node.node_id == scalar_aggregate_id)
        .expect("scalar threshold aggregate should have a node capability");
    assert_eq!(
        scalar_capability.max_output_rows,
        Some(1),
        "{context} scalar threshold aggregate should prove at most one row"
    );
}

#[tokio::test]
async fn tpch_q13_lowers_left_join_filter_aggregates_and_final_sort() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q13");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q13 = items.into_iter().find(|item| item.query_id == "q13").expect("TPC-H workload should include q13");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q13.input else {
        panic!("q13 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let plan = result.selected_plan();
    let joins = native_joins(plan);
    assert_eq!(joins.len(), 1, "q13 should lower to exactly one native join");
    let join = joins[0];
    assert_eq!(join.spec.join_type, NativeJoinType::Left);
    assert_eq!(join.spec.build_side, NativeJoinSide::Right);
    assert_eq!(join.spec.keys.len(), 1);
    assert_eq!(join.spec.keys[0].left.to_string(), "input0.c_custkey");
    assert_eq!(join.spec.keys[0].right.to_string(), "input1.o_custkey");
    assert!(
        join.spec.residual_predicate.is_none(),
        "q13 orders-side NOT LIKE should lower as a pre-join filter, not a join residual"
    );

    let right_input = plan.node(join.inputs[1]).expect("q13 join right input should exist");
    let PlanNode::Projection(right_projection) = &right_input.kind else {
        panic!("q13 join right input should be the filtered orders projection");
    };
    let filter_input = plan.node(right_projection.input).expect("q13 orders filter should exist");
    let PlanNode::Filter(filter) = &filter_input.kind else {
        panic!("q13 orders-side NOT LIKE should be a Filter node before the join");
    };
    assert!(matches!(
        &filter.predicate,
        Expr::Like { expr, pattern, escape, negated }
            if *negated
                && pattern == "%special%requests%"
                && escape.as_deref() == Some("\\")
                && matches!(expr.as_ref(), Expr::Column(column) if column.to_string() == "o_comment")
    ));

    let aggregates = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) => Some(aggregate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(aggregates.len(), 2, "q13 should lower both GROUP BY levels");
    assert_eq!(aggregates[0].spec.group_exprs.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 1);
    assert_eq!(aggregates[1].spec.group_exprs.len(), 1);
    assert_eq!(aggregates[1].spec.aggr_exprs.len(), 1);

    let PlanNode::Projection(proj) = &plan.root_node().kind else {
        panic!("q13 native root should be the final projection: {plan}");
    };
    let sort_node = plan.node(proj.input).expect("q13 sort input should exist");
    let PlanNode::Sort(sort) = &sort_node.kind else {
        panic!("q13 native projection input should be the final sort: {plan}");
    };
    assert_eq!(sort.spec.expressions.len(), 2);
    assert_eq!(sort.spec.expressions[0].expr.to_string(), "count(Int64(1))");
    assert!(sort.spec.expressions[0].descending);
    assert_eq!(sort.spec.expressions[1].expr.to_string(), "c_count");
    assert!(sort.spec.expressions[1].descending);
    assert_eq!(proj.outputs.len(), 2);
    assert_eq!(proj.outputs[0].field.name(), "c_count");
    assert_eq!(proj.outputs[1].field.name(), "custdist");

    let optimized = optimize_query_plan(
        plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    let aggregate_join_pushdown = &optimized.diagnostics.aggregate_join_pushdown;
    assert_eq!(
        aggregate_join_pushdown.applied_count, 1,
        "q13 should pre-aggregate orders counts before the left join: {aggregate_join_pushdown:?}"
    );
    let optimized_left_joins = native_join_ids(&optimized.plan, NativeJoinType::Left);
    assert_eq!(optimized_left_joins.len(), 1, "optimized q13 should still contain one semantic left join");
    let (_join_id, optimized_join) = optimized_left_joins[0];
    let optimized_right_input =
        optimized.plan.node(optimized_join.inputs[1]).expect("optimized q13 right join input should exist");
    let PlanNode::Aggregate(pre_aggregate) = &optimized_right_input.kind else {
        panic!("optimized q13 should pre-aggregate the orders side before the left join:\n{}", optimized.plan);
    };
    assert_eq!(
        pre_aggregate.spec.group_exprs.iter().map(ToString::to_string).collect::<Vec<_>>(),
        vec!["o_custkey AS o_custkey".to_string()]
    );
    assert_eq!(pre_aggregate.spec.aggr_exprs.len(), 1);
    assert_eq!(pre_aggregate.spec.aggr_exprs[0].function, AggregateFunction::Count);
    assert_eq!(pre_aggregate.spec.aggr_exprs[0].expr.to_string(), "o_custkey");
    assert_eq!(optimized.diagnostics.count_non_null_column_pruning.applied_count, 1);
    let optimized_orders_source = native_local_parquet_sources(&optimized.plan)
        .into_iter()
        .find(|source| {
            source.read_columns.iter().any(|column| column == "o_custkey")
                && source.read_columns.iter().any(|column| column == "o_comment")
        })
        .expect("optimized q13 should retain the filtered orders source");
    assert!(
        !optimized_orders_source.read_columns.iter().any(|column| column == "o_orderkey"),
        "footer-proven COUNT(o_orderkey) should not force reading o_orderkey after COUNT(1) rewrite"
    );
    assert!(
        optimized.plan.nodes().iter().any(|node| match &node.kind {
            PlanNode::Projection(projection) => projection.outputs.iter().any(|output| {
                matches!(
                    &output.expr,
                    Expr::Alias { expr, alias }
                        if alias.starts_with("__algeon_native_preagg_sum_") && matches!(expr.as_ref(), Expr::Case { .. })
                )
            }),
            _ => false,
        }),
        "optimized q13 should coalesce unmatched left-join counts to zero before the final aggregate:\n{}",
        optimized.plan
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q16_lowers_multi_join_leaves_then_serial_aggregates_and_final_sort() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-tpch-q16");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q16 = items.into_iter().find(|item| item.query_id == "q16").expect("TPC-H workload should include q16");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q16.input else {
        panic!("q16 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("q16 should lower to native");
    };
    assert_eq!(
        candidate.shape.to_string(),
        "ProjectionExec(SortExec(AggregateExec(AggregateExec(HashJoinExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan),FilterExec(LocalParquetScan))))))"
    );

    let plan = result.selected_plan();
    assert!(
        admit(plan, Default::default(), PlanningPolicy::default(), rapids_analysis_context(PlanningPolicy::default()))
            .expect("q16 bounded admission should not fail")
            .admission()
            .is_some(),
        "q16 should be admitted as one grouped aggregate pipeline:\n{plan}"
    );
    let optimized = optimize_query_plan(
        plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert!(
        admit(
            &optimized.plan,
            Default::default(),
            PlanningPolicy::default(),
            rapids_analysis_context(PlanningPolicy::default())
        )
        .expect("optimized q16 bounded admission should not fail")
        .admission()
        .is_some(),
        "optimized q16 should remain admitted as one grouped aggregate pipeline:\n{}",
        optimized.plan
    );
    let PlanNode::Projection(projection) = &plan.root_node().kind else {
        panic!("q16 native root should be the final projection: {plan}");
    };
    let sort_node = plan.node(projection.input).expect("q16 sort input should exist");
    let PlanNode::Sort(sort) = &sort_node.kind else {
        panic!("q16 projection input should be sort: {plan}");
    };
    assert_eq!(sort.spec.expressions.len(), 4);
    assert_eq!(sort.spec.expressions[0].expr.to_string(), "count(alias1)");
    assert!(sort.spec.expressions[0].descending);
    assert_eq!(sort.spec.expressions[1].expr.to_string(), "p_brand");
    assert!(!sort.spec.expressions[1].descending);
    assert_eq!(sort.spec.expressions[2].expr.to_string(), "p_type");
    assert!(!sort.spec.expressions[2].descending);
    assert_eq!(sort.spec.expressions[3].expr.to_string(), "p_size");
    assert!(!sort.spec.expressions[3].descending);
    assert_eq!(projection.outputs.len(), 4);
    assert_eq!(projection.outputs[3].field.name(), "supplier_cnt");

    let outer_aggregate_node = plan.node(sort.input).expect("q16 sort input should exist");
    let PlanNode::Aggregate(outer_aggregate) = &outer_aggregate_node.kind else {
        panic!("q16 sort input should be the outer aggregate: {plan}");
    };
    let inner_aggregate_id = skip_projection_nodes(plan, outer_aggregate.input);
    let inner_aggregate_node = plan.node(inner_aggregate_id).expect("q16 outer aggregate input should exist");
    let PlanNode::Aggregate(inner_aggregate) = &inner_aggregate_node.kind else {
        panic!("q16 should lower DataFusion's distinct/grouping stack as serial aggregate nodes");
    };
    let join_stack_id = skip_projection_nodes(plan, inner_aggregate.input);
    let join_stack_node = plan.node(join_stack_id).expect("q16 inner aggregate input should exist");
    assert!(
        matches!(join_stack_node.kind, PlanNode::Join(_)),
        "q16 aggregate stack should sit above the multi-join pipeline"
    );

    let aggregates = native_aggregate_nodes(plan);
    assert_eq!(aggregates.len(), 2, "q16 should lower only the serial aggregate/distinct stack above joins");
    let joins = native_joins(plan);
    assert_eq!(joins.len(), 2, "q16 should lower the part/partsupp join plus anti supplier branch");
    assert!(joins.iter().any(|join| join.spec.join_type == NativeJoinType::Inner));
    assert!(joins.iter().any(|join| join.spec.join_type == NativeJoinType::LEFT_ANTI));
    for join in joins {
        for input in join.inputs {
            assert_q16_join_leaf_subtree_has_no_aggregate_sort_or_limit(plan, input);
        }
    }

    let sources = native_local_parquet_sources(plan);
    assert_eq!(sources.len(), 3, "q16 join leaves should be local Parquet sources");
    assert_eq!(native_filter_exprs(plan).len(), 2, "q16 should keep part and supplier predicates as leaf filters");

    fs::remove_dir_all(dir).unwrap();
}

fn assert_q16_join_leaf_subtree_has_no_aggregate_sort_or_limit(plan: &QueryPlan, root: NodeId) {
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        let node = plan.node(node_id).expect("q16 native join subtree node should exist");
        match &node.kind {
            PlanNode::Aggregate(_) => panic!("q16 join leaf subtree must not contain aggregate nodes below joins"),
            PlanNode::Window(_) => panic!("q16 join leaf subtree must not contain window nodes below joins"),
            PlanNode::Sort(_) => panic!("q16 join leaf subtree must not contain sort nodes below joins"),
            PlanNode::Limit(_) => panic!("q16 join leaf subtree must not contain limit nodes below joins"),
            PlanNode::GraphAlgorithm(_) => {
                panic!("q16 join leaf subtree must not contain graph algorithm nodes below joins")
            },
            PlanNode::VectorAlgorithm(_) | PlanNode::MlAlgorithm(_) => {
                panic!("q16 join leaf subtree must not contain dense algorithm nodes below joins")
            },
            PlanNode::EdgeNormalize(_) => {
                panic!("q16 join leaf subtree must not contain edge normalize nodes below joins")
            },
            PlanNode::Source(_)
            | PlanNode::Filter(_)
            | PlanNode::Projection(_)
            | PlanNode::Join(_)
            | PlanNode::Union(_) => {
                stack.extend(node.input_ids().iter().copied());
            },
        }
    }
}

fn skip_projection_nodes(plan: &QueryPlan, mut node_id: NodeId) -> NodeId {
    while let Some(node) = plan.node(node_id) {
        let PlanNode::Projection(projection) = &node.kind else {
            break;
        };
        node_id = projection.input;
    }
    node_id
}

fn projection_wrapped_aggregate<'a>(plan: &'a QueryPlan, node_id: NodeId, context: &str) -> &'a AggregateNode {
    let projection_node = plan.node(node_id).unwrap_or_else(|| panic!("{context} projection should exist"));
    let PlanNode::Projection(projection) = &projection_node.kind else {
        panic!("{context} should be projection-wrapped");
    };
    let aggregate_node = plan.node(projection.input).unwrap_or_else(|| panic!("{context} aggregate should exist"));
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        panic!("{context} projection input should be an aggregate");
    };
    aggregate
}

fn source_tables_below(plan: &QueryPlan, root: NodeId) -> BTreeSet<String> {
    let mut tables = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        let node = plan.node(node_id).expect("native subtree node should exist");
        match &node.kind {
            PlanNode::Source(source_node) => {
                if let SourceNode::LocalParquet(source) = source_node.as_ref()
                    && let Some(table) = local_parquet_source_table(source)
                {
                    tables.insert(table.to_string());
                }
            },
            PlanNode::Filter(_)
            | PlanNode::Projection(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Join(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_)
            | PlanNode::MlAlgorithm(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::Union(_) => stack.extend(node.input_ids().iter().copied()),
        }
    }
    tables
}

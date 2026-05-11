use super::*;

#[path = "tpch_shapes/q22.rs"]
mod q22;

#[tokio::test]
async fn tpch_recent_bounded_topologies_expose_modeled_facts() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-tpch-recent-topologies");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();

    let q2 = items.iter().find(|item| item.query_id == "q02").expect("TPC-H workload should include q02");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q2.input else {
        panic!("q02 should produce a physical plan");
    };
    let result = lower_physical_plan_to_native(Arc::clone(plan));
    assert_eq!(result.rewritten_count(), 1, "q02 outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    let analysis = native_plan.capability_analysis();
    assert!(
        native_plan.has_bounded_executable_multi_join_aggregate_over_join_subplan_topk_pipeline(),
        "q02 should lower to the executable aggregate-over-join subplan TopK pipeline: {:?}\n{native_plan}",
        analysis.multi_join_aggregate_over_join_subplan_topk_admission,
    );
    assert!(
        analysis.admits_multi_join_aggregate_over_join_subplan_topk_pipeline(),
        "q02 should be admitted by the capability proof:\n{native_plan}"
    );
    let facts = native_plan
        .bounded_join_modeled_facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .expect("q02 should expose modeled bounded join facts");
    assert_eq!(
        facts.get("bounded_join_execution_family").map(String::as_str),
        Some("multi_join_with_aggregate_over_join_subplan_topk")
    );

    let q15 = items.iter().find(|item| item.query_id == "q15").expect("TPC-H workload should include q15");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q15.input else {
        panic!("q15 should produce a physical plan");
    };
    let result = lower_physical_plan_to_native(Arc::clone(plan));
    assert_eq!(result.rewritten_count(), 1, "q15 outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    assert!(
        native_plan.has_bounded_executable_keyed_hash_join_resident_subplan_pipeline(),
        "q15 should lower to the executable keyed hash join resident subplan pipeline:\n{native_plan}"
    );
    let facts = native_plan
        .bounded_join_modeled_facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .expect("q15 should expose modeled bounded join facts");
    assert_eq!(
        facts.get("bounded_join_execution_family").map(String::as_str),
        Some("keyed_hash_join_resident_subplan_pipeline")
    );

    let q18 = items.iter().find(|item| item.query_id == "q18").expect("TPC-H workload should include q18");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q18.input else {
        panic!("q18 should produce a physical plan");
    };
    let result = lower_physical_plan_to_native(Arc::clone(plan));
    assert_eq!(result.rewritten_count(), 1, "q18 outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    assert!(
        native_plan
            .has_bounded_executable_hash_join_resident_grouped_aggregate_input_post_join_aggregate_topk_pipeline(),
        "q18 should lower to the executable resident grouped aggregate TopK pipeline:\n{native_plan}"
    );
    let facts = native_plan
        .bounded_join_modeled_facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .expect("q18 should expose modeled bounded join facts");
    assert_eq!(
        facts.get("bounded_join_execution_family").map(String::as_str),
        Some("hash_join_with_resident_grouped_aggregate_input_post_join_aggregate_topk")
    );

    let q20 = items.iter().find(|item| item.query_id == "q20").expect("TPC-H workload should include q20");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q20.input else {
        panic!("q20 should produce a physical plan");
    };
    let result = lower_physical_plan_to_native(Arc::clone(plan));
    assert_eq!(result.rewritten_count(), 1, "q20 outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    assert!(
        native_plan.has_bounded_executable_multi_join_resident_grouped_aggregate_subquery_filter_pipeline(),
        "q20 should lower to the executable resident grouped aggregate subquery-filter pipeline:\n{native_plan}"
    );
    let facts = native_plan
        .bounded_join_modeled_facts(cudf_query_engine::ExecutionBoundaryPolicy::default())
        .expect("q20 should expose modeled bounded join facts");
    assert_eq!(
        facts.get("bounded_join_execution_family").map(String::as_str),
        Some("multi_join_with_resident_grouped_aggregate_subquery_filter")
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q21_decorrelates_to_native_left_only_mixed_residual_joins() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-tpch-q21");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q21 = items.into_iter().find(|item| item.query_id == "q21").expect("TPC-H workload should include q21");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q21.input else {
        panic!("q21 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let joins = native_joins(&result.plans[0]);
    assert!(
        joins.iter().any(|join| q21_left_only_mixed_residual_join_matches(join, NativeJoinType::LeftSemi)),
        "q21 should lower its EXISTS branch to a right-build native left-semi residual join"
    );
    assert!(
        joins.iter().any(|join| q21_left_only_mixed_residual_join_matches(join, NativeJoinType::LeftAnti)),
        "q21 should lower its NOT EXISTS branch to a right-build native left-anti residual join"
    );

    fs::remove_dir_all(dir).unwrap();
}

fn q21_left_only_mixed_residual_join_matches(
    join: &cudf_query_engine::JoinNode,
    expected_join_type: NativeJoinType,
) -> bool {
    join.spec.join_type == expected_join_type
        && join.spec.build_side == NativeJoinSide::Right
        && !join.spec.keys.is_empty()
        && join.spec.residual_predicate.is_some()
        && !join.spec.output_projection.is_empty()
        && join.spec.output_projection.iter().all(|column| column.side == NativeJoinSide::Left)
}

#[tokio::test]
async fn tpch_q7_lowers_duplicate_nation_names_through_distinct_native_identity_paths() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-tpch-q7");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q7 = items.into_iter().find(|item| item.query_id == "q07").expect("TPC-H workload should include q07");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q7.input else {
        panic!("q7 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-tpch-q8");
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

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
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
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-tpch-q11");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q11 = items.into_iter().find(|item| item.query_id == "q11").expect("TPC-H workload should include q11");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q11.input else {
        panic!("q11 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("q11 should lower to native");
    };
    assert_eq!(
        candidate.shape,
        "SortExec(ProjectionExec(NestedLoopJoinExec(ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),ProjectionExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))),ProjectionExec(AggregateExec(HashJoinExec(FilterExec(LocalParquetScan),ProjectionExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))))))"
    );

    let plan = &result.plans[0];
    assert!(plan.has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline());
    assert!(plan.bounded_join_unsupported_shape_facts(cudf_query_engine::ExecutionBoundaryPolicy::default()).is_none());
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

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q13_lowers_left_join_filter_aggregates_and_final_sort() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-tpch-q13");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q13 = items.into_iter().find(|item| item.query_id == "q13").expect("TPC-H workload should include q13");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q13.input else {
        panic!("q13 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let plan = &result.plans[0];
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

    let PlanNode::Sort(sort) = &plan.root_node().kind else {
        panic!("q13 native root should be the final ORDER BY sort");
    };
    assert_eq!(sort.spec.expressions.len(), 2);
    assert_eq!(sort.spec.expressions[0].expr.to_string(), "custdist");
    assert!(sort.spec.expressions[0].descending);
    assert_eq!(sort.spec.expressions[1].expr.to_string(), "c_count");
    assert!(sort.spec.expressions[1].descending);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_q16_lowers_multi_join_leaves_then_serial_aggregates_and_final_sort() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-tpch-q16");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q16 = items.into_iter().find(|item| item.query_id == "q16").expect("TPC-H workload should include q16");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q16.input else {
        panic!("q16 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("q16 should lower to native");
    };
    assert_eq!(
        candidate.shape,
        "SortExec(ProjectionExec(AggregateExec(AggregateExec(HashJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),FilterExec(LocalParquetScan))))))"
    );

    let plan = &result.plans[0];
    let PlanNode::Sort(sort) = &plan.root_node().kind else {
        panic!("q16 native root should be the final ORDER BY sort");
    };
    let projection_node = plan.node(sort.input).expect("q16 sort input should exist");
    let PlanNode::Projection(projection) = &projection_node.kind else {
        panic!("q16 sort input should be the final projection");
    };
    let outer_aggregate_node = plan.node(projection.input).expect("q16 projection input should exist");
    let PlanNode::Aggregate(outer_aggregate) = &outer_aggregate_node.kind else {
        panic!("q16 projection input should be the outer aggregate");
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
    assert!(joins.iter().any(|join| join.spec.join_type == NativeJoinType::LeftAnti));
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
            PlanNode::Sort(_) => panic!("q16 join leaf subtree must not contain sort nodes below joins"),
            PlanNode::Limit(_) => panic!("q16 join leaf subtree must not contain limit nodes below joins"),
            PlanNode::Source(_) | PlanNode::Filter(_) | PlanNode::Projection(_) | PlanNode::Join(_) => {
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

fn projection_wrapped_aggregate<'a>(
    plan: &'a QueryPlan,
    node_id: NodeId,
    context: &str,
) -> &'a cudf_query_engine::AggregateNode {
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
            | PlanNode::Sort(_)
            | PlanNode::Limit(_)
            | PlanNode::Join(_) => stack.extend(node.input_ids().iter().copied()),
        }
    }
    tables
}

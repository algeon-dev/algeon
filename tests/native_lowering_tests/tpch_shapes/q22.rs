use super::*;

#[tokio::test]
async fn tpch_q22_lowers_scalar_avg_anti_join_conditional_join_and_final_grouping() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-tpch-q22");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q22 = items.into_iter().find(|item| item.query_id == "q22").expect("TPC-H workload should include q22");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = q22.input else {
        panic!("q22 should produce a physical plan");
    };

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("q22 should lower to native");
    };
    assert_eq!(
        candidate.shape,
        "SortExec(ProjectionExec(AggregateExec(ProjectionExec(NestedLoopJoinExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),LocalParquetScan)),AggregateExec(FilterExec(LocalParquetScan)))))))"
    );

    let plan = &result.plans[0];
    let output_fields = plan.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>();
    assert_eq!(output_fields, vec!["cntrycode", "numcust", "totacctbal"]);
    assert!(
        output_fields.iter().all(|field| !field.to_ascii_lowercase().contains("avg")),
        "scalar AVG branch must not leak into the final Q22 output schema"
    );
    assert!(plan.to_string().contains("SUBSTRING(c_phone FROM 1 FOR 2)"));
    assert!(!plan.to_string().contains("SUBSTRING(c_phone FROM c_custkey"));

    let PlanNode::Sort(sort) = &plan.root_node().kind else {
        panic!("q22 native root should be the final ORDER BY sort");
    };
    assert_eq!(sort.spec.expressions.len(), 1);
    assert_eq!(sort.spec.expressions[0].expr.to_string(), "cntrycode");
    assert!(!sort.spec.expressions[0].descending);

    let final_projection_node = plan.node(sort.input).expect("q22 sort input should exist");
    let PlanNode::Projection(final_projection) = &final_projection_node.kind else {
        panic!("q22 sort input should be the final output projection");
    };
    assert_eq!(final_projection.outputs.len(), 3);
    let final_projection_text = final_projection.expressions().map(ToString::to_string).collect::<Vec<_>>();
    for expected_output in ["cntrycode", "numcust", "totacctbal"] {
        assert!(
            final_projection_text.iter().any(|expr| expr.contains(expected_output)),
            "q22 final projection should expose {expected_output}: {final_projection_text:?}"
        );
    }
    let final_aggregate_node = plan.node(final_projection.input).expect("q22 final projection input should exist");
    let PlanNode::Aggregate(final_aggregate) = &final_aggregate_node.kind else {
        panic!("q22 final projection input should be the GROUP BY aggregate");
    };
    let mut aggregate_input_projection_chain = Vec::new();
    let mut current = final_aggregate.input;
    while let Some(node) = plan.node(current) {
        let PlanNode::Projection(projection) = &node.kind else {
            break;
        };
        aggregate_input_projection_chain.push(projection.expressions().map(ToString::to_string).collect::<Vec<_>>());
        current = projection.input;
    }
    assert!(
        aggregate_input_projection_chain.iter().any(|expressions| {
            expressions.len() == 2
                && expressions.iter().any(|expr| expr == "SUBSTRING(c_phone FROM 1 FOR 2) AS cntrycode")
                && expressions.iter().any(|expr| expr == "c_acctbal")
        }),
        "q22 scalar comparison output should be explicitly projected to cntrycode and c_acctbal before grouping: {aggregate_input_projection_chain:?}"
    );
    assert_eq!(
        final_aggregate.spec.group_exprs.iter().map(ToString::to_string).collect::<Vec<_>>(),
        vec!["cntrycode".to_string()]
    );
    assert_eq!(final_aggregate.spec.aggr_exprs.len(), 2);
    let final_aggrs = final_aggregate
        .spec
        .aggr_exprs
        .iter()
        .map(|expr| (expr.function, expr.expr.to_string(), expr.alias.clone()))
        .collect::<Vec<_>>();
    assert!(
        final_aggrs.iter().any(|(function, _expr, _alias)| *function == AggregateFunction::Count),
        "q22 final aggregate should include COUNT(*): {final_aggrs:?}"
    );
    assert!(
        final_aggrs.iter().any(|(function, _expr, alias)| {
            *function == AggregateFunction::Sum && alias.to_ascii_lowercase().contains("c_acctbal")
        }),
        "q22 final aggregate should include SUM(c_acctbal), possibly via a synthetic aggregate argument: {final_aggrs:?}"
    );

    let aggregates = native_aggregate_nodes(plan);
    let scalar_avg = aggregates
        .iter()
        .find(|aggregate| {
            aggregate.spec.group_exprs.is_empty()
                && aggregate.spec.aggr_exprs.len() == 1
                && aggregate.spec.aggr_exprs[0].function == AggregateFunction::Avg
        })
        .expect("q22 scalar subquery should lower to a global AVG aggregate");
    assert_eq!(scalar_avg.spec.aggr_exprs[0].expr.to_string(), "c_acctbal");
    let scalar_avg_input = skip_projection_nodes(plan, scalar_avg.input);
    let scalar_avg_filter_node = plan.node(scalar_avg_input).expect("q22 scalar AVG input should exist");
    let PlanNode::Filter(scalar_avg_filter) = &scalar_avg_filter_node.kind else {
        panic!("q22 scalar AVG should read from a filtered customer source");
    };
    assert!(scalar_avg_filter.predicate.to_string().contains("c_acctbal"));
    assert!(scalar_avg_filter.predicate.to_string().contains("SUBSTRING(c_phone FROM 1 FOR 2)"));
    let scalar_avg_source_node = plan.node(scalar_avg_filter.input).expect("q22 scalar AVG source should exist");
    let PlanNode::Source(scalar_avg_source) = &scalar_avg_source_node.kind else {
        panic!("q22 scalar AVG filter input should be a source");
    };
    assert!(matches!(
        scalar_avg_source.as_ref(),
        SourceNode::LocalParquet(source) if local_parquet_source_table(source) == Some("customer")
    ));

    let joins = native_joins(plan);
    assert_eq!(joins.len(), 2, "q22 should lower to the anti join plus the scalar comparison join");
    assert!(joins.iter().all(|join| join.spec.join_type != NativeJoinType::Cross));
    let anti_join = joins
        .iter()
        .find(|join| join.spec.join_type == NativeJoinType::LeftAnti)
        .expect("q22 NOT EXISTS orders branch should lower to a left-anti hash join");
    assert_eq!(anti_join.spec.build_side, NativeJoinSide::Right);
    assert_eq!(anti_join.spec.keys.len(), 1);
    assert_eq!(anti_join.spec.keys[0].left.to_string(), "input0.c_custkey");
    assert_eq!(anti_join.spec.keys[0].right.to_string(), "input1.o_custkey");
    assert!(anti_join.spec.residual_predicate.is_none());
    assert!(!anti_join.spec.output_projection.is_empty());
    assert!(anti_join.spec.output_projection.iter().all(|column| column.side == NativeJoinSide::Left));

    let conditional_join = joins
        .iter()
        .find(|join| join.spec.keys.is_empty() && join.spec.residual_predicate.is_some())
        .expect("q22 scalar comparison should lower to a keyless conditional join");
    assert_eq!(conditional_join.spec.join_type, NativeJoinType::Inner);
    assert_join_output_projection(
        &conditional_join.spec.output_projection,
        &[(NativeJoinSide::Left, "input0.c_phone", "c_phone"), (NativeJoinSide::Left, "input0.c_acctbal", "c_acctbal")],
    );

    let sources = native_local_parquet_sources(plan);
    assert_eq!(sources.len(), 3, "q22 should lower outer customer, orders, and scalar AVG customer sources");
    assert_eq!(result.source_diagnostics.len(), 3);
    let customer_sources =
        sources.iter().filter(|source| local_parquet_source_table(source) == Some("customer")).collect::<Vec<_>>();
    assert_eq!(customer_sources.len(), 2);
    let outer_customer = customer_sources
        .iter()
        .copied()
        .find(|source| source.read_columns.iter().any(|column| column == "c_custkey"))
        .expect("q22 outer customer source should read c_custkey for the anti join");
    assert_eq!(
        outer_customer.read_columns,
        vec!["c_acctbal".to_string(), "c_custkey".to_string(), "c_phone".to_string()]
    );
    let scalar_avg_customer = customer_sources
        .iter()
        .copied()
        .find(|source| !source.read_columns.iter().any(|column| column == "c_custkey"))
        .expect("q22 scalar AVG customer source should not read c_custkey");
    assert_eq!(scalar_avg_customer.read_columns, vec!["c_acctbal".to_string(), "c_phone".to_string()]);
    let orders_source = sources
        .iter()
        .find(|source| local_parquet_source_table(source) == Some("orders"))
        .expect("q22 orders source should be present");
    assert_eq!(orders_source.read_columns, vec!["o_custkey".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

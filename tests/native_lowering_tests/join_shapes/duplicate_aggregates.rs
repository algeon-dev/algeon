use super::*;

#[test]
fn duplicate_named_root_aggregate_uses_unique_executable_schema_and_restores_host_sink_schema() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-duplicate-root-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = duplicate_named_root_aggregate_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    let aggregate_node = native_plan
        .nodes()
        .iter()
        .find(|node| matches!(node.kind, PlanNode::Aggregate(_)))
        .expect("native plan should contain root aggregate node");
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        panic!("native plan should contain root aggregate node");
    };
    assert!(
        matches!(&aggregate.spec.group_exprs[0], Expr::Alias { alias, .. } if alias == "__nexus_native_field_0_dup"),
        "aggregate group output alias must use executable schema: {:?}",
        aggregate.spec.group_exprs
    );
    assert_eq!(aggregate.spec.aggr_exprs[0].alias, "__nexus_native_field_1_dup");
    assert_eq!(
        aggregate_node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["__nexus_native_field_0_dup", "__nexus_native_field_1_dup"]
    );
    assert_eq!(
        native_plan.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["dup", "dup"]
    );
    assert!(
        native_plan.nodes().iter().all(|node| node
            .output_schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<BTreeSet<_>>()
            .len()
            == node.output_schema().fields().len()),
        "native executable node schemas must be unique:\n{native_plan}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duplicate_group_alias_root_aggregate_uses_unique_synthetic_input_projection() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-duplicate-root-group-aliases");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = duplicate_group_alias_root_aggregate_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    let aggregate_node = native_plan
        .nodes()
        .iter()
        .find(|node| matches!(node.kind, PlanNode::Aggregate(_)))
        .expect("native plan should contain root aggregate node");
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        panic!("native plan should contain root aggregate node");
    };
    let projection_node = native_plan.node(aggregate.input).expect("aggregate input projection should exist");
    let PlanNode::Projection(projection) = &projection_node.kind else {
        panic!("aggregate input should be the synthetic projection");
    };
    assert_eq!(
        projection_node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["dup", "__nexus_native_group_expr_1_dup", "__nexus_native_aggregate_arg_0"]
    );
    assert!(
        projection.expressions().any(|expr| {
            matches!(
                expr,
                Expr::Alias { expr, alias }
                    if alias == "__nexus_native_group_expr_1_dup"
                        && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "payload")
            )
        }),
        "second duplicate group alias should be materialized under a unique executable name: {:?}",
        projection.expressions().collect::<Vec<_>>()
    );
    assert!(
        matches!(
            &aggregate.spec.group_exprs[0],
            Expr::Alias { expr, alias }
                if alias == "__nexus_native_field_0_dup"
                    && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "dup")
        ),
        "first group key should read the first synthetic group column and output its executable aggregate name: {:?}",
        aggregate.spec.group_exprs
    );
    assert!(
        matches!(
            &aggregate.spec.group_exprs[1],
            Expr::Alias { expr, alias }
                if alias == "__nexus_native_field_1_dup"
                    && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "__nexus_native_group_expr_1_dup")
        ),
        "second group key should read the unique synthetic group column and output its executable aggregate name: {:?}",
        aggregate.spec.group_exprs
    );
    assert_eq!(
        aggregate_node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["__nexus_native_field_0_dup", "__nexus_native_field_1_dup", "sum_m00"]
    );
    assert_eq!(
        native_plan.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["dup", "dup", "sum_m00"]
    );
    assert!(
        native_plan.nodes().iter().all(|node| node
            .output_schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<BTreeSet<_>>()
            .len()
            == node.output_schema().fields().len()),
        "native executable node schemas must be unique:\n{native_plan}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duplicate_group_alias_recursive_aggregate_input_renames_aggregate_spec_outputs() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-duplicate-recursive-group-aliases");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_over_duplicate_group_alias_input_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    let inner_aggregate_node = native_plan
        .nodes()
        .iter()
        .find(|node| {
            node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>()
                == vec!["__nexus_native_field_0_dup", "__nexus_native_field_1_dup", "sum_m00"]
        })
        .expect("native plan should contain the recursive aggregate input node");
    let PlanNode::Aggregate(inner_aggregate) = &inner_aggregate_node.kind else {
        panic!("recursive aggregate input node should be an aggregate");
    };
    assert!(
        matches!(&inner_aggregate.spec.group_exprs[0], Expr::Alias { alias, .. } if alias == "__nexus_native_field_0_dup"),
        "recursive aggregate input first group output should use executable output name: {:?}",
        inner_aggregate.spec.group_exprs
    );
    assert!(
        matches!(&inner_aggregate.spec.group_exprs[1], Expr::Alias { alias, .. } if alias == "__nexus_native_field_1_dup"),
        "recursive aggregate input second group output should use executable output name: {:?}",
        inner_aggregate.spec.group_exprs
    );
    assert_eq!(inner_aggregate.spec.aggr_exprs[0].alias, "sum_m00");
    assert!(
        native_plan.nodes().iter().all(|node| node
            .output_schema()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<BTreeSet<_>>()
            .len()
            == node.output_schema().fields().len()),
        "native executable node schemas must be unique:\n{native_plan}"
    );

    fs::remove_dir_all(dir).unwrap();
}

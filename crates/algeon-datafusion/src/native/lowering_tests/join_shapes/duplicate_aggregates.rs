use super::*;
use crate::native::ParquetReadStrategyConfig;

#[test]
fn duplicate_named_root_aggregate_uses_unique_executable_schema_and_restores_host_sink_schema() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-duplicate-root-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = duplicate_named_root_aggregate_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let aggregate_node = native_plan
        .nodes()
        .iter()
        .find(|node| matches!(node.kind, PlanNode::Aggregate(_)))
        .expect("native plan should contain root aggregate node");
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        panic!("native plan should contain root aggregate node");
    };
    assert!(
        matches!(&aggregate.spec.group_exprs[0], Expr::Alias { alias, .. } if alias == "__algeon_native_field_0_dup"),
        "aggregate group output alias must use executable schema: {:?}",
        aggregate.spec.group_exprs
    );
    assert_eq!(aggregate.spec.aggr_exprs[0].alias, "__algeon_native_field_1_dup");
    assert_eq!(
        aggregate_node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["__algeon_native_field_0_dup", "__algeon_native_field_1_dup"]
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
fn duplicate_group_alias_root_aggregate_reads_input_columns_under_unique_output_names() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-duplicate-root-group-aliases");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = duplicate_group_alias_root_aggregate_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let aggregate_node = native_plan
        .nodes()
        .iter()
        .find(|node| matches!(node.kind, PlanNode::Aggregate(_)))
        .expect("native plan should contain root aggregate node");
    let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
        panic!("native plan should contain root aggregate node");
    };
    for (group_expr, (input, output)) in aggregate
        .spec
        .group_exprs
        .iter()
        .zip([("key", "__algeon_native_field_0_dup"), ("payload", "__algeon_native_field_1_dup")])
    {
        assert!(
            matches!(
                group_expr,
                Expr::Alias { expr, alias }
                    if alias == output && matches!(expr.as_ref(), Expr::Column(column) if column.name() == input)
            ),
            "each duplicate group alias should read its own input column under a unique executable name: {:?}",
            aggregate.spec.group_exprs
        );
    }
    assert_eq!(
        aggregate_node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["__algeon_native_field_0_dup", "__algeon_native_field_1_dup", "sum_m00"]
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
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-duplicate-recursive-group-aliases");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = aggregate_over_duplicate_group_alias_input_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = result.selected_plan();
    let inner_aggregate_node = native_plan
        .nodes()
        .iter()
        .find(|node| {
            node.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>()
                == vec!["__algeon_native_field_0_dup", "__algeon_native_field_1_dup", "sum_m00"]
        })
        .expect("native plan should contain the recursive aggregate input node");
    let PlanNode::Aggregate(inner_aggregate) = &inner_aggregate_node.kind else {
        panic!("recursive aggregate input node should be an aggregate");
    };
    assert!(
        matches!(&inner_aggregate.spec.group_exprs[0], Expr::Alias { alias, .. } if alias == "__algeon_native_field_0_dup"),
        "recursive aggregate input first group output should use executable output name: {:?}",
        inner_aggregate.spec.group_exprs
    );
    assert!(
        matches!(&inner_aggregate.spec.group_exprs[1], Expr::Alias { alias, .. } if alias == "__algeon_native_field_1_dup"),
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

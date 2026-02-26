use super::*;

#[path = "join_shapes/duplicate_aggregates.rs"]
mod duplicate_aggregates;
#[path = "join_shapes/residual_variants.rs"]
mod residual_variants;

#[tokio::test]
async fn physical_cross_join_exec_lowers_to_native_cross_join() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-cross-join-exec");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT a.key, b.payload FROM bench AS a CROSS JOIN bench AS b ORDER BY a.key LIMIT 2",
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = result.plans[0]
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a cross join node");
    assert_eq!(join.spec.join_type, NativeJoinType::Cross);
    assert!(join.spec.keys.is_empty());
    assert!(join.spec.residual_predicate.is_none());
    assert_eq!(join.spec.output_projection.len(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_filter_lowers_to_native_residual_predicate() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-hash-join-filter");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = hash_join_filter_aggregate_sort_limit_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    let join = native_plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node");
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(left.as_ref(), Expr::Column(column) if column.to_string() == "input0.l_payload")
                && matches!(right.as_ref(), Expr::Column(column) if column.to_string() == "input1.r_payload")
    ));

    let sources = native_plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source_node) => match source_node.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                SourceNode::GpuResident(_) => None,
                SourceNode::LiteralBatch(_) => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0].read_columns, vec!["key".to_string(), "l_m00".to_string(), "l_payload".to_string()]);
    assert_eq!(sources[1].read_columns, vec!["key".to_string(), "r_m01".to_string(), "r_payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn expression_hash_join_keys_materialize_as_synthetic_projection_columns() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-expression-join-key");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let left = parquet_scan(&left_path);
    let right = parquet_scan(&right_path);
    let left_schema = left.schema();
    let right_schema = right.schema();
    let left_key =
        binary(col("key", &left_schema).unwrap(), Operator::Plus, lit(ScalarValue::Int32(Some(1))), &left_schema)
            .unwrap();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(left_key, col("key", &right_schema).unwrap())],
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );

    let result = lower_physical_plan_to_native(join);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    let join = native_plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Join(join) => Some(join),
            _ => None,
        })
        .expect("native plan should contain a join node");
    assert_eq!(join.spec.keys.len(), 1);
    assert_eq!(join.spec.keys[0].left.name(), "__cudf_join_key_left_0_0");
    assert_eq!(join.spec.keys[0].right.name(), "key");
    assert!(
        native_plan.nodes().iter().any(|node| match &node.kind {
            PlanNode::Projection(projection) =>
                projection.outputs.iter().any(|output| output.field.name() == "__cudf_join_key_left_0_0"),
            _ => false,
        }),
        "expression join key should be materialized by a pre-join projection"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn predicate_free_left_nested_loop_join_with_scalar_right_lowers_to_native_cross_join() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-left-nlj-scalar-right");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let left = parquet_scan(&left_path);
    let right_scan = parquet_scan(&right_path);
    let right_scan_schema = right_scan.schema();
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(ScalarValue::Int64(Some(1)))])
        .schema(Arc::clone(&right_scan_schema))
        .alias("right_count")
        .build()
        .unwrap();
    let right: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![Arc::new(count)],
            vec![None],
            right_scan,
            right_scan_schema,
        )
        .unwrap(),
    );
    let left_schema = left.schema();
    let right_schema = right.schema();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        NestedLoopJoinExec::try_new(
            left,
            right,
            None,
            &JoinType::Left,
            Some(vec![
                left_schema.index_of("key").unwrap(),
                left_schema.index_of("l_m00").unwrap(),
                left_schema.fields().len() + right_schema.index_of("right_count").unwrap(),
            ]),
        )
        .unwrap(),
    );

    let result = lower_physical_plan_to_native(join);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let join = rewritten_native_join(&result);
    assert_eq!(join.spec.join_type, NativeJoinType::Cross);
    assert!(join.spec.keys.is_empty());
    assert!(join.spec.residual_predicate.is_none());
    assert_join_output_projection(
        &join.spec.output_projection,
        &[
            (NativeJoinSide::Left, "input0.key", "key"),
            (NativeJoinSide::Left, "input0.l_m00", "l_m00"),
            (NativeJoinSide::Right, "input1.right_count", "right_count"),
        ],
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duplicate_same_named_canonical_swapped_join_fields_project_by_ordinal_to_distinct_native_columns() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-duplicate-join-fields");
    let left_path = dir.join("supplier_nation.parquet");
    let right_path = dir.join("customer_nation.parquet");
    write_duplicate_nation_parquet_file(&left_path, &["FRANCE", "GERMANY"]);
    write_duplicate_nation_parquet_file(&right_path, &["BRAZIL", "CANADA"]);
    let left = parquet_scan(&left_path);
    let right = parquet_scan(&right_path);
    let left_schema = left.schema();
    let right_schema = right.schema();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("id", &left_schema).unwrap(), col("id", &right_schema).unwrap())],
            None,
            &JoinType::Right,
            Some(vec![
                left_schema.index_of("n_name").unwrap(),
                left_schema.fields().len() + right_schema.index_of("n_name").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );
    let join_schema = join.schema();
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr {
                    expr: Arc::new(datafusion::physical_expr::expressions::Column::new("n_name", 0)),
                    alias: "supp_nation".to_string(),
                },
                ProjectionExpr {
                    expr: Arc::new(datafusion::physical_expr::expressions::Column::new("n_name", 1)),
                    alias: "cust_nation".to_string(),
                },
            ],
            join,
        )
        .unwrap(),
    );
    let projection_schema = projection.schema();
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        col("supp_nation", &projection_schema).unwrap(),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, projection).with_fetch(Some(10)));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}, join_schema={join_schema:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    let join = native_joins(native_plan).into_iter().next().expect("native plan should contain a join");
    let join_output_names =
        join.spec.output_projection.iter().map(|column| column.output_name.as_str()).collect::<Vec<_>>();
    assert_eq!(join_output_names.len(), 2);
    assert_ne!(join_output_names[0], join_output_names[1]);
    assert!(join_output_names.iter().all(|name| name.starts_with("__cudf_field_")));
    assert_join_output_projection(
        &join.spec.output_projection,
        &[
            (NativeJoinSide::Right, "input1.n_name", join_output_names[0]),
            (NativeJoinSide::Left, "input0.n_name", join_output_names[1]),
        ],
    );

    let projection_node = native_plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Projection(projection) => {
                let exprs = projection.expressions().map(ToString::to_string).collect::<Vec<_>>();
                if exprs.iter().any(|expr| expr.contains("supp_nation"))
                    && exprs.iter().any(|expr| expr.contains("cust_nation"))
                {
                    Some(exprs)
                } else {
                    None
                }
            },
            _ => None,
        })
        .expect("native plan should contain the aliasing projection");
    assert_eq!(
        projection_node,
        vec![format!("{} AS supp_nation", join_output_names[0]), format!("{} AS cust_nation", join_output_names[1]),]
    );
    assert_eq!(
        native_plan.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["supp_nation", "cust_nation"]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bare_duplicate_frontend_join_fields_restore_at_native_host_sink_without_executable_name_leak() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-bare-duplicate-join-fields");
    let left_path = dir.join("supplier_nation.parquet");
    let right_path = dir.join("customer_nation.parquet");
    write_duplicate_nation_parquet_file(&left_path, &["FRANCE", "GERMANY"]);
    write_duplicate_nation_parquet_file(&right_path, &["BRAZIL", "CANADA"]);
    let left = parquet_scan(&left_path);
    let right = parquet_scan(&right_path);
    let left_schema = left.schema();
    let right_schema = right.schema();
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(col("id", &left_schema).unwrap(), col("id", &right_schema).unwrap())],
            None,
            &JoinType::Right,
            Some(vec![
                left_schema.index_of("n_name").unwrap(),
                left_schema.fields().len() + right_schema.index_of("n_name").unwrap(),
            ]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        Arc::new(datafusion::physical_expr::expressions::Column::new("n_name", 0)),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, join).with_fetch(Some(10)));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let native_plan = &result.plans[0];
    assert_eq!(
        native_plan.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["n_name", "n_name"]
    );
    let executable_root_schema = native_plan.node(native_plan.root_id()).unwrap().output_schema();
    assert_eq!(
        executable_root_schema.fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["__cudf_field_0_n_name", "__cudf_field_1_n_name"]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn recursive_hash_join_input_filters_lower_to_native_filters() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-recursive-input-filters");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let left = filtered_payload_scan(parquet_scan(&left_path), "l_payload", 10);
    let right = filtered_payload_scan(parquet_scan(&right_path), "r_payload", 20);
    let plan = hash_join_aggregate_sort_limit_plan_with_type(left, right, JoinType::Inner);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("filtered recursive hash-join inputs should lower to native");
    };
    assert_eq!(
        candidate.shape,
        "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(HashJoinExec(FilterExec(LocalParquetScan),FilterExec(LocalParquetScan))))))"
    );
    assert_eq!(candidate.operator_count, 9);

    let filters = native_filter_exprs(&result.plans[0]);
    assert_eq!(filters.len(), 2);
    assert!(filters.iter().any(|filter| {
        matches!(
            filter,
            Expr::Binary { left, right, .. }
                if matches!(left.as_ref(), Expr::Column(column) if column.name() == "l_payload")
                    && matches!(right.as_ref(), Expr::Literal(LiteralValue::Int32(10)))
        )
    }));
    assert!(filters.iter().any(|filter| {
        matches!(
            filter,
            Expr::Binary { left, right, .. }
                if matches!(left.as_ref(), Expr::Column(column) if column.name() == "r_payload")
                    && matches!(right.as_ref(), Expr::Literal(LiteralValue::Int32(20)))
        )
    }));

    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0].pure_filter_columns, vec!["l_payload".to_string()]);
    assert_eq!(sources[1].pure_filter_columns, vec!["r_payload".to_string()]);
    assert_eq!(sources[0].diagnostics.pure_filter_column_count, 1);
    assert_eq!(sources[1].diagnostics.pure_filter_column_count, 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn post_join_filter_lowers_to_native_filter_input() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-post-join-filter");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let plan = post_join_filter_aggregate_sort_limit_plan(parquet_scan(&left_path), parquet_scan(&right_path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("post-join filter should lower as a native input filter");
    };
    assert_eq!(
        candidate.shape,
        "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(HashJoinExec(LocalParquetScan,LocalParquetScan))))))"
    );

    let filters = native_filter_exprs(&result.plans[0]);
    assert_eq!(filters.len(), 1);
    assert!(matches!(
        filters[0],
        Expr::Binary { left, right, .. }
            if matches!(left.as_ref(), Expr::Column(column) if column.name() == "l_m00")
                && matches!(right.as_ref(), Expr::Column(column) if column.name() == "r_m01")
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duplicate_named_residual_columns_bind_executable_names_after_canonical_swap() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-duplicate-residual-bindings");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);

    let left_scan = parquet_scan(&left_path);
    let left_scan_schema = left_scan.schema();
    let left: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &left_scan_schema).unwrap(), alias: "dup".to_string() },
                ProjectionExpr { expr: col("l_payload", &left_scan_schema).unwrap(), alias: "dup".to_string() },
            ],
            left_scan,
        )
        .unwrap(),
    );
    let right_scan = parquet_scan(&right_path);
    let right_scan_schema = right_scan.schema();
    let right: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            [
                ProjectionExpr { expr: col("key", &right_scan_schema).unwrap(), alias: "dup".to_string() },
                ProjectionExpr { expr: col("r_payload", &right_scan_schema).unwrap(), alias: "dup".to_string() },
            ],
            right_scan,
        )
        .unwrap(),
    );
    let left_schema = left.schema();
    let right_schema = right.schema();
    let filter_schema = Arc::new(Schema::new(vec![left_schema.field(1).clone(), right_schema.field(1).clone()]));
    let filter_expr = binary(
        Arc::new(datafusion::physical_expr::expressions::Column::new("dup", 0)),
        Operator::Gt,
        Arc::new(datafusion::physical_expr::expressions::Column::new("dup", 1)),
        &filter_schema,
    )
    .unwrap();
    let filter = JoinFilter::new(
        filter_expr,
        vec![ColumnIndex { index: 1, side: JoinSide::Left }, ColumnIndex { index: 1, side: JoinSide::Right }],
        filter_schema,
    );
    let join: Arc<dyn ExecutionPlan> = Arc::new(
        HashJoinExec::try_new(
            left,
            right,
            vec![(
                Arc::new(datafusion::physical_expr::expressions::Column::new("dup", 0)),
                Arc::new(datafusion::physical_expr::expressions::Column::new("dup", 0)),
            )],
            Some(filter),
            &JoinType::Right,
            Some(vec![1, left_schema.fields().len() + 1]),
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    );
    let sort_exprs = LexOrdering::new([PhysicalSortExpr::new(
        Arc::new(datafusion::physical_expr::expressions::Column::new("dup", 0)),
        ArrowSortOptions::new(false, false),
    )])
    .unwrap();
    let plan: Arc<dyn ExecutionPlan> = Arc::new(SortExec::new(sort_exprs, join).with_fetch(Some(10)));

    let result = lower_physical_plan_to_native(plan);

    let join = rewritten_native_join(&result);
    assert_eq!(join.spec.join_type, NativeJoinType::Left);
    let residual = join.spec.residual_predicate.as_ref().expect("join filter should lower as residual predicate");
    assert!(matches!(
        residual,
        Expr::Binary { left, right, .. }
            if matches!(
                left.as_ref(),
                Expr::Column(column) if column.to_string() == "input1.__cudf_field_1_dup"
            )
                && matches!(
                    right.as_ref(),
                    Expr::Column(column) if column.to_string() == "input0.__cudf_field_1_dup"
                )
    ));
    assert_join_output_projection(
        &join.spec.output_projection,
        &[
            (NativeJoinSide::Right, "input1.__cudf_field_1_dup", "__cudf_field_0_dup"),
            (NativeJoinSide::Left, "input0.__cudf_field_1_dup", "__cudf_field_1_dup"),
        ],
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn hash_join_aggregate_inputs_lower_to_native_aggregate_inputs() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-join-aggregate-inputs");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_left_parquet_file(&left_path);
    write_join_right_parquet_file(&right_path);
    let left = grouped_sum_join_input(parquet_scan(&left_path), "key", "l_m00", "l_m00");
    let right = grouped_sum_join_input(parquet_scan(&right_path), "key", "r_m01", "r_m01");
    let plan = hash_join_aggregate_sort_limit_plan_with_type(left, right, JoinType::Inner);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("aggregate join inputs should lower as native aggregate input nodes");
    };
    assert_eq!(
        candidate.shape,
        "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(HashJoinExec(AggregateExec(LocalParquetScan),AggregateExec(LocalParquetScan))))))"
    );
    assert_eq!(candidate.operator_count, 9);

    let aggregates = native_aggregate_nodes(&result.plans[0]);
    assert_eq!(aggregates.len(), 3);
    assert_eq!(aggregates[0].spec.aggr_exprs[0].alias, "l_m00");
    assert_eq!(aggregates[1].spec.aggr_exprs[0].alias, "r_m01");
    assert_eq!(aggregates[2].spec.aggr_exprs[0].alias, "sum_x00");

    let sources = native_local_parquet_sources(&result.plans[0]);
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0].read_columns, vec!["key".to_string(), "l_m00".to_string()]);
    assert_eq!(sources[1].read_columns, vec!["key".to_string(), "r_m01".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

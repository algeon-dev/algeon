#[test]
fn aggregate_join_pushdown_reuses_physical_group_identity_and_preserves_output_aliases() {
    for group_aliases in [vec!["group_key"], vec!["group_key", "second_group_key"]] {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(estimated_local_source_with_fields(
            "aliased_fact", &["physical_key", "value"], 1024 * 1024, 10_000,
        ));
        let right = builder.add_source(estimated_local_source_with_fields(
            "aliased_dimension", &["physical_key"], 1024, 100,
        ));
        let mut output_names = group_aliases.clone();
        output_names.push("value");
        let mut outputs = group_aliases.iter().map(|alias| {
            JoinOutputColumn::left(ColumnRef::input(0, "physical_key", *alias), *alias)
        }).collect::<Vec<_>>();
        outputs.push(JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"));
        let join = builder.add_join(left, right, JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "physical_key", "display_key"),
                ColumnRef::input(1, "physical_key", "display_key"),
            )],
        ).with_output_projection(outputs).with_build_side(JoinSide::Right), schema(&output_names));
        let mut projected_names = group_aliases.clone();
        projected_names.push("measure");
        let mut expressions = group_aliases.iter().map(|alias| Expr::column(*alias)).collect::<Vec<_>>();
        expressions.push(Expr::binary(
            Expr::column("value"), BinaryOp::Add, Expr::column("value"),
        ).alias("measure"));
        let projection = builder.add_projection(join, projection_outputs(expressions, schema(&projected_names)));
        let mut final_names = group_aliases.clone();
        final_names.push("sum_value");
        let aggregate = builder.add_aggregate(projection, AggregateSpec::new(
            group_aliases.iter().map(|alias| Expr::column(*alias)).collect(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("measure"), "sum_value")],
        ), schema(&final_names));
        let plan = builder.finish_with_host_sink(aggregate);
        let optimized = optimize_query_plan(plan, &aggregate_join_pushdown_only_options(), rapids_analysis_context());
        assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
        assert_eq!(optimized.plan.root_node().output_schema(), &schema(&final_names));
        let PlanNode::Aggregate(final_aggregate) = &optimized.plan.root_node().kind else { panic!("final aggregate") };
        let PlanNode::Join(reduced_join) = &optimized.plan.node(final_aggregate.input).unwrap().kind else { panic!("reduced join") };
        let preaggregate_node = optimized.plan.node(reduced_join.inputs[0]).unwrap();
        let PlanNode::Aggregate(preaggregate) = &preaggregate_node.kind else { panic!("preaggregate") };
        assert_eq!(preaggregate.spec.group_exprs.len(), group_aliases.len(),
            "the equality key must reuse an existing physical group rather than append its display alias");
        assert_eq!(reduced_join.spec.keys[0].left.name(), "group_key");
        assert_eq!(reduced_join.spec.keys[0].left.display_name(), "display_key");
        assert_eq!(reduced_join.spec.keys[0].right.name(), "physical_key");
        for alias in &group_aliases {
            assert!(preaggregate_node.output_schema().field_with_name(alias).is_ok());
            assert!(reduced_join.spec.output_projection.iter().any(|output| output.column.name() == *alias));
        }
        let PlanNode::Projection(measure_input) = &optimized.plan.node(preaggregate.input).unwrap().kind else { panic!("computed measure projection") };
        assert_eq!(measure_input.outputs.iter().filter(|output| output.field.name() == "physical_key").count(), 1,
            "materializing multiple observable aliases must read each physical group column only once");
    }
}

#[test]
fn aggregate_join_pushdown_reduces_plain_sum_inner_join_without_semijoining_duplicates() {
    let source_schema = schema(&["zip_prefix", "s_store_name", "ss_net_profit"]);
    let right_schema = schema(&["zip_prefix"]);
    let join_schema = schema(&["ss_net_profit", "s_store_name"]);
    let aggregate_input_schema = schema(&["s_store_name", "__nexus_native_aggregate_arg_0"]);
    let aggregate_schema = schema(&["s_store_name", "sum_profit"]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/q08-class-left.parquet")
                .with_projected_row_group_bytes(vec![512 * 1024 * 1024], vec![512 * 1024 * 1024])
                .with_row_group_rows(vec![10_000_000]),
        ],
        Arc::clone(&source_schema),
        vec!["zip_prefix".into(), "s_store_name".into(), "ss_net_profit".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/q08-class-right.parquet")
                .with_projected_row_group_bytes(vec![1024], vec![1024])
                .with_row_group_rows(vec![100]),
        ],
        Arc::clone(&right_schema),
        vec!["zip_prefix".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "zip_prefix", "zip_prefix"),
                ColumnRef::input(1, "zip_prefix", "zip_prefix"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_net_profit", "ss_net_profit"), "ss_net_profit"),
            JoinOutputColumn::left(ColumnRef::input(0, "s_store_name", "s_store_name"), "s_store_name"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let projection = builder.add_projection(
        join,
        projection_outputs(
            vec![Expr::column("s_store_name"), Expr::column("ss_net_profit").alias("__nexus_native_aggregate_arg_0")],
            aggregate_input_schema,
        ),
    );
    let aggregate = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("s_store_name")],
            vec![AggregateExpr::new(
                AggregateFunction::Sum,
                Expr::column("__nexus_native_aggregate_arg_0"),
                "sum_profit",
            )],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert!(optimized.changed(), "plain q08-class aggregate-over-join should be pre-aggregated");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let root = optimized.plan.root_node();
    let PlanNode::Aggregate(final_aggregate) = &root.kind else {
        panic!("optimized root should remain a final aggregate");
    };
    assert!(final_aggregate.spec.grouping_sets.is_none());
    assert_eq!(final_aggregate.spec.group_exprs, vec![Expr::column("s_store_name")]);

    let PlanNode::Join(reduced_join) =
        &optimized.plan.node(final_aggregate.input).expect("final aggregate input should exist").kind
    else {
        panic!("final aggregate should consume the reduced join");
    };
    assert_eq!(
        reduced_join.spec.join_type,
        JoinType::Inner,
        "right-side duplicate keys must retain inner join multiplicity"
    );
    assert!(
        reduced_join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left),
        "q08-class reduced join should only emit pre-aggregated left columns"
    );

    let preaggregate_id = reduced_join.inputs[0];
    let PlanNode::Aggregate(preaggregate) =
        &optimized.plan.node(preaggregate_id).expect("pre-aggregate input should exist").kind
    else {
        panic!("left input to reduced join should be the pre-aggregate");
    };
    assert_eq!(
        preaggregate.spec.group_exprs,
        vec![Expr::column("s_store_name").alias("s_store_name"), Expr::column("zip_prefix").alias("zip_prefix")],
        "pre-aggregate must include the join key so the reduced inner join preserves right-key multiplicity"
    );
}

#[test]
fn aggregate_join_pushdown_rejects_large_fixed_width_detail_key_preaggregate_without_reduction_evidence() {
    let join_schema = schema(&["part_key", "supp_key", "value"]);
    let aggregate_schema = schema(&["part_key", "sum_value"]);

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "detail_fact",
        &["part_key", "supp_key", "value"],
        4 * 1024 * 1024 * 1024,
        80_000_000,
    ));
    let dim = builder.add_source(estimated_local_source_with_fields("filtered_dim", &["supp_key"], 16 * 1024, 1));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "supp_key", "supp_key"),
                ColumnRef::input(1, "supp_key", "supp_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "part_key", "part_key"), "part_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "supp_key", "supp_key"), "supp_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("part_key")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let mut options = NativeOptimizerOptions::disabled();
    options.aggregate_join_pushdown = AggregateJoinPushdownOptions::default();
    let optimized = optimize_query_plan(plan.clone(), &options, rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Unchanged);
    assert_eq!(optimized.plan, plan);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 0);
    assert_eq!(
        optimized.diagnostics.aggregate_join_pushdown.candidates[0].rejected_reason,
        Some(AggregateJoinPushdownRejectedReason::DetailKeyPreAggregateReductionNotEvidenced)
    );
}

#[test]
fn aggregate_join_pushdown_rejects_large_single_join_key_detail_preaggregate_without_reduction_evidence() {
    let join_schema = schema(&["order_key", "value"]);
    let aggregate_schema = schema(&["order_key", "sum_value"]);

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "large_detail_fact",
        &["order_key", "value"],
        8 * 1024 * 1024 * 1024,
        160_000_000,
    ));
    let filtered_orders =
        builder.add_source(estimated_local_source_with_fields("filtered_orders", &["order_key"], 64 * 1024, 10_000));
    let join = builder.add_join(
        fact,
        filtered_orders,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_key", "order_key"),
                ColumnRef::input(1, "order_key", "order_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_key", "order_key"), "order_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("order_key")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let mut options = NativeOptimizerOptions::disabled();
    options.aggregate_join_pushdown = AggregateJoinPushdownOptions::default();
    let optimized = optimize_query_plan(plan.clone(), &options, rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Unchanged);
    assert_eq!(optimized.plan, plan);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 0);
    assert_eq!(
        optimized.diagnostics.aggregate_join_pushdown.candidates[0].rejected_reason,
        Some(AggregateJoinPushdownRejectedReason::DetailKeyPreAggregateReductionNotEvidenced)
    );
}

#[test]
fn aggregate_join_pushdown_admits_fixed_width_detail_key_below_reduction_evidence_threshold() {
    let join_schema = schema(&["order_key", "value"]);
    let aggregate_schema = schema(&["order_key", "sum_value"]);

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "bounded_detail_fact",
        &["order_key", "value"],
        4 * 1024 * 1024 * 1024,
        63_999_999,
    ));
    let filtered_orders =
        builder.add_source(estimated_local_source_with_fields("small_orders", &["order_key"], 64 * 1024, 10_000));
    let join = builder.add_join(
        fact,
        filtered_orders,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_key", "order_key"),
                ColumnRef::input(1, "order_key", "order_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_key", "order_key"), "order_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("order_key")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let mut options = NativeOptimizerOptions::disabled();
    options.aggregate_join_pushdown = AggregateJoinPushdownOptions::default();
    let optimized = optimize_query_plan(plan, &options, rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Changed);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.candidates[0].rejected_reason, None);
}

#[test]
fn aggregate_join_pushdown_admits_large_variable_width_group_key_preaggregate() {
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("store_name", DataType::Utf8, false),
        Field::new("dim_key", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("store_name", DataType::Utf8, false),
        Field::new("value", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("store_name", DataType::Utf8, false),
        Field::new("sum_value", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/large-store-detail.parquet")
                .with_projected_row_group_bytes(vec![8 * 1024 * 1024 * 1024], vec![8 * 1024 * 1024 * 1024])
                .with_row_group_rows(vec![80_000_000]),
        ],
        Arc::clone(&fact_schema),
        vec!["store_name".into(), "dim_key".into(), "value".into()],
    )));
    let dim = builder.add_source(estimated_local_source_with_fields("filtered_dim_store", &["dim_key"], 64 * 1024, 10));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "dim_key", "dim_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "store_name", "store_name"), "store_name"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("store_name")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let mut options = NativeOptimizerOptions::disabled();
    options.aggregate_join_pushdown = AggregateJoinPushdownOptions::default();
    let optimized = optimize_query_plan(plan, &options, rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Changed);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.candidates[0].rejected_reason, None);
}

#[test]
fn aggregate_join_pushdown_preserves_grouping_id_output_schema_for_grouping_sets() {
    let sales_schema = schema(&["store_sk", "state", "county", "net_profit"]);
    let store_schema = schema(&["store_sk"]);
    let join_schema = schema(&["state", "county", "net_profit"]);
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("state", DataType::Int32, true),
        Field::new("county", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("sum_profit", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("sales", Arc::clone(&sales_schema))));
    let store = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("store", Arc::clone(&store_schema))));
    let join = builder.add_join(
        sales,
        store,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "store_sk", "store_sk"),
                ColumnRef::input(1, "store_sk", "store_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "state", "state"), "state"),
            JoinOutputColumn::left(ColumnRef::input(0, "county", "county"), "county"),
            JoinOutputColumn::left(ColumnRef::input(0, "net_profit", "net_profit"), "net_profit"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("state"), Expr::column("county")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("net_profit"), "sum_profit")],
        )
        .with_grouping_sets(nexus_query_engine::expr::GroupingSetsSpec::new(
            vec![vec![true, true], vec![false, true], vec![false, false]],
            "__grouping_id",
        )),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert!(optimized.changed(), "grouping-set aggregate-over-join should still be eligible for pre-aggregation");
    assert_all_node_inputs_exist(&optimized.plan);
    let root = optimized.plan.root_node();
    let PlanNode::Aggregate(final_aggregate) = &root.kind else {
        panic!("optimized root should remain a final grouping-sets aggregate");
    };
    assert!(
        final_aggregate.spec.grouping_sets.is_some(),
        "aggregate join pushdown must preserve the grouping-set contract"
    );
    assert_eq!(
        root.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["state", "county", "__grouping_id", "sum_profit"],
        "final aggregate schema must retain the generated grouping-id field consumed by downstream GROUPING() projections"
    );
}

#[test]
fn aggregate_join_pushdown_avg_grouping_sets_final_projection_facts_match_typed_outputs() {
    let sales_schema = schema(&["store_sk", "state", "county", "qty"]);
    let store_schema = schema(&["store_sk"]);
    let join_schema = schema(&["state", "county", "qty"]);
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("state", DataType::Int32, true),
        Field::new("county", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("avg_qty", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("sales", Arc::clone(&sales_schema))));
    let store = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("store", Arc::clone(&store_schema))));
    let join = builder.add_join(
        sales,
        store,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "store_sk", "store_sk"),
                ColumnRef::input(1, "store_sk", "store_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "state", "state"), "state"),
            JoinOutputColumn::left(ColumnRef::input(0, "county", "county"), "county"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("state"), Expr::column("county")],
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("qty"), "avg_qty")],
        )
        .with_grouping_sets(nexus_query_engine::expr::GroupingSetsSpec::new(
            vec![vec![true, true], vec![false, true], vec![false, false]],
            "__grouping_id",
        )),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert!(optimized.changed(), "AVG grouping-set aggregate-over-join should be pre-aggregated");
    let root = optimized.plan.root_node();
    let PlanNode::Projection(final_projection) = &root.kind else {
        panic!("AVG pushdown should project the public average after explicit state merge");
    };
    assert_eq!(
        final_projection.output_schema().as_ref(),
        root.facts.output_schema.as_ref(),
        "optimizer-created final projection facts must mirror typed projection outputs"
    );
    assert_eq!(
        final_projection.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["state", "county", "__grouping_id", "avg_qty"],
        "the AVG final projection must preserve the public grouping id after computing the public aggregate output"
    );
    let PlanNode::Aggregate(final_aggregate) =
        &optimized.plan.node(final_projection.input).expect("final projection input should exist").kind
    else {
        panic!("AVG final projection should consume a state-merging grouping-set aggregate");
    };
    assert!(
        final_aggregate.spec.grouping_sets.is_some(),
        "the state-merging aggregate must keep the grouping-set contract"
    );
}

#[test]
fn aggregate_join_pushdown_reduces_left_join_count_on_non_preserved_side() {
    let left_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int64, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customer =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("customer", Arc::clone(&left_schema))));
    let orders =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("orders", Arc::clone(&right_schema))));
    let join = builder.add_join(
        customer,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "c_custkey", "c_custkey"),
                ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert!(optimized.changed(), "left count-over-join aggregate should be pre-aggregated");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert_all_node_inputs_exist(&optimized.plan);

    let PlanNode::Aggregate(final_aggregate) = &optimized.plan.root_node().kind else {
        panic!("optimized root should remain a final aggregate");
    };
    assert_eq!(final_aggregate.spec.group_exprs, vec![Expr::column("c_custkey")]);
    assert_eq!(final_aggregate.spec.aggr_exprs.len(), 1);
    assert_eq!(final_aggregate.spec.aggr_exprs[0].function, AggregateFunction::Sum);
    assert_eq!(final_aggregate.spec.aggr_exprs[0].alias, "c_count");
    let Expr::Column(preagg_count_column) = &final_aggregate.spec.aggr_exprs[0].expr else {
        panic!("final aggregate should sum the projected pre-aggregate count");
    };
    assert!(preagg_count_column.name().starts_with("__nexus_native_preagg_sum_"));

    let PlanNode::Projection(coalesce_projection) =
        &optimized.plan.node(final_aggregate.input).expect("final aggregate input should exist").kind
    else {
        panic!("left join count rewrite should coalesce missing right-side counts before final aggregation");
    };
    assert!(matches!(
        &coalesce_projection.outputs[1].expr,
        Expr::Alias { expr, alias }
            if alias.starts_with("__nexus_native_preagg_sum_")
                && matches!(expr.as_ref(), Expr::Case { .. })
    ));

    let PlanNode::Join(reduced_join) =
        &optimized.plan.node(coalesce_projection.input).expect("coalesce projection input should exist").kind
    else {
        panic!("coalesce projection should consume the reduced left join");
    };
    assert_eq!(reduced_join.spec.join_type, JoinType::Left);
    assert_eq!(reduced_join.spec.build_side, JoinSide::Right);

    let PlanNode::Aggregate(preaggregate) =
        &optimized.plan.node(reduced_join.inputs[1]).expect("right pre-aggregate should exist").kind
    else {
        panic!("right input to reduced left join should be the pushed pre-aggregate");
    };
    assert_eq!(preaggregate.spec.group_exprs, vec![Expr::column("o_custkey").alias("o_custkey")]);
    assert_eq!(preaggregate.spec.aggr_exprs.len(), 1);
    assert_eq!(preaggregate.spec.aggr_exprs[0].function, AggregateFunction::Count);
    assert_eq!(preaggregate.spec.aggr_exprs[0].expr, Expr::column("o_orderkey"));
}

#[test]
fn aggregate_join_pushdown_reduces_avg_inner_join_with_explicit_sum_count_state() {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("reason_sk", DataType::Int32, false),
        Field::new("qty", DataType::Float64, true),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("reason_sk", DataType::Int32, false),
        Field::new("reason_desc", DataType::Utf8, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Float64, true),
        Field::new("reason_desc", DataType::Utf8, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("reason_desc", DataType::Utf8, true),
        Field::new("avg_qty", DataType::Float64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("fact", Arc::clone(&left_schema))));
    let reason =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("reason", Arc::clone(&right_schema))));
    let join = builder.add_join(
        fact,
        reason,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "reason_sk", "reason_sk"),
                ColumnRef::input(1, "reason_sk", "reason_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(ColumnRef::input(1, "reason_desc", "reason_desc"), "reason_desc"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("reason_desc")],
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("qty"), "avg_qty")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let optimized = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert!(optimized.changed(), "AVG aggregate-over-join should be pre-aggregated as sum/count state");
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert_all_node_inputs_exist(&optimized.plan);

    let PlanNode::Projection(final_projection) = &optimized.plan.root_node().kind else {
        panic!("AVG pushdown should project the public average after merging explicit state");
    };
    assert_eq!(final_projection.outputs.len(), 2);
    assert_eq!(final_projection.outputs[1].field.name(), "avg_qty");
    assert!(
        matches!(&final_projection.outputs[1].expr, Expr::Alias { expr, alias }
            if alias == "avg_qty" && matches!(expr.as_ref(), Expr::Case { .. })),
        "AVG final projection must divide merged SUM by merged COUNT with null-on-zero handling"
    );

    let PlanNode::Aggregate(final_aggregate) =
        &optimized.plan.node(final_projection.input).expect("final projection input should exist").kind
    else {
        panic!("AVG final projection should consume a state-merging aggregate");
    };
    assert_eq!(final_aggregate.spec.group_exprs, vec![Expr::column("reason_desc")]);
    assert_eq!(
        final_aggregate.spec.aggr_exprs.iter().map(|expr| expr.function).collect::<Vec<_>>(),
        vec![AggregateFunction::Sum, AggregateFunction::Sum],
        "final AVG merge must sum partial SUM and COUNT state, not average partial averages"
    );
    assert!(final_aggregate.spec.aggr_exprs.iter().all(|expr| expr.alias.starts_with("__nexus_native_preagg_avg_")));

    let PlanNode::Join(reduced_join) =
        &optimized.plan.node(final_aggregate.input).expect("final aggregate input should exist").kind
    else {
        panic!("state-merging aggregate should consume the reduced join");
    };
    let PlanNode::Aggregate(preaggregate) =
        &optimized.plan.node(reduced_join.inputs[0]).expect("left pre-aggregate should exist").kind
    else {
        panic!("left input to reduced join should be the AVG state pre-aggregate");
    };
    assert_eq!(
        preaggregate.spec.aggr_exprs.iter().map(|expr| expr.function).collect::<Vec<_>>(),
        vec![AggregateFunction::Sum, AggregateFunction::Count]
    );
    assert_eq!(
        preaggregate.spec.group_exprs,
        vec![Expr::column("reason_sk").alias("reason_sk")],
        "pre-aggregate must retain the join key before joining the reason dimension"
    );
}

#[test]
fn aggregate_join_pushdown_splits_large_union_before_small_dimension_join_preaggregate() {
    let branch_schema = schema(&["store_sk", "date_sk", "sales"]);
    let date_join_schema = schema(&["store_sk", "sales"]);
    let store_join_schema = schema(&["store_id", "sales"]);
    let aggregate_schema = schema(&["store_id", "sum_sales"]);

    let mut builder = QueryPlanBuilder::new();
    let sales_a = builder.add_source(estimated_local_source_with_fields(
        "sales_a",
        &["store_sk", "date_sk", "sales"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let sales_b = builder.add_source(estimated_local_source_with_fields(
        "sales_b",
        &["store_sk", "date_sk", "sales"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let sales_union = builder.add_union(vec![sales_a, sales_b], Arc::clone(&branch_schema));
    let date_dim = builder.add_source(estimated_local_source_with_fields("date_dim", &["date_sk"], 1024 * 1024, 365));
    let date_join = builder.add_join(
        sales_union,
        date_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "date_sk", "date_sk"), ColumnRef::input(1, "date_sk", "date_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "store_sk", "store_sk"), "store_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
        ])
        .with_build_side(JoinSide::Right),
        date_join_schema,
    );
    let store =
        builder.add_source(estimated_local_source_with_fields("store", &["store_sk", "store_id"], 16 * 1024, 1_000));
    let store_join = builder.add_join(
        date_join,
        store,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "store_sk", "store_sk"),
                ColumnRef::input(1, "store_sk", "store_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "store_id", "store_id"), "store_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
        ])
        .with_build_side(JoinSide::Right),
        store_join_schema,
    );
    let aggregate = builder.add_aggregate(
        store_join,
        AggregateSpec::new(
            vec![Expr::column("store_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let mut options = NativeOptimizerOptions::disabled();
    options.aggregate_join_pushdown = AggregateJoinPushdownOptions::default();
    let optimized = optimize_query_plan(plan, &options, rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Changed);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert_all_node_inputs_exist(&optimized.plan);

    let PlanNode::Aggregate(final_aggregate) = &optimized.plan.root_node().kind else {
        panic!("optimized root should be the store-id final aggregate");
    };
    let PlanNode::Join(store_join) =
        &optimized.plan.node(final_aggregate.input).expect("final aggregate input should exist").kind
    else {
        panic!("final aggregate should consume the reduced store join");
    };
    let PlanNode::Aggregate(date_preaggregate_merge) = &optimized
        .plan
        .node(store_join.inputs[0])
        .expect("store join should consume the date pre-aggregate merge")
        .kind
    else {
        panic!("large union date pre-aggregate should be merged before the store join");
    };
    assert_eq!(date_preaggregate_merge.spec.group_exprs, vec![Expr::column("store_sk")]);
    assert_eq!(date_preaggregate_merge.spec.aggr_exprs[0].function, AggregateFunction::Sum);

    let PlanNode::Union(partial_union) = &optimized
        .plan
        .node(date_preaggregate_merge.input)
        .expect("date pre-aggregate merge should consume branch partials")
        .kind
    else {
        panic!("date pre-aggregate merge should consume a union of branch partial aggregates");
    };
    assert_eq!(partial_union.inputs.len(), 2);

    let mut replayed_date_input = None;
    for branch_preaggregate_id in &partial_union.inputs {
        let PlanNode::Aggregate(branch_preaggregate) =
            &optimized.plan.node(*branch_preaggregate_id).expect("branch pre-aggregate should exist").kind
        else {
            panic!("partial union input should be a branch pre-aggregate");
        };
        let PlanNode::Join(branch_date_join) =
            &optimized.plan.node(branch_preaggregate.input).expect("branch pre-aggregate input should exist").kind
        else {
            panic!("branch pre-aggregate should consume a branch-local date join");
        };
        assert!(
            !matches!(
                optimized.plan.node(branch_date_join.inputs[0]).expect("branch date join input should exist").kind,
                PlanNode::Union(_)
            ),
            "branch-local date joins must consume individual union branches"
        );
        if let Some(first_date_input) = replayed_date_input {
            assert_eq!(
                branch_date_join.inputs[1], first_date_input,
                "branch-local date joins should share the same small dimension input"
            );
        } else {
            replayed_date_input = Some(branch_date_join.inputs[1]);
        }
    }
}

#[test]
fn aggregate_join_pushdown_does_not_split_union_when_any_branch_size_is_unknown() {
    let branch_schema = schema(&["store_sk", "date_sk", "sales"]);
    let date_join_schema = schema(&["store_sk", "sales"]);
    let store_join_schema = schema(&["store_id", "sales"]);
    let aggregate_schema = schema(&["store_id", "sum_sales"]);

    let mut builder = QueryPlanBuilder::new();
    let known_sales = builder.add_source(estimated_local_source_with_fields(
        "known_sales",
        &["store_sk", "date_sk", "sales"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let unknown_sales = builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("unknown_sales", Arc::clone(&branch_schema))));
    let sales_union = builder.add_union(vec![known_sales, unknown_sales], Arc::clone(&branch_schema));
    let date_dim = builder.add_source(estimated_local_source_with_fields(
        "date_dim_unknown_branch",
        &["date_sk"],
        1024 * 1024,
        365,
    ));
    let date_join = builder.add_join(
        sales_union,
        date_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "date_sk", "date_sk"), ColumnRef::input(1, "date_sk", "date_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "store_sk", "store_sk"), "store_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
        ])
        .with_build_side(JoinSide::Right),
        date_join_schema,
    );
    let store = builder.add_source(estimated_local_source_with_fields(
        "store_unknown_branch",
        &["store_sk", "store_id"],
        16 * 1024,
        1_000,
    ));
    let store_join = builder.add_join(
        date_join,
        store,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "store_sk", "store_sk"),
                ColumnRef::input(1, "store_sk", "store_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "store_id", "store_id"), "store_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales", "sales"), "sales"),
        ])
        .with_build_side(JoinSide::Right),
        store_join_schema,
    );
    let aggregate = builder.add_aggregate(
        store_join,
        AggregateSpec::new(
            vec![Expr::column("store_id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "sum_sales")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let mut options = NativeOptimizerOptions::disabled();
    options.aggregate_join_pushdown = AggregateJoinPushdownOptions::default();
    let optimized = optimize_query_plan(plan, &options, rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Changed);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let PlanNode::Aggregate(final_aggregate) = &optimized.plan.root_node().kind else {
        panic!("optimized root should be the store-id final aggregate");
    };
    let PlanNode::Join(store_join) =
        &optimized.plan.node(final_aggregate.input).expect("final aggregate input should exist").kind
    else {
        panic!("final aggregate should consume the reduced store join");
    };
    let PlanNode::Aggregate(preaggregate) =
        &optimized.plan.node(store_join.inputs[0]).expect("store join should consume the date pre-aggregate").kind
    else {
        panic!("store join should consume the unsplit date pre-aggregate");
    };
    let PlanNode::Join(date_join) =
        &optimized.plan.node(preaggregate.input).expect("date pre-aggregate input should exist").kind
    else {
        panic!("unsplit date pre-aggregate should still consume the date join");
    };
    assert!(
        matches!(
            optimized.plan.node(date_join.inputs[0]).expect("date join left input should exist").kind,
            PlanNode::Union(_)
        ),
        "unknown union branch size should keep the original join-over-union pre-aggregate"
    );
}

fn left_semi_keyset_aggregate_plan(
    group_columns: &[&str],
    grouping_sets: Option<nexus_query_engine::expr::GroupingSetsSpec>,
) -> nexus_query_engine::plan::QueryPlan {
    let semi_schema = schema(&["order_key", "category", "value"]);
    let mut aggregate_fields = group_columns.to_vec();
    if grouping_sets.is_some() {
        aggregate_fields.push("__grouping_id");
    }
    aggregate_fields.push("sum_value");
    let aggregate_schema = schema(&aggregate_fields);

    let mut builder = QueryPlanBuilder::new();
    let detail = builder.add_source(estimated_local_source_with_fields(
        "semi_detail",
        &["order_key", "category", "value"],
        4 * 1024 * 1024 * 1024,
        80_000_000,
    ));
    let keyset = builder.add_source(estimated_local_source_with_fields("semi_keyset", &["order_key"], 64 * 1024, 1_000));
    let semi = builder.add_join(
        detail,
        keyset,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_key", "order_key"),
                ColumnRef::input(1, "order_key", "order_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_key", "order_key"), "order_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "category", "category"), "category"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        semi_schema,
    );
    let mut spec = AggregateSpec::new(
        group_columns.iter().map(|column| Expr::column(*column)).collect(),
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );
    if let Some(grouping_sets) = grouping_sets {
        spec = spec.with_grouping_sets(grouping_sets);
    }
    let aggregate = builder.add_aggregate(semi, spec, aggregate_schema);
    builder.finish_with_host_sink(aggregate)
}

fn aggregate_join_pushdown_only_options() -> NativeOptimizerOptions {
    let mut options = NativeOptimizerOptions::disabled();
    options.aggregate_join_pushdown = AggregateJoinPushdownOptions::default();
    options
}

#[test]
fn aggregate_join_pushdown_keeps_left_semi_on_final_group_keys_below_the_aggregate() {
    let plan = left_semi_keyset_aggregate_plan(&["order_key", "category"], None);

    let optimized = optimize_query_plan(plan.clone(), &aggregate_join_pushdown_only_options(), rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Unchanged);
    assert_eq!(optimized.plan, plan);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 0);
    assert_eq!(
        optimized.diagnostics.aggregate_join_pushdown.candidates[0].rejected_reason,
        Some(AggregateJoinPushdownRejectedReason::LeftSemiAlreadyFiltersFinalGroups)
    );
}

#[test]
fn aggregate_join_pushdown_still_preaggregates_below_left_semi_whose_key_is_not_a_group_key() {
    let plan = left_semi_keyset_aggregate_plan(&["category"], None);

    let optimized = optimize_query_plan(plan, &aggregate_join_pushdown_only_options(), rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Changed);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    let PlanNode::Aggregate(final_aggregate) = &optimized.plan.root_node().kind else {
        panic!("optimized root should remain the final aggregate");
    };
    assert_eq!(final_aggregate.spec.group_exprs, vec![Expr::column("category")]);
    let PlanNode::Join(semi) = &optimized.plan.node(final_aggregate.input).expect("final input").kind else {
        panic!("final aggregate should consume the left-semi join");
    };
    assert_eq!(semi.spec.join_type, JoinType::LeftSemi);
    let PlanNode::Aggregate(preaggregate) = &optimized.plan.node(semi.inputs[0]).expect("semi left input").kind
    else {
        panic!("semi key outside the final grouping needs the pre-aggregate to carry it");
    };
    assert_eq!(
        preaggregate.spec.group_exprs,
        vec![Expr::column("category").alias("category"), Expr::column("order_key").alias("order_key")]
    );
}

#[test]
fn aggregate_join_pushdown_still_preaggregates_grouping_sets_below_left_semi_on_group_keys() {
    let plan = left_semi_keyset_aggregate_plan(
        &["order_key", "category"],
        Some(nexus_query_engine::expr::GroupingSetsSpec::new(
            vec![vec![true, true], vec![true, false], vec![false, false]],
            "__grouping_id",
        )),
    );

    let optimized = optimize_query_plan(plan, &aggregate_join_pushdown_only_options(), rapids_analysis_context());

    assert_eq!(optimized.change, NativeOptimizerChange::Changed);
    assert_eq!(optimized.diagnostics.aggregate_join_pushdown.applied_count, 1);
    let PlanNode::Aggregate(final_aggregate) = &optimized.plan.root_node().kind else {
        panic!("optimized root should remain the rollup final aggregate");
    };
    assert!(final_aggregate.spec.grouping_sets.is_some());
    let PlanNode::Join(semi) = &optimized.plan.node(final_aggregate.input).expect("final input").kind else {
        panic!("rollup final aggregate should consume the left-semi join");
    };
    assert_eq!(semi.spec.join_type, JoinType::LeftSemi);
    let PlanNode::Aggregate(preaggregate) = &optimized.plan.node(semi.inputs[0]).expect("semi left input").kind
    else {
        panic!("rollup pre-aggregate should still reduce detail rows before the grouping-set expansion");
    };
    assert!(preaggregate.spec.grouping_sets.is_none());
}

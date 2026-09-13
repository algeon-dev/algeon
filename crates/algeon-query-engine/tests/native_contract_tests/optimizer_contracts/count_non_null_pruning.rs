#[test]
fn count_non_null_pruning_rewrites_footer_proven_count_column_to_retained_carrier_and_prunes_count_only_read_column() {
    let plan = count_non_null_pruning_source_adjacent_plan(
        Some(vec![Some(0), Some(0), Some(0)]),
        Some(vec![Some(0), Some(3), Some(0)]),
    );
    let options = NativeOptimizerOptions {
        count_non_null_column_pruning: CountNonNullColumnPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.count_non_null_column_pruning.considered_count, 1);
    assert_eq!(result.diagnostics.count_non_null_column_pruning.applied_count, 1);
    let candidate = &result.diagnostics.count_non_null_column_pruning.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.column, "order_count_input");
    assert_eq!(candidate.file_column_index, Some(1));
    assert_eq!(candidate.rejected_reason, None);

    let PlanNode::Aggregate(aggregate) = &result.plan.root_node().kind else {
        panic!("root should remain aggregate");
    };
    assert_eq!(aggregate.spec.aggr_exprs[0].function, AggregateFunction::Count);
    assert_eq!(aggregate.spec.aggr_exprs[0].expr, Expr::column("o_custkey"));

    let PlanNode::Projection(projection) = &result.plan.node(aggregate.input).expect("aggregate input").kind else {
        panic!("aggregate should consume the source-adjacent projection");
    };
    assert_eq!(projection.outputs.len(), 1);
    assert_eq!(projection.outputs[0].field.name(), "o_custkey");

    let PlanNode::Filter(filter) = &result.plan.node(projection.input).expect("projection input").kind else {
        panic!("projection should consume the source-adjacent filter");
    };
    assert_eq!(filter.predicate.to_string(), "(o_orderstatus = 1)");

    let PlanNode::Source(source) = &result.plan.node(filter.input).expect("filter input").kind else {
        panic!("filter should consume the local Parquet source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("expected local Parquet source");
    };
    assert_eq!(source.read_columns, vec!["o_custkey", "o_orderstatus"]);
    assert_eq!(source.read_column_file_indices, vec![0, 2]);
    assert_eq!(source.projected_output_columns, vec!["o_custkey", "o_orderstatus"]);
    assert_eq!(source.unique_key_columns, vec![BTreeSet::from(["o_custkey".to_string()])]);
    assert!(
        !source.read_columns.iter().any(|column| column == "o_orderkey"),
        "count-only column should not be read after COUNT(column) becomes COUNT(1)"
    );
}

#[test]
fn count_non_null_pruning_prunes_physical_read_columns_when_source_uses_logical_output_schema_override() {
    let plan = count_non_null_pruning_overridden_source_plan();
    let options = NativeOptimizerOptions {
        count_non_null_column_pruning: CountNonNullColumnPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.count_non_null_column_pruning.applied_count, 1);
    let PlanNode::Aggregate(aggregate) = &result.plan.root_node().kind else {
        panic!("root should remain aggregate");
    };
    assert_eq!(aggregate.spec.aggr_exprs[0].expr, Expr::column("o_custkey"));

    let PlanNode::Projection(projection) = &result.plan.node(aggregate.input).expect("aggregate input").kind else {
        panic!("aggregate should consume projection");
    };
    let PlanNode::Filter(filter) = &result.plan.node(projection.input).expect("projection input").kind else {
        panic!("projection should consume filter");
    };
    let PlanNode::Source(source) = &result.plan.node(filter.input).expect("filter input").kind else {
        panic!("filter should consume source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("expected local Parquet source");
    };
    assert_eq!(source.read_columns, vec!["raw_custkey", "raw_status"]);
    assert_eq!(source.read_column_file_indices, vec![0, 2]);
    assert_eq!(source.projected_output_columns, vec!["o_custkey", "o_orderstatus"]);
    assert_eq!(
        source.output_schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        vec!["o_custkey", "o_orderstatus"]
    );
    assert!(
        !source.read_columns.iter().any(|column| column == "raw_orderkey"),
        "logical schema overrides must not prevent pruning the physical count-only column"
    );
}

#[test]
fn count_non_null_pruning_remaps_retained_synthetic_source_refs_and_partition_values() {
    let plan = count_non_null_pruning_synthetic_source_plan();
    let options = NativeOptimizerOptions {
        count_non_null_column_pruning: CountNonNullColumnPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.count_non_null_column_pruning.applied_count, 1);
    let PlanNode::Aggregate(aggregate) = &result.plan.root_node().kind else {
        panic!("root should remain aggregate");
    };
    assert_eq!(aggregate.spec.aggr_exprs[0].expr, Expr::column("k"));

    let PlanNode::Projection(projection) = &result.plan.node(aggregate.input).expect("aggregate input").kind else {
        panic!("aggregate should consume projection");
    };
    let PlanNode::Source(source) = &result.plan.node(projection.input).expect("projection input").kind else {
        panic!("projection should consume source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("expected local Parquet source");
    };
    validate_local_parquet_source(source, rapids_analysis_context()).unwrap();
    assert_eq!(source.read_columns, vec!["k"]);
    assert_eq!(source.read_column_file_indices, vec![0]);
    assert_eq!(source.projected_output_columns, vec!["k", "p1", "m1"]);
    assert_eq!(
        source.projected_output_column_source_refs,
        vec![SourceColumnRef::File(0), SourceColumnRef::Partition(0), SourceColumnRef::MissingNull(0)]
    );
    assert_eq!(source.partition_columns.iter().map(|column| column.name.as_str()).collect::<Vec<_>>(), vec!["p1"]);
    assert_eq!(source.missing_columns.iter().map(|column| column.name.as_str()).collect::<Vec<_>>(), vec!["m1"]);
    assert_eq!(source.files[0].partition_values, vec![LiteralValue::Int32(9)]);
}

#[test]
fn count_non_null_pruning_does_not_mutate_sources_when_no_count_rewrite_applies() {
    let plan = non_count_projection_pruning_plan();
    let options = NativeOptimizerOptions {
        count_non_null_column_pruning: CountNonNullColumnPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result =
        optimize_query_plan(plan.clone(), &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.count_non_null_column_pruning.is_empty());
}

#[test]
fn count_non_null_pruning_keeps_count_column_when_footer_proof_is_missing() {
    let plan = count_non_null_pruning_source_adjacent_plan(None, None);
    let options = NativeOptimizerOptions {
        count_non_null_column_pruning: CountNonNullColumnPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result =
        optimize_query_plan(plan.clone(), &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert_eq!(result.diagnostics.count_non_null_column_pruning.considered_count, 1);
    assert_eq!(result.diagnostics.count_non_null_column_pruning.applied_count, 0);
    let candidate = &result.diagnostics.count_non_null_column_pruning.candidates[0];
    assert!(!candidate.applied);
    assert_eq!(candidate.column, "order_count_input");
    assert_eq!(candidate.file_column_index, Some(1));
    assert_eq!(candidate.rejected_reason, Some(CountNonNullColumnPruningRejectedReason::NullCountProofUnavailable));
}

#[test]
fn count_non_null_pruning_keeps_grouped_count_column_when_no_retained_non_null_carrier_is_proven() {
    let plan = count_non_null_pruning_source_adjacent_plan(None, Some(vec![Some(0), Some(3), Some(0)]));
    let options = NativeOptimizerOptions {
        count_non_null_column_pruning: CountNonNullColumnPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result =
        optimize_query_plan(plan.clone(), &options, rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert_eq!(result.diagnostics.count_non_null_column_pruning.considered_count, 1);
    assert_eq!(result.diagnostics.count_non_null_column_pruning.applied_count, 0);
    let candidate = &result.diagnostics.count_non_null_column_pruning.candidates[0];
    assert!(!candidate.applied);
    assert_eq!(candidate.rejected_reason, Some(CountNonNullColumnPruningRejectedReason::NonNullCarrierUnavailable));
}

fn count_non_null_pruning_source_adjacent_plan(
    carrier_null_counts: Option<Vec<Option<u64>>>,
    count_null_counts: Option<Vec<Option<u64>>>,
) -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_orderstatus", DataType::Int32, false),
    ]));
    let projection_schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("order_count_input", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut file = LocalParquetFile::new("/tmp/count-non-null-orders.parquet").with_row_groups(vec![0, 2]);
    if let Some(null_counts) = carrier_null_counts {
        file = file.with_row_group_null_counts(0, null_counts);
    }
    if let Some(null_counts) = count_null_counts {
        file = file.with_row_group_null_counts(1, null_counts);
    }

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![file],
            Arc::clone(&source_schema),
            vec!["o_custkey".to_string(), "o_orderkey".to_string(), "o_orderstatus".to_string()],
        )
        .with_unique_key_columns(vec![
            BTreeSet::from(["o_custkey".to_string()]),
            BTreeSet::from(["o_orderkey".to_string()]),
        ]),
    ));
    let filtered = builder.add_filter(
        source,
        Expr::binary(Expr::column("o_orderstatus"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1))),
        Arc::clone(&source_schema),
    );
    let projected = builder.add_projection(
        filtered,
        projection_outputs(
            vec![Expr::column("o_custkey"), Expr::column("o_orderkey").alias("order_count_input")],
            projection_schema,
        ),
    );
    let aggregate = builder.add_aggregate(
        projected,
        AggregateSpec::new(
            vec![Expr::column("o_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_count_input"), "c_count")],
        ),
        aggregate_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

fn count_non_null_pruning_synthetic_source_plan() -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("cnt", DataType::Int32, false),
        Field::new("p0", DataType::Int32, false),
        Field::new("p1", DataType::Int32, false),
        Field::new("m0", DataType::Int64, true),
        Field::new("m1", DataType::Int64, true),
    ]));
    let projection_schema = Arc::new(Schema::new(vec![
        Field::new("p1", DataType::Int32, false),
        Field::new("m1", DataType::Int64, true),
        Field::new("k", DataType::Int32, false),
        Field::new("count_input", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("p1", DataType::Int32, false),
        Field::new("m1", DataType::Int64, true),
        Field::new("k", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));
    let file = LocalParquetFile::new("/tmp/count-non-null-synthetic.parquet")
        .with_partition_values(vec![LiteralValue::Int32(7), LiteralValue::Int32(9)])
        .with_row_group_null_counts(0, vec![Some(0)])
        .with_row_group_null_counts(1, vec![Some(0)]);
    let source = LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&source_schema),
        vec!["k".to_string(), "cnt".to_string()],
    )
    .with_partition_columns(vec![
        PartitionColumn::new("p0", DataType::Int32, false),
        PartitionColumn::new("p1", DataType::Int32, false),
    ])
    .with_missing_columns(vec![
        MissingColumn::new("m0", DataType::Int64, true),
        MissingColumn::new("m1", DataType::Int64, true),
    ])
    .with_column_roles(
        vec![
            "k".to_string(),
            "cnt".to_string(),
            "p0".to_string(),
            "p1".to_string(),
            "m0".to_string(),
            "m1".to_string(),
        ],
        Vec::new(),
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projected = builder.add_projection(
        source,
        projection_outputs(
            vec![Expr::column("p1"), Expr::column("m1"), Expr::column("k"), Expr::column("cnt").alias("count_input")],
            projection_schema,
        ),
    );
    let aggregate = builder.add_aggregate(
        projected,
        AggregateSpec::new(
            vec![Expr::column("p1"), Expr::column("m1"), Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("count_input"), "c_count")],
        ),
        aggregate_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

fn non_count_projection_pruning_plan() -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("unused", DataType::Int64, false),
    ]));
    let projection_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("sum_payload", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/non-count-source-pruning.parquet")],
        Arc::clone(&source_schema),
        vec!["k".to_string(), "payload".to_string(), "unused".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projected = builder.add_projection(
        source,
        projection_outputs(vec![Expr::column("k"), Expr::column("payload")], projection_schema),
    );
    let aggregate = builder.add_aggregate(
        projected,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "sum_payload")],
        ),
        aggregate_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

fn count_non_null_pruning_overridden_source_plan() -> QueryPlan {
    let physical_schema = Arc::new(Schema::new(vec![
        Field::new("raw_custkey", DataType::Int32, false),
        Field::new("raw_orderkey", DataType::Int32, false),
        Field::new("raw_status", DataType::Int32, false),
    ]));
    let source_output_schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_orderstatus", DataType::Int32, false),
    ]));
    let projection_schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("order_count_input", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));
    let file = LocalParquetFile::new("/tmp/count-non-null-overridden-orders.parquet")
        .with_row_groups(vec![0, 2])
        .with_row_group_null_counts(0, vec![Some(0), Some(0), Some(0)])
        .with_row_group_null_counts(1, vec![Some(0), Some(3), Some(0)]);
    let source = LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&physical_schema),
        vec!["raw_custkey".to_string(), "raw_orderkey".to_string(), "raw_status".to_string()],
    )
    .with_column_roles(vec!["o_custkey".to_string(), "o_orderkey".to_string(), "o_orderstatus".to_string()], Vec::new())
    .with_column_source_refs(
        vec![0, 1, 2],
        vec![SourceColumnRef::File(0), SourceColumnRef::File(1), SourceColumnRef::File(2)],
        Vec::new(),
    )
    .with_output_schema_override(Arc::clone(&source_output_schema));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filtered = builder.add_filter(
        source,
        Expr::binary(Expr::column("o_orderstatus"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1))),
        Arc::clone(&source_output_schema),
    );
    let projected = builder.add_projection(
        filtered,
        projection_outputs(
            vec![Expr::column("o_custkey"), Expr::column("o_orderkey").alias("order_count_input")],
            projection_schema,
        ),
    );
    let aggregate = builder.add_aggregate(
        projected,
        AggregateSpec::new(
            vec![Expr::column("o_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_count_input"), "c_count")],
        ),
        aggregate_schema,
    );
    builder.finish_with_host_sink(aggregate)
}

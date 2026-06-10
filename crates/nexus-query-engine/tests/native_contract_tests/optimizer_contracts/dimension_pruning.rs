#[test]
fn dimension_pruning_rejects_shared_target_lineage_before_mutating_source_row_groups() {
    let plan = shared_fact_target_dimension_pruning_plan();
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    let diagnostics = &result.diagnostics.dimension_derived_row_group_pruning;
    assert_eq!(diagnostics.considered_count, 4);
    assert_eq!(diagnostics.applied_count, 0);
    assert_eq!(
        diagnostics
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.target_key_column.as_deref() == Some("fact_key")
                    && candidate.rejected_reason
                        == Some(DimensionDerivedRowGroupPruningRejectedReason::TargetLineageReusedInPlan)
            })
            .count(),
        2
    );
    assert!(diagnostics.candidates.iter().all(|candidate| !candidate.applied));
}

#[test]
fn dimension_pruning_seeds_date_reader_predicate_from_aggregate_group_key_filter() {
    let dir = unique_contract_temp_dir("dimension-group-filter");
    let plan = aggregate_group_filter_dimension_pruning_plan(&dir);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.dimension_derived_row_group_pruning;
    assert_eq!(diagnostics.applied_count, 1);
    let applied = diagnostics.candidates.iter().find(|candidate| candidate.applied).expect("one candidate prunes");
    assert_eq!(applied.dimension_key_column.as_deref(), Some("d_date_sk"));
    assert_eq!(applied.target_key_column.as_deref(), Some("ss_sold_date_sk"));
    assert_eq!(applied.derived_min_key, Some(2));
    assert_eq!(applied.derived_max_key, Some(3));
    assert_eq!(applied.selected_row_groups_before, Some(4));
    assert_eq!(applied.selected_row_groups_after, Some(2));

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
    let date_source = sources
        .iter()
        .find(|source| source.read_columns.iter().any(|column| column == "d_year"))
        .expect("date source should remain present");
    assert_eq!(date_source.reader_predicate.as_ref().map(ToString::to_string).as_deref(), Some("(d_year = 2000)"));
    let fact_source = sources
        .iter()
        .find(|source| source.read_columns.iter().any(|column| column == "ss_sold_date_sk"))
        .expect("fact source should remain present");
    assert_eq!(fact_source.files[0].row_groups.as_deref(), Some(&[1, 2][..]));

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_reads_stats_from_leaf_for_nested_fact_schema() {
    let dir = unique_contract_temp_dir("dimension-nested-fact");
    let fact_path = dir.join("fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let payload_x = Arc::new(Field::new("x", DataType::Int32, false));
    let payload_y = Arc::new(Field::new("y", DataType::Int32, false));
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Struct(Fields::from(vec![payload_x.clone(), payload_y.clone()])), false),
        Field::new("fact_key", DataType::Int32, false),
    ]));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dim_key", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_nested_fact_parquet(
        &fact_path,
        Arc::clone(&fact_schema),
        payload_x,
        payload_y,
        Int32Array::from(vec![0, 0, 0]),
        Int32Array::from(vec![20, 100, 100]),
        Int32Array::from(vec![10, 20, 30]),
        1,
    );
    write_i32_record_batch_parquet(
        &dim_path,
        Arc::clone(&dim_schema),
        vec![Int32Array::from(vec![20, 40]), Int32Array::from(vec![2000, 1999])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["fact_key".to_string()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&dim_path)],
        Arc::clone(&dim_schema),
        vec!["dim_key".to_string(), "d_year".to_string()],
    )));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("fact_key", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    let joined = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_year", "d_year"), "d_year"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("d_year", DataType::Int32, false),
        Field::new("cnt", DataType::Int64, true),
    ]));
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("d_year")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "cnt")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let filtered = builder.add_filter(
        aggregate,
        Expr::binary(Expr::column("d_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
        aggregate_schema,
    );
    let plan = builder.finish_gpu_resident(filtered);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let applied = result
        .diagnostics
        .dimension_derived_row_group_pruning
        .candidates
        .iter()
        .find(|candidate| candidate.applied)
        .expect("nested fact source should be pruned by fact_key stats");
    assert_eq!(applied.target_key_column.as_deref(), Some("fact_key"));
    assert_eq!(applied.selected_row_groups_before, Some(3));
    assert_eq!(applied.selected_row_groups_after, Some(1));

    let fact_source = result
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
        .find(|source| source.read_columns.iter().any(|column| column == "fact_key"))
        .expect("fact source should remain present");
    assert_eq!(fact_source.files[0].row_groups.as_deref(), Some(&[1][..]));

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_seeded_plan_exposes_group_key_filter_discovery_request() {
    let dir = unique_contract_temp_dir("dimension-group-filter-discovery");
    let plan = aggregate_group_filter_dimension_pruning_plan(&dir);

    assert!(discover_dimension_derived_pruning_requests(&plan).is_empty());

    let seeded = seed_dimension_derived_reader_predicates(plan);
    let requests = discover_dimension_derived_pruning_requests(&seeded);

    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].dimension_key_column, "d_date_sk");
    assert_eq!(requests[0].target_key_column, "ss_sold_date_sk");

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_does_not_seed_unsupported_reader_predicate() {
    let dir = unique_contract_temp_dir("dimension-group-filter-unsupported");
    let plan = aggregate_group_filter_dimension_pruning_plan_with_predicate(
        &dir,
        Expr::binary(
            Expr::try_cast(Expr::column("ss_sold_year"), DataType::Int64),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Int64(2000)),
        ),
    );

    let seeded = seed_dimension_derived_reader_predicates(plan.clone());

    assert_eq!(seeded, plan);
    assert!(discover_dimension_derived_pruning_requests(&seeded).is_empty());

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_external_stats_narrow_existing_row_group_selection() {
    let mut builder = QueryPlanBuilder::new();
    let schema = Arc::new(Schema::new(vec![Field::new("fact_key", DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/external-stats-row-groups.parquet").with_row_groups(vec![1, 3]);
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&schema),
        vec!["fact_key".to_string()],
    )));
    let plan = builder.finish_gpu_resident(source);
    let result = apply_dimension_derived_pruning_with_stats(
        plan,
        source,
        &[vec![
            Some(nexus_query_engine::DimensionDerivedKeyRange { min: 10, max: 19 }),
            Some(nexus_query_engine::DimensionDerivedKeyRange { min: 30, max: 39 }),
        ]],
        &nexus_query_engine::DimensionKeyFilter::from_bounds(nexus_query_engine::DimensionDerivedKeyRange {
            min: 35,
            max: 35,
        }),
    );

    assert!(result.applied);
    assert_eq!(result.before, 2);
    assert_eq!(result.after, 1);
    let PlanNode::Source(source) = &result.plan.node(source).expect("source should remain present").kind else {
        panic!("node should remain a source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("source should remain local parquet");
    };
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[3][..]));
}

#[test]
fn dimension_pruning_scattered_keyset_prunes_gaps_a_single_range_would_keep() {
    // The qualifying keyset {5, 35} bounds to [5, 35]. A single-range filter
    // overlaps BOTH row groups ([10,19] and [30,39]); the set-aware filter keeps
    // only the row group containing 35, proving runs prune the inner gap.
    let mut builder = QueryPlanBuilder::new();
    let schema = Arc::new(Schema::new(vec![Field::new("fact_key", DataType::Int32, false)]));
    let file = LocalParquetFile::new("/tmp/scattered-keyset-row-groups.parquet").with_row_groups(vec![1, 3]);
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&schema),
        vec!["fact_key".to_string()],
    )));
    let plan = builder.finish_gpu_resident(source);

    let scattered =
        nexus_query_engine::DimensionKeyFilter::from_keys(&[5, 35]).expect("non-empty keyset builds a filter");
    assert_eq!(scattered.run_count(), 2);

    let result = apply_dimension_derived_pruning_with_stats(
        plan,
        source,
        &[vec![
            Some(nexus_query_engine::DimensionDerivedKeyRange { min: 10, max: 19 }),
            Some(nexus_query_engine::DimensionDerivedKeyRange { min: 30, max: 39 }),
        ]],
        &scattered,
    );

    assert!(result.applied);
    assert_eq!(result.before, 2);
    assert_eq!(result.after, 1);
    let PlanNode::Source(source) = &result.plan.node(source).expect("source should remain present").kind else {
        panic!("node should remain a source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("source should remain local parquet");
    };
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[3][..]));
}

#[test]
fn dimension_pruning_does_not_seed_right_side_left_join_group_key_filter() {
    let dir = unique_contract_temp_dir("dimension-left-right-filter");
    let plan = aggregate_right_group_filter_left_join_plan(&dir);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dimension_derived_row_group_pruning.applied_count, 0);
    assert!(result.plan.nodes().iter().all(|node| match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => source.reader_predicate.is_none(),
            _ => true,
        },
        _ => true,
    }));

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_does_not_seed_reader_predicate_through_shared_aggregate_subtree() {
    let dir = unique_contract_temp_dir("dimension-shared-group-filter");
    let plan = shared_aggregate_group_filter_dimension_pruning_plan(&dir);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options);

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dimension_derived_row_group_pruning.applied_count, 0);
    assert!(result.plan.nodes().iter().all(|node| match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => source.reader_predicate.is_none(),
            _ => true,
        },
        _ => true,
    }));

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

fn shared_fact_target_dimension_pruning_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source("store_sales", "fact_key", 512 * 1024 * 1024, 10_000_000));
    let output_schema = schema(&["dim_key", "fact_key"]);
    let mut joins = Vec::new();

    for (name, day) in [("date_dim_a", 1), ("date_dim_b", 2)] {
        let dim = builder.add_source(estimated_date_source_with_day_predicate(name, day));
        let join = builder.add_join(
            dim,
            fact,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "dim_key", "dim_key"),
                    ColumnRef::input(1, "fact_key", "fact_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "dim_key", "dim_key"), "dim_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "fact_key", "fact_key"), "fact_key"),
            ]),
            Arc::clone(&output_schema),
        );
        joins.push(join);
    }

    let union = builder.add_union(joins, output_schema);
    builder.finish_gpu_resident(union)
}

fn aggregate_group_filter_dimension_pruning_plan(dir: &Path) -> nexus_query_engine::QueryPlan {
    aggregate_group_filter_dimension_pruning_plan_with_predicate(
        dir,
        Expr::binary(Expr::column("ss_sold_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
    )
}

fn aggregate_group_filter_dimension_pruning_plan_with_predicate(
    dir: &Path,
    filter_predicate: Expr,
) -> nexus_query_engine::QueryPlan {
    let fact_path = dir.join("store_sales.parquet");
    let date_path = dir.join("date_dim.parquet");
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_date_sk", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
    ]));
    let date_schema = Arc::new(Schema::new(vec![
        Field::new("d_date_sk", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(
        &fact_path,
        Arc::clone(&fact_schema),
        vec![
            Int32Array::from(vec![1, 2, 3, 4]),
            Int32Array::from(vec![10, 20, 30, 40]),
            Int32Array::from(vec![1, 1, 1, 1]),
        ],
        1,
    );
    write_i32_record_batch_parquet(
        &date_path,
        Arc::clone(&date_schema),
        vec![Int32Array::from(vec![1, 2, 3, 4]), Int32Array::from(vec![1999, 2000, 2000, 2001])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["ss_sold_date_sk".to_string(), "ss_item_sk".to_string(), "ss_quantity".to_string()],
    )));
    let date = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&date_path)],
        Arc::clone(&date_schema),
        vec!["d_date_sk".to_string(), "d_year".to_string()],
    )));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_date_sk", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    let joined = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"), "ss_sold_date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "ss_item_sk", "ss_item_sk"), "ss_item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "ss_quantity", "ss_quantity"), "ss_quantity"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_year", "d_year"), "d_year"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_year", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_qty", DataType::Int64, true),
    ]));
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("d_year").alias("ss_sold_year"), Expr::column("ss_item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("ss_quantity"), "ss_qty")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let filtered = builder.add_filter(aggregate, filter_predicate, aggregate_schema);
    builder.finish_gpu_resident(filtered)
}

fn shared_aggregate_group_filter_dimension_pruning_plan(dir: &Path) -> nexus_query_engine::QueryPlan {
    let fact_path = dir.join("store_sales.parquet");
    let date_path = dir.join("date_dim.parquet");
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_date_sk", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
    ]));
    let date_schema = Arc::new(Schema::new(vec![
        Field::new("d_date_sk", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(
        &fact_path,
        Arc::clone(&fact_schema),
        vec![
            Int32Array::from(vec![1, 2, 3, 4]),
            Int32Array::from(vec![10, 20, 30, 40]),
            Int32Array::from(vec![1, 1, 1, 1]),
        ],
        1,
    );
    write_i32_record_batch_parquet(
        &date_path,
        Arc::clone(&date_schema),
        vec![Int32Array::from(vec![1, 2, 3, 4]), Int32Array::from(vec![1999, 2000, 2000, 2001])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["ss_sold_date_sk".to_string(), "ss_item_sk".to_string(), "ss_quantity".to_string()],
    )));
    let date = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&date_path)],
        Arc::clone(&date_schema),
        vec!["d_date_sk".to_string(), "d_year".to_string()],
    )));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_date_sk", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    let joined = builder.add_join(
        fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"), "ss_sold_date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "ss_item_sk", "ss_item_sk"), "ss_item_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "ss_quantity", "ss_quantity"), "ss_quantity"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_year", "d_year"), "d_year"),
        ]),
        Arc::clone(&join_schema),
    );
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("ss_sold_year", DataType::Int32, false),
        Field::new("ss_item_sk", DataType::Int32, false),
        Field::new("ss_qty", DataType::Int64, true),
    ]));
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("d_year").alias("ss_sold_year"), Expr::column("ss_item_sk")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("ss_quantity"), "ss_qty")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let filtered = builder.add_filter(
        aggregate,
        Expr::binary(Expr::column("ss_sold_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
        Arc::clone(&aggregate_schema),
    );
    let union = builder.add_union(vec![filtered, aggregate], aggregate_schema);
    builder.finish_gpu_resident(union)
}

fn aggregate_right_group_filter_left_join_plan(dir: &Path) -> nexus_query_engine::QueryPlan {
    let fact_path = dir.join("left_fact.parquet");
    let dim_path = dir.join("right_dim.parquet");
    let fact_schema = Arc::new(Schema::new(vec![Field::new("fact_key", DataType::Int32, false)]));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dim_key", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(&fact_path, Arc::clone(&fact_schema), vec![Int32Array::from(vec![1, 2, 3])], 1);
    write_i32_record_batch_parquet(
        &dim_path,
        Arc::clone(&dim_schema),
        vec![Int32Array::from(vec![1, 2]), Int32Array::from(vec![1999, 2000])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["fact_key".to_string()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&dim_path)],
        Arc::clone(&dim_schema),
        vec!["dim_key".to_string(), "d_year".to_string()],
    )));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("fact_key", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, true),
    ]));
    let joined = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_year", "d_year"), "d_year"),
        ]),
        join_schema,
    );
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("d_year", DataType::Int32, true),
        Field::new("cnt", DataType::Int64, true),
    ]));
    let aggregate = builder.add_aggregate(
        joined,
        AggregateSpec::new(
            vec![Expr::column("d_year")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "cnt")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let filtered = builder.add_filter(
        aggregate,
        Expr::binary(Expr::column("d_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
        aggregate_schema,
    );
    builder.finish_gpu_resident(filtered)
}

fn write_i32_record_batch_parquet(path: &Path, schema: Arc<Schema>, columns: Vec<Int32Array>, row_group_rows: usize) {
    let arrays = columns.into_iter().map(|column| Arc::new(column) as Arc<dyn arrow::array::Array>).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).expect("test batch should be valid");
    let file = File::create(path).expect("test parquet file should be creatable");
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_group_rows)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("test parquet writer should be creatable");
    writer.write(&batch).expect("test parquet batch should be writable");
    writer.close().expect("test parquet writer should close");
}

fn write_nested_fact_parquet(
    path: &Path,
    schema: Arc<Schema>,
    payload_x: Arc<Field>,
    payload_y: Arc<Field>,
    x_values: Int32Array,
    y_values: Int32Array,
    fact_key: Int32Array,
    row_group_rows: usize,
) {
    let x_values = Arc::new(x_values) as ArrayRef;
    let y_values = Arc::new(y_values) as ArrayRef;
    let payload = Arc::new(StructArray::from(vec![(payload_x, x_values), (payload_y, y_values)])) as ArrayRef;
    let fact_key = Arc::new(fact_key) as ArrayRef;
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![payload, fact_key]).expect("test batch should be valid");
    let file = File::create(path).expect("test parquet file should be creatable");
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_group_rows)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("test parquet writer should be creatable");
    writer.write(&batch).expect("test parquet batch should be writable");
    writer.close().expect("test parquet writer should close");
}

fn unique_contract_temp_dir(label: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock should be monotonic").as_nanos();
    let dir = std::env::temp_dir().join(format!("datafusion-nexus-{label}-{id}"));
    fs::create_dir_all(&dir).expect("temporary test directory should be creatable");
    dir
}

fn estimated_date_source_with_day_predicate(name: &str, day: i32) -> SourceNode {
    let source_schema = schema(&["dim_key", "day"]);
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet"))
        .with_stable_id(name)
        .with_projected_row_group_bytes(vec![1024 * 1024], vec![1024 * 1024])
        .with_row_group_rows(vec![365]);
    let predicate = Expr::binary(Expr::column("day"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(day)));
    SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![file], source_schema, vec!["dim_key".to_string(), "day".to_string()])
            .with_reader_predicate(predicate),
    )
}

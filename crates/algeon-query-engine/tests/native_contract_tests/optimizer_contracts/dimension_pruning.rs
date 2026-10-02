use arrow::array::Array;
use arrow::array::Int64Array;
use arrow::array::StringArray;
use arrow::array::StringViewArray;
use query_engine::optimizer::DimensionKeyFilter;
use query_engine::optimizer::KeyRange;
use query_engine::optimizer::dimension_key_filter_from_batch;

#[test]
fn dimension_pruning_rejects_shared_target_lineage_before_mutating_source_row_groups() {
    let plan = shared_fact_target_dimension_pruning_plan();
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan.clone(), &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

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

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

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
fn dimension_pruning_traces_target_union_branch_aliases() {
    let dir = unique_contract_temp_dir("dimension-union-target-aliases");
    let fact_a_path = dir.join("fact_a.parquet");
    let fact_b_path = dir.join("fact_b.parquet");
    let date_path = dir.join("date_dim.parquet");
    let fact_a_schema = Arc::new(Schema::new(vec![Field::new("a_date_sk", DataType::Int32, false)]));
    let fact_b_schema = Arc::new(Schema::new(vec![Field::new("b_date_sk", DataType::Int32, false)]));
    let date_schema = Arc::new(Schema::new(vec![
        Field::new("d_date_sk", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(
        &fact_a_path,
        Arc::clone(&fact_a_schema),
        vec![Int32Array::from(vec![1, 2, 3, 4])],
        1,
    );
    write_i32_record_batch_parquet(
        &fact_b_path,
        Arc::clone(&fact_b_schema),
        vec![Int32Array::from(vec![3, 5, 2, 6])],
        1,
    );
    write_i32_record_batch_parquet(
        &date_path,
        Arc::clone(&date_schema),
        vec![Int32Array::from(vec![1, 2, 3, 4]), Int32Array::from(vec![1999, 2000, 2000, 2001])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact_a = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_a_path)],
        Arc::clone(&fact_a_schema),
        vec!["a_date_sk".to_string()],
    )));
    let fact_b = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_b_path)],
        Arc::clone(&fact_b_schema),
        vec!["b_date_sk".to_string()],
    )));
    let union_schema = Arc::new(Schema::new(vec![Field::new("date_sk", DataType::Int32, false)]));
    let fact_a_projected = builder.add_projection(
        fact_a,
        projection_outputs(vec![Expr::column("a_date_sk").alias("date_sk")], Arc::clone(&union_schema)),
    );
    let fact_b_projected = builder.add_projection(
        fact_b,
        projection_outputs(vec![Expr::column("b_date_sk").alias("date_sk")], Arc::clone(&union_schema)),
    );
    let union = builder.add_union(vec![fact_a_projected, fact_b_projected], Arc::clone(&union_schema));
    let date = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&date_path)],
            Arc::clone(&date_schema),
            vec!["d_date_sk".to_string(), "d_year".to_string()],
        )
        .with_reader_predicate(Expr::binary(
            Expr::column("d_year"),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Int32(2000)),
        )),
    ));
    let date_key_schema = Arc::new(Schema::new(vec![Field::new("d_date_sk", DataType::Int32, false)]));
    let date_projected =
        builder.add_projection(date, projection_outputs(vec![Expr::column("d_date_sk")], Arc::clone(&date_key_schema)));
    let joined = builder.add_join(
        union,
        date_projected,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk")]),
        Arc::clone(&union_schema),
    );
    let plan = builder.finish_gpu_resident(joined);

    let requests = discover_dimension_derived_pruning_requests(&plan);
    assert_eq!(requests.len(), 2);
    let mut request_target_columns =
        requests.iter().map(|request| request.target_key_column.as_str()).collect::<Vec<_>>();
    request_target_columns.sort_unstable();
    assert_eq!(request_target_columns, vec!["a_date_sk", "b_date_sk"]);

    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };
    let result = optimize_query_plan(plan, &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let diagnostics = &result.diagnostics.dimension_derived_row_group_pruning;
    assert_eq!(diagnostics.considered_count, 3);
    assert_eq!(diagnostics.applied_count, 2);
    assert_eq!(
        diagnostics.candidates[0].rejected_reason,
        Some(DimensionDerivedRowGroupPruningRejectedReason::DimensionKeyLineageUnavailable)
    );
    assert_eq!(diagnostics.candidates[1].target_source_node_id, Some(fact_a));
    assert_eq!(diagnostics.candidates[2].target_source_node_id, Some(fact_b));
    assert!(diagnostics.candidates.iter().any(|candidate| {
        candidate.applied
            && candidate.target_source_node_id == Some(fact_a)
            && candidate.target_key_column.as_deref() == Some("a_date_sk")
            && candidate.selected_row_groups_after == Some(2)
    }));
    assert!(diagnostics.candidates.iter().any(|candidate| {
        candidate.applied
            && candidate.target_source_node_id == Some(fact_b)
            && candidate.target_key_column.as_deref() == Some("b_date_sk")
            && candidate.selected_row_groups_after == Some(2)
    }));

    let PlanNode::Source(source) = &result.plan.node(fact_a).expect("fact_a source should remain").kind else {
        panic!("fact_a node should remain a source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("fact_a node should remain local parquet");
    };
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1, 2][..]));

    let PlanNode::Source(source) = &result.plan.node(fact_b).expect("fact_b source should remain").kind else {
        panic!("fact_b node should remain a source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("fact_b node should remain local parquet");
    };
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[0, 2][..]));

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
        Field::new("payload", DataType::Struct(Fields::from(vec![Arc::clone(&payload_x), Arc::clone(&payload_y)])), false),
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

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

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
fn dimension_pruning_key_accumulator_supports_q16_string_predicate_with_utf8view() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int32, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8View, false),
        Field::new("p_size", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Brand#11", "Brand#45", "Brand#22", "Brand#33", "Brand#44"]))
                as ArrayRef,
            Arc::new(StringViewArray::from(vec![
                "SMALL BRUSHED",
                "SMALL BRUSHED",
                "MEDIUM POLISHED COPPER",
                "LARGE ANODIZED",
                "MEDIUM POLISHED STEEL",
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![49, 49, 49, 3, 14])) as ArrayRef,
        ],
    )
    .expect("test batch should be valid");

    let filter = dimension_key_filter_from_batch(
        &batch,
        "p_partkey",
        &part_brand_type_size_predicate(),
    )
    .expect("part string predicate should be evaluable")
    .expect("qualifying part keys should build a filter");

    assert_eq!(filter.bounds(), KeyRange { min: 10, max: 40 });
    assert_eq!(filter.run_count(), 2);
    assert!(filter.overlaps(KeyRange { min: 10, max: 10 }));
    assert!(filter.overlaps(KeyRange { min: 40, max: 40 }));
    assert!(!filter.overlaps(KeyRange { min: 20, max: 30 }));
}

#[test]
fn dimension_pruning_derives_key_filter_from_string_not_like_part_predicate() {
    let dir = unique_contract_temp_dir("dimension-q16-part-string-filter");
    let fact_path = dir.join("partsupp.parquet");
    let part_path = dir.join("part.parquet");
    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Int32, false),
    ]));
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int32, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(
        &fact_path,
        Arc::clone(&fact_schema),
        vec![Int32Array::from(vec![10, 20, 30, 40, 50]), Int32Array::from(vec![1, 1, 1, 1, 1])],
        1,
    );
    write_part_brand_type_size_dimension_parquet(&part_path, Arc::clone(&part_schema), 1);

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["ps_partkey".to_string(), "ps_supplycost".to_string()],
    )));
    let part = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&part_path)],
            Arc::clone(&part_schema),
            vec!["p_partkey".to_string(), "p_brand".to_string(), "p_type".to_string(), "p_size".to_string()],
        )
        .with_reader_predicate(part_brand_type_size_predicate()),
    ));
    let join_schema = Arc::new(Schema::new(vec![Field::new("ps_partkey", DataType::Int32, false)]));
    let joined = builder.add_join(
        fact,
        part,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                ColumnRef::input(1, "p_partkey", "p_partkey"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "ps_partkey", "ps_partkey"),
            "ps_partkey",
        )]),
        Arc::clone(&join_schema),
    );
    let plan = builder.finish_gpu_resident(joined);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let applied = result
        .diagnostics
        .dimension_derived_row_group_pruning
        .candidates
        .iter()
        .find(|candidate| candidate.applied)
        .expect("part brand/type/size predicate should prune partsupp row groups");
    assert_eq!(applied.dimension_key_column.as_deref(), Some("p_partkey"));
    assert_eq!(applied.target_key_column.as_deref(), Some("ps_partkey"));
    assert_eq!(applied.derived_min_key, Some(10));
    assert_eq!(applied.derived_max_key, Some(40));
    assert_eq!(applied.selected_row_groups_before, Some(5));
    assert_eq!(applied.selected_row_groups_after, Some(2));

    let PlanNode::Source(source) = &result.plan.node(fact).expect("fact source node should remain").kind else {
        panic!("fact node should remain a source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("fact node should remain local parquet");
    };
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[0, 3][..]));

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_does_not_trace_target_lineage_through_limit() {
    let dir = unique_contract_temp_dir("dimension-limit-target");
    let fact_path = dir.join("fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let fact_schema = Arc::new(Schema::new(vec![Field::new("fact_key", DataType::Int32, false)]));
    let dim_schema = Arc::new(Schema::new(vec![Field::new("dim_key", DataType::Int32, false)]));
    write_i32_record_batch_parquet(&fact_path, Arc::clone(&fact_schema), vec![Int32Array::from(vec![1, 2, 3])], 1);
    write_i32_record_batch_parquet(&dim_path, Arc::clone(&dim_schema), vec![Int32Array::from(vec![2])], 1);

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&fact_schema),
        vec!["fact_key".to_string()],
    )));
    let limited_fact = builder.add_limit(fact, LimitSpec::fetch(1), Arc::clone(&fact_schema));
    let dim = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&dim_path)],
            Arc::clone(&dim_schema),
            vec!["dim_key".to_string()],
        )
        .with_reader_predicate(Expr::binary(
            Expr::column("dim_key"),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Int32(2)),
        )),
    ));
    let joined = builder.add_join(
        limited_fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")]),
        Arc::clone(&fact_schema),
    );
    let plan = builder.finish_gpu_resident(joined);
    let options = NativeOptimizerOptions {
        dimension_derived_row_group_pruning: DimensionDerivedRowGroupPruningOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.dimension_derived_row_group_pruning.applied_count, 0);
    assert!(result.diagnostics.dimension_derived_row_group_pruning.candidates.iter().any(|candidate| {
        candidate.target_source_node_id.is_none()
            && candidate.rejected_reason
                == Some(DimensionDerivedRowGroupPruningRejectedReason::TargetKeyLineageUnavailable)
    }));
    let fact_source = result
        .plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) if source.read_columns.iter().any(|column| column == "fact_key") => {
                    Some(source)
                },
                _ => None,
            },
            _ => None,
        })
        .expect("fact source should remain present");
    assert_eq!(fact_source.files[0].row_groups, None);

    fs::remove_dir_all(dir).expect("temporary dimension pruning fixture should be removable");
}

#[test]
fn dimension_pruning_seeded_plan_exposes_group_key_filter_discovery_request() {
    let dir = unique_contract_temp_dir("dimension-group-filter-discovery");
    let plan = aggregate_group_filter_dimension_pruning_plan(&dir);

    assert!(discover_dimension_derived_pruning_requests(&plan).is_empty());

    let seeded = seed_dimension_derived_reader_predicates(plan, rapids_analysis_context(PlanningPolicy::default()));
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

    let seeded = seed_dimension_derived_reader_predicates(plan.clone(), rapids_analysis_context(PlanningPolicy::default()));

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
            Some(KeyRange { min: 10, max: 19 }),
            Some(KeyRange { min: 30, max: 39 }),
        ]],
        &DimensionKeyFilter::from_bounds(KeyRange {
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

    let scattered = DimensionKeyFilter::from_keys(&[5, 35])
        .expect("non-empty keyset builds a filter");
    assert_eq!(scattered.run_count(), 2);

    let result = apply_dimension_derived_pruning_with_stats(
        plan,
        source,
        &[vec![
            Some(KeyRange { min: 10, max: 19 }),
            Some(KeyRange { min: 30, max: 39 }),
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

    let result = optimize_query_plan(plan.clone(), &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

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

    let result = optimize_query_plan(plan.clone(), &options, rapids_analysis_context(PlanningPolicy::default()), local_dimension_pruning_resolver());

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

fn shared_fact_target_dimension_pruning_plan() -> QueryPlan {
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

fn aggregate_group_filter_dimension_pruning_plan(dir: &Path) -> QueryPlan {
    aggregate_group_filter_dimension_pruning_plan_with_predicate(
        dir,
        Expr::binary(Expr::column("ss_sold_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
    )
}

fn aggregate_group_filter_dimension_pruning_plan_with_predicate(
    dir: &Path,
    filter_predicate: Expr,
) -> QueryPlan {
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

fn shared_aggregate_group_filter_dimension_pruning_plan(dir: &Path) -> QueryPlan {
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

fn aggregate_right_group_filter_left_join_plan(dir: &Path) -> QueryPlan {
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

fn part_brand_type_size_predicate() -> Expr {
    let brand_ok = Expr::binary(
        Expr::column("p_brand"),
        BinaryOp::NotEq,
        Expr::literal(LiteralValue::Utf8("Brand#45".to_string())),
    );
    let type_ok = Expr::not_like_escape(Expr::column("p_type"), "MEDIUM POLISHED%", "\\");
    let size_ok = Expr::in_list(
        Expr::column("p_size"),
        [49, 14, 23, 45, 19, 3, 36, 9].into_iter().map(|size| Expr::literal(LiteralValue::Int32(size))).collect(),
    );
    Expr::binary(Expr::binary(brand_ok, BinaryOp::And, type_ok), BinaryOp::And, size_ok)
}

fn write_i32_record_batch_parquet(path: &Path, schema: Arc<Schema>, columns: Vec<Int32Array>, row_group_rows: usize) {
    let arrays = columns.into_iter().map(|column| Arc::new(column) as Arc<dyn Array>).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).expect("test batch should be valid");
    write_record_batch_parquet(path, schema, &batch, row_group_rows);
}

fn write_part_brand_type_size_dimension_parquet(path: &Path, schema: Arc<Schema>, row_group_rows: usize) {
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Brand#11", "Brand#45", "Brand#22", "Brand#33", "Brand#44"]))
                as ArrayRef,
            Arc::new(StringArray::from(vec![
                "SMALL BRUSHED",
                "SMALL BRUSHED",
                "MEDIUM POLISHED COPPER",
                "LARGE ANODIZED",
                "MEDIUM POLISHED STEEL",
            ])) as ArrayRef,
            Arc::new(Int32Array::from(vec![49, 49, 49, 3, 14])) as ArrayRef,
        ],
    )
    .expect("test batch should be valid");
    write_record_batch_parquet(path, schema, &batch, row_group_rows);
}

fn write_record_batch_parquet(path: &Path, schema: Arc<Schema>, batch: &RecordBatch, row_group_rows: usize) {
    let file = File::create(path).expect("test parquet file should be creatable");
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_group_rows)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).expect("test parquet writer should be creatable");
    writer.write(batch).expect("test parquet batch should be writable");
    writer.close().expect("test parquet writer should close");
}

#[expect(clippy::too_many_arguments, reason = "test fixture parameters are independent fixture axes")]
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
    let dir = std::env::temp_dir().join(format!("algeon-datafusion-{label}-{id}"));
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

#[test]
fn seed_dimension_derived_reader_predicates_preserves_dynamic_filter_routes() {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let file = LocalParquetFile::new("/tmp/test.parquet");
    let source = LocalParquetSource::from_files(vec![file], Arc::clone(&schema), vec!["x".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_node_id = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(
        source_node_id,
        AggregateSpec::new(vec![Expr::column("x")], Vec::new()),
        Arc::clone(&schema),
    );
    let filter = builder.add_filter(
        aggregate,
        Expr::binary(Expr::column("x"), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(5))),
        Arc::clone(&schema),
    );
    let route = test_dynamic_filter_route(source_node_id);
    let plan = builder.finish_with_host_sink(filter).with_dynamic_filter_routes(vec![route.clone()]);

    let result = seed_dimension_derived_reader_predicates(plan, rapids_analysis_context(PlanningPolicy::default()));

    assert_eq!(result.dynamic_filter_routes(), &[route]);
    let source = local_parquet_source(&result, source_node_id).unwrap();
    assert!(source.reader_predicate.as_ref().is_some_and(|predicate| {
        *predicate == Expr::binary(Expr::column("x"), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(5)))
    }));
}

#[test]
fn optimize_dimension_derived_row_group_pruning_preserves_dynamic_filter_routes_when_pruning_applies() {
    let dir = unique_contract_temp_dir("dimension-pruning-routes");
    let fact_path = dir.join("fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    write_i64_parquet(&fact_path, Arc::clone(&schema), &[0, 1, 5, 6, 10, 11], 2);
    write_i64_parquet(&dim_path, Arc::clone(&schema), &[5], 1);

    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(vec![LocalParquetFile::new(&dim_path)], Arc::clone(&schema), vec![
            "x".to_string()
        ])
        .with_reader_predicate(Expr::binary(Expr::column("x"), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(5)))),
    ));
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&fact_path)],
        Arc::clone(&schema),
        vec!["x".to_string()],
    )));
    let join = builder.add_join(
        dim,
        fact,
        JoinSpec::new(JoinType::Inner, vec![JoinKey::new(ColumnRef::input(0, "x", "x"), ColumnRef::input(1, "x", "x"))])
            .with_output_projection(vec![JoinOutputColumn::right(ColumnRef::input(1, "x", "x"), "x")]),
        Arc::clone(&schema),
    );
    let route = test_dynamic_filter_route(fact);
    let plan = builder.finish_with_host_sink(join).with_dynamic_filter_routes(vec![route.clone()]);

    let result = optimize_dimension_derived_row_group_pruning(
        plan,
        &DimensionDerivedRowGroupPruningOptions::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    );

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.applied_count, 1);
    assert_eq!(result.plan.dynamic_filter_routes(), &[route]);
    let source = local_parquet_source(&result.plan, fact).unwrap();
    assert_eq!(source.files[0].row_groups, Some(vec![1]));

    fs::remove_dir_all(dir).expect("temporary dimension pruning directory should be removable");
}

fn test_dynamic_filter_route(source_node_id: NodeId) -> DynamicFilterRoute {
    DynamicFilterRoute::new(
        DynamicFilterId::new(0),
        DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id: source_node_id },
        DynamicFilterConsumer::LocalParquetSource { source_node_id },
        vec!["x".to_string()],
        vec!["x".to_string()],
        vec![DynamicFilterKind::Range],
        DynamicFilterConsumerLowering::ReaderPredicate,
    )
}

fn write_i64_parquet(path: &Path, schema: Arc<Schema>, values: &[i64], row_group_rows: usize) {
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(
        values.to_vec(),
    ))])
    .expect("test record batch should be valid");
    write_record_batch_parquet(path, schema, &batch, row_group_rows);
}

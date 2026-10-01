use query_engine::{
    admission::{admit, bounded_join_unsupported_shape_facts},
    dimension_pruning::local_dimension_pruning_resolver,
    dynamic_filter::{
        DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterProducer,
        LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON,
    },
    optimizer::{
        DynamicFilterRouteDiscoveryOptions, JoinBuildSideSelectionOptions, NativeOptimizerOptions, optimize_query_plan,
    },
    rapids_backend::rapids_analysis_context,
};

#[test]
fn bounded_native_query_applies_hash_join_build_dynamic_filter_to_probe_source() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-dynamic-filter");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    write_join_right_keys_parquet(&runtime, &right_path, &[2]);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimized = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &optimized.plan).unwrap();
    let output = result.output().as_host_arrow().unwrap();
    assert_eq!(output.num_rows(), 2);
    assert_eq!(result.metrics().row_groups_committed, 2);

    fs::remove_dir_all(dir).unwrap();
}

// A full sort between a dynamic-filter consumer source and its join probe
// accepts only a resident frame, so the source cannot stay a deferred leaf.
#[test]
fn bounded_native_query_applies_dynamic_filter_to_source_under_full_sort_probe() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-dynamic-filter-sorted-probe");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    write_join_right_keys_parquet(&runtime, &right_path, &[2, 3]);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let qty_descending = || SortSpec::new(vec![SortExpr::new(Expr::column("qty"), true, false)]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let sorted_left = builder.add_sort(left, qty_descending(), Arc::clone(&left_schema));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    )));
    let join = builder.add_join(
        sorted_left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sorted_join = builder.add_sort(join, qty_descending(), Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sorted_join);
    let optimized = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );
    assert_eq!(optimized.plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Sort(_))).count(), 2);
    assert!(optimized.plan.dynamic_filter_routes().iter().any(|route| matches!(
        route.consumer,
        DynamicFilterConsumer::LocalParquetSource { source_node_id } if source_node_id == left
    )));
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &optimized.plan).unwrap();
    let output = result.output().as_host_arrow().unwrap();
    assert_eq!(int32_array(output, 0).values().as_ref(), &[3, 3, 2, 2]);
    assert_eq!(int32_array(output, 1).values().as_ref(), &[31, 30, 21, 20]);
    // One build row group plus the two probe row groups inside the build-key range.
    assert_eq!(result.metrics().row_groups_committed, 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_applies_post_decode_filter_to_shared_local_source_frame() {
    let dir = unique_temp_dir("query-engine-bounded-shared-source-post-decode-dynamic-filter");
    let part_path = dir.join("part.parquet");
    let partsupp_path = dir.join("partsupp.parquet");
    let supplier_path = dir.join("supplier.parquet");
    let part_schema = write_single_i32_parquet(&part_path, "p_partkey", &[1, 3]);
    let partsupp_schema = write_partsupp_dynamic_filter_parquet(&partsupp_path);
    let supplier_schema = write_single_i32_parquet(&supplier_path, "s_suppkey", &[10, 20, 30]);
    let result_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Int32, false),
    ]));
    let partsupp_result_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_supplycost", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let part = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&part_path)
                .with_file_size_bytes(fs::metadata(&part_path).unwrap().len())
                .with_row_group_rows(vec![2]),
        ],
        Arc::clone(&part_schema),
        vec!["p_partkey".to_string()],
    )));
    let partsupp = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&partsupp_path)
                .with_file_size_bytes(fs::metadata(&partsupp_path).unwrap().len())
                .with_row_group_rows(vec![4]),
        ],
        Arc::clone(&partsupp_schema),
        vec!["ps_partkey".to_string(), "ps_suppkey".to_string(), "ps_supplycost".to_string()],
    )));
    let supplier = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&supplier_path)
                .with_file_size_bytes(fs::metadata(&supplier_path).unwrap().len())
                .with_row_group_rows(vec![3]),
        ],
        Arc::clone(&supplier_schema),
        vec!["s_suppkey".to_string()],
    )));
    let part_partsupp_join = builder.add_join(
        part,
        partsupp,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "p_partkey", "p_partkey"),
                ColumnRef::input(1, "ps_partkey", "ps_partkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "p_partkey", "p_partkey"), "p_partkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
        ])
        .with_build_side(JoinSide::Left),
        Arc::clone(&result_schema),
    );
    let supplier_partsupp = builder.add_join(
        supplier,
        partsupp,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "s_suppkey", "s_suppkey"),
                ColumnRef::input(1, "ps_suppkey", "ps_suppkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
        ])
        .with_build_side(JoinSide::Left),
        Arc::clone(&partsupp_result_schema),
    );
    let partsupp_min = builder.add_aggregate(
        supplier_partsupp,
        AggregateSpec::new(
            vec![Expr::column("ps_partkey")],
            vec![AggregateExpr::new(AggregateFunction::Min, Expr::column("ps_supplycost"), "min_supplycost")],
        ),
        Arc::new(Schema::new(vec![
            Field::new("ps_partkey", DataType::Int32, false),
            Field::new("min_supplycost", DataType::Int32, false),
        ])),
    );
    let final_join = builder.add_join(
        part_partsupp_join,
        partsupp_min,
        JoinSpec::new(
            JoinType::Inner,
            vec![
                JoinKey::new(
                    ColumnRef::input(0, "p_partkey", "p_partkey"),
                    ColumnRef::input(1, "ps_partkey", "ps_partkey"),
                ),
                JoinKey::new(
                    ColumnRef::input(0, "ps_supplycost", "ps_supplycost"),
                    ColumnRef::input(1, "min_supplycost", "min_supplycost"),
                ),
            ],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "p_partkey", "p_partkey"), "p_partkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&result_schema),
    );
    let plan = builder.finish_with_host_sink(final_join);
    let optimizer_options = NativeOptimizerOptions {
        dynamic_filter_route_discovery: DynamicFilterRouteDiscoveryOptions::default(),
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::disabled()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert!(optimized.plan.dynamic_filter_routes().iter().any(|route| {
        route.consumer_columns == ["ps_partkey".to_string()]
            && route.consumer_lowering == DynamicFilterConsumerLowering::PostDecodeOnly
            && matches!(
                route.consumer,
                DynamicFilterConsumer::LocalParquetSource { source_node_id }
                    if source_node_id == partsupp
            )
    }));
    let context = backend_context_with_grant(1024 * 1024);
    let result = execute_query_plan(&context, &optimized.plan).unwrap();
    let QueryOutput::HostArrow(output) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let metrics = result.metrics();
    assert_eq!(output.num_rows(), 2);
    assert_eq!(int32_array(output, 0).values().as_ref(), &[1, 3]);
    assert_eq!(int32_array(output, 1).values().as_ref(), &[5, 4]);
    assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
        row.event == "applied"
            && row.consumer_kind == "local_parquet_source"
            && row.consumer_node_id == partsupp.to_string()
            && row.consumer_lowering == "post_decode_only"
    }));
    let source_post_decode_rows = metrics
        .dynamic_filter_runtime_rows
        .iter()
        .filter(|row| {
            row.event == "applied_post_decode"
                && row.consumer_kind == "local_parquet_source"
                && row.consumer_node_id == partsupp.to_string()
                && row.consumer_lowering == "post_decode_only"
        })
        .collect::<Vec<_>>();
    assert_eq!(
        source_post_decode_rows.len(),
        1,
        "shared source post-decode dynamic filter should be materialized once for the shared frame"
    );
    assert!(source_post_decode_rows.iter().any(|row| {
        row.event == "applied_post_decode"
            && row.consumer_kind == "local_parquet_source"
            && row.consumer_node_id == partsupp.to_string()
            && row.consumer_lowering == "post_decode_only"
            && row.post_decode_rows_before == Some(4)
            && row.post_decode_rows_after == Some(2)
    }));
    assert_eq!(metrics.dynamic_filter_source_post_decode_rows_filtered, 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_batches_dynamic_reader_predicate_probe_source_chunks() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-batched-dynamic-reader-filter");
    let left_paths = [dir.join("lineitem-0.parquet"), dir.join("lineitem-1.parquet"), dir.join("lineitem-2.parquet")];
    let left_schema = write_dynamic_filter_probe_rows_parquet(&left_paths[0], &[1, 2], &[10, 20]);
    write_dynamic_filter_probe_rows_parquet(&left_paths[1], &[2, 3], &[21, 30]);
    write_dynamic_filter_probe_rows_parquet(&left_paths[2], &[2, 4], &[22, 40]);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        left_paths
            .iter()
            .map(|path| {
                LocalParquetFile::new(path)
                    .with_file_size_bytes(fs::metadata(path).unwrap().len())
                    .with_row_group_rows(vec![2])
            })
            .collect(),
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right_keys", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(8 * 1024 * 1024);

    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "right_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );
    {
        let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
        let output = result.output().as_host_arrow().unwrap();
        let metrics = result.metrics();
        assert_eq!(output.num_rows(), 3);
        assert_eq!(int32_array(output, 0).values().as_ref(), &[2, 2, 2]);
        assert_eq!(int32_array(output, 1).values().as_ref(), &[20, 21, 22]);
        assert_eq!(metrics.source_chunks_planned, 3);
        assert_eq!(metrics.source_chunks_attempted, 3);
        assert_eq!(metrics.source_chunks_committed, 3);
        assert_eq!(metrics.row_groups_committed, 3);
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_before, 6);
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_after, 3);
        assert_eq!(metrics.source_read_runtime_rows.len(), 1);
        let source_read_row = &metrics.source_read_runtime_rows[0];
        assert_eq!(source_read_row.read_kind, "chunk_batch");
        assert_eq!(source_read_row.read_strategy, "local_kvikio");
        assert_eq!(source_read_row.batch_index, 0);
        assert_eq!(source_read_row.chunk_count, 3);
        assert_eq!(source_read_row.file_count, 3);
        assert_eq!(source_read_row.row_group_count, 3);
        assert_eq!(source_read_row.read_columns_per_file, 2);
        assert_eq!(source_read_row.read_columns_total, 6);
        assert_eq!(source_read_row.output_rows, 3);
        assert!(source_read_row.reader_predicate);
        assert_eq!(source_read_row.reader_dynamic_filter_count, 1);
        assert!(source_read_row.elapsed > std::time::Duration::ZERO);
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "reader_predicate_rows"
                && row.source_file_index.is_none()
                && row.source_chunk_index.is_none()
                && row.reader_predicate_rows_before == Some(6)
                && row.reader_predicate_rows_after == Some(3)
        }));
    }
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_applies_residual_equality_dynamic_filter_before_downstream_dimension_build() {
    let dir = unique_temp_dir("query-engine-bounded-residual-equality-dynamic-filter");
    let sales_path = dir.join("catalog_sales.parquet");
    let d1_path = dir.join("date_dim_d1.parquet");
    let inventory_path = dir.join("inventory.parquet");
    let d2_path = dir.join("date_dim_d2.parquet");
    let sales_schema = write_q72_sales_parquet(&sales_path);
    let d1_schema = write_q72_d1_parquet(&d1_path);
    let inventory_schema = write_q72_inventory_parquet(&inventory_path);
    let d2_schema = write_q72_d2_parquet(&d2_path);
    let sales_with_d1_schema = Arc::new(Schema::new(vec![
        Field::new("cs_item_sk", DataType::Int32, false),
        Field::new("d_week_seq", DataType::Int32, false),
    ]));
    let inventory_with_d2_schema = Arc::new(Schema::new(vec![
        Field::new("inv_item_sk", DataType::Int32, false),
        Field::new("d_week_seq", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("cs_item_sk", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&sales_path)
                .with_file_size_bytes(fs::metadata(&sales_path).unwrap().len())
                .with_row_group_rows(vec![2]),
        ],
        Arc::clone(&sales_schema),
        vec!["cs_sold_date_sk".to_string(), "cs_item_sk".to_string()],
    )));
    let d1 = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(&d1_path)
                    .with_file_size_bytes(fs::metadata(&d1_path).unwrap().len())
                    .with_row_group_rows(vec![1]),
            ],
            Arc::clone(&d1_schema),
            vec!["d_date_sk".to_string(), "d_week_seq".to_string(), "d_year".to_string()],
        )
        .with_reader_predicate(Expr::binary(
            Expr::column("d_year"),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Int32(1999)),
        )),
    ));
    let inventory = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&inventory_path)
                .with_file_size_bytes(fs::metadata(&inventory_path).unwrap().len())
                .with_row_group_rows(vec![1, 1]),
        ],
        Arc::clone(&inventory_schema),
        vec!["inv_item_sk".to_string(), "inv_date_sk".to_string()],
    )));
    let d2 = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&d2_path)
                .with_file_size_bytes(fs::metadata(&d2_path).unwrap().len())
                .with_row_group_rows(vec![1, 1]),
        ],
        Arc::clone(&d2_schema),
        vec!["d_date_sk".to_string(), "d_week_seq".to_string()],
    )));
    let sales_with_d1 = builder.add_join(
        sales,
        d1,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "cs_sold_date_sk", "cs_sold_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "cs_item_sk", "cs_item_sk"), "cs_item_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_week_seq", "d_week_seq"), "d_week_seq"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&sales_with_d1_schema),
    );
    let inventory_with_d2 = builder.add_join(
        inventory,
        d2,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "inv_date_sk", "inv_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "inv_item_sk", "inv_item_sk"), "inv_item_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_week_seq", "d_week_seq"), "d_week_seq"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&inventory_with_d2_schema),
    );
    let root = builder.add_join(
        sales_with_d1,
        inventory_with_d2,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "cs_item_sk", "cs_item_sk"),
                ColumnRef::input(1, "inv_item_sk", "inv_item_sk"),
            )],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "d_week_seq", "d_week_seq"),
            BinaryOp::Eq,
            Expr::input_column(1, "d_week_seq", "d_week_seq"),
        ))
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "cs_item_sk", "cs_item_sk"),
            "cs_item_sk",
        )])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(root);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    let residual_route = optimized
        .plan
        .dynamic_filter_routes()
        .iter()
        .find(|route| {
            matches!(
                route.producer,
                DynamicFilterProducer::HashJoinBuild {
                    join_node_id,
                    build_side: JoinSide::Right,
                    ..
                } if join_node_id == sales_with_d1
            ) && matches!(
                route.consumer,
                DynamicFilterConsumer::LocalParquetSource {
                    source_node_id
                } if source_node_id == d2
            ) && route.producer_columns == ["d_week_seq".to_string()]
                && route.consumer_columns == ["d_week_seq".to_string()]
        })
        .cloned()
        .expect("residual equality should route filtered d1 week evidence into d2");
    let context = backend_context_with_grant(8 * 1024 * 1024);

    let result = execute_query_plan(&context, &optimized.plan).unwrap();
    let output = result.output().as_host_arrow().unwrap();
    let metrics = result.metrics();
    assert_eq!(output.num_rows(), 1);
    assert_eq!(int32_array(output, 0).values().as_ref(), &[10]);
    assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
        row.dynamic_filter_id == residual_route.id.to_string()
            && row.event == "applied"
            && row.consumer_node_id == d2.to_string()
            && row.selected_row_groups_before == Some(2)
            && row.selected_row_groups_after == Some(1)
    }));
    assert!(metrics.source_read_runtime_rows.iter().any(|row| {
        row.source_node_id == d2.to_string()
            && row.output_rows == 1
            && row.reader_predicate
            && row.reader_dynamic_filter_count == 1
    }));
    assert!(metrics.source_read_runtime_rows.iter().any(|row| {
        row.source_node_id == inventory.to_string()
            && row.output_rows == 1
            && row.reader_predicate
            && row.reader_dynamic_filter_count == 1
    }));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_prunes_hash_set_dynamic_filter_gap_row_group_off_probe_source() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-source-hash-set-dynamic-filter");
    let left_path = dir.join("lineitem.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let mut build_keys = vec![1, 3];
    build_keys.extend(4..=1026);
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(build_keys))]).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right_keys", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(1024 * 1024);

    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "right_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );
    {
        let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
        let output = result.output().as_host_arrow().unwrap();
        let metrics = result.metrics();
        assert_eq!(output.num_rows(), 4);
        assert_eq!(int32_array(output, 0).values().as_ref(), &[1, 1, 3, 3]);
        assert_eq!(int32_array(output, 1).values().as_ref(), &[10, 11, 30, 31]);
        assert_eq!(metrics.dynamic_filter_produced_count, 1);
        assert_eq!(metrics.dynamic_filter_applied_count, 1);
        // The build keyset is {1, 3, 4..=1026}; value 2 is absent. The probe's three
        // row groups carry order_id stats [1,1], [2,2], [3,3]. Set-aware pruning
        // coalesces the keyset into runs [1,1] and [3,1026] and drops the [2,2] row
        // group, whose only value sits in the gap and matches no build key. A
        // single-range filter ([1,1026]) would have retained all three. The output is
        // unchanged because the pruned row group joins to nothing.
        assert_eq!(metrics.dynamic_filter_row_groups_pruned, 1);
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "produced" && row.filter_kind == "hash_set_contains" && row.payload_value_count == Some(1025)
        }));
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "applied"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "hash_set_contains"
                && row.selected_row_groups_before == Some(3)
                && row.selected_row_groups_after == Some(2)
        }));
        assert!(!metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "applied_post_decode"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "hash_set_contains"
                && row.selected_row_groups_before.is_none()
        }));
        assert_eq!(metrics.dynamic_filter_source_post_decode_rows_filtered, 0);
        assert_eq!(metrics.dynamic_filter_filter_post_decode_rows_filtered, 0);
        assert_eq!(metrics.dynamic_filter_join_probe_post_decode_rows_filtered, 0);
    }
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_defers_large_small_in_list_reader_filter_when_sibling_reader_filter_remains() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-deferred-small-in-list-reader-filter");
    let fact_path = dir.join("lineitem.parquet");
    let fact_schema = write_dynamic_filter_probe_parquet(&fact_path);
    let order_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let qty_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));
    let mut order_keys = vec![1];
    order_keys.extend(100..228);
    let order_batch =
        RecordBatch::try_new(Arc::clone(&order_schema), vec![Arc::new(Int32Array::from(order_keys))]).unwrap();
    let qty_batch =
        RecordBatch::try_new(Arc::clone(&qty_schema), vec![Arc::new(Int32Array::from(vec![10, 11, 30, 31]))]).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&fact_path)
                .with_file_size_bytes(fs::metadata(&fact_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&fact_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let order_keys =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("order_keys", Arc::clone(&order_schema))));
    let order_join = builder.add_join(
        fact,
        order_keys,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let qty_keys =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("qty_keys", Arc::clone(&qty_schema))));
    let qty_join = builder.add_join(
        order_join,
        qty_keys,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "qty", "qty"), ColumnRef::input(1, "qty", "qty"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(qty_join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 2);
    let context = backend_context_with_grant(1024 * 1024);

    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "order_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &order_batch).unwrap()),
            Arc::clone(&order_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "qty_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &qty_batch).unwrap()),
            Arc::clone(&qty_schema),
        )
        .unwrap(),
    );
    {
        let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
        let output = result.output().as_host_arrow().unwrap();
        let metrics = result.metrics();
        assert_eq!(output.num_rows(), 2);
        assert_eq!(int32_array(output, 0).values().as_ref(), &[1, 1]);
        assert_eq!(int32_array(output, 1).values().as_ref(), &[10, 11]);
        assert_eq!(metrics.dynamic_filter_produced_count, 2);
        assert_eq!(metrics.dynamic_filter_applied_count, 2);
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "reader_predicate_rows"
                && row.filter_kind == "small_in_list"
                && row.payload_value_count == Some(4)
        }));
        let deferred_rows = metrics
            .dynamic_filter_runtime_rows
            .iter()
            .filter(|row| {
                row.event == "applied_post_decode"
                    && row.consumer_kind == "local_parquet_source"
                    && row.filter_kind == "small_in_list"
                    && row.payload_value_count == Some(129)
            })
            .collect::<Vec<_>>();
        assert!(!deferred_rows.is_empty());
        // The deferred order keyset {1, 100..=227} coalesces to runs [1,1] and
        // [100,227]; the probe's [2,2] and [3,3] row groups are both in the gap and
        // are pruned set-aware, leaving only the [1,1] row group. The deferred
        // post-decode filter therefore sees 2 rows that already match and filters
        // none; the deferral still fires (asserted above), but row-group pruning now
        // removes the non-matching rows upstream instead of post-decode.
        assert_eq!(deferred_rows.iter().filter_map(|row| row.post_decode_rows_before).sum::<u64>(), 2);
        assert_eq!(deferred_rows.iter().filter_map(|row| row.post_decode_rows_after).sum::<u64>(), 2);
        assert_eq!(deferred_rows.iter().map(|row| row.post_decode_rows_filtered).sum::<u64>(), 0);
        assert_eq!(metrics.dynamic_filter_source_post_decode_rows_filtered, 0);
    }
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_produces_range_only_dynamic_filter_for_large_local_parquet_keyset() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-large-source-range-filter");
    let left_path = dir.join("lineitem.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let build_keys = (0..=65_536).collect::<Vec<i32>>();
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(build_keys))]).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_projected_row_group_bytes(
                    vec![128 * 1024 * 1024, 128 * 1024 * 1024, 128 * 1024 * 1024],
                    vec![128 * 1024 * 1024, 128 * 1024 * 1024, 128 * 1024 * 1024],
                )
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right_keys", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(8 * 1024 * 1024);

    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "right_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );
    {
        let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
        let output = result.output().as_host_arrow().unwrap();
        let metrics = result.metrics();
        assert_eq!(output.num_rows(), 6);
        assert_eq!(metrics.dynamic_filter_produced_count, 1);
        assert_eq!(metrics.dynamic_filter_applied_count, 1);
        assert_eq!(metrics.dynamic_filter_skipped_count, 0);
        assert_eq!(metrics.dynamic_filter_source_bytes_pruned, 0);
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "produced"
                && row.filter_kind == "range"
                && row.payload_value_count.is_none()
                && row.payload_min_i64 == Some(0)
                && row.payload_max_i64 == Some(65_536)
                && row.producer_rows == Some(65_537)
        }));
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "applied"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "range"
                && row.selected_row_groups_before == Some(3)
                && row.selected_row_groups_after == Some(3)
        }));
        assert!(!metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "reader_predicate_rows"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "range"
        }));
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_filtered, 0);
    }
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_skips_large_dynamic_filter_for_dimension_sized_probe_source() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-large-dimension-filter-skip");
    let left_path = dir.join("lineitem.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let build_keys = (0..=65_536).collect::<Vec<i32>>();
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(build_keys))]).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right_keys", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(8 * 1024 * 1024);

    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "right_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );
    {
        let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
        let output = result.output().as_host_arrow().unwrap();
        let metrics = result.metrics();
        assert_eq!(output.num_rows(), 6);
        assert_eq!(metrics.dynamic_filter_produced_count, 0);
        assert_eq!(metrics.dynamic_filter_applied_count, 0);
        assert_eq!(metrics.dynamic_filter_skipped_count, 1);
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "skipped"
                && row.skipped_reason.as_deref() == Some("large_keyset_dimension_sized_local_parquet_consumer")
        }));
    }
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_splits_large_dynamic_filter_between_dimension_and_fact_sources() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-large-filter-mixed-source-policy");
    let small_path = dir.join("small_probe.parquet");
    let large_path = dir.join("large_probe.parquet");
    let small_schema = write_dynamic_filter_probe_parquet(&small_path);
    let large_schema = write_dynamic_filter_probe_parquet(&large_path);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let build_keys = (0..=65_536).collect::<Vec<i32>>();
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(build_keys))]).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let small = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&small_path)
                .with_file_size_bytes(fs::metadata(&small_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&small_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let large = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&large_path)
                .with_file_size_bytes(fs::metadata(&large_path).unwrap().len())
                .with_projected_row_group_bytes(
                    vec![128 * 1024 * 1024, 128 * 1024 * 1024, 128 * 1024 * 1024],
                    vec![128 * 1024 * 1024, 128 * 1024 * 1024, 128 * 1024 * 1024],
                )
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&large_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let union = builder.add_union(vec![small, large], Arc::clone(&output_schema));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right_keys", Arc::clone(&right_schema))));
    let join = builder.add_join(
        union,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 2);
    let context = backend_context_with_grant(8 * 1024 * 1024);

    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "right_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );
    {
        let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
        let output = result.output().as_host_arrow().unwrap();
        let metrics = result.metrics();
        assert_eq!(output.num_rows(), 12);
        assert_eq!(metrics.dynamic_filter_produced_count, 1);
        assert_eq!(metrics.dynamic_filter_applied_count, 1);
        assert_eq!(metrics.dynamic_filter_skipped_count, 1);
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "skipped" && row.skipped_reason.as_deref() == Some(LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON)
        }));
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "produced"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "range"
                && row.payload_value_count.is_none()
                && row.payload_min_i64 == Some(0)
                && row.payload_max_i64 == Some(65_536)
        }));
    }
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_does_not_duplicate_small_in_list_reader_predicate_after_probe_source_decode() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-source-small-in-list-reader-predicate");
    let left_path = dir.join("lineitem.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(vec![1, 3]))]).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right_keys", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(1024 * 1024);

    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "right_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );
    {
        let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
        let output = result.output().as_host_arrow().unwrap();
        let metrics = result.metrics();
        assert_eq!(output.num_rows(), 4);
        assert_eq!(int32_array(output, 0).values().as_ref(), &[1, 1, 3, 3]);
        assert_eq!(int32_array(output, 1).values().as_ref(), &[10, 11, 30, 31]);
        assert_eq!(metrics.dynamic_filter_produced_count, 1);
        assert_eq!(metrics.dynamic_filter_applied_count, 1);
        // The build keyset {1, 3} coalesces to runs [1,1] and [3,3]; the probe's
        // [2,2] row group sits in the gap and is pruned set-aware (a single [1,3]
        // range would have kept it). Only the [1,1] and [3,3] row groups reach the
        // reader predicate, so it evaluates 4 rows that all match and filters none;
        // the row-group prune subsumes the per-row filtering the range path deferred.
        assert_eq!(metrics.dynamic_filter_row_groups_pruned, 1);
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_before, 4);
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_after, 4);
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_filtered, 0);
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "produced" && row.filter_kind == "small_in_list" && row.payload_value_count == Some(2)
        }));
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "applied"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "small_in_list"
                && row.selected_row_groups_before == Some(3)
                && row.selected_row_groups_after == Some(2)
        }));
        assert!(!metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "applied_post_decode"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "small_in_list"
        }));
        let reader_rows = metrics
            .dynamic_filter_runtime_rows
            .iter()
            .filter(|row| {
                row.event == "reader_predicate_rows"
                    && row.consumer_kind == "local_parquet_source"
                    && row.filter_kind == "small_in_list"
                    && row.payload_value_count == Some(2)
            })
            .collect::<Vec<_>>();
        assert!(!reader_rows.is_empty());
        assert_eq!(reader_rows.iter().filter_map(|row| row.reader_predicate_rows_before).sum::<u64>(), 4);
        assert_eq!(reader_rows.iter().filter_map(|row| row.reader_predicate_rows_after).sum::<u64>(), 4);
        assert_eq!(reader_rows.iter().map(|row| row.reader_predicate_rows_filtered).sum::<u64>(), 0);
    }
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_does_not_duplicate_tpcds_sized_small_in_list_reader_predicate_after_probe_source_decode() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-tpcds-small-in-list-reader-predicate");
    let left_path = dir.join("lineitem.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let mut build_keys = vec![1, 3];
    build_keys.extend(4..=401);
    assert_eq!(build_keys.len(), 400);
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(build_keys))]).unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right_keys", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(1024 * 1024);

    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "right_keys".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );
    {
        let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
        let output = result.output().as_host_arrow().unwrap();
        let metrics = result.metrics();
        assert_eq!(output.num_rows(), 4);
        assert_eq!(int32_array(output, 0).values().as_ref(), &[1, 1, 3, 3]);
        assert_eq!(int32_array(output, 1).values().as_ref(), &[10, 11, 30, 31]);
        assert_eq!(metrics.dynamic_filter_produced_count, 1);
        assert_eq!(metrics.dynamic_filter_applied_count, 1);
        // Build keyset {1, 3, 4..=401} coalesces to runs [1,1] and [3,401]; the probe's
        // [2,2] row group is in the gap and is pruned set-aware. The reader predicate
        // then sees only the 4 surviving rows (all matching) and filters none; pruning
        // subsumes the per-row reader filtering the single-range path performed.
        assert_eq!(metrics.dynamic_filter_row_groups_pruned, 1);
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_before, 4);
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_after, 4);
        assert_eq!(metrics.dynamic_filter_reader_predicate_rows_filtered, 0);
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "produced"
                && row.filter_kind == "small_in_list"
                && row.payload_value_count == Some(400)
                && row.producer_rows == Some(400)
        }));
        assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "applied"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "small_in_list"
                && row.payload_value_count == Some(400)
                && row.selected_row_groups_before == Some(3)
                && row.selected_row_groups_after == Some(2)
        }));
        assert!(!metrics.dynamic_filter_runtime_rows.iter().any(|row| {
            row.event == "applied_post_decode"
                && row.consumer_kind == "local_parquet_source"
                && row.filter_kind == "small_in_list"
                && row.payload_value_count == Some(400)
        }));
        let reader_rows = metrics
            .dynamic_filter_runtime_rows
            .iter()
            .filter(|row| {
                row.event == "reader_predicate_rows"
                    && row.consumer_kind == "local_parquet_source"
                    && row.filter_kind == "small_in_list"
                    && row.payload_value_count == Some(400)
            })
            .collect::<Vec<_>>();
        assert!(!reader_rows.is_empty());
        assert_eq!(reader_rows.iter().filter_map(|row| row.reader_predicate_rows_before).sum::<u64>(), 4);
        assert_eq!(reader_rows.iter().filter_map(|row| row.reader_predicate_rows_after).sum::<u64>(), 4);
        assert_eq!(reader_rows.iter().map(|row| row.reader_predicate_rows_filtered).sum::<u64>(), 0);
    }
    drop(gpu_inputs);
    drop(runtime);
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_handles_dynamic_filter_pruned_empty_probe_source() {
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-empty-dynamic-filter");
    let left_path = dir.join("lineitem.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    let right_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimized = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &optimized.plan).unwrap();
    let output = result.output().as_host_arrow().unwrap();
    let metrics = result.metrics();
    assert_eq!(output.num_rows(), 0);
    assert_eq!(output.num_columns(), 2);
    assert_eq!(metrics.dynamic_filter_produced_count, 1);
    assert_eq!(metrics.dynamic_filter_applied_count, 1);
    assert_eq!(metrics.dynamic_filter_row_groups_pruned, 3);
    assert_eq!(metrics.row_groups_committed, 0);
    assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
        row.event == "applied"
            && row.filter_kind == "range"
            && row.selected_row_groups_before == Some(3)
            && row.selected_row_groups_after == Some(0)
    }));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_applies_hash_join_build_dynamic_filter_to_resident_probe() {
    let fixture_runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-resident-probe-dynamic-filter");
    let right_path = dir.join("orders.parquet");
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = write_join_right_keys_parquet(&fixture_runtime, &right_path, &[2]);
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&left_schema))));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    assert!(matches!(optimized.plan.dynamic_filter_routes()[0].consumer, DynamicFilterConsumer::HashJoinProbe { .. }));

    drop(fixture_runtime);
    let context = backend_context_with_grant(1024 * 1024);
    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&left_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
    let output = result.output().as_host_arrow().unwrap();
    let metrics = result.metrics();
    assert_eq!(output.num_rows(), 1);
    assert_eq!(int32_array(output, 0).values().as_ref(), &[2]);
    assert_eq!(int32_array(output, 1).values().as_ref(), &[20]);
    assert_eq!(metrics.dynamic_filter_produced_count, 1);
    assert_eq!(metrics.dynamic_filter_applied_count, 1);
    assert_eq!(metrics.dynamic_filter_join_probe_post_decode_rows_filtered, 3);
    assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
        row.event == "applied_post_decode"
            && row.consumer_kind == "hash_join_probe"
            && row.consumer_lowering == "post_decode_only"
    }));
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_applies_hash_join_build_dynamic_filter_at_resident_probe_filter() {
    let fixture_runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-resident-probe-filter-dynamic-filter");
    let right_path = dir.join("orders.parquet");
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = write_join_right_keys_parquet(&fixture_runtime, &right_path, &[2]);
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&left_schema))));
    let filtered_left = builder.add_filter(
        left,
        Expr::binary(Expr::column("qty"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&left_schema),
    );
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    )));
    let join = builder.add_join(
        filtered_left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    assert!(matches!(optimized.plan.dynamic_filter_routes()[0].consumer, DynamicFilterConsumer::Filter { .. }));

    drop(fixture_runtime);
    let context = backend_context_with_grant(1024 * 1024);
    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&left_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
    let output = result.output().as_host_arrow().unwrap();
    let metrics = result.metrics();
    assert_eq!(output.num_rows(), 1);
    assert_eq!(int32_array(output, 0).values().as_ref(), &[2]);
    assert_eq!(int32_array(output, 1).values().as_ref(), &[20]);
    assert_eq!(metrics.dynamic_filter_produced_count, 1);
    assert_eq!(metrics.dynamic_filter_applied_count, 1);
    assert_eq!(metrics.dynamic_filter_filter_post_decode_rows_filtered, 3);
    assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
        row.event == "applied_post_decode"
            && row.consumer_kind == "filter"
            && row.consumer_lowering == "post_decode_only"
            && row.post_decode_rows_before == Some(4)
            && row.post_decode_rows_after == Some(1)
    }));
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_applies_filter_dynamic_filter_through_projection_alias() {
    let fixture_runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-projected-filter-dynamic-filter");
    let right_path = dir.join("orders.parquet");
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("raw_order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let right_schema = write_join_right_keys_parquet(&fixture_runtime, &right_path, &[2]);
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    let output_schema = Arc::clone(&projected_schema);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&left_schema))));
    let filtered_left = builder.add_filter(
        left,
        Expr::binary(Expr::column("qty"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&left_schema),
    );
    let projected_left = builder.add_projection(
        filtered_left,
        vec![
            ProjectionOutput::new(Expr::column("raw_order_id"), Field::new("order_id", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("qty"), Field::new("qty", DataType::Int32, false)),
        ],
    );
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    )));
    let join = builder.add_join(
        projected_left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    assert_eq!(optimized.plan.dynamic_filter_routes()[0].consumer_columns, ["raw_order_id".to_string()]);
    assert!(matches!(optimized.plan.dynamic_filter_routes()[0].consumer, DynamicFilterConsumer::Filter { .. }));

    drop(fixture_runtime);
    let context = backend_context_with_grant(1024 * 1024);
    let admitted = context.admit_plan(&optimized.plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&left_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &optimized.plan, &gpu_inputs).unwrap();
    let output = result.output().as_host_arrow().unwrap();
    let metrics = result.metrics();
    assert_eq!(output.num_rows(), 1);
    assert_eq!(int32_array(output, 0).values().as_ref(), &[2]);
    assert_eq!(int32_array(output, 1).values().as_ref(), &[20]);
    assert_eq!(metrics.dynamic_filter_produced_count, 1);
    assert_eq!(metrics.dynamic_filter_applied_count, 1);
    assert!(metrics.dynamic_filter_runtime_rows.iter().any(|row| {
        row.event == "applied_post_decode"
            && row.consumer_kind == "filter"
            && row.consumer_lowering == "post_decode_only"
            && row.consumer_node_id != "none"
            && row.post_decode_rows_before == Some(4)
            && row.post_decode_rows_after == Some(1)
    }));
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_aggregate_keyset_dynamic_filter_join_input() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-aggregate-dynamic-filter");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_dynamic_filter_probe_parquet(&left_path);
    let right_schema = write_join_right_keys_parquet(&runtime, &right_path, &[2, 2]);
    let aggregate_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    )));
    let keyset = builder.add_aggregate(
        right,
        AggregateSpec::new(vec![Expr::column("order_id")], Vec::new()),
        Arc::clone(&aggregate_schema),
    );
    let join = builder.add_join(
        left,
        keyset,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty")])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    let context = backend_context_with_grant(1024 * 1024);

    let decision = admit(&optimized.plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context()).unwrap();
    assert!(bounded_join_unsupported_shape_facts(&optimized.plan, &decision, rapids_analysis_context()).is_none());
    let result = execute_query_plan(&context, &optimized.plan).expect("aggregate keyset join input should execute");
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut rows = (0..batch.num_rows()).map(|row| int32_array(batch, 0).value(row)).collect::<Vec<_>>();
    rows.sort_unstable();
    // The build keyset is {2}, so the inner join keeps both order_id = 2 probe
    // rows; a dynamic filter may prune row groups but never join results.
    assert_eq!(rows, vec![20, 21]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_build_dynamic_filter_below_aggregate_probe() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-hash-join-aggregate-probe-dynamic-filter");
    let fact_path = dir.join("store_returns.parquet");
    let dim_path = dir.join("store.parquet");
    let fact_schema = write_dynamic_filter_probe_parquet(&fact_path);
    let dim_schema = write_join_right_keys_parquet(&runtime, &dim_path, &[2]);
    let aggregate_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&dim_path).with_file_size_bytes(fs::metadata(&dim_path).unwrap().len())],
        Arc::clone(&dim_schema),
        vec!["order_id".to_string()],
    )));
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&fact_path)
                .with_file_size_bytes(fs::metadata(&fact_path).unwrap().len())
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&fact_schema),
        vec!["order_id".to_string()],
    )));
    let grouped_fact = builder.add_aggregate(
        fact,
        AggregateSpec::new(vec![Expr::column("order_id")], Vec::new()),
        Arc::clone(&aggregate_schema),
    );
    let join = builder.add_join(
        dim,
        grouped_fact,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![JoinOutputColumn::right(
                ColumnRef::input(1, "order_id", "order_id"),
                "order_id",
            )])
            .with_build_side(JoinSide::Left),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let optimized =
        optimize_query_plan(plan, &optimizer_options, rapids_analysis_context(), local_dimension_pruning_resolver());
    assert_eq!(optimized.diagnostics.dynamic_filter_route_discovery.route_count, 1);
    assert!(matches!(
        optimized.plan.dynamic_filter_routes()[0].consumer,
        DynamicFilterConsumer::LocalParquetSource { .. }
    ));
    let context = backend_context_with_grant(1024 * 1024);

    let decision = admit(&optimized.plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context()).unwrap();
    assert!(bounded_join_unsupported_shape_facts(&optimized.plan, &decision, rapids_analysis_context()).is_none());
    let result = execute_query_plan(&context, &optimized.plan).expect("aggregate probe input should execute");
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut rows = (0..batch.num_rows()).map(|row| int32_array(batch, 0).value(row)).collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, vec![2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_hash_join_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert!(metrics.reservation_requested_bytes > 0);
    assert!(metrics.reservation_granted_bytes > 0);
    assert!(metrics.reservation_live_peak_bytes > 0);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert!(metrics.hash_join_hash_table_bytes > 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.build_side, "right");
    assert_eq!(join_metrics.probe_side, "left");
    assert!(join_metrics.gather_map_bytes > 0);
    assert_eq!(join_metrics.output_rows, 3);
    assert_granted_reservation_trace(
        metrics,
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_input_projection"),
        "schema-only join input selection must alias or transfer the resident input instead of charging a projection"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[20, 25, 30]);
    assert_eq!(string_array(batch, 2).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_uses_single_pass_reusable_probe_for_unique_inner_build() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-single-pass-unique-build");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right)
            .with_build_artifact_id("orders_by_order_id"),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert!(result.metrics().operators.iter().any(|row| {
        row.join.as_ref().is_some_and(|join| join.execution_strategy == "bounded_reusable_hash_join_single_pass_probe")
    }));

    fs::remove_dir_all(dir).unwrap();
}

fn join_source_with_projected_row_group_estimates(
    path: &Path,
    schema: &Arc<Schema>,
    columns: &[&str],
    row_group_count: usize,
    bytes_per_row_group: u64,
) -> LocalParquetSource {
    let row_group_bytes = vec![bytes_per_row_group; row_group_count];
    LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(path)
                .with_file_size_bytes(fs::metadata(path).unwrap().len())
                .with_row_group_bytes(row_group_bytes.clone())
                .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
                .with_row_group_rows(vec![1; row_group_count]),
        ],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

fn write_dynamic_filter_probe_parquet(path: &Path) -> Arc<Schema> {
    write_dynamic_filter_probe_rows_parquet(path, &[1, 1, 2, 2, 3, 3], &[10, 11, 20, 21, 30, 31])
}

fn write_dynamic_filter_probe_rows_parquet(path: &Path, order_ids: &[i32], quantities: &[i32]) -> Arc<Schema> {
    assert_eq!(order_ids.len(), quantities.len());
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(order_ids.to_vec())), Arc::new(Int32Array::from(quantities.to_vec()))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 2);
    schema
}

fn write_q72_sales_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("cs_sold_date_sk", DataType::Int32, false),
        Field::new("cs_item_sk", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 1])), Arc::new(Int32Array::from(vec![10, 20]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 2);
    schema
}

fn write_q72_d1_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("d_date_sk", DataType::Int32, false),
        Field::new("d_week_seq", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![1999])),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 1);
    schema
}

fn write_q72_inventory_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("inv_item_sk", DataType::Int32, false),
        Field::new("inv_date_sk", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![10, 10])), Arc::new(Int32Array::from(vec![5, 6]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 1);
    schema
}

fn write_q72_d2_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("d_date_sk", DataType::Int32, false),
        Field::new("d_week_seq", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![5, 6])), Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 1);
    schema
}

fn write_row_grouped_join_left_with_payload_parquet(path: &Path) -> Arc<Schema> {
    let row_groups = 4;
    let rows_per_row_group = 256;
    let total_rows = row_groups * rows_per_row_group;
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let mut order_ids = Vec::with_capacity(total_rows);
    let mut quantities = Vec::with_capacity(total_rows);
    for row_group in 0..row_groups {
        let order_id = i32::try_from(row_group + 1).unwrap();
        order_ids.extend(std::iter::repeat_n(order_id, rows_per_row_group));
        quantities.extend((0..rows_per_row_group).map(|row| order_id * 1000 + i32::try_from(row).unwrap()));
    }
    let payload = "x".repeat(8192);
    let payloads = vec![payload.as_str(); total_rows];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(order_ids)),
            Arc::new(Int32Array::from(quantities)),
            Arc::new(StringArray::from(payloads)),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, rows_per_row_group);
    schema
}

fn write_category_bucket_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("bucket", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["category_2", "category_4"])), Arc::new(Int32Array::from(vec![7, 9]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_source_adjacent_join_filter_left_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("partkey", DataType::Int32, false),
        Field::new("brand", DataType::Utf8, false),
        Field::new("container", DataType::Utf8, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("size", DataType::Int32, false),
        Field::new("shipmode", DataType::Utf8, false),
        Field::new("shipinstruct", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(StringArray::from(vec![
                "Brand#12", "Brand#23", "Brand#34", "Brand#12", "Brand#12", "Brand#23", "Brand#34", "Brand#34",
            ])),
            Arc::new(StringArray::from(vec![
                "SM CASE", "MED BOX", "LG PACK", "SM CASE", "WRAP", "MED BOX", "LG PACK", "LG PACK",
            ])),
            Arc::new(Float64Array::from(vec![5.0, 15.0, 25.0, 12.0, 5.0, 9.0, 25.0, 25.0])),
            Arc::new(Int32Array::from(vec![3, 8, 12, 3, 3, 8, 12, 12])),
            Arc::new(StringArray::from(vec!["AIR", "AIR REG", "AIR", "AIR", "AIR", "AIR", "SHIP", "AIR"])),
            Arc::new(StringArray::from(vec![
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "TAKE BACK RETURN",
            ])),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 2);
    schema
}

fn write_source_adjacent_join_filter_right_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("partkey", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(StringArray::from(vec![
                "brand12",
                "brand23",
                "brand34",
                "qty_high",
                "container",
                "qty_low",
                "mode",
                "instruction",
            ])),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 8);
    schema
}

fn source_adjacent_join_filter_predicate() -> Expr {
    let branch_1 = q19_brand_branch("Brand#12", &["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1.0, 11.0, 5);
    let branch_2 = q19_brand_branch("Brand#23", &["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10.0, 20.0, 10);
    let branch_3 = q19_brand_branch("Brand#34", &["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20.0, 30.0, 15);
    Expr::binary(Expr::binary(branch_1, BinaryOp::Or, branch_2), BinaryOp::Or, branch_3)
}

fn q19_brand_branch(brand: &str, containers: &[&str], quantity_min: f64, quantity_max: f64, size_max: i32) -> Expr {
    and_all(vec![
        Expr::binary(Expr::column("brand"), BinaryOp::Eq, utf8_lit(brand)),
        Expr::in_list(Expr::column("container"), containers.iter().map(|container| utf8_lit(container)).collect()),
        Expr::binary(Expr::column("quantity"), BinaryOp::GtEq, Expr::literal(LiteralValue::Float64(quantity_min))),
        Expr::binary(Expr::column("quantity"), BinaryOp::LtEq, Expr::literal(LiteralValue::Float64(quantity_max))),
        Expr::binary(Expr::column("size"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(1))),
        Expr::binary(Expr::column("size"), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(size_max))),
        Expr::in_list(Expr::column("shipmode"), vec![utf8_lit("AIR"), utf8_lit("AIR REG")]),
        Expr::binary(Expr::column("shipinstruct"), BinaryOp::Eq, utf8_lit("DELIVER IN PERSON")),
    ])
}

fn and_all(mut predicates: Vec<Expr>) -> Expr {
    let first = predicates.remove(0);
    predicates.into_iter().fold(first, |left, right| Expr::binary(left, BinaryOp::And, right))
}

fn utf8_lit(value: &str) -> Expr {
    Expr::literal(LiteralValue::Utf8(value.to_string()))
}

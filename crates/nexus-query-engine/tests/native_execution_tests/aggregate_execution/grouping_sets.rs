use super::*;

#[test]
fn bounded_compiled_join_rollup_avg_final_merge_concatenates_typed_null_partials() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-compiled-rollup-avg-typed-null-partials");
    let inventory_path = dir.join("inventory.parquet");
    let item_path = dir.join("item.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let inventory_schema = write_inventory_rollup_parquet(&inventory_path);
    let item_schema = write_item_rollup_parquet(&item_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
        Field::new("quantity_on_hand", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("avg_quantity_on_hand", DataType::Float64, true),
    ]));

    let inventory_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&inventory_path).with_file_size_bytes(fs::metadata(&inventory_path).unwrap().len())],
        inventory_schema,
        vec!["item_sk".to_string(), "quantity_on_hand".to_string()],
    )
    .with_column_roles(vec!["item_sk".to_string()], vec!["quantity_on_hand".to_string()]);
    let mut item_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&item_path).with_file_size_bytes(fs::metadata(&item_path).unwrap().len())],
        item_schema,
        vec![
            "item_sk".to_string(),
            "product_name".to_string(),
            "brand".to_string(),
            "class".to_string(),
            "category".to_string(),
        ],
    );
    item_source.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let inventory = builder.add_source(SourceNode::LocalParquet(inventory_source));
    let item = builder.add_source(SourceNode::LocalParquet(item_source));
    let join = builder.add_join(
        inventory,
        item,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::ColumnRef::input(0, "item_sk", "inventory_item_sk"),
                nexus_query_engine::ColumnRef::input(1, "item_sk", "item_item_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(
                nexus_query_engine::ColumnRef::input(1, "product_name", "product_name"),
                "product_name",
            ),
            JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "brand", "brand"), "brand"),
            JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "class", "class"), "class"),
            JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            JoinOutputColumn::left(
                nexus_query_engine::ColumnRef::input(0, "quantity_on_hand", "quantity_on_hand"),
                "quantity_on_hand",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("product_name"), Expr::column("brand"), Expr::column("class"), Expr::column("category")],
            vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("quantity_on_hand"), "avg_quantity_on_hand")],
        )
        .with_grouping_sets(GroupingSetsSpec::new(
            vec![
                vec![false, false, false, false],
                vec![false, false, false, true],
                vec![false, false, true, true],
                vec![false, true, true, true],
                vec![true, true, true, true],
            ],
            "__grouping_id",
        )),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("__grouping_id"), false, false),
            SortExpr::new(Expr::column("product_name"), false, true),
            SortExpr::new(Expr::column("brand"), false, true),
            SortExpr::new(Expr::column("class"), false, true),
            SortExpr::new(Expr::column("category"), false, true),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(
        metrics
            .operators
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .any(|join| join.execution_strategy == "bounded_partitioned_hash_equi_join"),
        "q22-style ROLLUP AVG regression should exercise the partitioned join stream path"
    );
    assert!(metrics.source_chunks_committed >= 2);
    assert!(
        metrics.partial_aggregate_count >= 2,
        "join-fed ROLLUP AVG should aggregate more than one streamed partial"
    );
    assert!(metrics.final_merge_attempts >= 1);
    assert!(
        metrics
            .reservation_events
            .iter()
            .any(|event| event.outcome.as_str() == "granted"
                && event.memory_boundary == "compiled_aggregate_final_merge"),
        "ROLLUP AVG over a joined stream should merge partials at the compiled aggregate final-merge boundary"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 13);
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Utf8);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::Utf8);
    assert_eq!(batch.schema().field(2).data_type(), &DataType::Utf8);
    assert_eq!(batch.schema().field(3).data_type(), &DataType::Utf8);
    assert_eq!(batch.schema().field(4).data_type(), &DataType::UInt32);
    assert_eq!(batch.schema().field(5).data_type(), &DataType::Float64);
    assert!(string_array(batch, 0).null_count() > 0);
    assert!(string_array(batch, 1).null_count() > 0);
    assert!(string_array(batch, 2).null_count() > 0);
    assert!(string_array(batch, 3).null_count() > 0);
    assert_rollup_avg_rows_eq(
        rollup_avg_rows(batch),
        vec![
            (
                Some("p1".to_string()),
                Some("BrandA".to_string()),
                Some("Class1".to_string()),
                Some("CatA".to_string()),
                0,
                12.0,
            ),
            (
                Some("p2".to_string()),
                Some("BrandA".to_string()),
                Some("Class1".to_string()),
                Some("CatA".to_string()),
                0,
                6.0,
            ),
            (
                Some("p3".to_string()),
                Some("BrandB".to_string()),
                Some("Class2".to_string()),
                Some("CatB".to_string()),
                0,
                24.0,
            ),
            (Some("p1".to_string()), Some("BrandA".to_string()), Some("Class1".to_string()), None, 1, 12.0),
            (Some("p2".to_string()), Some("BrandA".to_string()), Some("Class1".to_string()), None, 1, 6.0),
            (Some("p3".to_string()), Some("BrandB".to_string()), Some("Class2".to_string()), None, 1, 24.0),
            (Some("p1".to_string()), Some("BrandA".to_string()), None, None, 3, 12.0),
            (Some("p2".to_string()), Some("BrandA".to_string()), None, None, 3, 6.0),
            (Some("p3".to_string()), Some("BrandB".to_string()), None, None, 3, 24.0),
            (Some("p1".to_string()), None, None, None, 7, 12.0),
            (Some("p2".to_string()), None, None, None, 7, 6.0),
            (Some("p3".to_string()), None, None, None, 7, 24.0),
            (None, None, None, None, 15, 114.0 / 7.0),
        ],
        "q22-style ROLLUP AVG rows should preserve typed NULL grouping keys and merged AVG state",
    );

    fs::remove_dir_all(dir).unwrap();
}

fn rollup_avg_rows(
    batch: &RecordBatch,
) -> Vec<(Option<String>, Option<String>, Option<String>, Option<String>, u32, f64)> {
    let product_names = string_array(batch, 0);
    let brands = string_array(batch, 1);
    let classes = string_array(batch, 2);
    let categories = string_array(batch, 3);
    let grouping_ids = batch.column(4).as_any().downcast_ref::<UInt32Array>().unwrap();
    let averages = float64_array(batch, 5);
    (0..batch.num_rows())
        .map(|row| {
            (
                string_value(product_names, row),
                string_value(brands, row),
                string_value(classes, row),
                string_value(categories, row),
                grouping_ids.value(row),
                averages.value(row),
            )
        })
        .collect()
}

fn assert_rollup_avg_rows_eq(
    actual: Vec<(Option<String>, Option<String>, Option<String>, Option<String>, u32, f64)>,
    expected: Vec<(Option<String>, Option<String>, Option<String>, Option<String>, u32, f64)>,
    message: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{message}: row count mismatch");
    for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            (&actual.0, &actual.1, &actual.2, &actual.3, actual.4),
            (&expected.0, &expected.1, &expected.2, &expected.3, expected.4),
            "{message}: row {row} grouping columns mismatch"
        );
        assert!(
            (actual.5 - expected.5).abs() <= 1e-9,
            "{message}: row {row} avg mismatch, actual={}, expected={}",
            actual.5,
            expected.5
        );
    }
}

fn string_value(array: &StringArray, row: usize) -> Option<String> {
    (!array.is_null(row)).then(|| array.value(row).to_string())
}

fn write_inventory_rollup_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("quantity_on_hand", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 3, 3, 3])),
            Arc::new(Int32Array::from(vec![10, 12, 14, 6, 20, 24, 28])),
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, Arc::clone(&schema), batch);
    schema
}

fn write_item_rollup_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int32, false),
        Field::new("product_name", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("category", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("p1"), Some("p2"), Some("p3")])),
            Arc::new(StringArray::from(vec![Some("BrandA"), Some("BrandA"), Some("BrandB")])),
            Arc::new(StringArray::from(vec![Some("Class1"), Some("Class1"), Some("Class2")])),
            Arc::new(StringArray::from(vec![Some("CatA"), Some("CatA"), Some("CatB")])),
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, Arc::clone(&schema), batch);
    schema
}

fn write_single_batch_parquet(path: &Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

use std::{fs::File, path::Path, sync::Arc};

use arrow::record_batch::RecordBatchOptions;
use query_planning::expr::ColumnRef;

use super::*;

type RollupAvgRow = (Option<String>, Option<String>, Option<String>, Option<String>, u32, f64);

#[test]
fn bounded_native_query_emits_rollup_grand_total_row_for_empty_literal_input() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let empty_batch = RecordBatch::try_new_with_options(
        Arc::clone(&input_schema),
        vec![Arc::new(Int32Array::from(Vec::<i32>::new())) as _, Arc::new(Int64Array::from(Vec::<i64>::new())) as _],
        &RecordBatchOptions::new().with_row_count(Some(0)),
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("s", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&input_schema), vec![empty_batch])));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "s")],
            )
            .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1, "empty-input ROLLUP must keep the grand-total row");
    assert!(int32_array(batch, 0).is_null(0));
    assert!(int64_array(batch, 2).is_null(0));
}

#[test]
fn bounded_native_query_emits_rollup_grand_total_row_for_runtime_filtered_empty_input() {
    let dir = unique_temp_dir("query-engine-bounded-native-filtered-empty-rollup");
    let path = dir.join("bench.parquet");
    let context = backend_context();
    let input_schema = write_filtered_empty_rollup_input_parquet(&path);

    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("s", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("key"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(100_000)));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filtered = builder.add_filter(source, filter).unwrap();
    let aggregate = builder
        .add_aggregate(
            filtered,
            AggregateSpec::new(
                vec![Expr::column("key")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("payload"), "s")],
            )
            .with_grouping_sets(GroupingSetsSpec::new(vec![vec![false], vec![true]], "__grouping_id")),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics().clone();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_eq!(
        batch.num_rows(),
        1,
        "runtime-filtered empty ROLLUP must keep the grand-total row; metrics partials={} final_merges={} source_chunks_committed={} input_rows={}",
        metrics.partial_aggregate_count,
        metrics.final_merge_attempts,
        metrics.source_chunks_committed,
        metrics.input_rows
    );
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Int32);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::UInt32);
    assert_eq!(batch.schema().field(2).data_type(), &DataType::Int64);
    assert!(int32_array(batch, 0).is_null(0), "grand-total key must be NULL");
    let grouping_ids = batch.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
    assert_eq!(grouping_ids.value(0), 1, "grand-total grouping id must mark null key");
    assert!(int64_array(batch, 2).is_null(0), "grand-total SUM over empty input must be NULL");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_compiled_join_rollup_avg_final_merge_concatenates_typed_null_partials() {
    let dir = unique_temp_dir("query-engine-bounded-compiled-rollup-avg-typed-null-partials");
    let inventory_path = dir.join("inventory.parquet");
    let item_path = dir.join("item.parquet");
    let context = backend_context();
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
    let join = builder
        .add_join(
            inventory,
            item,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "item_sk", "inventory_item_sk"),
                    ColumnRef::input(1, "item_sk", "item_item_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::right(ColumnRef::input(1, "product_name", "product_name"), "product_name"),
                JoinOutputColumn::right(ColumnRef::input(1, "brand", "brand"), "brand"),
                JoinOutputColumn::right(ColumnRef::input(1, "class", "class"), "class"),
                JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                JoinOutputColumn::left(ColumnRef::input(0, "quantity_on_hand", "quantity_on_hand"), "quantity_on_hand"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&join_schema),
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![
                    Expr::column("product_name"),
                    Expr::column("brand"),
                    Expr::column("class"),
                    Expr::column("category"),
                ],
                vec![AggregateExpr::new(
                    AggregateFunction::Avg,
                    Expr::column("quantity_on_hand"),
                    "avg_quantity_on_hand",
                )],
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
        )
        .unwrap();
    let sort = builder
        .add_sort(
            aggregate,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("__grouping_id"), false, false),
                SortExpr::new(Expr::column("product_name"), false, true),
                SortExpr::new(Expr::column("brand"), false, true),
                SortExpr::new(Expr::column("class"), false, true),
                SortExpr::new(Expr::column("category"), false, true),
            ]),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

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

fn rollup_avg_rows(batch: &RecordBatch) -> Vec<RollupAvgRow> {
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

fn assert_rollup_avg_rows_eq(actual: Vec<RollupAvgRow>, expected: Vec<RollupAvgRow>, message: &str) {
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

fn write_filtered_empty_rollup_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((0..32).map(|i| i % 4).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..32).map(|i| i as i64).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, Arc::clone(&schema), batch);
    schema
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

use super::*;

#[path = "parquet_fixtures/aggregate_inputs.rs"]
mod aggregate_inputs;

pub(super) use aggregate_inputs::*;

pub(super) fn write_input_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 3])),
            Arc::new(Int64Array::from(vec![5, 10, 20, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_join_left_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 2, 3])), Arc::new(Int32Array::from(vec![10, 20, 25, 30]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_supplier_cost_partsupp_parquet(
    runtime: &cudf::Runtime,
    path: &std::path::Path,
    partkeys: &[i32],
    suppkeys: &[i32],
    qtys: &[i64],
) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
        Field::new("qty", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(partkeys.to_vec())),
            Arc::new(Int32Array::from(suppkeys.to_vec())),
            Arc::new(Int64Array::from(qtys.to_vec())),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_supplier_cost_supplier_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("s_suppkey", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![10, 20, 30]))]).unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_mixed_aggregate_input_fact_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity_id", DataType::Int32, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("measure", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2])),
            Arc::new(Float64Array::from(vec![1.0, 5.0, 2.0, 20.0])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_mixed_aggregate_input_fact_row_grouped_parquet(path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("entity_id", DataType::Int32, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("measure", DataType::Float64, false),
    ]));
    let mut entity_ids = Vec::with_capacity(50_000);
    let mut quantities = Vec::with_capacity(50_000);
    let mut measures = Vec::with_capacity(50_000);
    for _ in 0..12_500 {
        entity_ids.extend([1, 1, 2, 2]);
        quantities.extend([1.0, 5.0, 2.0, 20.0]);
        measures.extend([10.0, 20.0, 30.0, 40.0]);
    }
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(entity_ids)),
            Arc::new(Float64Array::from(quantities)),
            Arc::new(Float64Array::from(measures)),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 25_000);
    schema
}

pub(super) fn write_mixed_aggregate_input_dim_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("entity_id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 99]))]).unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_join_left_decimal_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2, 3])),
            Arc::new(
                Decimal128Array::from_iter_values([12_345, 20_000, 999, 30_000])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_join_right_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![2, 3, 4])), Arc::new(StringArray::from(vec!["two", "three", "four"]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_row_grouped_join_left_parquet(path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 2, 3])), Arc::new(Int32Array::from(vec![10, 20, 25, 30]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 1);
    schema
}

pub(super) fn write_row_grouped_join_right_parquet(path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![2, 3, 4])), Arc::new(StringArray::from(vec!["two", "three", "four"]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 1);
    schema
}

pub(super) fn write_single_i32_row_grouped_parquet(path: &std::path::Path, max_row_group_rows: usize) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))]).unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, max_row_group_rows);
    schema
}

pub(super) fn write_i32_parquet_with_row_group_rows(
    path: &std::path::Path,
    values: &[i32],
    max_row_group_rows: usize,
) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values.to_vec()))]).unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, max_row_group_rows);
    schema
}

pub(super) fn write_record_batch_with_row_group_rows(
    path: &std::path::Path,
    schema: Arc<Schema>,
    batch: RecordBatch,
    max_row_group_rows: usize,
) {
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(max_row_group_rows)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_join_right_keys_parquet(
    runtime: &cudf::Runtime,
    path: &std::path::Path,
    order_ids: &[i32],
) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let categories = order_ids.iter().map(|key| format!("category_{key}")).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(order_ids.to_vec())), Arc::new(StringArray::from(categories))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_customer_order_count_customer_parquet(
    runtime: &cudf::Runtime,
    path: &std::path::Path,
) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))]).unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_customer_order_count_orders_parquet(
    runtime: &cudf::Runtime,
    path: &std::path::Path,
) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![10, 11, 20, 50])), Arc::new(Int32Array::from(vec![1, 1, 2, 5]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_part_distinct_count_part_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int32, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["Brand#11", "Brand#11", "Brand#22", "Brand#11"])),
            Arc::new(StringArray::from(vec!["SMALL BRUSHED", "SMALL BRUSHED", "MEDIUM POLISHED", "SMALL BRUSHED"])),
            Arc::new(Int32Array::from(vec![1, 1, 2, 1])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_part_distinct_count_partsupp_parquet(
    runtime: &cudf::Runtime,
    path: &std::path::Path,
) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![10, 11, 10, 11, 12, 13, 14])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_part_distinct_count_supplier_parquet(
    runtime: &cudf::Runtime,
    path: &std::path::Path,
) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("s_suppkey", DataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![10, 11, 12, 13]))]).unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_join_region_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("region", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![2, 3, 5])), Arc::new(StringArray::from(vec!["east", "west", "unused"]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(super) fn write_tpch_like_types_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(
                Decimal128Array::from_iter_values([12_345, 67_890, 22_222]).with_precision_and_scale(15, 2).unwrap(),
            ),
            Arc::new(Date32Array::from(vec![18_262, 18_293, 18_322])),
            Arc::new(TimestampNanosecondArray::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["O", "F", "O"])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

use super::*;

pub(super) fn write_source_unary_parquet(path: &Path) -> Arc<Schema> {
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(super) fn write_join_parquets(left_path: &Path, right_path: &Path) {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int32Array::from(vec![10, 20, 30, 40]))],
    )
    .unwrap();
    write_parquet(left_path, Arc::clone(&left_schema), &left_batch);

    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![Arc::new(Int32Array::from(vec![2, 4, 5])), Arc::new(Int32Array::from(vec![200, 400, 500]))],
    )
    .unwrap();
    write_parquet(right_path, Arc::clone(&right_schema), &right_batch);
}

pub(super) fn write_aggregate_join_input_parquets(left_path: &Path, right_path: &Path) {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 1, 2, 3])), Arc::new(Int64Array::from(vec![10, 5, 20, 30]))],
    )
    .unwrap();
    write_parquet(left_path, Arc::clone(&left_schema), &left_batch);

    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let right_batch =
        RecordBatch::try_new(Arc::clone(&right_schema), vec![Arc::new(Int32Array::from(vec![1, 2, 4]))]).unwrap();
    write_parquet(right_path, Arc::clone(&right_schema), &right_batch);
}

pub(super) fn write_self_join_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 4])), Arc::new(Int32Array::from(vec![10, 20, 40]))],
    )
    .unwrap();
    write_parquet(path, Arc::clone(&schema), &batch);
}

pub(super) fn write_multi_join_parquets(a_path: &Path, b_path: &Path, c_path: &Path) {
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
    ]));
    let a_batch = RecordBatch::try_new(
        Arc::clone(&a_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Int32Array::from(vec![10, 20, 30]))],
    )
    .unwrap();
    write_parquet(a_path, Arc::clone(&a_schema), &a_batch);

    let b_schema = Arc::new(Schema::new(vec![
        Field::new("b_id", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, false),
    ]));
    let b_batch = RecordBatch::try_new(
        Arc::clone(&b_schema),
        vec![Arc::new(Int32Array::from(vec![2, 3, 4])), Arc::new(Int32Array::from(vec![200, 300, 400]))],
    )
    .unwrap();
    write_parquet(b_path, Arc::clone(&b_schema), &b_batch);

    let c_schema = Arc::new(Schema::new(vec![
        Field::new("c_id", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, false),
    ]));
    let c_batch = RecordBatch::try_new(
        Arc::clone(&c_schema),
        vec![Arc::new(Int32Array::from(vec![3, 2, 5])), Arc::new(Int32Array::from(vec![3000, 2000, 5000]))],
    )
    .unwrap();
    write_parquet(c_path, Arc::clone(&c_schema), &c_batch);
}

pub(super) fn write_residual_stream_join_parquets(a_path: &Path, b_path: &Path, c_path: &Path) {
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("a_id", DataType::Int32, false),
        Field::new("a_value", DataType::Int32, false),
    ]));
    let a_batch_0 = RecordBatch::try_new(
        Arc::clone(&a_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int32Array::from(vec![10, 20]))],
    )
    .unwrap();
    let a_batch_1 = RecordBatch::try_new(
        Arc::clone(&a_schema),
        vec![Arc::new(Int32Array::from(vec![3, 4])), Arc::new(Int32Array::from(vec![30, 40]))],
    )
    .unwrap();
    write_parquet_batches(a_path, Arc::clone(&a_schema), &[a_batch_0, a_batch_1]);

    let b_schema = Arc::new(Schema::new(vec![
        Field::new("b_id", DataType::Int32, false),
        Field::new("b_value", DataType::Int32, false),
    ]));
    let b_batch = RecordBatch::try_new(
        Arc::clone(&b_schema),
        vec![Arc::new(Int32Array::from(vec![2, 4])), Arc::new(Int32Array::from(vec![200, 400]))],
    )
    .unwrap();
    write_parquet(b_path, Arc::clone(&b_schema), &b_batch);

    let c_schema = Arc::new(Schema::new(vec![
        Field::new("c_id", DataType::Int32, false),
        Field::new("c_value", DataType::Int32, false),
    ]));
    let c_batch = RecordBatch::try_new(
        Arc::clone(&c_schema),
        vec![Arc::new(Int32Array::from(vec![2, 4])), Arc::new(Int32Array::from(vec![15, 45]))],
    )
    .unwrap();
    write_parquet(c_path, Arc::clone(&c_schema), &c_batch);
}

pub(super) fn write_single_join_post_root_parquets(customers_path: &Path, orders_path: &Path) {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let customer_batch =
        RecordBatch::try_new(Arc::clone(&customer_schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    write_parquet(customers_path, Arc::clone(&customer_schema), &customer_batch);

    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let orders_batch = RecordBatch::try_new(
        Arc::clone(&orders_schema),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30])), Arc::new(Int32Array::from(vec![1, 1, 2]))],
    )
    .unwrap();
    write_parquet(orders_path, Arc::clone(&orders_schema), &orders_batch);
}

pub(super) fn write_string_group_join_post_root_parquets(customers_path: &Path, orders_path: &Path) {
    let customer_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_name", DataType::Utf8, false),
    ]));
    let customer_batch = RecordBatch::try_new(
        Arc::clone(&customer_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(StringArray::from(vec!["A", "BBBB", "CCC"]))],
    )
    .unwrap();
    write_parquet(customers_path, Arc::clone(&customer_schema), &customer_batch);

    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let orders_batch = RecordBatch::try_new(
        Arc::clone(&orders_schema),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30])), Arc::new(Int32Array::from(vec![1, 1, 2]))],
    )
    .unwrap();
    write_parquet(orders_path, Arc::clone(&orders_schema), &orders_batch);
}

pub(super) fn write_nonlinear_topk_contract_parquets() {
    remove_nonlinear_topk_contract_parquets();
    for label in [
        "aggregate-over-join-topk-a",
        "aggregate-over-join-topk-b",
        "aggregate-over-join-topk-c",
        "aggregate-over-join-topk-d",
        "aggregate-over-join-topk-e",
        "aggregate-over-join-topk-f",
        "resident-aggregate-post-aggregate-topk-a",
        "resident-aggregate-post-aggregate-topk-b",
        "resident-aggregate-post-aggregate-topk-c",
        "resident-aggregate-subquery-filter-a",
        "resident-aggregate-subquery-filter-b",
        "resident-aggregate-subquery-filter-c",
        "resident-aggregate-subquery-filter-d",
    ] {
        write_key_value_parquet(Path::new(&format!("/tmp/{label}.parquet")), &[1, 2, 3], &[10, 20, 30]);
    }
    for label in ["resident-aggregate-post-aggregate-topk-lineitem", "resident-aggregate-subquery-filter-lineitem"] {
        write_key_value_parquet(Path::new(&format!("/tmp/{label}.parquet")), &[1, 2, 2, 3], &[1, 2, 3, 4]);
    }
}

pub(super) fn remove_nonlinear_topk_contract_parquets() {
    for label in [
        "aggregate-over-join-topk-a",
        "aggregate-over-join-topk-b",
        "aggregate-over-join-topk-c",
        "aggregate-over-join-topk-d",
        "aggregate-over-join-topk-e",
        "aggregate-over-join-topk-f",
        "resident-aggregate-post-aggregate-topk-a",
        "resident-aggregate-post-aggregate-topk-b",
        "resident-aggregate-post-aggregate-topk-c",
        "resident-aggregate-post-aggregate-topk-lineitem",
        "resident-aggregate-subquery-filter-a",
        "resident-aggregate-subquery-filter-b",
        "resident-aggregate-subquery-filter-c",
        "resident-aggregate-subquery-filter-d",
        "resident-aggregate-subquery-filter-lineitem",
    ] {
        let _ = fs::remove_file(format!("/tmp/{label}.parquet"));
    }
}

pub(super) fn write_key_value_parquet(path: &Path, keys: &[i32], values: &[i64]) {
    assert_eq!(keys.len(), values.len());
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys.to_vec())), Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap();
    write_parquet(path, schema, &batch);
}

pub(super) fn write_parquet(path: &Path, schema: Arc<Schema>, batch: &RecordBatch) {
    write_parquet_batches(path, schema, std::slice::from_ref(batch));
}

fn write_parquet_batches(path: &Path, schema: Arc<Schema>, batches: &[RecordBatch]) {
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
}

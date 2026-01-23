use super::*;

pub(crate) fn write_discounted_lineitem_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_quantity", DataType::Decimal128(15, 2), false),
        Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
        Field::new("l_discount", DataType::Decimal128(15, 2), false),
        Field::new("l_shipdate", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(
                Decimal128Array::from_iter_values([2_300, 2_500, 1_000, 1_200, 1_200, 1_200, 1_200])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from_iter_values([10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(Decimal128Array::from_iter_values([5, 6, 7, 6, 4, 8, 6]).with_precision_and_scale(15, 2).unwrap()),
            Arc::new(Date32Array::from(vec![
                date32(1994, 1, 15),
                date32(1994, 6, 1),
                date32(1994, 9, 30),
                date32(1995, 1, 1),
                date32(1994, 3, 15),
                date32(1994, 11, 15),
                date32(1993, 12, 31),
            ])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_phase2_scalar_expression_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Date32Array::from(vec![18_262, 18_293, 18_322, 18_336])),
            Arc::new(TimestampNanosecondArray::from(vec![100, 200, 300, 400])),
            Arc::new(StringArray::from(vec![Some("O"), Some("F"), Some("O"), None])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

pub(crate) fn write_single_id_parquet(path: &Path, id: i64) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![id]))]).unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_row_grouped_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 3, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 10, 20, 20, 30, 5])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400, 500, 600])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_uneven_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2, 1, 1, 3])),
            Arc::new(Int64Array::from(vec![10, 10, 10, 10, 10, 10])),
            Arc::new(Int64Array::from(vec![0, 100, 200, 30, 30, 300])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_uneven_global_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("m00", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![0, 100, 200, 30, 30]))]).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(4)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_narrow_global_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("m00", DataType::Int8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int8Array::from(vec![100, 100, 100, 100]))]).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_decimal_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(15, 2), false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Decimal128Array::from_iter_values([100, 200, 300]).with_precision_and_scale(15, 2).unwrap())],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_decimal_scale0_grouped_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(20, 0), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2])),
            Arc::new(Decimal128Array::from_iter_values([120, 340, 500]).with_precision_and_scale(20, 0).unwrap()),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_decimal_ratio_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("numerator", DataType::Decimal128(38, 8), false),
        Field::new("denominator", DataType::Decimal128(38, 8), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(
                Decimal128Array::from_iter_values([100_000_000, 300_000_000]).with_precision_and_scale(38, 8).unwrap(),
            ),
            Arc::new(
                Decimal128Array::from_iter_values([400_000_000, 600_000_000]).with_precision_and_scale(38, 8).unwrap(),
            ),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_market_share_decimal_ratio_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("o_year", DataType::Int32, false),
        Field::new("is_target", DataType::Boolean, false),
        Field::new("volume", DataType::Decimal128(38, 8), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![2020, 2020, 2021, 2021])),
            Arc::new(BooleanArray::from(vec![true, false, true, false])),
            Arc::new(
                Decimal128Array::from_iter_values([100_000_000, 300_000_000, 300_000_000, 300_000_000])
                    .with_precision_and_scale(38, 8)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_fractional_decimal_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(15, 2), false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Decimal128Array::from_iter_values([100, 200, 200]).with_precision_and_scale(15, 2).unwrap())],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_nullable_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3])),
            Arc::new(Int64Array::from(vec![None, None, Some(10), None, Some(30)])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_nullable_global_aggregate_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4])), Arc::new(Int64Array::from(vec![None, None, None, None]))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_final_merge_retry_input_parquet(path: &Path, keys_per_row_group: i32) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let keys = (0..3).flat_map(|_| 0..keys_per_row_group).collect::<Vec<_>>();
    let values = vec![1_i64; keys.len()];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys)), Arc::new(Int64Array::from(values))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(keys_per_row_group as usize)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

pub(crate) fn write_single_row_group_final_sort_input_parquet(path: &Path, rows: i32) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let keys = (0..rows).collect::<Vec<_>>();
    let values = vec![1_i64; rows as usize];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys)), Arc::new(Int64Array::from(values))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows as usize)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

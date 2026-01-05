use std::{fs::File, sync::Arc};

use arrow::{
    array::{
        BinaryArray, Date32Array, Date64Array, Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray,
        TimestampNanosecondArray,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

pub(super) fn write_numeric_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_left_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("l_payload", DataType::Int32, false),
        Field::new("l_m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 20, 30])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_right_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("r_payload", DataType::Int32, false),
        Field::new("r_m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 10, 10])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_duplicate_nation_parquet_file(path: &std::path::Path, names: &[&str]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("n_name", DataType::Utf8, false),
    ]));
    let ids = (0..names.len()).map(|index| i32::try_from(index + 1).unwrap()).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(StringArray::from(names.to_vec()))],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_left_decimal_payload_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("l_payload", DataType::Decimal128(15, 2), false),
        Field::new("l_m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Decimal128Array::from_iter_values([500, 2_000, 3_000]).with_precision_and_scale(15, 2).unwrap()),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_right_decimal_payload_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("r_payload", DataType::Decimal128(15, 2), false),
        Field::new("r_m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Decimal128Array::from_iter_values([1_000, 1_000, 1_000]).with_precision_and_scale(15, 2).unwrap()),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_binary_schema_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
        Field::new("blob", DataType::Binary, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(BinaryArray::from(vec![b"a" as &[u8], b"b" as &[u8], b"c" as &[u8]])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_float_measure_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_date64_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("event_date64", DataType::Date64, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Date64Array::from(vec![1_577_836_800_000, 1_580_428_800_000, 1_583_020_800_000])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_row_group_pruning_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 2, 1, 2, 1, 2])),
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 10, 11, 20, 21])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_nullable_row_group_pruning_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, true),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(2), Some(4), None, Some(5), None])),
            Arc::new(Int64Array::from(vec![10, 20, 40, 400, 50, 500])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_parquet_file_with_tpch_like_columns(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
        Field::new("label", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(15, 2), false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![5, 10, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
            Arc::new(
                Decimal128Array::from_iter_values([12_345, 67_890, 22_222]).with_precision_and_scale(15, 2).unwrap(),
            ),
            Arc::new(Date32Array::from(vec![18_262, 18_293, 18_322])),
            Arc::new(TimestampNanosecondArray::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_q6_lineitem_parquet_file(path: &std::path::Path) {
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
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn write_batch(path: &std::path::Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn date32(year: i32, month: u32, day: u32) -> i32 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 { adjusted_year } else { adjusted_year - 399 } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month = month as i32;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

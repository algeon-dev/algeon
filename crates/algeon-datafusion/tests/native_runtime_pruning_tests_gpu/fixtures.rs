use std::{fs::File, sync::Arc};

use arrow::{
    array::{
        BooleanArray, Date32Array, Decimal128Array, FixedSizeListArray, Int32Array, Int64Array, ListArray, StringArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::parquet::{
    arrow::{ArrowSchemaConverter, ArrowWriter},
    data_type::{Int32Type, Int64Type},
    file::{metadata::SortingColumn, properties::WriterProperties, writer::SerializedFileWriter},
};

pub(super) fn write_sorting_columns_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_sorting_columns(Some(vec![SortingColumn { column_idx: 0, descending: false, nulls_first: false }]))
        .build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Rows ascend by `key` across 16 row groups, and the footer declares that
/// order through `sorting_columns`.
pub(super) fn write_multi_row_group_sorting_columns_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
    ]));
    let keys = (0..4_096).collect::<Vec<i32>>();
    let payloads = keys.iter().map(|key| (key * 7_919) % 1_000).collect::<Vec<i32>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys)), Arc::new(Int32Array::from(payloads))],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(256))
        .set_sorting_columns(Some(vec![SortingColumn { column_idx: 0, descending: false, nulls_first: false }]))
        .build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_filtered_empty_rollup_parquet_file(path: &std::path::Path) {
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
    let props = WriterProperties::builder().build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
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

pub(super) fn write_physically_empty_parquet_file(path: &std::path::Path) {
    let file = File::create(path).unwrap();
    let writer = ArrowWriter::try_new(file, empty_parquet_schema(), Some(WriterProperties::builder().build())).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_zero_row_group_parquet_file(path: &std::path::Path) {
    let schema = empty_parquet_schema();
    let file = File::create(path).unwrap();
    let properties = Arc::new(WriterProperties::builder().build());
    let parquet_schema = ArrowSchemaConverter::new().convert(&schema).unwrap();
    let mut writer =
        SerializedFileWriter::new(file, parquet_schema.root_schema_ptr(), Arc::clone(&properties)).unwrap();
    let mut row_group = writer.next_row_group().unwrap();
    for column_index in 0..4 {
        let mut column = row_group.next_column().unwrap().expect("zero-row group must contain every schema column");
        match column_index {
            0 | 1 => column.typed::<Int32Type>().write_batch(&[], None, None).unwrap(),
            2 | 3 => column.typed::<Int64Type>().write_batch(&[], None, None).unwrap(),
            _ => unreachable!(),
        };
        column.close().unwrap();
    }
    row_group.close().unwrap();
    writer.close().unwrap();
}

fn empty_parquet_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]))
}

pub(super) fn write_reader_pushdown_selective_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let row_count = 1024;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((0..row_count).map(|value| value % 8).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..row_count).map(i64::from).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..row_count).map(i64::from).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..row_count).map(|value| i64::from(value * 2)).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_count as usize)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_decimal32_reader_pushdown_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Decimal128(7, 2), false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let row_count = 1024;
    let payload = (0..row_count).map(i128::from).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((0..row_count).map(|value| value % 8).collect::<Vec<_>>())),
            Arc::new(Decimal128Array::from_iter_values(payload).with_precision_and_scale(7, 2).unwrap()),
            Arc::new(Int64Array::from((0..row_count).map(i64::from).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..row_count).map(|value| i64::from(value * 2)).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(row_count as usize)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_bare_boolean_filter_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("flag", DataType::Boolean, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(BooleanArray::from(vec![true, false, true, false, true, false])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_scalar_expression_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("status", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Date32Array::from(vec![18_262, 18_293, 18_322, 18_353])),
            Arc::new(StringArray::from(vec!["O", "F", "O", "P"])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_discounted_revenue_lineitem_parquet(path: &std::path::Path) {
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

pub(super) fn write_nullable_scalar_expression_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("orderkey", DataType::Int64, true),
        Field::new("shipdate", DataType::Date32, true),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        Field::new("status", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(4), Some(5), None])),
            Arc::new(Date32Array::from(vec![Some(18_262), None, Some(18_322), Some(18_353), None, Some(18_414)])),
            Arc::new(TimestampNanosecondArray::from(vec![
                Some(1_577_840_523_123_456_789),
                None,
                Some(1_577_844_184_987_654_321),
                Some(1_577_934_245_111_222_333),
                None,
                Some(1_578_024_306_000_001_234),
            ])),
            Arc::new(StringArray::from(vec![Some("O"), Some("F"), None, Some("P"), Some("OX"), Some("R")])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_string_length_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(StringArray::from(vec![
                Some("abc"),
                Some("é"),
                Some("中文"),
                Some("🙂"),
                Some("e\u{301}"),
                Some(""),
                None,
                Some("naïve"),
            ])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Rows: 2020-03-04 12:34:56.789123456, the tick before the epoch, NULL,
/// 1970-01-01 00:00:59, 1970-01-01 00:59:59.999 and 2001-09-09 01:46:40.5,
/// each stored in every timestamp unit. `epoch_s` holds integer seconds
/// including negative ones.
pub(super) fn write_timestamp_extract_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int32, false),
        Field::new("ts_s", DataType::Timestamp(TimeUnit::Second, None), true),
        Field::new("ts_ms", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("ts_us", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("ts_ns", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        Field::new("shipdate", DataType::Date32, true),
        Field::new("epoch_s", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5])),
            Arc::new(TimestampSecondArray::from(vec![
                Some(1_583_325_296),
                Some(-1),
                None,
                Some(59),
                Some(3_599),
                Some(1_000_000_000),
            ])),
            Arc::new(TimestampMillisecondArray::from(vec![
                Some(1_583_325_296_789),
                Some(-1),
                None,
                Some(59_000),
                Some(3_599_999),
                Some(1_000_000_000_500),
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1_583_325_296_789_123),
                Some(-1),
                None,
                Some(59_000_000),
                Some(3_599_999_000),
                Some(1_000_000_000_500_000),
            ])),
            Arc::new(TimestampNanosecondArray::from(vec![
                Some(1_583_325_296_789_123_456),
                Some(-1),
                None,
                Some(59_000_000_000),
                Some(3_599_999_000_000),
                Some(1_000_000_000_500_000_000),
            ])),
            Arc::new(Date32Array::from(vec![
                date32(2020, 3, 4),
                date32(1969, 12, 31),
                date32(1970, 1, 1),
                date32(1970, 1, 1),
                date32(1970, 1, 1),
                date32(2001, 9, 9),
            ])),
            Arc::new(Int64Array::from(vec![Some(1_583_325_296), Some(-1), None, Some(59), Some(-61), Some(-3_600)])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_nullable_sort_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Int32, false),
        Field::new("tie_key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![10, 20, 30, 40, 30, 20, 10, 50, 60])),
            Arc::new(Int32Array::from(vec![3, 1, 2, 1, 3, 2, 2, 1, 1])),
            Arc::new(Int64Array::from(vec![
                None,
                Some(20),
                Some(12),
                None,
                Some(13),
                Some(22),
                None,
                Some(31),
                Some(11),
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

pub(super) fn write_nullable_avg_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 10, 10, 10, 10])),
            Arc::new(Int64Array::from(vec![None, None, Some(10), None, Some(30)])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Keys cover a group whose `x` values are all null, a group whose `y` and `d`
/// values are all null, a null key, and an `x` value repeated in a later row
/// group. The `d` averages of keys 1 and 3 and of all rows do not terminate
/// at the AVG output scale, so they check its truncation toward zero.
pub(super) fn write_nullable_single_distinct_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("x", DataType::Int64, true),
        Field::new("y", DataType::Int64, true),
        Field::new("d", DataType::Decimal128(9, 2), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                Some(3),
                Some(3),
                None,
                None,
                Some(1),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(10),
                Some(10),
                None,
                Some(20),
                None,
                None,
                Some(30),
                Some(31),
                Some(30),
                Some(40),
                Some(40),
                Some(20),
            ])),
            Arc::new(Int64Array::from(vec![
                Some(5),
                None,
                Some(7),
                Some(1),
                Some(4),
                None,
                None,
                None,
                None,
                Some(2),
                Some(3),
                Some(9),
            ])),
            Arc::new(
                Decimal128Array::from(vec![
                    Some(1_000),
                    Some(1_001),
                    None,
                    Some(1_001),
                    None,
                    None,
                    Some(-1),
                    Some(-1),
                    Some(0),
                    Some(6),
                    None,
                    None,
                ])
                .with_precision_and_scale(9, 2)
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn write_uneven_avg_input_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2, 1, 1, 3])),
            Arc::new(Int32Array::from(vec![10, 10, 10, 10, 10, 10])),
            Arc::new(Int64Array::from(vec![0, 100, 200, 30, 30, 300])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

pub(super) fn date32(year: i32, month: u32, day: u32) -> i32 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 { adjusted_year } else { adjusted_year - 399 } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month = month as i32;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

pub(super) fn write_list_count_input_parquet(path: &std::path::Path) {
    let item = Arc::new(Field::new("element", DataType::Int64, true));
    let pair = Arc::new(Field::new("element", DataType::Int32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("items", DataType::List(Arc::clone(&item)), true),
        Field::new("pairs", DataType::FixedSizeList(Arc::clone(&pair), 2), true),
    ]));
    // items: NULL, [], [NULL], [1, NULL], [2, 3], NULL. Child nulls and empty
    // lists count as valid rows; only the two top-level NULLs are excluded.
    let items = ListArray::new(
        item,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 0, 0, 1, 3, 5, 5])),
        Arc::new(Int64Array::from(vec![None, Some(1), None, Some(2), Some(3)])),
        Some(NullBuffer::from(vec![false, true, true, true, true, false])),
    );
    // pairs: [1, 2], NULL, [NULL, NULL], [3, 4], NULL, [5, 6].
    let pairs = FixedSizeListArray::new(
        pair,
        2,
        Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            None,
            None,
            None,
            None,
            Some(3),
            Some(4),
            None,
            None,
            Some(5),
            Some(6),
        ])),
        Some(NullBuffer::from(vec![true, false, true, true, false, true])),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])), Arc::new(items), Arc::new(pairs)],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

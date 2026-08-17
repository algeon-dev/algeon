use std::{fs::File, sync::Arc};

use arrow::{
    array::{
        ArrayRef, BinaryArray, BooleanArray, Date32Array, Date64Array, Decimal128Array, FixedSizeListArray,
        Float64Array, Int32Array, Int32Builder, Int64Array, ListArray, MapBuilder, StringArray, StringBuilder,
        StructArray, TimestampMicrosecondArray, TimestampNanosecondArray,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{DataType, Field, Fields, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::parquet::{
    arrow::ArrowWriter,
    file::{metadata::SortingColumn, properties::WriterProperties},
};

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

pub(super) fn write_sorted_numeric_parquet_file(path: &std::path::Path) {
    write_sorted_numeric_parquet_file_with_keys(path, &[1, 1, 2]);
}

pub(super) fn write_sorted_numeric_parquet_file_with_sorting_columns(path: &std::path::Path) {
    write_sorted_numeric_parquet_file_with_keys_and_props(
        path,
        &[1, 1, 2],
        Some(
            WriterProperties::builder()
                .set_sorting_columns(Some(vec![SortingColumn { column_idx: 0, descending: false, nulls_first: false }]))
                .build(),
        ),
    );
}

pub(super) fn write_sorted_numeric_parquet_file_with_keys(path: &std::path::Path, keys: &[i32]) {
    write_sorted_numeric_parquet_file_with_keys_and_props(path, keys, None);
}

fn write_sorted_numeric_parquet_file_with_keys_and_props(
    path: &std::path::Path,
    keys: &[i32],
    props: Option<WriterProperties>,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let row_ids = (0..keys.len()).map(|index| i64::try_from(index + 1).unwrap()).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(keys.to_vec())),
            Arc::new(Int32Array::from(keys.iter().map(|key| key * 10).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(row_ids.iter().map(|index| index * 100).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(row_ids)),
        ],
    )
    .unwrap();
    write_batch_with_props(path, schema, batch, props);
}

pub(super) fn write_snap_edge_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("edge_id", DataType::Int64, false),
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
        Field::new("src_bucket", DataType::Int32, false),
        Field::new("dst_bucket", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![0, 1, 2, 3])),
            Arc::new(Int64Array::from(vec![5, 12, 20, 30])),
            Arc::new(Int64Array::from(vec![50, 40, 30, 20])),
            Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0, 1.0])),
            Arc::new(Int32Array::from(vec![0, 0, 0, 0])),
            Arc::new(Int32Array::from(vec![0, 0, 0, 0])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_full_gpu_gap_analysis_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, true),
        Field::new("score", DataType::Float64, true),
        Field::new("name", DataType::Utf8, true),
        Field::new("flag", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![10, 10, 20, 30])),
            Arc::new(Int64Array::from(vec![Some(5), Some(15), None, Some(25)])),
            Arc::new(Float64Array::from(vec![Some(1.0), Some(4.0), Some(9.0), None])),
            Arc::new(StringArray::from(vec![Some("alpha"), Some("beta"), Some("gamma"), None])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None, Some(true)])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_decimal_filter_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Decimal128(7, 2), false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Decimal128Array::from_iter_values([500, 2_000, 3_000]).with_precision_and_scale(7, 2).unwrap()),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_household_demographics_like_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("hd_dep_count", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("hd_vehicle_count", DataType::Int32, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(Int32Array::from(vec![4, 0, 3])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
            Arc::new(Int32Array::from(vec![2, 0, 1])),
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

pub(super) fn write_join_left_list_parquet_file(path: &std::path::Path) {
    let item = Arc::new(Field::new("element", DataType::Utf8, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("l_tags", DataType::List(Arc::clone(&item)), true),
        Field::new("l_m00", DataType::Int64, false),
    ]));
    let tags = ListArray::new(
        item,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 1, 2])),
        Arc::new(StringArray::from(vec![Some("red"), Some("blue")])),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(tags),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_right_list_parquet_file(path: &std::path::Path) {
    let item = Arc::new(Field::new("element", DataType::Utf8, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("r_tags", DataType::List(Arc::clone(&item)), true),
        Field::new("r_m01", DataType::Int64, false),
    ]));
    let tags = ListArray::new(
        item,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2, 2])),
        Arc::new(StringArray::from(vec![Some("cyan"), Some("gold")])),
        Some(NullBuffer::from(vec![true, true, false])),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(tags), Arc::new(Int64Array::from(vec![10, 20, 30]))],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_left_timestamp_tz_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("l_event_ts", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), false),
        Field::new("l_m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(TimestampMicrosecondArray::from(vec![100, 200, 300]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_right_timestamp_tz_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("r_event_ts", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), false),
        Field::new("r_m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(TimestampMicrosecondArray::from(vec![150, 250, 350]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![1, 2, 3])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_left_string_payload_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("l_zip", DataType::Utf8, false),
        Field::new("l_country", DataType::Utf8, false),
        Field::new("l_m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1])),
            Arc::new(StringArray::from(vec!["12345-0000", "22222-0000", "33333-0000"])),
            Arc::new(StringArray::from(vec!["US", "CA", "MX"])),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_join_right_string_payload_parquet_file(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("r_zip", DataType::Utf8, false),
        Field::new("r_country", DataType::Utf8, false),
        Field::new("r_m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["12345-9999", "44444-0000", "33333-9999"])),
            Arc::new(StringArray::from(vec!["us", "ca", "mx"])),
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

pub(super) fn write_list_utf8_parquet_file(path: &std::path::Path) {
    let item = Arc::new(Field::new("element", DataType::Utf8, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("tags", DataType::List(Arc::clone(&item)), true),
        Field::new("m00", DataType::Int64, false),
    ]));
    let tags = ListArray::new(
        item,
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 3])),
        Arc::new(StringArray::from(vec![Some("red"), Some("blue"), Some("green")])),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(tags), Arc::new(Int64Array::from(vec![10, 20, 30]))],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_fixed_size_list_int32_parquet_file(path: &std::path::Path) {
    let item = Arc::new(Field::new("element", DataType::Int32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("features", DataType::FixedSizeList(Arc::clone(&item), 2), true),
        Field::new("m00", DataType::Int64, false),
    ]));
    let features = FixedSizeListArray::new(
        item,
        2,
        Arc::new(Int32Array::from(vec![Some(10), None, Some(30), Some(40), Some(50), Some(60)])),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(features),
            Arc::new(Int64Array::from(vec![100, 200, 300])),
        ],
    )
    .unwrap();
    write_batch(path, schema, batch);
}

pub(super) fn write_struct_map_parquet_file(path: &std::path::Path) {
    let struct_x = Arc::new(Field::new("x", DataType::Int32, false));
    let struct_y = Arc::new(Field::new("y", DataType::Utf8, true));
    let details = Arc::new(StructArray::from(vec![
        (Arc::clone(&struct_x), Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef),
        (Arc::clone(&struct_y), Arc::new(StringArray::from(vec![Some("alpha"), None, Some("gamma")])) as ArrayRef),
    ])) as ArrayRef;

    let mut map_builder = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
    map_builder.keys().append_value("a");
    map_builder.values().append_value(1);
    map_builder.append(true).unwrap();
    map_builder.keys().append_value("b");
    map_builder.values().append_value(2);
    map_builder.keys().append_value("c");
    map_builder.values().append_value(3);
    map_builder.append(true).unwrap();
    map_builder.append(false).unwrap();
    let attrs = Arc::new(map_builder.finish()) as ArrayRef;

    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new(
            "details",
            DataType::Struct(Fields::from(vec![Arc::clone(&struct_x), Arc::clone(&struct_y)])),
            false,
        ),
        Field::new("attrs", attrs.data_type().clone(), true),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            details,
            attrs,
            Arc::new(Int64Array::from(vec![100, 200, 300])),
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

pub(super) fn write_nested_prefix_row_group_pruning_parquet_file(path: &std::path::Path) {
    let payload_x = Arc::new(Field::new("x", DataType::Int32, false));
    let payload_y = Arc::new(Field::new("y", DataType::Int32, false));
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Struct(Fields::from(vec![payload_x.clone(), payload_y.clone()])), false),
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let payload = Arc::new(StructArray::from(vec![
        (payload_x, Arc::new(Int32Array::from(vec![0, 0])) as ArrayRef),
        (payload_y, Arc::new(Int32Array::from(vec![200, 0])) as ArrayRef),
    ])) as ArrayRef;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![payload, Arc::new(Int32Array::from(vec![100, 200])), Arc::new(Int64Array::from(vec![10, 20]))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
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

pub(super) fn write_discounted_revenue_lineitem_parquet_file(path: &std::path::Path) {
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
    write_batch_with_props(path, schema, batch, None);
}

fn write_batch_with_props(
    path: &std::path::Path,
    schema: Arc<Schema>,
    batch: RecordBatch,
    props: Option<WriterProperties>,
) {
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, props).unwrap();
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

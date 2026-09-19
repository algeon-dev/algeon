use std::{collections::BTreeSet, fs::File, path::Path, sync::Arc};

use parquet::file::reader::{FileReader, SerializedFileReader};
use query_engine::parquet_stats::{ColumnByteKind, row_group_column_bytes};

use super::*;

#[path = "parquet_fixtures/aggregate_inputs.rs"]
mod aggregate_inputs;

pub(super) use aggregate_inputs::*;

pub(super) mod fixture_local_parquet_source {
    use super::*;

    pub(crate) fn from_path(
        path: impl Into<std::path::PathBuf>,
        schema: Arc<Schema>,
        read_columns: Vec<String>,
    ) -> NativeLocalParquetSource {
        let path = path.into();
        if path.is_file() {
            from_files(vec![LocalParquetFile::new(&path)], schema, read_columns)
        } else {
            NativeLocalParquetSource::new(path, schema, read_columns)
        }
    }

    pub(crate) fn from_files(
        files: Vec<LocalParquetFile>,
        schema: Arc<Schema>,
        read_columns: Vec<String>,
    ) -> NativeLocalParquetSource {
        let (read_column_indexes, projected_column_indexes) = fixture_column_indexes(&schema, &read_columns);
        NativeLocalParquetSource::from_files(
            files
                .into_iter()
                .map(|file| enrich_fixture_local_parquet_file(file, &read_column_indexes, &projected_column_indexes))
                .collect(),
            schema,
            read_columns,
        )
    }

    pub(crate) fn zero_column_count(files: Vec<LocalParquetFile>) -> NativeLocalParquetSource {
        let read_column_indexes = BTreeSet::new();
        NativeLocalParquetSource::zero_column_count(
            files
                .into_iter()
                .map(|file| enrich_fixture_local_parquet_file(file, &read_column_indexes, &read_column_indexes))
                .collect(),
        )
    }

    pub(crate) fn count_star_probe(
        files: Vec<LocalParquetFile>,
        read_schema: Arc<Schema>,
        read_columns: Vec<String>,
    ) -> NativeLocalParquetSource {
        let (read_column_indexes, projected_column_indexes) = fixture_column_indexes(&read_schema, &read_columns);
        let files = files
            .into_iter()
            .map(|file| enrich_fixture_local_parquet_file(file, &read_column_indexes, &projected_column_indexes))
            .collect();
        NativeLocalParquetSource::count_star_probe(files, read_schema, read_columns)
    }
}

fn fixture_local_parquet_file(
    path: &Path,
    read_column_indexes: &BTreeSet<usize>,
    projected_column_indexes: &BTreeSet<usize>,
) -> LocalParquetFile {
    let reader = SerializedFileReader::try_from(path).unwrap_or_else(|error| {
        panic!("written native execution fixture must be readable as Parquet: {}: {error}", path.display())
    });
    let metadata = reader.metadata();
    let file_size = fs::metadata(path)
        .unwrap_or_else(|error| {
            panic!("written native execution fixture must be stat-able: {}: {error}", path.display())
        })
        .len();
    let row_group_rows = metadata
        .row_groups()
        .iter()
        .map(|row_group| u64::try_from(row_group.num_rows()).expect("fixture row count fits u64"))
        .collect::<Vec<_>>();
    let row_group_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| {
            row_group_column_bytes(row_group, read_column_indexes, ColumnByteKind::Compressed).unwrap_or_else(|| {
                panic!("fixture Parquet row group is missing a read-column byte count: {}", path.display())
            })
        })
        .collect::<Vec<_>>();
    let projected_compressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| {
            row_group_column_bytes(row_group, projected_column_indexes, ColumnByteKind::Compressed).unwrap_or_else(
                || panic!("fixture Parquet row group is missing a projected-column byte count: {}", path.display()),
            )
        })
        .collect::<Vec<_>>();
    let projected_uncompressed_bytes = metadata
        .row_groups()
        .iter()
        .map(|row_group| {
            row_group_column_bytes(row_group, projected_column_indexes, ColumnByteKind::Uncompressed).unwrap_or_else(
                || panic!("fixture Parquet row group is missing a projected-column byte count: {}", path.display()),
            )
        })
        .collect::<Vec<_>>();
    LocalParquetFile::new(path)
        .with_file_size_bytes(file_size)
        .with_row_groups((0..metadata.num_row_groups()).collect())
        .with_row_group_rows(row_group_rows)
        .with_row_group_bytes(row_group_bytes)
        .with_projected_row_group_bytes(projected_compressed_bytes, projected_uncompressed_bytes)
}

fn enrich_fixture_local_parquet_file(
    mut file: LocalParquetFile,
    read_column_indexes: &BTreeSet<usize>,
    projected_column_indexes: &BTreeSet<usize>,
) -> LocalParquetFile {
    if !file.path.is_file() {
        return file;
    }
    let exact = fixture_local_parquet_file(&file.path, read_column_indexes, projected_column_indexes);
    if file.file_size_bytes.is_none() {
        file.file_size_bytes = exact.file_size_bytes;
    }
    if file.row_groups.is_none() {
        file.row_groups = exact.row_groups;
    }
    if file.row_group_bytes.is_none() {
        file.row_group_bytes = exact.row_group_bytes;
    }
    if file.projected_row_group_compressed_bytes.is_none() {
        file.projected_row_group_compressed_bytes = exact.projected_row_group_compressed_bytes;
    }
    if file.projected_row_group_uncompressed_bytes.is_none() {
        file.projected_row_group_uncompressed_bytes = exact.projected_row_group_uncompressed_bytes;
    }
    if file.row_group_rows.is_none() {
        file.row_group_rows = exact.row_group_rows;
    }
    file
}

fn fixture_column_indexes(schema: &Arc<Schema>, read_columns: &[String]) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let source = NativeLocalParquetSource::from_files(Vec::new(), Arc::clone(schema), read_columns.to_vec());
    let read_column_indexes = source.read_column_file_indices.iter().copied().collect::<BTreeSet<_>>();
    let projected_column_indexes = source
        .projected_output_column_source_refs
        .iter()
        .filter_map(|source_ref| match source_ref {
            query_engine::source::SourceColumnRef::File(index) => Some(*index),
            query_engine::source::SourceColumnRef::Partition(_)
            | query_engine::source::SourceColumnRef::MissingNull(_) => None,
        })
        .collect::<BTreeSet<_>>();
    (read_column_indexes, projected_column_indexes)
}

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

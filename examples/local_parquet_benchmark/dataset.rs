use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct ParquetDatasetLayout {
    pub(super) file_count: usize,
    pub(super) row_group_count: usize,
}

pub(super) fn write_benchmark_parquet_dataset(
    path: &Path,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if config.file_count == 1 {
        return write_benchmark_parquet_file(path, 0, config.input_rows, config);
    }

    fs::create_dir_all(path)?;
    let mut start = 0usize;
    for file_index in 0..config.file_count {
        let remaining_rows = config.input_rows - start;
        let remaining_files = config.file_count - file_index;
        let rows = remaining_rows.div_ceil(remaining_files);
        let file_path = path.join(format!("part-{file_index:05}.parquet"));
        write_benchmark_parquet_file(&file_path, start, rows, config)?;
        start += rows;
    }
    Ok(())
}

pub(super) fn write_benchmark_parquet_file(
    path: &Path,
    start_row: usize,
    rows: usize,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    let file = File::create(path)?;
    let schema = benchmark_schema(config.measure_columns);
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(config.row_group_size))
        .set_dictionary_enabled(false)
        .set_compression(config.compression.parquet())
        .build();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props))?;
    let mut offset = 0usize;

    while offset < rows {
        let len = (rows - offset).min(config.batch_size);
        writer.write(&benchmark_batch(Arc::clone(&schema), start_row + offset, offset, rows, len, config)?)?;
        offset += len;
    }

    writer.close()?;
    Ok(())
}

pub(super) fn benchmark_schema(measure_columns: usize) -> SchemaRef {
    let mut fields = Vec::with_capacity(measure_columns + 2);
    fields.push(Field::new("key", DataType::Int32, false));
    fields.push(Field::new("payload", DataType::Int32, false));
    fields.extend((0..measure_columns).map(|index| Field::new(measure_column_name(index), DataType::Int64, false)));
    Arc::new(Schema::new(fields))
}

pub(super) fn benchmark_batch(
    schema: SchemaRef,
    start: usize,
    file_local_start: usize,
    file_rows: usize,
    len: usize,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<RecordBatch> {
    let mut keys = Vec::with_capacity(len);
    let mut payload = Vec::with_capacity(len);
    let mut measures = (0..config.measure_columns).map(|_| Vec::with_capacity(len)).collect::<Vec<_>>();

    for offset in 0..len {
        let row = start + offset;
        let file_local_row = file_local_start + offset;
        keys.push((row % config.group_count) as i32);
        payload.push(benchmark_payload_value(file_local_row, file_rows, config));
        for (column_index, values) in measures.iter_mut().enumerate() {
            values.push(measure_value(row, column_index));
        }
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(config.measure_columns + 2);
    columns.push(Arc::new(Int32Array::from(keys)));
    columns.push(Arc::new(Int32Array::from(payload)));
    columns.extend(measures.into_iter().map(|values| Arc::new(Int64Array::from(values)) as ArrayRef));
    Ok(RecordBatch::try_new(schema, columns)?)
}

pub(super) fn benchmark_payload_value(
    file_local_row: usize,
    file_rows: usize,
    config: &LocalParquetBenchmarkConfig,
) -> i32 {
    let total_row_groups = file_rows.div_ceil(config.row_group_size);
    let pruned_row_groups = total_row_groups / 2;
    let row_group = file_local_row / config.row_group_size;
    if row_group < pruned_row_groups {
        (file_local_row % 10) as i32
    } else {
        10 + (splitmix64(file_local_row as u64 ^ 0x9e37_79b9_7f4a_7c15) % 90) as i32
    }
}

pub(super) fn measure_value(row: usize, column_index: usize) -> i64 {
    let row = row as u64;
    let column = column_index as u64;
    let mixed = splitmix64(row.wrapping_mul(0xd6e8_feb8_6659_fd93) ^ column.wrapping_mul(0xa076_1d64_78bd_642f));
    ((mixed ^ row.rotate_left((column_index % 31) as u32)) & 0xffff_ffff) as i64
}

pub(super) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub(super) fn validate_file_size(file_bytes: u64, config: &LocalParquetBenchmarkConfig) -> DataFusionResult<()> {
    if file_bytes < config.minimum_file_bytes {
        return Err(plan_error(format!(
            "generated Parquet file is below benchmark minimum: parquet_file_bytes={file_bytes}, minimum_file_bytes={}. Increase CUDF_PARQUET_BENCH_ROWS, CUDF_PARQUET_BENCH_MEASURE_COLUMNS, or use CUDF_PARQUET_BENCH_COMPRESSION=uncompressed",
            config.minimum_file_bytes
        )));
    }
    Ok(())
}

pub(super) fn validate_dataset_layout(
    dataset_layout: &ParquetDatasetLayout,
    config: &LocalParquetBenchmarkConfig,
) -> DataFusionResult<()> {
    if dataset_layout.file_count != config.file_count {
        return Err(plan_error(format!(
            "generated Parquet file count did not match benchmark config: actual_files={}, configured_files={}",
            dataset_layout.file_count, config.file_count
        )));
    }
    if dataset_layout.row_group_count <= dataset_layout.file_count {
        return Err(plan_error(format!(
            "generated Parquet dataset must contain at least one multi-row-group file for the direct native Parquet contract: files={}, row_groups={}, row_group_size={}. Lower CUDF_PARQUET_BENCH_ROW_GROUP_SIZE or increase CUDF_PARQUET_BENCH_ROWS",
            dataset_layout.file_count, dataset_layout.row_group_count, config.row_group_size
        )));
    }
    Ok(())
}

pub(super) fn parquet_dataset_bytes(path: &Path) -> DataFusionResult<u64> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }

    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

pub(super) fn parquet_dataset_layout(path: &Path) -> DataFusionResult<ParquetDatasetLayout> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        return Ok(ParquetDatasetLayout { file_count: 1, row_group_count: parquet_file_row_group_count(path)? });
    }

    let mut file_count = 0usize;
    let mut row_group_count = 0usize;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_file() {
            file_count += 1;
            row_group_count += parquet_file_row_group_count(&entry.path())?;
        }
    }
    Ok(ParquetDatasetLayout { file_count, row_group_count })
}

pub(super) fn parquet_file_row_group_count(path: &Path) -> DataFusionResult<usize> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|error| plan_error(format!("failed to read Parquet metadata for {}: {error}", path.display())))?;
    Ok(reader.metadata().num_row_groups())
}

pub(super) fn default_benchmark_dir() -> PathBuf {
    unique_benchmark_dir(Path::new(DEFAULT_BENCHMARK_DATA_DIR))
}

pub(super) fn benchmark_input_path(dir: &Path, config: &LocalParquetBenchmarkConfig) -> PathBuf {
    if config.file_count == 1 { dir.join("wide.parquet") } else { dir.join("wide_parquet_dataset") }
}

pub(super) fn unique_benchmark_dir(base_dir: &Path) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    base_dir.join(format!("datafusion-cudf-local-parquet-bench-{}-{nonce}", std::process::id()))
}

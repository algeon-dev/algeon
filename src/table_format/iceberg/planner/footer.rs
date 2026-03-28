use nexus_query_engine::parquet_stats::row_group_null_counts_by_root_column;

use super::*;

pub(super) fn scan_plan_needs_field_id_footer_schema(plan: &TableFormatScanPlan) -> bool {
    plan.format == TableFormat::Iceberg
        && plan.projected_columns.iter().chain(plan.pure_filter_columns.iter()).any(|mapping| {
            mapping.source == TableFormatColumnSource::DirectTopLevelFileColumn && mapping.field_id.is_some()
        })
}

pub(super) async fn remote_iceberg_parquet_footer_row_count(
    file_io: &FileIO,
    location: &str,
    known_file_size: Option<u64>,
) -> Option<u64> {
    remote_iceberg_parquet_footer(file_io, location, known_file_size).await.and_then(|footer| footer.row_count)
}

pub(super) struct RemoteIcebergParquetFooter {
    pub(super) schema: SchemaRef,
    pub(super) row_count: Option<u64>,
    pub(super) row_group_null_counts: BTreeMap<usize, Vec<Option<u64>>>,
}

pub(super) async fn remote_iceberg_parquet_footer(
    file_io: &FileIO,
    location: &str,
    known_file_size: Option<u64>,
) -> Option<RemoteIcebergParquetFooter> {
    let input = file_io.new_input(location).ok()?;
    let file_size = match known_file_size {
        Some(file_size) => file_size,
        None => input.metadata().await.ok()?.size,
    };
    if file_size < FOOTER_SIZE as u64 {
        return None;
    }

    let reader = input.reader().await.ok()?;
    let prefetch = (64 * 1024).min(file_size);
    let tail_start = file_size - prefetch;
    let tail = reader.read(tail_start..file_size).await.ok()?;
    if tail.len() < FOOTER_SIZE {
        return None;
    }
    let footer_tail_bytes: [u8; FOOTER_SIZE] = tail[tail.len() - FOOTER_SIZE..].try_into().ok()?;
    let footer_tail = FooterTail::try_from(footer_tail_bytes).ok()?;
    if footer_tail.is_encrypted_footer() {
        return None;
    }

    let total_footer_len = footer_tail.metadata_length() + FOOTER_SIZE;
    if total_footer_len as u64 > file_size {
        return None;
    }
    let metadata_bytes = if total_footer_len as u64 <= prefetch {
        let start = tail.len() - total_footer_len;
        let end = tail.len() - FOOTER_SIZE;
        tail.slice(start..end)
    } else {
        let metadata_start = file_size - total_footer_len as u64;
        let metadata_end = file_size - FOOTER_SIZE as u64;
        reader.read(metadata_start..metadata_end).await.ok()?
    };

    let mut footer = BytesMut::with_capacity(total_footer_len);
    footer.extend_from_slice(&metadata_bytes);
    footer.extend_from_slice(&footer_tail_bytes);
    let footer = footer.freeze();

    let mut metadata_reader = ParquetMetaDataReader::new();
    metadata_reader.try_parse_sized(&footer, file_size).ok()?;
    let metadata = metadata_reader.finish().ok()?;
    let file_metadata = metadata.file_metadata();
    let schema =
        parquet::arrow::parquet_to_arrow_schema(file_metadata.schema_descr(), file_metadata.key_value_metadata())
            .ok()
            .map(|schema| Arc::new(schema) as SchemaRef)?;
    let row_count = u64::try_from(file_metadata.num_rows()).ok();
    let row_group_null_counts = parquet_footer_row_group_null_counts(&metadata);
    Some(RemoteIcebergParquetFooter { schema, row_count, row_group_null_counts })
}

pub(super) fn parquet_footer_row_group_null_counts(metadata: &ParquetMetaData) -> BTreeMap<usize, Vec<Option<u64>>> {
    row_group_null_counts_by_root_column(metadata)
}

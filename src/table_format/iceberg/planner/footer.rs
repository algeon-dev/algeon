use std::{collections::BTreeMap, sync::Arc};

use nexus_query_engine::parquet_stats::row_group_null_counts_by_root_column;

use super::{
    BytesMut, Error, ErrorCode, FOOTER_SIZE, FileIO, FileRead, FooterTail, ParquetMetaData, ParquetMetaDataReader,
    Result, SchemaRef, TableFormat, TableFormatColumnSource, TableFormatScanPlan,
};

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
) -> Result<Option<u64>> {
    Ok(remote_iceberg_parquet_footer(file_io, location, known_file_size).await?.and_then(|footer| footer.row_count))
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
) -> Result<Option<RemoteIcebergParquetFooter>> {
    let input = file_io
        .new_input(location)
        .map_err(|error| iceberg_footer_probe_error("failed to open Iceberg Parquet input", "open_input", error))?;
    let file_size = match known_file_size {
        Some(file_size) => file_size,
        None => {
            input
                .metadata()
                .await
                .map_err(|error| {
                    iceberg_footer_probe_error(
                        "failed to read Iceberg Parquet object metadata",
                        "read_object_metadata",
                        error,
                    )
                })?
                .size
        },
    };
    if file_size < FOOTER_SIZE as u64 {
        return Err(invalid_footer_probe_error(
            "Iceberg Parquet object is shorter than the footer tail",
            "validate_object_size",
        ));
    }

    let reader = input.reader().await.map_err(|error| {
        iceberg_footer_probe_error("failed to open Iceberg Parquet object reader", "open_reader", error)
    })?;
    let prefetch = (64 * 1024).min(file_size);
    let tail_start = file_size - prefetch;
    let tail = reader.read(tail_start..file_size).await.map_err(|error| {
        iceberg_footer_probe_error("failed to read Iceberg Parquet footer tail", "read_footer_tail", error)
    })?;
    if tail.len() as u64 != prefetch {
        return Err(invalid_footer_probe_error(
            "Iceberg Parquet footer tail read returned an unexpected byte count",
            "validate_footer_tail_read",
        ));
    }
    let footer_tail_bytes: [u8; FOOTER_SIZE] = tail[tail.len() - FOOTER_SIZE..].try_into().map_err(|error| {
        footer_probe_source_error("failed to extract Iceberg Parquet footer tail", "extract_footer_tail", error)
    })?;
    let footer_tail = FooterTail::try_from(footer_tail_bytes).map_err(|error| {
        footer_probe_source_error("failed to decode Iceberg Parquet footer tail", "decode_footer_tail", error)
    })?;
    if footer_tail.is_encrypted_footer() {
        return Ok(None);
    }

    let total_footer_len = footer_tail.metadata_length().checked_add(FOOTER_SIZE).ok_or_else(|| {
        invalid_footer_probe_error("Iceberg Parquet footer length overflowed", "validate_footer_length")
    })?;
    if total_footer_len as u64 > file_size {
        return Err(invalid_footer_probe_error(
            "Iceberg Parquet footer length exceeds the object size",
            "validate_footer_length",
        ));
    }
    let metadata_bytes = if total_footer_len as u64 <= prefetch {
        let start = tail.len() - total_footer_len;
        let end = tail.len() - FOOTER_SIZE;
        tail.slice(start..end)
    } else {
        let metadata_start = file_size - total_footer_len as u64;
        let metadata_end = file_size - FOOTER_SIZE as u64;
        let metadata = reader.read(metadata_start..metadata_end).await.map_err(|error| {
            iceberg_footer_probe_error("failed to read Iceberg Parquet footer metadata", "read_footer_metadata", error)
        })?;
        if metadata.len() != footer_tail.metadata_length() {
            return Err(invalid_footer_probe_error(
                "Iceberg Parquet footer metadata read returned an unexpected byte count",
                "validate_footer_metadata_read",
            ));
        }
        metadata
    };

    let mut footer = BytesMut::with_capacity(total_footer_len);
    footer.extend_from_slice(&metadata_bytes);
    footer.extend_from_slice(&footer_tail_bytes);
    let footer = footer.freeze();

    let mut metadata_reader = ParquetMetaDataReader::new();
    metadata_reader.try_parse_sized(&footer, file_size).map_err(|error| {
        footer_probe_source_error("failed to parse Iceberg Parquet footer metadata", "parse_footer_metadata", error)
    })?;
    let metadata = metadata_reader.finish().map_err(|error| {
        footer_probe_source_error(
            "failed to finish Iceberg Parquet footer metadata parsing",
            "finish_footer_metadata",
            error,
        )
    })?;
    let file_metadata = metadata.file_metadata();
    let schema =
        parquet::arrow::parquet_to_arrow_schema(file_metadata.schema_descr(), file_metadata.key_value_metadata())
            .map_err(|error| {
                footer_probe_source_error(
                    "failed to convert Iceberg Parquet footer schema",
                    "convert_footer_schema",
                    error,
                )
            })
            .map(|schema| Arc::new(schema) as SchemaRef)?;
    let row_count = Some(u64::try_from(file_metadata.num_rows()).map_err(|error| {
        footer_probe_source_error("Iceberg Parquet footer has a negative row count", "convert_row_count", error)
    })?);
    let row_group_null_counts = parquet_footer_row_group_null_counts(&metadata);
    Ok(Some(RemoteIcebergParquetFooter { schema, row_count, row_group_null_counts }))
}

const REMOTE_FOOTER_PROBE_OPERATION: &str = "datafusion_nexus.table_format.iceberg.inspect_remote_parquet_footer";

fn iceberg_footer_probe_error(message: &'static str, stage: &'static str, source: iceberg::Error) -> Error {
    let error = Error::new(
        crate::table_format::iceberg::catalog::classify_iceberg_scan_error(&source),
        REMOTE_FOOTER_PROBE_OPERATION,
        message,
    )
    .with_fact("table_format", TableFormat::Iceberg.code())
    .with_fact("footer_probe_stage", stage);
    crate::table_format::iceberg::catalog::apply_iceberg_catalog_error_facts(error, &source).with_source(source)
}

fn footer_probe_source_error(
    message: &'static str,
    stage: &'static str,
    source: impl std::error::Error + Send + Sync + 'static,
) -> Error {
    Error::new(ErrorCode::SourceDependencyFailure, REMOTE_FOOTER_PROBE_OPERATION, message)
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_fact("footer_probe_stage", stage)
        .with_source(source)
}

fn invalid_footer_probe_error(message: &'static str, stage: &'static str) -> Error {
    Error::new(ErrorCode::SourceDependencyFailure, REMOTE_FOOTER_PROBE_OPERATION, message)
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_fact("footer_probe_stage", stage)
}

pub(super) fn parquet_footer_row_group_null_counts(metadata: &ParquetMetaData) -> BTreeMap<usize, Vec<Option<u64>>> {
    row_group_null_counts_by_root_column(metadata)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn corrupt_remote_footer_is_a_typed_inspection_failure() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("corrupt.parquet");
        fs::write(&path, [0_u8; FOOTER_SIZE]).unwrap();

        let result =
            remote_iceberg_parquet_footer(&FileIO::new_with_fs(), path.to_str().unwrap(), Some(FOOTER_SIZE as u64))
                .await;
        let Err(error) = result else {
            panic!("a corrupt footer must not be treated as unavailable proof");
        };

        assert_eq!(error.code(), ErrorCode::SourceDependencyFailure);
        assert_eq!(error.operation(), REMOTE_FOOTER_PROBE_OPERATION);
        assert_eq!(error.fact_value("table_format"), Some("iceberg"));
        assert_eq!(error.fact_value("footer_probe_stage"), Some("decode_footer_tail"));
    }

    #[tokio::test]
    async fn encrypted_remote_footer_remains_unavailable_native_proof() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().join("encrypted.parquet");
        fs::write(&path, [0, 0, 0, 0, b'P', b'A', b'R', b'E']).unwrap();

        let footer =
            remote_iceberg_parquet_footer(&FileIO::new_with_fs(), path.to_str().unwrap(), Some(FOOTER_SIZE as u64))
                .await
                .unwrap();

        assert!(footer.is_none(), "encrypted footer metadata cannot provide native field-id proof");
    }
}

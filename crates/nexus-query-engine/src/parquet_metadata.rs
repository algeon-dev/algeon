//! Parquet footer metadata helpers shared by source planning and diagnostics.

use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use object_store::{ObjectStore, ObjectStoreExt, path::Path as ObjectStorePath};
use parquet::file::{
    FOOTER_SIZE,
    metadata::{FooterTail, ParquetMetaData, ParquetMetaDataReader},
};

use crate::error::{QueryError, QueryErrorCode, Result};

/// Speculative tail size for footer-only object-store metadata reads.
const FOOTER_PREFETCH_BYTES: u64 = 64 * 1024;

/// Parses Parquet metadata from a complete in-memory Parquet payload.
pub(crate) fn parse_complete_parquet_metadata(
    bytes: &[u8],
    operation: &'static str,
    path: impl std::fmt::Display,
) -> Result<Arc<ParquetMetaData>> {
    let file_size = u64::try_from(bytes.len()).map_err(|_error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            operation,
            "Parquet payload length exceeds native metadata range",
        )
        .with_fact("path", path.to_string())
    })?;
    if bytes.len() < FOOTER_SIZE {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            operation,
            "Parquet payload is smaller than the footer",
        )
        .with_fact("path", path.to_string())
        .with_fact("file_size", file_size));
    }

    let footer_tail_bytes: [u8; FOOTER_SIZE] =
        bytes[bytes.len() - FOOTER_SIZE..].try_into().expect("footer tail slice was length-checked");
    let footer_tail = parse_footer_tail(footer_tail_bytes, operation, &path)?;
    let metadata_len = footer_tail.metadata_length();
    let total_footer_len = metadata_len + FOOTER_SIZE;
    if total_footer_len > bytes.len() {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            operation,
            "Parquet footer metadata length exceeds payload size",
        )
        .with_fact("path", path.to_string())
        .with_fact("file_size", file_size)
        .with_fact("metadata_length", metadata_len));
    }

    let metadata_start = bytes.len() - total_footer_len;
    let metadata_end = bytes.len() - FOOTER_SIZE;
    parse_metadata_from_footer_bytes(
        Bytes::copy_from_slice(&bytes[metadata_start..metadata_end]),
        footer_tail_bytes,
        file_size,
        operation,
        path,
    )
}

/// Fetches Parquet footer metadata from object storage without staging the full object.
pub(crate) async fn fetch_object_store_parquet_metadata(
    store: Arc<dyn ObjectStore>,
    location: ObjectStorePath,
) -> Result<(Arc<ParquetMetaData>, u64)> {
    let operation = "nexus_query_engine.source_metadata.object_store_footer";
    let head = store
        .head(&location)
        .await
        .map_err(|error| object_store_metadata_error("failed to HEAD object-store Parquet file", error, &location))?;
    let file_size = head.size;
    if file_size < FOOTER_SIZE as u64 {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            operation,
            "object-store Parquet file is smaller than the footer",
        )
        .with_fact("path", location.as_ref())
        .with_fact("file_size", file_size));
    }

    let prefetch = FOOTER_PREFETCH_BYTES.min(file_size);
    let tail_start = file_size - prefetch;
    let tail = store
        .get_range(&location, tail_start..file_size)
        .await
        .map_err(|error| object_store_metadata_error("failed to GET Parquet footer range", error, &location))?;
    if tail.len() < FOOTER_SIZE {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            operation,
            "object-store Parquet tail fetch returned fewer bytes than the footer size",
        )
        .with_fact("path", location.as_ref())
        .with_fact("file_size", file_size)
        .with_fact("tail_bytes", tail.len()));
    }

    let footer_tail_bytes: [u8; FOOTER_SIZE] =
        tail[tail.len() - FOOTER_SIZE..].try_into().expect("footer tail slice was length-checked");
    let footer_tail = parse_footer_tail(footer_tail_bytes, operation, &location.as_ref())?;
    let metadata_len = footer_tail.metadata_length();
    let total_footer_len = metadata_len + FOOTER_SIZE;
    if (total_footer_len as u64) > file_size {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            operation,
            "Parquet footer metadata length exceeds file size",
        )
        .with_fact("path", location.as_ref())
        .with_fact("file_size", file_size)
        .with_fact("metadata_length", metadata_len));
    }

    let metadata_bytes = if (total_footer_len as u64) <= prefetch {
        let start = tail.len() - total_footer_len;
        let end = tail.len() - FOOTER_SIZE;
        tail.slice(start..end)
    } else {
        let metadata_start = file_size - total_footer_len as u64;
        let metadata_end = file_size - FOOTER_SIZE as u64;
        store.get_range(&location, metadata_start..metadata_end).await.map_err(|error| {
            object_store_metadata_error("failed to GET Parquet footer metadata range", error, &location)
        })?
    };
    let metadata =
        parse_metadata_from_footer_bytes(metadata_bytes, footer_tail_bytes, file_size, operation, &location)?;
    Ok((metadata, file_size))
}

fn parse_footer_tail(
    footer_tail_bytes: [u8; FOOTER_SIZE],
    operation: &'static str,
    path: &impl std::fmt::Display,
) -> Result<FooterTail> {
    let footer_tail = FooterTail::try_from(footer_tail_bytes).map_err(|error| {
        QueryError::new(QueryErrorCode::CudfDependencyFailure, operation, "failed to decode Parquet footer tail")
            .with_fact("path", path.to_string())
            .with_source(error)
    })?;
    if footer_tail.is_encrypted_footer() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            operation,
            "encrypted Parquet footers are not supported by native footer metadata loading",
        )
        .with_fact("path", path.to_string()));
    }
    Ok(footer_tail)
}

fn parse_metadata_from_footer_bytes(
    metadata_bytes: Bytes,
    footer_tail_bytes: [u8; FOOTER_SIZE],
    file_size: u64,
    operation: &'static str,
    path: impl std::fmt::Display,
) -> Result<Arc<ParquetMetaData>> {
    let mut reader = ParquetMetaDataReader::new();
    let mut footer_with_tail = BytesMut::with_capacity(metadata_bytes.len() + FOOTER_SIZE);
    footer_with_tail.extend_from_slice(&metadata_bytes);
    footer_with_tail.extend_from_slice(&footer_tail_bytes);
    reader.try_parse_sized(&footer_with_tail.freeze(), file_size).map_err(|error| {
        QueryError::new(QueryErrorCode::CudfDependencyFailure, operation, "failed to decode Parquet footer metadata")
            .with_fact("path", path.to_string())
            .with_source(error)
    })?;
    reader.finish().map(Arc::new).map_err(|error| {
        QueryError::new(QueryErrorCode::CudfDependencyFailure, operation, "failed to finalize Parquet metadata")
            .with_fact("path", path.to_string())
            .with_source(error)
    })
}

fn object_store_metadata_error(
    message: &'static str,
    error: object_store::Error,
    location: &ObjectStorePath,
) -> QueryError {
    QueryError::new(
        QueryErrorCode::CudfDependencyFailure,
        "nexus_query_engine.source_metadata.object_store_footer",
        message,
    )
    .with_fact("path", location.as_ref())
    .with_source(error)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use object_store::{ObjectStoreExt, memory::InMemory, path::Path as ObjectStorePath};
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use super::*;

    fn write_test_parquet(row_count: usize, rows_per_group: usize) -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let values = (0..row_count).map(|value| value as i32).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(values))]).unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        bytes
    }

    #[test]
    fn parses_complete_payload_metadata() {
        let bytes = write_test_parquet(6, 2);

        let metadata =
            parse_complete_parquet_metadata(&bytes, "nexus_query_engine.test.parquet_metadata", "memory.parquet")
                .unwrap();

        assert_eq!(metadata.num_row_groups(), 3);
        assert_eq!(metadata.row_group(0).num_rows(), 2);
    }

    #[tokio::test]
    async fn fetches_object_store_footer_without_full_payload() {
        let bytes = write_test_parquet(8, 2);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("table/part-0.parquet");
        store.put(&location, bytes.clone().into()).await.unwrap();

        let (metadata, file_size) = fetch_object_store_parquet_metadata(store, location).await.unwrap();

        assert_eq!(file_size, bytes.len() as u64);
        assert_eq!(metadata.num_row_groups(), 4);
    }
}

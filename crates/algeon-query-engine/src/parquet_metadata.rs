//! Parquet footer metadata helpers shared by source planning and diagnostics.

use std::{
    error::Error as StdError,
    fmt, fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use bytes::{Bytes, BytesMut};
use object_store::{ObjectStore, ObjectStoreExt, path::Path as ObjectStorePath};
use parquet::file::{
    FOOTER_SIZE,
    metadata::{FooterTail, ParquetMetaData, ParquetMetaDataReader},
};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    source::redacted_object_store_path_diagnostic,
    util::fnv1a64,
};

use crate::object_store_error::{
    object_store_error_detail, object_store_error_kind, query_error_code_for_object_store_error,
};

/// Speculative tail size for footer-only object-store metadata reads.
pub(crate) const PARQUET_FOOTER_PREFETCH_BYTES: u64 = 64 * 1024;

pub(crate) const fn parquet_footer_prefetch_bytes(file_size: u64) -> u64 {
    if file_size < PARQUET_FOOTER_PREFETCH_BYTES { file_size } else { PARQUET_FOOTER_PREFETCH_BYTES }
}

/// Footer sidecar file signature: "ALGEONPF", the Algeon Parquet Footer
/// sidecar. Readers check it before interpreting the following little-endian
/// u64 fields as file size, content-identity hash, and file modified time; a
/// file without it is ignored, and the caller rewrites the sidecar.
const FOOTER_SIDECAR_MAGIC: &[u8; 8] = b"ALGEONPF";
const FOOTER_SIDECAR_HEADER_BYTES: usize =
    FOOTER_SIDECAR_MAGIC.len() + std::mem::size_of::<u64>() + std::mem::size_of::<u64>() + std::mem::size_of::<u64>();
/// File-name suffix of a footer sidecar. The lakehouse cache directory scan
/// skips files with it.
pub(crate) const PARQUET_FOOTER_SIDECAR_SUFFIX: &str = ".footer";

/// Fetches Parquet footer metadata from object storage without staging the full object.
pub(crate) async fn fetch_object_store_parquet_metadata(
    store: Arc<dyn ObjectStore>,
    location: ObjectStorePath,
) -> Result<(Arc<ParquetMetaData>, u64)> {
    let operation = "query_engine.source_metadata.object_store_footer";
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
        .with_fact("path", object_store_path_fact(&location))
        .with_fact("file_size", file_size));
    }

    let prefetch = parquet_footer_prefetch_bytes(file_size);
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
        .with_fact("path", object_store_path_fact(&location))
        .with_fact("file_size", file_size)
        .with_fact("tail_bytes", tail.len()));
    }

    let footer_tail_bytes: [u8; FOOTER_SIZE] =
        tail[tail.len() - FOOTER_SIZE..].try_into().expect("footer tail slice was length-checked");
    let footer_tail = parse_footer_tail(footer_tail_bytes, operation, &object_store_path_fact(&location))?;
    let metadata_len = footer_tail.metadata_length();
    let total_footer_len = metadata_len + FOOTER_SIZE;
    if (total_footer_len as u64) > file_size {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            operation,
            "Parquet footer metadata length exceeds file size",
        )
        .with_fact("path", object_store_path_fact(&location))
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
    let metadata = parse_metadata_from_footer_bytes(
        metadata_bytes,
        footer_tail_bytes,
        file_size,
        operation,
        object_store_path_fact(&location),
    )?;
    Ok((metadata, file_size))
}

/// Footer sidecar path for a complete lakehouse cache data file.
pub(crate) fn parquet_footer_sidecar_path(data_path: &Path) -> PathBuf {
    let file_name = data_path.file_name().and_then(|value| value.to_str()).unwrap_or("cache-entry");
    data_path.with_file_name(format!(".{file_name}{PARQUET_FOOTER_SIDECAR_SUFFIX}"))
}

/// Reads a persisted footer sidecar for a complete cache file.
///
/// Stale sidecars whose recorded file size or content identity does not match
/// the data file are ignored so callers can fall back to the complete file.
pub(crate) fn read_parquet_footer_sidecar(
    data_path: &Path,
    expected_content_identity: &str,
) -> Result<Option<Arc<ParquetMetaData>>> {
    let sidecar_path = parquet_footer_sidecar_path(data_path);
    let Ok(bytes) = fs::read(&sidecar_path) else {
        return Ok(None);
    };
    if bytes.len() < FOOTER_SIDECAR_HEADER_BYTES + FOOTER_SIZE
        || &bytes[..FOOTER_SIDECAR_MAGIC.len()] != FOOTER_SIDECAR_MAGIC
    {
        return Ok(None);
    }
    let file_size_end = FOOTER_SIDECAR_MAGIC.len() + std::mem::size_of::<u64>();
    let identity_hash_end = file_size_end + std::mem::size_of::<u64>();
    let modified_nanos_end = identity_hash_end + std::mem::size_of::<u64>();
    let file_size = u64::from_le_bytes(
        bytes[FOOTER_SIDECAR_MAGIC.len()..file_size_end].try_into().expect("sidecar file-size slice is length-checked"),
    );
    let identity_hash = u64::from_le_bytes(
        bytes[file_size_end..identity_hash_end].try_into().expect("sidecar identity-hash slice is length-checked"),
    );
    let modified_nanos = u64::from_le_bytes(
        bytes[identity_hash_end..modified_nanos_end].try_into().expect("sidecar modified-time slice is length-checked"),
    );
    let Ok(data_metadata) = fs::metadata(data_path) else {
        return Ok(None);
    };
    if data_metadata.len() != file_size {
        return Ok(None);
    }
    if identity_hash != fnv1a64(expected_content_identity.as_bytes()) {
        return Ok(None);
    }
    if file_modified_nanos(&data_metadata) != Some(modified_nanos) {
        return Ok(None);
    }
    let footer_bytes = &bytes[FOOTER_SIDECAR_HEADER_BYTES..];
    let footer_tail_bytes: [u8; FOOTER_SIZE] = footer_bytes[footer_bytes.len() - FOOTER_SIZE..]
        .try_into()
        .expect("sidecar footer tail slice is length-checked");
    let metadata_bytes = Bytes::copy_from_slice(&footer_bytes[..footer_bytes.len() - FOOTER_SIZE]);
    parse_metadata_from_footer_bytes(
        metadata_bytes,
        footer_tail_bytes,
        file_size,
        "query_engine.source_metadata.footer_sidecar",
        cache_metadata_path_fact(&sidecar_path),
    )
    .map(Some)
}

/// Persists a footer sidecar beside a complete lakehouse cache data file.
pub(crate) fn write_parquet_footer_sidecar_from_file(data_path: &Path, content_identity: &str) -> Result<()> {
    let mut file = fs::File::open(data_path).map_err(|error| {
        local_metadata_io_error("failed to open complete Parquet cache file for footer sidecar", error, data_path)
    })?;
    let data_metadata = file.metadata().map_err(|error| {
        local_metadata_io_error("failed to stat complete Parquet cache file for footer sidecar", error, data_path)
    })?;
    let file_size = data_metadata.len();
    if file_size < FOOTER_SIZE as u64 {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "query_engine.source_metadata.footer_sidecar",
            "complete Parquet cache file is smaller than the footer",
        )
        .with_fact("path", cache_metadata_path_fact(data_path))
        .with_fact("file_size", file_size));
    }
    let mut footer_tail_bytes = [0u8; FOOTER_SIZE];
    file.seek(SeekFrom::End(-(FOOTER_SIZE as i64))).map_err(|error| {
        local_metadata_io_error("failed to seek complete Parquet cache footer tail", error, data_path)
    })?;
    file.read_exact(&mut footer_tail_bytes).map_err(|error| {
        local_metadata_io_error("failed to read complete Parquet cache footer tail", error, data_path)
    })?;
    let footer_tail = parse_footer_tail(
        footer_tail_bytes,
        "query_engine.source_metadata.footer_sidecar",
        &cache_metadata_path_fact(data_path),
    )?;
    let total_footer_len = footer_tail.metadata_length() + FOOTER_SIZE;
    if total_footer_len as u64 > file_size {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "query_engine.source_metadata.footer_sidecar",
            "Parquet footer metadata length exceeds complete cache file size",
        )
        .with_fact("path", cache_metadata_path_fact(data_path))
        .with_fact("file_size", file_size)
        .with_fact("metadata_length", footer_tail.metadata_length()));
    }

    let mut footer_bytes = vec![0u8; total_footer_len];
    file.seek(SeekFrom::Start(file_size - total_footer_len as u64)).map_err(|error| {
        local_metadata_io_error("failed to seek complete Parquet cache footer metadata", error, data_path)
    })?;
    file.read_exact(&mut footer_bytes).map_err(|error| {
        local_metadata_io_error("failed to read complete Parquet cache footer metadata", error, data_path)
    })?;

    let sidecar_path = parquet_footer_sidecar_path(data_path);
    let temp_path = parquet_footer_sidecar_temp_path(&sidecar_path);
    let mut sidecar = fs::OpenOptions::new().write(true).create_new(true).open(&temp_path).map_err(|error| {
        local_metadata_io_error("failed to create Parquet footer sidecar temp file", error, &sidecar_path)
            .with_fact("temp_path", cache_metadata_path_fact(&temp_path))
    })?;
    sidecar.write_all(FOOTER_SIDECAR_MAGIC).map_err(|error| {
        sidecar_temp_io_error("failed to write Parquet footer sidecar header", error, &sidecar_path, &temp_path)
    })?;
    sidecar.write_all(&file_size.to_le_bytes()).map_err(|error| {
        sidecar_temp_io_error("failed to write Parquet footer sidecar file size", error, &sidecar_path, &temp_path)
    })?;
    sidecar.write_all(&fnv1a64(content_identity.as_bytes()).to_le_bytes()).map_err(|error| {
        sidecar_temp_io_error(
            "failed to write Parquet footer sidecar content identity",
            error,
            &sidecar_path,
            &temp_path,
        )
    })?;
    sidecar.write_all(&file_modified_nanos(&data_metadata).unwrap_or(0).to_le_bytes()).map_err(|error| {
        sidecar_temp_io_error("failed to write Parquet footer sidecar modified time", error, &sidecar_path, &temp_path)
    })?;
    sidecar.write_all(&footer_bytes).map_err(|error| {
        sidecar_temp_io_error("failed to write Parquet footer sidecar bytes", error, &sidecar_path, &temp_path)
    })?;
    sidecar.sync_all().map_err(|error| {
        sidecar_temp_io_error("failed to sync Parquet footer sidecar", error, &sidecar_path, &temp_path)
    })?;
    // Close the temp file descriptor before atomic rename into the final sidecar path.
    drop(sidecar);
    fs::rename(&temp_path, &sidecar_path).map_err(|error| {
        sidecar_temp_io_error("failed to publish Parquet footer sidecar", error, &sidecar_path, &temp_path)
    })?;
    Ok(())
}

fn sidecar_temp_io_error(
    message: &'static str,
    error: std::io::Error,
    sidecar_path: &Path,
    temp_path: &Path,
) -> QueryError {
    let _ = fs::remove_file(temp_path);
    local_metadata_io_error(message, error, sidecar_path).with_fact("temp_path", cache_metadata_path_fact(temp_path))
}

fn parquet_footer_sidecar_temp_path(sidecar_path: &Path) -> PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |duration| duration.as_nanos());
    let file_name = sidecar_path.file_name().and_then(|value| value.to_str()).unwrap_or("footer-sidecar");
    sidecar_path.with_file_name(format!(".{file_name}.tmp.{}.{}", std::process::id(), nonce))
}

fn file_modified_nanos(metadata: &fs::Metadata) -> Option<u64> {
    let modified = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_nanos();
    u64::try_from(modified).ok()
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
    source: object_store::Error,
    location: &ObjectStorePath,
) -> QueryError {
    object_store_query_error("query_engine.source_metadata.object_store_footer", message, source)
        .with_fact("path", object_store_path_fact(location))
}

pub(crate) fn object_store_query_error(
    operation: &'static str,
    message: &'static str,
    source: object_store::Error,
) -> QueryError {
    let code = query_error_code_for_object_store_error(&source);
    let kind = object_store_error_kind(&source);
    let detail = object_store_error_detail(&source);
    let error = QueryError::new(code, operation, message).with_fact("object_store_error_kind", kind);
    let error = match detail {
        Some(detail) => error.with_fact("object_store_error_detail", detail),
        None => error,
    };
    error.with_source(RedactedObjectStoreError { kind, detail, source })
}

#[derive(Debug)]
struct RedactedObjectStoreError {
    kind: &'static str,
    detail: Option<&'static str>,
    source: object_store::Error,
}

impl fmt::Display for RedactedObjectStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "object-store error kind={}", self.kind)?;
        if let Some(detail) = self.detail {
            write!(f, " detail={detail}")?;
        }
        Ok(())
    }
}

impl StdError for RedactedObjectStoreError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.source)
    }
}

fn object_store_path_fact(location: &ObjectStorePath) -> String {
    redacted_object_store_path_diagnostic(location.as_ref())
}

fn local_metadata_io_error(message: &'static str, error: std::io::Error, path: &Path) -> QueryError {
    QueryError::new(QueryErrorCode::CudfDependencyFailure, "query_engine.source_metadata.footer_sidecar", message)
        .with_fact("path", cache_metadata_path_fact(path))
        .with_source(error)
}

fn cache_metadata_path_fact(path: &Path) -> String {
    format!("lakehouse_cache_metadata_path:fnv64:{:016x}", stable_cache_metadata_hash(&path.display().to_string()))
}

/// Stable, non-cryptographic fingerprint for cache metadata paths in diagnostics.
///
/// This is FNV-1a 64-bit, chosen only so logs can correlate the same local path
/// across runs without printing credential-bearing cache paths. The hex value is
/// not reversible, not a security boundary, and must not be used as a cache key.
fn stable_cache_metadata_hash(value: &str) -> u64 {
    fnv1a64(value.as_bytes())
}

#[cfg(test)]
mod tests {
    use std::{fmt, fs, sync::Arc};

    use arrow::{
        array::Int32Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use async_trait::async_trait;
    use futures::{
        StreamExt,
        stream::{BoxStream, empty},
    };
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStoreExt,
        PutMultipartOptions, PutOptions, PutPayload, PutResult, memory::InMemory, path::Path as ObjectStorePath,
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use super::*;

    fn write_test_parquet(row_count: usize, rows_per_group: usize) -> Vec<u8> {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let values = (0..row_count).map(|value| value as i32).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(values))]).unwrap();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        bytes
    }

    #[derive(Debug, Clone, Copy)]
    enum FaultKind {
        PermissionDenied,
        Unauthenticated,
        NotFound,
        Generic,
        GenericHttp403,
        GenericHttp401,
        GenericMissingCredentials,
        GenericInvalidAccessKey,
        GenericExpiredToken,
        GenericExpiredTokenHttp403,
        GenericSignatureMismatch,
    }

    #[derive(Debug)]
    struct FaultObjectStore {
        kind: FaultKind,
    }

    impl FaultObjectStore {
        const fn new(kind: FaultKind) -> Self {
            Self { kind }
        }

        fn error(&self, location: &ObjectStorePath) -> object_store::Error {
            let path = location.to_string();
            match self.kind {
                FaultKind::PermissionDenied => object_store::Error::PermissionDenied {
                    path,
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "simulated access denied",
                    )),
                },
                FaultKind::Unauthenticated => object_store::Error::Unauthenticated {
                    path,
                    source: Box::new(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "simulated unauthenticated",
                    )),
                },
                FaultKind::NotFound => object_store::Error::NotFound {
                    path,
                    source: Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "simulated missing object")),
                },
                FaultKind::Generic => object_store::Error::Generic {
                    store: "fault",
                    source: Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, "simulated timeout")),
                },
                FaultKind::GenericHttp403 => object_store::Error::Generic {
                    store: "fault",
                    source: Box::new(std::io::Error::other(
                        "Server returned non-2xx status code: 403 Forbidden: AccessDenied",
                    )),
                },
                FaultKind::GenericHttp401 => object_store::Error::Generic {
                    store: "fault",
                    source: Box::new(std::io::Error::other("Server returned non-2xx status code: 401 Unauthorized")),
                },
                FaultKind::GenericMissingCredentials => object_store::Error::Generic {
                    store: "fault",
                    source: Box::new(std::io::Error::other("could not load credentials from AWS provider chain")),
                },
                FaultKind::GenericInvalidAccessKey => object_store::Error::Generic {
                    store: "fault",
                    source: Box::new(std::io::Error::other("InvalidAccessKeyId: access key is not valid")),
                },
                FaultKind::GenericExpiredToken => object_store::Error::Generic {
                    store: "fault",
                    source: Box::new(NestedSourceError {
                        message: "catalog request failed",
                        source: std::io::Error::other("ExpiredToken: security token expired"),
                    }),
                },
                FaultKind::GenericExpiredTokenHttp403 => object_store::Error::Generic {
                    store: "fault",
                    source: Box::new(std::io::Error::other(
                        "Server returned non-2xx status code: 403 Forbidden; ExpiredToken: security token expired",
                    )),
                },
                FaultKind::GenericSignatureMismatch => object_store::Error::Generic {
                    store: "fault",
                    source: Box::new(std::io::Error::other(
                        "SignatureDoesNotMatch: the request signature did not match",
                    )),
                },
            }
        }
    }

    #[derive(Debug)]
    struct NestedSourceError {
        message: &'static str,
        source: std::io::Error,
    }

    impl fmt::Display for NestedSourceError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.message)
        }
    }

    impl StdError for NestedSourceError {
        fn source(&self) -> Option<&(dyn StdError + 'static)> {
            Some(&self.source)
        }
    }

    impl fmt::Display for FaultObjectStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "fault-object-store")
        }
    }

    #[async_trait]
    impl ObjectStore for FaultObjectStore {
        async fn put_opts(
            &self,
            location: &ObjectStorePath,
            _payload: PutPayload,
            _opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            Err(self.error(location))
        }

        async fn put_multipart_opts(
            &self,
            location: &ObjectStorePath,
            _opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            Err(self.error(location))
        }

        async fn get_opts(&self, location: &ObjectStorePath, _options: GetOptions) -> object_store::Result<GetResult> {
            Err(self.error(location))
        }

        fn delete_stream(
            &self,
            _locations: BoxStream<'static, object_store::Result<ObjectStorePath>>,
        ) -> BoxStream<'static, object_store::Result<ObjectStorePath>> {
            empty().boxed()
        }

        fn list(&self, _prefix: Option<&ObjectStorePath>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            empty().boxed()
        }

        async fn list_with_delimiter(&self, _prefix: Option<&ObjectStorePath>) -> object_store::Result<ListResult> {
            Ok(ListResult { common_prefixes: Vec::new(), objects: Vec::new() })
        }

        async fn copy_opts(
            &self,
            from: &ObjectStorePath,
            _to: &ObjectStorePath,
            _options: CopyOptions,
        ) -> object_store::Result<()> {
            Err(self.error(from))
        }
    }

    #[test]
    fn footer_prefetch_size_is_capped_by_file_size() {
        assert_eq!(parquet_footer_prefetch_bytes(0), 0);
        assert_eq!(parquet_footer_prefetch_bytes(1024), 1024);
        assert_eq!(parquet_footer_prefetch_bytes(PARQUET_FOOTER_PREFETCH_BYTES), PARQUET_FOOTER_PREFETCH_BYTES);
        assert_eq!(parquet_footer_prefetch_bytes(PARQUET_FOOTER_PREFETCH_BYTES + 1), PARQUET_FOOTER_PREFETCH_BYTES);
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

    #[tokio::test]
    async fn object_store_footer_fetch_preserves_source_storage_error_classification() {
        let cases = [
            (
                FaultKind::PermissionDenied,
                QueryErrorCode::SourceAccessDenied,
                "permission_denied",
                Some("access_denied"),
            ),
            (
                FaultKind::Unauthenticated,
                QueryErrorCode::SourceUnauthenticated,
                "unauthenticated",
                Some("unauthenticated"),
            ),
            (FaultKind::NotFound, QueryErrorCode::SourceObjectNotFound, "not_found", Some("not_found")),
            (FaultKind::Generic, QueryErrorCode::SourceTemporaryFailure, "generic", Some("timeout")),
            (FaultKind::GenericHttp403, QueryErrorCode::SourceAccessDenied, "generic", Some("access_denied")),
            (FaultKind::GenericHttp401, QueryErrorCode::SourceUnauthenticated, "generic", Some("http_401")),
            (
                FaultKind::GenericMissingCredentials,
                QueryErrorCode::SourceUnauthenticated,
                "generic",
                Some("missing_credentials"),
            ),
            (
                FaultKind::GenericInvalidAccessKey,
                QueryErrorCode::SourceUnauthenticated,
                "generic",
                Some("invalid_access_key"),
            ),
            (FaultKind::GenericExpiredToken, QueryErrorCode::SourceUnauthenticated, "generic", Some("expired_token")),
            (
                FaultKind::GenericExpiredTokenHttp403,
                QueryErrorCode::SourceUnauthenticated,
                "generic",
                Some("expired_token"),
            ),
            (
                FaultKind::GenericSignatureMismatch,
                QueryErrorCode::SourceAccessDenied,
                "generic",
                Some("signature_mismatch"),
            ),
        ];

        for (fault, expected_code, expected_kind, expected_detail) in cases {
            let store: Arc<dyn ObjectStore> = Arc::new(FaultObjectStore::new(fault));
            let location = ObjectStorePath::from("secret-bucket/private/file.parquet?X-Amz-Signature=do-not-leak");

            let error = fetch_object_store_parquet_metadata(store, location).await.unwrap_err();

            assert_eq!(error.code(), expected_code);
            assert_eq!(error.operation(), "query_engine.source_metadata.object_store_footer");
            assert_eq!(error.fact_value("object_store_error_kind"), Some(expected_kind));
            assert_eq!(error.fact_value("object_store_error_detail"), expected_detail);
            assert!(error.source_error().is_some());
            let diagnostic = error.fact_value("path").expect("redacted path fact");
            assert!(diagnostic.starts_with("object_store_path:fnv64:"));
            for sensitive in ["secret-bucket", "private/file", "X-Amz-Signature", "do-not-leak"] {
                assert!(!error.to_string().contains(sensitive), "diagnostic leaked {sensitive}: {error}");
            }
        }
    }

    #[test]
    fn object_store_footer_path_diagnostics_are_redacted() {
        let location = ObjectStorePath::from("secret-bucket/private/file.parquet?X-Amz-Signature=do-not-leak");
        let diagnostic = object_store_path_fact(&location);

        assert!(diagnostic.starts_with("object_store_path:fnv64:"));
        for sensitive in ["secret-bucket", "private/file", "X-Amz-Signature", "do-not-leak"] {
            assert!(!diagnostic.contains(sensitive), "diagnostic leaked {sensitive}: {diagnostic}");
        }
    }

    #[test]
    fn footer_sidecar_path_diagnostics_are_redacted() {
        let path = Path::new("/tmp/cache/secret-bucket/private/file.parquet?X-Amz-Signature=do-not-leak");
        let diagnostic = cache_metadata_path_fact(path);

        assert!(diagnostic.starts_with("lakehouse_cache_metadata_path:fnv64:"));
        for sensitive in ["secret-bucket", "private/file", "X-Amz-Signature", "do-not-leak"] {
            assert!(!diagnostic.contains(sensitive), "diagnostic leaked {sensitive}: {diagnostic}");
        }
    }

    #[test]
    fn footer_sidecar_round_trips_and_rejects_stale_file_size() {
        let root = std::env::temp_dir().join(format!(
            "query-engine-footer-sidecar-{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let data_path = root.join("part-0.parquet");
        fs::write(&data_path, write_test_parquet(12, 4)).unwrap();

        write_parquet_footer_sidecar_from_file(&data_path, "content-a").unwrap();
        let metadata = read_parquet_footer_sidecar(&data_path, "content-a").unwrap().unwrap();
        assert_eq!(metadata.file_metadata().num_rows(), 12);
        assert_eq!(metadata.num_row_groups(), 3);
        assert!(read_parquet_footer_sidecar(&data_path, "content-b").unwrap().is_none());

        fs::write(&data_path, b"stale").unwrap();
        assert!(read_parquet_footer_sidecar(&data_path, "content-a").unwrap().is_none());

        let _ = fs::remove_dir_all(root);
    }
}

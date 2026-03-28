use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use parquet::file::{
    metadata::ParquetMetaData,
    reader::{FileReader, SerializedFileReader},
};

use super::source_file_needs_footer_metadata;
use crate::{
    error::{QueryError, QueryErrorCode, Result},
    parquet_metadata::{fetch_object_store_parquet_metadata, read_parquet_footer_sidecar},
    runtime::{ExecutionContext, S3ObjectStoreKey},
    source::{LocalParquetFile, LocalParquetSource, ParquetReadStrategy, ParquetSourceLocation},
};

const OBJECT_STORE_FOOTER_METADATA_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Footer metadata cache shared across repeated source chunk planning calls.
#[derive(Debug, Default)]
pub struct SourceMetadataCache {
    footers: HashMap<SourceMetadataCacheKey, Option<Arc<ParquetMetaData>>>,
}

impl SourceMetadataCache {
    /// Returns the number of cached footer entries.
    ///
    /// Local-cache misses are not cached while the cache file is absent or
    /// unreadable, because fill-before-read may make the footer available to a
    /// later planning pass that reuses this cache.
    pub fn cached_footer_count(&self) -> usize {
        self.footers.len()
    }

    pub(super) fn footer_metadata(
        &mut self,
        file: &LocalParquetFile,
        context: Option<&SourceMetadataLoadContext<'_>>,
    ) -> Result<FooterMetadataLoad> {
        let key = SourceMetadataCacheKey::from_file(file);
        if let Some(metadata) = self.footers.get(&key) {
            return Ok(FooterMetadataLoad { metadata: metadata.clone(), load_time: Duration::ZERO, load_count: 0 });
        }

        let timer = Instant::now();
        let metadata = match load_local_parquet_footer_metadata(file) {
            Some(metadata) => Some(metadata),
            None => {
                context.map(|context| load_object_store_parquet_footer_metadata(context, file)).transpose()?.flatten()
            },
        };
        let load_time = timer.elapsed();
        if metadata.is_some() || cache_footer_metadata_miss(file) {
            self.footers.insert(key, metadata.clone());
        }
        Ok(FooterMetadataLoad { metadata, load_time, load_count: 1 })
    }

    pub(super) fn prefetch_footer_metadata_for_source(
        &mut self,
        source: &LocalParquetSource,
        context: Option<&SourceMetadataLoadContext<'_>>,
    ) -> Result<FooterMetadataLoad> {
        let Some(context) = context else {
            return Ok(FooterMetadataLoad { metadata: None, load_time: Duration::ZERO, load_count: 0 });
        };
        let timer = Instant::now();
        let mut load_count = 0u64;
        let mut pending = Vec::<PendingFooterMetadataFetch>::new();

        for file in &source.files {
            if !source_file_needs_footer_metadata(source, file) {
                continue;
            }
            let key = SourceMetadataCacheKey::from_file(file);
            if self.footers.contains_key(&key) {
                continue;
            }
            if let Some(metadata) = load_local_parquet_footer_metadata(file) {
                self.footers.insert(key, Some(metadata));
                load_count = load_count.saturating_add(1);
                continue;
            }
            if !source_file_can_fetch_object_store_footer(file) {
                continue;
            }
            load_count = load_count.saturating_add(1);
            let fetch_location = match source_metadata_fetch_location(file) {
                Ok(fetch_location) => fetch_location,
                Err(_error) => {
                    self.footers.insert(key, None);
                    continue;
                },
            };
            let Some(fetch) = spawn_footer_metadata_fetch(
                context,
                fetch_location,
                "nexus_query_engine.source_metadata.object_store_footer_prefetch",
            )?
            else {
                self.footers.insert(key, None);
                continue;
            };
            let FooterMetadataFetch { handle, receiver, wait_started: _ } = fetch;
            pending.push(PendingFooterMetadataFetch { key, handle, receiver });
        }

        while !pending.is_empty() {
            let mut index = 0;
            while index < pending.len() {
                match pending[index].receiver.try_recv() {
                    Ok(Ok(metadata)) => {
                        let completed = pending.swap_remove(index);
                        self.footers.insert(completed.key.clone(), Some(metadata));
                    },
                    Ok(Err(error)) => {
                        if error.code() == QueryErrorCode::Cancelled {
                            return Err(error);
                        }
                        let completed = pending.swap_remove(index);
                        self.footers.insert(completed.key.clone(), None);
                    },
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        index += 1;
                    },
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        let completed = pending.swap_remove(index);
                        self.footers.insert(completed.key.clone(), None);
                    },
                }
            }
            if pending.is_empty() {
                break;
            }
            context
                .execution_context
                .check_cancellation("nexus_query_engine.source_metadata.object_store_footer_prefetch")?;
            let Some(remaining) = OBJECT_STORE_FOOTER_METADATA_FETCH_TIMEOUT.checked_sub(timer.elapsed()) else {
                for pending_fetch in pending.drain(..) {
                    match pending_fetch.receiver.try_recv() {
                        Ok(Ok(metadata)) => {
                            self.footers.insert(pending_fetch.key.clone(), Some(metadata));
                        },
                        Ok(Err(error)) => {
                            if error.code() == QueryErrorCode::Cancelled {
                                return Err(error);
                            }
                            self.footers.insert(pending_fetch.key.clone(), None);
                        },
                        Err(std::sync::mpsc::TryRecvError::Empty | std::sync::mpsc::TryRecvError::Disconnected) => {
                            self.footers.insert(pending_fetch.key.clone(), None);
                        },
                    }
                }
                break;
            };
            match pending[0].receiver.recv_timeout(remaining.min(Duration::from_millis(100))) {
                Ok(Ok(metadata)) => {
                    let completed = pending.swap_remove(0);
                    self.footers.insert(completed.key.clone(), Some(metadata));
                },
                Ok(Err(error)) => {
                    if error.code() == QueryErrorCode::Cancelled {
                        return Err(error);
                    }
                    let completed = pending.swap_remove(0);
                    self.footers.insert(completed.key.clone(), None);
                },
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {},
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    let completed = pending.swap_remove(0);
                    self.footers.insert(completed.key.clone(), None);
                },
            }
        }

        let load_time = if load_count > 0 { timer.elapsed() } else { Duration::ZERO };
        Ok(FooterMetadataLoad { metadata: None, load_time, load_count })
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SourceMetadataLoadContext<'a> {
    pub(super) execution_context: &'a ExecutionContext,
}

struct PendingFooterMetadataFetch {
    key: SourceMetadataCacheKey,
    handle: tokio::task::JoinHandle<()>,
    receiver: std::sync::mpsc::Receiver<Result<Arc<ParquetMetaData>>>,
}

impl Drop for PendingFooterMetadataFetch {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct FooterMetadataFetch {
    wait_started: Instant,
    handle: tokio::task::JoinHandle<()>,
    receiver: std::sync::mpsc::Receiver<Result<Arc<ParquetMetaData>>>,
}

fn spawn_footer_metadata_fetch(
    context: &SourceMetadataLoadContext<'_>,
    fetch_location: SourceMetadataFetchLocation,
    operation: &'static str,
) -> Result<Option<FooterMetadataFetch>> {
    let store = match context.execution_context.s3_object_store(S3ObjectStoreKey {
        bucket: fetch_location.bucket.clone(),
        region: fetch_location.region.clone(),
        storage_profile_id: fetch_location.storage_profile_id.clone(),
    }) {
        Ok(store) => store,
        Err(_error) => return Ok(None),
    };
    let wait_started = Instant::now();
    let Some(operation_permit) = context
        .execution_context
        .acquire_source_fetch_operation_with_timeout(OBJECT_STORE_FOOTER_METADATA_FETCH_TIMEOUT, operation)?
    else {
        return Ok(None);
    };
    let (sender, receiver) = std::sync::mpsc::channel();
    let key = fetch_location.key;
    let handle = context.execution_context.spawn_object_store_custom_fetch(
        move || fetch_object_store_parquet_metadata(store, key),
        move |fetch_result| {
            let metadata = fetch_result.value.map(|(metadata, _file_size)| metadata);
            let _ = sender.send(metadata);
            drop(operation_permit);
        },
    );
    Ok(Some(FooterMetadataFetch { wait_started, handle, receiver }))
}

fn load_local_parquet_footer_metadata(file: &LocalParquetFile) -> Option<Arc<ParquetMetaData>> {
    local_parquet_footer_metadata_path(file).and_then(|path| {
        let content_identity = sidecar_content_identity(file);
        if matches!(
            file.resolved_source.read_strategy,
            ParquetReadStrategy::LocalCacheKvikio | ParquetReadStrategy::RemoteDirect
        ) && let Ok(Some(metadata)) = read_parquet_footer_sidecar(path, content_identity)
        {
            return Some(metadata);
        }
        SerializedFileReader::try_from(path).ok().map(|reader| {
            if matches!(
                file.resolved_source.read_strategy,
                ParquetReadStrategy::LocalCacheKvikio | ParquetReadStrategy::RemoteDirect
            ) {
                let _ = crate::parquet_metadata::write_parquet_footer_sidecar_from_file(path, content_identity);
            }
            Arc::new(reader.metadata().clone())
        })
    })
}

pub(super) fn sidecar_content_identity(file: &LocalParquetFile) -> &str {
    file.resolved_source.content_identity.as_deref().unwrap_or(file.stable_id.as_str())
}

pub(super) fn local_parquet_footer_metadata_path(file: &LocalParquetFile) -> Option<&std::path::Path> {
    if let Some(path) = file.location.as_local_path() {
        return Some(path);
    }
    if file.resolved_source.read_strategy == ParquetReadStrategy::LocalCacheKvikio && file.path.is_file() {
        return Some(file.path.as_path());
    }
    None
}

fn cache_footer_metadata_miss(file: &LocalParquetFile) -> bool {
    file.resolved_source.read_strategy != ParquetReadStrategy::LocalCacheKvikio
}

fn load_object_store_parquet_footer_metadata(
    context: &SourceMetadataLoadContext<'_>,
    file: &LocalParquetFile,
) -> Result<Option<Arc<ParquetMetaData>>> {
    if !source_file_can_fetch_object_store_footer(file) {
        return Ok(None);
    }
    const OPERATION: &str = "nexus_query_engine.source_metadata.object_store_footer";
    context.execution_context.check_cancellation(OPERATION)?;
    let fetch_location = match source_metadata_fetch_location(file) {
        Ok(fetch_location) => fetch_location,
        Err(_error) => return Ok(None),
    };
    let Some(fetch) = spawn_footer_metadata_fetch(context, fetch_location, OPERATION)? else {
        return Ok(None);
    };
    let FooterMetadataFetch { wait_started, handle: fetch_handle, receiver } = fetch;
    loop {
        if let Err(error) = context.execution_context.check_cancellation(OPERATION) {
            fetch_handle.abort();
            return Err(error);
        }
        let Some(remaining) = OBJECT_STORE_FOOTER_METADATA_FETCH_TIMEOUT.checked_sub(wait_started.elapsed()) else {
            fetch_handle.abort();
            return Ok(None);
        };
        match receiver.recv_timeout(remaining.min(Duration::from_millis(100))) {
            Ok(Ok(metadata)) => return Ok(Some(metadata)),
            Ok(Err(error)) => {
                if error.code() == QueryErrorCode::Cancelled {
                    return Err(error);
                }
                return Ok(None);
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
        }
    }
}

fn source_file_can_fetch_object_store_footer(file: &LocalParquetFile) -> bool {
    matches!(
        file.resolved_source.read_strategy,
        ParquetReadStrategy::RemoteHostBuffer | ParquetReadStrategy::RemoteDirect
    )
}

struct SourceMetadataFetchLocation {
    bucket: String,
    region: String,
    key: object_store::path::Path,
    storage_profile_id: Option<String>,
}

fn source_metadata_fetch_location(file: &LocalParquetFile) -> Result<SourceMetadataFetchLocation> {
    let ParquetSourceLocation::ObjectStoreUri { scheme, uri } = &file.location else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_metadata.location",
            "source metadata fetch requires an object-store URI source location",
        )
        .with_fact("read_capability", file.location.read_capability()));
    };
    if scheme != "s3" {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.source_metadata.location",
            "object-store source metadata fetch currently supports only S3 URIs",
        )
        .with_fact("scheme", scheme));
    }
    parse_s3_metadata_source_uri(uri, file.resolved_source.storage_profile_id.clone())
}

fn parse_s3_metadata_source_uri(uri: &str, storage_profile_id: Option<String>) -> Result<SourceMetadataFetchLocation> {
    let rest = uri.strip_prefix("s3://").ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_metadata.s3_uri",
            "S3 source URI must start with s3://",
        )
        .with_fact("reason", "missing_s3_scheme")
    })?;
    let (bucket, key) = rest.split_once('/').ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_metadata.s3_uri",
            "S3 source URI must include a bucket and key",
        )
        .with_fact("reason", "missing_key")
    })?;
    if bucket.is_empty() || key.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.source_metadata.s3_uri",
            "S3 source URI bucket and key must be non-empty",
        )
        .with_fact("reason", "empty_bucket_or_key"));
    }
    let region = std::env::var("AWS_REGION")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|value| !value.is_empty()))
        .unwrap_or_else(|| "us-east-1".to_string());
    Ok(SourceMetadataFetchLocation {
        bucket: bucket.to_string(),
        region,
        key: object_store::path::Path::from(key),
        storage_profile_id,
    })
}

// Cache identity for footer metadata, including source resolution contracts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct SourceMetadataCacheKey {
    location: ParquetSourceLocation,
    path: PathBuf,
    stable_id: String,
    file_size_bytes: Option<u64>,
    read_strategy: ParquetReadStrategy,
    storage_profile_id: Option<String>,
    content_identity: String,
}

impl SourceMetadataCacheKey {
    pub(super) fn from_file(file: &LocalParquetFile) -> Self {
        Self {
            location: file.location.clone(),
            path: file.path.clone(),
            stable_id: file.stable_id.clone(),
            file_size_bytes: file.file_size_bytes,
            read_strategy: file.resolved_source.read_strategy,
            storage_profile_id: file.resolved_source.storage_profile_id.clone(),
            content_identity: sidecar_content_identity(file).to_string(),
        }
    }
}

// Footer metadata result plus the load evidence reported to callers.
pub(super) struct FooterMetadataLoad {
    pub(super) metadata: Option<Arc<ParquetMetaData>>,
    pub(super) load_time: Duration,
    pub(super) load_count: u64,
}

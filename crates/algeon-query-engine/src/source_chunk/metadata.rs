use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use object_store::path::Path as ObjectStorePath;
use parquet::file::{
    metadata::ParquetMetaData,
    reader::{FileReader, SerializedFileReader},
};
use query_planning::{
    error::{QueryError, QueryErrorCode, Result},
    source::{LocalParquetFile, LocalParquetSource, ParquetReadStrategy, ParquetSourceLocation},
};
use query_runtime::runtime::{ExecutionContext, S3ObjectStoreKey};
use tokio::task::JoinHandle;

use crate::parquet_metadata::{
    fetch_object_store_parquet_metadata, read_parquet_footer_sidecar, write_parquet_footer_sidecar_from_file,
};

const OBJECT_STORE_FOOTER_METADATA_FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Footer metadata cache shared across repeated source chunk planning calls.
#[derive(Debug, Default)]
pub struct SourceMetadataCache {
    footers: HashMap<SourceMetadataCacheKey, FooterMetadataCacheEntry>,
}

impl SourceMetadataCache {
    /// Returns the number of cached footer entries.
    ///
    /// Local-cache misses are not cached while the cache file is absent or
    /// unreadable, because fill-before-read may make the footer available to a
    /// later planning pass that reuses this cache.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn cached_footer_count(&self) -> usize {
        self.footers.len()
    }

    pub(super) fn footer_metadata(
        &mut self,
        source: &LocalParquetSource,
        file: &LocalParquetFile,
        context: Option<&ExecutionContext>,
    ) -> Result<FooterMetadataLoad> {
        let requirement =
            FooterMetadataRequirement::for_source_file_requirement(source_file_requires_footer_metadata(source, file));
        let key = SourceMetadataCacheKey::from_file(file);
        if let Some(entry) = self.footers.get(&key)
            && entry.usable_for(requirement)
        {
            return Ok(FooterMetadataLoad {
                metadata: entry.metadata.clone(),
                load_time: Duration::ZERO,
                load_count: 0,
            });
        }

        let timer = Instant::now();
        let metadata = match load_local_parquet_footer_metadata(file) {
            Some(metadata) => Some(metadata),
            None => context
                .map(|context| load_object_store_parquet_footer_metadata(context, file, requirement))
                .transpose()?
                .flatten(),
        };
        let load_time = timer.elapsed();
        if let Some(metadata) = &metadata {
            self.footers.insert(key, FooterMetadataCacheEntry::metadata(Arc::clone(metadata)));
        } else if cache_footer_metadata_miss(file) && !requirement.is_required() {
            self.footers.insert(key, FooterMetadataCacheEntry::miss(requirement));
        }
        Ok(FooterMetadataLoad { metadata, load_time, load_count: 1 })
    }

    pub(super) fn prefetch_footer_metadata_for_source(
        &mut self,
        source: &LocalParquetSource,
        context: Option<&ExecutionContext>,
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
            let requirement = FooterMetadataRequirement::for_source_file_requirement(
                source_file_requires_footer_metadata(source, file),
            );
            if self.footers.get(&key).is_some_and(|entry| entry.usable_for(requirement)) {
                continue;
            }
            if let Some(metadata) = load_local_parquet_footer_metadata(file) {
                self.footers.insert(key, FooterMetadataCacheEntry::metadata(metadata));
                load_count = load_count.saturating_add(1);
                continue;
            }
            if !source_file_can_fetch_object_store_footer(file) {
                continue;
            }
            load_count = load_count.saturating_add(1);
            let fetch_location = match source_metadata_fetch_location(file) {
                Ok(fetch_location) => fetch_location,
                Err(error) if requirement.is_required() => return Err(error),
                Err(_error) => {
                    self.footers.insert(key, FooterMetadataCacheEntry::miss(requirement));
                    continue;
                },
            };
            let Some(fetch) = spawn_footer_metadata_fetch(
                context,
                fetch_location,
                "query_engine.source_metadata.object_store_footer_prefetch",
                requirement,
            )?
            else {
                self.footers.insert(key, FooterMetadataCacheEntry::miss(requirement));
                continue;
            };
            let FooterMetadataFetch { handle, receiver, wait_started: _ } = fetch;
            pending.push(PendingFooterMetadataFetch { key, handle, receiver, requirement });
        }

        while !pending.is_empty() {
            let mut index = 0;
            while index < pending.len() {
                match pending[index].receiver.try_recv() {
                    Ok(Ok(metadata)) => {
                        let completed = pending.swap_remove(index);
                        self.footers.insert(completed.key.clone(), FooterMetadataCacheEntry::metadata(metadata));
                    },
                    Ok(Err(error)) => {
                        if should_propagate_footer_metadata_error(&error, pending[index].requirement) {
                            return Err(error);
                        }
                        let completed = pending.swap_remove(index);
                        self.footers
                            .insert(completed.key.clone(), FooterMetadataCacheEntry::miss(completed.requirement));
                    },
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        index += 1;
                    },
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        if pending[index].requirement.is_required() {
                            return Err(footer_metadata_fetch_disconnected_error(
                                "query_engine.source_metadata.object_store_footer_prefetch",
                            ));
                        }
                        let completed = pending.swap_remove(index);
                        self.footers
                            .insert(completed.key.clone(), FooterMetadataCacheEntry::miss(completed.requirement));
                    },
                }
            }
            if pending.is_empty() {
                break;
            }
            context.check_cancellation("query_engine.source_metadata.object_store_footer_prefetch")?;
            let Some(remaining) = OBJECT_STORE_FOOTER_METADATA_FETCH_TIMEOUT.checked_sub(timer.elapsed()) else {
                for pending_fetch in pending.drain(..) {
                    match pending_fetch.receiver.try_recv() {
                        Ok(Ok(metadata)) => {
                            self.footers
                                .insert(pending_fetch.key.clone(), FooterMetadataCacheEntry::metadata(metadata));
                        },
                        Ok(Err(error)) => {
                            if should_propagate_footer_metadata_error(&error, pending_fetch.requirement) {
                                return Err(error);
                            }
                            if pending_fetch.requirement.is_required() {
                                return Err(footer_metadata_fetch_timeout_error(
                                    "query_engine.source_metadata.object_store_footer_prefetch",
                                ));
                            }
                            self.footers.insert(
                                pending_fetch.key.clone(),
                                FooterMetadataCacheEntry::miss(pending_fetch.requirement),
                            );
                        },
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            if pending_fetch.requirement.is_required() {
                                return Err(footer_metadata_fetch_timeout_error(
                                    "query_engine.source_metadata.object_store_footer_prefetch",
                                ));
                            }
                            self.footers.insert(
                                pending_fetch.key.clone(),
                                FooterMetadataCacheEntry::miss(pending_fetch.requirement),
                            );
                        },
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            if pending_fetch.requirement.is_required() {
                                return Err(footer_metadata_fetch_disconnected_error(
                                    "query_engine.source_metadata.object_store_footer_prefetch",
                                ));
                            }
                            self.footers.insert(
                                pending_fetch.key.clone(),
                                FooterMetadataCacheEntry::miss(pending_fetch.requirement),
                            );
                        },
                    }
                }
                break;
            };
            match pending[0].receiver.recv_timeout(remaining.min(Duration::from_millis(100))) {
                Ok(Ok(metadata)) => {
                    let completed = pending.swap_remove(0);
                    self.footers.insert(completed.key.clone(), FooterMetadataCacheEntry::metadata(metadata));
                },
                Ok(Err(error)) => {
                    if should_propagate_footer_metadata_error(&error, pending[0].requirement) {
                        return Err(error);
                    }
                    let completed = pending.swap_remove(0);
                    self.footers.insert(completed.key.clone(), FooterMetadataCacheEntry::miss(completed.requirement));
                },
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {},
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    if pending[0].requirement.is_required() {
                        return Err(footer_metadata_fetch_disconnected_error(
                            "query_engine.source_metadata.object_store_footer_prefetch",
                        ));
                    }
                    let completed = pending.swap_remove(0);
                    self.footers.insert(completed.key.clone(), FooterMetadataCacheEntry::miss(completed.requirement));
                },
            }
        }

        let load_time = if load_count > 0 { timer.elapsed() } else { Duration::ZERO };
        Ok(FooterMetadataLoad { metadata: None, load_time, load_count })
    }
}

#[derive(Debug, Clone)]
struct FooterMetadataCacheEntry {
    metadata: Option<Arc<ParquetMetaData>>,
    miss_requirement: Option<FooterMetadataRequirement>,
}

impl FooterMetadataCacheEntry {
    fn metadata(metadata: Arc<ParquetMetaData>) -> Self {
        Self { metadata: Some(metadata), miss_requirement: None }
    }

    fn miss(requirement: FooterMetadataRequirement) -> Self {
        Self { metadata: None, miss_requirement: Some(requirement) }
    }

    fn usable_for(&self, requirement: FooterMetadataRequirement) -> bool {
        self.metadata.is_some() || (!requirement.is_required() && self.miss_requirement.is_some())
    }
}

struct PendingFooterMetadataFetch {
    key: SourceMetadataCacheKey,
    handle: JoinHandle<()>,
    receiver: std::sync::mpsc::Receiver<Result<Arc<ParquetMetaData>>>,
    requirement: FooterMetadataRequirement,
}

impl Drop for PendingFooterMetadataFetch {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct FooterMetadataFetch {
    wait_started: Instant,
    handle: JoinHandle<()>,
    receiver: std::sync::mpsc::Receiver<Result<Arc<ParquetMetaData>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FooterMetadataRequirement {
    Optional,
    Required,
}

impl FooterMetadataRequirement {
    const fn for_source_file_requirement(required: bool) -> Self {
        if required { Self::Required } else { Self::Optional }
    }

    const fn is_required(self) -> bool {
        matches!(self, Self::Required)
    }
}

fn spawn_footer_metadata_fetch(
    context: &ExecutionContext,
    fetch_location: SourceMetadataFetchLocation,
    _operation: &'static str,
    requirement: FooterMetadataRequirement,
) -> Result<Option<FooterMetadataFetch>> {
    let store = match context.s3_object_store(S3ObjectStoreKey {
        bucket: fetch_location.bucket.clone(),
        region: fetch_location.region.clone(),
        storage_profile_id: fetch_location.storage_profile_id.clone(),
    }) {
        Ok(store) => store,
        Err(error) if requirement.is_required() => return Err(error),
        Err(_error) => return Ok(None),
    };
    let wait_started = Instant::now();
    let (sender, receiver) = std::sync::mpsc::channel();
    let key = fetch_location.key;
    let handle = context.spawn_object_store_control_task(
        move || fetch_object_store_parquet_metadata(store, key),
        move |fetch_result| {
            let metadata = fetch_result.value.map(|(metadata, _file_size)| metadata);
            let _ = sender.send(metadata);
        },
    );
    Ok(Some(FooterMetadataFetch { wait_started, handle, receiver }))
}

fn load_local_parquet_footer_metadata(file: &LocalParquetFile) -> Option<Arc<ParquetMetaData>> {
    local_parquet_footer_metadata_path(file).and_then(|path| {
        let content_identity = sidecar_content_identity(file);
        if matches!(
            file.resolved_source.read_strategy(),
            ParquetReadStrategy::LocalCacheKvikio | ParquetReadStrategy::RemoteDirect
        ) && let Ok(Some(metadata)) = read_parquet_footer_sidecar(path, content_identity)
        {
            return Some(metadata);
        }
        SerializedFileReader::try_from(path).ok().map(|reader| {
            if matches!(
                file.resolved_source.read_strategy(),
                ParquetReadStrategy::LocalCacheKvikio | ParquetReadStrategy::RemoteDirect
            ) {
                let _ = write_parquet_footer_sidecar_from_file(path, content_identity);
            }
            Arc::new(reader.metadata().clone())
        })
    })
}

pub(crate) fn sidecar_content_identity(file: &LocalParquetFile) -> &str {
    file.resolved_source.content_identity.as_deref().unwrap_or(file.stable_id.as_str())
}

pub(crate) fn local_parquet_footer_metadata_path(file: &LocalParquetFile) -> Option<&std::path::Path> {
    if let Some(path) = file.location.as_local_path() {
        return Some(path);
    }
    if file.resolved_source.read_strategy() == ParquetReadStrategy::LocalCacheKvikio && file.path.is_file() {
        return Some(file.path.as_path());
    }
    None
}

fn cache_footer_metadata_miss(file: &LocalParquetFile) -> bool {
    file.resolved_source.read_strategy() != ParquetReadStrategy::LocalCacheKvikio
}

fn load_object_store_parquet_footer_metadata(
    context: &ExecutionContext,
    file: &LocalParquetFile,
    requirement: FooterMetadataRequirement,
) -> Result<Option<Arc<ParquetMetaData>>> {
    if !source_file_can_fetch_object_store_footer(file) {
        return Ok(None);
    }
    const OPERATION: &str = "query_engine.source_metadata.object_store_footer";
    context.check_cancellation(OPERATION)?;
    let fetch_location = match source_metadata_fetch_location(file) {
        Ok(fetch_location) => fetch_location,
        Err(error) if requirement.is_required() => return Err(error),
        Err(_error) => return Ok(None),
    };
    let Some(fetch) = spawn_footer_metadata_fetch(context, fetch_location, OPERATION, requirement)? else {
        return Ok(None);
    };
    let FooterMetadataFetch { wait_started, handle: fetch_handle, receiver } = fetch;
    loop {
        if let Err(error) = context.check_cancellation(OPERATION) {
            fetch_handle.abort();
            return Err(error);
        }
        let Some(remaining) = OBJECT_STORE_FOOTER_METADATA_FETCH_TIMEOUT.checked_sub(wait_started.elapsed()) else {
            fetch_handle.abort();
            if requirement.is_required() {
                return Err(footer_metadata_fetch_timeout_error(OPERATION));
            }
            return Ok(None);
        };
        match receiver.recv_timeout(remaining.min(Duration::from_millis(100))) {
            Ok(Ok(metadata)) => return Ok(Some(metadata)),
            Ok(Err(error)) => {
                if should_propagate_footer_metadata_error(&error, requirement) {
                    return Err(error);
                }
                return Ok(None);
            },
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                if requirement.is_required() {
                    return Err(footer_metadata_fetch_disconnected_error(OPERATION));
                }
                return Ok(None);
            },
        }
    }
}

fn should_propagate_footer_metadata_error(error: &QueryError, requirement: FooterMetadataRequirement) -> bool {
    match error.code() {
        QueryErrorCode::Cancelled
        | QueryErrorCode::SourceAccessDenied
        | QueryErrorCode::SourceUnauthenticated
        | QueryErrorCode::SourceObjectNotFound => true,
        QueryErrorCode::SourceTemporaryFailure => requirement.is_required(),
        _ => requirement.is_required(),
    }
}

fn footer_metadata_fetch_timeout_error(operation: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::SourceTemporaryFailure,
        operation,
        "timed out fetching required object-store Parquet footer metadata",
    )
    .with_fact("reason", "required_footer_metadata_fetch_timeout")
    .with_fact("object_store_error_kind", "timeout")
    .with_fact("timeout_ms", OBJECT_STORE_FOOTER_METADATA_FETCH_TIMEOUT.as_millis())
}

fn footer_metadata_fetch_disconnected_error(operation: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::SourceTemporaryFailure,
        operation,
        "object-store Parquet footer metadata fetch stopped before returning required metadata",
    )
    .with_fact("reason", "required_footer_metadata_fetch_disconnected")
    .with_fact("object_store_error_kind", "fetch_worker_disconnected")
}

fn source_file_can_fetch_object_store_footer(file: &LocalParquetFile) -> bool {
    matches!(file.resolved_source.read_strategy(), ParquetReadStrategy::RemoteDirect)
}

struct SourceMetadataFetchLocation {
    bucket: String,
    region: String,
    key: ObjectStorePath,
    storage_profile_id: Option<String>,
}

fn source_metadata_fetch_location(file: &LocalParquetFile) -> Result<SourceMetadataFetchLocation> {
    let ParquetSourceLocation::ObjectStoreUri { scheme, uri } = &file.location else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.source_metadata.location",
            "source metadata fetch requires an object-store URI source location",
        )
        .with_fact("read_capability", file.location.read_capability()));
    };
    if scheme != "s3" {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "query_engine.source_metadata.location",
            "object-store source metadata fetch currently supports only S3 URIs",
        )
        .with_fact("scheme", scheme));
    }
    parse_s3_metadata_source_uri(uri, file.resolved_source.storage_profile_id().map(str::to_owned))
}

fn parse_s3_metadata_source_uri(uri: &str, storage_profile_id: Option<String>) -> Result<SourceMetadataFetchLocation> {
    let rest = uri.strip_prefix("s3://").ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.source_metadata.s3_uri",
            "S3 source URI must start with s3://",
        )
        .with_fact("reason", "missing_s3_scheme")
    })?;
    let (bucket, key) = rest.split_once('/').ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.source_metadata.s3_uri",
            "S3 source URI must include a bucket and key",
        )
        .with_fact("reason", "missing_key")
    })?;
    if bucket.is_empty() || key.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.source_metadata.s3_uri",
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
        key: ObjectStorePath::from(key),
        storage_profile_id,
    })
}

// Cache identity for footer metadata, including source resolution contracts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SourceMetadataCacheKey {
    location: ParquetSourceLocation,
    path: PathBuf,
    stable_id: String,
    file_size_bytes: Option<u64>,
    read_strategy: ParquetReadStrategy,
    storage_profile_id: Option<String>,
    content_identity: String,
}

impl SourceMetadataCacheKey {
    pub(crate) fn from_file(file: &LocalParquetFile) -> Self {
        Self {
            location: file.location.clone(),
            path: file.path.clone(),
            stable_id: file.stable_id.clone(),
            file_size_bytes: file.file_size_bytes,
            read_strategy: file.resolved_source.read_strategy(),
            storage_profile_id: file.resolved_source.storage_profile_id().map(str::to_owned),
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

use query_planning::{
    chunk::{
        decimal::{
            object_store_decimal_reader_predicate_requires_footer_contract, parquet_reader_decimal_types,
            reader_decimal_type_metadata_unavailable_error, validate_object_store_reader_decimal_types,
            validate_supplied_reader_decimal_types_match_footer,
        },
        row_group::{RowGroupMetadata, row_group_metadata_from_footer, source_row_group_metadata},
    },
    source::ParquetReaderDecimalType,
};

pub(super) struct SourceMetadataLoad {
    pub(super) metadata: Option<RowGroupMetadata>,
    pub(super) load_time: Duration,
    pub(super) load_count: u64,
    pub(super) reader_decimal_types: Vec<ParquetReaderDecimalType>,
    pub(super) reader_decimal_types_footer_verified: bool,
}

pub(super) fn read_parquet_metadata(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    metadata_cache: &mut SourceMetadataCache,
    metadata_load_context: Option<&ExecutionContext>,
) -> Result<SourceMetadataLoad> {
    let needs_object_store_decimal_contract =
        object_store_decimal_reader_predicate_requires_footer_contract(source, file);
    let footer_load = if source_file_needs_footer_metadata(source, file) {
        metadata_cache.footer_metadata(source, file, metadata_load_context)?
    } else {
        FooterMetadataLoad { metadata: None, load_time: Duration::ZERO, load_count: 0 }
    };
    let footer_reader_decimal_types =
        footer_load.metadata.as_ref().map(|metadata| parquet_reader_decimal_types(metadata, source));
    let reader_decimal_types = if needs_object_store_decimal_contract {
        let Some(footer_reader_decimal_types) = footer_reader_decimal_types else {
            return Err(reader_decimal_type_metadata_unavailable_error(source, file));
        };
        validate_supplied_reader_decimal_types_match_footer(
            source,
            file,
            &file.reader_decimal_types,
            &footer_reader_decimal_types,
        )?;
        footer_reader_decimal_types
    } else if file.reader_decimal_types.is_empty() {
        footer_reader_decimal_types.unwrap_or_default()
    } else {
        file.reader_decimal_types.clone()
    };
    let reader_decimal_types_footer_verified = needs_object_store_decimal_contract;
    validate_object_store_reader_decimal_types(source, file, &reader_decimal_types)?;
    let footer_metadata =
        footer_load.metadata.as_ref().map(|metadata| row_group_metadata_from_footer(metadata, source));
    Ok(SourceMetadataLoad {
        metadata: source_row_group_metadata(file, footer_metadata),
        load_time: footer_load.load_time,
        load_count: footer_load.load_count,
        reader_decimal_types,
        reader_decimal_types_footer_verified,
    })
}

/// Loads one footer slot per `source.files` entry, in order, through the
/// shared footer cache.
///
/// A `None` slot means no footer is available for that file (a non-local file
/// with no context, or a load miss), and callers that consume footers treat it
/// as "skip the footer-dependent decision for this file" rather than an error.
/// Reuses [`SourceMetadataCache::footer_metadata`] so this shares the same
/// cache entries, requirement handling, and load-time accounting as ordinary
/// chunk-planning footer loads instead of loading footers a second time.
pub(crate) fn source_footer_metadata(
    source: &LocalParquetSource,
    cache: &mut SourceMetadataCache,
    context: Option<&ExecutionContext>,
) -> Result<Vec<Option<Arc<ParquetMetaData>>>> {
    source.files.iter().map(|file| Ok(cache.footer_metadata(source, file, context)?.metadata)).collect()
}

pub(super) fn source_file_needs_footer_metadata(source: &LocalParquetSource, file: &LocalParquetFile) -> bool {
    file.row_group_bytes.is_none()
        || file.projected_row_group_compressed_bytes.is_none()
        || file.projected_row_group_uncompressed_bytes.is_none()
        || file.pure_filter_row_group_compressed_bytes.is_none()
        || file.pure_filter_row_group_uncompressed_bytes.is_none()
        || file.row_group_rows.is_none()
        || object_store_decimal_reader_predicate_requires_footer_contract(source, file)
        || (file.reader_decimal_types.is_empty()
            && source
                .reader_predicate
                .as_ref()
                .is_some_and(|predicate| source.reader_predicate_requires_source_contract(predicate)))
}

pub(super) fn source_file_requires_footer_metadata(source: &LocalParquetSource, file: &LocalParquetFile) -> bool {
    source.scan_limit.is_some()
        || source.is_zero_column_count_source()
        || object_store_decimal_reader_predicate_requires_footer_contract(source, file)
}

pub(crate) fn footer_verified_object_store_reader_decimal_types(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> Result<Option<Vec<ParquetReaderDecimalType>>> {
    footer_verified_object_store_reader_decimal_types_with_supplied(context, source, file, &file.reader_decimal_types)
}

pub(crate) fn footer_verified_object_store_reader_decimal_types_with_supplied(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    supplied_reader_decimal_types: &[ParquetReaderDecimalType],
) -> Result<Option<Vec<ParquetReaderDecimalType>>> {
    let mut metadata_cache = SourceMetadataCache::default();
    footer_verified_object_store_reader_decimal_types_with_supplied_cache(
        context,
        source,
        file,
        supplied_reader_decimal_types,
        &mut metadata_cache,
    )
}

pub(crate) fn footer_verified_object_store_reader_decimal_types_with_supplied_cache(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    supplied_reader_decimal_types: &[ParquetReaderDecimalType],
    metadata_cache: &mut SourceMetadataCache,
) -> Result<Option<Vec<ParquetReaderDecimalType>>> {
    if !object_store_decimal_reader_predicate_requires_footer_contract(source, file) {
        return Ok(None);
    }
    let file_with_supplied_types;
    let file = if supplied_reader_decimal_types == file.reader_decimal_types {
        file
    } else {
        file_with_supplied_types = file.clone().with_reader_decimal_types(supplied_reader_decimal_types.to_vec());
        &file_with_supplied_types
    };
    let metadata_load = read_parquet_metadata(source, file, metadata_cache, Some(context))?;
    Ok(Some(metadata_load.reader_decimal_types))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn source_from_file(file: LocalParquetFile) -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::Int64, true)]));
        LocalParquetSource::from_files(vec![file], schema, vec!["c0".to_string()])
    }

    fn footer_error(code: QueryErrorCode) -> QueryError {
        QueryError::new(code, "query_engine.test.footer_metadata", "test footer metadata error")
    }

    #[test]
    fn footer_metadata_error_policy_preserves_fatal_source_errors() {
        for code in [
            QueryErrorCode::Cancelled,
            QueryErrorCode::SourceAccessDenied,
            QueryErrorCode::SourceUnauthenticated,
            QueryErrorCode::SourceObjectNotFound,
        ] {
            let error = footer_error(code);
            assert!(should_propagate_footer_metadata_error(&error, FooterMetadataRequirement::Optional));
            assert!(should_propagate_footer_metadata_error(&error, FooterMetadataRequirement::Required));
        }
    }

    #[test]
    fn footer_metadata_error_policy_keeps_optional_temporary_misses_soft() {
        let temporary = footer_error(QueryErrorCode::SourceTemporaryFailure);
        assert!(!should_propagate_footer_metadata_error(&temporary, FooterMetadataRequirement::Optional));
        assert!(should_propagate_footer_metadata_error(&temporary, FooterMetadataRequirement::Required));

        let dependency = footer_error(QueryErrorCode::CudfDependencyFailure);
        assert!(!should_propagate_footer_metadata_error(&dependency, FooterMetadataRequirement::Optional));
        assert!(should_propagate_footer_metadata_error(&dependency, FooterMetadataRequirement::Required));
    }

    #[test]
    fn cached_optional_footer_metadata_miss_is_not_reused_for_required_metadata() {
        let file =
            LocalParquetFile::object_store_uri("s3://bucket/table/part-0.parquet").expect("valid object-store file");
        let optional_source = source_from_file(file.clone());
        let mut required_source = source_from_file(file.clone());
        required_source.scan_limit = Some(1);
        let mut cache = SourceMetadataCache::default();

        let optional_load = cache.footer_metadata(&optional_source, &file, None).unwrap();
        assert!(optional_load.metadata.is_none());
        assert_eq!(optional_load.load_count, 1);
        assert_eq!(cache.cached_footer_count(), 1);

        let required_load = cache.footer_metadata(&required_source, &file, None).unwrap();
        assert!(required_load.metadata.is_none());
        assert_eq!(
            required_load.load_count, 1,
            "required metadata must ignore a previously cached optional miss and attempt a fresh load"
        );
    }
}
